"""Captured process identities survive loss of the writer's original evidence."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from qqtools.plugins.qexp.infrastructure.host import host_instance_id
from qqtools.plugins.qexp.runtime.responsibility import responsibility_root
from qqtools.plugins.qexp.runtime.responsibility_cleanup import CLEANUP_FORMAT, CleanupRequest, complete_cleanup
from qqtools.plugins.qexp.runtime.responsibility_store import (
    CAPTURED_WRITER_LIMIT,
    Conflict,
    Ledger,
    Unavailable,
    identity_key,
)

pytestmark = pytest.mark.integration
PAYLOAD = {"task_id": "deleted-task", "attempt_number": 1}
IDENTITY = "deleted-task-attempt-1"


def process_identity(pid):
    fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    return {
        "host_id": host_instance_id(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "pid_namespace": Path("/proc/self/ns/pid").stat().st_ino,
        "pid": pid,
        "start_time_ticks": int(fields[19]),
    }


def cleanup_request():
    return CleanupRequest(
        IDENTITY,
        PAYLOAD,
        {"format": CLEANUP_FORMAT, "task_id": PAYLOAD["task_id"], "attempt_id": IDENTITY, "basis": "terminal_attempt"},
    )


def test_writer_capture_is_durable_idempotent_and_retains_multiple_writers(tmp_path, monkeypatch):
    ledger = Ledger.create(tmp_path / "ledger")
    writer = process_identity(os.getpid())
    generation = ledger.capture_writer(IDENTITY, PAYLOAD, writer)
    reopened = Ledger(ledger.root)
    assert reopened.lookup(IDENTITY)["captured_writers"] == [writer]
    with monkeypatch.context() as patch:
        patch.setattr(reopened.io, "replace", lambda *_args, **_kwargs: pytest.fail("exact retry wrote a file"))
        assert reopened.capture_writer(IDENTITY, PAYLOAD, writer) == generation
    second = {**writer, "start_time_ticks": writer["start_time_ticks"] + 1}
    assert reopened.capture_writer(IDENTITY, PAYLOAD, second) > generation
    assert reopened.lookup(IDENTITY)["captured_writers"] == [writer, second]
    assert reopened.lookup(IDENTITY)["payload"] == PAYLOAD


def test_capture_cannot_add_writer_after_cleanup_handoff(tmp_path):
    ledger = Ledger.create(tmp_path / "ledger")
    writer = process_identity(os.getpid())
    generation = ledger.capture_writer(IDENTITY, PAYLOAD, writer)
    generation = ledger.handoff(IDENTITY, generation, cleanup_receipt={"fixture": "proof"})
    assert ledger.capture_writer(IDENTITY, PAYLOAD, writer) == generation
    before = ledger.lookup(IDENTITY)
    with pytest.raises(Conflict, match="cleanup"):
        ledger.capture_writer(IDENTITY, PAYLOAD, {**writer, "pid": writer["pid"] + 1})
    assert ledger.lookup(IDENTITY) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("host_id", ""),
        ("boot_id", "unknown"),
        ("pid_namespace", 0),
        ("pid", True),
        ("pid", -1),
        ("start_time_ticks", -1),
    ],
)
def test_invalid_writer_identity_never_publishes_membership(tmp_path, field, value):
    ledger = Ledger.create(tmp_path / "ledger")
    writer = {**process_identity(os.getpid()), field: value}
    with pytest.raises(ValueError):
        ledger.capture_writer(IDENTITY, PAYLOAD, writer)
    assert ledger.find(IDENTITY) is None


def test_cleanup_waits_for_captured_writer_after_all_original_evidence_is_gone(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    ledger = Ledger.create(responsibility_root(root))
    child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.buffer.read()"], stdin=subprocess.PIPE)
    try:
        writer = process_identity(child.pid)
        ledger.capture_writer(IDENTITY, PAYLOAD, writer)
        # No Task, Attempt, intent, registration or process manifest exists.
        assert not complete_cleanup(ledger, root, cleanup_request())
        assert ledger.lookup(IDENTITY)["stage"] == "active"
        assert "cleanup_receipt" not in ledger.lookup(IDENTITY)
        child.communicate(timeout=5)
        assert child.returncode == 0
        assert complete_cleanup(Ledger(ledger.root), root, cleanup_request())
        assert ledger.find(IDENTITY) is None
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=5)


@pytest.mark.parametrize(
    "damage", ["host", "namespace", "unreadable", "malformed", "boot_unreadable", "boot_malformed"]
)
def test_unresolved_captured_writer_retains_membership(tmp_path, monkeypatch, damage):
    root = tmp_path / "runtime"
    root.mkdir()
    ledger = Ledger.create(responsibility_root(root))
    writer = process_identity(os.getpid())
    if damage == "host":
        writer["host_id"] = "another-host"
        writer["boot_id"] = "10000000-0000-0000-0000-000000000001"
    if damage == "namespace":
        writer["pid_namespace"] += 1
    ledger.capture_writer(IDENTITY, PAYLOAD, writer)
    original = Path.read_text

    def read(path, *args, **kwargs):
        if path == Path("/proc/sys/kernel/random/boot_id"):
            if damage == "boot_unreadable":
                raise PermissionError("injected inaccessible boot identity")
            if damage == "boot_malformed":
                return "malformed"
        if path == Path("/proc") / str(os.getpid()) / "stat" and damage != "namespace":
            if damage == "unreadable":
                raise PermissionError("injected inaccessible process identity")
            return "malformed"
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", read)
        assert not complete_cleanup(ledger, root, cleanup_request())
    assert ledger.lookup(IDENTITY)["stage"] == "active"


@pytest.mark.parametrize("change", ["reboot", "pid_reuse"])
def test_old_boot_or_reused_pid_cannot_keep_a_finished_writer_alive(tmp_path, change):
    root = tmp_path / "runtime"
    root.mkdir()
    ledger = Ledger.create(responsibility_root(root))
    writer = process_identity(os.getpid())
    if change == "reboot":
        writer["boot_id"] = "10000000-0000-0000-0000-000000000001"
    else:
        writer["start_time_ticks"] += 1
    ledger.capture_writer(IDENTITY, PAYLOAD, writer)
    assert complete_cleanup(ledger, root, cleanup_request())
    assert ledger.find(IDENTITY) is None


def test_capture_helper_uses_cleanup_guard_and_canonical_locator(tmp_path):
    from qqtools.plugins.qexp.runtime.responsibility_backfill import capture_local_writer
    from qqtools.plugins.qexp.runtime.responsibility_cleanup import evidence_write_guard

    root = tmp_path / "runtime"
    root.mkdir()
    ledger = Ledger.create(responsibility_root(root))
    writer = process_identity(os.getpid())
    with evidence_write_guard(root, IDENTITY) as acquired:
        assert acquired
        with pytest.raises(Conflict, match="busy"):
            capture_local_writer(ledger, root, IDENTITY, {"task_id": "deleted-task"}, writer)
    assert ledger.find(IDENTITY) is None
    capture_local_writer(ledger, root, IDENTITY, {"task_id": "deleted-task"}, writer)
    assert ledger.lookup(IDENTITY)["payload"] == PAYLOAD


def test_writer_bound_fails_without_discarding_previously_captured_writers(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    ledger = Ledger.create(responsibility_root(root))
    live_writer = process_identity(os.getpid())
    writer = {**live_writer, "boot_id": "10000000-0000-0000-0000-000000000001"}
    for number in range(CAPTURED_WRITER_LIMIT):
        ledger.capture_writer(IDENTITY, PAYLOAD, {**writer, "pid": number + 1})
    before = ledger.lookup(IDENTITY)
    with pytest.raises(Unavailable, match="bound"):
        ledger.capture_writer(IDENTITY, PAYLOAD, live_writer)
    retained = Ledger(ledger.root).lookup(IDENTITY)
    assert retained["captured_writers"] == before["captured_writers"]
    assert retained["payload"] == before["payload"]
    assert retained["writer_capture_incomplete"] is True
    assert not complete_cleanup(Ledger(ledger.root), root, cleanup_request())
    with pytest.raises(Conflict, match="incomplete"):
        ledger.handoff(IDENTITY, retained["generation"])


@pytest.mark.parametrize("has_marker_room", [True, False])
def test_writer_encoding_refusal_retains_cleanup_even_when_entry_has_no_marker_room(
    tmp_path, monkeypatch, has_marker_room
):
    from qqtools.plugins.qexp.runtime import responsibility_store as storage

    ledger = Ledger.create(tmp_path / "ledger")
    old = {**process_identity(os.getpid()), "boot_id": "10000000-0000-0000-0000-000000000001"}
    ledger.capture_writer(IDENTITY, PAYLOAD, old)
    before = ledger.lookup(IDENTITY)
    size = len(storage.encode(before))
    monkeypatch.setattr(storage, "INITIAL_ENTRY_BYTES", size)
    if not has_marker_room:
        monkeypatch.setattr(storage, "ENTRY_BYTES", size)
    with pytest.raises(Unavailable, match="encoding"):
        ledger.capture_writer(IDENTITY, PAYLOAD, process_identity(os.getpid()))
    current = Ledger(ledger.root).lookup(IDENTITY)
    assert current["captured_writers"] == before["captured_writers"]
    with pytest.raises(Conflict, match="incomplete"):
        ledger.handoff(IDENTITY, current["generation"])
    monkeypatch.setattr(ledger.io, "replace", lambda *_args, **_kwargs: pytest.fail("repeated refusal wrote"))
    with pytest.raises(Unavailable, match="encoding"):
        ledger.capture_writer(IDENTITY, PAYLOAD, process_identity(os.getpid()))


def test_first_writer_encoding_refusal_marks_bucket_without_creating_partial_member(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runtime import responsibility_store as storage

    ledger = Ledger.create(tmp_path / "ledger")
    generation = ledger.publish(IDENTITY, PAYLOAD)
    other = next(
        f"other-{number}" for number in range(1000) if identity_key(f"other-{number}")[0] == identity_key(IDENTITY)[0]
    )
    monkeypatch.setattr(storage, "INITIAL_ENTRY_BYTES", 1)
    with pytest.raises(Unavailable, match="encoding"):
        ledger.capture_writer(other, PAYLOAD, process_identity(os.getpid()))
    assert ledger.find(other) is None
    with pytest.raises(Conflict, match="incomplete"):
        Ledger(ledger.root).handoff(IDENTITY, generation)


def test_capture_rejects_a_ledger_owned_by_another_runtime(tmp_path):
    from qqtools.plugins.qexp.runtime.responsibility_backfill import capture_local_writer

    ledger = Ledger.create(tmp_path / "ledger")
    with pytest.raises(ValueError, match="does not belong"):
        capture_local_writer(ledger, tmp_path / "other", IDENTITY, PAYLOAD, process_identity(os.getpid()))
    assert ledger.find(IDENTITY) is None


def test_unavailable_host_identity_cannot_authorize_cleanup(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runtime import responsibility_cleanup

    root = tmp_path / "runtime"
    root.mkdir()
    ledger = Ledger.create(responsibility_root(root))
    writer = process_identity(os.getpid())
    writer["boot_id"] = "10000000-0000-0000-0000-000000000001"
    ledger.capture_writer(IDENTITY, PAYLOAD, writer)

    def unavailable():
        raise RuntimeError("injected unavailable host identity")

    monkeypatch.setattr(responsibility_cleanup, "host_instance_id", unavailable)
    assert not complete_cleanup(ledger, root, cleanup_request())
    assert ledger.lookup(IDENTITY)["stage"] == "active"


@pytest.mark.parametrize("writers", [None, [], [{}], [{"pid": 42}]])
def test_corrupt_writer_inventory_is_not_an_empty_inventory(tmp_path, writers):
    ledger = Ledger.create(tmp_path / "ledger")
    ledger.capture_writer(IDENTITY, PAYLOAD, process_identity(os.getpid()))
    value = ledger.lookup(IDENTITY)
    value["captured_writers"] = writers
    key = identity_key(IDENTITY)
    ledger.io.replace(ledger.root / str(int(key[0], 16)) / f"e{key}", value)
    with pytest.raises(Unavailable, match="captured writer"):
        ledger.lookup(IDENTITY)
