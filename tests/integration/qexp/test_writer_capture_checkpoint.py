"""Capture intent precedes observation and survives both observation and replay loss."""

import multiprocessing
import os
from pathlib import Path

import pytest

from qqtools.plugins.qexp.infrastructure.host import host_instance_id
from qqtools.plugins.qexp.runtime import responsibility_capture as capture
from qqtools.plugins.qexp.runtime.paths import local_paths
from qqtools.plugins.qexp.runtime.responsibility import responsibility_root
from qqtools.plugins.qexp.runtime.responsibility_cleanup import CLEANUP_FORMAT, CleanupRequest, complete_cleanup
from qqtools.plugins.qexp.runtime.responsibility_import import move_legacy_record
from qqtools.plugins.qexp.runtime.responsibility_store import Conflict, Ledger, Unavailable
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json

pytestmark = pytest.mark.integration
IDENTITY = "task-attempt-1"
PAYLOAD = {"task_id": "task", "attempt_number": 1}


def setup_capture(root):
    root.mkdir(parents=True, exist_ok=True)
    ledger = Ledger.open_or_create(responsibility_root(root))
    return ledger, capture.WriterCaptureCheckpoint(ledger, root)


def observation():
    fields = Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()
    return {
        "identity": IDENTITY,
        "payload": PAYLOAD,
        "writer": {
            "host_id": host_instance_id(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "pid_namespace": Path("/proc/self/ns/pid").stat().st_ino,
            "pid": os.getpid(),
            "start_time_ticks": int(fields[19]),
        },
    }


def request():
    return CleanupRequest(
        IDENTITY,
        PAYLOAD,
        {"format": CLEANUP_FORMAT, "task_id": "task", "attempt_id": IDENTITY, "basis": "terminal_attempt"},
    )


def test_pending_is_durable_before_observation_and_survives_empty_batch(tmp_path):
    root = tmp_path / "runtime"
    ledger, checkpoint = setup_capture(root)
    with checkpoint.observe() as current:
        state = read_json(root / capture.CAPTURE_FILE)
        assert state["instance"] == ledger.instance and state["phase"] == "pending"
        assert state["pending"] == []
        current.record([])
        assert not complete_cleanup(ledger, root, request())
    assert read_json(root / capture.CAPTURE_FILE) == state
    assert not complete_cleanup(ledger, root, request())
    assert ledger.find(IDENTITY) is None


@pytest.mark.parametrize("is_maintenance", [False, True])
def test_pending_preserves_evidence_and_already_handed_off_cleanup(tmp_path, is_maintenance):
    root = tmp_path / "runtime"
    ledger, checkpoint = setup_capture(root)
    generation = ledger.publish(IDENTITY, PAYLOAD)
    cleanup = request()
    if is_maintenance:
        ledger.handoff(IDENTITY, generation, cleanup_receipt=cleanup.receipt)
        cleanup = CleanupRequest.from_entry(ledger.lookup(IDENTITY))
    path = local_paths(root)["observations"] / f"{IDENTITY}.json"
    atomic_replace(path, {"exit_observation": {"attempt_id": IDENTITY}})
    before = ledger.lookup(IDENTITY)
    with checkpoint.observe():
        pass
    assert not complete_cleanup(Ledger(ledger.root), root, cleanup)
    assert ledger.lookup(IDENTITY) == before and path.exists()


def crash_capture(root, boundary):
    ledger = Ledger(responsibility_root(root))
    checkpoint = capture.WriterCaptureCheckpoint(ledger, root)
    original = capture.atomic_replace
    writes = 0

    def write(path, value):
        nonlocal writes
        original(path, value)
        writes += 1
        if (boundary, writes) in {("initial", 1), ("pending", 2), ("cleared", 3)}:
            os._exit(86)

    capture.atomic_replace = write
    original_capture = ledger.capture_writer

    def publish(*args):
        value = original_capture(*args)
        if boundary == "member":
            os._exit(86)
        return value

    ledger.capture_writer = publish
    with checkpoint.observe() as current:
        observed = observation()
        if boundary == "observed":
            os._exit(86)
        current.record([observed])


def run_crash(root, boundary):
    process = multiprocessing.get_context("fork").Process(target=crash_capture, args=(root, boundary))
    process.start()
    try:
        process.join(10)
        assert not process.is_alive() and process.exitcode == 86
        pid = process.pid
    finally:
        if process.is_alive():
            process.kill()
        process.join(5)
        process.close()
    return pid


@pytest.mark.parametrize("boundary", ["initial", "observed", "pending", "member", "cleared"])
def test_crash_retains_capture_and_replays_exact_identity_after_process_exit(tmp_path, boundary):
    root = tmp_path / "runtime"
    ledger, checkpoint = setup_capture(root)
    pid = run_crash(root, boundary)
    before = read_json(root / capture.CAPTURE_FILE)
    if boundary == "pending":
        run_crash(root, "member")
        assert read_json(root / capture.CAPTURE_FILE) == before
    assert not complete_cleanup(ledger, root, request())
    with checkpoint.observe():
        member = ledger.find(IDENTITY)
        if boundary in {"pending", "member", "cleared"}:
            assert member["captured_writers"][0]["pid"] == pid
        else:
            assert member is None  # Interrupted observation must be rediscovered.
    after = read_json(root / capture.CAPTURE_FILE)
    assert after["capture_id"] == before["capture_id"] and after["pending"] == []
    assert not complete_cleanup(ledger, root, request())


def test_failed_initial_barrier_never_admits_observer(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    _, checkpoint = setup_capture(root)
    original = capture.atomic_replace

    def failed(path, value):
        original(path, value)
        raise OSError("injected checkpoint barrier failure")

    monkeypatch.setattr(capture, "atomic_replace", failed)
    with pytest.raises(OSError, match="barrier failure"):
        with checkpoint.observe():
            pytest.fail("observer ran before its checkpoint was acknowledged")
    assert capture.has_pending_writer_capture(root)
    with capture.capture_cleanup_guard(root) as can_cleanup:
        assert not can_cleanup


def test_reopen_retries_checkpoint_directory_barrier_before_observing(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    _, checkpoint = setup_capture(root)
    with checkpoint.observe():
        pass

    def failed(*_args):
        raise OSError("injected retry barrier failure")

    monkeypatch.setattr(capture.DurableIO, "sync_directory", failed)
    with pytest.raises(OSError, match="retry barrier"):
        with checkpoint.observe():
            pytest.fail("observer ran before checkpoint barrier retry")


def test_inflight_cleanup_excludes_observation_without_creating_checkpoint(tmp_path):
    root = tmp_path / "runtime"
    _, checkpoint = setup_capture(root)
    with capture.capture_cleanup_guard(root) as can_cleanup:
        assert can_cleanup
        with pytest.raises(Conflict, match="busy"):
            with checkpoint.observe():
                pytest.fail("observation overlapped cleanup")
    assert not capture.has_pending_writer_capture(root)


@pytest.mark.parametrize("damage", ["malformed", "dangling", "instance"])
def test_invalid_checkpoint_retains_evidence_and_cannot_restart_as_empty(tmp_path, damage):
    root = tmp_path / "runtime"
    _, checkpoint = setup_capture(root)
    path = root / capture.CAPTURE_FILE
    if damage == "malformed":
        path.write_text("invalid")
    elif damage == "dangling":
        path.symlink_to(root / "missing")
    else:
        with checkpoint.observe():
            pass
        state = read_json(path)
        state["instance"] = "0" * 32
        atomic_replace(path, state)
    with capture.capture_cleanup_guard(root) as can_cleanup:
        assert not can_cleanup
    with pytest.raises((ValueError, Unavailable)):
        with checkpoint.observe():
            pytest.fail("invalid checkpoint admitted an observation")


def test_pending_source_prevents_import_and_cleanup_through_another_runtime(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    _, checkpoint = setup_capture(source)
    ledger, _ = setup_capture(target)
    path = local_paths(source)["observations"] / f"{IDENTITY}.json"
    value = {"exit_observation": {"attempt_id": IDENTITY, "task_id": "task", "protocol_version": 1}}
    atomic_replace(path, value)
    generation = ledger.capture_source(IDENTITY, PAYLOAD, source)
    ledger.handoff(IDENTITY, generation, cleanup_receipt=request().receipt)
    with checkpoint.observe():
        pass
    with pytest.raises(Conflict, match="writer capture"):
        move_legacy_record(
            target,
            source,
            "observations",
            path,
            local_paths(target)["observations"] / path.name,
            is_destination_authoritative=False,
        )
    assert not complete_cleanup(ledger, target, CleanupRequest.from_entry(ledger.lookup(IDENTITY)))
    assert read_json(path) == value and ledger.find(IDENTITY) is not None


def test_failed_membership_publication_leaves_exact_replay_batch(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    ledger, checkpoint = setup_capture(root)
    observed = observation()
    with monkeypatch.context() as patch:

        def fail(*_args):
            raise OSError("injected capture write failure")

        patch.setattr(ledger, "capture_writer", fail)
        with pytest.raises(OSError, match="capture write failure"):
            with checkpoint.observe() as current:
                current.record([observed])
    assert read_json(root / capture.CAPTURE_FILE)["pending"] == [observed]
    with checkpoint.observe():
        assert ledger.lookup(IDENTITY)["captured_writers"] == [observed["writer"]]
    assert read_json(root / capture.CAPTURE_FILE)["pending"] == []


def test_record_requires_observation_scope_and_bounds_pending_batch(tmp_path):
    root = tmp_path / "runtime"
    ledger, checkpoint = setup_capture(root)
    observed = observation()
    with pytest.raises(RuntimeError, match="active observation"):
        checkpoint.record([observed])
    with checkpoint.observe() as current:
        with pytest.raises(ValueError, match="at most 64"):
            current.record([observed] * 65)
    assert ledger.find(IDENTITY) is None and capture.has_pending_writer_capture(root)


def test_uncertain_journal_write_invalidates_scope_before_another_batch(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    ledger, checkpoint = setup_capture(root)
    observed = observation()
    original = capture.atomic_replace

    def uncertain(path, value):
        original(path, value)
        if value["pending"]:
            raise OSError("injected uncertain pending commit")

    with checkpoint.observe() as current:
        with monkeypatch.context() as patch:
            patch.setattr(capture, "atomic_replace", uncertain)
            with pytest.raises(OSError, match="uncertain"):
                current.record([observed])
        with pytest.raises(RuntimeError, match="active observation"):
            current.record([])
    assert read_json(root / capture.CAPTURE_FILE)["pending"] == [observed]
    with checkpoint.observe():
        assert ledger.lookup(IDENTITY)["captured_writers"] == [observed["writer"]]


def test_capture_journal_size_refusal_keeps_global_retention(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    ledger, checkpoint = setup_capture(root)
    with checkpoint.observe() as current:
        before = read_json(root / capture.CAPTURE_FILE)
        monkeypatch.setattr(capture, "CAPTURE_BYTES", len((root / capture.CAPTURE_FILE).read_bytes()))
        with pytest.raises(ValueError, match="pending_writer_capture"):
            current.record([observation()])
    assert read_json(root / capture.CAPTURE_FILE) == before and ledger.find(IDENTITY) is None
    assert not complete_cleanup(ledger, root, request())


def test_stale_ledger_object_cannot_begin_capture_in_a_replaced_store(tmp_path):
    root = tmp_path / "runtime"
    ledger, checkpoint = setup_capture(root)
    ledger.root.rename(root / "retired-ledger")
    Ledger.create(ledger.root)
    with pytest.raises(Conflict, match="replaced"):
        with checkpoint.observe():
            pytest.fail("stale Ledger admitted observation into another instance")
    assert not capture.has_pending_writer_capture(root)
