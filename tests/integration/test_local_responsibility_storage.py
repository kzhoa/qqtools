"""Real filesystem/process checks for an isolated experiment, not qexp lifecycle."""

from __future__ import annotations

import importlib.util
import itertools
import json
import multiprocessing
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tests/helpers/qexp/local_responsibility.py"
SPEC = importlib.util.spec_from_file_location("local_responsibility", SCRIPT)
prototype = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prototype)
Ledger, DurableIO = prototype.Ledger, prototype.DurableIO
Conflict, Unavailable = prototype.Conflict, prototype.Unavailable


@pytest.fixture(scope="module", autouse=True)
def preload_crash_worker_imports():
    # Forkserver children otherwise repeat these imports at every crash boundary.
    # Preload definitions only; each child still opens its own ledger.
    multiprocessing.set_forkserver_preload(["__main__", "pytest", "qqtools.plugins.qexp.runtime.responsibility_store"])


def identities(count: int, bucket: int = 0) -> list[str]:
    result = []
    value = 0
    while len(result) < count:
        identity = f"project/attempt-{value}"
        if int(prototype.identity_key(identity)[0], 16) == bucket:
            result.append(identity)
        value += 1
    return result


def scan(ledger, limit=64):
    result = []
    for bucket in range(prototype.BUCKETS):
        cursor = None
        while True:
            records, cursor = ledger.page(bucket, cursor, limit)
            assert len(records) <= limit
            result.extend(records)
            if cursor is None:
                break
    assert len({record["identity"] for record in result}) == len(result)
    return sorted(result, key=lambda record: record["identity"])


def assert_compact(root, count):
    pages = 0
    entries = 0
    for bucket in range(16):
        directory = root / str(bucket)
        header = json.loads((directory / "header").read_text())
        expected = (header["count"] + 63) // 64
        assert {path.name for path in directory.glob("p[0-9]*")} == {f"p{i}" for i in range(expected)}
        assert not (directory / "pending").exists()
        assert not (directory / "scratch").exists()
        pages += expected
        entries += len(list(directory.glob("e*")))
        stages = header.get("stages")
        if stages is not None:
            assert stages["active"] + stages["maintenance"] == header["count"]
            for stage, prefix in (("active", "a"), ("maintenance", "m")):
                stage_count = stages[stage]
                assert {path.name for path in directory.glob(f"{prefix}[0-9]*")} == {
                    f"{prefix}{i}" for i in range((stage_count + 63) // 64)
                }
                keys = []
                for i in range((stage_count + 63) // 64):
                    keys.extend(json.loads((directory / f"{prefix}{i}").read_text())["entries"])
                assert len(keys) == len(set(keys)) == stage_count
                for slot, key in enumerate(keys):
                    entry = json.loads((directory / f"e{key}").read_text())
                    assert (entry["stage"], entry["stage_slot"]) == (stage, slot)
    assert entries == count
    return pages


def test_pagination_boundaries_and_no_inventory(tmp_path, monkeypatch):
    ledger = Ledger.create(tmp_path / "ledger")
    names = identities(130)
    for identity in names:
        ledger.publish(identity, {"task": "task", "attempt_number": 1})

    def forbid(*args, **kwargs):
        raise AssertionError("normal access must not inventory directories")

    monkeypatch.setattr(os, "scandir", forbid)
    monkeypatch.setattr(os, "listdir", forbid)
    for size in (1, 7, 63, 64):
        assert {entry["identity"] for entry in scan(Ledger(ledger.root), size)} == set(names)
    io = DurableIO()
    records, cursor = Ledger(ledger.root, io).page(0)
    assert len(records) == 64 and cursor is not None
    # Marker + fixed pending probe + header + one page + 64 locators.
    assert io.counts["reads"] == 68
    assert io.counts["writes"] == io.counts["directory_fsync"] == 0


def test_stale_cursor_generation_and_exact_retry(tmp_path):
    ledger = Ledger.create(tmp_path / "ledger")
    names = identities(66)
    for identity in names:
        ledger.publish(identity, {})
    _, cursor = ledger.page(0)
    generation = ledger.lookup(names[0])["generation"]
    ledger.io.counts.clear()
    assert ledger.publish(names[0], {}) == generation
    assert ledger.io.counts["writes"] == 0
    maintenance = ledger.handoff(names[0], generation)
    with pytest.raises(Conflict):
        ledger.page(0, cursor)
    with pytest.raises(Conflict):
        ledger.retire(names[0], generation)
    assert ledger.retire(names[0], maintenance)
    assert not ledger.retire(names[0], maintenance)
    replacement = ledger.publish(names[0], {})
    assert replacement > maintenance
    with pytest.raises(Conflict):
        ledger.retire(names[0], maintenance)
    assert ledger.lookup(names[0])["generation"] == replacement
    other = Ledger.create(tmp_path / "other")
    with pytest.raises(Conflict):
        other.page(0, cursor)


@pytest.mark.parametrize(
    "count,reuse_rounds",
    [
        pytest.param(65, 1, id="cross-page-reuse"),
        pytest.param(140, 4, id="repeated-churn", marks=pytest.mark.stress),
    ],
)
def test_churn_random_removal_and_page_reuse(tmp_path, count, reuse_rounds):
    ledger = Ledger.create(tmp_path / "ledger")
    randomizer = random.Random(487)
    names = identities(count)
    live = {}
    for identity in names:
        live[identity] = ledger.publish(identity, {})
    randomizer.shuffle(names)
    for index, identity in enumerate(names):
        generation = ledger.handoff(identity, live.pop(identity))
        assert ledger.retire(identity, generation)
        assert {entry["identity"] for entry in scan(ledger)} == set(live)
        assert_compact(ledger.root, len(live))
        if index % 20 == 0:
            ledger = Ledger(ledger.root)
    for _ in range(reuse_rounds):
        for identity in names[:70]:
            generation = ledger.publish(identity, {})
            assert ledger.retire(identity, ledger.handoff(identity, generation))
        assert_compact(ledger.root, 0)
    # Fixed 16 headers and locks + one marker; no historical locators or pages.
    assert sum(path.is_file() for path in ledger.root.rglob("*")) == 33


def prepare(root, case):
    ledger = Ledger.create(root)
    names = identities(68)
    count = {
        "publish_empty": 0,
        "publish_page": 64,
        "handoff": 1,
        "retire_tail": 1,
        "retire_same": 3,
        "retire_cross": 65,
        "capture_source": 0,
        "attach_source": 1,
        "resolve_locator": 1,
        "capture_local": 0,
        "resolve_local": 1,
        "capture_writer": 0,
        "attach_writer": 1,
        "capture_writer_source": 0,
        "attach_writer_source": 1,
        "reject_writer_entry": 1,
        "reject_writer_bucket": 1,
        "handoff_cross": 66,
        "retire_dual": 68,
        "build_stage": 3,
    }[case]
    for identity in names[:count]:
        if case == "resolve_local":
            ledger.capture_local(identity, {"task_id": None, "attempt_number": None})
        elif case == "resolve_locator":
            ledger.capture_source(identity, {"task_id": None, "attempt_number": None}, Path("/legacy-fixture"))
        elif case in {
            "attach_source",
            "attach_writer",
            "attach_writer_source",
            "reject_writer_entry",
            "reject_writer_bucket",
        }:
            ledger.publish(identity, {"task_id": "task", "attempt_number": 1})
        else:
            ledger.publish(identity, {"source_runtime": "fixture", "task_id": "task"})
    if case.startswith("retire"):
        ledger.handoff(names[0], ledger.lookup(names[0])["generation"])
    if case == "retire_dual":
        # Different members compact the all-members and maintenance indexes.
        for name in names[1:66]:
            ledger.handoff(name, ledger.lookup(name)["generation"])
    if case == "build_stage":
        remove_stage_projection(ledger)
    return names[count] if case.startswith("publish") else names[0]


def operation(ledger, case, identity):
    if case.startswith("publish"):
        ledger.publish(identity, {"source_runtime": "fixture", "task_id": "task"})
    elif case in {"capture_local", "resolve_local"}:
        ledger.capture_local(identity, {"task_id": "task", "attempt_number": 1})
    elif case in {"capture_source", "attach_source"}:
        ledger.capture_source(identity, {"task_id": "task", "attempt_number": 1}, Path("/legacy-fixture"))
    elif case == "resolve_locator":
        ledger.resolve_locator(identity, {"task_id": "task", "attempt_number": 1})
    elif case.startswith("reject_writer"):
        from qqtools.plugins.qexp.runtime import responsibility_store as storage

        size = len(storage.encode(ledger.lookup(identity)))
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(storage, "INITIAL_ENTRY_BYTES", size)
            if case == "reject_writer_bucket":
                patch.setattr(storage, "ENTRY_BYTES", size)
            with pytest.raises(Unavailable, match="encoding"):
                operation(ledger, "attach_writer", identity)
    elif case in {"capture_writer", "attach_writer", "capture_writer_source", "attach_writer_source"}:
        ledger.capture_writer(
            identity,
            {"task_id": "task", "attempt_number": 1},
            {
                "host_id": "fixture-host",
                "boot_id": "10000000-0000-0000-0000-000000000001",
                "pid_namespace": 42,
                "pid": 10,
                "start_time_ticks": 20,
            },
            source_root=Path("/legacy-fixture") if case.endswith("_source") else None,
        )
    elif case.startswith("handoff"):
        ledger.handoff(identity, ledger.lookup(identity)["generation"])
    elif case == "build_stage":
        ledger.build_stage_index(0, 1)
    else:
        ledger.retire(identity, ledger.lookup(identity)["generation"])


def test_writer_attachment_has_one_transaction_and_exact_retry_has_no_writes(tmp_path):
    identity = prepare(tmp_path / "ledger", "attach_writer")
    io = DurableIO()
    ledger = Ledger(tmp_path / "ledger", io)
    operation(ledger, "attach_writer", identity)
    assert io.counts["file_fsync"] == 3
    assert io.counts["directory_fsync"] == 3
    io.counts.clear()
    operation(ledger, "attach_writer", identity)
    assert io.counts["writes"] == io.counts["file_fsync"] == io.counts["directory_fsync"] == 0


def crash_child(root, case, identity, boundary, is_replay=False):
    calls = 0

    def crash(label):
        nonlocal calls
        calls += 1
        if calls == boundary:
            os._exit(86)  # No Python cleanup; kernel releases the leaf flock.

    ledger = Ledger(root, DurableIO(crash))
    if is_replay:
        ledger.page(0)
    else:
        operation(ledger, case, identity)


def run_child(target, *args):
    child = multiprocessing.get_context("forkserver").Process(target=target, args=args)
    child.start()
    try:
        child.join(15)
        assert not child.is_alive(), "isolated child exceeded 15s"
        return child.exitcode
    finally:
        if child.is_alive():
            child.kill()
        child.join(5)
        child.close()


@pytest.mark.parametrize("case", ["publish_page", "handoff", "retire_cross"])
def test_process_crash_around_durable_intent_recovers(tmp_path, case):
    seed = tmp_path / "seed"
    identity = prepare(seed, case)
    before = scan(Ledger(seed))
    expected_root = tmp_path / "expected"
    shutil.copytree(seed, expected_root)
    trace = DurableIO(lambda _: None)
    operation(Ledger(expected_root, trace), case, identity)
    after = scan(Ledger(expected_root))
    intent = trace.events.index("pending:directory_fsync") + 1
    for boundary in (intent - 1, intent):
        trial = tmp_path / f"cut-{boundary}"
        shutil.copytree(seed, trial)
        assert run_child(crash_child, trial, case, identity, boundary) == 86
        actual = scan(Ledger(trial))
        assert actual in (before, after)
        if boundary >= intent:
            assert actual == after
        assert scan(Ledger(trial)) == actual
        assert_compact(trial, len(actual))


@pytest.mark.parametrize("case", ["reject_writer_entry", "reject_writer_bucket"])
@pytest.mark.slow
def test_incomplete_writer_marker_survives_transaction_and_replay_crashes(tmp_path, case):
    seed = tmp_path / "seed"
    identity = prepare(seed, case)

    def snapshot(root):
        ledger = Ledger(root)
        records = scan(ledger)
        header = json.loads((root / "0/header").read_text())
        return records, header.get("writer_capture_incomplete", False)

    def assert_retained(root, expected):
        assert snapshot(root) == expected
        ledger = Ledger(root)
        entry = ledger.lookup(identity)
        with pytest.raises(Conflict, match="incomplete"):
            ledger.handoff(identity, entry["generation"])
        assert_compact(root, 1)

    before = snapshot(seed)
    trace_root = tmp_path / "trace"
    shutil.copytree(seed, trace_root)
    trace = DurableIO(lambda _: None)
    operation(Ledger(trace_root, trace), case, identity)
    after = snapshot(trace_root)
    assert before != after
    assert_retained(trace_root, after)
    intent = trace.events.index("pending:directory_fsync") + 1
    for boundary in range(1, len(trace.events) + 1):
        trial = tmp_path / f"cut-{boundary}"
        shutil.copytree(seed, trial)
        assert run_child(crash_child, trial, case, identity, boundary) == 86
        assert snapshot(trial) in (before, after)
        if boundary >= intent:
            assert_retained(trial, after)

    # A second crash during recovery must not discard the cleanup prohibition.
    assert run_child(crash_child, seed, case, identity, intent) == 86
    replay_root = tmp_path / "replay-trace"
    shutil.copytree(seed, replay_root)
    replay = DurableIO(lambda _: None)
    Ledger(replay_root, replay).page(0)
    for boundary in range(1, len(replay.events) + 1):
        trial = tmp_path / f"replay-cut-{boundary}"
        shutil.copytree(seed, trial)
        assert run_child(crash_child, trial, case, identity, boundary, True) == 86
        assert_retained(trial, after)


@pytest.mark.parametrize(
    "case",
    [
        "publish_empty",
        "publish_page",
        "handoff",
        "retire_tail",
        "retire_same",
        "retire_cross",
        "capture_source",
        "attach_source",
        "resolve_locator",
        "capture_local",
        "resolve_local",
        "capture_writer",
        "attach_writer",
        "capture_writer_source",
        "attach_writer_source",
        "handoff_cross",
        "retire_dual",
    ],
)
@pytest.mark.slow
def test_process_crash_at_every_storage_boundary(tmp_path, case):
    seed = tmp_path / "seed"
    identity = prepare(seed, case)
    before = scan(Ledger(seed))
    trace_root = tmp_path / "trace"
    shutil.copytree(seed, trace_root)
    io = DurableIO(lambda _: None)
    operation(Ledger(trace_root, io), case, identity)
    after = scan(Ledger(trace_root))
    durable_intent = io.events.index("pending:directory_fsync") + 1
    for boundary in range(1, len(io.events) + 1):
        trial = tmp_path / f"cut-{boundary}"
        shutil.copytree(seed, trial)
        assert run_child(crash_child, trial, case, identity, boundary) == 86
        actual = scan(Ledger(trial))
        assert actual in (before, after), (case, boundary, io.events[boundary - 1])
        if boundary >= durable_intent:
            assert actual == after
        assert scan(Ledger(trial)) == actual  # Replay is idempotent after reopen.
        assert_compact(trial, len(actual))


@pytest.mark.parametrize(
    "case",
    [
        "publish_page",
        "handoff",
        "retire_cross",
        "handoff_cross",
        "retire_dual",
        "capture_writer",
        "attach_writer",
        "capture_writer_source",
        "attach_writer_source",
    ],
)
@pytest.mark.slow
def test_replay_itself_can_crash_repeatedly(tmp_path, case):
    seed = tmp_path / "seed"
    identity = prepare(seed, case)
    expected_root = tmp_path / "expected"
    shutil.copytree(seed, expected_root)
    operation(Ledger(expected_root), case, identity)
    expected = scan(Ledger(expected_root))
    # The fourth event commits the durable pending transaction, before its images.
    assert run_child(crash_child, seed, case, identity, 4) == 86
    trace_root = tmp_path / "trace"
    shutil.copytree(seed, trace_root)
    io = DurableIO(lambda _: None)
    Ledger(trace_root, io).page(0)
    for boundary in range(1, len(io.events) + 1):
        trial = tmp_path / f"replay-{boundary}"
        shutil.copytree(seed, trial)
        assert run_child(crash_child, trial, case, identity, boundary, True) == 86
        assert scan(Ledger(trial)) == expected
        assert_compact(trial, len(expected))


def writer_child(root, names):
    ledger = Ledger(root)
    for identity in names:
        ledger.publish(identity, {})


def test_concurrent_publishers_have_one_membership_per_identity(tmp_path):
    ledger = Ledger.create(tmp_path / "ledger")
    names = identities(80)
    children = [
        multiprocessing.get_context("forkserver").Process(target=writer_child, args=(ledger.root, names))
        for _ in range(4)
    ]
    try:
        for child in children:
            child.start()
        for child in children:
            child.join(20)
            assert child.exitcode == 0
    finally:
        for child in children:
            if child.is_alive():
                child.kill()
            if child.pid is not None:
                child.join(5)
                child.close()
    assert {record["identity"] for record in scan(ledger)} == set(names)
    assert_compact(ledger.root, 80)


def test_missing_locator_and_stale_redo_fail_closed(tmp_path):
    ledger = Ledger.create(tmp_path / "ledger")
    names = identities(2)
    ledger.publish(names[0], {})
    key = prototype.identity_key(names[0])
    locator = ledger.root / "0" / f"e{key}"
    original = locator.read_bytes()
    locator.unlink()
    with pytest.raises(Unavailable):
        ledger.page(0)
    locator.write_bytes(original)  # Explicit test-only corruption restoration.
    assert run_child(crash_child, ledger.root, "publish_page", names[1], 4) == 86
    pending = (ledger.root / "0/pending").read_bytes()
    ledger.page(0)
    ledger.handoff(names[0], ledger.lookup(names[0])["generation"])
    (ledger.root / "0/pending").write_bytes(pending)
    with pytest.raises(Unavailable, match="different generation"):
        ledger.page(0)
    assert ledger.page(1) == ([], None)  # One broken bucket does not block another.


def test_fsync_failure_does_not_acknowledge_publication(tmp_path, monkeypatch):
    ledger = Ledger.create(tmp_path / "ledger")

    def fail(fd):
        raise OSError("injected fsync failure")

    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(Unavailable, match="fsync failure"):
        ledger.publish(identities(1)[0], {})
    monkeypatch.undo()
    assert scan(Ledger(ledger.root)) == []


def test_encoding_budget_is_rejected_before_mutation(tmp_path):
    ledger = Ledger.create(tmp_path / "ledger")
    with pytest.raises(Unavailable, match="encoding budget"):
        ledger.publish("long-input", {"value": "x" * prototype.ENTRY_BYTES})
    assert scan(ledger) == []
    assert ledger.io.counts["writes"] == 0
    with pytest.raises(FileExistsError):
        Ledger.create(ledger.root)


def test_handoff_covers_idempotent_external_cleanup(tmp_path):
    ledger = Ledger.create(tmp_path / "ledger")
    evidence = tmp_path / "evidence"
    evidence.write_text("terminal fixture")
    identity = identities(1)[0]
    generation = ledger.publish(identity, {"evidence": str(evidence)})
    maintenance = ledger.handoff(identity, generation)
    # Agent stops after external cleanup but before retirement.
    evidence.unlink()
    reopened = Ledger(ledger.root)
    entry = reopened.lookup(identity)
    assert entry["stage"] == "maintenance"
    Path(entry["payload"]["evidence"]).unlink(missing_ok=True)
    assert reopened.retire(identity, maintenance)
    assert scan(reopened) == []


def test_continuous_mutation_invalidates_dense_cursors(tmp_path):
    """Characterize a known integration blocker, not a fairness acceptance test."""
    ledger = Ledger.create(tmp_path / "ledger")
    names = identities(75)
    for identity in names[:65]:
        ledger.publish(identity, {})
    seen = set()
    for identity in names[65:]:
        records, cursor = ledger.page(0)
        seen.update(record["identity"] for record in records)
        ledger.publish(identity, {})
        with pytest.raises(Conflict):
            ledger.page(0, cursor)
    assert names[64] not in seen


@pytest.mark.parametrize("case", ["publish_page", "handoff", "retire_cross"])
@pytest.mark.slow
def test_power_cut_model_restores_last_synced_directory(tmp_path, case):
    """Model one allowed power-loss outcome: discard unsynced namespace changes.

    Real fsync is retained. This is not hardware/filesystem qualification and does
    not exhaust every possible persistence reordering on an actual device.
    """
    seed = tmp_path / "seed"
    identity = prepare(seed, case)
    before = scan(Ledger(seed))
    expected_root = tmp_path / "expected"
    shutil.copytree(seed, expected_root)
    trace = DurableIO(lambda _: None)
    operation(Ledger(expected_root, trace), case, identity)
    after = scan(Ledger(expected_root))
    intent = trace.events.index("pending:directory_fsync") + 1

    class PowerCut(Exception):
        pass

    for boundary in range(1, len(trace.events) + 1):
        root = tmp_path / f"model-{boundary}"
        shutil.copytree(seed, root)
        directory = root / "0"
        durable = {path.name: path.read_bytes() for path in directory.iterdir()}
        events = 0

        def checkpoint(label):
            nonlocal durable, events
            if label.endswith(":directory_fsync"):
                durable = {path.name: path.read_bytes() for path in directory.iterdir()}
            events += 1
            if events == boundary:
                raise PowerCut

        with pytest.raises(PowerCut):
            operation(Ledger(root, DurableIO(checkpoint)), case, identity)
        # Crash-model reset applies only to this private fixture, after the lock
        # has been released. No other process has a handle to this directory.
        for path in directory.iterdir():
            path.unlink()
        for name, content in durable.items():
            (directory / name).write_bytes(content)
        assert scan(Ledger(root)) == (after if boundary >= intent else before)


def test_large_locator_cross_page_retirement_stays_bounded(tmp_path):
    ledger = Ledger.create(tmp_path / "ledger")
    names = identities(65)
    for identity in names:
        ledger.publish(identity, {"reason": "x" * 60000})
    generation = ledger.handoff(names[0], ledger.lookup(names[0])["generation"])
    ledger.io = DurableIO()
    assert ledger.retire(names[0], generation)
    assert ledger.io.counts["max_images"] == 6
    assert ledger.io.counts["max_transaction_bytes"] < 80 * 1024
    assert ledger.io.counts["file_fsync"] + ledger.io.counts["directory_fsync"] == 7
    assert len(scan(ledger)) == 64


def test_corrupt_pending_transaction_is_not_applied(tmp_path):
    root = tmp_path / "ledger"
    identity = prepare(root, "publish_empty")
    assert run_child(crash_child, root, "publish_empty", identity, 4) == 86
    path = root / "0/pending"
    value = json.loads(path.read_text())
    value["images"]["header"]["count"] += 1
    path.write_text(json.dumps(value))
    with pytest.raises(Unavailable, match="checksum"):
        Ledger(root).page(0)
    assert json.loads((root / "0/header").read_text())["count"] == 0


def test_profiler_cli_records_counters_and_refuses_existing_output(tmp_path):
    output = tmp_path / "profile"
    args = [
        sys.executable,
        "-m",
        "scripts.qualification.profile_local_responsibility",
        "--output",
        str(output),
        "--cycles",
        "1",
        "--samples",
        "1",
        "--live",
        "4",
    ]
    result = subprocess.run(args, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "results.json").read_text())
    assert report["live_counts"] == [4]
    assert len(report["churn_remaining_files"]) == 39
    assert len(report["source_sha256"]) == 3
    assert report["summary"]["publish"]["max_counts"]["file_fsync"] == 5
    assert report["summary"]["publish_retry"]["max_counts"].get("writes", 0) == 0
    before = (output / "results.json").read_bytes()
    result = subprocess.run(args, capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert (output / "results.json").read_bytes() == before


@pytest.mark.parametrize("limit", [1, 7, 63, 64])
def test_reverse_service_sweep_finishes_during_continuous_appends(tmp_path, limit):
    ledger = Ledger.create(tmp_path / "ledger")
    names = identities(400)
    initial = set(names[:130])
    for identity in initial:
        ledger.publish(identity, {})
    cursor = None
    seen = set()
    for step in range(130):
        records, cursor = ledger.service_page(0, cursor, limit)
        assert len(records) <= limit
        seen.update(entry["identity"] for entry in records)
        ledger.publish(names[130 + step], {})
        if cursor is None:
            break
    else:
        pytest.fail("reverse sweep did not finish within its initial slot count")
    assert initial <= seen
    assert len(seen) == 130  # New appends did not extend this sweep.


def test_reverse_service_compaction_cannot_hide_retained_members(tmp_path):
    ledger = Ledger.create(tmp_path / "ledger")
    names = identities(300)
    live = set(names[:130])
    for identity in names[:130]:
        ledger.publish(identity, {})
    initial = set(live)
    seen = set()
    cursor = None
    randomizer = random.Random(812)
    for step in range(130):
        records, cursor = ledger.service_page(0, cursor, 7)
        seen.update(entry["identity"] for entry in records)
        # Delete ahead of and behind the cursor; move tail members across pages.
        for identity in randomizer.sample(sorted(live), min(3, len(live))):
            generation = ledger.handoff(identity, ledger.lookup(identity)["generation"])
            ledger.retire(identity, generation)
            live.remove(identity)
        identity = names[130 + step]
        ledger.publish(identity, {})
        live.add(identity)
        if cursor is None:
            break
    else:
        pytest.fail("compaction prevented finite sweep completion")
    assert initial & live <= seen
    assert_compact(ledger.root, len(live))


def test_round_robin_service_does_not_starve_other_buckets_or_hide_failures(tmp_path):
    ledger = Ledger.create(tmp_path / "ledger")
    large = identities(100)
    small = identities(1, bucket=1)[0]
    for identity in large:
        ledger.publish(identity, {})
    ledger.publish(small, {})
    traversal = prototype.ServiceTraversal(ledger)
    assert len(traversal.take(1)) == 1
    assert traversal.take(1)[0]["identity"] == small
    assert not traversal.has_completed_initial_sweeps
    for _ in range(16 * 100):
        traversal.take(1)
        if traversal.has_completed_initial_sweeps:
            break
    assert traversal.has_completed_initial_sweeps
    # A broken bucket must not pin the service turn or turn into a healthy empty set.
    traversal.next_bucket = 0
    (ledger.root / "0/header").write_text("broken")
    with pytest.raises(Unavailable):
        traversal.take(1)
    assert traversal.take(1)[0]["identity"] == small


@pytest.mark.parametrize("case", ["publish_page", "handoff", "retire_cross", "handoff_cross", "retire_dual"])
@pytest.mark.slow
def test_redo_recovers_every_mixture_of_persisted_after_images(tmp_path, case):
    seed = tmp_path / "seed"
    identity = prepare(seed, case)
    before = {path.name: path.read_bytes() for path in (seed / "0").iterdir()}
    expected_root = tmp_path / "expected"
    shutil.copytree(seed, expected_root)
    operation(Ledger(expected_root), case, identity)
    expected = scan(Ledger(expected_root))
    assert run_child(crash_child, seed, case, identity, 4) == 86
    transaction = json.loads((seed / "0/pending").read_text())
    names = sorted(transaction["images"])
    # After intent durability, the filesystem may persist any subset of renamed
    # or unlinked names before the grouped directory barrier. Include header-first.
    for number, bits in enumerate(itertools.product((False, True), repeat=len(names))):
        root = tmp_path / f"mixture-{number}"
        shutil.copytree(seed, root)
        for name, has_after_image in zip(names, bits, strict=True):
            path = root / "0" / name
            value = transaction["images"][name]
            if has_after_image:
                if value is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(prototype.encode(value))
            elif name in before:
                path.write_bytes(before[name])
            else:
                path.unlink(missing_ok=True)
        assert scan(Ledger(root)) == expected
        assert_compact(root, len(expected))


def initialize_with_io(root, io):
    class InitializingLedger(Ledger):
        io_type = staticmethod(lambda: io)

    return InitializingLedger.open_or_create(root)


def crash_initialization(root, boundary):
    calls = 0

    def crash(_label):
        nonlocal calls
        calls += 1
        if calls == boundary:
            os._exit(86)

    initialize_with_io(root, DurableIO(crash))


@pytest.mark.slow
def test_initialization_resumes_after_every_process_crash_boundary(tmp_path):
    io = DurableIO(lambda _: None)
    initialize_with_io(tmp_path / "trace", io)
    assert io.counts["file_fsync"] + io.counts["directory_fsync"] == 55
    assert io.events[-3:] == ["root:directory_fsync", "initializing:unlink", "initializing:directory_fsync"]
    for boundary in range(1, len(io.events) + 1):
        root = tmp_path / f"cut-{boundary}"
        assert run_child(crash_initialization, root, boundary) == 86
        ledger = Ledger.open_or_create(root)
        assert scan(ledger) == [], (boundary, io.events[boundary - 1])
        ledger.publish("surviving-attempt", {"attempt_number": 1})
        reopened = Ledger.open_or_create(root)
        assert reopened.lookup("surviving-attempt")["payload"] == {"attempt_number": 1}
        assert not root.with_name(f".{root.name}.building").exists()
        assert not (root / "initializing").exists()
        assert_compact(root, 1)


def test_initialization_retries_failed_root_rename_barrier(tmp_path):
    class FailedParentBarrier(DurableIO):
        def sync_directory(self, path, label):
            if label == "root":
                raise OSError("root rename not durable")
            super().sync_directory(path, label)

    root = tmp_path / "ledger"
    with pytest.raises(OSError, match="root rename not durable"):
        initialize_with_io(root, FailedParentBarrier())
    assert (root / "initializing").exists()
    io = DurableIO(lambda _: None)
    initialize_with_io(root, io)
    assert io.events == ["root:directory_fsync", "initializing:unlink", "initializing:directory_fsync"]
    quiet = DurableIO()
    initialize_with_io(root, quiet)
    assert quiet.counts["writes"] == quiet.counts["directory_fsync"] == 0


@pytest.mark.parametrize("damage", ["published", "occupied", "unexpected", "symlink", "bad_marker"])
def test_initialization_never_resets_unknown_or_occupied_storage(tmp_path, damage):
    root = tmp_path / "ledger"
    staging = root.with_name(f".{root.name}.building")
    if damage == "published":
        root.mkdir()
        (root / "marker").write_text("broken")
        protected = root / "marker"
    elif damage == "occupied":
        Ledger.create(staging).publish("retained", {})
        protected = staging / str(int(prototype.identity_key("retained")[0], 16)) / "header"
    elif damage == "symlink":
        protected = tmp_path / "foreign"
        protected.mkdir()
        staging.symlink_to(protected, target_is_directory=True)
    else:
        staging.mkdir()
        protected = staging / ("foreign" if damage == "unexpected" else "marker")
        protected.write_text("not initialization data")
    before = protected.read_bytes() if protected.is_file() else sorted(protected.iterdir())
    with pytest.raises(Unavailable):
        Ledger.open_or_create(root)
    after = protected.read_bytes() if protected.is_file() else sorted(protected.iterdir())
    assert before == after
    if damage != "published":
        assert not root.exists()


def initialize_and_publish(root, identity):
    from qqtools.plugins.qexp.runtime.locks import exclusive

    with exclusive(root.parent / "initialization.lock"):
        ledger = Ledger.open_or_create(root)
    ledger.publish(identity, {})


def test_concurrent_first_publishers_share_one_initialized_store(tmp_path):
    root = tmp_path / "ledger"
    children = [
        multiprocessing.get_context("forkserver").Process(
            target=initialize_and_publish,
            args=(root, f"attempt-{number}"),
        )
        for number in range(4)
    ]
    try:
        for child in children:
            child.start()
        for child in children:
            child.join(15)
            assert child.exitcode == 0
    finally:
        for child in children:
            if child.is_alive():
                child.kill()
            if child.pid is not None:
                child.join(5)
                child.close()
    assert {entry["identity"] for entry in scan(Ledger(root))} == {f"attempt-{number}" for number in range(4)}
    assert_compact(root, 4)


def cleanup_fixture(runtime):
    from qqtools.plugins.qexp.runtime.paths import local_paths
    from qqtools.plugins.qexp.runtime.responsibility_cleanup import CLEANUP_FORMAT, FLAT_EVIDENCE, CleanupRequest

    runtime.mkdir()
    ledger = Ledger.create(runtime / "members")
    request = CleanupRequest(
        "task-attempt-1",
        {"task_id": "task", "attempt_number": 1},
        {"format": CLEANUP_FORMAT, "task_id": "task", "attempt_id": "task-attempt-1", "basis": "terminal_attempt"},
    )
    ledger.publish(request.identity, request.payload)
    paths = local_paths(runtime)
    evidence = []
    for name in FLAT_EVIDENCE:
        paths[name].mkdir(parents=True, exist_ok=True)
        path = paths[name] / f"{request.identity}.json"
        path.write_text("{}")
        evidence.append(path)
    decisions = paths["termination_decisions"] / request.identity
    decisions.mkdir(parents=True)
    for index in range(2):
        path = decisions / f"{index}.json"
        path.write_text("{}")
        evidence.append(path)
    return request, evidence


def crash_cleanup(runtime, request, boundary):
    from qqtools.plugins.qexp.runtime.responsibility_cleanup import complete_cleanup

    count = 0

    def crash(_event):
        nonlocal count
        count += 1
        if count == boundary:
            os._exit(86)

    complete_cleanup(Ledger(runtime / "members", DurableIO(crash)), runtime, request)


@pytest.mark.slow
def test_cleanup_process_crashes_preserve_receipt_until_durable_deletion(tmp_path):
    from qqtools.plugins.qexp.runtime.responsibility_cleanup import CleanupRequest, complete_cleanup

    seed = tmp_path / "seed"
    request, evidence = cleanup_fixture(seed)
    trial = tmp_path / "trace"
    shutil.copytree(seed, trial)
    io = DurableIO(lambda _: None)
    assert complete_cleanup(Ledger(trial / "members", io), trial, request)
    assert scan(Ledger(trial / "members")) == []
    assert any(event.endswith(":unlink") and request.identity in event for event in io.events)
    for boundary in range(1, len(io.events) + 1):
        runtime = tmp_path / f"cleanup-cut-{boundary}"
        shutil.copytree(seed, runtime)
        assert run_child(crash_cleanup, runtime, request, boundary) == 86
        ledger = Ledger(runtime / "members")
        entries = scan(ledger)
        surviving = [(runtime / path.relative_to(seed)).exists() for path in evidence]
        if not entries:
            assert not any(surviving), (boundary, io.events[boundary - 1])
            continue
        assert len(entries) == 1
        entry = entries[0]
        if entry["stage"] == "active":
            assert all(surviving), "evidence was deleted before durable handoff"
            replay = request  # External terminal proof is still available.
        else:
            assert entry["cleanup_receipt"] == request.receipt
            replay = CleanupRequest.from_entry(entry)
        assert complete_cleanup(ledger, runtime, replay)
        assert scan(Ledger(runtime / "members")) == []
        assert not any((runtime / path.relative_to(seed)).exists() for path in evidence)


def test_cleanup_failed_final_barrier_keeps_standalone_replay_proof(tmp_path):
    from qqtools.plugins.qexp.runtime.responsibility_cleanup import CleanupRequest, complete_cleanup

    class FailedDeletionBarrier(DurableIO):
        def sync_directory(self, path, label):
            if label == "cleanup_evidence_directories":
                raise OSError("injected deletion barrier failure")
            super().sync_directory(path, label)

    runtime = tmp_path / "runtime"
    request, evidence = cleanup_fixture(runtime)
    with pytest.raises(OSError, match="deletion barrier"):
        complete_cleanup(Ledger(runtime / "members", FailedDeletionBarrier()), runtime, request)
    assert not any(path.exists() for path in evidence)
    ledger = Ledger(runtime / "members")
    replay = CleanupRequest.from_entry(ledger.lookup(request.identity))
    assert replay is not None
    # No evidence or shared Task truth remains; the receipt alone owns the retry.
    assert complete_cleanup(ledger, runtime, replay)
    assert scan(ledger) == []


def test_stale_cleanup_receipt_cannot_delete_republished_membership(tmp_path):
    from qqtools.plugins.qexp.runtime.responsibility_cleanup import CleanupRequest, complete_cleanup

    runtime = tmp_path / "runtime"
    request, evidence = cleanup_fixture(runtime)
    ledger = Ledger(runtime / "members")
    generation = ledger.handoff(
        request.identity, ledger.lookup(request.identity)["generation"], cleanup_receipt=request.receipt
    )
    replay = CleanupRequest.from_entry(ledger.lookup(request.identity))
    ledger.retire(request.identity, generation)
    ledger.publish(request.identity, request.payload)
    with pytest.raises(Conflict, match="different membership"):
        complete_cleanup(ledger, runtime, replay)
    assert all(path.exists() for path in evidence)
    assert ledger.lookup(request.identity)["stage"] == "active"


def test_old_bare_maintenance_stage_is_not_cleanup_permission(tmp_path):
    from qqtools.plugins.qexp.runtime.responsibility_cleanup import CleanupRequest

    ledger = Ledger.create(tmp_path / "ledger")
    generation = ledger.publish("identity", {"task_id": "task", "attempt_number": 1})
    ledger.handoff("identity", generation)
    assert CleanupRequest.from_entry(ledger.lookup("identity")) is None


def remove_stage_projection(ledger):
    """Construct the supported base-only format, without changing membership truth."""
    for bucket in range(16):
        directory = ledger.root / str(bucket)
        header = json.loads((directory / "header").read_text())
        header.pop("stages", None)
        (directory / "header").write_bytes(prototype.encode(header))
        for path in directory.glob("e*"):
            entry = json.loads(path.read_text())
            entry.pop("stage_slot", None)
            path.write_bytes(prototype.encode(entry))
        for prefix in ("a", "m"):
            for path in directory.glob(f"{prefix}[0-9]*"):
                path.unlink()


def stage_scan(ledger, stage, limit=64):
    records = []
    for bucket in range(16):
        cursor = None
        while True:
            page, cursor = ledger.service_page(bucket, cursor, limit, stage=stage)
            records.extend(page)
            if cursor is None:
                break
    return records


def test_active_traversal_never_reads_maintenance_or_base_pages(tmp_path):
    ledger = Ledger.create(tmp_path / "ledger")
    names = identities(134)
    for identity in names:
        generation = ledger.publish(identity, {})
        if identity in names[:130]:
            ledger.handoff(identity, generation)
    forbidden = {f"e{prototype.identity_key(identity)}" for identity in names[:130]}

    class ActiveReadsOnly(DurableIO):
        def read(self, path, limit):
            assert path.name not in forbidden
            assert not (path.name[0] in "pm" and path.name[1:].isdigit())
            return super().read(path, limit)

    ledger.io = ActiveReadsOnly()
    assert {entry["identity"] for entry in stage_scan(ledger, "active")} == set(names[130:])
    # 16 pending probes, 16 headers, one active page and four active locators.
    assert ledger.io.counts["reads"] == 37
    assert ledger.io.counts["writes"] == 0


@pytest.mark.parametrize("stage", ["active", "maintenance"])
def test_stage_sweep_finishes_under_publication_handoff_and_retirement(tmp_path, stage):
    ledger = Ledger.create(tmp_path / "ledger")
    names = identities(400)
    initial = set(names[:130])
    retained = set(initial)
    for identity in names[:130]:
        generation = ledger.publish(identity, {})
        if stage == "maintenance":
            ledger.handoff(identity, generation)
    seen = set()
    cursor = None
    for step in range(130):
        records, cursor = ledger.service_page(0, cursor, 7, stage=stage)
        seen.update(entry["identity"] for entry in records)
        removed = names[step]
        generation = ledger.lookup(removed)["generation"]
        if stage == "active":
            ledger.handoff(removed, generation)
        else:
            ledger.retire(removed, generation)
        retained.discard(removed)
        generation = ledger.publish(names[130 + step], {})
        if stage == "maintenance":
            ledger.handoff(names[130 + step], generation)
        if cursor is None:
            break
    else:
        pytest.fail("stage sweep did not finish")
    assert retained <= seen
    assert_compact(ledger.root, len(scan(ledger)))


def test_stage_cursor_cannot_cross_stage_boundary(tmp_path):
    ledger = Ledger.create(tmp_path / "ledger")
    for identity in identities(2):
        ledger.publish(identity, {})
    _, cursor = ledger.service_page(0, limit=1, stage="active")
    assert cursor is not None
    for stage in (None, "maintenance"):
        with pytest.raises(Conflict):
            ledger.service_page(0, cursor, stage=stage)


def test_dual_compaction_preserves_both_locators_and_bounds(tmp_path):
    identity = prepare(tmp_path / "ledger", "retire_dual")
    ledger = Ledger(tmp_path / "ledger")
    names = identities(68)
    maintenance_tail = ledger.lookup(names[65])
    all_tail = ledger.lookup(names[67])
    ledger.io.counts.clear()
    operation(ledger, "retire_dual", identity)
    assert ledger.io.counts["max_images"] == 8
    assert ledger.io.counts["max_transaction_bytes"] < prototype.TRANSACTION_BYTES
    assert ledger.io.counts["file_fsync"] + ledger.io.counts["directory_fsync"] == 11
    assert ledger.lookup(names[65]) == {**maintenance_tail, "stage_slot": 0}
    assert ledger.lookup(names[67]) == {**all_tail, "slot": 0}
    assert_compact(ledger.root, 67)


def test_resumable_stage_build_preserves_generations_during_mutation(tmp_path):
    ledger = Ledger.create(tmp_path / "ledger")
    names = identities(160)
    for identity in names[:130]:
        generation = ledger.publish(identity, {})
        if identity in names[:65]:
            ledger.handoff(identity, generation, cleanup_receipt={"proof": identity})
    original = {entry["identity"]: entry for entry in scan(ledger)}
    remove_stage_projection(ledger)
    assert len(scan(ledger)) == 130
    with pytest.raises(Unavailable, match="incomplete"):
        ledger.service_page(0, stage="active")
    for step in range(20):
        ledger = Ledger(ledger.root)
        if ledger.build_stage_index(0, 7):
            break
        # Retire an unindexed entry; an indexed tail moves below the build cursor.
        ledger.retire(names[step], original[names[step]]["generation"])
        ledger.publish(names[130 + step], {})
    else:
        pytest.fail("stage build did not finish under churn")
    for bucket in range(1, 16):
        assert ledger.build_stage_index(bucket)
    for entry in scan(ledger):
        if entry["identity"] in original:
            before = original[entry["identity"]]
            assert entry["generation"] == before["generation"]
            assert entry.get("cleanup_receipt") == before.get("cleanup_receipt")
    for stage in ("active", "maintenance"):
        assert {entry["identity"] for entry in stage_scan(ledger, stage)} == {
            entry["identity"] for entry in scan(ledger) if entry["stage"] == stage
        }
    assert_compact(ledger.root, 130)
    ledger.io.counts.clear()
    assert ledger.build_stage_index(0)
    assert ledger.io.counts["writes"] == 0


@pytest.mark.slow
def test_stage_build_replays_every_process_crash_without_changing_ownership(tmp_path):
    seed = tmp_path / "seed"
    identity = prepare(seed, "build_stage")
    before = scan(Ledger(seed))
    trace_root = tmp_path / "trace"
    shutil.copytree(seed, trace_root)
    io = DurableIO(lambda _: None)
    operation(Ledger(trace_root, io), "build_stage", identity)
    for boundary in range(1, len(io.events) + 1):
        root = tmp_path / f"build-cut-{boundary}"
        shutil.copytree(seed, root)
        assert run_child(crash_child, root, "build_stage", identity, boundary) == 86
        ledger = Ledger(root)
        for _ in range(3):
            if ledger.build_stage_index(0, 1):
                break
        else:
            pytest.fail("build did not finish after replay")
        after = scan(ledger)
        assert [{key: value for key, value in entry.items() if key != "stage_slot"} for entry in after] == before
        assert {entry["identity"] for entry in ledger.service_page(0, stage="active")[0]} == {
            entry["identity"] for entry in before
        }
        assert_compact(root, 3)


def test_cleanup_handoff_cannot_replace_existing_proof(tmp_path):
    runtime = tmp_path / "runtime"
    request, _evidence = cleanup_fixture(runtime)
    ledger = Ledger(runtime / "members")
    generation = ledger.handoff(
        request.identity, ledger.lookup(request.identity)["generation"], cleanup_receipt=request.receipt
    )
    ledger.io.counts.clear()
    with pytest.raises(Conflict, match="immutable"):
        ledger.handoff(request.identity, generation, cleanup_receipt={**request.receipt, "format": "future-proof"})
    assert ledger.io.counts["writes"] == 0
    assert ledger.lookup(request.identity)["cleanup_receipt"] == request.receipt
