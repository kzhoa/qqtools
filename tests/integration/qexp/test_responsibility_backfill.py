"""Local inventory capture checkpoints survive interruption without replaying work."""

import multiprocessing
import os
from pathlib import Path

import pytest

from qqtools.plugins.qexp.runtime import responsibility_backfill as backfill
from qqtools.plugins.qexp.runtime.locks import exclusive
from qqtools.plugins.qexp.runtime.paths import local_paths
from qqtools.plugins.qexp.runtime.responsibility import ResponsibilityReader, responsibility_root
from qqtools.plugins.qexp.runtime.responsibility_backfill import ResponsibilityBackfill
from qqtools.plugins.qexp.runtime.responsibility_import import RECORD_KEYS
from qqtools.plugins.qexp.runtime.responsibility_store import Conflict, Ledger, Unavailable
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json

pytestmark = pytest.mark.integration


def evidence(root, lane, identity, **fields):
    directory = local_paths(root)[lane]
    path = directory / identity / "decision.json" if lane == "termination_decisions" else directory / f"{identity}.json"
    atomic_replace(path, {RECORD_KEYS[lane]: {"attempt_id": identity, **fields}})
    return path


def finish(capture, limit=64):
    for _ in range(1000):
        progress = capture.take(limit)
        assert progress is not None
        assert progress.entries_visited <= limit and progress.records_processed <= limit
        if progress.is_sweep_complete:
            capture.close()
            return progress
    pytest.fail("finite backfill did not finish")


def test_backfill_captures_all_evidence_lanes_without_shared_task_truth(tmp_path):
    root = tmp_path / "runtime"
    originals = {}
    identities = set()
    for number, lane in enumerate(RECORD_KEYS):
        identity = f"deleted-task-attempt-{number + 1}"
        path = evidence(root, lane, identity)
        originals[path] = path.read_bytes()
        identities.add(identity)
    capture = ResponsibilityBackfill(root)
    progress = finish(capture, 2)
    ledger = Ledger(responsibility_root(root))
    for identity in identities:
        record = ledger.lookup(identity)
        assert record["stage"] == "active"
        assert record["payload"]["task_id"] == "deleted-task"
        assert "legacy_source" not in record
    assert {path: path.read_bytes() for path in originals} == originals
    assert ResponsibilityBackfill(root).take().capture_id == progress.capture_id


def test_backfill_fills_unknown_local_locator_without_replacing_known_fields(tmp_path):
    root = tmp_path / "runtime"
    evidence(root, "processes", "custom")
    evidence(root, "registrations", "custom", task_id="task", attempt_number=3)
    finish(ResponsibilityBackfill(root), 1)
    ledger = Ledger(responsibility_root(root))
    record = ledger.lookup("custom")
    assert record["payload"] == {"task_id": "task", "attempt_number": 3}
    assert ledger.capture_local("custom", {"task_id": None, "attempt_number": None}) == record["generation"]
    with pytest.raises(Conflict, match="different identity"):
        ledger.capture_local("custom", {"task_id": "other", "attempt_number": 3})
    generation = ledger.handoff("custom", record["generation"], cleanup_receipt={"fixture": "retained"})
    assert ledger.capture_local("custom", record["payload"]) == generation
    assert ledger.lookup("custom")["cleanup_receipt"] == {"fixture": "retained"}


def test_restart_replays_pending_batch_before_resuming_scan(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    paths = [evidence(root, "processes", f"task-attempt-{number}") for number in (1, 2)]
    original = Ledger.capture_local
    calls = 0

    def fail_second(ledger, *args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("capture interrupted")
        return original(ledger, *args)

    with monkeypatch.context() as patch:
        patch.setattr(Ledger, "capture_local", fail_second)
        with pytest.raises(OSError, match="capture interrupted"):
            ResponsibilityBackfill(root).take()
    state = read_json(root / "responsibility-backfill.json")
    assert len(state["pending"]) == 2 and state["lane"] == 0
    ledger = Ledger(responsibility_root(root))
    first = Path(state["pending"][0]).stem
    generation = ledger.lookup(first)["generation"]
    restarted = ResponsibilityBackfill(root)
    assert restarted.take(1).pending_records == 1
    assert ledger.lookup(first)["generation"] == generation
    finish(restarted)
    assert all(path.exists() and ledger.lookup(path.stem)["identity"] == path.stem for path in paths)


def test_restart_does_not_rescan_completed_lanes(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    path = evidence(root, "processes", "task-attempt-1")
    evidence(root, "registrations", "task-attempt-2")
    capture = ResponsibilityBackfill(root)
    assert capture.take().completed_lanes == 1
    capture.close()
    original = os.scandir

    def forbidden(directory):
        if directory == path.parent:
            raise AssertionError("completed lane was rescanned")
        return original(directory)

    with monkeypatch.context() as patch:
        patch.setattr(os, "scandir", forbidden)
        finish(ResponsibilityBackfill(root))
    assert Ledger(responsibility_root(root)).lookup("task-attempt-2")


def test_every_inventory_entry_consumes_budget(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    path = evidence(root, "processes", "task-attempt-1")
    for number in range(25):
        (path.parent / f"temporary-{number}").touch()
    original = os.scandir
    visited = []

    class CountedDirectory:
        def __init__(self, directory):
            self.entries = original(directory)
            self.is_counted = directory == path.parent

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.entries.close()

        def __iter__(self):
            return self

        def __next__(self):
            entry = next(self.entries)
            if self.is_counted:
                visited.append(entry.name)
            return entry

    capture = ResponsibilityBackfill(root)
    with monkeypatch.context() as patch:
        patch.setattr(os, "scandir", CountedDirectory)
        for _ in range(10):
            before = len(visited)
            progress = capture.take(3)
            assert len(visited) - before <= 3
            if progress.completed_lanes:
                break
    capture.close()
    assert len(visited) == 26 and len(set(visited)) == 26


@pytest.mark.parametrize("mode", ["malformed_record", "identity_conflict", "nested_record"])
def test_invalid_evidence_does_not_advance_lane(tmp_path, mode):
    root = tmp_path / "runtime"
    path = evidence(root, "processes", "task-attempt-1")
    if mode == "malformed_record":
        path.write_text("{}")
    elif mode == "identity_conflict":
        atomic_replace(path, {"process": {"attempt_id": "another-attempt-1"}})
    else:
        nested = path.parent / "unexpected"
        nested.mkdir()
        path = path.rename(nested / path.name)
    with pytest.raises(Unavailable):
        ResponsibilityBackfill(root).take()
    assert read_json(root / "responsibility-backfill.json")["lane"] == 0
    assert path.exists()


def test_backfill_fails_closed_on_replaced_store_and_corrupt_progress(tmp_path):
    root = tmp_path / "runtime"
    evidence(root, "processes", "task-attempt-1")
    capture = ResponsibilityBackfill(root)
    capture.take(1)
    capture.close()
    original = responsibility_root(root)
    original.rename(root / "old-ledger")
    Ledger.open_or_create(original)
    before = capture.path.read_bytes()
    with pytest.raises(Unavailable, match="replaced"):
        ResponsibilityBackfill(root).take()
    assert capture.path.read_bytes() == before
    capture.path.write_text("broken")
    with pytest.raises(ValueError):
        ResponsibilityBackfill(root).take()
    assert capture.path.read_text() == "broken"


def test_backfill_defers_to_existing_builder_lock(tmp_path):
    root = tmp_path / "runtime"
    evidence(root, "processes", "task-attempt-1")
    capture = ResponsibilityBackfill(root)
    with exclusive(root / "locks" / "responsibility-backfill.lock"):
        assert capture.take() is None
    finish(capture)


@pytest.mark.parametrize("is_parent", [False, True])
def test_pending_replay_rejects_replaced_symlink_evidence(tmp_path, monkeypatch, is_parent):
    root = tmp_path / "runtime"
    source = evidence(root, "termination_decisions", "task-attempt-1")
    capture = ResponsibilityBackfill(root)

    def interrupted(*_args):
        raise OSError("before pending capture")

    with monkeypatch.context() as patch:
        patch.setattr(backfill, "capture_local_record", interrupted)
        with pytest.raises(OSError, match="before pending capture"):
            finish(capture)
    checkpoint = capture.path.read_bytes()
    target = source.parent if is_parent else source
    moved = target.rename(tmp_path / "outside")
    target.symlink_to(moved, target_is_directory=is_parent)
    with pytest.raises(Unavailable, match="regular|directory"):
        ResponsibilityBackfill(root).take()
    assert capture.path.read_bytes() == checkpoint
    assert Ledger(responsibility_root(root)).find("task-attempt-1") is None


def test_background_reader_backfills_old_evidence_and_resumes_completed_checkpoint(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    evidence(root, "observations", "historical-attempt-7")

    def read_until_discovered(reader):
        found = False
        try:
            for _ in range(100):
                entry = reader.take()
                found |= entry is not None and entry["identity"] == "historical-attempt-7"
                if found and reader._is_backfill_complete:
                    return
                if reader._done is not None:
                    assert reader._done.wait(5)
            pytest.fail("background capture did not discover the old Attempt")
        finally:
            reader.close()
            if reader._done is not None:
                assert reader._done.wait(5)

    read_until_discovered(ResponsibilityReader(root))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("completed inventory was scanned or rewritten on restart")

    with monkeypatch.context() as patch:
        patch.setattr(backfill, "iter_evidence_entries", forbidden)
        patch.setattr(backfill, "atomic_replace", forbidden)
        read_until_discovered(ResponsibilityReader(root))


def test_failed_backfill_does_not_hide_already_indexed_responsibility(tmp_path):
    root = tmp_path / "runtime"
    path = evidence(root, "processes", "bad-attempt-1")
    path.write_text("{}")
    ledger = Ledger.open_or_create(responsibility_root(root))
    ledger.publish("retained-attempt-1", {"task_id": "retained", "attempt_number": 1})
    reader = ResponsibilityReader(root)
    try:
        for _ in range(50):
            entry = reader.take()
            if entry is not None and entry["identity"] == "retained-attempt-1":
                assert reader.failures > 0
                assert not reader._is_backfill_complete
                break
            if reader._done is not None:
                assert reader._done.wait(5)
        else:
            pytest.fail("failed capture suppressed unrelated recovery")
    finally:
        reader.close()
        if reader._done is not None:
            assert reader._done.wait(5)


def crash_capture(root, boundary):
    original_save = backfill.atomic_replace
    original_capture = Ledger.capture_local

    def save(path, value):
        original_save(path, value)
        if path.name == "responsibility-backfill.json":
            if boundary == "pending" and value["pending"]:
                os._exit(86)
            if boundary == "checkpoint" and value["lane"] > 0:
                os._exit(86)

    def publish(ledger, *args):
        result = original_capture(ledger, *args)
        if boundary == "membership":
            os._exit(86)
        return result

    backfill.atomic_replace = save
    Ledger.capture_local = publish
    finish(ResponsibilityBackfill(root))


@pytest.mark.parametrize("boundary", ["pending", "membership", "checkpoint"])
def test_real_crash_replays_capture_without_deleting_evidence(tmp_path, boundary):
    root = tmp_path / "runtime"
    paths = [evidence(root, "processes", f"task-attempt-{number}") for number in (1, 2)]
    before = {path: path.read_bytes() for path in paths}
    process = multiprocessing.get_context("fork").Process(target=crash_capture, args=(root, boundary))
    process.start()
    try:
        process.join(10)
        assert process.exitcode == 86
    finally:
        if process.is_alive():
            process.kill()
        process.join(5)
        process.close()
    finish(ResponsibilityBackfill(root), 1)
    ledger = Ledger(responsibility_root(root))
    assert all(ledger.lookup(path.stem)["stage"] == "active" for path in paths)
    assert {path: path.read_bytes() for path in paths} == before


def process_census(tmp_path, monkeypatch, *, should_finish=True):
    from qqtools.plugins.qexp.config_types import RootConfig
    from qqtools.plugins.qexp.runtime import responsibility_process_capture as processes

    root, source, proc = (tmp_path / name for name in ("target/runtime", "legacy/runtime", "proc"))
    root.mkdir(parents=True)
    source.mkdir(parents=True)
    (proc / "sys/kernel/random").mkdir(parents=True)
    (proc / "sys/kernel/random/boot_id").write_text("10000000-0000-0000-0000-000000000001")
    (proc / "self/ns").mkdir(parents=True)
    (proc / "self/ns/pid").touch()
    monkeypatch.setattr(processes, "PROC_ROOT", proc)
    cfg = RootConfig(tmp_path / ".qexp", tmp_path, "fixture", root)
    ledger = Ledger.open_or_create(responsibility_root(root))
    census = processes.RunnerProcessCapture(cfg, ledger, legacy_source=source)
    if should_finish:
        assert census.take(64).is_sweep_complete
    return census, source


@pytest.mark.parametrize("limit", [1, 7, 63, 64])
def test_post_process_backfill_is_fresh_and_captures_both_roots(tmp_path, monkeypatch, limit):
    from qqtools.plugins.qexp.runtime.responsibility_capture import CAPTURE_FILE

    census, source = process_census(tmp_path, monkeypatch)
    root = census.cfg.runtime_root
    old = finish(ResponsibilityBackfill(root))
    originals = {}
    for number, lane in enumerate(RECORD_KEYS, 1):
        for location, prefix in ((root, "local"), (source, "legacy")):
            identity = f"{prefix}-attempt-{number}"
            path = evidence(location, lane, identity)
            originals[path] = path.read_bytes()
    assert ResponsibilityBackfill(root).take().capture_id == old.capture_id
    assert not census.checkpoint.ledger.has_members()
    progress = finish(ResponsibilityBackfill(root, process_capture=census), limit)
    assert progress.total_lanes == 2 * len(RECORD_KEYS)
    assert progress.capture_id != old.capture_id
    for number in range(1, len(RECORD_KEYS) + 1):
        for prefix in ("local", "legacy"):
            entry = census.checkpoint.ledger.lookup(f"{prefix}-attempt-{number}")
            assert entry["payload"] == {"task_id": prefix, "attempt_number": number}
            assert entry.get("legacy_source") == (str(source) if prefix == "legacy" else None)
    assert {path: path.read_bytes() for path in originals} == originals
    assert not responsibility_root(source).exists() and not census.cfg.shared_root.exists()
    assert read_json(root / CAPTURE_FILE)["phase"] == read_json(source / CAPTURE_FILE)["phase"] == "pending"
    with monkeypatch.context() as patch:
        patch.setattr(backfill, "iter_evidence_entries", lambda *_a, **_k: pytest.fail("completed capture rescanned"))
        patch.setattr(backfill, "atomic_replace", lambda *_a, **_k: pytest.fail("completed capture rewritten"))
        repeated = ResponsibilityBackfill(root, process_capture=census).take(limit)
    assert repeated.is_sweep_complete and repeated.entries_visited == repeated.records_processed == 0
    assert repeated.capture_id == progress.capture_id


def test_evidence_sweep_requires_complete_process_context_and_retention(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runtime.responsibility_capture import CAPTURE_FILE

    census, source = process_census(tmp_path, monkeypatch, should_finish=False)
    root = census.cfg.runtime_root
    sweep = ResponsibilityBackfill(root, process_capture=census)
    with pytest.raises(Unavailable, match="completed process sweep"):
        sweep.take()
    assert not sweep.path.exists()
    assert census.take().is_sweep_complete
    sweep.take()
    before = sweep.path.read_bytes()
    (source / CAPTURE_FILE).unlink()
    with pytest.raises(Unavailable, match="source hold disappeared"):
        sweep.take()
    assert sweep.path.read_bytes() == before


def test_post_process_sweep_without_legacy_source(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runtime.responsibility_process_capture import RunnerProcessCapture

    prepared, _ = process_census(tmp_path, monkeypatch, should_finish=False)
    census = RunnerProcessCapture(prepared.cfg, prepared.checkpoint.ledger)
    assert census.take().is_sweep_complete
    root = census.cfg.runtime_root
    evidence(root, "observations", "task-attempt-1")
    with pytest.raises(ValueError, match="another runtime"):
        ResponsibilityBackfill(tmp_path / "unrelated", process_capture=census)
    progress = finish(ResponsibilityBackfill(root, process_capture=census))
    assert progress.total_lanes == len(RECORD_KEYS)
    assert "legacy_source" not in census.checkpoint.ledger.lookup("task-attempt-1")


def test_final_evidence_checkpoint_cannot_adopt_another_capture(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runtime.responsibility_capture import CAPTURE_FILE

    census, source = process_census(tmp_path, monkeypatch)
    root = census.cfg.runtime_root
    finish(ResponsibilityBackfill(root, process_capture=census))
    for location in (root, source):
        path = location / CAPTURE_FILE
        state = read_json(path)
        state["capture_id"] = "f" * 32
        atomic_replace(path, state)
    with pytest.raises(Unavailable, match="invalid local backfill checkpoint"):
        ResponsibilityBackfill(root, process_capture=census).take()


@pytest.mark.parametrize("has_revision", [False, True])
def test_completed_evidence_sweep_is_not_reused_after_reboot(tmp_path, monkeypatch, has_revision):
    from qqtools.plugins.qexp.runtime import responsibility_process_capture as processes

    census, source = process_census(tmp_path, monkeypatch)
    root = census.cfg.runtime_root
    sweep = ResponsibilityBackfill(root, process_capture=census)
    first = finish(sweep)
    if not has_revision:
        state = read_json(sweep.path)
        state.pop("writer_sweep_revision", None)
        atomic_replace(sweep.path, state)
    (processes.PROC_ROOT / "sys/kernel/random/boot_id").write_text("20000000-0000-0000-0000-000000000002")
    path = evidence(source, "observations", "late-attempt-1")
    assert census.restart_after_reboot()
    with pytest.raises(Unavailable, match="completed process sweep"):
        sweep.take()
    assert census.take().is_sweep_complete
    second = finish(sweep, 1)
    assert second.capture_id != first.capture_id
    assert census.checkpoint.ledger.lookup(path.stem)["legacy_source"] == str(source)
    assert path.exists()
    with monkeypatch.context() as patch:
        patch.setattr(backfill, "iter_evidence_entries", lambda *_args: pytest.fail("completed sweep rescanned"))
        assert sweep.take().is_sweep_complete


@pytest.mark.parametrize("is_legacy", [False, True])
def test_retained_pending_evidence_cannot_disappear_silently(tmp_path, monkeypatch, is_legacy):
    census, source = process_census(tmp_path, monkeypatch)
    root = census.cfg.runtime_root
    path = evidence(source if is_legacy else root, "processes", "task-attempt-1")
    original = path.read_bytes()
    sweep = ResponsibilityBackfill(root, process_capture=census)

    def interrupt(*_args, **_kwargs):
        raise OSError("before capture")

    with monkeypatch.context() as patch:
        patch.setattr(backfill, "capture_local_record", interrupt)
        with pytest.raises(OSError, match="before capture"):
            finish(sweep)
    before = sweep.path.read_bytes()
    path.unlink()
    with pytest.raises(Unavailable, match="retained backfill evidence disappeared"):
        ResponsibilityBackfill(root, process_capture=census).take()
    assert sweep.path.read_bytes() == before
    assert census.checkpoint.ledger.find("task-attempt-1") is None
    path.write_bytes(original)
    finish(ResponsibilityBackfill(root, process_capture=census))
    assert census.checkpoint.ledger.lookup("task-attempt-1")


@pytest.mark.parametrize("boundary", ["membership", "restart_checkpoint"])
def test_reboot_evidence_replay_crash_keeps_old_batch_and_restarts_all_lanes(tmp_path, monkeypatch, boundary):
    from qqtools.plugins.qexp.runtime import responsibility_process_capture as processes

    census, source = process_census(tmp_path, monkeypatch)
    root = census.cfg.runtime_root
    old = evidence(source, "observations", "old-attempt-1")
    sweep = ResponsibilityBackfill(root, process_capture=census)
    original_capture = backfill.capture_local_record

    def fail(*_args, **_kwargs):
        raise OSError("stop with retained batch")

    with monkeypatch.context() as patch:
        patch.setattr(backfill, "capture_local_record", fail)
        with pytest.raises(OSError, match="retained batch"):
            finish(sweep)
    before = read_json(sweep.path)
    assert before["pending"] == [old.name]
    new = evidence(root, "processes", "new-attempt-1")
    (processes.PROC_ROOT / "sys/kernel/random/boot_id").write_text("20000000-0000-0000-0000-000000000002")
    census.restart_after_reboot()
    assert census.take().is_sweep_complete

    def crash():
        original_write = backfill.atomic_replace

        def capture_record(*args, **kwargs):
            result = original_capture(*args, **kwargs)
            if boundary == "membership":
                os._exit(86)
            return result

        def write(path, value):
            original_write(path, value)
            if boundary == "restart_checkpoint" and value.get("capture_id") != before["capture_id"]:
                os._exit(86)

        backfill.capture_local_record = capture_record
        backfill.atomic_replace = write
        ResponsibilityBackfill(root, process_capture=census).take(1)

    child = multiprocessing.get_context("fork").Process(target=crash)
    child.start()
    try:
        child.join(10)
        assert child.exitcode == 86
    finally:
        if child.is_alive():
            child.kill()
        child.join(5)
        child.close()
    result = finish(ResponsibilityBackfill(root, process_capture=census), 1)
    assert result.capture_id != before["capture_id"]
    assert census.checkpoint.ledger.lookup(old.stem)["legacy_source"] == str(source)
    assert census.checkpoint.ledger.lookup(new.stem)["stage"] == "active"
    assert old.exists() and new.exists()


@pytest.mark.parametrize("revision", [True, 0, -1, "1", 999999])
def test_invalid_process_revision_cannot_reset_retained_evidence(tmp_path, monkeypatch, revision):
    census, _ = process_census(tmp_path, monkeypatch)
    sweep = ResponsibilityBackfill(census.cfg.runtime_root, process_capture=census)
    finish(sweep)
    state = read_json(sweep.path)
    state["writer_sweep_revision"] = revision
    atomic_replace(sweep.path, state)
    before = sweep.path.read_bytes()
    with pytest.raises(Unavailable, match="process sweep revision"):
        sweep.take()
    assert sweep.path.read_bytes() == before


def test_unversioned_completed_evidence_gets_one_fresh_sweep(tmp_path, monkeypatch):
    census, source = process_census(tmp_path, monkeypatch)
    sweep = ResponsibilityBackfill(census.cfg.runtime_root, process_capture=census)
    first = finish(sweep)
    state = read_json(sweep.path)
    state.pop("writer_sweep_revision")
    atomic_replace(sweep.path, state)
    path = evidence(source, "registrations", "late-attempt-1")
    second = finish(sweep)
    assert second.capture_id != first.capture_id
    assert census.checkpoint.ledger.lookup(path.stem)["stage"] == "active"
    assert sweep.take().capture_id == second.capture_id


def test_reboot_cannot_discard_a_missing_retained_evidence_batch(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runtime import responsibility_process_capture as processes

    census, source = process_census(tmp_path, monkeypatch)
    sweep = ResponsibilityBackfill(census.cfg.runtime_root, process_capture=census)
    path = evidence(source, "registrations", "pending-attempt-1")
    original = path.read_bytes()

    def fail(*_args, **_kwargs):
        raise OSError("stop before membership")

    with monkeypatch.context() as patch:
        patch.setattr(backfill, "capture_local_record", fail)
        with pytest.raises(OSError, match="before membership"):
            finish(sweep)
    before = sweep.path.read_bytes()
    (processes.PROC_ROOT / "sys/kernel/random/boot_id").write_text("20000000-0000-0000-0000-000000000002")
    census.restart_after_reboot()
    assert census.take().is_sweep_complete
    path.unlink()
    with pytest.raises(Unavailable, match="retained backfill evidence disappeared"):
        sweep.take()
    assert sweep.path.read_bytes() == before
    path.write_bytes(original)
    finish(sweep)
    assert census.checkpoint.ledger.lookup(path.stem)["legacy_source"] == str(source)


@pytest.mark.parametrize("is_post_process", [False, True])
def test_alternating_backfills_finish_without_restarting_each_other(tmp_path, monkeypatch, is_post_process):
    census, _ = process_census(tmp_path, monkeypatch)
    root = census.cfg.runtime_root
    for number in range(70):
        evidence(root, "processes", f"task-{number}-attempt-1")
    captures = [ResponsibilityBackfill(root, process_capture=census if is_post_process else None) for _ in range(2)]
    try:
        for _ in range(30):
            pages = [capture.take(7) for capture in captures]
            if all(page.is_sweep_complete for page in pages):
                break
        else:
            pytest.fail("concurrent checkpoint revisions starved evidence traversal")
        assert census.checkpoint.ledger.lookup("task-69-attempt-1")
    finally:
        for capture in captures:
            capture.close()


@pytest.mark.parametrize("boundary", ["pending", "membership", "checkpoint"])
def test_crashed_source_sweep_replays_into_target_without_source_deletion(tmp_path, monkeypatch, boundary):
    census, source = process_census(tmp_path, monkeypatch)
    root = census.cfg.runtime_root
    paths = [evidence(source, "processes", f"legacy-attempt-{number}") for number in (1, 2)]
    before = {path: path.read_bytes() for path in paths}

    def crash():
        original_save = backfill.atomic_replace
        original_capture = Ledger.capture_source

        def save(path, value):
            original_save(path, value)
            if path.name == "responsibility-capture-backfill.json":
                if boundary == "pending" and value["lane"] >= len(RECORD_KEYS) and value["pending"]:
                    os._exit(86)
                if boundary == "checkpoint" and value["lane"] > len(RECORD_KEYS):
                    os._exit(86)

        def publish(ledger, *args):
            result = original_capture(ledger, *args)
            if boundary == "membership":
                os._exit(86)
            return result

        backfill.atomic_replace = save
        Ledger.capture_source = publish
        finish(ResponsibilityBackfill(root, process_capture=census))

    child = multiprocessing.get_context("fork").Process(target=crash)
    child.start()
    try:
        child.join(10)
        assert not child.is_alive() and child.exitcode == 86
    finally:
        if child.is_alive():
            child.kill()
        child.join(5)
        child.close()
    sweep = ResponsibilityBackfill(root, process_capture=census)
    state = read_json(sweep.path)
    if state["pending"]:
        with monkeypatch.context() as patch:
            patch.setattr(backfill, "iter_evidence_entries", lambda *_a, **_k: pytest.fail("enumerated before replay"))
            assert sweep.take(1).records_processed == 1
    finish(sweep, 1)
    for path in paths:
        entry = census.checkpoint.ledger.lookup(path.stem)
        assert entry["legacy_source"] == str(source) and entry["stage"] == "active"
    assert {path: path.read_bytes() for path in paths} == before
    assert not responsibility_root(source).exists()
