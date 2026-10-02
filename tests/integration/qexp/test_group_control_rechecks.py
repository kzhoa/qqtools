from __future__ import annotations

from pathlib import Path

import pytest

from qqtools.plugins.qexp.runtime import operation_store
from qqtools.plugins.qexp.runtime.group_discovery import locator
from qqtools.plugins.qexp.runtime.group_discovery.control import GroupControlCursor, advance_group_control_rechecks
from qqtools.plugins.qexp.runtime.group_discovery.rechecks import GroupRechecks
from qqtools.plugins.qexp.runtime.locks import group_writer_lock
from qqtools.plugins.qexp.runtime.paths import shared_paths
from qqtools.plugins.qexp.runtime.store import atomic_replace, fenced_mutations
from tests.helpers.qexp_discovery import isolated_group

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _case(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    locator.ensure_group_service_layout(cfg)
    with group_writer_lock(cfg, "experiment"):
        observed = locator.publish_group_locator_locked(cfg, "experiment", "control", "task_change")
    return cfg, observed["generation"]


def _operation(cfg, index, *, group="other", state="converging"):
    path = shared_paths(cfg.shared_root)["group_control_active"] / f"operation-{index:04d}.json"
    atomic_replace(path, {"group_control": {"group_name": group, "state": state}})
    return path


def test_group_control_requires_full_stable_operation_sweep_not_one_page(tmp_path, monkeypatch):
    cfg, generation = _case(tmp_path)
    paths = [_operation(cfg, index) for index in range(65)]
    # Pick the directory's actual final entry; directory iteration isn't lexical.
    ordered = list(operation_store.iter_active_operation_paths(cfg, "group_control", limit=100, cursor={}))
    pending = ordered[-1]
    atomic_replace(pending, {"group_control": {"group_name": "experiment", "state": "converging"}})
    cursor = GroupControlCursor()
    monkeypatch.setattr(
        operation_store, "local_paths", lambda *_args: pytest.fail("shared slice accessed local cursors")
    )
    before = locator.group_locator_path(cfg.shared_root, "experiment", "control").read_bytes()
    assert advance_group_control_rechecks(cfg, "experiment", generation, cursor) == "progress"
    assert locator.group_locator_path(cfg.shared_root, "experiment", "control").read_bytes() == before
    for _ in range(3):
        assert advance_group_control_rechecks(cfg, "experiment", generation, cursor) == "progress"
        assert locator.read_group_locator(cfg.shared_root, "experiment", "control") is not None
    atomic_replace(pending, {"group_control": {"group_name": "experiment", "state": "completed"}})
    for _ in range(4):
        if advance_group_control_rechecks(cfg, "experiment", generation, cursor) == "quiescent":
            break
    else:
        raise AssertionError("completed operations did not permit exact locator retirement")
    assert locator.read_group_locator(cfg.shared_root, "experiment", "control") is None
    assert all(path.exists() for path in paths)


def test_group_control_directory_mutation_invalidates_prefix_absence_proof(tmp_path):
    cfg, generation = _case(tmp_path)
    for index in range(65):
        _operation(cfg, index)
    cursor = GroupControlCursor()
    assert advance_group_control_rechecks(cfg, "experiment", generation, cursor) == "progress"
    before = cursor.operation_witness
    pending = _operation(cfg, 100, group="experiment")
    for _ in range(4):
        assert advance_group_control_rechecks(cfg, "experiment", generation, cursor) == "progress"
    assert cursor.operation_witness != before
    assert locator.read_group_locator(cfg.shared_root, "experiment", "control") is not None
    pending.unlink()
    for _ in range(4):
        if advance_group_control_rechecks(cfg, "experiment", generation, cursor) == "quiescent":
            break
    else:
        raise AssertionError("a new stable whole sweep did not converge")


def test_group_control_restarting_cursor_cannot_turn_partial_scan_into_quiescence(tmp_path):
    cfg, generation = _case(tmp_path)
    for index in range(65):
        _operation(cfg, index)
    for _ in range(3):
        assert advance_group_control_rechecks(cfg, "experiment", generation, GroupControlCursor()) == "progress"
        assert locator.read_group_locator(cfg.shared_root, "experiment", "control") is not None
    cursor = GroupControlCursor()
    assert advance_group_control_rechecks(cfg, "experiment", generation, cursor) == "progress"
    assert advance_group_control_rechecks(cfg, "experiment", generation, cursor) == "quiescent"


def test_group_control_locator_generation_and_mutation_fence_are_exact(tmp_path):
    cfg, generation = _case(tmp_path)
    with group_writer_lock(cfg, "experiment"):
        newer = locator.publish_group_locator_locked(cfg, "experiment", "control", "group_operation")
    assert advance_group_control_rechecks(cfg, "experiment", generation, GroupControlCursor()) == "stale"

    def revoked():
        raise RuntimeError("executor authority revoked")

    with fenced_mutations(cfg.shared_root, revoked), pytest.raises(RuntimeError, match="revoked"):
        advance_group_control_rechecks(cfg, "experiment", newer["generation"], GroupControlCursor())
    assert locator.read_group_locator(cfg.shared_root, "experiment", "control") == newer
    assert advance_group_control_rechecks(cfg, "experiment", newer["generation"], GroupControlCursor()) == "quiescent"


def test_group_control_operation_scan_is_bounded_and_closes_all_descriptors(tmp_path, monkeypatch):
    import os

    from qqtools.plugins.qexp.runtime.group_discovery import control

    cfg, generation = _case(tmp_path)
    for index in range(130):
        _operation(cfg, index)
    reads = []
    real_read = control.read_json_limited

    def read(path: Path, **kwargs):
        reads.append(path)
        return real_read(path, **kwargs)

    monkeypatch.setattr(control, "read_json_limited", read)
    before_fds = len(os.listdir("/proc/self/fd"))
    cursor = GroupControlCursor()
    for _ in range(3):
        before = len(reads)
        result = advance_group_control_rechecks(cfg, "experiment", generation, cursor)
        assert len(reads) - before <= 64
        assert len(os.listdir("/proc/self/fd")) == before_fds
    assert result == "quiescent"


@pytest.mark.parametrize("kind", ["fifo", "directory", "symlink", "malformed"])
def test_group_control_unsafe_operation_cannot_be_skipped_on_retry(tmp_path, kind):
    import os

    cfg, generation = _case(tmp_path)
    directory = shared_paths(cfg.shared_root)["group_control_active"]
    directory.mkdir(parents=True, exist_ok=True)
    unsafe = directory / "unsafe.json"
    if kind == "fifo":
        os.mkfifo(unsafe)
    elif kind == "directory":
        unsafe.mkdir()
    elif kind == "symlink":
        target = _operation(cfg, 0)
        unsafe.symlink_to(target)
    else:
        atomic_replace(unsafe, {"unexpected": {}})
    cursor = GroupControlCursor()
    for _ in range(3):
        with pytest.raises(ValueError):
            advance_group_control_rechecks(cfg, "experiment", generation, cursor)
        assert cursor.operation_cursor == {}
        assert locator.read_group_locator(cfg.shared_root, "experiment", "control") is not None
    if kind == "directory":
        unsafe.rmdir()
    else:
        unsafe.unlink()
    assert advance_group_control_rechecks(cfg, "experiment", generation, cursor) == "quiescent"


def _committed_recheck(journal, task):
    ticket = journal.begin(task, "submission-a", 1, "retry", {"task_revision_before": 3})
    assert ticket is not None
    journal.resolve(ticket)
    return ticket


def test_group_control_consumes_one_event_per_slice_and_restarts_generation(tmp_path):
    cfg, generation = _case(tmp_path)
    journal = GroupRechecks(cfg.shared_root, "experiment")
    with group_writer_lock(cfg, "experiment"):
        initial = journal.snapshot(initialize=True)
        _committed_recheck(journal, "task-a")
        _committed_recheck(journal, "task-b")
    cursor = GroupControlCursor()
    assert advance_group_control_rechecks(cfg, "experiment", generation, cursor) == "progress"
    assert cursor.recheck_sequence == 2
    assert cursor.recheck_generation == initial.generation
    assert locator.read_group_locator(cfg.shared_root, "experiment", "control") is not None
    with group_writer_lock(cfg, "experiment"):
        renewed = journal.invalidate()
        _committed_recheck(journal, "task-c")
    assert advance_group_control_rechecks(cfg, "experiment", generation, cursor) == "progress"
    assert cursor.recheck_generation == renewed.generation
    assert cursor.recheck_sequence == 2
    assert advance_group_control_rechecks(cfg, "experiment", generation, cursor) == "quiescent"


def test_group_control_new_tail_at_retirement_requires_fresh_journal_proof(tmp_path, monkeypatch):
    cfg, generation = _case(tmp_path)
    journal = GroupRechecks(cfg.shared_root, "experiment")
    with group_writer_lock(cfg, "experiment"):
        journal.snapshot(initialize=True)
    original = locator.acknowledge_group_locator_locked

    def append_before_retirement(*args, **kwargs):
        # The real acknowledgement must recheck truth, not reuse the slice's
        # snapshot. This injects a publication while its writer lock is held.
        _committed_recheck(journal, "task-a")
        return original(*args, **kwargs)

    cursor = GroupControlCursor()
    with monkeypatch.context() as patch:
        patch.setattr(locator, "acknowledge_group_locator_locked", append_before_retirement)
        assert advance_group_control_rechecks(cfg, "experiment", generation, cursor) == "stale"
    assert locator.read_group_locator(cfg.shared_root, "experiment", "control") is not None
    assert advance_group_control_rechecks(cfg, "experiment", generation, cursor) == "progress"
    assert advance_group_control_rechecks(cfg, "experiment", generation, cursor) == "quiescent"
