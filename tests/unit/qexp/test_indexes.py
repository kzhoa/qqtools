from __future__ import annotations

import pytest

from qqtools.plugins.qexp.indexes import (
    collect_index_drift_report,
    load_index,
    rebuild_all_indexes,
    remove_index_on_delete,
    sync_task_state_index,
    update_index_on_phase_change,
    update_index_on_submit,
)
from qqtools.plugins.qexp.layout import init_shared_root
from qqtools.plugins.qexp.models import (
    PHASE_QUEUED,
    PHASE_RUNNING,
    PHASE_SUCCEEDED,
    Meta,
    Task,
    TaskLineage,
    TaskResult,
    TaskRuntime,
    TaskSpec,
    TaskStatus,
    TaskTimestamps,
    utc_now_iso,
)
from qqtools.plugins.qexp.storage import save_task


@pytest.fixture()
def cfg(tmp_path):
    return init_shared_root(tmp_path / ".qexp", "dev1", runtime_root=tmp_path / "runtime")


def _make_task(task_id: str, phase: str = PHASE_QUEUED) -> Task:
    now = utc_now_iso()
    return Task(
        meta=Meta.new("dev1"),
        task_id=task_id,
        name=None,
        group=None,
        batch_id=None,
        machine_name="dev1",
        attempt=1,
        spec=TaskSpec(command=["echo"], requested_gpus=1),
        status=TaskStatus(phase=phase),
        runtime=TaskRuntime(),
        timestamps=TaskTimestamps(created_at=now, queued_at=now),
        result=TaskResult(),
        lineage=TaskLineage(),
    )


class TestStateIndexes:
    def test_submit_indexes_state_only(self, cfg):
        task = _make_task("t1")
        update_index_on_submit(cfg, task)
        assert "t1" in load_index(cfg, "state", PHASE_QUEUED)

    def test_idempotent_submit(self, cfg):
        task = _make_task("t1")
        update_index_on_submit(cfg, task)
        update_index_on_submit(cfg, task)
        assert load_index(cfg, "state", PHASE_QUEUED).count("t1") == 1

    def test_phase_change_moves_between_state_indexes(self, cfg):
        task = _make_task("t1")
        update_index_on_submit(cfg, task)
        update_index_on_phase_change(cfg, "t1", PHASE_QUEUED, PHASE_RUNNING)
        assert "t1" not in load_index(cfg, "state", PHASE_QUEUED)
        assert "t1" in load_index(cfg, "state", PHASE_RUNNING)

    def test_sync_task_state_index_removes_stale_residuals(self, cfg):
        task = _make_task("t1")
        update_index_on_submit(cfg, task)
        update_index_on_phase_change(cfg, "t1", PHASE_QUEUED, PHASE_RUNNING)

        from qqtools.plugins.qexp.layout import index_by_state_dir
        from qqtools.plugins.qexp.storage import write_atomic_json

        write_atomic_json(index_by_state_dir(cfg) / f"{PHASE_QUEUED}.json", {"task_ids": ["t1"]})
        write_atomic_json(index_by_state_dir(cfg) / f"{PHASE_SUCCEEDED}.json", {"task_ids": ["t1"]})

        sync_task_state_index(cfg, "t1", PHASE_RUNNING)

        assert "t1" not in load_index(cfg, "state", PHASE_QUEUED)
        assert "t1" in load_index(cfg, "state", PHASE_RUNNING)
        assert "t1" not in load_index(cfg, "state", PHASE_SUCCEEDED)

    def test_remove_index_on_delete_only_removes_state_index(self, cfg):
        task = _make_task("t-delete")
        save_task(cfg, task)
        update_index_on_submit(cfg, task)
        remove_index_on_delete(cfg, task)
        assert "t-delete" not in load_index(cfg, "state", PHASE_QUEUED)


class TestRebuild:
    def test_rebuild_matches_incremental_and_prunes_legacy_dirs(self, cfg):
        from qqtools.plugins.qexp.layout import global_indexes_dir

        for i in range(5):
            task = _make_task(f"t{i}")
            save_task(cfg, task)
            update_index_on_submit(cfg, task)

        legacy_dir = global_indexes_dir(cfg) / "tasks_by_machine"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        (legacy_dir / "dev1.json").write_text('{"task_ids":["ghost"]}', encoding="utf-8")

        from qqtools.plugins.qexp.layout import index_by_state_dir
        for path in index_by_state_dir(cfg).glob("*.json"):
            path.unlink()

        stats = rebuild_all_indexes(cfg)
        assert stats["total_tasks"] == 5
        assert load_index(cfg, "state", PHASE_QUEUED) == [f"t{i}" for i in range(5)]
        assert stats["removed_legacy_index_dirs"] == ["tasks_by_machine"]
        assert not legacy_dir.exists()

    def test_rebuild_empty(self, cfg):
        stats = rebuild_all_indexes(cfg)
        assert stats["total_tasks"] == 0

    def test_invalid_index_type(self, cfg):
        with pytest.raises(ValueError):
            load_index(cfg, "machine", "dev1")


class TestIndexDriftReport:
    def test_reports_missing_and_unexpected_state_entries(self, cfg):
        task = _make_task("t1")
        save_task(cfg, task)
        update_index_on_submit(cfg, task)

        from qqtools.plugins.qexp.layout import index_by_state_dir
        from qqtools.plugins.qexp.storage import write_atomic_json

        write_atomic_json(index_by_state_dir(cfg) / f"{PHASE_QUEUED}.json", {"task_ids": []})
        write_atomic_json(index_by_state_dir(cfg) / "ghost.json", {"task_ids": ["ghost-task"]})

        report = collect_index_drift_report(cfg)
        assert report["ok"] is False
        assert report["families"]["state"]["missing_count"] == 1
        assert report["families"]["state"]["unexpected_count"] == 1
