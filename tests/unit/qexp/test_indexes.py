from __future__ import annotations

import pytest

from qqtools.plugins.qexp.indexes import (
    load_index,
    rebuild_all_indexes,
    update_index_on_phase_change,
    update_index_on_submit,
)
from qqtools.plugins.qexp.layout import init_shared_root
from qqtools.plugins.qexp.models import (
    Meta,
    PHASE_QUEUED,
    PHASE_RUNNING,
    PHASE_SUCCEEDED,
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


def _make_task(
    task_id: str,
    phase: str = PHASE_QUEUED,
    batch_id: str | None = None,
    group: str | None = None,
) -> Task:
    now = utc_now_iso()
    return Task(
        meta=Meta.new("dev1"),
        task_id=task_id,
        name=None,
        group=group,
        batch_id=batch_id,
        machine_name="dev1",
        attempt=1,
        spec=TaskSpec(command=["echo"], requested_gpus=1),
        status=TaskStatus(phase=phase),
        runtime=TaskRuntime(),
        timestamps=TaskTimestamps(created_at=now, queued_at=now),
        result=TaskResult(),
        lineage=TaskLineage(),
    )


class TestIndexOnSubmit:
    def test_state_index(self, cfg):
        t = _make_task("t1")
        update_index_on_submit(cfg, t)
        assert "t1" in load_index(cfg, "state", PHASE_QUEUED)

    def test_machine_index(self, cfg):
        t = _make_task("t1")
        update_index_on_submit(cfg, t)
        assert "t1" in load_index(cfg, "machine", "dev1")

    def test_batch_index(self, cfg):
        t = _make_task("t1", batch_id="b1")
        update_index_on_submit(cfg, t)
        assert "t1" in load_index(cfg, "batch", "b1")

    def test_group_index(self, cfg):
        t = _make_task("t1", group="contract_n_4and6")
        update_index_on_submit(cfg, t)
        assert "t1" in load_index(cfg, "group", "contract_n_4and6")

    def test_no_batch_index_when_none(self, cfg):
        t = _make_task("t1")
        update_index_on_submit(cfg, t)
        assert load_index(cfg, "batch", "none") == []

    def test_idempotent(self, cfg):
        t = _make_task("t1")
        update_index_on_submit(cfg, t)
        update_index_on_submit(cfg, t)
        assert load_index(cfg, "state", PHASE_QUEUED).count("t1") == 1


class TestPhaseChange:
    def test_moves_between_state_indexes(self, cfg):
        t = _make_task("t1")
        update_index_on_submit(cfg, t)
        update_index_on_phase_change(cfg, "t1", PHASE_QUEUED, PHASE_RUNNING)
        assert "t1" not in load_index(cfg, "state", PHASE_QUEUED)
        assert "t1" in load_index(cfg, "state", PHASE_RUNNING)


class TestRebuild:
    def test_rebuild_matches_incremental(self, cfg):
        for i in range(5):
            t = _make_task(
                f"t{i}",
                batch_id="b1" if i < 3 else None,
                group="contract_n_4and6" if i < 2 else None,
            )
            save_task(cfg, t)
            update_index_on_submit(cfg, t)

        # Corrupt: manually clear an index
        from qqtools.plugins.qexp.layout import index_by_state_dir
        for f in index_by_state_dir(cfg).glob("*.json"):
            f.unlink()

        stats = rebuild_all_indexes(cfg)
        assert stats["total_tasks"] == 5
        assert load_index(cfg, "state", PHASE_QUEUED) == [f"t{i}" for i in range(5)]
        assert len(load_index(cfg, "batch", "b1")) == 3
        assert len(load_index(cfg, "group", "contract_n_4and6")) == 2

    def test_rebuild_empty(self, cfg):
        stats = rebuild_all_indexes(cfg)
        assert stats["total_tasks"] == 0

    def test_invalid_index_type(self, cfg):
        with pytest.raises(ValueError):
            load_index(cfg, "invalid", "key")
