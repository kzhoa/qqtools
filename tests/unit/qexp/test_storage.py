from __future__ import annotations

import json

import pytest

from qqtools.plugins.qexp.layout import RootConfig, init_shared_root
from qqtools.plugins.qexp.models import (
    Batch,
    BATCH_COMMIT_COMMITTED,
    BatchPolicy,
    BatchSummary,
    GpuInventory,
    Machine,
    Meta,
    PHASE_QUEUED,
    PHASE_RUNNING,
    Task,
    TaskLineage,
    TaskResult,
    TaskRuntime,
    TaskSpec,
    TaskStatus,
    TaskTimestamps,
    utc_now_iso,
    AGENT_MODE_ON_DEMAND,
    AGENT_STATE_STOPPED,
)
from qqtools.plugins.qexp.storage import (
    CASConflict,
    cas_update_batch,
    cas_update_task,
    iter_all_batches,
    iter_all_tasks,
    iter_machines,
    load_batch,
    load_machine,
    load_task,
    release_claim,
    save_batch,
    save_claim,
    save_machine,
    save_task,
)


@pytest.fixture()
def cfg(tmp_path):
    return init_shared_root(tmp_path / ".qexp", "dev1", runtime_root=tmp_path / "runtime")


def _make_task(task_id: str = "t-001", machine: str = "dev1") -> Task:
    now = utc_now_iso()
    return Task(
        meta=Meta.new(machine),
        task_id=task_id,
        name=None,
        group=None,
        batch_id=None,
        machine_name=machine,
        attempt=1,
        spec=TaskSpec(command=["python", "train.py"], requested_gpus=1),
        status=TaskStatus(phase=PHASE_QUEUED),
        runtime=TaskRuntime(),
        timestamps=TaskTimestamps(created_at=now, queued_at=now),
        result=TaskResult(),
        lineage=TaskLineage(),
    )


def _make_batch(batch_id: str = "b-001", machine: str = "dev1") -> Batch:
    return Batch(
        meta=Meta.new(machine),
        batch_id=batch_id,
        name=None,
        group=None,
        source_manifest=None,
        machine_name=machine,
        commit_state=BATCH_COMMIT_COMMITTED,
        expected_task_count=0,
        task_ids=[],
        summary=BatchSummary(),
        policy=BatchPolicy(),
    )


# ---------------------------------------------------------------------------
# Task persistence
# ---------------------------------------------------------------------------


class TestTaskPersistence:
    def test_save_and_load(self, cfg):
        t = _make_task()
        save_task(cfg, t)
        loaded = load_task(cfg, "t-001")
        assert loaded.task_id == "t-001"
        assert loaded.meta.revision == 1
        assert loaded.spec.command == ["python", "train.py"]

    def test_load_missing(self, cfg):
        with pytest.raises(FileNotFoundError):
            load_task(cfg, "nonexistent")

    def test_load_validates_id(self, cfg):
        with pytest.raises(ValueError):
            load_task(cfg, "../evil")

    def test_atomic_write_no_partial(self, cfg):
        t = _make_task()
        save_task(cfg, t)
        from qqtools.plugins.qexp.layout import task_path
        p = task_path(cfg, "t-001")
        assert p.is_file()
        tmp = p.with_suffix(".json.tmp")
        assert not tmp.exists()


# ---------------------------------------------------------------------------
# CAS
# ---------------------------------------------------------------------------


class TestCAS:
    def test_cas_update_increments_revision(self, cfg):
        t = _make_task()
        save_task(cfg, t)
        t.status.phase = PHASE_RUNNING
        updated = cas_update_task(cfg, t, expected_revision=1)
        assert updated.meta.revision == 2

    def test_cas_conflict(self, cfg):
        t = _make_task()
        save_task(cfg, t)
        t.status.phase = PHASE_RUNNING
        cas_update_task(cfg, t, expected_revision=1)
        t2 = _make_task()
        with pytest.raises(CASConflict) as exc_info:
            cas_update_task(cfg, t2, expected_revision=1)
        assert exc_info.value.expected == 1
        assert exc_info.value.actual == 2

    def test_cas_batch(self, cfg):
        b = _make_batch()
        save_batch(cfg, b)
        b.task_ids = ["t1"]
        updated = cas_update_batch(cfg, b, expected_revision=1)
        assert updated.meta.revision == 2
        assert updated.task_ids == ["t1"]


# ---------------------------------------------------------------------------
# Batch persistence
# ---------------------------------------------------------------------------


class TestBatchPersistence:
    def test_save_and_load(self, cfg):
        b = _make_batch()
        save_batch(cfg, b)
        loaded = load_batch(cfg, "b-001")
        assert loaded.batch_id == "b-001"

    def test_load_missing(self, cfg):
        with pytest.raises(FileNotFoundError):
            load_batch(cfg, "nonexistent")


# ---------------------------------------------------------------------------
# Machine persistence
# ---------------------------------------------------------------------------


class TestMachinePersistence:
    def test_load_own_machine(self, cfg):
        m = load_machine(cfg)
        assert m.machine_name == "dev1"

    def test_load_other_machine(self, cfg):
        init_shared_root(cfg.shared_root, "gpu2")
        m = load_machine(cfg, "gpu2")
        assert m.machine_name == "gpu2"

    def test_load_missing_machine(self, cfg):
        with pytest.raises(FileNotFoundError):
            load_machine(cfg, "nosuch")


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------


class TestClaims:
    def test_save_and_release(self, cfg):
        save_claim(cfg, "t-001", utc_now_iso(), 1)
        from qqtools.plugins.qexp.layout import machine_claims_active_dir, machine_claims_released_dir
        assert (machine_claims_active_dir(cfg) / "t-001.json").is_file()

        release_claim(cfg, "t-001", "completed")
        assert not (machine_claims_active_dir(cfg) / "t-001.json").exists()
        released = machine_claims_released_dir(cfg) / "t-001.json"
        assert released.is_file()
        data = json.loads(released.read_text(encoding="utf-8"))
        assert data["release_reason"] == "completed"

    def test_release_without_active(self, cfg):
        release_claim(cfg, "t-002", "orphaned")
        from qqtools.plugins.qexp.layout import machine_claims_released_dir
        assert (machine_claims_released_dir(cfg) / "t-002.json").is_file()


# ---------------------------------------------------------------------------
# Iterators
# ---------------------------------------------------------------------------


class TestIterators:
    def test_iter_all_tasks(self, cfg):
        save_task(cfg, _make_task("t1"))
        save_task(cfg, _make_task("t2"))
        tasks = iter_all_tasks(cfg)
        assert len(tasks) == 2
        assert {t.task_id for t in tasks} == {"t1", "t2"}

    def test_iter_all_tasks_empty(self, cfg):
        assert iter_all_tasks(cfg) == []

    def test_iter_all_batches(self, cfg):
        save_batch(cfg, _make_batch("b1"))
        batches = iter_all_batches(cfg)
        assert len(batches) == 1

    def test_iter_machines(self, cfg):
        machines = iter_machines(cfg)
        assert len(machines) == 1
        assert machines[0].machine_name == "dev1"
