from __future__ import annotations

import pytest

from qqtools.plugins.qexp.indexes import load_index, update_index_on_submit
from qqtools.plugins.qexp.layout import init_shared_root
from qqtools.plugins.qexp.models import (
    Meta,
    PHASE_FAILED,
    PHASE_QUEUED,
    PHASE_RUNNING,
    PHASE_STARTING,
    Task,
    TaskLineage,
    TaskResult,
    TaskRuntime,
    TaskSpec,
    TaskStatus,
    TaskTimestamps,
    utc_now_iso,
)
from qqtools.plugins.qexp.scheduler import (
    Scheduler,
    reconcile_running_tasks,
    run_dispatch_cycle,
)
from qqtools.plugins.qexp.storage import cas_update_task, load_task, save_task


@pytest.fixture()
def cfg(tmp_path):
    return init_shared_root(tmp_path / ".qexp", "dev1", runtime_root=tmp_path / "runtime")


def _make_task(task_id: str, machine: str = "dev1") -> Task:
    now = utc_now_iso()
    return Task(
        meta=Meta.new(machine),
        task_id=task_id,
        name=None,
        group=None,
        batch_id=None,
        machine_name=machine,
        attempt=1,
        spec=TaskSpec(command=["echo", "hi"], requested_gpus=1),
        status=TaskStatus(phase=PHASE_QUEUED),
        runtime=TaskRuntime(),
        timestamps=TaskTimestamps(created_at=now, queued_at=now),
        result=TaskResult(),
        lineage=TaskLineage(),
    )


class TestDispatchCycle:
    def test_empty_queue(self, cfg):
        launched = run_dispatch_cycle(cfg)
        assert launched == []

    def test_dispatch_without_tracker(self, cfg):
        t = _make_task("t1")
        save_task(cfg, t)
        update_index_on_submit(cfg, t)

        launched = run_dispatch_cycle(cfg, tracker=None)
        assert launched == ["t1"]

        loaded = load_task(cfg, "t1")
        assert loaded.status.phase == PHASE_STARTING
        assert loaded.meta.revision == 3  # 1 (create) + 2 (dispatching + starting)

    def test_dispatch_skips_other_machine(self, cfg):
        t = _make_task("t1", machine="other-machine")
        save_task(cfg, t)
        update_index_on_submit(cfg, t)

        launched = run_dispatch_cycle(cfg)
        assert launched == []

    def test_dispatch_multiple(self, cfg):
        for i in range(3):
            t = _make_task(f"t{i}")
            save_task(cfg, t)
            update_index_on_submit(cfg, t)

        launched = run_dispatch_cycle(cfg)
        assert len(launched) == 3

    def test_index_updated_after_dispatch(self, cfg):
        t = _make_task("t1")
        save_task(cfg, t)
        update_index_on_submit(cfg, t)

        run_dispatch_cycle(cfg)
        assert "t1" not in load_index(cfg, "state", PHASE_QUEUED)
        assert "t1" in load_index(cfg, "state", PHASE_STARTING)


class _FakeTracker:
    def __init__(self, available: int = 8):
        self._available = available
        self.allocated: dict[str, list[int]] = {}
        self.visible_gpu_ids: list[int] = list(range(available))
        self.reserved_gpu_ids: set[int] = set()
        self.task_id_to_gpu_ids: dict[str, list[int]] = {}

    def allocate(self, task_id: str, num_gpus: int) -> list[int] | None:
        if num_gpus > self._available:
            return None
        gpus = list(range(self._available - num_gpus, self._available))
        self._available -= num_gpus
        self.allocated[task_id] = gpus
        return gpus

    def release(self, task_id: str) -> None:
        gpus = self.allocated.pop(task_id, [])
        self._available += len(gpus)


class _FakeExecutor:
    """Executor that doesn't require tmux."""

    def __init__(self):
        self.launched: list[str] = []

    def launch_in_window(self, cfg, task_id, session_name="experiments"):
        self.launched.append(task_id)
        return f"@fake-window-{task_id}"

    def launch_task(self, cfg, task_id, group):
        session_name = group or "experiments"
        self.launched.append(f"{task_id}:{session_name}")
        return f"@fake-window-{task_id}"

    def cleanup_window(self, window_id):
        pass


class TestSchedulerWithTracker:
    def test_allocates_gpus(self, cfg):
        t = _make_task("t1")
        t.spec.requested_gpus = 2
        save_task(cfg, t)
        update_index_on_submit(cfg, t)

        tracker = _FakeTracker(available=4)
        executor = _FakeExecutor()
        scheduler = Scheduler(tracker=tracker, executor=executor)
        launched = scheduler.run_dispatch_cycle(cfg)
        assert launched == ["t1"]
        loaded = load_task(cfg, "t1")
        assert len(loaded.runtime.assigned_gpus) == 2
        assert executor.launched == ["t1:experiments"]

    def test_routes_group_to_tmux_session(self, cfg):
        t = _make_task("t1")
        t.group = "contract_n_4and6"
        save_task(cfg, t)
        update_index_on_submit(cfg, t)

        tracker = _FakeTracker(available=2)
        executor = _FakeExecutor()
        scheduler = Scheduler(tracker=tracker, executor=executor)
        launched = scheduler.run_dispatch_cycle(cfg)

        assert launched == ["t1"]
        assert executor.launched == ["t1:contract_n_4and6"]

    def test_insufficient_gpus_rollback(self, cfg):
        t = _make_task("t1")
        t.spec.requested_gpus = 4
        save_task(cfg, t)
        update_index_on_submit(cfg, t)

        tracker = _FakeTracker(available=2)
        executor = _FakeExecutor()
        scheduler = Scheduler(tracker=tracker, executor=executor)
        launched = scheduler.run_dispatch_cycle(cfg)
        assert launched == []
        loaded = load_task(cfg, "t1")
        assert loaded.status.phase == PHASE_QUEUED

    def test_multiple_tasks_gpu_backfill(self, cfg):
        """Tasks that fit get scheduled; those that don't are rolled back."""
        t1 = _make_task("t1")
        t1.spec.requested_gpus = 2
        save_task(cfg, t1)
        update_index_on_submit(cfg, t1)

        t2 = _make_task("t2")
        t2.spec.requested_gpus = 4
        save_task(cfg, t2)
        update_index_on_submit(cfg, t2)

        t3 = _make_task("t3")
        t3.spec.requested_gpus = 1
        save_task(cfg, t3)
        update_index_on_submit(cfg, t3)

        tracker = _FakeTracker(available=4)
        executor = _FakeExecutor()
        scheduler = Scheduler(tracker=tracker, executor=executor)
        launched = scheduler.run_dispatch_cycle(cfg)

        # t1 (2 gpus) fits, t2 (4 gpus) doesn't (only 2 left), t3 (1 gpu) fits
        assert "t1" in launched
        assert "t3" in launched
        assert "t2" not in launched
        assert load_task(cfg, "t2").status.phase == PHASE_QUEUED


class TestReconcileRunning:
    def test_no_orphans(self, cfg):
        orphaned = reconcile_running_tasks(cfg)
        assert orphaned == []

    def test_detects_crashed_wrapper(self, cfg):
        t = _make_task("t1")
        save_task(cfg, t)
        update_index_on_submit(cfg, t)

        # Move to running with a fake dead PID
        from qqtools.plugins.qexp.indexes import update_index_on_phase_change
        t = load_task(cfg, "t1")
        t.status.phase = PHASE_RUNNING
        t.runtime.wrapper_pid = 99999999  # dead PID
        t.timestamps.started_at = "2020-01-01T00:00:00Z"  # long ago
        cas_update_task(cfg, t, t.meta.revision)
        update_index_on_phase_change(cfg, "t1", PHASE_QUEUED, PHASE_RUNNING)

        tracker = _FakeTracker()
        executor = _FakeExecutor()
        scheduler = Scheduler(tracker=tracker, executor=executor)
        failed = scheduler.reconcile_running_tasks(cfg)
        assert "t1" in failed
        assert load_task(cfg, "t1").status.phase == PHASE_FAILED

    def test_startup_timeout(self, cfg):
        t = _make_task("t1")
        save_task(cfg, t)
        update_index_on_submit(cfg, t)

        from qqtools.plugins.qexp.indexes import update_index_on_phase_change
        t = load_task(cfg, "t1")
        t.status.phase = PHASE_STARTING
        t.timestamps.started_at = "2020-01-01T00:00:00Z"  # long ago
        cas_update_task(cfg, t, t.meta.revision)
        update_index_on_phase_change(cfg, "t1", PHASE_QUEUED, PHASE_STARTING)

        tracker = _FakeTracker()
        executor = _FakeExecutor()
        scheduler = Scheduler(
            tracker=tracker, executor=executor, startup_grace_seconds=1
        )
        failed = scheduler.reconcile_running_tasks(cfg)
        assert "t1" in failed
