from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.machine_agent import MachineRuntime, dispatch_machine_cycle_locked
from qqtools.plugins.qexp.runtime.filesystem_qualification import (
    FilesystemProbeEvidence,
    evaluate_filesystem_probe,
)
from qqtools.plugins.qexp.runtime.store import read_json
from qqtools.plugins.qexp.runtime.tasks import load_task
from qqtools.plugins.qexp.runtime.work_budget import (
    AdaptiveBatchSizer,
    RuntimeDiagnostics,
    SliceBudget,
    WorkBudgetPolicy,
    activate_diagnostics,
    bounded_records,
)


class _Clock:
    def __init__(self) -> None:
        self.now_ns = 0

    def __call__(self) -> int:
        return self.now_ns

    def advance(self, elapsed_ns: int) -> None:
        self.now_ns += elapsed_ns


class _FailOnExtraRead:
    def __init__(self, count: int) -> None:
        self.count = count
        self.reads = 0

    def __iter__(self):
        for value in range(self.count):
            self.reads += 1
            yield value


@pytest.mark.parametrize("history_size", [0, 1, 100, 1000, 10000])
@pytest.mark.parametrize("scenario", ["full", "idle_empty", "idle_ready"])
def test_slice_operation_count_is_independent_of_terminal_history(history_size: int, scenario: str) -> None:
    policy = WorkBudgetPolicy(record_hard_limit=64, operation_hard_limit=256)
    budget = SliceBudget(policy=policy)
    active_count = {"full": 0, "idle_empty": 0, "idle_ready": 1000}[scenario]
    project = {
        "terminal_task_count": history_size,
        "ready_candidates": range(active_count),
    }

    inspected = list(bounded_records(project["ready_candidates"], budget))

    expected = min(active_count, policy.record_hard_limit)
    assert len(inspected) == expected
    assert budget.records_used == expected
    assert budget.operations_used <= policy.operation_hard_limit
    assert project["terminal_task_count"] == history_size


def test_slice_budget_uses_monotonic_deadline_between_records() -> None:
    clock = _Clock()
    budget = SliceBudget(
        policy=WorkBudgetPolicy(soft_deadline_ms=10, initial_batch_size=1),
        clock_ns=clock,
    )
    inspected = []
    for record in bounded_records(range(10), budget):
        inspected.append(record)
        clock.advance(4_000_000)

    assert inspected == [0, 1, 2]
    assert budget.records_used == 3


def test_slice_budget_does_not_prefetch_after_hard_limit() -> None:
    records = _FailOnExtraRead(10)
    budget = SliceBudget(
        policy=WorkBudgetPolicy(
            record_hard_limit=4,
            operation_hard_limit=4,
            initial_batch_size=4,
        )
    )

    assert list(bounded_records(records, budget)) == [0, 1, 2, 3]
    assert records.reads == 4


def test_work_budget_rejects_boolean_limits() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        WorkBudgetPolicy(record_hard_limit=True)


def test_adaptive_batch_shrinks_on_delay_and_recovers_one_step_at_a_time() -> None:
    policy = WorkBudgetPolicy(
        record_hard_limit=8,
        soft_deadline_ms=8,
        initial_batch_size=8,
        growth_observations=2,
    )
    sizer = AdaptiveBatchSizer(policy)

    assert sizer.observe(20_000_000) == 1
    recovery = [sizer.observe(100_000) for _ in range(20)]
    assert recovery == sorted(recovery)
    assert recovery[-1] == policy.record_hard_limit
    assert sizer.batch_size <= policy.record_hard_limit


def test_failed_record_does_not_hide_hard_limit_or_cursor_progress() -> None:
    policy = WorkBudgetPolicy(record_hard_limit=4, operation_hard_limit=4)
    budget = SliceBudget(policy=policy)
    visited = []

    for record in bounded_records(range(10), budget):
        visited.append(record)
        try:
            if record == 1:
                raise OSError("injected read failure")
        except OSError:
            continue

    assert visited == [0, 1, 2, 3]
    assert budget.records_used == policy.record_hard_limit
    assert visited[-1] + 1 == 4


def test_runtime_diagnostics_count_task_reads_and_machine_stages(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp",
        "gpu-1",
        runtime_root=tmp_path / "legacy-runtime",
    )
    task = submit(cfg, ["echo", "ok"], working_dir=work_dir)
    diagnostics = RuntimeDiagnostics()

    with activate_diagnostics(diagnostics):
        assert load_task(cfg, task.task_id).task_id == task.task_id

    assert diagnostics.snapshot()["counters"]["task_json_read.records"] == 1
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.add_binding(cfg.shared_root, cfg.machine_name)
    dispatch_machine_cycle_locked(
        runtime,
        available_gpus=[],
        supervise=False,
        publish_snapshots=False,
    )
    value = read_json(runtime.paths["diagnostics"] / "scheduler-cycle.json")
    cycle = value["scheduler_diagnostic"]
    assert cycle["counters"]["maintain_project.calls"] == 1
    assert cycle["counters"]["offer_due_tasks.calls"] == 1
    assert cycle["counters"]["scheduler.work.skipped_no_capacity"] == 1
    assert "run_dispatch_cycle.calls" not in cycle["counters"]
    assert cycle["counters"].get("task_json_read.records", 0) == 0
    assert "reservation_enumeration" in cycle["timings"]


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"peer_host": "host-a"}, "probe hosts must be distinct"),
        ({"initiator_host": "   "}, "both host identities are required"),
        ({"exclusive_lock": False}, "cross-host exclusive lock failed"),
        ({"exclusive_lock": 1}, "cross-host exclusive lock failed"),
        ({"atomic_replace": False}, "atomic replace visibility failed"),
        ({"fsync_visibility": False}, "fsync durability visibility failed"),
        ({"failure_cleanup": False}, "failure cleanup behavior failed"),
    ],
)
def test_filesystem_probe_fails_closed(changes: dict[str, object], reason: str) -> None:
    values = {
        "initiator_host": "host-a",
        "peer_host": "host-b",
        "exclusive_lock": True,
        "atomic_replace": True,
        "fsync_visibility": True,
        "failure_cleanup": True,
    }
    values.update(changes)

    result = evaluate_filesystem_probe(FilesystemProbeEvidence(**values))

    assert result.is_qualified is False
    assert reason in result.reasons


def test_filesystem_probe_accepts_complete_two_host_evidence() -> None:
    result = evaluate_filesystem_probe(FilesystemProbeEvidence("host-a", "host-b", True, True, True, True))

    assert result.is_qualified is True
    assert result.reasons == ()
