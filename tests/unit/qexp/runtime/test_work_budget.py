import pytest

from qqtools.plugins.qexp.runtime.work_budget import (
    AdaptiveBatchSizer,
    SliceBudget,
    WorkBudgetPolicy,
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
