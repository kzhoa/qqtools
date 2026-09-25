"""Portable scheduler work accounting and adaptive slice sizing."""

from __future__ import annotations

import time
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, TypeVar

DEFAULT_SLICE_DEADLINE_MS = 50
DEFAULT_RECORD_HARD_LIMIT = 64
DEFAULT_OPERATION_HARD_LIMIT = 256
DEFAULT_INITIAL_BATCH_SIZE = 4
DEFAULT_GROWTH_OBSERVATIONS = 3
DIAGNOSTIC_PUBLISH_INTERVAL_SECONDS = 30
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class WorkBudgetPolicy:
    """Environment-independent limits for one scheduler work slice."""

    record_hard_limit: int = DEFAULT_RECORD_HARD_LIMIT
    operation_hard_limit: int = DEFAULT_OPERATION_HARD_LIMIT
    soft_deadline_ms: int = DEFAULT_SLICE_DEADLINE_MS
    minimum_batch_size: int = 1
    initial_batch_size: int = DEFAULT_INITIAL_BATCH_SIZE
    growth_observations: int = DEFAULT_GROWTH_OBSERVATIONS

    def __post_init__(self) -> None:
        values = (
            self.record_hard_limit,
            self.operation_hard_limit,
            self.soft_deadline_ms,
            self.minimum_batch_size,
            self.initial_batch_size,
            self.growth_observations,
        )
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("work budget limits must be positive.")
        if self.minimum_batch_size > self.initial_batch_size:
            raise ValueError("minimum_batch_size must not exceed initial_batch_size.")
        if self.initial_batch_size > self.record_hard_limit:
            raise ValueError("initial_batch_size must not exceed record_hard_limit.")


@dataclass(slots=True)
class RuntimeDiagnostics:
    """Collect bounded operation counts and monotonic stage timings."""

    clock_ns: Callable[[], int] = time.monotonic_ns
    counters: Counter[str] = field(default_factory=Counter)
    elapsed_ns: Counter[str] = field(default_factory=Counter)
    maximum_ns: Counter[str] = field(default_factory=Counter)

    def increment(self, name: str, amount: int = 1) -> None:
        if amount < 0:
            raise ValueError("diagnostic increments must not be negative.")
        self.counters[name] += amount

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        started_ns = self.clock_ns()
        self.increment(f"{name}.calls")
        try:
            yield
        finally:
            duration_ns = max(0, self.clock_ns() - started_ns)
            self.elapsed_ns[name] += duration_ns
            self.maximum_ns[name] = max(self.maximum_ns[name], duration_ns)

    def snapshot(self) -> dict[str, object]:
        names = sorted(set(self.elapsed_ns) | set(self.maximum_ns))
        return {
            "counters": dict(sorted(self.counters.items())),
            "timings": {
                name: {
                    "total_ns": self.elapsed_ns[name],
                    "maximum_ns": self.maximum_ns[name],
                }
                for name in names
            },
        }


_ACTIVE_DIAGNOSTICS: ContextVar[RuntimeDiagnostics | None] = ContextVar("qexp_runtime_diagnostics", default=None)


@contextmanager
def activate_diagnostics(
    diagnostics: RuntimeDiagnostics | None,
) -> Iterator[RuntimeDiagnostics | None]:
    """Activate a collector without masking an outer collector when omitted."""
    if diagnostics is None:
        yield _ACTIVE_DIAGNOSTICS.get()
        return
    token = _ACTIVE_DIAGNOSTICS.set(diagnostics)
    try:
        yield diagnostics
    finally:
        _ACTIVE_DIAGNOSTICS.reset(token)


@contextmanager
def diagnostic_span(name: str) -> Iterator[None]:
    diagnostics = _ACTIVE_DIAGNOSTICS.get()
    if diagnostics is None:
        yield
        return
    with diagnostics.measure(name):
        yield


def diagnostic_increment(name: str, amount: int = 1) -> None:
    diagnostics = _ACTIVE_DIAGNOSTICS.get()
    if diagnostics is not None:
        diagnostics.increment(name, amount)


def diagnostic_observe_ns(name: str, duration_ns: int) -> None:
    """Record an observed duration without treating it as measured operation time."""
    if type(duration_ns) is not int or duration_ns < 0:
        raise ValueError("observed duration must be a nonnegative integer")
    diagnostics = _ACTIVE_DIAGNOSTICS.get()
    if diagnostics is not None:
        diagnostics.increment(f"{name}.observations")
        diagnostics.elapsed_ns[name] += duration_ns
        diagnostics.maximum_ns[name] = max(diagnostics.maximum_ns[name], duration_ns)


@dataclass(slots=True)
class SliceBudget:
    """Enforce count limits and a monotonic soft deadline between records."""

    policy: WorkBudgetPolicy = field(default_factory=WorkBudgetPolicy)
    clock_ns: Callable[[], int] = time.monotonic_ns
    records_used: int = 0
    operations_used: int = 0
    _deadline_ns: int = field(init=False)

    def __post_init__(self) -> None:
        self._deadline_ns = self.clock_ns() + self.policy.soft_deadline_ms * 1_000_000

    def can_start_record(self, *, operations: int = 1, check_deadline: bool = True) -> bool:
        if type(operations) is not int or operations <= 0:
            raise ValueError("record operations must be a positive integer.")
        return self.records_used < self.policy.record_hard_limit and self.can_start_operation(
            operations=operations, check_deadline=check_deadline
        )

    def can_start_operation(self, *, operations: int = 1, check_deadline: bool = True) -> bool:
        """Return whether more bounded I/O may begin in this slice."""
        if type(operations) is not int or operations <= 0:
            raise ValueError("operation count must be a positive integer.")
        return self.operations_used + operations <= self.policy.operation_hard_limit and (
            not check_deadline or self.clock_ns() < self._deadline_ns
        )

    def consume_operation(self, *, operations: int = 1) -> None:
        """Account for bounded I/O that is not a candidate record."""
        if type(operations) is not int or operations <= 0:
            raise ValueError("operation count must be a positive integer.")
        if self.operations_used + operations > self.policy.operation_hard_limit:
            raise RuntimeError("scheduler operation hard limit exceeded.")
        self.operations_used += operations

    def consume_record(self, *, operations: int = 1) -> None:
        if type(operations) is not int or operations <= 0:
            raise ValueError("record operations must be a positive integer.")
        if self.records_used >= self.policy.record_hard_limit:
            raise RuntimeError("scheduler record hard limit exceeded.")
        if self.operations_used + operations > self.policy.operation_hard_limit:
            raise RuntimeError("scheduler operation hard limit exceeded.")
        self.records_used += 1
        self.operations_used += operations


@dataclass(slots=True)
class OperationReservation:
    """A maximum operation-cost reservation owned by one invocation ledger."""

    ledger: InvocationLedger
    operations: int
    _active: bool = True

    def commit(self, *, actual_operations: int) -> None:
        """Commit actual usage and return any unused reserved operations."""
        self.ledger.commit(self, actual_operations=actual_operations)

    def release(self) -> None:
        """Return this reservation without charging operations."""
        self.ledger.release(self)


@dataclass(slots=True)
class InvocationLedger:
    """Account one end-to-end maintenance invocation across nested phases.

    Operation reservations are included in admission. Each admitted phase step
    reserves its worst-case cost before it consumes a semantic item; after the
    step, callers commit the measured operation count or release the unused
    reservation. The deadline is cooperative and only prevents later admission.
    """

    semantic_item_limit: int
    operation_limit: int = DEFAULT_OPERATION_HARD_LIMIT
    deadline_ms: int = DEFAULT_SLICE_DEADLINE_MS
    clock_ns: Callable[[], int] = time.monotonic_ns
    semantic_items_consumed: int = 0
    operations_consumed: int = 0
    operations_reserved: int = 0
    _started_ns: int = field(init=False)
    _deadline_ns: int = field(init=False)
    _exhaustion_reason: str | None = None
    deadline_overrun_ms: int = 0

    def __post_init__(self) -> None:
        values = (self.semantic_item_limit, self.operation_limit, self.deadline_ms)
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("invocation ledger limits must be positive integers.")
        self._started_ns = self.clock_ns()
        self._deadline_ns = self._started_ns + self.deadline_ms * 1_000_000

    @property
    def semantic_items_remaining(self) -> int:
        return max(0, self.semantic_item_limit - self.semantic_items_consumed)

    @property
    def operations_remaining(self) -> int:
        return max(0, self.operation_limit - self.operations_consumed - self.operations_reserved)

    @property
    def elapsed_ms(self) -> int:
        return max(0, (self.clock_ns() - self._started_ns) // 1_000_000)

    @property
    def exhaustion_reason(self) -> str | None:
        """Return the first boundary that stopped admission, if any."""
        if self._exhaustion_reason is not None:
            return self._exhaustion_reason
        if self.semantic_items_remaining <= 0:
            return "semantic_items"
        if self.operations_remaining <= 0:
            return "operations"
        if self.clock_ns() >= self._deadline_ns:
            return "deadline"
        return None

    def can_admit(self, *, maximum_operations: int = 1) -> bool:
        """Return whether one more semantic item can reserve its exit cost."""
        self._validate_count(maximum_operations, "maximum_operations", positive=True)
        if self.semantic_items_remaining <= 0:
            self._exhaustion_reason = self._exhaustion_reason or "semantic_items"
            return False
        if self.operations_remaining < maximum_operations:
            self._exhaustion_reason = self._exhaustion_reason or "operations"
            return False
        if self.clock_ns() >= self._deadline_ns:
            self._exhaustion_reason = self._exhaustion_reason or "deadline"
            return False
        return True

    def reserve_operations(self, *, maximum_operations: int) -> OperationReservation | None:
        """Reserve a step's maximum operation cost before beginning the step."""
        self._validate_count(maximum_operations, "maximum_operations", positive=True)
        if not self.can_admit(maximum_operations=maximum_operations):
            return None
        self.operations_reserved += maximum_operations
        return OperationReservation(self, maximum_operations)

    def consume_semantic_item(self) -> None:
        """Charge one source examination or standalone durable transition."""
        if self.semantic_items_remaining <= 0:
            self._exhaustion_reason = self._exhaustion_reason or "semantic_items"
            raise RuntimeError("invocation semantic-item limit exceeded.")
        self.semantic_items_consumed += 1

    def charge_operations(self, *, operations: int) -> None:
        """Charge unreserved setup/report operations against the hard limit."""
        self._validate_count(operations, "operations", positive=True)
        if self.operations_remaining < operations:
            self._exhaustion_reason = self._exhaustion_reason or "operations"
            raise RuntimeError("invocation operation hard limit exceeded.")
        self.operations_consumed += operations

    def commit(self, reservation: OperationReservation, *, actual_operations: int) -> None:
        """Commit usage bounded by a live reservation, releasing its remainder."""
        self._validate_count(actual_operations, "actual_operations", positive=False)
        self._validate_reservation(reservation)
        if actual_operations > reservation.operations:
            raise ValueError("actual_operations must not exceed the reserved operation count.")
        self.operations_reserved -= reservation.operations
        self.operations_consumed += actual_operations
        reservation._active = False
        self._record_deadline_overrun()

    def release(self, reservation: OperationReservation) -> None:
        """Release unused reserved operations without charging the step."""
        self._validate_reservation(reservation)
        self.operations_reserved -= reservation.operations
        reservation._active = False
        self._record_deadline_overrun()

    def report(self) -> dict[str, int | str | None]:
        """Return the bounded public counters for this invocation."""
        self._record_deadline_overrun()
        return {
            "semantic_items_consumed": self.semantic_items_consumed,
            "operations_consumed": self.operations_consumed,
            "elapsed_ms": self.elapsed_ms,
            "semantic_items_remaining": self.semantic_items_remaining,
            "operations_remaining": self.operations_remaining,
            "exhaustion_reason": self.exhaustion_reason,
            "deadline_overrun_ms": self.deadline_overrun_ms,
        }

    def _validate_reservation(self, reservation: OperationReservation) -> None:
        if not isinstance(reservation, OperationReservation) or reservation.ledger is not self:
            raise ValueError("operation reservation belongs to another ledger.")
        if not reservation._active:
            raise ValueError("operation reservation is no longer active.")

    @staticmethod
    def _validate_count(value: int, name: str, *, positive: bool) -> None:
        if type(value) is not int or value < (1 if positive else 0):
            qualifier = "positive" if positive else "nonnegative"
            raise ValueError(f"{name} must be a {qualifier} integer.")

    def _record_deadline_overrun(self) -> None:
        now = self.clock_ns()
        if now > self._deadline_ns:
            overrun = (now - self._deadline_ns + 999_999) // 1_000_000
            self.deadline_overrun_ms = max(self.deadline_overrun_ms, overrun)
            if self.semantic_items_remaining > 0 and self.operations_remaining > 0:
                self._exhaustion_reason = self._exhaustion_reason or "deadline"


def bounded_records(records: Iterable[T], budget: SliceBudget) -> Iterator[T]:
    """Yield only records admitted before the portable count/deadline boundary."""
    iterator = iter(records)
    while budget.can_start_record():
        try:
            record = next(iterator)
        except StopIteration:
            return
        budget.consume_record()
        yield record


@dataclass(slots=True)
class AdaptiveBatchSizer:
    """Conservatively resize an in-process batch from monotonic observations."""

    policy: WorkBudgetPolicy = field(default_factory=WorkBudgetPolicy)
    batch_size: int = field(init=False)
    estimated_record_ns: int | None = None
    _growth_streak: int = 0

    def __post_init__(self) -> None:
        self.batch_size = self.policy.initial_batch_size

    def observe(self, elapsed_ns: int) -> int:
        if elapsed_ns <= 0:
            raise ValueError("elapsed_ns must be positive.")
        if self.estimated_record_ns is None:
            self.estimated_record_ns = elapsed_ns
        else:
            # A 3:1 EWMA plus the latest sample is conservative on sudden slowdowns.
            average_ns = (3 * self.estimated_record_ns + elapsed_ns + 3) // 4
            self.estimated_record_ns = max(average_ns, elapsed_ns)
        deadline_ns = self.policy.soft_deadline_ms * 1_000_000
        target = max(
            self.policy.minimum_batch_size,
            min(self.policy.record_hard_limit, deadline_ns // self.estimated_record_ns),
        )
        if target < self.batch_size:
            self.batch_size = max(
                self.policy.minimum_batch_size,
                min(target, max(self.policy.minimum_batch_size, self.batch_size // 2)),
            )
            self._growth_streak = 0
        elif target > self.batch_size:
            self._growth_streak += 1
            if self._growth_streak >= self.policy.growth_observations:
                self.batch_size = min(self.policy.record_hard_limit, self.batch_size + 1)
                self._growth_streak = 0
        else:
            self._growth_streak = 0
        return self.batch_size
