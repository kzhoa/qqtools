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

    def can_start_record(self) -> bool:
        return (
            self.records_used < self.policy.record_hard_limit
            and self.operations_used < self.policy.operation_hard_limit
            and self.clock_ns() < self._deadline_ns
        )

    def consume_record(self, *, operations: int = 1) -> None:
        if operations <= 0:
            raise ValueError("record operations must be positive.")
        if self.records_used >= self.policy.record_hard_limit:
            raise RuntimeError("scheduler record hard limit exceeded.")
        if self.operations_used + operations > self.policy.operation_hard_limit:
            raise RuntimeError("scheduler operation hard limit exceeded.")
        self.records_used += 1
        self.operations_used += operations


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
