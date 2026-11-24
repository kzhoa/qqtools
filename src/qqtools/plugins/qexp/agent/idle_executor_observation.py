"""Deadline-bound machine-local reads for Idle Job decisions."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from threading import Condition, Event, Thread

from ..runtime.locks import exclusive
from .project_io_executor import ProjectIOExecutor
from .project_io_protocol import ProjectIORequest

_CONTENTION_RETRY_SECONDS = 0.002


@dataclass(frozen=True, slots=True)
class IdleExecutorEvidence:
    epoch: str
    envelope: str
    unresolved: tuple[ProjectIORequest, ...]


@dataclass(slots=True)
class _ObservationRequest:
    executor: ProjectIOExecutor
    deadline: float
    pre_executor_check: Callable[[], bool] | None
    done: Event = field(default_factory=Event)
    cancelled: bool = False
    lease: ExitStack | None = None
    evidence: IdleExecutorEvidence | None = None


class IdleExecutorObservation:
    """One outstanding read. The decision thread can revoke its lock during a stalled scan."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._request: _ObservationRequest | None = None
        self._closed = False
        Thread(target=self._run, name="qexp-idle-executor-observation", daemon=True).start()

    @contextmanager
    def guard(
        self,
        executor: ProjectIOExecutor,
        *,
        deadline: float,
        pre_executor_check: Callable[[], bool] | None = None,
    ) -> Iterator[IdleExecutorEvidence | None]:
        """Run an optional read-only check in the observer before acquiring the executor lock."""
        request = _ObservationRequest(executor, deadline, pre_executor_check)
        with self._condition:
            accepted = not self._closed and self._request is None and time.monotonic() < deadline
            if accepted:
                self._request = request
                self._condition.notify()
        if not accepted:
            yield None
            return
        try:
            ready = request.done.wait(max(0, deadline - time.monotonic()))
            with self._condition:
                evidence = request.evidence if ready and self._is_active_locked(request) else None
            yield evidence
        finally:
            with self._condition:
                request.cancelled = True
                lease, request.lease = request.lease, None
            if lease is not None:
                lease.close()

    def observe(self, executor: ProjectIOExecutor, *, deadline: float) -> IdleExecutorEvidence | None:
        with self.guard(executor, deadline=deadline) as evidence:
            return evidence

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._request is not None or self._closed)
                if self._closed:
                    return
                request = self._request
            if request is None:
                continue
            try:
                evidence = self._read(request)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError, KeyError):
                evidence = None
            with self._condition:
                if self._is_active_locked(request):
                    request.evidence = evidence
                self._request = None
                request.done.set()
            del request

    def _is_active_locked(self, request: _ObservationRequest) -> bool:
        return not request.cancelled and not self._closed and time.monotonic() < request.deadline

    def _read(self, request: _ObservationRequest) -> IdleExecutorEvidence | None:
        if request.pre_executor_check is not None:
            with self._condition:
                if not self._is_active_locked(request):
                    return None
            try:
                if not request.pre_executor_check():
                    return None
            except Exception:
                return None

        # A stalled check can return after the caller released its registry guard.
        while True:
            with self._condition:
                if not self._is_active_locked(request):
                    return None
            with ExitStack() as lock:
                acquired = lock.enter_context(exclusive(request.executor.paths["project_io_lock"], blocking=False))
                if acquired:
                    with self._condition:
                        if not self._is_active_locked(request):
                            return None
                        # No read or reconciliation may mutate transport after a timeout.
                        # The control thread releases this exact lock, even if I/O is still pending.
                        request.lease = lock.pop_all()
                    status, unresolved = request.executor.read_idle_evidence_locked()
                    epoch, envelope = status["executor_epoch"], status["envelope"]
                    if isinstance(epoch, str) and epoch and envelope in {"healthy", "degraded", "exceeded"}:
                        return IdleExecutorEvidence(epoch, envelope, unresolved)
                    return None
            time.sleep(max(0, min(_CONTENTION_RETRY_SECONDS, request.deadline - time.monotonic())))
