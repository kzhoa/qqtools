"""Deadline-bound machine-local reads for Idle Job decisions."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
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
    registry_lock_path: Path | None
    pre_executor_check: Callable[[], bool] | None
    done: Event = field(default_factory=Event)
    cancelled: bool = False
    lease: ExitStack | None = None
    evidence: IdleExecutorEvidence | None = None


class IdleExecutorObservation:
    """One outstanding read with locks the decision thread can release if it stalls."""

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
        registry_lock_path: Path | None = None,
        pre_executor_check: Callable[[], bool] | None = None,
    ) -> Iterator[IdleExecutorEvidence | None]:
        """Run an optional read-only check under the registry lock before fencing the executor."""
        if (registry_lock_path is None) != (pre_executor_check is None):
            raise ValueError("The registry lock path and pre-executor check must be supplied together.")
        request = _ObservationRequest(executor, deadline, registry_lock_path, pre_executor_check)
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
            if evidence is not None:
                yield evidence
                return
        finally:
            with self._condition:
                request.cancelled = True
                lease, request.lease = request.lease, None
            if lease is not None:
                lease.close()
        yield None

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

    def _publish_lease(self, request: _ObservationRequest, lock: ExitStack) -> bool:
        # Move each acquired lock to the caller before starting potentially stalled reads.
        with self._condition:
            if not self._is_active_locked(request):
                return False
            if request.lease is None:
                request.lease = lock.pop_all()
            else:
                request.lease.callback(lock.pop_all().close)
            return True

    def _read(self, request: _ObservationRequest) -> IdleExecutorEvidence | None:
        if request.registry_lock_path is not None and request.pre_executor_check is not None:
            with self._condition:
                if not self._is_active_locked(request):
                    return None
            with ExitStack() as lock:
                acquired = lock.enter_context(exclusive(request.registry_lock_path, blocking=False))
                if not acquired or not self._publish_lease(request, lock):
                    return None
            try:
                if not request.pre_executor_check():
                    return None
            except Exception:
                return None

        # A stalled registry read can return after the caller has released the locks.
        while True:
            with self._condition:
                if not self._is_active_locked(request):
                    return None
            with ExitStack() as lock:
                acquired = lock.enter_context(exclusive(request.executor.paths["project_io_lock"], blocking=False))
                if acquired:
                    if not self._publish_lease(request, lock):
                        return None
                    status, unresolved = request.executor.read_idle_evidence_locked()
                    epoch, envelope = status["executor_epoch"], status["envelope"]
                    if isinstance(epoch, str) and epoch and envelope in {"healthy", "degraded", "exceeded"}:
                        return IdleExecutorEvidence(epoch, envelope, unresolved)
                    return None
            time.sleep(max(0, min(_CONTENTION_RETRY_SECONDS, request.deadline - time.monotonic())))
