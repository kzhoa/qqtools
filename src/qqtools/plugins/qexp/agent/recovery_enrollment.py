"""Bounded background admission enrollment for retained local recovery."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread

from ..runtime.recovery_admission import fence_recovery_admission
from ..runtime.responsibility_capture import CaptureBusy
from .context import MachineRuntime, ProjectBinding
from .recovery_capture import RecoveryCapture, RecoveryProgress

PROJECTS_PER_PASS = 4
PASS_INTERVAL_SECONDS = 1.0
ACTIVE_RETRY_SECONDS = 0.05
BUSY_RETRY_SECONDS = 0.01
PRODUCTIVE_YIELD_SECONDS = 0.01


class RecoveryEnrollment:
    """Prepare fences and retained capture without shared-root I/O in poll().

    The owning machine-agent loop holds scheduler authority for this object's
    lifetime. Only the worker touches shared roots and capture storage. Completion
    is cached for the exact binding; sole-index supervision has its own gate.
    """

    def __init__(self, runtime: MachineRuntime) -> None:
        self.runtime = runtime
        # Only the service thread mutates scheduling and capture state. The
        # foreground reads an immutable completion snapshot for idle policy.
        self._settled: frozenset[ProjectBinding] = frozenset()
        self._preparation_attempted: set[ProjectBinding] = set()
        self._next_project: str | None = None
        self._retry_at: dict[ProjectBinding, float] = {}
        self._stop = Event()
        self._thread: Thread | None = None
        self._captures: dict[ProjectBinding, RecoveryCapture] = {}

    def _refresh_pending(self) -> None:
        _revision, bindings = self.runtime.load_registry()
        settled = self._settled
        self.runtime.recovery_enrollment_pending_projects = {
            binding.project_id for binding in bindings if binding not in settled
        }

    def poll(self) -> None:
        """Start the service and consume completion without shared-project I/O."""
        if self._stop.is_set():
            return
        self._refresh_pending()
        if self._thread is not None:
            if self._thread.is_alive():
                return
            self._thread.join()
            self._thread = None
        if not self.runtime.recovery_enrollment_pending_projects:
            return
        self._thread = Thread(target=self._run, name="qexp-recovery-enrollment", daemon=True)
        try:
            self._thread.start()
        except RuntimeError:
            self._thread = None
            raise

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self._thread is None or not self._thread.is_alive():
            for capture in self._captures.values():
                capture.close()
        # Capture slices also mutate outside publication fences. Join the whole
        # worker before the lifecycle owner may release scheduler authority.

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    delay = self._advance_pass()
                except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                    delay = PASS_INTERVAL_SECONDS
                if delay is None:
                    return
                self._stop.wait(delay)
        finally:
            for capture in self._captures.values():
                capture.close()

    def _advance_pass(self) -> float | None:
        """Advance at most four projects; return the next useful wakeup delay."""
        _revision, bindings = self.runtime.load_registry()
        current = set(bindings)
        for binding in set(self._captures) - current:
            self._captures.pop(binding).close()
        self._settled = self._settled.intersection(current)
        self._preparation_attempted.intersection_update(current)
        self._retry_at = {binding: due for binding, due in self._retry_at.items() if binding in current}
        pending = [binding for binding in bindings if binding not in self._settled]
        now = time.monotonic()
        ready = [binding for binding in pending if self._retry_at.get(binding, 0.0) <= now]
        if ready:
            start = next((index for index, item in enumerate(ready) if item.project_id == self._next_project), 0)
            ordered = ready[start:] + ready[:start]
            unvisited = [binding for binding in ordered if binding not in self._preparation_attempted]
            visited = [binding for binding in ordered if binding in self._preparation_attempted]
            ordered = unvisited + visited
            selected = ordered[:PROJECTS_PER_PASS]
            self._next_project = ordered[len(selected) % len(ordered)].project_id
            should_prepare_only = len(unvisited) > len(selected)
            prepared: dict[ProjectBinding, RecoveryProgress | Exception] = {}
            if should_prepare_only:
                # Registration roots are disjoint. Prepare the bounded first
                # batch concurrently before capture work begins.
                with ThreadPoolExecutor(max_workers=len(selected)) as pool:
                    futures = {
                        binding: pool.submit(self._advance_binding, binding, should_prepare_only=True)
                        for binding in selected
                    }
                    for binding, future in futures.items():
                        try:
                            prepared[binding] = future.result()
                        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                            prepared[binding] = exc
            for binding in selected:
                if self._stop.is_set():
                    break
                retry_delay = PASS_INTERVAL_SECONDS
                try:
                    prepared_result = prepared.get(binding)
                    if isinstance(prepared_result, Exception):
                        raise prepared_result
                    progress = (
                        prepared_result
                        if isinstance(prepared_result, RecoveryProgress)
                        else self._advance_binding(binding, should_prepare_only=False)
                    )
                    if progress is RecoveryProgress.WAITING:
                        if self.runtime.registration_status(binding)["state"] == "superseded":
                            progress = RecoveryProgress.COMPLETE
                except CaptureBusy:
                    # Capture and cleanup share a short-lived parent lock. Keep
                    # the bounded worker responsive when a peer owns one slice.
                    progress = RecoveryProgress.WAITING
                    retry_delay = BUSY_RETRY_SECONDS
                except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                    progress = RecoveryProgress.WAITING
                    if binding in self._captures:
                        retry_delay = ACTIVE_RETRY_SECONDS
                self._preparation_attempted.add(binding)
                if progress is RecoveryProgress.WAITING and binding in self._captures:
                    retry_delay = min(retry_delay, ACTIVE_RETRY_SECONDS)
                if progress is RecoveryProgress.COMPLETE:
                    self._settled = self._settled.union((binding,))
                self._retry_at[binding] = (
                    time.monotonic() + retry_delay if progress is RecoveryProgress.WAITING else 0.0
                )
        pending = [binding for binding in bindings if binding not in self._settled]
        if not pending:
            return None
        next_due = min(self._retry_at.get(binding, 0.0) for binding in pending)
        # Productive batches yield briefly to peers but never wait for dispatch.
        return max(PRODUCTIVE_YIELD_SECONDS, min(PASS_INTERVAL_SECONDS, next_due - time.monotonic()))

    def _advance_binding(self, binding: ProjectBinding, *, should_prepare_only: bool) -> RecoveryProgress:
        """Run one bounded project step; durable guards remain the authority."""
        if not self.runtime.prepare_recovery_registration(binding, blocking=True):
            return RecoveryProgress.WAITING
        if should_prepare_only:
            return RecoveryProgress.ADVANCED
        if not fence_recovery_admission(self.runtime, binding).is_fenced:
            return RecoveryProgress.WAITING
        if binding not in self._captures:
            self._captures[binding] = RecoveryCapture(self.runtime, binding)
        capture = self._captures[binding]
        progress = capture.advance_step()
        if progress is not RecoveryProgress.COMPLETE:
            return progress
        is_group_active = capture.activate_group_authority()
        is_source_released = capture.release_source()
        return RecoveryProgress.COMPLETE if is_group_active and is_source_released else RecoveryProgress.WAITING
