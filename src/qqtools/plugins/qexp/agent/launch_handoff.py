"""Machine-local runner launch handoff polling and observer provisioning."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..config_types import RootConfig
from ..executor import (
    Executor,
    ExecutorLaunchHandoffTimeout,
    LaunchHandle,
    LaunchHandoff,
    append_launch_failure_diagnostic,
    launch_failure_handle,
)
from ..observer_provisioning import submit_observer_job
from ..runtime.paths import local_paths
from ..runtime.resources.reservations import ReservationIdentity, reservation_snapshot
from ..runtime.work_budget import diagnostic_increment, diagnostic_span
from ..scheduler import fail_attempt
from .context import MachineRuntime


@dataclass(frozen=True, slots=True)
class _PendingLaunchHandoff:
    cfg: RootConfig
    task_id: str
    attempt_id: str
    fencing_token: int
    project_id: str
    handoff: LaunchHandoff
    handle: LaunchHandle | None = None
    compensation_committed: bool = False
    retry_not_before: float = 0.0
    retry_failures: int = 0
    reservation_id: str | None = None


def _launch_intent_exists(path: Path) -> bool:
    """Check for intent without treating inaccessible metadata as missing."""
    try:
        path.stat()
    except FileNotFoundError:
        return False
    return True


def _queue_observer_provisioning(
    executor: Executor,
    pending: _PendingLaunchHandoff,
) -> bool:
    """Queue observer creation while keeping dispatch poll non-blocking."""
    handle = pending.handle
    if handle is None:
        return False
    provision = getattr(executor, "provision_observer", None)
    if callable(provision):
        try:
            provision(pending.cfg, pending.task_id, pending.attempt_id, handle)
        except Exception as exc:
            append_launch_failure_diagnostic(pending.cfg, pending.task_id, pending.attempt_id, exc)
        return True

    attach = getattr(executor, "attach_observer", None)
    if not callable(attach):
        return False
    key = (str(pending.cfg.shared_root), pending.task_id, pending.attempt_id)

    def attach_later() -> None:
        try:
            attach(pending.cfg, pending.task_id, pending.attempt_id, handle)
        except Exception as exc:
            append_launch_failure_diagnostic(pending.cfg, pending.task_id, pending.attempt_id, exc)

    try:
        accepted = submit_observer_job(key, attach_later)
    except Exception as exc:
        append_launch_failure_diagnostic(pending.cfg, pending.task_id, pending.attempt_id, exc)
        return False
    if not accepted:
        append_launch_failure_diagnostic(
            pending.cfg,
            pending.task_id,
            pending.attempt_id,
            RuntimeError("bounded observer provisioning worker is occupied"),
        )
    return accepted


def _queue_isolated_observer_provisioning(executor: Executor, pending: _PendingLaunchHandoff) -> bool:
    """Queue observer work without any controller-side Project diagnostics."""
    handle = pending.handle
    attach = getattr(executor, "attach_observer", None)
    if handle is None or not callable(attach):
        return False
    key = (str(pending.cfg.shared_root), pending.task_id, pending.attempt_id)

    def attach_later() -> None:
        try:
            attach(pending.cfg, pending.task_id, pending.attempt_id, handle)
        except Exception:
            # Observer creation is optional. Its Project-side diagnostics must
            # never fall back into the machine controller thread.
            pass

    try:
        accepted = submit_observer_job(key, attach_later)
    except Exception:
        diagnostic_increment("scheduler.isolated.observer_queue_unavailable")
        return False
    if not accepted:
        diagnostic_increment("scheduler.isolated.observer_queue_busy")
    return accepted


class _LaunchHandoffBatch:
    """Track runner handoffs without blocking a machine dispatch cycle.

    The MachineRuntime retains pending handoffs across batches while runners
    publish durable intent. An overdue isolated handoff stays pending until
    intent is observed or reauthorization can safely proceed.
    """

    def __init__(self, executor: Executor, runtime: MachineRuntime | str | Path) -> None:
        self._executor = executor
        if not isinstance(runtime, (str, Path)) and hasattr(runtime, "pending_launch_handoffs"):
            self._runtime = runtime
            self._runtime_root = Path(runtime.root)
        else:
            # Keep the old construction seam useful for focused callers that
            # only exercise a batch in isolation.  Machine dispatch always
            # passes MachineRuntime, which is what provides cross-cycle state.
            self._runtime = None
            self._runtime_root = Path(runtime)
            self._pending: list[_PendingLaunchHandoff] = []

    def _pending_items(self) -> list[_PendingLaunchHandoff]:
        if self._runtime is None:
            return list(self._pending)
        return list(self._runtime.pending_launch_handoffs.values())

    def _remember(self, pending: _PendingLaunchHandoff) -> None:
        key = (pending.project_id, pending.attempt_id)
        if self.has_pending(*key):
            raise RuntimeError(f"launch handoff is already pending for {key!r}")
        if self._runtime is None:
            self._pending.append(pending)
        else:
            self._runtime.pending_launch_handoffs[key] = pending

    def has_pending(self, project_id: str, attempt_id: str) -> bool:
        """Return whether exact local launch ownership is already retained."""
        if self._runtime is None:
            return any(item.project_id == project_id and item.attempt_id == attempt_id for item in self._pending)
        return (project_id, attempt_id) in self._runtime.pending_launch_handoffs

    def _remove(self, pending: _PendingLaunchHandoff) -> None:
        key = (pending.project_id, pending.attempt_id)
        if self._runtime is None:
            self._pending = [item for item in self._pending if item is not pending]
        elif self._runtime.pending_launch_handoffs.get(key) is pending:
            del self._runtime.pending_launch_handoffs[key]

    def _replace(self, pending: _PendingLaunchHandoff) -> None:
        key = (pending.project_id, pending.attempt_id)
        if self._runtime is None:
            self._pending = [pending if (item.project_id, item.attempt_id) == key else item for item in self._pending]
        elif self._runtime.pending_launch_handoffs.get(key) is not None:
            self._runtime.pending_launch_handoffs[key] = pending

    def _defer_retry(self, pending: _PendingLaunchHandoff, now: float) -> None:
        failures = pending.retry_failures + 1
        delay = min(5.0, 0.1 * (2 ** min(failures - 1, 6)))
        self._replace(replace(pending, retry_not_before=now + delay, retry_failures=failures))

    def launch(self, cfg: RootConfig, task_id: str, attempt: Any, project_id: str) -> None:
        initiate = getattr(self._executor, "initiate_attempt", None)
        if initiate is None:
            self._executor.launch_attempt(cfg, task_id, attempt)
            return
        key = (project_id, attempt.attempt_id)
        if self.has_pending(*key):
            raise RuntimeError(f"launch handoff is already pending for {key!r}")
        with diagnostic_span("executor.launch.initiate"):
            handle, handoff = initiate(cfg, task_id, attempt)
        if not isinstance(handle, LaunchHandle):
            candidate = getattr(handoff, "handle", None)
            handle = candidate if isinstance(candidate, LaunchHandle) else None
        pending = _PendingLaunchHandoff(
            cfg,
            task_id,
            attempt.attempt_id,
            attempt.current_fencing_token,
            project_id,
            handoff,
            handle,
        )
        self._remember(pending)

    def launch_authorized(self, cfg: RootConfig, evidence: Mapping[str, Any], project_id: str) -> None:
        """Start one runner from exact isolated authorization evidence."""
        if not isinstance(evidence, Mapping) or set(evidence) != {
            "outcome",
            "claim_identity",
            "launch_id",
            "launch_handoff_timeout_seconds",
        }:
            raise ValueError("authorized launch evidence is malformed.")
        if evidence.get("outcome") != "authorized":
            raise ValueError("launch handoff requires an authorized completion.")
        claim_identity = evidence.get("claim_identity")
        claim_fields = {"task_id", "attempt_id", "attempt_number", "fencing_token", "reservation_id"}
        if not isinstance(claim_identity, Mapping) or set(claim_identity) != claim_fields:
            raise ValueError("authorized launch claim identity is malformed.")
        task_id = claim_identity.get("task_id")
        attempt_id = claim_identity.get("attempt_id")
        fencing_token = claim_identity.get("fencing_token")
        if not isinstance(project_id, str) or not project_id:
            raise ValueError("authorized launch project_id must be a nonempty string.")
        if not isinstance(task_id, str) or not task_id or not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError("authorized launch Task and Attempt identity is invalid.")
        if type(fencing_token) is not int or fencing_token < 1:
            raise ValueError("authorized launch fencing_token must be a positive integer.")

        key = (project_id, attempt_id)
        if self.has_pending(*key):
            raise RuntimeError(f"launch handoff is already pending for {key!r}")

        initiate = getattr(self._executor, "initiate_authorized_attempt", None)
        if not callable(initiate):
            raise RuntimeError("executor does not support authorized launch tickets.")
        try:
            with diagnostic_span("executor.launch.initiate"):
                handle, handoff = initiate(
                    cfg,
                    claim_identity=claim_identity,
                    launch_id=evidence["launch_id"],
                    launch_handoff_timeout_seconds=evidence["launch_handoff_timeout_seconds"],
                )
        except Exception as exc:
            handle = launch_failure_handle(exc)
            if handle is not None:
                timeout = float(evidence["launch_handoff_timeout_seconds"])
                intent_path = local_paths(cfg.runtime_root)["launch_intents"] / f"{attempt_id}.json"
                self._remember(
                    _PendingLaunchHandoff(
                        cfg,
                        task_id,
                        attempt_id,
                        fencing_token,
                        project_id,
                        LaunchHandoff(attempt_id, intent_path, time.monotonic() + timeout, handle),
                        handle,
                        reservation_id=claim_identity["reservation_id"],
                    )
                )
            raise
        if not isinstance(handle, LaunchHandle):
            candidate = getattr(handoff, "handle", None)
            handle = candidate if isinstance(candidate, LaunchHandle) else None
        self._remember(
            _PendingLaunchHandoff(
                cfg,
                task_id,
                attempt_id,
                fencing_token,
                project_id,
                handoff,
                handle,
                reservation_id=claim_identity["reservation_id"],
            )
        )

    def _retry_isolated_launch_after_exit(self, pending: _PendingLaunchHandoff) -> bool:
        """Forget an authorization only after its exact runner is proven absent."""
        if self._runtime is None or pending.reservation_id is None:
            return False
        controller = getattr(self._runtime, "project_io_controller", None)
        retry = getattr(controller, "retry_scheduler_launch_authorization", None)
        if not callable(retry):
            return False
        try:
            records = reservation_snapshot(self._runtime_root).active
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError):
            return False
        for record in records:
            if not isinstance(record, dict) or record.get("reservation_id") != pending.reservation_id:
                continue
            try:
                identity = ReservationIdentity.from_record(record)
            except (TypeError, ValueError):
                return False
            if (
                identity.project_id != pending.project_id
                or identity.task_id != pending.task_id
                or identity.attempt_id != pending.attempt_id
                or identity.fencing_token != pending.fencing_token
            ):
                return False
            retry(identity)
            self._remove(pending)
            return True
        return False

    def poll(
        self,
        result_by_project: dict[str, dict[str, Any]] | None = None,
        *,
        isolated: bool = False,
    ) -> None:
        """Poll pending handoffs once, never sleeping or waiting on a child."""
        pending_items = self._pending_items()
        if not pending_items:
            return
        for pending in pending_items:
            now = time.monotonic()
            if isolated:
                if now < pending.retry_not_before:
                    continue
                try:
                    is_published = _launch_intent_exists(pending.handoff.intent_path)
                except Exception:
                    self._defer_retry(pending, now)
                    continue
                if is_published:
                    _queue_isolated_observer_provisioning(self._executor, pending)
                    self._remove(pending)
                    continue
                try:
                    is_expired = now >= pending.handoff.deadline
                except Exception:
                    self._defer_retry(pending, now)
                    continue
                if not is_expired:
                    continue
                handle = pending.handle
                process = None
                if isinstance(handle, LaunchHandle):
                    process = handle.runner_process if handle.runner_process is not None else handle.reference
                poll = getattr(process, "poll", None)
                if not callable(poll):
                    self._defer_retry(pending, now)
                    continue
                try:
                    has_exited = poll() is not None
                except Exception:
                    has_exited = False
                if has_exited:
                    # A dead exact child cannot publish after this point. Check
                    # the intent once more to close the timeout observation race.
                    try:
                        is_published = _launch_intent_exists(pending.handoff.intent_path)
                    except Exception:
                        self._defer_retry(pending, now)
                        continue
                    if is_published:
                        _queue_isolated_observer_provisioning(self._executor, pending)
                        self._remove(pending)
                        continue
                    if self._retry_isolated_launch_after_exit(pending):
                        continue
                diagnostic_increment("scheduler.isolated.launch_handoff_overdue")
                self._defer_retry(pending, now)
                continue
            if now < pending.retry_not_before:
                continue
            if pending.compensation_committed:
                handle = pending.handle
                if handle is None:
                    self._remove(pending)
                    continue
                try:
                    self._executor.cleanup_launch(handle)
                except Exception as exc:
                    append_launch_failure_diagnostic(pending.cfg, pending.task_id, pending.attempt_id, exc)
                    self._defer_retry(pending, now)
                    continue
                self._remove(pending)
                self._record_result_failure(
                    pending,
                    ExecutorLaunchHandoffTimeout("launch cleanup completed after compensation"),
                    result_by_project,
                )
                continue
            try:
                published = pending.handoff.intent_path.exists()
            except Exception as exc:
                append_launch_failure_diagnostic(pending.cfg, pending.task_id, pending.attempt_id, exc)
                self._defer_retry(pending, now)
                continue
            if published:
                _queue_observer_provisioning(self._executor, pending)
                self._remove(pending)
                continue

            try:
                is_expired = now >= pending.handoff.deadline
            except Exception as exc:
                append_launch_failure_diagnostic(pending.cfg, pending.task_id, pending.attempt_id, exc)
                self._defer_retry(pending, now)
                continue
            if not is_expired:
                continue

            handle = pending.handle
            if not isinstance(handle, LaunchHandle):
                candidate = getattr(pending.handoff, "handle", None)
                handle = candidate if isinstance(candidate, LaunchHandle) else None
            if pending.handle is None and handle is not None:
                pending = replace(pending, handle=handle)
                self._replace(pending)
            timeout = ExecutorLaunchHandoffTimeout(
                f"runner did not publish launch intent for {pending.attempt_id!r}",
                handle=handle,
            )
            append_launch_failure_diagnostic(pending.cfg, pending.task_id, pending.attempt_id, timeout)
            try:
                did_fail = fail_attempt(
                    pending.cfg,
                    pending.task_id,
                    pending.attempt_id,
                    pending.fencing_token,
                    "executor_launch_handoff_timeout",
                    should_require_unstarted=True,
                    reservation_runtime_root=self._runtime_root,
                )
            except Exception as exc:
                append_launch_failure_diagnostic(pending.cfg, pending.task_id, pending.attempt_id, exc)
                self._defer_retry(pending, now)
                continue
            if not did_fail:
                # Evidence or authority changed after the timeout.  Leave the
                # live attempt to normal recovery and never kill its handle.
                self._remove(pending)
                continue
            if handle is None:
                self._remove(pending)
                self._record_result_failure(pending, timeout, result_by_project)
                continue
            try:
                self._executor.cleanup_launch(handle)
            except Exception as exc:
                # Compensation committed, but exact resource cleanup did not.
                # Retain the record and retry cleanup in a later cycle without
                # attempting fail_attempt a second time.
                append_launch_failure_diagnostic(pending.cfg, pending.task_id, pending.attempt_id, exc)
                pending = replace(pending, compensation_committed=True)
                self._replace(pending)
                self._defer_retry(pending, now)
                continue
            self._remove(pending)
            self._record_result_failure(pending, timeout, result_by_project)

    def _record_result_failure(
        self,
        pending: _PendingLaunchHandoff,
        failure: BaseException,
        result_by_project: dict[str, dict[str, Any]] | None,
    ) -> None:
        if result_by_project is None:
            return
        item = result_by_project.get(pending.project_id)
        if item is None:
            return
        launched = item.get("launched", [])
        if pending.task_id in launched:
            launched.remove(pending.task_id)
        item["status"] = "error"
        item["error"] = str(failure)

    def finish(
        self,
        result_by_project: dict[str, dict[str, Any]] | None = None,
        *,
        isolated: bool = False,
    ) -> None:
        """Compatibility alias for one non-blocking poll."""
        self.poll(result_by_project, isolated=isolated)
