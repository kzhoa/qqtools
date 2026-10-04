"""Machine scheduling and admission."""

from __future__ import annotations

import os
import shlex
import signal
import threading
import time
import uuid
from collections.abc import Collection, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, ContextManager

from ..authority import AuthoritySupervisor
from ..config_types import RootConfig
from ..executor import (
    Executor,
    ExecutorLaunchHandoffTimeout,
    LaunchHandle,
    LaunchHandoff,
    append_launch_failure_diagnostic,
    launch_failure_handle,
)
from ..gpu_policy import (
    GpuDiscovery,
    GpuReservationPolicy,
    parse_environment_gpu_ids,
    persist_gpu_policy_observation,
    resolve_gpu_policy,
)
from ..layout import load_machine_record, load_root_config, machine_state_path, runtime_pid_path
from ..legacy_agent import get_agent_status
from ..machine_config import is_legacy_agent_project, load_machine_policy, save_machine_config
from ..machine_dispatch_plan import (
    MachineDispatchSnapshot,
    build_machine_dispatch_plan,
    order_dispatch_project_ids,
    reduce_dispatch_cursor,
)
from ..machine_state import publish_machine_snapshots, publish_machine_stop_snapshot
from ..observer_provisioning import submit_observer_job
from ..project_maintenance import maintain_project, reconcile_reservation
from ..runtime.locks import exclusive
from ..runtime.paths import local_paths, machine_project_paths, shared_paths
from ..runtime.ready import advance_ready_index_build, peek_ready_marker, read_ready_index_state
from ..runtime.records import TaskSpec, utc_now
from ..runtime.resources.cpu_lane import cpu_reservation_snapshot
from ..runtime.resources.reservations import (
    ReservationIdentity,
    ReservationSnapshot,
    reconcile_snapshot,
    reservation_snapshot,
)
from ..runtime.responsibility_capture import CaptureBusy
from ..runtime.store import atomic_replace, iter_json, read_json
from ..runtime.upgrade.machine import MachineUpgradeWorker, discover_registered_upgrades, inspect_registered_upgrades
from ..runtime.work_budget import (
    DIAGNOSTIC_PUBLISH_INTERVAL_SECONDS,
    AdaptiveBatchSizer,
    RuntimeDiagnostics,
    SliceBudget,
    WorkBudgetPolicy,
    activate_diagnostics,
    diagnostic_increment,
    diagnostic_span,
)
from ..scheduler import (
    _BorrowAdmissionGrant,
    _BorrowAdmissionRevision,
    fail_attempt,
    resume_starting_attempt,
    run_dispatch_cycle,
)
from . import helpers as _helpers
from .context import MachineRuntime, ProjectBinding, default_machine_runtime_root
from .helpers import (
    _publish_project_snapshots,
    _read_pid,
    _reconcile_machine_reservations,
    _record_bad_task_spec,
    _recover_starting_reservations,
    _working_directory_reason,
)
from .inventory import InventoryReconciliation, exact_binding_for_entry
from .primary_demand import PrimaryDemandProbe, probe_primary_demand
from .project_io_controller import ProjectIOController
from .project_io_executor import ProjectIOExecutor
from .project_io_supervision import AttemptSupervisionCoordinator
from .scheduler_diagnostics import SchedulerDiagnosticStore

_SCHEDULER_DIAGNOSTIC_PUBLISH_INTERVAL_NS = 5_000_000_000
_PRIMARY_PROBE_FAULT_REASONS = frozenset(
    {
        "registry_unreadable",
        "project_not_probeable",
        "ready_index_unreadable",
        "ready_index_unresolved",
        "group_ready_members_unresolved",
        "index_unreadable",
        "marker_unreadable",
        "task_truth_unreadable",
        "group_unreadable",
        "task_invalid",
        "route_mismatch",
        "marker_invalid",
        "marker_identity_invalid",
        "submission_identity_missing",
        "submission_missing",
        "submission_invalid",
        "submission_state_invalid",
        "group_missing",
        "group_invalid",
        "dependency_invalid",
    }
)


def _enablement_reconciliation_probe(
    runtime: MachineRuntime,
    reconciliation: InventoryReconciliation | None,
) -> dict[str, object]:
    """Translate one exact registry/inventory observation into derived evidence."""
    runtime_id = runtime.instance_id
    base_identity: dict[str, object] = {
        "producer": "enablement_reconciliation",
        "component": "scheduler",
        "stage": "reconciliation",
        "check": "enablement",
        "runtime_id": runtime_id,
    }
    if reconciliation is None:
        identity = {
            **base_identity,
            "reason_code": "registry_enablement_unknown",
            "scope_type": "machine",
        }
        return {
            "identity": identity,
            "outcome": "blocked",
            "coverage": "unknown",
            "source_revision": {},
            "details": {"progress": "blocked"},
            "findings": (
                {
                    "identity": identity,
                    "severity": "fault",
                    "source_revision": {},
                    "details": {"progress": "blocked"},
                },
            ),
            "covered_bindings": [],
        }

    source_revision = {
        "registry_revision": reconciliation.registry_revision,
        "inventory_revision": reconciliation.inventory_revision,
    }
    findings: list[dict[str, object]] = []
    mirror_incomplete = "inventory_mirror_incomplete" in reconciliation.blockers
    mirror_reason = "inventory_mirror_incomplete" if mirror_incomplete else "enablement_mirror_diverged"
    entries_by_exact_binding = {(entry.project_id, entry.shared_root): entry for entry in reconciliation.entries}
    for binding in reconciliation.bindings:
        entry = entries_by_exact_binding.get((binding.project_id, binding.shared_root))
        _exact_binding, conflict = (
            exact_binding_for_entry(entry, reconciliation.bindings) if entry is not None else (None, True)
        )
        if entry is not None and not conflict and entry.enabled == binding.enabled:
            continue
        identity: dict[str, object] = {
            **base_identity,
            "reason_code": mirror_reason,
            "scope_type": "project",
            "project_id": binding.project_id,
            "registry_revision": reconciliation.registry_revision,
        }
        if binding.registration_generation is not None:
            identity["registration_generation"] = binding.registration_generation
        findings.append(
            {
                "identity": identity,
                "severity": "fault",
                "source_revision": source_revision,
                "details": {
                    "progress": ("mirror_write_pending" if mirror_incomplete and entry is not None else "blocked")
                },
            }
        )
    if not reconciliation.converged and not findings:
        findings.append(
            {
                "identity": {
                    **base_identity,
                    "reason_code": mirror_reason,
                    "scope_type": "machine",
                    "registry_revision": reconciliation.registry_revision,
                },
                "severity": "fault",
                "source_revision": source_revision,
                "details": {"progress": "blocked"},
            }
        )
    decision_identity: dict[str, object] = {
        **base_identity,
        "reason_code": "enablement_converged" if reconciliation.converged else mirror_reason,
        "scope_type": "machine",
        "registry_revision": reconciliation.registry_revision,
    }
    return {
        "identity": decision_identity,
        "outcome": "converged" if reconciliation.converged else "blocked",
        "coverage": "complete",
        "source_revision": source_revision,
        "details": {"progress": "converged" if reconciliation.converged else "blocked"},
        "findings": findings,
        "covered_bindings": [
            {
                "project_id": binding.project_id,
                "registration_generation": binding.registration_generation,
            }
            for binding in reconciliation.bindings
        ],
    }


def _scheduler_diagnostic_probe(
    runtime: MachineRuntime,
    *,
    lane: str,
    probe: PrimaryDemandProbe,
    registry_revision: int,
    bindings: dict[str, ProjectBinding],
    covered_project_ids: set[str],
    available_capacity: int,
    borrow_denied: bool = False,
) -> dict[str, object]:
    """Translate a completed primary probe without carrying rendered errors."""
    runtime_id = runtime.instance_id
    source_revision = {"registry_revision": registry_revision}
    findings: list[dict[str, object]] = []
    blockers: list[dict[str, object]] = []
    for diagnostic in probe.diagnostics:
        reason = diagnostic.get("reason")
        if not isinstance(reason, str) or reason not in _PRIMARY_PROBE_FAULT_REASONS:
            continue
        project_id = diagnostic.get("project_id")
        binding = bindings.get(project_id) if isinstance(project_id, str) else None
        identity: dict[str, object] = {
            "producer": "primary_probe",
            "reason_code": reason,
            "component": "scheduler",
            "stage": "admission",
            "check": "primary_demand",
            "scope_type": "project_route" if project_id is not None else "machine_lane",
            "runtime_id": runtime_id,
            "resource_lane": lane,
        }
        if isinstance(project_id, str):
            identity["project_id"] = project_id
        if binding is not None and binding.registration_generation is not None:
            identity["registration_generation"] = binding.registration_generation
        task_id = diagnostic.get("task_id")
        if isinstance(task_id, str):
            identity["task_id"] = task_id
        task_generation = diagnostic.get("task_generation")
        if type(task_generation) is int:
            identity["task_generation"] = task_generation
        route_scope = diagnostic.get("route_scope")
        if isinstance(route_scope, str):
            identity["route_scope"] = route_scope
        details: dict[str, object] = {}
        exception_type = diagnostic.get("exception_type")
        if isinstance(exception_type, str):
            details["exception_type"] = exception_type
        findings.append(
            {
                "identity": identity,
                "severity": "fault",
                "source_revision": source_revision,
                "details": details,
            }
        )
        blockers.append(identity)
    decision_identity: dict[str, object] = {
        "producer": "primary_probe",
        "reason_code": "borrow_blocked_unresolved_primary" if borrow_denied else "primary_demand",
        "component": "scheduler",
        "stage": "admission",
        "check": "primary_demand",
        "scope_type": "machine_lane",
        "runtime_id": runtime_id,
        "resource_lane": lane,
    }
    decision_project_id = next(
        (
            item.get("project_id")
            for item in probe.diagnostics
            if isinstance(item.get("project_id"), str) and item.get("project_id") in bindings
        ),
        None,
    )
    if borrow_denied and decision_project_id is None:
        decision_project_id = next(
            (project_id for project_id, binding in sorted(bindings.items()) if binding.enabled),
            None,
        )
    if isinstance(decision_project_id, str):
        binding = bindings[decision_project_id]
        decision_identity["scope_type"] = "project_route"
        decision_identity["project_id"] = decision_project_id
        if binding.registration_generation is not None:
            decision_identity["registration_generation"] = binding.registration_generation
        route_scope = next(
            (
                item.get("route_scope")
                for item in probe.diagnostics
                if item.get("project_id") == decision_project_id and isinstance(item.get("route_scope"), str)
            ),
            None,
        )
        if route_scope in {"home", "shared"}:
            route = runtime.primary_probe.route((decision_project_id, route_scope, lane))
        else:
            route = next(
                (
                    runtime.primary_probe.route((decision_project_id, scope, lane))
                    for scope in ("shared", "home")
                    if runtime.primary_probe.route((decision_project_id, scope, lane)).revision is not None
                ),
                None,
            )
        if route is not None and route.revision is not None:
            decision_identity["ready_revision"] = route.revision
            source_revision["ready_revision"] = route.revision
    if borrow_denied:
        decision_identity["registry_revision"] = registry_revision
    return {
        "identity": decision_identity,
        "outcome": "blocked" if borrow_denied else probe.state,
        "coverage": "complete" if probe.state == "no_primary_demand" else "incomplete",
        "capacity_context": {"available": available_capacity},
        "blocker_identities": blockers,
        "source_revision": source_revision,
        "details": {"probe_state": probe.state},
        "findings": findings,
        "covered_bindings": [
            {
                "project_id": project_id,
                "registration_generation": bindings[project_id].registration_generation,
            }
            for project_id in sorted(covered_project_ids)
            if project_id in bindings
        ],
    }


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

    A runner can be alive and authorized while its launch-intent publication is
    delayed by local startup work.  The reservation is deliberately retained
    during that interval; the machine agent polls the durable intent once per
    cycle and only compensates after the bounded deadline expires.
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

    @property
    def _pending_store(self) -> dict[tuple[str, str], _PendingLaunchHandoff]:
        if self._runtime is None:
            return {}
        return self._runtime.pending_launch_handoffs

    def _pending_items(self) -> list[_PendingLaunchHandoff]:
        """Return persistent and compatibility-local records exactly once."""
        items: list[_PendingLaunchHandoff] = []
        seen: set[tuple[str, str]] = set()
        for pending in (*self._pending_store.values(), *self._pending):
            key = (pending.project_id, pending.attempt_id)
            if key in seen:
                continue
            seen.add(key)
            items.append(pending)
        return items

    def _remember(self, pending: _PendingLaunchHandoff) -> None:
        key = (pending.project_id, pending.attempt_id)
        if key in self._pending_store or any(
            item.project_id == pending.project_id and item.attempt_id == pending.attempt_id for item in self._pending
        ):
            raise RuntimeError(f"launch handoff is already pending for {key!r}")
        if self._runtime is not None:
            self._runtime.pending_launch_handoffs[key] = pending
        self._pending.append(pending)

    def has_pending(self, project_id: str, attempt_id: str) -> bool:
        """Return whether exact local launch ownership is already retained."""
        key = (project_id, attempt_id)
        return key in self._pending_store or any(
            item.project_id == project_id and item.attempt_id == attempt_id for item in self._pending
        )

    def _remove(self, pending: _PendingLaunchHandoff) -> None:
        key = (pending.project_id, pending.attempt_id)
        if self._runtime is not None and self._runtime.pending_launch_handoffs.get(key) is pending:
            del self._runtime.pending_launch_handoffs[key]
        self._pending = [item for item in self._pending if item is not pending]

    def _replace(self, pending: _PendingLaunchHandoff) -> None:
        key = (pending.project_id, pending.attempt_id)
        if self._runtime is not None and self._runtime.pending_launch_handoffs.get(key) is not None:
            self._runtime.pending_launch_handoffs[key] = pending
        self._pending = [pending if (item.project_id, item.attempt_id) == key else item for item in self._pending]

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
        if key in self._pending_store or any(
            item.project_id == project_id and item.attempt_id == attempt.attempt_id for item in self._pending
        ):
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
        if key in self._pending_store or any(
            item.project_id == project_id and item.attempt_id == attempt_id for item in self._pending
        ):
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
                    published = pending.handoff.intent_path.exists()
                except Exception:
                    continue
                if published:
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
                        if pending.handoff.intent_path.exists():
                            _queue_isolated_observer_provisioning(self._executor, pending)
                            self._remove(pending)
                            continue
                    except Exception:
                        pass
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


@dataclass(frozen=True, slots=True)
class _ProjectLaunchExecutor:
    batch: _LaunchHandoffBatch
    project_id: str

    def launch_attempt(self, cfg: RootConfig, task_id: str, attempt: Any) -> None:
        self.batch.launch(cfg, task_id, attempt, self.project_id)


def _probe_primary_demand(
    runtime: MachineRuntime,
    readable: dict[str, RootConfig],
    dispatchable: dict[str, RootConfig],
    visible: list[int],
    free: list[int],
    budget: SliceBudget,
    reservations: tuple[dict[str, Any], ...] = (),
    *,
    lane: str = "gpu",
    excluded_project_ids: set[str] | frozenset[str] = frozenset(),
) -> PrimaryDemandProbe:
    """Scan primary candidates through an independent bounded ready cursor."""
    try:
        _, bindings = runtime.load_registry()
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        return PrimaryDemandProbe(
            "unresolved",
            ({"reason": "registry_unreadable", "exception_type": type(exc).__name__},),
        )
    enabled_ids = {binding.project_id for binding in bindings if binding.enabled}
    return probe_primary_demand(
        runtime.primary_probe,
        enabled_ids,
        readable,
        dispatchable,
        visible,
        free,
        budget,
        reservations,
        lane=lane,
        excluded_project_ids=excluded_project_ids,
    )


def _authority_recovery_has_possible_demand(readable: dict[str, RootConfig], project_ids: set[str]) -> bool:
    """Fail closed unless every recovering project's ready routes are empty."""

    # This one-shot idle proof may inspect both routes for several recovering
    # projects. Keep the fixed operation cap while allowing loaded machines a
    # full second to prove emptiness; timeout remains fail closed.
    budget = SliceBudget(WorkBudgetPolicy(soft_deadline_ms=1000))
    for project_id in sorted(project_ids):
        cfg = readable.get(project_id)
        if cfg is None:
            return True
        try:
            if read_ready_index_state(cfg) != "active":
                return True
            for scope in ("shared", "home"):
                peek = peek_ready_marker(cfg, project_id, scope, None, budget)
                if peek.reference is not None or peek.unresolved or peek.exhausted or not peek.wrapped:
                    return True
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            return True
    return False


def _build_borrow_admission_grant(
    runtime: MachineRuntime,
    dispatchable: dict[str, RootConfig],
    probe: PrimaryDemandProbe,
    enabled_ids: set[str],
    *,
    lane: str = "gpu",
) -> _BorrowAdmissionGrant | None:
    """Create a grant only after every enabled project's primary probe completed."""
    if probe.state != "no_primary_demand":
        return None
    revisions: list[_BorrowAdmissionRevision] = []
    route_keys = [(project_id, scope, lane) for project_id in sorted(enabled_ids) for scope in ("shared", "home")]
    revision_snapshot = runtime.primary_probe.completed_revisions(lane, route_keys)
    if revision_snapshot is None:
        return None
    for project_id in sorted(enabled_ids):
        cfg = dispatchable.get(project_id)
        if cfg is None:
            return None
        for scope in ("shared", "home"):
            cursor_key = (project_id, scope, lane)
            revision = revision_snapshot[cursor_key]
            revisions.append(_BorrowAdmissionRevision(project_id, cfg, scope, revision))
    grant = _BorrowAdmissionGrant(runtime.root, tuple(revisions), lane)
    if not grant.is_valid(runtime.root):
        # A route checked in an earlier slice may have changed.  Reject the
        # observation here instead of passing a known-stale grant to a claim.
        runtime.primary_probe.invalidate_routes(route_keys)
        return None
    return grant


def _observe_gpu_policy(
    runtime: MachineRuntime,
    *,
    available_gpus: list[int] | None,
    instance_id: str,
    reserved_gpu_ids: set[int] | frozenset[int] = frozenset(),
) -> tuple[GpuReservationPolicy, Any]:
    """Obtain one bounded raw observation and persist its effective policy view."""
    if available_gpus is not None:
        injected = tuple(sorted(set(available_gpus)))
        discovery = GpuDiscovery(injected, "empty" if not injected else "available", "injected")
    else:
        from ..gpu_policy import discover_gpu_inventory

        discovery = discover_gpu_inventory()
    environment_gpu_ids, environment_status = parse_environment_gpu_ids()
    reservation_policy = GpuReservationPolicy(discovery, environment_gpu_ids, environment_status)
    policy_view = resolve_gpu_policy(
        runtime.root,
        discovery=discovery,
        environment_gpu_ids=environment_gpu_ids,
        environment_status=environment_status,
    ).with_reservations(reserved_gpu_ids)
    try:
        persist_gpu_policy_observation(
            runtime,
            instance_id=instance_id,
            pid=_read_pid(runtime),
            view=policy_view,
            policy=reservation_policy,
        )
    except (OSError, RuntimeError, ValueError, TypeError):
        diagnostic_increment("gpu_policy.observation_unavailable")
    return reservation_policy, policy_view


def dispatch_machine_cycle_locked(
    runtime: MachineRuntime,
    *,
    available_gpus: list[int] | None = None,
    executor: Executor | None = None,
    instance_id: str = "machine-agent",
    heartbeat_interval_seconds: float = 5.0,
    started_at: str | None = None,
    supervisors: dict[str, AuthoritySupervisor] | None = None,
    supervise: bool = True,
    publish_snapshots: bool = True,
) -> list[dict[str, Any]]:
    """Supervise bindings and fill capacity in fair one-claim-per-project rounds."""
    project_io_controller = _project_io_controller(runtime)
    diagnostics = RuntimeDiagnostics()
    with activate_diagnostics(diagnostics), diagnostic_span("dispatch_machine_cycle"):
        with project_io_controller.admission_turn() if project_io_controller is not None else nullcontext():
            results = _dispatch_machine_cycle_locked(
                runtime,
                available_gpus=available_gpus,
                executor=executor,
                instance_id=instance_id,
                heartbeat_interval_seconds=heartbeat_interval_seconds,
                started_at=started_at,
                supervisors=supervisors,
                supervise=supervise,
                publish_snapshots=publish_snapshots,
            )
    if project_io_controller is not None and project_io_controller.has_pending_admission:
        runtime.last_cycle_had_demand = True
    now_ns = time.monotonic_ns()
    should_publish_scheduler_diagnostics = (
        runtime.last_scheduler_diagnostics_publish_ns is None
        or now_ns - runtime.last_scheduler_diagnostics_publish_ns >= _SCHEDULER_DIAGNOSTIC_PUBLISH_INTERVAL_NS
    )
    if should_publish_scheduler_diagnostics:
        try:
            store = SchedulerDiagnosticStore(runtime)
            store.recover()
            store.publish_cycle(
                counters=diagnostics.snapshot(),
                timings=None,
                probes=runtime.last_scheduler_diagnostic_probes,
                working_set=runtime.working_set.snapshot(),
            )
            store.reconcile_cycle(runtime.last_scheduler_diagnostic_probes)
            runtime.last_scheduler_diagnostics_publish_ns = now_ns
        except (OSError, RuntimeError, TypeError, ValueError):
            pass
    publish_interval_ns = DIAGNOSTIC_PUBLISH_INTERVAL_SECONDS * 1_000_000_000
    should_publish_diagnostic = (
        runtime.last_diagnostic_publish_ns is None or now_ns - runtime.last_diagnostic_publish_ns >= publish_interval_ns
    )
    if not should_publish_diagnostic:
        return results
    try:
        atomic_replace(
            runtime.paths["diagnostics"] / "scheduler-cycle.json",
            {
                "scheduler_diagnostic": {
                    "recorded_at": utc_now(),
                    **diagnostics.snapshot(),
                }
            },
        )
        runtime.last_diagnostic_publish_ns = now_ns
    except OSError:
        pass
    return results


def _dispatch_admission_layer(
    runtime: MachineRuntime,
    ordered_enabled: list[ProjectBinding],
    dispatchable: dict[str, RootConfig],
    executor: Executor,
    free: list[int],
    free_cpu_slots: int,
    visible: list[int],
    result_by_project: dict[str, dict[str, Any]],
    batch_sizers: dict[str, AdaptiveBatchSizer],
    budget: SliceBudget,
    launch_batch: _LaunchHandoffBatch,
    *,
    admission_role: str,
    borrow_admission_grant: _BorrowAdmissionGrant | None = None,
    lane: str,
    admission_blocked_project_ids: set[str] | frozenset[str] = frozenset(),
    gpu_policy: GpuReservationPolicy | None = None,
) -> tuple[list[int], int, ProjectBinding | None]:
    """Run one fair admission layer and return remaining GPUs and last winner."""
    inspected_ready: set[tuple[str, str, str, str]] = set()
    round_bindings = list(ordered_enabled)
    last_successful_binding: ProjectBinding | None = None
    has_retried_empty_round = False
    while (free or free_cpu_slots) and round_bindings:
        if not budget.can_start_record():
            diagnostic_increment("scheduler.work.immediate_slices")
            time.sleep(0)
            budget = SliceBudget(budget.policy)
        round_claims = 0
        records_before_round = budget.records_used
        round_last_success: ProjectBinding | None = None
        for binding in round_bindings:
            if binding.project_id in admission_blocked_project_ids:
                continue
            cfg = dispatchable.get(binding.project_id)
            if cfg is None or (not free and not free_cpu_slots) or not budget.can_start_record():
                continue
            claimed_task_ids: list[str] = []
            try:
                with diagnostic_span("scheduler.work"):
                    launched = run_dispatch_cycle(
                        cfg,
                        available_gpus=free,
                        available_cpus=free_cpu_slots,
                        executor=_ProjectLaunchExecutor(launch_batch, binding.project_id),
                        reservation_runtime_root=runtime.root,
                        project_id=binding.project_id,
                        admission_role=admission_role,
                        borrow_admission_grant=borrow_admission_grant,
                        ready_cursor_namespace=f"{binding.project_id}.{admission_role}",
                        preflight=lambda spec: _working_directory_reason(spec) is None,
                        preflight_rejected=lambda task: _record_bad_task_spec(
                            runtime, binding, task.task_id, task.spec
                        ),
                        claim_guard=lambda: runtime.enabled_claim_guard(binding),
                        max_new_claims=1,
                        on_claim=claimed_task_ids.append,
                        # Starting attempts are reconciled once for all bindings before
                        # admission layers run.  Repeating recovery here would relaunch
                        # an attempt for every role/lane pass when an executor has not
                        # yet published local launch evidence.
                        should_recover_starting=False,
                        work_budget=budget,
                        batch_sizer=batch_sizers[binding.project_id],
                        inspected_ready=inspected_ready,
                        lane=lane,
                        gpu_policy=gpu_policy,
                    )
                result_by_project[binding.project_id]["launched"].extend(launched)
                if claimed_task_ids:
                    round_claims += 1
                    round_last_success = binding
                    last_successful_binding = binding
                    snapshot = reconcile_snapshot(runtime.root)
                    free = [gpu_id for gpu_id in visible if gpu_id not in snapshot.reserved_gpu_ids]
                    policy, cpu_reservations = cpu_reservation_snapshot(runtime.root)
                    free_cpu_slots = policy.capacity - sum(item.get("cpu_slots", 0) for item in cpu_reservations)
            except (OSError, RuntimeError, ValueError) as exc:
                item = result_by_project[binding.project_id]
                item["status"] = "error"
                item["error"] = str(exc)
        if (
            round_claims == 0
            and not budget.can_start_record()
            and budget.records_used == records_before_round
            and not has_retried_empty_round
        ):
            diagnostic_increment("scheduler.work.immediate_slices")
            time.sleep(0)
            budget = SliceBudget(budget.policy)
            round_bindings = round_bindings[1:] + round_bindings[:1]
            has_retried_empty_round = True
            continue
        if round_claims == 0:
            break
        diagnostic_increment("scheduler.work.rounds")
        if round_last_success is not None:
            index = round_bindings.index(round_last_success)
            round_bindings = round_bindings[index + 1 :] + round_bindings[: index + 1]
    return free, free_cpu_slots, last_successful_binding


def _project_io_controller(runtime: MachineRuntime) -> ProjectIOController | None:
    executor = getattr(runtime, "project_io_executor", None)
    if executor is None:
        return None
    if not isinstance(executor, ProjectIOExecutor):
        raise TypeError("runtime.project_io_executor must be a ProjectIOExecutor.")
    controller = getattr(runtime, "project_io_controller", None)
    if controller is None:
        controller = ProjectIOController(runtime, executor)
        runtime.project_io_controller = controller
    elif not isinstance(controller, ProjectIOController):
        raise TypeError("runtime.project_io_controller must be a ProjectIOController.")
    elif controller.runtime is not runtime or controller.executor is not executor:
        raise RuntimeError("runtime.project_io_controller belongs to a different runtime or executor.")

    coordinator = getattr(runtime, "attempt_supervision_coordinator", None)
    if coordinator is None:
        runtime.attempt_supervision_coordinator = AttemptSupervisionCoordinator(runtime, controller)
    elif not isinstance(coordinator, AttemptSupervisionCoordinator):
        raise TypeError("runtime.attempt_supervision_coordinator must be an AttemptSupervisionCoordinator.")
    elif coordinator.runtime is not runtime or coordinator.controller is not controller:
        raise RuntimeError("runtime.attempt_supervision_coordinator has a different owner.")
    return controller


def _advance_machine_snapshot_publications(
    runtime: MachineRuntime,
    controller: ProjectIOController,
    validated_project_ids: Collection[str],
) -> None:
    """Advance one newest heartbeat snapshot intent on the dispatch thread."""
    intent = runtime.machine_snapshot_intent(validated_project_ids=tuple(validated_project_ids))
    if intent is None:
        return
    try:
        controller.advance_machine_snapshot_publications(
            intent.bindings,
            intent.registry_revision,
            instance_id=intent.instance_id,
            pid=intent.pid,
            visible_gpu_ids=intent.visible_gpu_ids,
            reservations=intent.reservations,
            heartbeat_interval_seconds=intent.heartbeat_interval_seconds,
            started_at=intent.started_at,
            gpu_policy=intent.gpu_policy,
        )
    except (OSError, RuntimeError, ValueError):
        # A later heartbeat replaces this intent and retries current state.
        return


def _advance_registration_renewals(
    runtime: MachineRuntime,
    controller: ProjectIOController,
    registry_revision: int | None = None,
    registered_bindings: Sequence[ProjectBinding] | None = None,
) -> None:
    """Advance the newest heartbeat renewal intent on the dispatch thread."""
    intent = runtime.registration_renewal_intent()
    if intent is not None and registry_revision is not None and intent.registry_revision != registry_revision:
        runtime.complete_registration_renewal_intent(intent)
        intent = None
    current_bindings = (
        tuple(registered_bindings) if registered_bindings is not None else (() if intent is None else intent.bindings)
    )
    binding_by_owner = {
        (binding.project_id, binding.registration_generation): binding
        for binding in current_bindings
        if binding.enabled
    }
    service_bindings = list(intent.bindings) if intent is not None else []
    executor = getattr(controller, "executor", None)
    try:
        unresolved_before = executor.unresolved_requests() if executor is not None else ()
    except (OSError, RuntimeError, ValueError):
        unresolved_before = ()
    renewal_requests = [request for request in unresolved_before if request.operation_kind == "registration_renew"]
    for request in unresolved_before:
        if request.operation_kind != "registration_renew" or (
            registry_revision is not None and request.registry_revision != registry_revision
        ):
            continue
        binding = binding_by_owner.get((request.project_id, request.registration_generation))
        if binding is not None and binding not in service_bindings:
            service_bindings.append(binding)
    if not service_bindings and not renewal_requests:
        return
    if intent is not None:
        renewal_horizon_seconds = intent.renewal_horizon_seconds
        service_revision = intent.registry_revision
    else:
        matching_requests = [
            request
            for request in renewal_requests
            if registry_revision is None or request.registry_revision == registry_revision
        ]
        horizon_requests = matching_requests or renewal_requests
        renewal_horizon_seconds = max(request.parameters["renewal_horizon_seconds"] for request in horizon_requests)
        service_revision = registry_revision if registry_revision is not None else horizon_requests[0].registry_revision
    try:
        completions = controller.advance_registration_renewals(
            tuple(service_bindings),
            service_revision,
            renewal_horizon_seconds=renewal_horizon_seconds,
        )
    except (OSError, RuntimeError, ValueError):
        # Retain the latest intent so a later dispatch cycle can retry it.
        return
    if intent is None:
        return
    accepted_project_ids = set(completions)
    try:
        unresolved_after = executor.unresolved_requests() if executor is not None else ()
    except (OSError, RuntimeError, ValueError):
        unresolved_after = ()
    intent_owners = {(binding.project_id, binding.registration_generation) for binding in intent.bindings}
    unresolved_after_owners = {
        (request.project_id, request.registration_generation)
        for request in unresolved_after
        if request.operation_kind == "registration_renew"
    }
    occupied_after_owners = {
        (request.project_id, request.registration_generation)
        for request in unresolved_after
        if request.registry_revision == intent.registry_revision
    }
    accepted_project_ids.update(
        request.project_id
        for request in unresolved_after
        if request.operation_kind == "registration_renew"
        and request.registry_revision == intent.registry_revision
        and (request.project_id, request.registration_generation) in intent_owners
    )
    accepted_project_ids.update(
        request.project_id
        for request in unresolved_before
        if request.operation_kind == "registration_renew"
        and (request.project_id, request.registration_generation) in intent_owners
        and (request.project_id, request.registration_generation) not in unresolved_after_owners
    )
    # Executor serialization prevents a renewal from being prepared while the
    # same exact binding owns any other request. Retire that transient renewal
    # hint from this batch so one hung claim cannot pin the other 63 renewals;
    # the fair working-set cursor will offer the occupied owner again later.
    accepted_project_ids.update(
        project_id for project_id, generation in intent_owners if (project_id, generation) in occupied_after_owners
    )
    accepted, remaining = runtime.advance_registration_renewal_intent(intent, accepted_project_ids)
    if not accepted or remaining is not None:
        return
    if intent.bindings:
        next_bindings, next_horizon = runtime.working_set.select_registration_renewals(
            limit=64,
            heartbeat_interval_seconds=intent.heartbeat_interval_seconds,
        )
        if next_bindings:
            runtime.publish_registration_renewal_intent(
                bindings=next_bindings,
                registry_revision=intent.registry_revision,
                renewal_horizon_seconds=next_horizon,
                heartbeat_interval_seconds=intent.heartbeat_interval_seconds,
            )


def _advance_activation_working_set(
    runtime: MachineRuntime,
    controller: ProjectIOController,
    registry_revision: int,
    registered_bindings: Sequence[ProjectBinding],
) -> None:
    """Advance activation consumer state without coordinator-side Project I/O."""
    working_set = runtime.working_set
    current_by_owner = {
        (binding.project_id, binding.registration_generation): binding for binding in registered_bindings
    }
    try:
        unresolved = controller.executor.unresolved_requests()
    except (OSError, RuntimeError, ValueError):
        unresolved = ()

    retirement_bindings: list[ProjectBinding] = []
    retirement_owners: set[tuple[str, str | None]] = set()
    retirement_projects: set[str] = set()
    for request in unresolved:
        if request.operation_kind != "activation_consumer_retire":
            continue
        binding = runtime.activation_consumer_retirements.select_exact(
            runtime_id=request.runtime_id,
            project_id=request.project_id,
            registration_generation=request.registration_generation,
            shared_root=Path(request.canonical_shared_root),
            machine_name=request.parameters["machine_name"],
            snapshot=registered_bindings,
        )
        if binding is None:
            continue
        owner = (binding.project_id, binding.registration_generation)
        if owner not in retirement_owners and binding.project_id not in retirement_projects:
            retirement_bindings.append(binding)
            retirement_owners.add(owner)
            retirement_projects.add(binding.project_id)
    for binding in runtime.activation_consumer_retirements.select_pending(
        registered_bindings,
        limit=max(1, 64 - len(retirement_bindings)),
    ):
        owner = (binding.project_id, binding.registration_generation)
        if (
            owner not in retirement_owners
            and binding.project_id not in retirement_projects
            and len(retirement_bindings) < 64
        ):
            retirement_bindings.append(binding)
            retirement_owners.add(owner)
            retirement_projects.add(binding.project_id)
    if retirement_bindings or any(request.operation_kind == "activation_consumer_retire" for request in unresolved):
        try:
            completions = controller.advance_activation_consumer_retirements(retirement_bindings, registry_revision)
        except (OSError, RuntimeError, ValueError):
            completions = {}
        runtime.activation_consumer_retirements.apply_completions(retirement_bindings, completions)

    registration_bindings: list[ProjectBinding] = []
    registration_owners: set[tuple[str, str | None]] = set()
    for request in unresolved:
        if request.operation_kind != "activation_consumer_register":
            continue
        binding = current_by_owner.get((request.project_id, request.registration_generation))
        if binding is not None and (binding.project_id, binding.registration_generation) not in registration_owners:
            registration_bindings.append(binding)
            registration_owners.add((binding.project_id, binding.registration_generation))
    for binding in working_set.select_activation_registrations(limit=64 - len(registration_bindings)):
        owner = (binding.project_id, binding.registration_generation)
        if owner not in registration_owners:
            registration_bindings.append(binding)
            registration_owners.add(owner)
    if registration_bindings or any(request.operation_kind == "activation_consumer_register" for request in unresolved):
        try:
            completions = controller.advance_activation_consumer_registrations(
                registration_bindings,
                registry_revision,
                process_fence=working_set.process_fence,
            )
        except (OSError, RuntimeError, ValueError):
            completions = {}
        working_set.apply_activation_registrations(registration_bindings, completions)

    observation_intents = []
    observation_owners: set[tuple[str, str | None]] = set()
    for request in unresolved:
        if request.operation_kind != "activation_observe":
            continue
        binding = current_by_owner.get((request.project_id, request.registration_generation))
        if binding is None or (binding.project_id, binding.registration_generation) in observation_owners:
            continue
        intent = working_set.retain_activation_observation_intent(
            binding,
            replay_epoch=request.parameters["replay_epoch"],
            replay_sequence=request.parameters["replay_sequence"],
        )
        if intent is not None:
            observation_intents.append(intent)
            observation_owners.add((binding.project_id, binding.registration_generation))
    for intent in working_set.select_activation_observations(limit=64 - len(observation_intents)):
        owner = (intent.binding.project_id, intent.binding.registration_generation)
        if owner not in observation_owners:
            observation_intents.append(intent)
            observation_owners.add(owner)
    observation_bindings = [intent.binding for intent in observation_intents]
    replay_cursors = {
        intent.binding.project_id: {"epoch": intent.replay_epoch, "sequence": intent.replay_sequence}
        for intent in observation_intents
    }
    if observation_intents or any(request.operation_kind == "activation_observe" for request in unresolved):
        try:
            completions = controller.advance_activation_observations(
                observation_bindings,
                registry_revision,
                replay_cursors,
            )
        except (OSError, RuntimeError, ValueError):
            completions = {}
        working_set.apply_activation_observations(observation_intents, completions)

    acknowledgement_intents = []
    acknowledgement_owners: set[tuple[str, str | None]] = set()
    for request in unresolved:
        if request.operation_kind != "activation_consumer_ack":
            continue
        binding = current_by_owner.get((request.project_id, request.registration_generation))
        if binding is None or (binding.project_id, binding.registration_generation) in acknowledgement_owners:
            continue
        intent = working_set.retain_activation_acknowledgement_intent(
            binding,
            epoch=request.parameters["epoch"],
            sequence=request.parameters["sequence"],
            reconstructed_floor=request.parameters["reconstructed_floor"],
            require_current=request.parameters["require_current"],
        )
        if intent is not None:
            acknowledgement_intents.append(intent)
            acknowledgement_owners.add((binding.project_id, binding.registration_generation))
    for intent in working_set.select_activation_acknowledgements(limit=64 - len(acknowledgement_intents)):
        owner = (intent.binding.project_id, intent.binding.registration_generation)
        if owner not in acknowledgement_owners:
            acknowledgement_intents.append(intent)
            acknowledgement_owners.add(owner)
    acknowledgement_bindings = [intent.binding for intent in acknowledgement_intents]
    acknowledgements = {
        intent.binding.project_id: {
            "process_fence": working_set.process_fence,
            "epoch": intent.epoch,
            "sequence": intent.sequence,
            "reconstructed_floor": intent.reconstructed_floor,
            "require_current": intent.require_current,
        }
        for intent in acknowledgement_intents
    }
    if acknowledgement_intents or any(request.operation_kind == "activation_consumer_ack" for request in unresolved):
        try:
            completions = controller.advance_activation_consumer_acks(
                acknowledgement_bindings,
                registry_revision,
                acknowledgements,
            )
        except (OSError, RuntimeError, ValueError):
            completions = {}
        working_set.apply_activation_acknowledgements(acknowledgement_intents, completions)


def _isolated_claim_identity(reservation: ReservationIdentity) -> dict[str, Any] | None:
    """Build the exact launch ticket key from one active local reservation."""
    attempt_id = reservation.attempt_id
    prefix = f"{reservation.task_id}-attempt-"
    if (
        reservation.project_id is None
        or reservation.shared_root is None
        or not isinstance(attempt_id, str)
        or not attempt_id.startswith(prefix)
        or type(reservation.fencing_token) is not int
        or reservation.fencing_token < 1
        or not reservation.reservation_id
    ):
        return None
    suffix = attempt_id[len(prefix) :]
    if not suffix.isascii() or not suffix.isdigit() or suffix.startswith("0"):
        return None
    attempt_number = int(suffix)
    if attempt_id != f"{reservation.task_id}-attempt-{attempt_number}":
        return None
    return {
        "task_id": reservation.task_id,
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "fencing_token": reservation.fencing_token,
        "reservation_id": reservation.reservation_id,
    }


def _isolated_active_reservations(
    records: tuple[dict[str, Any], ...],
    bindings_by_project: dict[str, ProjectBinding],
    trusted: set[ReservationIdentity] | None,
    *,
    require_current_generation: bool,
) -> dict[str, tuple[ReservationIdentity, ProjectBinding, dict[str, Any]]]:
    """Convert exact binding-local active records, optionally requiring shared proof."""
    trusted_by_id = {item.reservation_id: item for item in trusted} if trusted is not None else None
    active: dict[str, tuple[ReservationIdentity, ProjectBinding, dict[str, Any]]] = {}
    for record in records:
        if not isinstance(record, dict) or record.get("state") != "active":
            continue
        try:
            identity = ReservationIdentity.from_record(record)
        except (TypeError, ValueError):
            continue
        binding = bindings_by_project.get(identity.project_id or "")
        if (
            binding is None
            or (
                require_current_generation
                and identity.registration_generation not in {None, binding.registration_generation}
            )
            or identity.shared_root != str(binding.shared_root)
            or record.get("machine_name") != binding.machine_name
            or _isolated_claim_identity(identity) is None
            or (trusted_by_id is not None and trusted_by_id.get(identity.reservation_id) != identity)
        ):
            continue
        active[identity.reservation_id] = (identity, binding, record)
    return active


def _dispatch_isolated_machine_cycle(
    runtime: MachineRuntime,
    *,
    controller: ProjectIOController,
    bindings: list[ProjectBinding],
    validated_configs: dict[str, RootConfig],
    registry_revision: int,
    launch_batch: _LaunchHandoffBatch,
    available_gpus: list[int] | None,
    instance_id: str,
    omitted_dormant_enabled: bool,
) -> list[dict[str, Any]]:
    """Advance one bounded asynchronous scheduler pass using local snapshots."""
    runtime.last_cycle_consumed_binding = bool(validated_configs)
    results = [
        {
            "project_id": binding.project_id,
            "launched": [],
            "status": (
                "binding_validation_pending"
                if binding.enabled and binding.project_id not in validated_configs
                else "pending"
                if not binding.enabled
                else "dispatched"
            ),
        }
        for binding in bindings
    ]
    result_by_project = {item["project_id"]: item for item in results}
    supervision_bindings = [binding for binding in bindings if binding.project_id in validated_configs]
    reconciliation_bindings = list(supervision_bindings)
    reconciliation_binding_by_project = {binding.project_id: binding for binding in reconciliation_bindings}
    coordinator = getattr(runtime, "attempt_supervision_coordinator", None)
    if not isinstance(coordinator, AttemptSupervisionCoordinator):
        raise RuntimeError("isolated dispatch requires the runtime-owned attempt supervision coordinator.")
    coordinator.advance_all(supervision_bindings, registry_revision)
    ready_generations = getattr(runtime, "authority_ready_generations", None)
    if not isinstance(ready_generations, dict):
        ready_generations = {}
        runtime.authority_ready_generations = ready_generations
    current_generations = {binding.project_id: binding.registration_generation for binding in bindings}
    for project_id, generation in tuple(ready_generations.items()):
        if current_generations.get(project_id) != generation:
            ready_generations.pop(project_id, None)
    for binding in supervision_bindings:
        if binding.enabled and coordinator.initially_reconciled(binding, registry_revision):
            previous = ready_generations.get(binding.project_id)
            ready_generations[binding.project_id] = binding.registration_generation
            if previous != binding.registration_generation:
                # A newly ready binding should wake admission immediately.
                runtime.last_cycle_had_demand = True
        else:
            ready_generations.pop(binding.project_id, None)
    enabled_bindings = [binding for binding in bindings if binding.enabled]
    validated_bindings = [binding for binding in enabled_bindings if binding.project_id in validated_configs]
    authority_ready = {
        binding.project_id
        for binding in validated_bindings
        if ready_generations.get(binding.project_id) == binding.registration_generation
    }
    for binding in validated_bindings:
        if binding.project_id not in authority_ready:
            result_by_project[binding.project_id]["status"] = "authority_recovering"
    validated_bindings = [binding for binding in validated_bindings if binding.project_id in authority_ready]
    binding_by_project = {binding.project_id: binding for binding in validated_bindings}
    blocked_projects = set(runtime.upgrade_admission_blocked_projects)
    if runtime.upgrade_discovery_unknown:
        blocked_projects.update(binding.project_id for binding in enabled_bindings)
    for project_id in blocked_projects:
        item = result_by_project.get(project_id)
        if item is not None and item["status"] not in {"binding_validation_pending", "authority_recovering"}:
            item["status"] = "upgrade_blocked"
            item["upgrade"] = {"admission_blocked": True, "state": "cached"}

    cursor = runtime.load_cursor()
    ordered_project_ids = order_dispatch_project_ids(tuple(binding_by_project), cursor)
    ordered_bindings = [binding_by_project[project_id] for project_id in ordered_project_ids]
    primary_candidates: dict[str, Mapping[str, Any]] = {}
    primary_unknown = False

    try:
        due_offer_completions = controller.advance_scheduler_due_offers(
            [
                binding
                for binding in ordered_bindings
                if binding.project_id not in blocked_projects
                and not controller.scheduler_is_quiescent(binding, registry_revision)
            ],
            registry_revision,
        )
        if any(
            evidence.get("outcome") == "offered"
            for evidence in due_offer_completions.values()
            if isinstance(evidence, Mapping)
        ):
            runtime.last_cycle_had_demand = True
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError):
        runtime.last_cycle_had_demand = True

    def local_capacity() -> tuple[list[int], int, int, int]:
        snapshot = reservation_snapshot(runtime.root)
        _gpu_policy, policy_view = _observe_gpu_policy(
            runtime,
            available_gpus=available_gpus,
            instance_id=instance_id,
            reserved_gpu_ids=snapshot.reserved_gpu_ids,
        )
        visible = list(policy_view.visible_gpu_ids or ())
        free_gpu_ids = [gpu_id for gpu_id in visible if gpu_id not in snapshot.reserved_gpu_ids]
        cpu_policy, cpu_reservations = cpu_reservation_snapshot(runtime.root)
        reserved_cpu_slots = 0
        cpu_valid = True
        for record in cpu_reservations:
            slots = record.get("cpu_slots") if isinstance(record, dict) else None
            if type(slots) is not int or slots < 0:
                cpu_valid = False
                break
            reserved_cpu_slots += slots
        free_cpu_slots = max(0, cpu_policy.capacity - reserved_cpu_slots) if cpu_valid else 0
        return free_gpu_ids, free_cpu_slots, len(visible), cpu_policy.capacity

    try:
        initial_free_gpu_ids, initial_free_cpu_slots, _visible_gpu_count, _cpu_capacity = local_capacity()
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError):
        initial_free_gpu_ids, initial_free_cpu_slots = [], 0
        primary_unknown = bool(ordered_bindings)
    if initial_free_gpu_ids or initial_free_cpu_slots:
        try:
            ready_completions = controller.advance_scheduler_ready_index_builds(
                [binding for binding in ordered_bindings if binding.project_id not in blocked_projects],
                registry_revision,
                observed_only=True,
            )
            if any(
                evidence.get("state") == "building"
                for evidence in ready_completions.values()
                if isinstance(evidence, Mapping)
            ):
                runtime.last_cycle_had_demand = True
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError):
            runtime.last_cycle_had_demand = True

    for lane in ("gpu", "cpu"):
        observations: dict[str, Mapping[str, Any]] = {}
        borrow_admission = None
        try:
            free_gpu_ids, free_cpu_slots, visible_gpu_count, cpu_capacity = local_capacity()
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError):
            free_gpu_ids, free_cpu_slots = [], 0
            visible_gpu_count, cpu_capacity = 0, 0
            diagnostic_increment(f"scheduler.isolated.{lane}.capacity_unavailable")
            if ordered_bindings:
                primary_unknown = True
        lane_has_capacity = bool(free_gpu_ids) if lane == "gpu" else free_cpu_slots > 0
        try:
            observations = dict(
                controller.advance_scheduler_observations(
                    [
                        binding
                        for binding in ordered_bindings
                        if lane_has_capacity
                        and binding.project_id not in blocked_projects
                        and not controller.scheduler_is_quiescent(binding, registry_revision)
                    ],
                    registry_revision,
                    lane=lane,
                    admission_role="primary",
                )
            )
            if any(
                evidence.get("outcome") == "candidate"
                for evidence in observations.values()
                if isinstance(evidence, Mapping)
            ):
                primary_candidates.update(
                    {
                        project_id: evidence
                        for project_id, evidence in observations.items()
                        if isinstance(evidence, Mapping) and evidence.get("outcome") == "candidate"
                    }
                )
            if lane_has_capacity and any(
                binding.project_id not in observations
                and not controller.scheduler_is_quiescent(binding, registry_revision)
                for binding in ordered_bindings
            ):
                primary_unknown = True
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError):
            diagnostic_increment(f"scheduler.isolated.{lane}.observation_failed")
            primary_unknown = True

        all_current_validated = len(validated_bindings) == len(enabled_bindings)
        no_upgrade_blockers = not blocked_projects.intersection(binding_by_project)
        try:
            empty_scans = {
                project_id: evidence
                for project_id, evidence in observations.items()
                if isinstance(evidence, Mapping) and evidence.get("outcome") == "none"
            }
            candidates = {
                project_id: evidence
                for project_id, evidence in observations.items()
                if project_id not in blocked_projects
                and isinstance(evidence, Mapping)
                and evidence.get("outcome") == "candidate"
            }
            claim_completions = controller.advance_scheduler_claims(
                ordered_bindings,
                registry_revision,
                lane=lane,
                admission_role="primary",
                observations=candidates,
                available_gpu_ids=free_gpu_ids,
                available_cpu_slots=free_cpu_slots,
            )
            if any(evidence.get("outcome") == "claimed" for evidence in claim_completions.values()):
                runtime.last_cycle_had_demand = True

            borrow_observations = dict(
                controller.advance_scheduler_observations(
                    [
                        binding
                        for binding in ordered_bindings
                        if lane_has_capacity
                        and binding.project_id not in blocked_projects
                        and not controller.scheduler_is_quiescent(binding, registry_revision)
                    ],
                    registry_revision,
                    lane=lane,
                    admission_role="borrow",
                )
            )
            borrow_candidates = {
                project_id: evidence
                for project_id, evidence in borrow_observations.items()
                if project_id not in blocked_projects
                and isinstance(evidence, Mapping)
                and evidence.get("outcome") == "candidate"
            }
            if (
                borrow_candidates
                and lane_has_capacity
                and all_current_validated
                and not omitted_dormant_enabled
                and no_upgrade_blockers
            ):
                borrow_admission = controller.issue_borrow_admission(
                    ordered_bindings,
                    registry_revision,
                    lane=lane,
                    visible_capacity=visible_gpu_count if lane == "gpu" else cpu_capacity,
                    free_capacity=len(free_gpu_ids) if lane == "gpu" else free_cpu_slots,
                )
            else:
                controller.invalidate_borrow_admission(lane)
            controller.advance_scheduler_cursor_commits(
                ordered_bindings,
                registry_revision,
                lane=lane,
                admission_role="borrow",
                observations={
                    project_id: evidence
                    for project_id, evidence in borrow_observations.items()
                    if isinstance(evidence, Mapping) and evidence.get("outcome") == "none"
                },
            )
            borrow_completions = controller.advance_scheduler_claims(
                ordered_bindings,
                registry_revision,
                lane=lane,
                admission_role="borrow",
                observations=borrow_candidates,
                available_gpu_ids=free_gpu_ids,
                available_cpu_slots=free_cpu_slots,
                borrow_admission=borrow_admission,
            )
            if any(evidence.get("outcome") == "claimed" for evidence in borrow_completions.values()):
                runtime.last_cycle_had_demand = True
            controller.advance_scheduler_cursor_commits(
                ordered_bindings,
                registry_revision,
                lane=lane,
                admission_role="primary",
                observations=empty_scans,
            )
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError):
            diagnostic_increment(f"scheduler.isolated.{lane}.advance_failed")
            if ordered_bindings:
                primary_unknown = True

    try:
        final_snapshot = reservation_snapshot(runtime.root)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError):
        final_snapshot = None

    try:
        executor_status = controller.executor.poll()
        unresolved = controller.executor.unresolved_requests()
        if unresolved or executor_status.get("envelope") == "unknown":
            runtime.last_cycle_had_demand = True
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError):
        runtime.last_cycle_had_demand = True

    if final_snapshot is not None:
        if final_snapshot.active or final_snapshot.provisional:
            runtime.last_cycle_had_demand = True
        reconcilable_records = _isolated_active_reservations(
            final_snapshot.active,
            reconciliation_binding_by_project,
            None,
            require_current_generation=False,
        )
        try:
            trusted = set(
                controller.advance_scheduler_reservation_reconciliations(
                    reconciliation_bindings,
                    registry_revision,
                    reservations=[item[0] for item in reconcilable_records.values()],
                )
            )
            reconciled_snapshot = reservation_snapshot(runtime.root)
            active_records = _isolated_active_reservations(
                reconciled_snapshot.active,
                binding_by_project,
                trusted,
                require_current_generation=True,
            )
            reservations = [item[0] for item in active_records.values()]
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError):
            active_records = {}
            reservations = []
            trusted = set()
            runtime.last_cycle_had_demand = True
    else:
        active_records = {}
        reservations = []
        trusted = set()
        runtime.last_cycle_had_demand = True

    try:
        completions = controller.advance_scheduler_launch_authorizations(
            ordered_bindings,
            registry_revision,
            reservations=reservations,
        )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError):
        completions = {}
        runtime.last_cycle_had_demand = True

    if completions and final_snapshot is not None:
        try:
            latest_snapshot = reservation_snapshot(runtime.root)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError):
            latest_snapshot = None
            runtime.last_cycle_had_demand = True
        latest_active = (
            _isolated_active_reservations(
                latest_snapshot.active,
                binding_by_project,
                trusted,
                require_current_generation=True,
            )
            if latest_snapshot is not None
            else {}
        )
        for reservation_id, evidence in completions.items():
            if not isinstance(evidence, Mapping) or evidence.get("outcome") != "authorized":
                continue
            original = active_records.get(reservation_id)
            current = latest_active.get(reservation_id)
            if original is None or current is None:
                continue
            identity, binding, _record = original
            current_identity, current_binding, current_record = current
            claim_identity = _isolated_claim_identity(identity)
            if (
                claim_identity is None
                or current_identity != identity
                or current_binding != binding
                or current_identity.reservation_id != reservation_id
                or current_record.get("state") != "active"
                or not identity.matches(current_record)
                or evidence.get("claim_identity") != claim_identity
            ):
                continue
            cfg = validated_configs.get(binding.project_id)
            if cfg is None:
                continue
            try:
                launch_batch.launch_authorized(cfg, evidence, binding.project_id)
            except (OSError, RuntimeError, ValueError, TypeError) as exc:
                if not launch_batch.has_pending(binding.project_id, identity.attempt_id or ""):
                    controller.retry_scheduler_launch_authorization(identity)
                result_by_project[binding.project_id]["status"] = "error"
                result_by_project[binding.project_id]["error"] = str(exc)
                runtime.last_cycle_had_demand = True
                continue
            result_by_project[binding.project_id]["launched"].append(identity.task_id)
            runtime.save_cursor(binding.project_id)
            runtime.last_cycle_had_demand = True

    try:
        descriptor_completions = controller.advance_maintenance_descriptor_work(
            [binding for binding in ordered_bindings if binding.project_id not in blocked_projects],
            registry_revision,
        )
        for project_id, evidence in descriptor_completions.items():
            state = evidence.get("maintenance_state") if isinstance(evidence, Mapping) else None
            if state not in {"idle", "waiting", "completed"}:
                result_by_project[project_id]["maintenance"] = dict(evidence)
            if state in {"pending", "running"}:
                runtime.last_cycle_had_demand = True
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError):
        runtime.last_cycle_had_demand = True

    try:
        maintenance_completions = controller.advance_maintenance_event_flushes(
            ordered_bindings,
            registry_revision,
        )
        if maintenance_completions:
            runtime.last_cycle_had_demand = True
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError):
        runtime.last_cycle_had_demand = True

    if final_snapshot is not None:
        occupied_projects = {
            record.get("project_id")
            for record in (*final_snapshot.active, *final_snapshot.provisional)
            if isinstance(record, dict)
        }
        occupied_projects.update(identity[0] for identity in runtime.pending_launch_identities())
        try:
            # Quiescence is a per-Project proof. A candidate retained for one
            # binding must not prevent every other binding from advancing its
            # scheduler lane. The controller already excludes identities with
            # retained candidate evidence, so offer closure independently for
            # every otherwise-unoccupied Project on every pass.
            controller.advance_scheduler_quiescence(
                [
                    binding
                    for binding in ordered_bindings
                    if binding.project_id not in blocked_projects and binding.project_id not in occupied_projects
                ],
                registry_revision,
            )
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError):
            runtime.last_cycle_had_demand = True
    if any(not controller.scheduler_is_quiescent(binding, registry_revision) for binding in ordered_bindings):
        runtime.last_cycle_had_demand = True

    launch_batch.poll(result_by_project, isolated=True)
    if any(binding.enabled and binding.project_id not in validated_configs for binding in bindings):
        runtime.last_cycle_had_demand = True
    if primary_candidates or (
        primary_unknown
        and any(not controller.scheduler_is_quiescent(binding, registry_revision) for binding in ordered_bindings)
    ):
        runtime.last_cycle_had_demand = True
    if getattr(runtime, "pending_launch_handoffs", {}):
        runtime.last_cycle_had_demand = True
    runtime.last_cycle_validated_project_ids = frozenset(validated_configs)
    runtime.last_scheduler_diagnostic_probes = ()
    return results


def _dispatch_machine_cycle_locked(
    runtime: MachineRuntime,
    *,
    available_gpus: list[int] | None = None,
    executor: Executor | None = None,
    instance_id: str = "machine-agent",
    heartbeat_interval_seconds: float = 5.0,
    started_at: str | None = None,
    supervisors: dict[str, AuthoritySupervisor] | None = None,
    supervise: bool = True,
    publish_snapshots: bool = True,
) -> list[dict[str, Any]]:
    executor = executor or Executor()
    launch_batch = _LaunchHandoffBatch(executor, runtime)
    runtime.last_scheduler_diagnostic_probes = ()
    runtime.last_cycle_had_demand = False
    runtime.last_cycle_consumed_binding = False
    runtime.last_cycle_validated_project_ids = frozenset()
    has_isolation_executor = isinstance(getattr(runtime, "project_io_executor", None), ProjectIOExecutor)
    if not has_isolation_executor:
        # Keep the existing synchronous caller's pre-registry handoff poll
        # order. Isolated launches are polled only after safe controller setup.
        launch_batch.poll()
        if getattr(runtime, "pending_launch_handoffs", {}):
            runtime.last_cycle_had_demand = True
    with diagnostic_span("machine.registry.load"):
        registry_revision, registered = runtime.load_registry_snapshot()
    runtime.working_set.reconcile(registered, revision=registry_revision)
    if has_isolation_executor:
        try:
            runtime.activation_wake.capture(registry_revision, registered)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            runtime.last_cycle_had_demand = True
    runtime.working_set.poll_dormant(limit=4)
    _activate_due_maintenance_retries(runtime, registered)
    resident_bindings = runtime.working_set.resident_bindings()
    project_io_controller = _project_io_controller(runtime)
    isolated_mode = project_io_controller is not None
    validated_configs = (
        project_io_controller.advance_binding_validation(resident_bindings, registry_revision)
        if project_io_controller is not None
        else {}
    )
    if project_io_controller is not None:
        _advance_machine_snapshot_publications(runtime, project_io_controller, validated_configs)
        _advance_registration_renewals(runtime, project_io_controller, registry_revision, registered)
        _advance_activation_working_set(runtime, project_io_controller, registry_revision, registered)
        project_io_controller.advance_upgrade_work(registered, registry_revision)
        resident_bindings = runtime.working_set.resident_bindings()
        project_io_controller.advance_submission_control(resident_bindings, registry_revision)
        project_io_controller.advance_observation_maintenance(resident_bindings, registry_revision)
        project_io_controller.advance_notification_maintenance(registered, registry_revision)
        project_io_controller.advance_progress_work(registered, registry_revision)
        recovery_enrollment = getattr(runtime, "recovery_enrollment", None)
        if recovery_enrollment is not None:
            recovery_enrollment.advance(project_io_controller, registered, registry_revision)
        project_io_controller.advance_group_service_probes(resident_bindings, registry_revision)
        launch_batch.poll(isolated=True)
        if getattr(runtime, "pending_launch_handoffs", {}):
            runtime.last_cycle_had_demand = True
    if isolated_mode:
        runtime.last_cycle_validated_project_ids = frozenset(validated_configs)
    if isolated_mode:
        resident_supervised = list(resident_bindings)
    else:
        resident_supervised = [
            binding for binding in resident_bindings if runtime.registration_status(binding)["state"] != "superseded"
        ]
    omitted_dormant_enabled = runtime.working_set.has_dormant_enabled_binding()
    supervised = resident_supervised
    if supervisors is not None:
        supervised_ids = {binding.project_id for binding in supervised}
        for project_id in set(supervisors) - supervised_ids:
            del supervisors[project_id]
            getattr(runtime, "supervisor_generations", {}).pop(project_id, None)
    if project_io_controller is not None:
        return _dispatch_isolated_machine_cycle(
            runtime,
            controller=project_io_controller,
            bindings=supervised,
            validated_configs=validated_configs,
            registry_revision=registry_revision,
            launch_batch=launch_batch,
            available_gpus=available_gpus,
            instance_id=instance_id,
            omitted_dormant_enabled=omitted_dormant_enabled,
        )
    if not supervised:
        try:
            _observe_gpu_policy(
                runtime,
                available_gpus=available_gpus,
                instance_id=instance_id,
                reserved_gpu_ids=reservation_snapshot(runtime.root).reserved_gpu_ids,
            )
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            pass
        return []
    readable: dict[str, RootConfig] = {}
    readable_bindings: dict[str, ProjectBinding] = {}
    dispatchable: dict[str, RootConfig] = {}
    results: list[dict[str, Any]] = []
    upgrade_blocked: dict[str, dict[str, Any]] = {}
    scheduler_turns = {
        binding.project_id: runtime.working_set.begin_turn(binding, "scheduler") for binding in supervised
    }
    maintenance_turns = {
        binding.project_id: runtime.working_set.begin_turn(binding, "maintenance") for binding in supervised
    }
    for binding in supervised:
        if isolated_mode:
            cfg = validated_configs.get(binding.project_id)
            if cfg is None:
                results.append(
                    {
                        "project_id": binding.project_id,
                        "launched": [],
                        "status": "binding_validation_pending",
                    }
                )
                continue
            readable[binding.project_id] = cfg
            readable_bindings[binding.project_id] = binding
            runtime.last_cycle_consumed_binding = True
            continue
        if not binding.enabled and runtime.binding_state(binding) != "draining":
            continue
        if not runtime.binding_write_eligible(binding, renew=True):
            results.append(
                {
                    "project_id": binding.project_id,
                    "launched": [],
                    "status": "registration_ineligible",
                    "error": (
                        "current registration generation or machine write eligibility is unavailable; "
                        "new admission is blocked while retained execution evidence is reconciled."
                    ),
                }
            )
            continue
        try:
            with diagnostic_span("machine.binding.load"):
                cfg = _helpers._binding_config(runtime, binding)
                try:
                    runtime.drain_legacy_runner_evidence(binding)
                except CaptureBusy:
                    diagnostic_increment("legacy_inbox.capture_deferred")
            readable[binding.project_id] = cfg
            readable_bindings[binding.project_id] = binding
            runtime.last_cycle_consumed_binding = True
        except (OSError, RuntimeError, ValueError) as exc:
            results.append(
                {
                    "project_id": binding.project_id,
                    "launched": [],
                    "status": "error",
                    "error": str(exc),
                }
            )
    with diagnostic_span("machine.reservations.reconcile_projects"):
        snapshot = _reconcile_machine_reservations(runtime, readable)
    for binding in supervised:
        cfg = readable.get(binding.project_id)
        if cfg is None:
            continue
        try:
            with diagnostic_span("maintenance.project"):
                maintain_project(
                    cfg,
                    reservation_runtime_root=runtime.root,
                    project_id=binding.project_id,
                    should_reconcile_reservations=False,
                )
            if supervise:
                supervisor = None if supervisors is None else supervisors.get(binding.project_id)
                supervisor_generations = getattr(runtime, "supervisor_generations", {})
                if supervisor is not None and supervisor_generations.get(binding.project_id) != (
                    binding.registration_generation
                ):
                    del supervisors[binding.project_id]
                    supervisor_generations.pop(binding.project_id, None)
                    supervisor = None
                if supervisor is None:
                    supervisor = AuthoritySupervisor(cfg, reservation_runtime_root=runtime.root)
                    if supervisors is not None:
                        supervisors[binding.project_id] = supervisor
                        supervisor_generations[binding.project_id] = binding.registration_generation
                supervisor.tick()
            ready_generations = getattr(runtime, "authority_ready_generations", None)
            if (
                ready_generations is not None
                and ready_generations.get(binding.project_id) != binding.registration_generation
                and binding.enabled
            ):
                results.append({"project_id": binding.project_id, "launched": [], "status": "authority_recovering"})
                continue
            if not binding.enabled:
                continue
            dispatchable[binding.project_id] = cfg
            if runtime.upgrade_discovery_unknown or binding.project_id in runtime.upgrade_admission_blocked_projects:
                upgrade_blocked[binding.project_id] = {"admission_blocked": True, "state": "cached"}
        except (OSError, RuntimeError, ValueError) as exc:
            results.append(
                {
                    "project_id": binding.project_id,
                    "launched": [],
                    "status": "error",
                    "error": str(exc),
                }
            )
    executor = executor or Executor()
    # A deadline may have elapsed while maintenance and reservation
    # reconciliation ran. Poll once more immediately before starting
    # recovery, then exclude any still-pending identity from relaunch.
    launch_batch.poll()
    with diagnostic_span("machine.reservations.snapshot"):
        snapshot = reconcile_snapshot(runtime.root)
    pending_identities = (
        runtime.pending_launch_identities()
        if hasattr(runtime, "pending_launch_identities")
        else set(getattr(runtime, "pending_launch_handoffs", {}))
    )
    recovered = _recover_starting_reservations(
        runtime,
        dispatchable,
        snapshot.active,
        executor,
        excluded_pending=pending_identities,
        launch_recovered=launch_batch.launch,
    )
    authority_recovering = {
        item["project_id"]
        for item in results
        if item.get("status") == "authority_recovering" and isinstance(item.get("project_id"), str)
    }
    has_other_result = any(item.get("status") != "authority_recovering" for item in results)
    authority_recovery_has_demand = _authority_recovery_has_possible_demand(readable, authority_recovering)
    runtime.last_cycle_had_demand = (
        runtime.last_cycle_had_demand or has_other_result or any(recovered.values()) or authority_recovery_has_demand
    )
    with diagnostic_span("machine.reservations.snapshot"):
        snapshot = reconcile_snapshot(runtime.root)
    reservation_policy, policy_view = _observe_gpu_policy(
        runtime,
        available_gpus=available_gpus,
        instance_id=instance_id,
        reserved_gpu_ids=snapshot.reserved_gpu_ids,
    )
    visible = list(policy_view.visible_gpu_ids or ())
    free = [gpu_id for gpu_id in visible if gpu_id not in snapshot.reserved_gpu_ids]
    cpu_policy, cpu_reservations = cpu_reservation_snapshot(runtime.root)
    free_cpu_slots = cpu_policy.capacity - sum(item.get("cpu_slots", 0) for item in cpu_reservations)
    cursor_project_id = runtime.load_cursor()
    enabled_by_project = {binding.project_id: binding for binding in supervised if binding.enabled}
    ordered_project_ids = order_dispatch_project_ids(
        tuple(enabled_by_project),
        cursor_project_id,
    )
    ordered_enabled = [enabled_by_project[project_id] for project_id in ordered_project_ids]
    diagnostic_increment("scheduler.capacity.visible_gpus", len(visible))
    diagnostic_increment("scheduler.capacity.reserved_gpus", len(snapshot.reserved_gpu_ids))
    diagnostic_increment("scheduler.capacity.free_gpus", len(free))
    result_by_project = {item["project_id"]: item for item in results if item.get("project_id") is not None}
    for project_id, upgrade_status in upgrade_blocked.items():
        item = result_by_project.get(project_id)
        if item is None:
            item = {"project_id": project_id, "launched": [], "status": "upgrade_blocked"}
            results.append(item)
            result_by_project[project_id] = item
        item["status"] = "upgrade_blocked"
        item["upgrade"] = upgrade_status
    admission_dispatchable = {
        project_id: cfg for project_id, cfg in dispatchable.items() if project_id not in upgrade_blocked
    }
    for binding in ordered_enabled:
        if binding.project_id in dispatchable and binding.project_id not in result_by_project:
            item = {
                "project_id": binding.project_id,
                "launched": list(recovered.get(binding.project_id, [])),
                "status": "dispatched",
            }
            results.append(item)
            result_by_project[binding.project_id] = item
    if free or free_cpu_slots:
        for binding in ordered_enabled:
            cfg = dispatchable.get(binding.project_id)
            if cfg is None:
                continue
            with diagnostic_span("machine.ready.state"):
                ready_state = read_ready_index_state(cfg)
            if ready_state not in {"absent", "building"}:
                continue
            try:
                for _ in range(8):
                    with diagnostic_span("machine.ready.build"):
                        state = advance_ready_index_build(cfg)
                    if state.get("state") != "building":
                        break
            except (OSError, RuntimeError, ValueError) as exc:
                item = result_by_project[binding.project_id]
                item["status"] = "error"
                item["error"] = str(exc)
                del dispatchable[binding.project_id]
    if not free and not free_cpu_slots:
        diagnostic_increment("scheduler.work.skipped_no_capacity", len(ordered_enabled))
    budget = SliceBudget(WorkBudgetPolicy())
    batch_sizers: dict[str, AdaptiveBatchSizer] = {}
    enabled_ids = {binding.project_id for binding in ordered_enabled}
    for project_id in set(runtime.ready_batch_sizers) - enabled_ids:
        del runtime.ready_batch_sizers[project_id]
    for binding in ordered_enabled:
        sizer = runtime.ready_batch_sizers.get(binding.project_id)
        if not isinstance(sizer, AdaptiveBatchSizer):
            sizer = AdaptiveBatchSizer(budget.policy)
            runtime.ready_batch_sizers[binding.project_id] = sizer
        batch_sizers[binding.project_id] = sizer

    last_successful_project_id: str | None = None
    scheduler_diagnostic_probes: list[dict[str, object]] = []
    if runtime.last_enablement_reconciliation_probe is not None:
        scheduler_diagnostic_probes.append(runtime.last_enablement_reconciliation_probe)
    diagnostic_bindings = {binding.project_id: binding for binding in registered}
    # CPU and GPU borrowing are separate admission domains.  A busy primary queue in one
    # lane must never turn the other lane's complete no-demand probe into a denial.
    for lane, has_capacity in (("gpu", bool(free)), ("cpu", bool(free_cpu_slots))):
        probe = (
            _probe_primary_demand(
                runtime,
                readable,
                admission_dispatchable,
                visible if lane == "gpu" else list(range(cpu_policy.capacity)),
                free if lane == "gpu" else list(range(free_cpu_slots)),
                SliceBudget(WorkBudgetPolicy()),
                snapshot.reservations,
                lane=lane,
                excluded_project_ids=set(upgrade_blocked),
            )
            if has_capacity
            else PrimaryDemandProbe("unresolved")
        )
        if omitted_dormant_enabled and probe.state == "no_primary_demand":
            probe = PrimaryDemandProbe(
                "unresolved",
                (*probe.diagnostics, {"reason": "dormant_primary_binding_not_probed"}),
            )
        borrow_admission_grant = None
        if has_capacity and not omitted_dormant_enabled and probe.state == "no_primary_demand":
            borrow_admission_grant = _build_borrow_admission_grant(
                runtime, admission_dispatchable, probe, enabled_ids - set(upgrade_blocked), lane=lane
            )
            if borrow_admission_grant is None:
                probe = PrimaryDemandProbe("unresolved", probe.diagnostics)
        dispatch_plan = build_machine_dispatch_plan(
            MachineDispatchSnapshot(
                enabled_project_ids=ordered_project_ids,
                cursor_project_id=cursor_project_id,
                has_free_capacity=has_capacity,
                primary_demand_state=probe.state,
            )
        )
        scheduler_diagnostic_probes.append(
            _scheduler_diagnostic_probe(
                runtime,
                lane=lane,
                probe=probe,
                registry_revision=registry_revision,
                bindings=diagnostic_bindings,
                covered_project_ids=(enabled_ids - set(upgrade_blocked)),
                available_capacity=(len(free) if lane == "gpu" else free_cpu_slots),
                borrow_denied=has_capacity and probe.state == "unresolved",
            )
        )
        diagnostic_increment(f"scheduler.primary_probe.{lane}.{probe.state}")
        unresolved_only_for_empty_authority = bool(probe.diagnostics) and all(
            item.get("reason") == "project_not_probeable"
            and item.get("project_id") in authority_recovering
            and not authority_recovery_has_demand
            for item in probe.diagnostics
        )
        if has_capacity and (
            probe.state in {"runnable_now", "waiting_for_aggregation"}
            or (probe.state == "unresolved" and not unresolved_only_for_empty_authority)
        ):
            runtime.last_cycle_had_demand = True
        budget = SliceBudget(WorkBudgetPolicy())
        for admission_role in dispatch_plan.admission_roles:
            lane_gpus = free if lane == "gpu" else []
            lane_cpus = free_cpu_slots if lane == "cpu" else 0
            lane_gpus, lane_cpus, layer_winner = _dispatch_admission_layer(
                runtime,
                ordered_enabled,
                admission_dispatchable,
                executor,
                lane_gpus,
                lane_cpus,
                visible,
                result_by_project,
                batch_sizers,
                budget,
                launch_batch,
                admission_role=admission_role,
                borrow_admission_grant=(borrow_admission_grant if admission_role == "borrow" else None),
                lane=lane,
                admission_blocked_project_ids=set(upgrade_blocked),
                gpu_policy=reservation_policy,
            )
            if lane == "gpu":
                free = lane_gpus
            else:
                free_cpu_slots = lane_cpus
            if layer_winner is not None:
                last_successful_project_id = layer_winner.project_id
                runtime.last_cycle_had_demand = True
        cursor_effect = reduce_dispatch_cursor(dispatch_plan, last_successful_project_id)
        if cursor_effect is not None:
            runtime.save_cursor(cursor_effect.project_id)
    runtime.last_scheduler_diagnostic_probes = tuple(scheduler_diagnostic_probes)
    _advance_resident_maintenance_turn(
        runtime,
        supervised,
        readable,
        maintenance_turns,
        result_by_project,
    )
    launch_batch.finish(result_by_project)
    if publish_snapshots:
        reservations = list(reservation_snapshot(runtime.root).reservations)
        _publish_project_snapshots(
            readable,
            instance_id=instance_id,
            pid=_read_pid(runtime),
            visible=visible,
            reservations=reservations,
            gpu_policy=policy_view,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
            started_at=started_at,
            write_guard=lambda project_id: runtime.binding_write_guard(readable_bindings[project_id]),
        )
    _acknowledge_scheduler_turns(runtime, supervised, scheduler_turns, result_by_project, recovered)
    return results


def _acknowledge_scheduler_turns(
    runtime: MachineRuntime,
    bindings: list[ProjectBinding],
    turns: dict[str, Any],
    results: dict[str, dict[str, Any]],
    recovered: dict[str, list[str]],
) -> None:
    """Retire only scheduler turns with no local or primary-probe obligation."""
    try:
        reservations = reservation_snapshot(runtime.root).reservations
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError):
        reservations = None
    pending_handoffs = runtime.pending_launch_identities()
    reservations_unknown = reservations is None or any(not isinstance(item, dict) for item in reservations)
    reservation_projects = (
        set()
        if reservations is None
        else {
            item.get("project_id")
            for item in reservations
            if isinstance(item, dict)
            and item.get("state") in ("active", "provisional")
            and isinstance(item.get("project_id"), str)
        }
    )
    handoff_projects = {identity[0] for identity in pending_handoffs}
    upgrade_or_recovery = (
        set(runtime.upgrade_pending_projects)
        | set(runtime.upgrade_idle_blocked_projects)
        | set(runtime.upgrade_runnable_projects)
        | set(runtime.upgrade_admission_blocked_projects)
        | set(runtime.recovery_enrollment_pending_projects)
        | set(runtime.upgrade_probe_deadlines)
    )
    for binding in bindings:
        turn = turns.get(binding.project_id)
        if turn is None:
            continue
        project_id = binding.project_id
        result = results.get(project_id)
        has_reservation = reservations_unknown or project_id in reservation_projects
        has_handoff = project_id in handoff_projects
        has_recovered = bool(recovered.get(project_id))
        has_launched = bool(result and result.get("launched"))
        has_flag = project_id in upgrade_or_recovery
        if has_reservation or has_handoff or has_recovered or has_launched or has_flag:
            try:
                runtime.working_set.activate(binding, "scheduler_obligation")
            except ValueError:
                pass
        route_quiescent = not binding.enabled or all(
            (route := runtime.primary_probe.route((project_id, scope, lane))).is_complete and route.recheck is None
            for lane in ("gpu", "cpu")
            for scope in ("shared", "home")
        )
        result_quiescent = not binding.enabled or (result is not None and result.get("status") == "dispatched")
        quiescent = bool(
            result_quiescent
            and route_quiescent
            and not has_reservation
            and not has_handoff
            and not has_recovered
            and not has_launched
            and not has_flag
        )
        runtime.working_set.acknowledge(turn, quiescent=quiescent)


def _activate_due_maintenance_retries(
    runtime: MachineRuntime,
    registered: Sequence[ProjectBinding],
) -> None:
    """Wake backoff-delayed Projects when their durable retry becomes due."""
    now = time.monotonic()
    current_by_project = {binding.project_id: binding for binding in registered}
    for project_id, (binding, due_at) in tuple(runtime.maintenance_retry_deadlines.items()):
        current = current_by_project.get(project_id)
        if current != binding or not current.enabled:
            runtime.maintenance_retry_deadlines.pop(project_id, None)
            runtime.maintenance_retry_idle_blocked_projects.discard(project_id)
            continue
        if now < due_at:
            continue
        try:
            runtime.working_set.activate(binding, "maintenance_retry_due")
        except ValueError:
            runtime.maintenance_retry_deadlines.pop(project_id, None)
            runtime.maintenance_retry_idle_blocked_projects.discard(project_id)
        else:
            runtime.maintenance_retry_deadlines.pop(project_id, None)
            runtime.maintenance_retry_idle_blocked_projects.discard(project_id)


def _advance_resident_maintenance_turn(
    runtime: MachineRuntime,
    bindings: list[ProjectBinding],
    readable: dict[str, RootConfig],
    turns: dict[str, Any],
    results: dict[str, dict[str, Any]],
) -> None:
    """Advance one due descriptor after primary admission, regardless of capacity."""
    candidates = [binding for binding in bindings if binding.project_id in readable]
    if not candidates:
        return
    next_project_id = runtime.load_maintenance_cursor()
    project_ids = [binding.project_id for binding in candidates]
    try:
        start = project_ids.index(next_project_id)
    except ValueError:
        start = 0
    ordered = candidates[start:] + candidates[:start]
    selected = ordered[0]
    next_index = (start + 1) % len(candidates)
    runtime.save_maintenance_cursor(project_ids[next_index])

    from ..runtime.maintenance import advance_maintenance_work

    project_id = selected.project_id
    runtime.maintenance_retry_deadlines.pop(project_id, None)
    runtime.maintenance_retry_idle_blocked_projects.discard(project_id)
    try:
        progress = advance_maintenance_work(
            readable[project_id],
            reservation_runtime_root=runtime.root,
        )
        state = progress.get("maintenance_state")
        if state not in {"idle", "waiting", "completed"}:
            result = results.setdefault(
                project_id,
                {"project_id": project_id, "launched": [], "status": "dispatched"},
            )
            result["maintenance"] = progress
        if state in {"pending", "running"}:
            runtime.last_cycle_had_demand = True
            quiescent = False
        else:
            quiescent = state in {"idle", "waiting", "completed", "intervention"}
            if state == "waiting":
                due_at = progress.get("next_due_at")
                if isinstance(due_at, str):
                    try:
                        parsed = datetime.fromisoformat(due_at.replace("Z", "+00:00"))
                    except ValueError:
                        parsed = None
                    if parsed is not None and parsed.tzinfo is not None:
                        delay = max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())
                        runtime.maintenance_retry_deadlines[project_id] = (selected, time.monotonic() + delay)
                        if progress.get("idle_blocking", True):
                            runtime.maintenance_retry_idle_blocked_projects.add(project_id)
        turn = turns.get(project_id)
        if turn is not None:
            runtime.working_set.acknowledge(turn, quiescent=quiescent)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        result = results.setdefault(project_id, {"project_id": project_id, "launched": [], "status": "dispatched"})
        result["maintenance"] = {"maintenance_state": "error", "error": str(exc)}
        turn = turns.get(project_id)
        if turn is not None:
            runtime.working_set.acknowledge(turn, quiescent=False)


def dispatch_machine_cycle(
    runtime: MachineRuntime,
    *,
    available_gpus: list[int] | None = None,
    executor: Executor | None = None,
) -> list[dict[str, Any]]:
    """Dispatch registered projects once in stable round-robin order."""
    with runtime.scheduler_authority(blocking=False) as acquired:
        if not acquired:
            return []
        with runtime.migration_read_guard() as is_migration_clear:
            if not is_migration_clear:
                return []
            return dispatch_machine_cycle_locked(runtime, available_gpus=available_gpus, executor=executor)
