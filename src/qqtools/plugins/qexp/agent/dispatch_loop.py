"""Machine scheduling and admission."""

from __future__ import annotations

import os
import shlex
import signal
import threading
import time
import uuid
from dataclasses import dataclass, replace
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
    PrimaryCandidateObservation,
    build_machine_dispatch_plan,
    evaluate_primary_candidate,
    order_dispatch_project_ids,
    reduce_dispatch_cursor,
)
from ..machine_state import publish_machine_snapshots, publish_machine_stop_snapshot
from ..observer_provisioning import submit_observer_job
from ..project_maintenance import maintain_project, reconcile_reservation
from ..runtime.group_namespace import read_group
from ..runtime.locks import exclusive
from ..runtime.paths import local_paths, shared_paths
from ..runtime.ready import (
    ReadyProbeBudgetExhausted,
    advance_ready_index_build,
    classify_ready_marker,
    peek_primary_ready_marker,
    peek_ready_marker,
    read_ready_index_state,
    ready_index_route_revision,
)
from ..runtime.ready.group_members import is_group_ready_member_projection_usable
from ..runtime.records import TaskSpec, normalize_group_record, utc_now
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
    _eligible,
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


@dataclass(frozen=True, slots=True)
class PrimaryDemandProbe:
    state: str
    diagnostics: tuple[dict[str, str], ...] = ()


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

    def poll(self, result_by_project: dict[str, dict[str, Any]] | None = None) -> None:
        """Poll pending handoffs once, never sleeping or waiting on a child."""
        pending_items = self._pending_items()
        if not pending_items:
            return
        for pending in pending_items:
            now = time.monotonic()
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

    def finish(self, result_by_project: dict[str, dict[str, Any]] | None = None) -> None:
        """Compatibility alias for one non-blocking poll."""
        self.poll(result_by_project)


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
    diagnostics: list[dict[str, str]] = []
    try:
        _, bindings = runtime.load_registry()
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        return PrimaryDemandProbe("unresolved", ({"reason": f"registry_unreadable:{exc}"},))
    enabled_ids = {binding.project_id for binding in bindings if binding.enabled} - set(excluded_project_ids)
    unavailable = sorted(
        project_id for project_id in enabled_ids if project_id not in readable or project_id not in dispatchable
    )
    if unavailable:
        diagnostics.extend(
            {
                "project_id": project_id,
                "reason": "project_not_probeable",
            }
            for project_id in unavailable
        )
        return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
    probe_project_ids = sorted(
        project_id for project_id in readable if project_id in dispatchable and project_id in enabled_ids
    )
    probe_route_keys = [(project_id, scope, lane) for project_id in probe_project_ids for scope in ("shared", "home")]
    pending_routes = runtime.primary_probe.begin_round(lane, probe_route_keys)
    selected_recheck_route = runtime.primary_probe.next_recheck(lane)
    pending_route_keys = set(pending_routes)
    if selected_recheck_route is not None:
        pending_route_keys.add(selected_recheck_route)
    has_completed_selected_recheck = False
    for project_id in probe_project_ids:
        cfg = readable[project_id]
        try:
            ready_state = read_ready_index_state(cfg)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            diagnostics.append({"project_id": project_id, "reason": f"ready_index_unreadable:{exc}"})
            return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
        if ready_state != "active":
            diagnostics.append({"project_id": project_id, "reason": "ready_index_unresolved"})
            return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
        if not is_group_ready_member_projection_usable(cfg):
            diagnostics.append({"project_id": project_id, "reason": "group_ready_members_unresolved"})
            return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
        for scope in ("shared", "home"):
            cursor_key = (project_id, scope, lane)
            if cursor_key not in pending_route_keys:
                continue
            has_recheck_candidate = False
            is_rechecking_pending_candidate = cursor_key == selected_recheck_route
            route = runtime.primary_probe.route(cursor_key)
            cursor = route.cursor
            recheck_start_cursor = None
            if is_rechecking_pending_candidate:
                # The cached revision is checked by the borrow grant immediately
                # before a claim.  Do not spend this slice rereading every stable
                # route before revisiting the one dependency candidate selected
                # for this round.
                recheck_start_cursor = route.recheck.cursor
                cursor = route.recheck.cursor
            elif route.is_complete:
                try:
                    current_revision = ready_index_route_revision(cfg, scope, budget, primary_only=True, lane=lane)
                except ReadyProbeBudgetExhausted:
                    diagnostics.append({"project_id": project_id, "reason": "probe_budget_exhausted"})
                    return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
                except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                    diagnostics.append({"project_id": project_id, "reason": f"index_unreadable:{exc}"})
                    return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
                decision = runtime.primary_probe.begin_route(cursor_key, current_revision)
                if decision.has_index_changed:
                    diagnostics.append({"project_id": project_id, "reason": "ready_index_changed"})
                    return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
                if not decision.should_scan:
                    continue
                route = runtime.primary_probe.route(cursor_key)
                cursor = route.cursor
            elif (route.cursor is None or route.revision is None) and not is_rechecking_pending_candidate:
                try:
                    observed_revision = ready_index_route_revision(cfg, scope, budget, primary_only=True, lane=lane)
                except ReadyProbeBudgetExhausted:
                    diagnostics.append({"project_id": project_id, "reason": "probe_budget_exhausted"})
                    return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
                except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                    diagnostics.append({"project_id": project_id, "reason": f"index_unreadable:{exc}"})
                    return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
                runtime.primary_probe.begin_route(cursor_key, observed_revision)
                route = runtime.primary_probe.route(cursor_key)
                cursor = route.cursor
            while True:
                cursor_before = cursor
                try:
                    peek = peek_primary_ready_marker(cfg, project_id, scope, cursor, budget)
                except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                    runtime.primary_probe.record_progress(cursor_key, cursor_before)
                    diagnostics.append({"project_id": project_id, "reason": f"index_unreadable:{exc}"})
                    return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
                if peek.exhausted or peek.unresolved:
                    runtime.primary_probe.record_progress(cursor_key, cursor_before)
                    diagnostics.append(
                        {
                            "project_id": project_id,
                            "reason": ("probe_budget_exhausted" if peek.exhausted else "ready_index_unresolved"),
                        }
                    )
                    return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
                cursor = peek.cursor
                runtime.primary_probe.record_progress(cursor_key, cursor)
                if peek.reference is None:
                    if is_rechecking_pending_candidate:
                        # Recheck cursors move through every dependency candidate
                        # in this route.  At the end, wrap to the route start so a
                        # still-blocked first candidate cannot starve later ones.
                        should_retain_recheck = has_recheck_candidate or recheck_start_cursor is not None
                        if should_retain_recheck:
                            runtime.primary_probe.record_dependency_wait(cursor_key, None)
                        runtime.primary_probe.finish_recheck(
                            cursor_key,
                            has_waiting_candidate=should_retain_recheck,
                        )
                        has_completed_selected_recheck = True
                        break
                    break
                reference = peek.reference
                if not budget.can_start_record(operations=3):
                    runtime.primary_probe.record_progress(cursor_key, cursor_before)
                    diagnostics.append({"project_id": project_id, "reason": "probe_budget_exhausted"})
                    return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
                budget.consume_record(operations=3)
                try:
                    result = classify_ready_marker(cfg, reference)
                except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                    runtime.primary_probe.record_progress(cursor_key, cursor_before)
                    diagnostics.append(
                        {"project_id": project_id, "task_id": reference.task_id, "reason": f"marker_unreadable:{exc}"}
                    )
                    return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
                if result.classification == "corrupt":
                    diagnostics.append(
                        {
                            "project_id": project_id,
                            "task_id": reference.task_id,
                            "reason": result.reason,
                        }
                    )
                    return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
                if result.classification == "temporarily_unavailable":
                    if result.diagnostic is not None:
                        diagnostics.append(
                            {
                                "project_id": project_id,
                                "task_id": reference.task_id,
                                "reason": result.reason,
                                "diagnostic": result.diagnostic.as_dict(),
                            }
                        )
                    if result.reason.startswith("dependency_"):
                        # A dependency can become claimable without changing this
                        # route's revision.  Preserve its cursor across bounded scan
                        # batches, but do not count it as resource demand: a dependency
                        # wait must not prevent an otherwise eligible borrow claim.
                        has_recheck_candidate = True
                        if is_rechecking_pending_candidate:
                            runtime.primary_probe.record_dependency_wait(cursor_key, cursor)
                            # The completed baseline scan already established that
                            # this route has no runnable primary work.  Recheck only
                            # its dependency candidate this round, then let the next
                            # pending route consume the shared budget.
                            runtime.primary_probe.finish_recheck(cursor_key, has_waiting_candidate=True)
                            has_completed_selected_recheck = True
                            break
                        runtime.primary_probe.record_dependency_wait(cursor_key, cursor_before)
                    continue
                if result.classification != "claimable" or result.task is None:
                    continue
                task = result.task
                if (lane == "cpu") != task.spec.is_cpu_only:
                    continue
                try:
                    is_eligible = _eligible(cfg, task)
                except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                    runtime.primary_probe.record_progress(cursor_key, cursor_before)
                    diagnostics.append(
                        {"project_id": project_id, "task_id": task.task_id, "reason": f"task_truth_unreadable:{exc}"}
                    )
                    return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
                if not is_eligible:
                    diagnostics.append(
                        {
                            "project_id": project_id,
                            "task_id": task.task_id,
                            "reason": "placement_rejected",
                        }
                    )
                    continue
                has_primary_group_worker = True
                if task.group_name is not None:
                    try:
                        group = read_group(cfg.shared_root, task.group_name)
                        normalize_group_record(group)
                    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                        runtime.primary_probe.record_progress(cursor_key, cursor_before)
                        diagnostics.append(
                            {"project_id": project_id, "task_id": task.task_id, "reason": f"group_unreadable:{exc}"}
                        )
                        return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
                    worker = group["group"]["worker_set"].get(cfg.machine_name)
                    if worker is None or worker["scheduling_role"] != "primary":
                        has_primary_group_worker = False
                group_gpu_limit = (
                    worker["gpu_limit_gpus"]
                    if lane == "gpu" and task.group_name is not None and has_primary_group_worker
                    else None
                )
                group_gpu_usage = 0
                if group_gpu_limit is not None:
                    group_gpu_usage = sum(
                        len(reservation.get("gpu_ids", []))
                        for reservation in reservations
                        if reservation.get("project_id") == project_id
                        and reservation.get("group_name") == task.group_name
                        and reservation.get("machine_name") == cfg.machine_name
                    )
                decision = evaluate_primary_candidate(
                    PrimaryCandidateObservation(
                        is_eligible=is_eligible,
                        has_primary_group_worker=has_primary_group_worker,
                        working_directory_reason=_working_directory_reason(task.spec),
                        requested_gpus=(task.spec.requested_cpus or 0) if lane == "cpu" else task.spec.requested_gpus,
                        # CPU primary admission needs the policy capacity here, rather than
                        # the currently free slots.  A request that fits the machine but not
                        # its current free capacity must retain priority over borrowing.
                        visible_gpu_count=len(visible),
                        free_gpu_count=len(free),
                        group_gpu_limit=group_gpu_limit,
                        group_gpu_usage=group_gpu_usage,
                    )
                )
                if decision.reason is not None:
                    diagnostics.append(
                        {
                            "project_id": project_id,
                            "task_id": task.task_id,
                            "reason": decision.reason,
                        }
                    )
                if decision.outcome == "runnable_now":
                    runtime.primary_probe.hold_candidate(cursor_key, cursor_before)
                    return PrimaryDemandProbe("runnable_now", tuple(diagnostics[-32:]))
                if decision.outcome == "waiting_for_aggregation":
                    # Keep real primary demand at the resume position until it
                    # disappears or can claim resources.  Later dependency-only
                    # candidates must not turn this route into a negative cache.
                    runtime.primary_probe.hold_candidate(cursor_key, cursor_before)
                    return PrimaryDemandProbe("waiting_for_aggregation", tuple(diagnostics[-32:]))
            if is_rechecking_pending_candidate and has_completed_selected_recheck:
                continue
            try:
                end_revision = ready_index_route_revision(cfg, scope, budget, primary_only=True, lane=lane)
            except ReadyProbeBudgetExhausted:
                diagnostics.append({"project_id": project_id, "reason": "probe_budget_exhausted"})
                return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                diagnostics.append({"project_id": project_id, "reason": f"index_unreadable:{exc}"})
                return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
            decision = runtime.primary_probe.finish_route(cursor_key, end_revision)
            if decision.has_index_changed:
                diagnostics.append({"project_id": project_id, "reason": "ready_index_changed"})
                return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
    return PrimaryDemandProbe("no_primary_demand", tuple(diagnostics[-32:]))


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
    diagnostics = RuntimeDiagnostics()
    with activate_diagnostics(diagnostics), diagnostic_span("dispatch_machine_cycle"):
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
    now_ns = time.monotonic_ns()
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
    runtime.last_cycle_had_demand = False
    runtime.last_cycle_consumed_binding = False
    # Poll handoffs before registry/recovery work, including the no-binding
    # case. A pending reservation must not keep an idle agent alive forever
    # without making progress on its bounded launch deadline.
    launch_batch.poll()
    if getattr(runtime, "pending_launch_handoffs", {}):
        runtime.last_cycle_had_demand = True
    with diagnostic_span("machine.registry.load"):
        _, registered = runtime.load_registry()
    supervised = [
        binding
        for binding in registered
        if (binding.enabled or runtime.binding_state(binding) == "draining")
        and runtime.registration_status(binding)["state"] != "superseded"
    ]
    if supervisors is not None:
        supervised_ids = {binding.project_id for binding in supervised}
        for project_id in set(supervisors) - supervised_ids:
            del supervisors[project_id]
            getattr(runtime, "supervisor_generations", {}).pop(project_id, None)
    if not supervised:
        for binding in registered:
            if runtime.registration_status(binding)["state"] == "superseded":
                continue
            try:
                if runtime.binding_write_eligible(binding, renew=True):
                    _helpers._binding_config(runtime, binding)
                    runtime.last_cycle_consumed_binding = True
                    break
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                continue
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
    for binding in supervised:
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
            ):
                results.append({"project_id": binding.project_id, "launched": [], "status": "authority_recovering"})
                continue
            dispatchable[binding.project_id] = cfg
            if binding.project_id in runtime.upgrade_admission_blocked_projects:
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
        borrow_admission_grant = None
        if probe.state == "no_primary_demand":
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
        diagnostic_increment(f"scheduler.primary_probe.{lane}.{probe.state}")
        unresolved_only_for_empty_authority = bool(probe.diagnostics) and all(
            item.get("reason") == "project_not_probeable"
            and item.get("project_id") in authority_recovering
            and not authority_recovery_has_demand
            for item in probe.diagnostics
        )
        if probe.state in {"runnable_now", "waiting_for_aggregation"} or (
            has_capacity and probe.state == "unresolved" and not unresolved_only_for_empty_authority
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
    return results


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
