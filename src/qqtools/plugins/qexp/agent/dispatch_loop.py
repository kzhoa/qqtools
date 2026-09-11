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

from ..legacy_agent import _visible_gpus, get_agent_status
from ..authority import AuthoritySupervisor
from ..config_types import RootConfig
from ..executor import Executor
from ..layout import load_machine_record, load_root_config, machine_state_path, runtime_pid_path
from ..machine_config import is_legacy_agent_project, load_machine_policy, save_machine_config
from ..machine_dispatch_plan import (
    MachineDispatchSnapshot,
    PrimaryCandidateObservation,
    PrimaryProbeRouteState,
    begin_primary_probe_route,
    build_machine_dispatch_plan,
    evaluate_primary_candidate,
    finish_primary_probe_route,
    order_dispatch_project_ids,
    reduce_dispatch_cursor,
)
from .context import MachineRuntime, ProjectBinding, default_machine_runtime_root
from ..machine_state import publish_machine_snapshots, publish_machine_stop_snapshot
from ..project_maintenance import maintain_project, reconcile_reservation
from ..runtime.locks import exclusive
from ..runtime.paths import local_paths, shared_paths
from ..runtime.ready import (
    ReadyProbeBudgetExhausted,
    advance_ready_index_build,
    classify_ready_marker,
    peek_primary_ready_marker,
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


from .helpers import (
    _binding_config, _record_bad_task_spec,
    _reconcile_machine_reservations, _recover_starting_reservations,
    _working_directory_reason, _publish_project_snapshots, _read_pid,
)
from . import helpers as _helpers

@dataclass(frozen=True, slots=True)
class PrimaryDemandProbe:
    state: str
    diagnostics: tuple[dict[str, str], ...] = ()


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
    pending_routes = runtime.primary_probe_pending_routes.setdefault(lane, set())
    pending_routes.intersection_update(probe_route_keys)
    pending_routes.update(key for key in probe_route_keys if not runtime.primary_probe_complete.get(key, False))
    if not pending_routes:
        pending_routes.update(probe_route_keys)
    has_incomplete_route = any(
        not runtime.primary_probe_complete.get(cursor_key, False) for cursor_key in probe_route_keys
    )
    pending_recheck_routes = [
        cursor_key for cursor_key in probe_route_keys if cursor_key in runtime.primary_probe_recheck_cursors
    ]
    recheck_route_cursor = runtime.primary_probe_recheck_round_cursors.get(lane)
    if has_incomplete_route or not pending_recheck_routes:
        selected_recheck_route = None
    elif recheck_route_cursor in pending_recheck_routes:
        selected_recheck_route = recheck_route_cursor
    else:
        selected_recheck_route = pending_recheck_routes[0]
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
            if cursor_key not in pending_routes:
                continue
            has_recheck_candidate = False
            is_rechecking_pending_candidate = False
            completed_route_cursor = None
            cursor = runtime.primary_probe_cursors.get(cursor_key)
            route_state = PrimaryProbeRouteState(
                cursor,
                runtime.primary_probe_revisions.get(cursor_key),
                runtime.primary_probe_complete.get(cursor_key, False),
            )
            if route_state.is_complete and cursor_key == selected_recheck_route:
                # Record the next route before any budget-limited work.  If this
                # slice ends early, the incomplete route resumes via pending_routes.
                selected_index = pending_recheck_routes.index(cursor_key)
                runtime.primary_probe_recheck_round_cursors[lane] = pending_recheck_routes[
                    (selected_index + 1) % len(pending_recheck_routes)
                ]
                # The cached revision is checked by the borrow grant immediately
                # before a claim.  Do not spend this slice rereading every stable
                # route before revisiting the one dependency candidate selected
                # for this round.
                completed_route_cursor = route_state.cursor
                cursor = runtime.primary_probe_recheck_cursors[cursor_key]
                route_state = PrimaryProbeRouteState(cursor, route_state.revision, False)
                runtime.primary_probe_cursors[cursor_key] = cursor
                runtime.primary_probe_complete[cursor_key] = False
                is_rechecking_pending_candidate = True
            elif route_state.is_complete:
                try:
                    current_revision = ready_index_route_revision(cfg, scope, budget, primary_only=True, lane=lane)
                except ReadyProbeBudgetExhausted:
                    diagnostics.append({"project_id": project_id, "reason": "probe_budget_exhausted"})
                    return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
                except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                    diagnostics.append({"project_id": project_id, "reason": f"index_unreadable:{exc}"})
                    return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
                decision = begin_primary_probe_route(route_state, current_revision)
                runtime.primary_probe_cursors[cursor_key] = decision.state.cursor
                runtime.primary_probe_revisions[cursor_key] = decision.state.revision
                runtime.primary_probe_complete[cursor_key] = decision.state.is_complete
                if decision.has_index_changed:
                    diagnostics.append({"project_id": project_id, "reason": "ready_index_changed"})
                    return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
                if not decision.should_scan:
                    pending_routes.discard(cursor_key)
                    continue
            if (route_state.cursor is None or route_state.revision is None) and not is_rechecking_pending_candidate:
                try:
                    start_revision = ready_index_route_revision(cfg, scope, budget, primary_only=True, lane=lane)
                except ReadyProbeBudgetExhausted:
                    diagnostics.append({"project_id": project_id, "reason": "probe_budget_exhausted"})
                    return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
                except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                    diagnostics.append({"project_id": project_id, "reason": f"index_unreadable:{exc}"})
                    return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
                decision = begin_primary_probe_route(route_state, start_revision)
                runtime.primary_probe_revisions[cursor_key] = decision.state.revision
                cursor = decision.state.cursor
            else:
                start_revision = route_state.revision
            runtime.primary_probe_complete[cursor_key] = False
            while True:
                cursor_before = cursor
                try:
                    peek = peek_primary_ready_marker(cfg, project_id, scope, cursor, budget)
                except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                    diagnostics.append({"project_id": project_id, "reason": f"index_unreadable:{exc}"})
                    return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
                if peek.exhausted or peek.unresolved:
                    diagnostics.append(
                        {
                            "project_id": project_id,
                            "reason": ("probe_budget_exhausted" if peek.exhausted else "ready_index_unresolved"),
                        }
                    )
                    return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
                cursor = peek.cursor
                runtime.primary_probe_cursors[cursor_key] = cursor
                if peek.reference is None:
                    if is_rechecking_pending_candidate:
                        # Recheck cursors move through every dependency candidate
                        # in this route.  At the end, wrap to the route start so a
                        # still-blocked first candidate cannot starve later ones.
                        runtime.primary_probe_recheck_cursors[cursor_key] = None
                        runtime.primary_probe_cursors[cursor_key] = completed_route_cursor
                        runtime.primary_probe_complete[cursor_key] = True
                        has_completed_selected_recheck = True
                    runtime.primary_probe_complete[cursor_key] = True
                    break
                reference = peek.reference
                if not budget.can_start_record(operations=3):
                    runtime.primary_probe_cursors[cursor_key] = cursor_before
                    diagnostics.append({"project_id": project_id, "reason": "probe_budget_exhausted"})
                    return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
                budget.consume_record(operations=3)
                try:
                    result = classify_ready_marker(cfg, reference)
                except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                    runtime.primary_probe_cursors[cursor_key] = cursor_before
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
                    if result.reason.startswith("dependency_"):
                        # A dependency can become claimable without changing this
                        # route's revision.  Preserve its cursor across bounded scan
                        # batches, but do not count it as resource demand: a dependency
                        # wait must not prevent an otherwise eligible borrow claim.
                        has_recheck_candidate = True
                        runtime.primary_probe_recheck_cursors.setdefault(cursor_key, cursor_before)
                        if is_rechecking_pending_candidate:
                            runtime.primary_probe_recheck_cursors[cursor_key] = cursor
                            # The completed baseline scan already established that
                            # this route has no runnable primary work.  Recheck only
                            # its dependency candidate this round, then let the next
                            # pending route consume the shared budget.
                            runtime.primary_probe_cursors[cursor_key] = completed_route_cursor
                            runtime.primary_probe_complete[cursor_key] = True
                            has_completed_selected_recheck = True
                            break
                    continue
                if result.classification != "claimable" or result.task is None:
                    continue
                task = result.task
                if (lane == "cpu") != task.spec.is_cpu_only:
                    continue
                try:
                    is_eligible = _eligible(cfg, task)
                except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                    runtime.primary_probe_cursors[cursor_key] = cursor_before
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
                        group = read_json(cfg.shared_root / "groups" / f"{task.group_name}.json")
                        normalize_group_record(group)
                    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                        runtime.primary_probe_cursors[cursor_key] = cursor_before
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
                    runtime.primary_probe_cursors[cursor_key] = cursor_before
                    return PrimaryDemandProbe("runnable_now", tuple(diagnostics[-32:]))
                if decision.outcome == "waiting_for_aggregation":
                    # Keep real primary demand at the resume position until it
                    # disappears or can claim resources.  Later dependency-only
                    # candidates must not turn this route into a negative cache.
                    runtime.primary_probe_cursors[cursor_key] = cursor_before
                    return PrimaryDemandProbe("waiting_for_aggregation", tuple(diagnostics[-32:]))
            if is_rechecking_pending_candidate and has_completed_selected_recheck:
                pending_routes.discard(cursor_key)
                continue
            try:
                end_revision = ready_index_route_revision(cfg, scope, budget, primary_only=True, lane=lane)
            except ReadyProbeBudgetExhausted:
                diagnostics.append({"project_id": project_id, "reason": "probe_budget_exhausted"})
                return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                diagnostics.append({"project_id": project_id, "reason": f"index_unreadable:{exc}"})
                return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
            decision = finish_primary_probe_route(
                PrimaryProbeRouteState(
                    runtime.primary_probe_cursors.get(cursor_key),
                    start_revision,
                    runtime.primary_probe_complete.get(cursor_key, False),
                ),
                end_revision,
            )
            runtime.primary_probe_cursors[cursor_key] = decision.state.cursor
            runtime.primary_probe_revisions[cursor_key] = decision.state.revision
            runtime.primary_probe_complete[cursor_key] = decision.state.is_complete
            if decision.has_index_changed:
                diagnostics.append({"project_id": project_id, "reason": "ready_index_changed"})
                return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
            if is_rechecking_pending_candidate and not has_recheck_candidate and not has_completed_selected_recheck:
                runtime.primary_probe_recheck_cursors.pop(cursor_key, None)
            if cursor_key == selected_recheck_route:
                has_completed_selected_recheck = True
            pending_routes.discard(cursor_key)
    return PrimaryDemandProbe("no_primary_demand", tuple(diagnostics[-32:]))


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
    for project_id in sorted(enabled_ids):
        cfg = dispatchable.get(project_id)
        if cfg is None:
            return None
        for scope in ("shared", "home"):
            cursor_key = (project_id, scope, lane)
            revision = runtime.primary_probe_revisions.get(cursor_key)
            if not runtime.primary_probe_complete.get(cursor_key) or not isinstance(revision, int):
                return None
            revisions.append(_BorrowAdmissionRevision(project_id, cfg, scope, revision))
    grant = _BorrowAdmissionGrant(runtime.root, tuple(revisions), lane)
    if not grant.is_valid(runtime.root):
        # A route checked in an earlier slice may have changed.  Reject the
        # observation here instead of passing a known-stale grant to a claim.
        for item in revisions:
            key = (item.project_id, item.queue_scope, lane)
            runtime.primary_probe_cursors[key] = None
            runtime.primary_probe_complete[key] = False
            runtime.primary_probe_revisions.pop(key, None)
        return None
    return grant


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
    *,
    admission_role: str,
    borrow_admission_grant: _BorrowAdmissionGrant | None = None,
    lane: str,
    admission_blocked_project_ids: set[str] | frozenset[str] = frozenset(),
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
                        executor=executor,
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
                        should_recover_starting=True,
                        work_budget=budget,
                        batch_sizer=batch_sizers[binding.project_id],
                        inspected_ready=inspected_ready,
                        lane=lane,
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
    runtime.last_cycle_had_demand = False
    runtime.last_cycle_consumed_binding = False
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
            cfg = _helpers._binding_config(runtime, binding)
            runtime.drain_legacy_runner_evidence(binding)
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
    snapshot = reconcile_snapshot(runtime.root)
    recovered = _recover_starting_reservations(
        runtime,
        dispatchable,
        snapshot.active,
        executor,
    )
    runtime.last_cycle_had_demand = bool(results) or any(recovered.values())
    snapshot = reconcile_snapshot(runtime.root)
    visible = (
        list(available_gpus)
        if available_gpus is not None
        else _visible_gpus(next(iter(readable.values())))
        if readable
        else []
    )
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
            if cfg is None or read_ready_index_state(cfg) not in {"absent", "building"}:
                continue
            try:
                for _ in range(8):
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
        if probe.state in {"runnable_now", "waiting_for_aggregation"} or (has_capacity and probe.state == "unresolved"):
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
                admission_role=admission_role,
                borrow_admission_grant=(borrow_admission_grant if admission_role == "borrow" else None),
                lane=lane,
                admission_blocked_project_ids=set(upgrade_blocked),
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
    if publish_snapshots:
        reservations = list(reservation_snapshot(runtime.root).reservations)
        _publish_project_snapshots(
            readable,
            instance_id=instance_id,
            pid=_read_pid(runtime),
            visible=visible,
            reservations=reservations,
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
        with runtime.migration_guard() as is_migration_clear:
            if not is_migration_clear:
                return []
            return dispatch_machine_cycle_locked(runtime, available_gpus=available_gpus, executor=executor)
