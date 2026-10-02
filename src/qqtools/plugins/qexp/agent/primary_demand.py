"""Bounded primary-demand scanning for machine dispatch adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..config_types import RootConfig
from ..machine_dispatch_plan import PrimaryCandidateObservation, evaluate_primary_candidate
from ..runtime.group_namespace import read_group
from ..runtime.ready import (
    ReadyProbeBudgetExhausted,
    classify_ready_marker,
    peek_primary_ready_marker,
    read_ready_index_state,
    ready_index_route_revision,
)
from ..runtime.ready.group_members import is_group_ready_member_projection_usable
from ..runtime.records import normalize_group_record
from ..runtime.work_budget import SliceBudget
from ..scheduler import _eligible
from .dispatch_probe import PrimaryProbeSession
from .helpers import _working_directory_reason


@dataclass(frozen=True, slots=True)
class PrimaryDemandProbe:
    state: str
    diagnostics: tuple[dict[str, Any], ...] = ()


def probe_primary_demand(
    session: PrimaryProbeSession,
    enabled_ids: set[str],
    readable: dict[str, RootConfig],
    dispatchable: dict[str, RootConfig],
    visible: list[int],
    free: list[int],
    budget: SliceBudget,
    reservations: tuple[dict[str, Any], ...] = (),
    *,
    lane: str = "gpu",
    excluded_project_ids: set[str] | frozenset[str] = frozenset(),
    group_gpu_usage: Mapping[str, int] | None = None,
) -> PrimaryDemandProbe:
    """Scan primary candidates through an independent bounded ready cursor."""
    diagnostics: list[dict[str, str]] = []
    enabled_ids = set(enabled_ids) - set(excluded_project_ids)
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
    pending_routes = session.begin_round(lane, probe_route_keys)
    selected_recheck_route = session.next_recheck(lane)
    pending_route_keys = set(pending_routes)
    if selected_recheck_route is not None:
        pending_route_keys.add(selected_recheck_route)
    has_completed_selected_recheck = False
    for project_id in probe_project_ids:
        cfg = readable[project_id]
        try:
            ready_state = read_ready_index_state(cfg)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            diagnostics.append(
                {
                    "project_id": project_id,
                    "reason": "ready_index_unreadable",
                    "exception_type": type(exc).__name__,
                }
            )
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
            route = session.route(cursor_key)
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
                    diagnostics.append(
                        {
                            "project_id": project_id,
                            "route_scope": scope,
                            "reason": "index_unreadable",
                            "exception_type": type(exc).__name__,
                        }
                    )
                    return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
                decision = session.begin_route(cursor_key, current_revision)
                if decision.has_index_changed:
                    diagnostics.append({"project_id": project_id, "reason": "ready_index_changed"})
                    return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
                if not decision.should_scan:
                    continue
                route = session.route(cursor_key)
                cursor = route.cursor
            elif (route.cursor is None or route.revision is None) and not is_rechecking_pending_candidate:
                try:
                    observed_revision = ready_index_route_revision(cfg, scope, budget, primary_only=True, lane=lane)
                except ReadyProbeBudgetExhausted:
                    diagnostics.append({"project_id": project_id, "reason": "probe_budget_exhausted"})
                    return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
                except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                    diagnostics.append(
                        {
                            "project_id": project_id,
                            "route_scope": scope,
                            "reason": "index_unreadable",
                            "exception_type": type(exc).__name__,
                        }
                    )
                    return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
                session.begin_route(cursor_key, observed_revision)
                route = session.route(cursor_key)
                cursor = route.cursor
            while True:
                cursor_before = cursor
                try:
                    peek = peek_primary_ready_marker(cfg, project_id, scope, cursor, budget)
                except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                    session.record_progress(cursor_key, cursor_before)
                    diagnostics.append(
                        {
                            "project_id": project_id,
                            "route_scope": scope,
                            "reason": "index_unreadable",
                            "exception_type": type(exc).__name__,
                        }
                    )
                    return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
                if peek.exhausted or peek.unresolved:
                    session.record_progress(cursor_key, cursor_before)
                    diagnostics.append(
                        {
                            "project_id": project_id,
                            "reason": ("probe_budget_exhausted" if peek.exhausted else "ready_index_unresolved"),
                        }
                    )
                    return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
                cursor = peek.cursor
                session.record_progress(cursor_key, cursor)
                if peek.reference is None:
                    if is_rechecking_pending_candidate:
                        # Recheck cursors move through every dependency candidate
                        # in this route.  At the end, wrap to the route start so a
                        # still-blocked first candidate cannot starve later ones.
                        should_retain_recheck = has_recheck_candidate or recheck_start_cursor is not None
                        if should_retain_recheck:
                            session.record_dependency_wait(cursor_key, None)
                        session.finish_recheck(
                            cursor_key,
                            has_waiting_candidate=should_retain_recheck,
                        )
                        has_completed_selected_recheck = True
                        break
                    break
                reference = peek.reference
                if not budget.can_start_record(operations=3):
                    session.record_progress(cursor_key, cursor_before)
                    diagnostics.append({"project_id": project_id, "reason": "probe_budget_exhausted"})
                    return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
                budget.consume_record(operations=3)
                try:
                    result = classify_ready_marker(cfg, reference)
                except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                    session.record_progress(cursor_key, cursor_before)
                    diagnostics.append(
                        {
                            "project_id": project_id,
                            "task_id": reference.task_id,
                            "task_generation": reference.generation,
                            "route_scope": scope,
                            "reason": "marker_unreadable",
                            "exception_type": type(exc).__name__,
                        }
                    )
                    return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
                if result.classification == "corrupt":
                    corrupt_diagnostic: dict[str, Any] = {
                        "project_id": project_id,
                        "task_id": reference.task_id,
                        "task_generation": reference.generation,
                        "route_scope": scope,
                        "reason": result.reason,
                    }
                    if result.diagnostic is not None:
                        exception_type = result.diagnostic.as_dict().get("exception_type")
                        if isinstance(exception_type, str):
                            corrupt_diagnostic["exception_type"] = exception_type
                    diagnostics.append(corrupt_diagnostic)
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
                            session.record_dependency_wait(cursor_key, cursor)
                            # The completed baseline scan already established that
                            # this route has no runnable primary work.  Recheck only
                            # its dependency candidate this round, then let the next
                            # pending route consume the shared budget.
                            session.finish_recheck(cursor_key, has_waiting_candidate=True)
                            has_completed_selected_recheck = True
                            break
                        session.record_dependency_wait(cursor_key, cursor_before)
                    continue
                if result.classification != "claimable" or result.task is None:
                    continue
                task = result.task
                if (lane == "cpu") != task.spec.is_cpu_only:
                    continue
                try:
                    is_eligible = _eligible(cfg, task)
                except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                    session.record_progress(cursor_key, cursor_before)
                    diagnostics.append(
                        {
                            "project_id": project_id,
                            "task_id": task.task_id,
                            "task_generation": task.ready_generation,
                            "route_scope": scope,
                            "reason": "task_truth_unreadable",
                            "exception_type": type(exc).__name__,
                        }
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
                        session.record_progress(cursor_key, cursor_before)
                        diagnostics.append(
                            {
                                "project_id": project_id,
                                "task_id": task.task_id,
                                "task_generation": task.ready_generation,
                                "route_scope": scope,
                                "reason": "group_unreadable",
                                "exception_type": type(exc).__name__,
                            }
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
                used_group_gpus = 0
                if group_gpu_limit is not None:
                    used_group_gpus = (
                        group_gpu_usage.get(task.group_name, 0)
                        if group_gpu_usage is not None
                        else sum(
                            len(reservation.get("gpu_ids", []))
                            for reservation in reservations
                            if reservation.get("project_id") == project_id
                            and reservation.get("group_name") == task.group_name
                            and reservation.get("machine_name") == cfg.machine_name
                        )
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
                        group_gpu_usage=used_group_gpus,
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
                    session.hold_candidate(cursor_key, cursor_before)
                    return PrimaryDemandProbe("runnable_now", tuple(diagnostics[-32:]))
                if decision.outcome == "waiting_for_aggregation":
                    # Keep real primary demand at the resume position until it
                    # disappears or can claim resources.  Later dependency-only
                    # candidates must not turn this route into a negative cache.
                    session.hold_candidate(cursor_key, cursor_before)
                    return PrimaryDemandProbe("waiting_for_aggregation", tuple(diagnostics[-32:]))
            if is_rechecking_pending_candidate and has_completed_selected_recheck:
                continue
            try:
                end_revision = ready_index_route_revision(cfg, scope, budget, primary_only=True, lane=lane)
            except ReadyProbeBudgetExhausted:
                diagnostics.append({"project_id": project_id, "reason": "probe_budget_exhausted"})
                return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                diagnostics.append(
                    {
                        "project_id": project_id,
                        "route_scope": scope,
                        "reason": "index_unreadable",
                        "exception_type": type(exc).__name__,
                    }
                )
                return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
            decision = session.finish_route(cursor_key, end_revision)
            if decision.has_index_changed:
                diagnostics.append({"project_id": project_id, "reason": "ready_index_changed"})
                return PrimaryDemandProbe("unresolved", tuple(diagnostics[-32:]))
    return PrimaryDemandProbe("no_primary_demand", tuple(diagnostics[-32:]))
