from __future__ import annotations

from typing import Any

from .agent import get_agent_status
from .batch_state import build_batch_summary_from_counts, collect_batch_task_counts
from .events import query_events
from .indexes import load_index
from .layout import RootConfig
from .lifecycle import (
    build_machine_workset,
    read_agent_snapshot,
    read_summary_snapshot,
)
from .models import (
    AGENT_STATE_STOPPED,
    ALL_PHASES,
    BATCH_COMMIT_COMMITTED,
    PHASE_QUEUED,
    PHASE_RUNNING,
    validate_group_name,
)
from .storage import (
    iter_all_batches,
    iter_all_tasks,
    iter_machines,
    load_batch,
    load_task,
    read_json,
)
from .layout import gpu_state_path


# ---------------------------------------------------------------------------
# Task views
# ---------------------------------------------------------------------------


def list_tasks(
    cfg: RootConfig,
    phase: str | None = None,
    batch_id: str | None = None,
    group: str | None = None,
    machine: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    validated_group = validate_group_name(group)

    task_ids: list[str]
    if phase:
        task_ids = load_index(cfg, "state", phase)
    elif batch_id:
        task_ids = load_index(cfg, "batch", batch_id)
    elif validated_group:
        task_ids = load_index(cfg, "group", validated_group)
    elif machine:
        task_ids = load_index(cfg, "machine", machine)
    else:
        task_ids = [t.task_id for t in iter_all_tasks(cfg)]

    allowed_batch_ids = set(load_index(cfg, "batch", batch_id)) if batch_id else None
    allowed_group_ids = (
        set(load_index(cfg, "group", validated_group)) if validated_group else None
    )
    allowed_machine_ids = set(load_index(cfg, "machine", machine)) if machine else None
    allowed_phase_ids = set(load_index(cfg, "state", phase)) if phase else None

    results: list[dict[str, Any]] = []
    for tid in task_ids:
        if allowed_phase_ids is not None and tid not in allowed_phase_ids:
            continue
        if allowed_batch_ids is not None and tid not in allowed_batch_ids:
            continue
        if allowed_group_ids is not None and tid not in allowed_group_ids:
            continue
        if allowed_machine_ids is not None and tid not in allowed_machine_ids:
            continue
        try:
            t = load_task(cfg, tid)
            results.append({
                "task_id": t.task_id,
                "name": t.name,
                "group": t.group,
                "phase": t.status.phase,
                "machine": t.machine_name,
                "gpus": t.spec.requested_gpus,
                "attempt": t.attempt,
                "created_at": t.timestamps.created_at,
                "batch_id": t.batch_id,
            })
        except FileNotFoundError:
            continue
        if len(results) >= limit:
            break
    return results


def inspect_task(cfg: RootConfig, task_id: str) -> dict[str, Any]:
    t = load_task(cfg, task_id)
    return t.to_dict()


# ---------------------------------------------------------------------------
# Batch views
# ---------------------------------------------------------------------------


def _compute_batch_summary(cfg: RootConfig, batch) -> dict[str, int]:
    """Recompute batch summary from actual task states."""
    surviving_ids, counts = collect_batch_task_counts(
        cfg,
        batch.task_ids,
        ignore_missing=True,
    )
    summary = build_batch_summary_from_counts(total=len(surviving_ids), counts=counts)
    return {
        "total": summary.total,
        "queued": summary.queued,
        "running": summary.running,
        "succeeded": summary.succeeded,
        "failed": summary.failed,
        "cancelled": summary.cancelled,
    }


def list_batches(cfg: RootConfig, limit: int = 50) -> list[dict[str, Any]]:
    batches = iter_all_batches(cfg)
    results: list[dict[str, Any]] = []
    for b in batches:
        if b.commit_state != BATCH_COMMIT_COMMITTED:
            continue
        summary = _compute_batch_summary(cfg, b)
        results.append({
            "batch_id": b.batch_id,
            "name": b.name,
            "group": b.group,
            "machine": b.machine_name,
            "commit_state": b.commit_state,
            **summary,
        })
        if len(results) >= limit:
            break
    return results


def inspect_batch(cfg: RootConfig, batch_id: str) -> dict[str, Any]:
    b = load_batch(cfg, batch_id)
    if b.commit_state != BATCH_COMMIT_COMMITTED:
        raise FileNotFoundError(
            f"Batch {batch_id} is not committed and is hidden from observer surfaces."
        )
    data = b.to_dict()
    # Override stale summary with live computation
    data["batch"]["summary"] = _compute_batch_summary(cfg, b)
    return data


# ---------------------------------------------------------------------------
# Machine views
# ---------------------------------------------------------------------------


def list_machines(cfg: RootConfig) -> list[dict[str, Any]]:
    machines = iter_machines(cfg)
    results: list[dict[str, Any]] = []
    for m in machines:
        machine_cfg = RootConfig(
            shared_root=cfg.shared_root,
            project_root=cfg.project_root,
            machine_name=m.machine_name,
            runtime_root=m.runtime_root,
        )
        agent_snapshot = read_agent_snapshot(machine_cfg)
        summary_snapshot = read_summary_snapshot(machine_cfg)
        workset = agent_snapshot.workset if agent_snapshot is not None else build_machine_workset(
            cfg,
            machine_name=m.machine_name,
        )
        counts_by_phase = (
            dict(summary_snapshot.counts_by_phase)
            if summary_snapshot is not None
            else {
                "queued": workset.queued_count,
                "dispatching": workset.dispatching_count,
                "starting": workset.starting_count,
                "running": workset.running_count,
            }
        )
        agent_status = (
            get_agent_status(machine_cfg, probe_local_pid=False)
            if agent_snapshot is not None
            else None
        )
        agent_state = (
            agent_status["agent_state"] if agent_status is not None else AGENT_STATE_STOPPED
        )
        results.append({
            "machine_name": m.machine_name,
            "hostname": m.hostname,
            "agent_mode": m.agent_mode,
            "agent_state": agent_state,
            "gpu_count": m.gpu_inventory.count,
            "last_heartbeat": agent_snapshot.last_heartbeat if agent_snapshot else None,
            "idle_deadline_at": agent_snapshot.idle_deadline_at if agent_snapshot else None,
            "drain_started_at": agent_snapshot.drain_started_at if agent_snapshot else None,
            "counts_by_phase": counts_by_phase,
            "workset": workset.to_dict(),
        })
    return results


# ---------------------------------------------------------------------------
# Top view
# ---------------------------------------------------------------------------


def top_view(cfg: RootConfig, all_machines: bool = False) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for phase in ALL_PHASES:
        ids = load_index(cfg, "state", phase)
        counts[phase] = len(ids)

    machines_list = list_machines(cfg)

    recent_events = query_events(cfg, limit=10)

    result: dict[str, Any] = {
        "counts": counts,
        "machines": machines_list if all_machines else [
            m for m in machines_list if m["machine_name"] == cfg.machine_name
        ],
        "recent_events": recent_events,
    }

    # Try to include GPU state for current machine
    gp = gpu_state_path(cfg)
    if gp.is_file():
        try:
            result["gpu"] = read_json(gp)
        except Exception:
            pass

    return result
