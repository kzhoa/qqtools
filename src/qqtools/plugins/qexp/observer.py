from __future__ import annotations

from datetime import datetime, timezone
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
    load_resubmit_operation,
    load_batch,
    load_task,
    read_json,
)
from .layout import gpu_state_path

GPU_STATE_FRESHNESS_SECONDS = 30


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

    results: list[dict[str, Any]] = []
    if phase:
        for task_id in load_index(cfg, "state", phase):
            try:
                t = load_task(cfg, task_id)
            except FileNotFoundError:
                continue
            if t.status.phase != phase:
                continue
            if batch_id and t.batch_id != batch_id:
                continue
            if validated_group and t.group != validated_group:
                continue
            if machine and t.machine_name != machine:
                continue
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
            if len(results) >= limit:
                break
        return results

    for t in iter_all_tasks(cfg):
        if batch_id and t.batch_id != batch_id:
            continue
        if validated_group and t.group != validated_group:
            continue
        if machine and t.machine_name != machine:
            continue
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
        if len(results) >= limit:
            break
    return results


def inspect_task(cfg: RootConfig, task_id: str) -> dict[str, Any]:
    try:
        t = load_task(cfg, task_id)
    except FileNotFoundError as exc:
        try:
            operation = load_resubmit_operation(cfg, task_id)
        except FileNotFoundError:
            raise exc
        return {
            "task": None,
            "operation": operation.to_dict()["operation"],
            "meta": operation.meta.to_dict(),
            "message": (
                f"Task {task_id} has no visible task truth because an unfinished "
                f"resubmit operation is converging. Run 'qexp doctor repair' if needed."
            ),
        }
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


def _parse_iso_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _is_gpu_snapshot_stale(updated_at: str | None, now: datetime | None = None) -> bool:
    updated_dt = _parse_iso_timestamp(updated_at)
    if updated_dt is None:
        return True
    current = now or datetime.now(timezone.utc)
    return (current - updated_dt).total_seconds() > GPU_STATE_FRESHNESS_SECONDS


def _fallback_gpu_count(machine) -> int | None:
    if machine.gpu_inventory.visible_gpu_ids:
        return len(machine.gpu_inventory.visible_gpu_ids)
    if machine.gpu_inventory.count > 0:
        return machine.gpu_inventory.count
    return None


def _read_gpu_snapshot(machine_cfg: RootConfig) -> dict[str, Any] | None:
    path = gpu_state_path(machine_cfg)
    if not path.is_file():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def _normalize_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("GPU snapshot field must be a list.")
    normalized: list[int] = []
    for item in value:
        if not isinstance(item, int):
            raise ValueError("GPU snapshot list items must be integers.")
        normalized.append(item)
    return normalized


def _build_machine_gpu_view(machine_cfg: RootConfig, machine) -> dict[str, Any]:
    gpu_snapshot = _read_gpu_snapshot(machine_cfg)
    fallback_gpu_count = _fallback_gpu_count(machine)

    gpu_visible_ids: list[int] = []
    gpu_reserved_ids: list[int] = []
    gpu_backend = None
    gpu_probe_error = None
    gpu_updated_at = None
    gpu_count: int | None = None
    gpu_status = "unknown"

    if gpu_snapshot is not None:
        try:
            gpu_visible_ids = _normalize_int_list(gpu_snapshot.get("visible_gpu_ids"))
            gpu_reserved_ids = _normalize_int_list(gpu_snapshot.get("reserved_gpu_ids"))
            gpu_backend = gpu_snapshot.get("backend")
            gpu_probe_error = gpu_snapshot.get("probe_error")
            gpu_updated_at = gpu_snapshot.get("updated_at")

            snapshot_count = gpu_snapshot.get("gpu_count")
            if isinstance(snapshot_count, int):
                gpu_count = snapshot_count
            elif gpu_visible_ids:
                gpu_count = len(gpu_visible_ids)

            if gpu_snapshot.get("probe_succeeded") is False:
                gpu_status = "error"
                if gpu_count is None:
                    gpu_count = fallback_gpu_count
            elif _is_gpu_snapshot_stale(gpu_updated_at):
                gpu_status = "stale"
            else:
                gpu_status = "live"
        except Exception:
            gpu_snapshot = None

    if gpu_snapshot is None and fallback_gpu_count is not None:
        gpu_count = fallback_gpu_count
        gpu_status = "fallback"

    gpu_visible_count = gpu_count
    gpu_reserved_count = (
        len(gpu_reserved_ids)
        if gpu_snapshot is not None and gpu_status in {"live", "stale"}
        else None
    )
    gpu_free_count = None
    if isinstance(gpu_visible_count, int) and isinstance(gpu_reserved_count, int):
        gpu_free_count = max(gpu_visible_count - gpu_reserved_count, 0)

    return {
        "gpu_count": gpu_count,
        "gpu_status": gpu_status,
        "gpu_visible_ids": gpu_visible_ids,
        "gpu_reserved_ids": gpu_reserved_ids,
        "gpu_backend": gpu_backend,
        "gpu_probe_error": gpu_probe_error,
        "gpu_updated_at": gpu_updated_at,
        "gpu_visible_count": gpu_visible_count,
        "gpu_reserved_count": gpu_reserved_count,
        "gpu_free_count": gpu_free_count,
    }


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
        gpu_view = _build_machine_gpu_view(machine_cfg, m)
        results.append({
            "machine_name": m.machine_name,
            "hostname": m.hostname,
            "agent_mode": m.agent_mode,
            "agent_state": agent_state,
            "last_heartbeat": agent_snapshot.last_heartbeat if agent_snapshot else None,
            "idle_deadline_at": agent_snapshot.idle_deadline_at if agent_snapshot else None,
            "drain_started_at": agent_snapshot.drain_started_at if agent_snapshot else None,
            "counts_by_phase": counts_by_phase,
            "workset": workset.to_dict(),
            **gpu_view,
        })
    return results


# ---------------------------------------------------------------------------
# Top view
# ---------------------------------------------------------------------------


def top_view(cfg: RootConfig, all_machines: bool = False) -> dict[str, Any]:
    counts: dict[str, int] = {phase: 0 for phase in ALL_PHASES}
    for task in iter_all_tasks(cfg):
        counts[task.status.phase] = counts.get(task.status.phase, 0) + 1

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
