"""Truth-derived qexp projections."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config_types import RootConfig
from .runtime.dependencies import dependency_gate
from .runtime.group_namespace import group_directory, read_group
from .runtime.locks import schema_reader_lock
from .runtime.paths import (
    attempt_path,
    group_path,
    machine_path,
    shared_log_path,
    shared_paths,
    submission_path,
    task_path,
)
from .runtime.progress import inspect_progress
from .runtime.progress_types import ProgressObservation
from .runtime.records import AttemptRecord, TaskRecord, normalize_group_record
from .runtime.resources.reservations import reservation_snapshot
from .runtime.store import iter_json, read_json
from .runtime.tasks import load_task

_TERMINAL_PHASES = frozenset({"succeeded", "failed", "cancelled"})


class CurrentObservationError(RuntimeError):
    """Raised when a selected current Attempt is malformed or unreadable."""


def _load_observation_task(cfg: RootConfig, task_id: str) -> TaskRecord:
    """Load Task truth while translating malformed records to observation errors."""
    try:
        return load_task(cfg, task_id)
    except FileNotFoundError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise CurrentObservationError(f"Task {task_id!r} is malformed or unreadable.") from exc


def _task_selection_signature(task: TaskRecord) -> tuple[Any, ...]:
    """Return the Task fields that fence one current observation selection."""
    claim = task.claim_control.get("active_claim") or {}
    return (
        task.meta.get("revision"),
        task.state.get("projection"),
        task.attempt_control.get("current_attempt_id"),
        task.attempt_control.get("current_attempt_number"),
        claim.get("attempt_id"),
        claim.get("attempt_number"),
        claim.get("fencing_token"),
        claim.get("machine_name"),
    )


def _unavailable_current_view(task: TaskRecord, reason: str) -> dict[str, Any]:
    """Build an explicit frame when a Task transition prevents selection."""
    revision = task.meta.get("revision")
    progress: ProgressObservation = {
        "status": "unavailable",
        "observation_state": "unavailable",
        "reason": "identity_mismatch",
    }
    return {
        "task_id": task.task_id,
        "name": task.name,
        "phase": task.state["projection"],
        "reason": task.state.get("reason"),
        "revision": revision,
        "terminal": task.state["projection"] in _TERMINAL_PHASES,
        "observation_state": "unavailable",
        "observation_reason": reason,
        "selected_attempt": None,
        "progress": progress,
    }


def _read_current_attempt(cfg: RootConfig, task: TaskRecord) -> tuple[AttemptRecord | None, str | None]:
    """Read the one Attempt permitted by Task truth, without history enumeration."""
    projection = task.state.get("projection")
    current_id = task.attempt_control.get("current_attempt_id")
    current_number = task.attempt_control.get("current_attempt_number")
    reason = task.state.get("reason")

    if projection == "queued" and current_id is None:
        return None, None
    if (
        current_id is None
        and projection not in _TERMINAL_PHASES
        and not (projection == "blocked" and reason == "orphaned_attempt_requires_recovery")
    ):
        return None, "identity_mismatch"
    if type(current_number) is not int or current_number < 1:
        return None, "identity_mismatch"

    path = attempt_path(cfg.shared_root, task.task_id, current_number)
    try:
        attempt = AttemptRecord.from_dict(read_json(path))
    except FileNotFoundError as exc:
        raise CurrentObservationError(
            f"selected Attempt {task.task_id!r} number {current_number} was not found."
        ) from exc
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise CurrentObservationError(
            f"selected Attempt {task.task_id!r} number {current_number} is malformed or unreadable."
        ) from exc

    if attempt.task_id != task.task_id or attempt.attempt_number != current_number:
        raise CurrentObservationError(
            f"selected Attempt {task.task_id!r} number {current_number} has mismatched identity."
        )
    if current_id is not None and attempt.attempt_id != current_id:
        raise CurrentObservationError(f"selected Attempt {task.task_id!r} does not match current_attempt_id.")

    if projection == "running":
        claim = task.claim_control.get("active_claim") or {}
        if (
            current_id is None
            or attempt.attempt_id != current_id
            or attempt.phase not in {"starting", "running"}
            or claim.get("attempt_id") != attempt.attempt_id
            or claim.get("attempt_number") != attempt.attempt_number
            or claim.get("fencing_token") != attempt.current_fencing_token
            or claim.get("machine_name") != attempt.machine_name
        ):
            raise CurrentObservationError(
                f"running Task {task.task_id!r} has inconsistent active claim or Attempt identity."
            )
    elif projection in _TERMINAL_PHASES:
        if attempt.phase != projection:
            raise CurrentObservationError(f"terminal Task {task.task_id!r} does not match its preserved Attempt.")
    elif projection == "blocked":
        if current_id is None and attempt.phase != "orphaned":
            return None, "identity_mismatch"
        if attempt.phase not in {"claimed", "starting", "running", "orphaned"}:
            return None, "identity_mismatch"
    elif projection == "queued":
        if attempt.phase not in {"claimed", "starting", "running"}:
            return None, "identity_mismatch"
    else:
        return None, "identity_mismatch"
    return attempt, None


def _selected_attempt_view(cfg: RootConfig, task: TaskRecord, attempt: AttemptRecord) -> dict[str, Any]:
    """Return the compact Attempt descriptor used by both continuous viewers."""
    process = attempt.process
    if not isinstance(process, dict):
        raise CurrentObservationError(f"Attempt {attempt.attempt_id!r} process data is malformed.")
    references = process.get("log_references")
    if references is None:
        references = []
    if not isinstance(references, list):
        raise CurrentObservationError(f"Attempt {attempt.attempt_id!r} log references are malformed.")
    if references:
        reference = references[0]
        if not isinstance(reference, str) or not reference or "\x00" in reference:
            raise CurrentObservationError(f"Attempt {attempt.attempt_id!r} has an invalid log reference.")
        log_path = Path(reference)
    else:
        log_path = shared_log_path(cfg.shared_root, task.task_id, attempt.attempt_id)
    return {
        "attempt_id": attempt.attempt_id,
        "attempt_number": attempt.attempt_number,
        "phase": attempt.phase,
        "machine_name": attempt.machine_name,
        "log_path": str(log_path),
    }


def inspect_current_task(cfg: RootConfig, task_id: str) -> dict[str, Any]:
    """Collect one bounded, identity-checked current Task observation."""
    task = _load_observation_task(cfg, task_id)
    first_signature = _task_selection_signature(task)
    try:
        attempt, selection_reason = _read_current_attempt(cfg, task)
    except CurrentObservationError:
        latest_task = _load_observation_task(cfg, task_id)
        if _task_selection_signature(latest_task) != first_signature:
            return _unavailable_current_view(latest_task, "concurrent_transition")
        raise
    try:
        latest_task = _load_observation_task(cfg, task_id)
    except FileNotFoundError:
        raise
    if _task_selection_signature(latest_task) != first_signature:
        return _unavailable_current_view(latest_task, "concurrent_transition")
    if attempt is None:
        if selection_reason == "identity_mismatch":
            progress: ProgressObservation = {
                "status": "unavailable",
                "observation_state": "unavailable",
                "reason": "identity_mismatch",
            }
        else:
            progress = inspect_progress(cfg, latest_task)
        return {
            "task_id": latest_task.task_id,
            "name": latest_task.name,
            "phase": latest_task.state["projection"],
            "reason": latest_task.state.get("reason"),
            "revision": latest_task.meta.get("revision"),
            "terminal": latest_task.state["projection"] in _TERMINAL_PHASES,
            "observation_state": progress["observation_state"],
            "observation_reason": progress.get("reason"),
            "selected_attempt": None,
            "progress": progress,
        }

    selected = _selected_attempt_view(cfg, latest_task, attempt)
    progress = inspect_progress(cfg, latest_task)
    return {
        "task_id": latest_task.task_id,
        "name": latest_task.name,
        "phase": latest_task.state["projection"],
        "reason": latest_task.state.get("reason"),
        "revision": latest_task.meta.get("revision"),
        "terminal": latest_task.state["projection"] in _TERMINAL_PHASES,
        "observation_state": progress["observation_state"],
        "observation_reason": progress.get("reason"),
        "selected_attempt": selected,
        "progress": progress,
    }


def _task_view(task: TaskRecord) -> dict[str, Any]:
    claim = task.claim_control.get("active_claim") or {}
    return {
        "task_id": task.task_id,
        "name": task.name,
        "group": task.group_name,
        "phase": task.state["projection"],
        "reason": task.state.get("reason"),
        "gpus": task.spec.requested_gpus,
        "home_machine": task.placement_policy["home_machine"],
        "queue_scope": task.placement_runtime["queue_scope"],
        "current_attempt_id": task.attempt_control["current_attempt_id"],
        "claim_machine": claim.get("machine_name"),
    }


def list_tasks(
    cfg: RootConfig, *, phase: str | None = None, group: str | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    """Return the first matching Task views without scanning past a positive limit."""
    if limit == 0:
        return []
    result = []
    for path in iter_json(shared_paths(cfg.shared_root)["tasks"]):
        task = TaskRecord.from_dict(read_json(path))
        view = _task_view(task)
        if phase and view["phase"] != phase:
            continue
        if group and view["group"] != group:
            continue
        gate = dependency_gate(cfg, task)
        view["depends_on_task_ids"] = task.depends_on_task_ids
        view["dependency_state"] = gate.state
        view["dependency_reasons"] = list(gate.reasons)
        result.append(view)
        if limit > 0 and len(result) >= limit:
            break
    return result[:limit]


def list_tasks_page(
    cfg: RootConfig,
    *,
    phase: str | None = None,
    group: str | None = None,
    page_size: int = 50,
    cursor: str | None = None,
) -> dict[str, Any]:
    """Return one live, indexed page of Task views."""
    from .runtime.observation.api import list_tasks_page as _list_tasks_page

    return _list_tasks_page(cfg, phase=phase, group=group, page_size=page_size, cursor=cursor)


def inspect_task(cfg: RootConfig, task_id: str) -> dict[str, Any]:
    task = load_task(cfg, task_id)
    result = task.to_dict()
    progress: ProgressObservation = inspect_progress(cfg, task)
    result["progress"] = progress
    gate = dependency_gate(cfg, task)
    result["dependency_gate"] = {"state": gate.state, "reasons": list(gate.reasons)}
    attempts_dir = shared_paths(cfg.shared_root)["attempts"] / task_id
    result["attempts"] = [read_json(path) for path in iter_json(attempts_dir)]
    operation_id = task.submission_operation_id
    if operation_id:
        operation_path = submission_path(cfg.shared_root, operation_id)
        if operation_path.exists():
            submission = read_json(operation_path).get("submission", {})
            result["submission"] = {
                "operation_id": operation_id,
                "original_submitting_machine": submission.get("original_submitting_machine"),
            }
    return result


def list_groups(cfg: RootConfig) -> list[dict[str, Any]]:
    groups = []
    with schema_reader_lock(cfg.shared_root):
        for path in iter_json(group_directory(cfg.shared_root)):
            group = read_json(path)
            normalize_group_record(group)
            groups.append(group)
    return groups


def list_group_machines(cfg: RootConfig, name: str, *, reservation_runtime_root: Path | None = None) -> dict[str, Any]:
    """Return normalized Worker roles, usage, limits, and machine observations."""
    group = read_group(cfg.shared_root, name)
    normalize_group_record(group)
    machines = []
    for machine, worker in sorted(group["group"]["worker_set"].items()):
        summary_path = shared_paths(cfg.shared_root)["machines"] / machine / "state" / "summary.json"
        reservations: list[dict[str, Any]] = []
        agent_state = "unknown"
        if summary_path.exists():
            try:
                summary = read_json(summary_path).get("summary", {})
                reservations = [item for item in summary.get("machine_reservations", []) if isinstance(item, dict)]
                agent_state = summary.get("agent_state", "unknown")
            except (OSError, KeyError, TypeError, ValueError):
                agent_state = "unknown"
        if machine == cfg.machine_name and reservation_runtime_root is not None:
            try:
                reservations = list(reservation_snapshot(reservation_runtime_root).reservations)
                reservations = [
                    item for item in reservations if item.get("shared_root") in {None, str(cfg.shared_root)}
                ]
            except (OSError, KeyError, TypeError, ValueError):
                pass
        usage = sum(len(item.get("gpu_ids", [])) for item in reservations if item.get("group_name") == name)
        limit = worker["gpu_limit_gpus"]
        state = worker["state"]
        if limit is not None and usage > limit:
            state = "over_limit"
        elif limit is not None and usage >= limit:
            state = "full"
        machines.append(
            {
                "machine_name": machine,
                "scheduling_role": worker["scheduling_role"],
                "gpu_usage": usage,
                "gpu_limit_gpus": limit,
                "state": state,
                "agent": "registered" if agent_state in {"active", "idle"} else agent_state,
            }
        )
    return {"group": name, "machines": machines}


def list_machines(cfg: RootConfig) -> list[dict[str, Any]]:
    machines = shared_paths(cfg.shared_root)["machines"]
    return [_machine_view(cfg, path) for path in sorted(machines.glob("*/machine.json"))]


def _machine_view(cfg: RootConfig, machine_path_value: Any) -> dict[str, Any]:
    record = read_json(machine_path_value)
    machine_name = record.get("machine", {}).get("machine_name")
    if not isinstance(machine_name, str):
        return record
    state_dir = shared_paths(cfg.shared_root)["machines"] / machine_name / "state"
    state = {
        name: read_json(path) for name in ("agent", "gpu", "summary") if (path := state_dir / f"{name}.json").exists()
    }
    agent = state.get("agent", {})
    heartbeat = agent.get("heartbeat_at") if isinstance(agent, dict) else None
    interval = agent.get("heartbeat_interval_seconds") if isinstance(agent, dict) else None
    if isinstance(heartbeat, str) and isinstance(interval, (int, float)):
        try:
            elapsed = (
                datetime.now(timezone.utc) - datetime.fromisoformat(heartbeat.replace("Z", "+00:00"))
            ).total_seconds()
            state["freshness"] = "stale" if elapsed > interval * 3 else "fresh"
        except ValueError:
            state["freshness"] = "unknown"
    else:
        state["freshness"] = "unknown"
    record["state"] = state
    return record


def top_view(cfg: RootConfig, *, all_machines: bool = False) -> dict[str, Any]:
    tasks = list_tasks(cfg, limit=10**9)
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task["phase"]] = counts.get(task["phase"], 0) + 1
    return {
        "counts": counts,
        "tasks": tasks if all_machines else [t for t in tasks if t["home_machine"] == cfg.machine_name],
        "machines": list_machines(cfg),
    }
