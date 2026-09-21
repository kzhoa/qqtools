"""Durable per-Task observer choices and effective tmux decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .runtime.paths import submission_path
from .runtime.records import validate_identifier
from .runtime.store import read_json
from .runtime.tasks import load_task
from .tmux_policy import resolve_tmux_policy

TASK_OBSERVATION_VERSION = 1
_TASK_OBSERVATION_FIELDS = frozenset({"version", "tasks"})
_TASK_FIELDS = frozenset({"task_id", "tmux_override"})
_DIAGNOSTIC_LIMIT = 160


def validate_tmux_override(value: Any, label: str = "tmux_override") -> bool | None:
    """Validate a nullable, exact boolean observer override."""
    if value is not None and type(value) is not bool:
        raise ValueError(f"{label} must be a boolean or null.")
    return value


def build_task_observation(task_ids: Sequence[str], overrides: Sequence[bool | None]) -> dict[str, Any]:
    """Build strict top-level Submission Operation observation metadata."""
    if not isinstance(task_ids, Sequence) or isinstance(task_ids, (str, bytes)):
        raise ValueError("task_observation task_ids must be a sequence.")
    if not isinstance(overrides, Sequence) or isinstance(overrides, (str, bytes)):
        raise ValueError("task_observation overrides must be a sequence.")
    if len(task_ids) != len(overrides) or not task_ids:
        raise ValueError("task_observation task IDs and overrides must be non-empty and have equal length.")
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, (task_id, override) in enumerate(zip(task_ids, overrides, strict=True)):
        validate_identifier(task_id, f"task_observation.tasks[{index}].task_id")
        if task_id in seen:
            raise ValueError("task_observation task IDs must be unique.")
        seen.add(task_id)
        tasks.append(
            {
                "task_id": task_id,
                "tmux_override": validate_tmux_override(override, f"task_observation.tasks[{index}].tmux_override"),
            }
        )
    return {"version": TASK_OBSERVATION_VERSION, "tasks": tasks}


def validate_task_observation(value: Any, task_ids: Sequence[str] | None = None) -> dict[str, Any]:
    """Validate and normalize operation metadata, optionally fencing its Task identity."""
    if not isinstance(value, Mapping) or frozenset(value) != _TASK_OBSERVATION_FIELDS:
        raise ValueError("task_observation has invalid fields.")
    if type(value.get("version")) is not int or value["version"] != TASK_OBSERVATION_VERSION:
        raise ValueError("task_observation version is unsupported.")
    raw_tasks = value.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("task_observation.tasks must be a non-empty list.")
    expected_ids = None if task_ids is None else list(task_ids)
    result_tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_task in enumerate(raw_tasks):
        if not isinstance(raw_task, Mapping) or frozenset(raw_task) != _TASK_FIELDS:
            raise ValueError(f"task_observation.tasks[{index}] has invalid fields.")
        task_id = raw_task.get("task_id")
        validate_identifier(task_id, f"task_observation.tasks[{index}].task_id")
        if task_id in seen:
            raise ValueError("task_observation task IDs must be unique.")
        seen.add(task_id)
        result_tasks.append(
            {
                "task_id": task_id,
                "tmux_override": validate_tmux_override(
                    raw_task.get("tmux_override"), f"task_observation.tasks[{index}].tmux_override"
                ),
            }
        )
    if expected_ids is not None:
        if result_tasks and expected_ids != [item["task_id"] for item in result_tasks]:
            raise ValueError("task_observation task IDs do not match submission task order.")
    return {"version": TASK_OBSERVATION_VERSION, "tasks": result_tasks}


def decode_task_observation(value: Any, task_ids: Sequence[str]) -> dict[str, Any]:
    """Decode metadata while requiring exact identity with a submission plan."""
    return validate_task_observation(value, task_ids)


def _operation_for_task(cfg: Any, task_id: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
    task = load_task(cfg, task_id)
    operation_id = task.submission_operation_id
    if operation_id is None:
        return None
    try:
        validate_identifier(operation_id, "submission_operation_id")
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Task {task_id!r} has an invalid submission operation identity.") from exc
    path = submission_path(cfg.shared_root, operation_id)
    try:
        operation = read_json(path)
    except FileNotFoundError as exc:
        raise RuntimeError(f"submission operation {operation_id!r} is unreadable.") from exc
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f"submission operation {operation_id!r} is unreadable.") from exc
    submission = operation.get("submission")
    if not isinstance(submission, dict) or submission.get("operation_id") != operation_id:
        raise RuntimeError(f"Task {task_id!r} has mismatched submission operation identity.")
    return operation, submission


def read_task_tmux_override(cfg: Any, task_id: str) -> bool | None:
    """Read the stored Task override, with historical metadata inheriting."""
    loaded = _operation_for_task(cfg, task_id)
    if loaded is None:
        return None
    operation, submission = loaded
    if "task_observation" not in operation:
        return None
    metadata = operation["task_observation"]
    # Operations written before this additive boundary intentionally inherit.
    context = submission.get("resolved_context")
    task_ids = context.get("task_ids") if isinstance(context, Mapping) else None
    if not isinstance(task_ids, list):
        raise RuntimeError("submission operation has invalid task identity context.")
    decoded = decode_task_observation(metadata, task_ids)
    for item in decoded["tasks"]:
        if item["task_id"] == task_id:
            return item["tmux_override"]
    raise RuntimeError(f"submission operation has no observation choice for Task {task_id!r}.")


def stored_tmux_override(cfg: Any, task_id: str) -> bool | None:
    """Return a Task's persisted nullable override."""
    return read_task_tmux_override(cfg, task_id)


def tmux_override_label(value: bool | None) -> str:
    """Render a nullable override as its stable public label."""
    validate_tmux_override(value)
    return "inherit" if value is None else ("enabled" if value else "disabled")


def task_tmux_observation_label(cfg: Any, task_id: str) -> str:
    """Read and render a Task's stored observer choice."""
    return tmux_override_label(read_task_tmux_override(cfg, task_id))


def _diagnostic(value: Any) -> str:
    return str(value).replace("\x00", "\\0").replace("\r", "\\r").replace("\n", "\\n")[:_DIAGNOSTIC_LIMIT]


def resolve_task_tmux_observation(cfg: Any, task_id: str) -> dict[str, Any]:
    """Resolve one Task's effective observer decision exactly once per caller."""
    try:
        override = read_task_tmux_override(cfg, task_id)
    except Exception as exc:
        return {
            "enabled": False,
            "source": "default",
            "diagnostic_reason": _diagnostic(f"stored override: {type(exc).__name__}: {exc}"),
        }
    if override is not None:
        return {"enabled": override, "source": "task_override"}
    policy = resolve_tmux_policy(cfg)
    return {
        "enabled": policy["enabled"],
        "source": policy["source"],
        **({"diagnostic_reason": policy["diagnostic_reason"]} if "diagnostic_reason" in policy else {}),
    }


__all__ = [
    "TASK_OBSERVATION_VERSION",
    "build_task_observation",
    "decode_task_observation",
    "read_task_tmux_override",
    "resolve_task_tmux_observation",
    "stored_tmux_override",
    "task_tmux_observation_label",
    "tmux_override_label",
    "validate_task_observation",
    "validate_tmux_override",
]
