"""Durable per-Task live-progress choices owned by Submission Operations."""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

from .runtime.paths import submission_path
from .runtime.records import validate_group_name, validate_identifier
from .runtime.store import read_json
from .runtime.tasks import load_task

LIVE_PROGRESS_SELECTION_VERSION = 1
_SELECTION_FIELDS = frozenset({"version", "selection_digest", "tasks"})
_TASK_FIELDS = frozenset({"task_id", "requested", "enabled", "source", "group_policy"})
_GROUP_POLICY_FIELDS = frozenset({"group_identity", "revision"})
_GROUP_IDENTITY_FIELDS = frozenset({"name", "created_at", "creation_operation_id"})
_SOURCE_VALUES = frozenset({"explicit", "group", "default", "unavailable"})
_MAX_POLICY_REVISION = (1 << 63) - 1
_DIAGNOSTIC_LIMIT = 160
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def validate_live_progress_request(value: Any, label: str = "live_progress") -> bool | None:
    """Validate the nullable, exact-boolean Task request value."""
    if value is not None and type(value) is not bool:
        raise ValueError(f"{label} must be a boolean or null.")
    return value


def _selection_digest(tasks: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        list(tasks),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_group_identity(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != _GROUP_IDENTITY_FIELDS:
        raise ValueError(f"{label} must contain exactly name, created_at, and creation_operation_id.")
    name = validate_group_name(value.get("name"))
    if name is None:
        raise ValueError(f"{label}.name must be a non-null Group name.")
    created_at = value.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise ValueError(f"{label}.created_at must be a non-empty string.")
    creation_operation_id = value.get("creation_operation_id")
    if creation_operation_id is not None:
        validate_identifier(creation_operation_id, f"{label}.creation_operation_id")
    return {
        "name": name,
        "created_at": created_at,
        "creation_operation_id": creation_operation_id,
    }


def validate_live_progress_selection(
    value: Any,
    task_ids: Sequence[str] | None = None,
    *,
    group_name: str | None = None,
) -> dict[str, Any]:
    """Strictly validate and normalize one Operation selection namespace."""
    if not isinstance(value, Mapping) or frozenset(value) != _SELECTION_FIELDS:
        raise ValueError("live_progress_selection has invalid fields.")
    if type(value.get("version")) is not int or value["version"] != LIVE_PROGRESS_SELECTION_VERSION:
        raise ValueError("live_progress_selection version is unsupported.")

    raw_tasks = value.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("live_progress_selection.tasks must be a non-empty list.")
    normalized_tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_task in enumerate(raw_tasks):
        label = f"live_progress_selection.tasks[{index}]"
        if not isinstance(raw_task, Mapping) or frozenset(raw_task) != _TASK_FIELDS:
            raise ValueError(f"{label} has invalid fields.")
        task_id = raw_task.get("task_id")
        validate_identifier(task_id, f"{label}.task_id")
        if task_id in seen:
            raise ValueError("live_progress_selection Task IDs must be unique.")
        seen.add(task_id)

        requested = validate_live_progress_request(raw_task.get("requested"), f"{label}.requested")
        enabled = raw_task.get("enabled")
        if type(enabled) is not bool:
            raise ValueError(f"{label}.enabled must be a boolean.")
        source = raw_task.get("source")
        if not isinstance(source, str) or source not in _SOURCE_VALUES:
            raise ValueError(f"{label}.source is invalid.")

        group_policy = raw_task.get("group_policy")
        if source == "explicit":
            if requested is None or enabled is not requested or group_policy is not None:
                raise ValueError(f"{label} has contradictory explicit selection fields.")
            normalized_policy = None
        elif source == "group":
            if requested is not None:
                raise ValueError(f"{label}.requested must be null for Group selection.")
            if not isinstance(group_policy, Mapping) or frozenset(group_policy) != _GROUP_POLICY_FIELDS:
                raise ValueError(f"{label}.group_policy is invalid.")
            identity = _validate_group_identity(
                group_policy.get("group_identity"), f"{label}.group_policy.group_identity"
            )
            if group_name is None or identity["name"] != group_name:
                raise ValueError(f"{label}.group_policy identity does not match the submission Group.")
            revision = group_policy.get("revision")
            if type(revision) is not int or not 1 <= revision <= _MAX_POLICY_REVISION:
                raise ValueError(f"{label}.group_policy.revision is invalid.")
            normalized_policy = {"group_identity": identity, "revision": revision}
        else:
            if requested is not None or enabled or group_policy is not None:
                raise ValueError(f"{label} has contradictory inherited selection fields.")
            normalized_policy = None

        normalized_tasks.append(
            {
                "task_id": task_id,
                "requested": requested,
                "enabled": enabled,
                "source": source,
                "group_policy": normalized_policy,
            }
        )

    if task_ids is not None:
        if not isinstance(task_ids, Sequence) or isinstance(task_ids, (str, bytes)):
            raise ValueError("submission task_ids must be a sequence of Task IDs.")
        expected_ids = list(task_ids)
        for index, task_id in enumerate(expected_ids):
            validate_identifier(task_id, f"submission task_ids[{index}]")
        if [item["task_id"] for item in normalized_tasks] != expected_ids:
            raise ValueError("live_progress_selection Task IDs do not match submission task order.")

    digest = value.get("selection_digest")
    if not isinstance(digest, str) or not _DIGEST_PATTERN.fullmatch(digest):
        raise ValueError("live_progress_selection selection_digest is invalid.")
    if digest != _selection_digest(normalized_tasks):
        raise ValueError("live_progress_selection selection_digest does not match tasks.")
    return {
        "version": LIVE_PROGRESS_SELECTION_VERSION,
        "selection_digest": digest,
        "tasks": normalized_tasks,
    }


def build_live_progress_selection(
    tasks: Sequence[Mapping[str, Any]], *, group_name: str | None = None
) -> dict[str, Any]:
    """Build a version-1 selection and its digest from resolved Task entries."""
    normalized_tasks = [dict(item) for item in tasks]
    value = {
        "version": LIVE_PROGRESS_SELECTION_VERSION,
        "selection_digest": _selection_digest(normalized_tasks),
        "tasks": normalized_tasks,
    }
    return validate_live_progress_selection(value, group_name=group_name)


def disabled_live_progress_selection(task_ids: Sequence[str]) -> dict[str, Any]:
    """Return the safe all-off advisory selection for historical or invalid metadata."""
    if not isinstance(task_ids, Sequence) or isinstance(task_ids, (str, bytes)):
        raise ValueError("submission task_ids must be a sequence of Task IDs.")
    tasks = [
        {
            "task_id": task_id,
            "requested": None,
            "enabled": False,
            "source": "default",
            "group_policy": None,
        }
        for task_id in task_ids
    ]
    return build_live_progress_selection(tasks)


def _bounded_diagnostic(value: object) -> str:
    try:
        message = str(value)
    except Exception:
        message = "live-progress selection is unavailable"
    safe_message = " ".join("".join(char if char.isprintable() else " " for char in message).split())
    return (safe_message or "live-progress selection is unavailable")[:_DIAGNOSTIC_LIMIT]


def _warn_disabled(task_id: str, reason: object, *, emit: bool = True) -> None:
    if not emit:
        return
    try:
        warnings.warn(
            _bounded_diagnostic(f"Task {task_id!r} live progress is disabled: {reason}"),
            RuntimeWarning,
            stacklevel=3,
        )
    except Exception:
        pass


def warn_invalid_live_progress_selection(reason: object) -> None:
    """Report a bounded diagnostic when advisory Operation metadata is ignored."""
    try:
        warnings.warn(
            f"Submission live-progress selection is disabled: {_bounded_diagnostic(reason)}",
            RuntimeWarning,
            stacklevel=2,
        )
    except Exception:
        pass


def warn_live_progress_policy_unavailable(reason: object) -> None:
    """Report a bounded diagnostic when inherited policy falls back to false."""
    try:
        warnings.warn(
            f"Group live-progress policy unavailable; inherited Tasks are disabled: {_bounded_diagnostic(reason)}",
            RuntimeWarning,
            stacklevel=2,
        )
    except Exception:
        pass


def read_task_live_progress(cfg: Any, task_id: str, *, warn: bool = True) -> bool:
    """Read a Task's frozen advisory choice; invalid or historical metadata means off."""
    try:
        validate_identifier(task_id, "task_id")
    except (TypeError, ValueError) as exc:
        _warn_disabled(str(task_id), exc, emit=warn)
        return False
    try:
        task = load_task(cfg, task_id)
    except Exception as exc:
        _warn_disabled(task_id, f"Task record is unavailable: {type(exc).__name__}: {exc}", emit=warn)
        return False
    if task.task_id != task_id:
        _warn_disabled(task_id, "Task record identity does not match the requested Task", emit=warn)
        return False

    operation_id = task.submission_operation_id
    if operation_id is None:
        _warn_disabled(task_id, "Task has no Submission Operation identity", emit=warn)
        return False
    try:
        validate_identifier(operation_id, "submission_operation_id")
        operation = read_json(submission_path(cfg.shared_root, operation_id))
    except Exception as exc:
        _warn_disabled(task_id, f"Submission Operation is unavailable: {type(exc).__name__}: {exc}", emit=warn)
        return False

    submission = operation.get("submission")
    if not isinstance(submission, Mapping) or submission.get("operation_id") != operation_id:
        _warn_disabled(task_id, "Submission Operation identity does not match the Task", emit=warn)
        return False
    if "live_progress_selection" not in operation:
        _warn_disabled(task_id, "historical Submission Operation has no live-progress selection", emit=warn)
        return False

    context = submission.get("resolved_context")
    task_ids = context.get("task_ids") if isinstance(context, Mapping) else None
    if not isinstance(task_ids, list):
        _warn_disabled(task_id, "Submission Operation has invalid Task identity context", emit=warn)
        return False
    try:
        selection = validate_live_progress_selection(
            operation["live_progress_selection"],
            task_ids,
            group_name=submission.get("target_group"),
        )
    except Exception as exc:
        _warn_disabled(task_id, exc, emit=warn)
        return False

    for item in selection["tasks"]:
        if item["task_id"] == task_id:
            return item["enabled"]
    _warn_disabled(task_id, "Submission Operation selection has no matching Task", emit=warn)
    return False
