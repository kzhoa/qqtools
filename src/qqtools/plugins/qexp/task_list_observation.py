"""Bounded optional observations used by Task list and page views.

The ordinary Task list is intentionally kept in :mod:`observer`.  This module
owns the extra reads for explicitly selected location and progress fields so a
basic list remains a Task truth query with no observation side effects.
"""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qqtools.qexp._progress_protocol import read_advisory_snapshot

from .progress_selection import select_progress_protocol
from .runtime.paths import attempt_path, submission_path
from .runtime.progress import _running_registration_generation, _validate_projection, shared_progress_path
from .runtime.progress_v2 import _shared_path as shared_progress_v2_path
from .runtime.progress_v2 import _validate_projection_v2
from .runtime.progress_v3 import MAX_SNAPSHOT_V3_BYTES, _validate_projection_v3
from .runtime.progress_v3 import _shared_path as shared_progress_v3_path
from .runtime.records import AttemptRecord, TaskRecord, validate_identifier
from .runtime.store import read_json
from .runtime.tasks import load_task
from .task_live_progress import validate_live_progress_selection

MAX_POLICY_BYTES = 8 * 1024 * 1024
_MAX_SNAPSHOT_BYTES = 16 * 1024
_TERMINAL_PHASES = frozenset({"succeeded", "failed", "cancelled"})
_PROGRESS_FIELDS = frozenset({"overall-progress", "activity", "report-age"})
_ENRICHED_FIELDS = _PROGRESS_FIELDS | {"location"}
_IDENTITY_FIELDS = (
    "task_id",
    "attempt_id",
    "attempt_number",
    "machine_name",
    "launch_id",
    "wrapper_pid",
    "wrapper_start_time_ticks",
)


def _observation(state: str, reason: str | None, **values: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": state if state in {"available", "absent"} else "unavailable",
        "observation_state": state,
        "reason": reason,
    }
    result.update(values)
    return result


def _task_signature(task: TaskRecord) -> tuple[Any, ...]:
    claim = task.claim_control.get("active_claim")
    if not isinstance(claim, Mapping):
        claim = {}
    control = task.control if isinstance(task.control, Mapping) else {}
    return (
        task.meta.get("revision"),
        task.state.get("projection"),
        task.submission_operation_id,
        task.attempt_control.get("current_attempt_id"),
        task.attempt_control.get("current_attempt_number"),
        claim.get("attempt_id"),
        claim.get("attempt_number"),
        claim.get("fencing_token"),
        claim.get("machine_name"),
        control.get("cleanup_operation_id"),
        control.get("cleanup_state"),
    )


def _is_cleanup(task: TaskRecord) -> bool:
    return bool(task.control.get("cleanup_operation_id") or task.control.get("cleanup_state"))


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"unsupported JSON constant {value!r}")


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _read_bounded_json(path: Path, *, max_bytes: int) -> dict[str, Any]:
    """Read one regular JSON file without decoding bytes beyond its cap."""
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("record is not a regular file")
        encoded = b""
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            encoded = handle.read(max_bytes + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(encoded) > max_bytes:
        raise _OversizedPolicyError
    value = json.loads(
        encoded.decode("utf-8"),
        object_pairs_hook=_unique_pairs,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("record must be an object")
    return value


class _OversizedPolicyError(ValueError):
    """The bounded policy reader consumed more than its permitted bytes."""


def _policy_result(state: str, reason: str | None) -> dict[str, Any]:
    return {"state": state, "reason": reason}


def _policy_for_operation(
    cfg: Any,
    operation_id: str,
    task_ids: set[str],
) -> dict[str, dict[str, Any]]:
    try:
        validate_identifier(operation_id, "submission_operation_id")
    except (TypeError, ValueError):
        return {task_id: _policy_result("unknown", "policy_identity_mismatch") for task_id in task_ids}

    path = submission_path(cfg.shared_root, operation_id)
    try:
        operation = _read_bounded_json(path, max_bytes=MAX_POLICY_BYTES)
    except FileNotFoundError:
        return {task_id: _policy_result("unknown", "policy_operation_missing") for task_id in task_ids}
    except _OversizedPolicyError:
        return {task_id: _policy_result("unknown", "policy_oversized") for task_id in task_ids}
    except (OSError, PermissionError):
        return {task_id: _policy_result("unknown", "policy_read_failed") for task_id in task_ids}
    except (UnicodeError, ValueError, TypeError, RecursionError):
        return {task_id: _policy_result("unknown", "policy_invalid") for task_id in task_ids}
    except Exception:
        return {task_id: _policy_result("unknown", "policy_read_failed") for task_id in task_ids}

    submission = operation.get("submission")
    if not isinstance(submission, Mapping) or submission.get("operation_id") != operation_id:
        return {task_id: _policy_result("unknown", "policy_identity_mismatch") for task_id in task_ids}
    context = submission.get("resolved_context")
    resolved_ids = context.get("task_ids") if isinstance(context, Mapping) else None
    if not isinstance(resolved_ids, list):
        return {task_id: _policy_result("unknown", "policy_invalid") for task_id in task_ids}
    if any(not isinstance(item, str) for item in resolved_ids):
        return {task_id: _policy_result("unknown", "policy_invalid") for task_id in task_ids}
    if "live_progress_selection" not in operation:
        return {task_id: _policy_result("unknown", "policy_selection_missing") for task_id in task_ids}
    try:
        selection = validate_live_progress_selection(
            operation["live_progress_selection"],
            resolved_ids,
            group_name=submission.get("target_group"),
        )
    except (TypeError, ValueError, KeyError, AttributeError):
        return {task_id: _policy_result("unknown", "policy_invalid") for task_id in task_ids}
    except Exception:
        return {task_id: _policy_result("unknown", "policy_read_failed") for task_id in task_ids}

    entries = {item["task_id"]: item for item in selection["tasks"]}
    return {
        task_id: (
            _policy_result("enabled" if entries[task_id]["enabled"] else "disabled", None)
            if task_id in entries
            else _policy_result("unknown", "policy_identity_mismatch")
        )
        for task_id in task_ids
    }


def observe_reporting_policies(cfg: Any, tasks: Sequence[TaskRecord]) -> dict[str, dict[str, Any]]:
    """Observe frozen reporting policy once per referenced Submission Operation.

    The returned mapping retains only Task IDs from ``tasks``.  All successful
    and failed Operation observations are cached for this call, including
    bounded read failures, so sibling rows never trigger a retry.
    """
    by_operation: dict[str, set[str]] = {}
    results: dict[str, dict[str, Any]] = {}
    for task in tasks:
        task_id = task.task_id
        operation_id = task.submission_operation_id
        if operation_id is None:
            results[task_id] = _policy_result("unknown", "policy_reference_missing")
            continue
        if not isinstance(operation_id, str) or not operation_id:
            results[task_id] = _policy_result("unknown", "policy_identity_mismatch")
            continue
        by_operation.setdefault(operation_id, set()).add(task_id)

    cache: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    root_key = str(Path(cfg.shared_root).resolve())
    for operation_id, task_ids in by_operation.items():
        key = (root_key, operation_id)
        operation_result = cache.get(key)
        if operation_result is None:
            operation_result = _policy_for_operation(cfg, operation_id, task_ids)
            cache[key] = operation_result
        results.update(
            {
                task_id: operation_result.get(task_id, _policy_result("unknown", "policy_identity_mismatch"))
                for task_id in task_ids
            }
        )
    return results


def _read_attempt(cfg: Any, task: TaskRecord) -> tuple[AttemptRecord | None, str | None]:
    """Read only the Attempt selected by Task truth."""
    projection = task.state.get("projection")
    current_id = task.attempt_control.get("current_attempt_id")
    number = task.attempt_control.get("current_attempt_number")
    if _is_cleanup(task):
        return None, "cleanup"
    if projection == "queued" and current_id is None:
        return None, None
    if number is None:
        if projection in _TERMINAL_PHASES and current_id is None:
            return None, None
        return None, "identity_mismatch"
    if type(number) is not int or number < 1:
        return None, "identity_mismatch"
    if (
        current_id is None
        and projection not in _TERMINAL_PHASES
        and not (projection == "blocked" and task.state.get("reason") == "orphaned_attempt_requires_recovery")
    ):
        return None, "identity_mismatch"
    try:
        attempt = AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task.task_id, number)))
    except (FileNotFoundError, OSError, ValueError, TypeError, KeyError, AttributeError):
        return None, "attempt_unreadable"
    except Exception:
        return None, "attempt_unreadable"
    if (
        attempt.task_id != task.task_id
        or type(attempt.attempt_number) is not int
        or attempt.attempt_number < 1
        or attempt.attempt_number != number
    ):
        return None, "identity_mismatch"
    if current_id is not None and attempt.attempt_id != current_id:
        return None, "identity_mismatch"
    if projection == "running":
        claim = task.claim_control.get("active_claim") or {}
        if (
            current_id is None
            or attempt.phase not in {"starting", "running"}
            or claim.get("attempt_id") != attempt.attempt_id
            or claim.get("attempt_number") != attempt.attempt_number
            or claim.get("fencing_token") != attempt.current_fencing_token
            or claim.get("machine_name") != attempt.machine_name
        ):
            return None, "identity_mismatch"
    elif projection in _TERMINAL_PHASES:
        if attempt.phase != projection:
            return None, "identity_mismatch"
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
    try:
        validate_identifier(attempt.attempt_id, "attempt_id")
        validate_identifier(attempt.task_id, "task_id")
    except (TypeError, ValueError):
        return None, "identity_mismatch"
    return attempt, None


def _location(task: TaskRecord, attempt: AttemptRecord | None, selection_reason: str | None) -> dict[str, Any]:
    if selection_reason == "cleanup":
        return _observation("unavailable", "cleanup")
    if attempt is None:
        if selection_reason is None:
            return _observation("absent", None)
        if selection_reason == "attempt_unreadable":
            return _observation("unavailable", "attempt_unreadable")
        return _observation("unavailable", "identity_mismatch")
    projection = task.state.get("projection")
    if projection in _TERMINAL_PHASES:
        return _observation("absent", None)
    if attempt.phase not in {"claimed", "starting", "running", "orphaned"}:
        return _observation("unavailable", "identity_mismatch")
    if attempt.phase == "orphaned" and projection != "blocked":
        return _observation("unavailable", "identity_mismatch")
    gpus = attempt.assigned_gpus
    if type(gpus) is not list or any(type(gpu) is not int or gpu < 0 for gpu in gpus):
        return _observation("unavailable", "identity_mismatch")
    return _observation(
        "available",
        None,
        machine_name=attempt.machine_name,
        assigned_gpus=sorted(gpus),
        orphaned=attempt.phase == "orphaned",
    )


def _attempt_identity(task: TaskRecord, attempt: AttemptRecord) -> dict[str, Any]:
    process = attempt.process if isinstance(attempt.process, Mapping) else {}
    authorization = attempt.authorization if isinstance(attempt.authorization, Mapping) else {}
    return {
        "task_id": task.task_id,
        "attempt_id": attempt.attempt_id,
        "attempt_number": attempt.attempt_number,
        "machine_name": attempt.machine_name,
        "launch_id": authorization.get("launch_id"),
        "wrapper_pid": process.get("wrapper_pid"),
        "wrapper_start_time_ticks": process.get("wrapper_start_time_ticks"),
        "fencing_token": attempt.current_fencing_token,
    }


def _channel_result(
    cfg: Any,
    path: Path,
    identity: dict[str, Any],
    version: int,
    *,
    terminal: bool,
    validator: Any,
    max_bytes: int = _MAX_SNAPSHOT_BYTES,
) -> dict[str, Any]:
    try:
        value = read_advisory_snapshot(path, max_bytes=max_bytes)
    except FileNotFoundError:
        return _observation("no_report", "no_snapshot")
    except (OSError, PermissionError):
        return _observation("unavailable", "read_failed")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        return _observation("unavailable", "invalid_snapshot")
    except Exception:
        return _observation("unavailable", "read_failed")
    try:
        normalized = validator(
            value,
            identity,
            require_token=True,
            require_generation=not terminal,
        )
    except (TypeError, KeyError, AttributeError) as exc:
        message = str(exc)
        return _observation("unavailable", "identity_mismatch" if "identity" in message else "invalid_snapshot")
    except ValueError as exc:
        message = str(exc)
        if any(token in message for token in ("identity", "fencing", "superseded", "registration")):
            return _observation("unavailable", "identity_mismatch")
        return _observation("unavailable", "invalid_snapshot")
    except Exception:
        return _observation("unavailable", "read_failed")
    return _observation("available", None, **normalized)


def _progress_observations(
    cfg: Any,
    task: TaskRecord,
    attempt: AttemptRecord | None,
    selection_reason: str | None,
    policy: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int | None]:
    if _is_cleanup(task):
        unavailable = _observation("unavailable", "cleanup")
        return unavailable, dict(unavailable), dict(unavailable), None
    if attempt is None:
        if selection_reason is None:
            absent = _observation(
                "pending" if task.state.get("projection") == "queued" else "no_report",
                "not_started" if task.state.get("projection") == "queued" else "no_snapshot",
            )
            return absent, dict(absent), dict(absent), None
        unavailable = _observation(
            "unavailable", "read_failed" if selection_reason == "attempt_unreadable" else "identity_mismatch"
        )
        return unavailable, dict(unavailable), dict(unavailable), None

    projection = task.state.get("projection")
    terminal = projection in _TERMINAL_PHASES
    process = attempt.process if isinstance(attempt.process, Mapping) else {}
    if not terminal and (
        projection != "running"
        or attempt.phase not in {"starting", "running"}
        or type(process.get("wrapper_pid")) is not int
        or process.get("wrapper_pid") < 1
        or type(process.get("wrapper_start_time_ticks")) is not int
        or process.get("wrapper_start_time_ticks") < 0
    ):
        unavailable = _observation("unavailable", "identity_mismatch")
        return unavailable, dict(unavailable), dict(unavailable), None

    identity = _attempt_identity(task, attempt)
    generation: str | None = None
    if not terminal:
        try:
            generation = _running_registration_generation(cfg, attempt.machine_name)
        except (FileNotFoundError, OSError, PermissionError):
            unavailable = _observation("unavailable", "read_failed")
            return unavailable, dict(unavailable), dict(unavailable), None
        except (TypeError, ValueError, KeyError):
            unavailable = _observation("unavailable", "identity_mismatch")
            return unavailable, dict(unavailable), dict(unavailable), None
        except Exception:
            unavailable = _observation("unavailable", "read_failed")
            return unavailable, dict(unavailable), dict(unavailable), None
        identity["registration_generation"] = generation

    disabled = policy.get("state") == "disabled"
    candidates: dict[int, dict[str, Any]] = {}
    # v1 remains eligible even when the frozen selection explicitly disables
    # extended reporting.  Unknown policy leaves every channel eligible.
    candidates[1] = _channel_result(
        cfg,
        shared_progress_path(cfg.shared_root, task.task_id, attempt.attempt_id),
        identity,
        1,
        terminal=terminal,
        validator=_validate_projection,
    )
    if not disabled:
        candidates[2] = _channel_result(
            cfg,
            shared_progress_v2_path(cfg, task.task_id, attempt.attempt_id),
            identity,
            2,
            terminal=terminal,
            validator=_validate_projection_v2,
        )
        candidates[3] = _channel_result(
            cfg,
            shared_progress_v3_path(cfg, task.task_id, attempt.attempt_id),
            identity,
            3,
            terminal=terminal,
            validator=_validate_projection_v3,
            max_bytes=MAX_SNAPSHOT_V3_BYTES,
        )

    progress = candidates.get(1, _observation("no_report", "no_snapshot"))
    progress_extended = candidates.get(2, _observation("no_report", "no_snapshot"))
    progress_scoped = candidates.get(3, _observation("no_report", "no_snapshot"))

    if not terminal and generation is not None:
        try:
            if _running_registration_generation(cfg, attempt.machine_name) != generation:
                unavailable = _observation("unavailable", "identity_mismatch")
                return unavailable, unavailable, unavailable, None
        except (FileNotFoundError, OSError, PermissionError):
            unavailable = _observation("unavailable", "read_failed")
            return unavailable, unavailable, unavailable, None
        except (TypeError, ValueError, KeyError):
            unavailable = _observation("unavailable", "identity_mismatch")
            return unavailable, unavailable, unavailable, None
        except Exception:
            unavailable = _observation("unavailable", "read_failed")
            return unavailable, unavailable, unavailable, None

    selected_version = select_progress_protocol(
        progress,
        progress_extended,
        progress_scoped,
        attempt_id=attempt.attempt_id,
        attempt_number=attempt.attempt_number,
    )
    return progress, progress_extended, progress_scoped, selected_version


def _mark_changed(row: dict[str, Any], fields: set[str]) -> None:
    unavailable = _observation("unavailable", "changed_during_read")
    if fields & {"location"}:
        row["location"] = dict(unavailable)
    if fields & _PROGRESS_FIELDS:
        row["progress"] = dict(unavailable)
        row["progress_extended"] = dict(unavailable)
        row["progress_scoped"] = dict(unavailable)
        row["selected_progress_protocol_version"] = None
        row["reporting_policy"] = _policy_result("unknown", "changed_during_read")


def _enrich_one(
    cfg: Any,
    task: TaskRecord,
    row: dict[str, Any],
    fields: set[str],
    policy: dict[str, Any] | None,
) -> dict[str, Any]:
    first_signature = _task_signature(task)
    attempt, selection_reason = _read_attempt(cfg, task)
    if "location" in fields:
        row["location"] = _location(task, attempt, selection_reason)
    if fields & _PROGRESS_FIELDS:
        effective_policy = policy or _policy_result("unknown", "policy_reference_missing")
        row["reporting_policy"] = dict(effective_policy)
        progress, progress_extended, progress_scoped, selected_version = _progress_observations(
            cfg, task, attempt, selection_reason, effective_policy
        )
        row["progress"] = progress
        row["progress_extended"] = progress_extended
        row["progress_scoped"] = progress_scoped
        row["selected_progress_protocol_version"] = selected_version

    # A final Task truth read is the sole transition fence for every enriched
    # row.  Required Task failures intentionally escape as command failures.
    latest = load_task(cfg, task.task_id)
    if _task_signature(latest) != first_signature:
        _mark_changed(row, fields)
    return row


def enrich_task_rows(
    cfg: Any,
    tasks: Sequence[TaskRecord],
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> list[dict[str, Any]]:
    """Enrich bounded returned rows with only their selected observations."""
    selected = {field for field in fields if isinstance(field, str)}
    needs_progress = bool(selected & _PROGRESS_FIELDS)
    policies = observe_reporting_policies(cfg, tasks) if needs_progress else {}
    enriched: list[dict[str, Any]] = []
    for task, row in zip(tasks, rows, strict=True):
        copied = dict(row)
        enriched.append(_enrich_one(cfg, task, copied, selected, policies.get(task.task_id)))
    if selected & _ENRICHED_FIELDS:
        observation_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        for row in enriched:
            row["observation_time"] = observation_time
    return enriched


__all__ = ["MAX_POLICY_BYTES", "enrich_task_rows", "observe_reporting_policies"]
