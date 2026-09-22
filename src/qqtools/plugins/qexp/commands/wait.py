"""Read-only waiting for one selected qexp Task lifecycle."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..config_types import RootConfig
from ..runtime.dependencies import dependency_gate
from ..runtime.paths import attempt_path
from ..runtime.records import TASK_ID_PATTERN, AttemptRecord, TaskRecord
from ..runtime.store import read_json
from ..runtime.tasks import load_task

_TERMINAL_PHASES = frozenset({"succeeded", "failed", "cancelled"})
_WAITING_BLOCK_REASONS = frozenset(
    {
        "resource_waiting",
        "resource_unavailable",
        "waiting_for_resources",
        "queue_waiting",
        "waiting_for_dependency",
        "dependency_waiting",
    }
)
_TIMEOUT_PATTERN = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)([smh]?)\Z")
_LIFECYCLE_OUTCOMES = frozenset({"succeeded", "failed", "cancelled", "blocked", "timeout", "superseded"})


@dataclass(slots=True)
class _Selection:
    """The Task lifecycle selected by the initial coherent observation."""

    task_id: str
    attempt_number: int | None
    attempt_id: str | None = None


@dataclass(frozen=True, slots=True)
class _ObservationProblem(Exception):
    code: str
    message: str


def _error(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def make_wait_result(
    *,
    project: object | None = None,
    task_id: str | None = None,
    selected_attempt_number: int | None = None,
    selected_attempt_id: str | None = None,
    outcome: str,
    reason: str | None = None,
    task_exit_code: int | None = None,
    error: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build the complete machine-readable result for ``task wait``.

    ``project`` and all selection fields are nullable so callers handling an
    invocation before configuration or Task resolution can still emit the
    same result shape as the polling implementation.
    """
    normalized_error = None if outcome in _LIFECYCLE_OUTCOMES else (dict(error) if error is not None else None)
    return {
        "schema_version": 1,
        "task_id": task_id if isinstance(task_id, str) else None,
        "project": str(project) if project is not None else None,
        "selected_attempt_number": selected_attempt_number if type(selected_attempt_number) is int else None,
        "selected_attempt_id": selected_attempt_id if isinstance(selected_attempt_id, str) else None,
        "outcome": outcome,
        "reason": reason if isinstance(reason, str) else None,
        "task_exit_code": task_exit_code if type(task_exit_code) is int else None,
        "error": normalized_error,
    }


def _result(
    cfg: RootConfig,
    task_id: str,
    *,
    selected_attempt_number: int | None,
    selected_attempt_id: str | None,
    outcome: str,
    reason: str | None = None,
    task_exit_code: int | None = None,
    error: dict[str, str] | None = None,
) -> dict[str, object]:
    return make_wait_result(
        project=cfg.shared_root,
        task_id=task_id,
        selected_attempt_number=selected_attempt_number,
        selected_attempt_id=selected_attempt_id,
        outcome=outcome,
        reason=reason,
        task_exit_code=task_exit_code,
        error=error,
    )


def _selection_values(selection: _Selection | None) -> tuple[int | None, str | None]:
    if selection is None:
        return None, None
    return selection.attempt_number, selection.attempt_id


def _problem_result(
    cfg: RootConfig,
    task_id: str,
    problem: _ObservationProblem,
    selection: _Selection | None = None,
) -> dict[str, object]:
    number, attempt_id = _selection_values(selection)
    return _result(
        cfg,
        task_id,
        selected_attempt_number=number,
        selected_attempt_id=attempt_id,
        outcome="observation_failed",
        reason=problem.code,
        error=_error(problem.code, problem.message),
    )


def parse_wait_timeout(value: str | int | float | None) -> float | None:
    """Parse a finite non-negative duration expressed in seconds, minutes, or hours."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("wait timeout must be a finite non-negative duration.")
    if isinstance(value, (int, float)):
        try:
            parsed = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError("wait timeout must be a finite non-negative duration.") from exc
        if not math.isfinite(parsed) or parsed < 0:
            raise ValueError("wait timeout must be a finite non-negative duration.")
        return parsed
    if not isinstance(value, str):
        raise ValueError("wait timeout must be a finite non-negative duration.")
    match = _TIMEOUT_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError("wait timeout must be a decimal number with an optional s, m, or h suffix.")
    try:
        parsed = float(value[: -len(match.group(1))] if match.group(1) else value)
    except (OverflowError, ValueError) as exc:
        raise ValueError("wait timeout must be a finite non-negative duration.") from exc
    multiplier = {"": 1.0, "s": 1.0, "m": 60.0, "h": 3600.0}[match.group(1)]
    parsed *= multiplier
    if not math.isfinite(parsed):
        raise ValueError("wait timeout must be a finite non-negative duration.")
    return parsed


def _valid_task_id(task_id: object) -> bool:
    return isinstance(task_id, str) and 1 <= len(task_id) <= 250 and TASK_ID_PATTERN.fullmatch(task_id) is not None


def _validate_seconds(value: object, label: str, *, positive: bool = False) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"{label} must be a finite number of seconds."
    try:
        parsed = float(value)
    except (OverflowError, ValueError):
        return f"{label} must be a finite number of seconds."
    if not math.isfinite(parsed) or parsed < 0 or (positive and parsed == 0):
        return f"{label} must be a finite {('positive' if positive else 'non-negative')} number of seconds."
    return None


def _validate_task_snapshot(task: TaskRecord, task_id: str) -> tuple[str, str | None, int | None, int]:
    if task.task_id != task_id:
        raise _ObservationProblem("identity_mismatch", "Task truth does not match the requested Task ID.")
    state = task.state
    control = task.attempt_control
    if not isinstance(state, dict) or not isinstance(control, dict):
        raise _ObservationProblem("malformed_task", "Task truth has malformed lifecycle fields.")
    phase = state.get("projection")
    if phase not in {"queued", "running", "succeeded", "failed", "cancelled", "blocked"}:
        raise _ObservationProblem("malformed_task", "Task truth has an unsupported lifecycle phase.")
    current_id = control.get("current_attempt_id")
    current_number = control.get("current_attempt_number")
    next_number = control.get("next_attempt_number")
    if current_id is not None and not _valid_task_id(current_id):
        raise _ObservationProblem("malformed_task", "Task truth has an invalid current Attempt ID.")
    if current_number is not None and (type(current_number) is not int or current_number < 1):
        raise _ObservationProblem("malformed_task", "Task truth has an invalid current Attempt number.")
    if type(next_number) is not int or next_number < 1:
        raise _ObservationProblem("malformed_task", "Task truth has an invalid next Attempt number.")
    if current_id is not None and current_number is None:
        raise _ObservationProblem("identity_mismatch", "Task truth names an Attempt without its number.")
    if phase == "running" and current_id is None:
        raise _ObservationProblem("identity_mismatch", "running Task truth does not name a current Attempt.")
    return phase, current_id, current_number, next_number


def _load_task_checked(cfg: RootConfig, task_id: str) -> TaskRecord:
    try:
        task = load_task(cfg, task_id)
    except FileNotFoundError as exc:
        raise _ObservationProblem("task_missing", f"Task {task_id!r} was not found.") from exc
    except (OSError, KeyError, TypeError, ValueError, AttributeError, UnicodeError, RuntimeError, IndexError) as exc:
        raise _ObservationProblem("malformed_task", f"Task {task_id!r} is malformed or unreadable.") from exc
    try:
        _validate_task_snapshot(task, task_id)
    except _ObservationProblem:
        raise
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise _ObservationProblem("malformed_task", f"Task {task_id!r} has malformed lifecycle fields.") from exc
    return task


def _load_attempt(cfg: RootConfig, task_id: str, attempt_number: int) -> AttemptRecord | None:
    path = attempt_path(cfg.shared_root, task_id, attempt_number)
    try:
        value = read_json(path)
    except FileNotFoundError:
        return None
    except (OSError, KeyError, TypeError, ValueError, AttributeError, UnicodeError, RuntimeError, IndexError) as exc:
        raise _ObservationProblem(
            "attempt_unreadable",
            f"Attempt {task_id!r} number {attempt_number} is unreadable.",
        ) from exc
    try:
        attempt = AttemptRecord.from_dict(value)
    except (OSError, KeyError, TypeError, ValueError, AttributeError, UnicodeError, RuntimeError, IndexError) as exc:
        raise _ObservationProblem(
            "malformed_attempt",
            f"Attempt {task_id!r} number {attempt_number} is malformed.",
        ) from exc
    if attempt.task_id != task_id or attempt.attempt_number != attempt_number:
        raise _ObservationProblem(
            "identity_mismatch",
            f"Attempt {task_id!r} number {attempt_number} has mismatched identity.",
        )
    return attempt


def _attempt_exit_code(attempt: AttemptRecord) -> int | None:
    result = attempt.result
    if not isinstance(result, Mapping):
        raise _ObservationProblem("malformed_attempt", f"Attempt {attempt.attempt_id!r} result is malformed.")
    exit_code = result.get("exit_code")
    if exit_code is not None and (type(exit_code) is not int):
        raise _ObservationProblem("malformed_attempt", f"Attempt {attempt.attempt_id!r} exit code is invalid.")
    return exit_code


def _attempt_result(
    cfg: RootConfig,
    task_id: str,
    selection: _Selection,
    attempt: AttemptRecord,
    *,
    fallback_reason: str | None = None,
) -> dict[str, object] | None:
    if attempt.phase not in _TERMINAL_PHASES:
        return None
    exit_code = _attempt_exit_code(attempt)
    result = attempt.result
    reason = result.get("reason") if isinstance(result, Mapping) else None
    if reason is None:
        reason = fallback_reason
    if reason is not None and not isinstance(reason, str):
        raise _ObservationProblem("malformed_attempt", f"Attempt {attempt.attempt_id!r} reason is invalid.")
    return _result(
        cfg,
        task_id,
        selected_attempt_number=selection.attempt_number,
        selected_attempt_id=attempt.attempt_id,
        outcome=attempt.phase,
        reason=reason,
        task_exit_code=exit_code,
    )


def _task_result(
    cfg: RootConfig,
    task_id: str,
    selection: _Selection,
    task: TaskRecord,
    *,
    attempt: AttemptRecord | None = None,
) -> dict[str, object]:
    phase = task.state.get("projection")
    reason = task.state.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise _ObservationProblem("malformed_task", f"Task {task_id!r} reason is invalid.")
    selected_id = selection.attempt_id
    exit_code: int | None = None
    if attempt is not None:
        if attempt.phase != phase:
            raise _ObservationProblem(
                "identity_mismatch",
                f"Task {task_id!r} terminal truth does not match its selected Attempt.",
            )
        selected_id = attempt.attempt_id
        exit_code = _attempt_exit_code(attempt)
        if reason is None and isinstance(attempt.result, Mapping):
            reason = attempt.result.get("reason")
            if reason is not None and not isinstance(reason, str):
                raise _ObservationProblem("malformed_attempt", f"Attempt {attempt.attempt_id!r} reason is invalid.")
    return _result(
        cfg,
        task_id,
        selected_attempt_number=selection.attempt_number,
        selected_attempt_id=selected_id,
        outcome=phase,
        reason=reason,
        task_exit_code=exit_code,
    )


def _gate(cfg: RootConfig, task: TaskRecord) -> Any:
    try:
        gate = dependency_gate(cfg, task)
    except (OSError, KeyError, TypeError, ValueError, AttributeError, UnicodeError, RuntimeError, IndexError) as exc:
        raise _ObservationProblem("dependency_unavailable", "Task dependency truth is unreadable.") from exc
    state = getattr(gate, "state", None)
    if state not in {"ready", "waiting", "blocked", "invalid"}:
        raise _ObservationProblem("malformed_dependency_gate", "Task dependency gate returned invalid state.")
    reasons = getattr(gate, "reasons", ())
    if not isinstance(reasons, (tuple, list)):
        raise _ObservationProblem("malformed_dependency_gate", "Task dependency gate returned invalid reasons.")
    return gate


def _gate_reason(gate: Any) -> str:
    state = gate.state
    if state == "invalid":
        return "dependency_invalid"
    if state == "blocked":
        return "dependency_blocked"
    return f"dependency_{state}"


def _requires_intervention(task: TaskRecord, gate: Any) -> bool:
    if gate.state in {"blocked", "invalid"}:
        return True
    if task.state.get("projection") != "blocked":
        return False
    reason = task.state.get("reason")
    return reason not in _WAITING_BLOCK_REASONS


def _waiting_or_blocked(
    cfg: RootConfig,
    task_id: str,
    selection: _Selection,
    task: TaskRecord,
) -> dict[str, object] | None:
    gate = _gate(cfg, task)
    if _requires_intervention(task, gate):
        reason = task.state.get("reason") if task.state.get("projection") == "blocked" else _gate_reason(gate)
        if reason is not None and not isinstance(reason, str):
            raise _ObservationProblem("malformed_task", f"Task {task_id!r} blocked reason is invalid.")
        return _result(
            cfg,
            task_id,
            selected_attempt_number=selection.attempt_number,
            selected_attempt_id=selection.attempt_id,
            outcome="blocked",
            reason=reason,
        )
    return None


def _initial_selection(task: TaskRecord, task_id: str) -> _Selection:
    phase, current_id, current_number, next_number = _validate_task_snapshot(task, task_id)
    if current_id is not None:
        return _Selection(task_id, current_number, current_id)
    if phase == "queued":
        return _Selection(task_id, next_number)
    if phase in _TERMINAL_PHASES:
        return _Selection(task_id, current_number)
    if phase == "blocked" and current_number is not None:
        return _Selection(task_id, current_number)
    if phase == "blocked":
        return _Selection(task_id, next_number)
    raise _ObservationProblem("identity_mismatch", f"Task {task_id!r} has no selectable lifecycle Attempt.")


def _initial_observation(
    cfg: RootConfig,
    task_id: str,
) -> tuple[_Selection, dict[str, object] | None]:
    task = _load_task_checked(cfg, task_id)
    selection = _initial_selection(task, task_id)
    phase, current_id, current_number, _ = _validate_task_snapshot(task, task_id)
    attempt: AttemptRecord | None = None
    should_read_attempt = current_id is not None or (
        phase in (_TERMINAL_PHASES | {"blocked"}) and current_number is not None
    )
    if should_read_attempt and selection.attempt_number is not None:
        attempt = _load_attempt(cfg, task_id, selection.attempt_number)
        if attempt is None:
            if phase in _TERMINAL_PHASES and current_id is None and current_number is None:
                attempt = None
            else:
                raise _ObservationProblem(
                    "attempt_missing",
                    f"Attempt {task_id!r} number {selection.attempt_number} was not found.",
                )
        elif current_id is not None and attempt.attempt_id != current_id:
            raise _ObservationProblem(
                "identity_mismatch",
                f"Task {task_id!r} does not match its current Attempt.",
            )
    if attempt is not None:
        terminal = _attempt_result(cfg, task_id, selection, attempt, fallback_reason=task.state.get("reason"))
        if terminal is not None:
            if phase in _TERMINAL_PHASES and phase != attempt.phase:
                raise _ObservationProblem(
                    "identity_mismatch",
                    f"Task {task_id!r} terminal truth does not match its selected Attempt.",
                )
            selection.attempt_id = attempt.attempt_id
            return selection, terminal
        if phase in _TERMINAL_PHASES:
            raise _ObservationProblem(
                "identity_mismatch",
                f"Task {task_id!r} terminal truth names a non-terminal Attempt.",
            )
    if phase in _TERMINAL_PHASES:
        return selection, _task_result(cfg, task_id, selection, task, attempt=attempt)
    if phase == "queued" and task.control.get("cancellation_requested_at") and attempt is None:
        return (
            selection,
            _result(
                cfg,
                task_id,
                selected_attempt_number=selection.attempt_number,
                selected_attempt_id=None,
                outcome="cancelled",
                reason="cancelled_by_user",
            ),
        )
    if attempt is not None:
        blocked = _waiting_or_blocked(cfg, task_id, selection, task)
        return selection, blocked
    blocked = _waiting_or_blocked(cfg, task_id, selection, task)
    return selection, blocked


def _observe_selected(
    cfg: RootConfig,
    selection: _Selection,
) -> dict[str, object] | None:
    task = _load_task_checked(cfg, selection.task_id)
    phase, current_id, current_number, next_number = _validate_task_snapshot(task, selection.task_id)
    expected_number = selection.attempt_number
    if expected_number is None:
        if phase in _TERMINAL_PHASES:
            return _task_result(cfg, selection.task_id, selection, task)
        raise _ObservationProblem("identity_mismatch", "Task lifecycle changed after a terminal selection.")

    names_selected = current_number == expected_number and (
        (selection.attempt_id is not None and current_id == selection.attempt_id)
        or (selection.attempt_id is None and current_id is not None)
    )
    if selection.attempt_id is not None and current_id is None and phase in (_TERMINAL_PHASES | {"blocked"}):
        names_selected = current_number == expected_number

    if selection.attempt_id is None and current_id is None and phase == "blocked" and current_number == expected_number:
        attempt = _load_attempt(cfg, selection.task_id, expected_number)
        if attempt is None:
            raise _ObservationProblem(
                "attempt_missing",
                f"Attempt {selection.task_id!r} number {expected_number} was not found.",
            )
        selection.attempt_id = attempt.attempt_id
        terminal = _attempt_result(cfg, selection.task_id, selection, attempt, fallback_reason=task.state.get("reason"))
        if terminal is not None:
            return terminal
        return _waiting_or_blocked(cfg, selection.task_id, selection, task)

    if selection.attempt_id is None and current_id is not None and current_number == expected_number:
        attempt = _load_attempt(cfg, selection.task_id, expected_number)
        if attempt is None:
            raise _ObservationProblem(
                "attempt_missing",
                f"Attempt {selection.task_id!r} number {expected_number} was not found.",
            )
        if attempt.attempt_id != current_id:
            raise _ObservationProblem(
                "identity_mismatch",
                f"Task {selection.task_id!r} does not match its current Attempt.",
            )
        selection.attempt_id = attempt.attempt_id
        terminal = _attempt_result(cfg, selection.task_id, selection, attempt, fallback_reason=task.state.get("reason"))
        if terminal is not None:
            return terminal
        if phase in _TERMINAL_PHASES:
            raise _ObservationProblem(
                "identity_mismatch",
                f"Task {selection.task_id!r} terminal truth names a non-terminal Attempt.",
            )
        blocked = _waiting_or_blocked(cfg, selection.task_id, selection, task)
        return blocked

    if names_selected:
        attempt = _load_attempt(cfg, selection.task_id, expected_number)
        if attempt is None:
            raise _ObservationProblem(
                "attempt_missing",
                f"Attempt {selection.task_id!r} number {expected_number} was not found.",
            )
        if selection.attempt_id is not None and attempt.attempt_id != selection.attempt_id:
            raise _ObservationProblem(
                "identity_mismatch",
                f"Attempt {selection.task_id!r} changed its pinned identity.",
            )
        terminal = _attempt_result(cfg, selection.task_id, selection, attempt, fallback_reason=task.state.get("reason"))
        if terminal is not None:
            if phase in _TERMINAL_PHASES and phase != attempt.phase:
                raise _ObservationProblem(
                    "identity_mismatch",
                    f"Task {selection.task_id!r} terminal truth does not match its selected Attempt.",
                )
            return terminal
        if phase in _TERMINAL_PHASES:
            raise _ObservationProblem(
                "identity_mismatch",
                f"Task {selection.task_id!r} terminal truth names a non-terminal Attempt.",
            )
        return _waiting_or_blocked(cfg, selection.task_id, selection, task)

    # The Task moved away from the selected lifecycle. Read that exact Attempt
    # once so a retained terminal result wins over supersession.
    retained = _load_attempt(cfg, selection.task_id, expected_number)
    if retained is not None:
        terminal = _attempt_result(cfg, selection.task_id, selection, retained)
        if terminal is not None:
            selection.attempt_id = retained.attempt_id
            return terminal
    if phase == "cancelled" and selection.attempt_id is None and current_number is None:
        return _task_result(cfg, selection.task_id, selection, task)
    if phase in {"queued", "blocked"} and current_id is None and next_number == expected_number:
        if phase == "queued" and task.control.get("cancellation_requested_at"):
            return _result(
                cfg,
                selection.task_id,
                selected_attempt_number=selection.attempt_number,
                selected_attempt_id=selection.attempt_id,
                outcome="cancelled",
                reason="cancelled_by_user",
            )
        return _waiting_or_blocked(cfg, selection.task_id, selection, task)
    return _result(
        cfg,
        selection.task_id,
        selected_attempt_number=selection.attempt_number,
        selected_attempt_id=selection.attempt_id,
        outcome="superseded",
        reason="lifecycle_superseded",
    )


def wait_for_task(
    cfg: RootConfig,
    task_id: str,
    *,
    timeout_seconds: float | None = None,
    poll_interval_seconds: float = 2.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, object], int]:
    """Wait for the lifecycle selected by one coherent Task observation."""
    if not _valid_task_id(task_id):
        result = _result(
            cfg,
            task_id if isinstance(task_id, str) else None,
            selected_attempt_number=None,
            selected_attempt_id=None,
            outcome="invalid_input",
            reason="invalid_task_id",
            error=_error("invalid_input", "task_id is invalid."),
        )
        return result, 2
    timeout_error = None if timeout_seconds is None else _validate_seconds(timeout_seconds, "timeout_seconds")
    interval_error = _validate_seconds(poll_interval_seconds, "poll_interval_seconds")
    if timeout_error is not None or interval_error is not None:
        message = timeout_error or interval_error or "wait options are invalid."
        result = _result(
            cfg,
            task_id,
            selected_attempt_number=None,
            selected_attempt_id=None,
            outcome="invalid_input",
            reason="invalid_input",
            error=_error("invalid_input", message),
        )
        return result, 2

    selection: _Selection | None = None
    try:
        started = monotonic()
        deadline = None if timeout_seconds is None else started + float(timeout_seconds)
        selection, initial = _initial_observation(cfg, task_id)
        if initial is not None:
            return initial, _exit_code(initial["outcome"])
        if timeout_seconds == 0:
            timeout = _result(
                cfg,
                task_id,
                selected_attempt_number=selection.attempt_number,
                selected_attempt_id=selection.attempt_id,
                outcome="timeout",
                reason="timeout",
            )
            return timeout, 4
        while True:
            now = monotonic()
            if deadline is not None and now >= deadline:
                timeout = _result(
                    cfg,
                    task_id,
                    selected_attempt_number=selection.attempt_number,
                    selected_attempt_id=selection.attempt_id,
                    outcome="timeout",
                    reason="timeout",
                )
                return timeout, 4
            delay = float(poll_interval_seconds)
            if deadline is not None:
                delay = min(delay, max(0.0, deadline - now))
            if delay <= 0:
                timeout = _result(
                    cfg,
                    task_id,
                    selected_attempt_number=selection.attempt_number,
                    selected_attempt_id=selection.attempt_id,
                    outcome="timeout",
                    reason="timeout",
                )
                return timeout, 4
            sleep(delay)
            if deadline is not None and monotonic() >= deadline:
                timeout = _result(
                    cfg,
                    task_id,
                    selected_attempt_number=selection.attempt_number,
                    selected_attempt_id=selection.attempt_id,
                    outcome="timeout",
                    reason="timeout",
                )
                return timeout, 4
            observed = _observe_selected(cfg, selection)
            if observed is not None:
                return observed, _exit_code(observed["outcome"])
    except KeyboardInterrupt:
        number, attempt_id = _selection_values(selection)
        interrupted = _result(
            cfg,
            task_id,
            selected_attempt_number=number,
            selected_attempt_id=attempt_id,
            outcome="interrupted",
            reason="interrupted",
            error=_error("interrupted", "wait interrupted."),
        )
        return interrupted, 130
    except _ObservationProblem as problem:
        return _problem_result(cfg, task_id, problem, selection), 6
    except (OSError, TypeError, ValueError, AttributeError, UnicodeError, RuntimeError, IndexError) as exc:
        problem = _ObservationProblem("observation_error", f"Task observation failed: {exc}")
        return _problem_result(cfg, task_id, problem, selection), 6


def _exit_code(outcome: object) -> int:
    return {
        "succeeded": 0,
        "failed": 1,
        "cancelled": 1,
        "blocked": 3,
        "timeout": 4,
        "superseded": 5,
        "observation_failed": 6,
        "invalid_input": 2,
        "interrupted": 130,
    }.get(outcome, 6)


__all__ = ["make_wait_result", "parse_wait_timeout", "wait_for_task"]
