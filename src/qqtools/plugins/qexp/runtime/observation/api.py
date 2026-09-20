"""Public paginated Task observation API."""

from __future__ import annotations

import base64
import binascii
import json
import re
from typing import Any

from ...config_types import RootConfig
from ...layout import project_id
from ..records import TASK_ID_PATTERN

_CURSOR_FIELDS = frozenset({"v", "project", "phase", "group", "order", "generation", "last"})
_CURSOR_ORDER = "task_id-asc-v1"
_MAX_CURSOR_LENGTH = 4096
_MAX_TASK_ID_LENGTH = 250
_GENERATION_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_CANONICAL_BASE64_PATTERN = re.compile(r"^(?:[A-Za-z0-9_-]{4})*(?:[A-Za-z0-9_-]{2}(?:==)?|[A-Za-z0-9_-]{3}=?){0,1}$")
_OBSERVATION_EXIT_CODES = {
    "invalid_argument": 2,
    "invalid_cursor": 2,
    "cursor_expired": 1,
    "index_not_ready": 1,
    "index_unavailable": 1,
}


class ObservationError(RuntimeError):
    """A stable error raised while producing a paginated observation."""

    code: str
    message: str
    exit_code: int

    def __init__(self, code: str, message: str, exit_code: int | None = None) -> None:
        self.code = code
        self.message = message
        self.exit_code = exit_code if exit_code is not None else _OBSERVATION_EXIT_CODES.get(code, 1)
        super().__init__(message)


def _invalid_argument(message: str) -> ObservationError:
    return ObservationError("invalid_argument", message, 2)


def _invalid_cursor(message: str) -> ObservationError:
    return ObservationError("invalid_cursor", message, 2)


def _index_unavailable(message: str, cause: BaseException | None = None) -> ObservationError:
    error = ObservationError("index_unavailable", message, 1)
    if cause is not None:
        error.__cause__ = cause
    return error


def _normalize_filter(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _invalid_argument(f"{label} must be a string or null.")
    if not value:
        return None
    if len(value) > 250:
        raise _invalid_argument(f"{label} must be at most 250 characters.")
    if label == "group" and TASK_ID_PATTERN.fullmatch(value) is None:
        raise _invalid_argument("group contains characters outside the allowed identifier pattern.")
    return value


def _validate_page_size(page_size: int) -> int:
    if type(page_size) is not int or not 1 <= page_size <= 1000:
        raise _invalid_argument("page_size must be an integer from 1 through 1000.")
    return page_size


def _is_valid_task_id(value: object) -> bool:
    return isinstance(value, str) and len(value) <= _MAX_TASK_ID_LENGTH and TASK_ID_PATTERN.fullmatch(value) is not None


def _canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _encode_cursor(*, project: str, phase: str | None, group: str | None, generation: str, last: str) -> str:
    value = {
        "generation": generation,
        "group": group,
        "last": last,
        "order": _CURSOR_ORDER,
        "phase": phase,
        "project": project,
        "v": 1,
    }
    token = base64.urlsafe_b64encode(_canonical_json(value)).decode("ascii")
    if len(token) > _MAX_CURSOR_LENGTH:
        raise _index_unavailable("the continuation cursor exceeds the maximum supported length.")
    return token


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("cursor contains duplicate fields.")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"cursor contains unsupported JSON constant {value!r}.")


def _decode_cursor(cursor: str, *, project: str, phase: str | None, group: str | None) -> tuple[str, str]:
    if not isinstance(cursor, str) or not cursor or len(cursor) > _MAX_CURSOR_LENGTH:
        raise _invalid_cursor("cursor is missing, too long, or not a string.")
    if _CANONICAL_BASE64_PATTERN.fullmatch(cursor) is None:
        raise _invalid_cursor("cursor is not canonical URL-safe base64.")
    try:
        padded_cursor = cursor + "=" * (-len(cursor) % 4)
        decoded = base64.b64decode(padded_cursor.encode("ascii"), altchars=b"-_", validate=True)
        value = json.loads(
            decoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_fields,
            parse_constant=_reject_json_constant,
        )
    except (
        UnicodeDecodeError,
        UnicodeEncodeError,
        UnicodeError,
        ValueError,
        binascii.Error,
        json.JSONDecodeError,
    ) as exc:
        raise _invalid_cursor("cursor is not valid canonical JSON.") from exc
    if not isinstance(value, dict) or set(value) != _CURSOR_FIELDS:
        raise _invalid_cursor("cursor fields are invalid.")
    version = value["v"]
    if type(version) is not int:
        raise _invalid_cursor("cursor version is not an integer.")
    if version != 1:
        raise ObservationError("cursor_expired", "cursor version is no longer supported.", 1)
    if not isinstance(value["project"], str) or not isinstance(value["order"], str):
        raise _invalid_cursor("cursor project or ordering fields are invalid.")
    if value["project"] != project:
        raise _invalid_cursor("cursor belongs to a different project.")
    if value["order"] != _CURSOR_ORDER:
        raise _invalid_cursor("cursor ordering does not match this API.")
    for label in ("phase", "group"):
        if value[label] is not None and not isinstance(value[label], str):
            raise _invalid_cursor(f"cursor {label} filter is invalid.")
        if isinstance(value[label], str) and len(value[label]) > 250:
            raise _invalid_cursor(f"cursor {label} filter is too long.")
    if value["group"] is not None and TASK_ID_PATTERN.fullmatch(value["group"]) is None:
        raise _invalid_cursor("cursor group filter is invalid.")
    if value["phase"] != phase or value["group"] != group:
        raise _invalid_cursor("cursor filters do not match this request.")
    generation = value["generation"]
    if not isinstance(generation, str) or _GENERATION_PATTERN.fullmatch(generation) is None:
        raise _invalid_cursor("cursor index generation is invalid.")
    last = value["last"]
    if not _is_valid_task_id(last):
        raise _invalid_cursor("cursor last Task ID is invalid.")
    canonical = base64.urlsafe_b64encode(_canonical_json(value)).decode("ascii")
    if cursor not in {canonical, canonical.rstrip("=")}:
        raise _invalid_cursor("cursor is not canonical.")
    return generation, last


def _candidate_read_fields(candidate_read: Any) -> tuple[str, tuple[str, ...], bool, int]:
    try:
        generation = candidate_read.generation
        keys = tuple(candidate_read.keys)
        exhausted = candidate_read.exhausted
        revision = candidate_read.revision
    except (AttributeError, TypeError) as exc:
        raise _index_unavailable("Task index returned an invalid candidate page.", exc) from exc
    if not isinstance(generation, str) or _GENERATION_PATTERN.fullmatch(generation) is None:
        raise _index_unavailable("Task index returned an invalid generation.")
    if type(exhausted) is not bool or type(revision) is not int or revision < 0:
        raise _index_unavailable("Task index returned invalid page metadata.")
    return generation, keys, exhausted, revision


def list_tasks_page(
    cfg: RootConfig,
    *,
    phase: str | None = None,
    group: str | None = None,
    page_size: int = 50,
    cursor: str | None = None,
) -> dict[str, Any]:
    """Return one live, indexed page of Task views.

    Args:
        cfg: Initialized qexp project configuration.
        phase: Optional exact Task phase filter.
        group: Optional exact Group filter.
        page_size: Number of matching Task views to return, from 1 through 1000.
        cursor: Opaque continuation returned by a previous page.

    Returns:
        A page with live Task views and continuation metadata.

    Raises:
        ObservationError: If arguments, the cursor, truth, or the required index is invalid.
    """
    phase = _normalize_filter(phase, "phase")
    group = _normalize_filter(group, "group")
    page_size = _validate_page_size(page_size)
    current_project = project_id(cfg.shared_root)
    expected_generation: str | None = None
    after: str | None = None
    if cursor is not None:
        expected_generation, after = _decode_cursor(cursor, project=current_project, phase=phase, group=group)

    from . import projection

    try:
        candidate_read = projection.read_candidates(
            cfg,
            phase,
            group,
            after,
            expected_generation,
            max(page_size, 256),
        )
    except ObservationError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise _index_unavailable("Task index could not be read.", exc) from exc
    generation, keys, exhausted, revision = _candidate_read_fields(candidate_read)
    if expected_generation is not None and generation != expected_generation:
        raise ObservationError("cursor_expired", "cursor index generation has expired.", 1)

    from ...observer import _task_view
    from ..dependencies import dependency_gate
    from ..tasks import load_task

    items: list[dict[str, Any]] = []
    last_inspected = after
    previous_key = after
    stop_reason: str | None = None
    for candidate_key in keys:
        if not _is_valid_task_id(candidate_key):
            raise _index_unavailable("Task index returned an invalid Task ID.")
        if previous_key is not None and candidate_key <= previous_key:
            raise _index_unavailable("Task index returned candidates out of order.")
        previous_key = candidate_key
        last_inspected = candidate_key
        try:
            task = load_task(cfg, candidate_key)
        except FileNotFoundError:
            continue
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise _index_unavailable(f"Task truth for candidate {candidate_key!r} is unreadable.", exc) from exc
        try:
            if task.task_id != candidate_key:
                raise ValueError("Task ID does not match its indexed key.")
            view = _task_view(task)
            if phase is not None and view["phase"] != phase:
                continue
            if group is not None and view["group"] != group:
                continue
            gate = dependency_gate(cfg, task)
            view["depends_on_task_ids"] = task.depends_on_task_ids
            view["dependency_state"] = gate.state
            view["dependency_reasons"] = list(gate.reasons)
            items.append(view)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise _index_unavailable(f"Task truth for candidate {candidate_key!r} is unreadable.", exc) from exc
        if len(items) >= page_size:
            stop_reason = "page_full"
            break

    if stop_reason is None:
        if exhausted:
            stop_reason = "exhausted"
        elif last_inspected is None or last_inspected == after:
            raise _index_unavailable("Task index made no progress within its candidate budget.")
        else:
            stop_reason = "budget_exhausted"

    try:
        projection.check_read(cfg, generation, revision)
    except ObservationError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise _index_unavailable("Task index changed while the page was being read.", exc) from exc

    next_cursor = None
    if stop_reason != "exhausted":
        if last_inspected is None:
            raise _index_unavailable("Task index cannot create a continuation cursor.")
        next_cursor = _encode_cursor(
            project=current_project,
            phase=phase,
            group=group,
            generation=generation,
            last=last_inspected,
        )
    return {
        "items": items,
        "next_cursor": next_cursor,
        "consistency": "live",
        "index_generation": generation,
        "stop_reason": stop_reason,
    }


__all__ = ["ObservationError", "list_tasks_page"]
