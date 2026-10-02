"""Shared-only progress observation and exact advisory publication transactions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from qqtools.qexp._progress_protocol import MAX_SNAPSHOT_BYTES, read_advisory_snapshot, replace_advisory_snapshot

from .locks import exclusive
from .progress import _progress_lock_path, _validate_projection, resolve_progress_binding, shared_progress_path
from .progress_v2 import _validate_projection_v2


def _snapshot_path(cfg: Any, context: dict[str, Any]) -> Path:
    directory = "progress" if context["protocol_version"] == 1 else "progress-v2"
    return Path(cfg.shared_root) / directory / context["task_id"] / f"{context['attempt_id']}.json"


def validate_progress_snapshot(
    value: Any, binding: dict[str, Any], version: int, *, require_token: bool = False
) -> dict:
    if type(version) is not int or version not in {1, 2}:
        raise ValueError("progress snapshot version is invalid")
    validate = _validate_projection if version == 1 else _validate_projection_v2
    return validate(value, binding, require_token=require_token, require_generation=True)


def observe_progress_snapshot(cfg: Any, context: dict[str, Any], registration_generation: str) -> dict[str, Any]:
    """Read one current Attempt binding and its optional valid shared projection."""
    binding = resolve_progress_binding(cfg, context)
    if binding is None:
        return {"state": "retired", "binding": None, "snapshot": None}
    binding = {**binding, "registration_generation": registration_generation}
    snapshot = None
    try:
        snapshot = validate_progress_snapshot(
            read_advisory_snapshot(_snapshot_path(cfg, context), max_bytes=MAX_SNAPSHOT_BYTES),
            binding,
            context["protocol_version"],
        )
    except (OSError, ValueError, KeyError, TypeError, RecursionError):
        pass
    if resolve_progress_binding(cfg, context) != {
        key: value for key, value in binding.items() if key != "registration_generation"
    }:
        raise RuntimeError("progress Attempt identity changed during observation")
    return {"state": "observed", "binding": binding, "snapshot": snapshot}


def publish_progress_snapshot(
    cfg: Any,
    context: dict[str, Any],
    binding: dict[str, Any],
    snapshot: dict[str, Any],
    *,
    resolver: Callable[[Any, dict[str, Any]], dict[str, Any] | None] = resolve_progress_binding,
    context_alive: Callable[[], bool],
    writer: Callable[..., None] = replace_advisory_snapshot,
    before_replace: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Publish under the existing cleanup fence without any local write or process effect."""
    with exclusive(_progress_lock_path(cfg.shared_root, binding["task_id"]), blocking=False) as acquired:
        if not acquired:
            return {"state": "blocked", "binding": None, "snapshot": None}
        current = resolver(cfg, context)
        if current is None or not context_alive():
            return {"state": "retired", "binding": None, "snapshot": None}
        current = {**current, "registration_generation": binding["registration_generation"]}
        if current["fencing_token"] != binding["fencing_token"]:
            return {"state": "stale", "binding": None, "snapshot": None}
        validated = validate_progress_snapshot(snapshot, current, context["protocol_version"], require_token=True)
        path = _snapshot_path(cfg, context)
        existing = None
        try:
            existing = validate_progress_snapshot(
                read_advisory_snapshot(path, max_bytes=MAX_SNAPSHOT_BYTES),
                current,
                context["protocol_version"],
                require_token=True,
            )
        except (OSError, ValueError, KeyError, TypeError, RecursionError):
            pass
        if existing == validated:
            return {"state": "published", "binding": current, "snapshot": validated}
        if existing is not None and existing["sequence"] >= validated["sequence"]:
            return {"state": "stale", "binding": None, "snapshot": None}
        if before_replace is not None:
            before_replace()
        path.parent.mkdir(parents=True, exist_ok=True)
        kwargs = {"max_bytes": MAX_SNAPSHOT_BYTES} if context["protocol_version"] == 2 else {}
        if before_replace is not None:
            kwargs["before_replace"] = before_replace
        writer(path, validated, **kwargs)
        if context["protocol_version"] == 2 and not context_alive():
            if before_replace is not None:
                before_replace()
            path.unlink(missing_ok=True)
            return {"state": "retired", "binding": None, "snapshot": None}
        return {"state": "published", "binding": current, "snapshot": validated}
