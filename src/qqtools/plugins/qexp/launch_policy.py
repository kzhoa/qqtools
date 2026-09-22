"""Project-scoped timeout policy for runner launch handoffs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .runtime.locks import exclusive
from .runtime.store import atomic_replace, read_json

DEFAULT_LAUNCH_HANDOFF_TIMEOUT_SECONDS = 10
MIN_LAUNCH_HANDOFF_TIMEOUT_SECONDS = 1
MAX_LAUNCH_HANDOFF_TIMEOUT_SECONDS = 300
LAUNCH_HANDOFF_POLICY_VERSION = 1
LAUNCH_HANDOFF_POLICY_FILENAME = "launch-handoff-policy.json"
LAUNCH_HANDOFF_POLICY_LOCK_FILENAME = "launch-handoff-policy.lock"
_POLICY_FIELDS = frozenset(("version", "timeout_seconds"))


def _shared_root(value: Any) -> Path:
    root = getattr(value, "shared_root", value)
    return Path(root).expanduser().resolve()


def launch_handoff_policy_path(shared_root: Any) -> Path:
    """Return the durable project policy path."""
    return _shared_root(shared_root) / LAUNCH_HANDOFF_POLICY_FILENAME


def launch_handoff_policy_lock_path(shared_root: Any) -> Path:
    """Return the lock used only for policy reads and writes."""
    return _shared_root(shared_root) / "locks" / LAUNCH_HANDOFF_POLICY_LOCK_FILENAME


def validate_launch_handoff_timeout_seconds(value: Any) -> int | float:
    """Validate a finite launch handoff timeout in the inclusive supported range."""
    if type(value) not in (int, float):
        raise ValueError("launch handoff timeout_seconds must be a number of seconds.")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("launch handoff timeout_seconds must be finite.")
    if value < MIN_LAUNCH_HANDOFF_TIMEOUT_SECONDS:
        raise ValueError("launch handoff timeout_seconds must be at least 1 second.")
    if value > MAX_LAUNCH_HANDOFF_TIMEOUT_SECONDS:
        raise ValueError("launch handoff timeout_seconds must be at most 300 seconds.")
    return value


def _document(value: Any, path: Path) -> int | float:
    if not isinstance(value, dict) or frozenset(value) != _POLICY_FIELDS:
        raise ValueError(f"malformed launch handoff policy at {path}: expected exactly version and timeout_seconds.")
    if type(value.get("version")) is not int or value["version"] != LAUNCH_HANDOFF_POLICY_VERSION:
        raise ValueError(f"malformed launch handoff policy at {path}: unsupported version.")
    try:
        return validate_launch_handoff_timeout_seconds(value.get("timeout_seconds"))
    except ValueError as exc:
        raise ValueError(f"malformed launch handoff policy at {path}: {exc}") from exc


def _read_configured(shared_root: Any) -> int | float | None:
    path = launch_handoff_policy_path(shared_root)
    try:
        value = read_json(path)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"unable to read launch handoff policy at {path}: {exc}") from exc
    return _document(value, path)


def _result(
    timeout_seconds: int | float,
    source: str,
    *,
    diagnostic_reason: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "timeout_seconds": timeout_seconds,
        "source": source,
        "applies_to": "new_launches",
    }
    if diagnostic_reason is not None:
        result["diagnostic_reason"] = diagnostic_reason[:160]
    return result


def show_launch_handoff_policy(shared_root: Any) -> dict[str, Any]:
    """Load the project policy, treating only a missing file as default."""
    configured = _read_configured(shared_root)
    if configured is None:
        return _result(DEFAULT_LAUNCH_HANDOFF_TIMEOUT_SECONDS, "default")
    return _result(configured, "configured")


def set_launch_handoff_policy(shared_root: Any, timeout_seconds: Any) -> dict[str, Any]:
    """Durably set the project policy after validating the existing document."""
    timeout = validate_launch_handoff_timeout_seconds(timeout_seconds)
    root = _shared_root(shared_root)
    path = launch_handoff_policy_path(root)
    with exclusive(launch_handoff_policy_lock_path(root)):
        # A malformed existing file is an operator error, never silently replaced
        # by a set operation.
        _read_configured(root)
        atomic_replace(
            path,
            {"version": LAUNCH_HANDOFF_POLICY_VERSION, "timeout_seconds": timeout},
        )
    return _result(timeout, "configured")


def reset_launch_handoff_policy(shared_root: Any) -> dict[str, Any]:
    """Remove the explicit policy after strictly validating any existing file."""
    root = _shared_root(shared_root)
    path = launch_handoff_policy_path(root)
    with exclusive(launch_handoff_policy_lock_path(root)):
        _read_configured(root)
        path.unlink(missing_ok=True)
    return _result(DEFAULT_LAUNCH_HANDOFF_TIMEOUT_SECONDS, "default")


def resolve_launch_handoff_policy(shared_root: Any) -> dict[str, Any]:
    """Resolve the launch policy, falling back safely with a bounded reason."""
    try:
        configured = _read_configured(shared_root)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        return _result(
            DEFAULT_LAUNCH_HANDOFF_TIMEOUT_SECONDS,
            "default",
            diagnostic_reason=reason,
        )
    if configured is None:
        return _result(DEFAULT_LAUNCH_HANDOFF_TIMEOUT_SECONDS, "default")
    return _result(configured, "configured")


load_launch_handoff_policy = show_launch_handoff_policy


__all__ = [
    "DEFAULT_LAUNCH_HANDOFF_TIMEOUT_SECONDS",
    "LAUNCH_HANDOFF_POLICY_FILENAME",
    "LAUNCH_HANDOFF_POLICY_LOCK_FILENAME",
    "LAUNCH_HANDOFF_POLICY_VERSION",
    "MAX_LAUNCH_HANDOFF_TIMEOUT_SECONDS",
    "MIN_LAUNCH_HANDOFF_TIMEOUT_SECONDS",
    "launch_handoff_policy_lock_path",
    "launch_handoff_policy_path",
    "load_launch_handoff_policy",
    "resolve_launch_handoff_policy",
    "reset_launch_handoff_policy",
    "set_launch_handoff_policy",
    "show_launch_handoff_policy",
    "validate_launch_handoff_timeout_seconds",
]
