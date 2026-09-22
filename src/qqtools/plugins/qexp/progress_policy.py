"""Project-scoped reporting frequency policy for qexp progress observations.

The policy is deliberately separate from Task and Attempt records.  It is read
once at launch preparation and the resulting value is frozen in the local
Attempt context; running processes never resample this file.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .runtime.locks import exclusive
from .runtime.store import atomic_replace, read_json

DEFAULT_PROGRESS_INTERVAL_SECONDS = 30
PROGRESS_POLICY_VERSION = 1
PROGRESS_POLICY_FILENAME = "progress-policy.json"
PROGRESS_POLICY_LOCK_FILENAME = "progress-policy.lock"
_POLICY_FIELDS = frozenset(("version", "interval_seconds"))


def _shared_root(value: Any) -> Path:
    root = getattr(value, "shared_root", value)
    return Path(root).expanduser().resolve()


def progress_policy_path(shared_root: Any) -> Path:
    """Return the durable project policy path."""
    return _shared_root(shared_root) / PROGRESS_POLICY_FILENAME


def progress_policy_lock_path(shared_root: Any) -> Path:
    """Return the lock used only for policy reads/writes."""
    return _shared_root(shared_root) / "locks" / PROGRESS_POLICY_LOCK_FILENAME


def validate_interval_seconds(value: Any) -> int | float:
    """Validate a user or environment interval without accepting bool-like values."""
    if type(value) not in (int, float):
        raise ValueError("progress interval_seconds must be an int or float number of seconds.")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("progress interval_seconds must be finite.")
    if value < 1:
        raise ValueError("progress interval_seconds must be at least 1 second.")
    return value


def canonical_interval_seconds(value: Any) -> str:
    """Serialize a validated interval for the child-process environment."""
    validated = validate_interval_seconds(value)
    if isinstance(validated, int) or float(validated).is_integer():
        return str(int(validated))
    return format(float(validated), ".15g")


def _document(value: Any, path: Path) -> int | float:
    if not isinstance(value, dict) or frozenset(value) != _POLICY_FIELDS:
        raise ValueError(f"malformed progress policy at {path}: expected exactly version and interval_seconds.")
    if type(value.get("version")) is not int or value["version"] != PROGRESS_POLICY_VERSION:
        raise ValueError(f"malformed progress policy at {path}: unsupported version.")
    try:
        return validate_interval_seconds(value.get("interval_seconds"))
    except ValueError as exc:
        raise ValueError(f"malformed progress policy at {path}: {exc}") from exc


def _read_configured(shared_root: Any) -> int | float | None:
    path = progress_policy_path(shared_root)
    try:
        value = read_json(path)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"unable to read progress policy at {path}: {exc}") from exc
    return _document(value, path)


def _result(interval_seconds: int | float, source: str, *, diagnostic_reason: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "interval_seconds": interval_seconds,
        "source": source,
        "applies_to": "new_launches",
    }
    if diagnostic_reason is not None:
        result["diagnostic_reason"] = diagnostic_reason[:160]
    return result


def show_progress_policy(shared_root: Any) -> dict[str, Any]:
    """Load policy for an explicit read, treating only a missing file as default."""
    configured = _read_configured(shared_root)
    if configured is None:
        return _result(DEFAULT_PROGRESS_INTERVAL_SECONDS, "default")
    return _result(configured, "configured")


def set_progress_policy(shared_root: Any, interval_seconds: Any) -> dict[str, Any]:
    """Durably set the project policy after validating the previous document."""
    interval = validate_interval_seconds(interval_seconds)
    root = _shared_root(shared_root)
    path = progress_policy_path(root)
    with exclusive(progress_policy_lock_path(root)):
        # A malformed existing file is an explicit operator error, never silently
        # replaced by a set operation.
        _read_configured(root)
        atomic_replace(path, {"version": PROGRESS_POLICY_VERSION, "interval_seconds": interval})
    return _result(interval, "configured")


def reset_progress_policy(shared_root: Any) -> dict[str, Any]:
    """Remove the explicit policy after strictly validating any existing file."""
    root = _shared_root(shared_root)
    path = progress_policy_path(root)
    with exclusive(progress_policy_lock_path(root)):
        _read_configured(root)
        path.unlink(missing_ok=True)
    return _result(DEFAULT_PROGRESS_INTERVAL_SECONDS, "default")


def resolve_progress_policy(shared_root: Any) -> dict[str, Any]:
    """Resolve launch policy, falling back safely with a bounded local reason."""
    try:
        configured = _read_configured(shared_root)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        return _result(DEFAULT_PROGRESS_INTERVAL_SECONDS, "default", diagnostic_reason=reason)
    if configured is None:
        return _result(DEFAULT_PROGRESS_INTERVAL_SECONDS, "default")
    return _result(configured, "configured")


# Short aliases make the boundary discoverable to callers that use the policy
# as a load/set pair while retaining the explicit CLI-oriented names above.
load_progress_policy = show_progress_policy


__all__ = [
    "DEFAULT_PROGRESS_INTERVAL_SECONDS",
    "PROGRESS_POLICY_VERSION",
    "canonical_interval_seconds",
    "load_progress_policy",
    "progress_policy_lock_path",
    "progress_policy_path",
    "reset_progress_policy",
    "resolve_progress_policy",
    "set_progress_policy",
    "show_progress_policy",
    "validate_interval_seconds",
]
