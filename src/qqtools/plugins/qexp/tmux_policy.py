"""Project-scoped default policy for qexp tmux log observers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .runtime.locks import exclusive
from .runtime.store import atomic_replace, read_json

TMUX_POLICY_VERSION = 1
TMUX_POLICY_FILENAME = "tmux-policy.json"
TMUX_POLICY_LOCK_FILENAME = "tmux-policy.lock"
_POLICY_FIELDS = frozenset({"version", "enabled"})
_DIAGNOSTIC_LIMIT = 160


def _shared_root(value: Any) -> Path:
    root = getattr(value, "shared_root", value)
    return Path(root).expanduser().resolve()


def tmux_policy_path(shared_root: Any) -> Path:
    """Return the durable project policy path."""
    return _shared_root(shared_root) / TMUX_POLICY_FILENAME


def tmux_policy_lock_path(shared_root: Any) -> Path:
    """Return the lock used only for tmux policy reads and writes."""
    return _shared_root(shared_root) / "locks" / TMUX_POLICY_LOCK_FILENAME


def validate_tmux_policy_enabled(value: Any) -> bool:
    """Validate an actual boolean policy value."""
    if type(value) is not bool:
        raise ValueError("tmux policy enabled must be a boolean.")
    return value


def _document(value: Any, path: Path) -> bool:
    if not isinstance(value, dict) or frozenset(value) != _POLICY_FIELDS:
        raise ValueError(f"malformed tmux policy at {path}: expected exactly version and enabled.")
    if type(value.get("version")) is not int or value["version"] != TMUX_POLICY_VERSION:
        raise ValueError(f"malformed tmux policy at {path}: unsupported version.")
    try:
        return validate_tmux_policy_enabled(value.get("enabled"))
    except ValueError as exc:
        raise ValueError(f"malformed tmux policy at {path}: {exc}") from exc


def _read_configured(shared_root: Any) -> bool | None:
    path = tmux_policy_path(shared_root)
    try:
        value = read_json(path)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"unable to read tmux policy at {path}: {exc}") from exc
    return _document(value, path)


def _result(enabled: bool, source: str, *, diagnostic_reason: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "enabled": enabled,
        "source": source,
        "applies_to": "new_observer_decisions",
    }
    if diagnostic_reason is not None:
        result["diagnostic_reason"] = str(diagnostic_reason)[:_DIAGNOSTIC_LIMIT]
    return result


def show_tmux_policy(shared_root: Any) -> dict[str, Any]:
    """Load the project policy, treating only a missing file as the default."""
    configured = _read_configured(shared_root)
    if configured is None:
        return _result(False, "default")
    return _result(configured, "configured")


def set_tmux_policy(shared_root: Any, enabled: Any) -> dict[str, Any]:
    """Set the project policy after validating any existing document."""
    enabled = validate_tmux_policy_enabled(enabled)
    root = _shared_root(shared_root)
    path = tmux_policy_path(root)
    with exclusive(tmux_policy_lock_path(root)):
        # Never hide an operator-visible corrupt or unsupported policy by replacing it.
        _read_configured(root)
        atomic_replace(path, {"version": TMUX_POLICY_VERSION, "enabled": enabled})
    return _result(enabled, "configured")


def reset_tmux_policy(shared_root: Any) -> dict[str, Any]:
    """Remove the explicit policy after strictly validating any existing file."""
    root = _shared_root(shared_root)
    path = tmux_policy_path(root)
    with exclusive(tmux_policy_lock_path(root)):
        _read_configured(root)
        path.unlink(missing_ok=True)
    return _result(False, "default")


def resolve_tmux_policy(shared_root: Any) -> dict[str, Any]:
    """Resolve the policy safely for an observer decision."""
    try:
        configured = _read_configured(shared_root)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"[:_DIAGNOSTIC_LIMIT]
        return _result(False, "default", diagnostic_reason=reason)
    if configured is None:
        return _result(False, "default")
    return _result(configured, "configured")


load_tmux_policy = show_tmux_policy


__all__ = [
    "TMUX_POLICY_FILENAME",
    "TMUX_POLICY_LOCK_FILENAME",
    "TMUX_POLICY_VERSION",
    "load_tmux_policy",
    "resolve_tmux_policy",
    "reset_tmux_policy",
    "set_tmux_policy",
    "show_tmux_policy",
    "tmux_policy_lock_path",
    "tmux_policy_path",
    "validate_tmux_policy_enabled",
]
