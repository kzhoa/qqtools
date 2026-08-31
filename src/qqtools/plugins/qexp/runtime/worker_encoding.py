"""Versioned primary/borrow Worker wire encoding compatibility."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .paths import shared_paths
from .store import atomic_replace, read_json

ENCODING_COMPAT_V1 = "compat_v1"
# QQTOOLS-COMPAT-0001: remove compat_v1 runtime support in 1.3.15.
ENCODING_CANONICALIZING = "canonicalizing"
ENCODING_CANONICAL_V2 = "canonical_v2"
WORKER_ENCODING_STATES = {
    ENCODING_COMPAT_V1,
    ENCODING_CANONICALIZING,
    ENCODING_CANONICAL_V2,
}


def primary_borrow_encoding_path(shared_root: Path) -> Path:
    """Return the persistent primary/borrow encoding marker path."""
    return shared_paths(shared_root)["schema"] / "primary-borrow-encoding.json"


def _default_marker() -> dict[str, Any]:
    return {
        "state": ENCODING_COMPAT_V1,
        "revision": 1,
        "started_at": None,
        "completed_at": None,
        "started_by_agent": None,
    }


def _validate_marker(value: dict[str, Any]) -> dict[str, Any]:
    marker = value.get("primary_borrow_encoding")
    required = {"state", "revision", "started_at", "completed_at", "started_by_agent"}
    if not isinstance(marker, dict) or set(marker) != required:
        raise RuntimeError("primary/borrow encoding marker is malformed.")
    if marker["state"] not in WORKER_ENCODING_STATES:
        raise RuntimeError("primary/borrow encoding marker state is unsupported.")
    if type(marker["revision"]) is not int or marker["revision"] <= 0:
        raise RuntimeError("primary/borrow encoding marker revision is invalid.")
    for field in ("started_at", "completed_at", "started_by_agent"):
        if marker[field] is not None and not isinstance(marker[field], str):
            raise RuntimeError(f"primary/borrow encoding marker {field} is invalid.")
    return marker


def read_primary_borrow_encoding(cfg: object) -> dict[str, Any]:
    """Read the marker, treating pre-marker roots as compat_v1."""
    path = primary_borrow_encoding_path(cfg.shared_root)
    if not path.exists():
        return _default_marker()
    try:
        return _validate_marker(read_json(path))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("primary/borrow encoding marker is unreadable.") from exc


def ensure_primary_borrow_encoding(cfg: object) -> dict[str, Any]:
    """Create the initial compatibility marker without changing existing Groups."""
    path = primary_borrow_encoding_path(cfg.shared_root)
    if path.exists():
        return read_primary_borrow_encoding(cfg)
    marker = _default_marker()
    atomic_replace(path, {"primary_borrow_encoding": marker})
    return marker


def write_primary_borrow_encoding(cfg: object, marker: dict[str, Any]) -> None:
    """Persist a validated marker; callers hold the project schema lock."""
    value = {"primary_borrow_encoding": dict(marker)}
    _validate_marker(value)
    atomic_replace(primary_borrow_encoding_path(cfg.shared_root), value)


def prepare_group_record_for_write(cfg: object, data: dict[str, Any]) -> dict[str, Any]:
    """Encode borrow Workers according to the current marker without changing semantics."""
    marker = read_primary_borrow_encoding(cfg)
    if marker["state"] not in {ENCODING_CANONICALIZING, ENCODING_CANONICAL_V2}:
        return data
    try:
        workers = data["group"]["worker_set"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Group Worker Set is malformed.") from exc
    for worker in workers.values():
        if worker.get("scheduling_role") == "borrow":
            worker["state"] = "active"
    return data


def write_group_record(cfg: object, path: Path, data: dict[str, Any]) -> None:
    """Persist a Group after applying the marker-selected Worker encoding."""
    prepare_group_record_for_write(cfg, data)
    atomic_replace(path, data)
