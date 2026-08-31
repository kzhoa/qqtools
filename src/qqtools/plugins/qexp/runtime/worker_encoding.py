"""Versioned primary/borrow Worker wire encoding compatibility."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .locks import group_lock, schema_lock
from .paths import group_path, shared_paths
from .records import normalize_group_record, utc_now
from .store import atomic_replace, iter_json, read_json

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


def _canonicalizing_marker(marker: dict[str, Any], started_by_agent: str) -> dict[str, Any]:
    """Return the durable marker for an in-progress encoding migration."""
    return {
        "state": ENCODING_CANONICALIZING,
        "revision": marker["revision"] + 1,
        "started_at": utc_now(),
        "completed_at": None,
        "started_by_agent": started_by_agent,
    }


def _canonical_marker(marker: dict[str, Any]) -> dict[str, Any]:
    """Return the completed marker after all Group records are canonical."""
    return {
        **marker,
        "state": ENCODING_CANONICAL_V2,
        "revision": marker["revision"] + 1,
        "completed_at": utc_now(),
    }


def _canonicalize_group_record(data: dict[str, Any]) -> dict[str, Any]:
    """Rewrite one Group's legacy encoding without changing its scheduling semantics."""
    normalize_group_record(data, allow_legacy=True)
    for worker in data["group"]["worker_set"].values():
        if worker["scheduling_role"] == "borrow" and worker["state"] == "borrow":
            worker["state"] = "active"
    normalize_group_record(data)
    return data


def ensure_canonical_primary_borrow_encoding(
    cfg: object, *, started_by_agent: str
) -> dict[str, Any]:
    """Idempotently migrate Group Worker encoding before N+1 dispatch or mutation."""
    if not isinstance(started_by_agent, str) or not started_by_agent:
        raise ValueError("primary/borrow upgrade requires a non-empty agent identifier.")
    with schema_lock(cfg.shared_root):
        marker = read_primary_borrow_encoding(cfg)
        if marker["state"] == ENCODING_COMPAT_V1:
            for path in iter_json(shared_paths(cfg.shared_root)["groups"]):
                _canonicalize_group_record(read_json(path))
            marker = _canonicalizing_marker(marker, started_by_agent)
            write_primary_borrow_encoding(cfg, marker)
        elif marker["state"] == ENCODING_CANONICAL_V2:
            return marker
    for path in iter_json(shared_paths(cfg.shared_root)["groups"]):
        with group_lock(cfg.shared_root, path.stem):
            if not path.exists():
                continue
            data = _canonicalize_group_record(read_json(path))
            atomic_replace(path, data)
    with schema_lock(cfg.shared_root):
        marker = read_primary_borrow_encoding(cfg)
        if marker["state"] == ENCODING_CANONICALIZING:
            marker = _canonical_marker(marker)
            write_primary_borrow_encoding(cfg, marker)
        if marker["state"] != ENCODING_CANONICAL_V2:
            raise RuntimeError("primary/borrow encoding upgrade did not reach canonical_v2.")
        return marker


def prepare_group_record_for_write(cfg: object, data: dict[str, Any]) -> dict[str, Any]:
    """Encode borrow Workers according to the current marker without changing semantics."""
    marker = read_primary_borrow_encoding(cfg)
    if marker["state"] != ENCODING_CANONICAL_V2:
        raise RuntimeError("primary/borrow encoding must be canonical_v2 before Group writes.")
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
