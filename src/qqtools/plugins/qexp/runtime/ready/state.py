"""Durable ready-index state, initialization, and writer compatibility gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from ..locks import exclusive
from ..paths import ready_state_path, shared_paths
from ..records import utc_now
from ..store import atomic_replace, read_json

READY_PROTOCOL_VERSION = 1
READY_WRITER_CAPABILITY = "ready-v1"
ReadyIndexState = Literal["absent", "building", "active", "degraded"]


def ensure_ready_layout(cfg: object) -> None:
    """Create the additive ready layout without activating ready-only scheduling."""
    paths = shared_paths(cfg.shared_root)
    for name in (
        "ready",
        "ready_home",
        "ready_shared",
        "ready_catalogs",
        "ready_reservations",
        "ready_cursors",
        "ready_builds",
        "ready_locks",
        "ready_primary",
    ):
        paths[name].mkdir(parents=True, exist_ok=True)
    path = ready_state_path(cfg.shared_root)
    if not path.exists():
        atomic_replace(
            path,
            {
                "ready_index": {
                    "schema_version": READY_PROTOCOL_VERSION,
                    "state": "absent",
                    "writer_capability": None,
                    "revision": 0,
                    "build": None,
                    "updated_at": utc_now(),
                    "degraded_reasons": [],
                }
            },
        )


def read_ready_index_state(cfg: object) -> ReadyIndexState:
    """Return the scheduling gate for this project's ready projection."""
    path = ready_state_path(cfg.shared_root)
    if not path.exists():
        return "absent"
    try:
        record = read_json(path)["ready_index"]
        current_state = record["state"]
        if record["schema_version"] != READY_PROTOCOL_VERSION:
            return "degraded"
        if current_state not in {"absent", "building", "active", "degraded"}:
            return "degraded"
        if current_state in {"building", "active"} and record.get("writer_capability") != READY_WRITER_CAPABILITY:
            return "degraded"
        return current_state
    except (KeyError, TypeError, ValueError):
        return "degraded"


def state_lock_path(cfg: object) -> Path:
    """Return the ready-index state lock path without acquiring it."""
    return shared_paths(cfg.shared_root)["ready_locks"] / "state.lock"


def read_state_record(cfg: object) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read and validate the ready-index state record without acquiring its lock."""
    value = read_json(ready_state_path(cfg.shared_root))
    record = value["ready_index"]
    if record.get("schema_version") != READY_PROTOCOL_VERSION:
        raise ValueError("ready index state schema is unsupported.")
    if record.get("state") not in {"absent", "building", "active", "degraded"}:
        raise ValueError("ready index state is invalid.")
    record.setdefault("writer_capability", None)
    record.setdefault("revision", 0)
    record.setdefault("build", None)
    record.setdefault("degraded_reasons", [])
    if type(record["revision"]) is not int or record["revision"] < 0:
        raise ValueError("ready index revision is invalid.")
    return value, record


def read_ready_index_status(cfg: object) -> dict[str, Any]:
    """Return the durable build, cursor, watermark, and degradation status."""
    try:
        _value, record = read_state_record(cfg)
        return record
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return {
            "schema_version": READY_PROTOCOL_VERSION,
            "state": "degraded",
            "writer_capability": None,
            "revision": 0,
            "build": None,
            "degraded_reasons": ["state_invalid"],
            "updated_at": None,
        }


def commit_state_under_lock(path: Path, value: dict[str, Any], record: dict[str, Any]) -> None:
    """Commit a state record while the caller holds the ready state lock."""
    record["revision"] += 1
    record["updated_at"] = utc_now()
    atomic_replace(path, value)


def schema_capability_path(cfg: object) -> Path:
    """Return the schema writer-capability record path."""
    return shared_paths(cfg.shared_root)["schema"] / "version.json"


def install_writer_capability_gate(cfg: object) -> None:
    """Make pre-ready schema readers reject the root before they can mutate Tasks."""
    path = schema_capability_path(cfg)
    value = read_json(path)
    schema = value.get("schema")
    if not isinstance(schema, dict):
        raise RuntimeError("qexp schema/version.json is malformed.")
    capabilities = schema.get("writer_capabilities")
    if capabilities is None:
        schema["writer_capabilities"] = [READY_WRITER_CAPABILITY]
    elif (
        not isinstance(capabilities, list)
        or not all(isinstance(item, str) for item in capabilities)
        or READY_WRITER_CAPABILITY not in capabilities
    ):
        raise RuntimeError("qexp schema writer capability gate is incompatible.")
    else:
        return
    atomic_replace(path, value)


def assert_ready_writer_compatible(
    cfg: object,
    writer_capability: str | None = READY_WRITER_CAPABILITY,
) -> None:
    """Reject an incompatible writer before authoritative Task mutation."""
    current_state = read_ready_index_state(cfg)
    if current_state == "absent":
        return
    try:
        _value, record = read_state_record(cfg)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("ready index state is invalid; Task mutation is disabled.") from exc
    required = record.get("writer_capability")
    if required != READY_WRITER_CAPABILITY or writer_capability != required:
        raise RuntimeError(
            f"ready index requires writer capability {required!r}; writer declared {writer_capability!r}."
        )
    try:
        schema = read_json(schema_capability_path(cfg))["schema"]
        capabilities = schema["writer_capabilities"]
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("ready writer schema capability gate is missing.") from exc
    if (
        not isinstance(capabilities, list)
        or not all(isinstance(item, str) for item in capabilities)
        or READY_WRITER_CAPABILITY not in capabilities
    ):
        raise RuntimeError("ready writer schema capability gate is incompatible.")


def mark_ready_index_degraded(cfg: object, reason: str) -> None:
    """Fail closed after detecting a corrupt active projection."""
    path = ready_state_path(cfg.shared_root)
    try:
        with exclusive(state_lock_path(cfg)):
            value, record = read_state_record(cfg)
            degrade_state_record(record, reason)
            commit_state_under_lock(path, value, record)
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
        return


def degrade_state_record(record: dict[str, Any], reason: str) -> None:
    """Set a state record to degraded without writing it."""
    reasons = record.get("degraded_reasons", [])
    if not isinstance(reasons, list):
        reasons = []
    if reason not in reasons:
        reasons.append(reason)
    record["state"] = "degraded"
    record["degraded_reasons"] = reasons
