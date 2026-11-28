"""Read machine-local Idle Job status and validate its persisted records."""

from __future__ import annotations

import math
import stat
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..runtime.resources.idle_job import idle_job_instance_path, read_idle_job_ownership
from ..runtime.store import read_json, read_json_limited
from .idle_job_config import load_idle_job_config
from .idle_job_diagnostics import (
    DEMAND_DETAIL_MEMBER,
    INTERVAL_DETAIL_MEMBER,
    MAX_STATUS_BYTES,
    validate_demand_detail,
    validate_interval_detail,
)
from .idle_job_process import current_boot_id

IDLE_JOB_STATUS_VERSION = 1
IDLE_JOB_STATE_VERSION = 1
_ACTIVE_STATES = frozenset({"launching", "prepared", "running", "stopping"})
_STATUS_STATES = frozenset({"disabled", "waiting", "running", "stopping", "backoff", "blocked"})
_STATUS_FIELDS = frozenset(
    {
        "version",
        "state",
        "reason",
        "configured_gpu_ids",
        "resolved_gpu_ids",
        "process",
        "started_at",
        "last_exit_reason",
        "next_retry_at",
        "log_path",
        "config_revision",
        "supervision_active",
        "runtime_id",
        "failure_count",
    }
)
_STATE_FIELDS = frozenset(
    {
        "version",
        "observation_key",
        "idle_since",
        "retry_at",
        "failure_count",
        "last_exit",
        "started_at",
        "log_path",
    }
)


def runtime_root(value: Any) -> Path:
    if isinstance(value, (str, Path)):
        return Path(value).expanduser().resolve()
    candidate = getattr(value, "root", None)
    if candidate is None:
        raise TypeError("runtime must provide a root path")
    return Path(candidate).expanduser().resolve()


def runtime_id_without_initializing(value: Any, root: Path) -> str:
    candidate = getattr(value, "instance_id", None)
    if isinstance(candidate, str) and candidate:
        return candidate
    try:
        raw = read_json(root / "identity.json")
        machine_runtime = raw.get("machine_runtime")
        if isinstance(machine_runtime, dict):
            effective = machine_runtime.get("runtime_id")
            seed = machine_runtime.get("instance_id")
            if isinstance(effective, str) and effective:
                return effective
            if isinstance(seed, str) and seed:
                return seed
    except (FileNotFoundError, OSError, TypeError, ValueError, KeyError):
        pass
    return "machine-runtime"


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def utc_epoch(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        return None
    return parsed.timestamp()


def validate_status(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _STATUS_FIELDS:
        raise ValueError("Idle Job status envelope is malformed")
    if type(value["version"]) is not int or value["version"] != IDLE_JOB_STATUS_VERSION:
        raise ValueError("Idle Job status version is invalid")
    if value["state"] not in _STATUS_STATES:
        raise ValueError("Idle Job status state is invalid")
    if not isinstance(value["reason"], str) or not value["reason"]:
        raise ValueError("Idle Job status reason is invalid")
    for field in ("configured_gpu_ids", "resolved_gpu_ids"):
        ids = value[field]
        if ids is not None and (
            not isinstance(ids, list)
            or any(type(gpu_id) is not int or gpu_id < (-1 if field == "configured_gpu_ids" else 0) for gpu_id in ids)
            or (field == "resolved_gpu_ids" and len(set(ids)) != len(ids))
        ):
            raise ValueError(f"Idle Job status {field} is invalid")
    if value["process"] is not None and not isinstance(value["process"], dict):
        raise ValueError("Idle Job status process is invalid")
    if value["last_exit_reason"] is not None and (
        not isinstance(value["last_exit_reason"], str) or not value["last_exit_reason"]
    ):
        raise ValueError("Idle Job status last_exit_reason is invalid")
    if value["started_at"] is not None and utc_epoch(value["started_at"]) is None:
        raise ValueError("Idle Job status started_at is invalid")
    if value["next_retry_at"] is not None and not _finite_number(value["next_retry_at"]):
        raise ValueError("Idle Job status next_retry_at is invalid")
    if value["log_path"] is not None and (not isinstance(value["log_path"], str) or not value["log_path"]):
        raise ValueError("Idle Job status log_path is invalid")
    revision = value["config_revision"]
    if revision is not None and (type(revision) is not int or revision < 0):
        raise ValueError("Idle Job status config_revision is invalid")
    if type(value["supervision_active"]) is not bool:
        raise ValueError("Idle Job status supervision_active is invalid")
    if value["runtime_id"] is not None and (not isinstance(value["runtime_id"], str) or not value["runtime_id"]):
        raise ValueError("Idle Job status runtime_id is invalid")
    if type(value["failure_count"]) is not int or value["failure_count"] < 0:
        raise ValueError("Idle Job status failure_count is invalid")
    return dict(value)


def validate_state(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"idle_job_state"}:
        raise ValueError("Idle Job state envelope is malformed")
    raw = value["idle_job_state"]
    if not isinstance(raw, dict) or set(raw) != _STATE_FIELDS:
        raise ValueError("Idle Job state fields are malformed")
    if type(raw["version"]) is not int or raw["version"] != IDLE_JOB_STATE_VERSION:
        raise ValueError("Idle Job state version is invalid")
    key = raw["observation_key"]
    if key is not None and (not isinstance(key, str) or len(key) > 4096):
        raise ValueError("Idle Job observation key is invalid")
    for field in ("idle_since", "retry_at"):
        if raw[field] is not None and not _finite_number(raw[field]):
            raise ValueError(f"Idle Job state {field} is invalid")
    if type(raw["failure_count"]) is not int or raw["failure_count"] < 0:
        raise ValueError("Idle Job state failure_count is invalid")
    if raw["last_exit"] is not None and not isinstance(raw["last_exit"], dict):
        raise ValueError("Idle Job state last_exit is invalid")
    if raw["started_at"] is not None and utc_epoch(raw["started_at"]) is None:
        raise ValueError("Idle Job state started_at is invalid")
    if raw["log_path"] is not None and (not isinstance(raw["log_path"], str) or not raw["log_path"]):
        raise ValueError("Idle Job state log_path is invalid")
    return dict(raw)


def status_path(root: Path) -> Path:
    return root / "idle-job" / "status.json"


def state_path(root: Path) -> Path:
    return root / "idle-job" / "state.json"


def _read_status_payload(root: Path) -> dict[str, Any]:
    path = status_path(root)
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ValueError("Idle Job status must be a regular file")
    envelope = read_json_limited(path, max_bytes=MAX_STATUS_BYTES, record_type="idle_job_status")
    payload = validate_status(envelope.get("idle_job_status"))
    payload["demand_detail"] = validate_demand_detail(envelope.get(DEMAND_DETAIL_MEMBER))
    payload["interval_detail"] = validate_interval_detail(envelope.get(INTERVAL_DETAIL_MEMBER))
    return payload


def reports_unknown_demand(payload: Mapping[str, Any]) -> bool:
    return payload["reason"] == "unknown_demand" and payload["state"] in {"waiting", "stopping"}


def _ownership_file_exists(root: Path) -> bool:
    try:
        root.joinpath("idle-job", "ownership.json").lstat()
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return True


def _default_status(runtime_id: str | None, *, state: str, reason: str) -> dict[str, Any]:
    return {
        "version": IDLE_JOB_STATUS_VERSION,
        "state": state,
        "reason": reason,
        "configured_gpu_ids": None,
        "resolved_gpu_ids": None,
        "process": None,
        "started_at": None,
        "last_exit_reason": None,
        "next_retry_at": None,
        "log_path": None,
        "config_revision": None,
        "supervision_active": state in _ACTIVE_STATES,
        "runtime_id": runtime_id,
        "failure_count": 0,
    }


def owner_is_preparing(owner: Mapping[str, Any]) -> bool:
    return owner["version"] == 2 and owner["phase"] in {"prepared", "launching"}


def _idle_job_status(runtime: Any, *, runtime_id: str | None = None) -> dict[str, Any]:
    """Read the independent machine-local Idle Job status projection.

    A malformed status or ownership record is reported as ``blocked``.  The
    function never infers ``disabled`` while an ownership file is present.
    """
    root = runtime_root(runtime)
    runtime_id = runtime_id or runtime_id_without_initializing(runtime, root)
    ownership_exists = _ownership_file_exists(root)
    try:
        ownership = read_idle_job_ownership(root)
    except (OSError, RuntimeError, TypeError, ValueError, KeyError):
        ownership = None

    try:
        payload = _read_status_payload(root)
    except FileNotFoundError:
        if ownership is not None and owner_is_preparing(ownership):
            payload = _default_status(runtime_id, state="waiting", reason="preparing")
        else:
            payload = _default_status(
                runtime_id,
                state="blocked" if ownership_exists else "disabled",
                reason="malformed_status" if ownership_exists else "disabled",
            )
    except (OSError, TypeError, ValueError, KeyError):
        payload = _default_status(runtime_id, state="blocked", reason="malformed_status")
    if payload.get("runtime_id") is None:
        payload["runtime_id"] = runtime_id

    if ownership is None:
        if ownership_exists:
            payload.update(state="blocked", reason="malformed_ownership", supervision_active=True)
            return payload
        try:
            config = load_idle_job_config(runtime)
            payload["configured_gpu_ids"] = list(config.gpu_ids)
            if not config.enabled:
                payload.update(state="disabled", reason="disabled", process=None, started_at=None)
            elif payload["state"] in {"running", "stopping", "disabled"} or payload["reason"] in {
                "launching",
                "preparing",
            }:
                state = validate_state(
                    read_json_limited(state_path(root), max_bytes=65_536, record_type="idle_job_state")
                )
                retry_at = state["retry_at"]
                payload.update(
                    state="backoff" if retry_at is not None and retry_at > time.time() else "waiting",
                    reason="ownership_released",
                    process=None,
                    started_at=None,
                    next_retry_at=retry_at,
                    failure_count=state["failure_count"],
                    last_exit_reason=(state["last_exit"] or {}).get("reason"),
                )
        except FileNotFoundError:
            payload.update(state="waiting", reason="awaiting_supervision", process=None, started_at=None)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError):
            payload.update(state="blocked", reason="config_or_state_unavailable", process=None, started_at=None)
        return payload

    if ownership["runtime_id"] != runtime_id:
        payload["state"] = "blocked"
        payload["reason"] = "foreign_ownership"
        payload["supervision_active"] = True
        return payload
    try:
        boot_id = current_boot_id()
    except (OSError, RuntimeError, TypeError, ValueError):
        boot_id = None
    if boot_id is None or ownership["boot_id"] != boot_id:
        payload["state"] = "blocked"
        payload["reason"] = "foreign_boot"
        payload["supervision_active"] = True
        return payload
    expected_state = ownership["phase"]
    preparing = owner_is_preparing(ownership)
    if preparing:
        payload.update(process=None, started_at=None)
        if payload["state"] != "blocked":
            payload.update(state="waiting", reason="preparing")
    elif payload.get("state") not in {"waiting", "running", "stopping"}:
        visible_state = "waiting" if expected_state == "launching" else expected_state
        payload["state"] = visible_state
    payload["supervision_active"] = True
    payload["runtime_id"] = ownership["runtime_id"]
    payload["config_revision"] = ownership["config_revision"]
    payload["resolved_gpu_ids"] = list(ownership["gpu_ids"])
    payload["log_path"] = str(idle_job_instance_path(root, ownership["instance_id"]) / "command.log")
    if not preparing and (ownership["version"] == 1 or expected_state == "running"):
        payload["started_at"] = payload.get("started_at") or ownership["created_at"]
    return payload


def idle_job_status(runtime: Any, *, runtime_id: str | None = None) -> dict[str, Any]:
    """Read Idle Job status and its last observed interval without refreshing evidence."""
    payload = _idle_job_status(runtime, runtime_id=runtime_id)
    payload["demand_detail"] = payload.get("demand_detail") if reports_unknown_demand(payload) else None
    payload.setdefault("interval_detail", None)
    return payload
