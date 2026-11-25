"""Machine-local supervision for the optional Idle Job.

The supervisor deliberately knows only about machine-local state.  Project
registries, Tasks, and shared roots are outside this module's authority.  A
single durable ownership record fences training admission; this module keeps
that record until the process helper proves that the complete process group is
gone.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import stat
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, ExitStack, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from socket import socket
from subprocess import Popen
from typing import Any

from ..gpu_policy import GpuReservationPolicy, resolve_gpu_policy, validate_gpu_reservation
from ..infrastructure.process import process_start_time_ticks
from ..runtime.locks import exclusive
from ..runtime.paths import machine_runtime_paths
from ..runtime.records import utc_now
from ..runtime.resources.cpu_lane import get_cpu_lane_policy
from ..runtime.resources.idle_job import (
    clear_idle_job_ownership,
    idle_job_guard,
    idle_job_instance_path,
    read_idle_job_demand,
    read_idle_job_ownership,
    write_idle_job_ownership,
)
from ..runtime.resources.reservations import reservation_snapshot
from ..runtime.responsibility_store import DurableIO
from ..runtime.store import atomic_replace, json_encoded_size, read_json, read_json_limited
from .bindings import ProjectBinding
from .config import load_agent_config
from .idle_guard_digest import IDLE_DECISION_SECONDS
from .idle_job_config import (
    IdleJobConfig,
    command_argv,
    load_idle_job_config,
    resolve_idle_job_gpus,
    validate_idle_job_mode,
)
from .idle_job_diagnostics import (
    DEMAND_DETAIL_MEMBER,
    INTERVAL_DETAIL_MEMBER,
    MAX_STATUS_BYTES,
    build_demand_observation,
    demand_check,
    observe_demand_detail,
    validate_demand_detail,
    validate_interval_detail,
)
from .idle_job_process import (
    current_boot_id,
    idle_job_intent_matches_owner,
    idle_job_processes_exited,
    read_idle_job_process,
    request_idle_job_stop,
    spawn_idle_job,
)
from .idle_launch_permit import create_idle_launch_permit, poll_idle_launch_ready, send_idle_launch_permit
from .launch_handoff_state import LaunchHandoffState

IDLE_JOB_STATUS_VERSION = 1
IDLE_JOB_STATE_VERSION = 1
MAX_IDLE_JOB_FAILURE_BACKOFF_SECONDS = 60.0
NORMAL_EXIT_RETRY_SECONDS = 1.0
MAX_RETAINED_IDLE_JOB_INSTANCES = 5
DEFAULT_SHUTDOWN_EXTRA_SECONDS = 5.0
SUPERVISOR_POLL_SECONDS = 0.05
PREPARATION_TIMEOUT_SECONDS = 30.0

_INSTANCE_ID_PATTERN = re.compile(r"[0-9a-f]{32}\Z")
_ACTIVE_STATES = frozenset({"launching", "prepared", "running", "stopping"})
_STATUS_STATES = frozenset({"disabled", "waiting", "running", "stopping", "backoff", "blocked"})
_STOP_REASONS = frozenset(
    {
        "busy",
        "unknown",
        "unknown_demand",
        "disabled",
        "config_changed",
        "gpu_changed",
        "gpu_policy_unavailable",
        "gpu_reserved",
        "on_demand_mode",
        "agent_stop",
        "training_demand",
    }
)

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
_EXIT_FIELDS = frozenset({"version", "runtime_id", "instance_id", "boot_id", "returncode", "reason", "exited_at"})


class _SupervisorBlocked(RuntimeError):
    """Internal signal for a fail-closed local state observation."""


class _GpuReserved(_SupervisorBlocked):
    """The selected GPU is currently occupied by another local reservation."""


class _IdleDecisionRejected(RuntimeError):
    """A covered fence could not authorize the current Idle Job decision."""


@dataclass(frozen=True, slots=True)
class IdleDecisionEvidence:
    """Versioned evidence that must match to preserve continuous idleness."""

    registry_revision: int
    bindings: tuple[ProjectBinding, ...]
    gpu_policy_revision: int
    visible_gpu_ids: tuple[int, ...]
    cpu_policy_revision: int
    cpu_capacity: int
    executor_epoch: str
    checkpoint_digest: str

    def matches_gpu_policy(self, policy: Mapping[str, Any]) -> bool:
        """Check the observed persisted revision and usable inventory."""
        visible = policy.get("visible_gpu_ids")
        return (
            policy.get("revision") == self.gpu_policy_revision
            and isinstance(visible, (list, tuple))
            and tuple(visible) == self.visible_gpu_ids
        )


class _ConfigObservation:
    __slots__ = ("config", "agent_mode")

    def __init__(self, config: IdleJobConfig, agent_mode: str) -> None:
        self.config = config
        self.agent_mode = agent_mode


@dataclass(slots=True)
class _PreparedIdleJob:
    owner: dict[str, Any]
    intent: dict[str, Any]
    directory: Path
    observation_key: str
    idle_since: float
    ready_until: float
    endpoint: socket | None = None
    child: Popen[bytes] | None = None
    ready: bool = False
    permit_sent: bool = False
    spawn_failed: bool = False

    def close_permit(self) -> None:
        endpoint, self.endpoint = self.endpoint, None
        if endpoint is not None:
            endpoint.close()

    def matches(self, owner: dict[str, Any]) -> bool:
        return self.owner == {**owner, "phase": "prepared"}

    def intent_matches(self) -> bool:
        path = self.directory / "intent.json"
        try:
            if not stat.S_ISDIR(self.directory.lstat().st_mode):
                return False
            mode = path.lstat().st_mode
        except FileNotFoundError:
            return False
        if not stat.S_ISREG(mode):
            return False
        return read_json_limited(path, max_bytes=65_536, record_type="idle_job_intent") == {
            "idle_job_intent": self.intent
        }


_UNOBSERVED_OWNER = object()


def _same_owner(left: dict[str, Any] | None, right: object) -> bool:
    if left is None or not isinstance(right, dict):
        return left is right
    return {**left, "phase": right["phase"]} == right


def _root(value: Any) -> Path:
    if isinstance(value, (str, Path)):
        return Path(value).expanduser().resolve()
    candidate = getattr(value, "root", None)
    if candidate is None:
        raise TypeError("runtime must provide a root path")
    return Path(candidate).expanduser().resolve()


def _runtime_id_without_initializing(value: Any, root: Path) -> str:
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


def _safe_instance_id(value: object) -> bool:
    return isinstance(value, str) and _INSTANCE_ID_PATTERN.fullmatch(value) is not None


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _copy_json(value: Any) -> Any:
    """Copy JSON-shaped state without importing a second persistence model."""
    return json.loads(json.dumps(value, ensure_ascii=False))


def _utc_epoch(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        return None
    return parsed.timestamp()


def _instance_directory_is_safe(path: Path) -> bool:
    """Return whether a terminal instance tree contains no symlink surprise."""
    try:
        metadata = path.lstat()
    except OSError:
        return False
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        return False
    try:
        for entry in path.iterdir():
            item_metadata = entry.lstat()
            if stat.S_ISLNK(item_metadata.st_mode):
                return False
            if stat.S_ISDIR(item_metadata.st_mode):
                return False
            if not stat.S_ISREG(item_metadata.st_mode):
                return False
    except OSError:
        return False
    return True


def _read_exit_record(path: Path) -> dict[str, Any] | None:
    try:
        envelope = read_json_limited(path / "exit.json", max_bytes=65_536, record_type="idle_job_exit")
    except FileNotFoundError:
        return None
    raw = envelope.get("idle_job_exit")
    if not isinstance(raw, dict) or set(raw) != _EXIT_FIELDS:
        raise ValueError("Idle Job exit record is malformed")
    if type(raw["version"]) is not int or raw["version"] != 1:
        raise ValueError("Idle Job exit record version is invalid")
    if not isinstance(raw["runtime_id"], str) or not raw["runtime_id"]:
        raise ValueError("Idle Job exit runtime_id is invalid")
    if not _safe_instance_id(raw["instance_id"]):
        raise ValueError("Idle Job exit instance_id is invalid")
    if not isinstance(raw["boot_id"], str) or not raw["boot_id"]:
        raise ValueError("Idle Job exit boot_id is invalid")
    if raw["returncode"] is not None and type(raw["returncode"]) is not int:
        raise ValueError("Idle Job exit returncode is invalid")
    if not isinstance(raw["reason"], str) or not raw["reason"]:
        raise ValueError("Idle Job exit reason is invalid")
    if _utc_epoch(raw["exited_at"]) is None:
        raise ValueError("Idle Job exit exited_at is invalid")
    return dict(raw)


def _read_stop_reason(path: Path) -> str | None:
    try:
        value = read_json_limited(path / "stop.json", max_bytes=65_536, record_type="idle_job_stop")
    except FileNotFoundError:
        return None
    except (OSError, TypeError, ValueError, KeyError):
        return "invalid_stop_request"
    reason = value.get("reason")
    return reason if isinstance(reason, str) and reason else "invalid_stop_request"


def _canonical_status(value: object) -> dict[str, Any]:
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
    if value["started_at"] is not None and _utc_epoch(value["started_at"]) is None:
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


def _canonical_state(value: object) -> dict[str, Any]:
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
    if raw["started_at"] is not None and _utc_epoch(raw["started_at"]) is None:
        raise ValueError("Idle Job state started_at is invalid")
    if raw["log_path"] is not None and (not isinstance(raw["log_path"], str) or not raw["log_path"]):
        raise ValueError("Idle Job state log_path is invalid")
    return dict(raw)


def _status_path(root: Path) -> Path:
    return root / "idle-job" / "status.json"


def _state_path(root: Path) -> Path:
    return root / "idle-job" / "state.json"


def _read_status_payload(root: Path, *, include_diagnostics: bool = False) -> dict[str, Any]:
    path = _status_path(root)
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ValueError("Idle Job status must be a regular file")
    envelope = read_json_limited(path, max_bytes=MAX_STATUS_BYTES, record_type="idle_job_status")
    payload = _canonical_status(envelope.get("idle_job_status"))
    if include_diagnostics:
        payload["demand_detail"] = validate_demand_detail(envelope.get(DEMAND_DETAIL_MEMBER))
        payload["interval_detail"] = validate_interval_detail(envelope.get(INTERVAL_DETAIL_MEMBER))
    return payload


def _reports_unknown_demand(payload: Mapping[str, Any]) -> bool:
    return payload["reason"] == "unknown_demand" and payload["state"] in {"waiting", "stopping"}


def _observation_reset_check(previous: str | None, current: str) -> str:
    if previous is None:
        return "observation_changed"
    try:
        before, after = json.loads(previous), json.loads(current)
        if (
            not isinstance(before, dict)
            or before.keys() != after.keys()
            or before["demand"] != "idle"
            or any(
                type(before[field]) is not int or before[field] < 0 for field in ("demand_revision", "config_revision")
            )
        ):
            return "observation_changed"
        previous_policy = json.loads(before["policy"])
        if not isinstance(previous_policy, dict) or "resolved_gpu_ids" not in previous_policy:
            return "observation_changed"
    except (TypeError, ValueError, RecursionError):
        return "observation_changed"
    for field, check in (
        ("demand_revision", "demand_revision_changed"),
        ("config_revision", "config_revision_changed"),
        ("policy", "gpu_policy_changed"),
    ):
        if before[field] != after[field]:
            return check
    return "observation_changed"


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


def _owner_preparing(owner: Mapping[str, Any]) -> bool:
    return owner["version"] == 2 and owner["phase"] in {"prepared", "launching"}


def _idle_job_status(runtime: Any) -> dict[str, Any]:
    """Read the independent machine-local Idle Job status projection.

    A malformed status or ownership record is reported as ``blocked``.  The
    function never infers ``disabled`` while an ownership file is present.
    """
    root = _root(runtime)
    runtime_id = _runtime_id_without_initializing(runtime, root)
    ownership_exists = _ownership_file_exists(root)
    try:
        ownership = read_idle_job_ownership(root)
    except (OSError, RuntimeError, TypeError, ValueError, KeyError):
        ownership = None
        ownership_malformed = ownership_exists
    else:
        ownership_malformed = False

    try:
        payload = _read_status_payload(root, include_diagnostics=True)
    except FileNotFoundError:
        if ownership is not None and _owner_preparing(ownership):
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

    if ownership_malformed:
        payload["state"] = "blocked"
        payload["reason"] = "malformed_ownership"
        payload["supervision_active"] = True
        return payload
    if ownership is None:
        if ownership_exists:
            payload["state"] = "blocked"
            payload["reason"] = "malformed_ownership"
            payload["supervision_active"] = True
        else:
            try:
                config = load_idle_job_config(runtime)
                payload["configured_gpu_ids"] = list(config.gpu_ids)
                if not config.enabled:
                    payload.update(state="disabled", reason="disabled", process=None, started_at=None)
                elif payload["state"] in {"running", "stopping", "disabled"} or payload["reason"] in {
                    "launching",
                    "preparing",
                }:
                    state = _canonical_state(
                        read_json_limited(_state_path(root), max_bytes=65_536, record_type="idle_job_state")
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
    preparing = _owner_preparing(ownership)
    if preparing:
        payload.update(process=None, started_at=None)
        if payload["state"] != "blocked":
            payload.update(state="waiting", reason="preparing")
    elif payload.get("state") == "disabled" or payload.get("state") not in {"waiting", "running", "stopping"}:
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


def idle_job_status(runtime: Any) -> dict[str, Any]:
    """Read Idle Job status and its last observed interval without refreshing evidence."""
    payload = _idle_job_status(runtime)
    payload["demand_detail"] = payload.get("demand_detail") if _reports_unknown_demand(payload) else None
    payload.setdefault("interval_detail", None)
    return payload


class IdleJobSupervisor:
    """Coordinate one machine-local Idle Job without owning shared work."""

    def __init__(
        self,
        runtime: Any,
        runtime_id: str | None = None,
        *,
        agent_mode: str | None = None,
        clock: Callable[[], float] | None = None,
        monotonic: Callable[[], float] | None = None,
        wall_clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.runtime = runtime
        self.root = _root(runtime)
        self.runtime_id = runtime_id or _runtime_id_without_initializing(runtime, self.root)
        if not isinstance(self.runtime_id, str) or not self.runtime_id or len(self.runtime_id) > 128:
            raise ValueError("runtime_id must be a bounded non-empty string")
        if clock is not None and monotonic is not None:
            raise ValueError("provide clock or monotonic, not both")
        self._monotonic = clock or monotonic or (lambda: time.monotonic())
        self._wall_clock = wall_clock or (lambda: time.time())
        self._sleep = sleep or (lambda seconds: time.sleep(seconds))
        self._agent_mode_override = agent_mode
        self._observation_key: str | None = None
        self._last_handoff_revision: int | None = None
        self._last_decision_evidence: IdleDecisionEvidence | None = None
        self._deferred_stop: tuple[str, str | None, dict[str, Any] | None | object] | None = None
        # Restart interrupts proof of continuous idleness.
        self._idle_since: float | None = None
        self._retry_at: float | None = None
        self._failure_count = 0
        self._last_exit: dict[str, Any] | None = None
        self._started_at: str | None = None
        self._log_path: str | None = None
        self._instance_id: str | None = None
        self._process: dict[str, Any] | None = None
        self._configured_gpu_ids: list[int] | None = None
        self._resolved_gpu_ids: list[int] | None = None
        self._config_revision: int | None = None
        self._status_state = "disabled"
        self._status_reason = "disabled"
        self._pending_demand_check: str | None = None
        self._pending_demand_detail: dict[str, Any] | None = None
        self._pending_agent_instance_id: str | None = None
        self._last_demand_detail: dict[str, Any] | None = None
        self._interval_observed = False
        self._last_interval_reset: dict[str, Any] | None = None
        self._last_stop: dict[str, Any] | None = None
        self._last_stop_instance_id: str | None = None
        self._state_corrupt = False
        self._children = []
        self._preparation: _PreparedIdleJob | None = None
        self._supervision_active = True
        self._load_state()

    def _now(self) -> float:
        value = float(self._monotonic())
        if not math.isfinite(value):
            raise _SupervisorBlocked("monotonic clock is unavailable")
        return value

    def _wall_now(self) -> float:
        value = float(self._wall_clock())
        if not math.isfinite(value):
            raise _SupervisorBlocked("wall clock is unavailable")
        return value

    def _load_state(self) -> None:
        try:
            value = read_json_limited(_state_path(self.root), max_bytes=65_536, record_type="idle_job_state")
        except FileNotFoundError:
            return
        except (OSError, TypeError, ValueError, KeyError):
            self._state_corrupt = True
            return
        try:
            state = _canonical_state(value)
        except (TypeError, ValueError, KeyError):
            self._state_corrupt = True
            return
        self._observation_key = state["observation_key"]
        self._retry_at = float(state["retry_at"]) if state["retry_at"] is not None else None
        self._failure_count = state["failure_count"]
        self._last_exit = _copy_json(state["last_exit"]) if state["last_exit"] is not None else None
        self._started_at = state["started_at"]
        self._log_path = state["log_path"]

    def _persist_state(self) -> None:
        state = {
            "version": IDLE_JOB_STATE_VERSION,
            "observation_key": self._observation_key,
            "idle_since": self._idle_since,
            "retry_at": self._retry_at,
            "failure_count": self._failure_count,
            "last_exit": self._last_exit,
            "started_at": self._started_at,
            "log_path": self._log_path,
        }
        atomic_replace(_state_path(self.root), {"idle_job_state": state})
        self._state_corrupt = False

    def _payload(self, *, state: str | None = None, reason: str | None = None) -> dict[str, Any]:
        selected_state = state or self._status_state
        selected_reason = reason or self._status_reason
        visible_state = "waiting" if selected_state == "launching" else selected_state
        payload = {
            "version": IDLE_JOB_STATUS_VERSION,
            "state": visible_state,
            "reason": selected_reason,
            "configured_gpu_ids": list(self._configured_gpu_ids) if self._configured_gpu_ids is not None else None,
            "resolved_gpu_ids": list(self._resolved_gpu_ids) if self._resolved_gpu_ids is not None else None,
            "process": _copy_json(self._process) if self._process is not None else None,
            "started_at": self._started_at,
            "last_exit_reason": self._last_exit.get("reason") if self._last_exit is not None else None,
            "next_retry_at": self._retry_at,
            "log_path": self._log_path,
            "config_revision": self._config_revision,
            "supervision_active": self._supervision_active,
            "runtime_id": self.runtime_id,
            "failure_count": self._failure_count,
        }
        return _canonical_status(payload)

    def _publish(self, *, state: str, reason: str) -> dict[str, Any]:
        self._status_state = "waiting" if state == "launching" else state
        self._status_reason = reason
        payload = self._payload(state=state, reason=reason)
        # Only a completed decision establishes an interval for reset reporting.
        # A timer can start and clear within one tick while validating reservations.
        self._interval_observed = self._idle_since is not None
        base = {"idle_job_status": payload}
        detail, self._pending_demand_detail = self._pending_demand_detail, None
        self._pending_demand_check = None
        self._pending_agent_instance_id = None
        self._last_demand_detail = None
        if not _reports_unknown_demand(payload):
            detail = None
        try:
            interval = self._interval_detail()
        except Exception:
            interval = None
        enriched = base
        for member, value in ((DEMAND_DETAIL_MEMBER, detail), (INTERVAL_DETAIL_MEMBER, interval)):
            if value is None:
                continue
            try:
                candidate = {**enriched, member: value}
                if json_encoded_size(candidate) <= MAX_STATUS_BYTES:
                    enriched = candidate
            except Exception:
                pass
        path = _status_path(self.root)
        if enriched is not base:
            try:
                atomic_replace(path, enriched)
            except Exception:
                # Optional status must not interrupt supervision if the enriched write fails.
                pass
            else:
                self._last_demand_detail = enriched.get(DEMAND_DETAIL_MEMBER)
                return payload
        atomic_replace(path, base)
        return payload

    def set_demand_observation(
        self, demand: str, detail: Mapping[str, Any] | None, *, agent_instance_id: str | None
    ) -> None:
        """Stage the current observation for the next decision publication."""
        self._pending_agent_instance_id = agent_instance_id
        self._pending_demand_check = None
        self._pending_demand_detail = None
        if demand == "unknown" and isinstance(detail, Mapping):
            try:
                self._pending_demand_check = demand_check(detail.get("code"))
                if agent_instance_id is not None:
                    self._pending_demand_detail = observe_demand_detail(
                        detail,
                        agent_instance_id=agent_instance_id,
                        observed_at=self._wall_now(),
                        previous=self._last_demand_detail,
                    )
            except Exception:
                pass
        if demand == "unknown" and self._pending_demand_detail is None:
            self._last_demand_detail = None

    def _blocked(self, reason: str) -> dict[str, Any]:
        self._close_prepared_permit()
        self._clear_idle_interval(reason)
        self._status_state = "blocked"
        self._status_reason = reason
        try:
            return self._publish(state="blocked", reason=reason)
        except (OSError, RuntimeError, TypeError, ValueError):
            return self._payload(state="blocked", reason=reason)

    def _load_config(self) -> _ConfigObservation:
        config = load_idle_job_config(self.runtime)
        mode = self._agent_mode_override
        try:
            loaded_mode = load_agent_config(self.runtime).agent_mode
        except (OSError, RuntimeError, TypeError, ValueError, KeyError):
            if mode is None and config.enabled:
                raise
            if mode is None:
                mode = "daemon"
        else:
            mode = loaded_mode
        if not isinstance(mode, str):
            raise ValueError("agent mode is invalid")
        validate_idle_job_mode(config, mode)
        return _ConfigObservation(config, mode)

    def _set_config_fields(self, config: IdleJobConfig) -> None:
        self._configured_gpu_ids = list(config.gpu_ids)
        self._config_revision = config.revision

    def _policy_mapping(
        self, gpu_policy: GpuReservationPolicy | None
    ) -> tuple[Mapping[str, Any], GpuReservationPolicy]:
        if not isinstance(gpu_policy, GpuReservationPolicy):
            raise ValueError("Idle Job requires a current GPU discovery and policy observation")
        value = resolve_gpu_policy(
            self.root,
            discovery=gpu_policy.discovery,
            environment_gpu_ids=gpu_policy.environment_gpu_ids,
            environment_status=gpu_policy.environment_status,
        ).to_dict()
        return value, gpu_policy

    def _resolve_gpus(
        self, config: IdleJobConfig, gpu_policy: Any, *, decision_evidence: IdleDecisionEvidence | None = None
    ) -> tuple[tuple[int, ...], Mapping[str, Any], GpuReservationPolicy | None]:
        policy, reservation_policy = self._policy_mapping(gpu_policy)
        if decision_evidence is not None and not decision_evidence.matches_gpu_policy(policy):
            raise _IdleDecisionRejected("demand_census_unknown")
        visible = policy.get("visible_gpu_ids")
        if visible is None and "gpu_ids" in policy:
            visible = policy["gpu_ids"]
        if visible is None:
            raise ValueError("GPU policy has no visible GPU inventory")
        if policy.get("visible_status") in {"pending", "unavailable", "invalid"}:
            raise ValueError("GPU policy visibility is unavailable")
        resolved = resolve_idle_job_gpus(config.gpu_ids, visible)
        reserved = policy.get("reserved_gpu_ids", [])
        if not isinstance(reserved, (list, tuple, set, frozenset)) or any(
            type(gpu_id) is not int or gpu_id < 0 for gpu_id in reserved
        ):
            raise ValueError("GPU policy reserved_gpu_ids is invalid")
        if set(resolved).intersection(reserved):
            raise _GpuReserved("selected GPU is reserved")
        return resolved, policy, reservation_policy

    def _policy_key(self, policy: Mapping[str, Any], resolved: tuple[int, ...]) -> str:
        fields = {
            "revision": policy.get("revision"),
            "visible_gpu_ids": policy.get("visible_gpu_ids"),
            "resolved_gpu_ids": list(resolved),
            "visible_status": policy.get("visible_status"),
        }
        try:
            return json.dumps(fields, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            return json.dumps({"resolved_gpu_ids": list(resolved)}, sort_keys=True)

    def _observation_key_for(
        self,
        demand: str,
        demand_revision: int,
        config: IdleJobConfig,
        policy_key: str,
    ) -> str:
        return json.dumps(
            {
                "demand": demand,
                "demand_revision": demand_revision,
                "config_revision": config.revision,
                "policy": policy_key,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def _reset_observation_if_changed(self, key: str, now_wall: float) -> bool:
        changed = key != self._observation_key
        if changed:
            interval_start = self._now()
            self._record_interval_reset("observation_changed", replacement_key=key, now=interval_start)
            self._observation_key = key
            self._idle_since = interval_start
            self._retry_at = None
            self._failure_count = 0
            self._persist_state()
        elif self._idle_since is None:
            self._idle_since = self._now()
            self._persist_state()
        return changed

    def _diagnostic_timestamp(self) -> str:
        return datetime.fromtimestamp(self._wall_now(), timezone.utc).isoformat().replace("+00:00", "Z")

    def _record_interval_reset(
        self, check: str, *, replacement_key: str | None = None, now: float | None = None
    ) -> None:
        if self._idle_since is None or not self._interval_observed:
            return
        self._interval_observed = False
        self._last_interval_reset = None
        try:
            if replacement_key is not None:
                check = _observation_reset_check(self._observation_key, replacement_key)
            elif check == "unknown_demand":
                check = self._pending_demand_check or check
            observed = self._now() if now is None else now
            elapsed = max(0.0, observed - self._idle_since)
            if math.isfinite(elapsed):
                self._last_interval_reset = {
                    "check": check,
                    "elapsed_seconds": elapsed,
                    "reset_at": self._diagnostic_timestamp(),
                }
        except Exception:
            # Optional evidence cannot interrupt interval invalidation.
            pass

    def _clear_idle_interval(self, check: str, *, replacement_key: str | None = None) -> None:
        self._record_interval_reset(check, replacement_key=replacement_key)
        self._idle_since = None
        self._last_decision_evidence = None

    def _record_stop_request(self, instance_id: str, reason: str) -> None:
        if self._last_stop_instance_id == instance_id:
            return
        self._last_stop_instance_id = instance_id
        self._last_stop = None
        try:
            check = self._pending_demand_check if reason == "unknown_demand" else None
            self._last_stop = {
                "check": check or reason,
                "instance_id": instance_id,
                "stopped_at": self._diagnostic_timestamp(),
            }
        except Exception:
            # The persisted stop request remains authoritative.
            pass

    def _interval_detail(self) -> dict[str, Any] | None:
        if self._idle_since is None and self._last_interval_reset is None and self._last_stop is None:
            return None
        elapsed = max(0.0, self._now() - self._idle_since) if self._idle_since is not None else 0.0
        return validate_interval_detail(
            {"idle_elapsed_seconds": elapsed, "last_reset": self._last_interval_reset, "last_stop": self._last_stop}
        )

    def _close_prepared_permit(self) -> None:
        if self._preparation is not None:
            self._preparation.close_permit()

    def _set_owner_fields(self, owner: dict[str, Any], process: dict[str, Any] | None) -> None:
        self._instance_id = owner["instance_id"]
        self._resolved_gpu_ids = list(owner["gpu_ids"])
        self._config_revision = owner["config_revision"]
        self._started_at = owner["created_at"] if owner["version"] == 1 or process is not None else None
        self._log_path = str(idle_job_instance_path(self.root, owner["instance_id"]) / "command.log")
        self._process = _copy_json(process) if process is not None else None

    def _terminal_exit(self, owner: dict[str, Any], exit_record: dict[str, Any] | None, fallback: str) -> None:
        if self._last_exit is not None and self._last_exit.get("instance_id") == owner["instance_id"]:
            self._persist_state()
            return
        record = (
            dict(exit_record)
            if exit_record is not None
            else {
                "version": 1,
                "runtime_id": owner["runtime_id"],
                "instance_id": owner["instance_id"],
                "boot_id": owner["boot_id"],
                "returncode": None,
                "reason": fallback,
                "exited_at": utc_now(),
            }
        )
        self._last_exit = record
        self._process = None
        stop_reason = record.get("reason") in _STOP_REASONS or record.get("reason") in {
            "parent_exit",
            "supervisor_signal",
            "invalid_stop_request",
        }
        if stop_reason:
            self._retry_at = None
        elif record.get("returncode") == 0:
            self._failure_count = 0
            self._retry_at = self._wall_now() + NORMAL_EXIT_RETRY_SECONDS
        else:
            self._failure_count += 1
            delay = min(2.0 ** min(self._failure_count - 1, 6), MAX_IDLE_JOB_FAILURE_BACKOFF_SECONDS)
            self._retry_at = self._wall_now() + delay
        self._persist_state()
        self._status_state = "backoff" if self._retry_at is not None else "waiting"
        self._status_reason = record.get("reason") or fallback

    def _clear_owner_after_exit(self, owner: dict[str, Any], *, fallback: str) -> bool:
        instance_dir = idle_job_instance_path(self.root, owner["instance_id"])
        try:
            exit_record = _read_exit_record(instance_dir)
            if exit_record is not None and any(
                exit_record[key] != owner[key] for key in ("instance_id", "runtime_id", "boot_id")
            ):
                return False
            self._terminal_exit(owner, exit_record, fallback)
            if exit_record is None:
                atomic_replace(instance_dir / "exit.json", {"idle_job_exit": self._last_exit})
            clear_idle_job_ownership(self.root, owner["instance_id"])
            current = read_idle_job_ownership(self.root)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError):
            return False
        if current is not None:
            return False
        if self._preparation is not None and self._preparation.matches(owner):
            self._close_prepared_permit()
            self._preparation = None
        self._instance_id = None
        self._started_at = None
        self._log_path = self._log_path or str(instance_dir / "command.log")
        try:
            self._prune_terminal_instances_locked()
        except (OSError, RuntimeError, TypeError, ValueError, KeyError):
            pass
        return True

    def _reconcile_locked(self, boot_id: str) -> tuple[dict[str, Any] | None, str | None]:
        try:
            owner = read_idle_job_ownership(self.root)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError):
            return None, "malformed_ownership"
        preparation = self._preparation
        if preparation is not None and (owner is None or not preparation.matches(owner)):
            preparation.close_permit()
            self._preparation = preparation = None
        if owner is None:
            self._instance_id = None
            self._process = None
            return None, None
        if owner["runtime_id"] != self.runtime_id:
            self._set_owner_fields(owner, None)
            return owner, "foreign_ownership"
        instance_dir = idle_job_instance_path(self.root, owner["instance_id"])
        try:
            metadata = instance_dir.lstat()
        except FileNotFoundError:
            self._set_owner_fields(owner, None)
            return owner, "cleanup_uncertain"
        except OSError:
            self._set_owner_fields(owner, None)
            return owner, "cleanup_uncertain"
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            self._set_owner_fields(owner, None)
            return owner, "foreign_instance"
        try:
            if not idle_job_intent_matches_owner(instance_dir, owner):
                return owner, "foreign_instance"
            process = read_idle_job_process(instance_dir)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError):
            self._set_owner_fields(owner, None)
            return owner, "malformed_process"
        if process is not None and (
            process.get("runtime_id") != self.runtime_id
            or process.get("instance_id") != owner["instance_id"]
            or process.get("boot_id") != owner["boot_id"]
        ):
            self._set_owner_fields(owner, process)
            return owner, "foreign_process"
        try:
            exited = idle_job_processes_exited(instance_dir)
            if (
                not exited
                and preparation is not None
                and (
                    preparation.spawn_failed or (preparation.child is not None and preparation.child.poll() is not None)
                )
            ):
                exited = self._seal_unstarted_locked(preparation)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError):
            self._set_owner_fields(owner, process)
            return owner, "cleanup_uncertain"
        if exited:
            if not self._clear_owner_after_exit(owner, fallback="natural_exit"):
                self._set_owner_fields(owner, process)
                return owner, "cleanup_uncertain"
            return None, None
        command_started = process is not None and (owner["version"] == 1 or process.get("command_started") is True)
        if owner["phase"] == "running" and not command_started:
            self._set_owner_fields(owner, None)
            return owner, "process_unregistered"
        if owner["phase"] == "prepared" and command_started:
            self._set_owner_fields(owner, None)
            return owner, "unexpected_command_start"
        if owner["phase"] == "launching" and command_started:
            running = dict(owner)
            running["phase"] = "running"
            try:
                write_idle_job_ownership(self.root, running)
            except (OSError, RuntimeError, TypeError, ValueError, KeyError):
                self._set_owner_fields(owner, process)
                return owner, "ownership_update_failed"
            owner = running
        self._set_owner_fields(owner, process if command_started else None)
        return owner, None

    def _stop_locked(
        self, reason: str, owner: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any] | None, str | None]:
        self._close_prepared_permit()
        self._clear_idle_interval(reason)
        self._observation_key = None
        current = owner
        if current is None:
            try:
                current = read_idle_job_ownership(self.root)
            except (OSError, RuntimeError, TypeError, ValueError, KeyError):
                return None, "malformed_ownership"
        if current is None:
            self._deferred_stop = None
            return None, None
        if current["runtime_id"] != self.runtime_id:
            self._set_owner_fields(current, None)
            return current, "foreign_ownership"
        instance_dir = idle_job_instance_path(self.root, current["instance_id"])
        already_stopping = current["phase"] == "stopping"
        if not already_stopping or _read_stop_reason(instance_dir) is None:
            first_request = False
            if not already_stopping:
                try:
                    (instance_dir / "stop.json").lstat()
                except FileNotFoundError:
                    first_request = True
                except Exception:
                    # An unreadable marker cannot prove who initiated the stop.
                    pass
            try:
                request_idle_job_stop(instance_dir, reason)
                if first_request:
                    self._record_stop_request(current["instance_id"], reason)
                stopping = dict(current)
                stopping["phase"] = "stopping"
                write_idle_job_ownership(self.root, stopping)
                current = stopping
            except (OSError, RuntimeError, TypeError, ValueError, KeyError):
                self._set_owner_fields(current, None)
                return current, "stop_request_failed"
        self._set_owner_fields(current, self._process)
        self._deferred_stop = None
        return current, None

    def _make_instance_directory(self, instance_id: str) -> Path:
        instances = self.root / "idle-job" / "instances"
        if instances.exists():
            metadata = instances.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise _SupervisorBlocked("Idle Job instance root is unsafe")
        else:
            instances.mkdir(parents=True, exist_ok=True)
        path = idle_job_instance_path(self.root, instance_id)
        if path.exists() or path.is_symlink():
            raise _SupervisorBlocked("Idle Job instance already exists")
        path.mkdir()
        durability = DurableIO()
        for parent in (instances, instances.parent, self.root):
            durability.sync_directory(parent, "idle_job_prepare")
        return path

    def _record_failure_locked(self, reason: str, *, error: BaseException | None = None) -> dict[str, Any]:
        self._failure_count += 1
        delay = min(2.0 ** min(self._failure_count - 1, 6), MAX_IDLE_JOB_FAILURE_BACKOFF_SECONDS)
        self._retry_at = self._wall_now() + delay
        record = {
            "version": 1,
            "runtime_id": self.runtime_id,
            "instance_id": self._instance_id or uuid.uuid4().hex,
            "boot_id": None,
            "returncode": None,
            "reason": reason,
            "exited_at": utc_now(),
        }
        if error is not None:
            record["error"] = str(error)[:1024]
        self._last_exit = record
        self._process = None
        self._persist_state()
        return self._publish(state="backoff", reason=reason)

    def _persist_preparation_locked(
        self, config: IdleJobConfig, resolved: tuple[int, ...], boot_id: str
    ) -> _PreparedIdleJob:
        instance_id = uuid.uuid4().hex
        parent_pid = os.getpid()
        parent_start_ticks = process_start_time_ticks(parent_pid)
        idle_since = self._idle_since
        observation_key = self._observation_key
        if parent_start_ticks is None or idle_since is None or observation_key is None:
            raise _SupervisorBlocked("cannot establish Idle Job preparation identity")
        instance_dir = self._make_instance_directory(instance_id)
        owner = {
            "version": 2,
            "runtime_id": self.runtime_id,
            "boot_id": boot_id,
            "instance_id": instance_id,
            "phase": "prepared",
            "config_revision": config.revision,
            "gpu_ids": list(resolved),
            "parent_pid": parent_pid,
            "parent_start_ticks": parent_start_ticks,
            "created_at": utc_now(),
        }
        intent = {
            "version": 2,
            "runtime_id": self.runtime_id,
            "instance_id": instance_id,
            "command": list(command_argv(config.command)),
            "working_directory": str(Path(config.working_directory) if config.working_directory else self.root),
            "gpu_ids": list(resolved),
            "stop_timeout_seconds": config.stop_timeout_seconds,
            "parent_pid": parent_pid,
            "parent_start_ticks": parent_start_ticks,
            "boot_id": boot_id,
        }
        preparation = _PreparedIdleJob(
            owner, intent, instance_dir, observation_key, idle_since, time.monotonic() + PREPARATION_TIMEOUT_SECONDS
        )
        self._preparation = preparation
        atomic_replace(instance_dir / "intent.json", {"idle_job_intent": intent})
        write_idle_job_ownership(self.root, owner)
        self._set_owner_fields(owner, None)
        return preparation

    def _seal_unstarted_locked(self, preparation: _PreparedIdleJob) -> bool:
        """Seal an exact preparation after its only spawn attempt was ruled out or reaped."""
        child = preparation.child
        if (child is None and not preparation.spawn_failed) or (child is not None and child.poll() is None):
            return False
        if not preparation.intent_matches():
            return False
        for name in ("supervisor.json", "process.json"):
            try:
                (preparation.directory / name).lstat()
            except FileNotFoundError:
                continue
            return False
        exit_path = preparation.directory / "exit.json"
        try:
            exit_path.lstat()
        except FileNotFoundError:
            atomic_replace(
                exit_path,
                {
                    "idle_job_exit": {
                        "version": 1,
                        "runtime_id": preparation.owner["runtime_id"],
                        "instance_id": preparation.owner["instance_id"],
                        "boot_id": preparation.owner["boot_id"],
                        "returncode": None,
                        "reason": "launch_failed",
                        "exited_at": utc_now(),
                    }
                },
            )
        return idle_job_processes_exited(preparation.directory)

    def _preparation_failed_locked(self, error: BaseException) -> dict[str, Any]:
        preparation = self._preparation
        if preparation is not None:
            preparation.spawn_failed = True
        self._close_prepared_permit()
        owner = read_idle_job_ownership(self.root)
        if owner is not None:
            if (
                preparation is None
                or not preparation.matches(owner)
                or not self._seal_unstarted_locked(preparation)
                or not self._clear_owner_after_exit(owner, fallback="launch_failed")
            ):
                return self._blocked("cleanup_uncertain")
            return self._publish(state="backoff", reason="launch_failed")
        self._preparation = None
        self._instance_id = None
        if preparation is not None:
            self._seal_unstarted_locked(preparation)
        return self._record_failure_locked("launch_failed", error=error)

    def _spawn_prepared(self, preparation: _PreparedIdleJob) -> None:
        """Start a gated helper after relinquishing the Project and admission locks."""
        if time.monotonic() >= preparation.ready_until:
            raise _SupervisorBlocked("Idle Job preparation expired before spawn")
        controller, helper = create_idle_launch_permit()
        preparation.endpoint = controller
        intent = preparation.intent
        try:
            with helper:
                preparation.child = spawn_idle_job(
                    preparation.directory,
                    runtime_id=intent["runtime_id"],
                    instance_id=intent["instance_id"],
                    command=tuple(intent["command"]),
                    working_directory=Path(intent["working_directory"]),
                    gpu_ids=tuple(intent["gpu_ids"]),
                    stop_timeout_seconds=intent["stop_timeout_seconds"],
                    parent_pid=intent["parent_pid"],
                    parent_start_ticks=intent["parent_start_ticks"],
                    permit_socket=helper,
                )
            self._children.append(preparation.child)
        except BaseException:
            preparation.close_permit()
            raise

    def _verified_terminal(self, path: Path, boot_id: str) -> tuple[float, str] | None:
        if not _safe_instance_id(path.name) or not _instance_directory_is_safe(path):
            return None
        try:
            intent_envelope = read_json_limited(path / "intent.json", max_bytes=65_536, record_type="idle_job_intent")
            intent = intent_envelope.get("idle_job_intent")
            exit_record = _read_exit_record(path)
        except (OSError, TypeError, ValueError, KeyError):
            return None
        if not isinstance(intent, dict) or exit_record is None:
            return None
        if (
            intent.get("runtime_id") != self.runtime_id
            or intent.get("instance_id") != path.name
            or exit_record.get("runtime_id") != self.runtime_id
            or exit_record.get("instance_id") != path.name
            or exit_record.get("boot_id") != intent.get("boot_id")
        ):
            return None
        exited_at = _utc_epoch(exit_record.get("exited_at"))
        if exited_at is None:
            return None
        if not idle_job_processes_exited(path):
            return None
        return exited_at, path.name

    def _prune_terminal_instances_locked(self) -> None:
        instances = self.root / "idle-job" / "instances"
        try:
            metadata = instances.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            return
        boot_id = current_boot_id()
        verified: list[tuple[float, Path]] = []
        for entry in instances.iterdir():
            item_metadata = entry.lstat()
            if stat.S_ISLNK(item_metadata.st_mode) or not stat.S_ISDIR(item_metadata.st_mode):
                continue
            terminal = self._verified_terminal(entry, boot_id)
            if terminal is not None and entry.name != self._instance_id:
                verified.append((terminal[0], entry))
        verified.sort(key=lambda item: item[0])
        for _timestamp, path in verified[: max(0, len(verified) - MAX_RETAINED_IDLE_JOB_INSTANCES)]:
            if not _instance_directory_is_safe(path):
                continue
            try:
                shutil.rmtree(path)
            except OSError:
                continue

    def _status_after_no_owner(self, config: IdleJobConfig, *, state: str, reason: str) -> dict[str, Any]:
        self._set_config_fields(config)
        self._resolved_gpu_ids = self._resolved_gpu_ids if state == "waiting" else None
        self._process = None
        self._started_at = None
        self._instance_id = None
        return self._publish(state=state, reason=reason)

    def _defer_stop(
        self, reason: str, expected_owner: dict[str, Any] | None | object = _UNOBSERVED_OWNER
    ) -> dict[str, Any]:
        self._close_prepared_permit()
        if self._deferred_stop is None:
            self._deferred_stop = (reason, self._pending_demand_check, expected_owner)
        self._clear_idle_interval(reason)
        self._observation_key = None
        if self._instance_id is None and reason == "unknown_demand":
            return self._publish(state="waiting", reason=reason)
        return self._blocked("ownership_reconciliation_failed")

    def _settle_stop(
        self,
        boot_id: str,
        config: IdleJobConfig | None,
        reason: str,
        *,
        resume_if_unowned: bool = False,
        expected_owner: dict[str, Any] | None | object = _UNOBSERVED_OWNER,
    ) -> dict[str, Any] | None:
        """Stop under the ownership fence, deferring an unverified owner to a later tick."""
        self._close_prepared_permit()
        try:
            with idle_job_guard(self.root, blocking=False) as acquired:
                if not acquired:
                    return self._defer_stop(reason, expected_owner)
                if expected_owner is not _UNOBSERVED_OWNER and not _same_owner(
                    read_idle_job_ownership(self.root), expected_owner
                ):
                    self._deferred_stop = None
                    self._clear_idle_interval(reason)
                    self._observation_key = None
                    self._last_handoff_revision = None
                    return self._blocked("ownership_changed")
                owner, error = self._reconcile_locked(boot_id)
                if error is not None:
                    return self._blocked(error)
                if resume_if_unowned and owner is None:
                    self._deferred_stop = None
                    return None

                owner, error = self._stop_locked(reason, owner)
                if error is not None:
                    return self._blocked(error)
                if reason in {"unknown_demand", "busy"}:
                    self._last_handoff_revision = None
                if reason == "config_invalid":
                    if config is None and owner is None:
                        self._configured_gpu_ids = None
                        self._config_revision = None
                    return self._blocked(reason)
                if owner is not None:
                    return self._publish(state="stopping", reason=reason)
                assert config is not None
                if reason == "gpu_policy_unavailable":
                    return self._blocked(reason)
                if reason != "gpu_reserved":
                    self._resolved_gpu_ids = None
                return self._status_after_no_owner(
                    config, state="disabled" if reason == "disabled" else "waiting", reason=reason
                )
        except (OSError, RuntimeError, TypeError, ValueError, KeyError):
            return self._blocked("supervision_failed")

    def _reject_idle_decision(
        self, boot_id: str, config: IdleJobConfig, check: str, owner: dict[str, Any] | None
    ) -> dict[str, Any]:
        self.set_demand_observation(
            "unknown", build_demand_observation(check), agent_instance_id=self._pending_agent_instance_id
        )
        status = self._settle_stop(boot_id, config, "unknown_demand", expected_owner=owner)
        assert status is not None
        return status

    def _observe_local_idle_locked(
        self,
        owner: dict[str, Any] | None,
        observation: _ConfigObservation,
        policy_key: str,
        resolved: tuple[int, ...],
        reservation_policy: GpuReservationPolicy | None,
        evidence: IdleDecisionEvidence | None,
    ) -> tuple[dict[str, Any] | None, str, str | None]:
        """Inspect local fences without publishing state or reconciling ownership."""
        current_owner = read_idle_job_ownership(self.root)
        if not _same_owner(current_owner, owner):
            raise _IdleDecisionRejected("demand_census_unknown")
        owner = current_owner
        current = self._load_config()
        demand_revision = read_idle_job_demand(self.root)
        key = self._observation_key_for("idle", demand_revision, current.config, policy_key)
        if current.config != observation.config or current.agent_mode != observation.agent_mode:
            return owner, key, "config_changed"
        if owner is not None:
            if owner["phase"] == "stopping":
                return owner, key, "stopping"
            if owner["config_revision"] != current.config.revision:
                return owner, key, "config_changed"
            if tuple(owner["gpu_ids"]) != resolved:
                return owner, key, "gpu_changed"
            if self._idle_since is not None and self._observation_key != key:
                raise _IdleDecisionRejected("demand_census_unknown")
        if evidence is not None:
            cpu_policy = get_cpu_lane_policy(self.root, blocking=False)
            if (
                cpu_policy.revision != evidence.cpu_policy_revision
                or cpu_policy.capacity != evidence.cpu_capacity
                or ((self._idle_since is not None or owner is not None) and self._last_decision_evidence != evidence)
            ):
                raise _IdleDecisionRejected("demand_census_unknown")
        try:
            snapshot = reservation_snapshot(self.root, blocking=False, reconcile_expired=False)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            raise _IdleDecisionRejected("demand_census_unknown") from exc
        if snapshot.active or snapshot.provisional:
            return owner, key, "gpu_reserved"
        paths = machine_runtime_paths(self.root)
        with exclusive(paths["reservation_lock"], blocking=False) as acquired:
            if not acquired:
                raise _IdleDecisionRejected("demand_census_unknown")
            current_policy, _ = self._policy_mapping(reservation_policy)
            if self._policy_key(current_policy, resolved) != policy_key or (
                evidence is not None and not evidence.matches_gpu_policy(current_policy)
            ):
                raise _IdleDecisionRejected("demand_census_unknown")
            try:
                if reservation_policy is not None:
                    validate_gpu_reservation(self.root, list(resolved), reservation_policy)
            except (OSError, RuntimeError, TypeError, ValueError, KeyError):
                return owner, key, "gpu_policy_unavailable"
        return owner, key, None

    def _finish_idle_observation_locked(
        self,
        owner: dict[str, Any] | None,
        config: IdleJobConfig,
        resolved: tuple[int, ...],
        key: str,
        reason: str | None,
        launch_allowed: bool,
        boot_id: str,
        now_wall: float,
    ) -> dict[str, Any] | _PreparedIdleJob:
        """Apply the observed outcome with the local admission fence still held."""
        if reason is not None:
            self._clear_idle_interval(reason, replacement_key=key if key != self._observation_key else None)
            if reason == "stopping":
                self._close_prepared_permit()
                return self._publish(state="stopping", reason=reason)
            if owner is not None:
                _, error = self._stop_locked(reason, owner)
                if error is not None:
                    return self._blocked(error)
                return self._publish(state="stopping", reason=reason)
            if reason == "gpu_policy_unavailable":
                return self._blocked(reason)
            self._resolved_gpu_ids = list(resolved)
            return self._publish(state="waiting", reason=reason)
        if owner is not None:
            if _owner_preparing(owner):
                return self._publish(state="waiting", reason="preparing")
            return self._publish(state=owner["phase"], reason=owner["phase"])
        if not launch_allowed:
            if key != self._observation_key:
                self._clear_idle_interval("observation_changed", replacement_key=key)
                self._observation_key = key
            return self._publish(state="waiting", reason="partition_scan_pending")

        self._reset_observation_if_changed(key, now_wall)
        self._resolved_gpu_ids = list(resolved)
        if self._idle_since is None or self._now() - self._idle_since < config.idle_delay_seconds:
            return self._publish(state="waiting", reason="idle_delay")
        if self._retry_at is not None and now_wall < self._retry_at:
            return self._publish(state="backoff", reason="backoff")
        try:
            return self._persist_preparation_locked(config, resolved, boot_id)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            return self._preparation_failed_locked(exc)

    def tick(
        self,
        demand: str = "idle",
        gpu_policy: Any = None,
        *,
        launch_allowed: bool = True,
        expected_handoff_revision: int | None = None,
        decision_evidence: IdleDecisionEvidence | None = None,
        decision_guard: AbstractContextManager[bool] | None = None,
        decision_deadline: float | None = None,
    ) -> dict[str, Any]:
        """Advance supervision once and return the flat status payload."""
        self._children = [child for child in self._children if child.poll() is None]
        observed_demand = demand if demand in {"idle", "busy", "unknown"} else "unknown"
        guarded = decision_evidence is not None and observed_demand == "idle"
        try:
            boot_id = current_boot_id()
            now_wall = self._wall_now()
        except (OSError, RuntimeError, TypeError, ValueError, _SupervisorBlocked):
            return self._blocked("boot_identity_unavailable")
        if self._state_corrupt:
            return self._blocked("malformed_state")

        try:
            observation = self._load_config()
            config = observation.config
        except (OSError, RuntimeError, TypeError, ValueError, KeyError):
            status = self._settle_stop(boot_id, None, "config_invalid")
            assert status is not None
            return status
        self._set_config_fields(config)

        resume_if_unowned = False
        expected_stop_owner = _UNOBSERVED_OWNER
        if not config.enabled:
            stop_reason = "disabled"
        elif observed_demand == "busy":
            stop_reason = "busy"
        elif observed_demand == "unknown":
            stop_reason = "unknown_demand"
        elif self._deferred_stop is not None:
            stop_reason, self._pending_demand_check, expected_stop_owner = self._deferred_stop
            resume_if_unowned = True
        else:
            stop_reason = None
        if stop_reason is not None:
            status = self._settle_stop(
                boot_id, config, stop_reason, resume_if_unowned=resume_if_unowned, expected_owner=expected_stop_owner
            )
            if status is not None:
                return status

        if guarded:
            # A selector outside the observed inventory already requires local
            # stopping, even if admission is too busy to reconcile the owner.
            try:
                resolve_idle_job_gpus(config.gpu_ids, decision_evidence.visible_gpu_ids)
            except ValueError:
                status = self._settle_stop(boot_id, config, "gpu_policy_unavailable")
                assert status is not None
                return status

        try:
            with idle_job_guard(self.root, blocking=False) as acquired:
                if not acquired:
                    return self._defer_stop("unknown_demand")
                owner, error = self._reconcile_locked(boot_id)
                if error is not None:
                    return self._blocked(error)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError):
            return self._blocked("ownership_reconciliation_failed")

        preparation = self._preparation
        if (
            owner is not None
            and owner["phase"] == "prepared"
            and (
                preparation is None
                or (not preparation.permit_sent and preparation.endpoint is None)
                or (not preparation.permit_sent and time.monotonic() >= preparation.ready_until)
            )
        ):
            return self._reject_idle_decision(boot_id, config, "demand_census_unknown", owner)
        if (
            not launch_allowed
            and owner is not None
            and (
                owner["phase"] not in {"prepared", "stopping"} or (preparation is not None and preparation.permit_sent)
            )
        ):
            return self._reject_idle_decision(boot_id, config, "demand_census_unknown", owner)

        try:
            resolved, policy, reservation_policy = (
                self._resolve_gpus(config, gpu_policy, decision_evidence=decision_evidence)
                if guarded
                else self._resolve_gpus(config, gpu_policy)
            )
            policy_key = self._policy_key(policy, resolved)
        except _IdleDecisionRejected as exc:
            return self._reject_idle_decision(boot_id, config, str(exc), owner)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            reason = "gpu_reserved" if isinstance(exc, _GpuReserved) else "gpu_policy_unavailable"
            status = self._settle_stop(boot_id, config, reason, expected_owner=owner)
            assert status is not None
            return status

        try:
            if preparation is not None and preparation.endpoint is not None and not preparation.ready:
                preparation.ready = poll_idle_launch_ready(preparation.endpoint)
            handoffs = getattr(self.runtime, "pending_launch_handoffs", None)
            has_handoff_fence = isinstance(handoffs, LaunchHandoffState) and expected_handoff_revision is not None
            handoff_guard = handoffs.read_guard(blocking=False) if has_handoff_fence else nullcontext(True)
            paths = machine_runtime_paths(self.root)
            deadline = decision_deadline if guarded else time.monotonic() + IDLE_DECISION_SECONDS
            if deadline is None:
                raise _IdleDecisionRejected("demand_census_unknown")
            with ExitStack() as admission:
                with ExitStack() as local:
                    with ExitStack() as project:
                        if guarded and (decision_guard is None or not project.enter_context(decision_guard)):
                            raise _IdleDecisionRejected("demand_census_unknown")
                        if not local.enter_context(exclusive(paths["config_lock"], blocking=False)):
                            raise _IdleDecisionRejected("demand_census_unknown")
                        if not local.enter_context(handoff_guard):
                            raise _IdleDecisionRejected("launch_handoff_pending")
                        if not admission.enter_context(idle_job_guard(self.root, blocking=False)):
                            raise _IdleDecisionRejected("demand_census_unknown")
                        if expected_handoff_revision is not None:
                            revision, pending = handoffs.snapshot() if has_handoff_fence else (None, False)
                            if (
                                pending
                                or revision != expected_handoff_revision
                                or self._last_handoff_revision
                                not in (
                                    None,
                                    revision,
                                )
                            ):
                                raise _IdleDecisionRejected("launch_handoff_pending")
                        owner, key, reason = self._observe_local_idle_locked(
                            owner, observation, policy_key, resolved, reservation_policy, decision_evidence
                        )
                        endpoint = None
                        if reason is None and owner is not None and owner["phase"] == "prepared":
                            if (
                                preparation is None
                                or self._idle_since != preparation.idle_since
                                or key != preparation.observation_key
                                or not preparation.intent_matches()
                                or _read_stop_reason(preparation.directory) is not None
                            ):
                                raise _IdleDecisionRejected("demand_census_unknown")
                            if not preparation.permit_sent and time.monotonic() >= preparation.ready_until:
                                raise _IdleDecisionRejected("demand_census_unknown")
                            if launch_allowed and preparation.ready:
                                endpoint = preparation.endpoint
                        if (guarded or endpoint is not None) and time.monotonic() >= deadline:
                            raise _IdleDecisionRejected("demand_census_unknown")
                    # The admission and local config/handoff fences still cover
                    # this decision. All Project proof locks have been released.
                    if endpoint is not None and preparation is not None:
                        try:
                            send_idle_launch_permit(endpoint, preparation.owner["instance_id"], deadline)
                            preparation.permit_sent = True
                        finally:
                            preparation.close_permit()
                        return self._payload(state="waiting", reason="preparing")
                self._last_handoff_revision = expected_handoff_revision
                self._last_decision_evidence = decision_evidence
                result = self._finish_idle_observation_locked(
                    owner, config, resolved, key, reason, launch_allowed, boot_id, now_wall
                )
            if isinstance(result, _PreparedIdleJob):
                try:
                    self._spawn_prepared(result)
                except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
                    result.spawn_failed = True
                    with idle_job_guard(self.root, blocking=False) as acquired:
                        if not acquired:
                            return self._blocked("cleanup_uncertain")
                        return self._preparation_failed_locked(exc)
                with idle_job_guard(self.root, blocking=False) as acquired:
                    if not acquired:
                        return self._defer_stop("unknown_demand", result.owner)
                    if read_idle_job_ownership(self.root) != result.owner:
                        return self._blocked("ownership_changed")
                    self._retry_at = None
                    return self._publish(state="waiting", reason="preparing")
            return result
        except _IdleDecisionRejected as exc:
            return self._reject_idle_decision(boot_id, config, str(exc), owner)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError):
            if guarded or preparation is not None:
                return self._reject_idle_decision(boot_id, config, "demand_census_unknown", owner)
            return self._blocked("supervision_failed")

    @property
    def has_local_instance(self) -> bool:
        return self._instance_id is not None

    def pending_wait_seconds(self, wait_seconds: float) -> float:
        """Revisit a gated candidate without sleeping for the full daemon interval."""
        preparation = self._preparation
        if preparation is not None and (preparation.endpoint is not None or preparation.permit_sent):
            if time.monotonic() >= preparation.ready_until:
                return wait_seconds
            return min(wait_seconds, SUPERVISOR_POLL_SECONDS)
        return wait_seconds

    def shutdown(self, timeout: float | None = None, *, timeout_seconds: float | None = None) -> bool:
        """Request ``agent_stop`` and wait a bounded period for ownership release."""
        if timeout is not None and timeout_seconds is not None:
            raise ValueError("provide timeout or timeout_seconds, not both")
        if timeout_seconds is not None:
            timeout = timeout_seconds
        if timeout is None:
            try:
                timeout = self._load_config().config.stop_timeout_seconds + DEFAULT_SHUTDOWN_EXTRA_SECONDS
            except (OSError, RuntimeError, TypeError, ValueError, KeyError):
                timeout = 15.0
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(float(timeout)):
            raise ValueError("timeout must be a finite nonnegative number")
        timeout = max(0.0, float(timeout))
        self._close_prepared_permit()
        self._clear_idle_interval("agent_stop")
        try:
            boot_id = current_boot_id()
        except (OSError, RuntimeError, TypeError, ValueError):
            return False
        deadline = self._now() + timeout
        while True:
            try:
                self._children = [child for child in self._children if child.poll() is None]
                with idle_job_guard(self.root):
                    owner, error = self._reconcile_locked(boot_id)
                    if error is not None:
                        return False
                    owner, error = self._stop_locked("agent_stop", owner)
                    if error is not None:
                        return False
                    if owner is None:
                        self._supervision_active = False
                        for child in self._children:
                            child.poll()
                        self._publish(state="disabled", reason="agent_stop")
                        return True
            except (OSError, RuntimeError, TypeError, ValueError, KeyError):
                return False
            now = self._now()
            if now >= deadline:
                return False
            self._sleep(min(SUPERVISOR_POLL_SECONDS, deadline - now))


__all__ = ["IDLE_JOB_STATUS_VERSION", "IdleJobSupervisor", "idle_job_status"]
