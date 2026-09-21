"""Machine-local GPU discovery, admission policy, and diagnostics.

The policy is deliberately independent from project state.  Hardware discovery is
performed before the GPU reservation lock is acquired; reservation admission only
re-reads this module's small local record while that lock is held.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .runtime.locks import exclusive
from .runtime.paths import machine_runtime_paths
from .runtime.records import utc_now
from .runtime.store import atomic_replace, iter_json, read_json
from .runtime.work_budget import diagnostic_increment

_GPU_LIST_RE = re.compile(r"[0-9]+(?:,[0-9]+)*\Z", re.ASCII)
_POLICY_SCHEMA_VERSION = 1
_MAX_WARNING_IDS = 256
_MAX_WARNING_TEXT = 1024


class GpuPolicyError(ValueError):
    """Raised when the persisted GPU policy cannot be trusted."""


@dataclass(frozen=True, slots=True)
class GpuDiscovery:
    """One bounded raw GPU inventory observation."""

    gpu_ids: tuple[int, ...] | None
    status: str
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.gpu_ids is not None:
            ids = _canonical_ids(self.gpu_ids, allow_empty=True)
            object.__setattr__(self, "gpu_ids", ids)
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("GPU discovery status must be a non-empty string.")
        if self.reason is not None and not isinstance(self.reason, str):
            raise ValueError("GPU discovery reason must be a string or null.")


@dataclass(frozen=True, slots=True)
class GpuReservationPolicy:
    """Pre-lock discovery and environment observations used for admission."""

    discovery: GpuDiscovery
    environment_gpu_ids: tuple[int, ...] | None
    environment_status: str

    def __post_init__(self) -> None:
        if self.environment_gpu_ids is not None:
            object.__setattr__(
                self,
                "environment_gpu_ids",
                _canonical_ids(self.environment_gpu_ids, allow_empty=True),
            )
        if self.environment_status == "absent":
            object.__setattr__(self, "environment_status", "unset")
        elif self.environment_status not in {"unset", "valid", "invalid"}:
            raise ValueError("GPU environment status must be unset, absent, valid, or invalid.")


@dataclass(frozen=True, slots=True)
class GpuPolicyView:
    """Effective GPU policy and machine-local reservation diagnostics."""

    mode: str
    source: str
    revision: int
    configured_gpu_ids: tuple[int, ...] | None
    discovered_gpu_ids: tuple[int, ...] | None
    visible_gpu_ids: tuple[int, ...] | None
    undiscovered_configured_gpu_ids: tuple[int, ...] | None
    reserved_gpu_ids: tuple[int, ...] = ()
    unreserved_gpu_ids: tuple[int, ...] | None = None
    draining_gpu_ids: tuple[int, ...] = ()
    discovery_status: str = "unavailable"
    visible_status: str = "unavailable"
    discovery_reason: str | None = None
    visible_reason: str | None = None
    warnings: tuple[dict[str, Any], ...] = ()
    agent_running: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"auto", "explicit"}:
            raise ValueError("GPU policy mode must be auto or explicit.")
        if not isinstance(self.revision, int) or isinstance(self.revision, bool) or self.revision < 0:
            raise ValueError("GPU policy revision must be a nonnegative integer.")
        for field_name in (
            "configured_gpu_ids",
            "discovered_gpu_ids",
            "visible_gpu_ids",
            "undiscovered_configured_gpu_ids",
            "unreserved_gpu_ids",
            "reserved_gpu_ids",
            "draining_gpu_ids",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _canonical_ids(value, allow_empty=True))
        object.__setattr__(self, "warnings", tuple(dict(item) for item in self.warnings))

    def with_reservations(self, reserved_gpu_ids: set[int] | tuple[int, ...] | list[int]) -> "GpuPolicyView":
        """Return this view with lock-consistent reservation-derived capacity."""
        reserved = _canonical_ids(reserved_gpu_ids, allow_empty=True)
        if self.visible_status == "pending":
            return replace(
                self,
                reserved_gpu_ids=reserved,
                unreserved_gpu_ids=None,
                draining_gpu_ids=(),
            )
        visible = set(self.visible_gpu_ids or ())
        return replace(
            self,
            reserved_gpu_ids=reserved,
            unreserved_gpu_ids=tuple(sorted(visible.difference(reserved))),
            draining_gpu_ids=tuple(sorted(set(reserved).difference(visible))),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the finite structured policy output contract."""
        return {
            "mode": self.mode,
            "source": self.source,
            "revision": self.revision,
            "configured_gpu_ids": _list_or_none(self.configured_gpu_ids),
            "discovered_gpu_ids": _list_or_none(self.discovered_gpu_ids),
            "visible_gpu_ids": _list_or_none(self.visible_gpu_ids),
            "undiscovered_configured_gpu_ids": _list_or_none(self.undiscovered_configured_gpu_ids),
            "reserved_gpu_ids": list(self.reserved_gpu_ids),
            "unreserved_gpu_ids": _list_or_none(self.unreserved_gpu_ids),
            "draining_gpu_ids": list(self.draining_gpu_ids),
            "discovery_status": self.discovery_status,
            "visible_status": self.visible_status,
            "discovery_reason": self.discovery_reason,
            "visible_reason": self.visible_reason,
            "warnings": [dict(item) for item in self.warnings],
            "agent_running": self.agent_running,
        }


@dataclass(frozen=True, slots=True)
class _PolicyRecord:
    revision: int
    mode: str
    configured_gpu_ids: tuple[int, ...] | None
    updated_at: str | None
    exists: bool


def _canonical_ids(value: Any, *, allow_empty: bool) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list, set, frozenset)):
        raise ValueError("GPU IDs must be a sequence of integers.")
    values = tuple(value)
    if any(type(item) is not int or item < 0 for item in values):
        raise ValueError("GPU IDs must be nonnegative integers.")
    if not allow_empty and not values:
        raise ValueError("GPU ID list must not be empty.")
    if len(set(values)) != len(values):
        raise ValueError("GPU IDs must not contain duplicates.")
    return tuple(sorted(values))


def parse_gpu_id_list(value: str) -> tuple[int, ...]:
    """Parse a strict, nonempty comma-separated ASCII GPU index list."""
    if not isinstance(value, str) or not value or _GPU_LIST_RE.fullmatch(value) is None:
        raise ValueError(
            "GPU IDs must be a nonempty comma-separated list of ASCII nonnegative integers "
            "without whitespace, signs, ranges, duplicates, or trailing separators."
        )
    try:
        values = tuple(int(item, 10) for item in value.split(","))
    except (TypeError, ValueError) as exc:
        raise ValueError("GPU IDs must be decimal integers.") from exc
    return _canonical_ids(values, allow_empty=False)


def parse_environment_gpu_ids(value: str | None = None) -> tuple[tuple[int, ...] | None, str]:
    """Parse the inherited environment, treating empty/unset as no override."""
    raw = os.environ.get("QEXP_VISIBLE_GPUS") if value is None else value
    if raw is None or raw == "":
        return None, "unset"
    try:
        return parse_gpu_id_list(raw), "valid"
    except ValueError:
        return None, "invalid"


def discover_gpu_inventory() -> GpuDiscovery:
    """Probe raw device indices without consulting the qexp allowlist."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=index", "--format=csv,noheader"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                ids: list[int] = []
                for line in result.stdout.splitlines():
                    item = line.strip()
                    if not item:
                        continue
                    if re.fullmatch(r"[0-9]+", item, re.ASCII) is None:
                        raise ValueError("nvidia_smi_invalid_output")
                    ids.append(int(item, 10))
                return GpuDiscovery(tuple(ids), "empty" if not ids else "available")
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    try:
        import torch

        count = torch.cuda.device_count()
        if type(count) is not int or count < 0:
            return GpuDiscovery(None, "unavailable", "torch_invalid_device_count")
        ids = tuple(range(count))
        return GpuDiscovery(ids, "empty" if not ids else "available")
    except Exception:
        return GpuDiscovery(None, "unavailable", "gpu_discovery_unavailable")


def _runtime_root(runtime: Any) -> Path:
    if isinstance(runtime, (str, Path)):
        return Path(runtime).expanduser().resolve()
    root = getattr(runtime, "root", None)
    if root is None:
        raise TypeError("runtime must be a MachineRuntime or Path")
    return Path(root).expanduser().resolve()


def _policy_path(root: Path) -> Path:
    return machine_runtime_paths(root)["gpu_policy"]


def _observation_path(root: Path) -> Path:
    return machine_runtime_paths(root)["gpu_policy_observation"]


def _warning_path(root: Path) -> Path:
    return machine_runtime_paths(root)["gpu_policy_warnings"]


def _validate_updated_at(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise GpuPolicyError("GPU policy updated_at must be a nonempty UTC timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GpuPolicyError("GPU policy updated_at is not an ISO timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise GpuPolicyError("GPU policy updated_at must include UTC timezone information.")
    return value


def _read_policy_record(root: Path) -> _PolicyRecord:
    path = _policy_path(root)
    if not path.exists():
        return _PolicyRecord(0, "auto", None, None, False)
    try:
        raw = read_json(path)
    except (OSError, TypeError, ValueError) as exc:
        raise GpuPolicyError(f"GPU policy record is unreadable: {type(exc).__name__}.") from exc
    if set(raw) != {"gpu_policy"} or not isinstance(raw.get("gpu_policy"), dict):
        raise GpuPolicyError("GPU policy record has an invalid top-level shape.")
    value = raw["gpu_policy"]
    required = {"schema_version", "revision", "mode", "configured_gpu_ids", "updated_at"}
    if set(value) != required:
        raise GpuPolicyError("GPU policy record has unknown or missing fields.")
    schema = value["schema_version"]
    revision = value["revision"]
    mode = value["mode"]
    configured = value["configured_gpu_ids"]
    if type(schema) is not int or schema != _POLICY_SCHEMA_VERSION:
        raise GpuPolicyError("GPU policy schema_version is unsupported.")
    if type(revision) is not int or revision < 0:
        raise GpuPolicyError("GPU policy revision must be a nonnegative integer.")
    if mode not in {"auto", "explicit"}:
        raise GpuPolicyError("GPU policy mode must be auto or explicit.")
    if configured is not None:
        if not isinstance(configured, list):
            raise GpuPolicyError("GPU policy configured_gpu_ids must be a list or null.")
        try:
            canonical = _canonical_ids(configured, allow_empty=True)
        except ValueError as exc:
            raise GpuPolicyError("GPU policy configured_gpu_ids is invalid.") from exc
        if configured != list(canonical):
            raise GpuPolicyError("GPU policy configured_gpu_ids must be sorted and unique.")
    else:
        canonical = None
    if (mode == "auto") != (canonical is None):
        raise GpuPolicyError("GPU policy mode and configured_gpu_ids do not match.")
    updated_at = _validate_updated_at(value["updated_at"])
    return _PolicyRecord(revision, mode, canonical, updated_at, True)


def _record_payload(record: _PolicyRecord) -> dict[str, Any]:
    if record.revision < 1 and not record.exists:
        raise ValueError("cannot write a nonexistent policy record")
    return {
        "gpu_policy": {
            "schema_version": _POLICY_SCHEMA_VERSION,
            "revision": record.revision,
            "mode": record.mode,
            "configured_gpu_ids": _list_or_none(record.configured_gpu_ids),
            "updated_at": record.updated_at or utc_now(),
        }
    }


def _list_or_none(value: tuple[int, ...] | list[int] | None) -> list[int] | None:
    return list(value) if value is not None else None


def _warning(reason: str, *, message: str, **fields: Any) -> dict[str, Any]:
    value = {"reason": reason, "message": message[:_MAX_WARNING_TEXT], **fields}
    return value


def _repair_commands(
    configured: tuple[int, ...], discovered: tuple[int, ...], visible: tuple[int, ...] = ()
) -> list[str]:
    commands = []
    suggested = (visible or discovered)[:_MAX_WARNING_IDS]
    if suggested:
        commands.append("qexp agent gpus set --visible " + ",".join(str(item) for item in suggested))
    commands.append("qexp agent gpus set --none")
    commands.append("qexp agent gpus reset")
    commands.append("qexp agent gpus show")
    return commands


def _build_warnings(
    *,
    mode: str,
    source: str,
    revision: int,
    configured: tuple[int, ...] | None,
    discovered: tuple[int, ...] | None,
    visible: tuple[int, ...] | None,
    discovery_status: str,
    visible_status: str,
    discovery_reason: str | None,
) -> tuple[dict[str, Any], ...]:
    if source == "unavailable" and discovery_reason == "gpu_policy_unavailable":
        return (
            _warning("gpu_policy_invalid", message="Persisted GPU policy is unavailable; GPU admission is blocked."),
        )
    if configured is None:
        return ()
    if not configured:
        return ()
    if discovery_status == "unavailable" or discovered is None:
        return (
            _warning(
                "gpu_discovery_unavailable",
                message="GPU validation is unavailable; new GPU admission is blocked until discovery recovers.",
                revision=revision,
                configured_gpu_ids=list(configured[:_MAX_WARNING_IDS]),
                discovered_gpu_ids=None,
                effective_visible_gpu_ids=None,
                undiscovered_configured_gpu_ids=None,
                repair_commands=["qexp agent gpus show"],
            ),
        )
    missing = tuple(item for item in configured if item not in discovered)
    if not missing:
        return ()
    effective = list(visible or ())
    message = (
        f"Configured GPU IDs {','.join(str(item) for item in missing)} were not discovered on this host. "
        f"Configured: {','.join(str(item) for item in configured)}. "
        f"Discovered: {','.join(str(item) for item in discovered) or 'none'}. "
        f"Effective visible GPUs: {','.join(str(item) for item in effective) or 'none'}. "
        "If these GPUs should exist, repair NVIDIA/container device exposure and run qexp agent gpus show."
    )
    return (
        _warning(
            "configured_gpu_ids_not_discovered",
            message=message,
            revision=revision,
            configured_gpu_ids=list(configured[:_MAX_WARNING_IDS]),
            discovered_gpu_ids=list(discovered[:_MAX_WARNING_IDS]),
            effective_visible_gpu_ids=effective[:_MAX_WARNING_IDS],
            visible_gpu_ids=effective[:_MAX_WARNING_IDS],
            missing_gpu_ids=list(missing[:_MAX_WARNING_IDS]),
            undiscovered_configured_gpu_ids=list(missing[:_MAX_WARNING_IDS]),
            repair_commands=_repair_commands(configured, discovered, visible),
        ),
    )


def _unavailable_view(
    discovery: GpuDiscovery,
    *,
    reason: str,
    revision: int = 0,
    agent_running: bool = False,
) -> GpuPolicyView:
    discovered = discovery.gpu_ids
    warning = _warning(
        "gpu_policy_invalid" if reason == "gpu_policy_unavailable" else "gpu_discovery_unavailable",
        message=(
            "Persisted GPU policy is unavailable; new GPU admission is blocked."
            if reason == "gpu_policy_unavailable"
            else "GPU validation is unavailable; new GPU admission is blocked."
        ),
    )
    return GpuPolicyView(
        "auto",
        "unavailable",
        revision,
        None,
        discovered,
        None,
        None,
        discovery_status=discovery.status,
        visible_status="unavailable",
        discovery_reason=reason if reason == "gpu_policy_unavailable" else discovery.reason,
        visible_reason=reason,
        warnings=(warning,),
        agent_running=agent_running,
    )


def _invalid_policy_view(root: Path, discovery: GpuDiscovery) -> GpuPolicyView:
    """Describe a malformed record without treating it as automatic mode."""
    mode = "auto"
    source = "persisted"
    revision = 0
    try:
        raw = read_json(_policy_path(root)).get("gpu_policy", {})
        if isinstance(raw, dict):
            if raw.get("mode") == "explicit":
                mode = "explicit"
            if type(raw.get("revision")) is int and raw["revision"] >= 0:
                revision = raw["revision"]
    except (OSError, TypeError, ValueError):
        source = "unavailable"
    warning = _warning(
        "gpu_policy_invalid",
        message="Persisted GPU policy is malformed or unreadable; GPU admission is blocked.",
        revision=revision,
    )
    return GpuPolicyView(
        mode,
        source,
        revision,
        None,
        discovery.gpu_ids,
        None,
        None,
        discovery_status=discovery.status,
        visible_status="unavailable",
        discovery_reason="gpu_policy_invalid",
        visible_reason="gpu_policy_invalid",
        warnings=(warning,),
    )


def resolve_gpu_policy(
    runtime_root: Path,
    *,
    discovery: GpuDiscovery,
    environment_gpu_ids: tuple[int, ...] | None,
    environment_status: str,
) -> GpuPolicyView:
    """Resolve persisted policy, inherited environment, and raw discovery."""
    root = Path(runtime_root).expanduser().resolve()
    try:
        record = _read_policy_record(root)
    except GpuPolicyError:
        return _invalid_policy_view(root, discovery)
    if environment_status == "absent":
        environment_status = "unset"
    if environment_status not in {"unset", "valid", "invalid"}:
        environment_status = "invalid"
    if record.mode == "explicit":
        source = "persisted"
        configured = record.configured_gpu_ids
    elif environment_status == "valid" and environment_gpu_ids is not None:
        source = "environment"
        configured = _canonical_ids(environment_gpu_ids, allow_empty=True)
    elif environment_status == "invalid":
        source = "environment"
        configured = None
        return GpuPolicyView(
            "explicit",
            source,
            record.revision,
            None,
            discovery.gpu_ids,
            None,
            None,
            discovery_status=discovery.status,
            visible_status="unavailable",
            discovery_reason=discovery.reason,
            visible_reason="invalid_environment",
            warnings=(
                _warning(
                    "gpu_environment_invalid",
                    message="QEXP_VISIBLE_GPUS is invalid; GPU admission is blocked.",
                ),
            ),
        )
    else:
        source = "discovery"
        configured = None
    discovered = discovery.gpu_ids
    if configured == ():
        visible = ()
        visible_status = "empty"
    elif discovered is None:
        visible = None
        visible_status = "unavailable"
    elif configured is None:
        visible = discovered
        visible_status = discovery.status
    else:
        visible = tuple(item for item in configured if item in discovered)
        visible_status = "empty" if not visible else "available"
    undiscovered = (
        None
        if configured is None or discovered is None
        else tuple(item for item in configured if item not in discovered)
    )
    warnings = _build_warnings(
        mode="explicit" if source == "persisted" else "auto",
        source=source,
        revision=record.revision,
        configured=configured,
        discovered=discovered,
        visible=visible,
        discovery_status=discovery.status,
        visible_status=visible_status,
        discovery_reason=discovery.reason,
    )
    return GpuPolicyView(
        "explicit" if source in {"persisted", "environment"} else "auto",
        source,
        record.revision,
        configured,
        discovered,
        visible,
        undiscovered,
        discovery_status=discovery.status,
        visible_status=visible_status,
        discovery_reason=discovery.reason,
        visible_reason=None if visible is not None else "discovery_unavailable",
        warnings=warnings,
    )


def _agent_identity(root: Path) -> tuple[bool, str | None, int | None]:
    try:
        status = read_json(machine_runtime_paths(root)["agent"] / "status.json").get("machine_agent", {})
    except (OSError, TypeError, ValueError):
        return False, None, None
    pid = status.get("pid")
    instance = status.get("instance_id")
    start_ticks = status.get("pid_start_time_ticks")
    try:
        pid_record = int((machine_runtime_paths(root)["pid"]).read_text(encoding="utf-8").strip())
        fields = (Path("/proc") / str(pid_record) / "stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        observed_start_ticks = None if fields[0] == "Z" else int(fields[19])
    except (FileNotFoundError, IndexError, OSError, ValueError):
        pid_record = None
        observed_start_ticks = None
    running = (
        status.get("state") == "active"
        and type(pid) is int
        and pid == pid_record
        and isinstance(instance, str)
        and bool(instance)
        and type(start_ticks) is int
        and start_ticks == observed_start_ticks
    )
    return running, instance if running else None, pid if running else None


def _read_observation(root: Path) -> dict[str, Any] | None:
    try:
        value = read_json(_observation_path(root)).get("gpu_policy_observation")
    except (OSError, TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _context_from_observation(value: Mapping[str, Any]) -> tuple[GpuDiscovery, tuple[int, ...] | None, str] | None:
    discovery = value.get("discovery")
    if not isinstance(discovery, Mapping):
        return None
    ids = discovery.get("gpu_ids")
    if ids is not None:
        try:
            ids = _canonical_ids(ids, allow_empty=True)
        except ValueError:
            return None
    status = discovery.get("status")
    reason = discovery.get("reason")
    if not isinstance(status, str) or (reason is not None and not isinstance(reason, str)):
        return None
    env_ids = value.get("environment_gpu_ids")
    if env_ids is not None:
        try:
            env_ids = _canonical_ids(env_ids, allow_empty=True)
        except ValueError:
            return None
    env_status = value.get("environment_status")
    if env_status == "absent":
        env_status = "unset"
    if env_status not in {"unset", "valid", "invalid"}:
        return None
    return GpuDiscovery(ids, status, reason), env_ids, env_status


def _reservation_ids_locked(root: Path) -> set[int]:
    result: set[int] = set()
    now = datetime.now(timezone.utc)
    for directory, is_provisional in (
        (root / "reservations" / "active", False),
        (root / "reservations" / "provisional", True),
    ):
        for path in iter_json(directory):
            try:
                value = read_json(path).get("reservation", {})
                if is_provisional:
                    expires_at = value.get("expires_at")
                    if expires_at:
                        expiry = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
                        if expiry <= now:
                            continue
                ids = value.get("gpu_ids", [])
                if isinstance(ids, list):
                    result.update(item for item in ids if type(item) is int and item >= 0)
            except (OSError, TypeError, ValueError, KeyError):
                continue
    return result


def _effective_context(root: Path) -> tuple[GpuDiscovery, tuple[int, ...] | None, str, bool]:
    running, instance_id, pid = _agent_identity(root)
    observation = _read_observation(root)
    if (
        running
        and observation is not None
        and observation.get("instance_id") == instance_id
        and observation.get("pid") == pid
    ):
        context = _context_from_observation(observation)
        if context is not None:
            discovery, env_ids, env_status = context
            return discovery, env_ids, env_status, True
    discovery = discover_gpu_inventory()
    env_ids, env_status = parse_environment_gpu_ids()
    return discovery, env_ids, env_status, running


def _pending_fallback_view(view: GpuPolicyView) -> GpuPolicyView:
    return replace(
        view,
        source="pending",
        configured_gpu_ids=None,
        undiscovered_configured_gpu_ids=None,
        visible_gpu_ids=None,
        unreserved_gpu_ids=None,
        draining_gpu_ids=(),
        visible_status="pending",
        visible_reason="agent_stopped_fallback_pending",
        agent_running=False,
    )


def show_gpu_policy(runtime: Any) -> dict[str, Any]:
    """Return effective policy and machine reservation diagnostics."""
    root = _runtime_root(runtime)
    discovery, env_ids, env_status, running = _effective_context(root)
    view = resolve_gpu_policy(
        root,
        discovery=discovery,
        environment_gpu_ids=env_ids,
        environment_status=env_status,
    )
    record_is_malformed = False
    try:
        record = _read_policy_record(root)
    except GpuPolicyError:
        record = None
        record_is_malformed = True
    if not running and not record_is_malformed and (record is None or record.mode == "auto"):
        view = _pending_fallback_view(view)
    else:
        view = replace(view, agent_running=running)
    paths = machine_runtime_paths(root)
    try:
        with exclusive(paths["reservation_lock"]):
            reserved = _reservation_ids_locked(root)
    except (OSError, RuntimeError, ValueError):
        reserved = set()
        view = replace(
            view,
            warnings=tuple(view.warnings)
            + (_warning("reservation_observation_unavailable", message="GPU reservations are unavailable."),),
        )
    return view.with_reservations(reserved).to_dict()


def _mutation_context(root: Path) -> tuple[GpuDiscovery, tuple[int, ...] | None, str, bool]:
    return _effective_context(root)


def _validate_expected_revision(expected_revision: int | None) -> int | None:
    if expected_revision is None:
        return None
    if type(expected_revision) is not int or expected_revision < 0:
        raise ValueError("expected_revision must be a nonnegative integer.")
    return expected_revision


def _mutate_gpu_policy(
    runtime: Any,
    *,
    mode: str,
    configured_gpu_ids: tuple[int, ...] | None,
    expected_revision: int | None,
) -> dict[str, Any]:
    root = _runtime_root(runtime)
    if configured_gpu_ids is not None:
        configured_gpu_ids = _canonical_ids(configured_gpu_ids, allow_empty=True)
    expected_revision = _validate_expected_revision(expected_revision)
    discovery, env_ids, env_status, running = _mutation_context(root)
    paths = machine_runtime_paths(root)
    paths["locks"].mkdir(parents=True, exist_ok=True)
    with exclusive(paths["reservation_lock"]):
        previous = _read_policy_record(root)
        if expected_revision is not None and previous.revision != expected_revision:
            raise ValueError(
                f"GPU policy revision mismatch: expected {expected_revision}, current {previous.revision}."
            )
        previous_view = resolve_gpu_policy(
            root,
            discovery=discovery,
            environment_gpu_ids=env_ids,
            environment_status=env_status,
        )
        current_record = _PolicyRecord(previous.revision + 1, mode, configured_gpu_ids, utc_now(), True)
        atomic_replace(_policy_path(root), _record_payload(current_record))
        current_view = resolve_gpu_policy(
            root,
            discovery=discovery,
            environment_gpu_ids=env_ids,
            environment_status=env_status,
        )
        reserved = _reservation_ids_locked(root)
        previous_view = previous_view.with_reservations(reserved)
        if not running and mode == "auto":
            current_view = _pending_fallback_view(current_view)
        current_view = replace(current_view, agent_running=running).with_reservations(reserved)
        result = current_view.to_dict()
        result.update(
            {
                "previous_revision": previous.revision,
                "current_revision": current_record.revision,
                "entered_draining_gpu_ids": sorted(
                    set(current_view.draining_gpu_ids).difference(previous_view.draining_gpu_ids)
                ),
            }
        )
        return result


def set_gpu_policy(
    runtime: Any,
    configured_gpu_ids: tuple[int, ...],
    *,
    expected_revision: int | None = None,
) -> dict[str, Any]:
    """Persist an explicit allowlist under the machine reservation lock."""
    return _mutate_gpu_policy(
        runtime,
        mode="explicit",
        configured_gpu_ids=_canonical_ids(configured_gpu_ids, allow_empty=True),
        expected_revision=expected_revision,
    )


def reset_gpu_policy(
    runtime: Any,
    *,
    expected_revision: int | None = None,
) -> dict[str, Any]:
    """Persist automatic fallback mode while preserving revision monotonicity."""
    return _mutate_gpu_policy(runtime, mode="auto", configured_gpu_ids=None, expected_revision=expected_revision)


def validate_gpu_reservation(
    runtime_root: Path,
    gpu_ids: list[int] | tuple[int, ...],
    policy: GpuReservationPolicy,
) -> None:
    """Validate requested GPU IDs against the current record without probing hardware."""
    root = Path(runtime_root).expanduser().resolve()
    view = resolve_gpu_policy(
        root,
        discovery=policy.discovery,
        environment_gpu_ids=policy.environment_gpu_ids,
        environment_status=policy.environment_status,
    )
    requested = _canonical_ids(gpu_ids, allow_empty=True)
    visible = set(view.visible_gpu_ids or ())
    if view.visible_gpu_ids is None:
        if view.visible_reason == "invalid_environment":
            diagnostic_increment("scheduler.gpu_policy_rejections.invalid_environment")
            raise ValueError("GPU environment is invalid; GPU admission is unavailable.")
        if any(warning.get("reason") == "gpu_policy_invalid" for warning in view.warnings):
            diagnostic_increment("scheduler.gpu_policy_rejections.policy_unavailable")
        else:
            diagnostic_increment("scheduler.gpu_policy_rejections.discovery_unavailable")
        raise ValueError("GPU admission is unavailable under the current policy or discovery observation.")
    missing = [item for item in requested if item not in visible]
    if missing:
        diagnostic_increment("scheduler.gpu_policy_rejections.not_visible")
        raise ValueError(
            "requested GPU is not visible under the current policy: " + ",".join(str(item) for item in missing)
        )


def gpu_policy_fingerprint(view: GpuPolicyView) -> str:
    """Return the bounded warning-dedup fingerprint for a policy observation."""
    payload = {
        "revision": view.revision,
        "source": view.source,
        "discovery_status": view.discovery_status,
        "discovered_gpu_ids": _list_or_none(view.discovered_gpu_ids),
        "warnings": [item.get("reason") for item in view.warnings],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def persist_gpu_policy_observation(
    runtime: Any,
    *,
    instance_id: str,
    pid: int | None,
    view: GpuPolicyView,
    policy: GpuReservationPolicy,
) -> tuple[dict[str, Any], bool]:
    """Persist the latest bounded machine observation and deduplicated warning."""
    root = _runtime_root(runtime)
    fingerprint = gpu_policy_fingerprint(view)
    observation = {
        "gpu_policy_observation": {
            "instance_id": instance_id,
            "pid": pid,
            "revision": view.revision,
            "fingerprint": fingerprint,
            "discovery": {
                "gpu_ids": _list_or_none(policy.discovery.gpu_ids),
                "status": policy.discovery.status,
                "reason": policy.discovery.reason,
            },
            "environment_gpu_ids": _list_or_none(policy.environment_gpu_ids),
            "environment_status": policy.environment_status,
            "view": view.to_dict(),
            "observed_at": utc_now(),
        }
    }
    paths = machine_runtime_paths(root)
    paths["agent"].mkdir(parents=True, exist_ok=True)
    prior = _read_observation(root)
    warning_changed = prior is None or prior.get("fingerprint") != fingerprint
    atomic_replace(_observation_path(root), observation)
    if warning_changed and view.warnings:
        atomic_replace(
            _warning_path(root),
            {
                "gpu_policy_warnings": {
                    "revision": view.revision,
                    "fingerprint": fingerprint,
                    "warnings": [dict(item) for item in view.warnings[:8]],
                    "updated_at": utc_now(),
                }
            },
        )
    return observation["gpu_policy_observation"], bool(warning_changed and view.warnings)


__all__ = [
    "GpuDiscovery",
    "GpuPolicyError",
    "GpuPolicyView",
    "GpuReservationPolicy",
    "discover_gpu_inventory",
    "gpu_policy_fingerprint",
    "parse_environment_gpu_ids",
    "parse_gpu_id_list",
    "persist_gpu_policy_observation",
    "reset_gpu_policy",
    "resolve_gpu_policy",
    "set_gpu_policy",
    "show_gpu_policy",
    "validate_gpu_reservation",
]
