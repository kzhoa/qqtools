"""Bounded read-only Project and machine status projections."""

from __future__ import annotations

import json
import shlex
import stat
from pathlib import Path
from typing import Any

from ..agent.bindings import ProjectBinding
from ..agent.context import MachineRuntime
from ..agent.inventory import ProjectInventoryEntry
from ..config_types import RootConfig
from ..layout import machine_path, project_id
from ..runtime.observation.projection import _validate_state, observation_path
from ..runtime.paths import shared_paths
from ..runtime.records import validate_identifier

STATUS_SCHEMA_VERSION = 1
MAX_READ_ATTEMPTS = 32
MAX_RECORD_BYTES = 256 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024
MACHINE_DETAIL_MAX_READ_ATTEMPTS = 8


class _BoundedReadFailure(ValueError):
    """An exact-path read failed without allowing an unbounded fallback."""

    def __init__(self, reason: str, path: Path) -> None:
        super().__init__(reason)
        self.reason = reason
        self.path = path


class _ReadBudget:
    """Track exact-file reads and enforce the status payload limits."""

    def __init__(
        self,
        *,
        max_read_attempts: int = MAX_READ_ATTEMPTS,
        max_record_bytes: int = MAX_RECORD_BYTES,
        max_total_bytes: int = MAX_TOTAL_BYTES,
    ) -> None:
        self.max_read_attempts = max_read_attempts
        self.max_record_bytes = max_record_bytes
        self.max_total_bytes = max_total_bytes
        self.read_attempts = 0
        self.bytes_read = 0

    def read_json(self, path: Path) -> dict[str, Any] | None:
        """Read one regular JSON file, returning ``None`` only when absent."""
        if self.read_attempts >= self.max_read_attempts:
            raise _BoundedReadFailure("read_budget_exhausted", path)
        self.read_attempts += 1
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise _BoundedReadFailure(f"lstat_failed:{type(exc).__name__}", path) from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise _BoundedReadFailure("not_a_regular_file", path)
        if metadata.st_size > self.max_record_bytes:
            raise _BoundedReadFailure("record_oversized", path)
        if self.bytes_read + metadata.st_size > self.max_total_bytes:
            raise _BoundedReadFailure("total_read_budget_exhausted", path)
        try:
            with path.open("rb") as handle:
                encoded = handle.read(self.max_record_bytes + 1)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise _BoundedReadFailure(f"read_failed:{type(exc).__name__}", path) from exc
        self.bytes_read += len(encoded)
        if len(encoded) > self.max_record_bytes:
            raise _BoundedReadFailure("record_oversized", path)
        try:
            value = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _BoundedReadFailure("malformed_json", path) from exc
        if not isinstance(value, dict):
            raise _BoundedReadFailure("json_object_required", path)
        return value

    def as_dict(self) -> dict[str, int]:
        return {
            "read_attempts": self.read_attempts,
            "max_read_attempts": self.max_read_attempts,
            "bytes_read": self.bytes_read,
            "max_record_bytes": self.max_record_bytes,
            "max_total_bytes": self.max_total_bytes,
        }


def _warning(component: str, failure: _BoundedReadFailure) -> str:
    return f"{component} unavailable ({failure.reason})."


def _optional_read(
    reader: _ReadBudget,
    path: Path,
    component: str,
    warnings: list[str],
) -> dict[str, Any] | None:
    try:
        value = reader.read_json(path)
    except _BoundedReadFailure as exc:
        warnings.append(_warning(component, exc))
        return None
    if value is None:
        return None
    return value


def _project_identity(cfg: RootConfig, reader: _ReadBudget) -> str:
    identity_path = shared_paths(cfg.shared_root)["project"] / "identity.json"
    try:
        identity = reader.read_json(identity_path)
    except _BoundedReadFailure as exc:
        raise ValueError(f"Project identity is unavailable: {exc.reason}.") from exc
    if identity is None:
        raise ValueError(f"Project identity is missing: {identity_path}.")
    project = identity.get("project")
    expected = project_id(cfg.shared_root)
    if (
        not isinstance(project, dict)
        or project.get("project_id") != expected
        or not isinstance(project.get("shared_root"), str)
        or Path(project["shared_root"]).expanduser().resolve() != cfg.shared_root
    ):
        raise ValueError("Project identity is malformed or does not match the selected Project.")
    return expected


def _decode_registry(value: dict[str, Any]) -> list[ProjectBinding]:
    registry = value.get("registry")
    if not isinstance(registry, dict) or registry.get("version") != 1:
        raise ValueError("machine registry is malformed or unsupported.")
    revision = registry.get("revision")
    bindings = registry.get("bindings")
    if type(revision) is not int or revision < 0 or not isinstance(bindings, list):
        raise ValueError("machine registry is malformed.")
    try:
        parsed = [ProjectBinding.from_dict(item) for item in bindings]
    except (TypeError, ValueError) as exc:
        raise ValueError("machine registry contains a malformed project binding.") from exc
    return parsed


def _decode_inventory(value: dict[str, Any]) -> list[ProjectInventoryEntry]:
    inventory = value.get("inventory")
    if not isinstance(inventory, dict) or inventory.get("version") != 1:
        raise ValueError("machine Project inventory is malformed or unsupported.")
    revision = inventory.get("revision")
    entries = inventory.get("entries")
    if type(revision) is not int or revision < 0 or not isinstance(entries, list):
        raise ValueError("machine Project inventory is malformed.")
    try:
        parsed = [ProjectInventoryEntry.from_dict(item) for item in entries]
    except (TypeError, ValueError) as exc:
        raise ValueError("machine Project inventory contains a malformed entry.") from exc
    return parsed


def _matching_participation(
    cfg: RootConfig,
    selected_project_id: str,
    registry: list[ProjectBinding] | None,
    inventory: list[ProjectInventoryEntry] | None,
    *,
    registry_failed: bool,
    inventory_failed: bool,
) -> tuple[str, str | None]:
    registered = any(
        binding.project_id == selected_project_id and binding.shared_root == cfg.shared_root
        for binding in registry or []
    )
    if registered:
        return "registered", None
    inventoried = any(
        entry.project_id == selected_project_id and entry.shared_root == cfg.shared_root for entry in inventory or []
    )
    if inventoried:
        return "inventory_only", "inventory_entry_without_binding"
    if registry_failed or inventory_failed:
        return "unavailable", "local_registry_or_inventory_unavailable"
    return "unregistered", "no_local_binding_or_inventory"


def _agent_observation(
    value: dict[str, Any] | None,
    *,
    failed: bool,
    missing: bool,
) -> dict[str, object]:
    if value is None:
        reason = "record_missing" if missing and not failed else "record_unreadable"
        return {"scope": "machine", "state": "unavailable", "observed_at": None, "reason": reason}
    record = value.get("machine_agent")
    if not isinstance(record, dict):
        return {"scope": "machine", "state": "unavailable", "observed_at": None, "reason": "record_malformed"}
    state = record.get("state")
    if not isinstance(state, str) or not state:
        return {"scope": "machine", "state": "unavailable", "observed_at": None, "reason": "state_unavailable"}
    observed_at = next(
        (record.get(key) for key in ("heartbeat_at", "observed_at", "timestamp") if isinstance(record.get(key), str)),
        None,
    )
    reason = record.get("stop_reason") if state == "stopped" else None
    return {"scope": "machine", "state": state, "observed_at": observed_at, "reason": reason}


def _task_observation(
    value: dict[str, Any] | None, *, failed: bool, missing: bool, cfg: RootConfig
) -> dict[str, object]:
    if value is None:
        return {
            "state": "unavailable" if failed else "absent",
            "reason": "record_unreadable" if failed else ("projection_missing" if missing else "state_unavailable"),
        }
    try:
        validated = _validate_state(cfg, value)
    except (TypeError, ValueError) as exc:
        return {"state": "unavailable", "reason": f"invalid_state:{type(exc).__name__}"}
    reason = "projection_dirty" if validated.get("dirty") else None
    return {"state": validated["state"], "reason": reason}


def _follow_up_actions(cfg: RootConfig, participation: str) -> list[str]:
    project = str(cfg.project_root)
    actions = [
        shlex.join(["qexp", "task", "list", "--project", project]),
        shlex.join(["qexp", "machine", "list", "--project", project]),
        shlex.join(["qexp", "agent", "status"]),
    ]
    if participation in {"inventory_only", "unregistered"}:
        actions.append(shlex.join(["qexp", "project", "register", project]))
    elif participation == "unavailable":
        actions.append(shlex.join(["qexp", "init", "--machine", cfg.machine_name]))
    return actions


def project_status(
    cfg: RootConfig,
    *,
    selection_source: str,
    machine_runtime: MachineRuntime,
) -> dict[str, object]:
    """Return a bounded Project overview without Task or machine enumeration."""
    reader = _ReadBudget()
    selected_project_id = _project_identity(cfg, reader)
    warnings: list[str] = []

    registry_path = machine_runtime.paths["registry"]
    inventory_path = machine_runtime.paths["inventory"]
    agent_path = machine_runtime.paths["agent"] / "status.json"
    observation_state_path = observation_path(cfg) / "state.json"

    warning_count = len(warnings)
    registry_raw = _optional_read(reader, registry_path, "local registry", warnings)
    registry_failed = len(warnings) != warning_count
    warning_count = len(warnings)
    inventory_raw = _optional_read(reader, inventory_path, "local inventory", warnings)
    inventory_failed = len(warnings) != warning_count
    agent_missing = False
    try:
        agent_raw = reader.read_json(agent_path)
        agent_missing = agent_raw is None
    except _BoundedReadFailure as exc:
        warnings.append(_warning("local agent status", exc))
        agent_raw = None
        agent_failed = True
    else:
        agent_failed = False
        if agent_raw is None:
            warnings.append("local agent status unavailable (record_missing).")

    observation_missing = False
    try:
        observation_raw = reader.read_json(observation_state_path)
        observation_missing = observation_raw is None
    except _BoundedReadFailure as exc:
        warnings.append(_warning("Task observation", exc))
        observation_raw = None
        observation_failed = True
    else:
        observation_failed = False

    registry: list[ProjectBinding] | None = None
    inventory: list[ProjectInventoryEntry] | None = None
    if registry_raw is not None:
        try:
            registry = _decode_registry(registry_raw)
        except (TypeError, ValueError) as exc:
            registry_failed = True
            warnings.append(f"local registry unavailable (malformed_record:{type(exc).__name__}).")
    if inventory_raw is not None:
        try:
            inventory = _decode_inventory(inventory_raw)
        except (TypeError, ValueError) as exc:
            inventory_failed = True
            warnings.append(f"local inventory unavailable (malformed_record:{type(exc).__name__}).")

    participation_state, participation_reason = _matching_participation(
        cfg,
        selected_project_id,
        registry,
        inventory,
        registry_failed=registry_failed,
        inventory_failed=inventory_failed,
    )
    local_agent = _agent_observation(agent_raw, failed=agent_failed, missing=agent_missing)
    task_observation = _task_observation(
        observation_raw,
        failed=observation_failed,
        missing=observation_missing,
        cfg=cfg,
    )
    if task_observation["state"] == "unavailable" and not any("Task observation" in item for item in warnings):
        warnings.append("Task observation unavailable (record_malformed).")
    if local_agent["state"] == "unavailable" and not any("local agent status" in item for item in warnings):
        warnings.append("local agent status unavailable (record_malformed).")

    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "status": "partial" if warnings else "complete",
        "project": {
            "path": str(cfg.project_root),
            "project_id": selected_project_id,
            "selection_source": selection_source,
        },
        "local_participation": {"state": participation_state, "reason": participation_reason},
        "local_agent": local_agent,
        "task_observation": task_observation,
        "totals": {"tasks": None, "machines": None, "reason": "not_available"},
        "next_actions": _follow_up_actions(cfg, participation_state),
        "budget": reader.as_dict(),
        "warnings": warnings,
    }


def _machine_state_path(cfg: RootConfig, name: str, state_name: str) -> Path:
    return shared_paths(cfg.shared_root)["machines"] / name / "state" / state_name


def _machine_optional_record(
    reader: _ReadBudget,
    path: Path,
    key: str,
    warnings: list[str],
) -> dict[str, Any] | None:
    value = _optional_read(reader, path, f"machine {key}", warnings)
    if value is None:
        if not any(f"machine {key} unavailable" in item for item in warnings):
            warnings.append(f"machine {key} unavailable (record_missing).")
        return None
    record = value.get(key)
    if not isinstance(record, dict):
        warnings.append(f"machine {key} unavailable (record_malformed).")
        return None
    return record


def machine_detail(cfg: RootConfig, name: str) -> dict[str, object]:
    """Return one declared machine and exact local observation snapshots."""
    validate_identifier(name, "machine_name")
    reader = _ReadBudget(max_read_attempts=MACHINE_DETAIL_MAX_READ_ATTEMPTS)
    declaration_path = machine_path(cfg.shared_root, name)
    try:
        declaration_raw = reader.read_json(declaration_path)
    except _BoundedReadFailure as exc:
        raise ValueError(f"machine declaration is malformed: {exc.reason}.") from exc
    if declaration_raw is None:
        raise FileNotFoundError(declaration_path)
    declaration = declaration_raw.get("machine")
    expected_project_id = project_id(cfg.shared_root)
    if (
        not isinstance(declaration, dict)
        or declaration.get("machine_name") != name
        or declaration.get("project_id") != expected_project_id
        or declaration.get("shared_root") != str(cfg.shared_root)
    ):
        raise ValueError("machine declaration is malformed or belongs to another Project.")

    warnings: list[str] = []
    agent = _machine_optional_record(reader, _machine_state_path(cfg, name, "agent.json"), "agent", warnings)
    gpu = _machine_optional_record(reader, _machine_state_path(cfg, name, "gpu.json"), "gpu", warnings)
    summary = _machine_optional_record(reader, _machine_state_path(cfg, name, "summary.json"), "summary", warnings)
    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "machine_name": name,
        "project": {"path": str(cfg.project_root), "project_id": expected_project_id},
        "declaration": declaration,
        "agent": agent,
        "gpu": gpu,
        "summary": summary,
        "complete": not warnings,
        "warnings": warnings,
        "budget": reader.as_dict(),
    }


__all__ = [
    "MACHINE_DETAIL_MAX_READ_ATTEMPTS",
    "MAX_READ_ATTEMPTS",
    "MAX_RECORD_BYTES",
    "MAX_TOTAL_BYTES",
    "STATUS_SCHEMA_VERSION",
    "machine_detail",
    "project_status",
]
