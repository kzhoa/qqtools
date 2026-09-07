"""Shared explicit activation for schema-6 capability protocols."""
# QQTOOLS-COMPAT-0007: restricted legacy dependency normalization is retired in 1.3.17.
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from .config_types import RootConfig
from .layout import CPU_LANE_CAPABILITY, TASK_DEPENDENCIES_CAPABILITY
from .machine_runtime import MachineRuntime, ProjectBinding
from .runtime.locks import schema_lock
from .runtime.paths import shared_paths
from .runtime.records import TaskRecord, utc_now
from .runtime.store import atomic_replace, iter_json, read_json

_JOURNAL = "schema6-upgrade.json"
_CAPABILITIES = frozenset({CPU_LANE_CAPABILITY, TASK_DEPENDENCIES_CAPABILITY})


def _runtime_binding(
    cfg: RootConfig, machine_runtime_root: str | Path | None
) -> tuple[MachineRuntime, ProjectBinding]:
    """Resolve the caller's verified machine-local binding for this project."""
    runtime = MachineRuntime(machine_runtime_root)
    identity_path = shared_paths(cfg.shared_root)["project"] / "identity.json"
    identity = read_json(identity_path).get("project", {})
    project_id = identity.get("project_id")
    if not isinstance(project_id, str) or not project_id:
        raise ValueError("project identity is malformed.")
    _revision, bindings = runtime.load_registry()
    matches = [
        binding
        for binding in bindings
        if binding.project_id == project_id and binding.shared_root == cfg.shared_root
    ]
    if len(matches) != 1:
        raise ValueError("no unique machine runtime binding matches this project.")
    binding = matches[0]
    record = read_json(
        shared_paths(cfg.shared_root)["machines"] / binding.machine_name / "machine.json"
    ).get("machine", {})
    if (
        not isinstance(record, dict)
        or record.get("machine_name") != binding.machine_name
        or record.get("project_id") != binding.project_id
        or record.get("shared_root") != str(cfg.shared_root)
        or record.get("agent_runtime") != "machine"
    ):
        raise ValueError("local machine runtime binding does not match Project truth.")
    return runtime, binding


def _validate_attestations(
    cfg: RootConfig,
    journal: dict[str, Any],
    *,
    runtime: MachineRuntime,
    binding: ProjectBinding,
) -> None:
    """Ensure attestations retain their binding and the resumer's runtime identity."""
    identity_path = shared_paths(cfg.shared_root)["project"] / "identity.json"
    identity = read_json(identity_path).get("project", {})
    project_id = identity.get("project_id")
    if not isinstance(project_id, str) or not project_id:
        raise ValueError("project identity is malformed.")
    attestations = journal.get("attestations")
    if not isinstance(attestations, dict):
        raise ValueError("schema-6 upgrade attestations are malformed.")
    for machine_name, attestation in attestations.items():
        if machine_name not in journal["participants"] or not isinstance(attestation, dict):
            raise ValueError("schema-6 upgrade attestation is not a declared participant.")
        if (
            attestation.get("machine_name") != machine_name
            or attestation.get("binding_machine_name") != machine_name
            or attestation.get("project_id") != project_id
            or attestation.get("shared_root") != str(cfg.shared_root)
        ):
            raise ValueError(
                f"schema-6 upgrade attestation does not match machine binding: {machine_name}."
            )
        if (
            machine_name == binding.machine_name
            and attestation.get("runtime_root") != str(runtime.root)
        ):
            raise ValueError(
                "schema-6 upgrade attestation does not match the local machine runtime."
            )


def _blockers(
    cfg: RootConfig, *, machine_runtime_root: str | Path | None = None
) -> list[str]:
    """Return active shared and local state that prevents a schema-6 activation."""
    blockers: list[str] = []
    for path in iter_json(shared_paths(cfg.shared_root)["tasks"]):
        task = read_json(path).get("task", {})
        phase = task.get("state", {}).get("projection")
        if task.get("claim_control", {}).get("active_claim") or phase in {"running", "blocked"}:
            blockers.append(f"task:{path.stem}:{phase}")
    for path in iter_json(shared_paths(cfg.shared_root)["submissions"]):
        if read_json(path).get("submission", {}).get("state") not in {"committed", "aborted"}:
            blockers.append(f"operation:submission:{path.stem}")
    for directory in ("availability_active", "group_control_active", "cleanup_active"):
        if iter_json(shared_paths(cfg.shared_root)[directory]):
            blockers.append(f"operation:{directory.removesuffix('_active')}")
    pending_claims = shared_paths(cfg.shared_root)["claim_pending"]
    if pending_claims.exists() and any(path.is_file() for path in pending_claims.rglob("*.json")):
        blockers.append("operation:claim_pending")
    try:
        runtime, binding = _runtime_binding(cfg, machine_runtime_root)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        blockers.append(f"runtime:binding_unresolved:{exc}")
        return blockers
    for directory in ("active", "provisional", "cpu_active", "cpu_provisional"):
        for path in iter_json(runtime.paths[directory]):
            if read_json(path).get("reservation", {}).get("project_id") == binding.project_id:
                blockers.append(f"runtime:{directory}")
                break
    project_paths = runtime.project_paths(binding.project_id)
    for directory in ("processes", "registrations", "launch_intents", "termination_decisions"):
        if iter_json(project_paths[directory]):
            blockers.append(f"runtime:{directory}")
    return blockers


def _path(cfg: RootConfig) -> Path:
    return shared_paths(cfg.shared_root)["schema"] / _JOURNAL


def _requested(capabilities: list[str] | None) -> list[str]:
    values = sorted(set(_CAPABILITIES if capabilities is None else capabilities))
    unknown = sorted(set(values) - _CAPABILITIES)
    if unknown:
        raise ValueError(f"unsupported schema-6 capabilities: {', '.join(unknown)}")
    if not values:
        raise ValueError("at least one schema-6 capability is required.")
    if set(values) != _CAPABILITIES:
        raise ValueError(
            "schema-6 activation requires cpu-lane-v1 and task-dependencies-v1 together."
        )
    return values


def schema6_upgrade_status(cfg: RootConfig) -> dict[str, Any]:
    if _path(cfg).exists():
        return read_json(_path(cfg))["schema6_upgrade"]
    return {"phase": "legacy", "activation_id": None, "capabilities": []}


def check_schema6_upgrade(
    cfg: RootConfig, *, capabilities: list[str] | None = None,
    machine_runtime_root: str | Path | None = None,
) -> dict[str, Any]:
    status = schema6_upgrade_status(cfg)
    if status["phase"] != "legacy":
        return {"shared_root": str(cfg.shared_root), **status}
    return {
        "shared_root": str(cfg.shared_root), "capabilities": _requested(capabilities),
        "phase": "legacy", "blockers": _blockers(cfg, machine_runtime_root=machine_runtime_root),
    }


def start_schema6_upgrade(
    cfg: RootConfig, *, capabilities: list[str] | None = None,
    machine_runtime_root: str | Path | None = None,
) -> dict[str, Any]:
    with schema_lock(cfg.shared_root):
        requested = _requested(capabilities)
        if _path(cfg).exists():
            value = read_json(_path(cfg))["schema6_upgrade"]
            if value.get("capabilities") != requested:
                raise ValueError("schema-6 activation capabilities do not match the existing session.")
            return value
        blockers = _blockers(cfg, machine_runtime_root=machine_runtime_root)
        if blockers:
            raise RuntimeError("schema-6 upgrade requires a drained root: " + ", ".join(blockers))
        value = {
            "activation_id": uuid.uuid4().hex, "phase": "awaiting_attestations",
            "capabilities": requested, "participants": sorted(
                path.name for path in shared_paths(cfg.shared_root)["machines"].iterdir() if path.is_dir()
            ), "attestations": {}, "normalized_tasks": 0, "created_at": utc_now(),
        }
        atomic_replace(_path(cfg), {"schema6_upgrade": value})
        return value


def attest_schema6_upgrade(
    cfg: RootConfig, *, activation_id: str, machine_name: str,
    machine_runtime_root: str | Path | None = None,
) -> dict[str, Any]:
    with schema_lock(cfg.shared_root):
        value = read_json(_path(cfg))["schema6_upgrade"]
        if value["activation_id"] != activation_id or value["phase"] != "awaiting_attestations":
            raise ValueError("activation ID does not identify an attestable schema-6 upgrade.")
        runtime, binding = _runtime_binding(cfg, machine_runtime_root)
        if binding.machine_name != machine_name or machine_name not in value["participants"]:
            raise ValueError("attestation machine is not a declared local participant.")
        blockers = _blockers(cfg, machine_runtime_root=machine_runtime_root)
        if blockers:
            raise RuntimeError("schema-6 upgrade attestation found blockers: " + ", ".join(blockers))
        value["attestations"][machine_name] = {
            "machine_name": machine_name,
            "binding_machine_name": binding.machine_name,
            "project_id": binding.project_id,
            "shared_root": str(binding.shared_root),
            "runtime_root": str(runtime.root),
            "attested_at": utc_now(),
        }
        atomic_replace(_path(cfg), {"schema6_upgrade": value})
        return value


def resume_schema6_upgrade(
    cfg: RootConfig, *, activation_id: str,
    machine_runtime_root: str | Path | None = None,
) -> dict[str, Any]:
    with schema_lock(cfg.shared_root):
        value = read_json(_path(cfg))["schema6_upgrade"]
        if value["activation_id"] != activation_id:
            raise ValueError("activation ID does not match the schema-6 upgrade.")
        if value["phase"] == "completed":
            return value
        if value["phase"] == "normalizing":
            value.update({
                "phase": "awaiting_attestations",
                "attestations": {},
                "recovery_required_at": utc_now(),
            })
            atomic_replace(_path(cfg), {"schema6_upgrade": value})
            raise RuntimeError(
                "schema-6 upgrade was interrupted during normalization; collect fresh "
                "machine attestations before resuming."
            )
        if value["phase"] != "awaiting_attestations":
            raise RuntimeError(f"schema-6 upgrade has unsupported phase {value['phase']!r}.")
        missing = sorted(set(value["participants"]) - set(value["attestations"]))
        if missing:
            raise RuntimeError("schema-6 upgrade is missing attestations: " + ", ".join(missing))
        runtime, binding = _runtime_binding(cfg, machine_runtime_root)
        _validate_attestations(cfg, value, runtime=runtime, binding=binding)
        if _blockers(cfg, machine_runtime_root=machine_runtime_root):
            raise RuntimeError("schema-6 upgrade requires a drained root.")
        value["phase"] = "normalizing"
        atomic_replace(_path(cfg), {"schema6_upgrade": value})
        schema_path = shared_paths(cfg.shared_root)["schema"] / "version.json"
        schema = read_json(schema_path)
        current = schema["schema"].get("required_capabilities", [])
        schema["schema"]["required_capabilities"] = sorted(set(current) | set(value["capabilities"]))
        atomic_replace(schema_path, schema)
        normalized = 0
        for path in iter_json(shared_paths(cfg.shared_root)["tasks"]):
            raw = read_json(path)
            task = TaskRecord.from_dict(raw)
            changed = False
            if CPU_LANE_CAPABILITY in value["capabilities"] and task.spec.lane is None:
                task.spec.lane = "gpu"
                changed = True
            if TASK_DEPENDENCIES_CAPABILITY in value["capabilities"] and "depends_on_task_ids" not in raw["task"]:
                task.depends_on_task_ids = []
                changed = True
            if changed:
                task.meta["revision"] += 1
                task.meta["updated_at"] = utc_now()
                atomic_replace(path, task.to_dict())
                normalized += 1
        value.update({"phase": "completed", "completed_at": utc_now(), "normalized_tasks": normalized})
        atomic_replace(_path(cfg), {"schema6_upgrade": value})
        return value
