"""Explicit, resumable activation of CPU lanes on drained schema-6 roots."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from .config_types import RootConfig
from .layout import CPU_LANE_CAPABILITY, is_cpu_lane_root
from .machine_runtime import MachineRuntime
from .runtime.locks import schema_lock
from .runtime.paths import local_paths, shared_paths
from .runtime.records import TaskRecord, utc_now
from .runtime.store import atomic_replace, iter_json, read_json

_JOURNAL = "cpu-lane-upgrade.json"


def _journal_path(cfg: RootConfig) -> Path:
    return shared_paths(cfg.shared_root)["schema"] / _JOURNAL


def _runtime_binding(
    cfg: RootConfig, machine_runtime_root: str | Path | None
) -> tuple[MachineRuntime, Any]:
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


def _validate_attestations(cfg: RootConfig, journal: dict[str, Any]) -> None:
    """Ensure recorded attestations still identify their original local binding."""
    identity_path = shared_paths(cfg.shared_root)["project"] / "identity.json"
    identity = read_json(identity_path).get("project", {})
    project_id = identity.get("project_id")
    if not isinstance(project_id, str) or not project_id:
        raise ValueError("project identity is malformed.")
    shared_root = str(cfg.shared_root)
    attestations = journal.get("attestations")
    if not isinstance(attestations, dict):
        raise ValueError("CPU lane upgrade attestations are malformed.")
    for machine_name, attestation in attestations.items():
        if machine_name not in journal["participants"] or not isinstance(attestation, dict):
            raise ValueError("CPU lane upgrade attestation is not a declared participant.")
        if (
            attestation.get("machine_name") != machine_name
            or attestation.get("binding_machine_name") != machine_name
            or attestation.get("project_id") != project_id
            or attestation.get("shared_root") != shared_root
        ):
            raise ValueError(
                f"CPU lane upgrade attestation does not match machine binding: {machine_name}."
            )


def _blockers(
    cfg: RootConfig, *, machine_runtime_root: str | Path | None = None
) -> list[str]:
    blockers: list[str] = []
    for path in iter_json(shared_paths(cfg.shared_root)["tasks"]):
        task = read_json(path).get("task", {})
        phase = task.get("state", {}).get("projection")
        if task.get("claim_control", {}).get("active_claim") or phase in {"running", "blocked"}:
            blockers.append(f"task:{path.stem}:{phase}")
    for path in iter_json(shared_paths(cfg.shared_root)["submissions"]):
        if read_json(path).get("submission", {}).get("state") not in {"committed", "aborted"}:
            blockers.append(f"operation:submission:{path.stem}")
    for directory in ("availability", "group_control", "cleanup", "claim_pending"):
        path = shared_paths(cfg.shared_root)[directory]
        if iter_json(path):
            blockers.append(f"operation:{directory}")
    try:
        runtime, binding = _runtime_binding(cfg, machine_runtime_root)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        blockers.append(f"runtime:binding_unresolved:{exc}")
        return blockers
    for directory in ("active", "provisional", "cpu_active", "cpu_provisional"):
        for path in iter_json(runtime.paths[directory]):
            reservation = read_json(path).get("reservation", {})
            if reservation.get("project_id") == binding.project_id:
                blockers.append(f"runtime:{directory}")
                break
    project_paths = runtime.project_paths(binding.project_id)
    for directory in ("processes", "registrations", "launch_intents", "termination_decisions"):
        if iter_json(project_paths[directory]):
            blockers.append(f"runtime:{directory}")
    return blockers


def check_cpu_lane_upgrade(
    cfg: RootConfig, *, machine_runtime_root: str | Path | None = None
) -> dict[str, Any]:
    """Return a read-only legacy-root activation preflight."""
    if _journal_path(cfg).exists():
        journal = read_json(_journal_path(cfg))["cpu_lane_upgrade"]
        protocol_state = "canonical" if journal["phase"] == "completed" else "preparing"
        return {"shared_root": str(cfg.shared_root), "protocol_state": protocol_state, **journal}
    canonical = is_cpu_lane_root(cfg)
    return {
        "shared_root": str(cfg.shared_root),
        "protocol_state": "canonical" if canonical else "legacy",
        "blockers": [] if canonical else _blockers(cfg, machine_runtime_root=machine_runtime_root),
        "journal": _journal_path(cfg).exists(),
    }


def start_cpu_lane_upgrade(
    cfg: RootConfig, *, machine_runtime_root: str | Path | None = None
) -> dict[str, Any]:
    """Create the sole activation session after the shared root is drained."""
    with schema_lock(cfg.shared_root):
        journal_path = _journal_path(cfg)
        if journal_path.exists():
            return read_json(journal_path)["cpu_lane_upgrade"]
        if is_cpu_lane_root(cfg):
            return {"protocol_state": "canonical", "phase": "completed", "activation_id": None}
        blockers = _blockers(cfg, machine_runtime_root=machine_runtime_root)
        if blockers:
            raise RuntimeError("CPU lane upgrade requires a drained root: " + ", ".join(blockers))
        value = {
            "activation_id": uuid.uuid4().hex,
            "phase": "awaiting_attestations",
            "created_at": utc_now(),
            "attestations": {},
            "participants": sorted(
                path.name
                for path in shared_paths(cfg.shared_root)["machines"].iterdir()
                if path.is_dir()
            ),
            "normalized_tasks": 0,
        }
        atomic_replace(journal_path, {"cpu_lane_upgrade": value})
        return value


def attest_cpu_lane_upgrade(
    cfg: RootConfig,
    *,
    activation_id: str,
    machine_name: str,
    machine_runtime_root: str | Path | None = None,
) -> dict[str, Any]:
    """Record one operator-confirmed participant after local shared-state checks."""
    with schema_lock(cfg.shared_root):
        journal = read_json(_journal_path(cfg))["cpu_lane_upgrade"]
        if journal["activation_id"] != activation_id:
            raise ValueError("activation ID does not match the pending CPU lane upgrade.")
        if machine_name not in journal["participants"]:
            raise ValueError("machine is not a participant in the pending CPU lane upgrade.")
        _runtime, binding = _runtime_binding(cfg, machine_runtime_root)
        if binding.machine_name != machine_name:
            raise ValueError(
                "CPU lane upgrade attestation machine does not match the local machine binding."
            )
        blockers = _blockers(cfg, machine_runtime_root=machine_runtime_root)
        if blockers:
            raise RuntimeError("CPU lane attestation found blockers: " + ", ".join(blockers))
        attestations = dict(journal["attestations"])
        attestations[machine_name] = {
            "attested_at": utc_now(),
            "machine_name": machine_name,
            "binding_machine_name": binding.machine_name,
            "project_id": binding.project_id,
            "shared_root": str(binding.shared_root),
        }
        journal["attestations"] = attestations
        atomic_replace(_journal_path(cfg), {"cpu_lane_upgrade": journal})
        return journal


def resume_cpu_lane_upgrade(
    cfg: RootConfig, *, activation_id: str, machine_runtime_root: str | Path | None = None
) -> dict[str, Any]:
    """Install the permanent gate and canonicalize drained legacy Task truth."""
    with schema_lock(cfg.shared_root):
        journal = read_json(_journal_path(cfg))["cpu_lane_upgrade"]
        if journal["activation_id"] != activation_id:
            raise ValueError("activation ID does not match the pending CPU lane upgrade.")
        if journal["phase"] == "completed":
            return journal
        if journal["phase"] == "normalizing":
            # A process died after entering the write phase.  Earlier drain acknowledgments
            # cannot prove that clients stayed stopped while the coordinator was unavailable.
            journal.update({
                "phase": "awaiting_attestations",
                "attestations": {},
                "recovery_required_at": utc_now(),
            })
            atomic_replace(_journal_path(cfg), {"cpu_lane_upgrade": journal})
            raise RuntimeError(
                "CPU lane upgrade was interrupted during normalization; collect fresh "
                "machine attestations before resuming."
            )
        if journal["phase"] != "awaiting_attestations":
            raise RuntimeError(f"CPU lane upgrade has unsupported phase {journal['phase']!r}.")
        missing = sorted(set(journal["participants"]) - set(journal["attestations"]))
        if missing:
            raise RuntimeError("CPU lane upgrade is missing machine attestations: " + ", ".join(missing))
        _validate_attestations(cfg, journal)
        blockers = _blockers(cfg, machine_runtime_root=machine_runtime_root)
        if blockers:
            raise RuntimeError("CPU lane upgrade requires a drained root: " + ", ".join(blockers))
        journal["phase"] = "normalizing"
        atomic_replace(_journal_path(cfg), {"cpu_lane_upgrade": journal})
        # QQTOOLS-COMPAT-0005: publish the schema-envelope gate before any canonical write.
        schema_path = shared_paths(cfg.shared_root)["schema"] / "version.json"
        schema = read_json(schema_path)
        schema["schema"]["required_capabilities"] = [CPU_LANE_CAPABILITY]
        atomic_replace(schema_path, schema)
        normalized = 0
        for path in iter_json(shared_paths(cfg.shared_root)["tasks"]):
            task = TaskRecord.from_dict(read_json(path))
            if task.spec.lane is None:
                task.spec.lane = "gpu"
                task.meta["revision"] += 1
                task.meta["updated_at"] = utc_now()
                atomic_replace(path, task.to_dict())
                normalized += 1
        journal.update({"phase": "completed", "completed_at": utc_now(), "normalized_tasks": normalized})
        atomic_replace(_journal_path(cfg), {"cpu_lane_upgrade": journal})
        return journal


def cpu_lane_upgrade_status(cfg: RootConfig) -> dict[str, Any]:
    """Return the current CPU lane activation status without writing state."""
    path = _journal_path(cfg)
    if path.exists():
        journal = read_json(path)["cpu_lane_upgrade"]
        return {"protocol_state": "canonical" if journal["phase"] == "completed" else "preparing", **journal}
    if is_cpu_lane_root(cfg):
        return {"protocol_state": "canonical", "phase": "completed", "activation_id": None}
    return {"protocol_state": "legacy", "phase": None, "activation_id": None}
