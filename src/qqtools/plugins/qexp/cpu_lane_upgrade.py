"""Explicit, resumable activation of CPU lanes on drained schema-6 roots."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from .config_types import RootConfig
from .layout import CPU_LANE_CAPABILITY, is_cpu_lane_root
from .runtime.locks import schema_lock
from .runtime.paths import local_paths, shared_paths
from .runtime.records import TaskRecord, utc_now
from .runtime.store import atomic_replace, iter_json, read_json

_JOURNAL = "cpu-lane-upgrade.json"


def _journal_path(cfg: RootConfig) -> Path:
    return shared_paths(cfg.shared_root)["schema"] / _JOURNAL


def _blockers(cfg: RootConfig) -> list[str]:
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
    for directory in (
        "active",
        "provisional",
        "cpu_active",
        "cpu_provisional",
        "processes",
        "registrations",
        "launch_intents",
        "termination_decisions",
    ):
        if iter_json(local_paths(cfg.runtime_root)[directory]):
            blockers.append(f"runtime:{directory}")
    return blockers


def check_cpu_lane_upgrade(cfg: RootConfig) -> dict[str, Any]:
    """Return a read-only legacy-root activation preflight."""
    if _journal_path(cfg).exists():
        journal = read_json(_journal_path(cfg))["cpu_lane_upgrade"]
        protocol_state = "canonical" if journal["phase"] == "completed" else "preparing"
        return {"shared_root": str(cfg.shared_root), "protocol_state": protocol_state, **journal}
    canonical = is_cpu_lane_root(cfg)
    return {
        "shared_root": str(cfg.shared_root),
        "protocol_state": "canonical" if canonical else "legacy",
        "blockers": [] if canonical else _blockers(cfg),
        "journal": _journal_path(cfg).exists(),
    }


def start_cpu_lane_upgrade(cfg: RootConfig) -> dict[str, Any]:
    """Create the sole activation session after the shared root is drained."""
    with schema_lock(cfg.shared_root):
        journal_path = _journal_path(cfg)
        if journal_path.exists():
            return read_json(journal_path)["cpu_lane_upgrade"]
        if is_cpu_lane_root(cfg):
            return {"protocol_state": "canonical", "phase": "completed", "activation_id": None}
        blockers = _blockers(cfg)
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


def attest_cpu_lane_upgrade(cfg: RootConfig, *, activation_id: str, machine_name: str) -> dict[str, Any]:
    """Record one operator-confirmed participant after local shared-state checks."""
    with schema_lock(cfg.shared_root):
        journal = read_json(_journal_path(cfg))["cpu_lane_upgrade"]
        if journal["activation_id"] != activation_id:
            raise ValueError("activation ID does not match the pending CPU lane upgrade.")
        if machine_name not in journal["participants"]:
            raise ValueError("machine is not a participant in the pending CPU lane upgrade.")
        blockers = _blockers(cfg)
        if blockers:
            raise RuntimeError("CPU lane attestation found blockers: " + ", ".join(blockers))
        attestations = dict(journal["attestations"])
        attestations[machine_name] = {"attested_at": utc_now(), "machine_name": machine_name}
        journal["attestations"] = attestations
        atomic_replace(_journal_path(cfg), {"cpu_lane_upgrade": journal})
        return journal


def resume_cpu_lane_upgrade(cfg: RootConfig, *, activation_id: str) -> dict[str, Any]:
    """Install the permanent gate and canonicalize drained legacy Task truth."""
    with schema_lock(cfg.shared_root):
        journal = read_json(_journal_path(cfg))["cpu_lane_upgrade"]
        if journal["activation_id"] != activation_id:
            raise ValueError("activation ID does not match the pending CPU lane upgrade.")
        missing = sorted(set(journal["participants"]) - set(journal["attestations"]))
        if missing:
            raise RuntimeError("CPU lane upgrade is missing machine attestations: " + ", ".join(missing))
        blockers = _blockers(cfg)
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
