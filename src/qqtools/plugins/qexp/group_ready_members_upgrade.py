"""Explicit resumable activation for group-ready-members-v1."""
# QQTOOLS-COMPAT-0008: legacy roots retain Task-scan Group synchronization through 1.3.16.
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from .config_types import RootConfig
from .layout import GROUP_READY_MEMBERS_CAPABILITY
from .runtime.ready.group_members import (
    advance_group_ready_members_build,
    begin_group_ready_members_build,
    read_group_ready_members_state,
)
from .runtime.locks import schema_lock
from .runtime.paths import shared_paths
from .runtime.records import utc_now
from .runtime.store import atomic_replace, read_json

_JOURNAL = "group-ready-members-upgrade.json"


def _path(cfg: RootConfig) -> Path:
    return shared_paths(cfg.shared_root)["schema"] / _JOURNAL


def _participants(cfg: RootConfig) -> list[str]:
    return sorted(path.name for path in shared_paths(cfg.shared_root)["machines"].iterdir() if path.is_dir())


def _incompatible_agents(cfg: RootConfig) -> list[str]:
    incompatible: list[str] = []
    for machine in _participants(cfg):
        path = shared_paths(cfg.shared_root)["machines"] / machine / "state" / "agent.json"
        if not path.exists():
            continue
        try:
            agent = read_json(path)["agent"]
        except (KeyError, TypeError, ValueError):
            incompatible.append(machine)
            continue
        if agent.get("observed_state") in {"active", "idle"} and GROUP_READY_MEMBERS_CAPABILITY not in agent.get("writer_capabilities", []):
            incompatible.append(machine)
    return incompatible


def group_ready_members_upgrade_status(cfg: RootConfig) -> dict[str, Any]:
    if _path(cfg).exists():
        return read_json(_path(cfg))["group_ready_members_upgrade"]
    try:
        state = read_group_ready_members_state(cfg)
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        state = {"state": "legacy"}
    return {"phase": state["state"], "activation_id": None, "projection": state}


def check_group_ready_members_upgrade(cfg: RootConfig) -> dict[str, Any]:
    return {
        "shared_root": str(cfg.shared_root), "capability": GROUP_READY_MEMBERS_CAPABILITY,
        "phase": group_ready_members_upgrade_status(cfg)["phase"],
        "incompatible_agents": _incompatible_agents(cfg),
    }


def start_group_ready_members_upgrade(cfg: RootConfig) -> dict[str, Any]:
    with schema_lock(cfg.shared_root):
        if _path(cfg).exists():
            return read_json(_path(cfg))["group_ready_members_upgrade"]
        incompatible = _incompatible_agents(cfg)
        if incompatible:
            raise RuntimeError("group-ready-members activation requires capable agents: " + ", ".join(incompatible))
        schema_path = shared_paths(cfg.shared_root)["schema"] / "version.json"
        schema = read_json(schema_path)
        required = schema["schema"].setdefault("required_capabilities", [])
        begin_group_ready_members_build(
            cfg, is_repair=GROUP_READY_MEMBERS_CAPABILITY not in required
        )
        if GROUP_READY_MEMBERS_CAPABILITY not in required:
            required.append(GROUP_READY_MEMBERS_CAPABILITY)
            required.sort()
            atomic_replace(schema_path, schema)
        value = {
            "activation_id": uuid.uuid4().hex, "phase": "building",
            "capability": GROUP_READY_MEMBERS_CAPABILITY, "participants": _participants(cfg),
            "attestations": {}, "created_at": utc_now(), "updated_at": utc_now(),
        }
        atomic_replace(_path(cfg), {"group_ready_members_upgrade": value})
        return value


def attest_group_ready_members_upgrade(
    cfg: RootConfig, *, activation_id: str, machine_name: str,
) -> dict[str, Any]:
    with schema_lock(cfg.shared_root):
        value = read_json(_path(cfg))["group_ready_members_upgrade"]
        if value["activation_id"] != activation_id or value["phase"] != "building":
            raise ValueError("activation ID does not identify a building group-ready-members upgrade.")
        if machine_name not in value["participants"]:
            raise ValueError("attestation machine is not a declared participant.")
        if machine_name in _incompatible_agents(cfg):
            raise RuntimeError(f"machine {machine_name!r} has not declared group-ready-members-v1.")
        value["attestations"][machine_name] = {"machine_name": machine_name, "attested_at": utc_now()}
        value["updated_at"] = utc_now()
        atomic_replace(_path(cfg), {"group_ready_members_upgrade": value})
        return value


def resume_group_ready_members_upgrade(
    cfg: RootConfig, *, activation_id: str, max_tasks: int = 64,
) -> dict[str, Any]:
    with schema_lock(cfg.shared_root):
        value = read_json(_path(cfg))["group_ready_members_upgrade"]
        if value["activation_id"] != activation_id:
            raise ValueError("activation ID does not match the group-ready-members upgrade.")
        if value["phase"] == "completed":
            return value
        missing = sorted(set(value["participants"]) - set(value["attestations"]))
        if missing:
            raise RuntimeError("group-ready-members upgrade is missing attestations: " + ", ".join(missing))
        incompatible = _incompatible_agents(cfg)
        if incompatible:
            raise RuntimeError("group-ready-members activation found incompatible agents: " + ", ".join(incompatible))
        projection = advance_group_ready_members_build(cfg, max_tasks=max_tasks)
        value["projection"] = projection
        value["updated_at"] = utc_now()
        if projection["state"] == "active":
            value["phase"] = "completed"
            value["completed_at"] = utc_now()
        elif projection["state"] == "degraded":
            value["phase"] = "blocked"
        atomic_replace(_path(cfg), {"group_ready_members_upgrade": value})
        return value
