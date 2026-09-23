"""Shared admission fencing before retained local writer capture."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..agent.registration import RECOVERY_REGISTRATION_VERSION
from ..config_types import RootConfig
from ..layout import LOCAL_RECOVERY_CAPABILITY, load_machine_record, load_machine_registration, validate_root_contract
from .locks import schema_lock
from .paths import machine_registration_path, shared_paths
from .responsibility_store import DurableIO
from .store import atomic_replace, read_json
from .upgrade.framework import UpgradeCoordinator, pending_upgrade_requires_completion
from .upgrade.production import UpgradeJournalMigration

if TYPE_CHECKING:
    from ..agent.bindings import ProjectBinding
    from ..agent.context import MachineRuntime


@dataclass(frozen=True)
class RecoveryAdmission:
    is_fenced: bool
    blockers: tuple[str, ...] = ()


def _upgrade_blocker(cfg: RootConfig) -> str | None:
    if UpgradeJournalMigration().is_applicable(cfg):
        return "upgrade_manifest_not_ready"
    status = UpgradeCoordinator(cfg).status()
    if pending_upgrade_requires_completion(status):
        return "upgrade_coordinator_pending"
    return None


def _participants(runtime: MachineRuntime, binding: ProjectBinding, cfg: RootConfig) -> tuple[list[Path], list[str]]:
    directories = []
    blockers = []
    with os.scandir(shared_paths(cfg.shared_root)["machines"]) as entries:
        for entry in entries:
            if not entry.is_dir(follow_symlinks=False):
                blockers.append(f"invalid_machine_entry:{entry.name}")
                continue
            peer = RootConfig(cfg.shared_root, cfg.project_root, entry.name, cfg.runtime_root)
            metadata = load_machine_record(peer)
            machine = metadata.get("machine") if isinstance(metadata, dict) else None
            if not isinstance(machine, dict) or any(
                machine.get(key) != value
                for key, value in {
                    "machine_name": entry.name,
                    "project_id": binding.project_id,
                    "shared_root": str(cfg.shared_root),
                    "agent_runtime": "machine",
                }.items()
            ):
                blockers.append(f"machine_not_prepared:{entry.name}")
                continue
            standalone_root = machine.get("runtime_root")
            # This path belongs to the peer host. Check its persisted canonical
            # spelling without resolving that host's symlinks on this machine.
            if (
                not isinstance(standalone_root, str)
                or not standalone_root
                or "\x00" in standalone_root
                or not Path(standalone_root).is_absolute()
                or str(Path(standalone_root)) != standalone_root
                or ".." in Path(standalone_root).parts
            ):
                blockers.append(f"machine_not_prepared:{entry.name}")
                continue
            raw = load_machine_registration(peer)
            record = raw.get("registration") if isinstance(raw, dict) else None
            try:
                if not isinstance(record, dict):
                    raise ValueError("missing registration")
                runtime.registration.validate_registration_record(record, binding.project_id, peer)
            except (RuntimeError, ValueError):
                blockers.append(f"invalid_registration:{entry.name}")
                continue
            if record.get("state") == "superseded":
                if (
                    not isinstance(record.get("superseded_by_generation"), str)
                    or not record["superseded_by_generation"]
                ):
                    blockers.append(f"invalid_supersession:{entry.name}")
                    continue
            elif record.get("state") != "eligible" or record["version"] != RECOVERY_REGISTRATION_VERSION:
                # An expired or offline registration still owns its logical name.
                # Heartbeat absence is never evidence of durable retirement.
                blockers.append(f"registration_not_prepared:{entry.name}")
                continue
            directories.append(machine_registration_path(cfg.shared_root, entry.name).parent)
    return directories, blockers


def fence_recovery_admission(runtime: MachineRuntime, binding: ProjectBinding) -> RecoveryAdmission:
    """Install the root capability only after every participant is fenced.

    The registration fence precedes the nonblocking schema fence, matching the
    registered claim path. A busy schema defers activation rather than introducing
    a registration/schema lock inversion. This inventories machine metadata only;
    it neither scans execution history nor certifies any local capture.
    """
    if runtime._scheduler_authority_pid != os.getpid():
        raise RuntimeError("recovery admission fencing requires machine scheduler authority")
    with runtime._scheduler_authority_gate:
        if runtime._scheduler_authority_pid != os.getpid():
            raise RuntimeError("recovery admission fencing requires machine scheduler authority")
        with runtime.registry_guard(blocking=False) as registered:
            if not registered:
                return RecoveryAdmission(False, ("registry_busy",))
            if binding not in runtime.load_registry()[1]:
                return RecoveryAdmission(False, ("binding_changed",))
            cfg = binding.root_config()
            with runtime.binding_write_guard(binding) as eligible:
                if not eligible:
                    return RecoveryAdmission(False, ("registration_ineligible",))
                if runtime.registration_status(binding)["registration_version"] != RECOVERY_REGISTRATION_VERSION:
                    return RecoveryAdmission(False, ("local_registration_not_prepared",))
                with schema_lock(cfg.shared_root, blocking=False) as acquired:
                    if not acquired:
                        return RecoveryAdmission(False, ("schema_busy",))
                    validate_root_contract(cfg)
                    path = shared_paths(cfg.shared_root)["schema"] / "version.json"
                    schema = read_json(path)
                    capabilities = schema["schema"]["required_capabilities"]
                    io = DurableIO()
                    if LOCAL_RECOVERY_CAPABILITY in capabilities:
                        # Finish an uncertain previous rename barrier before the
                        # caller treats a visible capability as durable admission.
                        io.sync_directory(path.parent, "recovery_admission")
                        return RecoveryAdmission(True)
                    upgrade_blocker = _upgrade_blocker(cfg)
                    if upgrade_blocker is not None:
                        return RecoveryAdmission(False, (upgrade_blocker,))
                    directories, blockers = _participants(runtime, binding, cfg)
                    if blockers:
                        return RecoveryAdmission(False, tuple(sorted(blockers)))
                    for directory in directories:
                        io.sync_directory(directory, "recovery_participant")
                    if not runtime.registration_status(binding)["write_eligible"]:
                        return RecoveryAdmission(False, ("registration_ineligible",))
                    schema["schema"]["required_capabilities"] = [*capabilities, LOCAL_RECOVERY_CAPABILITY]
                    atomic_replace(path, schema)
                    return RecoveryAdmission(True)


def inspect_recovery_admission(runtime: MachineRuntime, binding: ProjectBinding) -> dict:
    """Read an advisory enrollment snapshot without activating or repairing state."""
    try:
        cfg = binding.root_config()
        status = runtime.registration_status(binding)
        if status["state"] == "superseded":
            return {"state": "superseded", "blockers": [], "diagnostic_only": True}
        if not status["write_eligible"] or status["registration_version"] != RECOVERY_REGISTRATION_VERSION:
            blockers = ["local_registration_not_prepared"]
        elif (
            LOCAL_RECOVERY_CAPABILITY
            in read_json(shared_paths(cfg.shared_root)["schema"] / "version.json")["schema"]["required_capabilities"]
        ):
            return {"state": "admission_fenced", "blockers": [], "diagnostic_only": True}
        elif (upgrade_blocker := _upgrade_blocker(cfg)) is not None:
            blockers = [upgrade_blocker]
        else:
            _directories, blockers = _participants(runtime, binding, cfg)
            if not blockers:
                blockers = ["admission_fence_pending"]
        return {"state": "waiting", "blockers": sorted(blockers), "diagnostic_only": True}
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        return {"state": "unavailable", "blockers": [str(exc)], "diagnostic_only": True}
