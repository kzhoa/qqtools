"""Machine metadata lifecycle, including compatibility normalization."""
from __future__ import annotations

from pathlib import Path

from .config_types import MachinePolicy, RootConfig
from .layout import (initialize_shared_root, load_machine_record, project_id,
                     save_machine_record)
from .policy import normalize_agent_mode, resolve_machine_policy
from .runtime.locks import machine_lock
from .runtime.filesystem_qualification import (
    validate_existing_filesystem_qualification,
)
from .runtime.store import read_json

MACHINE_AGENT_RUNTIME = "machine"


def has_legacy_agent_metadata(cfg: RootConfig) -> bool:
    """Return whether any machine record in this project predates the global agent."""
    machines_root = cfg.shared_root / "machines"
    for record_path in machines_root.glob("*/machine.json"):
        record = read_json(record_path)
        if record.get("machine", {}).get("agent_runtime") != MACHINE_AGENT_RUNTIME:
            return True
    return False


def save_machine_config(cfg: RootConfig, *, agent_mode: str | None) -> None:
    """Normalize and persist the current machine's metadata record."""
    with machine_lock(cfg.shared_root, cfg.machine_name):
        current = load_machine_record(cfg) or {}
        current["machine"] = {
            "machine_name": cfg.machine_name,
            "project_id": project_id(cfg.shared_root),
            "shared_root": str(cfg.shared_root),
            "runtime_root": str(cfg.runtime_root),
            "agent_mode": normalize_agent_mode(agent_mode),
            "agent_runtime": MACHINE_AGENT_RUNTIME,
        }
        save_machine_record(cfg, current)


def load_machine_policy(cfg: RootConfig) -> MachinePolicy:
    """Read machine metadata and map legacy modes to the active policy."""
    record = load_machine_record(cfg)
    raw_mode = record.get("machine", {}).get("agent_mode") if record else None
    return resolve_machine_policy(raw_mode)


def is_legacy_agent_project(cfg: RootConfig) -> bool:
    """Return whether this machine record predates the global-agent runtime."""
    record = load_machine_record(cfg) or {}
    return record.get("machine", {}).get("agent_runtime") != MACHINE_AGENT_RUNTIME


def init_shared_root(shared_root: Path, machine_name: str, *, agent_mode: str = "on_demand",
                     runtime_root: Path | None = None) -> RootConfig:
    shared_root = Path(shared_root).expanduser().resolve()
    runtime_root = runtime_root or (
        Path.home() / ".qqtools" / "qexp-runtime" / project_id(shared_root) / machine_name
    )
    cfg = RootConfig(shared_root, shared_root.parent, machine_name, runtime_root)
    if has_legacy_agent_metadata(cfg):
        raise ValueError("legacy project metadata requires 'qexp agent migrate-project'.")
    initialize_shared_root(cfg)
    validate_existing_filesystem_qualification(cfg)
    save_machine_config(cfg, agent_mode=agent_mode)
    return cfg
