"""Schema-5 qexp root initialization and configuration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config_types import RootConfig
from .runtime.paths import local_paths, machine_path, shared_paths
from .runtime.records import SCHEMA_VERSION, utc_now
from .runtime.store import atomic_replace, read_json


def _schema_path(cfg: RootConfig) -> Path:
    return shared_paths(cfg.shared_root)["schema"] / "version.json"


def read_schema_version(cfg: RootConfig) -> int | None:
    path = _schema_path(cfg)
    if not path.exists():
        return None
    value = read_json(path)
    schema = value.get("schema")
    if not isinstance(schema, dict):
        raise RuntimeError("qexp schema/version.json is malformed.")
    version = schema.get("version")
    if version != SCHEMA_VERSION:
        return version
    if set(schema) != {"name", "version", "minimum_reader_version", "created_at"}:
        raise RuntimeError("qexp schema/version.json is malformed.")
    return version


def validate_root_contract(cfg: RootConfig) -> None:
    version = read_schema_version(cfg)
    if version != SCHEMA_VERSION:
        if version is None:
            raise RuntimeError("qexp root is uninitialized; run qexp init first.")
        raise RuntimeError(f"Unsupported qexp schema {version!r}; expected schema {SCHEMA_VERSION}.")
    forbidden = {"global", "batches", "resubmit", "resubmit_operations"}
    present = forbidden.intersection(path.name for path in cfg.shared_root.iterdir())
    if present:
        raise RuntimeError(f"Mixed qexp schema root contains obsolete paths: {sorted(present)}.")
    required = {"schema", "project", "groups", "tasks", "attempts", "operations", "idempotency",
                "claims", "machines", "locks", "events", "indexes"}
    missing = sorted(name for name in required if not (cfg.shared_root / name).exists())
    if missing:
        raise RuntimeError(f"qexp schema-5 root is incomplete; missing {missing}.")


def ensure_shared_layout(cfg: RootConfig) -> None:
    paths = shared_paths(cfg.shared_root)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    (paths["locks"] / "groups").mkdir(exist_ok=True)
    (paths["locks"] / "tasks").mkdir(exist_ok=True)


def ensure_machine_layout(cfg: RootConfig) -> None:
    for path in local_paths(cfg.runtime_root).values():
        path.mkdir(parents=True, exist_ok=True)
    machine_dir = shared_paths(cfg.shared_root)["machines"] / cfg.machine_name
    for name in ("state", "events"):
        (machine_dir / name).mkdir(parents=True, exist_ok=True)


def initialize_shared_root(cfg: RootConfig) -> None:
    existing = read_schema_version(cfg)
    if existing is not None and existing != SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported qexp schema {existing!r}; refusing mixed-schema initialization.")
    if existing == SCHEMA_VERSION:
        validate_root_contract(cfg)
    ensure_shared_layout(cfg)
    ensure_machine_layout(cfg)
    schema = {"schema": {"name": "qexp-runtime", "version": SCHEMA_VERSION,
                          "minimum_reader_version": SCHEMA_VERSION, "created_at": utc_now()}}
    atomic_replace(_schema_path(cfg), schema)
    identity_path = shared_paths(cfg.shared_root)["project"] / "identity.json"
    if not identity_path.exists():
        atomic_replace(identity_path, {"project": {"project_id": project_id(cfg.shared_root),
                                                     "shared_root": str(cfg.shared_root)}})


def load_machine_record(cfg: RootConfig) -> dict[str, Any] | None:
    path = machine_path(cfg.shared_root, cfg.machine_name)
    return read_json(path) if path.exists() else None


def save_machine_record(cfg: RootConfig, record: dict[str, Any]) -> None:
    atomic_replace(machine_path(cfg.shared_root, cfg.machine_name), record)


def project_id(shared_root: Path) -> str:
    import hashlib
    return hashlib.sha256(str(shared_root).encode()).hexdigest()[:16]


def load_root_config(shared_root: str | Path, machine_name: str, runtime_root: str | Path | None = None,
                     *, require_initialized: bool = False) -> RootConfig:
    shared_root = Path(shared_root).expanduser().resolve()
    cfg = RootConfig(shared_root, shared_root.parent, machine_name,
                     Path(runtime_root) if runtime_root else Path.home() / ".qqtools" / "qexp-runtime" / project_id(shared_root) / machine_name)
    if require_initialized:
        validate_root_contract(cfg)
    return cfg


_CONTEXT_PATH = Path.home() / ".qqtools" / "qexp-context.json"


def save_context(shared_root: str, machine: str, runtime_root: str | None = None) -> Path:
    _CONTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_replace(_CONTEXT_PATH, {"shared_root": shared_root, "machine": machine, "runtime_root": runtime_root})
    return _CONTEXT_PATH


def load_context() -> dict[str, Any] | None:
    return read_json(_CONTEXT_PATH) if _CONTEXT_PATH.exists() else None


def clear_context() -> bool:
    if not _CONTEXT_PATH.exists():
        return False
    _CONTEXT_PATH.unlink()
    return True


def machine_state_path(cfg: RootConfig, name: str) -> Path:
    return shared_paths(cfg.shared_root)["machines"] / cfg.machine_name / "state" / name


def runtime_log_path(cfg: RootConfig, task_id: str, attempt_id: str | None = None) -> Path:
    suffix = attempt_id or "current"
    path = cfg.runtime_root / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{task_id}-{suffix}.log"


def runtime_pid_path(cfg: RootConfig) -> Path:
    return cfg.runtime_root / "agent" / "agent.pid"
