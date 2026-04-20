from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from pathlib import Path

from .models import (
    AGENT_MODE_ON_DEMAND,
    AGENT_MODES,
    ROOT_SCOPE_PROJECT,
    SCHEMA_VERSION,
    GpuInventory,
    Machine,
    RootManifest,
    generate_id,
    utc_now_iso,
    validate_machine_name,
)

DEFAULT_RUNTIME_BASE = "~/.qqtools/qexp-runtime"
DEFAULT_CONTEXT_FILE = "~/.qqtools/qexp-context.json"
FORBIDDEN_TRUTH_LAYOUT_DIRS = ("groups", "experiments")


@dataclass(slots=True)
class RootConfig:
    shared_root: Path
    project_root: Path
    machine_name: str
    runtime_root: Path

    def __post_init__(self) -> None:
        validate_machine_name(self.machine_name)
        self.shared_root = Path(self.shared_root).expanduser().resolve()
        self.project_root = Path(self.project_root).expanduser().resolve()
        self.runtime_root = Path(self.runtime_root).expanduser().resolve()


# ---------------------------------------------------------------------------
# Path helpers — shared root
# ---------------------------------------------------------------------------


def global_dir(cfg: RootConfig) -> Path:
    return cfg.shared_root / "global"


def global_schema_dir(cfg: RootConfig) -> Path:
    return global_dir(cfg) / "schema"


def global_tasks_dir(cfg: RootConfig) -> Path:
    return global_dir(cfg) / "tasks"


def global_batches_dir(cfg: RootConfig) -> Path:
    return global_dir(cfg) / "batches"


def global_indexes_dir(cfg: RootConfig) -> Path:
    return global_dir(cfg) / "indexes"


def global_locks_dir(cfg: RootConfig) -> Path:
    return global_dir(cfg) / "locks"


def global_events_dir(cfg: RootConfig) -> Path:
    return global_dir(cfg) / "events"


def machine_dir(cfg: RootConfig) -> Path:
    return cfg.shared_root / "machines" / cfg.machine_name


def machine_claims_active_dir(cfg: RootConfig) -> Path:
    return machine_dir(cfg) / "claims" / "active"


def machine_claims_released_dir(cfg: RootConfig) -> Path:
    return machine_dir(cfg) / "claims" / "released"


def machine_events_dir(cfg: RootConfig) -> Path:
    return machine_dir(cfg) / "events"


def machine_state_dir(cfg: RootConfig) -> Path:
    return machine_dir(cfg) / "state"


# Specific file paths


def task_path(cfg: RootConfig, task_id: str) -> Path:
    return global_tasks_dir(cfg) / f"{task_id}.json"


def batch_path(cfg: RootConfig, batch_id: str) -> Path:
    return global_batches_dir(cfg) / f"{batch_id}.json"


def machine_json_path(cfg: RootConfig) -> Path:
    return machine_dir(cfg) / "machine.json"


def schema_version_path(cfg: RootConfig) -> Path:
    return global_schema_dir(cfg) / "version.json"


def root_manifest_path(cfg: RootConfig) -> Path:
    return global_schema_dir(cfg) / "root.json"


def agent_state_path(cfg: RootConfig) -> Path:
    return machine_state_dir(cfg) / "agent.json"


def gpu_state_path(cfg: RootConfig) -> Path:
    return machine_state_dir(cfg) / "gpu.json"


def summary_state_path(cfg: RootConfig) -> Path:
    return machine_state_dir(cfg) / "summary.json"


# Runtime paths (local only, not shared)


def runtime_logs_dir(cfg: RootConfig) -> Path:
    return cfg.runtime_root / "logs"


def runtime_log_path(cfg: RootConfig, task_id: str) -> Path:
    return runtime_logs_dir(cfg) / f"{task_id}.log"


def runtime_pid_path(cfg: RootConfig) -> Path:
    return cfg.runtime_root / "agent.pid"


# Index directories


def index_by_state_dir(cfg: RootConfig) -> Path:
    return global_indexes_dir(cfg) / "tasks_by_state"


def index_by_batch_dir(cfg: RootConfig) -> Path:
    return global_indexes_dir(cfg) / "tasks_by_batch"


def index_by_machine_dir(cfg: RootConfig) -> Path:
    return global_indexes_dir(cfg) / "tasks_by_machine"


def index_by_group_dir(cfg: RootConfig) -> Path:
    return global_indexes_dir(cfg) / "tasks_by_group"


def batch_index_by_group_dir(cfg: RootConfig) -> Path:
    return global_indexes_dir(cfg) / "batches_by_group"


# Lock paths


def submit_lock_path(cfg: RootConfig) -> Path:
    return global_locks_dir(cfg) / "submit"


def batch_lock_path(cfg: RootConfig) -> Path:
    return global_locks_dir(cfg) / "batch"


def migrate_lock_path(cfg: RootConfig) -> Path:
    return global_locks_dir(cfg) / "migrate"


def clean_lock_path(cfg: RootConfig) -> Path:
    return global_locks_dir(cfg) / "clean"


# ---------------------------------------------------------------------------
# Layout creation
# ---------------------------------------------------------------------------

_SHARED_DIRS = [
    global_schema_dir,
    global_tasks_dir,
    global_batches_dir,
    global_locks_dir,
    global_events_dir,
    index_by_state_dir,
    index_by_batch_dir,
    index_by_machine_dir,
    index_by_group_dir,
    batch_index_by_group_dir,
]

_MACHINE_DIRS = [
    machine_claims_active_dir,
    machine_claims_released_dir,
    machine_events_dir,
    machine_state_dir,
]


def ensure_shared_layout(cfg: RootConfig) -> None:
    for dir_fn in _SHARED_DIRS:
        dir_fn(cfg).mkdir(parents=True, exist_ok=True)


def ensure_machine_layout(cfg: RootConfig) -> None:
    for dir_fn in _MACHINE_DIRS:
        dir_fn(cfg).mkdir(parents=True, exist_ok=True)


def ensure_runtime_layout(cfg: RootConfig) -> None:
    try:
        cfg.runtime_root.mkdir(parents=True, exist_ok=True)
        runtime_logs_dir(cfg).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot create runtime directory at {cfg.runtime_root}: {exc}. "
            f"Set --runtime-root or QEXP_RUNTIME_ROOT to a writable location."
        ) from exc


def write_schema_version(cfg: RootConfig, version: str = SCHEMA_VERSION) -> None:
    path = schema_version_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    payload = json.dumps({"schema_version": version}, indent=2)
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def read_schema_version(cfg: RootConfig) -> str | None:
    path = schema_version_path(cfg)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("schema_version")


def _resolve_project_root(shared_root: Path) -> Path:
    resolved = Path(shared_root).expanduser().resolve()
    if resolved.name != ".qexp":
        raise ValueError(
            "shared_root must be the project control directory named '.qexp'. "
            f"Got {resolved}."
        )
    return resolved.parent


def create_root_manifest(cfg: RootConfig) -> RootManifest:
    return RootManifest(
        schema_version=SCHEMA_VERSION,
        root_scope=ROOT_SCOPE_PROJECT,
        control_plane_id=f"cp_{generate_id()}",
        shared_root=str(cfg.shared_root),
        project_root=str(cfg.project_root),
        layout_version=SCHEMA_VERSION,
        created_at=utc_now_iso(),
        created_by_machine=cfg.machine_name,
    )


def write_root_manifest(cfg: RootConfig, manifest: RootManifest | None = None) -> RootManifest:
    if manifest is None:
        manifest = create_root_manifest(cfg)
    path = root_manifest_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp, path)
    return manifest


def read_root_manifest(cfg: RootConfig) -> RootManifest | None:
    path = root_manifest_path(cfg)
    if not path.is_file():
        return None
    return RootManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def ensure_root_manifest(cfg: RootConfig) -> RootManifest:
    manifest = read_root_manifest(cfg)
    if manifest is not None:
        return manifest
    return write_root_manifest(cfg)


def list_forbidden_truth_layout_dirs(cfg: RootConfig) -> list[Path]:
    return [
        cfg.shared_root / dirname
        for dirname in FORBIDDEN_TRUTH_LAYOUT_DIRS
        if (cfg.shared_root / dirname).exists()
    ]


def validate_root_contract(cfg: RootConfig) -> RootManifest:
    manifest = read_root_manifest(cfg)
    if manifest is None:
        raise FileNotFoundError(
            f"Root manifest not found at {root_manifest_path(cfg)}. "
            "Run 'qexp init' to initialize the project control root."
        )
    if Path(manifest.shared_root).resolve() != cfg.shared_root:
        raise ValueError(
            "shared_root does not match the initialized control-plane manifest: "
            f"manifest={manifest.shared_root}, requested={cfg.shared_root}."
        )
    if Path(manifest.project_root).resolve() != cfg.project_root:
        raise ValueError(
            "project_root does not match the initialized control-plane manifest: "
            f"manifest={manifest.project_root}, inferred={cfg.project_root}."
        )

    for path in list_forbidden_truth_layout_dirs(cfg):
        if path.exists():
            raise ValueError(
                f"Forbidden truth-layout directory detected at {path}. "
                "qexp only permits global object truth and machine-private directories."
            )
    return manifest


# ---------------------------------------------------------------------------
# RootConfig construction
# ---------------------------------------------------------------------------


def load_root_config(
    shared_root: str | Path,
    machine_name: str,
    runtime_root: str | Path | None = None,
    require_initialized: bool = False,
) -> RootConfig:
    shared_root_path = Path(shared_root)
    project_root = _resolve_project_root(shared_root_path)
    if runtime_root is None:
        runtime_root = Path(DEFAULT_RUNTIME_BASE).expanduser() / machine_name
    cfg = RootConfig(
        shared_root=shared_root_path,
        project_root=project_root,
        machine_name=machine_name,
        runtime_root=Path(runtime_root),
    )
    if require_initialized:
        validate_root_contract(cfg)
    elif root_manifest_path(cfg).is_file():
        validate_root_contract(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Init command
# ---------------------------------------------------------------------------


def init_shared_root(
    shared_root: Path,
    machine_name: str,
    agent_mode: str = AGENT_MODE_ON_DEMAND,
    runtime_root: Path | None = None,
) -> RootConfig:
    if agent_mode not in AGENT_MODES:
        raise ValueError(f"agent_mode must be one of {AGENT_MODES}, got {agent_mode!r}.")

    cfg = load_root_config(shared_root, machine_name, runtime_root)
    ensure_shared_layout(cfg)
    ensure_machine_layout(cfg)
    ensure_runtime_layout(cfg)
    write_schema_version(cfg)
    ensure_root_manifest(cfg)
    validate_root_contract(cfg)

    machine = Machine(
        machine_name=cfg.machine_name,
        hostname=socket.gethostname(),
        shared_root=str(cfg.shared_root),
        runtime_root=str(cfg.runtime_root),
        agent_mode=agent_mode,
        gpu_inventory=GpuInventory(),
    )

    machine_path = machine_json_path(cfg)
    tmp = machine_path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(machine.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(tmp, machine_path)

    return cfg


# ---------------------------------------------------------------------------
# CLI context persistence
# ---------------------------------------------------------------------------

_context_file_override: str | None = None


def _context_path() -> Path:
    if _context_file_override is not None:
        return Path(_context_file_override)
    return Path(DEFAULT_CONTEXT_FILE).expanduser()


def save_context(
    shared_root: str,
    machine_name: str,
    runtime_root: str | None = None,
) -> Path:
    path = _context_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "shared_root": shared_root,
        "machine": machine_name,
    }
    if runtime_root is not None:
        payload["runtime_root"] = runtime_root
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_context() -> dict | None:
    path = _context_path()
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def clear_context() -> bool:
    path = _context_path()
    if path.is_file():
        path.unlink()
        return True
    return False
