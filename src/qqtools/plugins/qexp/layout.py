"""Schema-5 qexp root initialization and configuration."""
from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from .config_types import RootConfig
from .runtime.paths import local_paths, machine_path, shared_log_path, shared_paths
from .runtime.locks import exclusive, schema_lock
from .runtime.records import AttemptRecord, SCHEMA_VERSION, TaskRecord, utc_now
from .runtime.store import atomic_replace, read_json
from .lease import default_lease_policy_document


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
    base_fields = {"name", "version", "minimum_reader_version", "created_at"}
    schema_fields = frozenset(schema)
    if schema_fields not in {
        frozenset(base_fields), frozenset({*base_fields, "writer_capabilities"})
    }:
        raise RuntimeError("qexp schema/version.json is malformed.")
    capabilities = schema.get("writer_capabilities")
    if capabilities is not None and (
        not isinstance(capabilities, list)
        or not all(isinstance(item, str) for item in capabilities)
        or "ready-v1" not in capabilities
    ):
        raise RuntimeError("qexp schema/version.json has unsupported writer capabilities.")
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
    required = {"schema", "project", "clock-observations", "groups", "tasks", "attempts", "operations", "idempotency",
                "claims", "machines", "locks", "events", "indexes"}
    missing = sorted(name for name in required if not (cfg.shared_root / name).exists())
    if missing:
        raise RuntimeError(f"qexp schema-6 root is incomplete; missing {missing}.")
    required_subdirs = [
        shared_paths(cfg.shared_root)["availability"],
        shared_paths(cfg.shared_root)["offer_deadlines"],
    ]
    missing_subdirs = sorted(str(path.relative_to(cfg.shared_root))
                             for path in required_subdirs if not path.exists())
    if missing_subdirs:
        raise RuntimeError(f"qexp schema-6 root is incomplete; missing {missing_subdirs}.")


def ensure_shared_layout(cfg: RootConfig) -> None:
    paths = shared_paths(cfg.shared_root)
    for name, path in paths.items():
        if name in {
            "lease_policy", "notifications", "operations_migration",
            "offer_deadlines_migration",
        }:
            continue
        path.mkdir(parents=True, exist_ok=True)
    (paths["locks"] / "groups").mkdir(exist_ok=True)
    (paths["locks"] / "tasks").mkdir(exist_ok=True)
    from .runtime.ready import ensure_ready_layout

    ensure_ready_layout(cfg)


def ensure_machine_layout(cfg: RootConfig) -> None:
    paths = local_paths(cfg.runtime_root)
    for name, path in paths.items():
        if name in {"clock_health", "lease_policy_cache"}:
            continue
        path.mkdir(parents=True, exist_ok=True)
    machine_dir = shared_paths(cfg.shared_root)["machines"] / cfg.machine_name
    for name in ("state", "events"):
        (machine_dir / name).mkdir(parents=True, exist_ok=True)


def initialize_shared_root(cfg: RootConfig) -> None:
    observed = read_schema_version(cfg)
    if observed is not None and observed != SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported qexp schema {observed!r}; refusing mixed-schema initialization."
        )
    with schema_lock(cfg.shared_root):
        existing = read_schema_version(cfg)
        if existing is not None and existing != SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported qexp schema {existing!r}; refusing mixed-schema initialization."
            )
        if existing == SCHEMA_VERSION:
            validate_root_contract(cfg)
        ensure_shared_layout(cfg)
        if existing is None:
            schema = {"schema": {
                "name": "qexp-runtime",
                "version": SCHEMA_VERSION,
                "minimum_reader_version": SCHEMA_VERSION,
                "created_at": utc_now(),
            }}
            atomic_replace(_schema_path(cfg), schema)
    ensure_machine_layout(cfg)
    identity_path = shared_paths(cfg.shared_root)["project"] / "identity.json"
    if not identity_path.exists():
        atomic_replace(identity_path, {"project": {"project_id": project_id(cfg.shared_root),
                                                     "shared_root": str(cfg.shared_root)}})
    policy_path = shared_paths(cfg.shared_root)["lease_policy"]
    if not policy_path.exists():
        atomic_replace(policy_path, default_lease_policy_document())


def migrate_schema5_to_schema6(cfg: RootConfig) -> None:
    """Hard-cut a drained schema-5 root to schema-6 lease semantics."""
    migration_lock = cfg.shared_root.parent / f".{cfg.shared_root.name}.schema6-migration.lock"
    with exclusive(migration_lock):
        if _recover_parked_schema5_root(cfg):
            return
        with schema_lock(cfg.shared_root, blocking=False) as acquired:
            if not acquired:
                raise RuntimeError("schema migration cannot start while another schema writer is active.")
        _migrate_schema5_to_schema6_locked(cfg)


def _migrate_schema5_to_schema6_locked(cfg: RootConfig) -> None:
    journal = _migration_journal_path(cfg)
    if journal.exists():
        plan = _read_migration_plan(cfg)
        phase = plan["migration"]["phase"]
        if phase == "staging":
            # The source root is untouched. Keep the incomplete stage for forensics, then retry.
            journal.unlink()
            _fsync_directory(journal.parent)
        elif phase == "ready_to_promote":
            _promote_staged_root(cfg, plan)
            return
        elif phase == "committed":
            validate_root_contract(cfg)
            return
        else:
            raise RuntimeError(f"schema migration cannot resume journal phase {phase!r}.")
    path = _schema_path(cfg)
    if not path.exists():
        raise RuntimeError("qexp root is uninitialized; cannot migrate.")
    version = read_json(path).get("schema", {}).get("version")
    if version == SCHEMA_VERSION:
        validate_root_contract(cfg)
        return
    if version != 5:
        raise RuntimeError(f"schema migration supports only schema 5, got {version!r}.")
    blockers = _migration_blockers(cfg)
    if blockers:
        raise RuntimeError(
            "schema-6 migration requires no active claims or runtime evidence; blockers: "
            + ", ".join(blockers)
        )
    _stage_and_promote_schema6_root(cfg)


def _migration_blockers(cfg: RootConfig) -> list[str]:
    """Return all authority evidence that prevents a destructive protocol cutover."""
    blockers: list[str] = []
    for task_path in shared_paths(cfg.shared_root)["tasks"].glob("*.json"):
        task = read_json(task_path).get("task", {})
        phase = task.get("state", {}).get("projection")
        if task.get("claim_control", {}).get("active_claim") or phase in {"running", "blocked"}:
            blockers.append(f"task:{task_path.stem}:{phase}")
    active_attempt_phases = {"claimed", "starting", "running", "orphaned"}
    for task_dir in shared_paths(cfg.shared_root)["attempts"].glob("*"):
        if not task_dir.is_dir():
            continue
        for attempt_path in task_dir.glob("*.json"):
            phase = read_json(attempt_path).get("attempt", {}).get("phase")
            if phase in active_attempt_phases:
                blockers.append(f"attempt:{task_dir.name}:{attempt_path.stem}:{phase}")
    for operation_path in shared_paths(cfg.shared_root)["submissions"].glob("*.json"):
        state = read_json(operation_path).get("submission", {}).get("state")
        if state not in {"committed", "aborted"}:
            blockers.append(f"submission:{operation_path.stem}:{state}")
    for name in ("group_control", "cleanup", "availability", "claim_pending"):
        directory = shared_paths(cfg.shared_root)[name]
        if directory.exists() and any(path.is_file() for path in directory.iterdir()):
            blockers.append(f"shared_runtime_evidence:{directory.name}")
    for name in ("active", "provisional", "processes", "registrations", "launch_intents",
                 "termination_decisions"):
        directory = local_paths(cfg.runtime_root)[name]
        if directory.exists() and any(directory.iterdir()):
            blockers.append(f"runtime_evidence:{directory.name}")
    return blockers


def _migration_journal_path(cfg: RootConfig) -> Path:
    return cfg.shared_root.parent / f".{cfg.shared_root.name}.schema6-migration.json"


def _tree_manifest(root: Path) -> list[dict[str, Any]]:
    """Create a stable content manifest and reject inputs unsafe to clone as authority state."""
    entries: list[dict[str, Any]] = []
    for item in sorted(root.rglob("*")):
        relative = item.relative_to(root).as_posix()
        if item.is_symlink():
            raise RuntimeError(f"schema migration rejects symbolic link: {relative}")
        if item.is_dir():
            entries.append({"path": relative, "kind": "directory"})
            continue
        if not item.is_file():
            raise RuntimeError(f"schema migration rejects non-regular entry: {relative}")
        digest = hashlib.sha256()
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        entries.append({"path": relative, "kind": "file", "sha256": digest.hexdigest(),
                        "size": item.stat().st_size})
    return entries


def _stage_and_promote_schema6_root(cfg: RootConfig) -> None:
    """Build and validate a complete replacement root before the recoverable cutover."""
    token = uuid.uuid4().hex
    stage_parent = cfg.shared_root.parent / f"{cfg.shared_root.name}.schema6-stage-{token}"
    stage = stage_parent / cfg.shared_root.name
    backup = cfg.shared_root.parent / f"{cfg.shared_root.name}.schema5-backup-{token}"
    journal = _migration_journal_path(cfg)
    if journal.exists():
        raise RuntimeError(f"schema migration journal already exists: {journal}; recover it before retrying.")
    source_manifest = _tree_manifest(cfg.shared_root)
    plan = {"migration": {"from_schema": 5, "to_schema": SCHEMA_VERSION,
            "source_root": str(cfg.shared_root), "stage_root": str(stage), "backup_root": str(backup),
            "source_manifest": source_manifest, "phase": "staging", "created_at": utc_now()}}
    atomic_replace(journal, plan)
    shutil.copytree(cfg.shared_root, stage)
    _rewrite_staged_root(cfg, stage)
    _fsync_tree(stage)
    plan["migration"]["stage_manifest"] = _tree_manifest(stage)
    plan["migration"]["phase"] = "ready_to_promote"
    atomic_replace(journal, plan)

    _promote_staged_root(cfg, plan)


def _read_migration_plan(cfg: RootConfig) -> dict[str, Any]:
    plan = read_json(_migration_journal_path(cfg))
    migration = plan.get("migration")
    if not isinstance(migration, dict):
        raise RuntimeError("schema migration journal is malformed.")
    for field in ("source_root", "stage_root", "backup_root", "phase"):
        if not isinstance(migration.get(field), str):
            raise RuntimeError("schema migration journal is malformed.")
    if Path(migration["source_root"]) != cfg.shared_root:
        raise RuntimeError("schema migration journal targets a different root.")
    for field in ("stage_root", "backup_root"):
        candidate = Path(migration[field])
        is_stage_root = (candidate.name == cfg.shared_root.name
                         and candidate.parent.parent == cfg.shared_root.parent
                         and candidate.parent.name.startswith(f"{cfg.shared_root.name}.schema6-stage-"))
        if (candidate.parent != cfg.shared_root.parent and not is_stage_root) or candidate == cfg.shared_root:
            raise RuntimeError("schema migration journal contains an unsafe sibling path.")
    return plan


def _recover_parked_schema5_root(cfg: RootConfig) -> bool:
    """Complete the only crash window where the canonical root is temporarily absent."""
    journal = _migration_journal_path(cfg)
    if not journal.exists():
        return False
    plan = _read_migration_plan(cfg)
    if plan["migration"]["phase"] != "source_parked":
        return False
    stage = Path(plan["migration"]["stage_root"])
    backup = Path(plan["migration"]["backup_root"])
    if cfg.shared_root.exists() or not stage.is_dir() or not backup.is_dir():
        raise RuntimeError("schema migration recovery found inconsistent parked-root state.")
    os.rename(stage, cfg.shared_root)
    _fsync_directory(cfg.shared_root.parent)
    plan["migration"]["phase"] = "committed"
    atomic_replace(journal, plan)
    validate_root_contract(cfg)
    return True


def _promote_staged_root(cfg: RootConfig, plan: dict[str, Any]) -> None:
    migration = plan["migration"]
    stage = Path(migration["stage_root"])
    backup = Path(migration["backup_root"])
    if not stage.is_dir() or backup.exists():
        raise RuntimeError("schema migration staging root or backup target is invalid.")
    if _tree_manifest(stage) != migration.get("stage_manifest"):
        raise RuntimeError("schema migration staged root no longer matches its verified manifest.")
    os.rename(cfg.shared_root, backup)
    _fsync_directory(cfg.shared_root.parent)
    migration["phase"] = "source_parked"
    atomic_replace(_migration_journal_path(cfg), plan)
    os.rename(stage, cfg.shared_root)
    _fsync_directory(cfg.shared_root.parent)
    migration["phase"] = "committed"
    atomic_replace(_migration_journal_path(cfg), plan)


def _rewrite_staged_root(cfg: RootConfig, stage: Path) -> None:
    """Normalize schema-5 records in an unreachable staging root."""
    for record_path in stage.rglob("*.json"):
        record = read_json(record_path)
        if isinstance(record.get("meta"), dict):
            record["meta"]["schema_version"] = SCHEMA_VERSION
        if "task" in record:
            _normalize_staged_task(record)
        if "attempt" in record:
            _normalize_staged_attempt(record)
        atomic_replace(record_path, record)
    staged_cfg = RootConfig(stage, stage.parent, cfg.machine_name, cfg.runtime_root)
    ensure_shared_layout(staged_cfg)
    policy_path = shared_paths(stage)["lease_policy"]
    if not policy_path.exists():
        atomic_replace(policy_path, default_lease_policy_document())
    atomic_replace(_schema_path(staged_cfg), {"schema": {"name": "qexp-runtime", "version": SCHEMA_VERSION,
        "minimum_reader_version": SCHEMA_VERSION, "created_at": utc_now()}})
    validate_root_contract(staged_cfg)
    _validate_staged_records(stage)


def _normalize_staged_task(record: dict[str, Any]) -> None:
    task = record["task"]
    control = task.setdefault("control", {})
    control.setdefault("cleanup_operation_id", None)
    control.setdefault("cleanup_state", None)
    runtime = task.setdefault("placement_runtime", {})
    runtime.setdefault("offer_clock_evidence", None)


def _normalize_staged_attempt(record: dict[str, Any]) -> None:
    attempt = record["attempt"]
    attempt.setdefault("authority_mode", "legacy_migrated")
    attempt.setdefault("lease", {}).setdefault("clock_evidence", None)


def _validate_staged_records(stage: Path) -> None:
    for task_path in shared_paths(stage)["tasks"].glob("*.json"):
        TaskRecord.from_dict(read_json(task_path))
    for task_dir in shared_paths(stage)["attempts"].glob("*"):
        if task_dir.is_dir():
            for attempt_path in task_dir.glob("*.json"):
                AttemptRecord.from_dict(read_json(attempt_path))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    """Make the migration backup durable before the schema commit point."""
    directories = [root]
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        elif path.is_dir():
            directories.append(path)
    for path in directories:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


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


def save_context(shared_root: str | Path) -> Path:
    """Save the canonical local default-project locator."""
    _CONTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    root = Path(shared_root).expanduser().resolve()
    atomic_replace(_CONTEXT_PATH, {"shared_root": str(root)})
    return _CONTEXT_PATH


def load_context() -> dict[str, Any] | None:
    if not _CONTEXT_PATH.exists():
        return None
    context = read_json(_CONTEXT_PATH)
    if not isinstance(context, dict):
        raise ValueError("qexp CLI context must be an object.")
    shared_root = context.get("shared_root")
    if not isinstance(shared_root, str) or not shared_root:
        raise ValueError("qexp CLI context shared_root must be a non-empty string.")
    return context


def clear_context() -> bool:
    if not _CONTEXT_PATH.exists():
        return False
    _CONTEXT_PATH.unlink()
    return True


def machine_state_path(cfg: RootConfig, name: str) -> Path:
    return shared_paths(cfg.shared_root)["machines"] / cfg.machine_name / "state" / name


def shared_attempt_log_path(cfg: RootConfig, task_id: str, attempt_id: str) -> Path:
    path = shared_log_path(cfg.shared_root, task_id, attempt_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def runtime_pid_path(cfg: RootConfig) -> Path:
    return cfg.runtime_root / "agent" / "agent.pid"
