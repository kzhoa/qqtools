"""Project registration and migration workflows."""

from __future__ import annotations

import os
import shlex
import signal
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..config_types import RootConfig
from ..layout import (
    load_machine_record,
    load_machine_registration,
    load_root_config,
    runtime_pid_path,
    save_machine_registration,
)
from ..legacy_agent import get_agent_status
from ..machine_config import is_legacy_agent_project, save_machine_config
from ..runtime.locks import exclusive
from ..runtime.paths import local_paths
from ..runtime.records import utc_now
from ..runtime.store import atomic_replace, iter_json, read_json
from .context import MachineRuntime, ProjectBinding, default_machine_runtime_root
from .helpers import _active_machine_identity, _pid_start_time_ticks


@dataclass(frozen=True, slots=True)
class ProjectRegistration:
    """Result of an idempotent current-generation project registration."""

    binding: ProjectBinding
    is_added: bool
    is_adopted: bool = False

    @property
    def message(self) -> str:
        """Return the stable operator-facing registration result."""
        if self.binding.enabled:
            return (
                "Project registration checks passed; new task admission is enabled."
                if self.is_added
                else "Project is already registered and remains enabled. Registration checks passed."
            )
        return "Project is already registered and remains disabled. Registration checks passed; new task admission is disabled."


def register_project(
    runtime: MachineRuntime | str | Path | None,
    shared_root: str | Path,
    machine_name: str,
    *,
    adopt_existing: bool = False,
) -> ProjectRegistration:
    """Register a current-generation project with the machine agent.

    Args:
        runtime: Machine runtime instance or root.
        shared_root: Initialized project control root.
        machine_name: Project-local machine name.

    Returns:
        The persisted binding and whether this call created it.

    Raises:
        ValueError: If the project still requires explicit legacy migration.
    """
    machine_runtime = runtime if isinstance(runtime, MachineRuntime) else MachineRuntime(runtime)
    try:
        cfg = load_root_config(shared_root, machine_name, require_initialized=True)
    except RuntimeError as exc:
        from qqtools.version import __version__

        raise RuntimeError(
            f"qexp registration preflight failed for installed qqtools {__version__}: {exc} "
            "Install a supported qqtools version before registering this project."
        ) from exc
    if load_machine_record(cfg) is not None and is_legacy_agent_project(cfg):
        raise ValueError("legacy project metadata requires 'qexp agent migrate-project'.")
    binding, is_added = machine_runtime.ensure_binding(shared_root, machine_name, adopt_existing=adopt_existing)
    return ProjectRegistration(binding, is_added, adopt_existing and is_added)


def _legacy_pid_matches(cfg: RootConfig, pid: int) -> bool:
    """Verify a legacy agent PID before signalling it during migration."""
    try:
        argv = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    values = {item.decode("utf-8", errors="replace") for item in argv if item}
    return (
        "qqtools.plugins.qexp.agent_process" in values
        and str(cfg.shared_root) in values
        and cfg.machine_name in values
        and str(cfg.runtime_root) in values
    )


def _stop_verified_legacy_agent(cfg: RootConfig, *, timeout: float = 5.0) -> int | None:
    status = get_agent_status(cfg)
    pid = status.get("pid")
    if not isinstance(pid, int) or not status.get("is_running"):
        runtime_pid_path(cfg).unlink(missing_ok=True)
        return None
    if not _legacy_pid_matches(cfg, pid):
        raise RuntimeError("legacy agent PID cannot be verified; refusing to signal it.")
    start_ticks = _pid_start_time_ticks(pid)
    if start_ticks is None:
        raise RuntimeError("legacy agent process identity cannot be verified; refusing to signal it.")
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while _pid_start_time_ticks(pid) == start_ticks and time.monotonic() < deadline:
        time.sleep(0.05)
    if _pid_start_time_ticks(pid) == start_ticks:
        raise TimeoutError(f"legacy agent {pid} did not stop within {timeout:g} seconds.")
    runtime_pid_path(cfg).unlink(missing_ok=True)
    return pid


def _migration_record(
    cfg: RootConfig, *, state: str, prepared_at: str, detail: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build the durable, restartable handoff record for one legacy project."""
    value: dict[str, Any] = {
        "legacy_runtime_root": str(cfg.runtime_root),
        "project_id": None,
        "shared_root": str(cfg.shared_root),
        "machine_name": cfg.machine_name,
        "state": state,
        "prepared_at": prepared_at,
        "updated_at": utc_now(),
    }
    if detail is not None:
        value["detail"] = detail
    return {"migration": value}


def _save_migration_state(
    runtime: MachineRuntime,
    binding: ProjectBinding,
    cfg: RootConfig,
    *,
    state: str,
    prepared_at: str,
    detail: dict[str, Any] | None = None,
) -> None:
    value = _migration_record(cfg, state=state, prepared_at=prepared_at, detail=detail)
    value["migration"]["project_id"] = binding.project_id
    atomic_replace(runtime.migration_path(binding.project_id), value)


def _import_legacy_reservations(runtime: MachineRuntime, binding: ProjectBinding, cfg: RootConfig) -> None:
    """Move legacy reservations without overwriting a possibly unrelated machine record."""
    source_paths = local_paths(cfg.runtime_root)
    source_lock = source_paths["locks"] / "gpu-reservations.lock"
    if source_lock.resolve() == runtime.paths["reservation_lock"].resolve():
        raise RuntimeError("legacy and machine reservation roots must be different during migration.")
    with exclusive(source_lock):
        records: list[tuple[Path, Path, dict[str, Any]]] = []
        for name in ("active", "provisional"):
            for path in iter_json(source_paths[name]):
                value = read_json(path)
                reservation = value.get("reservation", {})
                reservation.update(
                    {
                        "project_id": binding.project_id,
                        "shared_root": str(cfg.shared_root),
                        "machine_name": cfg.machine_name,
                    }
                )
                records.append((path, runtime.paths[name] / path.name, value))
        with exclusive(runtime.paths["reservation_lock"]):
            imported_ids = {destination.name for _source, destination, _value in records}
            imported_gpus = {
                gpu_id
                for _source, _destination, value in records
                for gpu_id in value.get("reservation", {}).get("gpu_ids", [])
            }
            occupied_gpus = {
                gpu_id
                for name in ("active", "provisional")
                for path in iter_json(runtime.paths[name])
                if path.name not in imported_ids
                for gpu_id in read_json(path).get("reservation", {}).get("gpu_ids", [])
            }
            if imported_gpus.intersection(occupied_gpus):
                raise RuntimeError(
                    "legacy reservation GPUs conflict during migration; "
                    "the project remains disabled and no reservation was released."
                )
            for _source, destination, value in records:
                if destination.exists() and read_json(destination) != value:
                    raise RuntimeError(
                        f"legacy reservation ID conflicts during migration: {destination.stem}; "
                        "the project remains disabled and no reservation was released."
                    )
            for _source, destination, value in records:
                if not destination.exists():
                    atomic_replace(destination, value)
            for source, _destination, _value in records:
                source.unlink(missing_ok=True)


def migrate_project(runtime: MachineRuntime | str | Path | None, cfg: RootConfig) -> ProjectBinding:
    """Move one legacy project into the unique machine-agent runtime."""
    machine_runtime = runtime if isinstance(runtime, MachineRuntime) else MachineRuntime(runtime)
    machine_runtime.ensure_layout()
    existing = machine_runtime.matching_binding(cfg)
    if existing is None:
        if not is_legacy_agent_project(cfg):
            raise ValueError("project already uses the machine-agent runtime; use 'qexp agent add-project'.")
        try:
            binding = machine_runtime.add_binding(
                cfg.shared_root,
                cfg.machine_name,
                enabled=False,
                adopt_existing=True,
            )
        except ValueError as exc:
            # A bootstrap runtime may have registered this legacy project before
            # migration selected the machine runtime used today.  If that owner
            # has no verified live agent, fence its stale registration and retry
            # adoption.  A live owner remains a competing authority and is
            # rejected by the normal registration guard.
            if "active competing authority" not in str(exc):
                raise
            registration = load_machine_registration(cfg).get("registration", {})
            owner_root = registration.get("runtime_root") if isinstance(registration, dict) else None
            generation = registration.get("generation") if isinstance(registration, dict) else None
            if not isinstance(owner_root, str) or not isinstance(generation, str):
                raise
            owner_runtime = MachineRuntime(owner_root)
            if _active_machine_identity(owner_runtime) is not None:
                raise
            with machine_runtime._registration_guard(cfg):
                raw_current = load_machine_registration(cfg)
                current = raw_current.get("registration") if isinstance(raw_current, dict) else None
                if not isinstance(current, dict) or current.get("generation") != generation:
                    raise
                current = dict(current)
                current.update(
                    {
                        "state": "superseded",
                        "superseded_by_generation": uuid.uuid4().hex,
                        "updated_at": utc_now(),
                    }
                )
                save_machine_registration(cfg, {"registration": current})
            binding = machine_runtime.add_binding(
                cfg.shared_root,
                cfg.machine_name,
                enabled=False,
                adopt_existing=True,
            )
        prepared_at = utc_now()
        _save_migration_state(machine_runtime, binding, cfg, state="prepared", prepared_at=prepared_at)
    else:
        binding = existing
        migration_path = machine_runtime.migration_path(binding.project_id)
        if not migration_path.exists():
            if is_legacy_agent_project(cfg):
                raise RuntimeError("legacy project has a registry binding but no migration record.")
            return binding
        migration = read_json(migration_path).get("migration", {})
        prepared_at = migration.get("prepared_at")
        if not isinstance(prepared_at, str):
            raise RuntimeError("migration record is malformed: prepared_at is missing.")
        if migration.get("legacy_runtime_root") != str(cfg.runtime_root):
            raise RuntimeError("migration record does not match this project's legacy runtime root.")
        if migration.get("state") == "active":
            return binding

    with machine_runtime.migration_guard():
        try:
            _stop_verified_legacy_agent(cfg)
            _save_migration_state(machine_runtime, binding, cfg, state="legacy_agent_stopped", prepared_at=prepared_at)
            _import_legacy_reservations(machine_runtime, binding, cfg)
            _save_migration_state(
                machine_runtime,
                binding,
                cfg,
                state="reservations_imported",
                prepared_at=prepared_at,
            )
            machine_runtime.import_legacy_evidence(binding)
            save_machine_config(cfg, agent_mode=None)
            binding = machine_runtime.set_enabled(binding.project_id, True)
            _save_migration_state(machine_runtime, binding, cfg, state="active", prepared_at=prepared_at)
        except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
            if binding.enabled:
                binding = machine_runtime.set_enabled(binding.project_id, False)
            _save_migration_state(
                machine_runtime,
                binding,
                cfg,
                state="blocked",
                prepared_at=prepared_at,
                detail={"error": str(exc)},
            )
            raise

    return binding


def unregister_project(runtime: MachineRuntime | str | Path | None, identifier: str | Path) -> ProjectBinding:
    return (runtime if isinstance(runtime, MachineRuntime) else MachineRuntime(runtime)).remove_binding(identifier)


def set_project_enabled(
    runtime: MachineRuntime | str | Path | None, identifier: str | Path, enabled: bool
) -> ProjectBinding:
    return (runtime if isinstance(runtime, MachineRuntime) else MachineRuntime(runtime)).set_enabled(
        identifier, enabled
    )


def _enable_command(machine_runtime: MachineRuntime, binding: ProjectBinding) -> str:
    """Build a copyable command using the actual runtime selected by the caller."""
    project = shlex.quote(binding.project_id)
    if machine_runtime.root == default_machine_runtime_root():
        return f"qexp agent enable-project {project}"
    return f"qexp --machine-runtime-root {shlex.quote(str(machine_runtime.root))} agent enable-project {project}"


def enable_project(runtime: MachineRuntime | str | Path | None, identifier: str | Path) -> ProjectBinding:
    """Revalidate a binding's current generation and enable new admission."""
    machine_runtime = runtime if isinstance(runtime, MachineRuntime) else MachineRuntime(runtime)
    with machine_runtime.registry_guard():
        revision, bindings = machine_runtime.load_registry()
        current = machine_runtime._find_binding(bindings, identifier)
        if not machine_runtime.binding_write_eligible(current, renew=True):
            status = machine_runtime.registration_status(current)
            raise RuntimeError(
                f"cannot enable project {current.project_id!r}: registration is "
                f"{status.get('state', 'unavailable')}; resolve registration authority first."
            )
        updated = replace(current, enabled=True)
        machine_runtime._save_registry(revision + 1, [updated if item == current else item for item in bindings])
    return updated
