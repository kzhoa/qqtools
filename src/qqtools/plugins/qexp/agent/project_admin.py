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
from ..runtime.authority_scan import is_path_present, iter_evidence_files, validate_evidence_path
from ..runtime.locks import exclusive
from ..runtime.paths import local_paths
from ..runtime.records import utc_now, validate_identifier
from ..runtime.responsibility_store import DurableIO
from ..runtime.store import atomic_replace, read_json
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
        if self.is_adopted:
            state = "enabled" if self.binding.enabled else "disabled"
            admission = "enabled" if self.binding.enabled else "disabled"
            return (
                f"Project registration ownership was adopted and remains {state}. "
                f"Registration checks passed; new task admission is {admission}."
            )
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
    enabled: bool = True,
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
    previous = machine_runtime.matching_binding(cfg)
    binding, is_added = machine_runtime.ensure_binding(
        shared_root,
        machine_name,
        enabled=enabled,
        adopt_existing=adopt_existing,
    )
    is_generation_replaced = bool(
        previous is not None and previous.registration_generation != binding.registration_generation
    )
    return ProjectRegistration(
        binding,
        is_added,
        adopt_existing and (is_added or is_generation_replaced),
    )


def adoption_warning_generation(runtime: MachineRuntime, cfg: RootConfig) -> str | None:
    """Return the generation that explicit adoption would replace, when mutation is possible."""
    raw = load_machine_registration(cfg)
    registration = raw.get("registration") if isinstance(raw, dict) else None
    if not isinstance(registration, dict):
        return None
    generation = registration.get("generation")
    if not isinstance(generation, str) or not generation:
        return None
    current = runtime.matching_binding(cfg)
    is_same_owner = bool(
        current is not None
        and current.registration_generation == generation
        and current.runtime_instance_id == runtime.instance_id
        and registration.get("runtime_instance_id") == runtime.instance_id
        and registration.get("runtime_root") == str(runtime.root)
    )
    if is_same_owner or runtime._registration_state(registration) == "eligible":
        return None
    return generation


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
    """Move CPU and GPU occupancy without changing machine capacity policy."""
    if cfg.runtime_root.resolve() == runtime.root.resolve():
        raise RuntimeError("legacy and machine reservation roots must be different during migration.")
    # The independent capacity domains must never nest their reservation locks.
    for names, lock_name, released in (
        (("active", "provisional"), "gpu-reservations.lock", "released"),
        (("cpu_active", "cpu_provisional"), "cpu-lane.lock", "cpu_released"),
    ):
        _import_reservation_lane(runtime, binding, cfg, names, lock_name, released)


def _import_reservation_lane(
    runtime: MachineRuntime,
    binding: ProjectBinding,
    cfg: RootConfig,
    names: tuple[str, str],
    lock_name: str,
    released: str,
) -> None:
    source_paths = local_paths(cfg.runtime_root)
    io = DurableIO()
    with exclusive(source_paths["locks"] / lock_name):
        records: dict[Path, tuple[Path, dict[str, Any]]] = {}
        source_ids = set()
        for name in names:
            for path in iter_evidence_files(source_paths[name], recursive=True):
                if path.parent != source_paths[name]:
                    raise RuntimeError(f"legacy reservation has an unexpected nested path: {path}")
                if path.name in source_ids:
                    raise RuntimeError(f"legacy reservation ID occurs in multiple phases: {path}")
                source_ids.add(path.name)
                if not validate_evidence_path(path, cfg.runtime_root):
                    continue
                value = read_json(path)
                reservation = value.get("reservation")
                if not isinstance(reservation, dict) or reservation.get("reservation_id") != path.stem:
                    raise RuntimeError(f"legacy reservation identity is malformed: {path}")
                validate_identifier(reservation.get("task_id"), "task_id")
                if (
                    reservation.get("project_id") not in (None, binding.project_id)
                    or reservation.get("shared_root") not in (None, str(cfg.shared_root))
                    or reservation.get("machine_name") not in (None, cfg.machine_name)
                ):
                    raise RuntimeError(f"legacy reservation belongs to another project or machine: {path}")
                if reservation.get("state") != ("active" if name == names[0] else "provisional"):
                    raise RuntimeError(f"legacy reservation state does not match its directory: {path}")
                if released == "cpu_released":
                    if type(reservation.get("cpu_slots")) is not int or reservation["cpu_slots"] < 1:
                        raise RuntimeError(f"legacy CPU reservation has invalid slots: {path}")
                elif not isinstance(reservation.get("gpu_ids"), list) or any(
                    type(gpu) is not int for gpu in reservation["gpu_ids"]
                ):
                    raise RuntimeError(f"legacy GPU reservation has invalid devices: {path}")
                reservation.update(
                    project_id=binding.project_id, shared_root=str(cfg.shared_root), machine_name=cfg.machine_name
                )
                records[runtime.paths[name] / path.name] = path, value
        with exclusive(runtime.paths["locks"] / lock_name):
            imported_ids = {path.name for path in records}
            occupied_gpus = set()
            for name in names:
                for path in iter_evidence_files(runtime.paths[name], recursive=True):
                    if path.parent != runtime.paths[name]:
                        raise RuntimeError(f"machine reservation has an unexpected nested path: {path}")
                    if not validate_evidence_path(path, runtime.root):
                        continue
                    value = read_json(path)
                    if path.name in imported_ids:
                        if path not in records or value != records[path][1]:
                            raise RuntimeError(f"legacy reservation ID conflicts during migration: {path}")
                    else:
                        reservation = value.get("reservation")
                        if not isinstance(reservation, dict):
                            raise RuntimeError(f"machine reservation is malformed: {path}")
                        reservation_id = validate_identifier(reservation.get("reservation_id"), "reservation_id")
                        project_id = reservation.get("project_id")
                        is_foreign = isinstance(project_id, str) and project_id != binding.project_id
                        if is_foreign:
                            validate_identifier(project_id, "project_id")
                        if (reservation_id != path.stem and not is_foreign) or reservation.get("state") != (
                            "active" if name == names[0] else "provisional"
                        ):
                            raise RuntimeError(f"machine reservation is malformed: {path}")
                        validate_identifier(reservation.get("task_id"), "task_id")
                        if released == "cpu_released" and (
                            type(reservation.get("cpu_slots")) is not int or reservation["cpu_slots"] < 1
                        ):
                            raise RuntimeError(f"machine CPU reservation has invalid slots: {path}")
                        if released == "released":
                            gpus = reservation.get("gpu_ids")
                            if not isinstance(gpus, list) or any(type(gpu) is not int for gpu in gpus):
                                raise RuntimeError(f"machine reservation has invalid devices: {path}")
                            occupied_gpus.update(gpus)
            durable_targets = set()
            retired_targets = set()
            copies = []
            for destination, (source, value) in records.items():
                validate_evidence_path(destination, runtime.root)
                released_path = runtime.paths[released] / destination.name
                if validate_evidence_path(released_path, runtime.root):
                    retained = read_json(released_path)
                    receipt = retained.get("reservation")
                    if not isinstance(receipt, dict):
                        raise RuntimeError(f"legacy reservation release conflicts during migration: {released_path}")
                    expected = {
                        **value,
                        "reservation": {
                            **value["reservation"],
                            "state": "released",
                            "released_at": receipt.get("released_at"),
                            "release_reason": receipt.get("release_reason"),
                        },
                    }
                    if retained != expected or not isinstance(receipt.get("released_at"), str):
                        raise RuntimeError(f"legacy reservation release conflicts during migration: {released_path}")
                    durable_targets.add(released_path)
                    # Release may have committed before its occupancy unlink.
                    # The earlier destination check proved this is the exact copy.
                    retired_targets.add(destination)
                else:
                    copies.append((destination, value))
                    durable_targets.add(destination)
            imported_gpus = {gpu for _destination, value in copies for gpu in value["reservation"].get("gpu_ids", [])}
            if imported_gpus.intersection(occupied_gpus):
                raise RuntimeError(
                    "legacy reservation GPUs conflict during migration; the project remains disabled and no reservation was released."
                )
            # Validate every conflict before copying or deleting any source in this lane.
            for destination, value in copies:
                if not is_path_present(destination):
                    atomic_replace(destination, value)
            directories = {parent for target in durable_targets for parent in target.resolve().parents}
            for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
                io.sync_directory(directory, "migration_reservation_destination")
            for destination in retired_targets:
                io.delete(destination, should_sync_directory=False)
            for directory in {target.parent for target in retired_targets}:
                io.sync_directory(directory, "migration_reservation_retired")
            for source, _value in records.values():
                io.delete(source, should_sync_directory=False)
            # Retry also syncs an empty source after an interrupted unlink barrier.
            for name in names:
                if is_path_present(source_paths[name]):
                    io.sync_directory(source_paths[name], "migration_reservation_source")


def migrate_project(runtime: MachineRuntime | str | Path | None, cfg: RootConfig) -> ProjectBinding:
    """Move one legacy project into the unique machine-agent runtime."""
    machine_runtime = runtime if isinstance(runtime, MachineRuntime) else MachineRuntime(runtime)
    machine_runtime.ensure_layout()
    existing = machine_runtime.matching_binding(cfg)
    if existing is None:
        if not is_legacy_agent_project(cfg):
            raise ValueError("project already uses the machine-agent runtime; use 'qexp project register <PATH>'.")
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
            # The rejected binding attempt deliberately retains its transaction so the
            # next registration mutation can restore the exact pre-attempt state. Complete
            # that rollback before fencing the stale owner; otherwise the retry below would
            # restore the active registration after we supersede it.
            with machine_runtime.registry_guard():
                machine_runtime._rollback_registration_transaction()
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
        return f"qexp project enable {project}"
    return f"qexp --machine-runtime-root {shlex.quote(str(machine_runtime.root))} project enable {project}"


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
