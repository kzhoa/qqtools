"""Machine-scoped qexp agent for fair dispatch across registered projects."""
from __future__ import annotations

import os
import signal
import threading
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .agent import _visible_gpus, get_agent_status
from .authority import AuthoritySupervisor
from .config_types import RootConfig
from .machine_state import publish_machine_snapshots, publish_machine_stop_snapshot
from .executor import Executor
from .machine_runtime import MachineRuntime, ProjectBinding
from .machine_config import is_legacy_agent_project, save_machine_config
from .layout import load_root_config, machine_state_path, runtime_pid_path
from .project_maintenance import maintain_project
from .runtime.paths import local_paths
from .runtime.records import TaskSpec, utc_now
from .runtime.reservations import active_reservations, release_expired_provisionals, reserved_gpu_ids
from .runtime.locks import exclusive
from .runtime.store import atomic_replace, iter_json, read_json
from .scheduler import run_dispatch_cycle


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _pid_start_time_ticks(pid: int | None) -> int | None:
    """Return Linux process start ticks, used to reject a reused PID."""
    if not pid:
        return None
    try:
        fields = (
            (Path("/proc") / str(pid) / "stat")
            .read_text(encoding="utf-8")
            .rsplit(")", 1)[1]
            .split()
        )
        if fields[0] == "Z":
            return None
        return int(fields[19])
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return None


def _read_pid(runtime: MachineRuntime) -> int | None:
    path = runtime.paths["pid"]
    try:
        return int(path.read_text(encoding="utf-8").strip()) if path.exists() else None
    except (OSError, ValueError):
        return None


def _active_machine_identity(runtime: MachineRuntime) -> tuple[int, str, int] | None:
    """Return a verified machine-agent identity, never trusting a bare PID file."""
    pid = _read_pid(runtime)
    status_path = runtime.paths["agent"] / "status.json"
    try:
        status = read_json(status_path).get("machine_agent", {})
    except (OSError, ValueError):
        return None
    instance_id = status.get("instance_id")
    start_ticks = status.get("pid_start_time_ticks")
    if (
        status.get("state") != "active"
        or status.get("pid") != pid
        or not isinstance(instance_id, str)
        or not isinstance(start_ticks, int)
        or _pid_start_time_ticks(pid) != start_ticks
    ):
        return None
    return pid, instance_id, start_ticks


def _binding_config(runtime: MachineRuntime, binding: ProjectBinding) -> RootConfig:
    """Build the binding's isolated local runtime configuration."""
    cfg = binding.root_config()
    return replace(cfg, runtime_root=runtime.project_paths(binding.project_id)["root"])


def _working_directory_reason(spec: TaskSpec) -> str | None:
    """Return the machine-local reason a TaskSpec cannot be started."""
    root = Path(spec.working_directory)
    if not root.exists():
        return "missing"
    if not root.is_dir():
        return "not_directory"
    if not os.access(root, os.R_OK | os.X_OK):
        return "not_readable_or_searchable"
    return None


def _ordered_bindings(runtime: MachineRuntime, bindings: list[ProjectBinding]) -> list[ProjectBinding]:
    """Return enabled bindings in stable cursor order without touching project roots."""
    enabled = sorted((binding for binding in bindings if binding.enabled), key=lambda item: item.project_id)
    if not enabled:
        return []
    cursor = runtime.load_cursor()
    if cursor is None:
        return enabled
    for index, binding in enumerate(enabled):
        if binding.project_id == cursor:
            return enabled[index:] + enabled[:index]
    return enabled


def _record_bad_task_spec(runtime: MachineRuntime, binding: ProjectBinding, task_id: str, spec: TaskSpec) -> None:
    reason = _working_directory_reason(spec)
    if reason is None:
        return
    atomic_replace(
        runtime.paths["diagnostics"] / f"bad-task-spec-{binding.project_id}-{task_id}.json",
        {"machine_diagnostic": {"kind": "bad_task_spec_working_directory", "project_id": binding.project_id,
                                "task_id": task_id, "path": spec.working_directory, "reason": reason}},
    )


def _publish_project_snapshots(
    readable: dict[str, RootConfig], *, instance_id: str, pid: int | None, visible: list[int],
    reservations: list[dict[str, Any]], heartbeat_interval_seconds: float = 5.0,
    started_at: str | None = None,
) -> None:
    """Publish each readable project's view of the shared machine reservation state."""
    reserved = sorted({gpu_id for item in reservations for gpu_id in item.get("gpu_ids", [])})
    started_at = started_at or utc_now()
    for project_id, cfg in readable.items():
        attempts = [item.get("attempt_id") for item in reservations if item.get("project_id") == project_id]
        agent_path = machine_state_path(cfg, "agent.json")
        idle_since_at = None
        if not reserved:
            idle_since_at = utc_now()
            if agent_path.exists():
                try:
                    previous = read_json(agent_path).get("agent", {})
                    if previous.get("instance_id") == instance_id and previous.get("observed_state") == "idle":
                        previous_idle_since = previous.get("idle_since_at")
                        if isinstance(previous_idle_since, str):
                            idle_since_at = previous_idle_since
                except (OSError, ValueError):
                    pass
        try:
            publish_machine_snapshots(
                cfg, instance_id=instance_id, pid=pid, agent_mode="machine",
                observed_state="active" if reserved else "idle",
                active_attempt_ids=[item for item in attempts if isinstance(item, str)],
                visible_gpu_ids=visible, reserved_gpu_ids=reserved,
                reservation_summaries=reservations,
                heartbeat_interval_seconds=heartbeat_interval_seconds, started_at=started_at,
                idle_since_at=idle_since_at,
            )
        except OSError:
            continue


def dispatch_machine_cycle_locked(
    runtime: MachineRuntime,
    *,
    available_gpus: list[int] | None = None,
    executor: Executor | None = None,
    instance_id: str = "machine-agent",
    heartbeat_interval_seconds: float = 5.0,
    started_at: str | None = None,
    supervisors: dict[str, AuthoritySupervisor] | None = None,
) -> list[dict[str, Any]]:
    """Supervise readable bindings and make at most one fair new claim."""
    executor = executor or Executor()
    _, registered = runtime.load_registry()
    ordered_enabled = _ordered_bindings(runtime, registered)
    supervised = [
        binding
        for binding in registered
        if binding.enabled or runtime.binding_state(binding) == "draining"
    ]
    if supervisors is not None:
        supervised_ids = {binding.project_id for binding in supervised}
        for project_id in set(supervisors) - supervised_ids:
            del supervisors[project_id]
    if not supervised:
        return []
    release_expired_provisionals(runtime.root)
    readable: dict[str, RootConfig] = {}
    dispatchable: dict[str, RootConfig] = {}
    results: list[dict[str, Any]] = []
    for binding in supervised:
        try:
            cfg = _binding_config(runtime, binding)
            readable[binding.project_id] = cfg
            runtime.drain_legacy_runner_evidence(binding)
            maintain_project(
                cfg,
                reservation_runtime_root=runtime.root,
                project_id=binding.project_id,
            )
            supervisor = None if supervisors is None else supervisors.get(binding.project_id)
            if supervisor is None:
                supervisor = AuthoritySupervisor(cfg, reservation_runtime_root=runtime.root)
                if supervisors is not None:
                    supervisors[binding.project_id] = supervisor
            supervisor.tick()
            dispatchable[binding.project_id] = cfg
        except (OSError, RuntimeError, ValueError) as exc:
            results.append({"project_id": binding.project_id, "launched": [], "status": "error", "error": str(exc)})
    visible = list(available_gpus) if available_gpus is not None else _visible_gpus(next(iter(readable.values()))) if readable else []
    successful_binding: ProjectBinding | None = None
    for binding in ordered_enabled:
        cfg = dispatchable.get(binding.project_id)
        if cfg is None:
            continue
        claimed = False
        claimed_task_ids: list[str] = []
        try:
            free = [gpu_id for gpu_id in visible if gpu_id not in reserved_gpu_ids(runtime.root)]
            launched = run_dispatch_cycle(
                cfg, available_gpus=free, executor=executor, reservation_runtime_root=runtime.root,
                project_id=binding.project_id, preflight=lambda spec: _working_directory_reason(spec) is None,
                preflight_rejected=lambda task: _record_bad_task_spec(runtime, binding, task.task_id, task.spec),
                claim_guard=lambda: runtime.enabled_claim_guard(binding), max_new_claims=1,
                on_claim=claimed_task_ids.append,
            )
            claimed = bool(claimed_task_ids)
            results.append({"project_id": binding.project_id, "launched": launched, "status": "dispatched"})
        except (OSError, RuntimeError, ValueError) as exc:
            results.append({"project_id": binding.project_id, "launched": [], "status": "error", "error": str(exc)})
        if claimed:
            successful_binding = binding
            break
    if ordered_enabled:
        start = ordered_enabled[0]
        if successful_binding is not None:
            index = ordered_enabled.index(successful_binding)
            runtime.save_cursor(ordered_enabled[(index + 1) % len(ordered_enabled)].project_id)
        else:
            runtime.save_cursor(ordered_enabled[(ordered_enabled.index(start) + 1) % len(ordered_enabled)].project_id)
    reservations = active_reservations(runtime.root)
    _publish_project_snapshots(
        readable, instance_id=instance_id, pid=_read_pid(runtime), visible=visible, reservations=reservations,
        heartbeat_interval_seconds=heartbeat_interval_seconds, started_at=started_at,
    )
    return results


def dispatch_machine_cycle(
    runtime: MachineRuntime,
    *,
    available_gpus: list[int] | None = None,
    executor: Executor | None = None,
) -> list[dict[str, Any]]:
    """Dispatch registered projects once in stable round-robin order."""
    with runtime.scheduler_authority(blocking=False) as acquired:
        if not acquired:
            return []
        with runtime.migration_guard() as is_migration_clear:
            if not is_migration_clear:
                return []
            return dispatch_machine_cycle_locked(
                runtime, available_gpus=available_gpus, executor=executor
            )


def get_machine_agent_status(runtime: MachineRuntime | str | Path | None = None, *, probe_local_pid: bool = True) -> dict[str, Any]:
    """Return machine-agent process and project-registry status."""
    machine_runtime = runtime if isinstance(runtime, MachineRuntime) else MachineRuntime(runtime)
    identity = _active_machine_identity(machine_runtime)
    pid = identity[0] if identity is not None else _read_pid(machine_runtime)
    running = bool(identity or (pid and not probe_local_pid))
    revision, bindings = machine_runtime.load_registry()
    return {
        "machine_runtime_root": str(machine_runtime.root),
        "agent_state": "active" if running else "stopped",
        "pid": pid,
        "is_running": running,
        "registry_revision": revision,
        "projects": [
            {**binding.to_dict(), "state": machine_runtime.binding_state(binding)}
            for binding in bindings
        ],
    }


@dataclass(frozen=True, slots=True)
class ProjectRegistration:
    """Result of an idempotent current-generation project registration."""

    binding: ProjectBinding
    is_added: bool


def register_project(
    runtime: MachineRuntime | str | Path | None,
    shared_root: str | Path,
    machine_name: str,
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
    cfg = load_root_config(shared_root, machine_name, require_initialized=True)
    if is_legacy_agent_project(cfg):
        raise ValueError("legacy project metadata requires 'qexp agent migrate-project'.")
    binding, is_added = machine_runtime.ensure_binding(shared_root, machine_name)
    return ProjectRegistration(binding, is_added)


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
        raise RuntimeError(
            "legacy agent process identity cannot be verified; refusing to signal it."
        )
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


def _import_legacy_reservations(
    runtime: MachineRuntime, binding: ProjectBinding, cfg: RootConfig
) -> None:
    """Move legacy reservations without overwriting a possibly unrelated machine record."""
    source_paths = local_paths(cfg.runtime_root)
    source_lock = source_paths["locks"] / "gpu-reservations.lock"
    if source_lock.resolve() == runtime.paths["reservation_lock"].resolve():
        raise RuntimeError(
            "legacy and machine reservation roots must be different during migration."
        )
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
        binding = machine_runtime.add_binding(cfg.shared_root, cfg.machine_name, enabled=False)
        prepared_at = utc_now()
        _save_migration_state(
            machine_runtime, binding, cfg, state="prepared", prepared_at=prepared_at
        )
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
            _save_migration_state(
                machine_runtime, binding, cfg, state="legacy_agent_stopped", prepared_at=prepared_at
            )
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
            _save_migration_state(
                machine_runtime, binding, cfg, state="active", prepared_at=prepared_at
            )
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


def set_project_enabled(runtime: MachineRuntime | str | Path | None, identifier: str | Path, enabled: bool) -> ProjectBinding:
    return (runtime if isinstance(runtime, MachineRuntime) else MachineRuntime(runtime)).set_enabled(identifier, enabled)


def run_machine_agent_loop(
    runtime: MachineRuntime | str | Path | None = None,
    *,
    loop_interval: float = 5.0,
    available_gpus: list[int] | None = None,
    executor: Executor | None = None,
) -> None:
    """Run the persistent machine agent until SIGTERM or SIGINT."""
    if loop_interval <= 0:
        raise ValueError("loop_interval must be positive.")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("machine agent loop must run in the process main thread.")
    machine_runtime = runtime if isinstance(runtime, MachineRuntime) else MachineRuntime(runtime)
    machine_runtime.ensure_layout()
    pid_path = machine_runtime.paths["pid"]
    current_identity = _active_machine_identity(machine_runtime)
    if current_identity is not None and current_identity[0] != os.getpid():
        raise RuntimeError(f"machine agent is already running with pid {current_identity[0]}.")
    instance_id = uuid.uuid4().hex
    started_at = utc_now()
    supervisors: dict[str, AuthoritySupervisor] = {}
    stop = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    with machine_runtime.scheduler_authority(blocking=False) as acquired:
        if not acquired:
            raise RuntimeError("machine scheduler authority is already held.")
        start_ticks = _pid_start_time_ticks(os.getpid())
        if start_ticks is None:
            raise RuntimeError("could not determine machine agent process identity.")
        previous_term = None
        previous_int = None
        is_pid_published = False
        is_status_published = False
        try:
            previous_term = signal.signal(signal.SIGTERM, request_stop)
            previous_int = signal.signal(signal.SIGINT, request_stop)
            pid_path.write_text(str(os.getpid()), encoding="utf-8")
            is_pid_published = True
            atomic_replace(machine_runtime.paths["agent"] / "status.json", {
                "machine_agent": {
                    "instance_id": instance_id, "pid": os.getpid(),
                    "pid_start_time_ticks": start_ticks, "state": "active",
                }
            })
            is_status_published = True
            _, registered = machine_runtime.load_registry()
            for binding in registered:
                if not binding.enabled and machine_runtime.binding_state(binding) != "draining":
                    continue
                try:
                    supervisor = AuthoritySupervisor(
                        _binding_config(machine_runtime, binding),
                        reservation_runtime_root=machine_runtime.root,
                    )
                    supervisors[binding.project_id] = supervisor
                    supervisor.recover_startup()
                except (OSError, RuntimeError, ValueError):
                    continue
            while not stop:
                with machine_runtime.migration_guard() as is_migration_clear:
                    if is_migration_clear:
                        dispatch_machine_cycle_locked(
                            machine_runtime, available_gpus=available_gpus, executor=executor,
                            instance_id=instance_id, heartbeat_interval_seconds=loop_interval,
                            started_at=started_at,
                            supervisors=supervisors,
                        )
                time.sleep(loop_interval)
        finally:
            try:
                is_active_identity = _active_machine_identity(machine_runtime) == (
                    os.getpid(), instance_id, start_ticks
                )
                if is_active_identity:
                    try:
                        reservations = active_reservations(machine_runtime.root)
                        reserved = sorted({
                            gpu_id
                            for item in reservations
                            for gpu_id in item.get("gpu_ids", [])
                        })
                    except (KeyError, OSError, ValueError):
                        reserved = []
                    try:
                        _, registered = machine_runtime.load_registry()
                    except (OSError, RuntimeError, ValueError):
                        registered = []
                    for binding in registered:
                        try:
                            cfg = _binding_config(machine_runtime, binding)
                            publish_machine_stop_snapshot(
                                cfg,
                                instance_id=instance_id,
                                pid=None,
                                agent_mode="machine",
                                visible_gpu_ids=_visible_gpus(cfg),
                                reserved_gpu_ids=reserved,
                                heartbeat_interval_seconds=loop_interval,
                                started_at=started_at,
                                idle_since_at=None if reserved else utc_now(),
                                stop_reason="stopped_by_signal" if stop else "stopped",
                            )
                        except (OSError, RuntimeError, ValueError):
                            continue
                if is_pid_published:
                    pid_path.unlink(missing_ok=True)
                if is_status_published:
                    atomic_replace(machine_runtime.paths["agent"] / "status.json", {
                        "machine_agent": {
                            "instance_id": instance_id, "pid": None, "state": "stopped"
                        }
                    })
            finally:
                try:
                    if previous_int is not None:
                        signal.signal(signal.SIGINT, previous_int)
                finally:
                    if previous_term is not None:
                        signal.signal(signal.SIGTERM, previous_term)


def _start_machine_agent_locked(machine_runtime: MachineRuntime, *, stdin=None, stdout=None, stderr=None):
    status = get_machine_agent_status(machine_runtime)
    if status["is_running"]:
        raise RuntimeError(f"machine agent is already running with pid {status['pid']}.")
    from .machine_agent_process import spawn_machine_agent_process
    return spawn_machine_agent_process(machine_runtime, stdin=stdin, stdout=stdout, stderr=stderr)


def start_machine_agent(
        runtime: MachineRuntime | str | Path | None = None, *, stdin=None, stdout=None, stderr=None):
    """Spawn the unique persistent machine agent."""
    machine_runtime = runtime if isinstance(runtime, MachineRuntime) else MachineRuntime(runtime)
    with machine_runtime.agent_lifecycle_guard():
        return _start_machine_agent_locked(
            machine_runtime, stdin=stdin, stdout=stdout, stderr=stderr
        )


def ensure_machine_agent_started(
        runtime: MachineRuntime | str | Path | None = None, *, stdin=None, stdout=None, stderr=None):
    """Return the running agent status, starting it atomically when absent."""
    machine_runtime = runtime if isinstance(runtime, MachineRuntime) else MachineRuntime(runtime)
    with machine_runtime.agent_lifecycle_guard():
        status = get_machine_agent_status(machine_runtime)
        if status["is_running"]:
            return None, status
        try:
            process = _start_machine_agent_locked(
                machine_runtime, stdin=stdin, stdout=stdout, stderr=stderr
            )
        except RuntimeError:
            status = get_machine_agent_status(machine_runtime)
            if status["is_running"]:
                return None, status
            raise
        return process, {**get_machine_agent_status(machine_runtime), "pid": process.pid}


def _stop_machine_agent_locked(machine_runtime: MachineRuntime, *, timeout: float) -> bool:
    identity = _active_machine_identity(machine_runtime)
    if identity is None:
        machine_runtime.paths["pid"].unlink(missing_ok=True)
        return False
    pid, _instance_id, start_ticks = identity
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + timeout
    while _pid_start_time_ticks(pid) == start_ticks and time.monotonic() < deadline:
        time.sleep(0.05)
    if _pid_start_time_ticks(pid) == start_ticks:
        raise TimeoutError(f"machine agent {pid} did not stop within {timeout} seconds.")
    return True


def stop_machine_agent(runtime: MachineRuntime | str | Path | None = None, *, timeout: float = 10.0) -> bool:
    """Request a graceful machine-agent stop and wait for the process to exit."""
    machine_runtime = runtime if isinstance(runtime, MachineRuntime) else MachineRuntime(runtime)
    with machine_runtime.agent_lifecycle_guard():
        return _stop_machine_agent_locked(machine_runtime, timeout=timeout)


def restart_machine_agent(
        runtime: MachineRuntime | str | Path | None = None, *, stdin=None, stdout=None, stderr=None):
    """Replace a running machine agent without treating it as a cold start."""
    machine_runtime = runtime if isinstance(runtime, MachineRuntime) else MachineRuntime(runtime)
    with machine_runtime.agent_lifecycle_guard():
        _stop_machine_agent_locked(machine_runtime, timeout=10.0)
        return _start_machine_agent_locked(
            machine_runtime, stdin=stdin, stdout=stdout, stderr=stderr
        )
