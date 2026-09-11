"""Machine-scoped qexp agent for fair dispatch across registered projects."""

from __future__ import annotations

import os
import shlex
import signal
import threading
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, ContextManager

from ..legacy_agent import _visible_gpus, get_agent_status
from ..authority import AuthoritySupervisor
from ..config_types import RootConfig
from ..executor import Executor
from ..layout import load_machine_record, load_root_config, machine_state_path, runtime_pid_path
from ..machine_config import is_legacy_agent_project, load_machine_policy, save_machine_config
from ..machine_dispatch_plan import (
    MachineDispatchSnapshot,
    PrimaryCandidateObservation,
    PrimaryProbeRouteState,
    begin_primary_probe_route,
    build_machine_dispatch_plan,
    evaluate_primary_candidate,
    finish_primary_probe_route,
    order_dispatch_project_ids,
    reduce_dispatch_cursor,
)
from .context import MachineRuntime, ProjectBinding, default_machine_runtime_root
from .helpers import _active_machine_identity, _machine_is_true_idle, _read_pid, _pid_start_time_ticks, _publish_project_snapshots, _publish_process_status, _consume_first_registered_binding
from . import helpers as _helpers
from .control_plane import _MachineControlPlane
from . import dispatch_loop as _dispatch
from .dispatch_loop import dispatch_machine_cycle
from .project_admin import migrate_project, _stop_verified_legacy_agent
from ..machine_state import publish_machine_snapshots, publish_machine_stop_snapshot
from ..project_maintenance import maintain_project, reconcile_reservation
from ..runtime.locks import exclusive
from ..runtime.paths import local_paths, shared_paths
from ..runtime.ready import (
    ReadyProbeBudgetExhausted,
    advance_ready_index_build,
    classify_ready_marker,
    peek_primary_ready_marker,
    read_ready_index_state,
    ready_index_route_revision,
)
from ..runtime.ready.group_members import is_group_ready_member_projection_usable
from ..runtime.records import TaskSpec, normalize_group_record, utc_now
from ..runtime.resources.cpu_lane import cpu_reservation_snapshot
from ..runtime.resources.reservations import (
    ReservationIdentity,
    ReservationSnapshot,
    reconcile_snapshot,
    reservation_snapshot,
)
from ..runtime.store import atomic_replace, iter_json, read_json
from ..runtime.upgrade.machine import MachineUpgradeWorker, discover_registered_upgrades, inspect_registered_upgrades
from ..runtime.work_budget import (
    DIAGNOSTIC_PUBLISH_INTERVAL_SECONDS,
    AdaptiveBatchSizer,
    RuntimeDiagnostics,
    SliceBudget,
    WorkBudgetPolicy,
    activate_diagnostics,
    diagnostic_increment,
    diagnostic_span,
)
from ..scheduler import (
    _BorrowAdmissionGrant,
    _BorrowAdmissionRevision,
    _eligible,
    fail_attempt,
    resume_starting_attempt,
    run_dispatch_cycle,
)


def get_machine_agent_status(
    runtime: MachineRuntime | str | Path | None = None, *, probe_local_pid: bool = True
) -> dict[str, Any]:
    """Return machine-agent process and project-registry status."""
    machine_runtime = runtime if isinstance(runtime, MachineRuntime) else MachineRuntime(runtime)
    identity = _active_machine_identity(machine_runtime)
    pid = identity[0] if identity is not None else _read_pid(machine_runtime)
    running = bool(identity or (pid and not probe_local_pid))
    revision, bindings = machine_runtime.load_registry()
    upgrade = inspect_registered_upgrades(machine_runtime)
    waiting_for_first_registration = False
    if running:
        try:
            status = read_json(machine_runtime.paths["agent"] / "status.json").get("machine_agent", {})
            waiting_for_first_registration = bool(status.get("waiting_for_first_registration"))
        except (OSError, TypeError, ValueError):
            waiting_for_first_registration = False
    projects = []
    for binding in bindings:
        try:
            eligibility = machine_runtime.registration_status(binding)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            eligibility = {"state": "invalid", "write_eligible": False, "error": str(exc)}
        project = {
            **binding.to_dict(),
            "state": machine_runtime.binding_state(binding),
            "eligibility": eligibility,
            "write_eligible": eligibility.get("write_eligible", False),
            "upgrade": next(
                (
                    project.get("upgrade", {})
                    for project in upgrade["projects"]
                    if project.get("project_id") == binding.project_id
                ),
                {},
            ),
        }
        if eligibility.get("state") == "superseded":
            replacement_name = f"{binding.machine_name}-replacement"
            project["blocker"] = (
                f"registration generation {eligibility.get('generation')!r} superseded this environment; "
                "explicit takeover is required before this binding can dispatch or publish authoritative state."
            )
            project["recovery_command"] = (
                f"qexp --shared-root {shlex.quote(str(binding.shared_root))} "
                f"--machine {shlex.quote(replacement_name)} "
                f"--machine-runtime-root {shlex.quote(str(machine_runtime.root))} agent add-project"
            )
            project["recovery_note"] = (
                f"The example logical name {replacement_name!r} is illustrative; verify its availability before use."
            )
        projects.append(project)
    return {
        "machine_runtime_root": str(machine_runtime.root),
        "agent_state": "active" if running else "stopped",
        "pid": pid,
        "is_running": running,
        "waiting_for_first_registration": waiting_for_first_registration if running else False,
        "registry_revision": revision,
        "projects": projects,
        "upgrade": upgrade,
    }


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
    control_plane: _MachineControlPlane | None = None
    scheduler_wakeup = threading.Event()
    stop = False
    stop_reason: str | None = None
    has_consumed_binding = False
    idle_since: float | None = None
    upgrade_worker: MachineUpgradeWorker | None = None

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop, stop_reason
        stop = True
        stop_reason = "stopped_by_signal"
        scheduler_wakeup.set()

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
            _publish_process_status(
                machine_runtime,
                instance_id=instance_id,
                pid=os.getpid(),
                start_ticks=start_ticks,
                waiting_for_first_registration=True,
            )
            is_status_published = True
            control_plane = _MachineControlPlane(
                machine_runtime,
                instance_id=instance_id,
                loop_interval=loop_interval,
                started_at=started_at,
                available_gpus=available_gpus,
                scheduler_wakeup=scheduler_wakeup,
            )
            control_plane.start()
            while not stop:
                scheduler_wakeup.clear()
                if stop:
                    break
                try:
                    discovery = discover_registered_upgrades(machine_runtime)
                    if upgrade_worker is None or not upgrade_worker.is_alive:
                        if (
                            discovery.get("runnable_project_ids")
                            and time.monotonic() >= machine_runtime.upgrade_next_pass_at
                        ):
                            upgrade_worker = MachineUpgradeWorker(machine_runtime)
                            upgrade_worker.start()
                except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                    pass
                try:
                    with machine_runtime.migration_guard() as is_migration_clear:
                        if is_migration_clear:
                            _dispatch.dispatch_machine_cycle_locked(
                                machine_runtime,
                                available_gpus=available_gpus,
                                executor=executor,
                                instance_id=instance_id,
                                heartbeat_interval_seconds=loop_interval,
                                started_at=started_at,
                                supervise=False,
                                publish_snapshots=False,
                            )
                            if machine_runtime.last_cycle_consumed_binding or _consume_first_registered_binding(
                                machine_runtime
                            ):
                                has_consumed_binding = True
                except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                    # A transient shared-root failure must not stop supervision of other projects.
                    pass
                try:
                    if has_consumed_binding:
                        _publish_process_status(
                            machine_runtime,
                            instance_id=instance_id,
                            pid=os.getpid(),
                            start_ticks=start_ticks,
                            waiting_for_first_registration=False,
                        )
                except OSError:
                    pass
                if _machine_is_true_idle(machine_runtime, has_consumed_binding=has_consumed_binding):
                    if idle_since is None:
                        idle_since = time.monotonic()
                    elif time.monotonic() - idle_since >= loop_interval:
                        stop = True
                        stop_reason = "idle"
                        continue
                else:
                    idle_since = None
                scheduler_wakeup.wait(loop_interval)
        finally:
            if upgrade_worker is not None:
                upgrade_worker.stop()
            if control_plane is not None:
                control_plane.stop()
            try:
                is_active_identity = _active_machine_identity(machine_runtime) == (
                    os.getpid(),
                    instance_id,
                    start_ticks,
                )
                if is_active_identity:
                    try:
                        reservations = list(reservation_snapshot(machine_runtime.root).reservations)
                        reserved = sorted({gpu_id for item in reservations for gpu_id in item.get("gpu_ids", [])})
                    except (KeyError, OSError, ValueError):
                        reserved = []
                    try:
                        _, registered = machine_runtime.load_registry()
                    except (OSError, RuntimeError, ValueError):
                        registered = []
                    for binding in registered:
                        try:
                            cfg = _helpers._binding_config(machine_runtime, binding)
                            with machine_runtime.binding_write_guard(binding) as is_eligible:
                                if is_eligible:
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
                                        stop_reason=stop_reason or "stopped",
                                    )
                        except (OSError, RuntimeError, ValueError):
                            continue
                if is_pid_published:
                    pid_path.unlink(missing_ok=True)
                if is_status_published:
                    atomic_replace(
                        machine_runtime.paths["agent"] / "status.json",
                        {
                            "machine_agent": {
                                "instance_id": instance_id,
                                "pid": None,
                                "pid_start_time_ticks": start_ticks,
                                "state": "stopped",
                                "waiting_for_first_registration": False,
                            }
                        },
                    )
            finally:
                try:
                    if previous_int is not None:
                        signal.signal(signal.SIGINT, previous_int)
                finally:
                    if previous_term is not None:
                        signal.signal(signal.SIGTERM, previous_term)


def _start_machine_agent_locked(
    machine_runtime: MachineRuntime,
    *,
    available_gpus: list[int] | None = None,
    stdin=None,
    stdout=None,
    stderr=None,
):
    status = get_machine_agent_status(machine_runtime)
    if status["is_running"]:
        raise RuntimeError(f"machine agent is already running with pid {status['pid']}.")
    from .process import spawn_machine_agent_process

    return spawn_machine_agent_process(
        machine_runtime,
        available_gpus=available_gpus,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
    )


def start_machine_agent(
    runtime: MachineRuntime | str | Path | None = None,
    *,
    available_gpus: list[int] | None = None,
    stdin=None,
    stdout=None,
    stderr=None,
):
    """Spawn the unique persistent machine agent."""
    machine_runtime = runtime if isinstance(runtime, MachineRuntime) else MachineRuntime(runtime)
    with machine_runtime.agent_lifecycle_guard():
        return _start_machine_agent_locked(
            machine_runtime,
            available_gpus=available_gpus,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
        )


def ensure_machine_agent_started(
    runtime: MachineRuntime | str | Path | None = None,
    *,
    available_gpus: list[int] | None = None,
    stdin=None,
    stdout=None,
    stderr=None,
):
    """Return the running agent status, starting it atomically when absent."""
    machine_runtime = runtime if isinstance(runtime, MachineRuntime) else MachineRuntime(runtime)
    with machine_runtime.agent_lifecycle_guard():
        status = get_machine_agent_status(machine_runtime)
        if status["is_running"]:
            return None, status
        try:
            process = _start_machine_agent_locked(
                machine_runtime,
                available_gpus=available_gpus,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
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
    runtime: MachineRuntime | str | Path | None = None,
    *,
    available_gpus: list[int] | None = None,
    stdin=None,
    stdout=None,
    stderr=None,
):
    """Replace a running machine agent without treating it as a cold start."""
    machine_runtime = runtime if isinstance(runtime, MachineRuntime) else MachineRuntime(runtime)
    with machine_runtime.agent_lifecycle_guard():
        identity = _active_machine_identity(machine_runtime)
        previous_pid = identity[0] if identity is not None else None
        _stop_machine_agent_locked(machine_runtime, timeout=10.0)
        process = _start_machine_agent_locked(
            machine_runtime,
            available_gpus=available_gpus,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
        )
        process.previous_pid = previous_pid
        return process

# Public lifecycle entry point retained for internal test and CLI orchestration.
def dispatch_machine_cycle_locked(*args, **kwargs):
    return _dispatch.dispatch_machine_cycle_locked(*args, **kwargs)
