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
from .project_maintenance import maintain_project, reconcile_reservation
from .runtime.paths import local_paths
from .runtime.records import TaskSpec, utc_now
from .runtime.reservations import (
    ReservationIdentity,
    ReservationSnapshot,
    active_reservations,
    reconcile_snapshot,
)
from .runtime.ready import advance_ready_index_build, read_ready_index_state
from .runtime.locks import exclusive
from .runtime.store import atomic_replace, iter_json, read_json
from .runtime.work_budget import (
    AdaptiveBatchSizer,
    DIAGNOSTIC_PUBLISH_INTERVAL_SECONDS,
    RuntimeDiagnostics,
    SliceBudget,
    activate_diagnostics,
    diagnostic_increment,
    diagnostic_span,
)
from .scheduler import fail_attempt, resume_starting_attempt, run_dispatch_cycle


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


def _reconcile_machine_reservations(
    runtime: MachineRuntime,
    readable: dict[str, RootConfig],
) -> ReservationSnapshot:
    """Reconcile a machine-wide snapshot without holding its lock during shared I/O."""
    with diagnostic_span("recovery.reservation_reconciliation"):
        snapshot = reconcile_snapshot(runtime.root)
        diagnostic_increment("recovery.reservations.snapshotted", len(snapshot.active))
        for reservation in snapshot.active:
            project_id = reservation.get("project_id")
            cfg = readable.get(project_id) if isinstance(project_id, str) else None
            if cfg is None:
                diagnostic_increment("recovery.reservations.isolated")
                continue
            try:
                action = reconcile_reservation(
                    cfg,
                    reservation,
                    reservation_runtime_root=runtime.root,
                )
            except (KeyError, OSError, RuntimeError, ValueError):
                diagnostic_increment("recovery.reservations.errors")
                diagnostic_increment("recovery.reservations.isolated")
                continue
            diagnostic_increment(f"recovery.reservations.{action}")
        trusted = reconcile_snapshot(runtime.root)
        diagnostic_increment("recovery.reservations.trusted", len(trusted.active))
        return trusted


def _recover_starting_reservations(
    runtime: MachineRuntime,
    readable: dict[str, RootConfig],
    reservations: tuple[dict[str, Any], ...],
    executor: Executor,
) -> dict[str, list[str]]:
    """Recover starting Attempts from exact active-reservation identities."""
    launched: dict[str, list[str]] = {}
    with diagnostic_span("recovery.starting_attempts"):
        for reservation in reservations:
            try:
                identity = ReservationIdentity.from_record(reservation)
            except ValueError:
                diagnostic_increment("recovery.starting.invalid_reservation")
                continue
            cfg = readable.get(identity.project_id or "")
            if cfg is None or identity.attempt_id is None or identity.fencing_token is None:
                diagnostic_increment("recovery.starting.ineligible_reservation")
                continue
            diagnostic_increment("recovery.starting.checked")
            try:
                attempt = resume_starting_attempt(
                    cfg,
                    identity.task_id,
                    reservation_runtime_root=runtime.root,
                    expected_reservation=identity,
                )
            except (KeyError, OSError, RuntimeError, ValueError):
                diagnostic_increment("recovery.starting.errors")
                continue
            if attempt is None:
                continue
            try:
                executor.launch_attempt(cfg, identity.task_id, attempt)
                launched.setdefault(identity.project_id or "", []).append(identity.task_id)
                diagnostic_increment("recovery.starting.launched")
            except Exception:
                diagnostic_increment("recovery.starting.launch_failed")
                try:
                    fail_attempt(
                        cfg,
                        identity.task_id,
                        attempt.attempt_id,
                        attempt.current_fencing_token,
                        "executor_launch_failed",
                        reservation_runtime_root=runtime.root,
                    )
                except (KeyError, OSError, RuntimeError, ValueError):
                    diagnostic_increment("recovery.starting.compensation_errors")
    return launched


class _MachineControlPlane:
    """Run deadline-sensitive heartbeats and authority renewal outside dispatch scans."""

    def __init__(
            self, runtime: MachineRuntime, *, instance_id: str, loop_interval: float,
            started_at: str, available_gpus: list[int] | None,
            scheduler_wakeup: threading.Event | None = None) -> None:
        self._runtime = runtime
        self._instance_id = instance_id
        self._loop_interval = loop_interval
        self._started_at = started_at
        self._visible_gpus = list(available_gpus) if available_gpus is not None else None
        self._scheduler_wakeup = scheduler_wakeup
        self._registry_revision: int | None = None
        self._stop_event = threading.Event()
        self._supervisors: dict[str, AuthoritySupervisor] = {}
        self._authority_thread = threading.Thread(
            target=self._run_authority_loop,
            name="qexp-machine-authority",
            daemon=True,
        )
        self._heartbeat_thread = threading.Thread(
            target=self._run_heartbeat_loop,
            name="qexp-machine-heartbeat",
            daemon=True,
        )

    def start(self) -> None:
        """Publish initial liveness before starting the control loops."""
        self._refresh_visible_gpus()
        self._publish_heartbeat()
        self._authority_thread.start()
        self._heartbeat_thread.start()

    def stop(self) -> None:
        """Stop control loops before publishing the terminal machine snapshot."""
        self._stop_event.set()
        for thread in (self._authority_thread, self._heartbeat_thread):
            if thread.is_alive():
                thread.join()

    def _supervised_bindings(self) -> list[ProjectBinding]:
        revision, registered = self._runtime.load_registry()
        if (
            self._registry_revision is not None
            and revision != self._registry_revision
            and self._scheduler_wakeup is not None
        ):
            self._scheduler_wakeup.set()
        self._registry_revision = revision
        return [
            binding
            for binding in registered
            if binding.enabled or self._runtime.binding_state(binding) == "draining"
        ]

    def _run_authority_cycle(self) -> float:
        authority_interval = self._loop_interval
        reserved_before = {
            gpu_id
            for item in active_reservations(self._runtime.root)
            for gpu_id in item.get("gpu_ids", [])
        }
        try:
            bindings = self._supervised_bindings()
        except (OSError, RuntimeError, ValueError):
            return authority_interval
        supervised_ids = {binding.project_id for binding in bindings}
        for project_id in set(self._supervisors) - supervised_ids:
            del self._supervisors[project_id]
        for binding in bindings:
            try:
                cfg = _binding_config(self._runtime, binding)
                supervisor = self._supervisors.get(binding.project_id)
                if supervisor is None:
                    supervisor = AuthoritySupervisor(cfg, reservation_runtime_root=self._runtime.root)
                    supervisor.recover_startup()
                    self._supervisors[binding.project_id] = supervisor
                supervisor.tick()
                authority_interval = min(authority_interval, supervisor.renewal_interval_seconds)
            except (OSError, RuntimeError, ValueError):
                continue
        reserved_after = {
            gpu_id
            for item in active_reservations(self._runtime.root)
            for gpu_id in item.get("gpu_ids", [])
        }
        if (
            reserved_after < reserved_before
            and self._scheduler_wakeup is not None
        ):
            self._scheduler_wakeup.set()
        return authority_interval

    def _refresh_visible_gpus(self) -> None:
        if self._visible_gpus is not None:
            return
        try:
            bindings = self._supervised_bindings()
            if bindings:
                self._visible_gpus = _visible_gpus(_binding_config(self._runtime, bindings[0]))
        except (OSError, RuntimeError, ValueError):
            return

    def _publish_heartbeat(self) -> None:
        self._refresh_visible_gpus()
        try:
            bindings = self._supervised_bindings()
        except (OSError, RuntimeError, ValueError):
            return
        readable: dict[str, RootConfig] = {}
        for binding in bindings:
            try:
                readable[binding.project_id] = _binding_config(self._runtime, binding)
            except (OSError, RuntimeError, ValueError):
                continue
        try:
            _publish_project_snapshots(
                readable,
                instance_id=self._instance_id,
                pid=_read_pid(self._runtime),
                visible=self._visible_gpus or [],
                reservations=active_reservations(self._runtime.root),
                heartbeat_interval_seconds=self._loop_interval,
                started_at=self._started_at,
            )
        except (OSError, RuntimeError, ValueError):
            return

    def _run_authority_loop(self) -> None:
        deadline = time.monotonic()
        while not self._stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining > 0 and self._stop_event.wait(remaining):
                return
            deadline = time.monotonic() + self._run_authority_cycle()

    def _run_heartbeat_loop(self) -> None:
        self._run_deadline_loop(self._publish_heartbeat, initial_delay=self._loop_interval)

    def _run_deadline_loop(self, operation, *, initial_delay: float = 0.0) -> None:
        """Run an operation on monotonic deadlines without adding execution time to the period."""
        deadline = time.monotonic() + initial_delay
        while not self._stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining > 0 and self._stop_event.wait(remaining):
                return
            operation()
            deadline += self._loop_interval
            now = time.monotonic()
            if deadline <= now:
                missed_intervals = int((now - deadline) // self._loop_interval) + 1
                deadline += missed_intervals * self._loop_interval


def dispatch_machine_cycle_locked(
    runtime: MachineRuntime,
    *,
    available_gpus: list[int] | None = None,
    executor: Executor | None = None,
    instance_id: str = "machine-agent",
    heartbeat_interval_seconds: float = 5.0,
    started_at: str | None = None,
    supervisors: dict[str, AuthoritySupervisor] | None = None,
    supervise: bool = True,
    publish_snapshots: bool = True,
) -> list[dict[str, Any]]:
    """Supervise bindings and fill capacity in fair one-claim-per-project rounds."""
    diagnostics = RuntimeDiagnostics()
    with activate_diagnostics(diagnostics), diagnostic_span("dispatch_machine_cycle"):
        results = _dispatch_machine_cycle_locked(
            runtime,
            available_gpus=available_gpus,
            executor=executor,
            instance_id=instance_id,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
            started_at=started_at,
            supervisors=supervisors,
            supervise=supervise,
            publish_snapshots=publish_snapshots,
        )
    now_ns = time.monotonic_ns()
    publish_interval_ns = DIAGNOSTIC_PUBLISH_INTERVAL_SECONDS * 1_000_000_000
    should_publish_diagnostic = (
        runtime.last_diagnostic_publish_ns is None
        or now_ns - runtime.last_diagnostic_publish_ns >= publish_interval_ns
    )
    if not should_publish_diagnostic:
        return results
    try:
        atomic_replace(
            runtime.paths["diagnostics"] / "scheduler-cycle.json",
            {
                "scheduler_diagnostic": {
                    "recorded_at": utc_now(),
                    **diagnostics.snapshot(),
                }
            },
        )
        runtime.last_diagnostic_publish_ns = now_ns
    except OSError:
        pass
    return results


def _dispatch_machine_cycle_locked(
    runtime: MachineRuntime,
    *,
    available_gpus: list[int] | None = None,
    executor: Executor | None = None,
    instance_id: str = "machine-agent",
    heartbeat_interval_seconds: float = 5.0,
    started_at: str | None = None,
    supervisors: dict[str, AuthoritySupervisor] | None = None,
    supervise: bool = True,
    publish_snapshots: bool = True,
) -> list[dict[str, Any]]:
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
    readable: dict[str, RootConfig] = {}
    dispatchable: dict[str, RootConfig] = {}
    results: list[dict[str, Any]] = []
    for binding in supervised:
        try:
            cfg = _binding_config(runtime, binding)
            runtime.drain_legacy_runner_evidence(binding)
            readable[binding.project_id] = cfg
        except (OSError, RuntimeError, ValueError) as exc:
            results.append(
                {
                    "project_id": binding.project_id,
                    "launched": [],
                    "status": "error",
                    "error": str(exc),
                }
            )
    snapshot = _reconcile_machine_reservations(runtime, readable)
    for binding in supervised:
        cfg = readable.get(binding.project_id)
        if cfg is None:
            continue
        try:
            with diagnostic_span("maintenance.project"):
                maintain_project(
                    cfg,
                    reservation_runtime_root=runtime.root,
                    project_id=binding.project_id,
                    should_reconcile_reservations=False,
                )
            if supervise:
                supervisor = None if supervisors is None else supervisors.get(binding.project_id)
                if supervisor is None:
                    supervisor = AuthoritySupervisor(cfg, reservation_runtime_root=runtime.root)
                    if supervisors is not None:
                        supervisors[binding.project_id] = supervisor
                supervisor.tick()
            dispatchable[binding.project_id] = cfg
        except (OSError, RuntimeError, ValueError) as exc:
            results.append(
                {
                    "project_id": binding.project_id,
                    "launched": [],
                    "status": "error",
                    "error": str(exc),
                }
            )
    executor = executor or Executor()
    snapshot = reconcile_snapshot(runtime.root)
    recovered = _recover_starting_reservations(
        runtime,
        dispatchable,
        snapshot.active,
        executor,
    )
    snapshot = reconcile_snapshot(runtime.root)
    visible = (
        list(available_gpus)
        if available_gpus is not None
        else _visible_gpus(next(iter(readable.values()))) if readable else []
    )
    free = [gpu_id for gpu_id in visible if gpu_id not in snapshot.reserved_gpu_ids]
    diagnostic_increment("scheduler.capacity.visible_gpus", len(visible))
    diagnostic_increment("scheduler.capacity.reserved_gpus", len(snapshot.reserved_gpu_ids))
    diagnostic_increment("scheduler.capacity.free_gpus", len(free))
    result_by_project = {
        item["project_id"]: item for item in results if item.get("project_id") is not None
    }
    for binding in ordered_enabled:
        if binding.project_id in dispatchable and binding.project_id not in result_by_project:
            item = {
                "project_id": binding.project_id,
                "launched": list(recovered.get(binding.project_id, [])),
                "status": "dispatched",
            }
            results.append(item)
            result_by_project[binding.project_id] = item
    if free:
        for binding in ordered_enabled:
            cfg = dispatchable.get(binding.project_id)
            if cfg is None or read_ready_index_state(cfg) not in {"absent", "building"}:
                continue
            try:
                advance_ready_index_build(cfg)
            except (OSError, RuntimeError, ValueError) as exc:
                item = result_by_project[binding.project_id]
                item["status"] = "error"
                item["error"] = str(exc)
                del dispatchable[binding.project_id]
    if not free:
        diagnostic_increment("scheduler.work.skipped_no_capacity", len(ordered_enabled))
    budget = SliceBudget()
    batch_sizers: dict[str, AdaptiveBatchSizer] = {}
    enabled_ids = {binding.project_id for binding in ordered_enabled}
    for project_id in set(runtime.ready_batch_sizers) - enabled_ids:
        del runtime.ready_batch_sizers[project_id]
    for binding in ordered_enabled:
        sizer = runtime.ready_batch_sizers.get(binding.project_id)
        if not isinstance(sizer, AdaptiveBatchSizer):
            sizer = AdaptiveBatchSizer(budget.policy)
            runtime.ready_batch_sizers[binding.project_id] = sizer
        batch_sizers[binding.project_id] = sizer
    inspected_ready: set[tuple[str, str, str]] = set()
    round_bindings = list(ordered_enabled)
    last_successful_binding: ProjectBinding | None = None
    has_retried_empty_round = False
    while free and round_bindings:
        if not budget.can_start_record():
            diagnostic_increment("scheduler.work.immediate_slices")
            time.sleep(0)
            budget = SliceBudget()
        round_claims = 0
        records_before_round = budget.records_used
        round_last_success: ProjectBinding | None = None
        for binding in round_bindings:
            cfg = dispatchable.get(binding.project_id)
            if cfg is None or not free or not budget.can_start_record():
                continue
            claimed_task_ids: list[str] = []
            try:
                with diagnostic_span("scheduler.work"):
                    launched = run_dispatch_cycle(
                        cfg,
                        available_gpus=free,
                        executor=executor,
                        reservation_runtime_root=runtime.root,
                        project_id=binding.project_id,
                        preflight=lambda spec: _working_directory_reason(spec) is None,
                        preflight_rejected=lambda task: _record_bad_task_spec(
                            runtime, binding, task.task_id, task.spec
                        ),
                        claim_guard=lambda: runtime.enabled_claim_guard(binding),
                        max_new_claims=1,
                        on_claim=claimed_task_ids.append,
                        should_recover_starting=False,
                        work_budget=budget,
                        batch_sizer=batch_sizers[binding.project_id],
                        inspected_ready=inspected_ready,
                    )
                result_by_project[binding.project_id]["launched"].extend(launched)
                if claimed_task_ids:
                    round_claims += 1
                    round_last_success = binding
                    last_successful_binding = binding
                    snapshot = reconcile_snapshot(runtime.root)
                    free = [
                        gpu_id for gpu_id in visible
                        if gpu_id not in snapshot.reserved_gpu_ids
                    ]
            except (OSError, RuntimeError, ValueError) as exc:
                item = result_by_project[binding.project_id]
                item["status"] = "error"
                item["error"] = str(exc)
        if (
            round_claims == 0
            and not budget.can_start_record()
            and budget.records_used == records_before_round
            and not has_retried_empty_round
        ):
            diagnostic_increment("scheduler.work.immediate_slices")
            time.sleep(0)
            budget = SliceBudget()
            round_bindings = round_bindings[1:] + round_bindings[:1]
            has_retried_empty_round = True
            continue
        if round_claims == 0:
            break
        diagnostic_increment("scheduler.work.rounds")
        if round_last_success is not None:
            index = round_bindings.index(round_last_success)
            round_bindings = round_bindings[index + 1 :] + round_bindings[: index + 1]
    if ordered_enabled:
        if last_successful_binding is not None:
            index = ordered_enabled.index(last_successful_binding)
            next_binding = ordered_enabled[(index + 1) % len(ordered_enabled)]
        else:
            next_binding = ordered_enabled[1 % len(ordered_enabled)]
        runtime.save_cursor(next_binding.project_id)
    if publish_snapshots:
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
    control_plane: _MachineControlPlane | None = None
    scheduler_wakeup = threading.Event()
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
                with machine_runtime.migration_guard() as is_migration_clear:
                    if is_migration_clear:
                        dispatch_machine_cycle_locked(
                            machine_runtime, available_gpus=available_gpus, executor=executor,
                            instance_id=instance_id, heartbeat_interval_seconds=loop_interval,
                            started_at=started_at,
                            supervise=False,
                            publish_snapshots=False,
                        )
                scheduler_wakeup.wait(loop_interval)
        finally:
            if control_plane is not None:
                control_plane.stop()
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
