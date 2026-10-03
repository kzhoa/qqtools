"""Machine-scoped qexp agent for fair dispatch across registered projects."""

from __future__ import annotations

import json
import os
import shlex
import signal
import sys
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, ContextManager

from ..authority import AuthoritySupervisor
from ..config_types import RootConfig
from ..executor import Executor
from ..gpu_policy import show_gpu_policy
from ..layout import load_machine_record, load_root_config, machine_state_path, runtime_pid_path
from ..legacy_agent import _visible_gpus, get_agent_status
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
from ..machine_state import publish_machine_snapshots, publish_machine_stop_snapshot
from ..notification_cleanup import cleanup_credentials
from ..notification_migration import notification_migration_status
from ..notifications import notification_runtime
from ..project_maintenance import maintain_project, reconcile_reservation
from ..runtime.group_namespace import inspect_group_authority
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
from ..runtime.recovery_admission import inspect_recovery_admission
from ..runtime.resources.cpu_lane import cpu_reservation_snapshot
from ..runtime.resources.reservations import (
    ReservationIdentity,
    ReservationSnapshot,
    reconcile_snapshot,
    reservation_snapshot,
)
from ..runtime.store import atomic_replace, iter_json, read_json
from ..runtime.upgrade.machine import inspect_registered_upgrades
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
from . import dispatch_loop as _dispatch
from . import helpers as _helpers
from .config import load_agent_config
from .context import MachineRuntime, ProjectBinding, default_machine_runtime_root
from .control_plane import _MachineControlPlane
from .diagnostics import (
    AgentLogService,
    bounded_exception_evidence,
    diagnostics_status,
    prepare_agent_diagnostics,
    update_diagnostic_evidence,
)
from .dispatch_loop import dispatch_machine_cycle
from .helpers import (
    _active_machine_identity,
    _consume_first_registered_binding,
    _machine_is_true_idle,
    _pid_start_time_ticks,
    _publish_process_status,
    _publish_project_snapshots,
    _read_pid,
)
from .inventory import exact_binding_for_entry, load_inventory, reconcile_live_bindings_locked
from .project_admin import _stop_verified_legacy_agent, migrate_project
from .project_io_controller import ProjectIOController
from .project_io_executor import ProjectIOExecutor
from .project_io_stop import publish_project_stop_snapshots
from .recovery_capture import inspect_recovery_capture
from .recovery_enrollment import RecoveryEnrollment
from .scheduler_diagnostics import SchedulerDiagnosticStore, unavailable_summary


class MachineAgentStartBlockedError(RuntimeError):
    """A declared local replacement transaction prevents agent startup."""


class MachineAgentStartError(RuntimeError):
    """The agent process could not be spawned or establish authority."""


class MachineAgentStopError(RuntimeError):
    """The running agent process could not be stopped within policy."""


def get_machine_agent_status(
    runtime: MachineRuntime | str | Path | None = None, *, probe_local_pid: bool = True
) -> dict[str, Any]:
    """Return machine-agent process and project-registry status."""
    machine_runtime = runtime if isinstance(runtime, MachineRuntime) else MachineRuntime(runtime)
    machine_runtime.require_initialized()
    agent_config = load_agent_config(machine_runtime)
    identity = _active_machine_identity(machine_runtime)
    pid = identity[0] if identity is not None else _read_pid(machine_runtime)
    running = bool(identity or (pid and not probe_local_pid))
    with machine_runtime.inventory_guard():
        with machine_runtime.registry_guard():
            reconciliation = reconcile_live_bindings_locked(machine_runtime, repair=False)
    revision = reconciliation.registry_revision
    bindings = list(reconciliation.bindings)
    inventory_revision = reconciliation.inventory_revision
    inventory_entries = list(reconciliation.entries)
    try:
        process_status = read_json(machine_runtime.paths["agent"] / "status.json").get("machine_agent", {})
    except (OSError, TypeError, ValueError):
        process_status = {}
    if not isinstance(process_status, dict):
        process_status = {}
    try:
        project_io_isolation = ProjectIOExecutor(machine_runtime).status_view()
    except Exception:
        project_io_isolation = ProjectIOExecutor._unknown_status()
    upgrade = inspect_registered_upgrades(machine_runtime)
    try:
        diagnostics = diagnostics_status(
            machine_runtime,
            configured_log_max_bytes=agent_config.log_max_bytes,
            active_identity=identity,
        )
    except Exception:
        diagnostics = None
    try:
        notification_migration = notification_migration_status(machine_runtime)
    except (OSError, RuntimeError, ValueError):
        notification_migration = {"state": "unavailable", "pending": True}
    waiting_for_first_registration = bool(process_status.get("waiting_for_first_registration")) if running else False
    inventory_by_binding = {(entry.project_id, entry.shared_root): entry for entry in inventory_entries}
    projects = []
    for binding in bindings:
        inventory_entry = inventory_by_binding.get((binding.project_id, binding.shared_root))
        exact_binding, binding_conflict = (
            exact_binding_for_entry(inventory_entry, bindings) if inventory_entry is not None else (None, True)
        )
        inventory_converged = (
            exact_binding == binding
            and not binding_conflict
            and inventory_entry is not None
            and inventory_entry.enabled == binding.enabled
        )
        try:
            eligibility = machine_runtime.registration_status(binding)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            eligibility = {"state": "invalid", "write_eligible": False, "error": str(exc)}
        project = {
            **binding.to_dict(),
            "effective_enabled": binding.enabled,
            "inventory_enabled": inventory_entry.enabled if inventory_entry is not None else None,
            "inventory_converged": inventory_converged,
            "registry_revision": revision,
            "inventory_revision": inventory_revision,
            "state": machine_runtime.binding_state(binding),
            "eligibility": eligibility,
            "recovery_enrollment": inspect_recovery_admission(machine_runtime, binding),
            "recovery_capture": inspect_recovery_capture(machine_runtime, binding),
            "group_authority": inspect_group_authority(binding.root_config()),
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
                f"qexp --machine {shlex.quote(replacement_name)} "
                f"--machine-runtime-root {shlex.quote(str(machine_runtime.root))} project register {shlex.quote(str(binding.shared_root))}"
            )
            project["recovery_note"] = (
                f"The example logical name {replacement_name!r} is illustrative; verify its availability before use."
            )
        projects.append(project)
    bound_pairs = {(binding.project_id, binding.shared_root) for binding in bindings}
    for inventory_entry in inventory_entries:
        if (inventory_entry.project_id, inventory_entry.shared_root) in bound_pairs:
            continue
        _binding, conflict = exact_binding_for_entry(inventory_entry, bindings)
        projects.append(
            {
                **inventory_entry.to_dict(),
                "state": "conflicting" if conflict else "inventory_only",
                "status": "conflicting" if conflict else "inventory_only",
                "mount_available": inventory_entry.shared_root.is_dir(),
                "machine_name": inventory_entry.name_override,
                "registration_generation": None,
                "runtime_instance_id": None,
                "effective_enabled": None,
                "inventory_enabled": inventory_entry.enabled,
                "inventory_converged": not conflict,
                "registry_revision": revision,
                "inventory_revision": inventory_revision,
                "eligibility": {"state": "unregistered", "write_eligible": False},
                "write_eligible": False,
                "reason": "missing_mount" if not inventory_entry.shared_root.is_dir() else "not_registered",
            }
        )
    try:
        gpu_policy = show_gpu_policy(machine_runtime)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
        gpu_policy = {
            "mode": "auto",
            "source": "unavailable",
            "revision": 0,
            "configured_gpu_ids": None,
            "discovered_gpu_ids": None,
            "visible_gpu_ids": None,
            "undiscovered_configured_gpu_ids": None,
            "reserved_gpu_ids": [],
            "unreserved_gpu_ids": None,
            "draining_gpu_ids": [],
            "discovery_status": "unavailable",
            "visible_status": "unavailable",
            "warnings": [{"reason": "gpu_policy_unavailable", "message": "GPU policy status is unavailable."}],
            "agent_running": running,
        }
    result = {
        "machine_runtime_root": str(machine_runtime.root),
        "runtime_id": machine_runtime.instance_id,
        "configured_agent_mode": agent_config.agent_mode,
        "observed_agent_mode": process_status.get("observed_agent_mode"),
        "requested_policy_revision": process_status.get("policy_revision_requested", agent_config.revision),
        "acknowledged_policy_revision": process_status.get("policy_revision_acknowledged"),
        "policy_revision_requested": process_status.get("policy_revision_requested", agent_config.revision),
        "policy_revision_acknowledged": process_status.get("policy_revision_acknowledged"),
        "readiness": bool(process_status.get("ready")) and running and reconciliation.converged,
        "ready": bool(process_status.get("ready")) and running and reconciliation.converged,
        "inventory_revision": inventory_revision,
        "inventory_converged": reconciliation.converged,
        "enablement_blockers": list(reconciliation.blockers),
        "reconciled_project_ids": process_status.get("reconciled_project_ids", []),
        "agent_state": "active" if running else "stopped",
        "stop_reason": process_status.get("stop_reason"),
        "pid": pid,
        "is_running": running,
        "waiting_for_first_registration": waiting_for_first_registration if running else False,
        "registry_revision": revision,
        "projects": projects,
        "upgrade": upgrade,
        "notification_migration": notification_migration,
        "gpu_policy": gpu_policy,
        "project_io_isolation": project_io_isolation,
        "warnings": list(gpu_policy.get("warnings", [])),
    }
    if diagnostics is not None:
        result["diagnostics"] = diagnostics
    try:
        result["scheduler_diagnostics"] = SchedulerDiagnosticStore(machine_runtime).summary_view()
    except Exception:
        result["scheduler_diagnostics"] = unavailable_summary(machine_runtime.root, reason="summary_unavailable")
    return result


def run_machine_agent_loop(
    runtime: MachineRuntime | str | Path | None = None,
    *,
    loop_interval: float = 5.0,
    available_gpus: list[int] | None = None,
    executor: Executor | None = None,
    instance_id: str | None = None,
    startup_sequence: int | None = None,
    log_path: str | Path | None = None,
    effective_log_max_bytes: int | None = None,
    capture_mode: str | None = None,
    initial_capture_health: str | None = None,
    initial_capture_error: str | None = None,
    initial_reconciliation_degraded: bool = False,
) -> None:
    """Run the persistent machine agent until SIGTERM, SIGINT, or SIGHUP."""
    if loop_interval <= 0:
        raise ValueError("loop_interval must be positive.")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("machine agent loop must run in the process main thread.")
    machine_runtime = runtime if isinstance(runtime, MachineRuntime) else MachineRuntime(runtime)
    machine_runtime.require_initialized()
    if machine_runtime.paths["replacement_transaction"].exists():
        raise MachineAgentStartBlockedError("machine agent activation is blocked by a pending machine replacement.")
    machine_runtime.ensure_layout(create_identity=False)
    agent_config = load_agent_config(machine_runtime)
    instance_id = instance_id or uuid.uuid4().hex
    if effective_log_max_bytes is None:
        effective_log_max_bytes = agent_config.log_max_bytes
    prepared = None
    if startup_sequence is None:
        selected_capture_mode = capture_mode or "foreground_managed_only"
        prepared = prepare_agent_diagnostics(
            machine_runtime,
            instance_id=instance_id,
            capture_mode=selected_capture_mode,
            log_max_bytes=effective_log_max_bytes,
        )
        startup_sequence = prepared.startup_sequence
        log_path = log_path if log_path is not None else prepared.log_path
        capture_mode = capture_mode or prepared.capture_mode
        initial_capture_health = prepared.capture_health
        initial_capture_error = prepared.error
        initial_reconciliation_degraded = prepared.reconciliation_degraded
        prepared.close()
    else:
        capture_mode = capture_mode or "detached"
        initial_capture_health = initial_capture_health or ("healthy" if log_path is not None else "degraded")
        initial_capture_error = initial_capture_error or (
            "log_unavailable" if initial_capture_health != "healthy" else None
        )
    try:
        initial_inventory_revision, _initial_inventory = load_inventory(machine_runtime)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
        initial_inventory_revision = None
    pid_path = machine_runtime.paths["pid"]
    current_identity = _active_machine_identity(machine_runtime)
    if current_identity is not None and current_identity[0] != os.getpid():
        raise MachineAgentStartError(f"machine agent is already running with pid {current_identity[0]}.")
    started_at = utc_now()
    control_plane: _MachineControlPlane | None = None
    project_io_executor: ProjectIOExecutor | None = None
    scheduler_wakeup = threading.Event()
    stop = False
    stop_reason: str | None = None
    handled_signal: int | None = None
    has_consumed_binding = False
    policy_revision_requested = agent_config.revision
    policy_revision_acknowledged: int | None = None
    observed_agent_mode = agent_config.agent_mode
    observed_inventory_revision = initial_inventory_revision
    reconciled_project_ids: list[str] = []
    reconciled_inventory_revision: int | None = None
    reconciled_registry_revision: int | None = None
    reconciled_policy_revision: int | None = None
    enabled_project_ids: set[str] = set()
    enabled_projects_reconciled = False
    enablement_view_converged = False
    readiness_view_initialized = False
    cycle_completed = False
    idle_since: float | None = None
    recovery_enrollment = RecoveryEnrollment(machine_runtime)
    machine_runtime.recovery_enrollment = recovery_enrollment
    emitted_gpu_warning_fingerprint: str | None = None
    idle_shutdown_guard = None
    agent_revision = 0
    agent_revision_lock = threading.Lock()
    cleanup_steps: dict[str, dict[str, Any]] = {}
    cleanup_failures: list[str] = []
    capture_health_state = {
        "health": initial_capture_health,
        "error": initial_capture_error,
    }

    def capture_health_callback(health: str, error: str | None) -> None:
        if initial_reconciliation_degraded:
            capture_health_state["health"] = "degraded"
            capture_health_state["error"] = (
                "reconciliation_unavailable" if health == "healthy" else "reconciliation_and_capture_degraded"
            )
        else:
            capture_health_state["health"] = health
            capture_health_state["error"] = error
        publish_agent_evidence(
            capture_health=capture_health_state["health"],
            capture_error=capture_health_state["error"],
        )

    log_service = (
        AgentLogService(
            log_path,
            max_bytes=effective_log_max_bytes,
            capture_mode=capture_mode,
            health_callback=capture_health_callback,
        )
        if log_path is not None
        else None
    )

    def publish_agent_evidence(**evidence: object) -> bool:
        nonlocal agent_revision
        with agent_revision_lock:
            agent_revision += 1
            revision = agent_revision
        try:
            return update_diagnostic_evidence(
                machine_runtime,
                instance_id=instance_id,
                startup_sequence=startup_sequence,
                writer="agent",
                revision=revision,
                evidence=evidence,
            )
        except Exception:
            return False

    def write_managed_log(message: str) -> None:
        if log_service is None:
            return
        try:
            log_service.write(message)
        except Exception:
            return

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stop, stop_reason, handled_signal
        stop = True
        if stop_reason is None:
            stop_reason = "stopped_by_signal"
            handled_signal = signum
        scheduler_wakeup.set()

    @contextmanager
    def retain_idle_shutdown_guard():
        try:
            yield
        finally:
            if idle_shutdown_guard is not None:
                idle_shutdown_guard.__exit__(None, None, None)

    with (
        retain_idle_shutdown_guard(),
        machine_runtime.scheduler_authority(blocking=False) as acquired,
        notification_runtime(machine_runtime.root),
    ):
        if not acquired:
            raise RuntimeError("machine scheduler authority is already held.")
        start_ticks = _pid_start_time_ticks(os.getpid())
        if start_ticks is None:
            raise RuntimeError("could not determine machine agent process identity.")
        previous_term = None
        previous_int = None
        previous_hup = None
        is_pid_published = False
        is_status_published = False
        primary_exception: BaseException | None = None
        primary_traceback = None

        def safe_error_facts(exc: BaseException) -> dict[str, Any]:
            try:
                facts = bounded_exception_evidence(exc)
            except BaseException:
                return {"exception_type": type(exc).__name__}
            return facts if isinstance(facts, dict) else {"exception_type": type(exc).__name__}

        def record_cleanup_step(name: str, exc: BaseException | None = None) -> None:
            if exc is None:
                cleanup_steps[name] = {"outcome": "succeeded"}
                return
            error_facts = safe_error_facts(exc)
            cleanup_steps[name] = {
                "outcome": "failed",
                "error_type": error_facts.get("exception_type", type(exc).__name__),
            }
            cleanup_failures.append(name)

        def run_cleanup_step(name: str, action: Callable[[], Any]) -> None:
            try:
                action()
            except BaseException as exc:
                record_cleanup_step(name, exc)
            else:
                record_cleanup_step(name)

        def require_agent_diagnostic_write(**evidence: object) -> None:
            if not publish_agent_evidence(**evidence):
                raise RuntimeError("agent diagnostic publication was rejected")

        try:
            project_io_executor = ProjectIOExecutor(machine_runtime)
            project_io_executor.begin_epoch()
            machine_runtime.project_io_executor = project_io_executor
            # Establish the sole Project-I/O owner before any control-plane
            # thread starts.  The lifecycle must never expose a production
            # window where the legacy authority supervisor can claim a turn.
            _dispatch._project_io_controller(machine_runtime)
            previous_term = signal.signal(signal.SIGTERM, request_stop)
            previous_int = signal.signal(signal.SIGINT, request_stop)
            previous_hup = signal.signal(signal.SIGHUP, request_stop)
            pid_path.write_text(str(os.getpid()), encoding="utf-8")
            is_pid_published = True
            _publish_process_status(
                machine_runtime,
                instance_id=instance_id,
                pid=os.getpid(),
                start_ticks=start_ticks,
                waiting_for_first_registration=True,
                runtime_id=machine_runtime.instance_id,
                configured_agent_mode=agent_config.agent_mode,
                observed_agent_mode=observed_agent_mode,
                policy_revision_requested=policy_revision_requested,
                policy_revision_acknowledged=policy_revision_acknowledged,
                inventory_revision=observed_inventory_revision,
                reconciled_project_ids=reconciled_project_ids,
                ready=False,
                diagnostic_startup_sequence=startup_sequence,
                diagnostic_log_path=str(log_path) if log_path is not None else None,
                effective_log_max_bytes=effective_log_max_bytes,
                capture_mode=capture_mode,
            )
            is_status_published = True
            publish_agent_evidence(
                phase="active",
                admitted=True,
                pid=os.getpid(),
                pid_start_time_ticks=start_ticks,
                capture_health=capture_health_state["health"],
                capture_error=capture_health_state["error"],
                log_path=str(log_path) if log_path is not None else None,
                effective_log_max_bytes=effective_log_max_bytes,
                capture_mode=capture_mode,
            )
            if log_service is not None:
                try:
                    log_service.start()
                except Exception as exc:
                    capture_health_state.update(health="degraded", error="log_service_start_failed")
                    publish_agent_evidence(capture_health="degraded", capture_error="log_service_start_failed")
                    write_managed_log(f"machine agent log service startup failed: {type(exc).__name__}\n")
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
                    project_io_executor.poll()
                except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                    pass
                try:
                    agent_config = load_agent_config(machine_runtime)
                    policy_revision_requested = agent_config.revision
                    observed_agent_mode = agent_config.agent_mode
                except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                    # Keep the last valid policy observation while the durable
                    # config is being repaired or published by another actor.
                    pass
                cycle_completed = False
                try:
                    recovery_enrollment.poll()
                except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                    # Local registry failure is retried without occupying the
                    # independent authority and heartbeat control loops.
                    pass
                if time.monotonic() >= machine_runtime.notification_next_pass_at:
                    cleanup_credentials(machine_runtime.root, limit=8, blocking=False)
                    machine_runtime.notification_next_pass_at = time.monotonic() + 5.0
                try:
                    with machine_runtime.migration_read_guard() as is_migration_clear:
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
                            cycle_completed = True
                            if not has_consumed_binding:
                                if getattr(machine_runtime, "project_io_executor", None) is not None:
                                    has_consumed_binding = machine_runtime.last_cycle_consumed_binding
                                elif machine_runtime.last_cycle_consumed_binding or _consume_first_registered_binding(
                                    machine_runtime
                                ):
                                    has_consumed_binding = True
                except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                    # A transient shared-root failure must not stop supervision of other projects.
                    pass
                try:
                    gpu_status = show_gpu_policy(machine_runtime)
                    gpu_warnings = gpu_status.get("warnings", [])
                    if gpu_warnings:
                        fingerprint = json.dumps(
                            {
                                "revision": gpu_status.get("revision"),
                                "discovered": gpu_status.get("discovered_gpu_ids"),
                                "warnings": gpu_warnings,
                            },
                            sort_keys=True,
                        )
                        if fingerprint != emitted_gpu_warning_fingerprint:
                            message = gpu_warnings[0].get("message") if isinstance(gpu_warnings[0], dict) else None
                            if isinstance(message, str) and message:
                                print(f"Warning: {message}", file=sys.stderr, flush=True)
                                write_managed_log(f"Warning: {message}\n")
                            emitted_gpu_warning_fingerprint = fingerprint
                except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                    pass
                enablement_view_converged = False
                try:
                    reconciliation = None
                    with machine_runtime.agent_lifecycle_guard(blocking=False) as lifecycle_acquired:
                        if lifecycle_acquired:
                            with machine_runtime.inventory_guard():
                                with machine_runtime.registry_guard():
                                    reconciliation = reconcile_live_bindings_locked(machine_runtime)
                    if reconciliation is None:
                        raise RuntimeError("machine lifecycle is changing; readiness will be retried.")
                    observed_inventory_revision = reconciliation.inventory_revision
                    registry_revision = reconciliation.registry_revision
                    observed_entries = reconciliation.entries
                    observed_bindings = reconciliation.bindings
                    enablement_view_converged = reconciliation.converged
                    machine_runtime.last_enablement_reconciliation_probe = _dispatch._enablement_reconciliation_probe(
                        machine_runtime, reconciliation
                    )
                    if cycle_completed:
                        view_changed = (
                            not readiness_view_initialized
                            or not enabled_projects_reconciled
                            or reconciled_inventory_revision != observed_inventory_revision
                            or reconciled_registry_revision != registry_revision
                            or reconciled_policy_revision != policy_revision_requested
                            or (
                                project_io_executor is not None
                                and not enabled_project_ids.issubset(
                                    getattr(machine_runtime, "last_cycle_validated_project_ids", frozenset())
                                )
                            )
                        )
                        if view_changed:
                            enabled_project_ids = {
                                binding.project_id for binding in observed_bindings if binding.enabled
                            }
                            for entry in observed_entries:
                                binding, conflict = exact_binding_for_entry(entry, observed_bindings)
                                if (binding is None or conflict) and entry.enabled:
                                    enabled_project_ids.add(entry.project_id)
                            entry_by_binding = {
                                (entry.project_id, entry.shared_root): entry for entry in observed_entries
                            }
                            project_io_controller = getattr(machine_runtime, "project_io_controller", None)
                            if project_io_executor is not None and isinstance(
                                project_io_controller, ProjectIOController
                            ):
                                reconciled_project_ids = [
                                    binding.project_id
                                    for binding in observed_bindings
                                    if binding.enabled
                                    and (entry := entry_by_binding.get((binding.project_id, binding.shared_root)))
                                    is not None
                                    and not exact_binding_for_entry(entry, observed_bindings)[1]
                                    and project_io_controller.validated_config(binding, registry_revision) is not None
                                ]
                            else:
                                reconciled_project_ids = [
                                    binding.project_id
                                    for binding in observed_bindings
                                    if binding.enabled
                                    and (entry := entry_by_binding.get((binding.project_id, binding.shared_root)))
                                    is not None
                                    and not exact_binding_for_entry(entry, observed_bindings)[1]
                                    and machine_runtime.registration_status(binding).get("write_eligible", False)
                                ]
                            enabled_projects_reconciled = enablement_view_converged and enabled_project_ids.issubset(
                                set(reconciled_project_ids)
                            )
                            reconciled_inventory_revision = observed_inventory_revision
                            reconciled_registry_revision = registry_revision
                            reconciled_policy_revision = policy_revision_requested
                            readiness_view_initialized = True
                        if (
                            reconciled_inventory_revision == observed_inventory_revision
                            and reconciled_registry_revision == registry_revision
                            and reconciled_policy_revision == policy_revision_requested
                        ):
                            # A completed cycle acknowledges only the durable view
                            # whose enabled projects have been reconciled.
                            policy_revision_acknowledged = policy_revision_requested
                    ready = bool(
                        cycle_completed
                        and enablement_view_converged
                        and enabled_project_ids
                        and policy_revision_acknowledged == policy_revision_requested
                        and enabled_projects_reconciled
                        and (
                            project_io_executor is None
                            or enabled_project_ids.issubset(
                                getattr(machine_runtime, "last_cycle_validated_project_ids", frozenset())
                            )
                        )
                    )
                    _publish_process_status(
                        machine_runtime,
                        instance_id=instance_id,
                        pid=os.getpid(),
                        start_ticks=start_ticks,
                        waiting_for_first_registration=not has_consumed_binding,
                        runtime_id=machine_runtime.instance_id,
                        configured_agent_mode=agent_config.agent_mode,
                        observed_agent_mode=observed_agent_mode,
                        policy_revision_requested=policy_revision_requested,
                        policy_revision_acknowledged=policy_revision_acknowledged,
                        inventory_revision=observed_inventory_revision,
                        reconciled_project_ids=reconciled_project_ids,
                        ready=ready,
                        diagnostic_startup_sequence=startup_sequence,
                        diagnostic_log_path=str(log_path) if log_path is not None else None,
                        effective_log_max_bytes=effective_log_max_bytes,
                        capture_mode=capture_mode,
                    )
                except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                    machine_runtime.last_enablement_reconciliation_probe = _dispatch._enablement_reconciliation_probe(
                        machine_runtime, None
                    )
                    try:
                        _publish_process_status(
                            machine_runtime,
                            instance_id=instance_id,
                            pid=os.getpid(),
                            start_ticks=start_ticks,
                            waiting_for_first_registration=not has_consumed_binding,
                            runtime_id=machine_runtime.instance_id,
                            configured_agent_mode=agent_config.agent_mode,
                            observed_agent_mode=observed_agent_mode,
                            policy_revision_requested=policy_revision_requested,
                            policy_revision_acknowledged=policy_revision_acknowledged,
                            inventory_revision=observed_inventory_revision,
                            reconciled_project_ids=reconciled_project_ids,
                            ready=False,
                            diagnostic_startup_sequence=startup_sequence,
                            diagnostic_log_path=str(log_path) if log_path is not None else None,
                            effective_log_max_bytes=effective_log_max_bytes,
                            capture_mode=capture_mode,
                        )
                    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                        pass
                try:
                    # Dispatch can take a full multi-project cycle. Consume the
                    # service's latest completion before deciding to stay alive.
                    recovery_enrollment.poll()
                    is_idle = _machine_is_true_idle(machine_runtime, has_consumed_binding=has_consumed_binding)
                except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                    is_idle = False
                if is_idle:
                    if idle_since is None:
                        idle_since = time.monotonic()
                    elif time.monotonic() - idle_since >= loop_interval:
                        idle_shutdown_guard = _confirm_idle_shutdown(
                            machine_runtime,
                            has_consumed_binding=has_consumed_binding,
                            available_gpus=available_gpus,
                            executor=executor,
                            instance_id=instance_id,
                            loop_interval=loop_interval,
                            started_at=started_at,
                        )
                        if idle_shutdown_guard is not None:
                            stop = True
                            if stop_reason is None:
                                stop_reason = "idle"
                            continue
                        idle_since = None
                else:
                    idle_since = None
                pending_wait = getattr(machine_runtime, "pending_launch_wait_seconds", None)
                wait_seconds = pending_wait(loop_interval) if callable(pending_wait) else loop_interval
                project_io_controller = getattr(machine_runtime, "project_io_controller", None)
                if isinstance(project_io_controller, ProjectIOController):
                    try:
                        wait_seconds = project_io_controller.pending_wait_seconds(wait_seconds)
                    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                        pass
                scheduler_wakeup.wait(wait_seconds)
        except BaseException as exc:
            primary_exception = exc
            primary_traceback = exc.__traceback__
            stop = True
            stop_reason = "unhandled_exception"
            try:
                exception_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            except BaseException:
                exception_text = f"{type(exc).__name__}\n"
            write_managed_log(exception_text)
        finally:
            frozen_stop_reason = stop_reason or "stopped"
            stop_reason = frozen_stop_reason
            primary_exception_evidence = safe_error_facts(primary_exception) if primary_exception is not None else None

            # Fence Project I/O authority before any cleanup that may itself
            # block. Later shutdown performs the bounded signal/reap sequence.
            if project_io_executor is None:
                record_cleanup_step("project_io_executor_fence")
            else:
                run_cleanup_step("project_io_executor_fence", project_io_executor.fence_epoch)

            def publish_stopping_evidence() -> None:
                evidence: dict[str, object] = {
                    "phase": "stopping",
                    "admitted": False,
                    "pid": os.getpid(),
                    "pid_start_time_ticks": start_ticks,
                    "capture_health": capture_health_state["health"],
                    "capture_error": capture_health_state["error"],
                    "log_path": str(log_path) if log_path is not None else None,
                    "effective_log_max_bytes": effective_log_max_bytes,
                    "capture_mode": capture_mode,
                    "stop_reason": frozen_stop_reason,
                    "cleanup_outcome": "pending",
                }
                if handled_signal is not None:
                    evidence["handled_signal"] = handled_signal
                if primary_exception_evidence is not None:
                    evidence["primary_exception"] = primary_exception_evidence
                require_agent_diagnostic_write(**evidence)

            # Record the transition before stopping any long-lived service.
            run_cleanup_step("publish_stopping_evidence", publish_stopping_evidence)
            run_cleanup_step("recovery_enrollment_stop", recovery_enrollment.stop)

            if control_plane is None:
                record_cleanup_step("control_plane_stop")
            else:
                run_cleanup_step("control_plane_stop", control_plane.stop)

            def close_attempt_supervision() -> None:
                coordinator = getattr(machine_runtime, "attempt_supervision_coordinator", None)
                if coordinator is not None:
                    coordinator.close()

            run_cleanup_step("attempt_supervision_coordinator_close", close_attempt_supervision)

            def close_progress_observation() -> None:
                controller = getattr(machine_runtime, "project_io_controller", None)
                if isinstance(controller, ProjectIOController):
                    controller.progress.close()

            run_cleanup_step("progress_observation_close", close_progress_observation)

            if project_io_executor is None:
                record_cleanup_step("project_io_executor_shutdown")
            else:
                run_cleanup_step("project_io_executor_shutdown", project_io_executor.shutdown)

            def publish_project_stops() -> None:
                expected_identity = (os.getpid(), instance_id, start_ticks)
                if _active_machine_identity(machine_runtime) != expected_identity:
                    return
                if project_io_executor is None:
                    raise RuntimeError("Project stop publication requires the isolated executor.")
                publish_project_stop_snapshots(
                    machine_runtime,
                    project_io_executor,
                    instance_id=instance_id,
                    available_gpus=available_gpus,
                    heartbeat_interval_seconds=loop_interval,
                    started_at=started_at,
                    stop_reason=frozen_stop_reason,
                )

            run_cleanup_step("project_stop_publications", publish_project_stops)

            expected_identity = (os.getpid(), instance_id, start_ticks)

            def publish_local_stopped_status() -> None:
                if not is_status_published or _active_machine_identity(machine_runtime) != expected_identity:
                    return
                atomic_replace(
                    machine_runtime.paths["agent"] / "status.json",
                    {
                        "machine_agent": {
                            "instance_id": instance_id,
                            "pid": None,
                            "pid_start_time_ticks": start_ticks,
                            "state": "stopped",
                            "waiting_for_first_registration": False,
                            "runtime_id": machine_runtime.instance_id,
                            "configured_agent_mode": agent_config.agent_mode,
                            "observed_agent_mode": observed_agent_mode,
                            "policy_revision_requested": policy_revision_requested,
                            "policy_revision_acknowledged": policy_revision_acknowledged,
                            "inventory_revision": observed_inventory_revision,
                            "reconciled_project_ids": reconciled_project_ids,
                            "ready": False,
                            "stop_reason": frozen_stop_reason,
                            "diagnostic_startup_sequence": startup_sequence,
                            "diagnostic_log_path": str(log_path) if log_path is not None else None,
                            "effective_log_max_bytes": effective_log_max_bytes,
                            "capture_mode": capture_mode,
                        }
                    },
                )

            run_cleanup_step("local_stopped_status", publish_local_stopped_status)

            def remove_owned_pid() -> None:
                if not is_pid_published:
                    return
                try:
                    recorded_pid = int(pid_path.read_text(encoding="utf-8").strip())
                except (OSError, ValueError):
                    return
                if recorded_pid != os.getpid():
                    return
                if _active_machine_identity(machine_runtime) == expected_identity:
                    pid_path.unlink(missing_ok=True)
                    return
                try:
                    stopped_status = read_json(machine_runtime.paths["agent"] / "status.json").get("machine_agent", {})
                except (OSError, RuntimeError, ValueError, TypeError):
                    return
                if (
                    stopped_status.get("state") == "stopped"
                    and stopped_status.get("instance_id") == instance_id
                    and stopped_status.get("pid") is None
                    and stopped_status.get("pid_start_time_ticks") == start_ticks
                ):
                    pid_path.unlink(missing_ok=True)

            run_cleanup_step("identity_checked_pid_removal", remove_owned_pid)

            def release_idle_shutdown_guard() -> None:
                nonlocal idle_shutdown_guard
                if idle_shutdown_guard is not None:
                    guard = idle_shutdown_guard
                    idle_shutdown_guard = None
                    guard.__exit__(None, None, None)

            run_cleanup_step("idle_shutdown_guard_release", release_idle_shutdown_guard)

            if previous_hup is None:
                record_cleanup_step("restore_sighup_handler")
            else:
                run_cleanup_step("restore_sighup_handler", lambda: signal.signal(signal.SIGHUP, previous_hup))

            if previous_int is None:
                record_cleanup_step("restore_sigint_handler")
            else:
                run_cleanup_step("restore_sigint_handler", lambda: signal.signal(signal.SIGINT, previous_int))
            if previous_term is None:
                record_cleanup_step("restore_sigterm_handler")
            else:
                run_cleanup_step("restore_sigterm_handler", lambda: signal.signal(signal.SIGTERM, previous_term))

            if log_service is None:
                record_cleanup_step("log_service_final_check")
                record_cleanup_step("log_service_stop")
            else:

                def final_log_check() -> None:
                    try:
                        signal_detail = f" (signal={handled_signal})" if handled_signal is not None else ""
                        if not log_service.write(f"machine agent stopping: {frozen_stop_reason}{signal_detail}\n"):
                            raise OSError("final managed log write failed")
                    except BaseException:
                        capture_health_state.update(health="degraded", error="final_log_check_failed")
                        publish_agent_evidence(capture_health="degraded", capture_error="final_log_check_failed")
                        raise

                def stop_log_service() -> None:
                    try:
                        if not log_service.stop():
                            raise OSError("final managed log rotation or shutdown failed")
                    except BaseException:
                        capture_health_state.update(health="degraded", error="log_service_stop_failed")
                        publish_agent_evidence(capture_health="degraded", capture_error="log_service_stop_failed")
                        raise

                run_cleanup_step("log_service_final_check", final_log_check)
                run_cleanup_step("log_service_stop", stop_log_service)

            final_evidence: dict[str, object] = {
                "phase": "stopped",
                "admitted": False,
                "pid": None,
                "pid_start_time_ticks": start_ticks,
                "capture_health": capture_health_state["health"],
                "capture_error": capture_health_state["error"],
                "log_path": str(log_path) if log_path is not None else None,
                "effective_log_max_bytes": effective_log_max_bytes,
                "capture_mode": capture_mode,
                "stop_reason": frozen_stop_reason,
                "cleanup_outcome": "failed" if cleanup_failures else "succeeded",
                "cleanup_steps": dict(cleanup_steps),
                "finalized_at": utc_now(),
            }
            if handled_signal is not None:
                final_evidence["handled_signal"] = handled_signal
            if primary_exception_evidence is not None:
                final_evidence["primary_exception"] = primary_exception_evidence
            try:
                if not publish_agent_evidence(**final_evidence):
                    cleanup_failures.append("final_diagnostic_publication")
            except BaseException:
                cleanup_failures.append("final_diagnostic_publication")

        if primary_exception is not None:
            raise primary_exception.with_traceback(primary_traceback)
        if cleanup_failures:
            raise MachineAgentStopError("machine agent cleanup failed in step(s): " + ", ".join(cleanup_failures))


def _confirm_idle_shutdown(
    runtime: MachineRuntime,
    *,
    has_consumed_binding: bool,
    available_gpus: list[int] | None,
    executor: Executor | None,
    instance_id: str,
    loop_interval: float,
    started_at: str,
) -> ContextManager[bool] | None:
    """Recheck demand while excluding a concurrent activation decision.

    Submission publishes its durable ready work before entering the same
    lifecycle guard in ``ensure_machine_agent_started``. Isolated callers check
    captured local wake and registry identities instead of dispatching shared
    work under that guard. Hold it through stopped-status publication so a
    concurrent activation cannot accept an agent already committed to exit.
    """

    # The process already owns scheduler authority. Lifecycle operations take
    # this lock before waiting for that process, so an idle check must never
    # wait here and invert the lifecycle/scheduler lock order.
    guard = runtime.agent_lifecycle_guard(blocking=False)
    acquired = guard.__enter__()
    if not acquired:
        guard.__exit__(None, None, None)
        return None
    try:
        with runtime.migration_read_guard() as is_migration_clear:
            if not is_migration_clear:
                guard.__exit__(None, None, None)
                return None
            if getattr(runtime, "project_io_executor", None) is not None:
                if not runtime.activation_wake.is_current():
                    guard.__exit__(None, None, None)
                    return None
            else:
                _dispatch.dispatch_machine_cycle_locked(
                    runtime,
                    available_gpus=available_gpus,
                    executor=executor,
                    instance_id=instance_id,
                    heartbeat_interval_seconds=loop_interval,
                    started_at=started_at,
                    supervise=False,
                    publish_snapshots=False,
                )
        if not _machine_is_true_idle(runtime, has_consumed_binding=has_consumed_binding):
            guard.__exit__(None, None, None)
            return None
        return guard
    except BaseException:
        guard.__exit__(*sys.exc_info())
        raise


def _start_machine_agent_locked(
    machine_runtime: MachineRuntime,
    *,
    available_gpus: list[int] | None = None,
    loop_interval: float | None = None,
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
        loop_interval=loop_interval,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
    )


def start_machine_agent(
    runtime: MachineRuntime | str | Path | None = None,
    *,
    available_gpus: list[int] | None = None,
    loop_interval: float | None = None,
    stdin=None,
    stdout=None,
    stderr=None,
):
    """Spawn the unique persistent machine agent."""
    machine_runtime = runtime if isinstance(runtime, MachineRuntime) else MachineRuntime(runtime)
    machine_runtime.require_initialized()
    if machine_runtime.paths["replacement_transaction"].exists():
        raise MachineAgentStartBlockedError("machine agent start is blocked by a pending machine replacement.")
    with machine_runtime.agent_lifecycle_guard():
        return _start_machine_agent_locked(
            machine_runtime,
            available_gpus=available_gpus,
            loop_interval=loop_interval,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
        )


def ensure_machine_agent_started(
    runtime: MachineRuntime | str | Path | None = None,
    *,
    available_gpus: list[int] | None = None,
    loop_interval: float | None = None,
    stdin=None,
    stdout=None,
    stderr=None,
):
    """Return the running agent status, starting it atomically when absent."""
    machine_runtime = runtime if isinstance(runtime, MachineRuntime) else MachineRuntime(runtime)
    machine_runtime.require_initialized()
    if machine_runtime.paths["replacement_transaction"].exists():
        raise MachineAgentStartBlockedError("machine agent start is blocked by a pending machine replacement.")
    with machine_runtime.agent_lifecycle_guard():
        machine_runtime.activation_wake.publish_locked()
        status = get_machine_agent_status(machine_runtime)
        if status["is_running"]:
            return None, status
        try:
            process = _start_machine_agent_locked(
                machine_runtime,
                available_gpus=available_gpus,
                loop_interval=loop_interval,
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
    except OSError as exc:
        raise MachineAgentStopError(f"machine agent {pid} could not be signalled: {exc}") from exc
    deadline = time.monotonic() + timeout
    while _pid_start_time_ticks(pid) == start_ticks and time.monotonic() < deadline:
        time.sleep(0.05)
    if _pid_start_time_ticks(pid) == start_ticks:
        raise MachineAgentStopError(f"machine agent {pid} did not stop within {timeout} seconds.")
    return True


def stop_machine_agent(runtime: MachineRuntime | str | Path | None = None, *, timeout: float = 10.0) -> bool:
    """Request a graceful machine-agent stop and wait for the process to exit."""
    machine_runtime = runtime if isinstance(runtime, MachineRuntime) else MachineRuntime(runtime)
    machine_runtime.require_initialized()
    with machine_runtime.agent_lifecycle_guard():
        return _stop_machine_agent_locked(machine_runtime, timeout=timeout)


def restart_machine_agent(
    runtime: MachineRuntime | str | Path | None = None,
    *,
    available_gpus: list[int] | None = None,
    loop_interval: float | None = None,
    stdin=None,
    stdout=None,
    stderr=None,
):
    """Replace a running machine agent without treating it as a cold start."""
    machine_runtime = runtime if isinstance(runtime, MachineRuntime) else MachineRuntime(runtime)
    machine_runtime.require_initialized()
    with machine_runtime.agent_lifecycle_guard():
        if machine_runtime.paths["replacement_transaction"].exists():
            raise MachineAgentStartBlockedError("machine agent restart is blocked by a pending machine replacement.")
        identity = _active_machine_identity(machine_runtime)
        previous_pid = identity[0] if identity is not None else None
        _stop_machine_agent_locked(machine_runtime, timeout=10.0)
        process = _start_machine_agent_locked(
            machine_runtime,
            available_gpus=available_gpus,
            loop_interval=loop_interval,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
        )
        process.previous_pid = previous_pid
        return process


# Public lifecycle entry point retained for internal test and CLI orchestration.
def dispatch_machine_cycle_locked(*args, **kwargs):
    return _dispatch.dispatch_machine_cycle_locked(*args, **kwargs)
