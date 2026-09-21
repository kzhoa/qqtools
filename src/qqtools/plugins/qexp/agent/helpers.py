"""Shared machine agent helpers."""

from __future__ import annotations

import os
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, ContextManager, Mapping

from ..config_types import RootConfig
from ..executor import Executor, append_launch_failure_diagnostic, launch_failure_handle, launch_failure_reason
from ..gpu_policy import GpuPolicyView
from ..layout import machine_state_path
from ..machine_config import load_machine_policy
from ..machine_state import publish_machine_snapshots
from ..project_maintenance import reconcile_reservation
from ..runtime.paths import local_paths, shared_paths
from ..runtime.records import TaskSpec, utc_now
from ..runtime.resources.reservations import (
    ReservationIdentity,
    ReservationSnapshot,
    reconcile_snapshot,
    reservation_snapshot,
)
from ..runtime.store import atomic_replace, iter_json, read_json, replace_snapshot_if_changed
from ..runtime.work_budget import diagnostic_increment, diagnostic_span
from ..scheduler import fail_attempt, resume_starting_attempt
from .config import load_agent_config
from .context import MachineRuntime, ProjectBinding


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
        fields = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
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


def _machine_is_true_idle(runtime: MachineRuntime, *, has_consumed_binding: bool) -> bool:
    """Retain unfinished recovery and reservations before on-demand idle exit."""
    if getattr(runtime, "pending_launch_handoffs", {}):
        return False
    if getattr(runtime, "upgrade_pending_projects", set()):
        return False
    if getattr(runtime, "recovery_enrollment_pending_projects", set()):
        return False
    try:
        _revision, bindings = runtime.load_registry()
    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
        return False
    if not has_consumed_binding:
        # Startup waits until the current process has validated and consumed a binding.
        return False
    # Residency is machine-global.  Per-binding legacy metadata remains useful
    # for migration, but it must not decide whether this current agent exits.
    try:
        global_policy = load_agent_config(runtime)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
        return False
    for binding in bindings:
        try:
            with closing(runtime.iter_recovery_blockers(binding, should_use_capture=True)) as blockers:
                if next(blockers, None) is not None:
                    return False
            if not binding.enabled:
                continue
            cfg = _binding_config(runtime, binding)
            paths = shared_paths(cfg.shared_root)
            if any(
                next(paths[name].glob("*.json"), None) is not None
                for name in ("availability_active", "group_control_active", "cleanup_active")
            ):
                return False
            if not global_policy.exit_when_idle:
                return False
            if getattr(runtime, "last_cycle_had_demand", True):
                return False
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            return False
    try:
        snapshot = reservation_snapshot(runtime.root)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
        return False
    return not snapshot.active and not snapshot.provisional


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


def _publish_process_status(
    runtime: MachineRuntime,
    *,
    instance_id: str,
    pid: int,
    start_ticks: int,
    waiting_for_first_registration: bool,
    state: str = "active",
    runtime_id: str | None = None,
    configured_agent_mode: str | None = None,
    observed_agent_mode: str | None = None,
    policy_revision_requested: int | None = None,
    policy_revision_acknowledged: int | None = None,
    inventory_revision: int | None = None,
    reconciled_project_ids: list[str] | None = None,
    ready: bool | None = None,
) -> None:
    """Publish live process identity and its process-local registration wait state."""
    machine_agent: dict[str, Any] = {
        "instance_id": instance_id,
        "pid": pid if state == "active" else None,
        "pid_start_time_ticks": start_ticks,
        "state": state,
        "waiting_for_first_registration": waiting_for_first_registration if state == "active" else False,
    }
    optional = {
        "runtime_id": runtime_id,
        "configured_agent_mode": configured_agent_mode,
        "observed_agent_mode": observed_agent_mode,
        "policy_revision_requested": policy_revision_requested,
        "policy_revision_acknowledged": policy_revision_acknowledged,
        "inventory_revision": inventory_revision,
        "reconciled_project_ids": reconciled_project_ids,
        "ready": ready,
    }
    machine_agent.update({key: value for key, value in optional.items() if value is not None})
    replace_snapshot_if_changed(runtime.paths["agent"] / "status.json", {"machine_agent": machine_agent})


def _binding_config(runtime: MachineRuntime, binding: ProjectBinding) -> RootConfig:
    """Build the binding's isolated local runtime configuration."""
    cfg = binding.root_config()
    return replace(cfg, runtime_root=runtime.project_paths(binding.project_id)["root"])


def _consume_first_registered_binding(runtime: MachineRuntime) -> bool:
    """Validate one current binding so the process can leave its first-registration wait."""
    try:
        _revision, bindings = runtime.load_registry()
    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
        return False
    for binding in bindings:
        try:
            if not runtime.binding_write_eligible(binding, renew=True):
                continue
            _binding_config(runtime, binding)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            continue
        return True
    return False


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


def _record_bad_task_spec(runtime: MachineRuntime, binding: ProjectBinding, task_id: str, spec: TaskSpec) -> None:
    reason = _working_directory_reason(spec)
    if reason is None:
        return
    atomic_replace(
        runtime.paths["diagnostics"] / f"bad-task-spec-{binding.project_id}-{task_id}.json",
        {
            "machine_diagnostic": {
                "kind": "bad_task_spec_working_directory",
                "project_id": binding.project_id,
                "task_id": task_id,
                "path": spec.working_directory,
                "reason": reason,
            }
        },
    )


def _publish_project_snapshots(
    readable: dict[str, RootConfig],
    *,
    instance_id: str,
    pid: int | None,
    visible: list[int],
    reservations: list[dict[str, Any]],
    heartbeat_interval_seconds: float = 5.0,
    started_at: str | None = None,
    write_guard: Callable[[str], ContextManager[bool]] | None = None,
    gpu_policy: GpuPolicyView | Mapping[str, object] | None = None,
) -> None:
    """Publish each readable project's view of the shared machine reservation state."""
    reserved = sorted({gpu_id for item in reservations for gpu_id in item.get("gpu_ids", [])})
    started_at = started_at or utc_now()
    for project_id, cfg in readable.items():
        if write_guard is None:
            _publish_project_snapshot(
                cfg,
                instance_id=instance_id,
                pid=pid,
                visible=visible,
                reserved=reserved,
                reservations=reservations,
                project_id=project_id,
                heartbeat_interval_seconds=heartbeat_interval_seconds,
                started_at=started_at,
                gpu_policy=gpu_policy,
            )
            continue
        with write_guard(project_id) as is_eligible:
            if is_eligible:
                _publish_project_snapshot(
                    cfg,
                    instance_id=instance_id,
                    pid=pid,
                    visible=visible,
                    reserved=reserved,
                    reservations=reservations,
                    project_id=project_id,
                    heartbeat_interval_seconds=heartbeat_interval_seconds,
                    started_at=started_at,
                    gpu_policy=gpu_policy,
                )


def _publish_project_snapshot(
    cfg: RootConfig,
    *,
    instance_id: str,
    pid: int | None,
    visible: list[int],
    reserved: list[int],
    reservations: list[dict[str, Any]],
    project_id: str,
    heartbeat_interval_seconds: float,
    started_at: str,
    gpu_policy: GpuPolicyView | Mapping[str, object] | None = None,
) -> None:
    """Publish one project snapshot while the caller holds any required write fence."""
    attempts = [item.get("attempt_id") for item in reservations if item.get("project_id") == project_id]
    project_reservations = [item for item in reservations if item.get("project_id") == project_id]
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
        policy_value = (
            gpu_policy.with_reservations(reserved).to_dict()
            if isinstance(gpu_policy, GpuPolicyView)
            else dict(gpu_policy or {})
        )
        publish_machine_snapshots(
            cfg,
            instance_id=instance_id,
            pid=pid,
            agent_mode=load_machine_policy(cfg).agent_mode,
            observed_state="active" if reserved else "idle",
            active_attempt_ids=[item for item in attempts if isinstance(item, str)],
            visible_gpu_ids=visible,
            reserved_gpu_ids=reserved,
            reservation_summaries=project_reservations,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
            started_at=started_at,
            idle_since_at=idle_since_at,
            gpu_policy=policy_value,
        )
    except OSError:
        return


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
    *,
    excluded_pending: set[tuple[str, str]] | frozenset[tuple[str, str]] = frozenset(),
    launch_recovered: Callable[[RootConfig, str, Any, str], None] | None = None,
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
            if (identity.project_id, identity.attempt_id) in excluded_pending:
                diagnostic_increment("recovery.starting.pending_handoff")
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
                if launch_recovered is None:
                    executor.launch_attempt(cfg, identity.task_id, attempt)
                else:
                    launch_recovered(cfg, identity.task_id, attempt, identity.project_id or "")
                launched.setdefault(identity.project_id or "", []).append(identity.task_id)
                diagnostic_increment("recovery.starting.launched")
            except Exception as exc:
                diagnostic_increment("recovery.starting.launch_failed")
                append_launch_failure_diagnostic(cfg, identity.task_id, attempt.attempt_id, exc)
                try:
                    did_fail = fail_attempt(
                        cfg,
                        identity.task_id,
                        attempt.attempt_id,
                        attempt.current_fencing_token,
                        launch_failure_reason(exc),
                        should_require_unstarted=True,
                        reservation_runtime_root=runtime.root,
                    )
                except (KeyError, OSError, RuntimeError, ValueError) as compensation_error:
                    diagnostic_increment("recovery.starting.compensation_errors")
                    append_launch_failure_diagnostic(
                        cfg,
                        identity.task_id,
                        attempt.attempt_id,
                        compensation_error,
                    )
                else:
                    handle = launch_failure_handle(exc)
                    if did_fail and handle is not None:
                        try:
                            executor.cleanup_launch(handle)
                        except Exception as cleanup_error:
                            append_launch_failure_diagnostic(
                                cfg,
                                identity.task_id,
                                attempt.attempt_id,
                                cleanup_error,
                            )
    return launched
