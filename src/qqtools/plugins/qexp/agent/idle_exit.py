"""Machine-agent idle exit eligibility and final lifecycle guard acquisition."""

from __future__ import annotations

import sys
from contextlib import closing
from typing import ContextManager

from ..executor import Executor
from ..runtime.paths import shared_paths
from ..runtime.resources.reservations import reservation_snapshot
from . import dispatch_loop
from .config import load_agent_config
from .context import MachineRuntime
from .helpers import _binding_config
from .takeover_intent import read_takeover_intent


def can_exit_when_idle(runtime: MachineRuntime, *, has_consumed_binding: bool) -> bool:
    """Retain unfinished recovery and reservations before on-demand idle exit."""
    project_io_executor = getattr(runtime, "project_io_executor", None)
    if project_io_executor is not None:
        try:
            if read_takeover_intent(runtime.root) is not None or project_io_executor.has_unfinished_work():
                return False
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            return False
    if getattr(runtime, "pending_launch_handoffs", {}):
        return False
    if getattr(runtime, "upgrade_discovery_unknown", False):
        return False
    if getattr(runtime, "upgrade_idle_blocked_projects", set()):
        return False
    if getattr(runtime, "recovery_enrollment_pending_projects", set()):
        return False
    if getattr(runtime, "maintenance_retry_idle_blocked_projects", set()):
        return False
    if not has_consumed_binding:
        # Startup waits until the current process has validated and consumed a binding.
        return False
    if getattr(runtime, "last_cycle_had_demand", True):
        return False
    try:
        global_policy = load_agent_config(runtime)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
        return False
    try:
        if not runtime.working_set.is_reconciled:
            revision, bindings = runtime.load_registry_snapshot()
            runtime.working_set.reconcile(bindings, revision=revision)
        resident_bindings = runtime.working_set.resident_bindings()
    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
        return False
    if resident_bindings and not global_policy.exit_when_idle:
        return False
    if project_io_executor is not None:
        # Every lane and the activation replay have to settle asynchronously.
        # Never replace a missing proof by a synchronous shared-root scan here.
        if resident_bindings or not runtime.activation_wake.is_current():
            return False
        try:
            _revision, registered = runtime.load_registry_snapshot()
            if registered and not global_policy.exit_when_idle:
                return False
            snapshot = reservation_snapshot(runtime.root)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            return False
        return not snapshot.active and not snapshot.provisional
    for binding in resident_bindings:
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
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            return False
    try:
        snapshot = reservation_snapshot(runtime.root)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
        return False
    return not snapshot.active and not snapshot.provisional


def confirm_idle_shutdown(
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
                dispatch_loop.dispatch_machine_cycle_locked(
                    runtime,
                    available_gpus=available_gpus,
                    executor=executor,
                    instance_id=instance_id,
                    heartbeat_interval_seconds=loop_interval,
                    started_at=started_at,
                    supervise=False,
                    publish_snapshots=False,
                )
        if not can_exit_when_idle(runtime, has_consumed_binding=has_consumed_binding):
            guard.__exit__(None, None, None)
            return None
        return guard
    except BaseException:
        guard.__exit__(*sys.exc_info())
        raise
