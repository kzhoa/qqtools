"""qexp scheduler — dispatch queued tasks and reconcile running ones."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .events import write_event
from .executor import Executor
from .indexes import load_index, update_index_on_phase_change
from .layout import RootConfig, machine_claims_active_dir
from .models import (
    ACTIVE_PHASES,
    PHASE_DISPATCHING,
    PHASE_FAILED,
    PHASE_ORPHANED,
    PHASE_QUEUED,
    PHASE_RUNNING,
    PHASE_STARTING,
    utc_now_iso,
    validate_phase_transition,
)
from .storage import (
    CASConflict,
    cas_update_task,
    load_task,
    read_json,
    release_claim,
    save_claim,
)

log = logging.getLogger(__name__)

DEFAULT_STARTUP_GRACE_SECONDS = 30


def _is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@dataclass(slots=True)
class Scheduler:
    """Full scheduler with GPU allocation and tmux execution."""

    tracker: Any
    executor: Executor = field(default_factory=Executor)
    startup_grace_seconds: int = DEFAULT_STARTUP_GRACE_SECONDS
    process_alive_check: Callable[[int], bool] = _is_process_alive

    def run_dispatch_cycle(self, cfg: RootConfig) -> list[str]:
        """Dispatch queued tasks for this machine.

        Full flow per task: queued → dispatching → GPU alloc → starting → tmux launch.
        Returns list of launched task_ids.
        """
        queued_ids = load_index(cfg, "state", PHASE_QUEUED)
        if not queued_ids:
            return []

        launched: list[str] = []
        for task_id in queued_ids:
            try:
                task = load_task(cfg, task_id)
            except FileNotFoundError:
                continue

            if task.status.phase != PHASE_QUEUED:
                continue
            if task.machine_name != cfg.machine_name:
                continue

            # CAS: queued → dispatching
            old_rev = task.meta.revision
            try:
                task.status.phase = PHASE_DISPATCHING
                task = cas_update_task(cfg, task, old_rev)
                update_index_on_phase_change(cfg, task_id, PHASE_QUEUED, PHASE_DISPATCHING)
            except CASConflict:
                log.debug("CAS conflict dispatching %s, skipping.", task_id)
                continue

            # GPU allocation
            assigned = self.tracker.allocate(task_id, task.spec.requested_gpus)
            if assigned is None:
                # Rollback: dispatching → queued
                task.status.phase = PHASE_QUEUED
                try:
                    cas_update_task(cfg, task, task.meta.revision)
                    update_index_on_phase_change(cfg, task_id, PHASE_DISPATCHING, PHASE_QUEUED)
                except CASConflict:
                    pass
                continue
            task.runtime.assigned_gpus = assigned

            # CAS: dispatching → starting
            old_rev = task.meta.revision
            try:
                task.status.phase = PHASE_STARTING
                task.timestamps.started_at = utc_now_iso()
                task = cas_update_task(cfg, task, old_rev)
                update_index_on_phase_change(cfg, task_id, PHASE_DISPATCHING, PHASE_STARTING)
            except CASConflict:
                self.tracker.release(task_id)
                continue

            # Write claim
            save_claim(cfg, task_id, utc_now_iso(), task.meta.revision)

            # Launch in tmux via executor
            try:
                window_id = self.executor.launch_task(cfg, task_id, task.group)
                log.info(
                    "Launched task %s in window %s with GPUs %s (group=%s).",
                    task_id, window_id, assigned, task.group,
                )
                launched.append(task_id)
            except Exception:
                log.exception("Failed to launch task %s in tmux.", task_id)
                self.tracker.release(task_id)
                # Mark as failed
                task.status.phase = PHASE_FAILED
                task.status.reason = "tmux_launch_failed"
                task.timestamps.finished_at = utc_now_iso()
                try:
                    cas_update_task(cfg, task, task.meta.revision)
                    update_index_on_phase_change(cfg, task_id, PHASE_STARTING, PHASE_FAILED)
                except CASConflict:
                    pass
                write_event(cfg, "task_failed", task_id=task_id, details={
                    "reason": "tmux_launch_failed",
                })
                release_claim(cfg, task_id, "tmux_launch_failed")

        return launched

    def reconcile_running_tasks(self, cfg: RootConfig) -> list[str]:
        """Check tasks in active phases for liveness.

        For each active task owned by this machine:
        - If it has a wrapper_pid: check if the process is alive
        - If it has no wrapper_pid and startup grace exceeded: mark failed
        - Optionally check tmux window existence

        Returns list of task_ids that were failed/orphaned.
        """
        failed_ids: list[str] = []

        for phase in (PHASE_DISPATCHING, PHASE_STARTING, PHASE_RUNNING):
            task_ids = load_index(cfg, "state", phase)
            for task_id in task_ids:
                try:
                    task = load_task(cfg, task_id)
                except FileNotFoundError:
                    continue
                if task.machine_name != cfg.machine_name:
                    continue

                should_fail = False
                fail_reason = ""

                if phase == PHASE_RUNNING:
                    # Check wrapper_pid liveness
                    if task.runtime.wrapper_pid:
                        if not self.process_alive_check(task.runtime.wrapper_pid):
                            should_fail = True
                            fail_reason = "wrapper_crashed"
                    else:
                        # Running but no PID recorded yet — grace period
                        if self._startup_grace_exceeded(task):
                            should_fail = True
                            fail_reason = "no_wrapper_pid"

                elif phase in (PHASE_DISPATCHING, PHASE_STARTING):
                    # Check if startup grace exceeded
                    if self._startup_grace_exceeded(task):
                        should_fail = True
                        fail_reason = "startup_timeout"

                if should_fail:
                    old_phase = task.status.phase
                    task.status.phase = PHASE_FAILED
                    task.status.reason = fail_reason
                    task.timestamps.finished_at = utc_now_iso()
                    try:
                        cas_update_task(cfg, task, task.meta.revision)
                        update_index_on_phase_change(cfg, task_id, old_phase, PHASE_FAILED)
                    except CASConflict:
                        continue
                    self.tracker.release(task_id)
                    release_claim(cfg, task_id, fail_reason)
                    write_event(cfg, "task_failed", task_id=task_id, details={
                        "reason": fail_reason,
                    })
                    failed_ids.append(task_id)
                    log.warning(
                        "Reconciled task %s: %s (was %s).",
                        task_id, fail_reason, phase,
                    )

        return failed_ids

    def _startup_grace_exceeded(self, task: Any) -> bool:
        """Check if task has been in pre-running state beyond grace period."""
        started_at = task.timestamps.started_at
        if not started_at:
            # Use created_at as fallback
            started_at = task.timestamps.created_at
        try:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            elapsed = (datetime.now(timezone.utc) - dt).total_seconds()
            return elapsed > self.startup_grace_seconds
        except Exception:
            return True


# ---------------------------------------------------------------------------
# Standalone functions used by the agent loop
# ---------------------------------------------------------------------------


def run_dispatch_cycle(
    cfg: RootConfig,
    tracker: Any = None,
) -> list[str]:
    """Standalone dispatch function.

    When tracker is None, runs a simplified dispatch without GPU allocation
    or tmux (useful for testing). When tracker is provided, creates a full
    Scheduler and runs with executor.
    """
    if tracker is None:
        return _dispatch_without_executor(cfg)

    scheduler = Scheduler(tracker=tracker)
    return scheduler.run_dispatch_cycle(cfg)


def _dispatch_without_executor(cfg: RootConfig) -> list[str]:
    """Simplified dispatch: queued → dispatching → starting, no GPU or tmux.

    Used for unit testing and dry-run scenarios.
    """
    queued_ids = load_index(cfg, "state", PHASE_QUEUED)
    if not queued_ids:
        return []

    launched: list[str] = []
    for task_id in queued_ids:
        try:
            task = load_task(cfg, task_id)
        except FileNotFoundError:
            continue
        if task.status.phase != PHASE_QUEUED:
            continue
        if task.machine_name != cfg.machine_name:
            continue

        old_rev = task.meta.revision
        try:
            task.status.phase = PHASE_DISPATCHING
            task = cas_update_task(cfg, task, old_rev)
            update_index_on_phase_change(cfg, task_id, PHASE_QUEUED, PHASE_DISPATCHING)
        except CASConflict:
            continue

        old_rev = task.meta.revision
        try:
            task.status.phase = PHASE_STARTING
            task.timestamps.started_at = utc_now_iso()
            task = cas_update_task(cfg, task, old_rev)
            update_index_on_phase_change(cfg, task_id, PHASE_DISPATCHING, PHASE_STARTING)
        except CASConflict:
            continue

        save_claim(cfg, task_id, utc_now_iso(), task.meta.revision)
        launched.append(task_id)

    return launched


def reconcile_running_tasks(cfg: RootConfig) -> list[str]:
    """Standalone reconcile — no-op when called without a Scheduler."""
    return []
