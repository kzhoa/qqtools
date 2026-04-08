from __future__ import annotations

import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .agent import wake_agent_if_needed
from .indexes import (
    load_index,
    rebuild_all_indexes,
    update_index_on_phase_change,
    update_index_on_submit,
)
from .layout import RootConfig, ensure_machine_layout, ensure_shared_layout
from .locking import submit_lock, batch_lock
from .models import (
    Batch,
    BatchPolicy,
    BatchSummary,
    Meta,
    PHASE_CANCELLED,
    PHASE_DISPATCHING,
    PHASE_FAILED,
    PHASE_QUEUED,
    PHASE_RUNNING,
    PHASE_STARTING,
    CANCELLABLE_PHASES,
    TERMINAL_PHASES,
    Task,
    TaskLineage,
    TaskResult,
    TaskRuntime,
    TaskSpec,
    TaskStatus,
    TaskTimestamps,
    generate_id,
    utc_now_iso,
    validate_task_id,
)
from .storage import (
    CASConflict,
    cas_update_batch,
    cas_update_task,
    load_batch,
    load_task,
    save_batch,
    save_task,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


def submit(
    cfg: RootConfig,
    command: list[str],
    requested_gpus: int = 1,
    task_id: str | None = None,
    name: str | None = None,
    batch_id: str | None = None,
) -> Task:
    if task_id is None:
        task_id = generate_id()
    validate_task_id(task_id)

    now = utc_now_iso()
    task = Task(
        meta=Meta.new(cfg.machine_name),
        task_id=task_id,
        name=name,
        batch_id=batch_id,
        machine_name=cfg.machine_name,
        attempt=1,
        spec=TaskSpec(command=list(command), requested_gpus=requested_gpus),
        status=TaskStatus(phase=PHASE_QUEUED),
        runtime=TaskRuntime(),
        timestamps=TaskTimestamps(created_at=now, queued_at=now),
        result=TaskResult(),
        lineage=TaskLineage(),
    )

    ensure_shared_layout(cfg)
    ensure_machine_layout(cfg)

    with submit_lock(cfg):
        save_task(cfg, task)
        update_index_on_submit(cfg, task)

    if not wake_agent_if_needed(cfg):
        import sys
        print(
            f"Warning: task {task_id} queued, but agent could not be started. "
            f"Run 'qexp agent start' manually or start an agent on this machine.",
            file=sys.stderr,
        )
    return task


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


def cancel(
    cfg: RootConfig,
    task_id: str,
    grace_seconds: float = 3.0,
    poll_interval: float = 0.1,
) -> Task:
    task = load_task(cfg, task_id)
    phase = task.status.phase

    if phase not in CANCELLABLE_PHASES:
        raise ValueError(
            f"Cannot cancel task {task_id} in phase {phase!r}. "
            f"Cancellable phases: {sorted(CANCELLABLE_PHASES)}."
        )

    old_phase = phase
    old_rev = task.meta.revision

    # If running with a live process, signal with escalation
    if phase == PHASE_RUNNING and task.runtime.wrapper_pid:
        _kill_with_escalation(
            task.runtime.wrapper_pid,
            grace_seconds=grace_seconds,
            poll_interval=poll_interval,
        )
        # Also signal the process group if available
        if task.runtime.process_group_id:
            try:
                os.killpg(task.runtime.process_group_id, signal.SIGTERM)
            except OSError:
                pass

    task.status.phase = PHASE_CANCELLED
    task.status.reason = "cancelled_by_user"
    task.timestamps.finished_at = utc_now_iso()
    task = cas_update_task(cfg, task, old_rev)
    update_index_on_phase_change(cfg, task_id, old_phase, PHASE_CANCELLED)

    from .events import write_event
    write_event(cfg, "task_cancelled", task_id=task_id)

    return task


def _kill_with_escalation(
    pid: int,
    grace_seconds: float = 3.0,
    poll_interval: float = 0.1,
) -> None:
    """SIGTERM → wait grace → SIGKILL if still alive."""
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return

    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return  # Process exited
        time.sleep(poll_interval)

    # Still alive after grace period: escalate to SIGKILL
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------


def retry(cfg: RootConfig, task_id: str) -> Task:
    original = load_task(cfg, task_id)

    if original.status.phase not in TERMINAL_PHASES:
        raise ValueError(
            f"Cannot retry task {task_id} in phase {original.status.phase!r}. "
            f"Only terminal tasks can be retried."
        )

    new_id = generate_id()
    now = utc_now_iso()

    new_task = Task(
        meta=Meta.new(cfg.machine_name),
        task_id=new_id,
        name=original.name,
        batch_id=original.batch_id,
        machine_name=cfg.machine_name,
        attempt=original.attempt + 1,
        spec=TaskSpec(
            command=list(original.spec.command),
            requested_gpus=original.spec.requested_gpus,
        ),
        status=TaskStatus(phase=PHASE_QUEUED),
        runtime=TaskRuntime(),
        timestamps=TaskTimestamps(created_at=now, queued_at=now),
        result=TaskResult(),
        lineage=TaskLineage(retry_of=original.task_id),
    )

    with submit_lock(cfg):
        save_task(cfg, new_task)
        update_index_on_submit(cfg, new_task)

    # Update batch if applicable
    if original.batch_id:
        _add_task_to_batch(cfg, original.batch_id, new_id)

    wake_agent_if_needed(cfg)
    return new_task


def _add_task_to_batch(cfg: RootConfig, batch_id: str, task_id: str) -> None:
    try:
        batch = load_batch(cfg, batch_id)
    except FileNotFoundError:
        return
    batch.task_ids.append(task_id)
    try:
        cas_update_batch(cfg, batch, batch.meta.revision)
    except CASConflict:
        # Best-effort; batch summary can be rebuilt
        pass


# ---------------------------------------------------------------------------
# Batch submit
# ---------------------------------------------------------------------------


def batch_submit(
    cfg: RootConfig,
    manifest_path: Path,
) -> Batch:
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    batch_section = raw.get("batch", {})
    defaults = raw.get("defaults", {})
    tasks_section = raw.get("tasks", [])

    if not tasks_section:
        raise ValueError("Manifest must contain at least one task.")

    batch_id = generate_id()
    now = utc_now_iso()
    task_ids: list[str] = []

    ensure_shared_layout(cfg)
    ensure_machine_layout(cfg)

    with batch_lock(cfg):
        for entry in tasks_section:
            command = entry.get("command")
            if not command:
                raise ValueError("Each task in the manifest must have a 'command'.")

            gpus = entry.get("requested_gpus", defaults.get("requested_gpus", 1))
            task_name = entry.get("name")
            tid = entry.get("task_id", generate_id())

            task = Task(
                meta=Meta.new(cfg.machine_name),
                task_id=tid,
                name=task_name,
                batch_id=batch_id,
                machine_name=cfg.machine_name,
                attempt=1,
                spec=TaskSpec(command=list(command), requested_gpus=gpus),
                status=TaskStatus(phase=PHASE_QUEUED),
                runtime=TaskRuntime(),
                timestamps=TaskTimestamps(created_at=now, queued_at=now),
                result=TaskResult(),
                lineage=TaskLineage(),
            )
            save_task(cfg, task)
            update_index_on_submit(cfg, task)
            task_ids.append(tid)

        policy_raw = batch_section.get("policy", {})
        batch = Batch(
            meta=Meta.new(cfg.machine_name),
            batch_id=batch_id,
            name=batch_section.get("name"),
            source_manifest=str(manifest_path),
            machine_name=cfg.machine_name,
            task_ids=task_ids,
            summary=BatchSummary(total=len(task_ids), queued=len(task_ids)),
            policy=BatchPolicy(
                allow_retry_failed=policy_raw.get("allow_retry_failed", True),
                allow_retry_cancelled=policy_raw.get("allow_retry_cancelled", True),
            ),
        )
        save_batch(cfg, batch)

    wake_agent_if_needed(cfg)
    return batch


# ---------------------------------------------------------------------------
# Batch retry helpers
# ---------------------------------------------------------------------------


def batch_retry_failed(cfg: RootConfig, batch_id: str) -> list[Task]:
    batch = load_batch(cfg, batch_id)
    if not batch.policy.allow_retry_failed:
        raise ValueError(
            f"Batch {batch_id} policy disallows retrying failed tasks."
        )
    return _batch_retry_by_phase(cfg, batch, PHASE_FAILED)


def batch_retry_cancelled(cfg: RootConfig, batch_id: str) -> list[Task]:
    batch = load_batch(cfg, batch_id)
    if not batch.policy.allow_retry_cancelled:
        raise ValueError(
            f"Batch {batch_id} policy disallows retrying cancelled tasks."
        )
    return _batch_retry_by_phase(cfg, batch, PHASE_CANCELLED)


def _batch_retry_by_phase(
    cfg: RootConfig, batch: Batch, target_phase: str
) -> list[Task]:
    new_tasks: list[Task] = []

    for tid in list(batch.task_ids):
        try:
            task = load_task(cfg, tid)
        except FileNotFoundError:
            continue
        if task.status.phase == target_phase:
            new_task = retry(cfg, tid)
            new_tasks.append(new_task)

    return new_tasks


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------

DEFAULT_CLEAN_OLDER_THAN_SECONDS = 7 * 24 * 3600  # 7 days


def get_log_path(cfg: RootConfig, task_id: str) -> Path:
    from .layout import runtime_log_path
    return runtime_log_path(cfg, task_id)


def read_logs(cfg: RootConfig, task_id: str) -> str:
    log_path = get_log_path(cfg, task_id)
    if not log_path.is_file():
        raise FileNotFoundError(f"Log file for task {task_id} not found at {log_path}.")
    return log_path.read_text(encoding="utf-8", errors="replace")


def tail_log(cfg: RootConfig, task_id: str) -> None:
    """Follow log file output (blocking). Ctrl-C to stop."""
    log_path = get_log_path(cfg, task_id)
    if not log_path.is_file():
        raise FileNotFoundError(f"Log file for task {task_id} not found at {log_path}.")

    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        while True:
            chunk = fh.read()
            if chunk:
                sys.stdout.write(chunk)
                sys.stdout.flush()
                continue
            time.sleep(0.5)


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------


def clean(
    cfg: RootConfig,
    dry_run: bool = False,
    include_failed: bool = False,
    older_than_seconds: int = DEFAULT_CLEAN_OLDER_THAN_SECONDS,
) -> dict[str, Any]:
    """Remove old terminal task files and their logs.

    Only cleans succeeded (and optionally failed/cancelled) tasks
    older than the threshold.
    """
    from .layout import global_tasks_dir, runtime_log_path
    from .models import PHASE_SUCCEEDED, PHASE_FAILED as PF, PHASE_CANCELLED as PC
    from .storage import iter_all_tasks, delete_task_file

    phases_to_clean = {PHASE_SUCCEEDED}
    if include_failed:
        phases_to_clean.add(PF)
        phases_to_clean.add(PC)

    cutoff = datetime.now(timezone.utc).timestamp() - older_than_seconds
    deleted_tasks: list[str] = []
    deleted_logs: list[str] = []

    for task in iter_all_tasks(cfg):
        if task.status.phase not in phases_to_clean:
            continue

        finished = task.timestamps.finished_at
        if finished is None:
            continue

        try:
            dt = datetime.fromisoformat(finished.replace("Z", "+00:00"))
            if dt.timestamp() > cutoff:
                continue
        except Exception:
            continue

        if dry_run:
            deleted_tasks.append(task.task_id)
        else:
            delete_task_file(cfg, task.task_id)
            deleted_tasks.append(task.task_id)

        lp = runtime_log_path(cfg, task.task_id)
        if lp.is_file():
            if not dry_run:
                lp.unlink()
            deleted_logs.append(str(lp))

    # Rebuild indexes after cleaning
    if not dry_run and deleted_tasks:
        from .indexes import rebuild_all_indexes
        rebuild_all_indexes(cfg)

    return {
        "dry_run": dry_run,
        "task_ids": deleted_tasks,
        "deleted_task_count": len(deleted_tasks),
        "deleted_log_count": len(deleted_logs),
        "deleted_log_files": deleted_logs,
    }
