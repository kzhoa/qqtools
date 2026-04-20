from __future__ import annotations

import logging
import os
import signal
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .agent import wake_agent_if_needed
from .batch_state import build_batch_summary_from_counts, collect_batch_task_counts
from .doctor import repair_metadata
from .indexes import (
    load_index,
    rebuild_all_indexes,
    remove_index_on_delete,
    update_batch_index_on_create,
    update_index_on_phase_change,
    update_index_on_submit,
)
from .layout import (
    RootConfig,
    ensure_machine_layout,
    ensure_shared_layout,
    read_root_manifest,
    validate_root_contract,
)
from .locking import batch_lock, clean_lock, submit_lock
from .locking import task_operation_lock
from .models import (
    Batch,
    BATCH_COMMIT_ABORTED,
    BATCH_COMMIT_COMMITTED,
    BATCH_COMMIT_PREPARING,
    BatchPolicy,
    BatchSummary,
    Meta,
    PHASE_CANCELLED,
    PHASE_DISPATCHING,
    PHASE_FAILED,
    PHASE_QUEUED,
    PHASE_RUNNING,
    PHASE_SUCCEEDED,
    PHASE_STARTING,
    CANCELLABLE_PHASES,
    TERMINAL_PHASES,
    Task,
    TaskLineage,
    ResubmitOperation,
    ResubmitNewSubmission,
    ResubmitOldTaskSummary,
    RESUBMIT_STATE_ABORTED,
    RESUBMIT_STATE_COMMITTED,
    RESUBMIT_STATE_CREATING_NEW,
    RESUBMIT_STATE_DELETING_OLD,
    RESUBMIT_STATE_PREPARING,
    TaskResult,
    TaskRuntime,
    TaskSpec,
    TaskStatus,
    TaskTimestamps,
    generate_id,
    validate_group_name,
    utc_now_iso,
    validate_task_id,
)
from .storage import (
    CASConflict,
    cas_update_batch,
    cas_update_task,
    delete_resubmit_operation,
    delete_task_file,
    load_resubmit_operation,
    load_batch,
    load_machine,
    load_task,
    save_resubmit_operation,
    save_batch,
    save_task,
    iter_all_tasks,
)

log = logging.getLogger(__name__)

RESUBMIT_ALLOWED_PHASES = frozenset({PHASE_FAILED, PHASE_CANCELLED})


def _create_task(
    *,
    cfg: RootConfig,
    command: list[str],
    requested_gpus: int,
    task_id: str,
    name: str | None,
    batch_id: str | None,
    group: str | None,
    machine_name: str,
    attempt: int,
    lineage: TaskLineage | None = None,
) -> Task:
    now = utc_now_iso()
    return Task(
        meta=Meta.new(machine_name),
        task_id=task_id,
        name=name,
        group=group,
        batch_id=batch_id,
        machine_name=machine_name,
        attempt=attempt,
        spec=TaskSpec(command=list(command), requested_gpus=requested_gpus),
        status=TaskStatus(phase=PHASE_QUEUED),
        runtime=TaskRuntime(),
        timestamps=TaskTimestamps(created_at=now, queued_at=now),
        result=TaskResult(),
        lineage=lineage or TaskLineage(),
    )


def _ensure_no_pending_resubmit(cfg: RootConfig, task_id: str) -> None:
    try:
        operation = load_resubmit_operation(cfg, task_id)
    except FileNotFoundError:
        return
    if operation.state not in {RESUBMIT_STATE_COMMITTED, RESUBMIT_STATE_ABORTED}:
        raise RuntimeError(
            f"Task {task_id} has an unfinished resubmit operation in state "
            f"{operation.state!r}. Run 'qexp doctor repair' to converge it."
        )


def _advance_resubmit_operation(
    cfg: RootConfig,
    operation: ResubmitOperation,
    state: str,
) -> ResubmitOperation:
    operation.state = state
    operation.meta.revision += 1
    operation.meta.updated_at = utc_now_iso()
    operation.meta.updated_by_machine = cfg.machine_name
    save_resubmit_operation(cfg, operation)
    return operation


def _build_resubmit_operation(
    cfg: RootConfig,
    task: Task,
    *,
    command: list[str],
    requested_gpus: int,
    name: str | None,
    group: str | None,
) -> ResubmitOperation:
    prepared_task = _create_task(
        cfg=cfg,
        command=command,
        requested_gpus=requested_gpus,
        task_id=task.task_id,
        name=name,
        batch_id=None,
        group=group,
        machine_name=cfg.machine_name,
        attempt=1,
    )
    return ResubmitOperation(
        meta=Meta.new(cfg.machine_name),
        operation_type="resubmit",
        task_id=task.task_id,
        state=RESUBMIT_STATE_PREPARING,
        old_task_snapshot_path=f"global/tasks/{task.task_id}.json",
        new_submission=ResubmitNewSubmission(
            command=list(command),
            requested_gpus=requested_gpus,
            name=name,
            group=group,
            machine_name=cfg.machine_name,
        ),
        new_task_snapshot=prepared_task.to_dict(),
        old_task_summary=ResubmitOldTaskSummary(
            phase=task.status.phase,
            machine_name=task.machine_name,
            attempt=task.attempt,
            batch_id=task.batch_id,
            name=task.name,
            group=task.group,
        ),
    )


def _best_effort_delete_task_log(cfg: RootConfig, task: Task) -> None:
    try:
        _delete_task_log(_resolve_task_log_path(cfg, task))
    except Exception:
        log.warning("Failed to remove runtime log for task %s.", task.task_id, exc_info=True)


def _delete_task_truth(cfg: RootConfig, task: Task) -> None:
    delete_task_file(cfg, task.task_id)
    try:
        remove_index_on_delete(cfg, task)
    except Exception:
        log.warning("Failed to update indexes while deleting task %s.", task.task_id, exc_info=True)
    _best_effort_delete_task_log(cfg, task)


def _persist_submitted_task_truth(cfg: RootConfig, task: Task) -> None:
    save_task(cfg, task)
    try:
        update_index_on_submit(cfg, task)
    except Exception:
        log.warning("Failed to update indexes while creating task %s.", task.task_id, exc_info=True)


def _materialize_resubmit_task(operation: ResubmitOperation) -> Task:
    return Task.from_dict(operation.new_task_snapshot)


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
    group: str | None = None,
) -> Task:
    validate_root_contract(cfg)
    if task_id is None:
        task_id = generate_id()
    validate_task_id(task_id)
    group = validate_group_name(group)
    task = _create_task(
        cfg=cfg,
        command=command,
        requested_gpus=requested_gpus,
        task_id=task_id,
        name=name,
        batch_id=batch_id,
        group=group,
        machine_name=cfg.machine_name,
        attempt=1,
    )

    ensure_shared_layout(cfg)
    ensure_machine_layout(cfg)

    with submit_lock(cfg):
        _ensure_no_pending_resubmit(cfg, task_id)
        try:
            load_task(cfg, task_id)
        except FileNotFoundError:
            pass
        else:
            raise ValueError(f"Task {task_id!r} already exists.")
        _persist_submitted_task_truth(cfg, task)

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
    with task_operation_lock(cfg, task_id):
        _ensure_no_pending_resubmit(cfg, task_id)
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


def retry(cfg: RootConfig, task_id: str, group: str | None = None) -> Task:
    validate_root_contract(cfg)
    with task_operation_lock(cfg, task_id):
        _ensure_no_pending_resubmit(cfg, task_id)
        original = load_task(cfg, task_id)

        if original.status.phase not in TERMINAL_PHASES:
            raise ValueError(
                f"Cannot retry task {task_id} in phase {original.status.phase!r}. "
                f"Only terminal tasks can be retried."
            )

        resolved_group = original.group if group is None else validate_group_name(group)

        new_id = generate_id()
        new_task = _create_task(
            cfg=cfg,
            command=original.spec.command,
            requested_gpus=original.spec.requested_gpus,
            task_id=new_id,
            name=original.name,
            batch_id=original.batch_id,
            group=resolved_group,
            machine_name=cfg.machine_name,
            attempt=original.attempt + 1,
            lineage=TaskLineage(retry_of=original.task_id),
        )

        with submit_lock(cfg):
            _persist_submitted_task_truth(cfg, new_task)

    # Update batch if applicable
    if original.batch_id:
        _add_task_to_batch(cfg, original.batch_id, new_id)

    wake_agent_if_needed(cfg)
    return new_task


def resubmit(
    cfg: RootConfig,
    task_id: str,
    *,
    command: list[str],
    requested_gpus: int | None = None,
    name: str | None = None,
    group: str | None = None,
) -> Task:
    validate_root_contract(cfg)
    validate_task_id(task_id)

    with task_operation_lock(cfg, task_id):
        operation_persisted = False
        _ensure_no_pending_resubmit(cfg, task_id)
        original = load_task(cfg, task_id)

        if original.status.phase not in RESUBMIT_ALLOWED_PHASES:
            raise ValueError(
                f"Cannot resubmit task {task_id} in phase {original.status.phase!r}. "
                f"Allowed phases: {sorted(RESUBMIT_ALLOWED_PHASES)}."
            )
        if original.batch_id is not None:
            raise ValueError(
                f"Cannot resubmit batch member task {task_id}. "
                "Batch task resubmit is not supported."
            )

        resolved_group = original.group if group is None else validate_group_name(group)
        resolved_name = original.name if name is None else name
        resolved_gpus = (
            original.spec.requested_gpus if requested_gpus is None else requested_gpus
        )
        TaskSpec(command=list(command), requested_gpus=resolved_gpus)
        operation = _build_resubmit_operation(
            cfg,
            original,
            command=command,
            requested_gpus=resolved_gpus,
            name=resolved_name,
            group=resolved_group,
        )

        ensure_shared_layout(cfg)
        ensure_machine_layout(cfg)

        with submit_lock(cfg):
            try:
                save_resubmit_operation(cfg, operation)
                operation_persisted = True
                from .events import write_event

                write_event(
                    cfg,
                    "resubmit_started",
                    task_id=task_id,
                    details={"state": operation.state},
                )

                operation = _advance_resubmit_operation(
                    cfg, operation, RESUBMIT_STATE_DELETING_OLD
                )
                _delete_task_truth(cfg, original)

                operation = _advance_resubmit_operation(
                    cfg, operation, RESUBMIT_STATE_CREATING_NEW
                )
                new_task = _materialize_resubmit_task(operation)
                _persist_submitted_task_truth(cfg, new_task)
                _advance_resubmit_operation(cfg, operation, RESUBMIT_STATE_COMMITTED)
                delete_resubmit_operation(cfg, task_id)
                write_event(
                    cfg,
                    "resubmit_committed",
                    task_id=task_id,
                    details={"replaced_phase": original.status.phase},
                )
            except Exception as exc:
                if operation_persisted:
                    raise RuntimeError(
                        f"Resubmit for task {task_id} did not converge. "
                        "An unfinished resubmit operation was recorded. "
                        "Run 'qexp doctor repair' to continue convergence."
                    ) from exc
                raise

    if not wake_agent_if_needed(cfg):
        print(
            f"Warning: task {task_id} resubmitted, but agent could not be started. "
            f"Run 'qexp agent start' manually or start an agent on this machine.",
            file=sys.stderr,
        )
    return new_task


def _add_task_to_batch(cfg: RootConfig, batch_id: str, task_id: str) -> None:
    for _ in range(5):
        try:
            batch = load_batch(cfg, batch_id)
        except FileNotFoundError:
            return
        _ensure_batch_committed(batch)
        if task_id in batch.task_ids:
            return

        batch.task_ids.append(task_id)
        batch.expected_task_count = max(batch.expected_task_count, len(batch.task_ids))
        batch.summary = _build_batch_summary(cfg, batch.task_ids)
        try:
            cas_update_batch(cfg, batch, batch.meta.revision)
            return
        except CASConflict:
            continue

    raise RuntimeError(f"Failed to update batch {batch_id} after repeated CAS conflicts.")


# ---------------------------------------------------------------------------
# Batch submit
# ---------------------------------------------------------------------------


def batch_submit(
    cfg: RootConfig,
    manifest_path: Path,
) -> Batch:
    validate_root_contract(cfg)
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    batch_section = raw.get("batch", {})
    defaults = raw.get("defaults", {})
    tasks_section = raw.get("tasks", [])

    if not tasks_section:
        raise ValueError("Manifest must contain at least one task.")

    batch_id = generate_id()
    now = utc_now_iso()
    batch_group = validate_group_name(batch_section.get("group"))
    task_specs = _prepare_batch_task_specs(
        cfg,
        batch_id=batch_id,
        now=now,
        defaults=defaults,
        tasks_section=tasks_section,
        batch_group=batch_group,
    )

    ensure_shared_layout(cfg)
    ensure_machine_layout(cfg)

    with batch_lock(cfg):
        policy_raw = batch_section.get("policy", {})
        persisted_tasks: list[Task] = []
        batch = Batch(
            meta=Meta.new(cfg.machine_name),
            batch_id=batch_id,
            name=batch_section.get("name"),
            group=batch_group,
            source_manifest=str(manifest_path),
            machine_name=cfg.machine_name,
            commit_state=BATCH_COMMIT_PREPARING,
            expected_task_count=len(task_specs),
            task_ids=[],
            summary=BatchSummary(total=len(task_specs)),
            policy=BatchPolicy(
                allow_retry_failed=policy_raw.get("allow_retry_failed", True),
                allow_retry_cancelled=policy_raw.get("allow_retry_cancelled", True),
            ),
        )
        save_batch(cfg, batch)
        try:
            for task in task_specs:
                save_task(cfg, task)
                persisted_tasks.append(task)
                batch.task_ids.append(task.task_id)
                update_index_on_submit(cfg, task)

            batch.commit_state = BATCH_COMMIT_COMMITTED
            batch.summary = _build_batch_summary(cfg, batch.task_ids)
            save_batch(cfg, batch)
            update_batch_index_on_create(cfg, batch)
        except Exception:
            for task in persisted_tasks:
                remove_index_on_delete(cfg, task)
                delete_task_file(cfg, task.task_id)
            batch.task_ids = []
            batch.commit_state = BATCH_COMMIT_ABORTED
            batch.expected_task_count = 0
            batch.summary = BatchSummary()
            save_batch(cfg, batch)
            raise

    wake_agent_if_needed(cfg)
    return batch


# ---------------------------------------------------------------------------
# Batch retry helpers
# ---------------------------------------------------------------------------


def batch_retry_failed(cfg: RootConfig, batch_id: str) -> list[Task]:
    validate_root_contract(cfg)
    batch = load_batch(cfg, batch_id)
    _ensure_batch_committed(batch)
    if not batch.policy.allow_retry_failed:
        raise ValueError(
            f"Batch {batch_id} policy disallows retrying failed tasks."
        )
    return _batch_retry_by_phase(cfg, batch, PHASE_FAILED)


def batch_retry_cancelled(cfg: RootConfig, batch_id: str) -> list[Task]:
    validate_root_contract(cfg)
    batch = load_batch(cfg, batch_id)
    _ensure_batch_committed(batch)
    if not batch.policy.allow_retry_cancelled:
        raise ValueError(
            f"Batch {batch_id} policy disallows retrying cancelled tasks."
        )
    return _batch_retry_by_phase(cfg, batch, PHASE_CANCELLED)


def _batch_retry_by_phase(
    cfg: RootConfig, batch: Batch, target_phase: str
) -> list[Task]:
    _ensure_batch_committed(batch)
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


@dataclass(slots=True)
class _ResolvedLogPath:
    task_id: str
    status: str
    path: str | None
    message: str | None = None


def _parse_iso_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _build_batch_summary(cfg: RootConfig, task_ids: list[str]) -> BatchSummary:
    surviving_ids, counts = collect_batch_task_counts(cfg, task_ids)
    return build_batch_summary_from_counts(total=len(surviving_ids), counts=counts)


def _ensure_batch_committed(batch: Batch) -> None:
    if batch.commit_state != BATCH_COMMIT_COMMITTED:
        raise ValueError(
            f"Batch {batch.batch_id} is not committed. "
            f"Current commit_state={batch.commit_state!r}."
        )


def _prepare_batch_task_specs(
    cfg: RootConfig,
    *,
    batch_id: str,
    now: str,
    defaults: dict[str, Any],
    tasks_section: list[dict[str, Any]],
    batch_group: str | None,
) -> list[Task]:
    prepared_tasks: list[Task] = []
    seen_task_ids: set[str] = set()

    for entry in tasks_section:
        command = entry.get("command")
        if not command:
            raise ValueError("Each task in the manifest must have a 'command'.")

        tid = entry.get("task_id", generate_id())
        validate_task_id(tid)
        if tid in seen_task_ids:
            raise ValueError(f"Duplicate task_id in manifest: {tid!r}.")
        try:
            load_task(cfg, tid)
        except FileNotFoundError:
            pass
        else:
            raise ValueError(f"Task {tid!r} already exists.")

        seen_task_ids.add(tid)
        gpus = entry.get("requested_gpus", defaults.get("requested_gpus", 1))
        prepared_tasks.append(
            Task(
                meta=Meta.new(cfg.machine_name),
                task_id=tid,
                name=entry.get("name"),
                group=validate_group_name(entry.get("group", batch_group)),
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
        )
    return prepared_tasks


def _resolve_task_log_path(cfg: RootConfig, task: Task) -> _ResolvedLogPath:
    from .layout import runtime_log_path

    try:
        machine = load_machine(cfg, task.machine_name)
    except FileNotFoundError as exc:
        return _ResolvedLogPath(
            task_id=task.task_id,
            status="unresolved",
            path=None,
            message=str(exc),
        )

    machine_cfg = RootConfig(
        shared_root=cfg.shared_root,
        project_root=cfg.project_root,
        machine_name=machine.machine_name,
        runtime_root=Path(machine.runtime_root),
    )
    path = runtime_log_path(machine_cfg, task.task_id)
    return _ResolvedLogPath(
        task_id=task.task_id,
        status="resolved",
        path=str(path),
    )


def get_log_path(cfg: RootConfig, task_id: str) -> Path:
    task = load_task(cfg, task_id)
    resolved = _resolve_task_log_path(cfg, task)
    if resolved.path is None:
        raise FileNotFoundError(
            resolved.message or f"Cannot resolve log path for task {task_id}."
        )
    return Path(resolved.path)


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
    task_id: str | None = None,
    include_failed: bool = False,
    older_than_seconds: int | None = None,
) -> dict[str, Any]:
    validate_root_contract(cfg)
    """Clean terminal task records.

    Two modes are supported:
    - batch clean: select old terminal tasks by age/phase filters
    - single-task clean: delete one explicit terminal task and repair batch truth
    """
    if task_id is not None and include_failed:
        raise ValueError("--task-id cannot be combined with --include-failed.")

    if older_than_seconds is None:
        manifest = read_root_manifest(cfg)
        older_than_seconds = (
            manifest.lifecycle_policy.clean_after_seconds
            if manifest is not None
            else DEFAULT_CLEAN_OLDER_THAN_SECONDS
        )

    mode = "single_task" if task_id is not None else "batch"

    should_repair = False
    try:
        with clean_lock(cfg):
            task_lock = (
                task_operation_lock(cfg, task_id)
                if mode == "single_task" and task_id
                else nullcontext()
            )
            with task_lock:
                if mode == "single_task" and task_id is not None:
                    _ensure_no_pending_resubmit(cfg, task_id)
                    candidates = [_load_single_clean_candidate(cfg, task_id)]
                else:
                    candidates = _select_batch_clean_candidates(
                        cfg,
                        include_failed=include_failed,
                        older_than_seconds=older_than_seconds,
                    )

                planned_task_ids = [task.task_id for task in candidates]
                log_plans = [_resolve_task_log_path(cfg, task) for task in candidates]

                batch_updates = _plan_batch_updates(cfg, candidates)

                result: dict[str, Any] = {
                    "dry_run": dry_run,
                    "mode": mode,
                    "task_ids": planned_task_ids,
                    "deleted_task_count": len(planned_task_ids) if not dry_run else 0,
                    "planned_task_count": len(planned_task_ids),
                    "deleted_log_count": 0,
                    "deleted_log_files": [],
                    "repaired_batches": sorted(batch_updates.keys()),
                    "log_results": [asdict(plan) for plan in log_plans],
                }

                if dry_run:
                    return result

                deleted_task_ids: list[str] = []
                deleted_log_files: list[str] = []
                log_results: list[dict[str, Any]] = []
                mutations_started = False

                _apply_batch_updates(cfg, batch_updates)
                mutations_started = bool(batch_updates)
                should_repair = mutations_started

                for task in candidates:
                    delete_task_file(cfg, task.task_id)
                    deleted_task_ids.append(task.task_id)
                    mutations_started = True
                    should_repair = True

                rebuild_all_indexes(cfg)

                for resolved in log_plans:
                    log_result = _delete_task_log(resolved)
                    log_results.append(log_result)
                    if log_result["status"] == "deleted" and log_result["path"]:
                        deleted_log_files.append(log_result["path"])

                result.update(
                    {
                        "task_ids": deleted_task_ids,
                        "deleted_task_count": len(deleted_task_ids),
                        "deleted_log_count": len(deleted_log_files),
                        "deleted_log_files": deleted_log_files,
                        "log_results": log_results,
                    }
                )
                return result
    except Exception:
        if should_repair:
            repair_metadata(cfg)
        raise


def _load_single_clean_candidate(cfg: RootConfig, task_id: str) -> Task:
    task = load_task(cfg, task_id)
    if task.status.phase not in TERMINAL_PHASES:
        raise ValueError(
            f"Cannot clean task {task_id} in phase {task.status.phase!r}. "
            f"Only terminal tasks can be cleaned."
        )
    if task.batch_id:
        load_batch(cfg, task.batch_id)
    return task


def _select_batch_clean_candidates(
    cfg: RootConfig,
    *,
    include_failed: bool,
    older_than_seconds: int,
) -> list[Task]:
    phases_to_clean = {PHASE_SUCCEEDED}
    if include_failed:
        phases_to_clean.update({PHASE_FAILED, PHASE_CANCELLED})

    cutoff = datetime.now(timezone.utc).timestamp() - older_than_seconds
    selected: list[Task] = []
    for task in iter_all_tasks(cfg):
        if task.status.phase not in phases_to_clean:
            continue
        finished_at = task.timestamps.finished_at
        if finished_at is None:
            continue
        try:
            finished_dt = _parse_iso_timestamp(finished_at)
        except Exception:
            continue
        if finished_dt.timestamp() > cutoff:
            continue
        if task.batch_id:
            load_batch(cfg, task.batch_id)
        selected.append(task)
    return selected


def _plan_batch_updates(cfg: RootConfig, tasks: list[Task]) -> dict[str, set[str]]:
    by_batch: dict[str, set[str]] = {}
    for task in tasks:
        if task.batch_id:
            by_batch.setdefault(task.batch_id, set()).add(task.task_id)
    for batch_id in by_batch:
        load_batch(cfg, batch_id)
    return by_batch


def _apply_batch_updates(cfg: RootConfig, removals_by_batch: dict[str, set[str]]) -> None:
    for batch_id, removed_ids in removals_by_batch.items():
        for _ in range(5):
            batch = load_batch(cfg, batch_id)
            next_task_ids = [tid for tid in batch.task_ids if tid not in removed_ids]
            batch.task_ids = next_task_ids
            batch.expected_task_count = len(next_task_ids)
            batch.summary = _build_batch_summary(cfg, next_task_ids)
            try:
                cas_update_batch(cfg, batch, batch.meta.revision)
                break
            except CASConflict:
                continue
        else:
            raise RuntimeError(
                f"Failed to update batch {batch_id} after repeated CAS conflicts."
            )


def _delete_task_log(resolved: _ResolvedLogPath) -> dict[str, Any]:
    payload = asdict(resolved)
    if resolved.path is None:
        return payload

    path = Path(resolved.path)
    if not path.exists():
        payload["status"] = "missing"
        payload["message"] = f"Log file not found at {path}."
        return payload

    try:
        path.unlink()
    except OSError as exc:
        payload["status"] = "delete_failed"
        payload["message"] = str(exc)
        return payload

    payload["status"] = "deleted"
    payload["message"] = None
    return payload
