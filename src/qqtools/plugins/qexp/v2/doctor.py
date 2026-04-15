from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from .indexes import rebuild_all_indexes
from .locking import clean_lock
from .layout import (
    RootConfig,
    global_batches_dir,
    global_locks_dir,
    global_tasks_dir,
)
from .lifecycle import read_agent_snapshot
from .models import (
    ACTIVE_PHASES,
    AGENT_STATE_FAILED,
    AGENT_STATE_STALE,
    AGENT_STATE_STOPPED,
    BatchSummary,
    PHASE_ORPHANED,
    utc_now_iso,
    validate_phase_transition,
)
from .storage import (
    CASConflict,
    cas_update_task,
    iter_all_tasks,
    iter_machines,
    load_batch,
    load_task,
    save_batch,
    read_json,
)
from .indexes import update_index_on_phase_change

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# rebuild-index
# ---------------------------------------------------------------------------


def rebuild_indexes(cfg: RootConfig) -> dict[str, Any]:
    with clean_lock(cfg):
        return rebuild_all_indexes(cfg)


# ---------------------------------------------------------------------------
# repair-orphans
# ---------------------------------------------------------------------------


def repair_orphans(
    cfg: RootConfig,
    heartbeat_stale_seconds: float = 120.0,
) -> list[str]:
    """Find tasks in active phases whose machine heartbeat is stale.

    Checks the agent state file (state/agent.json) for each machine,
    which is where the agent writes heartbeats. A machine is considered
    stale if:
    - No agent state file exists, OR
    - The last_heartbeat is older than heartbeat_stale_seconds, OR
    - The agent process (PID) is not alive.

    Transitions stale tasks to orphaned and returns affected task_ids.
    """
    from datetime import datetime, timezone
    from .agent import get_agent_status

    machines = iter_machines(cfg)
    stale_machines: set[str] = set()
    now = datetime.now(timezone.utc)

    for m in machines:
        alt_cfg = RootConfig(
            shared_root=cfg.shared_root,
            machine_name=m.machine_name,
            runtime_root=cfg.runtime_root,
        )

        # Check agent state file (the actual heartbeat source)
        snapshot = read_agent_snapshot(alt_cfg)
        if snapshot is None:
            stale_machines.add(m.machine_name)
            continue

        if snapshot.agent_state in {
            AGENT_STATE_STOPPED,
            AGENT_STATE_FAILED,
            AGENT_STATE_STALE,
        }:
            stale_machines.add(m.machine_name)
            continue

        # Cross-machine health must be interpreted from the shared snapshot,
        # not by probing a foreign PID in the local process namespace.
        status = get_agent_status(alt_cfg, probe_local_pid=False)
        if status["is_running"]:
            continue

        # Agent not running — check how long since last heartbeat
        hb = snapshot.last_heartbeat
        if hb is None:
            stale_machines.add(m.machine_name)
            continue

        try:
            dt = datetime.fromisoformat(hb.replace("Z", "+00:00"))
            elapsed = (now - dt).total_seconds()
            if elapsed > heartbeat_stale_seconds:
                stale_machines.add(m.machine_name)
        except Exception:
            stale_machines.add(m.machine_name)

    orphaned: list[str] = []
    tasks = iter_all_tasks(cfg)
    for task in tasks:
        if task.status.phase not in ACTIVE_PHASES:
            continue
        if task.machine_name not in stale_machines:
            continue

        old_phase = task.status.phase
        try:
            validate_phase_transition(old_phase, PHASE_ORPHANED)
        except ValueError:
            continue

        task.status.phase = PHASE_ORPHANED
        task.status.reason = "machine_heartbeat_stale"
        try:
            cas_update_task(cfg, task, task.meta.revision)
            update_index_on_phase_change(cfg, task.task_id, old_phase, PHASE_ORPHANED)
            orphaned.append(task.task_id)
            log.info("Marked task %s as orphaned (was %s).", task.task_id, old_phase)
        except CASConflict:
            log.debug("CAS conflict repairing %s, skipping.", task.task_id)

    return orphaned


# ---------------------------------------------------------------------------
# cleanup-locks
# ---------------------------------------------------------------------------


def cleanup_stale_locks(
    cfg: RootConfig,
    max_age_seconds: float = 300.0,
) -> list[str]:
    locks_dir = global_locks_dir(cfg)
    if not locks_dir.is_dir():
        return []

    cleaned: list[str] = []
    for lock_file in locks_dir.iterdir():
        if not lock_file.is_file():
            continue
        try:
            mtime = lock_file.stat().st_mtime
            if (time.time() - mtime) > max_age_seconds:
                lock_file.unlink()
                cleaned.append(str(lock_file))
                log.info("Removed stale lock: %s", lock_file)
        except OSError:
            continue

    return cleaned


def repair_metadata(cfg: RootConfig) -> dict[str, Any]:
    """Repair batch truth and indexes into a converged state.

    This is the recovery entrypoint for partial clean failures:
    - prune missing task_ids from batch truth
    - recompute batch summaries from surviving tasks
    - rebuild all derived indexes
    """
    repaired_batches: list[str] = []
    pruned_task_refs = 0

    with clean_lock(cfg):
        batches_dir = global_batches_dir(cfg)
        if batches_dir.is_dir():
            for path in sorted(batches_dir.glob("*.json")):
                if path.name.endswith(".tmp"):
                    continue
                batch_id = path.stem
                batch = load_batch(cfg, batch_id)

                surviving_ids: list[str] = []
                counts: dict[str, int] = {}
                removed = 0

                for task_id in batch.task_ids:
                    try:
                        task = load_task(cfg, task_id)
                    except FileNotFoundError:
                        removed += 1
                        continue
                    surviving_ids.append(task_id)
                    phase = task.status.phase
                    counts[phase] = counts.get(phase, 0) + 1

                new_summary = BatchSummary(
                    total=len(surviving_ids),
                    queued=counts.get("queued", 0),
                    running=counts.get("running", 0),
                    succeeded=counts.get("succeeded", 0),
                    failed=counts.get("failed", 0),
                    cancelled=counts.get("cancelled", 0),
                    blocked=counts.get("blocked", 0),
                    orphaned=counts.get("orphaned", 0),
                )

                if surviving_ids != batch.task_ids or new_summary != batch.summary:
                    batch.task_ids = surviving_ids
                    batch.summary = new_summary
                    batch.meta.revision += 1
                    batch.meta.updated_at = utc_now_iso()
                    batch.meta.updated_by_machine = cfg.machine_name
                    save_batch(cfg, batch)
                    repaired_batches.append(batch.batch_id)
                    pruned_task_refs += removed

        index_stats = rebuild_all_indexes(cfg)

    return {
        "repaired_batch_count": len(repaired_batches),
        "repaired_batches": repaired_batches,
        "pruned_task_ref_count": pruned_task_refs,
        "index_stats": index_stats,
    }


# ---------------------------------------------------------------------------
# verify-integrity
# ---------------------------------------------------------------------------


def verify_integrity(cfg: RootConfig) -> dict[str, Any]:
    """Non-destructive integrity check.

    Returns dict with 'ok' bool and 'issues' list.
    """
    issues: list[str] = []
    tasks_dir = global_tasks_dir(cfg)
    batches_dir = global_batches_dir(cfg)

    if not tasks_dir.is_dir():
        return {
            "ok": True,
            "issues": [],
            "tasks_checked": 0,
            "batches_checked": 0,
        }

    checked = 0
    for path in tasks_dir.glob("*.json"):
        if path.name.endswith(".tmp"):
            continue
        checked += 1
        expected_id = path.stem
        try:
            data = read_json(path)
            task_id = data.get("task", {}).get("task_id")
            if task_id != expected_id:
                issues.append(
                    f"Filename/ID mismatch: file={path.name}, task_id={task_id}"
                )
            revision = data.get("meta", {}).get("revision")
            if not isinstance(revision, int) or revision < 1:
                issues.append(
                    f"Invalid revision for {expected_id}: {revision}"
                )
        except Exception as e:
            issues.append(f"Cannot read {path.name}: {e}")

    batches_checked = 0
    if batches_dir.is_dir():
        for path in batches_dir.glob("*.json"):
            if path.name.endswith(".tmp"):
                continue
            batches_checked += 1
            expected_id = path.stem
            try:
                data = read_json(path)
                batch_data = data.get("batch", {})
                batch_id = batch_data.get("batch_id")
                if batch_id != expected_id:
                    issues.append(
                        f"Batch filename/ID mismatch: file={path.name}, batch_id={batch_id}"
                    )
                    continue

                declared_task_ids = list(batch_data.get("task_ids", []))
                surviving_ids: list[str] = []
                counts: dict[str, int] = {}
                for task_id in declared_task_ids:
                    try:
                        task = load_task(cfg, task_id)
                    except FileNotFoundError:
                        issues.append(
                            f"Batch {expected_id} references missing task {task_id}"
                        )
                        continue
                    surviving_ids.append(task_id)
                    phase = task.status.phase
                    counts[phase] = counts.get(phase, 0) + 1

                expected_summary = {
                    "total": len(surviving_ids),
                    "queued": counts.get("queued", 0),
                    "running": counts.get("running", 0),
                    "succeeded": counts.get("succeeded", 0),
                    "failed": counts.get("failed", 0),
                    "cancelled": counts.get("cancelled", 0),
                    "blocked": counts.get("blocked", 0),
                    "orphaned": counts.get("orphaned", 0),
                }
                actual_summary = batch_data.get("summary", {})
                if actual_summary != expected_summary:
                    issues.append(
                        f"Batch {expected_id} summary drift: expected={expected_summary}, actual={actual_summary}"
                    )
            except Exception as e:
                issues.append(f"Cannot read batch {path.name}: {e}")

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "tasks_checked": checked,
        "batches_checked": batches_checked,
    }
