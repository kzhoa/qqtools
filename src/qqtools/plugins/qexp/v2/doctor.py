from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from .indexes import rebuild_all_indexes
from .layout import (
    RootConfig,
    global_locks_dir,
    global_tasks_dir,
)
from .models import (
    ACTIVE_PHASES,
    PHASE_ORPHANED,
    utc_now_iso,
    validate_phase_transition,
)
from .storage import (
    CASConflict,
    cas_update_task,
    iter_all_tasks,
    iter_machines,
    load_task,
    read_json,
)
from .indexes import update_index_on_phase_change

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# rebuild-index
# ---------------------------------------------------------------------------


def rebuild_indexes(cfg: RootConfig) -> dict[str, Any]:
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
    from .agent import read_agent_state, is_agent_running

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
        state = read_agent_state(alt_cfg)
        if state is None:
            stale_machines.add(m.machine_name)
            continue

        # If agent PID is recorded and still alive, machine is healthy
        if state.get("pid") and is_agent_running(alt_cfg):
            continue

        # Agent not running — check how long since last heartbeat
        hb = state.get("last_heartbeat")
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


# ---------------------------------------------------------------------------
# verify-integrity
# ---------------------------------------------------------------------------


def verify_integrity(cfg: RootConfig) -> dict[str, Any]:
    """Non-destructive integrity check.

    Returns dict with 'ok' bool and 'issues' list.
    """
    issues: list[str] = []
    tasks_dir = global_tasks_dir(cfg)

    if not tasks_dir.is_dir():
        return {"ok": True, "issues": [], "tasks_checked": 0}

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

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "tasks_checked": checked,
    }
