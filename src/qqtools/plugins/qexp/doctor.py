from __future__ import annotations

import logging
import time
from pathlib import Path

from .batch_state import (
    build_batch_summary_from_counts,
    collect_batch_task_counts,
    resolve_declared_batch_total,
)
from .indexes import rebuild_all_indexes
from .locking import clean_lock
from .layout import (
    RootConfig,
    global_batches_dir,
    global_events_dir,
    global_locks_dir,
    global_tasks_dir,
    list_forbidden_truth_layout_dirs,
    validate_root_contract,
    write_root_manifest,
)
from .lifecycle import read_agent_snapshot
from .models import (
    ACTIVE_PHASES,
    AGENT_STATE_FAILED,
    AGENT_STATE_STALE,
    AGENT_STATE_STOPPED,
    BATCH_COMMIT_ABORTED,
    BATCH_COMMIT_COMMITTED,
    BATCH_COMMIT_PREPARING,
    BatchSummary,
    PHASE_ORPHANED,
    ROOT_SCOPE_PROJECT,
    TERMINAL_PHASES,
    RootGovernanceSnapshot,
    Task,
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


def _count_event_files(base: Path) -> int:
    if not base.is_dir():
        return 0
    total = 0
    for path in base.rglob("*.json"):
        if not path.name.endswith(".tmp"):
            total += 1
    return total


def collect_governance_snapshot(cfg: RootConfig) -> RootGovernanceSnapshot:
    total_tasks = 0
    terminal_tasks = 0
    tasks_dir = global_tasks_dir(cfg)
    if tasks_dir.is_dir():
        for path in tasks_dir.glob("*.json"):
            if path.name.endswith(".tmp"):
                continue
            try:
                task = Task.from_dict(read_json(path))
            except Exception:
                continue
            total_tasks += 1
            if task.status.phase in TERMINAL_PHASES:
                terminal_tasks += 1
    return RootGovernanceSnapshot(
        total_tasks=total_tasks,
        terminal_tasks=terminal_tasks,
        total_batches=len(list(global_batches_dir(cfg).glob("*.json")))
        if global_batches_dir(cfg).is_dir()
        else 0,
        total_events=_count_event_files(global_events_dir(cfg)),
        total_machines=len(iter_machines(cfg)),
        updated_at=utc_now_iso(),
    )


def refresh_root_manifest_governance(cfg: RootConfig) -> RootGovernanceSnapshot:
    manifest = validate_root_contract(cfg)
    manifest.governance = collect_governance_snapshot(cfg)
    write_root_manifest(cfg, manifest)
    return manifest.governance


# ---------------------------------------------------------------------------
# rebuild-index
# ---------------------------------------------------------------------------


def rebuild_indexes(cfg: RootConfig) -> dict[str, Any]:
    with clean_lock(cfg):
        stats = rebuild_all_indexes(cfg)
        governance = refresh_root_manifest_governance(cfg)
        return {
            "index_stats": stats,
            "governance": governance.to_dict(),
        }


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
            project_root=cfg.project_root,
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
    committed_batches: list[str] = []
    aborted_batches: list[str] = []
    pruned_task_refs = 0

    with clean_lock(cfg):
        batches_dir = global_batches_dir(cfg)
        if batches_dir.is_dir():
            for path in sorted(batches_dir.glob("*.json")):
                if path.name.endswith(".tmp"):
                    continue
                batch_id = path.stem
                batch = load_batch(cfg, batch_id)

                removed = 0

                for task_id in batch.task_ids:
                    try:
                        load_task(cfg, task_id)
                    except FileNotFoundError:
                        removed += 1
                surviving_ids, counts = collect_batch_task_counts(
                    cfg,
                    batch.task_ids,
                    ignore_missing=True,
                )
                declared_count = resolve_declared_batch_total(
                    commit_state=batch.commit_state,
                    expected_task_count=batch.expected_task_count,
                    declared_task_ids=batch.task_ids,
                    surviving_ids=surviving_ids,
                )
                new_summary = build_batch_summary_from_counts(
                    total=declared_count,
                    counts=counts,
                )
                next_commit_state = batch.commit_state
                if batch.commit_state == BATCH_COMMIT_PREPARING:
                    if len(surviving_ids) == declared_count:
                        next_commit_state = BATCH_COMMIT_COMMITTED
                        committed_batches.append(batch.batch_id)
                    else:
                        next_commit_state = BATCH_COMMIT_ABORTED
                        aborted_batches.append(batch.batch_id)

                if (
                    surviving_ids != batch.task_ids
                    or new_summary != batch.summary
                    or next_commit_state != batch.commit_state
                    or batch.expected_task_count != declared_count
                ):
                    batch.task_ids = surviving_ids
                    batch.summary = new_summary
                    batch.commit_state = next_commit_state
                    batch.expected_task_count = declared_count
                    batch.meta.revision += 1
                    batch.meta.updated_at = utc_now_iso()
                    batch.meta.updated_by_machine = cfg.machine_name
                    save_batch(cfg, batch)
                    repaired_batches.append(batch.batch_id)
                    pruned_task_refs += removed

        index_stats = rebuild_all_indexes(cfg)
        governance = refresh_root_manifest_governance(cfg)

    return {
        "repaired_batch_count": len(repaired_batches),
        "repaired_batches": repaired_batches,
        "committed_batches": committed_batches,
        "aborted_batches": aborted_batches,
        "pruned_task_ref_count": pruned_task_refs,
        "index_stats": index_stats,
        "governance": governance.to_dict(),
    }


# ---------------------------------------------------------------------------
# verify-integrity
# ---------------------------------------------------------------------------


def verify_integrity(cfg: RootConfig) -> dict[str, Any]:
    """Non-destructive integrity check.

    Returns dict with 'ok' bool and 'issues' list.
    """
    issues: list[str] = []
    manifest: dict[str, Any] | None = None
    tasks_dir = global_tasks_dir(cfg)
    batches_dir = global_batches_dir(cfg)

    try:
        root_manifest = validate_root_contract(cfg)
        manifest = root_manifest.to_dict()["root_manifest"]
        if root_manifest.root_scope != ROOT_SCOPE_PROJECT:
            issues.append(
                f"Invalid root scope {root_manifest.root_scope!r}; expected {ROOT_SCOPE_PROJECT!r}."
            )
    except Exception as exc:
        issues.append(f"Invalid root manifest: {exc}")

    forbidden_truth_dirs = [
        str(path) for path in list_forbidden_truth_layout_dirs(cfg)
    ]
    for path in forbidden_truth_dirs:
        issues.append(
            f"Forbidden truth-layout directory detected: {path}. "
            "Only global object truth and machine-private subtrees are allowed."
        )

    checked = 0
    if tasks_dir.is_dir():
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
                stored_expected_task_count = batch_data.get(
                    "expected_task_count",
                    len(declared_task_ids),
                )
                commit_state = batch_data.get("commit_state")
                if commit_state not in {
                    BATCH_COMMIT_PREPARING,
                    BATCH_COMMIT_COMMITTED,
                    BATCH_COMMIT_ABORTED,
                }:
                    issues.append(
                        f"Batch {expected_id} has invalid commit_state {commit_state!r}"
                    )
                if (
                    not isinstance(stored_expected_task_count, int)
                    or stored_expected_task_count < 0
                ):
                    issues.append(
                        f"Batch {expected_id} has invalid expected_task_count {stored_expected_task_count!r}"
                    )
                    stored_expected_task_count = len(declared_task_ids)
                if len(declared_task_ids) > stored_expected_task_count:
                    issues.append(
                        f"Batch {expected_id} has more persisted tasks than expected_task_count."
                    )
                counts: dict[str, int] = {}
                surviving_ids: list[str] = []
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
                    if task.batch_id != expected_id:
                        issues.append(
                            f"Batch {expected_id} contains task {task_id} with mismatched batch_id {task.batch_id!r}"
                        )

                expected_task_count = resolve_declared_batch_total(
                    commit_state=commit_state,
                    expected_task_count=stored_expected_task_count,
                    declared_task_ids=declared_task_ids,
                    surviving_ids=surviving_ids,
                )

                expected_summary = build_batch_summary_from_counts(
                    total=expected_task_count,
                    counts=counts,
                ).to_dict()
                actual_summary = batch_data.get("summary", {})
                if actual_summary != expected_summary:
                    issues.append(
                        f"Batch {expected_id} summary drift: expected={expected_summary}, actual={actual_summary}"
                    )
                if (
                    commit_state == BATCH_COMMIT_COMMITTED
                    and len(surviving_ids) != expected_task_count
                ):
                    issues.append(
                        f"Batch {expected_id} is committed but only has "
                        f"{len(surviving_ids)}/{expected_task_count} tasks."
                    )
                if (
                    commit_state == BATCH_COMMIT_PREPARING
                    and len(surviving_ids) == expected_task_count
                ):
                    issues.append(
                        f"Batch {expected_id} is still preparing despite complete task set."
                    )
            except Exception as e:
                issues.append(f"Cannot read batch {path.name}: {e}")

    governance = collect_governance_snapshot(cfg).to_dict()
    if governance["terminal_tasks"] > governance["total_tasks"] * 0.8 and governance["total_tasks"] > 20:
        issues.append(
            "Terminal task ratio exceeds 80%; lifecycle cleanup or archive governance is overdue."
        )

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "root_manifest": manifest,
        "forbidden_truth_dirs": forbidden_truth_dirs,
        "governance": governance,
        "tasks_checked": checked,
        "batches_checked": batches_checked,
    }
