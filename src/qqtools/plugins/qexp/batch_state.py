from __future__ import annotations

from .layout import RootConfig
from .models import BATCH_COMMIT_PREPARING, BatchSummary, PHASE_CANCELLED, PHASE_FAILED, PHASE_QUEUED, PHASE_RUNNING, PHASE_SUCCEEDED
from .storage import load_task


def collect_batch_task_counts(
    cfg: RootConfig,
    task_ids: list[str],
    *,
    ignore_missing: bool = False,
) -> tuple[list[str], dict[str, int]]:
    surviving_ids: list[str] = []
    counts: dict[str, int] = {}
    for task_id in task_ids:
        try:
            task = load_task(cfg, task_id)
        except FileNotFoundError:
            if ignore_missing:
                continue
            raise
        surviving_ids.append(task_id)
        phase = task.status.phase
        counts[phase] = counts.get(phase, 0) + 1
    return surviving_ids, counts


def resolve_declared_batch_total(
    *,
    commit_state: str,
    expected_task_count: int,
    declared_task_ids: list[str],
    surviving_ids: list[str],
) -> int:
    if commit_state == BATCH_COMMIT_PREPARING:
        if expected_task_count <= 0:
            return len(declared_task_ids)
        return expected_task_count
    return len(surviving_ids)


def build_batch_summary_from_counts(
    *,
    total: int,
    counts: dict[str, int],
) -> BatchSummary:
    return BatchSummary(
        total=total,
        queued=counts.get(PHASE_QUEUED, 0),
        running=counts.get(PHASE_RUNNING, 0),
        succeeded=counts.get(PHASE_SUCCEEDED, 0),
        failed=counts.get(PHASE_FAILED, 0),
        cancelled=counts.get(PHASE_CANCELLED, 0),
        blocked=counts.get("blocked", 0),
        orphaned=counts.get("orphaned", 0),
    )
