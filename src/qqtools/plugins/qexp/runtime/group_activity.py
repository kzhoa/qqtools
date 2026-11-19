"""Durable working-set handoffs for Group and grouped Task mutations."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from ..config_types import RootConfig
from .group_namespace import GroupPublicationUnavailable, _submission_state, group_visibility
from .group_working_set import GroupWorkingSetUnavailable, is_index_initialized, next_check_time, refresh_group
from .group_working_set_migration import _read_state, read_schema, schema_capabilities
from .locks import exclusive, group_writer_lock
from .observation.tree import IndexTree
from .operation_store import locate_operation_path
from .paths import group_path, shared_paths, submission_path, task_path
from .project_activation import notify_group_activity
from .protocol_compatibility import GROUP_WORKING_SET_CAPABILITY
from .records import TaskRecord, new_id, validate_group_name, validate_identifier
from .responsibility_store import DurableIO
from .store import atomic_replace, check_mutation_fence, fenced_mutations, read_json, read_json_limited
from .tasks import load_task

_INTENT_BYTES = 4096
_ACTIVITY_VERSION = 1
_LEASE_FIELDS = frozenset({"lease_expires_at", "clock_error_bound_seconds", "clock_provider", "clock_observation_id"})


def _activity_directory(root: Path) -> Path:
    return shared_paths(root)["indexes"] / "group-activity"


def _pending_path(root: Path, name: str) -> Path:
    return _activity_directory(root) / "pending" / f"{name}.json"


def _submission_receipt_path(root: Path, name: str) -> Path:
    return _activity_directory(root) / "submission-receipts" / f"{name}.json"


def _queue(root: Path) -> IndexTree:
    return IndexTree(_activity_directory(root), phase="repair")


def _queue_lock_path(root: Path) -> Path:
    return shared_paths(root)["locks"] / "group-activity-queue.lock"


def _read_activity_record(path: Path) -> dict[str, Any] | None:
    try:
        return read_json_limited(path, max_bytes=_INTENT_BYTES, record_type="group_activity")
    except FileNotFoundError:
        return None


def has_pending_activity(root: Path, name: str) -> bool:
    """Check under the Group lock before using an inactivity deadline."""
    return _pending_path(root, name).exists()


def has_any_activity_repair(root: Path) -> bool:
    """Check for queued repair work before certifying an index inventory."""
    queue = _queue(root)
    if not queue.root_path.exists():
        return False
    with exclusive(_queue_lock_path(root), blocking=False) as has_queue_lock:
        return not has_queue_lock or bool(queue.candidates(None, max_candidates=1).keys)


# QQTOOLS-COMPAT-0025: only the creation of a new published name can leave
# inventory coverage incomplete. The normal repair queue continues to own all
# activity, including Tasks with arbitrary-size canonical truth.
def _bootstrap_queue(root: Path) -> IndexTree:
    return IndexTree(_activity_directory(root), phase="bootstrap")


def next_bootstrap_repair(root: Path) -> str | None:
    with exclusive(_queue_lock_path(root), blocking=False) as acquired:
        if not acquired:
            raise BlockingIOError("Group activity repair queue is busy")
        keys = _bootstrap_queue(root).candidates(None, max_candidates=1, max_pages=66, max_bytes=66 * 65536).keys
        return keys[0] if keys else None


def finish_bootstrap_repair(root: Path, name: str) -> None:
    with exclusive(_queue_lock_path(root)):
        _bootstrap_queue(root).discard(name)


def require_activity_writer(root: Path) -> None:
    """Require the working-set writer fence and projections before changing truth."""
    if GROUP_WORKING_SET_CAPABILITY not in schema_capabilities(read_schema(root)) or not is_index_initialized(root):
        raise RuntimeError("Group activity is not ready. Advance the Project upgrade before retrying the mutation.")


def _sync_directory(path: Path) -> None:
    DurableIO().sync_directory(path, "group_activity")


def _task_fingerprint(task: TaskRecord) -> str:
    value = task.to_dict()["task"]
    claim_control = dict(value["claim_control"])
    claim = claim_control.get("active_claim")
    if claim is not None:
        claim_control["active_claim"] = {field: data for field, data in claim.items() if field not in _LEASE_FIELDS}
    value["claim_control"] = claim_control
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prepare_locked(
    cfg: RootConfig, name: str, target: dict[str, Any], *, token: str, may_create: bool = False
) -> None:
    root = cfg.shared_root
    require_activity_writer(root)
    path = _pending_path(root, name)
    with exclusive(_queue_lock_path(root)):
        queue = _queue(root)
        queue.add(name)
        if may_create and _read_state(root) in {"inventory", "building"}:
            bootstrap = _bootstrap_queue(root)
            bootstrap.add(name)
            _sync_directory(bootstrap.root_path.parent)
            _sync_directory(bootstrap.root_path.parent.parent)
        _sync_directory(queue.root_path.parent)
        _sync_directory(queue.root_path.parent.parent)
        _sync_directory(_activity_directory(root))
        _sync_directory(_activity_directory(root).parent)
    atomic_replace(path, {"version": _ACTIVITY_VERSION, "group": name, "token": token, "target": target})
    _sync_directory(path.parent.parent)
    if next_check_time(root) is None:
        notify_group_activity(cfg)


def _truth_matches_task(cfg: RootConfig, name: str, target: dict[str, Any]) -> bool:
    task_id = validate_identifier(target["task_id"], "task_id")
    operation_id = validate_identifier(target["submission"], "submission_operation_id")
    cleanup_id = target.get("cleanup")
    try:
        task = load_task(cfg, task_id)
    except FileNotFoundError:
        if cleanup_id is None:
            return False
        try:
            operation_file = locate_operation_path(cfg, "cleanup", task_id)
            cleanup = read_json(operation_file)["cleanup"]
            task = TaskRecord.from_dict(cleanup["group_discovery_outcome"]["task"])
        except FileNotFoundError:
            return False
        if (
            cleanup.get("operation_id") != cleanup_id
            or cleanup.get("task_id") != task_id
            or cleanup.get("group_name") != name
            or cleanup.get("submission_operation_id") != operation_id
            or task.control.get("cleanup_operation_id") != cleanup_id
            or task.meta["revision"] != target["revision"] - 1
        ):
            return False
        _sync_directory(operation_file.parent)
    else:
        if task.meta["revision"] < target["revision"] or _task_fingerprint(task) == target["before"]:
            if not target.get("cleanup_start"):
                return False
            try:
                operation_file = locate_operation_path(cfg, "cleanup", task_id)
                cleanup = read_json(operation_file)["cleanup"]
            except FileNotFoundError:
                return False
            if (
                cleanup.get("operation_id"),
                cleanup.get("task_id"),
                cleanup.get("group_name"),
                cleanup.get("submission_operation_id"),
            ) != (cleanup_id, task_id, name, operation_id):
                return False
            _sync_directory(operation_file.parent)
    if (
        task.task_id != task_id
        or task.group_name != name
        or task.submission_operation_id != operation_id
        or task.group_membership_sequence != target["membership"]
    ):
        return False
    _sync_directory(task_path(cfg.shared_root, task_id).parent)
    return True


def _truth_matches_group(data: dict[str, Any], target: dict[str, Any]) -> bool:
    if data["meta"]["revision"] < target["revision"]:
        return False
    group = data["group"]
    if target.get("cancellation") is not None:
        return any(barrier["operation_id"] == target["cancellation"] for barrier in group["cancellation_barriers"])
    if target.get("worker") is not None:
        worker = group["worker_set"].get(target["worker"])
        if worker is None:
            return False
        proof = target["worker_proof"]
        return all(worker.get(field) == value for field, value in proof.items())
    if target.get("dispatch_epoch") is not None:
        return group["dispatch_epoch"] >= target["dispatch_epoch"]
    return group["creation_operation_id"] is None


def _submission_activity_published(root: Path, name: str, operation_id: str) -> bool:
    path = _submission_receipt_path(root, name)
    receipt = _read_activity_record(path)
    if receipt is None or receipt.get("operation_id") != operation_id:
        return False
    _sync_directory(path.parent)
    _sync_directory(path.parent.parent)
    return True


def _submission_committed(root: Path, operation_id: str) -> bool:
    return _submission_state(root, validate_identifier(operation_id, "submission_operation_id")) == "committed"


def _should_publish(cfg: RootConfig, name: str, target: dict[str, Any]) -> bool:
    root = cfg.shared_root
    file = group_path(root, name)
    try:
        data = read_json(file)
    except FileNotFoundError:
        return False
    group = data["group"]
    if group["name"] != name:
        raise ValueError(f"Group activity target identity is invalid for {name!r}")
    if (owner := group.get("creation_operation_id")) is not None and not _submission_committed(root, owner):
        return False
    kind = target["kind"]
    if kind == "task":
        has_truth = _truth_matches_task(cfg, name, target)
    elif kind == "group":
        has_truth = _truth_matches_group(data, target)
        if has_truth and (worker := target.get("worker")) is not None:
            owner = group["worker_set"][worker].get("added_by_operation")
            has_truth = owner is None or _submission_committed(root, owner)
    elif kind == "submission":
        operation_id = target["operation_id"]
        has_truth = not _submission_activity_published(root, name, operation_id) and _submission_committed(
            root, operation_id
        )
        if has_truth:
            _sync_directory(submission_path(root, operation_id).parent)
    elif kind == "wake":
        has_truth = True
    else:
        raise ValueError(f"Unknown Group activity target {kind!r}")
    if has_truth:
        _sync_directory(file.parent)
    return has_truth


def _finish_locked(cfg: RootConfig, name: str, *, blocking: bool = True, was_enqueued: bool = False) -> bool:
    root = cfg.shared_root
    path = _pending_path(root, name)
    intent = _read_activity_record(path)
    if intent is None and not was_enqueued:
        return False
    if intent is not None:
        if intent.get("version") != _ACTIVITY_VERSION or intent.get("group") != name:
            raise ValueError(f"Group activity intent identity is invalid for {name!r}")
        target = intent["target"]
        if _should_publish(cfg, name, target):
            # The shared filesystem timestamps the durable intent. A skewed
            # submitting machine cannot shorten or extend the shared deadline.
            refresh_group(root, name, intent["token"], now=path.stat().st_ctime_ns / 1_000_000_000, blocking=blocking)
            if target["kind"] == "submission":
                receipt = _submission_receipt_path(root, name)
                atomic_replace(receipt, {"operation_id": target["operation_id"]})
                _sync_directory(receipt.parent.parent)
    with exclusive(_queue_lock_path(root), blocking=blocking) as has_queue_lock:
        if not has_queue_lock:
            return False
        if intent is not None:
            check_mutation_fence(path)
            path.unlink()
        # The previous writer may have unlinked the intent before its parent
        # directory sync completed. Settle that unlink before retiring the key.
        if path.parent.exists():
            _sync_directory(path.parent)
        # A completed writer already settled publication. Leave only interrupted
        # creations for the converter, even under continuous post-inventory traffic.
        if _read_state(root) in {"inventory", "building"}:
            _bootstrap_queue(root).discard(name)
        _queue(root).discard(name)
    return True


@contextmanager
def record_group_activity(
    cfg: RootConfig,
    data: dict[str, Any],
    *,
    worker: str | None = None,
    cancellation: str | None = None,
    is_dispatch_change: bool = False,
) -> Iterator[None]:
    """Bracket one Group truth write while the Group writer lock is held."""
    group = data["group"]
    name = group["name"]
    target = {"kind": "group", "revision": data["meta"]["revision"]}
    if worker is not None:
        target["worker"] = worker
        target["worker_proof"] = {
            field: group["worker_set"][worker].get(field)
            for field in ("state", "state_epoch", "scheduling_role", "gpu_limit_gpus", "removal_operation_id")
        }
    elif cancellation is not None:
        target["cancellation"] = cancellation
    elif is_dispatch_change:
        target["dispatch_epoch"] = group["dispatch_epoch"]
    _finish_locked(cfg, name)
    _prepare_locked(
        cfg,
        name,
        target,
        token=new_id(),
        may_create=worker is None and cancellation is None and not is_dispatch_change,
    )
    yield
    _finish_locked(cfg, name)


@contextmanager
def record_group_task_activity(
    cfg: RootConfig,
    task: TaskRecord,
    *,
    cleanup_id: str | None = None,
    cleanup_start: bool = False,
    mutation_fence: Callable[[], None] | None = None,
) -> Iterator[None]:
    """Bracket a persisted Task mutation under the caller's Group and Task locks."""
    name = task.group_name
    operation_id = task.submission_operation_id
    if name is None or operation_id is None:
        yield
        return
    root = cfg.shared_root
    if not _submission_committed(root, operation_id):
        yield
        return
    target = {
        "kind": "task",
        "task_id": task.task_id,
        "revision": task.meta["revision"] + 1,
        "before": _task_fingerprint(task),
        "submission": operation_id,
        "membership": task.group_membership_sequence,
        "cleanup": cleanup_id,
        "cleanup_start": cleanup_start,
    }
    with fenced_mutations(root, mutation_fence):
        _finish_locked(cfg, name)
        _prepare_locked(cfg, name, target, token=new_id())
    yield
    with fenced_mutations(root, mutation_fence):
        _finish_locked(cfg, name)


def prepare_submission_activity(cfg: RootConfig, name: str, submission: dict[str, Any]) -> None:
    """Prepare a committed append handoff before its writer updates Submission truth."""
    operation_id = submission["operation_id"]
    root = cfg.shared_root
    group = read_json(group_path(root, name))["group"]
    pending = group.get("pending_submission_commit") or {}
    if pending.get("operation_id") != operation_id:
        raise RuntimeError(f"Submission {operation_id!r} no longer owns pending Group {name!r} activity")
    _finish_locked(cfg, name)
    if not _submission_activity_published(root, name, operation_id):
        _prepare_locked(
            cfg,
            name,
            {"kind": "submission", "operation_id": operation_id},
            token=operation_id,
            may_create=group.get("creation_operation_id") == operation_id,
        )


def finalize_submission_activity(cfg: RootConfig, name: str, submission: dict[str, Any]) -> None:
    """Publish a committed append while the caller still holds pending ownership proof."""
    prepare_submission_activity(cfg, name, submission)
    _finish_locked(cfg, name)
    if not _submission_activity_published(cfg.shared_root, name, submission["operation_id"]):
        raise RuntimeError(f"Submission {submission['operation_id']!r} has no durable Group activity receipt")


def wake_group(cfg: RootConfig, name: str) -> dict[str, Any]:
    """Refresh an existing published Group without changing Group truth."""
    validate_group_name(name)
    with group_writer_lock(cfg, name):
        root = cfg.shared_root
        group = group_visibility(root, name)
        _finish_locked(cfg, name)
        _prepare_locked(cfg, name, {"kind": "wake"}, token=new_id())
        _finish_locked(cfg, name)
        return group


def _repair_candidates(root: Path, limit: int) -> tuple[str, ...]:
    cursor_path = _activity_directory(root) / "repair-cursor.json"
    with exclusive(_queue_lock_path(root), blocking=False) as has_queue_lock:
        if not has_queue_lock:
            return ()
        cursor = _read_activity_record(cursor_path) or {}
        after = cursor.get("after")
        queue = _queue(root)
        batch = queue.candidates(after, max_candidates=limit)
        if after is not None and not batch.keys and batch.exhausted:
            batch = queue.candidates(None, max_candidates=limit)
        if batch.keys or after is not None:
            atomic_replace(cursor_path, {"after": None if batch.exhausted else batch.keys[-1]})
        return batch.keys


def repair_activity_slice(cfg: RootConfig, limit: int) -> int:
    """Recover a bounded slice of pending activity without waiting for writer locks."""
    if type(limit) is not int or limit < 1:
        raise ValueError("Group activity repair limit must be a positive integer")
    repaired = 0
    for name in _repair_candidates(cfg.shared_root, limit):
        validate_group_name(name)
        try:
            with group_writer_lock(cfg, name, blocking=False) as has_group_lock:
                if has_group_lock and _finish_locked(cfg, name, blocking=False, was_enqueued=True):
                    repaired += 1
        except (GroupWorkingSetUnavailable, GroupPublicationUnavailable, OSError, ValueError, KeyError, TypeError):
            continue
    return repaired
