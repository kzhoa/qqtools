"""Bounded proof that a due Group can leave the daily working set."""

from __future__ import annotations

import math
import os
import stat
import time
from pathlib import Path
from typing import Any

from ..config_types import RootConfig
from ..lease import clock_capability
from . import group_working_set as index
from .claim_archive_scan import is_pending_archive_clear
from .group_activity import has_pending_activity, repair_activity_slice
from .group_control_coverage import has_empty_control_coverage
from .group_discovery import locator
from .group_discovery.activation import group_service_routing_state, is_group_service_active
from .group_discovery.coverage import GroupCoverage
from .group_discovery.member_reader import MemberIdentity
from .group_namespace import read_group
from .group_working_set_migration import index_active
from .locks import group_writer_lock, task_lock
from .operation_store import active_operation_path, archived_operation_path
from .paths import attempt_path, shared_paths
from .records import AttemptRecord, TaskRecord
from .store import atomic_replace, check_mutation_fence, read_json_limited
from .tasks import load_task

_MEMBERS_PER_SLICE = 16
_RECORD_LIMIT = 1024 * 1024
_TERMINAL_STATES = frozenset({"failed", "cancelled", "succeeded"})
_SETTLED_CONTROLS = frozenset({"completed", "superseded"})
_PROOF_ERRORS = (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError)


def _qualified_time(cfg: RootConfig) -> float | None:
    try:
        capability = clock_capability(cfg)
        observation = capability.observation
        if capability.is_healthy and observation is not None:
            wall_time = time.time()
            monotonic_now = time.monotonic()
            bound = observation.bound_at(monotonic_now)
            if (
                0 <= observation.monotonic_observed_at <= monotonic_now
                and math.isfinite(bound)
                and 0 <= bound <= wall_time
                and math.isfinite(wall_time)
            ):
                return wall_time - bound
    except _PROOF_ERRORS:
        pass
    return None


def _group_has_control(data: dict[str, Any]) -> bool:
    group = data["group"]
    if group.get("pending_submission_commit") is not None:
        return True
    for key in ("cancellation_operation", "worker_control"):
        control = data.get(key)
        if control is not None and (
            control.get("state") not in _SETTLED_CONTROLS or control.get("pending_machine_acknowledgements")
        ):
            return True
    return any(
        worker.get("removal_operation_id") is not None and worker["state"] != "removing"
        for worker in group["worker_set"].values()
    )


def _no_locator_obligations(cfg: RootConfig, name: str, *, empty: bool) -> bool:
    root = cfg.shared_root
    if is_group_service_active(root):
        return all(locator.read_group_locator(root, name, lane) is None for lane in locator.LANES)
    if not empty or group_service_routing_state(root) not in {None, "preparing"}:
        return False
    if not has_empty_control_coverage(cfg, name):
        return False
    try:
        (locator.group_service_root(root) / "layout.json").lstat()
    except FileNotFoundError:
        for lane in locator.LANES:
            try:
                locator.group_locator_path(root, name, lane).lstat()
            except FileNotFoundError:
                continue
            return False
        return True
    return all(locator.read_group_locator(root, name, lane) is None for lane in locator.LANES)


def _terminal_member(cfg: RootConfig, name: str, member: MemberIdentity) -> bool:
    root = cfg.shared_root
    with task_lock(root, member.task_id, blocking=False) as acquired:
        if not acquired:
            return False
        if active_operation_path(cfg, "cleanup", member.task_id).exists() or not is_pending_archive_clear(
            cfg, member.task_id
        ):
            return False
        cleanup = None
        try:
            task = load_task(cfg, member.task_id)
        except FileNotFoundError:
            cleanup = read_json_limited(
                archived_operation_path(cfg, "cleanup", member.task_id),
                max_bytes=_RECORD_LIMIT,
                record_type="group_working_set_cleanup",
            )["cleanup"]
            if (
                cleanup.get("state") != "completed"
                or cleanup.get("task_id") != member.task_id
                or cleanup.get("group_name") != name
                or cleanup.get("submission_operation_id") != member.operation_id
            ):
                return False
            task = TaskRecord.from_dict(cleanup["group_discovery_outcome"]["task"])
        if (
            task.task_id != member.task_id
            or task.group_name != name
            or task.group_membership_sequence != member.sequence
            or task.submission_operation_id != member.operation_id
            or task.state.get("projection") not in _TERMINAL_STATES
            or task.claim_control.get("active_claim") is not None
        ):
            return False
        if cleanup is not None:
            return cleanup.get("terminal_state") == task.state["projection"]
        if task.control.get("cleanup_operation_id") is not None or task.control.get("cleanup_state") is not None:
            return False
        number = task.attempt_control.get("current_attempt_number")
        attempt_id = task.attempt_control.get("current_attempt_id")
        if number is None:
            return attempt_id is None
        if type(number) is not int or number < 1:
            return False
        try:
            attempt = AttemptRecord.from_dict(
                read_json_limited(
                    attempt_path(root, member.task_id, number),
                    max_bytes=_RECORD_LIMIT,
                    record_type="group_working_set_attempt",
                )
            )
        except FileNotFoundError:
            reason = task.state.get("reason")
            if (
                task.state["projection"] != "cancelled"
                or attempt_id is not None
                or reason
                not in {
                    "cancelled_before_launch",
                    "launch_gate_lost",
                    "launch_reservation_lost",
                    "group_cancelled_before_launch",
                    "worker_or_dispatch_changed",
                }
            ):
                return False
            token = task.claim_control["fencing_epoch"]
            archive = read_json_limited(
                shared_paths(root)["claim_archive"] / member.task_id / f"{token}.json",
                max_bytes=_RECORD_LIMIT,
                record_type="group_working_set_claim",
            )["claim_archive"]
            claim = archive["claim"]
            return (
                archive["task_id"] == member.task_id
                and archive["reason"] == reason
                and archive["fencing_token"] == claim["fencing_token"] == token
                and archive["attempt_id"] == claim["attempt_id"] == f"{member.task_id}-attempt-{number}"
                and claim["launch_state"] == "claimed"
            )
        return (
            attempt.task_id == member.task_id
            and attempt.attempt_number == number
            and attempt_id in {None, attempt.attempt_id}
            and attempt.phase in _TERMINAL_STATES
            and (
                not (attempt.termination.get("requested_at") or attempt.termination.get("requested_by_operation_id"))
                or bool(attempt.termination.get("acknowledged_at"))
            )
        )


def _cursor_path(root: Path, name: str) -> Path:
    return root / "indexes" / "group-working-set" / "expiry-cursors" / f"{name}.json"


def _cursor_start(path: Path, identity: dict[str, Any], tail: int) -> int:
    try:
        if not stat.S_ISDIR(path.parent.lstat().st_mode) or not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError("Group working-set expiry cursor path is invalid.")
        saved = read_json_limited(path, max_bytes=4096, record_type="group_working_set_expiry_cursor")
    except FileNotFoundError:
        return 1
    sequence = saved.get("next_sequence")
    if (
        saved.get("version") == 1
        and saved.get("identity") == identity
        and type(sequence) is int
        and 1 <= sequence <= tail
    ):
        return sequence
    return 1


def _save_cursor(path: Path, identity: dict[str, Any], sequence: int) -> None:
    check_mutation_fence(path)
    path.parent.mkdir(exist_ok=True)
    if not stat.S_ISDIR(path.parent.lstat().st_mode):
        raise ValueError("Group working-set expiry cursor directory is invalid.")
    descriptor = os.open(path.parent.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    atomic_replace(path, {"version": 1, "identity": identity, "next_sequence": sequence})


def _remove_cursor(path: Path) -> None:
    try:
        if stat.S_ISDIR(path.parent.lstat().st_mode):
            check_mutation_fence(path)
            path.unlink(missing_ok=True)
    except FileNotFoundError:
        return


def _group_is_settled(cfg: RootConfig, name: str, revision: int, cursor: Path) -> bool:
    root = cfg.shared_root
    data = read_group(root, name)
    group = data["group"]
    next_sequence = group.get("next_membership_sequence")
    group_revision = data["meta"]["revision"]
    if group["name"] != name or type(next_sequence) is not int or next_sequence < 1 or type(group_revision) is not int:
        return False
    if has_pending_activity(root, name) or _group_has_control(data):
        return False
    tail = next_sequence - 1
    if not _no_locator_obligations(cfg, name, empty=tail == 0):
        return False
    if tail == 0:
        return True
    coverage = GroupCoverage(root, name)
    status = coverage.status()
    if not status.is_complete or status.reason is not None or status.tail != tail or status.prefix != tail:
        return False
    identity = {
        "group_name": name,
        "activity_revision": revision,
        "group_revision": group_revision,
        "created_at": data["meta"]["created_at"],
        "creation_operation_id": group["creation_operation_id"],
        "membership_tail": tail,
    }
    start = _cursor_start(cursor, identity, tail)
    stop = min(start + _MEMBERS_PER_SLICE, tail + 1)
    sequence = start
    while sequence < stop:
        member = coverage.read_member(sequence)
        if member.sequence != sequence or not _terminal_member(cfg, name, member):
            break
        sequence += 1
    if sequence == tail + 1:
        return True
    if sequence > start:
        _save_cursor(cursor, identity, sequence)
    return False


def _finish_candidate(
    cfg: RootConfig,
    name: str,
    revision: int,
    next_check_ms: int,
    *,
    qualified_time: float,
) -> None:
    root = cfg.shared_root
    with group_writer_lock(cfg, name, blocking=False) as acquired:
        if acquired and next_check_ms <= math.floor(qualified_time * 1000):
            cursor = _cursor_path(root, name)
            try:
                settled = _group_is_settled(cfg, name, revision, cursor)
            except _PROOF_ERRORS:
                settled = False
            if settled:
                if index.finish_check(
                    root, name, revision, next_check_ms, now=qualified_time, retire=True, blocking=False
                ):
                    _remove_cursor(cursor)
                return

        # CAS also permits deferral when another writer holds the Group lock.
        # Move behind the currently due cohort on the qualified shared time.
        index.finish_check(root, name, revision, next_check_ms, now=qualified_time, retire=False, blocking=False)


def expire_due_groups(cfg: RootConfig, *, now: float | None = None, limit: int = 8) -> int:
    """Repair activity and examine at most ``limit`` scheduled working-set entries."""
    root = cfg.shared_root
    if not index_active(root):
        return 0
    index.recover_index(root, blocking=False)
    repair_activity_slice(cfg, limit)
    if now is None:
        first = index.next_check_time(root, blocking=False)
        if first is None or first > time.time():
            return 0
        now = _qualified_time(cfg)
        if now is None:
            return 0
    candidates = index.due_candidates(root, now=now, limit=limit, blocking=False)
    for name, revision, next_check_ms in candidates:
        try:
            _finish_candidate(cfg, name, revision, next_check_ms, qualified_time=now)
        except index.GroupWorkingSetUnavailable:
            # A competing reader or writer can leave this candidate due.
            continue
    return len(candidates)
