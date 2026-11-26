"""Advisory Task and Group observations for an explicit Project takeover preview."""

from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import Iterator
from heapq import nsmallest
from pathlib import Path
from typing import Any

from ..config_types import RootConfig
from ..domain.policies import group_allows, task_machine_matches
from ..runtime.authority_scan import iter_evidence_files, validate_evidence_path
from ..runtime.group_cancellation import has_active_cancellation
from ..runtime.group_namespace import group_visibility
from ..runtime.operation_store import operation_exists
from ..runtime.paths import attempt_path, group_path, shared_paths
from ..runtime.records import GROUP_DISPATCH_STATES, AttemptRecord, TaskRecord, validate_identifier
from ..runtime.store import read_json
from ..runtime.submission_control import read_submission_state
from ..runtime.tasks import load_task

_PAGE_SIZE = 25
_TERMINAL_PHASES = frozenset({"succeeded", "failed", "cancelled"})


class TaskImpactCursorError(ValueError):
    """The advisory Task continuation has invalid identifier syntax."""


def _directory_identity(directory: Path) -> tuple[int, int]:
    metadata = directory.stat(follow_symlinks=False)
    if not stat.S_ISDIR(metadata.st_mode):
        raise NotADirectoryError(f"Expected a Project Task directory at {directory}.")
    return metadata.st_dev, metadata.st_ino


def _published_target_tasks(cfg: RootConfig, directory: Path) -> Iterator[TaskRecord]:
    for path in iter_evidence_files(directory):
        try:
            task = load_task(cfg, path.stem)
            if task.task_id != path.stem:
                raise ValueError(f"Task ID {task.task_id!r} does not match its filename.")
            home_machine = task.placement_policy["home_machine"]
            if not isinstance(home_machine, str) or not home_machine:
                raise ValueError("Task home machine is invalid.")
            claim = task.claim_control.get("active_claim")
            claim_machine = claim.get("machine_name") if claim is not None else None
            if home_machine != cfg.machine_name and claim_machine != cfg.machine_name:
                if claim is not None and (not isinstance(claim_machine, str) or not claim_machine):
                    raise ValueError("Task active claim machine is invalid.")
                # A new primary worker can also admit offered Tasks from another home.
                if (
                    claim is not None
                    or task.group_name is None
                    or task.state["projection"] != "queued"
                    or not task_machine_matches(task, cfg.machine_name)
                ):
                    continue
            operation_id = task.submission_operation_id
            if operation_id is not None:
                state = read_submission_state(cfg, operation_id, should_request_repair=False)
                if state != "committed":
                    continue
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Cannot establish Task impact from {path}: {exc}") from exc
        yield task


def _observe_attempt(cfg: RootConfig, task: TaskRecord, claim: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(task.attempt_control, dict) or task.task_id in {".", ".."}:
        return None
    attempt_id = task.attempt_control.get("current_attempt_id")
    number = task.attempt_control.get("current_attempt_number")
    if not isinstance(attempt_id, str) or not attempt_id or type(number) is not int or number < 1:
        return None
    path = attempt_path(cfg.shared_root, task.task_id, number)
    try:
        if not validate_evidence_path(path, cfg.shared_root):
            return None
        attempt = AttemptRecord.from_dict(read_json(path))
    except (KeyError, OSError, TypeError, ValueError):
        return None
    if (
        attempt.task_id != task.task_id
        or attempt.attempt_id != attempt_id
        or type(attempt.attempt_number) is not int
        or attempt.attempt_number != number
        or not isinstance(attempt.reservation_id, str)
        or not isinstance(attempt.termination, dict)
        or not isinstance(attempt.result, dict)
    ):
        return None
    if claim is not None and (
        claim.get("machine_name") != attempt.machine_name
        or not attempt.reservation_id
        or claim.get("reservation_id") != attempt.reservation_id
        or claim.get("attempt_id") != attempt_id
        or type(claim.get("attempt_number")) is not int
        or claim["attempt_number"] != number
        or type(claim.get("fencing_token")) is not int
        or type(attempt.current_fencing_token) is not int
        or claim["fencing_token"] != attempt.current_fencing_token
    ):
        return None
    requested_at = attempt.termination.get("requested_at")
    exit_code = attempt.result.get("exit_code")
    return {
        "attempt_id": attempt_id,
        "attempt_number": number,
        "machine_name": attempt.machine_name,
        "phase": attempt.phase,
        "reservation_id": attempt.reservation_id,
        "process_state": "recorded_result" if attempt.phase in _TERMINAL_PHASES else "unknown",
        "stop_pending": isinstance(requested_at, str)
        and bool(requested_at)
        and attempt.termination.get("acknowledged_at") is None,
        "exit_code": exit_code if type(exit_code) is int else None,
    }


def _observe_group(root: Path, name: str) -> dict[str, Any] | None:
    try:
        path = group_path(root, name)
        if not validate_evidence_path(path, root):
            return None
        value = group_visibility(root, name, read_json(path), should_request_repair=False)
        if group_path(root, name) != path:
            return None
        group = value["group"]
        revision = value["meta"]["revision"]
        worker_epoch = group["worker_set_epoch"]
        if (
            group.get("name") != name
            or group.get("dispatch_state") not in GROUP_DISPATCH_STATES
            or type(revision) is not int
            or revision < 0
            or type(worker_epoch) is not int
            or worker_epoch < 0
        ):
            return None
    except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None
    return value


def _placement(state: str, reason: str | None = None, *, group_change: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"state": state, "reason": reason, "group_change": group_change}


def _observe_placement(cfg: RootConfig, task: TaskRecord, groups: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    if task.submission_operation_id is None:
        return _placement("unavailable", "submission_evidence_unavailable")
    if task.control.get("cancellation_requested_at"):
        return _placement("blocked", "task_cancellation_pending")
    if (
        task.control.get("cleanup_operation_id")
        or task.control.get("cleanup_state")
        or operation_exists(cfg, "cleanup", task.task_id)
    ):
        return _placement("blocked", "task_cleanup_pending")
    name = task.group_name
    if name is None:
        return _placement("permits")
    if name not in groups:
        groups[name] = _observe_group(cfg.shared_root, name)
    value = groups[name]
    if value is None:
        return _placement("unavailable", "group_evidence_unavailable")
    group = value["group"]
    if group["dispatch_state"] == "paused":
        return _placement("blocked", "group_paused")
    if has_active_cancellation(cfg, task, value):
        return _placement("blocked", "group_cancellation_active")

    machine = cfg.machine_name
    worker = group["worker_set"].get(machine)
    if worker is None:
        if not task_machine_matches(task, machine):
            return _placement("blocked", "task_machine_mismatch")
        if group.get("pending_submission_commit") is not None:
            return _placement("blocked", "group_submission_pending")
        try:
            validate_identifier(machine, "Group worker")
        except ValueError:
            return _placement("blocked", "worker_name_not_supported")
        return _placement(
            "blocked",
            "worker_missing",
            group_change={
                "action": "add_primary_worker",
                "group_name": name,
                "machine_name": machine,
                "group_revision": value["meta"]["revision"],
                "worker_set_epoch": group["worker_set_epoch"],
            },
        )
    if worker["state"] != "active":
        return _placement("blocked", f"worker_{worker['state']}")
    if not group_allows(value, task, machine):
        return _placement("blocked", "task_machine_mismatch")
    gpu_limit = worker["gpu_limit_gpus"]
    if not task.spec.is_cpu_only and gpu_limit is not None and task.spec.requested_gpus > gpu_limit:
        return _placement("blocked", "worker_gpu_limit")
    if worker["scheduling_role"] == "borrow":
        return _placement("permits", "borrow_admission_required")
    return _placement("permits")


def _describe_task(cfg: RootConfig, task: TaskRecord, groups: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    phase = task.state["projection"]
    claim = task.claim_control.get("active_claim")
    claim_machine = claim.get("machine_name") if claim is not None else None
    if not isinstance(task.control, dict):
        raise ValueError(f"Task {task.task_id!r} has malformed control evidence.")
    cancellation = None
    requested_at = task.control.get("cancellation_requested_at")
    if requested_at is not None:
        terminate_running = task.control.get("terminate_running")
        acknowledged_at = task.control.get("termination_acknowledged_at")
        if (
            not isinstance(requested_at, str)
            or not requested_at
            or type(terminate_running) is not bool
            or (acknowledged_at is not None and (not isinstance(acknowledged_at, str) or not acknowledged_at))
        ):
            raise ValueError(f"Task {task.task_id!r} has malformed cancellation evidence.")
        cancellation = {
            "requested_at": requested_at,
            "terminate_running": terminate_running,
            "acknowledged_at": acknowledged_at,
        }
    attempt = None
    placement = None
    if phase in _TERMINAL_PHASES:
        disposition = "recorded_result"
        attempt = _observe_attempt(cfg, task, claim)
    elif claim_machine == cfg.machine_name:
        attempt = _observe_attempt(cfg, task, claim)
        disposition = "review_interrupted_attempt" if attempt is not None else "attempt_evidence_unavailable"
    elif isinstance(claim_machine, str) and claim_machine:
        disposition = "claimed_elsewhere"
        attempt = _observe_attempt(cfg, task, claim)
    elif phase == "queued" and claim is None:
        disposition = "queued"
        if not isinstance(task.control, dict):
            raise RuntimeError(f"Task {task.task_id!r} has malformed control evidence.")
        placement = _observe_placement(cfg, task, groups)
    else:
        disposition = "task_state_unresolved"
    return {
        "task_id": task.task_id,
        "phase": phase,
        "group_name": task.group_name,
        "home_machine": task.placement_policy["home_machine"],
        "disposition": disposition,
        "claim_machine": claim_machine if isinstance(claim_machine, str) else None,
        "cancellation": cancellation,
        "attempt": attempt,
        "placement": placement,
    }


def observe_takeover_tasks(cfg: RootConfig, *, after_task_id: str | None = None) -> dict[str, Any]:
    """Observe published Task impact for the selected Project and logical machine.

    The count spans the directory and the cursor only limits the advisory page.
    Records may change across reads. Placement is a static policy observation;
    it conveys no execution, retry, process, or reservation authority.
    """
    if after_task_id is not None:
        try:
            validate_identifier(after_task_id, "Task continuation ID")
        except ValueError as exc:
            raise TaskImpactCursorError(str(exc)) from exc

    directory = shared_paths(cfg.shared_root)["tasks"]
    identity = _directory_identity(directory)
    count = 0

    def page_candidates() -> Iterator[TaskRecord]:
        nonlocal count
        for task in _published_target_tasks(cfg, directory):
            count += 1
            if after_task_id is None or task.task_id > after_task_id:
                yield task

    candidates = nsmallest(_PAGE_SIZE + 1, page_candidates(), key=lambda task: task.task_id)
    if _directory_identity(directory) != identity:
        raise RuntimeError(f"Project Task directory changed during observation: {directory}.")

    groups: dict[str, dict[str, Any] | None] = {}
    tasks = [_describe_task(cfg, task, groups) for task in candidates[:_PAGE_SIZE]]
    return {
        "status": "observed",
        "count": count,
        "tasks": tasks,
        "next_task": candidates[_PAGE_SIZE - 1].task_id if len(candidates) > _PAGE_SIZE else None,
    }


def observe_takeover_group_selection(cfg: RootConfig, names: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
    """Bind selected Group proposals to all matching published queued Tasks."""
    if not names:
        return ()

    affected_tasks: dict[str, list[str]] = {}
    for name in names:
        validate_identifier(name, "Group name")
        if name in affected_tasks:
            raise ValueError(f"Duplicate Group selection {name!r}.")
        affected_tasks[name] = []

    directory = shared_paths(cfg.shared_root)["tasks"]
    identity = _directory_identity(directory)
    groups: dict[str, dict[str, Any] | None] = {}
    changes: dict[str, dict[str, Any]] = {}
    for task in _published_target_tasks(cfg, directory):
        name = task.group_name
        if name is None or name not in affected_tasks:
            continue
        if task.state["projection"] != "queued" or task.claim_control.get("active_claim") is not None:
            continue
        if not isinstance(task.control, dict):
            raise RuntimeError(f"Task {task.task_id!r} has malformed control evidence.")
        change = _observe_placement(cfg, task, groups)["group_change"]
        if change is not None and change["action"] == "add_primary_worker":
            changes[name] = change
            affected_tasks[name].append(task.task_id)
    if _directory_identity(directory) != identity:
        raise RuntimeError(f"Project Task directory changed during observation: {directory}.")

    evidence = []
    for name in sorted(affected_tasks):
        task_ids = affected_tasks[name]
        if not task_ids:
            if name in groups and groups[name] is None:
                raise RuntimeError(f"Cannot establish Group {name!r} proposal (group_evidence_unavailable).")
            raise ValueError(f"Group {name!r} has no proposed primary worker addition for a queued Task.")
        encoded_task_ids = json.dumps(sorted(task_ids), separators=(",", ":")).encode("utf-8")
        evidence.append(
            {
                **changes[name],
                "affected_task_count": len(task_ids),
                "affected_tasks_digest": hashlib.sha256(encoded_task_ids).hexdigest(),
            }
        )
    return tuple(evidence)
