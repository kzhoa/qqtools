"""Common single and bulk submission transaction with resumable operations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable

from .locks import group_lock, schema_lock
from .paths import group_path, idempotency_path, submission_path, task_path
from .records import (
    TaskRecord,
    TaskSpec,
    new_group,
    new_id,
    new_submission,
    new_worker_member,
    utc_now,
    validate_identifier,
)
from .ready import (
    assert_ready_writer_compatible,
    delete_ready_marker,
    prepare_ready_transition,
    retire_previous_ready_generation,
)
from .active_operations import operation_exists
from .store import atomic_replace, create_if_absent, read_json
from .availability import remove_deadline_index, sync_deadline_index
from .tasks import save_task
from ..lease import clock_capability, new_timed_offer_proof, persist_clock_observation


class IdempotencyConflict(ValueError):
    pass


class SubmissionResult(list[TaskRecord]):
    def __init__(
        self,
        tasks: Iterable[TaskRecord],
        *,
        operation_id: str,
        idempotency_key: str,
        target_group: str | None,
        state: str,
    ) -> None:
        super().__init__(tasks)
        self.operation_id = operation_id
        self.idempotency_key = idempotency_key
        self.target_group = target_group
        self.state = state

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "idempotency_key": self.idempotency_key,
            "target_group": self.target_group,
            "task_ids": [task.task_id for task in self],
            "state": self.state,
        }


def _submission_result(tasks: Iterable[TaskRecord], submission: dict[str, Any]) -> SubmissionResult:
    return SubmissionResult(
        tasks,
        operation_id=submission["operation_id"],
        idempotency_key=submission["idempotency_key"],
        target_group=submission["target_group"],
        state=submission["state"],
    )


def semantic_digest(request: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _worker_additions(worker_set: list[str] | None) -> list[str]:
    additions = list(worker_set or [])
    seen: set[str] = set()
    for index, machine in enumerate(additions):
        validate_identifier(machine, f"worker_set[{index}]")
        if machine in seen:
            raise ValueError(f"worker_set must not contain duplicate machine {machine!r}.")
        seen.add(machine)
    return additions


def _resolved_home(value: str | None, submitting_machine: str) -> str:
    home = value or "current"
    if home == "current":
        return submitting_machine
    validate_identifier(home, "home_machine")
    return home


def _canonical_specs(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    canonical = []
    for raw in specs:
        item = dict(raw)
        item["home_machine"] = item.get("home_machine", "current")
        canonical.append(item)
    return canonical


def _resolved_specs(specs: list[dict[str, Any]], submitting_machine: str) -> list[dict[str, Any]]:
    result = []
    seen: set[str] = set()
    for raw in specs:
        task_id = raw.get("task_id") or new_id()
        if task_id in seen:
            raise ValueError(f"duplicate task_id {task_id!r} in submission.")
        seen.add(task_id)
        home_machine = _resolved_home(raw.get("home_machine"), submitting_machine)
        result.append(
            {
                "task_id": task_id,
                "name": raw.get("name"),
                "home_machine": home_machine,
                "command": list(raw["command"]),
                "working_directory": raw.get("working_directory", str(Path.cwd())),
                "requested_gpus": raw.get("requested_gpus", 1),
                "sharing_mode": raw.get("sharing_mode", "private"),
                "fallback_machines": raw.get("fallback_machines", "group"),
                "offer_after_seconds": raw.get("offer_after_seconds"),
            }
        )
    return result


def _active_workers(group: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        machine: worker for machine, worker in group["group"]["worker_set"].items() if worker.get("state") == "active"
    }


def _planned_worker_set(
    group: dict[str, Any] | None, submitting_machine: str, additions: list[str]
) -> dict[str, dict[str, Any]]:
    active_workers = _active_workers(group) if group else {}
    all_workers = group["group"]["worker_set"] if group else {}
    planned = dict(active_workers)
    for machine in [submitting_machine, *additions]:
        worker = all_workers.get(machine)
        if worker is not None and worker.get("state") != "active":
            raise ValueError(f"machine {machine!r} is not an active Group worker.")
        planned.setdefault(machine, {"state": "active", "state_epoch": 0, "added_by_operation": None})
    return planned


def _validate_placement_against_workers(
    resolved: list[dict[str, Any]], *, group_name: str | None, planned_workers: dict[str, dict[str, Any]]
) -> None:
    for item in resolved:
        home = item["home_machine"]
        if group_name is None:
            if item["sharing_mode"] != "private":
                raise ValueError("ungrouped tasks must use private placement.")
            if home not in planned_workers:
                raise ValueError("ungrouped tasks must use the submitting machine as home.")
            continue
        if home not in planned_workers:
            raise ValueError(f"tasks home_machine {home!r} is not an active worker in Group {group_name!r}.")
        if item["sharing_mode"] == "private":
            continue
        fallback = item["fallback_machines"]
        if fallback == "group":
            continue
        for machine in fallback:
            if machine not in planned_workers:
                raise ValueError(
                    f"tasks fallback_machines contains {machine!r}, which is not an active "
                    f"worker in Group {group_name!r}."
                )


def _group_precondition(group: dict[str, Any] | None) -> dict[str, Any]:
    if group is None:
        return {"exists": False, "revision": None, "worker_set_epoch": None}
    return {
        "exists": True,
        "revision": group["meta"]["revision"],
        "worker_set_epoch": group["group"]["worker_set_epoch"],
    }


def _validate_group_precondition(group: dict[str, Any], precondition: dict[str, Any], group_name: str) -> None:
    if group["group"]["admission_state"] != "open":
        raise ValueError(f"Group {group_name!r} is sealed.")
    if group["meta"]["revision"] != precondition["revision"]:
        raise RuntimeError(f"Group {group_name!r} changed during submission.")
    if group["group"]["worker_set_epoch"] != precondition["worker_set_epoch"]:
        raise RuntimeError(f"Group {group_name!r} Worker Set changed during submission.")


def _reject_cleanup_tombstones(cfg: Any, resolved: list[dict[str, Any]]) -> None:
    for item in resolved:
        if operation_exists(cfg, "cleanup", item["task_id"]):
            raise ValueError(f"Task {item['task_id']!r} was cleaned and its id cannot be reused.")


def _task_matches_resolved(
    current: TaskRecord, item: dict[str, Any], operation_id: str, group_name: str | None
) -> bool:
    spec = TaskSpec(item["command"], item["working_directory"], item["requested_gpus"])
    expected_policy = {
        "home_machine": item["home_machine"],
        "sharing_mode": item["sharing_mode"],
        "fallback_constraint": item["fallback_machines"],
        "offer_after_seconds": item["offer_after_seconds"],
    }
    return (
        current.submission_operation_id == operation_id
        and current.group_name == group_name
        and current.name == item["name"]
        and current.spec.to_dict() == spec.to_dict()
        and current.placement_policy == expected_policy
    )


def _remove_operation_added_workers(group: dict[str, Any], operation: dict[str, Any]) -> bool:
    origin = operation["submission"]["original_submitting_machine"]
    additions = operation["submission"]["resolved_context"].get("worker_set_additions", [])
    removed_worker = False
    for machine in dict.fromkeys([origin, *additions]):
        worker = group["group"]["worker_set"].get(machine)
        if worker and worker.get("added_by_operation") == operation["submission"]["operation_id"]:
            del group["group"]["worker_set"][machine]
            removed_worker = True
    if removed_worker:
        group["group"]["worker_set_epoch"] += 1
    return removed_worker


def submit_specs(
    cfg: Any,
    specs: list[dict[str, Any]],
    *,
    group_name: str | None = None,
    idempotency_key: str | None = None,
    kind: str = "single",
    worker_set: list[str] | None = None,
    on_prepared: Callable[[str, str], None] | None = None,
) -> SubmissionResult:
    if not specs:
        raise ValueError("submission must contain at least one task.")
    worker_additions = _worker_additions(worker_set)
    normalized = {"group": group_name, "tasks": _canonical_specs(specs), "worker_set": sorted(worker_additions)}
    raw_digest = semantic_digest(normalized)
    key = idempotency_key or new_id()
    mapping_path = idempotency_path(cfg.shared_root, semantic_digest({"project": str(cfg.shared_root), "key": key}))
    with schema_lock(cfg.shared_root):
        existing = mapping_path.exists()
        if existing:
            operation_id = read_json(mapping_path)["operation_id"]
            operation = read_json(submission_path(cfg.shared_root, operation_id))
            submission = operation["submission"]
            if submission["raw_request_digest"] != raw_digest:
                raise IdempotencyConflict("idempotency key was already used with different semantic input.")
            if submission["state"] == "committed":
                try:
                    return _submission_result(
                        [
                            TaskRecord.from_dict(read_json(task_path(cfg.shared_root, task_id)))
                            for task_id in submission["resolved_context"]["task_ids"]
                        ],
                        submission,
                    )
                except FileNotFoundError as exc:
                    raise RuntimeError("committed submission has missing Task truth; run qexp doctor repair.") from exc
            if submission["state"] == "aborted":
                raise RuntimeError(f"submission operation was aborted: {submission['failure_reason']}")
            group_name = submission["target_group"]
            resolved = submission["resolved_context"]["task_specs"]
            operation_id = submission["operation_id"]
            operation = read_json(submission_path(cfg.shared_root, operation_id))
        else:
            resolved = _resolved_specs(specs, cfg.machine_name)
            group_precondition = _group_precondition(None)
            planned_workers: dict[str, dict[str, Any]] = {}
            if group_name:
                with group_lock(cfg.shared_root, group_name):
                    group_file = group_path(cfg.shared_root, group_name)
                    group = read_json(group_file) if group_file.exists() else None
                    if group is not None and group["group"]["admission_state"] != "open":
                        raise ValueError(f"Group {group_name!r} is sealed.")
                    group_precondition = _group_precondition(group)
                    planned_workers = _planned_worker_set(group, cfg.machine_name, worker_additions)
                    _validate_placement_against_workers(
                        resolved, group_name=group_name, planned_workers=planned_workers
                    )
            else:
                _validate_placement_against_workers(resolved, group_name=None, planned_workers={cfg.machine_name: {}})
            if any(item["offer_after_seconds"] is not None for item in resolved):
                capability = clock_capability(cfg)
                if not capability.is_healthy or capability.observation is None:
                    raise ValueError("timed offer requires a healthy clock capability; use an immediate share instead.")
                persist_clock_observation(cfg, capability.observation)
                for item in resolved:
                    if item["offer_after_seconds"] is not None:
                        deadline, proof = new_timed_offer_proof(capability.observation, item["offer_after_seconds"])
                        item["offer_eligible_at"] = deadline
                        item["offer_clock_evidence"] = proof
            _reject_cleanup_tombstones(cfg, resolved)
            context = {
                "task_ids": [item["task_id"] for item in resolved],
                "task_specs": resolved,
                "create_group": bool(group_name and not group_precondition["exists"]),
                "worker_set_additions": list(worker_additions),
                "group_precondition": group_precondition,
                "planned_worker_set": sorted(planned_workers),
            }
            operation_id = new_id()
            operation = new_submission(
                operation_id=operation_id,
                kind=kind,
                key=key,
                raw_digest=raw_digest,
                machine=cfg.machine_name,
                target_group=group_name,
                resolved_context=context,
            )
            create_if_absent(submission_path(cfg.shared_root, operation_id), operation)
            create_if_absent(mapping_path, {"operation_id": operation_id})

        if group_name:
            with group_lock(cfg.shared_root, group_name):
                group_file = group_path(cfg.shared_root, group_name)
                if not group_file.exists():
                    atomic_replace(group_file, new_group(group_name, cfg.machine_name))
                group = read_json(group_file)
                precondition = operation["submission"]["resolved_context"].get("group_precondition") or {}
                pending = group["group"].get("pending_submission_commit") or {}
                has_own_pending_commit = pending.get("operation_id") == operation_id
                if pending and not has_own_pending_commit:
                    raise RuntimeError(
                        f"Group {group_name!r} has pending submission commit {pending['operation_id']!r}."
                    )
                if precondition.get("exists") and not has_own_pending_commit:
                    _validate_group_precondition(group, precondition, group_name)
                elif not precondition.get("exists") and group["group"]["worker_set"] and not has_own_pending_commit:
                    operation["submission"]["state"] = "aborted"
                    operation["submission"]["failure_reason"] = f"Group {group_name!r} changed during submission."
                    atomic_replace(submission_path(cfg.shared_root, operation_id), operation)
                    raise RuntimeError(f"Group {group_name!r} changed during submission.")
                if group["group"]["admission_state"] != "open":
                    operation["submission"]["state"] = "aborted"
                    operation["submission"]["failure_reason"] = f"Group {group_name!r} is sealed."
                    atomic_replace(submission_path(cfg.shared_root, operation_id), operation)
                    raise ValueError(f"Group {group_name!r} is sealed.")
                planned_workers = _planned_worker_set(
                    group,
                    operation["submission"]["original_submitting_machine"],
                    operation["submission"]["resolved_context"].get("worker_set_additions", []),
                )
                _validate_placement_against_workers(resolved, group_name=group_name, planned_workers=planned_workers)
                if operation["submission"]["commit_plan"]["group_membership_sequences"] is None:
                    start = group["group"]["next_membership_sequence"]
                    sequences = list(range(start, start + len(resolved)))
                    operation["submission"]["commit_plan"]["group_membership_sequences"] = sequences
                    operation["submission"]["state"] = "committing"
                    atomic_replace(submission_path(cfg.shared_root, operation_id), operation)
                else:
                    sequences = operation["submission"]["commit_plan"]["group_membership_sequences"]
                group["group"]["pending_submission_commit"] = {
                    "operation_id": operation_id,
                    "membership_sequences": sequences,
                    "worker_set_additions": operation["submission"]["resolved_context"].get("worker_set_additions", []),
                }
                workers = group["group"]["worker_set"]
                origin = operation["submission"]["original_submitting_machine"]
                added_workers: list[dict[str, Any]] = []
                for machine in dict.fromkeys(
                    [origin, *operation["submission"]["resolved_context"].get("worker_set_additions", [])]
                ):
                    if machine not in workers:
                        worker = new_worker_member(added_by_operation=operation_id)
                        workers[machine] = worker
                        added_workers.append(worker)
                if added_workers:
                    group["group"]["worker_set_epoch"] += 1
                    for worker in added_workers:
                        worker["state_epoch"] = group["group"]["worker_set_epoch"]
                group["meta"]["revision"] += 1
                group["meta"]["updated_at"] = utc_now()
                atomic_replace(group_file, group)
        else:
            sequences = [None] * len(resolved)

        staged: list[TaskRecord] = []
        try:
            if on_prepared:
                on_prepared(operation_id, operation["submission"]["idempotency_key"])
            for sequence, item in zip(sequences, resolved):
                path = task_path(cfg.shared_root, item["task_id"])
                _reject_cleanup_tombstones(cfg, [item])
                if path.exists():
                    current = TaskRecord.from_dict(read_json(path))
                    if not _task_matches_resolved(current, item, operation_id, group_name):
                        raise ValueError(f"Task {item['task_id']!r} already exists with different truth.")
                    if current.ready_generation == 0:
                        old_generation, _ = prepare_ready_transition(
                            cfg, current, "submission_resume"
                        )
                        current.meta["revision"] += 1
                        current.meta["updated_at"] = utc_now()
                        save_task(cfg, current)
                        retire_previous_ready_generation(cfg, old_generation, current)
                    sync_deadline_index(cfg, current)
                    staged.append(current)
                    continue
                task = TaskRecord.new(
                    task_id=item["task_id"],
                    machine=item["home_machine"],
                    spec=TaskSpec(item["command"], item["working_directory"], item["requested_gpus"]),
                    group_name=group_name,
                    name=item["name"],
                    sharing_mode=item["sharing_mode"],
                    fallback_machines=item["fallback_machines"],
                    offer_after_seconds=item["offer_after_seconds"],
                    offer_eligible_at=item.get("offer_eligible_at"),
                    offer_clock_evidence=item.get("offer_clock_evidence"),
                    operation_id=operation_id,
                )
                task.group_membership_sequence = sequence
                old_generation, _ = prepare_ready_transition(
                    cfg,
                    task,
                    "submission_stage",
                    target_revision=task.meta["revision"],
                )
                save_task(cfg, task)
                retire_previous_ready_generation(cfg, old_generation, task)
                sync_deadline_index(cfg, task)
                staged.append(task)
            if group_name:
                with group_lock(cfg.shared_root, group_name):
                    group = read_json(group_path(cfg.shared_root, group_name))
                    sequences = operation["submission"]["commit_plan"]["group_membership_sequences"]
                    group["group"]["next_membership_sequence"] = max(
                        group["group"]["next_membership_sequence"], max(sequences, default=0) + 1
                    )
                    group["meta"]["revision"] += 1
                    group["meta"]["updated_at"] = utc_now()
                    group["group"]["pending_submission_commit"] = None
                    atomic_replace(group_path(cfg.shared_root, group_name), group)
            operation["submission"]["state"] = "committed"
            operation["submission"]["committed_at"] = utc_now()
            atomic_replace(submission_path(cfg.shared_root, operation_id), operation)
            return _submission_result(staged, operation["submission"])
        except Exception as exc:
            operation["submission"]["state"] = "aborted"
            operation["submission"]["failure_reason"] = str(exc)
            atomic_replace(submission_path(cfg.shared_root, operation_id), operation)
            if group_name:
                with group_lock(cfg.shared_root, group_name):
                    group_file = group_path(cfg.shared_root, group_name)
                    if group_file.exists():
                        group = read_json(group_file)
                        has_pending_commit = (group["group"].get("pending_submission_commit") or {}).get(
                            "operation_id"
                        ) == operation_id
                        removed_worker = _remove_operation_added_workers(group, operation)
                        if has_pending_commit or removed_worker:
                            group["group"]["pending_submission_commit"] = None
                            group["meta"]["revision"] += 1
                            group["meta"]["updated_at"] = utc_now()
                            atomic_replace(group_file, group)
            for item in operation["submission"]["resolved_context"].get("task_specs", []):
                task_id = item["task_id"]
                path = task_path(cfg.shared_root, task_id)
                try:
                    current = TaskRecord.from_dict(read_json(path))
                except FileNotFoundError:
                    remove_deadline_index(cfg, task_id)
                    continue
                if current.submission_operation_id == operation_id:
                    remove_deadline_index(cfg, task_id)
                    try:
                        delete_ready_marker(cfg, task_id, current.ready_generation)
                    except (OSError, KeyError, TypeError, ValueError):
                        pass
                    assert_ready_writer_compatible(cfg)
                    path.unlink(missing_ok=True)
            raise
