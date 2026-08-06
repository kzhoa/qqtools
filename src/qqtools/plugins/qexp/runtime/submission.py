"""Common single and bulk submission transaction with resumable operations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from .locks import group_lock, schema_lock
from .paths import group_path, idempotency_path, shared_paths, submission_path, task_path
from .records import (TaskRecord, TaskSpec, new_group, new_id, new_submission,
                      new_worker_member, utc_now)
from .store import atomic_replace, create_if_absent, read_json
from ..lease import clock_capability, new_timed_offer_proof, persist_clock_observation


class IdempotencyConflict(ValueError):
    pass


def semantic_digest(request: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _resolved_specs(specs: list[dict[str, Any]], home_machine: str) -> list[dict[str, Any]]:
    result = []
    seen: set[str] = set()
    for raw in specs:
        task_id = raw.get("task_id") or new_id()
        if task_id in seen:
            raise ValueError(f"duplicate task_id {task_id!r} in submission.")
        seen.add(task_id)
        result.append({"task_id": task_id, "name": raw.get("name"), "home_machine": home_machine,
                       "command": list(raw["command"]),
                       "working_directory": raw.get("working_directory", str(Path.cwd())),
                       "requested_gpus": raw.get("requested_gpus", 1),
                       "sharing_mode": raw.get("sharing_mode", "private"),
                       "fallback_machines": raw.get("fallback_machines", "group"),
                       "offer_after_seconds": raw.get("offer_after_seconds")})
    return result


def _reject_cleanup_tombstones(cfg: Any, resolved: list[dict[str, Any]]) -> None:
    cleanup_dir = shared_paths(cfg.shared_root)["cleanup"]
    for item in resolved:
        if (cleanup_dir / f"{item['task_id']}.json").exists():
            raise ValueError(
                f"Task {item['task_id']!r} was cleaned and its id cannot be reused."
            )


def submit_specs(cfg: Any, specs: list[dict[str, Any]], *, group_name: str | None = None,
                 idempotency_key: str | None = None, kind: str = "single",
                 worker_set: list[str] | None = None,
                 on_prepared: Callable[[str, str], None] | None = None) -> list[TaskRecord]:
    if not specs:
        raise ValueError("submission must contain at least one task.")
    normalized = {"group": group_name, "tasks": specs, "worker_set": sorted(worker_set or [])}
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
                    return [TaskRecord.from_dict(read_json(task_path(cfg.shared_root, task_id)))
                            for task_id in submission["resolved_context"]["task_ids"]]
                except FileNotFoundError as exc:
                    raise RuntimeError("committed submission has missing Task truth; run qexp doctor repair.") from exc
            if submission["state"] == "aborted":
                raise RuntimeError(f"submission operation was aborted: {submission['failure_reason']}")
            group_name = submission["target_group"]
            resolved = submission["resolved_context"]["task_specs"]
        else:
            resolved = _resolved_specs(specs, cfg.machine_name)
            if any(item["offer_after_seconds"] is not None for item in resolved):
                capability = clock_capability(cfg)
                if not capability.is_healthy or capability.observation is None:
                    raise ValueError(
                        "timed offer requires a healthy clock capability; use an immediate share instead."
                    )
                persist_clock_observation(cfg, capability.observation)
                for item in resolved:
                    if item["offer_after_seconds"] is not None:
                        deadline, proof = new_timed_offer_proof(
                            capability.observation, item["offer_after_seconds"]
                        )
                        item["offer_eligible_at"] = deadline
                        item["offer_clock_evidence"] = proof
            _reject_cleanup_tombstones(cfg, resolved)
            context = {"task_ids": [item["task_id"] for item in resolved], "task_specs": resolved,
                       "create_group": bool(group_name), "worker_set_additions": list(worker_set or []),
                       "group_revision_precondition": None}
            operation_id = new_id()
            operation = new_submission(operation_id=operation_id, kind=kind, key=key,
                                       raw_digest=raw_digest, machine=cfg.machine_name,
                                       target_group=group_name, resolved_context=context)
            if group_name:
                with group_lock(cfg.shared_root, group_name):
                    group_file = group_path(cfg.shared_root, group_name)
                    if group_file.exists() and read_json(group_file)["group"]["admission_state"] != "open":
                        raise ValueError(f"Group {group_name!r} is sealed.")
            create_if_absent(submission_path(cfg.shared_root, operation_id), operation)
            create_if_absent(mapping_path, {"operation_id": operation_id})

        if group_name:
            with group_lock(cfg.shared_root, group_name):
                group_file = group_path(cfg.shared_root, group_name)
                if not group_file.exists():
                    atomic_replace(group_file, new_group(group_name, cfg.machine_name))
                group = read_json(group_file)
                if group["group"]["admission_state"] != "open":
                    operation["submission"]["state"] = "aborted"
                    operation["submission"]["failure_reason"] = f"Group {group_name!r} is sealed."
                    atomic_replace(submission_path(cfg.shared_root, operation_id), operation)
                    raise ValueError(f"Group {group_name!r} is sealed.")
                if operation["submission"]["commit_plan"]["group_membership_sequences"] is None:
                    start = group["group"]["next_membership_sequence"]
                    sequences = list(range(start, start + len(resolved)))
                    operation["submission"]["commit_plan"]["group_membership_sequences"] = sequences
                    operation["submission"]["state"] = "committing"
                    atomic_replace(submission_path(cfg.shared_root, operation_id), operation)
                else:
                    sequences = operation["submission"]["commit_plan"]["group_membership_sequences"]
                group["group"]["pending_submission_commit"] = {
                    "operation_id": operation_id, "membership_sequences": sequences,
                    "worker_set_additions": operation["submission"]["resolved_context"].get("worker_set_additions", [])}
                workers = group["group"]["worker_set"]
                origin = operation["submission"]["original_submitting_machine"]
                for machine in [origin, *operation["submission"]["resolved_context"].get("worker_set_additions", [])]:
                    if machine not in workers:
                        workers[machine] = new_worker_member(added_by_operation=operation_id)
                group["group"]["worker_set_epoch"] += 1
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
                    if current.submission_operation_id != operation_id or current.spec.to_dict() != TaskSpec(
                            item["command"], item["working_directory"], item["requested_gpus"]).to_dict():
                        raise ValueError(f"Task {item['task_id']!r} already exists with different truth.")
                    staged.append(current)
                    continue
                task = TaskRecord.new(task_id=item["task_id"], machine=item["home_machine"],
                    spec=TaskSpec(item["command"], item["working_directory"], item["requested_gpus"]),
                    group_name=group_name, name=item["name"], sharing_mode=item["sharing_mode"],
                    fallback_machines=item["fallback_machines"], offer_after_seconds=item["offer_after_seconds"],
                    offer_eligible_at=item.get("offer_eligible_at"),
                    offer_clock_evidence=item.get("offer_clock_evidence"),
                    operation_id=operation_id)
                task.group_membership_sequence = sequence
                atomic_replace(path, task.to_dict())
                staged.append(task)
            if group_name:
                with group_lock(cfg.shared_root, group_name):
                    group = read_json(group_path(cfg.shared_root, group_name))
                    sequences = operation["submission"]["commit_plan"]["group_membership_sequences"]
                    group["group"]["next_membership_sequence"] = max(group["group"]["next_membership_sequence"], max(sequences, default=0) + 1)
                    group["meta"]["revision"] += 1
                    group["meta"]["updated_at"] = utc_now()
                    group["group"]["pending_submission_commit"] = None
                    atomic_replace(group_path(cfg.shared_root, group_name), group)
            operation["submission"]["state"] = "committed"
            operation["submission"]["committed_at"] = utc_now()
            atomic_replace(submission_path(cfg.shared_root, operation_id), operation)
            return staged
        except Exception as exc:
            operation["submission"]["state"] = "aborted"
            operation["submission"]["failure_reason"] = str(exc)
            atomic_replace(submission_path(cfg.shared_root, operation_id), operation)
            if group_name:
                with group_lock(cfg.shared_root, group_name):
                    group_file = group_path(cfg.shared_root, group_name)
                    if group_file.exists():
                        group = read_json(group_file)
                        if (group["group"].get("pending_submission_commit") or {}).get("operation_id") == operation_id:
                            group["group"]["pending_submission_commit"] = None
                            for machine in operation["submission"]["resolved_context"].get("worker_set_additions", []):
                                worker = group["group"]["worker_set"].get(machine)
                                if worker and worker.get("added_by_operation") == operation_id:
                                    del group["group"]["worker_set"][machine]
                            group["meta"]["revision"] += 1
                            group["meta"]["updated_at"] = utc_now()
                            atomic_replace(group_file, group)
            for task in staged:
                if task.submission_operation_id == operation_id:
                    task_path(cfg.shared_root, task.task_id).unlink(missing_ok=True)
            raise
