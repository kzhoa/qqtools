"""Durable Task availability transitions for queued placement controls."""
from __future__ import annotations

import os
import heapq
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
from typing import Any, Literal

from ..config_types import RootConfig
from ..events import write_diagnostic_event
from ..lease import (ClockObservation, clock_capability, new_timed_offer_proof,
                     persist_clock_observation, timed_offer_deadline_upper)
from .locks import group_lock, task_lock
from .locks import schema_lock
from .paths import group_path, shared_paths, submission_path
from .records import SCHEMA_VERSION, TaskRecord, new_id, normalize_group_record, utc_now
from .ready import (
    discard_ready_generation,
    prepare_ready_transition,
    reserve_ready_generation,
    retire_previous_ready_generation,
)
from .store import atomic_replace, iter_json, read_json
from .tasks import load_task, save_task
from .active_operations import (
    active_operation_path,
    archive_operation,
    iter_active_operation_paths,
    locate_operation_path,
    operation_exists,
    write_active_operation,
)

AvailabilityAction = Literal["share_now", "share_after", "keep_local", "manual_offer", "elapsed_offer"]


@dataclass(slots=True)
class AvailabilityTransitionRequest:
    action: AvailabilityAction
    task_id: str
    helper_machines: list[str] | None = None
    after_seconds: int | None = None
    reason: str = "manual"
    operation_id: str | None = None


@dataclass(slots=True)
class AvailabilityTransitionResult:
    action: AvailabilityAction
    task_id: str
    group: str | None
    home_machine: str
    eligible_helper_machines: list[str]
    effective_at: str | None
    resulting_state: str
    idempotent: bool
    operation_id: str
    message: str
    task: TaskRecord

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values.pop("task")
        return values


def _operation_path(cfg: RootConfig, operation_id: str) -> Path:
    return locate_operation_path(cfg, "availability", operation_id)


def _operation_meta(cfg: RootConfig) -> dict[str, Any]:
    now = utc_now()
    return {"schema_version": SCHEMA_VERSION, "revision": 1, "created_at": now,
            "updated_at": now, "updated_by": {"actor_type": "cli",
            "machine_name": cfg.machine_name, "process_id": str(os.getpid())}}


def _create_operation(
        cfg: RootConfig, request: AvailabilityTransitionRequest) -> tuple[str, dict[str, Any]]:
    operation_id = request.operation_id or new_id()
    path = _operation_path(cfg, operation_id)
    if path.exists():
        return operation_id, read_json(path)
    now = utc_now()
    operation = {"meta": _operation_meta(cfg), "availability_operation": {
        "operation_id": operation_id, "operation_type": request.action,
        "task_id": request.task_id, "state": "prepared",
        "requested_by": cfg.machine_name, "reason": request.reason,
        "helper_machines": list(request.helper_machines) if request.helper_machines is not None else None,
        "after_seconds": request.after_seconds, "created_at": now, "updated_at": now,
        "completed_at": None, "blocked_reason": None, "task_revision_before": None,
        "task_revision_after": None, "result": None}}
    write_active_operation(cfg, "availability", operation_id, operation)
    return operation_id, operation


def _update_operation(
        cfg: RootConfig, operation: dict[str, Any], *, state: str,
        blocked_reason: str | None = None,
        result: AvailabilityTransitionResult | None = None,
        revision_before: int | None = None,
        revision_after: int | None = None) -> None:
    control = operation["availability_operation"]
    control["state"] = state
    control["updated_at"] = utc_now()
    control["blocked_reason"] = blocked_reason
    if revision_before is not None:
        control["task_revision_before"] = revision_before
    if revision_after is not None:
        control["task_revision_after"] = revision_after
    if result is not None:
        control["result"] = result.to_dict()
    if state == "completed":
        control["completed_at"] = control["completed_at"] or utc_now()
    operation["meta"]["revision"] += 1
    operation["meta"]["updated_at"] = utc_now()
    if state == "completed":
        archive_operation(cfg, "availability", control["operation_id"], operation)
    else:
        write_active_operation(cfg, "availability", control["operation_id"], operation)


def clock_evidence(cfg: RootConfig) -> tuple[ClockObservation, datetime, float]:
    capability = clock_capability(cfg)
    if not capability.is_healthy or capability.observation is None:
        raise ValueError(
            "timed sharing requires a healthy clock capability; use immediate share instead."
        )
    persist_clock_observation(cfg, capability.observation)
    return capability.observation, datetime.now(timezone.utc), time.monotonic()


def elapsed_offer_is_proven(cfg: RootConfig, task: TaskRecord) -> bool:
    proof = task.placement_runtime.get("offer_clock_evidence")
    deadline = task.placement_runtime.get("offer_eligible_at")
    if not isinstance(proof, dict) or not isinstance(deadline, str):
        return False
    try:
        observation, now, monotonic_now = clock_evidence(cfg)
    except ValueError:
        return False
    evaluator_lower = now - timedelta(seconds=observation.bound_at(monotonic_now))
    try:
        deadline_upper = timed_offer_deadline_upper(deadline, proof)
    except (TypeError, ValueError, OverflowError):
        return False
    return evaluator_lower >= deadline_upper


def _cleanup_blocked(cfg: RootConfig, task: TaskRecord) -> bool:
    return bool(task.control.get("cleanup_operation_id")
                or task.control.get("cleanup_state")
                or operation_exists(cfg, "cleanup", task.task_id))


def _submission_committed(cfg: RootConfig, task: TaskRecord) -> bool:
    operation_id = task.submission_operation_id
    if not operation_id:
        return False
    path = submission_path(cfg.shared_root, operation_id)
    return path.exists() and read_json(path).get("submission", {}).get("state") == "committed"


def _group_data(cfg: RootConfig, task: TaskRecord) -> dict[str, Any] | None:
    if not task.group_name:
        return None
    group = read_json(group_path(cfg.shared_root, task.group_name))
    normalize_group_record(group)
    return group


def _group_cancel_blocked(group: dict[str, Any], task: TaskRecord) -> bool:
    sequence = task.group_membership_sequence or 0
    for barrier in group["group"].get("cancellation_barriers", []):
        if sequence <= barrier.get("membership_high_watermark", 0):
            return True
    return False


def _validate_common(cfg: RootConfig, task: TaskRecord, group: dict[str, Any] | None) -> None:
    if _cleanup_blocked(cfg, task):
        raise ValueError(f"Task {task.task_id!r} is being cleaned and cannot change placement.")
    if task.state["projection"] != "queued" or task.claim_control.get("active_claim"):
        raise ValueError("placement can only change while a Task is queued and unclaimed.")
    if task.control.get("cancellation_requested_at") or task.control.get("cancellation_operation_id"):
        raise ValueError("cancelled Tasks cannot change placement.")
    if not _submission_committed(cfg, task):
        raise ValueError("placement can only change after Task submission is committed.")
    if group is None:
        return
    if group["group"].get("pending_submission_commit"):
        raise ValueError(f"Group {task.group_name!r} has a pending submission commit.")
    if _group_cancel_blocked(group, task):
        raise ValueError(f"Group {task.group_name!r} is being cancelled.")
    home = task.placement_policy["home_machine"]
    worker = group["group"]["worker_set"].get(home)
    if not worker or worker.get("state") != "active":
        raise ValueError(f"Task home machine {home!r} is not a claimable Group worker.")


def _normalize_helpers(
        task: TaskRecord, group: dict[str, Any] | None,
        helper_machines: list[str] | None) -> str | list[str]:
    if helper_machines is None:
        return "group"
    if len(set(helper_machines)) != len(helper_machines):
        raise ValueError("shared helper machines must be unique.")
    home = task.placement_policy["home_machine"]
    if home in helper_machines:
        raise ValueError("the home machine is already eligible and cannot be listed as a helper.")
    if group is None:
        raise ValueError(
            f"Task {task.task_id!r} is local-only because it does not belong to a Group. "
            "Submit the work to a Group to let other machines help."
        )
    workers = group["group"]["worker_set"]
    invalid = [
        machine
        for machine in helper_machines
        if workers.get(machine, {}).get("state") != "active"
    ]
    if invalid:
        raise ValueError(f"shared helpers are not active Group workers: {invalid}")
    return list(helper_machines) if helper_machines else "group"


def _eligible_helpers(task: TaskRecord, group: dict[str, Any] | None) -> list[str]:
    if group is None:
        return []
    home = task.placement_policy["home_machine"]
    fallback = task.placement_policy.get("fallback_constraint", "group")
    workers = group["group"]["worker_set"]
    if fallback == "group":
        return sorted(machine for machine, worker in workers.items()
                      if machine != home and worker.get("state") == "active")
    return sorted(
        machine
        for machine in fallback
        if machine != home and workers.get(machine, {}).get("state") == "active"
    )


def _is_same_immediate_share(
        task: TaskRecord, group: dict[str, Any] | None,
        helper_machines: list[str] | None) -> bool:
    fallback = _normalize_helpers(task, group, helper_machines)
    return (task.placement_policy.get("sharing_mode") == "spillover"
            and task.placement_policy.get("fallback_constraint") == fallback
            and task.placement_runtime.get("queue_scope") == "shared")


def _result_message(action: AvailabilityAction, task: TaskRecord, group: dict[str, Any] | None,
                    effective_at: str | None) -> str:
    if action == "keep_local":
        return (f"Task {task.task_id} is now restricted to its home machine "
                f"{task.placement_policy['home_machine']}.")
    if action == "share_after":
        return (f"Task {task.task_id} will stay on {task.placement_policy['home_machine']} until "
                f"{effective_at}, then become available to eligible Group workers.")
    group_name = task.group_name or ""
    suffix = ""
    if group is not None and group["group"].get("dispatch_state") == "paused":
        suffix = " The Group is paused, so workers can claim it after the Group resumes."
    return f"Task {task.task_id} is now available to eligible workers in Group {group_name}.{suffix}"


def _make_result(action: AvailabilityAction, task: TaskRecord, group: dict[str, Any] | None,
                 operation_id: str, *, idempotent: bool) -> AvailabilityTransitionResult:
    effective_at = (task.placement_runtime.get("offer_eligible_at")
                    if action == "share_after" else task.placement_runtime.get("offered_at"))
    if action == "keep_local":
        effective_at = task.placement_runtime.get("queued_home_at")
    return AvailabilityTransitionResult(
        action=action, task_id=task.task_id, group=task.group_name,
        home_machine=task.placement_policy["home_machine"],
        eligible_helper_machines=_eligible_helpers(task, group),
        effective_at=effective_at, resulting_state=task.placement_runtime["queue_scope"],
        idempotent=idempotent, operation_id=operation_id,
        message=_result_message(action, task, group, effective_at), task=task,
    )


def _write_audit_event(cfg: RootConfig, result: AvailabilityTransitionResult,
                       before: dict[str, Any], clock_proof: dict[str, Any] | None) -> None:
    timestamp = utc_now()
    day = timestamp[:10]
    event = {"event_id": result.operation_id, "event_type": "task_availability_changed",
             "task_id": result.task_id, "machine_name": cfg.machine_name,
             "timestamp": timestamp, "details": {"action": result.action,
             "operation_id": result.operation_id, "before": before,
             "after": {"placement_policy": result.task.placement_policy,
                       "placement_runtime": result.task.placement_runtime},
             "clock_evidence": clock_proof}}
    atomic_replace(shared_paths(cfg.shared_root)["events"] / day / f"{result.operation_id}.json", event)


def _deadline_index_path(cfg: RootConfig, task_id: str) -> Path:
    return shared_paths(cfg.shared_root)["offer_deadlines"] / f"{task_id}.json"


def _deadline_bucket(value: str) -> str:
    deadline = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return deadline.astimezone(timezone.utc).strftime("%Y%m%d%H")


def _active_deadline_path(cfg: RootConfig, task: TaskRecord) -> Path:
    bucket = _deadline_bucket(task.placement_runtime["offer_eligible_at"])
    return (
        shared_paths(cfg.shared_root)["offer_deadlines_active"]
        / task.placement_policy["home_machine"]
        / bucket
        / f"{task.task_id}.json"
    )


def remove_deadline_index(cfg: RootConfig, task_id: str) -> None:
    stable = _deadline_index_path(cfg, task_id)
    if stable.exists() or stable.is_symlink():
        try:
            target = stable.resolve(strict=True)
        except FileNotFoundError:
            target = None
        stable.unlink(missing_ok=True)
        if target is not None:
            target.unlink(missing_ok=True)
            bucket = target.parent
            home = bucket.parent
            try:
                bucket.rmdir()
            except OSError:
                pass
            try:
                home.rmdir()
            except OSError:
                pass


def sync_deadline_index(cfg: RootConfig, task: TaskRecord) -> None:
    path = _deadline_index_path(cfg, task.task_id)
    if (task.placement_runtime.get("queue_scope") == "home"
            and task.placement_policy.get("sharing_mode") == "spillover"
            and task.placement_runtime.get("offer_eligible_at")
            and task.placement_runtime.get("offer_clock_evidence")):
        desired = {"offer_deadline": {"task_id": task.task_id,
            "group_name": task.group_name, "home_machine": task.placement_policy["home_machine"],
            "offer_eligible_at": task.placement_runtime["offer_eligible_at"],
            "operation_id": task.placement_runtime.get("availability_operation_id"),
            "updated_at": task.meta["updated_at"]}}
        active = _active_deadline_path(cfg, task)
        if path.exists():
            try:
                if read_json(path) == desired:
                    return
            except (KeyError, TypeError, ValueError):
                pass
        remove_deadline_index(cfg, task.task_id)
        atomic_replace(active, desired)
        try:
            path.symlink_to(active.relative_to(path.parent))
        except FileExistsError:
            pass
        return
    remove_deadline_index(cfg, task.task_id)


def iter_due_deadline_paths(cfg: RootConfig, *, limit: int = 64):
    """Yield bounded due records for this home machine from time buckets only."""
    if limit <= 0:
        raise ValueError("deadline limit must be positive.")
    home = shared_paths(cfg.shared_root)["offer_deadlines_active"] / cfg.machine_name
    if not home.exists():
        return
    current_bucket = datetime.now(timezone.utc).strftime("%Y%m%d%H")
    yielded = 0
    due_buckets = heapq.nsmallest(
        limit,
        (
            Path(entry.path)
            for entry in os.scandir(home)
            if entry.is_dir() and entry.name <= current_bucket
        ),
        key=lambda path: path.name,
    )
    for bucket in due_buckets:
        with os.scandir(bucket) as entries:
            for entry in entries:
                if not entry.is_file() or not entry.name.endswith(".json"):
                    continue
                if yielded >= limit:
                    return
                yielded += 1
                yield Path(entry.path)


def migrate_legacy_deadline_indexes(cfg: RootConfig) -> None:
    """Move legacy flat deadline records into home/time buckets once."""
    marker = shared_paths(cfg.shared_root)["offer_deadlines_migration"]
    if marker.exists():
        return
    with schema_lock(cfg.shared_root):
        if marker.exists():
            return
        root = shared_paths(cfg.shared_root)["offer_deadlines"]
        for path in root.iterdir():
            if not path.is_file() or path.is_symlink() or path.name == marker.name:
                continue
            try:
                record = read_json(path)["offer_deadline"]
                home_machine = record["home_machine"]
                bucket = _deadline_bucket(record["offer_eligible_at"])
            except (KeyError, TypeError, ValueError):
                path.unlink(missing_ok=True)
                continue
            active = (
                shared_paths(cfg.shared_root)["offer_deadlines_active"]
                / home_machine
                / bucket
                / path.name
            )
            atomic_replace(active, {"offer_deadline": record})
            path.unlink(missing_ok=True)
            path.symlink_to(active.relative_to(path.parent))
        atomic_replace(marker, {"offer_deadline_layout": {"version": 1}})


def _same_delayed_share(task: TaskRecord, fallback: str | list[str],
                        after_seconds: int | None) -> bool:
    return (task.placement_policy.get("sharing_mode") == "spillover"
            and task.placement_policy.get("fallback_constraint") == fallback
            and task.placement_policy.get("offer_after_seconds") == after_seconds
            and task.placement_runtime.get("queue_scope") == "home"
            and bool(task.placement_runtime.get("offer_eligible_at"))
            and bool(task.placement_runtime.get("offer_clock_evidence")))


def apply_availability_transition(
        cfg: RootConfig, request: AvailabilityTransitionRequest) -> AvailabilityTransitionResult:
    if request.after_seconds is not None and request.after_seconds < 0:
        raise ValueError("share --after must be non-negative.")
    operation_id, operation = _create_operation(cfg, request)
    initial = load_task(cfg, request.task_id)
    lock_group = initial.group_name
    target_scope = (
        "shared" if request.action in {"share_now", "manual_offer", "elapsed_offer"}
        else "home"
    )
    reserved_generation = initial.ready_generation + 1
    ready_reference = reserve_ready_generation(
        cfg,
        initial.task_id,
        reserved_generation,
        target_scope,
        initial.placement_policy["home_machine"],
    )
    is_ready_committed = False
    try:
        with ExitStack() as stack:
            if lock_group:
                stack.enter_context(group_lock(cfg.shared_root, lock_group))
            stack.enter_context(task_lock(cfg.shared_root, request.task_id))
            task = load_task(cfg, request.task_id)
            group = _group_data(cfg, task)
            if request.action == "elapsed_offer":
                if task.placement_policy["home_machine"] != cfg.machine_name:
                    result = _make_result(request.action, task, group, operation_id, idempotent=True)
                    _update_operation(cfg, operation, state="completed", result=result)
                    return result
                if not elapsed_offer_is_proven(cfg, task):
                    result = _make_result(request.action, task, group, operation_id, idempotent=True)
                    sync_deadline_index(cfg, task)
                    _update_operation(cfg, operation, state="completed", result=result)
                    return result
            if request.action == "share_now" and task.group_name is None:
                raise ValueError(
                    f"Task {task.task_id!r} is local-only because it does not belong to a Group. "
                    "Submit the work to a Group to let other machines help."
                )
            if request.action == "manual_offer" and task.placement_policy["sharing_mode"] != "spillover":
                raise ValueError("private Tasks cannot be offered to shared workers; use task share.")
            _validate_common(cfg, task, group)
            before = {"placement_policy": dict(task.placement_policy),
                      "placement_runtime": dict(task.placement_runtime)}
            revision_before = task.meta["revision"]
            idempotent = False
            clock_proof = None
            if request.action == "keep_local":
                idempotent = (task.placement_policy["sharing_mode"] == "private"
                              and task.placement_runtime["queue_scope"] == "home"
                              and not task.placement_runtime.get("offer_eligible_at"))
                if not idempotent:
                    task.placement_policy.update({"sharing_mode": "private",
                        "fallback_constraint": "group", "offer_after_seconds": None})
                    task.placement_runtime.update({"queue_scope": "home", "queued_home_at": utc_now(),
                        "offer_eligible_at": None, "offer_clock_evidence": None,
                        "offered_at": None, "offer_reason": None, "offered_by": cfg.machine_name,
                        "availability_operation_id": operation_id})
            elif request.action == "share_after":
                if task.group_name is None:
                    raise ValueError(
                        f"Task {task.task_id!r} is local-only because it does not belong to a Group. "
                        "Submit the work to a Group to let other machines help."
                    )
                if task.placement_runtime["queue_scope"] != "home":
                    raise ValueError("share --after requires a home-queued Task; use keep-local first.")
                fallback = _normalize_helpers(task, group, request.helper_machines)
                idempotent = _same_delayed_share(task, fallback, request.after_seconds)
                if not idempotent:
                    observation, wall_now, monotonic_now = clock_evidence(cfg)
                    deadline, clock_proof = new_timed_offer_proof(
                        observation, request.after_seconds or 0,
                        wall_now=wall_now, monotonic_now=monotonic_now,
                    )
                    task.placement_policy.update({"sharing_mode": "spillover",
                        "fallback_constraint": fallback,
                        "offer_after_seconds": request.after_seconds})
                    task.placement_runtime.update({"queue_scope": "home", "queued_home_at": utc_now(),
                        "offer_eligible_at": deadline, "offer_clock_evidence": clock_proof,
                        "offered_at": None, "offer_reason": None, "offered_by": cfg.machine_name,
                        "availability_operation_id": operation_id})
            else:
                if request.action == "share_now":
                    idempotent = _is_same_immediate_share(task, group, request.helper_machines)
                    fallback = _normalize_helpers(task, group, request.helper_machines)
                    task.placement_policy["sharing_mode"] = "spillover"
                    task.placement_policy["fallback_constraint"] = fallback
                else:
                    idempotent = (task.placement_runtime["queue_scope"] == "shared"
                                  and task.placement_policy["sharing_mode"] == "spillover")
                if not idempotent:
                    task.placement_policy["offer_after_seconds"] = None
                    task.placement_runtime.update({"queue_scope": "shared",
                        "offer_eligible_at": None, "offer_clock_evidence": None,
                        "offered_at": utc_now(), "offer_reason": request.reason,
                        "offered_by": cfg.machine_name,
                        "availability_operation_id": operation_id})
            if not idempotent:
                old_generation, _ = prepare_ready_transition(
                    cfg,
                    task,
                    f"availability_{request.action}",
                    reference=ready_reference,
                )
                task.meta["revision"] += 1
                task.meta["updated_at"] = utc_now()
                save_task(cfg, task)
                is_ready_committed = True
                retire_previous_ready_generation(cfg, old_generation, task)
            sync_deadline_index(cfg, task)
            result = _make_result(request.action, task, group, operation_id, idempotent=idempotent)
            _write_audit_event(cfg, result, before, clock_proof)
            _update_operation(cfg, operation, state="completed", result=result,
                              revision_before=revision_before,
                              revision_after=task.meta["revision"])
            return result
    except Exception as exc:
        try:
            current = load_task(cfg, request.task_id)
            if current.placement_runtime.get("availability_operation_id") == operation_id:
                control = operation["availability_operation"]
                control["state"] = "prepared"
                control["blocked_reason"] = None
                control["updated_at"] = utc_now()
                operation["meta"]["revision"] += 1
                operation["meta"]["updated_at"] = utc_now()
                atomic_replace(_operation_path(cfg, operation_id), operation)
            else:
                _update_operation(cfg, operation, state="blocked", blocked_reason=str(exc))
        except Exception:
            _update_operation(cfg, operation, state="blocked", blocked_reason=str(exc))
        raise
    finally:
        if not is_ready_committed:
            discard_ready_generation(cfg, initial.task_id, reserved_generation)


def reconcile_availability_operations(
    cfg: RootConfig, *, include_legacy: bool = True,
) -> list[dict[str, Any]]:
    reconciled: list[dict[str, Any]] = []
    for path in iter_active_operation_paths(
        cfg, "availability", include_legacy=include_legacy
    ):
        operation = read_json(path)
        control = operation.get("availability_operation", {})
        if control.get("state") not in {"prepared", "blocked"}:
            continue
        if control.get("state") == "blocked" and control.get("blocked_reason"):
            archive_operation(cfg, "availability", control["operation_id"], operation)
            continue
        request = AvailabilityTransitionRequest(
            action=control["operation_type"], task_id=control["task_id"],
            helper_machines=control.get("helper_machines"),
            after_seconds=control.get("after_seconds"), reason=control.get("reason") or "manual",
            operation_id=control["operation_id"],
        )
        try:
            result = apply_availability_transition(cfg, request)
            reconciled.append(result.to_dict())
        except Exception as exc:
            write_diagnostic_event(
                cfg, "availability_operation_reconcile_failed", task_id=control.get("task_id"),
                details={"operation_id": control.get("operation_id"), "reason": str(exc)},
            )
    return reconciled


def rebuild_deadline_indexes(cfg: RootConfig) -> int:
    rebuilt = 0
    indexed: set[str] = set()
    for task_file in iter_json(shared_paths(cfg.shared_root)["tasks"]):
        task = TaskRecord.from_dict(read_json(task_file))
        before = _deadline_index_path(cfg, task.task_id).exists()
        sync_deadline_index(cfg, task)
        after = _deadline_index_path(cfg, task.task_id).exists()
        if after:
            indexed.add(task.task_id)
        if before != after or after:
            rebuilt += 1
    for index_file in iter_json(shared_paths(cfg.shared_root)["offer_deadlines"]):
        if index_file.stem not in indexed:
            index_file.unlink(missing_ok=True)
            rebuilt += 1
    return rebuilt
