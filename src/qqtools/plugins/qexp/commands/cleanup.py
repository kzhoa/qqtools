"""Cleanup command workflows for qexp."""

from __future__ import annotations

import os
import shutil
from contextlib import ExitStack, nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..config_types import RootConfig
from ..runtime.authority_scan import is_path_present
from ..runtime.availability import remove_deadline_index
from ..runtime.claims import reconcile_claim_archives
from ..runtime.group_discovery.changes import record_task_change
from ..runtime.group_namespace import read_group
from ..runtime.locks import exclusive, group_lock, schema_writer_lock, task_lock
from ..runtime.operation_store import (
    active_operation_path,
    archive_operation,
    iter_active_operation_paths,
    locate_operation_path,
    write_active_operation,
)
from ..runtime.paths import attempt_path, group_path, local_paths, shared_paths, task_path
from ..runtime.progress import cleanup_local_progress, cleanup_shared_progress
from ..runtime.progress_v2 import cleanup_local_progress_v2, cleanup_shared_progress_v2
from ..runtime.ready import assert_ready_writer_compatible, retire_current_ready_generation
from ..runtime.records import SCHEMA_VERSION, AttemptRecord, TaskRecord, new_id, utc_now
from ..runtime.responsibility import responsibility_root
from ..runtime.responsibility_capture import capture_cleanup_guard, cleanup_runtime_guard
from ..runtime.responsibility_cleanup import (
    CLEANUP_FORMAT,
    FLAT_EVIDENCE,
    CleanupRequest,
    complete_cleanup,
    writers_are_quiescent,
)
from ..runtime.responsibility_store import Conflict, Ledger
from ..runtime.store import atomic_replace, iter_json, read_json
from ..runtime.tasks import delete_task, load_task, save_task


def _machine_project_id(cfg: RootConfig, reservation_runtime_root: Path) -> str | None:
    if reservation_runtime_root == cfg.runtime_root:
        return None
    identity_path = shared_paths(cfg.shared_root)["project"] / "identity.json"
    if not identity_path.exists():
        return None
    value = read_json(identity_path).get("project", {}).get("project_id")
    return value if isinstance(value, str) else None


def _reservation_matches_task(reservation: dict[str, Any], task_id: str, machine_project_id: str | None) -> bool:
    return reservation.get("task_id") == task_id and (
        machine_project_id is None or reservation.get("project_id") == machine_project_id
    )


def _clean_blockers(cfg: RootConfig, task: TaskRecord, *, reservation_runtime_root: Path | None = None) -> list[str]:
    reservation_runtime_root = reservation_runtime_root or cfg.runtime_root
    machine_project_id = _machine_project_id(cfg, reservation_runtime_root)
    blockers: list[str] = []
    if task.state["projection"] not in {"succeeded", "failed", "cancelled"}:
        blockers.append(f"task_state:{task.state['projection']}")
    if task.claim_control.get("active_claim"):
        blockers.append("active_claim")
    for operation_path in iter_json(shared_paths(cfg.shared_root)["group_control"]):
        control = read_json(operation_path).get("group_control", {})
        if control.get("group_name") != task.group_name:
            continue
        if control.get("operation_type") == "cancel" and task.group_name:
            try:
                barriers = read_group(cfg.shared_root, task.group_name)["group"].get("cancellation_barriers", [])
            except FileNotFoundError:
                barriers = []
            if not any(item.get("operation_id") == control.get("operation_id") for item in barriers):
                blockers.append(f"group_control_barrier_missing:{control.get('operation_id')}")
                continue
        if control.get("state") == "completed":
            continue
        high_watermark = control.get("membership_high_watermark")
        if high_watermark is not None and (task.group_membership_sequence or 0) <= high_watermark:
            blockers.append(f"group_control:{control.get('operation_id')}")
    for manifest_path in iter_json(cfg.runtime_root / "processes"):
        process = read_json(manifest_path).get("process", {})
        if process.get("task_id") != task.task_id:
            continue
        if process.get("observed_state") not in {"exited", "missing", "quarantined"}:
            blockers.append(f"local_process:{process.get('attempt_id')}")
    for state in ("active", "provisional", "cpu_active", "cpu_provisional"):
        for reservation_path in iter_json(local_paths(reservation_runtime_root)[state]):
            reservation = read_json(reservation_path).get("reservation", {})
            if _reservation_matches_task(reservation, task.task_id, machine_project_id):
                blockers.append(f"local_{state}_reservation:{reservation.get('reservation_id')}")
    return blockers


def _cleanup_required_machines(cfg: RootConfig, task: TaskRecord) -> list[str]:
    machines = {cfg.machine_name, task.placement_policy["home_machine"]}
    attempts_dir = shared_paths(cfg.shared_root)["attempts"] / task.task_id
    for path in iter_json(attempts_dir):
        machines.add(AttemptRecord.from_dict(read_json(path)).machine_name)
    return sorted(machine for machine in machines if machine)


def _dependency_references(cfg: RootConfig, task: TaskRecord) -> list[str]:
    """Return all extant direct dependents that retain this Task's history."""
    if task.group_name is None:
        return []
    references: list[str] = []
    for path in iter_json(shared_paths(cfg.shared_root)["tasks"]):
        candidate = TaskRecord.from_dict(read_json(path))
        if candidate.group_name == task.group_name and task.task_id in candidate.depends_on_task_ids:
            references.append(candidate.task_id)
    return sorted(references)


def _start_cleanup_operation(cfg: RootConfig, task: TaskRecord) -> dict[str, Any]:
    from ..events import write_event

    references = _dependency_references(cfg, task)
    if references:
        raise ValueError(f"Task {task.task_id!r} cannot be cleaned while referenced by: {', '.join(references)}")
    operation_path = locate_operation_path(cfg, "cleanup", task.task_id)
    if operation_path.exists():
        operation = read_json(operation_path)
        cleanup = operation.get("cleanup", {})
        if cleanup.get("state") != "completed":
            if task.group_name:
                from ..runtime.group_discovery.service import publish_group_locator_for_transition

                # QQTOOLS-COMPAT-0017: refresh maintenance discovery before
                # repairing Task cleanup ownership from an existing operation.
                publish_group_locator_for_transition(cfg, task.group_name, "maintenance", "metadata_cleanup")
            task.control["cleanup_operation_id"] = cleanup.get("operation_id")
            task.control["cleanup_state"] = cleanup.get("state")
            task.meta["revision"] += 1
            task.meta["updated_at"] = utc_now()
            save_task(cfg, task)
        return operation
    now = utc_now()
    operation = {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "revision": 1,
            "created_at": now,
            "updated_at": now,
            "updated_by": {"actor_type": "cli", "machine_name": cfg.machine_name, "process_id": str(os.getpid())},
        },
        "cleanup": {
            "operation_id": new_id(),
            "task_id": task.task_id,
            "state": "preparing",
            "group_name": task.group_name,
            "submission_operation_id": task.submission_operation_id,
            "terminal_state": task.state["projection"],
            "created_at": now,
            "required_machines": _cleanup_required_machines(cfg, task),
            "acknowledgements": {},
            "completed_at": None,
        },
    }
    operation_path = active_operation_path(cfg, "cleanup", task.task_id)
    if task.group_name:
        from ..runtime.group_discovery.service import publish_group_locator_for_transition

        # QQTOOLS-COMPAT-0017: the maintenance locator precedes cleanup's first
        # durable operation or Task-side ownership effect.
        publish_group_locator_for_transition(cfg, task.group_name, "maintenance", "metadata_cleanup")
    write_active_operation(cfg, "cleanup", task.task_id, operation)
    task.control["cleanup_operation_id"] = operation["cleanup"]["operation_id"]
    task.control["cleanup_state"] = "preparing"
    task.meta["revision"] += 1
    task.meta["updated_at"] = now
    save_task(cfg, task)
    retire_current_ready_generation(cfg, task)
    write_event(
        cfg,
        "task_cleanup_started",
        task_id=task.task_id,
        details={
            "operation_id": operation["cleanup"]["operation_id"],
            "group_name": task.group_name,
            "submission_operation_id": task.submission_operation_id,
            "terminal_state": task.state["projection"],
        },
    )
    return operation


def _cleanup_known_attempt(cfg: RootConfig, attempt: AttemptRecord) -> tuple[list[str], list[str]]:
    """Retain terminal proof before deletion; caller holds the Task cleanup fence."""
    paths = local_paths(cfg.runtime_root)
    evidence = [paths[name] / f"{attempt.attempt_id}.json" for name in FLAT_EVIDENCE]
    evidence.append(paths["termination_decisions"] / attempt.attempt_id)
    existing = [path for path in evidence if path.exists()]
    root = responsibility_root(cfg.runtime_root)
    entry = None
    if root.exists():
        entry = Ledger(root).find(attempt.attempt_id)
    if not existing and entry is None:
        return [], []
    if attempt.phase not in {"succeeded", "failed", "cancelled"}:
        return [], [f"local_attempt_not_terminal:{attempt.attempt_id}"]
    payload = {"task_id": attempt.task_id, "attempt_number": attempt.attempt_number}
    if entry is not None and entry["payload"] != payload:
        can_resolve = (
            entry["stage"] == "active"
            and set(entry["payload"]) == set(payload)
            and all(entry["payload"][key] in (None, value) for key, value in payload.items())
        )
        if not can_resolve:
            raise Conflict("cleanup membership does not match Attempt truth")
    request = CleanupRequest.from_entry(entry) if entry is not None else None
    if request is None:
        if not writers_are_quiescent(cfg.runtime_root, attempt.task_id, attempt.attempt_id):
            return [], [f"local_writer_unresolved:{attempt.attempt_id}"]
        request = CleanupRequest(
            attempt.attempt_id,
            payload,
            {
                "format": CLEANUP_FORMAT,
                "task_id": attempt.task_id,
                "attempt_id": attempt.attempt_id,
                "basis": "terminal_attempt",
            },
        )
    with exclusive(cfg.runtime_root / "locks" / "responsibility-initialize.lock"):
        ledger = Ledger.open_or_create(root)
    is_complete = complete_cleanup(ledger, cfg.runtime_root, request)
    removed = [str(path) for path in existing if not path.exists()]
    blockers = [] if is_complete else [f"local_cleanup_pending:{attempt.attempt_id}"]
    return removed, blockers


def _cleanup_local_resources(
    cfg: RootConfig, cleanup: dict, *, reservation_runtime_root: Path | None = None
) -> tuple[list[str], list[str]]:
    with capture_cleanup_guard(cfg.runtime_root) as can_cleanup:
        if not can_cleanup:
            return [], ["writer_capture_pending"]
        return _cleanup_local_resources_guarded(cfg, cleanup, reservation_runtime_root=reservation_runtime_root)


def _cleanup_local_resources_guarded(
    cfg: RootConfig, cleanup: dict, *, reservation_runtime_root: Path | None = None
) -> tuple[list[str], list[str]]:
    from ..agent.context import resolve_execution_context
    from ..runtime.resources.reservations import release
    from ..runtime.responsibility_task_cleanup import cleanup_unmatched_task_evidence

    task_id = cleanup["task_id"]
    reservation_runtime_root = reservation_runtime_root or resolve_execution_context(cfg).reservation_root
    machine_project_id = _machine_project_id(cfg, reservation_runtime_root)
    removed: list[str] = []
    blockers: list[str] = []
    attempt_ids: set[str] = set()
    known_attempts: dict[str, AttemptRecord] = {}
    for path in iter_json(shared_paths(cfg.shared_root)["attempts"] / task_id):
        attempt = AttemptRecord.from_dict(read_json(path))
        if attempt.task_id != task_id:
            raise ValueError(f"Attempt {attempt.attempt_id!r} belongs to another Task")
        attempt_ids.add(attempt.attempt_id)
        known_attempts[attempt.attempt_id] = attempt
        if attempt.machine_name == cfg.machine_name:
            deleted, pending = _cleanup_known_attempt(cfg, attempt)
            removed.extend(deleted)
            blockers.extend(pending)
    if blockers:
        return removed, blockers
    deleted, pending, unmatched_ids = cleanup_unmatched_task_evidence(cfg, cleanup, known_attempts)
    removed.extend(deleted)
    blockers.extend(pending)
    attempt_ids.update(unmatched_ids)
    if blockers:
        return removed, blockers
    paths = local_paths(reservation_runtime_root)
    for state in ("active", "provisional", "cpu_active", "cpu_provisional"):
        for reservation_path in list(iter_json(paths[state])):
            reservation = read_json(reservation_path).get("reservation", {})
            if not _reservation_matches_task(reservation, task_id, machine_project_id):
                continue
            release(reservation_runtime_root, reservation["reservation_id"], "task_cleanup")
            removed.append(str(reservation_path))
    for log_path in sorted((cfg.runtime_root / "logs").glob(f"{task_id}-*.log")):
        log_path.unlink(missing_ok=True)
        removed.append(str(log_path))
    removed.extend(cleanup_local_progress_v2(cfg, task_id, attempt_ids))
    removed.extend(cleanup_local_progress(cfg, task_id, attempt_ids))
    return removed, []


def _cleanup_capture_roots(cfg: RootConfig, reservation_runtime_root: Path | None) -> list[Path]:
    """Keep migration source coverage after individual memberships retire."""
    from ..agent.context import resolve_execution_context

    roots = [cfg.runtime_root]
    if reservation_runtime_root == cfg.runtime_root:
        return roots
    context = resolve_execution_context(cfg, reservation_runtime_root)
    if context.binding is None:
        if cfg.runtime_root.is_relative_to(context.machine_runtime.paths["projects"]):
            raise RuntimeError("managed cleanup requires its registered project binding")
        return roots
    # An explicit reservation backend can accompany a pre-migration cfg. The
    # target checkpoint may precede the source hold after interrupted setup.
    roots.append(context.local_root)
    source_paths = context.machine_runtime._legacy_evidence_roots(context.binding)
    if source_paths is not None:
        roots.append(source_paths[0]["root"])
    elif is_path_present(context.machine_runtime.migration_path(context.binding.project_id)):
        raise RuntimeError("managed cleanup migration has no valid legacy source")
    return roots


def _finalize_cleanup_operation(cfg: RootConfig, operation: dict[str, Any]) -> list[str]:
    from ..events import write_event

    cleanup = operation["cleanup"]
    task_id = cleanup["task_id"]
    group_name = cleanup.get("group_name")
    if isinstance(group_name, str) and group_name:
        from ..runtime.group_discovery.service import publish_group_locator_for_transition

        # QQTOOLS-COMPAT-0017: refresh cleanup discovery before retiring the
        # cleanup owner and its Group membership evidence.
        publish_group_locator_for_transition(cfg, group_name, "maintenance", "metadata_cleanup")
    path = task_path(cfg.shared_root, task_id)
    task: TaskRecord | None = None
    if path.exists():
        task = load_task(cfg, task_id)
        references = _dependency_references(cfg, task)
        if references:
            raise RuntimeError(
                f"Task {task_id!r} cleanup retained truth because it is referenced by: {', '.join(references)}"
            )
    if not reconcile_claim_archives(cfg, task_id):
        return []
    removed: list[str] = []
    if path.exists():
        assert_ready_writer_compatible(cfg)
        existing = cleanup.get("group_discovery_outcome")
        task_value = task.to_dict()
        task_record = task_value.get("task", {})
        if not isinstance(task_record, dict):
            raise ValueError("Task serialization is malformed")
        if existing is not None:
            if not isinstance(existing, dict) or not isinstance(existing.get("task"), dict):
                raise Conflict("cleanup Group discovery outcome is malformed")
            existing_task = existing["task"].get("task")
            if not isinstance(existing_task, dict):
                raise Conflict("cleanup Group discovery outcome Task is malformed")
            if existing_task.get("task_id") != task_record.get("task_id") or existing_task.get(
                "submission_operation_id"
            ) != task_record.get("submission_operation_id"):
                raise Conflict("cleanup Group discovery outcome does not match Task truth")
        else:
            attempt = None
            attempt_number = task.attempt_control.get("current_attempt_number")
            if attempt_number is not None:
                try:
                    attempt = read_json(attempt_path(cfg.shared_root, task.task_id, attempt_number))
                except FileNotFoundError:
                    attempt = None
            cleanup["group_discovery_outcome"] = {"task": task_value, "attempt": attempt}
            operation["meta"]["revision"] += 1
            operation["meta"]["updated_at"] = utc_now()
            write_active_operation(cfg, "cleanup", task.task_id, operation)
    change_context = (
        record_task_change(
            cfg,
            task,
            "cleanup",
            details={"cleanup_operation_id": cleanup["operation_id"]},
        )
        if task is not None
        else nullcontext()
    )
    with change_context:
        if path.exists():
            delete_task(cfg, task_id)
            removed.append(str(path))
        deadline_index = shared_paths(cfg.shared_root)["offer_deadlines"] / f"{task_id}.json"
        if deadline_index.exists():
            remove_deadline_index(cfg, task_id)
            removed.append(str(deadline_index))
        attempts_dir = shared_paths(cfg.shared_root)["attempts"] / task_id
        if attempts_dir.exists():
            shutil.rmtree(attempts_dir)
            removed.append(str(attempts_dir))
        logs_dir = shared_paths(cfg.shared_root)["logs"] / task_id
        if logs_dir.exists():
            shutil.rmtree(logs_dir)
            removed.append(str(logs_dir))
        removed.extend(cleanup_shared_progress(cfg, task_id))
        removed.extend(cleanup_shared_progress_v2(cfg, task_id))
        write_event(
            cfg,
            "task_cleaned",
            task_id=task_id,
            details={
                "operation_id": cleanup["operation_id"],
                "group_name": cleanup.get("group_name"),
                "submission_operation_id": cleanup.get("submission_operation_id"),
                "terminal_state": cleanup.get("terminal_state"),
            },
        )
        cleanup.update({"state": "completed", "completed_at": utc_now()})
        operation["meta"]["revision"] += 1
        operation["meta"]["updated_at"] = utc_now()
        archive_operation(cfg, "cleanup", task_id, operation)
    return removed


def _finalize_cleanup_if_ready(cfg: RootConfig, operation_path: Path) -> list[str]:
    operation = read_json(operation_path)
    cleanup = operation.get("cleanup", {})
    task_id = cleanup.get("task_id")
    group_name = cleanup.get("group_name")
    if not task_id:
        return []

    def finalize_under_task_lock() -> list[str]:
        operation = read_json(operation_path)
        cleanup = operation["cleanup"]
        if cleanup.get("state") not in {"preparing", "waiting_ack"}:
            return []
        pending = cleanup.get("pending_machines")
        if pending is None:
            required = set(cleanup.get("required_machines", []))
            pending = sorted(required - set(cleanup.get("acknowledgements", {})))
            cleanup["pending_machines"] = pending
            atomic_replace(operation_path, operation)
        if pending:
            return []
        return _finalize_cleanup_operation(cfg, operation)

    with schema_writer_lock(cfg, blocking=False, require_narrow=True) as has_schema_lock:
        if not has_schema_lock:
            return []
        if group_name:
            with group_lock(cfg.shared_root, group_name, blocking=False) as has_group_lock:
                if not has_group_lock:
                    return []
                with task_lock(cfg.shared_root, task_id, blocking=False) as has_task_lock:
                    if not has_task_lock:
                        return []
                    return finalize_under_task_lock()
        with task_lock(cfg.shared_root, task_id, blocking=False) as has_task_lock:
            if not has_task_lock:
                return []
            return finalize_under_task_lock()


def reconcile_cleanup_operations(
    cfg: RootConfig,
    *,
    reservation_runtime_root: Path | None = None,
    include_legacy: bool = True,
) -> list[dict[str, Any]]:
    """Clean machine-local resources and finalize fully acknowledged cleanup operations."""
    # Active-operation enumeration writes a local cursor before yielding. Protect
    # that write too, and validate the binding before a stale caller recreates it.
    with cleanup_runtime_guard(cfg.runtime_root):
        capture_roots = _cleanup_capture_roots(cfg, reservation_runtime_root)
        return _reconcile_cleanup_operations(
            cfg, capture_roots, reservation_runtime_root=reservation_runtime_root, include_legacy=include_legacy
        )


def _reconcile_cleanup_operations(
    cfg: RootConfig,
    capture_roots: list[Path],
    *,
    reservation_runtime_root: Path | None,
    include_legacy: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for operation_path in iter_active_operation_paths(cfg, "cleanup", include_legacy=include_legacy):
        operation = read_json(operation_path)
        cleanup = operation.get("cleanup", {})
        if cleanup.get("state") not in {"preparing", "waiting_ack"}:
            continue
        task_id = cleanup.get("task_id")
        if not task_id:
            continue
        result = {
            "operation_id": cleanup.get("operation_id"),
            "task_id": task_id,
            "state": cleanup.get("state", "waiting_ack"),
            "pending_machines": cleanup.get("pending_machines", []),
            "removed": [],
            "blockers": [],
        }
        with capture_cleanup_guard(*capture_roots) as can_cleanup:
            with schema_writer_lock(cfg, blocking=False, require_narrow=True) as has_schema_lock:
                if not has_schema_lock:
                    result["blockers"] = ["schema_lock_busy"]
                    results.append(result)
                    continue
                task_file = task_path(cfg.shared_root, task_id)
                task = load_task(cfg, task_id) if task_file.exists() else None
                with ExitStack() as stack:
                    if task and task.group_name:
                        has_group_lock = stack.enter_context(
                            group_lock(cfg.shared_root, task.group_name, blocking=False)
                        )
                        if not has_group_lock:
                            result["blockers"] = ["group_lock_busy"]
                            results.append(result)
                            continue
                    has_task_lock = stack.enter_context(task_lock(cfg.shared_root, task_id, blocking=False))
                    if not has_task_lock:
                        result["blockers"] = ["task_lock_busy"]
                        results.append(result)
                        continue
                    operation = read_json(operation_path)
                    cleanup = operation["cleanup"]
                    if cleanup.get("state") not in {"preparing", "waiting_ack"}:
                        continue
                    required = set(cleanup.get("required_machines", []))
                    acknowledgements = cleanup.setdefault("acknowledgements", {})
                    removed: list[str] = []
                    blockers: list[str] = [] if can_cleanup else ["writer_capture_pending"]
                    if can_cleanup and cfg.machine_name in required and cfg.machine_name not in acknowledgements:
                        removed, blockers = _cleanup_local_resources(
                            cfg, cleanup, reservation_runtime_root=reservation_runtime_root
                        )
                        if not blockers:
                            acknowledgements[cfg.machine_name] = {"acknowledged_at": utc_now(), "removed": removed}
                    cleanup["state"] = "waiting_ack"
                    cleanup["pending_machines"] = sorted(required - set(acknowledgements))
                    if task_file.exists():
                        task = load_task(cfg, task_id)
                        if task.control.get("cleanup_operation_id") == cleanup.get("operation_id"):
                            task.control["cleanup_state"] = cleanup["state"]
                            task.meta["revision"] += 1
                            task.meta["updated_at"] = utc_now()
                            save_task(cfg, task)
                    operation["meta"]["revision"] += 1
                    operation["meta"]["updated_at"] = utc_now()
                    atomic_replace(operation_path, operation)
                    result = {
                        "operation_id": cleanup["operation_id"],
                        "task_id": task_id,
                        "state": cleanup["state"],
                        "pending_machines": cleanup.get("pending_machines", []),
                        "removed": removed,
                        "blockers": blockers,
                    }
            if can_cleanup and not result["pending_machines"]:
                result["removed"].extend(_finalize_cleanup_if_ready(cfg, operation_path))
                finalized_path = locate_operation_path(cfg, "cleanup", task_id)
                finalized = read_json(finalized_path).get("cleanup", {})
                result["state"] = finalized.get("state", result["state"])
                result["pending_machines"] = finalized.get("pending_machines", [])
            results.append(result)
    return results


def clean(
    cfg: RootConfig,
    *,
    task_id: str | None = None,
    group: str | None = None,
    older_than_days: int = 30,
    limit: int = 100,
    dry_run: bool = False,
    max_work_items: int = 64,
    reservation_runtime_root: Path | None = None,
) -> dict[str, Any]:
    """Remove terminal Task truth exactly or under a bounded retention policy."""
    if reservation_runtime_root is None:
        from ..agent.context import resolve_execution_context

        context = resolve_execution_context(cfg)
        cfg = context.local_cfg
        reservation_runtime_root = context.reservation_root
    if task_id and group:
        raise ValueError("task_id and group cannot be used together.")
    if older_than_days < 0:
        raise ValueError("older_than_days must be non-negative.")
    if limit <= 0:
        raise ValueError("limit must be positive.")
    if type(max_work_items) is not int or not 1 <= max_work_items <= 64:
        raise ValueError("max_work_items must be between 1 and 64.")
    if task_id:
        candidates = [load_task(cfg, task_id)]
    else:
        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        candidates = []
        for path in iter_json(shared_paths(cfg.shared_root)["tasks"]):
            task = TaskRecord.from_dict(read_json(path))
            if group and task.group_name != group:
                continue
            updated_at = datetime.fromisoformat(task.meta["updated_at"].replace("Z", "+00:00"))
            if task.state["projection"] in {"succeeded", "failed", "cancelled"} and updated_at <= cutoff:
                candidates.append(task)
        candidates.sort(key=lambda item: (item.meta["updated_at"], item.task_id))
        candidates = candidates[:limit]
    result: dict[str, Any] = {
        "dry_run": dry_run,
        "deletion": "none" if dry_run else "requested",
        "deletion_performed": False,
        "candidates": [task.task_id for task in candidates],
        "removed": [],
        "skipped": {},
        "removed_task_ids": [],
        "pending_task_ids": [],
        "skipped_task_ids": [],
        "blocked_task_ids": [],
    }
    from ..runtime.ready.group_members_rebuild import cleanup_group_ready_member_archives

    if not dry_run:
        result["group_ready_member_archive_cleanup"] = cleanup_group_ready_member_archives(
            cfg, max_work_items=max_work_items
        )
    from ..scheduler import authority_locks

    for candidate in candidates:
        with schema_writer_lock(cfg, require_narrow=True):
            with authority_locks(cfg, candidate):
                task = load_task(cfg, candidate.task_id)
                blockers = _clean_blockers(cfg, task, reservation_runtime_root=reservation_runtime_root)
                unsafe_blockers = [
                    item
                    for item in blockers
                    if not item.startswith(
                        (
                            "local_active_reservation:",
                            "local_provisional_reservation:",
                            "local_cpu_active_reservation:",
                            "local_cpu_provisional_reservation:",
                        )
                    )
                ]
                if unsafe_blockers:
                    result["skipped"][task.task_id] = unsafe_blockers
                    result["skipped_task_ids"].append(task.task_id)
                    result["blocked_task_ids"].append(task.task_id)
                    continue
                if not dry_run:
                    operation = _start_cleanup_operation(cfg, task)
                    result.setdefault("operations", {})[task.task_id] = operation["cleanup"]
    if not dry_run:
        for reconciliation in reconcile_cleanup_operations(cfg, reservation_runtime_root=reservation_runtime_root):
            if reconciliation["task_id"] in result["candidates"]:
                result["removed"].extend(reconciliation["removed"])
                result.setdefault("operations", {})[reconciliation["task_id"]] = reconciliation
    operations = result.get("operations", {})
    if isinstance(operations, dict):
        for candidate_id, operation in operations.items():
            if not isinstance(operation, dict):
                continue
            state = operation.get("state")
            if state == "completed":
                result["removed_task_ids"].append(str(candidate_id))
            elif state == "blocked":
                result["blocked_task_ids"].append(str(candidate_id))
            else:
                result["pending_task_ids"].append(str(candidate_id))
    result["removed_task_ids"] = sorted(set(result["removed_task_ids"]))
    result["pending_task_ids"] = sorted(
        set(result["pending_task_ids"]) - set(result["removed_task_ids"]) - set(result["blocked_task_ids"])
    )
    result["skipped_task_ids"] = sorted(set(result["skipped_task_ids"]))
    result["blocked_task_ids"] = sorted(set(result["blocked_task_ids"]))
    result["candidate_count"] = len(result["candidates"])
    result["removed_count"] = len(result["removed_task_ids"])
    result["pending_count"] = len(result["pending_task_ids"])
    result["skipped_count"] = len(result["skipped_task_ids"])
    result["blocked_count"] = len(result["blocked_task_ids"])
    result["deletion_performed"] = bool(result["removed"] or result["removed_task_ids"])
    if dry_run:
        result["outcome"] = "preview"
    elif result["blocked_task_ids"]:
        result["outcome"] = "blocked"
    elif result["pending_task_ids"]:
        result["outcome"] = "waiting"
    elif result["removed_task_ids"]:
        result["outcome"] = "completed"
    else:
        result["outcome"] = "no_change"
    if task_id and result["skipped"] and not dry_run:
        blockers = result["skipped"][task_id]
        raise ValueError(f"Task {task_id!r} cannot be cleaned: {', '.join(blockers)}")
    return result
