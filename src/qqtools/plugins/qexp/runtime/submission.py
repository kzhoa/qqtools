"""Common single and bulk submission transaction with resumable operations."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from ..layout import is_cpu_lane_root, is_group_ready_members_root, is_submission_group_publication_root
from ..task_observation import validate_tmux_override
from .availability import remove_deadline_index, sync_deadline_index
from .dependencies import validate_group_dependencies
from .group_namespace import provisional_group_reader
from .locks import group_lock, group_writer_lock, idempotency_lock, schema_writer_lock, task_lock, task_locks
from .operation_store import operation_exists
from .paths import group_path, idempotency_path, submission_path, task_path
from .ready import (
    assert_ready_writer_compatible,
    commit_ready_publication,
    delete_ready_marker,
    discard_ready_generation,
    prepare_ready_transition,
    primary_projection_routes_for_group,
    primary_route_update_transaction,
    retire_previous_ready_generation,
    sync_primary_ready_group,
)
from .ready.group_members import assert_group_ready_members_writable
from .records import (
    TaskRecord,
    new_group,
    new_id,
    new_worker_member,
    normalize_group_record,
    utc_now,
    validate_identifier,
)
from .store import atomic_replace, create_if_absent, read_json
from .submission_control import publish_submission as _publish_submission
from .submission_plan import (
    SubmissionPlan,
    _planned_worker_set,
    _resolve_submission_plan,
    _task_spec,
    _thaw,
    _validate_group_precondition,
    _validate_placement_against_workers,
    decode_submission_plan,
    encode_submission_plan,
    legacy_submission_request_digest,
    normalize_submission_request,
    prepare_submission_plan,
    semantic_digest,
)
from .tasks import save_task


def _write_group_record(cfg: object, path: Path, data: dict[str, Any]) -> None:
    """Persist a Group through this module's patchable atomic writer."""
    atomic_replace(path, data)


def _submission_atomic_writer(path: Path, value: dict[str, Any], **kwargs: Any) -> Any:
    """Keep Submission publication writes observable through this module's writer."""
    return atomic_replace(path, value, **kwargs)


def publish_submission(cfg: object, operation: dict[str, Any]) -> None:
    """Publish Submission truth through the control proof protocol."""
    _publish_submission(cfg, operation, _atomic_writer=_submission_atomic_writer)


@contextmanager
def _submission_protocol_lock(cfg: object, mapping_digest: str) -> Iterator[None]:
    """Fence submissions through the active member-projection protocol."""
    with schema_writer_lock(cfg, require_narrow=True):
        with idempotency_lock(cfg.shared_root, mapping_digest):
            yield


class IdempotencyConflict(ValueError):
    def __init__(self, message: str, *, operation_id: str | None = None, idempotency_key: str | None = None):
        super().__init__(message)
        self.operation_id = operation_id
        self.idempotency_key = idempotency_key


class SubmissionPending(RuntimeError):
    """Submission truth is durable but not yet proven committed."""

    def __init__(self, message: str, *, operation_id: str | None = None, idempotency_key: str | None = None):
        super().__init__(message)
        self.operation_id = operation_id
        self.idempotency_key = idempotency_key


class SubmissionUnknown(RuntimeError):
    """A commit write outcome cannot be proven from bounded durable truth."""

    def __init__(self, message: str, *, operation_id: str | None = None, idempotency_key: str | None = None):
        super().__init__(message)
        self.operation_id = operation_id
        self.idempotency_key = idempotency_key


class SubmissionResult(list[TaskRecord]):
    def __init__(
        self,
        tasks: Iterable[TaskRecord],
        *,
        operation_id: str,
        idempotency_key: str,
        target_group: str | None,
        state: str,
        group_disposition: str | None = None,
        mode: str | None = None,
        project: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(tasks)
        self.operation_id = operation_id
        self.idempotency_key = idempotency_key
        self.target_group = target_group
        self.state = state
        self.group_disposition = group_disposition
        self.mode = mode
        self.project = project

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "idempotency_key": self.idempotency_key,
            "target_group": self.target_group,
            "task_ids": [task.task_id for task in self],
            "state": self.state,
            "group_disposition": self.group_disposition,
            "mode": self.mode,
            "project": self.project,
        }


def _submission_result(tasks: Iterable[TaskRecord], submission: dict[str, Any]) -> SubmissionResult:
    context = submission.get("resolved_context") or {}
    return SubmissionResult(
        tasks,
        operation_id=submission["operation_id"],
        idempotency_key=submission["idempotency_key"],
        target_group=submission["target_group"],
        state=submission["state"],
        group_disposition=("created" if context.get("create_group") else "reused")
        if submission.get("target_group")
        else "none",
    )


@dataclass(frozen=True, slots=True)
class SubmissionPreview:
    """Read-only normalized submission result used by both input modes."""

    tasks: tuple[dict[str, Any], ...]
    group_action: str
    worker_additions: dict[str, dict[str, Any]]
    evidence_gaps: tuple[str, ...] = ()
    operation_id: str | None = None
    operation_state: str | None = None
    idempotency_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tasks": [dict(item) for item in self.tasks],
            "group_action": self.group_action,
            "worker_additions": {machine: dict(value) for machine, value in self.worker_additions.items()},
            "evidence_gaps": list(self.evidence_gaps),
            "operation_id": self.operation_id,
            "operation_state": self.operation_state,
            "idempotency_key": self.idempotency_key,
        }


def _reject_cleanup_tombstones(cfg: Any, resolved: list[dict[str, Any]]) -> None:
    for item in resolved:
        if operation_exists(cfg, "cleanup", item["task_id"]):
            raise ValueError(f"Task {item['task_id']!r} was cleaned and its id cannot be reused.")


def preview_specs(
    cfg: Any,
    specs: list[dict[str, Any]],
    *,
    group_name: str | None = None,
    idempotency_key: str | None = None,
    kind: str = "single",
    worker_set: list[str] | dict[str, Any] | None = None,
) -> SubmissionPreview:
    """Resolve submission semantics without reserving or writing any durable state."""
    request = normalize_submission_request(
        specs,
        group_name=group_name,
        kind=kind,
        worker_set=worker_set,
    )
    operation_id: str | None = None
    operation_state: str | None = None
    frozen_plan: SubmissionPlan | None = None
    if idempotency_key is not None:
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ValueError("idempotency_key must be a non-empty string or null.")
        mapping_path = idempotency_path(
            cfg.shared_root,
            semantic_digest({"project": str(cfg.shared_root), "key": idempotency_key}),
        )
        if mapping_path.exists():
            mapping = read_json(mapping_path)
            operation_id = mapping.get("operation_id") if isinstance(mapping, dict) else None
            try:
                validate_identifier(operation_id, "idempotency mapping operation_id")
                operation = read_json(submission_path(cfg.shared_root, operation_id))
                submission = operation["submission"]
                existing_digest = submission["raw_request_digest"]
                if existing_digest != request.raw_request_digest:
                    legacy = legacy_submission_request_digest(request)
                    if existing_digest != legacy:
                        raise IdempotencyConflict(
                            "idempotency key was already used with different semantic input.",
                            operation_id=operation_id,
                            idempotency_key=idempotency_key,
                        )
                frozen_plan = decode_submission_plan(operation)
                operation_state = submission.get("state")
            except IdempotencyConflict:
                raise
            except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
                raise RuntimeError("idempotency mapping does not identify a readable submission operation.") from exc

    if frozen_plan is None:
        plan = _resolve_submission_plan(
            cfg,
            request,
            idempotency_key=idempotency_key or "preview",
            operation_id="preview",
            allocate_task_ids=False,
            persist_clock_evidence=False,
            acquire_group_lock=False,
        )
    else:
        plan = frozen_plan

    tasks: list[dict[str, Any]] = []
    for index, frozen in enumerate(plan.task_specs):
        item = _thaw(frozen)
        if frozen_plan is None and request.specs[index].get("task_id") is None:
            item["task_id"] = None
        item["input_index"] = index
        item["tmux_override"] = plan.tmux_overrides[index]
        tasks.append(item)
    if plan.target_group is None:
        group_action = "none"
    elif plan.create_group:
        group_action = "create"
    else:
        group_action = "reuse"
    return SubmissionPreview(
        tasks=tuple(tasks),
        group_action=group_action,
        worker_additions=_thaw(plan.worker_set_additions),
        operation_id=operation_id,
        operation_state=operation_state,
        idempotency_key=idempotency_key,
    )


def _task_matches_resolved(
    current: TaskRecord, item: dict[str, Any], operation_id: str, group_name: str | None
) -> bool:
    spec = _task_spec(item, is_canonical=current.spec.lane is not None)
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
        and current.depends_on_task_ids == item["depends_on_task_ids"]
        and current.spec.to_dict() == spec.to_dict()
        and current.placement_policy == expected_policy
    )


def _new_task_from_resolved(
    item: dict[str, Any],
    *,
    group_name: str | None,
    operation_id: str,
    is_canonical: bool,
) -> TaskRecord:
    return TaskRecord.new(
        task_id=item["task_id"],
        machine=item["home_machine"],
        spec=_task_spec(item, is_canonical=is_canonical),
        group_name=group_name,
        name=item["name"],
        sharing_mode=item["sharing_mode"],
        fallback_machines=item["fallback_machines"],
        offer_after_seconds=item["offer_after_seconds"],
        offer_eligible_at=item.get("offer_eligible_at"),
        offer_clock_evidence=item.get("offer_clock_evidence"),
        operation_id=operation_id,
        depends_on_task_ids=item["depends_on_task_ids"],
    )


def _remove_operation_added_workers(group: dict[str, Any], operation_id: str, plan: SubmissionPlan) -> bool:
    additions = plan.worker_set_additions
    removed_worker = False
    for machine in dict.fromkeys(additions):
        worker = group["group"]["worker_set"].get(machine)
        if worker and worker.get("added_by_operation") == operation_id:
            del group["group"]["worker_set"][machine]
            removed_worker = True
    if removed_worker:
        group["group"]["worker_set_epoch"] += 1
    return removed_worker


def _finalize_submission_group_locked(cfg: Any, submission: dict[str, Any]) -> None:
    """Finalize a committed submission while the Group writer lock is held."""
    group_name = submission.get("target_group")
    if not group_name:
        return
    if is_group_ready_members_root(cfg):
        assert_group_ready_members_writable(cfg)
    group_file = group_path(cfg.shared_root, group_name)
    if not group_file.exists():
        raise RuntimeError(f"committed submission {submission['operation_id']!r} has a missing Group {group_name!r}.")
    group = read_json(group_file)
    normalize_group_record(group)
    pending = group["group"].get("pending_submission_commit") or {}
    if not pending:
        return
    if pending.get("operation_id") != submission["operation_id"]:
        raise RuntimeError(f"Group {group_name!r} has pending submission commit {pending.get('operation_id')!r}.")
    sequences = pending.get("membership_sequences")
    if sequences is None:
        sequences = submission["commit_plan"].get("group_membership_sequences")
    if not isinstance(sequences, list):
        raise RuntimeError(f"submission {submission['operation_id']!r} has no membership sequence plan.")
    group["group"]["next_membership_sequence"] = max(
        group["group"]["next_membership_sequence"], max(sequences, default=0) + 1
    )
    from .group_discovery.service import publish_submission_debt

    publish_submission_debt(cfg.shared_root, group_name, submission["operation_id"])
    group["meta"]["revision"] += 1
    group["meta"]["updated_at"] = utc_now()
    group["group"]["pending_submission_commit"] = None
    _write_group_record(cfg, group_file, group)


def finalize_submission_group(cfg: Any, submission: dict[str, Any]) -> None:
    """Finalize a committed submission's pending Group membership transaction."""
    group_name = submission.get("target_group")
    if not group_name:
        return
    with group_writer_lock(cfg, group_name):
        _finalize_submission_group_locked(cfg, submission)


def _plan_specs(plan: SubmissionPlan) -> list[dict[str, Any]]:
    """Return independent mutable task specifications from an immutable plan."""
    return [_thaw(item) for item in plan.task_specs]


def _validate_operation_plan(operation: dict[str, Any], plan: SubmissionPlan) -> None:
    try:
        persisted_plan = decode_submission_plan(operation)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise RuntimeError("submission operation has invalid immutable plan truth.") from exc
    if persisted_plan != plan:
        raise RuntimeError("submission operation immutable context does not match its execution plan.")


def _load_plan_tasks(cfg: Any, plan: SubmissionPlan) -> list[TaskRecord]:
    try:
        return [TaskRecord.from_dict(read_json(task_path(cfg.shared_root, task_id))) for task_id in plan.task_ids]
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("committed submission has missing Task truth; run qexp admin repair.") from exc


def _execute_submission_locked(
    cfg: Any,
    operation: dict[str, Any],
    plan: SubmissionPlan,
    *,
    on_prepared: Callable[[str, str], None] | None = None,
) -> SubmissionResult:
    """Execute a prepared operation while the schema and idempotency fences are held."""
    _validate_operation_plan(operation, plan)
    submission = operation["submission"]
    state = submission["state"]
    if state == "committed":
        tasks = _load_plan_tasks(cfg, plan)
        finalize_submission_group(cfg, submission)
        return _submission_result(tasks, submission)
    if state == "aborted":
        raise RuntimeError(f"submission operation was aborted: {submission['failure_reason']}")
    if state == "blocked":
        raise RuntimeError(f"submission operation is blocked: {submission['failure_reason']}")

    resolved = _plan_specs(plan)
    operation_id = plan.operation_id
    group_name = plan.target_group
    is_canonical = is_cpu_lane_root(cfg)
    prepared_candidates = [
        _new_task_from_resolved(
            item,
            group_name=group_name,
            operation_id=operation_id,
            is_canonical=is_canonical,
        )
        for item in resolved
    ]
    try:
        if group_name:
            with group_lock(cfg.shared_root, group_name):
                validate_group_dependencies(cfg, group_name, prepared_candidates)
        else:
            validate_group_dependencies(cfg, None, prepared_candidates)
    except Exception as exc:
        submission["state"] = "aborted"
        submission["failure_reason"] = str(exc)
        publish_submission(cfg, operation)
        raise

    # The operation, mapping, and whole-input preflight are durable/complete.
    # Disclose recovery identifiers before publishing any provisional Group,
    # Worker, ready-index, or Task truth.
    if on_prepared:
        try:
            on_prepared(plan.operation_id, submission["idempotency_key"])
        except Exception as exc:
            submission["state"] = "aborted"
            submission["failure_reason"] = str(exc)
            publish_submission(cfg, operation)
            raise

    if group_name:
        with group_lock(cfg.shared_root, group_name):
            if plan.create_group or plan.worker_set_additions:
                is_submission_group_publication_root(cfg, is_required=True)
            group_file = group_path(cfg.shared_root, group_name)
            group_was_missing = not group_file.exists()
            if group_was_missing:
                if not plan.create_group:
                    raise RuntimeError(f"Group {group_name!r} disappeared during submission.")
                group = new_group(
                    group_name,
                    plan.original_submitting_machine,
                    creation_operation_id=operation_id,
                )
            else:
                group = read_json(group_file)
                normalize_group_record(group)
            precondition = plan.group_precondition
            pending = group["group"].get("pending_submission_commit") or {}
            has_own_pending_commit = pending.get("operation_id") == operation_id
            has_own_creation = group["group"].get("creation_operation_id") == operation_id
            if pending and not has_own_pending_commit:
                raise RuntimeError(f"Group {group_name!r} has pending submission commit {pending['operation_id']!r}.")
            if precondition.get("exists") and not has_own_pending_commit:
                _validate_group_precondition(group, precondition, group_name)
            elif (
                not precondition.get("exists")
                and not group_was_missing
                and not has_own_pending_commit
                and not has_own_creation
            ):
                operation["submission"]["state"] = "aborted"
                operation["submission"]["failure_reason"] = f"Group {group_name!r} changed during submission."
                publish_submission(cfg, operation)
                raise RuntimeError(f"Group {group_name!r} changed during submission.")
            if group["group"]["admission_state"] != "open":
                operation["submission"]["state"] = "aborted"
                operation["submission"]["failure_reason"] = f"Group {group_name!r} is sealed."
                publish_submission(cfg, operation)
                raise ValueError(f"Group {group_name!r} is sealed.")
            additions = _thaw(plan.worker_set_additions)
            planned_workers = _planned_worker_set(group, additions)
            _validate_placement_against_workers(resolved, group_name=group_name, planned_workers=planned_workers)
            if operation["submission"]["commit_plan"]["group_membership_sequences"] is None:
                start = group["group"]["next_membership_sequence"]
                sequences = list(range(start, start + len(resolved)))
                operation["submission"]["commit_plan"]["group_membership_sequences"] = sequences
                operation["submission"]["state"] = "committing"
                publish_submission(cfg, operation)
            else:
                sequences = operation["submission"]["commit_plan"]["group_membership_sequences"]
            group["group"]["pending_submission_commit"] = {
                "operation_id": operation_id,
                "membership_sequences": sequences,
                "worker_set_additions": additions,
            }
            workers = group["group"]["worker_set"]
            previous_workers = {worker_name: dict(worker) for worker_name, worker in workers.items()}
            projection_routes = primary_projection_routes_for_group(cfg, group_name)
            added_workers: list[dict[str, Any]] = []
            added_worker_machines: list[str] = []
            for machine in dict.fromkeys(additions):
                declaration = additions[machine]
                if machine not in workers:
                    worker = new_worker_member(
                        scheduling_role=declaration["scheduling_role"],
                        gpu_limit_gpus=declaration["gpu_limit_gpus"],
                        added_by_operation=operation_id,
                    )
                    workers[machine] = worker
                    added_workers.append(worker)
                    added_worker_machines.append(machine)
            if added_workers:
                group["group"]["worker_set_epoch"] += 1
                for worker in added_workers:
                    worker["state_epoch"] = group["group"]["worker_set_epoch"]
            group["meta"]["revision"] += 1
            group["meta"]["updated_at"] = utc_now()
            if added_worker_machines:
                routes = projection_routes + [
                    (scope, machine) for machine in added_worker_machines for scope in ("shared", "home")
                ]
                with primary_route_update_transaction(cfg, routes):
                    with provisional_group_reader():
                        _write_group_record(cfg, group_file, group)
                        sync_primary_ready_group(cfg, group_name, previous_workers=previous_workers)
            else:
                _write_group_record(cfg, group_file, group)
    else:
        sequences = [None] * len(resolved)

    staged: list[TaskRecord] = []
    commit_durable = False

    def stage_and_commit() -> None:
        """Validate and publish one submission while its Group cannot change."""
        nonlocal commit_durable
        candidates = [
            _new_task_from_resolved(
                item,
                group_name=group_name,
                operation_id=operation_id,
                is_canonical=is_canonical,
            )
            for item in resolved
        ]
        validate_group_dependencies(cfg, group_name, candidates)
        for sequence, item in zip(sequences, resolved):
            path = task_path(cfg.shared_root, item["task_id"])
            _reject_cleanup_tombstones(cfg, [item])
            if path.exists():
                current = TaskRecord.from_dict(read_json(path))
                if not _task_matches_resolved(current, item, operation_id, group_name):
                    raise ValueError(f"Task {item['task_id']!r} already exists with different truth.")
                if current.ready_generation == 0:
                    old_generation, _ = prepare_ready_transition(cfg, current, "submission_resume")
                    current.meta["revision"] += 1
                    current.meta["updated_at"] = utc_now()
                    save_task(cfg, current)
                    commit_ready_publication(cfg, current)
                    retire_previous_ready_generation(cfg, old_generation, current)
                sync_deadline_index(cfg, current)
                staged.append(current)
                continue
            task = _new_task_from_resolved(
                item,
                group_name=group_name,
                operation_id=operation_id,
                is_canonical=is_canonical,
            )
            task.group_membership_sequence = sequence
            # A process can die after reserving the first generation but before
            # authoritative Task truth is written. The frozen operation owns
            # this previously absent Task ID, so discard only that exact stale
            # generation before idempotently staging it again.
            discard_ready_generation(cfg, task.task_id, 1)
            old_generation, _ = prepare_ready_transition(
                cfg,
                task,
                "submission_stage",
                target_revision=task.meta["revision"],
            )
            save_task(cfg, task)
            commit_ready_publication(cfg, task)
            retire_previous_ready_generation(cfg, old_generation, task)
            sync_deadline_index(cfg, task)
            staged.append(task)
        operation["submission"]["state"] = "committed"
        operation["submission"]["committed_at"] = utc_now()
        try:
            publish_submission(cfg, operation)
        except Exception:
            try:
                persisted = read_json(submission_path(cfg.shared_root, operation_id))
            except (OSError, ValueError):
                raise
            if persisted.get("submission", {}).get("state") != "committed":
                raise
        commit_durable = True

    try:
        # Validation and publication must be one Group-linearized operation. Otherwise a
        # prerequisite can begin cleanup between the final check and Task creation.
        if group_name:
            with group_lock(cfg.shared_root, group_name):
                with task_locks(cfg.shared_root, list(plan.task_ids)):
                    with provisional_group_reader():
                        stage_and_commit()
        else:
            with task_locks(cfg.shared_root, list(plan.task_ids)):
                stage_and_commit()
        finalize_submission_group(cfg, operation["submission"])
        return _submission_result(staged, operation["submission"])
    except Exception as exc:
        if commit_durable or operation["submission"].get("state") == "blocked":
            raise
        operation["submission"]["state"] = "aborted"
        operation["submission"]["failure_reason"] = str(exc)
        publish_submission(cfg, operation)
        if group_name:
            with group_lock(cfg.shared_root, group_name):
                group_file = group_path(cfg.shared_root, group_name)
                if group_file.exists():
                    group = read_json(group_file)
                    normalize_group_record(group)
                    has_pending_commit = (group["group"].get("pending_submission_commit") or {}).get(
                        "operation_id"
                    ) == operation_id
                    creation_owner = group["group"].get("creation_operation_id")
                    if has_pending_commit and plan.create_group and creation_owner == operation_id:
                        group_file.unlink()
                    else:
                        removed_worker = _remove_operation_added_workers(group, operation_id, plan)
                        if has_pending_commit or removed_worker:
                            group["group"]["pending_submission_commit"] = None
                            group["meta"]["revision"] += 1
                            group["meta"]["updated_at"] = utc_now()
                            _write_group_record(cfg, group_file, group)
        for item in resolved:
            task_id = item["task_id"]
            path = task_path(cfg.shared_root, task_id)
            with task_lock(cfg.shared_root, task_id):
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


def _staged_tasks_are_exact(cfg: Any, plan: SubmissionPlan) -> bool:
    """Return whether every Task in a stored plan is exact operation-owned truth."""
    try:
        for item in _plan_specs(plan):
            current = TaskRecord.from_dict(read_json(task_path(cfg.shared_root, item["task_id"])))
            if not _task_matches_resolved(current, item, plan.operation_id, plan.target_group):
                return False
    except (FileNotFoundError, OSError, KeyError, TypeError, ValueError):
        return False
    return True


def _abort_submission_owned(cfg: Any, operation: dict[str, Any], plan: SubmissionPlan, reason: str) -> None:
    """Abort and remove only staged truth owned by one Submission operation."""
    submission = operation["submission"]
    submission["state"] = "aborted"
    submission["failure_reason"] = reason
    publish_submission(cfg, operation)
    group_name = plan.target_group
    if group_name:
        with group_lock(cfg.shared_root, group_name):
            group_file = group_path(cfg.shared_root, group_name)
            if group_file.exists():
                group = read_json(group_file)
                normalize_group_record(group)
                pending = group["group"].get("pending_submission_commit") or {}
                own_pending = pending.get("operation_id") == plan.operation_id
                own_creation = group["group"].get("creation_operation_id") == plan.operation_id
                if own_creation and (not pending or own_pending):
                    group_file.unlink()
                else:
                    removed = _remove_operation_added_workers(group, plan.operation_id, plan)
                    if own_pending or removed:
                        group["group"]["pending_submission_commit"] = None
                        group["meta"]["revision"] += 1
                        group["meta"]["updated_at"] = utc_now()
                        _write_group_record(cfg, group_file, group)
    for item in _plan_specs(plan):
        path = task_path(cfg.shared_root, item["task_id"])
        with task_lock(cfg.shared_root, item["task_id"]):
            try:
                current = TaskRecord.from_dict(read_json(path))
            except FileNotFoundError:
                continue
            if current.submission_operation_id != plan.operation_id:
                continue
            remove_deadline_index(cfg, item["task_id"])
            try:
                delete_ready_marker(cfg, item["task_id"], current.ready_generation)
            except (OSError, KeyError, TypeError, ValueError):
                pass
            path.unlink(missing_ok=True)


def reconcile_submission(
    cfg: Any,
    operation_id: str,
    *,
    abort_incomplete: bool = False,
) -> str:
    """Reconcile one Submission operation and its operation-owned Group truth.

    Same-key retry leaves an incomplete operation resumable.  Group control and
    doctor pass ``abort_incomplete=True`` so an occupying provisional Group is
    safely removed instead of being treated as absent.
    """
    validate_identifier(operation_id, "operation_id")
    operation_file = submission_path(cfg.shared_root, operation_id)
    if not operation_file.exists():
        raise RuntimeError(f"Submission operation {operation_id!r} is unavailable.")
    initial = read_json(operation_file)
    initial_submission = initial.get("submission") if isinstance(initial, dict) else None
    if not isinstance(initial_submission, dict):
        raise RuntimeError(f"Submission operation {operation_id!r} is malformed.")
    key = initial_submission.get("idempotency_key")
    if not isinstance(key, str) or not key:
        raise RuntimeError(f"Submission operation {operation_id!r} has no idempotency key.")
    mapping_digest = semantic_digest({"project": str(cfg.shared_root), "key": key})
    with _submission_protocol_lock(cfg, mapping_digest):
        operation = read_json(operation_file)
        submission = operation["submission"]
        plan = decode_submission_plan(operation)
        if submission["state"] == "committed":
            finalize_submission_group(cfg, submission)
            return "committed"
        if submission["state"] == "aborted":
            if abort_incomplete:
                _abort_submission_owned(cfg, operation, plan, submission.get("failure_reason") or "aborted")
            return "aborted"
        if submission["state"] == "blocked":
            return submission["state"]
        group_name = plan.target_group
        if group_name:
            with group_lock(cfg.shared_root, group_name):
                group_file = group_path(cfg.shared_root, group_name)
                if group_file.exists():
                    group = read_json(group_file)
                    normalize_group_record(group)
                    owner = group["group"].get("creation_operation_id")
                    pending = group["group"].get("pending_submission_commit") or {}
                    if owner is not None and owner != operation_id and not plan.group_precondition.get("exists"):
                        submission["state"] = "blocked"
                        submission["failure_reason"] = (
                            f"Group {group_name!r} is occupied by Submission operation {owner!r}."
                        )
                        publish_submission(cfg, operation)
                        return "blocked"
                    if pending and pending.get("operation_id") not in {None, operation_id}:
                        submission["state"] = "blocked"
                        submission["failure_reason"] = (
                            f"Group {group_name!r} has pending Submission operation {pending.get('operation_id')!r}."
                        )
                        publish_submission(cfg, operation)
                        return "blocked"
        if abort_incomplete and not _staged_tasks_are_exact(cfg, plan):
            _abort_submission_owned(cfg, operation, plan, "incomplete staged truth was safely aborted")
            return "aborted"
        _execute_submission_locked(cfg, operation, plan)
        return "committed"


def submit_specs(
    cfg: Any,
    specs: list[dict[str, Any]],
    *,
    group_name: str | None = None,
    idempotency_key: str | None = None,
    kind: str = "single",
    worker_set: list[str] | dict[str, Any] | None = None,
    on_prepared: Callable[[str, str], None] | None = None,
) -> SubmissionResult:
    request = normalize_submission_request(specs, group_name=group_name, kind=kind, worker_set=worker_set)
    key = idempotency_key or new_id()
    if not isinstance(key, str):
        raise ValueError("idempotency_key must be a string or null.")
    raw_digest = request.raw_request_digest
    mapping_path = idempotency_path(cfg.shared_root, semantic_digest({"project": str(cfg.shared_root), "key": key}))
    with _submission_protocol_lock(cfg, mapping_path.stem):
        if mapping_path.exists():
            mapping = read_json(mapping_path)
            mapped_operation_id = mapping.get("operation_id") if isinstance(mapping, dict) else None
            try:
                validate_identifier(mapped_operation_id, "idempotency mapping operation_id")
            except (TypeError, ValueError) as exc:
                raise RuntimeError("idempotency mapping has an invalid operation_id.") from exc
            operation = read_json(submission_path(cfg.shared_root, mapped_operation_id))
            submission = operation.get("submission") if isinstance(operation, dict) else None
            if not isinstance(submission, dict):
                raise RuntimeError("mapped submission operation is invalid.")
            if submission.get("operation_id") != mapped_operation_id:
                raise RuntimeError("mapped operation_id does not match persisted submission operation_id.")
            if submission.get("idempotency_key") != key:
                raise RuntimeError("persisted idempotency_key does not match requested key.")
            if "raw_request_digest" not in submission:
                raise RuntimeError("persisted submission raw_request_digest is missing.")
            requested_overrides = tuple(validate_tmux_override(item.get("tmux_override")) for item in request.specs)
            digest_matches = submission["raw_request_digest"] == raw_digest
            legacy_inherit_replay = (
                "task_observation" not in operation
                and all(item is None for item in requested_overrides)
                and submission["raw_request_digest"] == legacy_submission_request_digest(request)
            )
            if not digest_matches and not legacy_inherit_replay:
                raise IdempotencyConflict(
                    "idempotency key was already used with different semantic input.",
                    operation_id=mapped_operation_id,
                    idempotency_key=key,
                )
            plan = decode_submission_plan(operation)
            if requested_overrides != plan.tmux_overrides:
                raise IdempotencyConflict(
                    "submission task observation metadata does not match the requested override.",
                    operation_id=mapped_operation_id,
                    idempotency_key=key,
                )
        else:
            plan = prepare_submission_plan(cfg, request, idempotency_key=key)
            operation = encode_submission_plan(plan)
            _reject_cleanup_tombstones(cfg, _plan_specs(plan))
            create_if_absent(submission_path(cfg.shared_root, plan.operation_id), operation)
            create_if_absent(mapping_path, {"operation_id": plan.operation_id})
        return _execute_submission_locked(cfg, operation, plan, on_prepared=on_prepared)
