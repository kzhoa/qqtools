"""Task command workflows for qexp."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from ..config_types import RootConfig
from ..layout import ensure_machine_layout, ensure_shared_layout, is_task_dependencies_root, validate_root_contract
from ..manifest import UNSET, parse_submission_manifest
from ..runtime.availability import (
    AvailabilityTransitionRequest,
    AvailabilityTransitionResult,
    apply_availability_transition,
)
from ..runtime.claims import archive_claim
from ..runtime.dependencies import is_committed_submission_task, normalize_dependency_ids, validate_group_dependencies
from ..runtime.group_discovery.changes import record_task_change
from ..runtime.locks import group_writer_lock, task_lock
from ..runtime.operation_store import operation_exists
from ..runtime.paths import attempt_path, shared_paths
from ..runtime.ready import (
    commit_ready_publication,
    discard_ready_generation,
    prepare_ready_transition,
    reserve_ready_generation,
    retire_previous_ready_generation,
)
from ..runtime.records import AttemptRecord, TaskRecord, utc_now, validate_group_name, validate_identifier
from ..runtime.store import atomic_replace, read_json
from ..runtime.submission import SubmissionResult as RuntimeSubmissionResult
from ..runtime.submission import submit_specs
from ..runtime.tasks import load_task, save_task
from ..submission_contracts import SubmissionRequest
from ..task_observation import validate_tmux_override


def is_cleanup_blocked(task: TaskRecord) -> bool:
    return bool(task.control.get("cleanup_operation_id") or task.control.get("cleanup_state"))


def has_cleanup_operation(cfg: RootConfig, task_id: str) -> bool:
    return operation_exists(cfg, "cleanup", task_id)


def reject_cleanup_blocked(cfg: RootConfig, task: TaskRecord, action: str) -> None:
    if is_cleanup_blocked(task) or has_cleanup_operation(cfg, task.task_id):
        raise ValueError(f"Task {task.task_id!r} is being cleaned and cannot be {action}.")


def submit(
    cfg: RootConfig,
    command: list[str],
    requested_gpus: int = 1,
    requested_cpus: int | None = None,
    task_id: str | None = None,
    name: str | None = None,
    group: str | None = None,
    working_dir: str | Path | None = None,
    home_machine: str | None = None,
    sharing_mode: str = "private",
    fallback_machines: str | list[str] = "group",
    offer_after_seconds: int | None = None,
    depends_on_task_ids: list[str] | None = None,
    idempotency_key: str | None = None,
    tmux_override: bool | None = None,
    on_prepared: Callable[[str, str], None] | None = None,
) -> TaskRecord:
    validate_tmux_override(tmux_override)
    validate_root_contract(cfg)
    ensure_shared_layout(cfg)
    ensure_machine_layout(cfg)
    item = {
        "task_id": task_id,
        "name": name,
        "command": list(command),
        "requested_gpus": requested_gpus,
        "requested_cpus": requested_cpus,
        "working_directory": str(Path(working_dir or Path.cwd()).resolve()),
        "home_machine": "current" if home_machine is None else home_machine,
        "sharing_mode": sharing_mode,
        "fallback_machines": fallback_machines,
        "offer_after_seconds": offer_after_seconds,
        "depends_on_task_ids": depends_on_task_ids or [],
        "tmux_override": tmux_override,
    }
    return submit_specs(cfg, [item], group_name=validate_group_name(group), idempotency_key=idempotency_key)[0]


def submit_request(
    cfg: RootConfig,
    request: SubmissionRequest,
    *,
    on_prepared: Callable[[str, str], None] | None = None,
) -> RuntimeSubmissionResult | Any:
    """Enter the single submission workflow for command and file requests."""
    if not isinstance(request, SubmissionRequest):
        raise TypeError("request must be a qexp SubmissionRequest.")
    validate_root_contract(cfg)
    if request.dry_run:
        # Preview is an explicitly separate runtime entry point.  Do not fall
        # back to submit_specs: a preview must not reserve an idempotency key or
        # create durable Task/operation truth.
        from ..runtime.submission import preview_specs

        return preview_specs(
            cfg,
            request.normalized_specs,
            group_name=request.group_name,
            idempotency_key=request.idempotency_key,
            kind="single" if request.mode == "command" else "bulk",
            worker_set=request.worker_set if request.workers_declared else None,
        )
    ensure_shared_layout(cfg)
    ensure_machine_layout(cfg)
    return submit_specs(
        cfg,
        request.normalized_specs,
        group_name=request.group_name,
        idempotency_key=request.idempotency_key,
        kind="single" if request.mode == "command" else "bulk",
        worker_set=request.worker_set if request.workers_declared else None,
        on_prepared=on_prepared,
    )


def batch_submit(
    cfg: RootConfig,
    manifest_path: Path,
    *,
    group: str | None = None,
    idempotency_key: str | None = None,
    tmux_override: bool | None = None,
    on_prepared: Callable[[str, str], None] | None = None,
) -> RuntimeSubmissionResult:
    validate_tmux_override(tmux_override)
    validate_root_contract(cfg)
    group_name = validate_group_name(group)
    manifest = parse_submission_manifest(
        Path(manifest_path),
        group_name=group_name if group_name is not None else UNSET,
        tmux_override=tmux_override if tmux_override is not None else UNSET,
    )
    return submit_specs(
        cfg,
        list(manifest.specs),
        group_name=manifest.group_name,
        idempotency_key=idempotency_key,
        kind="bulk",
        worker_set=manifest.workers if manifest.workers_declared else None,
        on_prepared=on_prepared,
    )


def cancel(
    cfg: RootConfig,
    task_id: str,
    *,
    terminate_running: bool = True,
    reservation_runtime_root: Path | None = None,
) -> TaskRecord:
    from ..agent.context import resolve_execution_context
    from ..scheduler import cancel_task

    reservation_runtime_root = reservation_runtime_root or resolve_execution_context(cfg).reservation_root
    return cancel_task(
        cfg,
        task_id,
        terminate_running=terminate_running,
        reservation_runtime_root=reservation_runtime_root,
    )


SUPERSESSION_WARNING = (
    "The old process may still be running and may continue external output writes; duplicate execution is possible."
)


def _supersession_operation_id(task_id: str, attempt_id: str, fencing_token: int, source_revision: int) -> str:
    """Build the stable operation identity for one exact supersession request."""
    identity = {
        "task_id": task_id,
        "attempt_id": attempt_id,
        "fencing_token": fencing_token,
        "source_task_revision": source_revision,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _supersession_observation(attempt: AttemptRecord) -> dict[str, Any]:
    """Keep a bounded summary of the Attempt evidence visible at supersession."""
    process = attempt.process
    result = attempt.result
    timestamps = attempt.timestamps
    return {
        "phase": attempt.phase,
        "process": {
            key: process.get(key)
            for key in (
                "wrapper_pid",
                "wrapper_start_time_ticks",
                "process_group_id",
                "process_group_start_time_ticks",
            )
        },
        "result": {key: result.get(key) for key in ("exit_code", "signal", "category", "reason")},
        "timestamps": {
            key: timestamps.get(key)
            for key in (
                "launch_authorized_at",
                "process_created_at",
                "running_at",
                "orphaned_at",
                "recovered_at",
                "finished_at",
            )
        },
    }


def _supersession_annotation(receipt: dict[str, Any]) -> dict[str, Any]:
    """Build the bounded historical Attempt annotation from its Task receipt."""
    return {
        "operation_id": receipt["operation_id"],
        "task_id": receipt["task_id"],
        "attempt_id": receipt["attempt_id"],
        "attempt_number": receipt["attempt_number"],
        "fencing_token": receipt["fencing_token"],
        "reservation_id": receipt["reservation_id"],
        "old_machine": receipt["old_machine"],
        "timestamp": receipt["timestamp"],
    }


def _validate_supersession_receipt(
    task: TaskRecord, requested_attempt_id: str, receipt: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Return a valid replay receipt for the requested old Attempt, if present."""
    receipt = receipt if receipt is not None else task.attempt_control.get("last_supersession")
    if receipt is None:
        return None
    if not isinstance(receipt, dict):
        raise ValueError("Task has a malformed attempt supersession receipt.")
    if receipt.get("attempt_id") != requested_attempt_id:
        return None
    required = {
        "operation_id",
        "task_id",
        "operator_machine",
        "timestamp",
        "source_task_revision",
        "old_machine",
        "attempt_id",
        "attempt_number",
        "fencing_token",
        "reservation_id",
        "last_observation",
        "observation_uncertain",
        "duplicate_risk_acknowledged",
        "terminate_old_process",
    }
    if not required.issubset(receipt):
        raise ValueError("Task has a malformed attempt supersession receipt.")
    if (
        receipt.get("task_id") != task.task_id
        or receipt.get("attempt_id") != requested_attempt_id
        or type(receipt.get("source_task_revision")) is not int
        or receipt["source_task_revision"] < 0
        or type(receipt.get("attempt_number")) is not int
        or receipt["attempt_number"] < 1
        or type(receipt.get("fencing_token")) is not int
        or receipt["fencing_token"] < 1
        or type(receipt.get("observation_uncertain")) is not bool
        or receipt.get("duplicate_risk_acknowledged") is not True
        or receipt.get("terminate_old_process") is not False
        or not isinstance(receipt.get("last_observation"), dict)
        or not isinstance(receipt.get("timestamp"), str)
        or not receipt["timestamp"]
    ):
        raise ValueError("Task has a malformed attempt supersession receipt.")
    try:
        validate_identifier(receipt["operation_id"], "supersession operation_id")
        validate_identifier(receipt["operator_machine"], "supersession operator machine")
        validate_identifier(receipt["old_machine"], "supersession old machine")
        validate_identifier(receipt["reservation_id"], "supersession reservation_id")
    except (TypeError, ValueError) as exc:
        raise ValueError("Task has a malformed attempt supersession receipt.") from exc
    if receipt["operation_id"] != _supersession_operation_id(
        task.task_id,
        requested_attempt_id,
        receipt["fencing_token"],
        receipt["source_task_revision"],
    ):
        raise ValueError("Task has a conflicting attempt supersession operation ID.")
    return receipt


def _validate_active_supersession(
    task: TaskRecord,
    requested_attempt_id: str,
    attempt: AttemptRecord,
) -> tuple[dict[str, Any], int, int]:
    """Validate the exact active durable ownership eligible for supersession."""
    if task.state.get("projection") != "running":
        raise ValueError("attempt supersession requires a running Task.")
    current_id = task.attempt_control.get("current_attempt_id")
    current_number = task.attempt_control.get("current_attempt_number")
    if current_id != requested_attempt_id:
        raise ValueError("--supersede-attempt must match the current Attempt ID.")
    if type(current_number) is not int or current_number < 1:
        raise ValueError("Task has no valid current Attempt number for supersession.")
    claim = task.claim_control.get("active_claim")
    if not isinstance(claim, dict):
        raise ValueError("attempt supersession requires an exact current active claim.")
    if claim.get("attempt_id") != requested_attempt_id:
        raise ValueError("--supersede-attempt must match the current active claim Attempt ID.")
    if claim.get("attempt_number") != current_number:
        raise ValueError("active claim and current Attempt number conflict.")
    if attempt.task_id != task.task_id or attempt.attempt_id != requested_attempt_id:
        raise ValueError("current Attempt identity does not match --supersede-attempt.")
    if attempt.attempt_number != current_number:
        raise ValueError("current Attempt number does not match the active claim.")
    if attempt.phase not in {"starting", "running"}:
        if attempt.phase == "claimed":
            raise ValueError("pre-launch claimed Attempts cannot be superseded.")
        raise ValueError(f"Attempt phase {attempt.phase!r} cannot be superseded.")
    if claim.get("authority_mode") != "holder_bound" or attempt.authority_mode != "holder_bound":
        raise ValueError("attempt supersession requires holder_bound ownership on the claim and Attempt.")
    token = claim.get("fencing_token")
    if type(token) is not int or token < 1:
        raise ValueError("active claim has an invalid fencing token.")
    if attempt.current_fencing_token != token:
        raise ValueError("active claim and Attempt fencing tokens conflict.")
    machine_name = claim.get("machine_name")
    if not isinstance(machine_name, str) or not machine_name or attempt.machine_name != machine_name:
        raise ValueError("active claim and Attempt machine identities conflict.")
    reservation_id = claim.get("reservation_id")
    if not isinstance(reservation_id, str) or not reservation_id or attempt.reservation_id != reservation_id:
        raise ValueError("active claim and Attempt reservation identities conflict.")
    if claim.get("claim_id") not in {None, requested_attempt_id}:
        raise ValueError("active claim identity evidence conflicts with the current Attempt.")
    if claim.get("launch_state") not in {"starting", "running"}:
        raise ValueError("active claim is not in a starting or running launch state.")
    launch_id = claim.get("launch_id")
    if not isinstance(launch_id, str) or not launch_id:
        raise ValueError("active claim is missing its durable launch identity.")
    ownership_transition = claim.get("ownership_transition")
    if ownership_transition is not None:
        if not isinstance(ownership_transition, dict):
            raise ValueError("active claim ownership transition evidence is malformed.")
        if any(
            ownership_transition.get(field) != expected
            for field, expected in (
                ("task_id", task.task_id),
                ("attempt_id", requested_attempt_id),
                ("attempt_number", current_number),
                ("fencing_token", token),
                ("machine_name", machine_name),
                ("reservation_id", reservation_id),
                ("launch_id", launch_id),
                ("target_authority_mode", "holder_bound"),
            )
        ):
            raise ValueError("active claim ownership transition identity conflicts with the Attempt.")
        if not isinstance(ownership_transition.get("source_active_lease_evidence"), dict):
            raise ValueError("active claim ownership transition lease evidence is malformed.")
    authorization = attempt.authorization
    if not isinstance(authorization, dict):
        raise ValueError("Attempt authorization evidence is malformed.")
    if authorization.get("launch_id") != launch_id:
        raise ValueError("active claim and Attempt launch identities conflict.")
    lease = attempt.lease
    if not isinstance(lease, dict) or lease.get("expires_at") is not None or lease.get("clock_evidence") is not None:
        raise ValueError("holder_bound Attempt contains conflicting lease evidence.")
    if any(
        claim.get(key) is not None
        for key in ("clock_error_bound_seconds", "clock_provider", "clock_observation_id", "lease_expires_at")
    ):
        raise ValueError("holder_bound claim contains conflicting lease evidence.")
    if authorization.get("ownership_superseded") is not None:
        raise ValueError("active Attempt already contains conflicting supersession evidence.")
    if (
        not isinstance(attempt.process, dict)
        or not isinstance(attempt.result, dict)
        or not isinstance(attempt.timestamps, dict)
    ):
        raise ValueError("Attempt observation evidence is malformed.")
    if not isinstance(attempt.termination, dict):
        raise ValueError("Attempt termination evidence is malformed.")
    if (
        not isinstance(attempt.token_history, list)
        or any(type(value) is not int or value < 1 for value in attempt.token_history)
        or token not in attempt.token_history
    ):
        raise ValueError("Attempt token history is malformed or conflicts with the active claim.")
    if not set(("expires_at", "clock_evidence")).issubset(lease):
        raise ValueError("holder_bound Attempt lease evidence is incomplete.")
    if any(
        attempt.termination.get(key) is not None
        for key in ("requested_by_operation_id", "requested_at", "acknowledged_at", "result")
    ):
        raise ValueError("Attempt has unresolved termination evidence.")
    if any(
        claim.get(key) is not None
        for key in ("termination_decision_id", "termination_decision_token", "termination_committed_at")
    ):
        raise ValueError("active claim has unresolved termination evidence.")
    if any(attempt.result.get(key) is not None for key in ("exit_code", "signal", "category", "reason")):
        raise ValueError("active Attempt contains conflicting terminal result evidence.")
    if (
        any(
            task.control.get(key) is not None and task.control.get(key) is not False
            for key in (
                "cancellation_requested_at",
                "cancellation_operation_id",
                "termination_acknowledged_at",
                "termination_result",
                "removal_operation_id",
                "removal_state",
            )
        )
        or task.control.get("terminate_running") is True
    ):
        raise ValueError("Task has unresolved cancellation or termination control.")
    fencing_epoch = task.claim_control.get("fencing_epoch")
    if type(fencing_epoch) is not int or fencing_epoch != token:
        raise ValueError("Task fencing epoch conflicts with the active Attempt.")
    attempt_revision = attempt.meta.get("revision") if isinstance(attempt.meta, dict) else None
    if type(attempt_revision) is not int or attempt_revision < 0:
        raise ValueError("Attempt metadata revision is malformed.")
    return claim, current_number, token


def _project_supersession_annotation(cfg: RootConfig, receipt: dict[str, Any]) -> None:
    """Atomically add only the matching ownership annotation to historical truth."""
    path = attempt_path(cfg.shared_root, receipt["task_id"], receipt["attempt_number"])
    try:
        stored = read_json(path)
    except FileNotFoundError as exc:
        raise ValueError("historical Attempt is missing; cannot repair supersession evidence.") from exc
    try:
        attempt = AttemptRecord.from_dict(stored)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError("historical Attempt is malformed; cannot repair supersession evidence.") from exc
    if (
        attempt.task_id != receipt["task_id"]
        or attempt.attempt_id != receipt["attempt_id"]
        or attempt.attempt_number != receipt["attempt_number"]
        or attempt.current_fencing_token != receipt["fencing_token"]
        or attempt.machine_name != receipt["old_machine"]
        or attempt.reservation_id != receipt["reservation_id"]
    ):
        raise ValueError("historical Attempt identity conflicts with the supersession receipt.")
    if not isinstance(attempt.authorization, dict):
        raise ValueError("historical Attempt authorization evidence is malformed.")
    annotation = _supersession_annotation(receipt)
    existing = attempt.authorization.get("ownership_superseded")
    if existing is not None:
        if existing != annotation:
            raise ValueError("historical Attempt has conflicting supersession evidence.")
        if attempt.authorization.get("supersession_receipt") == receipt:
            return
    attempt.authorization["supersession_receipt"] = dict(receipt)
    attempt.authorization["ownership_superseded"] = annotation
    if not isinstance(attempt.meta, dict) or type(attempt.meta.get("revision")) is not int:
        raise ValueError("historical Attempt metadata is malformed.")
    attempt.meta["revision"] += 1
    attempt.meta["updated_at"] = utc_now()
    if atomic_replace(path, attempt.to_dict()) is None:
        raise OSError("historical Attempt supersession projection durability is unknown.")


def _emit_supersession_event(cfg: RootConfig, receipt: dict[str, Any]) -> None:
    """Publish one deterministic audit event and repair it idempotently on replay."""
    timestamp = receipt["timestamp"]
    if (
        len(timestamp) < 10
        or timestamp[4] != "-"
        or timestamp[7] != "-"
        or not timestamp[:4].isdigit()
        or not timestamp[5:7].isdigit()
        or not timestamp[8:10].isdigit()
    ):
        raise ValueError("supersession receipt timestamp is malformed.")
    operation_id = receipt["operation_id"]
    path = shared_paths(cfg.shared_root)["events"] / timestamp[:10] / f"{operation_id}.json"
    details = {
        "operation_id": operation_id,
        "attempt_id": receipt["attempt_id"],
        "attempt_number": receipt["attempt_number"],
        "fencing_token": receipt["fencing_token"],
        "reservation_id": receipt["reservation_id"],
        "old_machine": receipt["old_machine"],
        "operator_machine": receipt["operator_machine"],
        "operator": receipt["operator_machine"],
        "source_task_revision": receipt["source_task_revision"],
        "timestamp": timestamp,
        "duplicate_execution_risk": True,
        "duplicate_risk_acknowledged": True,
        "terminate_old_process": False,
    }
    try:
        existing = read_json(path)
    except FileNotFoundError:
        existing = None
    if existing is not None:
        if (
            existing.get("event_type") != "attempt_superseded_by_retry"
            or existing.get("task_id") != receipt["task_id"]
            or existing.get("details") != details
        ):
            raise ValueError("supersession audit event conflicts with the committed receipt.")
        return
    event = {
        "event_id": operation_id,
        "event_type": "attempt_superseded_by_retry",
        "task_id": receipt["task_id"],
        "machine_name": receipt["operator_machine"],
        "timestamp": timestamp,
        "details": details,
    }
    if atomic_replace(path, event) is None:
        raise OSError("supersession audit event durability is unknown.")


def _retry_superseded_attempt(cfg: RootConfig, task_id: str, requested_attempt_id: str) -> TaskRecord:
    """Explicitly abandon one exact active durable Attempt and queue a successor."""
    validate_identifier(requested_attempt_id, "supersede_attempt")
    initial = load_task(cfg, task_id)
    reserved_generation: int | None = None
    is_committed = False
    try:
        from ..scheduler import authority_locks

        with authority_locks(cfg, initial):
            task = load_task(cfg, task_id)
            receipt = _validate_supersession_receipt(task, requested_attempt_id)
            if receipt is None:
                from ..runtime.terminal_evidence import canonical_attempt_number

                number = canonical_attempt_number(task_id, requested_attempt_id)
                if number is not None:
                    try:
                        historical = AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task_id, number)))
                    except FileNotFoundError:
                        historical = None
                    if historical is not None and historical.attempt_id == requested_attempt_id:
                        saved = historical.authorization.get("supersession_receipt")
                        if isinstance(saved, dict):
                            receipt = _validate_supersession_receipt(task, requested_attempt_id, saved)
            active_claim = task.claim_control.get("active_claim")
            if receipt is not None:
                if isinstance(active_claim, dict) and active_claim.get("attempt_id") == requested_attempt_id:
                    raise ValueError("Task supersession receipt conflicts with an active claim for the same Attempt.")
                _project_supersession_annotation(cfg, receipt)
                _emit_supersession_event(cfg, receipt)
                task.attempt_control["last_supersession"] = receipt
                return task

            reject_cleanup_blocked(cfg, task, "retried")
            previous = task.attempt_control.get("last_supersession")
            if isinstance(previous, dict):
                # Preserve the previous operation at its immutable Attempt
                # locator before replacing the bounded current Task receipt.
                _project_supersession_annotation(cfg, previous)
                _emit_supersession_event(cfg, previous)
            current_number = task.attempt_control.get("current_attempt_number")
            if type(current_number) is not int or current_number < 1:
                raise ValueError("Task has no current Attempt to supersede.")
            try:
                current = AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task_id, current_number)))
            except FileNotFoundError as exc:
                raise ValueError("current Attempt is missing; supersession cannot proceed.") from exc
            except (OSError, KeyError, TypeError, ValueError, AttributeError) as exc:
                raise ValueError("current Attempt is malformed; supersession cannot proceed.") from exc
            claim, attempt_number, fencing_token = _validate_active_supersession(task, requested_attempt_id, current)
            source_revision = task.meta.get("revision")
            if type(source_revision) is not int or source_revision < 0:
                raise ValueError("Task metadata revision is malformed.")
            reserved_generation = task.ready_generation + 1
            reference = reserve_ready_generation(
                cfg,
                task_id,
                reserved_generation,
                "home",
                task.placement_policy["home_machine"],
            )

            # Re-read the authoritative pair after ready capacity is reserved so
            # the receipt cannot bind a changed Task or identity.
            rechecked_task = load_task(cfg, task_id)
            if rechecked_task.meta.get("revision") != source_revision:
                raise ValueError("Task changed while preparing attempt supersession.")
            rechecked_number = rechecked_task.attempt_control.get("current_attempt_number")
            if rechecked_number != attempt_number or rechecked_task.ready_generation + 1 != reserved_generation:
                raise ValueError("Task Attempt identity changed while preparing supersession.")
            try:
                rechecked = AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task_id, rechecked_number)))
            except FileNotFoundError as exc:
                raise ValueError("current Attempt disappeared while preparing supersession.") from exc
            except (OSError, KeyError, TypeError, ValueError, AttributeError) as exc:
                raise ValueError("current Attempt became malformed while preparing supersession.") from exc
            _validate_active_supersession(rechecked_task, requested_attempt_id, rechecked)
            if (
                rechecked.current_fencing_token != fencing_token
                or rechecked.machine_name != current.machine_name
                or rechecked.reservation_id != current.reservation_id
            ):
                raise ValueError("Attempt identity changed while preparing supersession.")
            task = rechecked_task
            current = rechecked
            claim = task.claim_control["active_claim"]
            timestamp = utc_now()
            operation_id = _supersession_operation_id(task_id, requested_attempt_id, fencing_token, source_revision)
            receipt = {
                "operation_id": operation_id,
                "task_id": task_id,
                "operator_machine": cfg.machine_name,
                "timestamp": timestamp,
                "source_task_revision": source_revision,
                "old_machine": current.machine_name,
                "attempt_id": requested_attempt_id,
                "attempt_number": attempt_number,
                "fencing_token": fencing_token,
                "reservation_id": current.reservation_id,
                "last_observation": _supersession_observation(current),
                "observation_uncertain": True,
                "duplicate_risk_acknowledged": True,
                "terminate_old_process": False,
            }
            with record_task_change(
                cfg,
                task,
                "retry",
                details={
                    "reserved_generation": reserved_generation,
                    "operation_id": operation_id,
                    "attempt_id": requested_attempt_id,
                    "attempt_number": attempt_number,
                    "fencing_token": fencing_token,
                    "transition": "attempt_superseded_by_retry",
                },
            ):
                archive_claim(cfg, task_id, claim, "attempt_superseded_by_retry")
                task.claim_control["active_claim"] = None
                task.claim_control["fencing_epoch"] += 1
                task.attempt_control["last_supersession"] = receipt
                task.attempt_control["current_attempt_id"] = None
                task.state = {"projection": "queued", "reason": "attempt_superseded_by_retry"}
                task.control.update(
                    {
                        "cancellation_requested_at": None,
                        "cancellation_operation_id": None,
                        "terminate_running": False,
                        "requested_by": None,
                        "termination_acknowledged_at": None,
                        "termination_result": None,
                    }
                )
                task.placement_runtime.update(
                    {
                        "queue_scope": "home",
                        "queued_home_at": utc_now(),
                        "offered_at": None,
                        "offer_reason": None,
                        "offered_by": None,
                    }
                )
                old_generation, _ = prepare_ready_transition(cfg, task, "retry", reference=reference)
                task.meta["revision"] += 1
                task.meta["updated_at"] = utc_now()
                save_task(cfg, task)
                is_committed = True
            commit_ready_publication(cfg, task)
            retire_previous_ready_generation(cfg, old_generation, task)
            _project_supersession_annotation(cfg, receipt)
            _emit_supersession_event(cfg, receipt)
            return task
    finally:
        if not is_committed and reserved_generation is not None:
            try:
                from ..scheduler import authority_locks

                with authority_locks(cfg, initial):
                    try:
                        current_task = load_task(cfg, task_id)
                    except FileNotFoundError:
                        current_task = None
                    if current_task is None or current_task.ready_generation != reserved_generation:
                        discard_ready_generation(cfg, task_id, reserved_generation)
            except (OSError, KeyError, TypeError, ValueError, RuntimeError):
                pass


def retry(cfg: RootConfig, task_id: str, *, supersede_attempt: str | None = None) -> TaskRecord:
    """Queue the next Attempt after a failed or orphaned current Attempt."""
    if supersede_attempt is not None:
        return _retry_superseded_attempt(cfg, task_id, supersede_attempt)
    from ..scheduler import authority_locks

    initial = load_task(cfg, task_id)
    reserved_generation = initial.ready_generation + 1
    reference = reserve_ready_generation(
        cfg,
        task_id,
        reserved_generation,
        "home",
        initial.placement_policy["home_machine"],
    )
    is_committed = False
    try:
        with authority_locks(cfg, initial):
            task = load_task(cfg, task_id)
            reject_cleanup_blocked(cfg, task, "retried")
            if task.claim_control.get("active_claim"):
                raise ValueError("a Task with an active claim cannot be retried.")
            number = task.attempt_control.get("current_attempt_number")
            if number is None:
                raise ValueError("Task has no current Attempt to retry.")
            current_path = attempt_path(cfg.shared_root, task_id, number)
            current = AttemptRecord.from_dict(read_json(current_path))
            if task.state["projection"] == "failed" and current.phase == "failed":
                retry_mode = "failed"
            elif task.state["projection"] == "blocked" and current.phase == "orphaned":
                retry_mode = "orphaned"
            else:
                raise ValueError(
                    "only a failed Task or a blocked Task with an orphaned current Attempt can be retried."
                )
            with record_task_change(
                cfg,
                task,
                "retry",
                details={"reserved_generation": reserved_generation},
            ):
                if retry_mode == "failed":
                    task.state = {"projection": "queued", "reason": None}
                else:
                    superseded_at = utc_now()
                    task.claim_control["fencing_epoch"] += 1
                    from ..events import write_event

                    write_event(
                        cfg,
                        "orphan_superseded_by_retry",
                        task_id=task_id,
                        details={
                            "attempt_id": current.attempt_id,
                            "fencing_token": current.current_fencing_token,
                            "operator": cfg.machine_name,
                            "timestamp": superseded_at,
                        },
                    )
                    task.state = {"projection": "queued", "reason": "orphan_superseded_by_retry"}
                task.control.update(
                    {
                        "cancellation_requested_at": None,
                        "cancellation_operation_id": None,
                        "terminate_running": False,
                        "requested_by": None,
                        "termination_acknowledged_at": None,
                        "termination_result": None,
                    }
                )
                task.placement_runtime.update(
                    {
                        "queue_scope": "home",
                        "queued_home_at": utc_now(),
                        "offered_at": None,
                        "offer_reason": None,
                        "offered_by": None,
                    }
                )
                task.attempt_control["current_attempt_id"] = None
                old_generation, _ = prepare_ready_transition(cfg, task, "retry", reference=reference)
                task.meta["revision"] += 1
                task.meta["updated_at"] = utc_now()
                save_task(cfg, task)
                is_committed = True
                commit_ready_publication(cfg, task)
                retire_previous_ready_generation(cfg, old_generation, task)
                return task
    finally:
        if not is_committed:
            # A rename can commit Task truth before its durability call raises.
            # Never retire the generation now named by authoritative truth.
            try:
                with authority_locks(cfg, initial):
                    try:
                        current_task = load_task(cfg, task_id)
                    except FileNotFoundError:
                        current_task = None
                    if current_task is None or current_task.ready_generation != reserved_generation:
                        discard_ready_generation(cfg, task_id, reserved_generation)
            except (OSError, KeyError, TypeError, ValueError, RuntimeError):
                pass


def edit_dependencies(
    cfg: RootConfig,
    task_id: str,
    dependency_ids: list[str],
    *,
    action: str = "replace",
) -> TaskRecord:
    """Atomically replace, add, or remove dependencies from an unstarted Task."""
    validate_root_contract(cfg)
    if not is_task_dependencies_root(cfg):
        raise ValueError("Task dependencies require an activated task-dependencies-v1 root.")
    initial = load_task(cfg, task_id)
    if initial.group_name is None:
        raise ValueError("ungrouped tasks cannot declare dependencies.")
    requested = normalize_dependency_ids(dependency_ids)
    with group_writer_lock(cfg, initial.group_name):
        with task_lock(cfg.shared_root, task_id):
            task = load_task(cfg, task_id)
            reject_cleanup_blocked(cfg, task, "have dependencies edited")
            if not is_committed_submission_task(cfg, task):
                raise ValueError("a Task whose submission is not committed cannot have dependencies edited.")
            if task.group_name != initial.group_name:
                raise RuntimeError("Task Group changed while editing dependencies.")
            if task.state["projection"] in {"succeeded", "failed", "cancelled"}:
                raise ValueError("terminal tasks cannot have dependencies edited.")
            if task.claim_control.get("active_claim") or task.attempt_control["next_attempt_number"] != 1:
                raise ValueError("a Task that has created an Attempt cannot have dependencies edited.")
            current = set(task.depends_on_task_ids)
            if action == "replace":
                updated = requested
            elif action == "add":
                updated = sorted(current | set(requested))
            elif action == "remove":
                updated = sorted(current - set(requested))
            else:
                raise ValueError(f"unknown dependency edit action {action!r}.")
            candidate = TaskRecord.from_dict(task.to_dict())
            candidate.depends_on_task_ids = updated
            validate_group_dependencies(cfg, task.group_name, [candidate])
            with record_task_change(
                cfg,
                task,
                "dependency_edit",
                details={"action": action},
            ):
                task.depends_on_task_ids = updated
                task.meta["revision"] += 1
                task.meta["updated_at"] = utc_now()
                save_task(cfg, task)
            return task


def share(
    cfg: RootConfig, task_id: str, *, after_seconds: int | None = None, helper_machines: list[str] | None = None
) -> AvailabilityTransitionResult:
    if after_seconds is not None and after_seconds < 0:
        raise ValueError("share --after must be non-negative.")
    action = "share_after" if after_seconds is not None else "share_now"
    return apply_availability_transition(
        cfg,
        AvailabilityTransitionRequest(
            action=action,
            task_id=task_id,
            helper_machines=helper_machines,
            after_seconds=after_seconds,
            reason="manual",
        ),
    )


def keep_local(cfg: RootConfig, task_id: str) -> AvailabilityTransitionResult:
    return apply_availability_transition(
        cfg, AvailabilityTransitionRequest(action="keep_local", task_id=task_id, reason="manual")
    )


def offer(cfg: RootConfig, task_id: str, *, reason: str = "manual") -> AvailabilityTransitionResult:
    action = "elapsed_offer" if reason == "elapsed" else "manual_offer"
    return apply_availability_transition(
        cfg, AvailabilityTransitionRequest(action=action, task_id=task_id, reason=reason)
    )
