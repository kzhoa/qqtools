"""Durable bounded advancement for explicit whole-project maintenance."""

from __future__ import annotations

import errno
import random
import stat
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from ..config_types import RootConfig
from ..layout import project_id
from ..runtime.availability.offer_deadlines import advance_deadline_index_rebuild_step
from ..runtime.availability.transitions import reconcile_availability_operations
from ..runtime.directory_capture import read_directory_entry
from ..runtime.locks import exclusive, schema_lock
from ..runtime.maintenance_outbox import (
    activate_work,
    prepare_work,
    read_work,
    retire_work,
    select_due_work,
    update_work,
)
from ..runtime.observation.maintenance import ObservationMaintenance, request_rebuild
from ..runtime.observation.projection import inspect_observation
from ..runtime.operation_store import locate_operation_path
from ..runtime.paths import shared_paths, submission_path, task_path
from ..runtime.records import AttemptRecord, TaskRecord, new_id, utc_now, validate_identifier
from ..runtime.store import atomic_replace, read_json_limited, require_json_size
from ..runtime.submission import reconcile_submission
from ..runtime.submission_control import inspect_submission_control, request_control_rebuild
from ..runtime.work_budget import InvocationLedger, OperationReservation

PHASES = (
    "submission",
    "cleanup",
    "availability",
    "group_cancel",
    "deadline_index",
    "orphan_recovery",
    "ready_index",
    "group_ready_members",
    "task_observation",
    "submission_control",
    "final_verification",
)

# QQTOOLS-COMPAT-0018: an explicit full-audit request remains resumable across CLI slices.
# QQTOOLS-COMPAT-0019: read Stage 1 full-audit descriptors and migrate them to
# schema 2 without treating an unknown historical capture as complete.
DESCRIPTOR_SCHEMA_VERSION = 2
DESCRIPTOR_MAX_BYTES = 64 * 1024
_MAX_SOURCE_RECORD_BYTES = 1_048_576
_MAX_CURSOR_BYTES = 8 * 1024
_OPERATION_LIMIT = 256
_DEADLINE_MS = 50
_SETUP_OPERATIONS = 20
_EXIT_RESERVATION_OPERATIONS = 16
CONTEXT_RESOLUTION_OPERATIONS = 24

# Worst-case operation reservations include bounded source I/O, helper locks,
# helper commits, and the descriptor checkpoint. Keep every one-step cost below
# the invocation hard limit so every phase remains reachable on a fresh slice.
PHASE_OPERATION_RESERVATIONS = {
    "descriptor_initialization": 24,
    "submission": 24,
    "cleanup": 96,
    "availability": 80,
    "group_cancel": 112,
    "deadline_index": 112,
    "orphan_recovery": 128,
    "ready_index": 176,
    "group_ready_members": 176,
    "task_observation": 112,
    "submission_control": 112,
    "final_verification": 24,
}

_IDENTITY_FIELDS = frozenset({"project_id", "kind", "target_id", "work_generation"})
_STAGE1_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "identity",
        "capture_id",
        "state",
        "phase",
        "cursor",
        "progress_revision",
        "completed_phases",
        "phase_evidence",
        "prior_degraded_reasons",
        "last_progress_at",
        "created_at",
        "failure",
    }
)
_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "identity",
        "capture_id",
        "source_capture",
        "source_revision",
        "build_identity",
        "state",
        "phase",
        "cursor",
        "progress_revision",
        "completed_phases",
        "phase_evidence",
        "prior_degraded_reasons",
        "last_progress_at",
        "meaningful_progress_at",
        "created_at",
        "due_at",
        "retry_count",
        "failure",
    }
)
_POINTER_FIELDS = frozenset({"schema_version", "identity", "descriptor_file", "progress_revision"})


@contextmanager
def _repair_lock(cfg: RootConfig) -> Iterator[bool]:
    path = shared_paths(cfg.shared_root)["locks"] / "maintenance-v1.lock"
    with exclusive(path, blocking=False) as acquired:
        yield acquired


def _maintenance_root(cfg: RootConfig) -> Path:
    return shared_paths(cfg.shared_root)["submissions"].parent / "maintenance-v1"


def _current_path(cfg: RootConfig) -> Path:
    return _maintenance_root(cfg) / "full-audit.current.json"


def _descriptor_path(cfg: RootConfig, generation: str) -> Path:
    validate_identifier(generation, "work_generation")
    return _maintenance_root(cfg) / f"{generation}.json"


def _expected_identity(cfg: RootConfig, generation: str) -> dict[str, str]:
    return {
        "project_id": project_id(cfg.shared_root),
        "kind": "full_audit",
        "target_id": "project",
        "work_generation": generation,
    }


def _validate_identity(value: object, *, cfg: RootConfig) -> dict[str, str]:
    if type(value) is not dict or set(value) != _IDENTITY_FIELDS:
        raise ValueError("maintenance identity is invalid.")
    project = value.get("project_id")
    kind = value.get("kind")
    target = value.get("target_id")
    generation = value.get("work_generation")
    if (
        type(project) is not str
        or project != project_id(cfg.shared_root)
        or kind != "full_audit"
        or target != "project"
    ):
        raise ValueError("maintenance identity does not match this Project.")
    validate_identifier(generation, "work_generation")
    return {"project_id": project, "kind": kind, "target_id": target, "work_generation": generation}


def _read_bounded_record(path: Path, record_type: str) -> dict[str, Any]:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > DESCRIPTOR_MAX_BYTES:
        raise ValueError(f"{record_type} is not a bounded regular file.")
    return read_json_limited(path, max_bytes=DESCRIPTOR_MAX_BYTES, record_type=record_type)


def _read_pointer(cfg: RootConfig) -> dict[str, Any] | None:
    path = _current_path(cfg)
    try:
        pointer = _read_bounded_record(path, "maintenance_current")
    except FileNotFoundError:
        return None
    if set(pointer) != _POINTER_FIELDS or pointer.get("schema_version") not in {1, DESCRIPTOR_SCHEMA_VERSION}:
        raise ValueError("maintenance current pointer is malformed.")
    identity = _validate_identity(pointer.get("identity"), cfg=cfg)
    if pointer.get("descriptor_file") != f"{identity['work_generation']}.json":
        raise ValueError("maintenance current pointer target is invalid.")
    revision = pointer.get("progress_revision")
    if type(revision) is not int or revision < 0:
        raise ValueError("maintenance current pointer revision is invalid.")
    return {**pointer, "identity": identity}


def _validate_record(value: dict[str, Any], *, cfg: RootConfig, identity: dict[str, str]) -> dict[str, Any]:
    if set(value) != _RECORD_FIELDS or value.get("schema_version") != DESCRIPTOR_SCHEMA_VERSION:
        raise ValueError("maintenance descriptor schema is invalid.")
    record_identity = _validate_identity(value.get("identity"), cfg=cfg)
    if record_identity != identity or value.get("capture_id") != identity["work_generation"]:
        raise ValueError("maintenance descriptor identity does not match its current target.")
    state = value.get("state")
    if state not in {"pending", "running", "waiting", "completed", "intervention"}:
        raise ValueError("maintenance descriptor state is invalid.")
    phase = value.get("phase")
    if phase not in PHASES:
        raise ValueError("maintenance descriptor phase is invalid.")
    cursor = value.get("cursor")
    if type(cursor) is not dict:
        raise ValueError("maintenance descriptor cursor is invalid.")
    require_json_size({"cursor": cursor}, max_bytes=_MAX_CURSOR_BYTES, record_type="maintenance_cursor")
    revision = value.get("progress_revision")
    if type(revision) is not int or revision < 0:
        raise ValueError("maintenance descriptor progress revision is invalid.")
    completed = value.get("completed_phases")
    if type(completed) is not list or not all(type(item) is str for item in completed):
        raise ValueError("maintenance descriptor completion evidence is invalid.")
    if completed != list(PHASES[: len(completed)]) or len(completed) > len(PHASES):
        raise ValueError("maintenance descriptor completion evidence is not a phase prefix.")
    if state == "completed" and completed != list(PHASES):
        raise ValueError("completed maintenance descriptor lacks full phase evidence.")
    if state != "completed" and len(completed) >= len(PHASES):
        raise ValueError("nonterminal maintenance descriptor has terminal phase evidence.")
    evidence = value.get("phase_evidence")
    if type(evidence) is not dict or set(evidence) != set(completed):
        raise ValueError("maintenance descriptor phase evidence is invalid.")
    prior_reasons = value.get("prior_degraded_reasons")
    if type(prior_reasons) is not list or not all(type(reason) is str for reason in prior_reasons):
        raise ValueError("maintenance descriptor prior degradation evidence is invalid.")
    source_capture = value.get("source_capture")
    if (
        type(source_capture) is not dict
        or set(source_capture) != {"identity", "epoch", "sequence"}
        or source_capture.get("identity") != identity["project_id"]
        or (source_capture.get("epoch") is not None and type(source_capture.get("epoch")) is not str)
        or (
            source_capture.get("sequence") is not None
            and (type(source_capture.get("sequence")) is not int or source_capture["sequence"] < 1)
        )
    ):
        raise ValueError("maintenance source capture identity is invalid.")
    source_revision = value.get("source_revision")
    if source_revision is not None and (
        type(source_revision) is not dict
        or set(source_revision) != {"epoch", "sequence"}
        or type(source_revision.get("epoch")) is not str
        or type(source_revision.get("sequence")) is not int
        or source_revision["sequence"] < 0
    ):
        raise ValueError("maintenance source revision is invalid.")
    if source_revision is None:
        if source_capture.get("epoch") is not None or source_capture.get("sequence") is not None:
            raise ValueError("maintenance capture and source revision disagree.")
    elif (
        source_capture.get("epoch") != source_revision["epoch"]
        or source_capture.get("sequence") != source_revision["sequence"]
    ):
        raise ValueError("maintenance capture and source revision disagree.")
    build_identity = value.get("build_identity")
    if build_identity is not None and (
        type(build_identity) is not dict
        or any(type(key) is not str or type(item) is not str for key, item in build_identity.items())
    ):
        raise ValueError("maintenance build identity is invalid.")
    for timestamp in (
        value.get("last_progress_at"),
        value.get("meaningful_progress_at"),
        value.get("created_at"),
        value.get("due_at"),
    ):
        if type(timestamp) is not str or not timestamp:
            raise ValueError("maintenance descriptor timestamp is invalid.")
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("maintenance descriptor timestamp is invalid.") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("maintenance descriptor timestamp must include a timezone.")
    retry_count = value.get("retry_count")
    if type(retry_count) is not int or retry_count < 0:
        raise ValueError("maintenance retry count is invalid.")
    failure = value.get("failure")
    if failure is not None and (
        type(failure) is not dict
        or set(failure) != {"code", "phase", "type"}
        or any(type(failure.get(key)) is not str or not failure[key] for key in ("code", "phase", "type"))
    ):
        raise ValueError("maintenance descriptor failure is invalid.")
    if state == "intervention" and failure is None:
        raise ValueError("intervention descriptor is missing its failure code.")
    return value


def _load_descriptor(cfg: RootConfig, pointer: dict[str, Any]) -> dict[str, Any]:
    identity = pointer["identity"]
    value = _read_bounded_record(_descriptor_path(cfg, identity["work_generation"]), "maintenance_descriptor")
    if value.get("schema_version") == 1 and set(value) == _STAGE1_RECORD_FIELDS:
        value = _migrate_stage1_descriptor(value, cfg=cfg, identity=identity)
        _write_record(_descriptor_path(cfg, identity["work_generation"]), value)
        _publish_pointer(cfg, value)
        pointer["progress_revision"] = value["progress_revision"]
    record = _validate_record(value, cfg=cfg, identity=identity)
    if record["progress_revision"] == pointer["progress_revision"] + 1:
        # Recover the durable record-first half of an interrupted checkpoint.
        _publish_pointer(cfg, record)
        pointer["progress_revision"] = record["progress_revision"]
    elif record["progress_revision"] != pointer["progress_revision"]:
        raise ValueError("maintenance current pointer revision does not match its descriptor.")
    return record


def _write_record(path: Path, record: dict[str, Any]) -> None:
    require_json_size(record, max_bytes=DESCRIPTOR_MAX_BYTES, record_type="maintenance_descriptor")
    atomic_replace(path, record)


def _publish_pointer(cfg: RootConfig, record: dict[str, Any]) -> None:
    identity = record["identity"]
    pointer = {
        "schema_version": DESCRIPTOR_SCHEMA_VERSION,
        "identity": identity,
        "descriptor_file": f"{identity['work_generation']}.json",
        "progress_revision": record["progress_revision"],
    }
    require_json_size(pointer, max_bytes=DESCRIPTOR_MAX_BYTES, record_type="maintenance_current")
    atomic_replace(_current_path(cfg), pointer)


def _read_source_revision(cfg: RootConfig) -> dict[str, Any]:
    from .project_activation import read_project_activation

    checkpoint = read_project_activation(cfg.shared_root)
    if checkpoint is None:
        return {"epoch": None, "sequence": None}
    activation = checkpoint["project_activation"]
    return {"epoch": activation["epoch"], "sequence": activation["sequence"]}


def _source_changed(cfg: RootConfig, record: dict[str, Any]) -> bool:
    """Whether activation publication proves source changes after this capture."""
    from ..runtime.project_activation import activation_pending_path

    try:
        activation_pending_path(cfg.shared_root).lstat()
    except FileNotFoundError:
        pass
    else:
        return True
    captured = record["source_revision"]
    current = _read_source_revision(cfg)
    if current["epoch"] is None:
        return captured is not None
    if captured is None:
        return True
    if current["epoch"] != captured["epoch"]:
        return True
    return current["sequence"] > captured["sequence"]


def _migrate_stage1_descriptor(value: dict[str, Any], *, cfg: RootConfig, identity: dict[str, str]) -> dict[str, Any]:
    """Upgrade the accepted Stage 1 record while preserving uncertain coverage."""
    record = dict(value)
    record["schema_version"] = DESCRIPTOR_SCHEMA_VERSION
    record["source_capture"] = {"identity": identity["project_id"], "epoch": None, "sequence": None}
    record["source_revision"] = None
    record["build_identity"] = None
    record["meaningful_progress_at"] = record["last_progress_at"]
    record["due_at"] = record["created_at"]
    record["retry_count"] = 0
    if record["state"] not in {"completed", "intervention"}:
        record["state"] = "intervention"
        record["failure"] = {
            "code": "legacy_capture_identity_unknown",
            "phase": record["phase"],
            "type": "CompatibilityIntervention",
        }
    return record


def _initialize_descriptor(cfg: RootConfig, record: dict[str, Any], ledger: InvocationLedger) -> bool:
    reservation = ledger.reserve_operations(
        maximum_operations=PHASE_OPERATION_RESERVATIONS["descriptor_initialization"]
    )
    if reservation is None:
        return False
    semantic_before = ledger.semantic_items_consumed
    try:
        ledger.consume_semantic_item()
        identity = record["identity"]
        work = prepare_work(
            cfg,
            kind="full_audit",
            target_id="project",
            work_generation=identity["work_generation"],
            phase=record["phase"],
            cursor=record["cursor"],
        )
        activation_reason = f"mw:full_audit:project:{identity['work_generation']}:{work['progress_revision'] + 1}"
        from .project_activation import project_activation_transaction

        with project_activation_transaction(cfg, activation_reason) as activation_record:
            activation = activation_record["project_activation"]
            source_revision = {"epoch": activation["epoch"], "sequence": activation["sequence"]}
            record["source_capture"] = {
                "identity": record["identity"]["project_id"],
                **source_revision,
            }
            record["source_revision"] = source_revision
            update_work(
                cfg,
                kind="full_audit",
                target_id="project",
                work_generation=identity["work_generation"],
                state="pending",
                phase=record["phase"],
                cursor=record["cursor"],
                publish_activation=False,
            )
            _write_record(_descriptor_path(cfg, record["identity"]["work_generation"]), record)
            _publish_pointer(cfg, record)
        return True
    finally:
        if reservation._active:
            if ledger.semantic_items_consumed > semantic_before:
                reservation.commit(actual_operations=PHASE_OPERATION_RESERVATIONS["descriptor_initialization"])
            else:
                reservation.release()


def _persist_progress(cfg: RootConfig, record: dict[str, Any], *, meaningful_progress: bool) -> None:
    pointer = _read_pointer(cfg)
    if pointer is None or pointer["identity"] != record["identity"]:
        raise ValueError("maintenance target was superseded.")
    current = _load_descriptor(cfg, pointer)
    if current["progress_revision"] != record["progress_revision"]:
        raise ValueError("maintenance progress was superseded.")
    record["progress_revision"] += 1
    now = utc_now()
    record["last_progress_at"] = now
    if meaningful_progress:
        record["meaningful_progress_at"] = now
        if record["state"] not in {"waiting", "intervention"}:
            record["retry_count"] = 0
            record["due_at"] = now
            record["failure"] = None
    _write_record(_descriptor_path(cfg, record["identity"]["work_generation"]), record)
    _publish_pointer(cfg, record)
    work = read_work(
        cfg,
        kind="full_audit",
        target_id="project",
        work_generation=record["identity"]["work_generation"],
    )
    if work is None:
        work = prepare_work(
            cfg,
            kind="full_audit",
            target_id="project",
            work_generation=record["identity"]["work_generation"],
            phase=record["phase"],
            cursor=record["cursor"],
        )
    if record["state"] == "completed":
        retire_work(
            cfg,
            kind="full_audit",
            target_id="project",
            work_generation=record["identity"]["work_generation"],
            proof={
                "source": "full_audit_completion_fence",
                "capture_id": record["capture_id"],
                "source_revision": record["source_revision"],
            },
        )
    elif record["state"] == "intervention":
        retire_work(
            cfg,
            kind="full_audit",
            target_id="project",
            work_generation=record["identity"]["work_generation"],
            state="intervention",
            proof={"failure": record.get("failure")},
        )
    else:
        update_work(
            cfg,
            kind="full_audit",
            target_id="project",
            work_generation=record["identity"]["work_generation"],
            state="waiting" if record["state"] == "waiting" else "running",
            phase=record["phase"],
            cursor=record["cursor"],
            due_at=record["due_at"],
            retry_count=record["retry_count"],
            failure=record.get("failure"),
            meaningful_progress=meaningful_progress,
            publish_activation=False,
        )


def _retry_due(record: dict[str, Any], *, now: datetime | None = None) -> bool:
    due_at = datetime.fromisoformat(record["due_at"].replace("Z", "+00:00"))
    return due_at <= (now or datetime.now(timezone.utc))


def _schedule_retry(record: dict[str, Any], failure: dict[str, str]) -> None:
    retry_count = record["retry_count"] + 1
    base_delay = min(60.0, float(2 ** min(retry_count - 1, 6)))
    delay = random.SystemRandom().uniform(0.8 * base_delay, base_delay)
    record["retry_count"] = retry_count
    record["state"] = "waiting"
    record["failure"] = failure
    record["due_at"] = (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()


def _is_transient_failure(exception: BaseException) -> bool:
    if isinstance(exception, (BlockingIOError, TimeoutError)):
        return True
    return isinstance(exception, OSError) and exception.errno in {
        errno.EAGAIN,
        errno.EWOULDBLOCK,
        errno.EBUSY,
        errno.ETIMEDOUT,
        errno.ESTALE,
    }


def _new_descriptor(cfg: RootConfig) -> dict[str, Any]:
    generation = new_id()
    identity = _expected_identity(cfg, generation)
    now = utc_now()
    source_revision = _read_source_revision(cfg)
    try:
        from ..runtime.ready import read_ready_index_status

        prior_degraded_reasons = list(read_ready_index_status(cfg).get("degraded_reasons", []))
    except (AttributeError, FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        prior_degraded_reasons = []
    return {
        "schema_version": DESCRIPTOR_SCHEMA_VERSION,
        "identity": identity,
        "capture_id": generation,
        "source_capture": {
            "identity": identity["project_id"],
            "epoch": source_revision["epoch"],
            "sequence": source_revision["sequence"],
        },
        "source_revision": source_revision if source_revision["epoch"] is not None else None,
        "build_identity": None,
        "state": "pending",
        "phase": PHASES[0],
        "cursor": {"offset": 0},
        "progress_revision": 0,
        "completed_phases": [],
        "phase_evidence": {},
        "prior_degraded_reasons": prior_degraded_reasons,
        "last_progress_at": now,
        "meaningful_progress_at": now,
        "created_at": now,
        "due_at": now,
        "retry_count": 0,
        "failure": None,
    }


def _new_operation_cursor(kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "lane": "active",
        "cycle": 0,
        "active_offset": 0,
        "legacy_offset": 0,
    }


def _advance_operation_phase(
    cfg: RootConfig,
    record: dict[str, Any],
    phase: str,
    *,
    reservation_runtime_root: Path | None,
) -> tuple[
    dict[str, Any],
    bool,
    list[str],
    list[str],
    dict[str, str] | None,
    dict[str, Any] | None,
]:
    kind = {"cleanup": "cleanup", "availability": "availability", "group_cancel": "group_control"}[phase]
    phase_cursor = dict(record["cursor"])
    before = phase_cursor.get("operation_cursor")
    if type(before) is not dict or before.get("kind") != kind:
        before = _new_operation_cursor(kind)
    else:
        before = dict(before)
    operation_cursor = dict(before)
    repaired: list[str] = []
    blocked: list[str] = []
    phase_result: dict[str, Any] | None = None
    if phase == "cleanup":
        from ..commands.cleanup import advance_cleanup_maintenance_step
        from ..runtime.operation_store import iter_active_operation_paths

        pending_id = phase_cursor.get("pending_operation_id")
        pending_task_id = phase_cursor.get("pending_task_id")
        child_cursor = phase_cursor.get("cleanup_child")
        if isinstance(pending_id, str):
            if not isinstance(pending_task_id, str):
                blocked.append(pending_id)
                return (
                    phase_cursor,
                    False,
                    repaired,
                    blocked,
                    {
                        "code": "cleanup_pending_task_identity_missing",
                        "phase": phase,
                        "type": "Intervention",
                    },
                    None,
                )
            outcome = advance_cleanup_maintenance_step(
                cfg,
                pending_task_id,
                pending_id,
                child_cursor if isinstance(child_cursor, dict) else {},
                reservation_runtime_root=reservation_runtime_root,
            )
            if outcome.get("state") == "intervention":
                blocked.append(pending_id)
                return (
                    phase_cursor,
                    False,
                    repaired,
                    blocked,
                    {
                        "code": str(outcome.get("reason") or "cleanup_operation_incomplete"),
                        "phase": phase,
                        "type": "Intervention",
                    },
                    None,
                )
            if outcome.get("state") == "completed":
                repaired.append(pending_id)
                phase_cursor.pop("pending_operation_id", None)
                phase_cursor.pop("pending_task_id", None)
                phase_cursor.pop("cleanup_child", None)
            else:
                phase_cursor["cleanup_child"] = outcome.get("cursor", child_cursor or {})
                phase_result = {"state": "waiting"} if outcome.get("state") == "waiting" else None
            return phase_cursor, False, repaired, blocked, None, phase_result

        operation_path = next(
            iter(
                iter_active_operation_paths(
                    cfg,
                    "cleanup",
                    limit=1,
                    include_legacy=True,
                    cursor=operation_cursor,
                )
            ),
            None,
        )
        if operation_path is not None:
            operation = read_json_limited(
                operation_path,
                max_bytes=_MAX_SOURCE_RECORD_BYTES,
                record_type="maintenance_cleanup",
            )
            cleanup = operation.get("cleanup", {})
            if cleanup.get("state") in {"preparing", "waiting_ack"}:
                operation_id = cleanup.get("operation_id")
                task_id = cleanup.get("task_id")
                if not isinstance(operation_id, str) or not isinstance(task_id, str):
                    blocked.append(operation_path.stem)
                    return (
                        phase_cursor,
                        False,
                        repaired,
                        blocked,
                        {
                            "code": "cleanup_operation_identity_invalid",
                            "phase": phase,
                            "type": "Intervention",
                        },
                        None,
                    )
                phase_cursor["pending_operation_id"] = operation_id
                phase_cursor["pending_task_id"] = task_id
                phase_cursor["cleanup_child"] = {}
        after = operation_cursor
        phase_cursor["operation_cursor"] = after
        phase_done = (
            operation_path is None
            and type(before.get("cycle")) is int
            and type(after.get("cycle")) is int
            and after["cycle"] > before["cycle"]
        )
        return phase_cursor, phase_done, repaired, blocked, None, None
    elif phase == "availability":
        results = reconcile_availability_operations(cfg, include_legacy=True, limit=1, cursor=operation_cursor)
        for result in results:
            operation_id = result.get("operation_id")
            if not isinstance(operation_id, str):
                continue
            if result.get("state") == "blocked":
                blocked.append(operation_id)
                return (
                    record["cursor"],
                    False,
                    repaired,
                    blocked,
                    {
                        "code": "availability_operation_incomplete",
                        "phase": phase,
                        "type": "Intervention",
                    },
                    None,
                )
            repaired.append(operation_id)
    else:
        from ..commands.group import advance_legacy_cancel_step
        from ..commands.group_cancel import advance_indexed_cancel
        from ..runtime.group_namespace import is_group_authority_isolated
        from ..runtime.operation_store import iter_active_operation_paths

        pending_id = phase_cursor.get("pending_operation_id")
        child_cursor = phase_cursor.get("group_cancel_child")
        if isinstance(pending_id, str):
            path = locate_operation_path(cfg, "group_control", pending_id)
            if not path.exists():
                outcome = {"state": "intervention", "reason": "group_cancel_operation_missing"}
            else:
                operation = read_json_limited(
                    path,
                    max_bytes=_MAX_SOURCE_RECORD_BYTES,
                    record_type="maintenance_group_cancel",
                )
                control = operation.get("group_control", {})
                if control.get("state") == "blocked":
                    outcome = {
                        "state": "intervention",
                        "reason": control.get("blocked_reason") or "group_cancel_blocked",
                    }
                elif is_group_authority_isolated(cfg.shared_root):
                    outcome = advance_indexed_cancel(cfg, path, reservation_runtime_root=reservation_runtime_root) or {
                        "state": "completed",
                        "operation_id": pending_id,
                    }
                    outcome = {**outcome, "cursor": child_cursor or {}}
                else:
                    outcome = advance_legacy_cancel_step(
                        cfg,
                        pending_id,
                        child_cursor if isinstance(child_cursor, dict) else {},
                        reservation_runtime_root=reservation_runtime_root,
                    )
            if outcome.get("state") == "intervention":
                blocked.append(pending_id)
                return (
                    phase_cursor,
                    False,
                    repaired,
                    blocked,
                    {
                        "code": str(outcome.get("reason") or "group_cancel_operation_incomplete"),
                        "phase": phase,
                        "type": "Intervention",
                    },
                    None,
                )
            if outcome.get("state") == "completed":
                repaired.append(pending_id)
                phase_cursor.pop("pending_operation_id", None)
                phase_cursor.pop("group_cancel_child", None)
            else:
                phase_cursor["group_cancel_child"] = outcome.get("cursor", child_cursor or {})
                phase_result = {"state": "waiting"} if outcome.get("state") == "waiting" else None
            return phase_cursor, False, repaired, blocked, None, phase_result

        operation_path = next(
            iter(
                iter_active_operation_paths(
                    cfg,
                    "group_control",
                    limit=1,
                    include_legacy=True,
                    cursor=operation_cursor,
                )
            ),
            None,
        )
        if operation_path is not None:
            operation = read_json_limited(
                operation_path,
                max_bytes=_MAX_SOURCE_RECORD_BYTES,
                record_type="maintenance_group_cancel",
            )
            control = operation.get("group_control", {})
            if (
                control.get("operation_type") == "cancel"
                and control.get("state") in {"preparing", "converging", "waiting_ack", "blocked"}
                and isinstance(control.get("operation_id"), str)
            ):
                phase_cursor["pending_operation_id"] = control["operation_id"]
                phase_cursor["group_cancel_child"] = {}
        after = operation_cursor
        phase_cursor["operation_cursor"] = after
        phase_done = (
            operation_path is None
            and type(before.get("cycle")) is int
            and type(after.get("cycle")) is int
            and after["cycle"] > before["cycle"]
        )
        return phase_cursor, phase_done, repaired, blocked, None, None

    after = operation_cursor
    phase_cursor["operation_cursor"] = after
    phase_done = (
        type(before.get("cycle")) is int and type(after.get("cycle")) is int and after["cycle"] > before["cycle"]
    )
    return phase_cursor, phase_done, repaired, blocked, None, phase_result


def _advance_submission_phase(
    cfg: RootConfig,
    record: dict[str, Any],
) -> tuple[dict[str, Any], bool, list[str], list[str], dict[str, str] | None]:
    cursor = record["cursor"]
    pending_id = cursor.get("pending_submission_id")
    if isinstance(pending_id, str):
        from ..runtime.submission import advance_submission_cleanup_step

        try:
            outcome = advance_submission_cleanup_step(cfg, pending_id)
        except (KeyError, RuntimeError, TypeError, ValueError):
            return (
                cursor,
                False,
                [],
                [pending_id],
                {
                    "code": "submission_cleanup_invalid",
                    "phase": "submission",
                    "type": "Intervention",
                },
            )
        if outcome.get("state") == "intervention":
            return (
                cursor,
                False,
                [],
                [pending_id],
                {
                    "code": str(outcome.get("reason") or "submission_cleanup_intervention"),
                    "phase": "submission",
                    "type": "Intervention",
                },
            )
        if outcome.get("state") == "completed":
            return (
                {key: value for key, value in cursor.items() if key != "pending_submission_id"},
                False,
                [pending_id],
                [],
                None,
            )
        return cursor, False, [], [], None
    offset = cursor.get("offset", 0)
    if type(offset) is not int or offset < 0:
        raise ValueError("Submission audit cursor is invalid.")
    directory = shared_paths(cfg.shared_root)["submissions"]
    name, next_offset = read_directory_entry(directory, offset)
    if name is None:
        return {"offset": next_offset}, True, [], [], None
    next_cursor = {"offset": next_offset}
    if not name.endswith(".json"):
        return next_cursor, False, [], [], None
    path = directory / name
    operation = read_json_limited(
        path,
        max_bytes=_MAX_SOURCE_RECORD_BYTES,
        record_type="maintenance_submission",
    ).get("submission")
    if type(operation) is not dict:
        raise ValueError("Submission operation record is malformed.")
    operation_id = operation.get("operation_id") or path.stem
    validate_identifier(operation_id, "submission operation_id")
    state = operation.get("state")
    if state == "blocked":
        return (
            next_cursor,
            False,
            [],
            [operation_id],
            {
                "code": "submission_authority_blocked",
                "phase": "submission",
                "type": "Intervention",
            },
        )
    if state == "committed":
        # Committed Submission recovery only replays the bounded Group
        # finalizer; it never decodes or enumerates the immutable Task plan.
        from ..runtime.submission import _finalize_committed_submission

        _finalize_committed_submission(cfg, operation)
        return next_cursor, False, [operation_id], [], None
    if state in {"preparing", "committing", "aborted"}:
        # Child cleanup starts on the next invocation. The operation identity
        # and source cursor make that continuation independent of the local
        # runtime root and bound each following turn to one Group or Task child.
        return {**next_cursor, "pending_submission_id": operation_id}, False, [], [], None
    return (
        next_cursor,
        False,
        [],
        [operation_id],
        {"code": "submission_state_invalid", "phase": "submission", "type": "Intervention"},
    )


def _advance_orphan_phase(
    cfg: RootConfig,
    record: dict[str, Any],
    reservation_runtime_root: Path | None,
) -> tuple[dict[str, Any], bool, list[str], list[str], dict[str, str] | None]:
    cursor = dict(record["cursor"])
    pending = cursor.get("orphan")
    if isinstance(pending, dict):
        stage = pending.get("stage")
        task_id = pending.get("task_id")
        number = pending.get("attempt_number")
        if not isinstance(task_id, str) or type(number) is not int or number < 0:
            raise ValueError("orphan child cursor is invalid.")
        attempt_file = shared_paths(cfg.shared_root)["attempts"] / task_id / f"{number}.json"
        if stage == "attempt":
            if not attempt_file.exists():
                return (
                    cursor,
                    False,
                    [],
                    [task_id],
                    {
                        "code": "attempt_truth_missing",
                        "phase": "orphan_recovery",
                        "type": "Intervention",
                    },
                )
            attempt = AttemptRecord.from_dict(
                read_json_limited(
                    attempt_file,
                    max_bytes=_MAX_SOURCE_RECORD_BYTES,
                    record_type="maintenance_attempt",
                )
            )
            if attempt.task_id != task_id or attempt.attempt_number != number:
                raise ValueError("orphan Attempt identity does not match its Task.")
            if attempt.machine_name != cfg.machine_name:
                return (
                    cursor,
                    False,
                    [],
                    [task_id],
                    {
                        "code": "orphan_attempt_owned_remotely",
                        "phase": "orphan_recovery",
                        "type": "Intervention",
                    },
                )
            pending.update(
                {
                    "stage": "process",
                    "attempt_id": attempt.attempt_id,
                    "fencing_token": attempt.current_fencing_token,
                    "attempt_process_identity": {
                        key: attempt.process.get(key) for key in ("process_group_id", "process_group_start_time_ticks")
                    },
                }
            )
            return cursor, False, [], [], None
        if stage == "process":
            attempt_id = pending.get("attempt_id")
            fencing_token = pending.get("fencing_token")
            if not isinstance(attempt_id, str) or type(fencing_token) is not int:
                raise ValueError("orphan Attempt continuation is malformed.")
            manifest_path = cfg.runtime_root / "processes" / f"{attempt_id}.json"
            if not manifest_path.exists():
                return (
                    cursor,
                    False,
                    [],
                    [task_id],
                    {
                        "code": "orphan_process_evidence_missing",
                        "phase": "orphan_recovery",
                        "type": "Intervention",
                    },
                )
            process = read_json_limited(
                manifest_path,
                max_bytes=_MAX_SOURCE_RECORD_BYTES,
                record_type="maintenance_process",
            )
            process_record = process.get("process")
            if (
                not isinstance(process_record, dict)
                or process_record.get("task_id") != task_id
                or process_record.get("attempt_id") != attempt_id
            ):
                return (
                    cursor,
                    False,
                    [],
                    [task_id],
                    {
                        "code": "orphan_process_identity_mismatch",
                        "phase": "orphan_recovery",
                        "type": "Intervention",
                    },
                )
            from ..scheduler import finalize_orphaned_attempt
            from .attempt_recovery import recover_running_attempt
            from .process_evidence import inspect_group_identity

            evidence = inspect_group_identity(pending["attempt_process_identity"], process_record)
            if evidence.state == "alive":
                token = recover_running_attempt(
                    cfg,
                    task_id,
                    attempt_id,
                    fencing_token,
                    manifest=process_record,
                    reservation_runtime_root=reservation_runtime_root,
                )
                if token is None:
                    return (
                        cursor,
                        False,
                        [],
                        [task_id],
                        {
                            "code": "orphan_recovery_cas_rejected",
                            "phase": "orphan_recovery",
                            "type": "Intervention",
                        },
                    )
            elif evidence.state == "absent":
                if not finalize_orphaned_attempt(
                    cfg,
                    task_id,
                    attempt_id,
                    fencing_token,
                    exit_code=process_record.get("exit_code"),
                    was_terminated=bool(pending.get("was_terminated")),
                    reservation_runtime_root=reservation_runtime_root,
                ):
                    return (
                        cursor,
                        False,
                        [],
                        [task_id],
                        {
                            "code": "orphan_finalize_cas_rejected",
                            "phase": "orphan_recovery",
                            "type": "Intervention",
                        },
                    )
                process_record["observed_state"] = "exited"
                process_record["reconciled_at"] = utc_now()
                atomic_replace(manifest_path, {"process": process_record})
            else:
                return (
                    cursor,
                    False,
                    [],
                    [task_id],
                    {
                        "code": "orphan_process_identity_unverifiable",
                        "phase": "orphan_recovery",
                        "type": "Intervention",
                    },
                )
            del cursor["orphan"]
            return cursor, False, [task_id], [], None
        raise ValueError("orphan child stage is invalid.")

    offset = cursor.get("offset", 0)
    if type(offset) is not int or offset < 0:
        raise ValueError("orphan-recovery cursor is invalid.")
    directory = shared_paths(cfg.shared_root)["tasks"]
    name, next_offset = read_directory_entry(directory, offset)
    if name is None:
        return {"offset": next_offset}, True, [], [], None
    next_cursor = {"offset": next_offset}
    if not name.endswith(".json"):
        return next_cursor, False, [], [], None
    task = TaskRecord.from_dict(
        read_json_limited(
            directory / name,
            max_bytes=_MAX_SOURCE_RECORD_BYTES,
            record_type="maintenance_orphan_task",
        )
    )
    if task.state["projection"] != "blocked":
        return next_cursor, False, [], [], None
    number = task.attempt_control.get("current_attempt_number")
    if type(number) is not int or number < 0:
        return (
            next_cursor,
            False,
            [],
            [task.task_id],
            {
                "code": "orphan_current_attempt_missing",
                "phase": "orphan_recovery",
                "type": "Intervention",
            },
        )
    next_cursor["orphan"] = {
        "stage": "attempt",
        "task_id": task.task_id,
        "attempt_number": number,
        "was_terminated": bool(task.control.get("terminate_running")),
    }
    return next_cursor, False, [], [], None


def _advance_ready_phase(
    cfg: RootConfig, record: dict[str, Any]
) -> tuple[dict[str, Any], bool, list[str], list[str], dict[str, str] | None]:
    from ..doctor import mark_ready_index_degraded, parse_ready_reason, ready_task_projection_issue
    from .ready import read_ready_index_state, read_ready_index_status, repair_ready_index

    cursor = record["cursor"]
    mode = cursor.get("mode", "audit")
    if mode == "audit":
        offset = cursor.get("offset", 0)
        if type(offset) is not int or offset < 0:
            raise ValueError("ready-index audit cursor is invalid.")
        directory = shared_paths(cfg.shared_root)["tasks"]
        name, next_offset = read_directory_entry(directory, offset)
        if name is None:
            if read_ready_index_state(cfg) == "active":
                return {}, True, [], [], None
            return {"mode": "build"}, False, [], [], None
        next_cursor = {"mode": "audit", "offset": next_offset}
        if not name.endswith(".json"):
            return next_cursor, False, [], [], None
        task = TaskRecord.from_dict(
            read_json_limited(directory / name, max_bytes=_MAX_SOURCE_RECORD_BYTES, record_type="ready_audit_task")
        )
        if read_ready_index_state(cfg) != "active":
            return {"mode": "build"}, False, [], [], None
        issue = ready_task_projection_issue(cfg, task.task_id)
        if issue is None:
            return next_cursor, False, [], [], None
        try:
            diagnostic = parse_ready_reason(issue).diagnostic
        except ValueError:
            diagnostic = None
        if diagnostic is None:
            return (
                next_cursor,
                False,
                [],
                ["ready_index"],
                {
                    "code": "ready_index_audit_unverifiable",
                    "phase": "ready_index",
                    "type": "ProjectionEvidence",
                },
            )
        if issue not in record["prior_degraded_reasons"]:
            record["prior_degraded_reasons"].append(issue)
        mark_ready_index_degraded(cfg, diagnostic)
        return {"mode": "build"}, False, [], [], None

    if mode != "build":
        raise ValueError("ready-index phase cursor is invalid.")
    prior_state = read_ready_index_state(cfg)
    try:
        ready_record = repair_ready_index(cfg, max_tasks=1, bounded_initialization=True)
    except (AttributeError, FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return (
            cursor,
            False,
            [],
            ["ready_index"],
            {
                "code": "ready_index_rebuild_failed",
                "phase": "ready_index",
                "type": "ProjectionFailure",
            },
        )
    if ready_record.get("state") == "degraded":
        return (
            cursor,
            False,
            [],
            ["ready_index"],
            {
                "code": "ready_index_rebuild_failed",
                "phase": "ready_index",
                "type": "ProjectionFailure",
            },
        )
    if ready_record.get("state") == "active":
        build = ready_record.get("build") or {}
        if isinstance(build, dict) and isinstance(build.get("build_id"), str):
            record["build_identity"] = {"kind": "ready_index", "build_id": build["build_id"]}
        repaired = []
        if prior_state != "active":
            repaired.append(f"ready_index:{build.get('repaired', 0)}:{build.get('stale_removed', 0)}")
        return {}, True, repaired, [], None
    build = ready_record.get("build")
    if isinstance(build, dict) and isinstance(build.get("build_id"), str):
        record["build_identity"] = {"kind": "ready_index", "build_id": build["build_id"]}
    return cursor, False, [], [], None


def _advance_group_members_phase(
    cfg: RootConfig, descriptor: dict[str, Any]
) -> tuple[dict[str, Any], bool, list[str], list[str], dict[str, str] | None]:
    from ..runtime.ready.group_members import group_ready_members_state
    from ..runtime.ready.group_members_rebuild import repair_group_ready_members

    initial_state = group_ready_members_state(cfg)
    projection_record = repair_group_ready_members(cfg, max_work_items=1, bounded_initialization=True)
    state = projection_record.get("state")
    audit = projection_record.get("audit") if isinstance(projection_record.get("audit"), dict) else {}
    build = projection_record.get("build") if isinstance(projection_record.get("build"), dict) else {}
    identity = {"kind": "group_ready_members"}
    for key in ("build_id", "projection_id"):
        value = build.get(key) if key == "build_id" else projection_record.get(key)
        if isinstance(value, str):
            identity[key] = value
    if isinstance(audit.get("audit_id"), str):
        identity["audit_id"] = audit["audit_id"]
    descriptor["build_identity"] = identity if len(identity) > 1 else None
    if state == "degraded":
        return (
            {},
            False,
            [],
            ["group_ready_members"],
            {
                "code": "group_ready_members_rebuild_failed",
                "phase": "group_ready_members",
                "type": "ProjectionFailure",
            },
        )
    if state == "legacy" or (state == "active" and audit.get("state") == "completed"):
        repaired = ["group_ready_members"] if initial_state != "active" and state == "active" else []
        return {}, True, repaired, [], None
    return {}, False, [], [], None


def _advance_observation_phase(
    cfg: RootConfig, record: dict[str, Any]
) -> tuple[dict[str, Any], bool, dict[str, Any], dict[str, str] | None]:
    cursor = dict(record["cursor"])
    observation = inspect_observation(cfg)
    if observation.get("state") in {"degraded", "unavailable"} and not cursor.get("requested"):
        requested = request_rebuild(cfg)
        cursor["requested"] = requested.get("state") != "waiting"
        if requested.get("state") == "waiting":
            return cursor, False, requested, None
        return cursor, False, requested, None
    if observation.get("state") == "active" and observation.get("dirty") is False:
        return {}, True, observation, None
    maintenance = ObservationMaintenance(cfg)
    try:
        updated = maintenance.advance()
    finally:
        maintenance.close()
    if updated.get("state") == "degraded":
        return (
            cursor,
            False,
            updated,
            {
                "code": "task_observation_rebuild_failed",
                "phase": "task_observation",
                "type": "ProjectionFailure",
            },
        )
    return cursor, False, updated, None


def _advance_submission_control_phase(
    cfg: RootConfig, record: dict[str, Any]
) -> tuple[dict[str, Any], bool, dict[str, Any], dict[str, str] | None]:
    from ..runtime.submission_control_maintenance import SubmissionControlMaintenance

    cursor = dict(record["cursor"])
    control = inspect_submission_control(cfg)
    if control.get("state") == "active" and not cursor.get("requested"):
        return {}, True, control, None
    if control.get("state") == "unavailable" and not cursor.get("requested"):
        updated = request_control_rebuild(cfg)
        cursor["requested"] = updated.get("state") != "waiting"
        return cursor, False, updated, None
    maintenance = SubmissionControlMaintenance(cfg)
    try:
        updated = maintenance.advance()
    finally:
        maintenance.close()
    if updated.get("state") == "active":
        return {}, True, updated, None
    return cursor, False, updated, None


def _advance_phase(
    cfg: RootConfig,
    record: dict[str, Any],
    *,
    reservation_runtime_root: Path | None,
) -> tuple[dict[str, Any], bool, list[str], list[str], dict[str, str] | None, dict[str, Any] | None]:
    phase = record["phase"]
    if phase == "submission":
        cursor, done, repaired, blocked, failure = _advance_submission_phase(cfg, record)
        return cursor, done, repaired, blocked, failure, None
    if phase in {"cleanup", "availability", "group_cancel"}:
        cursor, done, repaired, blocked, failure, phase_result = _advance_operation_phase(
            cfg, record, phase, reservation_runtime_root=reservation_runtime_root
        )
        return cursor, done, repaired, blocked, failure, phase_result
    if phase == "deadline_index":
        outcome = advance_deadline_index_rebuild_step(cfg, cursor=record["cursor"] or None)
        repaired = []
        if outcome.get("rebuilt"):
            repaired.append(f"offer_deadline_indexes:{outcome['rebuilt']}")
        return outcome["cursor"], outcome.get("state") == "completed", repaired, [], None, None
    if phase == "orphan_recovery":
        cursor, done, repaired, blocked, failure = _advance_orphan_phase(cfg, record, reservation_runtime_root)
        return cursor, done, repaired, blocked, failure, None
    if phase == "ready_index":
        cursor, done, repaired, blocked, failure = _advance_ready_phase(cfg, record)
        return cursor, done, repaired, blocked, failure, None
    if phase == "group_ready_members":
        cursor, done, repaired, blocked, failure = _advance_group_members_phase(cfg, record)
        return cursor, done, repaired, blocked, failure, None
    if phase == "task_observation":
        cursor, done, result, failure = _advance_observation_phase(cfg, record)
        return cursor, done, [], [], failure, result
    if phase == "submission_control":
        cursor, done, result, failure = _advance_submission_control_phase(cfg, record)
        return cursor, done, [], [], failure, result
    if phase == "final_verification":
        completed = record.get("completed_phases")
        if completed != list(PHASES[:-1]) or record.get("failure") is not None:
            raise ValueError("full-audit completion evidence is incomplete.")
        return {}, True, [], [], None, {"complete": True}
    raise ValueError("maintenance phase is invalid.")


def _advance_phase_name(record: dict[str, Any], *, done: bool) -> None:
    if not done:
        return
    phase = record["phase"]
    index = PHASES.index(phase)
    record["completed_phases"].append(phase)
    record["phase_evidence"][phase] = {"completed_at": utc_now()}
    if phase == "final_verification":
        record["state"] = "completed"
        record["cursor"] = {}
        return
    record["phase"] = PHASES[index + 1]
    record["cursor"] = {}


def _work_operation_path(cfg: RootConfig, descriptor: dict[str, Any]) -> Path | None:
    identity = descriptor["identity"]
    kind = identity["kind"]
    operation_key = descriptor["cursor"].get("operation_key")
    if kind == "submission":
        return submission_path(cfg.shared_root, identity["target_id"])
    if kind not in {"cleanup", "availability", "group_cancel"}:
        return None
    if not isinstance(operation_key, str):
        return None
    operation_kind = "group_control" if kind == "group_cancel" else kind
    return locate_operation_path(cfg, operation_kind, operation_key)


def _operation_section(kind: str) -> str:
    return "availability_operation" if kind == "availability" else ("group_control" if kind == "group_cancel" else kind)


def _advance_operation_work(
    cfg: RootConfig,
    descriptor: dict[str, Any],
    *,
    reservation_runtime_root: Path | None,
) -> dict[str, Any]:
    identity = descriptor["identity"]
    kind = identity["kind"]
    operation_id = identity["target_id"]
    path = _work_operation_path(cfg, descriptor)
    if path is None:
        return {"state": "intervention", "reason": "maintenance_operation_key_missing"}
    try:
        operation = read_json_limited(
            path,
            max_bytes=_MAX_SOURCE_RECORD_BYTES,
            record_type=f"maintenance_{kind}_operation",
        )
    except FileNotFoundError:
        return {"state": "intervention", "reason": "maintenance_operation_missing_after_prepare"}
    section = operation.get(_operation_section(kind)) if type(operation) is dict else None
    if type(section) is not dict or section.get("operation_id") != operation_id:
        return {"state": "intervention", "reason": "maintenance_operation_identity_mismatch"}
    state = section.get("state")
    if state in {"completed", "superseded"}:
        return {"state": "completed", "proof": {"operation_state": state, "operation_id": operation_id}}
    if state == "blocked":
        return {
            "state": "intervention",
            "reason": section.get("blocked_reason") or "maintenance_operation_blocked",
        }

    cursor = descriptor["cursor"]
    if kind == "cleanup":
        from ..commands.cleanup import advance_cleanup_maintenance_step

        task_id = section.get("task_id")
        if not isinstance(task_id, str):
            return {"state": "intervention", "reason": "cleanup_operation_task_identity_missing"}
        outcome = advance_cleanup_maintenance_step(
            cfg,
            task_id,
            operation_id,
            cursor.get("child") if isinstance(cursor.get("child"), dict) else {},
            reservation_runtime_root=reservation_runtime_root,
        )
        if outcome.get("state") == "completed":
            return {"state": "completed", "proof": {"operation_id": operation_id, "source": "cleanup_step"}}
        if outcome.get("state") == "intervention":
            return {"state": "intervention", "reason": outcome.get("reason") or "cleanup_intervention"}
        next_cursor = dict(cursor)
        next_cursor["child"] = outcome.get("cursor", next_cursor.get("child", {}))
        return {
            "state": "waiting" if outcome.get("state") == "waiting" else "running",
            "cursor": next_cursor,
            "reason": outcome.get("reason"),
            "meaningful_progress": next_cursor != cursor,
        }

    if kind == "availability":
        from ..runtime.availability.transitions import AvailabilityTransitionRequest, apply_availability_transition

        outcome = apply_availability_transition(
            cfg,
            AvailabilityTransitionRequest(
                action=section.get("operation_type"),
                task_id=section.get("task_id"),
                helper_machines=section.get("helper_machines"),
                after_seconds=section.get("after_seconds"),
                reason=section.get("reason") or "manual",
                operation_id=operation_id,
            ),
        )
        return {"state": "completed", "proof": {"operation_id": operation_id, "source": "availability_transition"}}

    if kind == "submission":
        from ..runtime.submission import advance_submission_cleanup_step

        outcome = advance_submission_cleanup_step(cfg, operation_id)
        if outcome.get("state") == "completed":
            return {"state": "completed", "proof": {"operation_id": operation_id, "source": "submission_finalizer"}}
        if outcome.get("state") == "intervention":
            return {"state": "intervention", "reason": outcome.get("reason") or "submission_intervention"}
        return {
            "state": "running",
            "reason": outcome.get("stage"),
            "meaningful_progress": True,
        }

    operation_type = section.get("operation_type")
    if operation_type == "worker_remove_v2":
        from ..commands.worker_removal import reconcile_worker_removal

        outcome = reconcile_worker_removal(cfg, path, None, reservation_runtime_root=reservation_runtime_root)
        if outcome is None:
            return {"state": "waiting", "reason": "worker_removal_lock_or_state"}
        if outcome.get("state") in {"completed", "superseded"}:
            return {"state": "completed", "proof": {"operation_id": operation_id, "source": "worker_removal"}}
        if outcome.get("state") == "blocked":
            return {"state": "intervention", "reason": outcome.get("blocked_reason") or "worker_removal_blocked"}
        return {"state": "running", "reason": "worker_removal_progress", "meaningful_progress": True}
    if operation_type != "cancel":
        return {"state": "intervention", "reason": "legacy_worker_removal_requires_scoped_reconstruction"}
    if state not in {"preparing", "converging", "waiting_ack"}:
        return {"state": "intervention", "reason": "group_cancel_operation_state_unknown"}
    from ..runtime.group_namespace import is_group_authority_isolated

    if is_group_authority_isolated(cfg.shared_root):
        from ..commands.group_cancel import advance_indexed_cancel

        outcome = advance_indexed_cancel(cfg, path, reservation_runtime_root=reservation_runtime_root)
        if outcome is None:
            current = locate_operation_path(cfg, "group_control", operation_id)
            if not current.exists():
                return {"state": "intervention", "reason": "group_cancel_operation_missing_after_prepare"}
            outcome = {"state": "waiting", "reason": "group_cancel_no_progress"}
        if outcome.get("state") in {"completed", "superseded"}:
            return {"state": "completed", "proof": {"operation_id": operation_id, "source": "indexed_cancel"}}
        if outcome.get("state") == "blocked":
            return {"state": "intervention", "reason": outcome.get("blocked_reason") or "group_cancel_blocked"}
        return {"state": "running", "reason": "group_cancel_progress", "meaningful_progress": True}
    from ..commands.group import advance_legacy_cancel_step

    outcome = advance_legacy_cancel_step(
        cfg,
        operation_id,
        cursor.get("child", {}),
        reservation_runtime_root=reservation_runtime_root,
    )
    if outcome.get("state") == "completed":
        return {"state": "completed", "proof": {"operation_id": operation_id, "source": "legacy_cancel"}}
    if outcome.get("state") == "intervention":
        return {"state": "intervention", "reason": outcome.get("reason") or "group_cancel_intervention"}
    next_cursor = dict(cursor)
    next_cursor["child"] = outcome.get("cursor", {})
    return {
        "state": "running",
        "cursor": next_cursor,
        "meaningful_progress": next_cursor != cursor,
    }


def _advance_projection_work(
    cfg: RootConfig,
    descriptor: dict[str, Any],
    *,
    reservation_runtime_root: Path | None,
) -> dict[str, Any]:
    kind = descriptor["identity"]["kind"]
    if kind == "ready_index":
        cursor, done, repaired, blocked, failure = _advance_ready_phase(
            cfg,
            {"cursor": descriptor["cursor"], "prior_degraded_reasons": []},
        )
        return {
            "state": "intervention" if failure is not None else "completed" if done else "running",
            "cursor": cursor,
            "proof": {"source": "ready_index_fence", "repaired": repaired},
            "reason": None if failure is None else failure["code"],
            "meaningful_progress": cursor != descriptor["cursor"] or done,
        }
    if kind == "group_ready_members":
        cursor, done, repaired, blocked, failure = _advance_group_members_phase(
            cfg,
            {"cursor": descriptor["cursor"]},
        )
        return {
            "state": "intervention" if failure is not None else "completed" if done else "running",
            "cursor": cursor,
            "proof": {"source": "group_ready_members_fence", "repaired": repaired},
            "reason": None if failure is None else failure["code"],
            "meaningful_progress": cursor != descriptor["cursor"] or done,
        }
    if kind == "task_observation":
        cursor, done, result, failure = _advance_observation_phase(cfg, {"cursor": descriptor["cursor"]})
        return {
            "state": "intervention"
            if failure is not None
            else "completed"
            if done
            else ("waiting" if result.get("state") == "waiting" else "running"),
            "cursor": cursor,
            "reason": failure["code"] if failure else result.get("reason"),
            "meaningful_progress": cursor != descriptor["cursor"] or done or result.get("processed", 0) > 0,
        }
    if kind == "submission_control":
        cursor, done, result, failure = _advance_submission_control_phase(cfg, {"cursor": descriptor["cursor"]})
        return {
            "state": "intervention"
            if failure is not None
            else "completed"
            if done
            else ("waiting" if result.get("state") == "waiting" else "running"),
            "cursor": cursor,
            "reason": failure["code"] if failure else result.get("reason"),
            "meaningful_progress": cursor != descriptor["cursor"] or done,
        }
    if kind == "deadline_index":
        task_id = descriptor["identity"]["target_id"]
        from ..runtime.availability.offer_deadlines import remove_deadline_index, sync_deadline_index

        try:
            task = TaskRecord.from_dict(
                read_json_limited(task_path(cfg.shared_root, task_id), max_bytes=_MAX_SOURCE_RECORD_BYTES)
            )
        except FileNotFoundError:
            remove_deadline_index(cfg, task_id)
        else:
            sync_deadline_index(cfg, task)
        return {"state": "completed", "proof": {"source": "deadline_index_task", "task_id": task_id}}
    if kind == "orphan_recovery":
        cursor = descriptor["cursor"]
        task_id = cursor.get("task_id")
        attempt_id = descriptor["identity"]["target_id"]
        attempt_number = cursor.get("attempt_number")
        fencing_token = cursor.get("fencing_token")
        if (
            not isinstance(task_id, str)
            or cursor.get("attempt_id") != attempt_id
            or type(attempt_number) is not int
            or attempt_number < 0
            or type(fencing_token) is not int
            or fencing_token < 0
        ):
            return {"state": "intervention", "reason": "orphan_attempt_descriptor_invalid"}
        attempt_path_value = shared_paths(cfg.shared_root)["attempts"] / task_id / f"{attempt_number}.json"
        try:
            attempt = AttemptRecord.from_dict(
                read_json_limited(
                    attempt_path_value,
                    max_bytes=_MAX_SOURCE_RECORD_BYTES,
                    record_type="maintenance_known_attempt",
                )
            )
            task = TaskRecord.from_dict(
                read_json_limited(
                    task_path(cfg.shared_root, task_id),
                    max_bytes=_MAX_SOURCE_RECORD_BYTES,
                    record_type="maintenance_known_attempt_task",
                )
            )
        except FileNotFoundError:
            return {"state": "intervention", "reason": "orphan_attempt_truth_missing"}
        if attempt.task_id != task_id or attempt.attempt_number != attempt_number or attempt.attempt_id != attempt_id:
            return {"state": "intervention", "reason": "orphan_attempt_identity_mismatch"}
        active_claim = task.claim_control.get("active_claim") or {}
        if attempt.phase != "orphaned":
            if (
                task.state.get("projection") == "running"
                and active_claim.get("attempt_id") != attempt_id
                and type(active_claim.get("fencing_token")) is int
                and active_claim["fencing_token"] > fencing_token
            ):
                return {
                    "state": "completed",
                    "proof": {
                        "source": "attempt_recovery_superseded",
                        "attempt_id": attempt_id,
                        "fencing_token": active_claim["fencing_token"],
                    },
                }
            return {"state": "intervention", "reason": "orphan_attempt_truth_not_committed"}
        if (
            task.attempt_control.get("current_attempt_number") != attempt_number
            or task.state.get("projection") != "blocked"
            or active_claim
        ):
            return {"state": "intervention", "reason": "orphan_task_authority_mismatch"}
        child = {
            key: cursor[key]
            for key in (
                "stage",
                "task_id",
                "attempt_number",
                "attempt_id",
                "fencing_token",
                "attempt_process_identity",
                "was_terminated",
            )
            if key in cursor
        }
        next_cursor, done, repaired, blocked, failure = _advance_orphan_phase(
            cfg,
            {"cursor": {"orphan": child}},
            reservation_runtime_root,
        )
        child_after = next_cursor.get("orphan")
        if failure is not None:
            return {"state": "intervention", "reason": failure["code"]}
        if done or child_after is None:
            return {
                "state": "completed",
                "proof": {"source": "known_attempt_fence", "attempt_id": attempt_id, "repaired": repaired},
                "meaningful_progress": True,
            }
        return {
            "state": "running",
            "cursor": {**cursor, **child_after},
            "reason": "known_attempt_recovery_pending",
            "meaningful_progress": child_after != child,
        }
    return {"state": "intervention", "reason": "maintenance_descriptor_kind_unknown"}


def _next_descriptor_retry(record: dict[str, Any], reason: str | None) -> tuple[str, int, dict[str, str]]:
    retry_count = record["retry_count"] + 1
    base_delay = min(60.0, float(2 ** min(retry_count - 1, 6)))
    delay = random.SystemRandom().uniform(0.8 * base_delay, base_delay)
    due_at = (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()
    failure = {"code": reason or "maintenance_waiting", "phase": record["phase"], "type": "TransientWait"}
    return due_at, retry_count, failure


def _prepared_truth_committed(cfg: RootConfig, descriptor: dict[str, Any]) -> bool:
    """Recognize producer handoff from authoritative truth, never from elapsed time."""
    identity = descriptor["identity"]
    kind = identity["kind"]
    if kind in {"cleanup", "availability", "group_cancel", "submission"}:
        path = _work_operation_path(cfg, descriptor)
        if path is None:
            return False
        try:
            operation = read_json_limited(path, max_bytes=_MAX_SOURCE_RECORD_BYTES)
        except FileNotFoundError:
            return False
        section = operation.get(_operation_section(kind)) if type(operation) is dict else None
        return type(section) is dict and section.get("operation_id") == identity["target_id"]
    if kind == "ready_index":
        from ..runtime.ready import read_ready_index_status

        status = read_ready_index_status(cfg)
        build = status.get("build") if isinstance(status.get("build"), dict) else {}
        expected = descriptor["cursor"].get("build_id")
        if isinstance(expected, str):
            return build.get("build_id") == expected
        return status.get("state") == "degraded" and descriptor["cursor"].get("mode") == "build"
    if kind == "group_ready_members":
        from ..runtime.ready.group_members import read_group_ready_members_state

        status = read_group_ready_members_state(cfg)
        build = status.get("build") if isinstance(status.get("build"), dict) else {}
        park = status.get("park") if isinstance(status.get("park"), dict) else {}
        expected = descriptor["cursor"].get("build_id")
        return isinstance(expected, str) and expected in {build.get("build_id"), park.get("build_id")}
    if kind == "task_observation":
        status = inspect_observation(cfg)
        return status.get("state") in {"building", "degraded"} or bool(status.get("dirty"))
    if kind == "submission_control":
        return inspect_submission_control(cfg).get("state") in {"building", "degraded"}
    if kind == "orphan_recovery":
        cursor = descriptor["cursor"]
        task_id = cursor.get("task_id")
        attempt_number = cursor.get("attempt_number")
        if not isinstance(task_id, str) or type(attempt_number) is not int:
            return False
        try:
            attempt = AttemptRecord.from_dict(
                read_json_limited(
                    shared_paths(cfg.shared_root)["attempts"] / task_id / f"{attempt_number}.json",
                    max_bytes=_MAX_SOURCE_RECORD_BYTES,
                )
            )
            task = TaskRecord.from_dict(
                read_json_limited(
                    task_path(cfg.shared_root, task_id),
                    max_bytes=_MAX_SOURCE_RECORD_BYTES,
                )
            )
        except FileNotFoundError:
            return False
        return (
            attempt.attempt_id == identity["target_id"]
            and attempt.attempt_number == attempt_number
            and attempt.phase == "orphaned"
            and task.attempt_control.get("current_attempt_number") == attempt_number
            and task.attempt_control.get("current_attempt_id") is None
            and task.state.get("projection") == "blocked"
            and task.state.get("reason") == "orphaned_attempt_requires_recovery"
            and not task.claim_control.get("active_claim")
        )
    # Deadline repair is idempotently derived from current Task truth. The
    # full-audit adapter has its own durable descriptor as the handoff proof.
    return kind in {"deadline_index", "full_audit"}


def advance_maintenance_work(
    cfg: RootConfig,
    *,
    reservation_runtime_root: Path | None,
    max_scan: int = 1,
) -> dict[str, Any]:
    """Advance one durable descriptor selected from the Project-local outbox."""
    ledger = create_invocation_ledger(1)
    ledger.charge_operations(operations=32)
    _ensure_full_audit_outbox(cfg)
    selection = select_due_work(cfg, max_scan=max_scan)
    descriptor = selection["descriptor"]
    if descriptor is None:
        state = "pending" if selection.get("more") else "waiting" if selection.get("next_due_at") else "idle"
        return {
            "maintenance_state": state,
            "next_due_at": selection.get("next_due_at"),
            "more": selection.get("more", False),
            "budget": ledger.report(),
        }

    identity = descriptor["identity"]
    kind = identity["kind"]
    if descriptor["state"] == "prepared":
        if _prepared_truth_committed(cfg, descriptor):
            descriptor = activate_work(cfg, descriptor)
        else:
            due_at = (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat()
            update_work(
                cfg,
                kind=kind,
                target_id=identity["target_id"],
                work_generation=identity["work_generation"],
                state="prepared",
                due_at=due_at,
                failure={"code": "producer_handoff_pending", "type": "TransientWait"},
                publish_activation=False,
            )
            return {"maintenance_state": "waiting", "next_due_at": due_at, "budget": ledger.report()}

    if kind == "full_audit":
        progress = advance_full_audit(
            cfg,
            reservation_runtime_root=reservation_runtime_root,
            max_work_items=1,
            ledger=ledger,
            create_if_missing=False,
            create_successor=False,
            create_successor_on_change=True,
        )
        target_state = progress.get("maintenance_state")
        if target_state == "completed":
            retire_work(
                cfg,
                kind=kind,
                target_id=identity["target_id"],
                work_generation=identity["work_generation"],
                proof={"source": "full_audit", "capture_id": progress.get("scope", {}).get("capture_id")},
            )
        elif target_state == "intervention":
            retire_work(
                cfg,
                kind=kind,
                target_id=identity["target_id"],
                work_generation=identity["work_generation"],
                state="intervention",
                proof={"failure": progress.get("failure") or progress.get("intervention")},
            )
        else:
            phase = progress.get("phase") or descriptor["phase"]
            cursor = {"phase": phase, "cursor": progress.get("cursor", {})}
            if target_state == "waiting":
                due_at = progress.get("next_due_at") or descriptor["due_at"]
                update_work(
                    cfg,
                    kind=kind,
                    target_id=identity["target_id"],
                    work_generation=identity["work_generation"],
                    state="waiting",
                    phase=phase,
                    cursor=cursor,
                    due_at=due_at,
                )
            else:
                update_work(
                    cfg,
                    kind=kind,
                    target_id=identity["target_id"],
                    work_generation=identity["work_generation"],
                    state="running",
                    phase=phase,
                    cursor=cursor,
                    meaningful_progress=True,
                )
        return progress

    reservation_cost = PHASE_OPERATION_RESERVATIONS.get(kind, 128)
    reservation = ledger.reserve_operations(maximum_operations=reservation_cost)
    if reservation is None:
        due_at, retry_count, failure = _next_descriptor_retry(descriptor, "maintenance_budget_unavailable")
        update_work(
            cfg,
            kind=kind,
            target_id=identity["target_id"],
            work_generation=identity["work_generation"],
            state="waiting",
            due_at=due_at,
            retry_count=retry_count,
            failure=failure,
        )
        return {"maintenance_state": "waiting", "next_due_at": due_at, "budget": ledger.report()}
    ledger.consume_semantic_item()
    try:
        if kind in {"cleanup", "availability", "group_cancel", "submission"}:
            outcome = _advance_operation_work(
                cfg,
                descriptor,
                reservation_runtime_root=reservation_runtime_root,
            )
        else:
            outcome = _advance_projection_work(
                cfg,
                descriptor,
                reservation_runtime_root=reservation_runtime_root,
            )
    except (AttributeError, FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        if _is_transient_failure(exc):
            outcome = {"state": "waiting", "reason": type(exc).__name__}
        else:
            outcome = {"state": "intervention", "reason": type(exc).__name__}
    finally:
        if reservation._active:
            reservation.commit(actual_operations=reservation_cost)

    latest = read_work(
        cfg,
        kind=kind,
        target_id=identity["target_id"],
        work_generation=identity["work_generation"],
    )
    if latest is not None and latest["state"] in {"completed", "intervention", "superseded"}:
        return {"maintenance_state": latest["state"], "descriptor": latest}
    state = outcome.get("state")
    if state == "completed":
        retire_work(
            cfg,
            kind=kind,
            target_id=identity["target_id"],
            work_generation=identity["work_generation"],
            proof=outcome.get("proof") or {"source": "maintenance_completion_fence"},
        )
    elif state == "intervention":
        retire_work(
            cfg,
            kind=kind,
            target_id=identity["target_id"],
            work_generation=identity["work_generation"],
            state="intervention",
            proof={"code": outcome.get("reason") or "maintenance_intervention"},
        )
    elif state == "waiting":
        due_at, retry_count, failure = _next_descriptor_retry(descriptor, outcome.get("reason"))
        update_work(
            cfg,
            kind=kind,
            target_id=identity["target_id"],
            work_generation=identity["work_generation"],
            state="waiting",
            phase=str(outcome.get("phase") or descriptor["phase"]),
            cursor=outcome.get("cursor", descriptor["cursor"]),
            due_at=due_at,
            retry_count=retry_count,
            failure=failure,
            meaningful_progress=bool(outcome.get("meaningful_progress")),
        )
    else:
        update_work(
            cfg,
            kind=kind,
            target_id=identity["target_id"],
            work_generation=identity["work_generation"],
            state="running",
            phase=str(outcome.get("phase") or descriptor["phase"]),
            cursor=outcome.get("cursor", descriptor["cursor"]),
            due_at=utc_now(),
            retry_count=0 if outcome.get("meaningful_progress") else descriptor["retry_count"],
            failure=None if outcome.get("meaningful_progress") else descriptor.get("failure"),
            meaningful_progress=bool(outcome.get("meaningful_progress")),
        )
    return {
        "maintenance_state": state or "running",
        "descriptor_identity": identity,
        "phase": outcome.get("phase") or descriptor["phase"],
        "cursor": outcome.get("cursor", descriptor["cursor"]),
        "next_due_at": due_at if state == "waiting" else None,
        "budget": ledger.report(),
    }


def _ensure_full_audit_outbox(cfg: RootConfig) -> None:
    """Adopt one pre-Stage-3 full-audit checkpoint without scanning history."""
    pointer = _read_pointer(cfg)
    if pointer is None:
        return
    try:
        record = _load_descriptor(cfg, pointer)
    except (OSError, RuntimeError, TypeError, ValueError, KeyError):
        return
    if record["state"] in {"completed", "intervention"}:
        return
    identity = record["identity"]
    work = read_work(
        cfg,
        kind="full_audit",
        target_id="project",
        work_generation=identity["work_generation"],
    )
    if work is None:
        work = prepare_work(
            cfg,
            kind="full_audit",
            target_id="project",
            work_generation=identity["work_generation"],
            phase=record["phase"],
            cursor=record["cursor"],
        )
    if work["state"] == "prepared":
        activate_work(cfg, work)


def _intervene(record: dict[str, Any], failure: dict[str, str]) -> None:
    record["state"] = "intervention"
    record["failure"] = failure


def _public_progress(
    cfg: RootConfig,
    record: dict[str, Any] | None,
    ledger: InvocationLedger,
    *,
    repaired: list[str],
    blocked: list[str],
    observations: dict[str, Any] | None,
    submission_control: dict[str, Any] | None,
    intervention: dict[str, str] | None,
    busy: bool = False,
) -> dict[str, Any]:
    phase = record.get("phase") if record is not None else PHASES[0]
    phase_index = PHASES.index(phase) if phase in PHASES else 0
    deferred = [] if record is not None and record.get("state") == "completed" else list(PHASES[phase_index:])[:12]
    ready_status: dict[str, Any] = {}
    members_status: dict[str, Any] = {}
    prior_reasons = list(record.get("prior_degraded_reasons", [])) if record is not None else []
    try:
        from ..runtime.ready import read_ready_index_status

        ready_status = read_ready_index_status(cfg)
        if record is None:
            prior_reasons = list(ready_status.get("degraded_reasons", []))
    except (AttributeError, FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        ready_status = {"state": "unavailable"}
    try:
        from ..runtime.ready.group_members import group_ready_members_state, read_group_ready_members_state

        members_status = read_group_ready_members_state(cfg)
        members_status.setdefault("state", group_ready_members_state(cfg))
    except (AttributeError, FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        members_status = {"state": "unavailable"}
    if observations is None:
        try:
            observations = inspect_observation(cfg)
        except (OSError, RuntimeError, TypeError, ValueError):
            observations = {"state": "unavailable"}
    if submission_control is None:
        try:
            submission_control = inspect_submission_control(cfg)
        except (OSError, RuntimeError, TypeError, ValueError):
            submission_control = {"state": "unavailable"}
    complete = record is not None and record.get("state") == "completed"
    if record is None:
        scope = {"kind": "full_audit", "capture_id": None, "work_generation": None}
        cursor: dict[str, Any] = {}
        state = "intervention" if intervention is not None else "running"
    else:
        identity = record["identity"]
        scope = {
            "kind": "full_audit",
            "capture_id": record["capture_id"],
            "work_generation": identity["work_generation"],
        }
        cursor = dict(record["cursor"])
        state = record["state"]
    budget = ledger.report()
    if busy and budget["exhaustion_reason"] is None:
        budget["exhaustion_reason"] = None
    return {
        "repaired": repaired[:64],
        "blocked": blocked[:64],
        "complete": complete,
        "scope": scope,
        "budget": budget,
        "phase": phase,
        "cursor": cursor,
        "deferred_phases": deferred,
        "deferred_phase_count": 0
        if record is not None and record.get("state") == "completed"
        else len(PHASES[phase_index:]),
        "next_due_at": None if complete or intervention is not None or record is None else record["due_at"],
        "remaining_work": None,
        "maintenance_state": state,
        "source_capture": None if record is None else record["source_capture"],
        "source_revision": None if record is None else record["source_revision"],
        "build_identity": None if record is None else record["build_identity"],
        "due_at": None if record is None else record["due_at"],
        "retry_count": 0 if record is None else record["retry_count"],
        "meaningful_progress_at": None if record is None else record["meaningful_progress_at"],
        "failure": None if record is None else record["failure"],
        "intervention": intervention,
        "ready_index": {
            "state": ready_status.get("state"),
            "build": ready_status.get("build") or {},
            "degraded_reasons": ready_status.get("degraded_reasons", []),
        },
        "group_ready_members": members_status,
        "prior_degraded_reasons": prior_reasons,
        "task_observation": observations,
        "submission_control": submission_control,
    }


def create_invocation_ledger(max_work_items: int) -> InvocationLedger:
    """Create the single ledger before any repair context resolution."""
    if type(max_work_items) is not int or not 1 <= max_work_items <= 64:
        raise ValueError("max_work_items must be between 1 and 64.")
    return InvocationLedger(
        semantic_item_limit=max_work_items,
        operation_limit=_OPERATION_LIMIT,
        deadline_ms=_DEADLINE_MS,
    )


def advance_full_audit(
    cfg: RootConfig,
    *,
    reservation_runtime_root: Path | None,
    max_work_items: int,
    ledger: InvocationLedger | None = None,
    create_if_missing: bool = True,
    create_successor: bool = True,
    create_successor_on_change: bool = False,
    retry_intervention: bool = False,
) -> dict[str, Any]:
    """Attach to or advance one durable full-project audit within one ledger."""
    if type(max_work_items) is not int or not 1 <= max_work_items <= 64:
        raise ValueError("max_work_items must be between 1 and 64.")
    ledger = ledger or create_invocation_ledger(max_work_items)
    if ledger.semantic_item_limit != max_work_items:
        raise ValueError("invocation ledger semantic-item limit does not match max_work_items.")
    exit_reservation = ledger.reserve_operations(maximum_operations=_EXIT_RESERVATION_OPERATIONS)
    record: dict[str, Any] | None = None
    repaired: list[str] = []
    blocked: list[str] = []
    observations: dict[str, Any] | None = None
    submission_control: dict[str, Any] | None = None
    intervention: dict[str, str] | None = None
    busy = False
    try:
        if exit_reservation is not None:
            try:
                ledger.charge_operations(operations=_SETUP_OPERATIONS)
                with _repair_lock(cfg) as acquired:
                    if not acquired:
                        busy = True
                    else:
                        root = _maintenance_root(cfg)
                        root.mkdir(parents=True, exist_ok=True)
                        metadata = root.lstat()
                        if not stat.S_ISDIR(metadata.st_mode):
                            raise ValueError("maintenance descriptor root is not a directory.")
                        pointer = _read_pointer(cfg)
                        if pointer is None:
                            if create_if_missing:
                                pending = _new_descriptor(cfg)
                                record = pending if _initialize_descriptor(cfg, pending, ledger) else None
                        else:
                            try:
                                record = _load_descriptor(cfg, pointer)
                            except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
                                intervention = {
                                    "code": "maintenance_descriptor_untrusted",
                                    "phase": "descriptor_resolution",
                                    "type": type(exc).__name__,
                                }
                                blocked.append(intervention["code"])
                            source_changed = (
                                record is not None
                                and record["state"] == "completed"
                                and create_successor_on_change
                                and _source_changed(cfg, record)
                            )
                            if (
                                record is not None
                                and record["state"] == "completed"
                                and (create_successor or source_changed)
                            ):
                                successor = _new_descriptor(cfg)
                                record = successor if _initialize_descriptor(cfg, successor, ledger) else None

                        if record is not None and record["state"] == "intervention" and retry_intervention:
                            successor = _new_descriptor(cfg)
                            record = successor if _initialize_descriptor(cfg, successor, ledger) else None

                        if record is not None and record["state"] == "intervention":
                            intervention = record["failure"]
                            code = intervention.get("code") if isinstance(intervention, dict) else None
                            if isinstance(code, str):
                                blocked.append(code)
                        if record is not None and record["state"] == "waiting" and _retry_due(record):
                            record["state"] = "running"
                        while record is not None and record["state"] in {"pending", "running"}:
                            phase_lock = None
                            phase_lock_entered = False
                            try:
                                if record["phase"] == "group_ready_members":
                                    phase_lock = schema_lock(cfg.shared_root, blocking=False)
                                    has_schema_lock = phase_lock.__enter__()
                                    phase_lock_entered = True
                                    if not has_schema_lock:
                                        phase_lock.__exit__(None, None, None)
                                        phase_lock = None
                                        phase_lock_entered = False
                                        busy = True
                                        break
                                reservation_cost = PHASE_OPERATION_RESERVATIONS[record["phase"]]
                                reservation = ledger.reserve_operations(maximum_operations=reservation_cost)
                                if reservation is None:
                                    break
                                try:
                                    ledger.consume_semantic_item()
                                    previous_phase = record["phase"]
                                    previous_cursor = dict(record["cursor"])
                                    previous_build_identity = (
                                        dict(record["build_identity"])
                                        if isinstance(record.get("build_identity"), dict)
                                        else None
                                    )
                                    previous_completed = len(record["completed_phases"])
                                    try:
                                        cursor, phase_done, step_repaired, step_blocked, failure, phase_result = (
                                            _advance_phase(
                                                cfg,
                                                record,
                                                reservation_runtime_root=reservation_runtime_root,
                                            )
                                        )
                                        record["cursor"] = cursor
                                        record["state"] = "running"
                                        repaired.extend(step_repaired)
                                        blocked.extend(step_blocked)
                                        if phase_result is not None:
                                            if record["phase"] == "task_observation":
                                                observations = phase_result
                                            elif record["phase"] == "submission_control":
                                                submission_control = phase_result
                                        meaningful_progress = (
                                            cursor != previous_cursor
                                            or record["build_identity"] != previous_build_identity
                                            or phase_done
                                            or len(record["completed_phases"]) != previous_completed
                                            or bool(step_repaired)
                                        )
                                        transient_wait = (
                                            failure is None
                                            and isinstance(phase_result, dict)
                                            and phase_result.get("state") == "waiting"
                                        )
                                        if transient_wait:
                                            retry_failure = {
                                                "code": str(phase_result.get("reason") or "maintenance_waiting"),
                                                "phase": record["phase"],
                                                "type": "TransientWait",
                                            }
                                            if meaningful_progress:
                                                record["retry_count"] = 0
                                                record["meaningful_progress_at"] = utc_now()
                                            _schedule_retry(record, retry_failure)
                                        elif failure is not None:
                                            _intervene(record, failure)
                                            intervention = failure
                                        elif phase_done:
                                            _advance_phase_name(record, done=True)
                                        meaningful_progress = meaningful_progress or previous_phase != record["phase"]
                                        _persist_progress(
                                            cfg,
                                            record,
                                            meaningful_progress=meaningful_progress,
                                        )
                                    except (
                                        AttributeError,
                                        FileNotFoundError,
                                        KeyError,
                                        OSError,
                                        RuntimeError,
                                        TypeError,
                                        ValueError,
                                    ) as exc:
                                        failure = {
                                            "code": "maintenance_phase_failed",
                                            "phase": record["phase"],
                                            "type": type(exc).__name__,
                                        }
                                        meaningful_progress = (
                                            record["cursor"] != previous_cursor
                                            or len(record["completed_phases"]) != previous_completed
                                        )
                                        if _is_transient_failure(exc):
                                            if meaningful_progress:
                                                record["retry_count"] = 0
                                                record["meaningful_progress_at"] = utc_now()
                                            _schedule_retry(record, failure)
                                        else:
                                            _intervene(record, failure)
                                            intervention = failure
                                        try:
                                            _persist_progress(
                                                cfg,
                                                record,
                                                meaningful_progress=meaningful_progress,
                                            )
                                        except (
                                            AttributeError,
                                            FileNotFoundError,
                                            KeyError,
                                            OSError,
                                            RuntimeError,
                                            TypeError,
                                            ValueError,
                                        ):
                                            blocked.append("maintenance_progress_commit_failed")
                                finally:
                                    if reservation._active:
                                        reservation.commit(actual_operations=reservation_cost)
                            finally:
                                if phase_lock is not None and phase_lock_entered:
                                    phase_lock.__exit__(None, None, None)
                            if intervention is not None or record["state"] == "completed":
                                break
            except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
                intervention = {
                    "code": "maintenance_descriptor_failed",
                    "phase": "descriptor_resolution",
                    "type": type(exc).__name__,
                }
                blocked.append(intervention["code"])
    finally:
        # Keep the bounded exit reservation live through status collection and
        # rendering below; it is committed only after those reads finish.
        pass
    if intervention is not None and record is not None and record.get("failure") is None:
        record["failure"] = intervention
        record["state"] = "intervention"
    if intervention is not None and not blocked:
        blocked.append(intervention["code"])
    try:
        progress = _public_progress(
            cfg,
            record,
            ledger,
            repaired=repaired,
            blocked=blocked,
            observations=observations,
            submission_control=submission_control,
            intervention=intervention,
            busy=busy,
        )
    finally:
        if exit_reservation is not None and exit_reservation._active:
            exit_reservation.commit(actual_operations=_EXIT_RESERVATION_OPERATIONS)
    progress["budget"] = ledger.report()
    if not create_if_missing and record is None and intervention is None:
        progress["maintenance_state"] = "idle"
        progress["next_due_at"] = None
    return progress


__all__ = [
    "CONTEXT_RESOLUTION_OPERATIONS",
    "PHASES",
    "PHASE_OPERATION_RESERVATIONS",
    "advance_full_audit",
    "create_invocation_ledger",
]
