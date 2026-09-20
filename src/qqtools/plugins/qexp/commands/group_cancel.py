"""Bounded indexed Group cancellation convergence.

The canonical Group namespace owns a durable membership coverage projection.  A
Group cancellation consumes that projection one member at a time and then
services the ordered Task-change journal.  This module deliberately keeps the
legacy Task-directory replay in :mod:`group` for pre-isolation roots.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..config_types import RootConfig
from ..runtime.group_discovery.coverage import GroupCoverage, MemberIdentity
from ..runtime.group_namespace import group_authority_identity
from ..runtime.locks import group_writer_lock, task_lock
from ..runtime.operation_store import archive_operation, write_active_operation
from ..runtime.paths import attempt_path, group_path, task_path
from ..runtime.records import AttemptRecord, new_id, normalize_group_record, utc_now, validate_identifier
from ..runtime.store import atomic_replace, read_json
from ..runtime.tasks import load_task
from .task import is_cleanup_blocked

_ACTIVE_STATES = {"preparing", "converging", "waiting_ack", "blocked"}
_TERMINAL_STATES = {"succeeded", "failed", "cancelled"}
_PROGRESS_KEYS = (
    "target_tasks",
    "already_terminal",
    "queued_cancelled",
    "prelaunch_cancelled",
    "running_allowed",
    "termination_pending",
    "termination_acknowledged",
    "blocked",
)
_HISTORICAL_EFFECT_KEYS = frozenset({"queued_cancelled", "prelaunch_cancelled", "running_allowed"})
_RECEIPT_VERSION = 1
_DISCOVERY_VERSION = 1
_RECEIPT_MAX_BYTES = 65_536


def initialize_cancel_discovery(cfg: RootConfig, control: dict[str, Any]) -> None:
    """Initialize the canonical recheck journal and cancellation census.

    The caller must hold the Group writer fence.  No source sweep is started;
    GroupCoverage is advanced by the background discovery service and this
    consumer only reads its certified prefix.
    """

    group_name = control.get("group_name")
    if not isinstance(group_name, str) or not group_name:
        raise ValueError("Group cancellation control has no valid group name")
    from ..runtime.group_discovery.rechecks import GroupRechecks

    position = GroupRechecks(cfg.shared_root, group_name).snapshot(initialize=True)
    if position is None:
        raise RuntimeError("Group recheck journal did not initialize")
    authority = group_authority_identity(cfg.shared_root)
    control["discovery"] = _new_discovery(authority, position, is_legacy_adoption=False)


def advance_indexed_cancel(
    cfg: RootConfig,
    operation_path: Path,
    *,
    reservation_runtime_root: Path | None = None,
) -> dict[str, Any] | None:
    """Advance one canonical Group cancellation member or journal entry.

    At most one membership or recheck record is consumed by one invocation.
    The operation summary is returned after the Group fence is released.  A
    missing operation is treated as an already-reconciled caller race.
    """

    operation_path = Path(operation_path)
    operation = _read_optional_operation(operation_path)
    if operation is None:
        return None
    control = operation.get("group_control")
    if not isinstance(control, dict):
        return None
    group_name = control.get("group_name")
    if not isinstance(group_name, str) or not group_name:
        return _unwritable_blocked_summary(operation, "operation_group_missing")

    terminal_results: list[Any] = []
    summary: dict[str, Any] | None = None
    with group_writer_lock(cfg, group_name):
        if not operation_path.exists():
            return None
        operation = _read_optional_operation(operation_path)
        if operation is None:
            return None
        control = operation.get("group_control")
        if not isinstance(control, dict):
            return None
        summary, terminal_results = _advance_locked(
            cfg,
            operation_path,
            operation,
            control,
            reservation_runtime_root=reservation_runtime_root,
        )

    if terminal_results:
        # Keep this import lazy: group.py imports this module and owns the
        # post-lock reservation release and lifecycle-hook dispatch primitive.
        from .group import _dispatch_group_terminal_results

        _dispatch_group_terminal_results(cfg, terminal_results, reservation_runtime_root)
    return summary


def _advance_locked(
    cfg: RootConfig,
    operation_path: Path,
    operation: dict[str, Any],
    control: dict[str, Any],
    *,
    reservation_runtime_root: Path | None,
) -> tuple[dict[str, Any], list[Any]]:
    """Advance one operation while the caller holds the Group writer fence."""

    terminal_results: list[Any] = []
    operation_id = control.get("operation_id")
    group_name = control.get("group_name")
    if not isinstance(operation_id, str) or not _valid_identifier(operation_id):
        return _block_operation(cfg, operation, control, "operation_id_invalid", update_group=False), terminal_results
    if not isinstance(group_name, str) or not group_name:
        return _block_operation(
            cfg, operation, control, "operation_group_missing", update_group=False
        ), terminal_results
    if control.get("operation_type") != "cancel":
        return control, terminal_results
    state = control.get("state")
    if state not in _ACTIVE_STATES:
        return control, terminal_results
    high_watermark = control.get("membership_high_watermark")
    terminate_running = control.get("terminate_running")
    if type(high_watermark) is not int or high_watermark < 0 or type(terminate_running) is not bool:
        return _block_operation(cfg, operation, control, "cancellation_control_invalid"), terminal_results

    group_file = group_path(cfg.shared_root, group_name)
    if not group_file.exists():
        return _block_operation(cfg, operation, control, "group_missing", update_group=False), terminal_results
    try:
        group_data = read_json(group_file)
        normalize_group_record(group_data)
    except (OSError, KeyError, TypeError, ValueError):
        return _block_operation(cfg, operation, control, "group_malformed", update_group=False), terminal_results

    if not _has_exact_barrier(group_data, operation_id, high_watermark, terminate_running):
        return _block_operation(cfg, operation, control, "cancellation_barrier_mismatch"), terminal_results

    try:
        authority = group_authority_identity(cfg.shared_root)
    except (OSError, RuntimeError, TypeError, ValueError, KeyError):
        return _block_operation(
            cfg, operation, control, "group_authority_unavailable", update_group=False
        ), terminal_results

    discovery = control.get("discovery")
    if discovery is None:
        # This is the one-time adoption path for an active operation created by
        # the pre-isolation writer.  Existing effect counters remain historical.
        try:
            discovery = _adopt_discovery(cfg, control, authority)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            return _block_operation(
                cfg, operation, control, _reason("discovery_initialization_failed", exc)
            ), terminal_results
    elif not isinstance(discovery, dict):
        return _block_operation(cfg, operation, control, "discovery_malformed"), terminal_results

    if discovery.get("authority") != authority:
        return _block_operation(cfg, operation, control, "discovery_authority_mismatch"), terminal_results
    try:
        discovery = _validate_discovery(discovery, high_watermark)
    except (TypeError, ValueError, KeyError) as exc:
        return _block_operation(cfg, operation, control, _reason("discovery_malformed", exc)), terminal_results

    from ..runtime.group_discovery.rechecks import GroupRechecks

    journal = GroupRechecks(cfg.shared_root, group_name)
    try:
        position = journal.snapshot()
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        return _block_operation(cfg, operation, control, _reason("recheck_journal_unavailable", exc)), terminal_results
    if position is None:
        return _block_operation(cfg, operation, control, "recheck_journal_missing"), terminal_results

    if discovery["generation"] != _position_value(position, "generation"):
        _reset_discovery(control, discovery, position, legacy_adoption=bool(discovery["is_legacy_adoption"]))
        control["discovery"] = discovery
        _persist_operation(cfg, operation, control)
        return _finish_step(cfg, operation, control, group_data, terminal_results), terminal_results

    if discovery.get("pending_receipt") is not None:
        try:
            _finish_pending_receipt(cfg, operation, control, discovery)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            return _block_operation(cfg, operation, control, _reason("receipt_recovery_failed", exc)), terminal_results
        return _finish_step(cfg, operation, control, group_data, terminal_results), terminal_results

    current_member = discovery.get("current_member")
    if current_member is not None:
        try:
            member = _member_for_intent(cfg, group_name, current_member)
            terminal_results.extend(
                _consume_member(
                    cfg,
                    operation,
                    control,
                    discovery,
                    member,
                    group_name,
                    current_member,
                    terminal_results,
                )
            )
        except _BlockedStep as exc:
            return _block_operation(cfg, operation, control, exc.reason), terminal_results
        return _finish_step(cfg, operation, control, group_data, terminal_results), terminal_results

    coverage = GroupCoverage(cfg.shared_root, group_name)
    try:
        status = coverage.status()
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        return _block_operation(
            cfg, operation, control, _reason("membership_coverage_unavailable", exc)
        ), terminal_results
    if not _coverage_allows(status, high_watermark):
        reason = _coverage_reason(status, high_watermark)
        control["state"] = "converging"
        control["completed_at"] = None
        control["blocked_reason"] = reason
        _persist_operation(cfg, operation, control)
        _update_group_snapshot(cfg, group_data, control, group_file)
        return control, terminal_results

    member_cursor = discovery["member_cursor"]
    if member_cursor < high_watermark:
        sequence = member_cursor + 1
        try:
            member = coverage.read_member(sequence)
        except BlockingIOError:
            return _block_operation(cfg, operation, control, "membership_coverage_busy"), terminal_results
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            return _block_operation(
                cfg, operation, control, _reason("membership_provenance_invalid", exc)
            ), terminal_results
        if not _member_is_valid(member, sequence):
            return _block_operation(cfg, operation, control, "membership_identity_invalid"), terminal_results
        try:
            _prepare_member_intent(cfg, operation, control, discovery, member, "member")
            terminal_results.extend(
                _consume_member(
                    cfg,
                    operation,
                    control,
                    discovery,
                    member,
                    group_name,
                    discovery["current_member"],
                    terminal_results,
                )
            )
        except _BlockedStep as exc:
            return _block_operation(cfg, operation, control, exc.reason), terminal_results
        return _finish_step(cfg, operation, control, group_data, terminal_results), terminal_results

    try:
        current_position = journal.snapshot()
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        return _block_operation(cfg, operation, control, _reason("recheck_journal_unavailable", exc)), terminal_results
    if current_position is None:
        return _block_operation(cfg, operation, control, "recheck_journal_missing"), terminal_results
    if _position_value(current_position, "generation") != discovery["generation"]:
        _reset_discovery(control, discovery, current_position, legacy_adoption=bool(discovery["is_legacy_adoption"]))
        _persist_operation(cfg, operation, control)
        return _finish_step(cfg, operation, control, group_data, terminal_results), terminal_results

    journal_cursor = discovery["journal_cursor"]
    journal_tail = _position_value(current_position, "tail")
    if journal_cursor > journal_tail:
        return _block_operation(cfg, operation, control, "recheck_cursor_invalid"), terminal_results
    if journal_cursor < journal_tail:
        journal_sequence = journal_cursor + 1
        try:
            event = journal.read(current_position, journal_sequence)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            return _block_operation(cfg, operation, control, _reason("recheck_event_invalid", exc)), terminal_results
        try:
            target = _event_targets_group(cfg, coverage, group_name, high_watermark, event)
        except _BlockedStep as exc:
            return _block_operation(cfg, operation, control, exc.reason), terminal_results
        if target and event.get("state") == "in_flight":
            settled, results = _settle_event(cfg, event)
            terminal_results.extend(results)
            if not settled:
                return _block_operation(cfg, operation, control, "transition_in_flight"), terminal_results
            try:
                event = journal.read(journal.snapshot(), journal_sequence)
            except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
                return _block_operation(
                    cfg, operation, control, _reason("recheck_event_invalid", exc)
                ), terminal_results
            if event.get("state") not in {"committed", "aborted"}:
                return _block_operation(cfg, operation, control, "transition_in_flight"), terminal_results
        if target and event.get("state") == "committed":
            member = _member_for_event(cfg, coverage, event)
            _prepare_member_intent(cfg, operation, control, discovery, member, "journal", journal_sequence)
            try:
                terminal_results.extend(
                    _consume_member(
                        cfg,
                        operation,
                        control,
                        discovery,
                        member,
                        group_name,
                        discovery["current_member"],
                        terminal_results,
                        journal_sequence=journal_sequence,
                    )
                )
            except _BlockedStep as exc:
                return _block_operation(cfg, operation, control, exc.reason), terminal_results
        else:
            discovery["journal_cursor"] = journal_sequence
            _persist_operation(cfg, operation, control)
        return _finish_step(cfg, operation, control, group_data, terminal_results), terminal_results

    # No bounded work remains in this generation.  A fresh snapshot closes the
    # race where a Task writer appended a journal event after the first read.
    return _finish_step(cfg, operation, control, group_data, terminal_results), terminal_results


def _new_discovery(authority: dict[str, Any], position: Any, *, is_legacy_adoption: bool) -> dict[str, Any]:
    generation = _position_value(position, "generation")
    tail = _position_value(position, "tail")
    return {
        "version": _DISCOVERY_VERSION,
        "authority": dict(authority),
        "generation": generation,
        "member_cursor": 0,
        "journal_cursor": tail,
        "receipt_generation": new_id(),
        "pending_receipt": None,
        "current_member": None,
        "is_legacy_adoption": is_legacy_adoption,
        "current_progress": _zero_progress(),
    }


def _adopt_discovery(cfg: RootConfig, control: dict[str, Any], authority: dict[str, Any]) -> dict[str, Any]:
    from ..runtime.group_discovery.rechecks import GroupRechecks

    position = GroupRechecks(cfg.shared_root, control["group_name"]).snapshot(initialize=True)
    if position is None:
        raise RuntimeError("Group recheck journal did not initialize")
    progress = _ensure_progress(control)
    historical = {key: progress[key] for key in _HISTORICAL_EFFECT_KEYS}
    discovery = _new_discovery(authority, position, is_legacy_adoption=True)
    discovery["historical_progress"] = historical
    control["discovery"] = discovery
    return discovery


def _validate_discovery(discovery: dict[str, Any], high_watermark: int) -> dict[str, Any]:
    if discovery.get("version") != _DISCOVERY_VERSION:
        raise ValueError("discovery version is invalid")
    if type(discovery.get("generation")) is not int or discovery["generation"] < 1:
        raise ValueError("discovery generation is invalid")
    for key in ("member_cursor", "journal_cursor"):
        if type(discovery.get(key)) is not int or discovery[key] < 0:
            raise ValueError(f"discovery {key} is invalid")
    if discovery["member_cursor"] > high_watermark:
        raise ValueError("discovery member cursor exceeds cancellation watermark")
    if not isinstance(discovery.get("receipt_generation"), str) or not _valid_identifier(
        discovery["receipt_generation"]
    ):
        raise ValueError("discovery receipt generation is invalid")
    if type(discovery.get("is_legacy_adoption")) is not bool:
        raise ValueError("discovery legacy-adoption flag is invalid")
    if discovery.get("pending_receipt") is not None and not isinstance(discovery["pending_receipt"], dict):
        raise ValueError("discovery pending receipt is invalid")
    if discovery.get("current_member") is not None and not isinstance(discovery["current_member"], dict):
        raise ValueError("discovery current member is invalid")
    discovery.setdefault("current_progress", _zero_progress())
    _validate_progress(discovery["current_progress"], "current progress")
    if discovery["is_legacy_adoption"]:
        historical = discovery.setdefault("historical_progress", _zero_progress())
        for key in _HISTORICAL_EFFECT_KEYS:
            if type(historical.get(key)) is not int or historical[key] < 0:
                raise ValueError("historical cancellation progress is invalid")
    return discovery


def _reset_discovery(
    control: dict[str, Any], discovery: dict[str, Any], position: Any, *, legacy_adoption: bool
) -> None:
    generation = _position_value(position, "generation")
    tail = _position_value(position, "tail")
    interrupted = discovery.get("current_member")
    if interrupted is not None:
        interrupted = dict(interrupted, source="recovery", effect_event_sequence=None)
    discovery.update(
        {
            "version": _DISCOVERY_VERSION,
            "generation": generation,
            "member_cursor": 0,
            "journal_cursor": tail,
            "current_member": interrupted,
            "is_legacy_adoption": legacy_adoption,
        }
    )
    # Epoch invalidation revokes completion, not durable effect receipts. Keep
    # their WAL and contributions so the new census can replace each exactly
    # once, including already-proven historical effects and pending machines.


def _prepare_member_intent(
    cfg: RootConfig,
    operation: dict[str, Any],
    control: dict[str, Any],
    discovery: dict[str, Any],
    member: MemberIdentity,
    source: str,
    journal_sequence: int | None = None,
) -> None:
    from ..runtime.group_discovery.rechecks import GroupRechecks

    position = GroupRechecks(cfg.shared_root, control["group_name"]).snapshot()
    if position is None or position.generation != discovery["generation"]:
        raise _BlockedStep("recheck_generation_changed")
    task = None
    try:
        task = load_task(cfg, member.task_id)
    except FileNotFoundError:
        pass
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise _BlockedStep(_reason("task_invalid", exc)) from exc
    claim = (task.claim_control.get("active_claim") or {}) if task is not None else {}
    expected_effect = _expected_effect(task, control) if task is not None else None
    current_member = {
        "source": source,
        "sequence": member.sequence,
        "task_id": member.task_id,
        "submission_operation_id": member.operation_id,
        "effect_event_sequence": position.tail + 1,
        "before_task_revision": task.meta.get("revision") if task is not None else None,
        "expected_effect": expected_effect,
        "attempt_id": claim.get("attempt_id")
        if claim
        else (task.attempt_control.get("current_attempt_id") if task else None),
        "attempt_number": task.attempt_control.get("current_attempt_number") if task is not None else None,
        "fencing_token": claim.get("fencing_token") if claim else None,
    }
    if journal_sequence is not None:
        current_member["journal_sequence"] = journal_sequence
    discovery["current_member"] = current_member
    _persist_operation(cfg, operation, control)


def _member_for_intent(cfg: RootConfig, group_name: str, intent: dict[str, Any]) -> MemberIdentity:
    sequence = intent.get("sequence")
    task_id = intent.get("task_id")
    operation_id = intent.get("submission_operation_id")
    if type(sequence) is not int or sequence < 1 or not isinstance(task_id, str) or not isinstance(operation_id, str):
        raise _BlockedStep("current_member_invalid")
    try:
        validate_identifier(task_id, "current member task_id")
        validate_identifier(operation_id, "current member submission_operation_id")
        member = GroupCoverage(cfg.shared_root, group_name).read_member(sequence)
    except BlockingIOError as exc:
        raise _BlockedStep("membership_coverage_busy") from exc
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        raise _BlockedStep(_reason("membership_provenance_invalid", exc)) from exc
    if (member.task_id, member.operation_id, member.sequence) != (task_id, operation_id, sequence):
        raise _BlockedStep("current_member_identity_mismatch")
    return member


def _consume_member(
    cfg: RootConfig,
    operation: dict[str, Any],
    control: dict[str, Any],
    discovery: dict[str, Any],
    member: MemberIdentity,
    group_name: str,
    intent: dict[str, Any],
    terminal_results: list[Any],
    *,
    journal_sequence: int | None = None,
) -> list[Any]:
    del terminal_results
    operation_id = control["operation_id"]
    task_path_value = task_path(cfg.shared_root, member.task_id)
    if not task_path_value.exists():
        contribution, classification = _missing_task_contribution(cfg, member, operation_id, group_name)
        _commit_receipt(
            cfg,
            operation,
            control,
            discovery,
            member,
            intent,
            contribution,
            classification,
            None,
            journal_sequence=journal_sequence,
        )
        return []

    results: list[Any] = []
    with task_lock(cfg.shared_root, member.task_id):
        try:
            task = load_task(cfg, member.task_id)
        except FileNotFoundError:
            contribution, classification = _missing_task_contribution(cfg, member, operation_id, group_name)
            _commit_receipt(
                cfg,
                operation,
                control,
                discovery,
                member,
                intent,
                contribution,
                classification,
                None,
                journal_sequence=journal_sequence,
            )
            return []
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise _BlockedStep(_reason("task_invalid", exc)) from exc

        if (
            task.group_name != group_name
            or task.group_membership_sequence != member.sequence
            or task.submission_operation_id != member.operation_id
        ):
            raise _BlockedStep("task_membership_identity_mismatch")

        # The Group fence makes our first possible effect event identifiable.
        # Recover its typed owner before the Attempt/Task coherence audit; an
        # Attempt-before-Task crash is precisely why that owner is retained.
        from ..runtime.group_discovery.changes import settle_task_change
        from ..runtime.group_discovery.rechecks import GroupRechecks

        journal = GroupRechecks(cfg.shared_root, group_name)
        position = journal.snapshot()
        effect_sequence = intent.get("effect_event_sequence")
        if (
            position is not None
            and position.generation == discovery["generation"]
            and type(effect_sequence) is int
            and effect_sequence <= position.tail
        ):
            event = journal.read(position, effect_sequence)
            details = event.get("evidence", {}).get("details", {})
            if (
                event["task_id"] == member.task_id
                and event["submission_operation_id"] == member.operation_id
                and event["owner"] in {"terminal", "task_cancel"}
                and details.get("operation_id", details.get("cancellation_operation_id")) == operation_id
                and event["state"] == "in_flight"
            ):
                settled, recovered = settle_task_change(cfg, event)
                results.extend(recovered)
                if not settled:
                    raise _BlockedStep("transition_in_flight")
                task = load_task(cfg, member.task_id)

        audit_reason = _audit_task_transition(cfg, task, operation_id)
        if audit_reason is not None:
            raise _BlockedStep(audit_reason)
        try:
            from .group import _apply_group_cancel_locked

            progress_key, terminal = _apply_group_cancel_locked(cfg, task, control)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError):
            # The durable current_member intent remains in place for the next
            # bounded call; callers must see the storage failure.
            raise
        if terminal is not None:
            results.append(terminal)
        if progress_key == "blocked":
            raise _BlockedStep("task_cancellation_requires_resolution")
        try:
            task = load_task(cfg, member.task_id)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise _BlockedStep(_reason("task_invalid_after_cancel", exc)) from exc
        audit_reason = _audit_task_transition(cfg, task, operation_id)
        if audit_reason is not None:
            raise _BlockedStep(audit_reason)
        contribution, classification, pending_machine = _classify_task(
            cfg, task, control, operation_id, member, discovery
        )
        _commit_receipt(
            cfg,
            operation,
            control,
            discovery,
            member,
            intent,
            contribution,
            classification,
            pending_machine,
            task_revision=task.meta.get("revision"),
            journal_sequence=journal_sequence,
        )
    return results


def _classify_task(
    cfg: RootConfig,
    task: Any,
    control: dict[str, Any],
    operation_id: str,
    member: MemberIdentity,
    discovery: dict[str, Any],
) -> tuple[dict[str, int], str, str | None]:
    projection = task.state.get("projection")
    reason = task.state.get("reason")
    task_control = task.control
    existing = _read_receipt(cfg, control, discovery, member.sequence)
    if existing is not None and (
        existing["task_id"] != member.task_id or existing["submission_operation_id"] != member.operation_id
    ):
        raise _BlockedStep("receipt_membership_identity_mismatch")
    classification: str
    pending_machine: str | None = None
    if projection == "queued":
        if is_cleanup_blocked(task):
            classification = "already_terminal"
        else:
            classification = "blocked"
    elif (
        projection in _TERMINAL_STATES
        and control.get("terminate_running")
        and task_control.get("termination_acknowledged_at")
        and task_control.get("cancellation_operation_id") == operation_id
    ):
        classification = "termination_acknowledged"
    elif projection == "cancelled" and task_control.get("cancellation_operation_id") == operation_id:
        if reason == "group_cancelled_before_launch":
            classification = "prelaunch_cancelled"
        elif reason == "group_cancelled":
            classification = "queued_cancelled"
        else:
            classification = "already_terminal"
    elif projection in _TERMINAL_STATES:
        if existing and existing.get("classification") in {"queued_cancelled", "prelaunch_cancelled"}:
            classification = existing["classification"]
        else:
            classification = "already_terminal"
    elif projection == "running":
        if control.get("terminate_running"):
            pending_machine = _task_machine(task)
            classification = "termination_pending"
        else:
            classification = "running_allowed"
    else:
        classification = "blocked"
    contribution = _zero_progress()
    contribution["target_tasks"] = 1
    contribution[classification] = 1
    return contribution, classification, pending_machine


def _missing_task_contribution(
    cfg: RootConfig, member: MemberIdentity, operation_id: str, group_name: str
) -> tuple[dict[str, int], str]:
    from ..runtime.operation_store import locate_operation_path

    cleanup_path = locate_operation_path(cfg, "cleanup", member.task_id)
    if not cleanup_path.exists():
        raise _BlockedStep("missing_task_without_completed_cleanup")
    try:
        cleanup = read_json(cleanup_path).get("cleanup", {})
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise _BlockedStep(_reason("cleanup_proof_invalid", exc)) from exc
    if (
        cleanup.get("state") != "completed"
        or cleanup.get("task_id") != member.task_id
        or cleanup.get("group_name") != group_name
        or cleanup.get("submission_operation_id") != member.operation_id
    ):
        raise _BlockedStep("missing_task_without_matching_completed_cleanup")
    contribution = _zero_progress()
    witness = cleanup.get("group_discovery_outcome")
    if witness is not None:
        from ..runtime.records import TaskRecord

        try:
            task = TaskRecord.from_dict(witness["task"])
        except (TypeError, KeyError, ValueError) as exc:
            raise _BlockedStep("cleanup_outcome_invalid") from exc
        if (
            task.task_id != member.task_id
            or task.group_name != group_name
            or task.submission_operation_id != member.operation_id
            or task.group_membership_sequence != member.sequence
            or task.state.get("projection") not in _TERMINAL_STATES
            or task.claim_control.get("active_claim")
        ):
            raise _BlockedStep("cleanup_outcome_identity_mismatch")
        if task.control.get("cancellation_operation_id") == operation_id:
            if task.state.get("reason") == "group_cancelled":
                contribution["queued_cancelled"] = 1
            elif task.state.get("reason") == "group_cancelled_before_launch":
                contribution["prelaunch_cancelled"] = 1
    return contribution, "missing_completed_cleanup"


def _audit_task_transition(cfg: RootConfig, task: Any, operation_id: str) -> str | None:
    claim = task.claim_control.get("active_claim") or {}
    projection = task.state.get("projection")
    if not claim:
        if projection == "running":
            return "partial_transition"
        return None
    attempt_number = task.attempt_control.get("current_attempt_number")
    attempt_id = claim.get("attempt_id")
    token = claim.get("fencing_token")
    if type(attempt_number) is not int or not isinstance(attempt_id, str) or type(token) is not int:
        return "partial_transition"
    try:
        attempt = AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task.task_id, attempt_number)))
    except FileNotFoundError:
        if claim.get("launch_state") == "claimed":
            return None
        return "partial_transition"
    except (OSError, KeyError, TypeError, ValueError):
        return "partial_transition"
    if (
        attempt.task_id != task.task_id
        or attempt.attempt_id != attempt_id
        or attempt.current_fencing_token != token
        or attempt.attempt_number != attempt_number
    ):
        return "partial_transition"
    if attempt.phase in _TERMINAL_STATES or attempt.phase == "orphaned":
        if (
            claim.get("launch_state") == "claimed"
            and attempt.phase == "cancelled"
            and attempt.result.get("reason") == "group_cancelled_before_launch"
            and attempt.result.get("cancellation_operation_id") == operation_id
        ):
            return None
        return "partial_transition"
    if projection == "blocked" and (attempt.phase == "running" or attempt.current_fencing_token > token):
        return "partial_recovery"
    return None


def _expected_effect(task: Any, control: dict[str, Any]) -> str | None:
    if is_cleanup_blocked(task):
        return None
    claim = task.claim_control.get("active_claim") or {}
    if task.state.get("projection") == "queued" and not claim:
        return "queued_cancelled"
    if claim.get("launch_state") == "claimed":
        return "prelaunch_cancelled"
    return None


def _event_targets_group(
    cfg: RootConfig,
    coverage: GroupCoverage,
    group_name: str,
    high_watermark: int,
    event: dict[str, Any],
) -> bool:
    sequence = event.get("membership_sequence")
    if type(sequence) is not int or sequence < 1:
        raise _BlockedStep("recheck_membership_sequence_invalid")
    if sequence > high_watermark:
        return False
    evidence = event.get("evidence")
    if isinstance(evidence, dict):
        event_group = evidence.get("group_name", evidence.get("group"))
        if event_group is not None and event_group != group_name:
            return False
    try:
        member = coverage.read_member(sequence)
    except BlockingIOError as exc:
        raise _BlockedStep("membership_coverage_busy") from exc
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        raise _BlockedStep(_reason("membership_provenance_invalid", exc)) from exc
    if not (
        member.task_id == event.get("task_id")
        and member.operation_id == event.get("submission_operation_id")
        and member.sequence == sequence
    ):
        raise _BlockedStep("recheck_membership_identity_mismatch")
    return True


def _member_for_event(cfg: RootConfig, coverage: GroupCoverage, event: dict[str, Any]) -> MemberIdentity:
    sequence = event.get("membership_sequence")
    try:
        member = coverage.read_member(sequence)
    except BlockingIOError as exc:
        raise _BlockedStep("membership_coverage_busy") from exc
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        raise _BlockedStep(_reason("membership_provenance_invalid", exc)) from exc
    if member.task_id != event.get("task_id") or member.operation_id != event.get("submission_operation_id"):
        raise _BlockedStep("recheck_membership_identity_mismatch")
    return member


def _settle_event(cfg: RootConfig, event: dict[str, Any]) -> tuple[bool, list[Any]]:
    from ..runtime.group_discovery.changes import settle_task_change

    task_id = event.get("task_id")
    if not isinstance(task_id, str):
        raise _BlockedStep("recheck_task_id_invalid")
    with task_lock(cfg.shared_root, task_id):
        settled, results = settle_task_change(cfg, event)
    return bool(settled), list(results or [])


def _commit_receipt(
    cfg: RootConfig,
    operation: dict[str, Any],
    control: dict[str, Any],
    discovery: dict[str, Any],
    member: MemberIdentity,
    intent: dict[str, Any],
    contribution: dict[str, int],
    classification: str,
    pending_machine: str | None,
    *,
    task_revision: int | None = None,
    journal_sequence: int | None = None,
) -> None:
    operation_id = control["operation_id"]
    if task_revision is None:
        task_revision = intent.get("before_task_revision")
    if type(task_revision) is not int or task_revision < 0:
        task_revision = 0
    receipt = {
        "version": _RECEIPT_VERSION,
        "operation_id": operation_id,
        "receipt_generation": discovery["receipt_generation"],
        "task_id": member.task_id,
        "submission_operation_id": member.operation_id,
        "membership_sequence": member.sequence,
        "task_revision": task_revision,
        "classification": classification,
        "pending_machine": pending_machine,
        "contribution": dict(contribution),
    }
    old = _read_receipt(cfg, control, discovery, member.sequence)
    if old is not None:
        for key in _HISTORICAL_EFFECT_KEYS:
            receipt["contribution"][key] = max(receipt["contribution"][key], old["contribution"][key])
    _apply_contribution_delta(control, discovery, old, receipt)
    if intent.get("source") == "recovery":
        next_member_cursor = discovery["member_cursor"]
        next_journal_cursor = discovery["journal_cursor"]
    elif intent.get("source") == "journal":
        next_member_cursor = discovery["member_cursor"]
        next_journal_cursor = journal_sequence if journal_sequence is not None else intent.get("journal_sequence")
    else:
        next_member_cursor = max(discovery["member_cursor"], member.sequence)
        next_journal_cursor = discovery["journal_cursor"]
    pending = {"sequence": member.sequence, "receipt": receipt}
    discovery["pending_receipt"] = pending
    discovery["current_member"] = None
    discovery["member_cursor"] = next_member_cursor
    if next_journal_cursor is not None:
        discovery["journal_cursor"] = next_journal_cursor
    _persist_operation(cfg, operation, control)
    _write_receipt(cfg, control, discovery, member.sequence, receipt)
    discovery["pending_receipt"] = None
    _persist_operation(cfg, operation, control)


def _finish_pending_receipt(
    cfg: RootConfig,
    operation: dict[str, Any],
    control: dict[str, Any],
    discovery: dict[str, Any],
) -> None:
    pending = discovery.get("pending_receipt")
    if not isinstance(pending, dict) or type(pending.get("sequence")) is not int:
        raise ValueError("pending receipt is malformed")
    receipt = pending.get("receipt")
    _validate_receipt(receipt, control["operation_id"], discovery["receipt_generation"], pending["sequence"])
    _write_receipt(cfg, control, discovery, pending["sequence"], receipt)
    discovery["pending_receipt"] = None
    _persist_operation(cfg, operation, control)


def _read_receipt(
    cfg: RootConfig, control: dict[str, Any], discovery: dict[str, Any], sequence: int
) -> dict[str, Any] | None:
    path = _receipt_path(cfg, control, discovery, sequence)
    try:
        value = read_json(path)
    except FileNotFoundError:
        return None
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise _BlockedStep(_reason("receipt_malformed", exc)) from exc
    try:
        _validate_receipt(value, control["operation_id"], discovery["receipt_generation"], sequence)
    except (TypeError, ValueError, KeyError) as exc:
        raise _BlockedStep(_reason("receipt_malformed", exc)) from exc
    return value


def _write_receipt(
    cfg: RootConfig,
    control: dict[str, Any],
    discovery: dict[str, Any],
    sequence: int,
    receipt: dict[str, Any],
) -> None:
    path = _receipt_path(cfg, control, discovery, sequence)
    existing = None
    try:
        existing = read_json(path)
    except FileNotFoundError:
        pass
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("receipt path is malformed") from exc
    if existing is not None:
        _validate_receipt(existing, control["operation_id"], discovery["receipt_generation"], sequence)
    atomic_replace(path, receipt)


def _receipt_path(cfg: RootConfig, control: dict[str, Any], discovery: dict[str, Any], sequence: int) -> Path:
    if type(sequence) is not int or sequence < 1:
        raise ValueError("receipt sequence is invalid")
    operation_id = control.get("operation_id")
    generation = discovery.get("receipt_generation")
    if not isinstance(operation_id, str) or not _valid_identifier(operation_id):
        raise ValueError("receipt operation identifier is invalid")
    if not isinstance(generation, str) or not _valid_identifier(generation):
        raise ValueError("receipt generation identifier is invalid")
    coverage = GroupCoverage(cfg.shared_root, control["group_name"])
    directory = coverage.directory
    for path in (
        directory,
        directory / "operations",
        directory / "operations" / operation_id,
        directory / "operations" / operation_id / generation,
    ):
        _ensure_directory(path)
    _sync_directory_chain(directory / "operations" / operation_id / generation, directory)
    return directory / "operations" / operation_id / generation / f"{sequence}.json"


def _validate_receipt(value: Any, operation_id: str, generation: str, sequence: int) -> None:
    if not isinstance(value, dict):
        raise ValueError("receipt is not an object")
    if value.get("version") != _RECEIPT_VERSION:
        raise ValueError("receipt version is invalid")
    if value.get("operation_id") != operation_id or value.get("receipt_generation") != generation:
        raise ValueError("receipt identity is invalid")
    if value.get("membership_sequence") != sequence:
        raise ValueError("receipt membership sequence is invalid")
    for key in ("task_id", "submission_operation_id", "classification"):
        if not isinstance(value.get(key), str):
            raise ValueError(f"receipt {key} is invalid")
    if value.get("pending_machine") is not None and not isinstance(value["pending_machine"], str):
        raise ValueError("receipt pending machine is invalid")
    if type(value.get("task_revision")) is not int or value["task_revision"] < 0:
        raise ValueError("receipt Task revision is invalid")
    _validate_progress(value.get("contribution"), "receipt contribution")


def _apply_contribution_delta(
    control: dict[str, Any],
    discovery: dict[str, Any],
    old: dict[str, Any] | None,
    new: dict[str, Any],
) -> None:
    progress = _ensure_progress(control)
    current = discovery.setdefault("current_progress", _zero_progress())
    old_contribution = old.get("contribution", _zero_progress()) if old else _zero_progress()
    new_contribution = new.get("contribution", _zero_progress())
    for key in _PROGRESS_KEYS:
        current[key] += new_contribution.get(key, 0) - old_contribution.get(key, 0)
        if current[key] < 0:
            raise ValueError("receipt contribution counter underflow")
        if not discovery.get("is_legacy_adoption") or key not in _HISTORICAL_EFFECT_KEYS:
            progress[key] = current[key]
    pending = control.setdefault("pending_machine_acknowledgements", {})
    if old and old.get("pending_machine"):
        _remove_pending(pending, old["pending_machine"], old.get("task_id"))
    if new.get("pending_machine"):
        machine = new["pending_machine"]
        values = pending.setdefault(machine, [])
        if new["task_id"] not in values:
            values.append(new["task_id"])
            values.sort()


def _remove_pending(pending: dict[str, Any], machine: str, task_id: str | None) -> None:
    values = pending.get(machine)
    if not isinstance(values, list):
        pending.pop(machine, None)
        return
    pending[machine] = [value for value in values if value != task_id]
    if not pending[machine]:
        pending.pop(machine, None)


def _finish_step(
    cfg: RootConfig,
    operation: dict[str, Any],
    control: dict[str, Any],
    group_data: dict[str, Any],
    terminal_results: list[Any],
) -> dict[str, Any]:
    del terminal_results
    discovery = control["discovery"]
    progress = _ensure_progress(control)
    if discovery.get("pending_receipt") is not None or discovery.get("current_member") is not None:
        control["state"] = "converging"
        control["completed_at"] = None
    elif progress["blocked"]:
        control["state"] = "blocked"
        control["completed_at"] = None
        control["blocked_reason"] = "task_cancellation_requires_resolution"
    elif progress["termination_pending"]:
        control["state"] = "waiting_ack"
        control["completed_at"] = None
        control["blocked_reason"] = None
    else:
        coverage = GroupCoverage(cfg.shared_root, control["group_name"])
        status = coverage.status()
        position = _recheck_position(cfg, control["group_name"])
        complete = (
            discovery["member_cursor"] >= control["membership_high_watermark"]
            and position is not None
            and discovery["generation"] == _position_value(position, "generation")
            and discovery["journal_cursor"] == _position_value(position, "tail")
            and _coverage_allows(status, control["membership_high_watermark"])
            and not progress["blocked"]
            and not progress["termination_pending"]
        )
        if complete:
            control["state"] = "completed"
            control["completed_at"] = control.get("completed_at") or utc_now()
            control["blocked_reason"] = None
        else:
            control["state"] = "converging"
            control["completed_at"] = None
            control["blocked_reason"] = None
    _persist_operation(cfg, operation, control)
    if control.get("state") == "completed" and discovery.get("pending_receipt") is None:
        archive_operation(cfg, "group_control", control["operation_id"], operation)
    _update_group_snapshot(cfg, group_data, control, group_path(cfg.shared_root, control["group_name"]))
    return control


def _recheck_position(cfg: RootConfig, group_name: str) -> Any | None:
    from ..runtime.group_discovery.rechecks import GroupRechecks

    try:
        return GroupRechecks(cfg.shared_root, group_name).snapshot()
    except (OSError, RuntimeError, TypeError, ValueError, KeyError):
        return None


def _persist_operation(cfg: RootConfig, operation: dict[str, Any], control: dict[str, Any]) -> None:
    now = utc_now()
    control["updated_at"] = now
    meta = operation.setdefault("meta", {})
    meta["revision"] = int(meta.get("revision", 0)) + 1
    meta["updated_at"] = now
    write_active_operation(cfg, "group_control", control["operation_id"], operation)


def _update_group_snapshot(cfg: RootConfig, group_data: dict[str, Any], control: dict[str, Any], path: Path) -> None:
    snapshot = group_data.get("cancellation_operation") or {}
    if snapshot.get("operation_id") != control.get("operation_id"):
        return
    group_data["cancellation_operation"] = control
    group_data["meta"]["revision"] += 1
    group_data["meta"]["updated_at"] = utc_now()
    atomic_replace(path, group_data)


def _block_operation(
    cfg: RootConfig,
    operation: dict[str, Any],
    control: dict[str, Any],
    reason: str,
    *,
    update_group: bool = True,
) -> dict[str, Any]:
    control["state"] = "blocked"
    control["completed_at"] = None
    control["blocked_reason"] = reason
    _persist_operation(cfg, operation, control)
    if update_group:
        name = control.get("group_name")
        if isinstance(name, str):
            path = group_path(cfg.shared_root, name)
            if path.exists():
                try:
                    data = read_json(path)
                    normalize_group_record(data)
                    _update_group_snapshot(cfg, data, control, path)
                except (OSError, KeyError, TypeError, ValueError):
                    pass
    return control


def _coverage_allows(status: Any, high_watermark: int) -> bool:
    prefix = _status_value(status, "prefix", -1)
    reason = _status_value(status, "reason")
    if type(prefix) is not int or prefix < high_watermark:
        return False
    return reason in {None, "pending_submission"}


def _coverage_reason(status: Any, high_watermark: int) -> str:
    reason = _status_value(status, "reason")
    if reason == "busy":
        return "membership_coverage_busy"
    if reason in {"invalid_state", "invalid_member_slot", "invalid_group"}:
        return f"membership_coverage_{reason}"
    if _status_value(status, "prefix", 0) < high_watermark:
        return "membership_coverage_pending"
    return str(reason or "membership_coverage_pending")


def _has_exact_barrier(
    group_data: dict[str, Any], operation_id: str, high_watermark: int, terminate_running: bool
) -> bool:
    barriers = group_data.get("group", {}).get("cancellation_barriers", [])
    if not isinstance(barriers, list):
        return False
    return any(
        isinstance(item, dict)
        and item.get("operation_id") == operation_id
        and item.get("membership_high_watermark") == high_watermark
        and item.get("terminate_running") is terminate_running
        for item in barriers
    )


def _member_is_valid(member: Any, sequence: int) -> bool:
    return (
        isinstance(member, MemberIdentity)
        and member.sequence == sequence
        and isinstance(member.task_id, str)
        and isinstance(member.operation_id, str)
        and _valid_identifier(member.task_id)
        and _valid_identifier(member.operation_id)
    )


def _position_value(position: Any, key: str) -> int:
    value = position.get(key) if isinstance(position, dict) else getattr(position, key, None)
    if type(value) is not int or value < 0:
        raise ValueError(f"recheck position {key} is invalid")
    return value


def _status_value(status: Any, key: str, default: Any = None) -> Any:
    return status.get(key, default) if isinstance(status, dict) else getattr(status, key, default)


def _task_machine(task: Any) -> str:
    claim = task.claim_control.get("active_claim") or {}
    return claim.get("machine_name") or task.placement_policy.get("home_machine")


def _ensure_progress(control: dict[str, Any]) -> dict[str, int]:
    progress = control.setdefault("progress", {})
    for key in _PROGRESS_KEYS:
        value = progress.get(key, 0)
        if type(value) is not int or value < 0:
            raise ValueError(f"cancellation progress {key} is invalid")
        progress[key] = value
    control.setdefault("pending_machine_acknowledgements", {})
    if not isinstance(control["pending_machine_acknowledgements"], dict):
        raise ValueError("pending machine acknowledgements are invalid")
    return progress


def _zero_progress() -> dict[str, int]:
    return {key: 0 for key in _PROGRESS_KEYS}


def _validate_progress(progress: Any, label: str) -> None:
    if not isinstance(progress, dict):
        raise ValueError(f"{label} is invalid")
    for key in _PROGRESS_KEYS:
        value = progress.get(key, 0)
        if type(value) is not int or value < 0:
            raise ValueError(f"{label} {key} is invalid")


def _valid_identifier(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        validate_identifier(value, "identifier")
    except ValueError:
        return False
    return True


def _ensure_directory(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        path.mkdir(parents=True, exist_ok=True)
        info = path.lstat()
    if not info or not os.path.isdir(path) or path.is_symlink():
        raise ValueError(f"receipt directory is not a real directory: {path}")


def _sync_directory_chain(leaf: Path, coverage_directory: Path) -> None:
    current = leaf
    while True:
        _sync_directory(current)
        if current == coverage_directory:
            break
        if current.parent == current:
            raise ValueError("receipt directory escaped coverage directory")
        current = current.parent


def _sync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_optional_operation(path: Path) -> dict[str, Any] | None:
    try:
        return read_json(path)
    except FileNotFoundError:
        return None
    except (OSError, KeyError, TypeError, ValueError):
        return None


def _unwritable_blocked_summary(operation: dict[str, Any], reason: str) -> dict[str, Any]:
    control = operation.get("group_control")
    if isinstance(control, dict):
        control["state"] = "blocked"
        control["completed_at"] = None
        control["blocked_reason"] = reason
        return control
    return {"state": "blocked", "blocked_reason": reason, "completed_at": None}


def _reason(prefix: str, exc: BaseException) -> str:
    detail = str(exc).strip().replace("\n", " ")
    return f"{prefix}:{detail}" if detail else prefix


class _BlockedStep(RuntimeError):
    """A bounded step cannot safely classify its current record."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


__all__ = ["advance_indexed_cancel", "initialize_cancel_discovery"]
