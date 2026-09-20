"""Bounded indexed convergence for canonical Worker removal operations.

The canonical Group namespace owns the membership coverage and ordered recheck
journal.  This module consumes those two projections under the caller's Group
writer fence; it never scans the Task directory.
"""

from __future__ import annotations

from typing import Any

from ..config_types import RootConfig
from ..domain.policies import task_machine_matches
from ..lifecycle import TerminalCommitResult
from ..runtime.group_discovery.changes import record_task_change, settle_task_change
from ..runtime.group_discovery.coverage import GroupCoverage, MemberIdentity
from ..runtime.group_discovery.rechecks import GroupRechecks
from ..runtime.group_namespace import group_authority_identity
from ..runtime.locks import task_lock
from ..runtime.operation_store import locate_operation_path
from ..runtime.paths import attempt_path
from ..runtime.records import AttemptRecord, TaskRecord, validate_identifier
from ..runtime.store import read_json
from ..runtime.tasks import load_task, save_task

_DISCOVERY_VERSION = 1
_LANE_COUNT = 4
_TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})
_PENDING_STATES = frozenset({"preparing", "converging", "waiting_ack", "blocked"})
_DISCOVERY_KEYS = frozenset(
    {
        "version",
        "authority",
        "initial_high_watermark",
        "member_cursor",
        "generation",
        "journal_cursor",
        "sensitive",
        "policy_epoch",
        "policy_pending",
        "policy_target_epoch",
        "terminate_running",
        "pending_cursor",
        "sensitive_cursor",
        "next_lane",
    }
)
_RECEIPT_KEYS = frozenset(
    {
        "task_id",
        "submission_operation_id",
        "sequence",
        "task_revision",
        "attempt_id",
        "attempt_number",
        "fencing_token",
        "machine_name",
        "classification",
        "is_blocker",
    }
)


class _BlockedStep(RuntimeError):
    """A bounded consumer step cannot safely classify its current truth."""

    def __init__(self, reason: str, *, retain: bool = False, classification: str = "blocked") -> None:
        super().__init__(reason)
        self.reason = reason
        self.retain = retain
        self.classification = classification


def initialize_removal_discovery(cfg: RootConfig, control: dict[str, Any], data: dict[str, Any]) -> None:
    """Initialize the in-operation removal census and recheck position."""

    group = _group_record(data)
    group_name = _identifier(control.get("group_name"), "group_name")
    if group.get("name") != group_name:
        raise ValueError("worker removal Group identity does not match its control")
    _identifier(control.get("operation_id"), "operation_id")
    _identifier(control.get("machine_name"), "machine_name")
    terminate_running = control.get("terminate_running")
    if type(terminate_running) is not bool:
        raise ValueError("worker removal terminate_running is invalid")
    epoch = _counter(group.get("worker_set_epoch"), "worker_set_epoch")
    high_watermark = _high_watermark(group)
    position = GroupRechecks(cfg.shared_root, group_name).snapshot(initialize=True)
    if position is None:
        raise RuntimeError("Group recheck journal did not initialize")
    authority = group_authority_identity(cfg.shared_root)
    control["discovery"] = _new_discovery(
        authority,
        position,
        high_watermark,
        epoch,
        terminate_running,
    )


def advance_removal_discovery_locked(
    cfg: RootConfig,
    control: dict[str, Any],
    data: dict[str, Any],
    *,
    allow_settlement: bool = True,
) -> tuple[bool, list[TerminalCommitResult]]:
    """Consume at most one certified member, journal event, or sensitive Task.

    The caller owns the uninterrupted schema and Group locks and persists the
    mutated control record after this function returns.  This function acquires
    at most one Task lock and returns typed terminal results for post-lock
    dispatch.
    """

    if type(allow_settlement) is not bool:
        raise TypeError("allow_settlement must be a bool")
    results: list[TerminalCommitResult] = []
    try:
        group = _group_record(data)
        group_name = _identifier(control.get("group_name"), "group_name")
        if group.get("name") != group_name:
            raise _BlockedStep("worker_removal_group_identity_mismatch")
        if control.get("operation_type") != "worker_remove_v2":
            raise _BlockedStep("worker_removal_operation_type_invalid")
        if control.get("state") not in _PENDING_STATES:
            return False, results
        machine = _identifier(control.get("machine_name"), "machine_name")
        terminate_running = control.get("terminate_running")
        if type(terminate_running) is not bool:
            raise _BlockedStep("worker_removal_control_invalid")
        discovery = control.get("discovery")
        if type(discovery) is not dict:
            raise _BlockedStep("worker_removal_discovery_missing")
        authority = group_authority_identity(cfg.shared_root)
        _validate_discovery(discovery, authority, group_name, _high_watermark(group))
        current_epoch = _counter(group.get("worker_set_epoch"), "worker_set_epoch")
        current_high_watermark = _high_watermark(group)
        journal = GroupRechecks(cfg.shared_root, group_name)
        position = journal.snapshot()
        if position is None:
            raise _BlockedStep("recheck_journal_missing")
        if position.generation != discovery["generation"]:
            _reset_discovery(discovery, authority, position, current_high_watermark, current_epoch, terminate_running)
            _set_converging(control, "recheck_generation_changed")
            _refresh_blockers(control, discovery)
            return False, results
        if position.tail < discovery["journal_cursor"]:
            raise _BlockedStep("recheck_cursor_invalid")

        coverage = GroupCoverage(cfg.shared_root, group_name)
        status = coverage.status()
        coverage_reason = _coverage_reason(status, current_high_watermark)
        if coverage_reason is not None and coverage_reason.startswith("membership_coverage_invalid"):
            raise _BlockedStep(coverage_reason)

        completed_policy = _finish_policy_pass_if_ready(discovery, current_epoch, terminate_running)
        started_policy = False
        if not completed_policy and not discovery["policy_pending"]:
            if discovery["policy_epoch"] != current_epoch or discovery["terminate_running"] != terminate_running:
                _start_policy_pass(discovery, current_epoch)
                started_policy = True
                if not discovery["policy_pending"]:
                    discovery["policy_epoch"] = current_epoch
                    discovery["policy_target_epoch"] = current_epoch
                    discovery["terminate_running"] = terminate_running

        member_ready = (
            discovery["member_cursor"] < current_high_watermark
            and status.prefix >= discovery["member_cursor"] + 1
            and status.reason in {None, "pending_submission"}
        )
        journal_ready = discovery["journal_cursor"] < position.tail
        policy_ready = discovery["pending_cursor"] < len(discovery["policy_pending"])
        lane = _select_lane(
            discovery["next_lane"], member_ready, journal_ready, policy_ready, bool(discovery["sensitive"])
        )
        if lane is None:
            _refresh_blockers(control, discovery)
            if coverage_reason is not None:
                _set_converging(control, coverage_reason)
            elif discovery["policy_pending"] or discovery["member_cursor"] < current_high_watermark:
                _set_converging(control, "membership_coverage_pending" if not member_ready else None)
            else:
                _set_converging(control, None)
            return _final_proof(
                cfg,
                control,
                data,
                discovery,
                journal,
                coverage,
                current_high_watermark,
                current_epoch,
                terminate_running,
                results,
            )

        discovery["next_lane"] = (lane + 1) % _LANE_COUNT
        if lane == 0:
            sequence = discovery["member_cursor"] + 1
            member = _read_member(coverage, sequence, group_name)
            _consume_member_locked(cfg, control, data, discovery, member, group_name)
            discovery["member_cursor"] = sequence
        elif lane == 1:
            _consume_journal_locked(
                cfg,
                control,
                data,
                discovery,
                journal,
                position,
                coverage,
                group_name,
                machine,
                terminate_running,
                allow_settlement,
                results,
            )
        elif lane == 2:
            _consume_policy_locked(cfg, control, data, discovery, coverage, group_name)
        else:
            if not (member_ready or journal_ready or policy_ready):
                _consume_sensitive_locked(cfg, control, data, discovery, coverage, group_name)
            else:
                # A sensitive lane is only runnable when all other lanes are
                # empty.  Recompute the next invocation from the durable state.
                _set_converging(control, None)

        if started_policy and not discovery["policy_pending"]:
            _finish_policy_pass_if_ready(discovery, current_epoch, terminate_running)
        _refresh_blockers(control, discovery)
        return _final_proof(
            cfg,
            control,
            data,
            discovery,
            journal,
            coverage,
            current_high_watermark,
            current_epoch,
            terminate_running,
            results,
        )
    except _BlockedStep as exc:
        discovery = control.get("discovery")
        if isinstance(discovery, dict):
            _refresh_blockers(control, discovery)
            _set_blocked(control, exc.reason)
        else:
            control.update({"state": "blocked", "blocked_reason": exc.reason, "completed_at": None})
        return False, results
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        discovery = control.get("discovery")
        reason = _reason("worker_removal_discovery_failed", exc)
        if isinstance(discovery, dict):
            _refresh_blockers(control, discovery)
            _set_blocked(control, reason)
        else:
            control.update({"state": "blocked", "blocked_reason": reason, "completed_at": None})
        return False, results


def _group_record(data: dict[str, Any]) -> dict[str, Any]:
    group = data.get("group") if isinstance(data, dict) else None
    if type(group) is not dict:
        raise _BlockedStep("group_malformed")
    name = group.get("name")
    if type(name) is not str or not name:
        raise _BlockedStep("group_malformed")
    if type(group.get("worker_set")) is not dict:
        raise _BlockedStep("group_malformed")
    _counter(group.get("next_membership_sequence"), "next_membership_sequence", allow_zero=False)
    _counter(group.get("worker_set_epoch"), "worker_set_epoch")
    pending = group.get("pending_submission_commit")
    if pending is not None and type(pending) is not dict:
        raise _BlockedStep("group_pending_submission_invalid")
    return group


def _new_discovery(
    authority: dict[str, Any],
    position: Any,
    high_watermark: int,
    epoch: int,
    terminate_running: bool,
) -> dict[str, Any]:
    generation = _position_value(position, "generation")
    tail = _position_value(position, "tail")
    return {
        "version": _DISCOVERY_VERSION,
        "authority": dict(authority),
        "initial_high_watermark": high_watermark,
        "member_cursor": 0,
        "generation": generation,
        "journal_cursor": tail,
        "sensitive": {},
        "policy_epoch": epoch,
        "policy_pending": [],
        "policy_target_epoch": epoch,
        "terminate_running": terminate_running,
        "pending_cursor": 0,
        "sensitive_cursor": 0,
        "next_lane": 0,
    }


def _reset_discovery(
    discovery: dict[str, Any],
    authority: dict[str, Any],
    position: Any,
    high_watermark: int,
    epoch: int,
    terminate_running: bool,
) -> None:
    discovery.clear()
    discovery.update(_new_discovery(authority, position, high_watermark, epoch, terminate_running))


def _validate_discovery(
    discovery: dict[str, Any], authority: dict[str, Any], group_name: str, current_high_watermark: int
) -> None:
    if frozenset(discovery) != _DISCOVERY_KEYS:
        raise _BlockedStep("worker_removal_discovery_malformed")
    if discovery.get("version") != _DISCOVERY_VERSION or discovery.get("authority") != authority:
        raise _BlockedStep("worker_removal_discovery_identity_mismatch")
    for key in (
        "initial_high_watermark",
        "member_cursor",
        "generation",
        "journal_cursor",
        "policy_epoch",
        "policy_target_epoch",
        "pending_cursor",
        "sensitive_cursor",
        "next_lane",
    ):
        value = discovery.get(key)
        if type(value) is not int or value < 0:
            raise _BlockedStep(f"worker_removal_discovery_{key}_invalid")
    if discovery["generation"] < 1:
        raise _BlockedStep("worker_removal_discovery_generation_invalid")
    if discovery["initial_high_watermark"] > current_high_watermark:
        raise _BlockedStep("worker_removal_discovery_high_watermark_invalid")
    if discovery["member_cursor"] > current_high_watermark:
        raise _BlockedStep("worker_removal_discovery_member_cursor_invalid")
    if discovery["next_lane"] >= _LANE_COUNT:
        raise _BlockedStep("worker_removal_discovery_lane_invalid")
    if type(discovery.get("terminate_running")) is not bool:
        raise _BlockedStep("worker_removal_discovery_terminate_invalid")
    sensitive = discovery.get("sensitive")
    if type(sensitive) is not dict:
        raise _BlockedStep("worker_removal_discovery_sensitive_invalid")
    for key, entry in sensitive.items():
        if type(key) is not str or not key.isdecimal() or int(key) < 1 or str(int(key)) != key:
            raise _BlockedStep("worker_removal_discovery_sensitive_key_invalid")
        _validate_receipt(entry, int(key), group_name)
    pending = discovery.get("policy_pending")
    if type(pending) is not list or any(
        type(key) is not str or not key.isdecimal() or int(key) < 1 or str(int(key)) != key for key in pending
    ):
        raise _BlockedStep("worker_removal_discovery_policy_pending_invalid")
    if len(set(pending)) != len(pending):
        raise _BlockedStep("worker_removal_discovery_policy_pending_duplicate")
    if discovery["pending_cursor"] > len(pending):
        raise _BlockedStep("worker_removal_discovery_pending_cursor_invalid")


def _validate_receipt(entry: Any, sequence: int, group_name: str) -> None:
    if type(entry) is not dict or frozenset(entry) != _RECEIPT_KEYS:
        raise _BlockedStep("worker_removal_sensitive_receipt_malformed")
    if entry.get("sequence") != sequence:
        raise _BlockedStep("worker_removal_sensitive_receipt_sequence_invalid")
    for key in ("task_id", "submission_operation_id", "classification"):
        try:
            _identifier(entry.get(key), f"sensitive {key}")
        except ValueError as exc:
            raise _BlockedStep("worker_removal_sensitive_receipt_identity_invalid") from exc
    if type(entry.get("task_revision")) is not int or entry["task_revision"] < 0:
        raise _BlockedStep("worker_removal_sensitive_receipt_revision_invalid")
    for key in ("attempt_number", "fencing_token"):
        value = entry.get(key)
        if value is not None and (type(value) is not int or value < 0):
            raise _BlockedStep("worker_removal_sensitive_receipt_attempt_invalid")
    for key in ("attempt_id", "machine_name"):
        value = entry.get(key)
        if value is not None:
            try:
                _identifier(value, f"sensitive {key}")
            except ValueError as exc:
                raise _BlockedStep("worker_removal_sensitive_receipt_attempt_invalid") from exc
    if type(entry.get("is_blocker")) is not bool or not entry.get("classification"):
        raise _BlockedStep("worker_removal_sensitive_receipt_classification_invalid")


def _read_member(coverage: GroupCoverage, sequence: int, group_name: str) -> MemberIdentity:
    try:
        member = coverage.read_member(sequence)
    except BlockingIOError as exc:
        raise _BlockedStep("membership_coverage_busy") from exc
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        raise _BlockedStep(_reason("membership_provenance_invalid", exc)) from exc
    if not isinstance(member, MemberIdentity) or member.sequence != sequence:
        raise _BlockedStep("membership_identity_invalid")
    try:
        _identifier(member.task_id, "member task_id")
        _identifier(member.operation_id, "member submission_operation_id")
    except ValueError as exc:
        raise _BlockedStep("membership_identity_invalid") from exc
    del group_name
    return member


def _consume_member_locked(
    cfg: RootConfig,
    control: dict[str, Any],
    data: dict[str, Any],
    discovery: dict[str, Any],
    member: MemberIdentity,
    group_name: str,
) -> None:
    with task_lock(cfg.shared_root, member.task_id):
        _classify_locked(cfg, control, data, discovery, member, group_name)


def _consume_sensitive_locked(
    cfg: RootConfig,
    control: dict[str, Any],
    data: dict[str, Any],
    discovery: dict[str, Any],
    coverage: GroupCoverage,
    group_name: str,
) -> None:
    keys = sorted(discovery["sensitive"], key=int)
    if not keys:
        _set_converging(control, None)
        return
    key = next((key for key in keys if int(key) > discovery["sensitive_cursor"]), keys[0])
    # Persist the round-robin position even if this Task cannot be classified.
    # JSON key sorting cannot encode rotation through insertion order.
    discovery["sensitive_cursor"] = int(key)
    entry = discovery["sensitive"].get(key)
    sequence = int(key)
    if type(entry) is not dict:
        raise _BlockedStep("worker_removal_sensitive_receipt_malformed")
    member = _read_member(coverage, sequence, group_name)
    if member.task_id != entry["task_id"] or member.operation_id != entry["submission_operation_id"]:
        raise _BlockedStep("worker_removal_sensitive_membership_identity_mismatch")
    with task_lock(cfg.shared_root, member.task_id):
        _classify_locked(cfg, control, data, discovery, member, group_name)


def _consume_policy_locked(
    cfg: RootConfig,
    control: dict[str, Any],
    data: dict[str, Any],
    discovery: dict[str, Any],
    coverage: GroupCoverage,
    group_name: str,
) -> None:
    key = discovery["policy_pending"][discovery["pending_cursor"]]
    entry = discovery["sensitive"].get(key)
    if entry is None:
        discovery["pending_cursor"] += 1
        _finish_policy_pass_if_ready(
            discovery,
            _counter(data["group"].get("worker_set_epoch"), "worker_set_epoch"),
            bool(control["terminate_running"]),
        )
        return
    sequence = int(key)
    member = _read_member(coverage, sequence, group_name)
    if member.task_id != entry["task_id"] or member.operation_id != entry["submission_operation_id"]:
        raise _BlockedStep("worker_removal_sensitive_membership_identity_mismatch")
    with task_lock(cfg.shared_root, member.task_id):
        _classify_locked(cfg, control, data, discovery, member, group_name)
    discovery["pending_cursor"] += 1
    _finish_policy_pass_if_ready(
        discovery,
        _counter(data["group"].get("worker_set_epoch"), "worker_set_epoch"),
        bool(control["terminate_running"]),
    )


def _consume_journal_locked(
    cfg: RootConfig,
    control: dict[str, Any],
    data: dict[str, Any],
    discovery: dict[str, Any],
    journal: GroupRechecks,
    position: Any,
    coverage: GroupCoverage,
    group_name: str,
    machine: str,
    terminate_running: bool,
    allow_settlement: bool,
    results: list[TerminalCommitResult],
) -> None:
    sequence = discovery["journal_cursor"] + 1
    try:
        event = journal.read(position, sequence)
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        raise _BlockedStep(_reason("recheck_event_invalid", exc)) from exc
    membership_sequence = event.get("membership_sequence")
    if type(membership_sequence) is not int or membership_sequence < 1:
        raise _BlockedStep("recheck_membership_sequence_invalid")
    member = _read_member(coverage, membership_sequence, group_name)
    if member.task_id != event.get("task_id") or member.operation_id != event.get("submission_operation_id"):
        raise _BlockedStep("recheck_membership_identity_mismatch")
    state = event.get("state")
    if state == "in_flight":
        if not allow_settlement:
            _set_converging(control, "transition_in_flight")
            return
        with task_lock(cfg.shared_root, member.task_id):
            settled, settled_results = settle_task_change(cfg, event)
            results.extend(settled_results)
            if not settled:
                raise _BlockedStep("transition_in_flight")
            latest = journal.snapshot()
            if latest is None or latest.generation != discovery["generation"]:
                raise _BlockedStep("recheck_generation_changed")
            try:
                event = journal.read(latest, sequence)
            except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
                raise _BlockedStep(_reason("recheck_event_invalid", exc)) from exc
            if event.get("state") not in {"committed", "aborted"}:
                raise _BlockedStep("transition_in_flight")
            _classify_locked(
                cfg,
                control,
                data,
                discovery,
                member,
                group_name,
                task_already_locked=True,
            )
    elif state in {"committed", "aborted"}:
        with task_lock(cfg.shared_root, member.task_id):
            _classify_locked(cfg, control, data, discovery, member, group_name)
    else:
        raise _BlockedStep("recheck_event_state_invalid")
    del machine, terminate_running
    discovery["journal_cursor"] = sequence


def _classify_locked(
    cfg: RootConfig,
    control: dict[str, Any],
    data: dict[str, Any],
    discovery: dict[str, Any],
    member: MemberIdentity,
    group_name: str,
    *,
    task_already_locked: bool = True,
) -> None:
    task: TaskRecord | None
    try:
        task = load_task(cfg, member.task_id)
    except FileNotFoundError:
        if not task_already_locked:
            raise RuntimeError("missing Task was read without its Task lock")
        if _missing_task_is_proven(cfg, member, group_name):
            discovery["sensitive"].pop(str(member.sequence), None)
            return
        raise _BlockedStep("missing_task_without_matching_completed_cleanup")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise _BlockedStep(_reason("task_invalid", exc), retain=False) from exc

    if (
        task.group_name != group_name
        or task.task_id != member.task_id
        or task.submission_operation_id != member.operation_id
        or task.group_membership_sequence != member.sequence
    ):
        raise _BlockedStep("task_membership_identity_mismatch")
    try:
        attempt = _current_attempt(cfg, task)
        claim = task.claim_control.get("active_claim")
        if claim is not None and type(claim) is not dict:
            raise _BlockedStep("task_claim_coherence_invalid", retain=True, classification="partial_transition")
        projection = task.state.get("projection")
        if projection in _TERMINAL_STATES:
            if claim:
                raise _BlockedStep("terminal_task_has_claim", retain=True, classification="partial_transition")
            is_superseded_orphan = (
                attempt is not None
                and attempt.phase == "orphaned"
                and projection == "cancelled"
                and task.state.get("reason") == "cancelled_by_user"
                and task.claim_control.get("fencing_epoch", 0) > attempt.current_fencing_token
            )
            if attempt is not None and attempt.phase not in _TERMINAL_STATES and not is_superseded_orphan:
                raise _BlockedStep("task_attempt_coherence_invalid", retain=True, classification="partial_transition")
            discovery["sensitive"].pop(str(member.sequence), None)
            return

        if claim:
            _classify_active_claim(cfg, control, task, data, discovery, member, claim, attempt)
            return
        if projection == "running":
            raise _BlockedStep("task_claim_missing", retain=True, classification="partial_transition")
        if projection == "queued":
            is_superseded_orphan = (
                attempt is not None
                and attempt.phase == "orphaned"
                and task.state.get("reason") == "orphan_superseded_by_retry"
                and task.claim_control.get("fencing_epoch", 0) > attempt.current_fencing_token
            )
            is_withdrawn_claim = (
                task.attempt_control.get("current_attempt_id") is None
                and task.state.get("reason") == "attempt_materialization_failed"
            )
            if (
                attempt is not None
                and attempt.phase not in _TERMINAL_STATES
                and not (is_superseded_orphan or is_withdrawn_claim)
            ):
                raise _BlockedStep(
                    "queued_task_has_nonterminal_attempt", retain=True, classification="partial_transition"
                )
            _classify_queued(cfg, control, data, discovery, member, task)
            return
        if projection == "blocked" and attempt is not None and attempt.phase == "orphaned":
            # An orphan is detached from shared execution authority. Explicit retry
            # or locally proven recovery is a fresh journaled transition.
            discovery["sensitive"].pop(str(member.sequence), None)
            return
        if projection in {"blocked", "starting", "claimed"}:
            _set_sensitive(discovery, member, task, "nonterminal", True, attempt)
            raise _BlockedStep("task_state_requires_resolution", retain=True, classification="nonterminal")
        raise _BlockedStep("task_projection_invalid", retain=True, classification="partial_transition")
    except _BlockedStep as exc:
        if exc.retain:
            _set_sensitive(
                discovery, member, task, exc.classification, True, attempt if "attempt" in locals() else None
            )
        raise


def _classify_active_claim(
    cfg: RootConfig,
    control: dict[str, Any],
    task: TaskRecord,
    data: dict[str, Any],
    discovery: dict[str, Any],
    member: MemberIdentity,
    claim: dict[str, Any],
    attempt: AttemptRecord | None,
) -> None:
    if attempt is None:
        raise _BlockedStep("active_claim_attempt_missing", retain=True, classification="partial_transition")
    if task.state.get("projection") != "running" or attempt.phase not in {"claimed", "starting", "running"}:
        raise _BlockedStep("active_claim_phase_mismatch", retain=True, classification="partial_transition")
    if (
        claim.get("attempt_id") != attempt.attempt_id
        or claim.get("fencing_token") != attempt.current_fencing_token
        or claim.get("machine_name") != attempt.machine_name
        or attempt.task_id != task.task_id
        or task.attempt_control.get("current_attempt_id") != attempt.attempt_id
        or task.attempt_control.get("current_attempt_number") != attempt.attempt_number
    ):
        raise _BlockedStep("active_claim_coherence_invalid", retain=True, classification="partial_transition")
    target_machine = control.get("machine_name")
    claim_machine = attempt.machine_name
    if attempt.phase in _TERMINAL_STATES:
        is_blocker = True
        classification = "active_claim_terminal"
    else:
        is_blocker = claim_machine == target_machine
        classification = "active_claim" if is_blocker else "running_other"
    if is_blocker and control.get("terminate_running") and not task.control.get("terminate_running"):
        _request_termination(cfg, task, control)
    _set_sensitive(discovery, member, task, classification, is_blocker, attempt)


def _classify_queued(
    cfg: RootConfig,
    control: dict[str, Any],
    data: dict[str, Any],
    discovery: dict[str, Any],
    member: MemberIdentity,
    task: TaskRecord,
) -> None:
    group = data["group"]
    target_machine = control["machine_name"]
    queue_scope = task.placement_runtime.get("queue_scope")
    sharing_mode = task.placement_policy.get("sharing_mode")
    if queue_scope == "home" or sharing_mode == "private":
        is_blocker = task.placement_policy.get("home_machine") == target_machine
        classification = "queued_home" if is_blocker else "queued_other_home"
    else:
        workers = group.get("worker_set")
        if type(workers) is not dict:
            raise _BlockedStep("group_worker_set_invalid")
        has_other_worker = any(
            name != target_machine
            and type(worker) is dict
            and worker.get("state") == "active"
            and task_machine_matches(task, name)
            and (
                not task.spec.requested_gpus
                or worker.get("gpu_limit_gpus") is None
                or worker["gpu_limit_gpus"] >= task.spec.requested_gpus
            )
            for name, worker in workers.items()
        )
        is_blocker = not has_other_worker
        classification = "queued_shared_blocked" if is_blocker else "queued_shared"
    _set_sensitive(discovery, member, task, classification, is_blocker, None)


def _request_termination(cfg: RootConfig, task: TaskRecord, control: dict[str, Any]) -> None:
    operation_id = control["operation_id"]
    with record_task_change(
        cfg,
        task,
        "task_cancel",
        details={"operation_id": operation_id, "terminate_running": True, "expected_effect": None},
    ):
        if task.control.get("terminate_running"):
            return
        task.control.update(
            {
                "cancellation_requested_at": task.control.get("cancellation_requested_at"),
                "cancellation_operation_id": operation_id,
                "terminate_running": True,
                "requested_by": cfg.machine_name,
            }
        )
        from ..runtime.records import utc_now

        if not task.control.get("cancellation_requested_at"):
            task.control["cancellation_requested_at"] = utc_now()
        task.meta["revision"] += 1
        task.meta["updated_at"] = utc_now()
        save_task(cfg, task)


def _current_attempt(cfg: RootConfig, task: TaskRecord) -> AttemptRecord | None:
    current_id = task.attempt_control.get("current_attempt_id")
    current_number = task.attempt_control.get("current_attempt_number")
    if current_number is None:
        if current_id is None and not task.claim_control.get("active_claim"):
            return None
        raise _BlockedStep("task_attempt_number_missing", retain=True, classification="partial_transition")
    if type(current_number) is not int or current_number < 1:
        raise _BlockedStep("task_attempt_number_invalid", retain=True, classification="partial_transition")
    if current_id is not None and (type(current_id) is not str or not current_id):
        raise _BlockedStep("task_attempt_identity_invalid", retain=True, classification="partial_transition")
    try:
        attempt = AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task.task_id, current_number)))
    except FileNotFoundError:
        if (
            current_id is None
            and not task.claim_control.get("active_claim")
            and (
                task.state.get("projection") == "cancelled"
                or (
                    task.state.get("projection") == "queued"
                    and task.state.get("reason") == "attempt_materialization_failed"
                )
            )
        ):
            return None
        raise _BlockedStep("task_attempt_missing", retain=True, classification="partial_transition") from None
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise _BlockedStep(
            _reason("task_attempt_invalid", exc), retain=True, classification="partial_transition"
        ) from exc
    if (
        attempt.task_id != task.task_id
        or (current_id is not None and attempt.attempt_id != current_id)
        or attempt.attempt_number != current_number
    ):
        raise _BlockedStep("task_attempt_identity_mismatch", retain=True, classification="partial_transition")
    return attempt


def _missing_task_is_proven(cfg: RootConfig, member: MemberIdentity, group_name: str) -> bool:
    path = locate_operation_path(cfg, "cleanup", member.task_id)
    if not path.exists():
        return False
    try:
        cleanup = read_json(path).get("cleanup")
    except (OSError, KeyError, TypeError, ValueError):
        return False
    if type(cleanup) is not dict or (
        cleanup.get("state") != "completed"
        or cleanup.get("task_id") != member.task_id
        or cleanup.get("group_name") != group_name
        or cleanup.get("submission_operation_id") != member.operation_id
    ):
        return False
    witness = cleanup.get("group_discovery_outcome")
    if type(witness) is not dict or type(witness.get("task")) is not dict:
        return False
    try:
        task = TaskRecord.from_dict(witness["task"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        task.task_id == member.task_id
        and task.group_name == group_name
        and task.submission_operation_id == member.operation_id
        and task.group_membership_sequence == member.sequence
        and task.state.get("projection") in _TERMINAL_STATES
        and not task.claim_control.get("active_claim")
    )


def _set_sensitive(
    discovery: dict[str, Any],
    member: MemberIdentity,
    task: TaskRecord,
    classification: str,
    is_blocker: bool,
    attempt: AttemptRecord | None,
) -> None:
    key = str(member.sequence)
    old = discovery["sensitive"].get(key)
    if old is not None and (
        old.get("task_id") != member.task_id or old.get("submission_operation_id") != member.operation_id
    ):
        raise _BlockedStep("worker_removal_sensitive_identity_mismatch")
    revision = task.meta.get("revision")
    if type(revision) is not int or revision < 0:
        raise _BlockedStep("task_revision_invalid", retain=False)
    receipt = {
        "task_id": member.task_id,
        "submission_operation_id": member.operation_id,
        "sequence": member.sequence,
        "task_revision": revision,
        "attempt_id": attempt.attempt_id if attempt is not None else task.attempt_control.get("current_attempt_id"),
        "attempt_number": attempt.attempt_number
        if attempt is not None
        else task.attempt_control.get("current_attempt_number"),
        "fencing_token": attempt.current_fencing_token if attempt is not None else None,
        "machine_name": attempt.machine_name if attempt is not None else None,
        "classification": classification,
        "is_blocker": is_blocker,
    }
    discovery["sensitive"][key] = receipt


def _select_lane(start: int, member: bool, journal: bool, policy: bool, sensitive: bool) -> int | None:
    runnable = (member, journal, policy)
    for offset in range(_LANE_COUNT):
        lane = (start + offset) % _LANE_COUNT
        if lane < 3 and runnable[lane]:
            return lane
        if lane == 3 and sensitive and not any(runnable):
            return lane
    return None


def _start_policy_pass(discovery: dict[str, Any], epoch: int) -> None:
    discovery["policy_pending"] = sorted(discovery["sensitive"], key=int)
    discovery["policy_target_epoch"] = epoch
    discovery["pending_cursor"] = 0


def _finish_policy_pass_if_ready(discovery: dict[str, Any], epoch: int, terminate_running: bool) -> bool:
    pending = discovery["policy_pending"]
    if not pending or discovery["pending_cursor"] < len(pending):
        return False
    target = discovery["policy_target_epoch"]
    discovery["policy_epoch"] = target
    discovery["policy_pending"] = []
    discovery["pending_cursor"] = 0
    if target == epoch:
        discovery["terminate_running"] = terminate_running
    return True


def _refresh_blockers(control: dict[str, Any], discovery: dict[str, Any]) -> None:
    sensitive = discovery.get("sensitive")
    if not isinstance(sensitive, dict):
        return
    blockers = sorted(
        {
            entry.get("task_id")
            for entry in sensitive.values()
            if isinstance(entry, dict) and entry.get("is_blocker") is True and isinstance(entry.get("task_id"), str)
        }
    )
    control["blockers"] = blockers
    if blockers:
        control["state"] = "waiting_ack"
        control["blocked_reason"] = None
        control["completed_at"] = None


def _set_blocked(control: dict[str, Any], reason: str) -> None:
    control["state"] = "blocked"
    control["blocked_reason"] = reason
    control["completed_at"] = None


def _set_converging(control: dict[str, Any], reason: str | None) -> None:
    if control.get("state") == "completed":
        return
    control["state"] = "converging"
    control["blocked_reason"] = reason
    control["completed_at"] = None


def _final_proof(
    cfg: RootConfig,
    control: dict[str, Any],
    data: dict[str, Any],
    discovery: dict[str, Any],
    journal: GroupRechecks,
    coverage: GroupCoverage,
    high_watermark: int,
    epoch: int,
    terminate_running: bool,
    results: list[TerminalCommitResult],
) -> tuple[bool, list[TerminalCommitResult]]:
    try:
        status = coverage.status()
        position = journal.snapshot()
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        _set_blocked(control, _reason("worker_removal_final_proof_failed", exc))
        return False, results
    if position is None:
        _set_blocked(control, "recheck_journal_missing")
        return False, results
    if position.generation != discovery["generation"]:
        _set_converging(control, "recheck_generation_changed")
        return False, results
    _refresh_blockers(control, discovery)
    pending_submission = data["group"].get("pending_submission_commit")
    no_unresolved = all(
        not entry.get("is_blocker")
        and entry.get("classification") in {"queued_home", "queued_other_home", "queued_shared", "running_other"}
        for entry in discovery["sensitive"].values()
        if isinstance(entry, dict)
    )
    ready = (
        discovery["member_cursor"] >= high_watermark
        and type(status.prefix) is int
        and status.prefix >= high_watermark
        and status.reason is None
        and pending_submission is None
        and discovery["journal_cursor"] == position.tail
        and not discovery["policy_pending"]
        and discovery["pending_cursor"] == 0
        and discovery["policy_epoch"] == epoch
        and discovery["terminate_running"] == terminate_running
        and not control.get("blockers")
        and no_unresolved
    )
    if not ready and not control.get("blockers"):
        reason = None
        if status.reason is not None:
            reason = _coverage_reason(status, high_watermark)
        elif pending_submission is not None:
            reason = "pending_submission_commit"
        elif discovery["journal_cursor"] != position.tail:
            reason = "recheck_pending"
        elif discovery["policy_pending"] or discovery["policy_epoch"] != epoch:
            reason = "worker_policy_pending"
        elif discovery["terminate_running"] != terminate_running:
            reason = "termination_policy_pending"
        elif discovery["member_cursor"] < high_watermark:
            reason = "membership_coverage_pending"
        _set_converging(control, reason)
    return ready, results


def _coverage_reason(status: Any, high_watermark: int) -> str | None:
    reason = status.reason
    if reason == "busy":
        return "membership_coverage_busy"
    if reason in {"invalid_state", "invalid_member_slot", "invalid_group"}:
        return f"membership_coverage_invalid_{reason}"
    if type(status.prefix) is not int or status.prefix < high_watermark:
        return "membership_coverage_pending"
    if reason is not None:
        return f"membership_coverage_{reason}"
    return None


def _high_watermark(group: dict[str, Any]) -> int:
    next_sequence = _counter(group.get("next_membership_sequence"), "next_membership_sequence", allow_zero=False)
    return next_sequence - 1


def _counter(value: Any, label: str, *, allow_zero: bool = True) -> int:
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum:
        raise _BlockedStep(f"{label}_invalid")
    return value


def _identifier(value: Any, label: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{label} must be a string")
    validate_identifier(value, label)
    return value


def _position_value(position: Any, key: str) -> int:
    value = position.get(key) if isinstance(position, dict) else getattr(position, key, None)
    if type(value) is not int or value < 0:
        raise ValueError(f"recheck position {key} is invalid")
    return value


def _reason(prefix: str, exc: BaseException) -> str:
    detail = str(exc).strip().replace("\n", " ")
    return f"{prefix}:{detail}" if detail else prefix


__all__ = ["advance_removal_discovery_locked", "initialize_removal_discovery"]
