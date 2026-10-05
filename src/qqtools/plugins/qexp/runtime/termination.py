"""Crash-recoverable local termination decisions for fenced Attempts."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Generator, Iterator, Mapping

from ..config_types import RootConfig
from ..runtime.paths import attempt_control_lock_path, local_paths
from ..runtime.records import new_id, utc_now
from ..runtime.store import atomic_replace, iter_json, read_json
from .locks import exclusive
from .work_budget import diagnostic_increment, diagnostic_span

TERMINATION_STATES = frozenset(
    {"pending", "signal_committed", "sigterm_sent", "sigkill_sent", "confirmed", "superseded"}
)
_TERMINATION_TRANSITIONS = {
    "pending": {"signal_committed", "superseded"},
    "signal_committed": {"sigterm_sent", "confirmed"},
    "sigterm_sent": {"sigkill_sent", "confirmed"},
    "sigkill_sent": {"confirmed"},
    "confirmed": set(),
    "superseded": set(),
}
_DEADLINE_ONLY_SIGNAL_ERROR = "deadline-only termination decisions cannot authorize signals"
_TIMEOUT_RETIREMENT_MARKER = "QQTOOLS-COMPAT-0022"
TIMEOUT_RETIREMENT_STATES = frozenset({"suppressed", "shared_reconciled", "retired"})
_RETIREMENT_RECEIPT_VERSION = 1
_PROCESS_IDENTITY_FIELDS = (
    "wrapper_pid",
    "wrapper_start_time_ticks",
    "process_group_id",
    "process_group_start_time_ticks",
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def termination_decision_digest(decision: Mapping[str, Any]) -> str:
    """Return the stable digest of one immutable termination decision."""
    if not isinstance(decision, Mapping):
        raise ValueError("termination decision must be an object")
    return hashlib.sha256(_canonical_json(dict(decision))).hexdigest()


def is_deadline_only_decision(decision: Mapping[str, Any] | None) -> bool:
    """Return true only for the unambiguous historical timeout cause."""
    return isinstance(decision, Mapping) and (
        decision.get("authority_outcome") == "holder_safe_deadline_elapsed"
        and decision.get("reason") == "holder_safe_deadline_elapsed"
    )


def timeout_retirement_path(cfg: RootConfig, attempt_id: str, decision_id: str) -> Path:
    """Return the deterministic local receipt path for one old decision."""
    return local_paths(cfg.runtime_root)["termination_decisions"] / attempt_id / f"retirement-{decision_id}.json"


def _retirement_payload(
    decision: Mapping[str, Any],
    *,
    task_id: str,
    attempt_id: str,
    attempt_number: int,
    fencing_token: int,
    machine_name: str,
    process_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if not is_deadline_only_decision(decision):
        raise ValueError("only holder_safe_deadline_elapsed decisions can be retired")
    if decision.get("state") not in TERMINATION_STATES:
        raise ValueError("deadline-only decision state is invalid")
    if decision.get("shared_commitment") not in {"pending", "committed", "unavailable"}:
        raise ValueError("deadline-only shared commitment is invalid")
    decision_id = decision.get("decision_id")
    if not isinstance(decision_id, str) or not decision_id:
        raise ValueError("deadline-only decision ID is missing")
    if decision.get("task_id") != task_id or decision.get("attempt_id") != attempt_id:
        raise ValueError("deadline-only decision identity does not match the Attempt")
    if decision.get("decision_token") != fencing_token:
        raise ValueError("deadline-only decision token does not match the Attempt")
    if type(attempt_number) is not int or attempt_number < 1:
        raise ValueError("retirement Attempt number is invalid")
    if not isinstance(machine_name, str) or not machine_name:
        raise ValueError("retirement machine identity is missing")
    if not isinstance(process_identity, Mapping) or set(process_identity) != set(_PROCESS_IDENTITY_FIELDS):
        raise ValueError("retirement process identity is incomplete")
    if any(
        process_identity.get(field) != decision.get(field)
        for field in ("process_group_id", "process_group_start_time_ticks")
    ):
        raise ValueError("retirement process identity differs from the decision")
    operation_identity = {
        "receipt_version": _RETIREMENT_RECEIPT_VERSION,
        "operation": "timeout_decision_retirement",
        "decision_id": decision_id,
        "decision_digest": termination_decision_digest(decision),
        "task_id": task_id,
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "fencing_token": fencing_token,
        "machine_name": machine_name,
        "process_identity": dict(process_identity),
        "source_state": decision.get("state"),
        "source_shared_commitment": decision.get("shared_commitment"),
        "source_authority_outcome": decision.get("authority_outcome"),
        "source_reason": decision.get("reason"),
    }
    operation_id = hashlib.sha256(_canonical_json(operation_identity)).hexdigest()
    now = utc_now()
    return {
        **operation_identity,
        "operation_id": operation_id,
        "state": "suppressed",
        "delivery": "unknown" if decision.get("state") != "pending" else "not_attempted",
        "shared_receipt": None,
        "created_at": now,
        "updated_at": now,
        "marker": _TIMEOUT_RETIREMENT_MARKER,
    }


def _checked_retirement_receipt(receipt: object) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise ValueError("timeout retirement receipt must be an object")
    required = {
        "receipt_version",
        "operation",
        "operation_id",
        "decision_id",
        "decision_digest",
        "task_id",
        "attempt_id",
        "attempt_number",
        "fencing_token",
        "machine_name",
        "process_identity",
        "source_state",
        "source_shared_commitment",
        "source_authority_outcome",
        "source_reason",
        "state",
        "delivery",
        "shared_receipt",
        "created_at",
        "updated_at",
        "marker",
    }
    if set(receipt) != required:
        raise ValueError("timeout retirement receipt fields are invalid")
    value = deepcopy(dict(receipt))
    if value["receipt_version"] != _RETIREMENT_RECEIPT_VERSION or value["operation"] != "timeout_decision_retirement":
        raise ValueError("timeout retirement receipt version is invalid")
    if value["marker"] != _TIMEOUT_RETIREMENT_MARKER or value["state"] not in TIMEOUT_RETIREMENT_STATES:
        raise ValueError("timeout retirement receipt state is invalid")
    if value["delivery"] not in {"not_attempted", "unknown"}:
        raise ValueError("timeout retirement delivery state is invalid")
    if type(value["attempt_number"]) is not int or value["attempt_number"] < 1:
        raise ValueError("timeout retirement Attempt number is invalid")
    if type(value["fencing_token"]) is not int or value["fencing_token"] < 1:
        raise ValueError("timeout retirement fencing token is invalid")
    if not isinstance(value["process_identity"], Mapping) or set(value["process_identity"]) != set(
        _PROCESS_IDENTITY_FIELDS
    ):
        raise ValueError("timeout retirement process identity is invalid")
    identity = {
        key: value[key]
        for key in required
        if key not in {"operation_id", "state", "delivery", "shared_receipt", "created_at", "updated_at", "marker"}
    }
    expected_id = hashlib.sha256(_canonical_json(identity)).hexdigest()
    if value["operation_id"] != expected_id:
        raise ValueError("timeout retirement operation identity is invalid")
    if not isinstance(value["decision_digest"], str) or len(value["decision_digest"]) != 64:
        raise ValueError("timeout retirement decision digest is invalid")
    return value


def read_timeout_retirement(cfg: RootConfig, attempt_id: str, decision_id: str) -> dict[str, Any] | None:
    """Read and validate one local timeout-retirement receipt."""
    path = timeout_retirement_path(cfg, attempt_id, decision_id)
    try:
        value = read_json(path)
    except FileNotFoundError:
        return None
    envelope = value.get("timeout_decision_retirement") if isinstance(value, dict) else None
    return _checked_retirement_receipt(envelope)


def suppress_timeout_decision(
    cfg: RootConfig,
    *,
    decision: Mapping[str, Any],
    task_id: str,
    attempt_id: str,
    attempt_number: int,
    fencing_token: int,
    machine_name: str,
    process_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist the local suppression receipt before any shared retirement I/O."""
    payload = _retirement_payload(
        decision,
        task_id=task_id,
        attempt_id=attempt_id,
        attempt_number=attempt_number,
        fencing_token=fencing_token,
        machine_name=machine_name,
        process_identity=process_identity,
    )
    path = timeout_retirement_path(cfg, attempt_id, payload["decision_id"])
    existing = read_timeout_retirement(cfg, attempt_id, payload["decision_id"])
    if existing is not None:
        if existing["operation_id"] != payload["operation_id"] or any(
            existing.get(field) != payload.get(field)
            for field in (
                "decision_digest",
                "task_id",
                "attempt_id",
                "attempt_number",
                "fencing_token",
                "machine_name",
                "process_identity",
            )
        ):
            raise ValueError("timeout retirement receipt identity changed")
        return existing
    atomic_replace(path, {"timeout_decision_retirement": payload})
    return payload


def update_timeout_retirement(
    cfg: RootConfig,
    attempt_id: str,
    decision_id: str,
    *,
    state: str,
    shared_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Advance the local retirement receipt monotonically and idempotently."""
    if state not in TIMEOUT_RETIREMENT_STATES:
        raise ValueError("timeout retirement state is invalid")
    path = timeout_retirement_path(cfg, attempt_id, decision_id)
    value = read_json(path)
    receipt = _checked_retirement_receipt(value.get("timeout_decision_retirement"))
    current = receipt["state"]
    order = {"suppressed": 0, "shared_reconciled": 1, "retired": 2}
    if order[state] < order[current]:
        raise RuntimeError("timeout retirement state transition is not monotonic")
    if order[state] == order[current]:
        return receipt
    if shared_receipt is not None:
        receipt["shared_receipt"] = deepcopy(dict(shared_receipt))
    receipt["state"] = state
    receipt["updated_at"] = utc_now()
    atomic_replace(path, {"timeout_decision_retirement": receipt})
    return receipt


def retirement_matches_decision(
    receipt: Mapping[str, Any] | None,
    decision: Mapping[str, Any],
    *,
    task_id: str,
    attempt_id: str,
    attempt_number: int,
    fencing_token: int,
    machine_name: str,
    process_identity: Mapping[str, Any],
) -> bool:
    """Validate a receipt against immutable old decision and live identity."""
    if not isinstance(receipt, Mapping) or not is_deadline_only_decision(decision):
        return False
    try:
        expected = _retirement_payload(
            decision,
            task_id=task_id,
            attempt_id=attempt_id,
            attempt_number=attempt_number,
            fencing_token=fencing_token,
            machine_name=machine_name,
            process_identity=process_identity,
        )
        checked = _checked_retirement_receipt(receipt)
    except (TypeError, ValueError):
        return False
    return all(
        checked.get(field) == expected.get(field)
        for field in (
            "operation_id",
            "decision_digest",
            "task_id",
            "attempt_id",
            "attempt_number",
            "fencing_token",
            "machine_name",
            "process_identity",
        )
    )


def _retirement_proves_recovery_safe(receipt: Mapping[str, Any] | None, decision: Mapping[str, Any]) -> bool:
    """Require exact local and shared proof before lifting one old blocker."""
    if not isinstance(receipt, Mapping) or receipt.get("state") != "retired":
        return False
    if not retirement_matches_decision(
        receipt,
        decision,
        task_id=decision.get("task_id"),
        attempt_id=decision.get("attempt_id"),
        attempt_number=receipt.get("attempt_number"),
        fencing_token=decision.get("decision_token"),
        machine_name=receipt.get("machine_name"),
        process_identity=receipt.get("process_identity"),
    ):
        return False
    proof = receipt.get("shared_receipt")
    if not isinstance(proof, Mapping):
        return False
    shared_retirement = proof.get("retirement_receipt")
    revisions = proof.get("committed_revisions")
    return (
        proof.get("outcome") in {"committed", "already_committed"}
        and proof.get("shared_commitment") == "committed"
        and proof.get("authority_granted") is False
        and proof.get("local_effects") in ([], ())
        and all(
            proof.get(field) == receipt.get(field)
            for field in (
                "task_id",
                "attempt_id",
                "attempt_number",
                "fencing_token",
                "machine_name",
                "decision_id",
                "process_identity",
            )
        )
        and proof.get("decision_token") == receipt.get("fencing_token")
        and proof.get("authority_outcome") == receipt.get("source_authority_outcome")
        and proof.get("decision_reason") == receipt.get("source_reason")
        and isinstance(shared_retirement, Mapping)
        and shared_retirement.get("operation_id") == receipt.get("operation_id")
        and shared_retirement.get("decision_digest") == receipt.get("decision_digest")
        and isinstance(revisions, Mapping)
        and type(revisions.get("task")) is int
        and isinstance(revisions.get("attempt_digest"), str)
        and len(revisions["attempt_digest"]) == 64
    )


def _reject_deadline_only_signal(decision: dict[str, Any]) -> None:
    if decision.get("authority_outcome") == "holder_safe_deadline_elapsed" or decision.get("reason") == (
        "holder_safe_deadline_elapsed"
    ):
        raise RuntimeError(_DEADLINE_ONLY_SIGNAL_ERROR)


@contextmanager
def attempt_control_lock(cfg: RootConfig, attempt_id: str, *, blocking: bool = True) -> Iterator[bool]:
    with exclusive(attempt_control_lock_path(cfg.runtime_root, attempt_id), blocking=blocking) as acquired:
        yield acquired


def decision_path(cfg: RootConfig, attempt_id: str, decision_id: str) -> Path:
    return local_paths(cfg.runtime_root)["termination_decisions"] / attempt_id / f"{decision_id}.json"


def list_decisions(cfg: RootConfig, attempt_id: str | None = None) -> list[Path]:
    root = local_paths(cfg.runtime_root)["termination_decisions"]
    if attempt_id is not None:
        return iter_json(root / attempt_id)
    return [path for directory in sorted(root.glob("*")) if directory.is_dir() for path in iter_json(directory)]


def create_decision(
    cfg: RootConfig,
    *,
    task_id: str,
    attempt_id: str,
    fencing_token: int,
    process: dict[str, Any],
    authority_outcome: str,
    reason: str,
    decision_id: str | None = None,
) -> dict[str, Any]:
    """Create the only durable record that may authorize an external signal."""
    decision_id = decision_id or new_id()
    path = decision_path(cfg, attempt_id, decision_id)
    if path.exists():
        return read_json(path)["termination_decision"]
    value = {
        "termination_decision": {
            "decision_id": decision_id,
            "task_id": task_id,
            "attempt_id": attempt_id,
            "decision_token": fencing_token,
            "authority_outcome": authority_outcome,
            "reason": reason,
            "state": "pending",
            "shared_commitment": "pending",
            "shared_reconciliation": "pending",
            "process_group_id": process.get("process_group_id"),
            "process_group_start_time_ticks": process.get("process_group_start_time_ticks"),
            "signal_attempts": [],
            "observed_exit_code": None,
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
    }
    atomic_replace(path, value)
    return value["termination_decision"]


def update_decision(cfg: RootConfig, attempt_id: str, decision_id: str, **changes: Any) -> dict[str, Any]:
    path = decision_path(cfg, attempt_id, decision_id)
    value = read_json(path)
    decision = value["termination_decision"]
    if decision["state"] not in TERMINATION_STATES:
        raise RuntimeError("termination decision has an invalid state.")
    next_state = changes.get("state", decision["state"])
    if next_state not in TERMINATION_STATES:
        raise RuntimeError("termination decision has an invalid target state.")
    if next_state != decision["state"] and next_state not in _TERMINATION_TRANSITIONS[decision["state"]]:
        raise RuntimeError("termination decision state transition is not monotonic.")
    if next_state == "confirmed" and changes.get("confirmation") not in {"identity_absent", "process_absent"}:
        raise RuntimeError("termination confirmation requires absent process identity.")
    decision.update(changes)
    decision["updated_at"] = utc_now()
    atomic_replace(path, value)
    return decision


def termination_check_steps(
    cfg: RootConfig, attempt_id: str, limit: int = 8
) -> Generator[None, None, dict[str, Any] | None]:
    """Find a commitment in bounded pages while the caller owns the Attempt lock.

    A yielded value is unfinished work, never a negative proof. The caller must
    retain the lock through the returned result and its subsequent recovery CAS.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError("recovery page limit must be a positive integer")
    directory = local_paths(cfg.runtime_root)["termination_decisions"] / attempt_id
    try:
        entries = os.scandir(directory)
    except (FileNotFoundError, NotADirectoryError):
        return None
    with entries:
        visited = 0
        for entry in entries:
            diagnostic_increment("store.inventory_entries")
            visited += 1
            if entry.name.endswith(".json") and entry.is_file(follow_symlinks=False):
                record = read_json(Path(entry.path))
                if set(record) == {"timeout_decision_retirement"}:
                    continue
                decision = record.get("termination_decision") if isinstance(record, dict) else None
                if not isinstance(decision, dict):
                    raise ValueError("local termination decision must be an object")
                if is_deadline_only_decision(decision):
                    retirement = read_timeout_retirement(cfg, attempt_id, decision.get("decision_id", ""))
                    if _retirement_proves_recovery_safe(retirement, decision):
                        continue
                    return decision
                if decision.get("state") in {"signal_committed", "sigterm_sent", "sigkill_sent", "confirmed"}:
                    return decision
                if decision.get("shared_commitment") in {"committed", "unavailable"}:
                    return decision
            if visited == limit:
                yield
                visited = 0
    return None


def recovery_check_steps(cfg: RootConfig, attempt_id: str, limit: int = 8) -> Generator[None, None, bool]:
    """Return a negative recovery proof only at EOF under the caller's lock."""
    return (yield from termination_check_steps(cfg, attempt_id, limit)) is not None


def is_recovery_blocked(cfg: RootConfig, attempt_id: str) -> bool:
    """Synchronous recovery check for callers retaining the Attempt lock."""
    with diagnostic_span("termination.recovery_inventory"):
        steps = recovery_check_steps(cfg, attempt_id)
        try:
            while True:
                next(steps)
        except StopIteration as result:
            return result.value
        finally:
            steps.close()


def commit_local_unavailable(cfg: RootConfig, attempt_id: str, decision_id: str) -> dict[str, Any]:
    return update_decision(cfg, attempt_id, decision_id, shared_commitment="unavailable")


def commit_signal(cfg: RootConfig, attempt_id: str, decision_id: str) -> dict[str, Any]:
    decision = read_json(decision_path(cfg, attempt_id, decision_id))["termination_decision"]
    _reject_deadline_only_signal(decision)
    decision = update_decision(cfg, attempt_id, decision_id, state="signal_committed")
    return decision


def _matches_process_group(process_group_id: int | None, expected_start: int | None) -> bool:
    if not process_group_id or expected_start is None:
        return False
    try:
        stat = (Path("/proc") / str(process_group_id) / "stat").read_text(encoding="utf-8")
        return int(stat.rsplit(")", 1)[1].split()[19]) == expected_start
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return False


def send_signals(cfg: RootConfig, attempt_id: str, decision_id: str, *, grace_seconds: float = 5.0) -> dict[str, Any]:
    """Idempotently progress an irreversible committed decision to confirmed."""
    decision = read_json(decision_path(cfg, attempt_id, decision_id))["termination_decision"]
    _reject_deadline_only_signal(decision)
    if decision["state"] == "pending":
        raise RuntimeError("signal_committed must be durable before sending a signal.")
    pgid = decision.get("process_group_id")
    start = decision.get("process_group_start_time_ticks")
    if not _matches_process_group(pgid, start):
        return update_decision(
            cfg, attempt_id, decision_id, state="confirmed", observed_exit_code=None, confirmation="identity_absent"
        )
    if decision["state"] == "signal_committed":
        os.killpg(pgid, signal.SIGTERM)
        decision = update_decision(
            cfg,
            attempt_id,
            decision_id,
            state="sigterm_sent",
            signal_attempts=decision["signal_attempts"] + [{"signal": "SIGTERM", "at": utc_now()}],
        )
    if decision["state"] == "sigterm_sent":
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline and _matches_process_group(pgid, start):
            time.sleep(0.05)
    if decision["state"] == "sigterm_sent" and _matches_process_group(pgid, start):
        os.killpg(pgid, signal.SIGKILL)
        decision = update_decision(
            cfg,
            attempt_id,
            decision_id,
            state="sigkill_sent",
            signal_attempts=decision["signal_attempts"] + [{"signal": "SIGKILL", "at": utc_now()}],
        )
    if decision["state"] == "sigkill_sent":
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline and _matches_process_group(pgid, start):
            time.sleep(0.05)
    if not _matches_process_group(pgid, start):
        return update_decision(cfg, attempt_id, decision_id, state="confirmed", confirmation="process_absent")
    return decision


def advance_signals(
    cfg: RootConfig,
    attempt_id: str,
    decision_id: str,
    *,
    sigterm_deadline: float | None,
    grace_seconds: float = 5.0,
) -> tuple[dict[str, Any], float | None]:
    """Advance one committed signal step without sleeping under the Attempt lock.

    The caller owns the Attempt control lock and the advisory monotonic deadline.
    Losing that deadline grants a fresh grace period; it never accelerates SIGKILL.
    Every call reloads the durable decision and rechecks process-group identity.
    """
    decision = read_json(decision_path(cfg, attempt_id, decision_id))["termination_decision"]
    _reject_deadline_only_signal(decision)
    state = decision["state"]
    if state == "pending":
        raise RuntimeError("signal_committed must be durable before sending a signal.")
    if state in {"confirmed", "superseded"}:
        return decision, None
    pgid = decision.get("process_group_id")
    start = decision.get("process_group_start_time_ticks")
    if not _matches_process_group(pgid, start):
        return (
            update_decision(cfg, attempt_id, decision_id, state="confirmed", confirmation="identity_absent"),
            None,
        )
    if state == "signal_committed":
        os.killpg(pgid, signal.SIGTERM)
        decision = update_decision(
            cfg,
            attempt_id,
            decision_id,
            state="sigterm_sent",
            signal_attempts=decision["signal_attempts"] + [{"signal": "SIGTERM", "at": utc_now()}],
        )
        return decision, time.monotonic() + grace_seconds
    if state == "sigterm_sent":
        if sigterm_deadline is None:
            return decision, time.monotonic() + grace_seconds
        if time.monotonic() < sigterm_deadline:
            return decision, sigterm_deadline
        os.killpg(pgid, signal.SIGKILL)
        decision = update_decision(
            cfg,
            attempt_id,
            decision_id,
            state="sigkill_sent",
            signal_attempts=decision["signal_attempts"] + [{"signal": "SIGKILL", "at": utc_now()}],
        )
    if not _matches_process_group(pgid, start):
        decision = update_decision(cfg, attempt_id, decision_id, state="confirmed", confirmation="process_absent")
    return decision, None
