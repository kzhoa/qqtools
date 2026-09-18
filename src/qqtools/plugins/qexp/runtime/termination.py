"""Crash-recoverable local termination decisions for fenced Attempts."""

from __future__ import annotations

import os
import signal
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Iterator

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
                decision = record.get("termination_decision") if isinstance(record, dict) else None
                if not isinstance(decision, dict):
                    raise ValueError("local termination decision must be an object")
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
