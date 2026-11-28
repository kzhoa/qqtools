"""Machine-local orchestration for advisory scheduler diagnostic recovery."""

from __future__ import annotations

import os
import re
import shlex
import stat
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event
from typing import Any

from ..runtime.locks import exclusive
from ..runtime.store import atomic_replace, read_json_limited, require_json_size
from .context import MachineRuntime, _effective_runtime_id
from .helpers import _MAX_REPAIR_STATUS_BYTES, _REPAIR_IDENTITY_FILE, _pid_start_time_ticks, _status_file_witness
from .identity import decode_identity_record
from .scheduler_diagnostics import SchedulerDiagnosticStore

_REPORT_NAME = "scheduler-recovery.json"
_MAX_REPORT_BYTES = 4096
_RECOVERY_STATES = {"pending", "progressing", "awaiting_observation", "blocked", "idle"}
_COUNT_FIELDS = ("eligible", "deferred", "retained_current", "retained_uncertain")
_REASON_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_READ_ERRORS = (OSError, RuntimeError, ValueError, TypeError)
_ActiveIdentity = tuple[int, str, int]


def diagnostics_repair_command(runtime: MachineRuntime, *, apply: bool = False) -> str:
    """Build a copyable repair command tied to this MachineRuntime."""
    return _machine_command(runtime, ["agent", "diagnostics", "repair", "--apply" if apply else "--dry-run"])


def _machine_command(runtime: MachineRuntime, command: list[str]) -> str:
    root = str(runtime.root)
    if root.isprintable():
        return shlex.join(["qexp", "--machine-runtime-root", root, *command])

    # The suffix keeps command substitution from stripping trailing newlines.
    encoded_root = "".join(f"\\0{byte:03o}" for byte in os.fsencode(root))
    return (
        f"(qexp_runtime_root=$(printf '%b' '{encoded_root}_'); "
        f'qexp --machine-runtime-root "${{qexp_runtime_root%_}}" {shlex.join(command)})'
    )


def _scope(runtime: MachineRuntime) -> dict[str, Any]:
    return {"schema_version": 1, "scope": "machine_runtime", "machine_runtime_root": str(runtime.root)}


def _read_bounded_json(path: Path) -> dict[str, Any]:
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("expected a regular local JSON record")
    return read_json_limited(path, max_bytes=_MAX_REPORT_BYTES, record_type="scheduler_recovery")


def _verified_agent(runtime: MachineRuntime, runtime_id: str) -> _ActiveIdentity | None:
    try:
        if runtime.observe_instance_id() != runtime_id:
            return None
        pid_path = runtime.paths["pid"]
        if not stat.S_ISREG(pid_path.lstat().st_mode):
            return None
        with pid_path.open("rb") as handle:
            pid_record = handle.read(32)
        if len(pid_record) >= 32:
            return None
        pid = int(pid_record)
        status_path = runtime.paths["agent"] / "status.json"
        info = status_path.lstat()
        if not stat.S_ISREG(info.st_mode):
            return None
        if info.st_size > _MAX_REPAIR_STATUS_BYTES:
            record = _read_bounded_json(status_path.with_name(_REPAIR_IDENTITY_FILE))
            if record.get("status_witness") != _status_file_witness(status_path.lstat()):
                return None
        else:
            record = read_json_limited(status_path, max_bytes=_MAX_REPAIR_STATUS_BYTES, record_type="machine_agent")
        status = record.get("machine_agent")
        if not isinstance(status, dict) or status.get("state") != "active" or status.get("runtime_id") != runtime_id:
            return None
        instance_id = status.get("instance_id")
        start_ticks = status.get("pid_start_time_ticks")
        if (
            status.get("pid") != pid
            or not isinstance(instance_id, str)
            or type(start_ticks) is not int
            or _pid_start_time_ticks(pid) != start_ticks
        ):
            return None
        return pid, instance_id, start_ticks
    except _READ_ERRORS:
        return None


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "state": "blocked",
        "reason": reason,
        **dict.fromkeys(_COUNT_FIELDS),
        "retired": 0,
        "pending_transition": False,
        "reconciled_pending": False,
        "backup_path": None,
        "backup_verified": False,
    }


def _repair_store(
    runtime: MachineRuntime,
    *,
    runtime_id: str,
    owner: Callable[[], str | None],
    apply: bool,
    automatic: bool = False,
    mutation_allowed: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    try:
        return SchedulerDiagnosticStore(runtime).repair(
            runtime_id=runtime_id, owner=owner, apply=apply, automatic=automatic, mutation_allowed=mutation_allowed
        )
    except PermissionError:
        return _blocked("permission_denied")
    except _READ_ERRORS:
        return _blocked("store_unavailable")


def repair_scheduler_diagnostics(runtime: MachineRuntime, *, apply: bool) -> dict[str, Any]:
    """Inspect or repair the local scheduler diagnostic store under live ownership."""

    def repair() -> dict[str, Any]:
        try:
            runtime_id = runtime.observe_instance_id()
        except _READ_ERRORS:
            return _blocked("ownership_unverified")

        identity = _verified_agent(runtime, runtime_id)

        def owner() -> str | None:
            if identity is None or _verified_agent(runtime, runtime_id) != identity:
                return None
            return identity[1]

        if not apply:
            result = _repair_store(runtime, runtime_id=runtime_id, owner=owner, apply=False)
            if result["state"] == "idle" and identity is None:
                result.update(state="blocked", reason="ownership_unverified")
            return result

        with exclusive(_report_lock_path(runtime), blocking=False) as acquired:
            if not acquired:
                return _blocked("lock_unavailable")
            result = _repair_store(runtime, runtime_id=runtime_id, owner=owner, apply=True)
            if owner() is None:
                if result["state"] == "idle":
                    result.update(state="blocked", reason="ownership_unverified")
                return result
            instance_id = identity[1]
            report = _advance_report(runtime, runtime_id=runtime_id, instance_id=instance_id, result=result)
            if not _write_report(runtime, report):
                result["progress_report_unavailable"] = True
            return result

    if apply:
        try:
            with runtime.agent_lifecycle_guard(blocking=False) as acquired:
                result = repair() if acquired else _blocked("lock_unavailable")
        except PermissionError:
            result = _blocked("permission_denied")
        except _READ_ERRORS:
            result = _blocked("store_unavailable")
    else:
        result = repair()

    remaining = ("deferred", "pending_transition") if apply else ("eligible", "deferred", "pending_transition")
    apply_needed: bool | None = any(result.get(key) for key in remaining)
    next_action = None
    if result["state"] == "blocked":
        if not apply_needed:
            apply_needed = None
        if not apply and result["reason"] == "ownership_unverified":
            next_action = _machine_command(runtime, ["agent", "status"])
        else:
            next_action = diagnostics_repair_command(runtime)
    elif apply_needed:
        next_action = diagnostics_repair_command(runtime, apply=not apply)
    return {
        **result,
        **_scope(runtime),
        "action": "diagnostics_repair",
        "apply_needed": apply_needed,
        "next_action": next_action,
    }


def _is_utc_timestamp(value: object) -> bool:
    if not isinstance(value, str) or len(value) > 40 or not value.endswith("Z"):
        return False
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError:
        return False
    return timestamp.utcoffset() == timedelta(0)


def _is_count(value: object) -> bool:
    return type(value) is int and value >= 0


def _report_is_valid(report: dict[str, Any]) -> bool:
    state = report.get("state")
    reason = report.get("reason")
    return (
        type(report.get("schema_version")) is int
        and report["schema_version"] == 1
        and isinstance(state, str)
        and state in _RECOVERY_STATES
        and (reason is None or (isinstance(reason, str) and _REASON_CODE.fullmatch(reason) is not None))
        and _is_count(report.get("retired"))
        and all(report.get(key) is None or _is_count(report[key]) for key in _COUNT_FIELDS)
        and type(report.get("pending_transition")) is bool
        and all(
            report.get(key) is None or _is_utc_timestamp(report[key]) for key in ("last_progress_at", "next_check_at")
        )
        and (report.get("next_check_at") is None or state == "blocked")
    )


def _report(
    *,
    runtime_id: str | None,
    instance_id: str | None,
    state: str,
    result: dict[str, Any] | None = None,
    retired: int = 0,
    last_progress_at: str | None = None,
    next_check_at: str | None = None,
) -> dict[str, Any]:
    result = result or {}
    return {
        "schema_version": 1,
        "runtime_id": runtime_id,
        "agent_instance_id": instance_id,
        "state": state,
        "reason": result.get("reason"),
        "retired": retired,
        **{key: result.get(key) for key in _COUNT_FIELDS},
        "pending_transition": bool(result.get("pending_transition")),
        "last_progress_at": last_progress_at,
        "next_check_at": next_check_at,
    }


def read_scheduler_recovery(
    runtime: MachineRuntime,
    *,
    instance_id: str | None,
    read_json: Callable[[Path], dict | None] | None = None,
) -> dict[str, Any]:
    """Read bounded recovery evidence for the specified live agent instance."""
    reader = _read_bounded_json if read_json is None else read_json
    fallback = _report(runtime_id=None, instance_id=instance_id, state="pending" if instance_id else "idle")
    report = fallback
    if instance_id:
        try:
            raw_identity = reader(runtime.paths["identity"])
            identity = decode_identity_record(raw_identity) if isinstance(raw_identity, dict) else None
            if identity is not None:
                runtime_id = _effective_runtime_id(identity["instance_id"], identity.get("runtime_id"))
                fallback["runtime_id"] = runtime_id
                raw = reader(runtime.paths["agent"] / _REPORT_NAME)
                if (
                    isinstance(raw, dict)
                    and raw.keys() >= fallback.keys()
                    and raw.get("runtime_id") == runtime_id
                    and raw.get("agent_instance_id") == instance_id
                    and _report_is_valid(raw)
                ):
                    report = {key: raw.get(key) for key in fallback}
        except _READ_ERRORS:
            pass
    return {
        **report,
        **_scope(runtime),
        "next_action": diagnostics_repair_command(runtime) if report["state"] == "blocked" else None,
    }


def _report_lock_path(runtime: MachineRuntime) -> Path:
    return runtime.paths["locks"] / "scheduler-recovery.lock"


def _utc_timestamp(now: datetime) -> str:
    return now.isoformat().replace("+00:00", "Z")


def _advance_report(
    runtime: MachineRuntime,
    *,
    runtime_id: str,
    instance_id: str,
    result: dict[str, Any],
    next_check_at: str | None = None,
    unpublished_retired: int = 0,
    unpublished_progress_at: str | None = None,
) -> dict[str, Any]:
    previous = read_scheduler_recovery(runtime, instance_id=instance_id)
    progress_at = unpublished_progress_at or previous["last_progress_at"]
    if result["retired"] or result.get("reconciled_pending"):
        progress_at = _utc_timestamp(datetime.now(timezone.utc))
    state = result["state"]
    if state == "idle" and progress_at is not None:
        state = "awaiting_observation"
    counts = {
        **result,
        **{key: previous[key] for key in _COUNT_FIELDS if result.get(key) is None},
    }
    return _report(
        runtime_id=runtime_id,
        instance_id=instance_id,
        state=state,
        result=counts,
        retired=previous["retired"] + unpublished_retired + result["retired"],
        last_progress_at=progress_at,
        next_check_at=next_check_at,
    )


def _write_report(runtime: MachineRuntime, report: dict[str, Any], *, stop_event: Event | None = None) -> bool:
    def require_running(_witness: os.stat_result) -> None:
        if stop_event is not None and stop_event.is_set():
            raise OSError("scheduler recovery stopped")

    try:
        require_json_size(report, max_bytes=_MAX_REPORT_BYTES, record_type="scheduler_recovery")
        return atomic_replace(runtime.paths["agent"] / _REPORT_NAME, report, before_replace=require_running) is not None
    except _READ_ERRORS:
        return False


def publish_scheduler_recovery_start_failure(runtime: MachineRuntime, *, runtime_id: str, instance_id: str) -> None:
    """Expose a maintenance startup failure where the normal diagnostic store need not be writable."""
    result = _blocked("maintenance_unavailable")
    _write_report(runtime, _report(runtime_id=runtime_id, instance_id=instance_id, state="blocked", result=result))


def maintain_scheduler_diagnostics(
    runtime: MachineRuntime,
    *,
    identity: _ActiveIdentity,
    runtime_id: str,
    stop_event: Event,
) -> None:
    """Drain bounded repair batches while this lifecycle instance owns the agent."""
    instance_id = identity[1]

    def owner() -> str | None:
        if stop_event.is_set():
            return None
        current_identity = _verified_agent(runtime, runtime_id)
        return instance_id if current_identity == identity and not stop_event.is_set() else None

    blocked_delay = 1.0
    lock_misses = 0
    unpublished_retired = 0
    unpublished_progress_at = None
    while not stop_event.is_set():
        delay = 0.25
        try:
            with exclusive(_report_lock_path(runtime), blocking=False) as acquired:
                if acquired:
                    result = _repair_store(
                        runtime,
                        runtime_id=runtime_id,
                        owner=owner,
                        apply=True,
                        automatic=True,
                        mutation_allowed=lambda: not stop_event.is_set(),
                    )
                    if stop_event.is_set():
                        return
                    next_check_at = None
                    delay = 0.05
                    if result["state"] == "blocked" and result["reason"] == "lock_unavailable":
                        lock_misses += 1
                        delay = 0.25
                        if lock_misses < 5:
                            result["state"] = "pending"
                    else:
                        lock_misses = 0
                        if result["state"] == "blocked":
                            delay = blocked_delay
                            blocked_delay = min(30.0, blocked_delay * 2)
                        else:
                            blocked_delay = 1.0
                    if result["state"] == "blocked":
                        next_check_at = _utc_timestamp(datetime.now(timezone.utc) + timedelta(seconds=delay))
                    report = _advance_report(
                        runtime,
                        runtime_id=runtime_id,
                        instance_id=instance_id,
                        result=result,
                        next_check_at=next_check_at,
                        unpublished_retired=unpublished_retired,
                        unpublished_progress_at=unpublished_progress_at,
                    )
                    if report["state"] == "pending" and report["last_progress_at"] is not None:
                        report["state"] = "progressing"
                    if _write_report(runtime, report, stop_event=stop_event):
                        unpublished_progress_at = None
                        unpublished_retired = 0
                        if report["state"] == "awaiting_observation":
                            runtime.request_recovery_snapshots(report["last_progress_at"])
                            return
                        if report["state"] == "idle":
                            return
                    else:
                        unpublished_retired += result["retired"]
                        unpublished_progress_at = report["last_progress_at"]
                        delay = max(delay, 1.0)
        except _READ_ERRORS as exc:
            delay = blocked_delay
            blocked_delay = min(30.0, blocked_delay * 2)
            reason = "permission_denied" if isinstance(exc, PermissionError) else "store_unavailable"
            report = _advance_report(
                runtime,
                runtime_id=runtime_id,
                instance_id=instance_id,
                result=_blocked(reason),
                next_check_at=_utc_timestamp(datetime.now(timezone.utc) + timedelta(seconds=delay)),
                unpublished_retired=unpublished_retired,
                unpublished_progress_at=unpublished_progress_at,
            )
            if _write_report(runtime, report, stop_event=stop_event):
                unpublished_progress_at = None
                unpublished_retired = 0
        stop_event.wait(delay)
