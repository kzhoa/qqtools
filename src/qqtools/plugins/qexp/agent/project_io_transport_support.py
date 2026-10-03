"""Shared Project I/O transport identity and protocol-error support."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

from ..infrastructure.host import host_instance_id
from ..runtime.store import read_json_limited

_PROCESS_LIVE = "same_live_process"
_PROCESS_ABSENT = "positively_absent_or_reused"
_PROCESS_UNVERIFIED = "unverified"


class ProjectIOProtocolError(RuntimeError):
    """Executor-local evidence is invalid or cannot safely be reconciled."""


def _process_start_time_ticks(pid: int) -> int | None:
    """Return Linux proc start ticks, or None when the identity cannot be verified."""
    if not sys.platform.startswith("linux") or type(pid) is not int or pid <= 0:
        return None
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except (OSError, UnicodeError):
        return None
    closing = raw.rfind(")")
    if closing < 0:
        return None
    fields = raw[closing + 1 :].split()
    if len(fields) <= 19:
        return None
    try:
        ticks = int(fields[19])
    except ValueError:
        return None
    return ticks if ticks > 0 else None


def inspect_process_identity(pid: int, start_time_ticks: int) -> str:
    """Classify a PID/start-ticks pair without waiting for the process."""
    if not sys.platform.startswith("linux"):
        return _PROCESS_UNVERIFIED
    try:
        proc_root = Path("/proc")
        if not proc_root.is_dir():
            return _PROCESS_UNVERIFIED
        raw = (proc_root / str(pid) / "stat").read_text(encoding="ascii")
    except FileNotFoundError:
        return _PROCESS_ABSENT
    except (PermissionError, OSError, UnicodeError):
        return _PROCESS_UNVERIFIED
    closing = raw.rfind(")")
    if closing < 0:
        return _PROCESS_UNVERIFIED
    fields = raw[closing + 1 :].split()
    if len(fields) <= 19:
        return _PROCESS_UNVERIFIED
    try:
        observed_ticks = int(fields[19])
    except ValueError:
        return _PROCESS_UNVERIFIED
    return _PROCESS_LIVE if observed_ticks == start_time_ticks else _PROCESS_ABSENT


def _resolve_runtime_id(root: Path) -> str:
    identity = read_json_limited(root / "identity.json", max_bytes=4096, record_type="machine_runtime_identity")
    record = identity.get("machine_runtime")
    seed = record.get("instance_id") if isinstance(record, dict) else None
    effective = record.get("runtime_id") if isinstance(record, dict) else None
    if not isinstance(seed, str) or not seed:
        raise ProjectIOProtocolError("machine runtime identity is invalid.")
    if isinstance(effective, str) and re.fullmatch(r"[0-9a-f]{64}", effective):
        return effective
    return hashlib.sha256(f"{seed}\0{host_instance_id()}".encode()).hexdigest()
