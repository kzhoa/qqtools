"""Small, side-effectful process adapters."""

import errno
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ProcessObservationState = Literal["present", "absent", "unknown"]
ProcessReadReason = Literal["invalid_identity", "read_failed", "invalid_process_stat", "unsupported_probe"]

_PROC_ROOT = Path("/proc")
_PLATFORM = sys.platform


@dataclass(frozen=True, slots=True)
class ProcessIdentityRead:
    """Lossless observation of a process identifier and its start time."""

    state: ProcessObservationState
    start_time_ticks: int | None = None
    reason: ProcessReadReason | None = None

    def __post_init__(self) -> None:
        if self.state == "present":
            if type(self.start_time_ticks) is not int or self.start_time_ticks < 0 or self.reason is not None:
                raise ValueError("present process identity reads require nonnegative start-time ticks only")
        elif self.state == "absent":
            if self.start_time_ticks is not None or self.reason is not None:
                raise ValueError("absent process identity reads cannot include ticks or a reason")
        elif self.state == "unknown":
            if self.start_time_ticks is not None or self.reason not in {
                "invalid_identity",
                "read_failed",
                "invalid_process_stat",
                "unsupported_probe",
            }:
                raise ValueError("unknown process identity reads require a bounded reason only")
        else:
            raise ValueError("invalid process identity read state")


@dataclass(frozen=True, slots=True)
class ProcessPresence:
    """Lossless observation of process or process-group presence."""

    state: ProcessObservationState
    reason: ProcessReadReason | None = None

    def __post_init__(self) -> None:
        if self.state in {"present", "absent"}:
            if self.reason is not None:
                raise ValueError("present and absent process presence cannot include a reason")
        elif self.state == "unknown":
            if self.reason not in {
                "invalid_identity",
                "read_failed",
                "invalid_process_stat",
                "unsupported_probe",
            }:
                raise ValueError("unknown process presence requires a bounded reason")
        else:
            raise ValueError("invalid process presence state")


def _valid_process_identifier(process_id: object) -> bool:
    return type(process_id) is int and process_id > 0


def _proc_interface_supported() -> bool:
    if not isinstance(_PLATFORM, str) or not _PLATFORM.startswith("linux"):
        return False
    try:
        return _PROC_ROOT.is_dir()
    except (OSError, AttributeError, TypeError):
        return False


def _invalid_identity() -> ProcessIdentityRead:
    return ProcessIdentityRead(state="unknown", reason="invalid_identity")


def _read_process_stat(process_id: int) -> ProcessIdentityRead:
    if not _proc_interface_supported():
        return ProcessIdentityRead(state="unknown", reason="unsupported_probe")
    try:
        stat_bytes = (_PROC_ROOT / str(process_id) / "stat").read_bytes()
    except FileNotFoundError:
        return ProcessIdentityRead(state="absent")
    except UnicodeError:
        return ProcessIdentityRead(state="unknown", reason="invalid_process_stat")
    except OSError:
        return ProcessIdentityRead(state="unknown", reason="read_failed")
    try:
        stat = stat_bytes.decode("utf-8")
    except (AttributeError, UnicodeError):
        return ProcessIdentityRead(state="unknown", reason="invalid_process_stat")

    closing_parenthesis = stat.rfind(")")
    if closing_parenthesis < 0 or stat.find("(") < 0 or stat.find("(") > closing_parenthesis:
        return ProcessIdentityRead(state="unknown", reason="invalid_process_stat")
    fields = stat[closing_parenthesis + 1 :].split()
    if len(fields) <= 19:
        return ProcessIdentityRead(state="unknown", reason="invalid_process_stat")
    try:
        start_time_ticks = int(fields[19])
    except (TypeError, ValueError):
        return ProcessIdentityRead(state="unknown", reason="invalid_process_stat")
    if start_time_ticks < 0:
        return ProcessIdentityRead(state="unknown", reason="invalid_process_stat")
    return ProcessIdentityRead(state="present", start_time_ticks=start_time_ticks)


def read_process_identity(pid: int) -> ProcessIdentityRead:
    """Read a process start time without erasing observable failure states."""
    if not _valid_process_identifier(pid):
        return _invalid_identity()
    return _read_process_stat(pid)


def read_process_presence(pid: int, *, is_group: bool) -> ProcessPresence:
    """Observe process or process-group presence with a zero-signal probe."""
    if not _valid_process_identifier(pid):
        return ProcessPresence(state="unknown", reason="invalid_identity")

    if is_group:
        probe = getattr(os, "killpg", None)
        if probe is None:
            return ProcessPresence(state="unknown", reason="unsupported_probe")
    else:
        probe = getattr(os, "kill", None)
        if probe is None:
            return ProcessPresence(state="unknown", reason="unsupported_probe")
    try:
        probe(pid, 0)
    except OSError as error:
        if error.errno == errno.ESRCH or isinstance(error, ProcessLookupError):
            return ProcessPresence(state="absent")
        if error.errno in {
            errno.ENOSYS,
            getattr(errno, "ENOTSUP", errno.ENOSYS),
            getattr(errno, "EOPNOTSUPP", errno.ENOSYS),
        }:
            return ProcessPresence(state="unknown", reason="unsupported_probe")
        return ProcessPresence(state="unknown", reason="read_failed")
    except (AttributeError, NotImplementedError):
        return ProcessPresence(state="unknown", reason="unsupported_probe")
    except (OverflowError, ValueError):
        return ProcessPresence(state="unknown", reason="read_failed")
    return ProcessPresence(state="present")


def is_process_group_alive(process_group_id: int | None) -> bool:
    if not process_group_id:
        return False
    try:
        os.killpg(process_group_id, 0)
    except OSError:
        return False
    return True


def is_process_alive(process_id: int | None) -> bool:
    if not process_id:
        return False
    try:
        os.kill(process_id, 0)
    except OSError:
        return False
    return True


def process_start_time_ticks(process_id: int | None) -> int | None:
    if not process_id:
        return None
    try:
        stat = (Path("/proc") / str(process_id) / "stat").read_text(encoding="utf-8")
        fields = stat.rsplit(")", 1)[1].split()
        return int(fields[19])
    except (FileNotFoundError, IndexError, ValueError, OSError):
        return None


def terminate_process_group(process_group_id: int, *, grace_seconds: float = 5.0) -> bool:
    """Terminate a process group and return only after its absence is confirmed."""
    if not is_process_group_alive(process_group_id):
        return True
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except OSError:
        return not is_process_group_alive(process_group_id)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not is_process_group_alive(process_group_id):
            return True
        time.sleep(0.05)
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except OSError:
        pass
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not is_process_group_alive(process_group_id):
            return True
        time.sleep(0.05)
    return not is_process_group_alive(process_group_id)
