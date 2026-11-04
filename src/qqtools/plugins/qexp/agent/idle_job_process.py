"""Linux process supervision for machine-scoped idle jobs.

The caller owns the durable intent.  This module owns only the short-lived
supervisor, its private process-group guardian, and the resulting process and
exit observations.  The user command is deliberately kept out of the shell;
the process-group relationship is a lifecycle guard, not a security sandbox.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import math
import os
import select
import selectors
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..infrastructure.process import (
    ProcessIdentityRead,
    ProcessObservationState,
    read_process_identity,
    read_process_presence,
)
from ..runner import _configure_process_group_guardian
from ..runtime.store import atomic_replace, read_json_limited

IDLE_JOB_PROTOCOL_VERSION = 1
MAX_IDLE_JOB_RECORD_BYTES = 65_536
MAX_IDLE_JOB_STOP_REASON_BYTES = 1_024
MAX_IDLE_JOB_LOG_BYTES = 1_048_576
MAX_IDLE_JOB_LOG_CHUNK_BYTES = 65_536
IDLE_JOB_POLL_SECONDS = 0.05
IDLE_JOB_REGISTRATION_TIMEOUT_SECONDS = 10.0
_MODULE_NAME = "qqtools.plugins.qexp.agent.idle_job_process"

_PR_SET_PDEATHSIG = 1
_PR_SET_CHILD_SUBREAPER = 36
_PROC_ROOT = Path("/proc")
_BOOT_ID_PATH = _PROC_ROOT / "sys/kernel/random/boot_id"
_STOP_SENTINEL = object()

__all__ = [
    "IDLE_JOB_PROTOCOL_VERSION",
    "MAX_IDLE_JOB_LOG_BYTES",
    "MAX_IDLE_JOB_RECORD_BYTES",
    "MAX_IDLE_JOB_STOP_REASON_BYTES",
    "current_boot_id",
    "idle_job_processes_exited",
    "read_idle_job_process",
    "request_idle_job_stop",
    "spawn_idle_job",
]


def _require_linux() -> None:
    if not sys.platform.startswith("linux") or not _PROC_ROOT.is_dir():
        raise RuntimeError("idle job supervision requires Linux and /proc")


def current_boot_id() -> str:
    """Return the current Linux boot identity as a canonical UUID string."""
    _require_linux()
    try:
        value = _BOOT_ID_PATH.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError("could not read the Linux boot identity") from exc
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as exc:
        raise RuntimeError("Linux boot identity is not a UUID") from exc
    canonical = str(parsed)
    if value != canonical:
        raise RuntimeError("Linux boot identity is not canonical")
    return canonical


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _bounded_text(value: object, field: str, *, max_bytes: int, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{field} must be a non-empty string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field} must be valid UTF-8") from exc
    if len(encoded) > max_bytes:
        raise ValueError(f"{field} exceeds its {max_bytes}-byte limit")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain NUL")
    return value


def _positive_int(value: object, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _timeout(value: object, field: str = "stop_timeout_seconds") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0 or result > 86_400:
        raise ValueError(f"{field} must be a finite positive number no greater than 86400")
    return result


def _boot_id_value(value: object, field: str = "boot_id") -> str:
    value = _bounded_text(value, field, max_bytes=64)
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a UUID") from exc
    canonical = str(parsed)
    if value != canonical:
        raise ValueError(f"{field} must be a canonical UUID")
    return canonical


def _validate_command(command: object, field: str = "command") -> tuple[str, ...]:
    if type(command) is not tuple or not command:
        raise ValueError(f"{field} must be a non-empty tuple")
    result: list[str] = []
    for index, argument in enumerate(command):
        result.append(_bounded_text(argument, f"{field}[{index}]", max_bytes=32_768, allow_empty=index > 0))
    return tuple(result)


def _validate_gpu_ids(gpu_ids: object, field: str = "gpu_ids") -> tuple[int, ...]:
    if type(gpu_ids) is not tuple:
        raise ValueError(f"{field} must be a tuple")
    result: list[int] = []
    for index, gpu_id in enumerate(gpu_ids):
        if type(gpu_id) is not int or gpu_id < 0:
            raise ValueError(f"{field}[{index}] must be a non-negative integer")
        result.append(gpu_id)
    return tuple(result)


def _validate_working_directory(value: object, field: str = "working_directory") -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{field} must be a path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{field} must be absolute")
    return path


def _validate_intent(intent: object) -> dict[str, Any]:
    if not isinstance(intent, dict):
        raise ValueError("idle_job_intent must be an object")
    required = {
        "version",
        "runtime_id",
        "instance_id",
        "command",
        "working_directory",
        "gpu_ids",
        "stop_timeout_seconds",
        "parent_pid",
        "parent_start_ticks",
        "boot_id",
    }
    if set(intent) != required:
        raise ValueError("idle_job_intent has an unsupported or missing field")
    if type(intent["version"]) is not int or intent["version"] != IDLE_JOB_PROTOCOL_VERSION:
        raise ValueError("idle_job_intent version is unsupported")
    runtime_id = _bounded_text(intent["runtime_id"], "runtime_id", max_bytes=1_024)
    instance_id = _bounded_text(intent["instance_id"], "instance_id", max_bytes=1_024)
    command_value = intent["command"]
    if type(command_value) is not list or not command_value:
        raise ValueError("intent command must be a non-empty list")
    command = tuple(
        _bounded_text(argument, f"command[{index}]", max_bytes=32_768, allow_empty=index > 0)
        for index, argument in enumerate(command_value)
    )
    working_directory = _validate_working_directory(intent["working_directory"])
    gpu_value = intent["gpu_ids"]
    if type(gpu_value) is not list:
        raise ValueError("intent gpu_ids must be a list")
    gpu_ids = tuple(_nonnegative_int(gpu_id, f"gpu_ids[{index}]") for index, gpu_id in enumerate(gpu_value))
    return {
        "version": IDLE_JOB_PROTOCOL_VERSION,
        "runtime_id": runtime_id,
        "instance_id": instance_id,
        "command": command,
        "working_directory": working_directory,
        "gpu_ids": gpu_ids,
        "stop_timeout_seconds": _timeout(intent["stop_timeout_seconds"]),
        "parent_pid": _positive_int(intent["parent_pid"], "parent_pid"),
        "parent_start_ticks": _nonnegative_int(intent["parent_start_ticks"], "parent_start_ticks"),
        "boot_id": _boot_id_value(intent["boot_id"]),
    }


def _read_intent(instance_dir: Path) -> dict[str, Any]:
    envelope = read_json_limited(
        instance_dir / "intent.json",
        max_bytes=MAX_IDLE_JOB_RECORD_BYTES,
        record_type="idle_job_intent",
    )
    return _validate_intent(envelope.get("idle_job_intent"))


def _validate_process_record(process: object) -> dict[str, Any]:
    if not isinstance(process, dict):
        raise ValueError("idle_job_process must be an object")
    required = {
        "version",
        "runtime_id",
        "instance_id",
        "boot_id",
        "supervisor_pid",
        "supervisor_start_ticks",
        "pid",
        "start_time_ticks",
        "process_group_id",
        "started_at",
    }
    if not required.issubset(process):
        raise ValueError("idle_job_process is missing an identity field")
    if type(process["version"]) is not int or process["version"] != IDLE_JOB_PROTOCOL_VERSION:
        raise ValueError("idle_job_process version is unsupported")
    _bounded_text(process["runtime_id"], "idle_job_process.runtime_id", max_bytes=1_024)
    _bounded_text(process["instance_id"], "idle_job_process.instance_id", max_bytes=1_024)
    _boot_id_value(process["boot_id"], "idle_job_process.boot_id")
    _positive_int(process["supervisor_pid"], "idle_job_process.supervisor_pid")
    _nonnegative_int(process["supervisor_start_ticks"], "idle_job_process.supervisor_start_ticks")
    pid = _positive_int(process["pid"], "idle_job_process.pid")
    _nonnegative_int(process["start_time_ticks"], "idle_job_process.start_time_ticks")
    if process["process_group_id"] != pid:
        raise ValueError("idle_job_process process group must be led by the guardian")
    _positive_int(process["process_group_id"], "idle_job_process.process_group_id")
    _bounded_text(process["started_at"], "idle_job_process.started_at", max_bytes=128)
    return process


def _read_process_record(instance_dir: Path) -> dict[str, Any] | None:
    try:
        envelope = read_json_limited(
            instance_dir / "process.json",
            max_bytes=MAX_IDLE_JOB_RECORD_BYTES,
            record_type="idle_job_process",
        )
    except FileNotFoundError:
        return None
    process = envelope.get("idle_job_process")
    return _validate_process_record(process)


def _validate_supervisor_record(record: object) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("idle_job_supervisor must be an object")
    required = {"version", "runtime_id", "instance_id", "boot_id", "pid", "start_time_ticks"}
    if not required.issubset(record):
        raise ValueError("idle_job_supervisor is missing an identity field")
    if type(record["version"]) is not int or record["version"] != IDLE_JOB_PROTOCOL_VERSION:
        raise ValueError("idle_job_supervisor version is unsupported")
    _bounded_text(record["runtime_id"], "idle_job_supervisor.runtime_id", max_bytes=1_024)
    _bounded_text(record["instance_id"], "idle_job_supervisor.instance_id", max_bytes=1_024)
    _boot_id_value(record["boot_id"], "idle_job_supervisor.boot_id")
    _positive_int(record["pid"], "idle_job_supervisor.pid")
    _nonnegative_int(record["start_time_ticks"], "idle_job_supervisor.start_time_ticks")
    return record


def _read_supervisor_record(instance_dir: Path) -> dict[str, Any]:
    envelope = read_json_limited(
        instance_dir / "supervisor.json",
        max_bytes=MAX_IDLE_JOB_RECORD_BYTES,
        record_type="idle_job_supervisor",
    )
    return _validate_supervisor_record(envelope.get("idle_job_supervisor"))


def _identity_matches(pid: int, start_ticks: int, boot_id: str) -> bool:
    try:
        if current_boot_id() != boot_id:
            return False
    except RuntimeError:
        return False
    observed = read_process_identity(pid)
    return observed.state == "present" and observed.start_time_ticks == start_ticks


def _identity_state(pid: int, start_ticks: int, boot_id: str) -> ProcessObservationState:
    try:
        if current_boot_id() != boot_id:
            return "absent"
    except RuntimeError:
        return "unknown"
    observed = read_process_identity(pid)
    if observed.state == "present":
        return "present" if observed.start_time_ticks == start_ticks else "absent"
    return observed.state


def _pidfd_available() -> bool:
    return callable(getattr(os, "pidfd_open", None)) and callable(getattr(signal, "pidfd_send_signal", None))


def _signal_exact_process(pid: int, start_ticks: int, boot_id: str, signum: int) -> bool:
    """Signal one exact process identity through a pidfd, never a bare PID."""
    if not _pidfd_available() or not _identity_matches(pid, start_ticks, boot_id):
        return False
    descriptor: int | None = None
    try:
        descriptor = os.pidfd_open(pid, 0)
        if not _identity_matches(pid, start_ticks, boot_id):
            return False
        signal.pidfd_send_signal(descriptor, signum)
        return True
    except (OSError, ProcessLookupError, PermissionError, ValueError):
        return False
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _supports_prctl() -> bool:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        return callable(getattr(libc, "prctl"))
    except (AttributeError, OSError):
        return False


def _set_parent_death_signal(signum: int) -> None:
    if not _supports_prctl():
        raise RuntimeError("Linux prctl is unavailable")
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    if prctl(_PR_SET_PDEATHSIG, signum, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, "prctl(PR_SET_PDEATHSIG) failed")


def _set_child_subreaper() -> None:
    if not _supports_prctl():
        raise RuntimeError("Linux prctl is unavailable")
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    if prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, "prctl(PR_SET_CHILD_SUBREAPER) failed")


def spawn_idle_job(
    instance_dir: Path,
    *,
    runtime_id: str,
    instance_id: str,
    command: tuple[str, ...],
    working_directory: Path,
    gpu_ids: tuple[int, ...],
    stop_timeout_seconds: float,
    parent_pid: int,
    parent_start_ticks: int,
) -> subprocess.Popen[bytes]:
    """Start a supervisor after validating the caller's persisted intent.

    The caller must have atomically written ``intent.json`` before calling this
    function.  This function never creates or updates that intent.
    """
    _require_linux()
    if not isinstance(instance_dir, Path) or not instance_dir.is_dir():
        raise ValueError("instance_dir must be an existing directory")
    expected_boot_id = current_boot_id()
    intent = _read_intent(instance_dir)
    expected = {
        "runtime_id": _bounded_text(runtime_id, "runtime_id", max_bytes=1_024),
        "instance_id": _bounded_text(instance_id, "instance_id", max_bytes=1_024),
        "command": _validate_command(command),
        "working_directory": _validate_working_directory(working_directory),
        "gpu_ids": _validate_gpu_ids(gpu_ids),
        "stop_timeout_seconds": _timeout(stop_timeout_seconds),
        "parent_pid": _positive_int(parent_pid, "parent_pid"),
        "parent_start_ticks": _nonnegative_int(parent_start_ticks, "parent_start_ticks"),
        "boot_id": expected_boot_id,
    }
    for field, value in expected.items():
        if intent[field] != value:
            raise ValueError(f"intent {field} does not match spawn arguments")
    for path_name in (
        "instance.lock",
        "supervisor.json",
        "process.json",
        "exit.json",
        "stop.json",
        "command.log",
        "command.log.1",
    ):
        if (instance_dir / path_name).exists():
            raise RuntimeError("idle job instance directory has already been used")
    if not _pidfd_available():
        raise RuntimeError("idle job supervision requires pidfd_open and pidfd_send_signal")
    if not _identity_matches(parent_pid, parent_start_ticks, expected_boot_id):
        raise RuntimeError("idle job launch parent identity is not live")
    command_line = [sys.executable, "-m", _MODULE_NAME, "--supervisor", "--instance-dir", str(instance_dir)]
    return subprocess.Popen(
        command_line,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def request_idle_job_stop(instance_dir: Path, reason: str) -> None:
    """Publish a bounded stop request without trusting or signalling a PID."""
    _require_linux()
    if not isinstance(instance_dir, Path) or not instance_dir.is_dir():
        raise ValueError("instance_dir must be an existing directory")
    _bounded_text(reason, "reason", max_bytes=MAX_IDLE_JOB_STOP_REASON_BYTES)
    atomic_replace(instance_dir / "stop.json", {"reason": reason})


def read_idle_job_process(instance_dir: Path) -> dict[str, Any] | None:
    """Read the bounded guardian registration, returning its inner record."""
    _require_linux()
    if not isinstance(instance_dir, Path):
        raise ValueError("instance_dir must be a Path")
    return _read_process_record(instance_dir)


def _instance_lock_is_free(instance_dir: Path) -> bool:
    path = instance_dir / "instance.lock"
    if not path.exists():
        return True
    handle = None
    try:
        handle = path.open("a+")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EAGAIN}:
            return False
        return False
    else:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            return False
        return True
    finally:
        if handle is not None:
            handle.close()


def idle_job_processes_exited(instance_dir: Path) -> bool:
    """Return true only after the exact registered group is positively absent."""
    _require_linux()
    if not isinstance(instance_dir, Path) or not instance_dir.is_dir():
        return False
    try:
        intent = _read_intent(instance_dir)
        process = _read_process_record(instance_dir)
        boot_id = current_boot_id()
    except (OSError, RuntimeError, TypeError, ValueError, KeyError):
        return False
    if process is not None:
        if (
            process["runtime_id"] != intent["runtime_id"]
            or process["instance_id"] != intent["instance_id"]
            or process["boot_id"] != intent["boot_id"]
        ):
            return False
        if process["boot_id"] != boot_id:
            return True
        presence = read_process_presence(process["process_group_id"], is_group=True)
        return presence.state == "absent"
    # A supervisor that registered and then exited cannot issue a future GO.
    # The guardian persists process.json before accepting GO, so an absent
    # process record here excludes user execution even if the agent is alive.
    try:
        supervisor = _read_supervisor_record(instance_dir)
    except FileNotFoundError:
        supervisor = None
    except (OSError, RuntimeError, TypeError, ValueError, KeyError):
        return False
    if supervisor is not None:
        if any(supervisor[key] != intent[key] for key in ("runtime_id", "instance_id", "boot_id")):
            return False
        if _identity_state(supervisor["pid"], supervisor["start_time_ticks"], supervisor["boot_id"]) != "absent":
            return False
        try:
            if _read_process_record(instance_dir) is not None:
                return idle_job_processes_exited(instance_dir)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError):
            return False
        return _instance_lock_is_free(instance_dir)
    try:
        terminal = read_json_limited(
            instance_dir / "exit.json", max_bytes=MAX_IDLE_JOB_RECORD_BYTES, record_type="idle_job_exit"
        ).get("idle_job_exit")
    except FileNotFoundError:
        terminal = None
    except (OSError, RuntimeError, TypeError, ValueError, KeyError):
        return False
    if isinstance(terminal, dict) and (
        terminal.get("version") == 1
        and terminal.get("reason") == "launch_failed"
        and terminal.get("returncode") is None
        and all(terminal.get(key) == intent[key] for key in ("runtime_id", "instance_id", "boot_id"))
    ):
        # Only the launch caller writes this after synchronous spawn failure;
        # spawn_idle_job rejects any instance with a terminal record.
        return _instance_lock_is_free(instance_dir)
    parent_state = _identity_state(intent["parent_pid"], intent["parent_start_ticks"], intent["boot_id"])
    if parent_state in {"present", "unknown"}:
        return False
    return _instance_lock_is_free(instance_dir)


class _InstanceLock:
    """A nonblocking lock acquisition that can observe parent death while waiting."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = None

    def acquire(self, should_continue: Any) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        while should_continue():
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise
                time.sleep(IDLE_JOB_POLL_SECONDS)
                continue
            return True
        self.close()
        return False

    def close(self) -> None:
        if self.handle is None:
            return
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


class _BoundedCommandLog:
    def __init__(self, instance_dir: Path) -> None:
        self.path = instance_dir / "command.log"
        self.previous_path = instance_dir / "command.log.1"
        self.handle = self.path.open("ab")
        self.size = self.path.stat().st_size
        if self.size > MAX_IDLE_JOB_LOG_BYTES:
            self._rotate()

    def _rotate(self) -> None:
        self.handle.close()
        if self.previous_path.exists():
            self.previous_path.unlink()
        if self.path.exists():
            os.replace(self.path, self.previous_path)
        self.handle = self.path.open("ab")
        self.size = 0

    def write(self, value: bytes) -> None:
        offset = 0
        while offset < len(value):
            if self.size >= MAX_IDLE_JOB_LOG_BYTES:
                self._rotate()
            available = MAX_IDLE_JOB_LOG_BYTES - self.size
            chunk_size = min(available, MAX_IDLE_JOB_LOG_CHUNK_BYTES, len(value) - offset)
            chunk = value[offset : offset + chunk_size]
            self.handle.write(chunk)
            self.handle.flush()
            self.size += len(chunk)
            offset += len(chunk)

    def close(self) -> None:
        self.handle.close()


class _SupervisorOutput:
    def __init__(self, stream: Any, log: _BoundedCommandLog) -> None:
        self.stream = stream
        self.log = log
        self.selector = selectors.DefaultSelector()
        self.closed = False
        descriptor = stream.fileno()
        os.set_blocking(descriptor, False)
        self.selector.register(descriptor, selectors.EVENT_READ)

    def drain(self, timeout: float = 0.0) -> None:
        if self.closed:
            if timeout:
                time.sleep(timeout)
            return
        for key, _mask in self.selector.select(timeout):
            try:
                value = os.read(key.fd, MAX_IDLE_JOB_LOG_CHUNK_BYTES)
            except BlockingIOError:
                continue
            except OSError:
                value = b""
            if value:
                self.log.write(value)
            else:
                self.selector.unregister(key.fd)
                self.closed = True
                try:
                    self.stream.close()
                except OSError:
                    pass

    def close(self) -> None:
        # Teardown must not wait for a live writer or retry a failing log sink.
        self.selector.close()
        self.closed = True
        try:
            self.stream.close()
        except OSError:
            pass


def _proc_group_members(process_group_id: int, own_pid: int) -> set[int] | None:
    try:
        entries = list(_PROC_ROOT.iterdir())
    except OSError:
        return None
    members: set[int] = set()
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            pid = int(entry.name)
            raw = (entry / "stat").read_text(encoding="ascii")
            closing = raw.rfind(")")
            if closing < 0:
                return None
            fields = raw[closing + 1 :].split()
            if len(fields) <= 4:
                return None
            group_id = int(fields[2])
        except (OSError, UnicodeError, ValueError):
            continue
        if group_id == process_group_id and pid != own_pid:
            members.add(pid)
    return members


def _reap_orphan_children(leader_pid: int) -> int | None:
    leader_returncode = None
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return leader_returncode
        except InterruptedError:
            continue
        if pid == 0:
            return leader_returncode
        if pid == leader_pid:
            leader_returncode = os.waitstatus_to_exitcode(status)


_guardian_stop_requested = False
_guardian_group_term_sent = False


def _guardian_term_handler(_signum: int, _frame: Any) -> None:
    global _guardian_group_term_sent, _guardian_stop_requested
    _guardian_stop_requested = True
    if _guardian_group_term_sent:
        return
    _guardian_group_term_sent = True
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        os.killpg(os.getpgrp(), signal.SIGTERM)
    except OSError:
        pass


def _guardian_send_group_term() -> None:
    global _guardian_group_term_sent
    if _guardian_group_term_sent:
        return
    _guardian_group_term_sent = True
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        os.killpg(os.getpgrp(), signal.SIGTERM)
    except OSError:
        pass


def _guardian_relay_output(selector: selectors.BaseSelector, child_stream: Any, output_broken: list[bool]) -> bool:
    for key, _mask in selector.select(0.0):
        try:
            value = os.read(key.fd, MAX_IDLE_JOB_LOG_CHUNK_BYTES)
        except BlockingIOError:
            continue
        except OSError:
            value = b""
        if not value:
            selector.unregister(key.fd)
            child_stream.close()
            continue
        if output_broken[0]:
            continue
        offset = 0
        while offset < len(value):
            try:
                written = os.write(sys.stdout.fileno(), value[offset : offset + MAX_IDLE_JOB_LOG_CHUNK_BYTES])
            except BrokenPipeError:
                output_broken[0] = True
                break
            except OSError:
                output_broken[0] = True
                break
            if written <= 0:
                output_broken[0] = True
                break
            offset += written
    return bool(selector.get_map())


def _wait_for_go() -> bool:
    descriptor = sys.stdin.fileno()
    os.set_blocking(descriptor, False)
    while not _guardian_stop_requested:
        try:
            ready, _write, _error = select.select([descriptor], [], [], IDLE_JOB_POLL_SECONDS)
        except InterruptedError:
            continue
        if not ready:
            continue
        try:
            value = os.read(descriptor, 1)
        except BlockingIOError:
            continue
        return value == b"G"
    return False


def _run_guardian(instance_dir: Path) -> int:
    global _guardian_group_term_sent, _guardian_stop_requested
    _guardian_group_term_sent = False
    _guardian_stop_requested = False
    intent = _read_intent(instance_dir)
    boot_id = current_boot_id()
    if intent["boot_id"] != boot_id:
        raise RuntimeError("idle job intent belongs to another boot")
    supervisor = _read_supervisor_record(instance_dir)
    expected_supervisor_pid = os.getppid()
    if supervisor["pid"] != expected_supervisor_pid or supervisor["boot_id"] != boot_id:
        raise RuntimeError("idle job supervisor identity does not match guardian parent")
    if not _identity_matches(supervisor["pid"], supervisor["start_time_ticks"], boot_id):
        raise RuntimeError("idle job supervisor identity is not live")
    _configure_process_group_guardian(expected_supervisor_pid)
    _set_child_subreaper()
    if os.getpgrp() != os.getpid():
        raise RuntimeError("idle job guardian is not the process-group leader")
    guardian_identity = read_process_identity(os.getpid())
    if guardian_identity.state != "present" or guardian_identity.start_time_ticks is None:
        raise RuntimeError("could not determine idle job guardian identity")
    process_record = {
        "version": IDLE_JOB_PROTOCOL_VERSION,
        "runtime_id": intent["runtime_id"],
        "instance_id": intent["instance_id"],
        "boot_id": boot_id,
        "supervisor_pid": supervisor["pid"],
        "supervisor_start_ticks": supervisor["start_time_ticks"],
        "pid": os.getpid(),
        "start_time_ticks": guardian_identity.start_time_ticks,
        "process_group_id": os.getpid(),
        "started_at": _utc_now(),
    }
    _validate_process_record(process_record)
    atomic_replace(instance_dir / "process.json", {"idle_job_process": process_record})
    signal.signal(signal.SIGTERM, _guardian_term_handler)
    signal.signal(signal.SIGINT, _guardian_term_handler)
    if not _wait_for_go():
        return 128 + signal.SIGTERM if _guardian_stop_requested else 2
    if _guardian_stop_requested:
        return 128 + signal.SIGTERM
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in intent["gpu_ids"])
    try:
        child = subprocess.Popen(
            list(intent["command"]),
            cwd=intent["working_directory"],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=False,
        )
    except (OSError, ValueError):
        return 127
    if child.stdout is None:
        child.kill()
        child.wait()
        return 127
    child_descriptor = child.stdout.fileno()
    os.set_blocking(child_descriptor, False)
    child_selector = selectors.DefaultSelector()
    child_selector.register(child_descriptor, selectors.EVENT_READ)
    output_broken = [False]
    termination_deadline: float | None = None
    child_returncode: int | None = None
    try:
        while True:
            _guardian_relay_output(child_selector, child.stdout, output_broken)
            if child_returncode is None:
                child_returncode = child.poll()
            reaped_returncode = _reap_orphan_children(child.pid)
            if reaped_returncode is not None:
                child.returncode = child_returncode = reaped_returncode
            members = _proc_group_members(os.getpgrp(), os.getpid())
            if members is None:
                _guardian_send_group_term()
                os.killpg(os.getpgrp(), signal.SIGKILL)
            elif _guardian_stop_requested:
                if termination_deadline is None:
                    termination_deadline = time.monotonic() + intent["stop_timeout_seconds"]
                if time.monotonic() >= termination_deadline:
                    os.killpg(os.getpgrp(), signal.SIGKILL)
            elif child_returncode is not None and members:
                if termination_deadline is None:
                    _guardian_send_group_term()
                    termination_deadline = time.monotonic() + intent["stop_timeout_seconds"]
                elif time.monotonic() >= termination_deadline:
                    os.killpg(os.getpgrp(), signal.SIGKILL)
            if child_returncode is not None and not members:
                while child_selector.get_map():
                    _guardian_relay_output(child_selector, child.stdout, output_broken)
                    if child_selector.get_map():
                        time.sleep(IDLE_JOB_POLL_SECONDS)
                return child_returncode
            if not child_selector.get_map() and child_returncode is not None and not members:
                return child_returncode
            time.sleep(IDLE_JOB_POLL_SECONDS)
    finally:
        child_selector.close()
        if child.poll() is None:
            try:
                os.killpg(os.getpgrp(), signal.SIGKILL)
            except OSError:
                pass
        try:
            child.wait(timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            pass


_supervisor_stop_event: threading.Event | None = None


def _supervisor_stop_handler(_signum: int, _frame: Any) -> None:
    if _supervisor_stop_event is not None:
        _supervisor_stop_event.set()


def _read_stop_reason(instance_dir: Path) -> str | None | object:
    try:
        envelope = read_json_limited(
            instance_dir / "stop.json",
            max_bytes=MAX_IDLE_JOB_RECORD_BYTES,
            record_type="idle_job_stop",
        )
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError, KeyError):
        return _STOP_SENTINEL
    reason = envelope.get("reason")
    if not isinstance(reason, str):
        return _STOP_SENTINEL
    try:
        return _bounded_text(reason, "stop reason", max_bytes=MAX_IDLE_JOB_STOP_REASON_BYTES)
    except ValueError:
        return _STOP_SENTINEL


def _supervisor_stop_reason(instance_dir: Path, intent: dict[str, Any], boot_id: str) -> str | None:
    reason = _read_stop_reason(instance_dir)
    if reason is _STOP_SENTINEL:
        return "invalid_stop_request"
    if isinstance(reason, str):
        return reason
    if _supervisor_stop_event is not None and _supervisor_stop_event.is_set():
        return "supervisor_signal"
    if not _identity_matches(intent["parent_pid"], intent["parent_start_ticks"], boot_id):
        return "parent_exit"
    return None


def _validate_registration(process: dict[str, Any], intent: dict[str, Any], supervisor: dict[str, Any]) -> bool:
    return bool(
        process["runtime_id"] == intent["runtime_id"]
        and process["instance_id"] == intent["instance_id"]
        and process["boot_id"] == intent["boot_id"]
        and process["supervisor_pid"] == supervisor["pid"]
        and process["supervisor_start_ticks"] == supervisor["start_time_ticks"]
        and process["process_group_id"] == process["pid"]
    )


def _acquire_instance_lock(lock: _InstanceLock, intent: dict[str, Any], boot_id: str) -> bool:
    return lock.acquire(lambda: _identity_matches(intent["parent_pid"], intent["parent_start_ticks"], boot_id))


def _send_guardian_go(guardian: subprocess.Popen[bytes]) -> bool:
    if guardian.stdin is None:
        return False
    try:
        guardian.stdin.write(b"G")
        guardian.stdin.flush()
        guardian.stdin.close()
    except (BrokenPipeError, OSError, ValueError):
        return False
    return True


def _run_supervisor(instance_dir: Path) -> int:
    global _supervisor_stop_event
    _require_linux()
    intent = _read_intent(instance_dir)
    boot_id = current_boot_id()
    if intent["boot_id"] != boot_id:
        raise RuntimeError("idle job intent belongs to another boot")
    if not _pidfd_available():
        raise RuntimeError("idle job supervision requires pidfd_open and pidfd_send_signal")
    stop_event = threading.Event()
    _supervisor_stop_event = stop_event
    signal.signal(signal.SIGTERM, _supervisor_stop_handler)
    signal.signal(signal.SIGINT, _supervisor_stop_handler)
    lock = _InstanceLock(instance_dir / "instance.lock")
    guardian: subprocess.Popen[bytes] | None = None
    output: _SupervisorOutput | None = None
    process_record: dict[str, Any] | None = None
    guardian_identity: ProcessIdentityRead | None = None
    stop_reason: str | None = None
    try:
        if not _acquire_instance_lock(lock, intent, boot_id):
            return 2
        if not _identity_matches(intent["parent_pid"], intent["parent_start_ticks"], boot_id):
            return 2
        _set_parent_death_signal(signal.SIGTERM)
        if not _identity_matches(intent["parent_pid"], intent["parent_start_ticks"], boot_id):
            return 2
        supervisor_identity = read_process_identity(os.getpid())
        if supervisor_identity.state != "present" or supervisor_identity.start_time_ticks is None:
            raise RuntimeError("could not determine idle job supervisor identity")
        supervisor_record = {
            "version": IDLE_JOB_PROTOCOL_VERSION,
            "runtime_id": intent["runtime_id"],
            "instance_id": intent["instance_id"],
            "boot_id": boot_id,
            "pid": os.getpid(),
            "start_time_ticks": supervisor_identity.start_time_ticks,
        }
        _validate_supervisor_record(supervisor_record)
        atomic_replace(instance_dir / "supervisor.json", {"idle_job_supervisor": supervisor_record})
        if _supervisor_stop_reason(instance_dir, intent, boot_id) is not None:
            return 2
        guardian = subprocess.Popen(
            [sys.executable, "-m", _MODULE_NAME, "--guardian", "--instance-dir", str(instance_dir)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        if guardian.stdout is None:
            raise RuntimeError("idle job guardian did not expose stdout")
        log = _BoundedCommandLog(instance_dir)
        output = _SupervisorOutput(guardian.stdout, log)
        registration_deadline = time.monotonic() + IDLE_JOB_REGISTRATION_TIMEOUT_SECONDS
        while time.monotonic() < registration_deadline and guardian.poll() is None:
            output.drain(IDLE_JOB_POLL_SECONDS)
            try:
                candidate = _read_process_record(instance_dir)
            except (OSError, ValueError, TypeError, KeyError):
                candidate = None
                if (instance_dir / "process.json").exists():
                    break
            if candidate is not None:
                if _validate_registration(candidate, intent, supervisor_record):
                    process_record = candidate
                break
            if _supervisor_stop_reason(instance_dir, intent, boot_id) is not None:
                break
        guardian_identity = read_process_identity(guardian.pid)
        if process_record is not None:
            stop_reason = _supervisor_stop_reason(instance_dir, intent, boot_id)
            if stop_reason is None and _identity_matches(intent["parent_pid"], intent["parent_start_ticks"], boot_id):
                stop_reason = None if _send_guardian_go(guardian) else "guardian_start_failed"
            elif guardian.stdin is not None:
                try:
                    guardian.stdin.close()
                except OSError:
                    pass
            if stop_reason is not None:
                _signal_exact_process(
                    process_record["pid"],
                    process_record["start_time_ticks"],
                    process_record["boot_id"],
                    signal.SIGTERM,
                )
        else:
            stop_reason = _supervisor_stop_reason(instance_dir, intent, boot_id) or "guardian_registration_failed"
            if (
                guardian.poll() is None
                and guardian_identity.state == "present"
                and guardian_identity.start_time_ticks is not None
            ):
                _signal_exact_process(guardian.pid, guardian_identity.start_time_ticks, boot_id, signal.SIGTERM)
        stop_started = time.monotonic() if stop_reason is not None else None
        escalation_sent = False
        while guardian.poll() is None:
            output.drain(IDLE_JOB_POLL_SECONDS)
            if stop_reason is None:
                stop_reason = _supervisor_stop_reason(instance_dir, intent, boot_id)
                if stop_reason is not None:
                    stop_started = time.monotonic()
                    if process_record is not None:
                        _signal_exact_process(
                            process_record["pid"],
                            process_record["start_time_ticks"],
                            process_record["boot_id"],
                            signal.SIGTERM,
                        )
            elif stop_started is not None and not escalation_sent:
                if time.monotonic() - stop_started >= intent["stop_timeout_seconds"]:
                    if process_record is not None:
                        _signal_exact_process(
                            process_record["pid"],
                            process_record["start_time_ticks"],
                            process_record["boot_id"],
                            signal.SIGUSR1,
                        )
                    elif guardian_identity.state == "present" and guardian_identity.start_time_ticks is not None:
                        _signal_exact_process(guardian.pid, guardian_identity.start_time_ticks, boot_id, signal.SIGUSR1)
                    escalation_sent = True
            if guardian.poll() is None:
                continue
        output.drain(0.0)
        guardian_returncode = guardian.returncode
        if process_record is None or guardian_returncode is None:
            return 2
        if process_record["boot_id"] != current_boot_id():
            return 2
        presence = read_process_presence(process_record["process_group_id"], is_group=True)
        if presence.state != "absent":
            return 2
        exit_reason = stop_reason or "command_exit"
        atomic_replace(
            instance_dir / "exit.json",
            {
                "idle_job_exit": {
                    "version": IDLE_JOB_PROTOCOL_VERSION,
                    "runtime_id": intent["runtime_id"],
                    "instance_id": intent["instance_id"],
                    "boot_id": process_record["boot_id"],
                    "returncode": guardian_returncode,
                    "reason": exit_reason,
                    "exited_at": _utc_now(),
                }
            },
        )
        return 0
    finally:
        if guardian is not None and guardian.poll() is None:
            # Log failures and unexpected exceptions must also stop the group.
            # Popen still owns this unreaped child; verify its start identity
            # and signal through a pidfd before bounded output teardown.
            observed = read_process_identity(guardian.pid)
            if observed.state == "present" and observed.start_time_ticks is not None:
                _signal_exact_process(guardian.pid, observed.start_time_ticks, boot_id, signal.SIGUSR1)
            try:
                guardian.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
        if output is not None:
            try:
                output.close()
            finally:
                output.log.close()
        elif guardian is not None and guardian.stdout is not None:
            guardian.stdout.close()
        if guardian is not None and guardian.stdin is not None:
            try:
                guardian.stdin.close()
            except OSError:
                pass
        lock.close()
        _supervisor_stop_event = None


def main(argv: list[str] | None = None) -> int:
    """Run the private supervisor or guardian entry point."""
    import argparse

    parser = argparse.ArgumentParser(prog="qexp-idle-job")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--supervisor", action="store_true")
    mode.add_argument("--guardian", action="store_true")
    parser.add_argument("--instance-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        return (
            _run_supervisor(arguments.instance_dir) if arguments.supervisor else _run_guardian(arguments.instance_dir)
        )
    except (OSError, RuntimeError, TypeError, ValueError, KeyError):
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess entry points.
    raise SystemExit(main())
