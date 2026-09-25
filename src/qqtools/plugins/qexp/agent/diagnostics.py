"""Machine-local evidence and private log capture for qexp agents."""

from __future__ import annotations

import errno
import faulthandler
import fcntl
import json
import os
import re
import stat
import sys
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterator

from ....version import __version__
from ..infrastructure.host import host_instance_id
from ..infrastructure.process import read_process_identity
from ..runtime.paths import machine_runtime_paths
from ..runtime.store import atomic_replace

DEFAULT_LOG_MAX_BYTES = 10 * 1024 * 1024
MIN_LOG_MAX_BYTES = 64 * 1024
MAX_LOG_MAX_BYTES = 1024 * 1024 * 1024
_TMP_ROOT = Path("/tmp")
_MAX_DIAGNOSTIC_BYTES = 128 * 1024
_MAX_TEXT_LENGTH = 160
_MAX_EVICTION_COUNT = (1 << 31) - 1
_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z", re.ASCII)
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_SAFE_TOKEN_RE = re.compile(r"[A-Za-z0-9_.:-]{1,96}\Z", re.ASCII)
_TIMESTAMP_RE = re.compile(r"[0-9TZ:+.\-]{1,40}\Z", re.ASCII)
_PHASE_RANK = {"starting": 0, "active": 1, "stopping": 2, "stopped": 3}
_AGENT_FIELDS = {
    "phase",
    "admitted",
    "pid",
    "pid_start_time_ticks",
    "capture_health",
    "capture_error",
    "stop_reason",
    "handled_signal",
    "primary_exception",
    "cleanup_outcome",
    "cleanup_steps",
    "finalized_at",
}
_LAUNCHER_FIELDS = {
    "startup_outcome",
    "observed_at",
    "timeout_trigger",
    "timeout_triggered_at",
    "signal_attempts",
    "wait_status",
}
_OBSERVER_FIELDS = {"liveness", "observed_at", "reason", "coverage", "abnormal_exit_unknown"}


def parse_log_size(value: str | int) -> int:
    """Parse bytes or an exact binary unit, enforcing the supported range."""
    if type(value) is int:
        size = value
    elif isinstance(value, str):
        match = re.fullmatch(r"([0-9]+)(KiB|MiB|GiB)?", value, re.ASCII)
        if match is None:
            raise ValueError("log size must be decimal bytes or an integer followed by KiB, MiB, or GiB")
        amount = int(match.group(1), 10)
        unit = match.group(2)
        size = amount * {None: 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}[unit]
    else:
        raise ValueError("log size must be a string or integer")
    if not MIN_LOG_MAX_BYTES <= size <= MAX_LOG_MAX_BYTES:
        raise ValueError(f"log size must be between {MIN_LOG_MAX_BYTES} and {MAX_LOG_MAX_BYTES} bytes")
    return size


def format_log_size(value: int) -> str:
    """Render an exact binary size with its byte count."""
    if type(value) is not int or value < 0:
        raise ValueError("log size must be a nonnegative integer")
    for unit, factor in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if value and value % factor == 0:
            return f"{value // factor} {unit} ({value} bytes)"
    return f"{value} bytes"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _safe_token(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    bounded = value[:96]
    return bounded if _SAFE_TOKEN_RE.fullmatch(bounded) else None


def _safe_timestamp(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    bounded = value[:40]
    return bounded if _TIMESTAMP_RE.fullmatch(bounded) else None


def _safe_int(value: object, *, minimum: int = 0, maximum: int = (1 << 63) - 1) -> int | None:
    if type(value) is not int or value < minimum or value > maximum:
        return None
    return value


def bounded_exception_evidence(exc: BaseException) -> dict[str, Any]:
    """Return the exception type and final traceback location, without its message."""
    result: dict[str, Any] = {"exception_type": type(exc).__name__[:_MAX_TEXT_LENGTH]}
    try:
        frames = traceback.extract_tb(exc.__traceback__) if exc.__traceback__ else []
    except (ValueError, OSError):
        frames = []
    if frames:
        frame = frames[-1]
        result.update(
            filename=Path(frame.filename).name[:_MAX_TEXT_LENGTH],
            function=frame.name[:_MAX_TEXT_LENGTH],
            line=_safe_int(frame.lineno),
        )
    return result


@dataclass(slots=True)
class PreparedAgentDiagnostics:
    instance_id: str
    startup_sequence: int
    log_path: Path
    handle: BinaryIO | None
    capture_mode: str
    capture_health: str
    error: str | None
    reconciliation_degraded: bool

    def close(self) -> None:
        if self.handle is not None:
            try:
                self.handle.close()
            except OSError:
                pass
            self.handle = None


def _valid_component(value: object, label: str) -> str:
    if not isinstance(value, str) or not _COMPONENT_RE.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"{label} is not a safe path component")
    return value


def _runtime_identity(runtime: Any) -> tuple[Path, str, str]:
    root = Path(runtime.root).expanduser().resolve()
    runtime_id = runtime.instance_id
    if not isinstance(runtime_id, str) or not runtime_id:
        raise ValueError("runtime.instance_id must be a non-empty string")
    runtime_key = sha256(f"{root}\0{runtime_id}".encode("utf-8")).hexdigest()
    return root, runtime_id, runtime_key


def _mode(value: int) -> int:
    return stat.S_IMODE(value)


def _check_private_directory_fd(fd: int) -> None:
    info = os.fstat(fd)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or _mode(info.st_mode) != 0o700:
        raise PermissionError("private log directory ownership or mode is invalid")


def _open_tmp_root() -> int:
    root = Path(_TMP_ROOT)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(root, flags)
    if not stat.S_ISDIR(os.fstat(fd).st_mode):
        os.close(fd)
        raise NotADirectoryError("configured temporary root is not a directory")
    os.set_inheritable(fd, False)
    return fd


def _open_private_child(parent_fd: int, name: str, *, create: bool) -> int:
    if create:
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    child_fd = os.open(name, flags, dir_fd=parent_fd)
    try:
        info = os.fstat(child_fd)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
            raise PermissionError("private log directory ownership is invalid")
        os.fchmod(child_fd, 0o700)
        _check_private_directory_fd(child_fd)
        os.set_inheritable(child_fd, False)
        return child_fd
    except BaseException:
        os.close(child_fd)
        raise


def _log_directory_fd(log_path: Path, *, create: bool) -> int:
    root = Path(_TMP_ROOT)
    try:
        relative = log_path.parent.relative_to(root)
    except ValueError as exc:
        raise ValueError("agent log path is outside the configured /tmp root") from exc
    parts = relative.parts
    if len(parts) != 3 or parts[0] != f"qqtools-qexp-{os.getuid()}" or not _DIGEST_RE.fullmatch(parts[1]):
        raise ValueError("agent log path has an invalid private hierarchy")
    _valid_component(parts[2], "instance_id")
    current = _open_tmp_root()
    try:
        for part in parts:
            child = _open_private_child(current, part, create=create)
            os.close(current)
            current = child
        return current
    except BaseException:
        os.close(current)
        raise


def _open_log_file(log_path: Path) -> BinaryIO:
    directory_fd = _log_directory_fd(log_path, create=True)
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(log_path.name, flags, 0o600, dir_fd=directory_fd)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            os.close(fd)
            raise PermissionError("agent log must be a regular file owned by this user")
        os.fchmod(fd, 0o600)
        os.set_inheritable(fd, False)
        path_info = os.stat(log_path.name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(path_info.st_mode) or (path_info.st_dev, path_info.st_ino) != (info.st_dev, info.st_ino):
            os.close(fd)
            raise PermissionError("agent log path changed during open")
        return os.fdopen(fd, "ab", buffering=0)
    finally:
        os.close(directory_fd)


def _expected_log_path(runtime_key: str, instance_id: str) -> Path:
    return Path(_TMP_ROOT) / f"qqtools-qexp-{os.getuid()}" / runtime_key / instance_id / "agent.log"


def _ensure_runtime_diagnostic_dirs(runtime: Any) -> dict[str, Path]:
    paths = _runtime_paths(runtime)
    for key in ("diagnostics", "diagnostic_instances"):
        path = paths[key]
        path.mkdir(parents=True, exist_ok=True)
        info = os.lstat(path)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
            raise PermissionError("machine diagnostic directory ownership is invalid")
    return paths


def _runtime_paths(runtime: Any) -> dict[str, Path]:
    return runtime.paths if hasattr(runtime, "paths") else machine_runtime_paths(Path(runtime.root))


def _runtime_diagnostic_dirs_valid(paths: dict[str, Path]) -> bool:
    try:
        for key in ("diagnostics", "diagnostic_instances"):
            info = os.lstat(paths[key])
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
                return False
        return True
    except OSError:
        return False


def _default_metadata(runtime_id: str, next_sequence: int = 1) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "runtime_id": runtime_id,
        "next_startup_sequence": max(1, next_sequence),
        "coverage": {
            "evicted_count": 0,
            "first_evicted_sequence": None,
            "last_evicted_sequence": None,
            "terminal_evicted_through": 0,
            "unresolved_evicted_through": 0,
        },
    }


def _read_json_file(path: Path, *, max_bytes: int = _MAX_DIAGNOSTIC_BYTES) -> dict[str, Any] | None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except (FileNotFoundError, OSError):
        return None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_size > max_bytes:
            return None
        with os.fdopen(fd, "rb", closefd=False) as handle:
            encoded = handle.read(max_bytes + 1)
        if len(encoded) > max_bytes:
            return None
        value = json.loads(encoded.decode("utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeError, ValueError, TypeError):
        return None
    finally:
        os.close(fd)


def _write_json(path: Path, value: dict[str, Any]) -> bool:
    try:
        result = atomic_replace(path, value)
        return result is not None
    except (OSError, TypeError, ValueError):
        return False


def _instance_record_path(paths: dict[str, Path], instance_id: str) -> Path:
    return paths["diagnostic_instances"] / f"{_valid_component(instance_id, 'instance_id')}.json"


def _read_record(runtime: Any, instance_id: str) -> dict[str, Any] | None:
    try:
        paths = _runtime_paths(runtime)
        if not _runtime_diagnostic_dirs_valid(paths):
            return None
        return _read_json_file(_instance_record_path(paths, instance_id))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def read_diagnostic_record(runtime: Any, instance_id: str) -> dict[str, Any] | None:
    """Read one retained, identity-matching agent diagnostic record."""
    try:
        _, runtime_id, _ = _runtime_identity(runtime)
        record = _read_record(runtime, instance_id)
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    if (
        record is None
        or record.get("schema_version") != 1
        or record.get("runtime_id") != runtime_id
        or record.get("instance_id") != instance_id
    ):
        return None
    return record


@contextmanager
def _diagnostic_lock(runtime: Any) -> Iterator[bool]:
    """Acquire the independent diagnostic lock for at most one second."""
    fd: int | None = None
    acquired = False
    try:
        paths = _ensure_runtime_diagnostic_dirs(runtime)
        lock_path = paths["diagnostic_lock"]
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        parent_info = os.lstat(lock_path.parent)
        if not stat.S_ISDIR(parent_info.st_mode) or parent_info.st_uid != os.getuid():
            raise PermissionError("diagnostic lock directory ownership is invalid")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(lock_path, flags, 0o600)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise PermissionError("diagnostic lock ownership is invalid")
        os.fchmod(fd, 0o600)
        os.set_inheritable(fd, False)
        deadline = time.monotonic() + 1.0
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN} or time.monotonic() >= deadline:
                    break
                time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
    except (OSError, ValueError, KeyError, TypeError):
        acquired = False
    try:
        yield acquired
    finally:
        if fd is not None:
            if acquired:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
            os.close(fd)


def _record_files(paths: dict[str, Path]) -> list[Path]:
    try:
        if not _runtime_diagnostic_dirs_valid(paths):
            return []
        result = []
        with os.scandir(paths["diagnostic_instances"]) as entries:
            for entry in entries:
                if entry.name.endswith(".json") and _COMPONENT_RE.fullmatch(entry.name[:-5]):
                    try:
                        if entry.is_file(follow_symlinks=False):
                            result.append(Path(entry.path))
                    except OSError:
                        continue
        return sorted(result, key=lambda item: item.name)
    except OSError:
        return []


def _scan_records(runtime: Any, runtime_id: str) -> list[dict[str, Any]]:
    try:
        paths = _runtime_paths(runtime)
    except (OSError, KeyError, TypeError, AttributeError):
        return []
    records: list[dict[str, Any]] = []
    for path in _record_files(paths):
        record = _read_json_file(path)
        if (
            record is not None
            and record.get("schema_version") == 1
            and record.get("runtime_id") == runtime_id
            and isinstance(record.get("instance_id"), str)
            and type(record.get("startup_sequence")) is int
        ):
            records.append(record)
    return records


def _writers(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    writers = record.get("writers")
    if not isinstance(writers, dict):
        writers = {}
        record["writers"] = writers
    for name in ("agent", "launcher", "observer"):
        value = writers.get(name)
        if not isinstance(value, dict):
            value = {"revision": 0}
            writers[name] = value
        if type(value.get("revision")) is not int or value["revision"] < 0:
            value["revision"] = 0
    return writers


def _sanitize_exception(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    exception_type = value.get("exception_type")
    if not isinstance(exception_type, str):
        return None
    result: dict[str, Any] = {"exception_type": exception_type[:_MAX_TEXT_LENGTH]}
    filename = value.get("filename")
    function = value.get("function")
    line = _safe_int(value.get("line"), minimum=1)
    if isinstance(filename, str):
        result["filename"] = Path(filename).name[:_MAX_TEXT_LENGTH]
    if isinstance(function, str):
        result["function"] = function[:_MAX_TEXT_LENGTH]
    if line is not None:
        result["line"] = line
    return result


def _sanitize_cleanup_steps(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, Any] = {}
    for key in sorted(value, key=lambda item: str(item))[:32]:
        step = _safe_token(key)
        if step is None:
            continue
        outcome = value[key]
        if isinstance(outcome, str):
            token = _safe_token(outcome)
            if token is not None:
                result[step] = token
        elif isinstance(outcome, dict):
            cleaned: dict[str, Any] = {}
            for field in ("outcome", "error_type"):
                token = _safe_token(outcome.get(field))
                if token is not None:
                    cleaned[field] = token
            if cleaned:
                result[step] = cleaned
    return result


def _sanitize_signal_attempts(value: object) -> list[dict[str, Any]] | None:
    if not isinstance(value, list):
        return None
    result: list[dict[str, Any]] = []
    for attempt in value[-8:]:
        if not isinstance(attempt, dict):
            continue
        signal_number = _safe_int(attempt.get("signal"), minimum=1, maximum=255)
        attempted_at = _safe_timestamp(attempt.get("attempted_at"))
        if signal_number is None:
            continue
        item: dict[str, Any] = {"signal": signal_number}
        if attempted_at is not None:
            item["attempted_at"] = attempted_at
        if type(attempt.get("delivered")) is bool:
            item["delivered"] = attempt["delivered"]
        error_type = _safe_token(attempt.get("error_type"))
        if error_type is not None:
            item["error_type"] = error_type
        result.append(item)
    return result


def _sanitize_evidence(writer: str, evidence: dict[str, Any]) -> dict[str, Any]:
    allowed = {"agent": _AGENT_FIELDS, "launcher": _LAUNCHER_FIELDS, "observer": _OBSERVER_FIELDS}[writer]
    result: dict[str, Any] = {}
    for key in allowed:
        if key not in evidence:
            continue
        value = evidence[key]
        if writer == "agent":
            if key == "phase" and value in _PHASE_RANK:
                result[key] = value
            elif key == "admitted" and type(value) is bool:
                result[key] = value
            elif key == "pid":
                if value is None:
                    result[key] = None
                else:
                    number = _safe_int(value, minimum=1)
                    if number is not None:
                        result[key] = number
            elif key == "pid_start_time_ticks":
                if value is None:
                    result[key] = None
                else:
                    number = _safe_int(value)
                    if number is not None:
                        result[key] = number
            elif key == "capture_health" and value in {"healthy", "degraded", "unavailable"}:
                result[key] = value
            elif key == "capture_error":
                if value is None:
                    result[key] = None
                else:
                    token = _safe_token(value)
                    if token is not None:
                        result[key] = token
            elif key == "stop_reason":
                token = _safe_token(value)
                if token is not None:
                    result[key] = token
            elif key == "handled_signal":
                number = _safe_int(value, minimum=1, maximum=255)
                if number is not None:
                    result[key] = number
            elif key == "primary_exception":
                cleaned = _sanitize_exception(value)
                if cleaned is not None:
                    result[key] = cleaned
            elif key == "cleanup_outcome" and value in {"pending", "succeeded", "failed", "unknown"}:
                result[key] = value
            elif key == "cleanup_steps":
                cleaned_steps = _sanitize_cleanup_steps(value)
                if cleaned_steps is not None:
                    result[key] = cleaned_steps
            elif key == "finalized_at":
                timestamp = _safe_timestamp(value)
                if timestamp is not None:
                    result[key] = timestamp
        elif writer == "launcher":
            if key == "startup_outcome" and value in {"spawn_failed", "started", "timed_out", "exited"}:
                result[key] = value
            elif key in {"observed_at", "timeout_triggered_at"}:
                timestamp = _safe_timestamp(value)
                if timestamp is not None:
                    result[key] = timestamp
            elif key == "timeout_trigger":
                token = _safe_token(value)
                if token is not None:
                    result[key] = token
            elif key == "signal_attempts":
                attempts = _sanitize_signal_attempts(value)
                if attempts is not None:
                    result[key] = attempts
            elif key == "wait_status":
                if value is None:
                    result[key] = None
                else:
                    number = _safe_int(value, minimum=-(1 << 31), maximum=(1 << 31) - 1)
                    if number is not None:
                        result[key] = number
        else:
            if key == "liveness" and value in {"present", "absent", "unknown", "liveness_unknown"}:
                result[key] = value
            elif key == "observed_at":
                timestamp = _safe_timestamp(value)
                if timestamp is not None:
                    result[key] = timestamp
            elif key == "reason":
                token = _safe_token(value)
                if token is not None:
                    result[key] = token
            elif key == "coverage" and isinstance(value, dict):
                cleaned_coverage: dict[str, Any] = {}
                for field in ("evicted_count", "first_evicted_sequence", "last_evicted_sequence"):
                    number = _safe_int(value.get(field))
                    if number is not None:
                        cleaned_coverage[field] = number
                if cleaned_coverage:
                    result[key] = cleaned_coverage
            elif key == "abnormal_exit_unknown" and isinstance(value, dict):
                observed_at = _safe_timestamp(value.get("observed_at"))
                result[key] = {"observed_at": observed_at} if observed_at is not None else {}
    return result


def _merge_cleanup_steps(old: object, new: object) -> dict[str, Any]:
    merged = dict(old) if isinstance(old, dict) else {}
    if not isinstance(new, dict):
        return merged
    for step, outcome in new.items():
        previous = merged.get(step)
        if isinstance(previous, dict) and isinstance(outcome, dict):
            prior_outcome = previous.get("outcome")
            next_outcome = outcome.get("outcome")
            if prior_outcome in {"succeeded", "failed", "unknown"} and next_outcome == "pending":
                continue
            merged[step] = {**previous, **outcome}
        elif isinstance(previous, str) and previous in {"succeeded", "failed", "unknown"} and outcome == "pending":
            continue
        else:
            merged[step] = outcome
    return dict(sorted(merged.items())[:32])


def _merge_writer_evidence(writer: str, current: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(current)
    for key, value in incoming.items():
        if writer == "agent":
            if key == "phase":
                old_phase = merged.get(key)
                if old_phase not in _PHASE_RANK or _PHASE_RANK[value] >= _PHASE_RANK[old_phase]:
                    merged[key] = value
            elif key == "admitted":
                merged[key] = bool(merged.get(key)) or value
            elif key in {"pid", "pid_start_time_ticks"}:
                previous = merged.get(key)
                if value is not None and (previous is None or previous == value):
                    merged[key] = value
            elif key == "stop_reason":
                merged.setdefault(key, value)
            elif key in {"handled_signal", "primary_exception", "finalized_at"}:
                if key not in merged:
                    merged[key] = value
            elif key == "cleanup_outcome":
                previous = merged.get(key)
                if previous not in {"succeeded", "failed", "unknown"} or value != "pending":
                    merged[key] = value
            elif key == "cleanup_steps":
                merged[key] = _merge_cleanup_steps(merged.get(key), value)
            else:
                merged[key] = value
        elif key == "signal_attempts":
            attempts = merged.get(key, [])
            combined = [*attempts, *value] if isinstance(attempts, list) else list(value)
            unique = {json.dumps(item, sort_keys=True): item for item in combined}
            merged[key] = sorted(unique.values(), key=lambda item: (item.get("attempted_at", ""), item["signal"]))[-8:]
        elif key == "wait_status" and merged.get(key) is not None and value is None:
            continue
        elif key == "abnormal_exit_unknown" and merged.get(key):
            continue
        else:
            merged[key] = value
    return merged


def _project_agent_fields(record: dict[str, Any], agent: dict[str, Any]) -> None:
    phase = agent.get("phase")
    if phase in _PHASE_RANK:
        old_phase = record.get("phase")
        if old_phase not in _PHASE_RANK or _PHASE_RANK[phase] >= _PHASE_RANK[old_phase]:
            record["phase"] = phase
    record["admitted"] = bool(record.get("admitted")) or bool(agent.get("admitted"))
    for key in ("pid", "pid_start_time_ticks"):
        if record.get(key) is None and agent.get(key) is not None:
            record[key] = agent[key]
    if agent.get("capture_health") in {"healthy", "degraded", "unavailable"}:
        record["capture_health"] = agent["capture_health"]
    if agent.get("capture_error") is not None:
        record["capture_error"] = agent["capture_error"]
    elif "capture_error" in agent or agent.get("capture_health") == "healthy":
        record.pop("capture_error", None)


def _is_terminal(record: dict[str, Any]) -> bool:
    writers = record.get("writers", {})
    agent = writers.get("agent", {}) if isinstance(writers, dict) else {}
    launcher = writers.get("launcher", {}) if isinstance(writers, dict) else {}
    observer = writers.get("observer", {}) if isinstance(writers, dict) else {}
    return (
        (isinstance(agent, dict) and (agent.get("phase") == "stopped" or bool(agent.get("finalized_at"))))
        or (isinstance(launcher, dict) and type(launcher.get("wait_status")) is int)
        or (isinstance(observer, dict) and bool(observer.get("abnormal_exit_unknown")))
    )


def summarize_diagnostic_record(record: dict[str, Any]) -> dict[str, Any]:
    """Project a bounded primary exit summary from one versioned record."""
    writers = record.get("writers", {}) if isinstance(record, dict) else {}
    agent = writers.get("agent", {}) if isinstance(writers, dict) and isinstance(writers.get("agent"), dict) else {}
    launcher = (
        writers.get("launcher", {}) if isinstance(writers, dict) and isinstance(writers.get("launcher"), dict) else {}
    )
    observer = (
        writers.get("observer", {}) if isinstance(writers, dict) and isinstance(writers.get("observer"), dict) else {}
    )
    reason = agent.get("stop_reason")
    source = "agent" if reason else "unknown"
    if not reason and launcher.get("startup_outcome") in {"spawn_failed", "exited"}:
        reason, source = "startup_failed", "launcher"
    if not reason and observer.get("abnormal_exit_unknown"):
        reason, source = "abnormal_exit_unknown", "observer"
    summary: dict[str, Any] = {
        "instance_id": record.get("instance_id"),
        "startup_sequence": record.get("startup_sequence"),
        "reason": reason or "unknown",
        "source": source,
        "startup_outcome": launcher.get("startup_outcome"),
        "timeout_trigger": launcher.get("timeout_trigger"),
        "signal_attempts": launcher.get("signal_attempts", []),
        "wait_status": launcher.get("wait_status"),
        "termination_observation": (
            "abnormal_exit_unknown" if observer.get("abnormal_exit_unknown") else observer.get("liveness")
        ),
        "observed_at": observer.get("observed_at"),
        "cleanup_outcome": agent.get("cleanup_outcome"),
        "cleanup_steps": agent.get("cleanup_steps", {}),
        "handled_signal": agent.get("handled_signal"),
    }
    primary_exception = agent.get("primary_exception")
    if isinstance(primary_exception, dict):
        summary["primary_exception"] = {
            key: primary_exception[key]
            for key in ("exception_type", "filename", "function", "line")
            if key in primary_exception
        }
    return summary


def _metadata(runtime: Any, runtime_id: str, paths: dict[str, Path]) -> dict[str, Any]:
    value = _read_json_file(paths["diagnostic_metadata"])
    if (
        value is None
        or value.get("schema_version") != 1
        or value.get("runtime_id") != runtime_id
        or type(value.get("next_startup_sequence")) is not int
    ):
        max_sequence = max(
            (record["startup_sequence"] for record in _scan_records(runtime, runtime_id)),
            default=0,
        )
        return _default_metadata(runtime_id, max_sequence + 1)
    coverage = value.get("coverage")
    if not isinstance(coverage, dict):
        value["coverage"] = _default_metadata(runtime_id)["coverage"]
    else:
        defaults = _default_metadata(runtime_id)["coverage"]
        for key, default in defaults.items():
            if key not in coverage or (key == "evicted_count" and type(coverage.get(key)) is not int):
                coverage[key] = default
    return value


def _record_from_path(path: Path, runtime_id: str) -> dict[str, Any] | None:
    record = _read_json_file(path)
    if record is None or record.get("runtime_id") != runtime_id or record.get("schema_version") != 1:
        return None
    return record


def _refresh_last_exit_locked(runtime: Any, runtime_id: str, paths: dict[str, Path]) -> bool:
    records = [
        record
        for record in _scan_records(runtime, runtime_id)
        if record.get("admitted") is True and _is_terminal(record)
    ]
    if not records:
        return True
    latest = max(records, key=lambda item: item["startup_sequence"])
    summary = summarize_diagnostic_record(latest)
    old = _read_json_file(paths["diagnostic_last_exit"])
    old_sequence = old.get("startup_sequence", -1) if old and old.get("runtime_id") == runtime_id else -1
    sequence = latest["startup_sequence"]
    if type(old_sequence) is int and old_sequence > sequence:
        return True
    value = {
        "schema_version": 1,
        "runtime_id": runtime_id,
        "instance_id": latest["instance_id"],
        "startup_sequence": sequence,
        "summary": summary,
        "refreshed_at": _utc_now(),
    }
    return _write_json(paths["diagnostic_last_exit"], value)


def _reconcile_record(runtime: Any, record: dict[str, Any], runtime_id: str) -> tuple[str, bool]:
    if record.get("runtime_id") != runtime_id or _is_terminal(record):
        return ("terminal" if _is_terminal(record) else "unknown"), True
    writers = record.get("writers", {})
    agent = writers.get("agent", {}) if isinstance(writers, dict) else {}
    pid = record.get("pid") if record.get("pid") is not None else agent.get("pid")
    ticks = record.get("pid_start_time_ticks")
    if ticks is None:
        ticks = agent.get("pid_start_time_ticks")
    if record.get("host_id") != host_instance_id():
        state, reason = "unknown", "host_identity_mismatch"
    elif type(pid) is not int or pid <= 0 or type(ticks) is not int or ticks < 0:
        state, reason = "unknown", "identity_missing"
    else:
        observation = read_process_identity(pid)
        if observation.state == "absent":
            state, reason = "absent", "verified_absence"
        elif observation.state == "present" and observation.start_time_ticks != ticks:
            state, reason = "absent", "pid_reused"
        elif observation.state == "present":
            state, reason = "present", "identity_matches"
        else:
            state, reason = "unknown", "liveness_unknown"
    observer = writers.get("observer", {}) if isinstance(writers, dict) else {}
    if not isinstance(observer, dict):
        observer = {}
    if observer.get("liveness") != state or (state == "absent" and not observer.get("abnormal_exit_unknown")):
        evidence: dict[str, Any] = {"liveness": state, "reason": reason, "observed_at": _utc_now()}
        if state == "absent":
            evidence["abnormal_exit_unknown"] = {"observed_at": evidence["observed_at"]}
        revision = observer.get("revision", 0)
        updated = _update_evidence_internal(
            runtime,
            instance_id=record["instance_id"],
            startup_sequence=record["startup_sequence"],
            writer="observer",
            revision=(revision + 1) if type(revision) is int else 1,
            evidence=evidence,
            runtime_id=runtime_id,
        )
        return state, updated
    return state, True


def _prune_locked(runtime: Any, runtime_id: str, paths: dict[str, Path], current_instance_id: str | None) -> int:
    records = _scan_records(runtime, runtime_id)
    terminal = sorted(
        (record for record in records if _is_terminal(record) and record.get("instance_id") != current_instance_id),
        key=lambda item: item["startup_sequence"],
    )
    unresolved = sorted(
        (record for record in records if not _is_terminal(record) and record.get("instance_id") != current_instance_id),
        key=lambda item: item["startup_sequence"],
    )
    candidates = [(record, "terminal") for record in terminal[:-32]] + [
        (record, "unresolved") for record in unresolved[:-32]
    ]
    if not candidates:
        return 0
    meta = _metadata(runtime, runtime_id, paths)
    coverage = meta["coverage"]
    counted: list[tuple[dict[str, Any], str]] = []
    for record, category in sorted(candidates, key=lambda item: item[0]["startup_sequence"]):
        sequence = record["startup_sequence"]
        watermark_key = f"{category}_evicted_through"
        watermark = coverage.get(watermark_key, 0)
        if type(watermark) is not int:
            watermark = 0
        if sequence > watermark:
            counted.append((record, category))
            coverage[watermark_key] = sequence
    if counted:
        coverage["evicted_count"] = min(_MAX_EVICTION_COUNT, max(0, coverage.get("evicted_count", 0)) + len(counted))
        sequences = [record["startup_sequence"] for record, _ in counted]
        prior_min = coverage.get("first_evicted_sequence")
        prior_max = coverage.get("last_evicted_sequence")
        coverage["first_evicted_sequence"] = min(sequences + ([prior_min] if type(prior_min) is int else []))
        coverage["last_evicted_sequence"] = max(sequences + ([prior_max] if type(prior_max) is int else []))
    if not _write_json(paths["diagnostic_metadata"], meta):
        return 0
    removed = 0
    for record, _category in candidates:
        try:
            path = _instance_record_path(paths, record["instance_id"])
            info = os.lstat(path)
            if stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid():
                path.unlink()
                removed += 1
        except (OSError, ValueError):
            continue
    return removed


def reconcile_agent_diagnostics(runtime: Any, *, current_instance_id: str | None = None) -> dict[str, Any]:
    """Reconcile retained process identities, refresh last exit, then prune history."""
    try:
        _root, runtime_id, _key = _runtime_identity(runtime)
        paths = _ensure_runtime_diagnostic_dirs(runtime)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {"available": False, "reconciled": False, "coverage": {"evicted_count": 0}}
    try:
        current_instance_id = _valid_component(current_instance_id, "instance_id") if current_instance_id else None
    except ValueError:
        current_instance_id = None
    unknown = 0
    unresolved = 0
    inspected = 0
    update_failures = 0
    records = sorted(_scan_records(runtime, runtime_id), key=lambda item: item["startup_sequence"])
    for record in records:
        if record.get("instance_id") == current_instance_id:
            continue
        state, updated = _reconcile_record(runtime, record, runtime_id)
        inspected += 1
        if not updated:
            update_failures += 1
        if state == "unknown":
            unknown += 1
        if state in {"unknown", "present"}:
            unresolved += 1
    with _diagnostic_lock(runtime) as acquired:
        if not acquired:
            return {"available": False, "reconciled": False, "coverage": {"evicted_count": 0}}
        refreshed = update_failures == 0 and _refresh_last_exit_locked(runtime, runtime_id, paths)
        pruned = _prune_locked(runtime, runtime_id, paths, current_instance_id) if refreshed else 0
        meta = _metadata(runtime, runtime_id, paths)
        return {
            "available": True,
            "reconciled": refreshed,
            "inspected": inspected,
            "liveness_unknown": unknown,
            "unresolved": unresolved,
            "update_failures": update_failures,
            "pruned": pruned,
            "coverage": dict(meta.get("coverage", {})),
        }


def _update_evidence_internal(
    runtime: Any,
    *,
    instance_id: str,
    startup_sequence: int,
    writer: str,
    revision: int,
    evidence: dict[str, Any],
    runtime_id: str | None = None,
) -> bool:
    if writer not in {"agent", "launcher", "observer"} or type(revision) is not int or revision < 0:
        return False
    try:
        _root, actual_runtime_id, _key = _runtime_identity(runtime)
        if runtime_id is not None and actual_runtime_id != runtime_id:
            return False
        paths = _ensure_runtime_diagnostic_dirs(runtime)
        record_path = _instance_record_path(paths, instance_id)
        sanitized = _sanitize_evidence(writer, evidence)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False
    with _diagnostic_lock(runtime) as acquired:
        if not acquired:
            return False
        record = _record_from_path(record_path, actual_runtime_id)
        if (
            record is None
            or record.get("instance_id") != instance_id
            or type(record.get("startup_sequence")) is not int
            or record.get("startup_sequence") != startup_sequence
        ):
            return False
        writers = _writers(record)
        owned = writers[writer]
        current_revision = owned["revision"]
        if revision < current_revision:
            return False
        if revision == current_revision:
            return True
        next_owned = _merge_writer_evidence(writer, owned, sanitized)
        next_owned["revision"] = revision
        writers[writer] = next_owned
        if writer == "agent":
            _project_agent_fields(record, next_owned)
        if not _write_json(record_path, record):
            return False
        if record.get("admitted") is True and _is_terminal(record):
            _refresh_last_exit_locked(runtime, actual_runtime_id, paths)
        return True


def update_diagnostic_evidence(
    runtime: Any,
    *,
    instance_id: str,
    startup_sequence: int,
    writer: str,
    revision: int,
    evidence: dict[str, Any],
) -> bool:
    """Merge one writer's bounded evidence into an existing retained record."""
    if not isinstance(evidence, dict):
        return False
    return _update_evidence_internal(
        runtime,
        instance_id=instance_id,
        startup_sequence=startup_sequence,
        writer=writer,
        revision=revision,
        evidence=evidence,
    )


def _record_initial(
    *,
    runtime_id: str,
    instance_id: str,
    startup_sequence: int,
    log_path: Path,
    capture_mode: str,
    capture_health: str,
    capture_error: str | None,
    log_max_bytes: int,
) -> dict[str, Any]:
    started_at = _utc_now()
    agent: dict[str, Any] = {
        "revision": 0,
        "phase": "starting",
        "admitted": False,
        "capture_health": capture_health,
    }
    if capture_error is not None:
        agent["capture_error"] = capture_error
    return {
        "schema_version": 1,
        "runtime_id": runtime_id,
        "host_id": host_instance_id(),
        "instance_id": instance_id,
        "startup_sequence": startup_sequence,
        "package_version": __version__,
        "started_at": started_at,
        "phase": "starting",
        "admitted": False,
        "pid": None,
        "pid_start_time_ticks": None,
        "log_path": str(log_path),
        "capture_mode": capture_mode,
        "capture_health": capture_health,
        "effective_log_max_bytes": log_max_bytes,
        "writers": {
            "agent": agent,
            "launcher": {"revision": 0},
            "observer": {"revision": 0},
        },
        **({"capture_error": capture_error} if capture_error is not None else {}),
    }


def prepare_agent_diagnostics(
    runtime: Any,
    *,
    instance_id: str,
    capture_mode: str,
    log_max_bytes: int,
) -> PreparedAgentDiagnostics:
    """Reconcile old evidence and prepare a private per-instance agent log."""
    instance_id = _valid_component(instance_id, "instance_id")
    if capture_mode not in {"detached", "foreground_managed_only"}:
        raise ValueError("capture_mode must be 'detached' or 'foreground_managed_only'")
    log_max_bytes = parse_log_size(log_max_bytes)
    runtime_available = True
    try:
        _root, runtime_id, runtime_key = _runtime_identity(runtime)
    except Exception:
        runtime_available = False
        runtime_id, runtime_key = "unavailable", sha256(b"unavailable").hexdigest()
    log_path = _expected_log_path(runtime_key, instance_id)
    reconciliation_ok = True
    try:
        if not runtime_available:
            raise RuntimeError("runtime identity is unavailable")
        result = reconcile_agent_diagnostics(runtime)
        reconciliation_ok = bool(result.get("reconciled", result.get("available", False)))
    except Exception:
        reconciliation_ok = False
    handle: BinaryIO | None = None
    capture_health = "healthy"
    capture_error: str | None = None
    try:
        handle = _open_log_file(log_path)
    except Exception as exc:
        capture_health = "degraded"
        capture_error = f"log_open_{type(exc).__name__}"[:96]
    if not reconciliation_ok:
        capture_health = "degraded"
        capture_error = "reconciliation_unavailable"
    startup_sequence = 0
    try:
        if not runtime_available:
            raise RuntimeError("runtime identity is unavailable")
        paths = _ensure_runtime_diagnostic_dirs(runtime)
        with _diagnostic_lock(runtime) as acquired:
            if acquired:
                meta = _metadata(runtime, runtime_id, paths)
                records = _scan_records(runtime, runtime_id)
                highest = max((item["startup_sequence"] for item in records), default=0)
                sequence = max(meta.get("next_startup_sequence", 1), highest + 1, 1)
                meta["next_startup_sequence"] = sequence + 1
                if _write_json(paths["diagnostic_metadata"], meta):
                    startup_sequence = sequence
                    record = _record_initial(
                        runtime_id=runtime_id,
                        instance_id=instance_id,
                        startup_sequence=sequence,
                        log_path=log_path,
                        capture_mode=capture_mode,
                        capture_health=capture_health,
                        capture_error=capture_error,
                        log_max_bytes=log_max_bytes,
                    )
                    record_path = _instance_record_path(paths, instance_id)
                    if record_path.exists():
                        startup_sequence = 0
                        capture_health = "degraded"
                        capture_error = "instance_record_already_exists"
                        if handle is not None:
                            handle.close()
                            handle = None
                    elif not _write_json(record_path, record):
                        startup_sequence = 0
                        capture_health = "degraded"
                        capture_error = "record_write_failed"
                else:
                    capture_health = "degraded"
                    capture_error = "metadata_write_failed"
    except Exception:
        capture_health = "degraded"
        capture_error = capture_error or "diagnostics_store_unavailable"
    return PreparedAgentDiagnostics(
        instance_id=instance_id,
        startup_sequence=startup_sequence,
        log_path=log_path,
        handle=handle,
        capture_mode=capture_mode,
        capture_health=capture_health,
        error=capture_error,
        reconciliation_degraded=not reconciliation_ok,
    )


def _private_log_available(log_path: object, runtime_key: str, instance_id: str) -> bool:
    expected = _expected_log_path(runtime_key, instance_id)
    try:
        if Path(log_path) != expected:
            return False
        directory_fd = _log_directory_fd(expected, create=False)
        try:
            info = os.stat(expected.name, dir_fd=directory_fd, follow_symlinks=False)
        finally:
            os.close(directory_fd)
        return stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and _mode(info.st_mode) == 0o600
    except (OSError, TypeError, ValueError):
        return False


def _valid_threshold(value: object) -> int | None:
    try:
        return parse_log_size(value) if type(value) is int else None
    except ValueError:
        return None


def _read_last_exit(paths: dict[str, Path], runtime_id: str) -> dict[str, Any] | None:
    value = _read_json_file(paths["diagnostic_last_exit"])
    if value is None or value.get("schema_version") != 1 or value.get("runtime_id") != runtime_id:
        return None
    if not isinstance(value.get("summary"), dict):
        return None
    return value


def diagnostics_status(
    runtime: Any,
    *,
    configured_log_max_bytes: int,
    active_identity: tuple[int, str, int] | None,
) -> dict[str, Any]:
    """Return bounded diagnostic facts without opening or scanning agent logs."""
    configured = _valid_threshold(configured_log_max_bytes)
    if configured is None:
        raise ValueError("configured_log_max_bytes is outside the supported range")
    try:
        _root, runtime_id, runtime_key = _runtime_identity(runtime)
        paths = _runtime_paths(runtime)
        if not _runtime_diagnostic_dirs_valid(paths):
            raise PermissionError("diagnostic directory ownership is invalid")
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {
            "available": False,
            "state": "unavailable",
            "reason": "diagnostics_unavailable",
            "configured_log_max_bytes": configured,
            "configured_log_max_size": format_log_size(configured),
            "effective_log_max_bytes": None,
            "effective_log_max_size": None,
            "log_available": False,
            "last_exit": None,
            "last_exit_source": "unavailable",
            "coverage": {"evicted_count": 0},
        }
    records = _scan_records(runtime, runtime_id)
    chosen: dict[str, Any] | None = None
    if active_identity is not None and len(active_identity) == 3:
        _pid, active_instance_id, _ticks = active_identity
        chosen = next((item for item in records if item.get("instance_id") == active_instance_id), None)
    if chosen is None and records:
        chosen = max(records, key=lambda item: item["startup_sequence"])
    meta = _metadata(runtime, runtime_id, paths)
    coverage = dict(meta.get("coverage", {})) if isinstance(meta.get("coverage"), dict) else {"evicted_count": 0}
    last_exit_record = _read_last_exit(paths, runtime_id)
    last_exit_summary = last_exit_record.get("summary") if last_exit_record else None
    last_exit_seq = last_exit_record.get("startup_sequence") if last_exit_record else None
    newest_sequence = max((item["startup_sequence"] for item in records), default=0)
    last_exit_stale = type(last_exit_seq) is int and newest_sequence > last_exit_seq
    summary: dict[str, Any] | None = summarize_diagnostic_record(chosen) if chosen else None
    source = "record" if chosen else ("last_exit" if last_exit_summary else "unavailable")
    if summary is None and isinstance(last_exit_summary, dict):
        summary = dict(last_exit_summary)
    synthetic = False
    if (
        chosen is not None
        and chosen.get("host_id") == host_instance_id()
        and chosen.get("phase") in {"active", "stopping"}
    ):
        agent = chosen.get("writers", {}).get("agent", {})
        pid = chosen.get("pid") if chosen.get("pid") is not None else agent.get("pid")
        ticks = chosen.get("pid_start_time_ticks")
        if ticks is None:
            ticks = agent.get("pid_start_time_ticks")
        if type(pid) is int and pid > 0 and type(ticks) is int and ticks >= 0:
            observation = read_process_identity(pid)
            absent = observation.state == "absent" or (
                observation.state == "present" and observation.start_time_ticks != ticks
            )
            if absent:
                projected = dict(chosen)
                writers = {
                    name: dict(value) for name, value in chosen.get("writers", {}).items() if isinstance(value, dict)
                }
                observer = writers.setdefault("observer", {"revision": 0})
                observer["liveness"] = "absent"
                observer["reason"] = "verified_absence"
                observer["observed_at"] = _utc_now()
                observer["abnormal_exit_unknown"] = {"observed_at": observer["observed_at"]}
                projected["writers"] = writers
                summary = summarize_diagnostic_record(projected)
                source = "observer"
                synthetic = True
    current_sequence = chosen.get("startup_sequence") if chosen else newest_sequence or None
    is_historical = bool(chosen and active_identity is not None and chosen.get("instance_id") != active_identity[1])
    log_path = chosen.get("log_path") if chosen else None
    log_available = bool(chosen and _private_log_available(log_path, runtime_key, chosen["instance_id"]))
    effective = _valid_threshold(chosen.get("effective_log_max_bytes")) if chosen else None
    capture_mode = chosen.get("capture_mode") if chosen else None
    capture_health = chosen.get("capture_health") if chosen else "unavailable"
    return {
        "available": chosen is not None or last_exit_summary is not None,
        "state": "available" if chosen is not None or last_exit_summary is not None else "unavailable",
        "instance_id": chosen.get("instance_id")
        if chosen
        else (last_exit_record.get("instance_id") if last_exit_record else None),
        "startup_sequence": current_sequence,
        "log_path": log_path,
        "log_available": log_available,
        "configured_log_max_bytes": configured,
        "configured_log_max_size": format_log_size(configured),
        "effective_log_max_bytes": effective,
        "effective_log_max_size": format_log_size(effective) if effective is not None else None,
        "capture_mode": capture_mode,
        "capture_health": capture_health,
        "summary": summary,
        "summary_source": source,
        "summary_synthetic": synthetic,
        "last_exit": last_exit_summary,
        "last_exit_source": "last_exit" if last_exit_summary is not None else "unavailable",
        "last_exit_historical": last_exit_stale,
        "last_exit_stale": last_exit_stale,
        "historical": is_historical,
        "coverage": coverage,
    }


class AgentLogService:
    """Best-effort managed logging and detached stdout/stderr rotation."""

    def __init__(
        self,
        log_path: Path,
        *,
        max_bytes: int,
        capture_mode: str,
        health_callback: Callable[[str, str | None], None] | None = None,
    ) -> None:
        self.log_path = Path(log_path)
        self.max_bytes = parse_log_size(max_bytes)
        if capture_mode not in {"detached", "foreground_managed_only"}:
            raise ValueError("capture_mode must be 'detached' or 'foreground_managed_only'")
        self.capture_mode = capture_mode
        self.health_callback = health_callback
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle: BinaryIO | None = None
        self._archive_sequence = 0
        self._health = "unavailable"
        self._error: str | None = "not_started"
        self._next_retry_at = 0.0
        self._retry_delay = 0.25
        self._started = False
        self._stopped = False
        self._descriptors_bound = False

    def _notify_health(self, health: str, error: str | None) -> None:
        callback = self.health_callback
        if callback is None or (health == self._health and error == self._error):
            return
        self._health, self._error = health, error
        try:
            callback(health, error)
        except Exception:
            pass

    def _open(self) -> bool:
        if self._handle is None and time.monotonic() < self._next_retry_at:
            return False
        if self._handle is not None:
            return True
        try:
            self._handle = _open_log_file(self.log_path)
            self._retry_delay = 0.25
            self._next_retry_at = 0.0
            return True
        except Exception as exc:
            self._schedule_retry(exc)
            return False

    def _schedule_retry(self, exc: BaseException) -> None:
        error = f"capture_io_{type(exc).__name__}"[:96]
        self._next_retry_at = time.monotonic() + self._retry_delay
        self._retry_delay = min(30.0, self._retry_delay * 2)
        self._notify_health("degraded", error)

    def _handle_matches_path(self) -> bool:
        if self._handle is None:
            return False
        try:
            current = os.fstat(self._handle.fileno())
            directory_fd = _log_directory_fd(self.log_path, create=True)
            try:
                path_info = os.stat(self.log_path.name, dir_fd=directory_fd, follow_symlinks=False)
            finally:
                os.close(directory_fd)
            return (
                stat.S_ISREG(path_info.st_mode)
                and path_info.st_uid == os.getuid()
                and _mode(path_info.st_mode) == 0o600
                and (current.st_dev, current.st_ino) == (path_info.st_dev, path_info.st_ino)
            )
        except OSError:
            return False

    def _archive_name(self, directory_fd: int) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        for _ in range(100000):
            self._archive_sequence += 1
            name = f"_agent_{timestamp}_{self._archive_sequence:06d}.log"
            try:
                os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                return name
        raise FileExistsError("could not reserve a unique agent log archive name")

    def _bind_descriptors(self, handle: BinaryIO) -> None:
        os.dup2(handle.fileno(), 1, inheritable=True)
        os.dup2(handle.fileno(), 2, inheritable=True)
        self._descriptors_bound = True
        try:
            faulthandler.enable(file=sys.stderr, all_threads=True)
        except (RuntimeError, OSError, ValueError):
            pass

    def _rotate_locked(self) -> bool:
        if not self._open() or self._handle is None:
            return False
        try:
            self._handle.flush()
            current = os.fstat(self._handle.fileno())
            try:
                threshold = current.st_size >= self.max_bytes
            except OSError:
                threshold = False
            path_matches = self._handle_matches_path()
            if path_matches and not threshold:
                if self.capture_mode == "detached" and not self._descriptors_bound:
                    self._bind_descriptors(self._handle)
                self._notify_health("healthy", None)
                return True
            directory_fd = _log_directory_fd(self.log_path, create=True)
            try:
                temporary_name = f".agent-log-{uuid.uuid4().hex}.tmp"
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0)
                new_fd = os.open(temporary_name, flags, 0o600, dir_fd=directory_fd)
                os.fchmod(new_fd, 0o600)
                os.set_inheritable(new_fd, False)
                new_handle = os.fdopen(new_fd, "ab", buffering=0)
                archive_name: str | None = None
                try:
                    try:
                        old_path = os.stat(self.log_path.name, dir_fd=directory_fd, follow_symlinks=False)
                    except FileNotFoundError:
                        old_path = None
                    if old_path is not None:
                        if not stat.S_ISREG(old_path.st_mode) or old_path.st_uid != os.getuid():
                            raise PermissionError("active log path is not a regular owned file")
                        archive_name = self._archive_name(directory_fd)
                        os.rename(
                            self.log_path.name,
                            archive_name,
                            src_dir_fd=directory_fd,
                            dst_dir_fd=directory_fd,
                        )
                    os.replace(temporary_name, self.log_path.name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
                    old_handle = self._handle
                    self._handle = new_handle
                    if self.capture_mode == "detached":
                        self._descriptors_bound = False
                        self._bind_descriptors(new_handle)
                    marker = f"[qexp log rotation {_utc_now()} archive={archive_name or 'recreated'}]\n".encode()
                    new_handle.write(marker)
                    new_handle.flush()
                    try:
                        old_handle.close()
                    except OSError:
                        pass
                    self._retry_delay = 0.25
                    self._next_retry_at = 0.0
                    self._notify_health("healthy", None)
                    return True
                except BaseException:
                    if self._handle is not new_handle:
                        try:
                            new_handle.close()
                        except OSError:
                            pass
                    else:
                        try:
                            old_handle.close()
                        except (OSError, UnboundLocalError):
                            pass
                    raise
            finally:
                os.close(directory_fd)
        except Exception as exc:
            self._schedule_retry(exc)
            return False

    def _check_once(self) -> None:
        if time.monotonic() < self._next_retry_at:
            return
        with self._lock:
            if self._stopped:
                return
            self._rotate_locked()

    def _run(self) -> None:
        while not self._stop.wait(1.0):
            self._check_once()

    def start(self) -> None:
        with self._lock:
            if self._started or self._stopped:
                return
            self._started = True
            if self._open() and self._handle is not None:
                if self.capture_mode == "detached":
                    try:
                        self._bind_descriptors(self._handle)
                    except Exception as exc:
                        self._schedule_retry(exc)
                    else:
                        self._notify_health("healthy", None)
                else:
                    self._notify_health("healthy", None)
            self._thread = threading.Thread(target=self._run, name="qexp-agent-log-rotation", daemon=True)
            self._thread.start()

    def write(self, text: str) -> bool:
        if not isinstance(text, str):
            return False
        payload = text.encode("utf-8", errors="replace")
        try:
            with self._lock:
                if self._stopped or not self._open() or self._handle is None:
                    return False
                self._handle.write(payload)
                return True
        except Exception as exc:
            self._schedule_retry(exc)
            return False

    def stop(self) -> bool:
        with self._lock:
            if self._stopped:
                return self._health == "healthy"
            self._stopped = True
            self._stop.set()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        stopped_cleanly = thread is None or not thread.is_alive()
        with self._lock:
            self._stopped = False
            stopped_cleanly = self._rotate_locked() and stopped_cleanly
            self._stopped = True
            if self._handle is not None:
                try:
                    self._handle.close()
                except OSError:
                    stopped_cleanly = False
                self._handle = None
        return stopped_cleanly


__all__ = [
    "DEFAULT_LOG_MAX_BYTES",
    "MIN_LOG_MAX_BYTES",
    "MAX_LOG_MAX_BYTES",
    "PreparedAgentDiagnostics",
    "prepare_agent_diagnostics",
    "update_diagnostic_evidence",
    "read_diagnostic_record",
    "reconcile_agent_diagnostics",
    "diagnostics_status",
    "summarize_diagnostic_record",
    "AgentLogService",
    "bounded_exception_evidence",
    "parse_log_size",
    "format_log_size",
]
