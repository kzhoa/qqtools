"""Task log access commands for qexp."""

from __future__ import annotations

import codecs
import math
import os
import stat
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, TextIO

from .. import observer
from ..config_types import RootConfig
from ..layout import shared_attempt_log_path
from ..runtime.paths import attempt_path, shared_paths
from ..runtime.records import AttemptRecord, TaskRecord
from ..runtime.store import iter_json, read_json
from ..runtime.tasks import load_task


def _latest_attempt_for_logs(cfg: RootConfig, task: TaskRecord) -> AttemptRecord:
    number = task.attempt_control.get("current_attempt_number")
    if number is not None:
        path = attempt_path(cfg.shared_root, task.task_id, number)
        return AttemptRecord.from_dict(read_json(path))
    attempts_dir = shared_paths(cfg.shared_root)["attempts"] / task.task_id
    attempts = [AttemptRecord.from_dict(read_json(path)) for path in iter_json(attempts_dir)]
    if not attempts:
        raise FileNotFoundError(f"Task {task.task_id!r} has no Attempt records.")
    return max(attempts, key=lambda item: item.attempt_number)


def get_log_path(cfg: RootConfig, task_id: str) -> Path:
    task = load_task(cfg, task_id)
    attempt = _latest_attempt_for_logs(cfg, task)
    log_references = attempt.process.get("log_references") or []
    if log_references:
        return Path(log_references[0])
    return shared_attempt_log_path(cfg, task_id, attempt.attempt_id)


def _validate_read_tail(tail_lines: int | None) -> None:
    if tail_lines is not None and (type(tail_lines) is not int or tail_lines < 0):
        raise ValueError("tail_lines must be a non-negative integer or None.")


def read_logs(cfg: RootConfig, task_id: str, *, tail_lines: int | None = None) -> str:
    """Read the selected finite log, optionally limiting it to its last lines."""
    _validate_read_tail(tail_lines)
    path = get_log_path(cfg, task_id)
    if not path.exists():
        task = load_task(cfg, task_id)
        attempt = _latest_attempt_for_logs(cfg, task)
        raise FileNotFoundError(
            f"log for Task {task_id!r} Attempt {attempt.attempt_id!r} on machine "
            f"{attempt.machine_name!r} was not found at {path}"
        )
    if tail_lines is None:
        return path.read_text(encoding="utf-8", errors="replace")
    if tail_lines == 0:
        return ""
    with path.open("rb") as handle:
        file_stat = os.fstat(handle.fileno())
        _ensure_regular(path, file_stat.st_mode)
        offset = _tail_offset(handle, file_stat.st_size, tail_lines, 64 * 1024)
        handle.seek(offset)
        return handle.read().decode("utf-8", errors="replace")


def tail_log(cfg: RootConfig, task_id: str) -> None:
    print(read_logs(cfg, task_id), end="")


class _SelectionChanged(RuntimeError):
    """Raised when bounded tail setup observes a new current Attempt."""


class _GenerationChanged(RuntimeError):
    """Raised when a log path changes while an attachment is being opened."""


class _NonRegularLog(RuntimeError):
    """Raised when a selected log target is not a regular file."""


class _OutputClosed(RuntimeError):
    """Raised internally when the follower's stdout pipe is closed."""


@dataclass(frozen=True, slots=True)
class _LogDescriptor:
    attempt_id: str
    attempt_number: int
    phase: str
    machine_name: str
    log_path: Path

    @property
    def identity(self) -> tuple[str, int, str]:
        return (self.attempt_id, self.attempt_number, str(self.log_path))


@dataclass(slots=True)
class _Attachment:
    descriptor: _LogDescriptor
    handle: BinaryIO
    file_identity: tuple[int, int]
    offset: int
    decoder: Any


_UNSET = object()


def _validate_follow_options(tail_lines: int, interval_seconds: int | float, chunk_size: int) -> None:
    if type(tail_lines) is not int or tail_lines < 0:
        raise ValueError("tail_lines must be a non-negative integer.")
    if type(interval_seconds) not in (int, float) or not math.isfinite(interval_seconds) or interval_seconds < 1:
        raise ValueError("interval_seconds must be a finite number of at least 1 second.")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")


def _descriptor_from_payload(payload: Any) -> _LogDescriptor | None:
    if not isinstance(payload, Mapping):
        raise RuntimeError("Task observation returned a malformed payload.")
    selected = payload.get("selected_attempt")
    if selected is None:
        return None
    if not isinstance(selected, Mapping):
        raise RuntimeError("Task observation returned a malformed selected Attempt.")
    attempt_id = selected.get("attempt_id")
    attempt_number = selected.get("attempt_number")
    phase = selected.get("phase")
    machine_name = selected.get("machine_name")
    log_value = selected.get("log_path")
    if (
        not isinstance(attempt_id, str)
        or not attempt_id
        or type(attempt_number) is not int
        or attempt_number < 1
        or not isinstance(phase, str)
        or not isinstance(machine_name, str)
        or not isinstance(log_value, str)
        or not log_value
        or "\x00" in log_value
    ):
        raise RuntimeError("Task observation returned an invalid selected Attempt descriptor.")
    try:
        log_path = Path(log_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Task observation returned an invalid selected Attempt log path.") from exc
    return _LogDescriptor(attempt_id, attempt_number, phase, machine_name, log_path)


def _same_descriptor(left: _LogDescriptor | None, right: _LogDescriptor | None) -> bool:
    if left is None or right is None:
        return left is right
    return left.identity == right.identity


def _notice(stderr: TextIO, message: str) -> None:
    stderr.write(f"qexp: {message}\n")
    stderr.flush()


def _ensure_regular(path: Path, mode: int) -> None:
    if not stat.S_ISREG(mode):
        raise _NonRegularLog(f"log target {path} is not a regular file.")


def _tail_offset(
    handle: BinaryIO,
    size: int,
    tail_lines: int,
    chunk_size: int,
    *,
    revalidate: Callable[[], bool] | None = None,
) -> int:
    if tail_lines == 0 or size == 0:
        handle.seek(size)
        return size
    handle.seek(size - 1)
    last = handle.read(1)
    if len(last) != 1:
        handle.seek(0)
        return 0
    target_newlines = tail_lines + (1 if last == b"\n" else 0)
    cursor = size
    while cursor > 0:
        start = max(0, cursor - chunk_size)
        handle.seek(start)
        chunk = handle.read(cursor - start)
        if not chunk:
            break
        for index in range(len(chunk) - 1, -1, -1):
            if chunk[index] == 0x0A:
                target_newlines -= 1
                if target_newlines == 0:
                    offset = start + index + 1
                    if revalidate is not None and not revalidate():
                        raise _SelectionChanged
                    handle.seek(offset)
                    return offset
        cursor = start
        if cursor and revalidate is not None and not revalidate():
            raise _SelectionChanged
    handle.seek(0)
    return 0


def _open_attachment(
    descriptor: _LogDescriptor,
    tail_lines: int,
    chunk_size: int,
    *,
    revalidate: Callable[[], bool] | None = None,
) -> _Attachment:
    path = descriptor.log_path
    if revalidate is not None and not revalidate():
        raise _SelectionChanged
    initial_stat = path.stat()
    _ensure_regular(path, initial_stat.st_mode)
    handle = path.open("rb")
    try:
        opened_stat = os.fstat(handle.fileno())
        _ensure_regular(path, opened_stat.st_mode)
        file_identity = (opened_stat.st_dev, opened_stat.st_ino)
        if file_identity != (initial_stat.st_dev, initial_stat.st_ino):
            raise _GenerationChanged
        offset = _tail_offset(handle, opened_stat.st_size, tail_lines, chunk_size, revalidate=revalidate)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        handle.seek(offset)
        return _Attachment(descriptor, handle, file_identity, offset, decoder)
    except BaseException:
        handle.close()
        raise


def _close_attachment(attachment: _Attachment | None) -> None:
    if attachment is None:
        return
    try:
        attachment.handle.close()
    except OSError:
        pass


def _generation_status(attachment: _Attachment) -> str:
    handle_stat = os.fstat(attachment.handle.fileno())
    _ensure_regular(attachment.descriptor.log_path, handle_stat.st_mode)
    if (handle_stat.st_dev, handle_stat.st_ino) != attachment.file_identity or handle_stat.st_size < attachment.offset:
        return "changed"
    path_stat = attachment.descriptor.log_path.stat()
    _ensure_regular(attachment.descriptor.log_path, path_stat.st_mode)
    if (path_stat.st_dev, path_stat.st_ino) != attachment.file_identity or path_stat.st_size < attachment.offset:
        return "changed"
    if handle_stat.st_size > attachment.offset or path_stat.st_size > attachment.offset:
        return "grown"
    return "eof"


def _write_application(stdout: TextIO, text: str) -> None:
    if not text:
        return
    try:
        stdout.write(text)
        stdout.flush()
    except BrokenPipeError as exc:
        raise _OutputClosed from exc


def _finish_attachment(attachment: _Attachment, stdout: TextIO) -> None:
    _write_application(stdout, attachment.decoder.decode(b"", final=True))


def _transition_notice(stderr: TextIO, previous: _LogDescriptor | None, current: _LogDescriptor | None) -> None:
    if previous is not None and current is not None:
        _notice(
            stderr,
            f"Attempt boundary: switching from Attempt {previous.attempt_id} "
            f"(#{previous.attempt_number}) to Attempt {current.attempt_id} "
            f"(#{current.attempt_number}).",
        )
    elif previous is not None:
        _notice(
            stderr,
            f"Attempt {previous.attempt_id} (#{previous.attempt_number}) is no longer selected; "
            "waiting for the current Attempt.",
        )
    elif current is not None:
        _notice(
            stderr,
            f"following current Attempt {current.attempt_id} (#{current.attempt_number}).",
        )


def _discontinuity_notice(stderr: TextIO, descriptor: _LogDescriptor) -> None:
    _notice(
        stderr,
        f"log discontinuity for Attempt {descriptor.attempt_id} (#{descriptor.attempt_number}); "
        "a prefix may be skipped and lost bytes cannot be recovered.",
    )


def _format_descriptor(descriptor: _LogDescriptor) -> str:
    return f"Attempt {descriptor.attempt_id} (#{descriptor.attempt_number})"


def follow_logs(
    cfg: RootConfig,
    task_id: str,
    *,
    tail_lines: int = 100,
    interval_seconds: int | float = 2,
    follow_retries: bool = False,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    sleep: Callable[[float], None] = time.sleep,
    chunk_size: int = 64 * 1024,
) -> int:
    """Follow the selected current Task Attempt log without retaining its contents."""
    _validate_follow_options(tail_lines, interval_seconds, chunk_size)
    attachment: _Attachment | None = None
    observed: object = _UNSET
    pending_payload: Mapping[str, Any] | None = None
    wait_notice_key: tuple[Any, ...] | None = None
    transient_error_key: tuple[Any, ...] | None = None
    terminal_failure_key: tuple[str, int, str] | None = None

    def accept_descriptor(descriptor: _LogDescriptor | None) -> None:
        nonlocal attachment, observed, wait_notice_key, transient_error_key, terminal_failure_key
        if observed is _UNSET:
            observed = descriptor
            return
        previous = observed
        if _same_descriptor(previous if isinstance(previous, _LogDescriptor) else None, descriptor):
            return
        if attachment is not None:
            _close_attachment(attachment)
            attachment = None
        _transition_notice(
            stderr,
            previous if isinstance(previous, _LogDescriptor) else None,
            descriptor,
        )
        observed = descriptor
        wait_notice_key = None
        transient_error_key = None
        terminal_failure_key = None

    def revalidate_open(descriptor: _LogDescriptor, latest: list[Mapping[str, Any] | None]) -> bool:
        payload = observer.inspect_current_task(cfg, task_id)
        latest[0] = payload
        return _same_descriptor(descriptor, _descriptor_from_payload(payload))

    def report_wait(key: tuple[Any, ...], message: str) -> None:
        nonlocal wait_notice_key
        if wait_notice_key != key:
            _notice(stderr, message)
            wait_notice_key = key

    def report_transient(descriptor: _LogDescriptor, exc: OSError) -> None:
        nonlocal transient_error_key
        key = (descriptor.identity, type(exc).__name__)
        if transient_error_key != key:
            _notice(stderr, f"log for {_format_descriptor(descriptor)} is temporarily unavailable: {exc}")
            transient_error_key = key

    def report_success() -> None:
        nonlocal transient_error_key, terminal_failure_key, wait_notice_key
        if transient_error_key is not None and attachment is not None:
            _notice(stderr, f"log for {_format_descriptor(attachment.descriptor)} recovered.")
        transient_error_key = None
        terminal_failure_key = None
        wait_notice_key = None

    def wait_for_terminal_retry(descriptor: _LogDescriptor, exc: BaseException) -> None:
        nonlocal terminal_failure_key
        if terminal_failure_key == descriptor.identity:
            raise RuntimeError(
                f"log for Task {task_id!r} Attempt {descriptor.attempt_id!r} "
                f"remained unavailable after one terminal retry: {exc}"
            ) from exc
        terminal_failure_key = descriptor.identity
        _notice(
            stderr,
            f"terminal log for {_format_descriptor(descriptor)} is unavailable; retrying once.",
        )
        sleep(interval_seconds)

    def wait_for_missing(descriptor: _LogDescriptor, terminal: bool, reason: str) -> None:
        if terminal:
            wait_for_terminal_retry(descriptor, FileNotFoundError(reason))
            return
        report_wait(
            ("missing", descriptor.identity),
            f"waiting for log for {_format_descriptor(descriptor)} at {descriptor.log_path}.",
        )
        sleep(interval_seconds)

    def wait_for_transient(descriptor: _LogDescriptor, terminal: bool, exc: OSError) -> None:
        if terminal:
            report_transient(descriptor, exc)
            wait_for_terminal_retry(descriptor, exc)
            return
        report_transient(descriptor, exc)
        sleep(interval_seconds)

    try:
        while True:
            payload = pending_payload if pending_payload is not None else observer.inspect_current_task(cfg, task_id)
            pending_payload = None
            descriptor = _descriptor_from_payload(payload)
            accept_descriptor(descriptor)
            terminal = bool(payload.get("terminal"))

            if descriptor is None:
                if payload.get("observation_reason") == "concurrent_transition":
                    report_wait(
                        ("concurrent-transition", payload.get("revision")),
                        f"Task {task_id!r} changed during observation; retrying selection.",
                    )
                    sleep(interval_seconds)
                    continue
                if terminal and not follow_retries:
                    return 0
                if terminal:
                    report_wait(
                        ("terminal-retry", payload.get("phase"), payload.get("reason")),
                        f"Task {task_id!r} is terminal; waiting for a later retry.",
                    )
                else:
                    report_wait(
                        ("queued", payload.get("phase"), payload.get("reason")),
                        f"Task {task_id!r} has no selected current Attempt; waiting.",
                    )
                sleep(interval_seconds)
                continue

            if attachment is None:
                latest: list[Mapping[str, Any] | None] = [None]
                try:
                    attachment = _open_attachment(
                        descriptor,
                        tail_lines,
                        chunk_size,
                        revalidate=lambda: revalidate_open(descriptor, latest),
                    )
                except _SelectionChanged:
                    pending_payload = latest[0]
                    if pending_payload is None:
                        pending_payload = observer.inspect_current_task(cfg, task_id)
                    continue
                except _GenerationChanged:
                    continue
                except _NonRegularLog as exc:
                    raise RuntimeError(str(exc)) from exc
                except FileNotFoundError as exc:
                    wait_for_missing(descriptor, terminal, str(exc))
                    continue
                except (PermissionError, IsADirectoryError, NotADirectoryError) as exc:
                    raise RuntimeError(
                        f"log for Task {task_id!r} Attempt {descriptor.attempt_id!r} cannot be read: {exc}"
                    ) from exc
                except OSError as exc:
                    wait_for_transient(descriptor, terminal, exc)
                    continue

            try:
                generation = _generation_status(attachment)
            except _NonRegularLog as exc:
                raise RuntimeError(str(exc)) from exc
            except FileNotFoundError as exc:
                _discontinuity_notice(stderr, descriptor)
                _close_attachment(attachment)
                attachment = None
                wait_for_missing(descriptor, terminal, str(exc))
                continue
            except (PermissionError, IsADirectoryError, NotADirectoryError) as exc:
                raise RuntimeError(
                    f"log for Task {task_id!r} Attempt {descriptor.attempt_id!r} cannot be read: {exc}"
                ) from exc
            except OSError as exc:
                wait_for_transient(descriptor, terminal, exc)
                continue
            if generation == "changed":
                _discontinuity_notice(stderr, descriptor)
                _close_attachment(attachment)
                attachment = None
                continue

            try:
                read_offset = attachment.offset
                attachment.handle.seek(attachment.offset)
                data = attachment.handle.read(chunk_size)
            except (PermissionError, IsADirectoryError, NotADirectoryError) as exc:
                raise RuntimeError(
                    f"log for Task {task_id!r} Attempt {descriptor.attempt_id!r} cannot be read: {exc}"
                ) from exc
            except OSError as exc:
                latest_payload = observer.inspect_current_task(cfg, task_id)
                latest_descriptor = _descriptor_from_payload(latest_payload)
                if not _same_descriptor(descriptor, latest_descriptor):
                    accept_descriptor(latest_descriptor)
                    pending_payload = latest_payload
                    continue
                pending_payload = None
                wait_for_transient(descriptor, bool(latest_payload.get("terminal")), exc)
                continue

            latest_payload = observer.inspect_current_task(cfg, task_id)
            latest_descriptor = _descriptor_from_payload(latest_payload)
            if not _same_descriptor(descriptor, latest_descriptor):
                accept_descriptor(latest_descriptor)
                pending_payload = latest_payload
                continue
            pending_payload = latest_payload
            attachment.offset = read_offset + len(data)
            try:
                after_read = _generation_status(attachment)
            except _NonRegularLog as exc:
                raise RuntimeError(str(exc)) from exc
            except FileNotFoundError as exc:
                _discontinuity_notice(stderr, descriptor)
                _close_attachment(attachment)
                attachment = None
                pending_payload = None
                wait_for_missing(descriptor, bool(latest_payload.get("terminal")), str(exc))
                continue
            except (PermissionError, IsADirectoryError, NotADirectoryError) as exc:
                raise RuntimeError(
                    f"log for Task {task_id!r} Attempt {descriptor.attempt_id!r} cannot be read: {exc}"
                ) from exc
            except OSError as exc:
                attachment.offset = read_offset
                pending_payload = None
                wait_for_transient(descriptor, bool(latest_payload.get("terminal")), exc)
                continue
            if after_read == "changed":
                _discontinuity_notice(stderr, descriptor)
                _close_attachment(attachment)
                attachment = None
                continue

            report_success()
            try:
                _write_application(stdout, attachment.decoder.decode(data, final=False))
            except _OutputClosed:
                return 0
            latest_terminal = bool(latest_payload.get("terminal"))
            if after_read == "grown" or len(data) >= chunk_size:
                continue
            if not latest_terminal or follow_retries:
                pending_payload = None
                sleep(interval_seconds)
                continue
            try:
                _finish_attachment(attachment, stdout)
            except _OutputClosed:
                return 0
            return 0
    finally:
        _close_attachment(attachment)
