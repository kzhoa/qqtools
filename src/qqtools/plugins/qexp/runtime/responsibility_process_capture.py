"""Bounded Linux runner census backed by durable pending observations."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from ..config_types import RootConfig
from ..infrastructure.host import host_instance_id
from .records import validate_identifier
from .responsibility_capture import CAPTURE_BATCH_SIZE, WriterCaptureCheckpoint, capture_admission
from .responsibility_import import recovery_locator
from .responsibility_store import Ledger, Unavailable, validate_captured_writer

PROC_ROOT = Path("/proc")
SCAN_FORMAT = "qexp-runner-process-sweep-v1"
COMMAND_BYTES = 65536
STAT_BYTES = 8192
RUNNER_MODULE = "qqtools.plugins.qexp.runner"
RUNNER_OPTIONS = {
    "--shared-root",
    "--runtime-root",
    "--machine",
    "--task-id",
    "--attempt-id",
    "--fencing-token",
    "--launch-id",
}


def _read_bounded(path: Path, limit: int) -> bytes:
    with path.open("rb") as stream:
        value = stream.read(limit + 1)
    if len(value) > limit:
        raise Unavailable(f"process capture input exceeds its byte bound: {path.name}")
    return value


def _process_scope() -> dict:
    scope = {
        "host_id": host_instance_id(),
        "boot_id": _read_bounded(PROC_ROOT / "sys/kernel/random/boot_id", 128).decode("ascii").strip(),
        "pid_namespace": (PROC_ROOT / "self/ns/pid").stat().st_ino,
    }
    validate_captured_writer({**scope, "pid": 1, "start_time_ticks": 0})
    return scope


def _process_stat(path: Path, pid: int) -> tuple[str, int]:
    value = os.fsdecode(_read_bounded(path, STAT_BYTES))
    try:
        prefix, tail = value.rsplit(")", 1)
        if int(prefix.split("(", 1)[0].strip()) != pid:
            raise ValueError("PID mismatch")
        fields = tail.split()
        ticks = int(fields[19])
        if ticks < 0 or len(fields[0]) != 1:
            raise ValueError("invalid process state")
        return fields[0], ticks
    except (ValueError, IndexError) as exc:
        raise Unavailable("process capture could not classify process stat") from exc


@dataclass(frozen=True)
class ProcessCaptureProgress:
    entries_visited: int
    writers_recorded: int
    revision: int
    is_sweep_complete: bool


class RunnerProcessCapture:
    """Capture published CLI runner identities in bounded, restartable slices.

    The caller establishes retention under lifecycle ownership. Reopen replays the
    exact durable batch; an unfinished /proc iterator restarts from its beginning.
    EOF records a process sweep only, never admission fencing or full discovery.
    No Task history is read and no process is launched, killed or signalled.
    """

    def __init__(self, cfg: RootConfig, ledger: Ledger, *, legacy_source: Path | None = None) -> None:
        self.cfg = cfg
        self.checkpoint = WriterCaptureCheckpoint(ledger, cfg.runtime_root, legacy_source=legacy_source)
        self._entries = None
        self._iterator_scope: dict | None = None
        self._admission: dict | None = None

    def _check_admission(self, checkpoint: WriterCaptureCheckpoint) -> None:
        if checkpoint.admission != self._admission:
            raise Unavailable("process capture admission changed")

    def close(self) -> None:
        if self._entries is not None:
            self._entries.close()
        self._entries = None
        self._iterator_scope = None

    def restart_after_reboot(self) -> bool:
        """Replay retained writers and restart only on a new boot of the same host.

        Durable retention must already exclude destructive lifecycle work. Neither
        a foreign host nor a same-boot PID namespace change proves writer exit.
        Retention and enrollment identity survive; evidence must be swept again.
        """
        self.close()
        with self.checkpoint.observe() as checkpoint:
            self._check_admission(checkpoint)
            scope = _process_scope()
            validate_captured_writer({**scope, "pid": 1, "start_time_ticks": 0})
            previous = checkpoint.progress
            if previous is None:
                return False
            old_scope = {key: previous.get(key) for key in scope}
            try:
                validate_captured_writer({**old_scope, "pid": 1, "start_time_ticks": 0})
            except (ValueError, TypeError) as exc:
                raise Unavailable("invalid prior process sweep scope") from exc
            previous = self._progress(previous, old_scope)
            if old_scope == scope:
                return False
            if old_scope["host_id"] != scope["host_id"] or old_scope["boot_id"] == scope["boot_id"]:
                raise Unavailable("process sweep restart requires a new boot on the same host")
            checkpoint.record(
                [],
                progress={
                    **self._progress(None, scope),
                    "revision": previous["revision"] + 1,
                },
            )
            return True

    def prepare_admission(self, owner: dict) -> None:
        """Start a fresh post-fence census once, retaining prior durable writers."""
        self.close()
        with self.checkpoint.observe() as checkpoint:
            scope = _process_scope()
            previous = checkpoint.progress
            revision = 0
            if previous is not None:
                old_scope = {key: previous.get(key) for key in scope}
                try:
                    validate_captured_writer({**old_scope, "pid": 1, "start_time_ticks": 0})
                except (ValueError, TypeError) as exc:
                    raise Unavailable("invalid prior process sweep scope") from exc
                previous = self._progress(previous, old_scope)
                if old_scope["host_id"] != scope["host_id"] or (
                    old_scope["boot_id"] == scope["boot_id"] and old_scope != scope
                ):
                    raise Unavailable("capture admission belongs to another host or PID namespace")
                revision = previous["revision"]
            checkpoint.admit(owner, progress={**self._progress(None, scope), "revision": revision + 1})
            self._admission = capture_admission(owner)

    @contextmanager
    def completed_sweep(self) -> Iterator[WriterCaptureCheckpoint]:
        """Guard a subsequent evidence sweep with validated process enrollment.

        This verifies process-sweep ordering and context, not an admission fence
        or complete discovery. Target and source retention remain pending.
        """
        with self.checkpoint.observe() as checkpoint:
            self._check_admission(checkpoint)
            state = self._progress(checkpoint.progress, _process_scope())
            if not state["is_sweep_complete"]:
                raise Unavailable("evidence capture requires a completed process sweep")
            yield checkpoint

    def _progress(self, previous: dict | None, scope: dict) -> dict:
        context = {
            "format": SCAN_FORMAT,
            "shared_root": str(self.cfg.shared_root.resolve()),
            "machine_name": self.cfg.machine_name,
            **scope,
        }
        if previous is None:
            return {**context, "revision": 0, "is_sweep_complete": False}
        if (
            set(previous) != {*context, "revision", "is_sweep_complete"}
            or any(previous[key] != value for key, value in context.items())
            or type(previous["revision"]) is not int
            or previous["revision"] < 1
            or type(previous["is_sweep_complete"]) is not bool
        ):
            raise Unavailable("process sweep context changed or its progress is invalid")
        return previous

    def _runner_locator(self, raw: bytes) -> tuple[str, dict, Path | None] | None:
        arguments = os.fsdecode(raw).rstrip("\0").split("\0")
        if arguments[1:3] != ["-m", RUNNER_MODULE] or arguments[3:4] == ["--guardian"]:
            return None
        if not raw.endswith(b"\0"):
            raise Unavailable("runner command is incomplete")
        options = arguments[3:]
        if len(options) % 2 or len(set(options[::2])) != len(options[::2]):
            raise Unavailable("runner command options are ambiguous")
        fields = dict(zip(options[::2], options[1::2]))
        if fields.get("--machine") != self.cfg.machine_name:
            return None
        shared = fields.get("--shared-root")
        if not shared or not Path(shared).is_absolute():
            raise Unavailable("runner command has no absolute shared root")
        if Path(shared).resolve() != self.cfg.shared_root.resolve():
            return None
        root = fields.get("--runtime-root")
        if not root or not Path(root).is_absolute():
            raise Unavailable("runner command has no absolute runtime root")
        runtime = Path(root).resolve()
        if runtime not in {self.checkpoint.runtime_root, self.checkpoint.legacy_source}:
            return None
        if set(fields) != RUNNER_OPTIONS:
            raise Unavailable("runner command does not match the supported launch protocol")
        try:
            identity = validate_identifier(fields["--attempt-id"], "attempt_id")
            task_id = validate_identifier(fields["--task-id"], "task_id")
            validate_identifier(fields["--launch-id"], "launch_id")
            if int(fields["--fencing-token"]) < 1:
                raise ValueError("invalid fencing token")
            payload = recovery_locator(identity, {"task_id": task_id})
        except ValueError as exc:
            raise Unavailable("runner command has an invalid recovery identity") from exc
        source = runtime if runtime != self.checkpoint.runtime_root else None
        return identity, payload, source

    def _observe(self, entry: os.DirEntry, scope: dict) -> dict | None:
        if not entry.name.isascii() or not entry.name.isdecimal() or int(entry.name) < 1:
            return None
        pid = int(entry.name)
        process = Path(entry.path)
        try:
            metadata = entry.stat(follow_symlinks=False)
            if metadata.st_uid != os.getuid():
                return None
            if not stat.S_ISDIR(metadata.st_mode):
                raise Unavailable("process capture encountered a non-directory PID")
            with (process / "cmdline").open("rb") as stream:
                raw = stream.read(COMMAND_BYTES + 1)
            # Most same-user processes are unrelated.  Filter them before strict
            # identity parsing so a malformed transient /proc stat cannot stall
            # recovery enrollment for every project on the machine.
            if len(raw) > COMMAND_BYTES:
                if os.fsdecode(raw).split("\0")[1:3] == ["-m", RUNNER_MODULE]:
                    raise Unavailable("runner command exceeds its capture byte bound")
                return None
            if self._runner_locator(raw) is None:
                return None
            before = _process_stat(process / "stat", pid)
            if before[0] == "Z":
                return None
            with (process / "cmdline").open("rb") as stream:
                raw = stream.read(COMMAND_BYTES + 1)
            after = _process_stat(process / "stat", pid)
        except (FileNotFoundError, ProcessLookupError):
            return None
        if before[1] != after[1] or after[0] == "Z":
            return None
        if len(raw) > COMMAND_BYTES:
            raise Unavailable("runner command exceeds its capture byte bound")
        locator = self._runner_locator(raw)
        if locator is None:
            return None
        identity, payload, source = locator
        record = {
            "identity": identity,
            "payload": payload,
            "writer": {**scope, "pid": pid, "start_time_ticks": after[1]},
        }
        if source is not None:
            record["legacy_source"] = str(source)
        return record

    def take(self, limit: int = CAPTURE_BATCH_SIZE) -> ProcessCaptureProgress:
        """Visit at most limit directory entries after replaying durable observations."""
        if type(limit) is not int or not 1 <= limit <= CAPTURE_BATCH_SIZE:
            raise ValueError("process capture limit must be in 1..64")
        try:
            with self.checkpoint.observe() as checkpoint:
                self._check_admission(checkpoint)
                scope = _process_scope()
                state = self._progress(checkpoint.progress, scope)
                if state["is_sweep_complete"]:
                    self.close()
                    return ProcessCaptureProgress(0, 0, state["revision"], True)
                # Another scanner can advance the journal without invalidating
                # this iterator: each finite pass captures its own visited entries.
                if self._iterator_scope != scope:
                    self.close()
                if self._entries is None:
                    self._entries = os.scandir(PROC_ROOT)
                    self._iterator_scope = dict(scope)
                visited = 0
                records = []
                is_complete = False
                for _ in range(limit):
                    try:
                        entry = next(self._entries)
                    except StopIteration:
                        is_complete = True
                        break
                    visited += 1
                    observation = self._observe(entry, scope)
                    if observation is not None:
                        records.append(observation)
                progress = {**state, "revision": state["revision"] + 1, "is_sweep_complete": is_complete}
                checkpoint.record(records, progress=progress)
                if is_complete:
                    self.close()
                return ProcessCaptureProgress(visited, len(records), progress["revision"], is_complete)
        except BaseException:
            self.close()
            raise
