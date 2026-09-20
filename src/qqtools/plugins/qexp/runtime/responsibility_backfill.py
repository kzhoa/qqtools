"""Restartable local evidence capture; a completed sweep is not activation proof."""

from __future__ import annotations

import stat
import uuid
from collections.abc import Generator, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .authority_scan import iter_evidence_entries
from .locks import exclusive
from .paths import local_paths
from .records import validate_identifier
from .responsibility import responsibility_root
from .responsibility_cleanup import evidence_write_guard
from .responsibility_import import RECORD_KEYS, recovery_locator
from .responsibility_store import Conflict, Ledger, Unavailable
from .store import atomic_replace, read_json, read_json_limited

if TYPE_CHECKING:
    from .responsibility_process_capture import RunnerProcessCapture

FORMAT = "qexp-local-responsibility-backfill-v1"
LANES = tuple(RECORD_KEYS)
BATCH_SIZE = 64
STATE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class BackfillProgress:
    capture_id: str
    completed_lanes: int
    pending_records: int
    entries_visited: int
    records_processed: int
    total_lanes: int = len(LANES)

    @property
    def is_sweep_complete(self) -> bool:
        return self.completed_lanes == self.total_lanes


def _relative_record(name: str, value: object) -> Path:
    if not isinstance(value, str):
        raise Unavailable("backfill path must be relative JSON evidence")
    path = Path(value)
    depth = 2 if name == "termination_decisions" else 1
    if (
        path.is_absolute()
        or str(path) != value
        or len(path.parts) != depth
        or any(part in {".", ".."} for part in path.parts)
        or path.suffix != ".json"
    ):
        raise Unavailable(f"unexpected {name} evidence path: {value}")
    return path


def capture_local_record(
    ledger: Ledger,
    runtime_root: Path,
    name: str,
    relative: Path,
    *,
    source_root: Path | None = None,
    should_require_record: bool = False,
) -> None:
    """Keep source evidence intact and capture identity under the cleanup fence."""
    relative = _relative_record(name, str(relative))
    identity = relative.parent.name if name == "termination_decisions" else relative.stem
    validate_identifier(identity, "attempt_id")
    if source_root is not None and (not source_root.is_absolute() or source_root.resolve() == runtime_root.resolve()):
        raise ValueError("backfill source must be a different absolute runtime")
    directory = local_paths(source_root or runtime_root)[name]
    path = directory / relative
    with evidence_write_guard(runtime_root, identity) as acquired:
        if not acquired:
            raise Conflict(f"backfill evidence is busy: {identity}")
        try:
            for parent in {directory, path.parent}:
                if not stat.S_ISDIR(parent.stat(follow_symlinks=False).st_mode):
                    raise Unavailable(f"backfill evidence parent is not a directory: {parent}")
            if not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
                raise Unavailable(f"backfill evidence is not a regular file: {path}")
            value = read_json(path)
        except FileNotFoundError:
            if should_require_record:
                raise Unavailable(f"retained backfill evidence disappeared: {path}") from None
            # A cooperating cleanup may retire the source after enumeration.
            return
        record = value.get(RECORD_KEYS[name])
        if not isinstance(record, dict) or record.get("attempt_id", identity) != identity:
            raise Unavailable(f"backfill evidence identity does not match its path: {path}")
        payload = recovery_locator(identity, record)
        if source_root is None:
            ledger.capture_local(identity, payload)
        else:
            ledger.capture_source(identity, payload, source_root)


def capture_local_writer(
    ledger: Ledger,
    runtime_root: Path,
    identity: str,
    payload: dict,
    writer: dict,
    *,
    source_root: Path | None = None,
) -> int:
    """Persist an observed process locator under the same guard as cleanup.

    Caller-owned process discovery supplies the actual Linux identity. Capturing
    one writer neither proves a complete inventory nor authorizes its execution.
    """
    if ledger.root.resolve() != responsibility_root(runtime_root).resolve():
        raise ValueError("captured writer ledger does not belong to this runtime")
    validate_identifier(identity, "attempt_id")
    locator = recovery_locator(identity, payload)
    with evidence_write_guard(runtime_root, identity) as acquired:
        if not acquired:
            raise Conflict(f"captured writer evidence is busy: {identity}")
        if source_root is not None:
            if not source_root.is_absolute() or source_root.resolve() == runtime_root.resolve():
                raise ValueError("captured writer source must be a different absolute runtime")
            return ledger.capture_writer(identity, locator, writer, source_root=source_root)
        return ledger.capture_writer(identity, locator, writer)


class ResponsibilityBackfill:
    """Capture one lane in bounded slices, retaining a durable pending batch.

    Restart replays the pending batch and rescans only the unfinished lane. Prior
    captures are idempotent; completed lanes are not rescanned. The caller must
    separately fence old writers and cover new publications before activation.
    With process_capture, require its completed sweep and retain both roots while
    a separate checkpoint captures target and legacy evidence into the target.
    This class never deletes evidence, launches work, or declares coverage active.
    """

    def __init__(self, runtime_root: Path, *, process_capture: RunnerProcessCapture | None = None) -> None:
        self.runtime_root = runtime_root.resolve()
        self.process_capture = process_capture
        self.sources = [self.runtime_root]
        if process_capture is not None:
            checkpoint = process_capture.checkpoint
            if checkpoint.runtime_root != self.runtime_root:
                raise ValueError("backfill process capture belongs to another runtime")
            if checkpoint.legacy_source is not None:
                self.sources.append(checkpoint.legacy_source)
        self.total_lanes = len(LANES) * len(self.sources)
        filename = "responsibility-backfill.json" if process_capture is None else "responsibility-capture-backfill.json"
        self.path = self.runtime_root / filename
        self._entries: Generator[Path | None, None, None] | None = None
        self._cursor_key: tuple[str, int] | None = None

    def close(self) -> None:
        if self._entries is not None:
            self._entries.close()
        self._entries = None
        self._cursor_key = None

    def _load(self) -> tuple[Ledger, dict]:
        context = {}
        if self.process_capture is not None:
            context = {
                "writer_capture_id": self.process_capture.checkpoint.capture_id,
                "sources": [str(root) for root in self.sources],
            }
        try:
            state = read_json_limited(self.path, max_bytes=STATE_BYTES)
        except FileNotFoundError:
            if self.process_capture is None:
                with exclusive(self.runtime_root / "locks" / "responsibility-initialize.lock"):
                    ledger = Ledger.open_or_create(responsibility_root(self.runtime_root))
            else:
                ledger = self.process_capture.checkpoint.ledger
            state = {
                "format": FORMAT,
                **context,
                **self._process_revision(),
                "instance": ledger.instance,
                "capture_id": uuid.uuid4().hex,
                "revision": 0,
                "lane": 0,
                "pending": [],
                "at_end": False,
            }
            atomic_replace(self.path, state)
            return ledger, state
        # A checkpoint must never initialize a replacement for its missing ledger.
        ledger = Ledger(responsibility_root(self.runtime_root))
        if (
            state.get("format") != FORMAT
            or any(state.get(key) != value for key, value in context.items())
            or state.get("instance") != ledger.instance
            or not isinstance(state.get("capture_id"), str)
            or len(state["capture_id"]) != 32
            or any(character not in "0123456789abcdef" for character in state["capture_id"])
            or type(state.get("revision")) is not int
            or state["revision"] < 0
            or type(state.get("lane")) is not int
            or not 0 <= state["lane"] <= self.total_lanes
            or type(state.get("at_end")) is not bool
            or not isinstance(state.get("pending"), list)
            or len(state["pending"]) > BATCH_SIZE
            or (state["lane"] == self.total_lanes and (state["pending"] or state["at_end"]))
        ):
            raise Unavailable("invalid local backfill checkpoint or replaced responsibility ledger")
        if state["lane"] < self.total_lanes:
            for path in state["pending"]:
                _relative_record(LANES[state["lane"] % len(LANES)], path)
        if self.process_capture is not None and "writer_sweep_revision" in state:
            revision = state["writer_sweep_revision"]
            current = self._process_revision()["writer_sweep_revision"]
            if type(revision) is not int or not 1 <= revision <= current:
                raise Unavailable("invalid evidence process sweep revision")
        return ledger, state

    def _process_revision(self) -> dict:
        if self.process_capture is None:
            return {}
        return {"writer_sweep_revision": self.process_capture.checkpoint.progress["revision"]}

    def _needs_new_sweep(self, state: dict) -> bool:
        return any(state.get(key) != value for key, value in self._process_revision().items())

    def _restart_sweep(self, state: dict) -> None:
        if state["pending"]:
            raise Conflict("cannot restart evidence discovery before replaying its pending batch")
        self.close()
        state.update(
            self._process_revision(),
            capture_id=uuid.uuid4().hex,
            lane=0,
            at_end=False,
        )
        self._save(state)

    def _save(self, state: dict) -> None:
        state["revision"] += 1
        atomic_replace(self.path, state)

    def take(self, limit: int = BATCH_SIZE, *, should_cross_lanes: bool = False) -> BackfillProgress | None:
        """Bound new visits/record captures; None means the backfill lock is busy.

        Enrolled capture also replays at most 64 pending writer observations;
        busy or invalid process retention raises rather than admitting evidence.
        """
        if type(limit) is not int or not 1 <= limit <= BATCH_SIZE:
            raise ValueError("backfill limit must be in 1..64")
        retention = self.process_capture.completed_sweep() if self.process_capture is not None else nullcontext()
        try:
            with (
                retention,
                exclusive(self.runtime_root / "locks" / "responsibility-backfill.lock", blocking=False) as acquired,
            ):
                if not acquired:
                    return None
                remaining = limit
                visited = processed = 0
                while remaining:
                    progress = self._take_locked(remaining)
                    visited += progress.entries_visited
                    processed += progress.records_processed
                    remaining -= max(1, progress.entries_visited, progress.records_processed)
                    if not should_cross_lanes or progress.is_sweep_complete:
                        break
                return BackfillProgress(
                    progress.capture_id,
                    progress.completed_lanes,
                    progress.pending_records,
                    visited,
                    processed,
                    progress.total_lanes,
                )
        except BaseException:
            self.close()
            raise

    @contextmanager
    def completed_capture(self) -> Iterator[dict]:
        """Hold both retained sweeps while a lifecycle owner commits their proof."""
        if self.process_capture is None:
            raise Unavailable("completion requires retained process capture")
        with (
            self.process_capture.completed_sweep(),
            exclusive(self.runtime_root / "locks" / "responsibility-backfill.lock", blocking=False) as acquired,
        ):
            if not acquired:
                raise Conflict("evidence capture is busy")
            _ledger, state = self._load()
            if self._needs_new_sweep(state) or state["lane"] != self.total_lanes or state["pending"]:
                raise Unavailable("evidence capture is incomplete")
            yield state

    def _take_locked(self, limit: int) -> BackfillProgress:
        ledger, state = self._load()
        if self._needs_new_sweep(state) and not state["pending"]:
            self._restart_sweep(state)
        cursor_key = (state["capture_id"], state["lane"])
        if self._cursor_key != cursor_key:
            self.close()
            self._cursor_key = cursor_key
        visited = processed = 0
        if state["lane"] < self.total_lanes:
            source_index, lane = divmod(state["lane"], len(LANES))
            source_root = self.sources[source_index]
            name = LANES[lane]
            directory = local_paths(source_root)[name]
            if not state["pending"]:
                if self._entries is None:
                    self._entries = iter_evidence_entries(directory, recursive=True)
                for _ in range(limit):
                    try:
                        path = next(self._entries)
                    except StopIteration:
                        state["at_end"] = True
                        break
                    visited += 1
                    if path is not None:
                        relative = path.relative_to(directory)
                        _relative_record(name, str(relative))
                        state["pending"].append(str(relative))
                if state["pending"]:
                    # Never advance the durable batch past an uncaptured record.
                    self._save(state)
            for relative in state["pending"][:limit]:
                if self.process_capture is not None:
                    capture_local_record(
                        ledger,
                        self.runtime_root,
                        name,
                        Path(relative),
                        source_root=source_root if source_index else None,
                        should_require_record=True,
                    )
                else:
                    capture_local_record(ledger, self.runtime_root, name, Path(relative))
                processed += 1
            if processed:
                state["pending"] = state["pending"][processed:]
            if self._needs_new_sweep(state) and not state["pending"]:
                # A reboot cannot discard observations journaled before it.
                # Finish the old batch before restarting every evidence lane.
                self._restart_sweep(state)
            elif state["at_end"] and not state["pending"]:
                state["lane"] += 1
                state["at_end"] = False
                self.close()
                self._save(state)
            elif processed:
                self._save(state)
        return BackfillProgress(
            state["capture_id"], state["lane"], len(state["pending"]), visited, processed, self.total_lanes
        )
