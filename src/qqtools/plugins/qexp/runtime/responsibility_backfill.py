"""Restartable local evidence capture; a completed sweep is not activation proof."""

from __future__ import annotations

import stat
import uuid
from collections.abc import Generator, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

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
_MAX_EVIDENCE_RECORD_BYTES = 8 * 1024 * 1024


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


@dataclass(frozen=True, slots=True)
class CapturedEvidenceLocator:
    """Non-authorizing source identity; never contains commands or credentials."""

    source_root: Path
    lane: str
    relative: str
    identity: str
    task_id: str | None
    attempt_number: int | None

    @property
    def payload(self) -> dict:
        return {"task_id": self.task_id, "attempt_number": self.attempt_number}


def _relative_record(name: str, value: object) -> Path:
    if name not in RECORD_KEYS:
        raise ValueError(f"unknown evidence capture lane: {name}")
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


def _evidence_identity(name: str, relative: Path) -> str:
    relative = _relative_record(name, str(relative))
    identity = relative.parent.name if name == "termination_decisions" else relative.stem
    return validate_identifier(identity, "attempt_id")


def _canonical_capture_root(root: Path) -> Path:
    root = Path(root)
    if not root.is_absolute() or ".." in root.parts or "\x00" in str(root):
        raise ValueError("capture root must be a canonical absolute runtime path")
    return root


def read_capture_locator(
    source_root: Path,
    name: str,
    relative: Path,
    *,
    should_require_record: bool = False,
) -> CapturedEvidenceLocator | None:
    """Read one bounded source record without target locks or ledger effects.

    The caller owns source retention/cleanup exclusion and supplies its captured
    canonical root. A missing retained record is an error, not an empty result.
    """
    source_root = _canonical_capture_root(source_root)
    relative = _relative_record(name, str(relative))
    identity = _evidence_identity(name, relative)
    directory = local_paths(source_root)[name]
    path = directory / relative
    try:
        for parent in {directory, path.parent}:
            if not stat.S_ISDIR(parent.stat(follow_symlinks=False).st_mode):
                raise Unavailable(f"backfill evidence parent is not a directory: {parent}")
        if not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
            raise Unavailable(f"backfill evidence is not a regular file: {path}")
        value = read_json_limited(path, max_bytes=_MAX_EVIDENCE_RECORD_BYTES, record_type="capture_source_evidence")
    except FileNotFoundError:
        if should_require_record:
            raise Unavailable(f"retained backfill evidence disappeared: {path}") from None
        return None
    record = value.get(RECORD_KEYS[name])
    if not isinstance(record, dict) or record.get("attempt_id", identity) != identity:
        raise Unavailable(f"backfill evidence identity does not match its path: {path}")
    payload = recovery_locator(identity, record)
    return CapturedEvidenceLocator(
        source_root, name, str(relative), identity, payload["task_id"], payload["attempt_number"]
    )


def _apply_capture_locator_locked(ledger: Ledger, runtime_root: Path, captured: CapturedEvidenceLocator) -> None:
    """Apply only a validated locator while the target evidence-write guard is held."""
    if captured.source_root == runtime_root:
        ledger.capture_local(captured.identity, captured.payload)
    else:
        ledger.capture_source(captured.identity, captured.payload, captured.source_root)


def apply_capture_locator(ledger: Ledger, runtime_root: Path, captured: CapturedEvidenceLocator) -> None:
    """Apply a consumed source observation using target-local I/O only.

    Caller-owned retained capture supplies the exact source/binding identity.
    This neither reopens that source nor grants execution or cleanup authority.
    """
    runtime_root = _canonical_capture_root(runtime_root)
    if ledger.root.resolve() != responsibility_root(runtime_root).resolve():
        raise ValueError("captured locator ledger does not belong to this runtime")
    _canonical_capture_root(captured.source_root)
    identity = _evidence_identity(captured.lane, Path(captured.relative))
    if captured.identity != identity or recovery_locator(identity, captured.payload) != captured.payload:
        raise Unavailable("captured evidence locator identity is invalid")
    with evidence_write_guard(runtime_root, identity) as acquired:
        if not acquired:
            raise Conflict(f"backfill evidence is busy: {identity}")
        _apply_capture_locator_locked(ledger, runtime_root, captured)


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
    identity = _evidence_identity(name, relative)
    if source_root is not None and (not source_root.is_absolute() or source_root.resolve() == runtime_root.resolve()):
        raise ValueError("backfill source must be a different absolute runtime")
    with evidence_write_guard(runtime_root, identity) as acquired:
        if not acquired:
            raise Conflict(f"backfill evidence is busy: {identity}")
        captured = read_capture_locator(
            source_root or runtime_root, name, relative, should_require_record=should_require_record
        )
        if captured is not None:
            _apply_capture_locator_locked(ledger, runtime_root, captured)


def capture_local_writer(
    ledger: Ledger,
    runtime_root: Path,
    identity: str,
    payload: dict,
    writer: dict,
    *,
    source_root: Path | None = None,
) -> int:
    """Normalize an outside caller's source before the local capture transaction."""
    if source_root is not None and not source_root.is_absolute():
        raise ValueError("captured writer source must be absolute")
    source_root = source_root.resolve() if source_root is not None else None
    return apply_captured_writer(ledger, runtime_root, identity, payload, writer, source_root=source_root)


def apply_captured_writer(
    ledger: Ledger,
    runtime_root: Path,
    identity: str,
    payload: dict,
    writer: dict,
    *,
    source_root: Path | None = None,
) -> int:
    """Apply a retained writer using canonical captured roots and local I/O only.

    Caller-owned process discovery supplies the actual Linux identity. Capturing
    one writer neither proves a complete inventory nor authorizes its execution.
    """
    if ledger.root.resolve() != responsibility_root(runtime_root).resolve():
        raise ValueError("captured writer ledger does not belong to this runtime")
    validate_identifier(identity, "attempt_id")
    locator = recovery_locator(identity, payload)
    if source_root is not None and _canonical_capture_root(source_root) == runtime_root.resolve():
        raise ValueError("captured writer source must be a different absolute runtime")
    with evidence_write_guard(runtime_root, identity) as acquired:
        if not acquired:
            raise Conflict(f"captured writer evidence is busy: {identity}")
        if source_root is not None:
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
        if "source_cursor" in state:
            from .responsibility_source_scan import source_scan_cursor

            source_scan_cursor(state["source_cursor"])
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
        state.pop("source_cursor", None)
        self._save(state)

    def _save(self, state: dict) -> None:
        state["revision"] += 1
        atomic_replace(self.path, state)

    def take(
        self, limit: int = BATCH_SIZE, *, should_cross_lanes: bool = False, allow_source: bool = True
    ) -> BackfillProgress | None:
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
                    progress = self._take_locked(remaining, allow_source=allow_source)
                    visited += progress.entries_visited
                    processed += progress.records_processed
                    remaining -= max(1, progress.entries_visited, progress.records_processed)
                    if (
                        not should_cross_lanes
                        or progress.is_sweep_complete
                        or (not allow_source and progress.completed_lanes >= len(LANES))
                    ):
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

    def next_source_work(self) -> dict[str, Any] | None:
        """Recover the next source intent exclusively from target-local records."""
        if self.process_capture is None:
            raise Unavailable("source work requires retained process capture")
        with (
            self.process_capture.completed_sweep(),
            exclusive(self.runtime_root / "locks" / "responsibility-backfill.lock", blocking=False) as acquired,
        ):
            if not acquired:
                return None
            _ledger, state = self._load()
            if self._needs_new_sweep(state) and not state["pending"]:
                self._restart_sweep(state)
            return self._source_work(state)

    def _source_work(self, state: dict) -> dict[str, Any] | None:
        from .responsibility_source_scan import initial_source_scan_cursor

        if not len(LANES) <= state["lane"] < self.total_lanes:
            return None
        parameters = {
            "capture_id": state["writer_capture_id"],
            "backfill_id": state["capture_id"],
            "backfill_revision": state["revision"],
            "lane": LANES[state["lane"] % len(LANES)],
        }
        if state["pending"]:
            return {"operation": "read", "parameters": {**parameters, "relative": state["pending"][0]}}
        return {
            "operation": "scan",
            "parameters": {**parameters, "cursor": state.get("source_cursor", initial_source_scan_cursor())},
        }

    def apply_source_scan(self, parameters: Mapping[str, Any], scan: Mapping[str, Any] | None) -> bool:
        """Journal exact discovery before any source record can be consumed.

        None is a stale scan: discard only its advisory cursor and rescan, never
        a pending record or capture identity. Lost results simply repeat discovery.
        """
        from .responsibility_source_scan import source_scan_result

        if self.process_capture is None:
            raise Unavailable("source discovery requires retained process capture")
        with (
            self.process_capture.completed_sweep(),
            exclusive(self.runtime_root / "locks" / "responsibility-backfill.lock", blocking=False) as acquired,
        ):
            if not acquired:
                return False
            _ledger, state = self._load()
            if self._source_work(state) != {"operation": "scan", "parameters": dict(parameters)}:
                return False
            if self._needs_new_sweep(state):
                self._restart_sweep(state)
                return False
            if scan is None:
                state.pop("source_cursor", None)
            else:
                validated = source_scan_result(scan, parameters["lane"])
                state.update(
                    pending=validated["pending"], source_cursor=validated["cursor"], at_end=validated["at_end"]
                )
                if state["at_end"] and not state["pending"]:
                    state["lane"] += 1
                    state["at_end"] = False
                    state.pop("source_cursor", None)
            self._save(state)
            return True

    def apply_source_record(self, parameters: Mapping[str, Any], captured: CapturedEvidenceLocator) -> bool:
        """Apply and retire one journaled observation using local I/O only."""
        if self.process_capture is None:
            raise Unavailable("source observation requires retained process capture")
        with (
            self.process_capture.completed_sweep(),
            exclusive(self.runtime_root / "locks" / "responsibility-backfill.lock", blocking=False) as acquired,
        ):
            if not acquired:
                return False
            ledger, state = self._load()
            if self._source_work(state) != {"operation": "read", "parameters": dict(parameters)}:
                return False
            if (
                captured.source_root != self.process_capture.checkpoint.legacy_source
                or captured.lane != parameters["lane"]
                or captured.relative != parameters["relative"]
            ):
                raise Unavailable("source observation differs from its retained pending record")
            apply_capture_locator(ledger, self.runtime_root, captured)
            state["pending"] = state["pending"][1:]
            if not state["pending"]:
                if self._needs_new_sweep(state):
                    self._restart_sweep(state)
                    return True
                if state["at_end"]:
                    state["lane"] += 1
                    state["at_end"] = False
                    state.pop("source_cursor", None)
            self._save(state)
            return True

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

    def _take_locked(self, limit: int, *, allow_source: bool = True) -> BackfillProgress:
        ledger, state = self._load()
        if self._needs_new_sweep(state) and not state["pending"]:
            self._restart_sweep(state)
        cursor_key = (state["capture_id"], state["lane"])
        if self._cursor_key != cursor_key:
            self.close()
            self._cursor_key = cursor_key
        visited = processed = 0
        if not allow_source and state["lane"] >= len(LANES):
            return BackfillProgress(state["capture_id"], state["lane"], len(state["pending"]), 0, 0, self.total_lanes)
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
                state.pop("source_cursor", None)
                self.close()
                self._save(state)
            elif processed:
                self._save(state)
        return BackfillProgress(
            state["capture_id"], state["lane"], len(state["pending"]), visited, processed, self.total_lanes
        )
