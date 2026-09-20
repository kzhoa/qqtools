"""Bounded maintenance for the canonical Group discovery projections.

Maintenance is deliberately separate from discovery.  It only retires durable
projection data after the corresponding Group or source proof has been checked;
it never scans Task, Submission, or archived operation history and never applies
Task effects.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator

from ..group_namespace import group_authority_identity, is_group_authority_isolated
from ..locks import exclusive, group_writer_lock
from ..paths import shared_paths, submission_path
from ..records import validate_group_name, validate_identifier
from ..store import atomic_replace, read_json, read_json_limited, require_json_size
from .coverage import GroupCoverage
from .rechecks import GroupRechecks, RecheckPosition
from .recovery import _decode_receipt
from .source_revision import SourceRevision

_MAX_RECORD_BYTES = 64 * 1024
_MAX_RECEIPT_BYTES = 16 * 1024
_LANE_IDLE_SECONDS = 1.0
_TREE_MAX_DEPTH = 8
_STATUS_VERSION = 1
_REVISION_KEYS = frozenset({"device", "inode", "size", "mtime_ns", "ctime_ns"})
_PENDING_STATES = frozenset({"preparing", "converging", "waiting_ack", "blocked"})
_CANCEL_TERMINAL_STATES = frozenset({"completed"})
_WORKER_TERMINAL_STATES = frozenset({"completed", "superseded"})
_SOURCE_TARGETS = (
    "qualification/task.digests",
    "qualification/sequence.digests",
    "qualification/task-audit",
    "qualification/sequence-audit",
)


class _Missing:
    __slots__ = ()


_MISSING = _Missing()


@dataclass(frozen=True, slots=True)
class _DirectoryRevision:
    """The directory stamp used to certify one active-record pass."""

    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True, slots=True)
class _LaneResult:
    state: str
    reason: str | None = None
    deleted: bool = False
    diagnostic: bool = False


@dataclass(slots=True)
class _TreeFrame:
    path: Path
    iterator: os.ScandirIterator | None = None
    depth: int = 0
    identity: tuple[int, int] | None = None


class _DirectoryScanner:
    """Keep one directory iterator and consume no more than one entry at a time."""

    __slots__ = ("path", "_iterator", "_opened", "_missing", "_closed")

    def __init__(self, path: Path):
        self.path = path
        self._iterator: os.ScandirIterator | None = None
        self._opened = False
        self._missing = False
        self._closed = False

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def missing(self) -> bool:
        return self._missing

    def next_entry(self) -> os.DirEntry[str] | None | _Missing:
        """Return one entry, EOF, or an absent-directory sentinel."""

        if self._closed:
            return None
        if self._iterator is None and not self._opened:
            self._opened = True
            try:
                info = self.path.lstat()
            except FileNotFoundError:
                self._missing = True
                self.close()
                return _MISSING
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"maintenance directory is not a real directory: {self.path}")
            self._iterator = os.scandir(self.path)
        iterator = self._iterator
        if iterator is None:
            return None
        try:
            return next(iterator)
        except StopIteration:
            self.close()
            return None

    def close(self) -> None:
        if self._closed:
            return
        iterator = self._iterator
        self._iterator = None
        self._closed = True
        if iterator is not None:
            iterator.close()


class _TreeCleaner:
    """Depth-bounded DFS that deletes at most one object per step."""

    __slots__ = ("root", "remove_root", "_stack", "_initialized", "_done", "_closed")

    def __init__(self, root: Path, *, remove_root: bool):
        self.root = root
        self.remove_root = remove_root
        self._stack: list[_TreeFrame] = []
        self._initialized = False
        self._done = False
        self._closed = False

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def is_done(self) -> bool:
        return self._done

    def step(self) -> tuple[str, bool]:
        """Return ``(state, deleted)`` after one bounded cleanup action."""

        if self._closed:
            return "closed", False
        if self._done:
            return "complete", False
        if not self._initialized:
            self._initialized = True
            try:
                info = self.root.lstat()
            except FileNotFoundError:
                self._done = True
                return "complete", False
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"certified cleanup root is not a directory: {self.root}")
            self._stack.append(_TreeFrame(self.root, depth=0))

        while self._stack:
            frame = self._stack[-1]
            for ancestor in self._stack:
                info = ancestor.path.lstat()
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                    raise ValueError(f"cleanup directory was replaced: {ancestor.path}")
                if ancestor.identity is not None and ancestor.identity != (info.st_dev, info.st_ino):
                    raise ValueError(f"cleanup directory identity changed: {ancestor.path}")
                ancestor.identity = (info.st_dev, info.st_ino)
            if frame.iterator is None:
                try:
                    info = frame.path.lstat()
                except FileNotFoundError:
                    self._stack.pop()
                    continue
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                    raise ValueError(f"cleanup directory was replaced: {frame.path}")
                frame.iterator = os.scandir(frame.path)

            try:
                entry = next(frame.iterator)
            except StopIteration:
                iterator = frame.iterator
                frame.iterator = None
                if iterator is not None:
                    iterator.close()
                self._stack.pop()
                if frame.path == self.root:
                    if self.remove_root:
                        _remove_directory(frame.path)
                        self._done = True
                        return "progressed", True
                    self._done = True
                    return "complete", False
                _remove_directory(frame.path)
                return "progressed", True

            name = _entry_name(entry)
            child = frame.path / name
            try:
                info = child.lstat()
            except FileNotFoundError:
                return "progressed", False
            if stat.S_ISLNK(info.st_mode):
                _remove_file(child)
                return "progressed", True
            if stat.S_ISDIR(info.st_mode):
                depth = frame.depth + 1
                if depth > _TREE_MAX_DEPTH:
                    raise ValueError(f"cleanup tree exceeds depth {_TREE_MAX_DEPTH}: {child}")
                self._stack.append(_TreeFrame(child, depth=depth))
                return "progressed", False
            _remove_file(child)
            return "progressed", True

        self._done = True
        return "complete", False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for frame in self._stack:
            iterator = frame.iterator
            frame.iterator = None
            if iterator is not None:
                iterator.close()
        self._stack.clear()


@dataclass(slots=True)
class _JournalPass:
    position: RecheckPosition
    active_path: Path
    active_revision: _DirectoryRevision | None
    scanner: _DirectoryScanner | None
    minimum: int
    names: set[str] = field(default_factory=set)
    count: int = 0
    certified: bool = False


@dataclass(slots=True)
class _ReceiptJob:
    operation_id: str
    root: Path
    cleaner: _TreeCleaner


@dataclass(slots=True)
class _SourceJob:
    debt_path: Path
    operation_id: str
    revision: SourceRevision
    scratch: Path
    target_index: int = 0
    final_check_index: int = 0
    cleaner: _TreeCleaner | None = None


class GroupMaintenance:
    """Advance bounded maintenance for one canonical Group.

    Construction performs validation only.  The first :meth:`advance` binds
    the Group authority identity and opens any persistent iterators.
    """

    def __init__(self, root: Path, group: str) -> None:
        if not isinstance(root, (Path, str)):
            raise TypeError("root must be a path")
        if isinstance(root, str) and not root:
            raise ValueError("root must not be empty")
        if not isinstance(group, str):
            raise TypeError("group must be a string")
        validate_group_name(group)
        self._root = Path(os.path.abspath(os.fspath(root)))
        self._group = group
        self._cfg = SimpleNamespace(shared_root=self._root)
        self._coverage = GroupCoverage(self._root, group)
        self._rechecks = GroupRechecks(self._root, group)
        self._identity: dict[str, Any] | None = None
        self._coverage_directory: Path | None = None
        self._closed = False
        self._close_requested = False
        self._next_lane = 0
        self._cooldown = [0.0, 0.0, 0.0, 0.0]
        self._journal_pass: _JournalPass | None = None
        self._receipt_scan: _DirectoryScanner | None = None
        self._receipt_job: _ReceiptJob | None = None
        self._source_scan: _DirectoryScanner | None = None
        self._source_job: _SourceJob | None = None
        self._generation_scan: _DirectoryScanner | None = None
        self._generation_job: _TreeCleaner | None = None
        self._generation_number: int | None = None
        self._generation_root: Path | None = None
        self._counters: dict[str, dict[str, int]] = {
            "journal": {"steps": 0, "deleted": 0},
            "receipts": {"steps": 0, "deleted": 0},
            "sources": {"steps": 0, "deleted": 0},
            "generations": {"steps": 0, "deleted": 0},
        }
        self._last_diagnostic: tuple[str, str, str] | None = None
        self._last_errors: dict[str, str] = {}
        self._capture_complete: bool | None = None
        self._capture_turn = True
        self._capture_scan: _DirectoryScanner | None = None
        self._capture_operation_scan: _DirectoryScanner | None = None
        self._capture_root_revision: _DirectoryRevision | None = None
        self._capture_operation_revision: _DirectoryRevision | None = None
        self._capture_changed = False

    @property
    def is_closed(self) -> bool:
        """Whether all maintenance-owned iterators have been released."""

        return self._closed

    def request_close(self) -> None:
        """Request cooperative iterator cleanup."""

        if self._closed:
            return
        self._close_requested = True
        if not self._has_open_resources():
            self._closed = True

    def advance(self) -> dict[str, object]:
        """Perform one bounded step from the next round-robin lane."""

        if self._closed:
            return self._result("closed", None, "closed")
        if self._close_requested:
            self._close_resources()
            self._closed = True
            return self._result("closed", None, "closed")

        lane_index = self._next_lane
        self._next_lane = (self._next_lane + 1) % 4
        lane = ("journal", "receipts", "sources", "generations")[lane_index]
        self._counters[lane]["steps"] += 1
        if time.monotonic() < self._cooldown[lane_index]:
            return self._result("waiting", "cooldown", lane)
        try:
            if lane_index == 0:
                outcome = self._advance_journal()
            elif lane_index == 1:
                outcome = self._advance_receipts()
            elif lane_index == 2:
                outcome = self._advance_sources()
            else:
                outcome = self._advance_generations()
        except Exception as exc:
            self._abandon_failed_job(lane_index)
            self._set_cooldown(lane_index)
            outcome = _LaneResult("error", f"{type(exc).__name__}: {exc}", diagnostic=True)

        if outcome.deleted:
            self._counters[lane]["deleted"] += 1
        if outcome.diagnostic:
            persisted = self._persist_diagnostic("error", outcome.reason or "maintenance_error", lane)
            if persisted is not None:
                outcome = _LaneResult("error", persisted, deleted=outcome.deleted, diagnostic=False)
        elif outcome.deleted:
            persisted = self._persist_diagnostic("progressed", outcome.reason or "deleted", lane)
            if persisted is not None:
                outcome = _LaneResult("error", persisted, deleted=True, diagnostic=False)
        return self._result(outcome.state, outcome.reason, lane)

    def _advance_journal(self) -> _LaneResult:
        if not self._canonical_ready():
            return _LaneResult("waiting", "canonical_namespace_required")
        if self._journal_pass is None:
            result = self._begin_journal_pass()
            if result is not None:
                return result
        current = self._journal_pass
        if current is None:
            return _LaneResult("waiting", "journal_idle")
        if current.certified:
            return self._reclaim_journal(current)

        scanner = current.scanner
        if scanner is None:
            current.certified = True
            return _LaneResult("progressed", "journal_pass_certified")

        with self._writer_lock() as acquired:
            if not acquired:
                return _LaneResult("waiting", "group_writer_busy")
            entry = scanner.next_entry()
            if entry is _MISSING:
                self._journal_pass = None
                self._set_cooldown(0)
                return _LaneResult("waiting", "journal_directory_missing")
            if entry is None:
                if not self._journal_pass_is_stable(current):
                    self._discard_journal_pass()
                    return _LaneResult("waiting", "journal_pass_restarted")
                current.scanner = None
                current.certified = True
                return _LaneResult("progressed", "journal_pass_certified")
            try:
                self._inspect_active_record(current, entry)
            except Exception as exc:
                self._discard_journal_pass()
                self._set_cooldown(0)
                return _LaneResult("error", f"journal_active_record:{type(exc).__name__}: {exc}", diagnostic=True)
            return _LaneResult("progressed", "journal_entry")

    def _begin_journal_pass(self) -> _LaneResult | None:
        with self._writer_lock() as acquired:
            if not acquired:
                return _LaneResult("waiting", "group_writer_busy")
            position = self._rechecks.snapshot(initialize=False)
            if position is None:
                self._set_cooldown(0)
                return _LaneResult("waiting", "journal_idle")
            active_path = shared_paths(self._root)["group_control_active"]
            # Once a journal exists, an absent active namespace is not proof
            # that no subscriber remains. Flush its namespace before certifying
            # absences left by an interrupted archival unlink.
            _directory_revision(active_path, allow_missing=False)
            _sync_directory(active_path)
            active_revision = _directory_revision(active_path, allow_missing=False)
            scanner = _DirectoryScanner(active_path)
            self._journal_pass = _JournalPass(position, active_path, active_revision, scanner, position.tail)
            return None

    def _inspect_active_record(self, current: _JournalPass, entry: os.DirEntry[str]) -> None:
        name = _entry_name(entry)
        if name in current.names:
            raise ValueError("active operation directory yielded a duplicate entry")
        current.names.add(name)
        current.count += 1
        if not name.endswith(".json"):
            return
        operation_id = name[:-5]
        validate_identifier(operation_id, "active operation_id")
        path = current.active_path / name
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ValueError("active Group control record is not a regular file")
        value = read_json(path)
        if type(value) is not dict or type(value.get("group_control")) is not dict:
            raise ValueError("active Group control record is malformed")
        control = value["group_control"]
        if control.get("operation_id") != operation_id:
            raise ValueError("active Group control operation identity does not match its filename")
        group_name = control.get("group_name")
        if not isinstance(group_name, str) or not group_name:
            raise ValueError("active Group control has no Group identity")
        validate_group_name(group_name)
        operation_type = control.get("operation_type")
        state = control.get("state")
        if group_name != self._group:
            return
        if operation_type == "worker_remove":
            if state == "completed" or control.get("blocked_reason") == "legacy_worker_incarnation_unknown":
                return
            current.minimum = 0
            return
        if operation_type == "cancel":
            terminal = _CANCEL_TERMINAL_STATES
        elif operation_type == "worker_remove_v2":
            terminal = _WORKER_TERMINAL_STATES
        else:
            raise ValueError("unknown Group control operation type")
        if state in terminal:
            return
        if state not in _PENDING_STATES:
            raise ValueError("target Group control state is invalid")
        discovery = control.get("discovery")
        if type(discovery) is not dict:
            current.minimum = 0
            return
        authority = discovery.get("authority")
        generation = discovery.get("generation")
        cursor = discovery.get("journal_cursor")
        if authority != self._require_identity_without_group() or type(generation) is not int or generation < 1:
            current.minimum = 0
            return
        if type(cursor) is not int or cursor < 0:
            current.minimum = 0
            return
        if generation != current.position.generation:
            current.minimum = 0
            return
        position = self._rechecks.snapshot()
        if position is None or generation != position.generation or cursor > position.tail:
            raise ValueError("active Group control journal cursor exceeds current tail")
        current.minimum = min(current.minimum, cursor)

    def _journal_pass_is_stable(self, current: _JournalPass) -> bool:
        revision = _directory_revision(current.active_path, allow_missing=True)
        if revision != current.active_revision:
            return False
        latest = self._rechecks.snapshot(initialize=False)
        return (
            latest is not None
            and latest.generation == current.position.generation
            and current.count == len(current.names)
        )

    def _reclaim_journal(self, current: _JournalPass) -> _LaneResult:
        if current.minimum < 0:
            raise ValueError("journal retirement bound is invalid")
        with self._writer_lock() as acquired:
            if not acquired:
                return _LaneResult("waiting", "group_writer_busy")
            position = self._rechecks.snapshot(initialize=False)
            if position is None:
                self._discard_journal_pass()
                return _LaneResult("waiting", "journal_disappeared")
            if position.generation != current.position.generation:
                self._discard_journal_pass()
                return _LaneResult("waiting", "journal_generation_changed")
            reclaimed = self._rechecks.reclaim(position, through=current.minimum, max_events=1)
            retention = self._rechecks.retention(position)
            if retention.deleted >= current.minimum:
                self._discard_journal_pass()
                self._set_cooldown(0)
                if reclaimed:
                    return _LaneResult("progressed", "journal_event_deleted", deleted=True)
                return _LaneResult("waiting", "journal_retention_reached")
            if reclaimed:
                return _LaneResult("progressed", "journal_event_deleted", deleted=True)
            # An in-flight event is a live responsibility.  Restart the census
            # later so a consumer can publish its resolution without blocking
            # the other maintenance lanes.
            self._discard_journal_pass()
            self._set_cooldown(0)
            return _LaneResult("waiting", "journal_event_in_flight")

    def _advance_receipts(self) -> _LaneResult:
        if not self._canonical_ready():
            return _LaneResult("waiting", "canonical_namespace_required")
        job = self._receipt_job
        if job is not None:
            with self._writer_lock() as acquired:
                if not acquired:
                    return _LaneResult("waiting", "group_writer_busy")
                valid, reason = self._completed_cancel_proof(job.operation_id)
                if not valid:
                    job.cleaner.close()
                    self._receipt_job = None
                    return _LaneResult("waiting", reason or "receipt_proof_missing")
                _sync_directory(shared_paths(self._root)["group_control_active"])
                _check_directory_path(job.root.parent, self._coverage_directory_required())
                state, deleted = job.cleaner.step()
                if state == "complete":
                    self._receipt_job = None
                    return _LaneResult("waiting", "receipt_tree_complete")
                return _LaneResult(
                    "progressed", "receipt_object_deleted" if deleted else "receipt_entry", deleted=deleted
                )

        if self._receipt_scan is None:
            directory = self._coverage_directory_required() / "operations"
            if _directory_revision(directory, allow_missing=True) is None:
                self._set_cooldown(1)
                return _LaneResult("waiting", "receipt_idle")
            self._receipt_scan = _DirectoryScanner(directory)
        with self._writer_lock() as acquired:
            if not acquired:
                return _LaneResult("waiting", "group_writer_busy")
            entry = self._receipt_scan.next_entry()
            if entry is _MISSING:
                self._receipt_scan = None
                self._set_cooldown(1)
                return _LaneResult("waiting", "receipt_directory_missing")
            if entry is None:
                self._receipt_scan = None
                self._set_cooldown(1)
                return _LaneResult("waiting", "receipt_scan_complete")
            name = _entry_name(entry)
            try:
                validate_identifier(name, "coverage operation id")
            except (TypeError, ValueError):
                return _LaneResult("waiting", "receipt_entry_retained")
            operation_root = self._coverage_directory_required() / "operations" / name
            info = operation_root.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                return _LaneResult("error", "receipt_operation_tree_not_directory", diagnostic=True)
            valid, reason = self._completed_cancel_proof(name)
            if not valid:
                return _LaneResult("waiting", reason or "receipt_proof_missing")
            _sync_directory(shared_paths(self._root)["group_control_active"])
            self._receipt_job = _ReceiptJob(name, operation_root, _TreeCleaner(operation_root, remove_root=True))
            return _LaneResult("progressed", "receipt_tree_selected")

    def _completed_cancel_proof(self, operation_id: str) -> tuple[bool, str | None]:
        paths = shared_paths(self._root)
        active_directory = paths["group_control_active"]
        try:
            active_info = active_directory.lstat()
        except FileNotFoundError:
            return False, "group_control_active_missing"
        if stat.S_ISLNK(active_info.st_mode) or not stat.S_ISDIR(active_info.st_mode):
            raise ValueError("Group control active namespace is not a directory")
        active_path = active_directory / f"{operation_id}.json"
        try:
            active_path.lstat()
        except FileNotFoundError:
            pass
        else:
            return False, "receipt_operation_still_active"
        archive = paths["group_control"] / f"{operation_id}.json"
        info = archive.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            return False, "receipt_archive_not_regular"
        value = read_json(archive)
        control = value.get("group_control") if type(value) is dict else None
        if type(control) is not dict:
            return False, "receipt_archive_malformed"
        if (
            control.get("operation_id") != operation_id
            or control.get("group_name") != self._group
            or control.get("operation_type") != "cancel"
            or control.get("state") != "completed"
        ):
            return False, "receipt_archive_not_completed_cancel"
        return True, None

    def _advance_sources(self) -> _LaneResult:
        if not self._canonical_ready():
            return _LaneResult("waiting", "canonical_namespace_required")
        if self._capture_complete is not True:
            self._capture_turn = not self._capture_turn
            if self._capture_turn:
                try:
                    return self._advance_source_capture()
                except Exception:
                    self._close_source_capture()
                    raise
        job = self._source_job
        if job is not None:
            _check_directory_path(job.scratch, self._coverage_directory_required())
            lock_path = job.scratch.with_name(job.scratch.name + ".lock")
            if lock_path.is_symlink():
                raise ValueError("source cleanup lock must not be a symlink")
            with exclusive(lock_path, blocking=False) as acquired:
                if not acquired:
                    return _LaneResult("waiting", "source_scratch_busy")
                self._read_source_receipt(job)
                return self._advance_source_job(job)

        if self._source_scan is None:
            directory = self._coverage_directory_required() / "maintenance" / "sources"
            if _directory_revision(directory, allow_missing=True) is None:
                self._set_cooldown(2)
                return _LaneResult("waiting", "source_cleanup_idle")
            self._source_scan = _DirectoryScanner(directory)
        entry = self._source_scan.next_entry()
        if entry is _MISSING:
            self._source_scan = None
            self._set_cooldown(2)
            return _LaneResult("waiting", "source_cleanup_directory_missing")
        if entry is None:
            self._source_scan = None
            self._set_cooldown(2)
            return _LaneResult("waiting", "source_cleanup_scan_complete")
        try:
            job = self._source_job_from_entry(entry)
        except Exception as exc:
            return _LaneResult("error", f"source_cleanup_record:{type(exc).__name__}: {exc}", diagnostic=True)
        self._source_job = job
        return _LaneResult("progressed", "source_cleanup_selected")

    def _advance_source_capture(self) -> _LaneResult:
        """Capture pre-maintenance audit debt once, outside normal discovery."""

        directory = self._coverage_directory_required()
        marker = directory / "maintenance" / "source-capture.json"
        identity = {**self._require_identity_without_group(), "group": self._group}
        if self._capture_complete is None:
            try:
                value = json.loads(_read_bounded_file(marker, _MAX_RECORD_BYTES))
            except FileNotFoundError:
                self._capture_complete = False
            else:
                if value != {"version": 1, "identity": identity, "complete": True}:
                    raise ValueError("source cleanup capture record is invalid")
                self._capture_complete = True
                return _LaneResult("waiting", "source_capture_already_complete")

        sources = directory / "sources"
        if self._capture_scan is None:
            self._capture_root_revision = _directory_revision(sources, allow_missing=True)
            self._capture_scan = _DirectoryScanner(sources)
            self._capture_changed = False
            return _LaneResult("progressed", "source_capture_started")

        operation_scan = self._capture_operation_scan
        if operation_scan is not None:
            entry = operation_scan.next_entry()
            if entry is None or entry is _MISSING:
                if _directory_revision(operation_scan.path, allow_missing=True) != self._capture_operation_revision:
                    self._capture_changed = True
                self._capture_operation_scan = None
                return _LaneResult("progressed", "source_capture_operation_complete")
            scratch = operation_scan.path / _entry_name(entry)
            if not _is_digest_name(scratch.name):
                return _LaneResult("progressed", "source_capture_nonrevision_entry")
            try:
                _directory_revision(scratch, allow_missing=False)
                payload = _read_bounded_file(scratch / "receipt.json", _MAX_RECEIPT_BYTES)
            except FileNotFoundError:
                # An unfinished source still has its active locator. Its target
                # writer will publish cleanup debt before retiring that locator.
                return _LaneResult("progressed", "source_capture_unconfirmed")
            except Exception as exc:
                return _LaneResult("error", f"source_capture_path:{type(exc).__name__}: {exc}", diagnostic=True)
            try:
                raw = json.loads(payload)
                operation_id = raw.get("operation_id")
                validate_identifier(operation_id, "source capture operation_id")
                revision = _revision_from_value(raw.get("source_revision"))
                if scratch != self._coverage.source_scratch(operation_id, revision=revision):
                    raise ValueError("source capture receipt does not match its scratch path")
                receipt = _decode_receipt(
                    payload, submission_path(self._root, operation_id), operation_id, self._group, scratch
                )
                if receipt["status"] not in {"qualified", "irrelevant"}:
                    return _LaneResult("waiting", "source_capture_ambiguous_retained")
                enqueue_source_cleanup(self._root, self._group, operation_id, revision)
            except Exception as exc:
                return _LaneResult("error", f"source_capture_receipt:{type(exc).__name__}: {exc}", diagnostic=True)
            return _LaneResult("progressed", "source_cleanup_captured")

        entry = self._capture_scan.next_entry()
        if entry is not None and entry is not _MISSING:
            operation = sources / _entry_name(entry)
            if not _is_digest_name(operation.name):
                return _LaneResult("progressed", "source_capture_nonoperation_entry")
            try:
                self._capture_operation_revision = _directory_revision(operation, allow_missing=False)
            except Exception as exc:
                return _LaneResult("error", f"source_capture_operation:{type(exc).__name__}: {exc}", diagnostic=True)
            self._capture_operation_scan = _DirectoryScanner(operation)
            return _LaneResult("progressed", "source_capture_operation_selected")

        if self._capture_changed or _directory_revision(sources, allow_missing=True) != self._capture_root_revision:
            self._close_source_capture()
            return _LaneResult("waiting", "source_capture_namespace_changed")
        _ensure_directory_chain(marker.parent, directory)
        atomic_replace(marker, {"version": 1, "identity": identity, "complete": True})
        self._close_source_capture()
        self._capture_complete = True
        return _LaneResult("progressed", "source_capture_complete")

    def _close_source_capture(self) -> None:
        for scanner in (self._capture_operation_scan, self._capture_scan):
            if scanner is not None:
                scanner.close()
        self._capture_operation_scan = None
        self._capture_scan = None

    def _source_job_from_entry(self, entry: os.DirEntry[str]) -> _SourceJob:
        name = _entry_name(entry)
        if not name.endswith(".json"):
            raise ValueError("source cleanup entry is not a JSON debt record")
        path = self._coverage_directory_required() / "maintenance" / "sources" / name
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ValueError("source cleanup debt is not a regular file")
        value = read_json_limited(path, max_bytes=_MAX_RECORD_BYTES)
        if type(value) is not dict or set(value) != {"version", "group", "operation_id", "revision"}:
            raise ValueError("source cleanup debt record shape is invalid")
        if type(value.get("version")) is not int or value["version"] != 1 or value.get("group") != self._group:
            raise ValueError("source cleanup debt identity is invalid")
        operation_id = value.get("operation_id")
        validate_identifier(operation_id, "source cleanup operation_id")
        revision = _revision_from_value(value.get("revision"))
        expected = f"{_source_debt_digest(operation_id, revision)}.json"
        if name != expected:
            raise ValueError("source cleanup debt filename does not match its revision")
        scratch = self._coverage.source_scratch(operation_id, revision=revision)
        return _SourceJob(path, operation_id, revision, scratch)

    def _read_source_receipt(self, job: _SourceJob) -> dict[str, object]:
        path = job.scratch / "receipt.json"
        data = _read_bounded_file(path, _MAX_RECEIPT_BYTES)
        receipt = _decode_receipt(
            data,
            submission_path(self._root, job.operation_id),
            job.operation_id,
            self._group,
            job.scratch,
        )
        if receipt.get("status") not in {"qualified", "irrelevant"}:
            raise ValueError("source cleanup receipt is not terminal")
        revision = receipt.get("source_revision")
        if not isinstance(revision, SourceRevision) or revision != job.revision:
            raise ValueError("source cleanup receipt revision does not match debt")
        return receipt

    def _advance_source_job(self, job: _SourceJob) -> _LaneResult:
        qualification = job.scratch / "qualification"
        if qualification.exists() or qualification.is_symlink():
            _check_directory_path(qualification, self._coverage_directory_required())
        if job.cleaner is not None:
            state, deleted = job.cleaner.step()
            if state == "complete":
                job.cleaner = None
                job.target_index += 1
                return _LaneResult("progressed", "source_cleanup_tree_complete")
            return _LaneResult(
                "progressed", "source_cleanup_object_deleted" if deleted else "source_cleanup_entry", deleted=deleted
            )

        if job.target_index < len(_SOURCE_TARGETS):
            target = job.scratch / _SOURCE_TARGETS[job.target_index]
            try:
                info = target.lstat()
            except FileNotFoundError:
                job.target_index += 1
                return _LaneResult("progressed", "source_cleanup_target_absent")
            if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                job.cleaner = _TreeCleaner(target, remove_root=True)
                return _LaneResult("progressed", "source_cleanup_tree_selected")
            _remove_file(target)
            job.target_index += 1
            return _LaneResult("progressed", "source_cleanup_object_deleted", deleted=True)

        if job.final_check_index < len(_SOURCE_TARGETS):
            target = job.scratch / _SOURCE_TARGETS[job.final_check_index]
            job.final_check_index += 1
            try:
                target.lstat()
            except FileNotFoundError:
                return _LaneResult("progressed", "source_cleanup_recheck")
            job.target_index = job.final_check_index - 1
            job.final_check_index = 0
            job.cleaner = None
            return _LaneResult("waiting", "source_cleanup_target_reappeared")

        try:
            job.debt_path.lstat()
        except FileNotFoundError:
            self._source_job = None
            return _LaneResult("waiting", "source_cleanup_debt_missing")
        _remove_file(job.debt_path)
        self._source_job = None
        return _LaneResult("progressed", "source_cleanup_debt_deleted", deleted=True)

    def _advance_generations(self) -> _LaneResult:
        if not self._canonical_ready():
            return _LaneResult("waiting", "canonical_namespace_required")
        if self._generation_job is not None:
            with self._writer_lock() as acquired:
                if not acquired:
                    return _LaneResult("waiting", "group_writer_busy")
                position = self._rechecks.snapshot(initialize=False)
                if (
                    position is None
                    or self._generation_number is None
                    or position.generation != self._generation_number
                ):
                    self._reset_generation_lane()
                    return _LaneResult("waiting", "journal_generation_changed")
                _check_directory_path(self._generation_job.root.parent, self._coverage_directory_required())
                state, deleted = self._generation_job.step()
                if state == "complete":
                    self._reset_generation_lane(keep_scan=True)
                    return _LaneResult("waiting", "obsolete_generation_complete")
                return _LaneResult(
                    "progressed",
                    "obsolete_generation_object_deleted" if deleted else "obsolete_generation_entry",
                    deleted=deleted,
                )

        if self._generation_scan is None:
            with self._writer_lock() as acquired:
                if not acquired:
                    return _LaneResult("waiting", "group_writer_busy")
                position = self._rechecks.snapshot(initialize=False)
                if position is None:
                    self._set_cooldown(3)
                    return _LaneResult("waiting", "journal_idle")
                directory = self._coverage_directory_required() / "rechecks"
                if _directory_revision(directory, allow_missing=True) is None:
                    self._set_cooldown(3)
                    return _LaneResult("waiting", "rechecks_directory_missing")
                self._generation_number = position.generation
                self._generation_scan = _DirectoryScanner(directory)
        with self._writer_lock() as acquired:
            if not acquired:
                return _LaneResult("waiting", "group_writer_busy")
            position = self._rechecks.snapshot(initialize=False)
            if position is None or position.generation != self._generation_number:
                self._reset_generation_lane()
                return _LaneResult("waiting", "journal_generation_changed")
            entry = self._generation_scan.next_entry()
            if entry is _MISSING:
                self._reset_generation_lane()
                self._set_cooldown(3)
                return _LaneResult("waiting", "rechecks_directory_missing")
            if entry is None:
                self._generation_scan = None
                self._set_cooldown(3)
                return _LaneResult("waiting", "obsolete_generation_scan_complete")
            name = _entry_name(entry)
            if not name.isdecimal() or str(int(name)) != name:
                return _LaneResult("waiting", "rechecks_entry_retained")
            generation = int(name)
            # State publication is the authority boundary, not numeric order.
            # A crash-prepared next directory has no reachable ticket and can
            # be recreated by invalidate before its state is ever published.
            if generation == position.generation or generation <= 0:
                return _LaneResult("waiting", "rechecks_generation_retained")
            target = self._coverage_directory_required() / "rechecks" / name
            info = target.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                return _LaneResult("error", "obsolete_generation_not_directory", diagnostic=True)
            self._generation_number = position.generation
            self._generation_root = target
            self._generation_job = _TreeCleaner(target, remove_root=True)
            return _LaneResult("progressed", "obsolete_generation_selected")

    def _canonical_ready(self) -> bool:
        if not is_group_authority_isolated(self._root):
            return False
        directory = self._coverage.directory
        identity = group_authority_identity(self._root)
        if self._identity is None:
            self._identity = identity
            self._coverage_directory = directory
        elif identity != self._identity or directory != self._coverage_directory:
            raise RuntimeError("Group authority identity changed while maintenance was bound")
        return True

    def _coverage_directory_required(self) -> Path:
        if self._coverage_directory is None:
            self._canonical_ready()
        if self._coverage_directory is None:
            raise RuntimeError("Group coverage directory is unavailable")
        return self._coverage_directory

    def _require_identity_without_group(self) -> dict[str, Any]:
        identity = self._identity
        if type(identity) is not dict:
            raise RuntimeError("Group authority identity is unavailable")
        return {"project_id": identity["project_id"], "directory_identity": dict(identity["directory_identity"])}

    @contextmanager
    def _writer_lock(self) -> Iterator[bool]:
        with group_writer_lock(self._cfg, self._group, blocking=False) as acquired:
            yield acquired

    def _set_cooldown(self, lane: int) -> None:
        self._cooldown[lane] = time.monotonic() + _LANE_IDLE_SECONDS

    def _discard_journal_pass(self) -> None:
        current = self._journal_pass
        if current is not None and current.scanner is not None:
            current.scanner.close()
        self._journal_pass = None

    def _reset_generation_lane(self, *, keep_scan: bool = False) -> None:
        if self._generation_job is not None:
            self._generation_job.close()
        self._generation_job = None
        self._generation_root = None
        if not keep_scan and self._generation_scan is not None:
            self._generation_scan.close()
            self._generation_scan = None

    def _abandon_failed_job(self, lane: int) -> None:
        # Keep enumeration beyond a damaged job. Restarting the entire lane at
        # its first entry would starve later reclaimable records indefinitely.
        if lane == 0:
            self._discard_journal_pass()
        elif lane == 1:
            if self._receipt_job is not None:
                self._receipt_job.cleaner.close()
            self._receipt_job = None
        elif lane == 2:
            if self._source_job is not None and self._source_job.cleaner is not None:
                self._source_job.cleaner.close()
            self._source_job = None
        else:
            self._reset_generation_lane(keep_scan=True)

    def _reset_lane(self, lane: int) -> None:
        if lane == 0:
            self._discard_journal_pass()
        elif lane == 1:
            if self._receipt_scan is not None:
                self._receipt_scan.close()
            if self._receipt_job is not None:
                self._receipt_job.cleaner.close()
            self._receipt_scan = None
            self._receipt_job = None
        elif lane == 2:
            if self._source_scan is not None:
                self._source_scan.close()
            if self._source_job is not None and self._source_job.cleaner is not None:
                self._source_job.cleaner.close()
            self._source_scan = None
            self._source_job = None
        else:
            self._reset_generation_lane()

    def _has_open_resources(self) -> bool:
        return any(
            (
                self._journal_pass is not None and self._journal_pass.scanner is not None,
                self._receipt_scan is not None,
                self._receipt_job is not None,
                self._source_scan is not None,
                self._source_job is not None,
                self._generation_scan is not None,
                self._generation_job is not None,
                self._capture_scan is not None,
                self._capture_operation_scan is not None,
            )
        )

    def _close_resources(self) -> None:
        self._discard_journal_pass()
        self._reset_lane(1)
        self._reset_lane(2)
        self._reset_lane(3)
        self._close_source_capture()

    def _persist_diagnostic(self, state: str, reason: str, lane: str) -> str | None:
        reason = reason[:2048]
        key = (state, reason, lane)
        if state == "error":
            self._last_errors[lane] = reason
        if state == "error" and self._last_diagnostic == key:
            return None
        try:
            directory = self._coverage_directory_required()
            identity = self._identity
            if type(identity) is not dict:
                raise RuntimeError("Group authority identity is unavailable")
            record = {
                "version": _STATUS_VERSION,
                "identity": {**identity, "group": self._group},
                "state": state,
                "reason": reason,
                "lane": lane,
                "counter_kind": "service",
                "counters": self._counters,
                "last_errors": self._last_errors,
            }
            require_json_size(record, max_bytes=_MAX_RECORD_BYTES, record_type="group_maintenance_status")
            atomic_replace(directory / "maintenance" / "status.json", record)
        except Exception as exc:
            return f"diagnostic_persist:{type(exc).__name__}: {exc}"
        self._last_diagnostic = key
        return None

    def _result(self, state: str, reason: str | None, lane: str) -> dict[str, object]:
        return {
            "state": state,
            "reason": reason,
            "lane": lane,
            "counters": {name: dict(values) for name, values in self._counters.items()},
        }


def enqueue_source_cleanup(root: Path, group: str, operation_id: str, revision: SourceRevision) -> None:
    """Durably enqueue cleanup of one completed source's temporary audit data."""

    if not isinstance(root, (Path, str)):
        raise TypeError("root must be a path")
    root = Path(os.path.abspath(os.fspath(root)))
    validate_group_name(group)
    validate_identifier(operation_id, "operation_id")
    if not isinstance(revision, SourceRevision):
        raise TypeError("revision must be a SourceRevision")
    if not is_group_authority_isolated(root):
        return
    coverage = GroupCoverage(root, group)
    directory = coverage.directory
    record = {
        "version": 1,
        "group": group,
        "operation_id": operation_id,
        "revision": _revision_dict(revision),
    }
    require_json_size(record, max_bytes=_MAX_RECORD_BYTES, record_type="source_cleanup_debt")
    sources = directory / "maintenance" / "sources"
    _ensure_directory_chain(sources, directory)
    path = sources / f"{_source_debt_digest(operation_id, revision)}.json"
    atomic_replace(path, record)
    _sync_directory(directory / "maintenance")
    _sync_directory(directory)


def _entry_name(entry: os.DirEntry[str]) -> str:
    name = entry.name
    if not isinstance(name, str) or not name or name in {".", ".."} or "/" in name:
        raise ValueError("maintenance directory entry name is invalid")
    return name


def _check_directory_path(path: Path, ancestor: Path) -> None:
    relative = path.relative_to(ancestor)
    current = ancestor
    _directory_revision(current, allow_missing=False)
    for part in relative.parts:
        current = current / part
        _directory_revision(current, allow_missing=False)


def _is_digest_name(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _directory_revision(path: Path, *, allow_missing: bool) -> _DirectoryRevision | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        if allow_missing:
            return None
        raise
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"maintenance path is not a directory: {path}")
    return _DirectoryRevision(info.st_dev, info.st_ino, info.st_mtime_ns, info.st_ctime_ns)


def _sync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _sync_directory(path.parent)


def _remove_directory(path: Path) -> None:
    try:
        path.rmdir()
    except FileNotFoundError:
        return
    _sync_directory(path.parent)


def _revision_dict(revision: SourceRevision) -> dict[str, int]:
    return {key: getattr(revision, key) for key in ("device", "inode", "size", "mtime_ns", "ctime_ns")}


def _revision_from_value(value: object) -> SourceRevision:
    if type(value) is not dict or set(value) != _REVISION_KEYS:
        raise ValueError("source revision record is malformed")
    return SourceRevision(
        device=value["device"],
        inode=value["inode"],
        size=value["size"],
        mtime_ns=value["mtime_ns"],
        ctime_ns=value["ctime_ns"],
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _source_debt_digest(operation_id: str, revision: SourceRevision) -> str:
    payload = operation_id + ":" + _canonical_json(_revision_dict(revision)).decode("utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_bounded_file(path: Path, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    descriptor = os.open(path, flags)
    try:
        first = os.fstat(descriptor)
        if stat.S_ISLNK(first.st_mode) or not stat.S_ISREG(first.st_mode):
            raise ValueError(f"maintenance file is not regular: {path}")
        if first.st_size > max_bytes:
            raise ValueError(f"maintenance file exceeds {max_bytes} bytes: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError(f"maintenance file exceeds {max_bytes} bytes: {path}")
        second = os.fstat(descriptor)
        if (second.st_dev, second.st_ino, second.st_size) != (first.st_dev, first.st_ino, first.st_size):
            raise ValueError(f"maintenance file changed while reading: {path}")
        named = path.lstat()
        if stat.S_ISLNK(named.st_mode) or not stat.S_ISREG(named.st_mode):
            raise ValueError(f"maintenance file was replaced: {path}")
        if (named.st_dev, named.st_ino, named.st_size) != (first.st_dev, first.st_ino, first.st_size):
            raise ValueError(f"maintenance file was replaced: {path}")
        return data
    finally:
        os.close(descriptor)


def _ensure_directory_chain(path: Path, existing_ancestor: Path) -> None:
    missing: list[Path] = []
    candidate = path
    while candidate != existing_ancestor:
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            missing.append(candidate)
            candidate = candidate.parent
            continue
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"maintenance path is not a directory: {candidate}")
        break
    _directory_revision(candidate, allow_missing=False)
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            info = directory.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"maintenance path is not a directory: {directory}")
        _sync_directory(directory)
        _sync_directory(directory.parent)


__all__ = ["GroupMaintenance", "enqueue_source_cleanup"]
