"""Ordered Group activity projections and their durable update intent."""

from __future__ import annotations

import fcntl
import math
import os
import stat
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterator

from .observation.tree import IndexTree, TreeBatch
from .paths import shared_paths
from .records import validate_group_name
from .store import atomic_replace, check_mutation_fence, read_json_limited

_PARTITIONS = ("published", "working", "due")
_TIME_WIDTH = 20
_MAX_TIME_MS = 10**_TIME_WIDTH - 1
_ACTIVITY_WINDOW_MS = 7 * 24 * 60 * 60 * 1000
_CHECK_DELAY_MS = 60 * 1000
_MAX_RECORD_BYTES = 4096


class GroupWorkingSetUnavailable(RuntimeError):
    """Signal that the index cannot supply a complete, consistent projection."""


@dataclass(frozen=True, slots=True)
class _Activity:
    name: str
    revision: int
    deadline_ms: int
    next_check_ms: int | None
    is_visible: bool
    token: str | None

    @classmethod
    def from_record(cls, value: object) -> _Activity:
        if not isinstance(value, dict) or value.keys() != cls.__dataclass_fields__.keys():
            raise ValueError("Invalid Group working-set activity fields.")
        activity = cls(**value)
        _group_name(activity.name)
        _revision(activity.revision)
        _time_ms(activity.deadline_ms)
        if type(activity.is_visible) is not bool or activity.is_visible != (activity.next_check_ms is not None):
            raise ValueError("Invalid Group working-set visibility.")
        if activity.next_check_ms is not None and _time_ms(activity.next_check_ms) < activity.deadline_ms:
            raise ValueError("Group working-set check time precedes the activity deadline.")
        if activity.token is not None:
            _token(activity.token)
        return activity


def _group_name(value: str) -> str:
    name = validate_group_name(value)
    if name is None:
        raise ValueError("A Group name is required for the working-set index.")
    return name


def _revision(value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("Group activity revision must be a positive integer.")
    return value


def _time_ms(value: int) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_TIME_MS:
        raise ValueError("Group working-set time must be a nonnegative 20-digit millisecond value.")
    return value


def _clock_ms(now: float | None) -> int:
    seconds = time.time() if now is None else now
    if type(seconds) not in (int, float) or not 0 <= seconds <= _MAX_TIME_MS // 1000:
        raise ValueError("Group working-set clock must be finite nonnegative epoch seconds.")
    return _time_ms(math.floor(seconds * 1000))


def _token(value: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value.encode("utf-8")) <= 256:
        raise ValueError("Group activity token must be a nonempty string of at most 256 UTF-8 bytes.")
    return value


def _limit(value: int) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("Group working-set limit must be a positive integer.")
    return value


def _due_key(name: str, next_check_ms: int) -> str:
    return f"{next_check_ms:0{_TIME_WIDTH}d}-{name}"


def _parse_due_key(key: str) -> tuple[str, int]:
    timestamp, separator, name = key.partition("-")
    if not separator or len(timestamp) != _TIME_WIDTH or not timestamp.isascii() or not timestamp.isdecimal():
        raise ValueError("Invalid Group working-set due key.")
    return _group_name(name), _time_ms(int(timestamp))


def _directory(path: Path, *, create: bool = False) -> None:
    if create and not path.exists():
        check_mutation_fence(path)
        path.mkdir(exist_ok=True)
    if not stat.S_ISDIR(path.lstat().st_mode):
        raise ValueError(f"Group working-set path must be a real directory: {path}")


def _regular_file(path: Path, *, optional: bool = False) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        if optional:
            return False
        raise
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"Group working-set path must be a regular file: {path}")
    return True


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class _Index:
    def __init__(self, root: Path) -> None:
        self.indexes = shared_paths(root)["indexes"]
        self.base = self.indexes / "group-working-set"
        self.activities = self.base / "activities"
        self.intent = self.base / "redo.json"
        self.initialized = self.base / "initialized-v1.json"

    def tree(self, partition: str) -> IndexTree:
        return IndexTree(self.base, phase=partition)

    @contextmanager
    def lock(self, *, write: bool, initialize: bool = False, blocking: bool = True) -> Iterator[None]:
        try:
            _directory(self.indexes, create=initialize)
            _directory(self.base, create=initialize)
            path = self.base / "index.lock"
            flags = (os.O_RDWR if write else os.O_RDONLY) | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
            if initialize and not _regular_file(path, optional=True):
                check_mutation_fence(path)
                flags |= os.O_CREAT
            fd = os.open(path, flags, 0o600)
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise ValueError(f"Group working-set lock must be a regular file: {path}")
                operation = fcntl.LOCK_EX if write else fcntl.LOCK_SH
                if not blocking:
                    operation |= fcntl.LOCK_NB
                fcntl.flock(fd, operation)
                if not initialize:
                    self.require_initialized()
                if not write and _regular_file(self.intent, optional=True):
                    raise GroupWorkingSetUnavailable("Group working-set index has an unfinished update.")
                yield
            finally:
                os.close(fd)
        except (OSError, ValueError) as exc:
            raise GroupWorkingSetUnavailable(f"Group working-set index unavailable at {self.base}: {exc}") from exc

    def require_initialized(self) -> None:
        _regular_file(self.initialized)
        _directory(self.activities)
        _directory(self.base / "partitions")
        for partition in _PARTITIONS:
            root_path = self.tree(partition).root_path
            _directory(root_path.parent)
            _regular_file(root_path)

    def activity(self, name: str) -> _Activity | None:
        path = self.activities / f"{name}.json"
        if not _regular_file(path, optional=True):
            return None
        value = _Activity.from_record(read_json_limited(path, max_bytes=_MAX_RECORD_BYTES))
        if value.name != name:
            raise ValueError(f"Group working-set activity name does not match {path}.")
        return value

    def apply_intent(self, activity: _Activity, previous_check_ms: int | None, *, is_new: bool) -> None:
        modified = []
        if is_new:
            published = self.tree("published")
            published.add(activity.name)
            modified.append(published)
        if (previous_check_ms is not None) != activity.is_visible:
            working = self.tree("working")
            if activity.is_visible:
                working.add(activity.name)
            else:
                working.discard(activity.name)
            modified.append(working)
        if previous_check_ms != activity.next_check_ms:
            due = self.tree("due")
            if previous_check_ms is not None:
                due.discard(_due_key(activity.name, previous_check_ms))
            if activity.next_check_ms is not None:
                due.add(_due_key(activity.name, activity.next_check_ms))
            modified.append(due)
        path = self.activities / f"{activity.name}.json"
        _regular_file(path, optional=True)
        atomic_replace(path, asdict(activity))
        # A previous attempt could have crashed after replacing a tree node
        # but before flushing its directory. Replay can be a no-op in that tree.
        for tree in modified:
            _sync_directory(tree.root_path.parent)
        check_mutation_fence(self.intent)
        self.intent.unlink()
        _sync_directory(self.base)

    def recover(self) -> None:
        if not _regular_file(self.intent, optional=True):
            return
        value = read_json_limited(self.intent, max_bytes=_MAX_RECORD_BYTES)
        if value.keys() != {"activity", "previous_check_ms", "is_new"}:
            raise ValueError("Invalid Group working-set update intent fields.")
        activity = _Activity.from_record(value["activity"])
        previous_check_ms = value["previous_check_ms"]
        if previous_check_ms is not None:
            _time_ms(previous_check_ms)
        is_new = value["is_new"]
        if type(is_new) is not bool or (is_new and (previous_check_ms is not None or not activity.is_visible)):
            raise ValueError("Invalid Group working-set update intent transition.")
        # The intent may have become visible before its original directory fsync.
        _sync_directory(self.base)
        self.apply_intent(activity, previous_check_ms, is_new=is_new)

    def update(self, activity: _Activity, previous: _Activity | None) -> None:
        previous_check_ms = None if previous is None else previous.next_check_ms
        is_new = previous is None
        atomic_replace(
            self.intent, {"activity": asdict(activity), "previous_check_ms": previous_check_ms, "is_new": is_new}
        )
        self.apply_intent(activity, previous_check_ms, is_new=is_new)


def is_index_initialized(root: Path) -> bool:
    """Check the completion marker written after all initial roots are durable."""
    marker = _Index(root).initialized
    return _regular_file(marker, optional=True) and read_json_limited(marker, max_bytes=128) == {"version": 1}


def initialize_index(root: Path) -> None:
    """Durably create the three explicit empty roots for a new index."""
    index = _Index(root)
    with index.lock(write=True, initialize=True):
        if _regular_file(index.initialized, optional=True):
            index.require_initialized()
            _sync_directory(index.base)
            index.recover()
            return
        if _regular_file(index.intent, optional=True):
            raise GroupWorkingSetUnavailable("Group working-set index has an update before initialization.")
        _directory(index.activities, create=True)
        for partition in _PARTITIONS:
            tree = index.tree(partition)
            tree.ensure_root()
            _sync_directory(tree.root_path.parent)
        for directory in (index.base / "partitions", index.base, index.indexes, root):
            _sync_directory(directory)
        atomic_replace(index.initialized, {"version": 1})


def recover_index(root: Path, *, blocking: bool = True) -> None:
    """Replay an interrupted index update under the exclusive index lock."""
    index = _Index(root)
    with index.lock(write=True, blocking=blocking):
        index.recover()


def group_candidates(root: Path, after: str | None, *, all_groups: bool, limit: int) -> TreeBatch:
    """Read a bounded lexical page from the published or working partition."""
    if after is not None:
        _group_name(after)
    if type(all_groups) is not bool:
        raise ValueError("all_groups must be a Boolean.")
    _limit(limit)
    index = _Index(root)
    with index.lock(write=False):
        batch = index.tree("published" if all_groups else "working").candidates(after, max_candidates=limit)
        for name in batch.keys:
            _group_name(name)
        return batch


def _new_activity(name: str, token: str | None, now_ms: int, previous: _Activity | None) -> _Activity:
    deadline_ms = _time_ms(now_ms + _ACTIVITY_WINDOW_MS)
    if previous is not None:
        deadline_ms = max(deadline_ms, previous.deadline_ms)
    revision = 1 if previous is None else previous.revision + 1
    return _Activity(name, revision, deadline_ms, deadline_ms, True, token)


def enroll_group(root: Path, name: str, *, now: float) -> bool:
    """Seed a known published Group only when no activity for it has been recorded."""
    name = _group_name(name)
    now_ms = _clock_ms(now)
    index = _Index(root)
    with index.lock(write=True):
        index.recover()
        if index.activity(name) is not None:
            return False
        index.update(_new_activity(name, None, now_ms, None), None)
        return True


def refresh_group(root: Path, name: str, token: str, *, now: float | None = None, blocking: bool = True) -> bool:
    """Renew the activity window for a new token under the caller's Group writer lock."""
    name = _group_name(name)
    token = _token(token)
    now_ms = _clock_ms(now)
    index = _Index(root)
    with index.lock(write=True, blocking=blocking):
        index.recover()
        previous = index.activity(name)
        if previous is not None and previous.token == token:
            return False
        index.update(_new_activity(name, token, now_ms, previous), previous)
        return True


def due_candidates(root: Path, *, now: float, limit: int, blocking: bool = True) -> list[tuple[str, int, int]]:
    """Read the earliest due Group names, activity revisions, and millisecond times."""
    now_ms = _clock_ms(now)
    _limit(limit)
    index = _Index(root)
    with index.lock(write=False, blocking=blocking):
        batch = index.tree("due").candidates(None, max_candidates=limit)
        candidates = []
        for key in batch.keys:
            name, next_check_ms = _parse_due_key(key)
            if next_check_ms > now_ms:
                break
            activity = index.activity(name)
            if activity is None or activity.next_check_ms != next_check_ms:
                raise GroupWorkingSetUnavailable(f"Group working-set due key disagrees with activity for {name!r}.")
            candidates.append((name, activity.revision, next_check_ms))
        return candidates


def next_check_time(root: Path, *, blocking: bool = True) -> float | None:
    """Return the earliest scheduled check in wall-clock seconds."""
    index = _Index(root)
    with index.lock(write=False, blocking=blocking):
        batch = index.tree("due").candidates(None, max_candidates=1)
        if not batch.keys:
            return None
        _, next_check_ms = _parse_due_key(batch.keys[0])
        return next_check_ms / 1000


def finish_check(
    root: Path,
    name: str,
    revision: int,
    next_check_ms: int,
    *,
    now: float,
    retire: bool,
    blocking: bool = True,
) -> bool:
    """Retire an unchanged due check or defer it for 60 seconds.

    Retirement requires the caller's Group writer lock. Deferral changes only
    the next check time and can use the index lock alone if the Group is busy.
    """
    name = _group_name(name)
    _revision(revision)
    _time_ms(next_check_ms)
    now_ms = _clock_ms(now)
    if type(retire) is not bool:
        raise ValueError("retire must be a Boolean.")
    index = _Index(root)
    with index.lock(write=True, blocking=blocking):
        index.recover()
        previous = index.activity(name)
        if previous is None or (previous.revision, previous.next_check_ms) != (revision, next_check_ms):
            return False
        if next_check_ms > now_ms:
            return False
        deferred = None if retire else _time_ms(now_ms + _CHECK_DELAY_MS)
        index.update(replace(previous, next_check_ms=deferred, is_visible=not retire), previous)
        return True
