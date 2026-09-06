"""Short-lived filesystem locks with the schema-defined lock order."""
from __future__ import annotations

import errno
import fcntl
from contextvars import ContextVar
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Iterator

from .paths import lock_path, shared_paths
from .store import read_json


GROUP_READY_MEMBERS_CAPABILITY = "group-ready-members-v1"
_schema_writer_roots: ContextVar[frozenset[Path]] = ContextVar(
    "qexp_schema_writer_roots", default=frozenset()
)


def is_schema_narrow_protocol_active(cfg: object) -> bool:
    """Return whether the root selected the group-member narrow writer protocol."""
    schema_path = shared_paths(cfg.shared_root)["schema"] / "version.json"
    state_path = shared_paths(cfg.shared_root)["ready"] / "group-members" / "state.json"
    try:
        schema = read_json(schema_path).get("schema")
        state = read_json(state_path).get("group_ready_members")
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return False
    if not isinstance(schema, dict) or not isinstance(state, dict):
        return False
    capabilities = schema.get("required_capabilities", [])
    return (
        isinstance(capabilities, list)
        and GROUP_READY_MEMBERS_CAPABILITY in capabilities
        and state.get("state") == "active"
    )


@contextmanager
def exclusive(path: Path, *, blocking: bool = True) -> Iterator[bool]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    has_lock = False
    try:
        flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
        try:
            fcntl.flock(handle.fileno(), flags)
        except OSError as exc:
            if not blocking and exc.errno in {errno.EACCES, errno.EAGAIN}:
                yield False
                return
            raise
        has_lock = True
        yield True
    finally:
        if has_lock:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


@contextmanager
def shared(path: Path, *, blocking: bool = True) -> Iterator[bool]:
    """Hold a shared filesystem lock for a schema reader."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    has_lock = False
    try:
        flags = fcntl.LOCK_SH if blocking else fcntl.LOCK_SH | fcntl.LOCK_NB
        try:
            fcntl.flock(handle.fileno(), flags)
        except OSError as exc:
            if not blocking and exc.errno in {errno.EACCES, errno.EAGAIN}:
                yield False
                return
            raise
        has_lock = True
        yield True
    finally:
        if has_lock:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


@contextmanager
def schema_lock(root: Path, *, blocking: bool = True) -> Iterator[bool]:
    with exclusive(lock_path(root, "schema"), blocking=blocking) as acquired:
        yield acquired


@contextmanager
def schema_reader_lock(root: Path, *, blocking: bool = True) -> Iterator[bool]:
    """Hold the shared side of the schema fencing lock."""
    with shared(lock_path(root, "schema"), blocking=blocking) as acquired:
        yield acquired


@contextmanager
def schema_writer_lock(cfg: object, *, blocking: bool = True) -> Iterator[bool]:
    """Fence an authoritative writer against schema/capability mutation."""
    root = cfg.shared_root.resolve()
    held_roots = _schema_writer_roots.get()
    if root in held_roots:
        yield True
        return
    if is_schema_narrow_protocol_active(cfg):
        with schema_reader_lock(cfg.shared_root, blocking=blocking) as acquired:
            if acquired and is_schema_narrow_protocol_active(cfg):
                token = _schema_writer_roots.set(held_roots | {root})
                try:
                    yield True
                finally:
                    _schema_writer_roots.reset(token)
                return
    with schema_lock(cfg.shared_root, blocking=blocking) as acquired:
        if not acquired:
            yield False
            return
        token = _schema_writer_roots.set(held_roots | {root})
        try:
            yield True
        finally:
            _schema_writer_roots.reset(token)


@contextmanager
def group_lock(root: Path, name: str, *, blocking: bool = True) -> Iterator[bool]:
    with exclusive(lock_path(root, "groups", name), blocking=blocking) as acquired:
        yield acquired


@contextmanager
def group_writer_lock(cfg: object, name: str, *, blocking: bool = True) -> Iterator[bool]:
    """Acquire the schema and Group portions of the authoritative writer order."""
    with schema_writer_lock(cfg, blocking=blocking) as has_schema_lock:
        if not has_schema_lock:
            yield False
            return
        with group_lock(cfg.shared_root, name, blocking=blocking) as has_group_lock:
            yield has_group_lock


@contextmanager
def task_lock(root: Path, task_id: str, *, blocking: bool = True) -> Iterator[bool]:
    with exclusive(lock_path(root, "tasks", task_id), blocking=blocking) as acquired:
        yield acquired


@contextmanager
def task_writer_lock(
    cfg: object,
    task_id: str,
    group_name: str | None,
    *,
    blocking: bool = True,
) -> Iterator[bool]:
    """Acquire schema, optional Group, and Task writer fences in global order."""
    with schema_writer_lock(cfg, blocking=blocking) as has_schema_lock:
        if not has_schema_lock:
            yield False
            return
        with ExitStack() as stack:
            if group_name:
                has_group_lock = stack.enter_context(
                    group_lock(cfg.shared_root, group_name, blocking=blocking)
                )
                if not has_group_lock:
                    yield False
                    return
            with task_lock(cfg.shared_root, task_id, blocking=blocking) as has_task_lock:
                yield has_task_lock


@contextmanager
def task_locks(root: Path, task_ids: list[str]) -> Iterator[None]:
    """Acquire Task identity fences in the required global order."""
    unique_ids = sorted(set(task_ids))
    if len(unique_ids) != len(task_ids):
        raise ValueError("submission task IDs must be unique.")
    with ExitStack() as stack:
        for task_id in unique_ids:
            stack.enter_context(task_lock(root, task_id))
        yield


@contextmanager
def idempotency_lock(root: Path, digest: str, *, blocking: bool = True) -> Iterator[bool]:
    """Serialize one idempotency key without serializing unrelated submissions."""
    with exclusive(lock_path(root, "idempotency", digest), blocking=blocking) as acquired:
        yield acquired


@contextmanager
def machine_lock(root: Path, machine_name: str, *, blocking: bool = True) -> Iterator[bool]:
    with exclusive(lock_path(root, "machines", machine_name), blocking=blocking) as acquired:
        yield acquired
