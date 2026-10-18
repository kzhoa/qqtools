"""Rebuildable reverse projection for Project-wide Task dependencies.

Task records remain the only dependency authority.  This module stores a
generation-scoped reverse view solely so cleanup can find direct dependents
without walking the entire Task namespace.  Every mutating entry point must
hold the Project dependency fence while calling the mutation helpers.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .directory_capture import read_directory_entry
from .paths import (
    shared_paths,
    task_dependency_generation_path,
    task_dependency_record_path,
    task_dependency_state_path,
    task_path,
)
from .records import TaskRecord, utc_now, validate_identifier
from .store import atomic_replace, read_json

PROJECTION_VERSION = 1
PROJECTION_PROTOCOL = "task-dependencies-v2"
PROJECTION_STATUS = frozenset({"active", "building", "degraded"})
PROJECTION_PHASES = frozenset({"complete", "capture", "forward_audit", "reverse_audit", "select"})
MAX_FAILURE_REASON = 512


class ProjectionUnavailable(RuntimeError):
    """The reverse projection cannot prove complete cleanup coverage."""


def _storage_read(storage: Any | None, path: Path, *, max_bytes: int = 1_048_576) -> dict[str, Any]:
    _require_regular_file(path)
    if storage is not None:
        return storage.read_json_limited(path, max_bytes=max_bytes)
    return read_json(path)


def _storage_write(storage: Any | None, path: Path, value: dict[str, Any]) -> None:
    if storage is not None:
        storage.atomic_replace(path, value)
    else:
        atomic_replace(path, value)


def _require_regular_file(path: Path) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ProjectionUnavailable(f"task dependency projection evidence is not a regular file: {path}")
    return metadata


def _project_id(cfg: Any, *, storage: Any | None = None) -> str:
    identity_path = shared_paths(cfg.shared_root)["project"] / "identity.json"
    try:
        identity = _storage_read(storage, identity_path, max_bytes=16 * 1024).get("project", {})
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        raise ProjectionUnavailable(f"qexp Project identity is unavailable: {identity_path}") from exc
    project_id = identity.get("project_id") if isinstance(identity, dict) else None
    shared_root = identity.get("shared_root") if isinstance(identity, dict) else None
    if not isinstance(project_id, str) or not project_id or shared_root != str(cfg.shared_root):
        raise ProjectionUnavailable("task dependency projection identity does not match the Project root")
    return project_id


def _state_value(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"dependency_projection"}:
        raise ProjectionUnavailable("task dependency projection state has an invalid wrapper")
    state = value["dependency_projection"]
    if not isinstance(state, dict):
        raise ProjectionUnavailable("task dependency projection state is not an object")
    expected = {
        "version",
        "protocol",
        "project_id",
        "state",
        "revision",
        "active_generation",
        "building_generation",
        "phase",
        "cursor",
        "failure_reason",
        "updated_at",
    }
    if set(state) != expected:
        raise ProjectionUnavailable("task dependency projection state has an invalid shape")
    if state["version"] != PROJECTION_VERSION or state["protocol"] != PROJECTION_PROTOCOL:
        raise ProjectionUnavailable("task dependency projection protocol is unsupported")
    if state["state"] not in PROJECTION_STATUS:
        raise ProjectionUnavailable("task dependency projection state is invalid")
    if type(state["revision"]) is not int or state["revision"] < 0:
        raise ProjectionUnavailable("task dependency projection revision is invalid")
    for field in ("active_generation", "building_generation"):
        value = state[field]
        if value is not None and (not isinstance(value, str) or not value):
            raise ProjectionUnavailable(f"task dependency projection {field} is invalid")
    if state["state"] == "active":
        if not state["active_generation"] or state["building_generation"] is not None:
            raise ProjectionUnavailable("active task dependency projection has no sole generation")
        if state["phase"] != "complete" or state["cursor"] is not None or state["failure_reason"] is not None:
            raise ProjectionUnavailable("active task dependency projection has incomplete build evidence")
    elif state["state"] == "building":
        if not state["building_generation"] or state["phase"] not in PROJECTION_PHASES - {"complete"}:
            raise ProjectionUnavailable("building task dependency projection has invalid generation evidence")
        if not isinstance(state["cursor"], dict):
            raise ProjectionUnavailable("building task dependency projection cursor is invalid")
    else:
        if not isinstance(state["failure_reason"], str) or not state["failure_reason"]:
            raise ProjectionUnavailable("degraded task dependency projection has no bounded failure reason")
        if len(state["failure_reason"]) > MAX_FAILURE_REASON:
            raise ProjectionUnavailable("task dependency projection failure reason is unbounded")
    if not isinstance(state["updated_at"], str):
        raise ProjectionUnavailable("task dependency projection timestamp is invalid")
    return state


def _read_state(cfg: Any, *, storage: Any | None = None) -> dict[str, Any]:
    path = task_dependency_state_path(cfg.shared_root)
    value = _storage_read(storage, path, max_bytes=64 * 1024)
    state = _state_value(value)
    if state["project_id"] != _project_id(cfg, storage=storage):
        raise ProjectionUnavailable("task dependency projection belongs to a different Project")
    return state


def read_projection_state(cfg: Any, *, storage: Any | None = None) -> dict[str, Any]:
    """Read and validate the identity-bound projection state."""
    return dict(_read_state(cfg, storage=storage))


def inspect_projection(cfg: Any, *, storage: Any | None = None) -> dict[str, Any]:
    """Return bounded diagnostics without repairing or inferring Task edges."""
    path = task_dependency_state_path(cfg.shared_root)
    try:
        state = _read_state(cfg, storage=storage)
    except FileNotFoundError:
        return {"state": "missing", "path": str(path), "active_generation": None, "revision": None}
    except (OSError, ProjectionUnavailable, TypeError, ValueError) as exc:
        return {
            "state": "malformed",
            "path": str(path),
            "active_generation": None,
            "revision": None,
            "reason": str(exc)[:MAX_FAILURE_REASON],
        }
    return {
        "state": state["state"],
        "path": str(path),
        "active_generation": state["active_generation"],
        "building_generation": state["building_generation"],
        "phase": state["phase"],
        "cursor": state["cursor"],
        "revision": state["revision"],
        "failure_reason": state["failure_reason"],
    }


projection_status = inspect_projection


def _new_state(project_id: str, generation: str, *, status: str = "active") -> dict[str, Any]:
    return {
        "dependency_projection": {
            "version": PROJECTION_VERSION,
            "protocol": PROJECTION_PROTOCOL,
            "project_id": project_id,
            "state": status,
            "revision": 0,
            "active_generation": generation if status == "active" else None,
            "building_generation": None,
            "phase": "complete" if status == "active" else "capture",
            "cursor": None if status == "active" else {"offset": 0},
            "failure_reason": None,
            "updated_at": utc_now(),
        }
    }


def _ensure_generation(root: Path, generation: str) -> Path:
    generation_path = task_dependency_generation_path(root, generation)
    generation_path.mkdir(parents=True, exist_ok=True)
    records = generation_path / "records"
    records.mkdir(parents=True, exist_ok=True)
    return records


def initialize_empty_projection(cfg: Any) -> dict[str, Any]:
    """Initialize an empty active generation for a fresh v2 root."""
    path = task_dependency_state_path(cfg.shared_root)
    try:
        return _read_state(cfg)
    except FileNotFoundError:
        pass
    except ProjectionUnavailable:
        raise
    generation = uuid.uuid4().hex
    _ensure_generation(cfg.shared_root, generation)
    document = _new_state(_project_id(cfg), generation)
    _storage_write(None, path, document)
    return _state_value(document)


def _write_state(cfg: Any, state: dict[str, Any], *, storage: Any | None = None) -> dict[str, Any]:
    state = dict(state)
    state["revision"] += 1
    document = {"dependency_projection": state}
    _state_value(document)
    _storage_write(storage, task_dependency_state_path(cfg.shared_root), document)
    return _read_state(cfg, storage=storage)


def mark_degraded(
    cfg: Any,
    reason: str,
    *,
    storage: Any | None = None,
    preserve_build: bool = False,
) -> dict[str, Any]:
    """Durably disable cleanup coverage after an uncertain projection boundary."""
    try:
        state = _read_state(cfg, storage=storage)
    except ProjectionUnavailable:
        raise
    bounded = str(reason)[:MAX_FAILURE_REASON] or "projection_degraded"
    state["state"] = "degraded"
    state["failure_reason"] = bounded
    state["updated_at"] = utc_now()
    if not preserve_build:
        state["building_generation"] = None
        state["phase"] = "complete"
        state["cursor"] = None
    return _write_state(cfg, state, storage=storage)


def invalidate_legacy_rebuild(cfg: Any, reason: str, *, storage: Any | None = None) -> None:
    """Invalidate pre-v2 coverage before a legacy writer changes dependency truth.

    Current-version v1 writers use the Project dependency fence and call this
    before creating/deleting Tasks or editing edges. A migration slice holding
    the same fence therefore either observes the old truth or restarts from the
    new truth; it can never activate a generation assembled across the write.
    """
    try:
        state = _read_state(cfg, storage=storage)
    except FileNotFoundError:
        return
    if state["state"] in {"building", "active"}:
        mark_degraded(cfg, f"legacy dependency mutation: {reason}", storage=storage)


def begin_mutation(
    cfg: Any,
    *,
    storage: Any | None = None,
    allow_degraded: bool = False,
) -> dict[str, Any]:
    """Mark active coverage degraded before canonical Task truth is changed."""
    state = _read_state(cfg, storage=storage)
    if state["state"] != "active":
        if not (allow_degraded and state["state"] == "degraded"):
            raise ProjectionUnavailable("task dependency projection is not active for mutation")
        return state
    state["state"] = "degraded"
    state["failure_reason"] = "canonical dependency mutation in progress"
    state["updated_at"] = utc_now()
    return _write_state(cfg, state, storage=storage)


def _record_document(
    *,
    project_id: str,
    generation: str,
    prerequisite_id: str,
    dependent_ids: Iterable[str],
) -> dict[str, Any]:
    normalized_dependents = sorted(set(dependent_ids))
    digest_input = {
        "project_id": project_id,
        "generation": generation,
        "prerequisite_task_id": prerequisite_id,
        "dependent_task_ids": normalized_dependents,
    }
    return {
        "task_dependency_record": {
            "version": PROJECTION_VERSION,
            "protocol": PROJECTION_PROTOCOL,
            "project_id": project_id,
            "generation": generation,
            "prerequisite_task_id": prerequisite_id,
            "dependent_task_ids": normalized_dependents,
            "content_digest": hashlib.sha256(
                json.dumps(digest_input, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "updated_at": utc_now(),
        }
    }


def _record_value(
    value: Mapping[str, Any],
    *,
    project_id: str,
    generation: str,
    expected_id: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"task_dependency_record"}:
        raise ProjectionUnavailable("task dependency reverse record has an invalid wrapper")
    record = value["task_dependency_record"]
    required = {
        "version",
        "protocol",
        "project_id",
        "generation",
        "prerequisite_task_id",
        "dependent_task_ids",
        "content_digest",
        "updated_at",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise ProjectionUnavailable("task dependency reverse record has an invalid shape")
    if (
        record["version"] != PROJECTION_VERSION
        or record["protocol"] != PROJECTION_PROTOCOL
        or record["project_id"] != project_id
        or record["generation"] != generation
        or record["prerequisite_task_id"] != expected_id
    ):
        raise ProjectionUnavailable("task dependency reverse record identity is invalid")
    try:
        dependents = [validate_identifier(item, "projected dependent Task ID") for item in record["dependent_task_ids"]]
    except (TypeError, ValueError) as exc:
        raise ProjectionUnavailable("task dependency reverse record dependents are invalid") from exc
    if not isinstance(record["dependent_task_ids"], list) or dependents != sorted(set(dependents)):
        raise ProjectionUnavailable("task dependency reverse record dependents are not sorted and unique")
    digest_input = {
        "project_id": project_id,
        "generation": generation,
        "prerequisite_task_id": expected_id,
        "dependent_task_ids": dependents,
    }
    expected_digest = hashlib.sha256(
        json.dumps(digest_input, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if record["content_digest"] != expected_digest:
        raise ProjectionUnavailable("task dependency reverse record content digest is invalid")
    if not isinstance(record["updated_at"], str):
        raise ProjectionUnavailable("task dependency reverse record timestamp is invalid")
    return record


def _read_record(cfg: Any, generation: str, task_id: str, *, storage: Any | None = None) -> dict[str, Any]:
    path = task_dependency_record_path(cfg.shared_root, generation, task_id)
    try:
        value = _storage_read(storage, path, max_bytes=128 * 1024)
    except FileNotFoundError as exc:
        raise ProjectionUnavailable(f"task dependency reverse record is missing for {task_id!r}") from exc
    return _record_value(
        value, project_id=_project_id(cfg, storage=storage), generation=generation, expected_id=task_id
    )


def _write_record(
    cfg: Any,
    generation: str,
    task_id: str,
    dependent_ids: Iterable[str],
    *,
    storage: Any | None = None,
) -> None:
    validate_identifier(task_id, "prerequisite Task ID")
    records = _ensure_generation(cfg.shared_root, generation)
    path = records / f"{task_id}.json"
    _storage_write(
        storage,
        path,
        _record_document(
            project_id=_project_id(cfg, storage=storage),
            generation=generation,
            prerequisite_id=task_id,
            dependent_ids=dependent_ids,
        ),
    )


def _task_edges(task: TaskRecord | Mapping[str, Any] | None) -> tuple[str, tuple[str, ...]] | None:
    if task is None:
        return None
    if isinstance(task, TaskRecord):
        return task.task_id, tuple(task.depends_on_task_ids)
    raw_task = task.get("task") if isinstance(task, Mapping) else None
    if not isinstance(raw_task, Mapping):
        raise ValueError("Task edge input is malformed")
    task_id = validate_identifier(raw_task.get("task_id"), "Task ID")
    dependencies = raw_task.get("depends_on_task_ids", [])
    if not isinstance(dependencies, list):
        raise ValueError("Task dependency edge input is malformed")
    normalized = tuple(sorted({validate_identifier(item, "depends_on_task_ids") for item in dependencies}))
    return task_id, normalized


def synchronize_task_edges(
    cfg: Any,
    before: TaskRecord | Mapping[str, Any] | None,
    after: TaskRecord | Mapping[str, Any] | None,
    *,
    storage: Any | None = None,
) -> None:
    """Apply one canonical Task edge delta to the active generation records."""
    state = _read_state(cfg, storage=storage)
    if state["state"] not in {"active", "degraded"} or not state["active_generation"]:
        raise ProjectionUnavailable("task dependency projection cannot synchronize in its current state")
    old = _task_edges(before)
    new = _task_edges(after)
    if old is None and new is None:
        return
    generation = state["active_generation"]
    changed_prerequisites = set(old[1] if old else ()) | set(new[1] if new else ())
    if new is not None:
        changed_prerequisites.add(new[0])
    if old is not None and new is None:
        changed_prerequisites.add(old[0])
    for prerequisite_id in sorted(changed_prerequisites):
        try:
            record = _read_record(cfg, generation, prerequisite_id, storage=storage)
            dependents = set(record["dependent_task_ids"])
        except ProjectionUnavailable:
            # A newly created Task has no prior reverse record.  Any existing
            # prerequisite must already have one in active coverage.
            if new is None or prerequisite_id != new[0]:
                raise
            dependents = set()
        if old is not None and old[0] in dependents and prerequisite_id in old[1]:
            dependents.discard(old[0])
        if new is not None and prerequisite_id in new[1]:
            dependents.add(new[0])
        _write_record(cfg, generation, prerequisite_id, dependents, storage=storage)
    if old is not None and new is None:
        path = task_dependency_record_path(cfg.shared_root, generation, old[0])
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ProjectionUnavailable(f"task dependency reverse record is unsafe: {path}")
        path.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def synchronize_task_edge_batch(
    cfg: Any,
    changes: Iterable[tuple[TaskRecord | Mapping[str, Any] | None, TaskRecord | Mapping[str, Any] | None]],
    *,
    storage: Any | None = None,
) -> None:
    """Synchronize several before/after Task edge pairs."""
    for before, after in changes:
        synchronize_task_edges(cfg, before, after, storage=storage)


def complete_mutation(cfg: Any, *, storage: Any | None = None) -> dict[str, Any]:
    """Republish active coverage after every canonical mutation is durable."""
    state = _read_state(cfg, storage=storage)
    if state["state"] != "degraded" or not state["active_generation"]:
        raise ProjectionUnavailable("task dependency projection mutation was not fenced")
    state["state"] = "active"
    state["failure_reason"] = None
    state["phase"] = "complete"
    state["cursor"] = None
    state["updated_at"] = utc_now()
    return _write_state(cfg, state, storage=storage)


@contextmanager
def projection_mutation(cfg: Any, *, storage: Any | None = None) -> Iterator[None]:
    """Fence a batch so crashes leave explicit degraded cleanup evidence."""
    begin_mutation(cfg, storage=storage)
    try:
        yield
    except BaseException as exc:
        try:
            mark_degraded(cfg, f"mutation failed: {type(exc).__name__}", storage=storage)
        except Exception:
            pass
        raise
    complete_mutation(cfg, storage=storage)


def _directory_entry(directory: Path, offset: int, *, storage: Any | None = None) -> tuple[str | None, int]:
    try:
        metadata = directory.lstat()
    except FileNotFoundError:
        return None, offset
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ProjectionUnavailable(f"task dependency projection namespace is not a real directory: {directory}")
    if storage is not None:
        storage.account_metadata_ops(1)
    try:
        return read_directory_entry(directory, offset)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ProjectionUnavailable(f"task dependency projection cannot traverse {directory}") from exc


def _task_entry(cfg: Any, offset: int, *, storage: Any | None = None) -> tuple[str | None, int]:
    name, next_offset = _directory_entry(shared_paths(cfg.shared_root)["tasks"], offset, storage=storage)
    if name is None:
        return None, next_offset
    if not name.endswith(".json"):
        raise ProjectionUnavailable(f"qexp Task namespace contains unexpected entry: {name}")
    path = shared_paths(cfg.shared_root)["tasks"] / name
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise ProjectionUnavailable(f"qexp Task namespace changed during projection traversal: {name}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ProjectionUnavailable(f"qexp Task namespace contains unsafe entry: {name}")
    return validate_identifier(name[:-5], "Task ID"), next_offset


def _record_entry(cfg: Any, generation: str, offset: int, *, storage: Any | None = None) -> tuple[str | None, int]:
    directory = task_dependency_generation_path(cfg.shared_root, generation) / "records"
    name, next_offset = _directory_entry(directory, offset, storage=storage)
    if name is None:
        return None, next_offset
    if not name.endswith(".json"):
        raise ProjectionUnavailable(f"task dependency projection contains unexpected entry: {name}")
    path = directory / name
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise ProjectionUnavailable(f"task dependency projection changed during traversal: {name}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ProjectionUnavailable(f"task dependency projection contains unsafe entry: {name}")
    return validate_identifier(name[:-5], "projected prerequisite Task ID"), next_offset


def _load_task(cfg: Any, task_id: str, *, storage: Any | None = None) -> TaskRecord:
    return TaskRecord.from_dict(_storage_read(storage, task_path(cfg.shared_root, task_id), max_bytes=256 * 1024))


def _validate_forward_edges(cfg: Any, task: TaskRecord, *, storage: Any | None = None) -> None:
    if task.task_id in task.depends_on_task_ids:
        raise ProjectionUnavailable(f"Task {task.task_id!r} contains a self dependency")
    for prerequisite_id in task.depends_on_task_ids:
        try:
            prerequisite = _load_task(cfg, prerequisite_id, storage=storage)
        except FileNotFoundError as exc:
            raise ProjectionUnavailable(
                f"Task {task.task_id!r} references missing prerequisite {prerequisite_id!r}"
            ) from exc
        if not prerequisite.submission_operation_id:
            raise ProjectionUnavailable(f"Task {task.task_id!r} references an uncommitted prerequisite")


def _capture_task(cfg: Any, generation: str, task: TaskRecord, *, storage: Any | None = None) -> None:
    """Merge one Task's authoritative outgoing edges into a build generation."""
    try:
        own_record = _read_record(cfg, generation, task.task_id, storage=storage)
        own_dependents = own_record["dependent_task_ids"]
    except ProjectionUnavailable:
        own_dependents = []
    _write_record(cfg, generation, task.task_id, own_dependents, storage=storage)
    for prerequisite_id in task.depends_on_task_ids:
        try:
            record = _read_record(cfg, generation, prerequisite_id, storage=storage)
            dependents = set(record["dependent_task_ids"])
        except ProjectionUnavailable:
            dependents = set()
        dependents.add(task.task_id)
        _write_record(cfg, generation, prerequisite_id, dependents, storage=storage)


def _audit_forward_record(cfg: Any, generation: str, task: TaskRecord, *, storage: Any | None = None) -> None:
    _validate_forward_edges(cfg, task, storage=storage)
    try:
        _read_record(cfg, generation, task.task_id, storage=storage)
    except ProjectionUnavailable as exc:
        raise ProjectionUnavailable(f"task dependency reverse record is missing for Task {task.task_id!r}") from exc
    for prerequisite_id in task.depends_on_task_ids:
        record = _read_record(cfg, generation, prerequisite_id, storage=storage)
        if task.task_id not in record["dependent_task_ids"]:
            raise ProjectionUnavailable(
                f"task dependency reverse record omits {task.task_id!r} from {prerequisite_id!r}"
            )


def _audit_reverse_record(cfg: Any, generation: str, prerequisite_id: str, *, storage: Any | None = None) -> None:
    try:
        _load_task(cfg, prerequisite_id, storage=storage)
    except FileNotFoundError as exc:
        raise ProjectionUnavailable(
            f"task dependency reverse record has no canonical Task {prerequisite_id!r}"
        ) from exc
    record = _read_record(cfg, generation, prerequisite_id, storage=storage)
    for dependent_id in record["dependent_task_ids"]:
        try:
            dependent = _load_task(cfg, dependent_id, storage=storage)
        except FileNotFoundError as exc:
            raise ProjectionUnavailable(f"projected dependent Task {dependent_id!r} is missing") from exc
        if prerequisite_id not in dependent.depends_on_task_ids:
            raise ProjectionUnavailable(
                f"task dependency reverse record disagrees with canonical Task {dependent_id!r} truth"
            )


def start_rebuild(cfg: Any, *, storage: Any | None = None) -> dict[str, Any]:
    """Create a fresh generation and durable bounded rebuild cursor."""
    try:
        current = _read_state(cfg, storage=storage)
    except FileNotFoundError:
        current = {
            "version": PROJECTION_VERSION,
            "protocol": PROJECTION_PROTOCOL,
            "project_id": _project_id(cfg, storage=storage),
            "state": "degraded",
            "revision": 0,
            "active_generation": None,
            "building_generation": None,
            "phase": "complete",
            "cursor": None,
            "failure_reason": "projection build required",
            "updated_at": utc_now(),
        }
    if current["state"] == "building":
        return current
    generation = uuid.uuid4().hex
    _ensure_generation(cfg.shared_root, generation)
    next_state = dict(current)
    next_state.update(
        {
            "state": "building",
            "building_generation": generation,
            "phase": "capture",
            "cursor": {"offset": 0},
            "failure_reason": None,
            "updated_at": utc_now(),
        }
    )
    return _write_state(cfg, next_state, storage=storage)


def advance_rebuild(cfg: Any, *, storage: Any | None = None, max_items: int = 1) -> dict[str, Any]:
    """Advance at most ``max_items`` Task/reverse-record rebuild units."""
    if type(max_items) is not int or not 1 <= max_items <= 64:
        raise ValueError("max_items must be between 1 and 64")
    state = start_rebuild(cfg, storage=storage)
    if state["state"] != "building":
        return {"state": "complete", "phase": state["phase"], "cursor": state["cursor"], "work_items": 0}
    work = 0
    while work < max_items:
        state = _read_state(cfg, storage=storage)
        generation = state["building_generation"]
        cursor = dict(state["cursor"])
        offset = cursor.get("offset")
        if type(offset) is not int or offset < 0:
            raise ProjectionUnavailable("task dependency projection rebuild cursor is invalid")
        if state["phase"] == "capture":
            task_id, next_offset = _task_entry(cfg, offset, storage=storage)
            if task_id is None:
                state["phase"] = "forward_audit"
                state["cursor"] = {"offset": 0}
                _write_state(cfg, state, storage=storage)
                continue
            _capture_task(cfg, generation, _load_task(cfg, task_id, storage=storage), storage=storage)
            cursor["offset"] = next_offset
            state["cursor"] = cursor
            state["updated_at"] = utc_now()
            _write_state(cfg, state, storage=storage)
            work += 1
            continue
        if state["phase"] == "forward_audit":
            task_id, next_offset = _task_entry(cfg, offset, storage=storage)
            if task_id is None:
                state["phase"] = "reverse_audit"
                state["cursor"] = {"offset": 0}
                _write_state(cfg, state, storage=storage)
                continue
            _audit_forward_record(cfg, generation, _load_task(cfg, task_id, storage=storage), storage=storage)
            cursor["offset"] = next_offset
            state["cursor"] = cursor
            state["updated_at"] = utc_now()
            _write_state(cfg, state, storage=storage)
            work += 1
            continue
        if state["phase"] == "reverse_audit":
            prerequisite_id, next_offset = _record_entry(cfg, generation, offset, storage=storage)
            if prerequisite_id is None:
                state["phase"] = "select"
                state["cursor"] = {"offset": 0}
                _write_state(cfg, state, storage=storage)
                continue
            _audit_reverse_record(cfg, generation, prerequisite_id, storage=storage)
            cursor["offset"] = next_offset
            state["cursor"] = cursor
            state["updated_at"] = utc_now()
            _write_state(cfg, state, storage=storage)
            work += 1
            continue
        if state["phase"] == "select":
            state["state"] = "active"
            state["active_generation"] = generation
            state["building_generation"] = None
            state["phase"] = "complete"
            state["cursor"] = None
            state["failure_reason"] = None
            state["updated_at"] = utc_now()
            _write_state(cfg, state, storage=storage)
            return {"state": "complete", "phase": "complete", "cursor": None, "work_items": work}
        raise ProjectionUnavailable(f"unknown task dependency rebuild phase {state['phase']!r}")
    state = _read_state(cfg, storage=storage)
    return {"state": "in_progress", "phase": state["phase"], "cursor": state["cursor"], "work_items": work}


def rebuild_projection_slice(cfg: Any, *, storage: Any | None = None, max_items: int = 1) -> dict[str, Any]:
    """Compatibility alias for one bounded rebuild slice."""
    return advance_rebuild(cfg, storage=storage, max_items=max_items)


def activate_rebuilt_generation(cfg: Any, *, storage: Any | None = None) -> dict[str, Any]:
    """Select a fully audited generation as active."""
    state = _read_state(cfg, storage=storage)
    if state["state"] == "active":
        return state
    if state["state"] != "building" or state["phase"] != "select":
        raise ProjectionUnavailable("task dependency projection is not ready for activation")
    generation = state["building_generation"]
    if not generation:
        raise ProjectionUnavailable("task dependency projection has no build generation")
    state.update(
        {
            "state": "active",
            "active_generation": generation,
            "building_generation": None,
            "phase": "complete",
            "cursor": None,
            "failure_reason": None,
            "updated_at": utc_now(),
        }
    )
    return _write_state(cfg, state, storage=storage)


def projected_dependents(cfg: Any, task_id: str, *, storage: Any | None = None) -> list[str]:
    """Validate and return direct dependents proven by active coverage."""
    task_id = validate_identifier(task_id, "Task ID")
    state = _read_state(cfg, storage=storage)
    if state["state"] != "active" or not state["active_generation"]:
        raise ProjectionUnavailable("task dependency projection is not active")
    try:
        target = _load_task(cfg, task_id, storage=storage)
    except (FileNotFoundError, OSError, KeyError, TypeError, ValueError) as exc:
        raise ProjectionUnavailable(f"cleanup target Task {task_id!r} is unavailable") from exc
    del target
    record = _read_record(cfg, state["active_generation"], task_id, storage=storage)
    dependents = list(record["dependent_task_ids"])
    for dependent_id in dependents:
        try:
            dependent = _load_task(cfg, dependent_id, storage=storage)
        except (FileNotFoundError, OSError, KeyError, TypeError, ValueError) as exc:
            raise ProjectionUnavailable(f"projected dependent Task {dependent_id!r} is unavailable") from exc
        if task_id not in dependent.depends_on_task_ids:
            raise ProjectionUnavailable(
                f"task dependency projection disagrees with canonical Task {dependent_id!r} truth"
            )
    return dependents


validate_project_cleanup_dependencies = projected_dependents
cleanup_dependents = projected_dependents


def remove_task_from_projection(
    cfg: Any,
    task_id: str,
    *,
    before: TaskRecord | Mapping[str, Any] | None = None,
    storage: Any | None = None,
) -> None:
    """Remove a deleted Task's outgoing memberships and own reverse record."""
    if before is None:
        try:
            before = _load_task(cfg, task_id, storage=storage)
        except FileNotFoundError:
            before = {"task": {"task_id": task_id, "depends_on_task_ids": []}}
    synchronize_task_edges(cfg, before, None, storage=storage)


def projection_digest(cfg: Any, *, storage: Any | None = None) -> str:
    """Return a stable digest of validated projection state for diagnostics."""
    state = _read_state(cfg, storage=storage)
    return hashlib.sha256(str(sorted(state.items())).encode()).hexdigest()


__all__ = [
    "ProjectionUnavailable",
    "PROJECTION_PROTOCOL",
    "PROJECTION_VERSION",
    "activate_rebuilt_generation",
    "advance_rebuild",
    "begin_mutation",
    "cleanup_dependents",
    "complete_mutation",
    "initialize_empty_projection",
    "invalidate_legacy_rebuild",
    "inspect_projection",
    "mark_degraded",
    "projected_dependents",
    "projection_digest",
    "projection_mutation",
    "projection_status",
    "read_projection_state",
    "rebuild_projection_slice",
    "remove_task_from_projection",
    "start_rebuild",
    "synchronize_task_edge_batch",
    "synchronize_task_edges",
    "validate_project_cleanup_dependencies",
]
