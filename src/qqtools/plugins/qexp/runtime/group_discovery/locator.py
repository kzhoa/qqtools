"""Identity-bound, generation-fenced Group service locators.

Locators only make outstanding work discoverable. Existing Group, Task,
Submission, operation, and maintenance records remain the authority for any
effect. Callers publish and acknowledge while holding the schema and Group
writer fences.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Callable

from ..directory_capture import read_directory_entry
from ..group_namespace import group_authority_identity
from ..records import utc_now, validate_group_name
from ..store import atomic_replace

LANES = ("control", "maintenance", "membership")
SHARD_COUNT = 256
PATH_SCHEME = "sha256-first-byte-v1"
MAX_RECORD_BYTES = 16 * 1024
_MAX_GENERATION = (1 << 63) - 1
_LAYOUT_FIELDS = frozenset(
    {
        "version",
        "identity",
        "revision",
        "lanes",
        "shard_count",
        "path_scheme",
        "manifest_digest",
        "created_at",
    }
)
_LAYOUT_IDENTITY_FIELDS = frozenset({"project_id", "group_directory_identity"})
_LOCATOR_FIELDS = frozenset({"version", "identity", "service", "generation", "published_at", "reason"})
_LOCATOR_IDENTITY_FIELDS = frozenset({"project_id", "group_directory_identity", "group"})
_REASONS = {
    "control": frozenset({"task_change", "group_operation", "repair", "bootstrap"}),
    "maintenance": frozenset({"source_cleanup", "metadata_cleanup", "repair", "bootstrap"}),
    "membership": frozenset({"submission_commit", "submission_finalize", "repair", "bootstrap"}),
}


def group_service_root(root: Path) -> Path:
    """Return the shared Group service protocol root."""
    return Path(root) / "operations" / "group-service-v1"


def group_locator_path(root: Path, group: str, lane: str) -> Path:
    """Return the canonical locator path for one Group and lane."""
    _validate_lane(lane)
    validate_group_name(group)
    digest = hashlib.sha256(group.encode("utf-8")).hexdigest()
    return group_service_root(root) / lane / digest[:2] / f"{digest}.json"


def _validate_lane(lane: str) -> None:
    if lane not in LANES:
        raise ValueError(f"unsupported Group service lane: {lane!r}")


def _authority_identity(root: Path, storage: Any | None = None) -> dict[str, Any]:
    authority = group_authority_identity(root, storage=storage)
    project_id = authority.get("project_id")
    directory_identity = authority.get("directory_identity")
    if not isinstance(project_id, str) or not project_id or type(directory_identity) is not dict:
        raise RuntimeError("Group service authority identity is invalid")
    return {"project_id": project_id, "group_directory_identity": dict(directory_identity)}


def _manifest_digest() -> str:
    paths = [f"{lane}/{shard:02x}" for lane in LANES for shard in range(SHARD_COUNT)]
    encoded = json.dumps(paths, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _lstat(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _ensure_real_directory(path: Path) -> None:
    try:
        path.mkdir()
    except FileExistsError:
        pass
    metadata = _lstat(path)
    if metadata is None or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"Group service layout path is not a real directory: {path}")


def _open_directory(path: Path) -> int:
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)


def _sync_directory(path: Path) -> None:
    descriptor = _open_directory(path)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_json(path: Path, *, max_bytes: int = MAX_RECORD_BYTES, storage: Any | None = None) -> dict[str, Any]:
    if storage is not None:
        return storage.read_json_limited(path, max_bytes=max_bytes)
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > max_bytes:
            raise RuntimeError(f"Group service record is not a bounded regular file: {path}")
        pieces: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(8192, max_bytes + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise RuntimeError(f"Group service record exceeds {max_bytes} bytes: {path}")
            pieces.append(chunk)
    finally:
        os.close(descriptor)
    try:
        value = json.loads(b"".join(pieces))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Group service record is malformed: {path}") from exc
    if type(value) is not dict:
        raise RuntimeError(f"Group service record is not an object: {path}")
    return value


def _validate_layout_record(
    root: Path,
    value: object,
    *,
    storage: Any | None = None,
    authority: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _LAYOUT_FIELDS:
        raise RuntimeError("Group service layout marker has an invalid shape")
    expected_identity = _authority_identity(root, storage) if authority is None else authority
    if (
        type(value.get("version")) is not int
        or value["version"] != 1
        or type(value.get("revision")) is not int
        or value["revision"] != 1
        or type(value.get("identity")) is not dict
        or set(value["identity"]) != _LAYOUT_IDENTITY_FIELDS
        or value["identity"] != expected_identity
        or value.get("lanes") != list(LANES)
        or type(value.get("shard_count")) is not int
        or value["shard_count"] != SHARD_COUNT
        or value.get("path_scheme") != PATH_SCHEME
        or value.get("manifest_digest") != _manifest_digest()
        or not isinstance(value.get("created_at"), str)
    ):
        raise RuntimeError("Group service layout marker identity or manifest is invalid")
    return value


def _validate_layout_directories(root: Path) -> None:
    service_root = group_service_root(root)
    for path in (root / "operations", service_root, *(service_root / lane for lane in LANES)):
        metadata = _lstat(path)
        if metadata is None or not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(f"Group service layout directory is unavailable: {path}")
    for lane in LANES:
        for shard in range(SHARD_COUNT):
            path = service_root / lane / f"{shard:02x}"
            metadata = _lstat(path)
            if metadata is None or not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeError(f"Group service shard is unavailable: {path}")


def _layout_record(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": 1,
        "identity": identity,
        "revision": 1,
        "lanes": list(LANES),
        "shard_count": SHARD_COUNT,
        "path_scheme": PATH_SCHEME,
        "manifest_digest": _manifest_digest(),
        "created_at": utc_now(),
    }


def advance_group_service_layout(
    cfg: object,
    cursor: dict[str, Any],
    *,
    storage: Any,
    repair: bool = False,
    max_paths: int = 8,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Create or verify a bounded shard batch for coordinator activation."""

    if type(cursor) is not dict or cursor.get("phase") not in {"create", "verify"}:
        raise ValueError("Group service layout cursor is invalid")
    position = cursor.get("position")
    revisions = cursor.get("revisions", {})
    if (
        type(position) is not int
        or position < 0
        or position > len(LANES) * SHARD_COUNT
        or type(revisions) is not dict
        or any(lane not in LANES or type(value) is not list for lane, value in revisions.items())
    ):
        raise ValueError("Group service layout cursor is invalid")
    if type(max_paths) is not int or max_paths < 1:
        raise ValueError("Group service layout max_paths must be positive")
    root = Path(cfg.shared_root)
    identity = _authority_identity(root, storage)
    service_root = group_service_root(root)
    marker = service_root / "layout.json"
    marker_exists = storage.exists(marker)

    # Conservatively charge mkdir/lstat/fsync work that is deliberately outside
    # the migration JSON facade. The charge exceeds the maximum calls below.
    storage.account_metadata_ops(8)
    for ancestor in (root / "operations", service_root):
        _ensure_real_directory(ancestor)
    end = min(position + max_paths, len(LANES) * SHARD_COUNT)
    for current in range(position, end):
        lane = LANES[current // SHARD_COUNT]
        lane_path = service_root / lane
        shard_path = lane_path / f"{current % SHARD_COUNT:02x}"
        storage.account_metadata_ops(8)
        _ensure_real_directory(lane_path)
        if cursor["phase"] == "verify" and current % SHARD_COUNT == 0:
            metadata = lane_path.lstat()
            revisions[lane] = [metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns]
        metadata = _lstat(shard_path)
        if metadata is None:
            if cursor["phase"] == "verify" or (marker_exists and not repair):
                raise RuntimeError(f"Group service immutable layout shard is missing: {shard_path}")
            _ensure_real_directory(shard_path)
        elif not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(f"Group service layout path is not a real directory: {shard_path}")
        if cursor["phase"] == "create":
            _sync_directory(shard_path)
            _sync_directory(lane_path)
        if cursor["phase"] == "verify" and current % SHARD_COUNT == SHARD_COUNT - 1:
            metadata = lane_path.lstat()
            observed = [metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns]
            if observed != revisions.get(lane):
                return {"phase": "verify", "position": current - (SHARD_COUNT - 1), "revisions": {}}, None
    if end < len(LANES) * SHARD_COUNT:
        return {"phase": cursor["phase"], "position": end, "revisions": revisions}, None

    if cursor["phase"] == "create":
        return {"phase": "verify", "position": 0, "revisions": {}}, None

    storage.account_metadata_ops(8)
    for lane in LANES:
        lane_path = service_root / lane
        metadata = lane_path.lstat()
        observed = [metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns]
        if observed != revisions.get(lane):
            return {"phase": "verify", "position": 0, "revisions": {}}, None
    _sync_directory(service_root)
    _sync_directory(root / "operations")
    _sync_directory(root)
    if _authority_identity(root, storage) != identity:
        raise RuntimeError("Group service authority identity changed while creating its layout")
    if marker_exists:
        layout = _read_json(marker, storage=storage)
    else:
        layout = _layout_record(identity)
        _validate_layout_record(root, layout, storage=storage, authority=identity)
        storage.atomic_replace(marker, layout)
    return None, _validate_layout_record(root, layout, storage=storage, authority=identity)


def ensure_group_service_layout(cfg: object, *, storage: Any | None = None) -> dict[str, Any]:
    """Durably create and validate the fixed 768-shard locator layout.

    The marker is committed only after every shard and its ancestor chain have
    been created, synchronized, and rechecked without following symlinks.
    Existing markers are immutable evidence: a partial or mismatched layout is
    rejected rather than silently repaired.
    """
    root = Path(cfg.shared_root)
    identity = _authority_identity(root, storage)
    service_root = group_service_root(root)
    marker = service_root / "layout.json"
    if _lstat(marker) is not None:
        _validate_layout_directories(root)
        return _validate_layout_record(root, _read_json(marker, storage=storage), storage=storage, authority=identity)

    _ensure_real_directory(root / "operations")
    _ensure_real_directory(service_root)
    for lane in LANES:
        lane_path = service_root / lane
        _ensure_real_directory(lane_path)
        for shard in range(SHARD_COUNT):
            _ensure_real_directory(lane_path / f"{shard:02x}")

    # Synchronize leaves, then each containing directory up to the shared root.
    for lane in LANES:
        lane_path = service_root / lane
        for shard in range(SHARD_COUNT):
            _sync_directory(lane_path / f"{shard:02x}")
        _sync_directory(lane_path)
    _sync_directory(service_root)
    _sync_directory(root / "operations")
    _sync_directory(root)
    _validate_layout_directories(root)
    if _authority_identity(root, storage) != identity:
        raise RuntimeError("Group service authority identity changed while creating its layout")

    layout = _layout_record(identity)
    _validate_layout_record(root, layout, storage=storage, authority=identity)
    if storage is None:
        atomic_replace(marker, layout)
    else:
        storage.atomic_replace(marker, layout)
    return _validate_layout_record(root, _read_json(marker, storage=storage), storage=storage, authority=identity)


def read_group_service_layout(root: Path, *, storage: Any | None = None) -> dict[str, Any]:
    """Read and validate the immutable Group service layout marker."""
    root = Path(root)
    service_root = group_service_root(root)
    _validate_layout_directories(root)
    return _validate_layout_record(
        root,
        _read_json(service_root / "layout.json", storage=storage),
        storage=storage,
    )


def read_group_service_layout_marker(root: Path, *, storage: Any | None = None) -> dict[str, Any]:
    """Read coordinator-certified immutable layout evidence without a shard walk."""

    return _read_layout_marker(Path(root), storage=storage)


def _read_layout_marker(root: Path, *, storage: Any | None = None) -> dict[str, Any]:
    """Validate immutable layout evidence without rewalking all fixed shards."""

    return _validate_layout_record(
        root,
        _read_json(group_service_root(root) / "layout.json", storage=storage),
        storage=storage,
    )


def _validate_selected_shard(root: Path, group: str, lane: str) -> Path:
    digest = hashlib.sha256(group.encode("utf-8")).hexdigest()
    shard = group_service_root(root) / lane / digest[:2]
    descriptor = _open_directory(shard)
    os.close(descriptor)
    return shard


def _validate_locator(
    root: Path,
    group: str,
    lane: str,
    value: object,
    *,
    authority: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _LOCATOR_FIELDS:
        raise RuntimeError("Group locator has an invalid shape")
    try:
        validate_group_name(group)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Group locator identity is invalid") from exc
    expected_identity = _authority_identity(root) if authority is None else authority
    identity = value.get("identity")
    if (
        type(value.get("version")) is not int
        or value["version"] != 1
        or type(identity) is not dict
        or set(identity) != _LOCATOR_IDENTITY_FIELDS
        or identity.get("project_id") != expected_identity["project_id"]
        or identity.get("group_directory_identity") != expected_identity["group_directory_identity"]
        or identity.get("group") != group
    ):
        raise RuntimeError("Group locator identity does not match current authority")
    generation = value.get("generation")
    if type(generation) is not int or not 1 <= generation <= _MAX_GENERATION:
        raise RuntimeError("Group locator generation is invalid")
    if value.get("service") != lane:
        raise RuntimeError("Group locator service does not match its lane")
    if value.get("reason") not in _REASONS[lane]:
        raise RuntimeError("Group locator reason is invalid")
    if not isinstance(value.get("published_at"), str) or not value["published_at"]:
        raise RuntimeError("Group locator timestamp is invalid")
    return value


def read_group_locator(
    root: Path,
    group: str,
    lane: str,
    *,
    storage: Any | None = None,
) -> dict[str, Any] | None:
    """Read a locator only after validating its layout, path, and identity."""
    _validate_lane(lane)
    root = Path(root)
    layout = _read_layout_marker(root, storage=storage)
    path = group_locator_path(root, group, lane)
    if path.parent != _validate_selected_shard(root, group, lane):
        raise RuntimeError("Group service locator shard is unavailable")
    try:
        value = _read_json(path, storage=storage)
    except FileNotFoundError:
        return None
    return _validate_locator(root, group, lane, value, authority=layout["identity"])


def publish_group_locator_locked(
    cfg: object,
    group: str,
    lane: str,
    reason: str,
    *,
    storage: Any | None = None,
) -> dict[str, Any]:
    """Coalesce one obligation while the caller holds the Group writer fence."""
    _validate_lane(lane)
    if reason not in _REASONS[lane]:
        raise ValueError(f"unsupported reason {reason!r} for Group service lane {lane!r}")
    root = Path(cfg.shared_root)
    layout = _read_layout_marker(root, storage=storage)
    identity = dict(layout["identity"])
    if layout["identity"] != identity:
        raise RuntimeError("Group service layout identity no longer matches current authority")
    path = group_locator_path(root, group, lane)
    shard = _validate_selected_shard(root, group, lane)
    if path.parent != shard:
        raise RuntimeError("Group service locator shard is unavailable")
    previous: dict[str, Any] | None
    try:
        previous = _validate_locator(root, group, lane, _read_json(path, storage=storage), authority=identity)
    except FileNotFoundError:
        previous = None
    generation = 1 if previous is None else previous["generation"] + 1
    if generation > _MAX_GENERATION:
        raise RuntimeError("Group locator generation is exhausted")
    record = {
        "version": 1,
        "identity": {**identity, "group": group},
        "service": lane,
        "generation": generation,
        "published_at": utc_now(),
        "reason": reason,
    }
    _validate_locator(root, group, lane, record, authority=identity)
    encoded_size = len(json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8"))
    if encoded_size > MAX_RECORD_BYTES:
        raise RuntimeError("Group locator exceeds the 16 KiB record limit")
    if _authority_identity(root, storage) != identity:
        raise RuntimeError("Group service authority identity changed before locator publication")
    if storage is None:
        atomic_replace(path, record)
    else:
        storage.atomic_replace(path, record)
    persisted = _validate_locator(root, group, lane, _read_json(path, storage=storage), authority=identity)
    if persisted != record:
        raise RuntimeError("Group locator changed during publication")
    return persisted


def acknowledge_group_locator_locked(
    cfg: object,
    group: str,
    lane: str,
    generation: int,
    *,
    retirement_ready: Callable[[], bool],
) -> bool:
    """Unlink exactly the observed generation after authoritative retirement proof."""
    _validate_lane(lane)
    if type(generation) is not int or not 1 <= generation <= _MAX_GENERATION:
        raise ValueError("Group locator acknowledgement generation is invalid")
    if not callable(retirement_ready):
        raise TypeError("retirement_ready must be callable")
    root = Path(cfg.shared_root)
    observed = read_group_locator(root, group, lane)
    if observed is None or observed["generation"] != generation:
        return False
    if not retirement_ready():
        return False
    current = read_group_locator(root, group, lane)
    if (
        current is None
        or current["identity"] != observed["identity"]
        or current["service"] != observed["service"]
        or current["generation"] != generation
    ):
        return False
    path = group_locator_path(root, group, lane)
    try:
        os.unlink(path)
    except FileNotFoundError:
        return False
    _sync_directory(path.parent)
    return True


class GroupLocatorTraversal:
    """One-entry-at-a-time, lazy traversal over one locator lane."""

    def __init__(self, root: Path, lane: str):
        _validate_lane(lane)
        self._root = Path(root)
        self._lane = lane
        self._shard = 0
        self._offset = 0
        self._identity: dict[str, Any] | None = None
        self._layout_digest: str | None = None
        self._completed_passes = 0

    @property
    def completed_passes(self) -> int:
        """Return the number of complete shard passes finished by this traversal."""
        return self._completed_passes

    def advance(self) -> dict[str, Any] | None:
        """Inspect at most one directory entry and return a valid locator, if any."""
        if self._identity is None:
            layout = read_group_service_layout(self._root)
            self._identity = dict(layout["identity"])
            self._layout_digest = layout["manifest_digest"]
        else:
            # The immutable marker binds the full prevalidated layout. Rechecking
            # all 768 shards on every empty-directory step would make discovery
            # latency proportional to the layout size. The selected shard is
            # opened without following symlinks by read_directory_entry below.
            marker = group_service_root(self._root) / "layout.json"
            layout = _validate_layout_record(self._root, _read_json(marker))
            if layout["identity"] != self._identity or layout["manifest_digest"] != self._layout_digest:
                raise RuntimeError("Group service locator traversal identity changed")

        lane_root = group_service_root(self._root) / self._lane
        shard_path = lane_root / f"{self._shard:02x}"
        entry, next_offset = read_directory_entry(shard_path, self._offset)
        if entry is None:
            if self._shard == SHARD_COUNT - 1:
                self._completed_passes += 1
            self._shard = (self._shard + 1) % SHARD_COUNT
            self._offset = 0
            return None
        self._offset = next_offset
        if not entry.endswith(".json"):
            return None
        digest = entry[:-5]
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise RuntimeError("Group service shard contains a malformed locator name")
        path = shard_path / entry
        value = _read_json(path)
        if type(value) is not dict:
            raise RuntimeError("Group locator is not an object")
        identity = value.get("identity")
        group = identity.get("group") if type(identity) is dict else None
        if not isinstance(group, str) or hashlib.sha256(group.encode("utf-8")).hexdigest() != digest:
            raise RuntimeError("Group locator path does not match its identity")
        if group_locator_path(self._root, group, self._lane) != path:
            raise RuntimeError("Group locator path does not match its lane")
        return _validate_locator(self._root, group, self._lane, value)


__all__ = [
    "GroupLocatorTraversal",
    "LANES",
    "MAX_RECORD_BYTES",
    "PATH_SCHEME",
    "SHARD_COUNT",
    "acknowledge_group_locator_locked",
    "advance_group_service_layout",
    "ensure_group_service_layout",
    "group_locator_path",
    "group_service_root",
    "publish_group_locator_locked",
    "read_directory_entry",
    "read_group_locator",
    "read_group_service_layout",
    "read_group_service_layout_marker",
]
