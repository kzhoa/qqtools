"""Cooperative projection of one Group's membership coverage.

The service qualifies retained Submission sources and publishes membership
references. Coverage is complete when the projection closes every membership
gap. Each service owns its source and sweep objects and advances on one thread.
"""

from __future__ import annotations

import hashlib
import os
import stat
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from ..directory_capture import read_directory_entry
from ..group_namespace import group_authority_identity, is_group_authority_isolated, read_group
from ..locks import group_writer_lock
from ..paths import shared_paths, submission_path
from ..project_activation import project_activation_transaction
from ..records import validate_group_name, validate_identifier
from ..store import atomic_replace, read_json_limited, require_json_size
from .slice_io import SliceIO
from .source_revision import SourceRevision
from .source_sweep import SubmissionSourceSweep

_ACTIVE_RECORD_VERSION = 1
_BACKGROUND_RECORD_VERSION = 1
_MAX_RECORD_BYTES = 16_384
# Source slices cap bytes, operations, and cooperative parser time; they do
# not bound a blocking filesystem call's wall-clock latency.
_SOURCE_IO_BYTES = 262_144
_SOURCE_OPERATIONS = 32
_SOURCE_PROCESSED_BYTES = 65_536
_SOURCE_SLICE_SECONDS = 0.02
_MAX_PENDING_CANDIDATES = 64
_SWEEP_PAGE_SIZE = 1
_COVERAGE_PUBLISH_MEMBERS = 1
_COVERAGE_ADVANCE_MEMBERS = 64
_DEBT_DIRECTORY_NAME = "active"
_DEBT_RESCAN_SECONDS = 0.25


def _directory_revision(path: Path) -> dict[str, int] | None:
    """Return a restart-safe directory identity for a durable sweep."""

    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"Group discovery source path is not a directory: {path}")
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


class _DurableSubmissionSourceSweep:
    """One-entry sweep whose Linux directory cookie crosses worker processes."""

    def __init__(self, directory: Path, *, offset: int) -> None:
        self._directory = directory
        self.offset = offset
        self.is_closed = False
        self._close_requested = False

    def request_close(self) -> None:
        self._close_requested = True
        self.is_closed = True

    def advance(
        self,
        io: SliceIO,
        *,
        max_entries: int = 1,
        soft_deadline: float | None = None,
    ) -> Any:
        del io, soft_deadline
        if self._close_requested or self.is_closed:
            return SimpleNamespace(state="closed", candidates=(), reason=None)
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        try:
            name, next_offset = read_directory_entry(self._directory, self.offset)
        except FileNotFoundError:
            self.is_closed = True
            return SimpleNamespace(state="complete", candidates=(), reason=None)
        self.offset = next_offset
        if name is None:
            self.is_closed = True
            return SimpleNamespace(state="complete", candidates=(), reason=None)
        if not name.endswith(".json"):
            return SimpleNamespace(state="progressed", candidates=(), reason=None)
        try:
            validate_identifier(name[:-5], "submission operation id")
        except (TypeError, ValueError):
            return SimpleNamespace(state="progressed", candidates=(), reason=None)
        return SimpleNamespace(state="page", candidates=(self._directory / name,), reason=None)


def _absolute_path(value: Path, label: str) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise TypeError(f"{label} must be a path") from exc
    if isinstance(raw, bytes):
        raise TypeError(f"{label} must be a text path")
    return Path(os.path.abspath(os.path.normpath(raw)))


def _debt_bucket(group: str) -> str:
    return hashlib.sha256(group.encode("utf-8")).hexdigest()


def _debt_directory(root: Path, group: str) -> Path:
    return root / "operations" / "group-discovery" / _DEBT_DIRECTORY_NAME / _debt_bucket(group)


def _debt_path(root: Path, group: str, operation_id: str) -> Path:
    return _debt_directory(root, group) / f"{operation_id}.json"


def _revision_dict(revision: SourceRevision) -> dict[str, int]:
    return {
        "device": revision.device,
        "inode": revision.inode,
        "size": revision.size,
        "mtime_ns": revision.mtime_ns,
        "ctime_ns": revision.ctime_ns,
    }


def _revision_from_dict(value: object) -> SourceRevision:
    if type(value) is not dict or set(value) != {"device", "inode", "size", "mtime_ns", "ctime_ns"}:
        raise ValueError("active source revision has an invalid shape")
    try:
        return SourceRevision(
            device=value["device"],
            inode=value["inode"],
            size=value["size"],
            mtime_ns=value["mtime_ns"],
            ctime_ns=value["ctime_ns"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("active source revision is invalid") from exc


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _status_field(value: object, name: str, default: object = None) -> object:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _source_step_state(value: object) -> str:
    state = _status_field(value, "state")
    return state if isinstance(state, str) else "progressed"


def publish_submission_debt(root: Path, group: str, operation_id: str) -> Path | None:
    """Publish one advisory Group discovery debt record.

    The helper intentionally does nothing for the old ``groups`` namespace.
    A legacy writer therefore retains its existing path and does not silently
    create a second discovery protocol beside it.

    Args:
        root: Canonical shared qexp root.
        group: Target Group identifier.
        operation_id: Committed Submission operation identifier.

    Returns:
        The debt path when canonical Group authority is active, otherwise
        ``None``.
    """

    root = _absolute_path(root, "root")
    validate_group_name(group)
    validate_identifier(operation_id, "operation_id")
    if not is_group_authority_isolated(root):
        return None
    path = _debt_path(root, group, operation_id)
    record = {"version": 1, "group": group, "operation_id": operation_id}
    require_json_size(record, max_bytes=_MAX_RECORD_BYTES, record_type="group_discovery_debt")
    atomic_replace(path, record)
    # atomic_replace synchronizes the bucket, but mkdir(parents=True) does
    # not make its new ancestor links durable. Pending finalization may be
    # cleared only after the complete debt discovery path survives a crash.
    for directory in (path.parent.parent, path.parent.parent.parent, root / "operations"):
        _sync_directory(directory)
    return path


def publish_group_locator_for_transition(
    cfg: Any,
    group: str,
    lane: str,
    reason: str,
    *,
    before_write: Callable[[], None] | None = None,
) -> dict[str, Any] | None:
    """Dual-publish a Group service locator at an existing Group writer fence.

    Before the compatibility fence is installed, the historical discovery path
    remains authoritative and an unavailable locator layout is tolerated. Once
    activation fences old writers, locator publication becomes part of the
    transition and failure must stop the covered effect.

    The caller must already hold the schema and Group writer fences. This helper
    deliberately takes no additional lock so locator generation and retirement
    remain serialized by the existing Group lock.
    """

    if not is_group_authority_isolated(cfg.shared_root):
        return None

    if lane not in {"control", "maintenance", "membership"}:
        raise ValueError(f"unsupported Group service lane: {lane!r}")

    from . import activation, locator

    with project_activation_transaction(cfg, f"group_locator_{lane}", before_write=before_write):
        state = activation.group_service_routing_state(cfg.shared_root)
        if before_write is not None:
            before_write()
        try:
            return locator.publish_group_locator_locked(cfg, group, lane, reason)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            if state not in {None, "preparing"}:
                raise
            return None


class GroupDiscoveryService:
    """Advance one Group's source qualification and membership coverage."""

    def __init__(
        self,
        root: Path,
        group: str,
        *,
        mode: str = "legacy",
        locator_generation: int | None = None,
        continuation: dict[str, Any] | None = None,
    ) -> None:
        self._root = _absolute_path(root, "root")
        validated_group = validate_group_name(group)
        if validated_group is None:
            raise ValueError("group must be a nonempty identifier")
        self._group = validated_group
        if mode not in {"legacy", "locator"}:
            raise ValueError("Group discovery mode must be 'legacy' or 'locator'")
        if mode == "locator" and (type(locator_generation) is not int or locator_generation < 1):
            raise ValueError("locator mode requires a positive locator generation")
        self._mode = mode
        self._locator_generation = locator_generation
        self._durable_sweeps = continuation is not None
        continuation = continuation or {}
        self._bootstrap_offset = int(continuation.get("bootstrap_offset", 0))
        self._bootstrap_revision = continuation.get("bootstrap_revision")
        self._debt_offset = int(continuation.get("debt_offset", 0))
        self._debt_revision = continuation.get("debt_revision")

        self._coverage: Any | None = None
        self._coverage_directory: Path | None = None
        self._identity: object | None = None
        self._initialized = False
        self._closed = False
        self._close_requested = False

        self._background_path: Path | None = None
        self._active_path: Path | None = None
        self._bootstrap_complete = False
        self._bootstrap_debt_checked = False
        self._diagnostic_state: str | None = None
        self._diagnostic_reason: str | None = None

        self._phase = "initial"
        self._sweep: SubmissionSourceSweep | None = None
        self._sweep_kind: str | None = None
        self._sweep_complete = False
        self._sweep_missing = False
        self._next_sweep_at = 0.0
        self._pending_candidates: deque[tuple[Path, bool]] = deque(maxlen=_MAX_PENDING_CANDIDATES)

        self._active: dict[str, Any] | None = None
        self._source: Any | None = None
        self._source_close_requested = False
        self._confirmed: Any | None = None
        self._source_status: str | None = None
        self._source_error: tuple[str, str] | None = None
        self._publication_complete = False
        self._source_debt = False

        self._last_prefix = 0
        self._last_tail = 0
        self._last_status: dict[str, object] = {
            "state": "waiting",
            "reason": "initializing",
            "prefix": 0,
            "tail": 0,
        }

    @property
    def is_closed(self) -> bool:
        """Return whether the service has released all owned resources."""

        return self._closed

    @property
    def owns_source_parser(self) -> bool:
        """Whether this service currently retains its one source recovery owner."""

        return self._source is not None and not self._source.is_closed

    @property
    def continuation(self) -> dict[str, Any]:
        """Return restart-safe traversal state for an isolated worker slice."""

        sweep = self._sweep
        if isinstance(sweep, _DurableSubmissionSourceSweep) and not self._pending_candidates:
            if self._sweep_kind == "bootstrap":
                self._bootstrap_offset = sweep.offset
            elif self._sweep_kind == "debt":
                self._debt_offset = sweep.offset
        return {
            "bootstrap_offset": self._bootstrap_offset,
            "bootstrap_revision": self._bootstrap_revision,
            "debt_offset": self._debt_offset,
            "debt_revision": self._debt_revision,
        }

    def update_locator_generation(self, generation: int) -> None:
        """Refresh the generation observed by the active locator traversal."""

        if self._mode != "locator" or type(generation) is not int or generation < 1:
            raise ValueError("a positive generation is required for locator mode")
        changed = self._locator_generation is not None and generation != self._locator_generation
        self._locator_generation = generation
        if changed and self._phase == "blocked" and self._active is None and self._source is None:
            self._diagnostic_state = None
            self._diagnostic_reason = None
            self._debt_offset = 0
            self._phase = "locator_debt"

    def acknowledge_if_quiescent(self) -> bool:
        """Acknowledge this membership locator only after shared proof is complete."""

        if self._mode != "locator" or self._locator_generation is None:
            return False
        from . import activation, locator

        cfg = SimpleNamespace(shared_root=self._root)
        with group_writer_lock(cfg, self._group, blocking=False) as acquired:
            if not acquired or not activation.is_group_service_active(self._root):
                return False

            def retirement_ready() -> bool:
                if (
                    not self._initialized
                    or self._active is not None
                    or self._source is not None
                    or self._sweep is not None
                    or self._pending_candidates
                    or self._phase != "ready"
                    or self._diagnostic_state in {"error", "ambiguous"}
                ):
                    return False
                status = self._coverage_status()
                if not bool(_status_field(status, "is_complete", False)) or _status_field(status, "reason") is not None:
                    return False
                debt_directory = _debt_directory(self._root, self._group)
                try:
                    name, _offset = read_directory_entry(debt_directory, 0)
                except FileNotFoundError:
                    name = None
                if name is not None:
                    return False
                group = read_group(self._root, self._group)
                pending = group.get("group", {}).get("pending_submission_commit")
                if pending:
                    return False
                creation_owner = group.get("group", {}).get("creation_operation_id")
                if creation_owner:
                    operation = read_json_limited(
                        submission_path(self._root, creation_owner), max_bytes=_MAX_RECORD_BYTES
                    )
                    submission = operation.get("submission")
                    if not isinstance(submission, dict) or submission.get("state") not in {"committed", "aborted"}:
                        return False
                return True

            try:
                return locator.acknowledge_group_locator_locked(
                    cfg,
                    self._group,
                    "membership",
                    self._locator_generation,
                    retirement_ready=retirement_ready,
                )
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                return False

    def request_close(self) -> None:
        """Request cooperative cleanup; the next advances perform all I/O."""

        if self._closed:
            return
        self._close_requested = True
        if not self._initialized and self._source is None and self._sweep is None:
            self._closed = True

    def advance(self) -> dict[str, object]:
        """Perform one bounded discovery or coverage step."""

        if self._closed:
            return self._remember_status("closed", None)
        if self._close_requested:
            return self._advance_close()
        try:
            if not self._initialized:
                return self._initialize()
            if (
                self._mode == "locator"
                and self._active is None
                and self._source is None
                and not self._pending_candidates
                and self._phase in {"locator_debt", "ready", "coverage", "gap"}
            ):
                return self._advance_locator_debt()
            if self._diagnostic_state in {"error", "ambiguous"} and self._phase == "blocked":
                return self._remember_status(self._diagnostic_state, self._diagnostic_reason)
            if self._active is not None and self._source is None and self._phase != "blocked":
                return self._resume_active_source()
            if self._phase == "publish":
                return self._advance_publish()
            if self._phase == "close_source":
                return self._advance_source_close()
            if self._phase == "finish_source":
                return self._finish_source()
            if self._source is not None:
                return self._advance_source()
            if self._pending_candidates:
                return self._start_pending_candidate()
            if self._sweep is not None and not self._sweep_complete:
                return self._advance_sweep()
            if self._phase in {"bootstrap_sweep", "debt_sweep", "debt_check"} and self._sweep_complete:
                return self._after_sweep()
            if self._phase in {"coverage", "gap", "ready"}:
                return self._advance_coverage()
            if self._phase == "blocked":
                return self._remember_status(self._diagnostic_state or "error", self._diagnostic_reason)
            return self._advance_coverage()
        except Exception as exc:
            return self._record_error(f"{type(exc).__name__}: {exc}")

    def _initialize(self) -> dict[str, object]:
        from .coverage import GroupCoverage

        if self._mode == "locator":
            from .activation import is_group_service_active

            if not is_group_service_active(self._root):
                raise RuntimeError("locator-mode Group discovery requires active group-service-v1 evidence")

        # Construction is intentionally pure.  Canonical authority and the
        # coverage directory are validated only when the first service slice
        # is requested.
        self._identity = group_authority_identity(self._root)
        self._coverage = GroupCoverage(self._root, self._group)
        self._coverage_directory = _absolute_path(Path(self._coverage.directory), "coverage directory")
        self._coverage_directory.mkdir(parents=True, exist_ok=True)
        self._background_path = self._coverage_directory / "background.json"
        self._active_path = self._coverage_directory / "active-source.json"
        if self._mode == "legacy":
            self._load_background()
        self._load_active()
        self._initialized = True
        if self._active is not None:
            self._phase = "active_source"
            if self._mode == "legacy":
                self._start_sweep("debt" if self._bootstrap_complete else "bootstrap")
            return self._remember_status("waiting", "resuming_source")
        if self._mode == "locator":
            self._phase = "locator_debt"
            return self._remember_status("waiting", "initialized")
        self._phase = "ready" if self._bootstrap_complete else "bootstrap_sweep"
        if not self._bootstrap_complete:
            self._start_sweep("bootstrap")
        else:
            self._start_sweep("debt")
        return self._remember_status("waiting", "initialized")

    def _advance_locator_debt(self) -> dict[str, object]:
        """Inspect one durable membership debt entry without scanning history."""
        directory = _debt_directory(self._root, self._group)
        try:
            name, next_offset = read_directory_entry(directory, self._debt_offset)
        except FileNotFoundError:
            self._debt_offset = 0
            self._phase = "coverage"
            return self._advance_locator_coverage()
        if name is None:
            self._debt_offset = 0
            self._phase = "coverage"
            return self._advance_locator_coverage()
        if not name.endswith(".json"):
            return self._record_error(f"membership debt entry is malformed: {name!r}")
        try:
            validate_identifier(name[:-5], "source operation_id")
        except (TypeError, ValueError) as exc:
            return self._record_error(f"membership debt entry is malformed: {exc}")
        candidate = directory / name
        self._pending_candidates.append((candidate, True))
        self._phase = "ready"
        result = self._start_pending_candidate()
        self._debt_offset = next_offset
        return result

    def _advance_locator_coverage(self) -> dict[str, object]:
        if self._coverage is None:
            raise RuntimeError("coverage is unavailable")
        try:
            self._coverage.advance(max_members=_COVERAGE_ADVANCE_MEMBERS)
        except Exception as exc:
            return self._record_error(f"coverage advance failed: {type(exc).__name__}: {exc}")
        status = self._coverage_status()
        complete = bool(_status_field(status, "is_complete", False))
        if complete and self._diagnostic_state not in {"error", "ambiguous"}:
            self._phase = "ready"
            return self._remember_status("complete", None, status)
        self._phase = "blocked"
        reason = _status_field(status, "reason") or "membership_coverage_incomplete_without_source_debt"
        return self._record_error(str(reason))

    def _load_background(self) -> None:
        path = self._background_path
        if path is None or not path.exists():
            self._persist_background()
            return
        value = read_json_limited(path, max_bytes=_MAX_RECORD_BYTES)
        if type(value.get("version")) is not int or value["version"] != _BACKGROUND_RECORD_VERSION:
            raise ValueError("background record version is unsupported")
        if value.get("identity") != self._background_identity():
            raise ValueError("background record identity does not match canonical Group authority")
        if type(value.get("bootstrap_complete")) is not bool:
            raise ValueError("background bootstrap_complete is invalid")
        diagnostic = value.get("diagnostic")
        if type(diagnostic) is not dict:
            raise ValueError("background diagnostic is invalid")
        state = diagnostic.get("state")
        reason = diagnostic.get("reason")
        if state is not None and (not isinstance(state, str) or len(state) > 128):
            raise ValueError("background diagnostic state is invalid")
        if reason is not None and (not isinstance(reason, str) or len(reason) > 2048):
            raise ValueError("background diagnostic reason is invalid")
        self._bootstrap_complete = value["bootstrap_complete"]
        self._diagnostic_state = state
        self._diagnostic_reason = reason
        if state in {"error", "ambiguous"}:
            self._phase = "blocked"

    def _background_identity(self) -> dict[str, object]:
        identity = self._identity
        if type(identity) is not dict:
            raise RuntimeError("Group authority identity is unavailable")
        project_id = identity.get("project_id")
        directory_identity = identity.get("directory_identity")
        if type(project_id) is not str or type(directory_identity) is not dict:
            raise RuntimeError("Group authority identity is malformed")
        return {
            "project_id": project_id,
            "directory_identity": dict(directory_identity),
            "group": self._group,
        }

    def _persist_background(self) -> None:
        path = self._background_path
        if path is None:
            return
        record = {
            "version": _BACKGROUND_RECORD_VERSION,
            "identity": self._background_identity(),
            "bootstrap_complete": self._bootstrap_complete,
            "diagnostic": {"state": self._diagnostic_state, "reason": self._diagnostic_reason},
        }
        require_json_size(record, max_bytes=_MAX_RECORD_BYTES, record_type="group_discovery_background")
        atomic_replace(path, record)

    def _load_active(self) -> None:
        path = self._active_path
        if path is None or not path.exists():
            return
        value = read_json_limited(path, max_bytes=_MAX_RECORD_BYTES)
        if type(value.get("version")) is not int or value["version"] != _ACTIVE_RECORD_VERSION:
            raise ValueError("active source record version is unsupported")
        operation_id = value.get("operation_id")
        validate_identifier(operation_id, "active operation_id")
        if value.get("group") != self._group:
            raise ValueError("active source record Group does not match service")
        source = submission_path(self._root, operation_id)
        if value.get("source") != str(source):
            raise ValueError("active source record path is not the derived Submission path")
        debt = value.get("debt")
        if type(debt) is not bool:
            raise ValueError("active source record debt flag is invalid")
        revision = _revision_from_dict(value.get("revision"))
        self._active = {
            "operation_id": operation_id,
            "source": source,
            "debt": debt,
            "revision": revision,
            "debt_path": _debt_path(self._root, self._group, operation_id) if debt else None,
        }
        self._source_debt = debt

    def _persist_active(self, operation_id: str, source: Path, revision: SourceRevision, debt: bool) -> None:
        path = self._active_path
        if path is None:
            raise RuntimeError("active source path is unavailable")
        expected = submission_path(self._root, operation_id)
        if source != expected:
            raise ValueError("active source path is not the derived Submission path")
        record = {
            "version": _ACTIVE_RECORD_VERSION,
            "group": self._group,
            "operation_id": operation_id,
            "source": str(expected),
            "revision": _revision_dict(revision),
            "debt": debt,
        }
        require_json_size(record, max_bytes=_MAX_RECORD_BYTES, record_type="group_discovery_active_source")
        atomic_replace(path, record)
        self._active = {
            "operation_id": operation_id,
            "source": expected,
            "debt": debt,
            "revision": revision,
            "debt_path": _debt_path(self._root, self._group, operation_id) if debt else None,
        }
        self._source_debt = debt

    def _clear_active(self) -> None:
        path = self._active_path
        if path is None:
            raise RuntimeError("active source path is unavailable")
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        _sync_directory(path.parent)
        self._active = None
        self._source_debt = False

    def _resume_active_source(self) -> dict[str, object]:
        active = self._active
        if active is None:
            raise RuntimeError("no active source to resume")
        source = active["source"]
        operation_id = active["operation_id"]
        try:
            info = source.lstat()
        except OSError as exc:
            raise ValueError(f"active source is unavailable: {source}") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ValueError(f"active source is not a regular file: {source}")
        revision = SourceRevision.from_stat(info)
        if revision != active["revision"]:
            # A replacement must use a revision-keyed scratch path.  This
            # deliberately leaves any old publication cursor for coverage to
            # reject rather than accepting a stale irrelevant receipt.
            self._persist_active(operation_id, source, revision, bool(active["debt"]))
            active = self._active
            if active is None:
                raise RuntimeError("active source was lost while refreshing its revision")
        return self._construct_source(active, reason="active_source")

    def _construct_source(self, active: dict[str, Any], *, reason: str) -> dict[str, object]:
        from .recovery import RecoverableSource

        revision = active["revision"]
        scratch = self._coverage.source_scratch(active["operation_id"], revision=revision)
        self._source = RecoverableSource(
            active["source"],
            scratch,
            active["operation_id"],
            self._group,
        )
        self._source_close_requested = False
        self._confirmed = None
        self._source_status = None
        self._publication_complete = False
        self._source_error = None
        self._phase = "source"
        return self._remember_status("waiting", reason)

    def _start_pending_candidate(self) -> dict[str, object]:
        candidate, debt = self._pending_candidates[0]
        operation_id = candidate.stem
        validate_identifier(operation_id, "source operation_id")
        expected_candidate = (
            _debt_path(self._root, self._group, operation_id) if debt else submission_path(self._root, operation_id)
        )
        if candidate != expected_candidate:
            raise ValueError("source candidate path is not derived from its operation identifier")
        if debt:
            self._validate_debt(candidate, operation_id)
        try:
            info = candidate.lstat() if not debt else submission_path(self._root, operation_id).lstat()
        except OSError as exc:
            raise ValueError(f"source candidate is unavailable: {candidate}") from exc
        source = submission_path(self._root, operation_id)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ValueError(f"source candidate is not a regular file: {source}")
        revision = SourceRevision.from_stat(info)
        self._persist_active(operation_id, source, revision, debt)
        self._pending_candidates.popleft()
        return self._construct_source(self._active or {}, reason="source_started")

    def _validate_debt(self, path: Path, operation_id: str) -> None:
        value = read_json_limited(path, max_bytes=_MAX_RECORD_BYTES)
        if type(value.get("version")) is not int or value["version"] != 1:
            raise ValueError("debt record version is unsupported")
        if value.get("group") != self._group or value.get("operation_id") != operation_id:
            raise ValueError("debt record is foreign or inconsistent")

    def _advance_source(self) -> dict[str, object]:
        source = self._source
        if source is None:
            raise RuntimeError("source phase has no recoverable source")
        if source.is_complete:
            return self._classify_source_result()
        io = SliceIO(max_io_bytes=_SOURCE_IO_BYTES, max_operations=_SOURCE_OPERATIONS)
        try:
            step = source.advance(
                io,
                max_processed_bytes=_SOURCE_PROCESSED_BYTES,
                soft_deadline=time.monotonic() + _SOURCE_SLICE_SECONDS,
            )
        except Exception as exc:
            return self._fail_source(f"{type(exc).__name__}: {exc}")
        if source.is_complete:
            return self._classify_source_result()
        if source.is_closed:
            return self._fail_source("recoverable source closed before completion")
        state = _source_step_state(step)
        reason = _status_field(step, "reason")
        return self._remember_status("waiting" if state == "waiting" else "progressed", reason)

    def _classify_source_result(self) -> dict[str, object]:
        source = self._source
        if source is None:
            raise RuntimeError("completed source is unavailable")
        result = source.result
        if result is None:
            return self._fail_source("recoverable source completed without a result")
        status = _status_field(result, "status")
        if status not in {"qualified", "irrelevant", "ambiguous"}:
            return self._fail_source(f"recoverable source returned unknown status {status!r}")
        self._confirmed = result if status == "qualified" else None
        self._source_status = status
        if status == "qualified":
            self._phase = "publish"
            return self._remember_status("waiting", "source_qualified")
        if status == "irrelevant" and self._source_debt:
            return self._fail_source("debt_source_not_committed_for_group")
        self._source_error = ("ambiguous", f"source_{status}") if status == "ambiguous" else None
        self._phase = "close_source"
        return self._remember_status("waiting", f"source_{status}")

    def _advance_coverage(self) -> dict[str, object]:
        status = self._coverage_status()
        if self._phase == "gap" and time.monotonic() < self._next_sweep_at:
            return self._status_from_coverage(
                "waiting", status, str(_status_field(status, "reason") or "membership_gap")
            )
        if self._phase == "gap" and not self._bootstrap_complete:
            self._start_sweep("bootstrap")
            return self._remember_status("waiting", "fresh_sweep")
        if self._phase == "ready":
            self._start_sweep("debt")
            if self._sweep is not None:
                return self._remember_status("waiting", "debt_sweep")
        if self._coverage is None:
            raise RuntimeError("coverage is unavailable")
        try:
            self._coverage.advance(max_members=_COVERAGE_ADVANCE_MEMBERS)
        except Exception as exc:
            return self._record_error(f"coverage advance failed: {type(exc).__name__}: {exc}")
        status = self._coverage_status()
        complete = bool(_status_field(status, "is_complete", False))
        if complete:
            if not self._bootstrap_complete and not self._bootstrap_debt_checked:
                self._phase = "debt_check"
                self._start_sweep("debt")
                if self._sweep is None:
                    self._bootstrap_complete = True
                    self._diagnostic_state = "complete"
                    self._diagnostic_reason = None
                    self._persist_background()
                    self._phase = "ready"
                    return self._remember_status("complete", None, status)
                return self._remember_status("waiting", "debt_check", status)
            if not self._bootstrap_complete:
                self._bootstrap_complete = True
                self._diagnostic_state = "complete"
                self._diagnostic_reason = None
                self._persist_background()
            self._phase = "ready"
            return self._remember_status("complete", None, status)
        self._phase = "gap"
        self._next_sweep_at = time.monotonic() + _DEBT_RESCAN_SECONDS
        reason = _status_field(status, "reason") or "membership_gap"
        return self._status_from_coverage("waiting", status, str(reason))

    def _advance_sweep(self) -> dict[str, object]:
        sweep = self._sweep
        if sweep is None:
            self._sweep_complete = True
            return self._after_sweep()
        io = SliceIO(max_io_bytes=_SOURCE_IO_BYTES, max_operations=_SOURCE_OPERATIONS)
        # Until a candidate is persisted, restart must revisit its directory entry.
        if isinstance(sweep, _DurableSubmissionSourceSweep):
            if self._sweep_kind == "bootstrap":
                self._bootstrap_offset = sweep.offset
            elif self._sweep_kind == "debt":
                self._debt_offset = sweep.offset
        try:
            step = sweep.advance(
                io,
                max_entries=_SWEEP_PAGE_SIZE,
                soft_deadline=time.monotonic() + _SOURCE_SLICE_SECONDS,
            )
        except Exception as exc:
            return self._record_error(f"source sweep failed: {type(exc).__name__}: {exc}")
        if step.candidates:
            is_debt = self._sweep_kind == "debt"
            for candidate in step.candidates:
                if len(self._pending_candidates) >= _MAX_PENDING_CANDIDATES:
                    break
                self._pending_candidates.append((candidate, is_debt))
            return self._remember_status("progressed", "source_candidate")
        if step.state == "complete":
            self._sweep_complete = True
            return self._after_sweep()
        if step.state == "failed":
            return self._record_error(step.reason or "source sweep failed")
        return self._remember_status("waiting" if step.state == "waiting" else "progressed", step.reason)

    def _after_sweep(self) -> dict[str, object]:
        kind = self._sweep_kind
        self._sweep = None
        self._sweep_kind = None
        self._sweep_complete = False
        if kind == "bootstrap":
            self._phase = "coverage"
            return self._remember_status("waiting", "source_sweep_complete")
        if kind == "debt":
            self._bootstrap_debt_checked = True
            self._phase = "coverage"
            return self._remember_status("waiting", "debt_sweep_complete")
        return self._advance_coverage()

    def _start_sweep(self, kind: str) -> None:
        if self._sweep is not None:
            return
        if kind not in {"bootstrap", "debt"}:
            raise ValueError("unknown discovery sweep kind")
        directory = (
            shared_paths(self._root)["submissions"] if kind == "bootstrap" else _debt_directory(self._root, self._group)
        )
        if kind == "debt" and not directory.exists():
            self._sweep_missing = True
            self._sweep_complete = True
            self._sweep_kind = kind
            return
        self._sweep_missing = False
        if self._durable_sweeps:
            revision = _directory_revision(directory)
            if kind == "bootstrap":
                if revision != self._bootstrap_revision:
                    self._bootstrap_offset = 0
                    self._bootstrap_revision = revision
                offset = self._bootstrap_offset
            else:
                if revision != self._debt_revision:
                    self._debt_offset = 0
                    self._debt_revision = revision
                offset = self._debt_offset
            self._sweep = _DurableSubmissionSourceSweep(directory, offset=offset)
        else:
            self._sweep = SubmissionSourceSweep(directory, page_size=_SWEEP_PAGE_SIZE)
        self._sweep_kind = kind
        self._sweep_complete = False

    def _advance_source_close(self) -> dict[str, object]:
        source = self._source
        if source is None:
            self._phase = "finish_source"
            return self._finish_source()
        if not self._source_close_requested:
            source.request_close()
            self._source_close_requested = True
        if not source.is_closed:
            io = SliceIO(max_io_bytes=_SOURCE_IO_BYTES, max_operations=_SOURCE_OPERATIONS)
            try:
                source.advance(io, max_processed_bytes=_SOURCE_PROCESSED_BYTES, soft_deadline=None)
            except Exception as exc:
                if not source.is_closed:
                    return self._record_error(f"source close failed: {type(exc).__name__}: {exc}")
            if not source.is_closed:
                return self._remember_status("waiting", "source_closing")
        if self._source_error is not None:
            state, reason = self._source_error
            self._source = None
            self._phase = "blocked"
            self._record_diagnostic(state, reason)
            return self._remember_status(state, reason)
        if self._close_requested and not self._publication_complete and self._source_status != "irrelevant":
            # Keep the exact active-source record for a later service instance.
            # Closing a service while coverage is waiting must not discard a
            # qualified receipt or a source that was still being parsed.
            self._source = None
            self._phase = "blocked"
            return self._remember_status("waiting", "closing")
        self._phase = "finish_source"
        return self._remember_status("waiting", "source_closed")

    def _finish_source(self) -> dict[str, object]:
        if self._source is not None and not self._source.is_closed:
            self._phase = "close_source"
            return self._advance_source_close()
        if self._active is not None and self._source_status in {"qualified", "irrelevant"}:
            from .maintenance import enqueue_source_cleanup

            # Transfer temporary audit cleanup before retiring the recovery
            # locator. Cleanup has its own fair budget and cannot delay coverage.
            enqueue_source_cleanup(self._root, self._group, self._active["operation_id"], self._active["revision"])
        debt = self._source_debt
        sweep_kind = self._sweep_kind
        debt_path = self._active.get("debt_path") if self._active is not None else None
        self._source = None
        self._confirmed = None
        self._source_status = None
        self._publication_complete = False
        self._source_close_requested = False
        self._clear_active()
        if debt and debt_path is not None:
            self._delete_debt(Path(debt_path))
            if self._durable_sweeps:
                self._debt_offset = 0
                self._debt_revision = None
        if self._mode == "locator":
            # Source-debt removal changes its directory revision, so restart
            # from zero instead of trusting a cookie across that mutation.
            self._debt_offset = 0
            self._phase = "locator_debt"
            return self._remember_status("waiting", "source_finished")
        if self._sweep is not None and not self._sweep_complete and sweep_kind in {"bootstrap", "debt"}:
            self._phase = "bootstrap_sweep" if sweep_kind == "bootstrap" else "debt_sweep"
        else:
            self._phase = "coverage"
        return self._remember_status("waiting", "source_finished")

    def _advance_publish(self) -> dict[str, object]:
        if self._confirmed is None:
            raise RuntimeError("publish phase has no confirmed source")
        try:
            step = self._coverage.publish(self._confirmed, max_members=_COVERAGE_PUBLISH_MEMBERS)
        except Exception as exc:
            if type(exc).__name__ == "_PendingPublication":
                return self._remember_status("waiting", "pending_group_finalization")
            return self._record_error(f"coverage publication failed: {type(exc).__name__}: {exc}")
        state = _status_field(step, "state")
        reason = _status_field(step, "reason")
        if state == "blocked":
            return self._record_diagnostic("error", str(reason or "coverage publication blocked"))
        if state == "complete":
            self._publication_complete = True
            self._phase = "close_source"
            return self._remember_status("waiting", "publication_complete")
        if state == "waiting":
            return self._remember_status("waiting", reason or "coverage_waiting")
        return self._remember_status("progressed", reason)

    def _coverage_status(self) -> object:
        if self._coverage is None:
            raise RuntimeError("coverage is unavailable")
        status = self._coverage.status()
        prefix = _status_field(status, "prefix", 0)
        tail = _status_field(status, "tail", 0)
        if type(prefix) is not int or type(tail) is not int or prefix < 0 or tail < 0:
            raise ValueError("coverage status counters are invalid")
        self._last_prefix = prefix
        self._last_tail = tail
        return status

    def _status_from_coverage(self, state: str, status: object, reason: str | None = None) -> dict[str, object]:
        return self._remember_status(state, reason, status)

    def _remember_status(self, state: str, reason: object, coverage: object | None = None) -> dict[str, object]:
        if coverage is not None:
            self._last_prefix = int(_status_field(coverage, "prefix", self._last_prefix))
            self._last_tail = int(_status_field(coverage, "tail", self._last_tail))
        result: dict[str, object] = {
            "state": state,
            "reason": None if reason is None else str(reason),
            "prefix": self._last_prefix,
            "tail": self._last_tail,
        }
        self._last_status = result
        return result

    def _record_diagnostic(self, state: str, reason: str) -> dict[str, object]:
        self._diagnostic_state = state
        self._diagnostic_reason = reason
        self._persist_background()
        return self._remember_status(state, reason)

    def _record_error(self, reason: str) -> dict[str, object]:
        self._phase = "blocked"
        self._diagnostic_state = "error"
        self._diagnostic_reason = reason
        try:
            self._persist_background()
        except Exception as persist_exc:
            reason = f"{reason}; diagnostic_persist:{type(persist_exc).__name__}: {persist_exc}"
            self._diagnostic_reason = reason
        return self._remember_status("error", reason)

    def _fail_source(self, reason: str) -> dict[str, object]:
        self._source_error = ("error", reason)
        self._phase = "close_source" if self._source is not None and not self._source.is_closed else "blocked"
        if self._phase == "blocked":
            return self._record_diagnostic("error", reason)
        return self._advance_source_close()

    def _delete_debt(self, path: Path) -> None:
        expected = _debt_path(self._root, self._group, path.stem)
        if path != expected:
            raise ValueError("debt deletion path is not derived from its operation identifier")
        try:
            path.unlink()
        except FileNotFoundError:
            return
        _sync_directory(path.parent)

    def _advance_close(self) -> dict[str, object]:
        # Shutdown preserves the durable active locator. It is not source
        # completion and must neither retire debt nor restart a closed source.
        if self._source is not None:
            self._source.request_close()
            if not self._source.is_closed:
                self._source.advance(SliceIO(max_io_bytes=_SOURCE_IO_BYTES, max_operations=_SOURCE_OPERATIONS))
            if not self._source.is_closed:
                return self._remember_status("waiting", "closing")
            self._source = None
        if self._sweep is not None:
            self._sweep.request_close()
            if not self._sweep.is_closed:
                self._sweep.advance(SliceIO(), max_entries=1)
            if not self._sweep.is_closed:
                return self._remember_status("waiting", "closing")
            self._sweep = None
        self._closed = True
        return self._remember_status("closed", None)
