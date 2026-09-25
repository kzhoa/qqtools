"""Cooperative Group membership discovery and its machine-local worker.

The service is deliberately a locator projection.  It qualifies retained
Submission sources and publishes the resulting Group membership references;
it does not provide a cancellation or removal consumer cutover.  In
particular, a complete source sweep is never treated as Group coverage until
the coverage projection has closed every membership hole.

The service owns all source and sweep objects it creates.  A caller may
advance one service from one thread, while :class:`MachineGroupDiscoveryWorker`
keeps its services and directory iterators on its daemon thread.  This keeps
cleanup deterministic when the worker is stopped during a source projection.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ...agent.context import MachineRuntime, ProjectBinding
from ...agent.working_set import BindingTurn
from ..directory_capture import read_directory_entry
from ..group_namespace import (
    GroupNotPublished,
    GroupPublicationUnavailable,
    group_authority_identity,
    group_directory,
    is_group_authority_isolated,
    read_group,
)
from ..locks import group_writer_lock
from ..paths import group_path, shared_paths, submission_path
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
# These bounds cap cooperative work requested per slice; they are not hard
# latency guarantees for filesystem calls or synchronization.
_WORKER_WAIT_SECONDS = 0.001
_WORKER_IDLE_SECONDS = 1.0
_WORKER_COOLDOWN_SECONDS = 1.0
_LOCATOR_LANE_CYCLE = ("control", "control", "membership", "maintenance")
_MAX_RESIDENT_GROUPS = 64
_MAX_PARSER_OWNERS = 16
_MAX_CLOSE_WORK = 16
_MAX_LOCATOR_CANDIDATES = 64
_MAX_SERVICE_DESCRIPTORS = 256


@dataclass(slots=True)
class _GroupServiceEntry:
    """Retained discovery and maintenance owners for one Group key."""

    key: tuple[str, str, str, str]
    binding_key: tuple[str, str, str]
    binding: ProjectBinding
    group: str
    service: Any | None
    maintenance: Any | None
    locator_records: dict[str, dict[str, Any]] = field(default_factory=dict)
    locator_observed_at: dict[str, float] = field(default_factory=dict)
    locator_mode: bool = False
    pending_lane: str | None = None
    control_generation: int | None = None
    control_sequence: int = 1
    retry_delay: float = 1.0
    locator_retry_delay: dict[str, float] = field(default_factory=dict)
    locator_cooldown_until: dict[str, float] = field(default_factory=dict)
    service_turn: bool = True
    cooldown_until: float = 0.0
    restart_after_close: bool = False


@dataclass(slots=True)
class _CloseWork:
    """One cooperatively closed owner, advanced at most once per turn."""

    kind: str
    owner: Any
    binding_key: tuple[str, str, str] | None = None
    phase: str = "service"
    requested: bool = False


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

    with project_activation_transaction(cfg, f"group_locator_{lane}"):
        activation_path = cfg.shared_root / "schema" / "group-service.json"
        try:
            activation_record = read_json_limited(activation_path, max_bytes=_MAX_RECORD_BYTES)
        except FileNotFoundError:
            fenced = False
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            # A present but unreadable activation record cannot lower the writer
            # floor. Fail closed as though the fence had already been installed.
            fenced = True
        else:
            state = activation_record.get("state")
            fenced = state != "preparing"

        active = activation.is_group_service_active(cfg.shared_root)
        try:
            return locator.publish_group_locator_locked(cfg, group, lane, reason)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            if fenced or active:
                raise
            return None


class GroupDiscoveryService:
    """Advance one Group's source qualification and membership coverage."""

    def __init__(self, root: Path, group: str, *, mode: str = "legacy", locator_generation: int | None = None) -> None:
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
        self._debt_offset = 0

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
        self._debt_offset = next_offset
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
        return self._start_pending_candidate()

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
        candidate, debt = self._pending_candidates.popleft()
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


class MachineGroupDiscoveryWorker:
    """Run bounded Group discovery independently of machine scheduling."""

    def __init__(self, runtime: MachineRuntime) -> None:
        if not isinstance(runtime, MachineRuntime):
            raise TypeError("runtime must be a MachineRuntime")
        self._runtime = runtime
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="qexp-group-discovery",
            daemon=True,
        )
        self._entries: dict[tuple[str, str, str, str], _GroupServiceEntry] = {}
        self._entry_queue: deque[_GroupServiceEntry] = deque()
        self._restart_due: dict[tuple[str, str, str, str], _GroupServiceEntry] = {}
        self._close_queue: deque[_CloseWork] = deque()
        self._close_work_index: dict[tuple[str, int], _CloseWork] = {}
        self._closing_group_keys: set[tuple[str, str, str, str]] = set()
        self._closing_binding_keys: set[tuple[str, str, str]] = set()
        self._close_work_counts: dict[tuple[str, str, str], int] = {}
        self._bindings_by_key: dict[tuple[str, str, str], ProjectBinding] = {}
        self._locator_traversals: dict[tuple[tuple[str, str, str], str], Any] = {}
        self._lane_turns: dict[tuple[str, str, str], int] = {}
        self._group_turns: dict[tuple[str, str, str], tuple[BindingTurn, dict[str, int]]] = {}
        self._legacy_sweep_passes: dict[tuple[str, str, str], int] = {}
        self._pending_locator: tuple[str, dict[str, Any]] | None = None
        self._active_project_slice = False
        self._lane_metrics: dict[str, dict[str, int | float]] = {
            lane: {
                "pending_locators_encountered": 0,
                "resident_entries": 0,
                "completions": 0,
                "retries": 0,
                "evictions": 0,
                "bytes": 0,
                "reads": 0,
                "writes": 0,
                "descriptor_high_water": 0,
                "cap_refusals": 0,
                "oldest_pending_age_seconds": 0.0,
            }
            for lane in ("control", "membership", "maintenance")
        }

    @property
    def metrics(self) -> dict[str, dict[str, int | float]]:
        """Return the maintained counters without scanning shared history."""

        now = time.monotonic()
        for lane in self._lane_metrics:
            observed = [
                entry.locator_observed_at[lane] for entry in self._entries.values() if lane in entry.locator_observed_at
            ]
            self._lane_metrics[lane]["oldest_pending_age_seconds"] = max(
                (now - started for started in observed), default=0.0
            )
        return {lane: dict(values) for lane, values in self._lane_metrics.items()}

    @property
    def is_alive(self) -> bool:
        """Return whether the worker thread is currently running."""

        return self._thread.is_alive()

    def start(self) -> None:
        """Start the independent daemon worker."""

        self._thread.start()

    def stop(self) -> None:
        """Request stop and wait briefly without touching worker-owned state."""

        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        registry_revision: int | None = None
        bindings: list[ProjectBinding] = []
        sweeps: dict[tuple[str, str, str], SubmissionSourceSweep] = {}
        sweep_complete_at: dict[tuple[str, str, str], float] = {}
        binding_cursor = 0
        try:
            while not self._stop.is_set():
                try:
                    revision, registered = self._runtime.load_registry_snapshot()
                except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                    self._advance_close_one()
                    self._stop.wait(_WORKER_WAIT_SECONDS)
                    continue
                self._runtime.working_set.reconcile(registered, revision=revision)
                resident = self._runtime.working_set.resident_bindings()
                disabled = [binding for binding in resident if not binding.enabled]
                service_bindings = [binding for binding in resident if binding.enabled]
                if registry_revision is None or registry_revision != revision or bindings != service_bindings:
                    self._reconcile_bindings(service_bindings, sweeps)
                    sweep_complete_at = {
                        key: due for key, due in sweep_complete_at.items() if key in self._bindings_by_key
                    }
                    bindings = service_bindings
                    registry_revision = revision
                    binding_cursor %= max(1, len(bindings))

                # Cleanup receives one cooperative step even while healthy
                # Groups continue to receive their own fair work quantum.
                activity_keys: set[tuple[str, str, str]] = set()
                closed_binding = self._advance_close_one()
                if closed_binding is not None:
                    activity_keys.add(closed_binding)
                self._ack_disabled_bindings(disabled, sweeps)
                turn_binding = bindings[binding_cursor % len(bindings)] if bindings else None
                if turn_binding is not None:
                    self._ensure_group_turn(turn_binding, sweeps, sweep_complete_at)
                    retry_count_before = sum(values["retries"] for values in self._lane_metrics.values())
                selected = self._next_group(
                    bindings,
                    sweeps,
                    sweep_complete_at,
                    binding_cursor,
                )
                if bindings:
                    if selected is None:
                        binding_cursor = (binding_cursor + 1) % len(bindings)
                    else:
                        binding_cursor = (selected[0] + 1) % len(bindings)
                if selected is not None:
                    _, binding, group = selected
                    activity_keys.add(self._binding_key(binding))
                    self._admit_group(binding, group)
                    if self._pending_locator is not None:
                        lane, record = self._pending_locator
                        self._advance_active_locator(binding, group, lane, record)
                    elif not self._active_project_slice:
                        advanced_binding = self._advance_one_entry()
                        if advanced_binding is not None:
                            activity_keys.add(self._binding_key(advanced_binding))
                else:
                    advanced_binding = self._advance_one_entry()
                    if advanced_binding is not None:
                        activity_keys.add(self._binding_key(advanced_binding))
                restarted_binding = self._restart_one_due()
                if restarted_binding is not None:
                    activity_keys.add(self._binding_key(restarted_binding))
                if (
                    turn_binding is not None
                    and sum(values["retries"] for values in self._lane_metrics.values()) > retry_count_before
                ):
                    activity_keys.add(self._binding_key(turn_binding))
                for binding_key in activity_keys:
                    self._refresh_group_turn_for_key(binding_key, sweeps, sweep_complete_at)
                self._ack_group_turns([] if turn_binding is None else [turn_binding], sweeps, sweep_complete_at)
                has_work = bool(bindings or self._entries or self._restart_due or self._close_queue or sweeps)
                if self._stop.wait(_WORKER_WAIT_SECONDS if has_work else _WORKER_IDLE_SECONDS):
                    break
        finally:
            self._shutdown(sweeps)

    @staticmethod
    def _binding_key(binding: ProjectBinding) -> tuple[str, str, str]:
        return (
            binding.project_id,
            binding.registration_generation or "",
            str(binding.shared_root),
        )

    @classmethod
    def _group_key(cls, binding: ProjectBinding, group: str) -> tuple[str, str, str, str]:
        return (*cls._binding_key(binding), group)

    def _reconcile_bindings(
        self,
        registered: list[ProjectBinding],
        sweeps: dict[tuple[str, str, str], SubmissionSourceSweep],
    ) -> None:
        """Retire only binding objects that changed across a registry refresh."""

        current = {self._binding_key(binding): binding for binding in registered}
        for binding_key, previous in tuple(self._bindings_by_key.items()):
            replacement = current.get(binding_key)
            if replacement == previous:
                continue
            self._closing_binding_keys.add(binding_key)
            for key, entry in tuple(self._entries.items()):
                if entry.binding_key == binding_key:
                    self._queue_entry_close(entry, block_binding=True)
            for key, entry in tuple(self._restart_due.items()):
                if entry.binding_key == binding_key:
                    self._restart_due.pop(key, None)
            sweep = sweeps.get(binding_key)
            if sweep is not None and self._queue_sweep_close(sweep, binding_key, block_binding=True):
                sweeps.pop(binding_key, None)
            if self._close_work_counts.get(binding_key, 0) == 0:
                self._closing_binding_keys.discard(binding_key)
        for binding_key, sweep in tuple(sweeps.items()):
            if binding_key not in current and self._queue_sweep_close(sweep, binding_key, block_binding=True):
                sweeps.pop(binding_key, None)
        self._bindings_by_key = current
        self._lane_turns = {key: turn for key, turn in self._lane_turns.items() if key in current}
        self._group_turns = {key: turn for key, turn in self._group_turns.items() if key in current}
        self._legacy_sweep_passes = {key: count for key, count in self._legacy_sweep_passes.items() if key in current}
        self._locator_traversals = {
            key: traversal for key, traversal in self._locator_traversals.items() if key[0] in current
        }

    def _ensure_group_turn(
        self,
        binding: ProjectBinding,
        sweeps: dict[tuple[str, str, str], SubmissionSourceSweep],
        complete_at: dict[tuple[str, str, str], float],
    ) -> None:
        key = self._binding_key(binding)
        if key not in self._group_turns:
            self._refresh_group_turn_for_key(key, sweeps, complete_at)

    def _refresh_group_turn_for_key(
        self,
        binding_key: tuple[str, str, str],
        sweeps: dict[tuple[str, str, str], SubmissionSourceSweep],
        complete_at: dict[tuple[str, str, str], float],
    ) -> None:
        binding = self._bindings_by_key.get(binding_key)
        if binding is None:
            return
        previous = self._group_turns.get(binding_key)
        if previous is not None:
            self._runtime.working_set.acknowledge(previous[0], quiescent=False)
        from .locator import LANES as locator_lanes

        starts = {
            lane: getattr(self._locator_traversals.get((binding_key, lane)), "completed_passes", 0)
            for lane in locator_lanes
        }
        starts["legacy"] = self._legacy_sweep_passes.get(binding_key, 0)
        self._group_turns[binding_key] = (self._runtime.working_set.begin_turn(binding, "group"), starts)

    def _ack_group_turns(
        self,
        bindings: list[ProjectBinding],
        sweeps: dict[tuple[str, str, str], SubmissionSourceSweep],
        complete_at: dict[tuple[str, str, str], float],
    ) -> None:
        from .locator import LANES as locator_lanes

        resource_binding_keys = self._resource_binding_keys(sweeps)
        for binding in bindings:
            binding_key = self._binding_key(binding)
            turn_state = self._group_turns.get(binding_key)
            if turn_state is None:
                continue
            turn, starting_passes = turn_state
            if binding_key in resource_binding_keys:
                self._refresh_group_turn_for_key(binding_key, sweeps, complete_at)
                continue
            activation_state = self._activation_state(binding.shared_root)
            if activation_state == "active":
                complete = all(
                    getattr(self._locator_traversals.get((binding_key, lane)), "completed_passes", 0)
                    >= starting_passes.get(lane, 0) + 1
                    for lane in locator_lanes
                )
            elif activation_state in {None, "preparing", "fenced", "building"}:
                complete = self._legacy_sweep_passes.get(binding_key, 0) > starting_passes.get("legacy", 0)
            else:
                complete = False
            if not complete:
                continue
            if not self._runtime.working_set.acknowledge(turn, quiescent=True):
                self._refresh_group_turn_for_key(binding_key, sweeps, complete_at)

    def _ack_disabled_bindings(
        self,
        bindings: list[ProjectBinding],
        sweeps: dict[tuple[str, str, str], SubmissionSourceSweep],
    ) -> None:
        """Acknowledge disabled bindings only after their owned resources close."""

        resource_binding_keys = self._resource_binding_keys(sweeps)
        for binding in bindings:
            binding_key = self._binding_key(binding)
            if binding_key in resource_binding_keys:
                continue
            turn = self._runtime.working_set.begin_turn(binding, "group")
            self._runtime.working_set.acknowledge(turn, quiescent=True)

    def _resource_binding_keys(
        self,
        sweeps: dict[tuple[str, str, str], SubmissionSourceSweep],
    ) -> set[tuple[str, str, str]]:
        """Index binding-owned resources once for linear acknowledgement passes."""

        keys = {entry.binding_key for entry in self._entries.values()}
        keys.update(entry.binding_key for entry in self._restart_due.values())
        keys.update(key for key, count in self._close_work_counts.items() if count)
        keys.update(sweeps)
        return keys

    def _eligible_binding(self, binding: ProjectBinding) -> bool:
        """Check one binding's live eligibility without scanning the registry."""

        if not binding.enabled or binding.project_id in self._runtime.upgrade_admission_blocked_projects:
            return False
        try:
            if self._runtime.binding_state(binding) != "enabled":
                return False
            return is_group_authority_isolated(binding.shared_root)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            return False

    def _eligible_bindings(self, bindings: list[ProjectBinding]) -> list[ProjectBinding]:
        result: list[ProjectBinding] = []
        for binding in bindings:
            if self._eligible_binding(binding):
                result.append(binding)
        return result

    def _resident_entry_count(self) -> int:
        entries = {id(entry): entry for entry in self._entries.values()}
        entries.update({id(entry): entry for entry in self._restart_due.values()})
        entries.update({id(work.owner): work.owner for work in self._close_work_index.values() if work.kind == "entry"})
        return len(entries)

    def _parser_owner_count(self) -> int:
        entries = {id(entry): entry for entry in self._entries.values()}
        entries.update({id(work.owner): work.owner for work in self._close_work_index.values() if work.kind == "entry"})
        return sum(1 for entry in entries.values() if entry.service is not None and entry.service.owns_source_parser)

    @staticmethod
    def _activation_state(root: Path) -> str | None:
        path = root / "schema" / "group-service.json"
        try:
            value = read_json_limited(path, max_bytes=_MAX_RECORD_BYTES)
        except FileNotFoundError:
            return None
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            return "degraded"
        if not isinstance(value, dict):
            return "degraded"
        state = value.get("state")
        return state if state in {"preparing", "fenced", "building", "active", "degraded"} else "degraded"

    def _close_legacy_binding_owners(
        self,
        binding_key: tuple[str, str, str],
        sweeps: dict[tuple[str, str, str], SubmissionSourceSweep],
    ) -> None:
        sweep = sweeps.get(binding_key)
        if sweep is not None and self._queue_sweep_close(sweep, binding_key):
            sweeps.pop(binding_key, None)
        for entry in tuple(self._entries.values()):
            if entry.binding_key == binding_key and not entry.locator_mode:
                if len(self._close_work_index) >= _MAX_CLOSE_WORK:
                    return
                self._queue_entry_close(entry)

    def _advance_active_traversal(
        self,
        index: int,
        binding: ProjectBinding,
        key: tuple[str, str, str],
    ) -> tuple[int, ProjectBinding, str] | None:
        turn = self._lane_turns.get(key, 0)
        lane = _LOCATOR_LANE_CYCLE[turn]
        self._lane_turns[key] = (turn + 1) % len(_LOCATOR_LANE_CYCLE)
        traversal_key = (key, lane)
        try:
            from .locator import GroupLocatorTraversal

            traversal = self._locator_traversals.get(traversal_key)
            if traversal is None:
                traversal = GroupLocatorTraversal(binding.shared_root, lane)
                self._locator_traversals[traversal_key] = traversal
            record = traversal.advance()
            self._lane_metrics[lane]["reads"] += 1
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            self._lane_metrics[lane]["retries"] += 1
            return None
        if record is None:
            return None
        self._lane_metrics[lane]["pending_locators_encountered"] += 1
        self._lane_metrics[lane]["bytes"] += len(json.dumps(record, sort_keys=True, separators=(",", ":")).encode())
        group = record["identity"]["group"]
        try:
            read_group(binding.shared_root, group)
        except (FileNotFoundError, GroupNotPublished, GroupPublicationUnavailable):
            # Preserve the locator and let this lane retry after other digests.
            self._lane_metrics[lane]["retries"] += 1
            return None
        self._pending_locator = (lane, record)
        return index, binding, group

    def _next_group(
        self,
        bindings: list[ProjectBinding],
        sweeps: dict[tuple[str, str, str], SubmissionSourceSweep],
        complete_at: dict[tuple[str, str, str], float],
        start: int,
    ) -> tuple[int, ProjectBinding, str] | None:
        self._pending_locator = None
        self._active_project_slice = False
        if not bindings:
            return None
        index = start % len(bindings)
        binding = bindings[index]
        key = self._binding_key(binding)
        if key in self._closing_binding_keys:
            return None
        known_activation_state = self._activation_state(binding.shared_root)
        if (
            known_activation_state not in {"active", "degraded"}
            and key not in sweeps
            and complete_at.get(key, 0.0) > time.monotonic()
        ):
            return None
        if not self._eligible_binding(binding):
            return None
        activation_state = known_activation_state
        if activation_state == "active":
            self._active_project_slice = True
            self._close_legacy_binding_owners(key, sweeps)
            return self._advance_active_traversal(index, binding, key)
        if activation_state == "degraded":
            self._active_project_slice = True
            self._close_legacy_binding_owners(key, sweeps)
            return None
        if key not in sweeps:
            try:
                sweeps[key] = SubmissionSourceSweep(group_directory(binding.shared_root), page_size=1)
            except (OSError, RuntimeError, ValueError, TypeError):
                self._lane_metrics["control"]["retries"] += 1
                complete_at[key] = time.monotonic() + _WORKER_COOLDOWN_SECONDS
                return None
        sweep = sweeps[key]
        io = SliceIO(max_io_bytes=_SOURCE_IO_BYTES, max_operations=_SOURCE_OPERATIONS)
        try:
            step = sweep.advance(
                io,
                max_entries=1,
                soft_deadline=time.monotonic() + _SOURCE_SLICE_SECONDS,
            )
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            self._lane_metrics["control"]["retries"] += 1
            complete_at[key] = time.monotonic() + _WORKER_COOLDOWN_SECONDS
            if self._queue_sweep_close(sweep, key):
                sweeps.pop(key, None)
            return None
        if step.candidates:
            group = step.candidates[0].stem
            try:
                read_group(binding.shared_root, group)
            except (FileNotFoundError, GroupNotPublished, GroupPublicationUnavailable):
                self._lane_metrics["membership"]["retries"] += 1
                return None
            return index, binding, group
        if step.state == "complete":
            sweeps.pop(key, None)
            complete_at[key] = time.monotonic() + _WORKER_COOLDOWN_SECONDS
            self._legacy_sweep_passes[key] = self._legacy_sweep_passes.get(key, 0) + 1
        elif step.state == "failed":
            complete_at[key] = time.monotonic() + _WORKER_COOLDOWN_SECONDS
            if self._queue_sweep_close(sweep, key):
                sweeps.pop(key, None)
        return None

    def _new_entry(
        self,
        binding: ProjectBinding,
        group: str,
        *,
        locator_lane: str | None = None,
        locator_record: dict[str, Any] | None = None,
    ) -> _GroupServiceEntry:
        locator_mode = locator_record is not None
        service: GroupDiscoveryService | None = None
        maintenance: Any | None = None
        if not locator_mode:
            service = GroupDiscoveryService(binding.shared_root, group)
        elif locator_lane == "membership":
            service = GroupDiscoveryService(
                binding.shared_root,
                group,
                mode="locator",
                locator_generation=locator_record["generation"],
            )
        if not locator_mode or locator_lane == "maintenance":
            maintenance = self._new_maintenance(binding.shared_root, group)
        return _GroupServiceEntry(
            key=self._group_key(binding, group),
            binding_key=self._binding_key(binding),
            binding=binding,
            group=group,
            service=service,
            maintenance=maintenance,
            locator_records={locator_lane: locator_record} if locator_record is not None and locator_lane else {},
            locator_observed_at={locator_lane: time.monotonic()} if locator_record is not None and locator_lane else {},
            locator_mode=locator_mode,
            pending_lane=locator_lane,
        )

    @staticmethod
    def _new_maintenance(root: Path, group: str) -> Any:
        # Each retained Group owns a separately budgeted maintenance session.
        from .maintenance import GroupMaintenance

        return GroupMaintenance(root, group)

    def _admit_group(self, binding: ProjectBinding, group: str) -> None:
        key = self._group_key(binding, group)
        if self._pending_locator is not None:
            lane, record = self._pending_locator
            existing = self._entries.get(key)
            candidate_count = sum(len(entry.locator_records) for entry in self._entries.values())
            if existing is None and (
                key in self._closing_group_keys
                or self._resident_entry_count() >= _MAX_RESIDENT_GROUPS
                or candidate_count >= _MAX_LOCATOR_CANDIDATES
            ):
                self._lane_metrics[lane]["cap_refusals"] += 1
                if key not in self._closing_group_keys:
                    self._evict_one_locator_entry(exclude=key)
                return
            if (
                existing is not None
                and lane not in existing.locator_records
                and candidate_count >= _MAX_LOCATOR_CANDIDATES
            ):
                self._lane_metrics[lane]["cap_refusals"] += 1
                self._evict_one_locator_entry(exclude=key)
                return
            if existing is None:
                try:
                    existing = self._new_entry(
                        binding,
                        group,
                        locator_lane=lane,
                        locator_record=record,
                    )
                except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                    self._lane_metrics[lane]["retries"] += 1
                    return
                self._entries[key] = existing
                self._entry_queue.append(existing)
            else:
                existing.locator_mode = True
                existing.locator_records[lane] = record
                existing.locator_observed_at.setdefault(lane, time.monotonic())
                existing.pending_lane = lane
                if lane == "membership":
                    if existing.service is None:
                        existing.service = GroupDiscoveryService(
                            binding.shared_root,
                            group,
                            mode="locator",
                            locator_generation=record["generation"],
                        )
                    else:
                        existing.service.update_locator_generation(record["generation"])
                elif lane == "maintenance" and existing.maintenance is None:
                    existing.maintenance = self._new_maintenance(binding.shared_root, group)
            self._refresh_lane_residency()
            return
        if key in self._entries or key in self._restart_due or key in self._closing_group_keys:
            return
        if self._resident_entry_count() >= _MAX_RESIDENT_GROUPS:
            self._lane_metrics["membership"]["cap_refusals"] += 1
            return
        try:
            entry = self._new_entry(binding, group)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            return
        self._entries[key] = entry
        self._entry_queue.append(entry)

    def _evict_one_locator_entry(self, *, exclude: tuple[str, str, str, str]) -> bool:
        """Release one durable-locator owner so a later Group can enter the cap."""

        if len(self._close_work_index) >= _MAX_CLOSE_WORK:
            return False
        for entry in tuple(self._entry_queue):
            if entry.key == exclude or not entry.locator_mode or self._entries.get(entry.key) is not entry:
                continue
            self._queue_entry_close(entry)
            return True
        return False

    def _refresh_lane_residency(self) -> None:
        descriptors = self._service_descriptor_count()
        for lane in self._lane_metrics:
            self._lane_metrics[lane]["resident_entries"] = sum(
                lane in entry.locator_records for entry in self._entries.values()
            )
            self._lane_metrics[lane]["descriptor_high_water"] = max(
                self._lane_metrics[lane]["descriptor_high_water"], descriptors
            )

    def _restart_one_due(self) -> ProjectBinding | None:
        if not self._restart_due:
            return None
        key = next(iter(self._restart_due))
        old = self._restart_due.pop(key)
        binding = self._bindings_by_key.get(old.binding_key)
        if binding != old.binding:
            return None
        now = time.monotonic()
        if old.cooldown_until > now or not self._eligible_binding(binding):
            self._restart_due[key] = old
            return None
        try:
            read_group(binding.shared_root, old.group)
            if not group_path(binding.shared_root, old.group).exists():
                return binding
            entry = self._new_entry(binding, old.group)
        except FileNotFoundError:
            return binding
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            old.cooldown_until = now + _WORKER_COOLDOWN_SECONDS
            self._restart_due[key] = old
            return binding
        self._entries[key] = entry
        self._entry_queue.append(entry)
        return binding

    def _advance_one_entry(self) -> ProjectBinding | None:
        for _ in range(len(self._entry_queue)):
            entry = self._entry_queue.popleft()
            if self._entries.get(entry.key) is not entry:
                continue
            self._entry_queue.append(entry)
            if not self._eligible_binding(entry.binding):
                continue
            if entry.binding_key in self._closing_binding_keys:
                self._queue_entry_close(entry, block_binding=True)
                return entry.binding
            if entry.locator_mode:
                if not entry.locator_records:
                    self._queue_entry_close(entry)
                    return entry.binding
                lane = entry.pending_lane or next(iter(entry.locator_records), None)
                record = entry.locator_records.get(lane) if lane is not None else None
                if lane is not None and record is not None:
                    self._advance_active_locator(entry.binding, entry.group, lane, record)
                    return entry.binding
                continue
            try:
                read_group(entry.binding.shared_root, entry.group)
                if not group_path(entry.binding.shared_root, entry.group).exists():
                    self._queue_entry_close(entry)
                    continue
            except FileNotFoundError:
                self._queue_entry_close(entry)
                continue
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                continue
            if entry.service_turn:
                entry.service_turn = False
                if entry.cooldown_until > time.monotonic():
                    continue
                result = self._advance_current(entry.service)
                state = result.get("state")
                if state == "complete" or state == "ambiguous":
                    entry.cooldown_until = time.monotonic() + _WORKER_COOLDOWN_SECONDS
                elif state in {"error", "closed"}:
                    entry.cooldown_until = time.monotonic() + _WORKER_COOLDOWN_SECONDS
                    self._queue_entry_close(entry, restart=True)
                return entry.binding
            entry.service_turn = True
            try:
                entry.maintenance.advance()
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                # Maintenance owns its own retry state. A transient maintenance
                # failure never retires a healthy discovery service.
                return entry.binding
            return entry.binding
        return None

    def _advance_active_locator(
        self,
        binding: ProjectBinding,
        group: str,
        lane: str,
        observed: dict[str, Any],
    ) -> None:
        key = self._group_key(binding, group)
        entry = self._entries.get(key)
        if entry is None or not entry.locator_mode:
            return
        now = time.monotonic()
        if entry.locator_cooldown_until.get(lane, 0.0) > now:
            return
        from . import locator

        try:
            current = locator.read_group_locator(binding.shared_root, group, lane)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            self._schedule_entry_retry(entry, lane)
            return
        if current is None:
            entry.locator_records.pop(lane, None)
            entry.locator_observed_at.pop(lane, None)
            entry.locator_retry_delay.pop(lane, None)
            entry.locator_cooldown_until.pop(lane, None)
            entry.pending_lane = next(iter(entry.locator_records), None)
            self._refresh_lane_residency()
            if lane == "membership" and entry.service is not None:
                self._queue_entry_close(entry)
            elif not entry.locator_records:
                self._queue_entry_close(entry)
            return
        observed = current
        entry.locator_records[lane] = current
        entry.pending_lane = lane
        self._refresh_lane_residency()
        if lane == "membership":
            service = entry.service
            if service is None:
                service = GroupDiscoveryService(
                    binding.shared_root,
                    group,
                    mode="locator",
                    locator_generation=observed["generation"],
                )
                entry.service = service
            else:
                service.update_locator_generation(observed["generation"])
            parser_owners = self._parser_owner_count()
            if not service.owns_source_parser and parser_owners >= _MAX_PARSER_OWNERS:
                self._lane_metrics[lane]["cap_refusals"] += 1
                return
            if self._service_descriptor_count() >= _MAX_SERVICE_DESCRIPTORS:
                self._lane_metrics[lane]["cap_refusals"] += 1
                return
            result = self._advance_current(service)
            state = result.get("state")
            if state == "complete":
                try:
                    acknowledged = service.acknowledge_if_quiescent()
                except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                    acknowledged = False
                if acknowledged:
                    entry.locator_records.pop(lane, None)
                    entry.locator_observed_at.pop(lane, None)
                    entry.locator_retry_delay.pop(lane, None)
                    entry.locator_cooldown_until.pop(lane, None)
                    entry.pending_lane = next(iter(entry.locator_records), None)
                    entry.service = None
                    self._lane_metrics[lane]["completions"] += 1
                    self._lane_metrics[lane]["writes"] += 1
                    self._refresh_lane_residency()
                    if not entry.locator_records:
                        self._queue_entry_close(entry)
                return
            if state in {"error", "ambiguous"}:
                self._schedule_entry_retry(entry, lane)
            else:
                entry.locator_retry_delay[lane] = 1.0
                entry.locator_cooldown_until.pop(lane, None)
            return
        if lane == "control":
            if self._advance_control_locator(entry, observed):
                entry.locator_records.pop(lane, None)
                entry.locator_observed_at.pop(lane, None)
                entry.locator_retry_delay.pop(lane, None)
                entry.locator_cooldown_until.pop(lane, None)
                entry.pending_lane = next(iter(entry.locator_records), None)
                self._lane_metrics[lane]["completions"] += 1
                self._lane_metrics[lane]["writes"] += 1
                self._refresh_lane_residency()
                if not entry.locator_records:
                    self._queue_entry_close(entry)
            return
        if lane == "maintenance":
            maintenance = entry.maintenance
            if maintenance is None:
                maintenance = self._new_maintenance(binding.shared_root, group)
                entry.maintenance = maintenance
            try:
                result = maintenance.advance(locator_generation=observed["generation"])
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                self._schedule_entry_retry(entry, lane)
                return
            if result.get("state") in {"error", "ambiguous", "waiting"}:
                self._schedule_entry_retry(entry, lane)
                return
            entry.locator_retry_delay[lane] = 1.0
            entry.locator_cooldown_until.pop(lane, None)
            try:
                acknowledged = maintenance.acknowledge_if_quiescent(observed["generation"])
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                acknowledged = False
            if acknowledged:
                entry.locator_records.pop(lane, None)
                entry.locator_observed_at.pop(lane, None)
                entry.locator_retry_delay.pop(lane, None)
                entry.locator_cooldown_until.pop(lane, None)
                entry.pending_lane = next(iter(entry.locator_records), None)
                entry.maintenance = None
                self._lane_metrics[lane]["completions"] += 1
                self._lane_metrics[lane]["writes"] += 1
                self._refresh_lane_residency()
                if not entry.locator_records:
                    self._queue_entry_close(entry)
            return
        raise RuntimeError(f"unknown Group service lane {lane!r}")

    def _service_descriptor_count(self) -> int:
        entries = {id(entry): entry for entry in self._entries.values()}
        entries.update({id(work.owner): work.owner for work in self._close_work_index.values() if work.kind == "entry"})
        return sum(
            (1 if entry.service is not None and entry.service.owns_source_parser else 0)
            + (entry.maintenance.open_descriptor_count if entry.maintenance is not None else 0)
            for entry in entries.values()
        )

    def _schedule_entry_retry(self, entry: _GroupServiceEntry, lane: str) -> None:
        delay = entry.locator_retry_delay.get(lane, 1.0)
        entry.locator_cooldown_until[lane] = time.monotonic() + delay
        entry.locator_retry_delay[lane] = min(delay * 2, 60.0)
        self._lane_metrics[lane]["retries"] += 1

    def _advance_control_locator(self, entry: _GroupServiceEntry, observed: dict[str, Any]) -> bool:
        binding = entry.binding
        group = entry.group
        cfg = binding.root_config()
        try:
            from ...commands.group import reconcile_group_cancel_operations

            reconcile_group_cancel_operations(
                cfg,
                group,
                include_legacy=False,
                reservation_runtime_root=cfg.runtime_root,
            )
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            self._schedule_entry_retry(entry, "control")
            return False

        with group_writer_lock(cfg, group, blocking=False) as acquired:
            if not acquired:
                self._schedule_entry_retry(entry, "control")
                return False
            from .rechecks import GroupRechecks

            journal = GroupRechecks(cfg.shared_root, group)
            position = journal.snapshot()
            if position is None:
                entry.control_generation = None
                entry.control_sequence = 1
                return self._acknowledge_control_locked(cfg, entry, observed, journal, None)
            retention = journal.retention(position)
            if entry.control_generation != position.generation:
                entry.control_generation = position.generation
                entry.control_sequence = retention.deleted + 1
            if entry.control_sequence <= retention.deleted:
                entry.control_sequence = retention.deleted + 1
            if entry.control_sequence <= position.tail:
                sequence = entry.control_sequence
                try:
                    event = journal.read(position, sequence)
                except (OSError, RuntimeError, ValueError, KeyError):
                    self._schedule_entry_retry(entry, "control")
                    return False
                if event.get("state") == "in_flight":
                    from ..locks import task_lock
                    from .changes import settle_task_change

                    task_id = event.get("task_id")
                    if not isinstance(task_id, str):
                        self._schedule_entry_retry(entry, "control")
                        return False
                    with task_lock(cfg.shared_root, task_id, blocking=False) as has_task_lock:
                        if not has_task_lock:
                            self._schedule_entry_retry(entry, "control")
                            return False
                        settled, _results = settle_task_change(cfg, event)
                    if not settled:
                        self._schedule_entry_retry(entry, "control")
                        return False
                entry.control_sequence += 1
                entry.locator_retry_delay["control"] = 1.0
                entry.locator_cooldown_until.pop("control", None)
                return False
            return self._acknowledge_control_locked(cfg, entry, observed, journal, position)

    def _acknowledge_control_locked(
        self,
        cfg: Any,
        entry: _GroupServiceEntry,
        observed: dict[str, Any],
        journal: Any,
        position: Any | None,
    ) -> bool:
        from . import locator

        def retirement_ready() -> bool:
            latest = journal.snapshot()
            if position is None:
                if latest is not None:
                    return False
            elif (
                latest is None
                or latest.generation != position.generation
                or latest.tail != position.tail
                or entry.control_generation != latest.generation
                or entry.control_sequence <= latest.tail
            ):
                return False
            from ..operation_store import iter_active_operation_paths
            from ..store import read_json

            for operation_path in iter_active_operation_paths(cfg, "group_control", include_legacy=False):
                operation = read_json(operation_path)
                control = operation.get("group_control", {})
                if control.get("group_name") == entry.group and control.get("state") not in {
                    "completed",
                    "superseded",
                }:
                    return False
            return True

        try:
            return locator.acknowledge_group_locator_locked(
                cfg,
                entry.group,
                "control",
                observed["generation"],
                retirement_ready=retirement_ready,
            )
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            return False

    def _queue_entry_close(
        self,
        entry: _GroupServiceEntry,
        *,
        restart: bool = False,
        block_binding: bool = False,
    ) -> None:
        close_id = ("entry", id(entry))
        if close_id in self._close_work_index:
            return
        if len(self._close_work_index) >= _MAX_CLOSE_WORK:
            return
        if self._entries.get(entry.key) is entry:
            self._entries.pop(entry.key, None)
        for lane in entry.locator_records:
            self._lane_metrics[lane]["evictions"] += 1
        if block_binding:
            self._closing_binding_keys.add(entry.binding_key)
            entry.restart_after_close = False
            self._restart_due.pop(entry.key, None)
        elif restart:
            entry.restart_after_close = True
        self._close_work_index[close_id] = _CloseWork(
            "entry",
            entry,
            binding_key=entry.binding_key,
        )
        self._close_queue.append(self._close_work_index[close_id])
        self._closing_group_keys.add(entry.key)
        self._close_work_counts[entry.binding_key] = self._close_work_counts.get(entry.binding_key, 0) + 1

    def _queue_sweep_close(
        self,
        sweep: SubmissionSourceSweep,
        binding_key: tuple[str, str, str],
        *,
        block_binding: bool = False,
    ) -> bool:
        close_id = ("sweep", id(sweep))
        if close_id in self._close_work_index:
            return True
        if len(self._close_work_index) >= _MAX_CLOSE_WORK:
            return False
        if block_binding:
            self._closing_binding_keys.add(binding_key)
        work = _CloseWork("sweep", sweep, binding_key=binding_key)
        self._close_work_index[close_id] = work
        self._close_queue.append(work)
        self._close_work_counts[binding_key] = self._close_work_counts.get(binding_key, 0) + 1
        return True

    def _queue_simple_close(self, kind: str, owner: Any, binding_key: tuple[str, str, str]) -> None:
        close_id = (kind, id(owner))
        if close_id in self._close_work_index:
            return
        if len(self._close_work_index) >= _MAX_CLOSE_WORK:
            return
        work = _CloseWork(kind, owner, binding_key=binding_key)
        self._close_work_index[close_id] = work
        self._close_queue.append(work)
        self._close_work_counts[binding_key] = self._close_work_counts.get(binding_key, 0) + 1

    def _advance_close_one(self) -> tuple[str, str, str] | None:
        if not self._close_queue:
            return None
        work = self._close_queue.popleft()
        try:
            done = self._advance_close_work(work)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            done = False
        if done:
            self._finish_close_work(work)
        else:
            self._close_queue.append(work)
        return work.binding_key

    def _advance_close_work(self, work: _CloseWork) -> bool:
        if work.kind == "entry":
            entry = work.owner
            if work.phase == "service":
                if entry.service is None:
                    work.phase = "maintenance"
                    work.requested = False
                    return False
                if not work.requested:
                    entry.service.request_close()
                    work.requested = True
                if entry.service.is_closed:
                    work.phase = "maintenance"
                    work.requested = False
                    return False
                entry.service.advance()
                if entry.service.is_closed:
                    work.phase = "maintenance"
                    work.requested = False
                return False
            if entry.maintenance is None:
                return True
            if not work.requested:
                entry.maintenance.request_close()
                work.requested = True
            if entry.maintenance.is_closed:
                return True
            entry.maintenance.advance()
            return entry.maintenance.is_closed
        owner = work.owner
        if not work.requested:
            owner.request_close()
            work.requested = True
        if owner.is_closed:
            return True
        if work.kind == "sweep":
            owner.advance(
                SliceIO(max_io_bytes=_SOURCE_IO_BYTES, max_operations=_SOURCE_OPERATIONS),
                max_entries=1,
            )
        elif work.kind == "service":
            owner.advance()
        elif work.kind == "maintenance":
            owner.advance()
        else:
            raise RuntimeError(f"unknown Group discovery close work kind: {work.kind}")
        return owner.is_closed

    def _finish_close_work(self, work: _CloseWork) -> None:
        self._close_work_index.pop((work.kind, id(work.owner)), None)
        binding_key = work.binding_key
        if binding_key is not None:
            remaining = self._close_work_counts.get(binding_key, 1) - 1
            if remaining <= 0:
                self._close_work_counts.pop(binding_key, None)
                self._closing_binding_keys.discard(binding_key)
            else:
                self._close_work_counts[binding_key] = remaining
        if work.kind == "entry":
            entry = work.owner
            self._closing_group_keys.discard(entry.key)
            if (
                entry.restart_after_close
                and self._bindings_by_key.get(entry.binding_key) == entry.binding
                and entry.key not in self._entries
            ):
                self._restart_due[entry.key] = entry

    def _shutdown(self, sweeps: dict[tuple[str, str, str], SubmissionSourceSweep]) -> None:
        """Drain all remaining owners after the daemon loop stops."""

        entries: dict[int, _GroupServiceEntry] = {}
        for entry in self._entries.values():
            entries[id(entry)] = entry
        for entry in self._restart_due.values():
            entries[id(entry)] = entry
        for entry in self._entry_queue:
            entries[id(entry)] = entry
        for work in self._close_queue:
            if work.kind == "entry":
                entries[id(work.owner)] = work.owner
        for entry in entries.values():
            self._close_service(entry.service)
            self._close_maintenance(entry.maintenance)
        for work in self._close_queue:
            if work.kind == "sweep":
                self._close_sweep(work.owner)
            elif work.kind == "service":
                self._close_service(work.owner)
            elif work.kind == "maintenance":
                self._close_maintenance(work.owner)
        for sweep in sweeps.values():
            self._close_sweep(sweep)

    @staticmethod
    def _advance_current(service: GroupDiscoveryService) -> dict[str, object]:
        try:
            return service.advance()
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            return {
                "state": "error",
                "reason": f"{type(exc).__name__}: {exc}",
                "prefix": 0,
                "tail": 0,
            }

    @staticmethod
    def _should_rotate(state: object, reason: object) -> bool:
        return state in {"error", "closed"}

    @staticmethod
    def _close_service(service: GroupDiscoveryService | None) -> None:
        if service is None:
            return
        service.request_close()
        while not service.is_closed:
            try:
                service.advance()
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                if not service.is_closed:
                    time.sleep(_WORKER_WAIT_SECONDS)
        # Ownership remains on this daemon thread after stop's bounded join.

    @staticmethod
    def _close_sweep(sweep: SubmissionSourceSweep) -> None:
        sweep.request_close()
        while not sweep.is_closed:
            try:
                sweep.advance(
                    SliceIO(max_io_bytes=_SOURCE_IO_BYTES, max_operations=_SOURCE_OPERATIONS),
                    max_entries=1,
                )
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                if not sweep.is_closed:
                    time.sleep(_WORKER_WAIT_SECONDS)

    @staticmethod
    def _close_maintenance(maintenance: Any) -> None:
        if maintenance is None:
            return
        maintenance.request_close()
        while not maintenance.is_closed:
            try:
                maintenance.advance()
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                if not maintenance.is_closed:
                    time.sleep(_WORKER_WAIT_SECONDS)


__all__ = ["GroupDiscoveryService", "MachineGroupDiscoveryWorker", "publish_submission_debt"]
