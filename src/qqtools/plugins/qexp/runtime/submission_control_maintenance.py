"""Bounded background repair for committed Submission source proofs."""

from __future__ import annotations

import os
import stat
import threading
import time
from pathlib import Path
from typing import Any

from ..config_types import RootConfig
from . import submission_control as control
from .directory_capture import read_directory_entry
from .group_discovery.json_stream import Scanner
from .group_discovery.source_revision import BoundSource, SourceChangedError, SourceRevision
from .group_discovery.submission_projection import SubmissionProjection
from .locks import exclusive
from .paths import shared_paths, submission_path
from .records import validate_identifier
from .store import atomic_replace, require_json_size

_MAX_CONTROL_STATE_BYTES = 8 * 1024
_MAX_CHECKPOINT_BYTES = 512 * 1024
_SOURCE_STEP_BYTES = 65_536
_MAX_SCANNER_FRAGMENTS = 64
_EXPECTED_GROUP = "__qexp_control__"
_MAINTENANCE_LOCK_NAME = "maintenance.lock"
_WORKER_WAIT_SECONDS = 0.05
_WORKER_IDLE_SECONDS = 1.0


def _sync_directory(path: Path) -> None:
    """Fsync one directory after changing one of its derived records."""

    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_control_record(path: Path, *, max_bytes: int) -> dict[str, Any]:
    """Read one bounded control record through the core's strict helper."""

    return control.read_control_record(path, max_bytes=max_bytes)


def _directory_stamp(path: Path) -> list[int]:
    """Return the four scalar fields used by a resumable directory cursor."""

    value = path.stat(follow_symlinks=False)
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
        raise ValueError(f"Submission source directory is not a real directory: {path}")
    return [0, int(value.st_ino), int(value.st_mtime_ns), int(value.st_ctime_ns)]


def _revision_list(value: object) -> list[int]:
    """Validate the exact five-integer source revision representation."""

    if type(value) is not list or len(value) != 5 or not all(type(item) is int for item in value):
        raise ValueError("source revision must contain five integers")
    if value[0] != 0 or any(item < 0 for item in value[1:3]):
        raise ValueError("source revision identity and size must be nonnegative")
    return [int(item) for item in value]


def _source_revision(value: os.stat_result) -> tuple[list[int], SourceRevision]:
    """Obtain the core revision and its ``BoundSource`` representation."""

    encoded = _revision_list(control.source_revision(value))
    # The persisted protocol reserves the device field because shared roots
    # may be mounted with different client-local device numbers. BoundSource
    # still receives the actual local revision for descriptor verification.
    encoded[0] = 0
    return encoded, SourceRevision.from_stat(value)


def _noop(_value: object) -> None:
    return None


def _unlink_durable(path: Path) -> bool:
    """Unlink one derived record and fence its parent directory."""

    try:
        path.unlink()
    except FileNotFoundError:
        return False
    _sync_directory(path.parent)
    return True


def _replace_state(state: dict[str, Any], *, cursor: dict[str, Any] | None, phase: str) -> dict[str, Any]:
    """Build the core state shape while retaining its canonical root/version."""

    updated = dict(state)
    updated["state"] = phase
    updated["cursor"] = cursor
    return updated


def _cursor(value: object) -> dict[str, Any]:
    """Validate a persisted bootstrap cursor."""

    if type(value) is not dict:
        raise ValueError("submission control bootstrap cursor is invalid")
    stamp = value.get("stamp")
    offset = value.get("offset")
    current = value.get("current")
    next_offset = value.get("next_offset")
    errors = value.get("errors")
    if (
        not isinstance(stamp, list)
        or len(stamp) != 4
        or not all(type(item) is int and item >= 0 for item in stamp)
        or type(offset) is not int
        or offset < 0
        or current is not None
        and (not isinstance(current, str) or not current)
        or type(next_offset) is not int
        or next_offset < 0
        or type(errors) is not bool
    ):
        raise ValueError("submission control bootstrap cursor is invalid")
    return {
        "stamp": list(stamp),
        "offset": offset,
        "current": current,
        "next_offset": next_offset,
        "errors": errors,
    }


def _checkpoint_revision(value: object) -> list[int]:
    return _revision_list(value)


class _Checkpoint:
    """Validated parser checkpoint or a same-revision durable parser error."""

    __slots__ = ("value", "is_error", "is_changed")

    def __init__(self, value: dict[str, Any], *, is_error: bool, is_changed: bool = False) -> None:
        self.value = value
        self.is_error = is_error
        self.is_changed = is_changed


class _OperationResult:
    """Internal result used to keep cursor transitions explicit."""

    __slots__ = ("kind", "reason")

    def __init__(self, kind: str, reason: str) -> None:
        self.kind = kind
        self.reason = reason


class SubmissionControlMaintenance:
    """Advance one bounded Submission proof build or repair slice."""

    def __init__(self, cfg: RootConfig) -> None:
        if not isinstance(cfg, RootConfig):
            raise TypeError("cfg must be a RootConfig")
        self._cfg = cfg
        self._closed = False
        self._directories_ready = False
        self._pending_iterator: os.ScandirIterator[str] | None = None
        self._pending_lane = True

    @property
    def is_closed(self) -> bool:
        """Return whether this maintenance owner has released its iterator."""

        return self._closed

    def advance(self) -> dict[str, Any]:
        """Perform one metadata, directory, or source-parser maintenance slice."""

        if self._closed:
            return {"state": "closed", "reason": None}
        try:
            paths = control.control_paths(self._cfg)
            self._ensure_directories(paths)
            lock_path = paths["locks"] / _MAINTENANCE_LOCK_NAME
            with exclusive(lock_path, blocking=False) as acquired:
                if not acquired:
                    return {"state": "waiting", "reason": "maintenance_busy"}
                return self._advance_locked(paths)
        except control.SubmissionControlUnavailable as exc:
            return {"state": "waiting", "reason": f"control_unavailable:{exc}"}
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            return {"state": "waiting", "reason": f"maintenance:{type(exc).__name__}:{exc}"}

    def close(self) -> None:
        """Release the pending directory iterator owned by this instance."""

        if self._closed:
            return
        iterator = self._pending_iterator
        self._pending_iterator = None
        if iterator is not None:
            iterator.close()
        self._closed = True

    def _ensure_directories(self, paths: dict[str, Path]) -> None:
        if self._directories_ready:
            return
        control.initialize_build(self._cfg)
        self._directories_ready = True

    def _advance_locked(self, paths: dict[str, Path]) -> dict[str, Any]:
        try:
            state = control.read_control_state(self._cfg)
        except FileNotFoundError:
            state = None
        if state is None:
            control.initialize_build(self._cfg)
            state = control.read_control_state(self._cfg)
            if state is None:
                return {"state": "waiting", "reason": "control_state_absent"}
        phase = state.get("state")
        if phase not in {"building", "active"}:
            return {"state": "waiting", "reason": "control_state_unavailable"}
        if phase == "active":
            if state.get("cursor") is not None:
                return {"state": "waiting", "reason": "control_state_invalid"}
            return self._advance_pending(paths)
        use_pending = self._pending_lane
        self._pending_lane = not self._pending_lane
        if use_pending:
            return self._advance_pending(paths)
        return self._advance_bootstrap(paths, state)

    def _advance_bootstrap(self, paths: dict[str, Path], state: dict[str, Any]) -> dict[str, Any]:
        value = state.get("cursor")
        try:
            cursor = None if value is None else _cursor(value)
        except ValueError as exc:
            return {"state": "waiting", "reason": f"bootstrap_cursor_invalid:{exc}"}
        source_directory = shared_paths(self._cfg.shared_root)["submissions"]
        try:
            stamp = _directory_stamp(source_directory)
        except OSError as exc:
            return {"state": "waiting", "reason": f"source_directory_unavailable:{exc}"}
        if cursor is None:
            cursor = {"stamp": stamp, "offset": 0, "current": None, "next_offset": 0, "errors": False}
            control.write_control_state(self._cfg, _replace_state(state, cursor=cursor, phase="building"))
        elif cursor["stamp"] != stamp:
            reset = {"stamp": stamp, "offset": 0, "current": None, "next_offset": 0, "errors": False}
            control.write_control_state(self._cfg, _replace_state(state, cursor=reset, phase="building"))
            return {"state": "building", "reason": "source_directory_changed"}

        current = cursor["current"]
        if current is not None:
            return self._advance_bootstrap_operation(paths, state, cursor, current)

        try:
            name, new_offset = read_directory_entry(source_directory, cursor["offset"])
            after_stamp = _directory_stamp(source_directory)
        except OSError as exc:
            return {"state": "waiting", "reason": f"source_directory_unavailable:{exc}"}
        except (RuntimeError, ValueError) as exc:
            return {"state": "waiting", "reason": f"source_directory_invalid:{exc}"}
        if after_stamp != cursor["stamp"]:
            reset = {"stamp": after_stamp, "offset": 0, "current": None, "next_offset": 0, "errors": False}
            control.write_control_state(self._cfg, _replace_state(state, cursor=reset, phase="building"))
            return {"state": "building", "reason": "source_directory_changed"}
        if name is None:
            if cursor["errors"]:
                if new_offset != cursor["offset"]:
                    eof_cursor = dict(cursor)
                    eof_cursor["offset"] = new_offset
                    eof_cursor["next_offset"] = new_offset
                    control.write_control_state(self._cfg, _replace_state(state, cursor=eof_cursor, phase="building"))
                return {"state": "building", "reason": "source_invalid"}
            control.write_control_state(self._cfg, _replace_state(state, cursor=None, phase="active"))
            return {"state": "active", "reason": "bootstrap_complete"}

        next_cursor = dict(cursor)
        next_cursor["offset"] = new_offset
        next_cursor["next_offset"] = new_offset
        next_cursor["current"] = None
        if not name.endswith(".json"):
            control.write_control_state(self._cfg, _replace_state(state, cursor=next_cursor, phase="building"))
            return {"state": "building", "reason": "skipped_nonjson"}
        operation_id = name[:-5]
        try:
            validate_identifier(operation_id, "operation_id")
        except (TypeError, ValueError):
            next_cursor["errors"] = True
            control.write_control_state(self._cfg, _replace_state(state, cursor=next_cursor, phase="building"))
            return {"state": "building", "reason": "source_invalid"}
        next_cursor["current"] = operation_id
        control.write_control_state(self._cfg, _replace_state(state, cursor=next_cursor, phase="building"))
        return {"state": "building", "reason": "source_selected", "operation_id": operation_id}

    def _advance_bootstrap_operation(
        self,
        paths: dict[str, Path],
        state: dict[str, Any],
        cursor: dict[str, Any],
        operation_id: str,
    ) -> dict[str, Any]:
        outcome = self._process_operation(paths, operation_id)
        if outcome.kind in {"progressed", "waiting", "busy"}:
            return {"state": "building", "reason": outcome.reason, "operation_id": operation_id}
        next_cursor = dict(cursor)
        next_cursor["offset"] = cursor["next_offset"]
        next_cursor["current"] = None
        if outcome.kind == "invalid":
            next_cursor["errors"] = True
        control.write_control_state(self._cfg, _replace_state(state, cursor=next_cursor, phase="building"))
        return {"state": "building", "reason": outcome.reason, "operation_id": operation_id}

    def _advance_pending(self, paths: dict[str, Path]) -> dict[str, Any]:
        pending = paths["pending"]
        try:
            _directory_stamp(pending)
        except OSError as exc:
            return {"state": "waiting", "reason": f"pending_directory_unavailable:{exc}"}
        except ValueError as exc:
            return {"state": "waiting", "reason": f"pending_directory_invalid:{exc}"}
        if self._pending_iterator is None:
            try:
                self._pending_iterator = os.scandir(pending)
            except FileNotFoundError:
                return {"state": "waiting", "reason": "pending_directory_missing"}
            except OSError as exc:
                return {"state": "waiting", "reason": f"pending_directory_unavailable:{exc}"}
        try:
            entry = next(self._pending_iterator)
        except StopIteration:
            self._pending_iterator.close()
            self._pending_iterator = None
            return {"state": "waiting", "reason": "pending_idle"}
        name = entry.name
        if not isinstance(name, str) or not name.endswith(".json"):
            return {"state": "waiting", "reason": "pending_entry_skipped"}
        operation_id = name[:-5]
        try:
            validate_identifier(operation_id, "operation_id")
        except (TypeError, ValueError):
            return {"state": "waiting", "reason": "pending_entry_invalid"}
        outcome = self._process_operation(paths, operation_id)
        if outcome.kind == "complete":
            return {"state": "active", "reason": outcome.reason, "operation_id": operation_id}
        return {"state": "waiting", "reason": outcome.reason, "operation_id": operation_id}

    def _checkpoint_path(self, paths: dict[str, Path], operation_id: str) -> Path:
        return paths["checkpoints"] / f"{operation_id}.json"

    def _record_path(self, operation_id: str) -> Path:
        return control.record_path(self._cfg, operation_id)

    def _pending_path(self, operation_id: str) -> Path:
        return control.pending_path(self._cfg, operation_id)

    def _process_operation(self, paths: dict[str, Path], operation_id: str) -> _OperationResult:
        try:
            lock_path = control.operation_lock_path(self._cfg, operation_id)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            return _OperationResult("waiting", f"operation_lock_unavailable:{exc}")
        with exclusive(lock_path, blocking=False) as acquired:
            if not acquired:
                return _OperationResult("busy", "operation_busy")
            source = submission_path(self._cfg.shared_root, operation_id)
            try:
                source_info = source.lstat()
            except FileNotFoundError:
                self._remove_missing(paths, operation_id)
                return _OperationResult("complete", "source_missing")
            except OSError as exc:
                return _OperationResult("waiting", f"source_unreadable:{exc}")
            if stat.S_ISLNK(source_info.st_mode) or not stat.S_ISREG(source_info.st_mode):
                return _OperationResult("invalid", "source_invalid")
            try:
                revision, bound_revision = _source_revision(source_info)
            except (OSError, RuntimeError, ValueError, TypeError) as exc:
                return _OperationResult("waiting", f"source_revision_unavailable:{exc}")

            witness = self._recover_pending_witness(paths, operation_id, source, source_info)
            if witness is not None:
                return witness

            checkpoint_path = self._checkpoint_path(paths, operation_id)
            checkpoint = self._load_checkpoint(checkpoint_path, operation_id, revision)
            if checkpoint is not None and checkpoint.is_changed:
                return _OperationResult("waiting", "source_changed")
            if checkpoint is not None and checkpoint.is_error:
                return _OperationResult("invalid", "source_invalid")
            if self._receipt_matches(operation_id, revision):
                self._retire(paths, operation_id)
                return _OperationResult("complete", "proof_already_current")

            if checkpoint is None:
                try:
                    bound = BoundSource.open(source, expected_revision=bound_revision)
                except SourceChangedError:
                    self._clear_checkpoint(checkpoint_path)
                    return _OperationResult("waiting", "source_changed")
                except FileNotFoundError:
                    self._remove_missing(paths, operation_id)
                    return _OperationResult("complete", "source_missing")
                except (OSError, RuntimeError, ValueError) as exc:
                    return _OperationResult("waiting", f"source_unreadable:{exc}")
                return self._advance_source(
                    paths,
                    operation_id,
                    source,
                    bound,
                    bound_revision,
                    revision,
                    None,
                )

            try:
                bound = BoundSource.open(source, expected_revision=bound_revision)
            except SourceChangedError:
                self._clear_checkpoint(checkpoint_path)
                return _OperationResult("waiting", "source_changed")
            except (OSError, RuntimeError, ValueError) as exc:
                return _OperationResult("waiting", f"source_unreadable:{exc}")
            return self._advance_source(
                paths,
                operation_id,
                source,
                bound,
                bound_revision,
                revision,
                checkpoint.value,
            )

    def _recover_pending_witness(
        self,
        paths: dict[str, Path],
        operation_id: str,
        source: Path,
        source_info: os.stat_result,
    ) -> _OperationResult | None:
        """Recover a source rename whose proof publication was interrupted."""

        pending = self._pending_path(operation_id)
        try:
            value = _read_control_record(pending, max_bytes=_MAX_CONTROL_STATE_BYTES)
        except FileNotFoundError:
            return None
        except (OSError, RuntimeError, ValueError, TypeError, KeyError):
            return None
        if set(value) != {"version", "root", "operation_id", "state", "after_image"}:
            return None
        if value["operation_id"] != operation_id:
            return None
        if value["root"] != str(self._project_root().resolve()):
            return None
        if type(value["version"]) is not int or value["version"] != 1:
            return None
        state = value["state"]
        after_image = value["after_image"]
        if not isinstance(state, str) or state not in {"preparing", "committing", "committed", "aborted", "blocked"}:
            return None
        if (
            not isinstance(after_image, list)
            or len(after_image) != 4
            or not all(type(item) is int and item >= 0 for item in after_image)
        ):
            return None

        def matches(info: os.stat_result) -> bool:
            return [0, int(info.st_ino), int(info.st_size), int(info.st_mtime_ns)] == after_image

        if not matches(source_info):
            return None
        try:
            _sync_directory(source.parent)
            after_sync = source.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            return _OperationResult("waiting", f"source_recovery_unavailable:{exc}")
        if stat.S_ISLNK(after_sync.st_mode) or not stat.S_ISREG(after_sync.st_mode) or not matches(after_sync):
            return None
        try:
            revision, _bound_revision = _source_revision(after_sync)
            control.write_receipt(self._cfg, operation_id, state, revision)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _OperationResult("waiting", f"receipt_recovery_failed:{exc}")
        self._retire(paths, operation_id)
        return _OperationResult("complete", "proof_recovered")

    def _load_checkpoint(self, path: Path, operation_id: str, revision: list[int]) -> _Checkpoint | None:
        try:
            value = _read_control_record(path, max_bytes=_MAX_CHECKPOINT_BYTES)
        except FileNotFoundError:
            return None
        except (OSError, RuntimeError, ValueError, TypeError, KeyError):
            self._clear_checkpoint(path)
            return None
        try:
            if set(value) != {"version", "root", "operation_id", "revision", "scanner", "projection", "error"}:
                raise ValueError("checkpoint fields are invalid")
            if (
                type(value["version"]) is not int
                or value["version"] != 1
                or value["root"] != str(self._project_root().resolve())
            ):
                raise ValueError("checkpoint identity is invalid")
            if value["operation_id"] != operation_id:
                raise ValueError("checkpoint operation identity is invalid")
            stored_revision = _checkpoint_revision(value["revision"])
            error = value["error"]
            if error is not None and (not isinstance(error, str) or not error or len(error) > 2048):
                raise ValueError("checkpoint error is invalid")
            if stored_revision != revision:
                self._clear_checkpoint(path)
                return _Checkpoint({}, is_error=False, is_changed=True)
            if not isinstance(value["scanner"], dict) or not isinstance(value["projection"], dict):
                raise ValueError("checkpoint parser state is invalid")
            if error is not None:
                return _Checkpoint(value, is_error=True)
            return _Checkpoint(value, is_error=False)
        except (TypeError, ValueError, KeyError):
            self._clear_checkpoint(path)
            return None

    def _project_root(self) -> Path:
        return self._cfg.shared_root

    def _receipt_matches(self, operation_id: str, revision: list[int]) -> bool:
        try:
            value = _read_control_record(self._record_path(operation_id), max_bytes=_MAX_CONTROL_STATE_BYTES)
        except (FileNotFoundError, OSError, RuntimeError, ValueError, TypeError, KeyError):
            return False
        if set(value) != {"version", "root", "operation_id", "source_revision", "state"}:
            return False
        if value["operation_id"] != operation_id:
            return False
        if value["root"] != str(self._project_root().resolve()):
            return False
        if type(value["version"]) is not int or value["version"] != 1:
            return False
        stored = value["source_revision"]
        try:
            if _revision_list(stored) != revision:
                return False
            if value.get("state") not in {"preparing", "committing", "committed", "aborted", "blocked"}:
                return False
            return True
        except (OSError, RuntimeError, ValueError, TypeError, SourceChangedError):
            return False

    def _advance_source(
        self,
        paths: dict[str, Path],
        operation_id: str,
        source: Path,
        bound: BoundSource,
        bound_revision: SourceRevision,
        revision: list[int],
        checkpoint: dict[str, Any] | None,
    ) -> _OperationResult:
        checkpoint_path = self._checkpoint_path(paths, operation_id)
        try:
            projection = (
                SubmissionProjection.from_snapshot(
                    operation_id,
                    _EXPECTED_GROUP,
                    _noop,
                    _noop,
                    checkpoint["projection"],
                )
                if checkpoint is not None
                else SubmissionProjection(operation_id, _EXPECTED_GROUP, _noop, _noop)
            )
            scanner_snapshot = checkpoint["scanner"] if checkpoint is not None else None
            if scanner_snapshot is None:
                scanner = Scanner(bound, _noop, emit_bytes=projection.feed)
            else:
                offset = scanner_snapshot.get("offset")
                if type(offset) is not int or not 0 <= offset <= bound_revision.size:
                    raise ValueError("checkpoint scanner offset is invalid")
                bound.seek(offset)
                scanner = Scanner.from_snapshot(bound, _noop, scanner_snapshot, emit_bytes=projection.feed)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            self._clear_checkpoint(checkpoint_path)
            bound.close()
            return _OperationResult("waiting", f"checkpoint_rebuilt:{exc}")

        try:
            try:
                previous_scanner = scanner.snapshot()
                previous_projection = projection.snapshot()
            except (RuntimeError, TypeError, ValueError) as exc:
                return _OperationResult("waiting", f"checkpoint_snapshot_unavailable:{exc}")
            try:
                result = scanner.step(_SOURCE_STEP_BYTES, max_fragments=_MAX_SCANNER_FRAGMENTS)
            except (OSError, EOFError) as exc:
                try:
                    bound.verify()
                except SourceChangedError:
                    self._clear_checkpoint(checkpoint_path)
                    return _OperationResult("waiting", "source_changed")
                return _OperationResult("waiting", f"source_unreadable:{exc}")
            except (RuntimeError, TypeError, ValueError) as exc:
                try:
                    bound.verify()
                except SourceChangedError:
                    self._clear_checkpoint(checkpoint_path)
                    return _OperationResult("waiting", "source_changed")
                try:
                    envelope = self._checkpoint_value(
                        operation_id,
                        revision,
                        previous_scanner,
                        previous_projection,
                        str(exc),
                    )
                    self._write_checkpoint(checkpoint_path, envelope)
                except ValueError as checkpoint_exc:
                    if not self._write_error_checkpoint(checkpoint_path, operation_id, revision, checkpoint_exc):
                        return _OperationResult("waiting", f"checkpoint_write_failed:{checkpoint_exc}")
                return _OperationResult("invalid", "source_invalid")

            try:
                bound.verify()
            except SourceChangedError:
                self._clear_checkpoint(checkpoint_path)
                return _OperationResult("waiting", "source_changed")
            if not result.is_complete:
                try:
                    envelope = self._checkpoint_value(
                        operation_id,
                        revision,
                        scanner.snapshot(),
                        projection.snapshot(),
                        None,
                    )
                    self._write_checkpoint(checkpoint_path, envelope)
                    bound.verify()
                except SourceChangedError:
                    self._clear_checkpoint(checkpoint_path)
                    return _OperationResult("waiting", "source_changed")
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    if isinstance(exc, ValueError) and self._write_error_checkpoint(
                        checkpoint_path, operation_id, revision, exc
                    ):
                        return _OperationResult("invalid", "source_invalid")
                    return _OperationResult("waiting", f"checkpoint_write_failed:{exc}")
                return _OperationResult("progressed", "source_checkpointed")

            try:
                summary = projection.finish()
                if projection.source_schema_version != 6:
                    raise ValueError("source schema version is not 6")
                bound.verify()
            except SourceChangedError:
                self._clear_checkpoint(checkpoint_path)
                return _OperationResult("waiting", "source_changed")
            except (RuntimeError, TypeError, ValueError) as exc:
                try:
                    failed_projection = projection.snapshot()
                except (RuntimeError, TypeError, ValueError):
                    failed_projection = previous_projection
                try:
                    envelope = self._checkpoint_value(
                        operation_id,
                        revision,
                        scanner.snapshot(),
                        failed_projection,
                        str(exc),
                    )
                    self._write_checkpoint(checkpoint_path, envelope)
                except ValueError as checkpoint_exc:
                    if not self._write_error_checkpoint(checkpoint_path, operation_id, revision, checkpoint_exc):
                        return _OperationResult("waiting", f"checkpoint_write_failed:{checkpoint_exc}")
                return _OperationResult("invalid", "source_invalid")
            try:
                control.write_receipt(self._cfg, operation_id, summary.state, revision)
                bound.verify()
            except SourceChangedError:
                self._clear_checkpoint(checkpoint_path)
                return _OperationResult("waiting", "source_changed")
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                return _OperationResult("waiting", f"receipt_write_failed:{exc}")
            self._retire(paths, operation_id)
            return _OperationResult("complete", "proof_repaired")
        finally:
            bound.close()

    def _checkpoint_value(
        self,
        operation_id: str,
        revision: list[int],
        scanner: dict[str, Any],
        projection: dict[str, Any],
        error: str | None,
    ) -> dict[str, Any]:
        value = {
            "version": 1,
            "root": str(self._project_root().resolve()),
            "operation_id": operation_id,
            "revision": list(revision),
            "scanner": scanner,
            "projection": projection,
            "error": error[:2048] if isinstance(error, str) else None,
        }
        require_json_size(value, max_bytes=_MAX_CHECKPOINT_BYTES, record_type="submission_control_checkpoint")
        return value

    def _write_checkpoint(self, path: Path, value: dict[str, Any]) -> None:
        require_json_size(value, max_bytes=_MAX_CHECKPOINT_BYTES, record_type="submission_control_checkpoint")
        atomic_replace(path, value)

    def _write_error_checkpoint(
        self,
        path: Path,
        operation_id: str,
        revision: list[int],
        error: BaseException,
    ) -> bool:
        """Persist a minimal same-revision error marker when parser state is too large."""

        value = {
            "version": 1,
            "root": str(self._project_root().resolve()),
            "operation_id": operation_id,
            "revision": list(revision),
            "scanner": {},
            "projection": {},
            "error": str(error)[:2048] or "submission source is invalid",
        }
        try:
            self._write_checkpoint(path, value)
        except (OSError, RuntimeError, TypeError, ValueError):
            return False
        return True

    def _clear_checkpoint(self, path: Path) -> None:
        try:
            _unlink_durable(path)
        except FileNotFoundError:
            return

    def _retire(self, paths: dict[str, Path], operation_id: str) -> None:
        _unlink_durable(self._checkpoint_path(paths, operation_id))
        _unlink_durable(self._pending_path(operation_id))

    def _remove_missing(self, paths: dict[str, Path], operation_id: str) -> None:
        """Remove all derived evidence when its canonical source is gone."""

        _unlink_durable(self._record_path(operation_id))
        self._retire(paths, operation_id)


class MachineSubmissionControlWorker:
    """Run one retained Submission-control maintenance slice per project turn."""

    def __init__(self, runtime: Any) -> None:
        from ..agent.context import MachineRuntime

        if not isinstance(runtime, MachineRuntime):
            raise TypeError("runtime must be a MachineRuntime")
        self._runtime = runtime
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="qexp-submission-control", daemon=True)
        self._maintenance: dict[Any, SubmissionControlMaintenance] = {}
        self._cursor = 0
        self._idle_until: dict[Any, float] = {}

    @property
    def is_alive(self) -> bool:
        """Return whether the maintenance daemon is running."""

        return self._thread.is_alive()

    def start(self) -> None:
        """Start the independent Submission-control daemon."""

        self._thread.start()

    def stop(self) -> None:
        """Request stop and wait at most two seconds for iterator cleanup."""

        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    revision, bindings = self._runtime.load_registry_snapshot()
                except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                    if self._stop.wait(_WORKER_WAIT_SECONDS):
                        break
                    continue
                self._runtime.working_set.reconcile(bindings, revision=revision)
                resident = self._runtime.working_set.resident_bindings()
                enabled = [binding for binding in resident if binding.enabled]
                self._reconcile(enabled)
                retained_identities = {
                    (binding.project_id, binding.registration_generation, binding.shared_root)
                    for binding in self._maintenance
                }
                for dormant_candidate in resident:
                    identity = (
                        dormant_candidate.project_id,
                        dormant_candidate.registration_generation,
                        dormant_candidate.shared_root,
                    )
                    if not dormant_candidate.enabled and identity not in retained_identities:
                        turn = self._runtime.working_set.begin_turn(dormant_candidate, "submission")
                        self._runtime.working_set.acknowledge(turn, quiescent=True)
                binding = self._select(enabled)
                if binding is None:
                    if self._stop.wait(_WORKER_IDLE_SECONDS):
                        break
                    continue
                turn = self._runtime.working_set.begin_turn(binding, "submission")
                maintenance = self._maintenance[binding]
                try:
                    with self._runtime.binding_write_guard(binding) as eligible:
                        if eligible:
                            result = maintenance.advance()
                        else:
                            result = {"reason": "binding_ineligible"}
                except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                    result = {"reason": "maintenance_error"}
                if result.get("reason") in {"pending_idle", "binding_ineligible", "maintenance_error"}:
                    self._idle_until[binding] = time.monotonic() + _WORKER_IDLE_SECONDS
                else:
                    self._idle_until.pop(binding, None)
                self._runtime.working_set.acknowledge(
                    turn,
                    quiescent=result.get("reason") == "pending_idle",
                )
                if self._stop.wait(_WORKER_WAIT_SECONDS):
                    break
        finally:
            self._shutdown()

    def _reconcile(self, bindings: list[Any]) -> None:
        current = set(bindings)
        for binding in tuple(self._maintenance):
            if binding in current:
                continue
            maintenance = self._maintenance[binding]
            try:
                maintenance.close()
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                continue
            self._maintenance.pop(binding, None)
            self._idle_until.pop(binding, None)
        for binding in bindings:
            if binding not in self._maintenance:
                try:
                    self._maintenance[binding] = SubmissionControlMaintenance(binding.root_config())
                except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                    continue
        self._cursor = self._cursor % len(bindings) if bindings else 0

    def _select(self, bindings: list[Any]) -> Any | None:
        if not bindings:
            return None
        now = time.monotonic()
        for offset in range(len(bindings)):
            index = (self._cursor + offset) % len(bindings)
            binding = bindings[index]
            if binding not in self._maintenance or self._idle_until.get(binding, 0.0) > now:
                continue
            if not binding.enabled or binding.project_id in self._runtime.upgrade_admission_blocked_projects:
                continue
            try:
                if self._runtime.binding_state(binding) != "enabled":
                    continue
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                continue
            self._cursor = (index + 1) % len(bindings)
            return binding
        return None

    def _shutdown(self) -> None:
        for maintenance in tuple(self._maintenance.values()):
            try:
                maintenance.close()
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                pass
        self._maintenance.clear()
        self._idle_until.clear()


__all__ = ["MachineSubmissionControlWorker", "SubmissionControlMaintenance"]
