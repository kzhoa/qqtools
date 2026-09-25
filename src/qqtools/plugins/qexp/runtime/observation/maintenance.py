"""Bounded background maintenance for the query-only Task observation index."""

from __future__ import annotations

import errno
import os
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...config_types import RootConfig
from ...layout import LOCAL_RECOVERY_CAPABILITY
from ..directory_capture import read_directory_entry
from ..group_namespace import is_group_authority_isolated
from ..locks import exclusive, schema_lock, schema_writer_lock
from ..paths import shared_paths, task_path
from ..project_activation import project_activation_transaction
from ..protocol_compatibility import OBSERVATION_CAPABILITY
from ..records import TaskRecord, validate_identifier
from ..store import atomic_replace, read_json
from ..tasks import load_task
from . import projection

_OBSERVATION_LOCK_NAME = "task-observation.lock"
_MAX_GC_DEPTH = 5
_WORKER_WAIT_SECONDS = 0.05
_WORKER_IDLE_SECONDS = 0.25
_ACTIVE_POLL_SECONDS = 1.0


def _observation_lock_path(cfg: RootConfig) -> Path:
    """Return the global observation projection lock path."""

    return cfg.shared_root / "locks" / _OBSERVATION_LOCK_NAME


def _schema_path(cfg: RootConfig) -> Path:
    return shared_paths(cfg.shared_root)["schema"] / "version.json"


def _read_schema_capabilities(cfg: RootConfig) -> tuple[dict[str, Any], list[str]]:
    value = read_json(_schema_path(cfg))
    schema = value.get("schema")
    if not isinstance(schema, dict):
        raise RuntimeError("qexp schema/version.json is malformed.")
    capabilities = schema.get("required_capabilities")
    if not isinstance(capabilities, list) or not all(isinstance(item, str) for item in capabilities):
        raise RuntimeError("qexp schema/version.json has malformed required capabilities.")
    return value, capabilities


def _directory_stamp(path: Path) -> list[int]:
    """Return the identity and mutation stamp for the canonical Task directory."""

    value = path.stat(follow_symlinks=False)
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
        raise RuntimeError("Task source directory is not a real directory")
    return [int(value.st_dev), int(value.st_ino), int(value.st_mtime_ns), int(value.st_ctime_ns)]


def _revisioned(record: dict[str, Any], **changes: Any) -> dict[str, Any]:
    """Copy a projection state and advance its durable revision."""

    updated = dict(record)
    updated.update(changes)
    revision = updated.get("revision")
    if type(revision) is not int or revision < 0:
        raise ValueError("observation state revision is invalid")
    updated["revision"] = revision + 1
    return updated


def _build_cursor(value: object) -> dict[str, Any] | None:
    """Validate a resumable capture cursor without trusting its contents."""

    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("observation build cursor is invalid")
    directory = value.get("directory")
    offset = value.get("offset")
    processed = value.get("processed")
    if (
        not isinstance(directory, list)
        or len(directory) != 4
        or not all(type(item) is int and item >= 0 for item in directory)
        or type(offset) is not int
        or not 0 <= offset <= (1 << 63) - 1
        or type(processed) is not int
        or processed < 0
    ):
        raise ValueError("observation build cursor is invalid")
    return {"directory": list(directory), "offset": offset, "processed": processed}


def _build_error(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    error = value.get("error")
    return error if isinstance(error, str) and error else None


def _source_task_id(name: str) -> str | None:
    if not name.endswith(".json"):
        return None
    task_id = name[:-5]
    try:
        validate_identifier(task_id, "task_id")
        if len(task_id) > 250:
            raise ValueError("task_id exceeds the observation index limit")
        return task_id
    except (TypeError, ValueError):
        raise ValueError(f"Task source filename has an invalid identifier: {name!r}") from None


def _regular_task_path(path: Path) -> bool:
    try:
        value = path.lstat()
    except FileNotFoundError:
        raise
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
        raise ValueError(f"Task source entry is not a regular file: {path.name!r}")
    return True


@dataclass
class _DirectoryFrame:
    """One bounded, symlink-safe directory enumeration frame."""

    path: Path
    depth: int
    iterator: os.ScandirIterator[str]
    descriptor: int
    remove_after: bool

    def close(self) -> None:
        self.iterator.close()
        try:
            os.close(self.descriptor)
        except OSError:
            pass


class _GenerationGarbageCollector:
    """Reclaim one obsolete generation entry or node per maintenance slice."""

    def __init__(self, generations: Path) -> None:
        self._generations = generations
        self._keep_generation: str | None | object = _UNSET
        self._frames: list[_DirectoryFrame] = []
        self._complete = False
        self._blocked_reason: str | None = None

    @property
    def blocked_reason(self) -> str | None:
        return self._blocked_reason

    def reset(self, keep_generation: str | None) -> None:
        self.close()
        self._keep_generation = keep_generation
        self._complete = False
        self._blocked_reason = None

    def close(self) -> None:
        while self._frames:
            self._frames.pop().close()

    def advance(self, keep_generation: str | None) -> bool:
        """Perform one GC selection or unlink/rmdir and report completion."""

        if self._keep_generation is _UNSET or self._keep_generation != keep_generation:
            self.reset(keep_generation)
        if self._complete:
            return True
        if self._blocked_reason is not None:
            return False
        if not self._frames:
            try:
                generations_info = self._generations.lstat()
            except FileNotFoundError:
                self._complete = True
                return True
            if stat.S_ISLNK(generations_info.st_mode) or not stat.S_ISDIR(generations_info.st_mode):
                self._blocked_reason = f"generation_cleanup:invalid_root:{self._generations}"
                return False
            self._frames.append(self._open_frame(self._generations, 0, False))
            return False

        frame = self._frames[-1]
        try:
            entry = next(frame.iterator)
        except StopIteration:
            frame.close()
            self._frames.pop()
            if frame.remove_after:
                try:
                    parent_frame = self._frames[-1]
                    os.rmdir(frame.path.name, dir_fd=parent_frame.descriptor)
                    os.fsync(parent_frame.descriptor)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    if exc.errno == errno.ENOTEMPTY:
                        try:
                            self._frames.append(
                                self._open_frame(frame.path, frame.depth, True, dir_fd=parent_frame.descriptor)
                            )
                        except FileNotFoundError:
                            pass
                    elif exc.errno not in {errno.ENOENT, errno.ENOTDIR}:
                        self._blocked_reason = f"generation_cleanup:{type(exc).__name__}:{exc}"
                return False
            self._complete = not self._frames
            return self._complete

        name = entry.name
        child = frame.path / name
        if frame.depth == 0 and self._keep_generation is not None and name == self._keep_generation:
            return False
        try:
            info = entry.stat(follow_symlinks=False)
        except FileNotFoundError:
            return False
        if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
            depth = frame.depth + 1
            if depth > _MAX_GC_DEPTH:
                self._blocked_reason = f"generation_cleanup:depth_exceeded:{child}"
                return False
            try:
                self._frames.append(self._open_frame(child, depth, True, dir_fd=frame.descriptor))
            except FileNotFoundError:
                pass
            return False
        try:
            os.unlink(name, dir_fd=frame.descriptor)
            os.fsync(frame.descriptor)
        except FileNotFoundError:
            pass
        except IsADirectoryError:
            # A concurrent replacement can turn a selected non-directory into
            # a directory.  Revisit it on a later bounded slice.
            return False
        return False

    @staticmethod
    def _open_frame(path: Path, depth: int, remove_after: bool, *, dir_fd: int | None = None) -> _DirectoryFrame:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        descriptor = os.open(path if dir_fd is None else path.name, flags, dir_fd=dir_fd)
        try:
            iterator = os.scandir(descriptor)
        except BaseException:
            os.close(descriptor)
            raise
        return _DirectoryFrame(path, depth, iterator, descriptor, remove_after)


_UNSET = object()


def _initialize_generation(cfg: RootConfig, state: dict[str, Any]) -> None:
    """Create every empty generation partition before publishing its state."""

    projection.initialize_generation(cfg, state)


def _install_observation_capability(cfg: RootConfig) -> bool:
    """Append the observation gate while holding the exclusive schema fence."""

    path = _schema_path(cfg)
    value, capabilities = _read_schema_capabilities(cfg)
    if OBSERVATION_CAPABILITY in capabilities:
        return True
    schema = value["schema"]
    if not is_group_authority_isolated(cfg.shared_root) or LOCAL_RECOVERY_CAPABILITY not in capabilities:
        return False
    schema["required_capabilities"] = [*capabilities, OBSERVATION_CAPABILITY]
    atomic_replace(path, value)
    return True


def _observation_activation_allowed(cfg: RootConfig, capabilities: list[str]) -> bool:
    """Allow new roots or a fenced canonical old root to activate."""

    if OBSERVATION_CAPABILITY in capabilities:
        return True
    try:
        return is_group_authority_isolated(cfg.shared_root) and LOCAL_RECOVERY_CAPABILITY in capabilities
    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
        return False


def _write_rebuild_diagnostic(cfg: RootConfig, state: dict[str, Any], reason: str) -> dict[str, Any]:
    """Persist a non-queryable diagnostic without making corruption active."""

    return _revisioned(
        state,
        state="degraded",
        dirty=True,
        build={"error": reason[:2048]},
    )


class ObservationMaintenance:
    """Advance one bounded Task observation projection maintenance slice."""

    def __init__(self, cfg: RootConfig) -> None:
        if not isinstance(cfg, RootConfig):
            raise TypeError("cfg must be a RootConfig")
        self._cfg = cfg
        self._source = shared_paths(cfg.shared_root)["tasks"]
        self._gc = _GenerationGarbageCollector(projection.observation_path(cfg) / "generations")
        self._closed = False

    @property
    def is_closed(self) -> bool:
        """Return whether this maintenance owner has released its iterators."""

        return self._closed

    def advance(self) -> dict[str, Any]:
        """Perform at most one source-record capture or one generation GC step."""

        if self._closed:
            return {"state": "closed", "reason": None}
        try:
            state = projection.read_state(self._cfg)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            return {"state": "degraded", "reason": f"observation_state_unreadable:{exc}"}
        try:
            if state is None:
                result = self._initialize_absent()
                if result is not None:
                    return result
                state = projection.read_state(self._cfg)
                if state is None:
                    return {"state": "waiting", "reason": "observation_activation_deferred"}
            gate_result = self._finish_pending_gate(state)
            if gate_result is not None:
                return gate_result
            state = projection.read_state(self._cfg)
            if state is None:
                return {"state": "waiting", "reason": "observation_activation_deferred"}
            current_state = state.get("state")
            dirty = state.get("dirty")
            if current_state == "active" and dirty is False:
                return self._advance_clean_gc(state)
            if current_state == "degraded" and dirty is True and _build_error(state.get("build")) is not None:
                return {"state": "degraded", "reason": _build_error(state.get("build"))}
            if current_state not in {"building", "degraded", "active"}:
                return {"state": "degraded", "reason": "observation_state_invalid"}
            return self._advance_build(state)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            return {"state": "degraded", "reason": f"observation_maintenance:{type(exc).__name__}:{exc}"}

    def close(self) -> None:
        """Release the persistent generation scanner owned by this instance."""

        if self._closed:
            return
        self._gc.close()
        self._closed = True

    def _initialize_absent(self) -> dict[str, Any] | None:
        """Install an empty generation before fencing an existing root."""

        with schema_lock(self._cfg.shared_root, blocking=False) as acquired:
            if not acquired:
                return {"state": "waiting", "reason": "schema_busy"}
            with exclusive(_observation_lock_path(self._cfg), blocking=False) as observation_acquired:
                if not observation_acquired:
                    return {"state": "waiting", "reason": "observation_busy"}
                state = projection.read_state(self._cfg)
                if state is not None:
                    return None
                value, capabilities = _read_schema_capabilities(self._cfg)
                if not _observation_activation_allowed(self._cfg, capabilities):
                    return {"state": "waiting", "reason": "observation_activation_deferred"}
                while not self._gc.advance(None):
                    if self._gc.blocked_reason:
                        return {"state": "degraded", "reason": self._gc.blocked_reason}
                    return {"state": "waiting", "reason": "reclaiming_observation_generation"}
                new_state = projection.new_state(self._cfg, state="building")
                _initialize_generation(self._cfg, new_state)
                projection.write_state(self._cfg, new_state)
                if OBSERVATION_CAPABILITY not in capabilities:
                    value["schema"]["required_capabilities"] = [*capabilities, OBSERVATION_CAPABILITY]
                    atomic_replace(_schema_path(self._cfg), value)
                return {"state": "building", "reason": "observation_initialized"}

    def _finish_pending_gate(self, state: dict[str, Any]) -> dict[str, Any] | None:
        """Complete state-before-capability recovery before any source capture."""

        try:
            _value, capabilities = _read_schema_capabilities(self._cfg)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            return {"state": "degraded", "reason": f"observation_schema_unreadable:{exc}"}
        if OBSERVATION_CAPABILITY in capabilities:
            return None
        if state.get("state") not in {"building", "degraded", "active"}:
            return {"state": "degraded", "reason": "observation_state_invalid"}
        with schema_lock(self._cfg.shared_root, blocking=False) as acquired:
            if not acquired:
                return {"state": "waiting", "reason": "schema_busy"}
            try:
                schema_value, latest = _read_schema_capabilities(self._cfg)
                if OBSERVATION_CAPABILITY not in latest:
                    if not _observation_activation_allowed(self._cfg, latest):
                        return {"state": "waiting", "reason": "observation_activation_deferred"}
                    schema_value["schema"]["required_capabilities"] = [*latest, OBSERVATION_CAPABILITY]
                    atomic_replace(_schema_path(self._cfg), schema_value)
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                return {"state": "degraded", "reason": f"observation_gate:{exc}"}
        return None

    def _advance_clean_gc(self, state: dict[str, Any]) -> dict[str, Any]:
        with exclusive(_observation_lock_path(self._cfg), blocking=False) as acquired:
            if not acquired:
                return {"state": "waiting", "reason": "observation_busy"}
            latest = projection.read_state(self._cfg)
            if latest is None or latest.get("state") != "active" or latest.get("dirty") is not False:
                return {"state": "waiting", "reason": "observation_state_changed"}
            generation = latest.get("generation")
            if not isinstance(generation, str):
                return {"state": "degraded", "reason": "observation_generation_invalid"}
            try:
                complete = self._gc.advance(generation)
            except (OSError, RuntimeError, ValueError) as exc:
                return {"state": "degraded", "reason": f"observation_gc:{exc}"}
            if self._gc.blocked_reason:
                return {"state": "degraded", "reason": self._gc.blocked_reason}
            return {
                "state": "active",
                "reason": "idle" if complete else "garbage_collecting",
            }

    def _advance_build(self, state: dict[str, Any]) -> dict[str, Any]:
        with schema_writer_lock(self._cfg, blocking=False) as schema_acquired:
            if not schema_acquired:
                return {"state": "waiting", "reason": "schema_busy"}
            with exclusive(_observation_lock_path(self._cfg), blocking=False) as observation_acquired:
                if not observation_acquired:
                    return {"state": "waiting", "reason": "observation_busy"}
                latest = projection.read_state(self._cfg)
                if latest is None:
                    return {"state": "waiting", "reason": "observation_activation_deferred"}
                # A publisher may finish between the optimistic dirty-state
                # read and this lock. Only a still-unfinished projection rebuilds.
                if latest.get("state") == "active" and latest.get("dirty") is False:
                    return {"state": "active", "reason": "publication_completed"}
                if latest.get("state") == "degraded" and latest.get("dirty") is True:
                    error = _build_error(latest.get("build"))
                    if error is not None:
                        return {"state": "degraded", "reason": error}
                if latest.get("dirty") is True or latest.get("state") == "degraded":
                    result = self._replace_generation(latest)
                    if result is not None:
                        return result
                    latest = projection.read_state(self._cfg)
                    if latest is None:
                        return {"state": "degraded", "reason": "observation_state_lost"}
                    if latest.get("state") != "building":
                        return {"state": "degraded", "reason": "observation_rebuild_failed"}
                return self._capture_one(latest)

    def _replace_generation(self, state: dict[str, Any]) -> dict[str, Any] | None:
        """Rebuild only after every non-queryable generation is reclaimed."""

        complete = self._gc.advance(None)
        if self._gc.blocked_reason:
            return {"state": "degraded", "reason": self._gc.blocked_reason}
        if not complete:
            return {"state": "waiting", "reason": "reclaiming_observation_generation"}
        next_state = projection.new_state(self._cfg, state="building")
        try:
            _initialize_generation(self._cfg, next_state)
            projection.write_state(self._cfg, next_state)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            try:
                projection.write_state(self._cfg, _write_rebuild_diagnostic(self._cfg, state, str(exc)))
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                pass
            self._gc.reset(None)
            return {"state": "degraded", "reason": f"observation_rebuild:{exc}"}
        self._gc.reset(next_state["generation"])
        return None

    def _capture_one(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get("dirty") is not False:
            return {"state": "waiting", "reason": "observation_dirty"}
        try:
            cursor = _build_cursor(state.get("build"))
        except ValueError as exc:
            degraded = _write_rebuild_diagnostic(self._cfg, state, str(exc))
            projection.write_state(self._cfg, degraded)
            return {"state": "degraded", "reason": str(exc)}
        stamp = _directory_stamp(self._source)
        if cursor is None:
            cursor = {"directory": stamp, "offset": 0, "processed": 0}
            state = _revisioned(state, state="building", dirty=False, build=cursor)
            projection.write_state(self._cfg, state)
        elif cursor["directory"] != stamp:
            cursor = {"directory": stamp, "offset": 0, "processed": 0}
            state = _revisioned(state, state="building", dirty=False, build=cursor)
            projection.write_state(self._cfg, state)
            return {"state": "building", "reason": "task_source_changed"}

        name, new_offset = read_directory_entry(self._source, cursor["offset"])
        after_stamp = _directory_stamp(self._source)
        if after_stamp != cursor["directory"]:
            reset = {"directory": after_stamp, "offset": 0, "processed": 0}
            projection.write_state(
                self._cfg,
                _revisioned(state, state="building", dirty=False, build=reset),
            )
            return {"state": "building", "reason": "task_source_changed"}
        if name is None:
            activated = _revisioned(state, state="active", dirty=False, build=None)
            projection.write_state(self._cfg, activated)
            self._gc.reset(activated["generation"])
            return {"state": "active", "reason": "observation_build_complete"}

        next_cursor = {
            "directory": list(cursor["directory"]),
            "offset": new_offset,
            "processed": cursor["processed"] + 1,
        }
        try:
            task_id = _source_task_id(name)
        except ValueError as exc:
            reason = f"malformed_task_filename:{exc}"
            degraded = _write_rebuild_diagnostic(self._cfg, state, reason)
            projection.write_state(self._cfg, degraded)
            return {"state": "degraded", "reason": reason}
        if task_id is None:
            projection.write_state(
                self._cfg,
                _revisioned(state, state="building", dirty=False, build=next_cursor),
            )
            return {"state": "building", "reason": "skipped_nonjson"}
        path = task_path(self._cfg.shared_root, task_id)
        try:
            _regular_task_path(path)
            task = load_task(self._cfg, task_id)
        except FileNotFoundError:
            projection.write_state(
                self._cfg,
                _revisioned(state, state="building", dirty=False, build=next_cursor),
            )
            return {"state": "building", "reason": "task_disappeared"}
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            reason = f"malformed_task:{task_id}:{type(exc).__name__}:{exc}"
            degraded = _write_rebuild_diagnostic(self._cfg, state, reason)
            projection.write_state(self._cfg, degraded)
            return {"state": "degraded", "reason": reason}
        if not isinstance(task, TaskRecord) or task.task_id != task_id:
            reason = f"malformed_task:{task_id}:identity_mismatch"
            degraded = _write_rebuild_diagnostic(self._cfg, state, reason)
            projection.write_state(self._cfg, degraded)
            return {"state": "degraded", "reason": reason}

        dirty = _revisioned(state, state="building", dirty=True, build=cursor)
        projection.write_state(self._cfg, dirty)
        try:
            projection.sync_task(self._cfg, dirty, None, task)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            failure = _revisioned(dirty, state="degraded", dirty=True)
            try:
                projection.write_state(self._cfg, failure)
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                pass
            return {"state": "degraded", "reason": f"observation_publication:{exc}"}
        clean = _revisioned(dirty, state="building", dirty=False, build=next_cursor)
        projection.write_state(self._cfg, clean)
        return {"state": "building", "reason": "task_captured", "processed": next_cursor["processed"]}


def request_rebuild(cfg: RootConfig) -> dict[str, Any]:
    """Mark the observation generation non-queryable for background rebuild."""

    if not isinstance(cfg, RootConfig):
        raise TypeError("cfg must be a RootConfig")
    with schema_writer_lock(cfg, blocking=False) as schema_acquired:
        if not schema_acquired:
            return {"state": "waiting", "reason": "schema_busy"}
        with exclusive(_observation_lock_path(cfg), blocking=False) as observation_acquired:
            if not observation_acquired:
                return {"state": "waiting", "reason": "observation_busy"}
            try:
                state = projection.read_state(cfg)
            except (OSError, ValueError, TypeError):
                state = projection.new_state(cfg, state="degraded")
            if state is None:
                return {"state": "waiting", "reason": "observation_activation_deferred"}
            with project_activation_transaction(cfg, "observation_rebuild_request"):
                updated = _revisioned(state, state="degraded", dirty=True, build=None)
                projection.write_state(cfg, updated)
            return {"state": "degraded", "reason": "rebuild_requested", "revision": updated["revision"]}


class MachineObservationWorker:
    """Advance one registered project's observation maintenance per pass."""

    def __init__(self, runtime: Any) -> None:
        from ...agent.context import MachineRuntime

        if not isinstance(runtime, MachineRuntime):
            raise TypeError("runtime must be a MachineRuntime")
        self._runtime = runtime
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="qexp-task-observation", daemon=True)
        self._maintenance: dict[Any, ObservationMaintenance] = {}
        self._idle_until: dict[Any, float] = {}
        self._cursor = 0

    @property
    def is_alive(self) -> bool:
        """Return whether the worker thread is currently running."""

        return self._thread.is_alive()

    def start(self) -> None:
        """Start the independent observation daemon."""

        self._thread.start()

    def stop(self) -> None:
        """Request stop and wait up to two seconds for owned cleanup."""

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
                for binding in resident:
                    identity = (binding.project_id, binding.registration_generation, binding.shared_root)
                    if not binding.enabled and identity not in retained_identities:
                        turn = self._runtime.working_set.begin_turn(binding, "observation")
                        self._runtime.working_set.acknowledge(turn, quiescent=True)
                selected = self._select(enabled)
                if selected is None:
                    if self._stop.wait(_WORKER_IDLE_SECONDS):
                        break
                    continue
                binding = selected
                turn = self._runtime.working_set.begin_turn(binding, "observation")
                maintenance = self._maintenance[binding]
                try:
                    with self._runtime.binding_write_guard(binding) as eligible:
                        if eligible:
                            result = maintenance.advance()
                        else:
                            result = {"state": "waiting", "reason": "binding_ineligible"}
                except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                    result = {"state": "waiting", "reason": "project_observation_error"}
                if result.get("state") == "active" and result.get("reason") == "idle":
                    self._idle_until[binding] = time.monotonic() + _ACTIVE_POLL_SECONDS
                else:
                    self._idle_until.pop(binding, None)
                self._runtime.working_set.acknowledge(
                    turn,
                    quiescent=result.get("state") == "active" and result.get("reason") == "idle",
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
                    self._maintenance[binding] = ObservationMaintenance(binding.root_config())
                except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                    continue
        if bindings:
            self._cursor %= len(bindings)
        else:
            self._cursor = 0

    def _select(self, bindings: list[Any]) -> Any | None:
        if not bindings:
            return None
        now = time.monotonic()
        for offset in range(len(bindings)):
            index = (self._cursor + offset) % len(bindings)
            binding = bindings[index]
            if binding not in self._maintenance:
                continue
            if not binding.enabled or binding.project_id in self._runtime.upgrade_admission_blocked_projects:
                continue
            if self._idle_until.get(binding, 0.0) > now:
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
                continue
        self._maintenance.clear()
        self._idle_until.clear()


__all__ = ["MachineObservationWorker", "ObservationMaintenance", "request_rebuild"]
