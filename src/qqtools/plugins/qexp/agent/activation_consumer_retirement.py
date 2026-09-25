"""Crash-safe retirement intents for removed Project activation consumers."""

from __future__ import annotations

import json
import os
import re
import stat
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import Lock, RLock
from typing import TYPE_CHECKING, Any

from ..runtime.project_activation_consumers import retire_consumer_after_registration_removal
from ..runtime.store import atomic_replace
from .bindings import ProjectBinding

if TYPE_CHECKING:
    from .registration import MachineRegistration

_MAX_INTENT_BYTES = 16 * 1024
_MAX_RECOVERY_RECORDS = 16
_MAX_RECOVERY_QUEUE_ENTRIES = 64
_INTENT_DIRECTORY = "activation-consumer-retirements-v1"
_RECORD_NAME = "activation_consumer_retirement"
_RECORD_FIELDS = frozenset({"version", "runtime_id", "project_id", "registration_generation", "shared_root"})
_IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,128}\Z", re.ASCII)


@dataclass(frozen=True, slots=True)
class _RetirementIntent:
    runtime_id: str
    project_id: str
    registration_generation: str
    shared_root: str


class ActivationConsumerRetirements:
    """Store exact machine-local removal intents and retry them fairly."""

    def __init__(self, machine_root: Path) -> None:
        self.root = Path(machine_root).expanduser().resolve()
        self._lock = RLock()
        self._startup_guard = Lock()
        self._startup_recovered = False
        self._pending_paths: deque[Path] = deque()
        self._pending_set: set[Path] = set()

    def prepare(self, binding: ProjectBinding) -> None:
        """Durably record the exact consumer before local binding state is deleted."""
        intent = _intent_for_binding(binding)
        path = _intent_path(self.root, intent)
        if not _validate_directory_chain(self.root, path.parent, create=True):
            raise RuntimeError(f"activation consumer retirement directory is unavailable: {path.parent}")
        existing = _read_intent(self.root, path)
        if existing is not None and existing != intent:
            raise RuntimeError(f"activation consumer retirement intent conflicts with its identity path: {path}")
        encoded = _intent_record(intent)
        size = len(json.dumps(encoded, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8"))
        if size > _MAX_INTENT_BYTES:
            raise ValueError(f"activation consumer retirement intent exceeds {_MAX_INTENT_BYTES} bytes.")
        replaced = atomic_replace(path, encoded)
        if replaced is None or not stat.S_ISREG(replaced.st_mode) or replaced.st_size > _MAX_INTENT_BYTES:
            raise RuntimeError(f"activation consumer retirement intent could not be durably replaced: {path}")
        _sync_directory_chain_to_root(path.parent, self.root)
        self._enqueue(path)

    def complete(self, binding: ProjectBinding) -> None:
        """Retire and clear an intent after the caller committed registry removal.

        The internal caller must still hold the machine registry guard and have
        saved a registry revision in which this exact binding is absent.
        """
        intent = _intent_for_binding(binding)
        path = _intent_path(self.root, intent)
        stored = _read_intent(self.root, path)
        if stored is None:
            raise RuntimeError(f"activation consumer retirement intent is missing after registry removal: {path}")
        if stored != intent:
            raise RuntimeError(f"activation consumer retirement intent does not match the removed binding: {path}")
        self._retire_and_clear(intent)
        self._discard(path)

    def recover(self, snapshot: Sequence[ProjectBinding], registration: MachineRegistration) -> None:
        """Recover all startup absences once, then retry a fair bounded batch."""
        with self._lock:
            startup = not self._startup_recovered
        if startup:
            with self._startup_guard:
                with self._lock:
                    startup = not self._startup_recovered
                if startup:
                    paths = _list_intent_paths(self.root)
                    # Startup is explicitly allowed to inspect the full machine-local
                    # intent tree. Recheck each absence in its own short registry
                    # critical section so cross-root retirement I/O never pins the
                    # registry lock across the complete startup population.
                    for path in paths:
                        intent = _read_intent(self.root, path)
                        if intent is None:
                            continue
                        if _is_registered(intent, snapshot):
                            self._enqueue(path)
                            continue
                        with registration.registry_guard():
                            _revision, current = registration.load_registry()
                            current_intent = _read_intent(self.root, path)
                            if current_intent is None:
                                continue
                            if current_intent != intent:
                                raise RuntimeError(
                                    f"activation consumer retirement intent changed during recovery: {path}"
                                )
                            if _is_registered(current_intent, current):
                                self._enqueue(path)
                                continue
                            self._retire_and_clear(current_intent)
                    with self._lock:
                        self._startup_recovered = True
            return

        selected = self._take_batch()
        for index, path in enumerate(selected):
            try:
                intent = _read_intent(self.root, path)
                if intent is None:
                    continue
                if _is_registered(intent, snapshot):
                    self._enqueue(path)
                    continue
                # Nonblocking acquisition lets load_registry remain safe when it is
                # itself called inside an already-held registry guard.
                with registration.registry_guard(blocking=False) as acquired:
                    if not acquired:
                        self._enqueue(path)
                        continue
                    _revision, current = registration.load_registry()
                    current_intent = _read_intent(self.root, path)
                    if current_intent is None:
                        continue
                    if current_intent != intent:
                        raise RuntimeError(f"activation consumer retirement intent changed during recovery: {path}")
                    if _is_registered(current_intent, current):
                        self._enqueue(path)
                        continue
                    self._retire_and_clear(current_intent)
            except Exception:
                for unprocessed in selected[index:]:
                    self._enqueue(unprocessed)
                raise

    def recover_exact_locked(
        self,
        *,
        runtime_id: str,
        project_id: str,
        registration_generation: str,
        shared_root: Path,
        snapshot: Sequence[ProjectBinding],
    ) -> None:
        """Recover one target while the caller holds the machine registry guard."""
        intent = _RetirementIntent(
            _validate_identifier(runtime_id, "runtime_id"),
            _validate_identifier(project_id, "project_id"),
            _validate_identifier(registration_generation, "registration_generation"),
            _validate_shared_root(str(shared_root)),
        )
        path = _intent_path(self.root, intent)
        stored = _read_intent(self.root, path)
        if stored is None:
            return
        if stored != intent:
            raise RuntimeError(f"activation consumer retirement intent conflicts with its identity path: {path}")
        if _is_registered(intent, snapshot):
            return
        self._retire_and_clear(intent)
        self._discard(path)

    def _enqueue(self, path: Path) -> None:
        with self._lock:
            if path not in self._pending_set:
                self._pending_set.add(path)
                self._pending_paths.append(path)

    def _discard(self, path: Path) -> None:
        with self._lock:
            self._pending_set.discard(path)

    def _take_batch(self) -> list[Path]:
        selected: list[Path] = []
        with self._lock:
            for _ in range(_MAX_RECOVERY_QUEUE_ENTRIES):
                if not self._pending_paths or len(selected) == _MAX_RECOVERY_RECORDS:
                    break
                path = self._pending_paths.popleft()
                if path not in self._pending_set:
                    continue
                self._pending_set.remove(path)
                selected.append(path)
        return selected

    def _retire_and_clear(self, intent: _RetirementIntent) -> None:
        retire_consumer_after_registration_removal(
            Path(intent.shared_root),
            runtime_id=intent.runtime_id,
            project_id=intent.project_id,
            registration_generation=intent.registration_generation,
        )
        _clear_intent(self.root, intent)


def _intent_for_binding(binding: ProjectBinding) -> _RetirementIntent:
    runtime_id = _validate_identifier(binding.runtime_instance_id, "runtime_id")
    project_id = _validate_identifier(binding.project_id, "project_id")
    generation = _validate_identifier(binding.registration_generation, "registration_generation")
    shared_root = Path(binding.shared_root)
    if not shared_root.is_absolute() or shared_root != shared_root.resolve():
        raise ValueError("Project binding shared_root must be an absolute canonical path.")
    return _RetirementIntent(runtime_id, project_id, generation, str(shared_root))


def _intent_record(intent: _RetirementIntent) -> dict[str, Any]:
    return {
        _RECORD_NAME: {
            "version": 1,
            "runtime_id": intent.runtime_id,
            "project_id": intent.project_id,
            "registration_generation": intent.registration_generation,
            "shared_root": intent.shared_root,
        }
    }


def _intent_base(root: Path) -> Path:
    return Path(root) / "operations" / _INTENT_DIRECTORY


def _intent_path(root: Path, intent: _RetirementIntent) -> Path:
    root_key = sha256(_validate_shared_root(intent.shared_root).encode("utf-8")).hexdigest()
    return (
        _intent_base(root)
        / _validate_identifier(intent.runtime_id, "runtime_id")
        / _validate_identifier(intent.project_id, "project_id")
        / f"{_validate_identifier(intent.registration_generation, 'registration_generation')}.{root_key}.json"
    )


def _validate_identifier(value: object, label: str) -> str:
    if type(value) is not str or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must be 1 to 128 ASCII letters, digits, underscores, or hyphens.")
    return value


def _validate_shared_root(value: object) -> str:
    if type(value) is not str or not value:
        raise ValueError("activation consumer retirement shared_root must be an absolute canonical path.")
    path = Path(value)
    try:
        canonical = path.resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError("activation consumer retirement shared_root must be an absolute canonical path.") from exc
    if not path.is_absolute() or path != canonical or str(path) != value:
        raise ValueError("activation consumer retirement shared_root must be an absolute canonical path.")
    return value


def _validate_record(value: object, root: Path, path: Path) -> _RetirementIntent:
    if type(value) is not dict or set(value) != {_RECORD_NAME}:
        raise ValueError("activation consumer retirement record has missing or unknown fields.")
    record = value[_RECORD_NAME]
    if type(record) is not dict or set(record) != _RECORD_FIELDS:
        raise ValueError("activation consumer retirement intent has missing or unknown fields.")
    if type(record["version"]) is not int or record["version"] != 1:
        raise ValueError("activation consumer retirement intent version is unsupported.")
    intent = _RetirementIntent(
        _validate_identifier(record["runtime_id"], "runtime_id"),
        _validate_identifier(record["project_id"], "project_id"),
        _validate_identifier(record["registration_generation"], "registration_generation"),
        _validate_shared_root(record["shared_root"]),
    )
    if path != _intent_path(root, intent):
        raise ValueError("activation consumer retirement intent path does not match its identity.")
    return intent


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"activation consumer retirement JSON contains duplicate field {key!r}.")
        result[key] = value
    return result


def _read_intent(root: Path, path: Path) -> _RetirementIntent | None:
    if not _validate_directory_chain(root, path.parent, create=False):
        return None
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_INTENT_BYTES:
        raise RuntimeError(f"activation consumer retirement intent is not a bounded regular file: {path}")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_size > _MAX_INTENT_BYTES:
            raise RuntimeError(f"activation consumer retirement intent is not a bounded regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            encoded = handle.read(_MAX_INTENT_BYTES + 1)
        if len(encoded) > _MAX_INTENT_BYTES:
            raise ValueError(f"activation consumer retirement intent exceeds {_MAX_INTENT_BYTES} bytes.")
        value = json.loads(encoded.decode("utf-8"), object_pairs_hook=_reject_duplicate_fields)
    finally:
        os.close(descriptor)
    return _validate_record(value, root, path)


def _is_registered(intent: _RetirementIntent, bindings: Sequence[ProjectBinding]) -> bool:
    return any(
        binding.runtime_instance_id == intent.runtime_id
        and binding.project_id == intent.project_id
        and binding.registration_generation == intent.registration_generation
        and str(binding.shared_root) == intent.shared_root
        for binding in bindings
    )


def _list_intent_paths(root: Path) -> list[Path]:
    """Validate and close the complete startup intent tree."""
    base = _intent_base(root)
    if not _validate_directory_chain(root, base, create=False):
        return []
    paths: list[Path] = []
    with os.scandir(base) as runtime_entries:
        for runtime_entry in sorted(runtime_entries, key=lambda entry: entry.name):
            runtime_id = _validate_identifier(runtime_entry.name, "runtime_id")
            runtime_directory = base / runtime_id
            _require_directory(runtime_directory)
            with os.scandir(runtime_directory) as project_entries:
                for project_entry in sorted(project_entries, key=lambda entry: entry.name):
                    project_id = _validate_identifier(project_entry.name, "project_id")
                    project_directory = runtime_directory / project_id
                    _require_directory(project_directory)
                    with os.scandir(project_directory) as intent_entries:
                        for intent_entry in sorted(intent_entries, key=lambda entry: entry.name):
                            name = intent_entry.name
                            candidate = project_directory / name
                            metadata = candidate.lstat()
                            if name.startswith("."):
                                if not stat.S_ISREG(metadata.st_mode):
                                    raise RuntimeError(
                                        f"activation consumer retirement temporary file is invalid: {candidate}"
                                    )
                                continue
                            if not name.endswith(".json"):
                                raise RuntimeError(
                                    f"unexpected entry in activation consumer retirement directory: {candidate}"
                                )
                            generation, separator, root_key = name[:-5].rpartition(".")
                            _validate_identifier(generation, "registration_generation")
                            if not separator or not re.fullmatch(r"[0-9a-f]{64}", root_key, re.ASCII):
                                raise ValueError(f"activation consumer retirement filename is invalid: {candidate}")
                            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_INTENT_BYTES:
                                raise RuntimeError(
                                    f"activation consumer retirement intent is not a bounded regular file: {candidate}"
                                )
                            paths.append(candidate)
    return paths


def _require_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"activation consumer retirement directory disappeared: {path}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"activation consumer retirement path is not a real directory: {path}")


def _validate_directory_chain(root: Path, directory: Path, *, create: bool) -> bool:
    root = Path(root)
    directory = Path(directory)
    try:
        root_metadata = root.lstat()
    except FileNotFoundError:
        if create:
            raise RuntimeError(f"machine runtime root is missing: {root}") from None
        return False
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise RuntimeError(f"machine runtime root is not a real directory: {root}")
    try:
        parts = directory.relative_to(root).parts
    except ValueError as exc:
        raise ValueError("activation consumer retirement directory must be below the machine runtime root.") from exc
    current = root
    for part in parts:
        current /= part
        if create:
            try:
                current.mkdir()
            except FileExistsError:
                pass
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if create:
                raise RuntimeError(f"activation consumer retirement directory is missing: {current}") from None
            return False
        if not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(f"activation consumer retirement path is not a real directory: {current}")
    return True


def _sync_directory_chain_to_root(directory: Path, root: Path) -> None:
    try:
        directory.relative_to(root)
    except ValueError as exc:
        raise ValueError("activation consumer retirement directory must be below the machine runtime root.") from exc
    current = directory
    while True:
        metadata = current.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(f"activation consumer retirement path is not a real directory: {current}")
        descriptor = os.open(current, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if current == root:
            return
        current = current.parent


def _clear_intent(root: Path, intent: _RetirementIntent) -> None:
    path = _intent_path(root, intent)
    current = _read_intent(root, path)
    if current is None:
        return
    if current != intent:
        raise RuntimeError(f"activation consumer retirement intent changed before clearing: {path}")
    path.unlink()
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
