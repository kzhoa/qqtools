"""Publish Project takeover binding changes using machine-local state."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from ..runtime.registration_protocol import _read_protocol_record
from ..runtime.responsibility_capture import capture_cleanup_guard
from ..runtime.responsibility_store import DurableIO
from ..runtime.store import atomic_replace, check_mutation_fence, read_json
from .bindings import ProjectBinding
from .inventory import ProjectInventoryEntry, decode_inventory, save_inventory_locked
from .takeover_intent import TakeoverIntent

if TYPE_CHECKING:
    from .context import MachineRuntime

_ProjectRecord = TypeVar("_ProjectRecord", ProjectBinding, ProjectInventoryEntry)


def _require_scope(runtime: MachineRuntime, intent: TakeoverIntent, *phases: str) -> None:
    if intent.phase not in phases:
        raise RuntimeError(f"Cannot change the local Project binding in takeover phase {intent.phase!r}.")
    transition = intent.transition
    if (
        runtime.root != transition.successor_runtime_root
        or runtime.observe_instance_id() != transition.successor_runtime_id
    ):
        raise RuntimeError("Project takeover belongs to a different MachineRuntime.")


def _departing_binding(intent: TakeoverIntent) -> ProjectBinding | None:
    transition = intent.transition
    departing = transition.departing
    if departing is None:
        return None
    return ProjectBinding.from_canonical_paths(
        transition.project_id,
        transition.shared_root,
        departing.machine_name,
        False,
        departing.generation,
        departing.runtime_id,
        departing.runtime_root,
    )


@contextmanager
def _binding_mutation_guards(runtime: MachineRuntime, departing: ProjectBinding | None) -> Iterator[bool]:
    with ExitStack() as guards:
        if departing is not None and not guards.enter_context(runtime.binding_commit_guard(departing, blocking=False)):
            yield False
            return
        # The lifecycle caller may already hold migration read ownership.
        # Never wait for an inventory writer that may be waiting for migration.
        is_acquired = guards.enter_context(runtime.inventory_guard(blocking=False)) and guards.enter_context(
            runtime.registry_guard(blocking=False)
        )
        yield is_acquired


def _exact_project_record(records: list[_ProjectRecord], intent: TakeoverIntent, label: str) -> _ProjectRecord | None:
    transition = intent.transition
    project_id, shared_root = transition.project_id, transition.shared_root
    matches = [item for item in records if item.project_id == project_id or item.shared_root == shared_root]
    if not matches:
        return None
    if len(matches) != 1 or (matches[0].project_id, matches[0].shared_root) != (project_id, shared_root):
        raise RuntimeError(f"Project takeover conflicts with the local {label} for {project_id!r}.")
    return matches[0]


def _require_departing_binding(
    current: ProjectBinding | None, departing: ProjectBinding, original_enabled: bool
) -> ProjectBinding:
    if (
        current is None
        or current.machine_name != departing.machine_name
        or current.registration_generation != departing.registration_generation
        or current.runtime_instance_id != departing.runtime_instance_id
        or current.runtime_root not in (None, departing.runtime_root)
        or current.enabled not in (False, original_enabled)
    ):
        raise RuntimeError("The departing Project binding identity or enablement changed during takeover.")
    return current


def _load_existing_inventory(runtime: MachineRuntime) -> tuple[int, list[ProjectInventoryEntry]]:
    path = runtime.paths["inventory"]
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            raise RuntimeError("Machine Project inventory must be a regular non-symlink file.")
        return decode_inventory(read_json(path))
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError("Machine Project inventory is missing or malformed.") from exc


def observe_takeover_inventory(
    runtime: MachineRuntime, project_id: str, shared_root: Path
) -> ProjectInventoryEntry | None:
    """Read the exact inventory scope without initializing or repairing it."""
    _, entries = _load_existing_inventory(runtime)
    matches = [entry for entry in entries if entry.project_id == project_id or entry.shared_root == shared_root]
    if not matches:
        return None
    if len(matches) != 1 or matches[0].project_id != project_id or matches[0].shared_root != shared_root:
        raise RuntimeError("The selected Project conflicts with an existing inventory entry.")
    return matches[0]


def _save_binding(
    runtime: MachineRuntime,
    revision: int,
    bindings: list[ProjectBinding],
    current: ProjectBinding | None,
    updated: ProjectBinding,
) -> list[ProjectBinding]:
    updated_bindings = [updated if item == current else item for item in bindings]
    if current is None:
        updated_bindings.append(updated)
    if runtime.registration.save_registry_locked(revision + 1, updated_bindings) is None:
        raise RuntimeError("Could not verify durable Project takeover registry publication.")
    return updated_bindings


def _save_inventory_entry(
    runtime: MachineRuntime,
    revision: int,
    entries: list[ProjectInventoryEntry],
    current: ProjectInventoryEntry | None,
    updated: ProjectInventoryEntry,
) -> None:
    if current == updated:
        return
    updated_entries = [updated if item == current else item for item in entries]
    if current is None:
        updated_entries.append(updated)
    if save_inventory_locked(runtime, revision + 1, updated_entries) is None:
        raise RuntimeError("Could not verify durable Project takeover inventory publication.")


def _set_departing_enablement(runtime: MachineRuntime, intent: TakeoverIntent, *, is_enabled: bool) -> bool:
    departing = _departing_binding(intent)
    with _binding_mutation_guards(runtime, departing) as is_acquired:
        if not is_acquired:
            return False
        revision, bindings = runtime.load_registry_uncached()
        current = _exact_project_record(bindings, intent, "registry")
        if departing is None:
            if current is not None:
                raise RuntimeError("A local Project binding appeared during takeover.")
            if intent.phase != "reserved":
                return True
            entry = observe_takeover_inventory(runtime, intent.transition.project_id, intent.transition.shared_root)
            if (entry is None and not intent.original_enabled) or (
                entry is not None and entry.enabled != intent.original_enabled
            ):
                raise RuntimeError("The confirmed Project inventory enablement changed.")
            return True
        current = _require_departing_binding(current, departing, intent.original_enabled)
        inventory_revision, entries = _load_existing_inventory(runtime)
        entry = _exact_project_record(entries, intent, "inventory")
        if entry is None:
            raise RuntimeError("The departing Project binding is missing its inventory entry.")
        if current.enabled != is_enabled:
            updated = replace(current, enabled=is_enabled, _canonical_paths=True)
            _save_binding(runtime, revision, bindings, current, updated)
        _save_inventory_entry(
            runtime,
            inventory_revision,
            entries,
            entry,
            replace(entry, enabled=is_enabled, _canonical_paths=True),
        )
        # Both files share this directory, including visible writes from a failed sync.
        DurableIO().sync_directory(runtime.root, "takeover_departing_enablement")
        return True


def disable_departing_binding(runtime: MachineRuntime, intent: TakeoverIntent) -> bool:
    """Disable the reserved takeover's old binding, or return False for lock contention."""
    _require_scope(runtime, intent, "reserved")
    return _set_departing_enablement(runtime, intent, is_enabled=False)


def restore_departing_binding(runtime: MachineRuntime, intent: TakeoverIntent) -> bool:
    """Restore the old binding after cancellation proof, or return False if blocked."""
    _require_scope(runtime, intent, "cancelling", "aborting_stale")
    return _set_departing_enablement(runtime, intent, is_enabled=intent.original_enabled)


def _directory_stat(path: Path) -> os.stat_result | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"Project takeover path is not a real directory: {path}")
    return metadata


def _require_directory(path: Path) -> None:
    if _directory_stat(path) is None:
        raise RuntimeError(f"Project takeover directory is missing: {path}")


def _archive_directory(runtime: MachineRuntime, intent: TakeoverIntent, *, should_create: bool) -> Path:
    archive = runtime.paths["archives"]
    _require_directory(runtime.root)
    _require_directory(archive)
    for child in ("project-takeovers", intent.plan_id):
        archive /= child
        if should_create and _directory_stat(archive) is None:
            check_mutation_fence(archive)
            archive.mkdir()
        _require_directory(archive)
    return archive


def _archive_manifest(
    intent: TakeoverIntent,
    departing: ProjectBinding,
    source: Path,
    destination: Path,
    partition: os.stat_result | None,
) -> dict[str, str | int | None]:
    return {
        "version": 1,
        "plan_id": intent.plan_id,
        "project_id": departing.project_id,
        "shared_root": str(departing.shared_root),
        "runtime_id": departing.runtime_instance_id,
        "runtime_root": departing.runtime_root,
        "departing_machine_name": departing.machine_name,
        "departing_generation": departing.registration_generation,
        "source": str(source),
        "archive": str(destination),
        "partition_inode": str(partition.st_ino) if partition is not None else None,
    }


def _archive_departing_partition(runtime: MachineRuntime, intent: TakeoverIntent, departing: ProjectBinding) -> bool:
    source = runtime.paths["projects"] / departing.project_id
    _require_directory(runtime.root)
    _require_directory(source.parent)
    _directory_stat(source)
    with capture_cleanup_guard(source, should_remove_roots=True) as can_archive:
        if not can_archive:
            return False
        _require_directory(source.parent)
        source_stat = _directory_stat(source)
        archive = _archive_directory(runtime, intent, should_create=True)
        destination = archive / "partition"
        archived_stat = _directory_stat(destination)
        if source_stat is not None and archived_stat is not None:
            raise RuntimeError("Both the departing Project partition and its takeover archive exist.")
        path = archive / "manifest.json"
        expected = _archive_manifest(intent, departing, source, destination, source_stat or archived_stat)
        existing = _read_protocol_record(path)
        if existing is None:
            if archived_stat is not None:
                raise RuntimeError("The Project takeover archive has no manifest.")
            if atomic_replace(path, expected) is None:
                raise RuntimeError("Could not verify durable Project takeover archive manifest publication.")
        elif existing != expected or type(existing.get("version")) is not int:
            raise RuntimeError("The Project takeover archive manifest conflicts with the departing partition.")

        io = DurableIO()
        for directory in (archive, archive.parent, runtime.paths["archives"], runtime.root):
            io.sync_directory(directory, "takeover_archive_manifest")
        if source_stat is not None:
            check_mutation_fence(source)
            check_mutation_fence(destination)
            os.rename(source, destination)
        # Also finish barriers after a visible rename whose first sync failed.
        io.sync_directory(archive, "takeover_archive_partition_destination")
        io.sync_directory(source.parent, "takeover_archive_partition_source")
        return True


def _require_published_archive(runtime: MachineRuntime, intent: TakeoverIntent, departing: ProjectBinding) -> None:
    archive = _archive_directory(runtime, intent, should_create=False)
    source = runtime.paths["projects"] / departing.project_id
    destination = archive / "partition"
    expected = _archive_manifest(intent, departing, source, destination, _directory_stat(destination))
    existing = _read_protocol_record(archive / "manifest.json")
    if existing is None or existing != expected or type(existing.get("version")) is not int:
        raise RuntimeError("The published Project takeover archive manifest conflicts with the departing partition.")


def publish_takeover_binding(runtime: MachineRuntime, intent: TakeoverIntent) -> bool:
    """Publish the exact committed successor, or return False for a local hold."""
    _require_scope(runtime, intent, "publishing_binding")
    transition = intent.transition
    departing = _departing_binding(intent)
    successor = ProjectBinding.from_canonical_paths(
        transition.project_id,
        transition.shared_root,
        transition.target.machine_name,
        False,
        transition.successor_generation,
        transition.successor_runtime_id,
        str(transition.successor_runtime_root),
    )
    enabled_successor = replace(successor, enabled=intent.original_enabled, _canonical_paths=True)
    with _binding_mutation_guards(runtime, departing) as is_acquired:
        if not is_acquired:
            return False
        revision, bindings = runtime.load_registry_uncached()
        current = _exact_project_record(bindings, intent, "registry")
        inventory_revision, entries = _load_existing_inventory(runtime)
        entry = _exact_project_record(entries, intent, "inventory")
        if current in (successor, enabled_successor):
            witness = transition.departing
            if departing is not None and witness is not None:
                _require_published_archive(runtime, intent, departing)
                runtime.activation_consumer_retirements.recover_exact_locked(
                    runtime_id=witness.runtime_id,
                    project_id=transition.project_id,
                    registration_generation=witness.generation,
                    shared_root=transition.shared_root,
                    machine_name=witness.machine_name,
                    snapshot=bindings,
                )
        else:
            if departing is None:
                if current is not None:
                    raise RuntimeError("A local Project binding appeared before takeover publication.")
                parent = runtime.paths["projects"]
                _require_directory(runtime.root)
                _require_directory(parent)
                if _directory_stat(parent / transition.project_id) is not None:
                    raise RuntimeError("Cannot publish Project takeover over an orphan local partition.")
            else:
                current = _require_departing_binding(current, departing, False)
                if entry is None:
                    raise RuntimeError("The departing Project binding is missing its inventory entry.")
                if not _archive_departing_partition(runtime, intent, departing):
                    return False
                runtime.activation_consumer_retirements.prepare(departing)

            bindings = _save_binding(runtime, revision, bindings, current, successor)
            revision += 1
            current = successor
            if departing is not None:
                runtime.activation_consumer_retirements.complete(departing)

        updated = ProjectInventoryEntry(
            transition.project_id,
            transition.shared_root,
            intent.original_enabled,
            "explicit",
            transition.target.machine_name,
            _canonical_paths=True,
        )
        _save_inventory_entry(runtime, inventory_revision, entries, entry, updated)
        if current != enabled_successor:
            _save_binding(runtime, revision, bindings, current, enabled_successor)
        DurableIO().sync_directory(runtime.root, "takeover_publish_binding")
        return True
