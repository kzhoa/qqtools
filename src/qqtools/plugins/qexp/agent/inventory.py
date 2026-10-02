"""Machine-local Project inventory and the bounded registry-v1 import."""

from __future__ import annotations

import os
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..runtime.store import atomic_replace, read_json

INVENTORY_VERSION = 1
NAME_SOURCES = frozenset({"default", "explicit", "unresolved"})


def canonical_shared_root(value: str | Path, *, cwd: Path | None = None) -> Path:
    """Normalize a Project directory or explicit ``.qexp`` path."""
    raw = Path(value)
    if not raw.is_absolute():
        raw = (cwd or Path.cwd()) / raw
    if raw.name != ".qexp":
        raw = raw / ".qexp"
    return raw.expanduser().resolve()


def lexical_shared_root(value: str | Path, *, cwd: Path | None = None) -> Path:
    """Normalize a selector without following symlinks or requiring existence."""
    raw = Path(value)
    if not raw.is_absolute():
        raw = (cwd or Path.cwd()) / raw
    if raw.name != ".qexp":
        raw = raw / ".qexp"
    return Path(os.path.normpath(os.path.abspath(os.fspath(raw))))


@dataclass(frozen=True, slots=True)
class ProjectInventoryEntry:
    """Reusable local enrollment intent, independent from a live binding."""

    project_id: str
    shared_root: Path
    enabled: bool = True
    name_source: str = "default"
    name_override: str | None = None
    _canonical_paths: InitVar[bool] = field(default=False, kw_only=True)

    def __post_init__(self, _canonical_paths: bool) -> None:
        if not isinstance(self.project_id, str) or not self.project_id or "/" in self.project_id:
            raise ValueError("inventory project_id is invalid.")
        if type(_canonical_paths) is not bool:
            raise ValueError("_canonical_paths must be a bool.")
        if _canonical_paths:
            path = Path(self.shared_root)
            if not path.is_absolute() or ".." in path.parts or "\x00" in str(path):
                raise ValueError("shared_root must be a canonical absolute path.")
            object.__setattr__(self, "shared_root", path)
        else:
            object.__setattr__(self, "shared_root", Path(self.shared_root).expanduser().resolve())
        if type(self.enabled) is not bool:
            raise ValueError("inventory enabled must be a bool.")
        if not isinstance(self.name_source, str) or self.name_source not in NAME_SOURCES:
            raise ValueError("inventory name_source must be default, explicit, or unresolved.")
        if self.name_override is not None and (not isinstance(self.name_override, str) or not self.name_override):
            raise ValueError("inventory name_override is invalid.")
        if self.name_source == "default" and self.name_override is not None:
            raise ValueError("default inventory entries cannot have a name_override.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "shared_root": str(self.shared_root),
            "enabled": self.enabled,
            "name_source": self.name_source,
            "name_override": self.name_override,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProjectInventoryEntry":
        if not isinstance(value, dict):
            raise ValueError("inventory entry must be an object.")
        try:
            return cls(
                project_id=value["project_id"],
                shared_root=value["shared_root"],
                enabled=value["enabled"],
                name_source=value.get("name_source", "default"),
                name_override=value.get("name_override"),
                _canonical_paths=True,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("inventory entry is malformed.") from exc


InventoryEntry = ProjectInventoryEntry


@dataclass(frozen=True, slots=True)
class InventoryReconciliation:
    """Observed registry and inventory view after exact-binding repair."""

    registry_revision: int
    inventory_revision: int
    entries: tuple[ProjectInventoryEntry, ...]
    bindings: tuple[Any, ...]
    blockers: tuple[str, ...]
    converged: bool
    changed: bool


def exact_binding_for_entry(entry: ProjectInventoryEntry, bindings: Iterable[Any]) -> tuple[Any | None, bool]:
    """Match one binding only when both Project ID and canonical root agree uniquely."""
    values = tuple(bindings)
    id_matches = [binding for binding in values if binding.project_id == entry.project_id]
    root_matches = [binding for binding in values if binding.shared_root == entry.shared_root]
    exact_matches = [
        binding
        for binding in values
        if binding.project_id == entry.project_id and binding.shared_root == entry.shared_root
    ]
    has_conflict = bool(id_matches or root_matches) and (
        len(exact_matches) != 1 or len(id_matches) != 1 or len(root_matches) != 1
    )
    return (exact_matches[0] if len(exact_matches) == 1 and not has_conflict else None), has_conflict


def _as_runtime(runtime: Any):
    if hasattr(runtime, "paths") and hasattr(runtime, "load_registry"):
        return runtime
    from .context import MachineRuntime

    return MachineRuntime(runtime)


def _inventory_path(runtime: Any) -> Path:
    return _as_runtime(runtime).paths["inventory"]


def _decode(value: dict[str, Any]) -> tuple[int, list[ProjectInventoryEntry]]:
    raw = value.get("inventory")
    if not isinstance(raw, dict) or raw.get("version") != INVENTORY_VERSION:
        raise RuntimeError("machine Project inventory is malformed or unsupported.")
    revision = raw.get("revision")
    entries = raw.get("entries")
    if type(revision) is not int or revision < 0 or not isinstance(entries, list):
        raise RuntimeError("machine Project inventory is malformed.")
    try:
        parsed = [ProjectInventoryEntry.from_dict(item) for item in entries]
    except (TypeError, ValueError) as exc:
        raise RuntimeError("machine Project inventory contains a malformed entry.") from exc
    ids = [entry.project_id for entry in parsed]
    roots = [entry.shared_root for entry in parsed]
    if len(ids) != len(set(ids)) or len(roots) != len(set(roots)):
        raise RuntimeError("machine Project inventory contains duplicate identity or path.")
    return revision, parsed


def _encode(revision: int, entries: Iterable[ProjectInventoryEntry]) -> dict[str, Any]:
    return {
        "inventory": {
            "version": INVENTORY_VERSION,
            "revision": revision,
            "entries": [entry.to_dict() for entry in sorted(entries, key=lambda item: item.project_id)],
        }
    }


def _legacy_inventory(runtime: Any) -> list[ProjectInventoryEntry]:
    # QQTOOLS-COMPAT-0012: import registry-v1 bindings once and retain their
    # effective names as unresolved intent; name equality never proves origin.
    try:
        _revision, bindings = runtime.load_registry()
    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
        return []
    return [
        ProjectInventoryEntry(
            project_id=binding.project_id,
            shared_root=binding.shared_root,
            enabled=binding.enabled,
            name_source="unresolved",
            name_override=binding.machine_name,
            _canonical_paths=True,
        )
        for binding in bindings
    ]


def ensure_inventory(runtime: Any, *, require_initialized: bool = True) -> tuple[int, list[ProjectInventoryEntry]]:
    """Load inventory and import an existing registry exactly once."""
    machine_runtime = _as_runtime(runtime)
    if require_initialized:
        machine_runtime.require_initialized()
    path = _inventory_path(machine_runtime)
    if path.exists():
        return _decode(read_json(path))
    if not machine_runtime.has_identity and machine_runtime.paths["registry"].exists():
        machine_runtime.ensure_compatibility_identity()
    imported = _legacy_inventory(machine_runtime) if machine_runtime.paths["registry"].exists() else []
    if getattr(machine_runtime, "_inventory_lock_depth", 0):
        if path.exists():
            return _decode(read_json(path))
        atomic_replace(path, _encode(0, imported))
        return 0, imported
    with machine_runtime.inventory_guard():
        if path.exists():
            return _decode(read_json(path))
        atomic_replace(path, _encode(0, imported))
        return 0, imported


def load_inventory(runtime: Any, *, require_initialized: bool = True) -> tuple[int, list[ProjectInventoryEntry]]:
    return ensure_inventory(runtime, require_initialized=require_initialized)


def save_inventory_locked(runtime: Any, revision: int, entries: Iterable[ProjectInventoryEntry]):
    """Persist an inventory revision while the caller holds inventory_guard."""
    machine_runtime = _as_runtime(runtime)
    try:
        return atomic_replace(_inventory_path(machine_runtime), _encode(revision, entries))
    finally:
        machine_runtime.invalidate_inventory_cache()


def reconcile_live_bindings_locked(runtime: Any, *, repair: bool = True) -> InventoryReconciliation:
    """Inspect or repair exact live mirrors under inventory and registry locks."""
    machine_runtime = _as_runtime(runtime)
    if not machine_runtime._inventory_lock_depth or not machine_runtime._registry_guard_depth.get():
        raise RuntimeError("Project inventory reconciliation requires inventory and registry guards.")
    inventory_revision, entries = load_inventory(machine_runtime)
    registry_revision, bindings = machine_runtime.load_registry_uncached()
    next_entries = list(entries)
    blockers: list[str] = []
    runtime_id = machine_runtime.instance_id
    runtime_root = str(machine_runtime.root)

    for binding in bindings:
        id_matches = [entry for entry in entries if entry.project_id == binding.project_id]
        root_matches = [entry for entry in entries if entry.shared_root == binding.shared_root]
        exact_matches = [
            entry
            for entry in entries
            if entry.project_id == binding.project_id and entry.shared_root == binding.shared_root
        ]
        if len(exact_matches) != 1 or len(id_matches) != 1 or len(root_matches) != 1:
            blockers.append(
                "inventory_identity_path_conflict" if id_matches or root_matches else "inventory_entry_missing"
            )
            continue
        if (
            binding.runtime_instance_id != runtime_id
            or binding.runtime_root not in {None, runtime_root}
            or not isinstance(binding.registration_generation, str)
            or not binding.registration_generation
        ):
            blockers.append("binding_identity_changed")
            continue
        entry = exact_matches[0]
        if entry.enabled == binding.enabled:
            continue
        updated = ProjectInventoryEntry(
            entry.project_id,
            entry.shared_root,
            binding.enabled,
            entry.name_source,
            entry.name_override,
            _canonical_paths=True,
        )
        next_entries[next_entries.index(entry)] = updated

    changed = next_entries != entries
    if changed and not repair:
        blockers.append("enablement_mirror_diverged")
    elif changed:
        try:
            persisted = save_inventory_locked(machine_runtime, inventory_revision + 1, next_entries)
        except OSError:
            persisted = None
        if persisted is not None:
            inventory_revision += 1
            entries = next_entries
        else:
            blockers.append("inventory_mirror_incomplete")
            try:
                inventory_revision, entries = load_inventory(machine_runtime)
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                pass

    is_converged = not blockers and all(
        (binding.project_id, binding.shared_root) in {(entry.project_id, entry.shared_root) for entry in entries}
        and next(
            entry.enabled
            for entry in entries
            if entry.project_id == binding.project_id and entry.shared_root == binding.shared_root
        )
        == binding.enabled
        for binding in bindings
    )
    return InventoryReconciliation(
        registry_revision=registry_revision,
        inventory_revision=inventory_revision,
        entries=tuple(entries),
        bindings=tuple(bindings),
        blockers=tuple(sorted(set(blockers))),
        converged=is_converged,
        changed=repair and changed and "inventory_mirror_incomplete" not in blockers,
    )


def project_inventory_entry(
    project_id_value: str,
    shared_root: str | Path,
    *,
    enabled: bool = True,
    name_source: str = "default",
    name_override: str | None = None,
) -> ProjectInventoryEntry:
    return ProjectInventoryEntry(project_id_value, shared_root, enabled, name_source, name_override)


def inventory_payload(runtime: Any) -> dict[str, Any]:
    """Return inventory entries as stable dictionaries."""
    revision, entries = load_inventory(runtime)
    return {"revision": revision, "entries": [entry.to_dict() for entry in entries]}


def find_inventory_entries(
    entries: Iterable[ProjectInventoryEntry], selector: str | Path, *, cwd: Path | None = None
) -> list[ProjectInventoryEntry]:
    """Resolve an ID or lexical stored-path selector without touching Projects."""
    candidate = str(selector)
    by_id = [entry for entry in entries if entry.project_id == candidate]
    if by_id:
        return by_id
    lexical = lexical_shared_root(selector, cwd=cwd)
    return [entry for entry in entries if entry.shared_root == lexical]


def inventory_status(runtime: Any) -> dict[str, Any]:
    """Classify each inventory item against current effective bindings."""
    machine_runtime = _as_runtime(runtime)
    with machine_runtime.inventory_guard():
        with machine_runtime.registry_guard():
            revision, entries = load_inventory(machine_runtime)
            registry_revision, bindings = machine_runtime.load_registry_uncached()
    result: list[dict[str, Any]] = []
    for entry in entries:
        binding, conflict = exact_binding_for_entry(entry, bindings)
        mount_available = entry.shared_root.is_dir()
        if conflict:
            state = "conflicting"
        elif binding is None:
            state = "inventory_only"
        elif binding.enabled:
            state = "registered"
        else:
            state = "disabled"
        item: dict[str, Any] = {
            **entry.to_dict(),
            "status": state,
            "mount_available": mount_available,
            "machine_name": binding.machine_name if binding is not None else entry.name_override,
            "registration_generation": binding.registration_generation if binding is not None else None,
            "runtime_instance_id": binding.runtime_instance_id if binding is not None else None,
            "effective_enabled": binding.enabled if binding is not None and not conflict else None,
            "inventory_enabled": entry.enabled,
            "inventory_converged": (not conflict and (binding is None or binding.enabled == entry.enabled)),
            "registry_revision": registry_revision,
            "inventory_revision": revision,
        }
        if binding is not None:
            try:
                item["eligibility"] = machine_runtime.registration_status(binding)
                item["write_eligible"] = bool(item["eligibility"].get("write_eligible"))
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                item["eligibility"] = {"state": "invalid", "error": str(exc), "write_eligible": False}
                item["write_eligible"] = False
        else:
            item["eligibility"] = {"state": "unregistered", "write_eligible": False}
            item["write_eligible"] = False
        result.append(item)
    return {
        "revision": revision,
        "inventory_revision": revision,
        "registry_revision": registry_revision,
        "projects": result,
    }


__all__ = [
    "INVENTORY_VERSION",
    "NAME_SOURCES",
    "InventoryEntry",
    "InventoryReconciliation",
    "ProjectInventoryEntry",
    "canonical_shared_root",
    "ensure_inventory",
    "exact_binding_for_entry",
    "find_inventory_entries",
    "inventory_payload",
    "inventory_status",
    "lexical_shared_root",
    "load_inventory",
    "project_inventory_entry",
    "reconcile_live_bindings_locked",
    "save_inventory_locked",
]
