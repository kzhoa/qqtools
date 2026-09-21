"""Machine-local Project inventory and the bounded registry-v1 import."""

from __future__ import annotations

import os
from dataclasses import dataclass
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

    def __post_init__(self) -> None:
        if not isinstance(self.project_id, str) or not self.project_id or "/" in self.project_id:
            raise ValueError("inventory project_id is invalid.")
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
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("inventory entry is malformed.") from exc


InventoryEntry = ProjectInventoryEntry


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


def save_inventory_locked(runtime: Any, revision: int, entries: Iterable[ProjectInventoryEntry]) -> None:
    """Persist an inventory revision while the caller holds inventory_guard."""
    atomic_replace(_inventory_path(runtime), _encode(revision, entries))


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
    revision, entries = load_inventory(machine_runtime)
    _registry_revision, bindings = machine_runtime.load_registry()
    result: list[dict[str, Any]] = []
    for entry in entries:
        matching = [
            binding
            for binding in bindings
            if binding.project_id == entry.project_id or binding.shared_root == entry.shared_root
        ]
        binding = matching[0] if len(matching) == 1 else None
        conflict = len(matching) > 1 or any(
            item.project_id == entry.project_id and item.shared_root != entry.shared_root for item in matching
        )
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
    return {"revision": revision, "projects": result}


__all__ = [
    "INVENTORY_VERSION",
    "NAME_SOURCES",
    "InventoryEntry",
    "ProjectInventoryEntry",
    "canonical_shared_root",
    "ensure_inventory",
    "find_inventory_entries",
    "inventory_payload",
    "inventory_status",
    "lexical_shared_root",
    "load_inventory",
    "project_inventory_entry",
    "save_inventory_locked",
]
