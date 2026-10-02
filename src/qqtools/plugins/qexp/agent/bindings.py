"""Project-to-machine registration bindings."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, replace
from pathlib import Path
from typing import Any

from ..config_types import RootConfig
from ..layout import load_machine_record, load_root_config

REGISTRY_VERSION = 1


@dataclass(frozen=True, slots=True)
class ProjectBinding:
    project_id: str
    shared_root: Path
    machine_name: str
    enabled: bool = True
    registration_generation: str | None = None
    runtime_instance_id: str | None = None
    runtime_root: str | None = None
    _canonical_paths: InitVar[bool] = field(default=False, kw_only=True)

    def __post_init__(self, _canonical_paths: bool) -> None:
        if type(_canonical_paths) is not bool:
            raise ValueError("_canonical_paths must be a bool.")
        if _canonical_paths:
            path = Path(self.shared_root)
            if not path.is_absolute() or ".." in path.parts or "\x00" in str(path):
                raise ValueError("shared_root must be a canonical absolute path.")
            object.__setattr__(self, "shared_root", path)
        else:
            object.__setattr__(self, "shared_root", Path(self.shared_root).expanduser().resolve())
        if not self.project_id or "/" in self.project_id or "\\" in self.project_id or ".." in self.project_id:
            raise ValueError("project_id is invalid.")
        if not self.machine_name or "/" in self.machine_name or "\\" in self.machine_name or ".." in self.machine_name:
            raise ValueError("machine_name is invalid.")
        for value, label in (
            (self.registration_generation, "registration_generation"),
            (self.runtime_instance_id, "runtime_instance_id"),
            (self.runtime_root, "runtime_root"),
        ):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{label} is invalid.")
        if _canonical_paths and self.runtime_root is not None:
            path = Path(self.runtime_root)
            if not path.is_absolute() or ".." in path.parts or "\x00" in str(path):
                raise ValueError("runtime_root must be a canonical absolute path.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "shared_root": str(self.shared_root),
            "machine_name": self.machine_name,
            "enabled": self.enabled,
            "registration_generation": self.registration_generation,
            "runtime_instance_id": self.runtime_instance_id,
            "runtime_root": self.runtime_root,
        }

    @property
    def generation(self) -> str | None:
        """Compatibility alias for the logical registration generation."""
        return self.registration_generation

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProjectBinding":
        if not isinstance(value, dict):
            raise ValueError("project binding must be an object.")
        enabled = value.get("enabled")
        if not isinstance(enabled, bool):
            raise ValueError("project binding enabled must be a bool.")
        try:
            return cls(
                project_id=value["project_id"],
                shared_root=Path(value["shared_root"]),
                machine_name=value["machine_name"],
                enabled=enabled,
                registration_generation=value.get("registration_generation"),
                runtime_instance_id=value.get("runtime_instance_id"),
                runtime_root=value.get("runtime_root"),
                _canonical_paths=True,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("project binding has invalid identity fields.") from exc

    @classmethod
    def from_canonical_paths(
        cls,
        project_id: str,
        shared_root: Path,
        machine_name: str,
        enabled: bool = True,
        registration_generation: str | None = None,
        runtime_instance_id: str | None = None,
        runtime_root: str | None = None,
    ) -> ProjectBinding:
        """Build a binding from absolute paths without resolving filesystem links."""
        return cls(
            project_id=project_id,
            shared_root=shared_root,
            machine_name=machine_name,
            enabled=enabled,
            registration_generation=registration_generation,
            runtime_instance_id=runtime_instance_id,
            runtime_root=runtime_root,
            _canonical_paths=True,
        )

    def root_config(self) -> RootConfig:
        cfg = load_root_config(self.shared_root, self.machine_name, require_initialized=True)
        record = load_machine_record(cfg) or {}
        machine = record.get("machine")
        runtime_root = machine.get("runtime_root") if isinstance(machine, dict) else None
        if not isinstance(runtime_root, str) or not runtime_root:
            raise RuntimeError(
                f"machine {self.machine_name!r} has no valid standalone runtime_root in {self.shared_root}."
            )
        # Root validation is independent of the local runtime location. Reuse
        # this call's validated configuration, without caching it across calls.
        return replace(cfg, runtime_root=Path(runtime_root))


def decode_registry(value: dict[str, Any]) -> tuple[int, tuple[ProjectBinding, ...]]:
    """Decode an immutable local roster without resolving any Project path."""
    registry = value.get("registry")
    if not isinstance(registry, dict) or registry.get("version") != REGISTRY_VERSION:
        raise RuntimeError("machine registry is malformed or unsupported.")
    revision = registry.get("revision")
    bindings = registry.get("bindings")
    if type(revision) is not int or revision < 0 or not isinstance(bindings, list):
        raise RuntimeError("machine registry is malformed.")
    try:
        parsed = tuple(sorted((ProjectBinding.from_dict(item) for item in bindings), key=lambda item: item.project_id))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("machine registry contains a malformed project binding.") from exc
    return revision, parsed
