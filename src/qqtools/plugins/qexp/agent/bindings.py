"""Project-to-machine registration bindings."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..config_types import RootConfig
from ..layout import load_machine_record, load_root_config


@dataclass(frozen=True, slots=True)
class ProjectBinding:
    project_id: str
    shared_root: Path
    machine_name: str
    enabled: bool = True
    registration_generation: str | None = None
    runtime_instance_id: str | None = None
    runtime_root: str | None = None

    def __post_init__(self) -> None:
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
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("project binding has invalid identity fields.") from exc

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
