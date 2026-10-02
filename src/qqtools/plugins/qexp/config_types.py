"""Value types shared by qexp configuration modules."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class RootConfig:
    shared_root: Path
    project_root: Path
    machine_name: str
    runtime_root: Path
    _canonical_paths: InitVar[bool] = field(default=False, kw_only=True)

    def __post_init__(self, _canonical_paths: bool) -> None:
        if type(_canonical_paths) is not bool:
            raise ValueError("_canonical_paths must be a bool.")
        if _canonical_paths:
            for field_name in ("shared_root", "project_root", "runtime_root"):
                path = Path(getattr(self, field_name))
                if not path.is_absolute() or ".." in path.parts or "\x00" in str(path):
                    raise ValueError(f"{field_name} must be a canonical absolute path.")
                setattr(self, field_name, path)
        else:
            self.shared_root = Path(self.shared_root).expanduser().resolve()
            self.project_root = Path(self.project_root).expanduser().resolve()
            self.runtime_root = Path(self.runtime_root).expanduser().resolve()
        if self.shared_root.name != ".qexp":
            raise ValueError("shared_root must point to a project control root named '.qexp'.")
        if not self.machine_name or "/" in self.machine_name or "\\" in self.machine_name or ".." in self.machine_name:
            raise ValueError("machine_name is invalid.")

    @classmethod
    def from_canonical_paths(
        cls,
        shared_root: Path,
        project_root: Path,
        machine_name: str,
        runtime_root: Path,
    ) -> RootConfig:
        """Build a config from absolute paths without resolving filesystem links."""
        return cls(
            shared_root=shared_root,
            project_root=project_root,
            machine_name=machine_name,
            runtime_root=runtime_root,
            _canonical_paths=True,
        )


@dataclass(frozen=True, slots=True)
class MachinePolicy:
    agent_mode: str
    exit_when_idle: bool
