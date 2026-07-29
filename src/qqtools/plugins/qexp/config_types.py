"""Value types shared by qexp configuration modules."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RootConfig:
    shared_root: Path
    project_root: Path
    machine_name: str
    runtime_root: Path

    def __post_init__(self) -> None:
        self.shared_root = Path(self.shared_root).expanduser().resolve()
        self.project_root = Path(self.project_root).expanduser().resolve()
        self.runtime_root = Path(self.runtime_root).expanduser().resolve()
        if self.shared_root.name != ".qexp":
            raise ValueError("shared_root must point to a project control root named '.qexp'.")
        if not self.machine_name or "/" in self.machine_name or "\\" in self.machine_name or ".." in self.machine_name:
            raise ValueError("machine_name is invalid.")


@dataclass(frozen=True, slots=True)
class MachinePolicy:
    agent_mode: str
    exit_when_idle: bool
