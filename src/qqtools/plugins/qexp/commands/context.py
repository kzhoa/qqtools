"""Project selection for ordinary qexp commands.

Selection is deliberately read-only.  It normalizes a project directory or
its ``.qexp`` control directory, validates the selected root, and reports the
source used by the caller for bounded status output.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..config_types import RootConfig
from ..layout import load_context, load_root_config


class ProjectSelectionError(ValueError):
    """An explicit or discovered Project selector is missing or invalid."""


@dataclass(frozen=True, slots=True)
class ProjectSelection:
    """The canonical project control root and the selector that chose it."""

    shared_root: Path
    source: str


def normalize_project_path(value: str | os.PathLike[str]) -> Path:
    """Normalize a project directory or a ``.qexp`` control directory."""
    path = Path(value).expanduser()
    if path.name != ".qexp":
        path /= ".qexp"
    return path.resolve()


def _validate_project_path(path: Path) -> Path:
    if not path.exists():
        raise ProjectSelectionError(f"selected Project does not exist: {path}")
    if not path.is_dir():
        raise ProjectSelectionError(f"selected Project is not a directory: {path}")
    try:
        load_root_config(path, "unbound", require_initialized=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ProjectSelectionError(f"selected Project is invalid: {path}: {exc}") from exc
    return path


def _cwd_project() -> Path | None:
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        control = candidate / ".qexp"
        if control.exists():
            return _validate_project_path(control)
    return None


def resolve_project(explicit: str | os.PathLike[str] | None = None) -> ProjectSelection:
    """Resolve an ordinary Project using explicit, environment, cwd, saved order."""
    if explicit is not None:
        return ProjectSelection(_validate_project_path(normalize_project_path(explicit)), "explicit")

    environment = os.environ.get("QEXP_SHARED_ROOT")
    if environment:
        return ProjectSelection(_validate_project_path(normalize_project_path(environment)), "environment")

    discovered = _cwd_project()
    if discovered is not None:
        return ProjectSelection(discovered, "cwd")

    saved = load_context()
    if saved is not None:
        value = saved.get("shared_root")
        if not isinstance(value, str) or not value:
            raise ProjectSelectionError("saved qexp Project context is malformed")
        return ProjectSelection(_validate_project_path(normalize_project_path(value)), "saved")

    raise ProjectSelectionError("Project is required; pass --project PATH or run 'qexp use --project PATH'.")


def config_for_selection(
    selection: ProjectSelection,
    *,
    machine_name: str,
    runtime_root: str | os.PathLike[str] | None = None,
    require_initialized: bool = True,
) -> RootConfig:
    """Build a RootConfig for a previously validated selection."""
    return load_root_config(
        selection.shared_root,
        machine_name,
        runtime_root,
        require_initialized=require_initialized,
    )


__all__ = [
    "ProjectSelection",
    "ProjectSelectionError",
    "config_for_selection",
    "normalize_project_path",
    "resolve_project",
]
