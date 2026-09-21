"""Read-only Project locator resolution for qexp submission input."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .submission_contracts import ProjectSelection

_SCHEMA_RELATIVE_PATH = Path("schema") / "version.json"


def _canonical_candidate(value: str | Path, *, invocation_cwd: Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = invocation_cwd / candidate
    candidate = candidate.resolve()
    if candidate.name == ".qexp":
        return candidate
    return candidate / ".qexp"


def _is_initialized(root: Path) -> bool:
    return root.is_dir() and (root / _SCHEMA_RELATIVE_PATH).is_file()


def _looks_like_locator(value: Path) -> bool:
    """Whether a path contains a Project-shaped marker that must fail closed."""
    return value.name == ".qexp" or value.is_dir() and (value / ".qexp").exists()


def _select(value: str | Path, source: str, *, invocation_cwd: Path) -> ProjectSelection:
    root = _canonical_candidate(value, invocation_cwd=invocation_cwd)
    if not _is_initialized(root):
        raise ValueError(f"selected Project locator {Path(value)!s} is not an initialized .qexp root: {root}")
    return ProjectSelection(root.parent, source)  # type: ignore[arg-type]


def _ancestor_selection(start: Path, source: str) -> ProjectSelection | None:
    current = start if start.is_dir() else start.parent
    for directory in (current, *current.parents):
        marker = directory / ".qexp"
        if marker.exists():
            if not marker.is_dir() or not _is_initialized(marker):
                raise ValueError(f"discovered Project locator {marker} is not an initialized .qexp root.")
            return ProjectSelection(marker.parent, source)  # type: ignore[arg-type]
    return None


def _saved_value(saved_context: Mapping[str, Any] | str | Path | None) -> str | Path | None:
    if saved_context is None:
        return None
    if isinstance(saved_context, Mapping):
        return saved_context.get("shared_root") or saved_context.get("project")
    return saved_context


def _manifest_candidate(manifest_path: str | Path, *, invocation_cwd: Path) -> Path:
    path = Path(manifest_path).expanduser()
    if not path.is_absolute():
        path = invocation_cwd / path
    return path.resolve()


def resolve_submission_project(
    *,
    explicit_project: str | Path | None = None,
    manifest_path: str | Path | None = None,
    invocation_cwd: str | Path,
    environment_value: str | Path | None = None,
    saved_context: Mapping[str, Any] | str | Path | None = None,
) -> ProjectSelection:
    """Resolve one initialized Project without initializing or mutating it.

    Command mode uses explicit, environment, cwd ancestry, then saved context.
    File mode inserts the manifest's nearest ancestor between environment and
    cwd ancestry.  Every selected locator is validated before lower-precedence
    candidates are considered, so a malformed explicit root cannot silently
    redirect a submission to another Project.
    """
    cwd = Path(invocation_cwd).expanduser().resolve()
    saved = _saved_value(saved_context)
    if explicit_project is not None:
        return _select(explicit_project, "cli", invocation_cwd=cwd)
    if environment_value is not None:
        return _select(environment_value, "environment", invocation_cwd=cwd)

    if manifest_path is not None:
        manifest = _manifest_candidate(manifest_path, invocation_cwd=cwd)
        if not manifest.exists():
            raise ValueError(f"submission manifest does not exist: {manifest}")
        selected = _ancestor_selection(manifest.parent, "manifest_ancestor")
        if selected is not None:
            return selected

    selected = _ancestor_selection(cwd, "cwd_ancestor")
    if selected is not None:
        return selected
    if saved is not None:
        return _select(saved, "saved", invocation_cwd=cwd)
    raise ValueError("submission requires --project, QEXP_SHARED_ROOT, a Project ancestor, or saved qexp context.")


__all__ = ["ProjectSelection", "resolve_submission_project"]
