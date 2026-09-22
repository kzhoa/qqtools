"""Immutable values shared by qexp submission input and output boundaries.

The runtime submission protocol intentionally owns its own internal plan type.
These values describe the user-facing request before that protocol is entered,
and the result family presented by the CLI.  They contain no filesystem
writers and are safe to use while validating a request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

ProjectSource = Literal["cli", "environment", "manifest_ancestor", "cwd_ancestor", "saved"]
GroupSource = Literal["cli", "manifest", "none", None]
SubmissionMode = Literal["command", "file"]
SubmissionOutcome = Literal["committed", "rejected", "pending", "unknown", "preview"]


def _freeze(value: Any) -> Any:
    """Recursively freeze JSON-like request data without copying Path values."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, frozenset)):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ProjectSelection:
    """Canonical Project directory and its ``.qexp`` control root."""

    path: Path
    source: ProjectSource
    control_root: Path = field(init=False)

    def __post_init__(self) -> None:
        path = Path(self.path).expanduser().resolve()
        if path.name == ".qexp":
            path = path.parent
        control_root = path / ".qexp"
        if control_root.name != ".qexp":
            raise ValueError("submission project path must resolve to a .qexp control root.")
        if self.source not in {"cli", "environment", "manifest_ancestor", "cwd_ancestor", "saved"}:
            raise ValueError(f"invalid submission project source {self.source!r}.")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "control_root", control_root)

    @property
    def shared_root(self) -> Path:
        """Compatibility spelling used by the runtime configuration layer."""
        return self.control_root

    @property
    def canonical_path(self) -> Path:
        """Return the canonical ``.qexp`` locator requested by runtime APIs."""
        return self.control_root

    def to_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "source": self.source}


@dataclass(frozen=True, slots=True)
class SubmissionRequest:
    """Frozen, normalized input shared by preview and real submission."""

    mode: SubmissionMode
    specs: tuple[Mapping[str, Any], ...]
    group_name: str | None
    group_source: GroupSource
    workers: Mapping[str, Mapping[str, Any]]
    workers_declared: bool
    project: ProjectSelection
    invocation_cwd: Path
    manifest_path: Path | None
    idempotency_key: str | None
    no_activate: bool = False
    dry_run: bool = False
    output_format: str = "human"
    quiet: bool = False
    field_sources: tuple[Mapping[str, str], ...] = ()
    group_existed: bool | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"command", "file"}:
            raise ValueError(f"invalid submission mode {self.mode!r}.")
        if not self.specs:
            raise ValueError("submission must contain at least one task specification.")
        object.__setattr__(self, "specs", tuple(_freeze(item) for item in self.specs))
        object.__setattr__(self, "workers", _freeze(self.workers))
        object.__setattr__(self, "invocation_cwd", Path(self.invocation_cwd).expanduser().resolve())
        if self.manifest_path is not None:
            object.__setattr__(self, "manifest_path", Path(self.manifest_path).expanduser().resolve())
        if self.output_format not in {"human", "json"}:
            raise ValueError("submission output format must be 'human' or 'json'.")
        if self.quiet and self.output_format == "json":
            raise ValueError("--quiet cannot be combined with --format json.")
        if self.quiet and self.dry_run:
            raise ValueError("--quiet cannot be combined with --dry-run.")
        if self.group_source not in {"cli", "manifest", "none", None}:
            raise ValueError(f"invalid submission group source {self.group_source!r}.")
        if len(self.field_sources) == 0:
            object.__setattr__(self, "field_sources", tuple({} for _ in self.specs))
        else:
            object.__setattr__(self, "field_sources", tuple(_freeze(item) for item in self.field_sources))
        if len(self.field_sources) != len(self.specs):
            raise ValueError("submission field source maps must match task specifications.")

    @property
    def normalized_specs(self) -> list[dict[str, Any]]:
        """Return independent mutable dictionaries for the runtime boundary."""
        return [_thaw(item) for item in self.specs]

    @property
    def worker_set(self) -> dict[str, dict[str, Any]]:
        return _thaw(self.workers)

    @property
    def group(self) -> str | None:
        return self.group_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "specs": self.normalized_specs,
            "group": {
                "name": self.group_name,
                "source": self.group_source,
                "workers_declared": self.workers_declared,
                "workers": self.worker_set,
            },
            "project": self.project.to_dict(),
            "invocation_cwd": str(self.invocation_cwd),
            "manifest_path": str(self.manifest_path) if self.manifest_path is not None else None,
            "idempotency_key": self.idempotency_key,
            "no_activate": self.no_activate,
            "dry_run": self.dry_run,
            "format": self.output_format,
            "quiet": self.quiet,
            "field_sources": [_thaw(item) for item in self.field_sources],
            "group_existed": self.group_existed,
        }


@dataclass(frozen=True, slots=True)
class SubmissionResult:
    """Canonical submission result payload consumed by the CLI output boundary."""

    schema_version: int = 1
    mode: SubmissionMode | None = None
    outcome: SubmissionOutcome = "rejected"
    project: Mapping[str, Any] | None = None
    group: Mapping[str, Any] | None = None
    operation: Mapping[str, Any] | None = None
    idempotency_key: str | None = None
    task_ids: tuple[str, ...] = ()
    preview: Mapping[str, Any] | None = None
    error: Mapping[str, Any] | None = None
    activation: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("submission result schema_version must be 1.")
        if self.mode not in {"command", "file", None}:
            raise ValueError("submission result mode is invalid.")
        if self.outcome not in {"committed", "rejected", "pending", "unknown", "preview"}:
            raise ValueError("submission result outcome is invalid.")
        object.__setattr__(self, "task_ids", tuple(self.task_ids))
        object.__setattr__(self, "project", _freeze(self.project) if self.project is not None else None)
        object.__setattr__(self, "group", _freeze(self.group) if self.group is not None else None)
        object.__setattr__(self, "operation", _freeze(self.operation) if self.operation is not None else None)
        object.__setattr__(self, "preview", _freeze(self.preview) if self.preview is not None else None)
        object.__setattr__(self, "error", _freeze(self.error) if self.error is not None else None)
        object.__setattr__(self, "activation", _freeze(self.activation) if self.activation is not None else None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "outcome": self.outcome,
            "project": _thaw(self.project),
            "group": _thaw(self.group),
            "operation": _thaw(self.operation),
            "idempotency_key": self.idempotency_key,
            "task_ids": list(self.task_ids),
            "preview": _thaw(self.preview),
            "error": _thaw(self.error),
            "activation": _thaw(self.activation),
        }


def submission_result_payload(
    *,
    mode: SubmissionMode | None,
    outcome: SubmissionOutcome,
    project: ProjectSelection | Mapping[str, Any] | None,
    group: Mapping[str, Any] | None,
    operation: Mapping[str, Any] | None,
    idempotency_key: str | None,
    task_ids: list[str] | tuple[str, ...] = (),
    preview: Mapping[str, Any] | None = None,
    error: Mapping[str, Any] | None = None,
    activation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete JSON-compatible result, including every nullable field."""
    if isinstance(project, ProjectSelection):
        project = project.to_dict()
    return SubmissionResult(
        mode=mode,
        outcome=outcome,
        project=project,
        group=group,
        operation=operation,
        idempotency_key=idempotency_key,
        task_ids=tuple(task_ids),
        preview=preview,
        error=error,
        activation=activation,
    ).to_dict()


__all__ = [
    "GroupSource",
    "ProjectSelection",
    "ProjectSource",
    "SubmissionMode",
    "SubmissionOutcome",
    "SubmissionRequest",
    "SubmissionResult",
    "submission_result_payload",
]
