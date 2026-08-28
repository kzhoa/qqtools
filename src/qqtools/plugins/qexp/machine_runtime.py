"""Machine-local runtime ownership for qexp multi-project scheduling."""
from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator

from .config_types import RootConfig
from .layout import load_machine_record, load_root_config
from .runtime.locks import exclusive
from .runtime.filesystem_qualification import (
    validate_existing_filesystem_qualification,
)
from .runtime.paths import local_paths, machine_project_paths, machine_runtime_paths, shared_paths
from .runtime.records import utc_now
from .runtime.store import atomic_replace, iter_json, read_json
from .runtime.work_budget import AdaptiveBatchSizer

MACHINE_RUNTIME_ENV = "QEXP_MACHINE_RUNTIME_ROOT"
REGISTRY_VERSION = 1
LEGACY_AGENT_EVIDENCE = (
    "processes",
    "termination_decisions",
    "wrappers",
    "authority_diagnostics",
    "events",
)
LEGACY_RUNNER_INBOX = ("registrations", "observations", "launch_intents")


def resolve_machine_runtime_root(value: str | Path | None = None) -> Path:
    """Resolve the machine-local authority root without creating it."""
    configured = value if value is not None else os.environ.get(MACHINE_RUNTIME_ENV)
    root = Path(configured).expanduser().resolve() if configured else Path.home() / ".qqtools" / "qexp-machine"
    if root.name == ".qexp" or (root / "schema" / "version.json").exists():
        raise ValueError("QEXP_MACHINE_RUNTIME_ROOT must not point to a project .qexp root.")
    if root.exists() and not root.is_dir():
        raise ValueError("QEXP_MACHINE_RUNTIME_ROOT must be a directory.")
    return root


@dataclass(frozen=True, slots=True)
class ProjectBinding:
    project_id: str
    shared_root: Path
    machine_name: str
    enabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "shared_root", Path(self.shared_root).expanduser().resolve())
        if not self.project_id or "/" in self.project_id or "\\" in self.project_id or ".." in self.project_id:
            raise ValueError("project_id is invalid.")
        if not self.machine_name or "/" in self.machine_name or "\\" in self.machine_name or ".." in self.machine_name:
            raise ValueError("machine_name is invalid.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "shared_root": str(self.shared_root),
            "machine_name": self.machine_name,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProjectBinding":
        enabled = value.get("enabled")
        if not isinstance(enabled, bool):
            raise ValueError("project binding enabled must be a bool.")
        return cls(
            project_id=value["project_id"],
            shared_root=Path(value["shared_root"]),
            machine_name=value["machine_name"],
            enabled=enabled,
        )

    def root_config(self) -> RootConfig:
        cfg = load_root_config(self.shared_root, self.machine_name, require_initialized=True)
        record = load_machine_record(cfg) or {}
        runtime_root = record.get("machine", {}).get("runtime_root")
        if not isinstance(runtime_root, str) or not runtime_root:
            raise RuntimeError(
                f"machine {self.machine_name!r} has no valid standalone runtime_root in {self.shared_root}."
            )
        return load_root_config(
            self.shared_root, self.machine_name, runtime_root, require_initialized=True
        )


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Private pairing of project authority and its local resource backend."""

    cfg: RootConfig
    machine_runtime: "MachineRuntime"
    binding: ProjectBinding | None = None

    @property
    def project_id(self) -> str | None:
        return self.binding.project_id if self.binding else None

    @property
    def local_root(self) -> Path:
        if self.binding:
            return self.machine_runtime.project_paths(self.binding.project_id)["root"]
        return self.cfg.runtime_root

    @property
    def local_cfg(self) -> RootConfig:
        """Return the project configuration with its authoritative local runtime."""
        if self.binding is None:
            return self.cfg
        return replace(self.cfg, runtime_root=self.local_root)

    @property
    def is_machine_managed(self) -> bool:
        return self.binding is not None

    @property
    def reservation_root(self) -> Path:
        return self.machine_runtime.root if self.binding else self.cfg.runtime_root


def resolve_execution_context(
        cfg: RootConfig, machine_runtime_root: str | Path | None = None) -> ExecutionContext:
    """Resolve the registered machine reservation backend for a project operation."""
    return MachineRuntime(machine_runtime_root).execution_context(cfg)


class MachineRuntime:
    """Disposable local resource state shared by one qexp machine."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = resolve_machine_runtime_root(root)
        self.paths = machine_runtime_paths(self.root)
        self.last_diagnostic_publish_ns: int | None = None
        self.ready_batch_sizers: dict[str, AdaptiveBatchSizer] = {}

    def ensure_layout(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if not os.access(self.root, os.W_OK | os.X_OK):
            raise RuntimeError(f"machine runtime root is not writable: {self.root}")
        for name in ("locks", "agent", "provisional", "active", "released", "projects", "diagnostics"):
            self.paths[name].mkdir(parents=True, exist_ok=True)
        self.paths["cursor"].parent.mkdir(parents=True, exist_ok=True)

    def project_paths(self, project_id: str) -> dict[str, Path]:
        return machine_project_paths(self.root, project_id)

    def migration_path(self, project_id: str) -> Path:
        return self.project_paths(project_id)["root"] / "migration.json"

    @contextmanager
    def scheduler_authority(self, *, blocking: bool = False) -> Iterator[bool]:
        self.ensure_layout()
        user_id = os.getuid() if hasattr(os, "getuid") else 0
        global_lock = (
            Path(tempfile.gettempdir())
            / f"qqtools-qexp-machine-{user_id}"
            / "agent-authority.lock"
        )
        global_lock.parent.mkdir(parents=True, exist_ok=True)
        with exclusive(global_lock, blocking=blocking) as has_machine_authority:
            if not has_machine_authority:
                yield False
                return
            with exclusive(self.paths["scheduler_lock"], blocking=blocking) as acquired:
                yield acquired

    @contextmanager
    def migration_guard(self, *, blocking: bool = True) -> Iterator[bool]:
        """Serialize migration against dispatch and machine reservation changes."""
        self.ensure_layout()
        with exclusive(self.paths["locks"] / "migration.lock", blocking=blocking) as acquired:
            yield acquired

    @contextmanager
    def agent_lifecycle_guard(self, *, blocking: bool = True) -> Iterator[bool]:
        """Serialize machine-agent start, stop, restart, and readiness checks."""
        self.ensure_layout()
        with exclusive(self.paths["locks"] / "activation.lock", blocking=blocking) as acquired:
            yield acquired

    @contextmanager
    def registry_guard(self, *, blocking: bool = True) -> Iterator[bool]:
        self.ensure_layout()
        with exclusive(self.paths["registry_lock"], blocking=blocking) as acquired:
            yield acquired

    def load_registry(self) -> tuple[int, list[ProjectBinding]]:
        if not self.paths["registry"].exists():
            return 0, []
        value = read_json(self.paths["registry"])
        registry = value.get("registry")
        if not isinstance(registry, dict) or registry.get("version") != REGISTRY_VERSION:
            raise RuntimeError("machine registry is malformed or unsupported.")
        revision = registry.get("revision")
        bindings = registry.get("bindings")
        if not isinstance(revision, int) or revision < 0 or not isinstance(bindings, list):
            raise RuntimeError("machine registry is malformed.")
        return revision, [ProjectBinding.from_dict(item) for item in bindings]

    def _save_registry(self, revision: int, bindings: list[ProjectBinding]) -> None:
        atomic_replace(
            self.paths["registry"],
            {
                "registry": {
                    "version": REGISTRY_VERSION,
                    "revision": revision,
                    "updated_at": utc_now(),
                    "bindings": [binding.to_dict() for binding in sorted(bindings, key=lambda item: item.project_id)],
                }
            },
        )

    def ensure_binding(
            self, shared_root: str | Path, machine_name: str, *, enabled: bool = True
    ) -> tuple[ProjectBinding, bool]:
        """Persist one binding, returning whether this call created it."""
        cfg = load_root_config(shared_root, machine_name, require_initialized=True)
        identity_path = shared_paths(cfg.shared_root)["project"] / "identity.json"
        identity = read_json(identity_path).get("project", {}) if identity_path.exists() else {}
        stable_id = identity.get("project_id")
        if not isinstance(stable_id, str) or not stable_id:
            raise RuntimeError(f"qexp project identity is malformed: {identity_path}")
        validate_existing_filesystem_qualification(cfg)
        record = load_machine_record(cfg)
        if not record or record.get("machine", {}).get("machine_name") != machine_name:
            raise RuntimeError(f"machine {machine_name!r} is not initialized in {cfg.shared_root}.")
        binding = ProjectBinding(stable_id, cfg.shared_root, machine_name, enabled)
        with self.registry_guard():
            revision, bindings = self.load_registry()
            if binding in bindings:
                return binding, False
            if any(item.project_id == binding.project_id for item in bindings):
                raise ValueError(f"project {binding.project_id!r} is already registered.")
            if any(item.shared_root == binding.shared_root for item in bindings):
                raise ValueError(f"project root {binding.shared_root} is already registered.")
            self._save_registry(revision + 1, [*bindings, binding])
        return binding, True

    def add_binding(self, shared_root: str | Path, machine_name: str, *, enabled: bool = True) -> ProjectBinding:
        """Add one new binding, rejecting an already registered project."""
        binding, is_added = self.ensure_binding(shared_root, machine_name, enabled=enabled)
        if not is_added:
            raise ValueError(f"project {binding.project_id!r} is already registered.")
        return binding

    def _legacy_evidence_roots(
        self, binding: ProjectBinding
    ) -> tuple[dict[str, Path], dict[str, Path]] | None:
        migration_path = self.migration_path(binding.project_id)
        if not migration_path.exists():
            return None
        migration = read_json(migration_path).get("migration", {})
        source_value = migration.get("legacy_runtime_root")
        if not isinstance(source_value, str) or not source_value:
            return None
        return local_paths(Path(source_value)), self.project_paths(binding.project_id)

    def _move_legacy_evidence(
        self,
        binding: ProjectBinding,
        names: tuple[str, ...],
        *,
        is_destination_authoritative: bool,
    ) -> None:
        roots = self._legacy_evidence_roots(binding)
        if roots is None:
            return
        source_paths, destination_paths = roots
        for name in names:
            source_root = source_paths[name]
            if not source_root.is_dir():
                continue
            for path in sorted(source_root.rglob("*.json")):
                if not path.is_file():
                    continue
                destination = destination_paths[name] / path.relative_to(source_root)
                source_value = read_json(path)
                if destination.exists():
                    if (
                        not is_destination_authoritative
                        and read_json(destination) != source_value
                    ):
                        raise RuntimeError(
                            f"legacy evidence conflicts during migration: {destination}"
                        )
                else:
                    atomic_replace(destination, source_value)
                path.unlink(missing_ok=True)

    def import_legacy_evidence(self, binding: ProjectBinding) -> None:
        """Move evidence whose only writer was the stopped legacy agent."""
        self._move_legacy_evidence(
            binding,
            LEGACY_AGENT_EVIDENCE + LEGACY_RUNNER_INBOX,
            is_destination_authoritative=True,
        )

    def drain_legacy_runner_evidence(self, binding: ProjectBinding) -> None:
        """Move late immutable records written by a runner launched before migration."""
        self._move_legacy_evidence(
            binding,
            LEGACY_RUNNER_INBOX,
            is_destination_authoritative=False,
        )

    def _find_binding(self, bindings: list[ProjectBinding], identifier: str | Path) -> ProjectBinding:
        candidate = str(identifier)
        canonical = Path(candidate).expanduser().resolve() if candidate.endswith(".qexp") or "/" in candidate else None
        for binding in bindings:
            if binding.project_id == candidate or (canonical is not None and binding.shared_root == canonical):
                return binding
        raise ValueError(f"machine registry has no project {candidate!r}.")

    def set_enabled(self, identifier: str | Path, enabled: bool) -> ProjectBinding:
        with self.registry_guard():
            revision, bindings = self.load_registry()
            current = self._find_binding(bindings, identifier)
            updated = ProjectBinding(current.project_id, current.shared_root, current.machine_name, enabled)
            self._save_registry(revision + 1, [updated if item == current else item for item in bindings])
        return updated

    def remove_binding(self, identifier: str | Path) -> ProjectBinding:
        with self.migration_guard():
            with self.registry_guard():
                revision, bindings = self.load_registry()
                binding = self._find_binding(bindings, identifier)
                if binding.enabled:
                    raise ValueError(
                        "disable a project before removing it from the machine registry."
                    )
                blockers = self.binding_blockers(binding)
                if blockers:
                    raise RuntimeError(
                        "cannot remove project with active local evidence: "
                        + ", ".join(blockers)
                    )
                project_root = self.project_paths(binding.project_id)["root"]
                if project_root.exists():
                    shutil.rmtree(project_root)
                self._save_registry(
                    revision + 1, [item for item in bindings if item != binding]
                )
        return binding

    def binding_blockers(self, binding: ProjectBinding) -> list[str]:
        blockers: list[str] = []
        for name in ("provisional", "active"):
            for path in iter_json(self.paths[name]):
                reservation = read_json(path).get("reservation", {})
                if reservation.get("project_id") == binding.project_id:
                    blockers.append(f"reservation:{reservation.get('reservation_id', path.stem)}")
        for directory in (
            "processes",
            "registrations",
            "launch_intents",
            "observations",
            "termination_decisions",
        ):
            root = self.project_paths(binding.project_id)[directory]
            for path in sorted(root.rglob("*.json")) if root.is_dir() else []:
                blockers.append(f"{directory}:{path.stem}")
        return blockers

    def binding_state(self, binding: ProjectBinding) -> str:
        if binding.enabled:
            return "enabled"
        return "draining" if self.binding_blockers(binding) else "disabled"

    def load_cursor(self) -> str | None:
        if not self.paths["cursor"].exists():
            return None
        cursor = read_json(self.paths["cursor"]).get("cursor", {})
        value = cursor.get("next_project_id")
        return value if isinstance(value, str) else None

    def save_cursor(self, project_id: str | None) -> None:
        atomic_replace(self.paths["cursor"], {"cursor": {"next_project_id": project_id, "updated_at": utc_now()}})

    def execution_context(self, cfg: RootConfig) -> ExecutionContext:
        """Pair project operations with the shared machine reservation backend when registered."""
        return ExecutionContext(cfg, self, self.matching_binding(cfg))

    def claim_permitted(self, binding: ProjectBinding) -> bool:
        """Revalidate that an unchanged binding remains enabled for a new claim."""
        with self.registry_guard():
            _, bindings = self.load_registry()
        return any(item == binding and item.enabled for item in bindings)

    @contextmanager
    def enabled_claim_guard(self, binding: ProjectBinding) -> Iterator[bool]:
        """Hold registry state stable while a dispatcher creates one new claim."""
        with self.registry_guard():
            _, bindings = self.load_registry()
            yield any(item == binding and item.enabled for item in bindings)

    def matching_binding(self, cfg: RootConfig) -> ProjectBinding | None:
        _, bindings = self.load_registry()
        identity_path = shared_paths(cfg.shared_root)["project"] / "identity.json"
        if not identity_path.exists():
            return None
        project_id = read_json(identity_path).get("project", {}).get("project_id")
        for binding in bindings:
            if (
                binding.project_id == project_id
                and binding.shared_root == cfg.shared_root
                and binding.machine_name == cfg.machine_name
            ):
                return binding
        return None
