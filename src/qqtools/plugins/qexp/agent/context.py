"""Machine-local runtime ownership for qexp multi-project scheduling."""

from __future__ import annotations

import errno
import os
import shlex
import shutil
import tempfile
import time
import uuid
from contextlib import closing, contextmanager
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any, Iterator

from ..config_types import RootConfig
from ..infrastructure.host import host_instance_id as _host_instance_id
from ..layout import load_machine_record, load_root_config
from ..runtime.authority_scan import is_path_present, iter_evidence_files, validate_evidence_path
from ..runtime.group_discovery.session_owner import GroupSourceOwner
from ..runtime.locks import exclusive, shared
from ..runtime.paths import local_paths, machine_project_paths, machine_runtime_paths, shared_paths
from ..runtime.records import utc_now
from ..runtime.store import atomic_replace, iter_json, read_json
from ..runtime.work_budget import AdaptiveBatchSizer
from .bindings import ProjectBinding
from .dispatch_probe import PrimaryProbeSession
from .identity import MachineRuntimeUninitializedError, load_identity_record, require_fresh_runtime
from .registration import (
    RECOVERY_REGISTRATION_PROTOCOL,
    RECOVERY_REGISTRATION_VERSION,
    REGISTRATION_VERSION,
    REGISTRY_VERSION,
    MachineRegistration,
)

MACHINE_RUNTIME_ENV = "QEXP_MACHINE_RUNTIME_ROOT"
LEGACY_AGENT_EVIDENCE = (
    "processes",
    "termination_decisions",
    "wrappers",
    "authority_diagnostics",
)
LEGACY_RUNNER_INBOX = ("registrations", "observations", "launch_intents", "events")


class ProjectBindingRequiredError(ValueError):
    """The selected Project has no local machine binding."""


def resolve_machine_runtime_root(value: str | Path | None = None) -> Path:
    """Resolve the machine-local authority root without creating it."""
    configured = value if value is not None else os.environ.get(MACHINE_RUNTIME_ENV)
    root = Path(configured).expanduser().resolve() if configured else Path.home() / ".qqtools" / "qexp-machine"
    if root.name == ".qexp" or (root / "schema" / "version.json").exists():
        raise ValueError("QEXP_MACHINE_RUNTIME_ROOT must not point to a project .qexp root.")
    if root.exists() and not root.is_dir():
        raise ValueError("QEXP_MACHINE_RUNTIME_ROOT must be a directory.")
    return root


def default_machine_runtime_root() -> Path:
    """Return the canonical runtime root without consulting environment overrides."""
    return (Path.home() / ".qqtools" / "qexp-machine").resolve()


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


def resolve_execution_context(cfg: RootConfig, machine_runtime_root: str | Path | None = None) -> ExecutionContext:
    """Resolve the registered machine reservation backend for a project operation."""
    return MachineRuntime(machine_runtime_root).execution_context(cfg)


class MachineRuntime:
    """Disposable local resource state shared by one qexp machine."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = resolve_machine_runtime_root(root)
        self._scheduler_authority_gate = RLock()
        self._scheduler_authority_pid: int | None = None
        self._inventory_lock_depth = 0
        self._config_lock_depth = 0
        self.paths = machine_runtime_paths(self.root)
        self.last_diagnostic_publish_ns: int | None = None
        self.ready_batch_sizers: dict[str, AdaptiveBatchSizer] = {}
        self.primary_probe = PrimaryProbeSession()
        # Set by the most recent bounded dispatch cycle for on-demand idle exit.
        self.last_cycle_had_demand = True
        # Set when the current process validates and consumes a current binding.
        self.last_cycle_consumed_binding = False
        # Upgrade discovery is metadata-only when no project has pending work.  These fields are
        # intentionally process-local; the project journal remains the source of truth.
        self.upgrade_registry_revision: int | None = None
        self.upgrade_discovery_complete = False
        self.upgrade_pending_projects: set[str] = set()
        # Pending Group service activation is resumable maintenance and must
        # not by itself retain an otherwise idle on-demand agent.  Discovery
        # classifies pending upgrades that still carry an idle-exit obligation
        # into this separate set.
        self.upgrade_idle_blocked_projects: set[str] = set()
        self.recovery_enrollment_pending_projects: set[str] = set()
        self.upgrade_runnable_projects: set[str] = set()
        self.upgrade_probe_deadlines: dict[str, float] = {}
        self.upgrade_probe_budget = 4
        self.upgrade_next_pass_at = 0.0
        self.notification_next_pass_at = 0.0
        self.upgrade_admission_blocked_projects: set[str] = set()
        self.supervisor_generations: dict[str, str | None] = {}
        # A direct runner may need more than one scheduler cycle to publish its
        # durable launch intent.  Keep those handoffs in process memory while
        # the machine reservation remains authoritative.  The dispatch layer
        # owns the record shape; this map deliberately stores opaque values so
        # context.py does not depend on the executor implementation.
        self.pending_launch_handoffs: dict[tuple[str, str], Any] = {}
        self.group_source_owner = GroupSourceOwner()
        self.registration = MachineRegistration(
            self.root,
            ensure_layout=self.ensure_layout,
            current_instance_id=lambda: self.instance_id,
        )

    @property
    def instance_id(self) -> str:
        """Return identity bound to this runtime and the current host."""
        self.ensure_layout()
        try:
            value = read_json(self.paths["identity"]).get("machine_runtime", {})
            instance_id = value.get("instance_id")
            effective_id = value.get("runtime_id")
        except (OSError, TypeError, ValueError):
            instance_id = None
            effective_id = None
        if not isinstance(instance_id, str) or not instance_id:
            instance_id = uuid.uuid4().hex
            atomic_replace(self.paths["identity"], {"machine_runtime": {"instance_id": instance_id}})
            effective_id = None
        # A compatibility import can only preserve an already-effective ID;
        # it has no way to reverse the host-bound digest into the old random
        # seed.  New identities retain the random seed and derive the public
        # 64-hex value on the current host.
        if (
            isinstance(effective_id, str)
            and len(effective_id) == 64
            and all(char in "0123456789abcdef" for char in effective_id)
        ):
            return effective_id
        return sha256(f"{instance_id}\0{_host_instance_id()}".encode()).hexdigest()

    @property
    def has_identity(self) -> bool:
        """Return whether a durable identity exists without creating runtime state."""
        try:
            value = read_json(self.paths["identity"]).get("machine_runtime", {})
        except (FileNotFoundError, OSError, TypeError, ValueError):
            return False
        instance_id = value.get("instance_id") if isinstance(value, dict) else None
        return isinstance(instance_id, str) and bool(instance_id)

    def require_identity(self) -> None:
        """Distinguish fresh runtimes from missing, unreadable or corrupt identity."""
        if load_identity_record(self.paths["identity"]) is None:
            require_fresh_runtime(self.root)
            raise MachineRuntimeUninitializedError(
                f"qexp machine runtime is uninitialized at {self.root}; run 'qexp init --machine NAME'."
            )

    def require_initialized(self) -> None:
        """Reject operational use unless identity and global configuration are valid."""
        self.require_identity()
        from .config import load_agent_config

        load_agent_config(self, require_initialized=True)

    def ensure_layout(self, *, create_identity: bool = True) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if not os.access(self.root, os.W_OK | os.X_OK):
            raise RuntimeError(f"machine runtime root is not writable: {self.root}")
        for name in (
            "locks",
            "agent",
            "provisional",
            "active",
            "released",
            "cpu_provisional",
            "cpu_active",
            "cpu_released",
            "projects",
            "upgrades",
            "diagnostics",
            "archives",
            "generations",
        ):
            self.paths[name].mkdir(parents=True, exist_ok=True)
        self.paths["cpu_policy"].parent.mkdir(parents=True, exist_ok=True)
        self.paths["cursor"].parent.mkdir(parents=True, exist_ok=True)
        self.paths["gpu_policy_observation"].parent.mkdir(parents=True, exist_ok=True)
        self.paths["gpu_policy_warnings"].parent.mkdir(parents=True, exist_ok=True)
        if create_identity and not self.paths["identity"].exists():
            atomic_replace(self.paths["identity"], {"machine_runtime": {"instance_id": uuid.uuid4().hex}})

    def ensure_compatibility_identity(self) -> str:
        """Synthesize an identity only for an existing pre-feature registry."""
        if self.has_identity:
            return self.instance_id
        if not self.paths["registry"].exists():
            raise RuntimeError("qexp machine runtime is uninitialized; run 'qexp init --machine NAME'.")
        try:
            value = read_json(self.paths["registry"])
            bindings = value.get("registry", {}).get("bindings", [])
        except (OSError, TypeError, ValueError):
            bindings = []
        preserved = next(
            (
                item.get("runtime_instance_id")
                for item in bindings
                if isinstance(item, dict)
                and isinstance(item.get("runtime_instance_id"), str)
                and item["runtime_instance_id"]
            ),
            None,
        )
        self.ensure_layout()
        if isinstance(preserved, str) and len(preserved) == 64:
            atomic_replace(
                self.paths["identity"],
                {"machine_runtime": {"instance_id": preserved, "runtime_id": preserved, "compatibility": True}},
            )
        return self.instance_id

    def project_paths(self, project_id: str) -> dict[str, Path]:
        return machine_project_paths(self.root, project_id)

    def migration_path(self, project_id: str) -> Path:
        return self.project_paths(project_id)["root"] / "migration.json"

    @contextmanager
    def scheduler_authority(self, *, blocking: bool = False) -> Iterator[bool]:
        self.ensure_layout()
        user_id = os.getuid() if hasattr(os, "getuid") else 0
        global_lock = Path(tempfile.gettempdir()) / f"qqtools-qexp-machine-{user_id}" / "agent-authority.lock"
        global_lock.parent.mkdir(parents=True, exist_ok=True)
        with exclusive(global_lock, blocking=blocking) as has_machine_authority:
            if not has_machine_authority:
                yield False
                return
            with exclusive(self.paths["scheduler_lock"], blocking=blocking) as acquired:
                if not acquired:
                    yield False
                    return
                self._scheduler_authority_pid = os.getpid()
                try:
                    yield True
                finally:
                    with self._scheduler_authority_gate:
                        self._scheduler_authority_pid = None

    @contextmanager
    def migration_guard(self, *, blocking: bool = True) -> Iterator[bool]:
        """Serialize migration against dispatch and machine reservation changes."""
        self.ensure_layout()
        with exclusive(self.paths["locks"] / "migration.lock", blocking=blocking) as acquired:
            yield acquired

    @contextmanager
    def migration_read_guard(self, *, blocking: bool = True) -> Iterator[bool]:
        """Exclude migration/removal while allowing dispatch and enrollment together.

        Scheduler authority still serializes dispatch. Registry and registration
        guards serialize binding mutations; this guard only protects migration
        ownership and reservation import, not ordinary project activity.
        """
        self.ensure_layout()
        with shared(self.paths["locks"] / "migration.lock", blocking=blocking) as acquired:
            yield acquired

    @contextmanager
    def agent_lifecycle_guard(self, *, blocking: bool = True) -> Iterator[bool]:
        """Serialize machine-agent start, stop, restart, and readiness checks."""
        # Read-only lifecycle ownership must not implicitly create a machine
        # identity.  ``qexp init`` relies on this distinction to tell an empty
        # runtime from a replacement, while initialized callers still have
        # their durable layout in place.
        self.ensure_layout(create_identity=False)
        with exclusive(self.paths["locks"] / "activation.lock", blocking=blocking) as acquired:
            yield acquired

    @contextmanager
    def inventory_guard(self, *, blocking: bool = True) -> Iterator[bool]:
        """Serialize inventory publication independently from the registry."""
        self.ensure_layout(create_identity=False)
        self._inventory_lock_depth += 1
        try:
            with exclusive(self.paths["inventory_lock"], blocking=blocking) as acquired:
                yield acquired
        finally:
            self._inventory_lock_depth -= 1

    @contextmanager
    def config_guard(self, *, blocking: bool = True) -> Iterator[bool]:
        """Serialize global configuration revisions."""
        self.ensure_layout(create_identity=False)
        self._config_lock_depth += 1
        try:
            with exclusive(self.paths["config_lock"], blocking=blocking) as acquired:
                yield acquired
        finally:
            self._config_lock_depth -= 1

    @contextmanager
    def registry_guard(self, *, blocking: bool = True) -> Iterator[bool]:
        with self.registration.registry_guard(blocking=blocking) as acquired:
            yield acquired

    def load_registry(self) -> tuple[int, list[ProjectBinding]]:
        return self.registration.load_registry()

    def _save_registry(self, revision: int, bindings: list[ProjectBinding]) -> None:
        self.registration.save_registry_locked(revision, bindings)

    def _save_registration_transaction(
        self,
        *,
        revision: int,
        bindings: list[ProjectBinding],
        registrations: list[tuple[RootConfig, dict[str, Any] | None]],
        machine_records: list[tuple[RootConfig, dict[str, Any] | None]],
    ) -> None:
        self.registration._save_registration_transaction(
            revision=revision,
            bindings=bindings,
            registrations=registrations,
            machine_records=machine_records,
        )

    def _rollback_registration_transaction(self) -> None:
        self.registration.rollback_pending_locked()

    def ensure_binding(
        self,
        shared_root: str | Path,
        machine_name: str,
        *,
        enabled: bool = True,
        adopt_existing: bool = False,
    ) -> tuple[ProjectBinding, bool]:
        return self.registration.ensure_binding(
            shared_root,
            machine_name,
            enabled=enabled,
            adopt_existing=adopt_existing,
        )

    def _acquire_registration(
        self,
        cfg: RootConfig,
        project_id: str,
        runtime_instance_id: str,
        current: ProjectBinding | None,
        *,
        adopt_existing: bool,
    ) -> dict[str, str]:
        return self.registration._acquire_registration(
            cfg,
            project_id,
            runtime_instance_id,
            current,
            adopt_existing=adopt_existing,
        )

    def _acquire_registration_locked(
        self,
        cfg: RootConfig,
        project_id: str,
        runtime_instance_id: str,
        current: ProjectBinding | None,
        *,
        adopt_existing: bool,
    ) -> dict[str, str]:
        return self.registration._acquire_registration_locked(
            cfg,
            project_id,
            runtime_instance_id,
            current,
            adopt_existing=adopt_existing,
        )

    def _replace_registration(
        self,
        cfg: RootConfig,
        project_id: str,
        runtime_instance_id: str,
        old: ProjectBinding,
        *,
        adopt_existing: bool,
    ) -> dict[str, str]:
        return self.registration._replace_registration(
            cfg,
            project_id,
            runtime_instance_id,
            old,
            adopt_existing=adopt_existing,
        )

    @contextmanager
    def _registration_guard(self, cfg: RootConfig) -> Iterator[None]:
        with self.registration._registration_guard(cfg):
            yield

    @contextmanager
    def _registration_guards(self, *configs: RootConfig) -> Iterator[None]:
        with self.registration._registration_guards(*configs):
            yield

    def _supersede_registration(self, binding: ProjectBinding, replacement_generation: str) -> None:
        self.registration._supersede_registration(binding, replacement_generation)

    @staticmethod
    def _supersede_registration_locked(
        binding: ProjectBinding,
        cfg: RootConfig,
        replacement_generation: str,
    ) -> None:
        MachineRegistration._supersede_registration_locked(binding, cfg, replacement_generation)

    @staticmethod
    def _validate_registration_record(record: dict[str, Any], project_id: str, cfg: RootConfig) -> None:
        MachineRegistration.validate_registration_record(record, project_id, cfg)

    def prepare_recovery_registration(self, binding: ProjectBinding, *, blocking: bool = False) -> bool:
        """Fence old agent admission for a current machine-owned binding.

        This is only enrollment preparation, not process-capture completion or
        permission to use sole-index discovery. The lifecycle owner calls it
        while holding scheduler authority; ordinary registration stays version 1.
        Busy local lifecycle locks defer it unless a background caller opts into
        waiting. Unfinished legacy migration always defers preparation.
        """
        if self._scheduler_authority_pid != os.getpid():
            raise RuntimeError("recovery registration preparation requires machine scheduler authority.")
        with self._scheduler_authority_gate:
            if self._scheduler_authority_pid != os.getpid():
                raise RuntimeError("recovery registration preparation requires machine scheduler authority.")
            return self._prepare_recovery_registration(binding, blocking=blocking)

    def _prepare_recovery_registration(self, binding: ProjectBinding, *, blocking: bool) -> bool:
        with self.migration_read_guard(blocking=blocking) as acquired:
            if not acquired:
                return False
            with self.registry_guard(blocking=blocking) as registered:
                if not registered:
                    return False
                if not is_path_present(self.paths["registration_transaction"]):
                    return self.registration.publish_recovery_registration_locked(binding)
        # Rollback may restore several bindings and version-1 snapshots. Release
        # the shared fence before taking the exclusive one; never upgrade a lock.
        with self.migration_guard(blocking=blocking) as acquired:
            if not acquired:
                return False
            with self.registry_guard(blocking=blocking) as registered:
                if not registered:
                    return False
                self.registration.rollback_pending_locked()
                return self.registration.publish_recovery_registration_locked(binding)

    def _publish_recovery_registration_locked(self, binding: ProjectBinding) -> bool:
        return self.registration.publish_recovery_registration_locked(binding)

    @staticmethod
    def _registration_state(record: dict[str, Any]) -> str:
        return MachineRegistration.registration_state(record)

    def registration_status(self, binding: ProjectBinding) -> dict[str, Any]:
        return self.registration.registration_status(binding)

    def refresh_binding_eligibility(self, binding: ProjectBinding) -> bool:
        return self.registration.refresh_binding_eligibility(binding)

    def reactivate_binding(self, binding: ProjectBinding) -> bool:
        return self.registration.reactivate_binding(binding)

    @contextmanager
    def binding_write_guard(self, binding: ProjectBinding) -> Iterator[bool]:
        with self.registration.binding_write_guard(binding) as eligible:
            yield eligible

    def binding_write_eligible(self, binding: ProjectBinding, *, renew: bool = False) -> bool:
        return self.registration.binding_write_eligible(binding, renew=renew)

    def add_binding(
        self,
        shared_root: str | Path,
        machine_name: str,
        *,
        enabled: bool = True,
        adopt_existing: bool = False,
    ) -> ProjectBinding:
        binding, is_added = self.ensure_binding(
            shared_root,
            machine_name,
            enabled=enabled,
            adopt_existing=adopt_existing,
        )
        if not is_added:
            raise ValueError(f"project {binding.project_id!r} is already registered.")
        return binding

    def _legacy_evidence_roots(self, binding: ProjectBinding) -> tuple[dict[str, Path], dict[str, Path]] | None:
        migration_path = self.migration_path(binding.project_id)
        if not is_path_present(migration_path):
            return None
        migration = read_json(migration_path).get("migration", {})
        source_value = migration.get("legacy_runtime_root")
        if not isinstance(source_value, str) or not source_value:
            return None
        source_root = Path(source_value)
        if not source_root.is_absolute():
            raise RuntimeError("migration legacy source must be an absolute runtime path")
        source_root = source_root.resolve()
        return {"root": source_root, **local_paths(source_root)}, self.project_paths(binding.project_id)

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
        from ..runtime.responsibility_capture import CaptureBusy, capture_cleanup_guard
        from ..runtime.responsibility_import import RECORD_KEYS, move_legacy_record
        from ..runtime.responsibility_store import DurableIO

        with capture_cleanup_guard(source_paths["root"], destination_paths["root"]) as can_cleanup:
            if not can_cleanup:
                raise CaptureBusy("legacy evidence import is blocked by writer capture")
            for name in names:
                source_root = source_paths[name]
                for path in sorted(iter_evidence_files(source_root, recursive=True)):
                    destination = destination_paths[name] / path.relative_to(source_root)
                    validate_evidence_path(destination, self.root)
                    if name in RECORD_KEYS:
                        move_legacy_record(
                            destination_paths["root"],
                            source_paths["root"],
                            name,
                            path,
                            destination,
                            is_destination_authoritative=is_destination_authoritative,
                        )
                        continue
                    if not validate_evidence_path(path, source_paths["root"]):
                        continue
                    source_value = read_json(path)
                    if validate_evidence_path(destination, self.root):
                        if not is_destination_authoritative and read_json(destination) != source_value:
                            raise RuntimeError(f"legacy evidence conflicts during migration: {destination}")
                    else:
                        atomic_replace(destination, source_value)
                    for directory in destination.resolve().parents:
                        DurableIO().sync_directory(directory, "legacy_event_destination")
                    DurableIO().delete(path)

    def import_legacy_evidence(self, binding: ProjectBinding) -> None:
        """Move evidence whose only writer was the stopped legacy agent."""
        self._move_legacy_evidence(
            binding,
            LEGACY_AGENT_EVIDENCE + LEGACY_RUNNER_INBOX,
            is_destination_authoritative=True,
        )

    def drain_legacy_runner_evidence(self, binding: ProjectBinding) -> None:
        """Move late immutable records written by a runner launched before migration."""
        from .recovery_discovery import read_binding_capture

        proof = read_binding_capture(self, binding)
        names = LEGACY_RUNNER_INBOX
        if proof is not None:
            # Captured Attempt inboxes are refreshed by direct membership paths.
            # Events have their own pending outbox and are not capture members.
            if proof["legacy_source"] is None:
                return
            names = ("events",)
            events = local_paths(Path(proof["legacy_source"]))["events"]
            with closing(iter_evidence_files(events, recursive=True)) as pending:
                if next(pending, None) is None:
                    return
        self._move_legacy_evidence(
            binding,
            names,
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
        return self.registration.set_enabled(identifier, enabled)

    def remove_binding(self, identifier: str | Path) -> ProjectBinding:
        """Remove a disabled partition, allowing an active capture to finish first."""
        from ..runtime.responsibility_capture import CaptureBusy

        deadline = time.monotonic() + 30.0
        while True:
            try:
                return self._remove_binding_once(identifier)
            except CaptureBusy:
                # Never keep lifecycle locks while the background worker needs
                # them to finish its bounded census and publish completion.
                if self._scheduler_authority_pid == os.getpid() or time.monotonic() >= deadline:
                    raise
                with exclusive(self.paths["scheduler_lock"], blocking=False) as is_stopped:
                    if is_stopped:
                        raise
                time.sleep(0.05)

    def _remove_binding_once(self, identifier: str | Path) -> ProjectBinding:
        from ..runtime.responsibility_capture import CaptureBusy, capture_cleanup_guard
        from ..runtime.responsibility_completion import COMPLETION_FILE, release_completed_source

        with self.migration_guard():
            with self.registry_guard():
                revision, bindings = self.load_registry()
                binding = self._find_binding(bindings, identifier)
                if binding.enabled:
                    raise ValueError("disable a project before removing it from the machine registry.")
                project_root = self.project_paths(binding.project_id)["root"]
                has_migration = is_path_present(self.migration_path(binding.project_id))
                try:
                    roots = self._legacy_evidence_roots(binding) if has_migration else None
                except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError) as exc:
                    raise RuntimeError(
                        "cannot remove project with active local evidence: legacy_migration_unavailable"
                    ) from exc
                capture_roots = [project_root] if roots is None else [project_root, roots[0]["root"]]
                if is_path_present(project_root / COMPLETION_FILE):
                    try:
                        release_completed_source(project_root)
                    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                        raise RuntimeError("cannot remove project with unavailable capture completion") from exc
                with capture_cleanup_guard(*capture_roots, should_remove_roots=True) as can_cleanup:
                    if not can_cleanup:
                        blockers = self.binding_blockers(binding)
                        active = [
                            item
                            for item in blockers
                            if item not in {"writer_capture_pending", "legacy:writer_capture_pending"}
                        ]
                        if active:
                            raise RuntimeError(
                                "cannot remove project while writer capture is pending; active local evidence: "
                                + ", ".join(active)
                            )
                        raise CaptureBusy("cannot remove project while writer capture is pending")
                    blockers = self.binding_blockers(binding)
                    if blockers:
                        raise RuntimeError("cannot remove project with active local evidence: " + ", ".join(blockers))
                    if is_path_present(project_root):
                        try:
                            shutil.rmtree(project_root)
                        except OSError as exc:
                            if exc.errno != errno.ENOTEMPTY:
                                raise
                            # A worker that passed its binding check before the
                            # disable may finish one bounded staging write while
                            # removal walks the tree. Release lifecycle locks and
                            # retry after that writer observes the registry.
                            raise CaptureBusy("project runtime removal is busy") from exc
                self.registration.save_registry_locked(revision + 1, [item for item in bindings if item != binding])
        return binding

    def binding_blockers(self, binding: ProjectBinding) -> list[str]:
        blockers: list[str] = []
        for name in ("provisional", "active", "cpu_provisional", "cpu_active"):
            for path in iter_evidence_files(self.paths[name], recursive=True):
                reservation = read_json(path).get("reservation")
                if (
                    path.parent != self.paths[name]
                    or not isinstance(reservation, dict)
                    or reservation.get("reservation_id") != path.stem
                    or not isinstance(reservation.get("project_id"), str)
                    or not reservation["project_id"]
                ):
                    blockers.append(f"reservation_unavailable:{path.stem}")
                elif reservation["project_id"] == binding.project_id:
                    blockers.append(f"reservation:{reservation.get('reservation_id', path.stem)}")
        blockers.extend(sorted(self.iter_recovery_blockers(binding)))
        return blockers

    def iter_recovery_blockers(self, binding: ProjectBinding, *, should_use_capture: bool = False) -> Iterator[str]:
        """Retain local, imported and incomplete-migration recovery obligations."""
        from ..runtime.responsibility import responsibility_root
        from ..runtime.responsibility_capture import has_pending_writer_capture
        from ..runtime.responsibility_cleanup import FLAT_EVIDENCE
        from ..runtime.responsibility_completion import is_source_released
        from ..runtime.responsibility_store import Ledger
        from .recovery_discovery import read_binding_capture

        paths = self.project_paths(binding.project_id)
        is_qualified = False
        proof = None
        if should_use_capture:
            try:
                proof = read_binding_capture(self, binding)
                is_qualified = proof is not None
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                yield "recovery_capture_unavailable"
                return
        evidence_names = () if is_qualified else (*FLAT_EVIDENCE, "termination_decisions")
        sources = [("", paths, (*evidence_names, "events"))]
        migration_path = self.migration_path(binding.project_id)
        if is_path_present(migration_path):
            try:
                migration = read_json(migration_path).get("migration")
                if not isinstance(migration, dict):
                    raise ValueError("invalid migration record")
                source = migration.get("legacy_runtime_root")
                if not isinstance(source, str) or not Path(source).is_absolute():
                    raise ValueError("migration has no absolute legacy source")
                if migration.get("state") != "active":
                    yield "legacy_migration_incomplete"
                roots = self._legacy_evidence_roots(binding)
                if roots is None:
                    raise ValueError("migration has no legacy source")
                sources.append(
                    (
                        "legacy:",
                        roots[0],
                        (
                            *evidence_names,
                            "events",
                            "active",
                            "provisional",
                            "cpu_active",
                            "cpu_provisional",
                        ),
                    )
                )
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                yield "legacy_migration_unavailable"
        for prefix, source_paths, directories in sources:
            try:
                # Qualified target completion already rules out pending capture
                # without a directory sync. An unreleased source hold still
                # requires its owning enrollment worker to finish release.
                if is_qualified:
                    is_pending = bool(prefix) and not is_source_released(paths["root"], proof)
                else:
                    is_pending = has_pending_writer_capture(source_paths["root"])
                if is_pending:
                    yield f"{prefix}writer_capture_pending"
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                yield f"{prefix}writer_capture_unavailable"
            for directory in directories:
                root = source_paths[directory]
                try:
                    with closing(iter_evidence_files(root, recursive=True)) as evidence:
                        for path in evidence:
                            yield f"{prefix}{directory}:{path.stem}"
                except OSError:
                    yield f"{prefix}{directory}:unavailable"
        membership = responsibility_root(paths["root"])
        if is_path_present(membership):
            try:
                if Ledger(membership).has_members():
                    yield "recovery_responsibilities"
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                yield "recovery_responsibilities_unavailable"

    def binding_state(self, binding: ProjectBinding) -> str:
        registration_state = self.registration_status(binding)["state"]
        if registration_state == "superseded":
            return "superseded"
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

    def pending_launch_identities(self) -> set[tuple[str, str]]:
        """Return pending ``(project_id, attempt_id)`` handoff identities."""
        return set(self.pending_launch_handoffs)

    def pending_launch_wait_seconds(self, maximum: float) -> float:
        """Bound an agent sleep by the earliest in-process handoff deadline."""
        if maximum < 0:
            raise ValueError("maximum wait must not be negative")
        if not self.pending_launch_handoffs:
            return maximum
        now = time.monotonic()
        earliest: float | None = None
        for pending in self.pending_launch_handoffs.values():
            handoff = getattr(pending, "handoff", pending)
            deadline = getattr(handoff, "deadline", None)
            if isinstance(deadline, (int, float)):
                retry_not_before = getattr(pending, "retry_not_before", 0.0)
                wake_at = max(deadline, retry_not_before) if isinstance(retry_not_before, (int, float)) else deadline
                earliest = wake_at if earliest is None else min(earliest, wake_at)
        if earliest is None:
            return maximum
        return min(maximum, max(0.0, earliest - now))

    def execution_context(self, cfg: RootConfig) -> ExecutionContext:
        """Pair project operations with the shared machine reservation backend when registered."""
        return ExecutionContext(cfg, self, self.matching_binding(cfg))

    def resolve_project_binding(self, shared_root: str | Path) -> ProjectBinding:
        """Resolve and validate the unique local binding for a shared project root."""
        root = Path(shared_root).expanduser().resolve()
        identity_path = shared_paths(root)["project"] / "identity.json"
        if not identity_path.exists():
            raise ValueError(f"qexp project identity is missing: {identity_path}")
        identity = read_json(identity_path).get("project")
        if not isinstance(identity, dict):
            raise ValueError(f"qexp project identity is malformed: {identity_path}")
        stable_id = identity.get("project_id")
        identity_root = identity.get("shared_root")
        if not isinstance(stable_id, str) or not stable_id:
            raise ValueError(f"qexp project identity has no stable project ID: {identity_path}")
        if not isinstance(identity_root, str) or Path(identity_root).expanduser().resolve() != root:
            raise ValueError("project identity shared_root does not match the canonical shared root.")

        _, bindings = self.load_registry()
        project_matches = [binding for binding in bindings if binding.project_id == stable_id]
        root_matches = [binding for binding in bindings if binding.shared_root == root]
        matches = [binding for binding in project_matches if binding.shared_root == root]
        if len(project_matches) > 1 or len(root_matches) > 1:
            candidates = project_matches or root_matches
            machines = ", ".join(sorted(binding.machine_name for binding in candidates))
            raise RuntimeError(f"local project binding is ambiguous for {root}: {machines}.")
        if project_matches and root_matches and project_matches[0] != root_matches[0]:
            raise RuntimeError(f"machine registry binding does not match Project truth for {root}.")
        if not matches:
            for record_path in sorted(shared_paths(root)["machines"].glob("*/machine.json")):
                try:
                    record = read_json(record_path)
                    machine = record.get("machine") if isinstance(record, dict) else None
                except (OSError, TypeError, ValueError):
                    raise ValueError(f"qexp machine record is malformed: {record_path}") from None
                if not isinstance(machine, dict) or machine.get("agent_runtime") != "machine":
                    command = shlex.join(
                        [
                            "qexp",
                            "admin",
                            "migrate",
                            "agent",
                            "--project",
                            str(root),
                            "--machine",
                            record_path.parent.name,
                        ]
                    )
                    raise ValueError(f"legacy project metadata detected; run '{command}'.")
            raise ProjectBindingRequiredError(
                f"no local project binding exists for {root}; run 'qexp project register {root}'."
            )
        if len(matches) > 1:
            machines = ", ".join(sorted(binding.machine_name for binding in matches))
            raise RuntimeError(f"local project binding is ambiguous for {root}: {machines}.")

        binding = matches[0]
        cfg = load_root_config(root, binding.machine_name, require_initialized=True)
        record = load_machine_record(cfg)
        machine = record.get("machine", {}) if isinstance(record, dict) else {}
        if not isinstance(machine, dict):
            raise ValueError(f"machine record for {binding.machine_name!r} is malformed in {root}.")
        if (
            machine.get("machine_name") != binding.machine_name
            or machine.get("project_id") != binding.project_id
            or machine.get("shared_root") != str(root)
        ):
            raise ValueError(f"local binding for machine {binding.machine_name!r} does not match Project truth.")
        if machine.get("agent_runtime") != "machine":
            command = shlex.join(
                [
                    "qexp",
                    "admin",
                    "migrate",
                    "agent",
                    "--project",
                    str(root),
                    "--machine",
                    binding.machine_name,
                ]
            )
            raise ValueError(f"legacy project metadata detected; run '{command}'.")
        return binding

    def verified_execution_context(self, shared_root: str | Path) -> ExecutionContext:
        """Build an operational context from the binding-owned local runtime partition."""
        binding = self.resolve_project_binding(shared_root)
        local_root = self.project_paths(binding.project_id)["root"]
        cfg = load_root_config(
            binding.shared_root,
            binding.machine_name,
            local_root,
            require_initialized=True,
        )
        return ExecutionContext(cfg, self, binding)

    def claim_permitted(self, binding: ProjectBinding) -> bool:
        """Revalidate that an unchanged binding remains enabled for a new claim."""
        with self.registry_guard():
            _, bindings = self.load_registry()
        return any(item == binding and item.enabled and self.binding_write_eligible(item) for item in bindings)

    @contextmanager
    def enabled_claim_guard(self, binding: ProjectBinding) -> Iterator[bool]:
        """Fence a claim to an enabled binding's current registration generation."""
        with self.registry_guard():
            _, bindings = self.load_registry()
            if not any(item == binding and item.enabled for item in bindings):
                yield False
                return
            with self.binding_write_guard(binding) as is_eligible:
                yield is_eligible

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
