"""Machine-local runtime ownership for qexp multi-project scheduling."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any, Iterator

from qqtools.version import __version__

from ..config_types import RootConfig
from ..infrastructure.host import host_instance_id as _host_instance_id
from ..layout import load_machine_record, load_machine_registration, load_root_config, save_machine_registration
from ..lease import LeasePolicy, lease_expiry, load_lease_policy, parse_utc
from ..runtime.authority_scan import is_path_present, iter_evidence_files, validate_evidence_path
from ..runtime.group_discovery.session_owner import GroupSourceOwner
from ..runtime.locks import exclusive, machine_lock, shared
from ..runtime.paths import (
    local_paths,
    machine_path,
    machine_project_paths,
    machine_registration_path,
    machine_runtime_paths,
    shared_paths,
)
from ..runtime.ready import ReadyCursor
from ..runtime.records import utc_now
from ..runtime.store import atomic_replace, iter_json, read_json
from ..runtime.work_budget import AdaptiveBatchSizer, diagnostic_increment, diagnostic_observe_ns

MACHINE_RUNTIME_ENV = "QEXP_MACHINE_RUNTIME_ROOT"
REGISTRY_VERSION = 1
REGISTRATION_VERSION = 1
RECOVERY_REGISTRATION_VERSION = 2
RECOVERY_REGISTRATION_PROTOCOL = "qexp-local-responsibility-v1"
LEGACY_AGENT_EVIDENCE = (
    "processes",
    "termination_decisions",
    "wrappers",
    "authority_diagnostics",
)
LEGACY_RUNNER_INBOX = ("registrations", "observations", "launch_intents", "events")


def _registration_renewal_interval(policy: LeasePolicy) -> float:
    # Leave a second service opportunity when a valid Attempt interval is near TTL.
    return min(policy.renew_interval_seconds, policy.ttl_seconds / 2)


def _observe_registration_renewal(previous_expiry: str, policy: LeasePolicy, *, is_reactivation: bool) -> None:
    """Observe completed publication against the previous renewal target."""
    diagnostic_increment("registration.reactivation" if is_reactivation else "registration.renewal")
    try:
        target = parse_utc(previous_expiry) - timedelta(
            seconds=policy.ttl_seconds - _registration_renewal_interval(policy)
        )
        lateness = max(0.0, (datetime.now(timezone.utc) - target).total_seconds())
    except (ValueError, TypeError, OverflowError):
        diagnostic_increment("registration.renewal_lateness_unavailable")
        return
    diagnostic_observe_ns("registration.renewal_lateness", int(lateness * 1_000_000_000))


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
        self.paths = machine_runtime_paths(self.root)
        self.last_diagnostic_publish_ns: int | None = None
        self.ready_batch_sizers: dict[str, AdaptiveBatchSizer] = {}
        self.primary_probe_cursors: dict[tuple[str, ...], ReadyCursor | None] = {}
        self.primary_probe_revisions: dict[tuple[str, ...], int] = {}
        self.primary_probe_complete: dict[tuple[str, ...], bool] = {}
        # Finish each route once per probe round, even when the budget spans calls.
        self.primary_probe_pending_routes: dict[str, set[tuple[str, str, str]]] = {}
        # A temporarily unavailable marker can become claimable without an index
        # revision change.  Keep its position separately from scan completion:
        # dependency waiting is not resource demand and must not deny borrowing.
        self.primary_probe_recheck_cursors: dict[tuple[str, str, str], ReadyCursor | None] = {}
        # The next dependency route to revisit after every route has completed
        # its baseline scan.  Advancing one route at a time prevents an early
        # project from exhausting each slice before later projects are scanned.
        self.primary_probe_recheck_round_cursors: dict[str, tuple[str, str, str]] = {}
        # Set by the most recent bounded dispatch cycle for on-demand idle exit.
        self.last_cycle_had_demand = True
        # Set when the current process validates and consumes a current binding.
        self.last_cycle_consumed_binding = False
        # Upgrade discovery is metadata-only when no project has pending work.  These fields are
        # intentionally process-local; the project journal remains the source of truth.
        self.upgrade_registry_revision: int | None = None
        self.upgrade_discovery_complete = False
        self.upgrade_pending_projects: set[str] = set()
        self.recovery_enrollment_pending_projects: set[str] = set()
        self.upgrade_runnable_projects: set[str] = set()
        self.upgrade_probe_deadlines: dict[str, float] = {}
        self.upgrade_probe_budget = 4
        self.upgrade_next_pass_at = 0.0
        self.upgrade_admission_blocked_projects: set[str] = set()
        self.supervisor_generations: dict[str, str | None] = {}
        self.group_source_owner = GroupSourceOwner()

    @property
    def instance_id(self) -> str:
        """Return identity bound to this runtime and the current host."""
        self.ensure_layout()
        try:
            value = read_json(self.paths["identity"]).get("machine_runtime", {})
            instance_id = value.get("instance_id")
        except (OSError, TypeError, ValueError):
            instance_id = None
        if not isinstance(instance_id, str) or not instance_id:
            instance_id = uuid.uuid4().hex
            atomic_replace(self.paths["identity"], {"machine_runtime": {"instance_id": instance_id}})
        return sha256(f"{instance_id}\0{_host_instance_id()}".encode()).hexdigest()

    def ensure_layout(self) -> None:
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
        ):
            self.paths[name].mkdir(parents=True, exist_ok=True)
        self.paths["cpu_policy"].parent.mkdir(parents=True, exist_ok=True)
        self.paths["cursor"].parent.mkdir(parents=True, exist_ok=True)
        if not self.paths["identity"].exists():
            atomic_replace(self.paths["identity"], {"machine_runtime": {"instance_id": uuid.uuid4().hex}})

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
        try:
            parsed = [ProjectBinding.from_dict(item) for item in bindings]
        except (TypeError, ValueError) as exc:
            raise RuntimeError("machine registry contains a malformed project binding.") from exc
        return revision, parsed

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

    def _save_registration_transaction(
        self,
        *,
        revision: int,
        bindings: list[ProjectBinding],
        registrations: list[tuple[RootConfig, dict[str, Any] | None]],
        machine_records: list[tuple[RootConfig, dict[str, Any] | None]],
    ) -> None:
        """Record enough pre-state to roll back an interrupted registration publication."""
        atomic_replace(
            self.paths["registration_transaction"],
            {
                "registration_transaction": {
                    "revision": revision,
                    "bindings": [binding.to_dict() for binding in bindings],
                    "registrations": [
                        {
                            "shared_root": str(cfg.shared_root),
                            "machine_name": cfg.machine_name,
                            "record": record,
                        }
                        for cfg, record in registrations
                    ],
                    "machine_records": [
                        {
                            "shared_root": str(cfg.shared_root),
                            "machine_name": cfg.machine_name,
                            "record": record,
                        }
                        for cfg, record in machine_records
                    ],
                }
            },
        )

    def _rollback_registration_transaction(self) -> None:
        """Restore the last incomplete registration before accepting a new mutation."""
        path = self.paths["registration_transaction"]
        if not path.exists():
            return
        value = read_json(path).get("registration_transaction")
        if not isinstance(value, dict):
            raise RuntimeError("machine registration transaction is malformed.")
        try:
            revision = value["revision"]
            bindings = [ProjectBinding.from_dict(item) for item in value["bindings"]]
            registrations = value["registrations"]
            machine_records = value["machine_records"]
            if (
                not isinstance(revision, int)
                or not isinstance(registrations, list)
                or not isinstance(machine_records, list)
            ):
                raise ValueError
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("machine registration transaction is malformed.") from exc
        configs: list[RootConfig] = []
        for item in [*registrations, *machine_records]:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("shared_root"), str)
                or not isinstance(item.get("machine_name"), str)
            ):
                raise RuntimeError("machine registration transaction is malformed.")
            cfg = load_root_config(item["shared_root"], item["machine_name"])
            if not any(
                existing.shared_root == cfg.shared_root and existing.machine_name == cfg.machine_name
                for existing in configs
            ):
                configs.append(cfg)
        with self._registration_guards(*configs):
            for item in registrations:
                cfg = load_root_config(item["shared_root"], item["machine_name"])
                registration_path = machine_registration_path(cfg.shared_root, cfg.machine_name)
                record = item.get("record")
                if record is None:
                    registration_path.unlink(missing_ok=True)
                else:
                    save_machine_registration(cfg, record)
            for item in machine_records:
                cfg = load_root_config(item["shared_root"], item["machine_name"])
                record_path = machine_path(cfg.shared_root, cfg.machine_name)
                record = item.get("record")
                if record is None:
                    record_path.unlink(missing_ok=True)
                else:
                    from ..layout import save_machine_record

                    save_machine_record(cfg, record)
            self._save_registry(revision, bindings)
            path.unlink(missing_ok=True)

    def ensure_binding(
        self,
        shared_root: str | Path,
        machine_name: str,
        *,
        enabled: bool = True,
        adopt_existing: bool = False,
    ) -> tuple[ProjectBinding, bool]:
        """Persist one binding after project-owned generation and eligibility checks."""
        cfg = load_root_config(shared_root, machine_name, require_initialized=True)
        identity_path = shared_paths(cfg.shared_root)["project"] / "identity.json"
        identity = read_json(identity_path).get("project", {}) if identity_path.exists() else {}
        stable_id = identity.get("project_id")
        if not isinstance(stable_id, str) or not stable_id:
            raise RuntimeError(f"qexp project identity is malformed: {identity_path}")
        record = load_machine_record(cfg)
        if record is not None and record.get("machine", {}).get("machine_name") != machine_name:
            raise RuntimeError(f"machine {machine_name!r} is not initialized in {cfg.shared_root}.")
        runtime_instance_id = self.instance_id
        with self.registry_guard():
            self._rollback_registration_transaction()
            revision, bindings = self.load_registry()
            current = next(
                (
                    item
                    for item in bindings
                    if item.project_id == stable_id
                    and item.shared_root == cfg.shared_root
                    and item.machine_name == machine_name
                ),
                None,
            )
            same_project = [item for item in bindings if item.project_id == stable_id]
            if current is None and not same_project and any(item.shared_root == cfg.shared_root for item in bindings):
                raise ValueError(f"project root {cfg.shared_root} is already registered.")
            if current is None and same_project:
                old = same_project[0]
                old_status = self.registration_status(old)["state"]
                if old_status == "eligible":
                    raise ValueError(
                        f"project {stable_id!r} is already registered as {old.machine_name!r} "
                        "with active write eligibility."
                    )
                if old_status != "superseded" and not adopt_existing:
                    raise ValueError(
                        f"project {stable_id!r} remains registered as {old.machine_name!r} ({old_status}); "
                        "explicit --adopt-existing is required to replace its logical name."
                    )
            affected_configs = [cfg]
            if current is None and same_project:
                affected_configs.append(same_project[0].root_config())
            registration_records = [(item, load_machine_registration(item)) for item in affected_configs]
            machine_records = [(item, load_machine_record(item)) for item in affected_configs]
            self._save_registration_transaction(
                revision=revision,
                bindings=bindings,
                registrations=registration_records,
                machine_records=machine_records,
            )
            try:
                if current is None and same_project:
                    registration = self._replace_registration(
                        cfg,
                        stable_id,
                        runtime_instance_id,
                        same_project[0],
                        adopt_existing=adopt_existing,
                    )
                else:
                    registration = self._acquire_registration(
                        cfg,
                        stable_id,
                        runtime_instance_id,
                        current,
                        adopt_existing=adopt_existing,
                    )
                binding = ProjectBinding(
                    stable_id,
                    cfg.shared_root,
                    machine_name,
                    current.enabled if current is not None else enabled,
                    registration["generation"],
                    runtime_instance_id,
                    str(self.root),
                )
                if record is None:
                    from ..machine_config import save_machine_config

                    save_machine_config(cfg, agent_mode=None)
                if current is not None:
                    updated = [binding if item == current else item for item in bindings]
                    if updated != bindings:
                        self._save_registry(revision + 1, updated)
                    self.paths["registration_transaction"].unlink(missing_ok=True)
                    return binding, False
                if same_project:
                    bindings = [item for item in bindings if item != same_project[0]]
                if any(item.shared_root == binding.shared_root for item in bindings):
                    raise ValueError(f"project root {binding.shared_root} is already registered.")
                self._save_registry(revision + 1, [*bindings, binding])
                self.paths["registration_transaction"].unlink(missing_ok=True)
            except Exception:
                # Keep the durable transaction for the next registration attempt to
                # inspect and roll back after all nested locks have been released.
                raise
        return binding, True

    def _acquire_registration(
        self,
        cfg: RootConfig,
        project_id: str,
        runtime_instance_id: str,
        current: ProjectBinding | None,
        *,
        adopt_existing: bool,
    ) -> dict[str, str]:
        """Acquire or renew one project-owned logical-machine generation."""
        with self._registration_guard(cfg):
            return self._acquire_registration_locked(
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
        """Acquire registration while the caller holds the project registration fence."""
        policy = load_lease_policy(cfg)
        raw = load_machine_registration(cfg)
        registration = raw.get("registration") if isinstance(raw, dict) else None
        if registration is not None:
            self._validate_registration_record(registration, project_id, cfg)
        current_generation = current.registration_generation if current is not None else None
        same_owner = bool(
            isinstance(registration, dict)
            and registration.get("runtime_instance_id") == runtime_instance_id
            and registration.get("runtime_root") == str(self.root)
            and (current is None or current.runtime_root in {None, str(self.root)})
            and (current_generation is None or registration.get("generation") == current_generation)
        )
        if (
            current is not None
            and registration is None
            and current.runtime_instance_id == runtime_instance_id
            and current.runtime_root in {None, str(self.root)}
        ):
            same_owner = True
        if registration is not None and not same_owner:
            state = self._registration_state(registration)
            if not adopt_existing:
                if state == "eligible":
                    raise ValueError(
                        f"logical machine name {cfg.machine_name!r} is owned by registration "
                        f"generation {registration['generation']!r} with active write eligibility; "
                        "use --adopt-existing only after it is safely invalidated."
                    )
                raise ValueError(
                    f"logical machine name {cfg.machine_name!r} remains owned by previous registration "
                    f"generation {registration['generation']!r} ({state}); explicit --adopt-existing is required."
                )
            if state == "eligible":
                raise ValueError(
                    f"logical machine name {cfg.machine_name!r} still has active competing authority "
                    f"in generation {registration['generation']!r}; adoption is not safe."
                )
        generation = (
            registration.get("generation")
            if same_owner and registration
            else current.registration_generation
            if same_owner and current is not None and current.registration_generation
            else uuid.uuid4().hex
        )
        now = utc_now()
        # The protocol floor belongs to the logical name, including an explicit
        # ownership replacement. Returning old agents must remain fenced.
        version = registration["version"] if registration else REGISTRATION_VERSION
        value = {
            "version": version,
            "project_id": project_id,
            "shared_root": str(cfg.shared_root),
            "machine_name": cfg.machine_name,
            "generation": generation,
            "protocol_version": version,
            "client_version": __version__,
            "runtime_instance_id": runtime_instance_id,
            "runtime_root": str(self.root),
            "state": "eligible",
            "eligibility_expires_at": lease_expiry(policy),
            "created_at": registration.get("created_at", now) if same_owner and registration else now,
            "updated_at": now,
        }
        if version == RECOVERY_REGISTRATION_VERSION:
            value["recovery_protocol"] = RECOVERY_REGISTRATION_PROTOCOL
        save_machine_registration(cfg, {"registration": value})
        return value

    def _replace_registration(
        self,
        cfg: RootConfig,
        project_id: str,
        runtime_instance_id: str,
        old: ProjectBinding,
        *,
        adopt_existing: bool,
    ) -> dict[str, str]:
        """Acquire a replacement name and fence the stale local name atomically."""
        old_cfg = old.root_config()
        with self._registration_guards(cfg, old_cfg):
            registration = self._acquire_registration_locked(
                cfg,
                project_id,
                runtime_instance_id,
                None,
                adopt_existing=adopt_existing,
            )
            self._supersede_registration_locked(old, old_cfg, registration["generation"])
            return registration

    @contextmanager
    def _registration_guard(self, cfg: RootConfig) -> Iterator[None]:
        """Serialize registration replacement and generation-authorized writes."""
        with self._registration_guards(cfg):
            yield

    @contextmanager
    def _registration_guards(self, *configs: RootConfig) -> Iterator[None]:
        """Hold the project fence and all affected logical-machine locks."""
        if not configs:
            raise ValueError("at least one registration config is required.")
        shared_root = configs[0].shared_root
        if any(cfg.shared_root != shared_root for cfg in configs):
            raise ValueError("registration replacement must stay within one project.")
        with exclusive(shared_paths(shared_root)["locks"] / "registrations.lock"):
            with ExitStack() as stack:
                for guarded_cfg in sorted(configs, key=lambda item: item.machine_name):
                    stack.enter_context(machine_lock(guarded_cfg.shared_root, guarded_cfg.machine_name))
                yield

    def _supersede_registration(self, binding: ProjectBinding, replacement_generation: str) -> None:
        """Fence an old logical-name record after a project changes its registered name."""
        cfg = binding.root_config()
        with self._registration_guard(cfg):
            self._supersede_registration_locked(binding, cfg, replacement_generation)

    @staticmethod
    def _supersede_registration_locked(binding: ProjectBinding, cfg: RootConfig, replacement_generation: str) -> None:
        """Supersede a matching generation while its registration fence is held."""
        raw = load_machine_registration(cfg)
        record = raw.get("registration") if isinstance(raw, dict) else None
        if not isinstance(record, dict) or record.get("generation") != binding.registration_generation:
            return
        record = dict(record)
        record.update(
            {
                "state": "superseded",
                "superseded_by_generation": replacement_generation,
                "updated_at": utc_now(),
            }
        )
        save_machine_registration(cfg, {"registration": record})

    @staticmethod
    def _validate_registration_record(record: dict[str, Any], project_id: str, cfg: RootConfig) -> None:
        version = record.get("version")
        if type(version) is not int or version not in {REGISTRATION_VERSION, RECOVERY_REGISTRATION_VERSION}:
            raise RuntimeError("project machine registration uses an unsupported protocol version.")
        if version == RECOVERY_REGISTRATION_VERSION and (
            type(record.get("protocol_version")) is not int
            or record["protocol_version"] != version
            or record.get("recovery_protocol") != RECOVERY_REGISTRATION_PROTOCOL
        ):
            raise RuntimeError("project machine registration has an unsupported recovery protocol.")
        if record.get("project_id") != project_id or record.get("shared_root") != str(cfg.shared_root):
            raise RuntimeError("project machine registration does not match Project identity.")
        if record.get("machine_name") != cfg.machine_name:
            raise RuntimeError("project machine registration does not match the requested logical name.")
        for key in ("generation", "runtime_instance_id", "runtime_root", "eligibility_expires_at"):
            if not isinstance(record.get(key), str) or not record[key]:
                raise RuntimeError("project machine registration is malformed.")

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
                    return self._publish_recovery_registration_locked(binding)
        # Rollback may restore several bindings and version-1 snapshots. Release
        # the shared fence before taking the exclusive one; never upgrade a lock.
        with self.migration_guard(blocking=blocking) as acquired:
            if not acquired:
                return False
            with self.registry_guard(blocking=blocking) as registered:
                if not registered:
                    return False
                self._rollback_registration_transaction()
                return self._publish_recovery_registration_locked(binding)

    def _publish_recovery_registration_locked(self, binding: ProjectBinding) -> bool:
        """Publish under migration exclusion, registry and registration ownership."""
        from ..runtime.responsibility_store import DurableIO

        if binding not in self.load_registry()[1]:
            return False
        cfg = binding.root_config()
        record = load_machine_record(cfg)
        machine = record.get("machine") if isinstance(record, dict) else None
        if not isinstance(machine, dict) or machine.get("agent_runtime") != "machine":
            return False
        migration_path = self.migration_path(binding.project_id)
        if is_path_present(migration_path):
            migration = read_json(migration_path).get("migration")
            if not isinstance(migration, dict) or migration.get("state") != "active":
                return False
        with self.binding_write_guard(binding) as eligible:
            if not eligible:
                return False
            registration = load_machine_registration(cfg)["registration"]
            if registration["version"] == RECOVERY_REGISTRATION_VERSION:
                # A prior replace may be visible after a failed directory
                # sync. Readable preparation is not yet a durable fence.
                DurableIO().sync_directory(
                    machine_registration_path(cfg.shared_root, cfg.machine_name).parent,
                    "recovery_registration_fence",
                )
                return True
            # A reboot must not resurrect a retired version-1 rollback
            # snapshot after the shared protocol fence becomes durable.
            DurableIO().sync_directory(self.paths["registration_transaction"].parent, "recovery_registration")
            registration = dict(registration)
            registration.update(
                version=RECOVERY_REGISTRATION_VERSION,
                protocol_version=RECOVERY_REGISTRATION_VERSION,
                recovery_protocol=RECOVERY_REGISTRATION_PROTOCOL,
                client_version=__version__,
                updated_at=utc_now(),
            )
            save_machine_registration(cfg, {"registration": registration})
            return True

    @staticmethod
    def _registration_state(record: dict[str, Any]) -> str:
        try:
            expires_at = parse_utc(record["eligibility_expires_at"])
        except (KeyError, TypeError, ValueError):
            return "invalid"
        if record.get("state") == "superseded":
            return "superseded"
        return "eligible" if expires_at > datetime.now(timezone.utc) else "expired"

    def registration_status(self, binding: ProjectBinding) -> dict[str, Any]:
        """Return current shared ownership and write-eligibility diagnostics."""
        protocol = {"registration_version": None, "recovery_protocol": None}
        try:
            cfg = binding.root_config()
            raw = load_machine_registration(cfg)
            record = raw.get("registration") if isinstance(raw, dict) else None
            if not isinstance(record, dict):
                return {"state": "unregistered", "write_eligible": False, "generation": None, **protocol}
            protocol = {
                "registration_version": record.get("version"),
                "recovery_protocol": record.get("recovery_protocol"),
            }
            self._validate_registration_record(record, binding.project_id, cfg)
            runtime_instance_id = self.instance_id
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            return {"state": "invalid", "write_eligible": False, "generation": None, "error": str(exc), **protocol}
        same_generation = record.get("generation") == binding.registration_generation
        same_runtime = record.get("runtime_instance_id") == binding.runtime_instance_id == runtime_instance_id
        same_root = record.get("runtime_root") == str(self.root) and binding.runtime_root in {None, str(self.root)}
        state = self._registration_state(record) if same_generation and same_runtime and same_root else "superseded"
        return {
            "state": state,
            "write_eligible": state == "eligible",
            "generation": record.get("generation"),
            "eligibility_expires_at": record.get("eligibility_expires_at"),
            "runtime_instance_id": record.get("runtime_instance_id"),
            **protocol,
        }

    def refresh_binding_eligibility(self, binding: ProjectBinding) -> bool:
        """Renew a current generation, returning false for stale or replaced bindings."""
        with self.binding_write_guard(binding) as is_eligible:
            return is_eligible

    def reactivate_binding(self, binding: ProjectBinding) -> bool:
        """Renew an expired registration when its generation was not superseded."""
        cfg = binding.root_config()
        policy = load_lease_policy(cfg)
        with self._registration_guard(cfg):
            raw = load_machine_registration(cfg)
            record = raw.get("registration") if isinstance(raw, dict) else None
            if not isinstance(record, dict):
                return False
            self._validate_registration_record(record, binding.project_id, cfg)
            if (
                record.get("state") == "superseded"
                or record.get("generation") != binding.registration_generation
                or record.get("runtime_instance_id") != binding.runtime_instance_id
                or binding.runtime_instance_id != self.instance_id
                or record.get("runtime_root") != str(self.root)
                or binding.runtime_root not in {None, str(self.root)}
            ):
                return False
            record = dict(record)
            previous_expiry = record["eligibility_expires_at"]
            record["eligibility_expires_at"] = lease_expiry(policy)
            record["updated_at"] = utc_now()
            save_machine_registration(cfg, {"registration": record})
            _observe_registration_renewal(previous_expiry, policy, is_reactivation=True)
            return True

    @contextmanager
    def binding_write_guard(self, binding: ProjectBinding) -> Iterator[bool]:
        """Fence an authoritative write, renewing the current generation when due."""
        cfg = binding.root_config()
        policy = load_lease_policy(cfg)
        with self._registration_guard(cfg):
            raw = load_machine_registration(cfg)
            record = raw.get("registration") if isinstance(raw, dict) else None
            if not isinstance(record, dict):
                yield False
                return
            self._validate_registration_record(record, binding.project_id, cfg)
            if (
                self._registration_state(record) != "eligible"
                or record.get("generation") != binding.registration_generation
                or record.get("runtime_instance_id") != binding.runtime_instance_id
                or binding.runtime_instance_id != self.instance_id
                or record.get("runtime_root") != str(self.root)
                or binding.runtime_root not in {None, str(self.root)}
            ):
                yield False
                return
            previous_expiry = record["eligibility_expires_at"]
            expires_at = parse_utc(previous_expiry)
            next_expiry = lease_expiry(policy)
            renew_at = expires_at - timedelta(seconds=policy.ttl_seconds - _registration_renewal_interval(policy))
            # Derive scheduling from fenced durable state, never cached authority.
            # A shorter policy horizon also takes effect without waiting for renewal.
            if parse_utc(next_expiry) != expires_at and (
                datetime.now(timezone.utc) >= renew_at or parse_utc(next_expiry) < expires_at
            ):
                record = dict(record)
                record["eligibility_expires_at"] = next_expiry
                record["updated_at"] = utc_now()
                save_machine_registration(cfg, {"registration": record})
                _observe_registration_renewal(previous_expiry, policy, is_reactivation=False)
            yield True

    def binding_write_eligible(self, binding: ProjectBinding, *, renew: bool = False) -> bool:
        """Check current generation authority before an identity-scoped write."""
        if renew:
            try:
                return self.refresh_binding_eligibility(binding)
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                return False
        return bool(self.registration_status(binding).get("write_eligible"))

    def add_binding(
        self,
        shared_root: str | Path,
        machine_name: str,
        *,
        enabled: bool = True,
        adopt_existing: bool = False,
    ) -> ProjectBinding:
        """Add one new binding, rejecting an already registered project."""
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
        with self.registry_guard():
            revision, bindings = self.load_registry()
            current = self._find_binding(bindings, identifier)
            updated = replace(current, enabled=enabled)
            self._save_registry(revision + 1, [updated if item == current else item for item in bindings])
        return updated

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
                        shutil.rmtree(project_root)
                self._save_registry(revision + 1, [item for item in bindings if item != binding])
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
                    raise ValueError("legacy project metadata detected; run 'qexp agent migrate-project'.")
            raise ValueError(f"no local project binding exists for {root}; run 'qexp agent add-project'.")
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
            raise ValueError("legacy project metadata detected; run 'qexp agent migrate-project'.")
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
