"""Machine registration ownership for qexp multi-project scheduling."""

from __future__ import annotations

import math
import stat
import uuid
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from qqtools.version import __version__

from ..config_types import RootConfig
from ..layout import (
    load_machine_record,
    load_machine_registration,
    load_root_config,
    save_machine_record,
    save_machine_registration,
)
from ..lease import LeasePolicy, lease_expiry, load_lease_policy, parse_utc
from ..runtime.locks import exclusive, machine_lock
from ..runtime.paths import (
    machine_path,
    machine_project_paths,
    machine_registration_path,
    machine_runtime_paths,
    shared_paths,
)
from ..runtime.project_activation_consumers import read_consumer_progress
from ..runtime.records import utc_now
from ..runtime.responsibility_store import DurableIO
from ..runtime.store import atomic_replace, read_json
from ..runtime.work_budget import diagnostic_increment, diagnostic_observe_ns
from .bindings import ProjectBinding

REGISTRY_VERSION = 1
REGISTRATION_VERSION = 1
RECOVERY_REGISTRATION_VERSION = 2
RECOVERY_REGISTRATION_PROTOCOL = "qexp-local-responsibility-v1"


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


def _is_path_present(path: Path) -> bool:
    """Only a missing path proves absence; preserve inaccessible metadata as present."""
    try:
        path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


class MachineRegistration:
    """Own durable machine registration and project binding mutations."""

    def __init__(
        self,
        root: Path,
        *,
        ensure_layout: Callable[[], None],
        current_instance_id: Callable[[], str],
        recover_removed_consumer: Callable[[RootConfig, str, str, list[ProjectBinding]], None] | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.paths = machine_runtime_paths(self.root)
        self._ensure_layout = ensure_layout
        self._current_instance_id = current_instance_id
        self._recover_removed_consumer = recover_removed_consumer
        self._registry_cache_lock = RLock()
        self._registry_cache_witness: tuple[int, int, int, int, int] | None = None
        self._registry_cache_revision: int | None = None
        self._registry_cache_bindings: tuple[ProjectBinding, ...] | None = None

    @contextmanager
    def registry_guard(self, *, blocking: bool = True) -> Iterator[bool]:
        self._ensure_layout()
        with exclusive(self.paths["registry_lock"], blocking=blocking) as acquired:
            yield acquired

    def load_registry(self) -> tuple[int, list[ProjectBinding]]:
        """Load bindings as a fresh mutable list for existing callers."""
        revision, bindings = self.load_registry_snapshot()
        return revision, list(bindings)

    def load_registry_snapshot(self) -> tuple[int, tuple[ProjectBinding, ...]]:
        """Load an immutable snapshot, reusing its tuple while the file is unchanged."""
        path = self.paths["registry"]
        while True:
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                witness = None
                with self._registry_cache_lock:
                    if (
                        self._registry_cache_witness == witness
                        and self._registry_cache_revision is not None
                        and self._registry_cache_bindings is not None
                    ):
                        return self._registry_cache_revision, self._registry_cache_bindings
                revision, parsed = 0, ()
                with self._registry_cache_lock:
                    self._registry_cache_witness = None
                    self._registry_cache_revision = revision
                    self._registry_cache_bindings = parsed
                return revision, parsed
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError("machine registry must be a regular non-symlink file.")
            witness = (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )
            with self._registry_cache_lock:
                if (
                    self._registry_cache_witness == witness
                    and self._registry_cache_revision is not None
                    and self._registry_cache_bindings is not None
                ):
                    return self._registry_cache_revision, self._registry_cache_bindings
            value = read_json(path)
            registry = value.get("registry")
            if not isinstance(registry, dict) or registry.get("version") != REGISTRY_VERSION:
                raise RuntimeError("machine registry is malformed or unsupported.")
            revision = registry.get("revision")
            bindings = registry.get("bindings")
            if not isinstance(revision, int) or revision < 0 or not isinstance(bindings, list):
                raise RuntimeError("machine registry is malformed.")
            try:
                parsed = tuple(
                    sorted(
                        (ProjectBinding.from_dict(item) for item in bindings),
                        key=lambda item: item.project_id,
                    )
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError("machine registry contains a malformed project binding.") from exc
            try:
                after = path.lstat()
            except FileNotFoundError:
                continue
            after_witness = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            if not stat.S_ISREG(after.st_mode) or after_witness != witness:
                continue
            with self._registry_cache_lock:
                if (
                    self._registry_cache_witness == witness
                    and self._registry_cache_revision is not None
                    and self._registry_cache_bindings is not None
                ):
                    return self._registry_cache_revision, self._registry_cache_bindings
                self._registry_cache_witness = witness
                self._registry_cache_revision = revision
                self._registry_cache_bindings = parsed
            return revision, parsed

    def save_registry_locked(self, revision: int, bindings: list[ProjectBinding]) -> None:
        """Persist a registry revision while the caller retains the registry guard."""
        with self._registry_cache_lock:
            self._registry_cache_witness = None
            self._registry_cache_revision = None
            self._registry_cache_bindings = None
        persisted = atomic_replace(
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
        if persisted is None:
            try:
                persisted = self.paths["registry"].lstat()
            except OSError:
                persisted = None
        with self._registry_cache_lock:
            self._registry_cache_witness = (
                None
                if persisted is None
                else (
                    persisted.st_dev,
                    persisted.st_ino,
                    persisted.st_size,
                    persisted.st_mtime_ns,
                    persisted.st_ctime_ns,
                )
            )
            self._registry_cache_revision = revision
            self._registry_cache_bindings = tuple(sorted(bindings, key=lambda item: item.project_id))

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

    def rollback_pending_locked(self) -> None:
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
            current_revision, _current_bindings = self.load_registry()
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
                    save_machine_record(cfg, record)
            self.save_registry_locked(max(revision, current_revision) + 1, bindings)
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
        runtime_instance_id = self._current_instance_id()
        with self.registry_guard():
            self.rollback_pending_locked()
            revision, bindings = self.load_registry()
            if self._recover_removed_consumer is not None:
                self._recover_removed_consumer(cfg, stable_id, runtime_instance_id, bindings)
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
                        self.save_registry_locked(revision + 1, updated)
                    self.paths["registration_transaction"].unlink(missing_ok=True)
                    return binding, False
                if same_project:
                    bindings = [item for item in bindings if item != same_project[0]]
                if any(item.shared_root == binding.shared_root for item in bindings):
                    raise ValueError(f"project root {binding.shared_root} is already registered.")
                self.save_registry_locked(revision + 1, [*bindings, binding])
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
            self.validate_registration_record(registration, project_id, cfg)
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
        retired_same_owner = False
        if same_owner and current is None and isinstance(registration, dict):
            consumer = read_consumer_progress(
                cfg.shared_root,
                runtime_id=runtime_instance_id,
                project_id=project_id,
                registration_generation=registration["generation"],
            )
            retired_same_owner = bool(
                consumer is not None and consumer["project_activation_consumer"]["state"] == "retired"
            )
        if registration is not None and not same_owner:
            state = self.registration_state(registration)
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
            if same_owner and registration and not retired_same_owner
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
            "created_at": (
                registration.get("created_at", now) if same_owner and registration and not retired_same_owner else now
            ),
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
    def validate_registration_record(record: dict[str, Any], project_id: str, cfg: RootConfig) -> None:
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

    _validate_registration_record = validate_registration_record

    @staticmethod
    def registration_state(record: dict[str, Any]) -> str:
        try:
            expires_at = parse_utc(record["eligibility_expires_at"])
        except (KeyError, TypeError, ValueError):
            return "invalid"
        if record.get("state") == "superseded":
            return "superseded"
        return "eligible" if expires_at > datetime.now(timezone.utc) else "expired"

    _registration_state = registration_state

    def publish_recovery_registration_locked(self, binding: ProjectBinding) -> bool:
        """Publish under migration exclusion, registry and registration ownership."""
        if binding not in self.load_registry()[1]:
            return False
        cfg = binding.root_config()
        record = load_machine_record(cfg)
        machine = record.get("machine") if isinstance(record, dict) else None
        if not isinstance(machine, dict) or machine.get("agent_runtime") != "machine":
            return False
        migration_path = machine_project_paths(self.root, binding.project_id)["root"] / "migration.json"
        if _is_path_present(migration_path):
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
            self.validate_registration_record(record, binding.project_id, cfg)
            runtime_instance_id = self._current_instance_id()
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            return {"state": "invalid", "write_eligible": False, "generation": None, "error": str(exc), **protocol}
        same_generation = record.get("generation") == binding.registration_generation
        same_runtime = record.get("runtime_instance_id") == binding.runtime_instance_id == runtime_instance_id
        same_root = record.get("runtime_root") == str(self.root) and binding.runtime_root in {None, str(self.root)}
        state = self.registration_state(record) if same_generation and same_runtime and same_root else "superseded"
        return {
            "state": state,
            "write_eligible": state == "eligible",
            "generation": record.get("generation"),
            "eligibility_expires_at": record.get("eligibility_expires_at"),
            "runtime_instance_id": record.get("runtime_instance_id"),
            **protocol,
        }

    def refresh_binding_eligibility(
        self,
        binding: ProjectBinding,
        *,
        renewal_horizon_seconds: float = 0.0,
    ) -> bool:
        """Renew a current generation, returning false for stale or replaced bindings."""
        with self.binding_write_guard(binding, renewal_horizon_seconds=renewal_horizon_seconds) as is_eligible:
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
            self.validate_registration_record(record, binding.project_id, cfg)
            if (
                record.get("state") == "superseded"
                or record.get("generation") != binding.registration_generation
                or record.get("runtime_instance_id") != binding.runtime_instance_id
                or binding.runtime_instance_id != self._current_instance_id()
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
    def binding_write_guard(
        self,
        binding: ProjectBinding,
        *,
        renewal_horizon_seconds: float = 0.0,
    ) -> Iterator[bool]:
        """Fence an authoritative write, renewing the current generation when due."""
        if (
            not isinstance(renewal_horizon_seconds, (int, float))
            or isinstance(renewal_horizon_seconds, bool)
            or not math.isfinite(renewal_horizon_seconds)
            or renewal_horizon_seconds < 0
        ):
            raise ValueError("renewal_horizon_seconds must be a finite nonnegative number.")
        cfg = binding.root_config()
        policy = load_lease_policy(cfg)
        with self._registration_guard(cfg):
            raw = load_machine_registration(cfg)
            record = raw.get("registration") if isinstance(raw, dict) else None
            if not isinstance(record, dict):
                yield False
                return
            self.validate_registration_record(record, binding.project_id, cfg)
            if (
                self.registration_state(record) != "eligible"
                or record.get("generation") != binding.registration_generation
                or record.get("runtime_instance_id") != binding.runtime_instance_id
                or binding.runtime_instance_id != self._current_instance_id()
                or record.get("runtime_root") != str(self.root)
                or binding.runtime_root not in {None, str(self.root)}
            ):
                yield False
                return
            previous_expiry = record["eligibility_expires_at"]
            expires_at = parse_utc(previous_expiry)
            next_expiry = lease_expiry(policy)
            parsed_next_expiry = parse_utc(next_expiry)
            renew_at = expires_at - timedelta(seconds=policy.ttl_seconds - _registration_renewal_interval(policy))
            now = datetime.now(timezone.utc)
            horizon = now + timedelta(seconds=renewal_horizon_seconds)
            # Derive scheduling from fenced durable state, never cached authority.
            # Dormant round-robin callers renew early when the current lease
            # would not survive until their next bounded service opportunity.
            if parsed_next_expiry != expires_at and (
                now >= renew_at or expires_at <= horizon or parsed_next_expiry < expires_at
            ):
                record = dict(record)
                record["eligibility_expires_at"] = next_expiry
                record["updated_at"] = utc_now()
                save_machine_registration(cfg, {"registration": record})
                _observe_registration_renewal(previous_expiry, policy, is_reactivation=False)
            yield True

    def binding_write_eligible(
        self,
        binding: ProjectBinding,
        *,
        renew: bool = False,
        renewal_horizon_seconds: float = 0.0,
    ) -> bool:
        """Check current generation authority before an identity-scoped write."""
        if renew:
            try:
                return self.refresh_binding_eligibility(
                    binding,
                    renewal_horizon_seconds=renewal_horizon_seconds,
                )
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                return False
        return bool(self.registration_status(binding).get("write_eligible"))

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
            self.save_registry_locked(revision + 1, [updated if item == current else item for item in bindings])
        return updated
