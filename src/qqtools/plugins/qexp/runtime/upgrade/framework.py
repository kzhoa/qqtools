"""Durable, resumable machine-rolling upgrade coordination for qexp."""

from __future__ import annotations

import hashlib
import uuid
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from ...config_types import RootConfig
from ...runtime.locks import exclusive, schema_lock
from ...runtime.paths import shared_paths
from ...runtime.records import utc_now
from ...runtime.store import atomic_replace, migration_json_io_guard, read_json
from .contracts import (
    DeterministicUpgradeError,
    MigrationPhase,
    MigrationPlugin,
    MigrationSpec,
    PhaseResult,
    TransientUpgradeError,
    UpgradeContext,
    UpgradeError,
    UpgradeSliceBudget,
    UpgradeStorage,
)
from .production import UpgradeJournalMigration

UPGRADE_JOURNAL_VERSION = 1
UPGRADE_STATES = frozenset(
    {
        "idle",
        "runnable",
        "waiting",
        "paused",
        "pause_pending",
        "repair_required",
        "ready_to_resume",
        "completed",
    }
)


class MigrationRegistry:
    """Process-local registry of protocol-specific migration plugins."""

    def __init__(self, plugins: Iterable[MigrationPlugin] = ()) -> None:
        self._plugins: dict[str, MigrationPlugin] = {}
        for plugin in plugins:
            if not isinstance(plugin, MigrationPlugin):
                raise TypeError("migration registry accepts MigrationPlugin instances")
            if plugin.spec.name in self._plugins:
                raise ValueError(f"migration {plugin.spec.name!r} is already registered")
            self._plugins[plugin.spec.name] = plugin
        self._ordered_names = self._compile()

    def register(self, plugin: MigrationPlugin) -> None:
        if not isinstance(plugin, MigrationPlugin):
            raise TypeError("migration registry accepts MigrationPlugin instances")
        name = plugin.spec.name
        if name in self._plugins:
            raise ValueError(f"migration {name!r} is already registered")
        self._plugins[name] = plugin
        try:
            self._ordered_names = self._compile()
        except Exception:
            del self._plugins[name]
            raise

    def get(self, name: str) -> MigrationPlugin:
        try:
            return self._plugins[name]
        except KeyError as exc:
            raise DeterministicUpgradeError(f"migration plugin {name!r} is not installed") from exc

    def values(self) -> tuple[MigrationPlugin, ...]:
        return tuple(self._plugins[name] for name in self._ordered_names)

    def _compile(self) -> tuple[str, ...]:
        """Reject invalid dependency declarations and precompute a stable execution order."""
        names = set(self._plugins)
        for name, plugin in self._plugins.items():
            spec = plugin.spec
            if name in spec.dependencies or name in spec.conflicts:
                raise ValueError(f"migration {name!r} cannot depend on or conflict with itself")
            unknown = (set(spec.dependencies) | set(spec.conflicts)) - names
            if unknown:
                raise ValueError(f"migration {name!r} references unregistered migrations: {sorted(unknown)}")
        # A conflict is serialized by waiting until the named migration is completed, so it is an
        # ordering edge too.  Compiling both declarations prevents a runtime-only stalemate.
        remaining = {
            name: set(plugin.spec.dependencies) | set(plugin.spec.conflicts)
            for name, plugin in self._plugins.items()
        }
        ordered: list[str] = []
        while remaining:
            ready = sorted(name for name, dependencies in remaining.items() if not dependencies)
            if not ready:
                raise ValueError("migration dependencies contain a cycle")
            ordered.extend(ready)
            for name in ready:
                del remaining[name]
            completed = set(ready)
            for dependencies in remaining.values():
                dependencies.difference_update(completed)
        return tuple(ordered)


DEFAULT_MIGRATIONS = MigrationRegistry([UpgradeJournalMigration()])


def register_migration(plugin: MigrationPlugin) -> None:
    """Register a production migration during qexp runtime setup."""
    DEFAULT_MIGRATIONS.register(plugin)


def _is_applicable(plugin: MigrationPlugin, cfg: RootConfig) -> bool:
    spec = plugin.spec
    budget = UpgradeSliceBudget(
        spec.max_records_per_slice,
        spec.max_metadata_ops_per_slice,
        spec.max_io_bytes_per_slice,
    )
    with migration_json_io_guard():
        return plugin.is_applicable_with_storage(cfg, UpgradeStorage(budget))


def _journal_path(cfg: RootConfig) -> Path:
    return shared_paths(cfg.shared_root)["upgrade_journal"]


def _pause_intent_path(cfg: RootConfig) -> Path:
    return shared_paths(cfg.shared_root)["upgrade_pause_intent"]


def _load_pause_intent(cfg: RootConfig) -> dict[str, Any] | None:
    path = _pause_intent_path(cfg)
    if not path.exists():
        return None
    value = read_json(path).get("pause_intent")
    if not isinstance(value, dict) or value.get("state") != "requested":
        return None
    return value


def _save_pause_intent(cfg: RootConfig, *, reason: str) -> dict[str, Any]:
    intent = {"state": "requested", "requested_at": utc_now(), "reason": reason}
    atomic_replace(_pause_intent_path(cfg), {"pause_intent": intent})
    return intent


def _clear_pause_intent(cfg: RootConfig) -> None:
    atomic_replace(_pause_intent_path(cfg), {"pause_intent": {"state": None}})


def _upgrade_lock_path(cfg: RootConfig) -> Path:
    return shared_paths(cfg.shared_root)["locks"] / "upgrade.lock"


def _current_protocol(cfg: RootConfig) -> str:
    path = shared_paths(cfg.shared_root)["schema"] / "version.json"
    value = read_json(path).get("schema", {})
    protocol = value.get("protocol")
    if isinstance(protocol, str) and protocol:
        return protocol
    version = value.get("version")
    if type(version) is int:
        return f"schema-{version}"
    raise DeterministicUpgradeError("qexp schema protocol is missing or malformed")


def _timestamp_after(seconds: float) -> str:
    return (
        (datetime.now(timezone.utc) + timedelta(seconds=max(0.0, seconds)))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _probe_after(spec: MigrationSpec, project_id: str) -> str:
    """Return a bounded, deterministic jittered deadline for a waiting migration."""
    digest = hashlib.sha256(f"{project_id}:{spec.name}".encode()).digest()
    jitter = (int.from_bytes(digest[:2]) / 65535.0) * min(1.0, spec.max_probe_interval_seconds * 0.1)
    return _timestamp_after(spec.max_probe_interval_seconds + jitter)


def _is_due(value: object) -> bool:
    if not isinstance(value, str):
        return True
    try:
        return datetime.now(timezone.utc) >= datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return True


def _empty_journal(cfg: RootConfig) -> dict[str, Any]:
    return {
        "upgrade": {
            "version": UPGRADE_JOURNAL_VERSION,
            "project_id": _project_id(cfg),
            "shared_root": str(cfg.shared_root),
            "source_protocol": _current_protocol(cfg),
            "migrations": {},
            "revision": 0,
            "pause": {"state": None, "requested_at": None, "reason": None},
            "repair": None,
            "updated_at": utc_now(),
        }
    }


def _project_id(cfg: RootConfig) -> str:
    path = shared_paths(cfg.shared_root)["project"] / "identity.json"
    try:
        value = read_json(path).get("project", {}).get("project_id")
    except (OSError, ValueError):
        value = None
    if not isinstance(value, str) or not value:
        raise DeterministicUpgradeError(f"qexp project identity is unavailable: {path}")
    return value


def _load_journal(cfg: RootConfig) -> dict[str, Any] | None:
    path = _journal_path(cfg)
    if not path.exists():
        return None
    value = read_json(path)
    upgrade = value.get("upgrade")
    if (
        not isinstance(upgrade, dict)
        or upgrade.get("version") != UPGRADE_JOURNAL_VERSION
        or not isinstance(upgrade.get("migrations"), dict)
        or not isinstance(upgrade.get("pause"), dict)
    ):
        raise DeterministicUpgradeError(f"qexp upgrade journal is malformed: {path}")
    if upgrade.get("project_id") != _project_id(cfg):
        raise DeterministicUpgradeError("qexp upgrade journal targets a different project")
    return value


def _save_journal(cfg: RootConfig, journal: dict[str, Any]) -> None:
    journal["upgrade"]["revision"] = int(journal["upgrade"].get("revision", 0)) + 1
    journal["upgrade"]["updated_at"] = utc_now()
    atomic_replace(_journal_path(cfg), journal)


def _status_from_journal(cfg: RootConfig, journal: dict[str, Any] | None) -> dict[str, Any]:
    protocol = _current_protocol(cfg)
    if journal is None:
        return {
            "project_id": _project_id(cfg),
            "shared_root": str(cfg.shared_root),
            "state": "idle",
            "phase": None,
            "source_protocol": protocol,
            "target_protocol": None,
            "migrations": [],
            "pending": False,
            "can_run": False,
            "admission_blocked": False,
            "migration_blocked": False,
            "last_progress_at": None,
            "blockers": [],
            "pause": None,
            "repair": None,
        }
    upgrade = journal["upgrade"]
    migrations = list(upgrade["migrations"].values())
    active = next(
        (item for item in migrations if item.get("state") not in {"completed", "idle"}),
        migrations[-1] if migrations else None,
    )
    if active is None:
        state = "completed" if migrations else "idle"
        phase = None
    else:
        state = active.get("state", "repair_required")
        phase = active.get("phase")
    blockers = [
        item.get("error") or item.get("blocker")
        for item in migrations
        if isinstance(item.get("error") or item.get("blocker"), str)
    ]
    pause = _load_pause_intent(cfg) or upgrade.get("pause")
    if pause and pause.get("state") == "requested" and state not in {"completed", "repair_required"}:
        state = "pause_pending" if active and active.get("in_flight") else "paused"
    return {
        "project_id": upgrade["project_id"],
        "shared_root": upgrade["shared_root"],
        "state": state,
        "phase": phase,
        "source_protocol": upgrade.get("source_protocol", protocol),
        "target_protocol": active.get("target_protocol") if active else None,
        "migrations": migrations,
        "pending": bool(migrations and any(item.get("state") != "completed" for item in migrations)),
        "can_run": bool(
            active
            and (
                active.get("state") == "runnable"
                or (
                    active.get("state") == "waiting"
                    and _is_due(active.get("next_retry_at") or active.get("next_probe_at"))
                )
            )
        ),
        "next_probe_at": active.get("next_retry_at") or active.get("next_probe_at") if active else None,
        "admission_blocked": bool(active and active.get("admission_blocked")),
        "migration_blocked": state in {"repair_required", "paused", "pause_pending"},
        "last_progress_at": upgrade.get("updated_at"),
        "blockers": blockers,
        "pause": pause,
        "repair": upgrade.get("repair"),
    }


class UpgradeCoordinator:
    """Advance one project's declared migrations in bounded, crash-resumable slices."""

    def __init__(
        self,
        cfg: RootConfig,
        *,
        registry: MigrationRegistry | None = None,
        holder_id: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.registry = registry or DEFAULT_MIGRATIONS
        self.holder_id = holder_id or uuid.uuid4().hex

    def status(self) -> dict[str, Any]:
        try:
            journal = _load_journal(self.cfg)
            status = _status_from_journal(self.cfg, journal)
            if journal is not None:
                drifted = self._completed_migrations_with_drift(journal)
                if drifted:
                    status.update(
                        {
                            "state": "repair_required",
                            "pending": True,
                            "can_run": False,
                            "migration_blocked": True,
                            "blockers": status["blockers"]
                            + [f"completed_migration_drift:{name}" for name in drifted],
                        }
                    )
            return status
        except (OSError, KeyError, TypeError, ValueError, UpgradeError) as exc:
            return {
                "project_id": _safe_project_id(self.cfg),
                "shared_root": str(self.cfg.shared_root),
                "state": "repair_required",
                "phase": None,
                "source_protocol": None,
                "target_protocol": None,
                "migrations": [],
                "pending": True,
                "can_run": False,
                "admission_blocked": True,
                "migration_blocked": True,
                "last_progress_at": None,
                "blockers": [f"journal_unreadable:{exc}"],
                "pause": None,
                "repair": None,
            }

    def discover(self) -> dict[str, Any]:
        """Create a journal only when a registered plugin applies to this root."""
        current = _load_journal(self.cfg)
        if current is not None:
            upgrade = current["upgrade"]
            drifted = self._completed_migrations_with_drift(current)
            known_names = set(upgrade["migrations"])
            additions = [
                plugin
                for plugin in self.registry.values()
                if plugin.spec.name not in known_names and _is_applicable(plugin, self.cfg)
            ]
            if additions or drifted:
                with exclusive(_upgrade_lock_path(self.cfg)) as acquired:
                    if acquired:
                        current = _load_journal(self.cfg) or current
                        upgrade = current["upgrade"]
                        current_names = set(upgrade["migrations"])
                        for name in self._completed_migrations_with_drift(current):
                            item = upgrade["migrations"][name]
                            item.update(
                                {
                                    "state": "repair_required",
                                    "error": "completed migration invariant no longer holds",
                                    "admission_blocked": not self.registry.get(name).spec.safe_legacy_path,
                                }
                            )
                        for plugin in additions:
                            if plugin.spec.name in current_names:
                                continue
                            upgrade["migrations"][plugin.spec.name] = _new_migration_state(plugin.spec)
                        _save_journal(self.cfg, current)
                    else:
                        return _status_from_journal(self.cfg, _load_journal(self.cfg))
            return _status_from_journal(self.cfg, current)
        protocol = _current_protocol(self.cfg)
        applicable: list[MigrationPlugin] = []
        for plugin in self.registry.values():
            spec = plugin.spec
            if protocol not in spec.compatible_readers and protocol != spec.source_protocol:
                continue
            if _is_applicable(plugin, self.cfg):
                applicable.append(plugin)
        if not applicable:
            return _status_from_journal(self.cfg, None)
        journal = _empty_journal(self.cfg)
        for plugin in applicable:
            spec = plugin.spec
            journal["upgrade"]["migrations"][spec.name] = _new_migration_state(spec)
        with exclusive(_upgrade_lock_path(self.cfg)) as acquired:
            if acquired:
                _save_journal(self.cfg, journal)
        return _status_from_journal(self.cfg, _load_journal(self.cfg))

    def advance(self, *, force_retry: bool = False) -> dict[str, Any]:
        """Run at most one bounded phase slice and return durable project status."""
        try:
            journal = self._discover_for_advance()
        except (OSError, KeyError, TypeError, ValueError, UpgradeError) as exc:
            return self._error_status(exc)
        if journal is None:
            return _status_from_journal(self.cfg, None)
        with exclusive(_upgrade_lock_path(self.cfg), blocking=False) as has_upgrade_lock:
            if not has_upgrade_lock:
                return _status_from_journal(self.cfg, journal)
            journal = _load_journal(self.cfg)
            if journal is None:
                return _status_from_journal(self.cfg, None)
            try:
                return self._advance_locked(journal, force_retry=force_retry)
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                active = next(
                    (item for item in journal["upgrade"]["migrations"].values() if item.get("state") != "completed"),
                    None,
                )
                if active is None:
                    raise
                return self._record_failure(journal, active, str(exc), is_transient=False, safe_old_path=False)

    def request_pause(self, reason: str) -> dict[str, Any]:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("pause reason must be a non-empty string")
        journal = self._discover_for_advance()
        if journal is None:
            raise ValueError("no migration is pending for this project")
        _save_pause_intent(self.cfg, reason=reason)
        with exclusive(_upgrade_lock_path(self.cfg), blocking=False) as acquired:
            if not acquired:
                return self.status() | {"state": "pause_pending", "blockers": ["coordinator_busy"]}
            journal = _load_journal(self.cfg)
            if journal is None:
                raise ValueError("no migration is pending for this project")
            upgrade = journal["upgrade"]
            pause = upgrade["pause"]
            pause.update(_load_pause_intent(self.cfg) or {"state": "requested", "requested_at": utc_now(), "reason": reason})
            for item in upgrade["migrations"].values():
                if item.get("state") not in {"completed", "repair_required"}:
                    item["state"] = "pause_pending" if item.get("in_flight") else "paused"
            _save_journal(self.cfg, journal)
        return _status_from_journal(self.cfg, journal)

    def inspect_repair(self, target: str) -> dict[str, Any]:
        journal, plugin, item = self._repair_context(target)
        context = self._context(journal, item)
        with migration_json_io_guard():
            plan = plugin.plan_repair(context, target)
        if not isinstance(plan, dict) or not plan.get("snapshot_ready"):
            raise DeterministicUpgradeError(
                "repair plan must include a durable snapshot_ready=true proof before mutation"
            )
        repair = {
            "repair_id": uuid.uuid4().hex,
            "target": target,
            "operator": self.holder_id,
            "created_at": utc_now(),
            "state": "planned",
            "plan": plan,
            "migration_repair_generation": int(item.get("repair_generation", 0)),
        }
        repair["plan"]["repair_id"] = repair["repair_id"]
        repair["plan"]["journal_revision"] = int(journal["upgrade"].get("revision", 0)) + 1
        with exclusive(_upgrade_lock_path(self.cfg)) as acquired:
            if not acquired:
                raise TransientUpgradeError("repair coordinator is busy")
            current = _load_journal(self.cfg)
            if current is None or current["upgrade"].get("revision") != repair["plan"]["journal_revision"] - 1:
                raise ValueError("repair plan became stale while it was being prepared")
            current["upgrade"]["repair"] = repair
            _save_journal(self.cfg, current)
        return repair

    def apply_repair(self, repair_id: str) -> dict[str, Any]:
        with exclusive(_upgrade_lock_path(self.cfg), blocking=False) as acquired:
            if not acquired:
                raise TransientUpgradeError("repair coordinator is busy")
            journal, plugin, item = self._repair_context()
            repair = journal["upgrade"].get("repair")
            if not isinstance(repair, dict) or repair.get("repair_id") != repair_id or repair.get("state") != "planned":
                raise ValueError("repair plan is missing, stale, or already applied")
            with schema_lock(self.cfg.shared_root, blocking=False) as schema_acquired:
                if not schema_acquired:
                    raise TransientUpgradeError("repair requires the project schema lock")
                if repair["plan"].get("journal_revision") != journal["upgrade"].get("revision"):
                    raise ValueError("repair plan is stale; create a new plan from current revisions")
                context = self._context(journal, item)
                snapshot_path = shared_paths(self.cfg.shared_root)["upgrade_repairs"] / f"{repair_id}.snapshot.json"
                with migration_json_io_guard():
                    affected_records = plugin.repair_snapshot(context)
                context.storage.atomic_replace(
                    snapshot_path,
                    {
                        "snapshot": {
                            "repair_id": repair_id,
                            "created_at": utc_now(),
                            "journal": deepcopy(journal),
                            "affected_records": affected_records,
                        }
                    },
                )
                repair["snapshot_path"] = str(snapshot_path)
                with migration_json_io_guard():
                    is_writer_safe = plugin.writer_safe(context)
                    is_activation_ready = plugin.activation_ready(context)
                if not is_writer_safe or not is_activation_ready:
                    raise TransientUpgradeError("repair safety predicates are not currently satisfied")
                try:
                    with migration_json_io_guard():
                        result = plugin.apply_repair(context, repair["plan"])
                except (OSError, RuntimeError, ValueError, UpgradeError) as exc:
                    repair["state"] = "failed"
                    repair["error"] = str(exc)
                    item["state"] = "repair_required"
                    _save_journal(self.cfg, journal)
                    raise
                repair["state"] = "applied"
                repair["applied_at"] = utc_now()
                repair["result"] = result
                _save_journal(self.cfg, journal)
        return repair

    def validate_repair(self, repair_id: str) -> dict[str, Any]:
        with exclusive(_upgrade_lock_path(self.cfg), blocking=False) as acquired:
            if not acquired:
                raise TransientUpgradeError("repair coordinator is busy")
            journal, plugin, item = self._repair_context()
            repair = journal["upgrade"].get("repair")
            if not isinstance(repair, dict) or repair.get("repair_id") != repair_id or repair.get("state") != "applied":
                raise ValueError("repair must be applied before validation")
            if repair.get("migration_repair_generation") != int(item.get("repair_generation", 0)):
                raise ValueError("repair generation changed; create a new repair plan")
            with schema_lock(self.cfg.shared_root, blocking=False) as schema_acquired:
                if not schema_acquired:
                    raise TransientUpgradeError("repair validation requires the project schema lock")
                context = self._context(journal, item)
                with migration_json_io_guard():
                    result = plugin.validate_repair(context, repair["plan"])
            if result.state == "complete":
                repair["state"] = "validated"
                item["state"] = "ready_to_resume"
            else:
                repair["state"] = "validation_failed"
                item["state"] = "repair_required"
                item["error"] = result.blocker or "repair validation did not complete"
            repair["validated_at"] = utc_now()
            repair["validation"] = result.detail
            _save_journal(self.cfg, journal)
            return repair

    def resume(self) -> dict[str, Any]:
        with exclusive(_upgrade_lock_path(self.cfg), blocking=False) as acquired:
            if not acquired:
                raise TransientUpgradeError("upgrade coordinator is busy")
            journal = _load_journal(self.cfg)
            if journal is None:
                raise ValueError("no migration is pending for this project")
            repair = journal["upgrade"].get("repair")
            if not isinstance(repair, dict) or repair.get("state") != "validated":
                raise ValueError("migration can resume only after explicit repair validation")
            upgrade = journal["upgrade"]
            item = next(
                (value for value in upgrade["migrations"].values() if value.get("state") == "ready_to_resume"),
                None,
            )
            if item is None:
                raise ValueError("repair validation did not leave a migration ready to resume")
            if repair.get("migration_repair_generation") != int(item.get("repair_generation", 0)):
                raise ValueError("validated repair generation is stale")
            with schema_lock(self.cfg.shared_root, blocking=False) as schema_acquired:
                if not schema_acquired:
                    raise TransientUpgradeError("resume requires the project schema lock")
                plugin = self.registry.get(item["name"])
                context = self._context(journal, item)
                with migration_json_io_guard():
                    validation = plugin.validate_repair(context, repair["plan"])
                    is_activation_ready = plugin.activation_ready(context)
                    is_writer_safe = plugin.writer_safe(context)
                if validation.state != "complete":
                    raise DeterministicUpgradeError(
                        validation.blocker or "repair terminal invariant no longer holds"
                    )
                if not is_activation_ready or not is_writer_safe:
                    raise TransientUpgradeError("repair safety predicates are not currently satisfied")
            upgrade["pause"] = {"state": None, "requested_at": None, "reason": None}
            _clear_pause_intent(self.cfg)
            for item in upgrade["migrations"].values():
                if item.get("state") == "ready_to_resume":
                    item.update(
                        {
                            "state": "runnable",
                            "error": None,
                            "blocker": None,
                            "repair_generation": item.get("repair_generation", 0) + 1,
                        }
                    )
            upgrade["repair"] = None
            _save_journal(self.cfg, journal)
            return _status_from_journal(self.cfg, journal)

    def _discover_for_advance(self) -> dict[str, Any] | None:
        journal = _load_journal(self.cfg)
        if journal is not None:
            self.discover()
            return _load_journal(self.cfg)
        status = self.discover()
        return _load_journal(self.cfg) if status.get("pending") else None

    def _advance_locked(self, journal: dict[str, Any], *, force_retry: bool) -> dict[str, Any]:
        upgrade = journal["upgrade"]
        pause = upgrade["pause"]
        active = self._next_active(upgrade)
        if active is None:
            return _status_from_journal(self.cfg, journal)
        if _load_pause_intent(self.cfg) is not None:
            pause.update(_load_pause_intent(self.cfg) or {})
        if pause.get("state") == "requested":
            if active.get("in_flight"):
                active["state"] = "pause_pending"
            else:
                active["state"] = "paused"
            _save_journal(self.cfg, journal)
            return _status_from_journal(self.cfg, journal)
        if active.get("state") in {"paused", "pause_pending", "repair_required", "ready_to_resume"}:
            return _status_from_journal(self.cfg, journal)
        if not force_retry and not _is_due(active.get("next_retry_at")):
            return _status_from_journal(self.cfg, journal)
        plugin = self.registry.get(active["name"])
        spec = plugin.spec
        if not self._prerequisites_satisfied(upgrade, spec):
            active.update({"state": "waiting", "blocker": "migration_prerequisite"})
            _save_journal(self.cfg, journal)
            return _status_from_journal(self.cfg, journal)
        if not self._conflicts_clear(upgrade, spec):
            active.update({"state": "waiting", "blocker": "migration_conflict"})
            _save_journal(self.cfg, journal)
            return _status_from_journal(self.cfg, journal)
        phase = active.get("phase")
        if phase not in spec.phases:
            raise DeterministicUpgradeError(f"migration {spec.name!r} has an invalid journal phase {phase!r}")
        if phase == "activation" and not active.get("audit_passed"):
            raise DeterministicUpgradeError("activation reached without a successful audit")
        active.update(
            {
                "in_flight": True,
                "state": "runnable",
                "holder_id": self.holder_id,
                "fence_token": int(active.get("fence_token", 0)) + 1,
            }
        )
        _save_journal(self.cfg, journal)
        try:
            if phase == "activation":
                with schema_lock(self.cfg.shared_root, blocking=False) as has_schema_lock:
                    if not has_schema_lock:
                        active["in_flight"] = False
                        active["state"] = "waiting"
                        return self._save_waiting(journal, "schema_lock_busy")
                    context = self._context(journal, active)
                    with migration_json_io_guard():
                        is_activation_ready = plugin.activation_ready(context)
                        is_writer_safe = plugin.writer_safe(context)
                    if not is_activation_ready:
                        result = PhaseResult(
                            "waiting",
                            blocker="activation_predicate",
                            detail={"next_probe_at": _probe_after(spec, upgrade["project_id"])},
                        )
                    elif not is_writer_safe:
                        result = PhaseResult(
                            "waiting", blocker="writer_safety_unproven", detail={"admission_blocked": True}
                        )
                    else:
                        with migration_json_io_guard():
                            result = plugin.phase(phase, context)
            else:
                context = self._context(journal, active)
                with migration_json_io_guard():
                    result = plugin.phase(phase, context)
            if not isinstance(result, PhaseResult):
                raise DeterministicUpgradeError("migration phase must return PhaseResult")
            self._validate_slice_result(spec, result, context.slice_budget)
        except TransientUpgradeError as exc:
            return self._record_failure(
                journal, active, str(exc), is_transient=True, safe_old_path=spec.safe_legacy_path
            )
        except (DeterministicUpgradeError, UpgradeError, OSError, RuntimeError, ValueError) as exc:
            return self._record_failure(
                journal, active, str(exc), is_transient=False, safe_old_path=spec.safe_legacy_path
            )
        active["in_flight"] = False
        active["work_items"] = active.get("work_items", 0) + result.work_items
        active["cursor"] = result.cursor
        active["detail"] = result.detail
        active["last_slice_usage"] = {
            "records": result.work_items,
            "metadata_ops": result.metadata_ops,
            "io_bytes": result.io_bytes,
        }
        active["blocker"] = result.blocker
        if result.state == "blocked":
            if result.detail.get("retryable") is True:
                active["state"] = "waiting"
                active["next_retry_at"] = result.detail.get("next_probe_at") or _probe_after(
                    spec, upgrade["project_id"]
                )
            else:
                active["state"] = "repair_required"
            active["admission_blocked"] = bool(result.detail.get("admission_blocked"))
        elif result.state == "waiting":
            active["state"] = "waiting"
            active["next_probe_at"] = result.detail.get("next_probe_at") or _probe_after(
                spec, upgrade["project_id"]
            )
        elif result.state == "complete":
            if phase == "audit":
                active["audit_passed"] = True
                active["audit_evidence"] = dict(result.detail)
            index = int(active.get("phase_index", 0)) + 1
            if index >= len(spec.phases):
                active["state"] = "completed"
                active["completed_at"] = utc_now()
            else:
                active.update({"phase_index": index, "phase": spec.phases[index], "state": "runnable"})
        else:
            active["state"] = "runnable"
        _save_journal(self.cfg, journal)
        return _status_from_journal(self.cfg, journal)

    def _record_failure(
        self,
        journal: dict[str, Any],
        item: dict[str, Any],
        error: str,
        *,
        is_transient: bool,
        safe_old_path: bool,
    ) -> dict[str, Any]:
        item["in_flight"] = False
        item["retry_count"] = int(item.get("retry_count", 0)) + 1
        item["error"] = error
        if is_transient and item["retry_count"] <= 3:
            item["state"] = "waiting"
            item["next_retry_at"] = _timestamp_after(min(30.0, 2 ** (item["retry_count"] - 1)))
        else:
            item["state"] = "repair_required"
            item["admission_blocked"] = not safe_old_path
        _save_journal(self.cfg, journal)
        return _status_from_journal(self.cfg, journal)

    def _save_waiting(self, journal: dict[str, Any], blocker: str) -> dict[str, Any]:
        upgrade = journal["upgrade"]
        for item in upgrade["migrations"].values():
            if item.get("state") in {"runnable", "waiting"}:
                spec = self.registry.get(item["name"]).spec
                item.update(
                    {
                        "state": "waiting",
                        "blocker": blocker,
                        "next_probe_at": _probe_after(spec, upgrade["project_id"]),
                    }
                )
        _save_journal(self.cfg, journal)
        return _status_from_journal(self.cfg, journal)

    def _context(self, journal: dict[str, Any], item: dict[str, Any]) -> UpgradeContext:
        slice_budget = UpgradeSliceBudget(
            int(item.get("max_records_per_slice", 1)),
            int(item.get("max_metadata_ops_per_slice", 1)),
            int(item.get("max_io_bytes_per_slice", 1)),
        )
        return UpgradeContext(
            self.cfg,
            journal,
            int(item.get("work_budget", 1)),
            _journal_path(self.cfg),
            slice_budget,
            UpgradeStorage(slice_budget),
        )

    def _validate_slice_result(
        self, spec: MigrationSpec, result: PhaseResult, budget: UpgradeSliceBudget
    ) -> None:
        if result.work_items > spec.max_records_per_slice:
            raise DeterministicUpgradeError("migration phase exceeded its declared record budget")
        if result.metadata_ops > spec.max_metadata_ops_per_slice:
            raise DeterministicUpgradeError("migration phase exceeded its declared metadata budget")
        if result.io_bytes > spec.max_io_bytes_per_slice:
            raise DeterministicUpgradeError("migration phase exceeded its declared I/O budget")
        used = budget.used(
            max_records=spec.max_records_per_slice,
            max_metadata_ops=spec.max_metadata_ops_per_slice,
            max_io_bytes=spec.max_io_bytes_per_slice,
        )
        reported = {"records": result.work_items, "metadata_ops": result.metadata_ops, "io_bytes": result.io_bytes}
        if any(used[name] > reported[name] for name in used):
            raise DeterministicUpgradeError("migration phase under-reported consumed resources")

    def _next_active(self, upgrade: dict[str, Any]) -> dict[str, Any] | None:
        migrations = upgrade["migrations"]
        for plugin in self.registry.values():
            item = migrations.get(plugin.spec.name)
            if item is not None and item.get("state") != "completed" and self._prerequisites_satisfied(upgrade, plugin.spec):
                return item
        return next((item for item in migrations.values() if item.get("state") != "completed"), None)

    def _completed_migrations_with_drift(self, journal: dict[str, Any]) -> list[str]:
        """Return completed migrations whose durable terminal invariant no longer holds."""
        drifted: list[str] = []
        for name, item in journal["upgrade"]["migrations"].items():
            if item.get("state") != "completed":
                continue
            plugin = self.registry.get(name)
            try:
                with migration_json_io_guard():
                    is_valid = plugin.validate_terminal_invariant(self._context(journal, item))
            except (OSError, RuntimeError, ValueError, UpgradeError):
                is_valid = False
            if not is_valid:
                drifted.append(name)
        return drifted

    def _prerequisites_satisfied(self, upgrade: dict[str, Any], spec: MigrationSpec) -> bool:
        migrations = upgrade["migrations"]
        return all(migrations.get(name, {}).get("state") == "completed" for name in spec.dependencies)

    def _conflicts_clear(self, upgrade: dict[str, Any], spec: MigrationSpec) -> bool:
        migrations = upgrade["migrations"]
        return all(migrations.get(name, {}).get("state") in {None, "completed"} for name in spec.conflicts)

    def _repair_context(self, target: str | None = None) -> tuple[dict[str, Any], MigrationPlugin, dict[str, Any]]:
        journal = _load_journal(self.cfg)
        if journal is None:
            raise ValueError("no migration is pending for this project")
        migrations = journal["upgrade"]["migrations"]
        item = next(
            (
                value
                for value in migrations.values()
                if value.get("state") in {"paused", "repair_required", "ready_to_resume"}
            ),
            None,
        )
        if item is None:
            raise ValueError("migration must be paused or repair-required before repair")
        if target is not None and target != item["name"] and target != item.get("phase"):
            # The target is intentionally explicit: either the migration name or its current phase.
            raise ValueError(f"repair target {target!r} does not match the paused migration")
        return journal, self.registry.get(item["name"]), item

    def _error_status(self, exc: Exception) -> dict[str, Any]:
        status = self.status()
        status["state"] = "repair_required"
        status["migration_blocked"] = True
        status["admission_blocked"] = True
        status["blockers"] = [str(exc)]
        return status


def _new_migration_state(spec: MigrationSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "source_protocol": spec.source_protocol,
        "target_protocol": spec.target_protocol,
        "phase": spec.phases[0],
        "phase_index": 0,
        "state": "runnable",
        "work_budget": spec.work_budget,
        "max_probe_interval_seconds": spec.max_probe_interval_seconds,
        "max_records_per_slice": spec.max_records_per_slice,
        "max_metadata_ops_per_slice": spec.max_metadata_ops_per_slice,
        "max_io_bytes_per_slice": spec.max_io_bytes_per_slice,
        "in_flight": False,
        "cursor": None,
        "work_items": 0,
        "retry_count": 0,
        "next_retry_at": None,
        "next_probe_at": None,
        "audit_passed": False,
        "admission_blocked": False,
        "created_at": utc_now(),
        "completed_at": None,
    }


def _safe_project_id(cfg: RootConfig) -> str | None:
    try:
        return _project_id(cfg)
    except (OSError, UpgradeError):
        return None


def project_upgrade_status(cfg: RootConfig, *, registry: MigrationRegistry | None = None) -> dict[str, Any]:
    """Return bounded project upgrade status without scanning historical work records."""
    return UpgradeCoordinator(cfg, registry=registry).status()


def upgrade_journal_path(cfg: RootConfig) -> Path:
    """Return the shared durable journal path for a project upgrade."""
    return _journal_path(cfg)


def load_upgrade_journal(cfg: RootConfig) -> dict[str, Any] | None:
    """Load the validated project journal without scanning historical work records."""
    return _load_journal(cfg)


__all__ = [
    "DEFAULT_MIGRATIONS",
    "MigrationRegistry",
    "UpgradeCoordinator",
    "project_upgrade_status",
    "upgrade_journal_path",
    "load_upgrade_journal",
    "register_migration",
]
