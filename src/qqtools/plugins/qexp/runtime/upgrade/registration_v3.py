"""Bounded conversion and activation of the qexp registration-v3 protocol."""

from __future__ import annotations

import os
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ...config_types import RootConfig
from ...layout import load_root_config
from ...runtime.locks import exclusive
from ...runtime.paths import machine_registration_path, shared_paths
from ...runtime.protocol_compatibility import REGISTRATION_REPLACEMENT_CAPABILITY, SUPPORTED_REQUIRED_CAPABILITIES
from ...runtime.records import SCHEMA_VERSION
from ...runtime.registration_protocol import convert_registration, validate_registration
from ...runtime.store import read_json_limited
from .contracts import (
    DeterministicUpgradeError,
    MigrationPlugin,
    MigrationSpec,
    PhaseResult,
    TransientUpgradeError,
    UpgradeContext,
    UpgradeStorage,
)

REGISTRATION_REPLACEMENT_TARGET_PROTOCOL = "registration-v3"
REGISTRATION_REPLACEMENT_SOURCE_PROTOCOL = "registration-v1-v2"
_MAX_RECORD_BYTES = 64 * 1024
_MAX_MACHINE_ENTRIES = 4096
_COMPATIBILITY_MARKER = "QQTOOLS-COMPAT-0024"


def _schema_path(cfg: RootConfig) -> Path:
    return shared_paths(cfg.shared_root)["schema"] / "version.json"


def _identity_path(cfg: RootConfig) -> Path:
    return shared_paths(cfg.shared_root)["project"] / "identity.json"


def _schema_value(cfg: RootConfig, storage: UpgradeStorage | None = None) -> dict[str, Any]:
    path = _schema_path(cfg)
    if storage is None:
        return read_json_limited(path, max_bytes=_MAX_RECORD_BYTES)
    return storage.read_json_limited(path, max_bytes=_MAX_RECORD_BYTES)


def _project_id(cfg: RootConfig, storage: UpgradeStorage | None = None) -> str:
    path = _identity_path(cfg)
    value = (
        read_json_limited(path, max_bytes=_MAX_RECORD_BYTES)
        if storage is None
        else storage.read_json_limited(path, max_bytes=_MAX_RECORD_BYTES)
    )
    identity = value.get("project")
    if not isinstance(identity, dict):
        raise DeterministicUpgradeError("qexp Project identity is malformed")
    project_id = identity.get("project_id")
    if not isinstance(project_id, str) or not project_id:
        raise DeterministicUpgradeError("qexp Project identity has no stable project ID")
    if identity.get("shared_root") != str(cfg.shared_root):
        raise DeterministicUpgradeError("qexp Project identity does not match the shared root")
    return project_id


def _capabilities(schema: dict[str, Any]) -> list[str]:
    value = schema.get("schema")
    if not isinstance(value, dict) or value.get("version") != SCHEMA_VERSION:
        raise DeterministicUpgradeError("qexp schema/version.json is malformed or unsupported")
    capabilities = value.get("required_capabilities")
    if capabilities is None:
        return []
    if not isinstance(capabilities, list) or not all(isinstance(item, str) and item for item in capabilities):
        raise DeterministicUpgradeError("qexp schema/version.json has malformed required capabilities")
    unknown = sorted(set(capabilities) - SUPPORTED_REQUIRED_CAPABILITIES)
    if unknown:
        raise DeterministicUpgradeError(f"qexp root requires unsupported capabilities: {', '.join(unknown)}")
    return list(capabilities)


def _next_probe_at() -> str:
    value = datetime.now(timezone.utc) + timedelta(seconds=5)
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _phase_cursor(context: UpgradeContext, phase: str) -> str | None:
    item = context.journal["upgrade"]["migrations"].get(RegistrationV3Migration.spec.name)
    if not isinstance(item, dict):
        return None
    cursor = item.get("cursor")
    prefix = f"{phase}:"
    if cursor is None:
        return None
    if not isinstance(cursor, str) or not cursor.startswith(prefix):
        return None
    value = cursor[len(prefix) :]
    if not value:
        raise DeterministicUpgradeError("registration migration cursor is malformed")
    return value


def _phase_result(
    context: UpgradeContext,
    state: str,
    *,
    phase: str,
    cursor: str | None = None,
    blocker: str | None = None,
    detail: dict[str, Any] | None = None,
) -> PhaseResult:
    usage = context.slice_budget.used(
        max_records=RegistrationV3Migration.spec.max_records_per_slice,
        max_metadata_ops=RegistrationV3Migration.spec.max_metadata_ops_per_slice,
        max_io_bytes=RegistrationV3Migration.spec.max_io_bytes_per_slice,
    )
    return PhaseResult(
        state,
        work_items=usage["records"],
        metadata_ops=usage["metadata_ops"],
        io_bytes=usage["io_bytes"],
        cursor=cursor if cursor is None else f"{phase}:{cursor}",
        blocker=blocker,
        detail=detail or {},
    )


def _machine_names(context: UpgradeContext) -> tuple[str, ...]:
    machines = shared_paths(context.cfg.shared_root)["machines"]
    context.storage.account_metadata_ops(1)
    metadata = machines.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise DeterministicUpgradeError("qexp machine registry path is not a real directory")
    context.storage.account_metadata_ops(1)
    names: list[str] = []
    with os.scandir(machines) as entries:
        for entry in entries:
            if len(names) >= _MAX_MACHINE_ENTRIES:
                raise DeterministicUpgradeError(
                    f"registration migration machine entry bound exceeded ({_MAX_MACHINE_ENTRIES})"
                )
            names.append(entry.name)
    return tuple(sorted(names))


def _next_machine_name(names: tuple[str, ...], cursor: str | None) -> str | None:
    return next((name for name in names if cursor is None or name > cursor), None)


def _registration_document(context: UpgradeContext, machine_name: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = machine_registration_path(context.cfg.shared_root, machine_name)
    context.storage.account_metadata_ops(1)
    metadata = path.parent.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise DeterministicUpgradeError(f"invalid qexp machine entry: {machine_name}")
    document = context.storage.read_json_limited(path, max_bytes=_MAX_RECORD_BYTES)
    registration = document.get("registration")
    if not isinstance(registration, dict):
        raise DeterministicUpgradeError(f"registration record for {machine_name!r} is malformed")
    return path, document, registration


def _advance_registration_record(context: UpgradeContext, phase: str, *, audit: bool) -> PhaseResult:
    cursor = _phase_cursor(context, phase)
    try:
        project_id = _project_id(context.cfg, context.storage)
    except OSError as exc:
        return _phase_result(
            context,
            "waiting",
            phase=phase,
            cursor=cursor,
            blocker="registration_storage_unavailable",
            detail={"next_probe_at": _next_probe_at(), "error": str(exc)},
        )

    try:
        with exclusive(
            shared_paths(context.cfg.shared_root)["locks"] / "registrations.lock", blocking=False
        ) as acquired:
            if not acquired:
                return _phase_result(
                    context,
                    "waiting",
                    phase=phase,
                    cursor=cursor,
                    blocker="registration_lock_busy",
                    detail={"next_probe_at": _next_probe_at()},
                )
            try:
                names = _machine_names(context)
            except OSError as exc:
                return _phase_result(
                    context,
                    "waiting",
                    phase=phase,
                    cursor=cursor,
                    blocker="registration_storage_unavailable",
                    detail={"next_probe_at": _next_probe_at(), "error": str(exc)},
                )
            machine_name = _next_machine_name(names, cursor)
            if machine_name is None:
                return _phase_result(context, "complete", phase=phase, detail={"compatibility": _COMPATIBILITY_MARKER})

            try:
                path, document, registration = _registration_document(context, machine_name)
            except FileNotFoundError:
                # A machine directory without a Project registration is a valid inventory entry.
                context.slice_budget.consume_records()
                return _phase_result(
                    context,
                    "progressed",
                    phase=phase,
                    cursor=machine_name,
                    detail={
                        "machine_name": machine_name,
                        "converted": False,
                        "compatibility": _COMPATIBILITY_MARKER,
                    },
                )
            except OSError as exc:
                return _phase_result(
                    context,
                    "waiting",
                    phase=phase,
                    cursor=cursor,
                    blocker="registration_storage_unavailable",
                    detail={"next_probe_at": _next_probe_at(), "error": str(exc)},
                )

            cfg = load_root_config(context.cfg.shared_root, machine_name)
            converted = False
            if audit:
                validate_registration(registration, project_id, cfg)
            else:
                validate_registration(registration, project_id, cfg, allow_source=True)
                target = convert_registration(registration, project_id, cfg)
                updated = dict(document)
                updated["registration"] = target
                # Replay the directory barrier even when a prior rename made
                # target contents visible before its durability failed.
                try:
                    context.storage.atomic_replace(path, updated)
                except OSError as exc:
                    return _phase_result(
                        context,
                        "waiting",
                        phase=phase,
                        cursor=cursor,
                        blocker="registration_storage_unavailable",
                        detail={"next_probe_at": _next_probe_at(), "error": str(exc)},
                    )
                converted = target != registration

    except OSError as exc:
        return _phase_result(
            context,
            "waiting",
            phase=phase,
            cursor=cursor,
            blocker="registration_storage_unavailable",
            detail={"next_probe_at": _next_probe_at(), "error": str(exc)},
        )

    context.slice_budget.consume_records()
    return _phase_result(
        context,
        "progressed",
        phase=phase,
        cursor=machine_name,
        detail={
            "machine_name": machine_name,
            "converted": converted,
            "compatibility": _COMPATIBILITY_MARKER,
        },
    )


class RegistrationV3Migration(MigrationPlugin):
    """Convert source registration records and activate the target writer floor."""

    spec = MigrationSpec(
        name="registration-replacement-v1",
        source_protocol=REGISTRATION_REPLACEMENT_SOURCE_PROTOCOL,
        target_protocol=REGISTRATION_REPLACEMENT_TARGET_PROTOCOL,
        compatible_readers=("schema-6", REGISTRATION_REPLACEMENT_TARGET_PROTOCOL),
        compatible_writers=("schema-6", REGISTRATION_REPLACEMENT_TARGET_PROTOCOL),
        phases=("readiness", "backfill", "audit", "activation", "contraction"),
        work_budget=1,
        max_probe_interval_seconds=5.0,
        max_records_per_slice=1,
        max_metadata_ops_per_slice=32,
        max_io_bytes_per_slice=512 * 1024,
        interruption_budget_seconds=1.0,
        cleanup_version="1.3.30",
        normal_level="L3",
        recovery_level="L3",
        safe_legacy_path=True,
    )

    def is_applicable(self, cfg: RootConfig) -> bool:
        return REGISTRATION_REPLACEMENT_CAPABILITY not in _capabilities(_schema_value(cfg))

    def is_applicable_with_storage(self, cfg: RootConfig, storage: UpgradeStorage) -> bool:
        return REGISTRATION_REPLACEMENT_CAPABILITY not in _capabilities(_schema_value(cfg, storage))

    def is_applicable_for_status(self, cfg: RootConfig, storage: UpgradeStorage) -> bool:
        return self.is_applicable_with_storage(cfg, storage)

    def terminal_invariant_holds(self, cfg: RootConfig) -> bool:
        return REGISTRATION_REPLACEMENT_CAPABILITY in _capabilities(_schema_value(cfg))

    def validate_terminal_invariant(self, context: UpgradeContext) -> bool:
        return REGISTRATION_REPLACEMENT_CAPABILITY in _capabilities(_schema_value(context.cfg, context.storage))

    def readiness(self, context: UpgradeContext) -> PhaseResult:
        try:
            project_id = _project_id(context.cfg, context.storage)
            _capabilities(_schema_value(context.cfg, context.storage))
        except OSError as exc:
            raise TransientUpgradeError(f"registration migration readiness storage unavailable: {exc}") from exc
        return _phase_result(
            context,
            "complete",
            phase="readiness",
            detail={
                "project_id": project_id,
                "deployment_precondition": "owner-coordinated qexp process downtime",
                "old_process_detection": False,
                "compatibility": _COMPATIBILITY_MARKER,
            },
        )

    def backfill(self, context: UpgradeContext) -> PhaseResult:
        return _advance_registration_record(context, "backfill", audit=False)

    def audit(self, context: UpgradeContext) -> PhaseResult:
        return _advance_registration_record(context, "audit", audit=True)

    def activation_ready(self, context: UpgradeContext) -> bool:
        try:
            _project_id(context.cfg, context.storage)
            _capabilities(_schema_value(context.cfg, context.storage))
        except OSError as exc:
            raise TransientUpgradeError(f"registration migration activation storage unavailable: {exc}") from exc
        return True

    def writer_safe(self, context: UpgradeContext) -> bool:
        # QQTOOLS-COMPAT-0024 relies on the owner-approved fleet downtime boundary;
        # target-only local writers make remote process inspection unnecessary.
        del context
        return True

    def activation(self, context: UpgradeContext) -> PhaseResult:
        try:
            with exclusive(
                shared_paths(context.cfg.shared_root)["locks"] / "registrations.lock", blocking=False
            ) as acquired:
                if not acquired:
                    return _phase_result(
                        context,
                        "waiting",
                        phase="activation",
                        blocker="registration_lock_busy",
                        detail={"next_probe_at": _next_probe_at()},
                    )
                project_id = _project_id(context.cfg, context.storage)
                schema = _schema_value(context.cfg, context.storage)
                capabilities = _capabilities(schema)
                if REGISTRATION_REPLACEMENT_CAPABILITY not in capabilities:
                    schema["schema"]["required_capabilities"] = [*capabilities, REGISTRATION_REPLACEMENT_CAPABILITY]
                # Re-publish even when the capability is already present so an
                # interruption after replacement still receives the directory barrier.
                context.storage.atomic_replace(_schema_path(context.cfg), schema)
                detail = {
                    "project_id": project_id,
                    "capability": REGISTRATION_REPLACEMENT_CAPABILITY,
                    "compatibility": _COMPATIBILITY_MARKER,
                }
        except OSError as exc:
            raise TransientUpgradeError(f"registration migration activation storage unavailable: {exc}") from exc
        context.slice_budget.consume_records()
        return _phase_result(context, "complete", phase="activation", detail=detail)

    def contraction(self, context: UpgradeContext) -> PhaseResult:
        context.slice_budget.consume_records()
        return _phase_result(
            context,
            "complete",
            phase="contraction",
            detail={"compatibility": _COMPATIBILITY_MARKER, "evidence": "retained"},
        )


__all__ = [
    "REGISTRATION_REPLACEMENT_SOURCE_PROTOCOL",
    "REGISTRATION_REPLACEMENT_TARGET_PROTOCOL",
    "RegistrationV3Migration",
]
