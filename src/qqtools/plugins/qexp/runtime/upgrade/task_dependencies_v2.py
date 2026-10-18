"""Machine-rolling activation of Project-wide Task dependencies."""

from __future__ import annotations

import os
import stat
from datetime import datetime, timezone
from typing import Any

from qqtools.version import __version__

from ...config_types import RootConfig
from ...lease import parse_utc
from ...runtime.dependency_projection import (
    ProjectionUnavailable,
    advance_rebuild,
    inspect_projection,
    read_projection_state,
    start_rebuild,
)
from ...runtime.locks import dependency_lock
from ...runtime.paths import shared_paths
from ...runtime.protocol_compatibility import (
    SUPPORTED_REQUIRED_CAPABILITIES,
    TASK_DEPENDENCIES_CAPABILITY,
    TASK_DEPENDENCIES_V2_CAPABILITY,
)
from ...runtime.records import utc_now
from ...runtime.store import atomic_replace, read_json
from .contracts import DeterministicUpgradeError, MigrationPlugin, MigrationSpec, PhaseResult, UpgradeContext

TASK_DEPENDENCIES_V2_TARGET_PROTOCOL = "metadata:task-dependencies-v2"
TASK_DEPENDENCIES_V2_WRITER_FLOOR = "1.3.27"
_MAX_RECORD_BYTES = 64 * 1024


def _version_tuple(value: object) -> tuple[int, ...] | None:
    if not isinstance(value, str):
        return None
    pieces = value.split(".")
    if not pieces or any(not piece.isdecimal() for piece in pieces):
        return None
    return tuple(int(piece) for piece in pieces)


def _meets_floor(value: object) -> bool:
    candidate = _version_tuple(value)
    floor = _version_tuple(TASK_DEPENDENCIES_V2_WRITER_FLOOR)
    return candidate is not None and floor is not None and candidate >= floor


def _schema_path(cfg: RootConfig):
    return shared_paths(cfg.shared_root)["schema"] / "version.json"


def _schema_value(cfg: RootConfig, context: UpgradeContext | None = None) -> dict[str, Any]:
    if context is None:
        return read_json(_schema_path(cfg))
    return context.storage.read_json_limited(_schema_path(cfg), max_bytes=_MAX_RECORD_BYTES)


def _capabilities(schema: dict[str, Any]) -> list[str]:
    value = schema.get("schema")
    if not isinstance(value, dict):
        raise DeterministicUpgradeError("qexp schema/version.json is malformed")
    capabilities = value.get("required_capabilities")
    if not isinstance(capabilities, list) or not all(isinstance(item, str) for item in capabilities):
        raise DeterministicUpgradeError("qexp schema/version.json has malformed required capabilities")
    unknown = sorted(set(capabilities) - SUPPORTED_REQUIRED_CAPABILITIES)
    if unknown:
        raise DeterministicUpgradeError(f"qexp root requires unsupported capabilities: {', '.join(unknown)}")
    return capabilities


def _registration_floor_diagnostic(cfg: RootConfig, *, storage: Any | None = None) -> str | None:
    if not _meets_floor(__version__):
        return f"installed qqtools {__version__!r} is below {TASK_DEPENDENCIES_V2_WRITER_FLOOR}"
    machines = shared_paths(cfg.shared_root)["machines"]
    try:
        metadata = machines.lstat()
    except FileNotFoundError:
        return "Project machine registrations are unavailable"
    if not stat.S_ISDIR(metadata.st_mode):
        return "Project machine registrations are not a real directory"
    with os.scandir(machines) as entries:
        for entry in entries:
            if not entry.is_dir(follow_symlinks=False):
                continue
            path = machines / entry.name / "registration.json"
            try:
                record = (
                    storage.read_json_limited(path, max_bytes=_MAX_RECORD_BYTES)
                    if storage is not None
                    else read_json(path)
                )
                registration = record.get("registration")
                if not isinstance(registration, dict):
                    continue
                if registration.get("state") == "superseded":
                    continue
                if registration.get("state") != "eligible":
                    return f"registration for {entry.name!r} is not eligible"
                expires_at = parse_utc(registration.get("eligibility_expires_at"))
                if expires_at <= datetime.now(timezone.utc):
                    continue
                if not _meets_floor(registration.get("client_version")):
                    return f"eligible machine {entry.name!r} has not reported {TASK_DEPENDENCIES_V2_WRITER_FLOOR}"
            except FileNotFoundError:
                continue
            except (OSError, KeyError, TypeError, ValueError) as exc:
                return f"registration for {entry.name!r} is unavailable: {exc}"
    return None


def _phase_result(context: UpgradeContext, state: str, *, blocker: str | None = None, detail=None) -> PhaseResult:
    usage = context.slice_budget.used(
        max_records=TaskDependenciesV2Migration.spec.max_records_per_slice,
        max_metadata_ops=TaskDependenciesV2Migration.spec.max_metadata_ops_per_slice,
        max_io_bytes=TaskDependenciesV2Migration.spec.max_io_bytes_per_slice,
    )
    return PhaseResult(
        state,
        work_items=usage["records"],
        metadata_ops=usage["metadata_ops"],
        io_bytes=usage["io_bytes"],
        blocker=blocker,
        detail=detail or {},
    )


class TaskDependenciesV2Migration(MigrationPlugin):
    """Build/audit reverse cleanup coverage, then install the v2 capability."""

    spec = MigrationSpec(
        name="task-dependencies-v2",
        source_protocol="schema-6",
        target_protocol=TASK_DEPENDENCIES_V2_TARGET_PROTOCOL,
        compatible_readers=("schema-6", TASK_DEPENDENCIES_V2_TARGET_PROTOCOL),
        compatible_writers=("schema-6", TASK_DEPENDENCIES_V2_TARGET_PROTOCOL),
        phases=("readiness", "backfill", "audit", "activation", "contraction"),
        work_budget=1,
        max_records_per_slice=1,
        max_metadata_ops_per_slice=256,
        max_io_bytes_per_slice=256 * 1024,
        interruption_budget_seconds=1.0,
        dependencies=("group-service-v1",),
        cleanup_version=TASK_DEPENDENCIES_V2_TARGET_PROTOCOL,
        normal_level="L1",
        recovery_level="L1",
        safe_legacy_path=True,
    )

    def is_applicable(self, cfg: RootConfig) -> bool:
        try:
            capabilities = _capabilities(_schema_value(cfg))
        except (OSError, KeyError, TypeError, ValueError, DeterministicUpgradeError):
            return True
        return TASK_DEPENDENCIES_CAPABILITY in capabilities and TASK_DEPENDENCIES_V2_CAPABILITY not in capabilities

    def is_applicable_with_storage(self, cfg: RootConfig, storage) -> bool:
        try:
            capabilities = _capabilities(storage.read_json_limited(_schema_path(cfg), max_bytes=_MAX_RECORD_BYTES))
        except (OSError, KeyError, TypeError, ValueError, DeterministicUpgradeError):
            return True
        return TASK_DEPENDENCIES_CAPABILITY in capabilities and TASK_DEPENDENCIES_V2_CAPABILITY not in capabilities

    def is_applicable_for_status(self, cfg: RootConfig, storage) -> bool:
        return self.is_applicable_with_storage(cfg, storage)

    def terminal_invariant_holds(self, cfg: RootConfig) -> bool:
        try:
            return (
                TASK_DEPENDENCIES_V2_CAPABILITY in _capabilities(_schema_value(cfg))
                and inspect_projection(cfg)["state"] == "active"
            )
        except (OSError, KeyError, TypeError, ValueError, ProjectionUnavailable):
            return False

    def validate_terminal_invariant(self, context: UpgradeContext) -> bool:
        try:
            capabilities = _capabilities(_schema_value(context.cfg, context))
            state = read_projection_state(context.cfg, storage=context.storage)
        except (OSError, KeyError, TypeError, ValueError, ProjectionUnavailable, DeterministicUpgradeError):
            return False
        return TASK_DEPENDENCIES_V2_CAPABILITY in capabilities and state["state"] == "active"

    def readiness(self, context: UpgradeContext) -> PhaseResult:
        diagnostic = _registration_floor_diagnostic(context.cfg, storage=context.storage)
        if diagnostic is not None:
            return _phase_result(context, "waiting", blocker="writer_floor", detail={"diagnostic": diagnostic})
        context.slice_budget.consume_records()
        return _phase_result(context, "complete", detail={"writer_floor": TASK_DEPENDENCIES_V2_WRITER_FLOOR})

    def writer_safe(self, context: UpgradeContext) -> bool:
        return _registration_floor_diagnostic(context.cfg, storage=context.storage) is None

    def backfill(self, context: UpgradeContext) -> PhaseResult:
        diagnostic = _registration_floor_diagnostic(context.cfg, storage=context.storage)
        if diagnostic is not None:
            return _phase_result(context, "waiting", blocker="writer_floor", detail={"diagnostic": diagnostic})
        with dependency_lock(context.cfg.shared_root):
            result = advance_rebuild(context.cfg, storage=context.storage, max_items=1)
        context.slice_budget.consume_records()
        if result["state"] == "complete":
            return _phase_result(context, "complete", detail={"projection": result})
        return _phase_result(context, "progressed", detail={"projection": result})

    def audit(self, context: UpgradeContext) -> PhaseResult:
        try:
            state = read_projection_state(context.cfg, storage=context.storage)
        except FileNotFoundError:
            state = None
        except (OSError, ProjectionUnavailable) as exc:
            raise DeterministicUpgradeError(f"task dependency projection audit cannot read state: {exc}") from exc
        if state is None or state["state"] != "active":
            with dependency_lock(context.cfg.shared_root):
                result = advance_rebuild(context.cfg, storage=context.storage, max_items=1)
            context.slice_budget.consume_records()
            if result["state"] != "complete":
                return _phase_result(context, "progressed", detail={"projection": result})
            state = read_projection_state(context.cfg, storage=context.storage)
        else:
            context.slice_budget.consume_records()
        return _phase_result(context, "complete", detail={"projection_revision": state["revision"]})

    def activation_ready(self, context: UpgradeContext) -> bool:
        # Activation owns the schema fence and repairs a projection invalidated
        # after audit.  Admit every valid recoverable state so the phase can
        # make bounded progress instead of waiting forever outside the plugin.
        return inspect_projection(context.cfg, storage=context.storage)["state"] in {
            "missing",
            "active",
            "building",
            "degraded",
        }

    def activation(self, context: UpgradeContext) -> PhaseResult:
        # The coordinator already holds schema exclusion here.  Keep the
        # dependency fence through the final rebuild check and capability write
        # so a legacy mutation cannot invalidate coverage in between them.
        with dependency_lock(context.cfg.shared_root):
            state = inspect_projection(context.cfg, storage=context.storage)["state"]
            if state != "active":
                result = advance_rebuild(context.cfg, storage=context.storage, max_items=1)
                context.slice_budget.consume_records()
                if result["state"] != "complete":
                    return _phase_result(context, "progressed", detail={"projection": result})
            schema = _schema_value(context.cfg, context)
            capabilities = _capabilities(schema)
            if TASK_DEPENDENCIES_V2_CAPABILITY not in capabilities:
                schema["schema"]["required_capabilities"] = [*capabilities, TASK_DEPENDENCIES_V2_CAPABILITY]
                context.storage.atomic_replace(_schema_path(context.cfg), schema)
            if state == "active":
                context.slice_budget.consume_records()
        return _phase_result(context, "complete", detail={"capability": TASK_DEPENDENCIES_V2_CAPABILITY})

    def contraction(self, context: UpgradeContext) -> PhaseResult:
        context.slice_budget.consume_records()
        return _phase_result(context, "complete", detail={"compatibility": "QQTOOLS-COMPAT-0023"})

    def plan_repair(self, context: UpgradeContext, target: str) -> dict[str, Any]:
        if target not in {self.spec.name, "backfill", "audit", "activation"}:
            raise DeterministicUpgradeError(f"unsupported Task dependency v2 repair target: {target!r}")
        return {
            "target": target,
            "projection": inspect_projection(context.cfg, storage=context.storage),
            "evidence": ["canonical Task truth", "identity-bound projection state"],
            "intended_changes": ["resume or restart bounded reverse projection rebuild"],
            "snapshot_ready": True,
        }

    def apply_repair(self, context: UpgradeContext, plan: dict[str, Any]) -> dict[str, Any]:
        if not plan.get("snapshot_ready"):
            raise DeterministicUpgradeError("Task dependency repair evidence is incomplete")
        with dependency_lock(context.cfg.shared_root):
            state = start_rebuild(context.cfg, storage=context.storage)
        return {"projection": state, "compatibility": "QQTOOLS-COMPAT-0023"}

    def validate_repair(self, context: UpgradeContext, plan: dict[str, Any]) -> PhaseResult:
        try:
            state = read_projection_state(context.cfg, storage=context.storage)
        except (OSError, KeyError, TypeError, ValueError, ProjectionUnavailable) as exc:
            return _phase_result(context, "blocked", blocker=f"projection_repair_invalid:{exc}")
        if state["state"] != "building":
            return _phase_result(context, "blocked", blocker="projection_repair_not_building")
        context.slice_budget.consume_records()
        return _phase_result(context, "complete", detail={"projection": state})


__all__ = [
    "TASK_DEPENDENCIES_V2_TARGET_PROTOCOL",
    "TASK_DEPENDENCIES_V2_WRITER_FLOOR",
    "TaskDependenciesV2Migration",
]
