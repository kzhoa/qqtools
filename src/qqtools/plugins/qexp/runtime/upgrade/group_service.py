"""Upgrade-coordinator adapter for Group service activation and recovery."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from ...runtime.group_discovery import locator
from ...runtime.group_discovery.activation import PROTOCOL as GROUP_SERVICE_PROTOCOL
from ...runtime.group_discovery.activation import (
    WRITER_FLOOR,
    advance_group_service_activation_locked,
    group_service_registration_diagnostic,
    is_group_service_active,
    prepare_group_service_activation,
    read_group_service_activation_record,
    transition_group_service_state_locked,
    writer_floor_satisfied,
)
from ...runtime.group_namespace import group_directory, is_group_authority_isolated
from .contracts import DeterministicUpgradeError, MigrationPlugin, MigrationSpec, PhaseResult, UpgradeContext

GROUP_SERVICE_TARGET_PROTOCOL = "metadata:group-service-v1"


def _has_group(root, *, storage=None) -> bool:
    """Inspect at most the identity entry and one canonical Group record."""

    directory = group_directory(root, storage=storage)
    with os.scandir(directory) as entries:
        for _ in range(2):
            try:
                entry = next(entries)
            except StopIteration:
                return False
            if entry.name == ".authority-identity":
                continue
            if entry.name.endswith(".json") and entry.is_file(follow_symlinks=False):
                return True
            raise RuntimeError(f"canonical Group namespace contains an invalid entry: {entry.name}")
    return False


def _next_probe_at() -> str:
    value = datetime.now(timezone.utc) + timedelta(seconds=5)
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _phase_result(
    context: UpgradeContext,
    state: str,
    *,
    blocker: str | None = None,
    cursor: str | None = None,
    detail=None,
) -> PhaseResult:
    spec = GroupServiceMigration.spec
    usage = context.slice_budget.used(
        max_records=spec.max_records_per_slice,
        max_metadata_ops=spec.max_metadata_ops_per_slice,
        max_io_bytes=spec.max_io_bytes_per_slice,
    )
    return PhaseResult(
        state,
        work_items=usage["records"],
        metadata_ops=usage["metadata_ops"],
        io_bytes=usage["io_bytes"],
        cursor=cursor,
        blocker=blocker,
        detail=detail or {},
    )


class GroupServiceMigration(MigrationPlugin):
    """Drive Group service fencing/bootstrap through project and machine upgrades."""

    spec = MigrationSpec(
        name="group-service-v1",
        source_protocol="schema-6",
        target_protocol=GROUP_SERVICE_TARGET_PROTOCOL,
        compatible_readers=("schema-6", GROUP_SERVICE_TARGET_PROTOCOL),
        compatible_writers=("schema-6", GROUP_SERVICE_TARGET_PROTOCOL),
        phases=("expansion", "audit", "activation"),
        work_budget=1,
        max_records_per_slice=1,
        max_metadata_ops_per_slice=128,
        max_io_bytes_per_slice=256 * 1024,
        interruption_budget_seconds=1.0,
        conflicts=("upgrade-journal-v1",),
        cleanup_version=GROUP_SERVICE_TARGET_PROTOCOL,
        normal_level="L1",
        recovery_level="L1",
        safe_legacy_path=True,
    )

    def is_applicable(self, cfg) -> bool:
        if not writer_floor_satisfied():
            return False
        if not is_group_authority_isolated(cfg.shared_root) or not _has_group(cfg.shared_root):
            return False
        try:
            return not is_group_service_active(cfg.shared_root)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            return True

    def is_applicable_with_storage(self, cfg, storage) -> bool:
        if not writer_floor_satisfied():
            return False
        if not is_group_authority_isolated(cfg.shared_root, storage=storage) or not _has_group(
            cfg.shared_root, storage=storage
        ):
            return False
        try:
            return not is_group_service_active(cfg.shared_root, storage=storage)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            return True

    def terminal_invariant_holds(self, cfg) -> bool:
        return is_group_service_active(cfg.shared_root)

    def validate_terminal_invariant(self, context: UpgradeContext) -> bool:
        try:
            return is_group_service_active(context.cfg.shared_root, storage=context.storage)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            return False

    def expansion(self, context: UpgradeContext) -> PhaseResult:
        item = context.journal["upgrade"]["migrations"][self.spec.name]
        raw_cursor = item.get("cursor")
        try:
            cursor = (
                {"phase": "create", "position": 0, "revisions": {}} if raw_cursor is None else json.loads(raw_cursor)
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DeterministicUpgradeError("Group service layout cursor is malformed") from exc
        next_cursor, layout = locator.advance_group_service_layout(
            context.cfg,
            cursor,
            storage=context.storage,
        )
        context.slice_budget.consume_records()
        if next_cursor is not None:
            return _phase_result(
                context,
                "progressed",
                cursor=json.dumps(next_cursor, sort_keys=True, separators=(",", ":")),
            )
        if layout is None:
            raise DeterministicUpgradeError("Group service layout completed without durable evidence")
        prepare_group_service_activation(context.cfg, storage=context.storage, layout=layout)
        return _phase_result(context, "complete")

    def audit(self, context: UpgradeContext) -> PhaseResult:
        record = read_group_service_activation_record(context.cfg.shared_root, storage=context.storage)
        if record is None:
            raise DeterministicUpgradeError("Group service preparing record disappeared during audit")
        locator.read_group_service_layout_marker(context.cfg.shared_root, storage=context.storage)
        diagnostic = group_service_registration_diagnostic(context.cfg.shared_root, storage=context.storage)
        context.slice_budget.consume_records()
        if diagnostic is not None:
            return _phase_result(
                context,
                "waiting",
                blocker="writer_fence",
                detail={
                    "diagnostic_code": diagnostic["code"],
                    "next_probe_at": _next_probe_at(),
                },
            )
        return _phase_result(
            context,
            "complete",
            detail={"activation_revision": record["revision"], "writer_floor": WRITER_FLOOR},
        )

    def activation_ready(self, context: UpgradeContext) -> bool:
        del context
        return True

    def writer_safe(self, context: UpgradeContext) -> bool:
        # The activation state machine verifies every registered writer immediately
        # before it installs the schema fence.
        del context
        return True

    def activation(self, context: UpgradeContext) -> PhaseResult:
        record = read_group_service_activation_record(context.cfg.shared_root, storage=context.storage)
        if record is not None and record["state"] == "degraded":
            item = context.journal["upgrade"]["migrations"][self.spec.name]
            detail = item.get("detail")
            raw_cursor = detail.get("repair_layout_cursor") if isinstance(detail, dict) else None
            if raw_cursor is None:
                raw_cursor = {"phase": "create", "position": 0, "revisions": {}}
            if type(raw_cursor) is not dict:
                raise DeterministicUpgradeError("Group service repair layout cursor is malformed")
            next_cursor, _layout = locator.advance_group_service_layout(
                context.cfg,
                raw_cursor,
                storage=context.storage,
                repair=True,
            )
            if next_cursor is not None:
                context.slice_budget.consume_records()
                return _phase_result(
                    context,
                    "progressed",
                    detail={
                        "activation_state": "degraded",
                        "activation_revision": record["revision"],
                        "repair_layout_cursor": next_cursor,
                    },
                )
        record = advance_group_service_activation_locked(
            context.cfg,
            storage=context.storage,
            layout_prevalidated=True,
        )
        context.slice_budget.consume_records()
        if record["state"] == "preparing" and record.get("diagnostic") is not None:
            return _phase_result(
                context,
                "waiting",
                blocker="writer_fence",
                detail={"diagnostic_code": record["diagnostic"]["code"], "next_probe_at": _next_probe_at()},
            )
        if record["state"] == "active":
            return _phase_result(context, "complete", detail={"activation_revision": record["revision"]})
        return _phase_result(
            context,
            "progressed",
            detail={"activation_state": record["state"], "activation_revision": record["revision"]},
        )

    def plan_repair(self, context: UpgradeContext, target: str) -> dict[str, Any]:
        if target not in {self.spec.name, "activation"}:
            raise DeterministicUpgradeError(f"unsupported Group service repair target: {target!r}")
        record = read_group_service_activation_record(context.cfg.shared_root, storage=context.storage)
        if record is None or record["state"] not in {"active", "degraded"}:
            raise DeterministicUpgradeError("Group service repair requires active or degraded activation evidence")
        return {
            "target": target,
            "activation_state": record["state"],
            "activation_revision": record["revision"],
            "evidence": ["identity-bound activation record", "coordinator owns schema fence during repair"],
            "intended_changes": ["persist active-to-degraded transition", "resume bounded degraded rebuild"],
            "snapshot_ready": True,
        }

    def repair_snapshot(self, context: UpgradeContext) -> dict[str, Any]:
        record = read_group_service_activation_record(context.cfg.shared_root, storage=context.storage)
        if record is None:
            raise DeterministicUpgradeError("Group service activation record is missing")
        return {"activation_record": record}

    def apply_repair(self, context: UpgradeContext, plan: dict[str, Any]) -> dict[str, Any]:
        record = read_group_service_activation_record(context.cfg.shared_root, storage=context.storage)
        if record is None:
            raise DeterministicUpgradeError("Group service activation record is missing")
        if record["revision"] != plan.get("activation_revision") or record["state"] != plan.get("activation_state"):
            raise DeterministicUpgradeError("Group service repair plan is stale")
        if record["state"] == "active":
            record = transition_group_service_state_locked(
                context.cfg,
                "degraded",
                storage=context.storage,
            )
        return {"activation_state": record["state"], "activation_revision": record["revision"]}

    def validate_repair(self, context: UpgradeContext, plan: dict[str, Any]) -> PhaseResult:
        record = read_group_service_activation_record(context.cfg.shared_root, storage=context.storage)
        plan_revision = plan.get("activation_revision")
        was_active = plan.get("activation_state") == "active"
        if (
            record is None
            or record["state"] != "degraded"
            or type(plan_revision) is not int
            or record["revision"] < plan_revision
            or (was_active and record["revision"] <= plan_revision)
        ):
            return _phase_result(
                context,
                "blocked",
                blocker="degraded_transition_not_durable",
                detail={"retryable": False},
            )
        context.slice_budget.consume_records()
        return _phase_result(
            context,
            "complete",
            detail={"activation_state": record["state"], "activation_revision": record["revision"]},
        )


__all__ = ["GROUP_SERVICE_TARGET_PROTOCOL", "GroupServiceMigration"]
