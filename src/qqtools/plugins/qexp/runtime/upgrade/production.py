"""Production migration for the additive upgrade-journal capability."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ...runtime.paths import shared_paths
from ...runtime.records import utc_now
from ...runtime.store import atomic_replace, read_json
from .contracts import (
    DeterministicUpgradeError,
    MigrationPlugin,
    MigrationSpec,
    PhaseResult,
    UpgradeContext,
    UpgradeStorage,
)

UPGRADE_JOURNAL_CAPABILITY = "upgrade-journal-v1"
# QQTOOLS-COMPAT-0011: keep schema-6 readers untouched during the 1.4.x rolling window.
UPGRADE_JOURNAL_METADATA_PROTOCOL = "metadata:upgrade-journal-v1"
_MANIFEST_NAME = "protocol-manifest.json"


def _schema_path(context: UpgradeContext):
    return shared_paths(context.cfg.shared_root)["schema"] / "version.json"


def _manifest_path(context: UpgradeContext):
    return shared_paths(context.cfg.shared_root)["upgrade"] / _MANIFEST_NAME


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _schema_protocol(schema: dict[str, Any]) -> str:
    value = schema.get("schema", {})
    protocol = value.get("protocol")
    return protocol if isinstance(protocol, str) and protocol else "schema-6"


def _manifest_document(schema: dict[str, Any], *, protocol: str) -> dict[str, Any]:
    return {
        "upgrade_protocol_manifest": {
            "version": 1,
            "capability": UPGRADE_JOURNAL_CAPABILITY,
            "source_protocol": "schema-6",
            "target_protocol": UPGRADE_JOURNAL_METADATA_PROTOCOL,
            "protocol": protocol,
            "schema_digest": _digest(schema),
            "created_at": utc_now(),
        }
    }


def initialize_upgrade_journal_manifest(cfg: Any) -> None:
    """Install the target-state coordinator metadata for newly created roots."""
    path = shared_paths(cfg.shared_root)["upgrade"] / _MANIFEST_NAME
    if not path.exists():
        schema = read_json(shared_paths(cfg.shared_root)["schema"] / "version.json")
        atomic_replace(path, _manifest_document(schema, protocol=UPGRADE_JOURNAL_METADATA_PROTOCOL))


class UpgradeJournalMigration(MigrationPlugin):
    """Materialize and activate the additive journal capability on schema-6 roots."""

    spec = MigrationSpec(
        name="upgrade-journal-v1",
        source_protocol="schema-6",
        target_protocol=UPGRADE_JOURNAL_METADATA_PROTOCOL,
        compatible_readers=("schema-6",),
        compatible_writers=("schema-6",),
        phases=("expansion", "audit", "activation", "contraction"),
        work_budget=1,
        max_records_per_slice=1,
        max_metadata_ops_per_slice=8,
        max_io_bytes_per_slice=64 * 1024,
        interruption_budget_seconds=0.25,
        cleanup_version=UPGRADE_JOURNAL_METADATA_PROTOCOL,
        safe_legacy_path=True,
    )

    def is_applicable(self, cfg) -> bool:
        try:
            manifest = read_json(shared_paths(cfg.shared_root)["upgrade"] / _MANIFEST_NAME)
            schema = read_json(shared_paths(cfg.shared_root)["schema"] / "version.json")
        except (OSError, KeyError, TypeError, ValueError):
            return True
        return self._is_target_state_missing(manifest, schema)

    def _is_target_state_missing(
        self, manifest_document: dict[str, Any], schema: dict[str, Any]
    ) -> bool:
        value = manifest_document.get("upgrade_protocol_manifest", {})
        return not (
            value.get("version") == 1
            and value.get("capability") == UPGRADE_JOURNAL_CAPABILITY
            and value.get("source_protocol") == self.spec.source_protocol
            and value.get("target_protocol") == self.spec.target_protocol
            and value.get("protocol") == self.spec.target_protocol
            and _schema_protocol(schema) == self.spec.source_protocol
            and value.get("schema_digest") == _digest(schema)
        )

    def is_applicable_with_storage(self, cfg, storage: UpgradeStorage) -> bool:
        try:
            manifest = storage.read_json(
                shared_paths(cfg.shared_root)["upgrade"] / _MANIFEST_NAME
            )
            schema = storage.read_json(shared_paths(cfg.shared_root)["schema"] / "version.json")
        except (OSError, KeyError, TypeError, ValueError):
            return True
        return self._is_target_state_missing(manifest, schema)

    def terminal_invariant_holds(self, cfg) -> bool:
        return not self.is_applicable(cfg)

    def validate_terminal_invariant(self, context: UpgradeContext) -> bool:
        try:
            manifest = context.storage.read_json(_manifest_path(context))["upgrade_protocol_manifest"]
            schema = context.storage.read_json(_schema_path(context))
            self._validate_terminal_manifest(schema, manifest)
        except (OSError, KeyError, TypeError, ValueError, DeterministicUpgradeError):
            return False
        return True

    def expansion(self, context: UpgradeContext) -> PhaseResult:
        manifest_path = _manifest_path(context)
        schema = context.storage.read_json(_schema_path(context))
        if not context.storage.exists(manifest_path):
            context.storage.atomic_replace(
                manifest_path, _manifest_document(schema, protocol=_schema_protocol(schema))
            )
        context.slice_budget.consume_records()
        usage = context.slice_budget.used(
            max_records=self.spec.max_records_per_slice,
            max_metadata_ops=self.spec.max_metadata_ops_per_slice,
            max_io_bytes=self.spec.max_io_bytes_per_slice,
        )
        return PhaseResult(
            "complete",
            work_items=1,
            metadata_ops=usage["metadata_ops"],
            io_bytes=usage["io_bytes"],
            detail={"manifest": str(manifest_path)},
        )

    def audit(self, context: UpgradeContext) -> PhaseResult:
        try:
            manifest = context.storage.read_json(_manifest_path(context))["upgrade_protocol_manifest"]
            schema = context.storage.read_json(_schema_path(context))
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise DeterministicUpgradeError(f"upgrade protocol manifest is invalid: {exc}") from exc
        required = {
            "version": 1,
            "capability": UPGRADE_JOURNAL_CAPABILITY,
            "source_protocol": self.spec.source_protocol,
            "target_protocol": self.spec.target_protocol,
        }
        if any(manifest.get(key) != value for key, value in required.items()):
            raise DeterministicUpgradeError("upgrade protocol manifest metadata is invalid")
        protocol = _schema_protocol(schema)
        if manifest.get("protocol") != protocol:
            raise DeterministicUpgradeError("upgrade protocol manifest protocol is stale")
        if protocol != self.spec.source_protocol:
            raise DeterministicUpgradeError("upgrade protocol manifest is outside the migration boundary")
        if manifest.get("schema_digest") != _digest(schema):
            raise DeterministicUpgradeError("upgrade protocol manifest schema digest is stale")
        context.slice_budget.consume_records()
        usage = context.slice_budget.used(
            max_records=self.spec.max_records_per_slice,
            max_metadata_ops=self.spec.max_metadata_ops_per_slice,
            max_io_bytes=self.spec.max_io_bytes_per_slice,
        )
        return PhaseResult(
            "complete",
            work_items=1,
            metadata_ops=usage["metadata_ops"],
            io_bytes=usage["io_bytes"],
            detail={
                "audited": True,
                "schema_digest": _digest(schema),
                "manifest_digest": _digest(manifest),
            },
        )

    def activation(self, context: UpgradeContext) -> PhaseResult:
        manifest = context.storage.read_json(_manifest_path(context))["upgrade_protocol_manifest"]
        schema = context.storage.read_json(_schema_path(context))
        self._validate_manifest(
            schema, manifest, context.journal["upgrade"]["migrations"][self.spec.name]
        )
        if _schema_protocol(schema) != self.spec.source_protocol:
            raise DeterministicUpgradeError("schema protocol is outside the migration boundary")
        context.storage.atomic_replace(
            _manifest_path(context), _manifest_document(schema, protocol=self.spec.target_protocol)
        )
        context.slice_budget.consume_records()
        usage = context.slice_budget.used(
            max_records=self.spec.max_records_per_slice,
            max_metadata_ops=self.spec.max_metadata_ops_per_slice,
            max_io_bytes=self.spec.max_io_bytes_per_slice,
        )
        return PhaseResult(
            "complete",
            work_items=1,
            metadata_ops=usage["metadata_ops"],
            io_bytes=usage["io_bytes"],
            detail={"capability": UPGRADE_JOURNAL_CAPABILITY},
        )

    def _validate_manifest(
        self, schema: dict[str, Any], manifest: dict[str, Any], migration: dict[str, Any]
    ) -> None:
        """Verify the audited schema and manifest still form a safe activation boundary."""
        expected = {
            "version": 1,
            "capability": UPGRADE_JOURNAL_CAPABILITY,
            "source_protocol": self.spec.source_protocol,
            "target_protocol": self.spec.target_protocol,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise DeterministicUpgradeError("upgrade protocol manifest metadata is invalid")
        schema_digest = _digest(schema)
        if manifest.get("schema_digest") != schema_digest:
            raise DeterministicUpgradeError("upgrade protocol manifest schema digest is stale")
        if manifest.get("protocol") != _schema_protocol(schema):
            raise DeterministicUpgradeError("upgrade protocol manifest protocol is stale")
        evidence = migration.get("audit_evidence")
        if not isinstance(evidence, dict) or evidence.get("schema_digest") != schema_digest:
            raise DeterministicUpgradeError("schema version record changed after audit")
        if evidence.get("manifest_digest") != _digest(manifest):
            raise DeterministicUpgradeError("upgrade protocol manifest changed after audit")

    def contraction(self, context: UpgradeContext) -> PhaseResult:
        context.slice_budget.consume_records()
        return PhaseResult("complete", work_items=1, detail={"legacy_path": "retained"})

    def writer_safe(self, context: UpgradeContext) -> bool:
        # Activation only writes coordinator-owned metadata outside records parsed by legacy
        # readers and writers.  It does not activate a new shared schema writer contract.
        del context
        return True

    def plan_repair(self, context: UpgradeContext, target: str) -> dict[str, Any]:
        if target not in {self.spec.name, "expansion", "audit", "activation", "contraction"}:
            raise ValueError(f"unsupported repair target {target!r}")
        schema = context.storage.read_json(_schema_path(context))
        manifest_path = _manifest_path(context)
        manifest = context.storage.read_json(manifest_path) if context.storage.exists(manifest_path) else None
        migration = context.journal["upgrade"]["migrations"][self.spec.name]
        expected_protocol = (
            self.spec.target_protocol if migration.get("completed_at") else self.spec.source_protocol
        )
        return {
            "target": target,
            "observed_revisions": {"journal": int(context.journal["upgrade"].get("revision", 0))},
            "evidence": ["upgrade_protocol_manifest", "schema-version-record"],
            "intended_changes": ["recreate additive upgrade capability marker"],
            "snapshot_ready": True,
            "schema_digest": _digest(schema),
            "manifest_digest": _digest(manifest),
            "manifest_missing": manifest is None,
            "expected_manifest_protocol": expected_protocol,
        }

    def apply_repair(self, context: UpgradeContext, plan: dict[str, Any]) -> dict[str, Any]:
        if not plan.get("snapshot_ready") or not plan.get("evidence"):
            raise DeterministicUpgradeError("repair evidence is incomplete")
        manifest_path = _manifest_path(context)
        schema = context.storage.read_json(_schema_path(context))
        if _digest(schema) != plan.get("schema_digest"):
            raise DeterministicUpgradeError("repair plan is stale: schema version record changed")
        manifest = context.storage.read_json(manifest_path) if context.storage.exists(manifest_path) else None
        if _digest(manifest) != plan.get("manifest_digest"):
            raise DeterministicUpgradeError("repair plan is stale: protocol manifest changed")
        protocol = _schema_protocol(schema)
        if protocol != self.spec.source_protocol:
            raise DeterministicUpgradeError("repair cannot reconcile an unknown schema protocol")
        expected_protocol = plan.get("expected_manifest_protocol")
        if expected_protocol not in {self.spec.source_protocol, self.spec.target_protocol}:
            raise DeterministicUpgradeError("repair plan has an invalid expected manifest protocol")
        repaired = _manifest_document(schema, protocol=expected_protocol)
        context.storage.atomic_replace(manifest_path, repaired)
        repaired_manifest = repaired["upgrade_protocol_manifest"]
        return {
            "applied": True,
            "target": plan["target"],
            "manifest_rebuilt": manifest is None,
            "schema_digest": _digest(schema),
            "manifest_digest": _digest(repaired_manifest),
        }

    def validate_repair(self, context: UpgradeContext, plan: dict[str, Any]) -> PhaseResult:
        manifest_path = _manifest_path(context)
        try:
            manifest = context.storage.read_json(manifest_path)["upgrade_protocol_manifest"]
            schema = context.storage.read_json(_schema_path(context))
        except (OSError, KeyError, TypeError, ValueError) as exc:
            return PhaseResult("blocked", blocker=f"repair_terminal_invariant:{exc}")
        try:
            self._validate_repaired_manifest(
                schema, manifest, expected_protocol=plan.get("expected_manifest_protocol")
            )
        except DeterministicUpgradeError as exc:
            return PhaseResult("blocked", blocker=f"repair_terminal_invariant:{exc}")
        repair = context.journal["upgrade"].get("repair")
        result = repair.get("result") if isinstance(repair, dict) else None
        if (
            plan.get("schema_digest") != _digest(schema)
            or not isinstance(result, dict)
            or result.get("schema_digest") != _digest(schema)
            or result.get("manifest_digest") != _digest(manifest)
        ):
            return PhaseResult("blocked", blocker="repair_validation_digest_mismatch")
        return PhaseResult(
            "complete",
            detail={
                "validated": True,
                "target": plan["target"],
                "schema_digest": _digest(schema),
                "manifest_digest": _digest(manifest),
            },
        )

    def _validate_terminal_manifest(self, schema: dict[str, Any], manifest: dict[str, Any]) -> None:
        self._validate_repaired_manifest(
            schema, manifest, expected_protocol=self.spec.target_protocol
        )

    def _validate_repaired_manifest(
        self, schema: dict[str, Any], manifest: dict[str, Any], *, expected_protocol: object
    ) -> None:
        expected = {
            "version": 1,
            "capability": UPGRADE_JOURNAL_CAPABILITY,
            "source_protocol": self.spec.source_protocol,
            "target_protocol": self.spec.target_protocol,
            "protocol": expected_protocol,
        }
        if expected_protocol not in {self.spec.source_protocol, self.spec.target_protocol}:
            raise DeterministicUpgradeError("repair expected manifest protocol is invalid")
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise DeterministicUpgradeError("upgrade protocol manifest terminal metadata is invalid")
        if _schema_protocol(schema) != self.spec.source_protocol:
            raise DeterministicUpgradeError("source schema protocol changed")
        if manifest.get("schema_digest") != _digest(schema):
            raise DeterministicUpgradeError("upgrade protocol manifest schema digest is stale")

    def repair_snapshot(self, context: UpgradeContext) -> dict[str, Any]:
        manifest_path = _manifest_path(context)
        return {
            "schema_version": context.storage.read_json(_schema_path(context)),
            "protocol_manifest": (
                context.storage.read_json(manifest_path)
                if context.storage.exists(manifest_path)
                else None
            ),
        }


__all__ = [
    "UPGRADE_JOURNAL_CAPABILITY",
    "UPGRADE_JOURNAL_METADATA_PROTOCOL",
    "UpgradeJournalMigration",
    "initialize_upgrade_journal_manifest",
]
