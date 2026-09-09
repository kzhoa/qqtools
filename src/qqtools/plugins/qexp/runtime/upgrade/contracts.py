"""Stable authoring contracts for online qexp protocol migrations.

The coordinator deliberately knows only phase ordering, persistence and retry policy.  A
migration plugin owns the meaning of its records and must prove that its write path is safe with
the source writers it declares.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ...config_types import RootConfig
from ...runtime.store import atomic_replace, authorized_migration_json_io, json_encoded_size, read_json_limited

MigrationPhase = Literal[
    "expansion",
    "readiness",
    "backfill",
    "audit",
    "activation",
    "contraction",
]
PhaseState = Literal["progressed", "waiting", "complete", "blocked"]

PHASES: tuple[MigrationPhase, ...] = (
    "expansion",
    "readiness",
    "backfill",
    "audit",
    "activation",
    "contraction",
)


class UpgradeError(RuntimeError):
    """Base class for migration failures that are safe to persist in the journal."""


class TransientUpgradeError(UpgradeError):
    """A bounded retry may make progress without operator repair."""


class DeterministicUpgradeError(UpgradeError):
    """A repeat will not make progress until an operator repairs or changes state."""


@dataclass(frozen=True, slots=True)
class MigrationSpec:
    """Declarative safety and scheduling contract for one persisted protocol migration."""

    name: str
    source_protocol: str
    target_protocol: str
    compatible_readers: tuple[str, ...]
    compatible_writers: tuple[str, ...]
    phases: tuple[MigrationPhase, ...] = PHASES
    work_budget: int = 64
    max_probe_interval_seconds: float = 30.0
    max_records_per_slice: int = 64
    max_metadata_ops_per_slice: int = 32
    max_io_bytes_per_slice: int = 256 * 1024
    interruption_budget_seconds: float = 1.0
    dependencies: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    cleanup_version: str | None = None
    normal_level: Literal["L0", "L1", "L2"] = "L1"
    recovery_level: Literal["L0", "L1", "L2"] = "L2"
    safe_legacy_path: bool = True

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name or "\\" in self.name:
            raise ValueError("migration name must be a non-empty path-safe value")
        if not self.source_protocol or not self.target_protocol:
            raise ValueError("migration source_protocol and target_protocol are required")
        if not self.compatible_readers or not self.compatible_writers:
            raise ValueError("migration must declare compatible readers and writers")
        if not self.phases or len(set(self.phases)) != len(self.phases):
            raise ValueError("migration phases must be non-empty and unique")
        if any(phase not in PHASES for phase in self.phases):
            raise ValueError("migration contains an unsupported phase")
        if type(self.work_budget) is not int or self.work_budget <= 0:
            raise ValueError("migration work_budget must be a positive integer")
        if self.max_probe_interval_seconds <= 0:
            raise ValueError("migration max_probe_interval_seconds must be positive")
        if type(self.max_records_per_slice) is not int or self.max_records_per_slice <= 0:
            raise ValueError("migration max_records_per_slice must be a positive integer")
        if type(self.max_metadata_ops_per_slice) is not int or self.max_metadata_ops_per_slice <= 0:
            raise ValueError("migration max_metadata_ops_per_slice must be a positive integer")
        if type(self.max_io_bytes_per_slice) is not int or self.max_io_bytes_per_slice <= 0:
            raise ValueError("migration max_io_bytes_per_slice must be a positive integer")
        if self.interruption_budget_seconds <= 0:
            raise ValueError("migration interruption budget must be positive")
        if self.normal_level not in {"L0", "L1", "L2"}:
            raise ValueError("migration normal_level must be L0, L1, or L2")
        if self.recovery_level not in {"L0", "L1", "L2"}:
            raise ValueError("L3 workload drain is not a supported qexp recovery level")
        if self.normal_level == "L2" and self.recovery_level != "L2":
            raise ValueError("migration recovery level cannot be less disruptive than normal level")
        if self.normal_level == "L1" and self.recovery_level == "L0":
            raise ValueError("migration recovery level cannot be less disruptive than normal level")
        if type(self.safe_legacy_path) is not bool:
            raise ValueError("migration safe_legacy_path must be a bool")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_protocol": self.source_protocol,
            "target_protocol": self.target_protocol,
            "compatible_readers": list(self.compatible_readers),
            "compatible_writers": list(self.compatible_writers),
            "phases": list(self.phases),
            "work_budget": self.work_budget,
            "max_probe_interval_seconds": self.max_probe_interval_seconds,
            "max_records_per_slice": self.max_records_per_slice,
            "max_metadata_ops_per_slice": self.max_metadata_ops_per_slice,
            "max_io_bytes_per_slice": self.max_io_bytes_per_slice,
            "interruption_budget_seconds": self.interruption_budget_seconds,
            "dependencies": list(self.dependencies),
            "conflicts": list(self.conflicts),
            "cleanup_version": self.cleanup_version,
            "normal_level": self.normal_level,
            "recovery_level": self.recovery_level,
            "safe_legacy_path": self.safe_legacy_path,
        }


@dataclass(frozen=True, slots=True)
class PhaseResult:
    """Result returned by a plugin after one bounded phase slice."""

    state: PhaseState
    work_items: int = 0
    metadata_ops: int = 0
    io_bytes: int = 0
    cursor: str | None = None
    blocker: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.state not in {"progressed", "waiting", "complete", "blocked"}:
            raise ValueError("phase result has an unsupported state")
        if self.work_items < 0:
            raise ValueError("phase work_items cannot be negative")
        if self.metadata_ops < 0 or self.io_bytes < 0:
            raise ValueError("phase resource usage cannot be negative")
        if self.state == "blocked" and not self.blocker:
            raise ValueError("blocked phase results require a blocker")


@dataclass(slots=True)
class UpgradeSliceBudget:
    """Consumable limits supplied to a migration phase."""

    records_remaining: int
    metadata_ops_remaining: int
    io_bytes_remaining: int

    def consume_records(self, count: int = 1) -> None:
        self.records_remaining = self._consume(self.records_remaining, count, "records")

    def consume_metadata_ops(self, count: int = 1) -> None:
        self.metadata_ops_remaining = self._consume(self.metadata_ops_remaining, count, "metadata operations")

    def consume_io_bytes(self, count: int) -> None:
        self.io_bytes_remaining = self._consume(self.io_bytes_remaining, count, "I/O bytes")

    @staticmethod
    def _consume(remaining: int, count: int, name: str) -> int:
        if type(count) is not int or count < 0:
            raise ValueError(f"{name} consumption must be a non-negative integer")
        if count > remaining:
            raise DeterministicUpgradeError(f"migration slice exceeded its {name} budget")
        return remaining - count

    def used(self, *, max_records: int, max_metadata_ops: int, max_io_bytes: int) -> dict[str, int]:
        return {
            "records": max_records - self.records_remaining,
            "metadata_ops": max_metadata_ops - self.metadata_ops_remaining,
            "io_bytes": max_io_bytes - self.io_bytes_remaining,
        }


class UpgradeStorage:
    """Budget-enforcing storage facade for migration-owned JSON I/O."""

    def __init__(self, budget: UpgradeSliceBudget) -> None:
        self._budget = budget

    def read_json(self, path: Path) -> dict[str, Any]:
        self._budget.consume_metadata_ops()
        size = path.stat().st_size
        self._budget.consume_io_bytes(size)
        with authorized_migration_json_io():
            return read_json_limited(path, max_bytes=size)

    def exists(self, path: Path) -> bool:
        self._budget.consume_metadata_ops()
        return path.exists()

    def atomic_replace(self, path: Path, value: dict[str, Any]) -> None:
        size = json_encoded_size(value)
        self._budget.consume_metadata_ops()
        self._budget.consume_io_bytes(size)
        with authorized_migration_json_io():
            atomic_replace(path, value)


@dataclass(frozen=True, slots=True)
class UpgradeContext:
    """Context passed to a plugin for one bounded, project-scoped operation."""

    cfg: RootConfig
    journal: dict[str, Any]
    work_budget: int
    journal_path: Path
    slice_budget: UpgradeSliceBudget
    storage: UpgradeStorage


class MigrationPlugin:
    """Base class for protocol-specific migrations.

    Subclasses should override ``is_applicable`` and only the phases they need.  Default phase
    methods are complete so an additive capability can declare a small, explicit phase set.
    """

    spec: MigrationSpec

    def is_applicable(self, cfg: RootConfig) -> bool:
        """Return whether this source protocol is present and needs this migration."""
        del cfg
        return False

    def is_applicable_with_storage(self, cfg: RootConfig, storage: UpgradeStorage) -> bool:
        """Run applicability through budgeted storage when the check performs I/O."""
        del storage
        return self.is_applicable(cfg)

    def terminal_invariant_holds(self, cfg: RootConfig) -> bool:
        """Return whether a completed migration's durable target state still holds.

        Migrations without a distinct target-state marker may override this check. By default,
        a completed migration must no longer report itself applicable.
        """
        return not self.is_applicable(cfg)

    def validate_terminal_invariant(self, context: UpgradeContext) -> bool:
        """Check completed state through the budgeted storage boundary.

        Plugins whose terminal check performs I/O must override this context-aware method and
        use ``context.storage``. The compatibility fallback is suitable only for pure checks.
        """
        return self.terminal_invariant_holds(context.cfg)

    def phase(self, phase: MigrationPhase, context: UpgradeContext) -> PhaseResult:
        """Run one phase slice using the protocol-specific implementation."""
        handler = getattr(self, phase)
        return handler(context)

    def expansion(self, context: UpgradeContext) -> PhaseResult:
        del context
        return PhaseResult("complete")

    def readiness(self, context: UpgradeContext) -> PhaseResult:
        del context
        return PhaseResult("complete")

    def backfill(self, context: UpgradeContext) -> PhaseResult:
        del context
        return PhaseResult("complete")

    def audit(self, context: UpgradeContext) -> PhaseResult:
        del context
        return PhaseResult("complete")

    def activation(self, context: UpgradeContext) -> PhaseResult:
        del context
        return PhaseResult("complete")

    def contraction(self, context: UpgradeContext) -> PhaseResult:
        del context
        return PhaseResult("complete")

    def activation_ready(self, context: UpgradeContext) -> bool:
        """Return whether the migration's declared activation predicate currently holds."""
        del context
        return True

    def writer_safe(self, context: UpgradeContext) -> bool:
        """Return whether every relevant writer honors the activation boundary."""
        del context
        return True

    def plan_repair(self, context: UpgradeContext, target: str) -> dict[str, Any]:
        """Return a protocol-specific repair plan, or reject unsupported mutation."""
        del context, target
        raise DeterministicUpgradeError("this migration does not declare a repair handler")

    def apply_repair(self, context: UpgradeContext, plan: dict[str, Any]) -> dict[str, Any]:
        """Apply a previously reviewed plan idempotently."""
        del context, plan
        raise DeterministicUpgradeError("this migration does not declare a repair handler")

    def validate_repair(self, context: UpgradeContext, plan: dict[str, Any]) -> PhaseResult:
        """Validate repaired records before the operator may resume migration."""
        del context, plan
        raise DeterministicUpgradeError("this migration does not declare a repair handler")

    def repair_snapshot(self, context: UpgradeContext) -> dict[str, Any]:
        """Return the authoritative records a repair can mutate."""
        del context
        return {}


# These names make the authoring contract discoverable without coupling plugin authors to the
# coordinator's implementation class names.
MigrationDefinition = MigrationSpec
UpgradeDefinition = MigrationSpec
