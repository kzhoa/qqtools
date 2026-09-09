"""Online qexp upgrade framework."""

from .contracts import (
    DeterministicUpgradeError,
    MigrationDefinition,
    MigrationPhase,
    MigrationPlugin,
    MigrationSpec,
    PhaseResult,
    TransientUpgradeError,
    UpgradeContext,
    UpgradeDefinition,
    UpgradeError,
    UpgradeSliceBudget,
    UpgradeStorage,
)
from .framework import (
    DEFAULT_MIGRATIONS,
    MigrationRegistry,
    UpgradeCoordinator,
    load_upgrade_journal,
    project_upgrade_status,
    register_migration,
    upgrade_journal_path,
)
from .machine import (
    MachineUpgradeBudget,
    MachineUpgradeWorker,
    advance_registered_upgrades,
    discover_registered_upgrades,
    inspect_registered_upgrades,
)
from .production import UPGRADE_JOURNAL_CAPABILITY, UpgradeJournalMigration

__all__ = [
    "DEFAULT_MIGRATIONS",
    "DeterministicUpgradeError",
    "MigrationPhase",
    "MigrationDefinition",
    "MigrationPlugin",
    "MigrationRegistry",
    "MigrationSpec",
    "PhaseResult",
    "TransientUpgradeError",
    "UpgradeContext",
    "UpgradeDefinition",
    "UpgradeCoordinator",
    "UpgradeError",
    "UpgradeSliceBudget",
    "UpgradeStorage",
    "MachineUpgradeBudget",
    "MachineUpgradeWorker",
    "advance_registered_upgrades",
    "discover_registered_upgrades",
    "inspect_registered_upgrades",
    "project_upgrade_status",
    "load_upgrade_journal",
    "upgrade_journal_path",
    "register_migration",
    "UPGRADE_JOURNAL_CAPABILITY",
    "UpgradeJournalMigration",
]
