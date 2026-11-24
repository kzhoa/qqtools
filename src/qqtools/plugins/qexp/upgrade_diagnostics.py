"""Read-only projection of repair evidence from a Project upgrade journal."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .runtime.upgrade.contracts import PHASES
from .runtime.upgrade.framework import UPGRADE_JOURNAL_VERSION, UPGRADE_STATES


def summarize_upgrade_journal(value: dict[str, Any], *, project_id: str, shared_root: Path) -> dict[str, object]:
    """Validate journal evidence and report the first repair in recorded order."""
    summary: dict[str, object] = {
        "state": "unknown",
        "migration": None,
        "phase": None,
        "admission_blocked": None,
        "reason": "journal_malformed",
    }
    upgrade = value.get("upgrade")
    if not isinstance(upgrade, dict):
        return summary
    version = upgrade.get("version")
    if type(version) is not int or version != UPGRADE_JOURNAL_VERSION:
        return {**summary, "reason": "journal_version_unsupported"}
    root = upgrade.get("shared_root")
    if upgrade.get("project_id") != project_id or not isinstance(root, str) or Path(root) != shared_root:
        return {**summary, "reason": "journal_identity_mismatch"}
    migrations = upgrade.get("migrations")
    if not isinstance(migrations, dict) or not isinstance(upgrade.get("pause"), dict):
        return summary

    repair = None
    repair_count = 0
    for name, migration in migrations.items():
        if (
            not isinstance(name, str)
            or not name
            or "/" in name
            or "\\" in name
            or not isinstance(migration, dict)
            or migration.get("name") != name
        ):
            return summary
        state = migration.get("state")
        if not isinstance(state, str) or state not in UPGRADE_STATES or migration.get("phase") not in PHASES:
            return summary
        if state == "repair_required":
            if repair is None:
                repair = migration
            repair_count += 1

    if repair is None:
        return {**summary, "state": "no_repair_recorded", "reason": None}
    admission_blocked = repair.get("admission_blocked")
    summary = {
        "state": "repair_required",
        "migration": repair["name"],
        "phase": repair["phase"],
        "admission_blocked": admission_blocked if isinstance(admission_blocked, bool) else None,
        "reason": None,
    }
    if repair_count > 1:
        summary["additional_repairs"] = repair_count - 1
    return summary
