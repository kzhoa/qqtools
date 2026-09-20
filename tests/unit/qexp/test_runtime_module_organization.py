from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[3]
RUNTIME_ROOT = PROJECT_ROOT / "src/qqtools/plugins/qexp/runtime"
OLD_RUNTIME_MODULES = {
    "active_operations.py",
    "recovery.py",
    "reservations.py",
    "cpu_lane.py",
    "group_members.py",
    "ready.py",
    "availability.py",
}
OLD_RUNTIME_IMPORTS = {
    "runtime.active_operations",
    "runtime.recovery",
    "runtime.reservations",
    "runtime.cpu_lane",
    "runtime.group_members",
}
OLD_PRIMARY_PROBE_FIELDS = {
    "primary_probe_cursors",
    "primary_probe_revisions",
    "primary_probe_complete",
    "primary_probe_pending_routes",
    "primary_probe_recheck_cursors",
    "primary_probe_recheck_round_cursors",
}


def _import_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_legacy_runtime_modules_and_repository_imports_are_absent() -> None:
    assert not OLD_RUNTIME_MODULES & {path.name for path in RUNTIME_ROOT.glob("*.py")}

    imported_modules = set()
    for directory in ("src", "tests", "scripts"):
        for path in (PROJECT_ROOT / directory).rglob("*.py"):
            imported_modules.update(_import_modules(path))
    assert not {
        imported
        for imported in imported_modules
        for old_import in OLD_RUNTIME_IMPORTS
        if imported == old_import or imported.startswith(f"{old_import}.")
    }


def test_runtime_packages_export_owner_objects_without_private_names() -> None:
    import qqtools.plugins.qexp as qexp
    from qqtools.plugins.qexp.runtime import availability, ready
    from qqtools.plugins.qexp.runtime.ready import index, records
    from qqtools.plugins.qexp.runtime.resources import cpu_lane

    assert all(not name.startswith("_") for name in ready.__all__)
    assert all(not name.startswith("_") for name in availability.__all__)
    assert ready.ReadyMarkerRef is records.ReadyMarkerRef
    assert ready.ReadyClassification is index.ReadyClassification
    assert qexp.CpuLanePolicy is cpu_lane.CpuLanePolicy
    assert qexp.get_cpu_lane_policy is cpu_lane.get_cpu_lane_policy
    assert qexp.set_cpu_lane_capacity is cpu_lane.set_cpu_lane_capacity


def test_primary_probe_state_is_only_mutated_by_its_owner() -> None:
    production_root = PROJECT_ROOT / "src/qqtools/plugins/qexp"
    offenders: list[str] = []
    for path in production_root.rglob("*.py"):
        if path.name == "dispatch_probe.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in OLD_PRIMARY_PROBE_FIELDS:
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}:{node.attr}")

    assert offenders == []
