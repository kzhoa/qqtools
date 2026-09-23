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
OLD_PRIMARY_PROBE_FIELDS = {
    "primary_probe_cursors",
    "primary_probe_revisions",
    "primary_probe_complete",
    "primary_probe_pending_routes",
    "primary_probe_recheck_cursors",
    "primary_probe_recheck_round_cursors",
}


def _import_modules(path: Path, source: str | None = None) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8") if source is None else source, filename=str(path))
    modules: set[str] = set()
    pending = [tree]
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        else:
            # Imports can occur in nested statement bodies, never in expressions.
            pending.extend(child for child in ast.iter_child_nodes(node) if not isinstance(child, ast.expr))
    return modules


def test_legacy_runtime_module_files_are_absent() -> None:
    assert not OLD_RUNTIME_MODULES & {path.name for path in RUNTIME_ROOT.glob("*.py")}


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


def test_registration_owner_is_composed_once_and_context_reexports_contract(tmp_path: Path) -> None:
    from qqtools.plugins.qexp.agent.bindings import ProjectBinding as OwnedProjectBinding
    from qqtools.plugins.qexp.agent.context import RECOVERY_REGISTRATION_PROTOCOL as context_recovery_protocol
    from qqtools.plugins.qexp.agent.context import RECOVERY_REGISTRATION_VERSION as context_recovery_version
    from qqtools.plugins.qexp.agent.context import MachineRuntime, ProjectBinding
    from qqtools.plugins.qexp.agent.registration import (
        RECOVERY_REGISTRATION_PROTOCOL,
        RECOVERY_REGISTRATION_VERSION,
        MachineRegistration,
    )

    runtime = MachineRuntime(tmp_path / "machine-runtime")

    assert ProjectBinding is OwnedProjectBinding
    assert context_recovery_version == RECOVERY_REGISTRATION_VERSION
    assert context_recovery_protocol == RECOVERY_REGISTRATION_PROTOCOL
    assert isinstance(runtime.registration, MachineRegistration)


def test_registration_modules_do_not_import_runtime_composition_or_dispatch() -> None:
    agent_root = PROJECT_ROOT / "src/qqtools/plugins/qexp/agent"
    forbidden = {
        "context",
        "dispatch_probe",
        "control_plane",
        "lifecycle",
        "qqtools.plugins.qexp.agent.context",
        "qqtools.plugins.qexp.agent.dispatch_probe",
        "qqtools.plugins.qexp.agent.control_plane",
        "qqtools.plugins.qexp.agent.lifecycle",
    }

    for name in ("bindings.py", "registration.py"):
        imports = _import_modules(agent_root / name)
        assert not imports & forbidden, f"{name} imports runtime composition or dispatch: {sorted(imports & forbidden)}"


def test_primary_probe_state_is_only_mutated_by_its_owner() -> None:
    production_root = PROJECT_ROOT / "src/qqtools/plugins/qexp"
    offenders: list[str] = []
    for path in production_root.rglob("*.py"):
        if path.name == "dispatch_probe.py":
            continue
        source = path.read_text(encoding="utf-8")
        if not any(field in source for field in OLD_PRIMARY_PROBE_FIELDS):
            continue
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in OLD_PRIMARY_PROBE_FIELDS:
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}:{node.attr}")

    assert offenders == []


def test_import_scan_keeps_nested_and_multiline_imports(tmp_path: Path) -> None:
    source = """
import os, sys
from qqtools.plugins.qexp.runtime.recovery import (
    recover,
)
def nested():
    try:
        import qqtools.plugins.qexp.runtime.cpu_lane
    except ImportError:
        from qqtools.plugins.qexp.runtime import reservations
    value = [number * 2 for number in range(100)]
"""
    assert _import_modules(tmp_path / "sample.py", source) == {
        "os",
        "sys",
        "qqtools.plugins.qexp.runtime.recovery",
        "qqtools.plugins.qexp.runtime.cpu_lane",
        "qqtools.plugins.qexp.runtime",
    }
