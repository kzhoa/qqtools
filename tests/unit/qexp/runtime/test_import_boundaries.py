"""Static dependency checks for the qexp ready projection packages."""

from __future__ import annotations

import ast
from pathlib import Path

READY_ROOT = Path(__file__).parents[4] / "src/qqtools/plugins/qexp/runtime/ready"
READY_IMPLEMENTATIONS = frozenset(
    {
        "group_members",
        "group_members_rebuild",
        "index",
        "primary_candidates",
        "rebuild",
        "routes",
        "state",
        "traversal",
    }
)


def _local_dependencies(path: Path) -> set[str]:
    """Collect ready sibling imports, including imports nested in functions."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    dependencies: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 1:
                if node.module:
                    dependency = node.module.split(".", 1)[0]
                    if dependency in READY_IMPLEMENTATIONS:
                        dependencies.add(dependency)
                else:
                    dependencies.update(alias.name for alias in node.names if alias.name in READY_IMPLEMENTATIONS)
            elif node.level == 0 and node.module:
                prefix = "qqtools.plugins.qexp.runtime.ready."
                if node.module.startswith(prefix):
                    dependency = node.module[len(prefix) :].split(".", 1)[0]
                    if dependency in READY_IMPLEMENTATIONS:
                        dependencies.add(dependency)
        elif isinstance(node, ast.Import):
            prefix = "qqtools.plugins.qexp.runtime.ready."
            for alias in node.names:
                if alias.name.startswith(prefix):
                    dependency = alias.name[len(prefix) :].split(".", 1)[0]
                    if dependency in READY_IMPLEMENTATIONS:
                        dependencies.add(dependency)
    return dependencies


def _dependencies() -> dict[str, set[str]]:
    return {
        path.stem: _local_dependencies(path) for path in READY_ROOT.glob("*.py") if path.stem in READY_IMPLEMENTATIONS
    }


def _assert_acyclic(dependencies: dict[str, set[str]]) -> None:
    """Reject cycles without allowing a delayed-import exception list."""
    pending = {module: set(imports) for module, imports in dependencies.items()}
    while pending:
        leaves = {module for module, imports in pending.items() if not imports & pending.keys()}
        assert leaves, f"ready implementation import cycle: {pending}"
        for module in leaves:
            pending.pop(module)


def test_ready_import_boundaries_cover_top_level_and_function_imports() -> None:
    dependencies = _dependencies()

    # diagnostics is a validation/support module shared by the ready owners;
    # it is deliberately outside the implementation graph checked here.
    assert not dependencies["state"] & READY_IMPLEMENTATIONS - {"state"}
    assert not dependencies["routes"] & {
        "index",
        "traversal",
        "rebuild",
        "group_members",
        "group_members_rebuild",
        "primary_candidates",
    }
    assert not dependencies["traversal"] & {
        "index",
        "rebuild",
        "group_members",
        "group_members_rebuild",
    }
    assert not dependencies["index"] & {"traversal", "rebuild", "group_members_rebuild"}
    assert not dependencies["group_members"] & {
        "index",
        "rebuild",
        "group_members_rebuild",
        "primary_candidates",
    }
    for module in {"state", "routes", "traversal", "index", "group_members", "primary_candidates"}:
        assert not dependencies[module] & {"rebuild", "group_members_rebuild"}
    assert dependencies["group_members_rebuild"] >= {"group_members", "index", "routes"}
    _assert_acyclic(dependencies)


def test_group_member_rebuild_does_not_reach_private_online_helpers() -> None:
    tree = ast.parse(
        (READY_ROOT / "group_members_rebuild.py").read_text(encoding="utf-8"),
        filename=str(READY_ROOT / "group_members_rebuild.py"),
    )

    private_imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module == "group_members"
        for alias in node.names
        if alias.name.startswith("_")
    ]
    assert private_imports == []

    private_attributes = [
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "group_members"
        and node.attr.startswith("_")
    ]
    assert private_attributes == []


def test_primary_candidate_collaboration_names_match_owner_contract() -> None:
    from qqtools.plugins.qexp.runtime.ready import primary_candidates

    for name in (
        "projection_lock",
        "_read_projection_state",
        "can_update_projection_under_lock",
        "_serialize_candidate",
        "remove_candidate_from_all_routes_under_lock",
        "sync_task_candidate_under_lock",
    ):
        assert hasattr(primary_candidates, name)
    for legacy_name in (
        "projection_rebuild_lock",
        "rebuild_record",
        "accepts_updates_under_lock",
        "_candidate_value",
        "remove_candidate_everywhere_under_lock",
        "sync_candidate_under_lock",
    ):
        assert not hasattr(primary_candidates, legacy_name)
