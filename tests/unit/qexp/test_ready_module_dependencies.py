from __future__ import annotations

import ast
from pathlib import Path

READY_ROOT = Path(__file__).parents[3] / "src/qqtools/plugins/qexp/runtime/ready"


def _ready_dependencies() -> dict[str, set[str]]:
    modules = {path.stem for path in READY_ROOT.glob("*.py") if path.name != "__init__.py"}
    dependencies = {module: set() for module in modules}
    for path in READY_ROOT.glob("*.py"):
        if path.name == "__init__.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if not isinstance(node, ast.ImportFrom) or node.level != 1:
                continue
            if node.module:
                target = node.module.split(".", 1)[0]
                if target in modules:
                    dependencies[path.stem].add(target)
                continue
            dependencies[path.stem].update(alias.name for alias in node.names if alias.name in modules)
    return dependencies


def test_ready_implementation_dependencies_are_acyclic_and_layered() -> None:
    dependencies = _ready_dependencies()

    assert not dependencies["index"] & {"traversal", "rebuild", "group_members_rebuild"}
    assert not dependencies["routes"] & {"index", "primary_candidates"}
    assert "index" not in dependencies["traversal"]
    assert not dependencies["group_members"] & {
        "index",
        "primary_candidates",
        "rebuild",
        "group_members_rebuild",
    }
    assert dependencies["group_members_rebuild"] >= {"group_members", "index", "routes"}

    remaining = {module: set(imports) for module, imports in dependencies.items()}
    while remaining:
        leaves = {module for module, imports in remaining.items() if not imports & remaining.keys()}
        assert leaves, f"ready implementation import cycle: {remaining}"
        for module in leaves:
            remaining.pop(module)


def test_group_member_rebuild_uses_no_private_online_imports() -> None:
    module = ast.parse((READY_ROOT / "group_members_rebuild.py").read_text(encoding="utf-8"))
    private_imports = [
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom) and node.module == "group_members"
        for alias in node.names
        if alias.name.startswith("_")
    ]
    assert private_imports == []


def test_ready_routes_do_not_depend_on_group_models() -> None:
    module = ast.parse((READY_ROOT / "routes.py").read_text(encoding="utf-8"))
    imported_names = {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "Group" not in imported_names


def test_ready_facade_exports_group_rebuild_owner_objects() -> None:
    from qqtools.plugins.qexp.runtime import ready
    from qqtools.plugins.qexp.runtime.ready import group_members_rebuild

    assert ready.begin_group_ready_members_build is group_members_rebuild.begin_group_ready_members_build
    assert ready.advance_group_ready_members_build is group_members_rebuild.advance_group_ready_members_build
    assert ready.audit_group_ready_members is group_members_rebuild.audit_group_ready_members
    assert ready.repair_group_ready_members is group_members_rebuild.repair_group_ready_members
