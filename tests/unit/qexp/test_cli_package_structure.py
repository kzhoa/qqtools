from __future__ import annotations

import ast
import importlib
import tomllib
from pathlib import Path

import qqtools.plugins.qexp.cli as cli_package

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CLI_ROOT = REPOSITORY_ROOT / "src" / "qqtools" / "plugins" / "qexp" / "cli"
CLI_MODULES = (
    "entrypoint",
    "parser",
    "submission",
    "local_handlers",
    "project_handlers",
)


def _cli_imports(module: str) -> list[ast.ImportFrom]:
    tree = ast.parse((CLI_ROOT / f"{module}.py").read_text(encoding="utf-8"))
    return [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]


def test_cli_package_initializer_exposes_no_implementation_symbols() -> None:
    assert not hasattr(cli_package, "main")
    assert not hasattr(cli_package, "build_parser")
    assert not hasattr(cli_package, "LOCAL_HANDLERS")
    assert not hasattr(cli_package, "PROJECT_HANDLERS")

    tree = ast.parse((CLI_ROOT / "__init__.py").read_text(encoding="utf-8"))
    assert not [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]


def test_each_cli_owner_module_imports_independently() -> None:
    for module in CLI_MODULES:
        assert importlib.import_module(f"qqtools.plugins.qexp.cli.{module}")


def test_console_script_names_the_true_entrypoint_owner() -> None:
    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["scripts"]["qexp"] == "qqtools.plugins.qexp.cli.entrypoint:main"
    assert importlib.import_module("qqtools.plugins.qexp.cli.entrypoint").main


def test_cli_dependency_direction_has_no_forbidden_back_edges() -> None:
    for module in CLI_MODULES:
        imports = _cli_imports(module)
        assert not [node for node in imports if node.level == 1 and node.module is None], module

    parser_imports = {node.module for node in _cli_imports("parser")}
    assert parser_imports.isdisjoint({"entrypoint", "submission", "local_handlers", "project_handlers"})

    for module in ("submission", "local_handlers", "project_handlers"):
        imports = {node.module for node in _cli_imports(module)}
        assert "entrypoint" not in imports, module
