from __future__ import annotations

import ast
from pathlib import Path

_QEXP_ROOT = Path(__file__).parents[4] / "src" / "qqtools" / "plugins" / "qexp"
_OUTPUT_ROOT = _QEXP_ROOT / "cli" / "output"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def test_output_family_modules_do_not_import_domain_io() -> None:
    forbidden = {
        "agent",
        "commands",
        "layout",
        "observer",
        "runtime",
        "scheduler",
        "store",
    }
    violations: list[str] = []
    for path in sorted(_OUTPUT_ROOT.glob("*.py")):
        for imported in _imports(path):
            components = set(imported.split("."))
            if components & forbidden:
                violations.append(f"{path.name}: {imported}")

    assert violations == []


def test_finite_dispatchers_return_outcomes_instead_of_calling_an_emitter() -> None:
    for name in ("local_handlers.py", "project_handlers.py", "submission.py"):
        path = _QEXP_ROOT / "cli" / name
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "emitter"
        ]
        dispatchers = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("dispatch_")
        ]

        assert calls == [], name
        assert dispatchers, name
        assert all("emitter" not in {argument.arg for argument in dispatcher.args.args} for dispatcher in dispatchers)


def test_handler_prints_are_limited_to_non_finite_streams_and_diagnostics() -> None:
    for name in ("local_handlers.py", "project_handlers.py", "submission.py"):
        path = _QEXP_ROOT / "cli" / name
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        unexpected: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "print":
                continue
            source = ast.unparse(node)
            is_stderr = any(
                keyword.arg == "file" and ast.unparse(keyword.value) == "sys.stderr" for keyword in node.keywords
            )
            is_raw_log = bool(node.args and "log_commands.read_logs" in ast.unparse(node.args[0]))
            is_quiet_id = bool(node.args and "task_id" in ast.unparse(node.args[0]))
            if not (is_stderr or is_raw_log or is_quiet_id):
                unexpected.append(source)

        assert unexpected == [], name
