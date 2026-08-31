#!/usr/bin/env python3
"""Validate a committed release candidate before creating a release commit."""
from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from scripts.checks.check_compatibility_registry import RegistryError, Version
except ModuleNotFoundError:  # Direct script execution adds scripts/ rather than the repo root.
    from checks.check_compatibility_registry import RegistryError, Version


REPO_ROOT = Path(__file__).resolve().parents[1]
VERSION_PATH = REPO_ROOT / "src" / "qqtools" / "version.py"


def _imported_names(module: ast.Module) -> set[str]:
    """Return names statically re-exported by a stub module."""
    names: set[str] = set()
    for node in module.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        names.update(alias.asname or alias.name for alias in node.names if alias.name != "*")
    return names


def _lazy_exported_names(module: ast.Module) -> set[str]:
    """Return names registered through lazy-export declarations."""
    names: set[str] = set()
    for node in module.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        if not isinstance(node.value.func, ast.Name):
            continue
        if node.value.func.id not in {"lazy_export", "_lazy_export"}:
            continue
        for argument in node.value.args[1:]:
            if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
                raise RuntimeError(
                    f"{node.value.func.id} arguments must be literal export names "
                    "for preflight validation."
                )
            names.add(argument.value)

        if len(node.value.args) < 2:
            raise RuntimeError(f"{node.value.func.id} requires at least one export name.")
    return names


def _lazy_imported_names(module: ast.Module) -> set[str]:
    """Return names assigned to LazyImport proxies."""
    names: set[str] = set()
    for node in module.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name):
            continue
        if value.func.id != "LazyImport":
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def _runtime_exported_names(module: ast.Module) -> set[str]:
    """Return public names made available by the package initializer."""
    names = _imported_names(module) | _lazy_exported_names(module) | _lazy_imported_names(module)

    getattr_nodes = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "__getattr__"
    ]
    for getattr_node in getattr_nodes:
        for node in ast.walk(getattr_node):
            if not isinstance(node, ast.Compare) or len(node.ops) != 1:
                continue
            if not isinstance(node.ops[0], ast.Eq) or len(node.comparators) != 1:
                continue
            left, right = node.left, node.comparators[0]
            if isinstance(left, ast.Name) and left.id == "name" and isinstance(right, ast.Constant):
                if isinstance(right.value, str):
                    names.add(right.value)
    return names


def _check_lazy_export_stubs() -> None:
    """Ensure package stubs match the public lazy-export surface."""
    packages = (
        ("qqtools", REPO_ROOT / "src" / "qqtools" / "__init__.py"),
        (
            "qqtools.plugins.qpipeline",
            REPO_ROOT / "src" / "qqtools" / "plugins" / "qpipeline" / "__init__.py",
        ),
    )
    for package_name, init_path in packages:
        stub_path = init_path.with_suffix(".pyi")
        if not stub_path.is_file():
            relative_stub_path = stub_path.relative_to(REPO_ROOT)
            raise RuntimeError(f"Missing IDE stub for {package_name}: {relative_stub_path}")

        runtime_module = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
        stub_module = ast.parse(stub_path.read_text(encoding="utf-8"), filename=str(stub_path))
        runtime_exports = _runtime_exported_names(runtime_module)
        lazy_exports = _lazy_exported_names(runtime_module) | _lazy_imported_names(runtime_module)
        stub_exports = _imported_names(stub_module)

        missing_lazy_exports = lazy_exports - stub_exports
        stale_stub_exports = stub_exports - runtime_exports
        if missing_lazy_exports or stale_stub_exports:
            details = []
            if missing_lazy_exports:
                details.append(f"missing lazy exports: {sorted(missing_lazy_exports)}")
            if stale_stub_exports:
                details.append(f"stale stub exports: {sorted(stale_stub_exports)}")
            raise RuntimeError(f"{package_name} stub drift: {'; '.join(details)}")


def _run(
    *args: str,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, env=env, text=True, check=True)


def _require_clean_head() -> None:
    _run("git", "rev-parse", "--verify", "HEAD")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT,
        text=True,
        check=True,
        capture_output=True,
    )
    if status.stdout:
        raise RuntimeError("Release preflight requires a clean, committed worktree.")


def _current_version() -> Version:
    module = ast.parse(VERSION_PATH.read_text(encoding="utf-8"), filename=str(VERSION_PATH))
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "__version__":
            value = ast.literal_eval(node.value)
            return Version.parse(value, "src/qqtools/version.py::__version__")
    raise RuntimeError("Could not resolve __version__ from src/qqtools/version.py.")


def _check_target_version(target: Version) -> None:
    current = _current_version()
    if target <= current:
        raise RuntimeError(
            f"Release target {target} must be later than current source version {current}."
        )


def _check_compatibility(target: Version) -> None:
    _run(
        sys.executable,
        "scripts/checks/check_compatibility_registry.py",
        "check",
        "--release-version",
        str(target),
    )


def _build_artifacts(output_dir: Path, *, env: dict[str, str]) -> Path:
    _run(
        sys.executable,
        "-m",
        "build",
        "--quiet",
        "--sdist",
        "--wheel",
        "--outdir",
        str(output_dir),
        env=env,
    )
    wheels = list(output_dir.glob("qqtools-*.whl"))
    sdists = list(output_dir.glob("qqtools-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise RuntimeError("Expected exactly one wheel and one sdist from the release build.")
    return wheels[0]


def _release_env() -> dict[str, str]:
    """Use a repository-local pip cache for repeatable release preflight installs."""
    pip_cache_dir = REPO_ROOT / ".tox" / "pip-cache"
    pip_cache_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PIP_CACHE_DIR"] = str(pip_cache_dir)
    return env


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-version", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        target = Version.parse(args.target_version, "--target-version")
    except RegistryError as exc:
        parser.error(str(exc))
    _require_clean_head()
    _check_target_version(target)
    _check_compatibility(target)
    _check_lazy_export_stubs()
    release_env = _release_env()
    _run("tox", "run", "-e", "qexp-full", env=release_env)
    with tempfile.TemporaryDirectory(prefix="qqtools-preflight-") as temporary_dir:
        wheel = _build_artifacts(Path(temporary_dir), env=release_env)
        _run(
            "tox",
            "run",
            "-e",
            "release-e2e",
            "--installpkg",
            str(wheel),
            env=release_env,
        )
    print(f"Release preflight passed for {target}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
