#!/usr/bin/env python3
"""Enforce the repository's Unit, Integration, and installed-E2E lane boundaries."""

from __future__ import annotations

import ast
import re
from configparser import ConfigParser, Error as ConfigError
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RETIRED_TOX_LANES = (
    "testenv:qexp-fast",
    "testenv:qexp-host-exclusive",
    "testenv:qexp-stress",
)
SOURCE_TOX_LANES = frozenset(("unit", "integration", "qexp", "preflight"))
TOX_ENVIRONMENT_PATTERN = re.compile(
    r"(?:^|\s)(?:python(?:\d+(?:\.\d+)*)?\s+-m\s+)?tox(?:\s+run)?"
    r"(?:\s+(?!--?e(?:\s|=|$))--?[A-Za-z][\w-]*(?:[= ][^\s]+)?)*\s+-e\s*=?\s*"
    r"([A-Za-z0-9_.-]+(?:,[A-Za-z0-9_.-]+)*)"
)


def _pytest_aliases(module: ast.Module) -> tuple[set[str], set[str]]:
    pytest_aliases: set[str] = set()
    mark_aliases: set[str] = set()
    for node in module.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pytest":
                    pytest_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "pytest":
            for alias in node.names:
                if alias.name == "mark":
                    mark_aliases.add(alias.asname or alias.name)
    return pytest_aliases, mark_aliases


def _uses_pytest_marker(path: Path, marker: str) -> bool:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    pytest_aliases, mark_aliases = _pytest_aliases(module)
    for node in ast.walk(module):
        if not isinstance(node, ast.Attribute) or node.attr != marker:
            continue
        owner = node.value
        if isinstance(owner, ast.Name) and owner.id in mark_aliases:
            return True
        if not isinstance(owner, ast.Attribute) or owner.attr != "mark":
            continue
        if isinstance(owner.value, ast.Name) and owner.value.id in pytest_aliases:
            return True
    return False


def _check_marker_boundaries(repo_root: Path) -> list[str]:
    errors: list[str] = []
    boundaries = (
        ("Integration", repo_root / "tests/integration", "host_exclusive"),
        ("E2E", repo_root / "tests/e2e", "machine_lab"),
    )
    for lane, test_root, forbidden_marker in boundaries:
        for path in test_root.rglob("test_*.py"):
            if _uses_pytest_marker(path, forbidden_marker):
                errors.append(
                    f"{lane} may not use {forbidden_marker}: {path.relative_to(repo_root)}"
                )
    return errors


def _load_tox_config(path: Path) -> ConfigParser:
    config = ConfigParser(interpolation=None)
    with path.open(encoding="utf-8") as file:
        config.read_file(file)
    return config


def _section_commands(config: ConfigParser, section: str) -> str | None:
    if not config.has_section(section):
        return None
    return config.get(section, "commands", fallback="")


def _check_tox_boundaries(repo_root: Path) -> list[str]:
    try:
        config = _load_tox_config(repo_root / "tox.ini")
    except ConfigError as error:
        return [f"tox.ini could not be parsed: {error}"]

    errors: list[str] = []
    if any(config.has_section(section) for section in RETIRED_TOX_LANES):
        errors.append("retired qexp tox lanes are present")

    preflight_commands = _section_commands(config, "testenv:preflight")
    if preflight_commands is None or "tests/e2e" in preflight_commands:
        errors.append("preflight must collect only Unit and Integration tests")

    artifact_commands = _section_commands(config, "testenv:artifact-e2e")
    if artifact_commands is None or "tests/e2e" not in artifact_commands:
        errors.append("artifact-e2e must collect installed E2E tests")
    return errors


def _check_ci_boundaries(repo_root: Path) -> list[str]:
    workflow = (repo_root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    errors: list[str] = []
    if "schedule:" in workflow:
        errors.append("ordinary CI may not define a schedule trigger")
    for match in TOX_ENVIRONMENT_PATTERN.finditer(workflow):
        command = match.group(0).strip()
        for environment in match.group(1).split(","):
            if environment in SOURCE_TOX_LANES:
                errors.append(
                    f"ordinary CI may not run source-test lane: {command}"
                )
    if "tox run -e artifact-e2e" not in workflow:
        errors.append("ordinary CI must run artifact-e2e")
    return errors


def check_test_lanes(repo_root: Path = REPO_ROOT) -> list[str]:
    """Return violations of the repository's executable test-lane boundaries."""
    return [
        *_check_marker_boundaries(repo_root),
        *_check_tox_boundaries(repo_root),
        *_check_ci_boundaries(repo_root),
    ]


def main() -> int:
    errors = check_test_lanes()
    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise SystemExit(f"test lane governance failed:\n{details}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
