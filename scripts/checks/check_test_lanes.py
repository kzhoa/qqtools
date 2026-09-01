#!/usr/bin/env python3
"""Enforce the repository's Unit, Integration, and installed-E2E lane boundaries."""
from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _fail(errors: list[str]) -> None:
    if errors:
        raise SystemExit("test lane governance failed:\n" + "\n".join(f"- {error}" for error in errors))


def main() -> int:
    errors: list[str] = []
    for path in (REPO_ROOT / "tests" / "integration").rglob("test_*.py"):
        if "host_exclusive" in path.read_text(encoding="utf-8"):
            errors.append(f"Integration may not use host_exclusive: {path.relative_to(REPO_ROOT)}")
    for path in (REPO_ROOT / "tests" / "e2e").rglob("test_*.py"):
        if "machine_lab" in path.read_text(encoding="utf-8"):
            errors.append(f"E2E may not use machine_lab: {path.relative_to(REPO_ROOT)}")

    tox = _read("tox.ini")
    if "[testenv:qexp-fast]" in tox or "[testenv:qexp-host-exclusive]" in tox or "[testenv:qexp-stress]" in tox:
        errors.append("retired qexp tox lanes are present")
    preflight = tox.split("[testenv:preflight]", 1)
    if len(preflight) != 2 or "tests/e2e" in preflight[1].split("[testenv:", 1)[0]:
        errors.append("preflight must collect only Unit and Integration tests")
    artifact = tox.split("[testenv:artifact-e2e]", 1)
    if len(artifact) != 2 or "tests/e2e" not in artifact[1].split("[testenv:", 1)[0]:
        errors.append("artifact-e2e must collect installed E2E tests")

    workflow = _read(".github/workflows/ci.yml")
    if "schedule:" in workflow:
        errors.append("ordinary CI may not define a schedule trigger")
    for retired in ("tox run -e unit", "tox run -e integration", "tox run -e qexp", "tox run -e preflight"):
        if retired in workflow:
            errors.append(f"ordinary CI may not run source-test lane: {retired}")
    if "tox run -e artifact-e2e" not in workflow:
        errors.append("ordinary CI must run artifact-e2e")

    matrix_path = REPO_ROOT / "tests" / "CONTRACT_MATRIX.md"
    matrix = matrix_path.read_text(encoding="utf-8")
    for link in re.findall(r"\]\(([^)]+)\)", matrix):
        if link.startswith(("http://", "https://", "#")):
            continue
        if not (matrix_path.parent / link).is_file():
            errors.append(f"contract matrix links to a missing test: {link}")
    for retired_term in ("hermetic", "qexp-fast", " / merge"):
        if retired_term in matrix:
            errors.append(f"contract matrix contains retired lane term: {retired_term}")

    _fail(errors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
