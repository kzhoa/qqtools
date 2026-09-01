#!/usr/bin/env python3
"""Validate local evidence links and lane terms in the public contract matrix."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RETIRED_LANE_TERMS = ("hermetic", "qexp-fast", " / merge")


def check_contract_matrix(repo_root: Path = REPO_ROOT) -> list[str]:
    """Return invalid links and retired lane terms from the contract matrix."""
    matrix_path = repo_root / "tests/CONTRACT_MATRIX.md"
    matrix = matrix_path.read_text(encoding="utf-8")
    errors: list[str] = []
    for link in re.findall(r"\]\(([^)]+)\)", matrix):
        if link.startswith(("http://", "https://", "#")):
            continue
        if not (matrix_path.parent / link).is_file():
            errors.append(f"contract matrix links to a missing test: {link}")
    for retired_term in RETIRED_LANE_TERMS:
        if retired_term in matrix:
            errors.append(f"contract matrix contains retired lane term: {retired_term}")
    return errors


def main() -> int:
    errors = check_contract_matrix()
    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise SystemExit(f"contract matrix governance failed:\n{details}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
