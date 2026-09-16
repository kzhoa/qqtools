#!/usr/bin/env python3
"""Run the shared qqtools preflight validation commands.

Environment setup is intentionally outside this script:
- local developers use ``tox -e preflight`` with ``.[full]``;
- GitHub-hosted CI installs an explicit CPU-only PyTorch environment.

Both paths execute this exact command list so validation semantics cannot drift.
The torch unit subtree runs in its own pytest process because it exercises
PyTorch/DataLoader multiprocessing behavior that should not inherit process and
thread state from the rest of the unit suite.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable

COMMANDS: tuple[tuple[str, ...], ...] = (
    (PYTHON, "-m", "ruff", "check", "src", "tests", "scripts"),
    (PYTHON, "-m", "ruff", "format", "--check", "src", "tests", "scripts"),
    (PYTHON, "scripts/checks/check_repository_governance.py"),
    (PYTHON, "scripts/checks/check_test_lanes.py"),
    (PYTHON, "scripts/checks/check_contract_matrix.py"),
    (PYTHON, "-m", "pytest", "tests/unit", "--ignore=tests/unit/torch", "-q"),
    (PYTHON, "-m", "pytest", "tests/unit/torch", "-q"),
    (
        PYTHON,
        "-m",
        "pytest",
        "tests/integration",
        "--ignore=tests/integration/qexp",
        "-m",
        "not slow and not gpu and not ddp",
        "-q",
    ),
    (PYTHON, "-m", "pytest", "tests/integration/qexp/test_resource_isolation.py", "-q"),
    (PYTHON, "-m", "pytest", "tests/integration/qexp/test_store_crash_boundaries.py", "-q"),
    (PYTHON, "-m", "pytest", "tests/integration/qexp/test_machine_lab.py", "-q"),
    (
        PYTHON,
        "-m",
        "pytest",
        "--lifecycle-gate=representative",
        "tests/integration/qexp/test_agent_lifecycle_independence.py",
        "-k",
        "li01 or li02 or li04",
        "-q",
    ),
)


def main() -> int:
    for command in COMMANDS:
        print(f"+ {' '.join(command)}", flush=True)
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
