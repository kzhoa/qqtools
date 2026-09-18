#!/usr/bin/env python3
"""Run the shared qqtools preflight validation commands.

Environment setup is intentionally outside this script:
- local developers use ``./scripts/dev preflight`` with ``.[full]``;
- GitHub-hosted CI installs an explicit CPU-only PyTorch environment.

Both paths execute this exact command list so validation semantics cannot drift.
The torch unit subtree runs in its own pytest process because it exercises
PyTorch/DataLoader multiprocessing behavior that should not inherit process and
thread state from the rest of the unit suite.
"""

from __future__ import annotations

import shutil
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


def check_prerequisites() -> str | None:
    if sys.version_info[:2] != (3, 13):
        return (
            f"Preflight requires Python 3.13; got {sys.version_info.major}.{sys.version_info.minor}. "
            "Locally, run: ./scripts/dev preflight. In CI, verify the configured Python interpreter."
        )
    if sys.platform != "linux":
        return "Preflight requires Linux for the real-process lifecycle gate. Use a Linux development host."
    if shutil.which("tmux") is None:
        return "Preflight requires tmux. Install tmux with your system package manager, then rerun ./scripts/dev preflight."
    return None


def main() -> int:
    error = check_prerequisites()
    if error:
        print(error, file=sys.stderr)
        return 1
    for command in COMMANDS:
        print(f"+ {' '.join(command)}", flush=True)
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
