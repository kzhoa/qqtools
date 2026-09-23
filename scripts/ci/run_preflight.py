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

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
RELEASE_SLOW_INTEGRATION_NODES = (
    "tests/integration/torch/test_qdataset_process_boundaries.py::test_dataloader_and_file_lock_process_boundaries",
    "tests/integration/functional/test_qpipeline/test_qexp_live_progress.py::"
    "test_unchanged_qpipeline_command_reaches_task_show_progress",
    "tests/integration/test_lmdb_reader_lifecycle.py::test_child_reopens_parent_readers_without_invalidating_parent",
    "tests/integration/test_lmdb_reader_lifecycle.py::test_gc_finalizer_cannot_deadlock_or_invalidate_inflight_lease",
    "tests/integration/test_local_responsibility_storage.py::test_incomplete_writer_marker_survives_transaction_and_replay_crashes",
    "tests/integration/test_local_responsibility_storage.py::test_process_crash_at_every_storage_boundary",
    "tests/integration/test_local_responsibility_storage.py::test_replay_itself_can_crash_repeatedly",
    "tests/integration/test_local_responsibility_storage.py::test_power_cut_model_restores_last_synced_directory",
    "tests/integration/test_local_responsibility_storage.py::test_redo_recovers_every_mixture_of_persisted_after_images",
    "tests/integration/test_local_responsibility_storage.py::test_initialization_resumes_after_every_process_crash_boundary",
    "tests/integration/test_local_responsibility_storage.py::test_cleanup_process_crashes_preserve_receipt_until_durable_deletion",
    "tests/integration/test_local_responsibility_storage.py::test_stage_build_replays_every_process_crash_without_changing_ownership",
)


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
        "--durations=20",
    ),
    (
        PYTHON,
        "-m",
        "pytest",
        "tests/integration/qexp/test_resource_isolation.py",
        "tests/integration/qexp/test_store_crash_boundaries.py",
        "tests/integration/qexp/test_machine_lab.py",
        "-q",
    ),
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

COMMON_COMMANDS: tuple[tuple[str, ...], ...] = COMMANDS[:-2]


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("feature", "release"), default="feature")
    parser.add_argument("--release-base")
    parser.add_argument("--release-head")
    parser.add_argument("--release-actor")
    return parser


def _commands(
    profile: str,
    release_base: str | None,
    release_head: str | None,
    release_actor: str | None,
) -> tuple[tuple[str, ...], ...]:
    if profile == "feature":
        return COMMANDS
    if not all((release_base, release_head, release_actor)):
        _parser().error("--profile release requires --release-base, --release-head, and --release-actor")
    return (
        (
            PYTHON,
            "scripts/checks/check_release_commit.py",
            "validate",
            "--base-ref",
            release_base,
            "--head-ref",
            release_head,
            "--actor",
            release_actor,
        ),
        *COMMON_COMMANDS,
        (PYTHON, "-m", "pytest", *RELEASE_SLOW_INTEGRATION_NODES, "-q", "--durations=20"),
        (
            PYTHON,
            "scripts/qexp_integration_gate.py",
            "--budget-seconds",
            "600",
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    commands = _commands(args.profile, args.release_base, args.release_head, args.release_actor)
    error = check_prerequisites()
    if error:
        print(error, file=sys.stderr)
        return 1
    for command in commands:
        print(f"+ {' '.join(command)}", flush=True)
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
