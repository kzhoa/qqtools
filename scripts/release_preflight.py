#!/usr/bin/env python3
"""Validate a committed release candidate before creating a release commit."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, check=True)


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


def _build_artifacts(output_dir: Path) -> Path:
    _run(
        sys.executable,
        "-m",
        "build",
        "--quiet",
        "--sdist",
        "--wheel",
        "--outdir",
        str(output_dir),
    )
    wheels = list(output_dir.glob("qqtools-*.whl"))
    sdists = list(output_dir.glob("qqtools-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise RuntimeError("Expected exactly one wheel and one sdist from the release build.")
    return wheels[0]


def main() -> int:
    _require_clean_head()
    with tempfile.TemporaryDirectory(prefix="qqtools-preflight-") as temporary_dir:
        wheel = _build_artifacts(Path(temporary_dir))
        _run("tox", "run", "-e", "release-e2e", "--installpkg", str(wheel))
    print("Release preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
