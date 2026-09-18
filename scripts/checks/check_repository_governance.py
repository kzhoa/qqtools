#!/usr/bin/env python3
"""Validate repository-level development governance invariants."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTS_PATH = REPO_ROOT / "AGENTS.md"
WORKFLOW_ROOT = REPO_ROOT / ".github" / "workflows"

REQUIRED_MARKERS = (
    "<!-- qqtools-governance:branch-model=v1 -->",
    "<!-- qqtools-governance:strip-dot-dev=v1 -->",
    "<!-- qqtools-governance:agents-owner=kzhoa -->",
    "<!-- qqtools-governance:no-pr-required=v1 -->",
    "<!-- qqtools-governance:workflow-policy=v1 -->",
    "<!-- qqtools-governance:release-model=v1 -->",
)

REQUIRED_SECTIONS = (
    "## Branch model",
    "## Feature-local agent state",
    "## Feature promotion",
    "## Dev release promotion",
    "## Validation",
    "## Workflow governance",
    "## Compatibility governance",
    "## Protected governance surface",
)

ALLOWED_WORKFLOWS = {
    "ci.yml",
    "dev-preflight.yml",
    "publish.yml",
    "repository-governance.yml",
}

PUBLIC_BRANCHES = {"dev", "main"}


def _current_branch() -> str | None:
    github_ref = os.environ.get("GITHUB_REF_NAME")
    if github_ref:
        return github_ref
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    branch = result.stdout.strip()
    return branch or None


def _tracked_dev_paths() -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "ls-files", ".dev"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return ()
    return tuple(line for line in result.stdout.splitlines() if line)


def _workflow_names() -> set[str]:
    if not WORKFLOW_ROOT.is_dir():
        return set()
    return {path.name for path in WORKFLOW_ROOT.iterdir() if path.is_file()}


def main() -> int:
    errors: list[str] = []

    try:
        content = AGENTS_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"repository-governance: could not read AGENTS.md: {exc}", file=sys.stderr)
        return 1

    for marker in REQUIRED_MARKERS:
        count = content.count(marker)
        if count != 1:
            errors.append(f"AGENTS.md must contain governance marker exactly once: {marker!r} (found {count})")

    for section in REQUIRED_SECTIONS:
        if section not in content:
            errors.append(f"AGENTS.md is missing required section: {section}")

    if "Only GitHub actor `kzhoa` may intentionally modify this protected governance surface." not in content:
        errors.append("AGENTS.md must retain the owner-only governance statement for GitHub actor kzhoa")

    workflow_rule = (
        "Agents must not create, delete, rename, or modify any file under `.github/workflows/**` "
        "unless the repository owner has explicitly approved that workflow change in the current task."
    )
    if workflow_rule not in content:
        errors.append("AGENTS.md must retain the explicit owner-approval rule for workflow changes")

    workflow_names = _workflow_names()
    missing_workflows = sorted(ALLOWED_WORKFLOWS - workflow_names)
    unexpected_workflows = sorted(workflow_names - ALLOWED_WORKFLOWS)
    if missing_workflows:
        errors.append(f"required stable workflows are missing: {missing_workflows}")
    if unexpected_workflows:
        errors.append(f"unexpected workflow files are forbidden: {unexpected_workflows}")

    branch = _current_branch()
    if branch in PUBLIC_BRANCHES:
        tracked = _tracked_dev_paths()
        if tracked:
            errors.append(f"{branch} may not contain tracked .dev/** state: {list(tracked)}")
        if (REPO_ROOT / ".dev").exists():
            errors.append(f"{branch} may not contain a .dev directory")

    if errors:
        print("repository-governance: validation failed", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    suffix = f" on {branch}" if branch else ""
    print(f"Repository governance is valid{suffix}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
