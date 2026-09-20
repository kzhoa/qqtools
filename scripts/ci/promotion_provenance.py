#!/usr/bin/env python3
"""Render the deterministic subject attested by feature promotion."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

SCHEMA = "https://x1q.cc/qqtools/promotion-provenance/v1"
PROVENANCE_MARKER = "github-attestation-v1"
SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
REPOSITORY_PATTERN = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
TRAILER_KEYS = (
    "Promoted-from",
    "Promotion-source",
    "Promotion-run",
    "Promotion-attempt",
    "Promotion-provenance",
)


@dataclass(frozen=True)
class PromotionProvenance:
    schema: str
    repository: str
    target_commit: str
    target_tree: str
    target_parent: str
    source_commit: str
    source_ref: str
    workflow_run: int
    workflow_attempt: int


def _git(*args: str, input_text: str | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ValueError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.rstrip("\n")


def _positive_integer(value: str, name: str) -> int:
    if not value.isdigit() or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _trailers(commit: str) -> dict[str, str]:
    message = _git("show", "-s", "--format=%B", commit)
    parsed = _git("interpret-trailers", "--parse", input_text=message)
    values: dict[str, list[str]] = {}
    for line in parsed.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            values.setdefault(key, []).append(value.strip())

    trailers: dict[str, str] = {}
    for key in TRAILER_KEYS:
        matches = values.get(key, [])
        if len(matches) != 1:
            raise ValueError(f"commit must contain exactly one {key} trailer")
        trailers[key] = matches[0]
    return trailers


def provenance_for_commit(commit: str, repository: str, expected_parent: str) -> PromotionProvenance:
    """Build and validate provenance fields for one promoted commit."""
    if not REPOSITORY_PATTERN.fullmatch(repository):
        raise ValueError("repository must use owner/name form")

    target_commit = _git("rev-parse", f"{commit}^{{commit}}")
    target_tree = _git("rev-parse", f"{target_commit}^{{tree}}")
    parents = _git("show", "-s", "--format=%P", target_commit).split()
    if len(parents) != 1:
        raise ValueError("promoted commit must have exactly one parent")
    if parents[0] != expected_parent:
        raise ValueError("promoted commit parent does not match the previous dev commit")

    trailers = _trailers(target_commit)
    source_branch = trailers["Promoted-from"]
    source_commit = trailers["Promotion-source"]
    if not source_branch.startswith(("feat/", "feature/")):
        raise ValueError("Promoted-from must name a feat/* or feature/* branch")
    _git("check-ref-format", "--branch", source_branch)
    if not SHA_PATTERN.fullmatch(source_commit):
        raise ValueError("Promotion-source must be a lowercase SHA-1 commit ID")
    if trailers["Promotion-provenance"] != PROVENANCE_MARKER:
        raise ValueError(f"Promotion-provenance must be {PROVENANCE_MARKER}")

    return PromotionProvenance(
        schema=SCHEMA,
        repository=repository,
        target_commit=target_commit,
        target_tree=target_tree,
        target_parent=parents[0],
        source_commit=source_commit,
        source_ref=f"refs/heads/{source_branch}",
        workflow_run=_positive_integer(trailers["Promotion-run"], "Promotion-run"),
        workflow_attempt=_positive_integer(trailers["Promotion-attempt"], "Promotion-attempt"),
    )


def _write_outputs(path: Path, provenance: PromotionProvenance) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write("valid=true\n")
        output.write(f"source_sha={provenance.source_commit}\n")
        output.write(f"source_ref={provenance.source_ref}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--expected-parent", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()

    try:
        provenance = provenance_for_commit(args.commit, args.repository, args.expected_parent)
        args.output.write_text(
            json.dumps(asdict(provenance), sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        if args.github_output is not None:
            _write_outputs(args.github_output, provenance)
    except (OSError, ValueError) as exc:
        print(f"promotion-provenance: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
