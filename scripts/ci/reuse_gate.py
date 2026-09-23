"""Reuse recent successful GitHub gate jobs for the exact candidate commit."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

GATES = {
    "feature-preflight": ("CPU-only preflight (Python 3.13)",),
    "release-preflight": ("Release preflight (Python 3.13)",),
    "artifact": (
        "artifact-e2e-dev-release / artifact smoke (Python 3.12)",
        "artifact-e2e-dev-release / artifact smoke (Python 3.14)",
        "artifact-e2e-dev-release / artifact e2e (Python 3.13)",
    ),
}
PROMOTION_PROVENANCE_JOB = "Trusted promotion provenance"


def eligible_run(run: dict, gate: str, env: dict, now: datetime) -> bool:
    """Accept only recent owner runs from the expected dev gate, excluding this run."""
    source_gate = gate in {"feature-preflight", "release-preflight"}
    workflow = "dev-preflight.yml" if source_gate else "repository-governance.yml"
    event = "push" if source_gate else "workflow_dispatch"
    return (
        str(run["id"]) != env["GITHUB_RUN_ID"]
        and run["head_sha"] == env["GITHUB_SHA"]
        and run["head_branch"] == "dev"
        and run["event"] == event
        and run["path"] == f".github/workflows/{workflow}"
        and run["actor"]["login"] == "kzhoa"
        and run.get("triggering_actor", {}).get("login") == "kzhoa"
        and now - timedelta(hours=24) <= datetime.fromisoformat(run["created_at"].replace("Z", "+00:00")) <= now
    )


def gate_state(jobs: list[dict], required: tuple[str, ...]) -> str:
    """Never accept skipped, cancelled, failed, or incomplete gate jobs as evidence."""
    selected = [job for job in jobs if job["name"] in required]
    if any(job["status"] == "completed" and job["conclusion"] != "success" for job in selected):
        return "failed"
    if {job["name"] for job in selected} == set(required) and all(
        job["status"] == "completed" and job["conclusion"] == "success" for job in selected
    ):
        return "passed"
    return "pending"


def is_attested_promotion_without_preflight(jobs: list[dict], required: tuple[str, ...]) -> bool:
    """Return whether a trusted promotion intentionally skipped the source gate."""
    provenance = [job for job in jobs if job["name"] == PROMOTION_PROVENANCE_JOB]
    selected = [job for job in jobs if job["name"] in required]
    return (
        len(provenance) == 1
        and provenance[0]["status"] == "completed"
        and provenance[0]["conclusion"] == "success"
        and {job["name"] for job in selected} == set(required)
        and all(job["status"] == "completed" and job["conclusion"] == "skipped" for job in selected)
    )


def get_json(path: str) -> dict:
    request = urllib.request.Request(
        f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}/{path}",
        headers={"Authorization": f"Bearer {os.environ['GH_TOKEN']}", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def find_evidence(gate: str) -> str | None:
    query = urllib.parse.urlencode({"head_sha": os.environ["GITHUB_SHA"], "per_page": 100})
    runs = get_json(f"actions/runs?{query}")["workflow_runs"]
    now = datetime.now(timezone.utc)
    candidates = [run for run in runs if eligible_run(run, gate, os.environ, now)]
    if not candidates:
        return None
    # The latest attempt is authoritative; an older green run cannot hide a newer failure.
    run = max(candidates, key=lambda item: item["id"])
    run_id = run["id"]
    attempt = run["run_attempt"]
    deadline = time.monotonic() + 1200
    while True:
        current = get_json(f"actions/runs/{run_id}")
        if current["run_attempt"] != attempt:
            raise RuntimeError("Evidence run was rerun; dispatch again after it finishes.")
        jobs = get_json(f"actions/runs/{run_id}/attempts/{attempt}/jobs?per_page=100")["jobs"]
        state = gate_state(jobs, GATES[gate])
        if current["status"] == "completed" and current["conclusion"] != "success":
            raise RuntimeError(f"Evidence workflow did not succeed: {run['html_url']}")
        if state == "passed":
            return f"{run['html_url']}/attempts/{attempt}"
        if gate == "feature-preflight" and is_attested_promotion_without_preflight(jobs, GATES[gate]):
            print("Dev push used promotion provenance only; running a fresh feature preflight.", flush=True)
            return None
        if state == "failed" or current["status"] == "completed":
            raise RuntimeError(f"Gate evidence is unsuccessful: {run['html_url']}")
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Timed out waiting for gate: {run['html_url']}")
        print(f"Waiting for {gate}: {run['html_url']}", flush=True)
        time.sleep(20)


def main() -> None:
    evidence = None
    if os.environ.get("ALLOW_REUSE") == "true":
        try:
            evidence = find_evidence(os.environ["GATE"])
        except (urllib.error.URLError, TimeoutError):
            print("Evidence API unavailable; running the complete gate.")
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        output.write(f"reused={'true' if evidence else 'false'}\n")
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as summary:
        summary.write(f"Gate evidence: {evidence or 'fresh execution required'}\n")


if __name__ == "__main__":
    main()
