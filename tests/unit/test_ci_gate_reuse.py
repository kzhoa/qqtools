from __future__ import annotations

from datetime import datetime, timezone

import pytest

from scripts.ci import reuse_gate

NOW = datetime(2026, 9, 18, tzinfo=timezone.utc)
ENV = {"GITHUB_SHA": "candidate", "GITHUB_RUN_ID": "20"}


def run_fixture(**updates):
    return {
        "id": 10,
        "head_sha": "candidate",
        "head_branch": "dev",
        "event": "push",
        "path": ".github/workflows/dev-preflight.yml",
        "actor": {"login": "kzhoa"},
        "triggering_actor": {"login": "kzhoa"},
        "created_at": "2026-09-17T23:00:00Z",
        "run_attempt": 1,
        "html_url": "https://github.com/kzhoa/qqtools/actions/runs/10",
        "status": "completed",
        "conclusion": "success",
        **updates,
    }


@pytest.mark.parametrize(
    "updates",
    [
        {"id": 20},
        {"head_sha": "other"},
        {"head_branch": "main"},
        {"event": "pull_request"},
        {"path": ".github/workflows/ci.yml"},
        {"actor": {"login": "other"}},
        {"triggering_actor": {"login": "other"}},
        {"triggering_actor": {}},
        {"created_at": "2026-09-16T00:00:00Z"},
        {"created_at": "2026-09-19T00:00:00Z"},
    ],
)
def test_ineligible_evidence(updates):
    assert not reuse_gate.eligible_run(run_fixture(**updates), "preflight", ENV, NOW)


def test_gate_sources_are_distinct():
    assert reuse_gate.eligible_run(run_fixture(), "preflight", ENV, NOW)
    assert not reuse_gate.eligible_run(run_fixture(), "artifact", ENV, NOW)
    assert reuse_gate.eligible_run(
        run_fixture(path=".github/workflows/repository-governance.yml", event="workflow_dispatch"), "artifact", ENV, NOW
    )


@pytest.mark.parametrize("conclusion", ["failure", "skipped", "cancelled", "timed_out"])
def test_unsuccessful_jobs_are_not_evidence(conclusion):
    assert reuse_gate.gate_state([{"name": "a", "status": "completed", "conclusion": conclusion}], ("a",)) == "failed"


def test_all_required_jobs_must_finish():
    success = {"name": "a", "status": "completed", "conclusion": "success"}
    assert reuse_gate.gate_state([success], ("a",)) == "passed"
    assert reuse_gate.gate_state([success], ("a", "b")) == "pending"
    assert reuse_gate.gate_state([success, success], ("a", "b")) == "pending"
    assert reuse_gate.gate_state([{**success, "status": "in_progress", "conclusion": None}], ("a",)) == "pending"


def test_attested_promotion_is_not_reusable_preflight_evidence():
    jobs = [
        {"name": reuse_gate.PROMOTION_PROVENANCE_JOB, "status": "completed", "conclusion": "success"},
        {"name": reuse_gate.GATES["preflight"][0], "status": "completed", "conclusion": "skipped"},
    ]
    assert reuse_gate.is_attested_promotion_without_preflight(jobs, reuse_gate.GATES["preflight"])
    assert not reuse_gate.is_attested_promotion_without_preflight(jobs[:1], reuse_gate.GATES["preflight"])


def setup_api(monkeypatch, runs, states):
    for key, value in ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(reuse_gate, "eligible_run", lambda run, *args: True)
    replies = iter(states)

    def get_json(path):
        if path.startswith("actions/runs?"):
            return {"workflow_runs": runs}
        return next(replies)

    monkeypatch.setattr(reuse_gate, "get_json", get_json)
    monkeypatch.setattr(reuse_gate.time, "sleep", lambda _: None)


def test_waits_for_running_gate(monkeypatch):
    run = run_fixture(status="in_progress", conclusion=None)
    name = reuse_gate.GATES["preflight"][0]
    setup_api(
        monkeypatch,
        [run],
        [run, {"jobs": []}, run_fixture(), {"jobs": [{"name": name, "status": "completed", "conclusion": "success"}]}],
    )
    assert reuse_gate.find_evidence("preflight").endswith("/10/attempts/1")


@pytest.mark.parametrize("run", [run_fixture(run_attempt=2), run_fixture(conclusion="cancelled")])
def test_reruns_and_cancelled_sources_fail(monkeypatch, run):
    setup_api(monkeypatch, [run_fixture()], [run, {"jobs": []}])
    with pytest.raises(RuntimeError):
        reuse_gate.find_evidence("preflight")


def test_no_candidate_requires_fresh_gate(monkeypatch):
    setup_api(monkeypatch, [], [])
    assert reuse_gate.find_evidence("preflight") is None


def test_attested_dev_push_selects_fresh_release_preflight(monkeypatch):
    run = run_fixture()
    jobs = [
        {"name": reuse_gate.PROMOTION_PROVENANCE_JOB, "status": "completed", "conclusion": "success"},
        {"name": reuse_gate.GATES["preflight"][0], "status": "completed", "conclusion": "skipped"},
    ]
    setup_api(monkeypatch, [run], [run, {"jobs": jobs}])
    assert reuse_gate.find_evidence("preflight") is None


def test_latest_failure_does_not_fall_back_to_older_success(monkeypatch):
    latest = run_fixture(id=11, conclusion="failure")
    setup_api(monkeypatch, [run_fixture(), latest], [latest, {"jobs": []}])
    with pytest.raises(RuntimeError):
        reuse_gate.find_evidence("preflight")


def test_api_unavailable_selects_full_execution(monkeypatch, tmp_path):
    import urllib.error

    output = tmp_path / "output"
    summary = tmp_path / "summary"
    monkeypatch.setenv("ALLOW_REUSE", "true")
    monkeypatch.setenv("GATE", "preflight")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    def unavailable(_):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(reuse_gate, "find_evidence", unavailable)
    reuse_gate.main()
    assert output.read_text() == "reused=false\n"
    assert "fresh execution required" in summary.read_text()


def test_wait_timeout_fails_instead_of_starting_duplicate_gate(monkeypatch):
    run = run_fixture(status="in_progress", conclusion=None)
    setup_api(monkeypatch, [run], [run, {"jobs": []}])
    clock = iter([0, 1201])
    monkeypatch.setattr(reuse_gate.time, "monotonic", lambda: next(clock))
    with pytest.raises(RuntimeError, match="Timed out"):
        reuse_gate.find_evidence("preflight")
