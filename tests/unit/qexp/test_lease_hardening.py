from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.lease import (LeasePolicy, LeaseRenewalOutcome, chrony_health,
                                        load_lease_policy, save_lease_policy)
from qqtools.plugins.qexp.layout import load_root_config, migrate_schema5_to_schema6
from qqtools.plugins.qexp.runtime.paths import attempt_path
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.termination import (commit_local_unavailable, commit_signal,
                                                       create_decision, is_recovery_blocked,
                                                       send_signals, update_decision)
from qqtools.plugins.qexp.scheduler import (authorize_launch, claim_task, commit_shared_termination,
                                             _process_start_time_ticks, expire_claim, reconcile_running_tasks,
                                             renew_attempt_lease)
from qqtools.plugins.qexp.runtime.recovery import recover_running_attempt


def test_renewal_error_is_classified_without_fencing(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    monkeypatch.setattr("qqtools.plugins.qexp.scheduler.load_task", lambda *_: (_ for _ in ()).throw(OSError("down")))
    result = renew_attempt_lease(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert result.outcome is LeaseRenewalOutcome.RETRYABLE_ERROR
    assert result.error and result.error.error_type == "OSError"


def test_local_irreversible_commitment_blocks_recovery(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    decision = create_decision(
        cfg, task_id="task", attempt_id="attempt", fencing_token=7,
        process={"process_group_id": 1, "process_group_start_time_ticks": 2},
        authority_outcome="authority_unavailable", reason="lease_authority_unavailable",
    )
    commit_local_unavailable(cfg, "attempt", decision["decision_id"])
    assert is_recovery_blocked(cfg, "attempt")
    commit_signal(cfg, "attempt", decision["decision_id"])
    assert is_recovery_blocked(cfg, "attempt")


def test_chrony_health_rejects_large_offset():
    class Completed:
        returncode = 0
        stdout = "System time     : 2.0 seconds slow of NTP time\nRoot dispersion : 0.1 seconds\nLeap status     : Normal\n"

    assert chrony_health(LeasePolicy(), run=lambda *_args, **_kwargs: Completed()) == (False, "chrony_offset_exceeds_policy")


def test_clock_gate_blocks_claim_renewal_recovery_and_expiry(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    monkeypatch.setattr("qqtools.plugins.qexp.scheduler.authority_clock_health",
                        lambda *_args: (False, "chrony_unavailable"))
    assert claim_task(cfg, task.task_id, [0]) is None

    monkeypatch.setattr("qqtools.plugins.qexp.scheduler.authority_clock_health",
                        lambda *_args: (True, "healthy"))
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    monkeypatch.setattr("qqtools.plugins.qexp.scheduler.authority_clock_health",
                        lambda *_args: (False, "chrony_unavailable"))
    renewal = renew_attempt_lease(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert renewal.outcome is LeaseRenewalOutcome.RETRYABLE_ERROR
    assert renewal.error and renewal.error.error_type == "ClockHealthError"
    assert not expire_claim(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    monkeypatch.setattr("qqtools.plugins.qexp.runtime.recovery.authority_clock_health",
                        lambda *_args: (False, "chrony_unavailable"))
    assert recover_running_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token) is None


def test_shared_termination_commitment_rejects_renewal(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    task_file = cfg.shared_root / "tasks" / f"{task.task_id}.json"
    value = read_json(task_file)
    claim = value["task"]["claim_control"]["active_claim"]
    claim["termination_decision_id"] = "decision-1"
    claim["termination_decision_token"] = attempt.current_fencing_token
    previous_expiry = claim["lease_expires_at"]
    atomic_replace(task_file, value)
    result = renew_attempt_lease(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert result.outcome is LeaseRenewalOutcome.TERMINATION_REQUESTED
    assert read_json(task_file)["task"]["claim_control"]["active_claim"]["lease_expires_at"] == previous_expiry


def test_committed_termination_is_completed_by_agent_and_blocks_recovery(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    decision = create_decision(
        cfg, task_id=task.task_id, attempt_id=attempt.attempt_id,
        fencing_token=attempt.current_fencing_token,
        process={"process_group_id": 1, "process_group_start_time_ticks": 2},
        authority_outcome="termination_required", reason="fenced",
    )
    assert commit_shared_termination(
        cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, decision["decision_id"]
    )
    atomic_replace(cfg.runtime_root / "processes" / f"{attempt.attempt_id}.json", {"process": {
        "task_id": task.task_id,
        "attempt_id": attempt.attempt_id,
        "fencing_token": attempt.current_fencing_token,
        "process_group_id": 1,
    }})
    monkeypatch.setattr("qqtools.plugins.qexp.scheduler._process_evidence_state",
                        lambda *_args: "alive")
    monkeypatch.setattr("qqtools.plugins.qexp.runtime.termination._matches_process_group",
                        lambda *_args: False)
    reconcile_running_tasks(cfg)
    stored = read_json(
        cfg.runtime_root / "termination-decisions" / attempt.attempt_id / f"{decision['decision_id']}.json"
    )["termination_decision"]
    assert stored["state"] == "confirmed"
    assert stored["shared_commitment"] == "committed"


def test_sigkill_is_not_confirmed_until_process_identity_is_absent(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    decision = create_decision(
        cfg, task_id="task", attempt_id="attempt", fencing_token=7,
        process={"process_group_id": 1, "process_group_start_time_ticks": 2},
        authority_outcome="termination_required", reason="fenced",
    )
    commit_signal(cfg, "attempt", decision["decision_id"])
    monkeypatch.setattr("qqtools.plugins.qexp.runtime.termination._matches_process_group",
                        lambda *_args: True)
    monkeypatch.setattr("qqtools.plugins.qexp.runtime.termination.os.killpg", lambda *_args: None)
    current = send_signals(cfg, "attempt", decision["decision_id"], grace_seconds=0)
    assert current["state"] == "sigkill_sent"
    monkeypatch.setattr("qqtools.plugins.qexp.runtime.termination._matches_process_group",
                        lambda *_args: False)
    assert send_signals(cfg, "attempt", decision["decision_id"], grace_seconds=0)["state"] == "confirmed"


def test_termination_decision_rejects_state_rollback_and_wait_confirmation(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    decision = create_decision(
        cfg, task_id="task", attempt_id="attempt", fencing_token=7,
        process={"process_group_id": 1, "process_group_start_time_ticks": 2},
        authority_outcome="termination_required", reason="fenced",
    )
    commit_signal(cfg, "attempt", decision["decision_id"])
    with pytest.raises(RuntimeError, match="confirmation requires absent process identity"):
        update_decision(cfg, "attempt", decision["decision_id"], state="confirmed",
                        confirmation="runner_waited")
    with pytest.raises(RuntimeError, match="not monotonic"):
        update_decision(cfg, "attempt", decision["decision_id"], state="pending")


def test_termination_confirmation_uses_real_process_group_identity(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    child = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        decision = create_decision(
            cfg, task_id="task", attempt_id="attempt", fencing_token=7,
            process={"process_group_id": child.pid,
                     "process_group_start_time_ticks": _process_start_time_ticks(child.pid)},
            authority_outcome="termination_required", reason="fenced",
        )
        commit_signal(cfg, "attempt", decision["decision_id"])
        state = send_signals(cfg, "attempt", decision["decision_id"], grace_seconds=0)
        assert state["state"] in {"sigterm_sent", "sigkill_sent", "confirmed"}
        child.wait(timeout=5)
        assert send_signals(cfg, "attempt", decision["decision_id"], grace_seconds=0)["state"] == "confirmed"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_recovery_uses_authoritative_policy_ttl(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    save_lease_policy(cfg, LeasePolicy(ttl_seconds=180))
    task = submit(cfg, ["echo", "ok"], group="exp", sharing_mode="spillover")
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert expire_claim(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    manifest_path = cfg.runtime_root / "processes" / f"{attempt.attempt_id}.json"
    atomic_replace(manifest_path, {"process": {
        "task_id": task.task_id, "attempt_id": attempt.attempt_id,
        "fencing_token": attempt.current_fencing_token,
    }})
    token = recover_running_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert token == attempt.current_fencing_token + 1
    recovered = read_json(attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number))["attempt"]
    expires_at = recovered["lease"]["expires_at"]
    from qqtools.plugins.qexp.lease import parse_utc
    from datetime import datetime, timezone
    remaining = (parse_utc(expires_at) - datetime.now(timezone.utc)).total_seconds()
    assert 175 <= remaining <= 180


def test_schema5_migration_requires_drain_then_writes_policy(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    schema = cfg.shared_root / "schema" / "version.json"
    value = read_json(schema)
    value["schema"]["version"] = 5
    value["schema"]["minimum_reader_version"] = 5
    atomic_replace(schema, value)
    migrate_schema5_to_schema6(cfg)
    upgraded = load_root_config(cfg.shared_root, "g1", cfg.runtime_root, require_initialized=True)
    assert load_lease_policy(upgraded).ttl_seconds == 120


def test_schema5_migration_rejects_active_claim(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    assert claim_task(cfg, task.task_id, [0])
    schema = cfg.shared_root / "schema" / "version.json"
    value = read_json(schema)
    value["schema"]["version"] = 5
    value["schema"]["minimum_reader_version"] = 5
    atomic_replace(schema, value)
    with pytest.raises(RuntimeError, match="requires no active claims"):
        migrate_schema5_to_schema6(cfg)
