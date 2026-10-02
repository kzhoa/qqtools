from __future__ import annotations

import hashlib
import signal
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp import layout as qexp_layout
from qqtools.plugins.qexp.commands.group import create_group
from qqtools.plugins.qexp.infrastructure.process import process_start_time_ticks as _process_start_time_ticks
from qqtools.plugins.qexp.layout import load_root_config, migrate_schema5_to_schema6
from qqtools.plugins.qexp.lease import (
    ClockCapability,
    ClockObservation,
    LeasePolicy,
    LeaseRenewalOutcome,
    lease_policy_path,
    load_lease_policy,
    save_lease_policy,
)
from qqtools.plugins.qexp.runtime.attempt_recovery import recover_running_attempt
from qqtools.plugins.qexp.runtime.paths import attempt_path, shared_paths, task_path
from qqtools.plugins.qexp.runtime.process_evidence import ProcessEvidence
from qqtools.plugins.qexp.runtime.records import AttemptRecord
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.termination import (
    commit_local_unavailable,
    commit_signal,
    create_decision,
    is_recovery_blocked,
    send_signals,
    update_decision,
)
from qqtools.plugins.qexp.scheduler import (
    authorize_launch,
    claim_task,
    commit_shared_termination,
    expire_claim,
    reconcile_running_tasks,
    renew_attempt_lease,
    renew_project_io_attempt_lease,
)

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def test_renewal_error_is_classified_without_fencing(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    monkeypatch.setattr("qqtools.plugins.qexp.scheduler.load_task", lambda *_: (_ for _ in ()).throw(OSError("down")))
    result = renew_attempt_lease(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert result.outcome is LeaseRenewalOutcome.RETRYABLE_ERROR
    assert result.error and result.error.error_type == "OSError"


def test_project_io_renewal_replays_after_task_commit_without_old_revision_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    observation = ClockObservation(
        observation_id="a" * 32,
        provider="linux_adjtimex",
        observed_at="2026-09-28T00:00:00Z",
        monotonic_observed_at=time.monotonic(),
        boot_id="test-boot",
        lower_error_seconds=-0.001,
        upper_error_seconds=0.001,
        max_drift_rate=0.0,
        provider_margin_seconds=0.001,
    )
    monkeypatch.setattr(
        "qqtools.plugins.qexp.scheduler.clock_capability",
        lambda *_args: ClockCapability("healthy", "healthy", observation, (observation.provider,)),
    )
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    attempt_file = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    stored_attempt = AttemptRecord.from_dict(read_json(attempt_file))
    stored_attempt.phase = "running"
    stored_attempt.process.update(
        {
            "wrapper_pid": 101,
            "wrapper_start_time_ticks": 202,
            "process_group_id": 303,
            "process_group_start_time_ticks": 404,
        }
    )
    atomic_replace(attempt_file, stored_attempt.to_dict())
    task_file = task_path(cfg.shared_root, task.task_id)
    task_value = read_json(task_file)
    task_value["task"]["state"]["projection"] = "running"
    task_value["task"]["claim_control"]["active_claim"]["launch_state"] = "running"
    atomic_replace(task_file, task_value)
    expected_task_revision = task_value["meta"]["revision"]
    expected_attempt_digest = hashlib.sha256(attempt_file.read_bytes()).hexdigest()
    request_id = "b" * 32
    fence_calls = 0

    def crash_before_attempt_commit() -> None:
        nonlocal fence_calls
        fence_calls += 1
        if fence_calls == 3:
            raise OSError("injected crash after Task renewal")

    parameters = {
        "request_id": request_id,
        "task_id": task.task_id,
        "attempt_id": stored_attempt.attempt_id,
        "attempt_number": stored_attempt.attempt_number,
        "fencing_token": stored_attempt.current_fencing_token,
        "reservation_id": stored_attempt.reservation_id,
        "process_identity": {
            key: stored_attempt.process[key]
            for key in (
                "wrapper_pid",
                "wrapper_start_time_ticks",
                "process_group_id",
                "process_group_start_time_ticks",
            )
        },
        "expected_task_revision": expected_task_revision,
        "expected_attempt_digest": expected_attempt_digest,
    }
    with pytest.raises(OSError, match="injected crash"):
        renew_project_io_attempt_lease(cfg, mutation_fence=crash_before_attempt_commit, **parameters)

    partial_attempt = AttemptRecord.from_dict(read_json(attempt_file))
    assert partial_attempt.lease["clock_evidence"]["observation_id"] != request_id
    assert read_json(task_file)["task"]["claim_control"]["active_claim"]["clock_observation_id"] == request_id
    clock_record = read_json(
        shared_paths(cfg.shared_root)["clock_observations"] / cfg.machine_name / f"{request_id}.json"
    )
    planned_renewed_at = clock_record["renewal_plan"]["renewed_at"]
    task_after_other_update = read_json(task_file)
    task_after_other_update["meta"]["revision"] += 1
    task_after_other_update["meta"]["updated_at"] = "2099-01-01T00:00:00Z"
    atomic_replace(task_file, task_after_other_update)
    atomic_replace(lease_policy_path(cfg), {"lease_policy": {"malformed": True}})

    replayed = renew_project_io_attempt_lease(cfg, mutation_fence=lambda: None, **parameters)

    assert replayed["outcome"] == "renewed"
    renewed_task = read_json(task_file)["task"]
    renewed_attempt = AttemptRecord.from_dict(read_json(attempt_file))
    assert renewed_task["claim_control"]["active_claim"]["clock_observation_id"] == request_id
    assert renewed_attempt.lease["clock_evidence"]["observation_id"] == request_id
    assert renewed_attempt.lease["renewed_at"] == planned_renewed_at
    assert renewed_attempt.lease["renewed_at"] != task_after_other_update["meta"]["updated_at"]
    assert renewed_task["claim_control"]["active_claim"]["lease_expires_at"] == renewed_attempt.lease["expires_at"]
    lease_policy_path(cfg).unlink()

    expired_request_id = "d" * 32
    now = datetime.now(timezone.utc).replace(microsecond=0)
    expired_observation = ClockObservation(
        observation_id=expired_request_id,
        provider="linux_adjtimex",
        observed_at=now.isoformat().replace("+00:00", "Z"),
        monotonic_observed_at=time.monotonic(),
        boot_id="test-boot",
        lower_error_seconds=-0.001,
        upper_error_seconds=0.001,
        max_drift_rate=0.0,
        provider_margin_seconds=0.001,
    )
    atomic_replace(
        shared_paths(cfg.shared_root)["clock_observations"] / cfg.machine_name / f"{expired_request_id}.json",
        {
            "clock_observation": expired_observation.to_dict(),
            "renewal_plan": {
                "renewed_at": (now - timedelta(minutes=2)).isoformat().replace("+00:00", "Z"),
                "lease_expires_at": (now - timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
                "renew_after_seconds": 10.0,
            },
        },
    )
    current_task_bytes = task_file.read_bytes()
    current_attempt_bytes = attempt_file.read_bytes()
    expired = renew_project_io_attempt_lease(
        cfg,
        mutation_fence=lambda: None,
        **{
            **parameters,
            "request_id": expired_request_id,
            "expected_task_revision": read_json(task_file)["meta"]["revision"],
            "expected_attempt_digest": hashlib.sha256(current_attempt_bytes).hexdigest(),
        },
    )
    assert expired["outcome"] == "observed_stale"
    assert expired["reason"] == "renewal_plan_expired"
    assert task_file.read_bytes() == current_task_bytes
    assert attempt_file.read_bytes() == current_attempt_bytes

    stale_request_id = "c" * 32
    stale_observation = ClockObservation(
        observation_id=stale_request_id,
        provider="linux_adjtimex",
        observed_at="2026-09-28T00:00:00Z",
        monotonic_observed_at=time.monotonic(),
        boot_id="old-boot",
        lower_error_seconds=-0.001,
        upper_error_seconds=0.001,
        max_drift_rate=-1.0,
        provider_margin_seconds=0.001,
    )
    atomic_replace(
        shared_paths(cfg.shared_root)["clock_observations"] / cfg.machine_name / f"{stale_request_id}.json",
        {
            "clock_observation": stale_observation.to_dict(),
            "renewal_plan": {
                "renewed_at": "2026-09-28T00:00:00Z",
                "lease_expires_at": "2026-09-28T00:02:00Z",
                "renew_after_seconds": 10.0,
            },
        },
    )
    task_before = task_file.read_bytes()
    attempt_before = attempt_file.read_bytes()
    stale_parameters = {
        **parameters,
        "request_id": stale_request_id,
        "expected_task_revision": read_json(task_file)["meta"]["revision"],
        "expected_attempt_digest": hashlib.sha256(attempt_before).hexdigest(),
    }

    with pytest.raises(ValueError, match="bounds are invalid"):
        renew_project_io_attempt_lease(cfg, mutation_fence=lambda: None, **stale_parameters)

    assert task_file.read_bytes() == task_before
    assert attempt_file.read_bytes() == attempt_before
    valid_but_old = stale_observation.to_dict()
    valid_but_old["max_drift_rate"] = 0.0
    atomic_replace(
        shared_paths(cfg.shared_root)["clock_observations"] / cfg.machine_name / f"{stale_request_id}.json",
        {
            "clock_observation": valid_but_old,
            "renewal_plan": {
                "renewed_at": "2026-09-28T00:00:00Z",
                "lease_expires_at": "2026-09-28T00:02:00Z",
                "renew_after_seconds": 10.0,
            },
        },
    )
    with pytest.raises(RuntimeError, match="no longer current"):
        renew_project_io_attempt_lease(cfg, mutation_fence=lambda: None, **stale_parameters)


def test_local_irreversible_commitment_blocks_recovery(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    decision = create_decision(
        cfg,
        task_id="task",
        attempt_id="attempt",
        fencing_token=7,
        process={"process_group_id": 1, "process_group_start_time_ticks": 2},
        authority_outcome="authority_unavailable",
        reason="lease_authority_unavailable",
    )
    commit_local_unavailable(cfg, "attempt", decision["decision_id"])
    assert is_recovery_blocked(cfg, "attempt")
    commit_signal(cfg, "attempt", decision["decision_id"])
    assert is_recovery_blocked(cfg, "attempt")


def test_unqualified_clock_creates_local_safe_claim_and_blocks_expiry(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    monkeypatch.setattr(
        "qqtools.plugins.qexp.scheduler.clock_capability",
        lambda *_args: ClockCapability("unavailable", "no_qualified_provider"),
    )
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert attempt.authority_mode == "holder_bound"
    renewal = renew_attempt_lease(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert renewal.outcome is LeaseRenewalOutcome.NOT_REQUIRED
    assert not expire_claim(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
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
        cfg,
        task_id=task.task_id,
        attempt_id=attempt.attempt_id,
        fencing_token=attempt.current_fencing_token,
        process={"process_group_id": 1, "process_group_start_time_ticks": 2},
        authority_outcome="termination_required",
        reason="fenced",
    )
    assert commit_shared_termination(
        cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, decision["decision_id"]
    )
    atomic_replace(
        cfg.runtime_root / "processes" / f"{attempt.attempt_id}.json",
        {
            "process": {
                "task_id": task.task_id,
                "attempt_id": attempt.attempt_id,
                "fencing_token": attempt.current_fencing_token,
                "process_group_id": 1,
            }
        },
    )
    monkeypatch.setattr(
        "qqtools.plugins.qexp.scheduler.inspect_group_identity",
        lambda *_args: ProcessEvidence(state="unknown", reason="read_failed"),
    )
    monkeypatch.setattr("qqtools.plugins.qexp.runtime.termination._matches_process_group", lambda *_args: False)
    reconcile_running_tasks(cfg)
    stored = read_json(
        cfg.runtime_root / "termination-decisions" / attempt.attempt_id / f"{decision['decision_id']}.json"
    )["termination_decision"]
    assert stored["state"] == "confirmed"
    assert stored["shared_commitment"] == "committed"


def test_sigkill_is_not_confirmed_until_process_identity_is_absent(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    decision = create_decision(
        cfg,
        task_id="task",
        attempt_id="attempt",
        fencing_token=7,
        process={"process_group_id": 1, "process_group_start_time_ticks": 2},
        authority_outcome="termination_required",
        reason="fenced",
    )
    commit_signal(cfg, "attempt", decision["decision_id"])
    monkeypatch.setattr("qqtools.plugins.qexp.runtime.termination._matches_process_group", lambda *_args: True)
    monkeypatch.setattr("qqtools.plugins.qexp.runtime.termination.os.killpg", lambda *_args: None)
    current = send_signals(cfg, "attempt", decision["decision_id"], grace_seconds=0)
    assert current["state"] == "sigkill_sent"
    monkeypatch.setattr("qqtools.plugins.qexp.runtime.termination._matches_process_group", lambda *_args: False)
    assert send_signals(cfg, "attempt", decision["decision_id"], grace_seconds=0)["state"] == "confirmed"


def test_termination_decision_rejects_state_rollback_and_wait_confirmation(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    decision = create_decision(
        cfg,
        task_id="task",
        attempt_id="attempt",
        fencing_token=7,
        process={"process_group_id": 1, "process_group_start_time_ticks": 2},
        authority_outcome="termination_required",
        reason="fenced",
    )
    commit_signal(cfg, "attempt", decision["decision_id"])
    with pytest.raises(RuntimeError, match="confirmation requires absent process identity"):
        update_decision(cfg, "attempt", decision["decision_id"], state="confirmed", confirmation="runner_waited")
    with pytest.raises(RuntimeError, match="not monotonic"):
        update_decision(cfg, "attempt", decision["decision_id"], state="pending")


def test_termination_confirmation_uses_real_process_group_identity(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    child = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        decision = create_decision(
            cfg,
            task_id="task",
            attempt_id="attempt",
            fencing_token=7,
            process={
                "process_group_id": child.pid,
                "process_group_start_time_ticks": _process_start_time_ticks(child.pid),
            },
            authority_outcome="termination_required",
            reason="fenced",
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
    create_group(cfg, "exp")
    save_lease_policy(cfg, LeasePolicy(ttl_seconds=180))
    task = submit(cfg, ["echo", "ok"], group="exp", sharing_mode="spillover")
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert expire_claim(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    manifest_path = cfg.runtime_root / "processes" / f"{attempt.attempt_id}.json"
    atomic_replace(
        manifest_path,
        {
            "process": {
                "task_id": task.task_id,
                "attempt_id": attempt.attempt_id,
                "fencing_token": attempt.current_fencing_token,
            }
        },
    )
    token = recover_running_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert token == attempt.current_fencing_token + 1
    recovered = read_json(attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number))["attempt"]
    expires_at = recovered["lease"]["expires_at"]
    from datetime import datetime, timezone

    from qqtools.plugins.qexp.lease import parse_utc

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
    upgraded = load_root_config(cfg.shared_root, "g1", cfg.runtime_root, require_initialized=False)
    with pytest.raises(RuntimeError, match="requires cpu-lane-v1"):
        qexp_layout.validate_root_contract(upgraded)
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


def test_schema5_migration_recovers_after_source_is_parked(tmp_path: Path, monkeypatch):
    import qqtools.plugins.qexp.layout as layout

    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    task_file = cfg.shared_root / "tasks" / f"{task.task_id}.json"
    legacy_task = read_json(task_file)
    legacy_task["meta"]["schema_version"] = 5
    legacy_task["task"]["control"].pop("cleanup_operation_id")
    legacy_task["task"]["control"].pop("cleanup_state")
    legacy_task["task"]["placement_runtime"].pop("offer_clock_evidence")
    atomic_replace(task_file, legacy_task)
    schema = cfg.shared_root / "schema" / "version.json"
    value = read_json(schema)
    value["schema"].update({"version": 5, "minimum_reader_version": 5})
    atomic_replace(schema, value)

    original_rename = layout.os.rename
    is_crash_injected = False

    def crash_after_parking(source, destination):
        nonlocal is_crash_injected
        if source.name == ".qexp" and ".schema6-stage-" in str(source.parent):
            is_crash_injected = True
            raise OSError("simulated crash before staged-root promotion")
        return original_rename(source, destination)

    monkeypatch.setattr(layout.os, "rename", crash_after_parking)
    with pytest.raises(OSError, match="simulated crash"):
        migrate_schema5_to_schema6(cfg)
    assert is_crash_injected
    assert not cfg.shared_root.exists()

    monkeypatch.setattr(layout.os, "rename", original_rename)
    migrate_schema5_to_schema6(cfg)
    upgraded = load_root_config(cfg.shared_root, "g1", cfg.runtime_root, require_initialized=False)
    with pytest.raises(RuntimeError, match="requires cpu-lane-v1"):
        qexp_layout.validate_root_contract(upgraded)
    restored = read_json(upgraded.shared_root / "tasks" / f"{task.task_id}.json")
    assert restored["meta"]["schema_version"] == 6
    assert restored["task"]["control"]["cleanup_operation_id"] is None
    backup = next(tmp_path.glob(".qexp.schema5-backup-*"))
    original = read_json(backup / "tasks" / f"{task.task_id}.json")
    assert original == legacy_task


def test_bounded_signal_progress_preserves_grace_and_restarts_conservatively(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from qqtools.plugins.qexp.runtime import termination

    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    decision = create_decision(
        cfg,
        task_id="task",
        attempt_id="attempt",
        fencing_token=1,
        process={"process_group_id": 12345678, "process_group_start_time_ticks": 42},
        authority_outcome="termination_required",
        reason="cancelled",
    )
    commit_signal(cfg, "attempt", decision["decision_id"])
    now = [0.0]
    is_alive = [True]
    signals = []
    monkeypatch.setattr(termination, "_matches_process_group", lambda *_args: is_alive[0])
    monkeypatch.setattr(termination, "os", SimpleNamespace(killpg=lambda pgid, sig: signals.append((pgid, sig))))

    def forbidden_sleep(_seconds):
        pytest.fail("bounded termination waited under the Attempt lock")

    monkeypatch.setattr(termination, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=forbidden_sleep))

    def advance(deadline):
        with termination.attempt_control_lock(cfg, "attempt"):
            return termination.advance_signals(
                cfg, "attempt", decision["decision_id"], sigterm_deadline=deadline, grace_seconds=5
            )

    first, deadline = advance(None)
    assert first["state"] == "sigterm_sent"
    assert deadline == 5
    assert len(signals) == 1
    now[0] = 4.9
    waiting, deadline = advance(deadline)
    assert waiting["state"] == "sigterm_sent"
    assert len(signals) == 1
    # A restart after the original grace elapsed still grants a full new grace.
    now[0] = 20
    waiting, deadline = advance(None)
    assert deadline == 25
    assert len(signals) == 1
    now[0] = 24.9
    waiting, deadline = advance(deadline)
    assert waiting["state"] == "sigterm_sent"
    assert len(signals) == 1
    now[0] = 25
    killed, deadline = advance(deadline)
    assert killed["state"] == "sigkill_sent"
    assert len(signals) == 2
    assert [item[1] for item in signals] == [signal.SIGTERM, signal.SIGKILL]
    assert deadline is None
    is_alive[0] = False
    confirmed, deadline = advance(None)
    assert confirmed["state"] == "confirmed"
    assert confirmed["confirmation"] == "identity_absent"
    assert advance(None)[0]["state"] == "confirmed"
    assert len(signals) == 2
