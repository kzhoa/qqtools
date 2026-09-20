"""Paged recovery retains its negative proof without monopolizing project service."""

import json

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.authority import AuthoritySupervisor
from qqtools.plugins.qexp.lease import ClockCapability
from qqtools.plugins.qexp.runtime.attempt_recovery import recovery_steps
from qqtools.plugins.qexp.runtime.paths import local_paths
from qqtools.plugins.qexp.runtime.process_evidence import ProcessEvidence
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task
from qqtools.plugins.qexp.runtime.termination import attempt_control_lock
from qqtools.plugins.qexp.runtime.work_budget import RuntimeDiagnostics, activate_diagnostics
from qqtools.plugins.qexp.scheduler import authorize_launch, claim_task, expire_claim

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _orphan(tmp_path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    task = submit(cfg, ["echo", "recovery"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert attempt.authority_mode == "bounded_lease"
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    process = {
        "protocol_version": 1,
        "task_id": task.task_id,
        "attempt_id": attempt.attempt_id,
        "fencing_token": attempt.current_fencing_token,
        "observed_state": "running",
    }
    paths = local_paths(cfg.runtime_root)
    atomic_replace(paths["processes"] / f"{attempt.attempt_id}.json", {"process": process})
    assert expire_claim(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    directory = paths["termination_decisions"] / attempt.attempt_id
    for index in range(17):
        atomic_replace(directory / f"{index}.json", {"termination_decision": {"state": "superseded"}})
    monkeypatch.setattr(
        "qqtools.plugins.qexp.runtime.attempt_recovery.inspect_group_identity",
        lambda *_args: ProcessEvidence(state="alive"),
    )
    return cfg, task, attempt, process


def test_recovery_pages_hold_lock_until_fenced_commit(tmp_path, monkeypatch):
    cfg, task, attempt, _process = _orphan(tmp_path, monkeypatch)
    steps = recovery_steps(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    diagnostics = RuntimeDiagnostics()
    try:
        with activate_diagnostics(diagnostics):
            for page in range(2):
                assert next(steps) is None
                assert diagnostics.counters["store.inventory_entries"] == (page + 1) * 8
                assert load_task(cfg, task.task_id).state["projection"] == "blocked"
                with attempt_control_lock(cfg, attempt.attempt_id, blocking=False) as acquired:
                    assert not acquired
            with pytest.raises(StopIteration) as finished:
                next(steps)
        assert finished.value.value == attempt.current_fencing_token + 1
        assert diagnostics.counters["store.inventory_entries"] == 17
        current = load_task(cfg, task.task_id)
        assert current.state["projection"] == "running"
        assert current.claim_control["active_claim"]["fencing_token"] == finished.value.value
    finally:
        steps.close()
    with attempt_control_lock(cfg, attempt.attempt_id, blocking=False) as acquired:
        assert acquired


@pytest.mark.parametrize("change", ["clock", "process", "manifest", "exit", "close"])
def test_recovery_revalidates_after_yield_and_releases_lock(tmp_path, monkeypatch, change):
    cfg, task, attempt, process = _orphan(tmp_path, monkeypatch)
    steps = recovery_steps(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    try:
        next(steps)
        if change == "clock":
            monkeypatch.setattr(
                "qqtools.plugins.qexp.runtime.attempt_recovery.clock_capability",
                lambda *_args: ClockCapability("unavailable", "test_clock_lost"),
            )
        elif change == "process":
            monkeypatch.setattr(
                "qqtools.plugins.qexp.runtime.attempt_recovery.inspect_group_identity",
                lambda *_args: ProcessEvidence(state="absent"),
            )
        elif change == "manifest":
            process["fencing_token"] += 1
            atomic_replace(
                local_paths(cfg.runtime_root)["processes"] / f"{attempt.attempt_id}.json", {"process": process}
            )
        elif change == "exit":
            atomic_replace(local_paths(cfg.runtime_root)["observations"] / f"{attempt.attempt_id}.json", {})
        if change != "close":
            next(steps)
            with pytest.raises(StopIteration) as finished:
                next(steps)
            assert finished.value.value is None
    finally:
        steps.close()
    assert load_task(cfg, task.task_id).state["projection"] == "blocked"
    with attempt_control_lock(cfg, attempt.attempt_id, blocking=False) as acquired:
        assert acquired


def test_busy_recovery_lock_does_not_wait(tmp_path, monkeypatch):
    cfg, task, attempt, _process = _orphan(tmp_path, monkeypatch)
    with attempt_control_lock(cfg, attempt.attempt_id):
        steps = recovery_steps(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
        with pytest.raises(StopIteration) as finished:
            next(steps)
        assert finished.value.value is None
    assert load_task(cfg, task.task_id).state["projection"] == "blocked"


def test_supervisor_services_other_lanes_and_cancels_pending_recovery(tmp_path, monkeypatch):
    cfg, _task, attempt, process = _orphan(tmp_path, monkeypatch)
    supervisor = AuthoritySupervisor(cfg, work_limit=64)
    supervisor.recover_startup()
    work = supervisor._work
    work.recover(process)
    # A second request cannot replace the cursor or grow a recovery queue.
    work.recover({"attempt_id": "other"})
    try:
        work.tick(1)
        with attempt_control_lock(cfg, attempt.attempt_id, blocking=False) as acquired:
            assert not acquired
        assert not work.is_startup_complete
        calls = []
        monkeypatch.setattr(supervisor, "_materialize_registrations", lambda **_kwargs: calls.append("registration"))
        monkeypatch.setattr(supervisor, "_supervise", lambda _process: calls.append("supervision"))
        # These paths must not try to reacquire the pending Attempt's lock.
        path = local_paths(cfg.runtime_root)["processes"] / f"{attempt.attempt_id}.json"
        work._process(path)
        work._registration(path)
        assert calls == []
        other = path.with_name("other.json")
        atomic_replace(other, {"process": {"protocol_version": 1, "attempt_id": "other"}})
        work._process(other)
        assert calls == ["supervision"]
        supervisor.cancel_pending_control()
        with attempt_control_lock(cfg, attempt.attempt_id, blocking=False) as acquired:
            assert acquired
        assert work.snapshot()["pending_control_attempt"] is None
        assert read_json(path)["process"]["fencing_token"] == attempt.current_fencing_token
    finally:
        supervisor.close()


@pytest.mark.parametrize("record", [[], {"process": []}, {"process": None}])
def test_malformed_manifest_releases_pending_lock_and_retains_evidence(tmp_path, monkeypatch, record):
    cfg, task, attempt, process = _orphan(tmp_path, monkeypatch)
    supervisor = AuthoritySupervisor(cfg, work_limit=1)
    supervisor.recover_startup()
    work = supervisor._work
    work.recover(process)
    try:
        work.tick(1)
        path = local_paths(cfg.runtime_root)["processes"] / f"{attempt.attempt_id}.json"
        atomic_replace(path, record)
        work.tick(1)
        work.tick(1)
        assert work.snapshot()["pending_control_attempt"] is None
        assert json.loads(path.read_text()) == record
        assert load_task(cfg, task.task_id).state["projection"] == "blocked"
        with attempt_control_lock(cfg, attempt.attempt_id, blocking=False) as acquired:
            assert acquired
    finally:
        supervisor.close()


def test_alias_registration_cannot_reenter_pending_recovery_lock(tmp_path, monkeypatch):
    cfg, _task, attempt, process = _orphan(tmp_path, monkeypatch)
    supervisor = AuthoritySupervisor(cfg, work_limit=1)
    supervisor.recover_startup()
    work = supervisor._work
    work.recover(process)
    try:
        work.tick(1)
        path = local_paths(cfg.runtime_root)["registrations"] / "alias.json"
        atomic_replace(path, {"process_registration": process})
        monkeypatch.setattr(
            supervisor, "_publish_running", lambda *_args: pytest.fail("pending Attempt lock would be reacquired")
        )
        work._registration(path)
        with attempt_control_lock(cfg, attempt.attempt_id, blocking=False) as acquired:
            assert not acquired
    finally:
        supervisor.close()


def test_missing_termination_identity_is_diagnosed_without_pending_recovery(tmp_path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    path = local_paths(cfg.runtime_root)["termination_decisions"] / "attempt" / "broken.json"
    atomic_replace(path, {"termination_decision": {"state": "signal_committed", "decision_id": "broken"}})
    supervisor = AuthoritySupervisor(cfg, work_limit=64)
    try:
        supervisor.tick()
        snapshot = supervisor.work_snapshot
        assert snapshot["pending_control_attempt"] is None
        assert snapshot["lanes"]["termination"]["failures"] > 0
        diagnostic = read_json(local_paths(cfg.runtime_root)["authority_diagnostics"] / "control-plane.json")
        assert diagnostic["authority_diagnostic"]["reason"] == "termination_unavailable"
        assert path.exists()
    finally:
        supervisor.close()
