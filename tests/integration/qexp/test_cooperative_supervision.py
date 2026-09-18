"""Ordinary supervision bounds termination inventory without losing its fence."""

import json

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.authority import AuthoritySupervisor
from qqtools.plugins.qexp.runtime.paths import local_paths
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task
from qqtools.plugins.qexp.runtime.termination import attempt_control_lock, create_decision, update_decision
from qqtools.plugins.qexp.runtime.work_budget import activate_diagnostics
from qqtools.plugins.qexp.scheduler import authorize_launch, cancel_task, claim_task, expire_claim

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _process(cfg, *, history=17, gpu=0):
    task = submit(cfg, ["echo", "supervision"])
    attempt = claim_task(cfg, task.task_id, [gpu])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    process = {
        "protocol_version": 1,
        "task_id": task.task_id,
        "attempt_id": attempt.attempt_id,
        "fencing_token": attempt.current_fencing_token,
        "authority_mode": attempt.authority_mode,
        "observed_state": "running",
    }
    atomic_replace(local_paths(cfg.runtime_root)["processes"] / f"{attempt.attempt_id}.json", {"process": process})
    for index in range(history):
        create_decision(
            cfg,
            task_id=task.task_id,
            attempt_id=attempt.attempt_id,
            fencing_token=attempt.current_fencing_token,
            process=process,
            authority_outcome="test_pending",
            reason="test_pending",
            decision_id=f"pending-{index:03}",
        )
    return task, attempt, process


def _setup(tmp_path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    supervisor = AuthoritySupervisor(cfg, work_limit=1)
    supervisor.recover_startup()
    return cfg, supervisor


def test_renewal_waits_for_bounded_negative_proof_under_lock(tmp_path, monkeypatch):
    cfg, supervisor = _setup(tmp_path)
    task, attempt, process = _process(cfg)
    revision = load_task(cfg, task.task_id).meta["revision"]
    work = supervisor._work
    original = supervisor._renew_or_isolate

    def renew(*args, **kwargs):
        with attempt_control_lock(cfg, attempt.attempt_id, blocking=False) as acquired:
            assert not acquired
        return original(*args, **kwargs)

    monkeypatch.setattr(supervisor, "_renew_or_isolate", renew)
    monkeypatch.setattr("qqtools.plugins.qexp.authority.iter_json", lambda *_args: pytest.fail("eager inventory"))
    try:
        with activate_diagnostics(work.diagnostics):
            supervisor._supervise(process)
        for entries in (8, 16):
            assert work.diagnostics.counters["store.inventory_entries"] == entries
            assert load_task(cfg, task.task_id).meta["revision"] == revision
            with attempt_control_lock(cfg, attempt.attempt_id, blocking=False) as acquired:
                assert not acquired
            work.tick(1)
        assert work.diagnostics.counters["store.inventory_entries"] == 17
        assert load_task(cfg, task.task_id).meta["revision"] == revision + 1
        assert work.snapshot()["pending_control_attempt"] is None
        with attempt_control_lock(cfg, attempt.attempt_id, blocking=False) as acquired:
            assert acquired
    finally:
        supervisor.close()


@pytest.mark.parametrize("mutation", ["manifest", "orphan", "close"])
def test_delayed_supervision_revalidates_identity_and_releases_lock(tmp_path, mutation):
    cfg, supervisor = _setup(tmp_path)
    task, attempt, process = _process(cfg)
    try:
        supervisor._supervise(process)
        if mutation == "manifest":
            changed = dict(process, fencing_token=process["fencing_token"] + 1)
            atomic_replace(
                local_paths(cfg.runtime_root)["processes"] / f"{attempt.attempt_id}.json", {"process": changed}
            )
        elif mutation == "orphan":
            assert expire_claim(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
        revision = load_task(cfg, task.task_id).meta["revision"]
        if mutation == "close":
            supervisor.cancel_pending_control()
        else:
            supervisor._work.tick(1)
            supervisor._work.tick(1)
        assert load_task(cfg, task.task_id).meta["revision"] == revision
        with attempt_control_lock(cfg, attempt.attempt_id, blocking=False) as acquired:
            assert acquired
        assert supervisor._work.snapshot()["pending_control_attempt"] is None
    finally:
        supervisor.close()


def test_short_inventory_keeps_other_attempt_renewing_while_long_scan_waits(tmp_path):
    cfg, supervisor = _setup(tmp_path)
    _task, attempt, process = _process(cfg)
    other, _other_attempt, other_process = _process(cfg, history=0, gpu=1)
    revision = load_task(cfg, other.task_id).meta["revision"]
    try:
        supervisor._supervise(process)
        supervisor._supervise(other_process)
        assert load_task(cfg, other.task_id).meta["revision"] == revision + 1
        assert supervisor._work.snapshot()["pending_control_attempt"] == attempt.attempt_id
        with attempt_control_lock(cfg, attempt.attempt_id, blocking=False) as acquired:
            assert not acquired
    finally:
        supervisor.close()


def test_exit_waits_for_termination_proof_and_hook_runs_without_control_lock(tmp_path, monkeypatch):
    cfg, supervisor = _setup(tmp_path)
    task, attempt, process = _process(cfg)
    hooks = []

    def hook(*_args):
        with attempt_control_lock(cfg, attempt.attempt_id, blocking=False) as acquired:
            assert acquired
        hooks.append("terminal")

    monkeypatch.setattr("qqtools.plugins.qexp.authority.dispatch_task_lifecycle_hooks_noexcept", hook)
    try:
        supervisor._supervise(process)
        observation = local_paths(cfg.runtime_root)["observations"] / f"{attempt.attempt_id}.json"
        atomic_replace(observation, {"exit_observation": {"attempt_id": attempt.attempt_id, "observed_exit_code": 0}})
        supervisor._work._observation(observation)
        assert load_task(cfg, task.task_id).state["projection"] != "succeeded"
        assert supervisor._work.snapshot()["pending_control_kind"] == "supervision"
        supervisor._work.tick(1)
        supervisor._work.tick(1)
        assert load_task(cfg, task.task_id).state["projection"] == "succeeded"
        assert hooks == ["terminal"]
    finally:
        supervisor.close()


@pytest.mark.parametrize("commitment", ["committed", "unavailable"])
def test_pending_irreversible_commitment_prevents_renewal_and_completes_signal(tmp_path, monkeypatch, commitment):
    cfg, supervisor = _setup(tmp_path)
    task, attempt, process = _process(cfg, history=1)
    update_decision(cfg, attempt.attempt_id, "pending-000", shared_commitment=commitment)
    monkeypatch.setattr(supervisor, "_renew_or_isolate", lambda *_args, **_kwargs: pytest.fail("renewed commitment"))
    try:
        supervisor._supervise(process)
        path = local_paths(cfg.runtime_root)["termination_decisions"] / attempt.attempt_id / "pending-000.json"
        # No process identity is present, so signal advancement confirms absence.
        # Finalization may have cleaned the decision once terminal truth is durable.
        if path.exists():
            assert read_json(path)["termination_decision"]["state"] == "confirmed"
        assert load_task(cfg, task.task_id).state["projection"] == "failed"
    finally:
        supervisor.close()


def test_cancel_during_inventory_commits_termination_without_reentering_lock(tmp_path):
    cfg, supervisor = _setup(tmp_path)
    task, attempt, process = _process(cfg)
    try:
        supervisor._supervise(process)
        cancel_task(cfg, task.task_id, terminate_running=True)
        supervisor._work.tick(1)
        supervisor._work.tick(1)
        claim = load_task(cfg, task.task_id).claim_control["active_claim"]
        assert claim["termination_decision_id"]
        with attempt_control_lock(cfg, attempt.attempt_id, blocking=False) as acquired:
            assert acquired
    finally:
        supervisor.close()


def test_second_long_inventory_does_not_retain_a_second_lock(tmp_path):
    cfg, supervisor = _setup(tmp_path)
    _task, attempt, process = _process(cfg)
    other, other_attempt, other_process = _process(cfg, gpu=1)
    revision = load_task(cfg, other.task_id).meta["revision"]
    try:
        supervisor._supervise(process)
        supervisor._supervise(other_process)
        assert supervisor._work.snapshot()["pending_control_attempt"] == attempt.attempt_id
        assert load_task(cfg, other.task_id).meta["revision"] == revision
        with attempt_control_lock(cfg, other_attempt.attempt_id, blocking=False) as acquired:
            assert acquired
        supervisor._work.tick(1)
        supervisor._work.tick(1)
        supervisor._supervise(other_process)
        supervisor._work.tick(1)
        supervisor._work.tick(1)
        assert load_task(cfg, other.task_id).meta["revision"] == revision + 1
    finally:
        supervisor.close()


@pytest.mark.parametrize(
    "damaged",
    [[], {"process": []}, {"process": None}, "task_id", "attempt_id", "fencing_token", "read_error", "write_error"],
)
def test_corrupt_manifest_after_terminal_cas_preserves_hooks_and_other_service(tmp_path, monkeypatch, damaged):
    from qqtools.plugins.qexp import authority

    cfg, supervisor = _setup(tmp_path)
    task, attempt, process = _process(cfg, history=1)
    other, _other_attempt, other_process = _process(cfg, history=0, gpu=1)
    other_revision = load_task(cfg, other.task_id).meta["revision"]
    paths = local_paths(cfg.runtime_root)
    manifest = paths["processes"] / f"{attempt.attempt_id}.json"
    atomic_replace(
        paths["observations"] / f"{attempt.attempt_id}.json",
        {"exit_observation": {"attempt_id": attempt.attempt_id, "observed_exit_code": 0}},
    )
    registration = paths["registrations"] / f"{attempt.attempt_id}.json"
    atomic_replace(registration, {"process_registration": dict(process)})
    decisions = list((paths["termination_decisions"] / attempt.attempt_id).glob("*.json"))
    original = authority.commit_terminal_transition_locked
    hooks = []
    expected = damaged

    def commit(*args, **kwargs):
        nonlocal expected
        result = original(*args, **kwargs)
        assert result.outcome == "committed"
        if isinstance(damaged, str):
            expected = {"process": dict(process)}
            if damaged in {"read_error", "write_error"}:
                operation = "read_json" if damaged == "read_error" else "atomic_replace"
                original_operation = getattr(authority, operation)

                def unavailable(path, *args, **kwargs):
                    if path == manifest:
                        raise PermissionError("injected manifest unavailable")
                    return original_operation(path, *args, **kwargs)

                monkeypatch.setattr(authority, operation, unavailable)
            else:
                expected["process"][damaged] = "another-identity"
        atomic_replace(manifest, expected)
        return result

    monkeypatch.setattr(authority, "commit_terminal_transition_locked", commit)
    monkeypatch.setattr(authority, "dispatch_task_lifecycle_hooks_noexcept", lambda *_args: hooks.append("terminal"))
    try:
        supervisor._supervise(process)
        assert load_task(cfg, task.task_id).state["projection"] == "succeeded"
        assert hooks == ["terminal"]
        assert json.loads(manifest.read_text()) == expected
        diagnostic = read_json(paths["authority_diagnostics"] / f"{attempt.attempt_id}.json")
        assert diagnostic["authority_diagnostic"]["reason"] == "terminal_manifest_unreadable"
        if damaged != "write_error":
            # Later active and cleanup turns must not erase identity-mismatched
            # or unreadable evidence simply because shared truth is terminal.
            for _ in range(3):
                supervisor.tick_bounded(64)
            retained = json.loads(manifest.read_text())
            if isinstance(expected, dict) and isinstance(expected.get("process"), dict):
                # A failed read of the foreign Task may add advisory health state;
                # every original identity/evidence field must remain untouched.
                assert retained["process"].items() >= expected["process"].items()
            else:
                assert retained == expected
            assert registration.exists()
            assert (paths["observations"] / f"{attempt.attempt_id}.json").exists()
            assert all(path.exists() for path in decisions)
        supervisor._supervise(other_process)
        assert load_task(cfg, other.task_id).meta["revision"] == other_revision + 1
        with attempt_control_lock(cfg, attempt.attempt_id, blocking=False) as acquired:
            assert acquired
    finally:
        supervisor.close()
