"""Lifecycle coordination separates productive steps from external waits."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent import dispatch_loop, recovery_enrollment
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.recovery_capture import RecoveryProgress
from qqtools.plugins.qexp.agent.recovery_enrollment import RecoveryEnrollment
from qqtools.plugins.qexp.runtime.group_discovery import service
from qqtools.plugins.qexp.runtime.store import read_json

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def register(tmp_path, runtime, name):
    cfg = init_shared_root(tmp_path / name / ".qexp", "gpu-1", runtime_root=tmp_path / f"legacy-{name}")
    return cfg, runtime.add_binding(cfg.shared_root, cfg.machine_name)


@pytest.mark.parametrize("has_rollback", [False, True])
def test_dispatch_allows_registration_but_excludes_migration_and_rollback(tmp_path, monkeypatch, has_rollback):
    runtime = MachineRuntime(tmp_path / "machine")
    cfg, binding = register(tmp_path, runtime, "project")
    path = cfg.shared_root / "machines/gpu-1/registration.json"
    if has_rollback:
        revision, bindings = runtime.load_registry()
        runtime._save_registration_transaction(
            revision=revision, bindings=bindings, registrations=[(cfg, read_json(path))], machine_records=[]
        )
    entered, release = Event(), Event()

    def dispatch(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return []

    monkeypatch.setattr(dispatch_loop, "_dispatch_machine_cycle_locked", dispatch)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(dispatch_loop.dispatch_machine_cycle, runtime, available_gpus=[])
        try:
            assert entered.wait(5)
            with runtime.migration_guard(blocking=False) as acquired:
                assert not acquired
            assert runtime.prepare_recovery_registration(binding) is not has_rollback
            assert read_json(path)["registration"]["version"] == (1 if has_rollback else 2)
            assert runtime.paths["registration_transaction"].exists() is has_rollback
        finally:
            release.set()
        assert future.result(timeout=5) == []
    with runtime.scheduler_authority() as acquired:
        assert acquired
        assert runtime.prepare_recovery_registration(binding)
    assert read_json(path)["registration"]["version"] == 2
    assert not runtime.paths["registration_transaction"].exists()


def test_productive_binding_continues_while_waiting_binding_observes_own_backoff(tmp_path, monkeypatch):
    runtime = MachineRuntime(tmp_path / "machine")
    _, waiting = register(tmp_path, runtime, "waiting")
    _, advancing = register(tmp_path, runtime, "advancing")
    now = [10.0]
    monkeypatch.setattr(recovery_enrollment, "time", SimpleNamespace(monotonic=lambda: now[0]))
    enrollment = RecoveryEnrollment(runtime)
    calls = []
    complete = False

    def advance(binding, *, should_prepare_only):
        calls.append(binding)
        if complete:
            return RecoveryProgress.COMPLETE
        return RecoveryProgress.WAITING if binding == waiting else RecoveryProgress.ADVANCED

    monkeypatch.setattr(enrollment, "_advance_binding", advance)

    def run_pass():
        enrollment._advance_pass()
        enrollment._refresh_pending()

    try:
        run_pass()
        assert len(calls) == 2 and set(calls) == {waiting, advancing}
        run_pass()
        assert len(calls) == 3 and calls[-1] == advancing
        assert runtime.recovery_enrollment_pending_projects == {waiting.project_id, advancing.project_id}
        complete = True
        run_pass()
        assert enrollment._settled == {advancing}
        run_pass()
        assert enrollment._thread is None
        now[0] += recovery_enrollment.PASS_INTERVAL_SECONDS
        run_pass()
        assert enrollment._settled == {waiting, advancing}
        assert not runtime.recovery_enrollment_pending_projects
    finally:
        enrollment.stop()


def test_group_discovery_cooldown_does_not_recheck_shared_authority(tmp_path, monkeypatch):
    runtime = MachineRuntime(tmp_path / "machine")
    _, binding = register(tmp_path, runtime, "project")
    worker = service.MachineGroupDiscoveryWorker(runtime)
    checks = []
    monkeypatch.setattr(worker, "_eligible_binding", lambda candidate: checks.append(candidate) or False)
    monkeypatch.setattr(service, "time", SimpleNamespace(monotonic=lambda: 10.0))
    key = worker._binding_key(binding)
    assert worker._next_group([binding], {}, {key: 11.0}, 0) is None
    assert not checks
    assert worker._next_group([binding], {}, {key: 10.0}, 0) is None
    assert checks == [binding]


def test_slow_capture_shutdown_retains_scheduler_authority_until_worker_stops(tmp_path, monkeypatch):
    runtime = MachineRuntime(tmp_path / "machine")
    register(tmp_path, runtime, "project")
    enrollment = RecoveryEnrollment(runtime)
    entered, release, stopping, stopped = Event(), Event(), Event(), Event()

    def capture_step(capture):
        entered.set()
        assert release.wait(10)
        return RecoveryProgress.WAITING

    monkeypatch.setattr(recovery_enrollment.RecoveryCapture, "advance_step", capture_step)

    def own_agent():
        with runtime.scheduler_authority() as acquired:
            assert acquired
            enrollment.poll()
            assert entered.wait(5)
            stopping.set()
            enrollment.stop()
        stopped.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        owner = pool.submit(own_agent)
        try:
            assert stopping.wait(5)
            # Exercise the former two-second join timeout while capture is held
            # at an explicit signal. A successor must remain excluded throughout.
            assert not stopped.wait(2.2)
            successor = MachineRuntime(runtime.root)
            with successor.scheduler_authority() as acquired:
                assert not acquired
        finally:
            release.set()
        owner.result(timeout=5)
    assert stopped.is_set()
    assert not enrollment._thread.is_alive()
    with MachineRuntime(runtime.root).scheduler_authority() as acquired:
        assert acquired


def test_binding_config_observes_new_runtime_and_revalidates_root_each_call(tmp_path):
    from qqtools.plugins.qexp.layout import load_machine_record, save_machine_record

    runtime = MachineRuntime(tmp_path / "machine")
    cfg, binding = register(tmp_path, runtime, "project")
    assert binding.root_config().runtime_root == cfg.runtime_root
    machine = load_machine_record(cfg)
    changed = tmp_path / "changed-runtime"
    machine["machine"]["runtime_root"] = str(changed)
    save_machine_record(cfg, machine)
    assert binding.root_config().runtime_root == changed
    (cfg.shared_root / "project").rename(cfg.shared_root / "missing-project")
    with pytest.raises(RuntimeError, match="incomplete"):
        binding.root_config()
