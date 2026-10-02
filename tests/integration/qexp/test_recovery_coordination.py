"""Lifecycle coordination separates productive steps from external waits."""

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent import dispatch_loop
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.project_io_controller import ProjectIOController
from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor
from qqtools.plugins.qexp.agent.recovery_enrollment import RecoveryEnrollment
from qqtools.plugins.qexp.runtime.group_discovery import service
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from tests.helpers.qexp.lifecycle import wait_until

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


@pytest.mark.parametrize("has_scheduler_services", [False, True])
def test_productive_binding_continues_while_waiting_binding_observes_own_backoff(
    tmp_path, monkeypatch, has_scheduler_services
):
    runtime = MachineRuntime(tmp_path / "machine")
    _, waiting = register(tmp_path, runtime, "waiting")
    _, advancing = register(tmp_path, runtime, "advancing")
    now = [10.0]
    atomic_replace(runtime.migration_path(waiting.project_id), {"migration": {"state": "prepared"}})
    enrollment = RecoveryEnrollment(runtime)
    executor = ProjectIOExecutor(runtime)
    controller = ProjectIOController(runtime, executor, monotonic=lambda: now[0])
    starts = []
    start = executor.start

    def observe_start(request_id, **kwargs):
        request = executor._load_request(request_id)
        if request.project_id == waiting.project_id and request.operation_kind == "recovery_admission":
            starts.append(request_id)
        return start(request_id, **kwargs)

    monkeypatch.setattr(executor, "start", observe_start)

    def advance():
        revision, bindings = runtime.load_registry_snapshot()
        with controller.admission_turn():
            if has_scheduler_services:
                controller.advance_scheduler_observations([advancing], revision, lane="gpu", admission_role="primary")
                controller.advance_scheduler_claims(
                    [advancing],
                    revision,
                    lane="gpu",
                    admission_role="primary",
                    observations={},
                    available_gpu_ids=[],
                    available_cpu_slots=0,
                )
            enrollment.advance(controller, bindings, revision)

    def healthy_completed():
        advance()
        return advancing in enrollment._settled

    def all_completed():
        advance()
        return not runtime.recovery_enrollment_pending_projects

    try:
        with runtime.scheduler_authority() as acquired:
            assert acquired
            executor.begin_epoch()
            wait_until("productive-binding-settled", healthy_completed, stage="recovery-enrollment:backoff", timeout=20)
            assert starts and len(starts) == 1
            assert runtime.recovery_enrollment_pending_projects == {waiting.project_id}
            runtime.migration_path(waiting.project_id).unlink()
            now[0] += 0.999
            for _ in range(3):
                advance()
                assert len(starts) == 1
            now[0] += 0.001
            wait_until("waiting-binding-resumed", all_completed, stage="recovery-enrollment:retry", timeout=20)
            assert len(starts) == 2
            assert enrollment._settled == {waiting, advancing}
    finally:
        enrollment.stop()
        executor.shutdown()


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


def test_slow_capture_shutdown_fences_workers_before_releasing_scheduler_authority(tmp_path):
    from qqtools.plugins.qexp.runtime.locks import machine_lock

    runtime = MachineRuntime(tmp_path / "machine")
    cfg, binding = register(tmp_path, runtime, "project")
    enrollment = RecoveryEnrollment(runtime)
    executor = ProjectIOExecutor(runtime)
    controller = ProjectIOController(runtime, executor)
    registration_path = cfg.shared_root / "machines" / cfg.machine_name / "registration.json"
    before = registration_path.read_bytes()
    try:
        with runtime.scheduler_authority() as acquired:
            assert acquired
            executor.begin_epoch()
            with machine_lock(binding.shared_root, binding.machine_name) as acquired:
                assert acquired
                revision, bindings = runtime.load_registry_snapshot()
                with controller.admission_turn():
                    enrollment.advance(controller, bindings, revision)
                assert executor.status_view()["active_worker_count"] == 1
                executor.fence_epoch()
                started = time.monotonic()
                enrollment.stop()
                assert time.monotonic() - started < 1.0
                with MachineRuntime(runtime.root).scheduler_authority() as acquired:
                    assert not acquired
                status = executor.shutdown()
                assert time.monotonic() - started < 5.0
                assert not executor._load_epoch().active
                assert status["active_worker_count"] == status["exit_unverified_worker_count"] == 0
                assert registration_path.read_bytes() == before
    finally:
        enrollment.stop()
        executor.shutdown()
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
