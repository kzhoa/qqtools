"""Exercise Idle Job authorization with real local admission and process ownership."""

import sys
import threading
import time
from contextlib import contextmanager

import pytest

from qqtools.plugins.qexp.agent import idle_job_integration
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.idle_job import IdleJobSupervisor
from qqtools.plugins.qexp.agent.idle_job_config import set_idle_job_config
from qqtools.plugins.qexp.agent.project_io_controller import ProjectIOController
from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.gpu_policy import GpuDiscovery, GpuReservationPolicy

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


@contextmanager
def held_briefly(lock):
    acquired = threading.Event()
    release = threading.Event()

    def hold():
        with lock:
            acquired.set()
            release.wait(1)

    thread = threading.Thread(target=hold)
    thread.start()
    try:
        assert acquired.wait(2)
        yield
    finally:
        release.set()
        thread.join(2)
        assert not thread.is_alive()


@pytest.fixture
def decision(tmp_path, monkeypatch, request):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "test-machine", agent_mode="daemon")
    set_idle_job_config(
        runtime,
        enabled=True,
        command=f'{sys.executable} -c "import time; time.sleep(90)"',
        idle_delay_seconds=60,
        stop_timeout_seconds=0.2,
    )
    now = [0.0]
    supervisor = IdleJobSupervisor(runtime, monotonic=lambda: now[0])
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    runtime.project_io_executor = executor
    runtime.project_io_controller = ProjectIOController(runtime, executor)
    policy = GpuReservationPolicy(GpuDiscovery((0,), "available"), None, "unset")
    monkeypatch.setattr(
        idle_job_integration, "_gpu_policy_observation", lambda *a, **kw: (policy, {"revision": 0}, (0,))
    )
    with_project = getattr(request, "param", None) == "registered"
    if with_project:
        from qqtools.plugins.qexp import init_shared_root

        project = init_shared_root(tmp_path / "project" / ".qexp", "test-machine")
        runtime.add_binding(project.shared_root, project.machine_name)
    revision, bindings = runtime.load_registry()
    runtime.authority_ready_generations = {b.project_id: b.registration_generation for b in bindings}

    def tick():
        return idle_job_integration.advance_machine_idle_job(
            runtime,
            supervisor,
            available_gpus=[0],
            instance_id="test",
            cycle_completed=True,
            bindings=bindings,
            registry_revision=revision,
            enablement_converged=True,
            partition_observation="clear",
        )

    try:
        if with_project:
            from qqtools.plugins.qexp.agent.dispatch_loop import dispatch_machine_cycle_locked

            deadline = time.monotonic() + 20
            while True:
                dispatch_machine_cycle_locked(runtime, available_gpus=[0], supervise=False, publish_snapshots=False)
                status = tick()
                if status["reason"] == "idle_delay":
                    break
                assert time.monotonic() < deadline, status
                time.sleep(0.02)
        else:
            deadline = time.monotonic() + 5
            while tick()["reason"] != "idle_delay":
                assert time.monotonic() < deadline
                time.sleep(0.01)
        yield runtime, executor, supervisor, now, tick
    finally:
        supervisor.shutdown()
        runtime.project_io_controller.disarm_idle_job_guard()
        executor.shutdown()


@pytest.mark.parametrize("running", [False, True])
@pytest.mark.parametrize("pulse", [False, True])
def test_handoff_after_census_invalidates_the_actual_decision(decision, monkeypatch, running, pulse):
    runtime, _, supervisor, now, tick = decision
    now[0] = 60
    if running:
        deadline = time.monotonic() + 10
        while tick()["state"] != "running":
            assert time.monotonic() < deadline
            time.sleep(0.02)
    original = supervisor.tick

    def interleave(*args, **kwargs):
        assert kwargs["demand"] == "idle"
        key = ("project", "attempt")
        runtime.pending_launch_handoffs[key] = object()
        if pulse:
            del runtime.pending_launch_handoffs[key]
        return original(*args, **kwargs)

    monkeypatch.setattr(supervisor, "tick", interleave)
    status = tick()
    assert status["state"] in {"waiting", "stopping"}
    assert status["reason"] == "unknown_demand"


@pytest.mark.parametrize("read_number", [1, 2])
@pytest.mark.parametrize("method", ["_scan_records", "_status_from_records"])
def test_slow_executor_evidence_cannot_hold_control_loop(decision, monkeypatch, method, read_number):
    from qqtools.plugins.qexp.runtime.locks import exclusive

    _, executor, _, now, tick = decision
    now[0] = 60
    release = threading.Event()
    blocked = threading.Event()
    original = getattr(executor, method)

    reads = 0

    def stall(*args, **kwargs):
        nonlocal reads
        reads += 1
        if reads == read_number:
            blocked.set()
            release.wait(2)
        return original(*args, **kwargs)

    monkeypatch.setattr(executor, method, stall)
    try:
        started = time.monotonic()
        status = tick()
        elapsed = time.monotonic() - started
        assert blocked.is_set()
        with exclusive(executor.paths["project_io_lock"], blocking=False) as training_can_proceed:
            assert training_can_proceed
        assert elapsed < 0.150
        assert status["state"] == "waiting"
        assert status["reason"] == "unknown_demand"
    finally:
        release.set()


def test_shared_writer_cannot_commit_until_final_decision_releases_all_projects(tmp_path, monkeypatch):
    from qqtools.plugins.qexp import init_shared_root, submit
    from qqtools.plugins.qexp.agent.idle_guard_client import IdleGuardClient
    from qqtools.plugins.qexp.agent.idle_guard_digest import idle_guard_digest
    from qqtools.plugins.qexp.commands.task import cancel
    from qqtools.plugins.qexp.runtime.locks import exclusive
    from qqtools.plugins.qexp.runtime.project_activation import read_project_activation

    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "test-machine", agent_mode="daemon")
    projects = [init_shared_root(tmp_path / name / ".qexp", "test-machine") for name in ("a", "b")]
    for cfg in projects:
        runtime.add_binding(cfg.shared_root, cfg.machine_name)
    revision, bindings = runtime.load_registry()

    def digest():
        rows = []
        for binding in bindings:
            checkpoint = read_project_activation(binding.shared_root)
            record = checkpoint["project_activation"] if checkpoint else {}
            rows.append(
                [
                    binding.project_id,
                    binding.registration_generation,
                    str(binding.shared_root),
                    record.get("epoch"),
                    record.get("sequence"),
                ]
            )
        return idle_guard_digest(rows)

    executor = ProjectIOExecutor(runtime)
    epoch = executor.begin_epoch()
    client = IdleGuardClient(executor)
    original = digest()
    try:
        client.start(revision)
        with client.authorize(revision, epoch, original, 2, time.monotonic() + 5) as granted:
            assert granted
            for cfg in projects:
                with exclusive(cfg.shared_root / "locks" / "project-activation-v1.lock", blocking=False) as writable:
                    assert not writable
        cfg = projects[1]
        task = submit(cfg, ["true"], requested_gpus=1, working_dir=tmp_path)
        cancel(cfg, task.task_id, reservation_runtime_root=runtime.root)
        changed = digest()
        assert changed != original
        with client.authorize(revision, epoch, original, 2, time.monotonic() + 1) as granted:
            assert not granted
        with client.authorize(revision, epoch, changed, 2, time.monotonic() + 1) as granted:
            assert granted
        from qqtools.plugins.qexp.runtime.paths import machine_registration_path
        from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json

        registration = machine_registration_path(cfg.shared_root, cfg.machine_name)
        initial = read_json(registration)
        for changed_fields in ({"generation": "1" * 32}, {"eligibility_expires_at": "2000-01-01T00:00:00Z"}):
            atomic_replace(registration, {"registration": {**initial["registration"], **changed_fields}})
            assert digest() == changed
            with client.authorize(revision, epoch, changed, 2, time.monotonic() + 1) as granted:
                assert not granted
        atomic_replace(registration, initial)
        # A separate status reader sees the verified response after the original
        # request has crossed its overdue threshold.
        later = time.time() + 12
        with monkeypatch.context() as clock:
            clock.setattr(time, "time", lambda: later)
            with client.authorize(revision, epoch, changed, 2, time.monotonic() + 1) as granted:
                assert granted
            active = executor.unresolved_requests()
            with monkeypatch.context() as unavailable:

                def fail_status(*args):
                    raise OSError("diagnostic unavailable")

                unavailable.setattr(executor, "publish_idle_guard_response", fail_status)
                client.service(revision, epoch, active)
                with client.authorize(revision, epoch, changed, 2, time.monotonic() + 1) as granted:
                    assert granted
            client.service(revision, epoch, active)
            assert ProjectIOExecutor(runtime).status_view()["envelope"] == "healthy"
            clock.setattr(time, "time", lambda: later + 11)
            assert ProjectIOExecutor(runtime).status_view()["overdue_worker_count"] == 1
        client.close()
        with held_briefly(exclusive(executor.paths["project_io_lock"])):
            # Guard eviction cannot reenter a blocking reconciliation while
            # training is waiting for a worker slot.
            started = time.monotonic()
            client.service(revision, epoch, active)
            assert time.monotonic() - started < 0.15
    finally:
        client.close()
        executor.shutdown()


@pytest.mark.parametrize("running", [False, True])
@pytest.mark.parametrize("sustained", [False, True])
def test_lock_miss_uses_one_turn_budget(decision, monkeypatch, running, sustained):
    from qqtools.plugins.qexp.agent import idle_executor_observation
    from qqtools.plugins.qexp.runtime.locks import exclusive

    _, executor, supervisor, now, tick = decision
    now[0] = 59
    assert tick()["reason"] == "idle_delay"
    now[0] = 60
    pid = None
    if running:
        end = time.monotonic() + 10
        while tick()["state"] != "running":
            assert time.monotonic() < end
            time.sleep(0.02)
        pid = supervisor._process["pid"]
    missed = threading.Event()
    locked = threading.Event()

    @contextmanager
    def record_miss(*args, **kwargs):
        with exclusive(*args, **kwargs) as acquired:
            if not acquired:
                missed.set()
            yield acquired

    monkeypatch.setattr(idle_executor_observation, "exclusive", record_miss)

    def holder():
        with exclusive(executor.paths["project_io_lock"]):
            locked.set()
            if missed.wait(2):
                time.sleep(0.300 if sustained else 0.005)

    thread = threading.Thread(target=holder)
    thread.start()
    try:
        assert locked.wait(2)
        start = time.monotonic()
        status = tick()
        assert time.monotonic() - start < 0.150
        assert missed.is_set()
        if sustained:
            assert status["reason"] == "unknown_demand"
            assert supervisor._idle_since is None
        elif running:
            assert status["state"] == "running"
            assert supervisor._process["pid"] == pid
        else:
            assert status["reason"] in {"launching", "running"}, status
    finally:
        thread.join(3)
        assert not thread.is_alive()


@pytest.mark.parametrize("decision", ["registered"], indirect=True)
@pytest.mark.parametrize("running", [False, True])
def test_real_submission_pulse_after_idle_census_rejects_stale_interval(decision, tmp_path, monkeypatch, running):
    from qqtools.plugins.qexp import submit
    from qqtools.plugins.qexp.commands.task import cancel
    from qqtools.plugins.qexp.runtime.resources.idle_job import read_idle_job_demand

    runtime, _, supervisor, now, tick = decision
    now[0] = 60
    if running:
        deadline = time.monotonic() + 5
        while tick()["state"] != "running":
            assert time.monotonic() < deadline
            time.sleep(0.02)
    cfg = runtime.load_registry()[1][0].root_config()
    original = supervisor.tick
    local_revision = read_idle_job_demand(runtime.root)
    # Isolate checkpoint fencing from the independent deadline rejection.
    monkeypatch.setattr(idle_job_integration, "IDLE_DECISION_SECONDS", 5)
    checked = []
    authorize = runtime.project_io_controller._idle_job_guard.authorize

    @contextmanager
    def capture(*args):
        with authorize(*args) as granted:
            checked.append((granted, time.monotonic() < args[-1]))
            yield granted

    monkeypatch.setattr(runtime.project_io_controller._idle_job_guard, "authorize", capture)

    def interleave(*args, **kwargs):
        assert kwargs["demand"] == "idle"
        task = submit(cfg, ["true"], requested_gpus=1, working_dir=tmp_path)
        cancel(cfg, task.task_id, reservation_runtime_root=runtime.root)
        assert read_idle_job_demand(runtime.root) == local_revision
        return original(*args, **kwargs)

    monkeypatch.setattr(supervisor, "tick", interleave)
    status = tick()
    assert checked == [(False, True)]
    assert status["reason"] == "unknown_demand", status
    assert status["state"] in {"stopping", "waiting"}
    assert supervisor._idle_since is None


def test_timeout_releases_shared_locks_while_checkpoint_read_is_still_blocked(tmp_path, monkeypatch):
    import socket
    from types import SimpleNamespace

    from qqtools.plugins.qexp import init_shared_root
    from qqtools.plugins.qexp.agent import idle_guard_worker
    from qqtools.plugins.qexp.agent.idle_guard_client import IdleGuardClient
    from qqtools.plugins.qexp.runtime.locks import schema_lock

    project = init_shared_root(tmp_path / "project" / ".qexp", "test-machine")
    (project.shared_root / "locks" / "registrations.lock").touch()
    binding = SimpleNamespace(shared_root=project.shared_root)
    reader, worker = socket.socketpair(type=socket.SOCK_SEQPACKET)
    reader.setblocking(False)
    client = IdleGuardClient(None)
    client._request = SimpleNamespace(registry_revision=1, executor_epoch="epoch")
    client._channel = reader
    client._is_ready = True
    client._last_response_at = time.monotonic()
    stalled = threading.Event()
    release = threading.Event()

    def read(_binding):
        stalled.set()
        release.wait(3)
        raise OSError("stalled Project unavailable")

    def serve():
        with worker:
            assert worker.recv(5) == b"check"
            try:
                idle_guard_worker._answer_check(worker, (binding,))
            except BrokenPipeError:
                pass

    monkeypatch.setattr(idle_guard_worker, "_read_checkpoint", read)
    thread = threading.Thread(target=serve)
    thread.start()
    try:
        with client.authorize(1, "epoch", "0" * 64, 1, time.monotonic() + 0.15) as authorized:
            assert not authorized
        assert stalled.is_set()
        with schema_lock(project.shared_root, blocking=False) as writable:
            assert writable
    finally:
        release.set()
        client.close()
        thread.join(3)
    assert not thread.is_alive()


@pytest.mark.parametrize("running", [False, True])
def test_contended_final_machine_fence_returns_without_waiting(decision, running):
    from qqtools.plugins.qexp.runtime.resources.idle_job import idle_job_guard

    runtime, _, supervisor, now, tick = decision
    now[0] = 60
    if running:
        deadline = time.monotonic() + 5
        while tick()["state"] != "running":
            assert time.monotonic() < deadline
            time.sleep(0.01)
    with held_briefly(idle_job_guard(runtime.root)):
        started = time.monotonic()
        status = tick()
        assert time.monotonic() - started < 0.15
        assert status["state"] == ("blocked" if running else "waiting")
    assert supervisor._idle_since is None
    if running:
        assert tick()["state"] == "stopping"
    else:
        assert tick()["reason"] == "idle_delay"


def test_decision_skips_slow_transport_reconciliation(decision, monkeypatch):
    _, executor, _, now, tick = decision
    reconciliations = []
    original = executor._reconcile_locked

    def slow_reconcile(**kwargs):
        reconciliations.append(None)
        time.sleep(0.080)
        return original(**kwargs)

    monkeypatch.setattr(executor, "_reconcile_locked", slow_reconcile)
    now[0] = 59
    started = time.monotonic()
    assert tick()["reason"] == "idle_delay"
    assert time.monotonic() - started < 0.15
    assert not reconciliations


@pytest.mark.parametrize("decision", ["registered"], indirect=True)
def test_resident_omission_preserves_exact_startup_proof(decision, monkeypatch):
    from qqtools.plugins.qexp.agent.dispatch_loop import dispatch_machine_cycle_locked

    runtime, _, _, _, _ = decision
    revision, (binding,) = runtime.load_registry()
    controller = runtime.project_io_controller
    assert controller.validated_config(binding, revision) is not None
    with monkeypatch.context() as omitted:
        omitted.setattr(runtime.working_set, "resident_bindings", lambda *args: [])
        dispatch_machine_cycle_locked(runtime, available_gpus=[0], supervise=False, publish_snapshots=False)
    assert runtime.authority_ready_generations.get(binding.project_id) == binding.registration_generation
    assert controller.validated_config(binding, revision) is not None
    assert controller.validated_config(binding, revision + 1) is None
