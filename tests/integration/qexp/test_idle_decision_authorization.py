"""Exercise Idle Job authorization with real local admission and process ownership."""

import multiprocessing
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
def held_briefly(lock, *, timeout=1.0):
    acquired = threading.Event()
    release = threading.Event()

    def hold():
        with lock:
            acquired.set()
            release.wait(timeout)

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
    project_count = {"registered": 1, "two_projects": 2, "five_projects": 5}.get(getattr(request, "param", None), 0)
    command = f'{sys.executable} -c "import time; time.sleep(90)"'
    if project_count == 2:
        command = f"{sys.executable} -c \"from pathlib import Path; Path('command-ran').touch()\""
    set_idle_job_config(
        runtime,
        enabled=True,
        command=command.replace("command-ran", str(tmp_path / "command-ran")),
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
    for number in range(project_count):
        from qqtools.plugins.qexp import init_shared_root

        project = init_shared_root(tmp_path / f"project-{number}" / ".qexp", "test-machine")
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
        if project_count:
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
            assert (status["state"], status["reason"]) == ("waiting", "preparing"), status
            deadline = time.monotonic() + 5
            while tick()["state"] != "running":
                assert time.monotonic() < deadline
                time.sleep(0.01)
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


@pytest.mark.parametrize("running", [False, True])
def test_executor_timeout_cannot_block_again_on_machine_guard(decision, running):
    from qqtools.plugins.qexp.agent.idle_job import idle_job_status
    from qqtools.plugins.qexp.runtime.locks import exclusive
    from qqtools.plugins.qexp.runtime.resources.idle_job import idle_job_guard

    runtime, executor, supervisor, now, tick = decision
    now[0] = 60
    if running:
        end = time.monotonic() + 5
        while tick()["state"] != "running":
            assert time.monotonic() < end
            time.sleep(0.01)

    with (
        held_briefly(exclusive(executor.paths["project_io_lock"]), timeout=0.32),
        held_briefly(idle_job_guard(runtime.root), timeout=0.32),
    ):
        start = time.monotonic()
        status = tick()
        assert time.monotonic() - start < 0.15
        assert supervisor._idle_since is None
        if running:
            assert status["state"] == "blocked"
        else:
            assert status["reason"] == "unknown_demand"
    assert idle_job_status(runtime)["interval_detail"]["last_reset"]["check"] == "executor_unready"
    if running:
        assert tick()["state"] == "stopping"
    else:
        assert tick()["reason"] == "idle_delay"


def test_stalled_final_registry_read_releases_decision_without_locking_executor(decision, monkeypatch):
    from qqtools.plugins.qexp.runtime.locks import exclusive

    runtime, executor, supervisor, now, tick = decision
    now[0] = 60
    read_started = threading.Event()
    release = threading.Event()
    returned = threading.Event()
    available = []
    original = runtime.load_registry_uncached

    def read_registry():
        with exclusive(executor.paths["project_io_lock"], blocking=False) as acquired:
            available.append(acquired)
        read_started.set()
        release.wait(0.27)
        try:
            return original()
        finally:
            returned.set()

    try:
        with monkeypatch.context() as delayed:
            delayed.setattr(runtime, "load_registry_uncached", read_registry)
            start = time.monotonic()
            status = tick()
            elapsed = time.monotonic() - start
            assert read_started.is_set()
            assert available == [True]
            assert elapsed < 0.15
            assert not returned.is_set()
            assert status["reason"] == "unknown_demand"
            assert supervisor._idle_since is None
            with runtime.registry_guard(blocking=False) as registry_available:
                assert registry_available
            assert tick()["reason"] == "unknown_demand"
            assert available == [True]
    finally:
        release.set()
    assert returned.wait(1)
    end = time.monotonic() + 2
    while tick()["reason"] != "idle_delay":
        assert time.monotonic() < end
        time.sleep(0.01)


@pytest.mark.parametrize("changed", ["registry", "executor"])
def test_final_local_reads_reject_changed_identity(decision, monkeypatch, changed):
    from qqtools.plugins.qexp.runtime.locks import exclusive

    runtime, executor, supervisor, now, tick = decision
    now[0] = 60
    original = runtime.load_registry_uncached
    observed = threading.Event()
    monkeypatch.setattr(idle_job_integration, "IDLE_DECISION_SECONDS", 3)

    def change_during_read():
        revision, bindings = original()
        if changed == "registry":
            observed.set()
            return revision + 1, bindings
        with exclusive(executor.paths["project_io_lock"], blocking=False) as available:
            if not available:
                return revision, bindings
        executor.fence_epoch()
        executor.begin_epoch()
        observed.set()
        return revision, bindings

    monkeypatch.setattr(runtime, "load_registry_uncached", change_during_read)
    status = tick()
    assert observed.is_set()
    assert status["reason"] == "unknown_demand"
    assert supervisor._idle_since is None


def test_unknown_config_read_keeps_machine_guard_available_to_training(decision, monkeypatch):
    from qqtools.plugins.qexp.runtime.resources.idle_job import idle_job_guard

    runtime, _, supervisor, _, _ = decision
    reader = supervisor._load_config
    availability = []

    def check_machine_guard():
        with idle_job_guard(runtime.root, blocking=False) as acquired:
            availability.append(acquired)
        return reader()

    monkeypatch.setattr(supervisor, "_load_config", check_machine_guard)
    assert supervisor.tick(demand="unknown")["reason"] == "unknown_demand"
    assert availability == [True]


def test_unavailable_gpu_stop_does_not_wait_for_machine_guard(decision):
    from qqtools.plugins.qexp.runtime.resources.idle_job import idle_job_guard

    runtime, _, supervisor, _, tick = decision
    set_idle_job_config(runtime, gpu_ids=(1,))
    with held_briefly(idle_job_guard(runtime.root), timeout=0.32):
        start = time.monotonic()
        assert tick()["state"] == "blocked"
        assert time.monotonic() - start < 0.15
        assert supervisor._idle_since is None
    assert tick()["reason"] == "gpu_policy_unavailable"


def test_unowned_deferred_disable_does_not_override_current_unknown(decision):
    from qqtools.plugins.qexp.runtime.resources.idle_job import idle_job_guard

    runtime, _, supervisor, _, _ = decision
    set_idle_job_config(runtime, enabled=False)
    with held_briefly(idle_job_guard(runtime.root), timeout=0.32):
        assert supervisor.tick(demand="unknown")["state"] == "blocked"
    set_idle_job_config(runtime, enabled=True)
    status = supervisor.tick(demand="unknown")
    assert (status["state"], status["reason"]) == ("waiting", "unknown_demand")


def _submit_and_cancel_every_project(configs, runtime_root, working_dir):
    from qqtools.plugins.qexp import submit
    from qqtools.plugins.qexp.commands.task import cancel

    for cfg in configs:
        task = submit(cfg, ["true"], requested_gpus=1, working_dir=working_dir)
        cancel(cfg, task.task_id, reservation_runtime_root=runtime_root)


@pytest.mark.parametrize("decision", ["two_projects"], indirect=True)
@pytest.mark.parametrize("stage", ["file_fsync", "directory_fsync"])
def test_remote_training_commits_during_ownership_persistence(decision, tmp_path, monkeypatch, stage):
    import os
    import posix

    from qqtools.plugins.qexp.runtime.resources import idle_job as admission

    monkeypatch.setattr(os, "fsync", posix.fsync)
    runtime, _, supervisor, now, tick = decision
    configs = [binding.root_config() for binding in runtime.load_registry()[1]]
    writer = multiprocessing.get_context("spawn").Process(
        target=_submit_and_cancel_every_project, args=(configs, runtime.root, tmp_path)
    )
    observed = []
    replace = admission.atomic_replace

    def publish(path, payload):
        if observed or path.name != "ownership.json":
            return replace(path, payload)

        def barrier(step):
            if step == stage:
                observed.append(path.exists())
                writer.start()
                try:
                    writer.join(8)
                    assert writer.exitcode == 0, "training writer could not finish while publication was paused"
                finally:
                    if writer.is_alive():
                        writer.kill()
                        writer.join(3)
                assert not (tmp_path / "command-ran").exists()

        return replace(path, payload, io_step_observer=barrier)

    monkeypatch.setattr(admission, "atomic_replace", publish)
    now[0] = 60
    end = time.monotonic() + 12
    while not observed or supervisor._idle_since is not None:
        status = tick()
        assert time.monotonic() < end, status
        time.sleep(0.01)
    assert observed == [stage == "directory_fsync"]
    assert not (tmp_path / "command-ran").exists()


def _timed_training_writes(configs, runtime_root, channel):
    from qqtools.plugins.qexp import submit
    from qqtools.plugins.qexp.commands.task import cancel

    with channel:
        for i in range(8):
            cfg = configs[i % len(configs)]
            started = time.monotonic()
            channel.send(("attempt", i, started))
            task = submit(cfg, ["true"], requested_gpus=1, working_dir=cfg.project_root)
            channel.send(("committed", i, time.monotonic()))
            cancel(cfg, task.task_id, reservation_runtime_root=runtime_root)
        channel.send(("finished", 8, time.monotonic()))


@pytest.mark.parametrize("guards_enabled", [False, True], ids=["control", "two_guards"])
def test_training_writer_progress_on_overlapping_projects(tmp_path, guards_enabled, record_property):
    from contextlib import ExitStack

    from qqtools.plugins.qexp import init_shared_root
    from qqtools.plugins.qexp.agent.idle_guard_client import IdleGuardClient
    from qqtools.plugins.qexp.agent.idle_guard_digest import idle_guard_digest
    from qqtools.plugins.qexp.runtime.project_activation import read_project_activation

    configs = [init_shared_root(tmp_path / root / ".qexp", "writer") for root in ("one", "two")]
    ctx = multiprocessing.get_context("spawn")
    reader, sender = ctx.Pipe(False)
    stop = threading.Event()

    def authorize(client, revision, epoch, bindings, deadline):
        rows = []
        for b in bindings:
            record = (read_project_activation(b.shared_root) or {}).get("project_activation", {})
            rows.append(
                (
                    b.project_id,
                    b.registration_generation,
                    str(b.shared_root),
                    record.get("epoch"),
                    record.get("sequence"),
                )
            )
        with client.authorize(revision, epoch, idle_guard_digest(rows), len(bindings), deadline) as granted:
            return granted

    def repeat(args, grants):
        while not stop.is_set():
            if authorize(*args, time.monotonic() + 0.05):
                grants.append(time.monotonic())
            stop.wait(0.003)

    with ExitStack() as stack:
        stack.enter_context(reader)
        stack.enter_context(sender)
        workers = []
        for name in ("left", "right"):
            runtime = MachineRuntime(tmp_path / name)
            initialize_machine(runtime, name, agent_mode="daemon")
            for cfg in configs:
                runtime.add_binding(cfg.shared_root, name)
            if guards_enabled:
                executor = ProjectIOExecutor(runtime)
                epoch = executor.begin_epoch()
                stack.callback(executor.shutdown)
                revision, bindings = runtime.load_registry()
                client = IdleGuardClient(executor)
                stack.callback(client.close)
                client.start(revision)
                args = (client, revision, epoch, bindings)
                assert authorize(*args, time.monotonic() + 4)
                grants = []
                workers.append((threading.Thread(target=repeat, args=(args, grants)), grants))

        writer = ctx.Process(target=_timed_training_writes, args=(configs, runtime.root, sender))
        events = []
        for thread, _ in workers:
            thread.start()
        writer.start()
        try:
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline and reader.poll(max(0, deadline - time.monotonic())):
                event = reader.recv()
                events.append(event)
                if event[0] == "finished":
                    break
            assert events and events[-1][0] == "finished", events
            writer.join(1)
            assert writer.exitcode == 0
        finally:
            if writer.is_alive():
                writer.kill()
                writer.join(3)
            stop.set()
            for thread, _ in workers:
                thread.join(2)
                assert not thread.is_alive()
        attempts = {i: t for kind, i, t in events if kind == "attempt"}
        committed = {i: t for kind, i, t in events if kind == "committed"}
        assert attempts.keys() == committed.keys() == set(range(8))
        assert all(any(attempts[0] <= t <= events[-1][2] for t in grants) for _, grants in workers)
        record_property("max_submit_seconds", max(committed[i] - attempts[i] for i in attempts))


def _acquire_training_lock(root):
    from qqtools.plugins.qexp.runtime.locks import schema_reader_lock

    with schema_reader_lock(root, blocking=False) as acquired:
        assert acquired


@pytest.mark.parametrize("decision", ["registered"], indirect=True)
@pytest.mark.parametrize("stage", ["spawn", "launch_status", "running_status", "failure_cleanup"])
def test_local_side_effects_release_project_exclusion(decision, monkeypatch, stage):
    from qqtools.plugins.qexp.agent import idle_job

    runtime, _, supervisor, now, tick = decision
    root = runtime.load_registry()[1][0].shared_root
    checked = []

    def barrier():
        if checked:
            return
        writer = multiprocessing.get_context("spawn").Process(target=_acquire_training_lock, args=(root,))
        writer.start()
        try:
            writer.join(5)
            assert writer.exitcode == 0
        finally:
            if writer.is_alive():
                writer.kill()
                writer.join(3)
        checked.append(True)

    original_spawn = idle_job.spawn_idle_job

    def spawn(*args, **kwargs):
        if stage == "spawn":
            barrier()
        if stage == "failure_cleanup":
            raise OSError("injected spawn failure")
        return original_spawn(*args, **kwargs)

    monkeypatch.setattr(idle_job, "spawn_idle_job", spawn)
    publish = supervisor._publish

    def publish_with_barrier(*, state, reason):
        if (stage, reason) in {("launch_status", "preparing"), ("running_status", "running")}:
            barrier()
        return publish(state=state, reason=reason)

    monkeypatch.setattr(supervisor, "_publish", publish_with_barrier)
    cleanup = supervisor._preparation_failed_locked

    def clean_failed(error):
        barrier()
        return cleanup(error)

    if stage == "failure_cleanup":
        monkeypatch.setattr(supervisor, "_preparation_failed_locked", clean_failed)
    now[0] = 60
    deadline = time.monotonic() + 8
    while not checked:
        status = tick()
        assert time.monotonic() < deadline, status
        time.sleep(0.01)


@pytest.mark.parametrize("decision", ["registered"], indirect=True)
def test_training_commit_to_reservation_timeline(decision, tmp_path, monkeypatch, record_property):
    from qqtools.plugins.qexp import submit
    from qqtools.plugins.qexp.agent import idle_job
    from qqtools.plugins.qexp.runtime import submission
    from qqtools.plugins.qexp.runtime.resources.idle_job import read_idle_job_ownership
    from qqtools.plugins.qexp.runtime.resources.reservations import reserve

    runtime, _, supervisor, now, tick = decision
    now[0] = 60
    timeout = time.monotonic() + 8
    while tick()["state"] != "running":
        assert time.monotonic() < timeout
        time.sleep(0.01)
    marks = {}
    write = submission._submission_atomic_writer
    request_stop = idle_job.request_idle_job_stop

    def writer(path, envelope, **kwargs):
        result = write(path, envelope, **kwargs)
        if envelope.get("submission", {}).get("state") == "committed":
            marks.setdefault("commit", time.monotonic())
        return result

    def stop(directory, reason):
        marks.setdefault("observed", time.monotonic())
        request_stop(directory, reason)
        marks.setdefault("stop", time.monotonic())

    monkeypatch.setattr(submission, "_submission_atomic_writer", writer)
    monkeypatch.setattr(idle_job, "request_idle_job_stop", stop)
    cfg = runtime.load_registry()[1][0].root_config()
    marks["attempt"] = time.monotonic()
    task = submit(cfg, ["true"], requested_gpus=1, working_dir=tmp_path)
    marks["due"] = time.monotonic() + 0.02
    time.sleep(0.025)
    marks["start"] = time.monotonic()
    tick()
    assert "stop" in marks, "first due decision must revoke after demand or incomplete proof"
    while read_idle_job_ownership(runtime.root) is not None:
        supervisor.tick("busy")
        assert time.monotonic() < timeout
        time.sleep(0.01)
    marks["cleaned"] = time.monotonic()
    assert reserve(runtime.root, task.task_id, [0])["reservation"]["task_id"] == task.task_id
    marks["reserved"] = time.monotonic()
    order = ("attempt", "commit", "due", "start", "observed", "stop", "cleaned", "reserved")
    assert [marks[k] for k in order] == sorted(marks.values())
    record_property("submission_operation", task.submission_operation_id)
    for earlier, later in zip(order, order[1:]):
        record_property(f"{earlier}_to_{later}_seconds", marks[later] - marks[earlier])


@pytest.mark.parametrize("decision", ["registered"], indirect=True)
def test_same_generation_renewal_after_prior_safe_horizon_breaks_countdown(decision):
    from datetime import datetime, timedelta, timezone

    from qqtools.plugins.qexp.lease import load_lease_policy
    from qqtools.plugins.qexp.runtime.locks import exclusive
    from qqtools.plugins.qexp.runtime.paths import machine_registration_path
    from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json

    runtime, _, supervisor, now, tick = decision
    binding = runtime.load_registry()[1][0]
    cfg = binding.root_config()
    path = machine_registration_path(cfg.shared_root, cfg.machine_name)
    original = read_json(path)
    skew = load_lease_policy(cfg).max_clock_skew_seconds
    near = datetime.now(timezone.utc) + timedelta(seconds=skew + 0.5)
    lock = cfg.shared_root / "locks" / "registrations.lock"
    with exclusive(lock):
        atomic_replace(path, {"registration": {**original["registration"], "eligibility_expires_at": near.isoformat()}})
    now[0] = 59
    assert tick()["reason"] == "idle_delay"
    time.sleep(0.55)
    with exclusive(lock):
        atomic_replace(path, original)
    now[0] = 60
    assert tick()["reason"] == "unknown_demand"
    assert supervisor._idle_since is None
    assert tick()["reason"] == "idle_delay"


@pytest.mark.parametrize("decision", ["five_projects"], indirect=True)
@pytest.mark.parametrize("overdue", [False, True], ids=["four_occupied", "two_overdue"])
def test_training_arrival_stops_locally_without_a_project_slot(decision, tmp_path, monkeypatch, overdue):
    import signal

    from qqtools.plugins.qexp import submit
    from qqtools.plugins.qexp.agent.project_io_protocol import PROJECT_IO_OVERDUE_SECONDS
    from qqtools.plugins.qexp.runtime.resources.idle_job import read_idle_job_ownership
    from qqtools.plugins.qexp.runtime.resources.reservations import reserve

    runtime, executor, supervisor, now, tick = decision
    now[0] = 60
    end = time.monotonic() + 8
    while tick()["state"] != "running":
        assert time.monotonic() < end
        time.sleep(0.01)
    revision, bindings = runtime.load_registry()
    ctrl = runtime.project_io_controller
    ctrl.disarm_idle_job_guard(clear_census=False)
    # Warm-up completed real startup/census requests. Consume their durable
    # results before occupying the four request slots with controlled workers.
    while pending := executor.unresolved_requests():
        executor.poll()
        for request in pending:
            if request.operation_kind == "idle_decision_guard":
                executor.resolve_stale_idle_decision_guard(request.request_id, request)
            else:
                executor.consume(request.request_id, request)
        assert time.monotonic() < end
        time.sleep(0.01)
    owner = read_idle_job_ownership(runtime.root)
    stopped = []
    try:
        count = 2 if overdue else 4
        for binding in bindings[:count]:
            request = executor.prepare_validate_binding(binding, revision)
            executor.start(request.request_id)
            process = executor._children[request.request_id]
            process.send_signal(signal.SIGSTOP)
            stopped.append(process)
        task = submit(bindings[-1].root_config(), ["true"], requested_gpus=1, working_dir=tmp_path)
        with monkeypatch.context() as age:
            if overdue:
                future = time.time() + PROJECT_IO_OVERDUE_SECONDS + 1
                age.setattr(time, "time", lambda: future)
            status = executor.status_view()
            assert status["overdue_worker_count"] == (2 if overdue else 0)
            assert status["free_slot_count"] == 4 - count
            started = time.monotonic()
            result = tick()
            assert result["state"] == "stopping", result
            assert time.monotonic() - started < 0.25
            revoked = read_idle_job_ownership(runtime.root)
            assert (revoked["instance_id"], revoked["phase"]) == (owner["instance_id"], "stopping")
            assert (runtime.root / "idle-job" / "instances" / owner["instance_id"] / "stop.json").is_file()
        while read_idle_job_ownership(runtime.root) is not None:
            supervisor.tick("busy")
            assert time.monotonic() < end
            time.sleep(0.01)
        assert reserve(runtime.root, task.task_id, [0])["reservation"]["task_id"] == task.task_id
        assert all(p.poll() is None for p in stopped)
    finally:
        for process in stopped:
            process.send_signal(signal.SIGCONT)
