"""Machine Idle Job lifecycle without Project or experiment identities."""

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pytest

from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.idle_job import IdleJobSupervisor, idle_job_status
from qqtools.plugins.qexp.agent.idle_job_config import set_idle_job_config
from qqtools.plugins.qexp.agent.idle_job_process import idle_job_processes_exited
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.gpu_policy import GpuDiscovery, GpuReservationPolicy, set_gpu_policy
from qqtools.plugins.qexp.runtime.resources.cpu_lane import reserve_cpu, set_cpu_lane_capacity
from qqtools.plugins.qexp.runtime.resources.idle_job import (
    IdleJobAdmissionBlocked,
    idle_job_instance_path,
    read_idle_job_ownership,
)
from qqtools.plugins.qexp.runtime.resources.reservations import reservation_snapshot, reserve

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]

POLICY = GpuReservationPolicy(GpuDiscovery((0, 1, 3), "available"), None, "unset")


def _runtime(tmp_path, **changes):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "test-machine", agent_mode="daemon")
    set_idle_job_config(
        runtime,
        enabled=True,
        command=f'{sys.executable} -c "import time; time.sleep(60)"',
        idle_delay_seconds=0,
        stop_timeout_seconds=0.2,
        **changes,
    )
    return runtime


def _until(supervisor, predicate, *, demand="idle", policy=POLICY, seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        status = supervisor.tick(demand=demand, gpu_policy=policy)
        if predicate(status):
            return status
        time.sleep(0.03)
    pytest.fail(f"Idle Job did not converge: {status}")


def test_empty_machine_launch_disable_and_no_experiment_reservations(tmp_path):
    runtime = _runtime(tmp_path)
    supervisor = IdleJobSupervisor(runtime)
    try:
        status = _until(supervisor, lambda value: value["state"] == "running")
        assert status["resolved_gpu_ids"] == [3]
        assert runtime.load_registry()[1] == []
        assert not reservation_snapshot(runtime.root).reservations
        set_idle_job_config(runtime, enabled=False)
        _until(supervisor, lambda value: value["state"] == "disabled")
        assert read_idle_job_ownership(runtime.root) is None
    finally:
        assert supervisor.shutdown()


@pytest.mark.parametrize("lane", ["gpu", "different_gpu", "cpu"])
def test_eligible_admission_waits_for_process_cleanup(tmp_path, lane):
    runtime = _runtime(tmp_path)
    set_cpu_lane_capacity(runtime.root, capacity=1)
    supervisor = IdleJobSupervisor(runtime)

    def admit():
        if lane == "cpu":
            return reserve_cpu(runtime.root, "training", 1)
        return reserve(runtime.root, "training", [3 if lane == "gpu" else 0])

    try:
        _until(supervisor, lambda value: value["state"] == "running")
        with pytest.raises(IdleJobAdmissionBlocked):
            admit()
        assert not reservation_snapshot(runtime.root).reservations
        _until(supervisor, lambda value: read_idle_job_ownership(runtime.root) is None, demand="busy")
        # The training participant issued this stop before the supervisor saw busy demand.
        assert idle_job_status(runtime)["interval_detail"]["last_stop"] is None
        assert admit()["reservation"]["task_id"] == "training"
    finally:
        assert supervisor.shutdown()


@pytest.mark.parametrize(
    ("config_changes", "cleanup_between_reads"),
    [(False, False), (True, False), (True, True)],
    ids=["stable_config", "concurrent_config_edit", "concurrent_config_edit_and_exit"],
)
def test_training_stop_invalidates_interval_even_when_census_returns_idle(
    tmp_path, monkeypatch, config_changes, cleanup_between_reads
):
    from qqtools.plugins.qexp.agent import idle_job

    runtime = _runtime(tmp_path)
    set_idle_job_config(runtime, idle_delay_seconds=60)
    clock = [0]
    supervisor = IdleJobSupervisor(runtime, monotonic=lambda: clock[0])
    try:
        assert supervisor.tick(gpu_policy=POLICY)["reason"] == "idle_delay"
        clock[0] = 60
        _until(supervisor, lambda value: value["state"] == "running")
        with pytest.raises(IdleJobAdmissionBlocked):
            reserve(runtime.root, "training", [0])
        with monkeypatch.context() as patch:
            # Keep the first ownership observation in stopping until the chosen interleaving.
            patch.setattr(idle_job, "idle_job_processes_exited", lambda directory: False)
            clock[0] = 75
            with monkeypatch.context() as config_patch:
                if config_changes:
                    resolve_gpus = supervisor._resolve_gpus

                    def resolve_after_edit(config, policy):
                        set_idle_job_config(runtime, idle_delay_seconds=61)
                        if cleanup_between_reads:
                            owner = read_idle_job_ownership(runtime.root)
                            directory = idle_job_instance_path(runtime.root, owner["instance_id"])
                            deadline = time.monotonic() + 5
                            while not idle_job_processes_exited(directory):
                                assert time.monotonic() < deadline
                                time.sleep(0.01)
                            patch.setattr(idle_job, "idle_job_processes_exited", idle_job_processes_exited)
                        return resolve_gpus(config, policy)

                    config_patch.setattr(supervisor, "_resolve_gpus", resolve_after_edit)
                status = supervisor.tick("idle", gpu_policy=POLICY)
            assert status["reason"] == ("config_changed" if config_changes else "stopping")
            detail = idle_job_status(runtime)["interval_detail"]
            assert detail["idle_elapsed_seconds"] == 0
            assert detail["last_reset"]["check"] == "demand_revision_changed"
            assert detail["last_reset"]["elapsed_seconds"] == 75
            assert detail["last_stop"] is None
            clock[0] += 15
            status = supervisor.tick("idle", gpu_policy=POLICY)
            if cleanup_between_reads:
                assert status["reason"] == "idle_delay"
            else:
                assert status["state"] == "stopping"
            assert idle_job_status(runtime)["interval_detail"] == detail
    finally:
        assert supervisor.shutdown()


def test_gpu_policy_removal_stops_and_reselects_allowed_device(tmp_path):
    runtime = _runtime(tmp_path)
    supervisor = IdleJobSupervisor(runtime)
    try:
        _until(supervisor, lambda value: value["state"] == "running")
        old = read_idle_job_ownership(runtime.root)["instance_id"]
        set_gpu_policy(runtime, (0, 1))
        status = _until(
            supervisor,
            lambda value: value["state"] == "running" and read_idle_job_ownership(runtime.root)["instance_id"] != old,
        )
        assert status["resolved_gpu_ids"] == [1]
    finally:
        assert supervisor.shutdown()


@pytest.mark.parametrize("demand", ["busy", "unknown"])
def test_no_launch_without_proven_idle(tmp_path, demand):
    runtime = _runtime(tmp_path)
    supervisor = IdleJobSupervisor(runtime)
    try:
        for _ in range(3):
            status = supervisor.tick(demand=demand, gpu_policy=POLICY)
            assert status["state"] != "running"
            assert read_idle_job_ownership(runtime.root) is None
    finally:
        assert supervisor.shutdown()


@pytest.mark.parametrize("lane", ["other_gpu", "cpu"])
def test_concurrent_launch_and_admission_share_exclusion(tmp_path, lane):
    runtime = _runtime(tmp_path)
    set_cpu_lane_capacity(runtime.root, capacity=1)
    supervisor = IdleJobSupervisor(runtime)
    gate = threading.Barrier(2)

    def idle_launch():
        gate.wait(timeout=5)
        return supervisor.tick(demand="idle", gpu_policy=POLICY)

    def training_admission():
        gate.wait(timeout=5)
        try:
            if lane == "cpu":
                return reserve_cpu(runtime.root, "training", 1)
            return reserve(runtime.root, "training", [0])
        except IdleJobAdmissionBlocked:
            return None

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            idle_future = pool.submit(idle_launch)
            training_future = pool.submit(training_admission)
            idle_future.result(timeout=10)
            admitted = training_future.result(timeout=10)
        if admitted is not None:
            assert read_idle_job_ownership(runtime.root) is None
            for _ in range(3):
                supervisor.tick(demand="idle", gpu_policy=POLICY)
                assert read_idle_job_ownership(runtime.root) is None
        else:
            assert not reservation_snapshot(runtime.root).reservations
    finally:
        assert supervisor.shutdown()


def test_command_exit_uses_backoff_and_training_supersedes_retry(tmp_path):
    runtime = _runtime(tmp_path)
    set_idle_job_config(runtime, command=f'{sys.executable} -c "raise SystemExit(7)"')
    supervisor = IdleJobSupervisor(runtime)
    try:
        status = _until(supervisor, lambda value: value["state"] == "backoff")
        assert status["next_retry_at"] is not None
        assert status["last_exit_reason"] is not None
        assert status["failure_count"] >= 1
        for _ in range(3):
            status = supervisor.tick(demand="busy", gpu_policy=POLICY)
            assert status["state"] == "waiting"
            assert read_idle_job_ownership(runtime.root) is None
    finally:
        assert supervisor.shutdown()


def test_unknown_evidence_requires_a_fresh_idle_interval(tmp_path):
    runtime = _runtime(tmp_path)
    set_idle_job_config(runtime, idle_delay_seconds=0.3)
    supervisor = IdleJobSupervisor(runtime)
    try:
        supervisor.tick(demand="idle", gpu_policy=POLICY)
        time.sleep(0.2)
        supervisor.tick(demand="unknown", gpu_policy=POLICY)
        time.sleep(0.15)
        status = supervisor.tick(demand="idle", gpu_policy=POLICY)
        assert status["state"] == "waiting"
        assert read_idle_job_ownership(runtime.root) is None
        _until(supervisor, lambda value: value["state"] == "running")
    finally:
        assert supervisor.shutdown()


def test_recreated_supervisor_adopts_only_one_instance(tmp_path):
    runtime = _runtime(tmp_path)
    first = IdleJobSupervisor(runtime)
    second = None
    try:
        _until(first, lambda value: value["state"] == "running")
        owner = read_idle_job_ownership(runtime.root)
        second = IdleJobSupervisor(runtime)
        _until(second, lambda value: value["state"] == "running")
        assert read_idle_job_ownership(runtime.root)["instance_id"] == owner["instance_id"]
        assert len(list((runtime.root / "idle-job" / "instances").iterdir())) == 1
    finally:
        if second is not None:
            assert second.shutdown()
        assert first.shutdown()


def test_adopted_instance_records_stop_even_without_an_idle_interval(tmp_path):
    from qqtools.plugins.qexp.agent.idle_job_diagnostics import build_demand_observation

    runtime = _runtime(tmp_path)
    first = IdleJobSupervisor(runtime)
    try:
        _until(first, lambda value: value["state"] == "running")
        owner = read_idle_job_ownership(runtime.root)
        adopted = IdleJobSupervisor(runtime)
        assert adopted.tick(gpu_policy=POLICY)["state"] == "running"
        adopted.set_demand_observation(
            "unknown", build_demand_observation("authority_unready"), agent_instance_id="agent-b"
        )
        before = time.time()
        assert adopted.tick("unknown", gpu_policy=POLICY)["state"] == "stopping"
        after = time.time()
        detail = idle_job_status(runtime)["interval_detail"]
        assert detail["last_reset"] is None
        stop = detail["last_stop"]
        assert stop["check"] == "authority_unready"
        assert stop["instance_id"] == owner["instance_id"]
        assert before <= datetime.fromisoformat(stop["stopped_at"]).timestamp() <= after
        _until(adopted, lambda value: value["state"] == "waiting", demand="unknown")
        assert idle_job_status(runtime)["interval_detail"]["last_stop"] == stop
    finally:
        assert first.shutdown()


def test_configuration_replacement_waits_for_old_group_exit(tmp_path):
    runtime = _runtime(tmp_path)
    supervisor = IdleJobSupervisor(runtime)
    try:
        _until(supervisor, lambda value: value["state"] == "running")
        old = read_idle_job_ownership(runtime.root)
        old_directory = idle_job_instance_path(runtime.root, old["instance_id"])
        config = set_idle_job_config(runtime, gpu_ids=(0, 1))
        _until(
            supervisor,
            lambda value: value["state"] == "running" and value["config_revision"] == config.revision,
        )
        assert idle_job_processes_exited(old_directory)
        assert read_idle_job_ownership(runtime.root)["instance_id"] != old["instance_id"]
    finally:
        assert supervisor.shutdown()


def test_reservation_observation_requires_fresh_idle_interval_after_release(tmp_path):
    from qqtools.plugins.qexp.runtime.resources.reservations import release

    runtime = _runtime(tmp_path)
    set_idle_job_config(runtime, idle_delay_seconds=0.3)
    supervisor = IdleJobSupervisor(runtime)
    try:
        reservation = reserve(runtime.root, "training", [0])["reservation"]
        supervisor.tick(demand="idle", gpu_policy=POLICY)
        time.sleep(0.35)
        supervisor.tick(demand="idle", gpu_policy=POLICY)
        release(runtime.root, reservation["reservation_id"])
        supervisor.tick(demand="idle", gpu_policy=POLICY)
        assert read_idle_job_ownership(runtime.root) is None
        _until(supervisor, lambda value: value["state"] == "running")
    finally:
        assert supervisor.shutdown()


def test_spawn_failure_retention_and_large_retry_count(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.agent import idle_job as module
    from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json

    runtime = _runtime(tmp_path)
    now = [time.time()]

    def fail_spawn(*args, **kwargs):
        raise OSError("injected spawn failure")

    monkeypatch.setattr(module, "spawn_idle_job", fail_spawn)
    supervisor = IdleJobSupervisor(runtime, wall_clock=lambda: now[0])
    try:
        for _ in range(8):
            status = supervisor.tick(demand="idle", gpu_policy=POLICY)
            assert status["state"] == "backoff"
            now[0] += 61
        instances = runtime.root / "idle-job" / "instances"
        assert len(list(instances.iterdir())) <= 5
        state_path = runtime.root / "idle-job" / "state.json"
        state = read_json(state_path)
        state["idle_job_state"]["failure_count"] = 1025
        atomic_replace(state_path, state)
        restarted = IdleJobSupervisor(runtime, wall_clock=lambda: now[0])
        status = restarted.tick(demand="idle", gpu_policy=POLICY)
        assert status["state"] == "backoff"
        assert status["failure_count"] == 1026
        assert status["next_retry_at"] == now[0] + 60
        assert restarted.shutdown()
    finally:
        assert supervisor.shutdown()


def test_crash_after_ownership_release_preserves_retry_accounting(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.agent import idle_job as module
    from qqtools.plugins.qexp.runtime.store import read_json

    class InjectedCrash(BaseException):
        pass

    runtime = _runtime(tmp_path)
    set_idle_job_config(runtime, command=f'{sys.executable} -c "raise SystemExit(7)"')
    supervisor = IdleJobSupervisor(runtime)
    original_clear = module.clear_idle_job_ownership

    def crash_after_release(*args, **kwargs):
        original_clear(*args, **kwargs)
        raise InjectedCrash

    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(module, "clear_idle_job_ownership", crash_after_release)
            with pytest.raises(InjectedCrash):
                _until(supervisor, lambda value: value["state"] == "backoff")
        state = read_json(runtime.root / "idle-job" / "state.json")["idle_job_state"]
        assert state["failure_count"] == 1
        assert state["retry_at"] > time.time()
        assert module.idle_job_status(runtime)["state"] != "running"
        restarted = IdleJobSupervisor(runtime)
        status = restarted.tick(demand="idle", gpu_policy=POLICY)
        assert status["state"] == "backoff"
        assert status["failure_count"] == 1
        assert read_idle_job_ownership(runtime.root) is None
        assert restarted.shutdown()
    finally:
        assert supervisor.shutdown()


def test_terminal_instances_from_previous_boot_are_pruned(tmp_path, monkeypatch):
    import uuid

    from qqtools.plugins.qexp.agent import idle_job as module
    from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json

    runtime = _runtime(tmp_path)
    now = [time.time()]

    def fail_spawn(*args, **kwargs):
        raise OSError("injected spawn failure")

    monkeypatch.setattr(module, "spawn_idle_job", fail_spawn)
    supervisor = IdleJobSupervisor(runtime, wall_clock=lambda: now[0])
    try:
        for _ in range(5):
            assert supervisor.tick(demand="idle", gpu_policy=POLICY)["state"] == "backoff"
            now[0] += 61
        instances = runtime.root / "idle-job" / "instances"
        previous = list(instances.iterdir())
        previous_boot = str(uuid.uuid4())
        for instance in previous:
            for filename, envelope in (("intent.json", "idle_job_intent"), ("exit.json", "idle_job_exit")):
                value = read_json(instance / filename)
                value[envelope]["boot_id"] = previous_boot
                atomic_replace(instance / filename, value)
        for _ in range(5):
            assert supervisor.tick(demand="idle", gpu_policy=POLICY)["state"] == "backoff"
            now[0] += 61
        assert len(list(instances.iterdir())) <= 5
        assert sum(path.exists() for path in previous) < len(previous)
    finally:
        assert supervisor.shutdown()


def test_bounded_partition_progress_preserves_only_established_idle_interval(tmp_path):
    runtime = _runtime(tmp_path)
    set_idle_job_config(runtime, idle_delay_seconds=60)
    clock = [0.0]
    supervisor = IdleJobSupervisor(runtime, monotonic=lambda: clock[0])
    try:
        supervisor.tick(demand="idle", gpu_policy=POLICY, launch_allowed=False)
        clock[0] = 100.0
        assert supervisor.tick(demand="idle", gpu_policy=POLICY)["reason"] == "idle_delay"
        clock[0] = 130.0
        assert (
            supervisor.tick(demand="idle", gpu_policy=POLICY, launch_allowed=False)["reason"]
            == "partition_scan_pending"
        )
        clock[0] = 161.0
        assert (
            supervisor.tick(demand="idle", gpu_policy=POLICY, launch_allowed=False)["reason"]
            == "partition_scan_pending"
        )
        assert read_idle_job_ownership(runtime.root) is None
        _until(supervisor, lambda status: status["state"] == "running")
    finally:
        # Shutdown must use a moving clock for its bounded process wait.
        supervisor._monotonic = time.monotonic
        assert supervisor.shutdown()


def test_agent_stop_reclaims_idle_ownership_without_live_agent(tmp_path):
    from qqtools.plugins.qexp.agent.lifecycle import stop_machine_agent

    runtime = _runtime(tmp_path)
    supervisor = IdleJobSupervisor(runtime)
    try:
        _until(supervisor, lambda status: status["state"] == "running")
        owner = read_idle_job_ownership(runtime.root)
        directory = idle_job_instance_path(runtime.root, owner["instance_id"])
        assert stop_machine_agent(runtime) is False
        assert idle_job_processes_exited(directory)
        assert read_idle_job_ownership(runtime.root) is None
    finally:
        assert supervisor.shutdown()


def test_reservation_pulse_is_visible_after_countdown_restarts(tmp_path):
    import os
    import subprocess

    from qqtools.plugins.qexp.agent.helpers import _pid_start_time_ticks, _publish_process_status
    from qqtools.plugins.qexp.runtime.resources.reservations import release

    runtime = _runtime(tmp_path)
    set_idle_job_config(runtime, idle_delay_seconds=60)
    clock = [0]
    supervisor = IdleJobSupervisor(runtime, monotonic=lambda: clock[0])
    assert supervisor.tick(gpu_policy=POLICY)["reason"] == "idle_delay"
    assert idle_job_status(runtime)["interval_detail"]["last_reset"] is None
    clock[0] = 59
    reservation = reserve(runtime.root, "training", [0])["reservation"]
    release(runtime.root, reservation["reservation_id"])
    before = time.time()
    assert supervisor.tick(gpu_policy=POLICY)["reason"] == "idle_delay"
    after = time.time()
    detail = idle_job_status(runtime)["interval_detail"]
    assert detail["idle_elapsed_seconds"] == 0
    reset = detail["last_reset"]
    assert reset["check"] == "demand_revision_changed"
    assert reset["elapsed_seconds"] == 59
    assert before <= datetime.fromisoformat(reset["reset_at"]).timestamp() <= after

    clock[0] += 1
    assert supervisor.tick(gpu_policy=POLICY)["reason"] == "idle_delay"
    assert idle_job_status(runtime)["interval_detail"]["idle_elapsed_seconds"] == 1
    # The process driving the supervisor publishes its real identity so the CLI
    # can distinguish a live idle_delay decision from a stopped agent snapshot.
    pid = os.getpid()
    runtime.paths["pid"].write_text(str(pid), encoding="utf-8")
    _publish_process_status(
        runtime,
        instance_id="interval-test",
        pid=pid,
        start_ticks=_pid_start_time_ticks(pid),
        waiting_for_first_registration=True,
    )
    output = subprocess.run(
        [
            sys.executable,
            "-c",
            "from qqtools.plugins.qexp.cli.entrypoint import main; raise SystemExit(main())",
            "agent",
            "status",
            "--machine-runtime-root",
            str(runtime.root),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout
    for value in ("idle_delay", "demand_revision_changed", "59", reset["reset_at"]):
        assert value in output


@pytest.mark.parametrize("running", [False, True], ids=["countdown", "running"])
def test_decisive_check_survives_optional_write_failure_and_recovery(tmp_path, monkeypatch, running):
    from qqtools.plugins.qexp.agent import idle_job as module
    from qqtools.plugins.qexp.agent.idle_job_diagnostics import build_demand_observation

    runtime = _runtime(tmp_path)
    set_idle_job_config(runtime, idle_delay_seconds=60)
    clock = [0]
    supervisor = IdleJobSupervisor(runtime, monotonic=lambda: clock[0])
    try:
        supervisor.tick(gpu_policy=POLICY)
        clock[0] = 60 if running else 59
        if running:
            _until(supervisor, lambda status: status["state"] == "running")
        supervisor.set_demand_observation(
            "unknown", build_demand_observation("launch_handoff_pending"), agent_instance_id="agent-a"
        )
        write = module.atomic_replace
        with monkeypatch.context() as patch:

            def fail_enriched(path, record, **kwargs):
                if "idle_job_interval_detail" in record:
                    raise OSError("optional status unavailable")
                return write(path, record, **kwargs)

            patch.setattr(module, "atomic_replace", fail_enriched)
            status = supervisor.tick(demand="unknown", gpu_policy=POLICY)
            assert status["state"] == ("stopping" if running else "waiting")
            assert status["reason"] == "unknown_demand"
            assert module.idle_job_status(runtime)["interval_detail"] is None
        clock[0] += 10
        supervisor.tick(demand="unknown", gpu_policy=POLICY)
        reset = module.idle_job_status(runtime)["interval_detail"]["last_reset"]
        assert reset["check"] == "launch_handoff_pending"
        assert reset["elapsed_seconds"] == (60 if running else 59)
        _until(supervisor, lambda status: status["reason"] == "idle_delay")
        assert module.idle_job_status(runtime)["interval_detail"]["last_reset"] == reset
        restarted = IdleJobSupervisor(runtime, monotonic=lambda: clock[0])
        assert restarted.tick(gpu_policy=POLICY)["reason"] == "idle_delay"
        assert module.idle_job_status(runtime)["interval_detail"]["last_reset"] is None
    finally:
        assert supervisor.shutdown()


@pytest.mark.parametrize("established", [False, True])
def test_held_reservation_cannot_create_repeated_interval_resets(tmp_path, established):
    runtime = _runtime(tmp_path)
    set_idle_job_config(runtime, idle_delay_seconds=60)
    clock = [0]
    supervisor = IdleJobSupervisor(runtime, monotonic=lambda: clock[0], wall_clock=lambda: 1700000000 + clock[0])
    if established:
        assert supervisor.tick(gpu_policy=POLICY)["reason"] == "idle_delay"
    clock[0] = 59
    reserve(runtime.root, "training", [0])
    assert supervisor.tick(gpu_policy=POLICY)["reason"] == "gpu_reserved"
    detail = idle_job_status(runtime)["interval_detail"] or {}
    if established:
        assert detail["last_reset"]["elapsed_seconds"] == 59
        assert detail["last_reset"]["check"] == "demand_revision_changed"
    else:
        assert detail.get("last_reset") is None
    clock[0] += 2
    assert supervisor.tick(gpu_policy=POLICY)["reason"] == "gpu_reserved"
    assert (idle_job_status(runtime)["interval_detail"] or {}) == detail


def _pause_controller_for_crash(root, stage, reached):
    from qqtools.plugins.qexp.agent import idle_job as idle

    runtime = MachineRuntime(root)
    clock = [0]
    supervisor = IdleJobSupervisor(runtime, monotonic=lambda: clock[0])

    def pause(*args, **kwargs):
        reached.set()
        time.sleep(30)
        raise AssertionError("crash barrier timed out")

    if stage == "prepared":
        supervisor._spawn_prepared = pause
    elif stage == "intent":
        idle.write_idle_job_ownership = pause
    else:
        idle.send_idle_launch_permit = pause
    while True:
        supervisor.tick(gpu_policy=POLICY)
        clock[0] = 90
        time.sleep(0.01)


@pytest.mark.parametrize("stage", ["intent", "prepared", "authorized"])
def test_controller_crash_cannot_replay_a_launch(tmp_path, stage):
    import multiprocessing

    runtime = _runtime(tmp_path)
    marker = tmp_path / "ran"
    set_idle_job_config(runtime, command=f"{sys.executable} -c \"from pathlib import Path; Path('{marker}').touch()\"")
    set_idle_job_config(runtime, idle_delay_seconds=90)
    ctx = multiprocessing.get_context("spawn")
    reached = ctx.Event()
    controller = ctx.Process(target=_pause_controller_for_crash, args=(runtime.root, stage, reached))
    controller.start()
    try:
        assert reached.wait(8)
        owner = read_idle_job_ownership(runtime.root)
        assert (owner is None) == (stage == "intent")
        if owner:
            assert owner["phase"] == "prepared"
        assert not marker.exists()
    finally:
        controller.kill()
        controller.join(5)
        assert not controller.is_alive()

    recovered = IdleJobSupervisor(runtime)
    try:
        # Give recovery idle demand, with a fresh long interval so a new
        # instance cannot obscure replay of the original.
        _until(recovered, lambda _: read_idle_job_ownership(runtime.root) is None)
        assert not marker.exists()
        assert reserve(runtime.root, "training", [0])["reservation"]["task_id"] == "training"
    finally:
        assert recovered.shutdown()


@pytest.mark.parametrize("corrupt", ["intent", "exit"])
def test_prepared_recovery_retains_mismatched_evidence(tmp_path, monkeypatch, corrupt):
    from qqtools.plugins.qexp.agent import idle_job
    from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json

    runtime = _runtime(tmp_path)
    supervisor = IdleJobSupervisor(runtime)

    def crash(_preparation):
        raise KeyboardInterrupt()

    monkeypatch.setattr(supervisor, "_spawn_prepared", crash)
    with pytest.raises(KeyboardInterrupt):
        supervisor.tick(gpu_policy=POLICY)
    owner = read_idle_job_ownership(runtime.root)
    directory = idle_job_instance_path(runtime.root, owner["instance_id"])
    path = directory / "intent.json"
    original = read_json(path)
    if corrupt == "intent":
        import copy

        value = copy.deepcopy(original)
        value["idle_job_intent"].update(instance_id="f" * 32, parent_pid=1, parent_start_ticks=0)
        atomic_replace(path, value)
    else:
        atomic_replace(
            directory / "exit.json",
            {
                "idle_job_exit": {
                    "version": 1,
                    "runtime_id": owner["runtime_id"],
                    "instance_id": "f" * 32,
                    "boot_id": owner["boot_id"],
                    "returncode": None,
                    "exited_at": idle_job.utc_now(),
                    "reason": "launch_failed",
                }
            },
        )
        monkeypatch.setattr(idle_job, "idle_job_processes_exited", lambda path: True)
    try:
        restarted = IdleJobSupervisor(runtime)
        assert restarted.tick(gpu_policy=POLICY)["state"] == "blocked"
        assert read_idle_job_ownership(runtime.root)["instance_id"] == owner["instance_id"]
        with pytest.raises(IdleJobAdmissionBlocked):
            reserve(runtime.root, "training", [0])
    finally:
        # No helper was started. Restore the unmodified record so the original
        # controller can seal its synchronously interrupted preparation.
        atomic_replace(path, original)
        (directory / "exit.json").unlink(missing_ok=True)
        supervisor._preparation.spawn_failed = True
        assert supervisor.shutdown()


def test_instance_parent_directory_is_synced_before_prepared_owner(tmp_path, monkeypatch):
    import os

    from qqtools.plugins.qexp.agent import idle_job

    runtime = _runtime(tmp_path)
    supervisor = IdleJobSupervisor(runtime)
    instances = runtime.root / "idle-job" / "instances"
    sync = os.fsync
    write = idle_job.write_idle_job_ownership
    observed = []
    owners = []

    def capture(fd):
        if instances.is_dir() and os.path.samestat(os.fstat(fd), instances.stat()):
            observed.append(True)
        return sync(fd)

    def publish(root, owner):
        if owner["phase"] == "prepared":
            assert observed
            owners.append(owner["instance_id"])
        return write(root, owner)

    monkeypatch.setattr(os, "fsync", capture)
    monkeypatch.setattr(idle_job, "write_idle_job_ownership", publish)
    try:
        assert supervisor.tick(gpu_policy=POLICY)["reason"] == "preparing"
        assert len(owners) == 1
    finally:
        assert supervisor.shutdown()


def _contending_training(root, lane, channel):
    def admit():
        return reserve(root, "competing-training", [0]) if lane == "gpu" else reserve_cpu(root, "competing-training", 1)

    with channel:
        channel.send(("attempt", time.monotonic()))
        try:
            admit()
        except IdleJobAdmissionBlocked:
            channel.send(("revoked", time.monotonic()))
        else:
            raise AssertionError("training acquired an unconfirmed owner")
        assert channel.recv() == "retry"
        with pytest.raises(IdleJobAdmissionBlocked):
            admit()
        channel.send("still_blocked")
        assert channel.recv() == "cleaned"
        assert admit()["reservation"]["task_id"] == "competing-training"
        channel.send(("reserved", time.monotonic()))


@pytest.mark.parametrize("lane", ["cpu", "gpu"])
def test_local_training_waits_for_slow_prepared_publication_and_cleanup(tmp_path, monkeypatch, lane, record_property):
    import multiprocessing

    from qqtools.plugins.qexp.runtime.resources import idle_job as admission

    runtime = _runtime(tmp_path)
    set_cpu_lane_capacity(runtime.root, capacity=1)
    supervisor = IdleJobSupervisor(runtime)
    ctx = multiprocessing.get_context("spawn")
    client, server = ctx.Pipe()
    child = ctx.Process(target=_contending_training, args=(runtime.root, lane, server))
    write = admission.atomic_replace
    timeline = {}

    def publish(path, value):
        def slow_disk(step):
            if step != "directory_fsync" or timeline or path.name != "ownership.json":
                return
            child.start()
            assert client.poll(5)
            event, timeline["attempt"] = client.recv()
            assert event == "attempt"
            assert not client.poll(0.12)
            timeline["disk_resumed"] = time.monotonic()

        return write(path, value, io_step_observer=slow_disk)

    try:
        with monkeypatch.context() as inject:
            inject.setattr(admission, "atomic_replace", publish)
            supervisor.tick(gpu_policy=POLICY)
        assert client.poll(5)
        event, timeline["revoked"] = client.recv()
        assert event == "revoked"
        assert read_idle_job_ownership(runtime.root)["phase"] == "stopping"
        client.send("retry")
        assert client.poll(3) and client.recv() == "still_blocked"
        _until(supervisor, lambda _: read_idle_job_ownership(runtime.root) is None, demand="busy")
        timeline["cleaned"] = time.monotonic()
        client.send("cleaned")
        assert client.poll(5)
        event, timeline["reserved"] = client.recv()
        assert event == "reserved"
        child.join(3)
        assert child.exitcode == 0
        assert list(timeline.values()) == sorted(timeline.values())
        record_property("attempt_to_revoked_seconds", timeline["revoked"] - timeline["attempt"])
        record_property("revoked_to_cleaned_seconds", timeline["cleaned"] - timeline["revoked"])
    finally:
        if child.is_alive():
            child.kill()
            child.join(3)
        client.close()
        server.close()
        assert supervisor.shutdown()


def test_ambiguous_owner_publication_retains_cleanup_until_verified(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.agent import idle_job
    from qqtools.plugins.qexp.runtime.resources import idle_job as admission

    runtime = _runtime(tmp_path)
    supervisor = IdleJobSupervisor(runtime)
    replace = admission.atomic_replace
    seen = []

    def ambiguous(path, value):
        def fail(stage):
            if stage == "directory_fsync" and path.name == "ownership.json":
                owner = read_idle_job_ownership(runtime.root)
                assert owner["phase"] == "prepared"
                seen.append(owner["instance_id"])
                raise OSError("cannot confirm directory persistence")

        return replace(path, value, io_step_observer=fail)

    try:
        with monkeypatch.context() as delayed:
            delayed.setattr(admission, "atomic_replace", ambiguous)
            delayed.setattr(idle_job, "spawn_idle_job", lambda *a, **kw: pytest.fail("spawn after unknown write"))
            delayed.setattr(supervisor, "_seal_unstarted_locked", lambda prep: False)
            assert supervisor.tick(gpu_policy=POLICY)["reason"] == "cleanup_uncertain"
        assert seen == [read_idle_job_ownership(runtime.root)["instance_id"]]
        with pytest.raises(IdleJobAdmissionBlocked):
            reserve(runtime.root, "training", [0])
        _until(supervisor, lambda _: read_idle_job_ownership(runtime.root) is None, demand="busy")
        assert reserve(runtime.root, "training", [0])["reservation"]["task_id"] == "training"
    finally:
        assert supervisor.shutdown()


def test_expired_gpu_provisional_allows_idle_and_late_attach_revokes_it(tmp_path):
    from qqtools.plugins.qexp.runtime.paths import local_paths
    from qqtools.plugins.qexp.runtime.resources.idle_job import read_idle_job_demand
    from qqtools.plugins.qexp.runtime.resources.reservations import attach
    from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json

    runtime = _runtime(tmp_path)
    supervisor = IdleJobSupervisor(runtime)
    record = reserve(runtime.root, "training", [0])["reservation"]
    reservation_id = record["reservation_id"]
    path = local_paths(runtime.root)["provisional"] / f"{reservation_id}.json"
    value = read_json(path)
    value["reservation"]["expires_at"] = "2000-01-01T00:00:00Z"
    atomic_replace(path, value)
    try:
        _until(supervisor, lambda status: status["state"] == "running")
        assert path.exists()
        before = read_idle_job_demand(runtime.root)
        with pytest.raises(IdleJobAdmissionBlocked):
            attach(runtime.root, reservation_id, "attempt", 1)
        assert read_idle_job_demand(runtime.root) == before + 1
        assert read_idle_job_ownership(runtime.root)["phase"] == "stopping"
        assert not reservation_snapshot(runtime.root).active
        assert path.exists()
        _until(supervisor, lambda status: read_idle_job_ownership(runtime.root) is None, demand="busy")
        attach(runtime.root, reservation_id, "attempt", 1)
        assert reservation_snapshot(runtime.root).active[0]["attempt_id"] == "attempt"
    finally:
        assert supervisor.shutdown()
