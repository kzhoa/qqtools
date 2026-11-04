"""Idle Job is supervised by a ready machine-only daemon and reclaimed on stop."""

import sys
import time

import pytest

from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.idle_job_config import set_idle_job_config
from qqtools.plugins.qexp.agent.idle_job_process import idle_job_processes_exited
from qqtools.plugins.qexp.agent.lifecycle import get_machine_agent_status, start_machine_agent, stop_machine_agent
from qqtools.plugins.qexp.agent.readiness import capture_readiness_snapshot, wait_for_readiness
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.runtime.resources.idle_job import idle_job_instance_path, read_idle_job_ownership
from qqtools.plugins.qexp.runtime.resources.reservations import reservation_snapshot

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


@pytest.mark.parametrize("partition_count", [0, 65])
def test_machine_only_ready_does_not_wait_for_idle_delay_and_stop_reclaims_job(tmp_path, partition_count):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "test-machine")
    for index in range(partition_count):
        (runtime.paths["projects"] / f"empty-{index}").mkdir(parents=True)
    set_idle_job_config(
        runtime,
        enabled=True,
        command=f'{sys.executable} -c "import time;print(123,flush=True);time.sleep(60)"',
        idle_delay_seconds=300,
        stop_timeout_seconds=0.2,
    )
    process = start_machine_agent(runtime, available_gpus=[0, 3], loop_interval=0.05)
    directory = None
    try:
        ready = wait_for_readiness(runtime, capture_readiness_snapshot(runtime), timeout_seconds=20)
        assert ready["ready"], ready
        status = get_machine_agent_status(runtime)
        assert status["idle_job"]["supervision_active"] is True
        assert status["idle_job"]["state"] == "waiting"
        assert read_idle_job_ownership(runtime.root) is None
        set_idle_job_config(runtime, idle_delay_seconds=1)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            status = get_machine_agent_status(runtime)
            if status["idle_job"]["state"] == "running":
                break
            time.sleep(0.05)
        assert status["idle_job"]["state"] == "running", status["idle_job"]
        owner = read_idle_job_ownership(runtime.root)
        directory = idle_job_instance_path(runtime.root, owner["instance_id"])
        assert owner["gpu_ids"] == [3]
        assert runtime.load_registry()[1] == []
        assert not reservation_snapshot(runtime.root).reservations
        assert stop_machine_agent(runtime)
        process.wait(timeout=5)
        assert idle_job_processes_exited(directory)
        assert read_idle_job_ownership(runtime.root) is None
    finally:
        stop_machine_agent(runtime)
        process.wait(timeout=5)
        if directory is not None:
            assert idle_job_processes_exited(directory)


def test_default_stop_waits_for_forced_idle_job_cleanup(tmp_path):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "test-machine")
    marker = tmp_path / "command-ready"
    script = (
        "import signal,time,pathlib;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        f"pathlib.Path({str(marker)!r}).touch();time.sleep(90)"
    )
    import shlex

    set_idle_job_config(runtime, enabled=True, command=shlex.join([sys.executable, "-c", script]), idle_delay_seconds=0)
    process = start_machine_agent(runtime, available_gpus=[0], loop_interval=0.05)
    try:
        deadline = time.monotonic() + 15
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.exists()
        owner = read_idle_job_ownership(runtime.root)
        directory = idle_job_instance_path(runtime.root, owner["instance_id"])
        assert stop_machine_agent(runtime)
        process.wait(timeout=5)
        assert idle_job_processes_exited(directory)
        assert read_idle_job_ownership(runtime.root) is None
    finally:
        stop_machine_agent(runtime)
        process.wait(timeout=5)
