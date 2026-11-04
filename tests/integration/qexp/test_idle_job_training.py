"""Normal Project dispatch preempts machine-only Idle Job before training."""

import shlex
import sys
import time

import pytest

from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.idle_job_config import set_idle_job_config
from qqtools.plugins.qexp.agent.idle_job_process import idle_job_processes_exited
from qqtools.plugins.qexp.agent.lifecycle import get_machine_agent_status, start_machine_agent, stop_machine_agent
from qqtools.plugins.qexp.agent.readiness import capture_readiness_snapshot, wait_for_readiness
from qqtools.plugins.qexp.agent.setup import initialize_machine, initialize_project, register_projects
from qqtools.plugins.qexp.commands.task import submit
from qqtools.plugins.qexp.runtime.resources.cpu_lane import set_cpu_lane_capacity
from qqtools.plugins.qexp.runtime.resources.idle_job import idle_job_instance_path, read_idle_job_ownership

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _running_owner(runtime):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if get_machine_agent_status(runtime)["idle_job"]["state"] == "running":
            return read_idle_job_ownership(runtime.root)
        time.sleep(0.05)
    pytest.fail(f"Idle Job did not start: {get_machine_agent_status(runtime)}")


@pytest.mark.parametrize("lane", ["gpu", "cpu"])
def test_later_registered_projects_preempt_idle_job_through_dispatch(tmp_path, lane):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "test-machine")
    set_cpu_lane_capacity(runtime.root, capacity=1)
    set_idle_job_config(
        runtime,
        enabled=True,
        command=shlex.join([sys.executable, "-c", "import time;time.sleep(90)"]),
        idle_delay_seconds=0,
        stop_timeout_seconds=0.2,
    )
    process = start_machine_agent(runtime, available_gpus=[0, 3], loop_interval=0.05)
    try:
        first_owner = _running_owner(runtime)
        first_directory = idle_job_instance_path(runtime.root, first_owner["instance_id"])
        projects = [tmp_path / "first", tmp_path / "second"]
        for project in projects:
            initialize_project(project)
        result = register_projects(runtime, projects)
        assert all(item["status"] == "registered" for item in result["projects"])
        ready = wait_for_readiness(runtime, capture_readiness_snapshot(runtime), timeout_seconds=30)
        assert ready["ready"], ready
        owner = _running_owner(runtime)
        directory = idle_job_instance_path(runtime.root, owner["instance_id"])
        cfg = runtime.load_registry()[1][1].root_config()
        marker = tmp_path / "training-started"
        script = (
            "from pathlib import Path;"
            "from qqtools.plugins.qexp.agent.idle_job_process import idle_job_processes_exited;"
            f"assert idle_job_processes_exited(Path({str(directory)!r}));"
            f"Path({str(marker)!r}).write_text('reclaimed');"
        )
        task = submit(
            cfg,
            [sys.executable, "-c", script],
            requested_gpus=1 if lane == "gpu" else 0,
            requested_cpus=1 if lane == "cpu" else None,
            working_dir=tmp_path,
            tmux_override=False,
        )
        deadline = time.monotonic() + 30
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.exists(), f"Training {task.task_id} failed to start: {get_machine_agent_status(runtime)}"
        assert marker.read_text() == "reclaimed"
        assert idle_job_processes_exited(directory)
        assert idle_job_processes_exited(first_directory)
    finally:
        stop_machine_agent(runtime)
        process.wait(timeout=5)
