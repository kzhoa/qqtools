"""Public Idle Job configuration and agent-mode conflict contracts."""

import json
import sys

import pytest

from qqtools.plugins.qexp.agent.config import initialize_agent_config, load_agent_config, set_agent_config
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.idle_job_config import load_idle_job_config, set_idle_job_config
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.cli.entrypoint import main

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def test_cli_show_set_disable_preserves_configuration(tmp_path, capsys):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "test-machine")
    prefix = ["--machine-runtime-root", str(runtime.root), "agent", "config", "idle-job"]
    assert main([*prefix, "show", "--format=json"]) == 0
    assert json.loads(capsys.readouterr().out)["idle_job"]["enabled"] is False
    command = f"{sys.executable} /opt/jobs/occupy.py"
    assert (
        main(
            [
                *prefix,
                "set",
                "--enable",
                "--command",
                command,
                "--gpu-ids=1,3",
                "--working-directory",
                str(tmp_path),
                "--idle-delay-seconds",
                "0.5",
                "--format=json",
            ]
        )
        == 0
    )
    configured = json.loads(capsys.readouterr().out)["idle_job"]
    assert configured["enabled"] is True
    assert configured["gpu_ids"] == [1, 3]
    assert configured["command"] == command
    assert main([*prefix, "disable", "--format=json"]) == 0
    disabled = json.loads(capsys.readouterr().out)["idle_job"]
    assert disabled["enabled"] is False
    assert disabled["command"] == command
    assert disabled["revision"] > configured["revision"]
    assert load_idle_job_config(runtime).enabled is False


@pytest.mark.parametrize("mutator", [set_agent_config, initialize_agent_config])
def test_enabled_idle_job_prevents_mode_change_atomically(tmp_path, mutator):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "test-machine")
    set_idle_job_config(runtime, enabled=True, command=f"{sys.executable} job.py")
    before = load_agent_config(runtime)
    with pytest.raises(ValueError, match="daemon"):
        mutator(runtime, name="test-machine", agent_mode="on_demand")
    assert load_agent_config(runtime) == before
    set_idle_job_config(runtime, enabled=False)
    mutator(runtime, name="test-machine", agent_mode="on_demand")
    assert load_agent_config(runtime).agent_mode == "on_demand"


def test_on_demand_enablement_fails_without_mutating_either_config(tmp_path, capsys):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "test-machine", agent_mode="on_demand")
    before = load_agent_config(runtime)
    assert (
        main(
            [
                "--machine-runtime-root",
                str(runtime.root),
                "agent",
                "config",
                "idle-job",
                "set",
                "--enable",
                "--command",
                f"{sys.executable} job.py",
                "--format=json",
            ]
        )
        == 2
    )
    assert "daemon" in capsys.readouterr().out.lower()
    assert not load_idle_job_config(runtime).enabled
    assert load_agent_config(runtime) == before


def test_cli_reports_gpu_environment_conflict(tmp_path, capsys):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "test-machine")
    assert (
        main(
            [
                "--machine-runtime-root",
                str(runtime.root),
                "agent",
                "config",
                "idle-job",
                "set",
                "--command",
                "env CUDA_VISIBLE_DEVICES=0 python job.py",
                "--format=json",
            ]
        )
        == 2
    )
    assert "gpu_ids" in capsys.readouterr().out
    assert load_idle_job_config(runtime).revision == 0


def test_missing_working_directory_can_be_disabled_and_repaired(tmp_path):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "test-machine")
    directory = tmp_path / "working"
    directory.mkdir()
    set_idle_job_config(runtime, enabled=True, command=f"{sys.executable} job.py", working_directory=str(directory))
    directory.rmdir()
    assert not set_idle_job_config(runtime, enabled=False).enabled
    repaired = set_idle_job_config(runtime, working_directory=str(tmp_path), enabled=True)
    assert repaired.working_directory == str(tmp_path)
    assert repaired.enabled
