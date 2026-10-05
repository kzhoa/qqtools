import pytest

from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.setup import initialize_machine, register_projects
from qqtools.plugins.qexp.cli.entrypoint import main
from qqtools.plugins.qexp.machine_config import init_shared_root
from qqtools.plugins.qexp.notification_policy import load_policy

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def test_shared_file_notification_options_are_removed(tmp_path, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    machine_runtime_root = cfg.runtime_root.parent / "machine-runtime"
    runtime = MachineRuntime(machine_runtime_root)
    initialize_machine(runtime, "gpu-1")
    register_projects(runtime, [cfg.shared_root], machine_name="gpu-1")
    arguments = [
        "--project",
        str(cfg.shared_root),
        "--runtime-root",
        str(cfg.runtime_root),
        "--machine",
        cfg.machine_name,
        "--machine-runtime-root",
        str(machine_runtime_root),
        "config",
        "set",
        "notifications",
        "--provider",
        "feishu",
        "--credential-source",
        "shared_file",
    ]

    with pytest.raises(SystemExit, match="2"):
        main(arguments)
    captured = capsys.readouterr()
    assert "unrecognized arguments" in captured.err
    assert "shared_file" in captured.err


def test_secret_env_can_be_cleared_alone_and_with_another_edit(tmp_path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    runtime_root = tmp_path / "machine-runtime"
    runtime = MachineRuntime(runtime_root)
    initialize_machine(runtime, "gpu-1")
    register_projects(runtime, [cfg.shared_root], machine_name="gpu-1")
    command = [
        "--project",
        str(cfg.shared_root),
        "--runtime-root",
        str(cfg.runtime_root),
        "--machine",
        cfg.machine_name,
        "--machine-runtime-root",
        str(runtime_root),
        "config",
        "set",
        "notifications",
        "--provider",
        "feishu",
    ]

    assert main(command + ["--webhook-env", "FEISHU_WEBHOOK", "--secret-env", "FEISHU_SIGNING_SECRET"]) == 0
    project_id = MachineRuntime(runtime_root).resolve_project_binding(cfg.shared_root).project_id
    assert main(command + ["--unset-secret-env"]) == 0
    assert load_policy(runtime_root, "project", project_id)["override"]["destination"]["signing"] == "unsigned"

    assert main(command + ["--secret-env", "FEISHU_SIGNING_SECRET"]) == 0
    assert main(command + ["--unset-secret-env", "--enabled"]) == 0
    override = load_policy(runtime_root, "project", project_id)["override"]
    assert override["destination"]["signing"] == "unsigned"
    assert override["enabled"] is True

    assert main(command + ["--secret-env", "FEISHU_SIGNING_SECRET", "--unset-secret-env"]) == 2
    assert load_policy(runtime_root, "project", project_id)["override"] == override
