import io

import pytest

from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.cli.entrypoint import main
from qqtools.plugins.qexp.machine_config import init_shared_root
from qqtools.plugins.qexp.notification_config import shared_feishu_webhook_path
from qqtools.plugins.qexp.notification_credentials import credential_path
from qqtools.plugins.qexp.notification_policy import load_policy

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def test_shared_file_webhook_cli_requires_acknowledgement(tmp_path, monkeypatch, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    machine_runtime_root = cfg.runtime_root.parent / "machine-runtime"
    MachineRuntime(machine_runtime_root).ensure_binding(cfg.shared_root, cfg.machine_name)
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

    assert main(arguments) == 2
    assert not shared_feishu_webhook_path(cfg).exists()

    webhook = "https://open.feishu.cn/open-apis/bot/v2/hook/persisted-webhook"
    monkeypatch.setattr("sys.stdin", io.StringIO(webhook + "\n"))
    assert (
        main(
            arguments
            + [
                "--webhook-stdin",
                "--acknowledge-shared-secret-risk",
                "--timeout-seconds",
                "31",
            ]
        )
        == 2
    )
    assert not shared_feishu_webhook_path(cfg).exists()

    monkeypatch.setattr("sys.stdin", io.StringIO(webhook + "\n"))
    assert main(arguments + ["--webhook-stdin", "--acknowledge-shared-secret-risk"]) == 0

    assert not shared_feishu_webhook_path(cfg).exists()
    binding = MachineRuntime(machine_runtime_root).resolve_project_binding(cfg.shared_root)
    destination = load_policy(machine_runtime_root, "project", binding.project_id)["override"]["destination"]
    assert credential_path(machine_runtime_root, destination["credential_id"]).exists()
    assert main(arguments[:8] + ["config", "show", "notifications"]) == 0
    assert webhook not in capsys.readouterr().out


def test_secret_env_can_be_cleared_alone_and_with_another_edit(tmp_path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    runtime_root = tmp_path / "machine-runtime"
    MachineRuntime(runtime_root).ensure_binding(cfg.shared_root, cfg.machine_name)
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
