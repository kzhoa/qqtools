import io

import pytest

from qqtools.plugins.qexp import cli
from qqtools.plugins.qexp.machine_config import init_shared_root
from qqtools.plugins.qexp.machine_runtime import MachineRuntime
from qqtools.plugins.qexp.notification_config import shared_feishu_webhook_path

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]

def test_shared_file_webhook_cli_requires_acknowledgement(tmp_path, monkeypatch, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    machine_runtime_root = cfg.runtime_root.parent / "machine-runtime"
    MachineRuntime(machine_runtime_root).ensure_binding(cfg.shared_root, cfg.machine_name)
    arguments = [
        "--shared-root", str(cfg.shared_root), "--runtime-root", str(cfg.runtime_root),
        "--machine", cfg.machine_name, "--machine-runtime-root", str(machine_runtime_root),
        "config", "notifications", "provider", "set", "feishu",
        "--credential-source", "shared_file",
    ]

    assert cli.main(arguments) == 2
    assert not shared_feishu_webhook_path(cfg).exists()

    monkeypatch.setattr("sys.stdin", io.StringIO("https://example.invalid/persisted-webhook\n"))
    assert cli.main(arguments + [
        "--webhook-stdin", "--acknowledge-shared-secret-risk", "--timeout-seconds", "31",
    ]) == 2
    assert not shared_feishu_webhook_path(cfg).exists()

    monkeypatch.setattr("sys.stdin", io.StringIO("https://example.invalid/persisted-webhook\n"))
    assert cli.main(arguments + ["--webhook-stdin", "--acknowledge-shared-secret-risk"]) == 0

    assert shared_feishu_webhook_path(cfg).exists()
    assert cli.main(arguments[:8] + ["config", "notifications", "show"]) == 0
    assert "https://example.invalid/persisted-webhook" not in capsys.readouterr().out
