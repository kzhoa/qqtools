from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.machine_config import init_shared_root
from qqtools.plugins.qexp.notification_config import (
    shared_feishu_webhook_path,
    update_notifications,
    write_shared_feishu_webhook,
)
from qqtools.plugins.qexp.notifications import NotificationHook, notification_runtime

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def test_legacy_shared_file_imports_into_owner_private_webhook(tmp_path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.ensure_binding(cfg.shared_root, cfg.machine_name)
    update_notifications(
        cfg,
        lambda current: {
            **current,
            "enabled": True,
            "providers": {
                "feishu": {
                    "enabled": True,
                    "credential_source": "shared_file",
                    "webhook_env": "UNUSED_WEBHOOK_ENV",
                    "secret_env": None,
                    "timeout_seconds": 5,
                }
            },
        },
    )
    webhook = "https://open.feishu.cn/open-apis/bot/v2/hook/shared-webhook"
    write_shared_feishu_webhook(cfg, webhook)
    calls = []

    class Notifier:
        name = "feishu"

        def send(self, event, *, webhook, secret, timeout_seconds):
            calls.append((webhook, secret, timeout_seconds))
            return {"http_status": 200, "business_code": "0"}

    event = SimpleNamespace(
        phase="failed",
        task_id="task-a",
        attempt_id="attempt-a",
        reason="nonzero_exit",
        exit_code=None,
        execution_machine_name="gpu-a",
        dispatching_machine_name="gpu-b",
        finished_at="2026-08-07T00:00:00Z",
        execution_started_at=None,
        duration_ms=None,
    )
    with notification_runtime(runtime.root):
        NotificationHook(registry={"feishu": Notifier()}).handle(cfg, event)

    assert calls == [(webhook, None, 5)]
    assert shared_feishu_webhook_path(cfg).exists()
