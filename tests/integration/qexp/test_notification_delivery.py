from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp.machine_config import init_shared_root
from qqtools.plugins.qexp.notification_config import (
    shared_feishu_webhook_path,
    update_notifications,
    write_shared_feishu_webhook,
)
from qqtools.plugins.qexp.notifications import NotificationHook

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]

def test_shared_file_credential_source_uses_owner_private_webhook(tmp_path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
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
    write_shared_feishu_webhook(cfg, "https://example.invalid/shared-webhook")
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
    NotificationHook(registry={"feishu": Notifier()}).handle(cfg, event)

    assert calls == [("https://example.invalid/shared-webhook", None, 5)]
    assert shared_feishu_webhook_path(cfg).exists()
