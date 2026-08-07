from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp.machine_config import init_shared_root
from qqtools.plugins.qexp.notification_config import (shared_feishu_webhook_path,
                                                       update_notifications,
                                                       validate_notifications,
                                                       write_shared_feishu_webhook)
from qqtools.plugins.qexp.notifications.feishu import FeishuNotifier, NotificationTransportError
from qqtools.plugins.qexp.notifications import NotificationHook, notification_key


def _event(**overrides):
    values = {"phase": "failed", "task_id": "task-a", "attempt_id": "attempt-a",
              "reason": "nonzero_exit", "exit_code": None,
              "execution_machine_name": "gpu-a", "dispatching_machine_name": "gpu-b",
              "finished_at": "2026-08-07T00:00:00Z"}
    values.update(overrides)
    return SimpleNamespace(**values)


def test_notification_key_is_stable_and_phase_specific():
    assert notification_key("feishu", _event()) == notification_key("feishu", _event())
    assert notification_key("feishu", _event(phase="succeeded")) != notification_key("feishu", _event())


@pytest.mark.parametrize("timeout, valid", [(True, False), (0.49, False), (0.5, True), (30, True), (30.01, False)])
def test_feishu_timeout_validation(timeout, valid):
    value = {"enabled": True, "providers": {"feishu": {"enabled": True, "timeout_seconds": timeout}}}
    if valid:
        assert validate_notifications(value)["providers"]["feishu"]["timeout_seconds"] == timeout
    else:
        with pytest.raises(ValueError):
            validate_notifications(value)


def test_shared_file_credential_source_uses_owner_private_webhook(tmp_path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    update_notifications(cfg, lambda current: {
        **current,
        "enabled": True,
        "providers": {"feishu": {
            "enabled": True,
            "credential_source": "shared_file",
            "webhook_env": "UNUSED_WEBHOOK_ENV",
            "secret_env": None,
            "timeout_seconds": 5,
        }},
    })
    write_shared_feishu_webhook(cfg, "https://example.invalid/shared-webhook")
    calls = []

    class Notifier:
        name = "feishu"

        def send(self, event, *, webhook, secret, timeout_seconds):
            calls.append((webhook, secret, timeout_seconds))
            return {"http_status": 200, "business_code": "0"}

    NotificationHook(registry={"feishu": Notifier()}).handle(cfg, _event())

    assert calls == [("https://example.invalid/shared-webhook", None, 5)]
    assert shared_feishu_webhook_path(cfg).exists()


def test_feishu_payload_and_business_success():
    seen = {}

    class Response:
        status = 200

        def read(self):
            return b'{"code": 0}'

    def urlopen(request, timeout):
        seen.update(data=request.data, timeout=timeout)
        return Response()

    result = FeishuNotifier(clock=lambda: 100, urlopen=urlopen).send(
        _event(), webhook="https://example.invalid/hook", secret="secret", timeout_seconds=5
    )
    assert result == {"http_status": 200, "business_code": "0"}
    assert b"qexp task failed" in seen["data"]
    assert b"secret" not in seen["data"]


def test_feishu_rejects_boolean_business_code():
    class Response:
        status = 200

        def read(self):
            return b'{"code": false}'

    with pytest.raises(NotificationTransportError) as error:
        FeishuNotifier(urlopen=lambda request, timeout: Response()).send(
            _event(), webhook="https://example.invalid/hook", secret=None, timeout_seconds=5
        )
    assert error.value.reason_code == "invalid_response"


def test_feishu_rejects_conflicting_business_codes_as_invalid_response():
    class Response:
        status = 200

        def read(self):
            return b'{"code": 0, "StatusCode": 1}'

    with pytest.raises(NotificationTransportError) as error:
        FeishuNotifier(urlopen=lambda request, timeout: Response()).send(
            _event(), webhook="https://example.invalid/hook", secret=None, timeout_seconds=5
        )

    assert error.value.reason_code == "invalid_response"
    assert error.value.error_type == "invalid_response"


def test_malformed_unknown_provider_does_not_abort_dispatch(monkeypatch):
    cfg = SimpleNamespace()
    diagnostics = []

    monkeypatch.setattr(
        "qqtools.plugins.qexp.layout.load_machine_record",
        lambda _cfg: {"notifications": {"enabled": True, "providers": {
            "future_provider": "malformed",
            "feishu": {"enabled": False},
        }}},
    )
    monkeypatch.setattr(
        "qqtools.plugins.qexp.notifications._safe_diagnostic",
        lambda _cfg, event_type, _event, key, reason, outcome, **_kwargs:
            diagnostics.append((event_type, key, reason, outcome)),
    )

    NotificationHook(registry={}).handle(cfg, _event())

    assert diagnostics == [(
        "notification_skipped",
        notification_key("future_provider", _event()),
        "unknown_provider",
        "skipped",
    )]
