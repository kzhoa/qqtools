import json
from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp.notification_config import validate_notifications
from qqtools.plugins.qexp.notifications import NotificationHook, notification_key
from qqtools.plugins.qexp.notifications.feishu import FeishuNotifier, NotificationTransportError


def _event(**overrides):
    values = {
        "phase": "failed",
        "task_id": "task-a",
        "attempt_id": "attempt-a",
        "reason": "nonzero_exit",
        "exit_code": None,
        "execution_machine_name": "gpu-a",
        "dispatching_machine_name": "gpu-b",
        "finished_at": "2026-08-07T00:00:00Z",
        "execution_started_at": None,
        "duration_ms": None,
    }
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
    payload = json.loads(seen["data"])
    assert payload["msg_type"] == "interactive"
    assert payload["card"]["header"] == {
        "template": "red",
        "title": {"tag": "plain_text", "content": "qexp task failed"},
    }
    assert payload["card"]["elements"] == [
        {
            "tag": "markdown",
            "content": "\n".join(
                (
                    "- **Project ID**: `(not recorded)`",
                    "- **Project**: `(not recorded)`",
                    "- **Task Name**: `(not recorded)`",
                    "- **Task ID**: `task-a`",
                    "- **Attempt ID**: `attempt-a`",
                    "- **Reason**: `nonzero_exit`",
                    "- **Exit Code**: `(not recorded)`",
                    "- **Execution Machine**: `gpu-a`",
                    "- **Notification Machine**: `gpu-b`",
                    "- **Start Time**: `(not recorded)`",
                    "- **Execution Duration**: `(not recorded)`",
                    "- **Notification Machine Time**: `2026-08-07T00:00:00Z`",
                )
            ),
        }
    ]
    assert b"secret" not in seen["data"]


@pytest.mark.parametrize(
    ("phase", "template", "title"),
    [
        ("succeeded", "green", "qexp task succeeded"),
        ("cancelled", "orange", "qexp task cancelled"),
    ],
)
def test_feishu_card_uses_phase_status_colours(phase, template, title):
    seen = {}

    class Response:
        status = 200

        def read(self):
            return b'{"code": 0}'

    def urlopen(request, timeout):
        seen["payload"] = json.loads(request.data)
        return Response()

    FeishuNotifier(urlopen=urlopen).send(
        _event(phase=phase), webhook="https://example.invalid/hook", secret=None, timeout_seconds=5
    )

    assert seen["payload"]["card"]["header"] == {
        "template": template,
        "title": {"tag": "plain_text", "content": title},
    }


@pytest.mark.parametrize(
    ("machine_name", "expected"),
    [
        (None, "(not recorded)"),
        (" \t ", "(not recorded)"),
        ("gpu`a\\b\nnext\rrow\tcell", "gpu\\`a\\\\b\\nnext\\rrow\\tcell"),
    ],
)
def test_feishu_card_safely_displays_machine_name(machine_name, expected):
    seen = {}

    class Response:
        status = 200

        def read(self):
            return b'{"code": 0}'

    def urlopen(request, timeout):
        seen["payload"] = json.loads(request.data)
        return Response()

    FeishuNotifier(urlopen=urlopen).send(
        _event(execution_machine_name=machine_name),
        webhook="https://example.invalid/hook",
        secret=None,
        timeout_seconds=5,
    )

    markdown = seen["payload"]["card"]["elements"][0]["content"]
    assert f"- **Execution Machine**: `{expected}`" in markdown


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
        lambda _cfg: {
            "notifications": {
                "enabled": True,
                "providers": {
                    "future_provider": "malformed",
                    "feishu": {"enabled": False},
                },
            }
        },
    )
    monkeypatch.setattr(
        "qqtools.plugins.qexp.notifications._safe_diagnostic",
        lambda _cfg, event_type, _event, key, reason, outcome, **_kwargs: diagnostics.append(
            (event_type, key, reason, outcome)
        ),
    )

    NotificationHook(registry={}).handle(cfg, _event())

    assert diagnostics == [
        (
            "notification_skipped",
            notification_key("future_provider", _event()),
            "unknown_provider",
            "skipped",
        )
    ]
