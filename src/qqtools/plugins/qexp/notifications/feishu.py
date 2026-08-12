"""Feishu Incoming Webhook notifier implemented with Python standard library."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable


class NotificationTransportError(RuntimeError):
    def __init__(self, reason_code: str, *, http_status: int | None = None,
                 business_code: str | None = None, error_type: str | None = None):
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.http_status = http_status
        self.business_code = business_code
        self.error_type = error_type


class FeishuNotifier:
    name = "feishu"

    def __init__(self, *, clock: Callable[[], float] = time.time,
                 urlopen: Callable[..., Any] | None = None):
        self._clock = clock
        self._urlopen = urlopen or urllib.request.urlopen

    def send(self, event: Any, *, webhook: str, secret: str | None,
             timeout_seconds: float) -> dict[str, Any]:
        timestamp = str(int(self._clock()))
        payload: dict[str, Any] = {
            "msg_type": "interactive",
            "card": {
                "config": {"wide_screen_mode": True},
                "header": {
                    "template": _phase_template(event.phase),
                    "title": {
                        "tag": "plain_text",
                        "content": f"qexp 任务{_phase_label(event.phase)}",
                    },
                },
                "elements": [{"tag": "markdown", "content": _card_markdown(event)}],
            },
        }
        if secret is not None:
            string_to_sign = timestamp + "\n" + secret
            digest = hmac.new(string_to_sign.encode("utf-8"), b"", hashlib.sha256).digest()
            payload.update({"timestamp": timestamp, "sign": base64.b64encode(digest).decode("ascii")})
        request = urllib.request.Request(webhook, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                         headers={"Content-Type": "application/json"}, method="POST")
        try:
            response = self._urlopen(request, timeout=timeout_seconds)
            status = getattr(response, "status", getattr(response, "code", None))
            body = response.read()
        except TimeoutError as exc:
            raise NotificationTransportError("timeout", error_type="timeout") from exc
        except urllib.error.HTTPError as exc:
            raise NotificationTransportError("http_error", http_status=exc.code,
                                             error_type="http_error") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise NotificationTransportError("network_error", error_type="network_error") from exc
        if not isinstance(status, int) or not 200 <= status < 300:
            raise NotificationTransportError("http_error", http_status=status if isinstance(status, int) else None,
                                             error_type="http_error")
        try:
            result = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise NotificationTransportError("invalid_response", http_status=status,
                                             error_type="invalid_response") from exc
        if not isinstance(result, dict):
            raise NotificationTransportError("invalid_response", http_status=status,
                                             error_type="invalid_response")
        present = [result[key] for key in ("code", "StatusCode") if key in result]
        if not present or any(isinstance(value, bool) or not isinstance(value, int) for value in present):
            raise NotificationTransportError("invalid_response", http_status=status,
                                             error_type="invalid_response")
        if len(present) == 2 and present[0] != present[1]:
            raise NotificationTransportError("invalid_response", http_status=status,
                                             business_code=str(present[0]),
                                             error_type="invalid_response")
        if any(value != 0 for value in present):
            raise NotificationTransportError("business_error", http_status=status,
                                             business_code=str(present[0]),
                                             error_type="business_error")
        return {"http_status": status, "business_code": "0"}


def _phase_label(phase: Any) -> str:
    return {"succeeded": "成功", "failed": "失败", "cancelled": "已取消"}.get(
        phase, str(phase)
    )


def _phase_template(phase: Any) -> str:
    return {"succeeded": "green", "failed": "red", "cancelled": "orange"}.get(phase, "grey")


def _card_markdown(event: Any) -> str:
    exit_code = "null" if event.exit_code is None else event.exit_code
    return "\n".join((
        f"- **Task ID**: `{_markdown_code(event.task_id)}`",
        f"- **Attempt ID**: `{_markdown_code(event.attempt_id)}`",
        f"- **原因**: `{_markdown_code(event.reason)}`",
        f"- **退出码**: `{_markdown_code(exit_code)}`",
        f"- **执行机器**: `{_markdown_code(event.execution_machine_name)}`",
        f"- **通知机器**: `{_markdown_code(event.dispatching_machine_name)}`",
        f"- **机器完成时间**: `{_markdown_code(event.finished_at)}`",
    ))


def _markdown_code(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("`", "\\`")
