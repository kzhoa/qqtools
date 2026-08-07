"""Static qexp notification providers and lifecycle hook."""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any

from ..config_types import RootConfig
from ..events import write_notification_diagnostic
from ..notification_config import load_shared_feishu_webhook, validate_notifications
from ..runtime.paths import shared_paths
from ..runtime.store import CASConflict, atomic_replace, create_if_absent, read_json
from .feishu import FeishuNotifier, NotificationTransportError
from .base import Notifier


REGISTRY: dict[str, Notifier] = {"feishu": FeishuNotifier()}


def notification_key(notifier: str, event: Any) -> str:
    canonical = json.dumps([notifier, event.task_id, event.attempt_id, event.phase],
                           ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _record_path(cfg: RootConfig, key: str):
    return shared_paths(cfg.shared_root)["notifications"] / f"{key}.json"


def claim_notification(cfg: RootConfig, notifier: str, event: Any, key: str) -> bool:
    record = {"notification_key": key, "notifier": notifier, "task_id": event.task_id,
              "attempt_id": event.attempt_id, "phase": event.phase, "state": "claimed",
              "claimed_at": event.finished_at}
    try:
        create_if_absent(_record_path(cfg, key), record)
    except CASConflict:
        return False
    return True


def _finish_claim(cfg: RootConfig, key: str, state: str, reason_code: str, **extra: Any) -> None:
    path = _record_path(cfg, key)
    try:
        record = read_json(path)
        record.update({"state": state, "reason_code": reason_code,
                       "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra})
        atomic_replace(path, record)
    except Exception:
        pass


@dataclass(slots=True)
class NotificationHook:
    """Read current machine configuration and dispatch each provider independently."""

    name: str = "notifications"
    registry: dict[str, Notifier] | None = None

    def handle(self, cfg: RootConfig, event: Any) -> None:
        registry = self.registry if self.registry is not None else REGISTRY
        try:
            from ..layout import load_machine_record
            raw = (load_machine_record(cfg) or {}).get("notifications")
            config = validate_notifications(raw, allow_unknown=True)
        except Exception:
            _safe_diagnostic(cfg, "notification_skipped", event,
                             notification_key("feishu", event), "invalid_config", "skipped")
            return
        if not config["enabled"]:
            return
        for provider_name, provider_cfg in config["providers"].items():
            key = notification_key(provider_name, event)
            if not isinstance(provider_cfg, dict):
                _safe_diagnostic(cfg, "notification_skipped", event, key,
                                 "unknown_provider", "skipped")
                continue
            if not provider_cfg.get("enabled"):
                continue
            provider = registry.get(provider_name)
            if provider is None:
                _safe_diagnostic(cfg, "notification_skipped", event, key, "unknown_provider", "skipped")
                continue
            credential_source = provider_cfg["credential_source"]
            if credential_source == "env":
                webhook = os.environ.get(provider_cfg["webhook_env"])
                reason_code = "missing_webhook_env"
            else:
                try:
                    webhook = load_shared_feishu_webhook(cfg)
                    reason_code = "shared_webhook_unavailable"
                except (OSError, ValueError):
                    webhook = None
                    reason_code = "shared_webhook_unavailable"
            if not webhook:
                _safe_diagnostic(cfg, "notification_skipped", event, key, reason_code, "skipped")
                continue
            secret = None
            if provider_cfg.get("secret_env"):
                secret = os.environ.get(provider_cfg["secret_env"])
                if not secret:
                    _safe_diagnostic(cfg, "notification_skipped", event, key, "missing_secret_env", "skipped")
                    continue
            if not claim_notification(cfg, provider_name, event, key):
                _safe_diagnostic(cfg, "notification_skipped", event, key, "already_claimed", "skipped")
                continue
            _safe_diagnostic(cfg, "notification_claimed", event, key, "send_claimed", "claimed")
            try:
                result = provider.send(event, webhook=webhook, secret=secret,
                                       timeout_seconds=provider_cfg["timeout_seconds"])
            except NotificationTransportError as exc:
                _finish_claim(cfg, key, "failed", exc.reason_code)
                _safe_diagnostic(cfg, "notification_failed", event, key, exc.reason_code, "failed",
                                 http_status=exc.http_status, business_code=exc.business_code,
                                 error_type=exc.error_type)
                continue
            except Exception:
                _finish_claim(cfg, key, "failed", "network_error")
                _safe_diagnostic(cfg, "notification_failed", event, key, "network_error", "failed",
                                 error_type="provider_error")
                continue
            _finish_claim(cfg, key, "sent", "delivered", **result)
            _safe_diagnostic(cfg, "notification_sent", event, key, "delivered", "sent",
                             http_status=result.get("http_status"), business_code=result.get("business_code"))


def _safe_diagnostic(cfg: RootConfig, event_type: str, event: Any, key: str,
                     reason_code: str, outcome: str, **kwargs: Any) -> None:
    try:
        write_notification_diagnostic(cfg, event_type, event, notification_key=key,
                                      reason_code=reason_code, outcome=outcome, **kwargs)
    except Exception:
        pass


__all__ = ["NotificationHook", "Notifier", "REGISTRY", "claim_notification", "notification_key"]
