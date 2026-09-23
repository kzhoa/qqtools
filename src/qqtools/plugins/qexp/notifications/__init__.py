"""Static qexp notification providers and lifecycle hook."""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..agent.context import MachineRuntime
from ..config_types import RootConfig
from ..events import write_notification_diagnostic
from ..notification_reconciliation import LegacyConflictError, reconcile_legacy
from ..notification_resolver import NotificationResolutionError, resolve_delivery, resolve_policy
from ..runtime.paths import shared_paths
from ..runtime.store import CASConflict, atomic_replace, create_if_absent, read_json
from .base import Notifier
from .feishu import FeishuNotifier, NotificationTransportError

REGISTRY: dict[str, Notifier] = {"feishu": FeishuNotifier()}
_selected_runtime_root: ContextVar[Path | None] = ContextVar("qexp_notification_runtime_root", default=None)


@contextmanager
def notification_runtime(runtime_root: Path):
    """Use the CLI's selected MachineRuntime for inline terminal commits."""
    token = _selected_runtime_root.set(runtime_root)
    try:
        yield
    finally:
        _selected_runtime_root.reset(token)


def notification_key(notifier: str, event: Any) -> str:
    canonical = json.dumps(
        [notifier, event.task_id, event.attempt_id, event.phase], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _record_path(cfg: RootConfig, key: str):
    return shared_paths(cfg.shared_root)["notifications"] / f"{key}.json"


def claim_notification(cfg: RootConfig, notifier: str, event: Any, key: str) -> bool:
    record = {
        "notification_key": key,
        "notifier": notifier,
        "task_id": event.task_id,
        "attempt_id": event.attempt_id,
        "phase": event.phase,
        "state": "claimed",
        "claimed_at": event.finished_at,
    }
    try:
        create_if_absent(_record_path(cfg, key), record)
    except CASConflict:
        return False
    return True


def _finish_claim(cfg: RootConfig, key: str, state: str, reason_code: str, **extra: Any) -> None:
    path = _record_path(cfg, key)
    try:
        record = read_json(path)
        record.update(
            {
                "state": state,
                "reason_code": reason_code,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                **extra,
            }
        )
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
        key = notification_key("feishu", event)
        try:
            runtime = MachineRuntime(_selected_runtime_root.get())
            project_id = reconcile_legacy(runtime, cfg)
            effective = resolve_policy(runtime.root, project_id)
        except LegacyConflictError:
            _safe_diagnostic(cfg, "notification_skipped", event, key, "legacy_conflict", "skipped")
            return
        except Exception:
            _safe_diagnostic(cfg, "notification_skipped", event, key, "invalid_config", "skipped")
            return
        if not effective["enabled"]:
            return
        try:
            snapshot = resolve_delivery(runtime.root, project_id)
        except NotificationResolutionError as exc:
            _safe_diagnostic(cfg, "notification_skipped", event, key, exc.code, "skipped")
            return
        except (OSError, ValueError):
            _safe_diagnostic(cfg, "notification_skipped", event, key, "invalid_config", "skipped")
            return
        provider = registry.get("feishu")
        if provider is None:
            _safe_diagnostic(cfg, "notification_skipped", event, key, "unknown_provider", "skipped")
            return
        if not claim_notification(cfg, "feishu", event, key):
            _safe_diagnostic(cfg, "notification_skipped", event, key, "already_claimed", "skipped")
            return
        _safe_diagnostic(cfg, "notification_claimed", event, key, "send_claimed", "claimed")
        try:
            result = provider.send(
                event,
                webhook=snapshot["webhook"],
                secret=snapshot["secret"],
                timeout_seconds=snapshot["timeout_seconds"],
            )
        except NotificationTransportError as exc:
            _finish_claim(cfg, key, "failed", exc.reason_code)
            _safe_diagnostic(
                cfg,
                "notification_failed",
                event,
                key,
                exc.reason_code,
                "failed",
                http_status=exc.http_status,
                business_code=exc.business_code,
                error_type=exc.error_type,
            )
            return
        except Exception:
            _finish_claim(cfg, key, "failed", "network_error")
            _safe_diagnostic(
                cfg, "notification_failed", event, key, "network_error", "failed", error_type="provider_error"
            )
            return
        _finish_claim(cfg, key, "sent", "delivered", **result)
        _safe_diagnostic(
            cfg,
            "notification_sent",
            event,
            key,
            "delivered",
            "sent",
            http_status=result.get("http_status"),
            business_code=result.get("business_code"),
        )


def _safe_diagnostic(
    cfg: RootConfig, event_type: str, event: Any, key: str, reason_code: str, outcome: str, **kwargs: Any
) -> None:
    try:
        write_notification_diagnostic(
            cfg, event_type, event, notification_key=key, reason_code=reason_code, outcome=outcome, **kwargs
        )
    except Exception:
        pass


__all__ = ["NotificationHook", "Notifier", "REGISTRY", "claim_notification", "notification_key"]
