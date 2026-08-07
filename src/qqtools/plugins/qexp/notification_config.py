"""Machine-scoped notification configuration without persisted credentials."""
from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from .config_types import RootConfig
from .layout import load_machine_record, save_machine_record
from .runtime.locks import machine_lock
from .runtime.paths import shared_paths
from .runtime.store import atomic_replace, read_json

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DEFAULT_WEBHOOK_ENV = "QEXP_FEISHU_WEBHOOK"
_CREDENTIAL_SOURCES = frozenset({"env", "shared_file"})


def default_notifications() -> dict[str, Any]:
    return {"enabled": False, "providers": {}}


def _validate_env(value: Any, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not _ENV_NAME.fullmatch(value):
        raise ValueError("environment variable name is invalid")
    return value


def _validate_credential_source(value: Any) -> str:
    if value not in _CREDENTIAL_SOURCES:
        raise ValueError("feishu.credential_source must be env or shared_file")
    return value


def shared_feishu_webhook_path(cfg: RootConfig) -> Path:
    return shared_paths(cfg.shared_root)["machines"] / cfg.machine_name / "secrets" / "feishu-webhook.json"


def write_shared_feishu_webhook(cfg: RootConfig, webhook: str) -> None:
    """Persist an explicitly opted-in shared-root Feishu webhook with restrictive permissions."""
    if not isinstance(webhook, str) or not webhook:
        raise ValueError("shared Feishu webhook must be a non-empty string")
    with machine_lock(cfg.shared_root, cfg.machine_name):
        path = shared_feishu_webhook_path(cfg)
        atomic_replace(path, {"webhook": webhook})
        os.chmod(path, 0o600)


def load_shared_feishu_webhook(cfg: RootConfig) -> str:
    """Load an explicitly opted-in shared-root webhook."""
    path = shared_feishu_webhook_path(cfg)
    webhook = read_json(path).get("webhook")
    if not isinstance(webhook, str) or not webhook:
        raise ValueError("shared Feishu webhook record is invalid")
    return webhook


def validate_notifications(value: Any, *, allow_unknown: bool = False) -> dict[str, Any]:
    if value is None:
        return default_notifications()
    if not isinstance(value, dict):
        raise ValueError("notifications must be an object")
    enabled = value.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("notifications.enabled must be boolean")
    providers = value.get("providers", {})
    if not isinstance(providers, dict):
        raise ValueError("notifications.providers must be an object")
    result: dict[str, Any] = {"enabled": enabled, "providers": {}}
    for name, raw in providers.items():
        if name != "feishu" and not allow_unknown:
            raise ValueError(f"unknown notification provider {name!r}")
        if name != "feishu":
            result["providers"][name] = dict(raw) if isinstance(raw, dict) else raw
            continue
        if not isinstance(raw, dict):
            raise ValueError("feishu configuration must be an object")
        provider = dict(raw)
        provider["enabled"] = provider.get("enabled", False)
        if not isinstance(provider["enabled"], bool):
            raise ValueError("feishu.enabled must be boolean")
        provider["credential_source"] = _validate_credential_source(
            provider.get("credential_source", "env")
        )
        provider["webhook_env"] = _validate_env(provider.get("webhook_env", DEFAULT_WEBHOOK_ENV))
        provider["secret_env"] = _validate_env(provider.get("secret_env"), nullable=True)
        timeout = provider.get("timeout_seconds", 5)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0.5 <= timeout <= 30:
            raise ValueError("feishu.timeout_seconds must be between 0.5 and 30")
        provider["timeout_seconds"] = timeout
        result["providers"][name] = provider
    return result


def load_notifications(cfg: RootConfig) -> dict[str, Any]:
    record = load_machine_record(cfg) or {}
    try:
        return validate_notifications(record.get("notifications"))
    except ValueError:
        return {"enabled": False, "providers": {}}


def update_notifications(cfg: RootConfig, updater) -> dict[str, Any]:
    with machine_lock(cfg.shared_root, cfg.machine_name):
        record = load_machine_record(cfg) or {}
        current = validate_notifications(record.get("notifications"))
        updated = validate_notifications(updater(deepcopy(current)))
        record["notifications"] = updated
        save_machine_record(cfg, record)
        return updated
