"""Resolve canonical notification policy and delivery credentials."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .notification_credentials import read_webhook
from .notification_policy import load_policy_unlocked, policy_guard

_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "destination": None,
    "timeout_seconds": 5,
}


class NotificationResolutionError(ValueError):
    """An unavailable destination has a stable diagnostic reason."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _override(record: Mapping[str, Any]) -> Mapping[str, Any]:
    override = record["override"]
    return {} if override is None else override


def _resolve_policy_unlocked(runtime_root: Path, project_id: str | None) -> dict[str, Any]:
    global_record = load_policy_unlocked(runtime_root, "global")
    project_record = None if project_id is None else load_policy_unlocked(runtime_root, "project", project_id)
    global_override = _override(global_record)
    project_override = {} if project_record is None else _override(project_record)

    effective: dict[str, Any] = {}
    provenance: dict[str, str] = {}
    for field, default in _DEFAULTS.items():
        if project_record is not None and field in project_override:
            value, source = project_override[field], "project"
        elif field in global_override:
            value, source = global_override[field], "global"
        else:
            value, source = default, "default"
        effective[field] = deepcopy(value) if field == "destination" else value
        provenance[field] = source

    return {
        **effective,
        "provenance": provenance,
        "global_revision": global_record["revision"],
        "project_revision": None if project_record is None else project_record["revision"],
    }


def resolve_policy(runtime_root: Path, project_id: str | None = None) -> dict[str, Any]:
    """Resolve effective policy metadata without reading credential contents.

    Args:
        runtime_root: Machine runtime root containing notification policy records.
        project_id: Stable project registry ID, or None for the global view.

    Returns:
        Effective settings, field provenance, and policy revisions.
    """
    if not (runtime_root / "notifications").exists():
        return _resolve_policy_unlocked(runtime_root, project_id)
    with policy_guard(runtime_root):
        return _resolve_policy_unlocked(runtime_root, project_id)


def _resolve_delivery_unlocked(
    runtime_root: Path,
    policy: Mapping[str, Any],
    environ: Mapping[str, str],
) -> dict[str, Any]:
    if not policy["enabled"]:
        raise NotificationResolutionError("disabled", "Notifications are disabled by the effective policy.")

    destination = policy["destination"]
    if not isinstance(destination, Mapping):
        raise NotificationResolutionError(
            "missing_destination", "Notifications are enabled but no destination is configured."
        )

    source = destination.get("source")
    if source == "private_file":
        credential_id = destination.get("credential_id")
        try:
            webhook = read_webhook(runtime_root, credential_id)
        except (OSError, TypeError, ValueError):
            raise NotificationResolutionError(
                "private_webhook_unavailable", "Private webhook is inaccessible or corrupt; reconfigure it."
            ) from None
        if not isinstance(webhook, str) or not webhook:
            raise NotificationResolutionError(
                "private_webhook_unavailable", "The configured private webhook is empty; configure it again."
            )
    elif source in {"env", "environment"}:
        webhook_env = destination.get("webhook_env")
        if not isinstance(webhook_env, str) or not webhook_env:
            raise NotificationResolutionError(
                "invalid_config", "The configured destination has no webhook environment variable."
            )
        webhook = environ.get(webhook_env)
        if not isinstance(webhook, str) or not webhook:
            raise NotificationResolutionError(
                "missing_webhook_env", f"Webhook environment variable {webhook_env!r} is missing or empty."
            )
    else:
        raise NotificationResolutionError(
            "invalid_config", "The configured notification destination has an unsupported webhook source."
        )

    signing = destination.get("signing")
    if signing == "unsigned":
        secret = None
    elif isinstance(signing, Mapping):
        signing_env = signing.get("env")
        if not isinstance(signing_env, str) or not signing_env:
            raise NotificationResolutionError(
                "invalid_config", "The configured signing environment variable is invalid."
            )
        secret = environ.get(signing_env)
        if not isinstance(secret, str) or not secret:
            raise NotificationResolutionError(
                "missing_secret_env", f"Signing environment variable {signing_env!r} is missing or empty."
            )
    else:
        raise NotificationResolutionError("invalid_config", "The configured notification signing mode is invalid.")

    return {
        "webhook": webhook,
        "secret": secret,
        "timeout_seconds": policy["timeout_seconds"],
        "provenance": policy["provenance"],
        "global_revision": policy["global_revision"],
        "project_revision": policy["project_revision"],
    }


def resolve_delivery(
    runtime_root: Path,
    project_id: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve a credential snapshot for delivery without performing network I/O.

    Args:
        runtime_root: Machine runtime root containing policy and private credentials.
        project_id: Stable project registry ID, or None for the global view.
        environ: Environment snapshot to use, defaulting to the current process environment.

    Returns:
        Webhook, optional signing secret, effective timeout, provenance, and revisions.

    Raises:
        ValueError: If notifications are disabled or their destination credentials are unavailable.
    """
    current_environ = os.environ if environ is None else environ
    with policy_guard(runtime_root):
        policy = _resolve_policy_unlocked(runtime_root, project_id)
        return _resolve_delivery_unlocked(runtime_root, policy, current_environ)


__all__ = ["NotificationResolutionError", "resolve_delivery", "resolve_policy"]
