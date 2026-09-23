"""Read-only legacy notification capture and conversion. QQTOOLS-COMPAT-0016

Call :func:`capture_legacy` while holding the machine lock for the configured
machine. Reconciliation must then take its notification policy lock. This
module never acquires locks or writes files.
"""

from __future__ import annotations

import base64
import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .config_types import RootConfig
from .layout import load_machine_record
from .notification_config import load_notifications_strict, load_shared_feishu_webhook, shared_feishu_webhook_path


@dataclass(frozen=True, slots=True)
class LegacySnapshot:
    fingerprint: str = field(repr=False)
    override: dict[str, Any] | None
    webhook: str | None = field(repr=False)
    is_absent: bool


def _fingerprint(
    raw_section: Any,
    *,
    section_present: bool,
    credential_state: dict[str, str] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "notifications": {"present": section_present, "value": raw_section if section_present else None}
    }
    if credential_state is not None:
        payload["shared_credential"] = credential_state
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def capture_legacy(cfg: RootConfig) -> LegacySnapshot:
    """Capture and validate the legacy machine notification policy without writes.

    The caller must hold ``machine_lock(cfg.shared_root, cfg.machine_name)``.
    """
    record = load_machine_record(cfg) or {}
    section_present = "notifications" in record
    raw_section = record.get("notifications")
    configured = load_notifications_strict(cfg)
    if section_present != (configured is not None):
        raise ValueError("legacy notifications changed during capture")

    if not section_present:
        return LegacySnapshot(
            fingerprint=_fingerprint(None, section_present=False),
            override=None,
            webhook=None,
            is_absent=True,
        )

    if raw_section is None:
        return LegacySnapshot(
            fingerprint=_fingerprint(None, section_present=True),
            override={"enabled": False},
            webhook=None,
            is_absent=False,
        )

    if configured is None:
        raise ValueError("legacy notifications changed during capture")
    provider = configured["providers"].get("feishu")
    if provider is None:
        return LegacySnapshot(
            fingerprint=_fingerprint(raw_section, section_present=True),
            override={"enabled": False},
            webhook=None,
            is_absent=False,
        )

    enabled = configured["enabled"] and provider["enabled"]
    signing: str | dict[str, str]
    if provider["secret_env"] is None:
        signing = "unsigned"
    else:
        signing = {"env": provider["secret_env"]}

    webhook: str | None = None
    credential_state: dict[str, str] | None = None
    source = provider["credential_source"]
    if source == "shared_file":
        path = shared_feishu_webhook_path(cfg)
        try:
            credential_bytes = path.read_bytes()
        except FileNotFoundError:
            credential_state = {"kind": "missing"}
        except OSError:
            credential_state = {"kind": "unreadable"}
        else:
            credential_state = {
                "kind": "present",
                "bytes": base64.b64encode(credential_bytes).decode("ascii"),
            }

        try:
            webhook = load_shared_feishu_webhook(cfg)
        except (OSError, ValueError):
            if enabled:
                raise ValueError("legacy shared Feishu webhook is unavailable") from None
            webhook = None
        if enabled and (webhook is None or credential_state.get("kind") != "present"):
            raise ValueError("legacy shared Feishu webhook is unavailable")

    destination: dict[str, Any]
    if source == "env":
        destination = {
            "provider": "feishu",
            "source": "env",
            "webhook_env": provider["webhook_env"],
            "signing": signing,
        }
    else:
        destination = {"provider": "feishu", "source": "shared_file", "signing": signing}

    override = {
        "enabled": enabled,
        "timeout_seconds": provider["timeout_seconds"],
        "destination": destination,
    }
    return LegacySnapshot(
        fingerprint=_fingerprint(raw_section, section_present=True, credential_state=credential_state),
        override=override,
        webhook=webhook,
        is_absent=False,
    )


def legacy_override(snapshot: LegacySnapshot, credential_id: str | None = None) -> dict[str, Any] | None:
    """Return the project override, resolving a staged private credential ID."""
    if snapshot.is_absent:
        return None
    if snapshot.override is None:
        raise ValueError("legacy notification snapshot has no override")

    override = deepcopy(snapshot.override)
    destination = override.get("destination")
    if isinstance(destination, dict) and destination.get("source") == "shared_file":
        if not isinstance(credential_id, str) or not credential_id:
            if not override["enabled"]:
                override.pop("destination")
                return override
            raise ValueError("shared-file notification conversion requires a credential ID")
        destination["source"] = "private_file"
        destination["credential_id"] = credential_id
    return override
