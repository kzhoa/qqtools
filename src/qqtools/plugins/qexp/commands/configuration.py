"""Typed project and machine-global configuration operations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields
from typing import Any

from ..agent.config import agent_config_payload, set_agent_config
from ..agent.context import MachineRuntime
from ..config_types import RootConfig
from ..launch_policy import reset_launch_handoff_policy, set_launch_handoff_policy, show_launch_handoff_policy
from ..lease import LeasePolicy, lease_policy_path, load_lease_policy, reset_lease_policy, save_lease_policy
from ..notification_config import (
    default_notifications,
    load_notifications_strict,
    reset_notifications,
    update_notifications,
    validate_notifications,
    write_shared_feishu_webhook,
)
from ..progress_policy import reset_progress_policy, set_progress_policy, show_progress_policy
from ..runtime.paths import shared_paths
from ..runtime.store import iter_json, read_json
from ..tmux_policy import reset_tmux_policy, set_tmux_policy, show_tmux_policy

PROJECT_CONFIG_SECTIONS = ("lease", "notifications", "progress", "tmux", "launch-handoff")
CONFIG_SECTIONS = PROJECT_CONFIG_SECTIONS + ("agent",)

_LEASE_FIELDS = frozenset(field.name for field in fields(LeasePolicy))
_LEASE_INTEGER_FIELDS = frozenset({"ttl_seconds"})
_LEASE_NUMERIC_FIELDS = frozenset(
    {
        "renew_interval_seconds",
        "retry_initial_seconds",
        "retry_max_seconds",
        "retry_jitter_ratio",
        "max_clock_skew_seconds",
        "renewal_commit_margin_seconds",
        "clock_observation_max_age_seconds",
        "clock_provider_margin_seconds",
    }
)
_LEASE_STRING_FIELDS = frozenset({"lease_loss_action"})
_NOTIFICATION_FIELDS = frozenset({"enabled"})
_FEISHU_FIELDS = frozenset(
    {
        "enabled",
        "webhook_env",
        "credential_source",
        "shared_webhook",
        "acknowledge_shared_secret_risk",
        "secret_env",
        "timeout_seconds",
    }
)
_AGENT_FIELDS = frozenset({"name", "agent_mode"})


def _require_section(section: str) -> str:
    if section not in CONFIG_SECTIONS:
        raise ValueError(f"unknown config section {section!r}")
    return section


def _require_project_cfg(cfg: RootConfig | None) -> RootConfig:
    if cfg is None:
        raise ValueError("project configuration requires cfg")
    return cfg


def _validate_provider(section: str, provider: str | None) -> None:
    if provider is None:
        return
    if section != "notifications":
        raise ValueError("provider is only supported for notifications")
    if provider != "feishu":
        raise ValueError(f"unknown notification provider {provider!r}")


def _values_mapping(values: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(values, Mapping):
        raise ValueError("config values must be a mapping")
    result = dict(values)
    if not result:
        raise ValueError("config set requires at least one value")
    return result


def _validate_keys(values: Mapping[str, object], allowed: frozenset[str], label: str) -> None:
    unknown = frozenset(values) - allowed
    if unknown:
        raise ValueError(f"unknown {label} option(s): {sorted(unknown)!r}")


def _result(action: str, section: str, scope: str, values: dict[str, Any]) -> dict[str, object]:
    return {"action": action, "section": section, "scope": scope, "values": values}


def _lease_values(policy: LeasePolicy, *, source: str) -> dict[str, Any]:
    return {
        "lease_policy": asdict(policy),
        "source": source,
        "applies_to": "new_claims",
    }


def _show_lease(cfg: RootConfig) -> dict[str, Any]:
    path = lease_policy_path(cfg)
    return _lease_values(load_lease_policy(cfg), source="configured" if path.exists() else "default")


def _redact_notifications(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "enabled": value.get("enabled", False),
        "providers": {},
    }
    providers = value.get("providers", {})
    if not isinstance(providers, dict):
        raise ValueError("notifications.providers must be an object")
    for name, raw in providers.items():
        if isinstance(raw, dict):
            redacted = dict(raw)
            for secret_field in ("shared_webhook", "webhook", "secret"):
                redacted.pop(secret_field, None)
            result["providers"][name] = redacted
        else:
            result["providers"][name] = raw
    return result


def _show_notifications(cfg: RootConfig) -> dict[str, Any]:
    configured = load_notifications_strict(cfg)
    value = default_notifications() if configured is None else configured
    result = _redact_notifications(value)
    result["source"] = "configured" if configured is not None else "default"
    result["applies_to"] = "future_notifications"
    return result


def _show_project(section: str, cfg: RootConfig) -> dict[str, Any]:
    if section == "lease":
        return _show_lease(cfg)
    if section == "notifications":
        return _show_notifications(cfg)
    if section == "progress":
        return show_progress_policy(cfg)
    if section == "tmux":
        return show_tmux_policy(cfg)
    if section == "launch-handoff":
        return show_launch_handoff_policy(cfg)
    raise ValueError(f"unknown project config section {section!r}")


def show_config(
    section: str | None,
    *,
    cfg: RootConfig | None,
    runtime: MachineRuntime,
) -> dict[str, object]:
    """Show one typed configuration section or the complete Project view."""
    if section is None:
        project_cfg = _require_project_cfg(cfg)
        sections: dict[str, object] = {}
        complete = True
        for name in PROJECT_CONFIG_SECTIONS:
            try:
                values = _show_project(name, project_cfg)
            except Exception as exc:
                complete = False
                sections[name] = {
                    "status": "error",
                    "error": {"code": "invalid_config", "message": str(exc)},
                }
            else:
                sections[name] = {"status": "ok", "values": values}
        return {"action": "show", "scope": "project", "complete": complete, "sections": sections}

    name = _require_section(section)
    if name == "agent":
        return _result("show", name, "global", agent_config_payload(runtime))
    return _result("show", name, "project", _show_project(name, _require_project_cfg(cfg)))


def _normalize_clock_provider_priority(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        values = tuple(item.strip() for item in value.split(",") if item.strip())
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        values = tuple(value)
    else:
        raise ValueError("lease clock_provider_priority must be a sequence or comma-separated string")
    if not all(isinstance(item, str) for item in values):
        raise ValueError("lease clock_provider_priority must contain strings")
    return tuple(item.strip() for item in values)


def _validate_lease_types(values: Mapping[str, object]) -> None:
    for name in _LEASE_INTEGER_FIELDS:
        if name in values and type(values[name]) is not int:
            raise ValueError(f"lease {name} must be an integer")
    for name in _LEASE_NUMERIC_FIELDS:
        if name not in values:
            continue
        value = values[name]
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise ValueError(f"lease {name} must be a finite number")
    for name in _LEASE_STRING_FIELDS:
        if name in values and not isinstance(values[name], str):
            raise ValueError(f"lease {name} must be a string")


def _set_lease(cfg: RootConfig, values: Mapping[str, object]) -> dict[str, Any]:
    _validate_keys(values, _LEASE_FIELDS, "lease")
    _validate_lease_types(values)
    normalized = dict(values)
    if "clock_provider_priority" in normalized:
        normalized["clock_provider_priority"] = _normalize_clock_provider_priority(
            normalized["clock_provider_priority"]
        )
    current = load_lease_policy(cfg)
    _assert_no_active_claim(cfg)
    merged = {**asdict(current), **normalized}
    try:
        policy = LeasePolicy(**merged)
    except (TypeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc
    save_lease_policy(cfg, policy)
    return _lease_values(policy, source="configured")


def _assert_no_active_claim(cfg: RootConfig) -> None:
    for path in iter_json(shared_paths(cfg.shared_root)["tasks"]):
        data = read_json(path)
        if bool(data.get("task", {}).get("claim_control", {}).get("active_claim")):
            raise RuntimeError("lease policy cannot change while an active claim exists.")


def _validate_bool(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be a boolean")
    return value


def _set_notifications_global(cfg: RootConfig, values: Mapping[str, object]) -> dict[str, Any]:
    _validate_keys(values, _NOTIFICATION_FIELDS, "notifications")
    enabled = _validate_bool(values["enabled"], "notifications.enabled")
    update_notifications(cfg, lambda current: {**current, "enabled": enabled})
    return _show_notifications(cfg)


def _set_notifications_provider(
    cfg: RootConfig,
    provider: str,
    values: Mapping[str, object],
) -> dict[str, Any]:
    _validate_keys(values, _FEISHU_FIELDS, "feishu")
    if "enabled" in values:
        _validate_bool(values["enabled"], "feishu.enabled")
    if "acknowledge_shared_secret_risk" in values:
        _validate_bool(
            values["acknowledge_shared_secret_risk"],
            "feishu.acknowledge_shared_secret_risk",
        )
    if "shared_webhook" in values:
        webhook = values["shared_webhook"]
        if not isinstance(webhook, str) or not webhook:
            raise ValueError("shared Feishu webhook must be a non-empty string")
        if values.get("credential_source") != "shared_file":
            raise ValueError("feishu.shared_webhook requires credential_source=shared_file")
        if values.get("acknowledge_shared_secret_risk") is not True:
            raise ValueError("feishu.shared_webhook requires acknowledge_shared_secret_risk=true")
    if values.get("credential_source") == "shared_file" and (values.get("acknowledge_shared_secret_risk") is not True):
        raise ValueError("feishu.credential_source shared_file requires acknowledge_shared_secret_risk=true")

    # Validate the existing section and the complete provider value before writing a
    # separately stored secret.
    current = load_notifications_strict(cfg) or default_notifications()
    shared_webhook = values.get("shared_webhook")

    current_providers = dict(current["providers"])
    candidate_provider = dict(
        current_providers.get(
            provider,
            {
                "enabled": False,
                "webhook_env": "QEXP_FEISHU_WEBHOOK",
                "secret_env": None,
                "timeout_seconds": 5,
                "credential_source": "env",
            },
        )
    )
    for name in ("enabled", "webhook_env", "credential_source", "secret_env", "timeout_seconds"):
        if name in values:
            candidate_provider[name] = values[name]
    validate_notifications({"enabled": current["enabled"], "providers": {provider: candidate_provider}})

    def update_provider(current: dict[str, Any]) -> dict[str, Any]:
        providers = dict(current["providers"])
        provider_value = dict(
            providers.get(
                provider,
                {
                    "enabled": False,
                    "webhook_env": "QEXP_FEISHU_WEBHOOK",
                    "secret_env": None,
                    "timeout_seconds": 5,
                    "credential_source": "env",
                },
            )
        )
        for name in ("enabled", "webhook_env", "credential_source", "secret_env", "timeout_seconds"):
            if name in values:
                provider_value[name] = values[name]
        providers[provider] = provider_value
        # Provider changes intentionally preserve the global switch.
        return {**current, "providers": providers}

    if shared_webhook is not None:
        write_shared_feishu_webhook(cfg, shared_webhook)
    update_notifications(cfg, update_provider)
    return _show_notifications(cfg)


def _set_agent(runtime: MachineRuntime, values: Mapping[str, object]) -> dict[str, Any]:
    _validate_keys(values, _AGENT_FIELDS, "agent")
    if "name" in values and not isinstance(values["name"], str):
        raise ValueError("agent name must be a string")
    if "agent_mode" in values and not isinstance(values["agent_mode"], str):
        raise ValueError("agent agent_mode must be a string")
    set_agent_config(
        runtime,
        name=values.get("name"),
        agent_mode=values.get("agent_mode"),
    )
    return agent_config_payload(runtime)


def set_config(
    section: str,
    *,
    cfg: RootConfig | None,
    runtime: MachineRuntime,
    provider: str | None = None,
    values: Mapping[str, object],
) -> dict[str, object]:
    """Validate and set one fixed-schema configuration section."""
    name = _require_section(section)
    _validate_provider(name, provider)
    options = _values_mapping(values)
    if name == "agent":
        if provider is not None:
            raise ValueError("provider is not supported for agent")
        return _result("set", name, "global", _set_agent(runtime, options))

    project_cfg = _require_project_cfg(cfg)
    if name == "lease":
        if provider is not None:
            raise ValueError("provider is not supported for lease")
        result = _set_lease(project_cfg, options)
    elif name == "notifications":
        result = (
            _set_notifications_provider(project_cfg, provider, options)
            if provider is not None
            else _set_notifications_global(project_cfg, options)
        )
    elif name == "progress":
        if provider is not None:
            raise ValueError("provider is not supported for progress")
        _validate_keys(options, frozenset({"interval_seconds"}), "progress")
        result = set_progress_policy(project_cfg, options["interval_seconds"])
    elif name == "tmux":
        if provider is not None:
            raise ValueError("provider is not supported for tmux")
        _validate_keys(options, frozenset({"enabled"}), "tmux")
        result = set_tmux_policy(project_cfg, _validate_bool(options["enabled"], "tmux.enabled"))
    elif name == "launch-handoff":
        if provider is not None:
            raise ValueError("provider is not supported for launch-handoff")
        _validate_keys(options, frozenset({"timeout_seconds"}), "launch-handoff")
        result = set_launch_handoff_policy(project_cfg, options["timeout_seconds"])
    else:
        raise ValueError(f"unknown project config section {name!r}")
    return _result("set", name, "project", result)


def reset_config(
    section: str,
    *,
    cfg: RootConfig | None,
    runtime: MachineRuntime,
    provider: str | None = None,
) -> dict[str, object]:
    """Reset one explicit configuration override."""
    name = _require_section(section)
    _validate_provider(name, provider)
    if name == "agent":
        raise ValueError("agent configuration cannot be reset")

    project_cfg = _require_project_cfg(cfg)
    if name == "lease":
        if provider is not None:
            raise ValueError("provider is not supported for lease")
        _assert_no_active_claim(project_cfg)
        result = _lease_values(reset_lease_policy(project_cfg), source="default")
    elif name == "notifications":
        result = _show_notifications_after_notification_reset(project_cfg, provider)
    elif name == "progress":
        if provider is not None:
            raise ValueError("provider is not supported for progress")
        result = reset_progress_policy(project_cfg)
    elif name == "tmux":
        if provider is not None:
            raise ValueError("provider is not supported for tmux")
        result = reset_tmux_policy(project_cfg)
    elif name == "launch-handoff":
        if provider is not None:
            raise ValueError("provider is not supported for launch-handoff")
        result = reset_launch_handoff_policy(project_cfg)
    else:
        raise ValueError(f"unknown project config section {name!r}")
    return _result("reset", name, "project", result)


def _show_notifications_after_notification_reset(
    cfg: RootConfig,
    provider: str | None,
) -> dict[str, Any]:
    reset_notifications(cfg, provider=provider)
    return _show_notifications(cfg)


__all__ = [
    "CONFIG_SECTIONS",
    "PROJECT_CONFIG_SECTIONS",
    "reset_config",
    "set_config",
    "show_config",
]
