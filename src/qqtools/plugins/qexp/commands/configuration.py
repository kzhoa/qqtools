"""Typed project and machine-global configuration operations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields
from typing import Any

from ..agent.config import agent_config_payload, set_agent_config, validate_agent_mode, validate_agent_name
from ..agent.context import MachineRuntime
from ..agent.diagnostics import parse_log_size
from ..config_types import RootConfig
from ..launch_policy import (
    reset_launch_handoff_policy,
    set_launch_handoff_policy,
    show_launch_handoff_policy,
    validate_launch_handoff_timeout_seconds,
)
from ..lease import LeasePolicy, lease_policy_path, load_lease_policy, reset_lease_policy, save_lease_policy
from ..notification_config import (
    default_notifications,
    load_notifications_strict,
    reset_notifications,
    update_notifications,
    validate_notifications,
    write_shared_feishu_webhook,
)
from ..progress_policy import (
    reset_progress_policy,
    set_progress_policy,
    show_progress_policy,
    validate_interval_seconds,
)
from ..runtime.paths import shared_paths
from ..runtime.store import iter_json, read_json
from ..tmux_policy import reset_tmux_policy, set_tmux_policy, show_tmux_policy, validate_tmux_policy_enabled

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
_AGENT_FIELDS = frozenset({"name", "agent_mode", "log_max_bytes"})


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
    """Build the stable envelope used by all typed configuration commands.

    ``values`` remains the historical field.  ``effective_values`` is an
    additive alias with a name that makes it clear that the payload contains
    the value in force after a mutation (or the value being inspected for a
    read).  Source and application timing are facts supplied by the policy
    implementation, not inferred by the renderer.
    """
    return {
        "action": action,
        "section": section,
        "scope": scope,
        "source": values.get("source", "unavailable"),
        "applies_to": values.get("applies_to"),
        "values": values,
        "effective_values": values,
    }


def _field_value(values: Mapping[str, Any], section: str, field: str, provider: str | None) -> Any:
    """Return one user-facing field from a policy result.

    Policy modules intentionally use small, section-specific result shapes.
    Keeping this translation here avoids leaking those implementation shapes
    into the CLI contract while still allowing truthful no-op detection.
    """
    if section == "lease":
        nested = values.get("lease_policy")
        return nested.get(field) if isinstance(nested, Mapping) else None
    if section == "notifications" and provider is not None:
        providers = values.get("providers")
        nested = providers.get(provider) if isinstance(providers, Mapping) else None
        return nested.get(field) if isinstance(nested, Mapping) else None
    if section == "notifications" and field == "enabled":
        return values.get("enabled")
    return values.get(field)


def _changed_fields(
    section: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    requested: Mapping[str, object],
    provider: str | None,
) -> list[str]:
    changed: list[str] = []
    for field, requested_value in requested.items():
        # The actual shared webhook is intentionally never loaded for output.
        # Supplying one always represents a write, even when the redacted view
        # cannot compare its bytes with the previous value.
        if field == "shared_webhook":
            changed.append(field)
            continue
        if _field_value(before, section, field, provider) != _field_value(after, section, field, provider):
            changed.append(field)
        elif requested_value is not None and _field_value(after, section, field, provider) is None:
            # A value that was accepted but is not represented in a section's
            # compact view still counts as a changed field.  This is mainly a
            # guard for future typed policy fields.
            changed.append(field)
    return changed


def _requested_changed_fields(
    section: str,
    current: Mapping[str, Any],
    requested: Mapping[str, object],
    provider: str | None,
) -> list[str]:
    """Compare requested values with the current effective policy values."""
    changed: list[str] = []
    for field, requested_value in requested.items():
        if field == "shared_webhook":
            changed.append(field)
            continue
        normalized = requested_value
        if section == "lease" and field == "clock_provider_priority":
            normalized = _normalize_clock_provider_priority(requested_value)
        elif section == "agent" and field == "log_max_bytes":
            normalized = parse_log_size(requested_value)
        if normalized != _field_value(current, section, field, provider):
            changed.append(field)
    return changed


def _mutation_result(
    action: str,
    section: str,
    scope: str,
    before: Mapping[str, Any],
    after: dict[str, Any],
    requested: Mapping[str, object],
    provider: str | None,
) -> dict[str, object]:
    fields_changed = _changed_fields(section, before, after, requested, provider)
    changed = bool(fields_changed)
    return {
        **_result(action, section, scope, after),
        "changed": changed,
        "outcome": "updated" if changed else "no_change",
        "changed_fields": fields_changed,
    }


def _reset_changed_fields(
    section: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    provider: str | None,
) -> list[str]:
    if section == "notifications" and provider is None and before.get("providers") != after.get("providers"):
        return ["providers"]
    candidates = {
        "lease": _LEASE_FIELDS,
        "notifications": _NOTIFICATION_FIELDS | _FEISHU_FIELDS,
        "progress": {"interval_seconds"},
        "tmux": {"enabled"},
        "launch-handoff": {"timeout_seconds"},
    }.get(section, set())
    return [
        field
        for field in sorted(candidates)
        if _field_value(before, section, field, provider) != _field_value(after, section, field, provider)
    ]


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
            redacted_fields: list[str] = []
            for secret_field in tuple(redacted):
                normalized = str(secret_field).lower()
                if normalized in {"webhook", "shared_webhook", "secret", "signing_secret", "token", "password"} or (
                    "secret" in normalized and not normalized.endswith("_env")
                ):
                    redacted.pop(secret_field, None)
                    redacted_fields.append(str(secret_field))
            if redacted_fields:
                # Tell operators why a credential is absent without exposing
                # its value.  Environment variable names remain visible as
                # non-secret source metadata (for example ``secret_env``).
                redacted["redacted_fields"] = sorted(redacted_fields)
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
                if name == "notifications":
                    from .notifications import show_notifications

                    values = show_notifications(runtime, "project", project_cfg)
                else:
                    values = _show_project(name, project_cfg)
            except Exception as exc:
                complete = False
                sections[name] = {
                    "status": "error",
                    "scope": "project",
                    "source": "unavailable",
                    "applies_to": None,
                    "error": {"code": "invalid_config", "message": str(exc)},
                }
            else:
                sections[name] = {
                    "status": "ok",
                    "scope": "project",
                    "source": values.get("source", "unavailable"),
                    "applies_to": values.get("applies_to"),
                    "values": values,
                    "effective_values": values,
                }
        return {"action": "show", "scope": "project", "complete": complete, "sections": sections}

    name = _require_section(section)
    if name == "agent":
        values = agent_config_payload(runtime)
        values["source"] = "default" if values.get("provenance") == "default" else "configured"
        values["applies_to"] = "new_agent_processes"
        return _result("show", name, "global", values)
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
    log_max_bytes = parse_log_size(values["log_max_bytes"]) if "log_max_bytes" in values else None
    set_agent_config(
        runtime,
        name=values.get("name"),
        agent_mode=values.get("agent_mode"),
        log_max_bytes=log_max_bytes,
    )
    return agent_config_payload(runtime)


def _validate_set_options(
    section: str,
    options: Mapping[str, object],
    *,
    provider: str | None,
    cfg: RootConfig | None,
) -> None:
    """Validate a mutation before equality can classify it as a no-op."""
    if section == "agent":
        _validate_keys(options, _AGENT_FIELDS, "agent")
        if "name" in options:
            validate_agent_name(options["name"])  # type: ignore[arg-type]
        if "agent_mode" in options:
            validate_agent_mode(options["agent_mode"])  # type: ignore[arg-type]
        if "log_max_bytes" in options:
            parse_log_size(options["log_max_bytes"])
        return
    if section == "lease":
        _validate_keys(options, _LEASE_FIELDS, "lease")
        _validate_lease_types(options)
        normalized = dict(options)
        if "clock_provider_priority" in normalized:
            normalized["clock_provider_priority"] = _normalize_clock_provider_priority(
                normalized["clock_provider_priority"]
            )
        current = load_lease_policy(_require_project_cfg(cfg))
        LeasePolicy(**{**asdict(current), **normalized})
        return
    if section == "notifications":
        if provider is None:
            _validate_keys(options, _NOTIFICATION_FIELDS, "notifications")
            _validate_bool(options.get("enabled"), "notifications.enabled")
            return
        _validate_keys(options, _FEISHU_FIELDS, "feishu")
        if "enabled" in options:
            _validate_bool(options["enabled"], "feishu.enabled")
        if "acknowledge_shared_secret_risk" in options:
            _validate_bool(options["acknowledge_shared_secret_risk"], "feishu.acknowledge_shared_secret_risk")
        if "shared_webhook" in options:
            webhook = options["shared_webhook"]
            if not isinstance(webhook, str) or not webhook:
                raise ValueError("shared Feishu webhook must be a non-empty string")
            if options.get("credential_source") != "shared_file":
                raise ValueError("feishu.shared_webhook requires credential_source=shared_file")
            if options.get("acknowledge_shared_secret_risk") is not True:
                raise ValueError("feishu.shared_webhook requires acknowledge_shared_secret_risk=true")
        if options.get("credential_source") == "shared_file" and (
            options.get("acknowledge_shared_secret_risk") is not True
        ):
            raise ValueError("feishu.credential_source shared_file requires acknowledge_shared_secret_risk=true")
        current = load_notifications_strict(_require_project_cfg(cfg)) or default_notifications()
        providers = current.get("providers", {})
        current_provider = providers.get(provider, {}) if isinstance(providers, Mapping) else {}
        candidate_provider = {
            "enabled": False,
            "webhook_env": "QEXP_FEISHU_WEBHOOK",
            "secret_env": None,
            "timeout_seconds": 5,
            "credential_source": "env",
            **(dict(current_provider) if isinstance(current_provider, Mapping) else {}),
        }
        for field in ("enabled", "webhook_env", "credential_source", "secret_env", "timeout_seconds"):
            if field in options:
                candidate_provider[field] = options[field]
        validate_notifications({"enabled": current["enabled"], "providers": {provider: candidate_provider}})
        return
    if section == "progress":
        _validate_keys(options, frozenset({"interval_seconds"}), "progress")
        validate_interval_seconds(options.get("interval_seconds"))
        return
    if section == "tmux":
        _validate_keys(options, frozenset({"enabled"}), "tmux")
        validate_tmux_policy_enabled(options.get("enabled"))
        return
    if section == "launch-handoff":
        _validate_keys(options, frozenset({"timeout_seconds"}), "launch-handoff")
        validate_launch_handoff_timeout_seconds(options.get("timeout_seconds"))


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
    _validate_set_options(name, options, provider=provider, cfg=cfg)
    if name == "agent":
        if provider is not None:
            raise ValueError("provider is not supported for agent")
        before = agent_config_payload(runtime)
        before["source"] = "default" if before.get("provenance") == "default" else "configured"
        before["applies_to"] = "new_agent_processes"
        # Do not advance the global configuration revision for an explicit
        # no-op.  The command still returns the complete effective payload.
        changed_fields = _requested_changed_fields(name, before, options, None)
        after = before if not changed_fields else _set_agent(runtime, options)
        if changed_fields:
            after["source"] = "default" if after.get("provenance") == "default" else "configured"
            after["applies_to"] = "new_agent_processes"
        return _mutation_result("set", name, "global", before, after, options, None)

    project_cfg = _require_project_cfg(cfg)
    before = _show_project(name, project_cfg)
    # Avoid rewriting a policy when every requested field is already effective.
    # A supplied shared webhook is deliberately treated as a write because its
    # value is never loaded into the redacted comparison view.
    no_op = not _requested_changed_fields(name, before, options, provider)
    if no_op:
        return _mutation_result("set", name, "project", before, dict(before), options, provider)
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
    return _mutation_result("set", name, "project", before, result, options, provider)


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
    before = _show_project(name, project_cfg)
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
    changed_fields = _reset_changed_fields(name, before, result, provider)
    return {
        **_result("reset", name, "project", result),
        "changed": bool(changed_fields),
        "outcome": "updated" if changed_fields else "no_change",
        "changed_fields": changed_fields,
    }


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
