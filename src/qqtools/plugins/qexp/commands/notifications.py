"""Canonical setup and delivery service for qexp notifications."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .. import notification_credentials, notification_policy, notification_resolver
from ..agent.context import MachineRuntime
from ..config_types import RootConfig
from ..notification_cleanup import cleanup_credentials, credential_retention_status, mark_reference
from ..notification_reconciliation import reconcile_legacy
from ..notifications.feishu import FeishuNotifier, NotificationTransportError
from ..runtime.store import CASConflict

_VALID_SCOPES = frozenset({"global", "project"})
_OVERRIDE_FIELDS = ("enabled", "destination", "timeout_seconds")
_PLACEHOLDER_CREDENTIAL_ID = "0" * 32
_TRANSPORT_REASONS = frozenset({"timeout", "http_error", "network_error", "invalid_response", "business_error"})


def _target(runtime: MachineRuntime, scope: str, cfg: RootConfig | None) -> tuple[Path, str | None, RootConfig | None]:
    try:
        runtime.require_initialized()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc

    if not isinstance(scope, str) or scope not in _VALID_SCOPES:
        raise ValueError("scope must be 'global' or 'project'.")
    if scope == "global":
        return runtime.root, None, None
    if not isinstance(cfg, RootConfig):
        raise ValueError("Project notification scope requires a RootConfig.")
    try:
        context = runtime.verified_execution_context(cfg.shared_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc
    reconcile_legacy(runtime, context.cfg)
    return runtime.root, context.binding.project_id, context.cfg


def _load_policy(runtime_root: Path, scope: str, project_id: str | None) -> dict[str, Any]:
    try:
        return notification_policy.load_policy(runtime_root, scope, project_id)
    except Exception:
        raise ValueError("Notification policy could not be read safely.") from None


def _resolve_policy(runtime_root: Path, project_id: str | None) -> dict[str, Any]:
    try:
        return notification_resolver.resolve_policy(runtime_root, project_id)
    except Exception:
        raise ValueError("Notification policy could not be resolved safely.") from None


def _validate_override(override: dict[str, Any]) -> None:
    try:
        notification_policy.validate_override(override)
    except Exception:
        raise ValueError("Notification policy override is invalid.") from None


def _replace_policy(
    runtime_root: Path, scope: str, project_id: str | None, revision: int, override: dict[str, Any] | None
) -> dict[str, Any]:
    try:
        with notification_policy.policy_guard(runtime_root):
            current = notification_policy.load_policy_unlocked(runtime_root, scope, project_id)
            if current["revision"] != revision:
                raise CASConflict("Notification policy revision changed before publication.")
            destination = override.get("destination") if override is not None else None
            if isinstance(destination, dict) and destination["source"] == "private_file":
                mark_reference(runtime_root, destination["credential_id"])
            return notification_policy.replace_policy_unlocked(runtime_root, scope, revision, override, project_id)
    except CASConflict:
        raise ValueError("Notification policy changed concurrently; retry the operation.") from None
    except Exception:
        raise ValueError("Notification policy could not be updated safely.") from None


def _own_override(record: dict[str, Any]) -> dict[str, Any]:
    override = record.get("override")
    return {} if override is None else deepcopy(override)


def _changed_fields(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    return sorted(
        field
        for field in set(before) | set(after)
        if before.get(field) != after.get(field) or (field in before) != (field in after)
    )


def _signing_choice(secret_env: str | None, unsigned: bool, previous_destination: Any) -> str | dict[str, str]:
    if secret_env is not None:
        return {"env": secret_env}
    if unsigned:
        return "unsigned"
    signing = previous_destination.get("signing") if isinstance(previous_destination, dict) else None
    if isinstance(signing, dict) and signing.get("env"):
        raise ValueError("Replacing a signed destination requires secret_env or unsigned=True.")
    return "unsigned"


def _stage_webhook(runtime_root: Path, webhook: str) -> str:
    try:
        return notification_credentials.stage_webhook(runtime_root, webhook)
    except Exception:
        raise ValueError("Webhook is invalid or private credential staging failed.") from None


def _redacted_destination(destination: Any) -> dict[str, Any] | None:
    if not isinstance(destination, dict):
        return None
    safe_fields = {"provider", "source", "credential_id", "webhook_env", "signing"}
    return {key: deepcopy(value) for key, value in destination.items() if key in safe_fields}


def _render(
    runtime_root: Path,
    scope: str,
    project_id: str | None,
    effective: dict[str, Any],
    *,
    action: str,
) -> dict[str, Any]:
    provenance = deepcopy(effective["provenance"])
    global_revision = effective["global_revision"]
    project_revision = effective["project_revision"]
    revision = global_revision if scope == "global" else project_revision
    result: dict[str, Any] = {
        "action": action,
        "section": "notifications",
        "scope": scope,
        "source": provenance["destination"],
        "provenance": provenance,
        "effective_values": {
            "enabled": effective["enabled"],
            "destination": _redacted_destination(effective["destination"]),
            "timeout_seconds": effective["timeout_seconds"],
        },
        "runtime_root": str(runtime_root),
        "revision": revision,
        "global_revision": global_revision,
        "project_revision": project_revision,
    }
    if project_id is not None:
        result["project_id"] = project_id
    return result


def _mutation_note(scope: str, provenance: dict[str, str]) -> str:
    if scope == "global":
        return "Projects without overrides inherit global settings. Delivery test not run."
    overridden = [field for field in _OVERRIDE_FIELDS if provenance.get(field) == "project"]
    inherited = [field for field in _OVERRIDE_FIELDS if provenance.get(field) == "global"]
    notes = []
    if overridden:
        notes.append(f"Project overrides {', '.join(overridden)}.")
    if inherited:
        prefix = "Other settings inherit globally" if overridden else "Settings inherit globally"
        notes.append(f"{prefix}: {', '.join(inherited)}.")
    if not overridden and not inherited:
        notes.append("Project uses built-in defaults where no global setting exists.")
    notes.append("Delivery test not run.")
    return " ".join(notes)


def _mutation_result(
    runtime_root: Path,
    scope: str,
    project_id: str | None,
    *,
    action: str,
    changed: bool,
    outcome: str,
    changed_fields: list[str],
    committed: bool,
) -> dict[str, Any]:
    try:
        effective = notification_resolver.resolve_policy(runtime_root, project_id)
        result = _render(runtime_root, scope, project_id, effective, action=action)
    except Exception:
        if committed:
            raise ValueError(
                "Notification policy was updated, but its status could not be rendered; run 'qexp notifications show'."
            ) from None
        raise ValueError("Notification policy status could not be rendered; retry show.") from None
    result.update(
        {
            "changed": changed,
            "outcome": outcome,
            "changed_fields": changed_fields,
            "note": _mutation_note(scope, result["provenance"]),
        }
    )
    if committed:
        try:
            result["cleanup"] = cleanup_credentials(runtime_root, limit=8)
        except (OSError, RuntimeError, ValueError):
            result["cleanup"] = {"diagnostic": "Private credential cleanup deferred; policy change remains committed."}
    return result


def _setup_destination_signing(
    secret_env: str | None, unsigned: bool, previous_destination: Any
) -> str | dict[str, str]:
    if secret_env is not None and unsigned:
        raise ValueError("Choose either secret_env or unsigned=True, not both.")
    return _signing_choice(secret_env, unsigned, previous_destination)


def show_notifications(runtime: MachineRuntime, scope: str, cfg: RootConfig | None = None) -> dict[str, Any]:
    """Return redacted effective notification settings for one scope."""
    runtime_root, project_id, _ = _target(runtime, scope, cfg)
    effective = _resolve_policy(runtime_root, project_id)
    result = _render(runtime_root, scope, project_id, effective, action="show")
    if scope == "global":
        result["retention"] = credential_retention_status(runtime_root)
    return result


def setup_notifications(
    runtime: MachineRuntime,
    scope: str,
    *,
    cfg: RootConfig | None = None,
    webhook: str | None = None,
    webhook_env: str | None = None,
    secret_env: str | None = None,
    unsigned: bool = False,
) -> dict[str, Any]:
    """Configure a destination and enable notifications for the selected scope."""
    runtime_root, project_id, _ = _target(runtime, scope, cfg)
    if (webhook is None) == (webhook_env is None):
        raise ValueError("Provide exactly one of webhook or webhook_env.")
    if type(unsigned) is not bool:
        raise ValueError("unsigned must be boolean.")

    record = _load_policy(runtime_root, scope, project_id)
    before = _own_override(record)
    effective_before = _resolve_policy(runtime_root, project_id)
    signing = _setup_destination_signing(secret_env, unsigned, effective_before["destination"])
    if webhook is not None:
        destination = {
            "provider": "feishu",
            "source": "private_file",
            "credential_id": _PLACEHOLDER_CREDENTIAL_ID,
            "signing": signing,
        }
    else:
        destination = {
            "provider": "feishu",
            "source": "env",
            "webhook_env": webhook_env,
            "signing": signing,
        }
    candidate = {**before, "enabled": True, "destination": destination}
    _validate_override(candidate)

    if webhook is not None:
        credential_id = _stage_webhook(runtime_root, webhook)
        candidate["destination"]["credential_id"] = credential_id
        _validate_override(candidate)

    changed_fields = _changed_fields(before, candidate)
    committed = False
    if changed_fields:
        _replace_policy(runtime_root, scope, project_id, record["revision"], candidate)
        committed = True
    return _mutation_result(
        runtime_root,
        scope,
        project_id,
        action="setup",
        changed=bool(changed_fields),
        outcome="updated" if changed_fields else "unchanged",
        changed_fields=changed_fields,
        committed=committed,
    )


def set_notifications(
    runtime: MachineRuntime,
    scope: str,
    *,
    cfg: RootConfig | None = None,
    enabled: bool | None = None,
    webhook: str | None = None,
    webhook_env: str | None = None,
    secret_env: str | None = None,
    unsigned: bool = False,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Update explicit sparse notification fields without replacing other fields."""
    runtime_root, project_id, _ = _target(runtime, scope, cfg)
    if type(unsigned) is not bool:
        raise ValueError("unsigned must be boolean.")
    if secret_env is not None and unsigned:
        raise ValueError("Choose either secret_env or unsigned=True, not both.")
    if webhook is not None and webhook_env is not None:
        raise ValueError("Provide either webhook or webhook_env, not both.")
    if all(value is None for value in (enabled, webhook, webhook_env, secret_env, timeout_seconds)) and not unsigned:
        raise ValueError("set requires at least one value.")

    record = _load_policy(runtime_root, scope, project_id)
    before = _own_override(record)
    candidate = deepcopy(before)
    if enabled is not None:
        candidate["enabled"] = enabled
    if timeout_seconds is not None:
        candidate["timeout_seconds"] = timeout_seconds

    if webhook is not None or webhook_env is not None:
        effective_before = _resolve_policy(runtime_root, project_id)
        signing = _signing_choice(secret_env, unsigned, effective_before["destination"])
        candidate["destination"] = (
            {
                "provider": "feishu",
                "source": "private_file",
                "credential_id": _PLACEHOLDER_CREDENTIAL_ID,
                "signing": signing,
            }
            if webhook is not None
            else {"provider": "feishu", "source": "env", "webhook_env": webhook_env, "signing": signing}
        )
    elif secret_env is not None or unsigned:
        effective_before = _resolve_policy(runtime_root, project_id)
        destination = candidate.get("destination", effective_before["destination"])
        if not isinstance(destination, dict):
            raise ValueError("Signing cannot be set without a configured destination.")
        updated_destination = deepcopy(destination)
        updated_destination["signing"] = {"env": secret_env} if secret_env is not None else "unsigned"
        candidate["destination"] = updated_destination

    _validate_override(candidate)
    if webhook is not None:
        candidate["destination"]["credential_id"] = _stage_webhook(runtime_root, webhook)
        _validate_override(candidate)
    changed_fields = _changed_fields(before, candidate)
    committed = False
    if changed_fields:
        _replace_policy(runtime_root, scope, project_id, record["revision"], candidate)
        committed = True
    return _mutation_result(
        runtime_root,
        scope,
        project_id,
        action="set",
        changed=bool(changed_fields),
        outcome="updated" if changed_fields else "unchanged",
        changed_fields=changed_fields,
        committed=committed,
    )


def reset_notifications(runtime: MachineRuntime, scope: str, *, cfg: RootConfig | None = None) -> dict[str, Any]:
    """Reset one scope to inherited or built-in defaults using a revisioned tombstone."""
    runtime_root, project_id, _ = _target(runtime, scope, cfg)
    record = _load_policy(runtime_root, scope, project_id)
    before = _own_override(record)
    changed_fields = sorted(before)
    _replace_policy(runtime_root, scope, project_id, record["revision"], None)
    return _mutation_result(
        runtime_root,
        scope,
        project_id,
        action="reset",
        changed=True,
        outcome="reset",
        changed_fields=changed_fields,
        committed=True,
    )


def test_notifications(runtime: MachineRuntime, scope: str, *, cfg: RootConfig | None = None) -> dict[str, Any]:
    """Send one labeled test card through the invoking CLI process."""
    runtime_root, project_id, selected_cfg = _target(runtime, scope, cfg)
    try:
        snapshot = notification_resolver.resolve_delivery(runtime_root, project_id)
    except ValueError:
        raise
    except Exception:
        raise ValueError("Notification delivery credentials could not be resolved safely.") from None

    finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    event = SimpleNamespace(
        phase="test",
        task_id="notification-test",
        attempt_id="notification-test",
        project_id=project_id or "global",
        project=str(selected_cfg.project_root) if selected_cfg is not None else "",
        task_name="notification delivery test",
        reason="Manual notification test requested by CLI",
        exit_code=None,
        execution_machine_name="CLI",
        dispatching_machine_name="CLI",
        execution_started_at=finished_at,
        duration_ms=0,
        finished_at=finished_at,
    )
    try:
        provider_outcome = FeishuNotifier().send(
            event,
            webhook=snapshot["webhook"],
            secret=snapshot["secret"],
            timeout_seconds=snapshot["timeout_seconds"],
        )
    except NotificationTransportError as exc:
        reason_code = (
            exc.reason_code
            if isinstance(exc.reason_code, str) and exc.reason_code in _TRANSPORT_REASONS
            else "provider_error"
        )
        status = f" (HTTP {exc.http_status})" if type(exc.http_status) is int else ""
        raise ValueError(f"Notification test failed: {reason_code}{status}.") from None
    except Exception:
        raise ValueError("Notification test failed: provider_error.") from None

    provenance = deepcopy(snapshot["provenance"])
    global_revision = snapshot["global_revision"]
    project_revision = snapshot["project_revision"]
    revision = global_revision if scope == "global" else project_revision
    result: dict[str, Any] = {
        "action": "test",
        "section": "notifications",
        "scope": scope,
        "tested_by": "cli",
        "source": provenance["destination"],
        "provenance": provenance,
        "runtime_root": str(runtime_root),
        "revision": revision,
        "global_revision": global_revision,
        "project_revision": project_revision,
        "outcome": "delivered",
        "provider_outcome": {
            "http_status": provider_outcome.get("http_status"),
            "business_code": provider_outcome.get("business_code"),
        },
    }
    if project_id is not None:
        result["project_id"] = project_id
    return result
