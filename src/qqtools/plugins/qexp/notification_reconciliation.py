"""QQTOOLS-COMPAT-0016: Reconcile older project writers before canonical use."""

from __future__ import annotations

from .agent.context import MachineRuntime
from .config_types import RootConfig
from .notification_cleanup import mark_reference
from .notification_compat import LegacySnapshot, capture_legacy, legacy_override
from .notification_credentials import read_webhook, stage_webhook
from .notification_policy import load_policy_unlocked, policy_guard, replace_policy_unlocked
from .runtime.locks import machine_lock


class LegacyConflictError(ValueError):
    """An older writer changed the source after a canonical policy edit."""


def _new_metadata(snapshot: LegacySnapshot, revision: int) -> dict[str, object]:
    return {"fingerprint": snapshot.fingerprint, "imported_revision": revision, "status": "ready"}


def _converted_override(runtime: MachineRuntime, snapshot: LegacySnapshot) -> dict | None:
    credential_id = stage_webhook(runtime.root, snapshot.webhook) if snapshot.webhook else None
    if credential_id is not None:
        read_webhook(runtime.root, credential_id)
    return legacy_override(snapshot, credential_id)


def _project_id(runtime: MachineRuntime, cfg: RootConfig) -> str:
    runtime.require_initialized()
    context = runtime.verified_execution_context(cfg.shared_root)
    if context.project_id is None:
        raise ValueError("Project notification scope requires a verified local binding.")
    return context.project_id


def reconcile_legacy(runtime: MachineRuntime, cfg: RootConfig) -> str:
    """Import a stable old snapshot, or fail closed on an actual mixed-writer conflict."""
    project_id = _project_id(runtime, cfg)
    with machine_lock(cfg.shared_root, cfg.machine_name, blocking=False) as acquired:
        if not acquired:
            raise RuntimeError("Legacy notification source is busy; retry when the current update completes.")
        try:
            snapshot = capture_legacy(cfg)
        except (OSError, TypeError, ValueError) as exc:
            with policy_guard(runtime.root, blocking=False):
                current = load_policy_unlocked(runtime.root, "project", project_id)
                baseline = current.get("legacy")
                if baseline is None or baseline["status"] != "source_invalid":
                    previous_revision = 0 if baseline is None else baseline["imported_revision"]
                    imported_revision = (
                        current["revision"] + 1 if current["revision"] == previous_revision else previous_revision
                    )
                    replace_policy_unlocked(
                        runtime.root,
                        "project",
                        current["revision"],
                        current["override"],
                        project_id,
                        legacy={
                            "fingerprint": None,
                            "imported_revision": imported_revision,
                            "status": "source_invalid",
                        },
                    )
            raise ValueError(
                "Legacy notification source is unavailable or invalid; inspect project configuration."
            ) from exc
        with policy_guard(runtime.root, blocking=False):
            current = load_policy_unlocked(runtime.root, "project", project_id)
            baseline = current.get("legacy")
            if baseline is not None and baseline["fingerprint"] == snapshot.fingerprint:
                if baseline["status"] != "ready":
                    raise LegacyConflictError(
                        "Notification legacy conflict; run qexp notifications resolve --scope project "
                        "--prefer canonical|legacy."
                    )
                return project_id
            if baseline is None and current["revision"] > 0 and not snapshot.is_absent:
                replace_policy_unlocked(
                    runtime.root,
                    "project",
                    current["revision"],
                    current["override"],
                    project_id,
                    legacy={
                        "fingerprint": snapshot.fingerprint,
                        "imported_revision": 0,
                        "status": "legacy_conflict",
                    },
                )
                raise LegacyConflictError(
                    "Notification legacy conflict; run qexp notifications resolve --scope project "
                    "--prefer canonical|legacy."
                )
            if baseline is not None and (
                baseline["status"] == "legacy_conflict" or current["revision"] != baseline["imported_revision"]
            ):
                if baseline["status"] != "legacy_conflict":
                    metadata = {
                        "fingerprint": snapshot.fingerprint,
                        "imported_revision": baseline["imported_revision"],
                        "status": "legacy_conflict",
                    }
                    replace_policy_unlocked(
                        runtime.root,
                        "project",
                        current["revision"],
                        current["override"],
                        project_id,
                        legacy=metadata,
                    )
                raise LegacyConflictError(
                    "Notification legacy conflict; run qexp notifications resolve --scope project "
                    "--prefer canonical|legacy."
                )
            override = current["override"] if snapshot.is_absent else _converted_override(runtime, snapshot)
            destination = override.get("destination") if override is not None else None
            if isinstance(destination, dict) and destination.get("source") == "private_file":
                mark_reference(runtime.root, destination["credential_id"])
            replace_policy_unlocked(
                runtime.root,
                "project",
                current["revision"],
                override,
                project_id,
                legacy=_new_metadata(snapshot, current["revision"] + 1),
            )
    return project_id


def resolve_legacy_conflict(runtime: MachineRuntime, cfg: RootConfig, *, prefer: str) -> str:
    """Acknowledge the current old snapshot or explicitly import it under both locks."""
    if prefer not in {"canonical", "legacy"}:
        raise ValueError("--prefer must be canonical or legacy.")
    project_id = _project_id(runtime, cfg)
    with machine_lock(cfg.shared_root, cfg.machine_name, blocking=False) as acquired:
        if not acquired:
            raise RuntimeError("Legacy notification source is busy; retry when the current update completes.")
        try:
            snapshot = capture_legacy(cfg)
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(
                "Legacy notification source is unavailable or invalid; repair it before resolving."
            ) from exc
        with policy_guard(runtime.root, blocking=False):
            current = load_policy_unlocked(runtime.root, "project", project_id)
            if current.get("legacy", {}).get("status") != "legacy_conflict":
                raise ValueError("No notification legacy conflict is pending for this project.")
            override = current["override"] if prefer == "canonical" else _converted_override(runtime, snapshot)
            destination = override.get("destination") if override is not None else None
            if isinstance(destination, dict) and destination.get("source") == "private_file":
                mark_reference(runtime.root, destination["credential_id"])
            replace_policy_unlocked(
                runtime.root,
                "project",
                current["revision"],
                override,
                project_id,
                legacy=_new_metadata(snapshot, current["revision"] + 1),
            )
    return project_id
