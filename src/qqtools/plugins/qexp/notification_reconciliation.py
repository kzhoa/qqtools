"""QQTOOLS-COMPAT-0016: Reconcile older project writers before canonical use."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .agent.context import MachineRuntime
from .config_types import RootConfig
from .notification_cleanup import mark_reference
from .notification_compat import LegacySnapshot, capture_legacy, legacy_override
from .notification_credentials import read_webhook, stage_webhook
from .notification_policy import load_policy_unlocked, policy_guard, replace_policy_unlocked
from .runtime.locks import machine_lock
from .runtime.store import fenced_mutations


class LegacyConflictError(ValueError):
    """An older writer changed the source after a canonical policy edit."""


class LegacySourceInvalidError(ValueError):
    """Shared legacy capture failed and its invalid status was persisted."""


class LegacySourceBusyError(RuntimeError):
    """The shared legacy source transaction has not acquired its lock."""


def _new_metadata(snapshot: LegacySnapshot, revision: int) -> dict[str, object]:
    return {"fingerprint": snapshot.fingerprint, "imported_revision": revision, "status": "ready"}


def _converted_override(runtime_root: Path, snapshot: LegacySnapshot) -> dict | None:
    credential_id = stage_webhook(runtime_root, snapshot.webhook) if snapshot.webhook else None
    if credential_id is not None:
        read_webhook(runtime_root, credential_id)
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
    return reconcile_captured_legacy(runtime.root, cfg, project_id)


def reconcile_captured_legacy(
    runtime_root: Path,
    cfg: RootConfig,
    project_id: str,
    *,
    before_shared_capture: Callable[[], None] | None = None,
    before_local_write: Callable[[], None] | None = None,
) -> str:
    """Reconcile one verified Project without taking any MachineRuntime authority.

    Shared capture precedes local bookkeeping. The local-write fence must perform
    only local identity checks, including when called beneath the policy lock.
    The capture fence runs under the shared machine lock, before any policy lock.
    Credential contents remain private and never leave this transaction.
    """
    with fenced_mutations(runtime_root, before_local_write):
        return _reconcile_captured_legacy(runtime_root, cfg, project_id, before_shared_capture, before_local_write)


def _reconcile_captured_legacy(
    runtime_root: Path,
    cfg: RootConfig,
    project_id: str,
    before_shared_capture: Callable[[], None] | None,
    before_local_write: Callable[[], None] | None,
) -> str:
    with machine_lock(cfg.shared_root, cfg.machine_name, blocking=False) as acquired:
        if not acquired:
            raise LegacySourceBusyError("Legacy notification source is busy; retry when the current update completes.")
        if before_shared_capture is not None:
            before_shared_capture()
        try:
            snapshot = capture_legacy(cfg)
        except (OSError, TypeError, ValueError) as exc:
            if before_local_write is not None:
                before_local_write()
            with policy_guard(runtime_root, blocking=False):
                current = load_policy_unlocked(runtime_root, "project", project_id)
                baseline = current.get("legacy")
                if baseline is None or baseline["status"] != "source_invalid":
                    previous_revision = 0 if baseline is None else baseline["imported_revision"]
                    imported_revision = (
                        current["revision"] + 1 if current["revision"] == previous_revision else previous_revision
                    )
                    replace_policy_unlocked(
                        runtime_root,
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
            raise LegacySourceInvalidError(
                "Legacy notification source is unavailable or invalid; inspect project configuration."
            ) from exc
        if before_local_write is not None:
            before_local_write()
        with policy_guard(runtime_root, blocking=False):
            current = load_policy_unlocked(runtime_root, "project", project_id)
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
                    runtime_root,
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
                        runtime_root,
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
            if before_local_write is not None:
                before_local_write()
            override = current["override"] if snapshot.is_absent else _converted_override(runtime_root, snapshot)
            destination = override.get("destination") if override is not None else None
            if isinstance(destination, dict) and destination.get("source") == "private_file":
                mark_reference(runtime_root, destination["credential_id"])
            replace_policy_unlocked(
                runtime_root,
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
            override = current["override"] if prefer == "canonical" else _converted_override(runtime.root, snapshot)
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
