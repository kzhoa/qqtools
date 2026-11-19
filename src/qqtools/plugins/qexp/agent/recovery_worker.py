"""Closed worker handler for isolated recovery transactions."""

from __future__ import annotations

import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config_types import RootConfig
from ..layout import load_machine_record, load_root_config
from ..runtime.paths import machine_project_paths, machine_runtime_paths
from ..runtime.registration_authority import RegistrationIdentity
from ..runtime.store import fenced_mutations, read_json_limited
from .project_io_protocol import PROJECT_IO_MAX_RECORD_BYTES, ProjectIORequest


class RecoveryExecutorEpochFenced(RuntimeError):
    """The recovery request lost its executor epoch."""


class _BindingAuthorityChanged(RuntimeError):
    """The recovery request lost its binding or transaction context."""


class _RecoveryPreparationDeferred(RuntimeError):
    """Local rollback or migration has not yet established recovery eligibility."""


@dataclass(frozen=True)
class RecoveryWorkerIdentity:
    """Live worker identity checks bound to the request being handled."""

    is_epoch_current: Callable[[], bool]
    is_registry_binding_current: Callable[[], bool]
    is_claim_binding_current: Callable[[RootConfig], bool]

    def require_current_epoch(self) -> None:
        if not self.is_epoch_current():
            raise RecoveryExecutorEpochFenced

    def require_current_registry_binding(self) -> None:
        self.require_current_epoch()
        if not self.is_registry_binding_current():
            raise _BindingAuthorityChanged

    def require_current_claim_binding(self, cfg: RootConfig) -> None:
        self.require_current_epoch()
        if not self.is_claim_binding_current(cfg):
            raise _BindingAuthorityChanged


def _recovery_admission(
    request: ProjectIORequest,
    runtime_root: Path,
    cfg: RootConfig,
    worker_identity: RecoveryWorkerIdentity,
    write_possible: list[bool],
) -> dict[str, Any]:
    """Prepare shared recovery without MachineRuntime locks or record writes."""
    from ..runtime.authority_scan import is_path_present
    from ..runtime.recovery_admission import prepare_shared_recovery_admission
    from ..runtime.responsibility_store import DurableIO, Unavailable

    project_root = machine_project_paths(runtime_root, request.project_id)["root"]
    paths = machine_runtime_paths(runtime_root)
    journal_path = paths["registration_transaction"]
    migration_path = project_root / "migration.json"
    identity = RegistrationIdentity(
        request.project_id, request.registration_generation, request.runtime_id, str(runtime_root)
    )

    def migration_context() -> dict | None:
        if not is_path_present(migration_path):
            return None
        if not stat.S_ISREG(migration_path.lstat().st_mode):
            raise Unavailable("recovery preparation requires regular migration metadata")
        migration = read_json_limited(migration_path, max_bytes=PROJECT_IO_MAX_RECORD_BYTES).get("migration")
        if not isinstance(migration, dict):
            raise Unavailable("recovery preparation migration metadata is invalid")
        if migration.get("state") != "active" or any(
            migration.get(key) != value
            for key, value in {
                "project_id": request.project_id,
                "shared_root": request.canonical_shared_root,
                "machine_name": request.parameters["machine_name"],
            }.items()
        ):
            raise _RecoveryPreparationDeferred
        return migration

    waiting = {"state": "waiting", "registration_prepared": False, "admission_fenced": False}

    def rollback_pending_registration() -> bool:
        """Replay one durable CLI snapshot under machine-local exclusion."""
        from ..runtime.locks import exclusive
        from .registration import MachineRegistration

        registration = MachineRegistration(
            runtime_root,
            ensure_layout=lambda: None,
            current_instance_id=lambda: request.runtime_id,
        )
        with exclusive(paths["locks"] / "migration.lock", blocking=False) as migration_acquired:
            if not migration_acquired:
                return False
            with registration.registry_guard(blocking=False) as registry_acquired:
                if not registry_acquired or not is_path_present(journal_path):
                    return False
                worker_identity.require_current_registry_binding()

                def before_effect() -> None:
                    worker_identity.require_current_epoch()
                    write_possible[0] = True

                registration.rollback_pending_locked(before_effect=before_effect)
                return True

    worker_identity.require_current_registry_binding()
    try:
        if is_path_present(journal_path):
            if rollback_pending_registration():
                return waiting
            if is_path_present(journal_path):
                return waiting
        migration = migration_context()
        from ..runtime.registration_authority import observe_registration_owner

        def observation_is_current() -> bool:
            return (
                worker_identity.is_epoch_current()
                and worker_identity.is_registry_binding_current()
                and not is_path_present(journal_path)
                and migration_context() == migration
            )

        if observe_registration_owner(cfg, identity, is_current=observation_is_current) == "superseded":
            return {"state": "superseded", "registration_prepared": False, "admission_fenced": False}
        worker_identity.require_current_claim_binding(cfg)
        machine_record = load_machine_record(cfg)
        machine = machine_record.get("machine") if isinstance(machine_record, dict) else None
        if not isinstance(machine, dict) or machine.get("agent_runtime") != "machine":
            return waiting

        def local_current() -> bool:
            return (
                worker_identity.is_registry_binding_current()
                and not is_path_present(journal_path)
                and migration_context() == migration
            )

        def before_shared_write() -> None:
            worker_identity.require_current_claim_binding(cfg)
            if not local_current() or load_machine_record(cfg) != machine_record:
                raise _RecoveryPreparationDeferred
            write_possible[0] = True

        def before_registration_conversion() -> None:
            worker_identity.require_current_claim_binding(cfg)
            if not local_current():
                raise _RecoveryPreparationDeferred
            # Shared registration exclusion prevents a current CLI from taking
            # an old snapshot in the gap between this absence barrier and v2.
            # This sync changes no MachineRuntime file and needs no local guard.
            DurableIO().sync_directory(journal_path.parent, "recovery_registration")
            worker_identity.require_current_claim_binding(cfg)
            if not local_current():
                raise _RecoveryPreparationDeferred

        prepared = prepare_shared_recovery_admission(
            cfg,
            identity,
            is_current=local_current,
            before_registration_conversion=before_registration_conversion,
            before_shared_write=before_shared_write,
        )
        worker_identity.require_current_claim_binding(cfg)
        if not local_current():
            raise _RecoveryPreparationDeferred
    except _RecoveryPreparationDeferred:
        if write_possible[0]:
            raise _BindingAuthorityChanged from None
        return waiting
    return {
        "state": "ready" if prepared.admission.is_fenced else "waiting",
        "registration_prepared": prepared.is_registration_prepared,
        "admission_fenced": prepared.admission.is_fenced,
    }


def _recovery_capture_transition(
    request: ProjectIORequest,
    runtime_root: Path,
    cfg: RootConfig,
    worker_identity: RecoveryWorkerIdentity,
    write_possible: list[bool],
) -> dict[str, Any]:
    """Fence generation handoff source effects; local reset belongs to the coordinator."""
    from ..runtime.authority_scan import is_path_present
    from ..runtime.responsibility_backfill import _canonical_capture_root
    from ..runtime.responsibility_capture import CAPTURE_BYTES, CAPTURE_FILE, GENERATION_FILE
    from ..runtime.responsibility_completion import COMPLETION_FILE, SOURCE_RELEASE_FILE, _digest
    from ..runtime.responsibility_generation import (
        load_capture_generation_transition,
        normalize_generation_source,
        retain_generation_source,
    )
    from ..runtime.responsibility_source_read import SourceCaptureStale
    from ..runtime.responsibility_store import Unavailable

    target = machine_project_paths(runtime_root, request.project_id)["root"]
    migration_path = target / "migration.json"
    owner = {
        "project_id": request.project_id,
        "shared_root": request.canonical_shared_root,
        "machine_name": request.parameters["machine_name"],
        "owner_root": str(runtime_root),
        "owner_instance": request.runtime_id,
        "registration_generation": request.registration_generation,
    }

    def transition_context():
        migration = None
        source = None
        if is_path_present(migration_path):
            if not stat.S_ISREG(migration_path.lstat().st_mode):
                raise Unavailable("capture transition requires regular migration metadata")
            migration = read_json_limited(migration_path, max_bytes=PROJECT_IO_MAX_RECORD_BYTES).get("migration")
            if not isinstance(migration, dict) or any(
                migration.get(key) != value
                for key, value in {
                    "project_id": request.project_id,
                    "shared_root": request.canonical_shared_root,
                    "machine_name": request.parameters["machine_name"],
                    "state": "active",
                }.items()
            ):
                raise SourceCaptureStale("capture transition migration is not current")
            raw_source = migration.get("legacy_runtime_root")
            if not isinstance(raw_source, str) or str(Path(raw_source)) != raw_source:
                raise Unavailable("capture transition source is not canonical")
            source = _canonical_capture_root(Path(raw_source))
        transition = load_capture_generation_transition(target, legacy_source=source, owner=owner)
        if _digest(transition.state["previous_completion"]) != request.parameters["completion_digest"]:
            raise SourceCaptureStale("capture transition completion identity changed")
        current = read_json_limited(target / CAPTURE_FILE, max_bytes=CAPTURE_BYTES)
        presence = tuple(
            is_path_present(target / name) for name in (GENERATION_FILE, COMPLETION_FILE, SOURCE_RELEASE_FILE)
        )
        if request.parameters["phase"] == "normalize":
            if not presence[0] or presence[1] or presence[2] or current != transition.after:
                raise SourceCaptureStale("source normalization requires the completed local reset")
        elif current != transition.before:
            raise SourceCaptureStale("source retention phase was already locally reset")
        return transition, migration, presence

    worker_identity.require_current_claim_binding(cfg)
    try:
        context = transition_context()

        def before_source_write() -> None:
            worker_identity.require_current_claim_binding(cfg)
            if transition_context() != context:
                raise SourceCaptureStale("capture transition local context changed before source effect")
            write_possible[0] = True

        transaction = (
            retain_generation_source if request.parameters["phase"] == "retain" else normalize_generation_source
        )
        transaction(context[0], before_source_write=before_source_write)
        worker_identity.require_current_claim_binding(cfg)
        if transition_context() != context:
            raise SourceCaptureStale("capture transition local context changed before result publication")
    except SourceCaptureStale:
        if write_possible[0]:
            raise _BindingAuthorityChanged from None
        return {"state": "stale"}
    return {"state": "retained" if request.parameters["phase"] == "retain" else "normalized"}


def _recovery_group_authority(
    request: ProjectIORequest,
    runtime_root: Path,
    cfg: RootConfig,
    worker_identity: RecoveryWorkerIdentity,
    write_possible: list[bool],
) -> dict[str, Any]:
    """Activate shared Group truth only from this binding's exact local completion."""
    from ..runtime.authority_scan import is_path_present
    from ..runtime.group_namespace import activate_group_authority_locked
    from ..runtime.locks import schema_lock
    from ..runtime.recovery_admission import _upgrade_blocker
    from ..runtime.registration_authority import registration_write_guard
    from ..runtime.registration_protocol import has_recovery_protocol
    from ..runtime.responsibility_capture import GENERATION_FILE, capture_admission
    from ..runtime.responsibility_completion import _digest, read_capture_completion
    from ..runtime.responsibility_source_read import SourceCaptureStale

    target = machine_project_paths(runtime_root, request.project_id)["root"]
    identity = RegistrationIdentity(
        request.project_id, request.registration_generation, request.runtime_id, str(runtime_root)
    )
    owner = capture_admission(
        {
            "project_id": request.project_id,
            "shared_root": request.canonical_shared_root,
            "machine_name": request.parameters["machine_name"],
            "owner_root": str(runtime_root),
            "owner_instance": request.runtime_id,
            "registration_generation": request.registration_generation,
        }
    )

    def completion_context() -> dict:
        if is_path_present(target / GENERATION_FILE):
            raise SourceCaptureStale("Group activation capture is transitioning")
        proof = read_capture_completion(target, should_sync=False)
        if (
            proof is None
            or capture_admission(proof) != owner
            or _digest(proof) != request.parameters["completion_digest"]
        ):
            raise SourceCaptureStale("Group activation requires this exact binding's completion")
        return proof

    worker_identity.require_current_claim_binding(cfg)
    try:
        proof = completion_context()

        def is_current() -> bool:
            worker_identity.require_current_claim_binding(cfg)
            return completion_context() == proof

        def before_shared_write() -> None:
            if not is_current():
                raise SourceCaptureStale("Group activation completion changed before mutation")
            write_possible[0] = True

        with registration_write_guard(
            cfg, identity, is_current=is_current, before_shared_write=before_shared_write
        ) as registration:
            if registration is None or not has_recovery_protocol(registration):
                return {"state": "waiting"}
            with schema_lock(cfg.shared_root, blocking=False) as acquired:
                if not acquired or _upgrade_blocker(cfg) is not None:
                    return {"state": "waiting"}
                with fenced_mutations(cfg.shared_root, before_shared_write):
                    is_active = activate_group_authority_locked(cfg)
                if not is_current():
                    raise SourceCaptureStale("Group activation completion changed before result")
    except SourceCaptureStale:
        if write_possible[0]:
            raise _BindingAuthorityChanged from None
        return {"state": "stale"}
    return {"state": "active" if is_active else "waiting"}


def _recovery_source_release(
    request: ProjectIORequest,
    runtime_root: Path,
    cfg: RootConfig,
    worker_identity: RecoveryWorkerIdentity,
    write_possible: list[bool],
) -> dict[str, Any]:
    """Consume a local durable retirement intent and release only source retention."""
    from ..runtime.authority_scan import is_path_present
    from ..runtime.responsibility_capture import GENERATION_FILE, capture_admission
    from ..runtime.responsibility_completion import (
        SOURCE_RELEASE_FILE,
        _digest,
        load_source_release_context,
        read_capture_completion,
        release_source_retention,
    )
    from ..runtime.responsibility_source_read import SourceCaptureStale

    target = machine_project_paths(runtime_root, request.project_id)["root"]
    owner = capture_admission(
        {
            "project_id": request.project_id,
            "shared_root": request.canonical_shared_root,
            "machine_name": request.parameters["machine_name"],
            "owner_root": str(runtime_root),
            "owner_instance": request.runtime_id,
            "registration_generation": request.registration_generation,
        }
    )

    def capture_context():
        if is_path_present(target / GENERATION_FILE) or not is_path_present(target / SOURCE_RELEASE_FILE):
            raise SourceCaptureStale("source release has no current local retirement intent")
        proof = read_capture_completion(target, should_sync=False)
        if (
            proof is None
            or capture_admission(proof) != owner
            or _digest(proof) != request.parameters["completion_digest"]
        ):
            raise SourceCaptureStale("source release completion belongs to another owner or revision")
        return load_source_release_context(target, completion_digest=request.parameters["completion_digest"])

    worker_identity.require_current_claim_binding(cfg)
    try:
        context = capture_context()

        def before_source_write() -> None:
            worker_identity.require_current_claim_binding(cfg)
            if capture_context() != context:
                raise SourceCaptureStale("source release context changed before source effect")
            write_possible[0] = True

        is_released = release_source_retention(context, before_source_write=before_source_write)
        worker_identity.require_current_claim_binding(cfg)
        if capture_context() != context:
            raise SourceCaptureStale("source release context changed before result publication")
    except SourceCaptureStale:
        if write_possible[0]:
            raise _BindingAuthorityChanged from None
        return {"state": "stale"}
    return {"state": "released" if is_released else "waiting"}


def _recovery_source_hold(
    request: ProjectIORequest,
    runtime_root: Path,
    cfg: RootConfig,
    worker_identity: RecoveryWorkerIdentity,
    write_possible: list[bool],
) -> dict[str, Any]:
    """Own source retention only; exact target metadata is read-only."""
    from ..runtime.responsibility_capture import retain_capture_source
    from ..runtime.responsibility_source_read import SourceCaptureStale, load_source_hold_context
    from ..runtime.responsibility_store import Unavailable

    project_root = machine_project_paths(runtime_root, request.project_id)["root"]
    owner = {
        "project_id": request.project_id,
        "shared_root": request.canonical_shared_root,
        "machine_name": request.parameters["machine_name"],
        "owner_root": str(runtime_root),
        "owner_instance": request.runtime_id,
        "registration_generation": request.registration_generation,
    }

    def capture_context() -> tuple[dict, dict]:
        capture = load_source_hold_context(project_root, owner, capture_id=request.parameters["capture_id"])
        migration_path = project_root / "migration.json"
        if not stat.S_ISREG(migration_path.lstat().st_mode):
            raise Unavailable("source retention requires regular completed migration metadata")
        migration = read_json_limited(migration_path, max_bytes=PROJECT_IO_MAX_RECORD_BYTES).get("migration")
        expected = {
            "state": "active",
            "legacy_runtime_root": capture["legacy_source"],
            "project_id": request.project_id,
            "shared_root": request.canonical_shared_root,
            "machine_name": request.parameters["machine_name"],
        }
        if not isinstance(migration, dict) or any(migration.get(key) != value for key, value in expected.items()):
            raise SourceCaptureStale("source retention migration identity changed")
        return capture, migration

    worker_identity.require_current_claim_binding(cfg)
    try:
        capture, migration = capture_context()

        def before_source_write() -> None:
            worker_identity.require_current_claim_binding(cfg)
            if capture_context() != (capture, migration):
                raise _BindingAuthorityChanged
            write_possible[0] = True

        hold = retain_capture_source(capture, before_source_write=before_source_write)
        worker_identity.require_current_claim_binding(cfg)
        if capture_context() != (capture, migration):
            raise _BindingAuthorityChanged
    except SourceCaptureStale:
        if write_possible[0]:
            raise _BindingAuthorityChanged from None
        return {"state": "stale", "hold": None}
    return {"state": "retained", "hold": hold.to_dict()}


_RECOVERY_HANDLERS = {
    "recovery_admission": _recovery_admission,
    "recovery_capture_transition": _recovery_capture_transition,
    "recovery_group_authority": _recovery_group_authority,
    "recovery_source_release": _recovery_source_release,
    "recovery_source_hold": _recovery_source_hold,
}
RECOVERY_OPERATION_KINDS = frozenset(_RECOVERY_HANDLERS)


def handle_recovery_request(
    request: ProjectIORequest,
    runtime_root: Path,
    worker_identity: RecoveryWorkerIdentity,
    write_possible: list[bool],
) -> dict[str, Any]:
    """Execute one closed recovery transaction with live worker identity checks.

    Mark the caller's write record before each attempted mutation so a lost
    result or stale identity after that point can be reported as ambiguous.
    """
    if not isinstance(request, ProjectIORequest):
        raise ValueError("Recovery handling requires a ProjectIORequest.")
    handler = _RECOVERY_HANDLERS.get(request.operation_kind)
    if handler is None:
        raise ValueError(f"unsupported recovery operation: {request.operation_kind!r}")

    cfg = load_root_config(Path(request.canonical_shared_root), request.parameters["machine_name"])
    return handler(request, runtime_root, cfg, worker_identity, write_possible)
