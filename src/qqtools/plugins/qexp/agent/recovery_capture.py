"""Machine-owned enrollment of retained writers and their final evidence sweep."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from enum import Enum
from pathlib import Path

from ..config_types import RootConfig
from ..layout import LOCAL_RECOVERY_CAPABILITY
from ..runtime.authority_scan import is_path_present
from ..runtime.group_namespace import activate_group_authority_locked
from ..runtime.locks import exclusive, schema_lock
from ..runtime.paths import shared_paths
from ..runtime.responsibility import responsibility_root
from ..runtime.responsibility_backfill import ResponsibilityBackfill
from ..runtime.responsibility_capture import CAPTURE_BYTES, CAPTURE_FILE, GENERATION_FILE, is_same_capture_owner
from ..runtime.responsibility_completion import (
    publish_capture_completion,
    read_capture_completion,
    release_completed_source,
)
from ..runtime.responsibility_generation import restart_capture_generation
from ..runtime.responsibility_process_capture import RunnerProcessCapture
from ..runtime.responsibility_store import Conflict, Ledger, Unavailable
from ..runtime.store import read_json, read_json_limited
from .context import RECOVERY_REGISTRATION_VERSION, MachineRuntime, ProjectBinding


class RecoveryProgress(Enum):
    """A bounded step either advances work, waits for a condition, or completes."""

    ADVANCED = "advanced"
    WAITING = "waiting"
    COMPLETE = "complete"


def recovery_owner(runtime: MachineRuntime, binding: ProjectBinding) -> dict:
    return {
        "project_id": binding.project_id,
        "shared_root": str(binding.shared_root),
        "machine_name": binding.machine_name,
        "owner_root": str(runtime.root),
        "owner_instance": runtime.instance_id,
        "registration_generation": binding.registration_generation,
    }


def recovery_source(runtime: MachineRuntime, binding: ProjectBinding) -> Path | None:
    path = runtime.migration_path(binding.project_id)
    if not is_path_present(path):
        return None
    migration = read_json(path).get("migration")
    if not isinstance(migration, dict) or any(
        migration.get(key) != value
        for key, value in {
            "project_id": binding.project_id,
            "shared_root": str(binding.shared_root),
            "machine_name": binding.machine_name,
            "state": "active",
        }.items()
    ):
        raise Unavailable("capture requires this binding's completed legacy migration")
    source = migration.get("legacy_runtime_root")
    if (
        not isinstance(source, str)
        or not source
        or not Path(source).is_absolute()
        or str(Path(source).resolve()) != source
        or Path(source) == runtime.project_paths(binding.project_id)["root"]
    ):
        raise Unavailable("capture migration has no canonical legacy source")
    return Path(source)


def inspect_recovery_capture(runtime: MachineRuntime, binding: ProjectBinding) -> dict:
    """Inspect fixed capture metadata without advancing or repairing enrollment."""
    root = runtime.project_paths(binding.project_id)["root"]
    try:
        proof = read_capture_completion(root, should_sync=False)
        if proof is not None:
            if (
                proof["project_id"] != binding.project_id
                or proof["shared_root"] != str(binding.shared_root)
                or proof["machine_name"] != binding.machine_name
                or proof["owner_root"] != str(runtime.root)
                or proof["owner_instance"] != binding.runtime_instance_id
                or proof["registration_generation"] != binding.registration_generation
            ):
                raise Unavailable("capture completion belongs to another binding")
            source = proof["legacy_source"]
            state = (
                "captured_source_retained"
                if source is not None and is_path_present(Path(source) / CAPTURE_FILE)
                else "captured"
            )
        elif not is_path_present(root / CAPTURE_FILE):
            state = "not_started"
        else:
            capture = read_json_limited(root / CAPTURE_FILE, max_bytes=CAPTURE_BYTES)
            progress = capture.get("progress")
            state = (
                "capturing_evidence"
                if isinstance(progress, dict) and progress.get("is_sweep_complete") is True
                else "capturing_processes"
            )
        return {"state": state, "diagnostic_only": True}
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        return {"state": "unavailable", "detail": str(exc), "diagnostic_only": True}


class RecoveryCapture:
    """Advance at most 64 process entries and 64 evidence work units per pass.

    Initial retention and final publication hold lifecycle/ownership guards.
    Intermediate capture owns only retained local storage, so slow history I/O
    cannot hold the machine migration or registry lock across a discovery slice.
    """

    def __init__(self, runtime: MachineRuntime, binding: ProjectBinding) -> None:
        self.runtime = runtime
        self.binding = binding
        self.root = runtime.project_paths(binding.project_id)["root"]
        self.processes: RunnerProcessCapture | None = None
        self.evidence: ResponsibilityBackfill | None = None

    def close(self) -> None:
        if self.processes is not None:
            self.processes.close()
        if self.evidence is not None:
            self.evidence.close()

    def _source(self) -> Path | None:
        return recovery_source(self.runtime, self.binding)

    def _owner(self) -> dict:
        return recovery_owner(self.runtime, self.binding)

    @contextmanager
    def _owned(self) -> Iterator[RootConfig]:
        runtime, binding = self.runtime, self.binding
        with runtime._scheduler_authority_gate:
            if runtime._scheduler_authority_pid != os.getpid():
                raise Conflict("retained capture requires scheduler authority")
            # Registry ownership excludes binding/root removal. Completed legacy
            # migration is immutable; capture's parent locks exclude evidence
            # movement. Do not wait for the machine-wide dispatch-cycle lock.
            with runtime.registry_guard() as registered:
                if not registered or binding not in runtime.load_registry()[1]:
                    raise Conflict("capture binding changed")
                with runtime.binding_write_guard(binding) as eligible:
                    if (
                        not eligible
                        or runtime.registration_status(binding)["registration_version"] != RECOVERY_REGISTRATION_VERSION
                    ):
                        raise Conflict("capture registration is not prepared")
                    cfg = replace(binding.root_config(), runtime_root=self.root)
                    with schema_lock(cfg.shared_root, blocking=False) as acquired:
                        if not acquired:
                            raise Conflict("capture admission schema is busy")
                        capabilities = read_json(shared_paths(cfg.shared_root)["schema"] / "version.json")["schema"][
                            "required_capabilities"
                        ]
                        if LOCAL_RECOVERY_CAPABILITY not in capabilities:
                            raise Unavailable("capture requires the shared admission fence")
                        yield cfg

    def _begin(self) -> None:
        with self._owned() as cfg:
            source = self._source()
            self.root.mkdir(parents=True, exist_ok=True)
            with exclusive(self.root / "locks" / "responsibility-initialize.lock"):
                ledger = Ledger.open_or_create(responsibility_root(self.root))
            processes = RunnerProcessCapture(cfg, ledger, legacy_source=source)
            # Establish both retention markers before releasing lifecycle locks.
            processes.prepare_admission(self._owner())
        try:
            processes.restart_after_reboot()
        except BaseException:
            processes.close()
            raise
        self.processes = processes
        self.evidence = ResponsibilityBackfill(self.root, process_capture=processes)

    def advance(self) -> bool:
        """Return capture completion for callers that do not schedule another step."""
        return self.advance_step() is RecoveryProgress.COMPLETE

    def advance_step(self) -> RecoveryProgress:
        """Advance bounded capture and report whether another step can do work."""
        if self.runtime._scheduler_authority_pid != os.getpid():
            raise Conflict("retained capture requires scheduler authority")
        if is_path_present(self.root / GENERATION_FILE):
            self._restart_generation()
        proof = read_capture_completion(self.root)
        if proof is not None:
            owner = self._owner()
            if not is_same_capture_owner(proof, owner):
                raise Unavailable("capture completion belongs to another binding")
            source = self._source()
            if proof["legacy_source"] != (str(source) if source is not None else None):
                raise Unavailable("capture source changed after completion")
            if proof["registration_generation"] == owner["registration_generation"]:
                self.close()
                return RecoveryProgress.COMPLETE
            self._restart_generation()
        if self.processes is None:
            self._begin()
        if not self.processes.take(64).is_sweep_complete:
            return RecoveryProgress.ADVANCED
        progress = self.evidence.take(64, should_cross_lanes=True)
        if progress is not None and progress.is_sweep_complete:
            with self._owned():
                if self._source() != self.processes.checkpoint.legacy_source:
                    raise Conflict("capture migration source changed")
                publish_capture_completion(self.evidence, owner=self._owner())
            self.close()
            return RecoveryProgress.COMPLETE
        return RecoveryProgress.WAITING if progress is None else RecoveryProgress.ADVANCED

    def _has_current_completion(self) -> bool:
        proof = read_capture_completion(self.root)
        if proof is None:
            return False
        owner = self._owner()
        source = self._source()
        if (
            not is_same_capture_owner(proof, owner)
            or proof["registration_generation"] != owner["registration_generation"]
            or proof["legacy_source"] != (str(source) if source is not None else None)
        ):
            raise Unavailable("capture completion belongs to another binding or source")
        return True

    def release_source(self) -> bool:
        """Reclaim the retained source only after its captured responsibilities retire."""
        with self.runtime._scheduler_authority_gate:
            if self.runtime._scheduler_authority_pid != os.getpid():
                raise Conflict("source release requires scheduler authority")
            with self.runtime.registry_guard() as registered:
                if not registered or self.binding not in self.runtime.load_registry()[1]:
                    raise Conflict("source release binding changed")
                if not self._has_current_completion():
                    return False
                return release_completed_source(self.root)

    def activate_group_authority(self) -> bool:
        """Isolate Group truth from capture proof, independently of source release."""
        from ..runtime.recovery_admission import _upgrade_blocker

        with self._owned() as cfg:
            if not self._has_current_completion():
                return False
            if _upgrade_blocker(cfg) is not None:
                return False
            return activate_group_authority_locked(cfg)

    def _restart_generation(self) -> None:
        with self._owned():
            restart_capture_generation(self.root, legacy_source=self._source(), owner=self._owner())
        self.close()
        self.processes = None
        self.evidence = None
