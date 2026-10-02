"""Local retained-capture coordinator using the common Project-I/O arbiter."""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config_types import RootConfig
from ..runtime.authority_scan import is_path_present
from ..runtime.locks import exclusive
from ..runtime.responsibility import responsibility_root
from ..runtime.responsibility_backfill import ResponsibilityBackfill, _canonical_capture_root
from ..runtime.responsibility_capture import CAPTURE_FILE, GENERATION_FILE, SourceHold, is_same_capture_owner
from ..runtime.responsibility_completion import (
    _digest,
    prepare_completed_source_release,
    publish_capture_completion,
    read_capture_completion,
)
from ..runtime.responsibility_generation import (
    CaptureGenerationTransition,
    apply_generation_reset,
    finish_generation_reset,
    generation_source_hold,
    load_capture_generation_transition,
)
from ..runtime.responsibility_process_capture import RunnerProcessCapture
from ..runtime.responsibility_store import Conflict, Ledger, Unavailable
from ..runtime.store import read_json
from .context import MachineRuntime, ProjectBinding
from .recovery_capture import recovery_owner
from .recovery_transport import decode_capture_locator, decode_source_hold

_MAX_CAPTURES = 64
_PROJECTS_PER_SLICE = 4
_OPERATIONS = {
    "recovery_admission": "advance_recovery_admission",
    "recovery_source_hold": "advance_recovery_source_holds",
    "legacy_capture_scan": "advance_legacy_capture_scans",
    "legacy_capture_read": "advance_legacy_capture_reads",
    "recovery_capture_transition": "advance_recovery_capture_transitions",
    "recovery_group_authority": "advance_recovery_group_authority",
    "recovery_source_release": "advance_recovery_source_releases",
}


@dataclass(slots=True)
class _Capture:
    binding: ProjectBinding
    operation: str = "recovery_admission"
    parameters: dict[str, Any] | None = None
    processes: RunnerProcessCapture | None = None
    evidence: ResponsibilityBackfill | None = None
    transition: CaptureGenerationTransition | None = None
    proof: dict | None = None
    is_parked: bool = False

    def close(self) -> None:
        if self.processes is not None:
            self.processes.close()
        if self.evidence is not None:
            self.evidence.close()

    @property
    def has_open_scan(self) -> bool:
        return self.processes is not None and self.processes.has_open_scan


class RecoveryEnrollment:
    """Advance local effects without threads, shared reads, or worker joins.

    Dispatch advances this coordinator inside its common admission turn. Fixed
    checkpoints own restart/eviction recovery; unfinished process iterators stay
    resident until EOF rather than repeatedly scanning a prefix at 65 bindings.
    """

    def __init__(self, runtime: MachineRuntime) -> None:
        self.runtime = runtime
        self._settled: frozenset[ProjectBinding] = frozenset()
        self._captures: OrderedDict[ProjectBinding, _Capture] = OrderedDict()
        self._cursor = 0
        self._is_stopped = False

    def _refresh_pending(self) -> None:
        _revision, bindings = self.runtime.load_registry_snapshot()
        self._settled = self._settled.intersection(bindings)
        self.runtime.recovery_enrollment_pending_projects = {
            binding.project_id for binding in bindings if binding not in self._settled
        }

    def poll(self) -> None:
        """Refresh idle obligations using only the local registry."""
        if not self._is_stopped:
            self._refresh_pending()

    def stop(self) -> None:
        self._is_stopped = True
        for capture in self._captures.values():
            capture.close()
        self._captures.clear()

    @contextmanager
    def _owned(self, binding: ProjectBinding) -> Iterator[None]:
        runtime = self.runtime
        with runtime._scheduler_authority_gate:
            if runtime._scheduler_authority_pid != os.getpid():
                raise Conflict("retained capture requires scheduler authority")
            with runtime.registry_guard(blocking=False) as acquired:
                if not acquired or binding not in runtime.load_registry_snapshot()[1]:
                    raise Conflict("capture binding changed")
                if is_path_present(runtime.paths["registration_transaction"]):
                    raise Conflict("capture registration rollback is pending")
                with runtime.binding_commit_guard(binding, blocking=False) as acquired:
                    if not acquired:
                        raise Conflict("capture local binding is busy")
                    yield

    def _root(self, binding: ProjectBinding) -> Path:
        return self.runtime.project_paths(binding.project_id)["root"]

    def _source(self, binding: ProjectBinding) -> Path | None:
        path = self.runtime.migration_path(binding.project_id)
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
            raise Unavailable("capture requires this binding's completed migration")
        raw = migration.get("legacy_runtime_root")
        if not isinstance(raw, str) or str(_canonical_capture_root(Path(raw))) != raw:
            raise Unavailable("capture migration source is not canonical")
        source = Path(raw)
        if source == self._root(binding):
            raise Unavailable("capture source equals its target")
        return source

    @staticmethod
    def _request(capture: _Capture, operation: str, parameters: Mapping[str, Any]) -> None:
        capture.operation = operation
        capture.parameters = dict(parameters)

    def _prepare(self, capture: _Capture) -> None:
        binding = capture.binding
        root, source = self._root(binding), self._source(binding)
        owner = recovery_owner(self.runtime, binding)
        has_transition = is_path_present(root / GENERATION_FILE)
        proof = None if has_transition else read_capture_completion(root)
        if proof is not None and (
            not is_same_capture_owner(proof, owner)
            or proof["legacy_source"] != (str(source) if source is not None else None)
        ):
            raise Unavailable("capture completion belongs to another binding or source")
        if has_transition or (
            proof is not None and proof["registration_generation"] != binding.registration_generation
        ):
            transition = load_capture_generation_transition(root, legacy_source=source, owner=owner)
            capture.transition = transition
            if source is None:
                apply_generation_reset(transition, None)
                finish_generation_reset(transition, None)
                capture.transition = None
                self._prepare(capture)
                return
            # Intent alone is not proof that the reset committed. Replaying
            # retention is required when publication stopped before CAP changed.
            is_reset = has_transition and read_json(root / CAPTURE_FILE) == transition.after
            self._request(
                capture,
                "recovery_capture_transition",
                {
                    "completion_digest": _digest(transition.state["previous_completion"]),
                    "phase": "normalize" if is_reset else "retain",
                },
            )
            return
        if proof is not None:
            capture.proof = proof
            self._request(capture, "recovery_group_authority", {"completion_digest": _digest(proof)})
            return
        root.mkdir(parents=True, exist_ok=True)
        with exclusive(root / "locks/responsibility-initialize.lock"):
            ledger = Ledger.open_or_create(responsibility_root(root))
        cfg = RootConfig.from_canonical_paths(
            binding.shared_root, binding.shared_root.parent, binding.machine_name, root
        )
        capture.processes = RunnerProcessCapture(cfg, ledger, legacy_source=source)
        state = capture.processes.checkpoint.prepare_local()
        if source is None:
            self._begin(capture, None)
        else:
            self._request(capture, "recovery_source_hold", {"capture_id": state["capture_id"]})

    def _begin(self, capture: _Capture, hold: SourceHold | None) -> None:
        previous = capture.processes
        previous.close()
        runner = RunnerProcessCapture(
            previous.cfg,
            previous.checkpoint.ledger,
            legacy_source=previous.checkpoint.legacy_source,
            source_hold=hold,
        )
        capture.processes = runner
        runner.prepare_admission(recovery_owner(self.runtime, capture.binding))
        runner.restart_after_reboot()
        capture.evidence = ResponsibilityBackfill(self._root(capture.binding), process_capture=runner)
        capture.parameters = None

    def _advance_local(self, capture: _Capture) -> None:
        if capture.parameters is not None:
            return
        if capture.proof is not None:
            receipt = prepare_completed_source_release(self._root(capture.binding))
            if receipt is None:
                capture.is_parked = True
                return
            if capture.proof["legacy_source"] is None:
                self._settled = self._settled.union((capture.binding,))
            else:
                self._request(capture, "recovery_source_release", {"completion_digest": receipt["completion_digest"]})
            return
        if capture.processes is None:
            self._request(capture, "recovery_admission", {})
            return
        if not capture.processes.take(64).is_sweep_complete:
            return
        progress = capture.evidence.take(64, should_cross_lanes=True, allow_source=False)
        if progress is None:
            return
        if progress.is_sweep_complete:
            capture.proof = publish_capture_completion(
                capture.evidence, owner=recovery_owner(self.runtime, capture.binding)
            )
            capture.close()
            self._request(capture, "recovery_group_authority", {"completion_digest": _digest(capture.proof)})
            return
        work = capture.evidence.next_source_work()
        if work is not None:
            operation = "legacy_capture_scan" if work["operation"] == "scan" else "legacy_capture_read"
            self._request(capture, operation, work["parameters"])

    def _apply(self, capture: _Capture, evidence: Mapping[str, Any]) -> None:
        state, operation = evidence.get("state"), capture.operation
        capture.is_parked = state in {"waiting", "stale", "blocked"} or evidence.get("outcome") == "unavailable"
        if state == "stale" and operation != "legacy_capture_scan":
            capture.close()
            capture.processes = None
            capture.evidence = None
            capture.transition = None
            capture.proof = None
            self._request(capture, "recovery_admission", {})
            return
        if operation == "recovery_admission" and state == "ready":
            self._prepare(capture)
        elif operation == "recovery_admission" and state == "superseded":
            self._settled = self._settled.union((capture.binding,))
            capture.parameters = None
        elif operation == "recovery_source_hold" and state == "retained":
            self._begin(capture, decode_source_hold(evidence["hold"], capture.parameters["capture_id"]))
        elif operation == "legacy_capture_scan" and state in {"observed", "stale"}:
            capture.evidence.apply_source_scan(capture.parameters, evidence["scan"])
            capture.parameters = None
        elif operation == "legacy_capture_read" and state == "observed":
            locator = decode_capture_locator(evidence["locator"], capture.parameters)
            capture.evidence.apply_source_record(capture.parameters, locator)
            capture.parameters = None
        elif operation == "recovery_group_authority" and state == "active":
            capture.parameters = None
        elif operation == "recovery_source_release" and state == "released":
            self._settled = self._settled.union((capture.binding,))
            capture.parameters = None
        elif operation == "recovery_capture_transition" and state == "retained":
            apply_generation_reset(capture.transition, generation_source_hold(capture.transition))
            self._request(capture, operation, {**capture.parameters, "phase": "normalize"})
        elif operation == "recovery_capture_transition" and state == "normalized":
            finish_generation_reset(capture.transition, generation_source_hold(capture.transition))
            capture.transition = None
            self._prepare(capture)

    def advance(self, controller: Any, bindings: Sequence[ProjectBinding], registry_revision: int) -> None:
        """Offer bounded recovery intents inside the dispatch admission turn."""
        if self._is_stopped or self.runtime._scheduler_authority_pid != os.getpid():
            return
        current = set(bindings)
        self._settled = self._settled.intersection(current)
        for binding in tuple(self._captures):
            if binding not in current or binding in self._settled:
                self._captures.pop(binding).close()
        protected = {request.project_id for request in controller.executor.unresolved_requests()}
        count = len(bindings)
        for offset in range(min(_PROJECTS_PER_SLICE, count)):
            binding = bindings[(self._cursor + offset) % count]
            if binding in self._settled:
                continue
            capture = self._captures.get(binding)
            if capture is None:
                if len(self._captures) >= _MAX_CAPTURES:
                    victim = next(
                        (
                            key
                            for key, entry in self._captures.items()
                            if entry.is_parked and not entry.has_open_scan and key.project_id not in protected
                        ),
                        None,
                    )
                    if victim is None:
                        continue
                    self._captures.pop(victim).close()
                capture = _Capture(binding, parameters={})
                self._captures[binding] = capture
            try:
                with self._owned(binding):
                    self._advance_local(capture)
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                capture.close()
                capture.is_parked = True
                continue
            self._captures.move_to_end(binding)
        if count:
            self._cursor = (self._cursor + min(_PROJECTS_PER_SLICE, count)) % count
        for operation, method in _OPERATIONS.items():
            parameters = {
                binding.project_id: capture.parameters
                for binding, capture in self._captures.items()
                if capture.operation == operation and capture.parameters is not None
            }
            results = getattr(controller, method)(bindings, registry_revision, parameters)
            for binding, capture in tuple(self._captures.items()):
                if capture.operation != operation or binding.project_id not in results:
                    continue
                try:
                    with self._owned(binding):
                        self._apply(capture, results[binding.project_id])
                except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                    # Consumed transport is never completion. Durable domain
                    # checkpoints reconstruct the exact step on the next turn.
                    capture.close()
                    self._captures.pop(binding)
        self._refresh_pending()
