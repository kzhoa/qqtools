"""Machine-local recovery of historical orphaned Attempts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from ..runtime.authority_scan import EvidenceScan
from ..runtime.paths import local_paths, machine_project_paths
from ..runtime.records import utc_now
from ..runtime.resources.cpu_lane import retag_cpu_if_matches
from ..runtime.resources.reservations import ReservationIdentity, retag_if_matches
from ..runtime.responsibility_cleanup import evidence_write_guard
from ..runtime.store import atomic_replace, read_json_limited
from ..runtime.termination import attempt_control_lock
from ..runtime.work_budget import diagnostic_increment
from .context import MachineRuntime, ProjectBinding
from .project_io_process import (
    authority_evidence_matches_intent,
    binding_signature,
    local_binding_config,
    read_isolated_authority_manifest,
    read_process_manifest,
)
from .project_io_terminal_proof import has_exact_terminal_revisions

if TYPE_CHECKING:
    from .project_io.controller import ProjectIOController

_MAX_BINDINGS_PER_TURN = 64
_MAX_SCANS = 256
_MAX_ENTRIES = 256
_SOURCE_ERRORS = (OSError, RuntimeError, ValueError, TypeError, KeyError, AttributeError)


class OrphanRecovery:
    """Scan local evidence, submit exact shared recovery, and apply the committed result."""

    def __init__(self, runtime: MachineRuntime, controller: ProjectIOController) -> None:
        self._runtime = runtime
        self._controller = controller
        self._scans: dict[tuple[str, ...], EvidenceScan] = {}
        self._scan_complete: set[tuple[str, ...]] = set()
        self._entries: dict[str, Mapping[str, Any]] = {}
        self._binding_offset = 0

    def has_pending(self, signature: tuple[str, ...]) -> bool:
        """Check for pending local work for the exact binding and registry revision."""
        return any(entry.get("binding_signature") == signature for entry in self._entries.values())

    def close(self) -> None:
        """Close open scans and reset the local recovery state."""
        for scan in self._scans.values():
            scan.close()
        self._scans.clear()
        self._scan_complete.clear()
        self._entries.clear()
        self._binding_offset = 0

    @staticmethod
    def _stable_binding_signature(binding: ProjectBinding) -> tuple[str | None, ...]:
        """Return the stable binding identity persisted by shared recovery."""
        return (
            binding.project_id,
            str(binding.shared_root),
            binding.machine_name,
            binding.registration_generation,
            binding.runtime_instance_id,
            binding.runtime_root,
        )

    def _read_reservation(
        self,
        reservation_id: str,
        *,
        allowed_states: frozenset[str] = frozenset({"active"}),
    ) -> tuple[str, ReservationIdentity] | None:
        paths = local_paths(self._runtime.root)
        found: list[tuple[str, ReservationIdentity]] = []
        lanes = (
            ("active", "active"),
            ("provisional", "provisional"),
            ("released", "released"),
            ("cpu_active", "active"),
            ("cpu_provisional", "provisional"),
            ("cpu_released", "released"),
        )
        for name, expected_state in lanes:
            try:
                envelope = read_json_limited(
                    paths[name] / f"{reservation_id}.json",
                    max_bytes=65_536,
                    record_type="reservation",
                )
            except FileNotFoundError:
                continue
            reservation = envelope.get("reservation") if isinstance(envelope, dict) else None
            if not isinstance(reservation, dict) or reservation.get("state") != expected_state:
                return None
            found.append((expected_state, ReservationIdentity.from_record(reservation)))
        return found[0] if len(found) == 1 and found[0][0] in allowed_states else None

    @staticmethod
    def _has_recovery_proof(
        evidence: Mapping[str, Any] | None,
        entry: Mapping[str, Any],
        stable_signature: tuple[str | None, ...],
    ) -> bool:
        if (
            not isinstance(evidence, Mapping)
            or evidence.get("outcome") not in {"recovered", "already_recovered"}
            or evidence.get("reason") is not None
            or not isinstance(evidence.get("binding_signature"), (list, tuple))
            or tuple(evidence["binding_signature"]) != stable_signature
            or evidence.get("authority_granted") is not False
            or evidence.get("local_effects") not in ([], ())
            or not has_exact_terminal_revisions(evidence.get("source_revisions"))
            or not has_exact_terminal_revisions(evidence.get("committed_revisions"))
            or not authority_evidence_matches_intent(evidence, entry)
        ):
            return False
        token = evidence.get("recovered_fencing_token")
        return type(token) is int and token > entry["parameters"]["fencing_token"]

    def _apply_local_effects(
        self,
        binding: ProjectBinding,
        entry: Mapping[str, Any],
        evidence: Mapping[str, Any],
    ) -> bool:
        parameters = entry["parameters"]
        attempt_id = parameters["attempt_id"]
        recovered_token = evidence["recovered_fencing_token"]
        paths = machine_project_paths(self._runtime.root, binding.project_id)
        cfg = local_binding_config(self._runtime, binding)
        try:
            with evidence_write_guard(paths["root"], attempt_id) as evidence_acquired:
                if not evidence_acquired:
                    return False
                with attempt_control_lock(cfg, attempt_id, blocking=False) as lock_acquired:
                    if not lock_acquired:
                        return False
                    current = read_process_manifest(
                        self._runtime,
                        binding,
                        entry["path"],
                        entry["binding_signature"],
                    )
                    if current is None or current["intent_signature"] != entry["intent_signature"]:
                        return False
                    reservation = self._read_reservation(
                        parameters["reservation_id"],
                        allowed_states=frozenset({"active", "released"}),
                    )
                    if reservation is None:
                        return False
                    reservation_state, identity = reservation
                    if (
                        identity.project_id != binding.project_id
                        or identity.task_id != parameters["task_id"]
                        or identity.attempt_id != attempt_id
                        or identity.fencing_token not in {parameters["fencing_token"], recovered_token}
                    ):
                        return False
                    if reservation_state == "active" and identity.fencing_token == parameters["fencing_token"]:
                        retag = retag_cpu_if_matches if identity.cpu_slots is not None else retag_if_matches
                        if not retag(self._runtime.root, identity, attempt_id, recovered_token):
                            return False
                    process = dict(current["process"])
                    process.update(
                        fencing_token=recovered_token,
                        recovered_at=utc_now(),
                        observed_state="running",
                        supervisor="agent",
                        authority_state="healthy",
                        lease_expires_at=evidence["lease_expires_at"],
                    )
                    atomic_replace(entry["path"], {"process": process})
                    return True
        except _SOURCE_ERRORS:
            return False

    def advance(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
    ) -> dict[str, Mapping[str, Any]]:
        """Recover bounded live orphan manifests through one shared-only CAS."""
        if type(registry_revision) is not int or registry_revision < 0:
            raise ValueError("registry_revision must be a nonnegative integer.")
        eligible = list(bindings)
        current_signatures = {binding_signature(binding, registry_revision) for binding in eligible}
        for signature in tuple(self._scans):
            if signature not in current_signatures or (
                signature in self._scan_complete and not self.has_pending(signature)
            ):
                self._scans.pop(signature).close()
                self._scan_complete.discard(signature)
        binding_by_project = {binding.project_id: binding for binding in eligible}
        for project_id, entry in tuple(self._entries.items()):
            binding = binding_by_project.get(project_id)
            if binding is None or entry.get("binding_signature") != binding_signature(binding, registry_revision):
                self._entries.pop(project_id, None)
        if eligible:
            start = self._binding_offset % len(eligible)
            rotated = eligible[start:] + eligible[:start]
            selected = rotated[: min(_MAX_BINDINGS_PER_TURN, len(rotated))]
            self._binding_offset = (start + len(selected)) % len(eligible)
        else:
            selected = []
            self._binding_offset = 0

        attempts: dict[str, Mapping[str, Any]] = {}
        entries: dict[str, Mapping[str, Any]] = {}
        request_bindings: list[ProjectBinding] = []
        for binding in selected:
            signature = binding_signature(binding, registry_revision)
            paths = machine_project_paths(self._runtime.root, binding.project_id)
            entry = self._entries.get(binding.project_id)
            if entry is not None:
                try:
                    current = read_process_manifest(
                        self._runtime,
                        binding,
                        entry["path"],
                        signature,
                    )
                except _SOURCE_ERRORS:
                    current = None
                if current is None or current["intent_signature"] != entry["intent_signature"]:
                    self._entries.pop(binding.project_id, None)
                    entry = None
            scan = self._scans.get(signature)
            if scan is None:
                if len(self._scans) >= _MAX_SCANS:
                    diagnostic_increment("scheduler.isolated.orphan_scan_capacity_exhausted")
                    continue
                scan = EvidenceScan(paths["processes"])
                self._scans[signature] = scan
                self._scan_complete.discard(signature)
            try:
                if entry is None:
                    page = scan.take(1)
                    if page.is_complete:
                        self._scan_complete.add(signature)
                    if not page.paths:
                        continue
                    entry = read_process_manifest(
                        self._runtime,
                        binding,
                        page.paths[0],
                        signature,
                    )
                    if entry is None or entry["process"].get("authority_state") in {"healthy", "local_safe"}:
                        continue
                    if len(self._entries) >= _MAX_ENTRIES:
                        continue
                reservation_id = entry["parameters"].get("reservation_id")
                if not isinstance(reservation_id, str):
                    continue
                reservation_record = self._read_reservation(
                    reservation_id,
                    allowed_states=frozenset({"active", "released"}),
                )
                if (
                    reservation_record is None
                    or reservation_record[1].project_id != binding.project_id
                    or reservation_record[1].task_id != entry["parameters"]["task_id"]
                    or reservation_record[1].attempt_id != entry["parameters"]["attempt_id"]
                    or type(reservation_record[1].fencing_token) is not int
                    or reservation_record[1].fencing_token < entry["parameters"]["fencing_token"]
                ):
                    self._entries.pop(binding.project_id, None)
                    continue
                if binding.project_id not in self._entries:
                    self._entries[binding.project_id] = entry
            except _SOURCE_ERRORS:
                continue
            entries[binding.project_id] = entry
            attempts[binding.project_id] = {
                **entry["parameters"],
                "binding_signature": self._stable_binding_signature(binding),
                "source_revisions": {"task": None, "attempt_digest": None},
                # A process may exit after shared recovery commits but before
                # local application. Replay that exact marker, never create
                # fresh live authority from a dead manifest.
                "replay_only": reservation_record[0] == "released"
                or read_isolated_authority_manifest(self._runtime, binding, entry["path"], signature) is None,
            }
            request_bindings.append(binding)
        if not attempts:
            return {}
        try:
            results = self._controller.advance_authority_orphan_recoveries(
                request_bindings,
                registry_revision,
                attempts,
            )
        except _SOURCE_ERRORS:
            return {}
        for binding in request_bindings:
            entry = entries[binding.project_id]
            evidence = results.get(binding.project_id)
            stable_signature = self._stable_binding_signature(binding)
            if self._has_recovery_proof(evidence, entry, stable_signature):
                if self._apply_local_effects(binding, entry, evidence):
                    self._entries.pop(binding.project_id, None)
            elif isinstance(evidence, Mapping) and evidence.get("outcome") == "stale":
                self._entries.pop(binding.project_id, None)
        for signature, scan in tuple(self._scans.items()):
            if signature in self._scan_complete and not self.has_pending(signature):
                self._scans.pop(signature, scan).close()
                self._scan_complete.discard(signature)
        return results
