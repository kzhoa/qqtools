"""Discover local running Attempts and publish their exact process identity."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ....runtime.authority_scan import EvidenceScan, is_path_present
from ....runtime.paths import machine_project_paths
from ....runtime.process_evidence import inspect_wrapper_identity
from ....runtime.records import validate_identifier
from ....runtime.responsibility_cleanup import evidence_write_guard
from ....runtime.store import atomic_replace
from ....runtime.work_budget import diagnostic_increment
from ...context import MachineRuntime, ProjectBinding
from ...project_io_process import binding_signature
from ..controller import ProjectIOController
from ..protocol import ProjectIORequest
from .evidence import (
    SOURCE_ERRORS,
    ProcessRecord,
    RunnerSource,
    read_launch_intent,
    read_process_record,
    read_registration,
)

_MAX_SCAN_STATES = 256
_MAX_PENDING_RUNNING_PUBLICATIONS = 256
_MAX_RUNNING_PUBLICATION_BINDINGS = 64


@dataclass(slots=True)
class _RunningPublicationScans:
    """Persistent discovery state for one binding signature."""

    registrations: EvidenceScan
    launch_intents: EvidenceScan
    next_lane: str = "registration"
    registrations_complete: bool = False
    launch_intents_complete: bool = False

    def close(self) -> None:
        self.registrations.close()
        self.launch_intents.close()

    @property
    def complete(self) -> bool:
        """Whether both source lanes have reached EOF for this sweep."""
        return self.registrations_complete and self.launch_intents_complete


def _registration_matches_intent(
    registration: RunnerSource,
    intent: RunnerSource,
) -> bool:
    """Match only identity fields present before the child process exists."""
    return all(
        registration.record.get(field) == intent.record.get(field)
        for field in (
            "task_id",
            "attempt_id",
            "fencing_token",
            "wrapper_pid",
            "wrapper_start_time_ticks",
        )
    )


def _ensure_running_manifest(
    paths: Mapping[str, Path],
    source: RunnerSource,
) -> dict[str, Any]:
    """Materialize a registration once and return its typed publication fields."""
    attempt_id = source.parameters["attempt_id"]
    manifest = paths["processes"] / f"{attempt_id}.json"
    if not is_path_present(manifest):
        process = dict(source.record)
        process.update(
            {
                "observed_state": "running",
                "supervisor": "agent",
                "authority_state": "healthy",
                "created_by": "agent",
            }
        )
        atomic_replace(manifest, {"process": process})
    return dict(source.parameters)


def _terminal_matches_running_request(
    request: ProjectIORequest,
    binding: ProjectBinding,
    terminal: ProcessRecord,
) -> bool:
    """Correlate an exited manifest with only its exact running publication."""
    return (
        request.operation_kind == "authority_running_publish"
        and request.project_id == binding.project_id
        and dict(request.parameters)
        == {
            "machine_name": binding.machine_name,
            **terminal.parameters,
            "process_created_at": terminal.record.get("process_created_at"),
        }
    )


class RunningPublications:
    """Own binding rotation, open evidence scans and pending publication intents."""

    def __init__(self, runtime: MachineRuntime, controller: ProjectIOController) -> None:
        self.runtime = runtime
        self.controller = controller
        self._scans: dict[tuple[str, ...], _RunningPublicationScans] = {}
        self._intents: dict[tuple[str, ...], Mapping[str, Any]] = {}
        self._binding_offset = 0

    def close(self) -> None:
        for scans in self._scans.values():
            scans.close()
        self._scans.clear()
        self._intents.clear()
        self._binding_offset = 0

    def advance(
        self,
        bindings: Sequence[ProjectBinding],
        registry_revision: int,
        *,
        pending_attempt_ids: Collection[str] = (),
    ) -> dict[str, Mapping[str, Any]]:
        """Advance one bounded local evidence opportunity per selected binding."""
        if type(registry_revision) is not int or registry_revision < 0:
            raise ValueError("registry_revision must be a nonnegative integer.")
        if isinstance(pending_attempt_ids, (str, bytes)) or not isinstance(pending_attempt_ids, Collection):
            raise ValueError("pending_attempt_ids must be a finite collection of identifiers.")
        if len(pending_attempt_ids) > _MAX_PENDING_RUNNING_PUBLICATIONS:
            raise ValueError("pending_attempt_ids cannot contain more than 256 identifiers.")
        pending: set[str] = set()
        for attempt_id in pending_attempt_ids:
            pending.add(validate_identifier(attempt_id, "pending attempt_id"))

        eligible = list(bindings)
        signatures = [binding_signature(binding, registry_revision) for binding in eligible]
        current_signatures = set(signatures)
        for signature in tuple(self._scans):
            scans = self._scans[signature]
            if signature not in current_signatures or scans.complete:
                self._scans.pop(signature).close()
        self._intents = {
            signature: parameters for signature, parameters in self._intents.items() if signature in current_signatures
        }

        if eligible:
            start = self._binding_offset % len(eligible)
            rotated = eligible[start:] + eligible[:start]
            selected_count = min(_MAX_RUNNING_PUBLICATION_BINDINGS, len(rotated))
            selected = rotated[:selected_count]
            self._binding_offset = (start + selected_count) % len(eligible)
        else:
            selected = []
            self._binding_offset = 0

        publications: dict[str, Mapping[str, Any]] = {}
        publication_bindings: list[ProjectBinding] = []
        reconciliation_bindings: list[ProjectBinding] = []
        unresolved_running_requests = tuple(
            request
            for request in self.controller.executor.unresolved_requests()
            if request.operation_kind == "authority_running_publish"
        )
        for binding in selected:
            signature = binding_signature(binding, registry_revision)
            paths = machine_project_paths(self.runtime.root, binding.project_id)
            scans = self._scans.get(signature)
            retained = self._intents.get(signature)
            if retained is not None:
                try:
                    source = read_registration(
                        paths["registrations"] / f"{retained['attempt_id']}.json",
                        paths["root"],
                        binding,
                    )
                    process = read_process_record(
                        paths["processes"] / f"{retained['attempt_id']}.json", paths["root"], binding
                    )
                except SOURCE_ERRORS:
                    continue
                if (
                    source is None
                    or source.parameters != retained
                    or process is None
                    or any(
                        process.parameters[field] != retained[field]
                        for field in (
                            "task_id",
                            "attempt_id",
                            "attempt_number",
                            "fencing_token",
                            "reservation_id",
                            "process_identity",
                        )
                    )
                    or process.record.get("observed_state") == "exited"
                ):
                    self._intents.pop(signature, None)
                else:
                    if retained["attempt_id"] not in pending:
                        publications[binding.project_id] = retained
                        publication_bindings.append(binding)
                    continue
            if scans is None:
                if len(self._scans) >= _MAX_SCAN_STATES:
                    diagnostic_increment("scheduler.isolated.running_scan_capacity_exhausted")
                    continue
                scans = _RunningPublicationScans(
                    registrations=EvidenceScan(paths["registrations"]),
                    launch_intents=EvidenceScan(paths["launch_intents"]),
                )
                self._scans[signature] = scans

            if scans.registrations_complete:
                lane = "launch_intent"
            elif scans.launch_intents_complete:
                lane = "registration"
            else:
                lane = scans.next_lane
                scans.next_lane = "launch_intent" if lane == "registration" else "registration"
            scan = scans.registrations if lane == "registration" else scans.launch_intents
            try:
                page = scan.take(1)
            except SOURCE_ERRORS:
                continue
            if lane == "registration":
                scans.registrations_complete = page.is_complete
            else:
                scans.launch_intents_complete = page.is_complete
            if not page.paths:
                if scans.complete:
                    self._scans.pop(signature, scans).close()
                continue

            source_path = page.paths[0]
            try:
                source = (
                    read_registration(
                        source_path,
                        paths["root"],
                        binding,
                    )
                    if lane == "registration"
                    else read_launch_intent(source_path, paths["root"])
                )
            except SOURCE_ERRORS:
                continue
            if source is None or source.record["attempt_id"] in pending:
                continue

            try:
                if lane == "registration":
                    with evidence_write_guard(paths["root"], source.record["attempt_id"]) as acquired:
                        if not acquired:
                            continue
                        current_source = read_registration(
                            source_path,
                            paths["root"],
                            binding,
                        )
                        if current_source is None or current_source.identity != source.identity:
                            continue
                        terminal_process = read_process_record(
                            paths["processes"] / f"{source.record['attempt_id']}.json",
                            paths["root"],
                            binding,
                        )
                        if terminal_process is not None and terminal_process.record.get("observed_state") == "exited":
                            if any(
                                _terminal_matches_running_request(request, binding, terminal_process)
                                for request in unresolved_running_requests
                            ):
                                reconciliation_bindings.append(binding)
                            continue
                        parameters = _ensure_running_manifest(paths, current_source)
                else:
                    parameters, terminal = self._advance_launch_intent(
                        binding,
                        paths,
                        source_path,
                        source,
                        pending,
                    )
                    if terminal is not None and any(
                        _terminal_matches_running_request(request, binding, terminal)
                        for request in unresolved_running_requests
                    ):
                        reconciliation_bindings.append(binding)
            except SOURCE_ERRORS:
                continue

            if parameters is not None:
                if len(self._intents) >= _MAX_RUNNING_PUBLICATION_BINDINGS:
                    continue
                self._intents[signature] = parameters
                publications[binding.project_id] = parameters
                publication_bindings.append(binding)
            if scans.complete:
                self._scans.pop(signature, scans).close()

        if not publications and not reconciliation_bindings:
            return {}
        publication_project_ids = {binding.project_id for binding in publication_bindings}
        reconciliation_bindings = [
            binding for binding in reconciliation_bindings if binding.project_id not in publication_project_ids
        ]
        results = self.controller.advance_authority_running_publications(
            [*publication_bindings, *reconciliation_bindings],
            registry_revision,
            publications,
        )
        for binding in publication_bindings:
            if results.get(binding.project_id, {}).get("outcome") == "processed":
                self._intents.pop(binding_signature(binding, registry_revision), None)
        return results

    def _advance_launch_intent(
        self,
        binding: ProjectBinding,
        paths: Mapping[str, Path],
        source_path: Path,
        source: RunnerSource,
        pending: set[str],
    ) -> tuple[dict[str, Any] | None, ProcessRecord | None]:
        attempt_id = source.record["attempt_id"]
        if attempt_id in pending:
            return None, None
        with evidence_write_guard(paths["root"], attempt_id) as acquired:
            if not acquired:
                return None, None
            current_intent = read_launch_intent(source_path, paths["root"])
            if current_intent is None or current_intent.identity != source.identity:
                return None, None

            manifest = paths["processes"] / f"{attempt_id}.json"
            terminal_process = read_process_record(manifest, paths["root"], binding)
            if terminal_process is not None and terminal_process.record.get("observed_state") == "exited":
                return None, terminal_process

            registration_path = paths["registrations"] / f"{attempt_id}.json"
            if is_path_present(registration_path):
                registration = read_registration(
                    registration_path,
                    paths["root"],
                    binding,
                )
                if registration is None or not _registration_matches_intent(registration, current_intent):
                    return None, None
                return _ensure_running_manifest(paths, registration), None

            if is_path_present(manifest):
                return None, None
            try:
                wrapper_state = inspect_wrapper_identity(current_intent.record).state
            except SOURCE_ERRORS:
                wrapper_state = "unknown"
            if wrapper_state == "alive":
                return None, None

            process = dict(current_intent.record)
            process.update(
                {
                    "observed_state": "launch_unverifiable",
                    "supervisor": "agent",
                    "authority_state": "isolated",
                    "created_by": "agent",
                }
            )
            atomic_replace(manifest, {"process": process})
            return None, None
