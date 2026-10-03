"""Bounded recovery of reservation locators omitted by released runners."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..runtime.authority_scan import EvidenceScan, validate_evidence_path
from ..runtime.paths import local_paths, machine_project_paths
from ..runtime.records import validate_identifier
from ..runtime.resources.reservations import ReservationIdentity
from ..runtime.responsibility_cleanup import evidence_write_guard
from ..runtime.store import atomic_replace, read_json_limited
from ..runtime.work_budget import diagnostic_increment
from .bindings import ProjectBinding
from .project_io_process import PROCESS_IDENTITY_FIELDS

_ERRORS = (OSError, RuntimeError, ValueError, TypeError, KeyError)
_REGISTRATION_IDENTITY = (
    "protocol_version",
    "task_id",
    "attempt_id",
    "fencing_token",
    "machine_name",
    "process_created_at",
    *PROCESS_IDENTITY_FIELDS,
)
_RESOURCE_FIELDS = ("gpu_ids", "cpu_slots")
_IGNORED_PATH_LIMIT = 1_024


def _same_registration(process: Mapping[str, Any], record: Mapping[str, Any]) -> bool:
    return all(field in record and process.get(field) == record[field] for field in _REGISTRATION_IDENTITY) and all(
        (field in process) == (field in record) and process.get(field) == record.get(field)
        for field in _RESOURCE_FIELDS
    )


def registration_reservation(record: Mapping[str, Any], project_root: Path) -> str | None:
    """Resolve only an omitted registration locator from the matching local manifest."""
    if "reservation_id" in record:
        return record["reservation_id"]
    attempt_id = validate_identifier(record.get("attempt_id"), "process attempt_id")
    path = local_paths(project_root)["processes"] / f"{attempt_id}.json"
    if not validate_evidence_path(path, project_root):
        return None
    process = read_json_limited(path, max_bytes=65_536, record_type="process").get("process")
    if not isinstance(process, dict) or not _same_registration(process, record):
        return None
    return process.get("reservation_id")


class ProcessReservationRecovery:
    """Recover local locators; shared transactions retain all authority checks.

    Released protocol-1 registrations omit reservation_id. The machine reservation
    ledger supplies a hint for that exact Project/Attempt/fence, never permission
    to renew, publish terminal truth, signal, or release capacity.
    """

    def __init__(self, runtime_root: Path) -> None:
        self.root = runtime_root
        paths = local_paths(runtime_root)
        self.scans = tuple(
            (EvidenceScan(paths[name]), state)
            for name, state in (
                ("active", "active"),
                ("cpu_active", "active"),
                ("released", "released"),
                ("cpu_released", "released"),
            )
        )
        self._ignored_paths: OrderedDict[Path, None] = OrderedDict()

    def close(self) -> None:
        for scan, _state in self.scans:
            scan.close()

    def advance(self, bindings: Sequence[ProjectBinding]) -> dict[str, Any]:
        by_project = {binding.project_id: binding for binding in bindings}
        for scan, state in self.scans:
            try:
                page = scan.take(8)
            except _ERRORS:
                diagnostic_increment("scheduler.isolated.reservation_recovery_scan_failed")
                continue
            for path in page.paths:
                if path in self._ignored_paths:
                    self._ignored_paths.move_to_end(path)
                    continue
                try:
                    ignored = self._recover(path, state, by_project)
                except _ERRORS:
                    diagnostic_increment("scheduler.isolated.reservation_recovery_failed")
                    continue
                if ignored:
                    self._ignore_path(path)
        return {}

    def _ignore_path(self, path: Path) -> None:
        self._ignored_paths[path] = None
        self._ignored_paths.move_to_end(path)
        while len(self._ignored_paths) > _IGNORED_PATH_LIMIT:
            self._ignored_paths.popitem(last=False)

    def _recover(self, path: Path, state: str, bindings: Mapping[str, ProjectBinding]) -> bool:
        if not validate_evidence_path(path, self.root):
            return False
        record = read_json_limited(path, max_bytes=65_536, record_type="reservation")["reservation"]
        identity = ReservationIdentity.from_record(record)
        binding = bindings.get(identity.project_id)
        if (
            binding is None
            or record.get("state") != state
            or path.stem != identity.reservation_id
            or record.get("machine_name") != binding.machine_name
            or identity.shared_root != str(binding.shared_root)
            or identity.attempt_id is None
            or type(identity.fencing_token) is not int
            or identity.fencing_token < 1
        ):
            return False
        attempt_id = validate_identifier(identity.attempt_id, "reservation attempt_id")
        paths = machine_project_paths(self.root, binding.project_id)
        with evidence_write_guard(paths["root"], attempt_id) as acquired:
            if not acquired:
                return False
            manifest = paths["processes"] / f"{attempt_id}.json"
            registration = paths["registrations"] / f"{attempt_id}.json"
            if not validate_evidence_path(registration, paths["root"]):
                return False
            source = read_json_limited(registration, max_bytes=65_536, record_type="registration").get(
                "process_registration"
            )
            if not isinstance(source, dict):
                return False
            # Protocol-1 registrations are immutable. A registration that
            # already carries its locator can never need this compatibility
            # lane, so avoid rereading it on every agent cycle.
            if "reservation_id" in source:
                return True
            if (
                type(source.get("protocol_version")) is not int
                or source["protocol_version"] != 1
                or source.get("task_id") != identity.task_id
                or source.get("attempt_id") != attempt_id
                or type(source.get("fencing_token")) is not int
                or source["fencing_token"] != identity.fencing_token
                or source.get("machine_name") != binding.machine_name
                or any(field not in source for field in _REGISTRATION_IDENTITY)
                or source.get("gpu_ids", []) != list(identity.gpu_ids)
                or ("cpu_slots" in source and source["cpu_slots"] != identity.cpu_slots)
            ):
                return False
            if validate_evidence_path(manifest, paths["root"]):
                process = read_json_limited(manifest, max_bytes=65_536, record_type="process").get("process")
                if (
                    not isinstance(process, dict)
                    or "reservation_id" in process
                    or not _same_registration(process, source)
                ):
                    return False
            else:
                process = dict(source)
                process.update(
                    observed_state="running", supervisor="agent", authority_state="isolated", created_by="agent"
                )
            current = read_json_limited(path, max_bytes=65_536, record_type="reservation")["reservation"]
            if not identity.exactly_matches(current) or current.get("state") != state:
                return False
            process["reservation_id"] = identity.reservation_id
            atomic_replace(manifest, {"process": process})
            return False
