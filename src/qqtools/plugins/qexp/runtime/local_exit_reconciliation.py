"""Reconcile finished local processes without consulting shared project state."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .authority_scan import EvidenceScan
from .paths import local_paths
from .process_evidence import inspect_group_identity
from .records import utc_now
from .resources.reservations import (
    ReservationIdentity,
    release_if_matches,
    release_termination_if_matches,
    reservation_snapshot,
)
from .store import atomic_replace, iter_json, read_json


class LocalExitReconciler:
    """Release local reservations only when retained evidence proves a process exited."""

    def __init__(
        self,
        runtime_root: Path,
        *,
        reservation_runtime_root: Path | None = None,
        project_id: str | None = None,
    ) -> None:
        self.runtime_root = runtime_root
        self.reservation_runtime_root = reservation_runtime_root or runtime_root
        self.project_id = project_id if project_id is not None else runtime_root.name
        self._allows_projectless_reservations = self.reservation_runtime_root.resolve() == runtime_root.resolve()
        capacity_paths = local_paths(self.reservation_runtime_root)
        self._capacity_scans = (
            EvidenceScan(capacity_paths["active"]),
            EvidenceScan(capacity_paths["cpu_active"]),
        )
        self._capacity_turn = 0

    def close(self) -> None:
        """Release advisory reservation scan cursors."""
        for scan in self._capacity_scans:
            scan.close()

    def reconcile(self, limit: int | None = None, *, bounded: bool = False) -> None:
        """Release verified finished local occupancy without reading shared task truth."""
        paths = local_paths(self.runtime_root)
        if limit is None:
            for observation in iter_json(paths["observations"]):
                self.reconcile_observation(observation, bounded=bounded)
            return
        if type(limit) is not int or limit <= 0:
            raise ValueError("local capacity discovery limit must be a positive integer")
        for identity in self.capacity_page("local-capacity", limit=limit):
            if not self.owns_reservation(identity):
                continue
            attempt_id = identity.attempt_id
            if not attempt_id or attempt_id in {".", ".."} or "/" in attempt_id or "\\" in attempt_id:
                continue
            self.reconcile_observation(
                paths["observations"] / f"{attempt_id}.json",
                reservation_identity=identity,
                bounded=bounded,
            )

    def reconcile_observation(
        self,
        observation: Path,
        *,
        reservation_identity: ReservationIdentity | None = None,
        bounded: bool = False,
    ) -> None:
        """Reconcile one exit observation against its local registration evidence."""
        paths = local_paths(self.runtime_root)
        attempt_id = observation.stem
        try:
            manifest = paths["processes"] / observation.name
            if manifest.exists():
                process = read_json(manifest)["process"]
            else:
                process = read_json(paths["registrations"] / observation.name)["process_registration"]
            if not isinstance(process, dict):
                raise ValueError("local process evidence must be an object")
            if process.get("attempt_id") == attempt_id:
                self.release_finished(
                    process,
                    reservation_identity=reservation_identity,
                    bounded=bounded,
                )
        except (OSError, KeyError, TypeError, ValueError):
            self.record_diagnostic({"attempt_id": attempt_id}, "local_capacity_reconciliation_unavailable")

    def release_finished(
        self,
        process: dict[str, object],
        *,
        reservation_identity: ReservationIdentity | None = None,
        bounded: bool = False,
    ) -> None:
        """Retain recovery evidence while releasing an identity-verified absent process."""
        task_id, attempt_id = process.get("task_id"), process.get("attempt_id")
        if not isinstance(task_id, str) or not isinstance(attempt_id, str):
            return
        paths = local_paths(self.runtime_root)
        try:
            registration = read_json(paths["registrations"] / f"{attempt_id}.json")["process_registration"]
            if not isinstance(registration, dict):
                raise ValueError("local process registration must be an object")
            for key in ("task_id", "attempt_id", "process_group_id", "process_group_start_time_ticks"):
                if registration.get(key) is None or registration.get(key) != process.get(key):
                    return
            group = registration["process_group_id"]
            if not isinstance(group, int) or group <= 0:
                return
            if inspect_group_identity(registration, process).state != "absent":
                return
            is_valid, _code = self.read_exit_observation(
                paths["observations"] / f"{attempt_id}.json", task_id, attempt_id, process
            )
            if not is_valid:
                return
            if reservation_identity is None and bounded:
                self.reconcile(limit=8)
                return
            identities = (
                (reservation_identity,)
                if reservation_identity is not None
                else (
                    ReservationIdentity.from_record(record)
                    for record in reservation_snapshot(self.reservation_runtime_root).reservations
                )
            )
            for identity in identities:
                if not self.owns_reservation(identity):
                    continue
                if (
                    identity.task_id == task_id
                    and identity.attempt_id == attempt_id
                    and identity.fencing_token == process.get("fencing_token")
                ):
                    release_if_matches(self.reservation_runtime_root, identity, "local_process_exited")
        except (OSError, KeyError, TypeError, ValueError):
            self.record_diagnostic(process, "local_capacity_reconciliation_unavailable")

    def release_confirmed_termination(
        self,
        process: dict[str, object],
        decision: dict[str, object],
    ) -> bool:
        """Release one exact owned reservation after a signaled process is confirmed absent."""
        task_id = process.get("task_id")
        attempt_id = process.get("attempt_id")
        reservation_id = process.get("reservation_id")
        if (
            not isinstance(task_id, str)
            or not isinstance(attempt_id, str)
            or not isinstance(reservation_id, str)
            or decision.get("task_id") != task_id
            or decision.get("attempt_id") != attempt_id
            or decision.get("decision_token") != process.get("fencing_token")
            or decision.get("process_group_id") != process.get("process_group_id")
            or decision.get("process_group_start_time_ticks") != process.get("process_group_start_time_ticks")
            or decision.get("state") != "confirmed"
            or decision.get("shared_commitment") != "committed"
            or not isinstance(decision.get("signal_attempts"), list)
            or decision.get("confirmation") not in {"identity_absent", "process_absent"}
        ):
            return False
        paths = local_paths(self.runtime_root)
        try:
            manifest_path = paths["processes"] / f"{attempt_id}.json"
            manifest_envelope = read_json(manifest_path)
            manifest = manifest_envelope.get("process") if isinstance(manifest_envelope, dict) else None
            if not isinstance(manifest, dict) or manifest != process:
                return False
            registration_envelope = read_json(paths["registrations"] / f"{attempt_id}.json")
            registration = (
                registration_envelope.get("process_registration") if isinstance(registration_envelope, dict) else None
            )
            if not isinstance(registration, dict):
                return False
            for field in (
                "task_id",
                "attempt_id",
                "fencing_token",
                "process_group_id",
                "process_group_start_time_ticks",
            ):
                if registration.get(field) != process.get(field):
                    return False
            group = registration.get("process_group_id")
            if not isinstance(group, int) or group <= 0:
                return False
            if inspect_group_identity(registration, process).state != "absent":
                return False
            capacity_paths = local_paths(self.reservation_runtime_root)
            matches: list[tuple[str, ReservationIdentity]] = []
            for state, name in (
                ("active", "active"),
                ("provisional", "provisional"),
                ("released", "released"),
                ("active", "cpu_active"),
                ("provisional", "cpu_provisional"),
                ("released", "cpu_released"),
            ):
                path = capacity_paths[name] / f"{reservation_id}.json"
                try:
                    envelope = read_json(path)
                except FileNotFoundError:
                    continue
                record = envelope.get("reservation") if isinstance(envelope, dict) else None
                if not isinstance(record, dict) or record.get("state") != state:
                    return False
                matches.append((state, ReservationIdentity.from_record(record)))
            if len(matches) == 1:
                state, identity = matches[0]
            elif (
                len(matches) == 2
                and {state for state, _identity in matches} in ({"active", "released"}, {"provisional", "released"})
                and matches[0][1] == matches[1][1]
            ):
                # Exact duplicate locations are the recoverable boundary after
                # released truth was replaced but before source truth was unlinked.
                state = next(state for state, _identity in matches if state != "released")
                identity = matches[0][1]
            else:
                return False
            if (
                not self.owns_reservation(identity)
                or identity.reservation_id != reservation_id
                or identity.task_id != task_id
                or identity.attempt_id != attempt_id
                or identity.fencing_token != process.get("fencing_token")
            ):
                return False
            if state not in {"active", "provisional", "released"}:
                return False
            return release_termination_if_matches(
                self.reservation_runtime_root,
                identity,
                "local_termination_confirmed",
            )
        except (OSError, KeyError, TypeError, ValueError):
            self.record_diagnostic(process, "local_capacity_reconciliation_unavailable")
        return False

    def owns_reservation(self, identity: ReservationIdentity) -> bool:
        """Return whether this local runtime owns a reservation identity."""
        return identity.project_id == self.project_id or (
            identity.project_id is None and self._allows_projectless_reservations
        )

    def capacity_page(self, attempt_id: str, *, limit: int = 8) -> Iterable[ReservationIdentity]:
        """Yield a rotating bounded page from local GPU and CPU reservation lanes."""
        turn = self._capacity_turn
        self._capacity_turn = (turn + 1) % len(self._capacity_scans)
        quota, remainder = divmod(limit, len(self._capacity_scans))
        for offset in range(len(self._capacity_scans)):
            budget = quota + (offset < remainder)
            if not budget:
                continue
            scan = self._capacity_scans[(turn + offset) % len(self._capacity_scans)]
            try:
                page = scan.take(budget)
            except OSError:
                self.record_diagnostic({"attempt_id": attempt_id}, "local_reservation_unreadable")
                continue
            for path in page.paths:
                try:
                    reservation = read_json(path)["reservation"]
                    if not isinstance(reservation, dict):
                        raise ValueError("local reservation must be an object")
                    yield ReservationIdentity.from_record(reservation)
                except (OSError, KeyError, TypeError, ValueError):
                    self.record_diagnostic({"attempt_id": attempt_id}, "local_reservation_unreadable")

    def read_exit_observation(
        self, path: Path, task_id: str, attempt_id: str, process: dict[str, object]
    ) -> tuple[bool, int | None]:
        """Validate immutable exit identity before permitting local reservation release."""
        try:
            record = read_json(path)
            observation = record.get("exit_observation") if isinstance(record, dict) else None
        except (OSError, ValueError, TypeError):
            self.record_diagnostic(process, "exit_observation_unreadable")
            return False, None
        if not isinstance(observation, dict):
            self.record_diagnostic(process, "exit_observation_unreadable")
            return False, None
        if observation.get("protocol_version", 1) != 1:
            self.record_diagnostic(process, "exit_observation_protocol_unsupported")
            return False, None
        if observation.get("attempt_id") != attempt_id or observation.get("task_id") not in {
            None,
            task_id,
        }:
            self.record_diagnostic(process, "exit_observation_identity_mismatch")
            return False, None
        code = observation.get("observed_exit_code")
        if type(code) is not int:
            self.record_diagnostic(process, "exit_observation_code_invalid")
            return False, None
        return True, code

    def record_diagnostic(self, process: dict[str, object], reason: str, error: Exception | None = None) -> None:
        """Persist a local diagnostic while keeping failures non-fatal."""
        attempt_id = process.get("attempt_id")
        if not isinstance(attempt_id, str):
            return
        value: dict[str, object] = {"attempt_id": attempt_id, "reason": reason, "at": utc_now()}
        if error is not None:
            value["error_type"] = type(error).__name__
            value["error"] = str(error)
        try:
            atomic_replace(
                local_paths(self.runtime_root)["authority_diagnostics"] / f"{attempt_id}.json",
                {"authority_diagnostic": value},
            )
        except OSError:
            pass
