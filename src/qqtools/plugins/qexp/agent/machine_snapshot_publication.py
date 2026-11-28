"""Machine snapshot inputs, publication decisions, and durable local receipts."""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_right
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ..runtime.records import validate_identifier
from ..runtime.store import atomic_replace
from .project_io.json_data import copy_json_data
from .snapshot_publication_health import SNAPSHOT_COVERAGE_FILENAME, SnapshotPublicationHealth

if TYPE_CHECKING:
    from .bindings import ProjectBinding
    from .context import MachineRuntime
    from .project_io.identity import ValidationIdentity
    from .project_io.protocol import ProjectIORequest
    from .scheduler_diagnostics import ProducerCoverageCheckpoint


def _snapshot_gpu_ids(value: Sequence[int], label: str) -> list[int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes)
        or len(value) > 4096
        or any(type(gpu_id) is not int or gpu_id < 0 for gpu_id in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{label} must be a bounded sequence of unique nonnegative integers.")
    return sorted(value)


def _snapshot_reservations(reservations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(reservations, Sequence) or isinstance(reservations, str | bytes) or len(reservations) > 4096:
        raise ValueError("reservations must be a bounded sequence.")
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in reservations:
        if not isinstance(item, Mapping):
            raise ValueError("reservations must contain reservation objects.")
        reservation_id = item.get("reservation_id")
        try:
            validate_identifier(reservation_id, "reservation_id")
        except ValueError as exc:
            raise ValueError("reservation identity is invalid.") from exc
        if reservation_id in seen_ids:
            raise ValueError("reservations must have unique reservation IDs.")
        seen_ids.add(reservation_id)
        project_id = item.get("project_id")
        if project_id is not None:
            try:
                validate_identifier(project_id, "project_id")
            except ValueError as exc:
                raise ValueError("reservation project identity is invalid.") from exc
        state = item.get("state")
        if state not in ("active", "provisional"):
            raise ValueError("reservation state is invalid.")
        gpu_ids_value = item.get("gpu_ids", [])
        if (
            not isinstance(gpu_ids_value, Sequence)
            or isinstance(gpu_ids_value, str | bytes)
            or len(gpu_ids_value) > 4096
            or any(type(gpu_id) is not int or gpu_id < 0 for gpu_id in gpu_ids_value)
            or len(set(gpu_ids_value)) != len(gpu_ids_value)
        ):
            raise ValueError("reservation GPU identity is invalid.")
        summary: dict[str, Any] = {
            "reservation_id": reservation_id,
            "project_id": project_id,
        }
        for field in ("group_name", "machine_name", "task_id", "attempt_id"):
            value = item.get(field)
            if value is not None:
                try:
                    validate_identifier(value, field)
                except ValueError as exc:
                    raise ValueError(f"reservation {field} is invalid.") from exc
            summary[field] = value
        summary["gpu_ids"] = list(gpu_ids_value)
        summary["state"] = state
        admission = item.get("admission")
        if admission is not None and not isinstance(admission, Mapping):
            raise ValueError("reservation admission must be an object or null.")
        summary["admission"] = copy_json_data(admission) if admission is not None else None
        normalized.append(summary)
    return normalized


def _machine_snapshot_signature(parameters: Mapping[str, Any]) -> str:
    names = (
        "instance_id",
        "pid",
        "visible_gpu_ids",
        "reserved_gpu_ids",
        "reservation_summaries",
        "heartbeat_interval_seconds",
        "started_at",
        "gpu_policy",
    )
    try:
        encoded = json.dumps(
            {name: copy_json_data(parameters[name]) for name in names},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("machine snapshot input is not bounded JSON data.") from exc
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class SnapshotInputs:
    instance_id: str
    pid: int | None
    visible_gpu_ids: list[int]
    reserved_gpu_ids: list[int]
    summaries_by_project: dict[str, list[dict[str, Any]]]
    heartbeat_interval_seconds: int | float
    started_at: str
    gpu_policy: dict[str, Any]

    @classmethod
    def from_observation(
        cls,
        instance_id: str,
        pid: int | None,
        visible_gpu_ids: Sequence[int],
        reservations: Sequence[Mapping[str, Any]],
        heartbeat_interval_seconds: int | float,
        started_at: str,
        gpu_policy: object,
    ) -> SnapshotInputs:
        visible = _snapshot_gpu_ids(visible_gpu_ids, "visible_gpu_ids")
        if pid is not None and (type(pid) is not int or pid <= 0):
            raise ValueError("pid must be a positive integer or null.")
        if (
            type(heartbeat_interval_seconds) not in {int, float}
            or (type(heartbeat_interval_seconds) is float and not math.isfinite(heartbeat_interval_seconds))
            or heartbeat_interval_seconds <= 0
            or heartbeat_interval_seconds > 86_400
        ):
            raise ValueError("heartbeat_interval_seconds is invalid.")
        if not isinstance(started_at, str):
            raise ValueError("started_at must be an ISO-8601 timestamp.")
        reservation_values = _snapshot_reservations(reservations)
        reserved = sorted({gpu_id for item in reservation_values for gpu_id in item["gpu_ids"]})
        summaries_by_project: dict[str, list[dict[str, Any]]] = {}
        for item in reservation_values:
            project_id = item["project_id"]
            if project_id is None:
                continue
            summaries_by_project.setdefault(project_id, []).append(item)
        policy_view = getattr(gpu_policy, "with_reservations", None)
        if callable(policy_view):
            gpu_policy = policy_view(reserved)
        policy_to_dict = getattr(gpu_policy, "to_dict", None)
        if callable(policy_to_dict):
            gpu_policy = policy_to_dict()
        if not isinstance(gpu_policy, Mapping):
            raise ValueError("gpu_policy must be a mapping or policy view.")
        policy = copy_json_data(gpu_policy)
        if not isinstance(policy, dict):
            raise ValueError("gpu_policy must be an object.")

        return cls(
            instance_id, pid, visible, reserved, summaries_by_project, heartbeat_interval_seconds, started_at, policy
        )

    def parameters_for(self, project_id: str) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "pid": self.pid,
            "visible_gpu_ids": self.visible_gpu_ids,
            "reserved_gpu_ids": self.reserved_gpu_ids,
            "reservation_summaries": self.summaries_by_project.get(project_id, []),
            "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
            "started_at": self.started_at,
            "gpu_policy": self.gpu_policy,
        }


class MachineSnapshotPublication:
    """Own heartbeat scheduling, publication health, and recovery coverage receipts."""

    def __init__(
        self, runtime: MachineRuntime, health: SnapshotPublicationHealth, monotonic: Callable[[], float]
    ) -> None:
        self.runtime = runtime
        self.health = health
        self._monotonic = monotonic
        self._coverage_cursor: tuple[str, str] | None = None

    def due_offers(
        self, identities: Iterable[ValidationIdentity], inputs: SnapshotInputs, *, can_start: bool
    ) -> Iterator[tuple[ValidationIdentity, dict[str, Any], float]]:
        """Keep intent age during overload and offer changed or overdue snapshots."""
        instance_id = inputs.instance_id
        interval = inputs.heartbeat_interval_seconds
        if not can_start:
            for identity in identities:
                if self.health.stage(identity, instance_id) in {None, "published"}:
                    self.health.begin_intent(identity, instance_id, interval, new_attempt=True)
            return

        now = self._monotonic()
        for identity in identities:
            parameters = inputs.parameters_for(identity.project_id)
            signature = _machine_snapshot_signature(parameters)
            last = self.health.last_publication(identity)
            if last is not None and last[0] == signature and now - last[1] < interval:
                continue
            stage = self.health.stage(identity, instance_id)
            started = self.health.begin_intent(
                identity, instance_id, interval, new_attempt=stage in {None, "published", "backoff"}
            )
            # Leave one heartbeat for publication before the three-interval stale diagnostic.
            yield identity, parameters, started + 2 * interval

    def accept_publication(
        self,
        identity: ValidationIdentity,
        request: ProjectIORequest,
        instance_id: str,
        evidence: Mapping[str, Any],
        heartbeat_interval: float,
        observed_registry_revision: int | None,
    ) -> None:
        if (
            request.parameters.get("instance_id") == instance_id
            and request.parameters.get("stop_reason") is None
            and (observed_registry_revision is None or identity.registry_revision == observed_registry_revision)
            and self.health.report_request(
                identity, instance_id, heartbeat_interval, request.request_id, stage="published"
            )
        ):
            self._record_snapshot_observation(identity, instance_id, evidence["snapshot_at"], heartbeat_interval)
        self.health.record_publication(identity, _machine_snapshot_signature(request.parameters), self._monotonic())

    def prune_machine_snapshot_observations(self, registered: Sequence[ProjectBinding]) -> None:
        """Retain publication evidence while an exact registration remains in the registry."""
        self.health.prune_observations(
            (binding.runtime_instance_id, binding.project_id, binding.registration_generation)
            for binding in registered
            if binding.runtime_instance_id is not None and binding.registration_generation is not None
        )

    def record_scheduler_coverage(self, checkpoint: ProducerCoverageCheckpoint, *, instance_id: str) -> None:
        """Persist a bounded, rotating slice of covered Project snapshot receipts."""
        bindings = sorted(checkpoint.observed_at_by_binding)
        if not bindings:
            return
        cursor = self._coverage_cursor
        start = bisect_right(bindings, cursor) if cursor is not None else 0
        window = (bindings[start:] + bindings[:start])[:16]
        self._coverage_cursor = window[-1]
        runtime_id = self.runtime.observe_instance_id()
        for project_id, generation in window:
            evidence = self.health.snapshot_evidence(
                project_id=project_id,
                registration_generation=generation,
                runtime_id=runtime_id,
                agent_instance_id=instance_id,
            )
            if evidence is None:
                continue
            snapshot_at, accepted_at = evidence
            observed_at = checkpoint.observed_at_by_binding[project_id, generation]
            try:
                if datetime.fromisoformat(observed_at) <= datetime.fromisoformat(accepted_at):
                    continue
                atomic_replace(
                    self.runtime.project_paths(project_id)["root"] / SNAPSHOT_COVERAGE_FILENAME,
                    {
                        "snapshot_coverage": {
                            "runtime_id": runtime_id,
                            "agent_instance_id": instance_id,
                            "registration_generation": generation,
                            "snapshot_at": snapshot_at,
                            "accepted_at": accepted_at,
                            "observed_at": observed_at,
                        }
                    },
                )
                self.runtime.complete_recovery_snapshot(project_id, generation, snapshot_at)
            except (OSError, ValueError):
                continue

    def _record_snapshot_observation(
        self, identity: ValidationIdentity, instance_id: str, snapshot_at: str, heartbeat_interval: float
    ) -> None:
        accepted_at = datetime.now(timezone.utc).isoformat()
        try:
            committed = atomic_replace(
                self.runtime.project_paths(identity.project_id)["root"] / "snapshot-observation.json",
                {
                    "snapshot_observation": {
                        "runtime_id": identity.runtime_id,
                        "agent_instance_id": instance_id,
                        "registration_generation": identity.registration_generation,
                        "snapshot_at": snapshot_at,
                        "accepted_at": accepted_at,
                        "heartbeat_interval_seconds": heartbeat_interval,
                    }
                },
            )
        except (OSError, ValueError):
            return
        if committed is not None:
            self.health.record_snapshot_evidence(identity, snapshot_at, accepted_at)
