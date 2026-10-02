"""Machine-global CPU-only lane policy and reservations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ..locks import exclusive
from ..paths import local_paths
from ..records import new_id, utc_now, validate_identifier
from ..store import atomic_replace, iter_json, read_json

PROVISIONAL_TTL_SECONDS = 30

if TYPE_CHECKING:
    from .reservations import ReservationIdentity


@dataclass(frozen=True, slots=True)
class CpuLanePolicy:
    capacity: int
    revision: int

    @property
    def to_dict(self) -> dict[str, int]:
        return {"capacity": self.capacity, "revision": self.revision}


def _runtime_root(value: Path | Any) -> Path:
    root = value if isinstance(value, Path) else getattr(value, "root", value)
    return Path(root)


def _is_expired(value: dict[str, Any]) -> bool:
    reservation = value["reservation"]
    if reservation.get("state") == "provisional" and "executor_owner" in reservation:
        return False
    expires_at = reservation.get("expires_at")
    return bool(expires_at) and datetime.fromisoformat(expires_at.replace("Z", "+00:00")) <= datetime.now(timezone.utc)


def _values(directory: Path) -> list[tuple[Path, dict[str, Any]]]:
    return [(path, read_json(path)) for path in iter_json(directory)]


def _release_expired(paths: dict[str, Path]) -> None:
    for path, value in _values(paths["cpu_provisional"]):
        if _is_expired(value):
            value["reservation"].update(
                {"state": "released", "released_at": utc_now(), "release_reason": "provisional_expired"}
            )
            atomic_replace(paths["cpu_released"] / path.name, value)
            path.unlink(missing_ok=True)


def _executor_owner(
    executor_epoch: str | None,
    executor_request_id: str | None,
    registration_generation: str | None,
    *,
    project_id: str | None,
    shared_root: str | None,
    task_id: str,
    attempt_id: str | None,
    fencing_token: int | None,
) -> dict[str, str] | None:
    values = (executor_epoch, executor_request_id, registration_generation)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("executor_epoch, executor_request_id, and registration_generation must be provided together.")
    normalized: list[str] = []
    for value, label in zip(values, ("executor_epoch", "executor_request_id", "registration_generation")):
        if not isinstance(value, str) or not value or len(value) > 128 or "\x00" in value:
            raise ValueError(f"{label} must be a bounded nonempty identifier.")
        try:
            normalized.append(validate_identifier(value, label))
        except ValueError as exc:
            raise ValueError(f"{label} must be a bounded nonempty identifier.") from exc
    if not project_id or not shared_root or not attempt_id or type(fencing_token) is not int or fencing_token <= 0:
        raise ValueError("executor-owned reservations require exact Project and Attempt identity.")
    for value, label in ((project_id, "project_id"), (task_id, "task_id"), (attempt_id, "attempt_id")):
        if not isinstance(value, str) or len(value) > 128 or "\x00" in value:
            raise ValueError(f"{label} must be a bounded nonempty identifier.")
        try:
            validate_identifier(value, label)
        except ValueError as exc:
            raise ValueError(f"{label} must be a bounded nonempty identifier.") from exc
    if (
        not isinstance(shared_root, str)
        or len(shared_root) > 4096
        or "\x00" in shared_root
        or not Path(shared_root).is_absolute()
        or str(Path(shared_root)) != shared_root
    ):
        raise ValueError("executor-owned reservations require an absolute bounded shared_root.")
    return {
        "executor_epoch": normalized[0],
        "request_id": normalized[1],
        "registration_generation": normalized[2],
    }


def _matches_executor_owner(reservation: dict[str, Any], identity: ReservationIdentity) -> bool:
    if not _identity_has_executor_owner(identity):
        return False
    return reservation.get("executor_owner") == {
        "executor_epoch": identity.executor_epoch,
        "request_id": identity.executor_request_id,
        "registration_generation": identity.registration_generation,
    }


def _identity_has_executor_owner(identity: ReservationIdentity) -> bool:
    return bool(identity.executor_epoch and identity.executor_request_id and identity.registration_generation)


def _policy(paths: dict[str, Path]) -> CpuLanePolicy:
    path = paths["cpu_policy"]
    if not path.exists():
        return CpuLanePolicy(0, 0)
    value = read_json(path).get("cpu_lane")
    if not isinstance(value, dict) or type(value.get("capacity")) is not int or value["capacity"] < 0:
        raise RuntimeError("CPU lane policy is malformed.")
    if type(value.get("revision")) is not int or value["revision"] < 0:
        raise RuntimeError("CPU lane policy is malformed.")
    return CpuLanePolicy(value["capacity"], value["revision"])


def get_cpu_lane_policy(runtime_root: Path | Any) -> CpuLanePolicy:
    """Return the current machine-global CPU-only lane policy."""
    paths = local_paths(_runtime_root(runtime_root))
    with exclusive(paths["locks"] / "cpu-lane.lock"):
        return _policy(paths)


def _reservations(paths: dict[str, Path]) -> list[dict[str, Any]]:
    return [value["reservation"] for _, value in _values(paths["cpu_active"])] + [
        value["reservation"] for _, value in _values(paths["cpu_provisional"]) if not _is_expired(value)
    ]


def set_cpu_lane_capacity(runtime_root: Path | Any, *, capacity: int) -> CpuLanePolicy:
    """Atomically set CPU-only capacity without invalidating live reservations.

    Args:
        runtime_root: Machine runtime root shared by all bound projects.
        capacity: Non-negative logical CPU slot budget.

    Returns:
        The persisted CPU lane policy.
    """
    if type(capacity) is not int or capacity < 0:
        raise ValueError("CPU lane capacity must be a non-negative integer.")
    paths = local_paths(_runtime_root(runtime_root))
    paths["cpu_policy"].parent.mkdir(parents=True, exist_ok=True)
    for name in ("cpu_provisional", "cpu_active", "cpu_released", "locks"):
        paths[name].mkdir(parents=True, exist_ok=True)
    with exclusive(paths["locks"] / "cpu-lane.lock"):
        _release_expired(paths)
        current = _policy(paths)
        reserved = sum(item.get("cpu_slots", 0) for item in _reservations(paths))
        if capacity < reserved:
            raise ValueError(f"CPU lane capacity {capacity} is below reserved CPU slots {reserved}.")
        if capacity == current.capacity and paths["cpu_policy"].exists():
            return current
        updated = CpuLanePolicy(capacity, current.revision + 1)
        atomic_replace(paths["cpu_policy"], {"cpu_lane": {**updated.to_dict, "updated_at": utc_now()}})
        return updated


def initialize_cpu_lane_capacity(runtime_root: Path | Any, *, capacity: int) -> CpuLanePolicy:
    """Set an initial CPU capacity, rejecting a conflicting shared policy."""
    if type(capacity) is not int or capacity < 0:
        raise ValueError("CPU lane capacity must be a non-negative integer.")
    paths = local_paths(_runtime_root(runtime_root))
    paths["cpu_policy"].parent.mkdir(parents=True, exist_ok=True)
    for name in ("cpu_provisional", "cpu_active", "cpu_released", "locks"):
        paths[name].mkdir(parents=True, exist_ok=True)
    with exclusive(paths["locks"] / "cpu-lane.lock"):
        _release_expired(paths)
        if paths["cpu_policy"].exists():
            current = _policy(paths)
            if current.capacity != capacity:
                raise ValueError(
                    "CPU lane capacity is already configured as "
                    f"{current.capacity}; use 'qexp agent config cpu set' to change it."
                )
            return current
        policy = CpuLanePolicy(capacity, 1)
        atomic_replace(paths["cpu_policy"], {"cpu_lane": {**policy.to_dict, "updated_at": utc_now()}})
        return policy


def cpu_reservation_snapshot(runtime_root: Path) -> tuple[CpuLanePolicy, tuple[dict[str, Any], ...]]:
    """Return a lock-consistent CPU policy and usage-bearing reservations."""
    paths = local_paths(runtime_root)
    with exclusive(paths["locks"] / "cpu-lane.lock"):
        _release_expired(paths)
        return _policy(paths), tuple(_reservations(paths))


def reserve_cpu(
    runtime_root: Path,
    task_id: str,
    cpu_slots: int,
    *,
    attempt_id: str | None = None,
    fencing_token: int | None = None,
    project_id: str | None = None,
    shared_root: str | None = None,
    machine_name: str | None = None,
    group_name: str | None = None,
    admitted_as_borrow: bool = False,
    worker_scheduling_role: str | None = None,
    group_worker_set_epoch: int | None = None,
    worker_state_epoch: int | None = None,
    executor_epoch: str | None = None,
    executor_request_id: str | None = None,
    registration_generation: str | None = None,
) -> dict[str, Any]:
    """Reserve CPU slots provisionally after checking machine-global capacity."""
    if type(cpu_slots) is not int or cpu_slots < 1:
        raise ValueError("CPU reservation cpu_slots must be a positive integer.")
    if type(admitted_as_borrow) is not bool:
        raise ValueError("CPU reservation admitted_as_borrow must be a boolean.")
    if worker_scheduling_role not in (None, "primary", "borrow"):
        raise ValueError("CPU reservation worker_scheduling_role must be None, 'primary', or 'borrow'.")
    for value, label in (
        (group_worker_set_epoch, "group_worker_set_epoch"),
        (worker_state_epoch, "worker_state_epoch"),
    ):
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"CPU reservation {label} must be a non-negative integer or null.")
    if admitted_as_borrow and (worker_scheduling_role != "borrow" or not isinstance(group_name, str) or not group_name):
        raise ValueError("CPU borrow admission requires worker_scheduling_role='borrow' and a nonempty group_name.")
    owner = _executor_owner(
        executor_epoch,
        executor_request_id,
        registration_generation,
        project_id=project_id,
        shared_root=shared_root,
        task_id=task_id,
        attempt_id=attempt_id,
        fencing_token=fencing_token,
    )
    paths = local_paths(runtime_root)
    with exclusive(paths["locks"] / "cpu-lane.lock"):
        _release_expired(paths)
        policy = _policy(paths)
        reserved = sum(item.get("cpu_slots", 0) for item in _reservations(paths))
        if reserved + cpu_slots > policy.capacity:
            raise ValueError("CPU lane has insufficient free slots.")
        reservation_id = new_id()
        reservation = {
            "reservation_id": reservation_id,
            "acquisition_id": new_id(),
            "project_id": project_id,
            "shared_root": shared_root,
            "group_name": group_name,
            "machine_name": machine_name,
            "task_id": task_id,
            "attempt_id": attempt_id,
            "fencing_token": fencing_token,
            "cpu_slots": cpu_slots,
            "admission": {
                "admitted_as_borrow": admitted_as_borrow,
                "worker_scheduling_role": worker_scheduling_role,
                "group_worker_set_epoch": group_worker_set_epoch,
                "worker_state_epoch": worker_state_epoch,
                "gpu_limit_gpus": None,
            },
            "state": "provisional",
            "created_at": utc_now(),
            "expires_at": (
                None
                if owner is not None
                else (datetime.now(timezone.utc) + timedelta(seconds=PROVISIONAL_TTL_SECONDS))
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            ),
            "released_at": None,
            "release_reason": None,
        }
        if owner is not None:
            reservation["executor_owner"] = owner
        value = {
            "reservation": reservation,
        }
        atomic_replace(paths["cpu_provisional"] / f"{reservation_id}.json", value)
        return value


def attach_cpu(runtime_root: Path, reservation_id: str, attempt_id: str, fencing_token: int) -> None:
    """Attach a matching, unexpired provisional CPU reservation to an Attempt."""
    paths = local_paths(runtime_root)
    with exclusive(paths["locks"] / "cpu-lane.lock"):
        source = paths["cpu_provisional"] / f"{reservation_id}.json"
        if not source.exists():
            if (paths["cpu_active"] / source.name).exists():
                return
            raise FileNotFoundError(source)
        value = read_json(source)
        if _is_expired(value):
            _release_expired(paths)
            raise RuntimeError("CPU provisional reservation has expired.")
        reservation = value["reservation"]
        if "executor_owner" in reservation:
            raise RuntimeError("executor-owned reservations require exact offer attachment.")
        if reservation.get("attempt_id") != attempt_id or reservation.get("fencing_token") != fencing_token:
            raise RuntimeError("CPU reservation identity does not match Attempt authority.")
        reservation["state"] = "active"
        atomic_replace(paths["cpu_active"] / source.name, value)
        source.unlink(missing_ok=True)


def attach_executor_offer(
    runtime_root: Path,
    identity: ReservationIdentity,
    attempt_id: str,
    fencing_token: int,
) -> bool:
    """Attach one exact executor-owned CPU offer, idempotently."""
    if identity.cpu_slots is None or not _identity_has_executor_owner(identity):
        raise ValueError("executor CPU offer identity is incomplete.")
    if identity.attempt_id != attempt_id or identity.fencing_token != fencing_token:
        return False
    paths = local_paths(runtime_root)
    with exclusive(paths["locks"] / "cpu-lane.lock"):
        name = f"{identity.reservation_id}.json"
        provisional = paths["cpu_provisional"] / name
        active = paths["cpu_active"] / name
        released = paths["cpu_released"] / name
        if active.exists():
            if released.exists():
                return False
            current = read_json(active).get("reservation", {})
            active_matches = (
                current.get("state") == "active"
                and identity.matches(current)
                and _matches_executor_owner(current, identity)
            )
            if not active_matches:
                return False
            if provisional.exists():
                source = read_json(provisional).get("reservation", {})
                if (
                    source.get("state") != "provisional"
                    or not identity.matches(source)
                    or not _matches_executor_owner(source, identity)
                ):
                    return False
                provisional.unlink(missing_ok=True)
            return True
        if not provisional.exists() or released.exists():
            return False
        value = read_json(provisional)
        reservation = value.get("reservation", {})
        if (
            reservation.get("state") != "provisional"
            or not identity.matches(reservation)
            or not _matches_executor_owner(reservation, identity)
        ):
            return False
        if _is_expired(value):
            return False
        reservation["state"] = "active"
        atomic_replace(active, value)
        provisional.unlink(missing_ok=True)
        return True


def release_executor_offer(
    runtime_root: Path,
    identity: ReservationIdentity,
    reason: str = "claim_not_committed",
) -> bool:
    """Release only an exact executor-owned provisional CPU offer."""
    if identity.cpu_slots is None or not _identity_has_executor_owner(identity):
        raise ValueError("executor CPU offer identity is incomplete.")
    paths = local_paths(runtime_root)
    with exclusive(paths["locks"] / "cpu-lane.lock"):
        source = paths["cpu_provisional"] / f"{identity.reservation_id}.json"
        active = paths["cpu_active"] / source.name
        released = paths["cpu_released"] / source.name
        if released.exists():
            if active.exists():
                return False
            current = read_json(released).get("reservation", {})
            if (
                current.get("state") != "released"
                or not identity.matches(current)
                or not _matches_executor_owner(current, identity)
            ):
                return False
            if source.exists():
                provisional = read_json(source).get("reservation", {})
                if (
                    provisional.get("state") != "provisional"
                    or not identity.matches(provisional)
                    or not _matches_executor_owner(provisional, identity)
                ):
                    return False
                source.unlink(missing_ok=True)
            return True
        if not source.exists() or active.exists():
            return False
        value = read_json(source)
        reservation = value.get("reservation", {})
        if (
            reservation.get("state") != "provisional"
            or not identity.matches(reservation)
            or not _matches_executor_owner(reservation, identity)
        ):
            return False
        reservation.update({"state": "released", "released_at": utc_now(), "release_reason": reason})
        atomic_replace(paths["cpu_released"] / source.name, value)
        source.unlink(missing_ok=True)
        return True


def classify_executor_offer(
    runtime_root: Path,
    identity: ReservationIdentity,
) -> Literal["absent", "matching_provisional", "matching_active", "matching_released", "conflict"]:
    """Read one CPU offer identity without expiring or changing it."""
    if identity.cpu_slots is None or not _identity_has_executor_owner(identity):
        raise ValueError("executor CPU offer identity is incomplete.")
    paths = local_paths(runtime_root)
    with exclusive(paths["locks"] / "cpu-lane.lock"):
        name = f"{identity.reservation_id}.json"
        found: list[tuple[str, Path]] = [
            (state, paths[f"cpu_{state}"] / name)
            for state in ("provisional", "active", "released")
            if (paths[f"cpu_{state}"] / name).exists()
        ]
        if not found:
            return "absent"
        if len(found) != 1:
            return "conflict"
        state, path = found[0]
        value = read_json(path)
        reservation = value.get("reservation", {})
        if (
            reservation.get("state") != state
            or not identity.matches(reservation)
            or not _matches_executor_owner(reservation, identity)
        ):
            return "conflict"
        if state == "provisional":
            return "matching_provisional"
        if state == "active":
            return "matching_active"
    return "matching_released"


def classify_exact_reservation(
    runtime_root: Path,
    identity: ReservationIdentity,
) -> Literal["absent", "matching_active", "matching_released", "matching_release_pair", "conflict"]:
    """Read one active/released CPU reservation using its complete identity."""
    paths = local_paths(runtime_root)
    with exclusive(paths["locks"] / "cpu-lane.lock"):
        name = f"{identity.reservation_id}.json"
        found = [
            (state, paths[f"cpu_{state}"] / name)
            for state in ("provisional", "active", "released")
            if (paths[f"cpu_{state}"] / name).exists()
        ]
        if not found:
            return "absent"
        states = {state for state, _path in found}
        if states == {"active", "released"}:
            if all(
                reservation.get("state") == state and identity.exactly_matches(reservation)
                for state, path in found
                for reservation in (read_json(path).get("reservation", {}),)
            ):
                return "matching_release_pair"
            return "conflict"
        if len(found) != 1 or "provisional" in states:
            return "conflict"
        state, path = found[0]
        reservation = read_json(path).get("reservation", {})
        if reservation.get("state") != state or not identity.exactly_matches(reservation):
            return "conflict"
        return "matching_active" if state == "active" else "matching_released"


def has_active_cpu_reservation(
    runtime_root: Path,
    reservation_id: str,
    *,
    task_id: str,
    attempt_id: str,
    fencing_token: int,
) -> bool:
    """Return whether the active CPU reservation still fences this launch."""
    paths = local_paths(runtime_root)
    with exclusive(paths["locks"] / "cpu-lane.lock"):
        _release_expired(paths)
        path = paths["cpu_active"] / f"{reservation_id}.json"
        if not path.exists():
            return False
        reservation = read_json(path).get("reservation", {})
        return (
            reservation.get("state") == "active"
            and reservation.get("reservation_id") == reservation_id
            and reservation.get("task_id") == task_id
            and reservation.get("attempt_id") == attempt_id
            and reservation.get("fencing_token") == fencing_token
        )


def release_cpu(runtime_root: Path, reservation_id: str, reason: str = "completed") -> bool:
    """Idempotently release one CPU reservation."""
    paths = local_paths(runtime_root)
    with exclusive(paths["locks"] / "cpu-lane.lock"):
        for source_root in (paths["cpu_active"], paths["cpu_provisional"]):
            source = source_root / f"{reservation_id}.json"
            if not source.exists():
                continue
            value = read_json(source)
            if source_root == paths["cpu_provisional"] and "executor_owner" in value.get("reservation", {}):
                return False
            value["reservation"].update({"state": "released", "released_at": utc_now(), "release_reason": reason})
            atomic_replace(paths["cpu_released"] / source.name, value)
            source.unlink(missing_ok=True)
            return True
    return False


def release_cpu_if_matches(runtime_root: Path, identity: Any, reason: str) -> bool:
    """Release a CPU reservation only when its complete identity is unchanged."""
    paths = local_paths(runtime_root)
    with exclusive(paths["locks"] / "cpu-lane.lock"):
        source = paths["cpu_active"] / f"{identity.reservation_id}.json"
        provisional = paths["cpu_provisional"] / source.name
        released = paths["cpu_released"] / source.name
        if not source.exists() or provisional.exists():
            return False
        value = read_json(source)
        reservation = value.get("reservation", {})
        if not identity.exactly_matches(reservation):
            return False
        if released.exists():
            released_reservation = read_json(released).get("reservation", {})
            if released_reservation.get("state") != "released" or not identity.exactly_matches(released_reservation):
                return False
        reservation.update({"state": "released", "released_at": utc_now(), "release_reason": reason})
        atomic_replace(released, value)
        source.unlink(missing_ok=True)
        return True


def retag_cpu_if_matches(
    runtime_root: Path,
    identity: Any,
    attempt_id: str,
    fencing_token: int,
) -> bool:
    """Retag an active CPU reservation only if its full identity is unchanged."""
    paths = local_paths(runtime_root)
    with exclusive(paths["locks"] / "cpu-lane.lock"):
        filename = f"{identity.reservation_id}.json"
        candidates = [paths[name] / filename for name in ("cpu_active", "cpu_provisional", "cpu_released")]
        present = [path for path in candidates if path.exists()]
        source = paths["cpu_active"] / filename
        if present != [source]:
            return False
        value = read_json(source)
        reservation = value.get("reservation", {})
        if not identity.exactly_matches(reservation) or reservation.get("attempt_id") != attempt_id:
            return False
        if reservation.get("fencing_token") == fencing_token:
            return True
        reservation["fencing_token"] = fencing_token
        reservation["retagged_at"] = utc_now()
        atomic_replace(source, value)
        return True
