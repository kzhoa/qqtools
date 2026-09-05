"""Machine-global CPU-only lane policy and reservations."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .locks import exclusive
from .paths import local_paths
from .records import new_id, utc_now
from .store import atomic_replace, iter_json, read_json

PROVISIONAL_TTL_SECONDS = 30


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
    expires_at = value["reservation"].get("expires_at")
    return bool(expires_at) and datetime.fromisoformat(
        expires_at.replace("Z", "+00:00")
    ) <= datetime.now(timezone.utc)


def _values(directory: Path) -> list[tuple[Path, dict[str, Any]]]:
    return [(path, read_json(path)) for path in iter_json(directory)]


def _release_expired(paths: dict[str, Path]) -> None:
    for path, value in _values(paths["cpu_provisional"]):
        if _is_expired(value):
            value["reservation"].update({"state": "released", "released_at": utc_now(),
                                         "release_reason": "provisional_expired"})
            atomic_replace(paths["cpu_released"] / path.name, value)
            path.unlink(missing_ok=True)


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
        value["reservation"] for _, value in _values(paths["cpu_provisional"])
        if not _is_expired(value)
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
            raise ValueError(
                f"CPU lane capacity {capacity} is below reserved CPU slots {reserved}."
            )
        if capacity == current.capacity and paths["cpu_policy"].exists():
            return current
        updated = CpuLanePolicy(capacity, current.revision + 1)
        atomic_replace(paths["cpu_policy"], {"cpu_lane": {**updated.to_dict, "updated_at": utc_now()}})
        return updated


def cpu_reservation_snapshot(runtime_root: Path) -> tuple[CpuLanePolicy, tuple[dict[str, Any], ...]]:
    """Return a lock-consistent CPU policy and usage-bearing reservations."""
    paths = local_paths(runtime_root)
    with exclusive(paths["locks"] / "cpu-lane.lock"):
        _release_expired(paths)
        return _policy(paths), tuple(_reservations(paths))


def reserve_cpu(
    runtime_root: Path, task_id: str, cpu_slots: int, *, attempt_id: str | None = None,
    fencing_token: int | None = None, project_id: str | None = None,
    shared_root: str | None = None, machine_name: str | None = None,
    group_name: str | None = None,
) -> dict[str, Any]:
    """Reserve CPU slots provisionally after checking machine-global capacity."""
    if type(cpu_slots) is not int or cpu_slots < 1:
        raise ValueError("CPU reservation cpu_slots must be a positive integer.")
    paths = local_paths(runtime_root)
    with exclusive(paths["locks"] / "cpu-lane.lock"):
        _release_expired(paths)
        policy = _policy(paths)
        reserved = sum(item.get("cpu_slots", 0) for item in _reservations(paths))
        if reserved + cpu_slots > policy.capacity:
            raise ValueError("CPU lane has insufficient free slots.")
        reservation_id = new_id()
        value = {"reservation": {
            "reservation_id": reservation_id, "acquisition_id": new_id(), "project_id": project_id,
            "shared_root": shared_root, "group_name": group_name, "machine_name": machine_name,
            "task_id": task_id, "attempt_id": attempt_id, "fencing_token": fencing_token,
            "cpu_slots": cpu_slots, "state": "provisional", "created_at": utc_now(),
            "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=PROVISIONAL_TTL_SECONDS))
            .replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "released_at": None, "release_reason": None,
        }}
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
        if reservation.get("attempt_id") != attempt_id or reservation.get("fencing_token") != fencing_token:
            raise RuntimeError("CPU reservation identity does not match Attempt authority.")
        reservation["state"] = "active"
        atomic_replace(paths["cpu_active"] / source.name, value)
        source.unlink(missing_ok=True)


def release_cpu(runtime_root: Path, reservation_id: str, reason: str = "completed") -> bool:
    """Idempotently release one CPU reservation."""
    paths = local_paths(runtime_root)
    with exclusive(paths["locks"] / "cpu-lane.lock"):
        for source_root in (paths["cpu_active"], paths["cpu_provisional"]):
            source = source_root / f"{reservation_id}.json"
            if not source.exists():
                continue
            value = read_json(source)
            value["reservation"].update({"state": "released", "released_at": utc_now(),
                                         "release_reason": reason})
            atomic_replace(paths["cpu_released"] / source.name, value)
            source.unlink(missing_ok=True)
            return True
    return False


def release_cpu_if_matches(runtime_root: Path, identity: Any, reason: str) -> bool:
    """Release a CPU reservation only when its complete identity is unchanged."""
    paths = local_paths(runtime_root)
    with exclusive(paths["locks"] / "cpu-lane.lock"):
        source = paths["cpu_active"] / f"{identity.reservation_id}.json"
        if not source.exists():
            return False
        value = read_json(source)
        reservation = value.get("reservation", {})
        if (
            reservation.get("reservation_id") != identity.reservation_id
            or reservation.get("acquisition_id") != identity.acquisition_id
            or reservation.get("project_id") != identity.project_id
            or reservation.get("task_id") != identity.task_id
            or reservation.get("attempt_id") != identity.attempt_id
            or reservation.get("fencing_token") != identity.fencing_token
            or reservation.get("cpu_slots") != identity.cpu_slots
        ):
            return False
        reservation.update({"state": "released", "released_at": utc_now(), "release_reason": reason})
        atomic_replace(paths["cpu_released"] / source.name, value)
        source.unlink(missing_ok=True)
        return True
