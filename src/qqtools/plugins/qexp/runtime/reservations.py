"""Machine-local GPU reservation truth."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .paths import local_paths
from .locks import exclusive
from .store import atomic_replace, iter_json, read_json
from .records import new_id, utc_now
from .work_budget import diagnostic_increment, diagnostic_span

PROVISIONAL_TTL_SECONDS = 30


@dataclass(frozen=True, slots=True)
class ReservationIdentity:
    """Stable identity used to fence reconciliation mutations."""

    reservation_id: str
    acquisition_id: str
    project_id: str | None
    task_id: str
    attempt_id: str | None
    fencing_token: int | None
    gpu_ids: tuple[int, ...]
    cpu_slots: int | None

    @classmethod
    def from_record(cls, reservation: dict[str, Any]) -> "ReservationIdentity":
        reservation_id = reservation.get("reservation_id")
        acquisition_id = reservation.get("acquisition_id")
        task_id = reservation.get("task_id")
        gpu_ids = reservation.get("gpu_ids")
        cpu_slots = reservation.get("cpu_slots")
        if (
            not isinstance(reservation_id, str)
            or not reservation_id
            or not isinstance(acquisition_id, str)
            or not acquisition_id
            or not isinstance(task_id, str)
            or not task_id
            or (gpu_ids is not None and (
                not isinstance(gpu_ids, list) or any(type(gpu_id) is not int for gpu_id in gpu_ids)
            ))
            or (cpu_slots is not None and (type(cpu_slots) is not int or cpu_slots < 1))
            or ((gpu_ids is None) == (cpu_slots is None))
        ):
            raise ValueError("active reservation has an invalid identity.")
        project_id = reservation.get("project_id")
        attempt_id = reservation.get("attempt_id")
        fencing_token = reservation.get("fencing_token")
        if project_id is not None and not isinstance(project_id, str):
            raise ValueError("active reservation project_id must be a string or null.")
        if attempt_id is not None and not isinstance(attempt_id, str):
            raise ValueError("active reservation attempt_id must be a string or null.")
        if fencing_token is not None and type(fencing_token) is not int:
            raise ValueError("active reservation fencing_token must be an integer or null.")
        return cls(
            reservation_id,
            acquisition_id,
            project_id,
            task_id,
            attempt_id,
            fencing_token,
            tuple(gpu_ids or ()),
            cpu_slots,
        )

    def matches(self, reservation: dict[str, Any]) -> bool:
        try:
            return self == type(self).from_record(reservation)
        except ValueError:
            return False


@dataclass(frozen=True, slots=True)
class ReservationSnapshot:
    """One lock-consistent view of active and unexpired provisional records."""

    active: tuple[dict[str, Any], ...]
    reserved_gpu_ids: frozenset[int]
    provisional: tuple[dict[str, Any], ...] = ()

    @property
    def reservations(self) -> tuple[dict[str, Any], ...]:
        """Return all usage-bearing records represented by this snapshot."""
        return self.active + self.provisional


def _reservation_entries(directory: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Read one reservation directory with shared diagnostic accounting."""
    with diagnostic_span("reservation_enumeration"):
        paths = iter_json(directory)
        diagnostic_increment("reservation_enumeration.entries", len(paths))
        return [(path, read_json(path)) for path in paths]


def _reservation_values(directory: Path) -> list[dict[str, Any]]:
    return [value for _, value in _reservation_entries(directory)]


def _is_expired(value: dict[str, Any]) -> bool:
    expires_at = value["reservation"].get("expires_at")
    if not expires_at:
        return False
    return datetime.fromisoformat(expires_at.replace("Z", "+00:00")) <= datetime.now(timezone.utc)


def _expire_provisionals(paths: dict[str, Path]) -> None:
    for path, value in _reservation_entries(paths["provisional"]):
        if not _is_expired(value):
            continue
        value["reservation"].update({
            "state": "released",
            "released_at": utc_now(),
            "release_reason": "provisional_expired",
        })
        atomic_replace(paths["released"] / path.name, value)
        path.unlink(missing_ok=True)


def _reserve_locked(
    paths: dict[str, Path], task_id: str, gpu_ids: list[int], *, attempt_id: str | None,
    fencing_token: int | None, project_id: str | None, shared_root: str | None,
    machine_name: str | None, group_name: str | None,
    worker_scheduling_role: str | None, gpu_limit_gpus: int | None,
    group_worker_set_epoch: int | None, worker_state_epoch: int | None,
    admitted_as_borrow: bool, enforce_gpu_limit: bool,
) -> dict[str, Any]:
    _expire_provisionals(paths)
    active = _reservation_values(paths["active"])
    provisional = [
        value for value in _reservation_values(paths["provisional"])
        if not _is_expired(value)
    ]
    if {
        gpu for value in active + provisional for gpu in value["reservation"]["gpu_ids"]
    }.intersection(gpu_ids):
        raise ValueError("requested GPU is already reserved by qexp.")
    if enforce_gpu_limit and gpu_limit_gpus is not None:
        usage = sum(
            len(value["reservation"].get("gpu_ids", []))
            for value in active + provisional
            if value["reservation"].get("project_id") == project_id
            and value["reservation"].get("group_name") == group_name
            and value["reservation"].get("machine_name") == machine_name
        )
        if usage + len(gpu_ids) > gpu_limit_gpus:
            raise ValueError(
                f"Group {group_name!r} GPU limit on machine {machine_name!r} would be exceeded."
            )
    reservation_id = new_id()
    value = {"reservation": {
        "reservation_id": reservation_id,
        "acquisition_id": new_id(),
        "project_id": project_id,
        "shared_root": shared_root,
        "group_name": group_name,
        "machine_name": machine_name,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "fencing_token": fencing_token,
        "gpu_ids": list(gpu_ids),
        "admission": {
            "worker_scheduling_role": worker_scheduling_role,
            "gpu_limit_gpus": gpu_limit_gpus,
            "group_worker_set_epoch": group_worker_set_epoch,
            "worker_state_epoch": worker_state_epoch,
            "admitted_as_borrow": admitted_as_borrow,
        },
        "state": "provisional",
        "created_at": utc_now(),
        "expires_at": (
            datetime.now(timezone.utc) + timedelta(seconds=PROVISIONAL_TTL_SECONDS)
        ).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "released_at": None,
        "release_reason": None,
    }}
    atomic_replace(paths["provisional"] / f"{reservation_id}.json", value)
    return value


def reserve(runtime_root: Path, task_id: str, gpu_ids: list[int], *, attempt_id: str | None = None,
            fencing_token: int | None = None, project_id: str | None = None,
            shared_root: str | None = None, machine_name: str | None = None,
            group_name: str | None = None, worker_scheduling_role: str | None = None,
            gpu_limit_gpus: int | None = None, group_worker_set_epoch: int | None = None,
            worker_state_epoch: int | None = None,
            admitted_as_borrow: bool = False) -> dict[str, Any]:
    paths = local_paths(runtime_root)
    with exclusive(paths["locks"] / "gpu-reservations.lock"):
        return _reserve_locked(
            paths, task_id, gpu_ids, attempt_id=attempt_id, fencing_token=fencing_token,
            project_id=project_id, shared_root=shared_root, machine_name=machine_name,
            group_name=group_name, worker_scheduling_role=worker_scheduling_role,
            gpu_limit_gpus=gpu_limit_gpus,
            group_worker_set_epoch=group_worker_set_epoch, worker_state_epoch=worker_state_epoch,
            admitted_as_borrow=admitted_as_borrow, enforce_gpu_limit=False,
        )


def reserve_admitted(
    runtime_root: Path, task_id: str, gpu_ids: list[int], *, project_id: str,
    group_name: str, machine_name: str, gpu_limit_gpus: int | None,
    worker_scheduling_role: str, group_worker_set_epoch: int, worker_state_epoch: int,
    attempt_id: str | None = None, fencing_token: int | None = None,
    shared_root: str | None = None, admitted_as_borrow: bool = True,
) -> dict[str, Any]:
    """Atomically reserve GPUs after Group/Task admission authorization."""
    paths = local_paths(runtime_root)
    with exclusive(paths["locks"] / "gpu-reservations.lock"):
        return _reserve_locked(
            paths, task_id, gpu_ids, attempt_id=attempt_id, fencing_token=fencing_token,
            project_id=project_id, shared_root=shared_root, machine_name=machine_name,
            group_name=group_name, worker_scheduling_role=worker_scheduling_role,
            gpu_limit_gpus=gpu_limit_gpus,
            group_worker_set_epoch=group_worker_set_epoch, worker_state_epoch=worker_state_epoch,
            admitted_as_borrow=admitted_as_borrow, enforce_gpu_limit=True,
        )


def attach(runtime_root: Path, reservation_id: str, attempt_id: str, fencing_token: int) -> None:
    paths = local_paths(runtime_root)
    with exclusive(paths["locks"] / "gpu-reservations.lock"):
        source = paths["provisional"] / f"{reservation_id}.json"
        if not source.exists():
            if (paths["active"] / source.name).exists():
                return
            raise FileNotFoundError(source)
        value = read_json(source)
        value["reservation"].update({"state": "active", "attempt_id": attempt_id, "fencing_token": fencing_token})
        atomic_replace(paths["active"] / source.name, value)
        source.unlink(missing_ok=True)


def retag(runtime_root: Path, reservation_id: str, attempt_id: str, fencing_token: int) -> bool:
    """Idempotently align an active reservation with recovered Attempt authority."""
    paths = local_paths(runtime_root)
    with exclusive(paths["locks"] / "gpu-reservations.lock"):
        path = paths["active"] / f"{reservation_id}.json"
        if not path.exists():
            return False
        value = read_json(path)
        reservation = value["reservation"]
        if reservation.get("attempt_id") != attempt_id:
            return False
        if reservation.get("fencing_token") == fencing_token:
            return True
        reservation["fencing_token"] = fencing_token
        reservation["retagged_at"] = utc_now()
        atomic_replace(path, value)
        return True


def retag_if_matches(
    runtime_root: Path,
    identity: ReservationIdentity,
    attempt_id: str,
    fencing_token: int,
) -> bool:
    """Retag an active reservation only if its full identity is unchanged."""
    paths = local_paths(runtime_root)
    with exclusive(paths["locks"] / "gpu-reservations.lock"):
        path = paths["active"] / f"{identity.reservation_id}.json"
        if not path.exists():
            return False
        value = read_json(path)
        reservation = value.get("reservation", {})
        if not identity.matches(reservation) or reservation.get("attempt_id") != attempt_id:
            return False
        if reservation.get("fencing_token") == fencing_token:
            return True
        reservation["fencing_token"] = fencing_token
        reservation["retagged_at"] = utc_now()
        atomic_replace(path, value)
        return True


def release(runtime_root: Path, reservation_id: str, reason: str = "completed") -> None:
    paths = local_paths(runtime_root)
    with exclusive(paths["locks"] / "gpu-reservations.lock"):
        source = next((path for path in (paths["active"], paths["provisional"])
                       if (path / f"{reservation_id}.json").exists()), None)
        if source is not None:
            source_file = source / f"{reservation_id}.json"
            value = read_json(source_file)
            value["reservation"].update({"state": "released", "released_at": utc_now(), "release_reason": reason})
            atomic_replace(paths["released"] / source_file.name, value)
            source_file.unlink(missing_ok=True)
            return
    from .cpu_lane import release_cpu
    release_cpu(runtime_root, reservation_id, reason)


def release_if_matches(
    runtime_root: Path,
    identity: ReservationIdentity,
    reason: str,
) -> bool:
    """Release an active reservation only if its full identity is unchanged."""
    if identity.cpu_slots is not None:
        from .cpu_lane import release_cpu_if_matches
        return release_cpu_if_matches(runtime_root, identity, reason)
    paths = local_paths(runtime_root)
    with exclusive(paths["locks"] / "gpu-reservations.lock"):
        source = paths["active"] / f"{identity.reservation_id}.json"
        if not source.exists():
            return False
        value = read_json(source)
        reservation = value.get("reservation", {})
        if not identity.matches(reservation):
            return False
        reservation.update(
            {
                "state": "released",
                "released_at": utc_now(),
                "release_reason": reason,
            }
        )
        atomic_replace(paths["released"] / source.name, value)
        source.unlink(missing_ok=True)
        return True


def reserved_gpu_ids(runtime_root: Path) -> set[int]:
    paths = local_paths(runtime_root)
    active = {gpu for value in _reservation_values(paths["active"])
              for gpu in value["reservation"]["gpu_ids"]}
    provisional = {
        gpu
        for value in _reservation_values(paths["provisional"])
        if not _is_expired(value)
        for gpu in value["reservation"]["gpu_ids"]
    }
    return active | provisional


def active_reservations(runtime_root: Path) -> list[dict[str, Any]]:
    return [value["reservation"] for value in _reservation_values(local_paths(runtime_root)["active"])]


def reservation_snapshot(runtime_root: Path) -> ReservationSnapshot:
    """Read active and unexpired provisional reservations without mutating state."""
    paths = local_paths(runtime_root)
    with exclusive(paths["locks"] / "gpu-reservations.lock"):
        active = [value["reservation"] for value in _reservation_values(paths["active"])]
        provisional = [
            value["reservation"]
            for value in _reservation_values(paths["provisional"])
            if not _is_expired(value)
        ]
        reserved = {
            gpu_id
            for reservation in active + provisional
            for gpu_id in reservation.get("gpu_ids", [])
            if type(gpu_id) is int
        }
        return ReservationSnapshot(tuple(active), frozenset(reserved), tuple(provisional))


def reconcile_snapshot(runtime_root: Path) -> ReservationSnapshot:
    """Expire provisional records and return one locked active-reservation snapshot."""
    paths = local_paths(runtime_root)
    with exclusive(paths["locks"] / "gpu-reservations.lock"):
        for path, value in _reservation_entries(paths["provisional"]):
            if not _is_expired(value):
                continue
            value["reservation"].update(
                {
                    "state": "released",
                    "released_at": utc_now(),
                    "release_reason": "provisional_expired",
                }
            )
            atomic_replace(paths["released"] / path.name, value)
            path.unlink(missing_ok=True)
        active = [
            value["reservation"]
            for value in _reservation_values(paths["active"])
        ]
        provisional = [
            value["reservation"]
            for value in _reservation_values(paths["provisional"])
            if not _is_expired(value)
        ]
        reserved = {
            gpu_id
            for reservation in active + provisional
            for gpu_id in reservation.get("gpu_ids", [])
            if type(gpu_id) is int
        }
        return ReservationSnapshot(tuple(active), frozenset(reserved), tuple(provisional))


def release_expired_provisionals(runtime_root: Path) -> list[str]:
    paths = local_paths(runtime_root)
    released: list[str] = []
    with exclusive(paths["locks"] / "gpu-reservations.lock"):
        for path, value in _reservation_entries(paths["provisional"]):
            if _is_expired(value):
                reservation_id = value["reservation"]["reservation_id"]
                value["reservation"].update({"state": "released", "released_at": utc_now(),
                                               "release_reason": "provisional_expired"})
                atomic_replace(paths["released"] / path.name, value)
                path.unlink(missing_ok=True)
                released.append(reservation_id)
    return released
