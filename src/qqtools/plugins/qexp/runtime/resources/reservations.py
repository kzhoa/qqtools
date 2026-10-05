"""Machine-local GPU reservation truth."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from ...gpu_policy import GpuReservationPolicy, validate_gpu_reservation
from ..locks import exclusive
from ..paths import local_paths
from ..records import new_id, utc_now, validate_identifier
from ..store import atomic_replace, iter_json, read_json
from ..work_budget import diagnostic_increment, diagnostic_span

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
    shared_root: str | None = None
    registration_generation: str | None = None
    executor_epoch: str | None = None
    executor_request_id: str | None = None

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
            or reservation_id in {".", ".."}
            or Path(reservation_id).name != reservation_id
            or not isinstance(acquisition_id, str)
            or not acquisition_id
            or not isinstance(task_id, str)
            or not task_id
            or (
                gpu_ids is not None
                and (not isinstance(gpu_ids, list) or any(type(gpu_id) is not int for gpu_id in gpu_ids))
            )
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
        shared_root = reservation.get("shared_root")
        if shared_root is not None and not isinstance(shared_root, str):
            raise ValueError("active reservation shared_root must be a string or null.")
        executor_owner = reservation.get("executor_owner")
        if executor_owner is None:
            registration_generation = executor_epoch = executor_request_id = None
        else:
            if not isinstance(executor_owner, dict) or set(executor_owner) != {
                "executor_epoch",
                "request_id",
                "registration_generation",
            }:
                raise ValueError("active reservation executor owner is malformed.")
            executor_epoch = _require_bounded_id(executor_owner["executor_epoch"], "executor_epoch")
            executor_request_id = _require_bounded_id(executor_owner["request_id"], "executor_request_id")
            registration_generation = _require_bounded_id(
                executor_owner["registration_generation"], "registration_generation"
            )
        return cls(
            reservation_id,
            acquisition_id,
            project_id,
            task_id,
            attempt_id,
            fencing_token,
            tuple(gpu_ids or ()),
            cpu_slots,
            shared_root,
            registration_generation,
            executor_epoch,
            executor_request_id,
        )

    def matches(self, reservation: dict[str, Any]) -> bool:
        try:
            observed = type(self).from_record(reservation)
        except ValueError:
            return False
        base_fields = (
            "reservation_id",
            "acquisition_id",
            "project_id",
            "task_id",
            "attempt_id",
            "fencing_token",
            "gpu_ids",
            "cpu_slots",
        )
        if any(getattr(self, field) != getattr(observed, field) for field in base_fields):
            return False
        if not _identity_has_executor_owner(self):
            return True
        executor_fields = (
            "shared_root",
            "registration_generation",
            "executor_epoch",
            "executor_request_id",
        )
        return all(getattr(self, field) == getattr(observed, field) for field in executor_fields)

    def exactly_matches(self, reservation: dict[str, Any]) -> bool:
        """Return whether every identity field, including absent ownership, matches."""
        try:
            observed = type(self).from_record(reservation)
        except ValueError:
            return False
        return self == observed


def _require_bounded_id(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128 or "\x00" in value:
        raise ValueError(f"{label} must be a bounded nonempty identifier.")
    try:
        validate_identifier(value, label)
    except ValueError as exc:
        raise ValueError(f"{label} must be a bounded nonempty identifier.") from exc
    return value


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
    epoch = _require_bounded_id(executor_epoch, "executor_epoch")
    request_id = _require_bounded_id(executor_request_id, "executor_request_id")
    generation = _require_bounded_id(registration_generation, "registration_generation")
    if not project_id or not shared_root or not attempt_id or type(fencing_token) is not int or fencing_token <= 0:
        raise ValueError("executor-owned reservations require exact Project and Attempt identity.")
    _require_bounded_id(project_id, "project_id")
    _require_bounded_id(task_id, "task_id")
    _require_bounded_id(attempt_id, "attempt_id")
    if (
        not isinstance(shared_root, str)
        or len(shared_root) > 4096
        or "\x00" in shared_root
        or not Path(shared_root).is_absolute()
        or str(Path(shared_root)) != shared_root
    ):
        raise ValueError("executor-owned reservations require an absolute bounded shared_root.")
    return {
        "executor_epoch": epoch,
        "request_id": request_id,
        "registration_generation": generation,
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
    reservation = value["reservation"]
    if reservation.get("state") == "provisional" and "executor_owner" in reservation:
        return False
    expires_at = reservation.get("expires_at")
    if not expires_at:
        return False
    return datetime.fromisoformat(expires_at.replace("Z", "+00:00")) <= datetime.now(timezone.utc)


def _expire_provisionals(paths: dict[str, Path]) -> None:
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


def _reserve_locked(
    paths: dict[str, Path],
    task_id: str,
    gpu_ids: list[int],
    *,
    attempt_id: str | None,
    fencing_token: int | None,
    project_id: str | None,
    shared_root: str | None,
    machine_name: str | None,
    group_name: str | None,
    worker_scheduling_role: str | None,
    gpu_limit_gpus: int | None,
    group_worker_set_epoch: int | None,
    worker_state_epoch: int | None,
    admitted_as_borrow: bool,
    enforce_gpu_limit: bool,
    gpu_policy: GpuReservationPolicy | None = None,
    executor_owner: dict[str, str] | None = None,
) -> dict[str, Any]:
    _expire_provisionals(paths)
    if gpu_policy is not None:
        # The caller obtained discovery and inherited-environment observations
        # before taking this lock.  Validation only rereads the local policy;
        # it never probes hardware or touches shared project truth here.
        validate_gpu_reservation(paths["locks"].parent, gpu_ids, gpu_policy)
    active = _reservation_values(paths["active"])
    provisional = [value for value in _reservation_values(paths["provisional"]) if not _is_expired(value)]
    if {gpu for value in active + provisional for gpu in value["reservation"]["gpu_ids"]}.intersection(gpu_ids):
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
            raise ValueError(f"Group {group_name!r} GPU limit on machine {machine_name!r} would be exceeded.")
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
            None
            if executor_owner is not None
            else (datetime.now(timezone.utc) + timedelta(seconds=PROVISIONAL_TTL_SECONDS))
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        ),
        "released_at": None,
        "release_reason": None,
    }
    if executor_owner is not None:
        reservation["executor_owner"] = executor_owner
    value = {
        "reservation": reservation,
    }
    atomic_replace(paths["provisional"] / f"{reservation_id}.json", value)
    return value


def reserve(
    runtime_root: Path,
    task_id: str,
    gpu_ids: list[int],
    *,
    attempt_id: str | None = None,
    fencing_token: int | None = None,
    project_id: str | None = None,
    shared_root: str | None = None,
    machine_name: str | None = None,
    group_name: str | None = None,
    worker_scheduling_role: str | None = None,
    gpu_limit_gpus: int | None = None,
    group_worker_set_epoch: int | None = None,
    worker_state_epoch: int | None = None,
    admitted_as_borrow: bool = False,
    gpu_policy: GpuReservationPolicy | None = None,
    executor_epoch: str | None = None,
    executor_request_id: str | None = None,
    registration_generation: str | None = None,
) -> dict[str, Any]:
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
    with exclusive(paths["locks"] / "gpu-reservations.lock"):
        return _reserve_locked(
            paths,
            task_id,
            gpu_ids,
            attempt_id=attempt_id,
            fencing_token=fencing_token,
            project_id=project_id,
            shared_root=shared_root,
            machine_name=machine_name,
            group_name=group_name,
            worker_scheduling_role=worker_scheduling_role,
            gpu_limit_gpus=gpu_limit_gpus,
            group_worker_set_epoch=group_worker_set_epoch,
            worker_state_epoch=worker_state_epoch,
            admitted_as_borrow=admitted_as_borrow,
            enforce_gpu_limit=False,
            gpu_policy=gpu_policy,
            executor_owner=owner,
        )


def reserve_admitted(
    runtime_root: Path,
    task_id: str,
    gpu_ids: list[int],
    *,
    project_id: str,
    group_name: str,
    machine_name: str,
    gpu_limit_gpus: int | None,
    worker_scheduling_role: str,
    group_worker_set_epoch: int,
    worker_state_epoch: int,
    attempt_id: str | None = None,
    fencing_token: int | None = None,
    shared_root: str | None = None,
    admitted_as_borrow: bool = True,
    gpu_policy: GpuReservationPolicy | None = None,
    executor_epoch: str | None = None,
    executor_request_id: str | None = None,
    registration_generation: str | None = None,
) -> dict[str, Any]:
    """Atomically reserve GPUs after Group/Task admission authorization."""
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
    with exclusive(paths["locks"] / "gpu-reservations.lock"):
        return _reserve_locked(
            paths,
            task_id,
            gpu_ids,
            attempt_id=attempt_id,
            fencing_token=fencing_token,
            project_id=project_id,
            shared_root=shared_root,
            machine_name=machine_name,
            group_name=group_name,
            worker_scheduling_role=worker_scheduling_role,
            gpu_limit_gpus=gpu_limit_gpus,
            group_worker_set_epoch=group_worker_set_epoch,
            worker_state_epoch=worker_state_epoch,
            admitted_as_borrow=admitted_as_borrow,
            enforce_gpu_limit=True,
            gpu_policy=gpu_policy,
            executor_owner=owner,
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
        if "executor_owner" in value.get("reservation", {}):
            raise RuntimeError("executor-owned reservations require exact offer attachment.")
        value["reservation"].update({"state": "active", "attempt_id": attempt_id, "fencing_token": fencing_token})
        atomic_replace(paths["active"] / source.name, value)
        source.unlink(missing_ok=True)


def attach_executor_offer(
    runtime_root: Path,
    identity: ReservationIdentity,
    attempt_id: str,
    fencing_token: int,
) -> bool:
    """Attach one exact executor-owned provisional offer, idempotently."""
    if identity.cpu_slots is not None:
        from .cpu_lane import attach_executor_offer as attach_cpu_executor_offer

        return attach_cpu_executor_offer(runtime_root, identity, attempt_id, fencing_token)
    if not _identity_has_executor_owner(identity):
        raise ValueError("executor offer identity is incomplete.")
    if identity.attempt_id != attempt_id or identity.fencing_token != fencing_token:
        return False
    paths = local_paths(runtime_root)
    with exclusive(paths["locks"] / "gpu-reservations.lock"):
        name = f"{identity.reservation_id}.json"
        provisional = paths["provisional"] / name
        active = paths["active"] / name
        released = paths["released"] / name
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
                # Destination publication precedes source retirement. An exact
                # pair is the durable crash image of this same transition.
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
    """Release only an exact executor-owned provisional offer."""
    if identity.cpu_slots is not None:
        from .cpu_lane import release_executor_offer as release_cpu_executor_offer

        return release_cpu_executor_offer(runtime_root, identity, reason)
    if not _identity_has_executor_owner(identity):
        raise ValueError("executor offer identity is incomplete.")
    paths = local_paths(runtime_root)
    with exclusive(paths["locks"] / "gpu-reservations.lock"):
        source = paths["provisional"] / f"{identity.reservation_id}.json"
        active = paths["active"] / source.name
        released = paths["released"] / source.name
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
        atomic_replace(paths["released"] / source.name, value)
        source.unlink(missing_ok=True)
        return True


def classify_executor_offer(
    runtime_root: Path,
    identity: ReservationIdentity,
) -> Literal["absent", "matching_provisional", "matching_active", "matching_released", "conflict"]:
    """Read one offer identity without expiring, attaching, or releasing it."""
    if identity.cpu_slots is not None:
        from .cpu_lane import classify_executor_offer as classify_cpu_executor_offer

        return classify_cpu_executor_offer(runtime_root, identity)
    if not _identity_has_executor_owner(identity):
        raise ValueError("executor offer identity is incomplete.")
    paths = local_paths(runtime_root)
    with exclusive(paths["locks"] / "gpu-reservations.lock"):
        name = f"{identity.reservation_id}.json"
        found: list[tuple[str, Path]] = [
            (state, paths[state] / name)
            for state in ("provisional", "active", "released")
            if (paths[state] / name).exists()
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
    """Read one active/released reservation using its complete local identity."""
    if identity.cpu_slots is not None:
        from .cpu_lane import classify_exact_reservation as classify_exact_cpu_reservation

        return classify_exact_cpu_reservation(runtime_root, identity)
    paths = local_paths(runtime_root)
    with exclusive(paths["locks"] / "gpu-reservations.lock"):
        name = f"{identity.reservation_id}.json"
        found = [
            (state, paths[state] / name)
            for state in ("provisional", "active", "released")
            if (paths[state] / name).exists()
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
        filename = f"{identity.reservation_id}.json"
        matches = [paths[name] / filename for name in ("active", "provisional", "released")]
        present = [path for path in matches if path.exists()]
        path = paths["active"] / filename
        if present != [path]:
            return False
        value = read_json(path)
        reservation = value.get("reservation", {})
        if not identity.exactly_matches(reservation) or reservation.get("attempt_id") != attempt_id:
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
        source = next(
            (path for path in (paths["active"], paths["provisional"]) if (path / f"{reservation_id}.json").exists()),
            None,
        )
        if source is not None:
            source_file = source / f"{reservation_id}.json"
            value = read_json(source_file)
            if source == paths["provisional"] and "executor_owner" in value.get("reservation", {}):
                return
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
        provisional = paths["provisional"] / source.name
        released = paths["released"] / source.name
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
        reservation.update(
            {
                "state": "released",
                "released_at": utc_now(),
                "release_reason": reason,
            }
        )
        atomic_replace(released, value)
        source.unlink(missing_ok=True)
        return True


def release_termination_if_matches(
    runtime_root: Path,
    identity: ReservationIdentity,
    reason: str,
) -> bool:
    """Release one exact active or provisional reservation, including replay pairs."""
    if identity.cpu_slots is not None:
        from .cpu_lane import release_termination_if_matches as release_cpu_termination_if_matches

        return release_cpu_termination_if_matches(runtime_root, identity, reason)
    paths = local_paths(runtime_root)
    with exclusive(paths["locks"] / "gpu-reservations.lock"):
        filename = f"{identity.reservation_id}.json"
        active = paths["active"] / filename
        provisional = paths["provisional"] / filename
        released = paths["released"] / filename
        present = [path for path in (active, provisional, released) if path.exists()]
        if not present:
            return False
        records: dict[Path, dict[str, Any]] = {}
        for path in present:
            value = read_json(path)
            reservation = value.get("reservation", {})
            expected = "active" if path == active else "provisional" if path == provisional else "released"
            if reservation.get("state") != expected or not identity.exactly_matches(reservation):
                return False
            records[path] = value
        if active.exists() and provisional.exists():
            return False
        if released.exists() and not active.exists() and not provisional.exists():
            return True
        source = active if active.exists() else provisional
        value = records[source]
        reservation = value["reservation"]
        reservation.update({"state": "released", "released_at": utc_now(), "release_reason": reason})
        atomic_replace(released, value)
        source.unlink(missing_ok=True)
        return True


def reserved_gpu_ids(runtime_root: Path) -> set[int]:
    paths = local_paths(runtime_root)
    active = {gpu for value in _reservation_values(paths["active"]) for gpu in value["reservation"]["gpu_ids"]}
    provisional = {
        gpu
        for value in _reservation_values(paths["provisional"])
        if not _is_expired(value)
        for gpu in value["reservation"]["gpu_ids"]
    }
    return active | provisional


def active_reservations(runtime_root: Path) -> list[dict[str, Any]]:
    return list(reservation_snapshot(runtime_root).active)


def has_reservation(runtime_root: Path, reservation_id: str) -> bool:
    """Check one usage-bearing identity without enumerating either capacity lane."""
    if (
        not isinstance(reservation_id, str)
        or not reservation_id
        or reservation_id in {".", ".."}
        or Path(reservation_id).name != reservation_id
    ):
        raise ValueError("reservation_id must be a nonempty filename component.")
    paths = local_paths(runtime_root)
    # Keep independent capacity locks unnested, as in reservation_snapshot.
    for lock_name, active, provisional in (
        ("gpu-reservations.lock", "active", "provisional"),
        ("cpu-lane.lock", "cpu_active", "cpu_provisional"),
    ):
        with exclusive(paths["locks"] / lock_name):
            for name in (active, provisional):
                try:
                    value = read_json(paths[name] / f"{reservation_id}.json")
                except FileNotFoundError:
                    continue
                if value["reservation"]["reservation_id"] != reservation_id:
                    raise ValueError("reservation record does not match its filename.")
                if name == active or not _is_expired(value):
                    return True
    return False


def reservation_snapshot(runtime_root: Path) -> ReservationSnapshot:
    """Read active and unexpired provisional reservations without mutating state."""
    paths = local_paths(runtime_root)
    with exclusive(paths["locks"] / "gpu-reservations.lock"):
        active = [value["reservation"] for value in _reservation_values(paths["active"])]
        provisional = [
            value["reservation"] for value in _reservation_values(paths["provisional"]) if not _is_expired(value)
        ]
        reserved = {
            gpu_id
            for reservation in active + provisional
            for gpu_id in reservation.get("gpu_ids", [])
            if type(gpu_id) is int
        }
        gpu_snapshot = ReservationSnapshot(tuple(active), frozenset(reserved), tuple(provisional))
    # CPU and GPU reservations have independent locks and capacity domains.  Do not nest
    # their locks: callers need one complete machine view, but neither lane may block a
    # mutation in the other while it performs shared-root work.
    from .cpu_lane import cpu_reservation_snapshot

    _policy, cpu_reservations = cpu_reservation_snapshot(runtime_root)
    return ReservationSnapshot(
        gpu_snapshot.active + tuple(item for item in cpu_reservations if item.get("state") == "active"),
        gpu_snapshot.reserved_gpu_ids,
        gpu_snapshot.provisional + tuple(item for item in cpu_reservations if item.get("state") == "provisional"),
    )


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
        active = [value["reservation"] for value in _reservation_values(paths["active"])]
        provisional = [
            value["reservation"] for value in _reservation_values(paths["provisional"]) if not _is_expired(value)
        ]
        reserved = {
            gpu_id
            for reservation in active + provisional
            for gpu_id in reservation.get("gpu_ids", [])
            if type(gpu_id) is int
        }
        gpu_snapshot = ReservationSnapshot(tuple(active), frozenset(reserved), tuple(provisional))
    from .cpu_lane import cpu_reservation_snapshot

    _policy, cpu_reservations = cpu_reservation_snapshot(runtime_root)
    return ReservationSnapshot(
        gpu_snapshot.active + tuple(item for item in cpu_reservations if item.get("state") == "active"),
        gpu_snapshot.reserved_gpu_ids,
        gpu_snapshot.provisional + tuple(item for item in cpu_reservations if item.get("state") == "provisional"),
    )


def release_expired_provisionals(runtime_root: Path) -> list[str]:
    paths = local_paths(runtime_root)
    released: list[str] = []
    with exclusive(paths["locks"] / "gpu-reservations.lock"):
        for path, value in _reservation_entries(paths["provisional"]):
            if _is_expired(value):
                reservation_id = value["reservation"]["reservation_id"]
                value["reservation"].update(
                    {"state": "released", "released_at": utc_now(), "release_reason": "provisional_expired"}
                )
                atomic_replace(paths["released"] / path.name, value)
                path.unlink(missing_ok=True)
                released.append(reservation_id)
    return released
