"""Machine-local GPU reservation truth."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .paths import local_paths
from .locks import exclusive
from .store import atomic_replace, iter_json, read_json
from .records import new_id, utc_now

PROVISIONAL_TTL_SECONDS = 30


def _is_expired(value: dict[str, Any]) -> bool:
    expires_at = value["reservation"].get("expires_at")
    if not expires_at:
        return False
    return datetime.fromisoformat(expires_at.replace("Z", "+00:00")) <= datetime.now(timezone.utc)


def reserve(runtime_root: Path, task_id: str, gpu_ids: list[int], *, attempt_id: str | None = None,
            fencing_token: int | None = None, project_id: str | None = None,
            shared_root: str | None = None, machine_name: str | None = None) -> dict[str, Any]:
    paths = local_paths(runtime_root)
    with exclusive(paths["locks"] / "gpu-reservations.lock"):
        for path in iter_json(paths["provisional"]):
            value = read_json(path)
            if _is_expired(value):
                value["reservation"].update({"state": "released", "released_at": utc_now(),
                                               "release_reason": "provisional_expired"})
                atomic_replace(paths["released"] / path.name, value)
                path.unlink(missing_ok=True)
        active = {gpu for path in iter_json(paths["active"]) for gpu in read_json(path)["reservation"]["gpu_ids"]}
        provisional = {gpu for path in iter_json(paths["provisional"]) for gpu in read_json(path)["reservation"]["gpu_ids"]}
        if active.intersection(gpu_ids) or provisional.intersection(gpu_ids):
            raise ValueError("requested GPU is already reserved by qexp.")
        reservation_id = new_id()
        value = {"reservation": {"reservation_id": reservation_id, "acquisition_id": new_id(),
            "project_id": project_id, "shared_root": shared_root, "machine_name": machine_name,
            "task_id": task_id, "attempt_id": attempt_id, "fencing_token": fencing_token,
            "gpu_ids": list(gpu_ids), "state": "provisional", "created_at": utc_now(),
            "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=PROVISIONAL_TTL_SECONDS)).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "released_at": None, "release_reason": None}}
        atomic_replace(paths["provisional"] / f"{reservation_id}.json", value)
        return value


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


def release(runtime_root: Path, reservation_id: str, reason: str = "completed") -> None:
    paths = local_paths(runtime_root)
    with exclusive(paths["locks"] / "gpu-reservations.lock"):
        source = next((path for path in (paths["active"], paths["provisional"])
                       if (path / f"{reservation_id}.json").exists()), None)
        if source is None:
            return
        source_file = source / f"{reservation_id}.json"
        value = read_json(source_file)
        value["reservation"].update({"state": "released", "released_at": utc_now(), "release_reason": reason})
        atomic_replace(paths["released"] / source_file.name, value)
        source_file.unlink(missing_ok=True)


def reserved_gpu_ids(runtime_root: Path) -> set[int]:
    paths = local_paths(runtime_root)
    active = {gpu for path in iter_json(paths["active"]) for gpu in read_json(path)["reservation"]["gpu_ids"]}
    provisional = {
        gpu
        for path in iter_json(paths["provisional"])
        for value in [read_json(path)]
        if not _is_expired(value)
        for gpu in value["reservation"]["gpu_ids"]
    }
    return active | provisional


def active_reservations(runtime_root: Path) -> list[dict[str, Any]]:
    return [read_json(path)["reservation"] for path in iter_json(local_paths(runtime_root)["active"])]


def release_expired_provisionals(runtime_root: Path) -> list[str]:
    paths = local_paths(runtime_root)
    released: list[str] = []
    with exclusive(paths["locks"] / "gpu-reservations.lock"):
        for path in iter_json(paths["provisional"]):
            value = read_json(path)
            if _is_expired(value):
                reservation_id = value["reservation"]["reservation_id"]
                value["reservation"].update({"state": "released", "released_at": utc_now(),
                                               "release_reason": "provisional_expired"})
                atomic_replace(paths["released"] / path.name, value)
                path.unlink(missing_ok=True)
                released.append(reservation_id)
    return released
