"""Bounded, read-only discovery of reservations owned by the selected runtime."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any

from qqtools.qexp._progress_protocol import read_advisory_snapshot

from .agent.bindings import ProjectBinding
from .agent.context import MachineRuntime
from .agent.control_plane_health import HEALTH_FILENAME, HEALTH_SCHEMA_VERSION, MAX_HEALTH_BYTES
from .agent.snapshot_publication_health import observation_freshness
from .config_types import RootConfig
from .prelaunch_recovery import expected_target, reconcile_prelaunch
from .runtime.paths import attempt_path, task_path
from .runtime.records import validate_identifier
from .runtime.resources.reservations import ReservationIdentity

_MAX_CANDIDATE_ENTRIES = 16
_MAX_RECORD_READS = 32
_MAX_RECORD_BYTES = 256 * 1024
_ACTIVE_LANES = ("active", "cpu_active")
_READ_FAILURES = (OSError, TypeError, ValueError, KeyError, RuntimeError, AttributeError)

RecordReader = Callable[[Path], dict[str, Any] | None]


def _bounded_reader(read_json: RecordReader | None) -> Callable[[Path], dict[str, Any]]:
    cache: dict[Path, dict[str, Any] | Exception] = {}

    def read(path: Path) -> dict[str, Any]:
        if path not in cache:
            if len(cache) >= _MAX_RECORD_READS:
                raise ValueError("pre-launch observation read budget exhausted")
            try:
                value = (
                    read_advisory_snapshot(path, max_bytes=_MAX_RECORD_BYTES) if read_json is None else read_json(path)
                )
                cache[path] = FileNotFoundError(path) if value is None else value
            except _READ_FAILURES as exc:
                cache[path] = exc
        value = cache[path]
        if isinstance(value, Exception):
            raise value
        return value

    return read


def _binding_matches(cfg: RootConfig, runtime: MachineRuntime, binding: ProjectBinding | None) -> bool:
    return bool(
        binding is not None
        and binding.shared_root == cfg.shared_root
        and binding.machine_name == cfg.machine_name
        and binding.runtime_root == str(runtime.root)
        and binding.runtime_instance_id
    )


def _valid_reservation_id(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_-][a-zA-Z0-9._-]{0,127}", value) is not None


def parse_prelaunch_target(value: str) -> tuple[str, str]:
    """Decode an observed target into its exact record locator and identity digest."""
    reservation_id, sep, digest = value.rpartition(".")
    if not sep or not _valid_reservation_id(reservation_id) or re.fullmatch(r"[a-f0-9]{64}", digest) is None:
        raise ValueError("--expected-target must be the token printed by qexp status.")
    return reservation_id, digest


def _read_owner_record(
    path: Path,
    binding: ProjectBinding,
    lane: str,
    read: Callable[[Path], dict[str, Any]],
    *,
    state: str = "active",
) -> ReservationIdentity | None:
    try:
        record = read(path).get("reservation")
        if (
            not isinstance(record, dict)
            or record.get("state") != state
            or record.get("machine_name") != binding.machine_name
        ):
            return None
        identity = ReservationIdentity.from_record(record)
        if (
            identity.reservation_id != path.stem
            or identity.project_id != binding.project_id
            or identity.shared_root != str(binding.shared_root)
            or identity.attempt_id is None
            or identity.fencing_token is None
            or identity.fencing_token < 1
            or (identity.cpu_slots is not None) != (lane == "cpu_active")
            or (
                identity.registration_generation is not None
                and identity.registration_generation != binding.registration_generation
            )
        ):
            return None
        return identity
    except _READ_FAILURES:
        return None


def _reservation_at(
    runtime: MachineRuntime,
    binding: ProjectBinding,
    reservation_id: str,
    read: Callable[[Path], dict[str, Any]],
) -> ReservationIdentity | None:
    if not _valid_reservation_id(reservation_id):
        return None
    matches = [
        identity
        for lane in _ACTIVE_LANES
        if (identity := _read_owner_record(runtime.paths[lane] / f"{reservation_id}.json", binding, lane, read))
        is not None
    ]
    return matches[0] if len(matches) == 1 else None


def recovery_owner_available(runtime: MachineRuntime, binding: ProjectBinding) -> bool:
    """Check whether the selected reservation owner has current supervision."""
    try:
        agent = read_advisory_snapshot(runtime.paths["agent"] / "status.json", max_bytes=_MAX_RECORD_BYTES)[
            "machine_agent"
        ]
        health = read_advisory_snapshot(runtime.root / HEALTH_FILENAME, max_bytes=MAX_HEALTH_BYTES)
        lane = health["heartbeat"]
        return (
            agent.get("state") == "active"
            and agent.get("runtime_id") == binding.runtime_instance_id
            and isinstance(agent.get("instance_id"), str)
            and bool(agent["instance_id"])
            and health.get("schema_version") == HEALTH_SCHEMA_VERSION
            and health.get("instance_id") == agent["instance_id"]
            and observation_freshness(
                lane["last_success_at"], lane["effective_stale_after_seconds"], now=datetime.now(timezone.utc)
            )
            == "fresh"
        )
    except (*_READ_FAILURES, AttributeError):
        return False


def resolve_prelaunch_target(
    cfg: RootConfig,
    runtime: MachineRuntime,
    binding: ProjectBinding | None,
    task_id: str,
    target: str,
) -> ReservationIdentity | None:
    """Find the exact live owner record addressed by a copied command."""
    validate_identifier(task_id, "task_id")
    reservation_id, digest = parse_prelaunch_target(target)
    if binding is None or not binding.enabled or not _binding_matches(cfg, runtime, binding):
        return None
    identity = _reservation_at(runtime, binding, reservation_id, _bounded_reader(None))
    if (
        identity is None
        or identity.task_id != task_id
        or expected_target(identity, binding.runtime_instance_id) != digest
    ):
        return None
    return identity


def _project_reservations(
    runtime: MachineRuntime, binding: ProjectBinding, read: Callable[[Path], dict[str, Any]]
) -> Iterator[ReservationIdentity]:
    for lane in _ACTIVE_LANES:
        try:
            with os.scandir(runtime.paths[lane]) as entries:
                for entry in islice(entries, _MAX_CANDIDATE_ENTRIES):
                    path = Path(entry.path)
                    if path.suffix != ".json" or not _valid_reservation_id(path.stem):
                        continue
                    identity = _read_owner_record(path, binding, lane, read)
                    if identity is not None:
                        yield identity
        except OSError:
            continue


def _task_reservation_evidence(
    cfg: RootConfig,
    task_id: str,
    read: Callable[[Path], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], bool] | None:
    try:
        validate_identifier(task_id, "task_id")
        if task_id in {".", ".."}:
            return None
        task = read(task_path(cfg.shared_root, task_id))["task"]
        if task["task_id"] != task_id:
            return None
        claim = task["claim_control"].get("active_claim")
        number = task["attempt_control"].get("current_attempt_number")
        if type(number) is not int or number < 1:
            return None
        attempt = read(attempt_path(cfg.shared_root, task_id, number))["attempt"]
        if (
            attempt.get("task_id") != task_id
            or attempt.get("attempt_number") != number
            or not _valid_reservation_id(attempt.get("reservation_id"))
        ):
            return None
        if claim is None and (
            task["attempt_control"].get("current_attempt_id") is None
            and task["state"].get("projection") == "queued"
            and task["state"].get("reason") == "lease_expired_before_launch"
            and attempt.get("phase") == "cancelled"
            and attempt.get("result", {}).get("reason") == "lease_expired_before_launch"
        ):
            return {}, attempt, True
        if isinstance(claim, dict) and (
            attempt.get("phase") == "claimed"
            and claim.get("attempt_number") == number
            and claim.get("attempt_id")
            == attempt.get("attempt_id")
            == task["attempt_control"].get("current_attempt_id")
            and claim.get("machine_name") == attempt.get("machine_name")
            and claim.get("reservation_id") == attempt.get("reservation_id")
            and claim.get("fencing_token") == attempt.get("current_fencing_token")
        ):
            return claim, attempt, False
    except _READ_FAILURES:
        return None
    return None


def _owner_unavailable(
    claim: dict[str, Any],
    attempt: dict[str, Any],
    is_requeued: bool,
) -> dict[str, Any]:
    age_seconds = None
    try:
        claimed_at = claim.get("claimed_at") or attempt["lease"]["claimed_at"]
        age_seconds = max(
            0.0,
            (datetime.now(timezone.utc) - datetime.fromisoformat(claimed_at.replace("Z", "+00:00"))).total_seconds(),
        )
    except (KeyError, AttributeError, TypeError, ValueError, OverflowError):
        pass
    gpu_ids = attempt.get("assigned_gpus")
    authorization = attempt.get("authorization")
    timestamps = attempt.get("timestamps")
    return {
        "reason": "owner_runtime_unavailable",
        "project_outcome": "already_requeued" if is_requeued else "retained",
        "machine_outcome": "cleanup_pending" if is_requeued else "retained",
        "gpu_ids": gpu_ids
        if isinstance(gpu_ids, list) and all(type(gpu) is int and gpu >= 0 for gpu in gpu_ids)
        else [],
        "task_id": attempt["task_id"],
        "attempt_id": attempt.get("attempt_id"),
        "attempt_phase": attempt.get("phase"),
        "machine_name": attempt.get("machine_name"),
        "runtime_id": None,
        "reservation_id": attempt["reservation_id"],
        "reservation_verified": False,
        "age_seconds": age_seconds,
        "safe_at": None,
        "overdue_seconds": None,
        "launch_evidence": {
            "claim_launch_state": claim.get("launch_state"),
            "launch_id": claim.get("launch_id")
            or (authorization.get("launch_id") if isinstance(authorization, dict) else None),
            "launch_authorized_at": claim.get("launch_authorized_at")
            or (timestamps.get("launch_authorized_at") if isinstance(timestamps, dict) else None),
            "ownership_transition": claim.get("ownership_transition") is not None,
        },
        "expected_target": None,
        "actionable": False,
    }


def _observe_reservation(
    cfg: RootConfig,
    runtime: MachineRuntime,
    binding: ProjectBinding,
    identity: ReservationIdentity,
    read: Callable[[Path], dict[str, Any]],
) -> dict[str, Any] | None:
    try:
        result = reconcile_prelaunch(
            cfg,
            identity,
            runtime_root=runtime.root,
            runtime_id=binding.runtime_instance_id,
            apply=False,
            read_record=read,
        )
    except _READ_FAILURES:
        return None
    if result.get("reason") in {"not_prelaunch", "changed_target", "reservation_missing"} or result.get(
        "machine_outcome"
    ) in {
        "missing",
        "already_released",
    }:
        return None
    if result.get("attempt_phase") in {"starting", "running", "orphaned"}:
        return None
    result["machine_name"] = binding.machine_name
    result["reservation_verified"] = result.get("reason") not in {"runtime_binding_changed", "reservation_unavailable"}
    result["actionable"] = (
        binding.enabled
        and result.get("actionable") is True
        and identity == _reservation_at(runtime, binding, identity.reservation_id, read)
    )
    result["expected_target"] = f"{identity.reservation_id}.{expected_target(identity, binding.runtime_instance_id)}"
    return result


def _observe_task(
    cfg: RootConfig,
    runtime: MachineRuntime,
    binding: ProjectBinding | None,
    task_id: str,
    read: Callable[[Path], dict[str, Any]],
) -> dict[str, Any] | None:
    evidence = _task_reservation_evidence(cfg, task_id, read)
    if evidence is None:
        return None
    claim, attempt, is_requeued = evidence
    reservation_id = attempt["reservation_id"]
    if binding is not None and attempt.get("machine_name") == binding.machine_name:
        identity = _reservation_at(runtime, binding, reservation_id, read)
        if identity is not None and (
            identity.task_id,
            identity.attempt_id,
            identity.fencing_token,
        ) == (task_id, attempt.get("attempt_id"), attempt.get("current_fencing_token")):
            return _observe_reservation(cfg, runtime, binding, identity, read)
        if is_requeued:
            for lane in _ACTIVE_LANES:
                directory = "cpu_released" if lane == "cpu_active" else "released"
                released = _read_owner_record(
                    runtime.paths[directory] / f"{reservation_id}.json", binding, lane, read, state="released"
                )
                if released is not None and (
                    released.task_id,
                    released.attempt_id,
                    released.fencing_token,
                ) == (task_id, attempt.get("attempt_id"), attempt.get("current_fencing_token")):
                    return None
    return _owner_unavailable(claim, attempt, is_requeued)


def _local_cfg(cfg: RootConfig, runtime: MachineRuntime, binding: ProjectBinding) -> RootConfig:
    return RootConfig.from_canonical_paths(
        cfg.shared_root, cfg.project_root, binding.machine_name, runtime.project_paths(binding.project_id)["root"]
    )


def observe_prelaunch(
    cfg: RootConfig,
    runtime: MachineRuntime,
    binding: ProjectBinding | None,
    *,
    task_id: str | None = None,
    read_json: RecordReader | None = None,
) -> list[dict[str, Any]]:
    """Sample the selected owner's active records or inspect one Task by exact paths."""
    if not _binding_matches(cfg, runtime, binding):
        binding = None
    if binding is not None:
        cfg = _local_cfg(cfg, runtime, binding)
    read = _bounded_reader(read_json)
    if task_id is not None:
        observed = _observe_task(cfg, runtime, binding, task_id, read)
        return [] if observed is None else [observed]
    if binding is None:
        return []
    return [
        observation
        for identity in _project_reservations(runtime, binding, read)
        if (observation := _observe_reservation(cfg, runtime, binding, identity, read)) is not None
    ]


def observe_task_prelaunch(
    cfg: RootConfig,
    runtime: MachineRuntime,
    binding: ProjectBinding | None,
    task_ids: Iterable[str],
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Observe selected Task rows within one shared additional record budget."""
    if not _binding_matches(cfg, runtime, binding):
        binding = None
    if binding is not None:
        cfg = _local_cfg(cfg, runtime, binding)
    read = _bounded_reader(None)
    for task_id in islice(task_ids, _MAX_RECORD_READS):
        observation = _observe_task(cfg, runtime, binding, task_id, read)
        if observation is not None:
            yield task_id, observation
