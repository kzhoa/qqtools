from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.runtime.paths import local_paths
from qqtools.plugins.qexp.runtime.resources.cpu_lane import (
    attach_cpu,
    cpu_reservation_snapshot,
    initialize_cpu_lane_capacity,
    reserve_cpu,
)
from qqtools.plugins.qexp.runtime.resources.reservations import (
    ReservationIdentity,
    attach,
    attach_executor_offer,
    classify_executor_offer,
    release,
    release_executor_offer,
    reservation_snapshot,
    reserve,
)
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json


def _runtime(tmp_path: Path) -> MachineRuntime:
    runtime = MachineRuntime(tmp_path / "machine")
    runtime.ensure_layout()
    return runtime


def _owner() -> dict[str, str]:
    return {
        "executor_epoch": "a" * 32,
        "executor_request_id": "b" * 32,
        "registration_generation": "registration-a",
    }


def test_gpu_executor_offer_survives_ttl_and_requires_exact_apis(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    value = reserve(
        runtime.root,
        "task-a",
        [0],
        attempt_id="task-a-attempt-1",
        fencing_token=1,
        project_id="project-a",
        shared_root="/shared/project-a/.qexp",
        machine_name="gpu-1",
        **_owner(),
    )
    record = value["reservation"]
    identity = ReservationIdentity.from_record(record)
    path = local_paths(runtime.root)["provisional"] / f"{identity.reservation_id}.json"
    expired = read_json(path)
    expired["reservation"]["expires_at"] = "2000-01-01T00:00:00Z"
    atomic_replace(path, expired)

    assert reservation_snapshot(runtime.root).reserved_gpu_ids == {0}
    assert classify_executor_offer(runtime.root, identity) == "matching_provisional"
    with pytest.raises(RuntimeError, match="exact offer"):
        attach(runtime.root, identity.reservation_id, identity.attempt_id or "", identity.fencing_token or 0)
    assert release(runtime.root, identity.reservation_id, "ordinary_cleanup") is None
    assert classify_executor_offer(runtime.root, identity) == "matching_provisional"
    assert classify_executor_offer(runtime.root, replace(identity, executor_request_id="c" * 32)) == "conflict"

    assert attach_executor_offer(runtime.root, identity, "task-a-attempt-1", 1)
    assert classify_executor_offer(runtime.root, identity) == "matching_active"
    assert attach_executor_offer(runtime.root, identity, "task-a-attempt-1", 1)


def test_gpu_executor_offer_release_is_exact_and_provisional_only(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    record = reserve(
        runtime.root,
        "task-a",
        [0],
        attempt_id="task-a-attempt-1",
        fencing_token=1,
        project_id="project-a",
        shared_root="/shared/project-a/.qexp",
        machine_name="gpu-1",
        **_owner(),
    )["reservation"]
    identity = ReservationIdentity.from_record(record)

    assert not release_executor_offer(runtime.root, replace(identity, acquisition_id="wrong"))
    assert release_executor_offer(runtime.root, identity, "definitive_no_claim")
    assert classify_executor_offer(runtime.root, identity) == "matching_released"


def test_gpu_executor_offer_attach_replays_exact_cross_directory_crash_image(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    record = reserve(
        runtime.root,
        "task-a",
        [0],
        attempt_id="task-a-attempt-1",
        fencing_token=1,
        project_id="project-a",
        shared_root="/shared/project-a/.qexp",
        machine_name="gpu-1",
        **_owner(),
    )["reservation"]
    identity = ReservationIdentity.from_record(record)
    active_record = {"reservation": {**record, "state": "active"}}
    atomic_replace(local_paths(runtime.root)["active"] / f"{identity.reservation_id}.json", active_record)

    assert classify_executor_offer(runtime.root, identity) == "conflict"
    assert attach_executor_offer(runtime.root, identity, "task-a-attempt-1", 1)
    assert classify_executor_offer(runtime.root, identity) == "matching_active"
    assert not (local_paths(runtime.root)["provisional"] / f"{identity.reservation_id}.json").exists()
    assert not release_executor_offer(runtime.root, identity)


def test_gpu_executor_offer_release_replays_exact_cross_directory_crash_image(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    record = reserve(
        runtime.root,
        "task-a",
        [0],
        attempt_id="task-a-attempt-1",
        fencing_token=1,
        project_id="project-a",
        shared_root="/shared/project-a/.qexp",
        machine_name="gpu-1",
        **_owner(),
    )["reservation"]
    identity = ReservationIdentity.from_record(record)
    released_record = {
        "reservation": {
            **record,
            "state": "released",
            "released_at": "2026-10-02T00:00:00Z",
            "release_reason": "definitive_no_claim",
        }
    }
    atomic_replace(local_paths(runtime.root)["released"] / f"{identity.reservation_id}.json", released_record)

    assert classify_executor_offer(runtime.root, identity) == "conflict"
    assert release_executor_offer(runtime.root, identity, "definitive_no_claim")
    assert classify_executor_offer(runtime.root, identity) == "matching_released"
    assert not (local_paths(runtime.root)["provisional"] / f"{identity.reservation_id}.json").exists()


def test_gpu_executor_offer_mutations_reject_wrong_embedded_state(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    record = reserve(
        runtime.root,
        "task-a",
        [0],
        attempt_id="task-a-attempt-1",
        fencing_token=1,
        project_id="project-a",
        shared_root="/shared/project-a/.qexp",
        machine_name="gpu-1",
        **_owner(),
    )["reservation"]
    identity = ReservationIdentity.from_record(record)
    path = local_paths(runtime.root)["provisional"] / f"{identity.reservation_id}.json"
    atomic_replace(path, {"reservation": {**record, "state": "active"}})

    assert classify_executor_offer(runtime.root, identity) == "conflict"
    assert not attach_executor_offer(runtime.root, identity, "task-a-attempt-1", 1)
    assert not release_executor_offer(runtime.root, identity)


def test_gpu_executor_offer_replay_rejects_mismatched_transition_pair(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    record = reserve(
        runtime.root,
        "task-a",
        [0],
        attempt_id="task-a-attempt-1",
        fencing_token=1,
        project_id="project-a",
        shared_root="/shared/project-a/.qexp",
        machine_name="gpu-1",
        **_owner(),
    )["reservation"]
    identity = ReservationIdentity.from_record(record)
    atomic_replace(
        local_paths(runtime.root)["active"] / f"{identity.reservation_id}.json",
        {"reservation": {**record, "state": "active", "acquisition_id": "mismatched"}},
    )

    assert classify_executor_offer(runtime.root, identity) == "conflict"
    assert not attach_executor_offer(runtime.root, identity, "task-a-attempt-1", 1)
    assert classify_executor_offer(runtime.root, identity) == "conflict"


def test_cpu_executor_offer_survives_ttl_and_attaches_exactly(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    initialize_cpu_lane_capacity(runtime.root, capacity=4)
    record = reserve_cpu(
        runtime.root,
        "task-cpu",
        2,
        attempt_id="task-cpu-attempt-1",
        fencing_token=1,
        project_id="project-a",
        shared_root="/shared/project-a/.qexp",
        machine_name="gpu-1",
        **_owner(),
    )["reservation"]
    identity = ReservationIdentity.from_record(record)
    path = local_paths(runtime.root)["cpu_provisional"] / f"{identity.reservation_id}.json"
    expired = read_json(path)
    expired["reservation"]["expires_at"] = "2000-01-01T00:00:00Z"
    atomic_replace(path, expired)

    _policy, reservations = cpu_reservation_snapshot(runtime.root)
    assert [item["cpu_slots"] for item in reservations] == [2]
    assert classify_executor_offer(runtime.root, identity) == "matching_provisional"
    with pytest.raises(RuntimeError, match="exact offer"):
        attach_cpu(runtime.root, identity.reservation_id, "task-cpu-attempt-1", 1)
    assert attach_executor_offer(runtime.root, identity, "task-cpu-attempt-1", 1)
    assert classify_executor_offer(runtime.root, identity) == "matching_active"


def test_cpu_executor_offer_release_is_classified_for_replay(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    initialize_cpu_lane_capacity(runtime.root, capacity=2)
    record = reserve_cpu(
        runtime.root,
        "task-cpu",
        2,
        attempt_id="task-cpu-attempt-1",
        fencing_token=1,
        project_id="project-a",
        shared_root="/shared/project-a/.qexp",
        machine_name="gpu-1",
        **_owner(),
    )["reservation"]
    identity = ReservationIdentity.from_record(record)

    assert release_executor_offer(runtime.root, identity, "definitive_no_claim")
    assert classify_executor_offer(runtime.root, identity) == "matching_released"


def test_cpu_executor_offer_rejects_empty_task_identity(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    initialize_cpu_lane_capacity(runtime.root, capacity=1)

    with pytest.raises(ValueError, match="task_id"):
        reserve_cpu(
            runtime.root,
            "",
            1,
            attempt_id="task-cpu-attempt-1",
            fencing_token=1,
            project_id="project-a",
            shared_root="/shared/project-a/.qexp",
            machine_name="gpu-1",
            **_owner(),
        )


def test_cpu_executor_offer_attach_replays_exact_cross_directory_crash_image(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    initialize_cpu_lane_capacity(runtime.root, capacity=1)
    record = reserve_cpu(
        runtime.root,
        "task-cpu",
        1,
        attempt_id="task-cpu-attempt-1",
        fencing_token=1,
        project_id="project-a",
        shared_root="/shared/project-a/.qexp",
        machine_name="gpu-1",
        **_owner(),
    )["reservation"]
    identity = ReservationIdentity.from_record(record)
    atomic_replace(
        local_paths(runtime.root)["cpu_active"] / f"{identity.reservation_id}.json",
        {"reservation": {**record, "state": "active"}},
    )

    assert classify_executor_offer(runtime.root, identity) == "conflict"
    assert attach_executor_offer(runtime.root, identity, "task-cpu-attempt-1", 1)
    assert classify_executor_offer(runtime.root, identity) == "matching_active"
    assert not (local_paths(runtime.root)["cpu_provisional"] / f"{identity.reservation_id}.json").exists()
    assert not release_executor_offer(runtime.root, identity)


def test_cpu_executor_offer_release_replays_exact_cross_directory_crash_image(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    initialize_cpu_lane_capacity(runtime.root, capacity=1)
    record = reserve_cpu(
        runtime.root,
        "task-cpu",
        1,
        attempt_id="task-cpu-attempt-1",
        fencing_token=1,
        project_id="project-a",
        shared_root="/shared/project-a/.qexp",
        machine_name="gpu-1",
        **_owner(),
    )["reservation"]
    identity = ReservationIdentity.from_record(record)
    released_record = {
        "reservation": {
            **record,
            "state": "released",
            "released_at": "2026-10-02T00:00:00Z",
            "release_reason": "definitive_no_claim",
        }
    }
    atomic_replace(
        local_paths(runtime.root)["cpu_released"] / f"{identity.reservation_id}.json",
        released_record,
    )

    assert classify_executor_offer(runtime.root, identity) == "conflict"
    assert release_executor_offer(runtime.root, identity, "definitive_no_claim")
    assert classify_executor_offer(runtime.root, identity) == "matching_released"
    assert not (local_paths(runtime.root)["cpu_provisional"] / f"{identity.reservation_id}.json").exists()


def test_legacy_reservation_identity_matching_remains_backward_compatible(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    record = reserve(
        runtime.root,
        "legacy-task",
        [0],
        attempt_id="legacy-attempt",
        fencing_token=1,
        project_id="project-a",
        shared_root="/shared/project-a/.qexp",
    )["reservation"]
    identity = ReservationIdentity(
        record["reservation_id"],
        record["acquisition_id"],
        "project-a",
        "legacy-task",
        "legacy-attempt",
        1,
        (0,),
        None,
    )

    assert identity.matches(record)
