from pathlib import Path

import pytest

import qqtools.plugins.qexp as qexp
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.runtime.records import TaskSpec
from qqtools.plugins.qexp.runtime.resources.cpu_lane import (
    CpuLanePolicy,
    attach_cpu,
    cpu_reservation_snapshot,
    get_cpu_lane_policy,
    initialize_cpu_lane_capacity,
    release_cpu,
    reserve_cpu,
    set_cpu_lane_capacity,
)


def test_cpu_lane_public_api_reexports_resource_owner_objects():
    assert qexp.CpuLanePolicy is CpuLanePolicy
    assert qexp.get_cpu_lane_policy is get_cpu_lane_policy
    assert qexp.set_cpu_lane_capacity is set_cpu_lane_capacity


def test_cpu_task_spec_omits_gpu_request_from_canonical_record():
    spec = TaskSpec(["echo", "ok"], "/tmp", 0, 2, "cpu")

    assert spec.is_cpu_only
    assert spec.to_dict() == {
        "command": ["echo", "ok"],
        "working_directory": "/tmp",
        "lane": "cpu",
        "requested_cpus": 2,
    }


def test_cpu_lane_capacity_serializes_reservations(tmp_path: Path):
    runtime = tmp_path / "runtime"
    policy = set_cpu_lane_capacity(runtime, capacity=2)
    reservation = reserve_cpu(runtime, "task-1", 2, attempt_id="attempt-1", fencing_token=1)

    assert policy.capacity == 2
    with pytest.raises(ValueError, match="insufficient free slots"):
        reserve_cpu(runtime, "task-2", 1)
    with pytest.raises(ValueError, match="below reserved CPU slots"):
        set_cpu_lane_capacity(runtime, capacity=1)

    attach_cpu(runtime, reservation["reservation"]["reservation_id"], "attempt-1", 1)
    policy, reservations = cpu_reservation_snapshot(runtime)
    assert policy.capacity == 2
    assert reservations[0]["state"] == "active"
    assert release_cpu(runtime, reservation["reservation"]["reservation_id"])
    assert get_cpu_lane_policy(runtime).capacity == 2
    assert set_cpu_lane_capacity(runtime, capacity=0).capacity == 0


def test_cpu_borrow_admission_survives_reservation_attachment(tmp_path: Path):
    runtime = tmp_path / "runtime"
    set_cpu_lane_capacity(runtime, capacity=2)
    value = reserve_cpu(
        runtime,
        "task-1",
        1,
        attempt_id="attempt-1",
        fencing_token=1,
        group_name="group-1",
        admitted_as_borrow=True,
        worker_scheduling_role="borrow",
        group_worker_set_epoch=3,
        worker_state_epoch=4,
    )
    attach_cpu(runtime, value["reservation"]["reservation_id"], "attempt-1", 1)
    _policy, reservations = cpu_reservation_snapshot(runtime)
    assert reservations[0]["admission"] == {
        "admitted_as_borrow": True,
        "worker_scheduling_role": "borrow",
        "group_worker_set_epoch": 3,
        "worker_state_epoch": 4,
        "gpu_limit_gpus": None,
    }


@pytest.mark.parametrize(
    "metadata",
    [
        {"admitted_as_borrow": 1},
        {"worker_state_epoch": True},
        {"group_worker_set_epoch": -1},
        {"admitted_as_borrow": True, "worker_scheduling_role": "primary", "group_name": "group-1"},
        {"admitted_as_borrow": True, "worker_scheduling_role": "borrow"},
    ],
)
def test_cpu_lane_rejects_invalid_admission_metadata(tmp_path: Path, metadata):
    with pytest.raises(ValueError):
        reserve_cpu(tmp_path, "task-1", 1, **metadata)


def test_cpu_lane_rejects_invalid_resource_requests():
    with pytest.raises(ValueError, match="requested_cpus"):
        TaskSpec(["echo"], "/tmp", 0, None, "cpu")
    with pytest.raises(ValueError, match="cannot contain requested_cpus"):
        TaskSpec(["echo"], "/tmp", 1, 1, "gpu")


def test_cpu_lane_public_policy_api_accepts_machine_runtime(tmp_path: Path):
    runtime = MachineRuntime(tmp_path / "machine")
    runtime.ensure_layout()

    assert set_cpu_lane_capacity(runtime, capacity=3).capacity == 3
    assert get_cpu_lane_policy(runtime).capacity == 3


def test_cpu_lane_initialization_rejects_a_conflicting_shared_capacity(tmp_path: Path):
    runtime = tmp_path / "machine"
    assert initialize_cpu_lane_capacity(runtime, capacity=4).capacity == 4
    assert initialize_cpu_lane_capacity(runtime, capacity=4).capacity == 4
    with pytest.raises(ValueError, match="already configured as 4"):
        initialize_cpu_lane_capacity(runtime, capacity=8)
