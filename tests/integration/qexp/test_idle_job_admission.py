"""Idle Job occupancy preempts before GPU or CPU reservation admission."""

import uuid

import pytest

from qqtools.plugins.qexp.runtime.resources.cpu_lane import reserve_cpu, set_cpu_lane_capacity
from qqtools.plugins.qexp.runtime.resources.idle_job import (
    IdleJobAdmissionBlocked,
    clear_idle_job_ownership,
    idle_job_guard,
    idle_job_instance_path,
    read_idle_job_demand,
    read_idle_job_ownership,
    write_idle_job_ownership,
)
from qqtools.plugins.qexp.runtime.resources.reservations import reservation_snapshot, reserve
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _occupy(root):
    record = {
        "version": 1,
        "runtime_id": "machine-runtime",
        "instance_id": uuid.uuid4().hex,
        "boot_id": str(uuid.uuid4()),
        "phase": "running",
        "config_revision": 1,
        "gpu_ids": [0],
        "parent_pid": 123,
        "parent_start_ticks": 1,
        "created_at": "2026-11-04T00:00:00Z",
    }
    with idle_job_guard(root):
        write_idle_job_ownership(root, record)
    return record


@pytest.mark.parametrize("lane", ["same_gpu", "different_gpu", "cpu"])
def test_training_waits_for_entire_idle_job_cleanup(tmp_path, lane):
    record = _occupy(tmp_path)
    set_cpu_lane_capacity(tmp_path, capacity=1)

    def admit():
        if lane == "cpu":
            return reserve_cpu(tmp_path, "training", 1)
        return reserve(tmp_path, "training", [0 if lane == "same_gpu" else 3])

    with pytest.raises(IdleJobAdmissionBlocked):
        admit()
    assert not reservation_snapshot(tmp_path).reservations
    assert read_idle_job_ownership(tmp_path)["phase"] == "stopping"
    assert read_idle_job_demand(tmp_path) == 1
    assert (
        read_json(idle_job_instance_path(tmp_path, record["instance_id"]) / "stop.json")["reason"] == "training_demand"
    )
    with idle_job_guard(tmp_path):
        clear_idle_job_ownership(tmp_path, record["instance_id"])
    assert admit()["reservation"]["task_id"] == "training"


def test_ineligible_cpu_work_does_not_interrupt_idle_timer(tmp_path):
    _occupy(tmp_path)
    set_cpu_lane_capacity(tmp_path, capacity=0)
    with pytest.raises(ValueError, match="insufficient"):
        reserve_cpu(tmp_path, "cannot-run", 1)
    assert read_idle_job_demand(tmp_path) == 0
    assert read_idle_job_ownership(tmp_path)["phase"] == "running"


def test_corrupt_ownership_blocks_admission_without_releasing_it(tmp_path):
    path = tmp_path / "idle-job" / "ownership.json"
    atomic_replace(path, {"idle_job": {"instance_id": "../foreign"}})
    with pytest.raises(ValueError):
        reserve(tmp_path, "training", [0])
    assert path.exists()
    assert not reservation_snapshot(tmp_path).reservations


def test_disabled_feature_does_not_create_idle_job_records(tmp_path):
    reserve(tmp_path, "training", [0])
    assert not (tmp_path / "idle-job").exists()


def test_machine_replacement_cannot_discard_uncertain_idle_job(tmp_path):
    from qqtools.plugins.qexp.agent.context import MachineRuntime
    from qqtools.plugins.qexp.agent.setup import SetupOperationalError, initialize_machine

    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "test-machine")
    record = _occupy(runtime.root)
    before = runtime.instance_id
    with pytest.raises(SetupOperationalError, match="idle_job"):
        initialize_machine(runtime, "replacement", detach_old_runtime=True, confirmed=True)
    assert runtime.instance_id == before
    assert read_idle_job_ownership(runtime.root) == record
