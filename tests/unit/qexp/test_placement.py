from pathlib import Path

import pytest

from qqtools.plugins.qexp.domain.policies import group_allows, task_machine_matches
from qqtools.plugins.qexp.runtime.records import TaskRecord, TaskSpec, new_group, new_worker_member


def _task(tmp_path: Path, *, queue_scope: str, fallback: str | list[str]) -> TaskRecord:
    task = TaskRecord.new(
        task_id="task-1",
        machine="g3",
        spec=TaskSpec(["true"], str(tmp_path), 1),
        group_name="exp",
        sharing_mode="spillover",
        fallback_machines=fallback,
    )
    task.placement_runtime["queue_scope"] = queue_scope
    return task


def _group(*, dispatch_state: str = "active") -> dict:
    group = new_group("exp", "g3")
    group["group"]["dispatch_state"] = dispatch_state
    group["group"]["worker_set"] = {machine: new_worker_member() for machine in ("g2", "g3", "g4")}
    return group


def test_home_queue_allows_only_home_regardless_of_fallback(tmp_path: Path):
    task = _task(tmp_path, queue_scope="home", fallback=["g2"])

    assert task_machine_matches(task, "g3")
    assert not task_machine_matches(task, "g2")


def test_shared_explicit_fallback_adds_helpers_without_excluding_home(tmp_path: Path):
    task = _task(tmp_path, queue_scope="shared", fallback=["g2"])

    assert task_machine_matches(task, "g3")
    assert task_machine_matches(task, "g2")
    assert not task_machine_matches(task, "g4")


@pytest.mark.parametrize("machine", ["g2", "g3", "g4"])
def test_shared_group_fallback_allows_group_candidates(tmp_path: Path, machine: str):
    task = _task(tmp_path, queue_scope="shared", fallback="group")

    assert task_machine_matches(task, machine)
    assert group_allows(_group(), task, machine)


@pytest.mark.parametrize("machine", ["g2", "g3"])
def test_paused_group_rejects_home_and_helper(tmp_path: Path, machine: str):
    task = _task(tmp_path, queue_scope="shared", fallback=["g2"])

    assert not group_allows(_group(dispatch_state="paused"), task, machine)


@pytest.mark.parametrize("machine", ["g2", "g3"])
def test_inactive_worker_rejects_home_and_helper(tmp_path: Path, machine: str):
    task = _task(tmp_path, queue_scope="shared", fallback=["g2"])
    group = _group()
    group["group"]["worker_set"][machine]["state"] = "draining"

    assert not group_allows(group, task, machine)
