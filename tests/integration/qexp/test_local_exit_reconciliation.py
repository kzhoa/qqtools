from __future__ import annotations

from pathlib import Path

import pytest

from qqtools.plugins.qexp.agent import control_plane
from qqtools.plugins.qexp.agent.bindings import ProjectBinding
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.runtime.paths import local_paths
from qqtools.plugins.qexp.runtime.resources.reservations import attach, reserve
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


@pytest.mark.parametrize("blocker", [None, "project", "fencing", "missing_exit", "wrong_exit", "unknown_group"])
def test_control_plane_local_exit_reconciliation_never_constructs_shared_authority(tmp_path, monkeypatch, blocker):
    runtime = MachineRuntime(tmp_path / "machine")
    shared_root = tmp_path / "unavailable-shared-root"
    binding = ProjectBinding("project-a", shared_root, "gpu-1", _canonical_paths=True)
    paths = local_paths(runtime.project_paths(binding.project_id)["root"])
    process = {
        "task_id": "task-a",
        "attempt_id": "task-a-attempt-1",
        "fencing_token": 1,
        "process_group_id": 2_000_000_000,
        "process_group_start_time_ticks": 1,
    }
    if blocker == "unknown_group":
        process["process_group_start_time_ticks"] = None
    registration_path = paths["registrations"] / "task-a-attempt-1.json"
    atomic_replace(registration_path, {"process_registration": process})
    observation_path = paths["observations"] / "task-a-attempt-1.json"
    observation = {
        "exit_observation": {
            "protocol_version": 1,
            "task_id": "task-a",
            "attempt_id": "wrong-attempt" if blocker == "wrong_exit" else "task-a-attempt-1",
            "observed_exit_code": 0,
        }
    }
    if blocker != "missing_exit":
        atomic_replace(observation_path, observation)
    reservation = reserve(
        runtime.root,
        "task-a",
        [0],
        attempt_id="task-a-attempt-1",
        fencing_token=2 if blocker == "fencing" else 1,
        project_id="other-project" if blocker == "project" else binding.project_id,
    )["reservation"]
    attach(runtime.root, reservation["reservation_id"], "task-a-attempt-1", 2 if blocker == "fencing" else 1)
    active_path = local_paths(runtime.root)["active"] / f"{reservation['reservation_id']}.json"
    before = read_json(active_path)

    def reject_shared_constructor(*_args, **_kwargs):
        pytest.fail("local exit reconciliation constructed shared configuration/authority")

    original_resolve = Path.resolve

    def guarded_resolve(path, *args, **kwargs):
        assert path != shared_root and shared_root not in path.parents, "resolved unavailable shared path"
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(control_plane, "RootConfig", reject_shared_constructor)
    monkeypatch.setattr(control_plane, "_AuthoritySupervisor", reject_shared_constructor)
    monkeypatch.setattr(Path, "resolve", guarded_resolve)
    plane = control_plane._MachineControlPlane(
        runtime,
        instance_id="test",
        loop_interval=5.0,
        started_at="2026-09-29T00:00:00Z",
        available_gpus=[0],
    )
    sample = {"phase_seconds": {}}
    try:
        for _ in range(3):
            plane._reconcile_local_exits(binding, sample)
        assert sample["local_reconciliation"] == "returned"
        if blocker is None:
            assert not active_path.exists()
        else:
            assert read_json(active_path) == before
        # Local capacity convergence does not discard later shared-recovery proof.
        assert read_json(registration_path) == {"process_registration": process}
        if blocker != "missing_exit":
            assert read_json(observation_path) == observation
        assert not shared_root.exists()
    finally:
        for reconciler in plane._outage_supervisors.values():
            reconciler.close()
