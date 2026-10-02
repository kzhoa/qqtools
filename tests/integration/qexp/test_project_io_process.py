from __future__ import annotations

import os
from pathlib import Path

import pytest

from qqtools.plugins.qexp.agent.context import MachineRuntime, ProjectBinding
from qqtools.plugins.qexp.agent.project_io_process import binding_signature, read_exit_code, read_process_manifest
from qqtools.plugins.qexp.runtime.store import atomic_replace

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _local_attempt(tmp_path: Path):
    runtime = MachineRuntime(tmp_path / "machine")
    # A FIFO stands in for inaccessible Project storage. Local evidence parsing
    # must never open or inspect Project truth to classify this exit.
    shared_root = tmp_path / "unavailable-project"
    os.mkfifo(shared_root)
    binding = ProjectBinding("project-a", shared_root, "gpu-1", _canonical_paths=True)
    paths = runtime.project_paths(binding.project_id)
    process = {
        "task_id": "task-a",
        "attempt_id": "task-a-attempt-1",
        "fencing_token": 3,
        "machine_name": "gpu-1",
        "observed_state": "running",
        "wrapper_pid": None,
        "wrapper_start_time_ticks": None,
        "process_group_id": None,
        "process_group_start_time_ticks": None,
    }
    path = paths["processes"] / "task-a-attempt-1.json"
    atomic_replace(path, {"process": process})
    entry = read_process_manifest(runtime, binding, path, binding_signature(binding, 7))
    assert entry is not None
    assert entry["process"] == process
    return runtime, binding, paths, entry


@pytest.mark.parametrize("exit_code", [0, 1, -9])
@pytest.mark.parametrize("has_task_id", [False, True])
def test_local_terminal_evidence_accepts_exact_integer_exit(tmp_path, exit_code, has_task_id):
    runtime, binding, paths, entry = _local_attempt(tmp_path)
    observation = {"attempt_id": "task-a-attempt-1", "observed_exit_code": exit_code}
    if has_task_id:
        observation["task_id"] = "task-a"
    atomic_replace(paths["observations"] / "task-a-attempt-1.json", {"exit_observation": observation})
    assert read_exit_code(runtime, binding, entry) == exit_code


@pytest.mark.parametrize(
    "changes",
    [
        {"observed_exit_code": True},
        {"observed_exit_code": None},
        {"observed_exit_code": "0"},
        {"protocol_version": 2},
        {"attempt_id": "task-a-attempt-2"},
        {"task_id": "task-b"},
    ],
)
def test_local_terminal_evidence_rejects_invalid_observation(tmp_path, changes):
    runtime, binding, paths, entry = _local_attempt(tmp_path)
    observation = {"attempt_id": "task-a-attempt-1", "observed_exit_code": 0, **changes}
    atomic_replace(paths["observations"] / "task-a-attempt-1.json", {"exit_observation": observation})
    assert read_exit_code(runtime, binding, entry) is None


@pytest.mark.parametrize("case", ["missing", "fifo", "symlink", "oversized"])
def test_local_terminal_evidence_does_not_follow_unusable_files(tmp_path, case):
    runtime, binding, paths, entry = _local_attempt(tmp_path)
    path = paths["observations"] / "task-a-attempt-1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if case == "fifo":
        os.mkfifo(path)
    elif case == "symlink":
        path.symlink_to(binding.shared_root)
    elif case == "oversized":
        atomic_replace(
            path,
            {"exit_observation": {"attempt_id": "task-a-attempt-1", "observed_exit_code": 0, "extra": "x" * 65536}},
        )
    assert read_exit_code(runtime, binding, entry) is None
