from __future__ import annotations

import time
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.machine_agent import dispatch_machine_cycle_locked
from qqtools.plugins.qexp.machine_runtime import MachineRuntime
from qqtools.plugins.qexp.runtime.paths import ready_state_path, shared_paths
from qqtools.plugins.qexp.runtime.ready import (
    delete_stale_ready_marker, load_ready_cursor, next_ready_marker,
)
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.work_budget import (
    AdaptiveBatchSizer, SliceBudget, WorkBudgetPolicy,
)
from qqtools.plugins.qexp.scheduler import run_dispatch_cycle


class _RecordingExecutor:
    def __init__(self) -> None:
        self.launched: list[str] = []

    def launch_attempt(self, _cfg, task_id, _attempt) -> None:
        self.launched.append(task_id)


def _activate_ready(cfg) -> None:
    value = read_json(ready_state_path(cfg.shared_root))
    value["ready_index"]["state"] = "active"
    value["ready_index"]["writer_capability"] = "ready-v1"
    atomic_replace(ready_state_path(cfg.shared_root), value)


def test_active_ready_dispatch_does_not_enumerate_task_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime"
    )
    task = submit(cfg, ["echo", "ready"], task_id="task-ready", working_dir=work)
    _activate_ready(cfg)
    executor = _RecordingExecutor()
    original_iter_json = __import__(
        "qqtools.plugins.qexp.scheduler", fromlist=["iter_json"]
    ).iter_json

    def reject_task_enumeration(directory):
        assert Path(directory) != shared_paths(cfg.shared_root)["tasks"]
        return original_iter_json(directory)

    monkeypatch.setattr(
        "qqtools.plugins.qexp.scheduler.iter_json", reject_task_enumeration
    )
    launched = run_dispatch_cycle(
        cfg,
        available_gpus=[0],
        executor=executor,
        should_recover_starting=False,
        work_budget=SliceBudget(WorkBudgetPolicy(soft_deadline_ms=60_000)),
    )
    assert launched == [task.task_id]
    assert executor.launched == [task.task_id]


def test_ready_cursor_advances_past_temporarily_unavailable_candidate(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime"
    )
    paused = submit(
        cfg, ["echo", "paused"], task_id="task-a", group="paused", working_dir=work
    )
    ready = submit(cfg, ["echo", "ready"], task_id="task-b", working_dir=work)
    group_path = shared_paths(cfg.shared_root)["groups"] / "paused.json"
    group = read_json(group_path)
    group["group"]["dispatch_state"] = "paused"
    atomic_replace(group_path, group)
    _activate_ready(cfg)

    launched = run_dispatch_cycle(
        cfg,
        available_gpus=[0],
        executor=_RecordingExecutor(),
        should_recover_starting=False,
        max_new_claims=1,
        work_budget=SliceBudget(WorkBudgetPolicy(soft_deadline_ms=60_000)),
    )

    assert launched == [ready.task_id]
    cursor = load_ready_cursor(cfg, "standalone", "home")
    assert cursor.after_name is None
    assert cursor.revision >= 2
    assert paused.task_id not in launched


def test_candidate_cursor_wraps_without_repeating_a_marker_in_one_slice(tmp_path: Path) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime"
    )
    task = submit(cfg, ["echo", "ready"], task_id="task-ready")
    _activate_ready(cfg)

    first, first_wrapped = next_ready_marker(cfg, "project-one", "home")
    end_of_partition, second_wrapped = next_ready_marker(cfg, "project-one", "home")
    repeated, third_wrapped = next_ready_marker(cfg, "project-one", "home")

    assert first is not None and first.identity == f"{task.task_id}.{task.ready_generation}"
    assert first_wrapped is False
    assert end_of_partition is None
    assert second_wrapped is True
    assert repeated is not None and repeated.identity == first.identity
    assert third_wrapped is False


def test_machine_cycle_fills_multiple_gpus_across_fair_rounds(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime"
    )
    tasks = [
        submit(cfg, ["echo", str(index)], task_id=f"task-{index}", working_dir=work)
        for index in range(3)
    ]
    _activate_ready(cfg)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    executor = _RecordingExecutor()

    results = dispatch_machine_cycle_locked(
        runtime,
        available_gpus=[0, 1, 2],
        executor=executor,
        supervise=False,
        publish_snapshots=False,
    )

    assert results == [{
        "project_id": binding.project_id,
        "launched": [task.task_id for task in tasks],
        "status": "dispatched",
    }]
    assert executor.launched == [task.task_id for task in tasks]


def test_degraded_ready_index_fails_closed_for_new_claims(tmp_path: Path) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime"
    )
    submit(cfg, ["echo", "ready"], task_id="task-ready")
    state = read_json(ready_state_path(cfg.shared_root))
    state["ready_index"]["state"] = "degraded"
    atomic_replace(ready_state_path(cfg.shared_root), state)

    assert run_dispatch_cycle(
        cfg, available_gpus=[0], should_recover_starting=False
    ) == []


def test_slow_ready_reads_shrink_the_process_local_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime"
    )
    submit(cfg, ["echo", "ready"], task_id="task-ready")
    _activate_ready(cfg)
    policy = WorkBudgetPolicy(soft_deadline_ms=50, initial_batch_size=4)
    budget = SliceBudget(policy)
    sizer = AdaptiveBatchSizer(policy)
    from qqtools.plugins.qexp import scheduler

    original_classify = scheduler.classify_ready_marker

    def slow_classify(*args, **kwargs):
        time.sleep(0.1)
        return original_classify(*args, **kwargs)

    monkeypatch.setattr(scheduler, "classify_ready_marker", slow_classify)

    run_dispatch_cycle(
        cfg,
        available_gpus=[],
        should_recover_starting=False,
        work_budget=budget,
        batch_sizer=sizer,
    )

    assert sizer.batch_size < policy.initial_batch_size


def test_full_capacity_machine_cycle_reads_no_ready_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime"
    )
    _activate_ready(cfg)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    from qqtools.plugins.qexp.runtime.reservations import attach, reserve

    reservation = reserve(
        runtime.root,
        "occupied",
        [0],
        attempt_id="occupied-attempt",
        fencing_token=1,
        project_id="external-project",
    )
    attach(runtime.root, reservation["reservation"]["reservation_id"], "occupied-attempt", 1)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.scheduler.next_ready_marker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ready candidates must not be read at full capacity")
        ),
    )

    assert dispatch_machine_cycle_locked(
        runtime, available_gpus=[0], supervise=False, publish_snapshots=False
    ) == [{"project_id": binding.project_id, "launched": [], "status": "dispatched"}]


def test_stale_generation_cleanup_cannot_delete_current_marker(tmp_path: Path) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime"
    )
    task = submit(cfg, ["echo", "ready"], task_id="task-ready")
    stale_reference, _ = next_ready_marker(cfg, "project-one", "home")
    assert stale_reference is not None
    task_path = shared_paths(cfg.shared_root)["tasks"] / f"{task.task_id}.json"
    value = read_json(task_path)
    value["task"]["ready_generation"] = task.ready_generation + 1
    atomic_replace(task_path, value)

    assert delete_stale_ready_marker(cfg, stale_reference) is True
    assert not (
        shared_paths(cfg.shared_root)["ready_reservations"]
        / f"{task.task_id}.{task.ready_generation}.json"
    ).exists()
    assert read_json(task_path)["task"]["ready_generation"] == task.ready_generation + 1


def test_project_round_robin_fairness_survives_dynamic_binding(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    projects = []
    for name in ("first", "second"):
        cfg = init_shared_root(
            tmp_path / name / ".qexp",
            "gpu-1",
            runtime_root=tmp_path / f"{name}-runtime",
        )
        task = submit(
            cfg, ["echo", name], task_id=f"task-{name}", working_dir=work
        )
        _activate_ready(cfg)
        binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
        projects.append((cfg, task, binding))
    runtime.save_cursor(projects[1][2].project_id)
    executor = _RecordingExecutor()

    results = dispatch_machine_cycle_locked(
        runtime,
        available_gpus=[0, 1],
        executor=executor,
        supervise=False,
        publish_snapshots=False,
    )

    result_by_project = {item["project_id"]: item for item in results}
    assert executor.launched == [projects[1][1].task_id, projects[0][1].task_id]
    assert all(
        result_by_project[binding.project_id]["launched"] == [task.task_id]
        for _cfg, task, binding in projects
    )
    assert runtime.load_cursor() == projects[1][2].project_id
