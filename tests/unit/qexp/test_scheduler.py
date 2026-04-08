from qqtools.plugins.qexp import fsqueue
from qqtools.plugins.qexp.executor import qExpExecutor
from qqtools.plugins.qexp.models import qExpTask
from qqtools.plugins.qexp.scheduler import qExpScheduler
from qqtools.plugins.qexp.tracker import qExpTracker


def test_scheduler_uses_fifo_with_backfill(tmp_path, monkeypatch):
    root = tmp_path / "scheduler-home"
    monkeypatch.setenv(fsqueue.QQTOOLS_HOME_ENV, str(root))

    first = qExpTask(
        task_id="job_big",
        argv=["python", "train_big.py"],
        num_gpus=2,
        created_at="2024-01-01T00:00:00Z",
    )
    second = qExpTask(
        task_id="job_small",
        argv=["python", "train_small.py"],
        num_gpus=1,
        created_at="2024-01-01T00:00:01Z",
    )
    fsqueue.save_task(first)
    fsqueue.save_task(second)

    tracker = qExpTracker(gpu_probe=lambda: ("stub", [0]))
    tracker.refresh(root, running_tasks=[])
    scheduler = qExpScheduler(
        tracker=tracker,
        create_window=lambda task_id, session_name: f"@{task_id}",
        executor=qExpExecutor(send_command=lambda _window_id, _command: None),
    )

    launched = scheduler.run_cycle(root)

    assert launched == ["job_small"]
    assert fsqueue.load_task_by_id("job_big", root).status == "pending"
    launched_task = fsqueue.load_task_by_id("job_small", root)
    assert launched_task.status == "running"
    assert launched_task.assigned_gpus == [0]
    assert launched_task.tmux_window_id == "@job_small"


def test_scheduler_kills_window_when_dispatch_races(tmp_path, monkeypatch):
    root = tmp_path / "scheduler-home"
    monkeypatch.setenv(fsqueue.QQTOOLS_HOME_ENV, str(root))

    task = qExpTask(
        task_id="job_race",
        argv=["python", "train.py"],
        num_gpus=1,
        created_at="2024-01-01T00:00:00Z",
    )
    fsqueue.save_task(task)

    tracker = qExpTracker(gpu_probe=lambda: ("stub", [0]))
    tracker.refresh(root, running_tasks=[])
    killed_windows: list[str | None] = []
    scheduler = qExpScheduler(
        tracker=tracker,
        create_window=lambda task_id, session_name: "@race",
        kill_window=lambda window_id: killed_windows.append(window_id),
        executor=qExpExecutor(send_command=lambda _window_id, _command: None),
    )
    original_dispatch = fsqueue.dispatch_task_to_running

    def _raise_dispatch_race(*args, **kwargs):
        raise FileNotFoundError("simulated cancel race")

    monkeypatch.setattr(fsqueue, "dispatch_task_to_running", _raise_dispatch_race)

    launched = scheduler.run_cycle(root)

    assert launched == []
    assert killed_windows == ["@race"]
    assert tracker.reserved_gpu_ids == set()
