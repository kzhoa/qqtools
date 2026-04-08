from qqtools.plugins.qexp import fsqueue
from qqtools.plugins.qexp.models import qExpTask
from qqtools.plugins.qexp.tracker import qExpTracker


def test_tracker_refresh_rebuilds_reservations_from_running_tasks(tmp_path, monkeypatch):
    root = tmp_path / "tracker-home"
    monkeypatch.setenv(fsqueue.QQTOOLS_HOME_ENV, str(root))

    running_task = qExpTask(
        task_id="job_running",
        argv=["python", "train.py"],
        num_gpus=2,
        status="running",
        assigned_gpus=[2, 3],
        scheduled_at="2024-01-01T00:00:00Z",
        tmux_session="experiments",
        tmux_window_id="@1",
    )
    fsqueue.save_task(running_task)

    tracker = qExpTracker(gpu_probe=lambda: ("stub", [0, 1, 2, 3]))
    tracker.refresh(root)

    assert tracker.visible_gpu_ids == [0, 1, 2, 3]
    assert tracker.reserved_gpu_ids == {2, 3}
    assert tracker.task_id_to_gpu_ids == {"job_running": [2, 3]}


def test_tracker_allocate_and_release_follow_in_memory_reservations():
    tracker = qExpTracker(gpu_probe=lambda: ("stub", [0, 1, 2]))
    tracker.refresh(running_tasks=[])

    assert tracker.allocate("job_a", 2) == [0, 1]
    assert tracker.get_allocatable_gpu_ids() == [2]

    tracker.release("job_a")

    assert tracker.reserved_gpu_ids == set()
    assert tracker.get_allocatable_gpu_ids() == [0, 1, 2]


def test_tracker_rebuild_reservations_does_not_reprobe_visibility():
    calls = {"probe": 0}

    def _probe():
        calls["probe"] += 1
        return "stub", [0, 1]

    tracker = qExpTracker(gpu_probe=_probe)
    tracker.refresh_visibility()
    tracker.rebuild_reservations(
        running_tasks=[
            qExpTask(
                task_id="job_running",
                argv=["python", "train.py"],
                num_gpus=1,
                status="running",
                assigned_gpus=[1],
                scheduled_at="2024-01-01T00:00:00Z",
                tmux_session="experiments",
                tmux_window_id="@1",
            )
        ]
    )

    assert calls["probe"] == 1
    assert tracker.visible_gpu_ids == [0, 1]
    assert tracker.reserved_gpu_ids == {1}
