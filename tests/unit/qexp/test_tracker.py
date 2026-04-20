from qqtools.plugins.qexp.tracker import Tracker


def test_tracker_allocate_and_release_follow_in_memory_reservations():
    tracker = Tracker(gpu_probe=lambda: ("stub", [0, 1, 2]))
    tracker.refresh_visibility()

    assert tracker.allocate("task_a", 2) == [0, 1]
    assert tracker.get_allocatable_gpu_ids() == [2]

    tracker.release("task_a")

    assert tracker.reserved_gpu_ids == set()
    assert tracker.get_allocatable_gpu_ids() == [0, 1, 2]


def test_tracker_refresh_visibility_updates_backend_and_visible_ids():
    tracker = Tracker(gpu_probe=lambda: ("stub", [2, 3]))

    tracker.refresh_visibility()

    assert tracker.backend_name == "stub"
    assert tracker.visible_gpu_ids == [2, 3]


def test_tracker_allocate_returns_none_when_capacity_is_insufficient():
    tracker = Tracker(gpu_probe=lambda: ("stub", [0]))
    tracker.refresh_visibility()

    assert tracker.allocate("task_a", 2) is None
    assert tracker.reserved_gpu_ids == set()
