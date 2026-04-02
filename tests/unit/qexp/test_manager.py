import multiprocessing
import time

import pytest

from qqtools.plugins.qexp import fsqueue
from qqtools.plugins.qexp import manager
from qqtools.plugins.qexp.models import qExpTask
from qqtools.plugins.qexp.tracker import qExpTracker


def _hold_lock(root: str, ready_queue):
    lock = manager.qExpDaemonLock(root)
    ready_queue.put(lock.acquire())
    time.sleep(2)
    lock.release()


def test_daemon_lock_allows_only_one_active_holder(tmp_path):
    root = tmp_path / "daemon-home"
    ready_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_hold_lock, args=(str(root), ready_queue))
    process.start()

    try:
        assert ready_queue.get(timeout=5) is True
        time.sleep(0.2)
        assert manager.has_active_daemon_lock(root) is True
    finally:
        process.join(timeout=5)


def test_preflight_checks_require_linux_tmux_libtmux_and_gpu(monkeypatch):
    monkeypatch.setattr(manager.platform, "system", lambda: "Linux")
    monkeypatch.setattr(manager.tmux, "is_tmux_executable_available", lambda: True)
    monkeypatch.setattr(manager.importlib, "import_module", lambda name: object())
    monkeypatch.setattr(manager, "probe_gpu_backend", lambda: ("stub", [0, 1]))

    result = manager.run_preflight_checks()

    assert result.gpu_backend == "stub"
    assert result.visible_gpu_ids == [0, 1]


def test_reconcile_marks_lost_running_task_failed(tmp_path, monkeypatch):
    root = tmp_path / "manager-home"
    monkeypatch.setenv(fsqueue.QQTOOLS_HOME_ENV, str(root))

    task = qExpTask(
        task_id="job_lost",
        argv=["python", "train.py"],
        num_gpus=1,
        status="running",
        assigned_gpus=[0],
        scheduled_at="2024-01-01T00:00:00Z",
        tmux_session="experiments",
        tmux_window_id="@dead",
    )
    fsqueue.save_task(task)

    tracker = qExpTracker(gpu_probe=lambda: ("stub", [0]))
    tracker.refresh(root)
    monkeypatch.setattr(manager.time, "time", lambda: 9999999999)

    failed = manager.reconcile_running_jobs(
        tracker=tracker,
        root=root,
        startup_grace_seconds=10,
        window_exists=lambda _window_id: True,
    )

    repaired_task = fsqueue.load_task_by_id("job_lost", root)
    assert failed == ["job_lost"]
    assert repaired_task.status == "failed"
    assert repaired_task.exit_reason == "wrapper_crashed"
    assert tracker.reserved_gpu_ids == set()


def test_get_idle_timeout_seconds_reads_env(monkeypatch):
    monkeypatch.setenv("QQTOOLS_DAEMON_IDLE_TIMEOUT", "42")
    assert manager.get_idle_timeout_seconds() == 42

def test_main_accepts_root_only(monkeypatch, tmp_path):
    expected_root = tmp_path / "daemon-root"
    captured = {}

    def _fake_run_daemon_foreground(root=None, **_kwargs):
        captured["root"] = root
        return 0

    monkeypatch.setattr(manager, "run_daemon_foreground", _fake_run_daemon_foreground)

    result = manager.main(["--root", str(expected_root)])

    assert result == 0
    assert captured["root"] == expected_root.resolve()


def test_main_accepts_foreground_flag_for_tmux_launcher_compatibility(monkeypatch, tmp_path):
    expected_root = tmp_path / "daemon-root"
    captured = {}

    def _fake_run_daemon_foreground(root=None, **_kwargs):
        captured["root"] = root
        return 0

    monkeypatch.setattr(manager, "run_daemon_foreground", _fake_run_daemon_foreground)

    result = manager.main(["--foreground", "--root", str(expected_root)])

    assert result == 0
    assert captured["root"] == expected_root.resolve()
