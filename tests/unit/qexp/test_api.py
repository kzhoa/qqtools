import os
import sys
import time

import pytest

from qqtools.plugins.qexp import api as qexp_api
from qqtools.plugins.qexp import cancel, submit
from qqtools.plugins.qexp import fsqueue
from qqtools.plugins.qexp import manager
from qqtools.plugins.qexp.models import qExpTask
from qqtools.plugins.qexp import observer


def test_submit_bootstraps_layout_and_persists_pending_task(tmp_path, monkeypatch):
    root = tmp_path / "submit-home"
    monkeypatch.setenv(fsqueue.QQTOOLS_HOME_ENV, str(root))
    monkeypatch.setattr(manager, "run_preflight_checks", lambda: object())
    monkeypatch.setattr(manager, "is_daemon_active", lambda _root=None: True)

    task = submit(argv=["python", "train.py"], num_gpus=1, job_name="demo")

    assert task.status == "pending"
    assert root.joinpath("jobs", "pending", f"{task.task_id}.json").is_file()


def test_submit_is_idempotent_for_explicit_job_id(tmp_path, monkeypatch):
    root = tmp_path / "submit-home"
    monkeypatch.setenv(fsqueue.QQTOOLS_HOME_ENV, str(root))
    monkeypatch.setattr(manager, "run_preflight_checks", lambda: object())
    monkeypatch.setattr(manager, "is_daemon_active", lambda _root=None: True)

    first = submit(argv=["python", "train.py"], num_gpus=1, job_id="job_same")
    second = submit(argv=["python", "different.py"], num_gpus=2, job_id="job_same")

    assert first.task_id == second.task_id == "job_same"
    assert first.argv == second.argv == ["python", "train.py"]


def test_import_qqtools_and_qexp_do_not_eager_import_optional_runtime_deps():
    for module_name in ("qqtools", "qqtools.plugins", "qqtools.plugins.qexp"):
        sys.modules.pop(module_name, None)
    for module_name in ("libtmux", "psutil", "pynvml"):
        sys.modules.pop(module_name, None)

    import qqtools

    assert "libtmux" not in sys.modules
    assert "psutil" not in sys.modules
    assert "pynvml" not in sys.modules


def test_submit_reports_daemon_start_failure_after_queueing(tmp_path, monkeypatch):
    root = tmp_path / "submit-home"
    monkeypatch.setenv(fsqueue.QQTOOLS_HOME_ENV, str(root))
    monkeypatch.setattr(manager, "run_preflight_checks", lambda: object())

    daemon_states = iter([False, False])
    monkeypatch.setattr(manager, "is_daemon_active", lambda _root=None: next(daemon_states))
    monkeypatch.setattr(manager, "start_daemon_background", lambda _root=None: object())
    monkeypatch.setattr(manager, "DEFAULT_STARTUP_WAIT_SECONDS", 0)

    with pytest.raises(RuntimeError, match="daemon startup failed"):
        submit(argv=["python", "train.py"], num_gpus=1, job_id="job_fail")

    assert root.joinpath("jobs", "pending", "job_fail.json").is_file()


def test_submit_rejects_path_traversal_job_id(tmp_path, monkeypatch):
    root = tmp_path / "submit-home"
    monkeypatch.setenv(fsqueue.QQTOOLS_HOME_ENV, str(root))
    monkeypatch.setattr(manager, "run_preflight_checks", lambda: object())
    monkeypatch.setattr(manager, "is_daemon_active", lambda _root=None: True)

    with pytest.raises(ValueError, match="illegal path characters|illegal characters"):
        submit(argv=["python", "train.py"], num_gpus=1, job_id="../../../tmp/pwn")


def test_get_logs_path_and_read_logs_reject_path_traversal_task_id(tmp_path):
    root = tmp_path / "logs-home"

    with pytest.raises(ValueError, match="illegal path characters|illegal characters"):
        qexp_api.get_logs_path("../../../tmp/pwn", root=root)

    with pytest.raises(ValueError, match="illegal path characters|illegal characters"):
        qexp_api.read_logs("../../../tmp/pwn", root=root)


def test_cancel_moves_pending_task_to_cancelled(tmp_path, monkeypatch):
    root = tmp_path / "cancel-home"
    monkeypatch.setenv(fsqueue.QQTOOLS_HOME_ENV, str(root))
    task = qExpTask(task_id="job_pending", argv=["python", "train.py"], num_gpus=1)
    fsqueue.save_task(task, root)

    cancelled = cancel("job_pending", root=root)

    assert cancelled.status == "cancelled"
    assert cancelled.exit_reason == "cancelled_before_start"


def test_cancel_running_task_signals_process_group_and_escalates(tmp_path, monkeypatch):
    root = tmp_path / "cancel-home"
    monkeypatch.setenv(fsqueue.QQTOOLS_HOME_ENV, str(root))
    task = qExpTask(
        task_id="job_running",
        argv=["python", "train.py"],
        num_gpus=1,
        status="running",
        assigned_gpus=[0],
        tmux_session="experiments",
        tmux_window_id="@job_running",
        process_group_id=1234,
    )
    fsqueue.save_task(task, root)

    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(qexp_api.os, "killpg", lambda pgid, sig: signals.append((pgid, sig)))
    monkeypatch.setattr(qexp_api, "_is_process_group_alive", lambda _pgid: True)
    monkeypatch.setattr(qexp_api.time, "sleep", lambda _seconds: None)
    time_values = iter([0.0, 0.05, 0.1, 0.25])
    monkeypatch.setattr(qexp_api.time, "time", lambda: next(time_values))

    current = cancel("job_running", root=root, grace_seconds=0.2, poll_interval_seconds=0.01)

    assert current.status == "running"
    assert signals[0][0] == 1234
    assert len(signals) == 2


def test_get_status_snapshot_reports_stale_daemon_and_recent_events(tmp_path, monkeypatch):
    root = tmp_path / "status-home"
    monkeypatch.setenv(fsqueue.QQTOOLS_HOME_ENV, str(root))

    pending = qExpTask(task_id="job_pending", argv=["python", "p.py"], num_gpus=2)
    running = qExpTask(
        task_id="job_running",
        argv=["python", "r.py"],
        num_gpus=1,
        status="running",
        assigned_gpus=[0],
        scheduled_at="2026-04-02T09:00:00Z",
        started_at="2026-04-02T09:00:10Z",
    )
    failed = qExpTask(
        task_id="job_failed",
        argv=["python", "f.py"],
        num_gpus=1,
        status="failed",
        assigned_gpus=[1],
        finished_at="2026-04-02T09:05:00Z",
        exit_reason="wrapper_crashed",
    )
    fsqueue.save_task(pending, root)
    fsqueue.save_task(running, root)
    failed_path = fsqueue.save_task(failed, root)
    log_path = fsqueue.get_log_path("job_failed", root)
    log_path.write_text("failed\n", encoding="utf-8")

    daemon_lock = manager.qExpDaemonLock(root)
    assert daemon_lock.acquire() is True
    daemon_lock.write_metadata({"pid": 1234, "started_at": "2026-04-02T09:00:00Z"})

    monkeypatch.setattr(manager, "_is_process_alive", lambda pid: pid == 1234)
    monkeypatch.setattr(manager.time, "time", lambda: failed_path.stat().st_mtime + 1000)
    monkeypatch.setattr(observer.platform, "node", lambda: "worker-a")
    monkeypatch.setattr(observer.platform, "platform", lambda: "Linux-x86_64")
    monkeypatch.setattr(observer, "probe_gpu_backend", lambda: ("stub", [0, 1]))

    snapshot = qexp_api.get_status_snapshot(root=root)

    daemon_lock.release()

    assert snapshot["daemon"]["state"] == "STALE (Unresponsive)"
    assert snapshot["counts"] == {
        "pending": 1,
        "running": 1,
        "done": 0,
        "failed": 1,
        "cancelled": 0,
    }
    task_map = {item["task_id"]: item for item in snapshot["tasks"]}
    assert task_map["job_running"]["state"] == "Running"
    assert snapshot["pending_preview"][0]["task_id"] == "job_pending"
    assert snapshot["gpus"]["slots"][0]["task_id"] == "job_running"
    assert snapshot["events"][0]["task_id"] == "job_failed"
    assert snapshot["warnings"] == []
    assert snapshot["summary"]["invalid_task_files"] == 0
    assert "env" not in task_map["job_running"]["task"]


def test_status_snapshot_does_not_create_pid_file_on_fresh_root(tmp_path, monkeypatch):
    root = tmp_path / "readonly-status-home"
    monkeypatch.setenv(fsqueue.QQTOOLS_HOME_ENV, str(root))
    monkeypatch.setattr(observer.platform, "node", lambda: "worker-a")
    monkeypatch.setattr(observer.platform, "platform", lambda: "Linux-x86_64")
    monkeypatch.setattr(observer, "probe_gpu_backend", lambda: (None, []))

    snapshot = qexp_api.get_status_snapshot(root=root)

    assert snapshot["daemon"]["state"] == "STOPPED"
    assert not root.joinpath("daemon.pid").exists()


def test_status_snapshot_skips_bad_task_files_with_warning(tmp_path, monkeypatch):
    root = tmp_path / "bad-task-home"
    monkeypatch.setenv(fsqueue.QQTOOLS_HOME_ENV, str(root))
    fsqueue.ensure_qexp_layout(root)
    broken_path = root.joinpath("jobs", "running", "job_broken.json")
    broken_path.write_text('{"task_id":"job_broken","status":"pending"}', encoding="utf-8")

    monkeypatch.setattr(observer.platform, "node", lambda: "worker-a")
    monkeypatch.setattr(observer.platform, "platform", lambda: "Linux-x86_64")
    monkeypatch.setattr(observer, "probe_gpu_backend", lambda: ("stub", [0]))

    snapshot = qexp_api.get_status_snapshot(root=root)

    assert snapshot["tasks"] == []
    assert snapshot["events"] == []
    assert snapshot["counts"] == {
        "pending": 0,
        "running": 0,
        "done": 0,
        "failed": 0,
        "cancelled": 0,
    }
    assert len(snapshot["warnings"]) == 1
    assert snapshot["summary"]["invalid_task_files"] == 1
    assert snapshot["gpus"]["slots"][0]["state"] == "Free"


def test_status_snapshot_masks_env_by_omission(tmp_path, monkeypatch):
    root = tmp_path / "env-home"
    monkeypatch.setenv(fsqueue.QQTOOLS_HOME_ENV, str(root))
    task = qExpTask(
        task_id="job_secret",
        argv=["python", "secret.py"],
        num_gpus=1,
        env={"kind": "none", "extra_env": {"API_KEY": "secret-value"}},
    )
    fsqueue.save_task(task, root)

    monkeypatch.setattr(observer.platform, "node", lambda: "worker-a")
    monkeypatch.setattr(observer.platform, "platform", lambda: "Linux-x86_64")
    monkeypatch.setattr(observer, "probe_gpu_backend", lambda: (None, []))

    snapshot = qexp_api.get_status_snapshot(root=root)

    assert snapshot["tasks"][0]["task"]["task_id"] == "job_secret"
    assert "env" not in snapshot["tasks"][0]["task"]


def test_status_snapshot_tolerates_task_file_removed_during_observation(tmp_path, monkeypatch):
    root = tmp_path / "vanish-home"
    monkeypatch.setenv(fsqueue.QQTOOLS_HOME_ENV, str(root))
    fsqueue.ensure_qexp_layout(root)
    path = root.joinpath("jobs", "running", "job_vanish.json")
    path.write_text("{}", encoding="utf-8")

    original_load_task = observer.fsqueue.load_task

    def _vanishing_load_task(target_path):
        if target_path == path:
            path.unlink(missing_ok=True)
            raise FileNotFoundError("task file vanished during observation")
        return original_load_task(target_path)

    monkeypatch.setattr(observer.fsqueue, "load_task", _vanishing_load_task)
    monkeypatch.setattr(observer.platform, "node", lambda: "worker-a")
    monkeypatch.setattr(observer.platform, "platform", lambda: "Linux-x86_64")
    monkeypatch.setattr(observer, "probe_gpu_backend", lambda: ("stub", [0]))

    snapshot = qexp_api.get_status_snapshot(root=root)

    assert snapshot["tasks"] == []
    assert snapshot["events"] == []
    assert snapshot["warnings"] == [
        f"Skipped unreadable task file '{path}': task file vanished during observation"
    ]


def test_clean_keeps_recent_done_and_cancelled_by_default(tmp_path, monkeypatch):
    root = tmp_path / "clean-home"
    monkeypatch.setenv(fsqueue.QQTOOLS_HOME_ENV, str(root))
    for state in ("done", "cancelled", "failed"):
        task = qExpTask(
            task_id=f"job_{state}",
            argv=["python", f"{state}.py"],
            num_gpus=1,
            status=state,
        )
        fsqueue.save_task(task, root)
        fsqueue.get_log_path(task.task_id, root).write_text(f"{state}\n", encoding="utf-8")

    dry_run = qexp_api.clean(root=root, dry_run=True)
    assert dry_run["task_ids"] == []
    assert fsqueue.load_task_by_id("job_done", root) is not None
    assert fsqueue.get_log_path("job_done", root).exists()

    result = qexp_api.clean(root=root)

    assert result["task_ids"] == []
    assert fsqueue.load_task_by_id("job_done", root) is not None
    assert fsqueue.load_task_by_id("job_cancelled", root) is not None
    assert fsqueue.load_task_by_id("job_failed", root) is not None
    assert fsqueue.get_log_path("job_done", root).exists()
    assert fsqueue.get_log_path("job_cancelled", root).exists()
    assert fsqueue.get_log_path("job_failed", root).exists()


def test_clean_removes_only_old_done_and_cancelled_logs(tmp_path, monkeypatch):
    root = tmp_path / "clean-old-home"
    monkeypatch.setenv(fsqueue.QQTOOLS_HOME_ENV, str(root))
    for state in ("done", "cancelled", "failed"):
        task = qExpTask(
            task_id=f"job_{state}",
            argv=["python", f"{state}.py"],
            num_gpus=1,
            status=state,
        )
        path = fsqueue.save_task(task, root)
        fsqueue.get_log_path(task.task_id, root).write_text(f"{state}\n", encoding="utf-8")
        old_time = time.time() - (qexp_api.DEFAULT_CLEAN_OLDER_THAN_SECONDS + 60)
        if state != "failed":
            os.utime(path, (old_time, old_time))
            log_path = fsqueue.get_log_path(task.task_id, root)
            os.utime(log_path, (old_time, old_time))

    result = qexp_api.clean(root=root)

    assert sorted(result["task_ids"]) == ["job_cancelled", "job_done"]
    assert fsqueue.load_task_by_id("job_done", root) is None
    assert fsqueue.load_task_by_id("job_cancelled", root) is None
    assert fsqueue.load_task_by_id("job_failed", root) is not None
    assert not fsqueue.get_log_path("job_done", root).exists()
    assert not fsqueue.get_log_path("job_cancelled", root).exists()
    assert fsqueue.get_log_path("job_failed", root).exists()
