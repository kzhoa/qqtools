import sys

import pytest

from qqtools.plugins.qexp import api as qexp_api
from qqtools.plugins.qexp import cancel, submit
from qqtools.plugins.qexp import fsqueue
from qqtools.plugins.qexp import manager
from qqtools.plugins.qexp.models import qExpTask


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
