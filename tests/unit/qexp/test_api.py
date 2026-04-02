import sys

import pytest

from qqtools.plugins.qexp import submit
from qqtools.plugins.qexp import fsqueue
from qqtools.plugins.qexp import manager


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
