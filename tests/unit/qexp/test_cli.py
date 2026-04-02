import io
from pathlib import Path

import pytest

from qqtools.plugins.qexp import cli
from qqtools.plugins.qexp.models import qExpTask


def test_submit_command_dispatches_to_api(monkeypatch, capsys, tmp_path):
    captured = {}

    def _fake_submit(**kwargs):
        captured.update(kwargs)
        return qExpTask(task_id="job_cli", argv=["python", "train.py"], num_gpus=1)

    monkeypatch.setattr(cli.api, "submit", _fake_submit)

    result = cli.main(
        [
            "--root",
            str(tmp_path),
            "submit",
            "--num-gpus",
            "1",
            "--job-id",
            "job_cli",
            "--job-name",
            "demo",
            "--env",
            "OMP_NUM_THREADS=8",
            "--",
            "python",
            "train.py",
        ]
    )

    assert result == 0
    assert captured["num_gpus"] == 1
    assert captured["job_id"] == "job_cli"
    assert captured["job_name"] == "demo"
    assert captured["root"] == tmp_path.resolve()
    assert captured["argv"] == ["python", "train.py"]
    assert captured["env"]["kind"] == "none"
    assert captured["env"]["extra_env"]["OMP_NUM_THREADS"] == "8"
    assert capsys.readouterr().out.strip() == "job_cli"


def test_submit_command_builds_conda_env(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        cli.api,
        "submit",
        lambda **kwargs: captured.update(kwargs) or qExpTask(task_id="job_conda", argv=["python"], num_gpus=1),
    )

    cli.main(
        [
            "submit",
            "--num-gpus",
            "1",
            "--conda-name",
            "torch",
            "--conda-activate-script",
            "/opt/conda.sh",
            "--",
            "python",
            "train.py",
        ]
    )

    assert captured["env"] == {
        "kind": "conda",
        "name": "torch",
        "activate_script": "/opt/conda.sh",
    }


def test_submit_requires_task_argv():
    with pytest.raises(ValueError, match="requires a task argv"):
        cli.main(["submit", "--num-gpus", "1"])


def test_daemon_background_starts_and_verifies(monkeypatch, capsys, tmp_path):
    calls = {"preflight": 0, "start": 0}
    monkeypatch.setattr(cli.manager, "run_preflight_checks", lambda: calls.__setitem__("preflight", calls["preflight"] + 1))
    monkeypatch.setattr(cli.manager, "start_daemon_background", lambda root=None: calls.__setitem__("start", calls["start"] + 1))
    monkeypatch.setattr(cli.manager, "is_daemon_active", lambda root=None: True)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    result = cli.main(["--root", str(tmp_path), "daemon", "--background"])

    assert result == 0
    assert calls == {"preflight": 1, "start": 1}
    assert "started in background" in capsys.readouterr().out


def test_daemon_foreground_calls_manager(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.manager, "run_daemon_foreground", lambda root=None: 7)

    result = cli.main(["--root", str(tmp_path), "daemon"])

    assert result == 7


def test_daemon_rejects_conflicting_mode_flags():
    with pytest.raises(ValueError, match="Choose either --background or --foreground"):
        cli.main(["daemon", "--background", "--foreground"])


def test_cancel_command_prints_task_state(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        cli.api,
        "cancel",
        lambda task_id, root=None: qExpTask(
            task_id=task_id,
            argv=["python", "train.py"],
            num_gpus=1,
            status="cancelled",
        ),
    )

    result = cli.main(["--root", str(tmp_path), "cancel", "job_cancelled"])

    assert result == 0
    assert capsys.readouterr().out.strip() == "job_cancelled cancelled"


def test_logs_command_prints_log_contents(monkeypatch, capsys, tmp_path):
    log_path = tmp_path / "jobs" / "logs" / "job_logs.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("hello\nworld\n", encoding="utf-8")
    monkeypatch.setattr(cli.api, "read_logs", lambda task_id, root=None: log_path.read_text(encoding="utf-8"))

    result = cli.main(["--root", str(tmp_path), "logs", "job_logs"])

    assert result == 0
    assert capsys.readouterr().out == "hello\nworld\n"


def test_logs_follow_tails_existing_log(monkeypatch, tmp_path):
    log_path = tmp_path / "jobs" / "logs" / "job_follow.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("line1\n", encoding="utf-8")
    buffer = io.StringIO()
    monkeypatch.setattr(cli.sys, "stdout", buffer)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: (_ for _ in ()).throw(KeyboardInterrupt()))

    with pytest.raises(KeyboardInterrupt):
        cli.main(["--root", str(tmp_path), "logs", "job_follow", "--follow"])

    assert buffer.getvalue() == "line1\n"
