import io

import pytest

from qqtools.plugins.qexp import fsqueue
from qqtools.plugins.qexp.models import qExpTask
from qqtools.plugins.qexp.runner import build_child_shell_command, run_job_file


class _FakeStdout:
    def __init__(self, chunks: list[bytes]):
        self._chunks = list(chunks)

    def read1(self, _size: int) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


class _FakeProcess:
    def __init__(self, return_code: int, chunks: list[bytes]):
        self.pid = 43210
        self.stdout = _FakeStdout(chunks)
        self._return_code = return_code

    def wait(self) -> int:
        return self._return_code


class _FakeStdoutBuffer:
    def __init__(self):
        self.buffer = io.BytesIO()


@pytest.fixture
def qexp_root(tmp_path, monkeypatch):
    root = tmp_path / "runner-home"
    monkeypatch.setenv(fsqueue.QQTOOLS_HOME_ENV, str(root))
    return root


def test_build_child_shell_command_supports_conda_activation():
    task = qExpTask(
        task_id="job_conda",
        argv=["python", "train.py", "--epochs", "1"],
        num_gpus=1,
        env={
            "kind": "conda",
            "name": "torch",
            "activate_script": "/opt/miniconda3/etc/profile.d/conda.sh",
        },
    )

    command = build_child_shell_command(task)

    assert command[:2] == ["bash", "-c"]
    assert "conda activate torch" in command[2]
    assert "exec python train.py --epochs 1" in command[2]


def test_run_job_file_persists_logs_and_moves_successful_task(qexp_root, monkeypatch):
    running_task = qExpTask(
        task_id="job_success",
        argv=["python", "train.py"],
        num_gpus=1,
        status="running",
        scheduled_at="2024-01-01T00:00:00Z",
        assigned_gpus=[0],
        tmux_session="experiments",
        tmux_window_id="@1",
    )
    job_file = fsqueue.save_task(running_task, qexp_root)
    fake_stdout = _FakeStdoutBuffer()
    monkeypatch.setattr("sys.stdout", fake_stdout)

    def _fake_popen(*args, **kwargs):
        assert kwargs["start_new_session"] is True
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
        return _FakeProcess(0, [b"hello\n", b"world\n"])

    result = run_job_file(job_file, popen_factory=_fake_popen)

    assert result == 0
    completed = fsqueue.load_task_by_id("job_success", qexp_root)
    assert completed is not None
    assert completed.status == "done"
    assert completed.wrapper_pid is not None
    assert completed.process_group_id == 43210
    assert completed.started_at is not None
    assert completed.exit_code == 0
    assert fsqueue.get_log_path("job_success", qexp_root).read_text(encoding="utf-8") == "hello\nworld\n"
    assert fake_stdout.buffer.getvalue() == b"hello\nworld\n"


def test_run_job_file_marks_sigterm_exit_as_cancelled(qexp_root, monkeypatch):
    running_task = qExpTask(
        task_id="job_cancelled",
        argv=["python", "train.py"],
        num_gpus=1,
        status="running",
        scheduled_at="2024-01-01T00:00:00Z",
        assigned_gpus=[1],
        tmux_session="experiments",
        tmux_window_id="@2",
    )
    job_file = fsqueue.save_task(running_task, qexp_root)
    monkeypatch.setattr("sys.stdout", _FakeStdoutBuffer())

    result = run_job_file(job_file, popen_factory=lambda *args, **kwargs: _FakeProcess(-15, []))

    assert result == -15
    completed = fsqueue.load_task_by_id("job_cancelled", qexp_root)
    assert completed is not None
    assert completed.status == "cancelled"
    assert completed.exit_reason == "cancelled_by_signal"
