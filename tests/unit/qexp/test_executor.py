from pathlib import Path

from qqtools.plugins.qexp.executor import qExpExecutor


def test_executor_builds_wrapper_command_with_runner_and_gpu_env(tmp_path):
    job_file = tmp_path / "jobs" / "running" / "job_demo.json"
    executor = qExpExecutor(send_command=lambda _window_id, _command: None)

    command = executor.build_wrapper_command(job_file, [3, 5])

    assert "CUDA_VISIBLE_DEVICES=3,5" in command
    assert "qqtools.plugins.qexp.runner" in command
    assert str(Path(job_file).resolve()) in command
