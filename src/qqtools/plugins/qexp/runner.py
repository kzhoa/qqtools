from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import fsqueue
from .models import qExpTask, utc_now_iso


def _infer_root_from_job_file(job_file: Path) -> Path:
    return job_file.expanduser().resolve().parents[2]


def _build_child_environment(task: qExpTask) -> dict[str, str]:
    child_env = os.environ.copy()
    if task.assigned_gpus:
        child_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in task.assigned_gpus)

    task_env = task.env or {"kind": "none"}
    extra_env = task_env.get("extra_env") or {}
    for key, value in extra_env.items():
        child_env[str(key)] = str(value)
    return child_env


def build_child_shell_command(task: qExpTask) -> list[str]:
    task_env = task.env or {"kind": "none"}
    argv = shlex.join(task.argv)
    env_kind = task_env.get("kind", "none")

    if env_kind == "conda":
        activate_script = shlex.quote(task_env["activate_script"])
        env_name = shlex.quote(task_env["name"])
        shell_command = (
            f"source {activate_script} && conda activate {env_name} && exec {argv}"
        )
    elif env_kind == "venv":
        activate_path = Path(task_env["path"]).expanduser().resolve().joinpath("bin", "activate")
        shell_command = f"source {shlex.quote(str(activate_path))} && exec {argv}"
    else:
        shell_command = f"exec {argv}"

    return ["bash", "-c", shell_command]


def _validate_stage1_metadata(task: qExpTask) -> None:
    if task.status != "running":
        raise RuntimeError(f"Wrapper expected a running task file, got status={task.status!r}.")
    if not task.tmux_session or not task.tmux_window_id:
        raise RuntimeError("Wrapper requires Stage 1 tmux metadata before startup.")
    if not task.assigned_gpus:
        raise RuntimeError("Wrapper requires Stage 1 assigned_gpus before startup.")


def _stream_child_output(stream: Any, log_handle) -> None:
    if stream is None:
        return

    while True:
        chunk = stream.read1(8192)
        if not chunk:
            break
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
        log_handle.write(chunk)
        log_handle.flush()


def _classify_terminal_state(return_code: int) -> tuple[str, str | None]:
    if return_code == 0:
        return "done", None
    if return_code < 0 and abs(return_code) in {signal.SIGTERM, signal.SIGKILL}:
        return "cancelled", "cancelled_by_signal"
    return "failed", "nonzero_exit"


def run_job_file(
    job_file: Path | str,
    *,
    popen_factory: Any = subprocess.Popen,
) -> int:
    job_path = Path(job_file).expanduser().resolve()
    root = _infer_root_from_job_file(job_path)
    task = fsqueue.load_task(job_path)
    _validate_stage1_metadata(task)

    fsqueue.update_running_task(task.task_id, root=root, wrapper_pid=os.getpid())
    log_path = fsqueue.get_log_path(task.task_id, root=root)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    child_process = None
    with log_path.open("ab") as log_handle:
        try:
            child_process = popen_factory(
                build_child_shell_command(task),
                cwd=task.workdir or None,
                env=_build_child_environment(task),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception as exc:
            fsqueue.complete_running_task(
                task.task_id,
                "failed",
                root=root,
                exit_code=None,
                exit_reason=f"launch_failed:{type(exc).__name__}",
            )
            raise

        fsqueue.update_running_task(
            task.task_id,
            root=root,
            process_group_id=child_process.pid,
            started_at=utc_now_iso(),
        )
        _stream_child_output(child_process.stdout, log_handle)
        return_code = child_process.wait()

    terminal_state, exit_reason = _classify_terminal_state(return_code)
    fsqueue.complete_running_task(
        task.task_id,
        terminal_state,
        root=root,
        exit_code=return_code,
        exit_reason=exit_reason,
    )
    return return_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="qexp task wrapper")
    parser.add_argument("--job-file", required=True, type=str)
    args = parser.parse_args(argv)
    return run_job_file(args.job_file)


if __name__ == "__main__":
    raise SystemExit(main())
