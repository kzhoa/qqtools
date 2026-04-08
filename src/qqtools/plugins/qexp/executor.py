from __future__ import annotations

import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import tmux


@dataclass(slots=True)
class qExpExecutor:
    send_command: Callable[[str, str], None] = tmux.send_command_to_window

    def build_wrapper_command(self, job_file: Path | str, assigned_gpu_ids: list[int]) -> str:
        job_path = Path(job_file).expanduser().resolve()
        command_parts = [
            f"CUDA_VISIBLE_DEVICES={shlex.quote(','.join(str(gpu_id) for gpu_id in assigned_gpu_ids))}",
            shlex.quote(sys.executable),
            "-m",
            "qqtools.plugins.qexp.runner",
            "--job-file",
            shlex.quote(str(job_path)),
        ]
        return " ".join(command_parts)

    def launch_wrapper(self, window_id: str, job_file: Path | str, assigned_gpu_ids: list[int]) -> str:
        command = self.build_wrapper_command(job_file, assigned_gpu_ids)
        self.send_command(window_id, command)
        return command
