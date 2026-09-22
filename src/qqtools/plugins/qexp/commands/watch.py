"""Continuous Task observation commands for qexp."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from typing import Any, TextIO

from .. import observer
from ..cli.output import CliOutput, OutputKind, render


def watch_task(
    cfg: Any,
    task_id: str,
    *,
    interval_seconds: int | float = 2,
    follow_retries: bool = False,
    stdout: TextIO = sys.stdout,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Refresh one compact Task frame until terminal or interruption."""
    first_frame = True
    while True:
        payload = observer.inspect_current_task(cfg, task_id)
        frame = render(CliOutput(OutputKind.TASK_WATCH, payload), "human")
        if not first_frame:
            stdout.write("\x1b[2J\x1b[H")
        stdout.write(frame)
        stdout.write("\n")
        stdout.flush()
        first_frame = False
        is_terminal = bool(payload.get("terminal"))
        is_transition = payload.get("observation_reason") == "concurrent_transition"
        if is_terminal and not is_transition and not follow_retries:
            return 0
        sleep(interval_seconds)


__all__ = ["watch_task"]
