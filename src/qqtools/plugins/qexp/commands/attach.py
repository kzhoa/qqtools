"""Read-only attachment to an independent Task observer pane."""

from __future__ import annotations

import subprocess

from ..config_types import RootConfig
from ..executor import Executor


def attach_task(cfg: RootConfig, task_id: str) -> int:
    """Create or reuse a viewer, then join its tmux pane without evicting clients."""
    attachment = Executor().attach_task_observer(cfg, task_id)
    try:
        return subprocess.call(["tmux", "attach-session", "-r", "-t", attachment.window_id])
    except KeyboardInterrupt:
        return 130


__all__ = ["attach_task"]
