from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import fsqueue, tmux
from .models import qExpTask
from .tracker import qExpTracker


@dataclass(slots=True)
class qExpScheduler:
    tracker: qExpTracker
    create_window: Callable[[str, str], str] = tmux.create_window_for_task
    kill_window: Callable[[str | None], None] = tmux.kill_window
    session_name: str = tmux.TMUX_SESSION_EXPERIMENTS

    def run_cycle(self, root=None) -> list[str]:
        launched_task_ids: list[str] = []
        pending_tasks = fsqueue.iter_tasks("pending", root)
        pending_tasks.sort(key=lambda task: (task.created_at, task.task_id))

        for task in pending_tasks:
            assigned_gpu_ids = self.tracker.allocate(task.task_id, task.num_gpus)
            if assigned_gpu_ids is None:
                continue

            window_id: str | None = None
            try:
                window_id = self.create_window(task.task_id, self.session_name)
                fsqueue.dispatch_task_to_running(
                    task.task_id,
                    assigned_gpus=assigned_gpu_ids,
                    tmux_session=self.session_name,
                    tmux_window_id=window_id,
                    root=root,
                )
            except FileNotFoundError:
                self.tracker.release(task.task_id)
                self.kill_window(window_id)
                continue
            except Exception:
                self.tracker.release(task.task_id)
                self.kill_window(window_id)
                raise

            launched_task_ids.append(task.task_id)

        return launched_task_ids
