from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from . import fsqueue, tmux
from .executor import qExpExecutor
from .tracker import qExpTracker


@dataclass(slots=True)
class qExpScheduler:
    tracker: qExpTracker
    create_window: Callable[[str, str], str] = tmux.create_window_for_task
    kill_window: Callable[[str | None], None] = tmux.kill_window
    executor: qExpExecutor = field(default_factory=qExpExecutor)
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
                running_path = fsqueue.dispatch_task_to_running(
                    task.task_id,
                    assigned_gpus=assigned_gpu_ids,
                    tmux_session=self.session_name,
                    tmux_window_id=window_id,
                    root=root,
                )
                self.executor.launch_wrapper(window_id, running_path, assigned_gpu_ids)
            except FileNotFoundError:
                self.tracker.release(task.task_id)
                self.kill_window(window_id)
                continue
            except Exception:
                self.tracker.release(task.task_id)
                self.kill_window(window_id)
                existing = fsqueue.load_task_by_id(task.task_id, root)
                if existing is not None and existing.status == "running":
                    fsqueue.complete_running_task(
                        task.task_id,
                        "failed",
                        root=root,
                        exit_code=None,
                        exit_reason="wrapper_launch_failed",
                    )
                raise

            launched_task_ids.append(task.task_id)

        return launched_task_ids
