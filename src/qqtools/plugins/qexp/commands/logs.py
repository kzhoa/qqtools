"""Task log access commands for qexp."""
from __future__ import annotations

from pathlib import Path

from ..layout import RootConfig, runtime_log_path
from ..runtime.paths import attempt_path, shared_paths
from ..runtime.records import AttemptRecord, TaskRecord
from ..runtime.store import iter_json, read_json
from ..runtime.tasks import load_task


def _latest_attempt_for_logs(cfg: RootConfig, task: TaskRecord) -> AttemptRecord:
    number = task.attempt_control.get("current_attempt_number")
    if number is not None:
        path = attempt_path(cfg.shared_root, task.task_id, number)
        return AttemptRecord.from_dict(read_json(path))
    attempts_dir = shared_paths(cfg.shared_root)["attempts"] / task.task_id
    attempts = [AttemptRecord.from_dict(read_json(path)) for path in iter_json(attempts_dir)]
    if not attempts:
        raise FileNotFoundError(f"Task {task.task_id!r} has no Attempt records.")
    return max(attempts, key=lambda item: item.attempt_number)


def get_log_path(cfg: RootConfig, task_id: str) -> Path:
    task = load_task(cfg, task_id)
    attempt = _latest_attempt_for_logs(cfg, task)
    log_references = attempt.process.get("log_references") or []
    if log_references:
        return Path(log_references[0])
    return runtime_log_path(cfg, task_id, attempt.attempt_id)


def read_logs(cfg: RootConfig, task_id: str) -> str:
    path = get_log_path(cfg, task_id)
    if not path.exists():
        task = load_task(cfg, task_id)
        attempt = _latest_attempt_for_logs(cfg, task)
        raise FileNotFoundError(
            f"log for Task {task_id!r} Attempt {attempt.attempt_id!r} on machine "
            f"{attempt.machine_name!r} was not found at {path}"
        )
    return path.read_text(encoding="utf-8", errors="replace")


def tail_log(cfg: RootConfig, task_id: str) -> None:
    print(read_logs(cfg, task_id), end="")
