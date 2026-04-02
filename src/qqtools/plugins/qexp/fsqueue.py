from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import STATE_DIRECTORY_MAP, TASK_STATES, qExpTask, get_state_directory_name, utc_now_iso

QQTOOLS_HOME_ENV = "QQTOOLS_HOME"
LEGAL_STATE_TRANSITIONS = {
    ("pending", "running"),
    ("running", "done"),
    ("running", "failed"),
    ("pending", "cancelled"),
    ("running", "cancelled"),
}


def _normalize_root(root: Path | str | None) -> Path:
    if root is None:
        return get_qexp_home()
    return Path(root).expanduser().resolve()


def get_qexp_home() -> Path:
    root = os.environ.get(QQTOOLS_HOME_ENV)
    if root:
        return Path(root).expanduser().resolve()
    return Path.home().joinpath(".qqtools")


def get_jobs_root(root: Path | str | None = None) -> Path:
    state_root = _normalize_root(root)
    return state_root.joinpath("jobs")


def ensure_qexp_layout(root: Path | str | None = None) -> Path:
    state_root = _normalize_root(root)
    jobs_root = get_jobs_root(state_root)

    state_root.mkdir(parents=True, exist_ok=True)
    jobs_root.mkdir(parents=True, exist_ok=True)
    jobs_root.joinpath("logs").mkdir(parents=True, exist_ok=True)
    for directory_name in STATE_DIRECTORY_MAP.values():
        jobs_root.joinpath(directory_name).mkdir(parents=True, exist_ok=True)
    return state_root


def get_state_dir(state: str, root: Path | str | None = None) -> Path:
    jobs_root = get_jobs_root(root)
    return jobs_root.joinpath(get_state_directory_name(state))


def get_log_path(task_id: str, root: Path | str | None = None) -> Path:
    jobs_root = get_jobs_root(root)
    return jobs_root.joinpath("logs", f"{task_id}.log")


def get_task_path(state: str, task_id: str, root: Path | str | None = None) -> Path:
    return get_state_dir(state, root).joinpath(f"{task_id}.json")


def _write_atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def read_task_payload(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Task payload at {path} must be a dictionary.")
    return payload


def _update_task_file(path: Path, updates: dict[str, Any]) -> qExpTask:
    task = load_task(path)
    for field_name, field_value in updates.items():
        if not hasattr(task, field_name):
            raise AttributeError(f"Task has no field named '{field_name}'.")
        setattr(task, field_name, field_value)
    _write_atomic_json(path, task.to_dict())
    return task


def save_task(task: qExpTask, root: Path | None = None) -> Path:
    ensure_qexp_layout(root)
    task_path = get_task_path(task.status, task.task_id, root)
    _write_atomic_json(task_path, task.to_dict())
    return task_path


def load_task(path: Path) -> qExpTask:
    data = read_task_payload(path)
    task = qExpTask.from_dict(data)

    parent_state = path.parent.name
    if parent_state not in TASK_STATES:
        raise ValueError(f"Task file is stored under unsupported state directory: {parent_state}")
    if task.status != parent_state:
        raise ValueError(
            f"Task status '{task.status}' does not match directory '{parent_state}'."
        )
    return task


def iter_tasks(state: str, root: Path | None = None) -> list[qExpTask]:
    ensure_qexp_layout(root)
    tasks: list[qExpTask] = []
    for path in get_state_task_paths(state, root):
        tasks.append(load_task(path))
    return tasks


def get_state_task_paths(state: str, root: Path | None = None) -> list[Path]:
    ensure_qexp_layout(root)
    state_dir = get_state_dir(state, root)
    paths: list[Path] = []
    for path in sorted(state_dir.glob("*.json")):
        if path.name.endswith(".tmp"):
            continue
        paths.append(path)
    return paths


def iter_all_task_paths(root: Path | None = None) -> list[Path]:
    ensure_qexp_layout(root)
    task_paths: list[Path] = []
    for state in TASK_STATES:
        task_paths.extend(get_state_task_paths(state, root))
    return task_paths


def count_tasks(state: str, root: Path | None = None) -> int:
    ensure_qexp_layout(root)
    return len(list(get_state_dir(state, root).glob("*.json")))


def find_task_path(task_id: str, root: Path | None = None) -> Path | None:
    for path in iter_all_task_paths(root):
        if path.stem == task_id:
            return path
    return None


def load_task_by_id(task_id: str, root: Path | None = None) -> qExpTask | None:
    path = find_task_path(task_id, root)
    if path is None:
        return None
    return load_task(path)


def move_task(task_id: str, from_state: str, to_state: str, root: Path | None = None) -> Path:
    ensure_qexp_layout(root)
    if (from_state, to_state) not in LEGAL_STATE_TRANSITIONS:
        raise ValueError(f"Illegal task state transition: {from_state} -> {to_state}")

    source = get_task_path(from_state, task_id, root)
    if not source.exists():
        raise FileNotFoundError(f"Task file not found for state transition: {source}")

    task = load_task(source)
    task.status = to_state
    destination = get_task_path(to_state, task_id, root)
    # This write-then-rename flow preserves atomic visibility at the destination path,
    # but it leaves a crash window where the source directory may temporarily contain a
    # payload whose JSON status already says `to_state`. Phase 2 reconciliation must treat
    # that directory/status mismatch as recoverable residue and repair it deterministically.
    _write_atomic_json(source, task.to_dict())
    os.replace(source, destination)
    return destination


def dispatch_task_to_running(
    task_id: str,
    assigned_gpus: list[int],
    tmux_session: str,
    tmux_window_id: str,
    root: Path | None = None,
) -> Path:
    ensure_qexp_layout(root)
    source = get_task_path("pending", task_id, root)
    if not source.exists():
        raise FileNotFoundError(f"Pending task file not found for dispatch: {source}")

    task = load_task(source)
    task.status = "running"
    task.scheduled_at = utc_now_iso()
    task.assigned_gpus = list(assigned_gpus)
    task.tmux_session = tmux_session
    task.tmux_window_id = tmux_window_id
    destination = get_task_path("running", task_id, root)
    _write_atomic_json(source, task.to_dict())
    os.replace(source, destination)
    return destination


def fail_running_task(task_id: str, exit_reason: str, root: Path | None = None) -> Path:
    ensure_qexp_layout(root)
    source = get_task_path("running", task_id, root)
    if not source.exists():
        raise FileNotFoundError(f"Running task file not found for failure transition: {source}")

    task = load_task(source)
    task.status = "failed"
    task.exit_reason = exit_reason
    task.finished_at = utc_now_iso()
    destination = get_task_path("failed", task_id, root)
    _write_atomic_json(source, task.to_dict())
    os.replace(source, destination)
    return destination


def update_running_task(task_id: str, root: Path | None = None, **updates: Any) -> qExpTask:
    ensure_qexp_layout(root)
    path = get_task_path("running", task_id, root)
    if not path.exists():
        raise FileNotFoundError(f"Running task file not found for update: {path}")
    return _update_task_file(path, updates)


def cancel_pending_task(task_id: str, root: Path | None = None) -> Path:
    ensure_qexp_layout(root)
    source = get_task_path("pending", task_id, root)
    if not source.exists():
        raise FileNotFoundError(f"Pending task file not found for cancellation: {source}")

    task = load_task(source)
    task.status = "cancelled"
    task.exit_reason = "cancelled_before_start"
    task.finished_at = utc_now_iso()
    destination = get_task_path("cancelled", task_id, root)
    _write_atomic_json(source, task.to_dict())
    os.replace(source, destination)
    return destination


def complete_running_task(
    task_id: str,
    terminal_state: str,
    root: Path | None = None,
    *,
    exit_code: int | None,
    exit_reason: str | None = None,
    finished_at: str | None = None,
) -> Path:
    ensure_qexp_layout(root)
    if terminal_state not in {"done", "failed", "cancelled"}:
        raise ValueError(
            "terminal_state must be one of done, failed, or cancelled for running completion."
        )

    source = get_task_path("running", task_id, root)
    if not source.exists():
        raise FileNotFoundError(f"Running task file not found for completion: {source}")

    task = load_task(source)
    task.status = terminal_state
    task.exit_code = exit_code
    task.exit_reason = exit_reason
    task.finished_at = finished_at or utc_now_iso()
    destination = get_task_path(terminal_state, task_id, root)
    _write_atomic_json(source, task.to_dict())
    os.replace(source, destination)
    return destination


def repair_state_mismatches(root: Path | None = None) -> list[Path]:
    repaired_paths: list[Path] = []
    for path in iter_all_task_paths(root):
        payload = read_task_payload(path)
        payload_state = payload.get("status")
        if payload_state not in TASK_STATES:
            continue
        if payload_state == path.parent.name:
            continue

        destination = get_task_path(payload_state, path.stem, root)
        if destination.exists():
            path.unlink()
            repaired_paths.append(destination)
            continue

        os.replace(path, destination)
        repaired_paths.append(destination)
    return repaired_paths


def parse_task_timestamp(value: str | None) -> float | None:
    if not value:
        return None

    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized).timestamp()
