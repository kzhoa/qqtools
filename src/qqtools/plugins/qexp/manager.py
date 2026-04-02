from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import os
import platform
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import fsqueue, tmux
from .models import TASK_FAILED, TASK_RUNNING, utc_now_iso
from .scheduler import qExpScheduler
from .tracker import qExpTracker, probe_gpu_backend

DAEMON_PID_FILE = "daemon.pid"
DEFAULT_IDLE_TIMEOUT_SECONDS = 900
DEFAULT_LOOP_INTERVAL_SECONDS = 5
DEFAULT_STARTUP_WAIT_SECONDS = 1.0
DEFAULT_STARTUP_GRACE_SECONDS = 10
HEARTBEAT_STALE_SECONDS = 30


@dataclass(slots=True)
class qExpPreflightResult:
    gpu_backend: str
    visible_gpu_ids: list[int]


class qExpDaemonLock:
    def __init__(self, root: Path | None = None):
        self.root = fsqueue.ensure_qexp_layout(root)
        self.pid_path = self.root.joinpath(DAEMON_PID_FILE)
        self._handle = None

    def acquire(self) -> bool:
        self.pid_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.pid_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return False

        self._handle = handle
        self.write_metadata({"pid": os.getpid(), "started_at": utc_now_iso()})
        return True

    def write_metadata(self, payload: dict) -> None:
        if self._handle is None:
            raise RuntimeError("Cannot write daemon metadata before acquiring the lock.")

        self._handle.seek(0)
        self._handle.truncate()
        json.dump(payload, self._handle, indent=2, sort_keys=True)
        self._handle.write("\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self.touch_heartbeat()

    def touch_heartbeat(self) -> None:
        if self._handle is None:
            raise RuntimeError("Cannot touch daemon heartbeat before acquiring the lock.")
        os.utime(self.pid_path, None)

    def release(self) -> None:
        if self._handle is None:
            return
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None


def get_idle_timeout_seconds() -> int:
    raw = os.environ.get("QQTOOLS_DAEMON_IDLE_TIMEOUT")
    if raw is None:
        return DEFAULT_IDLE_TIMEOUT_SECONDS
    return int(raw)


def _read_pid_metadata(root: Path | None = None) -> dict | None:
    pid_path = fsqueue.ensure_qexp_layout(root).joinpath(DAEMON_PID_FILE)
    if not pid_path.exists():
        return None

    try:
        with pid_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None
    return payload


def _is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _has_active_daemon_lock(root: Path | None = None, *, create_if_missing: bool) -> bool:
    pid_path = fsqueue.ensure_qexp_layout(root).joinpath(DAEMON_PID_FILE)
    if not pid_path.exists():
        if not create_if_missing:
            return False
        pid_path.touch(exist_ok=True)
    with pid_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False


def has_active_daemon_lock(root: Path | None = None) -> bool:
    return _has_active_daemon_lock(root, create_if_missing=True)


def has_active_daemon_lock_readonly(root: Path | None = None) -> bool:
    return _has_active_daemon_lock(root, create_if_missing=False)


def is_heartbeat_stale(root: Path | None = None, stale_seconds: int = HEARTBEAT_STALE_SECONDS) -> bool:
    pid_path = fsqueue.ensure_qexp_layout(root).joinpath(DAEMON_PID_FILE)
    if not pid_path.exists():
        return True
    return (time.time() - pid_path.stat().st_mtime) > stale_seconds


def is_daemon_active(root: Path | None = None) -> bool:
    if not has_active_daemon_lock(root):
        return False

    metadata = _read_pid_metadata(root)
    if metadata is None:
        return False

    pid = metadata.get("pid")
    if not isinstance(pid, int) or not _is_process_alive(pid):
        return False

    return not is_heartbeat_stale(root)


def run_preflight_checks() -> qExpPreflightResult:
    if platform.system() != "Linux":
        raise RuntimeError("qexp daemon is only supported on local Linux hosts.")

    if not tmux.is_tmux_executable_available():
        raise RuntimeError("tmux is required for qexp. Install it and try again.")

    try:
        importlib.import_module("libtmux")
    except Exception as exc:
        raise RuntimeError(
            "libtmux is required for qexp. Install optional dependencies with "
            "'pip install qqtools[exp]'."
        ) from exc

    backend_name, visible_gpu_ids = probe_gpu_backend()
    if backend_name is None or not visible_gpu_ids:
        raise RuntimeError(
            "qexp detected 0 GPU because neither pynvml nor nvidia-smi produced visible GPUs. "
            "Install optional dependencies with 'pip install qqtools[exp]' and verify the host GPUs."
        )

    return qExpPreflightResult(gpu_backend=backend_name, visible_gpu_ids=visible_gpu_ids)


def start_daemon_background(root: Path | None = None) -> None:
    root_path = fsqueue.ensure_qexp_layout(root)
    tmux.launch_background_daemon(root_path)


def reconcile_running_jobs(
    tracker: qExpTracker,
    root: Path | None = None,
    startup_grace_seconds: int = DEFAULT_STARTUP_GRACE_SECONDS,
    window_exists: Callable[[str | None], bool] = tmux.window_exists,
    process_exists: Callable[[int], bool] = _is_process_alive,
) -> list[str]:
    repaired_paths = fsqueue.repair_state_mismatches(root)
    failed_task_ids: list[str] = []
    if repaired_paths:
        tracker.rebuild_reservations(root=root)

    running_tasks = fsqueue.iter_tasks(TASK_RUNNING, root)
    for task in running_tasks:
        if not task.tmux_window_id:
            fsqueue.fail_running_task(task.task_id, "wrapper_crashed", root=root)
            tracker.release(task.task_id)
            failed_task_ids.append(task.task_id)
            continue

        if not window_exists(task.tmux_window_id):
            fsqueue.fail_running_task(task.task_id, "wrapper_crashed", root=root)
            tracker.release(task.task_id)
            failed_task_ids.append(task.task_id)
            continue

        if task.wrapper_pid is not None:
            if not process_exists(task.wrapper_pid):
                fsqueue.fail_running_task(task.task_id, "wrapper_crashed", root=root)
                tracker.release(task.task_id)
                failed_task_ids.append(task.task_id)
            continue

        scheduled_at_epoch = fsqueue.parse_task_timestamp(task.scheduled_at)
        if scheduled_at_epoch is None:
            fsqueue.fail_running_task(task.task_id, "wrapper_crashed", root=root)
            tracker.release(task.task_id)
            failed_task_ids.append(task.task_id)
            continue

        if (time.time() - scheduled_at_epoch) >= startup_grace_seconds:
            fsqueue.fail_running_task(task.task_id, "wrapper_crashed", root=root)
            tracker.release(task.task_id)
            failed_task_ids.append(task.task_id)

    return failed_task_ids


def run_daemon_foreground(
    root: Path | None = None,
    loop_interval_seconds: int = DEFAULT_LOOP_INTERVAL_SECONDS,
    startup_grace_seconds: int = DEFAULT_STARTUP_GRACE_SECONDS,
    idle_timeout_seconds: int | None = None,
    window_exists: Callable[[str | None], bool] = tmux.window_exists,
) -> int:
    idle_timeout = idle_timeout_seconds if idle_timeout_seconds is not None else get_idle_timeout_seconds()
    preflight = run_preflight_checks()

    daemon_lock = qExpDaemonLock(root)
    if not daemon_lock.acquire():
        raise RuntimeError("Another qexp daemon already holds the active daemon lock.")

    tracker = qExpTracker()
    scheduler = qExpScheduler(tracker=tracker)
    idle_seconds = 0

    try:
        daemon_lock.write_metadata(
            {
                "pid": os.getpid(),
                "started_at": utc_now_iso(),
                "gpu_backend": preflight.gpu_backend,
                "visible_gpu_ids": preflight.visible_gpu_ids,
            }
        )

        while True:
            tracker.refresh(root)
            failed_task_ids = reconcile_running_jobs(
                tracker=tracker,
                root=root,
                startup_grace_seconds=startup_grace_seconds,
                window_exists=window_exists,
            )
            if failed_task_ids:
                tracker.rebuild_reservations(root=root)
            scheduler.run_cycle(root)
            daemon_lock.touch_heartbeat()

            pending_count = fsqueue.count_tasks("pending", root)
            running_count = fsqueue.count_tasks("running", root)

            if pending_count == 0 and running_count == 0:
                idle_seconds += loop_interval_seconds
            else:
                idle_seconds = 0

            if idle_seconds > idle_timeout:
                return 0

            time.sleep(loop_interval_seconds)
    finally:
        daemon_lock.release()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="qexp daemon runner")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--root", type=str, default=None)
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser().resolve() if args.root else None
    return run_daemon_foreground(root=root)


if __name__ == "__main__":
    raise SystemExit(main())
