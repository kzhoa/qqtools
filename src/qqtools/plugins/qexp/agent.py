"""Local qexp agent lifecycle and scheduling loop."""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import Any

from .executor import Executor
from .layout import RootConfig, runtime_pid_path
from .runtime.reservations import active_reservations, release, release_expired_provisionals, reserved_gpu_ids, retag
from .commands.cleanup import reconcile_cleanup_operations
from .commands.group import reconcile_group_cancel_operations
from .commands.task import offer
from .runtime.paths import attempt_path, shared_paths
from .runtime.records import AttemptRecord
from .runtime.store import read_json
from .runtime.placement import offer_due
from .runtime.store import iter_json
from .runtime.tasks import load_task
from .scheduler import reconcile_running_tasks, run_dispatch_cycle


def get_agent_status(cfg: RootConfig, probe_local_pid: bool = True) -> dict[str, Any]:
    pid_path = runtime_pid_path(cfg)
    pid = int(pid_path.read_text().strip()) if pid_path.exists() else None
    running = bool(pid and (not probe_local_pid or _pid_alive(pid)))
    return {"machine_name": cfg.machine_name, "agent_state": "active" if running else "stopped",
            "pid": pid, "is_running": running}


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _visible_gpus(cfg: RootConfig) -> list[int]:
    value = os.environ.get("QEXP_VISIBLE_GPUS", "")
    if value:
        return [int(item) for item in value.split(",") if item.strip()]
    if shutil.which("nvidia-smi"):
        result = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                                check=False, capture_output=True, text=True)
        if result.returncode == 0:
            return [int(item.strip()) for item in result.stdout.splitlines() if item.strip()]
    try:
        import torch
        return list(range(torch.cuda.device_count()))
    except Exception:
        return []


def run_agent_loop(cfg: RootConfig, *, persistent: bool = False, loop_interval: float = 5.0,
                   idle_timeout: int = 600, available_gpus: list[int] | None = None,
                   executor: Executor | None = None) -> None:
    cfg.runtime_root.mkdir(parents=True, exist_ok=True)
    pid_path = runtime_pid_path(cfg)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    executor = executor or Executor()
    started = time.monotonic()
    try:
        while persistent or time.monotonic() - started < idle_timeout:
            release_expired_provisionals(cfg.runtime_root)
            _reconcile_reservations(cfg)
            reconcile_running_tasks(cfg, executor=executor)
            reconcile_group_cancel_operations(cfg)
            reconcile_cleanup_operations(cfg)
            _offer_due_tasks(cfg)
            visible = available_gpus if available_gpus is not None else _visible_gpus(cfg)
            free = [gpu for gpu in visible if gpu not in reserved_gpu_ids(cfg.runtime_root)]
            launched = run_dispatch_cycle(cfg, available_gpus=free, executor=executor)
            if launched:
                started = time.monotonic()
            time.sleep(loop_interval)
    finally:
        pid_path.unlink(missing_ok=True)


def _reconcile_reservations(cfg: RootConfig) -> None:
    for reservation in active_reservations(cfg.runtime_root):
        try:
            task = load_task(cfg, reservation["task_id"])
        except FileNotFoundError:
            release(cfg.runtime_root, reservation["reservation_id"], "task_missing")
            continue
        claim = task.claim_control.get("active_claim") or {}
        if (claim.get("reservation_id") != reservation["reservation_id"] or
                claim.get("fencing_token") != reservation.get("fencing_token")):
            if task.state["projection"] == "blocked":
                continue
            number = task.attempt_control.get("current_attempt_number")
            if number is not None:
                attempt_file = attempt_path(cfg.shared_root, task.task_id, number)
                if attempt_file.exists():
                    attempt = AttemptRecord.from_dict(read_json(attempt_file))
                    if (claim.get("attempt_id") == attempt.attempt_id
                            and claim.get("fencing_token") == attempt.current_fencing_token
                            and retag(cfg.runtime_root, reservation["reservation_id"], attempt.attempt_id,
                                      attempt.current_fencing_token)):
                        continue
            release(cfg.runtime_root, reservation["reservation_id"], "claim_missing")


def _offer_due_tasks(cfg: RootConfig) -> None:
    for path in iter_json(shared_paths(cfg.shared_root)["tasks"]):
        task = load_task(cfg, path.stem)
        if task.placement_policy["home_machine"] != cfg.machine_name or not offer_due(task):
            continue
        try:
            offer(cfg, task.task_id, reason="elapsed")
        except (ValueError, FileNotFoundError):
            continue


def start_agent(cfg: RootConfig, *, persistent: bool = False, idle_timeout: int = 600) -> None:
    run_agent_loop(cfg, persistent=persistent, idle_timeout=idle_timeout)
