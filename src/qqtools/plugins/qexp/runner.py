"""Passive local process wrapper for one already-authorized Attempt."""
from __future__ import annotations

import argparse
import ctypes
import os
import signal
import subprocess
import sys
from pathlib import Path

from .config_types import RootConfig
from .layout import load_root_config, shared_attempt_log_path
from .runtime.paths import local_paths
from .runtime.records import AttemptRecord, utc_now
from .runtime.store import atomic_replace, create_if_absent, read_json
from .runtime.tasks import load_task
from .scheduler import _process_start_time_ticks

LOCAL_PROCESS_PROTOCOL_VERSION = 1


def _load_attempt(cfg: RootConfig, task_id: str, attempt_id: str) -> AttemptRecord:
    for path in (cfg.shared_root / "attempts" / task_id).glob("*.json"):
        data = read_json(path)
        if data["attempt"]["attempt_id"] == attempt_id:
            return AttemptRecord.from_dict(data)
    raise FileNotFoundError(f"Attempt {attempt_id!r} not found for Task {task_id!r}.")


def registration_path(cfg: RootConfig, attempt_id: str) -> Path:
    return local_paths(cfg.runtime_root)["registrations"] / f"{attempt_id}.json"


def observation_path(cfg: RootConfig, attempt_id: str) -> Path:
    return local_paths(cfg.runtime_root)["observations"] / f"{attempt_id}.json"


def launch_intent_path(cfg: RootConfig, attempt_id: str) -> Path:
    return local_paths(cfg.runtime_root)["launch_intents"] / f"{attempt_id}.json"


def _publish_launch_intent(cfg: RootConfig, attempt: AttemptRecord, task: object) -> Path:
    """Persist wrapper identity before creating the training process."""
    path = launch_intent_path(cfg, attempt.attempt_id)
    if path.exists():
        raise RuntimeError(f"launch intent already exists for {attempt.attempt_id!r}.")
    create_if_absent(path, {"launch_intent": {
        "protocol_version": LOCAL_PROCESS_PROTOCOL_VERSION,
        "attempt_id": attempt.attempt_id,
        "task_id": attempt.task_id,
        "fencing_token": attempt.current_fencing_token,
        "wrapper_pid": os.getpid(),
        "wrapper_start_time_ticks": _process_start_time_ticks(os.getpid()),
        "gpu_ids": attempt.assigned_gpus,
        "command": task.spec.command,
        "working_directory": task.spec.working_directory,
        "lease_expires_at": (task.claim_control.get("active_claim") or {}).get("lease_expires_at"),
        "created_at": utc_now(),
    }})
    return path


def _kill_owned_process_group(_signal_number: int, _frame: object) -> None:
    """Immediately kill the runner's private process group after its parent dies."""
    try:
        os.killpg(os.getpgrp(), signal.SIGKILL)
    finally:
        os._exit(128 + signal.SIGUSR1)


def _configure_process_group_guardian(expected_parent_pid: int) -> None:
    """Configure the private guardian to kill its group if the runner dies."""
    if os.name != "posix":
        raise RuntimeError("qexp process-group guardian requires POSIX")
    if os.getppid() != expected_parent_pid:
        _kill_owned_process_group(signal.SIGUSR1, None)
    signal.signal(signal.SIGUSR1, _kill_owned_process_group)
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        prctl = libc.prctl
    except AttributeError as exc:
        raise RuntimeError("qexp process-group guardian requires Linux prctl") from exc
    if prctl(1, signal.SIGUSR1, 0, 0, 0) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, "prctl(PR_SET_PDEATHSIG) failed")
    if os.getppid() != expected_parent_pid:
        _kill_owned_process_group(signal.SIGUSR1, None)


def _run_guardian(command: list[str], expected_parent_pid: int) -> int:
    """Run training inside a group that dies when the passive runner dies."""
    _configure_process_group_guardian(expected_parent_pid)
    return subprocess.Popen(command).wait()


def _publish_registration(cfg: RootConfig, attempt: AttemptRecord, task: object, child: object) -> Path:
    """Publish immutable child identity; the agent owns the mutable process manifest."""
    path = registration_path(cfg, attempt.attempt_id)
    if path.exists():
        raise RuntimeError(f"process registration already exists for {attempt.attempt_id!r}.")
    create_if_absent(path, {"process_registration": {
        "protocol_version": LOCAL_PROCESS_PROTOCOL_VERSION,
        "attempt_id": attempt.attempt_id,
        "task_id": attempt.task_id,
        "fencing_token": attempt.current_fencing_token,
        "wrapper_pid": os.getpid(),
        "wrapper_start_time_ticks": _process_start_time_ticks(os.getpid()),
        "process_group_id": child.pid,
        "process_group_start_time_ticks": _process_start_time_ticks(child.pid),
        "gpu_ids": attempt.assigned_gpus,
        "command": task.spec.command,
        "working_directory": task.spec.working_directory,
        "lease_expires_at": (task.claim_control.get("active_claim") or {}).get("lease_expires_at"),
        "created_at": utc_now(),
    }})
    return path


def _publish_exit_observation(cfg: RootConfig, attempt_id: str, return_code: int) -> None:
    atomic_replace(observation_path(cfg, attempt_id), {"exit_observation": {
        "protocol_version": LOCAL_PROCESS_PROTOCOL_VERSION,
        "attempt_id": attempt_id,
        "observed_exit_code": return_code,
        "observed_at": utc_now(),
    }})


def run_attempt(cfg: RootConfig, task_id: str, attempt_id: str, fencing_token: int,
                popen_factory=subprocess.Popen, poll_interval: float = 0.5) -> int:
    """Start and observe a process without participating in execution authority."""
    del poll_interval
    task = load_task(cfg, task_id)
    attempt = _load_attempt(cfg, task_id, attempt_id)
    if attempt.current_fencing_token != fencing_token:
        raise RuntimeError("stale fencing token cannot launch Attempt.")
    environment = os.environ.copy()
    if attempt.assigned_gpus:
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, attempt.assigned_gpus))
    log_path = shared_attempt_log_path(cfg, task_id, attempt_id)
    _publish_launch_intent(cfg, attempt, task)
    with log_path.open("ab") as log:
        guardian_command = [sys.executable, "-m", "qqtools.plugins.qexp.runner", "--guardian",
                            "--parent-pid", str(os.getpid()), "--", *task.spec.command]
        child = popen_factory(guardian_command, cwd=task.spec.working_directory, env=environment,
                              stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        _publish_registration(cfg, attempt, task, child)
        return_code = child.wait()
    _publish_exit_observation(cfg, attempt_id, return_code)
    return return_code


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--guardian":
        if len(argv) < 4 or argv[1] != "--parent-pid":
            raise RuntimeError("qexp guardian requires its runner parent PID")
        try:
            expected_parent_pid = int(argv[2])
        except ValueError as exc:
            raise RuntimeError("qexp guardian parent PID must be an integer") from exc
        command = argv[3:]
        if command[:1] == ["--"]:
            command = command[1:]
        if not command:
            raise RuntimeError("qexp guardian requires a training command")
        return _run_guardian(command, expected_parent_pid)
    parser = argparse.ArgumentParser(description="qexp Attempt runner")
    parser.add_argument("--shared-root", required=True)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--fencing-token", required=True, type=int)
    parser.add_argument("--runtime-root")
    args = parser.parse_args(argv)
    cfg = load_root_config(args.shared_root, args.machine, args.runtime_root, require_initialized=True)
    return run_attempt(cfg, args.task_id, args.attempt_id, args.fencing_token)


if __name__ == "__main__":
    raise SystemExit(main())
