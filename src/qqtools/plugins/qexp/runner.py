"""Token-aware local process runner for one Attempt."""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path

from .config_types import RootConfig
from .layout import load_root_config, runtime_log_path
from .runtime.paths import attempt_path
from .runtime.records import AttemptRecord, utc_now
from .runtime.reservations import release
from .runtime.store import atomic_replace, read_json
from .runtime.tasks import load_task, save_task
from .scheduler import (authority_locks, authorize_launch, renew_attempt_lease,
                        _process_start_time_ticks)


def _terminate_child(child: object, *, timeout: float = 5.0) -> int:
    """Terminate a process group and wait before releasing execution resources."""
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except OSError:
            pass
        deadline = time.monotonic() + timeout
        while child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except OSError:
                pass
    try:
        return child.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, TypeError):
        return child.returncode if child.returncode is not None else 137


def _adopt_manifest_token(cfg: RootConfig, manifest_path: Path, task_id: str,
                          attempt_id: str, current_token: int) -> int:
    """Adopt a recovery-issued fencing token only from this local manifest."""
    manifest = read_json(manifest_path).get("process", {})
    token = manifest.get("fencing_token")
    if not isinstance(token, int) or token == current_token:
        return current_token
    task = load_task(cfg, task_id)
    claim = task.claim_control.get("active_claim") or {}
    attempt = _load_attempt(cfg, task_id, attempt_id)
    if (claim.get("attempt_id") == attempt_id and claim.get("fencing_token") == token
            and attempt.current_fencing_token == token):
        return token
    return current_token


def _load_attempt(cfg: RootConfig, task_id: str, attempt_id: str) -> AttemptRecord:
    for path in (cfg.shared_root / "attempts" / task_id).glob("*.json"):
        data = read_json(path)
        if data["attempt"]["attempt_id"] == attempt_id:
            return AttemptRecord.from_dict(data)
    raise FileNotFoundError(f"Attempt {attempt_id!r} not found for Task {task_id!r}.")


def _publish_process_manifest(cfg: RootConfig, attempt: AttemptRecord, task: object, child: object) -> Path:
    path = cfg.runtime_root / "processes" / f"{attempt.attempt_id}.json"
    wrapper_start = _process_start_time_ticks(os.getpid())
    process_start = _process_start_time_ticks(child.pid)
    atomic_replace(path, {"process": {"attempt_id": attempt.attempt_id, "task_id": attempt.task_id,
        "fencing_token": attempt.current_fencing_token, "wrapper_pid": os.getpid(),
        "wrapper_start_time_ticks": wrapper_start, "process_group_id": child.pid,
        "process_group_start_time_ticks": process_start, "gpu_ids": attempt.assigned_gpus,
        "command": task.spec.command, "working_directory": task.spec.working_directory,
        "created_at": utc_now(), "observed_state": "running", "supervisor": "runner",
        "exit_code": None, "signal": None}})
    return path


def run_attempt(cfg: RootConfig, task_id: str, attempt_id: str, fencing_token: int,
                popen_factory=subprocess.Popen, poll_interval: float = 0.5) -> int:
    task = load_task(cfg, task_id)
    attempt = _load_attempt(cfg, task_id, attempt_id)
    claim = task.claim_control.get("active_claim") or {}
    if claim.get("attempt_id") != attempt_id or claim.get("fencing_token") != fencing_token:
        raise RuntimeError("stale fencing token cannot launch Attempt.")
    if not authorize_launch(cfg, task_id, attempt_id, fencing_token):
        raise RuntimeError("launch authorization was fenced by current Group or Task control state.")
    attempt = _load_attempt(cfg, task_id, attempt_id)
    attempt.phase = "starting"
    attempt.timestamps["launch_authorized_at"] = utc_now()
    atomic_replace(attempt_path(cfg.shared_root, task_id, attempt.attempt_number), attempt.to_dict())
    environment = os.environ.copy()
    if attempt.assigned_gpus:
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, attempt.assigned_gpus))
    log_path = runtime_log_path(cfg, task_id, attempt_id)
    child = None
    manifest_path: Path | None = None
    return_code = 127
    try:
        with log_path.open("ab") as log:
            child = popen_factory(list(task.spec.command), cwd=task.spec.working_directory, env=environment,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            manifest_path = _publish_process_manifest(cfg, attempt, task, child)
            attempt.process.update({"wrapper_pid": os.getpid(), "process_group_id": child.pid,
                                    "wrapper_start_time_ticks":
                                        _process_start_time_ticks(os.getpid()),
                                    "process_group_start_time_ticks":
                                        _process_start_time_ticks(child.pid),
                                    "local_process_manifest": str(manifest_path),
                                    "log_references": [str(log_path)]})
            attempt.phase = "running"
            attempt.timestamps["process_created_at"] = utc_now()
            attempt.timestamps["running_at"] = utc_now()
            atomic_replace(attempt_path(cfg.shared_root, task_id, attempt.attempt_number), attempt.to_dict())
            while child.poll() is None:
                if manifest_path:
                    fencing_token = _adopt_manifest_token(cfg, manifest_path, task_id, attempt_id, fencing_token)
                current = load_task(cfg, task_id)
                if current.control.get("terminate_running"):
                    return_code = _terminate_child(child)
                    break
                if not renew_attempt_lease(cfg, task_id, attempt_id, fencing_token):
                    if manifest_path:
                        recovered_token = _adopt_manifest_token(
                            cfg, manifest_path, task_id, attempt_id, fencing_token
                        )
                        if recovered_token != fencing_token:
                            fencing_token = recovered_token
                            continue
                    # Lost fencing authority means this process must not keep producing side effects.
                    return_code = _terminate_child(child)
                    break
                time.sleep(poll_interval)
            if return_code == 127:
                return_code = child.returncode
    except Exception as exc:
        attempt.result["reason"] = f"launch_failed:{type(exc).__name__}"
        if child is not None:
            return_code = _terminate_child(child)
    finally:
        if manifest_path:
            manifest = read_json(manifest_path)
            manifest["process"].update({"observed_state": "exited", "exit_code": return_code})
            atomic_replace(manifest_path, manifest)
        _publish_terminal(cfg, task_id, attempt_id, fencing_token, return_code, attempt)
    return return_code


def _publish_terminal(cfg: RootConfig, task_id: str, attempt_id: str, fencing_token: int,
                      return_code: int, attempt: AttemptRecord) -> None:
    task = load_task(cfg, task_id)
    with authority_locks(cfg, task):
        task = load_task(cfg, task_id)
        claim = task.claim_control.get("active_claim") or {}
        if claim.get("attempt_id") != attempt_id or claim.get("fencing_token") != fencing_token:
            return
        path = attempt_path(cfg.shared_root, task_id, task.attempt_control["current_attempt_number"])
        current = AttemptRecord.from_dict(read_json(path))
        cancelled = task.control.get("terminate_running")
        current.phase = "cancelled" if cancelled else ("succeeded" if return_code == 0 else "failed")
        current.result.update({"exit_code": return_code, "reason": "cancelled_by_user" if cancelled else
                               ("completed" if return_code == 0 else "nonzero_exit")})
        current.timestamps["finished_at"] = utc_now()
        if cancelled:
            current.termination.update({"acknowledged_at": utc_now(), "result": "terminated"})
        atomic_replace(path, current.to_dict())
        release(cfg.runtime_root, claim["reservation_id"], current.result["reason"])
        task.claim_control["active_claim"] = None
        task.attempt_control["current_attempt_id"] = None
        task.state.update({"projection": current.phase, "reason": current.result["reason"]})
        if cancelled:
            task.control.update({"termination_acknowledged_at": utc_now(),
                                 "termination_result": "terminated"})
        task.meta["revision"] += 1
        task.meta["updated_at"] = utc_now()
        save_task(cfg, task)


def main(argv: list[str] | None = None) -> int:
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
