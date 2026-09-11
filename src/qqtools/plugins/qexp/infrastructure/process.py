"""Small, side-effectful process adapters."""

import os
import signal
import time
from pathlib import Path


def is_process_group_alive(process_group_id: int | None) -> bool:
    if not process_group_id:
        return False
    try:
        os.killpg(process_group_id, 0)
    except OSError:
        return False
    return True


def is_process_alive(process_id: int | None) -> bool:
    if not process_id:
        return False
    try:
        os.kill(process_id, 0)
    except OSError:
        return False
    return True


def process_start_time_ticks(process_id: int | None) -> int | None:
    if not process_id:
        return None
    try:
        stat = (Path("/proc") / str(process_id) / "stat").read_text(encoding="utf-8")
        fields = stat.rsplit(")", 1)[1].split()
        return int(fields[19])
    except (FileNotFoundError, IndexError, ValueError, OSError):
        return None


def terminate_process_group(process_group_id: int, *, grace_seconds: float = 5.0) -> bool:
    """Terminate a process group and return only after its absence is confirmed."""
    if not is_process_group_alive(process_group_id):
        return True
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except OSError:
        return not is_process_group_alive(process_group_id)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not is_process_group_alive(process_group_id):
            return True
        time.sleep(0.05)
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except OSError:
        pass
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not is_process_group_alive(process_group_id):
            return True
        time.sleep(0.05)
    return not is_process_group_alive(process_group_id)
