"""Shared advisory machine-observability snapshots."""
from __future__ import annotations

from typing import Iterable

from .config_types import RootConfig
from .layout import machine_state_path
from .runtime.locks import exclusive
from .runtime.paths import local_paths
from .runtime.records import utc_now
from .runtime.store import atomic_replace, read_json
from .runtime.ready import READY_WRITER_CAPABILITY


def publish_machine_snapshots(
        cfg: RootConfig, *, instance_id: str, pid: int | None, agent_mode: str,
        observed_state: str, active_attempt_ids: Iterable[str], visible_gpu_ids: Iterable[int],
        reserved_gpu_ids: Iterable[int], heartbeat_interval_seconds: float,
        started_at: str, idle_since_at: str | None, stop_reason: str | None = None,
        reservation_summaries: Iterable[dict[str, object]] | None = None) -> None:
    """Publish machine-owned advisory state without affecting scheduling authority."""
    with exclusive(local_paths(cfg.runtime_root)["locks"] / "machine-snapshot.lock"):
        _write_machine_snapshots(
            cfg, instance_id=instance_id, pid=pid, agent_mode=agent_mode,
            observed_state=observed_state, active_attempt_ids=active_attempt_ids,
            visible_gpu_ids=visible_gpu_ids, reserved_gpu_ids=reserved_gpu_ids,
            heartbeat_interval_seconds=heartbeat_interval_seconds, started_at=started_at,
            idle_since_at=idle_since_at, stop_reason=stop_reason,
            reservation_summaries=reservation_summaries,
        )


def _write_machine_snapshots(
        cfg: RootConfig, *, instance_id: str, pid: int | None, agent_mode: str,
        observed_state: str, active_attempt_ids: Iterable[str], visible_gpu_ids: Iterable[int],
        reserved_gpu_ids: Iterable[int], heartbeat_interval_seconds: float,
        started_at: str, idle_since_at: str | None, stop_reason: str | None = None,
        reservation_summaries: Iterable[dict[str, object]] | None = None) -> None:
    """Write snapshots while the local machine-snapshot lock is held."""
    now = utc_now()
    attempts = sorted(set(active_attempt_ids))
    visible = sorted(set(visible_gpu_ids))
    reserved = sorted(set(reserved_gpu_ids))
    reservations = list(reservation_summaries or [])
    agent = {"agent": {"machine_name": cfg.machine_name, "instance_id": instance_id,
                        "pid": pid, "configured_mode": agent_mode,
                        "observed_state": observed_state, "started_at": started_at,
                        "heartbeat_at": now, "heartbeat_interval_seconds": heartbeat_interval_seconds,
                        "writer_capability": READY_WRITER_CAPABILITY,
                        "idle_since_at": idle_since_at, "active_attempt_ids": attempts,
                        "stop_reason": stop_reason}}
    gpu = {"gpu": {"machine_name": cfg.machine_name, "observed_at": now,
                   "visible_gpu_ids": visible, "reserved_gpu_ids": reserved,
                   "free_gpu_ids": [gpu_id for gpu_id in visible if gpu_id not in reserved],
                   "active_attempt_ids": attempts}}
    summary = {"summary": {"machine_name": cfg.machine_name, "observed_at": now,
                             "agent_state": observed_state,
                             "active_attempt_ids": attempts, "visible_gpu_count": len(visible),
                             "reserved_gpu_ids": reserved,
                             "free_gpu_ids": gpu["gpu"]["free_gpu_ids"],
                             "machine_reservation_count": len(reservations),
                             "machine_reservation_ids": sorted(
                                 item["reservation_id"] for item in reservations
                                 if isinstance(item.get("reservation_id"), str)
                             ),
                             "machine_reservations": [
                                 {key: item.get(key) for key in (
                                 "reservation_id", "project_id", "group_name", "machine_name",
                                 "task_id", "attempt_id", "gpu_ids", "state", "admission"
                                 )}
                                 for item in reservations
                             ]}}
    atomic_replace(machine_state_path(cfg, "agent.json"), agent)
    atomic_replace(machine_state_path(cfg, "gpu.json"), gpu)
    atomic_replace(machine_state_path(cfg, "summary.json"), summary)


def publish_machine_stop_snapshot(
        cfg: RootConfig, *, instance_id: str, pid: int | None, agent_mode: str,
        visible_gpu_ids: Iterable[int], reserved_gpu_ids: Iterable[int],
        heartbeat_interval_seconds: float, started_at: str, idle_since_at: str | None,
        stop_reason: str) -> bool:
    """Publish a stop snapshot only when this agent still owns the machine view."""
    with exclusive(local_paths(cfg.runtime_root)["locks"] / "machine-snapshot.lock"):
        path = machine_state_path(cfg, "agent.json")
        if path.exists():
            try:
                current_instance_id = read_json(path).get("agent", {}).get("instance_id")
            except (OSError, ValueError):
                return False
            if current_instance_id != instance_id:
                return False
        _write_machine_snapshots(
            cfg, instance_id=instance_id, pid=pid, agent_mode=agent_mode,
            observed_state="stopped", active_attempt_ids=[], visible_gpu_ids=visible_gpu_ids,
            reserved_gpu_ids=reserved_gpu_ids, heartbeat_interval_seconds=heartbeat_interval_seconds,
            started_at=started_at, idle_since_at=idle_since_at, stop_reason=stop_reason,
        )
    return True
