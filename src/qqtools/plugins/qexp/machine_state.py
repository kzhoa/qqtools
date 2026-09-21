"""Shared advisory machine-observability snapshots."""

from __future__ import annotations

from typing import Iterable, Mapping

from .config_types import RootConfig
from .layout import machine_state_path
from .runtime.locks import exclusive
from .runtime.paths import local_paths
from .runtime.ready import READY_WRITER_CAPABILITY
from .runtime.ready.group_members import GROUP_READY_MEMBERS_CAPABILITY
from .runtime.records import utc_now
from .runtime.store import read_json, replace_snapshot_if_changed


def publish_machine_snapshots(
    cfg: RootConfig,
    *,
    instance_id: str,
    pid: int | None,
    agent_mode: str,
    observed_state: str,
    active_attempt_ids: Iterable[str],
    visible_gpu_ids: Iterable[int],
    reserved_gpu_ids: Iterable[int],
    heartbeat_interval_seconds: float,
    started_at: str,
    idle_since_at: str | None,
    stop_reason: str | None = None,
    reservation_summaries: Iterable[dict[str, object]] | None = None,
    gpu_policy: Mapping[str, object] | None = None,
) -> None:
    """Publish machine-owned advisory state without affecting scheduling authority."""
    with exclusive(local_paths(cfg.runtime_root)["locks"] / "machine-snapshot.lock"):
        _write_machine_snapshots(
            cfg,
            instance_id=instance_id,
            pid=pid,
            agent_mode=agent_mode,
            observed_state=observed_state,
            active_attempt_ids=active_attempt_ids,
            visible_gpu_ids=visible_gpu_ids,
            reserved_gpu_ids=reserved_gpu_ids,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
            started_at=started_at,
            idle_since_at=idle_since_at,
            stop_reason=stop_reason,
            reservation_summaries=reservation_summaries,
            gpu_policy=gpu_policy,
        )


def _write_machine_snapshots(
    cfg: RootConfig,
    *,
    instance_id: str,
    pid: int | None,
    agent_mode: str,
    observed_state: str,
    active_attempt_ids: Iterable[str],
    visible_gpu_ids: Iterable[int],
    reserved_gpu_ids: Iterable[int],
    heartbeat_interval_seconds: float,
    started_at: str,
    idle_since_at: str | None,
    stop_reason: str | None = None,
    reservation_summaries: Iterable[dict[str, object]] | None = None,
    gpu_policy: Mapping[str, object] | None = None,
) -> None:
    """Write snapshots while the local machine-snapshot lock is held."""
    now = utc_now()
    attempts = sorted(set(active_attempt_ids))
    visible = sorted(set(visible_gpu_ids))
    reserved = sorted(set(reserved_gpu_ids))
    reservations = list(reservation_summaries or [])
    policy = dict(gpu_policy or {})
    agent = {
        "agent": {
            "machine_name": cfg.machine_name,
            "instance_id": instance_id,
            "pid": pid,
            "configured_mode": agent_mode,
            "observed_state": observed_state,
            "started_at": started_at,
            "heartbeat_at": now,
            "heartbeat_interval_seconds": heartbeat_interval_seconds,
            "writer_capability": READY_WRITER_CAPABILITY,
            "writer_capabilities": [
                READY_WRITER_CAPABILITY,
                GROUP_READY_MEMBERS_CAPABILITY,
            ],
            "idle_since_at": idle_since_at,
            "active_attempt_ids": attempts,
            "stop_reason": stop_reason,
            "gpu_policy": policy,
            "warnings": list(policy.get("warnings", [])),
        }
    }
    gpu = {
        "gpu": {
            "machine_name": cfg.machine_name,
            "observed_at": now,
            "visible_gpu_ids": visible,
            "reserved_gpu_ids": reserved,
            "free_gpu_ids": [gpu_id for gpu_id in visible if gpu_id not in reserved],
            "active_attempt_ids": attempts,
            "gpu_policy": policy,
            "policy_revision": policy.get("revision"),
            "policy_mode": policy.get("mode"),
            "policy_source": policy.get("source"),
            "configured_gpu_ids": policy.get("configured_gpu_ids"),
            "discovered_gpu_ids": policy.get("discovered_gpu_ids"),
            "discovery_status": policy.get("discovery_status"),
            "visible_status": policy.get("visible_status"),
            "draining_gpu_ids": policy.get("draining_gpu_ids", []),
            "warnings": list(policy.get("warnings", [])),
        }
    }
    summary = {
        "summary": {
            "machine_name": cfg.machine_name,
            "observed_at": now,
            "agent_state": observed_state,
            "active_attempt_ids": attempts,
            "visible_gpu_count": len(visible),
            "reserved_gpu_ids": reserved,
            "free_gpu_ids": gpu["gpu"]["free_gpu_ids"],
            "machine_reservation_count": len(reservations),
            "machine_reservation_ids": sorted(
                item["reservation_id"] for item in reservations if isinstance(item.get("reservation_id"), str)
            ),
            "machine_reservations": [
                {
                    key: item.get(key)
                    for key in (
                        "reservation_id",
                        "project_id",
                        "group_name",
                        "machine_name",
                        "task_id",
                        "attempt_id",
                        "gpu_ids",
                        "state",
                        "admission",
                    )
                }
                for item in reservations
            ],
            "gpu_policy": policy,
            "warnings": list(policy.get("warnings", [])),
        }
    }
    replace_snapshot_if_changed(machine_state_path(cfg, "agent.json"), agent)
    replace_snapshot_if_changed(machine_state_path(cfg, "gpu.json"), gpu)
    replace_snapshot_if_changed(machine_state_path(cfg, "summary.json"), summary)


def publish_machine_stop_snapshot(
    cfg: RootConfig,
    *,
    instance_id: str,
    pid: int | None,
    agent_mode: str,
    visible_gpu_ids: Iterable[int],
    reserved_gpu_ids: Iterable[int],
    heartbeat_interval_seconds: float,
    started_at: str,
    idle_since_at: str | None,
    stop_reason: str,
    gpu_policy: Mapping[str, object] | None = None,
) -> bool:
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
            cfg,
            instance_id=instance_id,
            pid=pid,
            agent_mode=agent_mode,
            observed_state="stopped",
            active_attempt_ids=[],
            visible_gpu_ids=visible_gpu_ids,
            reserved_gpu_ids=reserved_gpu_ids,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
            started_at=started_at,
            idle_since_at=idle_since_at,
            stop_reason=stop_reason,
            gpu_policy=gpu_policy,
        )
    return True
