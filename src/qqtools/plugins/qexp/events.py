from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .layout import RootConfig, global_events_dir, machine_events_dir
from .models import generate_id, utc_now_iso
from .storage import read_json, write_atomic_json

EVENT_TYPES = (
    "submit_succeeded",
    "submit_failed",
    "task_started",
    "task_finished",
    "task_failed",
    "task_cancelled",
    "task_orphaned",
    "agent_started",
    "agent_stopped",
    "task_claimed",
)


def _date_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def write_event(
    cfg: RootConfig,
    event_type: str,
    task_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> str:
    event_id = generate_id()
    date = _date_str()
    payload: dict[str, Any] = {
        "event_id": event_id,
        "event_type": event_type,
        "machine_name": cfg.machine_name,
        "timestamp": utc_now_iso(),
    }
    if task_id is not None:
        payload["task_id"] = task_id
    if details:
        payload["details"] = details

    # Write to global events
    global_dir = global_events_dir(cfg) / date
    global_dir.mkdir(parents=True, exist_ok=True)
    write_atomic_json(global_dir / f"{event_id}.json", payload)

    # Write to machine events
    machine_dir = machine_events_dir(cfg) / date
    machine_dir.mkdir(parents=True, exist_ok=True)
    write_atomic_json(machine_dir / f"{event_id}.json", payload)

    return event_id


def query_events(
    cfg: RootConfig,
    date: str | None = None,
    event_type: str | None = None,
    machine: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    if machine and machine != cfg.machine_name:
        alt_cfg = RootConfig(
            shared_root=cfg.shared_root,
            project_root=cfg.project_root,
            machine_name=machine,
            runtime_root=cfg.runtime_root,
        )
        base = machine_events_dir(alt_cfg)
    elif machine:
        base = machine_events_dir(cfg)
    else:
        base = global_events_dir(cfg)

    if not base.is_dir():
        return []

    results: list[dict[str, Any]] = []
    date_dirs = sorted(base.iterdir(), reverse=True) if date is None else []
    if date is not None:
        d = base / date
        date_dirs = [d] if d.is_dir() else []

    for dd in date_dirs:
        if not dd.is_dir():
            continue
        for f in sorted(dd.glob("*.json"), reverse=True):
            if f.name.endswith(".tmp"):
                continue
            try:
                event = read_json(f)
            except (json.JSONDecodeError, OSError):
                continue
            if event_type and event.get("event_type") != event_type:
                continue
            results.append(event)
            if len(results) >= limit:
                return results

    return results
