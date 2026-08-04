"""Append-only event projection for diagnostics."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from .runtime.paths import shared_paths
from .runtime.paths import local_paths
from .runtime.records import new_id
from .runtime.store import atomic_replace, iter_json, read_json


def write_event(cfg: Any, event_type: str, *, task_id: str | None = None,
                details: dict[str, Any] | None = None) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    event = {"event_id": uuid.uuid4().hex, "event_type": event_type, "task_id": task_id,
             "machine_name": cfg.machine_name, "timestamp": datetime.now(timezone.utc).isoformat(),
             "details": details or {}}
    atomic_replace(shared_paths(cfg.shared_root)["events"] / now / f"{event['event_id']}.json", event)


def write_diagnostic_event(cfg: Any, event_type: str, *, task_id: str | None = None,
                           attempt_id: str | None = None, details: dict[str, Any] | None = None) -> None:
    """Publish diagnostics or preserve them locally when shared storage is unavailable."""
    try:
        write_event(cfg, event_type, task_id=task_id, details=details)
        return
    except OSError:
        pass
    event_id = new_id()
    event = {"event_id": event_id, "event_type": event_type, "task_id": task_id,
             "attempt_id": attempt_id, "machine_name": cfg.machine_name,
             "timestamp": datetime.now(timezone.utc).isoformat(), "details": details or {}}
    directory = local_paths(cfg.runtime_root)["events"] / (attempt_id or "machine")
    atomic_replace(directory / f"{event_id}.json", event)


def flush_local_events(cfg: Any) -> int:
    """Idempotently copy durable local diagnostics to the shared event stream."""
    flushed = 0
    root = local_paths(cfg.runtime_root)["events"]
    for directory in sorted(root.glob("*")):
        if not directory.is_dir():
            continue
        for path in iter_json(directory):
            event = read_json(path)
            try:
                now = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00")).strftime("%Y-%m-%d")
                atomic_replace(shared_paths(cfg.shared_root)["events"] / now / f"{event['event_id']}.json", event)
            except (OSError, ValueError, KeyError):
                continue
            path.unlink(missing_ok=True)
            flushed += 1
    return flushed
