"""Append-only event projection for diagnostics."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from .runtime.paths import shared_paths
from .runtime.store import atomic_replace


def write_event(cfg: Any, event_type: str, *, task_id: str | None = None,
                details: dict[str, Any] | None = None) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    event = {"event_id": uuid.uuid4().hex, "event_type": event_type, "task_id": task_id,
             "machine_name": cfg.machine_name, "timestamp": datetime.now(timezone.utc).isoformat(),
             "details": details or {}}
    atomic_replace(shared_paths(cfg.shared_root)["events"] / now / f"{event['event_id']}.json", event)
