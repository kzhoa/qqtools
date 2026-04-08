from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .layout import (
    RootConfig,
    index_by_batch_dir,
    index_by_machine_dir,
    index_by_state_dir,
)
from .models import ALL_PHASES, Task
from .storage import iter_all_tasks, write_atomic_json


# ---------------------------------------------------------------------------
# Low-level index file I/O
# ---------------------------------------------------------------------------


def _read_index_file(path: Path) -> list[str]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("task_ids", [])


def _write_index_file(path: Path, task_ids: list[str]) -> None:
    write_atomic_json(path, {"task_ids": task_ids})


def _add_to_index(path: Path, task_id: str) -> None:
    ids = _read_index_file(path)
    if task_id not in ids:
        ids.append(task_id)
        _write_index_file(path, ids)


def _remove_from_index(path: Path, task_id: str) -> None:
    ids = _read_index_file(path)
    if task_id in ids:
        ids.remove(task_id)
        _write_index_file(path, ids)


# ---------------------------------------------------------------------------
# Public index operations
# ---------------------------------------------------------------------------


def update_index_on_submit(cfg: RootConfig, task: Task) -> None:
    phase = task.status.phase
    _add_to_index(
        index_by_state_dir(cfg) / f"{phase}.json",
        task.task_id,
    )
    _add_to_index(
        index_by_machine_dir(cfg) / f"{task.machine_name}.json",
        task.task_id,
    )
    if task.batch_id:
        _add_to_index(
            index_by_batch_dir(cfg) / f"{task.batch_id}.json",
            task.task_id,
        )


def update_index_on_phase_change(
    cfg: RootConfig, task_id: str, old_phase: str, new_phase: str
) -> None:
    _remove_from_index(
        index_by_state_dir(cfg) / f"{old_phase}.json",
        task_id,
    )
    _add_to_index(
        index_by_state_dir(cfg) / f"{new_phase}.json",
        task_id,
    )


def load_index(cfg: RootConfig, index_type: str, key: str) -> list[str]:
    if index_type == "state":
        return _read_index_file(index_by_state_dir(cfg) / f"{key}.json")
    elif index_type == "batch":
        return _read_index_file(index_by_batch_dir(cfg) / f"{key}.json")
    elif index_type == "machine":
        return _read_index_file(index_by_machine_dir(cfg) / f"{key}.json")
    else:
        raise ValueError(f"Unknown index type: {index_type!r}")


# ---------------------------------------------------------------------------
# Full rebuild
# ---------------------------------------------------------------------------


def rebuild_all_indexes(cfg: RootConfig) -> dict[str, Any]:
    by_state: dict[str, list[str]] = {p: [] for p in ALL_PHASES}
    by_machine: dict[str, list[str]] = {}
    by_batch: dict[str, list[str]] = {}

    tasks = iter_all_tasks(cfg)
    for t in tasks:
        phase = t.status.phase
        by_state.setdefault(phase, []).append(t.task_id)
        by_machine.setdefault(t.machine_name, []).append(t.task_id)
        if t.batch_id:
            by_batch.setdefault(t.batch_id, []).append(t.task_id)

    # Clear and rewrite state indexes
    state_dir = index_by_state_dir(cfg)
    state_dir.mkdir(parents=True, exist_ok=True)
    for f in state_dir.glob("*.json"):
        f.unlink()
    for phase, ids in by_state.items():
        if ids:
            _write_index_file(state_dir / f"{phase}.json", ids)

    # Clear and rewrite machine indexes
    machine_dir = index_by_machine_dir(cfg)
    machine_dir.mkdir(parents=True, exist_ok=True)
    for f in machine_dir.glob("*.json"):
        f.unlink()
    for name, ids in by_machine.items():
        _write_index_file(machine_dir / f"{name}.json", ids)

    # Clear and rewrite batch indexes
    batch_dir = index_by_batch_dir(cfg)
    batch_dir.mkdir(parents=True, exist_ok=True)
    for f in batch_dir.glob("*.json"):
        f.unlink()
    for bid, ids in by_batch.items():
        _write_index_file(batch_dir / f"{bid}.json", ids)

    return {
        "total_tasks": len(tasks),
        "states": {k: len(v) for k, v in by_state.items() if v},
        "machines": {k: len(v) for k, v in by_machine.items()},
        "batches": {k: len(v) for k, v in by_batch.items()},
    }
