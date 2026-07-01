from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .layout import RootConfig, global_tasks_dir, index_by_state_dir
from .models import ALL_PHASES, Task
from .storage import iter_all_tasks, read_json, write_atomic_json


def _read_index_file(path: Path) -> list[str]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data.get("task_ids", [])
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _write_index_file(path: Path, task_ids: list[str]) -> None:
    write_atomic_json(path, {"task_ids": task_ids})


def _legacy_membership_index_dirs(cfg: RootConfig) -> list[Path]:
    indexes_dir = index_by_state_dir(cfg).parent
    return [
        indexes_dir / "tasks_by_batch",
        indexes_dir / "tasks_by_machine",
        indexes_dir / "tasks_by_group",
        indexes_dir / "batches_by_group",
    ]


def _add_to_index(path: Path, task_id: str) -> None:
    task_ids = _read_index_file(path)
    if task_id in task_ids:
        return
    task_ids.append(task_id)
    _write_index_file(path, task_ids)


def _remove_from_index(path: Path, task_id: str) -> None:
    task_ids = _read_index_file(path)
    if task_id not in task_ids:
        return
    task_ids.remove(task_id)
    _write_index_file(path, task_ids)


def collect_index_drift_report(cfg: RootConfig) -> dict[str, Any]:
    expected_by_state: dict[str, set[str]] = {phase: set() for phase in ALL_PHASES}

    tasks_dir = global_tasks_dir(cfg)
    if tasks_dir.is_dir():
        for path in sorted(tasks_dir.glob("*.json")):
            if path.name.endswith(".tmp"):
                continue
            try:
                task = Task.from_dict(read_json(path))
            except Exception:
                continue
            expected_by_state.setdefault(task.status.phase, set()).add(task.task_id)

    actual_by_state: dict[str, set[str]] = {}
    state_dir = index_by_state_dir(cfg)
    if state_dir.is_dir():
        for path in sorted(state_dir.glob("*.json")):
            actual_by_state[path.stem] = set(_read_index_file(path))

    entries: dict[str, Any] = {}
    missing_count = 0
    unexpected_count = 0
    for phase in sorted(set(expected_by_state) | set(actual_by_state)):
        expected_ids = expected_by_state.get(phase, set())
        actual_ids = actual_by_state.get(phase, set())
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        if not missing and not unexpected:
            continue
        missing_count += len(missing)
        unexpected_count += len(unexpected)
        entries[phase] = {
            "missing": missing,
            "unexpected": unexpected,
        }

    family = {
        "ok": missing_count == 0 and unexpected_count == 0,
        "missing_count": missing_count,
        "unexpected_count": unexpected_count,
        "entries": entries,
    }
    return {
        "ok": family["ok"],
        "missing_count": missing_count,
        "unexpected_count": unexpected_count,
        "families": {
            "state": family,
        },
    }


def update_index_on_submit(cfg: RootConfig, task: Task) -> None:
    sync_task_state_index(cfg, task.task_id, task.status.phase)


def remove_index_on_delete(cfg: RootConfig, task: Task) -> None:
    sync_task_state_index(cfg, task.task_id, None)


def update_index_on_phase_change(
    cfg: RootConfig, task_id: str, old_phase: str, new_phase: str
) -> None:
    sync_task_state_index(cfg, task_id, new_phase, stale_phases=[old_phase])


def sync_task_state_index(
    cfg: RootConfig,
    task_id: str,
    actual_phase: str | None,
    *,
    stale_phases: list[str] | tuple[str, ...] | None = None,
) -> None:
    state_dir = index_by_state_dir(cfg)
    phases_to_remove = set(ALL_PHASES)
    if actual_phase is not None:
        phases_to_remove.discard(actual_phase)
    if stale_phases is not None:
        phases_to_remove.update(stale_phases)

    for phase in phases_to_remove:
        _remove_from_index(state_dir / f"{phase}.json", task_id)

    if actual_phase is not None:
        _add_to_index(state_dir / f"{actual_phase}.json", task_id)


def load_index(cfg: RootConfig, index_type: str, key: str) -> list[str]:
    if index_type != "state":
        raise ValueError(f"Unknown index type: {index_type!r}")
    return _read_index_file(index_by_state_dir(cfg) / f"{key}.json")


def rebuild_all_indexes(cfg: RootConfig) -> dict[str, Any]:
    by_state: dict[str, list[str]] = {phase: [] for phase in ALL_PHASES}
    for task in iter_all_tasks(cfg):
        by_state.setdefault(task.status.phase, []).append(task.task_id)

    state_dir = index_by_state_dir(cfg)
    state_dir.mkdir(parents=True, exist_ok=True)
    for path in state_dir.glob("*.json"):
        path.unlink()
    for phase, task_ids in by_state.items():
        if task_ids:
            _write_index_file(state_dir / f"{phase}.json", task_ids)
    removed_legacy_dirs: list[str] = []
    for path in _legacy_membership_index_dirs(cfg):
        if not path.exists():
            continue
        shutil.rmtree(path)
        removed_legacy_dirs.append(path.name)

    return {
        "total_tasks": sum(len(task_ids) for task_ids in by_state.values()),
        "states": {phase: len(task_ids) for phase, task_ids in by_state.items() if task_ids},
        "removed_legacy_index_dirs": removed_legacy_dirs,
    }
