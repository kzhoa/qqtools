"""Best-effort cleanup for the independent progress-v3 advisory channel."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from qqtools.qexp._progress_protocol import identifier, read_advisory_snapshot

from .locks import exclusive
from .progress import _progress_lock_path

_LOCAL_MAILBOX_NAME = "latest-v3.json"


def _local_path(cfg: Any, directory: str, attempt_id: str) -> Path:
    return Path(cfg.runtime_root) / directory / f"{identifier(attempt_id)}.json"


def _local_mailbox_path(runtime_root: Path, attempt_id: str) -> Path:
    return Path(runtime_root) / "progress" / identifier(attempt_id) / _LOCAL_MAILBOX_NAME


def _context_path(cfg: Any, attempt_id: str) -> Path:
    return _local_path(cfg, "progress-v3-contexts", attempt_id)


def _observed_path(cfg: Any, attempt_id: str) -> Path:
    return _local_path(cfg, "progress-v3-observed", attempt_id)


def _coordinator_path(cfg: Any, attempt_id: str) -> Path:
    return _local_path(cfg, "progress-coordinator/v3", attempt_id)


def _unlink_record(path: Path, removed: list[str]) -> None:
    try:
        path.unlink()
        removed.append(str(path))
    except OSError:
        pass


def _unlink_with_temporaries(path: Path, removed: list[str]) -> None:
    _unlink_record(path, removed)
    try:
        temporaries = list(path.parent.glob(f".{path.name}.*"))
    except OSError:
        temporaries = []
    for temporary in temporaries:
        _unlink_record(temporary, removed)


def cleanup_local_progress_v3(cfg: Any, task_id: str, attempt_ids: set[str]) -> list[str]:
    """Remove v3 contexts before their local mailbox, cache, and cadence files."""
    task_id = identifier(task_id)
    removed: list[str] = []
    targets: set[str] = set()
    for attempt_id in attempt_ids:
        try:
            targets.add(identifier(attempt_id))
        except ValueError:
            continue

    context_root = Path(cfg.runtime_root) / "progress-v3-contexts"
    try:
        context_paths = list(context_root.glob("*.json")) if context_root.is_dir() else []
    except OSError:
        context_paths = []
    for path in context_paths:
        try:
            context = read_advisory_snapshot(path)
            if context.get("task_id") == task_id:
                targets.add(identifier(path.stem))
        except (OSError, ValueError, TypeError):
            continue

    # Remove every discoverable context before touching the associated channels.
    for attempt_id in targets:
        _unlink_with_temporaries(_context_path(cfg, attempt_id), removed)

    for attempt_id in targets:
        _unlink_with_temporaries(_coordinator_path(cfg, attempt_id), removed)
        _unlink_with_temporaries(_local_mailbox_path(cfg.runtime_root, attempt_id), removed)
        _unlink_with_temporaries(_observed_path(cfg, attempt_id), removed)
    return removed


def cleanup_shared_progress_v3(cfg: Any, task_id: str) -> list[str]:
    """Remove the v3 Task projection under the existing progress lock."""
    try:
        task_id = identifier(task_id)
        directory = Path(cfg.shared_root) / "progress-v3" / task_id
        with exclusive(_progress_lock_path(cfg.shared_root, task_id)):
            existed = directory.exists()
            shutil.rmtree(directory, ignore_errors=True)
            return [str(directory)] if existed and not directory.exists() else []
    except (OSError, RuntimeError, ValueError):
        return []
