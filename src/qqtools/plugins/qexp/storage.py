from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .layout import (
    RootConfig,
    batch_path,
    global_batches_dir,
    global_resubmit_operations_dir,
    global_tasks_dir,
    machine_claims_active_dir,
    machine_claims_released_dir,
    machine_json_path,
    resubmit_operation_path,
    task_path,
)
from .models import (
    Batch,
    Machine,
    ResubmitOperation,
    Task,
    utc_now_iso,
    validate_task_id,
)


class CASConflict(Exception):
    """Raised when revision does not match on write."""

    def __init__(self, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"CAS conflict: expected revision {expected}, found {actual}."
        )


# ---------------------------------------------------------------------------
# Atomic JSON write (reuses v1 pattern: tmp -> fsync -> rename)
# ---------------------------------------------------------------------------


def write_atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    data = json.dumps(payload, indent=2, sort_keys=True)

    # Retry loop: network/FUSE filesystems (e.g. WSL2 /mnt/c) can race
    # between mkdir and the first write or between tmp-write and rename.
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
            return
        except FileNotFoundError as exc:
            last_exc = exc
            # Re-ensure parent (may have been lost on FUSE)
            path.parent.mkdir(parents=True, exist_ok=True)
            time.sleep(0.05 * (attempt + 1))

    raise last_exc  # type: ignore[misc]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Task persistence
# ---------------------------------------------------------------------------


def save_task(cfg: RootConfig, task: Task) -> None:
    path = task_path(cfg, task.task_id)
    write_atomic_json(path, task.to_dict())


def load_task(cfg: RootConfig, task_id: str) -> Task:
    validate_task_id(task_id)
    path = task_path(cfg, task_id)
    if not path.is_file():
        raise FileNotFoundError(f"Task {task_id} not found at {path}.")
    return Task.from_dict(read_json(path))


def cas_update_task(
    cfg: RootConfig, task: Task, expected_revision: int
) -> Task:
    path = task_path(cfg, task.task_id)
    if path.is_file():
        current = Task.from_dict(read_json(path))
        if current.meta.revision != expected_revision:
            raise CASConflict(expected_revision, current.meta.revision)

    task.meta.revision = expected_revision + 1
    task.meta.updated_at = utc_now_iso()
    task.meta.updated_by_machine = cfg.machine_name
    save_task(cfg, task)
    return task


def delete_task_file(cfg: RootConfig, task_id: str) -> None:
    path = task_path(cfg, task_id)
    if path.is_file():
        path.unlink()


# ---------------------------------------------------------------------------
# Resubmit operation persistence
# ---------------------------------------------------------------------------


def save_resubmit_operation(cfg: RootConfig, operation: ResubmitOperation) -> None:
    path = resubmit_operation_path(cfg, operation.task_id)
    write_atomic_json(path, operation.to_dict())


def load_resubmit_operation(cfg: RootConfig, task_id: str) -> ResubmitOperation:
    validate_task_id(task_id)
    path = resubmit_operation_path(cfg, task_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"Resubmit operation for task {task_id} not found at {path}."
        )
    return ResubmitOperation.from_dict(read_json(path))


def delete_resubmit_operation(cfg: RootConfig, task_id: str) -> None:
    path = resubmit_operation_path(cfg, task_id)
    if path.is_file():
        path.unlink()


def iter_resubmit_operations(cfg: RootConfig) -> list[ResubmitOperation]:
    ops_dir = global_resubmit_operations_dir(cfg)
    if not ops_dir.is_dir():
        return []
    result = []
    for p in sorted(ops_dir.glob("*.json")):
        if p.suffix == ".json" and not p.name.endswith(".tmp"):
            result.append(ResubmitOperation.from_dict(read_json(p)))
    return result


# ---------------------------------------------------------------------------
# Batch persistence
# ---------------------------------------------------------------------------


def save_batch(cfg: RootConfig, batch: Batch) -> None:
    path = batch_path(cfg, batch.batch_id)
    write_atomic_json(path, batch.to_dict())


def load_batch(cfg: RootConfig, batch_id: str) -> Batch:
    validate_task_id(batch_id)
    path = batch_path(cfg, batch_id)
    if not path.is_file():
        raise FileNotFoundError(f"Batch {batch_id} not found at {path}.")
    return Batch.from_dict(read_json(path))


def cas_update_batch(
    cfg: RootConfig, batch: Batch, expected_revision: int
) -> Batch:
    path = batch_path(cfg, batch.batch_id)
    if path.is_file():
        current = Batch.from_dict(read_json(path))
        if current.meta.revision != expected_revision:
            raise CASConflict(expected_revision, current.meta.revision)

    batch.meta.revision = expected_revision + 1
    batch.meta.updated_at = utc_now_iso()
    batch.meta.updated_by_machine = cfg.machine_name
    save_batch(cfg, batch)
    return batch


# ---------------------------------------------------------------------------
# Machine persistence
# ---------------------------------------------------------------------------


def save_machine(cfg: RootConfig, machine: Machine) -> None:
    path = machine_json_path(cfg)
    write_atomic_json(path, machine.to_dict())


def load_machine(cfg: RootConfig, machine_name: str | None = None) -> Machine:
    if machine_name and machine_name != cfg.machine_name:
        alt_cfg = RootConfig(
            shared_root=cfg.shared_root,
            project_root=cfg.project_root,
            machine_name=machine_name,
            runtime_root=cfg.runtime_root,
        )
        path = machine_json_path(alt_cfg)
    else:
        path = machine_json_path(cfg)
    if not path.is_file():
        raise FileNotFoundError(f"Machine config not found at {path}.")
    return Machine.from_dict(read_json(path))


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------


def save_claim(
    cfg: RootConfig, task_id: str, claimed_at: str, revision_at_claim: int
) -> None:
    validate_task_id(task_id)
    claim = {
        "task_id": task_id,
        "machine_name": cfg.machine_name,
        "claimed_at": claimed_at,
        "revision_at_claim": revision_at_claim,
    }
    path = machine_claims_active_dir(cfg) / f"{task_id}.json"
    write_atomic_json(path, claim)


def load_claim(
    cfg: RootConfig,
    task_id: str,
    *,
    machine_name: str | None = None,
) -> dict[str, Any]:
    validate_task_id(task_id)
    if machine_name and machine_name != cfg.machine_name:
        claim_cfg = RootConfig(
            shared_root=cfg.shared_root,
            project_root=cfg.project_root,
            machine_name=machine_name,
            runtime_root=cfg.runtime_root,
        )
    else:
        claim_cfg = cfg
    active = machine_claims_active_dir(claim_cfg) / f"{task_id}.json"
    if not active.is_file():
        raise FileNotFoundError(f"Active claim for task {task_id} not found at {active}.")
    return read_json(active)


def release_claim(
    cfg: RootConfig,
    task_id: str,
    reason: str,
    *,
    machine_name: str | None = None,
) -> None:
    validate_task_id(task_id)
    if machine_name and machine_name != cfg.machine_name:
        claim_cfg = RootConfig(
            shared_root=cfg.shared_root,
            project_root=cfg.project_root,
            machine_name=machine_name,
            runtime_root=cfg.runtime_root,
        )
    else:
        claim_cfg = cfg

    active = machine_claims_active_dir(claim_cfg) / f"{task_id}.json"
    released_dir = machine_claims_released_dir(claim_cfg)
    released_dir.mkdir(parents=True, exist_ok=True)

    claim: dict[str, Any]
    if active.is_file():
        claim = read_json(active)
        active.unlink()
    else:
        claim = {
            "task_id": task_id,
            "machine_name": machine_name or cfg.machine_name,
        }

    claim["released_at"] = utc_now_iso()
    claim["release_reason"] = reason
    write_atomic_json(released_dir / f"{task_id}.json", claim)


# ---------------------------------------------------------------------------
# Iterators
# ---------------------------------------------------------------------------


def iter_all_tasks(cfg: RootConfig) -> list[Task]:
    tasks_dir = global_tasks_dir(cfg)
    if not tasks_dir.is_dir():
        return []
    result = []
    for p in sorted(tasks_dir.glob("*.json")):
        if p.suffix == ".json" and not p.name.endswith(".tmp"):
            result.append(Task.from_dict(read_json(p)))
    return result


def iter_all_batches(cfg: RootConfig) -> list[Batch]:
    batches_dir = global_batches_dir(cfg)
    if not batches_dir.is_dir():
        return []
    result = []
    for p in sorted(batches_dir.glob("*.json")):
        if p.suffix == ".json" and not p.name.endswith(".tmp"):
            result.append(Batch.from_dict(read_json(p)))
    return result


def iter_machines(cfg: RootConfig) -> list[Machine]:
    machines_base = cfg.shared_root / "machines"
    if not machines_base.is_dir():
        return []
    result = []
    for d in sorted(machines_base.iterdir()):
        mj = d / "machine.json"
        if mj.is_file():
            result.append(Machine.from_dict(read_json(mj)))
    return result
