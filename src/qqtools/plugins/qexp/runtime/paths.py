"""Schema-5 shared and machine-local paths."""
from __future__ import annotations

from pathlib import Path


def shared_paths(root: Path) -> dict[str, Path]:
    return {
        "schema": root / "schema",
        "project": root / "project",
        "groups": root / "groups",
        "tasks": root / "tasks",
        "attempts": root / "attempts",
        "submissions": root / "operations" / "submissions",
        "group_control": root / "operations" / "group-control",
        "cleanup": root / "operations" / "cleanup",
        "idempotency": root / "idempotency" / "submissions",
        "claim_archive": root / "claims" / "archive",
        "machines": root / "machines",
        "locks": root / "locks",
        "events": root / "events",
        "indexes": root / "indexes",
    }


def local_paths(root: Path) -> dict[str, Path]:
    return {
        "agent": root / "agent",
        "provisional": root / "reservations" / "provisional",
        "active": root / "reservations" / "active",
        "released": root / "reservations" / "released",
        "processes": root / "processes",
        "wrappers": root / "wrappers",
        "locks": root / "locks",
    }


def task_path(root: Path, task_id: str) -> Path:
    return shared_paths(root)["tasks"] / f"{task_id}.json"


def group_path(root: Path, name: str) -> Path:
    return shared_paths(root)["groups"] / f"{name}.json"


def attempt_path(root: Path, task_id: str, number: int) -> Path:
    return shared_paths(root)["attempts"] / task_id / f"{number}.json"


def submission_path(root: Path, operation_id: str) -> Path:
    return shared_paths(root)["submissions"] / f"{operation_id}.json"


def idempotency_path(root: Path, digest: str) -> Path:
    return shared_paths(root)["idempotency"] / f"{digest}.json"


def machine_path(root: Path, machine: str) -> Path:
    return shared_paths(root)["machines"] / machine / "machine.json"


def lock_path(root: Path, kind: str, identifier: str | None = None) -> Path:
    base = shared_paths(root)["locks"]
    if kind == "schema":
        return base / "schema.lock"
    if kind not in {"groups", "tasks"} or identifier is None:
        raise ValueError("kind must be schema, groups, or tasks with an identifier.")
    return base / kind / f"{identifier}.lock"
