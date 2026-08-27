"""Schema-5 shared and machine-local paths."""
from __future__ import annotations

from pathlib import Path


def shared_paths(root: Path) -> dict[str, Path]:
    return {
        "schema": root / "schema",
        "project": root / "project",
        "lease_policy": root / "project" / "lease-policy.json",
        "clock_observations": root / "clock-observations",
        "groups": root / "groups",
        "tasks": root / "tasks",
        "attempts": root / "attempts",
        "submissions": root / "operations" / "submissions",
        "availability": root / "operations" / "availability",
        "group_control": root / "operations" / "group-control",
        "cleanup": root / "operations" / "cleanup",
        "idempotency": root / "idempotency" / "submissions",
        "claim_archive": root / "claims" / "archive",
        "claim_pending": root / "claims" / "pending",
        "machines": root / "machines",
        "locks": root / "locks",
        "events": root / "events",
        "indexes": root / "indexes",
        "offer_deadlines": root / "indexes" / "offer-deadlines",
        "logs": root / "logs",
        "notifications": root / "notifications",
    }


def local_paths(root: Path) -> dict[str, Path]:
    return {
        "agent": root / "agent",
        "provisional": root / "reservations" / "provisional",
        "active": root / "reservations" / "active",
        "released": root / "reservations" / "released",
        "processes": root / "processes",
        "registrations": root / "process-registrations",
        "observations": root / "process-observations",
        "launch_intents": root / "launch-intents",
        "wrappers": root / "wrappers",
        "authority_diagnostics": root / "authority-diagnostics",
        "events": root / "events",
        "clock_health": root / "agent" / "clock-health.json",
        "lease_policy_cache": root / "agent" / "lease-policy.json",
        "termination_decisions": root / "termination-decisions",
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
    if kind not in {"groups", "tasks", "machines"} or identifier is None:
        raise ValueError("kind must be schema, groups, tasks, or machines with an identifier.")
    return base / kind / f"{identifier}.lock"


def attempt_control_lock_path(root: Path, attempt_id: str) -> Path:
    return local_paths(root)["locks"] / "attempt-control" / f"{attempt_id}.lock"


def machine_runtime_paths(root: Path) -> dict[str, Path]:
    root = Path(root).expanduser().resolve()
    return {
        "root": root,
        "locks": root / "locks",
        "scheduler_lock": root / "locks" / "scheduler.lock",
        "registry_lock": root / "locks" / "registry.lock",
        "reservation_lock": root / "locks" / "gpu-reservations.lock",
        "registry": root / "registry.json",
        "cursor": root / "scheduler" / "cursor.json",
        "agent": root / "agent",
        "pid": root / "agent" / "machine-agent.pid",
        "provisional": root / "reservations" / "provisional",
        "active": root / "reservations" / "active",
        "released": root / "reservations" / "released",
        "projects": root / "projects",
        "diagnostics": root / "diagnostics",
    }


def machine_project_paths(root: Path, project_id: str) -> dict[str, Path]:
    if not project_id or "/" in project_id or "\\" in project_id or ".." in project_id:
        raise ValueError("project_id is invalid.")
    project_root = machine_runtime_paths(root)["projects"] / project_id
    return {"root": project_root, **local_paths(project_root)}


def shared_log_path(root: Path, task_id: str, attempt_id: str) -> Path:
    return shared_paths(root)["logs"] / task_id / f"{attempt_id}.log"
