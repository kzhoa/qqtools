"""Machine-scoped discovery and fair scheduling for project upgrade coordinators."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Event, Thread
from typing import Any

from ...machine_runtime import MachineRuntime, ProjectBinding
from ...runtime.store import atomic_replace, read_json
from .framework import UpgradeCoordinator


def _probe_deadline(status: dict[str, Any]) -> float | None:
    value = status.get("next_probe_at")
    if not isinstance(value, str):
        return None
    try:
        remaining = (datetime.fromisoformat(value.replace("Z", "+00:00")) - datetime.now(timezone.utc)).total_seconds()
    except ValueError:
        return None
    return time.monotonic() + max(0.0, remaining)


def _inaccessible_probe_deadline() -> float:
    """Bound retries for a registered root whose storage is temporarily unavailable."""
    return time.monotonic() + 5.0


@dataclass(frozen=True, slots=True)
class MachineUpgradeBudget:
    """Hard upper bounds for one machine-agent upgrade pass."""

    max_projects: int = 4
    max_slices: int = 4
    max_records: int = 256
    max_metadata_ops: int = 128
    max_io_bytes: int = 1024 * 1024
    min_pass_interval_seconds: float = 1.0

    def __post_init__(self) -> None:
        if type(self.max_projects) is not int or self.max_projects <= 0:
            raise ValueError("max_projects must be a positive integer")
        if type(self.max_slices) is not int or self.max_slices <= 0:
            raise ValueError("max_slices must be a positive integer")
        for name in ("max_records", "max_metadata_ops", "max_io_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_pass_interval_seconds <= 0:
            raise ValueError("min_pass_interval_seconds must be positive")


class MachineUpgradeWorker:
    """Demand-started worker that never occupies the scheduling-critical call path."""

    def __init__(self, runtime: MachineRuntime, budget: MachineUpgradeBudget | None = None) -> None:
        self.runtime = runtime
        self.budget = budget or MachineUpgradeBudget()
        self._stop = Event()
        self._thread = Thread(target=self._run, name="qexp-machine-upgrade", daemon=True)

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        try:
            # One worker invocation is one machine budget.  The next agent cycle may demand-start
            # another invocation; a migration can never monopolize the machine process until its
            # full history is converted.
            if not self._stop.is_set():
                advance_registered_upgrades(self.runtime, budget=self.budget)
        finally:
            # No worker or timer is retained.  The cache contains only bounded journal deadlines;
            # it is refreshed after each executed slice or at the declared next probe deadline.
            self.runtime.upgrade_next_pass_at = time.monotonic() + self.budget.min_pass_interval_seconds


def inspect_registered_upgrades(runtime: MachineRuntime) -> dict[str, Any]:
    """Read every registered binding without scanning its historical Tasks or Attempts."""
    _revision, bindings = runtime.load_registry()
    projects: list[dict[str, Any]] = []
    inaccessible: list[dict[str, Any]] = []
    for binding in bindings:
        base = {
            **binding.to_dict(),
            "discovery_source": "machine_registry",
        }
        try:
            status = UpgradeCoordinator(binding.root_config()).status()
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            status = {
                "project_id": binding.project_id,
                "shared_root": str(binding.shared_root),
                "state": "inaccessible",
                "phase": None,
                "source_protocol": None,
                "target_protocol": None,
                "pending": True,
                "admission_blocked": False,
                "migration_blocked": True,
                "last_progress_at": None,
                "blockers": [f"registered_root_inaccessible:{exc}"],
                "pause": None,
                "repair": None,
                "migrations": [],
            }
            inaccessible.append({**base, "upgrade": status})
            projects.append({**base, "state": runtime.binding_state(binding), "upgrade": status})
            continue
        projects.append({**base, "state": runtime.binding_state(binding), "upgrade": status})
    pending = [item for item in projects if item["upgrade"].get("pending")]
    return {
        "projects": projects,
        "inaccessible_projects": inaccessible,
        "aggregate_state": "inaccessible" if inaccessible else ("pending" if pending else "complete"),
        "pending_project_ids": [item["project_id"] for item in pending],
        "all_roots_complete": not inaccessible and not pending,
        "discovery_source": "machine_registry",
        "discovery_boundary": "locally_registered_bindings",
    }


def discover_registered_upgrades(runtime: MachineRuntime, *, force: bool = False) -> dict[str, Any]:
    """Perform startup or binding-change discovery and cache only bounded metadata."""
    revision, bindings = runtime.load_registry()
    if (
        not force
        and getattr(runtime, "upgrade_discovery_complete", False)
        and getattr(runtime, "upgrade_registry_revision", None) == revision
    ):
        pending_ids = set(getattr(runtime, "upgrade_pending_projects", set()))
        deadlines = getattr(runtime, "upgrade_probe_deadlines", {})
        due_ids = [project_id for project_id in sorted(pending_ids) if deadlines.get(project_id, float("inf")) <= time.monotonic()]
        for binding in (item for item in bindings if item.project_id in due_ids[: runtime.upgrade_probe_budget]):
            try:
                coordinator = UpgradeCoordinator(binding.root_config())
                status = coordinator.status()
                if status.get("state") == "idle":
                    # An inaccessible first probe has no project journal to consult.  Re-run the
                    # bounded applicability check after storage recovers instead of caching idle.
                    status = coordinator.discover()
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                deadlines[binding.project_id] = _inaccessible_probe_deadline()
                continue
            if not status.get("pending"):
                pending_ids.discard(binding.project_id)
                runtime.upgrade_runnable_projects.discard(binding.project_id)
                runtime.upgrade_admission_blocked_projects.discard(binding.project_id)
                deadlines.pop(binding.project_id, None)
            else:
                deadlines[binding.project_id] = _probe_deadline(status) or time.monotonic()
                if status.get("can_run"):
                    runtime.upgrade_runnable_projects.add(binding.project_id)
                else:
                    runtime.upgrade_runnable_projects.discard(binding.project_id)
                if status.get("admission_blocked"):
                    runtime.upgrade_admission_blocked_projects.add(binding.project_id)
                else:
                    runtime.upgrade_admission_blocked_projects.discard(binding.project_id)
        runtime.upgrade_pending_projects = pending_ids
        return {
            "pending_project_ids": sorted(pending_ids),
            "runnable_project_ids": sorted(getattr(runtime, "upgrade_runnable_projects", set())),
            "discovery_source": "machine_registry",
        }
    pending_ids: set[str] = set()
    runnable_ids: set[str] = set()
    projects: list[dict[str, Any]] = []
    for binding in bindings:
        try:
            status = UpgradeCoordinator(binding.root_config()).discover()
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            status = {
                "project_id": binding.project_id,
                "state": "inaccessible",
                "pending": True,
                "can_run": False,
                "admission_blocked": False,
                "next_probe_at": None,
                "error": str(exc),
            }
        if status.get("pending"):
            pending_ids.add(binding.project_id)
        if status.get("can_run"):
            runnable_ids.add(binding.project_id)
        projects.append(status)
    runtime.upgrade_registry_revision = revision
    runtime.upgrade_pending_projects = pending_ids
    runtime.upgrade_runnable_projects = runnable_ids
    runtime.upgrade_admission_blocked_projects = {
        status["project_id"] for status in projects if status.get("admission_blocked")
    }
    runtime.upgrade_probe_deadlines = {
        status["project_id"]: (
            _probe_deadline(status) if _probe_deadline(status) is not None else _inaccessible_probe_deadline()
        )
        for status in projects
        if status.get("pending")
    }
    runtime.upgrade_discovery_complete = True
    return {
        "projects": projects,
        "pending_project_ids": sorted(pending_ids),
        "runnable_project_ids": sorted(runnable_ids),
        "discovery_source": "machine_registry",
    }


def advance_registered_upgrades(
    runtime: MachineRuntime,
    *,
    force_discovery: bool = False,
    budget: MachineUpgradeBudget | None = None,
) -> dict[str, Any]:
    """Discover applicable work and advance pending projects in fair bounded slices."""
    budget = budget or MachineUpgradeBudget()
    discovery = discover_registered_upgrades(runtime, force=force_discovery)
    _revision, bindings = runtime.load_registry()
    pending_ids: set[str] = set(discovery.get("pending_project_ids", ()))
    if not pending_ids:
        runtime.upgrade_pending_projects = set()
        return {
            "projects": [],
            "slices": 0,
            "pending_project_ids": [],
            "worker_state": "idle",
            "discovery_source": "machine_registry",
        }

    runnable_ids = set(getattr(runtime, "upgrade_runnable_projects", pending_ids))
    ordered = [binding for binding in bindings if binding.project_id in pending_ids and binding.project_id in runnable_ids]
    cursor = _load_upgrade_cursor(runtime)
    ordered = _rotate_bindings(ordered, cursor)
    results: list[dict[str, Any]] = []
    records_used = 0
    metadata_ops_used = 0
    io_bytes_used = 0
    for binding in ordered[: budget.max_projects]:
        if len(results) >= budget.max_slices:
            break
        # A project can only start a slice when its declared bounds fit the remaining machine
        # budget.  This preserves a hard per-pass cap independent of registered-root count.
        status = UpgradeCoordinator(binding.root_config()).status()
        active = next((item for item in status.get("migrations", ()) if item.get("state") != "completed"), {})
        if (
            records_used + int(active.get("max_records_per_slice", 1)) > budget.max_records
            or metadata_ops_used + int(active.get("max_metadata_ops_per_slice", 1)) > budget.max_metadata_ops
            or io_bytes_used + int(active.get("max_io_bytes_per_slice", 1)) > budget.max_io_bytes
        ):
            continue
        try:
            result = UpgradeCoordinator(binding.root_config()).advance()
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            result = {
                "project_id": binding.project_id,
                "shared_root": str(binding.shared_root),
                "state": "repair_required",
                "pending": True,
                "admission_blocked": True,
                "migration_blocked": True,
                "phase": None,
                "source_protocol": None,
                "target_protocol": None,
                "last_progress_at": None,
                "blockers": [str(exc)],
                "migrations": [],
                "pause": None,
                "repair": None,
            }
        results.append(result)
        records_used += int(active.get("max_records_per_slice", 1))
        metadata_ops_used += int(active.get("max_metadata_ops_per_slice", 1))
        io_bytes_used += int(active.get("max_io_bytes_per_slice", 1))
        if result.get("pending"):
            pending_ids.add(binding.project_id)
        else:
            pending_ids.discard(binding.project_id)
        cursor = binding.project_id
    runtime.upgrade_pending_projects = pending_ids
    next_runnable = set(runnable_ids)
    for item in results:
        if item.get("pending") and item.get("can_run"):
            next_runnable.add(item["project_id"])
        else:
            next_runnable.discard(item["project_id"])
        deadline = _probe_deadline(item)
        if deadline is None:
            runtime.upgrade_probe_deadlines.pop(item["project_id"], None)
        else:
            runtime.upgrade_probe_deadlines[item["project_id"]] = deadline
    runtime.upgrade_runnable_projects = next_runnable & pending_ids
    for item in results:
        if item.get("admission_blocked"):
            runtime.upgrade_admission_blocked_projects.add(item["project_id"])
        else:
            runtime.upgrade_admission_blocked_projects.discard(item["project_id"])
    if cursor is not None:
        _save_upgrade_cursor(runtime, cursor)
    worker_state = "runnable" if any(item.get("state") in {"runnable"} for item in results) else "waiting"
    return {
        "projects": results,
        "slices": len(results),
        "pending_project_ids": sorted(pending_ids),
        "worker_state": worker_state,
        "budget_used": {
            "records": records_used,
            "metadata_ops": metadata_ops_used,
            "io_bytes": io_bytes_used,
            "workers": 1,
            "queued_work": max(0, len(ordered) - len(results)),
        },
        "discovery_source": "machine_registry",
    }


def _rotate_bindings(bindings: list[ProjectBinding], cursor: str | None) -> list[ProjectBinding]:
    if not bindings or cursor is None:
        return bindings
    for index, binding in enumerate(bindings):
        if binding.project_id == cursor:
            return bindings[index + 1 :] + bindings[: index + 1]
    return bindings


def _load_upgrade_cursor(runtime: MachineRuntime) -> str | None:
    path = runtime.paths["upgrade_cursor"]
    if not path.exists():
        return None
    value = read_json(path).get("cursor", {}).get("next_project_id")
    return value if isinstance(value, str) else None


def _save_upgrade_cursor(runtime: MachineRuntime, project_id: str) -> None:
    atomic_replace(
        runtime.paths["upgrade_cursor"],
        {"cursor": {"next_project_id": project_id}},
    )


__all__ = [
    "MachineUpgradeBudget",
    "MachineUpgradeWorker",
    "advance_registered_upgrades",
    "discover_registered_upgrades",
    "inspect_registered_upgrades",
]
