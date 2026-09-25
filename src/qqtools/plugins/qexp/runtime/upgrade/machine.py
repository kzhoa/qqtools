"""Machine-scoped discovery and fair scheduling for project upgrade coordinators."""

from __future__ import annotations

import time
from bisect import bisect_left, insort
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Event, Thread
from typing import Any

from ...agent.context import MachineRuntime, ProjectBinding
from ...runtime.store import atomic_replace, read_json
from .framework import UpgradeCoordinator, pending_upgrade_requires_completion


def _idle_blocked_project_ids(projects: list[dict[str, Any]], pending_ids: set[str]) -> set[str]:
    """Return pending upgrades that must retain an on-demand agent."""

    blocked: set[str] = set()
    by_id = {item.get("project_id"): item for item in projects}
    for project_id in pending_ids:
        status = by_id.get(project_id)
        if not isinstance(status, dict) or pending_upgrade_requires_completion(status):
            blocked.add(project_id)
    return blocked


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


class _ProbeDeadlineMap(dict[str, float]):
    """Keep the due-probe heap synchronized with process-local deadline writes."""

    def __init__(self, runtime: MachineRuntime) -> None:
        super().__init__()
        self._runtime = runtime

    def __setitem__(self, project_id: str, deadline: float) -> None:
        super().__setitem__(project_id, deadline)
        _schedule_probe_deadline(self._runtime, project_id, deadline)

    def pop(self, project_id: str, *default: float) -> float:
        present = project_id in self
        value = super().pop(project_id, *default)
        if present:
            _invalidate_probe_deadline(self._runtime, project_id)
        return value

    def clear(self) -> None:
        project_ids = tuple(self)
        super().clear()
        for project_id in project_ids:
            _invalidate_probe_deadline(self._runtime, project_id)

    def update(self, values: Any = (), **kwargs: float) -> None:
        for project_id, deadline in dict(values, **kwargs).items():
            self[project_id] = deadline


def _schedule_probe_deadline(runtime: MachineRuntime, project_id: str, deadline: float) -> None:
    schedule = getattr(runtime, "upgrade_probe_schedule", None)
    positions = getattr(runtime, "upgrade_probe_schedule_positions", None)
    if not isinstance(schedule, list) or not isinstance(positions, dict):
        return

    _remove_probe_schedule_entry(runtime, project_id)
    sequence = getattr(runtime, "upgrade_probe_schedule_sequence", 0) + 1
    runtime.upgrade_probe_schedule_sequence = sequence
    schedule.append((deadline, sequence, project_id))
    index = len(schedule) - 1
    positions[project_id] = index
    _sift_probe_schedule_up(runtime, index)


def _invalidate_probe_deadline(runtime: MachineRuntime, project_id: str) -> None:
    _remove_probe_schedule_entry(runtime, project_id)


def _sift_probe_schedule_up(runtime: MachineRuntime, index: int) -> None:
    schedule = runtime.upgrade_probe_schedule
    positions = runtime.upgrade_probe_schedule_positions
    while index:
        parent = (index - 1) // 2
        if schedule[parent] <= schedule[index]:
            return
        schedule[parent], schedule[index] = schedule[index], schedule[parent]
        positions[schedule[parent][2]] = parent
        positions[schedule[index][2]] = index
        index = parent


def _sift_probe_schedule_down(runtime: MachineRuntime, index: int) -> None:
    schedule = runtime.upgrade_probe_schedule
    positions = runtime.upgrade_probe_schedule_positions
    while True:
        left = index * 2 + 1
        right = left + 1
        smallest = index
        if left < len(schedule) and schedule[left] < schedule[smallest]:
            smallest = left
        if right < len(schedule) and schedule[right] < schedule[smallest]:
            smallest = right
        if smallest == index:
            return
        schedule[index], schedule[smallest] = schedule[smallest], schedule[index]
        positions[schedule[index][2]] = index
        positions[schedule[smallest][2]] = smallest
        index = smallest


def _remove_probe_schedule_entry(runtime: MachineRuntime, project_id: str) -> None:
    schedule = getattr(runtime, "upgrade_probe_schedule", None)
    positions = getattr(runtime, "upgrade_probe_schedule_positions", None)
    if not isinstance(schedule, list) or not isinstance(positions, dict):
        return
    index = positions.pop(project_id, None)
    if index is None:
        return
    last_entry = schedule.pop()
    if index == len(schedule):
        return
    schedule[index] = last_entry
    positions[last_entry[2]] = index
    parent = (index - 1) // 2
    if index and schedule[index] < schedule[parent]:
        _sift_probe_schedule_up(runtime, index)
    else:
        _sift_probe_schedule_down(runtime, index)


def _pop_due_probe(runtime: MachineRuntime) -> tuple[float, int, str]:
    schedule = runtime.upgrade_probe_schedule
    due_entry = schedule[0]
    _remove_probe_schedule_entry(runtime, due_entry[2])
    return due_entry


def _replace_sorted_project_id(project_ids: list[str], project_id: str, include: bool) -> None:
    index = bisect_left(project_ids, project_id)
    present = index < len(project_ids) and project_ids[index] == project_id
    if include and not present:
        insort(project_ids, project_id)
    elif not include and present:
        project_ids.pop(index)


def _update_discovery_summary_project(
    runtime: MachineRuntime,
    status: dict[str, Any],
    *,
    update_runtime_sets: bool = True,
) -> None:
    """Update one cached project and its sorted result summaries in place."""
    project_id = status.get("project_id")
    if not isinstance(project_id, str):
        return
    project_indexes = getattr(runtime, "upgrade_discovery_project_index", None)
    if not isinstance(project_indexes, dict):
        return
    project_index = project_indexes.get(project_id)
    if not isinstance(project_index, int):
        return

    projects = runtime.upgrade_discovery_project_list
    pending = bool(status.get("pending"))
    projects[project_index] = status
    runtime.upgrade_discovery_projects[project_id] = status

    pending_project_ids = runtime.upgrade_discovery_pending_project_ids
    _replace_sorted_project_id(pending_project_ids, project_id, pending)
    if update_runtime_sets:
        if pending:
            runtime.upgrade_pending_projects.add(project_id)
        else:
            runtime.upgrade_pending_projects.discard(project_id)

    runnable = pending and bool(status.get("can_run"))
    _replace_sorted_project_id(runtime.upgrade_discovery_runnable_project_ids, project_id, runnable)
    if update_runtime_sets:
        if runnable:
            runtime.upgrade_runnable_projects.add(project_id)
        else:
            runtime.upgrade_runnable_projects.discard(project_id)

    admission_blocked = pending and bool(status.get("admission_blocked"))
    if update_runtime_sets:
        if admission_blocked:
            runtime.upgrade_admission_blocked_projects.add(project_id)
        else:
            runtime.upgrade_admission_blocked_projects.discard(project_id)

    if update_runtime_sets:
        if pending and pending_upgrade_requires_completion(status):
            runtime.upgrade_idle_blocked_projects.add(project_id)
        else:
            runtime.upgrade_idle_blocked_projects.discard(project_id)

    inaccessible_positions = runtime.upgrade_discovery_inaccessible_positions
    inaccessible_index = bisect_left(inaccessible_positions, project_index)
    was_inaccessible = (
        inaccessible_index < len(inaccessible_positions) and inaccessible_positions[inaccessible_index] == project_index
    )
    is_inaccessible = status.get("state") == "inaccessible"
    if is_inaccessible:
        if was_inaccessible:
            runtime.upgrade_discovery_inaccessible_projects[inaccessible_index] = status
        else:
            inaccessible_positions.insert(inaccessible_index, project_index)
            runtime.upgrade_discovery_inaccessible_projects.insert(inaccessible_index, status)
    elif was_inaccessible:
        inaccessible_positions.pop(inaccessible_index)
        runtime.upgrade_discovery_inaccessible_projects.pop(inaccessible_index)

    summary = runtime.upgrade_discovery_summary
    inaccessible = bool(runtime.upgrade_discovery_inaccessible_projects)
    has_pending = bool(pending_project_ids)
    summary["aggregate_state"] = "inaccessible" if inaccessible else "pending" if has_pending else "complete"
    summary["all_roots_complete"] = not inaccessible and not has_pending


def _cached_discovery_result(runtime: MachineRuntime) -> dict[str, Any]:
    return dict(runtime.upgrade_discovery_summary)


def _rebuild_discovery_cache(
    runtime: MachineRuntime,
    revision: int,
    bindings: tuple[ProjectBinding, ...],
    projects: list[dict[str, Any]],
) -> None:
    project_ids = tuple(binding.project_id for binding in bindings)
    project_index = {project_id: index for index, project_id in enumerate(project_ids)}
    project_by_id = {status["project_id"]: status for status in projects if isinstance(status.get("project_id"), str)}
    ordered_projects = [project_by_id[project_id] for project_id in project_ids if project_id in project_by_id]
    # Discovery returns one status per binding. Keeping the index aligned with
    # the registry makes later status replacements constant-time.
    if len(ordered_projects) != len(project_ids):
        project_ids = tuple(project_id for project_id in project_ids if project_id in project_by_id)
        project_index = {project_id: index for index, project_id in enumerate(project_ids)}

    pending_ids = {status["project_id"] for status in ordered_projects if status.get("pending")}
    runnable_ids = {status["project_id"] for status in ordered_projects if status.get("can_run")}
    admission_blocked_ids = {status["project_id"] for status in ordered_projects if status.get("admission_blocked")}
    inaccessible_entries = [
        (index, status) for index, status in enumerate(ordered_projects) if status.get("state") == "inaccessible"
    ]

    runtime.upgrade_pending_projects = pending_ids
    runtime.upgrade_idle_blocked_projects = _idle_blocked_project_ids(ordered_projects, pending_ids)
    runtime.upgrade_runnable_projects = runnable_ids
    runtime.upgrade_admission_blocked_projects = admission_blocked_ids
    runtime.upgrade_discovery_project_ids = project_ids
    runtime.upgrade_discovery_project_index = project_index
    runtime.upgrade_discovery_project_list = ordered_projects
    runtime.upgrade_discovery_projects = project_by_id
    runtime.upgrade_discovery_pending_project_ids = sorted(pending_ids)
    runtime.upgrade_discovery_runnable_project_ids = sorted(runnable_ids)
    runtime.upgrade_discovery_inaccessible_positions = [index for index, _ in inaccessible_entries]
    runtime.upgrade_discovery_inaccessible_projects = [status for _, status in inaccessible_entries]
    runtime.upgrade_binding_by_id = {binding.project_id: binding for binding in bindings}

    runtime.upgrade_probe_schedule = []
    runtime.upgrade_probe_schedule_positions = {}
    runtime.upgrade_probe_schedule_sequence = 0
    deadlines = _ProbeDeadlineMap(runtime)
    runtime.upgrade_probe_deadlines = deadlines
    for status in ordered_projects:
        if not status.get("pending"):
            continue
        deadline = _probe_deadline(status)
        deadlines[status["project_id"]] = deadline if deadline is not None else _inaccessible_probe_deadline()

    inaccessible = bool(runtime.upgrade_discovery_inaccessible_projects)
    has_pending = bool(runtime.upgrade_discovery_pending_project_ids)
    runtime.upgrade_discovery_summary = {
        "projects": ordered_projects,
        "inaccessible_projects": runtime.upgrade_discovery_inaccessible_projects,
        "pending_project_ids": runtime.upgrade_discovery_pending_project_ids,
        "runnable_project_ids": runtime.upgrade_discovery_runnable_project_ids,
        "aggregate_state": "inaccessible" if inaccessible else "pending" if has_pending else "complete",
        "all_roots_complete": not inaccessible and not has_pending,
        "discovery_source": "machine_registry",
    }
    runtime.upgrade_discovery_complete = True
    # Publish the registry revision last.  Readers that could not acquire the
    # discovery lock treat the cache as unknown for the whole rebuild, so they
    # can never accept a new revision with partially replaced cache contents.
    runtime.upgrade_registry_revision = revision


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
        owner = getattr(self.runtime, "group_source_owner", None)
        if owner is not None:
            owner.shutdown()
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
        project = {**base, "state": runtime.binding_state(binding), "upgrade": status}
        if status.get("state") == "inaccessible":
            inaccessible.append(project)
        projects.append(project)
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
    """Serialize cache mutation while letting hot readers reuse a busy cache."""
    acquired = runtime.upgrade_discovery_lock.acquire(blocking=force)
    if not acquired:
        if getattr(runtime, "upgrade_discovery_complete", False):
            current_revision = runtime.load_registry_snapshot()[0]
            if (
                not runtime.upgrade_discovery_unknown
                and getattr(runtime, "upgrade_registry_revision", None) == current_revision
            ):
                return _cached_discovery_result(runtime)
            runtime.upgrade_discovery_unknown = True
            stale = _cached_discovery_result(runtime)
            stale["discovery_stale"] = True
            stale["aggregate_state"] = "unknown"
            stale["all_roots_complete"] = False
            return stale
        runtime.upgrade_discovery_lock.acquire()
        acquired = True
    try:
        runtime.upgrade_discovery_unknown = True
        while True:
            starting_revision = runtime.load_registry_snapshot()[0]
            result = _discover_registered_upgrades_locked(runtime, force=force)
            if runtime.load_registry_snapshot()[0] == starting_revision:
                runtime.upgrade_discovery_unknown = False
                return result
            force = True
    finally:
        if acquired:
            runtime.upgrade_discovery_lock.release()


def _discover_registered_upgrades_locked(runtime: MachineRuntime, *, force: bool = False) -> dict[str, Any]:
    """Perform startup or binding-change discovery and cache only bounded metadata."""
    revision, bindings = runtime.load_registry_snapshot()
    if (
        not force
        and getattr(runtime, "upgrade_discovery_complete", False)
        and getattr(runtime, "upgrade_registry_revision", None) == revision
    ):
        deadlines = runtime.upgrade_probe_deadlines
        schedule = runtime.upgrade_probe_schedule
        binding_by_id = runtime.upgrade_binding_by_id
        pending_ids = runtime.upgrade_pending_projects
        probe_budget = max(0, runtime.upgrade_probe_budget)
        probes = 0
        stale_entries = 0
        now = time.monotonic()
        while schedule and probes < probe_budget and stale_entries < probe_budget:
            deadline, _sequence, project_id = schedule[0]
            if deadline > now:
                break
            _pop_due_probe(runtime)
            if deadlines.get(project_id) != deadline or project_id not in pending_ids:
                stale_entries += 1
                if project_id not in pending_ids:
                    deadlines.pop(project_id, None)
                continue
            probes += 1
            binding = binding_by_id.get(project_id)
            if binding is None:
                continue
            try:
                coordinator = UpgradeCoordinator(binding.root_config())
                status = coordinator.status()
                if status.get("state") == "idle":
                    # An inaccessible first probe has no project journal to consult.  Re-run the
                    # bounded applicability check after storage recovers instead of caching idle.
                    status = coordinator.discover()
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                deadlines[project_id] = _inaccessible_probe_deadline()
                continue
            status["project_id"] = project_id
            _update_discovery_summary_project(runtime, status)
            if not status.get("pending"):
                deadlines.pop(project_id, None)
            else:
                deadlines[project_id] = _probe_deadline(status) or time.monotonic()
        return _cached_discovery_result(runtime)

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
        status["project_id"] = binding.project_id
        projects.append(status)
    _rebuild_discovery_cache(runtime, revision, bindings, projects)
    return _cached_discovery_result(runtime)


def advance_registered_upgrades(
    runtime: MachineRuntime,
    *,
    force_discovery: bool = False,
    budget: MachineUpgradeBudget | None = None,
) -> dict[str, Any]:
    """Advance upgrades without allowing an older discovery to replace a newer cache."""
    with runtime.upgrade_discovery_lock:
        runtime.upgrade_discovery_unknown = True
        starting_revision = runtime.load_registry_snapshot()[0]
        result = _advance_registered_upgrades_locked(runtime, force_discovery=force_discovery, budget=budget)
        while runtime.load_registry_snapshot()[0] != starting_revision:
            starting_revision = runtime.load_registry_snapshot()[0]
            _discover_registered_upgrades_locked(runtime, force=True)
        runtime.upgrade_discovery_unknown = False
        return result


def _advance_registered_upgrades_locked(
    runtime: MachineRuntime,
    *,
    force_discovery: bool = False,
    budget: MachineUpgradeBudget | None = None,
) -> dict[str, Any]:
    """Discover applicable work and advance pending projects in fair bounded slices."""
    budget = budget or MachineUpgradeBudget()
    discovery = _discover_registered_upgrades_locked(runtime, force=force_discovery)
    _revision, bindings = runtime.load_registry()
    pending_ids: set[str] = set(discovery.get("pending_project_ids", ()))
    discovered_projects = list(discovery.get("projects", ()))
    inaccessible_projects = list(discovery.get("inaccessible_projects", ()))
    if not inaccessible_projects:
        inaccessible_projects = [item for item in discovered_projects if item.get("state") == "inaccessible"]
    discovery_facts = {
        "inaccessible_projects": inaccessible_projects,
        "aggregate_state": "inaccessible" if inaccessible_projects else "pending" if pending_ids else "complete",
        "all_roots_complete": not inaccessible_projects and not pending_ids,
        "discovery_boundary": "locally_registered_bindings",
    }
    if not pending_ids:
        runtime.upgrade_pending_projects = set()
        runtime.upgrade_idle_blocked_projects = set()
        return {
            "projects": [],
            "slices": 0,
            "pending_project_ids": [],
            "worker_state": "idle",
            "discovery_source": "machine_registry",
            **discovery_facts,
        }

    runnable_ids = set(getattr(runtime, "upgrade_runnable_projects", pending_ids))
    ordered = [
        binding for binding in bindings if binding.project_id in pending_ids and binding.project_id in runnable_ids
    ]
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
    current_projects = {
        item.get("project_id"): item for item in discovered_projects if isinstance(item.get("project_id"), str)
    }
    current_projects.update({item["project_id"]: item for item in results if isinstance(item.get("project_id"), str)})
    runtime.upgrade_idle_blocked_projects = _idle_blocked_project_ids(list(current_projects.values()), pending_ids)
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
    for item in results:
        _update_discovery_summary_project(runtime, item, update_runtime_sets=False)
    if cursor is not None:
        _save_upgrade_cursor(runtime, cursor)
    worker_state = "runnable" if any(item.get("state") in {"runnable"} for item in results) else "waiting"
    current_discovery_facts = {
        **discovery_facts,
        "aggregate_state": "inaccessible" if inaccessible_projects else "pending" if pending_ids else "complete",
        "all_roots_complete": not inaccessible_projects and not pending_ids,
    }
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
        **current_discovery_facts,
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
