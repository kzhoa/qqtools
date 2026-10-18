"""Authoritative Task dependency validation and derived scheduling gates."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Iterator

from ..layout import is_task_dependencies_v2_root
from .locks import group_lock, sorted_group_task_locks, task_lock, task_locks
from .operation_store import operation_exists
from .records import TaskRecord, validate_identifier
from .submission_control import read_submission_state


@dataclass(frozen=True, slots=True)
class DependencyGate:
    state: str
    reasons: tuple[dict[str, str], ...] = ()

    @property
    def is_ready(self) -> bool:
        return self.state == "ready"


def normalize_dependency_ids(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("depends_on_task_ids must be a list of task IDs.")
    result = [validate_identifier(item, "depends_on_task_ids") for item in value]
    if len(set(result)) != len(result):
        raise ValueError("depends_on_task_ids must not contain duplicate task IDs.")
    return sorted(result)


def _load_dependency_task(
    cfg: object,
    task_id: str,
    task_cache: dict[str, TaskRecord | None],
) -> TaskRecord | None:
    """Read exact Task truth once, caching missing records as absent edges."""
    if task_id not in task_cache:
        from .tasks import load_task

        try:
            task = load_task(cfg, task_id)
            task_cache[task_id] = task if task.task_id == task_id else None
        except FileNotFoundError:
            task_cache[task_id] = None
    return task_cache[task_id]


def is_committed_submission_task(cfg: object, task: TaskRecord) -> bool:
    """Return whether Task truth belongs to a durably committed submission."""
    operation_id = task.submission_operation_id
    if not operation_id:
        return False
    try:
        return read_submission_state(cfg, operation_id) == "committed"
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return False


def _dependency_context(task: TaskRecord | None) -> str:
    return task.group_name if task is not None and task.group_name is not None else "<ungrouped>"


def _edge_error(
    downstream: TaskRecord,
    prerequisite_id: str,
    message: str,
    prerequisite: TaskRecord | None = None,
) -> ValueError:
    return ValueError(
        f"dependency Task {prerequisite_id!r} {message}; downstream Task {downstream.task_id!r} "
        f"(Group {_dependency_context(downstream)}), prerequisite Group {_dependency_context(prerequisite)}."
    )


def _check_no_cycle(
    cfg: object,
    candidate_ids: Iterable[str],
    task_cache: dict[str, TaskRecord | None],
) -> None:
    """Check only the dependency closure reachable from candidate Tasks."""
    states: dict[str, str] = {}

    for candidate_id in candidate_ids:
        if states.get(candidate_id) == "visited":
            continue
        candidate = task_cache.get(candidate_id)
        if candidate is None:
            continue
        states[candidate_id] = "visiting"
        stack: list[tuple[TaskRecord, Iterator[str]]] = [(candidate, iter(candidate.depends_on_task_ids))]
        while stack:
            current, dependencies = stack[-1]
            try:
                dependency_id = next(dependencies)
            except StopIteration:
                states[current.task_id] = "visited"
                stack.pop()
                continue

            dependency = _load_dependency_task(cfg, dependency_id, task_cache)
            if dependency is None:
                continue
            state = states.get(dependency_id)
            if state == "visiting":
                raise _edge_error(current, dependency_id, "creates a dependency cycle", dependency)
            if state == "visited":
                continue

            states[dependency_id] = "visiting"
            stack.append((dependency, iter(dependency.depends_on_task_ids)))


def validate_group_dependencies(
    cfg: object,
    group_name: str | None,
    candidates: Iterable[TaskRecord],
) -> None:
    """Validate candidate edges against the selected Project's Task truth.

    V1 retains the historical same-Group and ungrouped restrictions. V2
    validates exact Project-local IDs and permits every Group orientation.
    """
    candidates = list(candidates)
    is_v2 = is_task_dependencies_v2_root(cfg)
    if not is_v2 and group_name is None and any(task.depends_on_task_ids for task in candidates):
        raise ValueError("ungrouped tasks cannot declare dependencies.")
    for task in candidates:
        if task.group_name != group_name:
            raise ValueError(
                f"dependency candidate Task {task.task_id!r} belongs to Group "
                f"{_dependency_context(task)}, not {_dependency_context(next(iter(candidates), None))}."
            )

    candidate_ids = {task.task_id for task in candidates}
    if len(candidate_ids) != len(candidates):
        raise ValueError("dependency candidate Task IDs must be unique.")
    task_cache: dict[str, TaskRecord | None] = {task.task_id: task for task in candidates}
    for task in candidates:
        for dependency_id in task.depends_on_task_ids:
            dependency = _load_dependency_task(cfg, dependency_id, task_cache)
            if dependency is None:
                raise _edge_error(task, dependency_id, "does not exist in this Project")
            if not is_v2 and dependency.group_name != group_name:
                raise _edge_error(task, dependency_id, f"is not in Group {group_name!r}", dependency)
            if dependency_id not in candidate_ids and not is_committed_submission_task(cfg, dependency):
                raise _edge_error(task, dependency_id, "has not been committed by its submission", dependency)
            if dependency_id == task.task_id:
                raise _edge_error(task, dependency_id, "cannot be the downstream Task itself", dependency)
            if (
                dependency.control.get("cleanup_operation_id")
                or dependency.control.get("cleanup_state")
                or operation_exists(cfg, "cleanup", dependency_id)
            ):
                raise _edge_error(task, dependency_id, "is being cleaned", dependency)
    _check_no_cycle(cfg, (task.task_id for task in candidates), task_cache)


def _load_gate_task(cfg: object, task_id: str) -> TaskRecord | None:
    from .tasks import load_task

    try:
        return load_task(cfg, task_id)
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return None


def dependency_gate(cfg: object, task: TaskRecord) -> DependencyGate:
    """Derive the scheduling state from direct prerequisite Task truth."""
    if not task.depends_on_task_ids:
        return DependencyGate("ready")
    is_v2 = is_task_dependencies_v2_root(cfg)
    if not is_v2 and task.group_name is None:
        return DependencyGate("invalid", ({"reason": "dependencies_without_group"},))
    invalid: list[dict[str, str]] = []
    blocked: list[dict[str, str]] = []
    waiting: list[dict[str, str]] = []
    for task_id in task.depends_on_task_ids:
        dependency = _load_gate_task(cfg, task_id)
        base = {"task_id": task_id}
        if dependency is None:
            invalid.append({**base, "reason": "missing_or_invalid"})
            continue
        if not is_v2 and dependency.group_name != task.group_name:
            invalid.append({**base, "reason": "cross_group"})
        elif (
            dependency.control.get("cleanup_operation_id")
            or dependency.control.get("cleanup_state")
            or operation_exists(cfg, "cleanup", task_id)
        ):
            invalid.append({**base, "reason": "cleanup_in_progress"})
        elif dependency.state["projection"] in {"failed", "cancelled"}:
            blocked.append({**base, "reason": dependency.state["projection"]})
        elif dependency.state["projection"] != "succeeded":
            waiting.append({**base, "reason": dependency.state["projection"]})
    if invalid:
        return DependencyGate("invalid", tuple(invalid))
    if blocked:
        return DependencyGate("blocked", tuple(blocked))
    if waiting:
        return DependencyGate("waiting", tuple(waiting))
    return DependencyGate("ready")


def resolve_affected_dependency_locks(
    cfg: object,
    candidates: Iterable[TaskRecord],
) -> tuple[list[str], list[str]]:
    """Resolve all Groups and Task IDs touched by candidate dependency edges."""
    candidate_list = list(candidates)
    task_ids = {task.task_id for task in candidate_list}
    groups = {task.group_name for task in candidate_list if task.group_name is not None}
    cache: dict[str, TaskRecord | None] = {task.task_id: task for task in candidate_list}
    for task in candidate_list:
        for dependency_id in task.depends_on_task_ids:
            dependency = _load_dependency_task(cfg, dependency_id, cache)
            if dependency is not None:
                task_ids.add(dependency.task_id)
                if dependency.group_name is not None:
                    groups.add(dependency.group_name)
    return sorted(groups), sorted(task_ids)


def resolve_dependency_lock_set(cfg: object, task: TaskRecord) -> tuple[list[str], list[str]]:
    """Resolve the downstream/direct-prerequisite lock set used by claim."""
    return resolve_affected_dependency_locks(cfg, [task])


@contextmanager
def dependency_locks(cfg: object, task: TaskRecord) -> Iterator[None]:
    """Acquire all affected Groups then Tasks in stable order.

    Claim deliberately does not acquire the Project dependency mutation fence;
    writers use that fence before entering this same Group/Task order.
    """
    if not is_task_dependencies_v2_root(cfg):
        if not task.group_name:
            with task_lock(cfg.shared_root, task.task_id):
                yield
            return
        with group_lock(cfg.shared_root, task.group_name):
            with task_locks(cfg.shared_root, sorted({task.task_id, *task.depends_on_task_ids})):
                yield
        return
    groups, task_ids = resolve_dependency_lock_set(cfg, task)
    with sorted_group_task_locks(cfg.shared_root, groups, task_ids):
        # The scheduler rereads Task truth immediately after this context is
        # entered. This local identity check closes the same race for callers
        # that only use dependency_locks directly.
        from .tasks import load_task

        reread = load_task(cfg, task.task_id)
        if reread.depends_on_task_ids != task.depends_on_task_ids:
            raise RuntimeError("Task dependency identity changed while dependency locks were acquired.")
        yield


__all__ = [
    "DependencyGate",
    "dependency_gate",
    "dependency_locks",
    "is_committed_submission_task",
    "normalize_dependency_ids",
    "resolve_affected_dependency_locks",
    "resolve_dependency_lock_set",
    "validate_group_dependencies",
]
