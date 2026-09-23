"""Authoritative Task dependency validation and derived scheduling gates."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Iterator

from .locks import group_lock, task_lock
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


def _check_no_cycle(
    cfg: object,
    group_name: str,
    candidate_ids: Iterable[str],
    task_cache: dict[str, TaskRecord | None],
) -> None:
    """Check cycles in the candidate-reachable same-Group dependency graph."""
    states: dict[str, str] = {}

    for candidate_id in candidate_ids:
        if states.get(candidate_id) == "visited":
            continue

        candidate = task_cache[candidate_id]
        states[candidate_id] = "visiting"
        stack: list[tuple[str, Iterator[str]]] = [(candidate_id, iter(candidate.depends_on_task_ids))]
        while stack:
            task_id, dependencies = stack[-1]
            try:
                dependency_id = next(dependencies)
            except StopIteration:
                states[task_id] = "visited"
                stack.pop()
                continue

            dependency = _load_dependency_task(cfg, dependency_id, task_cache)
            if dependency is None or dependency.group_name != group_name:
                continue

            state = states.get(dependency_id)
            if state == "visiting":
                raise ValueError("depends_on_task_ids creates a dependency cycle.")
            if state == "visited":
                continue

            states[dependency_id] = "visiting"
            stack.append((dependency_id, iter(dependency.depends_on_task_ids)))


def validate_group_dependencies(
    cfg: object,
    group_name: str | None,
    candidates: Iterable[TaskRecord],
) -> None:
    """Validate candidate dependency edges against current authoritative Group truth."""
    candidates = list(candidates)
    if group_name is None:
        if any(task.depends_on_task_ids for task in candidates):
            raise ValueError("ungrouped tasks cannot declare dependencies.")
        return

    for task in candidates:
        if task.group_name != group_name:
            raise ValueError("dependency candidate has a different Group.")

    candidate_ids = {task.task_id for task in candidates}
    task_cache: dict[str, TaskRecord | None] = {task.task_id: task for task in candidates}
    for task in candidates:
        for dependency_id in task.depends_on_task_ids:
            dependency = _load_dependency_task(cfg, dependency_id, task_cache)
            if dependency is None:
                raise ValueError(f"dependency Task {dependency_id!r} does not exist in Group {group_name!r}.")
            if dependency.group_name != group_name:
                raise ValueError(f"dependency Task {dependency_id!r} is not in Group {group_name!r}.")
            if dependency_id not in candidate_ids and not is_committed_submission_task(cfg, dependency):
                raise ValueError(f"dependency Task {dependency_id!r} has not been committed by its submission.")
            if dependency_id == task.task_id:
                raise ValueError("task cannot depend on itself.")
            if (
                dependency.control.get("cleanup_operation_id")
                or dependency.control.get("cleanup_state")
                or operation_exists(cfg, "cleanup", dependency_id)
            ):
                raise ValueError(f"dependency Task {dependency_id!r} is being cleaned.")
    _check_no_cycle(cfg, group_name, (task.task_id for task in candidates), task_cache)


def dependency_gate(cfg: object, task: TaskRecord) -> DependencyGate:
    """Derive the scheduling state from direct prerequisite Task truth."""
    if not task.depends_on_task_ids:
        return DependencyGate("ready")
    if task.group_name is None:
        return DependencyGate("invalid", ({"reason": "dependencies_without_group"},))
    invalid: list[dict[str, str]] = []
    blocked: list[dict[str, str]] = []
    waiting: list[dict[str, str]] = []
    for task_id in task.depends_on_task_ids:
        try:
            from .tasks import load_task

            dependency = load_task(cfg, task_id)
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            invalid.append({"task_id": task_id, "reason": "missing_or_invalid"})
            continue
        if dependency.group_name != task.group_name:
            invalid.append({"task_id": task_id, "reason": "cross_group"})
        elif (
            dependency.control.get("cleanup_operation_id")
            or dependency.control.get("cleanup_state")
            or operation_exists(cfg, "cleanup", task_id)
        ):
            invalid.append({"task_id": task_id, "reason": "cleanup_in_progress"})
        elif dependency.state["projection"] in {"failed", "cancelled"}:
            blocked.append({"task_id": task_id, "reason": dependency.state["projection"]})
        elif dependency.state["projection"] != "succeeded":
            waiting.append({"task_id": task_id, "reason": dependency.state["projection"]})
    if invalid:
        return DependencyGate("invalid", tuple(invalid))
    if blocked:
        return DependencyGate("blocked", tuple(blocked))
    if waiting:
        return DependencyGate("waiting", tuple(waiting))
    return DependencyGate("ready")


@contextmanager
def dependency_locks(cfg: object, task: TaskRecord) -> Iterator[None]:
    """Acquire Group then the downstream/direct-prerequisite Task locks in stable order."""
    if not task.group_name:
        with task_lock(cfg.shared_root, task.task_id):
            yield
        return
    with group_lock(cfg.shared_root, task.group_name):
        task_ids = sorted({task.task_id, *task.depends_on_task_ids})
        with _task_locks(cfg, task_ids):
            yield


@contextmanager
def _task_locks(cfg: object, task_ids: list[str]) -> Iterator[None]:
    if not task_ids:
        yield
        return
    with task_lock(cfg.shared_root, task_ids[0]):
        with _task_locks(cfg, task_ids[1:]):
            yield
