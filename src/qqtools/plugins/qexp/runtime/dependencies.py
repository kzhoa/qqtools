"""Authoritative Task dependency validation and derived scheduling gates."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Iterable

from .active_operations import operation_exists
from .locks import group_lock, task_lock
from .paths import shared_paths, submission_path
from .records import TaskRecord, validate_identifier
from .store import iter_json, read_json


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


def _group_tasks(cfg: object, group_name: str) -> dict[str, TaskRecord]:
    result: dict[str, TaskRecord] = {}
    for path in iter_json(shared_paths(cfg.shared_root)["tasks"]):
        task = TaskRecord.from_dict(read_json(path))
        if task.group_name == group_name:
            result[task.task_id] = task
    return result


def is_committed_submission_task(cfg: object, task: TaskRecord) -> bool:
    """Return whether Task truth belongs to a durably committed submission."""
    operation_id = task.submission_operation_id
    if not operation_id:
        return False
    try:
        submission = read_json(submission_path(cfg.shared_root, operation_id))["submission"]
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return False
    return submission.get("state") == "committed"


def _check_no_cycle(tasks: dict[str, TaskRecord]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visiting:
            raise ValueError("depends_on_task_ids creates a dependency cycle.")
        if task_id in visited:
            return
        visiting.add(task_id)
        for parent in tasks[task_id].depends_on_task_ids:
            if parent in tasks:
                visit(parent)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in tasks:
        visit(task_id)


def validate_group_dependencies(
    cfg: object, group_name: str | None, candidates: Iterable[TaskRecord],
) -> None:
    """Validate candidate dependency edges against current authoritative Group truth."""
    candidates = list(candidates)
    if group_name is None:
        if any(task.depends_on_task_ids for task in candidates):
            raise ValueError("ungrouped tasks cannot declare dependencies.")
        return
    tasks = _group_tasks(cfg, group_name)
    candidate_ids = {task.task_id for task in candidates}
    for task in candidates:
        if task.group_name != group_name:
            raise ValueError("dependency candidate has a different Group.")
        tasks[task.task_id] = task
    for task in candidates:
        for dependency_id in task.depends_on_task_ids:
            dependency = tasks.get(dependency_id)
            if dependency is None:
                raise ValueError(f"dependency Task {dependency_id!r} does not exist in Group {group_name!r}.")
            if dependency.group_name != group_name:
                raise ValueError(f"dependency Task {dependency_id!r} is not in Group {group_name!r}.")
            if (
                dependency_id not in candidate_ids
                and not is_committed_submission_task(cfg, dependency)
            ):
                raise ValueError(
                    f"dependency Task {dependency_id!r} has not been committed by its submission."
                )
            if dependency_id == task.task_id:
                raise ValueError("task cannot depend on itself.")
            if (
                dependency.control.get("cleanup_operation_id")
                or dependency.control.get("cleanup_state")
                or operation_exists(cfg, "cleanup", dependency_id)
            ):
                raise ValueError(f"dependency Task {dependency_id!r} is being cleaned.")
    _check_no_cycle(tasks)


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
