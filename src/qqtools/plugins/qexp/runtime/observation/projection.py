"""Durable Task observation projection and publication fencing."""

from __future__ import annotations

import os
import re
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from ..locks import exclusive
from ..paths import shared_paths, task_path
from ..protocol_compatibility import OBSERVATION_CAPABILITY
from ..records import TASK_ID_PATTERN
from ..store import atomic_replace, read_json, read_json_limited, require_json_size

if TYPE_CHECKING:
    from ..records import TaskRecord


OBSERVATION_VERSION = 1
MAX_STATE_BYTES = 64 * 1024
MAX_CANDIDATE_PAGES = 1024
MAX_CANDIDATE_BYTES = 8 * 1024 * 1024
_GENERATION_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_STATE_FIELDS = frozenset({"version", "project_id", "generation", "state", "revision", "dirty", "build"})
_STATES = frozenset({"building", "active", "degraded"})


@dataclass(frozen=True, slots=True)
class CandidateRead:
    """One bounded candidate page tied to a projection snapshot."""

    generation: str
    keys: tuple[str, ...]
    exhausted: bool
    revision: int


def observation_path(cfg: object) -> Path:
    """Return the filesystem root for the Task observation projection."""
    return shared_paths(cfg.shared_root)["indexes"] / "task-observation"


def _observation_lock_path(cfg: object) -> Path:
    return shared_paths(cfg.shared_root)["locks"] / "task-observation.lock"


def _project_id(cfg: object) -> str:
    # Keep layout and observation imports lazy: layout imports the runtime tree
    # during root initialization, while tasks import this module for writes.
    from ...layout import project_id

    return project_id(cfg.shared_root)


def _schema_has_capability(cfg: object) -> bool:
    """Read the authoritative capability marker without silently accepting corruption."""
    path = shared_paths(cfg.shared_root)["schema"] / "version.json"
    value = read_json(path)
    schema = value.get("schema")
    if not isinstance(schema, dict):
        raise ValueError("qexp schema/version.json is malformed.")
    capabilities = schema.get("required_capabilities")
    if capabilities is None:
        return False
    if not isinstance(capabilities, list) or not all(type(item) is str for item in capabilities):
        raise ValueError("qexp schema/version.json has malformed required capabilities.")
    return OBSERVATION_CAPABILITY in capabilities


def _validate_state(cfg: object, value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _STATE_FIELDS:
        raise ValueError("Task observation state has missing or unknown fields.")
    if type(value["version"]) is not int or value["version"] != OBSERVATION_VERSION:
        raise ValueError("Task observation state version is unsupported.")
    if value["project_id"] != _project_id(cfg):
        raise ValueError("Task observation state belongs to another project.")
    generation = value["generation"]
    if not isinstance(generation, str) or _GENERATION_PATTERN.fullmatch(generation) is None:
        raise ValueError("Task observation generation is invalid.")
    if value["state"] not in _STATES:
        raise ValueError("Task observation state value is invalid.")
    if type(value["revision"]) is not int or value["revision"] < 0:
        raise ValueError("Task observation revision is invalid.")
    if type(value["dirty"]) is not bool:
        raise ValueError("Task observation dirty marker is invalid.")
    if value["build"] is not None and not isinstance(value["build"], dict):
        raise ValueError("Task observation build state must be an object or null.")
    # Validate the exact encoded budget here as well as on write. This catches
    # a state that was replaced between the bounded read and its use.
    require_json_size(value, max_bytes=MAX_STATE_BYTES, record_type="task_observation_state")
    return value


def read_state(cfg: object) -> dict[str, Any] | None:
    """Read and validate the bounded observation state, or return ``None`` if absent."""
    path = observation_path(cfg) / "state.json"
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError("Task observation state must be a regular file.")
        value = read_json_limited(path, max_bytes=MAX_STATE_BYTES)
    except FileNotFoundError:
        return None
    return _validate_state(cfg, value)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory_chain(path: Path, stop: Path) -> None:
    current = path
    stop = stop.resolve()
    while True:
        _fsync_directory(current)
        if current.resolve() == stop or current.parent == current:
            return
        current = current.parent


def write_state(cfg: object, state: dict[str, Any]) -> None:
    """Atomically persist a validated, bounded observation state."""
    value = _validate_state(cfg, state)
    root = observation_path(cfg)
    root.mkdir(parents=True, exist_ok=True)
    require_json_size(value, max_bytes=MAX_STATE_BYTES, record_type="task_observation_state")
    atomic_replace(root / "state.json", value)
    # atomic_replace fences the leaf directory. Also fence newly-created
    # observation ancestors so a clean state cannot outrun their directory
    # entries on a crash.
    _fsync_directory_chain(root.parent, cfg.shared_root)


def new_state(cfg: object, *, state: str = "building") -> dict[str, Any]:
    """Create an in-memory state record for a fresh UUID generation."""
    if state not in _STATES:
        raise ValueError("Task observation state must be building, active, or degraded.")
    return {
        "version": OBSERVATION_VERSION,
        "project_id": _project_id(cfg),
        "generation": uuid.uuid4().hex,
        "state": state,
        "revision": 0,
        "dirty": False,
        "build": None,
    }


def _new_tree(base: Path, phase: str | None = None, group: str | None = None) -> Any:
    from .tree import IndexTree

    index_tree = IndexTree
    if phase is None and group is None:
        return index_tree(base)
    return index_tree(base, phase=phase, group=group)


def _tree_root(tree: Any) -> Path:
    root = tree.root_path
    if not isinstance(root, Path):
        raise ValueError("Task observation tree did not expose a root path.")
    return root


def _ensure_tree_root(tree: Any) -> None:
    tree.ensure_root()


def _partition_key(phase: str | None, group: str | None) -> str:
    """Return the tree-owned canonical digest for one partition."""
    from .tree import partition_key

    return partition_key(phase, group)


def _catalog_tree(base: Path) -> Any:
    return _new_tree(base / "catalog")


def initialize_generation(cfg: object, state: dict[str, Any]) -> Path:
    """Create a generation directory and its mandatory empty catalog root."""
    value = _validate_state(cfg, state)
    root = observation_path(cfg)
    generations = root / "generations"
    generations.mkdir(parents=True, exist_ok=True)
    generation = generations / value["generation"]
    generation.mkdir(parents=True, exist_ok=True)
    catalog = _catalog_tree(generation)
    _ensure_tree_root(catalog)
    _fsync_directory_chain(generation, cfg.shared_root)
    return generation


def initialize_empty(cfg: object) -> None:
    """Durably activate an empty projection for a provably new root."""
    lock_path = _observation_lock_path(cfg)
    with exclusive(lock_path):
        root = observation_path(cfg)
        if root.exists() and any(root.iterdir()):
            raise RuntimeError("empty Task observation initialization requires a new projection root.")
        state = new_state(cfg, state="active")
        initialize_generation(cfg, state)
        write_state(cfg, state)


def _status(state: dict[str, Any] | None, *, degraded: bool = False) -> dict[str, Any]:
    if degraded:
        return {"state": "degraded", "generation": None, "dirty": True, "revision": None}
    if state is None:
        return {"state": "absent", "generation": None, "dirty": False, "revision": None}
    return {key: state[key] for key in ("state", "generation", "dirty", "revision")}


def inspect_observation(cfg: object) -> dict[str, Any]:
    """Return advisory state without allowing malformed state to escape as healthy."""
    try:
        return _status(read_state(cfg))
    except (OSError, TypeError, ValueError):
        return _status(None, degraded=True)


def _validate_task_id(task_id: str) -> str:
    if not isinstance(task_id, str) or not 1 <= len(task_id) <= 250 or TASK_ID_PATTERN.fullmatch(task_id) is None:
        raise ValueError("task_id is invalid or exceeds the 250-character observation bound.")
    return task_id


def _load_previous_task(cfg: object, task_id: str) -> TaskRecord | None:
    from ..records import TaskRecord

    path = task_path(cfg.shared_root, task_id)
    try:
        value = read_json(path)
    except FileNotFoundError:
        return None
    previous = TaskRecord.from_dict(value)
    if previous.task_id != task_id:
        raise ValueError("Task truth ID does not match its path.")
    return previous


def _task_routes(task: TaskRecord | None) -> tuple[tuple[str | None, str | None], ...]:
    if task is None:
        return ()
    task_id = _validate_task_id(task.task_id)
    if task_id != task.task_id:
        raise ValueError("Task truth ID is invalid.")
    phase = task.state.get("projection")
    group = task.group_name
    if not isinstance(phase, str) or not phase:
        raise ValueError("Task phase is invalid for observation indexing.")
    if group is not None:
        _validate_task_id(group)
    routes: list[tuple[str | None, str | None]] = []
    for route in ((None, None), (phase, None), (None, group), (phase, group)):
        if route not in routes:
            routes.append(route)
    return tuple(routes)


def _partition_tree(base: Path, route: tuple[str | None, str | None]) -> Any:
    return _new_tree(base, phase=route[0], group=route[1])


def _candidate_page(tree: Any) -> Any:
    return tree.candidates(None, max_candidates=1, max_pages=MAX_CANDIDATE_PAGES, max_bytes=MAX_CANDIDATE_BYTES)


def _retire_empty_partition(base: Path, tree: Any, catalog: Any, catalog_key: str) -> None:
    page = _candidate_page(tree)
    keys = tuple(page.keys)
    if keys or not page.exhausted:
        return
    root = _tree_root(tree)
    root.unlink(missing_ok=True)
    parent = root.parent
    if parent != base and parent.exists():
        parent.rmdir()
        _fsync_directory(parent.parent)
    catalog.discard(catalog_key)
    catalog.ensure_root()


def sync_task(cfg: object, state: dict[str, Any], previous: TaskRecord | None, current: TaskRecord | None) -> None:
    """Synchronize one Task while the caller holds the observation lock."""
    state = _validate_state(cfg, state)
    if previous is None and current is None:
        return
    task_id = current.task_id if current is not None else previous.task_id
    _validate_task_id(task_id)
    if previous is not None and previous.task_id != task_id:
        raise ValueError("Previous Task ID does not match the publication path.")
    if current is not None and current.task_id != task_id:
        raise ValueError("Current Task ID does not match the publication path.")
    old_routes = set(_task_routes(previous))
    new_routes = set(_task_routes(current))
    if old_routes == new_routes:
        return

    base = observation_path(cfg) / "generations" / state["generation"]
    if not base.is_dir():
        raise OSError(f"Task observation generation is missing: {base}")
    catalog = _catalog_tree(base)
    if not _tree_root(catalog).exists():
        raise OSError("Task observation catalog root is missing.")

    for route in sorted(old_routes - new_routes, key=str):
        tree = _partition_tree(base, route)
        key = _partition_key(route[0], route[1])
        if catalog.contains(key):
            if not _tree_root(tree).exists():
                raise OSError("Task observation partition root is missing.")
            tree.discard(task_id)
            _retire_empty_partition(base, tree, catalog, key)

    for route in sorted(new_routes - old_routes, key=str):
        tree = _partition_tree(base, route)
        key = _partition_key(route[0], route[1])
        if catalog.contains(key):
            if not _tree_root(tree).exists():
                raise OSError("Task observation partition root is missing.")
        else:
            _ensure_tree_root(tree)
            catalog.add(key)
        tree.add(task_id)

    _fsync_directory_chain(base, cfg.shared_root)


def _degrade_after_publication(cfg: object, state: dict[str, Any]) -> None:
    degraded = dict(state)
    degraded["state"] = "degraded"
    degraded["dirty"] = True
    # Keep the normal build cursor (or None). Maintenance owns its shape and
    # can resume/rebuild it after a publication interruption.
    try:
        write_state(cfg, degraded)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError):
        pass


def _prepare_mutation(cfg: object, task_id: str) -> tuple[dict[str, Any], bool, TaskRecord | None, dict[str, Any]]:
    initialize_state = False
    try:
        state = read_state(cfg)
    except (OSError, TypeError, ValueError):
        state = None
        state = new_state(cfg, state="degraded")
        initialize_state = True
    if state is None:
        state = new_state(cfg, state="degraded")
        initialize_state = True

    is_clean = state["state"] in {"active", "building"} and not state["dirty"]
    previous = _load_previous_task(cfg, task_id) if is_clean else None
    from ..maintenance_outbox import prepare_target_work

    descriptor = prepare_target_work(
        cfg,
        kind="task_observation",
        target_id="project",
        phase="task_publication",
        cursor={"projection_generation": state["generation"], "task_id": task_id},
    )
    if initialize_state:
        write_state(cfg, state)
    intent = dict(state)
    intent["revision"] += 1
    intent["dirty"] = True
    write_state(cfg, intent)
    return intent, is_clean, previous, descriptor


def _publish(
    cfg: object,
    task_id: str,
    current: TaskRecord | None,
    write_truth: Callable[[], None],
) -> None:
    _validate_task_id(task_id)
    if current is not None and current.task_id != task_id:
        raise ValueError("Current Task ID does not match the publication path.")
    if not _schema_has_capability(cfg):
        write_truth()
        return

    with exclusive(_observation_lock_path(cfg)):
        state, is_clean, previous, descriptor = _prepare_mutation(cfg, task_id)
        # A callback failure leaves the durable dirty intent in place.
        write_truth()
        if not is_clean:
            from ..maintenance_outbox import activate_work

            activate_work(cfg, descriptor)
            return
        try:
            sync_task(cfg, state, previous, current)
            clean = dict(state)
            clean["dirty"] = False
            write_state(cfg, clean)
            from ..maintenance_outbox import retire_work

            retire_work(
                cfg,
                kind="task_observation",
                target_id="project",
                work_generation=descriptor["identity"]["work_generation"],
                proof={"source": "task_observation_publication", "generation": clean["generation"]},
            )
        except (OSError, ValueError, TypeError, KeyError, RuntimeError):
            # Truth is committed. Publication failure must never turn a
            # successful scheduler mutation into a failed mutation.
            _degrade_after_publication(cfg, state)
            from ..maintenance_outbox import activate_work

            activate_work(cfg, descriptor)


def publish_task(cfg: object, task: TaskRecord, write_truth: Callable[[], None]) -> None:
    """Publish Task truth and synchronize its live observation memberships."""
    _publish(cfg, task.task_id, task, write_truth)


def delete_task_truth(cfg: object, task_id: str, write_truth: Callable[[], None]) -> None:
    """Publish durable Task deletion and retire its observation memberships."""
    _publish(cfg, task_id, None, write_truth)


def read_candidates(
    cfg: object,
    phase: str | None,
    group: str | None,
    after: str | None,
    expected_generation: str | None,
    max_candidates: int,
) -> CandidateRead:
    """Read one bounded indexed candidate page under a nonblocking fence."""
    try:
        has_capability = _schema_has_capability(cfg)
    except (OSError, TypeError, ValueError) as exc:
        raise _observation_error("index_unavailable", "Task observation schema is unreadable.", exc) from exc
    if not has_capability:
        raise _observation_error("index_not_ready", "Task observation capability is not active.")
    if type(max_candidates) is not int or max_candidates <= 0:
        raise _observation_error("invalid_argument", "max_candidates must be a positive integer.")

    with exclusive(_observation_lock_path(cfg), blocking=False) as acquired:
        if not acquired:
            raise _observation_error("index_unavailable", "Task observation publication is in progress.")
        state = _read_query_state(cfg)
        _check_generation(state, expected_generation)
        base = observation_path(cfg) / "generations" / state["generation"]
        catalog = _catalog_tree(base)
        catalog_root = _tree_root(catalog)
        if not catalog_root.exists():
            raise _observation_error("index_unavailable", "Task observation catalog root is missing.")
        key = _partition_key(phase, group)
        try:
            contains = catalog.contains(key)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            raise _observation_error("index_unavailable", "Task observation catalog is unreadable.", exc) from exc
        if not contains:
            return CandidateRead(state["generation"], (), True, state["revision"])
        tree = _partition_tree(base, (phase, group))
        if not _tree_root(tree).exists():
            raise _observation_error("index_unavailable", "Task observation partition root is missing.")
        try:
            page = tree.candidates(
                after,
                max_candidates=max_candidates,
                max_pages=MAX_CANDIDATE_PAGES,
                max_bytes=MAX_CANDIDATE_BYTES,
            )
            keys = tuple(page.keys)
            exhausted = page.exhausted
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            raise _observation_error("index_unavailable", "Task observation partition is unreadable.", exc) from exc
        if type(exhausted) is not bool or any(not isinstance(key, str) for key in keys):
            raise _observation_error("index_unavailable", "Task observation partition returned invalid candidates.")
        return CandidateRead(state["generation"], keys, exhausted, state["revision"])


def _read_query_state(cfg: object) -> dict[str, Any]:
    try:
        state = read_state(cfg)
    except (OSError, TypeError, ValueError) as exc:
        raise _observation_error("index_unavailable", "Task observation state is unreadable.", exc) from exc
    if state is None:
        raise _observation_error("index_not_ready", "Task observation index is not initialized.")
    if state["state"] == "degraded" or state["dirty"]:
        raise _observation_error("index_unavailable", "Task observation index is unavailable.")
    if state["state"] == "building":
        raise _observation_error("index_not_ready", "Task observation index is building.")
    return state


def _check_generation(state: dict[str, Any], expected_generation: str | None) -> None:
    if expected_generation is not None and expected_generation != state["generation"]:
        raise _observation_error("cursor_expired", "Task observation cursor generation has expired.")


def check_read(cfg: object, generation: str, revision: int) -> None:
    """Verify that a page's projection snapshot remains valid."""
    try:
        has_capability = _schema_has_capability(cfg)
    except (OSError, TypeError, ValueError) as exc:
        raise _observation_error("index_unavailable", "Task observation schema is unreadable.", exc) from exc
    if not has_capability:
        raise _observation_error("index_not_ready", "Task observation capability is not active.")
    with exclusive(_observation_lock_path(cfg), blocking=False) as acquired:
        if not acquired:
            raise _observation_error("index_unavailable", "Task observation publication is in progress.")
        state = _read_query_state(cfg)
        if state["generation"] != generation:
            raise _observation_error("cursor_expired", "Task observation cursor generation has expired.")
        if state["revision"] != revision:
            raise _observation_error("index_unavailable", "Task observation index changed during the page read.")


def _observation_error(code: str, message: str, cause: BaseException | None = None) -> Exception:
    from .api import ObservationError

    error = ObservationError(code, message)
    if cause is not None:
        error.__cause__ = cause
    return error


__all__ = [
    "CandidateRead",
    "OBSERVATION_CAPABILITY",
    "check_read",
    "delete_task_truth",
    "initialize_empty",
    "initialize_generation",
    "inspect_observation",
    "new_state",
    "observation_path",
    "publish_task",
    "read_candidates",
    "read_state",
    "sync_task",
    "write_state",
]
