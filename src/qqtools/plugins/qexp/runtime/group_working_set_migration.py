"""Activation evidence for the shared Group working set."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .protocol_compatibility import GROUP_WORKING_SET_CAPABILITY
from .store import atomic_replace, read_json_limited

if TYPE_CHECKING:
    from .upgrade.contracts import UpgradeStorage

_SCHEMA_LIMIT = 16 * 1024
_MARKER_LIMIT = 4096
_ActivationState = Literal["inventory", "building", "active"]


def schema_path(root: Path) -> Path:
    return root / "schema" / "version.json"


def read_schema(root: Path, *, storage: UpgradeStorage | None = None) -> dict[str, Any]:
    read = storage.read_json_limited if storage is not None else read_json_limited
    return read(schema_path(root), max_bytes=_SCHEMA_LIMIT)


def schema_capabilities(document: dict[str, Any]) -> list[str]:
    schema = document.get("schema")
    capabilities = schema.get("required_capabilities", []) if isinstance(schema, dict) else None
    if not isinstance(capabilities, list) or not all(isinstance(item, str) for item in capabilities):
        raise ValueError("qexp schema has malformed required capabilities")
    return capabilities


def _marker_path(root: Path) -> Path:
    return root / "schema" / f"{GROUP_WORKING_SET_CAPABILITY}.json"


def _marker(root: Path, state: _ActivationState) -> dict[str, Any]:
    return {"version": 1, "root": str(root), "state": state}


def _read_state(root: Path, *, storage: UpgradeStorage | None = None) -> _ActivationState | None:
    read = storage.read_json_limited if storage is not None else read_json_limited
    try:
        value = read(_marker_path(root), max_bytes=_MARKER_LIMIT)
    except FileNotFoundError:
        return None
    state = value.get("state")
    if (
        state not in ("inventory", "building", "active")
        or type(value.get("version")) is not int
        or value != _marker(root, state)
    ):
        raise ValueError("Group working-set activation marker is malformed")
    return state


def index_active(root: Path, *, storage: UpgradeStorage | None = None) -> bool:
    """Return whether the Group working set has been certified for this root."""
    state = _read_state(root, storage=storage)
    try:
        capabilities = schema_capabilities(read_schema(root, storage=storage))
    except FileNotFoundError:
        # Fresh initialization publishes its schema after the empty index and marker.
        return False
    has_fence = GROUP_WORKING_SET_CAPABILITY in capabilities
    if (state is None and has_fence) or (state == "active" and not has_fence):
        raise ValueError("Group working-set activation and schema writer fence disagree")
    return state == "active"


def prepare_activation(root: Path, *, storage: UpgradeStorage) -> None:
    """Distinguish unconverted roots from damage to an activated root."""
    if _read_state(root, storage=storage) is None:
        storage.atomic_replace(_marker_path(root), _marker(root, "inventory"))


def publish_activation(root: Path, *, storage: UpgradeStorage | None = None) -> None:
    """Publish the marker after the caller establishes complete index coverage."""
    if _read_state(root, storage=storage) != "active":
        write = storage.atomic_replace if storage is not None else atomic_replace
        write(_marker_path(root), _marker(root, "active"))


def initialize_fresh(root: Path) -> None:
    """Initialize empty partitions before publishing a new project's schema."""
    from .group_working_set import initialize_index

    initialize_index(root)
    publish_activation(root)


# QQTOOLS-COMPAT-0025: the shared schema lock orders writers with both
# transitions of the temporary name-inventory fence.
def begin_audit(root: Path, storage: UpgradeStorage) -> None:
    if _read_state(root, storage=storage) == "inventory":
        storage.atomic_replace(_marker_path(root), _marker(root, "building"))


def group_stream_fence(root: Path) -> Path:
    return _marker_path(root).with_suffix(".stream.json")


def require_group_mutation(root: Path, name: str) -> None:
    state = _read_state(root)
    if state == "inventory":
        raise RuntimeError("Project Group name inventory is in progress. Retry the Group mutation after inventory.")

    if state == "building":
        try:
            streaming = read_json_limited(group_stream_fence(root), max_bytes=4096)["group"]
        except FileNotFoundError:
            return
        if streaming == name:
            raise RuntimeError(f"Group {name!r} is being converted. Retry its mutation after this record finishes.")


def sealed_proof_path(root: Path, name: str) -> Path:
    from .paths import shared_paths

    return shared_paths(root)["upgrade"] / GROUP_WORKING_SET_CAPABILITY / "sealed" / f"{name}.json"


def require_group_append(root: Path, group: dict[str, Any]) -> None:
    if _read_state(root) != "active" and (
        group.get("admission_state") == "sealed" or sealed_proof_path(root, group["name"]).exists()
    ):
        raise RuntimeError(f"Group {group['name']!r} is sealed until Project conversion activates.")
