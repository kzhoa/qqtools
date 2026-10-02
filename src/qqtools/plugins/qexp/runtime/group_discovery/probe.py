"""Closed, read-only Group-service census for machine-agent retirement."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Any, Mapping

from ..directory_capture import read_directory_entry
from ..group_namespace import GroupNotPublished, GroupPublicationUnavailable, group_directory, read_group
from ..records import validate_group_name
from ..store import read_json_limited
from . import activation, locator

_MAX_DIRECTORY_VISITS = 1024
_MAX_ACTIVATION_BYTES = 16 * 1024
_MAX_GROUP_BYTES = 8 * 1024 * 1024


def initial_group_service_probe_state() -> dict[str, Any]:
    """Return the canonical beginning of a Group-service census."""
    return {
        "mode": "start",
        "revision": None,
        "offset": 0,
        "lane_index": 0,
        "shard": 0,
    }


def validate_group_service_probe_state(value: object) -> dict[str, Any]:
    """Validate and detach a bounded Group-service census continuation."""
    if not isinstance(value, Mapping) or set(value) != {"mode", "revision", "offset", "lane_index", "shard"}:
        raise ValueError("Group-service probe state has missing or unknown fields.")
    mode = value["mode"]
    if mode not in {"start", "legacy", "locator"}:
        raise ValueError("Group-service probe mode is invalid.")
    revision = value["revision"]
    if revision is not None and (
        not isinstance(revision, str)
        or len(revision) != 64
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError("Group-service probe revision is invalid.")
    offset = value["offset"]
    lane_index = value["lane_index"]
    shard = value["shard"]
    if type(offset) is not int or not 0 <= offset <= (1 << 63) - 1:
        raise ValueError("Group-service probe offset is invalid.")
    if type(lane_index) is not int or not 0 <= lane_index <= len(locator.LANES):
        raise ValueError("Group-service probe lane index is invalid.")
    if type(shard) is not int or not 0 <= shard < locator.SHARD_COUNT:
        raise ValueError("Group-service probe shard is invalid.")
    if mode != "locator" and (lane_index != 0 or shard != 0):
        raise ValueError("legacy Group-service probe state cannot carry locator positions.")
    return {
        "mode": mode,
        "revision": revision,
        "offset": offset,
        "lane_index": lane_index,
        "shard": shard,
    }


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _directory_revision(path: Path) -> str:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"Group-service census path is not a directory: {path}")
    return _digest(
        {
            "device": info.st_dev,
            "inode": info.st_ino,
            "size": info.st_size,
            "mtime_ns": info.st_mtime_ns,
            "ctime_ns": info.st_ctime_ns,
        }
    )


def _activation_mode(root: Path) -> str:
    path = root / "schema" / "group-service.json"
    try:
        record = read_json_limited(path, max_bytes=_MAX_ACTIVATION_BYTES)
    except FileNotFoundError:
        return "legacy"
    state = record.get("state")
    if state == "active":
        if not activation.is_group_service_active(root):
            raise RuntimeError("active Group service has invalid activation evidence")
        return "locator"
    if state == "degraded":
        return "degraded"
    if state in {"preparing", "fenced", "building"}:
        return "legacy"
    raise ValueError("Group-service activation state is invalid")


def _candidate(group: str, lane: str, generation: int | None) -> dict[str, Any]:
    validate_group_name(group)
    return {"group": group, "lane": lane, "generation": generation}


def _legacy_probe(root: Path, state: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    directory = group_directory(root)
    revision = _directory_revision(directory)
    offset = state["offset"] if state["mode"] == "legacy" and state["revision"] == revision else 0
    for _ in range(_MAX_DIRECTORY_VISITS):
        name, next_offset = read_directory_entry(directory, offset)
        offset = next_offset
        continuation = {"mode": "legacy", "revision": revision, "offset": offset, "lane_index": 0, "shard": 0}
        if name is None:
            if _directory_revision(directory) != revision:
                return "pending", initial_group_service_probe_state(), None
            return "quiescent", continuation, None
        if not name.endswith(".json"):
            continue
        group = name[:-5]
        validate_group_name(group)
        try:
            read_group(root, group)
        except (GroupNotPublished, GroupPublicationUnavailable):
            return "pending", continuation, None
        return "active", continuation, _candidate(group, "legacy", None)
    return "pending", {"mode": "legacy", "revision": revision, "offset": offset, "lane_index": 0, "shard": 0}, None


def _locator_revision(root: Path) -> str:
    layout = locator.read_group_service_layout(root)
    return _digest({"identity": layout["identity"], "manifest_digest": layout["manifest_digest"]})


def _read_locator_candidate(root: Path, lane: str, shard: int, name: str) -> dict[str, Any]:
    if not name.endswith(".json"):
        raise ValueError("Group-service shard contains a malformed entry")
    digest = name[:-5]
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("Group-service locator name is malformed")
    path = locator.group_service_root(root) / lane / f"{shard:02x}" / name
    record = read_json_limited(path, max_bytes=locator.MAX_RECORD_BYTES)
    identity = record.get("identity")
    group = identity.get("group") if isinstance(identity, Mapping) else None
    if not isinstance(group, str) or hashlib.sha256(group.encode("utf-8")).hexdigest() != digest:
        raise ValueError("Group-service locator path does not match its identity")
    current = locator.read_group_locator(root, group, lane)
    if current is None or current != record:
        raise RuntimeError("Group-service locator changed during census")
    return _candidate(group, lane, current["generation"])


def _locator_probe(root: Path, state: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    revision = _locator_revision(root)
    if state["mode"] == "locator" and state["revision"] == revision:
        lane_index, shard, offset = state["lane_index"], state["shard"], state["offset"]
    else:
        lane_index, shard, offset = 0, 0, 0
    for _ in range(_MAX_DIRECTORY_VISITS):
        if lane_index == len(locator.LANES):
            if _locator_revision(root) != revision:
                return "pending", initial_group_service_probe_state(), None
            continuation = {
                "mode": "locator",
                "revision": revision,
                "offset": 0,
                "lane_index": lane_index,
                "shard": 0,
            }
            return "quiescent", continuation, None
        lane = locator.LANES[lane_index]
        directory = locator.group_service_root(root) / lane / f"{shard:02x}"
        name, next_offset = read_directory_entry(directory, offset)
        offset = next_offset
        if name is not None:
            continuation = {
                "mode": "locator",
                "revision": revision,
                "offset": offset,
                "lane_index": lane_index,
                "shard": shard,
            }
            return "active", continuation, _read_locator_candidate(root, lane, shard, name)
        shard += 1
        offset = 0
        if shard == locator.SHARD_COUNT:
            lane_index += 1
            shard = 0
    return (
        "pending",
        {"mode": "locator", "revision": revision, "offset": offset, "lane_index": lane_index, "shard": shard},
        None,
    )


def probe_group_service(root: Path, probe_state: object) -> dict[str, Any]:
    """Advance one fixed-bound read-only census without retaining descriptors."""
    root = Path(root)
    state = validate_group_service_probe_state(probe_state)
    mode = _activation_mode(root)
    if mode == "degraded":
        return {"state": "pending", "probe_state": initial_group_service_probe_state(), "candidate": None}
    if mode == "locator":
        outcome, continuation, candidate = _locator_probe(root, state)
    else:
        outcome, continuation, candidate = _legacy_probe(root, state)
    return {"state": outcome, "probe_state": continuation, "candidate": candidate}


__all__ = [
    "initial_group_service_probe_state",
    "probe_group_service",
    "validate_group_service_probe_state",
]
