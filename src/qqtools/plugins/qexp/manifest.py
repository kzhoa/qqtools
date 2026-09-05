"""Batch manifest parsing and normalization for qexp submissions."""

from __future__ import annotations

import warnings
from collections.abc import Hashable
from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from .runtime.records import validate_gpu_limit, validate_identifier

_ROOT_KEYS = {"group", "defaults", "tasks"}
_GROUP_KEYS = {"workers"}
_DEFAULTS_KEYS = {"requested_gpus", "requested_cpus", "working_directory", "placement"}
_TASK_KEYS = {
    "task_id",
    "name",
    "command",
    "requested_gpus",
    "requested_cpus",
    "working_directory",
    "placement",
    "sharing_mode",
    "fallback_machines",
    "offer_after_seconds",
    "depends_on_task_ids",
}
_PLACEMENT_KEYS = {"home_machine", "sharing"}
_SHARING_KEYS = {"mode", "fallback_machines", "offer"}
_OFFER_KEYS = {"after_seconds"}
_FLAT_PLACEMENT_FIELDS = {
    "sharing_mode": ("placement.sharing.mode", "sharing_mode"),
    "fallback_machines": ("placement.sharing.fallback_machines", "fallback_machines"),
    "offer_after_seconds": ("placement.sharing.offer.after_seconds", "offer_after_seconds"),
}


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Reject duplicate keys while PyYAML still has mapping pairs available."""

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[Any, Any]:
        if not isinstance(node, MappingNode):
            raise ConstructorError(
                None, None,
                f"expected a mapping node, but found {node.id}",
                node.start_mark,
            )
        explicit_keys: set[Hashable] = set()
        for key_node, _ in node.value:
            key = "<<" if key_node.tag == "tag:yaml.org,2002:merge" else self.construct_object(
                key_node, deep=deep
            )
            if not isinstance(key, Hashable):
                raise ConstructorError(
                    "while constructing a mapping", node.start_mark,
                    "found unhashable key", key_node.start_mark,
                )
            if key in explicit_keys:
                raise ValueError(
                    f"YAML mapping contains duplicate key {key!r} "
                    f"(line {key_node.start_mark.line + 1})."
                )
            explicit_keys.add(key)
        return super().construct_mapping(node, deep=deep)


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping.")
    return value


def _optional_mapping(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    return _mapping(value, path)


def _reject_unknown(value: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{path}.{unknown[0]} is not allowed.")


def _command(value: Any, path: str) -> list[str]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{path} must be a non-empty list of strings.")
    return list(value)


def _requested_gpus(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer.")
    return value


def _requested_cpus(value: Any, path: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 1:
        raise ValueError(f"{path} must be a positive integer.")
    return value


def _working_directory(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string.")
    return value


def _home_machine(value: Any, path: str) -> str:
    if value == "current":
        return "current"
    validate_identifier(value, path)
    return value


def _sharing_mode(value: Any, path: str) -> str:
    if value not in {"private", "spillover"}:
        raise ValueError(f"{path} must be 'private' or 'spillover'.")
    return value


def _fallback(value: Any, path: str) -> str | list[str]:
    if value == "group":
        return "group"
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be 'group' or a non-empty list of machine names.")
    seen: set[str] = set()
    result: list[str] = []
    for index, item in enumerate(value):
        validate_identifier(item, f"{path}[{index}]")
        if item in seen:
            raise ValueError(f"{path} must not contain duplicate machine {item!r}.")
        seen.add(item)
        result.append(item)
    return result


def _offer_after(value: Any, path: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer or null.")
    return value


def _worker_names(value: Any, path: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list of machine names.")
    seen: set[str] = set()
    result: list[str] = []
    for index, item in enumerate(value):
        validate_identifier(item, f"{path}[{index}]")
        if item in seen:
            raise ValueError(f"{path} must not contain duplicate machine {item!r}.")
        seen.add(item)
        result.append(item)
    return result


def _worker_pool(value: Any, path: str) -> dict[str, int | None]:
    if isinstance(value, list):
        return {machine: None for machine in _worker_names(value, path)}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a list or mapping of machine names to GPU limits.")
    result: dict[str, int | None] = {}
    for machine, limit in value.items():
        validate_identifier(machine, f"{path}.{machine}")
        if machine in result:
            raise ValueError(f"{path} must not contain duplicate machine {machine!r}.")
        result[machine] = validate_gpu_limit(limit, f"{path}.{machine}")
    return result


def _workers(value: Any, path: str) -> dict[str, dict[str, Any]]:
    if value is None:
        return {}
    if isinstance(value, list):
        return {
            machine: {"scheduling_role": "primary", "gpu_limit_gpus": None}
            for machine in _worker_names(value, path)
        }
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a list or mapping of Worker pools.")
    unknown = sorted(set(value) - {"primary", "borrow"})
    if unknown:
        raise ValueError(f"{path}.{unknown[0]} is not allowed.")
    primary = _worker_pool(value.get("primary", []), f"{path}.primary")
    borrow = _worker_pool(value.get("borrow", []), f"{path}.borrow")
    result: dict[str, dict[str, Any]] = {}
    for machine, limit in primary.items():
        result[machine] = {"scheduling_role": "primary", "gpu_limit_gpus": limit}
    for machine, limit in borrow.items():
        if machine in result:
            raise ValueError(f"{path} declares machine {machine!r} as both primary and borrow.")
        result[machine] = {"scheduling_role": "borrow", "gpu_limit_gpus": limit}
    return result


def _placement(value: Any, path: str) -> dict[str, Any]:
    placement = _optional_mapping(value, path)
    _reject_unknown(placement, _PLACEMENT_KEYS, path)
    result: dict[str, Any] = {}
    if "home_machine" in placement:
        result["home_machine"] = _home_machine(placement["home_machine"], f"{path}.home_machine")
    sharing = _optional_mapping(placement.get("sharing"), f"{path}.sharing")
    _reject_unknown(sharing, _SHARING_KEYS, f"{path}.sharing")
    if "mode" in sharing:
        result["sharing_mode"] = _sharing_mode(sharing["mode"], f"{path}.sharing.mode")
    if "fallback_machines" in sharing:
        result["fallback_machines"] = _fallback(sharing["fallback_machines"], f"{path}.sharing.fallback_machines")
    offer = _optional_mapping(sharing.get("offer"), f"{path}.sharing.offer")
    _reject_unknown(offer, _OFFER_KEYS, f"{path}.sharing.offer")
    if "after_seconds" in offer:
        result["offer_after_seconds"] = _offer_after(offer["after_seconds"], f"{path}.sharing.offer.after_seconds")
    return result


def _merge_placement(defaults: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    merged = {
        "home_machine": defaults.get("home_machine", "current"),
        "sharing_mode": defaults.get("sharing_mode", "private"),
        "fallback_machines": defaults.get("fallback_machines", "group"),
        "offer_after_seconds": defaults.get("offer_after_seconds"),
    }
    merged.update(task)
    if merged["sharing_mode"] == "private":
        merged["fallback_machines"] = "group"
        merged["offer_after_seconds"] = None
    return merged


def _apply_flat_fields(entry: dict[str, Any], placement: dict[str, Any], task_path: str) -> dict[str, Any]:
    result = dict(placement)
    used: list[str] = []
    for flat_name, (nested_label, canonical) in _FLAT_PLACEMENT_FIELDS.items():
        if flat_name not in entry:
            continue
        if canonical in result:
            raise ValueError(f"{task_path} declares {nested_label} and {flat_name}.")
        used.append(flat_name)
        if flat_name == "sharing_mode":
            result[canonical] = _sharing_mode(entry[flat_name], f"{task_path}.{flat_name}")
        elif flat_name == "fallback_machines":
            result[canonical] = _fallback(entry[flat_name], f"{task_path}.{flat_name}")
        else:
            result[canonical] = _offer_after(entry[flat_name], f"{task_path}.{flat_name}")
    if used:
        label = entry.get("name") or entry.get("task_id") or task_path
        warnings.warn(
            f"{task_path} ({label}) uses deprecated flat placement fields: {', '.join(used)}.",
            FutureWarning,
            stacklevel=3,
        )
    return result


def parse_batch_manifest(
    path: Path, *, group_name: str | None
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Parse a batch-submit manifest into canonical Task specs and Worker Set additions."""
    raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader) or {}
    root = _mapping(raw, "root")
    _reject_unknown(root, _ROOT_KEYS, "root")
    group = _optional_mapping(root.get("group"), "group")
    _reject_unknown(group, _GROUP_KEYS, "group")
    if group and group_name is None:
        raise ValueError("manifest group requires --group.")
    workers = _workers(group.get("workers"), "group.workers")
    defaults = _optional_mapping(root.get("defaults"), "defaults")
    _reject_unknown(defaults, _DEFAULTS_KEYS, "defaults")
    default_placement = _placement(defaults.get("placement"), "defaults.placement")
    if default_placement.get("sharing_mode") == "private" and (
        "fallback_machines" in default_placement or "offer_after_seconds" in default_placement
    ):
        raise ValueError("defaults.placement declares private sharing with fallback or offer.")
    tasks = root.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks must be a non-empty list.")
    normalized: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(tasks):
        task_path = f"tasks[{index}]"
        entry = _mapping(raw_entry, task_path)
        _reject_unknown(entry, _TASK_KEYS, task_path)
        if "command" not in entry:
            raise ValueError(f"{task_path}.command is required.")
        task_placement = _apply_flat_fields(
            entry, _placement(entry.get("placement"), f"{task_path}.placement"), task_path
        )
        if task_placement.get("sharing_mode") == "private" and (
            "fallback_machines" in task_placement or "offer_after_seconds" in task_placement
        ):
            raise ValueError(f"{task_path} declares private sharing with fallback or offer.")
        placement = _merge_placement(default_placement, task_placement)
        item = {
            "task_id": entry.get("task_id"),
            "name": entry.get("name"),
            "command": _command(entry["command"], f"{task_path}.command"),
            "requested_gpus": _requested_gpus(
                entry.get("requested_gpus", defaults.get("requested_gpus", 1)),
                f"{task_path}.requested_gpus",
            ),
            "requested_cpus": _requested_cpus(
                entry.get("requested_cpus", defaults.get("requested_cpus")),
                f"{task_path}.requested_cpus",
            ),
            "working_directory": _working_directory(
                entry.get("working_directory", defaults.get("working_directory", str(Path.cwd()))),
                f"{task_path}.working_directory",
            ),
            "depends_on_task_ids": entry.get("depends_on_task_ids", []),
            **placement,
        }
        if item["task_id"] is not None:
            validate_identifier(item["task_id"], f"{task_path}.task_id")
        if item["name"] is not None and not isinstance(item["name"], str):
            raise ValueError(f"{task_path}.name must be a string.")
        normalized.append(item)
    return normalized, workers
