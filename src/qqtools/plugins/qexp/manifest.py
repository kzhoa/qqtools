"""Batch manifest parsing and normalization for qexp submissions."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import yaml

from .runtime.records import validate_identifier

_ROOT_KEYS = {"group", "defaults", "tasks"}
_GROUP_KEYS = {"workers"}
_DEFAULTS_KEYS = {"requested_gpus", "working_directory", "placement"}
_TASK_KEYS = {
    "task_id",
    "name",
    "command",
    "requested_gpus",
    "working_directory",
    "placement",
    "sharing_mode",
    "fallback_machines",
    "offer_after_seconds",
}
_PLACEMENT_KEYS = {"home_machine", "sharing"}
_SHARING_KEYS = {"mode", "fallback_machines", "offer"}
_OFFER_KEYS = {"after_seconds"}
_FLAT_PLACEMENT_FIELDS = {
    "sharing_mode": ("placement.sharing.mode", "sharing_mode"),
    "fallback_machines": ("placement.sharing.fallback_machines", "fallback_machines"),
    "offer_after_seconds": ("placement.sharing.offer.after_seconds", "offer_after_seconds"),
}


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
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
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


def _workers(value: Any, path: str) -> list[str]:
    if value is None:
        return []
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


def parse_batch_manifest(path: Path, *, group_name: str | None) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse a batch-submit manifest into canonical Task specs and Worker Set additions."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
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
            "working_directory": _working_directory(
                entry.get("working_directory", defaults.get("working_directory", str(Path.cwd()))),
                f"{task_path}.working_directory",
            ),
            **placement,
        }
        if item["task_id"] is not None:
            validate_identifier(item["task_id"], f"{task_path}.task_id")
        if item["name"] is not None and not isinstance(item["name"], str):
            raise ValueError(f"{task_path}.name must be a string.")
        normalized.append(item)
    return normalized, workers
