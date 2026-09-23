"""Batch manifest parsing and normalization for qexp submissions."""

from __future__ import annotations

import warnings
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from .runtime.records import validate_gpu_limit, validate_group_name, validate_identifier
from .task_observation import validate_tmux_override

_ROOT_KEYS = {"group", "defaults", "tasks"}
_GROUP_KEYS = {"name", "workers"}
_DEFAULTS_KEYS = {
    "requested_gpus",
    "requested_cpus",
    "working_directory",
    "placement",
    "tmux",
    "live_progress",
}
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
    "tmux",
    "live_progress",
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
                None,
                None,
                f"expected a mapping node, but found {node.id}",
                node.start_mark,
            )
        explicit_keys: set[Hashable] = set()
        for key_node, _ in node.value:
            key = "<<" if key_node.tag == "tag:yaml.org,2002:merge" else self.construct_object(key_node, deep=deep)
            if not isinstance(key, Hashable):
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found unhashable key",
                    key_node.start_mark,
                )
            if key in explicit_keys:
                raise ValueError(f"YAML mapping contains duplicate key {key!r} (line {key_node.start_mark.line + 1}).")
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


def _tmux(value: Any, path: str) -> bool | None:
    """Validate a manifest tmux value without accepting YAML number/string coercion."""
    return validate_tmux_override(value, path)


def _live_progress(value: Any, path: str) -> bool | None:
    """Validate an optional manifest live-progress choice without coercion."""
    if value is None or type(value) is bool:
        return value
    raise ValueError(f"{path} must be a boolean or null.")


def _depends_on(value: Any, path: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{path} must be a list of task IDs.")
    seen: set[str] = set()
    result: list[str] = []
    for index, item in enumerate(value):
        validate_identifier(item, f"{path}[{index}]")
        if item in seen:
            raise ValueError(f"{path} must not contain duplicate task ID {item!r}.")
        seen.add(item)
        result.append(item)
    return result


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
            machine: {"scheduling_role": "primary", "gpu_limit_gpus": None} for machine in _worker_names(value, path)
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


@dataclass(frozen=True, slots=True)
class ManifestNormalization:
    """Normalized self-contained manifest plus input provenance."""

    specs: tuple[dict[str, Any], ...]
    workers: dict[str, dict[str, Any]]
    workers_declared: bool
    group_name: str | None
    group_source: str | None
    field_sources: tuple[dict[str, str], ...]
    manifest_path: Path

    def __iter__(self):
        """Keep the historical ``normalized, workers = ...`` unpacking useful."""
        yield [dict(item) for item in self.specs]
        yield {machine: dict(declaration) for machine, declaration in self.workers.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "specs": [dict(item) for item in self.specs],
            "workers": {machine: dict(declaration) for machine, declaration in self.workers.items()},
            "workers_declared": self.workers_declared,
            "group_name": self.group_name,
            "group_source": self.group_source,
            "field_sources": [dict(item) for item in self.field_sources],
            "manifest_path": str(self.manifest_path),
        }


UNSET = object()


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


def _resolve_working_directory(value: str, *, manifest_directory: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_directory / path
    return str(path.resolve())


def _validate_lane(spec: dict[str, Any], index: int) -> None:
    gpus = spec["requested_gpus"]
    cpus = spec.get("requested_cpus")
    if gpus > 0 and cpus is not None:
        raise ValueError(f"tasks[{index}] cannot request CPUs with requested_gpus > 0.")
    if gpus == 0 and (cpus is None or cpus <= 0):
        raise ValueError(f"tasks[{index}] requested_gpus == 0 requires positive requested_cpus.")


def _validate_group_placement(placement: dict[str, Any], path: str) -> None:
    if placement.get("sharing_mode") == "private" and (
        "fallback_machines" in placement or "offer_after_seconds" in placement
    ):
        raise ValueError(f"{path} declares private sharing with fallback or offer.")


def _validate_effective_placement(placement: dict[str, Any], path: str) -> None:
    """Validate an already merged placement while allowing default ``group`` fallback."""
    if placement.get("sharing_mode") == "private" and (
        placement.get("fallback_machines") not in (None, "group") or placement.get("offer_after_seconds") is not None
    ):
        raise ValueError(f"{path} declares private sharing with fallback or offer.")


def parse_submission_manifest(
    path: Path,
    *,
    group_name: str | None | object = UNSET,
    tmux_override: bool | None | object = UNSET,
    live_progress_override: bool | None | object = UNSET,
    requested_gpus: int | None | object = UNSET,
    requested_cpus: int | None | object = UNSET,
    home_machine: str | None | object = UNSET,
    working_directory: str | Path | None | object = UNSET,
    project_directory: Path | None = None,
    invocation_cwd: Path | None = None,
) -> ManifestNormalization:
    """Parse and normalize one self-contained submission manifest.

    ``UNSET`` is deliberately distinct from ``None`` for overrides: an omitted
    CLI option must not overwrite a file value, while an explicit null in the
    manifest may clear fields whose schema permits it.
    """
    manifest_path = Path(path).expanduser()
    if invocation_cwd is None:
        invocation_cwd = Path.cwd()
    invocation_cwd = Path(invocation_cwd).expanduser().resolve()
    if not manifest_path.is_absolute():
        manifest_path = invocation_cwd / manifest_path
    manifest_path = manifest_path.resolve()
    tmux_value = validate_tmux_override(None if tmux_override is UNSET else tmux_override, "tmux_override")
    live_progress_value = _live_progress(
        None if live_progress_override is UNSET else live_progress_override,
        "live_progress_override",
    )
    for label, value in (("requested_gpus", requested_gpus), ("requested_cpus", requested_cpus)):
        if value is not UNSET and value is not None and (type(value) is not int or value < 0):
            if label == "requested_cpus" and type(value) is int and value >= 1:
                continue
            raise ValueError(f"{label} override must be a non-negative integer or null.")
    if (
        requested_cpus is not UNSET
        and requested_cpus is not None
        and (type(requested_cpus) is not int or requested_cpus < 1)
    ):
        raise ValueError("requested_cpus override must be a positive integer or null.")
    if home_machine is not UNSET and home_machine is not None:
        _home_machine(home_machine, "home_machine override")
    if (
        working_directory is not UNSET
        and working_directory is not None
        and not isinstance(working_directory, (str, Path))
    ):
        raise ValueError("working_directory override must be a path or null.")

    raw = yaml.load(manifest_path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader) or {}
    root = _mapping(raw, "root")
    _reject_unknown(root, _ROOT_KEYS, "root")
    group = _optional_mapping(root.get("group"), "group")
    _reject_unknown(group, _GROUP_KEYS, "group")
    manifest_group = group.get("name")
    if manifest_group is not None and not isinstance(manifest_group, str):
        raise ValueError("group.name must be a string or null.")
    if manifest_group is not None:
        validate_group_name(manifest_group)
    workers_declared = "workers" in group and group.get("workers") is not None
    workers = _workers(group.get("workers"), "group.workers") if workers_declared else {}
    if group_name is UNSET:
        effective_group = manifest_group
        effective_group_source = "manifest" if manifest_group is not None else "none"
    else:
        if group_name is not None:
            validate_group_name(group_name)
        effective_group = group_name
        effective_group_source = "cli" if group_name is not None else "none"
    if workers_declared and effective_group is None:
        raise ValueError("manifest group.workers requires an effective Group name.")

    defaults = _optional_mapping(root.get("defaults"), "defaults")
    _reject_unknown(defaults, _DEFAULTS_KEYS, "defaults")
    default_placement = _apply_flat_fields(
        defaults, _placement(defaults.get("placement"), "defaults.placement"), "defaults"
    )
    _validate_group_placement(default_placement, "defaults.placement")
    default_tmux = _tmux(defaults.get("tmux"), "defaults.tmux") if "tmux" in defaults else None
    default_live_progress = None
    if "live_progress" in defaults:
        default_live_progress = _live_progress(defaults["live_progress"], "defaults.live_progress")
    if "requested_gpus" in defaults:
        default_gpus = _requested_gpus(defaults["requested_gpus"], "defaults.requested_gpus")
    else:
        default_gpus = 1
    default_cpus = _requested_cpus(defaults.get("requested_cpus"), "defaults.requested_cpus")
    default_cwd = defaults.get("working_directory")
    if default_cwd is not None:
        default_cwd = _resolve_working_directory(
            _working_directory(default_cwd, "defaults.working_directory"), manifest_directory=manifest_path.parent
        )
    tasks = root.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks must be a non-empty list.")

    normalized: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    project_base = project_directory.expanduser().resolve() if project_directory is not None else manifest_path.parent
    for index, raw_entry in enumerate(tasks):
        task_path = f"tasks[{index}]"
        entry = _mapping(raw_entry, task_path)
        _reject_unknown(entry, _TASK_KEYS, task_path)
        if "command" not in entry:
            raise ValueError(f"{task_path}.command is required.")
        task_placement = _apply_flat_fields(
            entry, _placement(entry.get("placement"), f"{task_path}.placement"), task_path
        )
        _validate_group_placement(task_placement, task_path)
        placement = _merge_placement(default_placement, task_placement)
        task_tmux = _tmux(entry.get("tmux"), f"{task_path}.tmux") if "tmux" in entry else None
        task_live_progress = (
            _live_progress(entry.get("live_progress"), f"{task_path}.live_progress")
            if "live_progress" in entry
            else None
        )
        if tmux_override is not UNSET:
            effective_tmux, tmux_source = tmux_value, "cli"
        elif task_tmux is not None:
            effective_tmux, tmux_source = task_tmux, "task"
        elif default_tmux is not None:
            effective_tmux, tmux_source = default_tmux, "defaults"
        else:
            effective_tmux, tmux_source = None, "project_policy"
        if live_progress_value is not None:
            effective_live_progress, live_progress_source = live_progress_value, "cli"
        elif task_live_progress is not None:
            effective_live_progress, live_progress_source = task_live_progress, "task"
        elif default_live_progress is not None:
            effective_live_progress, live_progress_source = default_live_progress, "defaults"
        else:
            effective_live_progress, live_progress_source = None, "group"

        if requested_gpus is not UNSET:
            effective_gpus, gpu_source = requested_gpus, "cli"
        elif "requested_gpus" in entry:
            effective_gpus, gpu_source = _requested_gpus(entry["requested_gpus"], f"{task_path}.requested_gpus"), "task"
        else:
            effective_gpus, gpu_source = default_gpus, "defaults" if "requested_gpus" in defaults else "builtin"
        if effective_gpus is None or type(effective_gpus) is not int or effective_gpus < 0:
            raise ValueError(f"{task_path}.requested_gpus must be a non-negative integer.")

        if requested_cpus is not UNSET:
            effective_cpus, cpu_source = requested_cpus, "cli"
        elif "requested_cpus" in entry:
            effective_cpus, cpu_source = _requested_cpus(entry["requested_cpus"], f"{task_path}.requested_cpus"), "task"
        elif "requested_cpus" in defaults:
            effective_cpus, cpu_source = default_cpus, "defaults"
        else:
            effective_cpus, cpu_source = None, "builtin"

        if home_machine is not UNSET:
            effective_home, home_source = home_machine, "cli"
        elif "home_machine" in placement:
            effective_home, home_source = (
                placement["home_machine"],
                "task" if "home_machine" in task_placement else "defaults",
            )
        else:
            effective_home, home_source = "current", "builtin"
        if effective_home is None:
            raise ValueError(f"{task_path}.placement.home_machine cannot be null.")
        effective_home = _home_machine(effective_home, f"{task_path}.home_machine")

        if working_directory is not UNSET:
            effective_cwd, cwd_source = working_directory, "cli"
            if effective_cwd is None:
                raise ValueError("working_directory override cannot be null.")
            effective_cwd = _resolve_working_directory(str(effective_cwd), manifest_directory=invocation_cwd)
        elif "working_directory" in entry:
            effective_cwd, cwd_source = (
                _resolve_working_directory(
                    _working_directory(entry["working_directory"], f"{task_path}.working_directory"),
                    manifest_directory=manifest_path.parent,
                ),
                "task",
            )
        elif default_cwd is not None:
            effective_cwd, cwd_source = default_cwd, "defaults"
        else:
            effective_cwd, cwd_source = str(project_base), "project_policy"

        task_id = entry.get("task_id")
        if task_id is not None:
            validate_identifier(task_id, f"{task_path}.task_id")
        name = entry.get("name")
        if name is not None and not isinstance(name, str):
            raise ValueError(f"{task_path}.name must be a string or null.")
        dependencies = _depends_on(entry.get("depends_on_task_ids", []), f"{task_path}.depends_on_task_ids")
        if "depends_on_task_ids" in entry and entry["depends_on_task_ids"] is None:
            raise ValueError(f"{task_path}.depends_on_task_ids cannot be null.")

        # Placement fields use task > defaults > built-in.  A task CPU/GPU lane
        # is checked after every supported bulk override has been applied.
        item = {
            "task_id": task_id,
            "name": name,
            "command": _command(entry["command"], f"{task_path}.command"),
            "requested_gpus": effective_gpus,
            "requested_cpus": effective_cpus,
            "working_directory": effective_cwd,
            "depends_on_task_ids": dependencies,
            "tmux_override": effective_tmux,
            "home_machine": effective_home,
            "sharing_mode": placement.get("sharing_mode", "private"),
            "fallback_machines": placement.get("fallback_machines", "group"),
            "offer_after_seconds": placement.get("offer_after_seconds"),
        }
        if effective_live_progress is not None:
            item["live_progress"] = effective_live_progress
        if "sharing_mode" not in item or item["sharing_mode"] is None:
            raise ValueError(f"{task_path}.placement.sharing.mode cannot be null.")
        _sharing_mode(item["sharing_mode"], f"{task_path}.sharing.mode")
        if item["fallback_machines"] is None:
            raise ValueError(f"{task_path}.placement.sharing.fallback_machines cannot be null.")
        item["fallback_machines"] = _fallback(item["fallback_machines"], f"{task_path}.sharing.fallback_machines")
        item["offer_after_seconds"] = _offer_after(
            item["offer_after_seconds"], f"{task_path}.sharing.offer.after_seconds"
        )
        _validate_effective_placement(
            {
                "sharing_mode": item["sharing_mode"],
                "fallback_machines": item["fallback_machines"],
                "offer_after_seconds": item["offer_after_seconds"],
            },
            task_path,
        )
        _validate_lane(item, index)
        normalized.append(item)
        sources.append(
            {
                "task_id": "task" if task_id is not None else "builtin",
                "name": "task" if name is not None else "builtin",
                "command": "task",
                "requested_gpus": gpu_source,
                "requested_cpus": cpu_source,
                "working_directory": cwd_source,
                "home_machine": home_source,
                "sharing_mode": "task"
                if "sharing_mode" in task_placement
                else "defaults"
                if "sharing_mode" in default_placement
                else "builtin",
                "fallback_machines": "task"
                if "fallback_machines" in task_placement
                else "defaults"
                if "fallback_machines" in default_placement
                else "builtin",
                "offer_after_seconds": "task"
                if "offer_after_seconds" in task_placement
                else "defaults"
                if "offer_after_seconds" in default_placement
                else "builtin",
                "depends_on_task_ids": "task" if "depends_on_task_ids" in entry else "builtin",
                "tmux_override": tmux_source,
                "live_progress": live_progress_source,
            }
        )
    return ManifestNormalization(
        tuple(normalized),
        workers,
        workers_declared,
        effective_group,
        effective_group_source,
        tuple(sources),
        manifest_path,
    )


def normalize_manifest(*args: Any, **kwargs: Any) -> ManifestNormalization:
    """Descriptive alias for callers that do not use the historical name."""
    return parse_submission_manifest(*args, **kwargs)


def normalize_command_submission(
    command: list[str],
    *,
    requested_gpus: int = 1,
    requested_cpus: int | None = None,
    task_id: str | None = None,
    name: str | None = None,
    group: str | None = None,
    working_directory: str | Path | None = None,
    home_machine: str = "current",
    sharing_mode: str = "private",
    fallback_machines: str | list[str] = "group",
    offer_after_seconds: int | None = None,
    depends_on_task_ids: list[str] | None = None,
    tmux_override: bool | None = None,
    live_progress_override: bool | None = None,
    invocation_cwd: Path | None = None,
    project_directory: Path | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Normalize command mode using the same runtime-facing Task keys."""
    if not command or any(not isinstance(item, str) for item in command):
        raise ValueError("submit requires a non-empty command argv.")
    cwd = Path(invocation_cwd or Path.cwd()).expanduser().resolve()
    directory = Path(working_directory or cwd)
    if not directory.is_absolute():
        directory = cwd / directory
    directory = directory.resolve()
    if task_id is not None:
        validate_identifier(task_id, "task_id")
    if name is not None and not isinstance(name, str):
        raise ValueError("name must be a string or null.")
    validate_group_name(group)
    _requested_gpus(requested_gpus, "requested_gpus")
    _requested_cpus(requested_cpus, "requested_cpus")
    live_progress_value = _live_progress(live_progress_override, "live_progress_override")
    item = {
        "task_id": task_id,
        "name": name,
        "command": list(command),
        "requested_gpus": requested_gpus,
        "requested_cpus": requested_cpus,
        "working_directory": str(directory),
        "home_machine": _home_machine(home_machine, "home_machine"),
        "sharing_mode": _sharing_mode(sharing_mode, "sharing_mode"),
        "fallback_machines": _fallback(fallback_machines, "fallback_machines"),
        "offer_after_seconds": _offer_after(offer_after_seconds, "offer_after_seconds"),
        "depends_on_task_ids": _depends_on(depends_on_task_ids or [], "depends_on_task_ids"),
        "tmux_override": _tmux(tmux_override, "tmux_override"),
    }
    if live_progress_value is not None:
        item["live_progress"] = live_progress_value
    _validate_effective_placement(item, "command placement")
    _validate_lane(item, 0)
    return item, {
        "task_id": "task" if task_id is not None else "builtin",
        "name": "task" if name is not None else "builtin",
        "command": "task",
        "requested_gpus": "cli",
        "requested_cpus": "cli" if requested_cpus is not None else "builtin",
        "working_directory": "cli" if working_directory is not None else "builtin",
        "home_machine": "cli",
        "sharing_mode": "cli",
        "fallback_machines": "builtin",
        "offer_after_seconds": "cli" if offer_after_seconds is not None else "builtin",
        "depends_on_task_ids": "cli" if depends_on_task_ids else "builtin",
        "tmux_override": "cli" if tmux_override is not None else "project_policy",
        "live_progress": "cli" if live_progress_value is not None else "group",
    }


def parse_batch_manifest(
    path: Path, *, group_name: str | None = None, tmux_override: bool | None = None
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Compatibility wrapper retaining the programmatic batch-submit API."""
    result = parse_submission_manifest(
        Path(path),
        group_name=group_name if group_name is not None else UNSET,
        tmux_override=tmux_override if tmux_override is not None else UNSET,
    )
    return list(result.specs), result.workers
