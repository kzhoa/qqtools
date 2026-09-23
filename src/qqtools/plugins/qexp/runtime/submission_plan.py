"""Immutable planning and durable decoding for qexp submissions."""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from ..layout import is_cpu_lane_root, is_group_ready_members_root, is_task_dependencies_root
from ..lease import clock_capability, new_timed_offer_proof, persist_clock_observation
from ..task_live_progress import (
    build_live_progress_selection,
    disabled_live_progress_selection,
    validate_live_progress_request,
    validate_live_progress_selection,
    warn_invalid_live_progress_selection,
    warn_live_progress_policy_unavailable,
)
from ..task_observation import build_task_observation, decode_task_observation, validate_tmux_override
from .dependencies import normalize_dependency_ids
from .group_namespace import read_group, read_group_raw
from .group_observation_policy import group_identity
from .locks import group_lock
from .paths import group_path, machine_path
from .ready.group_members import assert_group_ready_members_writable
from .records import TaskSpec, new_id, normalize_group_record, validate_group_name, validate_identifier
from .store import read_json

_SUBMISSION_STATES = frozenset({"preparing", "committing", "committed", "aborted", "blocked"})


class SubmissionTargetInvalid(ValueError):
    """The requested placement cannot be admitted in the current Project."""


def _freeze(value: Any) -> Any:
    """Recursively convert JSON-like values to immutable containers."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    """Recursively make independent mutable JSON-like containers."""
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, frozenset)):
        return [_thaw(item) for item in value]
    return value


def semantic_digest(request: Mapping[str, Any]) -> str:
    """Return the compact semantic digest used by submission idempotency."""
    return hashlib.sha256(json.dumps(_thaw(request), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _resolved_context_digest(context: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(_thaw(context), sort_keys=True).encode()).hexdigest()


def _worker_additions(
    worker_set: Mapping[str, Mapping[str, Any]] | list[str] | None,
) -> dict[str, dict[str, Any]]:
    if worker_set is None:
        return {}
    if isinstance(worker_set, list):
        seen: set[str] = set()
        for index, machine in enumerate(worker_set):
            validate_identifier(machine, f"worker_set[{index}]")
            if machine in seen:
                raise ValueError(f"worker_set must not contain duplicate machine {machine!r}.")
            seen.add(machine)
        worker_set = {machine: {"scheduling_role": "primary", "gpu_limit_gpus": None} for machine in worker_set}
    if not isinstance(worker_set, Mapping):
        raise ValueError("worker_set must be a Worker declaration mapping.")
    additions: dict[str, dict[str, Any]] = {}
    for machine, declaration in worker_set.items():
        validate_identifier(machine, f"worker_set.{machine}")
        if machine in additions:
            raise ValueError(f"worker_set must not contain duplicate machine {machine!r}.")
        if not isinstance(declaration, Mapping):
            raise ValueError(f"worker_set.{machine} must be a mapping.")
        role = declaration.get("scheduling_role", "primary")
        if "borrow_limit_gpus" in declaration:
            raise ValueError(f"worker_set.{machine} has obsolete borrow_limit_gpus.")
        limit = declaration.get("gpu_limit_gpus")
        if role not in {"primary", "borrow"}:
            raise ValueError(f"worker_set.{machine}.scheduling_role is invalid.")
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError(f"worker_set.{machine}.gpu_limit_gpus must be positive or null.")
        additions[machine] = {
            "scheduling_role": role,
            "gpu_limit_gpus": limit,
        }
    return {machine: additions[machine] for machine in sorted(additions)}


def _canonical_specs(specs: list[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    canonical: list[Mapping[str, Any]] = []
    for raw in specs:
        if not isinstance(raw, Mapping):
            raise ValueError("each submission task specification must be a mapping.")
        item = dict(raw)
        item["home_machine"] = item.get("home_machine", "current")
        item["tmux_override"] = validate_tmux_override(item.get("tmux_override"))
        live_progress = validate_live_progress_request(item.get("live_progress"))
        if live_progress is None:
            item.pop("live_progress", None)
        else:
            item["live_progress"] = live_progress
        canonical.append(_freeze(item))
    return tuple(canonical)


@dataclass(frozen=True, slots=True)
class SubmissionRequest:
    """Frozen caller request used to derive a submission plan."""

    specs: tuple[Mapping[str, Any], ...]
    group_name: str | None
    kind: str
    worker_set_additions: Mapping[str, Mapping[str, Any]]
    worker_set_declared: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "specs", tuple(_freeze(item) for item in self.specs))
        object.__setattr__(self, "group_name", self.group_name)
        object.__setattr__(self, "kind", self.kind)
        object.__setattr__(self, "worker_set_additions", _freeze(self.worker_set_additions))
        object.__setattr__(self, "worker_set_declared", bool(self.worker_set_declared))

    @property
    def target_group(self) -> str | None:
        return self.group_name

    @property
    def group(self) -> str | None:
        return self.group_name

    @property
    def task_specs(self) -> tuple[Mapping[str, Any], ...]:
        return self.specs

    @property
    def worker_set(self) -> Mapping[str, Mapping[str, Any]]:
        return self.worker_set_additions

    @property
    def raw_request(self) -> dict[str, Any]:
        value = {
            "group": self.group_name,
            "tasks": [_thaw(item) for item in self.specs],
            "worker_set": {
                machine: _thaw(self.worker_set_additions[machine]) for machine in sorted(self.worker_set_additions)
            },
        }
        # Keep omitted declarations byte-compatible with historical requests;
        # explicit empty declarations remain semantically distinct.
        if self.worker_set_declared and not self.worker_set_additions:
            value["worker_set_declared"] = True
        return value

    @property
    def raw_request_digest(self) -> str:
        return semantic_digest(self.raw_request)

    @property
    def raw_digest(self) -> str:
        return self.raw_request_digest


def normalize_submission_request(
    specs: list[dict[str, Any]], *, group_name: str | None, kind: str, worker_set: list[str] | Mapping[str, Any] | None
) -> SubmissionRequest:
    """Validate and freeze raw submission semantics without reading runtime state."""
    if not specs:
        raise ValueError("submission must contain at least one task.")
    if group_name is not None and not isinstance(group_name, str):
        raise ValueError("group must be a string or null.")
    if not isinstance(kind, str) or not kind:
        raise ValueError("submission kind must be a non-empty string.")
    additions = _worker_additions(worker_set)
    return SubmissionRequest(
        specs=_canonical_specs(specs),
        group_name=group_name,
        kind=kind,
        worker_set_additions=additions,
        worker_set_declared=worker_set is not None,
    )


def legacy_submission_request_digest(request: SubmissionRequest) -> str:
    """Return the pre-tmux-policy digest for historical operation replay."""
    raw_request = request.raw_request
    for task in raw_request["tasks"]:
        task.pop("tmux_override", None)
    return semantic_digest(raw_request)


def _resolved_home(value: str | None, submitting_machine: str) -> str:
    home = "current" if value is None else value
    if home == "current":
        return submitting_machine
    validate_identifier(home, "home_machine")
    return home


def _resolved_specs(
    specs: tuple[Mapping[str, Any], ...],
    submitting_machine: str,
    working_directory: str | None = None,
    *,
    allocate_task_ids: bool = True,
) -> list[dict[str, Any]]:
    if working_directory is None:
        working_directory = str(Path.cwd())
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for frozen_raw in specs:
        raw = _thaw(frozen_raw)
        command = raw.get("command")
        if not isinstance(command, list) or not command or any(not isinstance(item, str) for item in command):
            raise ValueError("command must be a non-empty list of strings.")
        task_id = raw.get("task_id")
        if task_id is None and allocate_task_ids:
            task_id = new_id()
        if task_id is not None:
            validate_identifier(task_id, "task_id")
        if task_id is not None and task_id in seen:
            raise ValueError(f"duplicate task_id {task_id!r} in submission.")
        if task_id is not None:
            seen.add(task_id)
        home_machine = _resolved_home(raw.get("home_machine"), submitting_machine)
        result.append(
            {
                "task_id": task_id,
                "name": raw.get("name"),
                "home_machine": home_machine,
                "command": list(command),
                "working_directory": raw.get("working_directory", working_directory),
                "requested_gpus": raw.get("requested_gpus", 1),
                "requested_cpus": raw.get("requested_cpus"),
                "sharing_mode": raw.get("sharing_mode", "private"),
                "fallback_machines": raw.get("fallback_machines", "group"),
                "offer_after_seconds": raw.get("offer_after_seconds"),
                "depends_on_task_ids": normalize_dependency_ids(raw.get("depends_on_task_ids")),
                "tmux_override": validate_tmux_override(raw.get("tmux_override")),
            }
        )
    return result


def _validate_target_machine_record(cfg: Any, machine_name: str) -> None:
    """Require a current-generation shared Project record for a remote home machine."""
    identity_path = cfg.shared_root / "project" / "identity.json"
    try:
        identity = read_json(identity_path)["project"]
        stable_id = identity["project_id"]
        identity_root = Path(identity["shared_root"]).expanduser().resolve()
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"qexp project identity is malformed: {identity_path}") from exc
    if not isinstance(stable_id, str) or not stable_id or identity_root != cfg.shared_root:
        raise RuntimeError("project identity does not match the canonical shared root.")

    record_path = machine_path(cfg.shared_root, machine_name)
    if not record_path.exists():
        raise SubmissionTargetInvalid(
            f"home machine {machine_name!r} has no current-generation Project machine record."
        )
    try:
        record = read_json(record_path)
        machine = record["machine"]
    except (KeyError, TypeError, ValueError) as exc:
        raise SubmissionTargetInvalid(f"home machine {machine_name!r} has an invalid Project machine record.") from exc
    if not isinstance(machine, dict):
        raise SubmissionTargetInvalid(f"home machine {machine_name!r} has an invalid Project machine record.")
    if (
        machine.get("machine_name") != machine_name
        or machine.get("project_id") != stable_id
        or machine.get("shared_root") != str(cfg.shared_root)
        or machine.get("agent_runtime") != "machine"
    ):
        raise SubmissionTargetInvalid(
            f"home machine {machine_name!r} does not have a current-generation Project machine record."
        )


def _active_workers(group: dict[str, Any]) -> dict[str, dict[str, Any]]:
    normalize_group_record(group)
    return {
        machine: worker for machine, worker in group["group"]["worker_set"].items() if worker.get("state") == "active"
    }


def _planned_worker_set(
    group: dict[str, Any] | None, additions: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    active_workers = _active_workers(group) if group else {}
    all_workers = group["group"]["worker_set"] if group else {}
    planned = dict(active_workers)
    for machine, declaration in additions.items():
        worker = all_workers.get(machine)
        if worker is not None and worker.get("state") != "active":
            raise ValueError(f"machine {machine!r} is not a claimable Group worker.")
        if worker is not None:
            if worker.get("scheduling_role") != declaration.get("scheduling_role") or worker.get(
                "gpu_limit_gpus"
            ) != declaration.get("gpu_limit_gpus"):
                raise ValueError(
                    f"Worker {machine!r} already exists with a conflicting role or GPU limit; "
                    "use group worker controls to change it."
                )
        planned.setdefault(
            machine,
            {
                "state": "active",
                **dict(declaration),
                "state_epoch": 0,
                "added_by_operation": None,
            },
        )
    return planned


def _validate_placement_against_workers(
    resolved: list[Mapping[str, Any]], *, group_name: str | None, planned_workers: Mapping[str, Any]
) -> None:
    for item in resolved:
        home = item["home_machine"]
        if group_name is None:
            if item["sharing_mode"] != "private":
                raise SubmissionTargetInvalid("ungrouped tasks must use private placement.")
            continue
        if home not in planned_workers:
            raise SubmissionTargetInvalid(
                f"tasks home_machine {home!r} is not an active worker in Group {group_name!r}."
            )
        if item["sharing_mode"] == "private":
            continue
        fallback = item["fallback_machines"]
        if fallback == "group":
            continue
        for machine in fallback:
            if machine not in planned_workers:
                raise SubmissionTargetInvalid(
                    f"tasks fallback_machines contains {machine!r}, which is not an active "
                    f"worker in Group {group_name!r}."
                )


def _group_precondition(group: dict[str, Any] | None) -> dict[str, Any]:
    if group is None:
        return {"exists": False, "revision": None, "worker_set_epoch": None}
    return {
        "exists": True,
        "revision": group["meta"]["revision"],
        "worker_set_epoch": group["group"]["worker_set_epoch"],
    }


def _validate_group_precondition(group: dict[str, Any], precondition: Mapping[str, Any], group_name: str) -> None:
    if group["group"]["admission_state"] != "open":
        raise ValueError(f"Group {group_name!r} is sealed.")
    if group["meta"]["revision"] != precondition["revision"]:
        raise RuntimeError(f"Group {group_name!r} changed during submission.")
    if group["group"]["worker_set_epoch"] != precondition["worker_set_epoch"]:
        raise RuntimeError(f"Group {group_name!r} Worker Set changed during submission.")


def _task_spec(item: Mapping[str, Any], *, is_canonical: bool) -> TaskSpec:
    requested_gpus = item["requested_gpus"]
    requested_cpus = item.get("requested_cpus")
    if requested_gpus == 0:
        if not is_canonical:
            raise ValueError("CPU-only tasks require a canonical CPU-lane root.")
        return TaskSpec(item["command"], item["working_directory"], 0, requested_cpus, "cpu")
    if requested_cpus is not None:
        raise ValueError("GPU tasks cannot declare requested_cpus.")
    return TaskSpec(
        item["command"],
        item["working_directory"],
        requested_gpus,
        None,
        "gpu" if is_canonical else None,
    )


def _validate_planned_dependencies(specs: list[Mapping[str, Any]], group_name: str | None) -> None:
    """Reject deterministic dependency contradictions before any journal mutation."""
    if group_name is None and any(item["depends_on_task_ids"] for item in specs):
        raise ValueError("ungrouped tasks cannot declare dependencies.")
    planned = {item["task_id"]: item for item in specs if item.get("task_id") is not None}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visiting:
            raise ValueError("depends_on_task_ids creates a dependency cycle.")
        if task_id in visited:
            return
        visiting.add(task_id)
        for dependency_id in planned[task_id]["depends_on_task_ids"]:
            if dependency_id in planned:
                visit(dependency_id)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in planned:
        visit(task_id)


@dataclass(frozen=True, slots=True)
class SubmissionPlan:
    """Immutable resolved decisions used by both fresh and replay execution."""

    operation_id: str
    key: str
    kind: str
    raw_request_digest: str
    resolved_context_digest: str
    original_submitting_machine: str
    target_group: str | None
    task_ids: tuple[str, ...]
    task_specs: tuple[Mapping[str, Any], ...]
    tmux_overrides: tuple[bool | None, ...]
    create_group: bool
    group_precondition: Mapping[str, Any]
    planned_worker_set: tuple[str, ...]
    worker_set_additions: Mapping[str, Mapping[str, Any]]
    worker_set_declared: bool = False
    live_progress_selection: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_ids", tuple(self.task_ids))
        object.__setattr__(self, "task_specs", tuple(_freeze(item) for item in self.task_specs))
        overrides = tuple(validate_tmux_override(item) for item in self.tmux_overrides)
        if len(overrides) != len(self.task_ids):
            raise ValueError("submission tmux overrides must match task IDs.")
        object.__setattr__(self, "tmux_overrides", overrides)
        object.__setattr__(self, "group_precondition", _freeze(self.group_precondition))
        object.__setattr__(self, "planned_worker_set", tuple(self.planned_worker_set))
        object.__setattr__(self, "worker_set_additions", _freeze(self.worker_set_additions))
        object.__setattr__(self, "worker_set_declared", bool(self.worker_set_declared))
        selection = self.live_progress_selection
        if selection is None:
            selection = disabled_live_progress_selection(self.task_ids)
        else:
            selection = validate_live_progress_selection(_thaw(selection), self.task_ids, group_name=self.target_group)
        object.__setattr__(self, "live_progress_selection", _freeze(selection))

    @property
    def idempotency_key(self) -> str:
        return self.key

    @property
    def raw_digest(self) -> str:
        return self.raw_request_digest

    @property
    def resolved_digest(self) -> str:
        return self.resolved_context_digest

    @property
    def resolved_context(self) -> dict[str, Any]:
        return _resolved_context(self)


def _resolved_context(plan: SubmissionPlan) -> dict[str, Any]:
    task_specs = []
    for frozen in plan.task_specs:
        item = _thaw(frozen)
        task_specs.append(item)
    context = {
        "task_ids": list(plan.task_ids),
        "task_specs": task_specs,
        "create_group": plan.create_group,
        "worker_set_additions": _thaw(plan.worker_set_additions),
        "group_precondition": _thaw(plan.group_precondition),
        "planned_worker_set": list(plan.planned_worker_set),
    }
    if plan.worker_set_declared and not plan.worker_set_additions:
        context["worker_set_declared"] = True
    elif not plan.worker_set_declared and plan.worker_set_additions:
        # Nonempty declarations predate this field and remain byte-compatible.
        # False is needed only when planning synthesized the default local Worker.
        context["worker_set_declared"] = False
    return context


def _resolve_submission_plan(
    cfg: Any,
    request: SubmissionRequest,
    *,
    idempotency_key: str,
    operation_id: str,
    allocate_task_ids: bool,
    persist_clock_evidence: bool,
    acquire_group_lock: bool = True,
    policy_snapshot: Mapping[str, Any] | None = None,
) -> SubmissionPlan:
    """Resolve a plan using only caller-selected read/write side effects."""
    if not isinstance(request, SubmissionRequest):
        raise TypeError("request must be a SubmissionRequest.")
    if not isinstance(idempotency_key, str) or not idempotency_key:
        raise ValueError("idempotency_key must be a non-empty string.")
    submitting_machine = cfg.machine_name
    working_directory = str(Path.cwd())
    is_canonical = is_cpu_lane_root(cfg)
    dependencies_are_canonical = is_task_dependencies_root(cfg)
    resolved = _resolved_specs(
        request.specs,
        submitting_machine,
        working_directory,
        allocate_task_ids=allocate_task_ids,
    )
    if not dependencies_are_canonical and any(item["depends_on_task_ids"] for item in resolved):
        raise ValueError("Task dependencies require an activated task-dependencies-v1 root.")
    for item in resolved:
        _task_spec(item, is_canonical=is_canonical)
    tmux_overrides = tuple(validate_tmux_override(item.pop("tmux_override", None)) for item in resolved)
    _validate_planned_dependencies(resolved, request.group_name)
    for machine in sorted({item["home_machine"] for item in resolved if item["home_machine"] != submitting_machine}):
        _validate_target_machine_record(cfg, machine)

    group = None
    group_precondition = _group_precondition(None)
    planned_workers: dict[str, dict[str, Any]] = {}
    if request.group_name:
        lock_context = group_lock(cfg.shared_root, request.group_name) if acquire_group_lock else nullcontext()
        with lock_context:
            if is_group_ready_members_root(cfg):
                assert_group_ready_members_writable(cfg)
            group_file = group_path(cfg.shared_root, request.group_name)
            if group_file.exists():
                group_reader = read_group_raw if acquire_group_lock else read_group
                group = group_reader(cfg.shared_root, request.group_name)
            else:
                group = None
            if group is not None:
                normalize_group_record(group)
            if group is not None and group["group"]["admission_state"] != "open":
                raise ValueError(f"Group {request.group_name!r} is sealed.")
            group_precondition = _group_precondition(group)
        worker_additions = dict(request.worker_set_additions)
        if group is None and not request.worker_set_declared:
            worker_additions = {submitting_machine: {"scheduling_role": "primary", "gpu_limit_gpus": None}}
        planned_workers = _planned_worker_set(group, worker_additions)
        _validate_placement_against_workers(resolved, group_name=request.group_name, planned_workers=planned_workers)
    else:
        if request.worker_set_additions:
            raise ValueError("ungrouped submissions cannot declare a Group Worker Set.")
        worker_additions = {}
        _validate_placement_against_workers(resolved, group_name=None, planned_workers={submitting_machine: {}})

    live_progress_entries: list[dict[str, Any]] = []
    policy_diagnostic: str | None = None
    for item, raw_spec in zip(resolved, request.specs, strict=True):
        requested = validate_live_progress_request(raw_spec.get("live_progress"))
        if requested is not None:
            live_progress_entries.append(
                {
                    "task_id": item["task_id"],
                    "requested": requested,
                    "enabled": requested,
                    "source": "explicit",
                    "group_policy": None,
                }
            )
            continue

        if not request.group_name or group is None:
            source = "default"
            enabled = False
            group_policy = None
        elif not isinstance(policy_snapshot, Mapping):
            source = "unavailable"
            enabled = False
            group_policy = None
            policy_diagnostic = "Group policy snapshot was not provided"
        else:
            snapshot_status = policy_snapshot.get("status")
            if snapshot_status == "missing":
                source = "default"
                enabled = False
                group_policy = None
            elif snapshot_status == "available":
                record = policy_snapshot.get("record")
                try:
                    if not isinstance(record, Mapping) or set(record) != {
                        "version",
                        "revision",
                        "group_identity",
                        "live_progress",
                    }:
                        raise ValueError("Group live-progress policy snapshot is malformed")
                    if type(record["version"]) is not int or record["version"] != 1:
                        raise ValueError("Group live-progress policy snapshot version is unsupported")
                    identity = record["group_identity"]
                    revision = record["revision"]
                    policy_enabled = record["live_progress"]
                    current_identity = group_identity(group)
                    if not isinstance(identity, Mapping) or dict(identity) != current_identity:
                        raise ValueError("Group live-progress policy identity does not match the resolved Group")
                    if (
                        type(revision) is not int
                        or not 1 <= revision <= (1 << 63) - 1
                        or type(policy_enabled) is not bool
                    ):
                        raise ValueError("Group live-progress policy snapshot is malformed")
                except (TypeError, ValueError) as exc:
                    source = "unavailable"
                    enabled = False
                    group_policy = None
                    policy_diagnostic = str(exc)
                else:
                    source = "group"
                    enabled = policy_enabled
                    group_policy = {"group_identity": current_identity, "revision": revision}
            else:
                source = "unavailable"
                enabled = False
                group_policy = None
                reason = policy_snapshot.get("reason")
                policy_diagnostic = reason if isinstance(reason, str) and reason else "Group policy read failed"

        live_progress_entries.append(
            {
                "task_id": item["task_id"],
                "requested": None,
                "enabled": enabled,
                "source": source,
                "group_policy": group_policy,
            }
        )

    if policy_diagnostic is not None:
        warn_live_progress_policy_unavailable(policy_diagnostic)
    live_progress_selection = build_live_progress_selection(live_progress_entries, group_name=request.group_name)

    if any(item["offer_after_seconds"] is not None for item in resolved):
        capability = clock_capability(cfg)
        if not capability.is_healthy or capability.observation is None:
            raise ValueError("timed offer requires a healthy clock capability; use an immediate share instead.")
        if persist_clock_evidence:
            persist_clock_observation(cfg, capability.observation)
        for item in resolved:
            if item["offer_after_seconds"] is not None:
                deadline, proof = new_timed_offer_proof(capability.observation, item["offer_after_seconds"])
                item["offer_eligible_at"] = deadline
                item["offer_clock_evidence"] = proof

    plan = SubmissionPlan(
        operation_id=operation_id,
        key=idempotency_key,
        kind=request.kind,
        raw_request_digest=request.raw_request_digest,
        resolved_context_digest="",
        original_submitting_machine=submitting_machine,
        target_group=request.group_name,
        task_ids=tuple(item["task_id"] for item in resolved),
        task_specs=tuple(resolved),
        tmux_overrides=tmux_overrides,
        create_group=bool(request.group_name and not group_precondition["exists"]),
        group_precondition=group_precondition,
        planned_worker_set=tuple(sorted(planned_workers)),
        worker_set_additions=worker_additions,
        worker_set_declared=request.worker_set_declared,
        live_progress_selection=live_progress_selection,
    )
    context = _resolved_context(plan)
    return SubmissionPlan(
        operation_id=plan.operation_id,
        key=plan.key,
        kind=plan.kind,
        raw_request_digest=plan.raw_request_digest,
        resolved_context_digest=_resolved_context_digest(context),
        original_submitting_machine=plan.original_submitting_machine,
        target_group=plan.target_group,
        task_ids=plan.task_ids,
        task_specs=plan.task_specs,
        tmux_overrides=plan.tmux_overrides,
        create_group=plan.create_group,
        group_precondition=plan.group_precondition,
        planned_worker_set=plan.planned_worker_set,
        worker_set_additions=plan.worker_set_additions,
        worker_set_declared=plan.worker_set_declared,
        live_progress_selection=plan.live_progress_selection,
    )


def prepare_submission_plan(
    cfg: Any,
    request: SubmissionRequest,
    *,
    idempotency_key: str,
    policy_snapshot: Mapping[str, Any] | None = None,
) -> SubmissionPlan:
    """Resolve one new submission while the submission fence is held."""
    return _resolve_submission_plan(
        cfg,
        request,
        idempotency_key=idempotency_key,
        operation_id=new_id(),
        allocate_task_ids=True,
        persist_clock_evidence=True,
        policy_snapshot=policy_snapshot,
    )


def encode_submission_plan(plan: SubmissionPlan) -> dict[str, Any]:
    """Encode a new plan into the existing preparing Submission envelope."""
    if not isinstance(plan, SubmissionPlan):
        raise TypeError("plan must be a SubmissionPlan.")
    from .records import new_submission

    context = plan.resolved_context
    digest = _resolved_context_digest(context)
    if digest != plan.resolved_context_digest:
        raise RuntimeError("submission plan resolved context digest is inconsistent.")
    operation = new_submission(
        operation_id=plan.operation_id,
        kind=plan.kind,
        key=plan.key,
        raw_digest=plan.raw_request_digest,
        machine=plan.original_submitting_machine,
        target_group=plan.target_group,
        resolved_context=context,
    )
    if operation["submission"]["resolved_context_digest"] != plan.resolved_context_digest:
        raise RuntimeError("encoded submission plan resolved context digest is inconsistent.")
    operation["task_observation"] = build_task_observation(plan.task_ids, plan.tmux_overrides)
    operation["live_progress_selection"] = _thaw(plan.live_progress_selection)
    return _thaw(operation)


def _validate_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"submission {label} is invalid.")
    return value


def _validate_task_spec_for_decode(raw: Mapping[str, Any], index: int) -> dict[str, Any]:
    if "depends_on_task_ids" not in raw:
        raise RuntimeError(
            "submission operation predates the canonical task-dependencies-v1 protocol; "
            "recovery requires qqtools 1.3.15."
        )
    value = dict(raw)
    required = {
        "task_id",
        "name",
        "home_machine",
        "command",
        "working_directory",
        "requested_gpus",
        "sharing_mode",
        "fallback_machines",
        "offer_after_seconds",
        "depends_on_task_ids",
    }
    missing = sorted(required - set(value))
    if missing:
        raise RuntimeError(f"submission task_specs[{index}] is missing required fields: {', '.join(missing)}.")
    optional = {"requested_cpus", "offer_eligible_at", "offer_clock_evidence"}
    unknown = sorted(set(value) - required - optional)
    if unknown:
        raise RuntimeError(f"submission task_specs[{index}] has unknown fields: {', '.join(unknown)}.")
    task_id = value["task_id"]
    validate_identifier(task_id, f"task_specs[{index}].task_id")
    if value["name"] is not None and not isinstance(value["name"], str):
        raise ValueError(f"task_specs[{index}].name must be a string or null.")
    validate_identifier(value["home_machine"], f"task_specs[{index}].home_machine")
    command = value["command"]
    if not isinstance(command, list) or not command or any(not isinstance(item, str) for item in command):
        raise ValueError(f"task_specs[{index}].command must be a non-empty list of strings.")
    if not isinstance(value["working_directory"], str):
        raise ValueError(f"task_specs[{index}].working_directory must be a string.")
    if not Path(value["working_directory"]).is_absolute():
        raise ValueError(f"task_specs[{index}].working_directory must be absolute.")
    if type(value["requested_gpus"]) is not int or value["requested_gpus"] < 0:
        raise ValueError(f"task_specs[{index}].requested_gpus must be a non-negative integer.")
    requested_cpus = value.get("requested_cpus")
    if requested_cpus is not None and (type(requested_cpus) is not int or requested_cpus < 1):
        raise ValueError(f"task_specs[{index}].requested_cpus must be a positive integer or null.")
    if value["requested_gpus"] == 0 and requested_cpus is None:
        raise ValueError(f"task_specs[{index}] CPU task is missing requested_cpus.")
    if value["requested_gpus"] > 0 and requested_cpus is not None:
        raise ValueError(f"task_specs[{index}] GPU task cannot contain requested_cpus.")
    if value["sharing_mode"] not in {"private", "spillover"}:
        raise ValueError(f"task_specs[{index}].sharing_mode is invalid.")
    fallback = value["fallback_machines"]
    if fallback != "group":
        if not isinstance(fallback, list) or not fallback:
            raise ValueError(f"task_specs[{index}].fallback_machines is invalid.")
        names = [validate_identifier(machine, f"task_specs[{index}].fallback_machines") for machine in fallback]
        if len(set(names)) != len(names):
            raise ValueError(f"task_specs[{index}].fallback_machines must not contain duplicates.")
    offer_after = value["offer_after_seconds"]
    if offer_after is not None and (type(offer_after) is not int or offer_after < 0):
        raise ValueError(f"task_specs[{index}].offer_after_seconds is invalid.")
    if offer_after is not None and value["sharing_mode"] != "spillover":
        raise ValueError(f"task_specs[{index}] timed offer requires spillover placement.")
    dependencies = value["depends_on_task_ids"]
    normalized = normalize_dependency_ids(dependencies)
    if dependencies != normalized:
        raise ValueError(f"task_specs[{index}].depends_on_task_ids are not canonical.")
    if task_id in normalized:
        raise ValueError(f"task_specs[{index}] cannot depend on itself.")
    offer_eligible_at = value.get("offer_eligible_at")
    offer_clock_evidence = value.get("offer_clock_evidence")
    if offer_eligible_at is not None and not isinstance(offer_eligible_at, str):
        raise ValueError(f"task_specs[{index}].offer_eligible_at is invalid.")
    if offer_clock_evidence is not None and not isinstance(offer_clock_evidence, dict):
        raise ValueError(f"task_specs[{index}].offer_clock_evidence is invalid.")
    if offer_after is None and (offer_eligible_at is not None or offer_clock_evidence is not None):
        raise ValueError(f"task_specs[{index}] has timed-offer proof without an offer delay.")
    if offer_after is not None and (offer_eligible_at is None or offer_clock_evidence is None):
        raise ValueError(f"task_specs[{index}] is missing timed-offer evidence.")
    if offer_clock_evidence is not None:
        proof_fields = {"creator_observation", "deadline_monotonic_at"}
        observation_fields = {
            "observation_id",
            "provider",
            "observed_at",
            "monotonic_observed_at",
            "boot_id",
            "lower_error_seconds",
            "upper_error_seconds",
            "max_drift_rate",
            "provider_margin_seconds",
        }
        observation = offer_clock_evidence.get("creator_observation")
        string_fields = {"observation_id", "provider", "observed_at", "boot_id"}
        number_fields = {
            "monotonic_observed_at",
            "lower_error_seconds",
            "upper_error_seconds",
            "max_drift_rate",
            "provider_margin_seconds",
        }
        valid_strings = isinstance(observation, dict) and all(
            isinstance(observation.get(field), str) and observation[field] for field in string_fields
        )
        valid_numbers = isinstance(observation, dict) and all(
            type(observation.get(field)) in {int, float} and math.isfinite(observation[field])
            for field in number_fields
        )
        deadline = offer_clock_evidence.get("deadline_monotonic_at")
        if (
            set(offer_clock_evidence) != proof_fields
            or not isinstance(observation, dict)
            or set(observation) != observation_fields
            or not valid_strings
            or not valid_numbers
            or type(deadline) not in {int, float}
            or not math.isfinite(deadline)
        ):
            raise ValueError(f"task_specs[{index}].offer_clock_evidence is invalid.")
    return value


def _validate_worker_additions(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError("submission worker_set_additions must be a declaration mapping.")
    result: dict[str, dict[str, Any]] = {}
    for machine, declaration in value.items():
        validate_identifier(machine, "worker_set_additions machine")
        if not isinstance(declaration, Mapping):
            raise ValueError(f"worker_set_additions.{machine} must be a mapping.")
        normalized = _worker_additions({machine: dict(declaration)})[machine]
        result[machine] = normalized
    return {machine: result[machine] for machine in sorted(result)}


def decode_submission_plan(operation: Mapping[str, Any]) -> SubmissionPlan:
    """Decode and validate an existing operation without consulting runtime state."""
    if not isinstance(operation, Mapping):
        raise RuntimeError("submission operation is not a mapping.")
    submission = operation.get("submission")
    if not isinstance(submission, Mapping):
        raise RuntimeError("submission operation has no valid submission envelope.")
    operation_id = submission.get("operation_id")
    try:
        validate_identifier(operation_id, "submission operation_id")
    except (TypeError, ValueError) as exc:
        raise RuntimeError("submission operation_id is invalid.") from exc
    key = submission.get("idempotency_key")
    if not isinstance(key, str) or not key:
        raise RuntimeError("submission idempotency_key is invalid.")
    kind = submission.get("kind")
    if not isinstance(kind, str) or not kind:
        raise RuntimeError("submission kind is invalid.")
    raw_digest = _validate_digest(submission.get("raw_request_digest"), "raw_request_digest")
    resolved_digest = _validate_digest(submission.get("resolved_context_digest"), "resolved_context_digest")
    original_machine = submission.get("original_submitting_machine")
    try:
        validate_identifier(original_machine, "original_submitting_machine")
    except (TypeError, ValueError) as exc:
        raise RuntimeError("submission original_submitting_machine is invalid.") from exc
    target_group = submission.get("target_group")
    try:
        target_group = validate_group_name(target_group)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("submission target_group is invalid.") from exc

    context = submission.get("resolved_context")
    if not isinstance(context, Mapping):
        raise RuntimeError("submission resolved_context is invalid.")
    required_context = {
        "task_ids",
        "task_specs",
        "create_group",
        "worker_set_additions",
        "group_precondition",
        "planned_worker_set",
    }
    allowed_context = required_context | {"worker_set_declared"}
    if set(context) - allowed_context or not required_context.issubset(context):
        raise RuntimeError("submission resolved_context has invalid fields.")
    persisted_worker_set_declared = context.get("worker_set_declared")
    if persisted_worker_set_declared is not None and type(persisted_worker_set_declared) is not bool:
        raise ValueError("submission worker_set_declared must be a boolean.")
    try:
        persisted_digest = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    except (TypeError, ValueError) as exc:
        raise RuntimeError("submission resolved_context is not canonical JSON.") from exc
    if persisted_digest != resolved_digest:
        raise RuntimeError("submission resolved_context_digest does not match resolved_context.")

    task_ids = context["task_ids"]
    task_specs = context["task_specs"]
    if not isinstance(task_ids, list) or any(not isinstance(item, str) for item in task_ids):
        raise ValueError("submission task_ids must be a list of strings.")
    if not task_ids:
        raise ValueError("submission task_ids must not be empty.")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("submission task_ids must be unique.")
    for index, task_id in enumerate(task_ids):
        validate_identifier(task_id, f"task_ids[{index}]")
    if not isinstance(task_specs, list) or len(task_specs) != len(task_ids):
        raise RuntimeError("submission task_specs and task_ids must have the same length.")
    canonical_specs = []
    for index, raw in enumerate(task_specs):
        if not isinstance(raw, Mapping):
            raise ValueError(f"submission task_specs[{index}] must be a mapping.")
        item = _validate_task_spec_for_decode(raw, index)
        if item["task_id"] != task_ids[index]:
            raise RuntimeError("submission task_specs and task_ids disagree.")
        canonical_specs.append(item)
    _validate_planned_dependencies(canonical_specs, target_group)
    if "task_observation" not in operation:
        tmux_overrides = tuple(None for _ in task_ids)
    else:
        metadata = operation["task_observation"]
        try:
            decoded_metadata = decode_task_observation(metadata, task_ids)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("submission task_observation metadata is invalid.") from exc
        tmux_overrides = tuple(item["tmux_override"] for item in decoded_metadata["tasks"])

    if "live_progress_selection" not in operation:
        live_progress_selection = disabled_live_progress_selection(task_ids)
    else:
        try:
            live_progress_selection = validate_live_progress_selection(
                operation["live_progress_selection"], task_ids, group_name=target_group
            )
        except Exception as exc:
            warn_invalid_live_progress_selection(exc)
            live_progress_selection = disabled_live_progress_selection(task_ids)

    create_group = context["create_group"]
    if type(create_group) is not bool:
        raise ValueError("submission create_group must be a boolean.")
    precondition = context["group_precondition"]
    if not isinstance(precondition, Mapping) or set(precondition) != {"exists", "revision", "worker_set_epoch"}:
        raise ValueError("submission group_precondition is invalid.")
    if type(precondition["exists"]) is not bool:
        raise ValueError("submission group_precondition.exists must be a boolean.")
    for field in ("revision", "worker_set_epoch"):
        value = precondition[field]
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"submission group_precondition.{field} is invalid.")
    if not precondition["exists"] and (
        precondition["revision"] is not None or precondition["worker_set_epoch"] is not None
    ):
        raise ValueError("missing Group precondition must not contain revisions.")
    if precondition["exists"] and (precondition["revision"] is None or precondition["worker_set_epoch"] is None):
        raise ValueError("existing Group precondition must contain revisions.")
    if create_group != bool(target_group and not precondition["exists"]):
        raise ValueError("submission create_group does not match its Group precondition.")
    planned_workers = context["planned_worker_set"]
    if not isinstance(planned_workers, list) or any(not isinstance(item, str) for item in planned_workers):
        raise ValueError("submission planned_worker_set must be a list of machine names.")
    for machine in planned_workers:
        validate_identifier(machine, "planned_worker_set machine")
    if planned_workers != sorted(set(planned_workers)):
        raise ValueError("submission planned_worker_set must be sorted and unique.")
    additions = _validate_worker_additions(context["worker_set_additions"])
    worker_set_declared = bool(additions) if persisted_worker_set_declared is None else persisted_worker_set_declared
    if target_group is None and (precondition["exists"] or planned_workers or additions):
        raise ValueError("ungrouped submission has contradictory Group planning fields.")
    if not set(additions).issubset(planned_workers):
        raise ValueError("submission worker_set_additions are outside planned_worker_set.")
    _validate_placement_against_workers(
        canonical_specs,
        group_name=target_group,
        planned_workers={machine: {} for machine in planned_workers},
    )

    staged_count = submission.get("staged_task_count")
    if type(staged_count) is not int or staged_count != len(task_ids):
        raise RuntimeError("submission staged_task_count is invalid.")
    state = submission.get("state")
    if state not in _SUBMISSION_STATES:
        raise RuntimeError("submission state is invalid.")
    commit_plan = submission.get("commit_plan")
    if not isinstance(commit_plan, Mapping) or set(commit_plan) != {
        "group_membership_sequences",
        "pending_group_revision",
    }:
        raise RuntimeError("submission commit_plan is invalid.")
    sequences = commit_plan["group_membership_sequences"]
    if sequences is not None and (
        not isinstance(sequences, list)
        or len(sequences) != len(task_ids)
        or any(type(item) is not int or item < 1 for item in sequences)
    ):
        raise RuntimeError("submission group membership sequence plan is invalid.")

    return SubmissionPlan(
        operation_id=operation_id,
        key=key,
        kind=kind,
        raw_request_digest=raw_digest,
        resolved_context_digest=resolved_digest,
        original_submitting_machine=original_machine,
        target_group=target_group,
        task_ids=tuple(task_ids),
        task_specs=tuple(canonical_specs),
        tmux_overrides=tmux_overrides,
        create_group=create_group,
        group_precondition=dict(precondition),
        planned_worker_set=tuple(planned_workers),
        worker_set_additions=additions,
        worker_set_declared=worker_set_declared,
        live_progress_selection=live_progress_selection,
    )
