"""Connect machine dispatch evidence to the machine-local Idle Job supervisor."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from contextlib import closing
from functools import partial
from typing import Any

from ..runtime.resources.cpu_lane import cpu_reservation_snapshot
from ..runtime.resources.reservations import reservation_snapshot
from . import dispatch_loop as _dispatch
from .context import MachineRuntime, ProjectBinding
from .helpers import _observe_unregistered_partitions
from .idle_job import IdleJobSupervisor
from .idle_job_config import load_idle_job_config
from .idle_job_diagnostics import build_demand_observation
from .project_io_controller import ProjectIOController

_ORDINARY_ERRORS = (AttributeError, OSError, RuntimeError, TypeError, ValueError, KeyError)
_DEMANDS = frozenset({"idle", "busy", "unknown"})


def _policy_value(policy: object, key: str, default: Any = None) -> Any:
    if isinstance(policy, Mapping):
        return policy.get(key, default)
    return getattr(policy, key, default)


def _safe_tick(
    supervisor: IdleJobSupervisor,
    demand: str,
    *,
    gpu_policy: object = None,
    launch_allowed: bool = True,
    demand_detail: Mapping[str, Any] | None = None,
    agent_instance_id: str | None = None,
) -> dict[str, Any]:
    """Tick the supervisor while keeping ordinary integration failures fail-closed."""

    def observe(selected_demand: str, detail: Mapping[str, Any] | None) -> None:
        try:
            setter = getattr(supervisor, "set_demand_observation", None)
            if callable(setter):
                setter(selected_demand, detail, agent_instance_id=agent_instance_id)
        except _ORDINARY_ERRORS:
            pass

    observe(demand, demand_detail)
    try:
        status = supervisor.tick(demand=demand, gpu_policy=gpu_policy, launch_allowed=launch_allowed)
        return dict(status) if isinstance(status, Mapping) else {}
    except _ORDINARY_ERRORS:
        if demand != "unknown":
            observe("unknown", build_demand_observation("observation_failed"))
            try:
                status = supervisor.tick(demand="unknown")
                return dict(status) if isinstance(status, Mapping) else {}
            except _ORDINARY_ERRORS:
                pass
        observe("unknown", None)
        return {}


def _unknown(
    supervisor: IdleJobSupervisor,
    code: str | None = None,
    *,
    agent_instance_id: str | None = None,
    runtime: MachineRuntime | None = None,
    registry_revision: int | None = None,
    project_ids: Collection[str] | None = None,
    bindings: Sequence[ProjectBinding] = (),
) -> dict[str, Any]:
    detail = None
    if code is not None:
        try:
            upgrade_bindings = upgrade_projects = None
            if (
                code == "upgrade_blocks_idle"
                and registry_revision is not None
                and getattr(runtime, "upgrade_registry_revision", None) == registry_revision
                and not getattr(runtime, "upgrade_discovery_unknown", False)
                and not isinstance(getattr(runtime, "project_io_controller", None), ProjectIOController)
            ):
                # Isolated upgrade results contain flags, so their presence supersedes the legacy cache.
                upgrade_bindings = getattr(runtime, "upgrade_binding_by_id", None)
                upgrade_projects = getattr(runtime, "upgrade_discovery_projects", None)
            detail = build_demand_observation(
                code,
                project_ids=project_ids,
                bindings=bindings,
                runtime_root=runtime.root if runtime is not None else None,
                upgrade_bindings=upgrade_bindings if isinstance(upgrade_bindings, Mapping) else None,
                upgrade_projects=upgrade_projects if isinstance(upgrade_projects, Mapping) else None,
            )
        except _ORDINARY_ERRORS:
            detail = build_demand_observation(code)
    return _safe_tick(supervisor, "unknown", demand_detail=detail, agent_instance_id=agent_instance_id)


def _binding_recovery_is_clear(runtime: MachineRuntime, bindings: Sequence[ProjectBinding]) -> bool:
    """Check bounded recovery evidence without reading shared task state synchronously."""
    for binding in bindings:
        if binding.enabled:
            continue
        try:
            with closing(runtime.iter_recovery_blockers(binding, should_use_capture=True)) as blockers:
                if next(blockers, None) is not None:
                    return False
        except _ORDINARY_ERRORS:
            return False
    return True


def _first_unready_authority(runtime: MachineRuntime, bindings: Sequence[ProjectBinding]) -> ProjectBinding | None:
    ready_generations = getattr(runtime, "authority_ready_generations", None)
    for binding in bindings:
        if binding.enabled and (
            not isinstance(ready_generations, Mapping)
            or ready_generations.get(binding.project_id) != binding.registration_generation
        ):
            return binding
    return None


def _gpu_policy_observation(
    runtime: MachineRuntime,
    *,
    available_gpus: list[int] | None,
    instance_id: str,
) -> tuple[object, object, tuple[int, ...]] | None:
    """Observe GPU visibility before acquiring either machine reservation lock."""
    _reservation_policy, policy_view = _dispatch._observe_gpu_policy(
        runtime,
        available_gpus=available_gpus,
        instance_id=instance_id,
        reserved_gpu_ids=frozenset(),
    )
    visible_status = _policy_value(policy_view, "visible_status")
    visible = _policy_value(policy_view, "visible_gpu_ids")
    if visible is None or visible_status in {"pending", "unavailable", "invalid"}:
        return None
    if not isinstance(visible, (tuple, list)):
        return None
    if any(type(gpu_id) is not int or gpu_id < 0 for gpu_id in visible):
        return None
    actual = tuple(sorted(set(visible)))
    if len(actual) != len(visible):
        return None
    return _reservation_policy, policy_view, actual


def _capacity_key(
    policy_view: object,
    allowed_gpu_ids: Sequence[int],
    *,
    cpu_revision: int,
    cpu_capacity: int,
) -> str:
    values = {
        "gpu_policy_revision": _policy_value(policy_view, "revision"),
        "allowed_gpu_ids": list(allowed_gpu_ids),
        "cpu_policy_revision": cpu_revision,
        "cpu_capacity": cpu_capacity,
    }
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def advance_machine_idle_job(
    runtime: MachineRuntime,
    supervisor: IdleJobSupervisor,
    *,
    available_gpus: list[int] | None,
    instance_id: str,
    cycle_completed: bool,
    bindings: Sequence[ProjectBinding],
    registry_revision: int,
    enablement_converged: bool,
    partition_observation: str | None = None,
) -> dict[str, Any]:
    """Advance one fail-closed Idle Job observation after a dispatch cycle.

    The function performs only machine-local evidence checks and delegates all
    Project-root demand observation to the typed ``ProjectIOController``.
    """
    unknown = partial(
        _unknown, supervisor, agent_instance_id=instance_id, runtime=runtime, registry_revision=registry_revision
    )
    try:
        config = load_idle_job_config(runtime)
        if not config.enabled:
            return unknown()

        # Hardware discovery is deliberately outside the reservation lock. A
        # reservation already held by training is busy evidence for both
        # lanes, even when a concurrent GPU discovery is unavailable.
        try:
            policy_observation = _gpu_policy_observation(
                runtime,
                available_gpus=available_gpus,
                instance_id=instance_id,
            )
        except _ORDINARY_ERRORS:
            policy_observation = None
        snapshot = reservation_snapshot(runtime.root)
        if snapshot.active or snapshot.provisional:
            reservation_policy = policy_observation[0] if policy_observation is not None else None
            return _safe_tick(supervisor, "busy", gpu_policy=reservation_policy, agent_instance_id=instance_id)

        if not cycle_completed:
            return unknown("dispatch_cycle_incomplete")
        if not enablement_converged:
            return unknown("enablement_unready")
        if not isinstance(bindings, Sequence) or isinstance(bindings, (str, bytes, bytearray)):
            return unknown("bindings_unknown")
        observed_bindings = tuple(bindings)
        if any(not isinstance(binding, ProjectBinding) for binding in observed_bindings):
            return unknown("bindings_unknown")

        if getattr(runtime, "pending_launch_handoffs", {}):
            return unknown("launch_handoff_pending")
        if getattr(runtime, "upgrade_discovery_unknown", False):
            return unknown("upgrade_evidence_unknown")
        if upgrade_blockers := getattr(runtime, "upgrade_idle_blocked_projects", set()):
            return unknown("upgrade_blocks_idle", project_ids=upgrade_blockers, bindings=observed_bindings)
        if enrollment_blockers := getattr(runtime, "recovery_enrollment_pending_projects", set()):
            return unknown("recovery_enrollment_pending", project_ids=enrollment_blockers)
        if retry_blockers := getattr(runtime, "maintenance_retry_idle_blocked_projects", set()):
            return unknown("maintenance_retry_pending", project_ids=retry_blockers)
        if partition_observation is None:
            partition_observation = _observe_unregistered_partitions(runtime, observed_bindings)
        if partition_observation not in {"clear", "pending"}:
            return unknown("partition_unknown")
        if not _binding_recovery_is_clear(runtime, observed_bindings):
            return unknown("recovery_unknown")
        unready = _first_unready_authority(runtime, observed_bindings)
        if unready is not None:
            return unknown("authority_unready", project_ids=(unready.project_id,))

        controller = getattr(runtime, "project_io_controller", None)
        if not isinstance(controller, ProjectIOController):
            return unknown("demand_census_unknown")
        executor = getattr(runtime, "project_io_executor", None)
        if executor is None or controller.executor is not executor:
            return unknown("executor_unready")
        executor_status = executor.status_view()
        if not isinstance(executor_status, Mapping) or executor_status.get("envelope") != "healthy":
            return unknown("executor_unready")

        if policy_observation is None:
            return unknown("gpu_policy_unknown")
        reservation_policy, policy_view, allowed_gpu_ids = policy_observation

        cpu_policy, _cpu_reservations = cpu_reservation_snapshot(runtime.root)
        if type(cpu_policy.revision) is not int or cpu_policy.revision < 0:
            return unknown("cpu_policy_unknown")
        if type(cpu_policy.capacity) is not int or cpu_policy.capacity < 0:
            return unknown("cpu_policy_unknown")

        enabled_bindings = tuple(binding for binding in observed_bindings if binding.enabled)
        demand = controller.advance_idle_job_demand(
            enabled_bindings,
            registry_revision,
            gpu_capacity=len(allowed_gpu_ids),
            cpu_capacity=cpu_policy.capacity,
            capacity_key=_capacity_key(
                policy_view,
                allowed_gpu_ids,
                cpu_revision=cpu_policy.revision,
                cpu_capacity=cpu_policy.capacity,
            ),
        )
        if demand not in _DEMANDS:
            demand = "unknown"
        return _safe_tick(
            supervisor,
            demand,
            gpu_policy=reservation_policy,
            launch_allowed=partition_observation == "clear",
            demand_detail=build_demand_observation("demand_census_unknown") if demand == "unknown" else None,
            agent_instance_id=instance_id,
        )
    except _ORDINARY_ERRORS:
        return unknown("observation_failed")


__all__ = ["advance_machine_idle_job"]
