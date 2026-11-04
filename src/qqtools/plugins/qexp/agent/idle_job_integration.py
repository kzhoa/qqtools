"""Connect machine dispatch evidence to the machine-local Idle Job supervisor."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from contextlib import closing
from typing import Any

from ..runtime.resources.cpu_lane import cpu_reservation_snapshot
from ..runtime.resources.reservations import reservation_snapshot
from . import dispatch_loop as _dispatch
from .context import MachineRuntime, ProjectBinding
from .helpers import _observe_unregistered_partitions
from .idle_job import IdleJobSupervisor
from .idle_job_config import load_idle_job_config
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
) -> dict[str, Any]:
    """Tick the supervisor while keeping ordinary integration failures fail-closed."""
    try:
        status = supervisor.tick(demand=demand, gpu_policy=gpu_policy, launch_allowed=launch_allowed)
        return dict(status) if isinstance(status, Mapping) else {}
    except _ORDINARY_ERRORS:
        if demand != "unknown":
            try:
                status = supervisor.tick(demand="unknown")
                return dict(status) if isinstance(status, Mapping) else {}
            except _ORDINARY_ERRORS:
                pass
        return {}


def _unknown(supervisor: IdleJobSupervisor) -> dict[str, Any]:
    return _safe_tick(supervisor, "unknown")


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


def _authority_is_ready(runtime: MachineRuntime, bindings: Sequence[ProjectBinding]) -> bool:
    ready_generations = getattr(runtime, "authority_ready_generations", None)
    if not isinstance(ready_generations, Mapping):
        return not any(binding.enabled for binding in bindings)
    return all(
        not binding.enabled or ready_generations.get(binding.project_id) == binding.registration_generation
        for binding in bindings
    )


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
    try:
        config = load_idle_job_config(runtime)
        if not config.enabled:
            return _unknown(supervisor)

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
            return _safe_tick(supervisor, "busy", gpu_policy=reservation_policy)

        if not cycle_completed or not enablement_converged:
            return _unknown(supervisor)
        if not isinstance(bindings, Sequence) or isinstance(bindings, (str, bytes, bytearray)):
            return _unknown(supervisor)
        observed_bindings = tuple(bindings)
        if any(not isinstance(binding, ProjectBinding) for binding in observed_bindings):
            return _unknown(supervisor)

        if (
            getattr(runtime, "pending_launch_handoffs", {})
            or getattr(runtime, "upgrade_discovery_unknown", False)
            or getattr(runtime, "upgrade_idle_blocked_projects", set())
            or getattr(runtime, "recovery_enrollment_pending_projects", set())
            or getattr(runtime, "maintenance_retry_idle_blocked_projects", set())
        ):
            return _unknown(supervisor)
        if partition_observation is None:
            partition_observation = _observe_unregistered_partitions(runtime, observed_bindings)
        if partition_observation not in {"clear", "pending"}:
            return _unknown(supervisor)
        if not _binding_recovery_is_clear(runtime, observed_bindings):
            return _unknown(supervisor)
        if not _authority_is_ready(runtime, observed_bindings):
            return _unknown(supervisor)

        controller = getattr(runtime, "project_io_controller", None)
        if not isinstance(controller, ProjectIOController):
            return _unknown(supervisor)
        executor = getattr(runtime, "project_io_executor", None)
        if executor is None or controller.executor is not executor:
            return _unknown(supervisor)
        executor_status = executor.status_view()
        if not isinstance(executor_status, Mapping) or executor_status.get("envelope") != "healthy":
            return _unknown(supervisor)

        if policy_observation is None:
            return _unknown(supervisor)
        reservation_policy, policy_view, allowed_gpu_ids = policy_observation

        cpu_policy, _cpu_reservations = cpu_reservation_snapshot(runtime.root)
        if type(cpu_policy.revision) is not int or cpu_policy.revision < 0:
            return _unknown(supervisor)
        if type(cpu_policy.capacity) is not int or cpu_policy.capacity < 0:
            return _unknown(supervisor)

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
        )
    except _ORDINARY_ERRORS:
        return _unknown(supervisor)


__all__ = ["advance_machine_idle_job"]
