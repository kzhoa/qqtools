"""Bounded isolated publication of final Project machine snapshots."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from ..gpu_policy import show_gpu_policy
from ..runtime.resources.reservations import reservation_snapshot
from .context import MachineRuntime, ProjectBinding
from .project_io_executor import ProjectIOExecutor

PROJECT_STOP_PUBLICATION_BUDGET_SECONDS = 2.0


def publish_project_stop_snapshots(
    runtime: MachineRuntime,
    executor: ProjectIOExecutor,
    *,
    instance_id: str,
    available_gpus: Sequence[int] | None,
    heartbeat_interval_seconds: float,
    started_at: str,
    stop_reason: str,
    budget_seconds: float = PROJECT_STOP_PUBLICATION_BUDGET_SECONDS,
) -> None:
    """Attempt every eligible stop snapshot through a fresh bounded executor epoch.

    Old unresolved requests retain their evidence and owner slots. Responsive
    bindings use any remaining slots; blocked bindings make this cleanup step
    fail without moving shared I/O back into the controller.
    """
    if budget_seconds <= 0:
        raise ValueError("Project stop publication budget must be positive.")
    try:
        gpu_policy = show_gpu_policy(runtime)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
        gpu_policy = {}
    try:
        reservations = list(reservation_snapshot(runtime.root).reservations)
        reserved_gpu_ids = sorted({gpu_id for item in reservations for gpu_id in item.get("gpu_ids", [])})
    except (KeyError, OSError, ValueError):
        reservations = []
        reserved_gpu_ids = []
    try:
        registry_revision, registered = runtime.load_registry()
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError("Project registry is unavailable during stop publication.") from exc

    visible_value = gpu_policy.get("visible_gpu_ids")
    visible_gpu_ids = (
        sorted(set(visible_value))
        if isinstance(visible_value, list) and all(type(item) is int and item >= 0 for item in visible_value)
        else sorted(set(available_gpus or ()))
    )
    pending = [binding for binding in registered if binding.enabled]
    started: dict[str, ProjectBinding] = {}
    failures: list[str] = []
    deadline = time.monotonic() + budget_seconds
    executor.begin_epoch()
    try:
        while time.monotonic() < deadline and (pending or started):
            made_progress = _consume_stop_results(executor, started, failures)
            unresolved = executor.unresolved_requests()
            blocked_owners = {
                (request.runtime_id, request.project_id, request.registration_generation) for request in unresolved
            }
            free_slots = max(0, executor.status_view()["capacity"] - len(unresolved))
            for binding in tuple(pending):
                if free_slots <= 0:
                    break
                owner = (executor.runtime_id, binding.project_id, binding.registration_generation)
                if owner in blocked_owners:
                    continue
                summaries = [dict(item) for item in reservations if item.get("project_id") == binding.project_id]
                try:
                    request = executor.prepare_machine_snapshot_publish(
                        binding,
                        registry_revision,
                        instance_id=instance_id,
                        pid=None,
                        visible_gpu_ids=visible_gpu_ids,
                        reserved_gpu_ids=reserved_gpu_ids,
                        reservation_summaries=summaries,
                        heartbeat_interval_seconds=heartbeat_interval_seconds,
                        started_at=started_at,
                        gpu_policy=gpu_policy,
                        stop_reason=stop_reason,
                    )
                    process = executor.start(request.request_id)
                except (OSError, RuntimeError, ValueError):
                    continue
                if process is None:
                    continue
                pending.remove(binding)
                started[request.request_id] = binding
                blocked_owners.add(owner)
                free_slots -= 1
                made_progress = True
            if not made_progress:
                time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
    finally:
        executor.shutdown()

    incomplete = {(binding.project_id, binding.registration_generation) for binding in pending} | {
        (binding.project_id, binding.registration_generation) for binding in started.values()
    }
    if failures or incomplete:
        raise RuntimeError(f"{len(failures) + len(incomplete)} Project stop publication(s) did not settle")


def _consume_stop_results(
    executor: ProjectIOExecutor,
    started: dict[str, ProjectBinding],
    failures: list[str],
) -> bool:
    made_progress = False
    executor.poll()
    for request_id, binding in tuple(started.items()):
        result = executor.load_result(request_id)
        if result is None or result.status == "outcome_unknown":
            continue
        consumed = executor.consume(request_id, result.request)
        if consumed is None:
            continue
        started.pop(request_id)
        if consumed.status != "completed" or consumed.evidence.get("outcome") not in {
            "published",
            "stale",
        }:
            failures.append(binding.project_id)
        made_progress = True
    return made_progress
