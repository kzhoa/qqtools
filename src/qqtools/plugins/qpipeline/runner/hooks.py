"""Frozen runner lifecycle-hook contracts.

This module intentionally contains no generic payload or command protocol.  A hook slot has one
context, one result contract, and one framework-reserved provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch.distributed as dist

from .runner_utils.best_model import BestMetricSnapshot
from .runner_utils.evaluation import EvaluationResult
from .runner_utils.types import RunMode


@dataclass(kw_only=True, frozen=True, slots=True)
class LoopDirective:
    """The only loop-control result accepted from a runner hook."""

    end_epoch: bool = False


@dataclass(kw_only=True, frozen=True, slots=True)
class ValidationHookContext:
    epoch: int
    global_step: int
    evaluation: EvaluationResult
    is_best: bool
    previous_best: Optional[BestMetricSnapshot]


@dataclass(kw_only=True, frozen=True, slots=True)
class RunnerBoundaryContext:
    epoch: int
    global_step: int
    run_mode: RunMode
    did_optimizer_step: bool
    is_epoch_end: bool
    terminal_candidate: bool
    latest_train_loss: Optional[float]


@dataclass(kw_only=True, frozen=True, slots=True)
class OptimizerStepEndContext:
    epoch: int
    global_step: int
    is_natural_epoch_end: bool


@dataclass(kw_only=True, frozen=True, slots=True)
class _HookRegistration:
    slot_name: str
    provider_id: str
    protocol_version: int
    callback: Callable


class RunnerHooks:
    """Composition-time runner hook slots.

    The object is deliberately small: there is no priority, replacement, chaining, or dynamic
    registration.  A distributed run validates its frozen descriptor before the Agent starts.
    """

    _SLOTS = (
        "after_validation",
        "boundary_cursor",
        "after_epoch_commit",
        "optimizer_step_end",
    )

    def __init__(self) -> None:
        self._registrations: dict[str, _HookRegistration] = {}
        self._is_frozen = False

    def _set(self, slot_name: str, callback: Callable, provider_id: str) -> None:
        self.set_hook_bundle(((slot_name, callback),), provider_id=provider_id)

    def set_hook_bundle(
        self,
        registrations: tuple[tuple[str, Callable], ...],
        *,
        provider_id: str,
    ) -> None:
        """Atomically install one provider's lifecycle-hook bundle."""
        if self._is_frozen:
            raise RuntimeError("Runner hook registration is frozen.")
        if not isinstance(provider_id, str) or not provider_id:
            raise ValueError("Runner hook provider_id must be a non-empty string.")
        if not registrations:
            raise ValueError("Runner hook bundle must contain at least one registration.")

        slot_names = tuple(slot_name for slot_name, _ in registrations)
        if len(set(slot_names)) != len(slot_names):
            raise ValueError(f"Runner hook bundle contains duplicate slots: {slot_names!r}.")
        for slot_name in slot_names:
            if slot_name not in self._SLOTS:
                raise ValueError(f"Unknown runner hook slot: {slot_name!r}.")
            if slot_name in self._registrations:
                existing = self._registrations[slot_name]
                raise RuntimeError(
                    f"Runner hook slot {slot_name!r} is already owned by {existing.provider_id!r}."
                )

        self._registrations.update(
            {
                slot_name: _HookRegistration(
                    slot_name=slot_name,
                    provider_id=provider_id,
                    protocol_version=1,
                    callback=callback,
                )
                for slot_name, callback in registrations
            }
        )

    def set_after_validation_hook(
        self, hook: Callable[[ValidationHookContext], None], *, provider_id: str
    ) -> None:
        self._set("after_validation", hook, provider_id)

    def set_boundary_cursor_hook(
        self, hook: Callable[[RunnerBoundaryContext], None], *, provider_id: str
    ) -> None:
        self._set("boundary_cursor", hook, provider_id)

    def set_after_epoch_commit_hook(
        self, hook: Callable[[RunnerBoundaryContext], None], *, provider_id: str
    ) -> None:
        self._set("after_epoch_commit", hook, provider_id)

    def set_optimizer_step_end_hook(
        self, hook: Callable[[OptimizerStepEndContext], LoopDirective], *, provider_id: str
    ) -> None:
        self._set("optimizer_step_end", hook, provider_id)

    @property
    def has_optimizer_step_end_hook(self) -> bool:
        return "optimizer_step_end" in self._registrations

    @property
    def has_collective_capable_validation_hook(self) -> bool:
        return "after_validation" in self._registrations

    def freeze(self) -> None:
        self._is_frozen = True

    def _require_frozen(self) -> None:
        if not self._is_frozen:
            raise RuntimeError("Runner hooks must be frozen before dispatch.")

    def validate_distributed_plan(self, distributed: bool) -> None:
        """Reject divergent lifecycle-hook plans before any hook can enter a collective."""
        if not self._is_frozen:
            raise RuntimeError("Runner hooks must be frozen before distributed plan validation.")
        if not distributed:
            return
        descriptor = tuple(
            (slot_name, registration.provider_id, registration.protocol_version)
            for slot_name, registration in sorted(self._registrations.items())
        )
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, descriptor)
        if any(item != descriptor for item in gathered):
            raise RuntimeError(f"Runner hook plan mismatch across ranks: {gathered!r}")

    def _dispatch(self, slot_name: str, context: object) -> None:
        self._require_frozen()
        registration = self._registrations.get(slot_name)
        if registration is not None:
            result = registration.callback(context)
            if result is not None:
                raise TypeError(
                    f"Runner hook {slot_name!r} must return None, got {type(result).__name__}."
                )

    def dispatch_after_validation(self, context: ValidationHookContext) -> None:
        self._dispatch("after_validation", context)

    def dispatch_at_boundary_cursor(self, context: RunnerBoundaryContext) -> None:
        self._dispatch("boundary_cursor", context)

    def dispatch_after_epoch_commit(self, context: RunnerBoundaryContext) -> None:
        self._dispatch("after_epoch_commit", context)

    def dispatch_optimizer_step_end(self, context: OptimizerStepEndContext) -> LoopDirective:
        self._require_frozen()
        registration = self._registrations.get("optimizer_step_end")
        if registration is None:
            raise RuntimeError("optimizer_step_end hook is not installed.")
        directive = registration.callback(context)
        if not isinstance(directive, LoopDirective):
            raise TypeError("optimizer_step_end hook must return LoopDirective.")
        return directive
