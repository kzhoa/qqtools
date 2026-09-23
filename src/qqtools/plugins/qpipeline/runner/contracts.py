"""Typed runner extension contracts.

The runner deliberately separates lifecycle reactions, committed observations, and control
operations.  None of these objects expose mutable runner state or a generic signal channel.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Optional

import torch.distributed as dist

from ..types import Stage
from .runner_utils.best_model import BestMetricSnapshot
from .runner_utils.evaluation import EvaluationResult


def freeze_scalar_metrics(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    """Copy scalar metrics into a read-only mapping for extension contexts."""
    frozen: dict[str, Any] = {}
    for key, value in metrics.items():
        if not isinstance(key, str):
            raise TypeError(f"Metric keys must be strings, got {type(key).__name__}.")
        item = getattr(value, "item", None)
        value = item() if callable(item) else value
        if not isinstance(value, (bool, int, float, str, type(None))):
            raise TypeError(f"Metric {key!r} must be a materialized scalar, got {type(value).__name__}.")
        frozen[key] = value
    return MappingProxyType(frozen)


@dataclass(kw_only=True, frozen=True, slots=True)
class TaskEpochStartContext:
    epoch: int
    global_step: int
    total_batches: int


@dataclass(kw_only=True, frozen=True, slots=True)
class TaskTrainBoundaryContext:
    epoch: int
    global_step: int
    batch_index: int
    total_batches: int
    did_optimizer_step: bool
    lr: Optional[float]
    batch_metrics: Mapping[str, Any]


@dataclass(kw_only=True, frozen=True, slots=True)
class TaskValidationContext:
    epoch: int
    global_step: int
    evaluation: EvaluationResult
    is_best: bool
    previous_best: Optional[BestMetricSnapshot]
    lr: Optional[float]


@dataclass(kw_only=True, frozen=True, slots=True)
class TaskEpochEndContext:
    completed_epoch: int
    global_step: int
    epoch_metrics: Mapping[str, Any]


@dataclass(kw_only=True, frozen=True, slots=True)
class EpochStartedFact:
    epoch: int
    global_step: int
    total_batches: int
    max_epochs: Optional[int] = None


@dataclass(kw_only=True, frozen=True, slots=True)
class ProgressTickFact:
    stage: Stage
    epoch: int
    global_step: int
    batch_index: int
    total_batches: int
    batch_metrics: Mapping[str, Any]
    average_metrics: Optional[Mapping[str, Any]]
    lr: Optional[float]


@dataclass(kw_only=True, frozen=True, slots=True)
class TrainBoundaryCommittedFact:
    epoch: int
    global_step: int
    batch_index: int
    total_batches: int
    did_optimizer_step: bool
    batch_metrics: Mapping[str, Any]
    lr: Optional[float]


@dataclass(kw_only=True, frozen=True, slots=True)
class EvaluationStartedFact:
    epoch: int
    global_step: int
    total_batches: int
    evaluation_stage: Optional[Literal["val", "test"]] = None
    loader_name: Optional[str] = None
    loader_index: Optional[int] = None
    model_variant: Optional[Literal["standard", "ema"]] = None


@dataclass(kw_only=True, frozen=True, slots=True)
class EvaluationBatchCommittedFact:
    stage: Stage
    epoch: int
    global_step: int
    batch_index: int
    total_batches: int


@dataclass(kw_only=True, frozen=True, slots=True)
class EvaluationCommittedFact:
    epoch: int
    global_step: int
    evaluation: EvaluationResult
    is_best: bool
    previous_best: Optional[BestMetricSnapshot]
    lr: Optional[float]


@dataclass(kw_only=True, frozen=True, slots=True)
class EpochCommittedFact:
    completed_epoch: int
    next_epoch: int
    global_step: int
    epoch_metrics: Mapping[str, Any]


@dataclass(kw_only=True, frozen=True, slots=True)
class StopCommittedFact:
    source: str
    message: str
    epoch: int
    global_step: int


@dataclass(kw_only=True, frozen=True, slots=True)
class EarlyStopDecision:
    should_stop: bool = False
    source: Optional[str] = None
    message: Optional[str] = None


class BoundaryDispatchError(RuntimeError):
    """A rank-consistent failure from a collective-free runner boundary."""


def _settle_error(
    *,
    local_error: Optional[BaseException],
    distributed: bool,
    error_type: type[RuntimeError],
    boundary_name: str,
) -> None:
    if not distributed:
        if local_error is not None:
            raise local_error
        return

    rank = dist.get_rank()
    local_wire = None
    if local_error is not None:
        local_wire = {
            "rank": rank,
            "error_type": type(local_error).__name__,
            "error_message": str(local_error),
        }
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_wire)
    errors = [item for item in gathered if item is not None]
    if not errors:
        return
    first = min(errors, key=lambda item: item["rank"])
    message = f"{boundary_name} failed on rank {first['rank']}: {first['error_type']}: {first['error_message']}"
    if local_error is not None:
        raise error_type(message) from local_error
    raise error_type(message)


def dispatch_protected_boundary(
    operation: Callable[[], None],
    *,
    distributed: bool,
    requires_settlement: bool,
    boundary_name: str,
) -> None:
    """Run local listener work and settle errors before a later collective phase."""
    if not requires_settlement:
        operation()
        return
    local_error: Optional[BaseException] = None
    try:
        operation()
    except BaseException as error:  # settle user callback failures across ranks
        local_error = error
    _settle_error(
        local_error=local_error,
        distributed=distributed,
        error_type=BoundaryDispatchError,
        boundary_name=boundary_name,
    )


class EventListenerBindings:
    """Static in-boundary task callback bindings."""

    _NAMES = ("epoch_start", "train_boundary", "validation", "epoch_end")

    def __init__(self) -> None:
        self._callbacks: dict[str, list[Callable[[object], None]]] = {name: [] for name in self._NAMES}
        self._is_frozen = False

    def bind(self, name: str, callback: Callable[[object], None]) -> None:
        if self._is_frozen:
            raise RuntimeError("Event listener bindings are frozen.")
        if name not in self._callbacks:
            raise ValueError(f"Unknown event-listener binding: {name!r}.")
        self._callbacks[name].append(callback)

    def freeze(self) -> None:
        self._is_frozen = True

    def dispatch(self, name: str, context: object) -> None:
        if not self._is_frozen:
            raise RuntimeError("Event listener bindings must be frozen before dispatch.")
        for callback in self._callbacks[name]:
            callback(context)

    def has(self, name: str) -> bool:
        return bool(self._callbacks[name])


@dataclass(slots=True)
class _ObserverRegistration:
    callback: Callable[[object], None]
    policy: Literal["best_effort", "settled_fatal"]
    optional: bool = False
    is_enabled: bool = True
    has_reported_failure: bool = False
    diagnostic: tuple[str, str] | None = None


class ObserverBindings:
    """Static committed-fact observer bindings with fixed failure policy."""

    _NAMES = (
        "epoch_started",
        "progress_tick",
        "table_update",
        "train_boundary",
        "evaluation_started",
        "evaluation_batch_committed",
        "evaluation_committed",
        "epoch_committed",
        "early_stop",
    )
    _OPTIONAL_NAMES = frozenset(
        (
            "epoch_started",
            "train_boundary",
            "evaluation_started",
            "evaluation_batch_committed",
            "evaluation_committed",
            "epoch_committed",
        )
    )
    _DIAGNOSTIC_LIMIT = 64

    def __init__(self, logger: Optional[Any] = None) -> None:
        self._logger = logger
        self._registrations: dict[str, list[_ObserverRegistration]] = {name: [] for name in self._NAMES}
        self._optional_diagnostics: deque[tuple[_ObserverRegistration, tuple[str, str]]] = deque()
        self._is_frozen = False

    def bind(
        self,
        name: str,
        callback: Callable[[object], None],
        *,
        policy: Literal["best_effort", "settled_fatal"] = "best_effort",
    ) -> None:
        if self._is_frozen:
            raise RuntimeError("Observer bindings are frozen.")
        if name not in self._registrations:
            raise ValueError(f"Unknown observer binding: {name!r}.")
        self._registrations[name].append(_ObserverRegistration(callback=callback, policy=policy))

    def bind_optional_bundle(
        self,
        subscriptions: Iterable[tuple[str, Callable[[object], None]]],
        *,
        policy: Literal["best_effort"] = "best_effort",
    ) -> None:
        """Atomically bind one optional observation plugin's committed-fact callbacks."""
        if self._is_frozen:
            raise RuntimeError("Observer bindings are frozen.")
        if type(policy) is not str or policy != "best_effort":
            raise ValueError("Optional observer callbacks only support the 'best_effort' policy.")

        try:
            bundle = tuple(subscriptions)
        except Exception as error:
            raise TypeError("Optional observer subscriptions must be an iterable of event/callback pairs.") from error

        validated: list[tuple[str, Callable[[object], None]]] = []
        for subscription in bundle:
            if type(subscription) is not tuple or len(subscription) != 2:
                raise TypeError("Each optional observer subscription must be a two-item tuple.")
            name, callback = subscription
            if type(name) is not str or name not in self._OPTIONAL_NAMES:
                raise ValueError("Optional observers may subscribe only to supported committed-fact events.")
            if not callable(callback):
                raise TypeError("Optional observer callbacks must be callable.")
            validated.append((name, callback))

        registrations = [
            (name, _ObserverRegistration(callback=callback, policy="best_effort", optional=True))
            for name, callback in validated
        ]
        for name, registration in registrations:
            self._registrations[name].append(registration)

    def freeze(self) -> None:
        self._is_frozen = True

    def has(self, name: str) -> bool:
        return bool(self._registrations[name])

    def dispatch(self, name: str, fact: object) -> None:
        if not self._is_frozen:
            raise RuntimeError("Observer bindings must be frozen before dispatch.")
        for registration in self._registrations[name]:
            if not registration.is_enabled:
                continue
            if registration.policy == "settled_fatal":
                registration.callback(fact)
                continue
            try:
                registration.callback(fact)
            except Exception as error:
                registration.is_enabled = False
                if not registration.has_reported_failure:
                    registration.has_reported_failure = True
                    if registration.optional:
                        registration.diagnostic = (name, "callback_failure")
                        if len(self._optional_diagnostics) == self._DIAGNOSTIC_LIMIT:
                            expired_registration, expired_diagnostic = self._optional_diagnostics.popleft()
                            if expired_registration.diagnostic == expired_diagnostic:
                                expired_registration.diagnostic = None
                        self._optional_diagnostics.append((registration, registration.diagnostic))
                    elif self._logger is not None:
                        self._logger.debug("Observer %s disabled after failure: %s", name, error, exc_info=True)

    def drain_optional_diagnostics(self, *, limit: int = _DIAGNOSTIC_LIMIT) -> tuple[tuple[str, str], ...]:
        """Drain a bounded set of primitive optional-callback diagnostics without logging."""
        if type(limit) is not int or limit < 0:
            raise ValueError("Diagnostic limit must be a non-negative integer.")
        remaining = min(limit, self._DIAGNOSTIC_LIMIT)
        if remaining == 0:
            return ()
        drained: list[tuple[str, str]] = []
        while self._optional_diagnostics and remaining:
            registration, diagnostic = self._optional_diagnostics.popleft()
            if registration.diagnostic == diagnostic:
                registration.diagnostic = None
                drained.append(diagnostic)
                remaining -= 1
        return tuple(drained)

    def dispatch_terminal(
        self,
        fact: StopCommittedFact,
        *,
        distributed: bool,
    ) -> None:
        local_error: Optional[BaseException] = None
        try:
            self.dispatch("early_stop", fact)
        except BaseException as error:
            local_error = error
        _settle_error(
            local_error=local_error,
            distributed=distributed,
            error_type=BoundaryDispatchError,
            boundary_name="terminal observer dispatch",
        )
