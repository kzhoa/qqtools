from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Type

from ..runner_utils.types import LoopSignal, RunningState, Stage


class EventName(Enum):
    ON_EPOCH_START = "on_epoch_start"
    ON_EPOCH_START_INTERNAL = "on_epoch_start_internal"
    ON_EPOCH_END = "on_epoch_end"
    ON_BATCH_START = "on_batch_start"
    ON_BATCH_END = "on_batch_end"
    ON_PROGRESS_TICK = "on_progress_tick"
    ON_TABLE_UPDATE = "on_table_update"
    ON_TRAIN_BATCH_END = "on_train_batch_end"
    ON_EVAL_START = "on_eval_start"
    ON_EVAL_END = "on_eval_end"
    ON_VALIDATION_END = "on_validation_end"
    ON_CHECKPOINT_REQUEST = "on_checkpoint_request"
    ON_EARLY_STOP = "on_early_stop"
    ON_LOSS_COMPUTED = "on_loss_computed"


@dataclass(kw_only=True, frozen=True)
class RunnerRuntimeView:
    run_state: RunningState
    stage: Optional[Stage]
    max_epochs: Optional[int]
    max_steps: Optional[int]


@dataclass(kw_only=True)
class BaseEventContext:
    runner: RunnerRuntimeView
    signal: Optional[LoopSignal] = None


@dataclass(kw_only=True)
class ProgressEventContext(BaseEventContext):
    batch_idx: int
    total_batches: int
    batch_metrics: Dict[str, Any]
    avg_bank: Optional[Dict[str, Any]] = None
    lr: Optional[float] = None


@dataclass(kw_only=True)
class ValidationEndEventContext(BaseEventContext):
    eval_results: Dict[str, Any]
    lr: Optional[float]
    previous_best: Optional[Dict[str, Any]]
    is_best: bool
    best_model_tracker: Optional[Any]
    signal: LoopSignal


@dataclass(kw_only=True)
class CheckpointRequestEventContext(BaseEventContext):
    checkpoint_type: str
    signal: LoopSignal


@dataclass(kw_only=True)
class LossComputedEventContext(BaseEventContext):
    batch_idx: int
    total_batches: int
    batch_data: Any
    loss_tensor: Any
    model_output: Any
    batch_replay_ref: Any = None
    lr: Optional[float] = None


@dataclass(kw_only=True)
class _EpochStartInternalContext(BaseEventContext):
    total_batches: int


@dataclass(kw_only=True)
class _BatchBoundaryInternalContext(BaseEventContext):
    batch_idx: int
    total_batches: int
    batch_metrics: Optional[Dict[str, Any]] = None
    avg_bank: Optional[Dict[str, Any]] = None
    lr: Optional[float] = None


@dataclass(kw_only=True)
class _EvalStartInternalContext(BaseEventContext):
    total_batches: int
    signal: LoopSignal


@dataclass(kw_only=True)
class _EvalEndInternalContext(BaseEventContext):
    eval_results: Dict[str, Any]
    signal: LoopSignal


@dataclass(frozen=True)
class EventSpec:
    name: EventName
    context_type: Type[BaseEventContext]
    # Default contract: run_state is passed through directly as the current
    # RunningState object. The framework does not enforce immutability for
    # event consumers; behavioral constraints are documented rather than
    # guarded at runtime.
    # Other payload fields intentionally remain ordinary Python objects. The
    # framework does not add generic immutability guards for them so callers
    # can keep that freedom when they explicitly want to mutate event data.


EVENT_SPECS: Dict[str, EventSpec] = {
    EventName.ON_EPOCH_START.value: EventSpec(
        name=EventName.ON_EPOCH_START,
        context_type=BaseEventContext,
    ),
    EventName.ON_EPOCH_START_INTERNAL.value: EventSpec(
        name=EventName.ON_EPOCH_START_INTERNAL,
        context_type=_EpochStartInternalContext,
    ),
    EventName.ON_EPOCH_END.value: EventSpec(
        name=EventName.ON_EPOCH_END,
        context_type=BaseEventContext,
    ),
    EventName.ON_BATCH_START.value: EventSpec(
        name=EventName.ON_BATCH_START,
        context_type=_BatchBoundaryInternalContext,
    ),
    EventName.ON_BATCH_END.value: EventSpec(
        name=EventName.ON_BATCH_END,
        context_type=_BatchBoundaryInternalContext,
    ),
    EventName.ON_PROGRESS_TICK.value: EventSpec(
        name=EventName.ON_PROGRESS_TICK,
        context_type=ProgressEventContext,
    ),
    EventName.ON_TABLE_UPDATE.value: EventSpec(
        name=EventName.ON_TABLE_UPDATE,
        context_type=ProgressEventContext,
    ),
    EventName.ON_TRAIN_BATCH_END.value: EventSpec(
        name=EventName.ON_TRAIN_BATCH_END,
        context_type=ProgressEventContext,
    ),
    EventName.ON_EVAL_START.value: EventSpec(
        name=EventName.ON_EVAL_START,
        context_type=_EvalStartInternalContext,
    ),
    EventName.ON_EVAL_END.value: EventSpec(
        name=EventName.ON_EVAL_END,
        context_type=_EvalEndInternalContext,
    ),
    EventName.ON_VALIDATION_END.value: EventSpec(
        name=EventName.ON_VALIDATION_END,
        context_type=ValidationEndEventContext,
    ),
    EventName.ON_CHECKPOINT_REQUEST.value: EventSpec(
        name=EventName.ON_CHECKPOINT_REQUEST,
        context_type=CheckpointRequestEventContext,
    ),
    EventName.ON_EARLY_STOP.value: EventSpec(
        name=EventName.ON_EARLY_STOP,
        context_type=BaseEventContext,
    ),
    EventName.ON_LOSS_COMPUTED.value: EventSpec(
        name=EventName.ON_LOSS_COMPUTED,
        context_type=LossComputedEventContext,
    ),
}
