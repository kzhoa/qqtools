from __future__ import annotations

from typing import Any, Dict, Optional

from ..runner_utils.types import LoopSignal, RunningState, Stage
from .dispatcher import EventDispatcher


def _dispatch_context(
    dispatcher: EventDispatcher,
    event: str,
    *,
    state: RunningState,
    stage: Optional[Stage],
    max_epochs: Optional[int],
    max_steps: Optional[int],
    signal: Optional[LoopSignal] = None,
    **payload: Any,
) -> None:
    dispatcher.dispatch(
        event,
        state=state,
        stage=stage,
        max_epochs=max_epochs,
        max_steps=max_steps,
        signal=signal,
        **payload,
    )


def emit_epoch_start(dispatcher: EventDispatcher, *, state: RunningState, stage: Stage, max_epochs: Optional[int], max_steps: Optional[int]) -> None:
    _dispatch_context(dispatcher, "on_epoch_start", state=state, stage=stage, max_epochs=max_epochs, max_steps=max_steps)


def emit_epoch_start_internal(
    dispatcher: EventDispatcher,
    *,
    state: RunningState,
    stage: Stage,
    total_batches: int,
    max_epochs: Optional[int],
    max_steps: Optional[int],
) -> None:
    _dispatch_context(
        dispatcher,
        "on_epoch_start_internal",
        state=state,
        stage=stage,
        total_batches=total_batches,
        max_epochs=max_epochs,
        max_steps=max_steps,
    )


def emit_epoch_end(dispatcher: EventDispatcher, *, state: RunningState, stage: Stage, max_epochs: Optional[int], max_steps: Optional[int]) -> None:
    _dispatch_context(dispatcher, "on_epoch_end", state=state, stage=stage, max_epochs=max_epochs, max_steps=max_steps)


def emit_batch_start(
    dispatcher: EventDispatcher,
    *,
    state: RunningState,
    stage: Stage,
    batch_idx: int,
    total_batches: int,
    max_epochs: Optional[int],
    max_steps: Optional[int],
) -> None:
    _dispatch_context(
        dispatcher,
        "on_batch_start",
        state=state,
        stage=stage,
        batch_idx=batch_idx,
        total_batches=total_batches,
        max_epochs=max_epochs,
        max_steps=max_steps,
    )


def emit_batch_end(
    dispatcher: EventDispatcher,
    *,
    state: RunningState,
    stage: Stage,
    batch_idx: int,
    total_batches: int,
    batch_metrics: Dict[str, Any],
    avg_bank: Optional[Dict[str, Any]],
    lr: Optional[float],
    max_epochs: Optional[int],
    max_steps: Optional[int],
) -> None:
    _dispatch_context(
        dispatcher,
        "on_batch_end",
        state=state,
        stage=stage,
        batch_idx=batch_idx,
        total_batches=total_batches,
        batch_metrics=batch_metrics,
        avg_bank=avg_bank,
        lr=lr,
        max_epochs=max_epochs,
        max_steps=max_steps,
    )


def emit_progress_tick(
    dispatcher: EventDispatcher,
    *,
    state: RunningState,
    stage: Stage,
    batch_idx: int,
    total_batches: int,
    batch_metrics: Dict[str, Any],
    avg_bank: Optional[Dict[str, Any]],
    lr: Optional[float],
    max_epochs: Optional[int],
    max_steps: Optional[int],
) -> None:
    _dispatch_context(
        dispatcher,
        "on_progress_tick",
        state=state,
        stage=stage,
        batch_idx=batch_idx,
        total_batches=total_batches,
        batch_metrics=batch_metrics,
        avg_bank=avg_bank,
        lr=lr,
        max_epochs=max_epochs,
        max_steps=max_steps,
    )


def emit_table_update(
    dispatcher: EventDispatcher,
    *,
    state: RunningState,
    stage: Stage,
    batch_idx: int,
    total_batches: int,
    batch_metrics: Dict[str, Any],
    avg_bank: Optional[Dict[str, Any]],
    lr: Optional[float],
    max_epochs: Optional[int],
    max_steps: Optional[int],
) -> None:
    _dispatch_context(
        dispatcher,
        "on_table_update",
        state=state,
        stage=stage,
        batch_idx=batch_idx,
        total_batches=total_batches,
        batch_metrics=batch_metrics,
        avg_bank=avg_bank,
        lr=lr,
        max_epochs=max_epochs,
        max_steps=max_steps,
    )


def emit_train_batch_end(
    dispatcher: EventDispatcher,
    *,
    state: RunningState,
    stage: Stage,
    batch_idx: int,
    total_batches: int,
    batch_metrics: Dict[str, Any],
    avg_bank: Optional[Dict[str, Any]],
    lr: Optional[float],
    max_epochs: Optional[int],
    max_steps: Optional[int],
) -> None:
    _dispatch_context(
        dispatcher,
        "on_train_batch_end",
        state=state,
        stage=stage,
        batch_idx=batch_idx,
        total_batches=total_batches,
        batch_metrics=batch_metrics,
        avg_bank=avg_bank,
        lr=lr,
        max_epochs=max_epochs,
        max_steps=max_steps,
    )


def emit_eval_start(
    dispatcher: EventDispatcher,
    *,
    state: RunningState,
    total_batches: int,
    signal: LoopSignal,
    max_epochs: Optional[int],
    max_steps: Optional[int],
) -> None:
    _dispatch_context(
        dispatcher,
        "on_eval_start",
        state=state,
        stage=None,
        total_batches=total_batches,
        signal=signal,
        max_epochs=max_epochs,
        max_steps=max_steps,
    )


def emit_eval_end(
    dispatcher: EventDispatcher,
    *,
    state: RunningState,
    eval_results: Dict[str, Any],
    signal: LoopSignal,
    max_epochs: Optional[int],
    max_steps: Optional[int],
) -> None:
    _dispatch_context(
        dispatcher,
        "on_eval_end",
        state=state,
        stage=None,
        eval_results=eval_results,
        signal=signal,
        max_epochs=max_epochs,
        max_steps=max_steps,
    )


def emit_validation_end(
    dispatcher: EventDispatcher,
    *,
    state: RunningState,
    eval_results: Dict[str, Any],
    lr: Optional[float],
    previous_best: Optional[Dict[str, Any]],
    is_best: bool,
    best_model_tracker: Any,
    signal: LoopSignal,
    max_epochs: Optional[int],
    max_steps: Optional[int],
) -> None:
    _dispatch_context(
        dispatcher,
        "on_validation_end",
        state=state,
        stage=None,
        eval_results=eval_results,
        lr=lr,
        previous_best=previous_best,
        is_best=is_best,
        best_model_tracker=best_model_tracker,
        signal=signal,
        max_epochs=max_epochs,
        max_steps=max_steps,
    )


def emit_checkpoint_request(
    dispatcher: EventDispatcher,
    *,
    state: RunningState,
    checkpoint_type: str,
    signal: LoopSignal,
    max_epochs: Optional[int],
    max_steps: Optional[int],
) -> None:
    _dispatch_context(
        dispatcher,
        "on_checkpoint_request",
        state=state,
        stage=None,
        checkpoint_type=checkpoint_type,
        signal=signal,
        max_epochs=max_epochs,
        max_steps=max_steps,
    )


def emit_early_stop(
    dispatcher: EventDispatcher,
    *,
    state: RunningState,
    signal: LoopSignal,
    max_epochs: Optional[int],
    max_steps: Optional[int],
) -> None:
    _dispatch_context(
        dispatcher,
        "on_early_stop",
        state=state,
        stage=None,
        signal=signal,
        max_epochs=max_epochs,
        max_steps=max_steps,
    )


def emit_loss_computed(
    dispatcher: EventDispatcher,
    *,
    state: RunningState,
    stage: Stage,
    batch_idx: int,
    total_batches: int,
    batch_data: Any,
    loss_tensor: Any,
    model_output: Any,
    batch_replay_ref: Any,
    lr: Optional[float],
    max_epochs: Optional[int],
    max_steps: Optional[int],
) -> None:
    _dispatch_context(
        dispatcher,
        "on_loss_computed",
        state=state,
        stage=stage,
        batch_idx=batch_idx,
        total_batches=total_batches,
        batch_data=batch_data,
        loss_tensor=loss_tensor,
        model_output=model_output,
        batch_replay_ref=batch_replay_ref,
        lr=lr,
        max_epochs=max_epochs,
        max_steps=max_steps,
    )
