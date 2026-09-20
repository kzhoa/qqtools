"""Optional rank-zero qexp observer, independent of terminal renderers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from qqtools.qexp import progress as progress_api


def _as_int(value: Any) -> int | None:
    """Return an integer fact value, excluding booleans and unknown values."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _stage_value(value: Any) -> str | None:
    """Normalize qPipeline stage values to the semantic val/test names."""
    value = getattr(value, "value", value)
    if value in {"val", "validation"}:
        return "val"
    if value == "test":
        return "test"
    return None


def _epoch_message(
    epoch: int | None,
    max_epochs: int | None,
    *,
    completed: bool = False,
) -> str | None:
    """Format a one-based active or completed epoch position."""
    if epoch is None or epoch < 0:
        return None
    message = f"Epoch {epoch + 1}"
    if max_epochs is not None:
        message += f"/{max_epochs}"
    if completed:
        message += " completed"
    return message


def _batch_message(batch_index: int | None, total_batches: int | None) -> str | None:
    """Format a one-based batch position without inventing an unknown bound."""
    if batch_index is None or batch_index < 0:
        return None
    message = f"Batch {batch_index + 1}"
    if total_batches is not None and total_batches > 0:
        message += f"/{total_batches}"
    return message


def _join_message(*parts: str | None) -> str | None:
    values = [part for part in parts if part]
    return " · ".join(values) or None


def _model_message(model_variant: str | None) -> str | None:
    if model_variant == "standard":
        return "Standard"
    if model_variant == "ema":
        return "EMA"
    return None


@dataclass(slots=True)
class _TrainingContext:
    epoch: int | None
    global_step: int | None
    batch_index: int | None
    total_batches: int | None
    max_epochs: int | None
    is_completed: bool = False


@dataclass(slots=True)
class _EvaluationContext:
    stage: str | None
    epoch: int | None
    global_step: int | None
    total_batches: int | None
    loader_name: str | None
    loader_index: int | None
    model_variant: str | None


class QexpProgressObserver:
    """Translate committed runner facts into the process singleton writer."""

    def __init__(self, config: Any) -> None:
        self._config = config
        self._max_epochs = _as_int(getattr(config, "max_epochs", None))
        self._training_context: _TrainingContext | None = None
        self._evaluation_context: _EvaluationContext | None = None

    def _training_message(self, context: _TrainingContext) -> str | None:
        return _join_message(
            _epoch_message(context.epoch, context.max_epochs, completed=context.is_completed),
            None if context.is_completed else _batch_message(context.batch_index, context.total_batches),
        )

    def _emit_training(self) -> None:
        context = self._training_context
        if context is None:
            return
        progress_api.update(
            stage="train",
            current=context.global_step,
            total=self._config.max_steps,
            unit="step",
            message=self._training_message(context),
        )

    def _evaluation_message(self, context: _EvaluationContext) -> str | None:
        loader = context.loader_name
        if not loader and context.loader_index is not None:
            loader = f"Loader {context.loader_index + 1}"
        return _join_message(
            loader,
            _model_message(context.model_variant),
            _epoch_message(context.epoch, self._max_epochs),
        )

    def _emit_evaluation_start(self) -> None:
        context = self._evaluation_context
        if context is None or context.stage is None:
            return
        total = context.total_batches if context.total_batches and context.total_batches > 0 else None
        current = 0 if total is not None else None
        progress_api.update(
            stage="validation" if context.stage == "val" else "test",
            current=current,
            total=total,
            unit="batch",
            message=self._evaluation_message(context),
        )

    def on_epoch_start(self, fact: Any) -> None:
        max_epochs = _as_int(getattr(fact, "max_epochs", None))
        if max_epochs is not None:
            self._max_epochs = max_epochs
        self._training_context = _TrainingContext(
            epoch=_as_int(getattr(fact, "epoch", None)),
            global_step=_as_int(getattr(fact, "global_step", None)),
            batch_index=None,
            total_batches=_as_int(getattr(fact, "total_batches", None)),
            max_epochs=self._max_epochs,
        )
        self._emit_training()

    def on_train_boundary(self, fact: Any) -> None:
        self._training_context = _TrainingContext(
            epoch=_as_int(getattr(fact, "epoch", None)),
            global_step=_as_int(getattr(fact, "global_step", None)),
            batch_index=_as_int(getattr(fact, "batch_index", None)),
            total_batches=_as_int(getattr(fact, "total_batches", None)),
            max_epochs=self._max_epochs,
        )
        self._emit_training()

    def on_train(self, fact: Any) -> None:
        """Retain the former adapter entry point for direct integrations."""
        self.on_train_boundary(fact)

    def on_epoch_committed(self, fact: Any) -> None:
        completed_epoch = _as_int(getattr(fact, "completed_epoch", None))
        if completed_epoch is None and self._training_context is not None:
            completed_epoch = self._training_context.epoch
        global_step = _as_int(getattr(fact, "global_step", None))
        if global_step is None and self._training_context is not None:
            global_step = self._training_context.global_step
        self._training_context = _TrainingContext(
            epoch=completed_epoch,
            global_step=global_step,
            batch_index=None,
            total_batches=None,
            max_epochs=self._max_epochs,
            is_completed=True,
        )
        self._emit_training()

    def on_evaluation_started(self, fact: Any) -> None:
        loader_name = getattr(fact, "loader_name", None)
        if not isinstance(loader_name, str) or not loader_name:
            loader_name = None
        self._evaluation_context = _EvaluationContext(
            stage=_stage_value(getattr(fact, "evaluation_stage", None)),
            epoch=_as_int(getattr(fact, "epoch", None)),
            global_step=_as_int(getattr(fact, "global_step", None)),
            total_batches=_as_int(getattr(fact, "total_batches", None)),
            loader_name=loader_name,
            loader_index=_as_int(getattr(fact, "loader_index", None)),
            model_variant=getattr(fact, "model_variant", None),
        )
        self._emit_evaluation_start()

    def on_progress_tick(self, fact: Any) -> None:
        stage = _stage_value(getattr(fact, "stage", None))
        if stage is None:
            return

        context = self._evaluation_context
        if context is not None and context.stage == stage:
            total = context.total_batches
            message = self._evaluation_message(context)
        else:
            total = _as_int(getattr(fact, "total_batches", None))
            message = None

        batch_index = _as_int(getattr(fact, "batch_index", None))
        current = batch_index + 1 if batch_index is not None and batch_index >= 0 else None
        if total is not None and total <= 0:
            current, total = None, None
        elif total is not None and current is not None and current > total:
            current, total = None, None
        progress_api.update(
            stage="validation" if stage == "val" else "test",
            current=current,
            total=total,
            unit="batch",
            message=message,
        )

    def on_evaluation_committed(self, fact: Any) -> None:
        self._evaluation_context = None
        self._emit_training()

    def close(self) -> None:
        progress_api.flush()


def bind_qexp_progress(observers: Any, config: Any) -> QexpProgressObserver | None:
    """Attach peer observers before bindings freeze; absent qexp means no work."""
    if not os.environ.get("QEXP_PROGRESS_PATH") or config.rank != 0:
        return None
    try:
        observer = QexpProgressObserver(config)
        for name, callback in (
            ("epoch_started", observer.on_epoch_start),
            ("train_boundary", observer.on_train_boundary),
            ("evaluation_started", observer.on_evaluation_started),
            ("progress_tick", observer.on_progress_tick),
            ("evaluation_committed", observer.on_evaluation_committed),
            ("epoch_committed", observer.on_epoch_committed),
        ):
            observers.bind(name, callback, policy="best_effort")
        return observer
    except Exception:
        return None
