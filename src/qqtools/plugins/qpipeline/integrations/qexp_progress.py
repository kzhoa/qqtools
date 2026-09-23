"""qexp's optional rank-zero progress observer for qPipeline facts."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from itertools import islice
from types import MappingProxyType
from typing import Any

from ..runner.contracts import (
    EpochCommittedFact,
    EpochStartedFact,
    EvaluationBatchCommittedFact,
    EvaluationCommittedFact,
    EvaluationStartedFact,
    TrainBoundaryCommittedFact,
)
from ..runner.observation_plugins import ObservationContext
from ..types import Stage

_MAX_PROGRESS_INTEGER = 2**63 - 1
_MAX_PROGRESS_METRICS = 32


def _progress_metrics(metrics: object, learning_rate: object = None) -> tuple[dict[object, object], int]:
    """Copy bounded committed facts, retaining their original omission count."""
    copied: dict[object, object] = {}
    source_count = 0
    if type(metrics) in (dict, MappingProxyType):
        source_count = len(metrics)
        copied.update(islice(metrics.items(), _MAX_PROGRESS_METRICS))

    if len(copied) < _MAX_PROGRESS_METRICS:
        if type(learning_rate) is int or (type(learning_rate) is float and math.isfinite(learning_rate)):
            if "lr" not in copied:
                copied["lr"] = learning_rate
                source_count += 1
    return copied, source_count


def _as_int(value: object) -> int | None:
    """Accept only bounded built-in integer facts; never invoke user conversion."""
    if type(value) is not int or not 0 <= value <= _MAX_PROGRESS_INTEGER:
        return None
    return value


def _as_position(value: object) -> int | None:
    integer = _as_int(value)
    return integer if integer is not None and integer < _MAX_PROGRESS_INTEGER else None


def _as_total(value: object) -> int | None:
    if type(value) is not int or not -(2**63) <= value <= _MAX_PROGRESS_INTEGER:
        return None
    return value


def _safe_text(value: object, *, limit: int = 256) -> str | None:
    if type(value) is not str or not value or len(value) > limit:
        return None
    try:
        if len(value.encode("utf-8")) > limit:
            return None
    except UnicodeEncodeError:
        return None
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        return None
    return value


def _stage_value(value: object) -> str | None:
    if type(value) is Stage:
        if value is Stage.VAL:
            return "val"
        if value is Stage.TEST:
            return "test"
        return None
    if type(value) is str:
        if len(value) <= len("validation") and value in {"val", "validation"}:
            return "val"
        if value == "test":
            return "test"
    return None


def _join_message(*parts: str | None) -> str | None:
    values = [part for part in parts if part]
    return " · ".join(values) or None


def _epoch_message(epoch: int | None, max_epochs: int | None, *, completed: bool = False) -> str | None:
    if epoch is None:
        return None
    message = f"Epoch {epoch + 1}"
    if max_epochs is not None and max_epochs > 0:
        message += f"/{max_epochs}"
    if completed:
        message += " completed"
    return message


def _batch_message(batch_index: int | None, total_batches: int | None) -> str | None:
    if batch_index is None:
        return None
    message = f"Batch {batch_index + 1}"
    if total_batches is not None and total_batches > 0:
        message += f"/{total_batches}"
    return message


def _model_message(model_variant: str | None) -> str | None:
    if model_variant == "standard":
        return "Standard"
    if model_variant == "ema":
        return "EMA"
    return None


def _render_message(parts: tuple[str | int | None, ...]) -> str | None:
    """Format already bounded primitive context on the producer writer thread."""
    if not parts:
        return None
    kind = parts[0]
    if kind == "training" and len(parts) == 6:
        _, epoch, max_epochs, state, batch_index, total_batches = parts
        completed = state == "completed"
        return _join_message(
            _epoch_message(
                epoch if type(epoch) is int else None,
                max_epochs if type(max_epochs) is int else None,
                completed=completed,
            ),
            None
            if completed
            else _batch_message(
                batch_index if type(batch_index) is int else None,
                total_batches if type(total_batches) is int else None,
            ),
        )
    if kind == "evaluation" and len(parts) == 6:
        _, loader_name, loader_index, model_variant, epoch, max_epochs = parts
        loader = loader_name if type(loader_name) is str else None
        if not loader and type(loader_index) is int:
            loader = f"Loader {loader_index + 1}"
        return _join_message(
            loader,
            _model_message(model_variant if type(model_variant) is str else None),
            _epoch_message(epoch if type(epoch) is int else None, max_epochs if type(max_epochs) is int else None),
        )
    return None


@dataclass(slots=True)
class _TrainingContext:
    epoch: int | None
    global_step: int | None
    batch_index: int | None
    total_batches: int | None
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
    """Translate bounded committed facts into producer-owned progress updates."""

    identifier = "qexp_progress"

    def __init__(self, progress_api: Any, context: ObservationContext) -> None:
        self._progress_api = progress_api
        self._has_v2_channel = bool(os.environ.get("QEXP_PROGRESS_V2_PATH"))
        self._max_epochs = _as_int(context.max_epochs)
        self._max_steps = _as_int(context.max_steps)
        self._training_context: _TrainingContext | None = None
        self._evaluation_context: _EvaluationContext | None = None
        self.subscriptions = (
            ("epoch_started", self.on_epoch_start),
            ("train_boundary", self.on_train_boundary),
            ("evaluation_started", self.on_evaluation_started),
            ("evaluation_batch_committed", self.on_evaluation_batch_committed),
            ("evaluation_committed", self.on_evaluation_committed),
            ("epoch_committed", self.on_epoch_committed),
        )

    def _training_parts(self, context: _TrainingContext) -> tuple[str | int | None, ...]:
        return (
            "training",
            context.epoch,
            self._max_epochs,
            "completed" if context.is_completed else "active",
            context.batch_index,
            context.total_batches,
        )

    def _emit_training(
        self, metrics: dict[object, object] | None = None, metrics_source_count: int | None = None
    ) -> None:
        context = self._training_context
        if context is None:
            return
        metrics_kwargs = (
            {"metrics": metrics, "_metrics_source_count": metrics_source_count} if self._has_v2_channel else {}
        )
        self._progress_api._offer_managed_progress(
            stage="train",
            current=context.global_step,
            total=self._max_steps,
            unit="step",
            message_parts=self._training_parts(context),
            render_message=_render_message,
            **metrics_kwargs,
        )

    def _evaluation_parts(self, context: _EvaluationContext) -> tuple[str | int | None, ...]:
        return (
            "evaluation",
            context.loader_name,
            context.loader_index,
            context.model_variant,
            context.epoch,
            self._max_epochs,
        )

    def _emit_evaluation_start(self) -> None:
        context = self._evaluation_context
        if context is None or context.stage is None:
            return
        total = context.total_batches if context.total_batches is not None and context.total_batches > 0 else None
        current = 0 if total is not None else None
        self._progress_api._offer_managed_progress(
            stage="validation" if context.stage == "val" else "test",
            current=current,
            total=total,
            unit="batch",
            message_parts=self._evaluation_parts(context),
            render_message=_render_message,
        )

    def on_epoch_start(self, fact: object) -> None:
        if type(fact) is not EpochStartedFact:
            return
        max_epochs = _as_int(fact.max_epochs)
        if max_epochs is not None:
            self._max_epochs = max_epochs
        self._training_context = _TrainingContext(
            epoch=_as_position(fact.epoch),
            global_step=_as_int(fact.global_step),
            batch_index=None,
            total_batches=_as_int(fact.total_batches),
        )
        self._emit_training()

    def on_train_boundary(self, fact: object) -> None:
        if type(fact) is not TrainBoundaryCommittedFact:
            return
        metrics, source_count = _progress_metrics(fact.batch_metrics, fact.lr) if self._has_v2_channel else (None, None)
        self._training_context = _TrainingContext(
            epoch=_as_position(fact.epoch),
            global_step=_as_int(fact.global_step),
            batch_index=_as_position(fact.batch_index),
            total_batches=_as_int(fact.total_batches),
        )
        self._emit_training(metrics=metrics, metrics_source_count=source_count)

    def on_epoch_committed(self, fact: object) -> None:
        if type(fact) is not EpochCommittedFact:
            return
        metrics, source_count = _progress_metrics(fact.epoch_metrics) if self._has_v2_channel else (None, None)
        completed_epoch = _as_position(fact.completed_epoch)
        if completed_epoch is None and self._training_context is not None:
            completed_epoch = self._training_context.epoch
        global_step = _as_int(fact.global_step)
        if global_step is None and self._training_context is not None:
            global_step = self._training_context.global_step
        self._training_context = _TrainingContext(
            epoch=completed_epoch,
            global_step=global_step,
            batch_index=None,
            total_batches=None,
            is_completed=True,
        )
        self._emit_training(metrics=metrics, metrics_source_count=source_count)

    def on_evaluation_started(self, fact: object) -> None:
        if type(fact) is not EvaluationStartedFact:
            return
        model_variant = fact.model_variant if type(fact.model_variant) is str else None
        self._evaluation_context = _EvaluationContext(
            stage=_stage_value(fact.evaluation_stage),
            epoch=_as_position(fact.epoch),
            global_step=_as_int(fact.global_step),
            total_batches=_as_total(fact.total_batches),
            loader_name=_safe_text(fact.loader_name),
            loader_index=_as_position(fact.loader_index),
            model_variant=model_variant if _safe_text(model_variant, limit=16) in {"standard", "ema"} else None,
        )
        self._emit_evaluation_start()

    def on_evaluation_batch_committed(self, fact: object) -> None:
        if type(fact) is not EvaluationBatchCommittedFact:
            return
        stage = _stage_value(fact.stage)
        if stage is None:
            return
        context = self._evaluation_context
        if context is not None and context.stage == stage:
            total = context.total_batches
            message_parts = self._evaluation_parts(context)
        else:
            total = _as_total(fact.total_batches)
            message_parts = ()
        batch_index = _as_position(fact.batch_index)
        current = batch_index + 1 if batch_index is not None else None
        if total is not None and (total <= 0 or (current is not None and current > total)):
            current, total = None, None
        self._progress_api._offer_managed_progress(
            stage="validation" if stage == "val" else "test",
            current=current,
            total=total,
            unit="batch",
            message_parts=message_parts,
            render_message=_render_message,
        )

    def on_evaluation_committed(self, fact: object) -> None:
        if type(fact) is not EvaluationCommittedFact:
            return
        self._evaluation_context = None
        self._emit_training()

    def close(self) -> None:
        self._progress_api._close_managed_progress()


def qexp_progress_factory(context: ObservationContext) -> QexpProgressObserver | None:
    """Create the connector only for a configured primary-rank progress channel."""
    if not os.environ.get("QEXP_PROGRESS_PATH") or type(context.rank) is not int or context.rank != 0:
        return None
    from qqtools.qexp import progress as progress_api

    return QexpProgressObserver(progress_api, context)
