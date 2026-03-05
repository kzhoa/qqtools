"""
Design Philosophy:
1. Trust the listener caller: Do not over-validate event types or callbacks.
2. Trust the task implementer: Do not over-validate return types or data formats.
3. Trust the config user: Do not over-validate configuration values unless they break critical paths.
4. Trust qt.ensure_scala: Assume it can handle various input types and raise if it cannot convert to a scalar.
"""

import gc
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader

import qqtools as qt

from ..entry_utils.qema import qEMA
from ..entry_utils.scheduler import qWarmupScheduler
from ..qlogger import ConsoleLogger, qLogger
from ..task.qtask import qTaskBase
from .runner_utils.avgbank import AvgBank
from .runner_utils.best_model import BestModelTracker
from .runner_utils.common import _is_periodic_trigger, move_batch_to_device
from .runner_utils.ema_context import EMAOffloadContext
from .runner_utils.tensorbank import TensorBank
from .runner_utils.types import EventContext, EventType, LoopSignal, RunConfig, RunningState


def _extract_batch_metric_views(
    batch_metrics: Dict[str, Any],
) -> Tuple[Dict[str, float], List[Tuple[str, float, float]]]:
    """Extract scalar metrics and weighted items in one pass."""
    scalar_metrics: Dict[str, float] = {}
    weighted_items: List[Tuple[str, float, float]] = []

    for key, item in batch_metrics.items():
        if isinstance(item, tuple):
            metric_value = item[0]
            metric_count = item[1] if len(item) > 1 else 1.0
        else:
            metric_value = item
            metric_count = 1.0

        value_float = qt.ensure_scala(metric_value)
        count_float = qt.ensure_scala(metric_count)

        scalar_metrics[key] = value_float
        weighted_items.append((key, value_float, count_float))

    return scalar_metrics, weighted_items


class TaskMetricTracker:
    """Collects batch metrics and optional tensor caches for epoch-level aggregation."""

    def __init__(
        self,
        task: qTaskBase,
        config: RunConfig,
        device: torch.device,
        logger: Optional[qLogger],
        use_tensor_bank: bool = None,
        avg_bank: Optional[AvgBank] = None,
    ):
        self.task = task
        self.config = config
        self.device = device
        self.logger = logger

        has_batch_cache = self.task.has_implemented("batch_cache")
        has_epoch_metric = self.task.has_implemented("epoch_metric")

        if use_tensor_bank is None:
            self.use_tensor_bank = has_batch_cache and has_epoch_metric
            if self.logger and has_batch_cache != has_epoch_metric:
                self.logger.warning(
                    "Task should implement both 'batch_cache' and 'epoch_metric' to enable tensor cache metrics."
                )
        else:
            self.use_tensor_bank = use_tensor_bank

        self.avg_bank = avg_bank or AvgBank()
        self.tensor_bank = TensorBank(logger=logger) if self.use_tensor_bank else None

    def add_weighted_metrics(self, metric_items: List[Tuple[str, float, float]]) -> None:
        for key, metric_value, metric_count in metric_items:
            self.avg_bank.add(key, metric_value, metric_count)

    def update_cache(self, out: Any, batch_data: Any) -> None:
        if not self.use_tensor_bank or self.tensor_bank is None:
            return
        self.tensor_bank.add(self.task.batch_cache(out, batch_data))

    def compute_epoch_metrics(self, reset_after_compute: bool = True) -> Dict[str, Any]:
        epoch_metrics = self.avg_bank.gather_average(self.config.distributed)

        if self.use_tensor_bank and self.tensor_bank is not None:
            gathered_cache = self.tensor_bank.gather(self.config.distributed, self.device)
            task_epoch_metrics = self.task.epoch_metric(gathered_cache)
            if task_epoch_metrics:
                epoch_metrics.update(task_epoch_metrics)

        if reset_after_compute:
            self.reset()

        return epoch_metrics

    def to_dict(self) -> Dict[str, Any]:
        return self.avg_bank.to_dict(self.config.distributed)

    def reset(self) -> None:
        self.avg_bank = AvgBank()
        self.reset_tensor_cache()

    def reset_tensor_cache(self) -> None:
        if self.tensor_bank is not None:
            self.tensor_bank.reset()


def _should_enable_offload(device: torch.device, model: nn.Module, logger: Optional[qLogger]) -> bool:
    """Check if model offloading should be enabled based on GPU capacity."""
    if device.type != "cuda":
        return False

    from ..entry_utils.info import get_model_size_bytes

    m_size = get_model_size_bytes(model)
    try:
        gpu_cap = torch.cuda.get_device_properties(device).total_memory
    except Exception:
        return False

    if m_size > (gpu_cap * 0.5):
        if logger:
            logger.info(
                f"Model size ({m_size / 1024**2:.2f} MB) > 50% of GPU capacity. "
                "Enabling mutual offloading during evaluation."
            )
        return True
    return False


class RunningAgent:
    def __init__(
        self,
        model: nn.Module,
        task: qTaskBase,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[qWarmupScheduler] = None,
        config: RunConfig = None,
        device: torch.device = None,
        ema_model: Optional[qEMA] = None,
        logger: Optional[qLogger] = None,
        listeners: Optional[Dict[str, List[Callable]]] = None,
        state: Optional[RunningState] = None,
        avg_bank: Optional[AvgBank] = None,
        best_model_tracker: Optional[BestModelTracker] = None,
    ):
        self.model = model
        self.task = task
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.config = config if config is not None else RunConfig()

        self.device = device or self.config.device
        self.ema_model = ema_model
        self.logger = logger or ConsoleLogger()

        self.state = state or RunningState()
        self.best_model_tracker = best_model_tracker or BestModelTracker(
            target=self.config.checkpoint.get("target", "val_metric"),
            mode=self.config.checkpoint.get("mode", "min"),
            min_delta=self.config.checkpoint.get("min_delta", 0.0),
        )
        self._ad_hoc_signal = LoopSignal()

        self.train_loader = task.train_loader
        self.val_loader = task.val_loader
        self.test_loader = task.test_loader

        self.interval_avg_bank = AvgBank()

        self.listeners = {event.value: [] for event in EventType}
        if listeners:
            for event, callbacks in listeners.items():
                if event not in self.listeners:
                    self.listeners[event] = []
                self.listeners[event].extend(list(callbacks))

        self.train_metric_tracker = TaskMetricTracker(
            task=self.task,
            config=self.config,
            device=self.device,
            logger=self.logger,
            avg_bank=avg_bank,
        )
        self.use_tensor_bank = self.train_metric_tracker.use_tensor_bank

        self._use_model_offload = False
        if self.ema_model is not None:
            self._use_model_offload = _should_enable_offload(self.device, self.model, self.logger)

    def add_listener(self, event: str, listener: Callable):
        if event in self.listeners:
            self.listeners[event].append(listener)
        else:
            warnings.warn(f"Unknown event: {event}")

    def _trigger(self, event: str, context: EventContext, snapshot: bool = True) -> EventContext:  # noqa: ARG002
        listeners = self.listeners.get(event)
        if not listeners:
            return context

        context.max_epochs = self.config.max_epochs
        context.max_steps = self.config.max_steps

        for listener in listeners:
            listener(context)
        return context

    def _has_listener(self, event: str) -> bool:
        return bool(self.listeners.get(event))

    def _prepare_batch(self, batch_data):
        batch_data = move_batch_to_device(batch_data, self.device)
        return batch_data

    def _forward_batch(self, model: nn.Module, batch_data):
        batch_data = self.task.pre_batch_forward(batch_data)
        out = self.task.batch_forward(model, batch_data)
        out = self.task.post_batch_forward(out, batch_data)
        return out, batch_data

    def _start_new_epoch(self):
        self.train_metric_tracker.reset_tensor_cache()

        if self.config.distributed and hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(self.state.epoch)

        self._trigger(
            "on_epoch_start",
            context=EventContext(state=self.state, total_batches=len(self.train_loader), stage="train"),
        )

    def _handle_epoch_end(self):
        epoch_metrics = self.train_metric_tracker.compute_epoch_metrics(reset_after_compute=True)
        self.state.update_current_metrics(epoch_metrics)
        self._trigger("on_epoch_end", context=EventContext(state=self.state))
        self.state.epoch += 1
        self.state.batch_idx_in_epoch = 0

    def train_batch(self, batch_data) -> Dict[str, Any]:
        return self._train_one_batch(batch_data, emit_events=False)

    def evaluate(self, model: nn.Module = None, use_ema: bool = False) -> Dict[str, Any]:
        base_model = model or self.model
        use_offload = self._use_model_offload and (base_model is self.model)

        with EMAOffloadContext(
            main_model=base_model,
            ema_model=self.ema_model,
            device=self.device,
            use_ema=use_ema,
            use_offload=use_offload,
            logger=self.logger,
        ) as eval_model:
            return self._evaluate_model(eval_model)

    def _evaluate_loaders(self, model: nn.Module) -> Dict[str, Any]:
        return self._evaluate_model(model)

    def _evaluate_model(self, model: nn.Module) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        if self.val_loader is not None:
            results.update(self._evaluate_loader(model, self.val_loader, prefix="val", stage="val"))

        if self.test_loader is not None:
            results.update(self._evaluate_loader(model, self.test_loader, prefix="test", stage="test"))

        return results

    def evaluate_all_models(self) -> Dict[str, Any]:
        eval_results = self.evaluate(self.model, use_ema=False)

        if self.ema_model is not None:
            ema_results = self.evaluate(self.model, use_ema=True)
            eval_results.update({f"ema_{key}": value for key, value in ema_results.items()})

        return eval_results

    def _run_evaluation(
        self, model: nn.Module, data_loader: DataLoader, prefix: str = "", stage: str = "val"
    ) -> Dict[str, Any]:
        return self._evaluate_loader(model=model, data_loader=data_loader, prefix=prefix, stage=stage)

    def _run_evaluation_loop(
        self, model: nn.Module, data_loader: DataLoader, prefix: str = "", stage: str = "val"
    ) -> Dict[str, Any]:
        return self._evaluate_loader(model=model, data_loader=data_loader, prefix=prefix, stage=stage)

    def _evaluate_loader(
        self, model: nn.Module, data_loader: DataLoader, prefix: str = "", stage: str = "val"
    ) -> Dict[str, Any]:
        was_training = model.training
        model.eval()
        total_batches = len(data_loader)

        eval_tracker = TaskMetricTracker(
            task=self.task,
            config=self.config,
            device=self.device,
            logger=self.logger,
        )
        has_progress_tick = self._has_listener("on_progress_tick")
        should_emit_batch = self._has_listener("on_batch_end")
        should_calc_avg = has_progress_tick or should_emit_batch

        try:
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(data_loader):
                    batch_data = self._prepare_batch(batch_data)
                    out, batch_data = self._forward_batch(model, batch_data)

                    raw_batch_metrics = self.task.batch_metric(out, batch_data)
                    scalar_batch_metrics, weighted_items = _extract_batch_metric_views(raw_batch_metrics)
                    eval_tracker.add_weighted_metrics(weighted_items)
                    eval_tracker.update_cache(out, batch_data)

                    avg_metrics = eval_tracker.to_dict() if should_calc_avg else None
                    if should_emit_batch:
                        batch_context = EventContext(
                            state=self.state,
                            batch_idx=batch_idx,
                            total_batches=total_batches,
                            batch_metrics=raw_batch_metrics,
                            avg_bank=avg_metrics,
                            stage=stage,
                        )
                        self._trigger("on_batch_end", context=batch_context)

                    if has_progress_tick:
                        progress_context = EventContext(
                            state=self.state,
                            batch_idx=batch_idx,
                            total_batches=total_batches,
                            batch_metrics=scalar_batch_metrics,
                            avg_bank=avg_metrics,
                            stage=stage,
                        )
                        self._trigger("on_progress_tick", context=progress_context)

            avg_metrics = eval_tracker.compute_epoch_metrics(reset_after_compute=True)
            prefixed_metrics = {f"{prefix}_{key}": value for key, value in avg_metrics.items()}
            prefixed_metrics[f"{prefix}_metric"] = self.task.post_metrics_to_value(avg_metrics)
            return prefixed_metrics
        finally:
            model.train(was_training)

    def _update_best_model_state(self, eval_results: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        tracker = self.best_model_tracker
        if not tracker:
            return False, None

        previous_best = {
            "metric": getattr(tracker, "best_metric", None),
            "epoch": getattr(tracker, "best_epoch", 0),
            "step": getattr(tracker, "best_step", 0),
        }

        target_key = getattr(tracker, "target", None)
        current_val = eval_results.get(target_key)
        if not target_key or current_val is None:
            return False, previous_best

        monitored_metric = qt.ensure_scala(current_val)
        if not tracker.update(monitored_metric, self.state.epoch, self.state.global_step):
            return False, previous_best

        # It's a new best model, update the state
        self.state.best_epoch = self.state.epoch
        self.state.best_step = self.state.global_step
        self.state.best_monitored_key = target_key
        self.state.best_monitored_metric = monitored_metric
        self.state.best_model_metrics_snapshot = dict(eval_results)
        return True, previous_best

    def _run_evaluation_and_update(self, signal: LoopSignal) -> None:
        total_eval_batches = sum(len(loader) for loader in (self.val_loader, self.test_loader) if loader is not None)

        self._trigger(
            "on_eval_start",
            context=EventContext(state=self.state, stage="eval", total_batches=total_eval_batches, signal=signal),
        )

        interval_metrics = self.interval_avg_bank.gather_average(self.config.distributed)
        self.interval_avg_bank = AvgBank()

        eval_results = self.evaluate_all_models()

        if interval_metrics:
            train_metric = self.task.post_metrics_to_value(interval_metrics)
            eval_results.update({f"train_{k}": v for k, v in interval_metrics.items()})
            eval_results["train_metric"] = train_metric

        self.state.update_current_metrics(eval_results)

        self._trigger(
            "on_eval_end",
            context=EventContext(state=self.state, stage="eval", eval_results=eval_results, signal=signal),
        )

        is_best, previous_best = self._update_best_model_state(eval_results)
        if is_best:
            self._request_checkpoint("best", signal=signal)

        validation_context = EventContext(
            state=self.state,
            stage="eval",
            eval_results=eval_results,
            previous_best=previous_best,
            is_best=is_best,
            best_model_tracker=self.best_model_tracker,
            signal=signal,
        )
        self._trigger("on_validation_end", context=validation_context, snapshot=False)

    def _request_checkpoint(
        self,
        checkpoint_type: Literal["best", "regular"],
        signal: Optional[LoopSignal] = None,
    ) -> None:
        target_signal = signal or self._ad_hoc_signal
        target_signal.request_checkpoint(checkpoint_type)

    def _flush_checkpoint_requests(self, signal: Optional[LoopSignal] = None) -> None:
        target_signal = signal or self._ad_hoc_signal
        while target_signal.pending_checkpoint_types:
            checkpoint_type = target_signal.pending_checkpoint_types.pop(0)
            request_context = EventContext(
                state=self.state,
                checkpoint_type=checkpoint_type,
                signal=target_signal,
            )
            self._trigger("on_checkpoint_request", context=request_context, snapshot=False)

            if request_context.checkpoint_path is None:
                continue

            if checkpoint_type == "best":
                self.state.best_ckp_file = Path(request_context.checkpoint_path).name

            self._trigger(
                "on_checkpoint_save",
                context=EventContext(
                    state=self.state,
                    checkpoint_type=checkpoint_type,
                    checkpoint_path=request_context.checkpoint_path,
                    signal=target_signal,
                ),
            )

    def _handle_periodic_events(self, is_epoch_end: bool) -> bool:
        signal = LoopSignal()

        is_eval_trigger = _is_periodic_trigger(
            run_mode=self.config.run_mode,
            interval=self.config.eval_interval,
            global_step=self.state.global_step,
            epoch=self.state.epoch,
            is_epoch_end=is_epoch_end,
        )

        if is_eval_trigger:
            self.logger.debug(
                f"eval trigger: run_mode={self.config.run_mode}, global_step={self.state.global_step}, epoch={self.state.epoch}, is_epoch_end={is_epoch_end}"
                f"\n eval_interval={self.config.eval_interval} save_interval={self.config.save_interval}"
            )
            self._run_evaluation_and_update(signal)

        is_save_trigger = _is_periodic_trigger(
            run_mode=self.config.run_mode,
            interval=self.config.save_interval,
            global_step=self.state.global_step,
            epoch=self.state.epoch,
            is_epoch_end=is_epoch_end,
        )
        if is_save_trigger:
            self._request_checkpoint("regular", signal=signal)

        self._flush_checkpoint_requests(signal)
        signal.synchronize_stop(self.device, self.config.distributed)

        if signal.should_stop:
            stop_message = signal.stop_message or "Early stopping triggered."
            self.logger.info(stop_message)
            self._trigger("on_early_stop", context=EventContext(state=self.state, signal=signal))
            return True

        return False

    def _reached_run_limits(self) -> bool:
        max_steps_limit = self.config.max_steps if self.config.max_steps is not None else float("inf")
        max_epochs_limit = self.config.max_epochs if self.config.max_epochs is not None else float("inf")

        is_reached = self.state.global_step >= max_steps_limit or self.state.epoch >= max_epochs_limit
        if is_reached:
            self.logger.info(f"Training loop stopping at epoch {self.state.epoch}, step {self.state.global_step}.")

        return is_reached

    def _step_warmup_and_maintenance(self) -> None:
        if self.scheduler is not None:
            self.scheduler.step_warmup()

        gc_freq = self.config.gc_freq
        if gc_freq is None or gc_freq <= 0:
            return

        completed_step = self.state.global_step + 1
        if completed_step % gc_freq != 0:
            return

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _train_one_batch(self, batch_data: Any, emit_events: bool = True) -> Dict[str, float]:
        total_batches = len(self.train_loader)
        state = self.state
        batch_idx = state.batch_idx_in_epoch - 1
        batch_start_time = time.time()

        if emit_events:
            self._trigger(
                "on_batch_start",
                context=EventContext(
                    state=state,
                    batch_idx=batch_idx,
                    total_batches=total_batches,
                ),
            )

        batch_data = self._prepare_batch(batch_data)
        out, batch_data = self._forward_batch(self.model, batch_data)

        raw_batch_metrics = self.task.batch_metric(out, batch_data)
        batch_metrics, weighted_items = _extract_batch_metric_views(raw_batch_metrics)
        self.train_metric_tracker.update_cache(out, batch_data)

        losses = self.task.batch_loss(out, batch_data, self.loss_fn)
        loss_tensor, _loss_cnt = losses.get("loss", (None, 1))

        if loss_tensor is not None:
            self.optimizer.zero_grad()
            loss_tensor.backward()

            if self.config.clip_grad is not None:
                clip_grad_norm_(self.model.parameters(), self.config.clip_grad)

            self.optimizer.step()

            if self.ema_model is not None:
                self.ema_model.update()

            loss_value = float(loss_tensor.item())
            batch_metrics["loss"] = loss_value
            weighted_items.append(("loss", loss_value, 1.0))

        if not emit_events:
            return batch_metrics

        batch_time = time.time() - batch_start_time
        state.total_train_time += batch_time
        batch_metrics["batch_time"] = batch_time
        weighted_items.append(("batch_time", float(batch_time), 1.0))

        self.train_metric_tracker.add_weighted_metrics(weighted_items)
        for key, metric_value, metric_count in weighted_items:
            self.interval_avg_bank.add(key, metric_value, metric_count)

        current_lr = self.optimizer.param_groups[0]["lr"] if self.optimizer else None
        self._step_warmup_and_maintenance()

        is_update_tick = state.batch_idx_in_epoch % self.config.print_freq == 0
        is_update_tick = is_update_tick or state.batch_idx_in_epoch == total_batches
        has_batch_end = self._has_listener("on_batch_end")
        has_train_batch_end = self._has_listener("on_train_batch_end")
        if has_batch_end or has_train_batch_end:
            batch_context = EventContext(
                state=state,
                batch_idx=batch_idx,
                total_batches=total_batches,
                batch_metrics=batch_metrics,
                lr=current_lr,
                stage="train",
            )
            if has_batch_end:
                self._trigger("on_batch_end", context=batch_context)
            if has_train_batch_end:
                self._trigger("on_train_batch_end", context=batch_context)

        has_progress_tick = self._has_listener("on_progress_tick")
        has_table_update = is_update_tick and self._has_listener("on_table_update")
        should_need_progress_avg = has_progress_tick or has_table_update
        if should_need_progress_avg:
            progress_avg_metrics = self.train_metric_tracker.to_dict()
            progress_context = EventContext(
                state=state,
                batch_idx=batch_idx,
                total_batches=total_batches,
                batch_metrics=batch_metrics,
                avg_bank=progress_avg_metrics,
                lr=current_lr,
                stage="train",
            )
            if has_progress_tick:
                self._trigger("on_progress_tick", context=progress_context)
            if has_table_update:
                self._trigger("on_table_update", context=progress_context)

        return batch_metrics

    def _training_loop(self) -> bool:
        try:
            while True:
                if self._reached_run_limits():
                    return False

                self._start_new_epoch()
                total_batches = len(self.train_loader)
                start_batch_idx = min(self.state.batch_idx_in_epoch, total_batches)

                for batch_idx, batch_data in enumerate(self.train_loader):
                    if batch_idx < start_batch_idx:
                        continue

                    self.state.batch_idx_in_epoch = batch_idx + 1
                    self._train_one_batch(batch_data, emit_events=True)

                    is_epoch_end = self.state.batch_idx_in_epoch >= total_batches
                    should_stop_mid_epoch = self._handle_periodic_events(is_epoch_end=is_epoch_end)
                    if should_stop_mid_epoch:
                        return True

                    self.state.global_step += 1
                    if self._reached_run_limits():
                        return False

                self._handle_epoch_end()
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user.")
            return True

    def run(self) -> bool:
        max_epochs_limit = self.config.max_epochs if self.config.max_epochs is not None else float("inf")
        max_steps_limit = self.config.max_steps if self.config.max_steps is not None else float("inf")
        self.logger.info(
            f"Starting training (mode={self.config.run_mode.value}, "
            f"eval_interval={self.config.eval_interval}, "
            f"save_interval={self.config.save_interval}, "
            f"max_epochs={max_epochs_limit}, "
            f"max_steps={max_steps_limit})"
        )

        early_stopped = self._training_loop()

        if not early_stopped:
            if self.state.global_step >= max_steps_limit:
                self.logger.info(f"Reached max_steps={max_steps_limit}")
            elif self.state.epoch >= max_epochs_limit:
                self.logger.info(f"Reached max_epochs={max_epochs_limit}")

        return early_stopped
