"""
Design Philosophy:
1. Trust the listener caller: Do not over-validate event types or callbacks.
2. Trust the task implementer: Do not over-validate return types or data formats.
3. Trust the config user: Do not over-validate configuration values unless they break critical paths.
4. Trust qt.ensure_scala: Assume it can handle various input types and raise if it cannot convert to a scalar.
"""

import copy
import gc
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

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
from .runner_utils.types import EventContext, EventType, LoopSignal, RunConfig, RunMode, RunningState


def _get_scalar_metrics(batch_metrics: Dict[str, Any]) -> Dict[str, float]:
    """Extracts scalar float values from a metric dictionary."""
    scalar_metrics = {}
    for key, item in batch_metrics.items():
        value = item[0] if isinstance(item, (tuple, list)) else item
        scalar_metrics[key] = qt.ensure_scala(value)
    return scalar_metrics


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
        allow_auto_offload: bool = True,
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

        self.train_avg_bank = avg_bank or AvgBank()

        has_batch_cache = self.task.has_implemented("batch_cache")
        has_epoch_metric = self.task.has_implemented("epoch_metric")
        self.use_tensor_bank = has_batch_cache and has_epoch_metric
        if self.logger and has_batch_cache != has_epoch_metric:
            self.logger.warning(
                "Task should implement both 'batch_cache' and 'epoch_metric' to enable tensor cache metrics."
            )

        self.train_tensor_bank = TensorBank(logger=logger) if self.use_tensor_bank else None

        self.interval_avg_bank = AvgBank()

        self.listeners = {event.value: [] for event in EventType}
        if listeners:
            for event, callbacks in listeners.items():
                event_name = self._validate_event_name(event)
                self.listeners[event_name].extend(list(callbacks))

        self._ema_offload_ctx = EMAOffloadContext(
            main_model=self.model,
            ema_model=self.ema_model,
            device=self.device,
            logger=self.logger,
            allow_auto_offload=allow_auto_offload,
        )
        config_accum_grad = self.config.accum_grad
        self.accum_grad = 1 if config_accum_grad in (None, 1) else int(config_accum_grad)
        self.has_accum_grad = self.accum_grad > 1
        self._apply_train_step_impl = (
            self._apply_train_step_accum if self.has_accum_grad else self._apply_train_step_plain
        )
        self._current_accum_loss_count = 0.0

    def _validate_event_name(self, event: Union[str, EventType]) -> str:
        event_name = event.value if isinstance(event, EventType) else str(event)
        if event_name not in self.listeners:
            allowed_events = ", ".join(sorted(self.listeners.keys()))
            raise ValueError(
                f"Unknown event: {event_name}. Register it in EventType first. " f"Allowed events: {allowed_events}"
            )
        return event_name

    def add_listener(self, event: Union[str, EventType], listener: Callable):
        event_name = self._validate_event_name(event)
        self.listeners[event_name].append(listener)

    def _check_run_period(self, interval: Optional[int], is_epoch_end: bool) -> bool:
        return _is_periodic_trigger(
            run_mode=self.config.run_mode,
            interval=interval,
            global_step=self.state.global_step,
            epoch=self.state.epoch,
            is_epoch_end=is_epoch_end,
        )

    def _trigger(self, event: str, snapshot: bool = True, **kwargs) -> None:
        """Trigger an event, building context from kwargs."""
        listeners = self.listeners.get(event)
        if not listeners:
            return

        state_to_pass = copy.deepcopy(self.state) if snapshot else self.state
        context = EventContext(
            state=state_to_pass, max_epochs=self.config.max_epochs, max_steps=self.config.max_steps, **kwargs
        )

        for listener in listeners:
            listener(context)

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
        if self.train_tensor_bank:
            self.train_tensor_bank.reset()

        if self.config.distributed and hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(self.state.epoch)

        self._trigger("on_epoch_start", total_batches=len(self.train_loader), stage="train")

    def _handle_epoch_end(self):
        epoch_metrics = self.train_avg_bank.gather_average(self.config.distributed)
        if self.use_tensor_bank and self.train_tensor_bank:
            gathered_cache = self.train_tensor_bank.gather(self.config.distributed, self.device)
            task_epoch_metrics = self.task.epoch_metric(gathered_cache)
            if task_epoch_metrics:
                epoch_metrics.update(task_epoch_metrics)
        self.train_avg_bank.reset()
        if self.train_tensor_bank:
            self.train_tensor_bank.reset()

        self.state.update_current_metrics(epoch_metrics)
        self._trigger("on_epoch_end")
        self.state.epoch += 1
        self.state.batch_idx_in_epoch = 0

    def evaluate(self, model: nn.Module = None, use_ema: bool = False) -> Dict[str, Any]:
        base_model = model or self.model

        with self._ema_offload_ctx(model=base_model, use_ema=use_ema) as eval_model:
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

        eval_avg_bank = AvgBank()
        has_batch_cache = self.task.has_implemented("batch_cache")
        has_epoch_metric = self.task.has_implemented("epoch_metric")
        use_tensor_bank_for_eval = has_batch_cache and has_epoch_metric
        eval_tensor_bank = TensorBank(logger=self.logger) if use_tensor_bank_for_eval else None

        has_progress_tick = self._has_listener("on_progress_tick")
        should_calc_avg = has_progress_tick

        try:
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(data_loader):
                    batch_data = self._prepare_batch(batch_data)
                    out, batch_data = self._forward_batch(model, batch_data)

                    raw_batch_metrics = self.task.batch_metric(out, batch_data)
                    eval_avg_bank.update_from_dict(raw_batch_metrics)
                    if eval_tensor_bank:
                        eval_tensor_bank.add(self.task.batch_cache(out, batch_data))

                    scalar_batch_metrics = _get_scalar_metrics(raw_batch_metrics)
                    avg_metrics = eval_avg_bank.to_dict(self.config.distributed) if should_calc_avg else None

                    self._trigger(
                        "on_batch_end",
                        batch_idx=batch_idx,
                        total_batches=total_batches,
                        batch_metrics=raw_batch_metrics,
                        avg_bank=avg_metrics,
                        stage=stage,
                    )

                    if has_progress_tick:
                        self._trigger(
                            "on_progress_tick",
                            batch_idx=batch_idx,
                            total_batches=total_batches,
                            batch_metrics=scalar_batch_metrics,
                            avg_bank=avg_metrics,
                            stage=stage,
                        )

            avg_metrics = eval_avg_bank.gather_average(self.config.distributed)
            if use_tensor_bank_for_eval and eval_tensor_bank:
                gathered_cache = eval_tensor_bank.gather(self.config.distributed, self.device)
                task_epoch_metrics = self.task.epoch_metric(gathered_cache)
                if task_epoch_metrics:
                    avg_metrics.update(task_epoch_metrics)
            eval_avg_bank.reset()
            if eval_tensor_bank:
                eval_tensor_bank.reset()

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

        self._trigger("on_eval_start", stage="eval", total_batches=total_eval_batches, signal=signal)

        interval_metrics = self.interval_avg_bank.gather_average(self.config.distributed)
        self.interval_avg_bank = AvgBank()

        eval_results = self.evaluate_all_models()

        if interval_metrics:
            train_metric = self.task.post_metrics_to_value(interval_metrics)
            eval_results.update({f"train_{k}": v for k, v in interval_metrics.items()})
            eval_results["train_metric"] = train_metric

        self.state.update_current_metrics(eval_results)

        self._trigger("on_eval_end", stage="eval", eval_results=eval_results, signal=signal)

        is_best, previous_best = self._update_best_model_state(eval_results)
        if is_best:
            self._request_checkpoint("best", signal=signal)

        self._trigger(
            "on_validation_end",
            snapshot=False,
            stage="eval",
            eval_results=eval_results,
            previous_best=previous_best,
            is_best=is_best,
            best_model_tracker=self.best_model_tracker,
            signal=signal,
        )

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

            self._trigger(
                "on_checkpoint_request",
                snapshot=False,
                signal=target_signal,
                checkpoint_type=checkpoint_type,
            )

            checkpoint_path = target_signal.checkpoint_path
            if checkpoint_path is None:
                continue

            if checkpoint_type == "best":
                self.state.best_ckp_file = Path(checkpoint_path).name

            # Reset checkpoint_path for the next listener to use
            target_signal.checkpoint_path = None

    def _handle_periodic_events(self, is_epoch_end: bool) -> bool:
        signal = LoopSignal()

        is_eval_trigger = self._check_run_period(self.config.eval_interval, is_epoch_end)
        if is_eval_trigger:
            self.logger.debug(
                f"eval trigger: run_mode={self.config.run_mode}, global_step={self.state.global_step}, epoch={self.state.epoch}, is_epoch_end={is_epoch_end}"
                f"\n eval_interval={self.config.eval_interval} save_interval={self.config.save_interval}"
            )
            self._run_evaluation_and_update(signal)

        is_save_trigger = self._check_run_period(self.config.save_interval, is_epoch_end)
        if is_save_trigger:
            self._request_checkpoint("regular", signal=signal)

        self._flush_checkpoint_requests(signal)
        signal.synchronize_stop(self.device, self.config.distributed)

        if signal.should_stop:
            stop_message = signal.stop_message or "Early stopping triggered."
            self.logger.info(stop_message)
            self._trigger("on_early_stop", signal=signal)
            return True

        return False

    def _reached_run_limits(self) -> bool:
        max_steps_limit = self.config.max_steps if self.config.max_steps is not None else float("inf")
        max_epochs_limit = self.config.max_epochs if self.config.max_epochs is not None else float("inf")

        is_reached = self.state.global_step >= max_steps_limit or self.state.epoch >= max_epochs_limit
        if is_reached:
            self.logger.info(f"Training loop stopping at epoch {self.state.epoch}, step {self.state.global_step}.")

        return is_reached

    def _get_current_lr(self) -> Optional[float]:
        if self.optimizer is None:
            return None
        return self.optimizer.param_groups[0]["lr"]

    def _handle_post_optimizer_step(self) -> None:
        if self.scheduler is not None:
            self.scheduler.step_after_optimizer_update()

        gc_freq = self.config.gc_freq
        if gc_freq is None or gc_freq <= 0:
            return

        completed_step = self.state.global_step
        if completed_step % gc_freq != 0:
            return

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _finalize_optimizer_step(self) -> bool:
        self.state.global_step += 1

        if self.ema_model is not None:
            self.ema_model.update()

        self._handle_post_optimizer_step()
        return True

    def _is_accum_window_start(self) -> bool:
        batch_idx_zero = self.state.batch_idx_in_epoch - 1
        return batch_idx_zero % self.accum_grad == 0

    def _should_step_optimizer(self, total_batches: int) -> bool:
        batch_idx_zero = self.state.batch_idx_in_epoch - 1
        is_accum_boundary = (batch_idx_zero + 1) % self.accum_grad == 0
        is_epoch_end = self.state.batch_idx_in_epoch >= total_batches
        return is_accum_boundary or is_epoch_end

    def _apply_train_step_plain(
        self,
        loss_tensor: torch.Tensor,
        loss_count: float,
        batch_metrics: Dict[str, float],
        total_batches: int,
    ) -> bool:
        self.optimizer.zero_grad()
        loss_tensor.backward()

        if self.config.clip_grad is not None:
            clip_grad_norm_(self.model.parameters(), self.config.clip_grad)

        self.optimizer.step()
        did_optimizer_step = self._finalize_optimizer_step()

        batch_metrics["loss"] = float(loss_tensor.item())
        return did_optimizer_step

    def _apply_train_step_accum(
        self,
        loss_tensor: torch.Tensor,
        loss_count: float,
        batch_metrics: Dict[str, float],
        total_batches: int,
    ) -> bool:
        did_optimizer_step = False
        if self._is_accum_window_start():
            self.optimizer.zero_grad()
            self._current_accum_loss_count = 0.0

        self._current_accum_loss_count += loss_count
        scaled_loss = loss_tensor * loss_count
        scaled_loss.backward()

        if self._should_step_optimizer(total_batches):
            if self._current_accum_loss_count <= 0:
                raise ValueError(
                    "task.batch_loss['loss'] must provide a positive sample count during training "
                    f"(epoch={self.state.epoch + 1}, batch_idx={self.state.batch_idx_in_epoch})."
                )

            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.div_(self._current_accum_loss_count)

            if self.config.clip_grad is not None:
                clip_grad_norm_(self.model.parameters(), self.config.clip_grad)

            self.optimizer.step()
            self._current_accum_loss_count = 0.0
            did_optimizer_step = self._finalize_optimizer_step()

        batch_metrics["loss"] = float(loss_tensor.item())
        return did_optimizer_step

    def _extract_training_loss(self, losses: Dict[str, Any]) -> Tuple[torch.Tensor, float]:
        loss_entry = losses.get("loss")
        if loss_entry is None:
            raise ValueError(
                "task.batch_loss must return a 'loss' entry during training "
                f"(epoch={self.state.epoch + 1}, batch_idx={self.state.batch_idx_in_epoch})."
            )

        loss_tensor = loss_entry[0] if isinstance(loss_entry, (tuple, list)) else loss_entry
        if not isinstance(loss_tensor, torch.Tensor):
            raise ValueError(
                "task.batch_loss['loss'] must contain a torch.Tensor during training "
                f"(epoch={self.state.epoch + 1}, batch_idx={self.state.batch_idx_in_epoch})."
            )

        loss_count = loss_entry[1] if isinstance(loss_entry, (tuple, list)) and len(loss_entry) >= 2 else 1.0
        loss_count = float(qt.ensure_scala(loss_count))

        return loss_tensor, loss_count

    def _train_one_batch(self, batch_data: Any, emit_events: bool = True) -> Tuple[Dict[str, float], bool]:
        total_batches = len(self.train_loader)
        state = self.state
        batch_idx = state.batch_idx_in_epoch - 1
        batch_start_time = time.time()

        if emit_events:
            self._trigger("on_batch_start", batch_idx=batch_idx, total_batches=total_batches)

        batch_data = self._prepare_batch(batch_data)
        out, batch_data = self._forward_batch(self.model, batch_data)

        raw_batch_metrics = self.task.batch_metric(out, batch_data)
        batch_metrics = _get_scalar_metrics(raw_batch_metrics)
        if self.train_tensor_bank:
            self.train_tensor_bank.add(self.task.batch_cache(out, batch_data))

        losses = self.task.batch_loss(out, batch_data, self.loss_fn)
        loss_tensor, loss_count = self._extract_training_loss(losses)
        did_optimizer_step = self._apply_train_step_impl(loss_tensor, loss_count, batch_metrics, total_batches)

        if not emit_events:
            return batch_metrics, did_optimizer_step

        batch_time = time.time() - batch_start_time
        state.total_train_time += batch_time
        batch_metrics["batch_time"] = batch_time

        self.train_avg_bank.update_from_dict(raw_batch_metrics)
        self.interval_avg_bank.update_from_dict(raw_batch_metrics)
        if "loss" in batch_metrics:
            self.train_avg_bank.add("loss", batch_metrics["loss"], 1.0)
            self.interval_avg_bank.add("loss", batch_metrics["loss"], 1.0)
        self.train_avg_bank.add("batch_time", batch_time, 1.0)
        self.interval_avg_bank.add("batch_time", batch_time, 1.0)

        current_lr = self._get_current_lr()

        is_update_tick = state.batch_idx_in_epoch % self.config.print_freq == 0
        is_update_tick = is_update_tick or state.batch_idx_in_epoch == total_batches

        common_args = dict(
            batch_idx=batch_idx,
            total_batches=total_batches,
            batch_metrics=batch_metrics,
            lr=current_lr,
            stage="train",
        )
        self._trigger("on_batch_end", **common_args)
        self._trigger("on_train_batch_end", **common_args)

        has_progress_tick = self._has_listener("on_progress_tick")
        has_table_update = is_update_tick and self._has_listener("on_table_update")
        should_need_progress_avg = has_progress_tick or has_table_update
        if should_need_progress_avg:
            progress_avg_metrics = self.train_avg_bank.to_dict(self.config.distributed)
            common_progress_args = dict(
                batch_idx=batch_idx,
                total_batches=total_batches,
                batch_metrics=batch_metrics,
                avg_bank=progress_avg_metrics,
                lr=current_lr,
                stage="train",
            )
            if has_progress_tick:
                self._trigger("on_progress_tick", **common_progress_args)
            if has_table_update:
                self._trigger("on_table_update", **common_progress_args)

        return batch_metrics, did_optimizer_step

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
                    _, did_optimizer_step = self._train_one_batch(batch_data, emit_events=True)

                    is_epoch_end = self.state.batch_idx_in_epoch >= total_batches
                    should_handle_periodic = False
                    if self.config.run_mode == RunMode.EPOCH:
                        should_handle_periodic = is_epoch_end
                    else:
                        should_handle_periodic = did_optimizer_step

                    if should_handle_periodic:
                        should_stop_mid_epoch = self._handle_periodic_events(is_epoch_end=is_epoch_end)
                        if should_stop_mid_epoch:
                            return True

                    if did_optimizer_step:
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
            f"accum_grad={self.config.accum_grad}, "
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


