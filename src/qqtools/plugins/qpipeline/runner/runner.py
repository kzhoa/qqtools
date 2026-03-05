"""
Unified Training Runner

Config
State
Agent
Special Features
"""

import copy
import gc
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader

import qqtools as qt

from ..entry_utils.qema import qEMA
from ..entry_utils.scheduler import qWarmupScheduler
from ..entry_utils.type_qconfig import CheckpointConfig, EarlyStopConfig, qConfig
from ..qlogger import ConsoleLogger, qLogger
from ..task.qtask import qTaskBase
from .avgbank import AvgBank
from .best_model import BestModelTracker
from .ckp_manager import CheckpointManager
from .earlystop import EarlyStopper
from .eval_summary_formatter import EvalSummaryFormatter
from .progress import ProgressTracker
from .tensorbank import TensorBank
from .types import EventContext, RunConfig, RunMode, RunningState

__all__ = ["train_runner", "infer_runner"]


class SheetLoggerListener:
    """
    A listener that writes metrics to a sheet logger (e.g., CSV) based on training events.
    It has distinct methods for different event types, allowing flexible attachment.
    """

    def __init__(self, logger: qLogger, run_config: RunConfig, log_granularity: List[str]):
        """
        Args:
            logger: An instance of qLogger.
            run_config: The overall run configuration to check for eval intervals.
            log_granularity: A list containing "eval" and/or "batch".
        """
        self.logger = logger
        self.config = run_config
        self.log_granularity = log_granularity

    def _prepare_data(self, context: EventContext, mode: str) -> Dict[str, Any]:
        """Prepares a flat dictionary of metrics for logging. Robust to missing metrics."""
        state = context.state
        data = {"epoch": state.epoch, "global_step": state.global_step}

        source_metrics = {}
        if mode == "eval":
            # Prefer context.eval_results, then fallback to a few common state metrics.
            if getattr(context, "eval_results", None):
                source_metrics.update(context.eval_results)
            else:
                # Backward-compatible fallback.
                for k in ["current_val_metric", "current_test_metric", "current_train_metric", "current_train_loss"]:
                    v = getattr(state, k, None)
                    if v is not None:
                        key = k.replace("current_", "")
                        source_metrics[key] = v
                if not source_metrics:
                    self.logger.warning("No eval metrics found in context or state; writing empty row.")
        elif mode == "batch":
            if getattr(context, "batch_metrics", None):
                source_metrics.update(context.batch_metrics)
            else:
                self.logger.warning("No batch_metrics found in context; writing empty row.")

        # Ensure all logger columns are present, fill with None if missing
        if self.logger.columns:
            for key in self.logger.columns:
                if key in data:
                    continue
                value = source_metrics.get(key)
                # Fallback for batch-level train metrics (e.g., train_loss -> loss)
                if value is None and mode == "batch" and key.startswith("train_"):
                    fallback_key = key[len("train_") :]
                    value = source_metrics.get(fallback_key)
                data[key] = value
        else:
            data.update(source_metrics)
        return data

    def on_eval_end(self, context: EventContext):
        """Callback for the end of an evaluation phase."""
        data = self._prepare_data(context, mode="eval")
        self.logger.write(data)

    def on_train_batch_end(self, context: EventContext):
        """Callback for the end of a training batch."""
        # If eval-level logging is also active, suppress batch logging on eval steps
        # to avoid duplicate entries for the same global_step.
        if "eval" in self.log_granularity:
            is_epoch_end = (
                context.batch_idx is not None
                and context.total_batches is not None
                and context.batch_idx == context.total_batches - 1
            )
            is_eval_trigger = _is_periodic_trigger(
                run_mode=self.config.run_mode,
                interval=self.config.eval_interval,
                global_step=context.state.global_step,
                epoch=context.state.epoch,
                is_epoch_end=is_epoch_end,
            )
            if is_eval_trigger:
                return  # Suppress write, on_eval_end will handle it

        data = self._prepare_data(context, mode="batch")
        self.logger.write(data)


# ============================================================================
# Pure Functions for Light-weight Agent
# ============================================================================
def move_batch_to_device(batch_data, device: torch.device):
    """Move batch data to device, handling various data structures.

    Args:
        batch_data: Batch data (Tensor, dict, tuple, or custom object)
        device: Target device

    Returns:
        Batch data moved to device
    """
    if batch_data is None:
        return batch_data

    if hasattr(batch_data, "to"):
        return batch_data.to(device)
    elif isinstance(batch_data, dict):
        return qt.qDict({k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_data.items()})
    elif isinstance(batch_data, (tuple, list)):
        return type(batch_data)(v.to(device) if isinstance(v, torch.Tensor) else v for v in batch_data)
    else:
        return batch_data


def _getattr_or_default(obj: Any, key: str, default: Any = None) -> Any:
    """Get attribute value and fallback when missing or None.

    If ``default`` is callable (e.g., ``dict``), it will be called lazily.
    """
    value = getattr(obj, key, None)
    if value is not None:
        return value
    return default() if callable(default) else default


def _is_periodic_trigger(
    run_mode: RunMode,
    interval: Optional[int],
    global_step: int,
    epoch: int,
    is_epoch_end: bool,
) -> bool:
    """Check if a periodic event should fire for current train state."""
    if interval is None or interval <= 0:
        return False

    if run_mode == RunMode.STEP:
        # global_step is incremented after periodic checks, so use completed-step semantics.
        return (global_step + 1) % interval == 0

    return is_epoch_end and (epoch + 1) % interval == 0


def _resolve_train_runner_policy(
    run_mode: Union[str, RunMode],
    max_epochs: Optional[int],
    max_steps: Optional[int],
    eval_interval: Optional[int],
    save_interval: Optional[int],
) -> Tuple[RunMode, int, int, Optional[int], Optional[int], List[str]]:
    """Resolve train-runner-owned policy fields into effective runtime values."""

    def _ensure_positive_int(value: Any, field_name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{field_name} must be a positive integer (>=1)")
        return value

    if run_mode is None:
        raise ValueError("run_mode cannot be None. Supported values are 'epoch' and 'step'.")

    resolved_run_mode = RunMode(run_mode)

    effective_eval_interval = 1 if eval_interval is None else eval_interval
    effective_eval_interval = _ensure_positive_int(effective_eval_interval, "eval_interval")

    effective_save_interval = effective_eval_interval if save_interval is None else save_interval
    effective_save_interval = _ensure_positive_int(effective_save_interval, "save_interval")

    effective_max_epochs = max_epochs
    effective_max_steps = max_steps
    policy_warnings: List[str] = []

    if resolved_run_mode == RunMode.EPOCH:
        if max_epochs is None:
            raise ValueError("max_epochs must be specified when run_mode='epoch'.")
        effective_max_epochs = _ensure_positive_int(max_epochs, "max_epochs")
        if max_steps is not None:
            policy_warnings.append(
                f"[run_mode=EPOCH] max_steps={max_steps} is ignored by mutual-exclusion policy; "
                f"training will be controlled by max_epochs={max_epochs}."
            )
        effective_max_steps = None
    else:  # RunMode.STEP
        if max_steps is None:
            raise ValueError("max_steps must be specified when run_mode='step'.")
        effective_max_steps = _ensure_positive_int(max_steps, "max_steps")
        if max_epochs is not None:
            policy_warnings.append(
                f"[run_mode=STEP] max_epochs={max_epochs} is ignored by mutual-exclusion policy; "
                f"training will be controlled by max_steps={max_steps}."
            )
        effective_max_epochs = None

    return (
        resolved_run_mode,
        effective_eval_interval,
        effective_save_interval,
        effective_max_epochs,
        effective_max_steps,
        policy_warnings,
    )

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
        checkpoint_manager: Optional[CheckpointManager] = None,
        early_stopper: Optional[EarlyStopper] = None,
        best_model_tracker: Optional[BestModelTracker] = None,
        best_model_manager: Optional[BestModelTracker] = None,
        listeners: Optional[Dict[str, List[Callable]]] = None,
        state: Optional[RunningState] = None,
        avg_bank: Optional[AvgBank] = None,
    ):
        """
        Training Agent with dependency injection support for better testability and extensibility.
        Args:
            model: Model instance
            task: Task instance
            loss_fn: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            config: Run configuration
            device: Device
            ema_model: EMA model
            logger: Logger instance
            checkpoint_manager: Checkpoint manager (optional, None to skip checkpoint functionality)
            early_stopper: Early stopping manager (optional, None to disable early stopping)
            best_model_tracker: Best model tracker (optional, None to skip best model tracking)
            best_model_manager: Deprecated alias of best_model_tracker
            listeners: Event listeners dictionary
            state: Running state
            avg_bank: Average metrics manager
        """
        self.model = model
        self.task = task
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Use the provided config object directly, or create a default one if none is given.
        # This prevents overwriting specific settings like `save_interval=None` with defaults.
        self.config = config if config is not None else RunConfig()

        self.device = device or self.config.device
        self.ema_model = ema_model
        self.logger = logger or ConsoleLogger()

        # initialization
        self.state = state or RunningState()

        self.train_loader = task.train_loader
        self.val_loader = task.val_loader
        self.test_loader = task.test_loader

        # Managers (Add-ons) - must be injected externally
        self.checkpoint_manager = checkpoint_manager
        self.early_stopper = early_stopper
        self.best_model_tracker = best_model_tracker or best_model_manager
        # Backward-compatibility alias.
        self.best_model_manager = self.best_model_tracker

        # Listeners for extensibility
        default_listeners = {
            "on_epoch_start": [],
            "on_epoch_end": [],
            "on_batch_start": [],
            "on_batch_end": [],
            "on_progress_tick": [],
            "on_train_batch_end": [],
            "on_table_update": [],
            "on_eval_start": [],
            "on_eval_end": [],
            "on_checkpoint_save": [],
            "on_early_stop": [],
        }
        self.listeners = listeners or default_listeners

        # Internal variables
        self._train_iter = None
        self.avg_bank = avg_bank or AvgBank()
        self.interval_avg_bank = AvgBank()

        # Check TensorBank capability hooks
        has_batch_cache = self.task.has_implemented("batch_cache")
        has_epoch_metric = self.task.has_implemented("epoch_metric")

        self.use_tensor_bank = has_batch_cache and has_epoch_metric
        self.tensor_bank = TensorBank(logger=self.logger) if self.use_tensor_bank else None
        if has_batch_cache and not has_epoch_metric:
            self.logger.warning(
                "Task implements 'batch_cache' but missing companion 'epoch_metric'. "
                "Tensor caching feature is disabled. Please implement both hooks to enable."
            )
        elif has_epoch_metric and not has_batch_cache:
            self.logger.warning(
                "Task implements 'epoch_metric' but missing companion 'batch_cache'. "
                "Tensor caching feature is disabled. Please implement both hooks to enable."
            )

        # Check for model offloading (EMA on GPU, Main Model on CPU during eval)
        self._use_model_offload = False
        if self.device.type == "cuda" and self.ema_model is not None:
            from ..entry_utils.info import get_model_size_bytes

            m_size = get_model_size_bytes(self.model)
            gpu_cap = torch.cuda.get_device_properties(self.device).total_memory
            if m_size > (gpu_cap * 0.5):
                self._use_model_offload = True
                self.logger.info(
                    f"Model size ({m_size / 1024**2:.2f} MB) > 50% of GPU capacity. "
                    "Enabling mutual offloading during evaluation."
                )

    def add_listener(self, event: str, listener: Callable):
        if event in self.listeners:
            self.listeners[event].append(listener)
        else:
            warnings.warn(f"Unknown event: {event}")

    def _trigger(self, event: str, context: EventContext):
        # Create a snapshot of the state for this event to prevent listeners
        # from getting a mutable state object that changes later.
        context_snapshot = copy.copy(context)
        context_snapshot.state = copy.copy(context.state)
        # Inject run limits into the context so listeners can access them
        try:
            context_snapshot.max_epochs = self.config.max_epochs
            context_snapshot.max_steps = self.config.max_steps
        except Exception:
            pass

        for listener in self.listeners.get(event, []):
            try:
                listener(context_snapshot)
            except Exception as e:
                self.logger.error(f"Listener error for event '{event}' in {listener.__name__}: {type(e).__name__}: {e}")
                raise

    def _prepare_batch(self, batch_data):
        """Prepare batch data: move to device and convert dtype if needed.

        Args:
            batch_data: Raw batch data

        Returns:
            Prepared batch data
        """
        batch_data = move_batch_to_device(batch_data, self.device)
        # if hasattr(batch_data, "to_dtype") and self.config.dtype is not None:
        #     batch_data = batch_data.to_dtype(self.config.dtype)
        return batch_data

    def _forward_batch(self, model: nn.Module, batch_data):
        """Forward pass through the model.

        Args:
            model: Model to use for forward pass
            batch_data: Prepared batch data

        Returns:
            Model output after post-processing
        """
        batch_data = self.task.pre_batch_forward(batch_data)
        out = self.task.batch_forward(model, batch_data)
        out = self.task.post_batch_forward(out, batch_data)
        return out, batch_data

    @staticmethod
    def _scalarize_batch_metrics(batch_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Convert batch metric payload into scalar-only dict for logging/progress.

        Supports both:
        - {name: scalar_or_tensor}
        - {name: (value, count)}
        """
        scalar_metrics: Dict[str, float] = {}

        for key, value in batch_metrics.items():
            metric_value = value
            if isinstance(value, tuple) and len(value) > 0:
                metric_value = value[0]

            if torch.is_tensor(metric_value):
                if metric_value.numel() != 1:
                    continue
                metric_value = metric_value.item()

            if isinstance(metric_value, (int, float)):
                scalar_metrics[key] = float(metric_value)

        return scalar_metrics

    def _emit_batch_events(
        self,
        *,
        batch_idx: int,
        total_batches: int,
        batch_metrics: Dict[str, Any],
        scalar_batch_metrics: Dict[str, float],
        stage: str,
        lr: Optional[float] = None,
        batch_avg_metrics: Optional[Dict[str, Any]] = None,
        progress_avg_metrics: Optional[Dict[str, Any]] = None,
        emit_train_batch_end: bool = False,
        emit_table_update: bool = False,
    ) -> None:
        """Emit batch-related events using shared context assembly."""
        batch_context = EventContext(
            state=self.state,
            batch_idx=batch_idx,
            total_batches=total_batches,
            batch_metrics=batch_metrics,
            avg_bank=batch_avg_metrics,
            lr=lr,
            stage=stage,
        )
        self._trigger("on_batch_end", context=batch_context)
        if emit_train_batch_end:
            self._trigger("on_train_batch_end", context=batch_context)

        progress_context = EventContext(
            state=self.state,
            batch_idx=batch_idx,
            total_batches=total_batches,
            batch_metrics=scalar_batch_metrics,
            avg_bank=progress_avg_metrics,
            lr=lr,
            stage=stage,
        )
        self._trigger("on_progress_tick", context=progress_context)
        if emit_table_update:
            self._trigger("on_table_update", context=progress_context)

    def _start_new_epoch(self):
        """Initialize a new epoch: create iterator and setup distributed sampling."""
        # Set iterator and reset metrics
        self._train_iter = iter(self.train_loader)
        if self.use_tensor_bank:
            self.tensor_bank.reset()

        # Update distributed sampler epoch for proper shuffling
        if self.config.distributed and hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(self.state.epoch)

        # Trigger epoch start event
        self._trigger(
            "on_epoch_start",
            context=EventContext(state=self.state, total_batches=len(self.train_loader), stage="train"),
        )

    def _get_next_batch(self) -> Tuple[Any, bool]:
        """
        Gets the next batch from the training data loader.

        Returns:
            A tuple containing:
            - The batch data (or None if epoch ends).
            - A boolean flag that is True if the current batch marks the end of an epoch.
        """
        is_epoch_end = False

        if self._train_iter is None:
            self._start_new_epoch()

            # If resuming from a checkpoint at mid-epoch, skip already-processed batches
            for _ in range(self.state.batch_idx_in_epoch):
                try:
                    next(self._train_iter)
                except StopIteration:
                    # If the entire epoch was skipped, reset batch_idx_in_epoch and break.
                    # _training_loop will detect the exhausted iterator and handle epoch end.
                    self.state.batch_idx_in_epoch = 0
                    break

        try:
            batch_data = next(self._train_iter)
            self.state.batch_idx_in_epoch += 1
            if self.state.batch_idx_in_epoch == len(self.train_loader):
                is_epoch_end = True
        except StopIteration:
            # Data loader exhausted for this epoch
            is_epoch_end = True
            batch_data = None  # Signal to _training_loop that epoch has ended

        return batch_data, is_epoch_end

    def _handle_epoch_end(self):
        """
        Handles tasks at the end of an epoch:
        1. Calculate and update average metrics
        2. Trigger epoch end event for listeners
        3. Reset metrics accumulator and increment epoch counter
        """
        # Calculate the average metrics of the current epoch
        epoch_metrics = self.avg_bank.gather_average(self.config.distributed)

        # Add tasks's cross-batch metrics if implemented
        if self.use_tensor_bank:
            gathered_cache = self.tensor_bank.gather(self.config.distributed, self.device)
            task_epoch_metrics = self.task.epoch_metric(gathered_cache)
            if task_epoch_metrics:
                epoch_metrics.update(task_epoch_metrics)

        self.state.update_current_metrics(epoch_metrics)

        # Trigger epoch end event for listeners
        self._trigger("on_epoch_end", context=EventContext(state=self.state))

        # Increment epoch counter and reset metrics for next epoch
        self.state.epoch += 1
        # Reset batch index at epoch boundary for mid-epoch checkpoint recovery
        self.state.batch_idx_in_epoch = 0
        self.avg_bank = AvgBank()  # Reset for next epoch

    def train_batch(self, batch_data) -> Dict[str, Any]:
        """Train a single batch"""
        # Prepare batch
        batch_data = self._prepare_batch(batch_data)

        # Forward pass
        out, batch_data = self._forward_batch(self.model, batch_data)

        # Calculate metrics
        raw_batch_metrics = self.task.batch_metric(out, batch_data)
        batch_metrics = self._scalarize_batch_metrics(raw_batch_metrics)

        # Cache tensors if implemented
        if self.use_tensor_bank:
            self.tensor_bank.add(self.task.batch_cache(out, batch_data))

        # Training mode: calculate loss and backward propagation
        losses = self.task.batch_loss(out, batch_data, self.loss_fn)
        loss_tensor, _loss_cnt = losses.get("loss", (None, 1))

        if loss_tensor is not None:
            self.optimizer.zero_grad()
            loss_tensor.backward()

            if self.config.clip_grad is not None:
                clip_grad_norm_(self.model.parameters(), self.config.clip_grad)

            self.optimizer.step()

            # EMA update
            if self.ema_model is not None:
                self.ema_model.update()

            batch_metrics["loss"] = loss_tensor.item()

        return batch_metrics

    def evaluate(self, model: nn.Module = None, use_ema: bool = False) -> Dict[str, Any]:
        """Evaluate the model.

        If use_ema is True and offloading is enabled, the main model will be moved to CPU
        to free up GPU memory for the EMA model.
        """
        eval_model = model or self.model
        ema_original_device = None
        offloaded = False

        if use_ema and self.ema_model is not None:
            eval_model = self.ema_model

            # Offload main model if needed to free GPU memory
            if self._use_model_offload:
                self.logger.debug("Offloading main model to 'cpu' for EMA evaluation.")
                self.model.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                offloaded = True

            # Temporarily move EMA model to target device for evaluation speed
            try:
                # Get current device of EMA model
                ema_original_device = next(eval_model.parameters()).device
                if ema_original_device != self.device:
                    self.logger.debug(
                        f"Moving EMA model from '{ema_original_device}' to '{self.device}' for evaluation."
                    )
                    eval_model.to(self.device)
            except (StopIteration, Exception):
                ema_original_device = None

        results = {}

        try:
            # Validation set
            if self.val_loader is not None:
                val_metrics = self._run_evaluation(eval_model, self.val_loader, "val", stage="val")
                results.update(val_metrics)

            # Test set
            if self.test_loader is not None:
                test_metrics = self._run_evaluation(eval_model, self.test_loader, "test", stage="test")
                results.update(test_metrics)
        finally:
            # Restore EMA model to its original device
            if ema_original_device is not None and ema_original_device != self.device:
                self.logger.debug(f"Moving EMA model back to '{ema_original_device}' after evaluation.")
                eval_model.to(ema_original_device)

            # Restore main model to GPU if it was offloaded
            if offloaded:
                self.logger.debug(f"Restoring main model to '{self.device}' after evaluation.")
                self.model.to(self.device)

        return results

    def _run_evaluation(
        self, model: nn.Module, data_loader: DataLoader, prefix: str = "", stage: str = "val"
    ) -> Dict[str, Any]:
        """Run evaluation"""
        was_training = model.training
        model.eval()
        avg_bank = AvgBank()
        tensor_bank = TensorBank(self.logger) if self.use_tensor_bank else None

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                # Prepare batch
                batch_data = self._prepare_batch(batch_data)

                # Forward pass
                out, batch_data = self._forward_batch(model, batch_data)

                # Calculate metrics
                batch_metrics = self.task.batch_metric(out, batch_data)

                # Accumulate metrics into avg_bank
                for key, (value, cnt) in batch_metrics.items():
                    if torch.is_tensor(value):
                        value = value.item()
                    avg_bank.add(key, value, cnt)

                # Cache tensors into tensor_bank if implemented
                if self.use_tensor_bank:
                    tensor_bank.add(self.task.batch_cache(out, batch_data))

                avg_metrics = avg_bank.to_dict(self.config.distributed)
                self._emit_batch_events(
                    batch_idx=batch_idx,
                    total_batches=len(data_loader),
                    batch_metrics=batch_metrics,
                    scalar_batch_metrics=self._scalarize_batch_metrics(batch_metrics),
                    stage=stage,
                    batch_avg_metrics=avg_metrics,
                    progress_avg_metrics=avg_metrics,
                )

        # Calculate average
        avg_metrics = avg_bank.gather_average(self.config.distributed)

        # Add tasks's cross-batch metrics if implemented
        if self.use_tensor_bank:
            gathered_cache = tensor_bank.gather(self.config.distributed, self.device)
            task_epoch_metrics = self.task.epoch_metric(gathered_cache)
            if task_epoch_metrics:
                avg_metrics.update(task_epoch_metrics)

        # Add prefix
        prefixed_metrics = {f"{prefix}_{key}": value for key, value in avg_metrics.items()}

        # Convert to task metrics
        task_metric = self.task.post_metrics_to_value(avg_metrics)
        prefixed_metrics[f"{prefix}_metric"] = task_metric

        model.train(was_training)
        return prefixed_metrics

    def _run_evaluation_and_update(self) -> bool:
        """
        Runs the full evaluation process and updates the best model/early stopping state.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        total_eval_batches = 0
        for loader in (self.val_loader, self.test_loader):
            if loader is None:
                continue
            try:
                total_eval_batches += len(loader)
            except TypeError:
                pass

        self._trigger(
            "on_eval_start",
            context=EventContext(state=self.state, stage="eval", total_batches=total_eval_batches),
        )

        # Gather and reset interval train metrics
        interval_metrics = self.interval_avg_bank.gather_average(self.config.distributed)
        self.interval_avg_bank = AvgBank()

        # Evaluate the original model
        eval_results = self.evaluate(self.model, use_ema=False)

        # Add train metrics to results
        if interval_metrics:
            train_metric = self.task.post_metrics_to_value(interval_metrics)
            eval_results.update({f"train_{k}": v for k, v in interval_metrics.items()})
            eval_results["train_metric"] = train_metric

        # Evaluate EMA model (if exists)
        if self.ema_model is not None:
            ema_results = self.evaluate(self.model, use_ema=True)
            # Merge results and add ema prefix
            for key, value in ema_results.items():
                eval_results[f"ema_{key}"] = value

        # Update state
        self.state.update_current_metrics(eval_results)

        # Trigger evaluation end event
        self._trigger(
            "on_eval_end",
            context=EventContext(state=self.state, stage="eval", eval_results=eval_results),
        )

        # Learning rate update (epoch-wise scheduler)
        if self.scheduler is not None:
            self.scheduler.step_main(metrics=eval_results.get("val_metric"))

        # Determine if best
        target = self.config.checkpoint.get("target", "val_metric")
        current_val = eval_results.get(target)

        is_best = False
        previous_best = None
        if self.best_model_tracker is not None:
            previous_best = {
                "metric": getattr(self.best_model_tracker, "best_metric", None),
                "epoch": getattr(self.best_model_tracker, "best_epoch", 0),
                "step": getattr(self.best_model_tracker, "best_step", 0),
            }
        if current_val is not None and self.best_model_tracker is not None:
            is_best = self.best_model_tracker.update(current_val, self.state.epoch, self.state.global_step)

        if is_best:
            self._save_best_checkpoint(
                monitored_key=target,
                monitored_metric=current_val,
                eval_results=eval_results,
            )

        # Early stop check
        if self.early_stopper is not None:
            early_stop_target = self.config.early_stop.get("target", "val_metric")

            metrics_for_early_stop = {}
            current_early_stop_metric = eval_results.get(early_stop_target)
            if current_early_stop_metric is not None:
                metrics_for_early_stop[early_stop_target] = current_early_stop_metric

            should_stop, stop_msg, debug_msg = self.early_stopper(metrics_for_early_stop)

            if debug_msg is not None:
                self.logger.info(debug_msg)

        # Log formatted summary
        self._log_eval_summary(eval_results, is_best, previous_best=previous_best)

        if self.early_stopper is not None and should_stop:
            self.logger.info(stop_msg)
            self._trigger("on_early_stop", context=EventContext(state=self.state))
            return True

        return False

    def _log_eval_summary(
        self,
        eval_results: Dict[str, Any],
        is_best: bool,
        previous_best: Optional[Dict[str, Any]] = None,
    ):
        """Log evaluation summary in both hierarchical and table-friendly formats."""
        target_key = self.config.checkpoint.get("target", "val_metric")
        target_mode = self.config.checkpoint.get("mode", "min")
        summary_lines, summary_has_markup, table_lines, table_has_markup = EvalSummaryFormatter.format_all(
            eval_results=eval_results,
            epoch=self.state.epoch,
            step=self.state.global_step,
            target_key=target_key,
            target_mode=target_mode,
            is_best=is_best,
            previous_best=previous_best,
            best_model_tracker=self.best_model_tracker,
            color_new_best=True,
        )

        if summary_has_markup:
            self.logger.info("\n".join(summary_lines), extra={"markup": True})
        else:
            self.logger.info("\n".join(summary_lines))

        if table_has_markup:
            self.logger.info("\n".join(table_lines), extra={"markup": True})
        else:
            self.logger.info("\n".join(table_lines))

    def _save_best_checkpoint(self, monitored_key: str, monitored_metric: Any, eval_results: Dict[str, Any]):
        """Saves the best model checkpoint and updates best metrics."""
        # Update best metrics
        self.state.best_epoch = self.state.epoch
        self.state.best_step = self.state.global_step
        self.state.best_monitored_key = monitored_key

        if monitored_metric is not None:
            monitored_metric = qt.ensure_scala(monitored_metric)
        self.state.best_monitored_metric = monitored_metric
        self.state.best_model_metrics_snapshot = copy.deepcopy(eval_results)

        self._save_checkpoint(checkpoint_type="best")

    def _save_regular_checkpoint(self):
        """Saves a checkpoint of the current state, not contingent on performance."""
        self._save_checkpoint(checkpoint_type="regular")

    def _save_checkpoint(self, checkpoint_type: Literal["best", "regular"]) -> Optional[str]:
        """Save a checkpoint and emit checkpoint event."""
        if self.checkpoint_manager is None:
            return None

        ckp_path = self.checkpoint_manager.save(
            copy.copy(self.state),
            self.model,
            self.task,
            self.optimizer,
            self.scheduler,
            self.ema_model,
            self.early_stopper,
            self.best_model_tracker,
            is_best=(checkpoint_type == "best"),
        )
        self._trigger(
            "on_checkpoint_save",
            context=EventContext(state=self.state, checkpoint_type=checkpoint_type, checkpoint_path=ckp_path),
        )
        return ckp_path

    def _handle_periodic_events(self, is_epoch_end: bool) -> bool:
        """
        Handles all periodic events like evaluation, checkpointing, and early stopping.
        This is the central place for managing the training lifecycle based on run_mode.

        Args:
            is_epoch_end: Flag indicating if the current step is the last in an epoch.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        is_eval_trigger = _is_periodic_trigger(
            run_mode=self.config.run_mode,
            interval=self.config.eval_interval,
            global_step=self.state.global_step,
            epoch=self.state.epoch,
            is_epoch_end=is_epoch_end,
        )

        should_stop = False
        if is_eval_trigger:
            self.logger.debug(
                f"eval trigger: run_mode={self.config.run_mode}, global_step={self.state.global_step}, epoch={self.state.epoch}, is_epoch_end={is_epoch_end}"
                f"\n eval_interval={self.config.eval_interval} save_interval={self.config.save_interval}"
            )
            should_stop = self._run_evaluation_and_update()

        is_save_trigger = _is_periodic_trigger(
            run_mode=self.config.run_mode,
            interval=self.config.save_interval,
            global_step=self.state.global_step,
            epoch=self.state.epoch,
            is_epoch_end=is_epoch_end,
        )
        if is_save_trigger:
            self._save_regular_checkpoint()

        return should_stop

    def _training_loop(self) -> bool:
        """
        The main training loop, abstracted from the `run` method.

        This loop iterates through the training data, manages state, and calls
        periodic event handlers. It is designed to be safe against infinite loops
        and to handle interruptions gracefully.

        Returns:
            bool: True if training was stopped early, False otherwise.
        """
        # Initialize training iterator
        self._train_iter = None

        while True:
            # Check for overall stopping conditions at the beginning of each potential epoch/step
            max_steps_limit = self.config.max_steps if self.config.max_steps is not None else float("inf")
            max_epochs_limit = self.config.max_epochs if self.config.max_epochs is not None else float("inf")
            if self.state.global_step >= max_steps_limit or self.state.epoch >= max_epochs_limit:
                self.logger.info(f"Training loop stopping at epoch {self.state.epoch}, step {self.state.global_step}.")
                return False  # Completed normally

            # This marks the start of an actual training iteration for a batch.
            # Evaluation/Checkpointing for the *previous* epoch happens just before epoch increment.
            # For step-mode, evaluation can happen mid-epoch.

            try:
                batch_data, is_epoch_end = self._get_next_batch()

                if batch_data is None:  # Epoch has truly ended, handle epoch-level events
                    # The event handling (eval, save) is triggered after the last batch is processed.
                    # This block now only handles the state update for the next epoch.
                    self._handle_epoch_end()  # Increment epoch, reset avg_bank, etc.
                    self._train_iter = None  # Force new iterator creation for next epoch
                    continue  # Start next iteration of while loop to check new state boundaries

                # Train a single batch
                batch_start_time = time.time()
                self._trigger(
                    "on_batch_start",
                    context=EventContext(
                        state=self.state,
                        batch_idx=self.state.batch_idx_in_epoch - 1,  # Use 0-indexed batch_idx
                        total_batches=len(self.train_loader),
                    ),
                )

                batch_metrics = self.train_batch(batch_data)

                # Unconditionally step warmup scheduler every batch
                if self.scheduler is not None:
                    self.scheduler.step_warmup()

                # Record batch time and update total training time
                batch_time = time.time() - batch_start_time
                self.state.total_train_time += batch_time
                batch_metrics["batch_time"] = batch_time

                # Update average banks
                for k, v in batch_metrics.items():
                    if isinstance(v, (int, float)):
                        self.avg_bank.add(k, v)
                        self.interval_avg_bank.add(k, v)

                # Trigger batch end event for listeners (e.g., progress bars, loggers)
                # Trigger table update based on print frequency
                freq = self.config.print_freq
                is_update_tick = self.state.batch_idx_in_epoch % freq == 0 or self.state.batch_idx_in_epoch == len(
                    self.train_loader
                )
                current_lr = self.optimizer.param_groups[0]["lr"] if self.optimizer else None
                self._emit_batch_events(
                    batch_idx=self.state.batch_idx_in_epoch - 1,
                    total_batches=len(self.train_loader),
                    batch_metrics=batch_metrics,
                    scalar_batch_metrics=self._scalarize_batch_metrics(batch_metrics),
                    stage="train",
                    lr=current_lr,
                    progress_avg_metrics=self.avg_bank.to_dict(self.config.distributed),
                    emit_train_batch_end=True,
                    emit_table_update=is_update_tick,
                )

                # Check periodic events that are tied to global_step, or epoch-end if not yet handled
                should_stop_mid_epoch = self._handle_periodic_events(is_epoch_end=is_epoch_end)
                if should_stop_mid_epoch:
                    return True  # Early stopping was triggered

                # Increment global step *after* all step-based logic is complete
                self.state.global_step += 1

                # Periodically clean up memory
                if self.state.global_step % self.config.gc_freq == 0:  # Using config.gc_freq
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except KeyboardInterrupt:
                self.logger.info("Training interrupted by user.")
                return True  # Treat as early stop
            except StopIteration:
                # This should ideally be handled by `batch_data is None` check from _get_next_batch
                # but as a defensive measure, if StopIteration happens here, it means the loader
                # was exhausted in a way not caught by `_get_next_batch` or it was the very last batch.
                self.logger.warning(
                    "StopIteration caught in _training_loop. This might be unexpected and could prematurely trigger "
                    "end-of-epoch events like evaluation. Forcing is_epoch_end=True."
                )
                should_stop_after_eval = self._handle_periodic_events(is_epoch_end=True)
                if should_stop_after_eval:
                    return True
                self._handle_epoch_end()
                self._train_iter = None
                continue  # Re-enter loop to check state boundaries

    def run(self) -> bool:
        """
        Runs the training process by setting up and invoking the main training loop.

        Returns:
            bool: True if training was stopped early, False otherwise.
        """
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

        # Final log message
        if not early_stopped:
            if self.state.global_step >= max_steps_limit:
                self.logger.info(f"Reached max_steps={max_steps_limit}")
            elif self.state.epoch >= max_epochs_limit:
                self.logger.info(f"Reached max_epochs={max_epochs_limit}")

        return early_stopped


# Design Rationale: Boundary Policy Ownership
# The orchestration layer (train_runner) owns the business policy for mutually
# exclusive run boundaries:
# - EPOCH mode keeps only max_epochs
# - STEP mode keeps only max_steps
#
# RunningAgent remains policy-agnostic and simply stops based on the concrete
# boundaries passed in via RunConfig.
def train_runner(
    model: nn.Module,
    task: qTaskBase,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[qWarmupScheduler] = None,
    args: Optional[qConfig] = None,
    max_epochs: Optional[int] = None,
    max_steps: Optional[int] = None,
    clip_grad: Optional[float] = None,
    distributed: bool = False,
    save_dir: str = "./logs",
    print_freq: int = 10,
    extra_log_keys: Optional[List[str]] = None,
    extra_ckp_caches: Optional[Dict[str, Any]] = None,
    use_profiler: bool = False,
    ema_model: Optional[qEMA] = None,
    run_mode: Union[str, RunMode] = "epoch",
    eval_interval: int = 1,
    save_interval: Optional[int] = None,
    log_granularity: Optional[List[Literal["eval", "batch"]]] = ["eval"],
) -> Dict[str, Any]:
    """
    Unified training runner

    Args:
        model: Model
        task: Task instance
        loss_fn: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        args: Object containing command-line arguments and other configurations.
              It is the single source for settings like device, rank, checkpoint, etc.
        max_epochs: Maximum number of epochs
        max_steps: Maximum number of steps
        clip_grad: Gradient clipping
        distributed: Whether to use distributed training
        save_dir: Directory to save checkpoints
        print_freq: Frequency of printing logs
        extra_log_keys: Extra log keys
        extra_ckp_caches: Extra checkpoint caches
        use_profiler: Whether to use profiler
        ema_model: EMA model
        run_mode: Running mode ("epoch" or "step")
        eval_interval: Evaluation interval (interpreted as epochs or steps depending on run_mode)
        save_interval: Regular checkpoint saving interval (interpreted as epochs or steps depending on run_mode)

    Returns:
        Dictionary with training results
    """
    # Handle compatibility parameters
    if args is None:
        raise ValueError("The 'args' parameter is required to configure the runner.")

    # Extract configuration from args
    device = _getattr_or_default(args, "device", lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    rank = _getattr_or_default(args, "rank", 0)
    runner_config = _getattr_or_default(args, "runner")
    if runner_config is None:
        raise AttributeError("args.runner is required")

    checkpoint_config: CheckpointConfig = _getattr_or_default(runner_config, "checkpoint", dict)
    early_stop_config: EarlyStopConfig = _getattr_or_default(runner_config, "early_stop", dict)
    ckp_file = _getattr_or_default(args, "ckp_file")
    init_file = _getattr_or_default(args, "init_file")
    render_type = _getattr_or_default(args, "render_type", "auto")

    (
        resolved_run_mode,
        effective_eval_interval,
        effective_save_interval,
        effective_max_epochs,
        effective_max_steps,
        boundary_policy_warnings,
    ) = _resolve_train_runner_policy(
        run_mode=run_mode,
        max_epochs=max_epochs,
        max_steps=max_steps,
        eval_interval=eval_interval,
        save_interval=save_interval,
    )
    # Configuration fallback logic
    if not checkpoint_config:
        checkpoint_config["target"] = early_stop_config.get("target", "val_metric")
        checkpoint_config["mode"] = early_stop_config.get("mode", "min")
        checkpoint_config["min_delta"] = early_stop_config.get("min_delta", 0.0)

    if "target" not in early_stop_config:
        early_stop_config["target"] = "val_metric"
    if "mode" not in early_stop_config:
        early_stop_config["mode"] = "min"

    # Create run configuration
    config = RunConfig(
        run_mode=resolved_run_mode,
        eval_interval=effective_eval_interval,
        save_interval=effective_save_interval,
        max_epochs=effective_max_epochs,
        max_steps=effective_max_steps,
        clip_grad=clip_grad,
        distributed=distributed,
        rank=rank,
        save_dir=save_dir,
        print_freq=print_freq,
        use_profiler=use_profiler,
        use_ema=ema_model is not None,
        render_type=render_type,
        ckp_file=ckp_file,
        init_file=init_file,
        device=device,
        checkpoint=checkpoint_config,
        early_stop=early_stop_config,
    )

    # Create loggerloss", "train_metric", "val_metric", "test_metric
    log_keys = ["epoch", "global_step", "train_metric", "val_metric", "test_metric", "train_loss"]
    if extra_log_keys:
        log_keys.extend(extra_log_keys)

    logger = qLogger(save_dir, columns=log_keys, console=True)
    for warning_msg in boundary_policy_warnings:
        logger.warning(warning_msg)

    effective_scheduler = None
    if scheduler is not None:
        if isinstance(scheduler, qWarmupScheduler):
            effective_scheduler = scheduler
        else:
            effective_scheduler = qWarmupScheduler(
                optimizer=optimizer, warmup_steps=0, warmup_factor=1.0, main_scheduler=scheduler
            )
    else:
        effective_scheduler = qWarmupScheduler(
            optimizer=optimizer, warmup_steps=0, warmup_factor=1.0, main_scheduler=qt.nn.DoNothing()
        )

    # Create managers
    checkpoint_manager = CheckpointManager(config.save_dir, config.rank)
    early_stopper = EarlyStopper.from_config(config.early_stop)
    best_model_tracker = BestModelTracker(
        target=config.checkpoint.get("target", "val_metric"),
        mode=config.checkpoint.get("mode", "min"),
        min_delta=config.checkpoint.get("min_delta", 0.0),
    )

    # Create training agent
    agent = RunningAgent(
        model=model,
        task=task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=effective_scheduler,
        config=config,
        device=device,
        ema_model=ema_model,
        logger=logger,
        checkpoint_manager=checkpoint_manager,
        early_stopper=early_stopper,
        best_model_tracker=best_model_tracker,
    )

    # Load checkpoint FIRST to restore training state
    checkpoint_loaded = False
    if config.ckp_file and Path(config.ckp_file).exists():
        agent.checkpoint_manager.load(
            config.ckp_file,
            device,
            model,
            task,
            optimizer,
            effective_scheduler,
            ema_model,
            agent.state,
            agent.early_stopper,
            agent.best_model_tracker,
        )
        logger.info(f"Loaded checkpoint from {config.ckp_file}")
        checkpoint_loaded = True

    # A training run must have a finite stopping condition.
    max_epochs_val = agent.config.max_epochs if agent.config.max_epochs is not None else float("inf")
    max_steps_val = agent.config.max_steps if agent.config.max_steps is not None else float("inf")
    if max_epochs_val == float("inf") and max_steps_val == float("inf"):
        # This can happen if the config has no boundaries and no explicit args are passed.
        # We check this after applying overrides to have the final picture.
        raise ValueError("Either max_epochs or max_steps must be specified for the training run.")

    # Unified LogListener handling
    progress_tracker = ProgressTracker(logger, config.print_freq, render_type=config.render_type)
    agent.add_listener("on_epoch_start", progress_tracker.on_epoch_start)
    agent.add_listener("on_progress_tick", progress_tracker.on_progress_tick)
    agent.add_listener("on_table_update", progress_tracker.on_table_update)
    agent.add_listener("on_epoch_end", progress_tracker.on_epoch_end)
    # Ensure progress tracker can pause/resume during mid-epoch evaluations
    agent.add_listener("on_eval_start", progress_tracker.on_eval_start)
    agent.add_listener("on_eval_end", progress_tracker.on_eval_end)

    # Sheet logger listener for structured data
    if log_granularity:
        sheet_logger_listener = SheetLoggerListener(logger, config, log_granularity)
        if "eval" in log_granularity:
            agent.add_listener("on_eval_end", sheet_logger_listener.on_eval_end)
        if "batch" in log_granularity:
            agent.add_listener("on_train_batch_end", sheet_logger_listener.on_train_batch_end)

    # Start profiler
    profiler = None
    if config.use_profiler:
        profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(Path(save_dir) / "profiler")),
            record_shapes=True,
        )
        profiler.start()

    # Run training
    early_stopped = False
    try:
        early_stopped = agent.run()

        if early_stopped:
            logger.info(f"Early stopping triggered at epoch {agent.state.epoch}, step {agent.state.global_step}")
        else:
            logger.info(f"Training completed: epoch {agent.state.epoch}, step {agent.state.global_step}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        early_stopped = True

    finally:
        # Stop profiler
        if profiler is not None:
            profiler.stop()
            if isinstance(profiler, profile):
                logger.info("Profiler results:")
                logger.info(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        # Ensure progress renderer resources are always released, even on exceptions.
        try:
            progress_tracker.on_run_end()
        except Exception as progress_cleanup_error:
            logger.debug(
                "ProgressTracker cleanup failed: %s",
                progress_cleanup_error,
                exc_info=True,
            )

        logger.close()
        logging.shutdown()

    # Return final results
    return {
        "best_epoch": agent.state.best_epoch,
        "best_step": agent.state.best_step,
        "best_monitored_key": agent.state.best_monitored_key,
        "best_monitored_metric": agent.state.best_monitored_metric,
        "best_model_metrics_snapshot": agent.state.best_model_metrics_snapshot,
        "final_epoch": agent.state.epoch,
        "final_step": agent.state.global_step,
        "total_train_time": agent.state.total_train_time,
        "early_stopped": early_stopped,
    }


def infer_runner(
    model: nn.Module,
    task: qTaskBase,
    dataloader: DataLoader,
    args: Optional[Any] = None,
    distributed: bool = False,
) -> List[Dict[str, Any]]:
    """
    Unified inference runner.

    Args:
        model: Model for inference.
        task: Task instance defining inference logic.
        dataloader: DataLoader for the inference dataset.
        args: Object containing command-line arguments and other configurations.
              It's the source for settings like device, rank, and ckp_file.
        distributed: Whether distributed data parallel is used.

    Returns:
        A list of dictionaries, where each dictionary contains the
        predictions and labels for a batch.
    """
    if args is None:
        raise ValueError("The 'args' parameter is required to configure the runner.")

    # Extract configuration from args
    device = _getattr_or_default(args, "device", lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    rank = _getattr_or_default(args, "rank", 0)
    ckp_file = _getattr_or_default(args, "ckp_file")
    render_type = _getattr_or_default(args, "render_type", "auto")

    # Setup logger (rank 0 only)
    logger = None
    if rank == 0:
        logger = ConsoleLogger(rank=rank)

    # Load checkpoint
    if ckp_file and Path(ckp_file).exists():
        checkpoint = torch.load(ckp_file, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:  # for backward compatibility
            model.load_state_dict(checkpoint)
        if logger:
            logger.info(f"Loaded model checkpoint from {ckp_file}")
    else:
        if logger:
            logger.warning("No checkpoint file provided or found. Running inference with initial model weights.")

    model.to(device)
    model.eval()

    all_results = []

    # DDP handling: only rank 0 collects results
    if distributed and rank != 0:
        # Non-primary processes still need to run forward pass but don't collect results
        try:
            with torch.no_grad():
                for batch_data in dataloader:
                    batch_data = move_batch_to_device(batch_data, device)
                    batch_data = task.pre_batch_forward(batch_data)
                    task.batch_forward(model, batch_data)
        except Exception as e:
            if logger:
                logger.error(f"Error during inference on rank {rank}: {e}")
            raise
        return all_results

    # Rank 0 process: run inference and collect results
    # Determine progress display strategy with fallback
    try:
        from tqdm import tqdm

        has_tqdm = True
    except ImportError:
        has_tqdm = False

    # Decide which progress strategy to use
    use_tqdm = has_tqdm and render_type != "plain"
    use_simple_logging = render_type != "plain" and not use_tqdm

    progress_iter = dataloader
    if use_tqdm:
        progress_iter = tqdm(dataloader, desc="Inference", dynamic_ncols=True)

    try:
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(progress_iter):
                # Forward pass
                batch_data = move_batch_to_device(batch_data, device)
                batch_data = task.pre_batch_forward(batch_data)
                out = task.batch_forward(model, batch_data)
                out = task.post_batch_forward(out, batch_data)

                # Extract predictions and labels
                batch_results = {"preds": out}
                if isinstance(batch_data, dict) and "y" in batch_data:
                    batch_results["labels"] = batch_data["y"]
                elif hasattr(batch_data, "y"):
                    batch_results["labels"] = batch_data.y

                # Detach tensors from the computation graph and move to CPU
                for key, value in batch_results.items():
                    if isinstance(value, torch.Tensor):
                        batch_results[key] = value.detach().cpu()

                all_results.append(batch_results)

                # Simple logging fallback (when tqdm not available but render_type != "plain")
                if use_simple_logging and (batch_idx + 1) % 10 == 0:
                    if logger:
                        logger.info(f"Inference progress: {batch_idx + 1} batches processed")

    except Exception as e:
        if logger:
            logger.error(f"Error during inference: {e}")
        raise

    return all_results

