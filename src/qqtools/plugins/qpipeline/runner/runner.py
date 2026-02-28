"""
Unified Training Runner

Config
State
Agent
Special Features
"""

import gc
import logging
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
from .best_model import BestModelManager
from .ckp_manager import CheckpointManager
from .earlystop import EarlyStopper
from .progress import ProgressTracker
from .types import EventContext, RunConfig, RunMode, RunningState

__all__ = ["train_runner", "infer_runner"]


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

    if isinstance(batch_data, torch.Tensor):
        return batch_data.to(device)
    elif isinstance(batch_data, dict):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_data.items()}
    elif isinstance(batch_data, (tuple, list)):
        return type(batch_data)(v.to(device) if isinstance(v, torch.Tensor) else v for v in batch_data)
    elif hasattr(batch_data, "to"):
        return batch_data.to(device)
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
        best_model_manager: Optional[BestModelManager] = None,
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
            best_model_manager: Best model manager (optional, None to skip best model tracking)
            listeners: Event listeners dictionary
            state: Running state
            avg_bank: Average metrics manager
        """
        self.model = model
        self.task = task
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or RunConfig()
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
        self.best_model_manager = best_model_manager

        # Listeners for extensibility
        default_listeners = {
            "on_epoch_start": [],
            "on_epoch_end": [],
            "on_batch_start": [],
            "on_batch_end": [],
            "on_train_batch_end": [],
            "on_eval_start": [],
            "on_eval_end": [],
            "on_checkpoint_save": [],
            "on_early_stop": [],
        }
        self.listeners = listeners or default_listeners

        # Internal variables
        self._train_iter = None
        self.avg_bank = avg_bank or AvgBank()

    def add_listener(self, event: str, listener: Callable):
        if event in self.listeners:
            self.listeners[event].append(listener)
        else:
            warnings.warn(f"Unknown event: {event}")

    def _trigger(self, event: str, context: EventContext):
        for listener in self.listeners.get(event, []):
            try:
                listener(context)
            except Exception as e:
                self.logger.error(f"Listener error for event '{event}' in {listener.__name__}: {type(e).__name__}: {e}")
                if self.config.fail_on_listener_error:
                    raise

    def _prepare_batch(self, batch_data):
        """Prepare batch data: move to device and convert dtype if needed.

        Args:
            batch_data: Raw batch data

        Returns:
            Prepared batch data
        """
        batch_data = move_batch_to_device(batch_data, self.device)
        if hasattr(batch_data, "to_dtype") and self.config.dtype is not None:
            batch_data = batch_data.to_dtype(self.config.dtype)
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

    def _start_new_epoch(self):
        """Initialize a new epoch: create iterator and setup distributed sampling."""
        self._train_iter = iter(self.train_loader)

        # Update distributed sampler epoch for proper shuffling
        if self.config.distributed and hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(self.state.epoch)

        # Trigger epoch start event
        self._trigger("on_epoch_start", context=EventContext(state=self.state, stage="train"))

    def _get_next_batch(self) -> Tuple[Any, bool]:
        """
        Gets the next batch from the training data loader. Also detects when an epoch ends.
        Supports resuming from mid-epoch checkpoints by skipping already-processed batches.

        Returns:
            A tuple containing:
            - The batch data.
            - A boolean flag that is True if the batch is the first of a new epoch.
        """
        is_epoch_end = False

        if self._train_iter is None:
            self._start_new_epoch()

            # If resuming from a checkpoint at mid-epoch, skip already-processed batches
            batches_to_skip = self.state.batch_idx_in_epoch
            for _ in range(batches_to_skip):
                try:
                    next(self._train_iter)
                except StopIteration:
                    # This shouldn't happen, but handle it defensively
                    self._handle_epoch_end()
                    self.state.batch_idx_in_epoch = 0
                    self._start_new_epoch()
                    break

        try:
            batch_data = next(self._train_iter)
            # Increment batch index after successfully getting the batch
            self.state.batch_idx_in_epoch += 1
        except StopIteration:
            is_epoch_end = True
            # At epoch end, reset batch index before starting new epoch
            self._handle_epoch_end()
            self._start_new_epoch()
            batch_data = next(self._train_iter)
            # First batch of new epoch
            self.state.batch_idx_in_epoch = 1

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
        batch_metrics = self.task.batch_metric(out, batch_data)

        # Training mode: calculate loss and backward propagation
        losses = self.task.batch_loss(out, batch_data, self.loss_fn)
        loss_tensor, loss_cnt = losses.get("loss", (None, 1))

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
        """Evaluate the model"""
        eval_model = self.ema_model if use_ema and self.ema_model is not None else (model or self.model)

        results = {}

        # Validation set
        if self.val_loader is not None:
            val_metrics = self._run_evaluation(eval_model, self.val_loader, "val", stage="val")
            results.update(val_metrics)

        # Test set
        if self.test_loader is not None:
            test_metrics = self._run_evaluation(eval_model, self.test_loader, "test", stage="test")
            results.update(test_metrics)

        return results

    def _run_evaluation(
        self, model: nn.Module, data_loader: DataLoader, prefix: str = "", stage: str = "val"
    ) -> Dict[str, Any]:
        """Run evaluation"""
        was_training = model.training
        model.eval()
        avg_bank = AvgBank()

        # Setup evaluation progress bar for rank 0
        is_rank_zero = getattr(self.config, "rank", 0) == 0
        use_tqdm = is_rank_zero and getattr(self.config, "render_type", "auto") != "plain"

        if use_tqdm:
            try:
                from tqdm import tqdm

                eval_loader = tqdm(data_loader, desc=f"Eval ({prefix})", dynamic_ncols=True, leave=False)
            except ImportError:
                eval_loader = data_loader
        else:
            eval_loader = data_loader

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(eval_loader):
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

                # Trigger batch end event with eval stage info
                batch_context = EventContext(
                    state=self.state,
                    batch_idx=batch_idx,
                    total_batches=len(data_loader),
                    batch_metrics=batch_metrics,
                    avg_bank=avg_bank.to_dict(self.config.distributed),
                    stage=stage,
                )
                self._trigger("on_batch_end", context=batch_context)

        # Calculate average
        avg_metrics = avg_bank.gather_average(self.config.distributed)

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
        self._trigger("on_eval_start", context=EventContext(state=self.state))

        # Evaluate the original model
        eval_results = self.evaluate(self.model, use_ema=False)

        # Evaluate EMA model (if exists)
        if self.ema_model is not None:
            ema_results = self.evaluate(self.model, use_ema=True)
            # Merge results and add ema prefix
            for key, value in ema_results.items():
                eval_results[f"ema_{key}"] = value

        # Update state
        self.state.update_current_metrics(eval_results)

        # Trigger evaluation end event
        self._trigger("on_eval_end", context=EventContext(state=self.state, eval_results=eval_results))

        # Learning rate update (epoch-wise scheduler)
        if self.scheduler is not None:
            self.scheduler.step_main(metrics=eval_results.get("val_metric"))

        # Determine if best
        target = self.config.checkpoint.get("target", "val_metric")
        current_val = eval_results.get(target)

        is_best = False
        if current_val is not None and self.best_model_manager is not None:
            is_best = self.best_model_manager.update(current_val, self.state.epoch, self.state.global_step)

        if is_best:
            self._save_best_checkpoint()

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
            if should_stop:
                self.logger.info(stop_msg)
                self._trigger("on_early_stop", context=EventContext(state=self.state))
                return True

        return False

    def _save_best_checkpoint(self):
        """Saves the best model checkpoint and updates best metrics."""
        # Update best metrics
        self.state.best_epoch = self.state.epoch
        self.state.best_step = self.state.global_step
        self.state.best_train_metric = self.state.current_train_metric
        self.state.best_val_metric = self.state.current_val_metric
        self.state.best_test_metric = self.state.current_test_metric

        # Save best checkpoint if manager is available
        if self.checkpoint_manager is not None:
            ckp_path = self.checkpoint_manager.save(
                self.state,
                self.model,
                self.optimizer,
                self.scheduler,
                self.ema_model,
                self.early_stopper,
                self.best_model_manager,
                self.task,
                is_best=True,
            )
            self.state.best_ckp_file = ckp_path
            self._trigger(
                "on_checkpoint_save",
                context=EventContext(state=self.state, checkpoint_type="best", checkpoint_path=ckp_path),
            )

    def _save_regular_checkpoint(self):
        """Saves a checkpoint of the current state, not contingent on performance."""
        if self.checkpoint_manager is None:
            return

        ckp_path = self.checkpoint_manager.save(
            self.state,
            self.model,
            self.optimizer,
            self.scheduler,
            self.ema_model,
            self.early_stopper,
            self.best_model_manager,
            self.task,
            is_best=False,
        )
        self._trigger(
            "on_checkpoint_save",
            context=EventContext(state=self.state, checkpoint_type="regular", checkpoint_path=ckp_path),
        )

    def _handle_periodic_events(self, is_epoch_end: bool) -> bool:
        """
        Handles all periodic events like evaluation, checkpointing, and early stopping.
        This is the central place for managing the training lifecycle based on run_mode.

        Args:
            is_epoch_end: Flag indicating if the current step is the last in an epoch.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        is_step_trigger = (
            self.config.run_mode == RunMode.STEP and self.state.global_step % self.config.eval_interval == 0
        )
        is_epoch_trigger = self.config.run_mode == RunMode.EPOCH and is_epoch_end
        is_eval_trigger = is_step_trigger or is_epoch_trigger

        should_stop = False
        if is_eval_trigger:
            should_stop = self._run_evaluation_and_update()

        should_save = False
        if self.config.save_interval:
            should_save = self.state.global_step % self.config.save_interval == 0
        else:
            # if no save_interval is set, default to saving at the same time as evaluation
            should_save = is_eval_trigger

        if should_save:
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

        while self.state.global_step < self.state.max_steps and self.state.epoch < self.state.max_epochs:
            try:
                batch_data, is_epoch_end = self._get_next_batch()

                # Train a single batch
                batch_start_time = time.time()
                self._trigger(
                    "on_batch_start",
                    context=EventContext(
                        state=self.state,
                        batch_idx=self.state.global_step % len(self.train_loader),
                        total_batches=len(self.train_loader),
                    ),
                )

                batch_metrics = self.train_batch(batch_data)

                # Unconditionally step warmup scheduler every batch
                if self.scheduler is not None:
                    self.scheduler.step_warmup()

                # Update average bank for epoch-level metrics
                for k, v in batch_metrics.items():
                    if isinstance(v, (int, float)):
                        self.avg_bank.add(k, v)

                # Record batch time and update total training time
                batch_time = time.time() - batch_start_time
                self.state.total_train_time += batch_time

                # Trigger batch end event for listeners (e.g., progress bars, loggers)
                batch_context = EventContext(
                    state=self.state,
                    batch_idx=self.state.global_step % len(self.train_loader),
                    total_batches=len(self.train_loader),
                    batch_metrics=batch_metrics,
                    avg_bank=self.avg_bank.to_dict(self.config.distributed),
                    lr=self.optimizer.param_groups[0]["lr"] if self.optimizer else None,
                    stage="train",
                )
                self._trigger("on_batch_end", context=batch_context)
                self._trigger("on_train_batch_end", context=batch_context)

                # Handle periodic events (evaluation, checkpointing, etc.)
                if self._handle_periodic_events(is_epoch_end):
                    return True  # Early stopping was triggered

                # Increment global step *after* all step-based logic is complete
                self.state.global_step += 1

                # Periodically clean up memory
                if self.state.global_step % 100 == 0:
                    gc.collect()
                    # if torch.cuda.is_available():
                    #     torch.cuda.empty_cache()

            except KeyboardInterrupt:
                self.logger.info("Training interrupted by user.")
                return True  # Treat as early stop

        return False  # Completed normally

    def run(self) -> bool:
        """
        Runs the training process by setting up and invoking the main training loop.

        Returns:
            bool: True if training was stopped early, False otherwise.
        """
        self.logger.info(
            f"Starting training (mode={self.config.run_mode.value}, "
            f"eval_interval={self.config.eval_interval}, "
            f"max_epochs={self.state.max_epochs}, "
            f"max_steps={self.state.max_steps})"
        )

        early_stopped = self._training_loop()

        # Final log message
        if not early_stopped:
            if self.state.global_step >= self.state.max_steps:
                self.logger.info(f"Reached max_steps={self.state.max_steps}")
            elif self.state.epoch >= self.state.max_epochs:
                self.logger.info(f"Reached max_epochs={self.state.max_epochs}")

        return early_stopped


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
        save_interval: Regular checkpoint saving interval (steps), None means no regular saving

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

    if run_mode is None:
        raise ValueError("run_mode cannot be None. Supported values are 'epoch' and 'step'.")
    eval_interval = 1 if eval_interval is None else eval_interval

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
        run_mode=RunMode(run_mode),
        eval_interval=eval_interval,
        save_interval=save_interval,
        max_epochs=max_epochs,
        max_steps=max_steps,
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

    # Create logger
    log_keys = ["epoch", "global_step", "train_metric", "val_metric", "test_metric", "train_loss"]
    if extra_log_keys:
        log_keys.extend(extra_log_keys)

    logger = qLogger(save_dir, columns=log_keys, console=True)

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
    best_model_manager = BestModelManager(
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
        best_model_manager=best_model_manager,
    )

    # Load checkpoint FIRST to restore training state
    checkpoint_loaded = False
    if config.ckp_file and Path(config.ckp_file).exists():
        agent.checkpoint_manager.load(
            config.ckp_file,
            device,
            model,
            optimizer,
            effective_scheduler,
            ema_model,
            agent.state,
            agent.early_stopper,
            agent.best_model_manager,
            task,
        )
        logger.info(f"Loaded checkpoint from {config.ckp_file}")
        checkpoint_loaded = True

    # Initialize/Override state AFTER checkpoint loading
    # This allows explicit overrides of max_epochs/max_steps if needed
    if max_epochs is not None:
        agent.state.max_epochs = max_epochs
    elif not checkpoint_loaded:
        agent.state.max_epochs = float("inf")
    # If checkpoint_loaded and max_epochs is None, keep the value from checkpoint

    if max_steps is not None:
        agent.state.max_steps = max_steps
    elif not checkpoint_loaded:
        agent.state.max_steps = float("inf")
    # If checkpoint_loaded and max_steps is None, keep the value from checkpoint

    # Validate run_mode and max_epochs/max_steps consistency
    if config.run_mode == RunMode.STEP:
        # In STEP mode, both max_epochs and max_steps can be used as stopping conditions
        if agent.state.max_epochs != float("inf") and agent.state.max_steps != float("inf"):
            logger.warning(
                f"[run_mode=STEP] Both max_epochs={agent.state.max_epochs} and "
                f"max_steps={agent.state.max_steps} are specified. "
                f"Training will stop when either condition is reached."
            )
    elif config.run_mode == RunMode.EPOCH:
        # In EPOCH mode, max_steps is ignored
        if agent.state.max_steps != float("inf"):
            logger.warning(
                f"[run_mode=EPOCH] max_steps={agent.state.max_steps} is specified but will be ignored "
                f"in EPOCH mode. Training will be controlled by max_epochs={agent.state.max_epochs}."
            )
    else:
        # Should not happen due to RunMode enum, but handle defensively
        logger.warning(f"Unknown run_mode: {config.run_mode}. Defaulting to EPOCH mode.")
        config.run_mode = RunMode.EPOCH

    # Unified LogListener handling
    progress_tracker = ProgressTracker(logger, config.print_freq, render_type=config.render_type)
    agent.add_listener("on_epoch_start", progress_tracker.on_epoch_start)
    agent.add_listener("on_batch_end", progress_tracker.on_batch_end)
    agent.add_listener("on_epoch_end", progress_tracker.on_epoch_end)
    # Ensure progress tracker can pause/resume during mid-epoch evaluations
    agent.add_listener("on_eval_start", progress_tracker.on_eval_start)
    agent.add_listener("on_eval_end", progress_tracker.on_eval_end)

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

        logger.close()
        logging.shutdown()

    # Return final results
    return {
        "best_epoch": agent.state.best_epoch,
        "best_step": agent.state.best_step,
        "best_val_metric": agent.state.best_val_metric,
        "best_test_metric": agent.state.best_test_metric,
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
