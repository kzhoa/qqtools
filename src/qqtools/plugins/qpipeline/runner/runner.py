"""
Unified Training Runner

Config
State
Agent
Special Features
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader

import qqtools as qt

from ..entry_utils.qema import qEMA
from ..entry_utils.scheduler import (
    SCHEDULER_STEP_ON_VALID_END,
    qWarmupScheduler,
)
from ..entry_utils.type_qconfig import CheckpointConfig, EarlyStopConfig, qConfig
from ..qlogger import ConsoleLogger, qLogger
from ..task.qtask import qTaskBase
from .agent import NaNDetectedError, RunningAgent
from .runner_utils.ckp_manager import CheckpointListener, CheckpointManager
from .runner_utils.common import _getattr_or_default, _is_periodic_trigger, move_batch_to_device
from .runner_utils.earlystop import EarlyStopListener, EarlyStopper
from .runner_utils.eval_formatter import EvalSummaryListener
from .runner_utils.progress import ProgressTracker
from .runner_utils.sheet_logger import SheetLogger, SheetLoggerListener
from .runner_utils.types import (
    EventContext,
    RunConfig,
    RunMode,
    RunningState,
    TerminalEvent,
    TerminalReason,
    TrainRunnerResult,
)

__all__ = ["train_runner", "infer_runner", "SheetLoggerListener"]

TerminalCause = Literal["normal_finish", "early_stop", "user_interrupt", "exception", "nan_detected"]


def _qconfig_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _is_oom_exception(exception: BaseException) -> bool:
    if isinstance(exception, torch.cuda.OutOfMemoryError):
        return True

    exception_message = str(exception).lower()
    oom_markers = (
        "out of memory",
        "cuda out of memory",
        "mps backend out of memory",
    )
    return any(marker in exception_message for marker in oom_markers)


def _build_terminal_event(
    *,
    status: Literal["finished", "stopped", "failed"],
    reason: TerminalReason,
    epoch: int,
    step: int,
    exception: Optional[BaseException] = None,
) -> TerminalEvent:
    terminal_event: TerminalEvent = {
        "status": status,
        "reason": reason,
        "text": f"Training {status}: reason={reason}",
        "epoch": epoch,
        "step": step,
    }
    if exception is not None:
        terminal_event["exception_type"] = type(exception).__name__
    return terminal_event


def _build_terminal_event_for_cause(
    *,
    terminal_cause: TerminalCause,
    state: RunningState,
    run_config: RunConfig,
    exception: Optional[BaseException] = None,
) -> TerminalEvent:
    if terminal_cause == "user_interrupt":
        return _build_terminal_event(
            status="stopped",
            reason="user_interrupt",
            epoch=state.epoch,
            step=state.global_step,
        )

    if terminal_cause == "exception":
        if exception is None:
            raise ValueError("exception must be provided when terminal_cause='exception'")
        return _build_terminal_event(
            status="failed",
            reason="oom" if _is_oom_exception(exception) else "exception",
            epoch=state.epoch,
            step=state.global_step,
            exception=exception,
        )

    if terminal_cause == "nan_detected":
        return _build_terminal_event(
            status="failed",
            reason="nan_detected",
            epoch=state.epoch,
            step=state.global_step,
        )

    if terminal_cause == "early_stop":
        return _build_terminal_event(
            status="finished",
            reason="early_stop",
            epoch=state.epoch,
            step=state.global_step,
        )

    if terminal_cause == "normal_finish":
        if run_config.max_steps is not None and state.global_step >= run_config.max_steps:
            return _build_terminal_event(
                status="finished",
                reason="max_steps",
                epoch=state.epoch,
                step=state.global_step,
            )

        if run_config.max_epochs is not None and state.epoch >= run_config.max_epochs:
            return _build_terminal_event(
                status="finished",
                reason="max_epochs",
                epoch=state.epoch,
                step=state.global_step,
            )

    raise RuntimeError(
        "Unable to classify training terminal state "
        f"(epoch={state.epoch}, step={state.global_step}, terminal_cause={terminal_cause}, "
        f"max_steps={run_config.max_steps}, max_epochs={run_config.max_epochs})."
    )


def _emit_terminal_event(
    logger: qLogger,
    terminal_event: TerminalEvent,
) -> None:
    log_method = logger.error if terminal_event["status"] == "failed" else logger.info
    log_kwargs: Dict[str, Any] = {"extra": {"terminal_event": terminal_event}}
    log_method(terminal_event["text"], **log_kwargs)


def _build_and_emit_terminal_event(
    *,
    logger: qLogger,
    terminal_cause: TerminalCause,
    state: RunningState,
    run_config: RunConfig,
    exception: Optional[BaseException] = None,
) -> TerminalEvent:
    terminal_event = _build_terminal_event_for_cause(
        terminal_cause=terminal_cause,
        state=state,
        run_config=run_config,
        exception=exception,
    )
    _emit_terminal_event(logger, terminal_event)
    return terminal_event


def _finalize_train_runner(
    *,
    logger: qLogger,
    sheet_logger: Optional[SheetLogger],
    progress_tracker: Optional[ProgressTracker],
    profiler: Optional[profile],
) -> None:
    if profiler is not None:
        profiler.stop()
        if isinstance(profiler, profile):
            logger.info("Profiler results:")
            logger.info(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    try:
        if progress_tracker is not None:
            progress_tracker.on_run_end()
    except Exception as progress_cleanup_error:
        logger.debug(
            "ProgressTracker cleanup failed: %s",
            progress_cleanup_error,
            exc_info=True,
        )

    if sheet_logger is not None:
        sheet_logger.close()
    logger.close()
    logging.shutdown()


def _prepare_training_session(
    *,
    config: RunConfig,
    save_dir: str,
    logger: qLogger,
    sheet_logger: Optional[SheetLogger],
    checkpoint_manager: CheckpointManager,
    agent: RunningAgent,
    early_stopper: EarlyStopper,
    eval_summary_listener: EvalSummaryListener,
    early_stop_listener: EarlyStopListener,
    checkpoint_listener: CheckpointListener,
    effective_scheduler: Optional[qWarmupScheduler],
    device: torch.device,
    model: nn.Module,
    task: qTaskBase,
    optimizer: torch.optim.Optimizer,
    ema_model: Optional[qEMA],
    log_granularity: Optional[List[Literal["eval", "batch"]]],
) -> Tuple[Optional[ProgressTracker], Optional[profile]]:
    progress_tracker: Optional[ProgressTracker] = None
    profiler: Optional[profile] = None

    if config.ckp_file and Path(config.ckp_file).exists():
        checkpoint_manager.load(
            config.ckp_file,
            device,
            model,
            task,
            optimizer,
            effective_scheduler,
            ema_model,
            agent.state,
            early_stopper,
            agent.best_model_tracker,
        )
        logger.info(f"Loaded checkpoint from {config.ckp_file}")

    def _on_validation_end_step_scheduler(context: EventContext) -> None:
        if effective_scheduler is None:
            return

        if effective_scheduler.step_on != SCHEDULER_STEP_ON_VALID_END:
            return

        eval_results = context.eval_results or {}
        effective_scheduler.step_main(metrics=eval_results.get("val_metric"))

    agent.add_listener("on_validation_end", _on_validation_end_step_scheduler)
    agent.add_listener("on_validation_end", eval_summary_listener.on_validation_end)
    agent.add_listener("on_validation_end", early_stop_listener.on_validation_end)
    agent.add_listener("on_checkpoint_request", checkpoint_listener.on_checkpoint_request)

    progress_tracker = ProgressTracker(logger, config.print_freq, render_type=config.render_type)
    agent.add_listener("on_epoch_start", progress_tracker.on_epoch_start)
    agent.add_listener("on_progress_tick", progress_tracker.on_progress_tick)
    agent.add_listener("on_table_update", progress_tracker.on_table_update)
    agent.add_listener("on_epoch_end", progress_tracker.on_epoch_end)
    agent.add_listener("on_eval_start", progress_tracker.on_eval_start)
    agent.add_listener("on_eval_end", progress_tracker.on_eval_end)

    if log_granularity and sheet_logger is not None:
        sheet_logger_listener = SheetLoggerListener(
            sheet_logger=sheet_logger,
            run_config=config,
            log_granularity=log_granularity,
            logger=logger,
        )
        if "eval" in log_granularity:
            agent.add_listener("on_eval_end", sheet_logger_listener.on_eval_end)
        if "batch" in log_granularity:
            agent.add_listener("on_train_batch_end", sheet_logger_listener.on_train_batch_end)

    if config.use_profiler:
        profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(Path(save_dir) / "profiler")),
            record_shapes=True,
        )
        profiler.start()

    return progress_tracker, profiler


def _resolve_train_runner_policy(
    run_mode: Union[str, RunMode],
    max_epochs: Optional[int],
    max_steps: Optional[int],
    eval_interval: int,
    save_interval: Optional[int],
) -> Tuple[RunMode, int, int, Optional[int], Optional[int], List[str]]:
    """Resolve train-runner-owned policy fields into effective runtime values."""

    def _is_positive_int(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 1

    if run_mode is None:
        raise ValueError("run_mode cannot be None")

    if not _is_positive_int(eval_interval):
        raise ValueError("eval_interval must be a positive integer")

    resolved_run_mode = RunMode(run_mode)
    effective_save_interval = save_interval if save_interval is not None else eval_interval
    if not _is_positive_int(effective_save_interval):
        raise ValueError("save_interval must be a positive integer")

    policy_warnings: List[str] = []
    effective_max_epochs: Optional[int] = None
    effective_max_steps: Optional[int] = None

    if resolved_run_mode == RunMode.EPOCH:
        if max_epochs is None:
            raise ValueError("max_epochs must be specified when run_mode='epoch'")
        if not _is_positive_int(max_epochs):
            raise ValueError("max_epochs must be a positive integer when run_mode='epoch'.")
        effective_max_epochs = max_epochs
        if max_steps is not None:
            policy_warnings.append(
                f"[run_mode=EPOCH] max_steps={max_steps} is ignored by mutual-exclusion policy; "
                f"training will be controlled by max_epochs={max_epochs}."
            )
    else:  # RunMode.STEP
        if max_steps is None:
            raise ValueError("max_steps must be specified when run_mode='step'")
        if not _is_positive_int(max_steps):
            raise ValueError("max_steps must be a positive integer when run_mode='step'.")
        effective_max_steps = max_steps
        if max_epochs is not None:
            policy_warnings.append(
                f"[run_mode=STEP] max_epochs={max_epochs} is ignored by mutual-exclusion policy; "
                f"training will be controlled by max_steps={max_steps}."
            )

    return (
        resolved_run_mode,
        eval_interval,
        effective_save_interval,
        effective_max_epochs,
        effective_max_steps,
        policy_warnings,
    )


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
    accum_grad: Optional[int] = None,
    log_granularity: Optional[List[Literal["eval", "batch"]]] = ["eval"],
    allow_auto_offload: bool = False,
) -> TrainRunnerResult:
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
        accum_grad: Optional gradient accumulation factor. `None` disables accumulation.
        allow_auto_offload: Whether to enable automatic EMA/model offload policy

    Returns:
        Structured training result with terminal contract fields
    """
    # Handle compatibility parameters
    if args is None:
        raise ValueError("The 'args' parameter is required to configure the runner.")
    if accum_grad is not None:
        if isinstance(accum_grad, bool) or not isinstance(accum_grad, int):
            raise ValueError("accum_grad must be an integer when specified")
        if accum_grad < 1:
            raise ValueError("accum_grad must be a positive integer when specified")

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

    if checkpoint_config is None:
        checkpoint_config = {}
    if early_stop_config is None:
        early_stop_config = {}

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
        eval_interval=1 if eval_interval is None else eval_interval,
        save_interval=save_interval,
    )
    # Configuration fallback logic
    if not checkpoint_config:
        checkpoint_config = {
            "target": _qconfig_get(early_stop_config, "target", "val_metric"),
            "mode": _qconfig_get(early_stop_config, "mode", "min"),
            "min_delta": _qconfig_get(early_stop_config, "min_delta", 0.0),
            "regular_latest_only": True,
        }

    if _qconfig_get(early_stop_config, "target") is None:
        early_stop_config["target"] = "val_metric"
    if _qconfig_get(early_stop_config, "mode") is None:
        early_stop_config["mode"] = "min"

    # Create run configuration
    config = RunConfig(
        run_mode=resolved_run_mode,
        eval_interval=effective_eval_interval,
        save_interval=effective_save_interval,
        max_epochs=effective_max_epochs,
        max_steps=effective_max_steps,
        clip_grad=clip_grad,
        accum_grad=accum_grad,
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

    # Define structured sheet-log columns.
    log_keys = ["epoch", "global_step", "train_metric", "val_metric", "test_metric", "train_loss"]
    if extra_log_keys:
        log_keys.extend(extra_log_keys)

    logger = qLogger(save_dir, console=True)
    sheet_logger = None
    if log_granularity and config.rank == 0:
        metrics_file = Path(save_dir) / "metrics.csv"
        sheet_logger = SheetLogger(metrics_file, columns=log_keys)
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

    # Create managers and callback listeners
    checkpoint_manager = CheckpointManager(
        config.save_dir,
        config.rank,
        keep_only_latest_regular=_qconfig_get(config.checkpoint, "regular_latest_only", True),
    )
    early_stopper = EarlyStopper.from_config(config.early_stop)

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
        allow_auto_offload=allow_auto_offload,
        logger=logger,
    )

    checkpoint_listener = CheckpointListener(
        checkpoint_manager=checkpoint_manager,
        model=model,
        task=task,
        optimizer=optimizer,
        scheduler=effective_scheduler,
        ema_model=ema_model,
        early_stopper=early_stopper,
        best_model_tracker=agent.best_model_tracker,
    )
    early_stop_listener = EarlyStopListener(
        early_stopper=early_stopper,
        target=config.early_stop.get("target", "val_metric"),
        logger=logger,
    )
    eval_summary_listener = EvalSummaryListener(
        logger=logger,
        target_key=_qconfig_get(config.checkpoint, "target", "val_metric"),
        target_mode=_qconfig_get(config.checkpoint, "mode", "min"),
    )

    progress_tracker = None
    profiler = None
    terminal_event: Optional[TerminalEvent] = None

    try:
        progress_tracker, profiler = _prepare_training_session(
            config=config,
            save_dir=save_dir,
            logger=logger,
            sheet_logger=sheet_logger,
            checkpoint_manager=checkpoint_manager,
            agent=agent,
            early_stopper=early_stopper,
            eval_summary_listener=eval_summary_listener,
            early_stop_listener=early_stop_listener,
            checkpoint_listener=checkpoint_listener,
            effective_scheduler=effective_scheduler,
            device=device,
            model=model,
            task=task,
            optimizer=optimizer,
            ema_model=ema_model,
            log_granularity=log_granularity,
        )
        loop_stopped_by_early_stop = agent.run()
        terminal_event = _build_and_emit_terminal_event(
            logger=logger,
            terminal_cause="early_stop" if loop_stopped_by_early_stop else "normal_finish",
            state=agent.state,
            run_config=config,
        )

    except KeyboardInterrupt:
        terminal_event = _build_and_emit_terminal_event(
            logger=logger,
            terminal_cause="user_interrupt",
            state=agent.state,
            run_config=config,
        )

    except NaNDetectedError:
        terminal_event = _build_and_emit_terminal_event(
            logger=logger,
            terminal_cause="nan_detected",
            state=agent.state,
            run_config=config,
        )

    except Exception as error:
        terminal_event = _build_and_emit_terminal_event(
            logger=logger,
            terminal_cause="exception",
            state=agent.state,
            run_config=config,
            exception=error,
        )
        raise

    finally:
        _finalize_train_runner(
            logger=logger,
            sheet_logger=sheet_logger,
            progress_tracker=progress_tracker,
            profiler=profiler,
        )

    # Return final results
    if terminal_event is None:
        raise RuntimeError("train_runner completed without terminal_event classification")

    return {
        "best_epoch": agent.state.best_epoch,
        "best_step": agent.state.best_step,
        "best_monitored_key": agent.state.best_monitored_key,
        "best_monitored_metric": agent.state.best_monitored_metric,
        "best_model_metrics_snapshot": agent.state.best_model_metrics_snapshot,
        "final_epoch": agent.state.epoch,
        "final_step": agent.state.global_step,
        "total_train_time": agent.state.total_train_time,
        "early_stopped": terminal_event["reason"] == "early_stop",
        "terminal_event": terminal_event,
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

