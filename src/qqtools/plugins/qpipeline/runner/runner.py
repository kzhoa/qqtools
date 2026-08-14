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

from ..entry_utils.optimizer import prepare_optimizer
from ..entry_utils.qema import qEMA
from ..entry_utils.scheduler import SCHEDULER_STEP_ON_VALID_END, prepare_scheduler, qWarmupScheduler
from ..entry_utils.type_qconfig import CheckpointConfig, EarlyStopConfig, qConfig
from ..qlogger import ConsoleLogger, qLogger
from ..task.qtask import TASK_LIFECYCLE_HOOKS, qTaskBase
from ..types import Stage
from .agent import NaNDetectedError, RunningAgent
from .contracts import EventListenerBindings, ObserverBindings, TaskValidationContext
from .hooks import RunnerHooks
from .runner_utils.best_model import BestModelTracker
from .runner_utils.ckp_manager import CheckpointManager, CheckpointPlugin, CheckpointPolicy
from .runner_utils.common import _getattr_or_default, _is_periodic_trigger, move_batch_to_device
from .runner_utils.earlystop import EarlyStopController, EarlyStopper
from .runner_utils.epoch_suffix import standardize_epoch_suffixes
from .runner_utils.evaluation import EvaluationResult
from .runner_utils.eval_formatter import EvalSummaryObserver
from .runner_utils.progress import ProgressTracker
from .runner_utils.metrics_jsonl import MetricsJsonlLogger, MetricsJsonlObserver
from .runner_utils.types import (
    RunConfig,
    RunMode,
    RunningState,
    TerminalEvent,
    TerminalReason,
    TrainRunnerResult,
)

__all__ = ["train_runner", "MetricsJsonlObserver"]

TerminalCause = Literal[
    "normal_finish", "early_stop", "user_interrupt", "exception", "nan_detected", "logger_failure",
]


def _qconfig_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


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

    if terminal_cause == "logger_failure":
        return _build_terminal_event(
            status="failed",
            reason="logger_failure",
            epoch=state.epoch,
            step=state.global_step,
            exception=exception,
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
    progress_tracker: Optional[ProgressTracker],
    profiler: Optional[profile],
    owns_logger: bool = True,
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



def _close_runner_logger(logger: qLogger, owns_logger: bool) -> None:
    if owns_logger:
        logger.close()
        logging.shutdown()


def _prepare_training_session(
    *,
    config: RunConfig,
    save_dir: str,
    logger: qLogger,
    checkpoint_plugin: CheckpointPlugin,
    device: torch.device,
    progress_tracker: Optional[ProgressTracker],
) -> Tuple[Optional[ProgressTracker], Optional[profile]]:
    profiler: Optional[profile] = None

    checkpoint_plugin.restore_if_requested(device)

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
            if not _is_positive_int(max_epochs):
                raise ValueError("max_epochs must be a positive integer when specified in run_mode='step'.")
            effective_max_epochs = max_epochs
            policy_warnings.append(
                f"[run_mode=STEP] max_epochs={max_epochs} is enabled as a secondary stopping boundary; "
                f"training will stop when max_steps={max_steps} or max_epochs={max_epochs} is reached first."
            )

    return (
        resolved_run_mode,
        eval_interval,
        effective_save_interval,
        effective_max_epochs,
        effective_max_steps,
        policy_warnings,
    )


def _resolve_step_mode_max_steps(
    run_mode: Union[str, RunMode],
    task: qTaskBase,
    max_epochs: Optional[int],
    max_steps: Optional[int],
    accum_grad: Optional[int],
) -> Tuple[Optional[int], Optional[int], List[str]]:
    if run_mode is None or RunMode(run_mode) != RunMode.STEP:
        return max_steps, max_epochs, []

    if max_steps is not None:
        return max_steps, max_epochs, []

    if max_epochs is None:
        return None, max_epochs, []

    if not _is_positive_int(max_epochs):
        raise ValueError(
            f"max_epochs={max_epochs!r} is not a positive integer; " "cannot infer max_steps when run_mode='step'."
        )

    effective_accum_grad = 1 if accum_grad is None else accum_grad
    if not _is_positive_int(effective_accum_grad):
        raise ValueError(
            f"accum_grad={accum_grad!r} is not a positive integer; " "cannot infer max_steps when run_mode='step'."
        )

    try:
        train_loader_length = len(task.train_loader)
    except (AttributeError, TypeError):
        train_loader_length = None

    if not _is_positive_int(train_loader_length):
        raise ValueError(
            "max_steps cannot be inferred when run_mode='step'; "
            "provide max_steps explicitly or ensure len(task.train_loader) is available as a positive integer."
        )

    optimizer_steps_per_epoch = (train_loader_length + effective_accum_grad - 1) // effective_accum_grad
    inferred_max_steps = optimizer_steps_per_epoch * max_epochs
    return (
        inferred_max_steps,
        max_epochs,
        [
            f"[run_mode=STEP] max_steps is not provided; inferred max_steps={inferred_max_steps} "
            f"from len(task.train_loader)={train_loader_length}, accum_grad={effective_accum_grad}, "
            f"and max_epochs={max_epochs}."
        ],
    )



# Design Rationale: Boundary Policy Ownership
# The orchestration layer (train_runner) owns the business policy for run
# boundaries:
# - EPOCH mode keeps max_epochs and ignores max_steps
# - STEP mode requires a concrete max_steps and may also keep max_epochs as a secondary cap
#
# RunningAgent remains policy-agnostic and simply stops based on the concrete
# boundaries passed in via RunConfig.
def train_runner(
    model: nn.Module,
    task: qTaskBase,
    loss_fn: Callable,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[qWarmupScheduler] = None,
    args: Optional[qConfig] = None,
    logger: Optional[qLogger] = None,
    max_epochs: Optional[int] = None,
    max_steps: Optional[int] = None,
    clip_grad: Optional[float] = None,
    distributed: bool = False,
    save_dir: str = "./logs",
    print_freq: int = 10,
    extra_ckp_caches: Optional[Dict[str, Any]] = None,
    use_profiler: bool = False,
    ema_model: Optional[qEMA] = None,
    run_mode: Union[str, RunMode] = "epoch",
    eval_interval: Union[int, str, None] = 1,
    save_interval: Optional[Union[int, str]] = None,
    accum_grad: Optional[int] = None,
    log_granularity: Optional[List[Literal["eval", "batch"]]] = ["eval"],
    auto_offload: bool = False,
) -> TrainRunnerResult:
    """
    Self-contained training runner.

    When called from QPipeline, optimizer/scheduler/logger are typically None
    and will be created internally from args. Independent callers may pass in
    pre-built instances to override the defaults.

    Args:
        model: Model
        task: Task instance
        loss_fn: Loss function
        optimizer: Optimizer. If None, created from args.optim config.
        scheduler: Learning rate scheduler. If None, created from args.optim config.
                   If a plain LRScheduler (non-qWarmupScheduler), it will be wrapped.
                   If already a qWarmupScheduler, used directly without re-wrapping.
        args: Object containing command-line arguments and other configurations.
              It is the single source for settings like device, rank, checkpoint, etc.
        logger: Optional logger instance. If None, a new qLogger is created for save_dir.
        max_epochs: Maximum number of epochs.
        max_steps: Maximum number of optimizer steps.
        clip_grad: Gradient clipping
        distributed: Whether to use distributed training
        save_dir: Directory to save checkpoints
        print_freq: Frequency of printing logs
        extra_ckp_caches: Extra checkpoint caches
        use_profiler: Whether to use profiler
        ema_model: EMA model
        run_mode: Running mode ("epoch" or "step")
        eval_interval: Evaluation interval. Epoch-suffix strings are converted to optimizer steps.
        save_interval: Checkpoint interval. Epoch-suffix strings are converted to optimizer steps.
        accum_grad: Optional gradient accumulation factor. `None` disables accumulation.
        auto_offload: Whether to offload the main model during EMA evaluation

    Returns:
        Structured training result with terminal contract fields
    """
    # Handle compatibility parameters
    if args is None:
        raise ValueError("The 'args' parameter is required to configure the runner.")
    if scheduler is not None and optimizer is None:
        raise ValueError(
            "Cannot pass scheduler without optimizer. A scheduler must be bound to "
            "the optimizer it was created with."
        )
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
    completion_config = _getattr_or_default(runner_config, "completion", dict)
    task_config = _getattr_or_default(args, "task", dict)
    ckp_file = _getattr_or_default(args, "ckp_file")
    render_type = _getattr_or_default(args, "render_type", "auto")

    if checkpoint_config is None:
        checkpoint_config = {}
    if early_stop_config is None:
        early_stop_config = {}
    if completion_config is None:
        completion_config = {}

    eval_interval, save_interval, epoch_suffix_logs = standardize_epoch_suffixes(
        args=args,
        task=task,
        run_mode=run_mode,
        eval_interval=eval_interval,
        save_interval=save_interval,
        accum_grad=accum_grad,
        distributed=distributed,
    )

    effective_input_max_steps, effective_input_max_epochs, inferred_max_step_warnings = _resolve_step_mode_max_steps(
        run_mode=run_mode,
        task=task,
        max_epochs=max_epochs,
        max_steps=max_steps,
        accum_grad=accum_grad,
    )

    (
        resolved_run_mode,
        effective_eval_interval,
        effective_save_interval,
        effective_max_epochs,
        effective_max_steps,
        boundary_policy_warnings,
    ) = _resolve_train_runner_policy(
        run_mode=run_mode,
        max_epochs=effective_input_max_epochs,
        max_steps=effective_input_max_steps,
        eval_interval=1 if eval_interval is None else eval_interval,
        save_interval=save_interval,
    )
    boundary_policy_warnings = [*inferred_max_step_warnings, *boundary_policy_warnings]
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
    completion_eval = _qconfig_get(completion_config, "eval", False)
    completion_save = _qconfig_get(completion_config, "save", False)
    if not isinstance(completion_eval, bool) or not isinstance(completion_save, bool):
        raise ValueError("runner.completion.eval and runner.completion.save must be booleans")

    checkpoint_target = _qconfig_get(checkpoint_config, "target", "val_metric")
    checkpoint_mode = _qconfig_get(checkpoint_config, "mode", "min")
    checkpoint_min_delta = _qconfig_get(checkpoint_config, "min_delta", 0.0)
    checkpoint_keep_only_latest_regular = _qconfig_get(
        checkpoint_config, "regular_latest_only", True
    )
    if not isinstance(checkpoint_keep_only_latest_regular, bool):
        raise ValueError("runner.checkpoint.regular_latest_only must be a boolean")
    checkpoint_policy = CheckpointPolicy(
        save_interval=effective_save_interval,
        restore_path=ckp_file,
        completion_save=completion_save,
        keep_only_latest_regular=checkpoint_keep_only_latest_regular,
    )
    node_aligned_output_keys = tuple(_qconfig_get(task_config, "node_aligned_output_keys", ()) or ())
    ddp_eval_dedup = _qconfig_get(runner_config, "ddp_eval_dedup", True)
    if not isinstance(ddp_eval_dedup, bool):
        raise ValueError("runner.ddp_eval_dedup must be a boolean")

    config = RunConfig(
        run_mode=resolved_run_mode,
        eval_interval=effective_eval_interval,
        completion_eval=completion_eval,
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
        device=device,
        ddp_eval_dedup=ddp_eval_dedup,
    )
    scheduler_target = _qconfig_get(
        _qconfig_get(getattr(args, "optim", None), "scheduler_params", None), "target", "val_metric"
    ) or "val_metric"
    targets = (
        checkpoint_target,
        _qconfig_get(early_stop_config, "target", "val_metric"),
        scheduler_target,
    )
    for target in targets:
        EvaluationResult.validate_target(target)

    # --- Logger ---
    owns_logger = logger is None
    if logger is None:
        logger = qLogger(save_dir, console=True)

    for log_line in epoch_suffix_logs:
        logger.info(log_line)

    metrics_logger = None
    if log_granularity and config.rank == 0:
        metrics_file = Path(save_dir) / "metrics.jsonl"
        metrics_logger = MetricsJsonlLogger(metrics_file)
    for warning_msg in boundary_policy_warnings:
        logger.warning(warning_msg)

    # --- Optimizer creation ---
    if optimizer is None:
        optimizer = prepare_optimizer(args, model, logger=logger)

    # --- Scheduler creation ---
    effective_scheduler = None
    if scheduler is not None:
        if isinstance(scheduler, qWarmupScheduler):
            effective_scheduler = scheduler
        else:
            effective_scheduler = qWarmupScheduler(
                optimizer=optimizer, warmup_steps=0, warmup_factor=1.0, main_scheduler=scheduler
            )
    else:
        optim_cfg = getattr(args, "optim", None)
        scheduler_name = getattr(optim_cfg, "scheduler", None) if optim_cfg else None
        if not scheduler_name:
            logger.info("[Scheduler] Learning rate scheduler is disabled")
            effective_scheduler = qWarmupScheduler(
                optimizer=optimizer, warmup_steps=0, warmup_factor=1.0, main_scheduler=qt.nn.DoNothing()
            )
        else:
            effective_scheduler = prepare_scheduler(args, optimizer)

    # Create managers and callback listeners
    checkpoint_manager = CheckpointManager(
        config.save_dir,
        config.rank if config.distributed else 0,
        keep_only_latest_regular=checkpoint_policy.keep_only_latest_regular,
    )
    early_stopper = EarlyStopper.from_config(early_stop_config)

    hooks = RunnerHooks()
    event_listeners = EventListenerBindings()
    observers = ObserverBindings(logger=logger)
    best_model_tracker = BestModelTracker(
        target=checkpoint_target,
        mode=checkpoint_mode,
        min_delta=checkpoint_min_delta,
    )

    for hook_name, binding_name in (
        ("on_epoch_start", "epoch_start"),
        ("on_train_batch_end", "train_boundary"),
        ("on_validation_end", "validation"),
        ("on_epoch_end", "epoch_end"),
    ):
        if task.has_implemented(hook_name):
            event_listeners.bind(binding_name, getattr(task, hook_name))

    if task.has_implemented("on_early_stop"):
        observers.bind("early_stop", task.on_early_stop, policy="settled_fatal")

    progress_tracker = ProgressTracker(
        logger, config.print_freq, render_type=config.render_type, rank=config.rank
    )
    observers.bind("epoch_started", progress_tracker.on_epoch_start)
    observers.bind("progress_tick", progress_tracker.on_progress_tick)
    observers.bind("table_update", progress_tracker.on_table_update)
    observers.bind("epoch_committed", progress_tracker.on_epoch_end)
    observers.bind("evaluation_started", progress_tracker.on_eval_start)
    observers.bind("evaluation_committed", progress_tracker.on_eval_end)

    eval_summary_observer = EvalSummaryObserver(
        logger=logger,
        target_key=checkpoint_target,
    )
    observers.bind("evaluation_committed", eval_summary_observer.on_evaluation_committed)
    if log_granularity and metrics_logger is not None:
        metrics_observer = MetricsJsonlObserver(
            logger=metrics_logger,
            run_config=config,
            log_granularity=log_granularity,
        )
        if "eval" in log_granularity:
            observers.bind("evaluation_committed", metrics_observer.on_evaluation_committed)
        if "batch" in log_granularity:
            observers.bind("train_boundary", metrics_observer.on_train_boundary)

    def _step_validation_scheduler(context: TaskValidationContext) -> None:
        if effective_scheduler is None or effective_scheduler.step_on != SCHEDULER_STEP_ON_VALID_END:
            return
        metric = context.evaluation.target_value(scheduler_target)
        if metric is None:
            logger.debug(
                "Plateau scheduler skipped: target=%r default=skip_metric_step epoch=%s step=%s.",
                scheduler_target,
                context.epoch,
                context.global_step,
            )
            return
        effective_scheduler.step_main(metrics=metric)

    early_stop_controller = EarlyStopController(
        early_stopper=early_stopper,
        target=_qconfig_get(early_stop_config, "target", "val_metric"),
        logger=logger,
    )
    event_listeners.freeze()
    observers.freeze()

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
        auto_offload=auto_offload,
        logger=logger,
        best_model_tracker=best_model_tracker,
        hooks=hooks,
        event_listeners=event_listeners,
        observers=observers,
        early_stop_controller=early_stop_controller,
        validation_scheduler_step=_step_validation_scheduler,
        node_aligned_output_keys=node_aligned_output_keys,
    )

    checkpoint_plugin = CheckpointPlugin(
        checkpoint_manager=checkpoint_manager,
        model=model,
        task=task,
        state=agent.state,
        policy=checkpoint_policy,
        optimizer=optimizer,
        scheduler=effective_scheduler,
        ema_model=ema_model,
        early_stopper=early_stopper,
        best_model_tracker=agent.best_model_tracker,
        logger=logger,
        event_logger=metrics_logger,
    )
    checkpoint_plugin.register(hooks)
    hooks.freeze()
    hooks.validate_distributed_plan(config.distributed)
    profiler = None
    terminal_event: Optional[TerminalEvent] = None
    terminal_cause: Optional[TerminalCause] = None
    primary_exception: Optional[BaseException] = None
    primary_traceback = None

    try:
        progress_tracker, profiler = _prepare_training_session(
            config=config,
            save_dir=save_dir,
            logger=logger,
            checkpoint_plugin=checkpoint_plugin,
            device=device,
            progress_tracker=progress_tracker,
        )
        agent_stop = agent.run()
        terminal_cause = "early_stop" if agent_stop == "early_stop" else "normal_finish"

    except KeyboardInterrupt:
        terminal_cause = "user_interrupt"

    except NaNDetectedError:
        terminal_cause = "nan_detected"

    except Exception as error:
        terminal_cause = "exception"
        primary_exception = error
        primary_traceback = error.__traceback__

    finally:
        _finalize_train_runner(
            logger=logger,
            progress_tracker=progress_tracker,
            profiler=profiler,
            owns_logger=owns_logger,
        )

    logger_error: Optional[BaseException] = None
    if metrics_logger is not None:
        try:
            if terminal_cause in {"normal_finish", "early_stop"}:
                metrics_logger.close()
            else:
                metrics_logger.abort()
        except BaseException as error:
            logger_error = error
            logger.error("Metrics JSONL finalization failed: %s", error, exc_info=True)

    if primary_exception is not None:
        if logger_error is not None:
            primary_exception.add_note(f"Metrics JSONL finalization failed: {logger_error!r}")
        terminal_event = _build_and_emit_terminal_event(
            logger=logger,
            terminal_cause="exception",
            state=agent.state,
            run_config=config,
            exception=primary_exception,
        )
        _close_runner_logger(logger, owns_logger)
        raise primary_exception.with_traceback(primary_traceback)

    if logger_error is not None:
        terminal_cause = "logger_failure"
    if terminal_cause is None:
        _close_runner_logger(logger, owns_logger)
        raise RuntimeError("train_runner completed without terminal cause")
    terminal_event = _build_and_emit_terminal_event(
        logger=logger,
        terminal_cause=terminal_cause,
        state=agent.state,
        run_config=config,
        exception=logger_error,
    )
    _close_runner_logger(logger, owns_logger)
    if logger_error is not None:
        raise logger_error

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
