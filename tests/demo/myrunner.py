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
from .ckp_manager import CheckpointListener, CheckpointManager
from .earlystop import EarlyStopListener, EarlyStopper
from .eval_formatter import EvalSummaryListener
from .progress import ProgressTracker
from .tensorbank import TensorBank
from .types import EventContext, FrozenRunningState, LoopSignal, RunConfig, RunMode, RunningState

__all__ = ["train_runner", "infer_runner"]


# ============================================================================
# Core Engine Add-ons (Listeners & Trackers)
# ============================================================================
class SheetLoggerListener:
    """A listener that writes metrics to a sheet logger (e.g., CSV) based on training events."""
    def __init__(self, logger: qLogger, run_config: RunConfig, log_granularity: List[str]):
        self.logger = logger
        self.config = run_config
        self.log_granularity = log_granularity

    def _prepare_data(self, context: EventContext, mode: str) -> Dict[str, Any]:
        state = context.state
        data = {"epoch": state.epoch, "global_step": state.global_step}
        source_metrics = {}

        if mode == "eval":
            if getattr(context, "eval_results", None):
                source_metrics.update(context.eval_results)
            else:
                for k in["current_val_metric", "current_test_metric", "current_train_metric", "current_train_loss"]:
                    v = getattr(state, k, None)
                    if v is not None:
                        source_metrics[k.replace("current_", "")] = v
                if not source_metrics:
                    self.logger.warning("No eval metrics found in context or state; writing empty row.")
        elif mode == "batch":
            if getattr(context, "batch_metrics", None):
                source_metrics.update(context.batch_metrics)

        if self.logger.columns:
            for key in self.logger.columns:
                if key in data: continue
                value = source_metrics.get(key)
                if value is None and mode == "batch" and key.startswith("train_"):
                    value = source_metrics.get(key[len("train_") :])
                data[key] = value
        else:
            data.update(source_metrics)
        return data

    def on_eval_end(self, context: EventContext):
        data = self._prepare_data(context, mode="eval")
        self.logger.write(data)

    def on_train_batch_end(self, context: EventContext):
        if "eval" in self.log_granularity:
            is_epoch_end = (context.batch_idx is not None and context.total_batches is not None and context.batch_idx == context.total_batches - 1)
            if _is_periodic_trigger(self.config.run_mode, self.config.eval_interval, context.state.global_step, context.state.epoch, is_epoch_end):
                return
        data = self._prepare_data(context, mode="batch")
        self.logger.write(data)


class SystemMaintenanceListener:
    """Handles GC and memory cleanup periodically without polluting the core loop."""
    def __init__(self, gc_freq: int):
        self.gc_freq = gc_freq

    def on_train_batch_end(self, context: EventContext):
        if context.state.global_step > 0 and context.state.global_step % self.gc_freq == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class WarmupSchedulerListener:
    """Delegates scheduler warmup steps as a side-effect hook."""
    def __init__(self, scheduler: Optional[qWarmupScheduler]):
        self.scheduler = scheduler

    def on_train_batch_end(self, context: EventContext):
        if self.scheduler is not None:
            self.scheduler.step_warmup()


class TaskMetricTracker:
    """Encapsulates metric accumulation (AvgBank) and caching (TensorBank)."""
    def __init__(self, task: qTaskBase, distributed: bool, device: torch.device, logger: Optional[qLogger] = None):
        self.task = task
        self.distributed = distributed
        self.device = device
        self.logger = logger

        has_cache = task.has_implemented("batch_cache")
        has_epoch = task.has_implemented("epoch_metric")
        self.use_tensor_bank = has_cache and has_epoch

        self.avg_bank = AvgBank()
        self.tensor_bank = TensorBank(logger=logger) if self.use_tensor_bank else None

    def update_on_batch(self, raw_metrics: dict, out: Any, batch_data: Any):
        for key, value in raw_metrics.items():
            cnt = 1
            if isinstance(value, tuple):
                if len(value) >= 2: val, cnt = value[0], value[1]
                elif len(value) == 1: val = value[0]
            else:
                val = value

            if torch.is_tensor(val): val = val.item()
            if isinstance(val, (int, float)):
                self.avg_bank.add(key, float(val), cnt)

        if self.use_tensor_bank:
            self.tensor_bank.add(self.task.batch_cache(out, batch_data))

    def compute_metrics(self) -> dict:
        metrics = self.avg_bank.gather_average(self.distributed)
        if self.use_tensor_bank:
            gathered = self.tensor_bank.gather(self.distributed, self.device)
            task_epoch_metrics = self.task.epoch_metric(gathered)
            if task_epoch_metrics:
                metrics.update(task_epoch_metrics)
        self.reset()
        return metrics

    def reset(self):
        self.avg_bank = AvgBank()
        if self.tensor_bank: self.tensor_bank.reset()


class EMAOffloadContext:
    """Context manager that temporarily moves models for EMA evaluation."""
    def __init__(self, main_model: nn.Module, ema_model: Optional[qEMA], device: torch.device, use_ema: bool, use_offload: bool, logger: Optional[qLogger] = None):
        self.main_model = main_model
        self.ema_model = ema_model
        self.device = device
        self.use_ema = use_ema
        self.use_offload = use_offload
        self.logger = logger
        self.ema_original_device = None
        self.offloaded = False

    def __enter__(self) -> nn.Module:
        if not self.use_ema or self.ema_model is None:
            return self.main_model

        if self.use_offload:
            if self.logger: self.logger.debug("Offloading main model to CPU for EMA eval.")
            self.main_model.cpu()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            self.offloaded = True

        try:
            self.ema_original_device = next(self.ema_model.parameters()).device
            if self.ema_original_device != self.device:
                self.ema_model.to(self.device)
        except Exception:
            self.ema_original_device = None

        return self.ema_model

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.use_ema and self.ema_model is not None:
            if self.ema_original_device and self.ema_original_device != self.device:
                self.ema_model.to(self.ema_original_device)
            if self.offloaded:
                if self.logger: self.logger.debug("Restoring main model to original device.")
                self.main_model.to(self.device)


# ============================================================================
# Pure Functions
# ============================================================================
def move_batch_to_device(batch_data, device: torch.device):
    if batch_data is None: return batch_data
    if hasattr(batch_data, "to"): return batch_data.to(device)
    elif isinstance(batch_data, dict): return qt.qDict({k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_data.items()})
    elif isinstance(batch_data, (tuple, list)): return type(batch_data)(v.to(device) if isinstance(v, torch.Tensor) else v for v in batch_data)
    else: return batch_data

def _getattr_or_default(obj: Any, key: str, default: Any = None) -> Any:
    value = getattr(obj, key, None)
    return value if value is not None else (default() if callable(default) else default)

def _is_periodic_trigger(run_mode: RunMode, interval: Optional[int], global_step: int, epoch: int, is_epoch_end: bool) -> bool:
    if interval is None or interval <= 0: return False
    if run_mode == RunMode.STEP: return (global_step + 1) % interval == 0
    return is_epoch_end and (epoch + 1) % interval == 0

def _resolve_train_runner_policy(run_mode, max_epochs, max_steps, eval_interval, save_interval):
    def _ensure_positive_int(val, name):
        if not isinstance(val, int) or val < 1 or isinstance(val, bool): raise ValueError(f"{name} must be >=1")
        return val

    run_mode = RunMode(run_mode)
    eff_eval = _ensure_positive_int(eval_interval or 1, "eval_interval")
    eff_save = _ensure_positive_int(save_interval or eff_eval, "save_interval")

    warnings_list =[]
    if run_mode == RunMode.EPOCH:
        if max_epochs is None: raise ValueError("max_epochs required for epoch mode.")
        if max_steps is not None: warnings_list.append(f"max_steps={max_steps} ignored in EPOCH mode.")
        return run_mode, eff_eval, eff_save, _ensure_positive_int(max_epochs, "max_epochs"), None, warnings_list
    else:
        if max_steps is None: raise ValueError("max_steps required for step mode.")
        if max_epochs is not None: warnings_list.append(f"max_epochs={max_epochs} ignored in STEP mode.")
        return run_mode, eff_eval, eff_save, None, _ensure_positive_int(max_steps, "max_steps"), warnings_list


# ============================================================================
# Core Running Agent
# ============================================================================
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
        self._train_iter = None

        # Delegate metric accumulation to Trackers
        self.train_tracker = TaskMetricTracker(self.task, self.config.distributed, self.device, self.logger)
        self.interval_tracker = TaskMetricTracker(self.task, self.config.distributed, self.device, self.logger)

        # Listeners initialization
        default_listeners = {
            "on_epoch_start": [], "on_epoch_end": [], "on_batch_start":[], "on_batch_end": [],
            "on_progress_tick": [], "on_train_batch_end":[], "on_table_update": [],
            "on_eval_start": [], "on_eval_end":[], "on_validation_end": [],
            "on_checkpoint_request":[], "on_checkpoint_save": [], "on_early_stop":[],
        }
        self.listeners = {ev: list(cbs) for ev, cbs in default_listeners.items()}
        if listeners:
            for ev, cbs in listeners.items():
                self.listeners.setdefault(ev,[]).extend(list(cbs))

        # Check for model offloading
        self._use_model_offload = False
        if self.device.type == "cuda" and self.ema_model is not None:
            from ..entry_utils.info import get_model_size_bytes
            if get_model_size_bytes(self.model) > (torch.cuda.get_device_properties(self.device).total_memory * 0.5):
                self._use_model_offload = True
                self.logger.info("Model size > 50% of GPU capacity. Enabling mutual offloading.")

    def add_listener(self, event: str, listener: Callable):
        if event in self.listeners:
            self.listeners[event].append(listener)
        else:
            warnings.warn(f"Unknown event: {event}")

    def _trigger(self, event: str, context: EventContext, snapshot: bool = True) -> EventContext:
        ctx = copy.copy(context) if snapshot else context
        ctx.state = FrozenRunningState.from_state(context.state)
        ctx.max_epochs = self.config.max_epochs
        ctx.max_steps = self.config.max_steps

        for listener in self.listeners.get(event,[]):
            try:
                listener(ctx)
            except Exception as e:
                lname = getattr(listener, "__name__", listener.__class__.__name__)
                self.logger.error(f"Listener error '{event}' in {lname}: {type(e).__name__}: {e}")
                raise
        return ctx

    def _prepare_batch(self, batch_data):
        return move_batch_to_device(batch_data, self.device)

    def _forward_batch(self, model: nn.Module, batch_data):
        batch_data = self.task.pre_batch_forward(batch_data)
        out = self.task.batch_forward(model, batch_data)
        return self.task.post_batch_forward(out, batch_data), batch_data

    @staticmethod
    def _scalarize_batch_metrics(batch_metrics: Dict[str, Any]) -> Dict[str, float]:
        scalars = {}
        for k, v in batch_metrics.items():
            val = v[0] if isinstance(v, tuple) and len(v) > 0 else v
            if torch.is_tensor(val) and val.numel() == 1: val = val.item()
            if isinstance(val, (int, float)): scalars[k] = float(val)
        return scalars

    def _emit_batch_events(self, **kwargs) -> None:
        """Helper to fire batch-level granular events."""
        ctx = EventContext(
            state=self.state, batch_idx=kwargs["batch_idx"], total_batches=kwargs["total_batches"],
            batch_metrics=kwargs["batch_metrics"], avg_bank=kwargs["batch_avg_metrics"],
            lr=kwargs["lr"], stage=kwargs["stage"]
        )
        self._trigger("on_batch_end", context=ctx)
        if kwargs.get("emit_train_batch_end"): self._trigger("on_train_batch_end", context=ctx)

        p_ctx = EventContext(
            state=self.state, batch_idx=kwargs["batch_idx"], total_batches=kwargs["total_batches"],
            batch_metrics=kwargs["scalar_batch_metrics"], avg_bank=kwargs["progress_avg_metrics"],
            lr=kwargs["lr"], stage=kwargs["stage"]
        )
        self._trigger("on_progress_tick", context=p_ctx)
        if kwargs.get("emit_table_update"): self._trigger("on_table_update", context=p_ctx)

    def _get_next_batch(self) -> Tuple[Any, bool]:
        if self._train_iter is None:
            self._train_iter = iter(self.train_loader)
            if self.config.distributed and hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(self.state.epoch)
            self._trigger("on_epoch_start", context=EventContext(state=self.state, total_batches=len(self.train_loader), stage="train"))

            # Skip iterations if resuming
            for _ in range(self.state.batch_idx_in_epoch):
                try: next(self._train_iter)
                except StopIteration:
                    self.state.batch_idx_in_epoch = 0; break

        try:
            batch_data = next(self._train_iter)
            self.state.batch_idx_in_epoch += 1
            return batch_data, (self.state.batch_idx_in_epoch == len(self.train_loader))
        except StopIteration:
            return None, True

    def _train_one_batch(self, batch_data) -> None:
        """Executes full lifecycle for a single training batch."""
        batch_start_time = time.time()
        batch_idx, total_batches = self.state.batch_idx_in_epoch - 1, len(self.train_loader)

        self._trigger("on_batch_start", context=EventContext(state=self.state, batch_idx=batch_idx, total_batches=total_batches))

        # Forward & Loss
        batch_data = self._prepare_batch(batch_data)
        out, batch_data = self._forward_batch(self.model, batch_data)
        raw_metrics = self.task.batch_metric(out, batch_data)
        losses = self.task.batch_loss(out, batch_data, self.loss_fn)
        loss_tensor, loss_cnt = losses.get("loss", (None, 1))

        if loss_tensor is not None:
            self.optimizer.zero_grad()
            loss_tensor.backward()
            if self.config.clip_grad: clip_grad_norm_(self.model.parameters(), self.config.clip_grad)
            self.optimizer.step()
            if self.ema_model: self.ema_model.update()
            raw_metrics["loss"] = (loss_tensor.item(), loss_cnt)

        # Tracking & Timing
        self.train_tracker.update_on_batch(raw_metrics, out, batch_data)
        self.interval_tracker.update_on_batch(raw_metrics, out, batch_data)
        batch_time = time.time() - batch_start_time
        self.state.total_train_time += batch_time

        scalar_metrics = self._scalarize_batch_metrics(raw_metrics)
        scalar_metrics["batch_time"] = batch_time

        # Events
        is_update_tick = (batch_idx + 1) % self.config.print_freq == 0 or (batch_idx + 1) == total_batches
        self._emit_batch_events(
            batch_idx=batch_idx, total_batches=total_batches,
            batch_metrics=raw_metrics, scalar_batch_metrics=scalar_metrics, stage="train",
            lr=self.optimizer.param_groups[0]["lr"] if self.optimizer else None,
            progress_avg_metrics=self.train_tracker.avg_bank.to_dict(self.config.distributed),
            emit_train_batch_end=True, emit_table_update=is_update_tick,
        )

    def _evaluate_loaders(self, model: nn.Module) -> Dict[str, Any]:
        results = {}
        for loader, stage in [(self.val_loader, "val"), (self.test_loader, "test")]:
            if not loader: continue

            was_training, model.eval() = model.training, model.eval()
            tracker = TaskMetricTracker(self.task, self.config.distributed, self.device, self.logger)

            with torch.no_grad():
                for batch_idx, batch_data in enumerate(loader):
                    batch_data = self._prepare_batch(batch_data)
                    out, batch_data = self._forward_batch(model, batch_data)
                    raw_metrics = self.task.batch_metric(out, batch_data)
                    tracker.update_on_batch(raw_metrics, out, batch_data)

                    scalars = self._scalarize_batch_metrics(raw_metrics)
                    self._emit_batch_events(
                        batch_idx=batch_idx, total_batches=len(loader), batch_metrics=scalars,
                        scalar_batch_metrics=scalars, stage=stage,
                        batch_avg_metrics=tracker.avg_bank.to_dict(self.config.distributed),
                        progress_avg_metrics=tracker.avg_bank.to_dict(self.config.distributed),
                    )

            avg_metrics = tracker.compute_metrics()
            stage_results = {f"{stage}_{k}": v for k, v in avg_metrics.items()}
            stage_results[f"{stage}_metric"] = self.task.post_metrics_to_value(avg_metrics)
            results.update(stage_results)
            model.train(was_training)

        return results

    def evaluate_all_models(self) -> Dict[str, Any]:
        """Core Evaluation logic covering main model and EMA."""
        eval_results = {}
        with EMAOffloadContext(self.model, self.ema_model, self.device, False, self._use_model_offload, self.logger) as m:
            eval_results.update(self._evaluate_loaders(m))

        if self.ema_model is not None:
            with EMAOffloadContext(self.model, self.ema_model, self.device, True, self._use_model_offload, self.logger) as m:
                eval_results.update({f"ema_{k}": v for k, v in self._evaluate_loaders(m).items()})
        return eval_results

    def _run_evaluation_and_update(self, signal: LoopSignal) -> None:
        total_batches = sum(len(l) for l in (self.val_loader, self.test_loader) if l)
        self._trigger("on_eval_start", context=EventContext(state=self.state, stage="eval", total_batches=total_batches, signal=signal))

        eval_results = self.evaluate_all_models()
        interval_metrics = self.interval_tracker.compute_metrics()
        if interval_metrics:
            eval_results.update({f"train_{k}": v for k, v in interval_metrics.items()})
            eval_results["train_metric"] = self.task.post_metrics_to_value(interval_metrics)

        self.state.update_current_metrics(eval_results)
        self._trigger("on_eval_end", context=EventContext(state=self.state, stage="eval", eval_results=eval_results, signal=signal))

        # Best Model Calculation
        is_best, prev_best = False, None
        if self.best_model_tracker and self.best_model_tracker.target:
            prev_best = {"metric": self.best_model_tracker.best_metric, "epoch": self.best_model_tracker.best_epoch, "step": getattr(self.best_model_tracker, "best_step", 0)}
            target_val = eval_results.get(self.best_model_tracker.target)
            if target_val is not None:
                try:
                    is_best = self.best_model_tracker.update(qt.ensure_scala(target_val), self.state.epoch, self.state.global_step)
                    if is_best:
                        self.state.best_epoch, self.state.best_step = self.state.epoch, self.state.global_step
                        self.state.best_monitored_key, self.state.best_monitored_metric = self.best_model_tracker.target, qt.ensure_scala(target_val)
                        self.state.best_model_metrics_snapshot = copy.deepcopy(eval_results)
                        signal.request_checkpoint("best")
                except Exception: pass

        validation_context = EventContext(
            state=self.state, stage="eval", eval_results=eval_results, previous_best=prev_best,
            is_best=is_best, best_model_tracker=self.best_model_tracker, signal=signal,
        )
        self._trigger("on_validation_end", context=validation_context, snapshot=False)

    def _flush_checkpoint_requests(self, signal: LoopSignal) -> None:
        while signal.pending_checkpoint_types:
            ckp_type = signal.pending_checkpoint_types.pop(0)
            req_ctx = EventContext(state=self.state, checkpoint_type=ckp_type, signal=signal)
            self._trigger("on_checkpoint_request", context=req_ctx, snapshot=False)

            if req_ctx.checkpoint_path:
                if ckp_type == "best": self.state.best_ckp_file = Path(req_ctx.checkpoint_path).name
                self._trigger("on_checkpoint_save", context=EventContext(state=self.state, checkpoint_type=ckp_type, checkpoint_path=req_ctx.checkpoint_path, signal=signal))

    def _handle_periodic_events(self, is_epoch_end: bool) -> bool:
        signal = LoopSignal()
        if _is_periodic_trigger(self.config.run_mode, self.config.eval_interval, self.state.global_step, self.state.epoch, is_epoch_end):
            self._run_evaluation_and_update(signal)

        if _is_periodic_trigger(self.config.run_mode, self.config.save_interval, self.state.global_step, self.state.epoch, is_epoch_end):
            signal.request_checkpoint("regular")

        self._flush_checkpoint_requests(signal)

        if signal.should_stop:
            self.logger.info(signal.stop_message or "Early stopping triggered.")
            self._trigger("on_early_stop", context=EventContext(state=self.state, signal=signal))
            return True
        return False

    def _reached_limits(self) -> bool:
        m_step = self.config.max_steps if self.config.max_steps is not None else float("inf")
        m_epoch = self.config.max_epochs if self.config.max_epochs is not None else float("inf")
        if self.state.global_step >= m_step or self.state.epoch >= m_epoch:
            self.logger.info(f"Training stopping at epoch {self.state.epoch}, step {self.state.global_step}.")
            return True
        return False

    def _training_loop(self) -> bool:
        self._train_iter = None
        while True:
            if self._reached_limits(): return False

            try:
                batch_data, is_epoch_end = self._get_next_batch()

                if batch_data is None:
                    epoch_metrics = self.train_tracker.compute_metrics()
                    self.state.update_current_metrics(epoch_metrics)
                    self._trigger("on_epoch_end", context=EventContext(state=self.state))
                    self.state.epoch, self.state.batch_idx_in_epoch = self.state.epoch + 1, 0
                    self._train_iter = None
                    continue

                self._train_one_batch(batch_data)

                if self._handle_periodic_events(is_epoch_end): return True
                self.state.global_step += 1

            except KeyboardInterrupt:
                self.logger.info("Training interrupted by user.")
                return True
            except StopIteration:
                self.logger.warning("Unexpected StopIteration in loop. Forcing epoch end boundary.")
                if self._handle_periodic_events(is_epoch_end=True): return True
                self._train_iter = None

    def run(self) -> bool:
        self.logger.info(f"Starting training (mode={self.config.run_mode.value}, eval={self.config.eval_interval}, max_e={self.config.max_epochs}, max_s={self.config.max_steps})")
        return self._training_loop()


# ============================================================================
# Main Orchestration Layers
# ============================================================================
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
    """Unified training runner."""
    if args is None: raise ValueError("The 'args' parameter is required.")

    device = _getattr_or_default(args, "device", lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    rank = _getattr_or_default(args, "rank", 0)
    runner_cfg = _getattr_or_default(args, "runner", {})
    chkpt_cfg = _getattr_or_default(runner_cfg, "checkpoint", dict)
    early_stop_cfg = _getattr_or_default(runner_cfg, "early_stop", dict)

    r_mode, e_eval, e_save, e_max_e, e_max_s, warnings_list = _resolve_train_runner_policy(run_mode, max_epochs, max_steps, eval_interval, save_interval)

    if not chkpt_cfg:
        chkpt_cfg.update({"target": early_stop_cfg.get("target", "val_metric"), "mode": early_stop_cfg.get("mode", "min"), "min_delta": early_stop_cfg.get("min_delta", 0.0)})
    early_stop_cfg.setdefault("target", "val_metric")
    early_stop_cfg.setdefault("mode", "min")

    config = RunConfig(
        run_mode=r_mode, eval_interval=e_eval, save_interval=e_save, max_epochs=e_max_e, max_steps=e_max_s,
        clip_grad=clip_grad, distributed=distributed, rank=rank, save_dir=save_dir, print_freq=print_freq,
        use_profiler=use_profiler, use_ema=ema_model is not None, render_type=_getattr_or_default(args, "render_type", "auto"),
        ckp_file=_getattr_or_default(args, "ckp_file"), init_file=_getattr_or_default(args, "init_file"),
        device=device, checkpoint=chkpt_cfg, early_stop=early_stop_cfg,
    )

    log_keys =["epoch", "global_step", "train_metric", "val_metric", "test_metric", "train_loss"] + (extra_log_keys or[])
    logger = qLogger(save_dir, columns=log_keys, console=True)
    for msg in warnings_list: logger.warning(msg)

    eff_scheduler = scheduler if isinstance(scheduler, qWarmupScheduler) else qWarmupScheduler(optimizer, 0, 1.0, scheduler or qt.nn.DoNothing())

    agent = RunningAgent(model, task, loss_fn, optimizer, eff_scheduler, config, device, ema_model, logger)

    # Attach Core Framework Listeners
    checkpoint_manager = CheckpointManager(config.save_dir, config.rank)
    early_stopper = EarlyStopper.from_config(config.early_stop)

    agent.add_listener("on_train_batch_end", SystemMaintenanceListener(config.gc_freq).on_train_batch_end)
    agent.add_listener("on_train_batch_end", WarmupSchedulerListener(eff_scheduler).on_train_batch_end)
    agent.add_listener("on_validation_end", lambda ctx: eff_scheduler.step_main(metrics=(ctx.eval_results or {}).get("val_metric")))
    agent.add_listener("on_validation_end", EvalSummaryListener(logger, config.checkpoint.get("target", "val_metric"), config.checkpoint.get("mode", "min")).on_validation_end)
    agent.add_listener("on_validation_end", EarlyStopListener(early_stopper, config.early_stop.get("target", "val_metric"), logger).on_validation_end)
    agent.add_listener("on_checkpoint_request", CheckpointListener(checkpoint_manager, model, task, optimizer, eff_scheduler, ema_model, early_stopper, agent.best_model_tracker).on_checkpoint_request)

    prog_tracker = ProgressTracker(logger, config.print_freq, render_type=config.render_type)
    for ev in["on_epoch_start", "on_progress_tick", "on_table_update", "on_epoch_end", "on_eval_start", "on_eval_end"]:
        agent.add_listener(ev, getattr(prog_tracker, ev))

    if log_granularity:
        sheet_logger = SheetLoggerListener(logger, config, log_granularity)
        if "eval" in log_granularity: agent.add_listener("on_eval_end", sheet_logger.on_eval_end)
        if "batch" in log_granularity: agent.add_listener("on_train_batch_end", sheet_logger.on_train_batch_end)

    if config.ckp_file and Path(config.ckp_file).exists():
        checkpoint_manager.load(config.ckp_file, device, model, task, optimizer, eff_scheduler, ema_model, agent.state, early_stopper, agent.best_model_tracker)
        logger.info(f"Loaded checkpoint: {config.ckp_file}")

    if config.max_epochs is None and config.max_steps is None:
        raise ValueError("Either max_epochs or max_steps must be specified.")

    profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, on_trace_ready=torch.profiler.tensorboard_trace_handler(str(Path(save_dir) / "profiler"))) if config.use_profiler else None
    if profiler: profiler.start()

    try:
        if agent.run(): logger.info(f"Early stop at ep {agent.state.epoch}, step {agent.state.global_step}")
        else: logger.info(f"Training completed: ep {agent.state.epoch}, step {agent.state.global_step}")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        if profiler: profiler.stop()
        try: prog_tracker.on_run_end()
        except Exception as e: logger.debug(f"ProgressTracker cleanup failed: {e}", exc_info=True)
        logger.close(); logging.shutdown()

    return {
        "best_epoch": agent.state.best_epoch, "best_step": getattr(agent.state, "best_step", 0),
        "best_monitored_key": agent.state.best_monitored_key, "best_monitored_metric": agent.state.best_monitored_metric,
        "best_model_metrics_snapshot": agent.state.best_model_metrics_snapshot,
        "final_epoch": agent.state.epoch, "final_step": agent.state.global_step, "total_train_time": agent.state.total_train_time,
    }

