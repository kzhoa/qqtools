import copy
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Literal, NotRequired, Optional, TypedDict, Union

import torch

TerminalReason = Literal[
    "max_steps", "max_epochs", "early_stop", "user_interrupt", "oom", "exception",
    "nan_detected", "logger_failure",
]
EpochResultMetricSource = Literal["current_eval", "latest_eval_reuse", "missing"]

__all__ = [
    "RunMode",
    "RunConfig",
    "RunningState",
    "FrozenRunningState",
    "TerminalReason",
    "EpochResultMetricSource",
    "TerminalEvent",
    "TrainRunnerResult",
]


class RunMode(Enum):
    EPOCH = "epoch"
    STEP = "step"


class TerminalEvent(TypedDict):
    status: Literal["finished", "stopped", "failed"]
    reason: TerminalReason
    text: str
    epoch: int
    step: int
    exception_type: NotRequired[str]


class TrainRunnerResult(TypedDict):
    best_epoch: int
    best_step: int
    best_monitored_key: Optional[str]
    best_monitored_metric: Optional[float]
    best_model_metrics_snapshot: Dict[str, Any]
    final_epoch: int
    final_step: int
    total_train_time: float
    early_stopped: bool
    terminal_event: TerminalEvent


@dataclass(frozen=True)
class RunConfig:

    # main loop
    run_mode: RunMode = RunMode.EPOCH
    eval_interval: int = 1  # depending on run_mode, this is either epoch interval or step interval
    completion_eval: bool = False

    # boundary
    # When not specified, max_epochs should be unlimited by default so that
    # STEP mode can rely on `max_steps` as the primary stopping condition.
    max_epochs: Optional[int] = None
    max_steps: Optional[int] = None

    # optimizer
    clip_grad: Optional[float] = None
    accum_grad: Optional[int] = None

    # ddp
    distributed: bool = False
    rank: int = 0

    # I/O
    save_dir: str = "./logs"
    print_freq: int = 10
    gc_freq: int = 1000  # Frequency of garbage collection and CUDA cache clearing

    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # special features
    use_profiler: bool = False
    use_ema: bool = False
    render_type: str = "auto"  # Options: "auto", "plain", "tqdm", "rich"; auto will fallback to rich->tqdm->plain
    ema_decay: float = 0.999

    ddp_eval_dedup: bool = True

    def __post_init__(self):
        if isinstance(self.run_mode, str):
            object.__setattr__(self, "run_mode", RunMode(self.run_mode))
        if isinstance(self.device, str):
            object.__setattr__(self, "device", torch.device(self.device))
        if not isinstance(self.print_freq, int) or self.print_freq <= 0:
            raise ValueError("print_freq must be a positive integer")

        # Validate eval_interval
        if not isinstance(self.eval_interval, int) or self.eval_interval < 1:
            raise ValueError("eval_interval must be a positive integer (>=1)")
        if not isinstance(self.completion_eval, bool):
            raise ValueError("completion_eval must be a boolean")
        if self.accum_grad is not None:
            if isinstance(self.accum_grad, bool) or not isinstance(self.accum_grad, int):
                raise ValueError("accum_grad must be an integer when specified")
            if self.accum_grad < 1:
                raise ValueError("accum_grad must be a positive integer (>=1) when specified")


@dataclass
class RunningState:
    # current state
    epoch: int = 0
    global_step: int = 0

    # best state
    best_epoch: int = 0
    best_step: int = 0
    best_monitored_key: Optional[str] = None
    best_monitored_metric: Optional[float] = None
    best_model_metrics_snapshot: Dict[str, Any] = field(default_factory=dict)
    best_ckp_file: Optional[str] = None

    # current metrics
    current_train_loss: Optional[float] = None
    current_train_metric: Optional[float] = None
    current_val_metric: Optional[float] = None
    current_test_metric: Optional[float] = None
    latest_val_metric: Optional[float] = None
    latest_test_metric: Optional[float] = None
    epoch_end_eval_triggered: bool = False
    epoch_result_val_metric_source: EpochResultMetricSource = "missing"
    epoch_result_test_metric_source: EpochResultMetricSource = "missing"

    # time related
    epoch_start_time: float = 0.0
    step_start_time: float = 0.0
    total_train_time: float = 0.0

    # Batch tracking for resuming from mid-epoch checkpoints
    batch_idx_in_epoch: int = 0

    def update_current_metrics(self, metrics: Dict[str, Any], *, is_evaluation_boundary: bool = False):
        if "train_metric" in metrics:
            self.current_train_metric = metrics["train_metric"]
        if is_evaluation_boundary:
            self.current_val_metric = metrics.get("val_metric")
            self.current_test_metric = metrics.get("test_metric")
            if "val_metric" in metrics:
                self.latest_val_metric = metrics["val_metric"]
            if "test_metric" in metrics:
                self.latest_test_metric = metrics["test_metric"]
        else:
            if "val_metric" in metrics:
                self.current_val_metric = metrics["val_metric"]
            if "test_metric" in metrics:
                self.current_test_metric = metrics["test_metric"]
        if "train_loss" in metrics:
            self.current_train_loss = metrics["train_loss"]

    def mark_epoch_end_eval_trigger(self, *, eval_triggered: bool) -> None:
        self.epoch_end_eval_triggered = eval_triggered

    def refresh_epoch_result_metric_sources(self) -> None:
        self.epoch_result_val_metric_source = self._resolve_epoch_result_metric_source(self.current_val_metric)
        self.epoch_result_test_metric_source = self._resolve_epoch_result_metric_source(self.current_test_metric)

    def _resolve_epoch_result_metric_source(self, metric_value: Optional[float]) -> EpochResultMetricSource:
        if metric_value is None:
            return "missing"
        if self.epoch_end_eval_triggered:
            return "current_eval"
        return "latest_eval_reuse"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_epoch": self.best_epoch,
            "best_step": self.best_step,
            "best_monitored_key": self.best_monitored_key,
            "best_monitored_metric": self.best_monitored_metric,
            "best_model_metrics_snapshot": self.best_model_metrics_snapshot,
            "best_ckp_file": self.best_ckp_file,
            "current_val_metric": self.current_val_metric,
            "current_test_metric": self.current_test_metric,
            "latest_val_metric": self.latest_val_metric,
            "latest_test_metric": self.latest_test_metric,
            "batch_idx_in_epoch": self.batch_idx_in_epoch,
        }

    def from_dict(self, state_dict: Dict[str, Any]):
        for key, value in state_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, MappingProxyType):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    if isinstance(value, frozenset):
        return {_deep_thaw(item) for item in value}
    return copy.deepcopy(value)


class FrozenRunningState:
    """Read-only state snapshot passed to listeners."""

    def __init__(self, state: RunningState):
        snapshot = {key: _deep_freeze(copy.deepcopy(value)) for key, value in vars(state).items()}
        object.__setattr__(self, "_snapshot", snapshot)

    @classmethod
    def from_state(cls, state: Union["RunningState", "FrozenRunningState"]) -> "FrozenRunningState":
        if isinstance(state, FrozenRunningState):
            return state
        return cls(state)

    def __getattr__(self, item: str) -> Any:
        snapshot = object.__getattribute__(self, "_snapshot")
        if item in snapshot:
            return snapshot[item]
        raise AttributeError(item)

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError("FrozenRunningState is read-only")

    def to_running_state(self) -> RunningState:
        state = RunningState()
        thawed = {key: _deep_thaw(value) for key, value in self._snapshot.items()}
        state.from_dict(thawed)
        return state
