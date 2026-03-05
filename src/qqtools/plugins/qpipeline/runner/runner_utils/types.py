import copy
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from qqtools.torch import qdist

__all__ = [
    "RunMode",
    "RunConfig",
    "RunningState",
    "LoopSignal",
    "FrozenRunningState",
    "EventType",
    "EventContext",
]


class RunMode(Enum):
    EPOCH = "epoch"
    STEP = "step"


@dataclass(frozen=True)
class RunConfig:

    # main loop
    run_mode: RunMode = RunMode.EPOCH
    eval_interval: int = 1  # depending on run_mode, this is either epoch interval or step interval
    save_interval: Optional[int] = None  # depending on run_mode, this is either epoch interval or step interval

    # boundary
    # When not specified, max_epochs should be unlimited by default so that
    # STEP mode can rely on `max_steps` as the primary stopping condition.
    max_epochs: Optional[int] = None
    max_steps: Optional[int] = None

    # optimizer
    clip_grad: Optional[float] = None

    # ddp
    distributed: bool = False
    rank: int = 0

    # I/O
    save_dir: str = "./logs"
    print_freq: int = 10
    gc_freq: int = 1000  # Frequency of garbage collection and CUDA cache clearing

    # recover
    ckp_file: Optional[str] = None
    init_file: Optional[str] = None

    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # special features
    use_profiler: bool = False
    use_ema: bool = False
    render_type: str = "auto"  # Options: "auto", "plain", "tqdm", "rich"; auto will fallback to rich->tqdm->plain
    ema_decay: float = 0.999

    # early stop
    checkpoint: Dict[str, Any] = field(default_factory=dict)
    early_stop: Dict[str, Any] = field(
        default_factory=lambda: {"target": "val_metric", "patience": 10, "mode": "min", "min_delta": 0.0}
    )

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


@dataclass
class RunningState:
    # current state
    epoch: int = 0
    global_step: int = 0
    max_epochs: int = 0
    max_steps: int = 0

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

    # time related
    epoch_start_time: float = 0.0
    step_start_time: float = 0.0
    total_train_time: float = 0.0

    # Batch tracking for resuming from mid-epoch checkpoints
    batch_idx_in_epoch: int = 0

    def update_current_metrics(self, metrics: Dict[str, Any]):
        if "train_metric" in metrics:
            self.current_train_metric = metrics["train_metric"]
        if "val_metric" in metrics:
            self.current_val_metric = metrics["val_metric"]
        if "test_metric" in metrics:
            self.current_test_metric = metrics["test_metric"]
        if "train_loss" in metrics:
            self.current_train_loss = metrics["train_loss"]

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
            "batch_idx_in_epoch": self.batch_idx_in_epoch,
        }

    def from_dict(self, state_dict: Dict[str, Any]):
        for key, value in state_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclass
class LoopSignal:
    """Mutable signal box consumed by the loop after listeners finish."""

    should_stop: bool = False
    stop_message: Optional[str] = None
    pending_checkpoint_types: List[Literal["best", "regular"]] = field(default_factory=list)

    def request_checkpoint(self, checkpoint_type: Literal["best", "regular"]) -> None:
        if checkpoint_type not in ("best", "regular"):
            raise ValueError(f"Unsupported checkpoint type: {checkpoint_type}")
        self.pending_checkpoint_types.append(checkpoint_type)

    def synchronize_stop(self, device: torch.device, distributed: bool) -> None:
        """Synchronize the stop signal across all DDP ranks."""
        if not distributed:
            return

        # Local decision
        local_flag = 1 if self.should_stop else 0
        # Reduce MAX: if anyone wants to stop, everyone stops
        reduced_flag = qdist.all_reduce(local_flag, device=device, reduceOp="max")

        if reduced_flag > 0:
            if not self.should_stop:
                self.stop_message = "Stopping triggered by another rank."
            self.should_stop = True


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


class EventType(Enum):
    ON_EPOCH_START = "on_epoch_start"
    ON_EPOCH_END = "on_epoch_end"
    ON_BATCH_START = "on_batch_start"
    ON_BATCH_END = "on_batch_end"
    ON_PROGRESS_TICK = "on_progress_tick"
    ON_TABLE_UPDATE = "on_table_update"
    ON_TRAIN_BATCH_END = "on_train_batch_end"
    ON_EVAL_START = "on_eval_start"
    ON_EVAL_END = "on_eval_end"
    ON_VALIDATION_END = "on_validation_end"
    ON_CHECKPOINT_REQUEST = "on_checkpoint_request"
    ON_CHECKPOINT_SAVE = "on_checkpoint_save"
    ON_EARLY_STOP = "on_early_stop"


@dataclass
class EventContext:
    state: Union[RunningState, FrozenRunningState]
    signal: Optional[LoopSignal] = None
    batch_idx: Optional[int] = None
    total_batches: Optional[int] = None
    batch_metrics: Optional[Dict[str, Any]] = None
    avg_bank: Optional[Dict[str, Any]] = None
    lr: Optional[float] = None
    eval_results: Optional[Dict[str, Any]] = None
    checkpoint_path: Optional[str] = None
    checkpoint_type: Optional[str] = None
    stage: Optional[str] = None  # Current stage: "training", "validation", "testing"
    previous_best: Optional[Dict[str, Any]] = None
    is_best: Optional[bool] = None
    best_model_tracker: Optional[Any] = None
    # Provide run limits on the context so listeners can access them without
    # coupling to RunningState (which no longer carries these fields).
    max_epochs: Optional[int] = None
    max_steps: Optional[int] = None
