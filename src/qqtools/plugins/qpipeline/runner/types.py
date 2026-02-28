from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

__all__ = ["RunMode", "RunConfig", "RunningState", "EventType", "EventContext"]


class RunMode(Enum):
    EPOCH = "epoch"
    STEP = "step"


@dataclass(frozen=True)
class RunConfig:

    # main loop
    run_mode: RunMode = RunMode.EPOCH
    eval_interval: int = 1  # depending on run_mode, this is either epoch interval or step interval
    save_interval: Optional[int] = None  # Regular checkpoint saving interval (steps), None means no regular saving

    # boundary
    max_epochs: Optional[int] = 1
    max_steps: Optional[int] = None

    # optimizer
    clip_grad: Optional[float] = None

    # ddp
    distributed: bool = False
    rank: int = 0

    # I/O
    save_dir: str = "./logs"
    print_freq: int = 10

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


class RunningState:

    def __init__(self):
        # current state
        self.epoch: int = 0
        self.global_step: int = 0
        self.max_epochs: int = 0
        self.max_steps: int = 0

        # best state
        self.best_epoch: int = 0
        self.best_step: int = 0
        self.best_train_metric: Optional[float] = None
        self.best_val_metric: Optional[float] = None
        self.best_test_metric: Optional[float] = None
        self.best_ckp_file: Optional[str] = None

        # curretn metrics
        self.current_train_loss: Optional[float] = None
        self.current_train_metric: Optional[float] = None
        self.current_val_metric: Optional[float] = None
        self.current_test_metric: Optional[float] = None

        # time related
        self.epoch_start_time: float = 0.0
        self.step_start_time: float = 0.0
        self.total_train_time: float = 0.0

        # Batch tracking for resuming from mid-epoch checkpoints
        self.batch_idx_in_epoch: int = 0

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
            "max_epochs": self.max_epochs,
            "max_steps": self.max_steps,
            "best_epoch": self.best_epoch,
            "best_step": self.best_step,
            "best_train_metric": self.best_train_metric,
            "best_val_metric": self.best_val_metric,
            "best_test_metric": self.best_test_metric,
            "best_ckp_file": self.best_ckp_file,
            "batch_idx_in_epoch": self.batch_idx_in_epoch,
        }

    def from_dict(self, state_dict: Dict[str, Any]):
        for key, value in state_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)


class EventType(Enum):
    ON_EPOCH_START = "on_epoch_start"
    ON_EPOCH_END = "on_epoch_end"
    ON_BATCH_START = "on_batch_start"
    ON_BATCH_END = "on_batch_end"
    ON_TRAIN_BATCH_END = "on_train_batch_end"
    ON_EVAL_START = "on_eval_start"
    ON_EVAL_END = "on_eval_end"
    ON_CHECKPOINT_SAVE = "on_checkpoint_save"
    ON_EARLY_STOP = "on_early_stop"


@dataclass
class EventContext:
    state: RunningState
    batch_idx: Optional[int] = None
    total_batches: Optional[int] = None
    batch_metrics: Optional[Dict[str, Any]] = None
    avg_bank: Optional[Dict[str, Any]] = None
    lr: Optional[float] = None
    eval_results: Optional[Dict[str, Any]] = None
    checkpoint_path: Optional[str] = None
    checkpoint_type: Optional[str] = None
    stage: Optional[str] = None  # Current stage: "training", "validation", "testing"
