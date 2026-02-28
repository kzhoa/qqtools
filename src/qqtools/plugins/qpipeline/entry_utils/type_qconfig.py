from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class OptimizerParamsConfig:
    """Optimizer parameters. Union of all possible fields for supported optimizers."""

    lr: float
    weight_decay: float = 0.0
    eps: float = 1e-8
    # Adam / AdamW specific
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    amsgrad: bool = False
    # SGD specific
    momentum: float = 0.0


@dataclass
class SchedulerParamsConfig:
    """Learning rate scheduler parameters. Union of all possible fields for supported schedulers."""

    # Cosine Annealing
    T_max: Optional[int] = None
    eta_min: float = 0.0

    # Step LR
    step_size: int = 30
    gamma: float = 0.1

    # MultiStep LR
    milestones: Optional[List[int]] = None

    # ReduceLROnPlateau
    mode: str = "min"
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4
    min_lr: float = 1e-6
    target: Optional[str] = None

    # Lambda LR
    lr_lambda: Optional[str] = None


@dataclass
class WarmupParamsConfig:
    """Warmup parameters."""

    warmup_steps: int = 0
    warmup_epochs: int = 0
    warmup_factor: float = 0.1


@dataclass
class EmaParamsConfig:
    """Exponential Moving Average parameters."""

    ema: bool = False
    ema_decay: float = 0.99


@dataclass
class OptimConfig:
    """Optimizer, loss function, and learning rate schedule configuration."""

    loss: Union[str, Dict[str, Any]]
    optimizer: str
    optimizer_params: OptimizerParamsConfig
    loss_params: Optional[Dict[str, Any]] = None
    scheduler: Optional[str] = None
    scheduler_params: Optional[SchedulerParamsConfig] = None
    warmup_params: Optional[WarmupParamsConfig] = None
    ema_params: Optional[EmaParamsConfig] = None


@dataclass
class EarlyStopConfig:
    """Early stopping configuration."""

    patience: int
    target: str = "val_metric"
    mode: str = "min"
    min_delta: float = 0.0
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None


@dataclass
class CheckpointConfig:
    """Best checkpoint configuration."""

    target: str = "val_metric"
    mode: str = "min"
    min_delta: float = 0.0


@dataclass
class RunnerConfig:
    """Training runner configuration."""

    early_stop: EarlyStopConfig
    run_mode: str = "epoch"
    max_epochs: Optional[int] = None
    max_steps: Optional[int] = None
    eval_interval: int = 1
    save_interval: Optional[int] = None
    clip_grad: Optional[float] = None
    checkpoint: Optional[CheckpointConfig] = None


@dataclass
class DataLoaderConfig:
    """Data loading configuration."""

    batch_size: int
    eval_batch_size: Optional[int] = None
    num_workers: int = 0
    pin_memory: bool = True


@dataclass
class TaskConfig:
    """Task configuration including dataset and data loading."""

    dataset: str
    dataloader: DataLoaderConfig
    target: Optional[str] = None


@dataclass
class qConfig:
    """Root configuration class for qConfig."""

    # Required user-specified fields
    seed: int
    log_dir: str
    optim: OptimConfig
    runner: RunnerConfig
    task: TaskConfig

    # Optional fields
    print_freq: int = 10
    ckp_file: Optional[str] = None
    init_file: Optional[str] = None
    render_type: str = "auto"
    model: Optional[Dict[str, Any]] = None

    # Framework-reserved fields (Automatically Set)
    device: Optional[str] = None
    rank: Optional[int] = None
    distributed: Optional[bool] = None
