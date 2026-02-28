from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LinearLR, LRScheduler, ReduceLROnPlateau

import qqtools as qt

__all__ = ["prepare_scheduler", "WarmupConfig", "SchedulerConfig"]


# ============================================================================
# Configuration Classes
# ============================================================================


@dataclass
class WarmupConfig:
    """Configuration for linear warmup phase.

    Attributes:
        steps: Absolute number of warmup steps. If > 0, this overrides epochs-based calculation.
               If <= 0, warmup_epochs will be used instead (if set).
        epochs: Number of warmup epochs. Only used when steps <= 0.
        initial_factor: Starting learning rate factor in [0, 1].
                       Typically 0.1 means start from 0.1x of base lr.
    """

    steps: int = 0
    epochs: int = 0
    initial_factor: float = 0.1

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not (0 <= self.initial_factor <= 1.0):
            raise ValueError(f"initial_factor must be in [0, 1], got {self.initial_factor}")

    @classmethod
    def from_dict(cls, config_dict: Optional[Dict]) -> "WarmupConfig":
        """Create WarmupConfig from dictionary.

        Args:
            config_dict: Dictionary with 'steps', 'epochs', and 'initial_factor' keys.
                        If None, returns a default config with warmup disabled.

        Returns:
            WarmupConfig instance.
        """
        if config_dict is None:
            return cls(steps=0, epochs=0)
        return cls(**config_dict)


@dataclass
class SchedulerConfig:
    """Configuration for scheduler with optional warmup.

    Attributes:
        name: Scheduler name. Supported values: 'cosine', 'step', 'multi_step', 'plateau', 'lambda'.

        params: Dictionary of scheduler-specific parameters. Structure depends on the scheduler:
            - 'cosine' (CosineAnnealingLR):
                T_max: Period of learning rate annealing.
                eta_min: Minimum learning rate.

            - 'step' (StepLR):
                step_size: Period of learning rate decay.
                gamma: Multiplicative factor of learning rate decay (default: 0.1).

            - 'multi_step' (MultiStepLR):
                milestones: List of epoch indices where learning rate will decrease.
                gamma: Multiplicative factor of learning rate decay (default: 0.1).

            - 'plateau' (ReduceLROnPlateau):
                factor: Factor by which learning rate will be reduced (default: 0.1).
                patience: Number of checks with no improvement after which lr will be reduced.
                mode: One of {'min', 'max'}. Default: 'min'.
                min_lr: A threshold value for learning rate (default: 0).

            - 'lambda' (LambdaLR):
                lr_lambda: Function or list of functions computing multiplicative factor given epoch.

        warmup: Optional WarmupConfig. If None, no warmup is applied.
    """

    name: str
    params: Optional[Dict] = None
    warmup: Optional[WarmupConfig] = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.name.lower() not in SCHEDULER_GETTERS:
            raise ValueError(f"Unsupported scheduler: '{self.name}'. " f"Supported: {list(SCHEDULER_GETTERS.keys())}")
        if self.params is None:
            self.params = {}
        if self.warmup is None:
            self.warmup = WarmupConfig(steps=0)
        elif isinstance(self.warmup, dict):
            self.warmup = WarmupConfig.from_dict(self.warmup)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "SchedulerConfig":
        """Create SchedulerConfig from dictionary.

        Args:
            config_dict: Dictionary with 'name', 'params', and optional 'warmup' keys.

        Returns:
            SchedulerConfig instance.

        Raises:
            ValueError: If scheduler name is invalid.
        """
        config_dict = config_dict.copy()
        warmup_data = config_dict.pop("warmup", None)
        warmup = WarmupConfig.from_dict(warmup_data) if warmup_data else WarmupConfig(steps=0)

        name = config_dict.pop("name")
        params = config_dict.pop("params", {})

        return cls(name=name, params=params, warmup=warmup)


CANONICAL_SCHEDULER_NAMES: Dict[str, str] = {
    "cosine": "CosineAnnealingLR",
    "step": "StepLR",
    "plateau": "ReduceLROnPlateau",
    "lambda": "LambdaLR",
    "multi_step": "MultiStepLR",
}


# ============================================================================
# Scheduler Getter Functions
# ============================================================================


def get_cosine_annealing_lr(scheduler_params: dict, optimizer: Optimizer) -> LRScheduler:
    """
    Create a CosineAnnealingLR scheduler with specified parameters.

    Args:
        scheduler_params: Configuration dict containing 'T_max' and 'eta_min' keys.
        optimizer: The optimizer instance to be scheduled.

    Returns:
        A configured CosineAnnealingLR scheduler instance.
    """
    T_max = scheduler_params["T_max"]
    eta_min = scheduler_params["eta_min"]

    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)


def get_step_lr(scheduler_params: dict, optimizer: Optimizer) -> LRScheduler:
    """
    Create a StepLR scheduler with specified parameters.

    Args:
        scheduler_params: Configuration dict with required keys:
            - step_size: Period of learning rate decay.
            - gamma: Multiplicative factor of learning rate decay (default: 0.1).
        optimizer: The optimizer instance to be scheduled.

    Returns:
        A configured StepLR scheduler instance.
    """
    step_size = scheduler_params["step_size"]
    gamma = scheduler_params["gamma"]

    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


def get_multi_step_lr(scheduler_params: dict, optimizer: Optimizer) -> LRScheduler:
    """
    Create a MultiStepLR scheduler with specified parameters.

    Args:
        scheduler_params: Configuration dict with required keys:
            - milestones: List of epoch indices where learning rate will decrease.
            - gamma: Multiplicative factor of learning rate decay (default: 0.1).
        optimizer: The optimizer instance to be scheduled.

    Returns:
        A configured MultiStepLR scheduler instance.
    """
    milestones = scheduler_params["milestones"]
    gamma = scheduler_params["gamma"]

    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)


def get_reduce_lr_on_plateau(scheduler_params: dict, optimizer: Optimizer) -> ReduceLROnPlateau:
    """
    Create a ReduceLROnPlateau scheduler with specified parameters.

    Args:
        scheduler_params: Configuration dict with common keys:
            - factor: Factor by which learning rate will be reduced (default: 0.1).
            - patience: Number of checks with no improvement after which lr will be reduced.
            - mode: One of {'min', 'max'}. In 'min' mode, lr will be reduced when quantity stops decreasing.
            - min_lr: A threshold value for learning rate.
        optimizer: The optimizer instance to be scheduled.

    Returns:
        A configured ReduceLROnPlateau scheduler instance.
    """
    factor = scheduler_params.get("factor", 0.1)
    patience = scheduler_params.get("patience", 10)
    mode = scheduler_params.get("mode", "min")
    min_lr = scheduler_params.get("min_lr", 0)

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=factor, patience=patience, mode=mode, min_lr=min_lr
    )


def get_lambda_lr(scheduler_params: dict, optimizer: Optimizer) -> LambdaLR:
    """
    Create a LambdaLR scheduler with specified parameters.

    Args:
        scheduler_params: Configuration dict with required keys:
            - lr_lambda: Function or list of functions which computes a multiplicative factor
              given an integer parameter epoch.
        optimizer: The optimizer instance to be scheduled.

    Returns:
        A configured LambdaLR scheduler instance.
    """
    lr_lambda = scheduler_params["lr_lambda"]

    if isinstance(lr_lambda, str):
        try:
            lr_lambda = eval(lr_lambda)
        except Exception as e:
            raise ValueError(f"Failed to evaluate lr_lambda string '{lr_lambda}': {e}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# Scheduler getter registry mapping scheduler names to their corresponding get functions
SCHEDULER_GETTERS: Dict[str, callable] = {
    "cosine": get_cosine_annealing_lr,
    "step": get_step_lr,
    "multi_step": get_multi_step_lr,
    "plateau": get_reduce_lr_on_plateau,
    "lambda": get_lambda_lr,
}


def prepare_scheduler(args: qt.qDict, optimizer: Optimizer, batches_per_epoch: int = 0) -> LRScheduler:
    """
    Prepare a learning rate scheduler with optional warmup support.

    This is the main entry point for scheduler creation. It supports:
    - CosineAnnealingLR (special handling)
    - All PyTorch built-in schedulers (StepLR, MultiStepLR, etc.)
    - Linear warmup phase with both absolute steps and epoch-based configuration
    - Null scheduler (no scheduling) if not configured

    Warmup logic:
        - If warmup_steps > 0: Use it directly (ignores warmup_epochs)
        - If warmup_steps <= 0 and warmup_epochs > 0: Convert warmup_epochs to steps
          (requires batches_per_epoch parameter)
        - If both warmup_steps and warmup_epochs <= 0: Warmup is disabled

    Configuration structure:
        args.optim.scheduler: Scheduler name (e.g., 'cosine', 'step')
        args.optim.scheduler_params: Parameters for the scheduler
        args.optim.warmup_params: Optional warmup configuration
            - warmup_steps: Absolute number of warmup steps (overrides warmup_epochs)
            - warmup_epochs: Number of warmup epochs (used if warmup_steps <= 0)
            - warmup_factor: Starting learning rate factor (0.0-1.0)

    Args:
        args: Configuration dict (qt.qDict) containing optimizer and scheduler settings.
        optimizer: The optimizer instance to be scheduled.
        batches_per_epoch: Number of batches per epoch. Required if using warmup_epochs.
                          Default: 0 (disabled epoch-based warmup).

    Returns:
        A qWarmupScheduler instance wrapping the main scheduler.

    Raises:
        ValueError: If scheduler configuration or parameters are invalid, or if
                   warmup_epochs is set but batches_per_epoch is 0.

    Example:
        Configuration via YAML:
        optim:
          scheduler: cosine
          scheduler_params:
            T_max: 100
            eta_min: 0.0
          warmup_params:
            warmup_steps: 1000
            warmup_factor: 0.1

        or using epochs:
        optim:
          scheduler: cosine
          scheduler_params:
            T_max: 100
            eta_min: 0.0
          warmup_params:
            warmup_epochs: 5
            warmup_factor: 0.1
    """
    optim_args = args.optim.copy()
    scheduler_name = optim_args.scheduler
    scheduler_params: qt.qDict = optim_args.scheduler_params
    warmup_params: qt.qDict = optim_args.warmup_params

    # Handle disabled scheduler
    if scheduler_name is None or scheduler_name == "":
        return build_null_scheduler()

    # Build warmup config from yaml parameters
    if warmup_params is not None:
        warmup_steps = getattr(warmup_params, "warmup_steps", 0) or 0
        warmup_epochs = getattr(warmup_params, "warmup_epochs", 0) or 0
        warmup_factor = getattr(warmup_params, "warmup_factor", 0.1)

        # Convert warmup_epochs to steps if warmup_steps is not set
        if warmup_steps <= 0 and warmup_epochs > 0:
            if batches_per_epoch <= 0:
                raise ValueError(
                    f"warmup_epochs is set to {warmup_epochs} but batches_per_epoch is {batches_per_epoch}. "
                    f"batches_per_epoch must be > 0 when using warmup_epochs."
                )
            warmup_steps = warmup_epochs * batches_per_epoch
            print(
                f"[qPipeline] Converting warmup_epochs={warmup_epochs} to warmup_steps={warmup_steps} "
                f"(batches_per_epoch={batches_per_epoch})"
            )

        warmup_config = WarmupConfig(
            steps=warmup_steps,
            epochs=warmup_epochs,
            initial_factor=warmup_factor,
        )
    else:
        warmup_config = WarmupConfig(steps=0, epochs=0)

    # Build scheduler config from yaml parameters
    scheduler_config = SchedulerConfig(
        name=scheduler_name,
        params=dict(scheduler_params) if scheduler_params else {},
        warmup=warmup_config,
    )

    # Create main scheduler using the corresponding getter function
    scheduler_name_lower = scheduler_name.lower()
    if scheduler_name_lower not in SCHEDULER_GETTERS:
        raise ValueError(f"Unsupported scheduler: '{scheduler_name}'. " f"Supported: {list(SCHEDULER_GETTERS.keys())}")

    getter = SCHEDULER_GETTERS[scheduler_name_lower]
    main_scheduler = getter(scheduler_config.params, optimizer)

    # Wrap with warmup scheduler
    return qWarmupScheduler(
        optimizer,
        scheduler_config.warmup.steps,
        scheduler_config.warmup.initial_factor,
        main_scheduler,
    )


def build_null_scheduler() -> LRScheduler:
    """
    Create a null scheduler (no learning rate adjustment).

    Returns:
        A qWarmupScheduler instance that performs no scheduling operations.
    """
    print("[qPipeline] Learning rate scheduler is disabled")
    main_scheduler = qt.nn.DoNothing()
    warmup_steps = -1
    warmup_factor = 0
    optimizer = NullOptimizer()
    return qWarmupScheduler(optimizer, warmup_steps, warmup_factor, main_scheduler)


class NullOptimizer(Optimizer, qt.nn.DoNothing):
    """
    A no-op optimizer for null scheduler scenarios.

    This optimizer performs no parameter updates and maintains an empty parameter group list.
    It is used when learning rate scheduling is disabled.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize NullOptimizer with empty parameter groups."""
        self.param_groups: List[Dict] = []
        self.defaults: Dict = {}

    def step(self, closure=None) -> None:
        """
        Perform a no-op optimization step.

        Args:
            closure: Optional closure (ignored).
        """
        pass


class qWarmupScheduler(LRScheduler):
    """
    A learning rate scheduler combining linear warmup and main scheduler phases.

    This scheduler manages two phases:
    1. Warmup phase: Linear increase from warmup_factor to 1.0 over warmup_steps
    2. Main phase: Delegate to the main scheduler after warmup finishes

    Typical usage:
        - Warmup steps are typically taken at batch end during training
        - Main scheduler steps are taken at epoch end after evaluation
        - Warmup usually finishes within the first epoch
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        warmup_factor: float,
        main_scheduler: Union[LRScheduler, ReduceLROnPlateau, object],
        last_epoch: int = -1,
    ) -> None:
        """
        Initialize qWarmupScheduler.

        Args:
            optimizer: The optimizer to schedule.
            warmup_steps: Number of warmup steps (-1 to disable warmup).
            warmup_factor: Starting learning rate factor (0.0-1.0).
            main_scheduler: The main scheduler to use after warmup phase.
            last_epoch: The index of last epoch (default: -1).

        Raises:
            TypeError: If main_scheduler is not a valid scheduler type.
            ValueError: If warmup_factor is not in valid range.
        """
        if not isinstance(main_scheduler, (LRScheduler, ReduceLROnPlateau, qt.nn.DoNothing)):
            raise TypeError(
                f"main_scheduler must be an instance of LRScheduler, ReduceLROnPlateau, "
                f"or DoNothing, got {type(main_scheduler).__name__}"
            )

        if warmup_steps > 0 and not (0 <= warmup_factor <= 1.0):
            raise ValueError(f"warmup_factor must be in [0, 1], got {warmup_factor}")

        self.warmup_factor = warmup_factor

        # Initialize warmup scheduler
        if warmup_steps > 0:
            self.warmup_scheduler = LinearLR(
                optimizer,
                start_factor=warmup_factor,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
        else:
            self.warmup_scheduler = qt.nn.DoNothing()

        self.main_scheduler: Union[LRScheduler, ReduceLROnPlateau, object] = main_scheduler
        self.current_step = 0
        self.warmup_steps = warmup_steps
        self._is_plateau = isinstance(main_scheduler, ReduceLROnPlateau)

        # Call parent class initializer after all instance variables are set
        super().__init__(optimizer, last_epoch)

    def _is_donothing(self, obj: object) -> bool:
        """
        Check if an object is a DoNothing instance.

        Args:
            obj: Object to check.

        Returns:
            True if the object is a DoNothing instance, False otherwise.
        """
        return isinstance(obj, qt.nn.DoNothing) or obj.__class__.__name__ == "DoNothing"

    def _initial_step(self) -> None:
        """
        Override to prevent automatic stepping during initialization.

        The parent LRScheduler._initial_step() would call self.step(), but our scheduler
        uses step_warmup() and step_main() instead of step(). Therefore we override it
        as a no-op. This is safe because:
        - base_lrs is already initialized by parent.__init__()
        - warmup steps are explicitly called during training loops
        - main scheduler steps are explicitly called after evaluation
        """
        pass

    def step_warmup(self) -> None:
        """
        Perform a warmup step (typically called at batch end).

        Increments the internal step counter and applies the warmup scheduler if within
        the warmup phase (current_step < warmup_steps). Once the warmup phase completes,
        subsequent calls to step_warmup() become no-ops to prevent unintended state changes.
        At that point, step_main() should be called instead.

        Note: current_step and warmup_scheduler.last_epoch are kept in sync:
        - LinearLR is initialized with _initial_step() automatically called, setting last_epoch=0
        - current_step also starts at 0, and increments in sync: 0, 1, 2, ..., warmup_steps
        - warmup_scheduler.last_epoch increments the same way: 0, 1, 2, ..., warmup_steps
        - The relationship is: current_step == warmup_scheduler.last_epoch
        """
        # Warmup phase already completed; this becomes a no-op
        if self.current_step >= self.warmup_steps:
            return

        if not self._is_donothing(self.warmup_scheduler):
            self.warmup_scheduler.step()

        self.current_step += 1

    def step_main(self, metrics: Optional[float] = None) -> None:
        """
        Perform a main scheduler step (typically called at epoch end).

        After the warmup phase completes, this delegates to the main scheduler.
        For ReduceLROnPlateau, pass the validation metric; for other schedulers,
        metrics is ignored.

        Args:
            metrics: Optional metric value (required for ReduceLROnPlateau schedulers).
        """
        if self.current_step < self.warmup_steps:
            return

        if self._is_donothing(self.main_scheduler):
            return

        if self._is_plateau:
            self.main_scheduler.step(metrics)
        else:
            self.main_scheduler.step()

    def get_lr(self) -> List[float]:
        """
        Get the current learning rates.

        Returns the learning rate from the appropriate scheduler (warmup or main).

        Returns:
            List of learning rates for each parameter group.
        """
        if self.current_step <= self.warmup_steps:
            if self._is_donothing(self.warmup_scheduler):
                return self.base_lrs
            warmup_lr = self.warmup_scheduler.get_last_lr()
            return warmup_lr if warmup_lr is not None else self.base_lrs
        else:
            if self._is_donothing(self.main_scheduler):
                return self.base_lrs
            main_lr = self.main_scheduler.get_last_lr()
            return main_lr if main_lr is not None else self.base_lrs

    def get_current_lr(self) -> List[float]:
        """Get the current learning rates (alias for get_last_lr)."""
        return self.get_last_lr()

    def state_dict(self) -> Dict:
        """
        Get the scheduler state as a dictionary for checkpointing.

        Returns:
            Dictionary containing warmup state, main scheduler state, current step,
            and plateau flag.
        """
        return {
            "warmup": (self.warmup_scheduler.state_dict() if not self._is_donothing(self.warmup_scheduler) else None),
            "main": (self.main_scheduler.state_dict() if not self._is_donothing(self.main_scheduler) else None),
            "current_step": self.current_step,
            "_is_plateau": self._is_plateau,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        """
        Load scheduler state from a checkpoint dictionary.

        Args:
            state_dict: Dictionary containing saved scheduler states (from state_dict()).
        """
        if "warmup" in state_dict and state_dict["warmup"] is not None:
            if not self._is_donothing(self.warmup_scheduler):
                self.warmup_scheduler.load_state_dict(state_dict["warmup"])

        if "main" in state_dict and state_dict["main"] is not None:
            if not self._is_donothing(self.main_scheduler):
                self.main_scheduler.load_state_dict(state_dict["main"])

        self.current_step = state_dict.get("current_step", 0)
        self._is_plateau = state_dict.get("_is_plateau", isinstance(self.main_scheduler, ReduceLROnPlateau))

    def re_init(self, lr: float) -> None:
        """
        Reinitialize the main scheduler with a new base learning rate.

        Args:
            lr: The new base learning rate.
        """
        base_lrs: List[float] = self.main_scheduler.base_lrs
        self.main_scheduler.base_lrs = [lr] * len(base_lrs)
        self.main_scheduler.last_epoch = -1
        self.main_scheduler._initial_step()
