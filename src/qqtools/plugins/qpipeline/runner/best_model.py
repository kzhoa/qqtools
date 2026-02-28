from typing import Any, Dict, Optional


class BestModelManager:
    """Manages and tracks the best model based on a specific metric."""

    def __init__(self, target: str = "val_metric", mode: str = "min", min_delta: float = 0.0):
        """
        Args:
            target (str): The metric to monitor (e.g., 'val_metric', 'train_loss').
            mode (str): One of {'min', 'max'}. In 'min' mode, training will stop when the
                        quantity monitored has stopped decreasing; in 'max' mode it will
                        stop when the quantity monitored has stopped increasing.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        if mode not in ["min", "max"]:
            raise ValueError(f"mode has to be 'min' or 'max', but got {mode}")

        self.target = target
        self.mode = mode
        self.min_delta = min_delta
        self.best_metric: Optional[float] = None
        self.best_epoch: int = 0
        self.best_step: int = 0

        if self.mode == "min":
            self.is_better = lambda a, b: a < b - self.min_delta
        else:
            self.is_better = lambda a, b: a > b + self.min_delta

    def update(self, current_metric: float, epoch: int, step: int) -> bool:
        """
        Update the manager with the current metric and determine if it's the best so far.

        Args:
            current_metric (float): The current value of the metric to check.
            epoch (int): Current epoch number.
            step (int): Current global step.

        Returns:
            bool: True if the current metric is the best so far, False otherwise.
        """
        if self.best_metric is None or self.is_better(current_metric, self.best_metric):
            self.best_metric = current_metric
            self.best_epoch = epoch
            self.best_step = step
            return True
        return False

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the manager."""
        return {
            "target": self.target,
            "mode": self.mode,
            "min_delta": self.min_delta,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "best_step": self.best_step,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads the manager's state."""
        self.target = state_dict["target"]
        self.mode = state_dict["mode"]
        self.min_delta = state_dict["min_delta"]
        self.best_metric = state_dict["best_metric"]
        self.best_epoch = state_dict["best_epoch"]
        self.best_step = state_dict["best_step"]

        # Re-initialize the is_better function based on loaded mode and min_delta
        if self.mode == "min":
            self.is_better = lambda a, b: a < b - self.min_delta
        else:
            self.is_better = lambda a, b: a > b + self.min_delta
