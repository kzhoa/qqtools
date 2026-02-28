from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..entry_utils.qema import qEMA
from .types import RunningState


def generate_checkpoint_filename(epoch: int, global_step: int, is_best: bool = False) -> str:
    """Generate checkpoint filename.

    Args:
        epoch: Current epoch
        global_step: Current global step
        is_best: Whether this is the best checkpoint

    Returns:
        Filename string
    """
    if is_best:
        return f"best_epoch{epoch}_step{global_step}.pt"
    else:
        return f"epoch{epoch}_step{global_step}.pt"


class CheckpointManager:
    def __init__(self, save_dir: str, rank: int = 0):
        self.save_dir = Path(save_dir)
        self.rank = rank

    def save(
        self,
        state: RunningState,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        ema_model: Optional[qEMA] = None,
        early_stopper: Optional[Any] = None,
        best_model_manager: Optional[Any] = None,  # Add this
        task: Optional[Any] = None,
        is_best: bool = False,
    ) -> str:
        """Save checkpoint to file"""
        if self.rank != 0:  # Only save in main process
            return ""

        checkpoint = self._create_checkpoint_dict(
            state, model, optimizer, scheduler, ema_model, early_stopper, best_model_manager, task
        )
        filename = generate_checkpoint_filename(state.epoch, state.global_step, is_best)

        # Save
        save_path = self.save_dir / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, str(save_path))

        # If it's the best model, delete the old one
        if is_best and state.best_ckp_file:
            # Convert best_ckp_file to Path object and handle both absolute and relative paths
            old_ckp_path = Path(state.best_ckp_file)
            # If it's an absolute path, use it directly; otherwise, it's relative to save_dir
            if not old_ckp_path.is_absolute():
                old_ckp_path = self.save_dir / old_ckp_path
            if old_ckp_path.exists():
                old_ckp_path.unlink()

        # Store only the filename (relative path) in state for portability
        state.best_ckp_file = filename

        return str(save_path)

    def load(
        self,
        checkpoint_path: str,
        device: torch.device,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        ema_model: Optional[qEMA] = None,
        state: Optional[RunningState] = None,
        early_stopper: Optional[Any] = None,
        best_model_manager: Optional[Any] = None,
        task: Optional[Any] = None,
    ):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        if checkpoint["optimizer_state_dict"] is not None and optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state
        if checkpoint["scheduler_state_dict"] is not None and scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load EMA state
        if "ema_state_dict" in checkpoint and ema_model is not None:
            ema_model.load_state_dict(checkpoint["ema_state_dict"])

        # Load training state
        if "state" in checkpoint and state is not None:
            state.from_dict(checkpoint["state"])

        # Load early stopper state
        if "earlystop_state_dict" in checkpoint and early_stopper is not None:
            early_stopper.load_state_dict(checkpoint["earlystop_state_dict"])

        if "best_model_state_dict" in checkpoint and best_model_manager is not None:
            best_model_manager.load_state_dict(checkpoint["best_model_state_dict"])

        # Load task state
        if "task_state_dict" in checkpoint and hasattr(task, "load_state_dict"):
            task.load_state_dict(checkpoint["task_state_dict"])

        return checkpoint

    def _create_checkpoint_dict(
        self,
        state: RunningState,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        ema_model: Optional[qEMA],
        early_stopper: Optional[Any],
        best_model_manager: Optional[Any],
        task: Optional[Any],
    ) -> Dict[str, Any]:
        """Create checkpoint dictionary with current model state."""
        checkpoint = {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "state": state.to_dict(),
        }

        if early_stopper is not None:
            checkpoint["earlystop_state_dict"] = early_stopper.state_dict()

        if best_model_manager is not None:
            checkpoint["best_model_state_dict"] = best_model_manager.state_dict()

        if ema_model is not None:
            checkpoint["ema_state_dict"] = ema_model.state_dict()

        # Save task-specific state
        if hasattr(task, "state_dict"):
            checkpoint["task_state_dict"] = task.state_dict()

        return checkpoint
