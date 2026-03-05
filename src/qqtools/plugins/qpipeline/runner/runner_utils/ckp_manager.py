import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ...entry_utils.qema import qEMA
from ...task.qtask import qTaskBase
from .types import EventContext, RunningState


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
    def __init__(self, save_dir: str, rank: int = 0, keep_only_latest_regular: bool = False):
        self.save_dir = Path(save_dir)
        self.rank = rank
        self.keep_only_latest_regular = keep_only_latest_regular
        self.latest_regular_ckp_file: Optional[str] = None

    def save(
        self,
        state: RunningState,
        model: nn.Module,
        task: qTaskBase,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        ema_model: Optional[qEMA] = None,
        early_stopper: Optional[Any] = None,
        best_model_tracker: Optional[Any] = None,
        best_model_manager: Optional[Any] = None,
        is_best: bool = False,
    ) -> str:
        """Save checkpoint to file"""
        if self.rank != 0:  # Only save in main process
            return ""

        if best_model_tracker is None:
            best_model_tracker = best_model_manager

        filename = generate_checkpoint_filename(state.epoch, state.global_step, is_best)
        checkpoint = self._create_checkpoint_dict(
            state,
            model,
            task,
            optimizer,
            scheduler,
            ema_model,
            early_stopper,
            best_model_tracker,
            filename=filename,
            is_best=is_best,
        )

        # Save the checkpoint to a file
        save_path = self.save_dir / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, str(save_path))

        # Manage checkpoint rotation to keep only the latest one if configured
        if is_best:
            self._rotate_checkpoint(state.best_ckp_file, save_path)
            # Store only the filename (relative path) in state for portability
            state.best_ckp_file = filename
        elif self.keep_only_latest_regular:
            self._rotate_checkpoint(self.latest_regular_ckp_file, save_path)
            self.latest_regular_ckp_file = filename

        return str(save_path)

    def _rotate_checkpoint(self, old_ckp_file_or_path: Optional[str], new_ckp_path: Path):
        """Delete the old checkpoint file."""
        if not old_ckp_file_or_path:
            return

        # Handle both absolute and relative paths for the old checkpoint
        old_ckp_path = Path(old_ckp_file_or_path)
        if not old_ckp_path.is_absolute():
            old_ckp_path = self.save_dir / old_ckp_path

        # Avoid deleting the file we just saved if names happen to collide
        if old_ckp_path.exists() and old_ckp_path.resolve() != new_ckp_path.resolve():
            try:
                old_ckp_path.unlink()
            except Exception:
                # Silently fail deletion (e.g., file locked on Windows)
                pass

    def load(
        self,
        checkpoint_path: str,
        device: torch.device,
        model: nn.Module,
        task: qTaskBase,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        ema_model: Optional[qEMA] = None,
        state: Optional[RunningState] = None,
        early_stopper: Optional[Any] = None,
        best_model_tracker: Optional[Any] = None,
        best_model_manager: Optional[Any] = None,
    ):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if best_model_tracker is None:
            best_model_tracker = best_model_manager

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

        if "best_model_state_dict" in checkpoint and best_model_tracker is not None:
            best_model_tracker.load_state_dict(checkpoint["best_model_state_dict"])

        if "latest_regular_ckp_file" in checkpoint and self.keep_only_latest_regular:
            self.latest_regular_ckp_file = checkpoint["latest_regular_ckp_file"]

        # Load task state
        if "task_state_dict" in checkpoint and task.has_implemented("load_state_dict"):
            task.load_state_dict(checkpoint["task_state_dict"])

        return checkpoint

    def _create_checkpoint_dict(
        self,
        state: RunningState,
        model: nn.Module,
        task: qTaskBase,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        ema_model: Optional[qEMA],
        early_stopper: Optional[Any],
        best_model_tracker: Optional[Any],
        filename: str,
        is_best: bool,
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

        if best_model_tracker is not None:
            checkpoint["best_model_state_dict"] = best_model_tracker.state_dict()

        if ema_model is not None:
            checkpoint["ema_state_dict"] = ema_model.state_dict()

        if self.keep_only_latest_regular:
            if not is_best:
                checkpoint["latest_regular_ckp_file"] = filename
            else:
                checkpoint["latest_regular_ckp_file"] = self.latest_regular_ckp_file

        # Save task-specific state
        if task.has_implemented("state_dict"):
            checkpoint["task_state_dict"] = task.state_dict()

        return checkpoint


class CheckpointListener:
    """Listener that persists checkpoints when requested by the agent."""

    def __init__(
        self,
        checkpoint_manager: Optional[CheckpointManager],
        model: nn.Module,
        task: qTaskBase,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        ema_model: Optional[qEMA] = None,
        early_stopper: Optional[Any] = None,
        best_model_tracker: Optional[Any] = None,
    ) -> None:
        self.checkpoint_manager = checkpoint_manager
        self.model = model
        self.task = task
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema_model = ema_model
        self.early_stopper = early_stopper
        self.best_model_tracker = best_model_tracker

    def _clone_state_for_save(self, context: EventContext):
        state = context.state
        if hasattr(state, "to_running_state"):
            return state.to_running_state()
        return copy.copy(state)

    def on_checkpoint_request(self, context: EventContext) -> None:
        if self.checkpoint_manager is None:
            return

        checkpoint_type = context.checkpoint_type or "regular"
        state_for_save = self._clone_state_for_save(context)
        ckp_path = self.checkpoint_manager.save(
            state_for_save,
            self.model,
            self.task,
            self.optimizer,
            self.scheduler,
            self.ema_model,
            self.early_stopper,
            self.best_model_tracker,
            is_best=(checkpoint_type == "best"),
        )
        context.checkpoint_path = ckp_path
