import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn

from ...entry_utils.qema import qEMA
from ...task.qtask import qTaskBase
from ..hooks import RunnerBoundaryContext, RunnerHooks, ValidationHookContext
from .common import _is_periodic_trigger
from .types import RunningState


@dataclass(frozen=True)
class CheckpointPolicy:
    """Resolved checkpoint settings owned by the checkpoint capability."""

    save_interval: int
    restore_path: Optional[str]
    completion_save: bool
    keep_only_latest_regular: bool

    def __post_init__(self) -> None:
        if isinstance(self.save_interval, bool) or not isinstance(self.save_interval, int):
            raise ValueError("checkpoint save_interval must be an integer")
        if self.save_interval < 1:
            raise ValueError("checkpoint save_interval must be a positive integer")
        if self.restore_path is not None and not isinstance(self.restore_path, str):
            raise ValueError("checkpoint restore_path must be a string or None")
        if not isinstance(self.completion_save, bool):
            raise ValueError("checkpoint completion_save must be a boolean")
        if not isinstance(self.keep_only_latest_regular, bool):
            raise ValueError("checkpoint keep_only_latest_regular must be a boolean")


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
        """Save on the owner and return its path on every participating rank."""
        is_distributed = dist.is_available() and dist.is_initialized()
        outcome = {"path": None, "error_type": None, "error_message": None}
        owner_error: Optional[BaseException] = None

        if self.rank == 0:
            try:
                if best_model_tracker is None:
                    best_model_tracker = best_model_manager
                previous_best_ckp_file = state.best_ckp_file
                state_for_save = copy.copy(state)
                filename = generate_checkpoint_filename(
                    state_for_save.epoch,
                    state_for_save.global_step,
                    is_best,
                )
                if is_best:
                    state_for_save.best_ckp_file = filename
                checkpoint = self._create_checkpoint_dict(
                    state_for_save,
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
                save_path = self.save_dir / filename
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, str(save_path))
                if is_best:
                    self._rotate_checkpoint(previous_best_ckp_file, save_path)
                elif self.keep_only_latest_regular:
                    self._rotate_checkpoint(self.latest_regular_ckp_file, save_path)
                    self.latest_regular_ckp_file = filename
                outcome["path"] = str(save_path)
            except Exception as error:
                owner_error = error
                outcome["error_type"] = type(error).__name__
                outcome["error_message"] = str(error)

        if is_distributed:
            outcome_box = [outcome]
            dist.broadcast_object_list(outcome_box, src=0)
            outcome = outcome_box[0]

        if outcome["error_type"] is not None:
            if owner_error is not None:
                raise owner_error
            raise RuntimeError(
                "Checkpoint persistence failed on owner: "
                f"{outcome['error_type']}: {outcome['error_message']}"
            )
        checkpoint_path = outcome["path"]
        if not isinstance(checkpoint_path, str) or not checkpoint_path:
            raise RuntimeError("Checkpoint persistence completed without an owner checkpoint path.")
        return checkpoint_path

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


class CheckpointPlugin:
    """Runner-composed checkpoint capability.

    The plugin owns checkpoint policy, persistence invocation, live checkpoint state and its
    projection.  The Agent only invokes lifecycle slots and never receives a checkpoint path.
    """

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        model: nn.Module,
        task: qTaskBase,
        state: RunningState,
        policy: CheckpointPolicy,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        ema_model: Optional[qEMA] = None,
        early_stopper: Optional[Any] = None,
        best_model_tracker: Optional[Any] = None,
        logger: Optional[Any] = None,
        event_logger: Optional[Any] = None,
    ) -> None:
        if checkpoint_manager is None:
            raise ValueError("checkpoint_manager is required")
        self.checkpoint_manager = checkpoint_manager
        self.model = model
        self.task = task
        self.state = state
        self.policy = policy
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema_model = ema_model
        self.early_stopper = early_stopper
        self.best_model_tracker = best_model_tracker
        self.logger = logger
        self.event_logger = event_logger

    def register(self, hooks: RunnerHooks) -> None:
        hooks.set_hook_bundle(
            (
                ("after_validation", self.after_validation),
                ("boundary_cursor", self.at_boundary_cursor),
                ("after_epoch_commit", self.after_epoch_commit),
            ),
            provider_id="checkpoint.v1",
        )

    def restore_if_requested(self, device: torch.device) -> bool:
        """Restore the composed checkpoint when its configured file is available."""
        checkpoint_file = self.policy.restore_path
        if not checkpoint_file or not Path(checkpoint_file).exists():
            return False

        self.checkpoint_manager.load(
            checkpoint_file,
            device,
            self.model,
            self.task,
            self.optimizer,
            self.scheduler,
            self.ema_model,
            self.state,
            self.early_stopper,
            self.best_model_tracker,
        )
        if self.logger is not None:
            self.logger.info(f"Loaded checkpoint from {checkpoint_file}")
        return True

    def after_validation(self, context: ValidationHookContext) -> None:
        if context.is_best:
            self._guard_save_nan(self.state.current_train_loss)
            self._save("best")

    def at_boundary_cursor(self, context: RunnerBoundaryContext) -> None:
        is_due = _is_periodic_trigger(
            run_mode=context.run_mode,
            interval=self.policy.save_interval,
            global_step=context.global_step,
            epoch=context.epoch,
            is_epoch_end=context.is_epoch_end,
        )
        completion_save = context.terminal_candidate and self.policy.completion_save
        if not (is_due or completion_save):
            return
        if context.is_epoch_end and context.terminal_candidate and completion_save:
            return
        self._guard_save_nan(context.latest_train_loss)
        self._save("regular")

    def after_epoch_commit(self, context: RunnerBoundaryContext) -> None:
        if context.terminal_candidate and self.policy.completion_save:
            self._guard_save_nan(context.latest_train_loss)
            self._save("regular")

    def _guard_save_nan(self, latest_train_loss: Optional[float]) -> None:
        """Fail every rank before checkpoint persistence when the boundary has a NaN loss."""
        local_nan = latest_train_loss is not None and math.isnan(latest_train_loss)
        if dist.is_available() and dist.is_initialized():
            gathered = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, local_nan)
            source_ranks = [rank for rank, value in enumerate(gathered) if value]
        else:
            source_ranks = [0] if local_nan else []
        if source_ranks:
            raise RuntimeError(
                f"NaN detected before checkpoint persistence on ranks={source_ranks}."
            )

    def _save(self, checkpoint_type: str) -> None:
        checkpoint_path = self.checkpoint_manager.save(
            self.state,
            self.model,
            self.task,
            self.optimizer,
            self.scheduler,
            self.ema_model,
            self.early_stopper,
            self.best_model_tracker,
            is_best=(checkpoint_type == "best"),
        )
        if checkpoint_type == "best":
            self.state.best_ckp_file = Path(checkpoint_path).name
        self._write_checkpoint_record(checkpoint_type, checkpoint_path)

    def _write_checkpoint_record(self, checkpoint_type: str, checkpoint_path: str) -> None:
        local_error = None
        if self.checkpoint_manager.rank == 0:
            try:
                path = str(Path(checkpoint_path).resolve())
                if self.logger is not None:
                    self.logger.info(f"[Checkpoint Saved] type={checkpoint_type} path={path}")
                if self.event_logger is not None:
                    self.event_logger.write(
                        {
                            "event": "checkpoint_saved",
                            "epoch": self.state.epoch,
                            "global_step": self.state.global_step,
                            "type": checkpoint_type,
                            "path": path,
                        }
                    )
            except Exception as error:
                local_error = (type(error).__name__, str(error))
        if dist.is_available() and dist.is_initialized():
            gathered = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, local_error)
        else:
            gathered = [local_error]
        for rank, error in enumerate(gathered):
            if error is not None:
                raise RuntimeError(f"Checkpoint logging failed on rank {rank}: {error[0]}: {error[1]}")
