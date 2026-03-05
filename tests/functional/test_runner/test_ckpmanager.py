import tempfile
from pathlib import Path

import pytest
import torch
import torch.optim as optim

from qqtools.plugins.qpipeline.runner.ckp_manager import CheckpointManager
from qqtools.plugins.qpipeline.runner.types import RunningState

from .conftest import SimpleModel, SimpleTask


class TestCheckpointManager:
    def setup_method(self):
        """Setup reusable components for each test."""
        self.model = SimpleModel(input_dim=10)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.task = SimpleTask(num_samples=10, num_features=10)
        self.state = RunningState(epoch=5, global_step=100)
        self.state.best_monitored_key = "ema_val_metric"
        self.state.best_monitored_metric = 0.321
        self.state.best_model_metrics_snapshot = {
            "val_metric": 0.456,
            "ema_val_metric": 0.321,
            "ema_test_metric": 0.222,
        }
        self.device = torch.device("cpu")

        # Optional components, kept as None for this test
        self.scheduler = None
        self.ema_model = None
        self.early_stopper = None
        self.best_model_manager = None

    def test_save_and_load_checkpoint(self):
        """
        Tests the core functionality of CheckpointManager: saving and loading a checkpoint.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Initialize manager and save the initial state
            manager = CheckpointManager(save_dir=tmpdir)

            # Manually advance the optimizer to create some state
            self.optimizer.step()

            ckp_path = manager.save(
                self.state,
                self.model,
                self.task,
                self.optimizer,
                self.scheduler,
                self.ema_model,
                self.early_stopper,
                self.best_model_manager,
                is_best=False,
            )

            # Verify that the checkpoint file was created
            assert Path(ckp_path).exists(), "Checkpoint file was not created."

            # 2. Create new, empty objects to load the state into
            model2 = SimpleModel(input_dim=10)
            optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
            state2 = RunningState()  # Initial state (epoch=0, step=0)
            task2 = SimpleTask(num_samples=10, num_features=10)

            # Ensure the new model's weights are different from the original's
            assert not torch.equal(self.model.fc1.weight, model2.fc1.weight)
            assert state2.epoch == 0

            # 3. Load the checkpoint into the new objects
            manager.load(
                ckp_path,
                self.device,
                model2,
                task2,
                optimizer2,
                self.scheduler,
                self.ema_model,
                state2,
                self.early_stopper,
                self.best_model_manager,
            )

            # 4. Assert that the states of the new objects match the original ones
            assert state2.epoch == self.state.epoch, "Epoch was not loaded correctly."
            assert state2.global_step == self.state.global_step, "Global step was not loaded correctly."
            assert state2.best_monitored_key == self.state.best_monitored_key
            assert state2.best_monitored_metric == self.state.best_monitored_metric
            assert state2.best_model_metrics_snapshot == self.state.best_model_metrics_snapshot
            assert torch.equal(self.model.fc1.weight, model2.fc1.weight), "Model state was not loaded correctly."
            assert self.optimizer.state_dict() == optimizer2.state_dict(), "Optimizer state was not loaded correctly."

    def test_state_from_legacy_checkpoint_dict(self):
        """Loading legacy checkpoint state should keep new fields at defaults."""
        state = RunningState()
        legacy_state = {
            "epoch": 3,
            "global_step": 12,
            "best_epoch": 2,
            "best_step": 10,
            "best_ckp_file": "best_epoch2_step10.pt",
            "batch_idx_in_epoch": 4,
            "best_val_metric": 0.5,  # legacy field, should be ignored
        }
        state.from_dict(legacy_state)

        assert state.epoch == 3
        assert state.global_step == 12
        assert state.best_epoch == 2
        assert state.best_step == 10
        assert state.best_ckp_file == "best_epoch2_step10.pt"
        assert state.batch_idx_in_epoch == 4
        assert state.best_monitored_key is None
        assert state.best_monitored_metric is None
        assert state.best_model_metrics_snapshot == {}
