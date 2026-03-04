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
            assert torch.equal(self.model.fc1.weight, model2.fc1.weight), "Model state was not loaded correctly."
            assert self.optimizer.state_dict() == optimizer2.state_dict(), "Optimizer state was not loaded correctly."
