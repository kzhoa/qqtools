"""
Test cases for runner2.py - Unified Training Runner
"""

import argparse
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from qqtools.plugins.qpipeline.runner.runner import RunningAgent, train_runner
from qqtools.plugins.qpipeline.runner.types import RunConfig, RunMode, RunningState

from .conftest import SimpleModel, SimpleTask

# ============================================================================
# Unit Tests for RunningState
# ============================================================================


class TestRunningState:
    """Test RunningState class"""

    def test_running_state_initialization(self):
        """Test RunningState initialization"""
        state = RunningState()
        assert state.epoch == 0
        assert state.global_step == 0
        assert state.best_epoch == 0
        assert state.best_val_metric is None

    def test_running_state_to_dict(self):
        """Test RunningState to_dict"""
        state = RunningState()
        state.epoch = 5
        state.global_step = 100
        state.best_val_metric = 0.5

        state_dict = state.to_dict()
        assert state_dict["epoch"] == 5
        assert state_dict["global_step"] == 100
        assert state_dict["best_val_metric"] == 0.5
        # Early stopper is managed separately, not in state dict
        assert "early_stop_counter" not in state_dict


# ============================================================================
# Unit Tests for RunConfig
# ============================================================================


class TestRunConfig:
    """Test RunConfig class"""

    def test_run_config_defaults(self):
        """Test RunConfig default values"""
        config = RunConfig()
        assert config.run_mode == RunMode.EPOCH
        assert config.max_epochs == 1
        assert config.max_steps is None
        assert config.eval_interval == 1

    def test_run_config_with_early_stop(self):
        """Test RunConfig with early stop configuration"""
        config = RunConfig(
            early_stop={
                "target": "val_metric",
                "patience": 5,
                "mode": "min",
                "min_delta": 1e-4,
            }
        )
        assert config.early_stop is not None
        assert config.early_stop["patience"] == 5


# ============================================================================
# Unit Tests for TrainingAgent
# ============================================================================


class TestTrainingAgent:
    """Test TrainingAgent class"""

    def setup_method(self):
        """Setup for each test"""
        self.task = SimpleTask(num_samples=100, num_features=10)
        self.model = SimpleModel(input_dim=10)
        self.device = torch.device("cpu")

    def test_training_agent_initialization(self):
        """Test TrainingAgent initialization"""
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        config = RunConfig()
        agent = RunningAgent(
            model=self.model,
            task=self.task,
            loss_fn=loss_fn,
            optimizer=optimizer,
            config=config,
        )
        assert agent.optimizer is optimizer
        assert agent.early_stopper is None  # Optional dependency, None if not injected

    def test_train_batch(self):
        """Test train_batch method"""
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        config = RunConfig()
        agent = RunningAgent(
            model=self.model,
            task=self.task,
            loss_fn=loss_fn,
            optimizer=optimizer,
            config=config,
            device=self.device,
        )

        batch_data, batch_target = next(iter(self.task.train_loader))
        batch = (batch_data, batch_target)
        metrics = agent.train_batch(batch)

        assert "loss" in metrics
        assert metrics["loss"] > 0

    def test_evaluate(self):
        """Test evaluate method"""
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        config = RunConfig()
        agent = RunningAgent(
            model=self.model,
            task=self.task,
            loss_fn=loss_fn,
            optimizer=optimizer,
            config=config,
            device=self.device,
        )

        results = agent.evaluate(use_ema=False)

        assert "val_mse" in results
        assert "val_metric" in results
        assert "test_mse" in results
        assert "test_metric" in results


# ============================================================================
# Integration Tests for train_runner
# ============================================================================


class TestTrainRunner:
    def setup_method(self):
        """Setup for each test"""
        self.args = argparse.Namespace(
            device=torch.device("cpu"),
            rank=0,
            checkpoint={},
            early_stop={},
            ckp_file=None,
            init_file=None,
        )

    def test_train_runner_epoch_mode(self):
        """Test train_runner in epoch mode"""
        with tempfile.TemporaryDirectory() as tmpdir:
            task = SimpleTask(num_samples=50, num_features=10)
            model = SimpleModel(input_dim=10)
            loss_fn = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            result = train_runner(
                model=model,
                task=task,
                loss_fn=loss_fn,
                optimizer=optimizer,
                args=self.args,
                max_epochs=2,
                eval_interval=1,
                save_dir=tmpdir,
                run_mode="epoch",
            )

            assert result["final_epoch"] == 2
            assert result["best_val_metric"] is not None
            assert result["total_train_time"] > 0

    def test_train_runner_step_mode(self):
        """Test train_runner in step mode"""
        with tempfile.TemporaryDirectory() as tmpdir:
            task = SimpleTask(num_samples=50, num_features=10)
            model = SimpleModel(input_dim=10)
            loss_fn = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            result = train_runner(
                model=model,
                task=task,
                loss_fn=loss_fn,
                optimizer=optimizer,
                args=self.args,
                max_steps=10,
                eval_interval=5,
                save_dir=tmpdir,
                run_mode="step",
            )

            assert result["final_step"] >= 10
            assert result["best_val_metric"] is not None

    def test_train_runner_with_early_stopping(self):
        """Test train_runner with early stopping"""
        with tempfile.TemporaryDirectory() as tmpdir:
            task = SimpleTask(num_samples=50, num_features=10)
            model = SimpleModel(input_dim=10)
            loss_fn = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            self.args.early_stop = {
                "target": "val_metric",
                "patience": 2,
                "mode": "min",
                "min_delta": 0.0,
            }

            result = train_runner(
                model=model,
                task=task,
                loss_fn=loss_fn,
                optimizer=optimizer,
                args=self.args,
                max_epochs=5,  # Reduced from 100 for faster testing
                eval_interval=1,
                save_dir=tmpdir,
            )

            # Verify training completed
            assert result["best_val_metric"] is not None

    def test_train_runner_with_checkpoint(self):
        """Test train_runner with checkpoint save and load"""
        with tempfile.TemporaryDirectory() as tmpdir:
            task = SimpleTask(num_samples=50, num_features=10)
            model = SimpleModel(input_dim=10)
            loss_fn = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # First run
            train_runner(
                model=model,
                task=task,
                loss_fn=loss_fn,
                optimizer=optimizer,
                args=self.args,
                max_epochs=2,
                eval_interval=1,
                save_interval=1,  # Save checkpoint every step
                save_dir=tmpdir,
            )

            # Check that checkpoint file was created
            checkpoint_files = list(Path(tmpdir).glob("*.pt"))
            assert len(checkpoint_files) > 0

            # Get the first checkpoint
            self.args.ckp_file = str(sorted(checkpoint_files)[0])

            # Create new model and resume training
            model2 = SimpleModel(input_dim=10)
            optimizer2 = optim.Adam(model2.parameters(), lr=0.001)

            result2 = train_runner(
                model=model2,
                task=task,
                loss_fn=loss_fn,
                optimizer=optimizer2,
                args=self.args,
                max_epochs=3,
                eval_interval=1,
                save_dir=tmpdir,
            )

            assert result2["final_epoch"] >= 2

    def test_train_runner_with_gradient_clipping(self):
        """Test train_runner with gradient clipping"""
        with tempfile.TemporaryDirectory() as tmpdir:
            task = SimpleTask(num_samples=50, num_features=10)
            model = SimpleModel(input_dim=10)
            loss_fn = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.1)

            result = train_runner(
                model=model,
                task=task,
                loss_fn=loss_fn,
                optimizer=optimizer,
                args=self.args,
                max_epochs=2,
                clip_grad=1.0,
                eval_interval=1,
                save_dir=tmpdir,
            )

            assert result["best_val_metric"] is not None

    def test_train_runner_with_lrscheduler(self):
        """Test train_runner with a learning rate scheduler"""
        with tempfile.TemporaryDirectory() as tmpdir:
            task = SimpleTask(num_samples=50, num_features=10)
            model = SimpleModel(input_dim=10)
            loss_fn = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

            result = train_runner(
                model=model,
                task=task,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                args=self.args,
                max_epochs=2,
                eval_interval=1,
                save_dir=tmpdir,
            )

            assert result["best_val_metric"] is not None
            # Check if learning rate has changed
            # Initial lr is 0.01, after 1 step (1 epoch), it should be 0.001
            # After 2 epochs, it should be 0.0001
            final_lr = optimizer.param_groups[0]["lr"]
            assert final_lr < 0.01

    def test_train_runner_with_warmup_cosine_lrscheduler(self):
        """Test train_runner with a warmup cosine learning rate scheduler"""
        with tempfile.TemporaryDirectory() as tmpdir:
            task = SimpleTask(num_samples=50, num_features=10)
            model = SimpleModel(input_dim=10)
            loss_fn = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)

            # Native PyTorch way to do warmup + cosine
            warmup_epochs = 1
            total_epochs = 5

            scheduler1 = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
            )
            scheduler2 = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs - warmup_epochs, eta_min=0.0001
            )

            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs]
            )

            # Run for a few epochs to observe LR changes
            max_epochs = 3  # Test with 3 epochs
            train_runner(
                model=model,
                task=task,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                args=self.args,
                max_epochs=max_epochs,
                eval_interval=1,
                save_dir=tmpdir,
            )

            # Verify learning rate changes over epochs
            # Initial LR: 0.01
            # After 1 epoch (warmup finishes), LR should be close to initial LR (0.01).
            # After more epochs, it should decay according to cosine.

            final_lr = optimizer.param_groups[0]["lr"]
            # After 3 epochs (1 warmup + 2 cosine), LR should have decayed from 0.01
            initial_lr = 0.01

            assert final_lr < initial_lr  # Should have decayed
            print(f"Final LR for warmup cosine scheduler: {final_lr}")

    def test_train_runner_with_cosine_lrscheduler(self):
        """Test train_runner with a standard CosineAnnealingLR scheduler"""
        with tempfile.TemporaryDirectory() as tmpdir:
            task = SimpleTask(num_samples=50, num_features=10)
            model = SimpleModel(input_dim=10)
            loss_fn = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)

            # Using standard PyTorch CosineAnnealingLR
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.0001)

            max_epochs = 5
            train_runner(
                model=model,
                task=task,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                args=self.args,
                max_epochs=max_epochs,
                eval_interval=1,
                save_dir=tmpdir,
            )

            final_lr = optimizer.param_groups[0]["lr"]
            # After 5 epochs, the LR should be close to eta_min (0.0001)
            assert final_lr == pytest.approx(0.0001, abs=1e-5)
            print(f"Final LR for cosine scheduler: {final_lr}")

    def test_train_runner_with_plateau_lrscheduler(self):
        """Test train_runner with a ReduceLROnPlateau scheduler"""
        with tempfile.TemporaryDirectory() as tmpdir:
            task = SimpleTask(num_samples=50, num_features=10)
            model = SimpleModel(input_dim=10)
            loss_fn = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)

            # Using standard PyTorch ReduceLROnPlateau
            # We'll set a small patience to trigger reduction quickly
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=1)

            max_epochs = 5
            result = train_runner(
                model=model,
                task=task,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                args=self.args,
                max_epochs=max_epochs,
                eval_interval=1,
                save_dir=tmpdir,
            )

            final_lr = optimizer.param_groups[0]["lr"]
            # After some epochs, the LR should have been reduced due to plateau
            # Since patience is 1, and we run for 5 epochs, it should reduce multiple times.
            # Initial LR: 0.01
            # After 1st reduction: 0.001
            # After 2nd reduction: 0.0001
            # etc.
            # Let's verify it's significantly lower than the initial LR.
            assert final_lr < 0.01
            print(f"Final LR for plateau scheduler: {final_lr}")

    def test_train_runner_with_distributed_flag(self):
        """Test train_runner with distributed flag"""
        with tempfile.TemporaryDirectory() as tmpdir:
            task = SimpleTask(num_samples=50, num_features=10)
            model = SimpleModel(input_dim=10)
            loss_fn = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            self.args.distributed = False
            self.args.rank = 0

            result = train_runner(
                model=model,
                task=task,
                loss_fn=loss_fn,
                optimizer=optimizer,
                args=self.args,
                max_epochs=2,
                eval_interval=1,
                save_dir=tmpdir,
            )

            assert result["best_val_metric"] is not None


# ============================================================================
# Tests for Config and RunMode
# ============================================================================


class TestConfigAndMode:
    """Test RunMode and RunConfig"""

    def test_run_mode_enum(self):
        """Test RunMode enum"""
        assert RunMode.EPOCH.value == "epoch"
        assert RunMode.STEP.value == "step"

    def test_run_config_string_conversion(self):
        """Test RunConfig string to enum conversion"""
        # RunConfig is frozen, so we pass run_mode as a string
        config = RunConfig(run_mode="epoch")
        assert isinstance(config.run_mode, RunMode)
        assert config.run_mode == RunMode.EPOCH


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
