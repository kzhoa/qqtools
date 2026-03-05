"""
Test cases for runner2.py - Unified Training Runner
"""

import argparse
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from qqtools.plugins.qpipeline.runner.runner import (
    RunningAgent,
    SheetLoggerListener,
    _is_periodic_trigger,
    _resolve_train_runner_policy,
    train_runner,
)
from qqtools.plugins.qpipeline.runner.runner_utils.ckp_manager import CheckpointListener
from qqtools.plugins.qpipeline.runner.runner_utils.types import EventContext, LoopSignal, RunConfig, RunMode, RunningState

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
        assert state.best_monitored_key is None
        assert state.best_monitored_metric is None
        assert state.best_model_metrics_snapshot == {}

    def test_running_state_to_dict(self):
        """Test RunningState to_dict"""
        state = RunningState()
        state.epoch = 5
        state.global_step = 100
        state.best_monitored_key = "val_metric"
        state.best_monitored_metric = 0.5
        state.best_model_metrics_snapshot = {"val_metric": 0.5, "val_mse": 0.5}

        state_dict = state.to_dict()
        assert state_dict["epoch"] == 5
        assert state_dict["global_step"] == 100
        assert state_dict["best_monitored_key"] == "val_metric"
        assert state_dict["best_monitored_metric"] == 0.5
        assert state_dict["best_model_metrics_snapshot"]["val_metric"] == 0.5
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
        assert config.max_epochs is None
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

    def test_trigger_freezes_state_but_keeps_signal_mutable(self):
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        agent = RunningAgent(
            model=self.model,
            task=self.task,
            loss_fn=loss_fn,
            optimizer=optimizer,
            config=RunConfig(),
            device=self.device,
        )

        def _listener(context: EventContext):
            with pytest.raises(AttributeError):
                context.state.epoch = 999
            assert context.state.epoch == agent.state.epoch
            context.signal.should_stop = True
            context.signal.stop_message = "stop now"

        signal = LoopSignal()
        agent.add_listener("on_validation_end", _listener)
        agent._trigger(
            "on_validation_end",
            context=EventContext(state=agent.state, signal=signal, eval_results={"val_metric": 1.0}),
            snapshot=False,
        )

        assert agent.state.epoch == 0
        assert signal.should_stop is True
        assert signal.stop_message == "stop now"

    def test_save_regular_checkpoint_calls_manager_with_regular_flag(self):
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        checkpoint_manager = Mock()
        checkpoint_manager.save.return_value = "regular.pt"
        logger = Mock()
        on_checkpoint_save = Mock()

        checkpoint_listener = CheckpointListener(
            checkpoint_manager=checkpoint_manager,
            model=self.model,
            task=self.task,
            optimizer=optimizer,
        )

        agent = RunningAgent(
            model=self.model,
            task=self.task,
            loss_fn=loss_fn,
            optimizer=optimizer,
            config=RunConfig(),
            device=self.device,
            logger=logger,
        )
        agent.add_listener("on_checkpoint_request", checkpoint_listener.on_checkpoint_request)
        agent.add_listener("on_checkpoint_save", on_checkpoint_save)

        agent._request_checkpoint("regular")
        agent._flush_checkpoint_requests()

        save_kwargs = checkpoint_manager.save.call_args.kwargs
        assert save_kwargs["is_best"] is False
        on_checkpoint_save.assert_called_once()
        checkpoint_context = on_checkpoint_save.call_args[0][0]
        assert checkpoint_context.checkpoint_type == "regular"
        assert checkpoint_context.checkpoint_path == "regular.pt"

    def test_save_best_checkpoint_calls_manager_with_best_flag(self):
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        checkpoint_manager = Mock()
        checkpoint_manager.save.return_value = "best.pt"
        logger = Mock()
        on_checkpoint_save = Mock()

        checkpoint_listener = CheckpointListener(
            checkpoint_manager=checkpoint_manager,
            model=self.model,
            task=self.task,
            optimizer=optimizer,
        )

        agent = RunningAgent(
            model=self.model,
            task=self.task,
            loss_fn=loss_fn,
            optimizer=optimizer,
            config=RunConfig(),
            device=self.device,
            logger=logger,
        )
        agent.add_listener("on_checkpoint_request", checkpoint_listener.on_checkpoint_request)
        agent.add_listener("on_checkpoint_save", on_checkpoint_save)

        agent._request_checkpoint("best")
        agent._flush_checkpoint_requests()

        save_kwargs = checkpoint_manager.save.call_args.kwargs
        assert save_kwargs["is_best"] is True
        on_checkpoint_save.assert_called_once()
        checkpoint_context = on_checkpoint_save.call_args[0][0]
        assert checkpoint_context.checkpoint_type == "best"
        assert checkpoint_context.checkpoint_path == "best.pt"
# ============================================================================
# Integration Tests for train_runner
# ============================================================================


class TestTrainRunner:
    @staticmethod
    def _make_args() -> argparse.Namespace:
        return argparse.Namespace(
            device=torch.device("cpu"),
            rank=0,
            distributed=False,
            runner=argparse.Namespace(
                checkpoint={},
                early_stop={},
                # Provide a default to ensure tests are consistent
                log_granularity=["eval"],
            ),
            ckp_file=None,
            init_file=None,
            render_type="plain",
        )

    @staticmethod
    def _create_training_components(num_samples: int = 50, num_features: int = 10, lr: float = 0.001):
        task = SimpleTask(num_samples=num_samples, num_features=num_features)
        model = SimpleModel(input_dim=num_features)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        return task, model, loss_fn, optimizer

    def setup_method(self):
        """Setup for each test"""
        self.args = self._make_args()

    def teardown_method(self, method):
        """Cleanup after each test to release any file locks (e.g. debug.log)."""
        import logging

        try:
            logging.shutdown()
        except Exception:
            pass

    def test_train_runner_epoch_mode(self):
        """Test train_runner in epoch mode"""
        with tempfile.TemporaryDirectory() as tmpdir:
            task, model, loss_fn, optimizer = self._create_training_components()

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
            assert result["best_monitored_key"] == "val_metric"
            assert result["best_monitored_metric"] is not None
            assert "val_metric" in result["best_model_metrics_snapshot"]
            assert result["total_train_time"] > 0

    def test_train_runner_step_mode(self):
        """Test train_runner in step mode"""
        with tempfile.TemporaryDirectory() as tmpdir:
            task, model, loss_fn, optimizer = self._create_training_components()

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
            assert result["best_monitored_key"] == "val_metric"
            assert result["best_monitored_metric"] is not None

    def test_train_runner_with_gradient_clipping(self):
        """Test train_runner with gradient clipping"""
        with tempfile.TemporaryDirectory() as tmpdir:
            task, model, loss_fn, optimizer = self._create_training_components(lr=0.1)

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

            assert result["best_monitored_metric"] is not None

    def test_train_runner_with_lrscheduler(self):
        """Test train_runner with a learning rate scheduler"""
        with tempfile.TemporaryDirectory() as tmpdir:
            task, model, loss_fn, optimizer = self._create_training_components(lr=0.01)
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

            assert result["best_monitored_metric"] is not None
            # Check if learning rate has changed
            # Initial lr is 0.01, after 1 step (1 epoch), it should be 0.001
            # After 2 epochs, it should be 0.0001
            final_lr = optimizer.param_groups[0]["lr"]
            assert final_lr < 0.01

    def test_train_runner_with_warmup_cosine_lrscheduler(self):
        """Test train_runner with a warmup cosine learning rate scheduler"""
        with tempfile.TemporaryDirectory() as tmpdir:
            task, model, loss_fn, optimizer = self._create_training_components(lr=0.01)

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
            task, model, loss_fn, optimizer = self._create_training_components(lr=0.01)

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
            task, model, loss_fn, optimizer = self._create_training_components(lr=0.01)

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

    def test_train_runner_without_runner_config_raises(self):
        """Test train_runner raises when args.runner is missing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            task, model, loss_fn, optimizer = self._create_training_components()
            args_without_runner = argparse.Namespace(
                device=torch.device("cpu"),
                rank=0,
                ckp_file=None,
                init_file=None,
            )

            with pytest.raises(AttributeError):
                train_runner(
                    model=model,
                    task=task,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    args=args_without_runner,
                    max_epochs=1,
                    eval_interval=1,
                    save_dir=tmpdir,
                )


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



class TestTrainRunnerPolicy:
    """Test policy resolution and boundary validation for train_runner."""

    @pytest.mark.parametrize(
        "run_mode,max_epochs,max_steps,expected_message",
        [
            ("epoch", "2", None, "max_epochs must be a positive integer"),
            ("epoch", 0, None, "max_epochs must be a positive integer"),
            ("epoch", -1, None, "max_epochs must be a positive integer"),
            ("step", None, "3", "max_steps must be a positive integer"),
            ("step", None, 0, "max_steps must be a positive integer"),
            ("step", None, -2, "max_steps must be a positive integer"),
        ],
    )
    def test_rejects_invalid_active_boundaries(
        self,
        run_mode,
        max_epochs,
        max_steps,
        expected_message,
    ):
        with pytest.raises(ValueError, match=expected_message):
            _resolve_train_runner_policy(
                run_mode=run_mode,
                max_epochs=max_epochs,
                max_steps=max_steps,
                eval_interval=1,
                save_interval=1,
            )

    @pytest.mark.parametrize(
        "run_mode,max_epochs,max_steps,expected_message",
        [
            ("epoch", None, 5, "max_epochs must be specified when run_mode='epoch'"),
            ("step", 5, None, "max_steps must be specified when run_mode='step'"),
        ],
    )
    def test_requires_boundary_for_selected_mode(
        self,
        run_mode,
        max_epochs,
        max_steps,
        expected_message,
    ):
        with pytest.raises(ValueError, match=expected_message):
            _resolve_train_runner_policy(
                run_mode=run_mode,
                max_epochs=max_epochs,
                max_steps=max_steps,
                eval_interval=1,
                save_interval=1,
            )

    @pytest.mark.parametrize(
        "eval_interval,save_interval,expected_message",
        [
            (True, 1, "eval_interval must be a positive integer"),
            (1, True, "save_interval must be a positive integer"),
        ],
    )
    def test_rejects_bool_interval_values(
        self,
        eval_interval,
        save_interval,
        expected_message,
    ):
        with pytest.raises(ValueError, match=expected_message):
            _resolve_train_runner_policy(
                run_mode="epoch",
                max_epochs=2,
                max_steps=None,
                eval_interval=eval_interval,
                save_interval=save_interval,
            )

    def test_keeps_mutual_exclusion_and_defaults(self):
        (
            resolved_run_mode,
            effective_eval_interval,
            effective_save_interval,
            effective_max_epochs,
            effective_max_steps,
            policy_warnings,
        ) = _resolve_train_runner_policy(
            run_mode="step",
            max_epochs=5,
            max_steps=4,
            eval_interval=2,
            save_interval=None,
        )

        assert resolved_run_mode == RunMode.STEP
        assert effective_eval_interval == 2
        assert effective_save_interval == 2
        assert effective_max_epochs is None
        assert effective_max_steps == 4
        assert len(policy_warnings) == 1


class TestPeriodicTrigger:
    """Test periodic trigger semantics shared by train loop and listeners."""

    @pytest.mark.parametrize(
        "run_mode,interval,global_step,epoch,is_epoch_end,expected",
        [
            (RunMode.STEP, 2, 0, 0, False, False),
            (RunMode.STEP, 2, 1, 0, False, True),
            (RunMode.STEP, 1, 0, 0, False, True),
            (RunMode.EPOCH, 2, 0, 0, True, False),
            (RunMode.EPOCH, 2, 0, 1, True, True),
            (RunMode.EPOCH, 2, 0, 1, False, False),
            (RunMode.STEP, None, 3, 0, False, False),
        ],
    )
    def test_is_periodic_trigger(self, run_mode, interval, global_step, epoch, is_epoch_end, expected):
        assert (
            _is_periodic_trigger(
                run_mode=run_mode,
                interval=interval,
                global_step=global_step,
                epoch=epoch,
                is_epoch_end=is_epoch_end,
            )
            == expected
        )

    def test_sheet_logger_listener_uses_completed_step_trigger_semantics(self):
        logger = Mock()
        logger.columns = []
        run_config = RunConfig(run_mode=RunMode.STEP, eval_interval=2, max_steps=4)
        listener = SheetLoggerListener(logger=logger, run_config=run_config, log_granularity=["eval", "batch"])

        state = RunningState(global_step=1, epoch=0)
        context = EventContext(
            state=state,
            batch_idx=0,
            total_batches=2,
            batch_metrics={"loss": 1.0},
            stage="train",
        )
        listener.on_train_batch_end(context)

        logger.write.assert_not_called()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

