import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from unittest.mock import MagicMock

from qqtools.plugins.qpipeline.runner.runner import RunningAgent
from qqtools.plugins.qpipeline.runner.runner_utils.types import RunConfig, RunMode
from qqtools.plugins.qpipeline.runner.hooks import RunnerBoundaryContext, RunnerHooks
from qqtools.plugins.qpipeline.runner.runner_utils.ckp_manager import (
    CheckpointManager,
    CheckpointPlugin,
    CheckpointPolicy,
)
from qqtools.plugins.qpipeline.runner.runner_utils.metrics_jsonl import MetricsJsonlLogger

# Re-using components from conftest for consistency
from .conftest import SimpleModel, SimpleTask


def _install_checkpoint_plugin(
    agent: RunningAgent,
    checkpoint_manager: CheckpointManager,
    *,
    event_logger=None,
    save_interval: int = 1,
) -> CheckpointPlugin:
    """Install the production checkpoint capability through runner hook composition."""
    plugin = CheckpointPlugin(
        checkpoint_manager=checkpoint_manager,
        model=agent.model,
        task=agent.task,
        state=agent.state,
        policy=CheckpointPolicy(
            save_interval=save_interval,
            restore_path=None,
            completion_save=False,
            keep_only_latest_regular=checkpoint_manager.keep_only_latest_regular,
        ),
        optimizer=agent.optimizer,
        logger=agent.logger,
        event_logger=event_logger,
    )
    plugin.register(agent.hooks)
    agent.hooks.freeze()
    return plugin


class TestCheckpointTriggerTiming:
    """Tests the precise timing of regular checkpoint save triggers."""

    @pytest.fixture
    def common_setup(self):
        """Provides a common setup for the agent and its components."""
        torch.manual_seed(42)  # Set seed for reproducibility
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir)
            task = SimpleTask(num_samples=40, num_features=10)  # Larger dataset to span more steps/epochs
            model = SimpleModel(input_dim=10)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            loss_fn = nn.MSELoss()
            device = torch.device("cpu")
            logger = MagicMock()
            # Use a real CheckpointManager to save actual files
            checkpoint_manager = CheckpointManager(save_dir=save_dir)
            yield task, model, optimizer, loss_fn, device, logger, checkpoint_manager, save_dir

    @pytest.mark.parametrize(
        "run_mode, max_val, save_interval, expected_triggers",
        [
            # === EPOCH Mode Cases ===
            (RunMode.EPOCH, 2, 1, [0, 1]),  # Save every epoch for 2 epochs
            (RunMode.EPOCH, 4, 2, [1, 3]),  # Save every 2 epochs for 4 epochs
            (RunMode.EPOCH, 5, 3, [2]),  # Save on the 3rd epoch (epoch index 2)
            (RunMode.EPOCH, 1, 2, []),  # Save interval is larger than max epochs
            # === STEP Mode Cases ===
            (RunMode.STEP, 5, 1, [1, 2, 3, 4, 5]),  # Save every completed step
            (RunMode.STEP, 5, 2, [2, 4]),  # Save every 2nd completed step
            (RunMode.STEP, 6, 3, [3, 6]),  # Save every 3rd completed step
            (RunMode.STEP, 2, 3, []),  # Save interval is larger than max steps
        ],
    )
    def test_save_trigger_timing(self, common_setup, run_mode, max_val, save_interval, expected_triggers):
        """
        Verifies that checkpoints are saved at correct intervals and contain correct information.
        """
        task, model, optimizer, loss_fn, device, logger, ckp_manager, save_dir = common_setup

        config_params = {
            "run_mode": run_mode,
            "eval_interval": max_val + 1,  # Disable evaluation to isolate checkpointing
            "device": device,
        }
        if run_mode == RunMode.EPOCH:
            config_params["max_epochs"] = max_val
        else:
            config_params["max_steps"] = max_val

        config = RunConfig(**config_params)
        agent = RunningAgent(
            model, task, loss_fn, optimizer, config=config, device=device, logger=logger, hooks=RunnerHooks()
        )

        _install_checkpoint_plugin(agent, ckp_manager, save_interval=save_interval)

        # Clone initial model state before training
        initial_weights = {k: v.clone() for k, v in model.state_dict().items()}

        agent.run()

        # --- Assertions ---
        saved_files = sorted(save_dir.glob("*.pt"))
        assert len(saved_files) == len(expected_triggers)

        if not expected_triggers:
            return

        # 1. Verify that the model state evolves over time if multiple checkpoints are saved.
        if len(saved_files) > 1:
            last_checkpoint = torch.load(saved_files[-1])
            last_weights = last_checkpoint["model_state_dict"]

            weights_are_different = any(not torch.equal(initial_weights[k], last_weights[k]) for k in initial_weights)
            assert weights_are_different, "Model weights did not change between the start and the last checkpoint."

        # 2. Verify the content of each individual checkpoint.
        for i, step_or_epoch in enumerate(expected_triggers):
            checkpoint_path = saved_files[i]
            checkpoint = torch.load(checkpoint_path)

            # Verify epoch/step number in the state dictionary
            state = checkpoint["state"]
            if run_mode == RunMode.EPOCH:
                assert state["epoch"] == step_or_epoch
            else:  # STEP mode
                assert state["global_step"] == step_or_epoch

            # Verify essential keys are present
            assert "model_state_dict" in checkpoint
            assert "optimizer_state_dict" in checkpoint
            optimizer_state = checkpoint["optimizer_state_dict"]
            assert "state" in optimizer_state
            assert "param_groups" in optimizer_state
            assert len(optimizer_state["param_groups"]) == len(optimizer.param_groups)

    def test_keep_only_latest_regular_keeps_single_regular_file(self, common_setup):
        task, model, optimizer, loss_fn, device, logger, _, save_dir = common_setup

        config = RunConfig(
            run_mode=RunMode.EPOCH,
            max_epochs=3,
            eval_interval=10,
            device=device,
        )
        agent = RunningAgent(
            model, task, loss_fn, optimizer, config=config, device=device, logger=logger, hooks=RunnerHooks()
        )

        _install_checkpoint_plugin(
            agent,
            CheckpointManager(save_dir=save_dir, keep_only_latest_regular=True),
            save_interval=1,
        )

        agent.run()

        saved_files = sorted(save_dir.glob("*.pt"))
        assert len(saved_files) == 1
        checkpoint = torch.load(saved_files[0])
        assert checkpoint["state"]["epoch"] == 2

    def test_checkpoint_plugin_logs_absolute_path_after_successful_save(self, common_setup):
        task, model, optimizer, loss_fn, device, logger, checkpoint_manager, save_dir = common_setup
        agent = RunningAgent(
            model,
            task,
            loss_fn,
            optimizer,
            config=RunConfig(run_mode=RunMode.STEP, max_steps=1, device=device),
            device=device,
            logger=logger,
            hooks=RunnerHooks(),
        )
        plugin = _install_checkpoint_plugin(agent, checkpoint_manager, save_interval=1)
        plugin.at_boundary_cursor(
            RunnerBoundaryContext(
                epoch=0,
                global_step=1,
                run_mode=RunMode.STEP,
                did_optimizer_step=True,
                is_epoch_end=False,
                terminal_candidate=False,
                latest_train_loss=0.1,
            )
        )

        checkpoint_path = next(save_dir.glob("*.pt")).resolve()
        logger.info.assert_any_call(f"[Checkpoint Saved] type=regular path={checkpoint_path}")

    def test_checkpoint_plugin_writes_jsonl_event_after_successful_save(self, common_setup):
        task, model, optimizer, loss_fn, device, logger, checkpoint_manager, save_dir = common_setup
        event_logger = MetricsJsonlLogger(save_dir / "metrics.jsonl")
        agent = RunningAgent(
            model,
            task,
            loss_fn,
            optimizer,
            config=RunConfig(run_mode=RunMode.STEP, max_steps=1, device=device),
            device=device,
            logger=logger,
            hooks=RunnerHooks(),
        )
        plugin = _install_checkpoint_plugin(
            agent,
            checkpoint_manager,
            event_logger=event_logger,
            save_interval=1,
        )
        plugin.at_boundary_cursor(
            RunnerBoundaryContext(
                epoch=0,
                global_step=1,
                run_mode=RunMode.STEP,
                did_optimizer_step=True,
                is_epoch_end=False,
                terminal_candidate=False,
                latest_train_loss=0.1,
            )
        )
        event_logger.close()

        event = json.loads((save_dir / "metrics.jsonl").read_text().strip())
        assert event["event"] == "checkpoint_saved"
        assert event["type"] == "regular"
        assert event["path"] == str(next(save_dir.glob("*.pt")).resolve())
