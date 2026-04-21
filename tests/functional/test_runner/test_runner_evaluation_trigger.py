import argparse
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from qqtools.plugins.qpipeline.runner.runner import RunningAgent
from qqtools.plugins.qpipeline.runner.runner_utils.types import RunConfig, RunMode, RunningState
from qqtools.plugins.qpipeline.task.qtask import qTaskBase


# Re-using SimpleModel and SimpleTask from conftest or redefining for clarity
class SimpleModel(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


class SimpleDataset(Dataset):
    def __init__(self, num_samples=100, num_features=10):
        self.num_samples = num_samples
        self.data = torch.randn(num_samples, num_features)
        self.labels = torch.randn(num_samples, 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class SimpleTask(qTaskBase):
    def __init__(self, num_samples=100, num_features=10):
        super().__init__()
        self.train_loader = DataLoader(SimpleDataset(num_samples, num_features), batch_size=4, shuffle=True)
        self.val_loader = DataLoader(SimpleDataset(num_samples // 2, num_features), batch_size=4, shuffle=False)
        self.test_loader = None  # Not used in these tests

    def batch_loss(self, out, batch_data, loss_fn):
        pred = out["pred"]
        _, labels = batch_data
        loss = loss_fn(pred, labels)
        return {"loss": (loss, 1)}

    def batch_metric(self, out, batch_data):
        # For simplicity, just return a dummy metric
        return {"dummy_metric": (1.0, 1)}

    def batch_forward(self, model, batch_data):
        pred = model(batch_data[0])
        return {"pred": pred}

    def post_metrics_to_value(self, metrics):
        # For simplicity, just return dummy value for now
        return metrics.get("val_dummy_metric", 0.0)


class TestEvaluationTiming:
    """Tests the precise timing of evaluation triggers."""

    @pytest.fixture
    def common_setup(self):
        task = SimpleTask(num_samples=20, num_features=10)  # Small dataset for fast tests
        model = SimpleModel(input_dim=10)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        device = torch.device("cpu")
        logger = MagicMock()
        return task, model, optimizer, loss_fn, device, logger

    def test_no_eval_before_epoch_0_end(self, common_setup):
        """
        1. 验证在 epoch 模式下，当 epoch 0 尚未结束时，不会触发评估。
        """
        task, model, optimizer, loss_fn, device, logger = common_setup
        num_batches_in_epoch = len(task.train_loader)  # e.g., 5 batches

        # Run for less than one full epoch
        max_steps_to_run = num_batches_in_epoch // 2  # e.g., 2 batches

        config = RunConfig(
            run_mode=RunMode.EPOCH,
            eval_interval=1,
            max_epochs=1,  # Ensure it stops after this
            max_steps=max_steps_to_run,
            device=device,
        )
        agent = RunningAgent(model, task, loss_fn, optimizer, config=config, device=device, logger=logger)

        # Manually set max_epochs/max_steps for agent state
        agent.state.max_epochs = config.max_epochs
        agent.state.max_steps = config.max_steps if config.max_steps is not None else float("inf")

        agent.add_listener("on_eval_start", logger.on_eval_start)
        agent.add_listener("on_eval_end", logger.on_eval_end)

        agent.run()

        # on_eval_start 和 on_eval_end 都不应该被调用
        logger.on_eval_start.assert_not_called()
        logger.on_eval_end.assert_not_called()

        # 验证训练确实执行了，且未达到完整 epoch
        assert agent.state.global_step == max_steps_to_run
        assert agent.state.epoch == 0  # Epoch counter should still be 0 as epoch didn't complete

    def test_no_eval_at_epoch_0_step_0_in_epoch_mode(self, common_setup):
        """
        验证在 epoch 模式和 eval_interval=1 的情况下，在 epoch 0, step 0 (即第一次迭代)之后不会触发评估。
        """
        task, model, optimizer, loss_fn, device, logger = common_setup

        config = RunConfig(
            run_mode=RunMode.EPOCH,
            eval_interval=1,
            max_epochs=1,
            max_steps=1,  # Run for exactly one step
            device=device,
        )
        agent = RunningAgent(model, task, loss_fn, optimizer, config=config, device=device, logger=logger)

        agent.add_listener("on_eval_start", logger.on_eval_start)
        agent.add_listener("on_eval_end", logger.on_eval_end)

        agent.run()

        # on_eval_start 和 on_eval_end 都不应该被调用
        logger.on_eval_start.assert_not_called()
        logger.on_eval_end.assert_not_called()

        # 验证只执行了一个 step
        assert agent.state.global_step == 1
        assert agent.state.epoch == 0

    @pytest.mark.parametrize(
        "max_epochs, eval_interval, expected_epochs",
        [
            (2, 1, [0, 1]),  # Eval every epoch for 2 epochs
            (4, 2, [1, 3]),  # Eval every 2 epochs for 4 epochs
            (5, 3, [2]),  # Eval every 3 epochs for 5 epochs
            (1, 2, []),  # Eval interval is larger than max epochs
        ],
    )
    def test_eval_trigger_in_epoch_mode(self, common_setup, max_epochs, eval_interval, expected_epochs):
        """
        验证在 EPOCH 模式下，评估根据 eval_interval 在正确的 epoch 结束时被触发。
        """
        task, model, optimizer, loss_fn, device, logger = common_setup
        num_batches_in_epoch = len(task.train_loader)

        config = RunConfig(
            run_mode=RunMode.EPOCH,
            eval_interval=eval_interval,
            max_epochs=max_epochs,
            max_steps=None,
            device=device,
        )
        agent = RunningAgent(model, task, loss_fn, optimizer, config=config, device=device, logger=logger)

        agent.add_listener("on_eval_start", logger.on_eval_start)
        agent.add_listener("on_eval_end", logger.on_eval_end)

        agent.run()

        # 验证 on_eval_start 和 on_eval_end 的调用次数
        expected_call_count = len(expected_epochs)
        assert (
            logger.on_eval_start.call_count == expected_call_count
        ), f"Expected {expected_call_count} eval calls, but got {logger.on_eval_start.call_count}"

        # 验证评估是否在正确的 epoch 触发
        if expected_call_count > 0:
            start_calls = [call_args[0][0].state.epoch for call_args in logger.on_eval_start.call_args_list]
            assert (
                start_calls == expected_epochs
            ), f"Expected on_eval_start at epochs {expected_epochs}, got {start_calls}"

        # 验证最终的训练状态
        assert agent.state.global_step == num_batches_in_epoch * max_epochs
        assert agent.state.epoch == max_epochs

    @pytest.mark.parametrize(
        "max_steps, eval_interval, expected_steps",
        [
            (1, 1, [1]),  # Eval after the first completed step
            (5, 2, [2, 4]),  # Eval every 2nd completed step
            (6, 3, [3, 6]),  # Eval every 3rd completed step
            (2, 3, []),  # New case: interval > max_steps
            (10, 1, list(range(1, 11))),  # Eval every completed step
        ],
    )
    def test_eval_trigger_in_step_mode(self, common_setup, max_steps, eval_interval, expected_steps):
        """
        验证在 STEP 模式下，评估根据 eval_interval 在正确的 global_step 完成后被触发。
        """
        task, model, optimizer, loss_fn, device, logger = common_setup

        config = RunConfig(
            run_mode=RunMode.STEP,
            eval_interval=eval_interval,
            max_steps=max_steps,
            max_epochs=None,  # In step mode, max_epochs should ideally not interfere
            device=device,
        )
        agent = RunningAgent(model, task, loss_fn, optimizer, config=config, device=device, logger=logger)

        agent.add_listener("on_eval_start", logger.on_eval_start)
        agent.add_listener("on_eval_end", logger.on_eval_end)

        agent.run()

        # 验证调用次数
        expected_call_count = len(expected_steps)
        assert (
            logger.on_eval_start.call_count == expected_call_count
        ), f"Expected {expected_call_count} calls, got {logger.on_eval_start.call_count}"

        # 验证触发时的 global_step
        if expected_call_count > 0:
            start_calls = [call_args[0][0].state.global_step for call_args in logger.on_eval_start.call_args_list]
            assert start_calls == expected_steps, f"Expected eval at steps {expected_steps}, got {start_calls}"

        # 验证最终状态
        assert agent.state.global_step == max_steps

    def test_step_mode_can_stop_at_secondary_max_epochs_boundary(self, common_setup):
        task, model, optimizer, loss_fn, device, logger = common_setup
        num_batches_in_epoch = len(task.train_loader)

        config = RunConfig(
            run_mode=RunMode.STEP,
            eval_interval=10,
            max_steps=999,
            max_epochs=1,
            device=device,
        )
        agent = RunningAgent(model, task, loss_fn, optimizer, config=config, device=device, logger=logger)

        agent.run()

        assert agent.state.epoch == 1
        assert agent.state.global_step == num_batches_in_epoch
