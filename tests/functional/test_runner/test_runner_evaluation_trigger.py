import argparse
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from qqtools.plugins.qpipeline.runner.events import CommandName
from qqtools.plugins.qpipeline.runner.runner import RunningAgent
from qqtools.plugins.qpipeline.runner.runner_utils.types import LoopSignal, RunConfig, RunMode, RunningState
from qqtools.plugins.qpipeline.task.qtask import qTaskBase


def _handle_checkpoint_command(context) -> str:
    return "test-checkpoint.pt"


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

    def post_metrics_to_value(self, result):
        # For simplicity, just return dummy value for now
        return result.get("val_dummy_metric", 0.0)


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

        observed_eval_epochs = []

        def capture_eval_start(context):
            observed_eval_epochs.append(context.runner.run_state.epoch)
            logger.on_eval_start(context)

        agent.add_listener("on_eval_start", capture_eval_start)
        agent.add_listener("on_eval_end", logger.on_eval_end)
        agent.dispatcher.set_handler(CommandName.SAVE_CHECKPOINT, _handle_checkpoint_command)

        agent.run()

        # 验证 on_eval_start 和 on_eval_end 的调用次数
        expected_call_count = len(expected_epochs)
        assert (
            logger.on_eval_start.call_count == expected_call_count
        ), f"Expected {expected_call_count} eval calls, but got {logger.on_eval_start.call_count}"

        # 验证评估是否在正确的 epoch 触发
        if expected_call_count > 0:
            assert (
                observed_eval_epochs == expected_epochs
            ), f"Expected on_eval_start at epochs {expected_epochs}, got {observed_eval_epochs}"

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

        observed_eval_steps = []

        def capture_eval_start(context):
            observed_eval_steps.append(context.runner.run_state.global_step)
            logger.on_eval_start(context)

        agent.add_listener("on_eval_start", capture_eval_start)
        agent.add_listener("on_eval_end", logger.on_eval_end)
        agent.dispatcher.set_handler(CommandName.SAVE_CHECKPOINT, _handle_checkpoint_command)

        agent.run()

        # 验证调用次数
        expected_call_count = len(expected_steps)
        assert (
            logger.on_eval_start.call_count == expected_call_count
        ), f"Expected {expected_call_count} calls, got {logger.on_eval_start.call_count}"

        # 验证触发时的 global_step
        if expected_call_count > 0:
            assert observed_eval_steps == expected_steps, f"Expected eval at steps {expected_steps}, got {observed_eval_steps}"

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

    def test_completion_actions_run_once_when_final_step_misses_periodic_intervals(self, common_setup):
        task, model, optimizer, loss_fn, device, logger = common_setup
        agent = RunningAgent(
            model,
            task,
            loss_fn,
            optimizer,
            config=RunConfig(
                run_mode=RunMode.STEP,
                eval_interval=10,
                save_interval=10,
                max_steps=2,
                completion={"eval": True, "save": True},
                device=device,
            ),
            device=device,
            logger=logger,
        )
        checkpoint_types = []
        agent.add_listener("on_eval_start", logger.on_eval_start)
        agent.dispatcher.set_handler(
            CommandName.SAVE_CHECKPOINT,
            lambda context: (checkpoint_types.append(context.checkpoint_type), "test-checkpoint.pt")[1],
        )

        agent.run()

        assert logger.on_eval_start.call_count == 1
        assert checkpoint_types == ["best", "regular"]

    def test_completion_save_is_independent_from_periodic_evaluation(self, common_setup):
        task, model, optimizer, loss_fn, device, logger = common_setup
        agent = RunningAgent(
            model,
            task,
            loss_fn,
            optimizer,
            config=RunConfig(
                run_mode=RunMode.STEP,
                eval_interval=2,
                save_interval=10,
                max_steps=2,
                completion={"eval": True, "save": True},
                device=device,
            ),
            device=device,
            logger=logger,
        )
        checkpoint_types = []
        agent.add_listener("on_eval_start", logger.on_eval_start)
        agent.dispatcher.set_handler(
            CommandName.SAVE_CHECKPOINT,
            lambda context: (checkpoint_types.append(context.checkpoint_type), "test-checkpoint.pt")[1],
        )

        agent.run()

        assert logger.on_eval_start.call_count == 1
        assert checkpoint_types == ["best", "regular"]

    def test_terminal_epoch_completion_save_uses_committed_cursor(self, common_setup):
        task, model, optimizer, loss_fn, device, logger = common_setup
        agent = RunningAgent(
            model,
            task,
            loss_fn,
            optimizer,
            config=RunConfig(
                run_mode=RunMode.EPOCH,
                eval_interval=2,
                save_interval=2,
                max_epochs=1,
                completion={"eval": True, "save": True},
                device=device,
            ),
            device=device,
            logger=logger,
        )
        checkpoint_cursors = []
        agent.add_listener("on_eval_start", logger.on_eval_start)

        def capture_checkpoint(context):
            checkpoint_cursors.append(
                (
                    context.checkpoint_type,
                    context.state.epoch,
                    context.state.batch_idx_in_epoch,
                )
            )
            return "test-checkpoint.pt"

        agent.dispatcher.set_handler(CommandName.SAVE_CHECKPOINT, capture_checkpoint)
        agent.run()

        assert logger.on_eval_start.call_count == 1
        assert agent.state.epoch_result_val_metric_source == "current_eval"
        assert checkpoint_cursors[-1] == ("regular", 1, 0)

    def test_early_stop_runs_requested_completion_save(self, common_setup):
        task, model, optimizer, loss_fn, device, logger = common_setup
        agent = RunningAgent(
            model,
            task,
            loss_fn,
            optimizer,
            config=RunConfig(
                run_mode=RunMode.STEP,
                eval_interval=1,
                save_interval=10,
                max_steps=99,
                completion={"save": True},
                device=device,
            ),
            device=device,
            logger=logger,
        )
        checkpoint_types = []

        def request_stop(context):
            context.signal.request_stop("test", "requested stop")

        def capture_checkpoint(context):
            checkpoint_types.append(context.checkpoint_type)
            return "test-checkpoint.pt"

        agent.add_listener("on_validation_end", request_stop)
        agent.dispatcher.set_handler(CommandName.SAVE_CHECKPOINT, capture_checkpoint)
        assert agent.run() == "early_stop"
        assert checkpoint_types == ["best", "regular"]

    def test_restored_state_at_limit_does_not_run_completion_actions(self, common_setup):
        task, model, optimizer, loss_fn, device, logger = common_setup
        agent = RunningAgent(
            model,
            task,
            loss_fn,
            optimizer,
            config=RunConfig(
                run_mode=RunMode.STEP,
                eval_interval=10,
                max_steps=1,
                completion={"eval": True, "save": True},
                device=device,
            ),
            state=RunningState(global_step=1),
            device=device,
            logger=logger,
        )
        agent.add_listener("on_eval_start", logger.on_eval_start)
        agent.dispatcher.set_handler(CommandName.SAVE_CHECKPOINT, _handle_checkpoint_command)

        assert agent.run() is None
        logger.on_eval_start.assert_not_called()

    def test_each_periodic_boundary_synchronizes_stop_once(self, common_setup, monkeypatch):
        task, model, optimizer, loss_fn, device, logger = common_setup
        synchronized_signals = []

        def capture_stop_synchronization(signal, device, distributed):
            synchronized_signals.append(signal)

        monkeypatch.setattr(LoopSignal, "synchronize_stop", capture_stop_synchronization)
        agent = RunningAgent(
            model,
            task,
            loss_fn,
            optimizer,
            config=RunConfig(
                run_mode=RunMode.STEP,
                eval_interval=10,
                max_steps=2,
                device=device,
            ),
            device=device,
            logger=logger,
        )

        assert agent.run() == "run_limit"
        assert len(synchronized_signals) == 2

    def test_completion_evaluation_stop_does_not_change_terminal_reason_or_sync(
        self,
        common_setup,
        monkeypatch,
    ):
        task, model, optimizer, loss_fn, device, logger = common_setup
        synchronized_signals = []
        early_stop_listener = MagicMock()

        def capture_stop_synchronization(signal, device, distributed):
            synchronized_signals.append(signal)

        monkeypatch.setattr(LoopSignal, "synchronize_stop", capture_stop_synchronization)
        agent = RunningAgent(
            model,
            task,
            loss_fn,
            optimizer,
            config=RunConfig(
                run_mode=RunMode.STEP,
                eval_interval=10,
                max_steps=1,
                completion={"eval": True},
                device=device,
            ),
            device=device,
            logger=logger,
        )
        agent.add_listener(
            "on_validation_end",
            lambda context: context.signal.request_stop("completion_eval", "requested stop"),
        )
        agent.dispatcher.set_handler(CommandName.SAVE_CHECKPOINT, _handle_checkpoint_command)
        agent.add_listener("on_early_stop", early_stop_listener)

        assert agent.run() == "run_limit"
        assert len(synchronized_signals) == 1
        early_stop_listener.assert_not_called()

    def test_completion_actions_share_one_signal_without_stop_synchronization(
        self,
        common_setup,
        monkeypatch,
    ):
        task, model, optimizer, loss_fn, device, logger = common_setup
        synchronized_signals = []
        completion_signals = {}

        def capture_stop_synchronization(signal, device, distributed):
            synchronized_signals.append(signal)

        def capture_validation_signal(context):
            completion_signals["evaluation"] = context.signal

        def capture_checkpoint_signal(context):
            completion_signals[context.checkpoint_type] = context
            return "test-checkpoint.pt"

        monkeypatch.setattr(LoopSignal, "synchronize_stop", capture_stop_synchronization)
        agent = RunningAgent(
            model,
            task,
            loss_fn,
            optimizer,
            config=RunConfig(
                run_mode=RunMode.STEP,
                eval_interval=10,
                save_interval=10,
                max_steps=1,
                completion={"eval": True, "save": True},
                device=device,
            ),
            device=device,
            logger=logger,
        )
        agent.add_listener("on_eval_start", capture_validation_signal)
        agent.dispatcher.set_handler(CommandName.SAVE_CHECKPOINT, capture_checkpoint_signal)

        assert agent.run() == "run_limit"

        completion_signal = completion_signals["evaluation"]
        assert completion_signals["best"].state is agent.state
        assert completion_signals["regular"].state is agent.state
        assert len(synchronized_signals) == 1
        assert completion_signal is not synchronized_signals[0]

    def test_periodic_early_stop_preserves_reasons_and_isolated_completion_stop(
        self,
        common_setup,
        monkeypatch,
    ):
        task, model, optimizer, loss_fn, device, logger = common_setup
        synchronized_signals = []
        early_stop_signals = []
        checkpoint_signals = {}

        def capture_stop_synchronization(signal, device, distributed):
            synchronized_signals.append(signal)

        def request_periodic_stop(context):
            context.signal.request_stop("periodic_eval", "requested stop")

        def capture_checkpoint(context):
            checkpoint_signals[context.checkpoint_type] = context
            return "test-checkpoint.pt"

        monkeypatch.setattr(LoopSignal, "synchronize_stop", capture_stop_synchronization)
        agent = RunningAgent(
            model,
            task,
            loss_fn,
            optimizer,
            config=RunConfig(
                run_mode=RunMode.STEP,
                eval_interval=1,
                save_interval=10,
                max_steps=99,
                completion={"save": True},
                device=device,
            ),
            device=device,
            logger=logger,
        )
        agent.add_listener("on_validation_end", request_periodic_stop)
        agent.dispatcher.set_handler(CommandName.SAVE_CHECKPOINT, capture_checkpoint)
        agent.add_listener("on_early_stop", lambda context: early_stop_signals.append(context.signal))

        assert agent.run() == "early_stop"
        assert len(synchronized_signals) == 1
        assert len(early_stop_signals) == 1
        assert checkpoint_signals["best"].state is agent.state
        assert checkpoint_signals["regular"].state is agent.state
        assert [reason.source for reason in early_stop_signals[0].stop_reasons] == ["periodic_eval"]

    def test_run_limit_wins_over_periodic_stop_and_defers_epoch_save(self, common_setup, monkeypatch):
        task, model, optimizer, loss_fn, device, logger = common_setup
        synchronized_signals = []
        checkpoint_cursors = []
        early_stop_listener = MagicMock()

        def capture_stop_synchronization(signal, device, distributed):
            synchronized_signals.append(signal)

        def request_periodic_stop(context):
            context.signal.request_stop("periodic_eval", "requested stop")

        def capture_checkpoint(context):
            checkpoint_cursors.append(
                (
                    context.checkpoint_type,
                    context.state.epoch,
                    context.state.batch_idx_in_epoch,
                )
            )
            return "test-checkpoint.pt"

        monkeypatch.setattr(LoopSignal, "synchronize_stop", capture_stop_synchronization)
        agent = RunningAgent(
            model,
            task,
            loss_fn,
            optimizer,
            config=RunConfig(
                run_mode=RunMode.EPOCH,
                eval_interval=1,
                save_interval=1,
                max_epochs=1,
                completion={"save": True},
                device=device,
            ),
            device=device,
            logger=logger,
        )
        agent.add_listener("on_validation_end", request_periodic_stop)
        agent.dispatcher.set_handler(CommandName.SAVE_CHECKPOINT, capture_checkpoint)
        agent.add_listener("on_early_stop", early_stop_listener)

        assert agent.run() == "run_limit"
        assert len(synchronized_signals) == 1
        assert early_stop_listener.call_count == 0
        assert [item[0] for item in checkpoint_cursors].count("regular") == 1
        assert checkpoint_cursors[-1] == ("regular", 1, 0)
