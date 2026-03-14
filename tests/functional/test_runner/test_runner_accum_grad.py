from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from qqtools.plugins.qpipeline.entry_utils.type_qconfig import EarlyStopConfig, RunnerConfig
from qqtools.plugins.qpipeline.runner.agent import RunningAgent
from qqtools.plugins.qpipeline.runner.runner import train_runner
from qqtools.plugins.qpipeline.runner.runner_utils.types import RunConfig, RunMode
from qqtools.plugins.qpipeline.task.qtask import qTaskBase


class TinyModel(nn.Module):
    def __init__(self, input_dim: int = 4):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


class FixedBatchTask(qTaskBase):
    def __init__(self, num_batches: int, batch_size: int = 2, num_features: int = 4):
        super().__init__()
        num_samples = num_batches * batch_size
        x = torch.randn(num_samples, num_features)
        y = torch.randn(num_samples, 1)
        dataset = TensorDataset(x, y)
        self.train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = None

    def batch_forward(self, model, batch_data):
        x, y = batch_data
        pred = model(x)
        return {"pred": pred, "target": y}

    def batch_loss(self, out, batch_data, loss_fn):
        loss = loss_fn(out["pred"], out["target"])
        return {"loss": (loss, out["pred"].shape[0])}

    def batch_metric(self, out, batch_data):
        mse = nn.MSELoss()(out["pred"], out["target"])
        return {"mse": (mse, out["pred"].shape[0])}

    def post_metrics_to_value(self, metrics):
        return metrics.get("val_mse", 0.0)

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        return None


class FixedTensorTask(qTaskBase):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, batch_size: int):
        super().__init__()
        dataset = TensorDataset(x, y)
        self.train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = None

    def batch_forward(self, model, batch_data):
        x, y = batch_data
        pred = model(x)
        return {"pred": pred, "target": y}

    def batch_loss(self, out, batch_data, loss_fn):
        loss = loss_fn(out["pred"], out["target"])
        return {"loss": (loss, out["pred"].shape[0])}

    def batch_metric(self, out, batch_data):
        mse = nn.MSELoss()(out["pred"], out["target"])
        return {"mse": (mse, out["pred"].shape[0])}

    def post_metrics_to_value(self, metrics):
        return metrics.get("val_mse", 0.0)

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        return None


class DummyEMA:
    def __init__(self):
        self.update_calls = 0

    def update(self):
        self.update_calls += 1


class DummyScheduler:
    def __init__(self):
        self.warmup_calls = 0

    def step_warmup(self):
        self.warmup_calls += 1


@pytest.mark.parametrize("invalid_value", [0, -1, 1.5, True])
def test_schema_runner_config_rejects_invalid_accum_grad(invalid_value):
    with pytest.raises(ValueError):
        RunnerConfig(early_stop=EarlyStopConfig(patience=1), accum_grad=invalid_value)


@pytest.mark.parametrize("invalid_value", [0, -1, 1.5, True])
def test_run_config_rejects_invalid_accum_grad(invalid_value):
    with pytest.raises(ValueError):
        RunConfig(accum_grad=invalid_value)


@pytest.mark.parametrize("invalid_value", [0, -1, 1.5, True])
def test_train_runner_rejects_invalid_accum_grad(invalid_value):
    model = TinyModel()
    task = FixedBatchTask(num_batches=2)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    with pytest.raises(ValueError):
        train_runner(
            model=model,
            task=task,
            loss_fn=loss_fn,
            optimizer=optimizer,
            args=SimpleNamespace(),
            max_epochs=1,
            accum_grad=invalid_value,
        )


def test_accum_grad_delays_optimizer_steps(monkeypatch):
    task = FixedBatchTask(num_batches=4)
    model = TinyModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    step_calls = []
    zero_grad_calls = []
    original_step = optimizer.step
    original_zero_grad = optimizer.zero_grad

    def counted_step(*args, **kwargs):
        step_calls.append("step")
        return original_step(*args, **kwargs)

    def counted_zero_grad(*args, **kwargs):
        zero_grad_calls.append("zero_grad")
        return original_zero_grad(*args, **kwargs)

    monkeypatch.setattr(optimizer, "step", counted_step)
    monkeypatch.setattr(optimizer, "zero_grad", counted_zero_grad)

    agent = RunningAgent(
        model=model,
        task=task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        config=RunConfig(run_mode=RunMode.STEP, max_steps=2, eval_interval=10, accum_grad=2),
        device=torch.device("cpu"),
    )

    agent.run()

    assert len(step_calls) == 2
    assert len(zero_grad_calls) == 2
    assert agent.state.global_step == 2


def test_accum_grad_one_matches_non_accumulated_step_count(monkeypatch):
    task = FixedBatchTask(num_batches=4)
    model = TinyModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    step_calls = []
    original_step = optimizer.step

    def counted_step(*args, **kwargs):
        step_calls.append("step")
        return original_step(*args, **kwargs)

    monkeypatch.setattr(optimizer, "step", counted_step)

    agent = RunningAgent(
        model=model,
        task=task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        config=RunConfig(run_mode=RunMode.STEP, max_steps=4, eval_interval=10, accum_grad=1),
        device=torch.device("cpu"),
    )

    agent.run()

    assert len(step_calls) == 4
    assert agent.state.global_step == 4


def test_partial_accum_window_is_flushed_at_epoch_end(monkeypatch):
    task = FixedBatchTask(num_batches=5)
    model = TinyModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    step_calls = []
    original_step = optimizer.step

    def counted_step(*args, **kwargs):
        step_calls.append("step")
        return original_step(*args, **kwargs)

    monkeypatch.setattr(optimizer, "step", counted_step)

    agent = RunningAgent(
        model=model,
        task=task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        config=RunConfig(run_mode=RunMode.EPOCH, max_epochs=1, eval_interval=1, accum_grad=2),
        device=torch.device("cpu"),
    )

    agent.run()

    assert len(step_calls) == 3
    assert agent.state.global_step == 3
    assert agent.state.epoch == 1


def test_step_mode_eval_interval_counts_optimizer_steps_under_accumulation():
    task = FixedBatchTask(num_batches=8)
    model = TinyModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    listener = MagicMock()

    agent = RunningAgent(
        model=model,
        task=task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        config=RunConfig(run_mode=RunMode.STEP, max_steps=4, eval_interval=2, accum_grad=2),
        device=torch.device("cpu"),
    )
    agent.add_listener("on_eval_start", listener)

    agent.run()

    eval_steps = [call_args[0][0].state.global_step for call_args in listener.call_args_list]
    assert eval_steps == [2, 4]
    assert agent.state.global_step == 4


def test_step_mode_save_interval_counts_optimizer_steps_under_accumulation():
    task = FixedBatchTask(num_batches=8)
    model = TinyModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    checkpoint_events = []

    def capture_checkpoint_request(context):
        checkpoint_events.append(
            {
                "step": context.state.global_step,
                "batch_idx_in_epoch": context.state.batch_idx_in_epoch,
                "checkpoint_type": context.checkpoint_type,
            }
        )

    agent = RunningAgent(
        model=model,
        task=task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        config=RunConfig(run_mode=RunMode.STEP, max_steps=4, eval_interval=99, save_interval=2, accum_grad=2),
        device=torch.device("cpu"),
    )
    agent.add_listener("on_checkpoint_request", capture_checkpoint_request)

    agent.run()

    assert checkpoint_events == [
        {"step": 2, "batch_idx_in_epoch": 4, "checkpoint_type": "regular"},
        {"step": 4, "batch_idx_in_epoch": 8, "checkpoint_type": "regular"},
    ]
    assert agent.state.global_step == 4


def test_step_mode_early_stop_halts_after_first_optimizer_step_under_accumulation(monkeypatch):
    task = FixedBatchTask(num_batches=8)
    model = TinyModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    processed_batches = []
    step_calls = []
    original_step = optimizer.step

    def counted_step(*args, **kwargs):
        step_calls.append("step")
        return original_step(*args, **kwargs)

    def stop_on_first_validation(context):
        context.signal.should_stop = True
        context.signal.stop_message = "stop after first accumulated validation"

    def capture_batch_end(context):
        if context.stage == "train":
            processed_batches.append(context.batch_idx)

    monkeypatch.setattr(optimizer, "step", counted_step)

    agent = RunningAgent(
        model=model,
        task=task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        config=RunConfig(run_mode=RunMode.STEP, max_steps=8, eval_interval=1, accum_grad=2),
        device=torch.device("cpu"),
    )
    agent.add_listener("on_validation_end", stop_on_first_validation)
    agent.add_listener("on_batch_end", capture_batch_end)

    early_stopped = agent.run()

    assert early_stopped is True
    assert len(step_calls) == 1
    assert processed_batches == [0, 1]
    assert agent.state.batch_idx_in_epoch == 2


def test_scheduler_warmup_counts_optimizer_steps_under_accumulation():
    task = FixedBatchTask(num_batches=5)
    model = TinyModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    scheduler = DummyScheduler()

    agent = RunningAgent(
        model=model,
        task=task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        config=RunConfig(run_mode=RunMode.EPOCH, max_epochs=1, eval_interval=10, accum_grad=2),
        device=torch.device("cpu"),
    )

    agent.run()

    assert scheduler.warmup_calls == 3
    assert agent.state.global_step == 3


def test_ema_updates_once_per_optimizer_step():
    task = FixedBatchTask(num_batches=5)
    model = TinyModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    ema_model = DummyEMA()

    agent = RunningAgent(
        model=model,
        task=task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        config=RunConfig(run_mode=RunMode.EPOCH, max_epochs=1, eval_interval=10, accum_grad=2),
        device=torch.device("cpu"),
        ema_model=ema_model,
    )

    agent.run()

    assert ema_model.update_calls == 3


def test_tail_window_gradient_scaling_matches_grouped_batches():
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = torch.tensor([[1.5], [2.5], [3.0], [4.5], [5.5]])

    accum_task = FixedTensorTask(x, y, batch_size=1)
    grouped_task = FixedTensorTask(x, y, batch_size=2)

    model_accum = nn.Linear(1, 1)
    model_grouped = nn.Linear(1, 1)
    model_grouped.load_state_dict(model_accum.state_dict())

    optimizer_accum = optim.SGD(model_accum.parameters(), lr=0.01)
    optimizer_grouped = optim.SGD(model_grouped.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    accum_agent = RunningAgent(
        model=model_accum,
        task=accum_task,
        loss_fn=loss_fn,
        optimizer=optimizer_accum,
        config=RunConfig(run_mode=RunMode.EPOCH, max_epochs=1, eval_interval=10, accum_grad=2),
        device=torch.device("cpu"),
    )
    grouped_agent = RunningAgent(
        model=model_grouped,
        task=grouped_task,
        loss_fn=loss_fn,
        optimizer=optimizer_grouped,
        config=RunConfig(run_mode=RunMode.EPOCH, max_epochs=1, eval_interval=10, accum_grad=1),
        device=torch.device("cpu"),
    )

    accum_agent.run()
    grouped_agent.run()

    for accum_param, grouped_param in zip(model_accum.parameters(), model_grouped.parameters()):
        assert torch.allclose(accum_param, grouped_param, atol=1e-6, rtol=1e-6)


def test_accum_grad_scales_by_sample_count_for_uneven_batches():
    x = torch.tensor([[1.0], [2.0], [10.0]])
    y = torch.tensor([[1.5], [2.5], [9.5]])

    accum_task = FixedTensorTask(x, y, batch_size=2)
    grouped_task = FixedTensorTask(x, y, batch_size=3)

    model_accum = nn.Linear(1, 1)
    model_grouped = nn.Linear(1, 1)
    model_grouped.load_state_dict(model_accum.state_dict())

    optimizer_accum = optim.SGD(model_accum.parameters(), lr=0.01)
    optimizer_grouped = optim.SGD(model_grouped.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    accum_agent = RunningAgent(
        model=model_accum,
        task=accum_task,
        loss_fn=loss_fn,
        optimizer=optimizer_accum,
        config=RunConfig(run_mode=RunMode.EPOCH, max_epochs=1, eval_interval=10, accum_grad=2),
        device=torch.device("cpu"),
    )
    grouped_agent = RunningAgent(
        model=model_grouped,
        task=grouped_task,
        loss_fn=loss_fn,
        optimizer=optimizer_grouped,
        config=RunConfig(run_mode=RunMode.EPOCH, max_epochs=1, eval_interval=10, accum_grad=1),
        device=torch.device("cpu"),
    )

    accum_agent.run()
    grouped_agent.run()

    for accum_param, grouped_param in zip(model_accum.parameters(), model_grouped.parameters()):
        assert torch.allclose(accum_param, grouped_param, atol=1e-6, rtol=1e-6)


class MissingLossTask(FixedBatchTask):
    def batch_loss(self, out, batch_data, loss_fn):
        return {}


def test_training_fails_fast_when_batch_loss_omits_loss():
    task = MissingLossTask(num_batches=2)
    model = TinyModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    agent = RunningAgent(
        model=model,
        task=task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        config=RunConfig(run_mode=RunMode.STEP, max_steps=1, eval_interval=10, accum_grad=2),
        device=torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="task\\.batch_loss must return a 'loss' entry"):
        agent.run()
