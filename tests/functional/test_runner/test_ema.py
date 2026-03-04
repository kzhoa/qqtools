import pytest
from unittest.mock import MagicMock

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from qqtools.plugins.qpipeline.entry_utils.qema import qEMA
from qqtools.plugins.qpipeline.runner.runner import RunningAgent
from qqtools.plugins.qpipeline.runner.types import RunConfig, RunningState
from qqtools.plugins.qpipeline.task.qtask import qTaskBase

# --- Mocking necessary classes and functions for isolated testing ---


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        x = x.to(self.linear.weight.device)
        return self.linear(x)


class SimpleDataset(Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.data = torch.randn(num_samples, 10)
        self.labels = torch.randn(num_samples, 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class SimpleTask(qTaskBase):
    def __init__(self):
        super().__init__()
        self.train_loader = DataLoader(SimpleDataset(), batch_size=4, shuffle=True)
        self.val_loader = DataLoader(SimpleDataset(), batch_size=4, shuffle=False)
        self.test_loader = None  # Not used in this test

    def batch_loss(self, out, batch_data, loss_fn):
        pred = out["pred"]
        _, labels = batch_data
        loss = loss_fn(pred, labels)
        return {"loss": (loss, 1)}

    def batch_metric(self, out, batch_data):
        # For simplicity, just return a dummy metric
        return {"dummy_metric": (1.0, 1)}

    def batch_forward(self, model, batch_data):
        # Simple pass-through for testing
        pred = model(batch_data[0])
        return {"pred": pred}

    def post_metrics_to_value(self, metrics):
        # Simple pass-through for testing
        return metrics


# --- Test Functions ---


@pytest.fixture(
    params=[
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")),
    ]
)
def setup_agent_with_ema(request):
    device = torch.device(request.param)  # Use CPU or CUDA

    model = SimpleModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    task = SimpleTask()
    config = RunConfig(
        device=device,
        max_epochs=2,
        eval_interval=1,
    )

    ema_model = qEMA(model, decay=0.999, device=device)

    agent = RunningAgent(
        model=model,
        task=task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        config=config,
        device=device,
        ema_model=ema_model,
        logger=MagicMock(),  # Mock logger
        checkpoint_manager=None,
        early_stopper=None,
        best_model_manager=None,
    )
    return agent, model, ema_model, device


def test_ema_update_and_evaluation(setup_agent_with_ema):
    agent, model, ema_model, device = setup_agent_with_ema

    # Store initial model and EMA model parameters
    initial_model_params = {n: p.clone() for n, p in model.named_parameters()}
    initial_ema_params = {n: p.clone() for n, p in ema_model.module.named_parameters()}

    # Run training for a few steps
    agent.run()

    # Check if any steps were actually taken
    assert agent.state.global_step > 0
    assert agent.state.epoch == 2

    # After training, model parameters should have changed
    for name, param in model.named_parameters():
        # Using a small epsilon to avoid float precision issues,
        # but torch.equal should already work if there's any change.
        assert not torch.equal(param, initial_model_params[name]), f"Parameter {name} did not change"

    # EMA model parameters should also have changed (updated from model) but be different from model's current params
    current_model_params = {n: p for n, p in model.named_parameters()}
    current_ema_params = {n: p for n, p in ema_model.module.named_parameters()}

    for name in current_model_params.keys():
        # Check if EMA model's parameter is different from the initial EMA parameter
        if name in initial_ema_params:
            assert not torch.equal(current_ema_params[name], initial_ema_params[name])
        # Check if EMA model's parameter is different from the current model's parameter (unless decay is 1)
        assert not torch.equal(current_ema_params[name], current_model_params[name])

    # Test evaluation with EMA
    # The `evaluate` method is a lower-level utility and does not trigger `on_eval_end` events itself.
    # That event is handled by the training loop. Here, we'll verify that the models' training states
    # are correctly restored after evaluation.
    assert agent.model.training, "Model should be in training mode after run()"
    assert ema_model.module.training, "EMA model should be in training mode after run()"

    # When use_ema is True, evaluate should use the ema_model and restore its state.
    agent.evaluate(use_ema=True)
    assert agent.model.training, "Original model's state should not be affected by EMA evaluation"
    assert ema_model.module.training, "EMA model's state should be restored after evaluation"

    # When use_ema is False, evaluate should use the base model and restore its state.
    agent.evaluate(use_ema=False)
    assert agent.model.training, "Original model's state should be restored after evaluation"
    assert ema_model.module.training, "EMA model's state should not be affected by original model evaluation"

    # Ensure batch_data and model parameters are on the same device during evaluation
    # This is implicitly tested by the fact that the evaluation runs without device errors.
    # We can add a more explicit check if needed, but _prepare_batch handles data movement.
    # The agent.device is 'cpu' and both models are moved to 'cpu' in setup_agent_with_ema.
    # _prepare_batch also moves batch_data to agent.device.
    assert agent.device == device
    assert next(model.parameters()).device == device
    assert next(ema_model.module.parameters()).device == device
