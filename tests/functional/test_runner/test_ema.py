import argparse
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


from qqtools.plugins.qpipeline.entry_utils.qema import qEMA
from qqtools.plugins.qpipeline.runner import runner as runner_module
from qqtools.plugins.qpipeline.task.qtask import qTaskBase

RunningAgent = runner_module.RunningAgent
train_runner = runner_module.train_runner
RunConfig = runner_module.RunConfig


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

    def post_metrics_to_value(self, result):
        # Follow qTaskBase contract: must return scalar.
        return float(result.get("dummy_metric", 0.0))


def _build_agent_for_offload_tests(
    auto_offload: bool,
    device: torch.device = torch.device("cpu"),
):
    model = SimpleModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    task = SimpleTask()
    config = RunConfig(
        device=device,
        max_epochs=1,
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
        auto_offload=auto_offload,
        logger=MagicMock(),
    )
    return agent, model, ema_model, device


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

    assert agent.device == device
    assert next(model.parameters()).device == device
    assert next(ema_model.module.parameters()).device == device


def test_direct_ema_evaluation_preserves_standard_metric_keys(setup_agent_with_ema):
    agent, _, _, _ = setup_agent_with_ema

    standard_results = agent.evaluate(use_ema=False)
    ema_results = agent.evaluate(use_ema=True)
    aggregated_results = agent.evaluate_all_models()

    assert standard_results.models[0].variant == "standard"
    assert ema_results.models[0].variant == "ema"
    assert ema_results.target_value("ema_val_metric") is not None
    assert aggregated_results.target_value("ema_val_metric") is not None

def test_enabled_auto_offload_always_offloads_for_ema_evaluation():
    agent, model, _, _ = _build_agent_for_offload_tests(auto_offload=True)

    with patch.object(model, "cpu", wraps=model.cpu) as cpu_spy:
        agent.evaluate(use_ema=True)

    cpu_spy.assert_called_once()
    assert agent._ema_offload_ctx._auto_offload_enabled is True


def test_disabled_auto_offload_never_offloads():
    agent, model, _, _ = _build_agent_for_offload_tests(auto_offload=False)

    with patch.object(model, "cpu", wraps=model.cpu) as cpu_spy:
        agent.evaluate(use_ema=True)

    cpu_spy.assert_not_called()
    assert agent._ema_offload_ctx._auto_offload_enabled is False


def test_non_main_model_eval_does_not_offload_main_model():
    agent, model, _, device = _build_agent_for_offload_tests(auto_offload=True)
    external_model = SimpleModel().to(device)

    with patch.object(model, "cpu", wraps=model.cpu) as cpu_spy:
        agent.evaluate(model=external_model, use_ema=True)

    cpu_spy.assert_not_called()


def test_best_snapshot_uses_ema_prefixed_metrics_when_target_is_ema_val_metric(tmp_path):
    class CheckpointSafeTask(SimpleTask):
        def state_dict(self):
            return {}

        def load_state_dict(self, state_dict):
            return None

        def post_metrics_to_value(self, result):
            return float(result.get("dummy_metric", 0.0))

    task = CheckpointSafeTask()
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    ema_model = qEMA(model, decay=0.999, device=torch.device("cpu"))

    args = argparse.Namespace(
        device=torch.device("cpu"),
        rank=0,
        distributed=False,
        runner=argparse.Namespace(
            checkpoint={"target": "ema_val_metric", "mode": "min", "min_delta": 0.0},
            early_stop={"target": "val_metric", "patience": 999, "mode": "min", "min_delta": 0.0},
            log_granularity=["eval"],
        ),
        ckp_file=None,
        init_file=None,
        render_type="plain",
    )

    result = train_runner(
        model=model,
        task=task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        args=args,
        max_epochs=2,
        eval_interval=1,
        ema_model=ema_model,
        run_mode="epoch",
        save_dir=str(tmp_path / "runner_logs"),
    )

    snapshot = result["best_model_metrics_snapshot"]
    assert result["best_monitored_key"] == "ema_val_metric"
    assert result["best_monitored_metric"] is not None
    assert any(model["variant"] == "ema" for model in snapshot["models"])
