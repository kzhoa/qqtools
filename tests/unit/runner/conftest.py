"""
Fixtures for runner tests
"""

from typing import Any, Dict, Tuple

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from qqtools.plugins.qpipeline.task.qtask import qTaskBase


class SimpleTask(qTaskBase):
    """Simple task mock for runner tests"""

    def __init__(self, num_samples=100, num_features=10):
        super().__init__()
        X = torch.randn(num_samples, num_features)
        y = torch.randn(num_samples, 1)

        train_size = int(0.7 * num_samples)
        val_size = int(0.15 * num_samples)

        train_data = TensorDataset(X[:train_size], y[:train_size])
        val_data = TensorDataset(X[train_size : train_size + val_size], y[train_size : train_size + val_size])
        test_data = TensorDataset(X[train_size + val_size :], y[train_size + val_size :])

        self.train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
        self.test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
        self.meta = {}

    def batch_forward(self, model: nn.Module, batch_data) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        if isinstance(batch_data, dict):
            x = batch_data["x"]
            y = batch_data["y"]
        else:
            x, y = batch_data
        pred = model(x)
        return {"pred": pred, "target": y}

    def batch_loss(self, out: Dict, batch_data, loss_fn=None) -> Dict[str, Tuple[torch.Tensor, int]]:
        """Compute loss"""
        pred = out["pred"]
        target = out["target"]
        loss = nn.MSELoss()(pred, target)
        return {"loss": (loss, pred.shape[0])}

    def batch_metric(self, out: Dict, batch_data) -> Dict[str, Tuple[torch.Tensor, int]]:
        """Compute metrics"""
        pred = out["pred"]
        target = out["target"]
        mse = nn.MSELoss()(pred, target)
        return {"mse": (mse, pred.shape[0])}

    def post_metrics_to_value(self, result: Dict[str, float]) -> float:
        """Convert metrics to single value for early stopping"""
        return result.get("val_mse", 0.0)

    @staticmethod
    def pre_batch_forward(batch_data):
        """Convert tuple batch data to dict"""
        if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
            x, y = batch_data
            return {"x": x, "y": y}
        return batch_data

    @staticmethod
    def post_batch_forward(output, batch_data):
        return output

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        pass


class SimpleModel(nn.Module):
    """Simple model mock for runner tests"""

    def __init__(self, input_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@pytest.fixture
def simple_task():
    """Fixture for SimpleTask"""
    return SimpleTask(num_samples=100, num_features=10)


@pytest.fixture
def simple_model():
    """Fixture for SimpleModel"""
    return SimpleModel(input_dim=10)


@pytest.fixture
def simple_optimizer(simple_model):
    """Fixture for optimizer"""
    return optim.Adam(simple_model.parameters(), lr=0.001)
