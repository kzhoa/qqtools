"""Small qpipeline fixtures shared by integration and e2e tests."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from qqtools.plugins.qpipeline.task.qtask import qTaskBase


class TinyTask(qTaskBase):
    """CPU-only regression task with stable, tiny data loaders."""

    def __init__(self, num_samples: int = 32, num_features: int = 6) -> None:
        super().__init__()
        x = torch.randn(num_samples, num_features)
        y = torch.randn(num_samples, 1)
        train_size = int(num_samples * 0.6)
        val_size = int(num_samples * 0.2)
        self.train_loader = DataLoader(
            TensorDataset(x[:train_size], y[:train_size]), batch_size=8, shuffle=False
        )
        self.val_loader = DataLoader(
            TensorDataset(x[train_size : train_size + val_size], y[train_size : train_size + val_size]),
            batch_size=8,
            shuffle=False,
        )
        self.test_loader = DataLoader(
            TensorDataset(x[train_size + val_size :], y[train_size + val_size :]), batch_size=8
        )
        self.meta = {}

    @staticmethod
    def pre_batch_forward(batch_data):
        x, y = batch_data
        return {"x": x, "y": y}

    def batch_forward(self, model, batch_data):
        return {"pred": model(batch_data["x"]), "target": batch_data["y"]}

    def batch_loss(self, out, batch_data, loss_fn=None):
        criterion = nn.MSELoss() if loss_fn is None else loss_fn
        loss = criterion(out["pred"], out["target"])
        return {"loss": (loss, out["pred"].shape[0])}

    def batch_metric(self, out, batch_data):
        metric = nn.MSELoss()(out["pred"], out["target"])
        return {"mse": (metric, out["pred"].shape[0])}

    def post_metrics_to_value(self, result):
        return result.get("val_mse", result.get("val_metric", 0.0))

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        return None


class TinyModel(nn.Module):
    """Small CPU model compatible with :class:`TinyTask`."""

    def __init__(self, in_dim: int = 6) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))
