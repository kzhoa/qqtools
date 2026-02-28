from argparse import Namespace
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import qqtools as qt
from qqtools.plugins.qpipeline.task.qtask import qTaskBase


class TinyTask(qTaskBase):
    def __init__(self, num_samples=32, num_features=6):
        super().__init__()
        x = torch.randn(num_samples, num_features)
        y = torch.randn(num_samples, 1)

        train_size = int(num_samples * 0.6)
        val_size = int(num_samples * 0.2)

        train_data = TensorDataset(x[:train_size], y[:train_size])
        val_data = TensorDataset(x[train_size : train_size + val_size], y[train_size : train_size + val_size])
        test_data = TensorDataset(x[train_size + val_size :], y[train_size + val_size :])

        self.train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=8, shuffle=False)
        self.test_loader = DataLoader(test_data, batch_size=8, shuffle=False)
        self.meta = {}

    @staticmethod
    def pre_batch_forward(batch_data):
        x, y = batch_data
        return {"x": x, "y": y}

    def batch_forward(self, model, batch_data):
        pred = model(batch_data["x"])
        return {"pred": pred, "target": batch_data["y"]}

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

    def load_state_dict(self, d):
        return None


class TinyModel(nn.Module):
    def __init__(self, in_dim=6):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture
def tiny_task():
    return TinyTask()


@pytest.fixture
def tiny_model():
    return TinyModel()


@pytest.fixture
def tiny_optimizer(tiny_model):
    return torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)


@pytest.fixture
def base_args(tmp_path):
    return qt.qDict(
        {
            "seed": 42,
            "log_dir": str(tmp_path / "logs"),
            "print_freq": 10,
            "device": torch.device("cpu"),
            "rank": 0,
            "distributed": False,
            "render_type": "plain",
            "ckp_file": None,
            "init_file": None,
            "runner": {
                "checkpoint": {},
                "early_stop": {
                    "target": "val_metric",
                    "patience": 3,
                    "mode": "min",
                    "min_delta": 0.0,
                },
            },
        }
    )


@pytest.fixture
def examples_dir():
    return Path(__file__).parent / "examples"


@pytest.fixture
def cmd_args_from_yaml(examples_dir):
    def _factory(name: str, **overrides):
        config_file = examples_dir / name
        ns = Namespace(config=str(config_file), ckp_file=None, test=False, local_rank=0, ddp_detect=False)
        for k, v in overrides.items():
            setattr(ns, k, v)
        return qt.qDict.from_namespace(ns)

    return _factory
