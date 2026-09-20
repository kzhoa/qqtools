"""Tiny real qPipeline training command used by qexp progress integration."""

from __future__ import annotations

import tempfile

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import qqtools as qt
from qqtools.plugins.qpipeline import qPipeline
from qqtools.plugins.qpipeline.task.qtask import qTaskBase


class TinyTask(qTaskBase):
    def __init__(self) -> None:
        super().__init__()
        features = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        targets = features.sum(dim=1, keepdim=True)
        data = TensorDataset(features, targets)
        self.train_loader = DataLoader(data, batch_size=2)
        self.val_loader = {"validation-set": DataLoader(data, batch_size=2)}
        self.test_loader = None

    def batch_forward(self, model, batch_data):
        features, targets = batch_data
        return {"prediction": model(features), "target": targets}

    def batch_loss(self, output, batch_data, loss_fn=None):
        loss = nn.functional.mse_loss(output["prediction"], output["target"])
        return {"loss": (loss, len(output["target"]))}

    def batch_metric(self, output, batch_data):
        error = (output["prediction"] - output["target"]).abs().mean()
        return {"mae": (error, len(output["target"]))}

    def post_metrics_to_value(self, result, *, stage=None):
        del stage
        if "mae" in result:
            return result["mae"]
        if "validation-set" in result:
            return result["validation-set"]["mae"]
        return None


class TinyPipeline(qPipeline):
    @staticmethod
    def prepare_model(args):
        del args
        return nn.Linear(2, 1)

    @staticmethod
    def prepare_task(args):
        del args
        return TinyTask()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="qexp-qpipeline-") as directory:
        args = qt.qDict(
            {
                "seed": 42,
                "log_dir": directory,
                "print_freq": 1,
                "ddp_detect": False,
                "ckp_file": None,
                "init_file": None,
                "optim": {
                    "loss": "mse",
                    "optimizer": "sgd",
                    "optimizer_params": {"lr": 0.01},
                    "scheduler": "",
                },
                "runner": {
                    "run_mode": "epoch",
                    "max_epochs": 1,
                    "max_steps": None,
                    "eval_interval": 1,
                },
                "task": {},
            }
        )
        TinyPipeline(args, mode="train").fit()


if __name__ == "__main__":
    main()
