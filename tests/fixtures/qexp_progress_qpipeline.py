"""Tiny real qPipeline training command used by qexp progress integration."""

from __future__ import annotations

import faulthandler
import os
import sys
import tempfile
import time

_START = time.monotonic()


def _phase(name: str) -> None:
    print(f"qexp-progress-fixture {time.monotonic() - _START:.3f}s {name}", flush=True)


_phase("imports-start")
_DIAGNOSTIC = os.environ.get("QEXP_PROGRESS_DIAGNOSTIC") == "1"
if _DIAGNOSTIC:
    faulthandler.dump_traceback_later(20, file=sys.stderr)

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

_phase("torch-import-done")

import qqtools as qt
from qqtools.plugins.qpipeline import qPipeline
from qqtools.plugins.qpipeline.task.qtask import qTaskBase

_phase("imports-done")
if _DIAGNOSTIC:
    faulthandler.cancel_dump_traceback_later()


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
        if not getattr(self, "_first_batch_seen", False):
            self._first_batch_seen = True
            _phase("first-batch")
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
    _phase("main-start")
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
        pipeline = TinyPipeline(args, mode="train")
        _phase("fit-start")
        pipeline.fit()
        _phase("fit-done")


if __name__ == "__main__":
    main()
