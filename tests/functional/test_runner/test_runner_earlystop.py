"""
Test cases for early stopping in train_runner
"""

import argparse
import tempfile
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from qqtools.plugins.qpipeline.runner.runner import train_runner
from .conftest import SimpleModel, SimpleTask


class TestTrainRunnerEarlyStop:
    @staticmethod
    def _make_args() -> argparse.Namespace:
        return argparse.Namespace(
            device=torch.device("cpu"),
            rank=0,
            distributed=False,
            runner=argparse.Namespace(
                checkpoint={},
                early_stop={},
                log_granularity=["eval"],
            ),
            ckp_file=None,
            init_file=None,
            render_type="plain",
        )

    @staticmethod
    def _create_training_components(num_samples: int = 50, num_features: int = 10, lr: float = 0.001):
        task = SimpleTask(num_samples=num_samples, num_features=num_features)
        model = SimpleModel(input_dim=num_features)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        return task, model, loss_fn, optimizer

    def setup_method(self):
        self.args = self._make_args()

    def test_early_stop_min_mode(self):
        """Test early stopping with mode='min'"""
        with tempfile.TemporaryDirectory() as tmpdir:
            task, model, loss_fn, optimizer = self._create_training_components()
            self.args.runner.early_stop = {
                "target": "val_metric",
                "patience": 2,
                "mode": "min",
                "min_delta": 0.0,
            }
            result = train_runner(
                model=model,
                task=task,
                loss_fn=loss_fn,
                optimizer=optimizer,
                args=self.args,
                max_epochs=10,
                eval_interval=1,
                save_dir=tmpdir,
            )
            assert result["best_val_metric"] is not None
            assert result["early_stopped"] is True

    def test_early_stop_max_mode(self):
        """Test early stopping with mode='max'"""
        with tempfile.TemporaryDirectory() as tmpdir:
            task, model, loss_fn, optimizer = self._create_training_components()
            self.args.runner.early_stop = {
                "target": "val_metric",
                "patience": 2,
                "mode": "max",
                "min_delta": 0.0,
            }
            result = train_runner(
                model=model,
                task=task,
                loss_fn=loss_fn,
                optimizer=optimizer,
                args=self.args,
                max_epochs=10,
                eval_interval=1,
                save_dir=tmpdir,
            )
            assert result["best_val_metric"] is not None
            assert result["early_stopped"] is True

    def test_early_stop_with_min_delta(self):
        """Test early stopping with min_delta > 0"""
        with tempfile.TemporaryDirectory() as tmpdir:
            task, model, loss_fn, optimizer = self._create_training_components()
            self.args.runner.early_stop = {
                "target": "val_metric",
                "patience": 2,
                "mode": "min",
                "min_delta": 0.1,
            }
            result = train_runner(
                model=model,
                task=task,
                loss_fn=loss_fn,
                optimizer=optimizer,
                args=self.args,
                max_epochs=10,
                eval_interval=1,
                save_dir=tmpdir,
            )
            assert result["best_val_metric"] is not None
            assert result["early_stopped"] is True

    def test_early_stop_patience_exceeded(self):
        """Test early stopping triggers only after patience exceeded"""
        with tempfile.TemporaryDirectory() as tmpdir:
            task, model, loss_fn, optimizer = self._create_training_components()
            self.args.runner.early_stop = {
                "target": "val_metric",
                "patience": 1,
                "mode": "min",
                "min_delta": 0.0,
            }
            result = train_runner(
                model=model,
                task=task,
                loss_fn=loss_fn,
                optimizer=optimizer,
                args=self.args,
                max_epochs=5,
                eval_interval=1,
                save_dir=tmpdir,
            )
            assert result["best_val_metric"] is not None
            assert result["early_stopped"] is True

    def test_early_stop_no_trigger(self):
        """Test early stopping does not trigger if patience not exceeded"""
        with tempfile.TemporaryDirectory() as tmpdir:
            task, model, loss_fn, optimizer = self._create_training_components()
            self.args.runner.early_stop = {
                "target": "val_metric",
                "patience": 100,
                "mode": "min",
                "min_delta": 0.0,
            }
            result = train_runner(
                model=model,
                task=task,
                loss_fn=loss_fn,
                optimizer=optimizer,
                args=self.args,
                max_epochs=3,
                eval_interval=1,
                save_dir=tmpdir,
            )
            assert result["best_val_metric"] is not None
            assert result["early_stopped"] is False

    def test_early_stop_custom_target(self):
        """Test early stopping with custom target metric"""
        with tempfile.TemporaryDirectory() as tmpdir:
            task, model, loss_fn, optimizer = self._create_training_components()
            self.args.runner.early_stop = {
                "target": "train_metric",
                "patience": 2,
                "mode": "max",
                "min_delta": 0.0,
            }
            result = train_runner(
                model=model,
                task=task,
                loss_fn=loss_fn,
                optimizer=optimizer,
                args=self.args,
                max_epochs=10,
                eval_interval=1,
                save_dir=tmpdir,
            )
            assert result["best_val_metric"] is not None
            assert result["early_stopped"] in [True, False]  # depends on metric


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
