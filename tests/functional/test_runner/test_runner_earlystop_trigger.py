"""
Test cases for early stopping trigger timing in train_runner
"""

import argparse
import tempfile
import logging
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from qqtools.plugins.qpipeline.runner.runner import train_runner
from .conftest import SimpleModel, SimpleTask


class DummyTask(SimpleTask):
    """A task that produces a fixed validation metric sequence for early stop simulation."""

    def __init__(self, val_metric_seq, num_samples=50, num_features=10):
        super().__init__(num_samples=num_samples, num_features=num_features)
        self._val_metric_seq = val_metric_seq
        self._eval_count = 0

    def post_metrics_to_value(self, metrics):
        # 根据当前 epoch 返回对应的 val_metric_seq 值，避免提前停滞
        epoch = getattr(self, "_last_epoch", 0)
        # 尝试从 metrics 或 RunningState 获取当前 epoch
        if "epoch" in metrics:
            epoch = metrics["epoch"]
        elif hasattr(self, "state") and hasattr(self.state, "epoch"):
            epoch = self.state.epoch
        # 取序列值
        if epoch < len(self._val_metric_seq):
            val = self._val_metric_seq[epoch]
        else:
            val = self._val_metric_seq[-1]
        self._last_epoch = epoch + 1
        return val


class TestEarlyStopTrigger:
    def setup_method(self):
        """Create a temporary directory before each test."""
        self.tmpdir = tempfile.TemporaryDirectory()

    def teardown_method(self, method):
        """Cleanup logging and the temporary directory after each test."""
        # Shutdown logging BEFORE cleaning up the temp dir
        logging.shutdown()
        self.tmpdir.cleanup()

    @staticmethod
    def _make_args():
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

    def test_trigger_on_patience(self):
        """Should trigger early stop when metric doesn't improve for patience steps."""
        # Simulate val_metric: [1.0, 0.9, 0.9, 0.9, 0.9] (patience=2, mode=min)
        val_metric_seq = [1.0, 0.9, 0.9, 0.9, 0.9]
        tmpdir = self.tmpdir.name
        args = self._make_args()
        args.runner.early_stop = {
            "target": "val_metric",
            "patience": 2,
            "mode": "min",
            "min_delta": 0.0,
        }
        model = SimpleModel(input_dim=10)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        task = DummyTask(val_metric_seq)
        result = train_runner(
            model=model,
            task=task,
            loss_fn=loss_fn,
            optimizer=optimizer,
            args=args,
            max_epochs=10,
            eval_interval=1,
            save_dir=tmpdir,
        )
        assert result["early_stopped"] is True
        # Should stop after patience epochs without improvement
        assert result["final_epoch"] <= 5

    def test_no_trigger_if_improving(self):
        """Should not trigger early stop if metric keeps improving."""
        val_metric_seq = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        max_epochs = len(val_metric_seq)
        patience = max_epochs + 1
        tmpdir = self.tmpdir.name
        args = self._make_args()
        args.runner.early_stop = {
            "target": "val_metric",
            "patience": patience,
            "mode": "min",
            "min_delta": 0.0,
        }
        model = SimpleModel(input_dim=10)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        task = DummyTask(val_metric_seq)
        result = train_runner(
            model=model,
            task=task,
            loss_fn=loss_fn,
            optimizer=optimizer,
            args=args,
            max_epochs=max_epochs,
            eval_interval=1,
            save_dir=tmpdir,
        )
        assert result["early_stopped"] is False
        assert result["final_epoch"] == max_epochs

    def test_trigger_with_min_delta(self):
        """Should trigger early stop only if improvement < min_delta."""
        val_metric_seq = [1.0, 0.95, 0.94, 0.93, 0.92]
        tmpdir = self.tmpdir.name
        args = self._make_args()
        args.runner.early_stop = {
            "target": "val_metric",
            "patience": 2,
            "mode": "min",
            "min_delta": 0.05,
        }
        model = SimpleModel(input_dim=10)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        task = DummyTask(val_metric_seq)
        result = train_runner(
            model=model,
            task=task,
            loss_fn=loss_fn,
            optimizer=optimizer,
            args=args,
            max_epochs=10,
            eval_interval=1,
            save_dir=tmpdir,
        )
        assert result["early_stopped"] is True
        assert result["final_epoch"] <= 5

    def test_trigger_max_mode(self):
        """Should trigger early stop in max mode when metric doesn't increase."""
        val_metric_seq = [0.5, 0.6, 0.6, 0.6, 0.6]
        tmpdir = self.tmpdir.name
        args = self._make_args()
        args.runner.early_stop = {
            "target": "val_metric",
            "patience": 2,
            "mode": "max",
            "min_delta": 0.0,
        }
        model = SimpleModel(input_dim=10)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        task = DummyTask(val_metric_seq)
        result = train_runner(
            model=model,
            task=task,
            loss_fn=loss_fn,
            optimizer=optimizer,
            args=args,
            max_epochs=10,
            eval_interval=1,
            save_dir=tmpdir,
        )
        assert result["early_stopped"] is True
        assert result["final_epoch"] <= 5

    def test_trigger_custom_target(self):
        """Should trigger early stop on custom metric target."""

        class CustomTask(DummyTask):
            def post_metrics_to_value(self, metrics):
                # Use train_metric sequence for early stop
                if self._eval_count < len(self._val_metric_seq):
                    val = self._val_metric_seq[self._eval_count]
                else:
                    val = self._val_metric_seq[-1]
                self._eval_count += 1
                metrics["train_metric"] = val
                return val

        train_metric_seq = [0.5, 0.6, 0.6, 0.6, 0.6]
        tmpdir = self.tmpdir.name
        args = self._make_args()
        args.runner.early_stop = {
            "target": "train_metric",
            "patience": 2,
            "mode": "max",
            "min_delta": 0.0,
        }
        model = SimpleModel(input_dim=10)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        task = CustomTask(train_metric_seq)
        result = train_runner(
            model=model,
            task=task,
            loss_fn=loss_fn,
            optimizer=optimizer,
            args=args,
            max_epochs=10,
            eval_interval=1,
            save_dir=tmpdir,
        )
        assert result["early_stopped"] is True
        assert result["final_epoch"] <= 5

    def test_earlystop_patience_zero(self):
        """Test early stopping with patience=0 should raise ValueError."""
        val_metric_seq = [1.0, 1.0, 1.0]
        tmpdir = self.tmpdir.name
        args = self._make_args()
        args.runner.early_stop = {
            "target": "val_metric",
            "patience": 0,
            "mode": "min",
            "min_delta": 0.0,
        }
        model = SimpleModel(input_dim=10)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        task = DummyTask(val_metric_seq)
        with pytest.raises(ValueError, match="Patience for val_metric must be positive integer"):
            train_runner(
                model=model,
                task=task,
                loss_fn=loss_fn,
                optimizer=optimizer,
                args=args,
                max_epochs=5,
                eval_interval=1,
                save_dir=tmpdir,
            )

    def test_earlystop_negative_min_delta_raises_error(self):
        """Test early stopping with min_delta < 0 should raise a ValueError."""
        val_metric_seq = [1.0, 0.99, 0.98, 0.97]
        tmpdir = self.tmpdir.name
        args = self._make_args()
        args.runner.early_stop = {
            "target": "val_metric",
            "patience": 2,
            "mode": "min",
            "min_delta": -0.1,
        }
        model = SimpleModel(input_dim=10)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        task = DummyTask(val_metric_seq)

        with pytest.raises(ValueError, match="Delta must be non-negative"):
            train_runner(
                model=model,
                task=task,
                loss_fn=loss_fn,
                optimizer=optimizer,
                args=args,
                max_epochs=4,
                eval_interval=1,
                save_dir=tmpdir,
            )

    def test_earlystop_metric_none(self):
        """Test early stopping when metric is None should not trigger early stop."""

        class NoneMetricTask(DummyTask):
            def post_metrics_to_value(self, metrics):
                return None

        tmpdir = self.tmpdir.name
        args = self._make_args()
        args.runner.early_stop = {
            "target": "val_metric",
            "patience": 2,
            "mode": "min",
            "min_delta": 0.0,
        }
        model = SimpleModel(input_dim=10)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        task = NoneMetricTask([None] * 5)
        result = train_runner(
            model=model,
            task=task,
            loss_fn=loss_fn,
            optimizer=optimizer,
            args=args,
            max_epochs=5,
            eval_interval=1,
            save_dir=tmpdir,
        )
        assert result["early_stopped"] is False
        assert result["final_epoch"] == 5

    def test_earlystop_metric_missing(self):
        """Test early stopping when metric is missing from eval_results should not trigger early stop."""

        class MissingMetricTask(DummyTask):
            def post_metrics_to_value(self, metrics):
                return 0.5

        tmpdir = self.tmpdir.name
        args = self._make_args()
        args.runner.early_stop = {
            "target": "not_exist_metric",
            "patience": 2,
            "mode": "min",
            "min_delta": 0.0,
        }
        model = SimpleModel(input_dim=10)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        task = MissingMetricTask([0.5] * 5)
        result = train_runner(
            model=model,
            task=task,
            loss_fn=loss_fn,
            optimizer=optimizer,
            args=args,
            max_epochs=5,
            eval_interval=1,
            save_dir=tmpdir,
        )
        assert result["early_stopped"] is False
        assert result["final_epoch"] == 5

    def test_earlystop_resume_count(self):
        """Test early stopping counter after simulated resume (patience计数是否正确)。"""
        val_metric_seq = [1.0, 0.9, 0.9, 0.9, 0.9]
        tmpdir = self.tmpdir.name
        args = self._make_args()
        args.runner.early_stop = {
            "target": "val_metric",
            "patience": 2,
            "mode": "min",
            "min_delta": 0.0,
        }
        model = SimpleModel(input_dim=10)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # 第一次训练，模拟中断
        task = DummyTask(val_metric_seq)
        train_runner(
            model=model,
            task=task,
            loss_fn=loss_fn,
            optimizer=optimizer,
            args=args,
            max_epochs=3,
            eval_interval=1,
            save_dir=tmpdir,
        )
        # 第二次训练，继续patience计数
        task2 = DummyTask(val_metric_seq)
        result2 = train_runner(
            model=model,
            task=task2,
            loss_fn=loss_fn,
            optimizer=optimizer,
            args=args,
            max_epochs=5,
            eval_interval=1,
            save_dir=tmpdir,
        )
        assert result2["early_stopped"] is True
        assert result2["final_epoch"] <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
