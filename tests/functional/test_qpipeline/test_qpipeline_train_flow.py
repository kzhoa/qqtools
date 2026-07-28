from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import qqtools.plugins.qpipeline.runner.runner as runner_module
from qqtools.plugins.qpipeline.runner.runner import train_runner
from qqtools.plugins.qpipeline.task.qtask import qTaskBase


class _LifecycleBaseTask(qTaskBase):
    def __init__(self):
        super().__init__()
        x = torch.randn(24, 6)
        y = torch.randn(24, 1)
        self.train_loader = DataLoader(TensorDataset(x[:16], y[:16]), batch_size=8, shuffle=False)
        self.val_loader = DataLoader(TensorDataset(x[16:20], y[16:20]), batch_size=4, shuffle=False)
        self.test_loader = DataLoader(TensorDataset(x[20:], y[20:]), batch_size=4, shuffle=False)
        self.meta = {}

    @staticmethod
    def pre_batch_forward(batch_data):
        x, y = batch_data
        return {"x": x, "y": y}

    def batch_forward(self, model, batch_data):
        pred = model(batch_data["x"])
        return {"pred": pred, "target": batch_data["y"]}

    def batch_loss(self, out, batch_data, loss_fn=None):
        criterion = torch.nn.MSELoss() if loss_fn is None else loss_fn
        loss = criterion(out["pred"], out["target"])
        return {"loss": (loss, out["pred"].shape[0])}

    def batch_metric(self, out, batch_data):
        metric = torch.nn.MSELoss()(out["pred"], out["target"])
        return {"mse": (metric, out["pred"].shape[0])}

    def post_metrics_to_value(self, result):
        return result.get("val_mse", result.get("val_metric", 0.0))


def test_train_runner_epoch_mode_uses_max_epochs(base_args, tiny_task, tiny_model):
    args = base_args.copy()
    args.runner.early_stop.patience = 999

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)

    result = train_runner(
        model=tiny_model,
        task=tiny_task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        args=args,
        run_mode="epoch",
        max_epochs=2,
        eval_interval=1,
        save_dir=args.log_dir,
        print_freq=5,
    )

    assert result["final_epoch"] == 2
    assert result["best_monitored_metric"] is not None


def test_train_runner_step_mode_dual_boundaries(base_args, tiny_task, tiny_model):
    args = base_args.copy()
    args.runner.early_stop.patience = 999

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)

    result = train_runner(
        model=tiny_model,
        task=tiny_task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        args=args,
        run_mode="step",
        max_epochs=3,
        max_steps=7,
        eval_interval=2,
        save_interval=3,
        save_dir=args.log_dir,
        print_freq=3,
    )

    assert result["final_step"] >= 6
    assert result["final_epoch"] <= 3
    assert result["best_monitored_metric"] is not None
    log_text = (Path(args.log_dir) / "debug.log").read_text(encoding="utf-8")
    assert "secondary stopping boundary" in log_text


def test_train_runner_step_mode_epoch_summary_logs_train_metrics_without_eval(
    base_args, tiny_task, tiny_model
):
    args = base_args.copy()
    args.runner.early_stop.patience = 999

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)

    train_runner(
        model=tiny_model,
        task=tiny_task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        args=args,
        run_mode="step",
        max_epochs=2,
        max_steps=None,
        eval_interval=9999999,
        save_dir=args.log_dir,
        print_freq=3,
    )

    log_text = (Path(args.log_dir) / "debug.log").read_text(encoding="utf-8")
    assert "--- Epoch 0 Results ---" in log_text
    assert "[train] loss:" in log_text
    assert "metric:" in log_text
    assert "[val] metric: n/a source=missing" in log_text
    assert "[test] metric: n/a source=missing" in log_text


def test_train_runner_step_mode_infers_max_steps_from_max_epochs(base_args, tiny_task, tiny_model):
    args = base_args.copy()
    args.runner.early_stop.patience = 999

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)
    expected_max_steps = len(tiny_task.train_loader) * 3

    result = train_runner(
        model=tiny_model,
        task=tiny_task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        args=args,
        run_mode="step",
        max_epochs=3,
        max_steps=None,
        eval_interval=2,
        save_dir=args.log_dir,
        print_freq=3,
    )

    assert result["terminal_event"]["reason"] == "max_steps"
    assert result["final_step"] == expected_max_steps
    log_text = (Path(args.log_dir) / "debug.log").read_text(encoding="utf-8")
    assert f"inferred max_steps={expected_max_steps}" in log_text
    assert "secondary stopping boundary" in log_text


def test_train_runner_step_mode_infers_max_steps_with_accum_grad(base_args, tiny_task, tiny_model):
    args = base_args.copy()
    args.runner.early_stop.patience = 999

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)
    train_loader_batches = len(tiny_task.train_loader)
    accum_grad = 2
    expected_max_steps = ((train_loader_batches + accum_grad - 1) // accum_grad) * 3

    result = train_runner(
        model=tiny_model,
        task=tiny_task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        args=args,
        run_mode="step",
        max_epochs=3,
        max_steps=None,
        eval_interval=2,
        accum_grad=accum_grad,
        save_dir=args.log_dir,
        print_freq=3,
    )

    assert result["terminal_event"]["reason"] == "max_steps"
    assert result["final_step"] == expected_max_steps
    assert result["final_epoch"] <= 3
    log_text = (Path(args.log_dir) / "debug.log").read_text(encoding="utf-8")
    assert f"inferred max_steps={expected_max_steps}" in log_text
    assert "accum_grad=2" in log_text
    assert "secondary stopping boundary" in log_text


def test_train_runner_step_mode_infer_requires_train_loader_len(base_args, tiny_model):
    args = base_args.copy()
    args.runner.early_stop.patience = 999

    class NoLenLoader:
        def __iter__(self):
            return iter(())

    class NoLenTask:
        train_loader = NoLenLoader()
        val_loader = None
        test_loader = None

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)

    with pytest.raises(
        ValueError,
        match="provide max_steps explicitly or ensure len\\(task.train_loader\\) is available as a positive integer",
    ):
        train_runner(
            model=tiny_model,
            task=NoLenTask(),
            loss_fn=loss_fn,
            optimizer=optimizer,
            args=args,
            run_mode="step",
            max_epochs=3,
            max_steps=None,
            eval_interval=2,
            save_dir=args.log_dir,
            print_freq=3,
        )


def test_train_runner_regular_checkpoint_saving(base_args, tiny_task, tiny_model, tmp_path):
    save_dir = tmp_path / "checkpoints"
    args = base_args.copy()
    args.runner.early_stop.patience = 999

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)

    train_runner(
        model=tiny_model,
        task=tiny_task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        args=args,
        run_mode="step",
        max_steps=6,
        eval_interval=3,
        save_interval=2,
        save_dir=str(save_dir),
    )

    regular_ckpts = list(Path(save_dir).glob("epoch*.pt"))
    best_ckpts = list(Path(save_dir).glob("best_*.pt"))

    assert len(regular_ckpts) == 1
    assert len(best_ckpts) >= 1


def test_train_runner_ckp_file_takes_effect(base_args, tiny_task, tiny_model, tmp_path):
    save_dir = tmp_path / "resume"
    args = base_args.copy()
    args.runner.early_stop.patience = 999

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)

    first = train_runner(
        model=tiny_model,
        task=tiny_task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        args=args,
        run_mode="step",
        max_steps=5,
        eval_interval=2,
        save_interval=2,
        save_dir=str(save_dir),
    )

    ckpts = sorted(Path(save_dir).glob("*.pt"))
    assert ckpts

    model2 = type(tiny_model)()
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1.0e-3)
    args2 = base_args.copy()
    args2.runner.early_stop.patience = 999
    args2.ckp_file = str(ckpts[-1])
    args2.init_file = str(tmp_path / "unused_init_file.pt")

    second = train_runner(
        model=model2,
        task=tiny_task,
        loss_fn=loss_fn,
        optimizer=optimizer2,
        args=args2,
        run_mode="step",
        max_steps=8,
        eval_interval=2,
        save_dir=str(save_dir),
    )

    assert second["final_step"] >= 4


def test_train_runner_missing_early_stop_field_uses_defaults(base_args, tiny_task, tiny_model):
    args = base_args.copy()
    args.runner = SimpleNamespace(checkpoint={})

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)

    result = train_runner(
        model=tiny_model,
        task=tiny_task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        args=args,
        run_mode="epoch",
        max_epochs=2,
        eval_interval=1,
        save_dir=args.log_dir,
        print_freq=5,
    )

    assert result["final_epoch"] == 2
    assert result["best_monitored_metric"] is not None


def test_train_runner_missing_checkpoint_and_early_stop_uses_defaults(base_args, tiny_task, tiny_model):
    args = base_args.copy()
    args.runner = SimpleNamespace()

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)

    result = train_runner(
        model=tiny_model,
        task=tiny_task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        args=args,
        run_mode="epoch",
        max_epochs=2,
        eval_interval=1,
        save_dir=args.log_dir,
        print_freq=5,
    )

    assert result["final_epoch"] == 2
    assert result["best_monitored_metric"] is not None


def test_train_runner_none_checkpoint_uses_defaults(base_args, tiny_task, tiny_model):
    args = base_args.copy()
    args.runner = SimpleNamespace(checkpoint=None, early_stop={"patience": 999})

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)

    result = train_runner(
        model=tiny_model,
        task=tiny_task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        args=args,
        run_mode="epoch",
        max_epochs=2,
        eval_interval=1,
        save_dir=args.log_dir,
        print_freq=5,
    )

    assert result["final_epoch"] == 2
    assert result["best_monitored_metric"] is not None


def test_train_runner_none_early_stop_uses_defaults(base_args, tiny_task, tiny_model):
    args = base_args.copy()
    args.runner = SimpleNamespace(checkpoint={}, early_stop=None)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)

    result = train_runner(
        model=tiny_model,
        task=tiny_task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        args=args,
        run_mode="epoch",
        max_epochs=2,
        eval_interval=1,
        save_dir=args.log_dir,
        print_freq=5,
    )

    assert result["final_epoch"] == 2
    assert result["best_monitored_metric"] is not None


def test_train_runner_none_run_mode_raises_value_error(base_args, tiny_task, tiny_model):
    args = base_args.copy()
    args.runner = SimpleNamespace(checkpoint={}, early_stop={"patience": 999})

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)

    with pytest.raises(ValueError, match="run_mode cannot be None"):
        train_runner(
            model=tiny_model,
            task=tiny_task,
            loss_fn=loss_fn,
            optimizer=optimizer,
            args=args,
            run_mode=None,
            max_epochs=2,
            eval_interval=1,
            save_dir=args.log_dir,
            print_freq=5,
        )


def test_train_runner_none_eval_interval_falls_back_to_one(base_args, tiny_task, tiny_model):
    args = base_args.copy()
    args.runner = SimpleNamespace(checkpoint={}, early_stop={"patience": 999})

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)

    result = train_runner(
        model=tiny_model,
        task=tiny_task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        args=args,
        run_mode="epoch",
        max_epochs=2,
        eval_interval=None,
        save_dir=args.log_dir,
        print_freq=5,
    )

    assert result["final_epoch"] == 2
    assert result["best_monitored_metric"] is not None


def test_train_runner_none_render_type_falls_back_to_auto(base_args, tiny_task, tiny_model):
    args = base_args.copy()
    args.runner = SimpleNamespace(checkpoint={}, early_stop={"patience": 999})
    args.render_type = None

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)

    result = train_runner(
        model=tiny_model,
        task=tiny_task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        args=args,
        run_mode="epoch",
        max_epochs=2,
        eval_interval=1,
        save_dir=args.log_dir,
        print_freq=5,
    )

    assert result["final_epoch"] == 2
    assert result["best_monitored_metric"] is not None


def test_train_runner_exception_still_cleans_progress_tracker(base_args, tiny_task, tiny_model, monkeypatch):
    tracker_instances = []

    class DummyProgressTracker:
        def __init__(self, logger, print_freq, render_type=None, rank=0):
            self.on_run_end_called = False
            tracker_instances.append(self)

        def on_epoch_start(self, context):
            return None

        def on_progress_tick(self, context):
            return None

        def on_table_update(self, context):
            return None

        def on_batch_end(self, context):
            return None

        def on_epoch_end(self, context):
            return None

        def on_eval_start(self, context):
            return None

        def on_eval_end(self, context):
            return None

        def on_run_end(self):
            self.on_run_end_called = True
            raise RuntimeError("cleanup failure")

    def _failing_run(self):
        self._start_new_epoch()
        raise ValueError("run failure")

    monkeypatch.setattr(runner_module, "ProgressTracker", DummyProgressTracker)
    monkeypatch.setattr(runner_module.RunningAgent, "run", _failing_run)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)

    with pytest.raises(ValueError, match="run failure"):
        train_runner(
            model=tiny_model,
            task=tiny_task,
            loss_fn=loss_fn,
            optimizer=optimizer,
            args=base_args,
            run_mode="epoch",
            max_epochs=2,
            eval_interval=1,
            save_dir=base_args.log_dir,
            print_freq=5,
        )

    assert tracker_instances
    assert tracker_instances[0].on_run_end_called is True


def test_train_runner_bridges_task_lifecycle_hooks(base_args, tiny_model):
    received = {}

    class LifecycleTask(_LifecycleBaseTask):
        def on_epoch_start(self, context):
            received["on_epoch_start"] = context

        def on_epoch_end(self, context):
            received["on_epoch_end"] = context
            received["on_epoch_end_epoch"] = context.runner.run_state.epoch

        def on_train_batch_end(self, context):
            received.setdefault("on_train_batch_end", []).append(context)

        def on_validation_end(self, context):
            received["on_validation_end"] = context

    args = base_args.copy()
    args.runner.early_stop.patience = 999
    task = LifecycleTask()
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)

    result = train_runner(
        model=tiny_model,
        task=task,
        loss_fn=torch.nn.MSELoss(),
        optimizer=optimizer,
        args=args,
        run_mode="epoch",
        max_epochs=1,
        eval_interval=1,
        save_dir=args.log_dir,
        print_freq=1,
    )

    assert result["final_epoch"] == 1
    assert received["on_epoch_start"].runner.stage == "train"
    assert not hasattr(received["on_epoch_start"], "total_batches")
    assert received["on_epoch_end_epoch"] == 0
    assert received["on_epoch_end"].runner.run_state.epoch == 1
    assert received["on_train_batch_end"]
    first_train_batch = received["on_train_batch_end"][0]
    assert first_train_batch.runner.stage == "train"
    assert "loss" in first_train_batch.batch_metrics
    assert first_train_batch.lr is not None
    validation_context = received["on_validation_end"]
    assert validation_context.runner.stage is None
    assert validation_context.eval_results is not None
    assert "val_metric" in validation_context.eval_results
    assert validation_context.signal is not None


def test_train_runner_does_not_bridge_legacy_task_lifecycle_names(base_args, tiny_model):
    calls = []

    class LegacyLifecycleTask(_LifecycleBaseTask):
        def onEpochStart(self, context):
            calls.append(("onEpochStart", context))

        def onEpochEnd(self, context):
            calls.append(("onEpochEnd", context))

        def onBatchEnd(self, context):
            calls.append(("onBatchEnd", context))

    args = base_args.copy()
    args.runner.early_stop.patience = 999
    task = LegacyLifecycleTask()
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)

    train_runner(
        model=tiny_model,
        task=task,
        loss_fn=torch.nn.MSELoss(),
        optimizer=optimizer,
        args=args,
        run_mode="epoch",
        max_epochs=1,
        eval_interval=1,
        save_dir=args.log_dir,
        print_freq=1,
    )

    assert calls == []


def test_train_runner_bridges_task_on_early_stop(base_args, tiny_model, monkeypatch):
    received = {}

    class EarlyStopTask(_LifecycleBaseTask):
        def on_early_stop(self, context):
            received["on_early_stop"] = context

    class ImmediateEarlyStopListener:
        def __init__(self, early_stopper, target="val_metric", logger=None):
            self.early_stopper = early_stopper
            self.target = target
            self.logger = logger

        def on_validation_end(self, context):
            if context.signal is not None:
                context.signal.request_stop("test", "forced stop for lifecycle hook test")

    args = base_args.copy()
    args.runner.early_stop.patience = 999
    monkeypatch.setattr(runner_module, "EarlyStopListener", ImmediateEarlyStopListener)

    train_runner(
        model=tiny_model,
        task=EarlyStopTask(),
        loss_fn=torch.nn.MSELoss(),
        optimizer=torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3),
        args=args,
        run_mode="epoch",
        max_epochs=3,
        eval_interval=1,
        save_dir=args.log_dir,
        print_freq=1,
    )

    assert received["on_early_stop"].signal is not None
    assert received["on_early_stop"].signal.should_stop is True
