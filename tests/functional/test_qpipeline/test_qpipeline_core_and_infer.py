from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

import qqtools as qt
import qqtools.plugins.qpipeline.qpipeline as qpipeline_module
import qqtools.plugins.qpipeline.runner.runner as runner_module
import qqtools.plugins.qpipeline.runner.runner_utils.progress as progress_module
from qqtools.plugins.qpipeline.entry import create_pipeline_class
from qqtools.plugins.qpipeline.qpipeline import prepare_logdir, qPipeline
from qqtools.plugins.qpipeline.runner.runner import evaluate_runner, infer_runner
from qqtools.plugins.qpipeline.task.qtask import OPTIONAL_METHODS, TASK_LIFECYCLE_HOOKS, qTaskBase


def test_prepare_logdir_writes_config_yaml(tmp_path):
    args = qt.qDict(
        {
            "log_dir": str(tmp_path / "run_a"),
            "config_file": None,
            "ckp_file": None,
            "seed": 1,
        }
    )

    prepare_logdir(args)
    assert Path(args.config_file).exists()
    assert Path(args.config_file).name == "config.yaml"


def test_prepare_logdir_writes_ckp_recover_yaml(tmp_path):
    args = qt.qDict(
        {
            "log_dir": str(tmp_path / "run_b"),
            "config_file": None,
            "ckp_file": str(tmp_path / "some_ckp.pt"),
            "seed": 1,
        }
    )

    prepare_logdir(args)
    assert Path(args.config_file).exists()
    assert Path(args.config_file).name == "config_ckprecover.yaml"


def test_create_pipeline_class_injects_prepare_methods():
    def _prepare_model(_args):
        return "model"

    def _prepare_task(_args):
        return "task"

    cls = create_pipeline_class(_prepare_model, _prepare_task)
    assert cls.prepare_model(None) == "model"
    assert cls.prepare_task(None) == "task"


def test_task_lifecycle_hooks_match_optional_methods_contract():
    lifecycle_hooks_in_optional_methods = {name for name in OPTIONAL_METHODS if name in TASK_LIFECYCLE_HOOKS}
    assert lifecycle_hooks_in_optional_methods == set(TASK_LIFECYCLE_HOOKS)


def test_prepare_training_session_registers_exact_task_lifecycle_hooks(monkeypatch, tmp_path):
    class LifecycleTask(qTaskBase):
        def __init__(self):
            super().__init__()
            self.train_loader = []
            self.val_loader = []
            self.test_loader = []
            self.meta = {}

        def batch_forward(self, model, batch_data):
            return {}

        def batch_loss(self, out, batch_data):
            return {"loss": (torch.tensor(0.0), 1)}

        def batch_metric(self, out, batch_data):
            return {"metric": (torch.tensor(0.0), 1)}

        def post_metrics_to_value(self, result):
            return 0.0

        def on_epoch_start(self, context):
            return None

        def on_epoch_end(self, context):
            return None

        def on_train_batch_end(self, context):
            return None

        def on_validation_end(self, context):
            return None

        def on_early_stop(self, context):
            return None

    class DummyProgressTracker:
        def __init__(self, logger, print_freq, render_type=None, rank=0):
            self.logger = logger
            self.print_freq = print_freq
            self.render_type = render_type
            self.rank = rank

        def on_epoch_start(self, context):
            return None

        def on_progress_tick(self, context):
            return None

        def on_table_update(self, context):
            return None

        def on_epoch_end(self, context):
            return None

        def on_eval_start(self, context):
            return None

        def on_eval_end(self, context):
            return None

    monkeypatch.setattr(runner_module, "ProgressTracker", DummyProgressTracker)

    task = LifecycleTask()
    agent = Mock()
    config = qt.qDict(
        {
            "ckp_file": None,
            "print_freq": 1,
            "render_type": None,
            "rank": 0,
            "use_profiler": False,
        }
    )

    runner_module._prepare_training_session(
        config=config,
        save_dir=str(tmp_path),
        logger=Mock(),
        sheet_logger=None,
        checkpoint_manager=Mock(),
        agent=agent,
        early_stopper=Mock(),
        eval_summary_listener=Mock(on_validation_end=Mock()),
        early_stop_listener=Mock(on_validation_end=Mock()),
        checkpoint_listener=Mock(on_checkpoint_request=Mock()),
        effective_scheduler=None,
        device=torch.device("cpu"),
        model=Mock(),
        task=task,
        optimizer=Mock(),
        ema_model=None,
        log_granularity=None,
    )

    registered_task_lifecycle_hooks = {
        call.args[0] for call in agent.add_listener.call_args_list if getattr(call.args[1], "__self__", None) is task
    }
    assert registered_task_lifecycle_hooks == set(TASK_LIFECYCLE_HOOKS)


def test_infer_runner_without_checkpoint(base_args, tiny_task, tiny_model):
    results = infer_runner(
        model=tiny_model,
        task=tiny_task,
        dataloader=tiny_task.test_loader,
        args=base_args,
        distributed=False,
    )
    assert results is not None
    assert "pred" in results
    assert results["pred"].shape[0] == len(tiny_task.test_loader.dataset)
    assert "labels" in results


def test_infer_runner_distributed_non_zero_rank_returns_none(monkeypatch, base_args, tiny_task, tiny_model):
    args = base_args.copy()
    args.rank = 1
    monkeypatch.setattr(runner_module.qt.qdist, "get_rank", lambda: 1)
    results = infer_runner(
        model=tiny_model,
        task=tiny_task,
        dataloader=tiny_task.test_loader,
        args=args,
        distributed=True,
    )
    assert results is None


def test_infer_runner_args_required(tiny_task, tiny_model):
    with pytest.raises(ValueError):
        infer_runner(
            model=tiny_model,
            task=tiny_task,
            dataloader=tiny_task.test_loader,
            args=None,
            distributed=False,
        )


def test_evaluate_once_runner_returns_prefixed_metrics(base_args, tiny_task, tiny_model):
    results = evaluate_runner(
        model=tiny_model,
        task=tiny_task,
        dataloader=tiny_task.test_loader,
        args=base_args,
        prefix="test",
        return_outputs=False,
    )

    assert results is not None
    assert "test_metric" in results
    assert "test_mse" in results
    assert "pred" not in results


def test_evaluate_once_runner_return_outputs_includes_metrics_and_predictions(base_args, tiny_task, tiny_model):
    results = evaluate_runner(
        model=tiny_model,
        task=tiny_task,
        dataloader=tiny_task.test_loader,
        args=base_args,
        prefix="test",
        return_outputs=True,
    )

    assert results is not None
    assert "test_metric" in results
    assert "test_mse" in results
    assert "pred" in results
    assert "labels" in results
    assert results["pred"].shape[0] == len(tiny_task.test_loader.dataset)


def test_qpipeline_uses_args_test_to_enter_test_mode(base_args, tiny_task, tiny_model):
    args = base_args.copy()
    args.test = True
    pipeline_cls = create_pipeline_class(lambda _args: tiny_model, lambda _args: tiny_task)

    pipeline = pipeline_cls(args)

    assert pipeline.mode == "test"
    assert pipeline.loss_fn is None


def test_qpipeline_mode_overrides_args_test(base_args, tiny_task, tiny_model):
    args = base_args.copy()
    args.test = True
    pipeline_cls = create_pipeline_class(lambda _args: tiny_model, lambda _args: tiny_task)

    pipeline = pipeline_cls(args, mode="test")
    assert pipeline.mode == "test"

    args2 = base_args.copy()
    args2.test = False
    pipeline2 = pipeline_cls(args2, mode="test")
    assert pipeline2.mode == "test"


def test_qpipeline_test_mode_evaluate_once_defaults_to_test_loader(base_args, tiny_task, tiny_model):
    args = base_args.copy()
    args.test = True
    pipeline_cls = create_pipeline_class(lambda _args: tiny_model, lambda _args: tiny_task)
    pipeline = pipeline_cls(args)

    results = pipeline.evaluate_once()

    assert results is not None
    assert "test_metric" in results
    assert "test_mse" in results


def test_qpipeline_infer_returns_outputs(base_args, tiny_task, tiny_model):
    args = base_args.copy()
    args.test = True
    pipeline_cls = create_pipeline_class(lambda _args: tiny_model, lambda _args: tiny_task)
    pipeline = pipeline_cls(args)

    results = pipeline.infer()

    assert results is not None
    assert "pred" in results
    assert "labels" in results


def test_qpipeline_fit_rejects_test_mode(base_args, tiny_task, tiny_model):
    args = base_args.copy()
    args.test = True
    pipeline_cls = create_pipeline_class(lambda _args: tiny_model, lambda _args: tiny_task)
    pipeline = pipeline_cls(args)

    with pytest.raises(RuntimeError, match="train mode"):
        pipeline.fit()


def test_progress_tracker_auto_mode_logs_info(monkeypatch):
    logger = Mock()

    class _DummyStrategy:
        def on_epoch_start(self, context):
            return None

        def on_progress_tick(self, context):
            return None

        def on_table_update(self, context):
            return None

        def on_epoch_end(self, context):
            return None

        def on_run_end(self):
            return None

        def on_eval_start(self, context):
            return None

        def on_eval_end(self, context):
            return None

    monkeypatch.setattr(progress_module, "HAS_RICH", True)
    monkeypatch.setattr(progress_module, "HAS_TQDM", True)
    monkeypatch.setattr(progress_module, "create_progress_strategy", lambda mode, logger, freq: _DummyStrategy())

    progress_module.ProgressTracker(logger=logger, print_freq=5, render_type="auto")

    logger.info.assert_called_with("Mode auto -> rich")
    logger.warning.assert_not_called()


def test_progress_tracker_explicit_unavailable_mode_logs_warning(monkeypatch):
    logger = Mock()

    class _DummyStrategy:
        def on_epoch_start(self, context):
            return None

        def on_progress_tick(self, context):
            return None

        def on_table_update(self, context):
            return None

        def on_epoch_end(self, context):
            return None

        def on_run_end(self):
            return None

        def on_eval_start(self, context):
            return None

        def on_eval_end(self, context):
            return None

    monkeypatch.setattr(progress_module, "HAS_RICH", False)
    monkeypatch.setattr(progress_module, "HAS_TQDM", True)
    monkeypatch.setattr(progress_module, "create_progress_strategy", lambda mode, logger, freq: _DummyStrategy())

    progress_module.ProgressTracker(logger=logger, print_freq=5, render_type="rich")

    logger.warning.assert_called_with("Mode rich -> tqdm")


def test_train_runner_progress_tick_contains_batch_time(monkeypatch, base_args, tiny_task, tiny_model, tiny_optimizer):
    captured_progress_contexts = []

    class _CaptureProgressTracker:
        def __init__(self, logger, print_freq=10, render_type=None, rank=0):
            return None

        def on_epoch_start(self, context):
            return None

        def on_progress_tick(self, context):
            captured_progress_contexts.append(context)

        def on_table_update(self, context):
            return None

        def on_epoch_end(self, context):
            return None

        def on_run_end(self):
            return None

        def on_eval_start(self, context):
            return None

        def on_eval_end(self, context):
            return None

    monkeypatch.setattr(runner_module, "ProgressTracker", _CaptureProgressTracker)

    runner_module.train_runner(
        model=tiny_model,
        task=tiny_task,
        loss_fn=torch.nn.MSELoss(),
        optimizer=tiny_optimizer,
        args=base_args,
        max_epochs=1,
        max_steps=1,
        run_mode="step",
        eval_interval=100,
        save_dir=str(base_args.log_dir),
        print_freq=1,
    )

    assert len(captured_progress_contexts) > 0

    first_context = captured_progress_contexts[0]
    assert first_context.batch_metrics is not None
    assert "batch_time" in first_context.batch_metrics
    assert isinstance(first_context.batch_metrics["batch_time"], float)
    assert first_context.batch_metrics["batch_time"] >= 0.0

    assert first_context.avg_bank is not None
    assert "batch_time" in first_context.avg_bank
    assert isinstance(first_context.avg_bank["batch_time"], (int, float))


def test_eval_emits_progress_tick_for_val_and_test_stages(tiny_task, tiny_model, tiny_optimizer):
    config = runner_module.RunConfig(
        render_type="plain",
        distributed=False,
        rank=0,
        max_epochs=1,
        max_steps=1,
        device=torch.device("cpu"),
    )

    agent = runner_module.RunningAgent(
        model=tiny_model,
        task=tiny_task,
        loss_fn=torch.nn.MSELoss(),
        optimizer=tiny_optimizer,
        config=config,
        device=torch.device("cpu"),
        logger=Mock(),
    )

    captured_stages = []

    def _capture_progress_tick(context):
        captured_stages.append(context.runner.stage)

    agent.add_listener("on_progress_tick", _capture_progress_tick)

    agent._evaluate_loader(tiny_model, tiny_task.val_loader, prefix="val", stage="val")
    agent._evaluate_loader(tiny_model, tiny_task.test_loader, prefix="test", stage="test")

    val_ticks = [stage for stage in captured_stages if stage == "val"]
    test_ticks = [stage for stage in captured_stages if stage == "test"]

    assert len(val_ticks) == len(tiny_task.val_loader)
    assert len(test_ticks) == len(tiny_task.test_loader)


def test_qpipeline_fit_forwards_accum_grad(monkeypatch, base_args):
    captured_kwargs = {}

    class DummyPipeline(qPipeline):
        @staticmethod
        def prepare_task(args):
            raise NotImplementedError

        @staticmethod
        def prepare_model(args):
            raise NotImplementedError

    def fake_train_runner(**kwargs):
        captured_kwargs.update(kwargs)
        return {"final_step": 0}

    args = base_args.copy()
    args.runner.update(
        {
            "run_mode": "step",
            "max_steps": 12,
            "eval_interval": 3,
            "save_interval": 6,
            "accum_grad": 4,
        }
    )

    monkeypatch.setattr(qpipeline_module, "train_runner", fake_train_runner)

    pipeline = object.__new__(DummyPipeline)
    pipeline.args = args
    pipeline.mode = "train"
    pipeline.model = Mock()
    pipeline.task = Mock()
    pipeline.loss_fn = Mock()
    pipeline.extra_ckp_caches = {"cache_key": "cache_value"}
    pipeline.ema_model = Mock()
    pipeline.logger = Mock()

    pipeline.fit(use_profiler=True)

    assert captured_kwargs["accum_grad"] == 4
    assert captured_kwargs["run_mode"] == "step"
    assert captured_kwargs["max_steps"] == 12
    assert captured_kwargs["eval_interval"] == 3
    assert captured_kwargs["save_interval"] == 6
    assert captured_kwargs["use_profiler"] is True
