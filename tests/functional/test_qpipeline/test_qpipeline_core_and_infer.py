from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

import qqtools as qt
import qqtools.plugins.qpipeline.runner.progress as progress_module
from qqtools.plugins.qpipeline.entry import create_pipeline_class
from qqtools.plugins.qpipeline.qpipeline import prepare_logdir
from qqtools.plugins.qpipeline.runner.runner import infer_runner


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


def test_infer_runner_without_checkpoint(base_args, tiny_task, tiny_model):
    results = infer_runner(
        model=tiny_model,
        task=tiny_task,
        dataloader=tiny_task.test_loader,
        args=base_args,
        distributed=False,
    )
    assert len(results) > 0
    assert "preds" in results[0]


def test_infer_runner_distributed_non_zero_rank_returns_empty(base_args, tiny_task, tiny_model):
    args = base_args.copy()
    args.rank = 1
    results = infer_runner(
        model=tiny_model,
        task=tiny_task,
        dataloader=tiny_task.test_loader,
        args=args,
        distributed=True,
    )
    assert results == []


def test_infer_runner_args_required(tiny_task, tiny_model):
    with pytest.raises(ValueError):
        infer_runner(
            model=tiny_model,
            task=tiny_task,
            dataloader=tiny_task.test_loader,
            args=None,
            distributed=False,
        )


def test_progress_tracker_auto_mode_logs_info(monkeypatch):
    logger = Mock()

    class _DummyStrategy:
        def on_epoch_start(self, context):
            return None

        def on_progress_tick(self, context):
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
