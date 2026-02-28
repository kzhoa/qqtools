from pathlib import Path

import pytest
import torch

import qqtools as qt
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
