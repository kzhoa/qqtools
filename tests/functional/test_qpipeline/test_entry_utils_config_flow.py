import pytest
import torch

import qqtools as qt
from qqtools.plugins.qpipeline.cmd_args import merge_basic_args, str2bool
from qqtools.plugins.qpipeline.entry_utils.loss import parse_loss_name, prepare_loss
from qqtools.plugins.qpipeline.entry_utils.optimizer import getCanonicalName, prepare_optimizer
from qqtools.plugins.qpipeline.entry_utils.scheduler import (
    SchedulerConfig,
    WarmupConfig,
    get_lambda_lr,
    prepare_scheduler,
)


def test_str2bool_valid_and_invalid():
    assert str2bool("true") is True
    assert str2bool("No") is False
    with pytest.raises(Exception):
        str2bool("not_bool")


def test_merge_basic_args_reads_yaml_and_keeps_extra(cmd_args_from_yaml):
    cmd_args = cmd_args_from_yaml("epoch_minimal.yaml", extra_flag="hello")
    args = merge_basic_args(cmd_args)

    assert args.runner.max_epochs == 3
    assert args.runner.run_mode == "epoch"
    assert args.extra_flag == "hello"


def test_optimizer_name_invalid_raises_keyerror():
    with pytest.raises(KeyError):
        getCanonicalName("nonexistent_optim")


def test_prepare_optimizer_invalid_optim_raises(tiny_model):
    args = qt.qDict({"optim": {"optimizer": "invalid", "optimizer_params": {"lr": 1.0e-3}}})
    with pytest.raises(KeyError):
        prepare_optimizer(args, tiny_model)


def test_parse_loss_name_unknown_raises():
    with pytest.raises(NotImplementedError):
        parse_loss_name("unknown_loss")


def test_prepare_loss_comboloss_success():
    args = qt.qDict(
        {
            "distributed": False,
            "optim": {
                "loss": "comboloss",
                "loss_params": {
                    "energy": ["mse", 1.0],
                    "force": ["l1", 0.5],
                },
            },
        }
    )
    loss_fn = prepare_loss(args)
    assert callable(loss_fn)


def test_prepare_loss_comboloss_none_loss_params_raises():
    args = qt.qDict(
        {
            "distributed": False,
            "optim": {
                "loss": "comboloss",
                "loss_params": None,
            },
        }
    )

    with pytest.raises(ValueError, match="optim.loss_params is required"):
        prepare_loss(args)


def test_warmup_config_invalid_factor_raises():
    with pytest.raises(ValueError):
        WarmupConfig(steps=10, epochs=0, initial_factor=1.2)


def test_scheduler_config_invalid_name_raises():
    with pytest.raises(ValueError):
        SchedulerConfig(name="invalid", params={})


def test_get_lambda_lr_invalid_expression_raises():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.tensor([1.0]))], lr=0.1)
    with pytest.raises(Exception):
        get_lambda_lr({"lr_lambda": "lambda epoch: bad + 1"}, optimizer)


def test_prepare_scheduler_unknown_scheduler_raises(base_args, tiny_model):
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)
    args = base_args.copy()
    args.optim = qt.qDict(
        {
            "scheduler": "unknown",
            "scheduler_params": qt.qDict(),
            "warmup_params": qt.qDict(),
        }
    )

    with pytest.raises(ValueError):
        prepare_scheduler(args, optimizer)


def test_prepare_scheduler_lambda_scheduler_success(base_args, tiny_model):
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)
    args = base_args.copy()
    args.optim = qt.qDict(
        {
            "scheduler": "lambda",
            "scheduler_params": qt.qDict({"lr_lambda": "lambda epoch: 0.95 ** epoch"}),
            "warmup_params": qt.qDict({"warmup_steps": 2, "warmup_factor": 0.2}),
        }
    )

    scheduler = prepare_scheduler(args, optimizer)
    scheduler.step_warmup()
    scheduler.step_warmup()

    optimizer.zero_grad()
    tiny_model(torch.randn(2, 6)).sum().backward()
    optimizer.step()

    scheduler.step_main(metrics=1.0)
    assert scheduler is not None
    assert hasattr(scheduler, "main_scheduler")
