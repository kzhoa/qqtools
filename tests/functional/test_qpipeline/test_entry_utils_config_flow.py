import pytest
import torch

import qqtools as qt
from qqtools.plugins.qpipeline.cmd_args import merge_basic_args, str2bool
from qqtools.plugins.qpipeline.entry_utils.loss import DDPMeanReducedLoss, FocalLoss, RMSELoss, parse_loss_name, prepare_loss
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


def test_merge_basic_args_reads_yaml_and_keeps_extra(cmd_args_from_yaml, tmp_path):
    cmd_args = cmd_args_from_yaml("epoch_minimal.yaml", extra_flag="hello")
    args = merge_basic_args(cmd_args)

    assert args.runner.max_epochs == 3
    assert args.runner.run_mode == "epoch"
    assert args.extra_flag == "hello"
    assert args.log_dir == str(tmp_path / "logs")


def test_merge_basic_args_reads_checkpoint_regular_latest_only(cmd_args_from_yaml, tmp_path):
    config_path = tmp_path / "regular_latest_only.yaml"
    config_path.write_text(
        """
log_dir: ./logs
optim:
  optimizer: adam
  optimizer_params:
    lr: 0.001
  loss: mse
runner:
  run_mode: epoch
  max_epochs: 3
  eval_interval: 1
  early_stop:
    patience: 2
  checkpoint:
    regular_latest_only: false
task:
  dataset: dummy
  dataloader:
    batch_size: 2
""".strip(),
        encoding="utf-8",
    )
    cmd_args = cmd_args_from_yaml(config_path.name)
    cmd_args.config = str(config_path)

    args = merge_basic_args(cmd_args)

    assert args.runner.checkpoint.regular_latest_only is False

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


def test_parse_loss_name_rmse_returns_rmse_loss():
    loss_fn = parse_loss_name("rmse")

    assert isinstance(loss_fn, RMSELoss)

    input_tensor = torch.tensor([1.0, 3.0])
    target_tensor = torch.tensor([1.0, 1.0])
    loss = loss_fn(input_tensor, target_tensor)

    assert torch.isclose(loss, torch.tensor((2.0**0.5)))


def test_parse_loss_name_rmse_ddp_returns_ddp_rmse_loss():
    with pytest.raises(NotImplementedError, match="rmse is not supported in DDP"):
        parse_loss_name("rmse", dpp=True)


def test_parse_loss_name_focal_ddp_returns_ddp_focal_loss():
    loss_fn = parse_loss_name("focal", dpp=True)

    assert isinstance(loss_fn, DDPMeanReducedLoss)
    assert isinstance(loss_fn.loss_fn, FocalLoss)
    assert loss_fn.loss_fn.reduction == "sum"


def test_parse_loss_name_focal_returns_mean_focal_loss():
    loss_fn = parse_loss_name("focal")

    assert isinstance(loss_fn, FocalLoss)
    assert loss_fn.reduction == "mean"


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
    assert isinstance(loss_fn, torch.nn.Module)
    assert isinstance(loss_fn.loss_fns, torch.nn.ModuleDict)
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


def test_prepare_loss_single_rmse_success():
    args = qt.qDict(
        {
            "distributed": False,
            "optim": {
                "loss": "rmse",
            },
        }
    )

    loss_fn = prepare_loss(args)

    assert isinstance(loss_fn, RMSELoss)


def test_comboloss_forwards_kwargs_to_child_losses():
    class CaptureLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.kwargs = None

        def forward(self, input, target, **kwargs):
            self.kwargs = kwargs
            return ((input - target) ** 2).mean()

    from qqtools.plugins.qpipeline.entry_utils.loss import ComboLoss

    child_loss = CaptureLoss()
    combo = ComboLoss({"energy": child_loss}, {"energy": 1.0})

    combo({"energy": torch.tensor([1.0])}, {"energy": torch.tensor([0.0])}, natoms=torch.tensor([3]))

    assert child_loss.kwargs == {"natoms": torch.tensor([3])}


def test_comboloss_rejects_empty_loss_fns():
    from qqtools.plugins.qpipeline.entry_utils.loss import ComboLoss

    with pytest.raises(AssertionError, match="loss_fns must not be empty"):
        ComboLoss({}, {})


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


def test_prepare_scheduler_non_plateau_defaults_step_on_optimizer_step(base_args, tiny_model):
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)
    args = base_args.copy()
    args.optim = qt.qDict(
        {
            "scheduler": "step",
            "scheduler_params": qt.qDict({"step_size": 1, "gamma": 0.1}),
            "warmup_params": qt.qDict(),
        }
    )

    scheduler = prepare_scheduler(args, optimizer)

    assert scheduler.step_on == "optimizer_step"


def test_prepare_scheduler_plateau_defaults_step_on_valid_end(base_args, tiny_model):
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)
    args = base_args.copy()
    args.optim = qt.qDict(
        {
            "scheduler": "plateau",
            "scheduler_params": qt.qDict({"factor": 0.1, "patience": 1}),
            "warmup_params": qt.qDict(),
        }
    )

    scheduler = prepare_scheduler(args, optimizer)

    assert scheduler.step_on == "valid_end"


def test_prepare_scheduler_plateau_rejects_non_valid_end_step_on(base_args, tiny_model):
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)
    args = base_args.copy()
    args.optim = qt.qDict(
        {
            "scheduler": "plateau",
            "scheduler_params": qt.qDict({"factor": 0.1, "patience": 1, "step_on": "optimizer_step"}),
            "warmup_params": qt.qDict(),
        }
    )

    with pytest.raises(ValueError, match="scheduler_params.step_on must be 'valid_end'"):
        prepare_scheduler(args, optimizer)


def test_prepare_scheduler_non_plateau_rejects_invalid_step_on(base_args, tiny_model):
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)
    args = base_args.copy()
    args.optim = qt.qDict(
        {
            "scheduler": "step",
            "scheduler_params": qt.qDict({"step_size": 1, "gamma": 0.1, "step_on": "epoch_end"}),
            "warmup_params": qt.qDict(),
        }
    )

    with pytest.raises(ValueError, match="scheduler_params.step_on must be one of"):
        prepare_scheduler(args, optimizer)


def test_prepare_scheduler_does_not_pass_step_on_to_scheduler_getter(base_args, tiny_model, monkeypatch):
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)
    args = base_args.copy()
    args.optim = qt.qDict(
        {
            "scheduler": "step",
            "scheduler_params": qt.qDict({"step_size": 1, "gamma": 0.1, "step_on": "valid_end"}),
            "warmup_params": qt.qDict(),
        }
    )
    captured = {}

    def _fake_getter(scheduler_params, passed_optimizer):
        captured["scheduler_params"] = dict(scheduler_params)
        captured["optimizer"] = passed_optimizer
        return torch.optim.lr_scheduler.StepLR(passed_optimizer, step_size=1, gamma=0.1)

    monkeypatch.setitem(
        __import__("qqtools.plugins.qpipeline.entry_utils.scheduler", fromlist=["SCHEDULER_GETTERS"]).SCHEDULER_GETTERS,
        "step",
        _fake_getter,
    )

    scheduler = prepare_scheduler(args, optimizer)

    assert scheduler.step_on == "valid_end"
    assert captured["optimizer"] is optimizer
    assert "step_on" not in captured["scheduler_params"]

