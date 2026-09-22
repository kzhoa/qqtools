"""YAML and real-runner configuration flow coverage for epoch-suffix intervals."""

from argparse import Namespace
from pathlib import Path

import pytest
import torch

import qqtools as qt
import qqtools.plugins.qpipeline.runner.runner as runner_module
from qqtools.plugins.qpipeline.cmd_args import merge_basic_args
from qqtools.plugins.qpipeline.entry import create_pipeline_class
from qqtools.plugins.qpipeline.runner.runner import train_runner
from tests.helpers.qpipeline import TinyModel, TinyTask

pytestmark = pytest.mark.integration


def test_step_epoch_suffix_yaml_completes_training(tmp_path):
    config_file = Path(__file__).parents[3] / "fixtures" / "qpipeline" / "step_epoch_suffix_minimal.yaml"
    command_args = qt.qDict.from_namespace(
        Namespace(
            config=str(config_file),
            ckp_file=None,
            test=False,
            local_rank=0,
            ddp_detect=False,
            log_dir=str(tmp_path / "logs"),
        )
    )
    args = merge_basic_args(command_args)
    args.config_file = str(config_file)
    args.init_file = None
    args.render_type = "plain"

    pipeline_class = create_pipeline_class(lambda _args: TinyModel(), lambda _args: TinyTask())
    result = pipeline_class(args, mode="train").fit()

    assert result["final_step"] == 2
    log_text = (Path(args.log_dir) / "debug.log").read_text(encoding="utf-8")
    assert "runner.eval_interval: 0.5epoch -> 1" in log_text
    assert "runner.save_interval: 1epoch -> 3" in log_text


def _make_args(tmp_path):
    return qt.qDict(
        {
            "device": torch.device("cpu"),
            "rank": 0,
            "log_dir": str(tmp_path / "logs"),
            "ckp_file": None,
            "init_file": None,
            "render_type": "plain",
            "optim": {"scheduler": ""},
            "runner": {
                "eval_interval": "99epoch",
                "save_interval": "99epoch",
                "checkpoint": {},
                "early_stop": {"target": "val_metric", "patience": 999, "mode": "min"},
            },
        }
    )


def _run_and_capture_config(monkeypatch, tmp_path, *, run_mode, eval_interval, save_interval):
    captured = {}
    original_agent = runner_module.RunningAgent
    original_checkpoint_plugin = runner_module.CheckpointPlugin

    class CapturingAgent(original_agent):
        def __init__(self, *args, **kwargs):
            captured["config"] = kwargs["config"]
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(runner_module, "RunningAgent", CapturingAgent)

    class CapturingCheckpointPlugin(original_checkpoint_plugin):
        def __init__(self, *args, **kwargs):
            captured["checkpoint_policy"] = kwargs["policy"]
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(runner_module, "CheckpointPlugin", CapturingCheckpointPlugin)
    args = _make_args(tmp_path)
    model = TinyModel()
    result = train_runner(
        model=model,
        task=TinyTask(),
        loss_fn=torch.nn.MSELoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1.0e-3),
        args=args,
        run_mode=run_mode,
        max_steps=1 if run_mode == "step" else None,
        max_epochs=1 if run_mode == "epoch" else None,
        eval_interval=eval_interval,
        save_interval=save_interval,
        save_dir=args.log_dir,
        print_freq=1,
    )
    return args, captured["config"], captured["checkpoint_policy"], result


@pytest.mark.parametrize(
    ("eval_interval", "save_interval", "expected_eval", "expected_save"),
    [
        ("0.5epoch", "0.25epoch", 1, 1),
        ("0.5epoch", None, 1, 1),
        (1, "0.25epoch", 1, 1),
    ],
)
def test_train_runner_standardizes_effective_interval_inputs(
    monkeypatch,
    tmp_path,
    eval_interval,
    save_interval,
    expected_eval,
    expected_save,
):
    args, config, checkpoint_policy, result = _run_and_capture_config(
        monkeypatch,
        tmp_path,
        run_mode="step",
        eval_interval=eval_interval,
        save_interval=save_interval,
    )

    assert result["final_step"] == 1
    assert config.eval_interval == expected_eval
    assert checkpoint_policy.save_interval == expected_save
    assert args.runner.eval_interval == "99epoch"
    assert args.runner.save_interval == "99epoch"


def test_train_runner_rejects_runner_suffixes_in_epoch_mode(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="run_mode='epoch'"):
        _run_and_capture_config(
            monkeypatch,
            tmp_path,
            run_mode="epoch",
            eval_interval="0.5epoch",
            save_interval=None,
        )
