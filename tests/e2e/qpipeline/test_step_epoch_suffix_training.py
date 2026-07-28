"""Public YAML-to-training regression coverage for epoch-suffix intervals."""

from argparse import Namespace
from pathlib import Path

import pytest

import qqtools as qt
from qqtools.plugins.qpipeline.cmd_args import merge_basic_args
from qqtools.plugins.qpipeline.entry import create_pipeline_class
from tests.support.qpipeline import TinyModel, TinyTask


pytestmark = pytest.mark.e2e


def test_step_epoch_suffix_yaml_completes_training(tmp_path):
    config_file = Path(__file__).parent / "examples" / "step_epoch_suffix_minimal.yaml"
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
