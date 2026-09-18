import pytest

from qqtools.plugins.qpipeline.entry_utils.scheduler import SchedulerConfig, WarmupConfig, get_lambda_lr


def test_warmup_config_rejects_invalid_factor() -> None:
    with pytest.raises(ValueError):
        WarmupConfig(steps=10, epochs=0, initial_factor=1.2)


def test_scheduler_config_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        SchedulerConfig(name="invalid", params={})


def test_get_lambda_lr_rejects_expression_that_cannot_be_evaluated() -> None:
    with pytest.raises(ValueError, match="Failed to evaluate lr_lambda string"):
        get_lambda_lr({"lr_lambda": "lambda epoch: ("}, object())
