"""Tests for EpochSuffixResolver — epoch-suffix auto-compute for step-mode fields."""

import pytest

import qqtools as qt
from qqtools.plugins.qpipeline.runner.runner_utils.epoch_suffix import (
    EpochSuffixResolver,
    _parse_epoch_suffix,
)


# ─── Unit tests for _parse_epoch_suffix ────────────────────────────────────────


class TestParseEpochSuffix:
    def test_integer_epoch(self):
        assert _parse_epoch_suffix("5epoch") == 5.0

    def test_float_epoch(self):
        assert _parse_epoch_suffix("0.5epoch") == 0.5

    def test_plain_integer(self):
        assert _parse_epoch_suffix(100) is None

    def test_plain_string_integer(self):
        assert _parse_epoch_suffix("100") is None

    def test_no_suffix(self):
        assert _parse_epoch_suffix("hello") is None

    def test_whitespace_stripped(self):
        assert _parse_epoch_suffix("  3epoch  ") == 3.0

    def test_zero_epoch(self):
        assert _parse_epoch_suffix("0epoch") == 0.0

    def test_large_float(self):
        assert _parse_epoch_suffix("10.5epoch") == 10.5


# ─── Fixtures ──────────────────────────────────────────────────────────────────


def _make_args(
    run_mode="step",
    accum_grad=None,
    scheduler="cosine",
    scheduler_params=None,
    warmup_params=None,
    eval_interval=1,
    save_interval=None,
    step_on=None,
):
    sp = scheduler_params or {}
    if step_on is not None:
        sp["step_on"] = step_on

    runner = {"run_mode": run_mode, "eval_interval": eval_interval, "save_interval": save_interval}
    if accum_grad is not None:
        runner["accum_grad"] = accum_grad

    return qt.qDict(
        {
            "optim": {
                "scheduler": scheduler,
                "scheduler_params": sp,
                "warmup_params": warmup_params,
            },
            "runner": runner,
            "distributed": False,
        }
    )


# ─── Step mode: basic resolution ──────────────────────────────────────────────


class TestStepModeBasicResolution:
    """run_mode=step, step_on=optimizer_step (default)."""

    def test_t_max_integer_epoch(self):
        # train_loader_length=100, accum_grad=2 => steps_per_epoch=50
        args = _make_args(scheduler_params={"T_max": "5epoch"}, accum_grad=2)
        resolver = EpochSuffixResolver(
            run_mode="step", step_on=None, train_loader_length=100, accum_grad=2
        )
        result = resolver.resolve(args)
        assert args.optim.scheduler_params.T_max == 250  # 5 * 50

    def test_t_max_float_epoch(self):
        args = _make_args(scheduler_params={"T_max": "0.5epoch"}, accum_grad=4)
        resolver = EpochSuffixResolver(
            run_mode="step", step_on=None, train_loader_length=1000, accum_grad=4
        )
        result = resolver.resolve(args)
        assert args.optim.scheduler_params.T_max == 125  # 0.5 * 250

    def test_eval_interval_epoch(self):
        args = _make_args(eval_interval="1epoch")
        resolver = EpochSuffixResolver(
            run_mode="step", step_on=None, train_loader_length=200, accum_grad=1
        )
        result = resolver.resolve(args)
        assert args.runner.eval_interval == 200

    def test_save_interval_epoch(self):
        args = _make_args(save_interval="0.5epoch")
        resolver = EpochSuffixResolver(
            run_mode="step", step_on=None, train_loader_length=200, accum_grad=2
        )
        result = resolver.resolve(args)
        assert args.runner.save_interval == 50  # 0.5 * 100

    def test_warmup_steps_epoch(self):
        args = _make_args(warmup_params={"warmup_steps": "0.1epoch", "warmup_factor": 0.1})
        resolver = EpochSuffixResolver(
            run_mode="step", step_on=None, train_loader_length=1000, accum_grad=4
        )
        result = resolver.resolve(args)
        assert args.optim.warmup_params.warmup_steps == 25  # 0.1 * 250

    def test_step_size_epoch(self):
        args = _make_args(scheduler="step", scheduler_params={"step_size": "2epoch", "gamma": 0.1})
        resolver = EpochSuffixResolver(
            run_mode="step", step_on=None, train_loader_length=100, accum_grad=1
        )
        result = resolver.resolve(args)
        assert args.optim.scheduler_params.step_size == 200

    def test_milestones_epoch(self):
        args = _make_args(
            scheduler="multi_step",
            scheduler_params={"milestones": ["0.3epoch", "0.6epoch", "0.9epoch"], "gamma": 0.1},
        )
        resolver = EpochSuffixResolver(
            run_mode="step", step_on=None, train_loader_length=100, accum_grad=1
        )
        result = resolver.resolve(args)
        assert args.optim.scheduler_params.milestones == [30, 60, 90]

    def test_no_epoch_suffix_passthrough(self):
        args = _make_args(scheduler_params={"T_max": 100}, eval_interval=5)
        resolver = EpochSuffixResolver(
            run_mode="step", step_on=None, train_loader_length=200, accum_grad=1
        )
        result = resolver.resolve(args)
        assert args.optim.scheduler_params.T_max == 100
        assert args.runner.eval_interval == 5
        assert not result.has_resolved

    def test_minimum_clamp_to_1(self):
        # 0.001 * 10 steps_per_epoch = 0.01 -> clamp to 1
        args = _make_args(scheduler_params={"T_max": "0.001epoch"})
        resolver = EpochSuffixResolver(
            run_mode="step", step_on=None, train_loader_length=10, accum_grad=1
        )
        result = resolver.resolve(args)
        assert args.optim.scheduler_params.T_max == 1


# ─── Step mode: accum_grad fallback ───────────────────────────────────────────


class TestAccumGradFallback:
    def test_accum_grad_none_defaults_to_1(self):
        args = _make_args(scheduler_params={"T_max": "1epoch"}, accum_grad=None)
        resolver = EpochSuffixResolver(
            run_mode="step", step_on=None, train_loader_length=100, accum_grad=None
        )
        result = resolver.resolve(args)
        assert args.optim.scheduler_params.T_max == 100  # 1 * (100/1)

    def test_accum_grad_explicit(self):
        args = _make_args(scheduler_params={"T_max": "1epoch"}, accum_grad=4)
        resolver = EpochSuffixResolver(
            run_mode="step", step_on=None, train_loader_length=100, accum_grad=4
        )
        result = resolver.resolve(args)
        assert args.optim.scheduler_params.T_max == 25  # 1 * ceil(100/4)


# ─── step_on=valid_end constraint ─────────────────────────────────────────────


class TestStepOnValidEnd:
    def test_scheduler_param_epoch_suffix_raises(self):
        args = _make_args(scheduler_params={"T_max": "5epoch"}, step_on="valid_end")
        resolver = EpochSuffixResolver(
            run_mode="step", step_on="valid_end", train_loader_length=100, accum_grad=1
        )
        with pytest.raises(ValueError, match="step_on='valid_end'"):
            resolver.resolve(args)

    def test_step_size_valid_end_raises(self):
        args = _make_args(scheduler="step", scheduler_params={"step_size": "2epoch"}, step_on="valid_end")
        resolver = EpochSuffixResolver(
            run_mode="step", step_on="valid_end", train_loader_length=100, accum_grad=1
        )
        with pytest.raises(ValueError, match="step_on='valid_end'"):
            resolver.resolve(args)

    def test_warmup_steps_allowed_with_valid_end(self):
        args = _make_args(
            scheduler_params={"T_max": 10},
            warmup_params={"warmup_steps": "0.1epoch", "warmup_factor": 0.1},
            step_on="valid_end",
        )
        resolver = EpochSuffixResolver(
            run_mode="step", step_on="valid_end", train_loader_length=100, accum_grad=1
        )
        result = resolver.resolve(args)
        assert args.optim.warmup_params.warmup_steps == 10  # 0.1 * 100

    def test_runner_fields_allowed_with_valid_end(self):
        args = _make_args(eval_interval="1epoch", step_on="valid_end")
        resolver = EpochSuffixResolver(
            run_mode="step", step_on="valid_end", train_loader_length=100, accum_grad=1
        )
        result = resolver.resolve(args)
        assert args.runner.eval_interval == 100


# ─── run_mode=epoch constraints ───────────────────────────────────────────────


class TestEpochModeConstraints:
    def test_runner_fields_epoch_suffix_raises(self):
        args = _make_args(run_mode="epoch", eval_interval="0.5epoch")
        resolver = EpochSuffixResolver(
            run_mode="epoch", step_on=None, train_loader_length=100, accum_grad=1
        )
        with pytest.raises(ValueError, match="run_mode='epoch'"):
            resolver.resolve(args)

    def test_save_interval_epoch_suffix_raises(self):
        args = _make_args(run_mode="epoch", save_interval="2epoch")
        resolver = EpochSuffixResolver(
            run_mode="epoch", step_on=None, train_loader_length=100, accum_grad=1
        )
        with pytest.raises(ValueError, match="run_mode='epoch'"):
            resolver.resolve(args)

    def test_scheduler_params_allowed_in_epoch_mode(self):
        args = _make_args(run_mode="epoch", scheduler_params={"T_max": "5epoch"})
        resolver = EpochSuffixResolver(
            run_mode="epoch", step_on=None, train_loader_length=100, accum_grad=1
        )
        result = resolver.resolve(args)
        assert args.optim.scheduler_params.T_max == 500  # 5 * 100

    def test_scheduler_params_valid_end_raises_in_epoch_mode(self):
        args = _make_args(run_mode="epoch", scheduler_params={"T_max": "5epoch"}, step_on="valid_end")
        resolver = EpochSuffixResolver(
            run_mode="epoch", step_on="valid_end", train_loader_length=100, accum_grad=1
        )
        with pytest.raises(ValueError, match="step_on='valid_end'"):
            resolver.resolve(args)

    def test_warmup_allowed_in_epoch_mode(self):
        args = _make_args(
            run_mode="epoch", warmup_params={"warmup_steps": "0.2epoch", "warmup_factor": 0.1}
        )
        resolver = EpochSuffixResolver(
            run_mode="epoch", step_on=None, train_loader_length=100, accum_grad=2
        )
        result = resolver.resolve(args)
        assert args.optim.warmup_params.warmup_steps == 10  # 0.2 * 50


# ─── Warmup coupling warning ──────────────────────────────────────────────────


class TestWarmupCouplingWarning:
    def test_both_epoch_suffix_triggers_warning(self):
        args = _make_args(
            scheduler_params={"T_max": "5epoch"},
            warmup_params={"warmup_steps": "0.1epoch", "warmup_factor": 0.1},
        )
        resolver = EpochSuffixResolver(
            run_mode="step", step_on=None, train_loader_length=100, accum_grad=1
        )
        result = resolver.resolve(args)
        assert len(result.warnings) == 1
        assert "warmup" in result.warnings[0].lower() or "AFTER" in result.warnings[0]

    def test_only_warmup_no_warning(self):
        args = _make_args(
            scheduler_params={"T_max": 100},
            warmup_params={"warmup_steps": "0.1epoch", "warmup_factor": 0.1},
        )
        resolver = EpochSuffixResolver(
            run_mode="step", step_on=None, train_loader_length=100, accum_grad=1
        )
        result = resolver.resolve(args)
        assert len(result.warnings) == 0

    def test_only_scheduler_no_warning(self):
        args = _make_args(
            scheduler_params={"T_max": "5epoch"},
            warmup_params={"warmup_steps": 10, "warmup_factor": 0.1},
        )
        resolver = EpochSuffixResolver(
            run_mode="step", step_on=None, train_loader_length=100, accum_grad=1
        )
        result = resolver.resolve(args)
        assert len(result.warnings) == 0


# ─── Error conditions ─────────────────────────────────────────────────────────


class TestErrorConditions:
    def test_zero_coefficient_raises(self):
        args = _make_args(scheduler_params={"T_max": "0epoch"})
        resolver = EpochSuffixResolver(
            run_mode="step", step_on=None, train_loader_length=100, accum_grad=1
        )
        with pytest.raises(ValueError, match="positive"):
            resolver.resolve(args)

    def test_invalid_train_loader_length_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            EpochSuffixResolver(
                run_mode="step", step_on=None, train_loader_length=0, accum_grad=1
            )

    def test_negative_train_loader_length_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            EpochSuffixResolver(
                run_mode="step", step_on=None, train_loader_length=-1, accum_grad=1
            )


# ─── ResolveResult ─────────────────────────────────────────────────────────────


class TestResolveResult:
    def test_has_resolved_true(self):
        args = _make_args(scheduler_params={"T_max": "5epoch"})
        resolver = EpochSuffixResolver(
            run_mode="step", step_on=None, train_loader_length=100, accum_grad=1
        )
        result = resolver.resolve(args)
        assert result.has_resolved

    def test_has_resolved_false(self):
        args = _make_args(scheduler_params={"T_max": 100})
        resolver = EpochSuffixResolver(
            run_mode="step", step_on=None, train_loader_length=100, accum_grad=1
        )
        result = resolver.resolve(args)
        assert not result.has_resolved

    def test_logs_contain_resolution_info(self):
        args = _make_args(scheduler_params={"T_max": "2epoch"})
        resolver = EpochSuffixResolver(
            run_mode="step", step_on=None, train_loader_length=100, accum_grad=1
        )
        result = resolver.resolve(args)
        assert len(result.logs) == 1
        log = result.logs[0]
        assert "len_loader=100" in log
        assert "steps_per_epoch=100" in log
        assert "T_max" in log
        assert "2epoch" in log
        assert "200" in log

    def test_steps_per_epoch_exposed(self):
        resolver = EpochSuffixResolver(
            run_mode="step", step_on=None, train_loader_length=100, accum_grad=4
        )
        args = _make_args(scheduler_params={"T_max": 100})
        result = resolver.resolve(args)
        assert result.steps_per_epoch == 25


# ─── Plateau scheduler infers step_on=valid_end ───────────────────────────────


class TestMilestoneMixed:
    """milestones list with mixed epoch-suffix and plain integers."""

    def test_mixed_milestones(self):
        args = _make_args(
            scheduler="multi_step",
            scheduler_params={"milestones": ["0.3epoch", 50, "0.9epoch"], "gamma": 0.1},
        )
        resolver = EpochSuffixResolver(
            run_mode="step", step_on=None, train_loader_length=100, accum_grad=1
        )
        result = resolver.resolve(args)
        assert args.optim.scheduler_params.milestones == [30, 50, 90]

    def test_milestones_valid_end_raises(self):
        args = _make_args(
            scheduler="multi_step",
            scheduler_params={"milestones": ["0.3epoch", "0.6epoch"], "gamma": 0.1},
            step_on="valid_end",
        )
        resolver = EpochSuffixResolver(
            run_mode="step", step_on="valid_end", train_loader_length=100, accum_grad=1
        )
        with pytest.raises(ValueError, match="step_on='valid_end'"):
            resolver.resolve(args)


class TestMultipleFieldsCombined:
    """Test that multiple fields are resolved in a single pass."""

    def test_all_tier1_fields_together(self):
        args = _make_args(
            scheduler_params={"T_max": "10epoch"},
            eval_interval="0.5epoch",
            save_interval="2epoch",
        )
        resolver = EpochSuffixResolver(
            run_mode="step", step_on=None, train_loader_length=200, accum_grad=2
        )
        result = resolver.resolve(args)
        # steps_per_epoch = ceil(200/2) = 100
        assert args.optim.scheduler_params.T_max == 1000
        assert args.runner.eval_interval == 50
        assert args.runner.save_interval == 200
        assert len(result.resolved_fields) == 3

    def test_no_scheduler_params_graceful(self):
        """args without optim.scheduler_params should not crash."""
        args = qt.qDict({
            "optim": {"scheduler": None, "scheduler_params": None, "warmup_params": None},
            "runner": {"run_mode": "step", "eval_interval": "1epoch"},
            "distributed": False,
        })
        resolver = EpochSuffixResolver(
            run_mode="step", step_on=None, train_loader_length=50, accum_grad=1
        )
        result = resolver.resolve(args)
        assert args.runner.eval_interval == 50


class TestPlateuSchedulerInference:
    def test_plateau_infers_valid_end(self):
        args = _make_args(
            scheduler="plateau",
            scheduler_params={"T_max": "5epoch", "factor": 0.1, "patience": 5},
        )
        resolver = EpochSuffixResolver(
            run_mode="step",
            step_on=None,
            train_loader_length=100,
            accum_grad=1,
            scheduler_name="plateau",
        )
        with pytest.raises(ValueError, match="step_on='valid_end'"):
            resolver.resolve(args)
