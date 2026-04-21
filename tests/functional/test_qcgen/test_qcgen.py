"""
Test suite for qcgen CLI command
Tests the qConfigGen interactive configuration generator
"""

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import yaml


def _load_runner_schema():
    schema_path = Path(__file__).resolve().parents[3] / "src/qqtools/plugins/qConfigGen/schemas/definitions/runner.json"
    with open(schema_path, "r") as f:
        return json.load(f)


def test_qcgen_import():
    """Test that qcgen CLI module can be imported"""
    from qqtools.cli import qcgen

    assert qcgen is not None
    assert hasattr(qcgen, "main")


def test_qcgen_main_callable():
    """Test that qcgen.main is callable"""
    from qqtools.cli.qcgen import main

    assert callable(main)


def test_qcgen_full_workflow():
    """Test qcgen complete workflow with mocked prompts and UI functions"""
    from qqtools.plugins.qConfigGen.main import main

    with (
        mock.patch("prompt_toolkit.prompt", return_value="test_value"),
        mock.patch("qqtools.plugins.qConfigGen.main.print_formatted_text"),
        mock.patch("qqtools.plugins.qConfigGen.main.pretty.install"),
        mock.patch("qqtools.plugins.qConfigGen.main.prompt_global_params") as mock_global,
        mock.patch("qqtools.plugins.qConfigGen.main.prompt_task_params") as mock_task,
        mock.patch("qqtools.plugins.qConfigGen.main.prompt_loss_params") as mock_loss,
        mock.patch("qqtools.plugins.qConfigGen.main.prompt_optimizer_params") as mock_optim,
        mock.patch("qqtools.plugins.qConfigGen.main.prompt_lr_scheduler_params") as mock_scheduler,
        mock.patch("qqtools.plugins.qConfigGen.main.prompt_ema_params") as mock_ema,
        mock.patch("qqtools.plugins.qConfigGen.main.prompt_model_params") as mock_model,
        mock.patch("qqtools.plugins.qConfigGen.main.prompt_runner_params") as mock_runner,
        mock.patch("qqtools.plugins.qConfigGen.main.prompt_save_location") as mock_save,
    ):
        # Mock return values for configuration steps
        mock_global.return_value = {"seed": 42, "log_dir": "./tmp/logs", "print_freq": 100}
        mock_task.return_value = {"dataset": "imagenet", "dataloader": {"batch_size": 32}}
        mock_loss.return_value = {"loss": "cross_entropy"}
        mock_optim.return_value = {"optimizer": "adamw", "lr": 0.001}
        mock_scheduler.return_value = {"scheduler": "cosine"}
        mock_ema.return_value = {"ema_decay": 0.99}
        mock_model.return_value = {"model_type": "resnet50"}
        mock_runner.return_value = {"run_mode": "epoch", "max_epochs": 100}

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_save.return_value = (tmpdir, "config.yaml")

            # Execute main workflow
            main()

            # Verify the config file was created
            config_file = Path(tmpdir) / "config.yaml"
            assert config_file.exists(), f"Config file not created at {config_file}"

            # Verify it's valid YAML and has expected structure
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            assert config is not None
            assert isinstance(config, dict)
            assert "seed" in config
            assert config["seed"] == 42
            assert "log_dir" in config
            assert "task" in config
            assert "optim" in config
            assert "runner" in config


def test_qcgen_yaml_output_format():
    """Test that generated YAML has proper formatting with gaps"""
    from qqtools.plugins.qConfigGen.main import dump_yaml_with_gaps

    test_config = {
        "seed": 42,
        "log_dir": "./tmp/logs",
        "task": {"dataset": "imagenet", "dataloader": {"batch_size": 32}},
        "optim": {"loss": "cross_entropy", "optimizer": "adamw"},
        "model": {"model_type": "resnet50"},
        "runner": {"run_mode": "epoch", "max_epochs": 100},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "test_output.yaml"

        dump_yaml_with_gaps(test_config, str(output_file), gap_lines=1)

        assert output_file.exists()

        # Verify file content
        with open(output_file, "r") as f:
            content = f.read()

        # Should have blank lines between sections
        # The format outputs \n\n (2 newlines) between sections
        assert "\n\n" in content, "Should have blank lines between sections"

        # Verify it loads correctly
        loaded_config = yaml.safe_load(content)
        assert loaded_config["seed"] == 42
        assert loaded_config["task"]["dataset"] == "imagenet"
        assert loaded_config["optim"]["optimizer"] == "adamw"


def test_qcgen_cli_entry_point():
    """Test that qcgen is properly registered as CLI entry point"""
    from qqtools.cli.qcgen import main

    # Mock the entire main workflow to avoid interactive prompts
    with (
        mock.patch("qqtools.plugins.qConfigGen.main.prompt_global_params") as mock_global,
        mock.patch("qqtools.plugins.qConfigGen.main.prompt_task_params") as mock_task,
        mock.patch("qqtools.plugins.qConfigGen.main.prompt_loss_params") as mock_loss,
        mock.patch("qqtools.plugins.qConfigGen.main.prompt_optimizer_params") as mock_optim,
        mock.patch("qqtools.plugins.qConfigGen.main.prompt_lr_scheduler_params") as mock_scheduler,
        mock.patch("qqtools.plugins.qConfigGen.main.prompt_ema_params") as mock_ema,
        mock.patch("qqtools.plugins.qConfigGen.main.prompt_model_params") as mock_model,
        mock.patch("qqtools.plugins.qConfigGen.main.prompt_runner_params") as mock_runner,
        mock.patch("qqtools.plugins.qConfigGen.main.prompt_save_location") as mock_save,
        mock.patch("qqtools.plugins.qConfigGen.main.print_formatted_text"),
        mock.patch("qqtools.plugins.qConfigGen.main.pretty.install"),
    ):

        # Set return values for all mocks
        mock_global.return_value = {"seed": 42, "log_dir": "./tmp"}
        mock_task.return_value = {"dataset": "test"}
        mock_loss.return_value = {"loss": "mse"}
        mock_optim.return_value = {"optimizer": "adam"}
        mock_scheduler.return_value = {"scheduler": "cosine"}
        mock_ema.return_value = {}
        mock_model.return_value = {}
        mock_runner.return_value = {"run_mode": "epoch", "max_epochs": 1}

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_save.return_value = (tmpdir, "config.yaml")

            # Should not raise any exceptions
            main()

            # Verify config was created
            config_file = Path(tmpdir) / "config.yaml"
            assert config_file.exists()


def test_runner_schema_supports_accum_grad():
    schema = _load_runner_schema()

    accum_grad_schema = schema["properties"]["accum_grad"]

    assert accum_grad_schema["type"] == ["integer", "null"]
    assert accum_grad_schema["minimum"] == 1
    assert accum_grad_schema["default"] is None


def test_runner_schema_requires_mode_specific_boundaries():
    schema = _load_runner_schema()
    branches = {branch["properties"]["run_mode"]["const"]: branch for branch in schema["oneOf"]}

    epoch_branch = branches["epoch"]
    step_branch = branches["step"]

    assert "max_epochs" in epoch_branch["required"]
    assert "max_steps" not in epoch_branch["required"]
    assert "max_steps" in step_branch["required"]
    assert "anyOf" not in step_branch
    assert step_branch["properties"]["max_steps"]["type"] == "integer"
    assert step_branch["properties"]["max_steps"]["minimum"] == 1
    assert "max_epochs" not in step_branch.get("required", [])


def test_runner_schema_describes_step_mode_secondary_epoch_boundary():
    schema = _load_runner_schema()

    assert "secondary stopping boundary" in schema["properties"]["max_epochs"]["description"]


def test_runner_schema_save_interval_description_matches_runtime_semantics():
    schema = _load_runner_schema()
    save_interval_desc = schema["properties"]["save_interval"]["description"]

    assert "eval_interval" in save_interval_desc
    assert "run_mode" in save_interval_desc


def test_runner_schema_supports_regular_latest_only():
    schema = _load_runner_schema()
    checkpoint_schema = schema["properties"]["checkpoint"]["properties"]["regular_latest_only"]

    assert checkpoint_schema["type"] == "boolean"
    assert checkpoint_schema["default"] is True


def test_qcgen_main_nests_regular_latest_only_under_checkpoint():
    from qqtools.plugins.qConfigGen.main import main

    with (
        mock.patch("qqtools.plugins.qConfigGen.main.print_formatted_text"),
        mock.patch("qqtools.plugins.qConfigGen.main.pretty.install"),
        mock.patch("qqtools.plugins.qConfigGen.main.prompt_global_params") as mock_global,
        mock.patch("qqtools.plugins.qConfigGen.main.prompt_task_params") as mock_task,
        mock.patch("qqtools.plugins.qConfigGen.main.prompt_loss_params") as mock_loss,
        mock.patch("qqtools.plugins.qConfigGen.main.prompt_optimizer_params") as mock_optim,
        mock.patch("qqtools.plugins.qConfigGen.main.prompt_lr_scheduler_params") as mock_scheduler,
        mock.patch("qqtools.plugins.qConfigGen.main.prompt_ema_params") as mock_ema,
        mock.patch("qqtools.plugins.qConfigGen.main.prompt_model_params") as mock_model,
        mock.patch("qqtools.plugins.qConfigGen.main.prompt_runner_params") as mock_runner,
        mock.patch("qqtools.plugins.qConfigGen.main.prompt_save_location") as mock_save,
    ):
        mock_global.return_value = {"seed": 42, "log_dir": "./tmp/logs"}
        mock_task.return_value = {"dataset": "imagenet", "dataloader": {"batch_size": 32}}
        mock_loss.return_value = {"loss": "cross_entropy"}
        mock_optim.return_value = {"optimizer": "adamw", "lr": 0.001}
        mock_scheduler.return_value = {}
        mock_ema.return_value = {}
        mock_model.return_value = {}
        mock_runner.return_value = {
            "run_mode": "epoch",
            "max_epochs": 100,
            "regular_latest_only": True,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_save.return_value = (tmpdir, "config.yaml")

            main()

            config_file = Path(tmpdir) / "config.yaml"
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

        assert config["runner"]["checkpoint"]["regular_latest_only"] is True
        assert "regular_latest_only" not in config["runner"]


def test_prompt_runner_params_can_disable_regular_latest_only():
    from qqtools.plugins.qConfigGen.pts.runnerPt import prompt_runner_params

    prompt_values = [
        "",
        "",
        "",
        "",
        "",
        "",
        "n",
    ]

    with (
        mock.patch("qqtools.plugins.qConfigGen.pts.runnerPt.prompt", side_effect=prompt_values),
        mock.patch("qqtools.plugins.qConfigGen.pts.runnerPt.print_formatted_text"),
        mock.patch("qqtools.plugins.qConfigGen.pts.runnerPt.prompt_early_stop", return_value={}),
    ):
        params = prompt_runner_params()

    assert params["regular_latest_only"] is False


def test_prompt_runner_params_accepts_step_mode_secondary_max_epochs():
    from qqtools.plugins.qConfigGen.pts.runnerPt import prompt_runner_params

    prompt_values = [
        "step",
        "500",
        "7",
        "",
        "",
        "",
        "",
        "y",
    ]

    with (
        mock.patch("qqtools.plugins.qConfigGen.pts.runnerPt.prompt", side_effect=prompt_values),
        mock.patch("qqtools.plugins.qConfigGen.pts.runnerPt.print_formatted_text"),
        mock.patch("qqtools.plugins.qConfigGen.pts.runnerPt.prompt_early_stop", return_value={}),
    ):
        params = prompt_runner_params()

    assert params["run_mode"] == "step"
    assert params["max_steps"] == 500
    assert params["max_epochs"] == 7


def test_prompt_lr_scheduler_params_defaults_non_plateau_step_on():
    from qqtools.plugins.qConfigGen.pts.lrSchedulerPt import prompt_lr_scheduler_params

    prompt_values = [
        "yes",
        "step",
        "",
        "",
        "",
        "no",
    ]

    with (
        mock.patch("qqtools.plugins.qConfigGen.pts.lrSchedulerPt.prompt", side_effect=prompt_values),
        mock.patch("qqtools.plugins.qConfigGen.pts.lrSchedulerPt.print_formatted_text"),
    ):
        result = prompt_lr_scheduler_params()

    assert result["scheduler"] == "step"
    assert result["scheduler_params"]["step_size"] == 30
    assert result["scheduler_params"]["gamma"] == 0.1
    assert result["scheduler_params"]["step_on"] == "optimizer_step"


def test_prompt_lr_scheduler_params_plateau_forces_valid_end():
    from qqtools.plugins.qConfigGen.pts.lrSchedulerPt import prompt_lr_scheduler_params

    prompt_values = [
        "yes",
        "plateau",
        "",
        "",
        "",
        "",
        "",
        "no",
    ]

    with (
        mock.patch("qqtools.plugins.qConfigGen.pts.lrSchedulerPt.prompt", side_effect=prompt_values),
        mock.patch("qqtools.plugins.qConfigGen.pts.lrSchedulerPt.print_formatted_text"),
    ):
        result = prompt_lr_scheduler_params()

    assert result["scheduler"] == "plateau"
    assert result["scheduler_params"]["mode"] == "min"
    assert result["scheduler_params"]["step_on"] == "valid_end"
