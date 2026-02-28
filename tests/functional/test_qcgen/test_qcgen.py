"""
Test suite for qcgen CLI command
Tests the qConfigGen interactive configuration generator
"""

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import yaml


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
        mock_global.return_value = {"seed": 42, "log_dir": "/tmp/logs", "print_freq": 100}
        mock_task.return_value = {"dataset": "imagenet", "dataloader": {"batch_size": 32}}
        mock_loss.return_value = {"loss": "cross_entropy"}
        mock_optim.return_value = {"optimizer": "adamw", "lr": 0.001}
        mock_scheduler.return_value = {"scheduler": "cosine"}
        mock_ema.return_value = {"ema_decay": 0.99}
        mock_model.return_value = {"model_type": "resnet50"}
        mock_runner.return_value = {"run_mode": "train", "num_epochs": 100}

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
        "log_dir": "/tmp/logs",
        "task": {"dataset": "imagenet", "dataloader": {"batch_size": 32}},
        "optim": {"loss": "cross_entropy", "optimizer": "adamw"},
        "model": {"model_type": "resnet50"},
        "runner": {"run_mode": "train", "num_epochs": 100},
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
        mock_global.return_value = {"seed": 42, "log_dir": "/tmp"}
        mock_task.return_value = {"dataset": "test"}
        mock_loss.return_value = {"loss": "mse"}
        mock_optim.return_value = {"optimizer": "adam"}
        mock_scheduler.return_value = {"scheduler": "cosine"}
        mock_ema.return_value = {}
        mock_model.return_value = {}
        mock_runner.return_value = {"run_mode": "train"}

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_save.return_value = (tmpdir, "config.yaml")

            # Should not raise any exceptions
            main()

            # Verify config was created
            config_file = Path(tmpdir) / "config.yaml"
            assert config_file.exists()
