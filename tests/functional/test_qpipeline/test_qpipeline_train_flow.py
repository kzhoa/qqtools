from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from qqtools.plugins.qpipeline.runner.runner import train_runner


def test_train_runner_epoch_mode_uses_max_epochs(base_args, tiny_task, tiny_model):
    args = base_args.copy()
    args.runner.early_stop.patience = 999

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)

    result = train_runner(
        model=tiny_model,
        task=tiny_task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        args=args,
        run_mode="epoch",
        max_epochs=2,
        eval_interval=1,
        save_dir=args.log_dir,
        print_freq=5,
    )

    assert result["final_epoch"] == 2
    assert result["best_val_metric"] is not None


def test_train_runner_step_mode_dual_boundaries(base_args, tiny_task, tiny_model):
    args = base_args.copy()
    args.runner.early_stop.patience = 999

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)

    result = train_runner(
        model=tiny_model,
        task=tiny_task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        args=args,
        run_mode="step",
        max_epochs=3,
        max_steps=7,
        eval_interval=2,
        save_interval=3,
        save_dir=args.log_dir,
        print_freq=3,
    )

    assert result["final_step"] >= 6
    assert result["final_epoch"] <= 3
    assert result["best_val_metric"] is not None


def test_train_runner_regular_checkpoint_saving(base_args, tiny_task, tiny_model, tmp_path):
    save_dir = tmp_path / "checkpoints"
    args = base_args.copy()
    args.runner.early_stop.patience = 999

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)

    train_runner(
        model=tiny_model,
        task=tiny_task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        args=args,
        run_mode="step",
        max_steps=6,
        eval_interval=3,
        save_interval=2,
        save_dir=str(save_dir),
    )

    ckpts = list(Path(save_dir).glob("*.pt"))
    assert len(ckpts) >= 2


def test_train_runner_ckp_file_takes_effect(base_args, tiny_task, tiny_model, tmp_path):
    save_dir = tmp_path / "resume"
    args = base_args.copy()
    args.runner.early_stop.patience = 999

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)

    first = train_runner(
        model=tiny_model,
        task=tiny_task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        args=args,
        run_mode="step",
        max_steps=5,
        eval_interval=2,
        save_interval=2,
        save_dir=str(save_dir),
    )

    ckpts = sorted(Path(save_dir).glob("*.pt"))
    assert ckpts

    model2 = type(tiny_model)()
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1.0e-3)
    args2 = base_args.copy()
    args2.runner.early_stop.patience = 999
    args2.ckp_file = str(ckpts[-1])
    args2.init_file = str(tmp_path / "unused_init_file.pt")

    second = train_runner(
        model=model2,
        task=tiny_task,
        loss_fn=loss_fn,
        optimizer=optimizer2,
        args=args2,
        run_mode="step",
        max_steps=8,
        eval_interval=2,
        save_dir=str(save_dir),
    )

    assert second["final_step"] >= 4


def test_train_runner_missing_early_stop_field_uses_defaults(base_args, tiny_task, tiny_model):
    args = base_args.copy()
    args.runner = SimpleNamespace(checkpoint={})

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)

    result = train_runner(
        model=tiny_model,
        task=tiny_task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        args=args,
        run_mode="epoch",
        max_epochs=2,
        eval_interval=1,
        save_dir=args.log_dir,
        print_freq=5,
    )

    assert result["final_epoch"] == 2
    assert result["best_val_metric"] is not None


def test_train_runner_missing_checkpoint_and_early_stop_uses_defaults(base_args, tiny_task, tiny_model):
    args = base_args.copy()
    args.runner = SimpleNamespace()

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)

    result = train_runner(
        model=tiny_model,
        task=tiny_task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        args=args,
        run_mode="epoch",
        max_epochs=2,
        eval_interval=1,
        save_dir=args.log_dir,
        print_freq=5,
    )

    assert result["final_epoch"] == 2
    assert result["best_val_metric"] is not None


def test_train_runner_none_checkpoint_uses_defaults(base_args, tiny_task, tiny_model):
    args = base_args.copy()
    args.runner = SimpleNamespace(checkpoint=None, early_stop={"patience": 999})

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)

    result = train_runner(
        model=tiny_model,
        task=tiny_task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        args=args,
        run_mode="epoch",
        max_epochs=2,
        eval_interval=1,
        save_dir=args.log_dir,
        print_freq=5,
    )

    assert result["final_epoch"] == 2
    assert result["best_val_metric"] is not None


def test_train_runner_none_early_stop_uses_defaults(base_args, tiny_task, tiny_model):
    args = base_args.copy()
    args.runner = SimpleNamespace(checkpoint={}, early_stop=None)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)

    result = train_runner(
        model=tiny_model,
        task=tiny_task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        args=args,
        run_mode="epoch",
        max_epochs=2,
        eval_interval=1,
        save_dir=args.log_dir,
        print_freq=5,
    )

    assert result["final_epoch"] == 2
    assert result["best_val_metric"] is not None


def test_train_runner_none_run_mode_raises_value_error(base_args, tiny_task, tiny_model):
    args = base_args.copy()
    args.runner = SimpleNamespace(checkpoint={}, early_stop={"patience": 999})

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)

    with pytest.raises(ValueError, match="run_mode cannot be None"):
        train_runner(
            model=tiny_model,
            task=tiny_task,
            loss_fn=loss_fn,
            optimizer=optimizer,
            args=args,
            run_mode=None,
            max_epochs=2,
            eval_interval=1,
            save_dir=args.log_dir,
            print_freq=5,
        )


def test_train_runner_none_eval_interval_falls_back_to_one(base_args, tiny_task, tiny_model):
    args = base_args.copy()
    args.runner = SimpleNamespace(checkpoint={}, early_stop={"patience": 999})

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)

    result = train_runner(
        model=tiny_model,
        task=tiny_task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        args=args,
        run_mode="epoch",
        max_epochs=2,
        eval_interval=None,
        save_dir=args.log_dir,
        print_freq=5,
    )

    assert result["final_epoch"] == 2
    assert result["best_val_metric"] is not None


def test_train_runner_none_render_type_falls_back_to_auto(base_args, tiny_task, tiny_model):
    args = base_args.copy()
    args.runner = SimpleNamespace(checkpoint={}, early_stop={"patience": 999})
    args.render_type = None

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1.0e-3)

    result = train_runner(
        model=tiny_model,
        task=tiny_task,
        loss_fn=loss_fn,
        optimizer=optimizer,
        args=args,
        run_mode="epoch",
        max_epochs=2,
        eval_interval=1,
        save_dir=args.log_dir,
        print_freq=5,
    )

    assert result["final_epoch"] == 2
    assert result["best_val_metric"] is not None
