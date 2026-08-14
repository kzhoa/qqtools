from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from qqtools.plugins.qpipeline.runner.hooks import (
    LoopDirective,
    OptimizerStepEndContext,
    RunnerBoundaryContext,
    RunnerHooks,
    ValidationHookContext,
)
from qqtools.plugins.qpipeline.runner.runner_utils.ckp_manager import CheckpointPlugin, CheckpointPolicy
from qqtools.plugins.qpipeline.runner.runner_utils.evaluation import EvaluationResult
from qqtools.plugins.qpipeline.runner.runner_utils.types import RunMode


def test_runner_hooks_reject_duplicate_slots_and_registration_after_freeze():
    hooks = RunnerHooks()
    hooks.set_after_validation_hook(lambda context: None, provider_id="checkpoint.v1")

    with pytest.raises(RuntimeError, match="already owned"):
        hooks.set_after_validation_hook(lambda context: None, provider_id="other.v1")

    hooks.freeze()
    with pytest.raises(RuntimeError, match="frozen"):
        hooks.set_boundary_cursor_hook(lambda context: None, provider_id="checkpoint.v1")


def test_optimizer_step_hook_requires_closed_loop_directive():
    hooks = RunnerHooks()
    hooks.set_optimizer_step_end_hook(
        lambda context: LoopDirective(end_epoch=True),
        provider_id="runtime-control.v1",
    )
    hooks.freeze()

    directive = hooks.dispatch_optimizer_step_end(
        OptimizerStepEndContext(epoch=2, global_step=11, is_natural_epoch_end=False)
    )

    assert directive.end_epoch is True


def test_empty_optimizer_dispatch_is_not_valid_without_registered_hook():
    hooks = RunnerHooks()
    hooks.freeze()
    with pytest.raises(RuntimeError, match="not installed"):
        hooks.dispatch_optimizer_step_end(
            OptimizerStepEndContext(epoch=0, global_step=1, is_natural_epoch_end=False)
        )


@pytest.mark.parametrize(
    "dispatch,context",
    [
        (
            lambda hooks: hooks.dispatch_after_validation,
            ValidationHookContext(
                epoch=0,
                global_step=1,
                evaluation=EvaluationResult(models=()),
                is_best=False,
                previous_best=None,
            ),
        ),
        (
            lambda hooks: hooks.dispatch_optimizer_step_end,
            OptimizerStepEndContext(epoch=0, global_step=1, is_natural_epoch_end=False),
        ),
    ],
)
def test_hook_dispatch_requires_frozen_composition(dispatch, context):
    hooks = RunnerHooks()

    with pytest.raises(RuntimeError, match="frozen before dispatch"):
        dispatch(hooks)(context)


@pytest.mark.parametrize("slot_name", ("after_validation", "boundary_cursor", "after_epoch_commit"))
def test_none_returning_hooks_reject_control_values(slot_name):
    hooks = RunnerHooks()
    callback = lambda context: {"unexpected": "control"}
    if slot_name == "after_validation":
        hooks.set_after_validation_hook(callback, provider_id="test.v1")
        context = ValidationHookContext(
            epoch=0,
            global_step=1,
            evaluation=EvaluationResult(models=()),
            is_best=False,
            previous_best=None,
        )
        dispatch = hooks.dispatch_after_validation
    else:
        callback_setter = (
            hooks.set_boundary_cursor_hook
            if slot_name == "boundary_cursor"
            else hooks.set_after_epoch_commit_hook
        )
        callback_setter(callback, provider_id="test.v1")
        context = RunnerBoundaryContext(
            epoch=0,
            global_step=1,
            run_mode=RunMode.STEP,
            did_optimizer_step=True,
            is_epoch_end=False,
            terminal_candidate=False,
            latest_train_loss=None,
        )
        dispatch = (
            hooks.dispatch_at_boundary_cursor
            if slot_name == "boundary_cursor"
            else hooks.dispatch_after_epoch_commit
        )
    hooks.freeze()

    with pytest.raises(TypeError, match=rf"{slot_name!r} must return None"):
        dispatch(context)


def test_checkpoint_plugin_registration_is_atomic_when_a_slot_is_owned():
    hooks = RunnerHooks()
    hooks.set_boundary_cursor_hook(lambda context: None, provider_id="other.v1")
    plugin = CheckpointPlugin(
        checkpoint_manager=Mock(),
        model=Mock(),
        task=Mock(),
        state=Mock(),
        policy=CheckpointPolicy(1, None, False, False),
    )

    with pytest.raises(RuntimeError, match="boundary_cursor.*already owned"):
        plugin.register(hooks)

    hooks.set_after_validation_hook(lambda context: None, provider_id="test.v1")
    hooks.set_after_epoch_commit_hook(lambda context: None, provider_id="test.v1")


def test_checkpoint_plugin_owns_configured_restore(tmp_path):
    checkpoint_file = tmp_path / "resume.pt"
    checkpoint_file.touch()
    checkpoint_manager = Mock()
    logger = Mock()
    model = Mock()
    task = Mock()
    optimizer = Mock()
    scheduler = Mock()
    ema_model = Mock()
    state = Mock()
    early_stopper = Mock()
    best_model_tracker = Mock()
    plugin = CheckpointPlugin(
        checkpoint_manager=checkpoint_manager,
        model=model,
        task=task,
        state=state,
        policy=CheckpointPolicy(1, str(checkpoint_file), False, False),
        optimizer=optimizer,
        scheduler=scheduler,
        ema_model=ema_model,
        early_stopper=early_stopper,
        best_model_tracker=best_model_tracker,
        logger=logger,
    )
    device = torch.device("cpu")

    assert plugin.restore_if_requested(device) is True
    checkpoint_manager.load.assert_called_once_with(
        str(checkpoint_file),
        device,
        model,
        task,
        optimizer,
        scheduler,
        ema_model,
        state,
        early_stopper,
        best_model_tracker,
    )
    logger.info.assert_called_once_with(f"Loaded checkpoint from {checkpoint_file}")


@pytest.mark.parametrize(
    "policy_args",
    [
        (True, None, False, False),
        (0, None, False, False),
        (1, 42, False, False),
        (1, None, "false", False),
        (1, None, False, "false"),
    ],
)
def test_checkpoint_policy_rejects_unresolved_values(policy_args):
    with pytest.raises(ValueError):
        CheckpointPolicy(*policy_args)
