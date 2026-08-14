"""Evaluation-boundary contracts after the listener-to-binding migration."""

from unittest.mock import MagicMock

import torch
import torch.nn as nn
import torch.optim as optim

from qqtools.plugins.qpipeline.runner.contracts import EventListenerBindings, ObserverBindings
from qqtools.plugins.qpipeline.runner.hooks import RunnerHooks
from qqtools.plugins.qpipeline.runner.runner import RunningAgent
from qqtools.plugins.qpipeline.runner.runner_utils.ckp_manager import CheckpointPlugin, CheckpointPolicy
from qqtools.plugins.qpipeline.runner.runner_utils.earlystop import EarlyStopController, EarlyStopper
from qqtools.plugins.qpipeline.runner.runner_utils.types import RunConfig, RunMode

from .conftest import SimpleModel, SimpleTask


def _build_agent(*, listeners=None, observers=None, controller=None, hooks=None, config=None):
    task = SimpleTask(num_samples=20, num_features=10)
    model = SimpleModel(input_dim=10)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return RunningAgent(
        model=model,
        task=task,
        loss_fn=nn.MSELoss(),
        optimizer=optimizer,
        config=config or RunConfig(run_mode=RunMode.STEP, max_steps=3, eval_interval=1),
        device=torch.device("cpu"),
        event_listeners=listeners,
        observers=observers,
        early_stop_controller=controller,
        hooks=hooks,
    )


def test_evaluation_observers_receive_committed_fact_after_checkpoint():
    order = []
    hooks = RunnerHooks()
    manager = MagicMock(rank=0)
    manager.save.side_effect = lambda *args, **kwargs: order.append("checkpoint") or "checkpoint.pt"
    agent = _build_agent(hooks=hooks)
    plugin = CheckpointPlugin(
        checkpoint_manager=manager,
        model=agent.model,
        task=agent.task,
        state=agent.state,
        policy=CheckpointPolicy(1, None, False, False),
        optimizer=agent.optimizer,
    )
    plugin.register(hooks)
    hooks.freeze()

    observers = ObserverBindings()
    observers.bind("evaluation_committed", lambda fact: order.append("observer"))
    observers.freeze()
    agent.observers = observers

    agent.run()

    assert order.index("checkpoint") < order.index("observer")


def test_validation_listener_cannot_mutate_control_and_runs_before_checkpoint():
    order = []
    listeners = EventListenerBindings()
    listeners.bind("validation", lambda context: order.append("task"))
    listeners.freeze()
    hooks = RunnerHooks()
    agent = _build_agent(listeners=listeners, hooks=hooks)
    manager = MagicMock(rank=0)
    manager.save.side_effect = lambda *args, **kwargs: order.append("checkpoint") or "checkpoint.pt"
    CheckpointPlugin(
        checkpoint_manager=manager,
        model=agent.model,
        task=agent.task,
        state=agent.state,
        policy=CheckpointPolicy(1, None, False, False),
        optimizer=agent.optimizer,
    ).register(hooks)
    hooks.freeze()

    agent.run()

    assert order.index("task") < order.index("checkpoint")


def test_early_stop_uses_typed_controller_and_terminal_observer():
    stopper = EarlyStopper(
        patiences={"val_metric": 1}, mode={"val_metric": "min"}, min_delta={"val_metric": 0.0}
    )
    controller = EarlyStopController(stopper, target="val_metric")
    received = []
    observers = ObserverBindings()
    observers.bind("early_stop", received.append, policy="settled_fatal")
    observers.freeze()
    agent = _build_agent(observers=observers, controller=controller)

    assert agent.run() == "early_stop"
    assert received and received[0].source == "early_stop"
