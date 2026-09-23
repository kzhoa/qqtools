from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from qqtools.plugins.qpipeline import Stage
from qqtools.plugins.qpipeline.runner import agent as agent_module
from qqtools.plugins.qpipeline.runner.agent import RunningAgent
from qqtools.plugins.qpipeline.runner.contracts import EvaluationCommittedFact, ObserverBindings
from qqtools.plugins.qpipeline.runner.runner_utils.best_model import BestMetricSnapshot
from qqtools.plugins.qpipeline.runner.runner_utils.eval_formatter import EvalFormatter, EvalSummaryObserver
from qqtools.plugins.qpipeline.runner.runner_utils.evaluation import (
    EvaluationResult,
    LoaderEvaluation,
    ModelEvaluation,
    ScoreTarget,
    StageEvaluation,
    TrainingResult,
    resolve_loader_group,
)
from qqtools.plugins.qpipeline.runner.runner_utils.types import RunConfig, RunningState

from .conftest import SimpleModel, SimpleTask


def _make_agent(task, *, distributed=False, observers=None):
    model = SimpleModel(input_dim=10)
    return RunningAgent(
        model=model,
        task=task,
        loss_fn=nn.MSELoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1.0e-3),
        config=RunConfig(device=torch.device("cpu"), distributed=distributed),
        device=torch.device("cpu"),
        observers=observers,
    )


def test_evaluation_boundaries_identify_each_loader_and_model_before_ticks():
    task = SimpleTask(num_samples=40)
    task.val_loader = {"dataset-a": task.val_loader, "dataset-b": task.test_loader}
    events = []
    observers = ObserverBindings()
    observers.bind("evaluation_started", lambda fact: events.append(("start", fact)))
    observers.bind("progress_tick", lambda fact: events.append(("tick", fact)))
    observers.freeze()
    agent = _make_agent(task, observers=observers)

    agent._evaluate_model(agent.model, *agent._resolve_evaluation_loaders(), "standard")

    starts = [fact for kind, fact in events if kind == "start"]
    assert [fact.evaluation_stage for fact in starts] == ["val", "val", "test"]
    assert [fact.loader_name for fact in starts] == ["dataset-a", "dataset-b", None]
    assert [fact.loader_index for fact in starts] == [0, 1, None]
    assert {fact.model_variant for fact in starts} == {"standard"}
    assert events[0] == ("start", starts[0])
    for index, (kind, fact) in enumerate(events):
        if kind == "tick" and getattr(fact.stage, "value", fact.stage) == "val":
            assert any(prior_kind == "start" for prior_kind, _ in events[:index])

    empty = DataLoader(TensorDataset(torch.empty(0, 10), torch.empty(0, 1)), batch_size=4)
    events.clear()
    agent._evaluate_model(agent.model, [(None, empty)], [], "ema")
    assert len(events) == 1
    kind, boundary = events[0]
    assert kind == "start"
    assert boundary.model_variant == "ema"
    assert boundary.loader_index is None
    assert boundary.total_batches == 0


def test_metric_free_batch_facts_cover_completed_iterations_without_progress_ticks():
    task = SimpleTask(num_samples=256)
    facts = []
    observers = ObserverBindings()
    observers.bind("evaluation_batch_committed", facts.append)
    observers.freeze()
    agent = _make_agent(task, observers=observers)

    agent._evaluate_loader(agent.model, task.val_loader, Stage.VAL)

    assert len(facts) == len(task.val_loader)
    assert [fact.batch_index for fact in facts] == list(range(len(task.val_loader)))
    assert {fact.stage for fact in facts} == {Stage.VAL}
    assert {fact.total_batches for fact in facts} == {len(task.val_loader)}
    assert all(not hasattr(fact, "batch_metrics") for fact in facts)


def test_failed_evaluation_does_not_emit_a_fact_for_failed_iteration():
    task = SimpleTask(num_samples=256)
    facts = []
    observers = ObserverBindings()
    observers.bind("evaluation_batch_committed", facts.append)
    observers.freeze()
    agent = _make_agent(task, observers=observers)
    original = task.batch_metric
    calls = 0

    def fail_second(out, batch):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("batch failed")
        return original(out, batch)

    task.batch_metric = fail_second
    with pytest.raises(RuntimeError, match="batch failed"):
        agent._evaluate_loader(agent.model, task.val_loader, Stage.VAL)
    assert [fact.batch_index for fact in facts] == [0]


def test_multi_loader_evaluation_preserves_order_and_uses_stage_score():
    task = SimpleTask(num_samples=40)
    task.val_loader = {"b2": task.test_loader, "b1": task.val_loader}
    task.test_loader = {"holdout": task.test_loader}

    def post_metrics_to_value(result, *, stage):
        if stage is Stage.VAL:
            return result["b1"]["mse"]
        if stage is Stage.TEST:
            return None
        return result["mse"]

    task.post_metrics_to_value = post_metrics_to_value
    evaluation = _make_agent(task).evaluate_all_models()
    standard = evaluation.models[0]

    assert [stage.stage for stage in standard.stages] == [Stage.VAL, Stage.TEST]
    assert [loader.name for loader in standard.stages[0].loaders] == ["b2", "b1"]
    assert standard.stages[0].score is not None
    assert standard.stages[1].score is None
    assert evaluation.target_value("val_metric") == standard.stages[0].score
    assert evaluation.target_value("test_metric") is None


def test_loader_groups_are_reread_between_evaluation_boundaries():
    task = SimpleTask(num_samples=40)
    task.val_loader = None
    task.test_loader = None
    agent = _make_agent(task)

    assert agent.evaluate_all_models().models[0].stages == ()
    task.val_loader = {"later": task.train_loader}

    evaluation = agent.evaluate_all_models()
    assert evaluation.models[0].stages[0].loaders[0].name == "later"


def test_training_eval_checks_batch_counts_only_for_first_nonempty_snapshot(monkeypatch):
    task = SimpleTask(num_samples=8)
    agent = _make_agent(task, distributed=True)
    validate_flags = []

    def prepare(loader, **kwargs):
        validate_flags.append(kwargs["should_validate_batch_counts"])
        return loader

    monkeypatch.setattr(agent_module, "prepare_eval_loader_for_ddp", prepare)

    agent._prepare_evaluation_loader_snapshot([], [])
    assert agent._eval_counts_pending is True

    val_loaders = [(None, task.val_loader)]
    agent._prepare_evaluation_loader_snapshot(val_loaders, [])
    agent._prepare_evaluation_loader_snapshot(val_loaders, [])

    assert validate_flags == [True, False]
    assert agent._eval_counts_pending is False


@pytest.mark.parametrize(
    "loader, message",
    [
        ({}, "non-empty"),
        ({"valid": object()}, "DataLoader"),
    ],
)
def test_loader_group_rejects_invalid_contracts(loader, message):
    with pytest.raises((TypeError, ValueError), match=message):
        resolve_loader_group(loader, group_name="validation")


def test_raw_metric_named_metric_does_not_collide_with_stage_score():
    task = SimpleTask(num_samples=8)
    task.batch_metric = lambda output, batch: {"metric": (torch.tensor(1.0), 1)}
    evaluation = _make_agent(task).evaluate()

    stage = evaluation.models[0].stages[0]
    assert stage.loaders[0].metrics["metric"] == 1.0
    assert evaluation.target_value("val_metric") == stage.score


def test_score_target_contract_and_training_score():
    evaluation = EvaluationResult(
        training=TrainingResult(metrics={"loss": 0.2}, score=0.15),
        models=(
            ModelEvaluation(
                variant="standard",
                stages=(StageEvaluation(Stage.VAL, (), 0.1), StageEvaluation(Stage.TEST, (), 0.2)),
            ),
            ModelEvaluation(variant="ema", stages=(StageEvaluation(Stage.VAL, (), 0.05),)),
        ),
    )

    assert evaluation.target_value(ScoreTarget.TRAIN) == 0.15
    assert evaluation.target_value("val_metric") == 0.1
    assert evaluation.target_value("test_metric") == 0.2
    assert evaluation.target_value("ema_val_metric") == 0.05
    assert evaluation.target_value("ema_test_metric") is None
    with pytest.raises(ValueError, match="Allowed targets"):
        evaluation.target_value("val_mse")


def test_evaluation_state_tracks_current_and_latest_metrics_separately():
    state = RunningState()
    state.update_current_metrics({"val_metric": 1.0, "test_metric": 2.0}, is_evaluation_boundary=True)
    state.update_current_metrics({}, is_evaluation_boundary=True)

    assert state.current_val_metric is None
    assert state.current_test_metric is None
    assert state.latest_val_metric == 1.0
    assert state.latest_test_metric == 2.0


def test_structured_formatter_uses_stage_and_loader_records():
    evaluation = EvaluationResult(
        training=TrainingResult(metrics={"loss": 0.2}, score=0.15),
        models=(
            ModelEvaluation(
                variant="standard",
                stages=(
                    StageEvaluation(
                        stage=Stage.VAL,
                        loaders=(LoaderEvaluation("b1", {"force_mae": 0.1}),),
                        score=0.1,
                    ),
                ),
            ),
        ),
    )
    summary_lines, _, table_lines, _ = EvalFormatter.format_evaluation(
        evaluation,
        epoch=1,
        step=2,
        target_key="val_metric",
        is_best=True,
        previous_best={"metric": 0.2, "epoch": 0, "step": 1},
        best=BestMetricSnapshot(metric=0.1, epoch=1, step=2),
        lr=None,
        color_new_best=False,
    )

    assert any("Best:" in line for line in summary_lines)
    assert any("NewBest" in line for line in summary_lines)
    assert any("train:interval" in line for line in table_lines)
    assert any("val:b1" in line for line in table_lines)


def test_eval_summary_observer_logs_current_best_on_new_best():
    logger = Mock()
    observer = EvalSummaryObserver(logger, color_new_best=False)
    evaluation = EvaluationResult(
        models=(
            ModelEvaluation(
                variant="standard",
                stages=(StageEvaluation(stage=Stage.VAL, loaders=(), score=0.1),),
            ),
        ),
    )

    observer.on_evaluation_committed(
        EvaluationCommittedFact(
            epoch=1,
            global_step=2,
            evaluation=evaluation,
            is_best=True,
            previous_best=BestMetricSnapshot(metric=0.2, epoch=0, step=1),
            lr=None,
        )
    )

    assert "Best: val_metric = 0.100000 @ epoch 1, step 2" in logger.info.call_args_list[0].args[0]
