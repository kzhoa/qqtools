import pytest
import torch
import torch.nn as nn

from qqtools.plugins.qpipeline import Stage
from qqtools.plugins.qpipeline.runner.agent import RunningAgent
from qqtools.plugins.qpipeline.runner.runner_utils.evaluation import (
    EvaluationResult,
    LoaderEvaluation,
    ModelEvaluation,
    ScoreTarget,
    StageEvaluation,
    TrainingResult,
    resolve_loader_group,
)
from qqtools.plugins.qpipeline.runner.runner_utils.eval_formatter import EvalFormatter
from qqtools.plugins.qpipeline.runner.runner_utils.types import RunConfig, RunningState

from .conftest import SimpleModel, SimpleTask


def _make_agent(task):
    model = SimpleModel(input_dim=10)
    return RunningAgent(
        model=model,
        task=task,
        loss_fn=nn.MSELoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1.0e-3),
        config=RunConfig(device=torch.device("cpu")),
        device=torch.device("cpu"),
    )


def test_multi_loader_evaluation_preserves_order_and_uses_stage_score():
    task = SimpleTask(num_samples=40)
    task.val_loader = {"b1": task.val_loader, "b2": task.test_loader}
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
    assert [loader.name for loader in standard.stages[0].loaders] == ["b1", "b2"]
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
        best_model_tracker=type("Tracker", (), {"mode": "min", "best_metric": 0.1, "best_epoch": 1, "best_step": 2})(),
        lr=None,
        color_new_best=False,
    )

    assert any("Best:" in line for line in summary_lines)
    assert any("NewBest" in line for line in summary_lines)
    assert any("train:interval" in line for line in table_lines)
    assert any("val:b1" in line for line in table_lines)
