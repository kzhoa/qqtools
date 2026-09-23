"""Pure adapter/TTY tests; no training, GPU, or external services required."""

from types import SimpleNamespace

import pytest

from qqtools.plugins.qpipeline.integrations import qexp_progress
from qqtools.plugins.qpipeline.runner.contracts import (
    EpochCommittedFact,
    EpochStartedFact,
    EvaluationBatchCommittedFact,
    EvaluationCommittedFact,
    EvaluationStartedFact,
    TrainBoundaryCommittedFact,
)
from qqtools.plugins.qpipeline.runner.observation_plugins import ObservationContext
from qqtools.plugins.qpipeline.runner.runner_utils import render_mode
from qqtools.plugins.qpipeline.runner.runner_utils.evaluation import EvaluationResult
from qqtools.plugins.qpipeline.types import Stage
from qqtools.qexp import progress as progress_api


class Sink:
    def __init__(self):
        self.updates = []
        self.closed = False

    def offer(self, **payload):
        parts = payload.pop("message_parts")
        renderer = payload.pop("render_message")
        payload["message"] = renderer(parts)
        self.updates.append(payload)
        return True

    def close(self):
        self.closed = True


@pytest.mark.parametrize("mode", [None, "auto"])
@pytest.mark.parametrize("rich,tqdm", [(True, True), (True, False), (False, True), (False, False)])
def test_non_tty_auto_never_uses_live_rendering(mode, rich, tqdm):
    assert render_mode.resolve_render_mode(mode, rich, tqdm, is_terminal=False)[0] == "plain"


@pytest.mark.parametrize(
    "mode,rich,tqdm,expected",
    [
        ("rich", True, True, "rich"),
        ("rich", False, True, "tqdm"),
        ("tqdm", True, True, "tqdm"),
        ("plain", True, True, "plain"),
    ],
)
def test_explicit_renderer_preserves_dependency_fallbacks(mode, rich, tqdm, expected):
    assert render_mode.resolve_render_mode(mode, rich, tqdm, is_terminal=False)[0] == expected


def test_actual_streams_are_probed_independently(monkeypatch):
    monkeypatch.setattr(render_mode.sys, "stdout", SimpleNamespace(isatty=lambda: False))
    monkeypatch.setattr(render_mode.sys, "stderr", SimpleNamespace(isatty=lambda: True))
    assert render_mode.resolve_render_mode("auto", True, True)[0] == "tqdm"
    monkeypatch.setattr(render_mode.sys, "stderr", object())
    assert render_mode.resolve_render_mode("auto", True, True)[0] == "plain"


def test_broken_isatty_is_noninteractive(monkeypatch):
    def broken():
        raise OSError("closed")

    monkeypatch.setattr(render_mode.sys, "stdout", SimpleNamespace(isatty=broken))
    monkeypatch.setattr(render_mode.sys, "stderr", SimpleNamespace(isatty=broken))
    assert render_mode.resolve_render_mode("auto", True, True)[0] == "plain"


def _observer(monkeypatch, *, rank=0, max_steps=15000, max_epochs=10):
    monkeypatch.setenv("QEXP_PROGRESS_PATH", "/unused")
    return qexp_progress.qexp_progress_factory(
        ObservationContext(rank=rank, max_steps=max_steps, max_epochs=max_epochs)
    )


@pytest.mark.parametrize("rank", [1, 2, 7])
def test_nonzero_rank_does_not_create_plugin(monkeypatch, rank):
    assert _observer(monkeypatch, rank=rank) is None


def test_adapter_absent_outside_qexp(monkeypatch):
    monkeypatch.delenv("QEXP_PROGRESS_PATH", raising=False)
    assert qexp_progress.qexp_progress_factory(ObservationContext(rank=0, max_steps=10, max_epochs=2)) is None


def test_fact_adapter_preserves_training_and_loader_local_evaluation(monkeypatch):
    sink = Sink()
    monkeypatch.setattr(progress_api, "_offer_managed_progress", sink.offer)
    observer = _observer(monkeypatch)
    callbacks = dict(observer.subscriptions)
    assert "progress_tick" not in callbacks
    assert "table_update" not in callbacks

    callbacks["epoch_started"](EpochStartedFact(global_step=7300, epoch=3, total_batches=100, max_epochs=10))
    assert sink.updates[-1] == dict(stage="train", current=7300, total=15000, unit="step", message="Epoch 4/10")
    callbacks["train_boundary"](
        TrainBoundaryCommittedFact(
            global_step=7340,
            epoch=3,
            batch_index=39,
            total_batches=100,
            did_optimizer_step=True,
            batch_metrics={},
            lr=None,
        )
    )
    assert sink.updates[-1]["message"] == "Epoch 4/10 · Batch 40/100"
    callbacks["evaluation_started"](
        EvaluationStartedFact(
            epoch=3,
            global_step=7340,
            total_batches=20,
            evaluation_stage="val",
            loader_name="Dataset B",
            loader_index=1,
            model_variant="standard",
        )
    )
    assert sink.updates[-1] == dict(
        stage="validation",
        current=0,
        total=20,
        unit="batch",
        message="Dataset B · Standard · Epoch 4/10",
    )
    callbacks["evaluation_batch_committed"](
        EvaluationBatchCommittedFact(stage=Stage.VAL, epoch=3, global_step=7340, batch_index=9, total_batches=20)
    )
    assert sink.updates[-1]["current"] == 10
    assert sink.updates[-1]["message"] == "Dataset B · Standard · Epoch 4/10"

    callbacks["evaluation_started"](
        EvaluationStartedFact(
            epoch=3,
            global_step=7340,
            total_batches=2,
            evaluation_stage="test",
            loader_name=None,
            loader_index=1,
            model_variant="ema",
        )
    )
    assert sink.updates[-1]["message"] == "Loader 2 · EMA · Epoch 4/10"
    callbacks["evaluation_batch_committed"](
        EvaluationBatchCommittedFact(stage=Stage.TEST, epoch=3, global_step=7340, batch_index=1, total_batches=2)
    )
    assert sink.updates[-1]["stage"] == "test"
    assert sink.updates[-1]["current"] == 2
    callbacks["evaluation_committed"](
        EvaluationCommittedFact(
            epoch=3,
            global_step=7340,
            evaluation=EvaluationResult(models=()),
            is_best=False,
            previous_best=None,
            lr=None,
        )
    )
    assert sink.updates[-1]["stage"] == "train"
    assert sink.updates[-1]["message"] == "Epoch 4/10 · Batch 40/100"
    callbacks["epoch_committed"](
        EpochCommittedFact(completed_epoch=3, next_epoch=4, global_step=7340, epoch_metrics={})
    )
    assert sink.updates[-1]["message"] == "Epoch 4/10 completed"


def test_empty_evaluation_and_unmatched_start_fallback(monkeypatch):
    sink = Sink()
    monkeypatch.setattr(progress_api, "_offer_managed_progress", sink.offer)
    observer = _observer(monkeypatch, max_steps=20, max_epochs=4)
    callbacks = dict(observer.subscriptions)
    callbacks["evaluation_started"](
        EvaluationStartedFact(
            epoch=1,
            global_step=5,
            total_batches=0,
            evaluation_stage="test",
            loader_name=None,
            loader_index=None,
            model_variant="ema",
        )
    )
    assert sink.updates[-1] == dict(stage="test", current=None, total=None, unit="batch", message="EMA · Epoch 2/4")
    callbacks["evaluation_batch_committed"](
        EvaluationBatchCommittedFact(stage=Stage.VAL, epoch=1, global_step=5, batch_index=2, total_batches=5)
    )
    assert sink.updates[-1] == dict(stage="validation", current=3, total=5, unit="batch", message=None)


@pytest.mark.parametrize("batch_index,total_batches", [(-1, 5), (5, 5), (0, 0)])
def test_invalid_counts_do_not_claim_batch_completion(monkeypatch, batch_index, total_batches):
    sink = Sink()
    monkeypatch.setattr(progress_api, "_offer_managed_progress", sink.offer)
    observer = _observer(monkeypatch)
    dict(observer.subscriptions)["evaluation_batch_committed"](
        EvaluationBatchCommittedFact(
            stage=Stage.VAL, epoch=0, global_step=0, batch_index=batch_index, total_batches=total_batches
        )
    )
    assert sink.updates[-1]["current"] is None
