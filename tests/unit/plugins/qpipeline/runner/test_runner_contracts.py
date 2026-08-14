import pytest

from qqtools.plugins.qpipeline.runner.contracts import (
    EventListenerBindings,
    ObserverBindings,
    ProgressTickFact,
    dispatch_protected_boundary,
    freeze_scalar_metrics,
)
from qqtools.plugins.qpipeline.types import Stage


def test_scalar_metric_snapshot_is_copied_and_read_only():
    source = {"loss": 1.0}
    snapshot = freeze_scalar_metrics(source)
    source["loss"] = 2.0

    assert snapshot["loss"] == 1.0
    with pytest.raises(TypeError):
        snapshot["loss"] = 3.0


def test_event_bindings_require_freeze_before_dispatch():
    bindings = EventListenerBindings()
    bindings.bind("epoch_start", lambda context: None)

    with pytest.raises(RuntimeError, match="frozen"):
        bindings.dispatch("epoch_start", object())


def test_best_effort_observer_disables_only_the_failed_callback():
    calls = []
    bindings = ObserverBindings()
    bindings.bind("progress_tick", lambda fact: (_ for _ in ()).throw(RuntimeError("sink failed")))
    bindings.bind("progress_tick", lambda fact: calls.append(fact.global_step))
    bindings.freeze()
    fact = ProgressTickFact(
        stage=Stage.TRAIN,
        epoch=0,
        global_step=1,
        batch_index=0,
        total_batches=1,
        batch_metrics=freeze_scalar_metrics({"loss": 1.0}),
        average_metrics=None,
        lr=None,
    )

    bindings.dispatch("progress_tick", fact)
    bindings.dispatch("progress_tick", fact)

    assert calls == [1, 1]


def test_local_protected_boundary_reraises_original_error():
    with pytest.raises(ValueError, match="task failure"):
        dispatch_protected_boundary(
            lambda: (_ for _ in ()).throw(ValueError("task failure")),
            distributed=False,
            requires_settlement=True,
            boundary_name="validation",
        )
