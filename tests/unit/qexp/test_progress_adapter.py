"""Pure adapter/TTY tests; no training, GPU, or external services required."""

from types import SimpleNamespace

import pytest

from qqtools.plugins.qpipeline.runner.runner_utils import qexp_progress, render_mode


class Sink:
    def __init__(self):
        self.updates = []
        self.closed = False

    def update(self, **payload):
        self.updates.append(payload)
        return True

    def close(self):
        self.closed = True


class Bindings:
    def __init__(self):
        self.callbacks = {}

    def bind(self, name, callback, *, policy):
        assert policy == "best_effort"
        self.callbacks[name] = callback


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


@pytest.mark.parametrize("rank", [1, 2, 7])
def test_nonzero_rank_does_not_bind_writer(monkeypatch, rank):
    monkeypatch.setenv("QEXP_PROGRESS_PATH", "/unused")
    monkeypatch.setattr(qexp_progress.progress_api, "update", lambda **_: pytest.fail("nonzero rank reported progress"))
    bindings = Bindings()
    assert qexp_progress.bind_qexp_progress(bindings, SimpleNamespace(rank=rank)) is None
    assert not bindings.callbacks


def test_adapter_absent_outside_qexp(monkeypatch):
    monkeypatch.delenv("QEXP_PROGRESS_PATH", raising=False)
    assert qexp_progress.bind_qexp_progress(Bindings(), SimpleNamespace(rank=0)) is None


def test_peer_fact_adapter_and_optimizer_step_semantics(monkeypatch):
    sink = Sink()
    monkeypatch.setenv("QEXP_PROGRESS_PATH", "/unused")
    monkeypatch.setattr(qexp_progress.progress_api, "update", sink.update)
    monkeypatch.setattr(qexp_progress.progress_api, "flush", lambda **_: sink.close())
    bindings = Bindings()
    observer = qexp_progress.bind_qexp_progress(bindings, SimpleNamespace(rank=0, max_steps=15000))
    train = SimpleNamespace(global_step=7340, epoch=3, batch_index=100, total_batches=200, stage="train")
    bindings.callbacks["train_boundary"](train)
    assert sink.updates[-1] == dict(stage="train", current=7340, total=15000, unit="step", message="epoch 3")
    bindings.callbacks["progress_tick"](train)
    assert len(sink.updates) == 1
    bindings.callbacks["evaluation_started"](train)
    assert sink.updates[-1]["stage"] == "evaluation"
    assert "current" not in sink.updates[-1]
    for stage, expected in (("val", "validation"), ("test", "test")):
        fact = SimpleNamespace(stage=SimpleNamespace(value=stage), batch_index=9, total_batches=100, global_step=7340)
        bindings.callbacks["progress_tick"](fact)
        assert sink.updates[-1]["stage"] == expected
        assert sink.updates[-1]["current"] == 10
    bindings.callbacks["evaluation_committed"](train)
    assert sink.updates[-1]["current"] == 7340
    assert sink.updates[-1]["stage"] == "train"
    observer.close()
    assert sink.closed


def test_epoch_mode_unknown_total(monkeypatch):
    sink = Sink()
    monkeypatch.setattr(qexp_progress.progress_api, "update", sink.update)
    observer = qexp_progress.QexpProgressObserver(SimpleNamespace(max_steps=None))
    observer.on_train(SimpleNamespace(global_step=10, epoch=2))
    assert sink.updates[-1]["total"] is None


def test_framework_and_user_calls_share_same_progress_api(monkeypatch):
    calls = []
    monkeypatch.setenv("QEXP_PROGRESS_PATH", "/unused")
    monkeypatch.setattr(qexp_progress.progress_api, "update", lambda **payload: calls.append(payload) or True)
    bindings = Bindings()
    observer = qexp_progress.bind_qexp_progress(bindings, SimpleNamespace(rank=0, max_steps=10))
    bindings.callbacks["train_boundary"](SimpleNamespace(global_step=2, epoch=0))
    qexp_progress.progress_api.update(stage="custom", current=1)
    assert [item["stage"] for item in calls] == ["train", "custom"]
    assert observer is not None
