"""Optional observation plugins cannot alter training dispatch or lifecycle."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from qqtools.plugins.qpipeline.runner.contracts import ObserverBindings
from qqtools.plugins.qpipeline.runner.observation_plugins import ObservationContext, install_observation_plugins


@dataclass
class Plugin:
    identifier: str
    subscriptions: tuple
    closes: list[str]

    def close(self):
        self.closes.append(self.identifier)


def context():
    return ObservationContext(rank=0, max_steps=10, max_epochs=2)


def test_plugins_install_in_order_and_close_in_reverse_once():
    calls, closes = [], []
    first = Plugin("first", (("epoch_started", lambda fact: calls.append("first")),), closes)
    second = Plugin("second", (("epoch_started", lambda fact: calls.append("second")),), closes)
    bindings = ObserverBindings()
    lifecycle = install_observation_plugins((lambda _: first, lambda _: second), context(), bindings)
    bindings.freeze()

    bindings.dispatch("epoch_started", SimpleNamespace())
    lifecycle.close()
    lifecycle.close()

    assert calls == ["first", "second"]
    assert closes == ["second", "first"]


def test_inactive_factory_has_no_failure_diagnostic():
    lifecycle = install_observation_plugins((lambda _: None,), context(), ObserverBindings())
    assert lifecycle.drain_diagnostics() == ()


@pytest.mark.parametrize("event", ["progress_tick", "table_update", "early_stop", "missing_event"])
def test_invalid_plugin_bundle_is_atomic_and_rejected_instance_is_closed(event):
    calls, closes = [], []
    bad = Plugin(
        "bad",
        (("epoch_started", lambda fact: calls.append("bad")), (event, lambda fact: calls.append("invalid"))),
        closes,
    )
    good = Plugin("good", (("epoch_started", lambda fact: calls.append("good")),), closes)
    bindings = ObserverBindings()
    lifecycle = install_observation_plugins((lambda _: bad, lambda _: good), context(), bindings)
    bindings.freeze()
    bindings.dispatch("epoch_started", object())
    lifecycle.close()

    assert calls == ["good"]
    assert closes == ["bad", "good"]


def test_duplicate_identity_and_factory_failure_do_not_disable_peers():
    calls, closes = [], []
    first = Plugin("same", (("epoch_started", lambda fact: calls.append(1)),), closes)
    duplicate = Plugin("same", (("epoch_started", lambda fact: calls.append(2)),), closes)
    last = Plugin("last", (("epoch_started", lambda fact: calls.append(3)),), closes)

    def broken(_):
        raise RuntimeError("factory failed")

    bindings = ObserverBindings()
    lifecycle = install_observation_plugins(
        (lambda _: first, broken, lambda _: duplicate, lambda _: last), context(), bindings
    )
    bindings.freeze()
    bindings.dispatch("epoch_started", object())
    lifecycle.close()

    assert calls == [3]
    assert closes == ["same", "same", "last"]


def test_close_failure_does_not_prevent_reverse_cleanup():
    closes = []

    class BrokenClose(Plugin):
        def close(self):
            closes.append(self.identifier)
            raise OSError("storage unavailable")

    first = Plugin("first", (), closes)
    broken = BrokenClose("broken", (), closes)
    last = Plugin("last", (), closes)
    lifecycle = install_observation_plugins(
        (lambda _: first, lambda _: broken, lambda _: last), context(), ObserverBindings()
    )

    lifecycle.close()
    assert closes == ["last", "broken", "first"]


def test_failed_bundle_install_closes_rejected_instance():
    closes = []
    plugin = Plugin("late", (("epoch_started", lambda fact: None),), closes)
    bindings = ObserverBindings()
    bindings.freeze()

    lifecycle = install_observation_plugins((lambda _: plugin,), context(), bindings)
    lifecycle.close()

    assert closes == ["late"]


def test_callback_failure_does_not_wait_on_logger_or_disable_peer():
    calls = []

    class BlockingLogger:
        def debug(self, *args, **kwargs):
            pytest.fail("optional callback dispatched into synchronous logger")

    def fail(_):
        raise ValueError("failed")

    broken = Plugin("broken", (("epoch_started", fail),), [])
    peer = Plugin("peer", (("epoch_started", lambda fact: calls.append(1)),), [])
    bindings = ObserverBindings(logger=BlockingLogger())
    lifecycle = install_observation_plugins((lambda _: broken, lambda _: peer), context(), bindings)
    bindings.freeze()
    bindings.dispatch("epoch_started", object())
    bindings.dispatch("epoch_started", object())
    assert calls == [1, 1]
    assert len(lifecycle.drain_diagnostics()) == 1
    assert lifecycle.drain_diagnostics() == ()
