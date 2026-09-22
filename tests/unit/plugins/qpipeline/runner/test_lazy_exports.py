import pytest

import qqtools.plugins.qpipeline.runner as runner_package


def test_public_runner_export_is_loaded_once_and_cached(monkeypatch):
    name = "train_runner"
    missing = object()
    previous = runner_package.__dict__.pop(name, missing)
    expected = object()
    calls = []

    class FakeLazyImport:
        def resolve(self):
            calls.append(name)
            return expected

    monkeypatch.setitem(runner_package._LAZY_EXPORTS, name, FakeLazyImport())
    try:
        assert runner_package.train_runner is expected
        assert runner_package.train_runner is expected
        assert calls == [name]
    finally:
        runner_package.__dict__.pop(name, None)
        if previous is not missing:
            runner_package.__dict__[name] = previous


def test_unknown_runner_export_is_rejected():
    with pytest.raises(AttributeError, match="has no attribute 'missing_runner'"):
        runner_package.missing_runner
