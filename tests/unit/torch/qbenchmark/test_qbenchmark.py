from types import SimpleNamespace

import qqtools.torch.qbenchmark as qb


class _DummyCudaDeviceContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class _DummyEvent:
    def __init__(self, enable_timing=True):
        self.enable_timing = enable_timing

    def record(self):
        return None

    def elapsed_time(self, other):
        return 2.0


class _DummyTimer:
    def __init__(self, cuda=True, verbose=False):
        self.duration = 0.123

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def test_get_duration(monkeypatch):
    calls = {"count": 0}

    def fn():
        calls["count"] += 1

    monkeypatch.setattr(qb.torch.cuda, "device", lambda *_args, **_kwargs: _DummyCudaDeviceContext())
    monkeypatch.setattr(qb.torch.cuda, "Event", _DummyEvent)
    monkeypatch.setattr(qb.torch.cuda, "synchronize", lambda: None)

    duration = qb.get_duration(4, fn, device="cuda", warm_up=True)
    assert duration == 0.008
    assert calls["count"] == 9


def test_get_duration_by_sync(monkeypatch):
    calls = {"count": 0}

    def fn():
        calls["count"] += 1

    monkeypatch.setattr(qb.qt, "Timer", _DummyTimer)

    duration = qb.get_duration_by_sync(3, fn)
    assert duration == 0.123
    assert calls["count"] == 3


def test_get_duration_by_device_interface(monkeypatch):
    calls = {"count": 0}

    def fn():
        calls["count"] += 1

    class DummyDeviceInterface:
        Event = _DummyEvent

        @staticmethod
        def synchronize():
            return None

    dummy_dynamo = SimpleNamespace(
        device_interface=SimpleNamespace(
            get_interface_for_device=lambda _device: DummyDeviceInterface,
        )
    )
    monkeypatch.setattr(qb.torch, "_dynamo", dummy_dynamo)

    duration = qb.get_duration_by_device_interface(2, fn, device="cuda", warm_up=False)
    assert duration == 0.004
    assert calls["count"] == 2
