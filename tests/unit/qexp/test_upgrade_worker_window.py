from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp.agent.project_io import worker
from qqtools.plugins.qexp.runtime.upgrade import framework


def _status():
    return {
        "state": "runnable",
        "pending": True,
        "can_run": True,
        "admission_blocked": False,
        "next_probe_at": None,
        "migrations": [{"name": "migration", "phase": "backfill", "state": "runnable"}],
        "invocation": {
            "slice_committed": True,
            "meaningful_progress": True,
            "contended_lock": None,
            "delta": None,
            "next_probe_at": None,
            "migration": "migration",
            "phase": "backfill",
        },
    }


def _run(monkeypatch, *, slice_seconds=0, discovery_seconds=0, transform=None):
    now = [0.0]
    calls = []

    class Coordinator:
        def __init__(self, cfg):
            pass

        def discover(self):
            now[0] += discovery_seconds
            return _status()

        def advance(self, **kwargs):
            calls.append(kwargs)
            now[0] += slice_seconds
            result = _status()
            if transform:
                transform(result)
            return result

    monkeypatch.setattr(framework, "UpgradeCoordinator", Coordinator)
    monkeypatch.setattr(worker, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(worker, "load_root_config", lambda *a: SimpleNamespace(shared_root="root"))
    monkeypatch.setattr(worker, "fenced_mutations", lambda *a: nullcontext())
    request = SimpleNamespace(canonical_shared_root="/unused", parameters={"machine_name": "machine"})
    result = worker._upgrade_service(request, None, {}, [False])
    return calls, result


@pytest.mark.parametrize(
    ("slice_seconds", "discovery_seconds", "expected"), [(0, 0, 8), (0.75, 0, 3), (3, 0, 1), (0.6, 1.5, 1), (0, 3, 1)]
)
def test_window_call_and_elapsed_limits(monkeypatch, slice_seconds, discovery_seconds, expected):
    calls, _ = _run(monkeypatch, slice_seconds=slice_seconds, discovery_seconds=discovery_seconds)
    assert calls == [{}] + [{"continuation": ("migration", "backfill")}] * (expected - 1)


@pytest.mark.parametrize(
    "stop",
    [
        "contention",
        "peer_progress",
        "no_progress",
        "uncommitted",
        "probe",
        "retry",
        "phase",
        "migration",
        "pause",
        "waiting",
        "io_failure",
        "missing_invocation",
    ],
)
def test_window_stops_without_own_runnable_progress(monkeypatch, stop):
    def transform(status):
        invocation = status["invocation"]
        if stop in {"contention", "peer_progress"}:
            invocation.update(contended_lock="upgrade", slice_committed=False, observed_progress=True)
        elif stop == "no_progress":
            invocation["meaningful_progress"] = False
        elif stop == "uncommitted":
            invocation["slice_committed"] = False
        elif stop == "probe":
            invocation["next_probe_at"] = "2999-01-01T00:00:00Z"
        elif stop == "retry":
            status["migrations"][0]["next_retry_at"] = "2999-01-01T00:00:00Z"
        elif stop in {"phase", "migration"}:
            status["migrations"][0]["phase" if stop == "phase" else "name"] = "other"
        elif stop in {"pause", "waiting", "io_failure"}:
            status["state"] = {"pause": "paused", "waiting": "waiting", "io_failure": "repair_required"}[stop]
        else:
            status.pop("invocation")

    calls, result = _run(monkeypatch, transform=transform)
    assert calls == [{}]
    expected = _status()
    transform(expected)
    assert result["state"] == expected["state"]
    assert set(result) == {"state", "pending", "can_run", "admission_blocked", "idle_blocking", "next_probe_at"}


def test_generic_invocation_delta_can_continue(monkeypatch):
    def transform(status):
        status["invocation"].pop("meaningful_progress")
        status["invocation"]["delta"] = {"kind": "units", "before": 1, "after": 2}

    calls, _ = _run(monkeypatch, transform=transform)
    assert len(calls) == 8


@pytest.mark.parametrize("processed", [None, 0, False, "1", 1])
def test_dependency_progress_uses_processed_entries_not_charged_records(processed):
    from qqtools.plugins.qexp.runtime.upgrade.task_dependencies_v2 import _phase_result

    context = SimpleNamespace(
        slice_budget=SimpleNamespace(used=lambda **kwargs: {"records": 1, "metadata_ops": 1, "io_bytes": 1})
    )
    result = _phase_result(context, "progressed", detail={"projection": {"work_items": processed}})
    assert result.work_items == 1
    assert result.meaningful_progress is (type(processed) is int and processed > 0)
