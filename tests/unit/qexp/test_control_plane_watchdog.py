from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp.agent import lifecycle
from qqtools.plugins.qexp.agent import scheduler_diagnostics as diagnostic_module
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.control_plane import ControlPlaneStopBlockedError, _MachineControlPlane
from qqtools.plugins.qexp.agent.control_plane_health import ControlPlaneHealth
from qqtools.plugins.qexp.agent.lifecycle import MachineAgentStopError
from qqtools.plugins.qexp.agent.scheduler_diagnostics import SchedulerDiagnosticStore


@pytest.fixture(autouse=True)
def _deterministic_diagnostic_publication_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep watchdog behavior tests independent of host filesystem latency."""
    monkeypatch.setattr(diagnostic_module, "time", SimpleNamespace(monotonic_ns=lambda: 0))


class _Clock:
    def __init__(self) -> None:
        self.monotonic_value = 0.0
        self.utc_value = datetime(2026, 10, 16, tzinfo=timezone.utc)

    def monotonic(self) -> float:
        return self.monotonic_value

    def utc_now(self) -> str:
        return self.utc_value.isoformat().replace("+00:00", "Z")

    def advance(self, seconds: float) -> None:
        self.monotonic_value += seconds
        self.utc_value += timedelta(seconds=seconds)


def _plane_with_clock(tmp_path) -> tuple[_MachineControlPlane, ControlPlaneHealth, _Clock]:
    runtime = MachineRuntime(tmp_path / "runtime")
    plane = _MachineControlPlane(
        runtime,
        instance_id="agent-instance",
        loop_interval=5.0,
        started_at="2026-10-16T00:00:00Z",
        available_gpus=[],
    )
    clock = _Clock()
    health = ControlPlaneHealth(
        runtime.root,
        instance_id="agent-instance",
        interval_seconds=5.0,
        monotonic=clock.monotonic,
        utc_clock=clock.utc_now,
    )
    plane._health = health
    return plane, health, clock


def test_watchdog_distinguishes_stalled_cycle_from_stopped_thread(tmp_path, monkeypatch) -> None:
    plane, health, clock = _plane_with_clock(tmp_path)
    sequence = health.begin_heartbeat_cycle()
    clock.advance(15.01)

    monkeypatch.setattr(plane, "_heartbeat_thread_liveness", lambda: True)
    stalled = plane.monitor_health()
    assert stalled["state"] == "stalled"
    assert stalled["heartbeat"]["state"] == "stalled"
    assert stalled["heartbeat"]["cycle_sequence"] == sequence

    monkeypatch.setattr(plane, "_heartbeat_thread_liveness", lambda: False)
    stopped = plane.monitor_health()
    assert stopped["state"] == "stopped"
    assert stopped["heartbeat"]["state"] == "stopped"


def test_watchdog_finding_survives_thread_exit_and_resolves_for_matching_cycle(tmp_path, monkeypatch) -> None:
    plane, health, clock = _plane_with_clock(tmp_path)
    sequence = health.begin_heartbeat_cycle()
    clock.advance(15.01)
    monkeypatch.setattr(plane, "_heartbeat_thread_liveness", lambda: True)

    plane.monitor_health()
    active = SchedulerDiagnosticStore(plane._runtime).active_view(producer="control_plane_health")
    assert active["total"] == 1
    assert active["items"][0]["identity"]["reason_code"] == "heartbeat_cycle_stalled"
    assert active["items"][0]["identity"]["target_id"] == "agent-instance"

    health.complete_heartbeat_cycle(sequence, "queued")
    plane.monitor_health()
    assert SchedulerDiagnosticStore(plane._runtime).active_view(producer="control_plane_health")["total"] == 0
    history = SchedulerDiagnosticStore(plane._runtime).history_view(limit=32)
    assert history["items"][0]["identity"]["reason_code"] == "heartbeat_cycle_stalled"


def test_watchdog_reports_repeated_responsive_failure_without_stalled_classification(tmp_path, monkeypatch) -> None:
    plane, health, clock = _plane_with_clock(tmp_path)
    first = health.begin_heartbeat_cycle()
    health.complete_heartbeat_cycle(first, "queued")
    clock.advance(15.01)
    second = health.begin_heartbeat_cycle()
    health.complete_heartbeat_cycle(second, "registry_unavailable")
    monkeypatch.setattr(plane, "_heartbeat_thread_liveness", lambda: True)

    snapshot = plane.monitor_health()

    assert snapshot["heartbeat"]["state"] == "degraded"
    active = SchedulerDiagnosticStore(plane._runtime).active_view(producer="control_plane_health")
    assert [item["identity"]["reason_code"] for item in active["items"]] == ["heartbeat_repeated_failure"]


def test_project_snapshot_overdue_finding_resolves_for_exact_publication(tmp_path) -> None:
    plane, health, _clock = _plane_with_clock(tmp_path)
    publication = {
        "runtime_id": plane._runtime.instance_id,
        "agent_instance_id": "agent-instance",
        "registry_revision": 7,
        "executor_epoch": "a" * 32,
        "project_id": "project-1",
        "registration_generation": "registration-1",
        "operation_kind": "machine_snapshot_publish",
        "request_id": "request-1",
        "stage": "started",
        "last_attempt_at": "2026-10-16T00:00:00Z",
        "last_published_at": None,
        "consecutive_failures": 0,
        "retry_at": None,
        "reason_code": None,
        "worker_elapsed_seconds": 15.1,
        "overdue": True,
        "effective_stale_after_seconds": 15.0,
    }
    health.update_project_snapshot_publications({"total": 1, "omitted_count": 0, "items": [publication]})
    plane._reconcile_scheduler_findings(health.snapshot(heartbeat_thread_alive=True))

    store = SchedulerDiagnosticStore(plane._runtime)
    active = store.active_view(producer="control_plane_health")
    assert active["total"] == 1
    identity = active["items"][0]["identity"]
    assert identity["reason_code"] == "project_snapshot_overdue"
    assert identity["project_id"] == "project-1"
    assert identity["executor_epoch"] == "a" * 32
    assert identity["request_id"] == "request-1"

    health.update_project_snapshot_publications({"total": 0, "omitted_count": 0, "items": []})
    plane._reconcile_scheduler_findings(health.snapshot(heartbeat_thread_alive=True))
    assert store.active_view(producer="control_plane_health")["total"] == 0
    history = store.history_view(limit=32)
    assert history["items"][0]["identity"]["reason_code"] == "project_snapshot_overdue"


def test_successor_watchdog_archives_predecessor_finding_without_claiming_recovery(tmp_path, monkeypatch) -> None:
    runtime = MachineRuntime(tmp_path / "runtime")
    predecessor = _MachineControlPlane(
        runtime,
        instance_id="predecessor-instance",
        loop_interval=5.0,
        started_at="2026-10-16T00:00:00Z",
        available_gpus=[],
    )
    predecessor._health.begin_heartbeat_cycle()
    monkeypatch.setattr(predecessor, "_heartbeat_thread_liveness", lambda: False)
    predecessor.monitor_health()

    successor = _MachineControlPlane(
        runtime,
        instance_id="successor-instance",
        loop_interval=5.0,
        started_at="2026-10-16T00:01:00Z",
        available_gpus=[],
    )
    monkeypatch.setattr(successor, "_heartbeat_thread_liveness", lambda: True)
    successor.monitor_health()

    store = SchedulerDiagnosticStore(runtime)
    assert store.active_view(producer="control_plane_health")["total"] == 0
    history = store.history_view(limit=32)
    assert history["items"][0]["identity"]["target_id"] == "predecessor-instance"
    assert history["items"][0]["identity"]["reason_code"] == "heartbeat_thread_stopped"


def test_scheduler_diagnostic_coverage_loss_becomes_active_then_resolves(tmp_path, monkeypatch) -> None:
    plane, _health, _clock = _plane_with_clock(tmp_path)
    monkeypatch.setattr(plane, "_heartbeat_thread_liveness", lambda: True)
    actual_active_view = plane._scheduler_diagnostics.active_view
    failed_once = False

    def fail_once(*args, **kwargs):
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise OSError("temporary scheduler diagnostic failure")
        return actual_active_view(*args, **kwargs)

    monkeypatch.setattr(plane._scheduler_diagnostics, "active_view", fail_once)
    first = plane.monitor_health()
    assert first["heartbeat"]["diagnostic_coverage_degraded"] is True

    plane.monitor_health()
    store = SchedulerDiagnosticStore(plane._runtime)
    active = store.active_view(producer="control_plane_health")
    assert [item["identity"]["reason_code"] for item in active["items"]] == ["heartbeat_diagnostic_coverage_loss"]

    third = plane.monitor_health()
    assert third["heartbeat"]["diagnostic_coverage_degraded"] is False
    assert store.active_view(producer="control_plane_health")["total"] == 0
    history = store.history_view(limit=32)
    assert history["items"][0]["identity"]["reason_code"] == "heartbeat_diagnostic_coverage_loss"


def test_control_plane_stop_is_bounded_when_heartbeat_thread_hangs(tmp_path) -> None:
    runtime = MachineRuntime(tmp_path / "runtime")
    plane = _MachineControlPlane(
        runtime,
        instance_id="agent-instance",
        loop_interval=5.0,
        started_at="2026-10-16T00:00:00Z",
        available_gpus=[],
    )
    release = threading.Event()
    hung = threading.Thread(target=release.wait, name="test-hung-heartbeat", daemon=True)
    hung.start()
    plane._heartbeat_thread = hung

    try:
        with pytest.raises(ControlPlaneStopBlockedError) as raised:
            plane.stop(join_timeout=0.01)
    finally:
        release.set()
        hung.join(1.0)

    assert raised.value.instance_id == "agent-instance"
    assert raised.value.heartbeat_thread_alive is True
    assert "recovery_blocked" in str(raised.value)


def test_process_stop_timeout_reports_recovery_blocked_without_starting_a_successor(tmp_path, monkeypatch) -> None:
    runtime = MachineRuntime(tmp_path / "runtime")
    health = ControlPlaneHealth(runtime.root, instance_id="agent-instance", interval_seconds=5.0)
    sequence = health.begin_heartbeat_cycle()
    health.complete_heartbeat_cycle(sequence, "queued")
    health.monitor_heartbeat(False)
    assert health.persist(heartbeat_thread_alive=False)

    monkeypatch.setattr(lifecycle, "_active_machine_identity", lambda _runtime: (123, "agent-instance", 456))
    monkeypatch.setattr(lifecycle.os, "kill", lambda _pid, _signal: None)
    monkeypatch.setattr(lifecycle, "_pid_start_time_ticks", lambda _pid: 456)
    monotonic_values = iter((0.0, 11.0))
    monkeypatch.setattr(lifecycle.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(lifecycle.time, "sleep", lambda _seconds: None)

    with pytest.raises(MachineAgentStopError, match="recovery_blocked") as raised:
        lifecycle._stop_machine_agent_locked(runtime, timeout=10.0)

    message = str(raised.value)
    assert "agent-instance" in message
    assert "heartbeat_state='stopped'" in message
    assert "heartbeat_thread_alive=False" in message
