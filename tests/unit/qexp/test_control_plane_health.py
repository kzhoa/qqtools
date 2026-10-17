from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone

import pytest

from qqtools.plugins.qexp.agent import lifecycle
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.control_plane import _MachineControlPlane
from qqtools.plugins.qexp.agent.control_plane_health import ControlPlaneHealth, read_control_plane_health
from qqtools.plugins.qexp.agent.setup import initialize_machine


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


def _health(tmp_path, clock: _Clock, **kwargs) -> ControlPlaneHealth:
    return ControlPlaneHealth(
        tmp_path,
        instance_id="agent-instance",
        interval_seconds=5.0,
        monotonic=clock.monotonic,
        utc_clock=clock.utc_now,
        **kwargs,
    )


def test_responsive_failures_degrade_after_three_intervals_and_recover(tmp_path) -> None:
    clock = _Clock()
    health = _health(tmp_path, clock)

    first = health.begin_heartbeat_cycle()
    clock.advance(0.25)
    health.complete_heartbeat_cycle(first, "queued")
    assert health.snapshot(heartbeat_thread_alive=True)["heartbeat"]["state"] == "healthy"

    clock.advance(1.0)
    second = health.begin_heartbeat_cycle()
    health.complete_heartbeat_cycle(second, "registry_unavailable")
    retrying = health.snapshot(heartbeat_thread_alive=True)["heartbeat"]
    assert retrying["state"] == "retrying"
    assert retrying["consecutive_failures"] == 1
    assert retrying["last_failure"] == {
        "reason_code": "registry_unavailable",
        "cycle_sequence": 2,
        "sanitized": True,
    }

    clock.advance(15.0)
    third = health.begin_heartbeat_cycle()
    health.complete_heartbeat_cycle(third, "capacity_unavailable")
    degraded = health.snapshot(heartbeat_thread_alive=True)["heartbeat"]
    assert degraded["state"] == "degraded"
    assert degraded["effective_stale_after_seconds"] == 15.0
    assert degraded["consecutive_failures"] == 2

    clock.advance(0.5)
    fourth = health.begin_heartbeat_cycle()
    health.complete_heartbeat_cycle(fourth, "queued")
    recovered = health.snapshot(heartbeat_thread_alive=True)["heartbeat"]
    assert recovered["state"] == "healthy"
    assert recovered["consecutive_failures"] == 0
    assert recovered["last_failure_at"] == degraded["last_failure_at"]


def test_failure_evidence_excludes_message_and_absolute_path(tmp_path) -> None:
    clock = _Clock()
    health = _health(tmp_path, clock)
    sequence = health.begin_heartbeat_cycle()

    try:
        raise RuntimeError("secret-token=/private/credentials.txt")
    except RuntimeError as exc:
        health.fail_heartbeat_cycle(sequence, exc)

    failure = health.snapshot(heartbeat_thread_alive=True)["heartbeat"]["last_failure"]
    encoded = json.dumps(failure, sort_keys=True)
    assert failure["reason_code"] == "heartbeat_cycle_exception"
    assert failure["exception_type"] == "RuntimeError"
    assert failure["filename"] == "test_control_plane_health.py"
    assert "secret-token" not in encoded
    assert "/private/credentials.txt" not in encoded


def test_persisted_health_is_instance_scoped_and_malformed_data_is_unknown(tmp_path) -> None:
    clock = _Clock()
    health = _health(tmp_path, clock)
    sequence = health.begin_heartbeat_cycle()
    health.complete_heartbeat_cycle(sequence, "queued")
    assert health.persist(heartbeat_thread_alive=True)

    current = read_control_plane_health(tmp_path, instance_id="agent-instance", thread_alive=True)
    assert current["state"] == "healthy"
    assert current["heartbeat"]["state"] == "healthy"

    stale = read_control_plane_health(tmp_path, instance_id="replacement-instance", thread_alive=True)
    assert stale["state"] == "unknown"
    assert stale["heartbeat"]["state"] == "unknown"

    (tmp_path / "control_plane_health.json").write_text("not-json", encoding="utf-8")
    malformed = read_control_plane_health(tmp_path, instance_id="agent-instance", thread_alive=True)
    assert malformed["state"] == "unknown"


def test_persisted_failure_reader_strips_unrecognized_sensitive_fields(tmp_path) -> None:
    clock = _Clock()
    health = _health(tmp_path, clock)
    sequence = health.begin_heartbeat_cycle()
    health.complete_heartbeat_cycle(sequence, "registry_unavailable")
    assert health.persist(heartbeat_thread_alive=True)

    path = tmp_path / "control_plane_health.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["heartbeat"]["last_failure"].update(
        {
            "message": "secret-token",
            "raw_path": "/private/credentials.txt",
            "filename": "/private/control_plane.py",
        }
    )
    path.write_text(json.dumps(record), encoding="utf-8")

    current = read_control_plane_health(tmp_path, instance_id="agent-instance", thread_alive=None)
    failure = current["heartbeat"]["last_failure"]
    assert failure["filename"] == "control_plane.py"
    assert "message" not in failure
    assert "raw_path" not in failure
    assert "secret-token" not in json.dumps(current, sort_keys=True)
    assert "/private" not in json.dumps(current, sort_keys=True)


def test_project_publication_reader_is_bounded_and_strips_tampered_fields(tmp_path) -> None:
    clock = _Clock()
    health = _health(tmp_path, clock)
    item = {
        "runtime_id": "runtime-1",
        "agent_instance_id": "agent-instance",
        "registry_revision": 7,
        "executor_epoch": "executor-1",
        "project_id": "project-1",
        "registration_generation": "registration-1",
        "operation_kind": "machine_snapshot_publish",
        "request_id": "request-1",
        "stage": "backoff",
        "last_attempt_at": clock.utc_now(),
        "last_published_at": None,
        "consecutive_failures": 1,
        "retry_at": clock.utc_now(),
        "reason_code": "worker_start_failed",
        "worker_elapsed_seconds": None,
        "overdue": True,
        "effective_stale_after_seconds": 15.0,
    }
    health.update_project_snapshot_publications({"total": 1, "omitted_count": 0, "items": [item]})
    assert health.persist(heartbeat_thread_alive=True)

    path = tmp_path / "control_plane_health.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["project_snapshot_publications"]["items"][0].update(
        {"message": "secret-token", "command": "curl credential", "path": "/private/project"}
    )
    record["project_snapshot_publications"]["items"].append(
        {**item, "project_id": "/private/project", "request_id": "secret token"}
    )
    path.write_text(json.dumps(record), encoding="utf-8")

    current = read_control_plane_health(tmp_path, instance_id="agent-instance", thread_alive=True)
    publications = current["project_snapshot_publications"]
    assert len(publications["items"]) == 1
    assert publications["items"][0]["project_id"] == "project-1"
    encoded = json.dumps(publications, sort_keys=True)
    assert "secret-token" not in encoded
    assert "curl credential" not in encoded
    assert "/private" not in encoded


def test_project_publication_logs_only_overdue_and_resolution_transitions(tmp_path) -> None:
    clock = _Clock()
    messages: list[str] = []
    health = _health(tmp_path, clock, private_log=messages.append)
    item = {
        "runtime_id": "runtime-1",
        "agent_instance_id": "agent-instance",
        "registry_revision": 7,
        "executor_epoch": "executor-1",
        "project_id": "project-1",
        "registration_generation": "registration-1",
        "operation_kind": "machine_snapshot_publish",
        "request_id": "request-1",
        "stage": "started",
        "last_attempt_at": clock.utc_now(),
        "last_published_at": None,
        "consecutive_failures": 0,
        "retry_at": None,
        "reason_code": None,
        "worker_elapsed_seconds": 15.0,
        "overdue": True,
        "effective_stale_after_seconds": 15.0,
    }
    view = {"total": 1, "omitted_count": 0, "items": [item]}

    health.update_project_snapshot_publications(view)
    health.update_project_snapshot_publications(view)
    health.update_project_snapshot_publications(
        {"total": 1, "omitted_count": 0, "items": [{**item, "stage": "published", "overdue": False}]}
    )

    events = [json.loads(message)["event"] for message in messages]
    assert events == ["project_snapshot_overdue", "project_snapshot_resolved"]


def test_instance_evidence_write_is_retried_on_a_later_cycle(tmp_path) -> None:
    clock = _Clock()
    attempts: list[dict[str, object]] = []

    def publish(evidence) -> bool:
        attempts.append(dict(evidence))
        return len(attempts) > 1

    health = _health(tmp_path, clock, publish_instance_evidence=publish)
    first = health.begin_heartbeat_cycle()
    health.complete_heartbeat_cycle(first, "registry_unavailable")
    second = health.begin_heartbeat_cycle()
    health.complete_heartbeat_cycle(second, "registry_unavailable")
    third = health.begin_heartbeat_cycle()
    health.complete_heartbeat_cycle(third, "queued")

    assert len(attempts) == 3
    assert attempts[1]["control_plane"]["maximum_consecutive_failures"] == 2
    assert attempts[1]["control_plane"]["coverage_loss_count"] == 1
    assert attempts[2]["control_plane"]["state"] == "healthy"
    assert attempts[2]["control_plane"]["last_success_at"] == "2026-10-16T00:00:00Z"
    assert attempts[2]["control_plane"]["last_failure_at"] == "2026-10-16T00:00:00Z"


def test_failed_health_persistence_is_counted_and_retried(tmp_path, monkeypatch) -> None:
    clock = _Clock()
    health = _health(tmp_path, clock)
    from qqtools.plugins.qexp.agent import control_plane_health

    actual_replace = control_plane_health.atomic_replace
    calls = 0

    def flaky_replace(path, value):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("temporary diagnostic storage failure")
        return actual_replace(path, value)

    monkeypatch.setattr(control_plane_health, "atomic_replace", flaky_replace)
    first = health.begin_heartbeat_cycle()
    health.complete_heartbeat_cycle(first, "registry_unavailable")
    assert health.persist(heartbeat_thread_alive=True) is False

    second = health.begin_heartbeat_cycle()
    health.complete_heartbeat_cycle(second, "queued")
    assert health.persist(heartbeat_thread_alive=True) is True

    current = read_control_plane_health(tmp_path, instance_id="agent-instance", thread_alive=None)
    assert calls == 2
    assert current["heartbeat"]["state"] == "healthy"
    assert current["heartbeat"]["diagnostic_coverage_loss_count"] == 1


def test_transition_logs_count_only_events_suppressed_since_the_previous_log(tmp_path) -> None:
    clock = _Clock()
    messages: list[str] = []
    health = _health(tmp_path, clock, private_log=messages.append)

    first = health.begin_heartbeat_cycle()
    health.complete_heartbeat_cycle(first, "registry_unavailable")
    clock.advance(1.0)
    second = health.begin_heartbeat_cycle()
    health.complete_heartbeat_cycle(second, "registry_unavailable")
    clock.advance(14.0)
    third = health.begin_heartbeat_cycle()
    health.complete_heartbeat_cycle(third, "registry_unavailable")
    fourth = health.begin_heartbeat_cycle()
    health.complete_heartbeat_cycle(fourth, "registry_unavailable")
    fifth = health.begin_heartbeat_cycle()
    health.complete_heartbeat_cycle(fifth, "queued")

    events = [json.loads(message) for message in messages]
    assert [event["event"] for event in events] == ["failure", "state_changed", "recovered"]
    assert [event["suppressed_identical_count"] for event in events] == [0, 1, 1]


def test_agent_status_does_not_substitute_process_liveness_for_heartbeat_liveness(tmp_path, monkeypatch) -> None:
    runtime = MachineRuntime(tmp_path / "runtime")
    initialize_machine(runtime, "gpu-1")
    observed: dict[str, object] = {}

    monkeypatch.setattr(lifecycle, "_active_machine_identity", lambda _runtime: (123, "agent-instance", 456))

    def read_health(_root, *, instance_id, thread_alive):
        observed.update(instance_id=instance_id, thread_alive=thread_alive)
        return {"instance_id": instance_id, "state": "unknown", "heartbeat": {}, "authority": {}}

    monkeypatch.setattr(lifecycle, "read_control_plane_health", read_health)

    status = lifecycle.get_machine_agent_status(runtime)

    assert status["is_running"] is True
    assert observed == {"instance_id": "agent-instance", "thread_alive": None}


def test_background_heartbeat_exception_is_contained_and_next_cycle_recovers(tmp_path, monkeypatch) -> None:
    runtime = MachineRuntime(tmp_path / "runtime")
    calls = 0
    recovered = threading.Event()
    plane = _MachineControlPlane(
        runtime,
        instance_id="agent-instance",
        loop_interval=0.01,
        started_at="2026-10-16T00:00:00Z",
        available_gpus=[],
    )

    def publish() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("one-cycle failure")
        recovered.set()
        plane._stop_event.set()
        return "queued"

    monkeypatch.setattr(plane, "_publish_project_heartbeats", publish)
    thread = threading.Thread(target=plane._run_heartbeat_loop)
    thread.start()
    assert recovered.wait(1.0)
    thread.join(1.0)

    assert not thread.is_alive()
    heartbeat = plane._health.snapshot(heartbeat_thread_alive=True)["heartbeat"]
    assert calls == 2
    assert heartbeat["cycle_sequence"] == 2
    assert heartbeat["state"] == "healthy"
    assert heartbeat["consecutive_failures"] == 0
    assert heartbeat["last_failure"]["exception_type"] == "RuntimeError"


def test_initial_heartbeat_exception_does_not_prevent_background_retry(tmp_path, monkeypatch) -> None:
    runtime = MachineRuntime(tmp_path / "runtime")
    calls = 0
    recovered = threading.Event()
    plane = _MachineControlPlane(
        runtime,
        instance_id="agent-instance",
        loop_interval=0.01,
        started_at="2026-10-16T00:00:00Z",
        available_gpus=[],
    )

    def publish() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("initial failure")
        recovered.set()
        return "queued"

    monkeypatch.setattr(plane, "_publish_project_heartbeats", publish)
    try:
        plane.start()
        assert recovered.wait(1.0)
        assert plane._heartbeat_thread.is_alive()
    finally:
        plane.stop()

    heartbeat = plane._health.snapshot(heartbeat_thread_alive=False)["heartbeat"]
    assert calls >= 2
    assert heartbeat["consecutive_failures"] == 0
    assert heartbeat["last_failure"]["exception_type"] == "RuntimeError"


def test_deadline_loop_does_not_swallow_base_exception(tmp_path) -> None:
    runtime = MachineRuntime(tmp_path / "runtime")
    plane = _MachineControlPlane(
        runtime,
        instance_id="agent-instance",
        loop_interval=0.01,
        started_at="2026-10-16T00:00:00Z",
        available_gpus=[],
    )

    def stop_now() -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        plane._run_deadline_loop(stop_now)
