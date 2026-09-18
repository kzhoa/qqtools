"""Comparison evidence must preserve phases and immutable counter snapshots."""

import json
from collections import deque
from pathlib import Path

import pytest

from tests.helpers.qexp.authority_measurement import AgentMeasurements
from tests.helpers.qexp.authority_reports import steady_operations, summarize


def test_counter_samples_do_not_change_when_observations_advance():
    measurements = AgentMeasurements()
    measurements.observe("store.read_json", 100)
    measurements.capture()
    measurements.observe("store.read_json", 200, has_failed=True)
    measurements.capture()
    first, second = measurements.samples
    assert first["operations"]["MainThread"]["store.read_json"] == {"calls": 1, "errors": 0, "total_ns": 100}
    assert second["operations"]["MainThread"]["store.read_json"] == {"calls": 2, "errors": 1, "total_ns": 300}
    assert first["monotonic"] <= second["monotonic"]


def test_sample_overflow_is_bounded_and_reported():
    measurements = AgentMeasurements()
    measurements.samples = deque(maxlen=2)
    for _ in range(3):
        measurements.capture()
    assert len(measurements.samples) == 2
    assert measurements.samples_dropped == 1


def test_same_expiry_does_not_hide_new_token_or_registration_generation():
    measurements = AgentMeasurements()
    attempt = {"attempt_id": "attempt", "phase": "running", "lease": {"expires_at": "2026-01-01T00:00:00Z"}}
    for token in (1, 1, 2):
        measurements.event("atomic_replace", {"attempt": dict(attempt, current_fencing_token=token)})
    registration = {"project_id": "project", "eligibility_expires_at": "2026-01-01T00:00:00Z", "state": "eligible"}
    for generation in ("first", "first", "second"):
        measurements.event("atomic_replace", {"registration": dict(registration, generation=generation)})
    assert len(measurements.events) == 4
    assert [event["fencing_token"] for event in measurements.events[:2]] == [1, 2]
    assert [event["generation"] for event in measurements.events[2:]] == ["first", "second"]


def test_steady_summary_uses_only_interior_phase_and_actual_elapsed_time():
    def sample(at, calls):
        return {
            "monotonic": at,
            "operations": {"authority": {"read": {"calls": calls, "errors": 0, "total_ns": calls * 100}}},
        }

    report = {
        "steady_started_monotonic": 10,
        "steady_finished_monotonic": 20,
        "agent": {"operation_samples": [sample(9, 100), sample(11, 120), sample(19, 200), sample(21, 1000)]},
    }
    result = steady_operations(report)
    assert result["seconds"] == 8
    assert result["sample_count"] == 2
    assert result["by_thread"]["authority"]["read"] == {
        "calls": 80,
        "errors": 0,
        "total_ns": 8000,
        "calls_per_second": 10,
    }
    report["steady_finished_monotonic"] = 12
    assert steady_operations(report) is None
    report["steady_finished_monotonic"] = 20
    report["agent"]["operation_samples"][2]["operations"]["authority"]["read"]["calls"] = 1
    with pytest.raises(ValueError, match="decreased"):
        steady_operations(report)


def test_renewal_lateness_does_not_cross_token_or_generation_boundaries(tmp_path):
    def event(record, expiry, at, **identity):
        return dict(record=record, expires_at=f"2026-01-01T00:{expiry}Z", at=f"2026-01-01T00:{at}Z", **identity)

    report = {
        "profile": {},
        "outcome": "passed",
        "source_sha": "test",
        "instrumentation_sha256": "test",
        "workload_test_sha256": "test",
        "agent": {
            "events": [
                event("attempt", "02:00", "00:00", attempt_id="a", fencing_token=1),
                event("attempt", "02:01", "00:20", attempt_id="a", fencing_token=2),
                event("attempt", "02:11", "00:21", attempt_id="a", fencing_token=2),
                event("registration", "02:00", "00:00", project_id="p", generation="first"),
                event("registration", "02:01", "00:20", project_id="p", generation="second"),
            ]
        },
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    result = summarize(path)
    assert result["expiry_publication_lateness"]["attempt"] == {"count": 1, "median_seconds": 10, "maximum_seconds": 10}
    assert "registration" not in result["expiry_publication_lateness"]


def test_write_counts_include_equivalent_publications_but_not_failed_immutable_create():
    measurements = AgentMeasurements()
    value = {"registration": {"project_id": "project", "state": "eligible"}}
    create = measurements.wrap(lambda *_args: False, "create_if_absent", is_store=True)
    assert create("path", value) is False
    assert measurements.events == []
    publish = measurements.wrap(lambda *_args: None, "atomic_replace", is_store=True)
    publish("path", value)
    publish("path", value)
    assert len(measurements.events) == 1
    assert measurements.operations["MainThread"]["record.registration.publications"]["calls"] == 2
    assert measurements.operations["MainThread"]["store.create_if_absent"]["calls"] == 1


def test_successful_immutable_create_with_none_return_is_observed():
    measurements = AgentMeasurements()
    value = {"process_registration": {"attempt_id": "a", "fencing_token": 1}}
    create = measurements.wrap(lambda *_args: None, "create_if_absent", is_store=True)
    assert create("path", value) is None
    assert len(measurements.events) == 1
    assert measurements.events[0]["record"] == "process_registration"


def test_write_trace_attributes_fsync_to_path_and_clears_failed_context():
    measurements = AgentMeasurements()
    fsync = measurements.wrap(lambda _fd: None, "os.fsync")

    def write(path, value):
        fsync(1)
        fsync(2)
        if path == "failed":
            raise OSError("write failed")

    publish = measurements.wrap(write, "atomic_replace", is_store=True)
    publish("first", {})
    with pytest.raises(OSError, match="write failed"):
        publish("failed", {})
    fsync(3)
    first, failed = measurements.writes
    assert first["path"] == "first"
    assert first["fsync_calls"] == failed["fsync_calls"] == 2
    assert not first["has_failed"]
    assert failed["has_failed"]
    assert first["total_ns"] >= first["fsync_ns"]
    assert measurements.write_context.current is None


def test_startup_summary_preserves_missing_profiles_and_excludes_later_writes():
    from tests.helpers.qexp.authority_reports import startup_summary

    report = {
        "startup_profiles": {
            "first.requested": "10",
            "first.shell": "10.5",
            "first.json": json.dumps(
                [
                    {"stage": "python_entered", "at": 10.6},
                    {"stage": "import_started", "at": 10.7},
                    {"stage": "import_finished", "at": 10.9},
                ]
            ),
            "missing.requested": "12",
        },
        "agent_requested_monotonic": 100,
        "all_running_seconds": 3,
        "agent": {
            "launch_calls": [{"stage": "tmux_create_window", "started": 10, "finished": 10.4}],
            "write_observations": [
                dict(path="/a/agent.json", finished_monotonic=at, total_ns=100, fsync_calls=2, fsync_ns=50)
                for at in (101, 102, 104)
            ],
        },
    }
    result = startup_summary(report)
    assert result["incomplete_launch_profiles"] == ["missing"]
    assert result["phases"]["command_requested_to_shell_entered"]["median_seconds"] == 0.5
    assert result["phases"]["import_started_to_import_finished"]["median_seconds"] == pytest.approx(0.2)
    assert result["phases"]["tmux_create_window"]["median_seconds"] == pytest.approx(0.4)
    assert result["agent_writes_by_filename"]["agent.json"] == {
        "calls": 2,
        "total_ns": 200,
        "fsync_calls": 4,
        "fsync_ns": 100,
    }


def test_profile_collection_records_invalid_and_missing_diagnostics_without_raising(tmp_path):
    from tests.helpers.qexp.startup_profile import collect_startup_profiles

    expected = dict(expected_source=Path("/checkout/runner.py"), expected_home=Path("/isolated/home"), expected_count=1)
    root = tmp_path / "profiles"
    assert collect_startup_profiles(root, **expected)["startup_profile_status"] == "incomplete"
    root.mkdir()
    path = root / "runner.json"
    for raw in ("truncated {", "{}", "[]", '[{"home":"/outside"}]'):
        path.write_text(raw)
        result = collect_startup_profiles(root, **expected)
        assert result["startup_profile_status"] == "invalid"
        assert not result["startup_environment_is_valid"]
        assert result["startup_profile_errors"]
        assert result["startup_profiles"][path.name] == raw
    path.write_text(json.dumps([{"home": "/isolated/home"}, {"source_file": "/checkout/runner.py"}]))
    assert collect_startup_profiles(root, **expected)["startup_profile_status"] == "valid"


def test_startup_summary_reports_truncated_events_and_unavailable_shell_timestamp():
    from tests.helpers.qexp.authority_reports import startup_summary

    result = startup_summary(
        {
            "startup_profiles": {
                "broken.requested": "10",
                "broken.json": "{",
                "no-shell.requested": "10",
                "no-shell.json": "[]",
            }
        }
    )
    assert "broken" in result["invalid_launch_profiles"]
    assert result["missing_shell_timestamps"] == ["no-shell"]


def test_summary_retains_control_plane_and_checks_generation_coverage(tmp_path):
    report = {
        "profile": {"bindings": 1},
        "outcome": "passed",
        "source_sha": "test",
        "instrumentation_sha256": "test",
        "workload_test_sha256": "test",
        "agent": {"events": [{"record": "registration", "project_id": "p", "generation": "current"}]},
    }
    path = tmp_path / "report.json"

    def result():
        path.write_text(json.dumps(report))
        return summarize(path)

    assert not result()["control_plane_coverage"]["is_available"]
    snapshot = {
        "authority_control_plane": {
            "projects": [{"project_id": "p", "registration_generation": "old", "maximum_service_gap_seconds": 2}],
            "heartbeat": {"operations": {"counters": {"publication_unavailable": 1}}},
            "schedule": {"skipped_intervals_total": 9},
        }
    }
    report["control_plane_snapshot"] = snapshot
    assert not result()["control_plane_coverage"]["has_complete_generation_coverage"]
    snapshot["authority_control_plane"]["projects"][0]["registration_generation"] = "current"
    summarized = result()
    assert summarized["control_plane_coverage"]["has_complete_generation_coverage"]
    assert summarized["control_plane_snapshot"] == snapshot
    snapshot["authority_control_plane"]["projects"].clear()
    assert not result()["control_plane_coverage"]["has_complete_generation_coverage"]
