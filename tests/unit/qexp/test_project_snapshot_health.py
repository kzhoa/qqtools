from __future__ import annotations

from datetime import datetime, timedelta, timezone

from qqtools.plugins.qexp.agent.project_io.identity import ValidationIdentity
from qqtools.plugins.qexp.agent.snapshot_publication_health import SnapshotPublicationHealth


class _Clock:
    def __init__(self) -> None:
        self.monotonic_value = 100.0
        self.utc_value = datetime(2026, 10, 16, tzinfo=timezone.utc)

    def monotonic(self) -> float:
        return self.monotonic_value

    def utc_now(self) -> str:
        return self.utc_value.isoformat().replace("+00:00", "Z")

    def advance(self, seconds: float) -> None:
        self.monotonic_value += seconds
        self.utc_value += timedelta(seconds=seconds)


def _identity(project_id: str = "project-1") -> ValidationIdentity:
    return ValidationIdentity(
        runtime_id="runtime",
        executor_epoch="epoch",
        project_id=project_id,
        shared_root=f"/shared/{project_id}",
        machine_name="gpu-1",
        registration_generation=f"registration-{project_id}",
        runtime_root="/runtime",
        registry_revision=7,
    )


def test_snapshot_progress_stages_use_deterministic_clocks_and_recover() -> None:
    clock = _Clock()
    tracker = SnapshotPublicationHealth(monotonic=clock.monotonic, utc_clock=clock.utc_now, max_retry_seconds=60)
    tracker.reset_instance("agent-instance")
    identity = _identity()
    tracker.begin_intent(identity, "agent-instance", 5.0, new_attempt=True)
    intent = tracker.view()["items"][0]
    assert intent["stage"] == "intent"
    assert intent["last_attempt_at"] == "2026-10-16T00:00:00Z"
    assert intent["consecutive_failures"] == 0

    # An admission turn may defer this intent without making it a failure.
    tracker.begin_intent(identity, "agent-instance", 5.0)
    deferred = tracker.view()["items"][0]
    assert deferred["stage"] == "intent"
    assert deferred["consecutive_failures"] == 0

    tracker.report_request(identity, "agent-instance", 5.0, "request-1", stage="queued")
    assert tracker.view()["items"][0]["stage"] == "queued"
    tracker.report_request(identity, "agent-instance", 5.0, "request-1", stage="started")
    clock.advance(3.0)
    started = tracker.view()["items"][0]
    assert started["stage"] == "started"
    assert started["worker_elapsed_seconds"] == 3.0

    tracker.report_request(identity, "agent-instance", 5.0, "request-1", stage="result_ready")
    assert tracker.view()["items"][0]["stage"] == "result_ready"
    tracker.report_request(identity, "agent-instance", 5.0, "request-1", stage="consumed")
    assert tracker.view()["items"][0]["stage"] == "consumed"

    tracker.report_failure(
        identity,
        "agent-instance",
        5.0,
        reason="publication_failed",
        request_id="request-1",
        retry_deadline=clock.monotonic() + 5,
    )
    failed = tracker.view()["items"][0]
    assert failed["stage"] == "backoff"
    assert failed["consecutive_failures"] == 1
    assert failed["retry_at"] == "2026-10-16T00:00:08Z"

    tracker.report_request(identity, "agent-instance", 5.0, "request-2", stage="consumed")
    tracker.report_request(identity, "agent-instance", 5.0, "request-2", stage="published")
    published = tracker.view()["items"][0]
    assert published["stage"] == "published"
    assert published["last_published_at"] == "2026-10-16T00:00:03Z"
    assert published["consecutive_failures"] == 0
    assert published["reason_code"] is None
    assert published["retry_at"] is None
    clock.advance(15.0)
    assert tracker.view()["items"][0]["overdue"] is True


def test_snapshot_health_view_is_bounded_and_worst_first() -> None:
    clock = _Clock()
    tracker = SnapshotPublicationHealth(monotonic=clock.monotonic, utc_clock=clock.utc_now, max_retry_seconds=60)
    tracker.reset_instance("agent-instance")
    for index in range(10):
        identity = _identity(f"project-{index:02d}")
        tracker.begin_intent(identity, "agent-instance", 5.0, new_attempt=True)
        if index < 2:
            tracker.report_failure(
                identity,
                "agent-instance",
                5.0,
                reason="worker_start_failed",
            )
    clock.advance(15.0)

    view = tracker.view(limit=8)
    assert view["total"] == 10
    assert view["omitted_count"] == 2
    assert len(view["items"]) == 8
    assert [item["project_id"] for item in view["items"][:2]] == ["project-00", "project-01"]
    assert all(item["overdue"] is True for item in view["items"])

    view["items"][0]["stage"] = "changed"
    assert tracker.view()["items"][0]["stage"] == "backoff"
    tracker.prune({_identity("project-09")})
    assert [item["project_id"] for item in tracker.view()["items"]] == ["project-09"]
    tracker.reset_instance("replacement")
    assert tracker.view() == {"items": [], "total": 0, "omitted_count": 0}


def test_rapid_retries_remain_overdue_and_old_completion_does_not_reset_freshness() -> None:
    clock = _Clock()
    tracker = SnapshotPublicationHealth(monotonic=clock.monotonic, utc_clock=clock.utc_now, max_retry_seconds=60)
    tracker.reset_instance("agent-instance")
    identity = _identity()
    for attempt in range(16):
        clock.advance(1.0)
        tracker.begin_intent(identity, "agent-instance", 5.0, new_attempt=True)
        tracker.report_failure(identity, "agent-instance", 5.0, reason="publication_failed", request_id=f"r{attempt}")
    overdue = tracker.view()["items"][0]
    assert overdue["overdue"] is True
    assert overdue["request_id"] == "r15"

    tracker.report_request(identity, "agent-instance", 5.0, "r16", stage="consumed")
    tracker.report_request(identity, "agent-instance", 5.0, "r15", stage="published")
    assert tracker.view()["items"][0]["overdue"] is True
    tracker.report_request(identity, "agent-instance", 5.0, "r16", stage="published")
    success = tracker.view()["items"][0]
    assert success["overdue"] is False
    assert success["publication_sequence"] == 1
