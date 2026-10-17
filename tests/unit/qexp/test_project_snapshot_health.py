from __future__ import annotations

from datetime import datetime, timedelta, timezone

from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.project_io_controller import ProjectIOController, _FailureBackoff, _ValidationIdentity
from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor
from qqtools.plugins.qexp.agent.setup import initialize_machine


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


def _controller(tmp_path):
    runtime = MachineRuntime(tmp_path / "runtime")
    initialize_machine(runtime, "gpu-1")
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    clock = _Clock()
    controller = ProjectIOController(runtime, executor, monotonic=clock.monotonic, utc_clock=clock.utc_now)
    controller._snapshot_health_instance_id = "agent-instance"
    return controller, executor, clock


def _identity(controller: ProjectIOController, project_id: str = "project-1") -> _ValidationIdentity:
    return _ValidationIdentity(
        runtime_id=controller.runtime.instance_id,
        executor_epoch=controller.executor.status_view()["executor_epoch"],
        project_id=project_id,
        shared_root=f"/shared/{project_id}",
        machine_name="gpu-1",
        registration_generation=f"registration-{project_id}",
        runtime_root=str(controller.runtime.root),
        registry_revision=7,
    )


def test_snapshot_progress_stages_use_deterministic_clocks_and_recover(tmp_path) -> None:
    controller, executor, clock = _controller(tmp_path)
    identity = _identity(controller)
    service_key = controller._snapshot_service_key(identity)
    try:
        controller._snapshot_begin_intent(identity, "agent-instance", 5.0, new_attempt=True)
        intent = controller.machine_snapshot_health_view()["items"][0]
        assert intent["stage"] == "intent"
        assert intent["last_attempt_at"] == "2026-10-16T00:00:00Z"
        assert intent["consecutive_failures"] == 0

        # An admission turn may defer this intent without making it a failure.
        controller._snapshot_begin_intent(identity, "agent-instance", 5.0)
        deferred = controller.machine_snapshot_health_view()["items"][0]
        assert deferred["stage"] == "intent"
        assert deferred["consecutive_failures"] == 0

        controller._snapshot_note_request(identity, "agent-instance", 5.0, "request-1", stage="queued")
        assert controller.machine_snapshot_health_view()["items"][0]["stage"] == "queued"
        controller._snapshot_note_worker_started(identity, "agent-instance", 5.0, "request-1")
        clock.advance(3.0)
        started = controller.machine_snapshot_health_view()["items"][0]
        assert started["stage"] == "started"
        assert started["worker_elapsed_seconds"] == 3.0

        controller._snapshot_mark_result_ready(identity, "agent-instance", 5.0, "request-1")
        assert controller.machine_snapshot_health_view()["items"][0]["stage"] == "result_ready"
        controller._snapshot_mark_consumed(identity, "agent-instance", 5.0, "request-1")
        assert controller.machine_snapshot_health_view()["items"][0]["stage"] == "consumed"

        controller._service_backoff[service_key] = _FailureBackoff(
            consecutive_failures=1, retry_at=clock.monotonic() + 5
        )
        controller._snapshot_record_failure(
            identity,
            "agent-instance",
            5.0,
            reason="publication_failed",
            request_id="request-1",
            service_key=service_key,
        )
        failed = controller.machine_snapshot_health_view()["items"][0]
        assert failed["stage"] == "backoff"
        assert failed["consecutive_failures"] == 1
        assert failed["retry_at"] == "2026-10-16T00:00:08Z"

        controller._snapshot_mark_published(identity, "agent-instance", 5.0, "request-1")
        published = controller.machine_snapshot_health_view()["items"][0]
        assert published["stage"] == "published"
        assert published["last_published_at"] == "2026-10-16T00:00:03Z"
        assert published["consecutive_failures"] == 0
        assert published["reason_code"] is None
        clock.advance(15.0)
        assert controller.machine_snapshot_health_view()["items"][0]["overdue"] is True
    finally:
        executor.shutdown()


def test_snapshot_health_view_is_bounded_and_worst_first(tmp_path) -> None:
    controller, executor, clock = _controller(tmp_path)
    try:
        for index in range(10):
            identity = _identity(controller, f"project-{index:02d}")
            controller._snapshot_begin_intent(identity, "agent-instance", 5.0, new_attempt=True)
            if index < 2:
                controller._snapshot_record_failure(
                    identity,
                    "agent-instance",
                    5.0,
                    reason="worker_start_failed",
                )
        clock.advance(15.0)

        view = controller.machine_snapshot_health_view(limit=8)
        assert view["total"] == 10
        assert view["omitted_count"] == 2
        assert len(view["items"]) == 8
        assert [item["project_id"] for item in view["items"][:2]] == ["project-00", "project-01"]
        assert all(item["overdue"] is True for item in view["items"])
    finally:
        executor.shutdown()
