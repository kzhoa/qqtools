from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from qqtools.plugins.qexp.agent import dispatch_loop
from qqtools.plugins.qexp.agent.bindings import ProjectBinding
from qqtools.plugins.qexp.agent.context import MachineRuntime


def test_machine_snapshot_intent_is_copied_and_newest_wins(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    binding = ProjectBinding(
        "project-a",
        tmp_path / "project-a" / ".qexp",
        "gpu-1",
        registration_generation="generation-1",
        runtime_instance_id="runtime-1",
        runtime_root=str(tmp_path / "machine" / "projects" / "project-a"),
    )
    reservations = [{"reservation_id": "reservation-1", "gpu_ids": [0]}]
    policy = {"mode": "auto", "visible_gpu_ids": [0]}

    runtime.publish_machine_snapshot_intent(
        bindings=[binding],
        registry_revision=1,
        instance_id="agent-1",
        pid=123,
        visible_gpu_ids=[0],
        reservations=reservations,
        heartbeat_interval_seconds=5.0,
        started_at="2026-09-28T00:00:00Z",
        gpu_policy=policy,
    )
    reservations[0]["gpu_ids"].append(9)
    policy["visible_gpu_ids"].append(9)
    runtime.publish_machine_snapshot_intent(
        bindings=[binding],
        registry_revision=2,
        instance_id="agent-2",
        pid=456,
        visible_gpu_ids=[1],
        reservations=[{"reservation_id": "reservation-2", "gpu_ids": [1]}],
        heartbeat_interval_seconds=10.0,
        started_at="2026-09-28T00:00:05Z",
        gpu_policy={"mode": "explicit", "visible_gpu_ids": [1]},
    )

    intent = runtime.machine_snapshot_intent()
    assert intent is not None
    assert intent.registry_revision == 2
    assert intent.instance_id == "agent-2"
    assert intent.visible_gpu_ids == (1,)
    assert intent.reservations == ({"reservation_id": "reservation-2", "gpu_ids": [1]},)
    assert intent.gpu_policy == {"mode": "explicit", "visible_gpu_ids": [1]}
    assert runtime.machine_snapshot_intent() is intent


def test_dispatch_snapshot_handoff_forwards_exact_intent_and_consumes_failures(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    binding = ProjectBinding(
        "project-a",
        tmp_path / "project-a" / ".qexp",
        "gpu-1",
        registration_generation="generation-1",
        runtime_instance_id="runtime-1",
        runtime_root=str(tmp_path / "machine" / "projects" / "project-a"),
    )
    runtime.publish_machine_snapshot_intent(
        bindings=[binding],
        registry_revision=7,
        instance_id="agent-1",
        pid=123,
        visible_gpu_ids=[0],
        reservations=[],
        heartbeat_interval_seconds=5.0,
        started_at="2026-09-28T00:00:00Z",
        gpu_policy={"mode": "auto"},
    )
    calls: list[tuple[object, ...]] = []

    def advance(bindings, revision, **parameters):
        calls.append((bindings, revision, parameters))
        raise OSError("injected controller failure")

    controller = SimpleNamespace(advance_machine_snapshot_publications=advance)
    assert runtime.machine_snapshot_intent(validated_project_ids=[]) is None
    dispatch_loop._advance_machine_snapshot_publications(runtime, controller, {binding.project_id})

    assert len(calls) == 1
    bindings, revision, parameters = calls[0]
    assert bindings == (binding,)
    assert revision == 7
    assert parameters == {
        "instance_id": "agent-1",
        "pid": 123,
        "visible_gpu_ids": (0,),
        "reservations": (),
        "heartbeat_interval_seconds": 5.0,
        "started_at": "2026-09-28T00:00:00Z",
        "gpu_policy": {"mode": "auto"},
    }
    assert runtime.machine_snapshot_intent() is not None

    controller.prune_machine_snapshot_observations = lambda bindings: None
    controller.is_binding_validated = lambda candidate, revision: candidate == binding and revision == 7
    for revision in (8, 7):
        dispatch_loop._advance_machine_snapshot_publications(
            runtime, controller, (), registry_snapshot=(revision, [binding])
        )
    assert len(calls) == 2
    assert calls[1] == ((binding,), 7, {**parameters, "observed_registry_revision": 7})


def test_recovery_keeps_dormant_snapshot_due_until_fresh_coverage(tmp_path, monkeypatch):
    runtime = MachineRuntime(tmp_path / "machine")
    binding = ProjectBinding("project-a", tmp_path / "project" / ".qexp", "gpu-1", registration_generation="gen")
    monkeypatch.setattr(runtime.registration, "load_registry_snapshot", lambda: (7, (binding,)))
    runtime.publish_registration_renewal_intent(
        bindings=(binding,), registry_revision=7, renewal_horizon_seconds=10, heartbeat_interval_seconds=5
    )

    def heartbeat():
        runtime.publish_machine_snapshot_intent(
            bindings=(),
            registry_revision=7,
            instance_id="agent",
            pid=1,
            visible_gpu_ids=[],
            reservations=[],
            heartbeat_interval_seconds=5,
            started_at="2026-09-28T00:00:00Z",
            gpu_policy={},
        )
        return runtime.machine_snapshot_intent().bindings

    now = datetime.now(timezone.utc)
    runtime.request_recovery_snapshots((now - timedelta(minutes=1)).isoformat())
    assert heartbeat() == (binding,)
    runtime.complete_recovery_snapshot(binding.project_id, "gen", (now - timedelta(seconds=30)).isoformat())
    assert heartbeat() == (binding,)
    runtime.complete_recovery_snapshot(binding.project_id, "old", now.isoformat())
    assert heartbeat() == (binding,)
    runtime.complete_recovery_snapshot(binding.project_id, "gen", now.isoformat())
    assert heartbeat() == ()


def test_registration_renewal_intent_is_latest_state_and_dispatch_forwards_it(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    first = ProjectBinding(
        "project-a",
        tmp_path / "project-a" / ".qexp",
        "gpu-1",
        registration_generation="generation-1",
        runtime_instance_id="runtime-1",
        runtime_root=str(tmp_path / "machine" / "projects" / "project-a"),
    )
    second = ProjectBinding(
        "project-b",
        tmp_path / "project-b" / ".qexp",
        "gpu-1",
        registration_generation="generation-2",
        runtime_instance_id="runtime-1",
        runtime_root=str(tmp_path / "machine" / "projects" / "project-b"),
    )
    source = [first]
    runtime.publish_registration_renewal_intent(
        bindings=source,
        registry_revision=1,
        renewal_horizon_seconds=10.0,
        heartbeat_interval_seconds=5.0,
    )
    source.append(second)
    runtime.publish_registration_renewal_intent(
        bindings=[second],
        registry_revision=2,
        renewal_horizon_seconds=20.0,
        heartbeat_interval_seconds=10.0,
    )
    calls = []

    def advance(bindings, revision, *, renewal_horizon_seconds):
        calls.append((bindings, revision, renewal_horizon_seconds))
        raise OSError("injected controller failure")

    intent = runtime.registration_renewal_intent()
    assert intent is not None
    assert intent.bindings == (second,)
    assert intent.registry_revision == 2
    assert intent.renewal_horizon_seconds == 20.0
    assert intent.heartbeat_interval_seconds == 10.0
    dispatch_loop._advance_registration_renewals(runtime, SimpleNamespace(advance_registration_renewals=advance))
    assert calls == [((second,), 2, 20.0)]
    assert runtime.registration_renewal_intent() is intent


def test_registration_renewal_batch_is_retained_until_every_binding_completes(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    bindings = tuple(
        ProjectBinding(
            f"project-{index}",
            tmp_path / f"project-{index}" / ".qexp",
            "gpu-1",
            registration_generation=f"generation-{index}",
            runtime_instance_id="runtime-1",
            runtime_root=str(tmp_path / "machine" / "projects" / f"project-{index}"),
        )
        for index in range(128)
    )
    next_batches = iter(((bindings[64:], 85.0), ((), 85.0)))
    runtime.working_set = SimpleNamespace(
        select_registration_renewals=lambda **_kwargs: next(next_batches),
    )
    runtime.publish_registration_renewal_intent(
        bindings=bindings[:64],
        registry_revision=7,
        renewal_horizon_seconds=85.0,
        heartbeat_interval_seconds=5.0,
    )
    completion_batches = iter(
        (
            {binding.project_id: {} for binding in bindings[:4]},
            {binding.project_id: {} for binding in bindings[4:64]},
            {binding.project_id: {} for binding in bindings[64:]},
        )
    )
    controller_inputs = []

    def advance(current, *_args, **_kwargs):
        controller_inputs.append(tuple(current))
        return next(completion_batches)

    controller = SimpleNamespace(advance_registration_renewals=advance)

    dispatch_loop._advance_registration_renewals(runtime, controller, 7)
    retained = runtime.registration_renewal_intent()
    assert retained is not None
    assert retained.bindings == bindings[4:64]

    dispatch_loop._advance_registration_renewals(runtime, controller, 7)
    retained = runtime.registration_renewal_intent()
    assert retained is not None
    assert retained.bindings == bindings[64:]

    dispatch_loop._advance_registration_renewals(runtime, controller, 7)
    assert runtime.registration_renewal_intent() is None
    assert controller_inputs == [bindings[:64], bindings[4:64], bindings[64:]]


def test_registration_renewal_batch_is_fenced_by_registry_revision(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    runtime.publish_registration_renewal_intent(
        bindings=[],
        registry_revision=7,
        renewal_horizon_seconds=10.0,
        heartbeat_interval_seconds=5.0,
    )
    controller = SimpleNamespace(
        advance_registration_renewals=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("stale batch must not reach the controller")
        )
    )

    dispatch_loop._advance_registration_renewals(runtime, controller, 8)

    assert runtime.registration_renewal_intent() is None


def test_hung_first_batch_member_does_not_block_the_next_registration_batch(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    bindings = tuple(
        ProjectBinding(
            f"project-{index}",
            tmp_path / f"project-{index}" / ".qexp",
            "gpu-1",
            registration_generation=f"generation-{index}",
            runtime_instance_id="runtime-1",
            runtime_root=str(tmp_path / "machine" / "projects" / f"project-{index}"),
        )
        for index in range(128)
    )
    next_batches = iter(((bindings[64:], 85.0), ((), 85.0)))
    runtime.working_set = SimpleNamespace(
        select_registration_renewals=lambda **_kwargs: next(next_batches),
    )
    runtime.publish_registration_renewal_intent(
        bindings=bindings[:64],
        registry_revision=7,
        renewal_horizon_seconds=85.0,
        heartbeat_interval_seconds=5.0,
    )

    class Executor:
        unresolved = ()

        def unresolved_requests(self):
            return self.unresolved

    executor = Executor()
    service_calls = []

    def advance(current, *_args, **_kwargs):
        service_calls.append(tuple(current))
        if len(service_calls) == 1:
            hung = bindings[0]
            executor.unresolved = (
                SimpleNamespace(
                    operation_kind="registration_renew",
                    registry_revision=7,
                    project_id=hung.project_id,
                    registration_generation=hung.registration_generation,
                    parameters={"renewal_horizon_seconds": 85.0},
                ),
            )
            return {binding.project_id: {} for binding in bindings[1:64]}
        return {binding.project_id: {} for binding in bindings[64:]}

    controller = SimpleNamespace(executor=executor, advance_registration_renewals=advance)

    dispatch_loop._advance_registration_renewals(runtime, controller, 7, bindings)
    next_intent = runtime.registration_renewal_intent()
    assert next_intent is not None
    assert next_intent.bindings == bindings[64:]

    dispatch_loop._advance_registration_renewals(runtime, controller, 7, bindings)
    assert runtime.registration_renewal_intent() is None
    dispatch_loop._advance_registration_renewals(runtime, controller, 7, bindings)
    assert service_calls == [bindings[:64], (*bindings[64:], bindings[0]), (bindings[0],)]


def test_other_request_for_binding_does_not_pin_registration_renewal_batch(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    bindings = tuple(
        ProjectBinding(
            f"project-{index}",
            tmp_path / f"project-{index}" / ".qexp",
            "gpu-1",
            registration_generation=f"generation-{index}",
            runtime_instance_id="runtime-1",
            runtime_root=str(tmp_path / "machine" / "projects" / f"project-{index}"),
        )
        for index in range(65)
    )
    runtime.working_set = SimpleNamespace(
        select_registration_renewals=lambda **_kwargs: (bindings[64:], 15.0),
    )
    runtime.publish_registration_renewal_intent(
        bindings=bindings[:64],
        registry_revision=7,
        renewal_horizon_seconds=15.0,
        heartbeat_interval_seconds=5.0,
    )
    occupied = bindings[0]
    claim = SimpleNamespace(
        operation_kind="scheduler_claim",
        registry_revision=7,
        project_id=occupied.project_id,
        registration_generation=occupied.registration_generation,
    )

    class Executor:
        def unresolved_requests(self):
            return (claim,)

    controller = SimpleNamespace(
        executor=Executor(),
        advance_registration_renewals=lambda *_args, **_kwargs: {binding.project_id: {} for binding in bindings[1:64]},
    )

    dispatch_loop._advance_registration_renewals(runtime, controller, 7, bindings)

    next_intent = runtime.registration_renewal_intent()
    assert next_intent is not None
    assert next_intent.bindings == bindings[64:]


def test_returned_renewal_error_does_not_pin_the_next_registration_batch(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    bindings = tuple(
        ProjectBinding(
            f"project-{index}",
            tmp_path / f"project-{index}" / ".qexp",
            "gpu-1",
            registration_generation=f"generation-{index}",
            runtime_instance_id="runtime-1",
            runtime_root=str(tmp_path / "machine" / "projects" / f"project-{index}"),
        )
        for index in range(128)
    )
    runtime.working_set = SimpleNamespace(
        select_registration_renewals=lambda **_kwargs: (bindings[64:], 85.0),
    )
    runtime.publish_registration_renewal_intent(
        bindings=bindings[:64],
        registry_revision=7,
        renewal_horizon_seconds=85.0,
        heartbeat_interval_seconds=5.0,
    )
    failed = bindings[0]
    failed_request = SimpleNamespace(
        operation_kind="registration_renew",
        registry_revision=7,
        project_id=failed.project_id,
        registration_generation=failed.registration_generation,
        parameters={"renewal_horizon_seconds": 85.0},
    )

    class Executor:
        unresolved = (failed_request,)

        def unresolved_requests(self):
            return self.unresolved

    executor = Executor()

    def advance(_current, *_args, **_kwargs):
        executor.unresolved = ()
        return {binding.project_id: {} for binding in bindings[1:64]}

    controller = SimpleNamespace(executor=executor, advance_registration_renewals=advance)

    dispatch_loop._advance_registration_renewals(runtime, controller, 7, bindings)

    next_intent = runtime.registration_renewal_intent()
    assert next_intent is not None
    assert next_intent.bindings == bindings[64:]


def test_unfinished_project_io_uses_a_short_controller_poll_interval(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    runtime.project_io_executor = SimpleNamespace(
        status_view=lambda: {"active_worker_count": 1, "overdue_worker_count": 0},
        has_ready_result=lambda: False,
    )

    assert runtime.pending_launch_wait_seconds(5.0) == 0.05

    runtime.project_io_executor = SimpleNamespace(
        status_view=lambda: {"active_worker_count": 1, "overdue_worker_count": 1},
        has_ready_result=lambda: False,
    )
    assert runtime.pending_launch_wait_seconds(5.0) == 5.0

    runtime.project_io_executor = SimpleNamespace(
        status_view=lambda: {"active_worker_count": 0, "overdue_worker_count": 0},
        has_ready_result=lambda: True,
    )
    assert runtime.pending_launch_wait_seconds(5.0) == 0.05


def test_renewal_prepare_backoff_retires_each_batch_until_deadline(tmp_path, monkeypatch):
    from collections import Counter

    from qqtools.plugins.qexp.agent.control_plane import _MachineControlPlane
    from qqtools.plugins.qexp.agent.control_plane_health import read_control_plane_health
    from qqtools.plugins.qexp.agent.project_io_controller import ProjectIOController

    runtime = MachineRuntime(tmp_path / "machine")
    bindings = tuple(
        ProjectBinding(
            f"project-{i}",
            tmp_path / str(i) / ".qexp",
            "gpu",
            registration_generation=f"generation-{i}",
            runtime_instance_id=runtime.instance_id,
            runtime_root=str(runtime.root),
        )
        for i in range(65)
    )
    runtime.working_set.reconcile(bindings, revision=1)
    monkeypatch.setattr(runtime, "load_registry_snapshot", lambda: (1, bindings))
    selected, horizon = runtime.working_set.select_registration_renewals()
    failed = selected[0]
    prepares = []

    def prepare(binding, *args, **kwargs):
        prepares.append(binding)
        raise OSError("test failure\n" + "x" * 500)

    executor = SimpleNamespace(
        runtime_id=runtime.instance_id,
        unresolved_requests=lambda: (),
        admission_snapshot=lambda **kwargs: (
            {
                "executor_epoch": "epoch",
                "envelope": "healthy",
                "capacity": 4,
                "overdue_worker_count": 0,
                "supported_hang_limit": 2,
            },
            (),
        ),
        prepare_registration_renew=prepare,
    )
    now = [100.0]
    controller = ProjectIOController(
        runtime, executor, monotonic=lambda: now[0], utc_clock=lambda: "2026-11-26T00:00:00Z"
    )
    for binding in bindings:
        if binding != failed:
            controller._registration_due[controller._binding_identity(binding, 1, "epoch")] = 1000.0
    runtime.publish_registration_renewal_intent(
        bindings=selected, registry_revision=1, renewal_horizon_seconds=horizon, heartbeat_interval_seconds=5.0
    )
    with controller.admission_turn():
        dispatch_loop._advance_registration_renewals(runtime, controller, 1, bindings)
        assert runtime.registration_renewal_intent().bindings == (failed,)
        assert prepares == []
    assert prepares == [failed]
    retry_at = next(iter(controller._service_backoff.values())).retry_at
    turns = Counter()
    for _ in range(131):
        batch = runtime.registration_renewal_intent().bindings
        turns.update(binding.project_id for binding in batch)
        dispatch_loop._advance_registration_renewals(runtime, controller, 1, bindings)
        assert len(runtime.registration_renewal_intent().bindings) == 64
        assert next(iter(controller._service_backoff.values())).retry_at == retry_at
        assert prepares == [failed]
    assert all(turns[binding.project_id] >= 2 for binding in bindings)
    item = controller.registration_renewal_health_view()["items"][0]
    assert (item["project_id"], item["registration_generation"]) == (failed.project_id, failed.registration_generation)
    assert item["blocking_reason"] == item["last_outcome"] == "request_prepare_failed"
    assert item["cause"] == "OSError"
    assert item["next_retry_at"] == "2026-11-26T00:00:05Z"
    runtime.project_io_controller = controller
    plane = _MachineControlPlane(
        runtime, instance_id="agent-instance", loop_interval=5, started_at="2026-11-26T00:00:00Z", available_gpus=[]
    )
    live = plane.monitor_health()
    persisted = read_control_plane_health(runtime.root, instance_id="agent-instance", thread_alive=None)
    for view in (live, persisted):
        assert view["project_registration_renewals"]["items"][0] == item
        assert "snapshot_batch_error" not in view
    now[0] = retry_at
    for _ in range(2):
        dispatch_loop._advance_registration_renewals(runtime, controller, 1, bindings)
    assert prepares == [failed, failed]
