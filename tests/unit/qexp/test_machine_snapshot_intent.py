from __future__ import annotations

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
