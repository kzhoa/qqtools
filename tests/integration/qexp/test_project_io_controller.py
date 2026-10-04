from __future__ import annotations

import builtins
import hashlib
import io
import os
import threading
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp import project_maintenance as project_maintenance_module
from qqtools.plugins.qexp.agent import dispatch_loop, project_io_supervision
from qqtools.plugins.qexp.agent import project_io_admission as project_io_admission_module
from qqtools.plugins.qexp.agent import project_io_controller as project_io_controller_module
from qqtools.plugins.qexp.agent import project_io_worker as project_io_worker_module
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.dispatch_loop import dispatch_machine_cycle_locked
from qqtools.plugins.qexp.agent.helpers import _machine_is_true_idle
from qqtools.plugins.qexp.agent.project_io_controller import ProjectIOController
from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor
from qqtools.plugins.qexp.agent.project_io_protocol import ProjectIOProcess, ProjectIOResult
from qqtools.plugins.qexp.agent.project_io_supervision import AttemptSupervisionCoordinator
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.agent.working_set import SERVICE_LANES
from qqtools.plugins.qexp.commands.group import change_worker, create_group
from qqtools.plugins.qexp.commands.task import share
from qqtools.plugins.qexp.executor import LaunchHandle, LaunchHandoff
from qqtools.plugins.qexp.layout import load_machine_registration, machine_state_path, save_machine_registration
from qqtools.plugins.qexp.runtime import maintenance as maintenance_module
from qqtools.plugins.qexp.runtime import maintenance_outbox as maintenance_outbox_module
from qqtools.plugins.qexp.runtime import project_activation as project_activation_module
from qqtools.plugins.qexp.runtime.availability import offer_deadlines as offer_deadline_module
from qqtools.plugins.qexp.runtime.availability import sync_deadline_index
from qqtools.plugins.qexp.runtime.availability import transitions as availability_transitions_module
from qqtools.plugins.qexp.runtime.paths import attempt_path, local_paths, ready_state_path, shared_paths, task_path
from qqtools.plugins.qexp.runtime.process_evidence import ProcessEvidence
from qqtools.plugins.qexp.runtime.project_activation import publish_project_activation
from qqtools.plugins.qexp.runtime.project_activation_consumers import read_consumer_progress, register_consumer
from qqtools.plugins.qexp.runtime.ready import read_ready_index_state, repair_ready_index
from qqtools.plugins.qexp.runtime.ready import state as ready_state_module
from qqtools.plugins.qexp.runtime.ready.traversal import load_ready_cursor
from qqtools.plugins.qexp.runtime.records import AttemptRecord
from qqtools.plugins.qexp.runtime.resources.cpu_lane import (
    attach_cpu,
    cpu_reservation_snapshot,
    reserve_cpu,
    set_cpu_lane_capacity,
)
from qqtools.plugins.qexp.runtime.resources.reservations import (
    ReservationIdentity,
    active_reservations,
    attach,
    attach_executor_offer,
    classify_executor_offer,
    reservation_snapshot,
    reserve,
)
from qqtools.plugins.qexp.runtime.store import atomic_replace, check_mutation_fence, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task, save_task
from qqtools.plugins.qexp.scheduler import claim_task, renew_project_io_attempt_lease
from tests.helpers.qexp.worker_diagnostics import describe_project_io_workers

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


@pytest.mark.parametrize("service", ["upgrade", "descriptor"])
def test_producer_offers_healthy_binding_even_when_first_binding_is_occupied(tmp_path, service):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    first_cfg, first, _revision, _bindings = _registered(tmp_path, "first", runtime)
    second_cfg, second, revision, bindings = _registered(tmp_path, "second", runtime)
    ordered = sorted(((first, first_cfg), (second, second_cfg)), key=lambda pair: pair[0].project_id)
    (blocked, blocked_cfg), (healthy, _healthy_cfg) = ordered
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    runtime.working_set.reconcile(bindings, revision=revision)
    try:
        for binding in bindings:
            _advance_until_validated(controller, bindings, revision, binding.project_id)
        blocked_state = ready_state_path(blocked_cfg.shared_root)
        blocked_state.unlink()
        os.mkfifo(blocked_state)
        blocked_request = executor.prepare_scheduler_observe(
            blocked,
            revision,
            lane="gpu",
            admission_role="primary",
            cursor_namespace=f"scheduler-{blocked.project_id}-primary-gpu",
        )
        assert executor.start(blocked_request.request_id) is not None
        advance = (
            controller.advance_upgrade_work if service == "upgrade" else controller.advance_maintenance_descriptor_work
        )
        operation = "upgrade_service" if service == "upgrade" else "maintenance_descriptor_advance"
        advance(bindings, revision)
        unresolved = executor.unresolved_requests()
        assert any(
            request.project_id == healthy.project_id and request.operation_kind == operation for request in unresolved
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            result = advance(bindings, revision)
            if service == "upgrade":
                if healthy.project_id not in runtime.upgrade_pending_projects:
                    break
            elif healthy.project_id in result:
                assert result[healthy.project_id]["maintenance_state"] == "idle"
                break
            time.sleep(0.02)
        else:
            raise AssertionError(
                f"eligible peer did not finish its {service} slice: {describe_project_io_workers(executor)}"
            )
        assert any(request.request_id == blocked_request.request_id for request in executor.unresolved_requests())
        assert executor.status_view()["active_worker_count"] <= 2
    finally:
        executor.shutdown()


def test_submission_control_service_repairs_healthy_peer_without_inline_domain_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qqtools.plugins.qexp.runtime import submission_control as control
    from qqtools.plugins.qexp.runtime import submission_control_maintenance as maintenance
    from qqtools.plugins.qexp.runtime.locks import exclusive
    from qqtools.plugins.qexp.runtime.paths import submission_path

    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg_blocked, blocked, _revision, _bindings = _registered(tmp_path, "blocked", runtime)
    cfg_healthy, healthy, revision, bindings = _registered(tmp_path, "healthy", runtime)
    task = submit(cfg_healthy, ["true"], working_dir=tmp_path)
    source = submission_path(cfg_healthy.shared_root, task.submission_operation_id)
    value = read_json(source)
    value["submission"]["resolved_context"]["retained_payload"] = "x" * 180_000
    atomic_replace(source, value)
    control.request_repair(cfg_healthy, task.submission_operation_id)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    runtime.working_set.reconcile(bindings, revision=revision)
    try:
        _advance_until_validated(controller, bindings, revision, blocked.project_id)
        _advance_until_validated(controller, bindings, revision, healthy.project_id)
        monkeypatch.setattr(
            maintenance.SubmissionControlMaintenance,
            "advance",
            lambda self: pytest.fail("inline shared Submission repair"),
        )
        with exclusive(control.control_paths(cfg_blocked)["locks"] / "maintenance.lock") as acquired:
            assert acquired
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                controller.advance_submission_control(bindings, revision)
                if not control.pending_path(cfg_healthy, task.submission_operation_id).exists():
                    break
                time.sleep(0.02)
            else:
                raise AssertionError("blocked Submission maintenance withheld its healthy peer")
            assert control.read_submission_state(cfg_healthy, task.submission_operation_id) == "committed"
            assert executor.status_view()["active_worker_count"] <= 4
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            controller.advance_submission_control(bindings, revision)
            if len(controller._submission_due) == 2:
                break
            time.sleep(0.02)
        else:
            raise AssertionError("Submission lanes did not settle after release")
    finally:
        executor.shutdown()


def test_upgrade_discovery_lock_does_not_withhold_healthy_peer(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg_blocked, blocked, _revision, _bindings = _registered(tmp_path, "blocked", runtime)
    _cfg_healthy, healthy, revision, bindings = _registered(tmp_path, "healthy", runtime)
    manifest = shared_paths(cfg_blocked.shared_root)["upgrade"] / "protocol-manifest.json"
    manifest.unlink()
    schema_path = shared_paths(cfg_blocked.shared_root)["schema"] / "version.json"
    schema = read_json(schema_path)
    schema["schema"].pop("protocol", None)
    atomic_replace(schema_path, schema)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    from qqtools.plugins.qexp.runtime.locks import exclusive

    try:
        with exclusive(shared_paths(cfg_blocked.shared_root)["locks"] / "upgrade.lock") as acquired:
            assert acquired
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                controller.advance_upgrade_work(bindings, revision)
                requests = executor.unresolved_requests()
                if (
                    runtime.upgrade_discovery_complete is False
                    and healthy.project_id not in runtime.upgrade_admission_blocked_projects
                    and any(request.project_id == blocked.project_id for request in requests)
                ):
                    break
                time.sleep(0.02)
            else:
                raise AssertionError("healthy upgrade summary was withheld by another Project's lock")
            assert blocked.project_id in runtime.upgrade_admission_blocked_projects
            assert blocked.project_id in runtime.upgrade_idle_blocked_projects
            assert executor.status_view()["active_worker_count"] <= 4
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            controller.advance_upgrade_work(bindings, revision)
            if not runtime.upgrade_pending_projects:
                break
            time.sleep(0.02)
        else:
            raise AssertionError("released upgrade journal did not converge")
        assert runtime.upgrade_discovery_complete
    finally:
        executor.shutdown()


def _registered(tmp_path: Path, name: str, runtime: MachineRuntime):
    cfg = init_shared_root(tmp_path / name / ".qexp", "gpu-1")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    revision, bindings = runtime.load_registry()
    return cfg, binding, revision, bindings


def _advance_until_validated(
    controller: ProjectIOController,
    bindings,
    revision: int,
    project_id: str,
    *,
    timeout: float = 5.0,
):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        configs = controller.advance_binding_validation(bindings, revision)
        if project_id in configs:
            return configs
        time.sleep(0.02)
    raise AssertionError(f"binding {project_id!r} did not validate: {describe_project_io_workers(controller.executor)}")


def test_group_service_controller_closes_exact_working_set_turn(tmp_path):
    from qqtools.plugins.qexp.runtime.group_discovery.coverage import GroupCoverage
    from qqtools.plugins.qexp.runtime.paths import submission_path
    from tests.helpers.qexp_discovery import isolated_group, source_file

    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg = isolated_group(tmp_path / "project", tail=1)
    source_file(submission_path(cfg.shared_root, "batch"), operation="batch")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    revision, bindings = runtime.load_registry()
    runtime.working_set.reconcile(bindings, revision=revision)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    try:
        _advance_until_validated(controller, bindings, revision, binding.project_id)
        _observe_working_set_activation(runtime, controller, bindings, revision)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            controller.advance_group_service_probes(bindings, revision)
            if runtime.working_set.is_lane_quiescent(binding, "group"):
                break
            time.sleep(0.01)
        else:
            raise AssertionError(f"Group lane did not close: {describe_project_io_workers(executor)}")
        assert controller.group_service_is_quiescent(binding, revision)
        assert GroupCoverage(cfg.shared_root, "experiment").status().is_complete
    finally:
        executor.shutdown()


def _observe_working_set_activation(runtime, controller, bindings, revision, *, project_ids=None):
    runtime.working_set.reconcile(bindings, revision=revision)
    selected = set(project_ids) if project_ids is not None else {binding.project_id for binding in bindings}
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        dispatch_loop._advance_activation_working_set(runtime, controller, revision, bindings)
        states = [
            runtime.working_set._states[runtime.working_set._identity(binding)]
            for binding in bindings
            if binding.project_id in selected
        ]
        pending = [
            request
            for request in controller.executor.unresolved_requests()
            if request.project_id in selected and request.operation_kind.startswith("activation_")
        ]
        if (
            states
            and not pending
            and all(
                state.consumer_registered
                and state.activation_observed
                and not state.observation_due
                and not state.unknown
                for state in states
            )
        ):
            return
        time.sleep(0.02)
    raise AssertionError(f"activation checkpoint was not observed: {describe_project_io_workers(controller.executor)}")


def _prime_activation_consumers(runtime, bindings, revision):
    """Establish local consumer state for tests whose subject starts at observation."""
    runtime.working_set.reconcile(bindings, revision=revision)
    runtime.working_set.apply_activation_registrations(
        bindings,
        {binding.project_id: {"outcome": "registered", "acknowledgement": None} for binding in bindings},
    )


def _prime_current_activation(runtime, bindings, revision):
    """Establish current local activation for tests of later service stages."""
    _prime_activation_consumers(runtime, bindings, revision)
    while intents := runtime.working_set.select_activation_observations(limit=64):
        runtime.working_set.apply_activation_observations(
            intents,
            {intent.binding.project_id: {"outcome": "observed", "checkpoint": None} for intent in intents},
        )


def _advance_until_observed(
    controller: ProjectIOController,
    bindings,
    revision: int,
    project_id: str,
    *,
    lane: str,
    admission_role: str = "primary",
    timeout: float = 5.0,
):
    _observe_working_set_activation(
        controller.runtime,
        controller,
        bindings,
        revision,
        project_ids={project_id},
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        observations = controller.advance_scheduler_observations(
            bindings,
            revision,
            lane=lane,
            admission_role=admission_role,
        )
        if project_id in observations:
            return observations
        time.sleep(0.02)
    raise AssertionError(
        f"binding {project_id!r} did not produce a scheduler observation: {describe_project_io_workers(controller.executor)}"
    )


def _claim_offer_identity(request) -> ReservationIdentity:
    offer = request.parameters["offer"]
    return ReservationIdentity.from_record(
        {
            "reservation_id": offer["reservation_id"],
            "acquisition_id": offer["acquisition_id"],
            "project_id": offer["project_id"],
            "shared_root": offer["shared_root"],
            "task_id": offer["task_id"],
            "attempt_id": offer["attempt_id"],
            "fencing_token": offer["fencing_token"],
            "gpu_ids": list(offer["gpu_ids"]) if offer["lane"] == "gpu" else None,
            "cpu_slots": offer["cpu_slots"] if offer["lane"] == "cpu" else None,
            "executor_owner": {
                "executor_epoch": request.executor_epoch,
                "request_id": request.request_id,
                "registration_generation": request.registration_generation,
            },
        }
    )


def test_project_io_owner_constructs_one_attempt_supervision_coordinator(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    runtime.project_io_executor = executor
    try:
        controller = dispatch_loop._project_io_controller(runtime)

        assert isinstance(controller, ProjectIOController)
        coordinator = runtime.attempt_supervision_coordinator
        assert isinstance(coordinator, AttemptSupervisionCoordinator)
        assert coordinator.runtime is runtime
        assert coordinator.controller is controller
        assert dispatch_loop._project_io_controller(runtime) is controller
        assert runtime.attempt_supervision_coordinator is coordinator
    finally:
        executor.shutdown()


def test_isolated_dispatch_advances_complete_attempt_supervision_and_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, bindings = _registered(tmp_path, "project", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    runtime.project_io_executor = executor
    controller = dispatch_loop._project_io_controller(runtime)
    assert controller is not None
    coordinator = runtime.attempt_supervision_coordinator
    calls: list[str] = []

    for name in (
        "advance_running_publications",
        "advance_terminal_completions",
        "advance_terminations",
        "advance_orphan_recoveries",
    ):
        original = getattr(coordinator, name)

        def record(*args, __name=name, __original=original, **kwargs):
            calls.append(__name)
            return __original(*args, **kwargs)

        monkeypatch.setattr(coordinator, name, record)

    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            dispatch_machine_cycle_locked(
                runtime,
                available_gpus=[],
                supervise=False,
                publish_snapshots=False,
            )
            if runtime.authority_ready_generations.get(binding.project_id) == binding.registration_generation:
                break
            time.sleep(0.02)
        else:
            raise AssertionError("isolated attempt supervision did not complete initial reconciliation")

        assert set(calls) == {
            "advance_running_publications",
            "advance_terminal_completions",
            "advance_terminations",
            "advance_orphan_recoveries",
        }
        assert runtime.load_registry()[0] == revision
    finally:
        coordinator.close()
        executor.shutdown()


def test_initial_reconciliation_fails_closed_on_malformed_local_evidence(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, _revision, _bindings = _registered(tmp_path, "project", runtime)
    paths = runtime.project_paths(binding.project_id)
    atomic_replace(paths["registrations"] / "malformed-attempt.json", {"unexpected": {}})
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    runtime.project_io_executor = executor
    try:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            dispatch_machine_cycle_locked(
                runtime,
                available_gpus=[],
                supervise=False,
                publish_snapshots=False,
            )
            time.sleep(0.02)

        assert binding.project_id not in runtime.authority_ready_generations
    finally:
        runtime.attempt_supervision_coordinator.close()
        executor.shutdown()


@pytest.mark.parametrize("lane", ["observation", "termination_decision"])
def test_initial_reconciliation_fails_closed_on_malformed_terminal_evidence(
    tmp_path: Path,
    lane: str,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, _revision, _bindings = _registered(tmp_path, "project", runtime)
    paths = runtime.project_paths(binding.project_id)
    if lane == "observation":
        atomic_replace(paths["observations"] / "malformed-attempt.json", {"exit_observation": {}})
    else:
        (paths["termination_decisions"] / "malformed-attempt").mkdir(parents=True)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    runtime.project_io_executor = executor
    try:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            dispatch_machine_cycle_locked(
                runtime,
                available_gpus=[],
                supervise=False,
                publish_snapshots=False,
            )
            time.sleep(0.02)

        assert binding.project_id not in runtime.authority_ready_generations
    finally:
        runtime.attempt_supervision_coordinator.close()
        executor.shutdown()


def test_initial_reconciliation_is_fenced_by_registry_revision(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    revision, _bindings = runtime.load_registry()
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    try:
        coordinator._advance_initial_reconciliation([binding], revision)
        coordinator._advance_initial_reconciliation([binding], revision)
        assert coordinator.initially_reconciled(binding, revision)
        runtime.authority_ready_generations = {
            binding.project_id: binding.registration_generation,
        }

        assert not coordinator.initially_reconciled(binding, revision + 1)
        coordinator._advance_initial_reconciliation([binding], revision + 1)
        assert not coordinator.initially_reconciled(binding, revision + 1)
    finally:
        coordinator.close()
        executor.shutdown()


def test_initial_reconciliation_stays_ready_after_multi_page_sweep(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    revision, _bindings = runtime.load_registry()
    processes = runtime.project_paths(binding.project_id)["processes"]
    for index in range(65):
        task_id = f"task-{index}"
        attempt_id = f"{task_id}-attempt-1"
        atomic_replace(
            processes / f"{attempt_id}.json",
            {
                "process": {
                    "protocol_version": 1,
                    "machine_name": binding.machine_name,
                    "task_id": task_id,
                    "attempt_id": attempt_id,
                    "fencing_token": 1,
                    "reservation_id": None,
                    "wrapper_pid": None,
                    "wrapper_start_time_ticks": None,
                    "process_group_id": None,
                    "process_group_start_time_ticks": None,
                    "observed_state": "running",
                    "authority_state": "local_safe",
                }
            },
        )
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    try:
        for _ in range(8):
            coordinator._advance_initial_reconciliation([binding], revision)
            if coordinator.initially_reconciled(binding, revision):
                break
        assert coordinator.initially_reconciled(binding, revision)

        coordinator._advance_initial_reconciliation([binding], revision)
        assert coordinator.initially_reconciled(binding, revision)
    finally:
        coordinator.close()
        executor.shutdown()


def test_initial_reconciliation_stays_ready_during_ongoing_supervision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    revision, _bindings = runtime.load_registry()
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    try:
        coordinator._advance_initial_reconciliation([binding], revision)
        coordinator._advance_initial_reconciliation([binding], revision)
        assert coordinator.initially_reconciled(binding, revision)

        monkeypatch.setattr(coordinator, "_initial_reconciliation_pending", lambda *_args: True)
        assert coordinator.initially_reconciled(binding, revision)
    finally:
        coordinator.close()
        executor.shutdown()


def test_initial_reconciliation_reaches_tail_beyond_scan_state_capacity(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    revision, _bindings = runtime.load_registry()
    bindings = [replace(binding, project_id=f"project-{index:03d}") for index in range(300)]
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    runtime.authority_ready_generations = {}
    try:
        for _ in range(10):
            coordinator._advance_initial_reconciliation(bindings, revision)
            for candidate in bindings:
                if coordinator.initially_reconciled(candidate, revision):
                    runtime.authority_ready_generations[candidate.project_id] = candidate.registration_generation

        assert runtime.authority_ready_generations[bindings[-1].project_id] == bindings[-1].registration_generation
        assert len(coordinator._initial_reconciliation_scans) <= 256
    finally:
        coordinator.close()
        executor.shutdown()


@pytest.mark.parametrize("change", ["none", "wake", "directory", "epoch"])
def test_authority_quiescence_requires_a_current_complete_local_census(tmp_path, change):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, bindings = _registered(tmp_path, "project", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    try:
        _advance_until_validated(controller, bindings, revision, binding.project_id)
        _observe_working_set_activation(runtime, controller, bindings, revision)
        coordinator._advance_initial_reconciliation(bindings, revision)
        coordinator._advance_initial_reconciliation(bindings, revision)
        assert coordinator.initially_reconciled(binding, revision)
        state = runtime.working_set._states[runtime.working_set._identity(binding)]
        assert "authority" not in state.acknowledged_lanes

        processes = runtime.project_paths(binding.project_id)["processes"]
        for index in range(65):
            atomic_replace(processes / f"irrelevant-{index:03d}.txt", {})
        coordinator._advance_initial_reconciliation(bindings, revision)
        assert "authority" not in state.acknowledged_lanes
        if change == "wake":
            runtime.working_set.activate(binding, "new_local_work")
        elif change == "directory":
            atomic_replace(processes / "new-evidence.txt", {})
            signature = coordinator._initial_reconciliation_signature(binding, revision)
            census = coordinator._authority_quiescence[signature]
            assert not census.has_unchanged_directories()
        elif change == "epoch":
            executor.begin_epoch()

        coordinator._advance_initial_reconciliation(bindings, revision)
        assert ("authority" in state.acknowledged_lanes) is (change == "none")
        assert coordinator.initially_reconciled(binding, revision)
        for _ in range(3):
            coordinator._advance_initial_reconciliation(bindings, revision)
        assert "authority" in state.acknowledged_lanes
        assert coordinator._initial_reconciliation_scans == {}
    finally:
        coordinator.close()
        executor.shutdown()


def test_new_malformed_evidence_invalidates_a_cached_authority_idle_proof(tmp_path):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, bindings = _registered(tmp_path, "project", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    try:
        _advance_until_validated(controller, bindings, revision, binding.project_id)
        _observe_working_set_activation(runtime, controller, bindings, revision)
        for _ in range(3):
            coordinator._advance_initial_reconciliation(bindings, revision)
        state = runtime.working_set._states[runtime.working_set._identity(binding)]
        assert "authority" in state.acknowledged_lanes

        atomic_replace(runtime.project_paths(binding.project_id)["processes"] / "unknown-attempt.json", {})
        coordinator._advance_initial_reconciliation(bindings, revision)
        assert "authority" not in state.acknowledged_lanes
        assert coordinator.initially_reconciled(binding, revision)
    finally:
        coordinator.close()
        executor.shutdown()


@pytest.mark.parametrize("observed_state", ["running", "exited"])
def test_startup_readiness_does_not_certify_identity_unknown_process_quiescence(tmp_path, observed_state):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, bindings = _registered(tmp_path, "project", runtime)
    process = {
        "protocol_version": 1,
        "machine_name": binding.machine_name,
        "task_id": "unknown-task",
        "attempt_id": "unknown-task-attempt-1",
        "fencing_token": 1,
        "reservation_id": None,
        "wrapper_pid": None,
        "wrapper_start_time_ticks": None,
        "process_group_id": None,
        "process_group_start_time_ticks": None,
        "observed_state": observed_state,
        "observed_exited_at": "2026-10-01T00:00:00+00:00",
        "authority_state": "local_safe",
    }
    atomic_replace(
        runtime.project_paths(binding.project_id)["processes"] / "unknown-task-attempt-1.json", {"process": process}
    )
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    try:
        _advance_until_validated(controller, bindings, revision, binding.project_id)
        _observe_working_set_activation(runtime, controller, bindings, revision)
        for _ in range(4):
            coordinator._advance_initial_reconciliation(bindings, revision)
        assert coordinator.initially_reconciled(binding, revision)
        state = runtime.working_set._states[runtime.working_set._identity(binding)]
        assert "authority" not in state.acknowledged_lanes
    finally:
        coordinator.close()
        executor.shutdown()


def test_blocked_binding_validation_does_not_delay_healthy_peer(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    blocked_cfg, blocked, _revision, _bindings = _registered(tmp_path, "blocked", runtime)
    healthy_cfg, healthy, revision, bindings = _registered(tmp_path, "healthy", runtime)
    identity = shared_paths(blocked_cfg.shared_root)["project"] / "identity.json"
    identity.unlink()
    os.mkfifo(identity)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)

    first = controller.advance_binding_validation(bindings, revision)
    assert first == {}

    configs = _advance_until_validated(controller, bindings, revision, healthy.project_id)

    assert blocked.project_id not in configs
    assert configs[healthy.project_id].shared_root == healthy_cfg.shared_root
    assert configs[healthy.project_id].runtime_root == runtime.project_paths(healthy.project_id)["root"]
    assert executor.status_view()["active_worker_count"] == 1
    executor.shutdown()


def test_dispatch_uses_isolated_validation_and_services_healthy_peer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    blocked_cfg, blocked, _revision, _bindings = _registered(tmp_path, "blocked", runtime)
    _healthy_cfg, healthy, _revision, _bindings = _registered(tmp_path, "healthy", runtime)
    identity = shared_paths(blocked_cfg.shared_root)["project"] / "identity.json"
    identity.unlink()
    os.mkfifo(identity)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    runtime.project_io_executor = executor

    def reject_direct_binding_io(*_args, **_kwargs):
        raise AssertionError("dispatch bypassed isolated binding validation")

    monkeypatch.setattr(runtime, "registration_status", reject_direct_binding_io)
    monkeypatch.setattr(runtime, "binding_write_eligible", reject_direct_binding_io)
    deadline = time.monotonic() + 5.0
    results = []
    while time.monotonic() < deadline:
        cycle_started = time.monotonic()
        results = dispatch_machine_cycle_locked(
            runtime,
            available_gpus=[],
            supervise=False,
            publish_snapshots=False,
        )
        assert time.monotonic() - cycle_started < 1.0
        healthy_result = next((item for item in results if item["project_id"] == healthy.project_id), None)
        if healthy_result is not None and healthy_result["status"] == "dispatched":
            break
        time.sleep(0.02)

    assert next(item for item in results if item["project_id"] == blocked.project_id)["status"] == (
        "binding_validation_pending"
    )
    assert next(item for item in results if item["project_id"] == healthy.project_id)["status"] == "dispatched"
    unresolved = executor.unresolved_requests()
    assert any(
        request.project_id == blocked.project_id and request.operation_kind == "validate_binding"
        for request in unresolved
    )
    assert executor.status_view()["active_worker_count"] <= 2
    executor.shutdown()


def test_isolated_idle_census_does_not_require_simultaneous_lane_receipts(tmp_path, monkeypatch):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, bindings = _registered(tmp_path, "empty", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    runtime.project_io_executor = executor
    runtime.project_io_controller = controller
    # Suppress transient dispatch receipts, not the independent Source census.
    monkeypatch.setattr(controller, "advance_scheduler_observations", lambda *_args, **_kwargs: {})
    offered_classes = set()
    admission_type = type(controller._admission)
    original_offer = admission_type.offer

    def offer(self, intent, action):
        if self is controller._admission and intent.operation_kind == "scheduler_quiescence_probe":
            offered_classes.add(intent.service_class)
        return original_offer(self, intent, action)

    monkeypatch.setattr(admission_type, "offer", offer)
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            dispatch_machine_cycle_locked(runtime, available_gpus=[0], publish_snapshots=False)
            if controller.scheduler_is_quiescent(binding, revision):
                break
            time.sleep(0.02)
        else:
            raise AssertionError(
                f"independent idle census remained gated by receipts: {describe_project_io_workers(executor)}"
            )
        assert runtime.working_set.is_lane_quiescent(binding, "scheduler")
        assert offered_classes == {"primary"}
        assert not runtime.pending_launch_identities()
        snapshot = reservation_snapshot(runtime.root)
        assert not snapshot.active and not snapshot.provisional
        with runtime.agent_lifecycle_guard():
            runtime.activation_wake.publish_locked()
        # The next actual dispatch captures the wake before advancing services.
        dispatch_machine_cycle_locked(runtime, available_gpus=[0], publish_snapshots=False)
        assert not controller.scheduler_is_quiescent(binding, revision)
    finally:
        executor.shutdown()


def test_isolated_dispatch_launches_healthy_peer_without_legacy_project_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    blocked_cfg, blocked, _revision, _bindings = _registered(tmp_path, "blocked", runtime)
    healthy_cfg, healthy, revision, bindings = _registered(tmp_path, "healthy", runtime)
    submit(blocked_cfg, ["echo", "blocked"], working_dir=tmp_path)
    healthy_task = submit(healthy_cfg, ["echo", "healthy"], working_dir=tmp_path)
    project_io_executor = ProjectIOExecutor(runtime)
    request.addfinalizer(project_io_executor.shutdown)
    request_timeline = []
    timeline_origin = time.monotonic()
    original_start = project_io_executor.start
    original_consume = project_io_executor.consume

    def record_start(request_id, **reconciliation_options):
        process = original_start(request_id, **reconciliation_options)
        if process is not None:
            request_timeline.append(
                (
                    round(time.monotonic() - timeline_origin, 3),
                    "start",
                    process.request.project_id,
                    process.request.operation_kind,
                )
            )
        return process

    def record_consume(request_id, current_request):
        result = original_consume(request_id, current_request)
        if result is not None:
            request_timeline.append(
                (
                    round(time.monotonic() - timeline_origin, 3),
                    "consume",
                    result.request.project_id,
                    result.request.operation_kind,
                )
            )
        return result

    monkeypatch.setattr(project_io_executor, "start", record_start)
    monkeypatch.setattr(project_io_executor, "consume", record_consume)
    project_io_executor.begin_epoch()
    controller = ProjectIOController(runtime, project_io_executor)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if set(controller.advance_binding_validation(bindings, revision)) == {
            binding.project_id for binding in bindings
        }:
            break
        time.sleep(0.02)
    else:
        raise AssertionError("bindings did not validate")
    runtime.project_io_executor = project_io_executor
    runtime.project_io_controller = controller
    runtime.authority_ready_generations = {}

    recovering = dispatch_machine_cycle_locked(
        runtime,
        available_gpus=[0],
        supervise=True,
        publish_snapshots=True,
    )
    assert {item["status"] for item in recovering} == {"authority_recovering"}
    assert all(request.operation_kind != "scheduler_observe" for request in project_io_executor.unresolved_requests())
    runtime.authority_ready_generations = {
        blocked.project_id: blocked.registration_generation,
        healthy.project_id: healthy.registration_generation,
    }

    blocked_state = ready_state_path(blocked_cfg.shared_root)
    blocked_state.unlink()
    os.mkfifo(blocked_state)
    launched: list[dict[str, object]] = []
    launch_attempts = 0

    class TicketExecutor:
        def initiate_authorized_attempt(self, cfg, **ticket):
            nonlocal launch_attempts
            launch_attempts += 1
            if launch_attempts == 1:
                raise OSError("synthetic Popen failure")
            launched.append({"cfg": cfg, **ticket})
            attempt_id = ticket["claim_identity"]["attempt_id"]
            intent = cfg.runtime_root / "launch-intents" / f"{attempt_id}.json"
            intent.parent.mkdir(parents=True, exist_ok=True)
            intent.touch()
            process = type("Process", (), {"pid": 4321})()
            handle = LaunchHandle("detached", process, runner_process=process)
            return handle, LaunchHandoff(attempt_id, intent, time.monotonic() + 60, handle)

    def reject_legacy_project_io(*_args, **_kwargs):
        raise AssertionError("isolated dispatch entered the legacy synchronous Project-I/O path")

    for name in (
        "_reconcile_machine_reservations",
        "maintain_project",
        "_recover_starting_reservations",
        "read_ready_index_state",
        "advance_ready_index_build",
        "_probe_primary_demand",
        "run_dispatch_cycle",
        "_publish_project_snapshots",
        "_acknowledge_scheduler_turns",
    ):
        monkeypatch.setattr(dispatch_loop, name, reject_legacy_project_io)

    results: list[dict[str, object]] = []
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline and not launched:
        cycle_started = time.monotonic()
        results = dispatch_machine_cycle_locked(
            runtime,
            available_gpus=[0],
            executor=TicketExecutor(),
            supervise=True,
            publish_snapshots=True,
        )
        assert time.monotonic() - cycle_started < 1.0
        time.sleep(0.02)

    assert len(launched) == 1, {
        "launch_attempts": launch_attempts,
        "results": results,
        "backoff": [(key[0].project_id, key[1:], state) for key, state in controller._service_backoff.items()],
        "requests": [
            (request.project_id, request.operation_kind) for request in project_io_executor.unresolved_requests()
        ],
        "workers": describe_project_io_workers(project_io_executor),
        "timeline": request_timeline,
    }
    assert launch_attempts == 2
    assert launched[0]["cfg"].shared_root == healthy_cfg.shared_root
    assert launched[0]["claim_identity"]["task_id"] == healthy_task.task_id
    assert launched[0]["launch_id"]
    healthy_result = next(item for item in results if item["project_id"] == healthy.project_id)
    assert healthy_result["launched"] == [healthy_task.task_id]
    # Shared background publication/repair is now isolated too, and may still
    # be running on the healthy binding when its foreground launch completes.
    # Preserve the actual isolation invariant: exactly one blocked request,
    # no duplicated owner, and bounded workers for these two bindings.
    status = project_io_executor.status_view()
    requests = project_io_executor.unresolved_requests()
    assert 1 <= status["active_worker_count"] <= 2
    assert sum(request.project_id == blocked.project_id for request in requests) == 1
    assert len(requests) == len({(request.project_id, request.registration_generation) for request in requests})
    project_io_executor.shutdown()


def test_isolated_dispatch_advances_due_offer_without_legacy_maintenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, _revision, _bindings = _registered(tmp_path, "project", runtime)
    create_group(cfg, "exp")
    task = submit(
        cfg,
        ["echo", "due"],
        group="exp",
        sharing_mode="spillover",
        offer_after_seconds=0,
    )
    stored = load_task(cfg, task.task_id)
    stored.placement_runtime["offer_eligible_at"] = "2000-01-01T00:00:00Z"
    save_task(cfg, stored)
    sync_deadline_index(cfg, stored)
    assert load_task(cfg, task.task_id).placement_runtime["queue_scope"] == "home"
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    runtime.project_io_executor = executor

    def reject_legacy(*_args, **_kwargs):
        raise AssertionError("isolated dispatch used synchronous Project maintenance")

    monkeypatch.setattr(dispatch_loop, "maintain_project", reject_legacy)
    try:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            dispatch_machine_cycle_locked(
                runtime,
                available_gpus=[],
                supervise=True,
                publish_snapshots=False,
            )
            if load_task(cfg, task.task_id).placement_runtime["queue_scope"] == "shared":
                break
            time.sleep(0.02)
        assert load_task(cfg, task.task_id).placement_runtime["queue_scope"] == "shared"
        isolated_cursor = (
            local_paths(runtime.project_paths(binding.project_id)["root"])["maintenance_cursors"]
            / "offer_deadlines.json"
        )
        legacy_cursor = local_paths(cfg.runtime_root)["maintenance_cursors"] / "offer_deadlines.json"
        assert isolated_cursor.exists()
        assert legacy_cursor != isolated_cursor
        assert not legacy_cursor.exists()
    finally:
        runtime.attempt_supervision_coordinator.close()
        executor.shutdown()


@pytest.mark.parametrize("lane", ["gpu", "cpu"])
@pytest.mark.parametrize("impossible_primary", [False, True])
def test_isolated_dispatch_admits_borrow_after_complete_primary_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lane: str,
    impossible_primary: bool,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    if lane == "cpu":
        set_cpu_lane_capacity(runtime.root, capacity=1)
    cfg, binding, _revision, _bindings = _registered(tmp_path, "borrow-project", runtime)
    if impossible_primary:
        submit(
            cfg,
            ["echo", "oversized-primary"],
            working_dir=tmp_path,
            requested_gpus=3 if lane == "gpu" else 0,
            requested_cpus=3 if lane == "cpu" else None,
        )
    create_group(cfg, "borrow-group")
    change_worker(cfg, "borrow-group", "gpu-1", "set", role="borrow")
    task = submit(
        cfg,
        ["echo", "borrow"],
        group="borrow-group",
        sharing_mode="spillover",
        working_dir=tmp_path,
        requested_gpus=1 if lane == "gpu" else 0,
        requested_cpus=1 if lane == "cpu" else None,
    )
    share(cfg, task.task_id)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    runtime.project_io_executor = executor

    class TicketExecutor:
        def initiate_authorized_attempt(self, launch_cfg, **ticket):
            attempt_id = ticket["claim_identity"]["attempt_id"]
            intent = launch_cfg.runtime_root / "launch-intents" / f"{attempt_id}.json"
            intent.parent.mkdir(parents=True, exist_ok=True)
            intent.touch()
            process = type("Process", (), {"pid": 4321})()
            handle = LaunchHandle("detached", process, runner_process=process)
            return handle, LaunchHandoff(attempt_id, intent, time.monotonic() + 60, handle)

    try:
        deadline = time.monotonic() + 30.0
        claim = None
        while time.monotonic() < deadline:
            dispatch_machine_cycle_locked(
                runtime,
                available_gpus=[0] if lane == "gpu" else [],
                executor=TicketExecutor(),
                supervise=True,
                publish_snapshots=False,
            )
            claim = load_task(cfg, task.task_id).claim_control.get("active_claim")
            if isinstance(claim, dict) and active_reservations(runtime.root):
                break
            time.sleep(0.02)

        assert isinstance(claim, dict), describe_project_io_workers(executor)
        assert claim["worker_scheduling_role"] == "borrow"
        assert claim["admitted_as_borrow"] is True
        reservations = active_reservations(runtime.root)
        assert len(reservations) == 1
        assert reservations[0]["project_id"] == binding.project_id
        assert reservations[0]["admission"]["admitted_as_borrow"] is True
    finally:
        coordinator = getattr(runtime, "attempt_supervision_coordinator", None)
        if coordinator is not None:
            coordinator.close()
        executor.shutdown()


@pytest.mark.parametrize("fence", ["expired", "copied", "capacity", "registry"])
def test_borrow_capability_is_exact_one_turn_and_rechecked_before_reservation(tmp_path, fence):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "borrow", runtime)
    create_group(cfg, "borrow-group")
    change_worker(cfg, "borrow-group", "gpu-1", "set", role="borrow")
    task = submit(cfg, ["echo", "borrow"], group="borrow-group", sharing_mode="spillover", working_dir=tmp_path)
    share(cfg, task.task_id)
    repair_ready_index(cfg)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    try:
        _advance_until_validated(controller, bindings, revision, binding.project_id)
        observations = _advance_until_observed(
            controller, bindings, revision, binding.project_id, lane="gpu", admission_role="borrow"
        )
        assert observations[binding.project_id]["outcome"] == "candidate"
        grant = None
        deadline = time.monotonic() + 5

        def try_claim(capability):
            controller.advance_scheduler_claims(
                bindings,
                revision,
                lane="gpu",
                admission_role="borrow",
                observations=observations,
                available_gpu_ids=[0, 1],
                available_cpu_slots=0,
                borrow_admission=capability,
            )

        while time.monotonic() < deadline:
            with controller.admission_turn():
                grant = controller.issue_borrow_admission(
                    bindings, revision, lane="gpu", visible_capacity=2, free_capacity=2
                )
                if grant is not None and fence != "expired":
                    if fence == "copied":
                        grant = replace(grant)
                    elif fence == "capacity":
                        reserve(runtime.root, "other-task", [1])
                    else:
                        _registered(tmp_path, "new-primary", runtime)
                    try_claim(grant)
            if grant is not None:
                break
            time.sleep(0.02)
        assert grant is not None
        if fence == "expired":
            try_claim(grant)
        assert not any(request.operation_kind == "scheduler_claim" for request in executor.unresolved_requests())
        assert load_task(cfg, task.task_id).claim_control.get("active_claim") is None
        assert all(
            record.get("project_id") != binding.project_id for record in reservation_snapshot(runtime.root).reservations
        )
    finally:
        executor.shutdown()


def test_unselected_borrow_grant_retains_scan_but_requires_fresh_verification(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, _bindings = _registered(tmp_path, "borrow", runtime)
    executor = ProjectIOExecutor(runtime)
    epoch = executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    try:
        identity = controller._binding_identity(binding, revision, epoch)
        assert identity is not None
        identities = frozenset({identity})
        state = project_io_controller_module._PrimaryProbeRound(
            "round",
            identities,
            "digest",
            {identity: {"continuation": "retained"}},
            {identity},
            {identity},
            "verify",
        )
        grant = project_io_controller_module.BorrowAdmissionGrant(
            executor.runtime_id, epoch, revision, "gpu", identities, "digest"
        )
        with controller.admission_turn():
            controller._primary_rounds["gpu"] = state
            controller._borrow_grants["gpu"] = grant
        assert controller._primary_rounds["gpu"] is state
        assert state.scanned == {identity}
        assert state.states == {identity: {"continuation": "retained"}}
        assert state.phase == "verify"
        assert state.verified == set()
        assert not controller._borrow_grants
    finally:
        executor.shutdown()


@pytest.mark.parametrize("transition", ["resume", "restart", "disable", "epoch"])
@pytest.mark.parametrize("lane", ["gpu", "cpu"])
def test_borrow_offer_survives_missed_request_grant_without_reusing_proof(
    tmp_path: Path, transition: str, lane: str
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    if lane == "cpu":
        set_cpu_lane_capacity(runtime.root, capacity=1)
    cfg, binding, revision, bindings = _registered(tmp_path, "borrow", runtime)
    create_group(cfg, "borrow-group")
    change_worker(cfg, "borrow-group", "gpu-1", "set", role="borrow")
    task = submit(
        cfg,
        ["echo", "borrow"],
        group="borrow-group",
        sharing_mode="spillover",
        working_dir=tmp_path,
        requested_gpus=1 if lane == "gpu" else 0,
        requested_cpus=1 if lane == "cpu" else None,
    )
    share(cfg, task.task_id)
    repair_ready_index(cfg)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    try:
        _advance_until_validated(controller, bindings, revision, binding.project_id)
        observations = _advance_until_observed(
            controller, bindings, revision, binding.project_id, lane=lane, admission_role="borrow"
        )
        background = []
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with controller.admission_turn():
                grant = controller.issue_borrow_admission(
                    bindings,
                    revision,
                    lane=lane,
                    visible_capacity=1,
                    free_capacity=1,
                )
                if grant is not None:
                    # Exhaust the bounded critical-chain preference so a real
                    # competing class opportunity wins this pass. The local
                    # offer must not depend on selection of a new request.
                    identity = controller._binding_identity(binding, revision, grant.executor_epoch)
                    candidate = observations[binding.project_id]["candidate"]
                    controller._admission._critical_chain_grants[identity.request_owner] = (
                        project_io_admission_module._CriticalChainCredit(
                            (grant.executor_epoch, candidate["attempt_id"]), grants=3
                        )
                    )
                    controller._admission._arbiter._class_cursor = 3
                    controller._offer_new_request(identity, "scheduler_due_offer", lambda: background.append(True))
                    for _ in range(2):
                        controller.advance_scheduler_claims(
                            bindings,
                            revision,
                            lane=lane,
                            admission_role="borrow",
                            observations=observations,
                            available_gpu_ids=[0] if lane == "gpu" else [],
                            available_cpu_slots=1 if lane == "cpu" else 0,
                            borrow_admission=grant,
                        )
            if grant is not None:
                break
            time.sleep(0.02)
        assert grant is not None
        assert background == [True], {
            "pending_admission": controller._admission.pending_count,
            "deferred_admission": controller._admission.has_deferred_work,
            "requests": [(request.operation_kind, request.project_id) for request in executor.unresolved_requests()],
        }
        assert not controller._borrow_grants
        assert len(controller._pending_claim_offers) == 1
        assert not any(request.operation_kind == "scheduler_claim" for request in executor.unresolved_requests())
        assert load_task(cfg, task.task_id).claim_control.get("active_claim") is None
        offers = reservation_snapshot(runtime.root).provisional
        assert len(offers) == 1
        offer_identity = ReservationIdentity.from_record(offers[0])
        assert offers[0]["admission"]["admitted_as_borrow"] is True

        if transition == "restart":
            executor.begin_epoch()
            controller = ProjectIOController(runtime, executor)
        elif transition == "disable":
            runtime.set_enabled(binding.project_id, False)
            revision, bindings = runtime.load_registry()
        elif transition == "epoch":
            executor.begin_epoch()

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            controller.advance_scheduler_claims(
                bindings,
                revision,
                lane=lane,
                admission_role="borrow",
                observations={},
                available_gpu_ids=[],
                available_cpu_slots=0,
            )
            state = classify_executor_offer(runtime.root, offer_identity)
            if state == ("matching_active" if transition == "resume" else "matching_released"):
                break
            time.sleep(0.02)
        else:
            raise AssertionError("deferred borrow offer did not converge")
        assert not controller._pending_claim_offers
        claim = load_task(cfg, task.task_id).claim_control.get("active_claim")
        if transition == "resume":
            assert isinstance(claim, dict)
            assert claim["admitted_as_borrow"] is True
        else:
            assert claim is None
    finally:
        executor.shutdown()


def test_due_offer_worker_reports_noop_and_disabled_binding_starts_nothing(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, bindings = _registered(tmp_path, "project", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)

    completions = {}
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        completions = controller.advance_scheduler_due_offers(bindings, revision)
        if binding.project_id in completions:
            break
        time.sleep(0.02)
    assert completions[binding.project_id] == {
        "outcome": "noop",
        "reason": "no_due_deadline",
        "task_id": None,
    }

    disabled = replace(binding, enabled=False)
    controller.advance_scheduler_due_offers([disabled], revision)
    assert all(request.operation_kind != "scheduler_due_offer" for request in executor.unresolved_requests())
    executor.shutdown()


@pytest.mark.parametrize("fence", ["binding", "epoch"])
def test_due_offer_rechecks_authority_inside_availability_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fence: str,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, _bindings = _registered(tmp_path, "project", runtime)
    create_group(cfg, "exp")
    task = submit(
        cfg,
        ["echo", "due"],
        group="exp",
        sharing_mode="spillover",
        offer_after_seconds=0,
    )
    stored = load_task(cfg, task.task_id)
    stored.placement_runtime["offer_eligible_at"] = "2000-01-01T00:00:00Z"
    save_task(cfg, stored)
    sync_deadline_index(cfg, stored)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    request = executor.prepare_scheduler_due_offer(binding, revision)
    real_offer = project_maintenance_module.offer

    def fence_then_offer(*args, **kwargs):
        if fence == "binding":
            runtime.set_enabled(cfg.shared_root, False)
        else:
            executor.fence_epoch()
        return real_offer(*args, **kwargs)

    monkeypatch.setattr(project_maintenance_module, "offer", fence_then_offer)
    with pytest.raises(
        (
            project_io_worker_module._BindingAuthorityChanged,
            project_io_worker_module._ExecutorEpochFenced,
        )
    ):
        project_io_worker_module._scheduler_due_offer(
            request,
            runtime.root,
            executor.paths,
            [False],
        )
    assert load_task(cfg, task.task_id).placement_runtime["queue_scope"] == "home"
    executor.shutdown()


def test_due_offer_result_reflects_locked_elapsed_proof(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, _binding, _revision, _bindings = _registered(tmp_path, "project", runtime)
    create_group(cfg, "exp")
    task = submit(
        cfg,
        ["echo", "due"],
        group="exp",
        sharing_mode="spillover",
        offer_after_seconds=0,
    )
    monkeypatch.setattr(project_maintenance_module, "elapsed_offer_is_proven", lambda *_args: True)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.runtime.availability.transitions.elapsed_offer_is_proven",
        lambda *_args: False,
    )

    progress = project_maintenance_module.advance_due_offer(cfg)

    assert progress.outcome == "noop"
    assert progress.reason == "elapsed_offer_unproven"
    assert progress.task_id == task.task_id
    assert load_task(cfg, task.task_id).placement_runtime["queue_scope"] == "home"


def test_due_offer_fences_clock_observation_before_shared_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, _bindings = _registered(tmp_path, "project", runtime)
    create_group(cfg, "exp")
    task = submit(
        cfg,
        ["echo", "due"],
        group="exp",
        sharing_mode="spillover",
        offer_after_seconds=0,
    )
    stored = load_task(cfg, task.task_id)
    stored.placement_runtime["offer_eligible_at"] = "2000-01-01T00:00:00Z"
    save_task(cfg, stored)
    sync_deadline_index(cfg, stored)
    observation_root = shared_paths(cfg.shared_root)["clock_observations"] / cfg.machine_name
    for path in observation_root.glob("*.json"):
        path.unlink()
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    request = executor.prepare_scheduler_due_offer(binding, revision)
    real_persist = availability_transitions_module.persist_clock_observation

    def fence_then_persist(*args, **kwargs):
        runtime.set_enabled(cfg.shared_root, False)
        return real_persist(*args, **kwargs)

    monkeypatch.setattr(availability_transitions_module, "persist_clock_observation", fence_then_persist)
    write_possible = [False]
    with pytest.raises(project_io_worker_module._BindingAuthorityChanged):
        project_io_worker_module._scheduler_due_offer(
            request,
            runtime.root,
            executor.paths,
            write_possible,
        )
    assert write_possible == [False]
    assert list(observation_root.glob("*.json")) == []
    assert load_task(cfg, task.task_id).placement_runtime["queue_scope"] == "home"
    executor.shutdown()


def test_due_offer_fences_deadline_directory_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, _bindings = _registered(tmp_path, "project", runtime)
    create_group(cfg, "exp")
    task = submit(
        cfg,
        ["echo", "due"],
        group="exp",
        sharing_mode="spillover",
        offer_after_seconds=0,
    )
    stored = load_task(cfg, task.task_id)
    stored.placement_runtime["offer_eligible_at"] = "2000-01-01T00:00:00Z"
    save_task(cfg, stored)
    sync_deadline_index(cfg, stored)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    request = executor.prepare_scheduler_due_offer(binding, revision)
    real_sync = offer_deadline_module._sync_directory
    sync_calls: list[Path] = []

    def fence_then_sync(path: Path) -> None:
        sync_calls.append(path)
        runtime.set_enabled(cfg.shared_root, False)
        real_sync(path)

    monkeypatch.setattr(offer_deadline_module, "_sync_directory", fence_then_sync)
    write_possible = [False]
    with pytest.raises(project_io_worker_module._BindingAuthorityChanged):
        project_io_worker_module._scheduler_due_offer(
            request,
            runtime.root,
            executor.paths,
            write_possible,
        )
    assert sync_calls
    assert write_possible == [True]
    executor.shutdown()


def test_due_offer_rotation_reaches_binding_outside_first_sixty_four(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, _bindings = _registered(tmp_path, "project", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor, monotonic=lambda: 100.0)
    executor_epoch = executor.status_view()["executor_epoch"]
    assert isinstance(executor_epoch, str)
    bindings = [replace(binding, project_id=f"project-{index:03d}") for index in range(65)]
    identities = []
    for item in bindings:
        identity = controller._binding_identity(item, revision, executor_epoch)
        assert identity is not None
        controller._validated.add(identity)
        identities.append(identity)
    for identity in identities[:64]:
        controller._record_service_failure((identity, "scheduler_due_offer", "deadline", "advance"))
    selected: list[str] = []

    def record_offer(identity, _kind, _prepare, *_key):
        selected.append(identity.project_id)

    monkeypatch.setattr(controller, "_offer_new_request", record_offer)
    controller.advance_scheduler_due_offers(bindings, revision)
    assert selected == []
    controller.advance_scheduler_due_offers(bindings, revision)
    assert selected == [bindings[-1].project_id]
    executor.shutdown()


def _reset_ready_index_for_isolated_build(cfg) -> None:
    state_path = ready_state_path(cfg.shared_root)
    value = read_json(state_path)
    value["ready_index"].update(
        state="absent",
        writer_capability=None,
        build=None,
        degraded_reasons=[],
    )
    atomic_replace(state_path, value)
    (shared_paths(cfg.shared_root)["ready_primary"] / "state.json").unlink(missing_ok=True)


def test_ready_index_worker_uses_project_machine_runtime_and_builds_to_active(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project", runtime)
    _reset_ready_index_for_isolated_build(cfg)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)

    completions = {}
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        completions = controller.advance_scheduler_ready_index_builds(bindings, revision)
        if completions.get(binding.project_id, {}).get("state") == "active":
            break
        time.sleep(0.02)
    assert read_ready_index_state(cfg) == "active"
    assert binding.project_id in completions
    assert completions[binding.project_id] == {
        "state": "active",
        "revision": completions[binding.project_id]["revision"],
        "build_id": None,
        "phase": None,
    }

    request = executor.prepare_scheduler_ready_index_build(binding, revision)
    observed_runtime_roots: list[Path] = []

    def observe_runtime_root(worker_cfg, **_kwargs):
        observed_runtime_roots.append(worker_cfg.runtime_root)
        return {"state": "active", "revision": 1, "build": None}

    original = project_io_worker_module.advance_ready_index_build
    project_io_worker_module.advance_ready_index_build = observe_runtime_root
    try:
        evidence = project_io_worker_module._scheduler_ready_index_build(
            request,
            runtime.root,
            executor.paths,
            [False],
        )
    finally:
        project_io_worker_module.advance_ready_index_build = original
    assert evidence == {"state": "active", "revision": 1, "build_id": None, "phase": None}
    assert observed_runtime_roots == [runtime.project_paths(binding.project_id)["root"]]
    assert observed_runtime_roots[0] != cfg.runtime_root
    executor.shutdown()


def test_isolated_dispatch_builds_ready_index_without_synchronous_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project", runtime)
    _reset_ready_index_for_isolated_build(cfg)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    runtime.project_io_executor = executor
    runtime.project_io_controller = controller
    runtime.authority_ready_generations = {binding.project_id: binding.registration_generation}

    def reject_synchronous_build(*_args, **_kwargs):
        raise AssertionError("isolated dispatch used the synchronous ready-index builder")

    monkeypatch.setattr(dispatch_loop, "read_ready_index_state", reject_synchronous_build)
    monkeypatch.setattr(dispatch_loop, "advance_ready_index_build", reject_synchronous_build)
    try:
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            dispatch_machine_cycle_locked(
                runtime,
                available_gpus=[0],
                supervise=True,
                publish_snapshots=False,
            )
            if read_ready_index_state(cfg) == "active":
                break
            time.sleep(0.02)
        assert read_ready_index_state(cfg) == "active", {
            "workers": describe_project_io_workers(executor),
            "needed": len(controller._ready_index_needed),
            "settled": len(controller._ready_index_settled),
            "queued": controller._admission.pending_count,
            "working_set": runtime.working_set.snapshot(),
            "backoff": [(key[1:], value) for key, value in controller._service_backoff.items()],
        }
    finally:
        runtime.attempt_supervision_coordinator.close()
        executor.shutdown()


def test_ready_index_inactive_observation_resumes_later_bounded_repair(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    _observe_working_set_activation(
        runtime,
        controller,
        bindings,
        revision,
        project_ids={binding.project_id},
    )

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        completion = controller.advance_scheduler_ready_index_builds(bindings, revision)
        if completion.get(binding.project_id, {}).get("state") == "active":
            break
        time.sleep(0.02)
    else:
        raise AssertionError("active ready index was not observed")

    state_path = ready_state_path(cfg.shared_root)
    value = read_json(state_path)
    value["ready_index"].update(state="degraded", degraded_reasons=[])
    atomic_replace(state_path, value)
    progress = repair_ready_index(cfg, max_tasks=1, bounded_initialization=True)
    assert progress["state"] == "building"

    observations = {}
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        observations = controller.advance_scheduler_observations(
            bindings,
            revision,
            lane="gpu",
            admission_role="primary",
        )
        if observations.get(binding.project_id, {}).get("reason") == "ready_index_inactive":
            break
        time.sleep(0.02)
    assert observations[binding.project_id]["reason"] == "ready_index_inactive"

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        controller.advance_scheduler_ready_index_builds(bindings, revision)
        if read_ready_index_state(cfg) == "active":
            break
        time.sleep(0.02)
    assert read_ready_index_state(cfg) == "active"
    executor.shutdown()


@pytest.mark.parametrize("admission_role", ["primary", "borrow"])
@pytest.mark.parametrize("lane", ["gpu", "cpu"])
def test_isolated_dispatch_drains_finished_observation_after_capacity_disappears(
    tmp_path: Path, admission_role: str, lane: str
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, bindings = _registered(tmp_path, "project", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    runtime.project_io_executor = executor
    runtime.project_io_controller = controller
    try:
        _advance_until_validated(controller, bindings, revision, binding.project_id)
        _observe_working_set_activation(runtime, controller, bindings, revision)
        request = executor.prepare_scheduler_observe(
            binding,
            revision,
            lane=lane,
            admission_role=admission_role,
            cursor_namespace=f"scheduler-{binding.project_id}-{admission_role}-{lane}",
        )
        assert executor.start(request.request_id) is not None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = executor.poll()
            if executor.load_result(request.request_id) is not None and status["active_worker_count"] == 0:
                break
            time.sleep(0.02)
        else:
            raise AssertionError("observation did not finish")
        # Capacity may be consumed by another binding before this result is
        # applied. Its owner slot must still become available to other lanes.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            dispatch_machine_cycle_locked(runtime, available_gpus=[], publish_snapshots=False)
            pending = executor.unresolved_requests()
            if all(item.request_id != request.request_id for item in pending):
                break
            time.sleep(0.02)
        else:
            raise AssertionError("completed observation retained its owner without launch capacity")
        assert all(item.operation_kind != "scheduler_observe" for item in pending)
        assert active_reservations(runtime.root) == []
    finally:
        coordinator = getattr(runtime, "attempt_supervision_coordinator", None)
        if coordinator is not None:
            coordinator.close()
        executor.shutdown()


def test_isolated_dispatch_does_not_start_ready_build_without_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project", runtime)
    _reset_ready_index_for_isolated_build(cfg)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    runtime.project_io_executor = executor
    runtime.project_io_controller = controller
    runtime.authority_ready_generations = {binding.project_id: binding.registration_generation}
    calls: list[object] = []
    monkeypatch.setattr(controller, "advance_scheduler_ready_index_builds", lambda *_args, **_kwargs: calls.append(1))
    try:
        dispatch_machine_cycle_locked(
            runtime,
            available_gpus=[],
            supervise=True,
            publish_snapshots=False,
        )
        assert calls == []
        assert read_ready_index_state(cfg) == "absent"
    finally:
        runtime.attempt_supervision_coordinator.close()
        executor.shutdown()


def test_ready_index_build_rechecks_binding_before_json_state_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, _bindings = _registered(tmp_path, "project", runtime)
    _reset_ready_index_for_isolated_build(cfg)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    request = executor.prepare_scheduler_ready_index_build(binding, revision)
    real_replace = ready_state_module.atomic_replace

    def fence_then_replace(*args, **kwargs):
        runtime.set_enabled(cfg.shared_root, False)
        return real_replace(*args, **kwargs)

    monkeypatch.setattr(ready_state_module, "atomic_replace", fence_then_replace)
    write_possible = [False]
    with pytest.raises(project_io_worker_module._BindingAuthorityChanged):
        project_io_worker_module._scheduler_ready_index_build(
            request,
            runtime.root,
            executor.paths,
            write_possible,
        )
    assert write_possible == [True]
    assert read_json(ready_state_path(cfg.shared_root))["ready_index"]["state"] == "absent"
    executor.shutdown()


def test_ready_index_build_fences_activation_directory_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, _bindings = _registered(tmp_path, "project", runtime)
    _reset_ready_index_for_isolated_build(cfg)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    request = executor.prepare_scheduler_ready_index_build(binding, revision)
    real_sync = project_activation_module._sync_directory
    sync_calls: list[Path] = []

    def fence_then_sync(path: Path) -> None:
        sync_calls.append(path)
        runtime.set_enabled(cfg.shared_root, False)
        real_sync(path)

    monkeypatch.setattr(project_activation_module, "_sync_directory", fence_then_sync)
    write_possible = [False]
    with pytest.raises(project_io_worker_module._BindingAuthorityChanged):
        project_io_worker_module._scheduler_ready_index_build(
            request,
            runtime.root,
            executor.paths,
            write_possible,
        )
    assert sync_calls
    assert write_possible == [True]
    executor.shutdown()


@pytest.mark.parametrize("fence", ["binding", "epoch"])
def test_ready_index_build_rechecks_authority_before_non_json_layout_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fence: str,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, _bindings = _registered(tmp_path, "project", runtime)
    _reset_ready_index_for_isolated_build(cfg)
    missing = shared_paths(cfg.shared_root)["ready_catalogs"]
    missing.rmdir()
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    request = executor.prepare_scheduler_ready_index_build(binding, revision)
    real_check = ready_state_module.check_mutation_fence

    def fence_then_check(path):
        if fence == "binding":
            runtime.set_enabled(cfg.shared_root, False)
        else:
            executor.fence_epoch()
        return real_check(path)

    monkeypatch.setattr(ready_state_module, "check_mutation_fence", fence_then_check)
    write_possible = [False]
    with pytest.raises(
        (project_io_worker_module._BindingAuthorityChanged, project_io_worker_module._ExecutorEpochFenced)
    ):
        project_io_worker_module._scheduler_ready_index_build(
            request,
            runtime.root,
            executor.paths,
            write_possible,
        )
    assert write_possible == [False]
    assert not missing.exists()
    executor.shutdown()


def test_ready_index_build_rotation_reaches_binding_outside_first_sixty_four(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, _bindings = _registered(tmp_path, "project", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor, monotonic=lambda: 100.0)
    executor_epoch = executor.status_view()["executor_epoch"]
    assert isinstance(executor_epoch, str)
    bindings = [replace(binding, project_id=f"project-{index:03d}") for index in range(65)]
    identities = []
    for item in bindings:
        identity = controller._binding_identity(item, revision, executor_epoch)
        assert identity is not None
        controller._validated.add(identity)
        identities.append(identity)
    for identity in identities[:64]:
        controller._record_service_failure((identity, "scheduler_ready_index_build", "ready_index", "advance"))
    selected: list[str] = []

    def record_offer(identity, _kind, _prepare, *_key):
        selected.append(identity.project_id)

    monkeypatch.setattr(controller, "_offer_new_request", record_offer)
    controller.advance_scheduler_ready_index_builds(bindings, revision)
    assert selected == []
    controller.advance_scheduler_ready_index_builds(bindings, revision)
    assert selected == [bindings[-1].project_id]
    executor.shutdown()


def test_ready_index_repair_intents_are_bounded_and_re_admit_healthy_eviction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, _bindings = _registered(tmp_path, "project", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor, monotonic=lambda: 100.0)
    executor_epoch = executor.status_view()["executor_epoch"]
    assert isinstance(executor_epoch, str)
    identities = []
    for index in range(65):
        identity = controller._binding_identity(
            replace(binding, project_id=f"project-{index:03d}"),
            revision,
            executor_epoch,
        )
        assert identity is not None
        identities.append(identity)
        controller._mark_ready_index_needed(identity)

    assert len(controller._ready_index_needed) == 64
    assert identities[0] not in controller._ready_index_needed
    assert identities[-1] in controller._ready_index_needed
    for identity in identities[1:]:
        controller._validated.add(identity)
        controller._record_service_failure((identity, "scheduler_ready_index_build", "ready_index", "advance"))
    controller._validated.add(identities[0])

    for identity in identities:
        controller._mark_ready_index_needed(identity)

    assert len(controller._ready_index_needed) == 64
    assert identities[0] in controller._ready_index_needed
    selected: list[str] = []

    def record_offer(identity, _kind, _prepare, *_key):
        selected.append(identity.project_id)

    monkeypatch.setattr(controller, "_offer_new_request", record_offer)
    bindings = [replace(binding, project_id=f"project-{index:03d}") for index in range(65)]
    controller.advance_scheduler_ready_index_builds(bindings, revision, observed_only=True)

    assert selected == [identities[0].project_id]
    executor.shutdown()


def test_ready_index_build_prewrite_binding_failure_is_retryable(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, _bindings = _registered(tmp_path, "project", runtime)
    _reset_ready_index_for_isolated_build(cfg)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    request = executor.prepare_scheduler_ready_index_build(binding, revision)
    runtime.set_enabled(cfg.shared_root, False)
    assert executor.start(request.request_id) is not None

    result = None
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        executor.poll()
        result = executor.load_result(request.request_id)
        if result is not None:
            break
        time.sleep(0.02)
    assert result is not None
    assert result.status == "retryable_error"
    assert result.reason_code == "project_io_scheduler_ready_index_build_failed"
    executor.shutdown()


def _activate_missing_deadline_descriptor(cfg, *, suffix: str = "maintenance"):
    prepared = maintenance_outbox_module.prepare_work(
        cfg,
        kind="deadline_index",
        target_id=f"missing-task-{suffix}",
        work_generation=f"generation-{suffix}",
        phase="repair",
    )
    return maintenance_outbox_module.activate_work(cfg, prepared)


def test_maintenance_descriptor_worker_uses_project_machine_runtime_and_one_slice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, _bindings = _registered(tmp_path, "project", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    request = executor.prepare_maintenance_descriptor_advance(binding, revision)
    observed: list[tuple[Path, Path | None, int]] = []

    def observe_slice(worker_cfg, *, reservation_runtime_root, max_scan):
        observed.append((worker_cfg.runtime_root, reservation_runtime_root, max_scan))
        return {"maintenance_state": "idle", "next_due_at": None, "more": False}

    monkeypatch.setattr(maintenance_module, "advance_maintenance_work", observe_slice)
    evidence = project_io_worker_module._maintenance_descriptor_advance(
        request,
        runtime.root,
        executor.paths,
        [False],
    )

    assert evidence == {
        "maintenance_state": "idle",
        "next_due_at": None,
        "more": False,
        "idle_blocking": False,
    }
    assert observed == [(runtime.project_paths(binding.project_id)["root"], runtime.root, 1)]
    assert observed[0][0] != cfg.runtime_root
    executor.shutdown()


def test_maintenance_descriptor_worker_marks_possible_write_only_after_immediate_fence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, _bindings = _registered(tmp_path, "project", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    request = executor.prepare_maintenance_descriptor_advance(binding, revision)
    write_possible = [False]

    def fail_after_fence(worker_cfg, **_kwargs):
        check_mutation_fence(worker_cfg.shared_root / "operations")
        raise OSError("synthetic descriptor write failure")

    monkeypatch.setattr(maintenance_module, "advance_maintenance_work", fail_after_fence)
    with pytest.raises(OSError, match="synthetic descriptor write failure"):
        project_io_worker_module._maintenance_descriptor_advance(
            request,
            runtime.root,
            executor.paths,
            write_possible,
        )
    assert write_possible == [True]
    executor.shutdown()


def test_maintenance_descriptor_prewrite_binding_failure_is_retryable(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, _bindings = _registered(tmp_path, "project", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    request = executor.prepare_maintenance_descriptor_advance(binding, revision)
    runtime.set_enabled(cfg.shared_root, False)
    assert executor.start(request.request_id) is not None

    result = None
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        executor.poll()
        result = executor.load_result(request.request_id)
        if result is not None:
            break
        time.sleep(0.02)
    assert result is not None
    assert result.status == "retryable_error"
    assert result.reason_code == "project_io_maintenance_descriptor_advance_failed"
    executor.shutdown()


@pytest.mark.parametrize("proof_change", ["unchanged", "wake", "lost_turn"])
def test_maintenance_descriptor_acknowledges_only_its_issued_activation_turn(tmp_path: Path, proof_change: str) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, bindings = _registered(tmp_path, "project", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    runtime.working_set.reconcile(bindings, revision=revision)
    state = runtime.working_set._states[runtime.working_set._identity(binding)]
    try:
        _advance_until_validated(controller, bindings, revision, binding.project_id)
        _observe_working_set_activation(runtime, controller, bindings, revision)

        controller.advance_maintenance_descriptor_work(bindings, revision)
        request = next(
            item for item in executor.unresolved_requests() if item.operation_kind == "maintenance_descriptor_advance"
        )
        issued_turn = controller._maintenance_descriptor_turns[request.request_id]
        assert not issued_turn.unknown
        assert "maintenance" not in state.acknowledged_lanes
        if proof_change == "wake":
            runtime.working_set.activate(binding, "new_local_obligation")
        elif proof_change == "lost_turn":
            controller._maintenance_descriptor_turns.clear()

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            completions = controller.advance_maintenance_descriptor_work(bindings, revision)
            if binding.project_id in completions:
                break
            time.sleep(0.02)
        else:
            raise AssertionError("descriptor idle confirmation did not complete")
        assert completions[binding.project_id]["maintenance_state"] == "idle"
        assert ("maintenance" in state.acknowledged_lanes) is (proof_change == "unchanged")
        assert bool(controller._maintenance_descriptor_settled) is (proof_change == "unchanged")
        assert request.request_id not in controller._maintenance_descriptor_turns
        if proof_change != "unchanged":
            controller.advance_maintenance_descriptor_work(bindings, revision)
            assert any(
                item.operation_kind == "maintenance_descriptor_advance" for item in executor.unresolved_requests()
            )
        else:
            for lane in SERVICE_LANES:
                if lane != "maintenance":
                    turn = runtime.working_set.begin_turn(binding, lane)
                    runtime.working_set.acknowledge(turn, quiescent=True)
            # Waiting for final activation confirmation must not invalidate
            # a captured proof or continuously restart maintenance workers.
            assert runtime.working_set.is_current_turn(issued_turn)
            assert controller.advance_maintenance_descriptor_work(bindings, revision) == {}
            assert not executor.unresolved_requests()
            runtime.working_set.activate(binding, "new_local_obligation")
            controller.advance_maintenance_descriptor_work(bindings, revision)
            assert not controller._maintenance_descriptor_settled
            assert any(
                item.operation_kind == "maintenance_descriptor_advance" for item in executor.unresolved_requests()
            )
    finally:
        executor.shutdown()


@pytest.mark.parametrize("invalidation", ["registry", "epoch"])
def test_scheduler_subset_preserves_other_service_backoff_until_identity_changes(
    tmp_path: Path, invalidation: str
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, deferred, _revision, _bindings = _registered(tmp_path, "deferred", runtime)
    _cfg, healthy, revision, _bindings = _registered(tmp_path, "healthy", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor, monotonic=lambda: 10.0)
    try:
        epoch = executor.status_view()["executor_epoch"]
        identity = controller._binding_identity(deferred, revision, epoch)
        assert identity is not None
        keys = [
            (identity, operation, "default", "")
            for operation in ("recovery_admission", "maintenance_descriptor_advance", "observation_service")
        ]
        for key in keys:
            controller._record_service_failure(key)
        retained = dict(controller._service_backoff)
        controller.advance_scheduler_observations([healthy], revision, lane="gpu", admission_role="primary")
        controller.advance_scheduler_claims(
            [healthy],
            revision,
            lane="gpu",
            admission_role="primary",
            observations={},
            available_gpu_ids=[],
            available_cpu_slots=0,
        )
        assert controller._service_backoff == retained
        assert all(not controller._service_retry_is_due(key) for key in keys)

        if invalidation == "registry":
            _cfg, _binding, revision, _bindings = _registered(tmp_path, "new-peer", runtime)
        else:
            executor.fence_epoch()
            executor.begin_epoch()
        controller.advance_scheduler_observations([healthy], revision, lane="gpu", admission_role="primary")
        assert not controller._service_backoff
    finally:
        executor.shutdown()


def test_maintenance_idle_proof_waits_for_activation_without_restarting_workers(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, bindings = _registered(tmp_path, "project", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    runtime.working_set.reconcile(bindings, revision=revision)
    state = runtime.working_set._states[runtime.working_set._identity(binding)]
    try:
        _advance_until_validated(controller, bindings, revision, binding.project_id)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            completions = controller.advance_maintenance_descriptor_work(bindings, revision)
            if binding.project_id in completions:
                break
            time.sleep(0.02)
        else:
            raise AssertionError("initial idle descriptor proof did not complete")
        assert completions[binding.project_id]["maintenance_state"] == "idle"
        provisional_turn = next(iter(controller._maintenance_descriptor_settled.values()))
        assert provisional_turn.unknown
        assert "maintenance" not in state.acknowledged_lanes
        for _ in range(3):
            assert controller.advance_maintenance_descriptor_work(bindings, revision) == {}
            assert not executor.unresolved_requests()

        _observe_working_set_activation(runtime, controller, bindings, revision)
        assert not runtime.working_set.is_turn_observation_pending(provisional_turn)
        controller.advance_maintenance_descriptor_work(bindings, revision)
        assert not controller._maintenance_descriptor_settled
        request = next(
            item for item in executor.unresolved_requests() if item.operation_kind == "maintenance_descriptor_advance"
        )
        assert not controller._maintenance_descriptor_turns[request.request_id].unknown
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            controller.advance_maintenance_descriptor_work(bindings, revision)
            if "maintenance" in state.acknowledged_lanes:
                break
            time.sleep(0.02)
        else:
            raise AssertionError("fresh activation-bound maintenance proof did not complete")
    finally:
        executor.shutdown()


def test_maintenance_descriptor_controller_advances_existing_outbox_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project", runtime)
    descriptor = _activate_missing_deadline_descriptor(cfg)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    acknowledgements = []
    acknowledge = runtime.working_set.acknowledge

    def record_acknowledgement(turn, *, quiescent):
        acknowledgements.append((turn.lane, quiescent))
        return acknowledge(turn, quiescent=quiescent)

    monkeypatch.setattr(runtime.working_set, "acknowledge", record_acknowledgement)

    completions = {}
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        completions = controller.advance_maintenance_descriptor_work(bindings, revision)
        if completions.get(binding.project_id, {}).get("maintenance_state") == "completed":
            break
        time.sleep(0.02)

    assert completions[binding.project_id] == {
        "maintenance_state": "completed",
        "next_due_at": None,
        "more": True,
        "idle_blocking": False,
    }
    assert acknowledgements == [("maintenance", False)]
    persisted = maintenance_outbox_module.read_work(
        cfg,
        kind="deadline_index",
        target_id=descriptor["identity"]["target_id"],
        work_generation=descriptor["identity"]["work_generation"],
    )
    assert persisted is not None
    assert persisted["state"] == "completed"
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        completions = controller.advance_maintenance_descriptor_work(bindings, revision)
        if completions.get(binding.project_id, {}).get("maintenance_state") == "idle":
            break
        time.sleep(0.02)
    assert completions[binding.project_id]["more"] is False
    assert acknowledgements[-1] == ("maintenance", True)
    executor.shutdown()


def test_maintenance_descriptor_terminal_slice_confirms_all_activated_work(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project", runtime)
    first = _activate_missing_deadline_descriptor(cfg, suffix="first")
    second = _activate_missing_deadline_descriptor(cfg, suffix="second")
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)

    states: list[str] = []
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        completions = controller.advance_maintenance_descriptor_work(bindings, revision)
        evidence = completions.get(binding.project_id)
        if evidence is not None:
            states.append(evidence["maintenance_state"])
        persisted = [
            maintenance_outbox_module.read_work(
                cfg,
                kind="deadline_index",
                target_id=descriptor["identity"]["target_id"],
                work_generation=descriptor["identity"]["work_generation"],
            )
            for descriptor in (first, second)
        ]
        if all(record is not None and record["state"] == "completed" for record in persisted) and states[-1:] == [
            "idle"
        ]:
            break
        time.sleep(0.02)
    else:
        raise AssertionError("all activated descriptors were not drained before idle settlement")

    assert states.count("completed") == 2
    assert states[-1] == "idle"
    executor.shutdown()


def test_maintenance_descriptor_producer_handoff_wait_does_not_inhibit_idle(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project", runtime)
    prepared = maintenance_outbox_module.prepare_work(
        cfg,
        kind="availability",
        target_id="producer-handoff",
        work_generation="producer-handoff",
    )
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        completions = controller.advance_maintenance_descriptor_work(bindings, revision)
        if completions.get(binding.project_id, {}).get("maintenance_state") == "waiting":
            break
        time.sleep(0.02)
    else:
        raise AssertionError("waiting descriptor state was not observed")
    first = maintenance_outbox_module.read_work(
        cfg,
        kind="availability",
        target_id=prepared["identity"]["target_id"],
        work_generation=prepared["identity"]["work_generation"],
    )
    assert first is not None
    assert binding.project_id in runtime.maintenance_retry_deadlines
    assert binding.project_id not in runtime.maintenance_retry_idle_blocked_projects

    _retry_binding, retry_at = runtime.maintenance_retry_deadlines[binding.project_id]
    while time.monotonic() < retry_at:
        time.sleep(0.02)
    dispatch_loop._activate_due_maintenance_retries(runtime, bindings)
    assert binding.project_id not in runtime.maintenance_retry_deadlines
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        completions = controller.advance_maintenance_descriptor_work(bindings, revision)
        if completions.get(binding.project_id, {}).get("maintenance_state") == "waiting":
            break
        time.sleep(0.02)
    else:
        raise AssertionError("due waiting descriptor was not retried")
    second = maintenance_outbox_module.read_work(
        cfg,
        kind="availability",
        target_id=prepared["identity"]["target_id"],
        work_generation=prepared["identity"]["work_generation"],
    )
    assert second is not None
    assert second["progress_revision"] > first["progress_revision"]
    executor.shutdown()


def test_maintenance_descriptor_machine_timer_inhibits_idle_and_retries(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project", runtime)
    descriptor = _activate_missing_deadline_descriptor(cfg, suffix="future-due")
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    due_at = (datetime.now(timezone.utc) + timedelta(seconds=0.5)).isoformat()
    maintenance_outbox_module.update_work(
        cfg,
        kind="deadline_index",
        target_id=descriptor["identity"]["target_id"],
        work_generation=descriptor["identity"]["work_generation"],
        state="waiting",
        due_at=due_at,
        publish_activation=False,
    )

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        completions = controller.advance_maintenance_descriptor_work(bindings, revision)
        if completions.get(binding.project_id, {}).get("maintenance_state") == "waiting":
            break
        time.sleep(0.02)
    else:
        raise AssertionError("future descriptor wait was not observed")

    assert binding.project_id in runtime.maintenance_retry_deadlines
    assert binding.project_id in runtime.maintenance_retry_idle_blocked_projects
    assert not _machine_is_true_idle(runtime, has_consumed_binding=True)
    _retry_binding, retry_at = runtime.maintenance_retry_deadlines[binding.project_id]
    while time.monotonic() < retry_at:
        time.sleep(0.01)
    dispatch_loop._activate_due_maintenance_retries(runtime, bindings)
    assert binding.project_id not in runtime.maintenance_retry_deadlines
    assert binding.project_id not in runtime.maintenance_retry_idle_blocked_projects

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        completions = controller.advance_maintenance_descriptor_work(bindings, revision)
        if completions.get(binding.project_id, {}).get("maintenance_state") == "completed":
            break
        time.sleep(0.02)
    else:
        raise AssertionError("due descriptor was not resumed")
    executor.shutdown()


@pytest.mark.parametrize("replacement", ["disabled", "generation"])
def test_maintenance_retry_timer_is_pruned_from_local_registry_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, _revision, _bindings = _registered(tmp_path, "project", runtime)
    runtime.maintenance_retry_deadlines[binding.project_id] = (binding, time.monotonic() + 3600.0)
    runtime.maintenance_retry_idle_blocked_projects.add(binding.project_id)

    current = (
        replace(binding, enabled=False)
        if replacement == "disabled"
        else replace(binding, registration_generation="b" * 32)
    )

    def reject_shared_binding_probe(_binding):
        raise AssertionError("maintenance retry cleanup performed synchronous Project I/O")

    monkeypatch.setattr(runtime, "binding_state", reject_shared_binding_probe)
    dispatch_loop._activate_due_maintenance_retries(runtime, [current])

    assert binding.project_id not in runtime.maintenance_retry_deadlines
    assert binding.project_id not in runtime.maintenance_retry_idle_blocked_projects


def test_maintenance_descriptor_rotation_reaches_binding_outside_first_sixty_four(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, _bindings = _registered(tmp_path, "project", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor, monotonic=lambda: 100.0)
    executor_epoch = executor.status_view()["executor_epoch"]
    assert isinstance(executor_epoch, str)
    bindings = [replace(binding, project_id=f"project-{index:03d}") for index in range(65)]
    identities = []
    for item in bindings:
        identity = controller._binding_identity(item, revision, executor_epoch)
        assert identity is not None
        controller._validated.add(identity)
        identities.append(identity)
    for identity in identities[:64]:
        controller._record_service_failure(controller._maintenance_descriptor_service_key(identity))
    selected: list[str] = []

    def record_offer(identity, _kind, _prepare, *_key):
        selected.append(identity.project_id)

    monkeypatch.setattr(controller, "_offer_new_request", record_offer)
    controller.advance_maintenance_descriptor_work(bindings, revision)
    assert selected == []
    controller.advance_maintenance_descriptor_work(bindings, revision)
    assert selected == [bindings[-1].project_id]
    executor.shutdown()


@pytest.mark.parametrize("replacement", ["disabled", "generation"])
def test_maintenance_descriptor_controller_resolves_stale_unstarted_request(
    tmp_path: Path,
    replacement: str,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, bindings = _registered(tmp_path, "project", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    request = executor.prepare_maintenance_descriptor_advance(binding, revision)
    current = (
        replace(binding, enabled=False)
        if replacement == "disabled"
        else replace(binding, registration_generation="b" * 32)
    )

    controller.advance_maintenance_descriptor_work([current], revision)

    assert all(item.request_id != request.request_id for item in executor.unresolved_requests())
    assert all(item.operation_kind != "maintenance_descriptor_advance" for item in executor.unresolved_requests())
    executor.shutdown()


def test_maintenance_descriptor_idle_settlement_reopens_after_activation(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    _observe_working_set_activation(runtime, controller, bindings, revision)

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        completions = controller.advance_maintenance_descriptor_work(bindings, revision)
        if completions.get(binding.project_id, {}).get("maintenance_state") == "idle":
            break
        time.sleep(0.02)
    else:
        raise AssertionError("idle descriptor state was not observed")

    for _ in range(3):
        assert controller.advance_maintenance_descriptor_work(bindings, revision) == {}
    assert all(request.operation_kind != "maintenance_descriptor_advance" for request in executor.unresolved_requests())

    executor_epoch = executor.status_view()["executor_epoch"]
    assert isinstance(executor_epoch, str)
    identity = controller._binding_identity(binding, revision, executor_epoch)
    assert identity is not None
    service_key = controller._maintenance_descriptor_service_key(identity)
    controller._record_service_failure(service_key)
    assert not controller._service_retry_is_due(service_key)
    descriptor = _activate_missing_deadline_descriptor(cfg, suffix="reactivated")
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        activation = controller.advance_activation_observations(
            bindings,
            revision,
            {binding.project_id: {"epoch": None, "sequence": 0}},
        )
        if binding.project_id in activation:
            break
        time.sleep(0.02)
    else:
        raise AssertionError(
            "descriptor activation was not observed: "
            f"status={executor.status_view()!r}, unresolved={executor.unresolved_requests()!r}, "
            f"backoff={controller._service_backoff!r}"
        )
    assert controller._service_retry_is_due(service_key)

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        completions = controller.advance_maintenance_descriptor_work(bindings, revision)
        if completions.get(binding.project_id, {}).get("maintenance_state") == "completed":
            break
        time.sleep(0.02)
    else:
        raise AssertionError("reactivated descriptor did not complete")
    persisted = maintenance_outbox_module.read_work(
        cfg,
        kind="deadline_index",
        target_id=descriptor["identity"]["target_id"],
        work_generation=descriptor["identity"]["work_generation"],
    )
    assert persisted is not None
    assert persisted["state"] == "completed"
    executor.shutdown()


def test_isolated_dispatch_advances_descriptor_without_synchronous_project_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, _revision, _bindings = _registered(tmp_path, "project", runtime)
    descriptor = _activate_missing_deadline_descriptor(cfg, suffix="dispatch")
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    runtime.project_io_executor = executor
    runtime.project_io_controller = controller
    runtime.authority_ready_generations = {binding.project_id: binding.registration_generation}

    def reject_synchronous_maintenance(*_args, **_kwargs):
        raise AssertionError("isolated dispatch used synchronous descriptor maintenance")

    monkeypatch.setattr(maintenance_module, "advance_maintenance_work", reject_synchronous_maintenance)
    try:
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            dispatch_machine_cycle_locked(
                runtime,
                available_gpus=[],
                supervise=True,
                publish_snapshots=False,
            )
            persisted = maintenance_outbox_module.read_work(
                cfg,
                kind="deadline_index",
                target_id=descriptor["identity"]["target_id"],
                work_generation=descriptor["identity"]["work_generation"],
            )
            if persisted is not None and persisted["state"] == "completed":
                break
            time.sleep(0.02)
        assert persisted is not None
        assert persisted["state"] == "completed"
    finally:
        runtime.attempt_supervision_coordinator.close()
        executor.shutdown()


@pytest.mark.parametrize("replacement", ["disabled", "generation"])
def test_ready_index_controller_resolves_stale_unstarted_request(
    tmp_path: Path,
    replacement: str,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, bindings = _registered(tmp_path, "project", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    request = executor.prepare_scheduler_ready_index_build(binding, revision)
    current = (
        replace(binding, enabled=False)
        if replacement == "disabled"
        else replace(binding, registration_generation="b" * 32)
    )

    controller.advance_scheduler_ready_index_builds([current], revision)

    assert all(item.request_id != request.request_id for item in executor.unresolved_requests())
    assert all(item.operation_kind != "scheduler_ready_index_build" for item in executor.unresolved_requests())
    executor.shutdown()


def test_isolated_dispatch_recovers_pre_epoch_starting_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, _revision, _bindings = _registered(tmp_path, "project", runtime)
    task = submit(cfg, ["echo", "recover"], working_dir=tmp_path)
    attempt = claim_task(
        cfg,
        task.task_id,
        [0],
        reservation_runtime_root=runtime.root,
        project_id=binding.project_id,
    )
    assert attempt is not None
    reservation = active_reservations(runtime.root)[0]
    assert reservation.get("executor_owner") is None

    project_io_executor = ProjectIOExecutor(runtime)
    project_io_executor.begin_epoch()
    runtime.project_io_executor = project_io_executor
    launched: list[dict[str, object]] = []

    class TicketExecutor:
        def initiate_authorized_attempt(self, launch_cfg, **ticket):
            launched.append({"cfg": launch_cfg, **ticket})
            attempt_id = ticket["claim_identity"]["attempt_id"]
            intent = launch_cfg.runtime_root / "launch-intents" / f"{attempt_id}.json"
            intent.parent.mkdir(parents=True, exist_ok=True)
            intent.touch()
            process = type("Process", (), {"pid": 4321})()
            handle = LaunchHandle("detached", process, runner_process=process)
            return handle, LaunchHandoff(attempt_id, intent, time.monotonic() + 60, handle)

    def reject_legacy(*_args, **_kwargs):
        raise AssertionError("isolated dispatch used legacy reservation/starting recovery")

    monkeypatch.setattr(dispatch_loop, "_reconcile_machine_reservations", reject_legacy)
    monkeypatch.setattr(dispatch_loop, "_recover_starting_reservations", reject_legacy)
    try:
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline and not launched:
            dispatch_machine_cycle_locked(
                runtime,
                available_gpus=[0],
                executor=TicketExecutor(),
                supervise=False,
                publish_snapshots=False,
            )
            time.sleep(0.02)
        assert len(launched) == 1
        assert launched[0]["claim_identity"]["attempt_id"] == attempt.attempt_id
        assert launched[0]["cfg"].shared_root == cfg.shared_root
    finally:
        runtime.attempt_supervision_coordinator.close()
        project_io_executor.shutdown()


def test_isolated_dispatch_releases_exact_missing_task_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, _revision, _bindings = _registered(tmp_path, "project", runtime)
    value = reserve(
        runtime.root,
        "missing-task",
        [0],
        attempt_id="missing-task-attempt-1",
        fencing_token=1,
        project_id=binding.project_id,
        shared_root=str(cfg.shared_root),
        machine_name=binding.machine_name,
    )
    reservation_id = value["reservation"]["reservation_id"]
    attach(runtime.root, reservation_id, "missing-task-attempt-1", 1)
    project_io_executor = ProjectIOExecutor(runtime)
    project_io_executor.begin_epoch()
    runtime.project_io_executor = project_io_executor

    def reject_legacy(*_args, **_kwargs):
        raise AssertionError("isolated dispatch used legacy reservation reconciliation")

    monkeypatch.setattr(dispatch_loop, "_reconcile_machine_reservations", reject_legacy)
    try:
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline and active_reservations(runtime.root):
            dispatch_machine_cycle_locked(
                runtime,
                available_gpus=[0],
                supervise=False,
                publish_snapshots=False,
            )
            time.sleep(0.02)
        assert active_reservations(runtime.root) == []
        released = read_json(local_paths(runtime.root)["released"] / f"{reservation_id}.json")["reservation"]
        assert released["reservation_id"] == reservation_id
        assert released["release_reason"] == "task_missing"
    finally:
        runtime.attempt_supervision_coordinator.close()
        project_io_executor.shutdown()


def test_isolated_dispatch_retags_exact_stale_reservation_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, _revision, _bindings = _registered(tmp_path, "project", runtime)
    task = submit(cfg, ["echo", "retag"], working_dir=tmp_path)
    attempt = claim_task(
        cfg,
        task.task_id,
        [0],
        reservation_runtime_root=runtime.root,
        project_id=binding.project_id,
    )
    assert attempt is not None
    original = ReservationIdentity.from_record(active_reservations(runtime.root)[0])
    shared_task = load_task(cfg, task.task_id)
    shared_task.claim_control["fencing_epoch"] = 2
    shared_task.claim_control["active_claim"]["fencing_token"] = 2
    shared_task.meta["revision"] += 1
    save_task(cfg, shared_task)
    attempt_file = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    shared_attempt = AttemptRecord.from_dict(read_json(attempt_file))
    shared_attempt.current_fencing_token = 2
    shared_attempt.token_history.append(2)
    atomic_replace(attempt_file, shared_attempt.to_dict())

    project_io_executor = ProjectIOExecutor(runtime)
    project_io_executor.begin_epoch()
    runtime.project_io_executor = project_io_executor

    def reject_legacy(*_args, **_kwargs):
        raise AssertionError("isolated dispatch used legacy reservation reconciliation")

    monkeypatch.setattr(dispatch_loop, "_reconcile_machine_reservations", reject_legacy)
    try:
        deadline = time.monotonic() + 8.0
        current = original
        while time.monotonic() < deadline and current.fencing_token != 2:
            dispatch_machine_cycle_locked(
                runtime,
                available_gpus=[0],
                supervise=False,
                publish_snapshots=False,
            )
            current = ReservationIdentity.from_record(active_reservations(runtime.root)[0])
            time.sleep(0.02)
        assert current == replace(original, fencing_token=2)
    finally:
        runtime.attempt_supervision_coordinator.close()
        project_io_executor.shutdown()


def test_isolated_dispatch_releases_exact_missing_task_cpu_reservation(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, _revision, _bindings = _registered(tmp_path, "project", runtime)
    set_cpu_lane_capacity(runtime.root, capacity=2)
    value = reserve_cpu(
        runtime.root,
        "missing-cpu-task",
        1,
        attempt_id="missing-cpu-task-attempt-1",
        fencing_token=1,
        project_id=binding.project_id,
        shared_root=str(cfg.shared_root),
        machine_name=binding.machine_name,
    )
    reservation_id = value["reservation"]["reservation_id"]
    attach_cpu(runtime.root, reservation_id, "missing-cpu-task-attempt-1", 1)
    project_io_executor = ProjectIOExecutor(runtime)
    project_io_executor.begin_epoch()
    runtime.project_io_executor = project_io_executor
    try:
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline and cpu_reservation_snapshot(runtime.root)[1]:
            dispatch_machine_cycle_locked(
                runtime,
                available_gpus=[],
                supervise=False,
                publish_snapshots=False,
            )
            time.sleep(0.02)
        assert cpu_reservation_snapshot(runtime.root)[1] == ()
        released = read_json(local_paths(runtime.root)["cpu_released"] / f"{reservation_id}.json")["reservation"]
        assert released["release_reason"] == "task_missing"
    finally:
        runtime.attempt_supervision_coordinator.close()
        project_io_executor.shutdown()


def test_reservation_reconcile_result_cannot_release_replaced_local_owner(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project", runtime)
    value = reserve(
        runtime.root,
        "missing-task",
        [0],
        attempt_id="missing-task-attempt-1",
        fencing_token=1,
        project_id=binding.project_id,
        shared_root=str(cfg.shared_root),
        machine_name=binding.machine_name,
    )
    reservation_id = value["reservation"]["reservation_id"]
    attach(runtime.root, reservation_id, "missing-task-attempt-1", 1)
    executor = ProjectIOExecutor(runtime)
    executor_epoch = executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    original = ReservationIdentity.from_record(active_reservations(runtime.root)[0])

    controller.advance_scheduler_reservation_reconciliations(
        bindings,
        revision,
        reservations=[original],
    )
    request = next(
        item for item in executor.unresolved_requests() if item.operation_kind == "scheduler_reservation_reconcile"
    )
    deadline = time.monotonic() + 5.0
    while executor.load_result(request.request_id) is None and time.monotonic() < deadline:
        executor.poll()
        time.sleep(0.02)
    assert executor.load_result(request.request_id) is not None

    active_path = local_paths(runtime.root)["active"] / f"{reservation_id}.json"
    active_value = read_json(active_path)
    active_value["reservation"]["executor_owner"] = {
        "executor_epoch": executor_epoch,
        "request_id": "f" * 32,
        "registration_generation": binding.registration_generation,
    }
    atomic_replace(active_path, active_value)
    replacement = ReservationIdentity.from_record(active_value["reservation"])
    assert replacement != original

    controller.advance_scheduler_reservation_reconciliations(
        bindings,
        revision,
        reservations=[replacement],
    )
    assert ReservationIdentity.from_record(active_reservations(runtime.root)[0]) == replacement
    assert not (local_paths(runtime.root)["released"] / f"{reservation_id}.json").exists()
    executor.shutdown()


def test_reservation_reconcile_finishes_exact_interrupted_release_pair(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project", runtime)
    value = reserve(
        runtime.root,
        "missing-task",
        [0],
        attempt_id="missing-task-attempt-1",
        fencing_token=1,
        project_id=binding.project_id,
        shared_root=str(cfg.shared_root),
        machine_name=binding.machine_name,
    )
    reservation_id = value["reservation"]["reservation_id"]
    attach(runtime.root, reservation_id, "missing-task-attempt-1", 1)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    identity = ReservationIdentity.from_record(active_reservations(runtime.root)[0])

    controller.advance_scheduler_reservation_reconciliations(bindings, revision, reservations=[identity])
    request = next(
        item for item in executor.unresolved_requests() if item.operation_kind == "scheduler_reservation_reconcile"
    )
    deadline = time.monotonic() + 5.0
    while executor.load_result(request.request_id) is None and time.monotonic() < deadline:
        executor.poll()
        time.sleep(0.02)
    assert executor.load_result(request.request_id) is not None

    paths = local_paths(runtime.root)
    active_path = paths["active"] / f"{reservation_id}.json"
    released_path = paths["released"] / active_path.name
    released_value = read_json(active_path)
    released_value["reservation"].update(
        state="released",
        released_at="2026-09-29T00:00:00Z",
        release_reason="task_missing",
    )
    atomic_replace(released_path, released_value)

    deadline = time.monotonic() + 5.0
    while active_reservations(runtime.root) and time.monotonic() < deadline:
        controller.advance_scheduler_reservation_reconciliations(bindings, revision, reservations=[identity])
        time.sleep(0.02)
    assert active_reservations(runtime.root) == []
    released = read_json(released_path)["reservation"]
    assert ReservationIdentity.from_record(released) == identity
    assert released["release_reason"] == "task_missing"
    executor.shutdown()


def test_reservation_reconcile_rotates_past_isolated_first_record(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project", runtime)
    blocked_task = submit(cfg, ["echo", "blocked"], working_dir=tmp_path)
    stored = load_task(cfg, blocked_task.task_id)
    stored.state["projection"] = "blocked"
    stored.claim_control["active_claim"] = None
    save_task(cfg, stored)
    values = [
        reserve(
            runtime.root,
            f"placeholder-{index}",
            [index],
            attempt_id=f"placeholder-{index}-attempt-1",
            fencing_token=1,
            project_id=binding.project_id,
            shared_root=str(cfg.shared_root),
            machine_name=binding.machine_name,
        )
        for index in range(2)
    ]
    for index, value in enumerate(values):
        attach(
            runtime.root,
            value["reservation"]["reservation_id"],
            f"placeholder-{index}-attempt-1",
            1,
        )
    reservation_ids = sorted(value["reservation"]["reservation_id"] for value in values)
    isolated_id, releasable_id = reservation_ids
    isolated_path = local_paths(runtime.root)["active"] / f"{isolated_id}.json"
    isolated_value = read_json(isolated_path)
    isolated_value["reservation"].update(
        task_id=blocked_task.task_id,
        attempt_id=f"{blocked_task.task_id}-attempt-1",
    )
    atomic_replace(isolated_path, isolated_value)

    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        identities = [ReservationIdentity.from_record(item) for item in active_reservations(runtime.root)]
        controller.advance_scheduler_reservation_reconciliations(bindings, revision, reservations=identities)
        if {item.reservation_id for item in identities} == {isolated_id}:
            break
        time.sleep(0.02)
    remaining = active_reservations(runtime.root)
    assert {item["reservation_id"] for item in remaining} == {isolated_id}
    released = read_json(local_paths(runtime.root)["released"] / f"{releasable_id}.json")["reservation"]
    assert released["release_reason"] == "task_missing"
    executor.shutdown()


def test_disabled_current_binding_reconciles_old_generation_reservation(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)

    value = reserve(
        runtime.root,
        "missing-old-generation-task",
        [0],
        attempt_id="missing-old-generation-task-attempt-1",
        fencing_token=1,
        project_id=binding.project_id,
        shared_root=str(cfg.shared_root),
        machine_name=binding.machine_name,
        executor_epoch="a" * 32,
        executor_request_id="b" * 32,
        registration_generation="old-generation",
    )
    reservation_id = value["reservation"]["reservation_id"]
    offer_identity = ReservationIdentity.from_record(value["reservation"])
    assert attach_executor_offer(runtime.root, offer_identity, "missing-old-generation-task-attempt-1", 1)
    disabled = replace(binding, enabled=False)
    deadline = time.monotonic() + 8.0
    while active_reservations(runtime.root) and time.monotonic() < deadline:
        identities = [ReservationIdentity.from_record(item) for item in active_reservations(runtime.root)]
        controller.advance_scheduler_reservation_reconciliations(
            [disabled],
            revision,
            reservations=identities,
        )
        time.sleep(0.02)
    assert active_reservations(runtime.root) == []
    released = read_json(local_paths(runtime.root)["released"] / f"{reservation_id}.json")["reservation"]
    assert released["release_reason"] == "task_missing"
    executor.shutdown()


def test_disabled_reservation_reconcile_retry_keeps_independent_backoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    now = [100.0]
    controller = ProjectIOController(runtime, executor, monotonic=lambda: now[0])
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    value = reserve(
        runtime.root,
        "missing-disabled-task",
        [0],
        attempt_id="missing-disabled-task-attempt-1",
        fencing_token=1,
        project_id=binding.project_id,
        shared_root=str(cfg.shared_root),
        machine_name=binding.machine_name,
    )
    identity = ReservationIdentity.from_record(value["reservation"])
    attach(runtime.root, identity.reservation_id, identity.attempt_id, identity.fencing_token)
    disabled = replace(binding, enabled=False)
    start_calls: list[str] = []

    def fail_start(request_id: str):
        start_calls.append(request_id)
        return None

    monkeypatch.setattr(executor, "start", fail_start)
    controller.advance_scheduler_reservation_reconciliations(
        [disabled],
        revision,
        reservations=[identity],
    )
    assert len(start_calls) == 1

    controller.advance_scheduler_observations(
        [disabled],
        revision,
        lane="gpu",
        admission_role="primary",
    )
    controller.advance_scheduler_reservation_reconciliations(
        [disabled],
        revision,
        reservations=[identity],
    )
    assert len(start_calls) == 1

    now[0] += 5.0
    controller.advance_scheduler_reservation_reconciliations(
        [disabled],
        revision,
        reservations=[identity],
    )
    assert len(start_calls) == 2
    executor.shutdown()


def test_reservation_reconcile_rotates_bounded_binding_slice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, _bindings = _registered(tmp_path, "project", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    executor_epoch = executor.status_view()["executor_epoch"]
    assert isinstance(executor_epoch, str)
    bindings = [replace(binding, project_id=f"project-{index:03d}") for index in range(65)]
    for item in bindings:
        identity = controller._binding_identity(item, revision, executor_epoch)
        assert identity is not None
        controller._validated.add(identity)
    reservations = [
        ReservationIdentity(
            reservation_id=f"{index + 1:032x}",
            acquisition_id=f"{index + 101:032x}",
            project_id=item.project_id,
            task_id=f"task-{index:03d}",
            attempt_id=f"task-{index:03d}-attempt-1",
            fencing_token=1,
            gpu_ids=(index,),
            cpu_slots=None,
            shared_root=str(cfg.shared_root),
            registration_generation=item.registration_generation,
            executor_epoch=executor_epoch,
            executor_request_id=f"{index + 201:032x}",
        )
        for index, item in enumerate(bindings)
    ]
    selected: list[str] = []

    def record_offer(identity, _kind, _prepare, *_key):
        selected.append(identity.project_id)

    monkeypatch.setattr(controller, "_offer_new_request", record_offer)
    controller.advance_scheduler_reservation_reconciliations(
        bindings,
        revision,
        reservations=reservations,
    )
    first = tuple(selected)
    assert len(first) == 64
    assert bindings[-1].project_id not in first

    selected.clear()
    controller.advance_scheduler_reservation_reconciliations(
        bindings,
        revision,
        reservations=reservations,
    )
    assert len(selected) == 64
    assert bindings[-1].project_id in selected
    executor.shutdown()


def test_isolated_dispatch_reconciles_disabled_binding_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, _revision, _bindings = _registered(tmp_path, "project", runtime)
    task = submit(cfg, ["echo", "disabled"], working_dir=tmp_path)
    attempt = claim_task(
        cfg,
        task.task_id,
        [0],
        reservation_runtime_root=runtime.root,
        project_id=binding.project_id,
    )
    assert attempt is not None
    reservation_id = attempt.reservation_id
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    runtime.project_io_executor = executor
    runtime.set_enabled(cfg.shared_root, False)
    task_path(cfg.shared_root, task.task_id).unlink()

    def reject_legacy(*_args, **_kwargs):
        raise AssertionError("isolated dispatch used legacy reservation reconciliation")

    monkeypatch.setattr(dispatch_loop, "_reconcile_machine_reservations", reject_legacy)
    try:
        deadline = time.monotonic() + 8.0
        while active_reservations(runtime.root) and time.monotonic() < deadline:
            dispatch_machine_cycle_locked(
                runtime,
                available_gpus=[0],
                supervise=False,
                publish_snapshots=False,
            )
            time.sleep(0.02)
        assert active_reservations(runtime.root) == []
        released = read_json(local_paths(runtime.root)["released"] / f"{reservation_id}.json")["reservation"]
        assert released["release_reason"] == "task_missing"
    finally:
        runtime.attempt_supervision_coordinator.close()
        executor.shutdown()


def test_old_registry_revision_validation_cannot_authorize_current_binding(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, old_revision, old_bindings = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    controller.advance_binding_validation(old_bindings, old_revision)

    deadline = time.monotonic() + 5.0
    while executor.status_view()["active_worker_count"] and time.monotonic() < deadline:
        executor.poll()
        time.sleep(0.02)
    _other_cfg, _other, revision, bindings = _registered(tmp_path, "project-b", runtime)

    current = controller.advance_binding_validation(bindings, revision)

    assert binding.project_id not in current
    assert controller.validated_config(binding, revision) is None
    configs = _advance_until_validated(controller, bindings, revision, binding.project_id)
    assert binding.project_id in configs


def test_executor_epoch_rollover_invalidates_cached_validation(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    assert controller.validated_config(binding, revision) is not None

    executor.fence_epoch()
    executor.begin_epoch()
    current = controller.advance_binding_validation(bindings, revision)

    assert current == {}
    assert controller.validated_config(binding, revision) is None
    assert len(executor.unresolved_requests()) == 1
    executor.shutdown()


def test_controller_flushes_and_retires_exact_local_event(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    event_id = "b" * 32
    bucket = "task-a-attempt-1"
    source = local_paths(runtime.project_paths(binding.project_id)["root"])["events"] / bucket / f"{event_id}.json"
    invalid_name = source.parent / "000.json"
    atomic_replace(invalid_name, {"invalid": True})
    atomic_replace(
        source,
        {
            "event_id": event_id,
            "event_type": "test_diagnostic",
            "task_id": "task-a",
            "attempt_id": bucket,
            "machine_name": "gpu-1",
            "timestamp": "2026-09-28T00:00:00+00:00",
            "details": {},
        },
    )
    original_digest = hashlib.sha256(source.read_bytes()).hexdigest()

    assert controller.advance_maintenance_event_flushes(bindings, revision) == {}
    deadline = time.monotonic() + 5.0
    completion = {}
    while time.monotonic() < deadline:
        completion = controller.advance_maintenance_event_flushes(bindings, revision)
        if binding.project_id in completion:
            break
        time.sleep(0.02)

    assert completion[binding.project_id] == {
        "outcome": "flushed",
        "event_id": event_id,
        "sha256": original_digest,
        "reason": None,
    }
    assert not source.exists()
    assert invalid_name.exists()
    assert (shared_paths(cfg.shared_root)["events"] / "2026-09-28" / f"{event_id}.json").exists()


@pytest.mark.parametrize("bad_kind", ["invalid_event", "identity_mismatch"])
def test_controller_retires_bad_event_and_flushes_next_event(
    tmp_path: Path,
    bad_kind: str,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    event_root = local_paths(runtime.project_paths(binding.project_id)["root"])["events"] / "machine"
    bad_event_id = "a" * 32
    bad_source = event_root / f"{bad_event_id}.json"
    if bad_kind == "invalid_event":
        bad_source.parent.mkdir(parents=True, exist_ok=True)
        bad_source.write_bytes(b"{")
    else:
        atomic_replace(
            bad_source,
            {
                "event_id": "f" * 32,
                "event_type": "wrong_identity",
                "task_id": None,
                "machine_name": "gpu-1",
                "timestamp": "2026-09-28T00:00:00+00:00",
                "details": {},
            },
        )
    good_event_id = "b" * 32
    good_source = event_root / f"{good_event_id}.json"
    atomic_replace(
        good_source,
        {
            "event_id": good_event_id,
            "event_type": "after_bad_event",
            "task_id": None,
            "machine_name": "gpu-1",
            "timestamp": "2026-09-28T00:00:01+00:00",
            "details": {},
        },
    )

    observed_reasons: list[str | None] = []
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        completion = controller.advance_maintenance_event_flushes(bindings, revision)
        if binding.project_id in completion:
            observed_reasons.append(completion[binding.project_id].get("reason"))
        if not bad_source.exists() and not good_source.exists():
            break
        time.sleep(0.02)

    assert bad_kind in observed_reasons
    assert None in observed_reasons
    assert not bad_source.exists()
    assert not good_source.exists()
    assert (shared_paths(cfg.shared_root)["events"] / "2026-09-28" / f"{good_event_id}.json").exists()


def test_controller_backs_off_when_event_retirement_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    now = [100.0]
    controller = ProjectIOController(runtime, executor, monotonic=lambda: now[0])
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    event_id = "c" * 32
    source = local_paths(runtime.project_paths(binding.project_id)["root"])["events"] / "machine" / f"{event_id}.json"
    atomic_replace(
        source,
        {
            "event_id": event_id,
            "event_type": "retirement_failure",
            "task_id": None,
            "machine_name": "gpu-1",
            "timestamp": "2026-09-28T00:00:00+00:00",
            "details": {},
        },
    )
    monkeypatch.setattr(executor, "retire_maintenance_event_source", lambda _request: False)

    controller.advance_maintenance_event_flushes(bindings, revision)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        completion = controller.advance_maintenance_event_flushes(bindings, revision)
        if binding.project_id in completion:
            break
        time.sleep(0.02)

    assert completion[binding.project_id]["outcome"] == "flushed"
    assert source.exists()
    assert executor.unresolved_requests() == ()
    assert controller.advance_maintenance_event_flushes(bindings, revision) == {}
    assert executor.unresolved_requests() == ()

    now[0] += 5.0
    controller.advance_maintenance_event_flushes(bindings, revision)
    assert len(executor.unresolved_requests()) == 1


@pytest.mark.parametrize("has_reservation", [False, True])
@pytest.mark.parametrize("has_warning", [False, True])
def test_controller_publishes_fenced_machine_snapshot(tmp_path: Path, has_reservation: bool, has_warning: bool) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    reservations = (
        [
            {
                "reservation_id": "reservation-1",
                "project_id": binding.project_id,
                "machine_name": binding.machine_name,
                "task_id": "task-1",
                "attempt_id": "task-1-attempt-1",
                "gpu_ids": [0],
                "state": "active",
                "admission": {"admitted_as_borrow": False, "gpu_limit_gpus": None},
            }
        ]
        if has_reservation
        else []
    )
    warnings = [{"code": "test_warning", "details": {"gpu_ids": [1]}}] if has_warning else []
    gpu_policy = {"mode": "all", "warnings": warnings}

    controller.advance_machine_snapshot_publications(
        bindings,
        revision,
        instance_id="agent-1",
        pid=123,
        visible_gpu_ids=[0, 1],
        reservations=reservations,
        heartbeat_interval_seconds=5.0,
        started_at="2026-09-28T00:00:00+00:00",
        gpu_policy=gpu_policy,
    )
    deadline = time.monotonic() + 5.0
    completions = {}
    while time.monotonic() < deadline:
        completions = controller.advance_machine_snapshot_publications(
            bindings,
            revision,
            instance_id="agent-1",
            pid=123,
            visible_gpu_ids=[0, 1],
            reservations=reservations,
            heartbeat_interval_seconds=5.0,
            started_at="2026-09-28T00:00:00+00:00",
            gpu_policy=gpu_policy,
        )
        if binding.project_id in completions:
            break
        time.sleep(0.02)

    assert completions[binding.project_id]["outcome"] == "published"
    agent = read_json(machine_state_path(cfg, "agent.json"))["agent"]
    assert agent["instance_id"] == "agent-1"
    assert agent["observed_state"] == ("active" if has_reservation else "idle")
    assert agent["heartbeat_interval_seconds"] == 5.0
    assert agent["warnings"] == warnings
    summary = read_json(machine_state_path(cfg, "summary.json"))["summary"]
    assert summary["machine_reservation_count"] == len(reservations)
    if has_reservation:
        assert summary["machine_reservations"][0]["admission"] == reservations[0]["admission"]
        assert summary["reserved_gpu_ids"] == [0]


def test_machine_snapshot_renews_registration_under_write_guard(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    envelope = load_machine_registration(cfg)
    registration = dict(envelope["registration"])
    previous_expiry = datetime.now(timezone.utc) + timedelta(seconds=2)
    registration["eligibility_expires_at"] = previous_expiry.isoformat()
    save_machine_registration(cfg, {"registration": registration})

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        completions = controller.advance_machine_snapshot_publications(
            bindings,
            revision,
            instance_id="agent-1",
            pid=123,
            visible_gpu_ids=[],
            reservations=[],
            heartbeat_interval_seconds=5.0,
            started_at="2026-09-28T00:00:00+00:00",
            gpu_policy={"mode": "none", "warnings": []},
        )
        if binding.project_id in completions:
            break
        time.sleep(0.02)

    assert completions[binding.project_id]["outcome"] == "published"
    renewed = load_machine_registration(cfg)["registration"]["eligibility_expires_at"]
    assert datetime.fromisoformat(renewed) > previous_expiry


def test_controller_renews_registration_with_exact_fenced_evidence(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    envelope = load_machine_registration(cfg)
    registration = dict(envelope["registration"])
    previous_expiry = datetime.now(timezone.utc) + timedelta(seconds=2)
    registration["eligibility_expires_at"] = previous_expiry.isoformat()
    save_machine_registration(cfg, {"registration": registration})

    completions = {}
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        completions = controller.advance_registration_renewals(
            bindings,
            revision,
            renewal_horizon_seconds=10.0,
        )
        if binding.project_id in completions:
            break
        time.sleep(0.02)

    evidence = completions[binding.project_id]
    assert evidence["outcome"] == "eligible"
    assert evidence["renewed"] is True
    assert datetime.fromisoformat(evidence["eligibility_expires_at"]) > previous_expiry
    assert (
        load_machine_registration(cfg)["registration"]["eligibility_expires_at"] == evidence["eligibility_expires_at"]
    )
    assert executor.unresolved_requests() == ()


def test_blocked_registration_renewal_does_not_delay_healthy_peer(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    blocked_cfg, blocked, _revision, _bindings = _registered(tmp_path, "blocked", runtime)
    healthy_cfg, healthy, revision, bindings = _registered(tmp_path, "healthy", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, blocked.project_id)
    _advance_until_validated(controller, bindings, revision, healthy.project_id)
    blocked_registration = blocked_cfg.shared_root / "machines" / "gpu-1" / "registration.json"
    original = blocked_registration.read_bytes()
    blocked_registration.unlink()
    os.mkfifo(blocked_registration)

    try:
        completions = {}
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            completions = controller.advance_registration_renewals(
                bindings,
                revision,
                renewal_horizon_seconds=120.0,
            )
            if healthy.project_id in completions:
                break
            time.sleep(0.02)

        assert completions[healthy.project_id]["outcome"] == "eligible"
        assert load_machine_registration(healthy_cfg)["registration"]["generation"] == (healthy.registration_generation)
        assert executor.status_view()["active_worker_count"] == 1
    finally:
        blocked_registration.unlink()
        blocked_registration.write_bytes(original)
        executor.shutdown()


def test_registration_renewal_cadence_avoids_repeated_shared_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, bindings = _registered(tmp_path, "project", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    now = [100.0]
    controller = ProjectIOController(runtime, executor, monotonic=lambda: now[0])
    prepares = []
    original = executor.prepare_registration_renew

    def record_prepare(*args, **kwargs):
        request = original(*args, **kwargs)
        prepares.append(request.request_id)
        return request

    monkeypatch.setattr(executor, "prepare_registration_renew", record_prepare)

    def advance():
        return controller.advance_registration_renewals(bindings, revision, renewal_horizon_seconds=120.0)

    def await_eligible():
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            value = advance().get(binding.project_id)
            if value is not None and value["outcome"] == "eligible":
                return value
            time.sleep(0.02)
        raise AssertionError("registration renewal did not complete")

    try:
        evidence = await_eligible()
        assert evidence["eligibility_expires_at"] is not None
        assert evidence["renew_after_seconds"] == pytest.approx(10.0)
        assert len(prepares) == 1
        for _ in range(20):
            controller.advance_scheduler_observations([], revision, lane="gpu", admission_role="primary")
            assert advance()[binding.project_id] == {"outcome": "not_due"}
        assert len(prepares) == 1
        assert not executor.has_unfinished_work()
        now[0] += 9.9
        assert advance()[binding.project_id] == {"outcome": "not_due"}
        assert len(prepares) == 1
        now[0] += 0.2
        assert await_eligible()["outcome"] == "eligible"
        assert len(prepares) == 2
        executor.begin_epoch()
        assert await_eligible()["outcome"] == "eligible"
        assert len(prepares) == 3, "a replacement epoch must not inherit old service cadence"
    finally:
        executor.shutdown()


def test_registration_renewal_progresses_past_executor_capacity_without_restarting_completed(
    tmp_path: Path,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    for index in range(8):
        _registered(tmp_path, f"project-{index}", runtime)
    revision, bindings = runtime.load_registry()
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    remaining = list(bindings)
    completed = set()
    deadline = time.monotonic() + 10.0

    while remaining and time.monotonic() < deadline:
        cycle = controller.advance_registration_renewals(
            remaining,
            revision,
            renewal_horizon_seconds=120.0,
        )
        assert completed.isdisjoint(cycle)
        completed.update(cycle)
        remaining = [binding for binding in remaining if binding.project_id not in cycle]
        time.sleep(0.02)

    assert completed == {binding.project_id for binding in bindings}
    assert remaining == []
    assert executor.unresolved_requests() == ()


def test_controller_advances_activation_registration_observation_and_ack(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    checkpoint = publish_project_activation(cfg, "work")["project_activation"]
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    runtime.working_set.reconcile(bindings, revision=revision)

    registration = {}
    deadline = time.monotonic() + 5.0
    while binding.project_id not in registration and time.monotonic() < deadline:
        registration = controller.advance_activation_consumer_registrations(
            bindings,
            revision,
            process_fence="process-a",
        )
        time.sleep(0.02)
    assert registration[binding.project_id] == {"outcome": "registered", "acknowledgement": None}
    runtime.working_set.apply_activation_registrations(bindings, registration)

    observation = {}
    deadline = time.monotonic() + 5.0
    while binding.project_id not in observation and time.monotonic() < deadline:
        observation = controller.advance_activation_observations(bindings, revision)
        time.sleep(0.02)
    assert observation[binding.project_id] == {
        "outcome": "observed",
        "checkpoint": {"epoch": checkpoint["epoch"], "sequence": checkpoint["sequence"]},
        "replay": {
            "epoch": checkpoint["epoch"],
            "sequence": checkpoint["sequence"],
            "reconstructed_floor": None,
            "complete": True,
        },
    }

    acknowledgement = {}
    deadline = time.monotonic() + 5.0
    while binding.project_id not in acknowledgement and time.monotonic() < deadline:
        acknowledgement = controller.advance_activation_consumer_acks(
            bindings,
            revision,
            {
                binding.project_id: {
                    "process_fence": "process-a",
                    "epoch": checkpoint["epoch"],
                    "sequence": checkpoint["sequence"],
                    "reconstructed_floor": None,
                    "require_current": True,
                }
            },
        )
        time.sleep(0.02)
    assert acknowledgement[binding.project_id]["outcome"] == "acknowledged"
    assert read_consumer_progress(
        cfg.shared_root,
        runtime_id=runtime.instance_id,
        project_id=binding.project_id,
        registration_generation=binding.registration_generation,
    )["project_activation_consumer"]["ack"] == {
        "epoch": checkpoint["epoch"],
        "sequence": checkpoint["sequence"],
    }


def test_outer_admission_turn_selects_across_public_service_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    try:
        _advance_until_validated(controller, bindings, revision, binding.project_id)
        _prime_current_activation(runtime, bindings, revision)
        monkeypatch.setattr(executor, "start", lambda _request_id: None)
        with controller.admission_turn():
            # Background is discovered first, but validation already spent the
            # first authority opportunity. Primary must receive the next one.
            controller.advance_notification_maintenance(bindings, revision)
            controller.advance_scheduler_observations(bindings, revision, lane="gpu", admission_role="primary")
            controller.advance_registration_renewals(bindings, revision, renewal_horizon_seconds=10.0)
            assert executor.unresolved_requests() == ()
        requests = executor.unresolved_requests()
        assert len(requests) == 1
        assert requests[0].operation_kind == "scheduler_observe"
        assert controller.has_pending_admission
        assert controller.pending_wait_seconds(5.0) == 0.1
    finally:
        executor.shutdown()


@pytest.mark.parametrize(
    ("method_name", "kwargs"),
    [
        ("advance_binding_validation", {"registry_revision": -1}),
        ("advance_scheduler_observations", {"lane": "invalid", "admission_role": "primary"}),
        (
            "advance_scheduler_claims",
            {
                "lane": "invalid",
                "admission_role": "primary",
                "observations": {},
                "available_gpu_ids": [],
                "available_cpu_slots": 0,
            },
        ),
        ("advance_scheduler_launch_authorizations", {"reservations": "invalid"}),
        ("advance_registration_renewals", {"renewal_horizon_seconds": -1}),
        ("advance_activation_observations", {"registry_revision": -1}),
    ],
)
def test_invalid_public_service_input_never_enters_executor_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method_name: str, kwargs: dict
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)

    def reject_mutation():
        raise AssertionError("invalid input reached mutating executor reconciliation")

    monkeypatch.setattr(executor, "poll", reject_mutation)
    monkeypatch.setattr(executor, "unresolved_requests", reject_mutation)
    with pytest.raises(ValueError):
        getattr(controller, method_name)([], **{"registry_revision": 0, **kwargs})


def test_failed_outer_turn_does_not_prepare_discovered_requests(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, _binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    try:
        with pytest.raises(ValueError, match="renewal_horizon_seconds"):
            with controller.admission_turn():
                controller.advance_binding_validation(bindings, revision)
                controller.advance_registration_renewals(bindings, revision, renewal_horizon_seconds=-1)
        assert executor.unresolved_requests() == ()
        assert not controller.has_pending_admission
        controller.advance_binding_validation(bindings, revision)
        assert len(executor.unresolved_requests()) == 1
    finally:
        executor.shutdown()


def test_consumption_wakes_successor_discovery_when_only_overdue_requests_remain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg_a, hung_a, _revision, _bindings = _registered(tmp_path, "hung-a", runtime)
    _cfg_b, hung_b, _revision, _bindings = _registered(tmp_path, "hung-b", runtime)
    _cfg_c, healthy, revision, _bindings = _registered(tmp_path, "healthy", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    try:
        _advance_until_validated(controller, [healthy], revision, healthy.project_id)
        _prime_activation_consumers(runtime, [healthy], revision)
        for binding in (hung_a, hung_b):
            executor.prepare_validate_binding(binding, revision)
        real_status = executor.status_view

        def overdue_status():
            status = real_status()
            status.update(overdue_worker_count=2, envelope="degraded")
            return status

        monkeypatch.setattr(executor, "status_view", overdue_status)
        controller.advance_activation_observations([healthy], revision)
        deadline = time.monotonic() + 5.0
        while not executor.has_ready_result() and time.monotonic() < deadline:
            executor.poll()
            time.sleep(0.02)
        assert executor.has_ready_result()
        completed = controller.advance_activation_observations([healthy], revision)
        assert completed[healthy.project_id]["outcome"] == "observed"
        assert len(executor.unresolved_requests()) == 2
        assert controller.has_pending_admission
        assert controller.pending_wait_seconds(5.0) == 0.1
        # The wakeup hint is not a permanent busy-poll state for held work.
        with controller.admission_turn():
            pass
        assert not controller.has_pending_admission
        assert controller.pending_wait_seconds(5.0) == 5.0
    finally:
        executor.shutdown()


def test_blocked_activation_observation_does_not_delay_healthy_peer(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    blocked_cfg, blocked, _revision, _bindings = _registered(tmp_path, "blocked", runtime)
    healthy_cfg, healthy, revision, bindings = _registered(tmp_path, "healthy", runtime)
    healthy_checkpoint = publish_project_activation(healthy_cfg, "work")["project_activation"]
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, blocked.project_id)
    _advance_until_validated(controller, bindings, revision, healthy.project_id)
    _prime_activation_consumers(runtime, bindings, revision)
    blocked_identity = shared_paths(blocked_cfg.shared_root)["project"] / "identity.json"
    original_identity = blocked_identity.read_bytes()
    blocked_identity.unlink()
    os.mkfifo(blocked_identity)

    try:
        completions = {}
        deadline = time.monotonic() + 5.0
        while healthy.project_id not in completions and time.monotonic() < deadline:
            completions = controller.advance_activation_observations(bindings, revision)
            time.sleep(0.02)

        assert completions[healthy.project_id] == {
            "outcome": "observed",
            "checkpoint": {
                "epoch": healthy_checkpoint["epoch"],
                "sequence": healthy_checkpoint["sequence"],
            },
            "replay": {
                "epoch": healthy_checkpoint["epoch"],
                "sequence": healthy_checkpoint["sequence"],
                "reconstructed_floor": None,
                "complete": True,
            },
        }
        assert executor.status_view()["active_worker_count"] == 1
    finally:
        blocked_identity.unlink()
        blocked_identity.write_bytes(original_identity)
        executor.shutdown()


def test_activation_controller_fairly_revisits_more_than_sixty_four_candidates(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    for index in range(65):
        _registered(tmp_path, f"project-{index}", runtime)
    revision, bindings = runtime.load_registry()
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    completed: set[str] = set()
    last_project_id = bindings[-1].project_id
    deadline = time.monotonic() + 20.0

    while last_project_id not in completed and time.monotonic() < deadline:
        completed.update(
            controller.advance_activation_consumer_registrations(
                bindings,
                revision,
                process_fence="process-a",
            )
        )
        time.sleep(0.02)

    assert last_project_id in completed


def test_binding_commit_guard_serializes_concurrent_disablement(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, _revision, _bindings = _registered(tmp_path, "project-a", runtime)
    started = threading.Event()
    finished = threading.Event()

    def disable() -> None:
        started.set()
        runtime.set_enabled(binding.project_id, False)
        finished.set()

    with runtime.binding_commit_guard(binding):
        thread = threading.Thread(target=disable)
        thread.start()
        assert started.wait(1.0)
        assert not finished.wait(0.1)
        _revision, current = runtime.load_registry()
        assert current[0].enabled is True

    assert finished.wait(1.0)
    thread.join()
    _revision, current = runtime.load_registry()
    assert current[0].enabled is False


def test_blocked_snapshot_publication_does_not_delay_healthy_peer(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    blocked_cfg, blocked, _revision, _bindings = _registered(tmp_path, "blocked", runtime)
    healthy_cfg, healthy, revision, bindings = _registered(tmp_path, "healthy", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, blocked.project_id)
    _advance_until_validated(controller, bindings, revision, healthy.project_id)
    blocked_machine_record = blocked_cfg.shared_root / "machines" / "gpu-1" / "machine.json"
    original = blocked_machine_record.read_bytes()
    blocked_machine_record.unlink()
    os.mkfifo(blocked_machine_record)

    try:
        controller.advance_machine_snapshot_publications(
            bindings,
            revision,
            instance_id="agent-1",
            pid=123,
            visible_gpu_ids=[],
            reservations=[],
            heartbeat_interval_seconds=5.0,
            started_at="2026-09-28T00:00:00+00:00",
            gpu_policy={"mode": "none", "warnings": []},
        )
        deadline = time.monotonic() + 5.0
        healthy_completion = None
        while time.monotonic() < deadline:
            completions = controller.advance_machine_snapshot_publications(
                bindings,
                revision,
                instance_id="agent-1",
                pid=123,
                visible_gpu_ids=[],
                reservations=[],
                heartbeat_interval_seconds=5.0,
                started_at="2026-09-28T00:00:00+00:00",
                gpu_policy={"mode": "none", "warnings": []},
            )
            healthy_completion = completions.get(healthy.project_id)
            if healthy_completion is not None:
                break
            time.sleep(0.02)

        assert healthy_completion["outcome"] == "published"
        assert machine_state_path(healthy_cfg, "agent.json").exists()
        assert executor.status_view()["active_worker_count"] == 1
    finally:
        blocked_machine_record.unlink()
        blocked_machine_record.write_bytes(original)
        executor.shutdown()


def test_controller_replays_ambiguous_event_flush_without_wedging_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    now = [100.0]
    controller = ProjectIOController(runtime, executor, monotonic=lambda: now[0])
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    event_id = "d" * 32
    source = local_paths(runtime.project_paths(binding.project_id)["root"])["events"] / "machine" / f"{event_id}.json"
    atomic_replace(
        source,
        {
            "event_id": event_id,
            "event_type": "ambiguous",
            "task_id": None,
            "machine_name": "gpu-1",
            "timestamp": "2026-09-28T00:00:00+00:00",
            "details": {},
        },
    )
    real_start = executor.start
    monkeypatch.setattr(executor, "start", lambda _request_id, **_kwargs: None)
    controller.advance_maintenance_event_flushes(bindings, revision)
    request = next(
        request for request in executor.unresolved_requests() if request.operation_kind == "maintenance_flush_event"
    )
    unknown = ProjectIOResult(
        request=request,
        status="outcome_unknown",
        reason_code="project_io_outcome_unknown",
        completed_at="2026-09-28T00:00:01Z",
        evidence={},
    )
    executor._write_record(
        executor._record_path("results", request.request_id),
        unknown.to_dict(),
        "project_io_result",
    )
    monkeypatch.setattr(executor, "start", real_start)

    controller.advance_maintenance_event_flushes(bindings, revision)
    now[0] += 300.0
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        controller.advance_maintenance_event_flushes(bindings, revision)
        if not executor.unresolved_requests():
            break
        time.sleep(0.02)

    assert executor.unresolved_requests() == ()
    assert not source.exists()


def test_controller_resolves_stale_event_request_after_binding_revision_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    event_id = "e" * 32
    source = local_paths(runtime.project_paths(binding.project_id)["root"])["events"] / "machine" / f"{event_id}.json"
    atomic_replace(
        source,
        {
            "event_id": event_id,
            "event_type": "stale",
            "task_id": None,
            "machine_name": "gpu-1",
            "timestamp": "2026-09-28T00:00:00+00:00",
            "details": {},
        },
    )
    monkeypatch.setattr(executor, "start", lambda _request_id, **_kwargs: None)
    controller.advance_maintenance_event_flushes(bindings, revision)
    assert executor.unresolved_requests()
    runtime.set_enabled(binding.project_id, False)
    current_revision, current_bindings = runtime.load_registry()

    controller.advance_maintenance_event_flushes(current_bindings, current_revision)

    assert executor.unresolved_requests() == ()
    assert source.exists()


def test_unknown_executor_state_hides_cached_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    unknown = executor.status_view()
    unknown["envelope"] = "unknown"
    monkeypatch.setattr(executor, "poll", lambda: unknown)

    current = controller.advance_binding_validation(bindings, revision)

    assert current == {}
    assert controller.validated_config(binding, revision) is None


def test_disabled_binding_remains_isolated_for_recovery_services(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, _revision, _bindings = _registered(tmp_path, "project-a", runtime)
    disabled = runtime.set_enabled(binding.project_id, False)
    revision, bindings = runtime.load_registry()
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)

    configs = _advance_until_validated(controller, bindings, revision, disabled.project_id)

    assert disabled.enabled is False
    assert disabled.project_id in configs
    assert controller.validated_config(disabled, revision) is not None


def test_binding_validation_requires_exact_runtime_registration(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, _bindings = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    request = executor.prepare_validate_binding(binding, revision)
    registration_path = shared_paths(cfg.shared_root)["machines"] / cfg.machine_name / "registration.json"
    value = read_json(registration_path)
    value["registration"]["runtime_instance_id"] = "f" * 64
    atomic_replace(registration_path, value)

    executor.start(request.request_id)
    deadline = time.monotonic() + 5.0
    result = None
    while time.monotonic() < deadline:
        executor.poll()
        result = executor.load_result(request.request_id)
        if result is not None:
            break
        time.sleep(0.02)

    assert result is not None
    assert result.status == "retryable_error"
    assert result.reason_code == "project_io_binding_validation_failed"


def test_binding_validation_requires_exact_shared_runtime_root(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, _bindings = _registered(tmp_path, "project-a", runtime)
    registration_path = shared_paths(cfg.shared_root)["machines"] / cfg.machine_name / "registration.json"
    value = read_json(registration_path)
    value["registration"]["runtime_root"] = str(tmp_path / "different-machine")
    atomic_replace(registration_path, value)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    request = executor.prepare_validate_binding(binding, revision)

    executor.start(request.request_id)
    deadline = time.monotonic() + 5.0
    result = None
    while time.monotonic() < deadline:
        executor.poll()
        result = executor.load_result(request.request_id)
        if result is not None:
            break
        time.sleep(0.02)

    assert result is not None
    assert result.status == "retryable_error"


@pytest.mark.parametrize("owner_field", ["runtime_instance_id", "runtime_root"])
def test_registry_runtime_owner_mismatch_never_starts_validation(tmp_path: Path, owner_field: str) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, _bindings = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)

    replacement = "f" * 64 if owner_field == "runtime_instance_id" else str(tmp_path / "different-runtime")
    configs = controller.advance_binding_validation([replace(binding, **{owner_field: replacement})], revision)

    assert configs == {}
    assert executor.unresolved_requests() == ()


def test_binding_validation_error_uses_bounded_retry_backoff(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    registration_path = shared_paths(cfg.shared_root)["machines"] / cfg.machine_name / "registration.json"
    valid_registration = read_json(registration_path)
    invalid_registration = read_json(registration_path)
    invalid_registration["registration"]["runtime_instance_id"] = "f" * 64
    atomic_replace(registration_path, invalid_registration)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    now = [100.0]
    controller = ProjectIOController(runtime, executor, monotonic=lambda: now[0])
    controller.advance_binding_validation(bindings, revision)

    deadline = time.monotonic() + 5.0
    while executor.unresolved_requests() and time.monotonic() < deadline:
        controller.advance_binding_validation(bindings, revision)
        time.sleep(0.02)
    assert executor.unresolved_requests() == ()
    assert controller.validated_config(binding, revision) is None

    atomic_replace(registration_path, valid_registration)
    controller.advance_binding_validation(bindings, revision)
    assert executor.unresolved_requests() == ()

    now[0] += 5.0
    controller.advance_binding_validation(bindings, revision)
    assert len(executor.unresolved_requests()) == 1
    executor.shutdown()


@pytest.mark.parametrize("failure_mode", ["raise", "none"])
def test_binding_validation_retries_same_request_after_start_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    real_start = executor.start
    calls = 0

    def fail_once(request_id: str):
        nonlocal calls
        calls += 1
        if calls == 1:
            if failure_mode == "raise":
                raise OSError("synthetic start failure")
            return None
        return real_start(request_id)

    monkeypatch.setattr(executor, "start", fail_once)
    now = [100.0]
    controller = ProjectIOController(runtime, executor, monotonic=lambda: now[0])
    controller.advance_binding_validation(bindings, revision)
    prepared = executor.unresolved_requests()
    assert len(prepared) == 1

    controller.advance_binding_validation(bindings, revision)
    assert calls == 1
    now[0] += 5.0
    controller.advance_binding_validation(bindings, revision)

    assert calls == 2
    assert executor.unresolved_requests()[0].request_id == prepared[0].request_id
    configs = _advance_until_validated(controller, bindings, revision, binding.project_id)
    assert binding.project_id in configs


def test_due_validation_retry_reuses_its_existing_slot_at_supported_hang_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, _binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    real_start = executor.start
    calls = 0

    def fail_once(request_id: str):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("synthetic start failure")
        return real_start(request_id)

    monkeypatch.setattr(executor, "start", fail_once)
    now = [100.0]
    controller = ProjectIOController(runtime, executor, monotonic=lambda: now[0])
    controller.advance_binding_validation(bindings, revision)
    prepared = executor.unresolved_requests()[0]
    now[0] += 5.0
    status = executor.status_view()
    status.update(overdue_worker_count=2, free_slot_count=2, envelope="degraded")
    monkeypatch.setattr(executor, "poll", lambda: status)

    controller.advance_binding_validation(bindings, revision)

    assert calls == 2
    assert executor.unresolved_requests() == (prepared,)
    executor.shutdown()


def test_unresolved_requests_are_sorted_and_do_not_consume_results(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg_a, binding_a, _revision, _bindings = _registered(tmp_path, "project-a", runtime)
    _cfg_b, binding_b, revision, _bindings = _registered(tmp_path, "project-b", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    second = executor.prepare_validate_binding(binding_a, revision)
    first = executor.prepare_validate_binding(binding_b, revision)

    requests = executor.unresolved_requests()

    assert requests == tuple(sorted((first, second), key=lambda request: request.request_id))
    assert executor.load_result(first.request_id) is None


def test_validation_uses_both_free_slots_when_supported_hang_limit_is_reached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, first, _revision, _bindings = _registered(tmp_path, "project-a", runtime)
    _cfg_b, second, _revision, _bindings = _registered(tmp_path, "project-b", runtime)
    _cfg_c, hung_a, _revision, _bindings = _registered(tmp_path, "hung-a", runtime)
    _cfg_d, hung_b, revision, bindings = _registered(tmp_path, "hung-b", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    for binding in (hung_a, hung_b):
        executor.prepare_activation_observe(binding, revision)
    status = executor.status_view()
    status.update(overdue_worker_count=2, free_slot_count=2, envelope="degraded")
    monkeypatch.setattr(executor, "poll", lambda: status)
    controller = ProjectIOController(runtime, executor)

    assert controller.advance_binding_validation(bindings, revision) == {}
    requests = executor.unresolved_requests()
    assert len(requests) == 4
    assert {request.project_id for request in requests if request.operation_kind == "validate_binding"} == {
        first.project_id,
        second.project_id,
    }
    executor.shutdown()


def test_running_validation_does_not_reserve_the_other_healthy_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    first_cfg, first, _revision, _bindings = _registered(tmp_path, "project-a", runtime)
    _second_cfg, second, _revision, _bindings = _registered(tmp_path, "project-b", runtime)
    _cfg_c, hung_a, _revision, _bindings = _registered(tmp_path, "hung-a", runtime)
    _cfg_d, hung_b, revision, bindings = _registered(tmp_path, "hung-b", runtime)
    identity_path = shared_paths(first_cfg.shared_root)["project"] / "identity.json"
    original = identity_path.read_bytes()
    identity_path.unlink()
    os.mkfifo(identity_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    for binding in (hung_a, hung_b):
        executor.prepare_activation_observe(binding, revision)
    running = executor.prepare_validate_binding(first, revision)
    executor.start(running.request_id)
    status = executor.status_view()
    status.update(overdue_worker_count=2, free_slot_count=1, envelope="degraded")
    monkeypatch.setattr(executor, "poll", lambda: status)
    controller = ProjectIOController(runtime, executor)

    try:
        assert controller.advance_binding_validation(bindings, revision) == {}
        requests = executor.unresolved_requests()
        assert running in requests
        assert len(requests) == 4
        assert any(request.project_id == second.project_id for request in requests)
    finally:
        identity_path.unlink()
        identity_path.write_bytes(original)
        executor.shutdown()


def test_validation_does_not_consume_or_replace_other_operation_kind(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg_a, binding_a, _revision, _bindings = _registered(tmp_path, "project-a", runtime)
    _cfg_b, binding_b, revision, bindings = _registered(tmp_path, "project-b", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    observation = executor.prepare_scheduler_observe(
        binding_a,
        revision,
        lane="gpu",
        admission_role="primary",
        cursor_namespace="scheduler-primary-gpu",
    )
    controller = ProjectIOController(runtime, executor)

    controller.advance_binding_validation(bindings, revision)
    requests = executor.unresolved_requests()

    assert any(request == observation for request in requests)
    assert any(
        request.operation_kind == "validate_binding" and request.project_id == binding_b.project_id
        for request in requests
    )
    assert all(
        request.operation_kind != "validate_binding" or request.project_id != binding_a.project_id
        for request in requests
    )
    executor.shutdown()


def test_blocked_scheduler_observation_does_not_delay_healthy_peer(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    blocked_cfg, blocked, _revision, _bindings = _registered(tmp_path, "blocked", runtime)
    healthy_cfg, healthy, revision, bindings = _registered(tmp_path, "healthy", runtime)
    submit(blocked_cfg, ["echo", "blocked"], working_dir=tmp_path)
    healthy_task = submit(healthy_cfg, ["echo", "healthy"], working_dir=tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        validated = controller.advance_binding_validation(bindings, revision)
        if set(validated) == {blocked.project_id, healthy.project_id}:
            break
        time.sleep(0.02)
    else:
        raise AssertionError("both bindings did not validate")
    _prime_current_activation(runtime, bindings, revision)
    state_path = ready_state_path(blocked_cfg.shared_root)
    state_path.unlink()
    os.mkfifo(state_path)

    observations = {}
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        cycle_started = time.monotonic()
        observations = controller.advance_scheduler_observations(
            bindings,
            revision,
            lane="gpu",
            admission_role="primary",
        )
        assert time.monotonic() - cycle_started < 1.0
        if healthy.project_id in observations:
            break
        time.sleep(0.02)

    assert blocked.project_id not in observations
    assert observations[healthy.project_id]["outcome"] == "candidate"
    assert observations[healthy.project_id]["candidate"]["task_id"] == healthy_task.task_id
    assert executor.status_view()["active_worker_count"] == 1
    controller.advance_scheduler_observations([healthy], revision, lane="gpu", admission_role="primary")
    assert any(request.project_id == blocked.project_id for request in executor.unresolved_requests())
    assert executor.status_view()["active_worker_count"] == 1
    replayed = controller.advance_scheduler_observations(
        bindings,
        revision,
        lane="gpu",
        admission_role="primary",
    )
    assert replayed[healthy.project_id] == observations[healthy.project_id]
    executor.shutdown()


@pytest.mark.parametrize("invalidation", ["activation", "selection"])
def test_finished_scheduler_observation_does_not_block_fresh_activation(tmp_path: Path, invalidation: str) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project", runtime)
    task = submit(cfg, ["echo", "eligible"], working_dir=tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    try:
        _advance_until_validated(controller, bindings, revision, binding.project_id)
        _observe_working_set_activation(runtime, controller, bindings, revision)
        request = executor.prepare_scheduler_observe(
            binding,
            revision,
            lane="gpu",
            admission_role="primary",
            cursor_namespace=f"scheduler-{binding.project_id}-primary-gpu",
        )
        assert executor.start(request.request_id) is not None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = executor.poll()
            result = executor.load_result(request.request_id)
            if result is not None and status["active_worker_count"] == 0:
                break
            time.sleep(0.02)
        else:
            raise AssertionError("candidate observation did not finish")
        assert result.evidence["candidate"]["task_id"] == task.task_id
        if invalidation == "activation":
            _prime_activation_consumers(runtime, bindings, revision)
            assert not runtime.working_set.has_current_activation(binding)
        selected = [] if invalidation == "selection" else bindings
        assert controller.advance_scheduler_observations(selected, revision, lane="gpu", admission_role="primary") == {}
        assert all(item.request_id != request.request_id for item in executor.unresolved_requests())
        assert load_task(cfg, task.task_id).claim_control.get("active_claim") is None
        # The completed read owns no durable cursor or claim. Fresh activation
        # can use its slot and rediscover the exact candidate without skipping it.
        fresh = _advance_until_observed(controller, bindings, revision, binding.project_id, lane="gpu")
        assert fresh[binding.project_id]["candidate"]["task_id"] == task.task_id
    finally:
        executor.shutdown()


def test_observation_selection_preserves_other_lane_candidate(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project", runtime)
    task = submit(cfg, ["echo", "eligible"], working_dir=tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    try:
        _advance_until_validated(controller, bindings, revision, binding.project_id)
        expected = _advance_until_observed(controller, bindings, revision, binding.project_id, lane="gpu")
        assert expected[binding.project_id]["candidate"]["task_id"] == task.task_id
        assert controller.advance_scheduler_observations([], revision, lane="cpu", admission_role="primary") == {}
        assert (
            controller.advance_scheduler_observations(bindings, revision, lane="gpu", admission_role="primary")
            == expected
        )
        assert not any(item.operation_kind == "scheduler_observe" for item in executor.unresolved_requests())
    finally:
        executor.shutdown()


def test_empty_scheduler_observation_does_not_hide_later_submission(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)

    empty = _advance_until_observed(
        controller,
        bindings,
        revision,
        binding.project_id,
        lane="gpu",
    )
    assert empty[binding.project_id]["outcome"] == "none"

    # Empty evidence owns the observed cursor until its typed commit finishes.
    # Re-observing from the same position would occupy the owner forever and
    # starve the commit that makes the next scan wrap safely.
    assert (
        controller.advance_scheduler_observations(
            bindings,
            revision,
            lane="gpu",
            admission_role="primary",
        )
        == {}
    )
    assert not any(request.operation_kind == "scheduler_observe" for request in executor.unresolved_requests())
    controller.advance_scheduler_cursor_commits(
        bindings,
        revision,
        lane="gpu",
        admission_role="primary",
        observations=empty,
    )
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        committed = controller.advance_scheduler_cursor_commits(
            bindings,
            revision,
            lane="gpu",
            admission_role="primary",
            observations={},
        )
        if binding.project_id in committed:
            break
        time.sleep(0.02)
    else:
        raise AssertionError("empty scheduler cursor did not commit")

    task = submit(cfg, ["echo", "later"], working_dir=tmp_path)
    observed = _advance_until_observed(
        controller,
        bindings,
        revision,
        binding.project_id,
        lane="gpu",
    )

    assert observed[binding.project_id]["outcome"] == "candidate"
    assert observed[binding.project_id]["candidate"]["task_id"] == task.task_id


def test_full_scheduler_candidate_window_claims_every_owner_without_phase_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(project_io_controller_module, "_MAX_RETAINED_SCHEDULER_INTENTS", 2)
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    bindings = []
    for index in range(3):
        cfg, binding, revision, bindings = _registered(tmp_path, f"project-{index}", runtime)
        submit(cfg, ["echo", str(index)], working_dir=tmp_path, requested_gpus=0, requested_cpus=1)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            validated = controller.advance_binding_validation(bindings, revision)
            if len(validated) == 3:
                break
            time.sleep(0.02)
        else:
            raise AssertionError("all scheduler-window bindings did not validate")
        _prime_current_activation(runtime, bindings, revision)

        observed_projects = set()
        selected_projects = set()

        def select_without_shared_mutation(
            _controller,
            _identity,
            selected_binding,
            _candidate,
            _cursor,
            _evidence,
            _lane,
            _admission_role,
            _allowed_gpu_ids,
            _batch_budget,
            _registry_revision,
            observation_key,
            _service_key,
            _borrow_admission,
            *,
            defer_request=False,
        ):
            assert defer_request is False
            selected_projects.add(selected_binding.project_id)
            controller._observations.pop(observation_key, None)

        monkeypatch.setattr(ProjectIOController, "_prepare_scheduler_claim_request", select_without_shared_mutation)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            with controller.admission_turn():
                observations = controller.advance_scheduler_observations(
                    bindings,
                    revision,
                    lane="cpu",
                    admission_role="primary",
                )
                controller.advance_scheduler_claims(
                    bindings,
                    revision,
                    lane="cpu",
                    admission_role="primary",
                    observations=observations,
                    available_gpu_ids=[],
                    available_cpu_slots=3,
                )
            observed_projects.update(
                project_id for project_id, evidence in observations.items() if evidence["outcome"] == "candidate"
            )
            if selected_projects == {binding.project_id for binding in bindings}:
                break
            time.sleep(0.02)
        else:
            raise AssertionError(
                f"full candidate window phase-locked: observed={observed_projects}, selected={selected_projects}"
            )
        assert observed_projects == {binding.project_id for binding in bindings}
        assert len(controller._observations) <= 2
    finally:
        executor.shutdown()


def test_unclaimable_scheduler_window_releases_evidence_for_omitted_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(project_io_controller_module, "_MAX_RETAINED_SCHEDULER_INTENTS", 2)
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    bindings = []
    for index in range(3):
        cfg, _binding, revision, bindings = _registered(tmp_path, f"project-{index}", runtime)
        submit(cfg, ["echo", str(index)], working_dir=tmp_path, requested_gpus=0, requested_cpus=1)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if len(controller.advance_binding_validation(bindings, revision)) == 3:
                break
            time.sleep(0.02)
        else:
            raise AssertionError("all unclaimable-window bindings did not validate")
        _prime_current_activation(runtime, bindings, revision)

        observed_projects = set()
        while time.monotonic() < deadline:
            with controller.admission_turn():
                observations = controller.advance_scheduler_observations(
                    bindings,
                    revision,
                    lane="cpu",
                    admission_role="primary",
                )
                observed_projects.update(
                    project_id for project_id, evidence in observations.items() if evidence["outcome"] == "candidate"
                )
                controller.advance_scheduler_claims(
                    bindings,
                    revision,
                    lane="cpu",
                    admission_role="primary",
                    observations=observations,
                    available_gpu_ids=[],
                    available_cpu_slots=0,
                )
            if observed_projects == {binding.project_id for binding in bindings}:
                break
            time.sleep(0.02)
        else:
            raise AssertionError(f"unclaimable candidate window omitted bindings: {observed_projects}")
        assert len(controller._observations) <= 2
    finally:
        executor.shutdown()


def test_claim_batch_limit_counts_candidates_not_empty_observations(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    empty = _advance_until_observed(
        controller,
        bindings,
        revision,
        binding.project_id,
        lane="gpu",
    )[binding.project_id]
    observations = {}
    for index in range(65):
        project_id = f"empty-{index:02d}"
        observations[project_id] = {
            **empty,
            "cursor": {**empty["cursor"], "namespace": f"scheduler-{project_id}-primary-gpu"},
        }

    assert (
        controller.advance_scheduler_claims(
            bindings,
            revision,
            lane="gpu",
            admission_role="primary",
            observations=observations,
            available_gpu_ids=[],
            available_cpu_slots=0,
        )
        == {}
    )


def test_scheduler_cursor_commit_consumes_exact_empty_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    observations = _advance_until_observed(
        controller,
        bindings,
        revision,
        binding.project_id,
        lane="gpu",
    )
    evidence = observations[binding.project_id]
    assert evidence["outcome"] == "none"

    retry_due = controller._service_retry_is_due
    monkeypatch.setattr(
        controller,
        "_service_retry_is_due",
        lambda key: False if key[1] == "scheduler_cursor_commit" else retry_due(key),
    )
    assert (
        controller.advance_scheduler_cursor_commits(
            bindings,
            revision,
            lane="gpu",
            admission_role="primary",
            observations=observations,
        )
        == {}
    )
    assert executor.unresolved_requests() == ()
    monkeypatch.setattr(controller, "_service_retry_is_due", retry_due)
    monkeypatch.setattr(controller, "scheduler_is_quiescent", lambda *_args: True)
    deadline = time.monotonic() + 5.0
    completions = {}
    while time.monotonic() < deadline:
        completions = controller.advance_scheduler_cursor_commits(
            bindings,
            revision,
            lane="gpu",
            admission_role="primary",
            observations={},
        )
        if binding.project_id in completions:
            break
        time.sleep(0.02)

    assert set(completions[binding.project_id]["routes"].values()) <= {
        "committed",
        "already_applied",
        "stale",
    }
    expected = evidence["cursor"]["routes"]["home"]["next"]
    saved = load_ready_cursor(cfg, evidence["cursor"]["namespace"], "home")
    assert {
        "catalog_page": saved.catalog_page,
        "partition": saved.partition,
        "after_name": saved.after_name,
        "revision": saved.revision,
    } == expected
    assert executor.unresolved_requests() == ()


@pytest.mark.parametrize("fence", ["unchanged", "wake", "lost_turn"])
def test_scheduler_quiescence_acknowledges_only_its_captured_complete_scan(tmp_path, fence):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project", runtime)
    repair_ready_index(cfg)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    try:
        _advance_until_validated(controller, bindings, revision, binding.project_id)
        _observe_working_set_activation(runtime, controller, bindings, revision)
        controller.advance_scheduler_quiescence(bindings, revision)
        request = next(
            item for item in executor.unresolved_requests() if item.operation_kind == "scheduler_quiescence_probe"
        )
        turn = controller._scheduler_quiescence_requests[request.request_id].turn
        assert not turn.unknown
        state = runtime.working_set._states[turn.identity]
        assert "scheduler" not in state.acknowledged_lanes
        if fence == "wake":
            runtime.working_set.activate(binding, "new_local_work")
        elif fence == "lost_turn":
            controller._scheduler_quiescence_rounds.clear()
            controller._scheduler_quiescence_requests.clear()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            controller.advance_scheduler_quiescence(bindings, revision)
            if request not in executor.unresolved_requests():
                break
            time.sleep(0.02)
        else:
            raise AssertionError("scheduler census result was not consumed")
        if fence != "unchanged":
            assert "scheduler" not in state.acknowledged_lanes
            assert not controller.scheduler_is_quiescent(binding, revision)
            assert any(item.operation_kind == "scheduler_quiescence_probe" for item in executor.unresolved_requests())
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            controller.advance_scheduler_quiescence(bindings, revision)
            if controller.scheduler_is_quiescent(binding, revision):
                break
            time.sleep(0.02)
        else:
            raise AssertionError("fresh complete scheduler census did not acknowledge its current turn")
        assert "scheduler" in state.acknowledged_lanes
        for _ in range(3):
            controller.advance_scheduler_quiescence(bindings, revision)
        assert executor.unresolved_requests() == ()
    finally:
        executor.shutdown()


@pytest.mark.parametrize("dependency", ["ready_index", "borrow_candidate"])
def test_scheduler_quiescence_defers_known_scheduling_dependencies(tmp_path, dependency):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project", runtime)
    if dependency == "ready_index":
        _reset_ready_index_for_isolated_build(cfg)
    else:
        create_group(cfg, "borrow-group")
        change_worker(cfg, "borrow-group", "gpu-1", "set", role="borrow")
        submit(cfg, ["echo", "work"], group="borrow-group", working_dir=tmp_path)
        repair_ready_index(cfg)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    try:
        _advance_until_validated(controller, bindings, revision, binding.project_id)
        _observe_working_set_activation(runtime, controller, bindings, revision)
        observation = _advance_until_observed(
            controller, bindings, revision, binding.project_id, lane="gpu", admission_role="borrow"
        )[binding.project_id]
        if dependency == "ready_index":
            assert observation["reason"] == "ready_index_inactive"
        else:
            assert observation["outcome"] == "candidate"
        assert executor.unresolved_requests() == ()
        for _ in range(5):
            controller.advance_scheduler_quiescence(bindings, revision)
        assert executor.unresolved_requests() == ()
        state = runtime.working_set._states[runtime.working_set._identity(binding)]
        assert "scheduler" not in state.acknowledged_lanes
        assert not controller.scheduler_is_quiescent(binding, revision)
    finally:
        executor.shutdown()


def test_scheduler_candidate_discovery_invalidates_an_old_quiescence_proof(tmp_path):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project", runtime)
    repair_ready_index(cfg)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    try:
        _advance_until_validated(controller, bindings, revision, binding.project_id)
        _observe_working_set_activation(runtime, controller, bindings, revision)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            controller.advance_scheduler_quiescence(bindings, revision)
            if controller.scheduler_is_quiescent(binding, revision):
                break
            time.sleep(0.02)
        else:
            raise AssertionError("empty Project did not produce a complete scheduler proof")
        task = submit(cfg, ["echo", "new-work"], working_dir=tmp_path)
        runtime.working_set.activate(binding, "test_new_task")
        observations = _advance_until_observed(controller, bindings, revision, binding.project_id, lane="gpu")
        assert observations[binding.project_id]["candidate"]["task_id"] == task.task_id
        assert not controller.scheduler_is_quiescent(binding, revision)
        state = runtime.working_set._states[runtime.working_set._identity(binding)]
        assert "scheduler" not in state.acknowledged_lanes
    finally:
        executor.shutdown()


def test_scheduler_cursor_commit_rejects_fabricated_empty_observation(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    observations = _advance_until_observed(
        controller,
        bindings,
        revision,
        binding.project_id,
        lane="gpu",
    )
    observations[binding.project_id]["cursor"]["routes"]["home"]["next"]["revision"] += 1

    with pytest.raises(ValueError, match="not produced"):
        controller.advance_scheduler_cursor_commits(
            bindings,
            revision,
            lane="gpu",
            admission_role="primary",
            observations=observations,
        )
    assert not any(request.operation_kind == "scheduler_cursor_commit" for request in executor.unresolved_requests())


def test_scheduler_cursor_commit_replays_outcome_unknown_exact_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    now = [100.0]
    controller = ProjectIOController(runtime, executor, monotonic=lambda: now[0])
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    observations = _advance_until_observed(
        controller,
        bindings,
        revision,
        binding.project_id,
        lane="gpu",
    )
    real_start = executor.start
    monkeypatch.setattr(executor, "start", lambda _request_id, **_kwargs: None)
    controller.advance_scheduler_cursor_commits(
        bindings,
        revision,
        lane="gpu",
        admission_role="primary",
        observations=observations,
    )
    request = next(
        request for request in executor.unresolved_requests() if request.operation_kind == "scheduler_cursor_commit"
    )
    unknown = ProjectIOResult(
        request=request,
        status="outcome_unknown",
        reason_code="project_io_scheduler_cursor_commit_failed",
        completed_at="2026-09-28T00:00:01Z",
        evidence={},
    )
    executor._write_record(
        executor._record_path("results", request.request_id),
        unknown.to_dict(),
        "project_io_result",
    )
    monkeypatch.setattr(executor, "start", real_start)
    now[0] += 5.0

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        controller.advance_scheduler_cursor_commits(
            bindings,
            revision,
            lane="gpu",
            admission_role="primary",
            observations={},
        )
        if not executor.unresolved_requests():
            break
        time.sleep(0.02)

    assert executor.unresolved_requests() == ()


def test_scheduler_cursor_commit_resolves_request_after_registry_revision_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    observations = _advance_until_observed(
        controller,
        bindings,
        revision,
        binding.project_id,
        lane="gpu",
    )
    monkeypatch.setattr(executor, "start", lambda _request_id, **_kwargs: None)
    controller.advance_scheduler_cursor_commits(
        bindings,
        revision,
        lane="gpu",
        admission_role="primary",
        observations=observations,
    )
    assert any(request.operation_kind == "scheduler_cursor_commit" for request in executor.unresolved_requests())
    _other_cfg, _other, current_revision, current_bindings = _registered(tmp_path, "project-b", runtime)

    controller.advance_scheduler_cursor_commits(
        current_bindings,
        current_revision,
        lane="gpu",
        admission_role="primary",
        observations={},
    )

    assert executor.unresolved_requests() == ()


def test_scheduler_observation_progresses_past_sixty_four_empty_projects(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    bindings = []
    target_cfg = None
    target = None
    revision = 0
    for index in range(65):
        cfg, binding, revision, bindings = _registered(tmp_path, f"project-{index:02d}", runtime)
        if index == 64:
            target_cfg = cfg
            target = binding
    assert target_cfg is not None and target is not None
    task = submit(target_cfg, ["echo", "last"], working_dir=tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _prime_current_activation(runtime, bindings, revision)

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        controller.advance_binding_validation(bindings, revision)
        observations = controller.advance_scheduler_observations(
            bindings,
            revision,
            lane="gpu",
            admission_role="primary",
        )
        if target.project_id in observations and observations[target.project_id]["outcome"] == "candidate":
            break
        time.sleep(0.02)
    else:
        raise AssertionError("scheduler observation did not progress beyond the first 64 projects")

    assert observations[target.project_id]["candidate"]["task_id"] == task.task_id
    assert len(controller._observations) <= 64
    executor.shutdown()


def test_scheduler_observation_retries_same_request_without_delaying_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    task = submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    now = [100.0]
    controller = ProjectIOController(runtime, executor, monotonic=lambda: now[0])
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    _prime_current_activation(runtime, bindings, revision)
    real_start = executor.start
    calls = 0

    def fail_once(request_id: str):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("synthetic observation start failure")
        return real_start(request_id)

    monkeypatch.setattr(executor, "start", fail_once)
    controller.advance_scheduler_observations(
        bindings,
        revision,
        lane="gpu",
        admission_role="primary",
    )
    prepared = executor.unresolved_requests()[0]
    assert controller.validated_config(binding, revision) is not None

    controller.advance_scheduler_observations(
        bindings,
        revision,
        lane="gpu",
        admission_role="primary",
    )
    assert calls == 1
    now[0] += 5.0
    controller.advance_scheduler_observations(
        bindings,
        revision,
        lane="gpu",
        admission_role="primary",
    )
    assert calls == 2
    assert executor.unresolved_requests()[0].request_id == prepared.request_id

    deadline = time.monotonic() + 5.0
    evidence = {}
    while time.monotonic() < deadline:
        evidence = controller.advance_scheduler_observations(
            bindings,
            revision,
            lane="gpu",
            admission_role="primary",
        )
        if binding.project_id in evidence:
            break
        time.sleep(0.02)
    assert evidence[binding.project_id]["candidate"]["task_id"] == task.task_id


def test_scheduler_observation_can_use_reserved_peer_slot_at_two_overdue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    _prime_current_activation(runtime, bindings, revision)
    status = executor.status_view()
    status.update(overdue_worker_count=2, free_slot_count=2, envelope="degraded")
    monkeypatch.setattr(executor, "poll", lambda: status)

    controller.advance_scheduler_observations(
        bindings,
        revision,
        lane="gpu",
        admission_role="primary",
    )

    assert len(executor.unresolved_requests()) == 1
    executor.shutdown()


def test_scheduler_observation_rejects_invalid_service_identity_before_mutation(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, _binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)

    with pytest.raises(ValueError, match="lane"):
        controller.advance_scheduler_observations(
            bindings,
            revision,
            lane="memory",
            admission_role="primary",
        )
    with pytest.raises(ValueError, match="admission_role"):
        controller.advance_scheduler_observations(
            bindings,
            revision,
            lane="gpu",
            admission_role="opportunistic",
        )
    assert executor.unresolved_requests() == ()


def test_scheduler_observation_worker_exit_is_retryable(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    now = [100.0]
    controller = ProjectIOController(runtime, executor, monotonic=lambda: now[0])
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    _prime_current_activation(runtime, bindings, revision)
    state_path = ready_state_path(cfg.shared_root)
    state_path.unlink()
    os.mkfifo(state_path)
    controller.advance_scheduler_observations(
        bindings,
        revision,
        lane="gpu",
        admission_role="primary",
    )
    request = executor.unresolved_requests()[0]
    process = executor._load_process(request.request_id)
    os.kill(process.pid, 9)

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        controller.advance_scheduler_observations(
            bindings,
            revision,
            lane="gpu",
            admission_role="primary",
        )
        if not executor.unresolved_requests():
            break
        now[0] += 5.0
        time.sleep(0.02)
    assert executor.unresolved_requests() == ()

    now[0] += 5.0
    controller.advance_scheduler_observations(
        bindings,
        revision,
        lane="gpu",
        admission_role="primary",
    )
    assert len(executor.unresolved_requests()) == 1
    assert executor.unresolved_requests()[0].request_id != request.request_id
    executor.shutdown()


def test_scheduler_observation_does_not_consume_foreign_cursor_namespace(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    foreign = executor.prepare_scheduler_observe(
        binding,
        revision,
        lane="gpu",
        admission_role="primary",
        cursor_namespace="foreign-primary-gpu",
    )
    executor.start(foreign.request_id)
    deadline = time.monotonic() + 5.0
    while executor.load_result(foreign.request_id) is None and time.monotonic() < deadline:
        executor.poll()
        time.sleep(0.02)

    assert (
        controller.advance_scheduler_observations(
            bindings,
            revision,
            lane="gpu",
            admission_role="primary",
        )
        == {}
    )
    assert executor.unresolved_requests() == (foreign,)
    assert executor.load_result(foreign.request_id) is not None


def test_scheduler_claim_allocates_and_attaches_exact_gpu_offer(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    task = submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    observations = _advance_until_observed(
        controller,
        bindings,
        revision,
        binding.project_id,
        lane="gpu",
    )

    completions = controller.advance_scheduler_claims(
        bindings,
        revision,
        lane="gpu",
        admission_role="primary",
        observations=observations,
        available_gpu_ids=[0],
        available_cpu_slots=0,
    )
    claim = next(request for request in executor.unresolved_requests() if request.operation_kind == "scheduler_claim")
    offer_identity = _claim_offer_identity(claim)
    assert completions == {}
    assert classify_executor_offer(runtime.root, offer_identity) == "matching_provisional"

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        completions = controller.advance_scheduler_claims(
            bindings,
            revision,
            lane="gpu",
            admission_role="primary",
            observations={},
            available_gpu_ids=[],
            available_cpu_slots=0,
        )
        if binding.project_id in completions:
            break
        time.sleep(0.02)

    assert completions[binding.project_id]["outcome"] == "claimed"
    assert completions[binding.project_id]["attempt_id"] == f"{task.task_id}-attempt-1"
    assert classify_executor_offer(runtime.root, offer_identity) == "matching_active"
    assert executor.unresolved_requests() == ()


@pytest.mark.parametrize("available_gpu_ids", [[0], [0, 1]])
def test_same_turn_claim_selection_allocates_only_selected_distinct_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    available_gpu_ids: list[int],
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    first_cfg, first, _revision, _bindings = _registered(tmp_path, "project-a", runtime)
    second_cfg, second, revision, bindings = _registered(tmp_path, "project-b", runtime)
    submit(first_cfg, ["echo", "first"], working_dir=tmp_path)
    submit(second_cfg, ["echo", "second"], working_dir=tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    try:
        for binding in (first, second):
            _advance_until_validated(controller, bindings, revision, binding.project_id)
        observations = {}
        for binding in (first, second):
            observations.update(_advance_until_observed(controller, bindings, revision, binding.project_id, lane="gpu"))
        assert set(observations) == {first.project_id, second.project_id}
        monkeypatch.setattr(executor, "start", lambda _request_id, **_kwargs: None)
        with controller.admission_turn():
            controller.advance_scheduler_claims(
                bindings,
                revision,
                lane="gpu",
                admission_role="primary",
                observations=observations,
                available_gpu_ids=available_gpu_ids,
                available_cpu_slots=0,
            )
            assert executor.unresolved_requests() == ()
            assert reservation_snapshot(runtime.root).provisional == ()
        requests = executor.unresolved_requests()
        assert len(requests) == len(available_gpu_ids)
        assert all(request.operation_kind == "scheduler_claim" for request in requests)
        offers = [_claim_offer_identity(request) for request in requests]
        allocated = [gpu for request in requests for gpu in request.parameters["offer"]["gpu_ids"]]
        assert sorted(allocated) == available_gpu_ids
        assert len(set(allocated)) == len(allocated)
        assert all(classify_executor_offer(runtime.root, offer) == "matching_provisional" for offer in offers)
        assert len(reservation_snapshot(runtime.root).provisional) == len(requests)
    finally:
        executor.shutdown()


@pytest.mark.parametrize("should_revalidate", [False, True])
def test_scheduler_launch_authorization_requires_and_rechecks_exact_active_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    should_revalidate: bool,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    task = submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    now = [100.0]
    controller = ProjectIOController(runtime, executor, monotonic=lambda: now[0])
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    observations = _advance_until_observed(
        controller,
        bindings,
        revision,
        binding.project_id,
        lane="gpu",
    )
    controller.advance_scheduler_claims(
        bindings,
        revision,
        lane="gpu",
        admission_role="primary",
        observations=observations,
        available_gpu_ids=[0],
        available_cpu_slots=0,
    )
    claim_request = next(
        request for request in executor.unresolved_requests() if request.operation_kind == "scheduler_claim"
    )
    reservation = _claim_offer_identity(claim_request)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        controller.advance_scheduler_claims(
            bindings,
            revision,
            lane="gpu",
            admission_role="primary",
            observations={},
            available_gpu_ids=[],
            available_cpu_slots=0,
        )
        if classify_executor_offer(runtime.root, reservation) == "matching_active":
            break
        time.sleep(0.02)

    assert classify_executor_offer(runtime.root, reservation) == "matching_active"
    real_start = executor.start
    start_calls: list[str] = []

    def exited_without_result(request_id: str, **_kwargs):
        start_calls.append(request_id)
        return object()

    monkeypatch.setattr(executor, "start", exited_without_result)
    assert (
        controller.advance_scheduler_launch_authorizations(
            bindings,
            revision,
            reservations=[reservation],
        )
        == {}
    )
    launch_request = next(
        request for request in executor.unresolved_requests() if request.operation_kind == "scheduler_launch_authorize"
    )
    assert launch_request.provisional_offer_id == reservation.reservation_id
    assert start_calls == [launch_request.request_id]
    assert (
        controller.advance_scheduler_launch_authorizations(
            bindings,
            revision,
            reservations=[reservation],
        )
        == {}
    )
    assert start_calls == [launch_request.request_id]
    now[0] += 5.0
    monkeypatch.setattr(executor, "start", real_start)
    assert (
        controller.advance_scheduler_launch_authorizations(
            bindings,
            revision,
            reservations=[reservation],
        )
        == {}
    )

    real_classify = project_io_controller_module.classify_exact_reservation
    monkeypatch.setattr(
        project_io_controller_module,
        "classify_exact_reservation",
        lambda _root, _identity: "matching_released",
    )
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        executor.poll()
        if executor.load_result(launch_request.request_id) is not None:
            break
        time.sleep(0.02)
    assert (
        controller.advance_scheduler_launch_authorizations(
            bindings,
            revision,
            reservations=[reservation],
        )
        == {}
    )
    assert launch_request in executor.unresolved_requests()
    monkeypatch.setattr(project_io_controller_module, "classify_exact_reservation", real_classify)

    original_launch_id = executor.load_result(launch_request.request_id).evidence["launch_id"]
    if should_revalidate:
        # A binding can leave the resident set while a completed request is
        # awaiting consumption. The request must not block its revalidation.
        controller.advance_binding_validation([], revision)
        _advance_until_validated(controller, bindings, revision, binding.project_id)
        assert classify_executor_offer(runtime.root, reservation) == "matching_active"

    completions = {}
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        completions = controller.advance_scheduler_launch_authorizations(
            bindings,
            revision,
            reservations=[reservation],
        )
        if reservation.reservation_id in completions:
            break
        time.sleep(0.02)

    assert completions[reservation.reservation_id]["outcome"] == "authorized"
    assert completions[reservation.reservation_id]["launch_id"] == original_launch_id
    current = load_task(cfg, task.task_id)
    assert current.claim_control["active_claim"]["launch_state"] == "starting"
    assert executor.unresolved_requests() == ()
    assert (
        controller.advance_scheduler_launch_authorizations(
            bindings,
            revision,
            reservations=[reservation],
        )
        == {}
    )
    assert executor.unresolved_requests() == ()


def test_scheduler_claim_allocates_and_attaches_exact_cpu_offer(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    set_cpu_lane_capacity(runtime.root, capacity=2)
    cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    task = submit(
        cfg,
        ["echo", "ok"],
        requested_gpus=0,
        requested_cpus=2,
        working_dir=tmp_path,
    )
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    observations = _advance_until_observed(
        controller,
        bindings,
        revision,
        binding.project_id,
        lane="cpu",
    )

    controller.advance_scheduler_claims(
        bindings,
        revision,
        lane="cpu",
        admission_role="primary",
        observations=observations,
        available_gpu_ids=[],
        available_cpu_slots=2,
    )
    claim = next(request for request in executor.unresolved_requests() if request.operation_kind == "scheduler_claim")
    offer_identity = _claim_offer_identity(claim)

    deadline = time.monotonic() + 5.0
    completions = {}
    while time.monotonic() < deadline:
        completions = controller.advance_scheduler_claims(
            bindings,
            revision,
            lane="cpu",
            admission_role="primary",
            observations={},
            available_gpu_ids=[],
            available_cpu_slots=0,
        )
        if binding.project_id in completions:
            break
        time.sleep(0.02)

    assert completions[binding.project_id]["outcome"] == "claimed"
    assert completions[binding.project_id]["attempt_id"] == f"{task.task_id}-attempt-1"
    assert classify_executor_offer(runtime.root, offer_identity) == "matching_active"


def test_deferred_cpu_claims_preserve_callers_batch_capacity_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    set_cpu_lane_capacity(runtime.root, capacity=4)
    first_cfg, first, _revision, _bindings = _registered(tmp_path, "project-a", runtime)
    second_cfg, second, revision, bindings = _registered(tmp_path, "project-b", runtime)
    for cfg in (first_cfg, second_cfg):
        submit(cfg, ["echo", "ok"], requested_gpus=0, requested_cpus=1, working_dir=tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    try:
        for binding in (first, second):
            _advance_until_validated(controller, bindings, revision, binding.project_id)
        observations = {}
        for binding in (first, second):
            observations.update(_advance_until_observed(controller, bindings, revision, binding.project_id, lane="cpu"))
        monkeypatch.setattr(executor, "start", lambda _request_id, **_kwargs: None)
        controller.advance_scheduler_claims(
            bindings,
            revision,
            lane="cpu",
            admission_role="primary",
            observations=observations,
            available_gpu_ids=[],
            available_cpu_slots=1,
        )
        requests = executor.unresolved_requests()
        assert len(requests) == 1
        assert requests[0].parameters["offer"]["cpu_slots"] == 1
        assert sum(item["cpu_slots"] for item in reservation_snapshot(runtime.root).provisional) == 1
    finally:
        executor.shutdown()


def test_authority_service_observes_exact_running_attempt_without_shared_writes(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    task = submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    observations = _advance_until_observed(
        controller,
        bindings,
        revision,
        binding.project_id,
        lane="gpu",
    )
    controller.advance_scheduler_claims(
        bindings,
        revision,
        lane="gpu",
        admission_role="primary",
        observations=observations,
        available_gpu_ids=[0],
        available_cpu_slots=0,
    )
    completions = {}
    deadline = time.monotonic() + 5.0
    while binding.project_id not in completions and time.monotonic() < deadline:
        completions = controller.advance_scheduler_claims(
            bindings,
            revision,
            lane="gpu",
            admission_role="primary",
            observations={},
            available_gpu_ids=[],
            available_cpu_slots=0,
        )
        time.sleep(0.02)
    claim = completions[binding.project_id]
    attempt_file = attempt_path(cfg.shared_root, task.task_id, claim["attempt_number"])
    attempt = AttemptRecord.from_dict(read_json(attempt_file))
    attempt.phase = "running"
    attempt.process.update(
        {
            "wrapper_pid": os.getpid(),
            "wrapper_start_time_ticks": 0,
            "process_group_id": os.getpgrp(),
            "process_group_start_time_ticks": 0,
        }
    )
    atomic_replace(attempt_file, attempt.to_dict())
    current_task = load_task(cfg, task.task_id)
    current_task.claim_control["active_claim"]["launch_state"] = "running"
    save_task(cfg, current_task)
    runtime.set_enabled(binding.project_id, False)
    revision, bindings = runtime.load_registry()
    binding = next(item for item in bindings if item.project_id == binding.project_id)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    assert (
        controller.advance_scheduler_observations(
            bindings,
            revision,
            lane="gpu",
            admission_role="primary",
        )
        == {}
    )
    task_before = (cfg.shared_root / "tasks" / f"{task.task_id}.json").read_bytes()
    attempt_before = attempt_file.read_bytes()
    service = {
        binding.project_id: {
            "task_id": task.task_id,
            "attempt_id": attempt.attempt_id,
            "attempt_number": attempt.attempt_number,
            "fencing_token": attempt.current_fencing_token,
            "reservation_id": attempt.reservation_id,
            "process_identity": {
                field: attempt.process[field]
                for field in (
                    "wrapper_pid",
                    "wrapper_start_time_ticks",
                    "process_group_id",
                    "process_group_start_time_ticks",
                )
            },
        }
    }

    authority = {}
    deadline = time.monotonic() + 5.0
    while binding.project_id not in authority and time.monotonic() < deadline:
        authority = controller.advance_authority_services(bindings, revision, service)
        time.sleep(0.02)

    assert authority[binding.project_id]["outcome"] == "observed_current"
    assert authority[binding.project_id]["attempt_phase"] == "running"
    assert authority[binding.project_id]["authority_granted"] is False
    assert not authority[binding.project_id]["local_effects"]
    assert (cfg.shared_root / "tasks" / f"{task.task_id}.json").read_bytes() == task_before
    assert attempt_file.read_bytes() == attempt_before

    _other_cfg, other_binding, other_revision, other_bindings = _registered(tmp_path, "project-b", runtime)
    with pytest.raises(ValueError, match="provenance"):
        controller.advance_authority_renewals(
            other_bindings,
            other_revision,
            {other_binding.project_id: authority[binding.project_id]},
        )
    revision, bindings = runtime.load_registry()
    _advance_until_validated(controller, bindings, revision, binding.project_id)

    clock_root = runtime.project_paths(binding.project_id)["root"]
    atomic_replace(
        local_paths(clock_root)["clock_health"],
        {
            "clock_capability": {
                "status": "healthy",
                "reason": "healthy",
                "providers": ["linux_adjtimex"],
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "observation": {
                    "observation_id": "a" * 32,
                    "provider": "linux_adjtimex",
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "monotonic_observed_at": time.monotonic(),
                    "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip(),
                    "lower_error_seconds": -0.001,
                    "upper_error_seconds": 0.001,
                    "max_drift_rate": 0.0,
                    "provider_margin_seconds": 0.001,
                },
            }
        },
    )
    bounded_task = load_task(cfg, task.task_id)
    bounded_claim = bounded_task.claim_control["active_claim"]
    bounded_claim["authority_mode"] = "bounded_lease"
    bounded_claim["lease_expires_at"] = "2026-09-28T01:00:00Z"
    save_task(cfg, bounded_task)
    bounded_attempt = AttemptRecord.from_dict(read_json(attempt_file))
    bounded_attempt.authority_mode = "bounded_lease"
    bounded_attempt.lease["expires_at"] = bounded_claim["lease_expires_at"]
    atomic_replace(attempt_file, bounded_attempt.to_dict())

    authority = {}
    deadline = time.monotonic() + 5.0
    while binding.project_id not in authority and time.monotonic() < deadline:
        authority = controller.advance_authority_services(bindings, revision, service)
        time.sleep(0.02)
    renewal = {}
    deadline = time.monotonic() + 5.0
    while binding.project_id not in renewal and time.monotonic() < deadline:
        renewal = controller.advance_authority_renewals(bindings, revision, authority)
        time.sleep(0.02)

    assert renewal[binding.project_id]["outcome"] == "renewed"
    assert renewal[binding.project_id]["authority_granted"] is False
    assert not renewal[binding.project_id]["local_effects"]
    renewed_task = load_task(cfg, task.task_id)
    renewed_attempt = AttemptRecord.from_dict(read_json(attempt_file))
    renewal_id = renewed_task.claim_control["active_claim"]["clock_observation_id"]
    assert renewal_id == renewed_attempt.lease["clock_evidence"]["observation_id"]
    assert renewed_task.claim_control["active_claim"]["lease_expires_at"] == renewed_attempt.lease["expires_at"]

    renewed_task_bytes = (cfg.shared_root / "tasks" / f"{task.task_id}.json").read_bytes()
    renewed_attempt_bytes = attempt_file.read_bytes()
    stale_renewal = {}
    deadline = time.monotonic() + 5.0
    while binding.project_id not in stale_renewal and time.monotonic() < deadline:
        stale_renewal = controller.advance_authority_renewals(bindings, revision, authority)
        time.sleep(0.02)
    assert stale_renewal[binding.project_id]["outcome"] == "observed_stale"
    assert (cfg.shared_root / "tasks" / f"{task.task_id}.json").read_bytes() == renewed_task_bytes
    assert attempt_file.read_bytes() == renewed_attempt_bytes

    stale = {binding.project_id: {**service[binding.project_id], "fencing_token": attempt.current_fencing_token + 1}}
    authority = {}
    deadline = time.monotonic() + 5.0
    while binding.project_id not in authority and time.monotonic() < deadline:
        authority = controller.advance_authority_services(bindings, revision, stale)
        time.sleep(0.02)
    assert authority[binding.project_id]["outcome"] == "observed_stale"
    assert authority[binding.project_id]["reason"] == "task_changed"


@pytest.mark.parametrize(
    "lifecycle",
    ["manifest_deleted", "manifest_exited", "binding_removed", "binding_replaced", "registry_changed"],
)
@pytest.mark.parametrize("crash_after_task", [False, True], ids=["clock-only", "task-committed"])
def test_authority_renewal_replay_only_reconciles_stale_request_after_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lifecycle: str,
    crash_after_task: bool,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    task = submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    process_identity = {
        "wrapper_pid": 99_999_999,
        "wrapper_start_time_ticks": 0,
        "process_group_id": 99_999_998,
        "process_group_start_time_ticks": 0,
    }
    shared_attempt = AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task.task_id, 1)))
    shared_attempt.phase = "running"
    shared_attempt.authority_mode = "bounded_lease"
    shared_attempt.process.update(process_identity)
    shared_attempt.lease["expires_at"] = "2026-09-30T00:00:00Z"
    attempt_file = attempt_path(cfg.shared_root, task.task_id, 1)
    atomic_replace(attempt_file, shared_attempt.to_dict())
    shared_task = load_task(cfg, task.task_id)
    claim = shared_task.claim_control["active_claim"]
    claim.update(
        {
            "launch_state": "running",
            "authority_mode": "bounded_lease",
            "lease_expires_at": shared_attempt.lease["expires_at"],
        }
    )
    save_task(cfg, shared_task)
    project_runtime_root = runtime.project_paths(binding.project_id)["root"]
    cfg = replace(cfg, runtime_root=project_runtime_root)
    atomic_replace(
        local_paths(project_runtime_root)["clock_health"],
        {
            "clock_capability": {
                "status": "healthy",
                "reason": "healthy",
                "providers": ["linux_adjtimex"],
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "observation": {
                    "observation_id": "a" * 32,
                    "provider": "linux_adjtimex",
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "monotonic_observed_at": time.monotonic(),
                    "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip(),
                    "lower_error_seconds": -0.001,
                    "upper_error_seconds": 0.001,
                    "max_drift_rate": 0.0,
                    "provider_margin_seconds": 0.001,
                },
            }
        },
    )
    source_revisions = {
        "task": load_task(cfg, task.task_id).meta["revision"],
        "attempt_digest": hashlib.sha256(attempt_file.read_bytes()).hexdigest(),
    }
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    request = executor.prepare_authority_renewal(
        binding,
        revision,
        task_id=task.task_id,
        attempt_id=attempt.attempt_id,
        attempt_number=attempt.attempt_number,
        fencing_token=attempt.current_fencing_token,
        reservation_id=attempt.reservation_id,
        process_identity=process_identity,
        source_revisions=source_revisions,
    )
    task_before = (cfg.shared_root / "tasks" / f"{task.task_id}.json").read_bytes()
    attempt_before = attempt_file.read_bytes()
    fence_calls = 0
    fail_at = 3 if crash_after_task else 2

    def crash_during_commit() -> None:
        nonlocal fence_calls
        fence_calls += 1
        if fence_calls == fail_at:
            raise RuntimeError("simulated worker interruption")

    with pytest.raises(RuntimeError, match="simulated worker interruption"):
        renew_project_io_attempt_lease(
            cfg,
            request_id=request.request_id,
            task_id=task.task_id,
            attempt_id=attempt.attempt_id,
            attempt_number=attempt.attempt_number,
            fencing_token=attempt.current_fencing_token,
            reservation_id=attempt.reservation_id,
            process_identity=process_identity,
            expected_task_revision=source_revisions["task"],
            expected_attempt_digest=source_revisions["attempt_digest"],
            mutation_fence=crash_during_commit,
        )
    assert attempt_file.read_bytes() == attempt_before
    if crash_after_task:
        assert load_task(cfg, task.task_id).claim_control["active_claim"]["clock_observation_id"] == request.request_id
    else:
        assert (cfg.shared_root / "tasks" / f"{task.task_id}.json").read_bytes() == task_before

    # Model SIGKILL after the Task replace but before publish_result: the
    # durable process record proves the worker is now absent, while no result
    # record exists yet. A fresh executor must synthesize outcome_unknown and
    # retain the exact request for replay-only reconciliation.
    result_missing_after_task = crash_after_task and lifecycle == "manifest_exited"
    if result_missing_after_task:
        executor._write_record(
            executor._record_path("processes", request.request_id),
            ProjectIOProcess(
                request=request,
                pid=2_000_000_000,
                start_time_ticks=1,
                started_at=datetime.now(timezone.utc).isoformat(),
                state="running",
            ).to_dict(),
            "project_io_process",
        )
    else:
        unknown_result = ProjectIOResult(
            request=request,
            status="outcome_unknown",
            reason_code="project_io_outcome_unknown",
            completed_at=datetime.now(timezone.utc).isoformat(),
            evidence={},
        )
        executor._write_record(
            executor._record_path("results", request.request_id),
            unknown_result.to_dict(),
            "project_io_result",
        )
    manifest = local_paths(project_runtime_root)["processes"] / f"{attempt.attempt_id}.json"
    if lifecycle in {"manifest_deleted", "manifest_exited"}:
        atomic_replace(
            manifest,
            {
                "process": {
                    "protocol_version": 1,
                    "task_id": task.task_id,
                    "attempt_id": attempt.attempt_id,
                    "fencing_token": attempt.current_fencing_token,
                    "machine_name": binding.machine_name,
                    "observed_state": "running",
                    **process_identity,
                }
            },
        )
        if lifecycle == "manifest_deleted":
            manifest.unlink()
        else:
            process_manifest = read_json(manifest)
            process_manifest["process"]["supervision_marker"] = "preserve-me"
            atomic_replace(manifest, process_manifest)
            monkeypatch.setattr(
                project_io_supervision, "inspect_wrapper_identity", lambda _process: ProcessEvidence("absent")
            )
            monkeypatch.setattr(
                project_io_supervision, "inspect_local_group_identity", lambda _process: ProcessEvidence("absent")
            )

    if lifecycle in {"binding_removed", "binding_replaced"}:
        # Record an activation consumer first so normal removal retires that
        # generation and re-registration allocates a genuinely new one.
        register_consumer(
            cfg.shared_root,
            runtime_id=runtime.instance_id,
            project_id=binding.project_id,
            registration_generation=binding.registration_generation,
            process_fence="test-process-fence",
        )
        runtime.set_enabled(binding.project_id, False)
        runtime.remove_binding(binding.project_id)
        if lifecycle == "binding_replaced":
            replacement = runtime.add_binding(cfg.shared_root, cfg.machine_name)
            assert replacement.registration_generation != binding.registration_generation
            revision, bindings = runtime.load_registry()
        else:
            revision, bindings = runtime.load_registry()
            assert not bindings
    elif lifecycle == "registry_changed":
        _registered(tmp_path, "unrelated-project", runtime)
        revision, bindings = runtime.load_registry()
        assert revision != request.registry_revision

    # A fresh executor/controller models restart. The old ambiguous request is
    # retained across the new epoch and replayed only in recovery mode.
    restarted_executor = ProjectIOExecutor(runtime)
    restarted_executor.begin_epoch()
    if result_missing_after_task:
        assert not restarted_executor._reconciliation_unknown
        assert not restarted_executor._record_path("processes", request.request_id).exists()
        synthesized = restarted_executor.load_result(request.request_id)
        assert synthesized is not None
        assert synthesized.status == "outcome_unknown"
    assert request in restarted_executor.unresolved_requests()
    controller_clock = [0.0]
    controller = ProjectIOController(runtime, restarted_executor, monotonic=lambda: controller_clock[0])

    inject_transient_replay_failure = crash_after_task and lifecycle == "registry_changed"
    if inject_transient_replay_failure:
        worker_pid = os.getpid()
        worker_ticks = project_io_worker_module._process_start_time_ticks(worker_pid)
        assert worker_ticks is not None
        live_process = ProjectIOProcess(
            request=request,
            pid=worker_pid,
            start_time_ticks=worker_ticks,
            started_at=datetime.now(timezone.utc).isoformat(),
            state="running",
        )
        restarted_executor._write_record(
            restarted_executor._record_path("processes", request.request_id),
            live_process.to_dict(),
            "project_io_process",
        )

        def fail_first_replay(*_args, **_kwargs):
            raise OSError("injected transient replay read failure")

        with monkeypatch.context() as replay_failure:
            replay_failure.setattr(project_io_worker_module, "_authority_renewal", fail_first_replay)
            assert (
                project_io_worker_module._run(
                    runtime.root,
                    request.request_id,
                    reconcile_authority_renewal=True,
                )
                == 0
            )
        failed_replay = restarted_executor.load_result(request.request_id)
        assert failed_replay is not None
        assert failed_replay.status == "retryable_error"
        assert failed_replay.reason_code == "project_io_authority_renewal_failed"
        restarted_executor._write_record(
            restarted_executor._record_path("processes", request.request_id),
            ProjectIOProcess(
                request=request,
                pid=2_000_000_000,
                start_time_ticks=1,
                started_at=live_process.started_at,
                state="running",
            ).to_dict(),
            "project_io_process",
        )
        project_io_supervision.advance_authority_renewals(runtime, controller, bindings, revision)
        retained = restarted_executor.load_result(request.request_id)
        assert retained is not None
        assert retained.status == "retryable_error"
        assert request in restarted_executor.unresolved_requests()
        controller_clock[0] = 5.1

    if crash_after_task:
        replay_fence_calls = 0

        def interrupt_replay_before_attempt() -> None:
            nonlocal replay_fence_calls
            replay_fence_calls += 1
            raise RuntimeError("simulated replay interruption")

        with pytest.raises(RuntimeError, match="simulated replay interruption"):
            renew_project_io_attempt_lease(
                cfg,
                request_id=request.request_id,
                task_id=task.task_id,
                attempt_id=attempt.attempt_id,
                attempt_number=attempt.attempt_number,
                fencing_token=attempt.current_fencing_token,
                reservation_id=attempt.reservation_id,
                process_identity=process_identity,
                expected_task_revision=source_revisions["task"],
                expected_attempt_digest=source_revisions["attempt_digest"],
                mutation_fence=interrupt_replay_before_attempt,
                replay_only=True,
            )
        assert replay_fence_calls == 1

    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        project_io_supervision.advance_authority_renewals(runtime, controller, bindings, revision)
        if not restarted_executor.unresolved_requests():
            break
        time.sleep(0.02)
    else:
        raise AssertionError("stale authority renewal remained unresolved after exact reconciliation")

    final_task = load_task(cfg, task.task_id)
    final_attempt = AttemptRecord.from_dict(read_json(attempt_file))
    if crash_after_task:
        assert final_task.claim_control["active_claim"]["clock_observation_id"] == request.request_id
        assert final_attempt.lease["clock_evidence"]["observation_id"] == request.request_id
        assert final_attempt.lease["expires_at"] == final_task.claim_control["active_claim"]["lease_expires_at"]
    else:
        assert (cfg.shared_root / "tasks" / f"{task.task_id}.json").read_bytes() == task_before
        assert attempt_file.read_bytes() == attempt_before
    if lifecycle == "manifest_deleted":
        assert not manifest.exists()
    elif lifecycle == "manifest_exited":
        final_manifest = read_json(manifest)["process"]
        assert final_manifest["supervision_marker"] == "preserve-me"
        assert final_manifest.get("authority_state") != "healthy"
    assert restarted_executor.unresolved_requests() == ()
    restarted_executor.shutdown()


@pytest.mark.parametrize("entry_point", ["supervision", "dispatch"])
def test_isolated_dispatch_authority_router_renews_from_local_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_point: str,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    task = submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    process_identity = {
        "wrapper_pid": 101,
        "wrapper_start_time_ticks": 202,
        "process_group_id": 303,
        "process_group_start_time_ticks": 404,
    }
    attempt_file = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    stored_attempt = AttemptRecord.from_dict(read_json(attempt_file))
    stored_attempt.phase = "running"
    stored_attempt.authority_mode = "bounded_lease"
    stored_attempt.process.update(process_identity)
    stored_attempt.lease["expires_at"] = (
        (datetime.now(timezone.utc) + timedelta(minutes=5)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    atomic_replace(attempt_file, stored_attempt.to_dict())
    stored_task = load_task(cfg, task.task_id)
    claim = stored_task.claim_control["active_claim"]
    claim.update(
        {
            "launch_state": "running",
            "authority_mode": "bounded_lease",
            "lease_expires_at": stored_attempt.lease["expires_at"],
        }
    )
    save_task(cfg, stored_task)

    project_runtime = runtime.project_paths(binding.project_id)["root"]
    atomic_replace(
        local_paths(project_runtime)["clock_health"],
        {
            "clock_capability": {
                "status": "healthy",
                "reason": "healthy",
                "providers": ["linux_adjtimex"],
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "observation": {
                    "observation_id": "a" * 32,
                    "provider": "linux_adjtimex",
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "monotonic_observed_at": time.monotonic(),
                    "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip(),
                    "lower_error_seconds": -0.001,
                    "upper_error_seconds": 0.001,
                    "max_drift_rate": 0.0,
                    "provider_margin_seconds": 0.001,
                },
            }
        },
    )
    manifest = local_paths(project_runtime)["processes"] / f"{attempt.attempt_id}.json"
    atomic_replace(
        manifest,
        {
            "process": {
                "protocol_version": 1,
                "task_id": task.task_id,
                "attempt_id": attempt.attempt_id,
                "fencing_token": attempt.current_fencing_token,
                "reservation_id": attempt.reservation_id,
                "machine_name": binding.machine_name,
                "authority_mode": "bounded_lease",
                "observed_state": "running",
                "custom_marker": {"preserve": True},
                "lease_expires_at": stored_attempt.lease["expires_at"],
                **process_identity,
            }
        },
    )
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    if entry_point == "dispatch":
        runtime.project_io_executor = executor
        runtime.project_io_controller = controller
    monkeypatch.setattr(project_io_supervision, "inspect_wrapper_identity", lambda _process: ProcessEvidence("alive"))
    monkeypatch.setattr(
        project_io_supervision, "inspect_local_group_identity", lambda _process: ProcessEvidence("alive")
    )

    def advance_with_local_io_only() -> None:
        # Guard only the controller process. Fresh worker interpreters must still
        # read and mutate the Project to complete the actual lease transaction.
        def reject_project_access(operation):
            def checked(path, *args, **kwargs):
                if isinstance(path, (str, bytes, os.PathLike)):
                    candidate = Path(os.path.abspath(os.fsdecode(path)))
                    assert not candidate.is_relative_to(cfg.shared_root), (
                        f"supervision coordinator accessed shared storage: {candidate}"
                    )
                return operation(path, *args, **kwargs)

            return checked

        with monkeypatch.context() as local_io:
            for module, name in (
                (builtins, "open"),
                (io, "open"),
                (os, "open"),
                (os, "stat"),
                (os, "lstat"),
                (os, "scandir"),
                (os, "listdir"),
                (os, "readlink"),
            ):
                local_io.setattr(module, name, reject_project_access(getattr(module, name)))
            if entry_point == "dispatch":
                dispatch_machine_cycle_locked(runtime, available_gpus=[], publish_snapshots=False)
            else:
                project_io_supervision.advance_authority_renewals(runtime, controller, bindings, revision)

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        advance_with_local_io_only()
        local_process = read_json(manifest)["process"]
        if local_process.get("authority_state") == "healthy":
            break
        time.sleep(0.02)
    else:
        unresolved = executor.unresolved_requests()
        results = {
            request.request_id: (
                None
                if executor.load_result(request.request_id) is None
                else executor.load_result(request.request_id).to_dict()
            )
            for request in unresolved
        }
        resolved = [read_json(path) for path in executor.paths["project_io_resolved"].glob("*.json")]
        raise AssertionError(
            "isolated authority router did not renew the local process: "
            f"manifest={read_json(manifest)!r}, unresolved={unresolved!r}, "
            f"results={results!r}, resolved={resolved!r}"
        )

    renewed_task = load_task(cfg, task.task_id)
    renewed_attempt = AttemptRecord.from_dict(read_json(attempt_file))
    assert renewed_task.claim_control["active_claim"]["clock_observation_id"]
    assert (
        renewed_task.claim_control["active_claim"]["clock_observation_id"]
        == renewed_attempt.lease["clock_evidence"]["observation_id"]
    )
    assert local_process["lease_expires_at"] == renewed_attempt.lease["expires_at"]
    assert local_process["custom_marker"] == {"preserve": True}
    executor.shutdown()


def test_isolated_authority_router_pins_one_manifest_until_definitive_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    process_root = local_paths(runtime.project_paths(binding.project_id)["root"])["processes"]
    for task_id, token in (("task-a", 1), ("task-b", 2)):
        attempt_id = f"{task_id}-attempt-1"
        atomic_replace(
            process_root / f"{attempt_id}.json",
            {
                "process": {
                    "protocol_version": 1,
                    "task_id": task_id,
                    "attempt_id": attempt_id,
                    "fencing_token": token,
                    "machine_name": binding.machine_name,
                    "observed_state": "running",
                    "wrapper_pid": token * 10 + 1,
                    "wrapper_start_time_ticks": token * 10 + 2,
                    "process_group_id": token * 10 + 3,
                    "process_group_start_time_ticks": token * 10 + 4,
                }
            },
        )

    runtime_id = ProjectIOExecutor(runtime).runtime_id
    monkeypatch.setattr(project_io_supervision, "inspect_wrapper_identity", lambda _process: ProcessEvidence("alive"))
    monkeypatch.setattr(
        project_io_supervision, "inspect_local_group_identity", lambda _process: ProcessEvidence("alive")
    )

    class FakeController:
        _observed_executor_epoch = "e" * 32

        def __init__(self) -> None:
            self.executor = type(
                "FakeExecutor",
                (),
                {"runtime_id": runtime_id, "unresolved_requests": lambda _self: ()},
            )()
            self.service_attempts: list[str] = []

        def advance_authority_services(self, _bindings, _revision, attempts):
            if not attempts:
                return {}
            project_id, parameters = next(iter(attempts.items()))
            self.service_attempts.append(parameters["attempt_id"])
            if len(self.service_attempts) == 1:
                return {}
            return {
                project_id: {
                    "outcome": "observed_current",
                    "reason": None,
                    "runtime_id": runtime_id,
                    "executor_epoch": self._observed_executor_epoch,
                    "project_id": project_id,
                    "canonical_shared_root": str(binding.shared_root),
                    "registration_generation": binding.registration_generation,
                    "registry_revision": revision,
                    "machine_name": binding.machine_name,
                    "service_action": "observe_current_attempt",
                    **parameters,
                    "source_revisions": {"task": 1, "attempt_digest": "a" * 64},
                    "attempt_phase": "running",
                    "authority_granted": False,
                    "local_effects": [],
                }
            }

        def advance_authority_renewals(self, _bindings, _revision, observations):
            return {
                project_id: {
                    "outcome": "renewed",
                    "lease_expires_at": "2026-09-29T00:02:00Z",
                    "renew_after_seconds": 10.0,
                }
                for project_id in observations
            }

    controller = FakeController()
    for _ in range(3):
        project_io_supervision.advance_authority_renewals(runtime, controller, bindings, revision)

    assert controller.service_attempts[0] == controller.service_attempts[1]
    assert controller.service_attempts[2] != controller.service_attempts[1]
    for _ in range(4):
        project_io_supervision.advance_authority_renewals(runtime, controller, bindings, revision)
    assert len(controller.service_attempts) == 3
    assert {path.stem for path in process_root.glob("*.json")} == {
        "task-a-attempt-1",
        "task-b-attempt-1",
    }


def test_isolated_authority_router_rehydrates_unresolved_owner_outside_next_sixty_four(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, target, revision, _bindings = _registered(tmp_path, "project-a", runtime)
    target = replace(target, enabled=False)
    fake_bindings = [replace(target, project_id=f"fake-{index:03d}") for index in range(64)]
    bindings = [*fake_bindings, target]
    parameters = {
        "task_id": "task-a",
        "attempt_id": "task-a-attempt-1",
        "attempt_number": 1,
        "fencing_token": 1,
        "reservation_id": None,
        "process_identity": {
            "wrapper_pid": 11,
            "wrapper_start_time_ticks": 12,
            "process_group_id": 13,
            "process_group_start_time_ticks": 14,
        },
    }
    manifest = runtime.project_paths(target.project_id)["processes"] / "task-a-attempt-1.json"
    atomic_replace(
        manifest,
        {
            "process": {
                "protocol_version": 1,
                "machine_name": target.machine_name,
                "observed_state": "running",
                **parameters,
                **parameters["process_identity"],
            }
        },
    )
    process = read_json(manifest)["process"]
    process.pop("attempt_number")
    process.pop("reservation_id")
    process.pop("process_identity")
    atomic_replace(manifest, {"process": process})
    monkeypatch.setattr(project_io_supervision, "inspect_wrapper_identity", lambda _process: ProcessEvidence("alive"))
    monkeypatch.setattr(
        project_io_supervision, "inspect_local_group_identity", lambda _process: ProcessEvidence("alive")
    )
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    request = executor.prepare_authority_service(target, revision, **parameters)

    class FakeController:
        def __init__(self) -> None:
            self.executor = executor
            self._observed_executor_epoch = request.executor_epoch
            self.selected: list[str] = []

        def advance_authority_services(self, selected_bindings, _revision, _attempts):
            self.selected = [binding.project_id for binding in selected_bindings]
            return {}

        def advance_authority_renewals(self, *_args, **_kwargs):
            observations = _args[2]
            if observations:
                raise AssertionError("no renewal observation exists")
            return {}

    controller = FakeController()
    runtime.authority_process_offset = 0
    project_io_supervision.advance_authority_renewals(runtime, controller, bindings, revision)

    assert target.project_id in controller.selected
    assert len(controller.selected) <= 64
    assert runtime.authority_process_intents[target.project_id]["parameters"] == parameters


def test_isolated_authority_scan_state_is_bounded_and_reaches_tail(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    revision, _bindings = runtime.load_registry()
    bindings = [replace(binding, project_id=f"project-{index:03d}") for index in range(300)]
    tail_processes = runtime.project_paths(bindings[-1].project_id)["processes"]
    tail_processes.mkdir(parents=True, exist_ok=True)
    atomic_replace(tail_processes / "invalid-a.json", {"unexpected": {}})
    atomic_replace(tail_processes / "invalid-b.json", {"unexpected": {}})
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    try:
        for _ in range(5):
            project_io_supervision.advance_authority_renewals(runtime, controller, bindings, revision)

        tail_signature = project_io_supervision._authority_binding_signature(bindings[-1], revision)
        assert tail_signature in runtime.authority_process_scans
        assert len(runtime.authority_process_scans) <= project_io_supervision._MAX_AUTHORITY_SCAN_STATES
    finally:
        for scan in runtime.authority_process_scans.values():
            scan.close()
        executor.shutdown()


def test_running_publication_scan_reaches_tail_pages_beyond_state_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    revision, _bindings = runtime.load_registry()
    bindings = [replace(binding, project_id=f"project-{index:03d}") for index in range(300)]
    for candidate in bindings:
        registrations = runtime.project_paths(candidate.project_id)["registrations"]
        for index in range(3):
            atomic_replace(registrations / f"bad-{index}.json", {"unexpected": {}})

    tail_registrations = runtime.project_paths(bindings[-1].project_id)["registrations"]
    tail_paths: list[Path] = []
    original = project_io_supervision._read_running_publication_source

    def record_tail(path, *args, **kwargs):
        if path.parent == tail_registrations:
            tail_paths.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(project_io_supervision, "_read_running_publication_source", record_tail)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    try:
        for _ in range(40):
            coordinator.advance_running_publications(bindings, revision)

        assert len(set(tail_paths)) >= 2
        assert len(coordinator._scans) <= 256
    finally:
        coordinator.close()
        executor.shutdown()


def test_isolated_authority_due_keys_are_bounded_and_prune_attempt_identity_churn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    process_root = local_paths(runtime.project_paths(binding.project_id)["root"])["processes"]
    attempt_id = "task-a-attempt-1"
    manifest = process_root / f"{attempt_id}.json"
    process_identity = {
        "wrapper_pid": 21,
        "wrapper_start_time_ticks": 22,
        "process_group_id": 23,
        "process_group_start_time_ticks": 24,
    }
    atomic_replace(
        manifest,
        {
            "process": {
                "protocol_version": 1,
                "task_id": "task-a",
                "attempt_id": attempt_id,
                "fencing_token": 1,
                "machine_name": binding.machine_name,
                "observed_state": "running",
                **process_identity,
            }
        },
    )
    monkeypatch.setattr(project_io_supervision, "inspect_wrapper_identity", lambda _process: ProcessEvidence("alive"))
    monkeypatch.setattr(
        project_io_supervision, "inspect_local_group_identity", lambda _process: ProcessEvidence("alive")
    )
    prefix = project_io_supervision._authority_due_binding_prefix(binding)
    old_identity_key = (*prefix, "task-a", attempt_id, "1", "1", "", "11", "12", "13", "14")
    runtime.authority_process_next_due[old_identity_key] = time.monotonic() + 60.0

    class FakeController:
        _observed_executor_epoch = "e" * 32

        def __init__(self) -> None:
            self.executor = type(
                "FakeExecutor",
                (),
                {"runtime_id": "r" * 32, "unresolved_requests": lambda _self: ()},
            )()
            self.attempts: list[dict[str, object]] = []

        def advance_authority_services(self, _bindings, _revision, attempts):
            self.attempts.append(dict(attempts))
            return {}

        def advance_authority_renewals(self, *_args, **_kwargs):
            return {}

    controller = FakeController()
    project_io_supervision.advance_authority_renewals(runtime, controller, bindings, revision)
    assert old_identity_key not in runtime.authority_process_next_due

    now = time.monotonic()
    for index in range(
        project_io_supervision._MAX_AUTHORITY_DUE_KEYS - project_io_supervision._AUTHORITY_DUE_INFLIGHT_RESERVE
    ):
        task_id = f"old-{index}"
        runtime.authority_process_next_due[
            (*prefix, task_id, f"{task_id}-attempt-1", "1", "1", "", "11", "12", "13", "14")
        ] = now + 60.0
    project_io_supervision.advance_authority_renewals(runtime, controller, bindings, revision)

    assert controller.attempts[-1] == {}
    assert len(runtime.authority_process_next_due) <= project_io_supervision._MAX_AUTHORITY_DUE_KEYS


@pytest.mark.parametrize("case", ["dead", "reused", "exit", "special_exit"])
def test_isolated_authority_manifest_requires_live_process_without_exit_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision, _bindings = _registered(tmp_path, "project-a", runtime)
    paths = runtime.project_paths(binding.project_id)
    attempt_id = "task-a-attempt-1"
    manifest = paths["processes"] / f"{attempt_id}.json"
    atomic_replace(
        manifest,
        {
            "process": {
                "protocol_version": 1,
                "task_id": "task-a",
                "attempt_id": attempt_id,
                "fencing_token": 1,
                "machine_name": binding.machine_name,
                "observed_state": "running",
                "wrapper_pid": 11,
                "wrapper_start_time_ticks": 12,
                "process_group_id": 13,
                "process_group_start_time_ticks": 14,
            }
        },
    )
    if case == "dead":
        wrapper = group = ProcessEvidence("absent")
    elif case == "reused":
        wrapper = ProcessEvidence("unknown", "identity_mismatch")
        group = ProcessEvidence("absent")
    else:
        wrapper = group = ProcessEvidence("alive")
        observation = paths["observations"] / f"{attempt_id}.json"
        if case == "exit":
            atomic_replace(observation, {"exit_observation": {"attempt_id": attempt_id}})
        else:
            observation.parent.mkdir(parents=True, exist_ok=True)
            os.mkfifo(observation)
    monkeypatch.setattr(project_io_supervision, "inspect_wrapper_identity", lambda _process: wrapper)
    monkeypatch.setattr(project_io_supervision, "inspect_local_group_identity", lambda _process: group)

    result = project_io_supervision._read_isolated_authority_manifest(
        runtime,
        binding,
        manifest,
        project_io_supervision._authority_binding_signature(binding, revision),
    )

    assert result is None


def test_scheduler_no_claim_releases_offer_only_after_definitive_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    task = submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    now = [100.0]
    controller = ProjectIOController(runtime, executor, monotonic=lambda: now[0])
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    observations = _advance_until_observed(
        controller,
        bindings,
        revision,
        binding.project_id,
        lane="gpu",
    )
    real_start = executor.start
    monkeypatch.setattr(executor, "start", lambda _request_id, **_kwargs: None)
    controller.advance_scheduler_claims(
        bindings,
        revision,
        lane="gpu",
        admission_role="primary",
        observations=observations,
        available_gpu_ids=[0],
        available_cpu_slots=0,
    )
    claim = next(request for request in executor.unresolved_requests() if request.operation_kind == "scheduler_claim")
    offer_identity = _claim_offer_identity(claim)
    assert classify_executor_offer(runtime.root, offer_identity) == "matching_provisional"

    changed = load_task(cfg, task.task_id)
    changed.meta["revision"] += 1
    save_task(cfg, changed)
    monkeypatch.setattr(executor, "start", real_start)
    now[0] += 5.0
    deadline = time.monotonic() + 5.0
    completions = {}
    while time.monotonic() < deadline:
        completions = controller.advance_scheduler_claims(
            bindings,
            revision,
            lane="gpu",
            admission_role="primary",
            observations={},
            available_gpu_ids=[],
            available_cpu_slots=0,
        )
        if binding.project_id in completions:
            break
        time.sleep(0.02)

    assert completions[binding.project_id]["outcome"] == "no_claim"
    assert completions[binding.project_id]["reason"] == "task_changed"
    assert classify_executor_offer(runtime.root, offer_identity) == "matching_released"
    assert executor.unresolved_requests() == ()


def test_scheduler_claim_result_replays_after_controller_restart(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    observations = _advance_until_observed(
        controller,
        bindings,
        revision,
        binding.project_id,
        lane="gpu",
    )
    controller.advance_scheduler_claims(
        bindings,
        revision,
        lane="gpu",
        admission_role="primary",
        observations=observations,
        available_gpu_ids=[0],
        available_cpu_slots=0,
    )
    claim = next(request for request in executor.unresolved_requests() if request.operation_kind == "scheduler_claim")
    offer_identity = _claim_offer_identity(claim)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        executor.poll()
        if executor.load_result(claim.request_id) is not None and executor.status_view()["active_worker_count"] == 0:
            break
        time.sleep(0.02)
    assert classify_executor_offer(runtime.root, offer_identity) == "matching_provisional"

    restarted = ProjectIOController(runtime, executor)
    completions = restarted.advance_scheduler_claims(
        bindings,
        revision,
        lane="gpu",
        admission_role="primary",
        observations={},
        available_gpu_ids=[],
        available_cpu_slots=0,
    )

    assert completions == {}
    assert classify_executor_offer(runtime.root, offer_identity) == "matching_active"
    assert executor.unresolved_requests() == ()


def test_scheduler_claim_prepare_failure_releases_unpublished_offer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    observations = _advance_until_observed(
        controller,
        bindings,
        revision,
        binding.project_id,
        lane="gpu",
    )

    monkeypatch.setattr(
        executor,
        "prepare_scheduler_claim",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("synthetic prepare failure")),
    )
    assert (
        controller.advance_scheduler_claims(
            bindings,
            revision,
            lane="gpu",
            admission_role="primary",
            observations=observations,
            available_gpu_ids=[0],
            available_cpu_slots=0,
        )
        == {}
    )

    assert executor.unresolved_requests() == ()
    assert not any(runtime.paths["provisional"].iterdir())


def test_scheduler_claim_rejects_observation_from_previous_executor_epoch(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    stale = _advance_until_observed(
        controller,
        bindings,
        revision,
        binding.project_id,
        lane="gpu",
    )

    executor.fence_epoch()
    executor.begin_epoch()
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    assert (
        controller.advance_scheduler_claims(
            bindings,
            revision,
            lane="gpu",
            admission_role="primary",
            observations=stale,
            available_gpu_ids=[0],
            available_cpu_slots=0,
        )
        == {}
    )
    assert executor.unresolved_requests() == ()
    assert not any(runtime.paths["provisional"].iterdir())


@pytest.mark.parametrize("ready_state", ["active", "degraded"])
def test_scheduler_claim_rechecks_ready_index_source_revision(tmp_path: Path, ready_state: str) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    task = submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    observations = _advance_until_observed(
        controller,
        bindings,
        revision,
        binding.project_id,
        lane="gpu",
    )
    state_path = ready_state_path(cfg.shared_root)
    state = read_json(state_path)
    state["ready_index"]["state"] = ready_state
    state["ready_index"]["revision"] += 1
    atomic_replace(state_path, state)

    controller.advance_scheduler_claims(
        bindings,
        revision,
        lane="gpu",
        admission_role="primary",
        observations=observations,
        available_gpu_ids=[0],
        available_cpu_slots=0,
    )
    claim = next(request for request in executor.unresolved_requests() if request.operation_kind == "scheduler_claim")
    offer_identity = _claim_offer_identity(claim)
    completions = {}
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        completions = controller.advance_scheduler_claims(
            bindings,
            revision,
            lane="gpu",
            admission_role="primary",
            observations={},
            available_gpu_ids=[],
            available_cpu_slots=0,
        )
        if binding.project_id in completions:
            break
        time.sleep(0.02)

    assert completions[binding.project_id]["outcome"] == "no_claim"
    assert completions[binding.project_id]["reason"] == "ready_changed"
    assert classify_executor_offer(runtime.root, offer_identity) == "matching_released"

    # An index revision change must reject stale evidence without skipping the
    # still-queued candidate, including after a fresh controller takes over.
    state["ready_index"]["state"] = "active"
    atomic_replace(state_path, state)
    resumed = ProjectIOController(runtime, executor)
    _advance_until_validated(resumed, bindings, revision, binding.project_id)
    refreshed = _advance_until_observed(resumed, bindings, revision, binding.project_id, lane="gpu")
    assert refreshed[binding.project_id]["candidate"] is not None
    assert refreshed[binding.project_id]["candidate"]["task_id"] == task.task_id
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        completions = resumed.advance_scheduler_claims(
            bindings,
            revision,
            lane="gpu",
            admission_role="primary",
            observations=refreshed,
            available_gpu_ids=[0],
            available_cpu_slots=0,
        )
        if binding.project_id in completions:
            break
        time.sleep(0.02)
    assert completions[binding.project_id]["outcome"] == "claimed"
    executor.shutdown()


def test_completed_claim_reconciles_after_unrelated_registry_revision_change(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    observations = _advance_until_observed(
        controller,
        bindings,
        revision,
        binding.project_id,
        lane="gpu",
    )
    controller.advance_scheduler_claims(
        bindings,
        revision,
        lane="gpu",
        admission_role="primary",
        observations=observations,
        available_gpu_ids=[0],
        available_cpu_slots=0,
    )
    claim = next(request for request in executor.unresolved_requests() if request.operation_kind == "scheduler_claim")
    offer_identity = _claim_offer_identity(claim)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        executor.poll()
        if executor.load_result(claim.request_id) is not None and executor.status_view()["active_worker_count"] == 0:
            break
        time.sleep(0.02)
    _other_cfg, _other, current_revision, current_bindings = _registered(tmp_path, "project-b", runtime)

    completions = controller.advance_scheduler_claims(
        current_bindings,
        current_revision,
        lane="gpu",
        admission_role="primary",
        observations={},
        available_gpu_ids=[],
        available_cpu_slots=0,
    )

    assert completions == {}
    assert classify_executor_offer(runtime.root, offer_identity) == "matching_active"
    assert executor.unresolved_requests() == ()


def test_scheduler_claim_worker_exit_retries_same_request_and_offer(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    task = submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    now = [100.0]
    controller = ProjectIOController(runtime, executor, monotonic=lambda: now[0])
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    observations = _advance_until_observed(
        controller,
        bindings,
        revision,
        binding.project_id,
        lane="gpu",
    )
    task_path = shared_paths(cfg.shared_root)["tasks"] / f"{task.task_id}.json"
    original = task_path.read_bytes()
    task_path.unlink()
    os.mkfifo(task_path)
    controller.advance_scheduler_claims(
        bindings,
        revision,
        lane="gpu",
        admission_role="primary",
        observations=observations,
        available_gpu_ids=[0],
        available_cpu_slots=0,
    )
    claim = next(request for request in executor.unresolved_requests() if request.operation_kind == "scheduler_claim")
    offer_identity = _claim_offer_identity(claim)
    process = executor._load_process(claim.request_id)
    os.kill(process.pid, 9)
    task_path.unlink()
    task_path.write_bytes(original)

    controller.advance_scheduler_claims(
        bindings,
        revision,
        lane="gpu",
        admission_role="primary",
        observations={},
        available_gpu_ids=[],
        available_cpu_slots=0,
    )
    assert classify_executor_offer(runtime.root, offer_identity) == "matching_provisional"
    now[0] += 5.0
    completions = {}
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        completions = controller.advance_scheduler_claims(
            bindings,
            revision,
            lane="gpu",
            admission_role="primary",
            observations={},
            available_gpu_ids=[],
            available_cpu_slots=0,
        )
        if binding.project_id in completions:
            break
        time.sleep(0.02)

    assert completions[binding.project_id]["outcome"] == "claimed"
    assert executor.unresolved_requests() == ()
    assert classify_executor_offer(runtime.root, offer_identity) == "matching_active"


def test_fenced_epoch_reconciles_dead_claim_worker_before_releasing_offer(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    task = submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    now = [100.0]
    controller = ProjectIOController(runtime, executor, monotonic=lambda: now[0])
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    observations = _advance_until_observed(
        controller,
        bindings,
        revision,
        binding.project_id,
        lane="gpu",
    )
    task_path = shared_paths(cfg.shared_root)["tasks"] / f"{task.task_id}.json"
    original = task_path.read_bytes()
    task_path.unlink()
    os.mkfifo(task_path)
    controller.advance_scheduler_claims(
        bindings,
        revision,
        lane="gpu",
        admission_role="primary",
        observations=observations,
        available_gpu_ids=[0],
        available_cpu_slots=0,
    )
    claim = next(request for request in executor.unresolved_requests() if request.operation_kind == "scheduler_claim")
    offer_identity = _claim_offer_identity(claim)
    process = executor._load_process(claim.request_id)
    os.kill(process.pid, 9)
    task_path.unlink()
    task_path.write_bytes(original)
    deadline = time.monotonic() + 5.0
    while executor._children[claim.request_id].poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    executor.begin_epoch()
    restarted = ProjectIOController(runtime, executor, monotonic=lambda: now[0])

    restarted.advance_scheduler_claims(
        bindings,
        revision,
        lane="gpu",
        admission_role="primary",
        observations={},
        available_gpu_ids=[],
        available_cpu_slots=0,
    )
    assert classify_executor_offer(runtime.root, offer_identity) == "matching_provisional"
    now[0] += 5.0
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        restarted.advance_scheduler_claims(
            bindings,
            revision,
            lane="gpu",
            admission_role="primary",
            observations={},
            available_gpu_ids=[],
            available_cpu_slots=0,
        )
        if not executor.unresolved_requests():
            break
        now[0] += 5.0
        time.sleep(0.02)

    assert executor.unresolved_requests() == ()
    assert classify_executor_offer(runtime.root, offer_identity) == "matching_released"


def test_disabled_binding_reconciles_ambiguous_claim_without_epoch_rollover(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    task = submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    now = [100.0]
    controller = ProjectIOController(runtime, executor, monotonic=lambda: now[0])
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    observations = _advance_until_observed(
        controller,
        bindings,
        revision,
        binding.project_id,
        lane="gpu",
    )
    task_record_path = shared_paths(cfg.shared_root)["tasks"] / f"{task.task_id}.json"
    original = task_record_path.read_bytes()
    task_record_path.unlink()
    os.mkfifo(task_record_path)
    controller.advance_scheduler_claims(
        bindings,
        revision,
        lane="gpu",
        admission_role="primary",
        observations=observations,
        available_gpu_ids=[0],
        available_cpu_slots=0,
    )
    claim = next(request for request in executor.unresolved_requests() if request.operation_kind == "scheduler_claim")
    offer_identity = _claim_offer_identity(claim)
    process = executor._load_process(claim.request_id)
    os.kill(process.pid, 9)
    task_record_path.unlink()
    task_record_path.write_bytes(original)
    deadline = time.monotonic() + 5.0
    while executor._children[claim.request_id].poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    runtime.set_enabled(binding.project_id, False)
    current_revision, current_bindings = runtime.load_registry()

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        now[0] += 5.0
        controller.advance_scheduler_claims(
            current_bindings,
            current_revision,
            lane="gpu",
            admission_role="primary",
            observations={},
            available_gpu_ids=[],
            available_cpu_slots=0,
        )
        if not executor.unresolved_requests():
            break
        time.sleep(0.02)

    assert executor.unresolved_requests() == ()
    assert classify_executor_offer(runtime.root, offer_identity) == "matching_released"


def test_terminal_attempt_cannot_reactivate_ambiguous_executor_offer(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision, bindings = _registered(tmp_path, "project-a", runtime)
    task = submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    now = [100.0]
    controller = ProjectIOController(runtime, executor, monotonic=lambda: now[0])
    _advance_until_validated(controller, bindings, revision, binding.project_id)
    observations = _advance_until_observed(
        controller,
        bindings,
        revision,
        binding.project_id,
        lane="gpu",
    )
    controller.advance_scheduler_claims(
        bindings,
        revision,
        lane="gpu",
        admission_role="primary",
        observations=observations,
        available_gpu_ids=[0],
        available_cpu_slots=0,
    )
    claim = next(request for request in executor.unresolved_requests() if request.operation_kind == "scheduler_claim")
    offer_identity = _claim_offer_identity(claim)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        executor.poll()
        if executor.load_result(claim.request_id) is not None and executor.status_view()["active_worker_count"] == 0:
            break
        time.sleep(0.02)
    assert executor.load_result(claim.request_id) is not None
    assert classify_executor_offer(runtime.root, offer_identity) == "matching_provisional"

    persisted = load_task(cfg, task.task_id)
    persisted.claim_control["active_claim"] = None
    persisted.attempt_control["current_attempt_id"] = None
    persisted.state.update({"projection": "failed", "reason": "synthetic_terminal"})
    persisted.meta["revision"] += 1
    save_task(cfg, persisted)
    attempt_record_path = attempt_path(cfg.shared_root, task.task_id, 1)
    attempt_record = read_json(attempt_record_path)
    attempt_record["attempt"]["phase"] = "failed"
    atomic_replace(attempt_record_path, attempt_record)
    executor.paths["project_io_results"].joinpath(f"{claim.request_id}.json").unlink()
    executor.begin_epoch()
    restarted = ProjectIOController(runtime, executor, monotonic=lambda: now[0])

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        now[0] += 5.0
        restarted.advance_scheduler_claims(
            bindings,
            revision,
            lane="gpu",
            admission_role="primary",
            observations={},
            available_gpu_ids=[],
            available_cpu_slots=0,
        )
        if not executor.unresolved_requests():
            break
        time.sleep(0.02)

    assert executor.unresolved_requests() == ()
    assert classify_executor_offer(runtime.root, offer_identity) == "matching_released"
