from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.agent import project_io_supervision
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.dispatch_loop import dispatch_machine_cycle_locked
from qqtools.plugins.qexp.agent.project_io_controller import ProjectIOController
from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor
from qqtools.plugins.qexp.agent.project_io_supervision import AttemptSupervisionCoordinator
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.commands.group import create_group
from qqtools.plugins.qexp.infrastructure.process import process_start_time_ticks
from qqtools.plugins.qexp.runtime import local_exit_reconciliation
from qqtools.plugins.qexp.runtime.authority_scan import EvidenceScan
from qqtools.plugins.qexp.runtime.availability import sync_deadline_index
from qqtools.plugins.qexp.runtime.paths import attempt_path, local_paths
from qqtools.plugins.qexp.runtime.records import AttemptRecord, utc_now
from qqtools.plugins.qexp.runtime.resources.reservations import reservation_snapshot
from qqtools.plugins.qexp.runtime.store import atomic_replace, iter_json, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task, save_task
from qqtools.plugins.qexp.scheduler import claim_task

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _until(operation, predicate, *, timeout: float = 15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = operation()
        if predicate(value):
            return value
        time.sleep(0.02)
    raise AssertionError(f"local supervision termination did not converge; last value: {value!r}")


def _advance_and_snapshot(coordinator, bindings, revision, cfg, task_id, child, manifest, runtime):
    result = coordinator.advance_terminations(bindings, revision)
    return (
        result,
        load_task(cfg, task_id).state["projection"],
        child.poll(),
        read_json(manifest)["process"].get("observed_state"),
        len(reservation_snapshot(runtime.root).active),
    )


def _running_case(tmp_path: Path, *, trigger: str):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    task = submit(cfg, ["sleep", "60"], working_dir=tmp_path)
    attempt = claim_task(
        cfg,
        task.task_id,
        [0],
        reservation_runtime_root=runtime.root,
        project_id=binding.project_id,
    )
    assert attempt is not None
    child = subprocess.Popen(["sleep", "60"], start_new_session=True)
    ticks = process_start_time_ticks(child.pid)
    assert ticks is not None
    process_identity = {
        "wrapper_pid": None,
        "wrapper_start_time_ticks": None,
        "process_group_id": child.pid,
        "process_group_start_time_ticks": ticks,
    }
    attempt_file = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    stored = AttemptRecord.from_dict(read_json(attempt_file))
    stored.phase = "running"
    stored.process.update(process_identity)
    atomic_replace(attempt_file, stored.to_dict())
    running = load_task(cfg, task.task_id)
    running.state.update(projection="running", reason="running")
    running.claim_control["active_claim"]["launch_state"] = "running"
    running.control["terminate_running"] = trigger == "cancel"
    running.meta["revision"] += 1
    save_task(cfg, running)
    revision, bindings = runtime.load_registry()
    binding = bindings[0]
    paths = local_paths(runtime.project_paths(binding.project_id)["root"])
    created_at = utc_now()
    process = {
        "protocol_version": 1,
        "machine_name": binding.machine_name,
        "task_id": task.task_id,
        "attempt_id": attempt.attempt_id,
        "fencing_token": attempt.current_fencing_token,
        "reservation_id": attempt.reservation_id,
        **process_identity,
        "process_created_at": created_at,
        "created_at": created_at,
        "observed_state": "running",
        "supervisor": "agent",
        "authority_state": "healthy",
        "created_by": "agent",
    }
    if trigger == "safe_deadline":
        process.update(
            lease_expires_at="2000-01-01T00:00:00Z",
            clock_error_bound_seconds=0.0,
        )
    atomic_replace(paths["registrations"] / f"{attempt.attempt_id}.json", {"process_registration": process})
    manifest = paths["processes"] / f"{attempt.attempt_id}.json"
    atomic_replace(manifest, {"process": process})
    return runtime, cfg, binding, revision, task, attempt, child, paths, manifest, attempt_file


@pytest.mark.parametrize("trigger", ["cancel", "safe_deadline"])
def test_termination_coordinator_commits_before_signaling_and_converges_capacity(
    tmp_path: Path,
    trigger: str,
) -> None:
    runtime, cfg, binding, revision, task, attempt, child, paths, manifest, _attempt_file = _running_case(
        tmp_path,
        trigger=trigger,
    )
    try:
        executor = ProjectIOExecutor(runtime)
        executor.begin_epoch()
        controller = ProjectIOController(runtime, executor)
        coordinator = AttemptSupervisionCoordinator(runtime, controller)
        try:
            _until(lambda: controller.advance_binding_validation([binding], revision), bool)
            _until(
                lambda: _advance_and_snapshot(
                    coordinator, [binding], revision, cfg, task.task_id, child, manifest, runtime
                ),
                lambda value: value[1:] == ("cancelled", child.returncode, "exited", 0),
            )
            terminal = load_task(cfg, task.task_id)
            assert terminal.state == {"projection": "cancelled", "reason": "terminated_by_agent"}
            assert terminal.control["termination_result"] == "terminated"
            decisions = iter_json(paths["termination_decisions"] / attempt.attempt_id)
            assert len(decisions) == 1
            decision = read_json(decisions[0])["termination_decision"]
            assert decision["state"] == "confirmed"
            assert decision["shared_commitment"] == "committed"
            assert [item["signal"] for item in decision["signal_attempts"]] == ["SIGTERM"]
            assert decision["observed_exit_code"] is None
            assert read_json(manifest)["process"]["observed_exit_code"] is None
            assert not executor.has_unfinished_work()
            for _ in range(8):
                coordinator._advance_initial_reconciliation([binding], revision)
                if coordinator.initially_reconciled(binding, revision):
                    break
            assert coordinator.initially_reconciled(binding, revision)
        finally:
            coordinator.close()
            executor.shutdown()
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_restart_with_signalled_exit_restores_offer_and_claim_admission(tmp_path: Path) -> None:
    runtime, cfg, binding, revision, task, attempt, child, paths, manifest, _attempt_file = _running_case(
        tmp_path,
        trigger="cancel",
    )
    first_executor = ProjectIOExecutor(runtime)
    first_executor.begin_epoch()
    first_controller = ProjectIOController(runtime, first_executor)
    first = AttemptSupervisionCoordinator(runtime, first_controller)
    try:
        _until(lambda: first_controller.advance_binding_validation([binding], revision), bool)
        _until(
            lambda: _advance_and_snapshot(first, [binding], revision, cfg, task.task_id, child, manifest, runtime),
            lambda value: value[1:] == ("cancelled", child.returncode, "exited", 0),
        )
        assert child.returncode == -15
        process_envelope = read_json(manifest)
        process_envelope["process"]["observed_exit_code"] = -15
        atomic_replace(manifest, process_envelope)
        capacity_paths = local_paths(runtime.root)
        released_path = capacity_paths["released"] / f"{attempt.reservation_id}.json"
        active_path = capacity_paths["active"] / released_path.name
        reservation_envelope = read_json(released_path)
        reservation_envelope["reservation"]["state"] = "active"
        reservation_envelope["reservation"].pop("released_at", None)
        reservation_envelope["reservation"].pop("release_reason", None)
        atomic_replace(active_path, reservation_envelope)
        released_path.unlink()
    finally:
        first.close()
        first_executor.shutdown()

    create_group(cfg, "spill")
    queued = submit(
        cfg,
        ["echo", "restart-safe"],
        group="spill",
        sharing_mode="spillover",
        offer_after_seconds=0,
    )
    queued_record = load_task(cfg, queued.task_id)
    queued_record.placement_runtime["offer_eligible_at"] = "2000-01-01T00:00:00Z"
    save_task(cfg, queued_record)
    sync_deadline_index(cfg, queued_record)

    restarted_executor = ProjectIOExecutor(runtime)
    restarted_executor.begin_epoch()
    restarted_controller = ProjectIOController(runtime, restarted_executor)
    restarted = AttemptSupervisionCoordinator(runtime, restarted_controller)
    runtime.project_io_executor = restarted_executor
    runtime.project_io_controller = restarted_controller
    runtime.attempt_supervision_coordinator = restarted
    runtime.authority_ready_generations = {}
    try:

        def advance_restart() -> tuple[list[dict[str, object]], str]:
            results = dispatch_machine_cycle_locked(
                runtime,
                available_gpus=[],
                supervise=True,
                publish_snapshots=False,
            )
            return results, load_task(cfg, queued.task_id).placement_runtime["queue_scope"]

        results, queue_scope = _until(
            advance_restart,
            lambda value: (
                value[1] == "shared" and all(item.get("status") != "authority_recovering" for item in value[0])
            ),
            timeout=30.0,
        )
        assert queue_scope == "shared"
        assert all(item.get("status") != "authority_recovering" for item in results)
        assert restarted.initially_reconciled(binding, revision)
        assert read_json(manifest)["process"]["observed_exit_code"] == -15
        decision_paths = iter_json(paths["termination_decisions"] / attempt.attempt_id)
        assert read_json(decision_paths[0])["termination_decision"]["observed_exit_code"] is None

        claimed = claim_task(
            cfg,
            queued.task_id,
            [0],
            reservation_runtime_root=runtime.root,
            project_id=binding.project_id,
        )
        assert claimed is not None
        assert claimed.task_id == queued.task_id
    finally:
        restarted.close()
        restarted_executor.shutdown()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_restart_reports_invalid_termination_exit_code(tmp_path: Path) -> None:
    runtime, cfg, binding, revision, task, attempt, child, _paths, manifest, _attempt_file = _running_case(
        tmp_path,
        trigger="cancel",
    )
    first_executor = ProjectIOExecutor(runtime)
    first_executor.begin_epoch()
    first_controller = ProjectIOController(runtime, first_executor)
    first = AttemptSupervisionCoordinator(runtime, first_controller)
    try:
        _until(lambda: first_controller.advance_binding_validation([binding], revision), bool)
        _until(
            lambda: _advance_and_snapshot(first, [binding], revision, cfg, task.task_id, child, manifest, runtime),
            lambda value: value[1:] == ("cancelled", child.returncode, "exited", 0),
        )
    finally:
        first.close()
        first_executor.shutdown()

    process_envelope = read_json(manifest)
    process_envelope["process"]["observed_exit_code"] = True
    atomic_replace(manifest, process_envelope)
    restarted_executor = ProjectIOExecutor(runtime)
    restarted_executor.begin_epoch()
    restarted_controller = ProjectIOController(runtime, restarted_executor)
    restarted = AttemptSupervisionCoordinator(runtime, restarted_controller)
    runtime.project_io_executor = restarted_executor
    runtime.project_io_controller = restarted_controller
    runtime.attempt_supervision_coordinator = restarted
    runtime.authority_ready_generations = {}
    try:

        def advance_restart():
            return dispatch_machine_cycle_locked(
                runtime,
                available_gpus=[],
                supervise=True,
                publish_snapshots=False,
            )

        results = _until(
            advance_restart,
            lambda value: any("termination_convergence" in item for item in value),
        )
        result = next(item for item in results if "termination_convergence" in item)
        assert result["status"] == "authority_recovering"
        assert result["termination_convergence"] == {
            "state": "invalid",
            "reason": "termination_exit_code_invalid",
            "attempt_id": attempt.attempt_id,
        }
        probe = next(
            item
            for item in runtime.last_scheduler_diagnostic_probes
            if item["identity"]["producer"] == "termination_convergence"
        )
        assert probe["findings"][0]["identity"]["reason_code"] == "termination_exit_code_invalid"
        assert "process" not in probe["findings"][0]
    finally:
        restarted.close()
        restarted_executor.shutdown()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_termination_registration_accepts_explicit_unassigned_reservation(tmp_path: Path) -> None:
    runtime, _cfg, binding, _revision, _task, _attempt, child, paths, manifest, _attempt_file = _running_case(
        tmp_path,
        trigger="cancel",
    )
    try:
        process_envelope = read_json(manifest)
        process_envelope["process"]["reservation_id"] = None
        atomic_replace(manifest, process_envelope)
        registration_path = paths["registrations"] / f"{process_envelope['process']['attempt_id']}.json"
        atomic_replace(registration_path, {"process_registration": process_envelope["process"]})

        registration = project_io_supervision._read_running_publication_source(
            registration_path,
            runtime.project_paths(binding.project_id)["root"],
            binding,
            "process_registration",
            allow_unassigned_reservation=True,
        )

        assert registration is not None
        assert registration.parameters["reservation_id"] is None
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_termination_directory_scan_retains_earlier_deferred_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _cfg, binding, revision, _task, _attempt, child, paths, _manifest, _attempt_file = _running_case(
        tmp_path,
        trigger="cancel",
    )
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    signature = coordinator._initial_reconciliation_signature(binding, revision)
    decisions = tmp_path / "termination-scan"
    for attempt_id in ("attempt-a", "attempt-b"):
        decision_dir = decisions / attempt_id
        decision_dir.mkdir(parents=True)
        atomic_replace(decision_dir / "decision.json", {"termination_decision": {}})

    def classify_path(_binding, _paths, current_signature, decision_path):
        if decision_path.parent.name == "attempt-a":
            coordinator._initial_deferred_lanes.add(current_signature)
            coordinator._initial_pending_termination_lanes.add(current_signature)
            return False
        return True

    monkeypatch.setattr(coordinator, "_initial_termination_decision_converged", classify_path)
    scan = EvidenceScan(decisions, directories=True)
    try:
        complete = False
        for _ in range(8):
            complete, invalid = coordinator._advance_initial_termination_lane(
                binding,
                paths,
                signature,
                scan,
            )
            if complete:
                break
        assert complete
        assert not invalid
        assert signature in coordinator._initial_deferred_lanes
        assert signature in coordinator._initial_pending_termination_lanes
    finally:
        scan.close()
        coordinator.close()
        executor.shutdown()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


@pytest.mark.parametrize("registry_changes", [False, True])
def test_termination_coordinator_recovers_commit_before_signal_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registry_changes: bool,
) -> None:
    runtime, cfg, binding, revision, task, attempt, child, paths, manifest, _attempt_file = _running_case(
        tmp_path,
        trigger="cancel",
    )
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    original = project_io_supervision.commit_signal

    def interrupt_after_commit(*_args, **_kwargs):
        raise OSError("injected crash after shared termination commit")

    monkeypatch.setattr(project_io_supervision, "commit_signal", interrupt_after_commit)
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)

        def advance_to_commit():
            coordinator.advance_terminations([binding], revision)
            decisions = iter_json(paths["termination_decisions"] / attempt.attempt_id)
            if not decisions:
                return None
            return read_json(decisions[0])["termination_decision"]

        decision = _until(
            advance_to_commit,
            lambda value: value is not None and value.get("shared_commitment") == "committed",
        )
        assert decision["state"] == "pending"
        assert decision["signal_attempts"] == []
        assert child.poll() is None
        assert (
            load_task(cfg, task.task_id).claim_control["active_claim"]["termination_decision_id"]
            == decision["decision_id"]
        )
    finally:
        coordinator.close()

    monkeypatch.setattr(project_io_supervision, "commit_signal", original)
    if registry_changes:
        runtime.set_enabled(binding.project_id, False)
        revision, bindings = runtime.load_registry()
        binding = next(item for item in bindings if item.project_id == binding.project_id)
    restarted = AttemptSupervisionCoordinator(runtime, controller)
    try:
        if registry_changes:
            _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        _until(
            lambda: _advance_and_snapshot(restarted, [binding], revision, cfg, task.task_id, child, manifest, runtime),
            lambda value: value[1:] == ("cancelled", child.returncode, "exited", 0),
        )
        decisions = iter_json(paths["termination_decisions"] / attempt.attempt_id)
        assert len(decisions) == 1
        recovered = read_json(decisions[0])["termination_decision"]
        assert recovered["state"] == "confirmed"
        assert [item["signal"] for item in recovered["signal_attempts"]] == ["SIGTERM"]
    finally:
        restarted.close()
        executor.shutdown()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_termination_coordinator_reconciles_exit_before_signal_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, cfg, binding, revision, task, attempt, child, paths, manifest, _attempt_file = _running_case(
        tmp_path,
        trigger="cancel",
    )
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    original = project_io_supervision.commit_signal

    def exit_after_authorization(*args, **kwargs):
        decision = original(*args, **kwargs)
        child.terminate()
        child.wait(timeout=5)
        return decision

    monkeypatch.setattr(project_io_supervision, "commit_signal", exit_after_authorization)
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        _until(
            lambda: _advance_and_snapshot(
                coordinator, [binding], revision, cfg, task.task_id, child, manifest, runtime
            ),
            lambda value: value[1:] == ("cancelled", child.returncode, "exited", 0),
        )
        terminal = load_task(cfg, task.task_id)
        assert terminal.state == {
            "projection": "cancelled",
            "reason": "termination_process_already_exited",
        }
        assert terminal.control["termination_result"] == "already_exited"
        decisions = iter_json(paths["termination_decisions"] / attempt.attempt_id)
        decision = read_json(decisions[0])["termination_decision"]
        assert decision["state"] == "confirmed"
        assert decision["signal_attempts"] == []
    finally:
        coordinator.close()
        executor.shutdown()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_termination_capacity_release_does_not_enumerate_machine_reservations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, cfg, binding, revision, task, _attempt, child, _paths, manifest, _attempt_file = _running_case(
        tmp_path,
        trigger="cancel",
    )
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)

    def reject_enumeration(*_args, **_kwargs):
        raise AssertionError("termination release must use exact reservation lookup")

    monkeypatch.setattr(local_exit_reconciliation, "reservation_snapshot", reject_enumeration)
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        _until(
            lambda: _advance_and_snapshot(
                coordinator, [binding], revision, cfg, task.task_id, child, manifest, runtime
            ),
            lambda value: value[1:] == ("cancelled", child.returncode, "exited", 0),
        )
    finally:
        coordinator.close()
        executor.shutdown()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_termination_capacity_release_repairs_interrupted_active_unlink(tmp_path: Path) -> None:
    runtime, cfg, binding, revision, task, attempt, child, _paths, manifest, _attempt_file = _running_case(
        tmp_path,
        trigger="cancel",
    )
    capacity = local_paths(runtime.root)
    active_path = capacity["active"] / f"{attempt.reservation_id}.json"
    released_path = capacity["released"] / active_path.name
    duplicate = read_json(active_path)
    duplicate["reservation"].update(
        state="released",
        released_at=utc_now(),
        release_reason="local_termination_confirmed",
    )
    atomic_replace(released_path, duplicate)

    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        _until(
            lambda: _advance_and_snapshot(
                coordinator, [binding], revision, cfg, task.task_id, child, manifest, runtime
            ),
            lambda value: value[1:] == ("cancelled", child.returncode, "exited", 0),
        )
        assert not active_path.exists()
        assert released_path.exists()
    finally:
        coordinator.close()
        executor.shutdown()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_termination_coordinator_never_signals_from_stale_shared_identity(tmp_path: Path) -> None:
    runtime, cfg, binding, revision, task, attempt, child, paths, manifest, attempt_file = _running_case(
        tmp_path,
        trigger="cancel",
    )
    stored = AttemptRecord.from_dict(read_json(attempt_file))
    stored.current_fencing_token += 1
    atomic_replace(attempt_file, stored.to_dict())
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        for _ in range(24):
            coordinator.advance_terminations([binding], revision)
            time.sleep(0.02)
        assert child.poll() is None
        assert load_task(cfg, task.task_id).state["projection"] == "running"
        assert iter_json(paths["termination_decisions"] / attempt.attempt_id) == []
        assert read_json(manifest)["process"]["observed_state"] == "running"
        assert reservation_snapshot(runtime.root).active
    finally:
        coordinator.close()
        executor.shutdown()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_healthy_termination_observation_does_not_pin_restart_readiness(tmp_path: Path) -> None:
    runtime, cfg, binding, revision, task, attempt, child, paths, manifest, _ = _running_case(tmp_path, trigger="none")
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    coordinator = AttemptSupervisionCoordinator(runtime, ProjectIOController(runtime, executor))
    try:
        _until(lambda: coordinator.controller.advance_binding_validation([binding], revision), bool)

        # Production collects all lanes before granting a request. Calling each
        # lane as a separate turn can keep readiness artificially occupied.
        def advance():
            with coordinator.controller.admission_turn():
                coordinator.advance_all([binding], revision)
            return coordinator.initially_reconciled(binding, revision)

        _until(advance, bool)
        assert child.poll() is None
        assert load_task(cfg, task.task_id).state["projection"] == "running"
        assert read_json(manifest)["process"]["observed_state"] == "running"
        assert len(reservation_snapshot(runtime.root).active) == 1
        assert not iter_json(paths["termination_decisions"] / attempt.attempt_id)
        # A later cancellation must still be discovered after the healthy probe
        # was retired; readiness cannot disable subsequent supervision.
        running = load_task(cfg, task.task_id)
        running.control["terminate_running"] = True
        running.meta["revision"] += 1
        save_task(cfg, running)

        def cancellation_snapshot():
            # Keep draining every lane started during recovery, as the machine
            # loop does; completed requests otherwise retain executor capacity.
            advance()
            return (
                load_task(cfg, task.task_id).state["projection"],
                child.poll(),
                read_json(manifest)["process"].get("observed_state"),
                len(reservation_snapshot(runtime.root).active),
            )

        _until(
            cancellation_snapshot,
            lambda value: value == ("cancelled", child.returncode, "exited", 0),
        )
    finally:
        coordinator.close()
        executor.shutdown()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
