from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.agent import project_io_supervision
from qqtools.plugins.qexp.agent.bindings import ProjectBinding
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.project_io_controller import ProjectIOController
from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor
from qqtools.plugins.qexp.agent.project_io_supervision import AttemptSupervisionCoordinator
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.commands.task import retry
from qqtools.plugins.qexp.runtime.paths import attempt_path, local_paths
from qqtools.plugins.qexp.runtime.records import AttemptRecord, utc_now
from qqtools.plugins.qexp.runtime.resources.reservations import reservation_snapshot
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task, save_task
from qqtools.plugins.qexp.scheduler import authorize_launch, claim_task

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _until(operation, predicate, *, timeout: float = 15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = operation()
        if predicate(value):
            return value
        time.sleep(0.02)
    raise AssertionError("local supervision terminal completion did not converge")


def _case(
    tmp_path: Path,
    *,
    exit_code: int,
    disabled: bool = False,
    cancel_requested: bool = False,
):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    task = submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    attempt = claim_task(
        cfg,
        task.task_id,
        [0],
        reservation_runtime_root=runtime.root,
        project_id=binding.project_id,
    )
    assert attempt is not None
    process_identity = {
        "wrapper_pid": None,
        "wrapper_start_time_ticks": None,
        "process_group_id": 2_000_000_000,
        "process_group_start_time_ticks": 1,
    }
    attempt_file = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    stored = AttemptRecord.from_dict(read_json(attempt_file))
    stored.phase = "running"
    stored.process.update(process_identity)
    atomic_replace(attempt_file, stored.to_dict())
    task = load_task(cfg, task.task_id)
    task.state.update(projection="running", reason="running")
    task.claim_control["active_claim"]["launch_state"] = "running"
    task.control["terminate_running"] = cancel_requested
    task.meta["revision"] += 1
    save_task(cfg, task)
    if disabled:
        runtime.set_enabled(binding.project_id, False)
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
    atomic_replace(paths["registrations"] / f"{attempt.attempt_id}.json", {"process_registration": process})
    manifest = paths["processes"] / f"{attempt.attempt_id}.json"
    atomic_replace(manifest, {"process": process})
    observation = paths["observations"] / f"{attempt.attempt_id}.json"
    atomic_replace(
        observation,
        {
            "exit_observation": {
                "protocol_version": 1,
                "task_id": task.task_id,
                "attempt_id": attempt.attempt_id,
                "observed_exit_code": exit_code,
                "observed_at": utc_now(),
            }
        },
    )
    return runtime, cfg, binding, revision, task, attempt, paths, manifest, observation


@pytest.mark.parametrize("exit_code, expected_phase", [(0, "succeeded"), (17, "failed")])
@pytest.mark.parametrize("disabled", [False, True], ids=["enabled", "disabled"])
def test_terminal_coordinator_commits_natural_exit_and_converges_local_capacity(
    tmp_path: Path,
    exit_code: int,
    expected_phase: str,
    disabled: bool,
) -> None:
    runtime, cfg, binding, revision, task, attempt, paths, manifest, observation = _case(
        tmp_path,
        exit_code=exit_code,
        disabled=disabled,
    )
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)

        def advance():
            value = coordinator.advance_terminal_completions([binding], revision)
            terminal = load_task(cfg, task.task_id).state["projection"] == expected_phase
            local = read_json(manifest)["process"].get("observed_state") == "exited"
            released = not reservation_snapshot(runtime.root).active
            return value, terminal and local and released

        value, _complete = _until(advance, lambda item: item[1])
        evidence = value[binding.project_id]
        assert evidence["outcome"] in {"committed", "already_committed", "already_terminal"}
        terminal_task = load_task(cfg, task.task_id)
        assert terminal_task.state["projection"] == expected_phase
        assert terminal_task.state["reason"] == ("completed" if exit_code == 0 else "nonzero_exit")
        stored = AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task.task_id, 1)))
        assert stored.phase == expected_phase
        assert stored.result["exit_code"] == exit_code
        process = read_json(manifest)["process"]
        assert process["observed_state"] == "exited"
        assert process["observed_exit_code"] == exit_code
        assert isinstance(process["observed_exited_at"], str)
        assert observation.exists()
        assert (paths["registrations"] / observation.name).exists()
        assert not executor.has_unfinished_work()
    finally:
        coordinator.close()
        executor.shutdown()


@pytest.mark.parametrize("exit_code", [0, -15])
def test_superseded_exit_settles_old_capacity_without_changing_successor(tmp_path: Path, exit_code: int) -> None:
    runtime, cfg, binding, revision, task, attempt, paths, manifest, _ = _case(tmp_path, exit_code=exit_code)
    path = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    historical = AttemptRecord.from_dict(read_json(path))
    historical.authority_mode = "holder_bound"
    historical.authorization["launch_id"] = "original-launch"
    historical.timestamps["launch_authorized_at"] = utc_now()
    historical.lease.update(expires_at=None, clock_evidence=None)
    atomic_replace(path, historical.to_dict())
    current = load_task(cfg, task.task_id)
    current.claim_control["active_claim"].update(
        authority_mode="holder_bound",
        launch_id="original-launch",
        launch_authorized_at=historical.timestamps["launch_authorized_at"],
        lease_expires_at=None,
        clock_error_bound_seconds=None,
        clock_provider=None,
        clock_observation_id=None,
    )
    save_task(cfg, current)
    retry(cfg, task.task_id, supersede_attempt=attempt.attempt_id)
    successor = claim_task(cfg, task.task_id, [1], reservation_runtime_root=runtime.root, project_id=binding.project_id)
    assert successor is not None
    assert authorize_launch(
        cfg,
        task.task_id,
        successor.attempt_id,
        successor.current_fencing_token,
        reservation_runtime_root=runtime.root,
    )
    before = load_task(cfg, task.task_id).to_dict()
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)

        def advance():
            coordinator.advance_terminal_completions([binding], revision)
            return read_json(manifest)["process"].get("observed_state"), len(reservation_snapshot(runtime.root).active)

        _until(advance, lambda value: value == ("exited", 1))
        assert load_task(cfg, task.task_id).to_dict() == before
        settled = AttemptRecord.from_dict(read_json(path))
        assert settled.phase == ("succeeded" if exit_code == 0 else "failed")
        assert settled.result["exit_code"] == exit_code
        assert reservation_snapshot(runtime.root).active[0]["attempt_id"] == successor.attempt_id
    finally:
        coordinator.close()
        executor.shutdown()


def test_terminal_coordinator_marks_natural_exit_after_cancel_as_already_exited(tmp_path: Path) -> None:
    runtime, cfg, binding, revision, task, attempt, _paths, manifest, _observation = _case(
        tmp_path,
        exit_code=0,
        cancel_requested=True,
    )
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        _until(
            lambda: coordinator.advance_terminal_completions([binding], revision),
            lambda _value: (
                load_task(cfg, task.task_id).state["projection"] == "succeeded"
                and read_json(manifest)["process"].get("observed_state") == "exited"
            ),
        )
        stored = AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task.task_id, 1)))
        assert stored.phase == "succeeded"
        assert stored.termination["result"] == "already_exited"
        assert read_json(manifest)["process"]["observed_state"] == "exited"
    finally:
        coordinator.close()
        executor.shutdown()


def test_terminal_coordinator_marks_signalled_exit_after_cancel_as_cancelled(tmp_path: Path) -> None:
    runtime, cfg, binding, revision, task, attempt, _paths, manifest, _observation = _case(
        tmp_path,
        exit_code=-15,
        cancel_requested=True,
    )
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        _until(
            lambda: coordinator.advance_terminal_completions([binding], revision),
            lambda _value: (
                load_task(cfg, task.task_id).state["projection"] == "cancelled"
                and read_json(manifest)["process"].get("observed_state") == "exited"
            ),
        )
        stored = AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task.task_id, 1)))
        assert stored.phase == "cancelled"
        assert stored.result["reason"] == "termination_process_already_exited"
        assert stored.termination["result"] == "already_exited"
    finally:
        coordinator.close()
        executor.shutdown()


def test_terminal_coordinator_rejects_mismatched_exit_identity_without_effect(tmp_path: Path) -> None:
    runtime, cfg, binding, revision, task, attempt, _paths, manifest, observation = _case(
        tmp_path,
        exit_code=0,
    )
    value = read_json(observation)
    value["exit_observation"]["attempt_id"] = "different-attempt"
    atomic_replace(observation, value)
    before_task = load_task(cfg, task.task_id).to_dict()
    before_manifest = manifest.read_bytes()
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        for _ in range(6):
            assert coordinator.advance_terminal_completions([binding], revision) == {}
        assert load_task(cfg, task.task_id).to_dict() == before_task
        assert manifest.read_bytes() == before_manifest
        assert reservation_snapshot(runtime.root).active
        assert not executor.has_unfinished_work()
    finally:
        coordinator.close()
        executor.shutdown()


def test_terminal_coordinator_replays_local_effects_after_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, cfg, binding, revision, task, _attempt, _paths, manifest, _observation = _case(
        tmp_path,
        exit_code=0,
    )
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    original = project_io_supervision._apply_terminal_local_effects
    monkeypatch.setattr(project_io_supervision, "_apply_terminal_local_effects", lambda *_args: True)
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        _until(
            lambda: coordinator.advance_terminal_completions([binding], revision),
            lambda _value: load_task(cfg, task.task_id).state["projection"] == "succeeded",
        )
        assert read_json(manifest)["process"]["observed_state"] == "running"
        # Exact local process/exit evidence releases machine capacity before
        # the independently replayable manifest effect and shared publication.
        assert not reservation_snapshot(runtime.root).active
    finally:
        coordinator.close()

    monkeypatch.setattr(project_io_supervision, "_apply_terminal_local_effects", original)
    assert any(request.operation_kind == "authority_terminal_publish" for request in executor.unresolved_requests())
    recovered_requests = 0
    recover = project_io_supervision._recover_terminal_transition

    def observe_recovery(request, candidate, registry_revision):
        nonlocal recovered_requests
        value = recover(request, candidate, registry_revision)
        recovered_requests += value is not None
        return value

    monkeypatch.setattr(project_io_supervision, "_recover_terminal_transition", observe_recovery)
    restarted = AttemptSupervisionCoordinator(runtime, controller)
    try:
        _until(
            lambda: restarted.advance_terminal_completions([binding], revision),
            lambda _value: (
                read_json(manifest)["process"].get("observed_state") == "exited"
                and not reservation_snapshot(runtime.root).active
            ),
        )
        assert recovered_requests > 0
    finally:
        restarted.close()
        executor.shutdown()


def test_terminal_coordinator_rejects_conflicting_settled_result(tmp_path: Path) -> None:
    runtime, cfg, binding, revision, task, attempt, _paths, manifest, _observation = _case(
        tmp_path,
        exit_code=0,
    )
    attempt_file = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    stored = AttemptRecord.from_dict(read_json(attempt_file))
    stored.phase = "succeeded"
    stored.result.update(reason="different_result", exit_code=99)
    atomic_replace(attempt_file, stored.to_dict())
    current = load_task(cfg, task.task_id)
    current.state.update(projection="succeeded", reason="different_result")
    current.claim_control["active_claim"] = None
    current.attempt_control["current_attempt_id"] = None
    current.meta["revision"] += 1
    save_task(cfg, current)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        for _ in range(12):
            coordinator.advance_terminal_completions([binding], revision)
            time.sleep(0.02)
        assert read_json(manifest)["process"]["observed_state"] == "running"
        # A conflicting shared settlement cannot authorize manifest cleanup,
        # but the proven exited local process no longer occupies capacity.
        assert not reservation_snapshot(runtime.root).active
    finally:
        coordinator.close()
        executor.shutdown()


def test_terminal_coordinator_advances_sequential_attempts_for_one_project(tmp_path: Path) -> None:
    runtime, cfg, binding, revision, task, _attempt, paths, first_manifest, _observation = _case(
        tmp_path,
        exit_code=17,
    )
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        _until(
            lambda: coordinator.advance_terminal_completions([binding], revision),
            lambda _value: (
                load_task(cfg, task.task_id).state["projection"] == "failed"
                and read_json(first_manifest)["process"].get("observed_state") == "exited"
            ),
        )
        retry(cfg, task.task_id)
        second = claim_task(
            cfg,
            task.task_id,
            [0],
            reservation_runtime_root=runtime.root,
            project_id=binding.project_id,
        )
        assert second is not None
        second_file = attempt_path(cfg.shared_root, task.task_id, second.attempt_number)
        second_stored = AttemptRecord.from_dict(read_json(second_file))
        process_identity = {
            "wrapper_pid": None,
            "wrapper_start_time_ticks": None,
            "process_group_id": 2_000_000_001,
            "process_group_start_time_ticks": 1,
        }
        second_stored.phase = "running"
        second_stored.process.update(process_identity)
        atomic_replace(second_file, second_stored.to_dict())
        second_task = load_task(cfg, task.task_id)
        second_task.state.update(projection="running", reason="running")
        second_task.claim_control["active_claim"]["launch_state"] = "running"
        second_task.meta["revision"] += 1
        save_task(cfg, second_task)
        created_at = utc_now()
        process = {
            "protocol_version": 1,
            "machine_name": binding.machine_name,
            "task_id": task.task_id,
            "attempt_id": second.attempt_id,
            "fencing_token": second.current_fencing_token,
            "reservation_id": second.reservation_id,
            **process_identity,
            "process_created_at": created_at,
            "created_at": created_at,
            "observed_state": "running",
            "supervisor": "agent",
            "authority_state": "healthy",
            "created_by": "agent",
        }
        atomic_replace(paths["registrations"] / f"{second.attempt_id}.json", {"process_registration": process})
        second_manifest = paths["processes"] / f"{second.attempt_id}.json"
        atomic_replace(second_manifest, {"process": process})
        atomic_replace(
            paths["observations"] / f"{second.attempt_id}.json",
            {
                "exit_observation": {
                    "protocol_version": 1,
                    "task_id": task.task_id,
                    "attempt_id": second.attempt_id,
                    "observed_exit_code": 0,
                    "observed_at": utc_now(),
                }
            },
        )
        _until(
            lambda: coordinator.advance_terminal_completions([binding], revision),
            lambda _value: (
                load_task(cfg, task.task_id).state["projection"] == "succeeded"
                and read_json(second_manifest)["process"].get("observed_state") == "exited"
            ),
        )
    finally:
        coordinator.close()
        executor.shutdown()


def test_terminal_coordinator_accepts_settled_cancelled_natural_exit_after_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, cfg, binding, revision, task, _attempt, _paths, manifest, _observation = _case(
        tmp_path,
        exit_code=17,
        cancel_requested=True,
    )
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    original = project_io_supervision._apply_terminal_local_effects
    monkeypatch.setattr(project_io_supervision, "_apply_terminal_local_effects", lambda *_args: "deferred")
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        _until(
            lambda: coordinator.advance_terminal_completions([binding], revision),
            lambda _value: load_task(cfg, task.task_id).state["projection"] == "cancelled",
        )
        assert read_json(manifest)["process"]["observed_state"] == "running"
    finally:
        coordinator.close()
    monkeypatch.setattr(project_io_supervision, "_apply_terminal_local_effects", original)
    restarted = AttemptSupervisionCoordinator(runtime, controller)
    try:
        _until(
            lambda: restarted.advance_terminal_completions([binding], revision),
            lambda _value: (
                read_json(manifest)["process"].get("observed_state") == "exited"
                and not reservation_snapshot(runtime.root).active
            ),
        )
    finally:
        restarted.close()
        executor.shutdown()


def test_terminal_coordinator_bounds_binding_scan_state(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    bindings = [
        ProjectBinding(
            project_id=f"project-{index}",
            shared_root=tmp_path / f"project-{index}" / ".qexp",
            machine_name="gpu-1",
            registration_generation=f"generation-{index}",
        )
        for index in range(300)
    ]
    coordinator = AttemptSupervisionCoordinator(runtime, object())  # type: ignore[arg-type]
    try:
        for _ in range(6):
            assert coordinator.advance_terminal_completions(bindings, 1) == {}
        assert len(coordinator._terminal_states) <= 256
    finally:
        coordinator.close()


@pytest.mark.parametrize("disabled", [False, True], ids=["enabled", "disabled"])
@pytest.mark.parametrize(
    ("cancel_requested", "expected_phase", "expected_reason"),
    [
        (False, "succeeded", "completed"),
        (True, "cancelled", "termination_process_already_exited"),
    ],
    ids=["natural", "cancelled"],
)
def test_terminal_coordinator_completes_real_detached_orphan(
    tmp_path: Path,
    disabled: bool,
    cancel_requested: bool,
    expected_phase: str,
    expected_reason: str,
) -> None:
    runtime, cfg, binding, _revision, task, attempt, _paths, manifest, _observation = _case(
        tmp_path,
        exit_code=0,
        cancel_requested=cancel_requested,
    )
    running = load_task(cfg, task.task_id)
    running.claim_control["active_claim"].update(
        lease_expires_at="2000-01-01T00:00:00Z",
        clock_error_bound_seconds=0.0,
    )
    running.meta["revision"] += 1
    save_task(cfg, running)
    # A supported source already persisted this detached orphan before upgrade.
    historical_path = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    historical = AttemptRecord.from_dict(read_json(historical_path))
    historical.phase = "orphaned"
    historical.timestamps["orphaned_at"] = utc_now()
    atomic_replace(historical_path, historical.to_dict())
    running.state.update(projection="blocked", reason="orphaned_attempt_requires_recovery")
    running.claim_control["active_claim"] = None
    running.attempt_control["current_attempt_id"] = None
    save_task(cfg, running)
    if disabled:
        runtime.set_enabled(binding.project_id, False)
    revision, bindings = runtime.load_registry()
    binding = bindings[0]
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        _until(
            lambda: coordinator.advance_terminal_completions([binding], revision),
            lambda _value: (
                load_task(cfg, task.task_id).state["projection"] == expected_phase
                and read_json(manifest)["process"].get("observed_state") == "exited"
                and not reservation_snapshot(runtime.root).active
            ),
        )
        terminal = load_task(cfg, task.task_id)
        assert terminal.state == {"projection": expected_phase, "reason": expected_reason}
        assert terminal.control["termination_result"] == ("already_exited" if cancel_requested else None)
    finally:
        coordinator.close()
        executor.shutdown()


def test_terminal_coordinator_rereads_exact_local_evidence_before_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, cfg, binding, revision, task, _attempt, _paths, manifest, observation = _case(
        tmp_path,
        exit_code=0,
    )
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    original = project_io_supervision._apply_terminal_local_effects
    mutated = False

    def mutate_before_effects(candidate, proof, paths, reconciler):
        nonlocal mutated
        if not mutated:
            value = read_json(observation)
            value["exit_observation"]["attempt_id"] = "different-attempt"
            atomic_replace(observation, value)
            mutated = True
        return original(candidate, proof, paths, reconciler)

    monkeypatch.setattr(project_io_supervision, "_apply_terminal_local_effects", mutate_before_effects)
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        _until(
            lambda: coordinator.advance_terminal_completions([binding], revision),
            lambda _value: load_task(cfg, task.task_id).state["projection"] == "succeeded",
        )
        _until(
            lambda: coordinator.advance_terminal_completions([binding], revision),
            lambda _value: mutated,
        )
        assert mutated
        assert read_json(manifest)["process"]["observed_state"] == "running"
        # The mutation is injected only after the earlier exact observation
        # released capacity; the final local-effect reread still blocks the
        # manifest transition.
        assert not reservation_snapshot(runtime.root).active
    finally:
        coordinator.close()
        executor.shutdown()


def test_terminal_coordinator_rejects_proof_for_different_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, cfg, binding, revision, task, _attempt, _paths, manifest, _observation = _case(
        tmp_path,
        exit_code=0,
    )
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    original_observation = project_io_supervision._is_terminal_observation_proof
    original_publication = project_io_supervision._is_terminal_publication_proof

    def mismatch(proof):
        return None if proof is None else replace(proof, transition_digest="0" * 64)

    monkeypatch.setattr(
        project_io_supervision,
        "_is_terminal_observation_proof",
        lambda evidence, candidate: mismatch(original_observation(evidence, candidate)),
    )
    monkeypatch.setattr(
        project_io_supervision,
        "_is_terminal_publication_proof",
        lambda evidence, candidate, transition: mismatch(original_publication(evidence, candidate, transition)),
    )
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        _until(
            lambda: coordinator.advance_terminal_completions([binding], revision),
            lambda _value: load_task(cfg, task.task_id).state["projection"] == "succeeded",
        )
        for _ in range(8):
            coordinator.advance_terminal_completions([binding], revision)
        assert read_json(manifest)["process"]["observed_state"] == "running"
    finally:
        coordinator.close()
        executor.shutdown()


@pytest.mark.parametrize("failure_point", ["replacement", "reconciliation"])
def test_terminal_coordinator_retries_interrupted_local_effects_without_changing_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    runtime, cfg, binding, revision, task, _attempt, _paths, manifest, _observation = _case(
        tmp_path,
        exit_code=0,
    )
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    interruption: dict[str, str] = {}
    successful_replays = 0
    original_apply = project_io_supervision._apply_terminal_local_effects

    def observe_replay(*args, **kwargs):
        nonlocal successful_replays
        outcome = original_apply(*args, **kwargs)
        if interruption and outcome == "applied":
            successful_replays += 1
        return outcome

    monkeypatch.setattr(project_io_supervision, "_apply_terminal_local_effects", observe_replay)

    if failure_point == "replacement":
        original_replace = project_io_supervision.atomic_replace

        def interrupt_replace(path, value):
            if path == manifest and value.get("process", {}).get("observed_state") == "exited" and not interruption:
                original_replace(path, value)
                interruption["observed_exited_at"] = value["process"]["observed_exited_at"]
                raise OSError("injected interruption after replacement")
            return original_replace(path, value)

        monkeypatch.setattr(project_io_supervision, "atomic_replace", interrupt_replace)
    else:
        original_reconcile = project_io_supervision.LocalExitReconciler.reconcile_observation

        def interrupt_reconcile(self, observation, **kwargs):
            process = read_json(manifest)["process"]
            if process.get("observed_state") == "exited" and not interruption:
                interruption["observed_exited_at"] = process["observed_exited_at"]
                raise OSError("injected interruption before reconciliation")
            return original_reconcile(self, observation, **kwargs)

        monkeypatch.setattr(
            project_io_supervision.LocalExitReconciler,
            "reconcile_observation",
            interrupt_reconcile,
        )

    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        _until(
            lambda: coordinator.advance_terminal_completions([binding], revision),
            lambda _value: (
                bool(interruption)
                and successful_replays > 0
                and read_json(manifest)["process"].get("observed_state") == "exited"
                and not reservation_snapshot(runtime.root).active
            ),
        )
        assert load_task(cfg, task.task_id).state["projection"] == "succeeded"
        assert binding.project_id not in coordinator._terminal_candidates
        assert read_json(manifest)["process"]["observed_exited_at"] == interruption["observed_exited_at"]
    finally:
        coordinator.close()
        executor.shutdown()
