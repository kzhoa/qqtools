from __future__ import annotations

import hashlib
import subprocess
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.agent import project_io_supervision
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.project_io_controller import ProjectIOController
from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor
from qqtools.plugins.qexp.agent.project_io_supervision import AttemptSupervisionCoordinator
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.infrastructure.process import process_start_time_ticks
from qqtools.plugins.qexp.runtime import attempt_recovery
from qqtools.plugins.qexp.runtime.paths import attempt_path, local_paths
from qqtools.plugins.qexp.runtime.resources.cpu_lane import set_cpu_lane_capacity
from qqtools.plugins.qexp.runtime.resources.reservations import ReservationIdentity, release_if_matches
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task, save_task
from qqtools.plugins.qexp.scheduler import (
    authorize_launch,
    cancel_task,
    claim_task,
    expire_claim,
    recover_project_io_orphaned_attempt,
    renew_project_io_attempt_lease,
)
from tests.fixtures.qexp_legacy_ownership import persist_legacy_orphan

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]
_LAST_RESULTS = {}


def _until(operation, predicate, *, timeout: float = 15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = operation()
        if predicate(value):
            return value
        time.sleep(0.02)
    raise AssertionError(f"isolated orphan recovery did not converge; last value: {value!r}")


def _orphan_case(tmp_path: Path, *, is_cpu: bool = False):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    atomic_replace(
        local_paths(runtime.project_paths(binding.project_id)["root"])["clock_health"],
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
    if is_cpu:
        set_cpu_lane_capacity(runtime.root, capacity=1)
    task = submit(
        cfg,
        ["sleep", "60"],
        working_dir=tmp_path,
        requested_gpus=0 if is_cpu else 1,
        requested_cpus=1 if is_cpu else None,
    )
    attempt = claim_task(
        cfg,
        task.task_id,
        [] if is_cpu else [0],
        reservation_runtime_root=runtime.root,
        project_id=binding.project_id,
    )
    assert attempt is not None
    assert authorize_launch(
        cfg,
        task.task_id,
        attempt.attempt_id,
        attempt.current_fencing_token,
        reservation_runtime_root=runtime.root,
    )
    child = subprocess.Popen(["sleep", "60"], start_new_session=True)
    ticks = process_start_time_ticks(child.pid)
    assert ticks is not None
    attempt_file = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    stored = read_json(attempt_file)
    stored["attempt"]["phase"] = "running"
    stored["attempt"]["process"].update(
        wrapper_pid=None,
        wrapper_start_time_ticks=None,
        process_group_id=child.pid,
        process_group_start_time_ticks=ticks,
    )
    atomic_replace(attempt_file, stored)
    persist_legacy_orphan(cfg, task.task_id)
    revision, bindings = runtime.load_registry()
    binding = bindings[0]
    paths = local_paths(runtime.project_paths(binding.project_id)["root"])
    manifest = paths["processes"] / f"{attempt.attempt_id}.json"
    process = {
        "protocol_version": 1,
        "machine_name": binding.machine_name,
        "task_id": task.task_id,
        "attempt_id": attempt.attempt_id,
        "fencing_token": attempt.current_fencing_token,
        "reservation_id": attempt.reservation_id,
        "wrapper_pid": None,
        "wrapper_start_time_ticks": None,
        "process_group_id": child.pid,
        "process_group_start_time_ticks": ticks,
        "observed_state": "running",
        "supervisor": "agent",
        "authority_state": "isolated",
        "created_by": "agent",
    }
    atomic_replace(manifest, {"process": process})
    reservation = local_paths(runtime.root)["cpu_active" if is_cpu else "active"] / f"{attempt.reservation_id}.json"
    return runtime, cfg, binding, revision, task, attempt, child, manifest, reservation


def _advance_snapshot(coordinator, binding, revision, cfg, task, attempt, manifest, reservation):
    result = coordinator.advance_orphan_recoveries([binding], revision)
    if result:
        _LAST_RESULTS[task.task_id] = result
    stored_task = load_task(cfg, task.task_id)
    reservation_record = read_json(reservation)["reservation"]
    return (
        result or _LAST_RESULTS.get(task.task_id, {}),
        stored_task.state["projection"],
        (stored_task.claim_control.get("active_claim") or {}).get("fencing_token"),
        reservation_record.get("fencing_token"),
        read_json(manifest)["process"].get("fencing_token"),
    )


def _recover_shared(cfg, binding, attempt, *, replay_only: bool = False):
    stored = read_json(attempt_path(cfg.shared_root, attempt.task_id, attempt.attempt_number))["attempt"]
    return recover_project_io_orphaned_attempt(
        cfg,
        machine_name=binding.machine_name,
        task_id=attempt.task_id,
        attempt_id=attempt.attempt_id,
        attempt_number=attempt.attempt_number,
        fencing_token=attempt.current_fencing_token,
        reservation_id=attempt.reservation_id,
        process_identity={
            field: stored["process"].get(field)
            for field in (
                "wrapper_pid",
                "wrapper_start_time_ticks",
                "process_group_id",
                "process_group_start_time_ticks",
            )
        },
        binding_signature=(
            binding.project_id,
            str(binding.shared_root),
            binding.machine_name,
            binding.registration_generation,
            binding.runtime_instance_id,
            binding.runtime_root,
        ),
        expected_task_revision=None,
        expected_attempt_digest=None,
        mutation_fence=lambda: None,
        replay_only=replay_only,
    )


@pytest.mark.parametrize("is_cpu", [False, True])
def test_orphan_recovery_commits_shared_truth_before_exact_local_retag(tmp_path: Path, is_cpu: bool) -> None:
    runtime, cfg, binding, revision, task, attempt, child, manifest, reservation = _orphan_case(
        tmp_path,
        is_cpu=is_cpu,
    )
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        recovered = _until(
            lambda: _advance_snapshot(coordinator, binding, revision, cfg, task, attempt, manifest, reservation),
            lambda value: value[1:] == ("running", 2, 2, 2),
        )
        assert recovered[0][binding.project_id]["outcome"] in {"recovered", "already_recovered"}
        assert child.poll() is None
    finally:
        coordinator.close()
        executor.shutdown()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_orphan_recovery_restart_finishes_local_effect_after_shared_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, cfg, binding, revision, task, attempt, child, manifest, reservation = _orphan_case(tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    original = project_io_supervision.retag_if_matches
    monkeypatch.setattr(project_io_supervision, "retag_if_matches", lambda *_args, **_kwargs: False)
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        committed = _until(
            lambda: _advance_snapshot(coordinator, binding, revision, cfg, task, attempt, manifest, reservation),
            lambda value: value[1] == "running" and value[2] == 2,
        )
        assert committed[3:] == (1, 1)
    finally:
        coordinator.close()

    monkeypatch.setattr(project_io_supervision, "retag_if_matches", original)
    restarted = AttemptSupervisionCoordinator(runtime, controller)
    try:
        _until(
            lambda: _advance_snapshot(restarted, binding, revision, cfg, task, attempt, manifest, reservation),
            lambda value: value[1:] == ("running", 2, 2, 2),
        )
        assert child.poll() is None
    finally:
        restarted.close()
        executor.shutdown()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_orphan_recovery_updates_manifest_after_local_capacity_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, cfg, binding, revision, task, attempt, child, manifest, reservation = _orphan_case(tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    original_advance = controller.advance_authority_orphan_recoveries

    def release_after_shared_commit(*args, **kwargs):
        result = original_advance(*args, **kwargs)
        if reservation.exists() and any(
            evidence.get("outcome") in {"recovered", "already_recovered"} for evidence in result.values()
        ):
            identity = ReservationIdentity.from_record(read_json(reservation)["reservation"])
            assert release_if_matches(runtime.root, identity, "local_process_exited")
        return result

    monkeypatch.setattr(controller, "advance_authority_orphan_recoveries", release_after_shared_commit)
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        _until(
            lambda: coordinator.advance_orphan_recoveries([binding], revision),
            lambda _value: (
                (load_task(cfg, task.task_id).claim_control.get("active_claim") or {}).get("fencing_token") == 2
                and read_json(manifest)["process"]["fencing_token"] == 2
            ),
        )
        released = local_paths(runtime.root)["released"] / reservation.name
        assert read_json(released)["reservation"]["fencing_token"] == 1
        assert read_json(manifest)["process"]["authority_state"] == "healthy"
    finally:
        coordinator.close()
        executor.shutdown()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


@pytest.mark.parametrize("is_cpu", [False, True])
@pytest.mark.parametrize("should_restart", [False, True])
def test_orphan_recovery_applies_committed_identity_after_process_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, is_cpu: bool, should_restart: bool
) -> None:
    runtime, cfg, binding, revision, task, attempt, child, manifest, reservation = _orphan_case(tmp_path, is_cpu=is_cpu)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    retag_name = "retag_cpu_if_matches" if is_cpu else "retag_if_matches"
    original = getattr(project_io_supervision, retag_name)
    monkeypatch.setattr(project_io_supervision, retag_name, lambda *_args, **_kwargs: False)
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        committed = _until(
            lambda: _advance_snapshot(coordinator, binding, revision, cfg, task, attempt, manifest, reservation),
            lambda value: value[1] == "running" and value[2] == 2,
        )
        assert committed[3:] == (1, 1)
        child.kill()
        child.wait(timeout=5)
        paths = local_paths(runtime.project_paths(binding.project_id)["root"])
        atomic_replace(
            paths["observations"] / f"{attempt.attempt_id}.json",
            {
                "exit_observation": {
                    "protocol_version": 1,
                    "task_id": task.task_id,
                    "attempt_id": attempt.attempt_id,
                    "observed_exit_code": 0,
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                }
            },
        )
        monkeypatch.setattr(project_io_supervision, retag_name, original)
        if should_restart:
            coordinator.close()
            coordinator = AttemptSupervisionCoordinator(runtime, controller)
        _until(
            lambda: _advance_snapshot(coordinator, binding, revision, cfg, task, attempt, manifest, reservation),
            lambda value: value[1:] == ("running", 2, 2, 2),
        )
        assert (paths["observations"] / f"{attempt.attempt_id}.json").exists()
    finally:
        coordinator.close()
        executor.shutdown()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_dead_orphan_replay_cannot_create_new_execution_authority(tmp_path: Path) -> None:
    runtime, cfg, binding, revision, task, attempt, child, manifest, reservation = _orphan_case(tmp_path)
    child.kill()
    child.wait(timeout=5)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        result = _until(
            lambda: _advance_snapshot(coordinator, binding, revision, cfg, task, attempt, manifest, reservation),
            lambda value: binding.project_id in value[0],
        )
        assert result[0][binding.project_id]["outcome"] == "stale"
        assert result[1:] == ("blocked", None, 1, 1)
    finally:
        coordinator.close()
        executor.shutdown()


def test_orphan_recovery_replays_attempt_first_partial_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, cfg, binding, _revision, task, attempt, child, _manifest, _reservation = _orphan_case(tmp_path)
    original = attempt_recovery.atomic_replace
    interrupted = False

    def interrupt_after_attempt_write(path, value):
        nonlocal interrupted
        original(path, value)
        if path == attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number) and not interrupted:
            interrupted = True
            raise OSError("injected after Attempt recovery publication")

    monkeypatch.setattr(attempt_recovery, "atomic_replace", interrupt_after_attempt_write)
    try:
        with pytest.raises(OSError, match="injected"):
            _recover_shared(cfg, binding, attempt)
        assert load_task(cfg, task.task_id).state["projection"] == "blocked"
        partial = read_json(attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number))["attempt"]
        assert partial["phase"] == "running"
        assert partial["current_fencing_token"] == 2

        monkeypatch.setattr(attempt_recovery, "atomic_replace", original)
        replay = _recover_shared(cfg, binding, attempt, replay_only=True)
        assert replay["outcome"] == "already_recovered"
        assert replay["recovered_fencing_token"] == 2
        assert load_task(cfg, task.task_id).claim_control["active_claim"]["fencing_token"] == 2
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_orphan_recovery_partial_replay_preserves_new_termination_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runtime, cfg, binding, _revision, task, attempt, child, _manifest, _reservation = _orphan_case(tmp_path)
    original = attempt_recovery.atomic_replace
    interrupted = False

    def interrupt_after_attempt_write(path, value):
        nonlocal interrupted
        original(path, value)
        if path == attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number) and not interrupted:
            interrupted = True
            raise OSError("injected after Attempt recovery publication")

    monkeypatch.setattr(attempt_recovery, "atomic_replace", interrupt_after_attempt_write)
    try:
        with pytest.raises(OSError, match="injected"):
            _recover_shared(cfg, binding, attempt)
        monkeypatch.setattr(attempt_recovery, "atomic_replace", original)
        cancelled = cancel_task(cfg, task.task_id, terminate_running=True)
        assert cancelled.control["terminate_running"] is True

        replay = _recover_shared(cfg, binding, attempt, replay_only=True)

        assert replay["outcome"] == "already_recovered"
        stored = load_task(cfg, task.task_id)
        assert stored.state["projection"] == "running"
        claim = stored.claim_control["active_claim"]
        assert claim["fencing_token"] == 2
        assert stored.control["terminate_running"] is True
        attempt_file = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
        encoded = attempt_file.read_bytes()
        recovered_attempt = read_json(attempt_file)["attempt"]
        termination = renew_project_io_attempt_lease(
            cfg,
            request_id="b" * 32,
            task_id=task.task_id,
            attempt_id=attempt.attempt_id,
            attempt_number=attempt.attempt_number,
            fencing_token=2,
            reservation_id=attempt.reservation_id,
            process_identity={
                field: recovered_attempt["process"].get(field)
                for field in (
                    "wrapper_pid",
                    "wrapper_start_time_ticks",
                    "process_group_id",
                    "process_group_start_time_ticks",
                )
            },
            expected_task_revision=stored.meta["revision"],
            expected_attempt_digest=hashlib.sha256(encoded).hexdigest(),
            mutation_fence=lambda: None,
        )
        assert termination["outcome"] == "termination_requested"
    finally:
        monkeypatch.setattr(attempt_recovery, "atomic_replace", original)
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_orphan_recovery_rejects_superseded_attempt_counter(tmp_path: Path) -> None:
    runtime, cfg, binding, _revision, task, attempt, child, _manifest, _reservation = _orphan_case(tmp_path)
    try:
        stored_task = load_task(cfg, task.task_id)
        stored_task.claim_control["fencing_epoch"] = 2
        stored_task.attempt_control.update(
            current_attempt_id=None,
            current_attempt_number=2,
            next_attempt_number=3,
        )
        stored_task.meta["revision"] += 1
        save_task(cfg, stored_task)

        result = _recover_shared(cfg, binding, attempt)
        assert result["outcome"] == "stale"
        assert result["reason"] == "recovery_target_mismatch"
        stored_attempt = read_json(attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number))["attempt"]
        assert stored_attempt["phase"] == "orphaned"
        assert stored_attempt["current_fencing_token"] == 1
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_orphan_recovery_restart_finishes_after_reservation_retag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, cfg, binding, revision, task, attempt, child, manifest, reservation = _orphan_case(tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    original = project_io_supervision.atomic_replace

    def fail_manifest_update(path, value):
        if path == manifest and value.get("process", {}).get("fencing_token") == 2:
            raise OSError("injected after reservation retag")
        return original(path, value)

    monkeypatch.setattr(project_io_supervision, "atomic_replace", fail_manifest_update)
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        _until(
            lambda: _advance_snapshot(coordinator, binding, revision, cfg, task, attempt, manifest, reservation),
            lambda value: value[1:4] == ("running", 2, 2),
        )
        assert read_json(manifest)["process"]["fencing_token"] == 1
    finally:
        coordinator.close()

    monkeypatch.setattr(project_io_supervision, "atomic_replace", original)
    restarted = AttemptSupervisionCoordinator(runtime, controller)
    try:
        _until(
            lambda: _advance_snapshot(restarted, binding, revision, cfg, task, attempt, manifest, reservation),
            lambda value: value[1:] == ("running", 2, 2, 2),
        )
    finally:
        restarted.close()
        executor.shutdown()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


@pytest.mark.parametrize("is_cpu", [False, True])
def test_orphan_recovery_fails_closed_for_active_released_twins(tmp_path: Path, is_cpu: bool) -> None:
    runtime, cfg, binding, revision, task, attempt, child, manifest, reservation = _orphan_case(
        tmp_path,
        is_cpu=is_cpu,
    )
    paths = local_paths(runtime.root)
    released = paths["cpu_released" if is_cpu else "released"] / reservation.name
    duplicate = read_json(reservation)
    duplicate["reservation"]["state"] = "released"
    duplicate["reservation"]["released_at"] = "2026-01-01T00:00:00Z"
    duplicate["reservation"]["release_reason"] = "injected_interrupted_release"
    atomic_replace(released, duplicate)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        for _ in range(5):
            coordinator.advance_orphan_recoveries([binding], revision)
        assert load_task(cfg, task.task_id).state["projection"] == "blocked"
        assert read_json(reservation)["reservation"]["fencing_token"] == 1
        assert read_json(manifest)["process"]["fencing_token"] == 1
    finally:
        coordinator.close()
        executor.shutdown()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


@pytest.mark.parametrize("is_cpu", [False, True])
def test_orphan_recovery_rechecks_twins_atomically_before_retag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    is_cpu: bool,
) -> None:
    runtime, cfg, binding, revision, task, attempt, child, manifest, reservation = _orphan_case(
        tmp_path,
        is_cpu=is_cpu,
    )
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    injected = False

    retag_name = "retag_cpu_if_matches" if is_cpu else "retag_if_matches"
    original_retag = getattr(project_io_supervision, retag_name)

    def inject_twin_before_locked_retag(runtime_root, identity, attempt_id, fencing_token):
        nonlocal injected
        if not injected:
            injected = True
            paths = local_paths(runtime.root)
            duplicate = read_json(reservation)
            duplicate["reservation"].update(
                state="released",
                released_at="2026-01-01T00:00:00Z",
                release_reason="injected_classification_retag_race",
            )
            atomic_replace(paths["cpu_released" if is_cpu else "released"] / reservation.name, duplicate)
        return original_retag(runtime_root, identity, attempt_id, fencing_token)

    monkeypatch.setattr(project_io_supervision, retag_name, inject_twin_before_locked_retag)
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        committed = _until(
            lambda: _advance_snapshot(coordinator, binding, revision, cfg, task, attempt, manifest, reservation),
            lambda _value: injected,
        )
        assert committed[1:3] == ("running", 2)
        assert committed[3:] == (1, 1)
    finally:
        coordinator.close()
        executor.shutdown()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_orphan_recovery_continues_for_disabled_existing_attempt(tmp_path: Path) -> None:
    runtime, cfg, binding, revision, task, attempt, child, manifest, reservation = _orphan_case(tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        disabled = replace(binding, enabled=False)
        _until(
            lambda: _advance_snapshot(coordinator, disabled, revision, cfg, task, attempt, manifest, reservation),
            lambda value: value[1:] == ("running", 2, 2, 2),
        )
    finally:
        coordinator.close()
        executor.shutdown()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_orphan_recovery_rejects_replaced_binding_generation(tmp_path: Path) -> None:
    runtime, cfg, binding, revision, task, attempt, child, manifest, reservation = _orphan_case(tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        replaced = replace(binding, registration_generation="replacement-generation")
        for _ in range(5):
            coordinator.advance_orphan_recoveries([replaced], revision + 1)
        assert load_task(cfg, task.task_id).state["projection"] == "blocked"
        assert read_json(reservation)["reservation"]["fencing_token"] == 1
        assert read_json(manifest)["process"]["fencing_token"] == 1
    finally:
        coordinator.close()
        executor.shutdown()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_orphan_recovery_scan_state_is_bounded_beyond_sixty_four_bindings(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    revision, _bindings = runtime.load_registry()
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    bindings = [replace(binding, project_id=f"project-{index:03d}") for index in range(300)]
    try:
        for _ in range(6):
            assert coordinator.advance_orphan_recoveries(bindings, revision) == {}
        assert len(coordinator._orphan_recovery_scans) <= 256
        assert len(coordinator._orphan_recovery_entries) <= 256
    finally:
        coordinator.close()
        executor.shutdown()


def test_orphan_recovery_scan_reaches_tail_pages_beyond_state_capacity(
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
        processes = runtime.project_paths(candidate.project_id)["processes"]
        for index in range(3):
            atomic_replace(processes / f"bad-{index}.json", {"unexpected": {}})

    tail_processes = runtime.project_paths(bindings[-1].project_id)["processes"]
    tail_paths: list[Path] = []
    original = project_io_supervision._read_process_manifest

    def record_tail(runtime_arg, binding_arg, path, signature):
        if path.parent == tail_processes:
            tail_paths.append(path)
        return original(runtime_arg, binding_arg, path, signature)

    monkeypatch.setattr(project_io_supervision, "_read_process_manifest", record_tail)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    try:
        for _ in range(30):
            coordinator.advance_orphan_recoveries(bindings, revision)

        assert len(set(tail_paths)) >= 2
        assert len(coordinator._orphan_recovery_scans) <= 256
    finally:
        coordinator.close()
        executor.shutdown()
