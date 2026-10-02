from __future__ import annotations

import time
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.project_io_controller import ProjectIOController
from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor
from qqtools.plugins.qexp.agent.project_io_supervision import AttemptSupervisionCoordinator
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.runtime.paths import attempt_path, local_paths
from qqtools.plugins.qexp.runtime.process_evidence import ProcessEvidence
from qqtools.plugins.qexp.runtime.records import AttemptRecord
from qqtools.plugins.qexp.runtime.responsibility_cleanup import evidence_write_guard
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task, save_task
from qqtools.plugins.qexp.runtime.termination import attempt_control_lock
from qqtools.plugins.qexp.scheduler import claim_task

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _until(operation, predicate, *, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = operation()
        if predicate(value):
            return value
        time.sleep(0.02)
    raise AssertionError("local supervision running publication did not complete")


def _case(tmp_path: Path, *, name: str = "project", disabled: bool = False):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg = init_shared_root(tmp_path / name / ".qexp", "gpu-1")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    task = submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    stored = AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task.task_id, 1)))
    stored.phase = "starting"
    atomic_replace(attempt_path(cfg.shared_root, task.task_id, 1), stored.to_dict())
    task = load_task(cfg, task.task_id)
    task.claim_control["active_claim"]["launch_state"] = "starting"
    save_task(cfg, task)
    if disabled:
        runtime.set_enabled(binding.project_id, False)
    revision, bindings = runtime.load_registry()
    binding = bindings[0]
    root = runtime.project_paths(binding.project_id)["root"]
    paths = local_paths(root)
    registration = {
        "protocol_version": 1,
        "machine_name": binding.machine_name,
        "task_id": task.task_id,
        "attempt_id": attempt.attempt_id,
        "fencing_token": attempt.current_fencing_token,
        "reservation_id": attempt.reservation_id,
        "wrapper_pid": 101,
        "wrapper_start_time_ticks": 202,
        "process_group_id": 303,
        "process_group_start_time_ticks": 404,
        "process_created_at": "2026-09-29T00:00:00+00:00",
    }
    atomic_replace(
        paths["registrations"] / f"{attempt.attempt_id}.json",
        {"process_registration": registration},
    )
    return runtime, cfg, binding, revision, task, attempt, paths, registration


@pytest.mark.parametrize("disabled", [False, True], ids=["enabled", "disabled"])
def test_running_coordinator_releases_local_guards_before_typed_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    disabled: bool,
) -> None:
    runtime, cfg, binding, revision, task, attempt, paths, registration = _case(tmp_path, disabled=disabled)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        local_cfg = controller.validated_config(binding, revision)
        assert local_cfg is not None
        original = controller.advance_authority_running_publications
        calls = []

        def checked_publication(bindings, current_revision, registrations):
            with evidence_write_guard(local_cfg.runtime_root, attempt.attempt_id) as acquired:
                assert acquired, "typed publication retained the local evidence guard"
            with attempt_control_lock(local_cfg, attempt.attempt_id, blocking=False) as acquired:
                assert acquired, "typed publication retained the local process-control lock"
            calls.append((tuple(bindings), current_revision, registrations))
            return original(bindings, current_revision, registrations)

        monkeypatch.setattr(controller, "advance_authority_running_publications", checked_publication)
        result = _until(
            lambda: coordinator.advance_running_publications([binding], revision),
            lambda value: binding.project_id in value,
        )[binding.project_id]

        assert result["outcome"] == "processed"
        assert result["authority_granted"] is False
        assert calls
        parameters = calls[-1][2][binding.project_id]
        assert parameters == {
            "task_id": task.task_id,
            "attempt_id": attempt.attempt_id,
            "attempt_number": attempt.attempt_number,
            "fencing_token": attempt.current_fencing_token,
            "reservation_id": attempt.reservation_id,
            "process_identity": {
                "wrapper_pid": 101,
                "wrapper_start_time_ticks": 202,
                "process_group_id": 303,
                "process_group_start_time_ticks": 404,
            },
            "process_created_at": registration["process_created_at"],
        }
        manifest = paths["processes"] / f"{attempt.attempt_id}.json"
        process = read_json(manifest)["process"]
        assert process["observed_state"] == "running"
        assert process["authority_state"] == "healthy"
        assert process["reservation_id"] == attempt.reservation_id
        assert AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task.task_id, 1))).phase == "running"
        assert load_task(cfg, task.task_id).claim_control["active_claim"]["launch_state"] == "running"
        assert not executor.has_unfinished_work()
    finally:
        coordinator.close()
        executor.shutdown()


def test_running_publication_remains_eligible_across_missed_service_grants(tmp_path: Path, monkeypatch) -> None:
    runtime, cfg, binding, revision, task, attempt, paths, _registration = _case(tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    original = controller.advance_authority_running_publications
    offers = []

    def decline(bindings, registry_revision, publications):
        assert registry_revision == revision
        assert list(bindings) == [binding]
        offers.append(publications[binding.project_id])
        return {}

    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        monkeypatch.setattr(controller, "advance_authority_running_publications", decline)
        for _ in range(6):
            assert coordinator.advance_running_publications([binding], revision) == {}
        assert len(offers) == 6, "a consumed directory page must not erase unselected work"
        assert all(value["attempt_id"] == attempt.attempt_id for value in offers)
        assert AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task.task_id, 1))).phase == "starting"
        monkeypatch.setattr(controller, "advance_authority_running_publications", original)
        _until(
            lambda: coordinator.advance_running_publications([binding], revision),
            lambda value: value.get(binding.project_id, {}).get("outcome") == "processed",
        )
        assert not coordinator._running_publication_intents
        assert AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task.task_id, 1))).phase == "running"
    finally:
        coordinator.close()
        executor.shutdown()


def test_running_coordinator_retains_replay_after_controller_failure(tmp_path: Path, monkeypatch) -> None:
    runtime, cfg, binding, revision, task, attempt, paths, _registration = _case(tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        original = controller.advance_authority_running_publications

        def unavailable(*_args, **_kwargs):
            raise OSError("typed publication unavailable")

        monkeypatch.setattr(controller, "advance_authority_running_publications", unavailable)
        with pytest.raises(OSError, match="typed publication unavailable"):
            coordinator.advance_running_publications([binding], revision)
        manifest = paths["processes"] / f"{attempt.attempt_id}.json"
        manifest_before = manifest.read_bytes()
        registration_path = paths["registrations"] / manifest.name
        registration_before = registration_path.read_bytes()
        assert AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task.task_id, 1))).phase == "starting"

        monkeypatch.setattr(controller, "advance_authority_running_publications", original)
        result = _until(
            lambda: coordinator.advance_running_publications([binding], revision),
            lambda value: binding.project_id in value,
        )[binding.project_id]
        assert result["outcome"] == "processed"
        assert manifest.read_bytes() == manifest_before
        assert registration_path.read_bytes() == registration_before
        assert AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task.task_id, 1))).phase == "running"
    finally:
        coordinator.close()
        executor.shutdown()


def test_running_coordinator_pending_control_defers_without_local_or_shared_effect(
    tmp_path: Path,
) -> None:
    runtime, cfg, binding, revision, task, attempt, paths, _registration = _case(tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        for _ in range(4):
            assert (
                coordinator.advance_running_publications([binding], revision, pending_attempt_ids={attempt.attempt_id})
                == {}
            )
        assert not (paths["processes"] / f"{attempt.attempt_id}.json").exists()
        assert AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task.task_id, 1))).phase == "starting"

        result = _until(
            lambda: coordinator.advance_running_publications([binding], revision),
            lambda value: binding.project_id in value,
        )
        assert result[binding.project_id]["outcome"] == "processed"
    finally:
        coordinator.close()
        executor.shutdown()


def test_supervision_fast_exit_materializes_running_before_terminal_completion(tmp_path: Path) -> None:
    runtime, cfg, binding, revision, task, attempt, paths, _registration = _case(tmp_path)
    atomic_replace(
        paths["observations"] / f"{attempt.attempt_id}.json",
        {
            "exit_observation": {
                "protocol_version": 1,
                "task_id": task.task_id,
                "attempt_id": attempt.attempt_id,
                "observed_exit_code": 0,
                "observed_at": "2026-09-29T00:00:01+00:00",
            }
        },
    )
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        _until(
            lambda: coordinator.advance_all([binding], revision),
            lambda _value: (
                load_task(cfg, task.task_id).state["projection"] == "succeeded" and not executor.has_unfinished_work()
            ),
        )

        assert (paths["processes"] / f"{attempt.attempt_id}.json").exists()
        assert load_task(cfg, task.task_id).state["projection"] == "succeeded"
        assert not executor.has_unfinished_work()
    finally:
        coordinator.close()
        executor.shutdown()


def test_supervision_restart_retires_obsolete_running_request_before_terminal_progress(tmp_path: Path) -> None:
    runtime, cfg, binding, revision, task, attempt, paths, registration = _case(tmp_path)
    atomic_replace(
        paths["processes"] / f"{attempt.attempt_id}.json",
        {
            "process": {
                **registration,
                "observed_state": "exited",
                "observed_exit_code": 0,
                "observed_exited_at": "2026-09-29T00:00:01+00:00",
                "supervisor": "agent",
                "authority_state": "healthy",
                "created_by": "agent",
            }
        },
    )
    atomic_replace(
        paths["observations"] / f"{attempt.attempt_id}.json",
        {
            "exit_observation": {
                "protocol_version": 1,
                "task_id": task.task_id,
                "attempt_id": attempt.attempt_id,
                "observed_exit_code": 0,
                "observed_at": "2026-09-29T00:00:01+00:00",
            }
        },
    )
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        obsolete = executor.prepare_authority_running_publish(
            binding,
            revision,
            task_id=task.task_id,
            attempt_id=attempt.attempt_id,
            attempt_number=attempt.attempt_number,
            fencing_token=attempt.current_fencing_token,
            reservation_id=attempt.reservation_id,
            process_identity={
                "wrapper_pid": registration["wrapper_pid"],
                "wrapper_start_time_ticks": registration["wrapper_start_time_ticks"],
                "process_group_id": registration["process_group_id"],
                "process_group_start_time_ticks": registration["process_group_start_time_ticks"],
            },
            process_created_at=registration["process_created_at"],
        )

        _until(
            lambda: coordinator.advance_all([binding], revision),
            lambda _value: obsolete not in executor.unresolved_requests(),
        )

        assert obsolete not in executor.unresolved_requests()
        _until(
            lambda: coordinator.advance_all([binding], revision),
            lambda _value: any(
                request.operation_kind.startswith("authority_terminal_") for request in executor.unresolved_requests()
            ),
        )
    finally:
        coordinator.close()
        executor.shutdown()


def test_exited_attempt_does_not_retire_another_attempts_running_request(tmp_path: Path) -> None:
    runtime, cfg, binding, revision, _task, exited_attempt, paths, exited_registration = _case(tmp_path)
    atomic_replace(
        paths["processes"] / f"{exited_attempt.attempt_id}.json",
        {
            "process": {
                **exited_registration,
                "observed_state": "exited",
                "observed_exit_code": 0,
                "observed_exited_at": "2026-09-29T00:00:01+00:00",
                "supervisor": "agent",
                "authority_state": "healthy",
                "created_by": "agent",
            }
        },
    )
    current_task = submit(cfg, ["echo", "current"], working_dir=tmp_path)
    current_attempt = claim_task(cfg, current_task.task_id, [1])
    assert current_attempt is not None
    current_attempt.phase = "starting"
    atomic_replace(
        attempt_path(cfg.shared_root, current_task.task_id, current_attempt.attempt_number),
        current_attempt.to_dict(),
    )
    current_task = load_task(cfg, current_task.task_id)
    current_task.claim_control["active_claim"]["launch_state"] = "starting"
    save_task(cfg, current_task)

    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        current = executor.prepare_authority_running_publish(
            binding,
            revision,
            task_id=current_task.task_id,
            attempt_id=current_attempt.attempt_id,
            attempt_number=current_attempt.attempt_number,
            fencing_token=current_attempt.current_fencing_token,
            reservation_id=current_attempt.reservation_id,
            process_identity={
                "wrapper_pid": 501,
                "wrapper_start_time_ticks": 502,
                "process_group_id": 503,
                "process_group_start_time_ticks": 504,
            },
            process_created_at="2026-09-29T00:00:02+00:00",
        )

        for _ in range(4):
            assert coordinator.advance_running_publications([binding], revision) == {}

        assert current in executor.unresolved_requests()
    finally:
        coordinator.close()
        executor.shutdown()


def test_running_coordinator_does_not_republish_exited_launch_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _cfg, binding, revision, task, attempt, paths, registration = _case(tmp_path)
    process = {
        **registration,
        "observed_state": "exited",
        "observed_exit_code": 0,
        "observed_exited_at": "2026-09-29T00:00:01+00:00",
        "supervisor": "agent",
        "authority_state": "healthy",
        "created_by": "agent",
    }
    atomic_replace(paths["processes"] / f"{attempt.attempt_id}.json", {"process": process})
    atomic_replace(
        paths["launch_intents"] / f"{attempt.attempt_id}.json",
        {
            "launch_intent": {
                "protocol_version": 1,
                "attempt_id": attempt.attempt_id,
                "task_id": task.task_id,
                "fencing_token": attempt.current_fencing_token,
                "launch_id": "launch-1",
                "wrapper_pid": registration["wrapper_pid"],
                "wrapper_start_time_ticks": registration["wrapper_start_time_ticks"],
                "gpu_ids": [0],
                "command": ["echo", "ok"],
                "working_directory": str(tmp_path),
                "lease_expires_at": None,
                "created_at": "2026-09-29T00:00:00+00:00",
            }
        },
    )
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)

    def reject_running_publication(*_args, **_kwargs):
        raise AssertionError("exited launch intent was republished as running")

    monkeypatch.setattr(controller, "advance_authority_running_publications", reject_running_publication)
    try:
        for _ in range(4):
            assert coordinator.advance_running_publications([binding], revision) == {}
        assert not executor.has_unfinished_work()
    finally:
        coordinator.close()
        executor.shutdown()


def test_running_coordinator_joins_real_launch_intent_to_later_registration(tmp_path: Path) -> None:
    runtime, _cfg, binding, revision, task, attempt, paths, registration = _case(tmp_path)
    launch_intent = {
        "protocol_version": 1,
        "attempt_id": attempt.attempt_id,
        "task_id": task.task_id,
        "fencing_token": attempt.current_fencing_token,
        "launch_id": "launch-1",
        "wrapper_pid": registration["wrapper_pid"],
        "wrapper_start_time_ticks": registration["wrapper_start_time_ticks"],
        "gpu_ids": [0],
        "command": ["echo", "ok"],
        "working_directory": str(tmp_path),
        "lease_expires_at": None,
        "created_at": "2026-09-29T00:00:00+00:00",
    }
    atomic_replace(
        paths["launch_intents"] / f"{attempt.attempt_id}.json",
        {"launch_intent": launch_intent},
    )
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)
    try:
        _until(lambda: controller.advance_binding_validation([binding], revision), bool)
        assert (
            coordinator.advance_running_publications([binding], revision, pending_attempt_ids={attempt.attempt_id})
            == {}
        )
        result = _until(
            lambda: coordinator.advance_running_publications([binding], revision),
            lambda value: binding.project_id in value,
        )
        assert result[binding.project_id]["outcome"] == "processed"
        process = read_json(paths["processes"] / f"{attempt.attempt_id}.json")["process"]
        assert process["process_group_id"] == registration["process_group_id"]
        assert process["observed_state"] == "running"
    finally:
        coordinator.close()
        executor.shutdown()


def test_running_coordinator_materializes_unverifiable_intent_without_shared_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, cfg, binding, revision, task, attempt, paths, _registration = _case(tmp_path)
    (paths["registrations"] / f"{attempt.attempt_id}.json").unlink()
    launch_intent = {
        "protocol_version": 1,
        "attempt_id": attempt.attempt_id,
        "task_id": task.task_id,
        "fencing_token": attempt.current_fencing_token,
        "launch_id": "launch-1",
        "wrapper_pid": 101,
        "wrapper_start_time_ticks": 202,
        "gpu_ids": [0],
        "command": ["echo", "ok"],
        "working_directory": str(tmp_path),
        "lease_expires_at": None,
        "created_at": "2026-09-29T00:00:00+00:00",
    }
    atomic_replace(
        paths["launch_intents"] / f"{attempt.attempt_id}.json",
        {"launch_intent": launch_intent},
    )
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)

    def unknown_wrapper(_record):
        return ProcessEvidence(state="unknown", reason="read_failed")

    def reject_publication(*_args, **_kwargs):
        raise AssertionError("unverifiable local launch attempted shared publication")

    monkeypatch.setattr(
        "qqtools.plugins.qexp.agent.project_io_supervision.inspect_wrapper_identity",
        unknown_wrapper,
    )
    monkeypatch.setattr(controller, "advance_authority_running_publications", reject_publication)
    try:
        for _ in range(4):
            coordinator.advance_running_publications([binding], revision)
            manifest = paths["processes"] / f"{attempt.attempt_id}.json"
            if manifest.exists():
                break
        process = read_json(manifest)["process"]
        assert process["observed_state"] == "launch_unverifiable"
        assert process["authority_state"] == "isolated"
        assert process["created_by"] == "agent"
        assert AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task.task_id, 1))).phase == "starting"
    finally:
        coordinator.close()
        executor.shutdown()


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(lambda envelope: envelope.update(extra=True), id="extra-envelope-field"),
        pytest.param(
            lambda envelope: envelope["process_registration"].update(protocol_version=True),
            id="boolean-protocol",
        ),
        pytest.param(
            lambda envelope: envelope["process_registration"].update(process_created_at="not-a-timestamp"),
            id="invalid-timestamp",
        ),
        pytest.param(
            lambda envelope: envelope["process_registration"].pop("process_group_start_time_ticks"),
            id="missing-process-identity",
        ),
    ],
)
def test_running_coordinator_rejects_malformed_registration_before_local_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
) -> None:
    runtime, _cfg, binding, revision, _task, attempt, paths, _registration = _case(tmp_path)
    registration_path = paths["registrations"] / f"{attempt.attempt_id}.json"
    envelope = read_json(registration_path)
    mutation(envelope)
    atomic_replace(registration_path, envelope)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    coordinator = AttemptSupervisionCoordinator(runtime, controller)

    def reject_publication(*_args, **_kwargs):
        raise AssertionError("malformed registration reached typed publication")

    monkeypatch.setattr(controller, "advance_authority_running_publications", reject_publication)
    try:
        for _ in range(4):
            assert coordinator.advance_running_publications([binding], revision) == {}
        assert not (paths["processes"] / f"{attempt.attempt_id}.json").exists()
        assert registration_path.exists()
    finally:
        coordinator.close()
        executor.shutdown()
