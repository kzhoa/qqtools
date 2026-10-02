from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.agent import project_io_worker as worker
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.project_io_controller import ProjectIOController
from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor
from qqtools.plugins.qexp.agent.project_io_protocol import ProjectIOProcess
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.runtime import running_publication
from qqtools.plugins.qexp.runtime.paths import attempt_path, local_paths, task_path
from qqtools.plugins.qexp.runtime.records import AttemptRecord, utc_now
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task, save_task
from qqtools.plugins.qexp.scheduler import claim_task

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _case(tmp_path: Path, *, disabled: bool = False):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    task = submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    path = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    attempt.phase = "starting"
    atomic_replace(path, attempt.to_dict())
    task = load_task(cfg, task.task_id)
    task.claim_control["active_claim"]["launch_state"] = "starting"
    save_task(cfg, task)
    if disabled:
        runtime.set_enabled(binding.project_id, False)
    revision, bindings = runtime.load_registry()
    parameters = {
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
        "process_created_at": "2026-09-29T00:00:00+00:00",
    }
    root = runtime.project_paths(binding.project_id)["root"]
    registration_path = local_paths(root)["registrations"] / f"{attempt.attempt_id}.json"
    atomic_replace(
        registration_path,
        {
            "process_registration": {
                **{key: value for key, value in parameters.items() if key != "process_identity"},
                **parameters["process_identity"],
                "protocol_version": 1,
            }
        },
    )
    return runtime, cfg, bindings, revision, parameters, path, registration_path


def _until(operation, predicate, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = operation()
        if predicate(value):
            return value
        time.sleep(0.02)
    raise AssertionError("running publication did not complete")


def _result(executor, request):
    def poll():
        executor.poll()
        return executor.load_result(request.request_id)

    return _until(poll, lambda result: result is not None)


@pytest.mark.parametrize("disabled", [False, True])
def test_controller_publishes_running_registration_without_local_effects(tmp_path: Path, disabled: bool) -> None:
    runtime, cfg, bindings, revision, parameters, path, registration = _case(tmp_path, disabled=disabled)
    binding = bindings[0]
    original_registration = registration.read_bytes()
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    try:
        _until(lambda: controller.advance_binding_validation(bindings, revision), bool)

        def advance():
            return controller.advance_authority_running_publications(
                bindings, revision, {binding.project_id: parameters}
            )

        result = _until(advance, lambda value: binding.project_id in value)[binding.project_id]
        assert result["outcome"] == "processed"
        assert result["transitioned_to_running"] is True
        assert result["authority_granted"] is False
        assert result["local_effects"] == ()
        stored = AttemptRecord.from_dict(read_json(path))
        assert stored.phase == "running"
        assert load_task(cfg, parameters["task_id"]).claim_control["active_claim"]["launch_state"] == "running"
        manifest = local_paths(runtime.project_paths(binding.project_id)["root"])["processes"] / (
            f"{parameters['attempt_id']}.json"
        )
        assert stored.process["local_process_manifest"] == str(manifest)
        assert not manifest.exists(), "shared worker must not materialize local process evidence"
        before = path.read_bytes(), task_path(cfg.shared_root, parameters["task_id"]).read_bytes()
        replay = _until(advance, lambda value: binding.project_id in value)[binding.project_id]
        assert replay["transitioned_to_running"] is False
        assert (path.read_bytes(), task_path(cfg.shared_root, parameters["task_id"]).read_bytes()) == before
        assert registration.read_bytes() == original_registration
        assert not executor.has_unfinished_work()
    finally:
        executor.shutdown()


def test_running_publication_conflict_does_not_grant_current_authority(tmp_path: Path) -> None:
    runtime, cfg, bindings, revision, parameters, path, _registration = _case(tmp_path)
    stored = AttemptRecord.from_dict(read_json(path))
    stored.process["process_group_id"] = 999
    atomic_replace(path, stored.to_dict())
    before = path.read_bytes(), task_path(cfg.shared_root, parameters["task_id"]).read_bytes()
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    try:
        request = executor.prepare_authority_running_publish(bindings[0], revision, **parameters)
        executor.start(request.request_id)
        result = _result(executor, request)
        assert result.status == "completed"
        assert result.evidence["outcome"] == "processed"
        assert result.evidence["transitioned_to_running"] is False
        assert result.evidence["authority_granted"] is False
        assert (path.read_bytes(), task_path(cfg.shared_root, parameters["task_id"]).read_bytes()) == before
    finally:
        executor.shutdown()


def test_exited_running_publication_retains_ambiguity_until_exact_retry(tmp_path: Path) -> None:
    runtime, cfg, bindings, revision, parameters, path, registration = _case(tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    try:
        request = executor.prepare_authority_running_publish(bindings[0], revision, **parameters)
        executor._write_record(
            executor._record_path("processes", request.request_id),
            ProjectIOProcess(request, 2_000_000_000, 1, utc_now(), "running").to_dict(),
            "project_io_process",
        )
        executor.poll()
        result = executor.load_result(request.request_id)
        assert result.status == "outcome_unknown"
        assert executor.consume(request.request_id, request) is None
        assert request in executor.unresolved_requests()
        assert registration.exists()
        assert executor.reset_ambiguous_authority_running_publish_for_retry(request.request_id, request)
        executor.start(request.request_id)
        assert _result(executor, request).status == "completed"
        assert AttemptRecord.from_dict(read_json(path)).phase == "running"
        assert load_task(cfg, parameters["task_id"]).claim_control["active_claim"]["launch_state"] == "running"
    finally:
        executor.shutdown()


@pytest.mark.parametrize("boundary", ["before_task", "before_result"])
def test_epoch_fence_preserves_unknown_and_restart_repairs_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    runtime, cfg, bindings, revision, parameters, path, registration = _case(tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    request = executor.prepare_authority_running_publish(bindings[0], revision, **parameters)
    ticks = worker._process_start_time_ticks(os.getpid())
    assert ticks is not None
    process = ProjectIOProcess(request, os.getpid(), ticks, utc_now(), "running")
    executor._write_record(
        executor._record_path("processes", request.request_id), process.to_dict(), "project_io_process"
    )
    original_save = running_publication.save_task
    original_publish = worker._publish_result

    def fence_before_task(*args, **kwargs):
        executor.fence_epoch()
        return original_save(*args, **kwargs)

    def fence_before_result(*args, **kwargs):
        executor.fence_epoch()
        return original_publish(*args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            if boundary == "before_task":
                patch.setattr(running_publication, "save_task", fence_before_task)
            else:
                patch.setattr(worker, "_publish_result", fence_before_result)
            assert worker._run(runtime.root, request.request_id) == 0
        result = executor.load_result(request.request_id)
        assert result.status == "outcome_unknown"
        assert AttemptRecord.from_dict(read_json(path)).phase == "running"
        assert load_task(cfg, parameters["task_id"]).claim_control["active_claim"]["launch_state"] == (
            "starting" if boundary == "before_task" else "running"
        )
    finally:
        # This worker call used the test process: never let shutdown signal it.
        executor._write_record(
            executor._record_path("processes", request.request_id),
            replace(process, pid=2_000_000_000, start_time_ticks=1).to_dict(),
            "project_io_process",
        )
        executor.shutdown()

    attempt_before_replay = path.read_bytes()
    registration_before_replay = registration.read_bytes()
    restarted = ProjectIOExecutor(runtime)
    restarted.begin_epoch()
    try:
        current = restarted.prepare_authority_running_publish(bindings[0], revision, **parameters)
        restarted.start(current.request_id)
        result = _result(restarted, current)
        assert result.status == "completed"
        assert result.evidence["transitioned_to_running"] is False
        assert path.read_bytes() == attempt_before_replay
        assert registration.read_bytes() == registration_before_replay
        assert load_task(cfg, parameters["task_id"]).claim_control["active_claim"]["launch_state"] == "running"
    finally:
        restarted.shutdown()
