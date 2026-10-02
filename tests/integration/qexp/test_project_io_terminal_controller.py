from __future__ import annotations

import hashlib
import os
import time
import uuid
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.project_io_controller import ProjectIOController
from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor
from qqtools.plugins.qexp.agent.project_io_protocol import authority_terminal_transition_digest
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.commands.task import retry
from qqtools.plugins.qexp.runtime.paths import attempt_path
from qqtools.plugins.qexp.runtime.records import AttemptRecord
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task, save_task
from qqtools.plugins.qexp.scheduler import claim_task, expire_claim, fail_attempt

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _until(operation, predicate, *, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = operation()
        if predicate(result):
            return result
        time.sleep(0.02)
    raise AssertionError("isolated service did not complete within its deadline")


def test_terminal_controller_consumes_historical_settlement_for_disabled_binding(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    task = submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert fail_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "nonzero_exit")
    retry(cfg, task.task_id)
    successor = claim_task(cfg, task.task_id, [0])
    assert successor is not None
    runtime.set_enabled(binding.project_id, False)
    revision, bindings = runtime.load_registry()
    old_path = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    successor_path = attempt_path(cfg.shared_root, task.task_id, successor.attempt_number)
    before = old_path.read_bytes(), successor_path.read_bytes(), load_task(cfg, task.task_id).to_dict()
    identity = {
        "task_id": task.task_id,
        "attempt_id": attempt.attempt_id,
        "attempt_number": attempt.attempt_number,
        "fencing_token": attempt.current_fencing_token,
        "reservation_id": attempt.reservation_id,
        "process_identity": {
            "wrapper_pid": None,
            "wrapper_start_time_ticks": None,
            "process_group_id": None,
            "process_group_start_time_ticks": None,
        },
        "mode": "active",
    }
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    try:
        _until(lambda: controller.advance_binding_validation(bindings, revision), lambda value: bool(value))
        observed = _until(
            lambda: controller.advance_authority_terminal_observations(
                bindings, revision, {binding.project_id: identity}
            ),
            lambda value: binding.project_id in value,
        )[binding.project_id]
        assert observed["outcome"] == "settled_terminal"
        assert observed["attempt_id"] == attempt.attempt_id
        assert observed["source_revisions"] == {
            "task": before[2]["meta"]["revision"],
            "attempt_digest": hashlib.sha256(before[0]).hexdigest(),
        }
        assert observed["authority_granted"] is False
        assert observed["local_effects"] in ([], ())
        assert (old_path.read_bytes(), successor_path.read_bytes(), load_task(cfg, task.task_id).to_dict()) == before
        assert not executor.has_unfinished_work()
    finally:
        executor.shutdown()


@pytest.mark.parametrize(
    ("attempt_phase", "expected_outcome"),
    [("orphaned", "current"), ("running", "stale")],
)
def test_terminal_controller_observes_real_detached_orphan_identity(
    tmp_path: Path,
    attempt_phase: str,
    expected_outcome: str,
) -> None:
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
    running = load_task(cfg, task.task_id)
    running.state.update(projection="running", reason="running")
    running.claim_control["active_claim"].update(
        launch_state="running",
        lease_expires_at="2000-01-01T00:00:00Z",
        clock_error_bound_seconds=0.0,
    )
    running.meta["revision"] += 1
    save_task(cfg, running)
    assert expire_claim(
        cfg,
        task.task_id,
        attempt.attempt_id,
        attempt.current_fencing_token,
        reservation_runtime_root=runtime.root,
    )
    orphaned = load_task(cfg, task.task_id)
    assert orphaned.state["projection"] == "blocked"
    assert orphaned.attempt_control["current_attempt_id"] is None
    if attempt_phase != "orphaned":
        inconsistent = AttemptRecord.from_dict(read_json(attempt_file))
        inconsistent.phase = attempt_phase
        atomic_replace(attempt_file, inconsistent.to_dict())
    revision, bindings = runtime.load_registry()
    identity = {
        "task_id": task.task_id,
        "attempt_id": attempt.attempt_id,
        "attempt_number": attempt.attempt_number,
        "fencing_token": attempt.current_fencing_token,
        "reservation_id": attempt.reservation_id,
        "process_identity": process_identity,
        "mode": "detached_orphan",
    }
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    try:
        _until(lambda: controller.advance_binding_validation(bindings, revision), bool)
        observed = _until(
            lambda: controller.advance_authority_terminal_observations(
                bindings,
                revision,
                {binding.project_id: identity},
            ),
            lambda value: binding.project_id in value,
        )[binding.project_id]
        assert observed["outcome"] == expected_outcome
        assert observed["task_phase"] == "blocked"
        assert observed["attempt_phase"] == attempt_phase
        if attempt_phase == "running":
            semantic = {
                **{key: value for key, value in identity.items() if key != "mode"},
                "mode": "detached_orphan",
                "phase": "succeeded",
                "reason": "completed",
                "exit_code": 0,
                "termination_result": None,
            }
            transition = {
                **semantic,
                "source_revisions": dict(observed["source_revisions"]),
                "transition_digest": authority_terminal_transition_digest(
                    machine_name=binding.machine_name,
                    **semantic,
                ),
            }
            published = _until(
                lambda: controller.advance_authority_terminal_publications(
                    bindings,
                    revision,
                    {binding.project_id: transition},
                ),
                lambda value: binding.project_id in value,
            )[binding.project_id]
            assert published["outcome"] == "stale"
            assert load_task(cfg, task.task_id).state["projection"] == "blocked"
    finally:
        executor.shutdown()


@pytest.mark.parametrize("disabled", [False, True], ids=["enabled", "disabled"])
@pytest.mark.parametrize("missing_identity", [None, "wrapper", "group"])
def test_terminal_controller_completes_observation_commit_and_publication(
    tmp_path: Path, disabled: bool, missing_identity: str | None
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    task = submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    path = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    stored = AttemptRecord.from_dict(read_json(path))
    stored.phase = "running"
    process_identity = {
        "wrapper_pid": 101,
        "wrapper_start_time_ticks": 202,
        "process_group_id": 303,
        "process_group_start_time_ticks": 404,
    }
    if missing_identity == "wrapper":
        process_identity.update(wrapper_pid=None, wrapper_start_time_ticks=None)
    elif missing_identity == "group":
        process_identity.update(process_group_id=None, process_group_start_time_ticks=None)
    stored.process.update(process_identity)
    atomic_replace(path, stored.to_dict())
    task = load_task(cfg, task.task_id)
    task.state.update(projection="running", reason="running")
    task.claim_control["active_claim"]["launch_state"] = "running"
    task.meta["revision"] += 1
    save_task(cfg, task)
    if disabled:
        runtime.set_enabled(binding.project_id, False)
    revision, bindings = runtime.load_registry()
    binding = bindings[0]
    identity = {
        "task_id": task.task_id,
        "attempt_id": attempt.attempt_id,
        "attempt_number": attempt.attempt_number,
        "fencing_token": attempt.current_fencing_token,
        "reservation_id": attempt.reservation_id,
        "process_identity": process_identity,
    }
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    try:
        _until(lambda: controller.advance_binding_validation(bindings, revision), lambda value: bool(value))
        observed = _until(
            lambda: controller.advance_authority_terminal_observations(
                bindings, revision, {binding.project_id: {**identity, "mode": "active"}}
            ),
            lambda value: binding.project_id in value,
        )[binding.project_id]
        assert observed["outcome"] == "current"
        assert observed["authority_granted"] is False
        decision = {
            **identity,
            "decision_id": uuid.uuid4().hex,
            "decision_token": attempt.current_fencing_token,
            "authority_outcome": "lease_expired",
            "reason": "authority_lost",
            "source_revisions": dict(observed["source_revisions"]),
        }
        committed = _until(
            lambda: controller.advance_authority_termination_commits(
                bindings, revision, {binding.project_id: decision}
            ),
            lambda value: binding.project_id in value,
        )[binding.project_id]
        assert committed["outcome"] == "committed"
        task = load_task(cfg, task.task_id)
        assert task.claim_control["active_claim"]["termination_decision_id"] == decision["decision_id"]
        semantic = {
            **identity,
            "mode": "active",
            "phase": "cancelled",
            "reason": "terminated_by_agent",
            "exit_code": -15,
            "termination_result": "terminated",
        }
        transition = {
            **semantic,
            "source_revisions": {
                "task": task.meta["revision"],
                "attempt_digest": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
            "transition_digest": authority_terminal_transition_digest(machine_name=binding.machine_name, **semantic),
        }
        published = _until(
            lambda: controller.advance_authority_terminal_publications(
                bindings, revision, {binding.project_id: transition}
            ),
            lambda value: binding.project_id in value,
        )[binding.project_id]
        assert published["outcome"] == "committed"
        assert published["lifecycle_event"]["phase"] == "cancelled"
        assert load_task(cfg, task.task_id).state["projection"] == "cancelled"
        assert not executor.has_unfinished_work()
    finally:
        executor.shutdown()


def test_two_overdue_workers_preserve_healthy_terminal_service(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    bindings = []
    for name in ("blocked-a", "blocked-b", "healthy"):
        cfg = init_shared_root(tmp_path / name / ".qexp", "gpu-1")
        bindings.append(runtime.add_binding(cfg.shared_root, cfg.machine_name))
    revision, bindings = runtime.load_registry()
    healthy = next(binding for binding in bindings if binding.shared_root.parent.name == "healthy")
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    try:
        _until(lambda: controller.advance_binding_validation(bindings, revision), lambda value: len(value) == 3)
        for binding in bindings:
            if binding == healthy:
                continue
            identity_path = binding.shared_root / "project" / "identity.json"
            identity_path.unlink()
            os.mkfifo(identity_path)
            request = executor.prepare_validate_binding(binding, revision)
            executor.start(request.request_id)
        _until(executor.poll, lambda value: value["overdue_worker_count"] == 2)
        started = time.monotonic()
        results = _until(
            lambda: controller.advance_authority_terminal_observations(
                [healthy],
                revision,
                {
                    healthy.project_id: {
                        "task_id": "absent-task",
                        "attempt_id": "absent-task-attempt-1",
                        "attempt_number": 1,
                        "fencing_token": 1,
                        "reservation_id": None,
                        "mode": "active",
                        "process_identity": {
                            "wrapper_pid": 101,
                            "wrapper_start_time_ticks": 202,
                            "process_group_id": 303,
                            "process_group_start_time_ticks": 404,
                        },
                    }
                },
            ),
            lambda value: healthy.project_id in value,
        )
        assert time.monotonic() - started < 15
        assert results[healthy.project_id]["outcome"] == "stale"
        assert results[healthy.project_id]["reason"] == "task_missing"
        assert executor.poll()["overdue_worker_count"] == 2
    finally:
        executor.shutdown()
