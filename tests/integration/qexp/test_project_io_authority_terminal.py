from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.agent import project_io_worker as project_io_worker_module
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.project_io_controller import ProjectIOController
from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor
from qqtools.plugins.qexp.agent.project_io_protocol import ProjectIOProcess, authority_terminal_transition_digest
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.commands.task import retry
from qqtools.plugins.qexp.runtime import claims as claims_module
from qqtools.plugins.qexp.runtime.observation import projection as observation_projection
from qqtools.plugins.qexp.runtime.paths import attempt_path, local_paths
from qqtools.plugins.qexp.runtime.ready.index import write_ready_marker
from qqtools.plugins.qexp.runtime.ready.routes import reference_for_generation
from qqtools.plugins.qexp.runtime.records import AttemptRecord
from qqtools.plugins.qexp.runtime.resources.reservations import reservation_snapshot
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task, save_task
from qqtools.plugins.qexp.scheduler import (
    claim_task,
    expire_claim,
    fail_attempt,
    observe_project_io_terminal_state,
    publish_project_io_terminal_transition,
)

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


_PROCESS_IDENTITY = {
    "wrapper_pid": 101,
    "wrapper_start_time_ticks": 202,
    "process_group_id": 303,
    "process_group_start_time_ticks": 404,
}


@dataclass(frozen=True)
class _RunningAttempt:
    runtime: MachineRuntime
    cfg: object
    binding: object
    registry_revision: int
    task_id: str
    attempt_id: str
    attempt_number: int
    fencing_token: int
    reservation_id: str | None
    attempt_file: Path

    @property
    def source_revisions(self) -> dict[str, object]:
        return {
            "task": load_task(self.cfg, self.task_id).meta["revision"],
            "attempt_digest": hashlib.sha256(self.attempt_file.read_bytes()).hexdigest(),
        }


def _running_attempt(tmp_path: Path) -> _RunningAttempt:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    revision, _bindings = runtime.load_registry()
    task = submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    path = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    stored_attempt = AttemptRecord.from_dict(read_json(path))
    stored_attempt.phase = "running"
    stored_attempt.process.update(_PROCESS_IDENTITY)
    atomic_replace(path, stored_attempt.to_dict())
    stored_task = load_task(cfg, task.task_id)
    stored_task.state.update({"projection": "running", "reason": "running"})
    stored_task.claim_control["active_claim"]["launch_state"] = "running"
    stored_task.meta["revision"] += 1
    save_task(cfg, stored_task)
    return _RunningAttempt(
        runtime,
        cfg,
        binding,
        revision,
        task.task_id,
        attempt.attempt_id,
        attempt.attempt_number,
        attempt.current_fencing_token,
        attempt.reservation_id,
        path,
    )


def _wait_for_result(executor: ProjectIOExecutor, request_id: str, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        executor.poll()
        result = executor.load_result(request_id)
        if result is not None:
            return result
        time.sleep(0.02)
    raise AssertionError(f"executor request {request_id} did not publish a result")


def _consume(executor: ProjectIOExecutor, request, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = executor.consume(request.request_id, request)
        if result is not None:
            return result
        time.sleep(0.02)
    raise AssertionError(f"executor request {request.request_id} was not consumed")


def _attempt_parameters(running: _RunningAttempt) -> dict[str, object]:
    return {
        "task_id": running.task_id,
        "attempt_id": running.attempt_id,
        "attempt_number": running.attempt_number,
        "fencing_token": running.fencing_token,
        "reservation_id": running.reservation_id,
        "process_identity": dict(_PROCESS_IDENTITY),
    }


def _terminal_publish_parameters(running: _RunningAttempt) -> dict[str, object]:
    semantic = {
        **_attempt_parameters(running),
        "machine_name": running.binding.machine_name,
        "mode": "active",
        "phase": "succeeded",
        "reason": "completed",
        "exit_code": 0,
        "termination_result": None,
    }
    return {
        "request_id": uuid.uuid4().hex,
        **semantic,
        "expected_task_revision": running.source_revisions["task"],
        "expected_attempt_digest": running.source_revisions["attempt_digest"],
        "transition_digest": authority_terminal_transition_digest(**semantic),
    }


@pytest.mark.parametrize("cancel_requested", [False, True])
def test_authority_terminal_observe_is_read_only_for_disabled_existing_attempt(
    tmp_path: Path, cancel_requested: bool
) -> None:
    running = _running_attempt(tmp_path)
    if cancel_requested:
        task = load_task(running.cfg, running.task_id)
        task.control["terminate_running"] = True
        task.meta["revision"] += 1
        save_task(running.cfg, task)
    running.runtime.set_enabled(running.binding.project_id, False)
    revision, bindings = running.runtime.load_registry()
    binding = next(item for item in bindings if item.project_id == running.binding.project_id)
    task_before = (running.cfg.shared_root / "tasks" / f"{running.task_id}.json").read_bytes()
    attempt_before = running.attempt_file.read_bytes()
    executor = ProjectIOExecutor(running.runtime)
    executor.begin_epoch()

    request = executor.prepare_authority_terminal_observe(
        binding,
        revision,
        **_attempt_parameters(running),
        mode="active",
    )
    executor.start(request.request_id)
    result = _wait_for_result(executor, request.request_id)

    assert result.status == "completed"
    assert result.evidence["outcome"] == "current"
    assert result.evidence["task_phase"] == "running"
    assert result.evidence["attempt_phase"] == "running"
    assert result.evidence["cancel_requested"] is cancel_requested
    assert result.evidence["source_revisions"] == running.source_revisions
    assert result.evidence["authority_granted"] is False
    assert result.evidence["local_effects"] == ()
    assert (running.cfg.shared_root / "tasks" / f"{running.task_id}.json").read_bytes() == task_before
    assert running.attempt_file.read_bytes() == attempt_before
    executor.shutdown()


@pytest.mark.parametrize("has_successor", [False, True])
@pytest.mark.parametrize("is_enabled", [False, True])
def test_terminal_observation_retains_settled_attempt_after_retry(
    tmp_path: Path, has_successor: bool, is_enabled: bool
) -> None:
    running = _running_attempt(tmp_path)
    assert fail_attempt(running.cfg, running.task_id, running.attempt_id, running.fencing_token, "nonzero_exit")
    retry(running.cfg, running.task_id)
    successor_file = None
    if has_successor:
        successor = claim_task(running.cfg, running.task_id, [0])
        assert successor is not None
        assert successor.attempt_number == running.attempt_number + 1
        successor_file = attempt_path(running.cfg.shared_root, running.task_id, successor.attempt_number)
    if not is_enabled:
        running.runtime.set_enabled(running.binding.project_id, False)
    revision, bindings = running.runtime.load_registry()
    binding = next(item for item in bindings if item.project_id == running.binding.project_id)
    task_file = running.cfg.shared_root / "tasks" / f"{running.task_id}.json"
    watched = [task_file, running.attempt_file]
    if successor_file is not None:
        watched.append(successor_file)
    before = {path: path.read_bytes() for path in watched}
    capacity_before = reservation_snapshot(running.cfg.runtime_root)
    executor = ProjectIOExecutor(running.runtime)
    executor.begin_epoch()
    try:
        request = executor.prepare_authority_terminal_observe(
            binding, revision, **_attempt_parameters(running), mode="active"
        )
        executor.start(request.request_id)
        result = _wait_for_result(executor, request.request_id)
        assert result.status == "completed"
        assert result.evidence["outcome"] == "settled_terminal"
        assert result.evidence["task_phase"] == ("running" if has_successor else "queued")
        assert result.evidence["attempt_phase"] == "failed"
        assert result.evidence["attempt_id"] == running.attempt_id
        assert result.evidence["source_revisions"] == running.source_revisions
        assert result.evidence["authority_granted"] is False
        assert result.evidence["local_effects"] == ()
        assert _consume(executor, request) == result
        assert {path: path.read_bytes() for path in watched} == before
        assert reservation_snapshot(running.cfg.runtime_root) == capacity_before
    finally:
        executor.shutdown()


@pytest.mark.parametrize(
    "conflict", ["attempt_first", "unfinished_retry", "next_number", "token", "reservation", "process"]
)
def test_settled_terminal_observation_rejects_partial_or_conflicting_evidence(tmp_path: Path, conflict: str) -> None:
    running = _running_attempt(tmp_path)
    if conflict == "attempt_first":
        attempt = AttemptRecord.from_dict(read_json(running.attempt_file))
        attempt.phase = "failed"
        atomic_replace(running.attempt_file, attempt.to_dict())
    else:
        assert fail_attempt(running.cfg, running.task_id, running.attempt_id, running.fencing_token, "nonzero_exit")
        retry(running.cfg, running.task_id)
    task = load_task(running.cfg, running.task_id)
    parameters = _attempt_parameters(running)
    if conflict == "unfinished_retry":
        task.attempt_control["current_attempt_id"] = running.attempt_id
        save_task(running.cfg, task)
    elif conflict == "next_number":
        task.attempt_control["next_attempt_number"] = running.attempt_number
        save_task(running.cfg, task)
    elif conflict == "token":
        parameters["fencing_token"] += 1
    elif conflict == "reservation":
        parameters["reservation_id"] = "other-reservation"
    elif conflict == "process":
        parameters["process_identity"]["wrapper_pid"] += 1
    before = running.attempt_file.read_bytes(), load_task(running.cfg, running.task_id).to_dict()

    result = observe_project_io_terminal_state(
        running.cfg, machine_name=running.binding.machine_name, **parameters, mode="active"
    )

    assert result["outcome"] == "stale"
    assert result["authority_granted"] is False
    assert result["local_effects"] == []
    assert (running.attempt_file.read_bytes(), load_task(running.cfg, running.task_id).to_dict()) == before


@pytest.mark.parametrize("locator_case", ["canonical", "opaque", "wrong_task", "conflicting_number"])
def test_terminal_settlement_helper_reads_only_shared_truth(tmp_path: Path, monkeypatch, locator_case: str) -> None:
    from qqtools.plugins.qexp.runtime.terminal_evidence import load_settled_terminal_attempt

    running = _running_attempt(tmp_path)
    assert fail_attempt(running.cfg, running.task_id, running.attempt_id, running.fencing_token, "nonzero_exit")
    retry(running.cfg, running.task_id)
    successor = claim_task(running.cfg, running.task_id, [0])
    assert successor is not None
    attempt_id = running.attempt_id
    locator = None
    number = 2 if locator_case == "conflicting_number" else None
    if locator_case in {"opaque", "wrong_task"}:
        attempt = AttemptRecord.from_dict(read_json(running.attempt_file))
        attempt_id = attempt.attempt_id = "imported-attempt"
        atomic_replace(running.attempt_file, attempt.to_dict())
        locator = {"task_id": "other-task" if locator_case == "wrong_task" else running.task_id, "attempt_number": 1}
    before = running.attempt_file.read_bytes(), load_task(running.cfg, running.task_id).to_dict()
    real_open, real_stat, real_resolve = Path.open, Path.stat, Path.resolve

    def reject_local_io(operation):
        def checked(path, *args, **kwargs):
            assert not path.is_relative_to(running.cfg.runtime_root)
            assert not path.is_relative_to(running.runtime.root)
            return operation(path, *args, **kwargs)

        return checked

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", reject_local_io(real_open))
        patch.setattr(Path, "stat", reject_local_io(real_stat))
        patch.setattr(Path, "resolve", reject_local_io(real_resolve))
        result = load_settled_terminal_attempt(
            running.cfg, running.task_id, attempt_id, attempt_number=number, recovery_locator=locator
        )
    if locator_case in {"wrong_task", "conflicting_number"}:
        assert result is None
    else:
        assert result is not None
        assert result.attempt_id == attempt_id
        assert result.attempt_number == 1
        assert result.phase == "failed"
    assert (running.attempt_file.read_bytes(), load_task(running.cfg, running.task_id).to_dict()) == before


def test_authority_termination_commit_is_exact_idempotent_and_conflict_safe(tmp_path: Path) -> None:
    running = _running_attempt(tmp_path)
    executor = ProjectIOExecutor(running.runtime)
    executor.begin_epoch()
    decision_id = uuid.uuid4().hex
    attempt_before = running.attempt_file.read_bytes()
    common = {
        **_attempt_parameters(running),
        "decision_token": running.fencing_token,
        "authority_outcome": "lease_expired",
        "reason": "authority_lost",
        "source_revisions": running.source_revisions,
    }

    request = executor.prepare_authority_termination_commit(
        running.binding,
        running.registry_revision,
        decision_id=decision_id,
        **common,
    )
    executor.start(request.request_id)
    result = _wait_for_result(executor, request.request_id)

    assert result.status == "completed"
    assert result.evidence["outcome"] == "committed"
    assert result.evidence["shared_commitment"] == "committed"
    committed_task = load_task(running.cfg, running.task_id)
    claim = committed_task.claim_control["active_claim"]
    assert claim["termination_decision_id"] == decision_id
    assert claim["termination_decision_token"] == running.fencing_token
    committed_revision = committed_task.meta["revision"]
    assert running.attempt_file.read_bytes() == attempt_before
    assert _consume(executor, request) == result

    replay = executor.prepare_authority_termination_commit(
        running.binding,
        running.registry_revision,
        decision_id=decision_id,
        **common,
    )
    executor.start(replay.request_id)
    replay_result = _wait_for_result(executor, replay.request_id)
    assert replay_result.status == "completed"
    assert replay_result.evidence["outcome"] == "already_committed"
    assert load_task(running.cfg, running.task_id).meta["revision"] == committed_revision
    assert _consume(executor, replay) == replay_result

    mismatches = (
        {"authority_outcome": "process_identity_lost"},
        {"reason": "different_reason"},
        {
            "source_revisions": {
                **common["source_revisions"],
                "task": common["source_revisions"]["task"] + 1,
            }
        },
    )
    for mismatch in mismatches:
        conflicting_common = {**common, **mismatch}
        semantic_conflict = executor.prepare_authority_termination_commit(
            running.binding,
            running.registry_revision,
            decision_id=decision_id,
            **conflicting_common,
        )
        executor.start(semantic_conflict.request_id)
        semantic_result = _wait_for_result(executor, semantic_conflict.request_id)
        assert semantic_result.status == "completed"
        assert semantic_result.evidence["outcome"] == "stale"
        assert semantic_result.evidence["reason"] == "decision_conflict"
        assert _consume(executor, semantic_conflict) == semantic_result

    conflict = executor.prepare_authority_termination_commit(
        running.binding,
        running.registry_revision,
        decision_id=uuid.uuid4().hex,
        **common,
    )
    executor.start(conflict.request_id)
    conflict_result = _wait_for_result(executor, conflict.request_id)
    assert conflict_result.status == "completed"
    assert conflict_result.evidence["outcome"] == "stale"
    assert conflict_result.evidence["reason"] == "decision_conflict"
    assert load_task(running.cfg, running.task_id).meta["revision"] == committed_revision
    executor.shutdown()


def test_authority_terminal_publish_commits_exact_transition_without_local_effects(tmp_path: Path) -> None:
    running = _running_attempt(tmp_path)
    manifest = local_paths(running.runtime.project_paths(running.binding.project_id)["root"])["processes"] / (
        f"{running.attempt_id}.json"
    )
    atomic_replace(manifest, {"process": {**_attempt_parameters(running), "custom_marker": "unchanged"}})
    manifest_before = manifest.read_bytes()
    executor = ProjectIOExecutor(running.runtime)
    executor.begin_epoch()
    observe = executor.prepare_authority_terminal_observe(
        running.binding,
        running.registry_revision,
        **_attempt_parameters(running),
        mode="active",
    )
    executor.start(observe.request_id)
    observed = _wait_for_result(executor, observe.request_id)
    assert observed.evidence["outcome"] == "current"
    assert _consume(executor, observe) == observed

    publish = executor.prepare_authority_terminal_publish(
        running.binding,
        running.registry_revision,
        **_attempt_parameters(running),
        mode="active",
        phase="succeeded",
        reason="completed",
        exit_code=0,
        termination_result=None,
        source_revisions=observed.evidence["source_revisions"],
        transition_digest=authority_terminal_transition_digest(
            mode="active",
            task_id=running.task_id,
            attempt_id=running.attempt_id,
            attempt_number=running.attempt_number,
            fencing_token=running.fencing_token,
            machine_name=running.binding.machine_name,
            reservation_id=running.reservation_id,
            process_identity=_PROCESS_IDENTITY,
            phase="succeeded",
            reason="completed",
            exit_code=0,
            termination_result=None,
        ),
    )
    executor.start(publish.request_id)
    result = _wait_for_result(executor, publish.request_id)

    assert result.status == "completed"
    assert result.evidence["outcome"] == "committed"
    assert result.evidence["authority_granted"] is False
    assert result.evidence["local_effects"] == ()
    assert result.evidence["lifecycle_event"]["phase"] == "succeeded"
    assert load_task(running.cfg, running.task_id).state["projection"] == "succeeded"
    terminal_attempt = AttemptRecord.from_dict(read_json(running.attempt_file))
    assert terminal_attempt.phase == "succeeded"
    assert terminal_attempt.result["exit_code"] == 0
    assert terminal_attempt.result["reason"] == "completed"
    assert manifest.read_bytes() == manifest_before
    assert _consume(executor, publish) == result

    fresh = executor.prepare_authority_terminal_publish(
        running.binding,
        running.registry_revision,
        **_attempt_parameters(running),
        mode="active",
        phase="succeeded",
        reason="completed",
        exit_code=0,
        termination_result=None,
        source_revisions={
            "task": observed.evidence["source_revisions"]["task"] + 1,
            "attempt_digest": observed.evidence["source_revisions"]["attempt_digest"],
        },
        transition_digest=publish.parameters["transition_digest"],
    )
    executor.start(fresh.request_id)
    fresh_result = _wait_for_result(executor, fresh.request_id)
    assert fresh_result.status == "completed"
    assert fresh_result.evidence["outcome"] == "stale"
    assert fresh_result.evidence["reason"] == "terminal_conflict"
    assert _consume(executor, fresh) == fresh_result
    executor.shutdown()


@pytest.mark.parametrize("name", ["长" * 70_000, "\x01" * 70_000, "\x00" * 70_000], ids=["unicode", "escaped", "nul"])
def test_authority_terminal_publish_always_returns_bounded_lifecycle_event(tmp_path: Path, name: str) -> None:
    running = _running_attempt(tmp_path)
    task = load_task(running.cfg, running.task_id)
    task.name = name
    save_task(running.cfg, task)
    parameters = _terminal_publish_parameters(running)

    result = publish_project_io_terminal_transition(
        running.cfg,
        **parameters,
        mutation_fence=lambda: None,
    )

    assert result["outcome"] == "committed"
    event = result["lifecycle_event"]
    assert event is not None
    assert event["task_name"]
    assert len(json.dumps(event["task_name"], ensure_ascii=False).encode("utf-8")) <= 16_384
    assert load_task(running.cfg, running.task_id).name == name


@pytest.mark.parametrize("fail_at", [2, 5], ids=["attempt_written", "task_written"])
def test_authority_terminal_publish_recovers_each_truth_write_boundary(tmp_path: Path, fail_at: int) -> None:
    running = _running_attempt(tmp_path)
    parameters = _terminal_publish_parameters(running)
    task = load_task(running.cfg, running.task_id)
    generation = task.ready_generation
    # Model a ready projection left behind by interrupted claim-time retirement.
    task.ready_generation -= 1
    write_ready_marker(
        running.cfg,
        task,
        generation=generation,
        source_transition="submit",
        source_revision=task.meta["revision"],
        target_revision=task.meta["revision"],
    )
    reference = reference_for_generation(running.cfg, running.task_id, generation)
    assert reference is not None

    def interrupt_after_boundary() -> None:
        attempt_written = AttemptRecord.from_dict(read_json(running.attempt_file)).phase == "succeeded"
        task_written = load_task(running.cfg, running.task_id).state["projection"] == "succeeded"
        if (fail_at == 2 and attempt_written) or (fail_at == 5 and task_written):
            raise RuntimeError("simulated terminal worker interruption")

    with pytest.raises(RuntimeError, match="simulated terminal worker interruption"):
        publish_project_io_terminal_transition(
            running.cfg,
            **parameters,
            mutation_fence=interrupt_after_boundary,
        )

    partial_attempt = AttemptRecord.from_dict(read_json(running.attempt_file))
    assert partial_attempt.phase == "succeeded"
    if fail_at == 2:
        assert load_task(running.cfg, running.task_id).state["projection"] == "running"
    else:
        assert load_task(running.cfg, running.task_id).state["projection"] == "succeeded"
    assert reference_for_generation(running.cfg, running.task_id, generation) == reference

    replay = publish_project_io_terminal_transition(
        running.cfg,
        **parameters,
        mutation_fence=lambda: None,
    )

    assert replay["outcome"] == ("committed" if fail_at == 2 else "already_committed")
    assert replay["lifecycle_event"] is not None
    assert replay["lifecycle_event"]["phase"] == "succeeded"
    assert load_task(running.cfg, running.task_id).state["projection"] == "succeeded"
    assert AttemptRecord.from_dict(read_json(running.attempt_file)).phase == "succeeded"
    assert reference_for_generation(running.cfg, running.task_id, generation) is None


def test_authority_terminal_publish_cross_epoch_worker_replay_survives_transient_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    running = _running_attempt(tmp_path)
    executor = ProjectIOExecutor(running.runtime)
    executor.begin_epoch()
    source_revisions = running.source_revisions
    request = executor.prepare_authority_terminal_publish(
        running.binding,
        running.registry_revision,
        **_attempt_parameters(running),
        mode="active",
        phase="succeeded",
        reason="completed",
        exit_code=0,
        termination_result=None,
        source_revisions=source_revisions,
        transition_digest=authority_terminal_transition_digest(
            mode="active",
            machine_name=running.binding.machine_name,
            **_attempt_parameters(running),
            phase="succeeded",
            reason="completed",
            exit_code=0,
            termination_result=None,
        ),
    )

    def interrupt_after_attempt_write() -> None:
        if AttemptRecord.from_dict(read_json(running.attempt_file)).phase == "succeeded":
            raise RuntimeError("simulated old worker loss")

    with pytest.raises(RuntimeError, match="simulated old worker loss"):
        publish_project_io_terminal_transition(
            running.cfg,
            request_id=request.request_id,
            machine_name=running.binding.machine_name,
            **_attempt_parameters(running),
            mode="active",
            phase="succeeded",
            reason="completed",
            exit_code=0,
            termination_result=None,
            expected_task_revision=source_revisions["task"],
            expected_attempt_digest=source_revisions["attempt_digest"],
            transition_digest=request.parameters["transition_digest"],
            mutation_fence=interrupt_after_attempt_write,
        )

    # A real worker publishes its process identity before entering shared I/O.
    # Retain that evidence with a positively absent PID to model its crash.
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
    restarted = ProjectIOExecutor(running.runtime)
    restarted.begin_epoch()
    assert request in restarted.unresolved_requests()
    pid = os.getpid()
    ticks = project_io_worker_module._process_start_time_ticks(pid)
    assert ticks is not None

    def install_live_replay_process() -> None:
        restarted._write_record(
            restarted._record_path("processes", request.request_id),
            ProjectIOProcess(
                request=request,
                pid=pid,
                start_time_ticks=ticks,
                started_at=datetime.now(timezone.utc).isoformat(),
                state="running",
            ).to_dict(),
            "project_io_process",
        )

    install_live_replay_process()
    with monkeypatch.context() as transient:
        transient.setattr(
            project_io_worker_module,
            "_authority_terminal_publish",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("transient replay read failure")),
        )
        assert (
            project_io_worker_module._run(
                running.runtime.root,
                request.request_id,
                reconcile_authority_terminal_publish=True,
            )
            == 0
        )
    failed = restarted.load_result(request.request_id)
    assert failed is not None
    assert failed.status == "retryable_error"
    assert request in restarted.unresolved_requests()

    install_live_replay_process()
    assert (
        project_io_worker_module._run(
            running.runtime.root,
            request.request_id,
            reconcile_authority_terminal_publish=True,
        )
        == 0
    )
    completed = restarted.load_result(request.request_id)
    assert completed is not None
    assert completed.status == "completed"
    assert completed.evidence["outcome"] == "committed"
    assert completed.evidence["lifecycle_event"]["phase"] == "succeeded"


def test_binding_validation_drains_cross_epoch_terminal_replay(tmp_path: Path) -> None:
    running = _running_attempt(tmp_path)
    executor = ProjectIOExecutor(running.runtime)
    executor.begin_epoch()
    source_revisions = running.source_revisions
    parameters = _terminal_publish_parameters(running)
    request = executor.prepare_authority_terminal_publish(
        running.binding,
        running.registry_revision,
        **_attempt_parameters(running),
        mode="active",
        phase="succeeded",
        reason="completed",
        exit_code=0,
        termination_result=None,
        source_revisions=source_revisions,
        transition_digest=parameters["transition_digest"],
    )
    parameters["request_id"] = request.request_id

    def interrupt_after_attempt_write() -> None:
        if AttemptRecord.from_dict(read_json(running.attempt_file)).phase == "succeeded":
            raise RuntimeError("simulated old worker loss")

    with pytest.raises(RuntimeError, match="simulated old worker loss"):
        publish_project_io_terminal_transition(
            running.cfg,
            **parameters,
            mutation_fence=interrupt_after_attempt_write,
        )
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

    restarted = ProjectIOExecutor(running.runtime)
    restarted.begin_epoch()
    controller = ProjectIOController(running.runtime, restarted)
    deadline = time.monotonic() + 5.0
    validated = {}
    try:
        while time.monotonic() < deadline:
            validated = controller.advance_binding_validation(
                [running.binding],
                running.registry_revision,
            )
            if (
                running.binding.project_id in validated
                and load_task(running.cfg, running.task_id).state["projection"] == "succeeded"
            ):
                break
            time.sleep(0.02)
        assert running.binding.project_id in validated
        assert load_task(running.cfg, running.task_id).state["projection"] == "succeeded"
        assert restarted.unresolved_requests() == ()
    finally:
        restarted.shutdown()


def test_detached_terminal_publish_cross_epoch_replays_attempt_first_write(tmp_path: Path) -> None:
    running = _running_attempt(tmp_path)
    task = load_task(running.cfg, running.task_id)
    task.claim_control["active_claim"].update(
        lease_expires_at="2000-01-01T00:00:00Z",
        clock_error_bound_seconds=0.0,
    )
    task.meta["revision"] += 1
    save_task(running.cfg, task)
    assert expire_claim(
        running.cfg,
        running.task_id,
        running.attempt_id,
        running.fencing_token,
    )
    orphan = load_task(running.cfg, running.task_id)
    assert orphan.state["projection"] == "blocked"
    assert orphan.attempt_control["current_attempt_id"] is None
    assert AttemptRecord.from_dict(read_json(running.attempt_file)).phase == "orphaned"

    executor = ProjectIOExecutor(running.runtime)
    executor.begin_epoch()
    source_revisions = running.source_revisions
    semantic = {
        **_attempt_parameters(running),
        "mode": "detached_orphan",
        "phase": "succeeded",
        "reason": "completed",
        "exit_code": 0,
        "termination_result": None,
    }
    request = executor.prepare_authority_terminal_publish(
        running.binding,
        running.registry_revision,
        **semantic,
        source_revisions=source_revisions,
        transition_digest=authority_terminal_transition_digest(
            machine_name=running.binding.machine_name,
            **semantic,
        ),
    )

    def interrupt_after_attempt_write() -> None:
        if AttemptRecord.from_dict(read_json(running.attempt_file)).phase == "succeeded":
            raise RuntimeError("simulated detached worker loss")

    with pytest.raises(RuntimeError, match="simulated detached worker loss"):
        publish_project_io_terminal_transition(
            running.cfg,
            request_id=request.request_id,
            machine_name=running.binding.machine_name,
            **semantic,
            expected_task_revision=source_revisions["task"],
            expected_attempt_digest=source_revisions["attempt_digest"],
            transition_digest=request.parameters["transition_digest"],
            mutation_fence=interrupt_after_attempt_write,
        )
    assert load_task(running.cfg, running.task_id).state["projection"] == "blocked"
    assert AttemptRecord.from_dict(read_json(running.attempt_file)).phase == "succeeded"

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
    restarted = ProjectIOExecutor(running.runtime)
    restarted.begin_epoch()
    assert request in restarted.unresolved_requests()
    pid = os.getpid()
    ticks = project_io_worker_module._process_start_time_ticks(pid)
    assert ticks is not None
    restarted._write_record(
        restarted._record_path("processes", request.request_id),
        ProjectIOProcess(
            request=request,
            pid=pid,
            start_time_ticks=ticks,
            started_at=datetime.now(timezone.utc).isoformat(),
            state="running",
        ).to_dict(),
        "project_io_process",
    )
    assert (
        project_io_worker_module._run(
            running.runtime.root,
            request.request_id,
            reconcile_authority_terminal_publish=True,
        )
        == 0
    )
    completed = restarted.load_result(request.request_id)
    assert completed is not None
    assert completed.status == "completed"
    assert completed.evidence["outcome"] == "committed"
    assert completed.evidence["mode"] == "detached_orphan"
    assert load_task(running.cfg, running.task_id).state["projection"] == "succeeded"


@pytest.mark.parametrize("has_committed_before_hook,is_replaced", [(False, False), (True, False), (True, True)])
def test_terminal_worker_delivers_once_including_replay_after_truth_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, has_committed_before_hook: bool, is_replaced: bool
) -> None:
    from qqtools.plugins.qexp import notifications
    from qqtools.plugins.qexp.notification_config import update_notifications, write_shared_feishu_webhook

    running = _running_attempt(tmp_path)
    update_notifications(
        running.cfg,
        lambda current: {
            **current,
            "enabled": True,
            "providers": {
                "feishu": {
                    "enabled": True,
                    "credential_source": "shared_file",
                    "webhook_env": "UNUSED_WEBHOOK_ENV",
                    "secret_env": None,
                    "timeout_seconds": 5,
                }
            },
        },
    )
    webhook = "https://open.feishu.cn/open-apis/bot/v2/hook/worker-private-test"
    write_shared_feishu_webhook(running.cfg, webhook)
    calls = []

    class Notifier:
        def send(self, event, *, webhook, secret, timeout_seconds):
            assert load_task(running.cfg, running.task_id).state["projection"] == "succeeded"
            calls.append((event.task_id, event.phase, webhook))
            return {"http_status": 200, "business_code": "0"}

    monkeypatch.setitem(notifications.REGISTRY, "feishu", Notifier())
    # A terminal worker must use captured scope, not construct a second local
    # authority owner or resolve shared binding configuration under local locks.
    monkeypatch.setattr(
        notifications,
        "MachineRuntime",
        lambda *_args, **_kwargs: pytest.fail("worker acquired MachineRuntime authority"),
    )
    executor = ProjectIOExecutor(running.runtime)
    executor.begin_epoch()
    parameters = _terminal_publish_parameters(running)
    request = executor.prepare_authority_terminal_publish(
        running.binding,
        running.registry_revision,
        **_attempt_parameters(running),
        mode="active",
        phase="succeeded",
        reason="completed",
        exit_code=0,
        termination_result=None,
        source_revisions=running.source_revisions,
        transition_digest=parameters["transition_digest"],
    )
    if has_committed_before_hook:
        # Model process loss after shared truth commits but before hook delivery.
        parameters["request_id"] = request.request_id
        result = publish_project_io_terminal_transition(running.cfg, **parameters, mutation_fence=lambda: None)
        assert result["outcome"] == "committed"
        assert calls == []

    replay_options = {}
    if has_committed_before_hook:
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
        restarted = ProjectIOExecutor(running.runtime)
        restarted.begin_epoch()
        replay_options = {
            "replay_only": True,
            "active_executor_epoch": project_io_worker_module._read_epoch(executor.paths).executor_epoch,
        }
    if is_replaced:
        registry_path = running.runtime.paths["registry"]
        registry = read_json(registry_path)
        registry["registry"]["bindings"][0]["registration_generation"] = uuid.uuid4().hex
        registry["registry"]["revision"] += 1
        atomic_replace(registry_path, registry)
        pid = os.getpid()
        ticks = project_io_worker_module._process_start_time_ticks(pid)
        assert ticks is not None
        process = ProjectIOProcess(
            request=request,
            pid=pid,
            start_time_ticks=ticks,
            started_at=datetime.now(timezone.utc).isoformat(),
            state="running",
        )
        restarted._write_record(
            restarted._record_path("processes", request.request_id), process.to_dict(), "project_io_process"
        )
        assert (
            project_io_worker_module._run(
                running.runtime.root, request.request_id, reconcile_authority_terminal_publish=True
            )
            == 0
        )
        result = restarted.load_result(request.request_id)
        assert result is not None
        assert result.status == "completed"
        assert result.evidence["outcome"] == "already_committed"
        # The direct worker invocation runs inside this test process; replace
        # its process receipt with positively absent crash evidence to model exit.
        from dataclasses import replace

        restarted._write_record(
            restarted._record_path("processes", request.request_id),
            replace(process, pid=2_000_000_000, start_time_ticks=1).to_dict(),
            "project_io_process",
        )
        assert restarted.resolve_stale_authority_terminal_publish(request.request_id, request)
        assert request not in restarted.unresolved_requests()
        assert calls == []
        from qqtools.plugins.qexp.runtime.paths import shared_paths

        assert not list(shared_paths(running.cfg.shared_root)["notifications"].glob("*.json"))
        assert load_task(running.cfg, running.task_id).state["projection"] == "succeeded"
        return

    for expected in (
        ["already_committed", "already_committed"] if has_committed_before_hook else ["committed", "already_committed"]
    ):
        with running.runtime.registry_guard(), running.runtime.binding_commit_guard(running.binding):
            result = project_io_worker_module._authority_terminal_publish(
                request, running.runtime.root, executor.paths, [False], **replay_options
            )
        assert result["outcome"] == expected
    assert calls == [(running.task_id, "succeeded", webhook)]
    assert webhook not in json.dumps(request.to_dict())
    assert webhook not in json.dumps(result)


@pytest.mark.parametrize("boundary", ["_prepare_mutation", "sync_task"], ids=["before_task", "after_task"])
def test_terminal_publication_epoch_fence_stops_nested_projection_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    running = _running_attempt(tmp_path)
    parameters = _terminal_publish_parameters(running)
    revoked = False
    at_revocation = {}
    original = getattr(observation_projection, boundary)

    def snapshot():
        return {
            str(path.relative_to(running.cfg.shared_root)): path.read_bytes()
            for path in running.cfg.shared_root.rglob("*")
            if path.is_file() and path.suffix in {".json", ".sqlite3"}
        }

    def revoke(*args, **kwargs):
        nonlocal revoked, at_revocation
        revoked = True
        at_revocation = snapshot()
        return original(*args, **kwargs)

    def fence():
        if revoked:
            raise RuntimeError("executor epoch revoked")

    with monkeypatch.context() as mutation:
        mutation.setattr(observation_projection, boundary, revoke)
        with pytest.raises(RuntimeError, match="executor epoch revoked"):
            publish_project_io_terminal_transition(running.cfg, **parameters, mutation_fence=fence)
    assert revoked
    assert snapshot() == at_revocation
    assert load_task(running.cfg, running.task_id).state["projection"] == (
        "running" if boundary == "_prepare_mutation" else "succeeded"
    )
    replay = publish_project_io_terminal_transition(running.cfg, **parameters, mutation_fence=lambda: None)
    assert replay["outcome"] in {"committed", "already_committed"}
    assert replay["lifecycle_event"]["phase"] == "succeeded"


def test_claim_archive_failure_rechecks_epoch_before_pending_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    running = _running_attempt(tmp_path)
    claim = load_task(running.cfg, running.task_id).claim_control["active_claim"]
    writes = []

    def fail_first_write(path, record):
        writes.append(path)
        raise OSError("archive unavailable")

    def fence():
        if writes:
            raise RuntimeError("executor epoch revoked")

    monkeypatch.setattr(claims_module, "create_if_absent", fail_first_write)
    with pytest.raises(RuntimeError, match="executor epoch revoked"):
        claims_module.archive_claim(running.cfg, running.task_id, claim, "completed", mutation_fence=fence)
    assert len(writes) == 1
