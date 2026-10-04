from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.agent import project_io_executor as project_io_executor_module
from qqtools.plugins.qexp.agent import project_io_worker as project_io_worker_module
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.dispatch_probe import PrimaryProbeSession
from qqtools.plugins.qexp.agent.lifecycle import get_machine_agent_status
from qqtools.plugins.qexp.agent.primary_probe_transport import encode_probe_session
from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor, ProjectIOProtocolError
from qqtools.plugins.qexp.agent.project_io_protocol import (
    PROJECT_IO_MAX_RECORD_BYTES,
    PROJECT_IO_MAX_RESOLVED_BYTES,
    PROJECT_IO_MAX_RESOLVED_RECORDS,
    ProjectIOProcess,
    ProjectIOResult,
)
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.runtime import project_activation as project_activation_module
from qqtools.plugins.qexp.runtime.locks import exclusive
from qqtools.plugins.qexp.runtime.paths import local_paths, shared_paths
from qqtools.plugins.qexp.runtime.project_activation import compact_project_activation, publish_project_activation
from qqtools.plugins.qexp.runtime.project_activation_consumers import read_consumer_progress
from qqtools.plugins.qexp.runtime.ready import (
    ReadyCursor,
    compare_and_commit_ready_cursor,
    load_ready_cursor,
    repair_ready_index,
)
from qqtools.plugins.qexp.runtime.ready import traversal as ready_traversal
from qqtools.plugins.qexp.runtime.resources.cpu_lane import initialize_cpu_lane_capacity, reserve_cpu
from qqtools.plugins.qexp.runtime.resources.reservations import (
    ReservationIdentity,
    attach_executor_offer,
    classify_executor_offer,
    reserve,
)
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json


def test_fresh_worker_import_does_not_load_executor() -> None:
    source_root = Path(project_io_worker_module.__file__).parents[4]
    script = f"""
import importlib.abc
import sys

sys.path.insert(0, {str(source_root)!r})

class RejectExecutor(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "qqtools.plugins.qexp.agent.project_io_executor":
            raise RuntimeError("worker imported executor orchestration")
        return None

sys.meta_path.insert(0, RejectExecutor())
import qqtools.plugins.qexp.agent.project_io_worker
assert "qqtools.plugins.qexp.agent.project_io_executor" not in sys.modules
"""

    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=10, check=False)

    assert result.returncode == 0, result.stderr


from qqtools.plugins.qexp.runtime.tasks import load_task, save_task
from qqtools.plugins.qexp.scheduler import claim_task
from qqtools.version import __version__
from tests.helpers.qexp.worker_diagnostics import describe_project_io_workers

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _registered(tmp_path: Path, name: str, runtime: MachineRuntime):
    cfg = init_shared_root(tmp_path / name / ".qexp", "gpu-1")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    revision, _bindings = runtime.load_registry()
    return cfg, binding, revision


def _wait_for_result(executor: ProjectIOExecutor, request_id: str, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        executor.poll()
        result = executor.load_result(request_id)
        if result is not None:
            return result
        time.sleep(0.02)
    raise AssertionError(f"executor request {request_id} did not publish a result")


def _wait_for_consumed(executor: ProjectIOExecutor, request, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = executor.consume(request.request_id, request)
        if result is not None:
            return result
        time.sleep(0.02)
    raise AssertionError(
        f"executor request {request.request_id} was not consumed: {describe_project_io_workers(executor)}"
    )


def _project_snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(root)): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in root.rglob("*")
        if path.is_file()
    }


def test_group_service_probe_and_advance_run_in_fresh_workers(tmp_path: Path, monkeypatch) -> None:
    from qqtools.plugins.qexp.agent.group_service_transport import (
        group_service_advance_request,
        group_service_probe_request,
        initial_group_service_continuation,
    )
    from qqtools.plugins.qexp.runtime.group_discovery.coverage import GroupCoverage
    from qqtools.plugins.qexp.runtime.group_discovery.probe import initial_group_service_probe_state
    from qqtools.plugins.qexp.runtime.paths import submission_path
    from tests.helpers.qexp_discovery import isolated_group, source_file

    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg = isolated_group(tmp_path / "project", tail=1)
    source_file(submission_path(cfg.shared_root, "batch"), operation="batch")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    revision, _bindings = runtime.load_registry()
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    try:
        probe = executor.prepare_closed_request(
            binding,
            revision,
            group_service_probe_request(binding.machine_name, initial_group_service_probe_state()),
        )
        executor.start(probe.request_id)
        observed = _wait_for_consumed(executor, probe)
        assert observed.status == "completed"
        assert observed.evidence["state"] == "active"
        candidate = observed.evidence["candidate"]
        continuation = initial_group_service_continuation(candidate)

        # A killed worker may have committed part of the fenced Group
        # transaction before result publication. Preserve the exact intent,
        # clear only the proven-dead execution image, and replay it.
        interrupted = executor.prepare_closed_request(
            binding,
            revision,
            group_service_advance_request(
                binding.machine_name,
                candidate,
                continuation,
                observed.evidence["probe_state"],
            ),
        )
        dead_process = ProjectIOProcess(interrupted, 2_000_000_000, 1, interrupted.prepared_at, "running")
        atomic_replace(
            executor.paths["project_io_processes"] / f"{interrupted.request_id}.json",
            dead_process.to_dict(),
        )
        with pytest.raises(ValueError, match="replay"):
            executor.reset_ambiguous_replayable_request_for_retry(probe.request_id, probe)
        inspect_process_identity = project_io_executor_module.inspect_process_identity
        for process_state in ("same_live_process", "unverified"):
            monkeypatch.setattr(
                project_io_executor_module,
                "inspect_process_identity",
                lambda *_args, process_state=process_state: process_state,
            )
            assert not executor.reset_ambiguous_replayable_request_for_retry(interrupted.request_id, interrupted)
        monkeypatch.setattr(project_io_executor_module, "inspect_process_identity", inspect_process_identity)
        executor.poll()
        ambiguous = executor.load_result(interrupted.request_id)
        assert ambiguous is not None and ambiguous.status == "outcome_unknown"
        mismatched = replace(interrupted, registry_revision=interrupted.registry_revision + 1)
        assert not executor.reset_ambiguous_replayable_request_for_retry(interrupted.request_id, mismatched)
        assert executor.reset_ambiguous_replayable_request_for_retry(interrupted.request_id, interrupted)
        assert executor.start(interrupted.request_id) is not None
        replayed = _wait_for_consumed(executor, interrupted)
        assert replayed.status == "completed"
        continuation = replayed.evidence["continuation"]
        if replayed.evidence["state"] == "quiescent":
            assert GroupCoverage(cfg.shared_root, "experiment").status().is_complete
            return

        for _ in range(100):
            request = executor.prepare_closed_request(
                binding,
                revision,
                group_service_advance_request(
                    binding.machine_name,
                    candidate,
                    continuation,
                    observed.evidence["probe_state"],
                ),
            )
            executor.start(request.request_id)
            result = _wait_for_consumed(executor, request)
            assert result.status == "completed"
            continuation = result.evidence["continuation"]
            if result.evidence["state"] == "quiescent":
                break
        else:
            raise AssertionError("isolated Group discovery did not converge")
        assert GroupCoverage(cfg.shared_root, "experiment").status().is_complete
    finally:
        executor.shutdown()


def test_idle_requires_consumption_of_returning_executor_work(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision = _registered(tmp_path, "project", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    try:
        assert not executor.has_unfinished_work()
        request = executor.prepare_validate_binding(binding, revision)
        assert executor.has_unfinished_work()

        executor.start(request.request_id)
        assert executor.status_view()["envelope"] == "degraded"
        assert executor.has_unfinished_work()
        result = _wait_for_result(executor, request.request_id)
        assert result.status == "completed"
        assert executor.has_unfinished_work()

        assert _wait_for_consumed(executor, request).status == "completed"
        assert not executor.has_unfinished_work()
    finally:
        executor.shutdown()


@pytest.mark.parametrize("evidence", ["none", "requests", "processes", "results", "symlink"])
def test_release_unprepared_offer_requires_absent_transport(tmp_path: Path, evidence: str) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, _revision = _registered(tmp_path, "project", runtime)
    executor = ProjectIOExecutor(runtime)
    epoch = executor.begin_epoch()
    request_id = uuid.uuid4().hex
    record = reserve(
        runtime.root,
        "task",
        [0],
        attempt_id="attempt",
        fencing_token=1,
        project_id=binding.project_id,
        shared_root=str(binding.shared_root),
        machine_name=binding.machine_name,
        executor_epoch=epoch,
        executor_request_id=request_id,
        registration_generation=binding.registration_generation,
    )["reservation"]
    identity = ReservationIdentity.from_record(record)
    before = _project_snapshot(cfg.shared_root)
    try:
        if evidence == "symlink":
            executor._record_path("requests", request_id).symlink_to(tmp_path / "missing")
            with pytest.raises(ProjectIOProtocolError):
                executor.release_unprepared_offer(identity)
            assert classify_executor_offer(runtime.root, identity) == "matching_provisional"
        elif evidence != "none":
            # Even malformed evidence prevents treating a transport as absent.
            atomic_replace(executor._record_path(evidence, request_id), {"invalid": True})
            assert executor.release_unprepared_offer(identity) is False
            assert classify_executor_offer(runtime.root, identity) == "matching_provisional"
        else:
            executor.begin_epoch()
            assert executor.release_unprepared_offer(identity) is True
            assert executor.release_unprepared_offer(identity) is True
            assert classify_executor_offer(runtime.root, identity) == "matching_released"
        assert _project_snapshot(cfg.shared_root) == before
    finally:
        executor.shutdown()


@pytest.mark.parametrize("resource", ["gpu", "cpu"])
def test_release_unprepared_offer_replays_exact_release_crash_pair(tmp_path: Path, resource: str) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, _revision = _registered(tmp_path, "project", runtime)
    executor = ProjectIOExecutor(runtime)
    epoch = executor.begin_epoch()
    request_id = uuid.uuid4().hex
    owner = {
        "attempt_id": "attempt",
        "fencing_token": 1,
        "project_id": binding.project_id,
        "shared_root": str(binding.shared_root),
        "machine_name": binding.machine_name,
        "executor_epoch": epoch,
        "executor_request_id": request_id,
        "registration_generation": binding.registration_generation,
    }
    if resource == "gpu":
        record = reserve(runtime.root, "task", [0], **owner)["reservation"]
        released_dir = local_paths(runtime.root)["released"]
        provisional_dir = local_paths(runtime.root)["provisional"]
    else:
        initialize_cpu_lane_capacity(runtime.root, capacity=1)
        record = reserve_cpu(runtime.root, "task", 1, **owner)["reservation"]
        released_dir = local_paths(runtime.root)["cpu_released"]
        provisional_dir = local_paths(runtime.root)["cpu_provisional"]
    identity = ReservationIdentity.from_record(record)
    released = {
        "reservation": {
            **record,
            "state": "released",
            "released_at": "2026-10-02T00:00:00Z",
            "release_reason": "claim_offer_not_prepared",
        }
    }
    atomic_replace(released_dir / f"{identity.reservation_id}.json", released)
    try:
        assert classify_executor_offer(runtime.root, identity) == "conflict"
        assert executor.release_unprepared_offer(identity) is True
        assert classify_executor_offer(runtime.root, identity) == "matching_released"
        assert not (provisional_dir / f"{identity.reservation_id}.json").exists()
    finally:
        executor.shutdown()


def test_submission_control_service_resumes_shared_proofs_in_fresh_workers(tmp_path: Path) -> None:
    from qqtools.plugins.qexp.runtime import submission_control as control
    from qqtools.plugins.qexp.runtime.paths import submission_path
    from qqtools.plugins.qexp.runtime.submission_control_continuation import submission_control_continuation

    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _registered(tmp_path, "project", runtime)
    task = submit(cfg, ["true"], working_dir=tmp_path)
    source = submission_path(cfg.shared_root, task.submission_operation_id)
    value = read_json(source)
    value["submission"]["resolved_context"]["retained_payload"] = "x" * 180_000
    atomic_replace(source, value)
    control.request_repair(cfg, task.submission_operation_id)
    control.request_control_rebuild(cfg)
    before_local = _project_snapshot(runtime.project_paths(binding.project_id)["root"])
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    continuation = submission_control_continuation()
    try:
        for _ in range(80):
            request = executor.prepare_submission_control_service(binding, revision, continuation=continuation)
            executor.start(request.request_id)
            result = _wait_for_consumed(executor, request)
            assert result.status == "completed"
            continuation = result.evidence["continuation"]
            if result.evidence["quiescent"]:
                break
        else:
            raise AssertionError("isolated Submission-control repair did not converge")
        assert control.read_control_state(cfg)["state"] == "active"
        assert control.read_submission_state(cfg, task.submission_operation_id) == "committed"
        assert not control.pending_path(cfg, task.submission_operation_id).exists()
        assert _project_snapshot(runtime.project_paths(binding.project_id)["root"]) == before_local
        assert not executor.unresolved_requests()
    finally:
        executor.shutdown()


@pytest.mark.parametrize("failure", ["worker_absence", "late_epoch"])
def test_submission_control_service_preserves_possible_write_ambiguity(tmp_path: Path, failure: str) -> None:
    from qqtools.plugins.qexp.runtime.submission_control_continuation import submission_control_continuation

    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision = _registered(tmp_path, "project", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    try:
        cursor = submission_control_continuation()
        request = executor.prepare_submission_control_service(binding, revision, continuation=cursor)
        process = ProjectIOProcess(request, 2_000_000_000, 1, request.prepared_at, "running")
        atomic_replace(executor.paths["project_io_processes"] / f"{request.request_id}.json", process.to_dict())
        if failure == "late_epoch":
            executor.fence_epoch()
            assert project_io_worker_module._publish_result(
                executor.root,
                executor.paths,
                request,
                process,
                "completed",
                None,
                {"state": "waiting", "quiescent": True, "reason_code": "idle", "continuation": cursor},
                shared_write_possible=True,
            )
        else:
            executor.poll()
        result = executor.load_result(request.request_id)
        assert result is not None and result.status == "outcome_unknown"
        assert executor.consume(request.request_id, request) is None
        if failure == "worker_absence":
            assert executor.reset_ambiguous_submission_control_service_for_retry(request.request_id, request)
            assert executor.start(request.request_id) is not None
            assert _wait_for_consumed(executor, request).status == "completed"
    finally:
        executor.shutdown()


def test_upgrade_service_resumes_shared_journal_in_fresh_workers(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    manifest_path = shared_paths(cfg.shared_root)["upgrade"] / "protocol-manifest.json"
    manifest_path.unlink()
    schema_path = shared_paths(cfg.shared_root)["schema"] / "version.json"
    schema = read_json(schema_path)
    schema["schema"].pop("protocol", None)
    atomic_replace(schema_path, schema)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    try:
        deadline = time.monotonic() + 20.0
        states = []
        while time.monotonic() < deadline:
            request = executor.prepare_upgrade_service(binding, revision)
            assert executor.start(request.request_id) is not None
            result = _wait_for_consumed(executor, request)
            assert result.status == "completed", result
            states.append(result.evidence["state"])
            if not result.evidence["pending"]:
                break
        else:
            raise AssertionError(f"upgrade did not converge: {states}")
        from qqtools.plugins.qexp.runtime.upgrade.framework import UpgradeCoordinator

        status = UpgradeCoordinator(cfg).status()
        assert not status["pending"]
        assert len(states) >= 3
        assert read_json(manifest_path)["upgrade_protocol_manifest"]["protocol"] == "metadata:upgrade-journal-v1"
        assert not executor.has_unfinished_work()
    finally:
        executor.shutdown()


def test_upgrade_late_publication_after_epoch_fence_retains_write_ambiguity(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    try:
        request = executor.prepare_upgrade_service(binding, revision)
        process = ProjectIOProcess(request, 2_000_000_000, 1, request.prepared_at, "running")
        atomic_replace(executor.paths["project_io_processes"] / f"{request.request_id}.json", process.to_dict())
        executor.fence_epoch()
        assert project_io_worker_module._publish_result(
            executor.root,
            executor.paths,
            request,
            process,
            "completed",
            None,
            {
                "state": "completed",
                "pending": False,
                "can_run": False,
                "admission_blocked": False,
                "idle_blocking": False,
                "next_probe_at": None,
            },
            shared_write_possible=True,
        )
        result = executor.load_result(request.request_id)
        assert result is not None and result.status == "outcome_unknown"
        assert executor.consume(request.request_id, request) is None
    finally:
        executor.shutdown()


def test_upgrade_worker_absence_preserves_ambiguous_journal_recovery(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    try:
        request = executor.prepare_upgrade_service(binding, revision)
        process = ProjectIOProcess(request, 2_000_000_000, 1, request.prepared_at, "running")
        atomic_replace(executor.paths["project_io_processes"] / f"{request.request_id}.json", process.to_dict())
        executor.poll()
        result = executor.load_result(request.request_id)
        assert result is not None and result.status == "outcome_unknown"
        assert executor.consume(request.request_id, request) is None
        assert executor.reset_ambiguous_upgrade_service_for_retry(request.request_id, request)
        assert executor.start(request.request_id) is not None
        assert _wait_for_consumed(executor, request).status == "completed"
    finally:
        executor.shutdown()


@pytest.mark.parametrize(
    ("prepare_name", "reset_name"),
    [
        ("prepare_scheduler_due_offer", "reset_ambiguous_scheduler_due_offer_for_retry"),
        ("prepare_scheduler_ready_index_build", "reset_ambiguous_scheduler_ready_index_build_for_retry"),
    ],
)
def test_scheduler_mutation_worker_absence_replays_exact_request(
    tmp_path: Path,
    prepare_name: str,
    reset_name: str,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    try:
        request = getattr(executor, prepare_name)(binding, revision)
        process = ProjectIOProcess(request, 2_000_000_000, 1, request.prepared_at, "running")
        atomic_replace(executor.paths["project_io_processes"] / f"{request.request_id}.json", process.to_dict())

        executor.poll()

        result = executor.load_result(request.request_id)
        assert result is not None and result.status == "outcome_unknown"
        assert not (executor.paths["project_io_processes"] / f"{request.request_id}.json").exists()
        assert getattr(executor, reset_name)(request.request_id, request)
        assert executor.start(request.request_id) is not None
        assert _wait_for_consumed(executor, request).status == "completed"
    finally:
        executor.shutdown()


def test_terminal_observation_worker_absence_is_retryable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    request = executor.prepare_authority_terminal_observe(
        binding,
        revision,
        task_id="task-a",
        attempt_id="task-a-attempt-1",
        attempt_number=1,
        fencing_token=1,
        reservation_id=None,
        process_identity={
            "wrapper_pid": None,
            "wrapper_start_time_ticks": None,
            "process_group_id": None,
            "process_group_start_time_ticks": None,
        },
        mode="active",
    )
    before = _project_snapshot(cfg.shared_root)
    # Persist a started worker whose PID/start identity is positively absent.
    # Missing output from this read-only operation cannot pin its slot forever.
    process = ProjectIOProcess(request, 2_000_000_000, 1, request.prepared_at, "running")
    atomic_replace(executor.paths["project_io_processes"] / f"{request.request_id}.json", process.to_dict())
    with monkeypatch.context() as patch:
        patch.setattr(project_io_executor_module, "inspect_process_identity", lambda *_args: "unverified")
        status = executor.poll()
        assert status["exit_unverified_worker_count"] == 1
        assert executor.load_result(request.request_id) is None
        assert executor.consume(request.request_id, request) is None
        assert (executor.paths["project_io_processes"] / f"{request.request_id}.json").exists()
    executor.poll()
    result = executor.load_result(request.request_id)
    assert result is not None
    assert result.status == "retryable_error"
    assert result.reason_code == "project_io_worker_exited_without_result"
    assert not (executor.paths["project_io_processes"] / f"{request.request_id}.json").exists()
    assert executor.consume(request.request_id, request) == result
    assert not executor.has_unfinished_work()
    assert _project_snapshot(cfg.shared_root) == before


def test_validate_binding_runs_in_fresh_worker_and_consumes_exact_result(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()

    request = executor.prepare_validate_binding(binding, revision, {"inventory_revision": 1})
    executor.start(request.request_id)
    result = _wait_for_result(executor, request.request_id)

    assert result.status == "completed"
    assert result.request == request
    assert result.evidence["project_id"] == binding.project_id
    consumed = _wait_for_consumed(executor, request)
    assert consumed == result
    assert executor.status_view()["active_worker_count"] == 0
    assert not executor.has_unfinished_work()


def test_resolved_history_evicts_to_record_and_byte_bounds(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    request = executor.prepare_validate_binding(binding, revision)
    result = ProjectIOResult(
        request=request,
        status="retryable_error",
        reason_code="project_io_worker_exited_without_result",
        completed_at=request.prepared_at,
        evidence={},
    )

    for _ in range(PROJECT_IO_MAX_RESOLVED_RECORDS + 32):
        executor._commit_resolution(request.request_id, "consumed", request, result)

    entries = list(executor.paths["project_io_resolved"].iterdir())
    sizes = [path.stat().st_size for path in entries]
    assert len(entries) == PROJECT_IO_MAX_RESOLVED_RECORDS
    assert all(0 < size <= PROJECT_IO_MAX_RECORD_BYTES for size in sizes)
    assert sum(sizes) <= PROJECT_IO_MAX_RESOLVED_BYTES


@pytest.mark.parametrize(
    "requested,free,demand",
    [
        (1, 1, "runnable_now"),
        (2, 1, "waiting_for_aggregation"),
        (3, 2, "no_primary_demand"),
    ],
)
def test_primary_probe_reuses_capacity_policy_in_fresh_worker(tmp_path, requested, free, demand):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _registered(tmp_path, "primary", runtime)
    submit(cfg, ["echo", "primary"], requested_gpus=requested, working_dir=tmp_path)
    repair_ready_index(cfg)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    try:
        before = _project_snapshot(cfg.shared_root)
        request = executor.prepare_scheduler_primary_probe(
            binding,
            revision,
            lane="gpu",
            round_id="a" * 32,
            phase="scan",
            capacity_digest="b" * 64,
            visible_capacity=2,
            free_capacity=free,
            group_gpu_usage={},
            probe_state=encode_probe_session(PrimaryProbeSession(), binding.project_id, "gpu"),
        )
        executor.start(request.request_id)
        result = _wait_for_result(executor, request.request_id)
        assert result.status == "completed"
        assert result.evidence["demand"] == demand
        assert (result.evidence["route_revisions"] is not None) == (demand == "no_primary_demand")
        assert _project_snapshot(cfg.shared_root) == before
        assert _wait_for_consumed(executor, request) == result
    finally:
        executor.shutdown()


@pytest.mark.parametrize("lane,role", [("gpu", "primary"), ("cpu", "primary"), ("gpu", "borrow"), ("cpu", "borrow")])
def test_scheduler_quiescence_probe_sees_both_lanes_and_roles_without_shared_mutation(tmp_path, lane, role):
    from qqtools.plugins.qexp.commands.group import change_worker, create_group

    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _registered(tmp_path, "project", runtime)
    kwargs = {}
    if role == "borrow":
        create_group(cfg, "borrow-group")
        change_worker(cfg, "borrow-group", "gpu-1", "set", role="borrow")
        kwargs["group"] = "borrow-group"
    submit(
        cfg,
        ["echo", "work"],
        requested_gpus=1 if lane == "gpu" else 0,
        requested_cpus=1 if lane == "cpu" else None,
        working_dir=tmp_path,
        **kwargs,
    )
    repair_ready_index(cfg)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    try:
        before = _project_snapshot(cfg.shared_root)
        request = executor.prepare_scheduler_quiescence_probe(
            binding, revision, probe_state=encode_probe_session(PrimaryProbeSession(), binding.project_id, "gpu")
        )
        assert executor.start(request.request_id) is not None
        result = _wait_for_consumed(executor, request)
        assert result.status == "completed"
        assert result.evidence["state"] == "active"
        assert _project_snapshot(cfg.shared_root) == before
        assert executor.unresolved_requests() == ()
    finally:
        executor.shutdown()


def test_primary_verification_rejects_changed_completed_route(tmp_path):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _registered(tmp_path, "primary", runtime)
    repair_ready_index(cfg)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    try:

        def run_probe(state, phase):
            request = executor.prepare_scheduler_primary_probe(
                binding,
                revision,
                lane="gpu",
                round_id="a" * 32,
                phase=phase,
                capacity_digest="b" * 64,
                visible_capacity=1,
                free_capacity=1,
                group_gpu_usage={},
                probe_state=state,
            )
            executor.start(request.request_id)
            result = _wait_for_result(executor, request.request_id)
            assert result.status == "completed"
            _wait_for_consumed(executor, request)
            return result

        scanned = run_probe(encode_probe_session(PrimaryProbeSession(), binding.project_id, "gpu"), "scan")
        assert scanned.evidence["demand"] == "no_primary_demand"
        submit(cfg, ["echo", "new-primary"], working_dir=tmp_path)
        verified = run_probe(scanned.evidence["probe_state"], "verify")
        assert verified.evidence["demand"] == "unresolved"
        assert verified.evidence["route_revisions"] is None
        rescanned = run_probe(verified.evidence["probe_state"], "scan")
        assert rescanned.evidence["demand"] == "runnable_now"
    finally:
        executor.shutdown()


def test_scheduler_quiescence_restarts_a_completed_route_after_new_work(tmp_path):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _registered(tmp_path, "project", runtime)
    repair_ready_index(cfg)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    try:
        state = encode_probe_session(PrimaryProbeSession(), binding.project_id, "gpu")
        request = executor.prepare_scheduler_quiescence_probe(binding, revision, probe_state=state)
        assert executor.start(request.request_id) is not None
        empty = _wait_for_consumed(executor, request)
        assert empty.status == "completed"
        assert empty.evidence["state"] == "quiescent"

        submit(cfg, ["echo", "new-cpu-work"], requested_gpus=0, requested_cpus=1, working_dir=tmp_path)
        before = _project_snapshot(cfg.shared_root)
        request = executor.prepare_scheduler_quiescence_probe(
            binding, revision, probe_state=empty.evidence["probe_state"]
        )
        assert executor.start(request.request_id) is not None
        changed = _wait_for_consumed(executor, request)
        assert changed.status == "completed"
        assert changed.evidence["state"] == "active"
        assert _project_snapshot(cfg.shared_root) == before
    finally:
        executor.shutdown()


def test_scheduler_quiescence_restarts_an_unfinished_census_after_revision_change(tmp_path):
    from qqtools.plugins.qexp.commands.group import change_worker, create_group

    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _registered(tmp_path, "project", runtime)
    create_group(cfg, "limited-group")
    change_worker(cfg, "limited-group", "gpu-1", "set", gpu_limit_gpus=1, has_gpu_limit=True)
    for index in range(65):
        submit(
            cfg,
            ["echo", "work"],
            requested_gpus=2,
            group="limited-group",
            task_id=f"work-{index:03d}",
            working_dir=tmp_path,
        )
    repair_ready_index(cfg)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    try:
        request = executor.prepare_scheduler_quiescence_probe(
            binding, revision, probe_state=encode_probe_session(PrimaryProbeSession(), binding.project_id, "gpu")
        )
        assert executor.start(request.request_id) is not None
        partial = _wait_for_consumed(executor, request)
        assert partial.status == "completed"
        assert partial.evidence["state"] == "pending"
        route = partial.evidence["probe_state"]["routes"]["home"]
        assert not route["is_complete"]
        assert route["cursor"] is not None

        change_worker(cfg, "limited-group", "gpu-1", "set", gpu_limit_gpus=2, has_gpu_limit=True)
        submit(cfg, ["echo", "new-work"], requested_gpus=2, group="limited-group", working_dir=tmp_path)
        before = _project_snapshot(cfg.shared_root)
        request = executor.prepare_scheduler_quiescence_probe(
            binding, revision, probe_state=partial.evidence["probe_state"]
        )
        assert executor.start(request.request_id) is not None
        changed = _wait_for_consumed(executor, request)
        assert changed.status == "completed"
        assert changed.evidence["state"] == "active"
        # The newly eligible prefix must be checked, not the old cursor suffix.
        assert changed.evidence["probe_state"]["routes"]["home"]["cursor"] is None
        assert _project_snapshot(cfg.shared_root) == before
    finally:
        executor.shutdown()


@pytest.mark.parametrize("operation", ["primary", "quiescence"])
def test_probe_does_not_inherit_dispatch_cursor_suffix(tmp_path, operation):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _registered(tmp_path, "primary", runtime)
    task = submit(cfg, ["echo", "primary"], working_dir=tmp_path)
    repair_ready_index(cfg)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    try:
        observation = executor.prepare_scheduler_observe(
            binding,
            revision,
            lane="gpu",
            admission_role="primary",
            cursor_namespace=f"scheduler-{binding.project_id}-primary-gpu",
        )
        executor.start(observation.request_id)
        observed = _wait_for_result(executor, observation.request_id)
        assert observed.evidence["candidate"]["task_id"] == task.task_id
        _wait_for_consumed(executor, observation)
        commit = executor.prepare_scheduler_cursor_commit(
            binding,
            revision,
            cursor=observed.evidence["cursor"],
            source_revisions=observed.evidence["source_revisions"],
        )
        executor.start(commit.request_id)
        assert _wait_for_result(executor, commit.request_id).status == "completed"
        _wait_for_consumed(executor, commit)
        state = encode_probe_session(PrimaryProbeSession(), binding.project_id, "gpu")
        if operation == "quiescence":
            request = executor.prepare_scheduler_quiescence_probe(binding, revision, probe_state=state)
        else:
            request = executor.prepare_scheduler_primary_probe(
                binding,
                revision,
                lane="gpu",
                round_id="a" * 32,
                phase="scan",
                capacity_digest="b" * 64,
                visible_capacity=1,
                free_capacity=1,
                group_gpu_usage={},
                probe_state=state,
            )
        before = _project_snapshot(cfg.shared_root)
        executor.start(request.request_id)
        result = _wait_for_consumed(executor, request)
        assert result.status == "completed"
        if operation == "quiescence":
            assert result.evidence["state"] == "active"
        else:
            assert result.evidence["demand"] == "runnable_now"
            assert result.evidence["route_revisions"] is None
        assert _project_snapshot(cfg.shared_root) == before
    finally:
        executor.shutdown()


@pytest.mark.parametrize("started", [False, True])
def test_stale_primary_probe_reclaims_only_readonly_progress(tmp_path, started):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _registered(tmp_path, "primary", runtime)
    repair_ready_index(cfg)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    try:
        request = executor.prepare_scheduler_primary_probe(
            binding,
            revision,
            lane="gpu",
            round_id="a" * 32,
            phase="scan",
            capacity_digest="b" * 64,
            visible_capacity=1,
            free_capacity=1,
            group_gpu_usage={},
            probe_state=encode_probe_session(PrimaryProbeSession(), binding.project_id, "gpu"),
        )
        if started:
            executor.start(request.request_id)
            assert _wait_for_result(executor, request.request_id).status == "completed"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            executor.poll()
            if executor.resolve_stale_primary_probe(request.request_id, request):
                break
            time.sleep(0.02)
        assert not executor.has_unfinished_work()
        assert executor.consume(request.request_id, request) is None
    finally:
        executor.shutdown()


def test_registration_renew_runs_in_fresh_worker_and_reports_durable_expiry(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    registration_path = cfg.shared_root / "machines" / cfg.machine_name / "registration.json"
    registration = read_json(registration_path)
    registration["registration"]["eligibility_expires_at"] = (
        # Keep eligibility valid through fresh-interpreter startup.  This is
        # still inside the 120-second renewal horizon exercised below.
        datetime.now(timezone.utc) + timedelta(seconds=30)
    ).isoformat()
    atomic_replace(registration_path, registration)

    request = executor.prepare_registration_renew(
        binding,
        revision,
        renewal_horizon_seconds=120.0,
    )
    executor.start(request.request_id)
    result = _wait_for_result(executor, request.request_id)

    assert result.status == "completed"
    assert result.evidence["outcome"] == "eligible"
    assert result.evidence["renewed"] is True
    assert result.evidence["renew_after_seconds"] == pytest.approx(10.0)
    assert (
        read_json(registration_path)["registration"]["eligibility_expires_at"]
        == result.evidence["eligibility_expires_at"]
    )
    deadline = time.monotonic() + 5.0
    while not executor.has_ready_result() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert executor.has_ready_result()
    assert _wait_for_consumed(executor, request) == result
    assert not executor.has_unfinished_work()


def test_fresh_worker_refreshes_stale_client_version_without_replacing_owner(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    registration_path = cfg.shared_root / "machines" / cfg.machine_name / "registration.json"
    envelope = read_json(registration_path)
    registration = envelope["registration"]
    registration["client_version"] = "0.0.1"
    preserved = {
        key: registration[key]
        for key in (
            "version",
            "project_id",
            "shared_root",
            "machine_name",
            "generation",
            "protocol_version",
            "runtime_instance_id",
            "runtime_root",
            "state",
            "created_at",
        )
    }
    atomic_replace(registration_path, envelope)

    request = executor.prepare_registration_renew(binding, revision, renewal_horizon_seconds=0.0)
    executor.start(request.request_id)
    result = _wait_for_result(executor, request.request_id)

    assert result.status == "completed"
    assert result.evidence["outcome"] == "eligible"
    assert result.evidence["renewed"] is True
    refreshed = read_json(registration_path)["registration"]
    assert refreshed["client_version"] == __version__
    assert {key: refreshed[key] for key in preserved} == preserved
    assert _wait_for_consumed(executor, request) == result
    executor.shutdown()


def test_registration_renew_stale_registry_does_not_write_shared_registration(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    request = executor.prepare_registration_renew(binding, revision, renewal_horizon_seconds=120.0)
    registration_path = cfg.shared_root / "machines" / cfg.machine_name / "registration.json"
    before = registration_path.read_bytes()
    runtime.set_enabled(binding.project_id, False)

    executor.start(request.request_id)
    result = _wait_for_result(executor, request.request_id)

    assert result.status == "completed"
    assert result.evidence == {
        "outcome": "stale",
        "renewed": False,
        "eligibility_expires_at": None,
        "renew_after_seconds": None,
        "reason": "binding_fence",
    }
    assert registration_path.read_bytes() == before


def test_missing_registration_renew_worker_result_is_outcome_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    request = executor.prepare_registration_renew(binding, revision, renewal_horizon_seconds=120.0)
    process = ProjectIOProcess(
        request=request,
        pid=999_999,
        start_time_ticks=1,
        started_at="2026-09-28T00:00:00+00:00",
        state="running",
    )
    executor._write_record(
        executor._record_path("processes", request.request_id),
        process.to_dict(),
        "project_io_process",
    )
    monkeypatch.setattr(project_io_executor_module, "inspect_process_identity", lambda *_args: "absent")

    executor.poll()
    result = executor.load_result(request.request_id)

    assert result is not None
    assert result.status == "outcome_unknown"
    assert result.reason_code == "project_io_outcome_unknown"
    assert result.evidence == {}


def test_registration_renew_exception_after_write_boundary_is_outcome_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    request = executor.prepare_registration_renew(binding, revision, renewal_horizon_seconds=120.0)
    pid = os.getpid()
    ticks = project_io_worker_module._process_start_time_ticks(pid)
    assert ticks is not None
    process = ProjectIOProcess(
        request=request,
        pid=pid,
        start_time_ticks=ticks,
        started_at="2026-09-28T00:00:00+00:00",
        state="running",
    )
    executor._write_record(
        executor._record_path("processes", request.request_id),
        process.to_dict(),
        "project_io_process",
    )

    def fail_after_write_boundary(_request, _root, _paths, write_possible):
        write_possible[0] = True
        raise OSError("injected post-boundary failure")

    monkeypatch.setattr(project_io_worker_module, "_registration_renew", fail_after_write_boundary)

    assert project_io_worker_module._run(runtime.root, request.request_id) == 0
    result = executor.load_result(request.request_id)
    assert result is not None
    assert result.status == "outcome_unknown"
    assert result.reason_code == "project_io_outcome_unknown"
    assert result.evidence == {}


def _authority_renewal_request(executor, binding, revision):
    return executor.prepare_authority_renewal(
        binding,
        revision,
        task_id="task-a",
        attempt_id="task-a-attempt-1",
        attempt_number=1,
        fencing_token=1,
        reservation_id="reservation-a",
        process_identity={
            "wrapper_pid": 101,
            "wrapper_start_time_ticks": 202,
            "process_group_id": 303,
            "process_group_start_time_ticks": 404,
        },
        source_revisions={"task": 1, "attempt_digest": "d" * 64},
    )


def test_missing_authority_renewal_worker_result_is_outcome_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    request = _authority_renewal_request(executor, binding, revision)
    process = ProjectIOProcess(
        request=request,
        pid=999_999,
        start_time_ticks=1,
        started_at="2026-09-28T00:00:00+00:00",
        state="running",
    )
    executor._write_record(
        executor._record_path("processes", request.request_id),
        process.to_dict(),
        "project_io_process",
    )
    monkeypatch.setattr(project_io_executor_module, "inspect_process_identity", lambda *_args: "absent")

    executor.poll()
    result = executor.load_result(request.request_id)

    assert result is not None
    assert result.status == "outcome_unknown"
    assert result.reason_code == "project_io_outcome_unknown"


def test_authority_renewal_exception_after_write_boundary_is_outcome_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    request = _authority_renewal_request(executor, binding, revision)
    pid = os.getpid()
    ticks = project_io_worker_module._process_start_time_ticks(pid)
    assert ticks is not None
    process = ProjectIOProcess(
        request=request,
        pid=pid,
        start_time_ticks=ticks,
        started_at="2026-09-28T00:00:00+00:00",
        state="running",
    )
    executor._write_record(
        executor._record_path("processes", request.request_id),
        process.to_dict(),
        "project_io_process",
    )

    def fail_after_write_boundary(
        _request,
        _runtime_root,
        _paths,
        write_possible,
        *,
        replay_only=False,
        active_executor_epoch=None,
    ):
        assert replay_only is False
        assert active_executor_epoch == request.executor_epoch
        write_possible[0] = True
        raise OSError("injected post-boundary failure")

    monkeypatch.setattr(project_io_worker_module, "_authority_renewal", fail_after_write_boundary)

    assert project_io_worker_module._run(runtime.root, request.request_id) == 0
    result = executor.load_result(request.request_id)
    assert result is not None
    assert result.status == "outcome_unknown"
    assert result.reason_code == "project_io_outcome_unknown"


def test_activation_observe_register_and_ack_run_in_fresh_workers(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    checkpoint = publish_project_activation(cfg, "test")["project_activation"]
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()

    observe = executor.prepare_activation_observe(binding, revision)
    executor.start(observe.request_id)
    observed = _wait_for_result(executor, observe.request_id)
    assert observed.status == "completed"
    assert observed.evidence == {
        "outcome": "observed",
        "checkpoint": {"epoch": checkpoint["epoch"], "sequence": checkpoint["sequence"]},
        "replay": {
            "epoch": checkpoint["epoch"],
            "sequence": checkpoint["sequence"],
            "reconstructed_floor": None,
            "complete": True,
        },
    }
    assert _wait_for_consumed(executor, observe) == observed

    register = executor.prepare_activation_consumer_register(
        binding,
        revision,
        process_fence="process-a",
    )
    executor.start(register.request_id)
    registered = _wait_for_result(executor, register.request_id)
    assert registered.status == "completed"
    assert registered.evidence == {"outcome": "registered", "acknowledgement": None}
    assert _wait_for_consumed(executor, register) == registered

    ack = executor.prepare_activation_consumer_ack(
        binding,
        revision,
        process_fence="process-a",
        epoch=checkpoint["epoch"],
        sequence=checkpoint["sequence"],
        require_current=True,
    )
    executor.start(ack.request_id)
    acknowledged = _wait_for_result(executor, ack.request_id)
    assert acknowledged.status == "completed"
    assert acknowledged.evidence == {
        "outcome": "acknowledged",
        "acknowledgement": {"epoch": checkpoint["epoch"], "sequence": checkpoint["sequence"]},
    }
    assert _wait_for_consumed(executor, ack) == acknowledged
    progress = read_consumer_progress(
        cfg.shared_root,
        runtime_id=runtime.instance_id,
        project_id=binding.project_id,
        registration_generation=binding.registration_generation,
    )
    assert progress is not None
    assert progress["project_activation_consumer"]["ack"] == acknowledged.evidence["acknowledgement"]


def test_activation_observe_derives_compacted_reconstruction_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    monkeypatch.setattr(project_activation_module, "_COMPACTION_BATCH", 3)
    checkpoint = None
    for index in range(3):
        checkpoint = publish_project_activation(cfg, f"work-{index}")["project_activation"]
    assert checkpoint is not None
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()

    request = executor.prepare_activation_observe(binding, revision)
    executor.start(request.request_id)
    result = _wait_for_result(executor, request.request_id)

    assert result.status == "completed"
    assert result.evidence["replay"] == {
        "epoch": checkpoint["epoch"],
        "sequence": checkpoint["sequence"],
        "reconstructed_floor": checkpoint["sequence"],
        "complete": True,
    }


def test_activation_observe_advances_replay_in_bounded_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    monkeypatch.setattr(project_activation_module, "_maybe_compact_activation_locked", lambda *_args: None)
    checkpoint = None
    for index in range(257):
        checkpoint = publish_project_activation(cfg, f"work-{index}")["project_activation"]
    assert checkpoint is not None
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()

    first = executor.prepare_activation_observe(binding, revision)
    executor.start(first.request_id)
    first_result = _wait_for_result(executor, first.request_id)
    assert first_result.evidence["replay"] == {
        "epoch": checkpoint["epoch"],
        "sequence": 256,
        "reconstructed_floor": None,
        "complete": False,
    }
    assert _wait_for_consumed(executor, first) == first_result

    second = executor.prepare_activation_observe(
        binding,
        revision,
        replay_epoch=checkpoint["epoch"],
        replay_sequence=256,
    )
    executor.start(second.request_id)
    second_result = _wait_for_result(executor, second.request_id)
    assert second_result.evidence["replay"] == {
        "epoch": checkpoint["epoch"],
        "sequence": 257,
        "reconstructed_floor": None,
        "complete": True,
    }


def test_activation_reobservation_refreshes_a_concurrently_advanced_compaction_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    monkeypatch.setattr(project_activation_module, "_maybe_compact_activation_locked", lambda *_args: None)
    checkpoint = None
    for index in range(556):
        checkpoint = publish_project_activation(cfg, f"work-{index}")["project_activation"]
    assert checkpoint is not None
    assert compact_project_activation(cfg.shared_root)["project_activation_snapshot"]["floor_sequence"] == 256
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()

    register = executor.prepare_activation_consumer_register(
        binding,
        revision,
        process_fence="process-a",
    )
    executor.start(register.request_id)
    assert _wait_for_result(executor, register.request_id).status == "completed"
    _wait_for_consumed(executor, register)

    first = executor.prepare_activation_observe(binding, revision)
    executor.start(first.request_id)
    first_result = _wait_for_result(executor, first.request_id)
    first_replay = first_result.evidence["replay"]
    assert first_replay == {
        "epoch": checkpoint["epoch"],
        "sequence": 512,
        "reconstructed_floor": 256,
        "complete": False,
    }
    _wait_for_consumed(executor, first)

    assert compact_project_activation(cfg.shared_root)["project_activation_snapshot"]["floor_sequence"] == 512
    stale_ack = executor.prepare_activation_consumer_ack(
        binding,
        revision,
        process_fence="process-a",
        epoch=checkpoint["epoch"],
        sequence=first_replay["sequence"],
        reconstructed_floor=first_replay["reconstructed_floor"],
    )
    executor.start(stale_ack.request_id)
    assert _wait_for_result(executor, stale_ack.request_id).status == "retryable_error"
    _wait_for_consumed(executor, stale_ack)

    refreshed = executor.prepare_activation_observe(
        binding,
        revision,
        replay_epoch=checkpoint["epoch"],
        replay_sequence=first_replay["sequence"],
    )
    executor.start(refreshed.request_id)
    refreshed_result = _wait_for_result(executor, refreshed.request_id)
    refreshed_replay = refreshed_result.evidence["replay"]
    assert refreshed_replay == {
        "epoch": checkpoint["epoch"],
        "sequence": 556,
        "reconstructed_floor": 512,
        "complete": True,
    }
    _wait_for_consumed(executor, refreshed)

    final_ack = executor.prepare_activation_consumer_ack(
        binding,
        revision,
        process_fence="process-a",
        epoch=checkpoint["epoch"],
        sequence=refreshed_replay["sequence"],
        reconstructed_floor=refreshed_replay["reconstructed_floor"],
        require_current=True,
    )
    executor.start(final_ack.request_id)
    final_result = _wait_for_result(executor, final_ack.request_id)
    assert final_result.status == "completed"
    assert final_result.evidence["outcome"] == "acknowledged"


@pytest.mark.parametrize("operation_kind", ["activation_consumer_register", "activation_consumer_ack"])
def test_missing_activation_mutation_worker_result_is_outcome_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation_kind: str,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    checkpoint = publish_project_activation(cfg, "test")["project_activation"]
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    if operation_kind == "activation_consumer_register":
        request = executor.prepare_activation_consumer_register(
            binding,
            revision,
            process_fence="process-a",
        )
    else:
        request = executor.prepare_activation_consumer_ack(
            binding,
            revision,
            process_fence="process-a",
            epoch=checkpoint["epoch"],
            sequence=checkpoint["sequence"],
        )
    process = ProjectIOProcess(
        request=request,
        pid=999_999,
        start_time_ticks=1,
        started_at="2026-09-28T00:00:00+00:00",
        state="running",
    )
    executor._write_record(
        executor._record_path("processes", request.request_id),
        process.to_dict(),
        "project_io_process",
    )
    monkeypatch.setattr(project_io_executor_module, "inspect_process_identity", lambda *_args: "absent")

    executor.poll()
    result = executor.load_result(request.request_id)

    assert result is not None
    assert result.status == "outcome_unknown"
    assert result.reason_code == "project_io_outcome_unknown"
    assert result.evidence == {}


def test_activation_mutation_exception_after_write_boundary_is_outcome_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    request = executor.prepare_activation_consumer_register(
        binding,
        revision,
        process_fence="process-a",
    )
    pid = os.getpid()
    ticks = project_io_worker_module._process_start_time_ticks(pid)
    assert ticks is not None
    process = ProjectIOProcess(
        request=request,
        pid=pid,
        start_time_ticks=ticks,
        started_at="2026-09-28T00:00:00+00:00",
        state="running",
    )
    executor._write_record(
        executor._record_path("processes", request.request_id),
        process.to_dict(),
        "project_io_process",
    )

    def fail_after_write_boundary(_request, _runtime_root, _paths, write_possible):
        write_possible[0] = True
        raise OSError("injected post-boundary failure")

    monkeypatch.setattr(project_io_worker_module, "_activation_consumer_mutation", fail_after_write_boundary)

    assert project_io_worker_module._run(runtime.root, request.request_id) == 0
    result = executor.load_result(request.request_id)
    assert result is not None
    assert result.status == "outcome_unknown"
    assert result.reason_code == "project_io_outcome_unknown"
    assert result.evidence == {}


def test_maintenance_flush_event_publishes_shared_but_retains_local_source(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    event_id = "a" * 32
    event = {
        "event_id": event_id,
        "event_type": "test_diagnostic",
        "task_id": "task-a",
        "attempt_id": "task-a-attempt-1",
        "machine_name": "gpu-1",
        "timestamp": "2026-09-28T00:00:00+00:00",
        "details": {"reason": "test"},
    }
    bucket = event["attempt_id"]
    source = local_paths(runtime.project_paths(binding.project_id)["root"])["events"] / bucket / f"{event_id}.json"
    atomic_replace(source, event)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()

    request = executor.prepare_maintenance_flush_event(
        binding,
        revision,
        bucket=bucket,
        filename=source.name,
        event_id=event_id,
        sha256=digest,
    )
    executor.start(request.request_id)
    result = _wait_for_result(executor, request.request_id)

    assert result.status == "completed"
    assert result.evidence == {"outcome": "flushed", "event_id": event_id, "sha256": digest, "reason": None}
    assert read_json(shared_paths(cfg.shared_root)["events"] / "2026-09-28" / source.name) == event
    assert source.exists()
    assert _wait_for_consumed(executor, request) == result
    assert executor.retire_maintenance_event_source(request)
    assert not source.exists()


def test_maintenance_event_retirement_does_not_delete_concurrent_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    event_id = "c" * 32
    bucket = "machine"
    source = local_paths(runtime.project_paths(binding.project_id)["root"])["events"] / bucket / f"{event_id}.json"
    original = {
        "event_id": event_id,
        "event_type": "original",
        "task_id": None,
        "machine_name": "gpu-1",
        "timestamp": "2026-09-28T00:00:00+00:00",
        "details": {},
    }
    replacement = {**original, "event_type": "replacement"}
    atomic_replace(source, original)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    request = executor.prepare_maintenance_flush_event(
        binding,
        revision,
        bucket=bucket,
        filename=source.name,
        event_id=event_id,
        sha256=digest,
    )
    real_replace = project_io_executor_module.os.replace
    injected = False

    def replace_after_concurrent_write(source_path, destination_path, **kwargs):
        nonlocal injected
        if source_path == source.name and not injected:
            injected = True
            atomic_replace(source, replacement)
        return real_replace(source_path, destination_path, **kwargs)

    monkeypatch.setattr(project_io_executor_module.os, "replace", replace_after_concurrent_write)

    assert executor.retire_maintenance_event_source(request) is False
    assert read_json(source) == replacement


def test_maintenance_event_retirement_rejects_bucket_symlink_swap(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    event_id = "f" * 32
    events_root = local_paths(runtime.project_paths(binding.project_id)["root"])["events"]
    bucket = events_root / "machine"
    source = bucket / f"{event_id}.json"
    event = {
        "event_id": event_id,
        "event_type": "original",
        "task_id": None,
        "machine_name": "gpu-1",
        "timestamp": "2026-09-28T00:00:00+00:00",
        "details": {},
    }
    atomic_replace(source, event)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    request = executor.prepare_maintenance_flush_event(
        binding,
        revision,
        bucket="machine",
        filename=source.name,
        event_id=event_id,
        sha256=digest,
    )
    original_bucket = events_root / "machine-original"
    bucket.rename(original_bucket)
    external = tmp_path / "external"
    atomic_replace(external / source.name, event)
    bucket.symlink_to(external, target_is_directory=True)

    assert executor.retire_maintenance_event_source(request) is False
    assert read_json(external / source.name) == event
    assert read_json(original_bucket / source.name) == event


def test_scheduler_observe_returns_exact_candidate_without_mutating_project(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    task = submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    before = _project_snapshot(cfg.shared_root)

    request = executor.prepare_scheduler_observe(
        binding,
        revision,
        lane="gpu",
        admission_role="primary",
        cursor_namespace="scheduler-primary-gpu",
    )
    executor.start(request.request_id)
    result = _wait_for_result(executor, request.request_id)

    assert result.status == "completed"
    assert result.evidence["outcome"] == "candidate"
    assert result.evidence["reason"] == "candidate_ready"
    assert result.evidence["candidate"]["task_id"] == task.task_id
    assert result.evidence["candidate"]["attempt_id"] == f"{task.task_id}-attempt-1"
    assert result.evidence["candidate"]["lane"] == "gpu"
    assert result.evidence["cursor"]["namespace"] == "scheduler-primary-gpu"
    assert _project_snapshot(cfg.shared_root) == before
    assert _wait_for_consumed(executor, request) == result


def test_scheduler_launch_authorize_commits_exact_gate_and_replays_launch_id(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    task = submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    attempt = claim_task(
        cfg,
        task.task_id,
        [0],
        reservation_runtime_root=runtime.root,
        project_id=binding.project_id,
        admission_role="primary",
    )
    assert attempt is not None
    identity = {
        "task_id": task.task_id,
        "attempt_id": attempt.attempt_id,
        "attempt_number": attempt.attempt_number,
        "fencing_token": attempt.current_fencing_token,
        "reservation_id": attempt.reservation_id,
    }
    executor = ProjectIOExecutor(runtime)
    executor_epoch = executor.begin_epoch()
    atomic_replace(
        local_paths(runtime.project_paths(binding.project_id)["root"])["clock_health"],
        {
            "clock_capability": {
                "status": "healthy",
                "reason": "healthy",
                "providers": ["linux_adjtimex"],
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "observation": {
                    "observation_id": "b" * 32,
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
    reservation_path = runtime.paths["active"] / f"{attempt.reservation_id}.json"
    reservation_record = read_json(reservation_path)
    reservation_record["reservation"]["executor_owner"] = {
        "executor_epoch": executor_epoch,
        "request_id": "a" * 32,
        "registration_generation": binding.registration_generation,
    }
    atomic_replace(reservation_path, reservation_record)
    reservation_identity = ReservationIdentity.from_record(reservation_record["reservation"])

    reconciliation = executor.prepare_scheduler_reservation_reconcile(
        binding,
        revision,
        reservation_identity=reservation_identity,
    )
    dead_reconciliation = ProjectIOProcess(
        reconciliation,
        2_000_000_000,
        1,
        reconciliation.prepared_at,
        "running",
    )
    atomic_replace(
        executor.paths["project_io_processes"] / f"{reconciliation.request_id}.json",
        dead_reconciliation.to_dict(),
    )
    executor.poll()
    retryable = executor.load_result(reconciliation.request_id)
    assert retryable is not None and retryable.status == "retryable_error"
    assert executor.consume(reconciliation.request_id, reconciliation) == retryable

    request = executor.prepare_scheduler_launch_authorize(
        binding,
        revision,
        claim_identity=identity,
        reservation_identity=reservation_identity,
    )
    dead_launch = ProjectIOProcess(request, 2_000_000_000, 1, request.prepared_at, "running")
    atomic_replace(
        executor.paths["project_io_processes"] / f"{request.request_id}.json",
        dead_launch.to_dict(),
    )
    executor.poll()
    ambiguous = executor.load_result(request.request_id)
    assert ambiguous is not None and ambiguous.status == "outcome_unknown"
    assert executor.reset_ambiguous_scheduler_launch_authorize_for_retry(request.request_id, request)
    executor.start(request.request_id)
    result = _wait_for_consumed(executor, request)

    assert result.status == "completed"
    assert result.evidence["outcome"] == "authorized"
    assert result.evidence["claim_identity"] == identity
    launch_id = result.evidence["launch_id"]
    current = load_task(cfg, task.task_id)
    assert current.claim_control["active_claim"]["launch_state"] == "starting"
    assert current.claim_control["active_claim"]["launch_id"] == launch_id
    persisted_attempt_record = read_json(
        shared_paths(cfg.shared_root)["attempts"] / task.task_id / f"{attempt.attempt_number}.json"
    )
    persisted_attempt = persisted_attempt_record["attempt"]
    assert persisted_attempt["phase"] == "starting"
    assert persisted_attempt["authorization"]["launch_id"] == launch_id

    # Replay repairs the only permitted partial pair: Task launch authority
    # committed, while the matching Attempt write did not become durable.
    persisted_attempt["phase"] = "claimed"
    persisted_attempt["authorization"].pop("launch_id")
    persisted_attempt["authorization"].pop("launch_handoff_timeout_seconds")
    persisted_attempt["timestamps"]["launch_authorized_at"] = None
    atomic_replace(
        shared_paths(cfg.shared_root)["attempts"] / task.task_id / f"{attempt.attempt_number}.json",
        persisted_attempt_record,
    )
    current.control["cancellation_requested_at"] = "2026-09-28T00:00:00Z"
    save_task(cfg, current)

    replay = executor.prepare_scheduler_launch_authorize(
        binding,
        revision,
        claim_identity=identity,
        reservation_identity=reservation_identity,
    )
    executor.start(replay.request_id)
    replayed = _wait_for_consumed(executor, replay)

    assert replayed.evidence["outcome"] == "authorized", replayed.evidence
    assert replayed.evidence["launch_id"] == launch_id
    repaired_attempt = read_json(
        shared_paths(cfg.shared_root)["attempts"] / task.task_id / f"{attempt.attempt_number}.json"
    )["attempt"]
    assert repaired_attempt["phase"] == "starting"
    assert repaired_attempt["authorization"]["launch_id"] == launch_id

    repaired_record = read_json(
        shared_paths(cfg.shared_root)["attempts"] / task.task_id / f"{attempt.attempt_number}.json"
    )
    repaired_record["attempt"]["timestamps"]["launch_authorized_at"] = "2000-01-01T00:00:00Z"
    atomic_replace(
        shared_paths(cfg.shared_root)["attempts"] / task.task_id / f"{attempt.attempt_number}.json",
        repaired_record,
    )
    inconsistent = executor.prepare_scheduler_launch_authorize(
        binding,
        revision,
        claim_identity=identity,
        reservation_identity=reservation_identity,
    )
    executor.start(inconsistent.request_id)
    inconsistent_result = _wait_for_consumed(executor, inconsistent)
    assert inconsistent_result.evidence["outcome"] == "denied"
    assert inconsistent_result.evidence["reason"] == "attempt_not_current"


@pytest.mark.parametrize("expires_at", ["2000-01-01T00:00:00Z", "not-a-timestamp"])
def test_scheduler_launch_authorize_denies_invalid_bounded_lease(tmp_path: Path, expires_at: str) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    task = submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    attempt = claim_task(
        cfg,
        task.task_id,
        [0],
        reservation_runtime_root=runtime.root,
        project_id=binding.project_id,
        admission_role="primary",
    )
    assert attempt is not None
    current = load_task(cfg, task.task_id)
    current.claim_control["active_claim"]["lease_expires_at"] = expires_at
    save_task(cfg, current)
    attempt_path = shared_paths(cfg.shared_root)["attempts"] / task.task_id / f"{attempt.attempt_number}.json"
    attempt_record = read_json(attempt_path)
    attempt_record["attempt"]["lease"]["expires_at"] = expires_at
    atomic_replace(attempt_path, attempt_record)
    identity = {
        "task_id": task.task_id,
        "attempt_id": attempt.attempt_id,
        "attempt_number": attempt.attempt_number,
        "fencing_token": attempt.current_fencing_token,
        "reservation_id": attempt.reservation_id,
    }
    executor = ProjectIOExecutor(runtime)
    executor_epoch = executor.begin_epoch()
    reservation_path = runtime.paths["active"] / f"{attempt.reservation_id}.json"
    reservation_record = read_json(reservation_path)
    reservation_record["reservation"]["executor_owner"] = {
        "executor_epoch": executor_epoch,
        "request_id": "a" * 32,
        "registration_generation": binding.registration_generation,
    }
    atomic_replace(reservation_path, reservation_record)
    reservation_identity = ReservationIdentity.from_record(reservation_record["reservation"])

    request = executor.prepare_scheduler_launch_authorize(
        binding,
        revision,
        claim_identity=identity,
        reservation_identity=reservation_identity,
    )
    executor.start(request.request_id)
    result = _wait_for_consumed(executor, request)

    assert result.evidence["outcome"] == "denied"
    assert result.evidence["reason"] == "authority_unavailable"
    assert load_task(cfg, task.task_id).claim_control["active_claim"]["launch_state"] == "claimed"


def test_scheduler_launch_authorize_denies_after_holder_safe_deadline(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    task = submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    attempt = claim_task(
        cfg,
        task.task_id,
        [0],
        reservation_runtime_root=runtime.root,
        project_id=binding.project_id,
        admission_role="primary",
    )
    assert attempt is not None
    expires_at = (datetime.now(timezone.utc) + timedelta(milliseconds=500)).isoformat().replace("+00:00", "Z")
    current = load_task(cfg, task.task_id)
    current.claim_control["active_claim"]["lease_expires_at"] = expires_at
    current.claim_control["active_claim"]["clock_error_bound_seconds"] = 1.0
    save_task(cfg, current)
    attempt_path = shared_paths(cfg.shared_root)["attempts"] / task.task_id / f"{attempt.attempt_number}.json"
    attempt_record = read_json(attempt_path)
    attempt_record["attempt"]["lease"]["expires_at"] = expires_at
    atomic_replace(attempt_path, attempt_record)
    identity = {
        "task_id": task.task_id,
        "attempt_id": attempt.attempt_id,
        "attempt_number": attempt.attempt_number,
        "fencing_token": attempt.current_fencing_token,
        "reservation_id": attempt.reservation_id,
    }
    executor = ProjectIOExecutor(runtime)
    executor_epoch = executor.begin_epoch()
    reservation_path = runtime.paths["active"] / f"{attempt.reservation_id}.json"
    reservation_record = read_json(reservation_path)
    reservation_record["reservation"]["executor_owner"] = {
        "executor_epoch": executor_epoch,
        "request_id": "a" * 32,
        "registration_generation": binding.registration_generation,
    }
    atomic_replace(reservation_path, reservation_record)
    reservation_identity = ReservationIdentity.from_record(reservation_record["reservation"])

    request = executor.prepare_scheduler_launch_authorize(
        binding,
        revision,
        claim_identity=identity,
        reservation_identity=reservation_identity,
    )
    executor.start(request.request_id)
    result = _wait_for_consumed(executor, request)

    assert result.evidence == {
        "outcome": "denied",
        "claim_identity": identity,
        "reason": "authority_unavailable",
    }


def test_scheduler_observe_uses_its_independent_cursor_namespace(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    task = submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    reservation = read_json(
        shared_paths(cfg.shared_root)["ready_reservations"] / f"{task.task_id}.{task.ready_generation}.json"
    )["ready_reservation"]
    namespace = "scheduler-primary-gpu"
    atomic_replace(
        shared_paths(cfg.shared_root)["ready_cursors"] / f"{namespace}.gpu-1.home.json",
        {
            "cursor": {
                "schema_version": 1,
                "project_id": namespace,
                "machine_name": "gpu-1",
                "queue_scope": "home",
                "catalog_page": str(reservation["catalog_page"]),
                "partition": reservation["partition"],
                "after_name": reservation["marker_name"],
                "revision": 1,
            }
        },
    )
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()

    request = executor.prepare_scheduler_observe(
        binding,
        revision,
        lane="gpu",
        admission_role="primary",
        cursor_namespace=namespace,
    )
    executor.start(request.request_id)
    result = _wait_for_result(executor, request.request_id)

    assert result.status == "completed"
    assert result.evidence["outcome"] == "none"
    assert result.evidence["cursor"]["routes"]["home"]["observed"]["after_name"] == reservation["marker_name"]


def test_scheduler_cursor_commit_advances_only_the_exact_observed_cursor(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    task = submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    namespace = "scheduler-primary-gpu"
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()

    observe = executor.prepare_scheduler_observe(
        binding,
        revision,
        lane="gpu",
        admission_role="primary",
        cursor_namespace=namespace,
    )
    executor.start(observe.request_id)
    observed = _wait_for_consumed(executor, observe)
    cursor = dict(observed.evidence["cursor"])
    assert observed.evidence["candidate"]["task_id"] == task.task_id

    commit = executor.prepare_scheduler_cursor_commit(binding, revision, cursor=cursor)
    executor.start(commit.request_id)
    committed = _wait_for_consumed(executor, commit)

    assert committed.status == "completed"
    assert committed.evidence["routes"] == {"home": "committed", "shared": "already_applied"}
    saved = load_ready_cursor(cfg, namespace, "home")
    expected = cursor["routes"]["home"]["next"]
    assert {
        "catalog_page": saved.catalog_page,
        "partition": saved.partition,
        "after_name": saved.after_name,
        "revision": saved.revision,
    } == expected

    stale_cursor = {
        **cursor,
        "routes": {
            **cursor["routes"],
            "home": {
                "observed": cursor["routes"]["home"]["observed"],
                "next": {
                    **cursor["routes"]["home"]["next"],
                    "revision": cursor["routes"]["home"]["next"]["revision"] + 1,
                },
            },
        },
    }
    stale = executor.prepare_scheduler_cursor_commit(binding, revision, cursor=stale_cursor)
    executor.start(stale.request_id)
    stale_result = _wait_for_consumed(executor, stale)

    assert stale_result.evidence["routes"]["home"] == "stale"
    assert load_ready_cursor(cfg, namespace, "home") == saved


def test_ready_cursor_commit_rechecks_authority_at_replace_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1")
    namespace = "scheduler-primary-gpu"
    observed = load_ready_cursor(cfg, namespace, "home")
    next_cursor = ReadyCursor(namespace, cfg.machine_name, "home", 0, "00", "task-a.1.json", 1)
    authority_current = [True]
    original_replace = ready_traversal.atomic_replace

    def fence_at_replace(path, value, *, before_replace=None, io_step_observer=None):
        assert before_replace is not None
        authority_current[0] = False
        return original_replace(
            path,
            value,
            before_replace=before_replace,
            io_step_observer=io_step_observer,
        )

    def require_authority() -> None:
        if not authority_current[0]:
            raise RuntimeError("authority fenced before replace")

    monkeypatch.setattr(ready_traversal, "atomic_replace", fence_at_replace)

    with pytest.raises(RuntimeError, match="fenced before replace"):
        compare_and_commit_ready_cursor(
            cfg,
            namespace,
            "home",
            observed,
            next_cursor,
            mutation_guard=require_authority,
        )

    assert load_ready_cursor(cfg, namespace, "home") == observed


def test_ready_cursor_compare_and_commit_serializes_divergent_writers(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1")
    namespace = "scheduler-primary-gpu"
    observed = load_ready_cursor(cfg, namespace, "home")
    first_next = ReadyCursor(namespace, cfg.machine_name, "home", 0, "00", "task-a.1.json", 1)
    second_next = ReadyCursor(namespace, cfg.machine_name, "home", 0, "00", "task-b.1.json", 1)
    first_at_guard = threading.Event()
    release_first = threading.Event()
    second_at_guard = threading.Event()
    outcomes: dict[str, str] = {}
    failures: list[BaseException] = []

    def run(name: str, next_cursor: ReadyCursor, guard) -> None:
        try:
            outcomes[name] = compare_and_commit_ready_cursor(
                cfg,
                namespace,
                "home",
                observed,
                next_cursor,
                mutation_guard=guard,
            )
        except BaseException as exc:
            failures.append(exc)

    def first_guard() -> None:
        first_at_guard.set()
        assert release_first.wait(5)

    first = threading.Thread(target=run, args=("first", first_next, first_guard))
    second = threading.Thread(target=run, args=("second", second_next, second_at_guard.set))
    first.start()
    assert first_at_guard.wait(5)
    second.start()
    assert not second_at_guard.wait(0.1)
    release_first.set()
    first.join(5)
    second.join(5)

    assert not first.is_alive() and not second.is_alive()
    assert failures == []
    assert outcomes == {"first": "committed", "second": "stale"}
    assert not second_at_guard.is_set()
    assert load_ready_cursor(cfg, namespace, "home") == first_next


def test_ready_cursor_commit_rejects_unknown_replace_outcome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1")
    namespace = "scheduler-primary-gpu"
    observed = load_ready_cursor(cfg, namespace, "home")
    next_cursor = ReadyCursor(namespace, cfg.machine_name, "home", 0, "00", "task-a.1.json", 1)

    def unknown_replace(_path, _value, *, before_replace=None, io_step_observer=None):
        del io_step_observer
        assert before_replace is not None
        before_replace(None)
        return None

    monkeypatch.setattr(ready_traversal, "atomic_replace", unknown_replace)

    with pytest.raises(OSError, match="durability"):
        compare_and_commit_ready_cursor(
            cfg,
            namespace,
            "home",
            observed,
            next_cursor,
            mutation_guard=lambda: None,
        )

    assert load_ready_cursor(cfg, namespace, "home") == observed


def test_cursor_commit_fence_after_first_route_is_unknown_and_replays(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    namespace = "scheduler-primary-gpu"
    initial = {"catalog_page": 0, "partition": None, "after_name": None, "revision": 0}
    cursor = {
        "namespace": namespace,
        "routes": {
            "home": {
                "observed": initial,
                "next": {"catalog_page": 0, "partition": "00", "after_name": "task-a.1.json", "revision": 1},
            },
            "shared": {
                "observed": initial,
                "next": {"catalog_page": 0, "partition": "00", "after_name": "task-b.1.json", "revision": 1},
            },
        },
    }
    request = executor.prepare_scheduler_cursor_commit(binding, revision, cursor=cursor)
    shared_lock = shared_paths(cfg.shared_root)["ready_locks"] / f"cursor.{namespace}.{cfg.machine_name}.shared.lock"

    with exclusive(shared_lock):
        executor.start(request.request_id)
        deadline = time.monotonic() + 5
        while load_ready_cursor(cfg, namespace, "home").revision != 1 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert load_ready_cursor(cfg, namespace, "home").revision == 1
        executor.fence_epoch()

    partial = _wait_for_result(executor, request.request_id)
    assert partial.status == "outcome_unknown"
    assert partial.reason_code == "project_io_scheduler_cursor_commit_failed"
    assert load_ready_cursor(cfg, namespace, "home").after_name == "task-a.1.json"
    assert load_ready_cursor(cfg, namespace, "shared").revision == 0

    deadline = time.monotonic() + 5
    while executor._record_path("processes", request.request_id).exists() and time.monotonic() < deadline:
        executor.poll()
        time.sleep(0.02)
    executor.begin_epoch()
    replay = executor.prepare_scheduler_cursor_commit(binding, revision, cursor=cursor)
    executor.start(replay.request_id)
    replayed = _wait_for_consumed(executor, replay)

    assert replayed.status == "completed"
    assert replayed.evidence["routes"] == {"home": "already_applied", "shared": "committed"}


def test_stale_cursor_resolution_retains_owned_live_child_without_process_record(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    initial = {"catalog_page": 0, "partition": None, "after_name": None, "revision": 0}
    request = executor.prepare_scheduler_cursor_commit(
        binding,
        revision,
        cursor={
            "namespace": "scheduler-project-a-primary-gpu",
            "routes": {
                "home": {"observed": initial, "next": initial},
                "shared": {"observed": initial, "next": initial},
            },
        },
    )

    class LiveChild:
        @staticmethod
        def poll():
            return None

    executor._children[request.request_id] = LiveChild()

    assert executor.resolve_stale_cursor_commit(request.request_id, request) is False
    assert executor._record_path("requests", request.request_id).exists()
    assert request.request_id in executor._children
    executor._children.clear()


def test_scheduler_observe_rejects_oversized_project_record_before_parsing(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    task = submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    namespace = "scheduler-primary-gpu"
    task_path = shared_paths(cfg.shared_root)["tasks"] / f"{task.task_id}.json"
    task_path.write_bytes(task_path.read_bytes() + b" " * (8 * 1024 * 1024 + 1))
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()

    request = executor.prepare_scheduler_observe(
        binding,
        revision,
        lane="gpu",
        admission_role="primary",
        cursor_namespace=namespace,
    )
    executor.start(request.request_id)
    result = _wait_for_result(executor, request.request_id)

    assert result.status == "completed"
    assert result.evidence["outcome"] == "none"
    assert result.evidence["reason"] == "candidate_unresolved"


def test_scheduler_claim_commits_shared_truth_without_touching_local_offer(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    task = submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor_epoch = executor.begin_epoch()
    observe = executor.prepare_scheduler_observe(
        binding,
        revision,
        lane="gpu",
        admission_role="primary",
        cursor_namespace="scheduler-primary-gpu",
    )
    executor.start(observe.request_id)
    observed = _wait_for_result(executor, observe.request_id)
    candidate = dict(observed.evidence["candidate"])
    cursor = dict(observed.evidence["cursor"])
    assert _wait_for_consumed(executor, observe) == observed
    request_id = uuid.uuid4().hex
    reservation = reserve(
        runtime.root,
        task.task_id,
        [0],
        attempt_id=candidate["attempt_id"],
        fencing_token=candidate["fencing_token"],
        project_id=binding.project_id,
        shared_root=str(binding.shared_root),
        machine_name=binding.machine_name,
        executor_epoch=executor_epoch,
        executor_request_id=request_id,
        registration_generation=binding.registration_generation,
    )["reservation"]
    offer = {
        "offer_id": reservation["reservation_id"],
        "acquisition_id": reservation["acquisition_id"],
        "reservation_id": reservation["reservation_id"],
        "executor_epoch": executor_epoch,
        "request_id": request_id,
        "project_id": binding.project_id,
        "shared_root": str(binding.shared_root),
        "registration_generation": binding.registration_generation,
        "task_id": task.task_id,
        "attempt_id": candidate["attempt_id"],
        "attempt_number": candidate["attempt_number"],
        "fencing_token": candidate["fencing_token"],
        "lane": "gpu",
        "gpu_ids": [0],
        "cpu_slots": 0,
        "group_name": None,
        "group_dispatch_epoch": None,
        "group_worker_set_epoch": None,
        "worker_state_epoch": None,
        "worker_scheduling_role": None,
        "gpu_limit_gpus": None,
        "admitted_as_borrow": False,
    }
    claim = executor.prepare_scheduler_claim(
        binding,
        revision,
        lane="gpu",
        admission_role="primary",
        candidate=candidate,
        offer=offer,
        cursor=cursor,
        request_id=request_id,
        source_revisions=observed.evidence["source_revisions"],
    )
    identity = ReservationIdentity.from_record(reservation)

    executor.start(claim.request_id)
    claimed = _wait_for_result(executor, claim.request_id)

    assert claimed.status == "completed"
    assert claimed.evidence["outcome"] == "claimed"
    assert claimed.evidence["cursor_routes"] is None
    assert claimed.evidence["attempt_id"] == candidate["attempt_id"]
    assert load_task(cfg, task.task_id).claim_control["active_claim"]["reservation_id"] == reservation["reservation_id"]
    assert classify_executor_offer(runtime.root, identity) == "matching_provisional"
    assert attach_executor_offer(runtime.root, identity, candidate["attempt_id"], candidate["fencing_token"])
    assert classify_executor_offer(runtime.root, identity) == "matching_active"
    assert _wait_for_consumed(executor, claim) == claimed


def test_scheduler_changed_task_retains_cursor_for_fresh_observation(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    task = submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor_epoch = executor.begin_epoch()
    namespace = "scheduler-primary-gpu"
    observe = executor.prepare_scheduler_observe(
        binding,
        revision,
        lane="gpu",
        admission_role="primary",
        cursor_namespace=namespace,
    )
    executor.start(observe.request_id)
    observed = _wait_for_consumed(executor, observe)
    candidate = dict(observed.evidence["candidate"])
    cursor = dict(observed.evidence["cursor"])
    request_id = uuid.uuid4().hex
    reservation = reserve(
        runtime.root,
        task.task_id,
        [0],
        attempt_id=candidate["attempt_id"],
        fencing_token=candidate["fencing_token"],
        project_id=binding.project_id,
        shared_root=str(binding.shared_root),
        machine_name=binding.machine_name,
        executor_epoch=executor_epoch,
        executor_request_id=request_id,
        registration_generation=binding.registration_generation,
    )["reservation"]
    offer = {
        "offer_id": reservation["reservation_id"],
        "acquisition_id": reservation["acquisition_id"],
        "reservation_id": reservation["reservation_id"],
        "executor_epoch": executor_epoch,
        "request_id": request_id,
        "project_id": binding.project_id,
        "shared_root": str(binding.shared_root),
        "registration_generation": binding.registration_generation,
        "task_id": task.task_id,
        "attempt_id": candidate["attempt_id"],
        "attempt_number": candidate["attempt_number"],
        "fencing_token": candidate["fencing_token"],
        "lane": "gpu",
        "gpu_ids": [0],
        "cpu_slots": 0,
        "group_name": None,
        "group_dispatch_epoch": None,
        "group_worker_set_epoch": None,
        "worker_state_epoch": None,
        "worker_scheduling_role": None,
        "gpu_limit_gpus": None,
        "admitted_as_borrow": False,
    }
    changed = load_task(cfg, task.task_id)
    changed.meta["revision"] += 1
    save_task(cfg, changed)
    claim = executor.prepare_scheduler_claim(
        binding,
        revision,
        lane="gpu",
        admission_role="primary",
        candidate=candidate,
        offer=offer,
        cursor=cursor,
        request_id=request_id,
        source_revisions=observed.evidence["source_revisions"],
    )
    identity = ReservationIdentity.from_record(reservation)

    executor.start(claim.request_id)
    result = _wait_for_consumed(executor, claim)

    assert result.status == "completed"
    assert result.evidence["outcome"] == "no_claim"
    assert result.evidence["reason"] == "task_changed"
    assert result.evidence["cursor_routes"] == {"home": "already_applied", "shared": "already_applied"}
    assert load_ready_cursor(cfg, namespace, "home").after_name == cursor["routes"]["home"]["observed"]["after_name"]
    assert classify_executor_offer(runtime.root, identity) == "matching_provisional"

    # Revision-only churn leaves this Task eligible. A subsequent observation
    # must still find it instead of silently abandoning it behind the cursor.
    refreshed = executor.prepare_scheduler_observe(
        binding,
        revision,
        lane="gpu",
        admission_role="primary",
        cursor_namespace=namespace,
    )
    executor.start(refreshed.request_id)
    current = _wait_for_consumed(executor, refreshed)
    assert current.evidence["candidate"]["task_id"] == task.task_id
    assert current.evidence["candidate"]["task_revision"] == changed.meta["revision"]


def test_scheduler_no_claim_binding_fence_makes_cursor_outcome_unknown(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    task = submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    executor = ProjectIOExecutor(runtime)
    executor_epoch = executor.begin_epoch()
    namespace = "scheduler-primary-gpu"
    observe = executor.prepare_scheduler_observe(
        binding,
        revision,
        lane="gpu",
        admission_role="primary",
        cursor_namespace=namespace,
    )
    executor.start(observe.request_id)
    observed = _wait_for_consumed(executor, observe)
    candidate = dict(observed.evidence["candidate"])
    cursor = dict(observed.evidence["cursor"])
    request_id = uuid.uuid4().hex
    reservation = reserve(
        runtime.root,
        task.task_id,
        [0],
        attempt_id=candidate["attempt_id"],
        fencing_token=candidate["fencing_token"],
        project_id=binding.project_id,
        shared_root=str(binding.shared_root),
        machine_name=binding.machine_name,
        executor_epoch=executor_epoch,
        executor_request_id=request_id,
        registration_generation=binding.registration_generation,
    )["reservation"]
    offer = {
        "offer_id": reservation["reservation_id"],
        "acquisition_id": reservation["acquisition_id"],
        "reservation_id": reservation["reservation_id"],
        "executor_epoch": executor_epoch,
        "request_id": request_id,
        "project_id": binding.project_id,
        "shared_root": str(binding.shared_root),
        "registration_generation": binding.registration_generation,
        "task_id": task.task_id,
        "attempt_id": candidate["attempt_id"],
        "attempt_number": candidate["attempt_number"],
        "fencing_token": candidate["fencing_token"],
        "lane": "gpu",
        "gpu_ids": [0],
        "cpu_slots": 0,
        "group_name": None,
        "group_dispatch_epoch": None,
        "group_worker_set_epoch": None,
        "worker_state_epoch": None,
        "worker_scheduling_role": None,
        "gpu_limit_gpus": None,
        "admitted_as_borrow": False,
    }
    changed = load_task(cfg, task.task_id)
    changed.meta["revision"] += 1
    save_task(cfg, changed)
    claim = executor.prepare_scheduler_claim(
        binding,
        revision,
        lane="gpu",
        admission_role="primary",
        candidate=candidate,
        offer=offer,
        cursor=cursor,
        request_id=request_id,
        source_revisions=observed.evidence["source_revisions"],
    )
    initial_cursor = load_ready_cursor(cfg, namespace, "home")
    runtime.set_enabled(binding.project_id, False)

    executor.start(claim.request_id)
    result = _wait_for_result(executor, claim.request_id)

    assert result.status == "outcome_unknown"
    assert result.reason_code == "project_io_outcome_unknown"
    assert load_ready_cursor(cfg, namespace, "home") == initial_cursor
    assert classify_executor_offer(runtime.root, ReservationIdentity.from_record(reservation)) == "matching_provisional"
    executor.shutdown()


def test_one_live_request_per_exact_binding_is_enforced(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    request = executor.prepare_validate_binding(binding, revision)

    with pytest.raises(RuntimeError, match="binding|request|occupied"):
        executor.prepare_validate_binding(binding, revision)

    executor.start(request.request_id)
    _wait_for_result(executor, request.request_id)
    _wait_for_consumed(executor, request)
    assert cfg.shared_root.exists()


def test_prepare_does_not_resolve_the_project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()

    def fail_resolve(*_args, **_kwargs):
        raise AssertionError("controller request preparation must not resolve the shared root")

    monkeypatch.setattr(Path, "resolve", fail_resolve)
    request = executor.prepare_validate_binding(binding, revision)

    assert request.canonical_shared_root == str(binding.shared_root)


def test_blocked_project_worker_does_not_block_healthy_peer_and_capacity_stays_bounded(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    blocked_cfg, blocked, revision = _registered(tmp_path, "blocked", runtime)
    _healthy_cfg, healthy, revision = _registered(tmp_path, "healthy", runtime)
    identity = blocked_cfg.shared_root / "project" / "identity.json"
    identity.unlink()
    os.mkfifo(identity)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()

    blocked_request = executor.prepare_validate_binding(blocked, revision)
    executor.start(blocked_request.request_id)
    healthy_request = executor.prepare_validate_binding(healthy, revision)
    executor.start(healthy_request.request_id)
    healthy_result = _wait_for_result(executor, healthy_request.request_id)

    assert healthy_result.status == "completed"
    assert _wait_for_consumed(executor, healthy_request) == healthy_result
    status = executor.status_view()
    assert status["active_worker_count"] == 1
    assert status["free_slot_count"] == 3
    assert status["blocking_project_ids"] == [blocked.project_id]

    executor.shutdown()
    stopped = executor.status_view()
    assert stopped["active_worker_count"] + stopped["exit_unverified_worker_count"] <= 1


def test_four_blocked_workers_prevent_a_fifth_request(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    bindings = []
    revision = 0
    for index in range(5):
        cfg, binding, revision = _registered(tmp_path, f"project-{index}", runtime)
        identity = cfg.shared_root / "project" / "identity.json"
        identity.unlink()
        os.mkfifo(identity)
        bindings.append(binding)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    requests = []
    for binding in bindings[:4]:
        request = executor.prepare_validate_binding(binding, revision)
        executor.start(request.request_id)
        requests.append(request)

    with pytest.raises(RuntimeError, match="capacity|occupied"):
        executor.prepare_validate_binding(bindings[4], revision)

    status = executor.status_view()
    assert status["active_worker_count"] == 4
    assert status["free_slot_count"] == 0
    assert len(status["blocking_project_ids"]) == 4
    rss_kib = []
    descriptor_counts = []
    for child in executor._children.values():
        process_root = Path("/proc") / str(child.pid)
        status_lines = (process_root / "status").read_text(encoding="ascii").splitlines()
        rss_line = next(line for line in status_lines if line.startswith("VmRSS:"))
        rss_kib.append(int(rss_line.split()[1]))
        descriptor_counts.append(len(tuple((process_root / "fd").iterdir())))
    assert len(rss_kib) == 4
    assert all(value <= 256 * 1024 for value in rss_kib)
    assert sum(rss_kib) <= 1024 * 1024
    # Fresh workers inherit only the executor's closed stdio and their blocked
    # shared-root handle; this guards against descriptor growth at capacity.
    assert all(value <= 16 for value in descriptor_counts)
    executor.shutdown()


def test_new_epoch_fences_a_late_result_before_consumption(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    first_epoch = executor.begin_epoch()
    request = executor.prepare_validate_binding(binding, revision)
    executor.start(request.request_id)
    assert _wait_for_result(executor, request.request_id).status == "completed"

    second_epoch = executor.begin_epoch()

    assert second_epoch != first_epoch
    assert executor.consume(request.request_id, request) is None
    assert executor.status_view()["executor_epoch"] == second_epoch


def test_live_worker_process_evidence_survives_published_result(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _registered(tmp_path, "blocked", runtime)
    identity = cfg.shared_root / "project" / "identity.json"
    identity.unlink()
    os.mkfifo(identity)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    request = executor.prepare_validate_binding(binding, revision)
    executor.start(request.request_id)
    result = ProjectIOResult(
        request=request,
        status="retryable_error",
        reason_code="project_io_binding_validation_failed",
        completed_at="2026-09-28T00:00:01Z",
        evidence={"exception_type": "SyntheticFailure"},
    )
    executor._write_record(
        executor._record_path("results", request.request_id),
        result.to_dict(),
        "project_io_result",
    )

    assert executor.consume(request.request_id, request) is None
    for lane in ("requests", "processes", "results"):
        assert executor._record_path(lane, request.request_id).exists()

    executor.shutdown()


def test_archived_resolution_replays_transient_cleanup_after_crash_window(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    request = executor.prepare_validate_binding(binding, revision)
    executor.start(request.request_id)
    result = _wait_for_result(executor, request.request_id)
    deadline = time.monotonic() + 5.0
    while executor._record_path("processes", request.request_id).exists() and time.monotonic() < deadline:
        executor.poll()
        time.sleep(0.02)
    assert not executor._record_path("processes", request.request_id).exists()
    with exclusive(executor.paths["project_io_lock"]):
        executor._commit_resolution(request.request_id, "consumed", request, result)

    executor.poll()
    assert executor._record_path("requests", request.request_id).exists()
    assert executor._record_path("results", request.request_id).exists()
    assert executor.consume(request.request_id, request) == result
    assert not executor._record_path("requests", request.request_id).exists()
    assert not executor._record_path("results", request.request_id).exists()


def test_restart_replays_archived_resolution_without_retaining_capacity(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    request = executor.prepare_validate_binding(binding, revision)
    executor.start(request.request_id)
    result = _wait_for_result(executor, request.request_id)
    deadline = time.monotonic() + 5.0
    while executor._record_path("processes", request.request_id).exists() and time.monotonic() < deadline:
        executor.poll()
        time.sleep(0.02)
    with exclusive(executor.paths["project_io_lock"]):
        executor._commit_resolution(request.request_id, "consumed", request, result)

    restarted = ProjectIOExecutor(runtime)
    restarted.begin_epoch()

    assert restarted.status_view()["free_slot_count"] == 4
    assert not restarted._record_path("requests", request.request_id).exists()
    assert not restarted._record_path("results", request.request_id).exists()


def test_large_snapshot_result_archives_without_repeating_request_payload(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision = _registered(tmp_path, "project-a", runtime)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    summaries = [
        {
            "reservation_id": uuid.uuid4().hex,
            "project_id": binding.project_id,
            "group_name": None,
            "machine_name": "gpu-1",
            "task_id": f"task-{index}",
            "attempt_id": f"task-{index}-attempt-1",
            "gpu_ids": [index],
            "state": "active",
            "admission": None,
        }
        for index in range(129)
    ]
    request = executor.prepare_machine_snapshot_publish(
        binding,
        revision,
        instance_id="agent-1",
        pid=123,
        visible_gpu_ids=list(range(129)),
        reserved_gpu_ids=list(range(129)),
        reservation_summaries=summaries,
        heartbeat_interval_seconds=5.0,
        started_at="2026-09-28T00:00:00+00:00",
        gpu_policy={"mode": "all", "warnings": []},
    )
    result = ProjectIOResult(
        request=request,
        status="completed",
        reason_code=None,
        completed_at="2026-09-28T00:00:01+00:00",
        evidence={
            "outcome": "published",
            "snapshot_at": request.parameters["snapshot_at"],
            "reason": None,
        },
    )
    executor._write_record(
        executor._record_path("results", request.request_id),
        result.to_dict(),
        "project_io_result",
    )

    assert executor.consume(request.request_id, request) == result
    resolution = next(executor.paths["project_io_resolved"].iterdir())
    assert resolution.stat().st_size <= 65_536
    assert executor._find_resolution(request.request_id, request) == ("consumed", result)


def test_restart_archives_unconsumed_old_epoch_requests_as_stale(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    bindings = []
    revision = 0
    for index in range(4):
        _cfg, binding, revision = _registered(tmp_path, f"project-{index}", runtime)
        bindings.append(binding)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    requests = [executor.prepare_validate_binding(binding, revision) for binding in bindings]
    assert executor.status_view()["free_slot_count"] == 0

    restarted = ProjectIOExecutor(runtime)
    restarted.begin_epoch()

    assert restarted.status_view()["free_slot_count"] == 4
    for request in requests:
        assert not restarted._record_path("requests", request.request_id).exists()


def test_unverified_worker_retains_stop_target_and_unreaped_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _registered(tmp_path, "blocked", runtime)
    identity = cfg.shared_root / "project" / "identity.json"
    identity.unlink()
    os.mkfifo(identity)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    request = executor.prepare_validate_binding(binding, revision)
    executor.start(request.request_id)
    assert executor._mark_live_workers_stop_targeted()

    with monkeypatch.context() as patch:
        patch.setattr(project_io_executor_module, "inspect_process_identity", lambda *_args: "unverified")
        status = executor.poll()
        process = executor._load_process(request.request_id)

    assert process.state == "stop_targeted"
    assert status["envelope"] == "unknown"
    assert status["exit_unverified_worker_count"] == 1
    assert status["unreaped_worker_count"] == 1
    executor.shutdown()


def test_agent_status_reads_executor_projection_without_project_io(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    executor = ProjectIOExecutor(runtime)
    epoch = executor.begin_epoch()

    status = get_machine_agent_status(runtime)

    assert status["project_io_isolation"] == {
        "protocol_version": 1,
        "executor_epoch": epoch,
        "capacity": 4,
        "active_worker_count": 0,
        "overdue_worker_count": 0,
        "exit_unverified_worker_count": 0,
        "unreaped_worker_count": 0,
        "free_slot_count": 4,
        "supported_hang_limit": 2,
        "envelope": "healthy",
        "oldest_overdue_at": None,
        "blocking_project_ids": [],
    }


def test_empty_executor_shutdown_does_not_spend_worker_grace_periods(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()

    grace_calls: list[float] = []
    with monkeypatch.context() as patch:
        patch.setattr(executor, "_grace_period", grace_calls.append)
        status = executor.shutdown()

    assert grace_calls == []
    assert status["active_worker_count"] == 0
    assert status["envelope"] == "degraded"


def test_unknown_executor_shutdown_does_not_infer_worker_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()

    grace_calls: list[float] = []
    with monkeypatch.context() as patch:
        patch.setattr(executor, "poll", executor._unknown_status)
        patch.setattr(executor, "_grace_period", grace_calls.append)
        status = executor.shutdown()

    assert grace_calls == [2.0]
    assert status["envelope"] == "degraded"
