"""Source retention is fenced worker I/O; target ownership and census stay local."""

import os
import time

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent import project_io_worker as worker
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.project_io_controller import ProjectIOController
from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor
from qqtools.plugins.qexp.agent.project_io_protocol import ProjectIOProcess
from qqtools.plugins.qexp.agent.recovery_transport import decode_source_hold
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.runtime import responsibility_capture as capture
from qqtools.plugins.qexp.runtime.responsibility import responsibility_root
from qqtools.plugins.qexp.runtime.responsibility_store import Ledger
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


@pytest.fixture
def case(tmp_path):
    source = tmp_path / "foreign/source"
    source.mkdir(parents=True)
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg = init_shared_root(tmp_path / "project/.qexp", "gpu-1", runtime_root=source)
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    root = runtime.project_paths(binding.project_id)["root"]
    root.mkdir(parents=True, exist_ok=True)
    atomic_replace(
        root / "migration.json",
        {
            "migration": {
                "state": "active",
                "legacy_runtime_root": str(source),
                "project_id": binding.project_id,
                "shared_root": str(binding.shared_root),
                "machine_name": binding.machine_name,
            }
        },
    )
    ledger = Ledger.open_or_create(responsibility_root(root))
    checkpoint = capture.WriterCaptureCheckpoint(ledger, root, legacy_source=source)
    state = checkpoint.prepare_local()
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    try:
        yield runtime, binding, source, checkpoint, state, executor
    finally:
        executor.shutdown()


def consume(executor, prepared):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        executor.poll()
        result = executor.consume(prepared.request_id, prepared)
        if result is not None:
            return result
        time.sleep(0.01)
    pytest.fail("source hold worker did not return while target ownership was retained")


@pytest.mark.parametrize("disabled", [False, True])
def test_source_hold_worker_owns_source_only_and_exact_result_enables_local_scope(case, disabled):
    runtime, binding, source, checkpoint, state, executor = case
    if disabled:
        binding = runtime.set_enabled(binding.project_id, False)
    revision, _ = runtime.load_registry_snapshot()
    before = {
        str(path.relative_to(checkpoint.runtime_root)): path.read_bytes()
        for path in checkpoint.runtime_root.rglob("*")
        if path.is_file()
    }
    request = executor.prepare_recovery_source_hold(binding, revision, capture_id=state["capture_id"])
    with runtime.registry_guard(), runtime.binding_commit_guard(binding):
        assert executor.start(request.request_id) is not None
        result = consume(executor, request)
    assert result.status == "completed" and result.evidence["state"] == "retained"
    hold = decode_source_hold(result.evidence["hold"], state["capture_id"])
    assert hold == capture.source_hold_for_capture(state)
    assert read_json(source / capture.CAPTURE_FILE) == hold.to_dict()
    after = {
        str(path.relative_to(checkpoint.runtime_root)): path.read_bytes()
        for path in checkpoint.runtime_root.rglob("*")
        if path.is_file()
    }
    assert after == before
    with checkpoint.observe_local(hold):
        pass
    replay = executor.prepare_recovery_source_hold(binding, revision, capture_id=state["capture_id"])
    assert executor.start(replay.request_id) is not None
    assert consume(executor, replay).evidence == result.evidence


def test_common_admission_consumes_source_retention_without_inline_domain_io(case, monkeypatch):
    runtime, binding, source, checkpoint, state, executor = case
    revision, _ = runtime.load_registry_snapshot()
    controller = ProjectIOController(runtime, executor)
    monkeypatch.setattr(
        worker, "_recovery_source_hold", lambda *_: pytest.fail("controller ran source transaction inline")
    )
    deadline = time.monotonic() + 5
    completed = {}
    while time.monotonic() < deadline:
        completed = controller.advance_recovery_source_holds(
            [binding], revision, {binding.project_id: {"capture_id": state["capture_id"]}}
        )
        if binding.project_id in completed:
            break
        time.sleep(0.01)
    hold = decode_source_hold(completed[binding.project_id]["hold"], state["capture_id"])
    assert read_json(source / capture.CAPTURE_FILE) == hold.to_dict()
    assert checkpoint.prepare_local() == state
    assert not executor.unresolved_requests()


@pytest.mark.parametrize("change", ["generation", "owner", "malformed"])
def test_pending_retention_can_replay_for_same_owner_not_foreign_or_malformed_admission(case, change):
    from qqtools.plugins.qexp.agent.recovery_capture import recovery_owner
    from qqtools.plugins.qexp.runtime.responsibility_process_capture import RunnerProcessCapture

    runtime, binding, source, checkpoint, state, executor = case
    owner = recovery_owner(runtime, binding)
    old_owner = {**owner, "registration_generation": "previous-generation"}
    old_hold = capture.retain_capture_source(state)
    from dataclasses import replace

    runner = RunnerProcessCapture(
        replace(binding.root_config(), runtime_root=checkpoint.runtime_root),
        checkpoint.ledger,
        source_hold=old_hold,
    )
    runner.prepare_admission(old_owner)
    admitted = read_json(checkpoint.path)
    if change == "owner":
        admitted["admission"]["owner_instance"] = "foreign-instance"
    elif change == "malformed":
        admitted["admission"]["unexpected"] = True
    atomic_replace(checkpoint.path, admitted)
    before = checkpoint.path.read_bytes()
    revision, _ = runtime.load_registry_snapshot()
    request = executor.prepare_recovery_source_hold(binding, revision, capture_id=state["capture_id"])
    try:
        assert executor.start(request.request_id) is not None
        result = consume(executor, request)
        assert checkpoint.path.read_bytes() == before
        if change != "generation":
            assert dict(result.evidence) == {"state": "stale", "hold": None}
            return
        assert result.status == "completed" and result.evidence["state"] == "retained"
        runner.prepare_admission(owner)
        refreshed = read_json(checkpoint.path)
        assert refreshed["admission"] == capture.capture_admission(owner)
        assert refreshed["progress"]["revision"] == admitted["progress"]["revision"] + 1
        assert read_json(source / capture.CAPTURE_FILE) == old_hold.to_dict()
    finally:
        runner.close()


@pytest.mark.parametrize("change", ["capture", "ledger", "migration", "complete", "generation"])
def test_stale_capture_does_not_establish_source_hold(case, change):
    runtime, binding, source, checkpoint, state, executor = case
    revision, _ = runtime.load_registry_snapshot()
    request = executor.prepare_recovery_source_hold(binding, revision, capture_id=state["capture_id"])
    if change == "capture":
        atomic_replace(checkpoint.path, {**state, "capture_id": "f" * 32})
    elif change == "ledger":
        marker_path = responsibility_root(checkpoint.runtime_root) / "marker"
        atomic_replace(marker_path, {**read_json(marker_path), "instance": "f" * 32})
    elif change == "migration":
        atomic_replace(runtime.migration_path(binding.project_id), {"migration": {"state": "active"}})
    elif change == "complete":
        from qqtools.plugins.qexp.runtime.responsibility_completion import COMPLETION_FILE

        atomic_replace(checkpoint.runtime_root / COMPLETION_FILE, {"unavailable": True})
    else:
        atomic_replace(checkpoint.runtime_root / capture.GENERATION_FILE, {"unavailable": True})
    assert executor.start(request.request_id) is not None
    result = consume(executor, request)
    assert result.status == "completed"
    assert dict(result.evidence) == {"state": "stale", "hold": None}
    assert not (source / capture.CAPTURE_FILE).exists()


def test_lost_source_hold_result_requires_positive_exit_then_exact_replay(case, monkeypatch):
    from qqtools.plugins.qexp.agent import project_io_executor as executor_module

    runtime, binding, source, checkpoint, state, executor = case
    revision, _ = runtime.load_registry_snapshot()
    request = executor.prepare_recovery_source_hold(binding, revision, capture_id=state["capture_id"])
    hold = capture.retain_capture_source(state)
    before = (source / capture.CAPTURE_FILE).read_bytes()
    process = ProjectIOProcess(request, 2_000_000_000, 1, request.prepared_at, "running")
    atomic_replace(executor.paths["project_io_processes"] / f"{request.request_id}.json", process.to_dict())
    with monkeypatch.context() as guarded:
        guarded.setattr(executor_module, "inspect_process_identity", lambda *_: "unverified")
        assert executor.poll()["exit_unverified_worker_count"] == 1
        assert executor.load_result(request.request_id) is None
        assert not executor.reset_ambiguous_recovery_source_hold_for_retry(request.request_id, request)
    executor.poll()
    result = executor.load_result(request.request_id)
    assert result is not None and result.status == "outcome_unknown"
    assert executor.consume(request.request_id, request) is None
    assert executor.reset_ambiguous_recovery_source_hold_for_retry(request.request_id, request)
    assert executor.start(request.request_id) is not None
    replayed = consume(executor, request)
    assert replayed.status == "completed"
    assert decode_source_hold(replayed.evidence["hold"], state["capture_id"]) == hold
    assert (source / capture.CAPTURE_FILE).read_bytes() == before
    assert checkpoint.prepare_local() == state


def test_source_alias_to_target_cannot_turn_retention_into_target_worker_write(case):
    runtime, binding, source, checkpoint, state, executor = case
    alias = source.parent / "alias"
    alias.symlink_to(checkpoint.runtime_root, target_is_directory=True)
    changed = {**state, "legacy_source": str(alias)}
    atomic_replace(checkpoint.path, changed)
    migration_path = runtime.migration_path(binding.project_id)
    migration = read_json(migration_path)["migration"]
    atomic_replace(migration_path, {"migration": {**migration, "legacy_runtime_root": str(alias)}})
    before = checkpoint.path.read_bytes()
    revision, _ = runtime.load_registry_snapshot()
    request = executor.prepare_recovery_source_hold(binding, revision, capture_id=state["capture_id"])
    assert executor.start(request.request_id) is not None
    result = consume(executor, request)
    assert result.status == "retryable_error"
    assert result.reason_code == "project_io_recovery_source_hold_failed"
    assert checkpoint.path.read_bytes() == before
    assert not (source / capture.CAPTURE_FILE).exists()


@pytest.mark.parametrize("failure", ["exception", "epoch", "capture_transition"])
def test_failure_after_source_retention_is_unknown_not_retryable_absence(case, monkeypatch, failure):
    runtime, binding, source, checkpoint, state, executor = case
    revision, _ = runtime.load_registry_snapshot()
    request = executor.prepare_recovery_source_hold(binding, revision, capture_id=state["capture_id"])
    ticks = worker._process_start_time_ticks(os.getpid())
    assert ticks is not None
    process = ProjectIOProcess(request, os.getpid(), ticks, request.prepared_at, "running")
    process_path = executor.paths["project_io_processes"] / f"{request.request_id}.json"
    atomic_replace(process_path, process.to_dict())
    original = capture.retain_capture_source

    def interrupted(captured, **kwargs):
        hold = original(captured, **kwargs)
        if failure == "exception":
            raise OSError("source hold result lost after publication")
        if failure == "epoch":
            executor.fence_epoch()
        else:
            atomic_replace(checkpoint.runtime_root / capture.GENERATION_FILE, {"unavailable": True})
        return hold

    monkeypatch.setattr(capture, "retain_capture_source", interrupted)
    try:
        assert worker._run(runtime.root, request.request_id) == 0
        result = executor.load_result(request.request_id)
        assert result is not None and result.status == "outcome_unknown"
        assert result.reason_code == "project_io_outcome_unknown"
        assert not result.evidence
        assert read_json(source / capture.CAPTURE_FILE) == capture.source_hold_for_capture(state).to_dict()
        assert read_json(checkpoint.path) == state
    finally:
        # This fixture's in-process worker record names pytest itself; never
        # leave it for lifecycle shutdown's exact-process signaling.
        process_path.unlink(missing_ok=True)
