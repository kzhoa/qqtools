"""Fresh-worker discovery and crash-safe local consumption of retained evidence."""

import os
import time
from dataclasses import replace
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.project_io_controller import ProjectIOController
from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor
from qqtools.plugins.qexp.agent.project_io_protocol import ProjectIOProcess
from qqtools.plugins.qexp.agent.recovery_capture import recovery_owner
from qqtools.plugins.qexp.agent.recovery_transport import decode_capture_locator
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.config_types import RootConfig
from qqtools.plugins.qexp.runtime import group_namespace
from qqtools.plugins.qexp.runtime import responsibility_completion as completion
from qqtools.plugins.qexp.runtime import responsibility_generation as generation
from qqtools.plugins.qexp.runtime import responsibility_process_capture as processes
from qqtools.plugins.qexp.runtime.paths import local_paths
from qqtools.plugins.qexp.runtime.responsibility import responsibility_root
from qqtools.plugins.qexp.runtime.responsibility_backfill import LANES, ResponsibilityBackfill
from qqtools.plugins.qexp.runtime.responsibility_capture import WriterCaptureCheckpoint, retain_capture_source
from qqtools.plugins.qexp.runtime.responsibility_import import RECORD_KEYS
from qqtools.plugins.qexp.runtime.responsibility_source_read import (
    load_source_read_context,
    read_retained_source_locator,
)
from qqtools.plugins.qexp.runtime.responsibility_source_scan import scan_retained_source
from qqtools.plugins.qexp.runtime.responsibility_store import Ledger, Unavailable
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


@pytest.mark.parametrize("guard_kind", ["registry", "binding"])
def test_recovery_local_application_defers_instead_of_waiting_for_machine_guard(case, guard_kind):
    from qqtools.plugins.qexp.agent.recovery_enrollment import RecoveryEnrollment

    runtime, binding, _source, target, *_middle, executor = case
    before = (target / completion.CAPTURE_FILE).read_bytes()
    controller = ProjectIOController(runtime, executor)
    enrollment = RecoveryEnrollment(runtime)
    revision, bindings = runtime.load_registry_snapshot()
    guard = runtime.registry_guard() if guard_kind == "registry" else runtime.binding_commit_guard(binding)
    try:
        with runtime.scheduler_authority() as acquired:
            assert acquired
            with guard as acquired:
                assert acquired
                started = time.monotonic()
                with controller.admission_turn():
                    enrollment.advance(controller, bindings, revision)
                assert time.monotonic() - started < 1.0
            assert (target / completion.CAPTURE_FILE).read_bytes() == before
            assert runtime.recovery_enrollment_pending_projects == {binding.project_id}
    finally:
        enrollment.stop()


@pytest.mark.parametrize("should_restart", [False, True])
@pytest.mark.parametrize("has_source_record", [False, True])
def test_production_enrollment_completes_source_capture_and_release_without_inline_shared_io(
    case, monkeypatch, should_restart, has_source_record
):
    from qqtools.plugins.qexp.agent.recovery_enrollment import RecoveryEnrollment
    from qqtools.plugins.qexp.runtime.responsibility_capture import CAPTURE_FILE

    runtime, binding, source, target, _scanner, _capture, _ledger, executor = case
    if has_source_record:
        atomic_replace(
            local_paths(source)["observations"] / "task-1-attempt-1.json",
            {"exit_observation": {"attempt_id": "task-1-attempt-1"}},
        )
    controller = ProjectIOController(runtime, executor)
    enrollment = RecoveryEnrollment(runtime)
    # Workers are fresh interpreters; prohibit these legacy parent-process paths.
    monkeypatch.setattr(
        runtime, "prepare_recovery_registration", lambda *_args, **_kwargs: pytest.fail("inline recovery registration")
    )
    monkeypatch.setattr(binding.__class__, "root_config", lambda *_args: pytest.fail("inline shared configuration"))
    operations = set()
    original_start = executor.start

    def started(request_id, **kwargs):
        operations.add(executor._load_request(request_id).operation_kind)
        return original_start(request_id, **kwargs)

    monkeypatch.setattr(executor, "start", started)
    deadline = time.monotonic() + 20
    has_restarted = False
    try:
        with runtime.scheduler_authority() as acquired:
            assert acquired
            while time.monotonic() < deadline:
                revision, bindings = runtime.load_registry_snapshot()
                with monkeypatch.context() as guarded, controller.admission_turn():
                    forbid_source_io(guarded, source)
                    forbid_source_io(guarded, binding.shared_root)
                    enrollment.advance(controller, bindings, revision)
                if binding in enrollment._settled:
                    break
                if has_source_record and completion.read_capture_completion(target, should_sync=False) is not None:
                    if group_namespace.is_group_authority_isolated(binding.shared_root):
                        break
                if should_restart and not has_restarted and "legacy_capture_scan" in operations:
                    enrollment.stop()
                    enrollment = RecoveryEnrollment(runtime)
                    has_restarted = True
                time.sleep(0.01)
            if not has_source_record:
                assert binding in enrollment._settled, {
                    entry.operation: entry.parameters for entry in enrollment._captures.values()
                }
        assert completion.read_capture_completion(target)["registration_generation"] == binding.registration_generation
        assert group_namespace.is_group_authority_isolated(binding.shared_root)
        assert (source / CAPTURE_FILE).exists() is has_source_record
        assert {
            "recovery_admission",
            "recovery_source_hold",
            "legacy_capture_scan",
            "recovery_group_authority",
        } <= operations
        if has_source_record:
            assert "legacy_capture_read" in operations
            assert "recovery_source_release" not in operations
            assert _ledger.has_members()
            assert runtime.recovery_enrollment_pending_projects == {binding.project_id}
        else:
            assert "recovery_source_release" in operations
            assert not runtime.recovery_enrollment_pending_projects
    finally:
        enrollment.stop()


@pytest.fixture
def case(tmp_path, monkeypatch):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    source = tmp_path / "source"
    cfg = init_shared_root(tmp_path / "project/.qexp", "gpu-1", runtime_root=source)
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    atomic_replace(
        runtime.migration_path(binding.project_id),
        {
            "migration": {
                "state": "active",
                "project_id": binding.project_id,
                "shared_root": str(binding.shared_root),
                "machine_name": binding.machine_name,
                "legacy_runtime_root": str(source),
            }
        },
    )
    target = runtime.project_paths(binding.project_id)["root"]
    target.mkdir(parents=True, exist_ok=True)
    ledger = Ledger.open_or_create(responsibility_root(target))
    checkpoint = WriterCaptureCheckpoint(ledger, target, legacy_source=source)
    hold = retain_capture_source(checkpoint.prepare_local())
    proc = tmp_path / "proc"
    (proc / "sys/kernel/random").mkdir(parents=True)
    (proc / "sys/kernel/random/boot_id").write_text(Path("/proc/sys/kernel/random/boot_id").read_text())
    (proc / "self/ns").mkdir(parents=True)
    (proc / "self/ns/pid").symlink_to("/proc/self/ns/pid")
    monkeypatch.setattr(processes, "PROC_ROOT", proc)
    local_cfg = RootConfig.from_canonical_paths(cfg.shared_root, cfg.project_root, cfg.machine_name, target)
    scanner = processes.RunnerProcessCapture(local_cfg, ledger, source_hold=hold)
    scanner.prepare_admission(recovery_owner(runtime, binding))
    assert scanner.take().is_sweep_complete
    capture = ResponsibilityBackfill(target, process_capture=scanner)
    assert capture.take(64, should_cross_lanes=True, allow_source=False).completed_lanes == len(LANES)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    try:
        yield runtime, binding, source, target, scanner, capture, ledger, executor
    finally:
        capture.close()
        scanner.close()
        executor.shutdown()


def source_only_context(case, parameters):
    runtime, binding, _source, target, *_rest = case
    return load_source_read_context(target, recovery_owner(runtime, binding), **parameters)


def forbid_source_io(monkeypatch, source):
    for obj, name in ((Path, "resolve"), (Path, "open"), (os, "open"), (os, "stat"), (os, "lstat"), (os, "scandir")):
        original = getattr(obj, name)

        def checked(path, *args, _original=original, **kwargs):
            if not isinstance(path, int):
                parsed = Path(os.fsdecode(path))
                if parsed == source or source in parsed.parents:
                    pytest.fail("local journal/capture touched source storage")
            return _original(path, *args, **kwargs)

        monkeypatch.setattr(obj, name, checked)


@pytest.mark.parametrize("lane", ["observations", "termination_decisions"])
def test_resumed_source_scan_and_journal_replay_finish_beyond_one_batch(case, monkeypatch, lane):
    runtime, binding, source, target, scanner, capture, ledger, _executor = case
    for number in range(65):
        identity = f"task-{number}-attempt-1"
        relative = f"{identity}/decision.json" if lane == "termination_decisions" else f"{identity}.json"
        key = "termination_decision" if lane == "termination_decisions" else "exit_observation"
        atomic_replace(
            local_paths(source)[lane] / relative, {key: {"attempt_id": identity, "env": {"SECRET": "hidden"}}}
        )
    scanned = set()
    for _ in range(400):
        work = capture.next_source_work()
        if work is None:
            break
        parameters = work["parameters"]
        if work["operation"] == "scan":
            context = source_only_context(
                case, {**{key: value for key, value in parameters.items() if key != "cursor"}, "relative": None}
            )
            result = scan_retained_source(context, parameters["lane"], parameters["cursor"], limit=7)
            assert result["entries_visited"] <= 7
            before = capture.path.read_bytes()
            # Simulate discovery result loss; identical durable input reproduces
            # the batch without advancing any target state.
            assert scan_retained_source(context, parameters["lane"], parameters["cursor"], limit=7) == result
            assert capture.path.read_bytes() == before
            with monkeypatch.context() as guarded:
                forbid_source_io(guarded, source)
                assert capture.apply_source_scan(parameters, result)
        else:
            context = source_only_context(case, parameters)
            captured = read_retained_source_locator(context, parameters["lane"], parameters["relative"])
            # Crash after local ledger publication but before pending retirement:
            # recover the same identity, not an empty/committed assumption.
            from qqtools.plugins.qexp.runtime.responsibility_backfill import apply_capture_locator

            with monkeypatch.context() as guarded:
                forbid_source_io(guarded, source)
                apply_capture_locator(ledger, target, captured)
                assert capture.apply_source_record(parameters, captured)
                assert not capture.apply_source_record(parameters, captured)
            scanned.add(captured.identity)
        capture.close()
        capture = ResponsibilityBackfill(target, process_capture=scanner)
    else:
        pytest.fail("retained source capture did not converge")
    assert scanned == {f"task-{number}-attempt-1" for number in range(65)}
    state = read_json(capture.path)
    assert state["lane"] == 2 * len(LANES) and not state["pending"]
    assert "source_cursor" not in state
    assert all(ledger.lookup(identity)["legacy_source"] == str(source) for identity in scanned)
    with monkeypatch.context() as guarded:
        forbid_source_io(guarded, source)
        proof = completion.publish_capture_completion(capture, owner=recovery_owner(runtime, binding))
        assert completion.read_capture_completion(target) == proof
        assert completion.prepare_completed_source_release(target) is None
    assert not (target / completion.SOURCE_RELEASE_FILE).exists()
    capture.close()


@pytest.mark.parametrize("change", ["none", "disabled", "cursor", "common_arbiter"])
def test_actual_scan_worker_has_no_machine_guard_or_local_write(case, change):
    runtime, binding, source, target, _scanner, capture, ledger, executor = case
    atomic_replace(
        local_paths(source)[LANES[0]] / "task-attempt-1.json",
        {RECORD_KEYS[LANES[0]]: {"attempt_id": "task-attempt-1"}},
    )
    work = capture.next_source_work()
    parameters = work["parameters"]
    if change == "disabled":
        binding = runtime.set_enabled(binding.project_id, False)
    if change == "cursor":
        parameters = {**parameters, "cursor": {**parameters["cursor"], "directory_identity": [1, 1]}}
    revision, _ = runtime.load_registry_snapshot()
    before = {str(path.relative_to(target)): path.read_bytes() for path in target.rglob("*") if path.is_file()}
    deadline = time.monotonic() + 5
    result = None
    if change == "common_arbiter":
        controller = ProjectIOController(runtime, executor)
        while time.monotonic() < deadline:
            results = controller.advance_legacy_capture_scans([binding], revision, {binding.project_id: parameters})
            if binding.project_id in results:
                evidence = results[binding.project_id]
                break
            time.sleep(0.01)
        else:
            pytest.fail("common arbiter did not complete retained source discovery")
    else:
        request = executor.prepare_legacy_capture_scan(binding, revision, **parameters)
        with runtime.registry_guard(), runtime.binding_commit_guard(binding), runtime.migration_guard():
            assert executor.start(request.request_id) is not None
            while time.monotonic() < deadline:
                executor.poll()
                result = executor.consume(request.request_id, request)
                if result is not None:
                    break
                time.sleep(0.01)
        assert result is not None and result.status == "completed"
        evidence = result.evidence
    after = {str(path.relative_to(target)): path.read_bytes() for path in target.rglob("*") if path.is_file()}
    assert after == before
    with pytest.raises(Unavailable, match="No such file"):
        ledger.lookup("task-attempt-1")
    if change == "cursor":
        assert dict(evidence) == {"state": "stale", "scan": None}
    else:
        assert evidence["state"] == "observed"
        assert list(evidence["scan"]["pending"]) == ["task-attempt-1.json"]
        assert evidence["scan"]["at_end"]
        assert capture.apply_source_scan(parameters, dict(evidence["scan"]))
        read_work = capture.next_source_work()
        read_request = executor.prepare_legacy_capture_read(binding, revision, **read_work["parameters"])
        assert executor.start(read_request.request_id) is not None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            executor.poll()
            read_result = executor.consume(read_request.request_id, read_request)
            if read_result is not None:
                break
            time.sleep(0.01)
        else:
            pytest.fail("journaled source record read did not finish")
        assert read_result.status == "completed" and read_result.evidence["state"] == "observed"
        captured = decode_capture_locator(read_result.evidence["locator"], read_request.parameters)
        assert capture.apply_source_record(read_work["parameters"], captured)
        assert ledger.lookup("task-attempt-1")["legacy_source"] == str(source)


def test_successive_fresh_workers_resume_exact_durable_directory_cookie(case):
    runtime, binding, source, _target, _scanner, capture, _ledger, executor = case
    expected = {f"task-{number}-attempt-1.json" for number in range(65)}
    for name in expected:
        atomic_replace(local_paths(source)[LANES[0]] / name, {RECORD_KEYS[LANES[0]]: {"attempt_id": name[:-5]}})
    revision, _ = runtime.load_registry_snapshot()
    seen = set()
    cookies = []
    for _ in range(3):
        work = capture.next_source_work()
        parameters = work["parameters"]
        assert work["operation"] == "scan"
        request = executor.prepare_legacy_capture_scan(binding, revision, **parameters)
        assert executor.start(request.request_id) is not None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            executor.poll()
            result = executor.consume(request.request_id, request)
            if result is not None:
                break
            time.sleep(0.01)
        else:
            pytest.fail("source directory continuation worker did not finish")
        assert result.status == "completed" and result.evidence["state"] == "observed"
        scan = result.evidence["scan"]
        assert scan["entries_visited"] <= 64
        cookies.append(scan["cursor"]["offset"])
        assert capture.apply_source_scan(parameters, scan)
        seen.update(scan["pending"])
        # Record consumption uses the already-covered source-only read and
        # local apply ports; the enumerations themselves are fresh processes.
        for _ in scan["pending"]:
            read_work = capture.next_source_work()
            read_parameters = read_work["parameters"]
            context = source_only_context(case, read_parameters)
            locator = read_retained_source_locator(context, read_parameters["lane"], read_parameters["relative"])
            assert capture.apply_source_record(read_parameters, locator)
        if scan["at_end"]:
            break
    else:
        pytest.fail("successive source workers kept rescanning their first batch")
    assert seen == expected
    assert len(cookies) == 2 and cookies[0] != cookies[1]


def consume(executor, request):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        executor.poll()
        result = executor.consume(request.request_id, request)
        if result is not None:
            return result
        time.sleep(0.01)
    pytest.fail("retained recovery worker did not finish")


def complete_empty_capture(case):
    runtime, binding, _source, _target, _scanner, capture, _ledger, executor = case
    revision, _ = runtime.load_registry_snapshot()
    admission = executor.prepare_recovery_admission(binding, revision)
    assert executor.start(admission.request_id) is not None
    assert consume(executor, admission).evidence["state"] == "ready"
    for _ in LANES:
        work = capture.next_source_work()
        parameters = work["parameters"]
        assert work["operation"] == "scan"
        request = executor.prepare_legacy_capture_scan(binding, revision, **parameters)
        assert executor.start(request.request_id) is not None
        result = consume(executor, request)
        assert result.evidence["state"] == "observed"
        assert not result.evidence["scan"]["pending"]
        assert capture.apply_source_scan(parameters, result.evidence["scan"])
    assert capture.next_source_work() is None
    return completion.publish_capture_completion(capture, owner=recovery_owner(runtime, binding))


@pytest.mark.parametrize("change", ["none", "disabled", "digest", "receipt", "conflict", "common_arbiter"])
def test_source_release_worker_uses_exact_local_receipt_without_target_writes(case, monkeypatch, change):
    runtime, binding, source, target, _scanner, _capture, _ledger, executor = case
    proof = complete_empty_capture(case)
    with monkeypatch.context() as guarded:
        forbid_source_io(guarded, source)
        receipt = completion.prepare_completed_source_release(target)
        assert receipt is not None
        context = completion.load_source_release_context(target, completion_digest=receipt["completion_digest"])
        assert context.proof == proof
    if change == "disabled":
        binding = runtime.set_enabled(binding.project_id, False)
    digest = "f" * 64 if change == "digest" else receipt["completion_digest"]
    if change == "receipt":
        (target / completion.SOURCE_RELEASE_FILE).unlink()
    if change == "conflict":
        path = source / completion.CAPTURE_FILE
        hold = read_json(path)
        atomic_replace(path, {**hold, "capture_id": "f" * 32})
    before = {str(path.relative_to(target)): path.read_bytes() for path in target.rglob("*") if path.is_file()}
    revision, _ = runtime.load_registry_snapshot()
    if change == "common_arbiter":
        controller = ProjectIOController(runtime, executor)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            results = controller.advance_recovery_source_releases(
                [binding], revision, {binding.project_id: {"completion_digest": digest}}
            )
            if binding.project_id in results:
                evidence = results[binding.project_id]
                break
            time.sleep(0.01)
        else:
            pytest.fail("common arbiter did not release completed source")
    else:
        request = executor.prepare_recovery_source_release(binding, revision, completion_digest=digest)
        with runtime.registry_guard(), runtime.binding_commit_guard(binding), runtime.migration_guard():
            assert executor.start(request.request_id) is not None
            result = consume(executor, request)
        if change == "conflict":
            assert result.status == "retryable_error"
            assert result.reason_code == "project_io_recovery_source_release_failed"
            assert (source / completion.CAPTURE_FILE).exists()
        else:
            assert result.status == "completed"
        evidence = result.evidence
    after = {str(path.relative_to(target)): path.read_bytes() for path in target.rglob("*") if path.is_file()}
    assert after == before
    if change in {"digest", "receipt"}:
        assert dict(evidence) == {"state": "stale"}
        assert (source / completion.CAPTURE_FILE).exists()
    elif change != "conflict":
        assert dict(evidence) == {"state": "released"}
        assert not (source / completion.CAPTURE_FILE).exists()
        replay = executor.prepare_recovery_source_release(binding, revision, completion_digest=digest)
        assert executor.start(replay.request_id) is not None
        assert consume(executor, replay).evidence["state"] == "released"
        assert completion.is_source_released(target, proof)


@pytest.mark.parametrize("failure", ["exception", "epoch", "receipt"])
def test_lost_source_release_result_is_unknown_until_exact_replay(case, monkeypatch, failure):
    from qqtools.plugins.qexp.agent import project_io_worker as worker

    runtime, binding, source, target, *_middle, executor = case
    proof = complete_empty_capture(case)
    receipt = completion.prepare_completed_source_release(target)
    revision, _ = runtime.load_registry_snapshot()
    request = executor.prepare_recovery_source_release(
        binding, revision, completion_digest=receipt["completion_digest"]
    )
    ticks = worker._process_start_time_ticks(os.getpid())
    assert ticks is not None
    process = ProjectIOProcess(request, os.getpid(), ticks, request.prepared_at, "running")
    process_path = executor.paths["project_io_processes"] / f"{request.request_id}.json"
    atomic_replace(process_path, process.to_dict())
    original = completion.release_source_retention

    def interrupted(*args, **kwargs):
        is_released = original(*args, **kwargs)
        assert is_released
        if failure == "exception":
            raise OSError("source release result lost after durable deletion")
        if failure == "epoch":
            executor.fence_epoch()
        else:
            (target / completion.SOURCE_RELEASE_FILE).unlink()
        return is_released

    monkeypatch.setattr(completion, "release_source_retention", interrupted)
    try:
        assert worker._run(runtime.root, request.request_id) == 0
        result = executor.load_result(request.request_id)
        assert result is not None and result.status == "outcome_unknown"
        assert result.reason_code == "project_io_outcome_unknown" and not result.evidence
        assert not (source / completion.CAPTURE_FILE).exists()
        assert completion.read_capture_completion(target) == proof
    finally:
        process_path.unlink(missing_ok=True)


@pytest.mark.parametrize("change", ["none", "disabled", "digest", "schema_busy", "members", "common_arbiter"])
def test_group_authority_worker_activates_shared_truth_without_local_effects(case, change):
    from qqtools.plugins.qexp.runtime.locks import schema_lock

    runtime, binding, source, target, _scanner, _capture, ledger, executor = case
    proof = complete_empty_capture(case)
    digest = "f" * 64 if change == "digest" else completion._digest(proof)
    if change == "disabled":
        binding = runtime.set_enabled(binding.project_id, False)
    if change == "members":
        ledger.capture_source("retained-attempt-1", {"task_id": "retained", "attempt_number": 1}, source)
        assert completion.prepare_completed_source_release(target) is None
    revision, _ = runtime.load_registry_snapshot()
    before = {str(path.relative_to(target)): path.read_bytes() for path in target.rglob("*") if path.is_file()}
    if change == "common_arbiter":
        controller = ProjectIOController(runtime, executor)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            results = controller.advance_recovery_group_authority(
                [binding], revision, {binding.project_id: {"completion_digest": digest}}
            )
            if binding.project_id in results:
                evidence = results[binding.project_id]
                break
            time.sleep(0.01)
        else:
            pytest.fail("common arbiter did not activate shared Group authority")
    else:
        request = executor.prepare_recovery_group_authority(binding, revision, completion_digest=digest)
        if change == "schema_busy":
            with schema_lock(binding.shared_root, blocking=False) as acquired:
                assert acquired
                assert executor.start(request.request_id) is not None
                result = consume(executor, request)
        else:
            with runtime.registry_guard(), runtime.binding_commit_guard(binding), runtime.migration_guard():
                assert executor.start(request.request_id) is not None
                result = consume(executor, request)
        assert result.status == "completed"
        evidence = result.evidence
    after = {str(path.relative_to(target)): path.read_bytes() for path in target.rglob("*") if path.is_file()}
    assert after == before
    assert (source / completion.CAPTURE_FILE).exists()
    if change in {"digest", "schema_busy"}:
        assert evidence["state"] == ("stale" if change == "digest" else "waiting")
        assert not group_namespace.is_group_authority_isolated(binding.shared_root)
    else:
        assert dict(evidence) == {"state": "active"}
        assert group_namespace.is_group_authority_isolated(binding.shared_root)
        assert not (binding.shared_root / "groups").exists()
        assert (binding.shared_root / "groups-v2").is_dir()
        replay = executor.prepare_recovery_group_authority(binding, revision, completion_digest=digest)
        assert executor.start(replay.request_id) is not None
        assert consume(executor, replay).evidence["state"] == "active"


@pytest.mark.parametrize("failure", ["epoch_before_move", "exception_after_move", "epoch_after_move"])
def test_group_directory_move_is_fenced_and_ambiguous_publication_remains_unknown(case, monkeypatch, failure):
    from qqtools.plugins.qexp.agent import project_io_worker as worker

    runtime, binding, _source, _target, *_middle, executor = case
    proof = complete_empty_capture(case)
    revision, _ = runtime.load_registry_snapshot()
    request = executor.prepare_recovery_group_authority(binding, revision, completion_digest=completion._digest(proof))
    ticks = worker._process_start_time_ticks(os.getpid())
    assert ticks is not None
    process = ProjectIOProcess(request, os.getpid(), ticks, request.prepared_at, "running")
    process_path = executor.paths["project_io_processes"] / f"{request.request_id}.json"
    atomic_replace(process_path, process.to_dict())
    original_check = group_namespace.check_mutation_fence
    original_rename = group_namespace.os.rename
    moves = []

    def check(path):
        if path == binding.shared_root / "groups-v2" and failure == "epoch_before_move":
            executor.fence_epoch()
        return original_check(path)

    def move(source_path, destination_path, *args, **kwargs):
        result = original_rename(source_path, destination_path, *args, **kwargs)
        if destination_path == binding.shared_root / "groups-v2":
            moves.append(destination_path)
            if failure == "exception_after_move":
                raise OSError("Group move result lost after namespace publication")
            if failure == "epoch_after_move":
                executor.fence_epoch()
        return result

    monkeypatch.setattr(group_namespace, "check_mutation_fence", check)
    monkeypatch.setattr(group_namespace.os, "rename", move)
    try:
        assert worker._run(runtime.root, request.request_id) == 0
        result = executor.load_result(request.request_id)
        assert result is not None and result.status == "outcome_unknown"
        assert not result.evidence
        if failure == "epoch_before_move":
            assert not moves and (binding.shared_root / "groups").exists()
        else:
            assert moves == [binding.shared_root / "groups-v2"]
            assert not (binding.shared_root / "groups").exists()
    finally:
        process_path.unlink(missing_ok=True)


def readopt(case):
    runtime, binding, *_rest = case
    path = binding.shared_root / "machines" / binding.machine_name / "registration.json"

    def expire():
        record = read_json(path)
        record["registration"]["eligibility_expires_at"] = "2000-01-01T00:00:00Z"
        atomic_replace(path, record)

    expire()
    other = MachineRuntime(runtime.root.parent / "other-owner")
    other.ensure_binding(binding.shared_root, binding.machine_name, adopt_existing=True)
    expire()
    adopted, _created = runtime.ensure_binding(binding.shared_root, binding.machine_name, adopt_existing=True)
    assert adopted.registration_generation != binding.registration_generation
    return adopted


@pytest.mark.parametrize("boundary", ["retained", "intent", "reset"])
def test_production_enrollment_replays_interrupted_generation_through_new_completion(case, monkeypatch, boundary):
    from qqtools.plugins.qexp.agent.recovery_enrollment import RecoveryEnrollment

    runtime, _binding, source, target, *_middle, executor = case
    old_proof = complete_empty_capture(case)
    binding = readopt(case)
    transition = generation.load_capture_generation_transition(
        target, legacy_source=source, owner=recovery_owner(runtime, binding)
    )
    revision, _ = runtime.load_registry_snapshot()
    retained = executor.prepare_recovery_capture_transition(
        binding, revision, completion_digest=completion._digest(old_proof), phase="retain"
    )
    assert executor.start(retained.request_id) is not None
    assert consume(executor, retained).evidence["state"] == "retained"
    if boundary == "intent":
        # A crash after intent publication but before reset must replay retain,
        # not infer normalization readiness from the intent's mere presence.
        atomic_replace(target / generation.GENERATION_FILE, transition.state)
    elif boundary == "reset":
        generation.apply_generation_reset(transition, generation.generation_source_hold(transition))
    controller = ProjectIOController(runtime, executor)
    enrollment = RecoveryEnrollment(runtime)
    phases = []
    original_start = executor.start

    def started(request_id, **kwargs):
        request = executor._load_request(request_id)
        if request.operation_kind == "recovery_capture_transition":
            phases.append(request.parameters["phase"])
        return original_start(request_id, **kwargs)

    monkeypatch.setattr(executor, "start", started)
    deadline = time.monotonic() + 20
    try:
        with runtime.scheduler_authority() as acquired:
            assert acquired
            while time.monotonic() < deadline:
                revision, bindings = runtime.load_registry_snapshot()
                with monkeypatch.context() as guarded, controller.admission_turn():
                    forbid_source_io(guarded, source)
                    forbid_source_io(guarded, binding.shared_root)
                    enrollment.advance(controller, bindings, revision)
                if binding in enrollment._settled:
                    break
                time.sleep(0.01)
            assert binding in enrollment._settled
        assert phases == (["normalize"] if boundary == "reset" else ["retain", "normalize"])
        proof = completion.read_capture_completion(target)
        assert proof["registration_generation"] == binding.registration_generation
        assert proof["capture_digest"] != old_proof["capture_digest"]
        assert not (target / generation.GENERATION_FILE).exists()
        assert not (source / completion.CAPTURE_FILE).exists()
        assert group_namespace.is_group_authority_isolated(binding.shared_root)
    finally:
        enrollment.stop()


@pytest.mark.parametrize("released", [False, True])
def test_fresh_generation_workers_and_local_reset_preserve_exact_replay_boundary(case, monkeypatch, released):
    runtime, _binding, source, target, _scanner, _capture, _ledger, executor = case
    proof = complete_empty_capture(case)
    if released:
        receipt = completion.prepare_completed_source_release(target)
        revision, _ = runtime.load_registry_snapshot()
        request = executor.prepare_recovery_source_release(
            _binding, revision, completion_digest=receipt["completion_digest"]
        )
        assert executor.start(request.request_id) is not None
        assert consume(executor, request).evidence["state"] == "released"
    binding = readopt(case)
    revision, _ = runtime.load_registry_snapshot()
    owner = recovery_owner(runtime, binding)
    digest = completion._digest(proof)
    with monkeypatch.context() as guarded:
        forbid_source_io(guarded, source)
        transition = generation.load_capture_generation_transition(target, legacy_source=source, owner=owner)
    before = {str(path.relative_to(target)): path.read_bytes() for path in target.rglob("*") if path.is_file()}
    retain = executor.prepare_recovery_capture_transition(binding, revision, completion_digest=digest, phase="retain")
    with runtime.registry_guard(), runtime.binding_commit_guard(binding), runtime.migration_guard():
        assert executor.start(retain.request_id) is not None
        assert consume(executor, retain).evidence["state"] == "retained"
    assert {str(path.relative_to(target)): path.read_bytes() for path in target.rglob("*") if path.is_file()} == before
    hold = generation.generation_source_hold(transition)
    assert read_json(source / completion.CAPTURE_FILE) == {
        **hold.to_dict(),
        "phase": "generation_pending",
        "admission": transition.state["owner"],
    }
    # A result may be lost after source fencing, before the local intent exists.
    repeat = executor.prepare_recovery_capture_transition(binding, revision, completion_digest=digest, phase="retain")
    assert executor.start(repeat.request_id) is not None
    assert consume(executor, repeat).evidence["state"] == "retained"
    with monkeypatch.context() as guarded:
        forbid_source_io(guarded, source)
        generation.apply_generation_reset(transition, hold)
        generation.apply_generation_reset(transition, hold)
    assert not (target / completion.COMPLETION_FILE).exists()
    assert (target / generation.GENERATION_FILE).exists()
    assert read_json(target / completion.CAPTURE_FILE) == transition.after
    assert transition.after["progress"]["revision"] == transition.before["progress"]["revision"] + 1
    before = {str(path.relative_to(target)): path.read_bytes() for path in target.rglob("*") if path.is_file()}
    normalize = executor.prepare_recovery_capture_transition(
        binding, revision, completion_digest=digest, phase="normalize"
    )
    with runtime.registry_guard(), runtime.binding_commit_guard(binding), runtime.migration_guard():
        assert executor.start(normalize.request_id) is not None
        assert consume(executor, normalize).evidence["state"] == "normalized"
    assert {str(path.relative_to(target)): path.read_bytes() for path in target.rglob("*") if path.is_file()} == before
    assert read_json(source / completion.CAPTURE_FILE) == hold.to_dict()
    repeat = executor.prepare_recovery_capture_transition(
        binding, revision, completion_digest=digest, phase="normalize"
    )
    assert executor.start(repeat.request_id) is not None
    assert consume(executor, repeat).evidence["state"] == "normalized"
    with monkeypatch.context() as guarded:
        forbid_source_io(guarded, source)
        generation.finish_generation_reset(transition, hold)
    assert not (target / generation.GENERATION_FILE).exists()
    assert read_json(target / completion.CAPTURE_FILE) == transition.after


def test_generation_source_normalization_cannot_skip_local_reset(case):
    runtime, _binding, source, target, *_middle, executor = case
    proof = complete_empty_capture(case)
    binding = readopt(case)
    revision, _ = runtime.load_registry_snapshot()
    before = (source / completion.CAPTURE_FILE).read_bytes()
    target_bytes = (target / completion.COMPLETION_FILE).read_bytes()
    request = executor.prepare_recovery_capture_transition(
        binding, revision, completion_digest=completion._digest(proof), phase="normalize"
    )
    assert executor.start(request.request_id) is not None
    result = consume(executor, request)
    assert result.status == "completed" and dict(result.evidence) == {"state": "stale"}
    assert (source / completion.CAPTURE_FILE).read_bytes() == before
    assert (target / completion.COMPLETION_FILE).read_bytes() == target_bytes
    assert not (target / generation.GENERATION_FILE).exists()


@pytest.mark.parametrize("phase", ["retain", "normalize"])
def test_generation_result_loss_after_source_effect_is_unknown_not_local_reset(case, monkeypatch, phase):
    from qqtools.plugins.qexp.agent import project_io_worker as worker

    runtime, _binding, source, target, *_middle, executor = case
    proof = complete_empty_capture(case)
    binding = readopt(case)
    revision, _ = runtime.load_registry_snapshot()
    transition = generation.load_capture_generation_transition(
        target, legacy_source=source, owner=recovery_owner(runtime, binding)
    )
    if phase == "normalize":
        retained = executor.prepare_recovery_capture_transition(
            binding, revision, completion_digest=completion._digest(proof), phase="retain"
        )
        assert executor.start(retained.request_id) is not None
        assert consume(executor, retained).evidence["state"] == "retained"
        generation.apply_generation_reset(transition, generation.generation_source_hold(transition))
    request = executor.prepare_recovery_capture_transition(
        binding, revision, completion_digest=completion._digest(proof), phase=phase
    )
    ticks = worker._process_start_time_ticks(os.getpid())
    assert ticks is not None
    process = ProjectIOProcess(request, os.getpid(), ticks, request.prepared_at, "running")
    process_path = executor.paths["project_io_processes"] / f"{request.request_id}.json"
    atomic_replace(process_path, process.to_dict())
    original = generation.retain_generation_source if phase == "retain" else generation.normalize_generation_source
    before = {str(path.relative_to(target)): path.read_bytes() for path in target.rglob("*") if path.is_file()}

    def interrupted(*args, **kwargs):
        hold = original(*args, **kwargs)
        assert hold is not None
        raise OSError("generation source result lost after publication")

    monkeypatch.setattr(
        generation, "retain_generation_source" if phase == "retain" else "normalize_generation_source", interrupted
    )
    try:
        assert worker._run(runtime.root, request.request_id) == 0
        result = executor.load_result(request.request_id)
        assert result is not None and result.status == "outcome_unknown" and not result.evidence
        assert {
            str(path.relative_to(target)): path.read_bytes() for path in target.rglob("*") if path.is_file()
        } == before
        assert read_json(source / completion.CAPTURE_FILE)["phase"] == (
            "generation_pending" if phase == "retain" else "pending"
        )
    finally:
        process_path.unlink(missing_ok=True)
