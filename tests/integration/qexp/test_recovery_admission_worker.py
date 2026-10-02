"""Shared recovery preparation uses worker ownership, never machine guards."""

import os
import time

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent import project_io_worker as worker
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.project_io_controller import ProjectIOController
from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor
from qqtools.plugins.qexp.agent.project_io_protocol import ProjectIOProcess
from qqtools.plugins.qexp.agent.registration import MachineRegistration
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.layout import load_machine_record, load_machine_registration
from qqtools.plugins.qexp.runtime import recovery_admission, registration_authority
from qqtools.plugins.qexp.runtime.locks import schema_lock
from qqtools.plugins.qexp.runtime.responsibility_store import DurableIO
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


@pytest.fixture
def case(tmp_path):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg = init_shared_root(tmp_path / "project/.qexp", "gpu-1")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    try:
        yield runtime, binding, cfg, executor
    finally:
        executor.shutdown()


def consume(executor, request):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        executor.poll()
        result = executor.consume(request.request_id, request)
        if result is not None:
            return result
        time.sleep(0.01)
    pytest.fail("recovery preparation worker did not finish")


def registration(cfg):
    return read_json(cfg.shared_root / "machines" / cfg.machine_name / "registration.json")["registration"]


def capability(cfg):
    return "local-recovery-v1" in read_json(cfg.shared_root / "schema/version.json")["schema"]["required_capabilities"]


@pytest.mark.parametrize("change", ["generation", "instance", "superseded", "invalid", "missing", "expired"])
def test_recovery_settles_only_positive_superseded_evidence_not_missing_or_invalid_ownership(case, change):
    from qqtools.plugins.qexp.agent.recovery_enrollment import RecoveryEnrollment

    runtime, binding, cfg, executor = case
    path = cfg.shared_root / "machines" / cfg.machine_name / "registration.json"
    record = registration(cfg)
    if change == "generation":
        record["generation"] = "replacement-generation"
    elif change == "instance":
        record["runtime_instance_id"] = "replacement-instance"
    elif change == "superseded":
        record["state"] = "superseded"
    elif change == "invalid":
        record["eligibility_expires_at"] = "not-a-timestamp"
    elif change == "expired":
        record["eligibility_expires_at"] = "2000-01-01T00:00:00Z"
    if change == "missing":
        path.unlink()
    else:
        atomic_replace(path, {"registration": record})
    before = None if change == "missing" else path.read_bytes()
    revision, _ = runtime.load_registry_snapshot()
    request = executor.prepare_recovery_admission(binding, revision)
    with runtime.registry_guard(), runtime.binding_commit_guard(binding):
        assert executor.start(request.request_id) is not None
        result = consume(executor, request)
    assert not capability(cfg)
    assert (path.read_bytes() if path.exists() else None) == before
    if change in {"invalid", "missing", "expired"}:
        assert result.evidence.get("state") != "superseded"
        return
    assert result.status == "completed"
    assert dict(result.evidence) == {"state": "superseded", "registration_prepared": False, "admission_fenced": False}
    controller = ProjectIOController(runtime, executor)
    enrollment = RecoveryEnrollment(runtime)
    try:
        with runtime.scheduler_authority() as acquired:
            assert acquired
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and binding not in enrollment._settled:
                with controller.admission_turn():
                    enrollment.advance(controller, [binding], revision)
                time.sleep(0.01)
            assert binding in enrollment._settled
        assert not runtime.recovery_enrollment_pending_projects
        assert not (runtime.project_paths(binding.project_id)["root"] / "responsibility-writer-capture.json").exists()
    finally:
        enrollment.stop()


@pytest.mark.parametrize("disabled", [False, True])
def test_actual_worker_prepares_registration_and_root_while_parent_holds_machine_guards(case, disabled):
    runtime, binding, cfg, executor = case
    if disabled:
        binding = runtime.set_enabled(binding.project_id, False)
    revision, _ = runtime.load_registry_snapshot()
    before = runtime.paths["registry"].read_bytes()
    original = registration(cfg)
    request = executor.prepare_recovery_admission(binding, revision)
    with runtime.registry_guard(), runtime.binding_commit_guard(binding), runtime.migration_guard():
        assert executor.start(request.request_id) is not None
        result = consume(executor, request)
    assert result.status == "completed"
    assert dict(result.evidence) == {"state": "ready", "registration_prepared": True, "admission_fenced": True}
    current = registration(cfg)
    assert current["version"] == current["protocol_version"] == 2
    for key in ("generation", "runtime_root", "runtime_instance_id", "eligibility_expires_at"):
        assert current[key] == original[key]
    assert capability(cfg)
    assert runtime.paths["registry"].read_bytes() == before
    assert not (runtime.project_paths(binding.project_id)["root"] / "responsibility-writer-capture.json").exists()


def test_valid_registration_journal_is_rolled_back_before_recovery_preparation(case):
    runtime, binding, cfg, executor = case
    revision, bindings = runtime.load_registry()
    original_registration = load_machine_registration(cfg)
    original_machine = load_machine_record(cfg)
    runtime.registration._save_registration_transaction(
        revision=revision,
        bindings=bindings,
        registrations=[(cfg, original_registration)],
        machine_records=[(cfg, original_machine)],
    )
    changed = read_json(cfg.shared_root / "machines" / cfg.machine_name / "registration.json")
    changed["registration"]["state"] = "superseded"
    atomic_replace(cfg.shared_root / "machines" / cfg.machine_name / "registration.json", changed)

    request = executor.prepare_recovery_admission(binding, revision)
    assert executor.start(request.request_id) is not None
    result = consume(executor, request)

    assert dict(result.evidence) == {"state": "waiting", "registration_prepared": False, "admission_fenced": False}
    assert not runtime.paths["registration_transaction"].exists()
    assert load_machine_registration(cfg) == original_registration
    assert load_machine_record(cfg) == original_machine
    current_revision, current_bindings = runtime.load_registry_uncached()
    assert current_revision == revision + 1
    assert current_bindings == bindings


def test_malformed_registration_journal_fails_closed_without_retiring_evidence(case):
    runtime, binding, _cfg, executor = case
    atomic_replace(runtime.paths["registration_transaction"], {"pending": True})
    revision, _ = runtime.load_registry_snapshot()
    request = executor.prepare_recovery_admission(binding, revision)
    result = run_in_process(executor, request)

    assert result.status == "retryable_error"
    assert result.reason_code == "project_io_recovery_admission_failed"
    assert read_json(runtime.paths["registration_transaction"]) == {"pending": True}


def test_registration_rollback_epoch_loss_retains_journal_for_exact_replay(case, monkeypatch):
    runtime, binding, cfg, executor = case
    revision, bindings = runtime.load_registry()
    original_registration = load_machine_registration(cfg)
    runtime.registration._save_registration_transaction(
        revision=revision,
        bindings=bindings,
        registrations=[(cfg, original_registration)],
        machine_records=[(cfg, load_machine_record(cfg))],
    )
    changed = read_json(cfg.shared_root / "machines" / cfg.machine_name / "registration.json")
    changed["registration"]["state"] = "superseded"
    atomic_replace(cfg.shared_root / "machines" / cfg.machine_name / "registration.json", changed)
    original_rollback = MachineRegistration.rollback_pending_locked

    def fence_between_effects(self, *, before_effect=None):
        calls = 0

        def fenced_effect():
            nonlocal calls
            calls += 1
            if calls == 2:
                executor.fence_epoch()
            assert before_effect is not None
            before_effect()

        return original_rollback(self, before_effect=fenced_effect)

    monkeypatch.setattr(MachineRegistration, "rollback_pending_locked", fence_between_effects)
    request = executor.prepare_recovery_admission(binding, revision)
    result = run_in_process(executor, request)

    assert result.status == "outcome_unknown"
    assert result.reason_code == "project_io_outcome_unknown"
    assert runtime.paths["registration_transaction"].exists()
    assert load_machine_registration(cfg) == original_registration


def test_local_pending_migration_defers_without_shared_writes(case):
    runtime, binding, cfg, executor = case
    atomic_replace(runtime.migration_path(binding.project_id), {"migration": {"state": "prepared"}})
    original = registration(cfg)
    revision, _ = runtime.load_registry_snapshot()
    request = executor.prepare_recovery_admission(binding, revision)
    assert executor.start(request.request_id) is not None
    result = consume(executor, request)
    assert dict(result.evidence) == {"state": "waiting", "registration_prepared": False, "admission_fenced": False}
    assert registration(cfg) == original
    assert not capability(cfg)


def test_schema_busy_does_not_confuse_prepared_registration_with_root_admission(case):
    runtime, binding, cfg, executor = case
    revision, _ = runtime.load_registry_snapshot()
    request = executor.prepare_recovery_admission(binding, revision)
    with schema_lock(cfg.shared_root, blocking=False) as acquired:
        assert acquired
        assert executor.start(request.request_id) is not None
        result = consume(executor, request)
    assert dict(result.evidence) == {"state": "waiting", "registration_prepared": True, "admission_fenced": False}
    assert registration(cfg)["version"] == 2
    assert not capability(cfg)
    retry = executor.prepare_recovery_admission(binding, revision)
    assert executor.start(retry.request_id) is not None
    assert consume(executor, retry).evidence["state"] == "ready"
    assert capability(cfg)


def test_common_arbiter_consumes_recovery_readiness_without_inline_shared_domain(case, monkeypatch):
    runtime, binding, cfg, executor = case
    revision, _ = runtime.load_registry_snapshot()
    controller = ProjectIOController(runtime, executor)
    monkeypatch.setattr(
        worker, "_recovery_admission", lambda *_: pytest.fail("coordinator ran shared preparation inline")
    )
    deadline = time.monotonic() + 5
    completed = {}
    while time.monotonic() < deadline:
        completed = controller.advance_recovery_admission([binding], revision, {binding.project_id: {}})
        if binding.project_id in completed:
            break
        time.sleep(0.01)
    assert completed[binding.project_id]["state"] == "ready"
    assert capability(cfg)
    assert not executor.unresolved_requests()


def run_in_process(executor, request):
    """Own the synthetic process record, which must never reach shutdown."""
    ticks = worker._process_start_time_ticks(os.getpid())
    assert ticks is not None
    process = ProjectIOProcess(request, os.getpid(), ticks, request.prepared_at, "running")
    path = executor.paths["project_io_processes"] / f"{request.request_id}.json"
    atomic_replace(path, process.to_dict())
    try:
        assert worker._run(executor.runtime.root, request.request_id) == 0
        result = executor.load_result(request.request_id)
        assert result is not None
        return result
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("failure", ["exception", "epoch", "journal"])
def test_lost_readiness_after_shared_publication_retains_ambiguous_outcome(case, monkeypatch, failure):
    runtime, binding, cfg, executor = case
    revision, _ = runtime.load_registry_snapshot()
    request = executor.prepare_recovery_admission(binding, revision)
    original = recovery_admission.prepare_shared_recovery_admission

    def interrupted(*args, **kwargs):
        prepared = original(*args, **kwargs)
        assert prepared.admission.is_fenced
        if failure == "exception":
            raise OSError("readiness result lost after shared publication")
        if failure == "epoch":
            executor.fence_epoch()
        else:
            atomic_replace(runtime.paths["registration_transaction"], {"pending": True})
        return prepared

    monkeypatch.setattr(recovery_admission, "prepare_shared_recovery_admission", interrupted)
    result = run_in_process(executor, request)
    assert result.status == "outcome_unknown"
    assert result.reason_code == "project_io_outcome_unknown"
    assert not result.evidence
    assert registration(cfg)["version"] == 2
    assert capability(cfg)
    assert not (runtime.project_paths(binding.project_id)["root"] / "responsibility-writer-capture.json").exists()


def test_local_namespace_durability_failure_prevents_registration_conversion(case, monkeypatch):
    runtime, binding, cfg, executor = case
    revision, _ = runtime.load_registry_snapshot()
    request = executor.prepare_recovery_admission(binding, revision)
    original = registration(cfg)
    original_sync = DurableIO.sync_directory

    def fail_local_barrier(io, path, prefix):
        if path == runtime.paths["registration_transaction"].parent and prefix == "recovery_registration":
            raise OSError("local rollback namespace is not durable")
        return original_sync(io, path, prefix)

    monkeypatch.setattr(DurableIO, "sync_directory", fail_local_barrier)
    result = run_in_process(executor, request)
    assert result.status == "retryable_error"
    assert result.reason_code == "project_io_recovery_admission_failed"
    assert registration(cfg) == original
    assert not capability(cfg)


def test_journal_arrival_before_conversion_defers_without_retiring_snapshot(case, monkeypatch):
    runtime, binding, cfg, executor = case
    revision, _ = runtime.load_registry_snapshot()
    request = executor.prepare_recovery_admission(binding, revision)
    original = registration(cfg)
    original_sync = DurableIO.sync_directory
    journal = {"pending": "must be restored before preparation"}

    def publish_journal(io, path, prefix):
        result = original_sync(io, path, prefix)
        if path == runtime.paths["registration_transaction"].parent and prefix == "recovery_registration":
            atomic_replace(runtime.paths["registration_transaction"], journal)
        return result

    monkeypatch.setattr(DurableIO, "sync_directory", publish_journal)
    result = run_in_process(executor, request)
    assert result.status == "completed"
    assert dict(result.evidence) == {"state": "waiting", "registration_prepared": False, "admission_fenced": False}
    assert registration(cfg) == original
    assert not capability(cfg)
    assert read_json(runtime.paths["registration_transaction"]) == journal


def test_replay_completes_shared_barriers_without_republishing_registration(case, monkeypatch):
    runtime, binding, cfg, executor = case
    revision, _ = runtime.load_registry_snapshot()
    request = executor.prepare_recovery_admission(binding, revision)
    assert executor.start(request.request_id) is not None
    assert consume(executor, request).evidence["state"] == "ready"
    original_registration = registration(cfg)
    original_schema = (cfg.shared_root / "schema/version.json").read_bytes()
    original_sync = DurableIO.sync_directory
    barriers = []

    def record_barrier(io, path, prefix):
        barriers.append((path, prefix))
        return original_sync(io, path, prefix)

    monkeypatch.setattr(DurableIO, "sync_directory", record_barrier)
    monkeypatch.setattr(
        registration_authority,
        "save_machine_registration",
        lambda *_: pytest.fail("prepared registration must not be republished"),
    )
    request = executor.prepare_recovery_admission(binding, revision)
    result = run_in_process(executor, request)
    assert result.status == "completed" and result.evidence["state"] == "ready"
    assert (cfg.shared_root / "machines" / cfg.machine_name, "recovery_registration_fence") in barriers
    assert (cfg.shared_root / "schema", "recovery_admission") in barriers
    assert (runtime.paths["registration_transaction"].parent, "recovery_registration") not in barriers
    assert registration(cfg) == original_registration
    assert (cfg.shared_root / "schema/version.json").read_bytes() == original_schema
