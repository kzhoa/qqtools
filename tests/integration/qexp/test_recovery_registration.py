"""Admission preparation is fenced, durable and distinct from discovery activation."""

import os
from dataclasses import replace

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.registration import RECOVERY_REGISTRATION_PROTOCOL
from qqtools.plugins.qexp.layout import load_machine_record, load_machine_registration, save_machine_record
from qqtools.plugins.qexp.runtime.responsibility_store import DurableIO
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


@pytest.fixture
def registered(tmp_path):
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy")
    runtime = MachineRuntime(tmp_path / "machine")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    path = cfg.shared_root / "machines" / cfg.machine_name / "registration.json"
    return runtime, binding, cfg, path


def prepare(runtime, binding):
    with runtime.scheduler_authority() as acquired:
        assert acquired
        return runtime.prepare_recovery_registration(binding)


def test_preparation_preserves_authority_and_retry_does_not_republish(registered, monkeypatch):
    runtime, binding, cfg, path = registered
    original = read_json(path)["registration"]
    assert original["version"] == 1
    assert prepare(runtime, binding)
    stored = read_json(path)["registration"]
    for key in ("generation", "runtime_instance_id", "runtime_root", "state", "eligibility_expires_at", "created_at"):
        assert stored[key] == original[key]
    assert stored["version"] == stored["protocol_version"] == 2
    assert stored["recovery_protocol"] == RECOVERY_REGISTRATION_PROTOCOL
    assert runtime.registration_status(binding)["write_eligible"]
    assert "local-recovery-v1" not in read_json(cfg.shared_root / "schema" / "version.json")["schema"].get(
        "required_capabilities", []
    )

    def unexpected_write(*args, **kwargs):
        pytest.fail("already prepared registration must not be republished")

    monkeypatch.setattr("qqtools.plugins.qexp.agent.registration.save_machine_registration", unexpected_write)
    assert prepare(runtime, binding)


def test_only_current_scheduler_owner_can_prepare(registered):
    runtime, binding, _cfg, path = registered
    original = path.read_bytes()
    with pytest.raises(RuntimeError, match="scheduler authority"):
        runtime.prepare_recovery_registration(binding)
    assert prepare(runtime, binding)
    with pytest.raises(RuntimeError, match="scheduler authority"):
        runtime.prepare_recovery_registration(binding)
    assert path.read_bytes() != original


def test_fork_cannot_reuse_inherited_scheduler_authority(registered):
    runtime, binding, _cfg, path = registered
    original = path.read_bytes()
    with runtime.scheduler_authority() as acquired:
        assert acquired
        pid = os.fork()
        if pid == 0:
            try:
                runtime.prepare_recovery_registration(binding)
            except RuntimeError:
                os._exit(0)
            os._exit(1)
        assert os.waitpid(pid, 0)[1] == 0
    assert path.read_bytes() == original


@pytest.mark.parametrize("guard", ["migration_guard", "registry_guard"])
def test_busy_local_lifecycle_defers_without_writing(registered, guard):
    runtime, binding, _cfg, path = registered
    original = path.read_bytes()
    with getattr(runtime, guard)() as acquired:
        assert acquired
        assert not prepare(runtime, binding)
    assert path.read_bytes() == original


@pytest.mark.parametrize("state", ["prepared", "legacy_agent_stopped", "blocked", "reservations_imported"])
def test_incomplete_legacy_migration_defers_preparation(registered, state):
    runtime, binding, _cfg, path = registered
    original = path.read_bytes()
    atomic_replace(runtime.migration_path(binding.project_id), {"migration": {"state": state}})
    assert not prepare(runtime, binding)
    assert path.read_bytes() == original


def test_legacy_machine_metadata_is_not_readiness(registered):
    runtime, binding, cfg, path = registered
    original = path.read_bytes()
    record = load_machine_record(cfg)
    record["machine"]["agent_runtime"] = "project"
    save_machine_record(cfg, record)
    assert not prepare(runtime, binding)
    assert path.read_bytes() == original


@pytest.mark.parametrize("change", ["stale_binding", "superseded", "expired", "wrong_runtime"])
def test_stale_or_ineligible_ownership_cannot_prepare(registered, change):
    runtime, binding, _cfg, path = registered
    record = read_json(path)
    if change == "stale_binding":
        binding = replace(binding, registration_generation="another-generation")
    elif change == "superseded":
        record["registration"]["state"] = "superseded"
    elif change == "expired":
        record["registration"]["eligibility_expires_at"] = "2000-01-01T00:00:00Z"
    else:
        record["registration"]["runtime_instance_id"] = "another-runtime"
    atomic_replace(path, record)
    original = path.read_bytes()
    assert not prepare(runtime, binding)
    assert path.read_bytes() == original


def test_disabled_binding_can_prepare_without_reenabling(registered):
    runtime, binding, _cfg, _path = registered
    binding = runtime.set_enabled(binding.project_id, False)
    assert prepare(runtime, binding)
    assert runtime.load_registry()[1] == [binding]
    assert not binding.enabled


def test_rebinding_reactivation_and_adoption_preserve_protocol_floor(registered, tmp_path):
    runtime, binding, cfg, path = registered
    assert prepare(runtime, binding)
    rebound, added = runtime.ensure_binding(cfg.shared_root, cfg.machine_name)
    assert not added and rebound == binding
    record = read_json(path)
    record["registration"]["eligibility_expires_at"] = "2000-01-01T00:00:00Z"
    atomic_replace(path, record)
    assert runtime.reactivate_binding(binding)
    assert read_json(path)["registration"]["version"] == 2
    runtime.registration._supersede_registration(binding, "replacement")
    replacement = MachineRuntime(tmp_path / "replacement")
    adopted, _added = replacement.ensure_binding(cfg.shared_root, cfg.machine_name, adopt_existing=True)
    stored = read_json(path)["registration"]
    assert stored["version"] == stored["protocol_version"] == 2
    assert stored["recovery_protocol"] == RECOVERY_REGISTRATION_PROTOCOL
    assert stored["generation"] == adopted.registration_generation != binding.registration_generation
    assert replacement.registration_status(adopted)["write_eligible"]


@pytest.mark.parametrize(
    "damage", [{"version": True}, {"version": 3}, {"protocol_version": 1}, {"recovery_protocol": "unknown"}]
)
def test_unsupported_protocol_cannot_prepare_or_rebind(registered, damage):
    runtime, binding, cfg, path = registered
    assert prepare(runtime, binding)
    record = read_json(path)
    record["registration"].update(damage)
    atomic_replace(path, record)
    original = path.read_bytes()
    status = runtime.registration_status(binding)
    assert not status["write_eligible"]
    assert status["registration_version"] == record["registration"].get("version")
    assert status["recovery_protocol"] == record["registration"].get("recovery_protocol")
    with pytest.raises(RuntimeError, match="unsupported"):
        prepare(runtime, binding)
    with pytest.raises(RuntimeError, match="unsupported"):
        runtime.ensure_binding(cfg.shared_root, cfg.machine_name)
    assert path.read_bytes() == original


def test_old_transaction_is_replayed_and_durably_retired_before_preparation(registered, monkeypatch):
    runtime, binding, cfg, path = registered
    revision, bindings = runtime.load_registry()
    old = load_machine_registration(cfg)
    runtime.registration._save_registration_transaction(
        revision=revision, bindings=bindings, registrations=[(cfg, old)], machine_records=[]
    )
    damaged = read_json(path)
    damaged["registration"]["generation"] = "uncommitted"
    atomic_replace(path, damaged)
    sync = DurableIO.sync_directory
    barriers = []

    def check_barrier(self, directory, label):
        if label == "recovery_registration":
            assert not runtime.paths["registration_transaction"].exists()
            assert read_json(path)["registration"]["version"] == 1
            assert read_json(path)["registration"]["generation"] == binding.registration_generation
            barriers.append(directory)
        return sync(self, directory, label)

    monkeypatch.setattr(DurableIO, "sync_directory", check_barrier)
    assert prepare(runtime, binding)
    assert barriers == [runtime.root]
    with runtime.registry_guard():
        runtime.registration.rollback_pending_locked()
    assert read_json(path)["registration"]["version"] == 2


def test_failed_retirement_barrier_does_not_publish_protocol_and_retry_syncs(registered, monkeypatch):
    runtime, binding, _cfg, path = registered
    sync = DurableIO.sync_directory

    def failed_sync(self, directory, label):
        if label == "recovery_registration":
            raise OSError("retirement fsync failed")
        return sync(self, directory, label)

    monkeypatch.setattr(DurableIO, "sync_directory", failed_sync)
    with pytest.raises(OSError, match="retirement fsync"):
        prepare(runtime, binding)
    assert read_json(path)["registration"]["version"] == 1
    monkeypatch.setattr(DurableIO, "sync_directory", sync)
    assert prepare(runtime, binding)
    assert read_json(path)["registration"]["version"] == 2


@pytest.mark.parametrize("boundary", ["retired_old_transaction", "published_protocol"])
def test_process_crash_retries_preparation_without_restoring_old_protocol(registered, boundary):
    runtime, binding, cfg, path = registered
    revision, bindings = runtime.load_registry()
    runtime.registration._save_registration_transaction(
        revision=revision,
        bindings=bindings,
        registrations=[(cfg, load_machine_registration(cfg))],
        machine_records=[],
    )
    pid = os.fork()
    if pid == 0:
        from qqtools.plugins.qexp.agent import registration as registration_owner

        sync = DurableIO.sync_directory
        save = registration_owner.save_machine_registration

        def interrupted_sync(self, directory, label):
            if boundary == "retired_old_transaction" and label == "recovery_registration":
                os._exit(41)
            return sync(self, directory, label)

        def interrupted_save(config, value):
            save(config, value)
            if boundary == "published_protocol" and value["registration"]["version"] == 2:
                os._exit(41)

        DurableIO.sync_directory = interrupted_sync
        registration_owner.save_machine_registration = interrupted_save
        try:
            prepare(runtime, binding)
        finally:
            os._exit(42)
    assert os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]) == 41
    assert read_json(path)["registration"]["version"] == (1 if boundary == "retired_old_transaction" else 2)
    restarted = MachineRuntime(runtime.root)
    assert prepare(restarted, binding)
    assert not restarted.paths["registration_transaction"].exists()
    with restarted.registry_guard():
        restarted.registration.rollback_pending_locked()
    stored = read_json(path)["registration"]
    assert stored["version"] == 2
    assert stored["generation"] == binding.registration_generation
    assert restarted.registration_status(binding)["write_eligible"]


def test_scheduler_does_not_release_authority_during_preparation(registered, monkeypatch):
    from threading import Event, Thread

    from qqtools.plugins.qexp.agent import registration as registration_owner

    runtime, binding, _cfg, path = registered
    acquired = Event()
    exit_requested = Event()
    leaving = Event()
    exited = Event()
    publishing = Event()
    finish_publication = Event()
    errors = []
    prepared = []
    save = registration_owner.save_machine_registration

    def slow_save(cfg, value):
        if value["registration"]["version"] == 2:
            publishing.set()
            if not finish_publication.wait(5):
                raise RuntimeError("test publication release timed out")
        save(cfg, value)

    def owner():
        try:
            with runtime.scheduler_authority() as owns:
                assert owns
                acquired.set()
                assert exit_requested.wait(5)
                leaving.set()
            exited.set()
        except BaseException as exc:
            errors.append(exc)

    def worker():
        try:
            prepared.append(runtime.prepare_recovery_registration(binding))
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(registration_owner, "save_machine_registration", slow_save)
    owner_thread = Thread(target=owner)
    worker_thread = Thread(target=worker)
    owner_thread.start()
    try:
        assert acquired.wait(5)
        worker_thread.start()
        assert publishing.wait(5)
        exit_requested.set()
        assert leaving.wait(5)
        assert not exited.wait(0.05)
        contender = MachineRuntime(runtime.root)
        with contender.scheduler_authority() as owns:
            assert not owns
    finally:
        finish_publication.set()
        exit_requested.set()
        if worker_thread.ident is not None:
            worker_thread.join(5)
        owner_thread.join(5)
    assert not errors
    assert not worker_thread.is_alive() and not owner_thread.is_alive()
    assert prepared == [True] and exited.is_set()
    assert read_json(path)["registration"]["version"] == 2
    with contender.scheduler_authority() as owns:
        assert owns
    with pytest.raises(RuntimeError, match="scheduler authority"):
        runtime.prepare_recovery_registration(binding)


def test_visible_protocol_after_failed_publication_requires_retry_barrier(registered, monkeypatch):
    from qqtools.plugins.qexp.agent import registration as registration_owner

    runtime, binding, _cfg, path = registered
    save = registration_owner.save_machine_registration
    sync = DurableIO.sync_directory

    def uncertain_save(cfg, value):
        save(cfg, value)
        if value["registration"]["version"] == 2:
            raise OSError("publication durability is uncertain")

    monkeypatch.setattr(registration_owner, "save_machine_registration", uncertain_save)
    with pytest.raises(OSError, match="durability is uncertain"):
        prepare(runtime, binding)
    assert read_json(path)["registration"]["version"] == 2
    barriers = []

    def failed_retry(self, directory, label):
        if label == "recovery_registration_fence":
            barriers.append(directory)
            raise OSError("retry barrier failed")
        return sync(self, directory, label)

    monkeypatch.setattr(DurableIO, "sync_directory", failed_retry)
    with pytest.raises(OSError, match="retry barrier"):
        prepare(runtime, binding)
    assert barriers == [path.parent]
    monkeypatch.setattr(DurableIO, "sync_directory", sync)
    assert prepare(runtime, binding)
    status = runtime.registration_status(binding)
    assert status["registration_version"] == 2
    assert status["recovery_protocol"] == RECOVERY_REGISTRATION_PROTOCOL


@pytest.mark.parametrize("machine", [None, [], "invalid"])
def test_malformed_machine_metadata_cannot_prepare(registered, machine):
    runtime, binding, cfg, path = registered
    original = path.read_bytes()
    save_machine_record(cfg, {"machine": machine})
    with pytest.raises(RuntimeError, match="no valid standalone runtime_root"):
        prepare(runtime, binding)
    assert path.read_bytes() == original


def test_missing_registration_reports_absent_protocol_diagnostics(registered):
    runtime, binding, _cfg, path = registered
    path.unlink()
    status = runtime.registration_status(binding)
    assert status["state"] == "unregistered" and not status["write_eligible"]
    assert status["registration_version"] is None
    assert status["recovery_protocol"] is None
