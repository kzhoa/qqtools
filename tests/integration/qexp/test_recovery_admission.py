"""Root admission fencing requires durable preparation of every participant."""

import os
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.layout import LOCAL_RECOVERY_CAPABILITY
from qqtools.plugins.qexp.runtime import recovery_admission
from qqtools.plugins.qexp.runtime.locks import schema_lock
from qqtools.plugins.qexp.runtime.recovery_admission import fence_recovery_admission
from qqtools.plugins.qexp.runtime.responsibility_store import DurableIO
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.upgrade import UpgradeCoordinator
from qqtools.plugins.qexp.runtime.upgrade.production import UpgradeJournalMigration

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


@pytest.fixture
def project(tmp_path):
    cfg = init_shared_root(tmp_path / ".qexp", "machine-a", runtime_root=tmp_path / "legacy-a")
    runtime = MachineRuntime(tmp_path / "runtime-a")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    return runtime, binding, cfg


def prepare(runtime, binding):
    with runtime.scheduler_authority() as owns:
        assert owns
        assert runtime.prepare_recovery_registration(binding)


def fence(runtime, binding):
    with runtime.scheduler_authority() as owns:
        assert owns
        return fence_recovery_admission(runtime, binding)


def schema_path(cfg):
    return cfg.shared_root / "schema" / "version.json"


def registration_path(cfg):
    return cfg.shared_root / "machines" / cfg.machine_name / "registration.json"


def add_peer(cfg, tmp_path):
    peer_cfg = init_shared_root(cfg.shared_root, "machine-b", runtime_root=tmp_path / "legacy-b")
    runtime = MachineRuntime(tmp_path / "runtime-b")
    binding = runtime.add_binding(cfg.shared_root, "machine-b")
    return runtime, binding, peer_cfg


def test_all_prepared_participants_activate_without_invalidating_upgrade_metadata(project, tmp_path, monkeypatch):
    runtime, binding, cfg = project
    peer_runtime, peer_binding, peer_cfg = add_peer(cfg, tmp_path)
    prepare(runtime, binding)
    prepare(peer_runtime, peer_binding)
    manifest = cfg.shared_root / "operations" / "upgrades" / "protocol-manifest.json"
    original_manifest = manifest.read_bytes()
    barriers = []
    sync = DurableIO.sync_directory

    def observe_sync(self, directory, label):
        if label == "recovery_participant":
            assert LOCAL_RECOVERY_CAPABILITY not in read_json(schema_path(cfg))["schema"]["required_capabilities"]
            barriers.append(directory)
        return sync(self, directory, label)

    monkeypatch.setattr(DurableIO, "sync_directory", observe_sync)
    assert fence(runtime, binding).is_fenced
    assert set(barriers) == {registration_path(cfg).parent, registration_path(peer_cfg).parent}
    assert LOCAL_RECOVERY_CAPABILITY in read_json(schema_path(cfg))["schema"]["required_capabilities"]
    assert manifest.read_bytes() == original_manifest
    assert not UpgradeJournalMigration().is_applicable(cfg)
    status = UpgradeCoordinator(cfg).discover()
    assert not status["admission_blocked"] and not status["pending"]

    scandir = os.scandir

    def forbidden_enumeration(path):
        if Path(path) != cfg.shared_root:
            pytest.fail("durable admission reentry must not enumerate participants or execution history")
        return scandir(path)

    with monkeypatch.context() as guarded:
        guarded.setattr(recovery_admission.os, "scandir", forbidden_enumeration)
        assert fence(runtime, binding).is_fenced
        assert runtime.registration_status(binding)["write_eligible"]


@pytest.mark.parametrize("peer_state", ["eligible", "expired", "offline", "advertised"])
def test_unprepared_peer_is_not_retired_by_expiry_or_heartbeat(project, tmp_path, peer_state):
    from qqtools.plugins.qexp.commands.task import submit
    from qqtools.plugins.qexp.scheduler import claim_task

    runtime, binding, cfg = project
    _peer_runtime, _peer_binding, peer_cfg = add_peer(cfg, tmp_path)
    prepare(runtime, binding)
    path = registration_path(peer_cfg)
    record = read_json(path)
    if peer_state == "expired":
        record["registration"]["eligibility_expires_at"] = "2000-01-01T00:00:00Z"
        atomic_replace(path, record)
    if peer_state == "advertised":
        atomic_replace(path.parent / "state" / "agent.json", {"writer_capabilities": [LOCAL_RECOVERY_CAPABILITY]})
    original = schema_path(cfg).read_bytes()
    outcome = fence(runtime, binding)
    assert not outcome.is_fenced
    assert outcome.blockers == ("registration_not_prepared:machine-b",)
    assert schema_path(cfg).read_bytes() == original
    # Waiting for the peer does not block ordinary target-code scheduling.
    task = submit(cfg, ["true"])
    assert claim_task(cfg, task.task_id, [0]) is not None


def test_explicit_supersession_can_retire_a_version_one_participant(project, tmp_path):
    runtime, binding, cfg = project
    peer_runtime, peer_binding, peer_cfg = add_peer(cfg, tmp_path)
    prepare(runtime, binding)
    peer_runtime._supersede_registration(peer_binding, binding.registration_generation)
    assert read_json(registration_path(peer_cfg))["registration"]["version"] == 1
    assert fence(runtime, binding).is_fenced


@pytest.mark.parametrize("damage", ["missing", "version", "identity", "legacy", "supersession"])
def test_missing_or_invalid_peer_prevents_activation(project, tmp_path, damage):
    runtime, binding, cfg = project
    peer_runtime, peer_binding, peer_cfg = add_peer(cfg, tmp_path)
    prepare(runtime, binding)
    prepare(peer_runtime, peer_binding)
    path = registration_path(peer_cfg)
    record = read_json(path)
    if damage == "missing":
        path.unlink()
    elif damage == "legacy":
        metadata_path = path.parent / "machine.json"
        metadata = read_json(metadata_path)
        metadata["machine"]["agent_runtime"] = "project"
        atomic_replace(metadata_path, metadata)
    else:
        if damage == "version":
            record["registration"]["version"] = 3
        elif damage == "identity":
            record["registration"]["shared_root"] = "/wrong/project"
        else:
            record["registration"]["state"] = "superseded"
        atomic_replace(path, record)
    original = schema_path(cfg).read_bytes()
    outcome = fence(runtime, binding)
    assert not outcome.is_fenced and outcome.blockers
    assert schema_path(cfg).read_bytes() == original


@pytest.mark.parametrize(
    "standalone_root",
    [
        None,
        "",
        3,
        {},
        "relative/root",
        "/root/../legacy",
        "/root/./legacy",
        "/root//legacy",
        "/root/legacy/",
        "/root/\x00legacy",
    ],
)
def test_invalid_peer_standalone_root_prevents_activation(project, tmp_path, standalone_root):
    runtime, binding, cfg = project
    peer_runtime, peer_binding, peer_cfg = add_peer(cfg, tmp_path)
    prepare(runtime, binding)
    prepare(peer_runtime, peer_binding)
    path = registration_path(peer_cfg).parent / "machine.json"
    metadata = read_json(path)
    if standalone_root is None:
        metadata["machine"].pop("runtime_root")
    else:
        metadata["machine"]["runtime_root"] = standalone_root
    atomic_replace(path, metadata)
    original = schema_path(cfg).read_bytes()
    assert fence(runtime, binding).blockers == ("machine_not_prepared:machine-b",)
    assert schema_path(cfg).read_bytes() == original


def test_caller_needs_owned_prepared_registration_and_current_binding(project):
    runtime, binding, cfg = project
    with pytest.raises(RuntimeError, match="scheduler authority"):
        fence_recovery_admission(runtime, binding)
    assert fence(runtime, binding).blockers == ("local_registration_not_prepared",)
    prepare(runtime, binding)
    runtime.set_enabled(binding.project_id, False)
    assert fence(runtime, binding).blockers == ("binding_changed",)
    assert LOCAL_RECOVERY_CAPABILITY not in read_json(schema_path(cfg))["schema"]["required_capabilities"]


def test_schema_contention_defers_without_deadlocking_registered_claims(project):
    runtime, binding, cfg = project
    prepare(runtime, binding)
    with schema_lock(cfg.shared_root):
        assert fence(runtime, binding).blockers == ("schema_busy",)
    assert fence(runtime, binding).is_fenced


def test_activation_waits_for_existing_upgrade_metadata(project):
    runtime, binding, cfg = project
    prepare(runtime, binding)
    manifest = cfg.shared_root / "operations" / "upgrades" / "protocol-manifest.json"
    manifest.unlink()
    assert fence(runtime, binding).blockers == ("upgrade_manifest_not_ready",)
    coordinator = UpgradeCoordinator(cfg)
    status = coordinator.discover()
    for _ in range(6):
        if not status["pending"]:
            break
        status = coordinator.advance()
    assert status["state"] == "completed"
    assert fence(runtime, binding).is_fenced
    assert not coordinator.status()["admission_blocked"]


def test_participant_durability_failure_cannot_publish_capability(project, monkeypatch):
    runtime, binding, cfg = project
    prepare(runtime, binding)
    sync = DurableIO.sync_directory

    def failed_sync(self, directory, label):
        if label == "recovery_participant":
            raise OSError("participant durability failed")
        return sync(self, directory, label)

    monkeypatch.setattr(DurableIO, "sync_directory", failed_sync)
    with pytest.raises(OSError, match="participant durability"):
        fence(runtime, binding)
    assert LOCAL_RECOVERY_CAPABILITY not in read_json(schema_path(cfg))["schema"]["required_capabilities"]
    monkeypatch.setattr(DurableIO, "sync_directory", sync)
    assert fence(runtime, binding).is_fenced


def test_uncertain_capability_publication_retries_barrier_without_enumeration(project, monkeypatch):
    runtime, binding, cfg = project
    prepare(runtime, binding)
    original = recovery_admission.atomic_replace

    def uncertain(path, value):
        original(path, value)
        raise OSError("uncertain capability durability")

    monkeypatch.setattr(recovery_admission, "atomic_replace", uncertain)
    with pytest.raises(OSError, match="uncertain capability"):
        fence(runtime, binding)
    assert LOCAL_RECOVERY_CAPABILITY in read_json(schema_path(cfg))["schema"]["required_capabilities"]
    sync = DurableIO.sync_directory
    barriers = []

    def failed_barrier(self, directory, label):
        if label == "recovery_admission":
            barriers.append(directory)
            raise OSError("retry barrier failed")
        return sync(self, directory, label)

    monkeypatch.setattr(DurableIO, "sync_directory", failed_barrier)
    with pytest.raises(OSError, match="retry barrier"):
        fence(runtime, binding)
    assert barriers == [schema_path(cfg).parent]
    monkeypatch.setattr(DurableIO, "sync_directory", sync)
    assert fence(runtime, binding).is_fenced


def test_crash_after_capability_commit_reenters_without_damaging_upgrade_state(project):
    runtime, binding, cfg = project
    prepare(runtime, binding)
    pid = os.fork()
    if pid == 0:
        original = recovery_admission.atomic_replace

        def crash(path, value):
            original(path, value)
            os._exit(43)

        recovery_admission.atomic_replace = crash
        try:
            fence(runtime, binding)
        finally:
            os._exit(44)
    assert os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]) == 43
    restarted = MachineRuntime(runtime.root)
    assert fence(restarted, binding).is_fenced
    assert not UpgradeJournalMigration().is_applicable(cfg)
    assert not UpgradeCoordinator(cfg).discover()["pending"]


@pytest.mark.parametrize("damage", ["base_capability", "unknown_capability", "created_at"])
def test_recovery_extension_does_not_hide_unrelated_schema_drift(project, damage):
    runtime, binding, cfg = project
    prepare(runtime, binding)
    assert fence(runtime, binding).is_fenced
    path = schema_path(cfg)
    document = read_json(path)
    if damage == "base_capability":
        document["schema"]["required_capabilities"].remove("cpu-lane-v1")
    elif damage == "unknown_capability":
        document["schema"]["required_capabilities"].append("unapproved-capability")
    else:
        document["schema"]["created_at"] = "changed"
    atomic_replace(path, document)
    assert UpgradeJournalMigration().is_applicable(cfg)


def test_upgrade_audit_still_fingerprints_full_schema_before_activation(project):
    _runtime, _binding, cfg = project
    manifest = cfg.shared_root / "operations" / "upgrades" / "protocol-manifest.json"
    manifest.unlink()
    coordinator = UpgradeCoordinator(cfg)
    coordinator.discover()
    assert coordinator.advance()["phase"] == "audit"
    assert coordinator.advance()["phase"] == "activation"
    # Out-of-protocol modification, unlike the fence's metadata-ready prerequisite.
    document = read_json(schema_path(cfg))
    document["schema"]["required_capabilities"].append(LOCAL_RECOVERY_CAPABILITY)
    atomic_replace(schema_path(cfg), document)
    status = coordinator.advance()
    assert status["state"] == "repair_required"
    assert "schema version record changed after audit" in status["blockers"][0]


def test_pending_upgrade_phase_defers_root_fence_even_after_metadata_activation(project):
    runtime, binding, cfg = project
    prepare(runtime, binding)
    manifest = cfg.shared_root / "operations" / "upgrades" / "protocol-manifest.json"
    manifest.unlink()
    coordinator = UpgradeCoordinator(cfg)
    coordinator.discover()
    for expected in ("audit", "activation", "contraction"):
        assert coordinator.advance()["phase"] == expected
    assert not UpgradeJournalMigration().is_applicable(cfg)
    assert fence(runtime, binding).blockers == ("upgrade_coordinator_pending",)
    assert coordinator.advance()["state"] == "completed"
    assert fence(runtime, binding).is_fenced
