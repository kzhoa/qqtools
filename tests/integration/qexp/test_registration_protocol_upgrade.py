"""QQTOOLS-COMPAT-0024: registration conversion preserves authority and replay fences."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.layout import load_machine_registration
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


@pytest.mark.parametrize("source_version", [1, 2])
def test_released_source_artifact_records_convert_without_identity_loss(tmp_path, source_version):
    from qqtools.plugins.qexp.layout import load_root_config
    from qqtools.plugins.qexp.runtime.registration_protocol import convert_registration

    path = Path(__file__).parents[2] / "fixtures" / "qexp_registration_source_1_3_27.json"
    source = json.loads(path.read_text().replace("${FIXTURE_ROOT}", str(tmp_path)))
    assert source["wheel_sha256"] == "1c996115980d188858f87766455166059210f0ee51be5fb5e8d07481dcc1e4e7"
    record = source[f"registration_v{source_version}"]["registration"]
    cfg = load_root_config(record["shared_root"], record["machine_name"])
    converted = convert_registration(record, record["project_id"], cfg)
    assert converted["version"] == 3
    for field in record.keys() - {"version", "protocol_version", "recovery_protocol"}:
        assert converted[field] == record[field]
    assert converted["recovery_protocol"] == record.get("recovery_protocol")


@pytest.fixture
def registered(tmp_path):
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "old", runtime_root=tmp_path / "legacy")
    runtime = MachineRuntime(tmp_path / "machine")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    return runtime, binding, cfg


def source_record(record, version):
    value = deepcopy(record)
    value.update(version=version, protocol_version=version)
    value.pop("registration_protocol", None)
    if version == 1:
        value.pop("recovery_protocol", None)
    else:
        value["recovery_protocol"] = "qexp-local-responsibility-v1"
    return value


@pytest.mark.parametrize("version", [1, 2])
def test_conversion_preserves_identity_and_does_not_grant_recovery(registered, version):
    from qqtools.plugins.qexp.runtime.registration_protocol import convert_registration, has_recovery_protocol

    _runtime, binding, cfg = registered
    original = source_record(load_machine_registration(cfg)["registration"], version)
    untouched = deepcopy(original)
    converted = convert_registration(original, binding.project_id, cfg)
    assert original == untouched
    assert converted["version"] == converted["protocol_version"] == 3
    assert converted["registration_protocol"] == "qexp-registration-replacement-v1"
    for field in (
        "project_id",
        "shared_root",
        "machine_name",
        "generation",
        "runtime_instance_id",
        "runtime_root",
        "state",
        "eligibility_expires_at",
        "created_at",
        "updated_at",
    ):
        assert converted[field] == original[field]
    assert has_recovery_protocol(converted) is (version == 2)
    assert convert_registration(converted, binding.project_id, cfg) == converted


@pytest.mark.parametrize(
    "changes",
    [
        {"version": 99},
        {"version": True},
        {"project_id": "different"},
        {"machine_name": "different"},
        {"eligibility_expires_at": "not-a-time"},
        {"state": "reserved"},
        {"recovery_protocol": "unknown"},
    ],
)
def test_invalid_target_never_reinterpreted_as_source(registered, changes):
    from qqtools.plugins.qexp.runtime.registration_protocol import convert_registration

    _runtime, binding, cfg = registered
    record = load_machine_registration(cfg)["registration"]
    record.update(changes)
    before = deepcopy(record)
    with pytest.raises((RuntimeError, ValueError)):
        convert_registration(record, binding.project_id, cfg)
    assert record == before


def test_recovery_preparation_keeps_target_envelope_and_generation(registered):
    runtime, binding, cfg = registered
    original = load_machine_registration(cfg)["registration"]
    assert original["version"] == 3
    assert original["recovery_protocol"] is None
    with runtime.scheduler_authority() as acquired:
        assert acquired
        assert runtime.prepare_recovery_registration(binding)
    prepared = load_machine_registration(cfg)["registration"]
    assert prepared["version"] == prepared["protocol_version"] == 3
    assert prepared["generation"] == original["generation"]
    assert prepared["recovery_protocol"] == "qexp-local-responsibility-v1"
    assert runtime.registration_status(binding)["write_eligible"]


@pytest.mark.parametrize("source_version", [1, 2])
def test_pending_source_rollback_converts_snapshot_before_replay(registered, source_version):
    runtime, binding, cfg = registered
    snapshot = {"registration": source_record(load_machine_registration(cfg)["registration"], source_version)}
    revision, bindings = runtime.load_registry()
    with runtime.registry_guard():
        runtime.registration._save_registration_transaction(
            revision=revision, bindings=bindings, registrations=[(cfg, snapshot)], machine_records=[]
        )
        journal = read_json(runtime.paths["registration_transaction"])
        journal["registration_transaction"].pop("version", None)
        atomic_replace(runtime.paths["registration_transaction"], journal)
        runtime.registration.rollback_pending_locked()
    stored = load_machine_registration(cfg)["registration"]
    assert stored["version"] == 3
    assert stored["generation"] == binding.registration_generation
    assert not runtime.paths["registration_transaction"].exists()
    assert read_json(runtime.paths["registry"])["registry"]["version"] == 2


@pytest.mark.parametrize("has_snapshot", [False, True])
def test_pending_source_rollback_cannot_overwrite_successor(registered, has_snapshot):
    runtime, _binding, cfg = registered
    snapshot = {"registration": source_record(load_machine_registration(cfg)["registration"], 1)}
    revision, bindings = runtime.load_registry()
    with runtime.registry_guard():
        runtime.registration._save_registration_transaction(
            revision=revision,
            bindings=bindings,
            registrations=[(cfg, snapshot if has_snapshot else None)],
            machine_records=[],
        )
        successor = deepcopy(load_machine_registration(cfg))
        successor["registration"].update(runtime_instance_id="other-runtime", generation="successor")
        path = cfg.shared_root / "machines" / cfg.machine_name / "registration.json"
        atomic_replace(path, successor)
        before = path.read_bytes()
        with pytest.raises(RuntimeError):
            runtime.registration.rollback_pending_locked()
        assert path.read_bytes() == before
        assert runtime.paths["registration_transaction"].exists()


def test_source_journal_after_registry_commit_preserves_published_generation(registered):
    runtime, binding, cfg = registered
    snapshot = {"registration": source_record(load_machine_registration(cfg)["registration"], 1)}
    revision, bindings = runtime.load_registry()
    with runtime.registry_guard():
        runtime.registration._save_registration_transaction(
            revision=revision, bindings=bindings, registrations=[(cfg, snapshot)], machine_records=[]
        )
        journal = read_json(runtime.paths["registration_transaction"])
        journal["registration_transaction"].pop("version", None)
        atomic_replace(runtime.paths["registration_transaction"], journal)
        committed = replace(binding, registration_generation="committed-generation")
        shared = deepcopy(snapshot)
        shared["registration"]["generation"] = committed.registration_generation
        atomic_replace(cfg.shared_root / "machines" / cfg.machine_name / "registration.json", shared)
        runtime.registration.save_registry_locked(revision + 1, [committed])
        runtime.registration.rollback_pending_locked()
    assert runtime.load_registry()[1] == [committed]
    assert load_machine_registration(cfg)["registration"]["generation"] == committed.registration_generation
    assert load_machine_registration(cfg)["registration"]["version"] == 3
    assert not runtime.paths["registration_transaction"].exists()


def legacy_project(registered, version):
    _runtime, _binding, cfg = registered
    schema_path = cfg.shared_root / "schema" / "version.json"
    schema = read_json(schema_path)
    schema["schema"]["required_capabilities"] = [
        item for item in schema["schema"]["required_capabilities"] if item != "registration-replacement-v1"
    ]
    atomic_replace(schema_path, schema)
    path = cfg.shared_root / "machines" / cfg.machine_name / "registration.json"
    record = source_record(load_machine_registration(cfg)["registration"], version)
    atomic_replace(path, {"registration": record})
    return cfg, path, record


def registration_coordinator(cfg):
    from qqtools.plugins.qexp.runtime.upgrade import MigrationRegistry, UpgradeCoordinator
    from qqtools.plugins.qexp.runtime.upgrade.registration_v3 import RegistrationV3Migration

    return UpgradeCoordinator(cfg, registry=MigrationRegistry([RegistrationV3Migration()]))


@pytest.mark.parametrize("version", [1, 2])
def test_coordinator_converts_source_then_audits_before_activation(registered, version):
    cfg, path, original = legacy_project(registered, version)
    coordinator = registration_coordinator(cfg)
    seen = set()
    for _ in range(20):
        status = coordinator.advance()
        migration = status["migrations"][0]
        seen.add(migration["phase"])
        if not status["pending"]:
            break
    assert not status["pending"], status
    assert {"backfill", "audit", "activation"}.issubset(seen)
    assert (
        "registration-replacement-v1"
        in read_json(cfg.shared_root / "schema" / "version.json")["schema"]["required_capabilities"]
    )
    converted = read_json(path)["registration"]
    assert converted["version"] == 3
    assert converted["generation"] == original["generation"]
    assert converted["recovery_protocol"] == original.get("recovery_protocol")
    assert registration_coordinator(cfg).status()["pending"] is False


def test_malformed_source_blocks_without_activating_or_deleting_evidence(registered):
    cfg, path, original = legacy_project(registered, 1)
    original["version"] = 99
    atomic_replace(path, {"registration": original})
    before = path.read_bytes()
    coordinator = registration_coordinator(cfg)
    for _ in range(5):
        status = coordinator.advance()
        if status["state"] == "repair_required":
            break
    assert status["pending"]
    assert status["migration_blocked"]
    assert path.read_bytes() == before
    assert (
        "registration-replacement-v1"
        not in read_json(cfg.shared_root / "schema" / "version.json")["schema"]["required_capabilities"]
    )


def test_new_root_is_target_ready_without_source_conversion(registered):
    from qqtools.plugins.qexp.runtime.upgrade.registration_v3 import RegistrationV3Migration

    _runtime, _binding, cfg = registered
    assert not RegistrationV3Migration().is_applicable(cfg)
    assert load_machine_registration(cfg)["registration"]["version"] == 3


@pytest.mark.parametrize("phase", ["backfill", "activation"])
@pytest.mark.parametrize("after_write", [False, True])
def test_shared_conversion_resumes_after_durable_write_interruption(registered, monkeypatch, phase, after_write):
    from qqtools.plugins.qexp.runtime.upgrade.contracts import UpgradeStorage

    cfg, path, original = legacy_project(registered, 2)
    coordinator = registration_coordinator(cfg)
    for _ in range(20):
        status = coordinator.advance(force_retry=True)
        if status["phase"] == phase:
            break
    assert status["phase"] == phase
    target = path if phase == "backfill" else cfg.shared_root / "schema" / "version.json"
    publish = UpgradeStorage.atomic_replace

    class Interrupted(BaseException):
        pass

    def interrupt(storage, destination, value, **kwargs):
        if destination == target:
            if after_write:
                publish(storage, destination, value, **kwargs)
            raise Interrupted
        return publish(storage, destination, value, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(UpgradeStorage, "atomic_replace", interrupt)
        with pytest.raises(Interrupted):
            coordinator.advance(force_retry=True)
    restarted = registration_coordinator(cfg)
    for _ in range(25):
        status = restarted.advance(force_retry=True)
        if not status["pending"]:
            break
    assert not status["pending"], status
    converted = read_json(path)["registration"]
    assert converted["version"] == 3
    assert converted["generation"] == original["generation"]
    assert converted["recovery_protocol"] == original["recovery_protocol"]


def test_registration_lock_contention_keeps_cursor_and_retries(registered):
    from qqtools.plugins.qexp.runtime.locks import exclusive

    cfg, path, _original = legacy_project(registered, 1)
    coordinator = registration_coordinator(cfg)
    assert coordinator.advance()["phase"] == "backfill"
    before = path.read_bytes()
    with exclusive(cfg.shared_root / "locks" / "registrations.lock"):
        waiting = coordinator.advance()
    assert waiting["pending"]
    assert "registration_lock_busy" in waiting["blockers"]
    assert path.read_bytes() == before
    for _ in range(20):
        status = coordinator.advance(force_retry=True)
        if not status["pending"]:
            break
    assert not status["pending"], status


def test_temporarily_unavailable_registration_retries_without_skipping(registered, monkeypatch):
    from qqtools.plugins.qexp.runtime.upgrade.contracts import UpgradeStorage

    cfg, path, _original = legacy_project(registered, 1)
    coordinator = registration_coordinator(cfg)
    assert coordinator.advance()["phase"] == "backfill"
    read = UpgradeStorage.read_json_limited

    def unavailable(storage, source, **kwargs):
        if source == path:
            raise PermissionError("temporarily unavailable")
        return read(storage, source, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(UpgradeStorage, "read_json_limited", unavailable)
        status = coordinator.advance()
    assert status["pending"]
    assert "registration_storage_unavailable" in status["blockers"]
    assert read_json(path)["registration"]["version"] == 1
    for _ in range(20):
        status = coordinator.advance(force_retry=True)
        if not status["pending"]:
            break
    assert not status["pending"], status
    assert read_json(path)["registration"]["version"] == 3


def test_first_registration_rollback_retries_after_registry_restore(tmp_path, monkeypatch):
    import os

    cfg = init_shared_root(tmp_path / "project" / ".qexp", "old", runtime_root=tmp_path / "legacy")
    runtime = MachineRuntime(tmp_path / "machine")
    runtime.ensure_layout()
    revision, bindings = runtime.load_registry()
    assert bindings == []
    journal = runtime.paths["registration_transaction"]
    runtime.registration._save_registration_transaction(
        revision=revision, bindings=bindings, registrations=[(cfg, None)], machine_records=[]
    )
    unlink = os.unlink

    class Interrupted(BaseException):
        pass

    def interrupt(path, *args, **kwargs):
        if Path(path) == journal:
            raise Interrupted
        return unlink(path, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "unlink", interrupt)
        with runtime.registry_guard(), pytest.raises(Interrupted):
            runtime.registration.rollback_pending_locked()
    assert runtime.load_registry()[1] == []
    assert journal.exists()
    with runtime.registry_guard():
        runtime.registration.rollback_pending_locked()
    assert not journal.exists()
    assert runtime.load_registry()[1] == []


def test_backfill_retries_directory_barrier_after_visible_conversion(registered, monkeypatch):
    from qqtools.plugins.qexp.runtime.upgrade.contracts import UpgradeStorage

    cfg, path, _original = legacy_project(registered, 1)
    coordinator = registration_coordinator(cfg)
    assert coordinator.advance()["phase"] == "backfill"
    publish = UpgradeStorage.atomic_replace
    calls = []

    def failed_barrier(storage, destination, value, **kwargs):
        if destination == path:
            # Model rename visibility followed by failed directory durability.
            path.write_text(json.dumps(value))
            raise OSError("directory barrier failed")
        return publish(storage, destination, value, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(UpgradeStorage, "atomic_replace", failed_barrier)
        assert coordinator.advance()["pending"]
    assert read_json(path)["registration"]["version"] == 3

    def observe(storage, destination, value, **kwargs):
        calls.append(destination)
        return publish(storage, destination, value, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(UpgradeStorage, "atomic_replace", observe)
        coordinator.advance(force_retry=True)
    assert path in calls


def test_nonregular_registration_fails_without_blocking_open(registered, monkeypatch):
    import os

    cfg, path, _original = legacy_project(registered, 1)
    coordinator = registration_coordinator(cfg)
    assert coordinator.advance()["phase"] == "backfill"
    path.unlink()
    os.mkfifo(path)
    real_open = os.open

    def bounded_open(source, flags, *args, **kwargs):
        if Path(source) == path:
            assert flags & os.O_NONBLOCK, "a FIFO open must not hang the upgrade worker"
        return real_open(source, flags, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "open", bounded_open)
        status = coordinator.advance()
    assert status["pending"]
    assert status["state"] == "repair_required", status
    assert path.is_fifo()


def test_real_agent_automatically_prepares_source_registry_and_shared_projects(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.agent.lifecycle import start_machine_agent, stop_machine_agent
    from qqtools.plugins.qexp.agent.setup import initialize_machine, initialize_project, register_projects
    from qqtools.plugins.qexp.layout import load_root_config
    from qqtools.plugins.qexp.runtime.upgrade import UpgradeCoordinator
    from tests.helpers.qexp.lifecycle import wait_until

    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "old", agent_mode="daemon")
    configs = []
    originals = []
    for version in (1, 2):
        root = tmp_path / f"project-{version}"
        initialize_project(root)
        register_projects(runtime, [root], machine_name="old")
        cfg = load_root_config(root / ".qexp", "old")
        binding = runtime.resolve_project_binding(cfg.shared_root)
        cfg, _path, original = legacy_project((runtime, binding, cfg), version)
        configs.append(cfg)
        originals.append(original)
    registry = read_json(runtime.paths["registry"])
    registry["registry"]["version"] = 1
    atomic_replace(runtime.paths["registry"], registry)
    original_id = runtime.instance_id
    monkeypatch.setenv("QEXP_VISIBLE_GPUS", "")
    process = start_machine_agent(runtime, available_gpus=[], loop_interval=0.05)
    try:

        def completed():
            return all(
                "registration-replacement-v1"
                in read_json(cfg.shared_root / "schema/version.json")["schema"]["required_capabilities"]
                and not UpgradeCoordinator(cfg).status()["pending"]
                for cfg in configs
            )

        wait_until(
            "automatic-registration-conversion",
            completed,
            timeout=60,
            on_timeout=lambda: [UpgradeCoordinator(cfg).status() for cfg in configs],
        )
        assert runtime.instance_id == original_id
        assert read_json(runtime.paths["registry"])["registry"]["version"] == 2
        for cfg, before in zip(configs, originals, strict=True):
            after = load_machine_registration(cfg)["registration"]
            assert after["version"] == 3
            for key in ("generation", "project_id", "runtime_instance_id", "runtime_root", "machine_name"):
                assert before[key] == after[key]
    finally:
        stop_machine_agent(runtime, timeout=10)
        process.wait(timeout=10)


def test_migration_rejects_symlink_machine_directory(registered, tmp_path):
    cfg, path, _original = legacy_project(registered, 1)
    coordinator = registration_coordinator(cfg)
    assert coordinator.advance()["phase"] == "backfill"
    moved = tmp_path / "moved-machine"
    path.parent.rename(moved)
    path.parent.symlink_to(moved, target_is_directory=True)
    before = (moved / "registration.json").read_bytes()
    status = coordinator.advance()
    assert status["state"] == "repair_required"
    assert (moved / "registration.json").read_bytes() == before


def test_machine_enumeration_does_not_stat_every_directory(registered, monkeypatch):
    from qqtools.plugins.qexp.runtime.upgrade import registration_v3

    cfg, _path, _original = legacy_project(registered, 1)
    coordinator = registration_coordinator(cfg)
    assert coordinator.advance()["phase"] == "backfill"
    scandir = registration_v3.os.scandir

    class NamesOnly:
        def __init__(self, path):
            self.source = scandir(path)

        def __enter__(self):
            return (type("Entry", (), {"name": entry.name})() for entry in self.source)

        def __exit__(self, *_args):
            self.source.close()

    with monkeypatch.context() as scoped:
        scoped.setattr(registration_v3.os, "scandir", NamesOnly)
        status = coordinator.advance()
    assert status["state"] == "runnable", status
    assert status["migrations"][0]["last_slice_usage"]["metadata_ops"] <= 32


def test_staggered_runtime_coordinators_share_one_conversion_journal(registered, tmp_path):
    from qqtools.plugins.qexp.layout import load_root_config
    from qqtools.plugins.qexp.runtime.upgrade import MigrationRegistry, UpgradeCoordinator
    from qqtools.plugins.qexp.runtime.upgrade.registration_v3 import RegistrationV3Migration

    runtime, old, cfg = registered
    peer = MachineRuntime(tmp_path / "peer")
    peer_binding = peer.add_binding(cfg.shared_root, "peer")
    peer_cfg = load_root_config(cfg.shared_root, "peer")
    _cfg, _path, old_before = legacy_project(registered, 1)
    _peer_cfg, _peer_path, peer_before = legacy_project((peer, peer_binding, peer_cfg), 2)
    first = UpgradeCoordinator(
        cfg, registry=MigrationRegistry([RegistrationV3Migration()]), holder_id=runtime.instance_id
    )
    second = UpgradeCoordinator(
        peer_cfg, registry=MigrationRegistry([RegistrationV3Migration()]), holder_id=peer.instance_id
    )
    assert first.advance()["phase"] == "backfill"
    assert first.advance()["pending"]
    for index in range(25):
        current = second if index % 2 == 0 else first
        status = current.advance(force_retry=True)
        if not status["pending"]:
            break
    assert not status["pending"], status
    assert not first.status()["pending"] and not second.status()["pending"]
    assert first.status()["migrations"] == second.status()["migrations"]
    for config, before, binding in ((cfg, old_before, old), (peer_cfg, peer_before, peer_binding)):
        after = load_machine_registration(config)["registration"]
        assert after["generation"] == before["generation"] == binding.registration_generation
        assert after["runtime_instance_id"] == before["runtime_instance_id"]
        assert after["recovery_protocol"] == before.get("recovery_protocol")
