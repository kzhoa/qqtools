"""Shared recovery fences remain independent of machine-local capture authority."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.layout import LOCAL_RECOVERY_CAPABILITY, load_machine_registration
from qqtools.plugins.qexp.runtime import registration_authority
from qqtools.plugins.qexp.runtime.recovery_admission import fence_shared_recovery_admission
from qqtools.plugins.qexp.runtime.registration_authority import (
    RegistrationIdentity,
    publish_recovery_registration_locked,
    registration_write_guard,
)
from qqtools.plugins.qexp.runtime.responsibility_store import DurableIO
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


@pytest.fixture
def registered(tmp_path):
    cfg = init_shared_root(tmp_path / "project/.qexp", "gpu-1", runtime_root=tmp_path / "legacy")
    runtime = MachineRuntime(tmp_path / "machine")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    identity = RegistrationIdentity(
        binding.project_id, binding.registration_generation, binding.runtime_instance_id, str(runtime.root)
    )
    return cfg, runtime, binding, identity


def _prepare(cfg, identity, *, before_shared_write=None):
    with registration_write_guard(cfg, identity, is_current=lambda: True) as registration:
        assert registration is not None
        publish_recovery_registration_locked(cfg, registration, before_shared_write=before_shared_write)


def _schema(cfg):
    return cfg.shared_root / "schema/version.json"


def test_shared_preparation_and_admission_need_no_machine_owner_or_local_writes(registered, monkeypatch):
    cfg, runtime, binding, identity = registered
    original = load_machine_registration(cfg)["registration"]
    with runtime.registry_guard(), runtime.migration_read_guard(), runtime.binding_commit_guard(binding):
        local_before = {
            path.relative_to(runtime.root): path.read_bytes() for path in runtime.root.rglob("*") if path.is_file()
        }

        def forbidden(*_args, **_kwargs):
            pytest.fail("shared recovery transaction used MachineRuntime authority")

        with monkeypatch.context() as guarded:
            guarded.setattr(MachineRuntime, "__init__", forbidden)
            guarded.setattr(MachineRuntime, "registry_guard", forbidden)
            guarded.setattr(MachineRuntime, "binding_write_guard", forbidden)
            guarded.setattr(MachineRuntime, "registration_status", forbidden)
            _prepare(cfg, identity)
            assert fence_shared_recovery_admission(cfg, identity, is_current=lambda: True).is_fenced
        assert local_before == {
            path.relative_to(runtime.root): path.read_bytes() for path in runtime.root.rglob("*") if path.is_file()
        }
    prepared = load_machine_registration(cfg)["registration"]
    assert prepared["version"] == 2
    assert prepared["generation"] == original["generation"]
    assert prepared["runtime_instance_id"] == original["runtime_instance_id"]
    assert LOCAL_RECOVERY_CAPABILITY in read_json(_schema(cfg))["schema"]["required_capabilities"]


@pytest.mark.parametrize("damage", ["generation", "runtime", "root", "revoked"])
def test_shared_admission_requires_exact_captured_registration(registered, damage):
    cfg, _runtime, _binding, identity = registered
    _prepare(cfg, identity)
    expected = _schema(cfg).read_bytes()
    if damage == "generation":
        identity = replace(identity, generation="other-generation")
    elif damage == "runtime":
        identity = replace(identity, runtime_id="other-runtime")
    elif damage == "root":
        identity = replace(identity, runtime_root=identity.runtime_root + "-replacement")
    outcome = fence_shared_recovery_admission(cfg, identity, is_current=lambda: damage != "revoked")
    assert not outcome.is_fenced
    assert outcome.blockers == ("registration_ineligible",)
    assert _schema(cfg).read_bytes() == expected


@pytest.mark.parametrize("capability_visible", [False, True])
def test_visible_capability_does_not_authorize_unprepared_registration(registered, capability_visible):
    cfg, _runtime, _binding, identity = registered
    if capability_visible:
        schema = read_json(_schema(cfg))
        schema["schema"]["required_capabilities"].append(LOCAL_RECOVERY_CAPABILITY)
        atomic_replace(_schema(cfg), schema)
    before = _schema(cfg).read_bytes()
    outcome = fence_shared_recovery_admission(cfg, identity, is_current=lambda: True)
    assert not outcome.is_fenced
    assert outcome.blockers == ("local_registration_not_prepared",)
    assert _schema(cfg).read_bytes() == before
    assert load_machine_registration(cfg)["registration"]["version"] == 1


@pytest.mark.parametrize("boundary", [1, 2])
def test_shared_admission_fences_participant_barrier_and_capability_replace(registered, boundary):
    cfg, _runtime, _binding, identity = registered
    _prepare(cfg, identity)
    before = _schema(cfg).read_bytes()
    calls = []

    def before_shared_write():
        calls.append(None)
        if len(calls) == boundary:
            raise RuntimeError("executor epoch revoked")

    with pytest.raises(RuntimeError, match="epoch revoked"):
        fence_shared_recovery_admission(cfg, identity, is_current=lambda: True, before_shared_write=before_shared_write)
    assert len(calls) == boundary
    assert _schema(cfg).read_bytes() == before
    assert fence_shared_recovery_admission(cfg, identity, is_current=lambda: True).is_fenced


def test_shared_preparation_and_visible_capability_replay_recheck_fences(registered):
    cfg, _runtime, _binding, identity = registered
    original = load_machine_registration(cfg)

    def revoked():
        raise RuntimeError("executor epoch revoked")

    with pytest.raises(RuntimeError, match="epoch revoked"):
        _prepare(cfg, identity, before_shared_write=revoked)
    assert load_machine_registration(cfg) == original
    _prepare(cfg, identity)
    prepared = load_machine_registration(cfg)
    with pytest.raises(RuntimeError, match="epoch revoked"):
        _prepare(cfg, identity, before_shared_write=revoked)
    assert load_machine_registration(cfg) == prepared
    assert fence_shared_recovery_admission(cfg, identity, is_current=lambda: True).is_fenced
    committed = _schema(cfg).read_bytes()
    with pytest.raises(RuntimeError, match="epoch revoked"):
        fence_shared_recovery_admission(cfg, identity, is_current=lambda: True, before_shared_write=revoked)
    assert _schema(cfg).read_bytes() == committed


def test_shared_admission_rechecks_local_fence_after_peer_inventory(registered):
    cfg, _runtime, _binding, identity = registered
    _prepare(cfg, identity)
    calls = []

    def is_current():
        calls.append(None)
        # The guard checks on entry and before yielding. Revoke at the final
        # capability check, after inventory and participant durability.
        return len(calls) < 4

    before = _schema(cfg).read_bytes()
    outcome = fence_shared_recovery_admission(cfg, identity, is_current=is_current)
    assert not outcome.is_fenced
    assert outcome.blockers == ("registration_ineligible",)
    assert len(calls) == 4
    assert _schema(cfg).read_bytes() == before


def test_registration_expiring_during_inventory_cannot_publish_capability(registered, monkeypatch):
    cfg, _runtime, _binding, identity = registered
    _prepare(cfg, identity)
    expired = [False]

    class AdvancingClock(datetime):
        @classmethod
        def now(cls, tz=None):
            current = datetime.now(tz)
            return current + timedelta(days=1) if expired[0] else current

    sync = DurableIO.sync_directory

    def expire_after_barrier(self, directory, label):
        sync(self, directory, label)
        if label == "recovery_participant":
            expired[0] = True

    monkeypatch.setattr(registration_authority, "datetime", AdvancingClock)
    monkeypatch.setattr(DurableIO, "sync_directory", expire_after_barrier)
    before = _schema(cfg).read_bytes()
    outcome = fence_shared_recovery_admission(cfg, identity, is_current=lambda: True)
    assert expired[0]
    assert not outcome.is_fenced
    assert outcome.blockers == ("registration_ineligible",)
    assert _schema(cfg).read_bytes() == before
