"""Registration write scheduling must retain fresh fenced authorization."""

import time
from datetime import datetime, timedelta, timezone

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.lease import LeasePolicy, save_lease_policy
from qqtools.plugins.qexp.runtime import registration_authority
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


@pytest.fixture
def registration(tmp_path, monkeypatch):
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy")
    runtime = MachineRuntime(tmp_path / "machine")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    path = cfg.shared_root / "machines" / cfg.machine_name / "registration.json"
    now = [datetime(2026, 9, 18, tzinfo=timezone.utc)]

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now[0]

    monkeypatch.setattr(registration_authority, "datetime", FixedDateTime)
    monkeypatch.setattr(
        registration_authority,
        "lease_expiry",
        lambda policy: (now[0] + timedelta(seconds=policy.ttl_seconds)).replace(microsecond=0).isoformat(),
    )
    value = read_json(path)
    value["registration"]["eligibility_expires_at"] = (now[0] + timedelta(seconds=120)).isoformat()
    atomic_replace(path, value)
    publications = []
    save = registration_authority.save_machine_registration

    def record_save(cfg, value):
        publications.append(value)
        save(cfg, value)

    monkeypatch.setattr(registration_authority, "save_machine_registration", record_save)
    return cfg, runtime, binding, path, now, publications


def test_only_due_guard_publishes_and_restart_uses_durable_deadline(registration):
    cfg, runtime, binding, path, now, publications = registration
    before = path.read_bytes()
    for _ in range(20):
        assert runtime.binding_write_eligible(binding, renew=True)
    assert publications == []
    assert path.read_bytes() == before
    now[0] += timedelta(seconds=10)
    assert runtime.binding_write_eligible(binding, renew=True)
    assert len(publications) == 1
    restarted = MachineRuntime(runtime.root)
    for _ in range(20):
        assert restarted.binding_write_eligible(binding, renew=True)
    assert len(publications) == 1
    now[0] += timedelta(seconds=10)
    assert restarted.binding_write_eligible(binding, renew=True)
    assert len(publications) == 2


@pytest.mark.parametrize("running_version", ["1.3.24", "1.3.20"])
def test_version_change_forces_one_guarded_publication_before_lease_is_due(registration, monkeypatch, running_version):
    _cfg, runtime, binding, path, now, publications = registration
    value = read_json(path)
    record = value["registration"]
    record["client_version"] = "1.3.22"
    record["updated_at"] = "2026-09-17T23:59:00Z"
    preserved = {
        key: record[key]
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
    atomic_replace(path, value)
    previous_expiry = record["eligibility_expires_at"]
    previous_updated_at = record["updated_at"]
    now[0] += timedelta(seconds=1)
    monkeypatch.setattr(registration_authority, "__version__", running_version)

    assert runtime.binding_write_eligible(binding, renew=True)

    refreshed = read_json(path)["registration"]
    assert refreshed["client_version"] == running_version
    assert datetime.fromisoformat(refreshed["eligibility_expires_at"]) > datetime.fromisoformat(previous_expiry)
    assert refreshed["updated_at"] != previous_updated_at
    assert {key: refreshed[key] for key in preserved} == preserved
    assert len(publications) == 1

    for _ in range(20):
        assert runtime.binding_write_eligible(binding, renew=True)
    assert len(publications) == 1


def test_expired_idle_registration_reactivates_after_agent_restart(registration):
    from qqtools.plugins.qexp.agent.control_plane import _MachineControlPlane

    _, runtime, binding, path, now, _publications = registration
    value = read_json(path)
    value["registration"]["eligibility_expires_at"] = (now[0] - timedelta(seconds=1)).isoformat()
    atomic_replace(path, value)

    restarted = MachineRuntime(runtime.root)
    restarted_binding = restarted.load_registry()[1][0]
    plane = _MachineControlPlane(
        restarted,
        instance_id="test",
        loop_interval=5,
        started_at=now[0].isoformat(),
        available_gpus=[],
    )
    try:
        assert restarted.registration_status(restarted_binding)["state"] == "expired"
        plane._run_authority_cycle()
        status = restarted.registration_status(restarted_binding)
        assert status["state"] == "eligible"
        assert status["generation"] == binding.registration_generation
        assert plane._authority_snapshot["projects"][0]["observation_status"] == "tick_returned"
    finally:
        plane.stop()


@pytest.mark.parametrize("mutation", ["generation", "runtime_instance_id", "runtime_root", "expired", "malformed"])
def test_fresh_guard_rejects_changed_authority_without_publication(registration, mutation):
    _, runtime, binding, path, now, publications = registration
    assert runtime.binding_write_eligible(binding, renew=True)
    value = read_json(path)
    record = value["registration"]
    if mutation == "expired":
        record["eligibility_expires_at"] = now[0].isoformat()
    elif mutation == "malformed":
        record["eligibility_expires_at"] = "invalid"
    else:
        record[mutation] = "replacement"
    atomic_replace(path, value)
    before = path.read_bytes()
    assert not runtime.binding_write_eligible(binding, renew=True)
    assert publications == []
    assert path.read_bytes() == before


@pytest.mark.parametrize("ttl", [60, 240])
def test_policy_horizon_change_is_applied_by_next_guard(registration, ttl):
    cfg, runtime, binding, path, now, publications = registration
    save_lease_policy(cfg, LeasePolicy(ttl_seconds=ttl))
    assert runtime.binding_write_eligible(binding, renew=True)
    assert len(publications) == 1
    assert read_json(path)["registration"]["eligibility_expires_at"] == (now[0] + timedelta(seconds=ttl)).isoformat()


def test_subsecond_due_guard_does_not_republish_identical_expiry(registration):
    cfg, runtime, binding, path, now, publications = registration
    save_lease_policy(cfg, LeasePolicy(ttl_seconds=120, renew_interval_seconds=0.5))
    now[0] += timedelta(seconds=0.75)
    assert runtime.binding_write_eligible(binding, renew=True)
    assert publications == []
    now[0] += timedelta(seconds=0.25)
    assert runtime.binding_write_eligible(binding, renew=True)
    assert len(publications) == 1


def test_registration_guard_calls_write_fence_immediately_before_renewal(registration, monkeypatch):
    _cfg, runtime, binding, path, now, _publications = registration
    value = read_json(path)
    value["registration"]["eligibility_expires_at"] = (now[0] + timedelta(seconds=1)).isoformat()
    atomic_replace(path, value)
    fenced = False
    real_save = registration_authority.save_machine_registration

    def before_shared_write() -> None:
        nonlocal fenced
        fenced = True

    def require_fence_before_save(cfg, record) -> None:
        assert fenced
        real_save(cfg, record)

    monkeypatch.setattr(registration_authority, "save_machine_registration", require_fence_before_save)
    with runtime.binding_write_guard(binding, before_shared_write=before_shared_write) as eligible:
        assert eligible
    assert fenced


def test_registration_guard_callback_failure_prevents_shared_save(registration, monkeypatch):
    _cfg, runtime, binding, path, now, publications = registration
    value = read_json(path)
    value["registration"]["eligibility_expires_at"] = (now[0] + timedelta(seconds=1)).isoformat()
    atomic_replace(path, value)
    before = path.read_bytes()

    def reject_shared_write() -> None:
        raise RuntimeError("injected write fence")

    with pytest.raises(RuntimeError, match="write fence"):
        with runtime.binding_write_guard(binding, before_shared_write=reject_shared_write):
            raise AssertionError("an unsuccessful guard must not enter its protected body")

    assert path.read_bytes() == before
    assert publications == []


def test_idle_heartbeat_keeps_registration_alive_with_near_ttl_renewal_interval(registration):
    from qqtools.plugins.qexp.agent.control_plane import _MachineControlPlane
    from qqtools.plugins.qexp.agent.dispatch_loop import dispatch_machine_cycle_locked
    from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor

    cfg, runtime, binding, path, _now, _publications = registration
    # The subprocess uses the system clock, so this integration slice replaces
    # the fixture's parent-only clock with a short real-time policy.  A 13-second
    # TTL retains the full clock/retry safety budget while a 0.2-second renewal
    # interval keeps the test bounded.
    registration_authority.datetime = datetime
    registration_authority.lease_expiry = lambda policy: (
        (datetime.now(timezone.utc) + timedelta(seconds=policy.ttl_seconds)).replace(microsecond=0).isoformat()
    )
    value = read_json(path)
    value["registration"]["eligibility_expires_at"] = (datetime.now(timezone.utc) + timedelta(seconds=120)).isoformat()
    atomic_replace(path, value)
    save_lease_policy(cfg, LeasePolicy(ttl_seconds=13, renew_interval_seconds=0.2))
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    runtime.project_io_executor = executor
    runtime.authority_ready_generations = {}
    plane = _MachineControlPlane(
        runtime,
        instance_id="test",
        loop_interval=0.2,
        started_at=datetime.now(timezone.utc).isoformat(),
        available_gpus=[],
    )
    expiries = []
    try:
        for _ in range(6):
            previous = read_json(path)["registration"]["eligibility_expires_at"]
            plane._publish_heartbeat()
            assert plane._heartbeat_snapshot["observation_status"] == "queued"
            deadline = time.monotonic() + 5.0
            current = previous
            while current == previous and time.monotonic() < deadline:
                dispatch_machine_cycle_locked(runtime, available_gpus=[], supervise=False, publish_snapshots=False)
                current = read_json(path)["registration"]["eligibility_expires_at"]
                if current == previous:
                    time.sleep(0.02)
            assert current != previous
            assert runtime.registration_status(binding)["write_eligible"]
            expiries.append(current)
            time.sleep(0.21)
        assert len(set(expiries)) == len(expiries)
    finally:
        executor.shutdown()


def test_slow_configured_loop_still_services_idle_registration_before_expiry(registration):
    from qqtools.plugins.qexp.agent.control_plane import _MachineControlPlane

    cfg, runtime, binding, path, now, publications = registration
    save_lease_policy(cfg, LeasePolicy(ttl_seconds=120, renew_interval_seconds=119))
    plane = _MachineControlPlane(
        runtime,
        instance_id="test",
        loop_interval=65,
        started_at=now[0].isoformat(),
        available_gpus=[],
    )
    # Registration predates this visit; do not assume it starts on a cycle boundary.
    now[0] += timedelta(seconds=55)
    try:
        for _ in range(10):
            interval = plane._run_authority_cycle()
            assert interval <= 30
            assert runtime.registration_status(binding)["write_eligible"]
            now[0] += timedelta(seconds=interval)
        assert len(publications) >= 4
    finally:
        plane.stop()


def test_idle_policy_shortening_refreshes_service_deadline(registration):
    from qqtools.plugins.qexp.agent.control_plane import _MachineControlPlane

    cfg, runtime, binding, path, now, publications = registration
    save_lease_policy(cfg, LeasePolicy(ttl_seconds=120, renew_interval_seconds=119))
    plane = _MachineControlPlane(
        runtime,
        instance_id="test",
        loop_interval=65,
        started_at=now[0].isoformat(),
        available_gpus=[],
    )
    try:
        assert plane._run_authority_cycle() == 30
        now[0] += timedelta(seconds=5)
        save_lease_policy(cfg, LeasePolicy(ttl_seconds=20, renew_interval_seconds=19))
        for _ in range(10):
            interval = plane._run_authority_cycle()
            assert interval == 5
            assert runtime.registration_status(binding)["write_eligible"]
            now[0] += timedelta(seconds=interval)
    finally:
        plane.stop()
