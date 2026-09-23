"""QQTOOLS-COMPAT-0017: obligation-driven Group service activation and locators."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.commands.task import edit_dependencies, submit
from qqtools.plugins.qexp.runtime.group_discovery import activation, locator
from qqtools.plugins.qexp.runtime.group_discovery.coverage import GroupCoverage
from qqtools.plugins.qexp.runtime.group_discovery.maintenance import GroupMaintenance
from qqtools.plugins.qexp.runtime.group_namespace import group_authority_identity
from qqtools.plugins.qexp.runtime.locks import group_writer_lock
from qqtools.plugins.qexp.runtime.paths import group_path
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.upgrade import UpgradeCoordinator
from tests.helpers.qexp_discovery import isolated_group

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


@pytest.fixture(autouse=True)
def target_writer(monkeypatch):
    """Exercise the target-release gate before version metadata is committed."""
    monkeypatch.setattr(activation, "__version__", "1.3.22")


def activate(cfg) -> dict[str, object]:
    for _ in range(20_000):
        state = activation.advance_group_service_activation(cfg)
        if state["state"] == "active":
            return state
    raise AssertionError("Group service activation did not converge")


def test_layout_precreates_exact_identity_bound_shards(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    layout = locator.ensure_group_service_layout(cfg)
    root = cfg.shared_root / "operations/group-service-v1"
    expected = [f"{lane}/{shard:02x}" for lane in locator.LANES for shard in range(256)]
    assert layout == read_json(root / "layout.json")
    assert layout["identity"] == {
        "project_id": group_authority_identity(cfg.shared_root)["project_id"],
        "group_directory_identity": group_authority_identity(cfg.shared_root)["directory_identity"],
    }
    assert layout["lanes"] == ["control", "maintenance", "membership"]
    assert (
        layout["manifest_digest"]
        == hashlib.sha256(json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    )
    assert [item.relative_to(root).as_posix() for item in sorted(root.glob("*/*")) if item.is_dir()] == expected


def test_locator_publication_coalesces_with_monotonic_generation(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    locator.ensure_group_service_layout(cfg)
    with group_writer_lock(cfg, "experiment"):
        first = locator.publish_group_locator_locked(cfg, "experiment", "membership", "submission_commit")
        second = locator.publish_group_locator_locked(cfg, "experiment", "membership", "submission_finalize")
    assert first["generation"] == 1
    assert second["generation"] == 2
    assert locator.read_group_locator(cfg.shared_root, "experiment", "membership") == second
    assert set(second) == {"version", "identity", "service", "generation", "published_at", "reason"}


def test_generation_fence_preserves_concurrent_publication(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    locator.ensure_group_service_layout(cfg)
    with group_writer_lock(cfg, "experiment"):
        observed = locator.publish_group_locator_locked(cfg, "experiment", "control", "task_change")
        newer = locator.publish_group_locator_locked(cfg, "experiment", "control", "group_operation")
        assert not locator.acknowledge_group_locator_locked(
            cfg, "experiment", "control", observed["generation"], retirement_ready=lambda: True
        )
    assert locator.read_group_locator(cfg.shared_root, "experiment", "control") == newer


def test_acknowledgement_requires_authoritative_retirement_proof(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    locator.ensure_group_service_layout(cfg)
    with group_writer_lock(cfg, "experiment"):
        record = locator.publish_group_locator_locked(cfg, "experiment", "maintenance", "metadata_cleanup")
        assert not locator.acknowledge_group_locator_locked(
            cfg, "experiment", "maintenance", record["generation"], retirement_ready=lambda: False
        )
        assert locator.acknowledge_group_locator_locked(
            cfg, "experiment", "maintenance", record["generation"], retirement_ready=lambda: True
        )
    assert locator.read_group_locator(cfg.shared_root, "experiment", "maintenance") is None


def test_identity_mismatch_fails_closed_without_replacing_record(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    locator.ensure_group_service_layout(cfg)
    path = locator.group_locator_path(cfg.shared_root, "experiment", "membership")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_replace(
        path,
        {
            "version": 1,
            "identity": {
                "project_id": "wrong",
                "group_directory_identity": {},
                "group": "experiment",
            },
            "service": "membership",
            "generation": 9,
            "published_at": "2026-09-23T00:00:00+00:00",
            "reason": "submission_commit",
        },
    )
    before = path.read_bytes()
    with group_writer_lock(cfg, "experiment"):
        with pytest.raises(RuntimeError, match="identity"):
            locator.publish_group_locator_locked(cfg, "experiment", "membership", "submission_commit")
    assert path.read_bytes() == before


def test_activation_fences_writer_then_bootstraps_to_active(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    state = activate(cfg)
    assert state["state"] == "active"
    assert state["writer_floor"] == "1.3.22"
    assert state["bootstrap"]["post_fence"] is True
    assert state["bootstrap"]["stable_pass"] >= 1
    assert state["bootstrap"]["phase"] == "complete"
    assert state["bootstrap"]["completed_at"] is not None
    schema = read_json(cfg.shared_root / "schema/version.json")["schema"]
    assert "group-service-v1" in schema["required_capabilities"]
    assert activation.is_group_service_active(cfg.shared_root)


def test_active_locator_traversal_does_not_enumerate_group_history(tmp_path, monkeypatch):
    cfg = isolated_group(tmp_path, tail=0)
    activate(cfg)
    with group_writer_lock(cfg, "experiment"):
        expected = locator.publish_group_locator_locked(cfg, "experiment", "control", "task_change")
    group_root = cfg.shared_root / "groups-v2"
    original = locator.read_directory_entry

    def guarded(path: Path, offset: int):
        assert path != group_root
        assert path != cfg.shared_root / "operations/submissions"
        return original(path, offset)

    monkeypatch.setattr(locator, "read_directory_entry", guarded)
    traversal = locator.GroupLocatorTraversal(cfg.shared_root, "control")
    for _ in range(1024):
        found = traversal.advance()
        if found is not None:
            break
    assert found == expected


def test_invalid_activation_transition_fails_closed(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    activate(cfg)
    with pytest.raises(RuntimeError, match="transition"):
        activation.transition_group_service_state(cfg, "preparing")


def test_source_writer_cannot_install_capability_fence(tmp_path, monkeypatch):
    cfg = isolated_group(tmp_path, tail=0)
    monkeypatch.setattr(activation, "__version__", "1.3.21")
    state = activation.advance_group_service_activation(cfg)
    assert state["state"] == "preparing"
    assert state["diagnostic"]["code"] == "writer_floor_not_met"
    schema = read_json(cfg.shared_root / "schema/version.json")["schema"]
    assert "group-service-v1" not in schema["required_capabilities"]


def test_eligible_old_participant_blocks_fence(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    atomic_replace(
        cfg.shared_root / "machines/g1/registration.json",
        {
            "registration": {
                "state": "eligible",
                "machine_name": "g1",
                "generation": uuid.uuid4().hex,
                "client_version": "1.3.21",
                "eligibility_expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            }
        },
    )
    state = activation.advance_group_service_activation(cfg)
    assert state["state"] == "preparing"
    assert state["diagnostic"]["code"] == "participant_below_writer_floor"


def test_expired_old_participant_does_not_block_fence(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    atomic_replace(
        cfg.shared_root / "machines/g1/registration.json",
        {
            "registration": {
                "state": "eligible",
                "machine_name": "g1",
                "generation": uuid.uuid4().hex,
                "client_version": "1.3.21",
                "eligibility_expires_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
            }
        },
    )
    state = activation.advance_group_service_activation(cfg)
    assert state["state"] == "fenced"


@pytest.mark.parametrize("cut", ["preparing", "fenced", "building", "active"])
def test_activation_recovers_after_each_state_record_is_durable(tmp_path, monkeypatch, cut):
    cfg = isolated_group(tmp_path, tail=0)
    original = activation.atomic_replace
    interrupted = False

    class SimulatedProcessExit(BaseException):
        pass

    def crash_after_replace(path, value):
        nonlocal interrupted
        original(path, value)
        if path == activation.activation_path(cfg.shared_root) and value.get("state") == cut and not interrupted:
            interrupted = True
            raise SimulatedProcessExit(f"crash after {cut}")

    with monkeypatch.context() as crashing:
        crashing.setattr(activation, "atomic_replace", crash_after_replace)
        with pytest.raises(SimulatedProcessExit, match=f"crash after {cut}"):
            for _ in range(20_000):
                activation.advance_group_service_activation(cfg)
    assert interrupted
    assert activate(cfg)["state"] == "active"


def test_active_metadata_damage_transitions_to_degraded(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    activate(cfg)
    (cfg.shared_root / "operations/group-service-v1/layout.json").unlink()
    state = activation.advance_group_service_activation(cfg)
    assert state["state"] == "degraded"
    assert state["diagnostic"]["code"] == "activation_degraded"


def test_active_worker_consumes_membership_locator_without_history_sweep(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runtime.group_discovery import service as service_module

    cfg = isolated_group(tmp_path, tail=0)
    activate(cfg)
    runtime = MachineRuntime(tmp_path / "machine")
    runtime.add_binding(cfg.shared_root, cfg.machine_name)
    _revision, bindings = runtime.load_registry()
    assert runtime.binding_state(bindings[0]) == "enabled"
    with group_writer_lock(cfg, "experiment"):
        locator.publish_group_locator_locked(cfg, "experiment", "membership", "submission_finalize")

    class ForbiddenSweep:
        def __init__(self, *args, **kwargs):
            pytest.fail("active Group service created a historical sweep")

    monkeypatch.setattr(service_module, "SubmissionSourceSweep", ForbiddenSweep)
    worker = service_module.MachineGroupDiscoveryWorker(runtime)
    worker.start()
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if locator.read_group_locator(cfg.shared_root, "experiment", "membership") is None:
                break
            time.sleep(0.02)
        assert locator.read_group_locator(cfg.shared_root, "experiment", "membership") is None, (
            worker.metrics,
            runtime.load_registry(),
            runtime.upgrade_admission_blocked_projects,
        )
    finally:
        worker.stop()
    assert not worker.is_alive


def test_upgrade_coordinator_discovers_and_activates_group_service(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    coordinator = UpgradeCoordinator(cfg, holder_id="group-service-test")

    available = {item["name"] for item in coordinator.status().get("available_migrations", [])}
    assert "group-service-v1" in available
    status = coordinator.discover()
    assert "group-service-v1" in {item["name"] for item in status["migrations"]}
    status = coordinator.advance(force_retry=True)
    migration = next(item for item in status["migrations"] if item["name"] == "group-service-v1")
    assert migration["phase"] == "expansion"
    assert json.loads(migration["cursor"]) == {"phase": "create", "position": 8, "revisions": {}}
    assert migration["last_slice_usage"]["metadata_ops"] <= migration["max_metadata_ops_per_slice"]
    layout_phases = {"create"}

    for _ in range(20_000):
        if not status["pending"]:
            break
        status = coordinator.advance(force_retry=True)
        migration = next(item for item in status["migrations"] if item["name"] == "group-service-v1")
        if migration["phase"] == "expansion" and migration["cursor"] is not None:
            layout_phases.add(json.loads(migration["cursor"])["phase"])
    else:
        raise AssertionError("Group service coordinator migration did not converge")

    assert status["state"] == "completed"
    assert layout_phases == {"create", "verify"}
    assert activation.is_group_service_active(cfg.shared_root)


def test_coordinator_layout_verification_detects_earlier_shard_loss(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    coordinator = UpgradeCoordinator(cfg, holder_id="group-service-layout-race-test")
    status = coordinator.discover()
    deleted = False

    for _ in range(1_000):
        status = coordinator.advance(force_retry=True)
        migration = next(item for item in status["migrations"] if item["name"] == "group-service-v1")
        if migration["phase"] == "expansion" and migration["cursor"] is not None:
            cursor = json.loads(migration["cursor"])
            if cursor["phase"] == "verify" and cursor["position"] >= 8 and not deleted:
                (cfg.shared_root / "operations/group-service-v1/control/00").rmdir()
                deleted = True
        if status["state"] == "repair_required":
            break

    assert deleted
    assert status["state"] == "repair_required"
    assert any("layout shard is missing" in blocker for blocker in status["blockers"])
    assert not (cfg.shared_root / "operations/group-service-v1/layout.json").exists()


def test_group_worker_does_not_advance_activation_record(tmp_path):
    from qqtools.plugins.qexp.runtime.group_discovery import service as service_module

    cfg = isolated_group(tmp_path, tail=0)
    record = activation.advance_group_service_activation(cfg)
    assert record["state"] == "fenced"
    before = activation.activation_path(cfg.shared_root).read_bytes()
    runtime = MachineRuntime(tmp_path / "machine")
    runtime.add_binding(cfg.shared_root, cfg.machine_name)
    worker = service_module.MachineGroupDiscoveryWorker(runtime)

    worker.start()
    try:
        time.sleep(0.1)
    finally:
        worker.stop()

    assert activation.activation_path(cfg.shared_root).read_bytes() == before


def test_coordinator_repair_rebuilds_degraded_group_service(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    coordinator = UpgradeCoordinator(cfg, holder_id="group-service-repair-test")
    status = coordinator.discover()
    for _ in range(20_000):
        if not status["pending"]:
            break
        status = coordinator.advance(force_retry=True)
    else:
        raise AssertionError("Group service coordinator migration did not converge")

    layout_path = cfg.shared_root / "operations/group-service-v1/layout.json"
    layout_path.unlink()
    assert coordinator.status()["state"] == "repair_required"
    assert coordinator.discover()["state"] == "repair_required"

    repair = coordinator.inspect_repair("group-service-v1")
    coordinator.apply_repair(repair["repair_id"])
    coordinator.validate_repair(repair["repair_id"])
    status = coordinator.resume()
    assert read_json(activation.activation_path(cfg.shared_root))["state"] == "degraded"

    for _ in range(20_000):
        if not status["pending"]:
            break
        status = coordinator.advance(force_retry=True)
    else:
        raise AssertionError("Group service degraded rebuild did not converge")

    assert activation.is_group_service_active(cfg.shared_root)


def test_maintenance_checkpoint_survives_and_acknowledges_exact_generation(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runtime.group_discovery import maintenance as maintenance_module

    cfg = isolated_group(tmp_path, tail=0)
    activate(cfg)
    with group_writer_lock(cfg, "experiment"):
        observed = locator.publish_group_locator_locked(cfg, "experiment", "maintenance", "metadata_cleanup")
    ticks = iter(range(0, 100_000, 2))
    monkeypatch.setattr(maintenance_module.time, "monotonic", lambda: float(next(ticks)))
    service = GroupMaintenance(cfg.shared_root, "experiment")
    for _ in range(2_000):
        service.advance(locator_generation=observed["generation"])
        if service.acknowledge_if_quiescent(observed["generation"]):
            break
    else:
        raise AssertionError("maintenance quiescence pass did not converge")
    checkpoint = read_json(GroupCoverage(cfg.shared_root, "experiment").directory / "maintenance/quiescence-v1.json")
    assert set(checkpoint) == {"version", "identity", "locator_generation", "pass_id", "next_lane", "lanes"}
    assert checkpoint["locator_generation"] == observed["generation"]
    assert all(lane["state"] == "complete" for lane in checkpoint["lanes"].values())
    assert locator.read_group_locator(cfg.shared_root, "experiment", "maintenance") is None


def test_maintenance_checkpoint_rebuilds_for_new_generation_and_corruption(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runtime.group_discovery import maintenance as maintenance_module

    cfg = isolated_group(tmp_path, tail=0)
    activate(cfg)
    with group_writer_lock(cfg, "experiment"):
        first = locator.publish_group_locator_locked(cfg, "experiment", "maintenance", "metadata_cleanup")
    ticks = iter(range(0, 100_000, 2))
    monkeypatch.setattr(maintenance_module.time, "monotonic", lambda: float(next(ticks)))
    service = GroupMaintenance(cfg.shared_root, "experiment")
    service.advance(locator_generation=first["generation"])
    path = GroupCoverage(cfg.shared_root, "experiment").directory / "maintenance/quiescence-v1.json"
    first_checkpoint = read_json(path)

    with group_writer_lock(cfg, "experiment"):
        second = locator.publish_group_locator_locked(cfg, "experiment", "maintenance", "metadata_cleanup")
    service.advance(locator_generation=second["generation"])
    second_checkpoint = read_json(path)
    assert second_checkpoint["locator_generation"] == second["generation"]
    assert second_checkpoint["pass_id"] != first_checkpoint["pass_id"]
    assert service.open_descriptor_count == 0

    atomic_replace(path, {"invalid": True})
    restarted = GroupMaintenance(cfg.shared_root, "experiment")
    restarted.advance(locator_generation=second["generation"])
    rebuilt = read_json(path)
    assert rebuilt["locator_generation"] == second["generation"]
    assert set(rebuilt) == {"version", "identity", "locator_generation", "pass_id", "next_lane", "lanes"}


def test_active_worker_enforces_lane_cycle_and_close_cap(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runtime.group_discovery import service as service_module

    cfg = isolated_group(tmp_path, tail=0)
    runtime = MachineRuntime(tmp_path / "machine")
    runtime.add_binding(cfg.shared_root, cfg.machine_name)
    _revision, bindings = runtime.load_registry()
    binding = bindings[0]
    worker = service_module.MachineGroupDiscoveryWorker(runtime)
    observed_lanes = []

    class EmptyTraversal:
        def __init__(self, root, lane):
            del root
            self.lane = lane

        def advance(self):
            observed_lanes.append(self.lane)
            return None

    monkeypatch.setattr(locator, "GroupLocatorTraversal", EmptyTraversal)
    key = worker._binding_key(binding)
    for _ in range(8):
        assert worker._advance_active_traversal(0, binding, key) is None
    assert observed_lanes == [
        "control",
        "control",
        "membership",
        "maintenance",
        "control",
        "control",
        "membership",
        "maintenance",
    ]

    observed_lanes.clear()
    for project in range(4):
        project_key = (f"project-{project}", key[1], f"{key[2]}-{project}")
        for _ in range(4):
            assert worker._advance_active_traversal(0, binding, project_key) is None
    assert observed_lanes == ["control", "control", "membership", "maintenance"] * 4

    owners = [object() for _ in range(17)]
    assert all(worker._queue_sweep_close(owner, key) for owner in owners[:16])
    assert not worker._queue_sweep_close(owners[16], key)
    assert len(worker._close_work_index) == 16


def test_active_worker_evicts_durable_locator_owner_at_resident_cap(tmp_path):
    from qqtools.plugins.qexp.runtime.group_discovery import service as service_module

    cfg = isolated_group(tmp_path, tail=0)
    runtime = MachineRuntime(tmp_path / "machine")
    runtime.add_binding(cfg.shared_root, cfg.machine_name)
    _revision, bindings = runtime.load_registry()
    binding = bindings[0]
    worker = service_module.MachineGroupDiscoveryWorker(runtime)
    for index in range(64):
        entry = worker._new_entry(
            binding,
            f"blocked-{index}",
            locator_lane="control",
            locator_record={"generation": 1},
        )
        worker._entries[entry.key] = entry
        worker._entry_queue.append(entry)

    worker._pending_locator = ("control", {"generation": 1})
    worker._admit_group(binding, "ready-group")

    assert len(worker._entries) == 63
    assert len(worker._close_queue) == 1
    assert worker.metrics["control"]["cap_refusals"] == 1


def test_new_membership_generation_rechecks_prior_pending_submission(tmp_path):
    from qqtools.plugins.qexp.runtime.group_discovery.service import GroupDiscoveryService

    cfg = isolated_group(tmp_path, tail=0)
    activate(cfg)
    path = group_path(cfg.shared_root, "experiment")
    group = read_json(path)
    group["group"]["pending_submission_commit"] = {"operation_id": uuid.uuid4().hex}
    atomic_replace(path, group)
    service = GroupDiscoveryService(cfg.shared_root, "experiment", mode="locator", locator_generation=1)

    for _ in range(20):
        result = service.advance()
        if result["state"] == "error":
            break
    assert result["state"] == "error"
    assert service._phase == "blocked"

    group = read_json(path)
    group["group"]["pending_submission_commit"] = None
    atomic_replace(path, group)
    service.update_locator_generation(2)
    for _ in range(20):
        result = service.advance()
        if result["state"] == "complete":
            break
    assert result["state"] == "complete"


def test_dependency_edit_publishes_control_locator_before_task_effect(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    activate(cfg)
    parent = submit(cfg, ["true"], task_id="parent", group="experiment", working_dir=tmp_path)
    child = submit(cfg, ["true"], task_id="child", group="experiment", working_dir=tmp_path)

    edit_dependencies(cfg, child.task_id, [parent.task_id], action="add")

    record = locator.read_group_locator(cfg.shared_root, "experiment", "control")
    assert record is not None
    assert record["reason"] == "task_change"
