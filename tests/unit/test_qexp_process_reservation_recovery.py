"""Released runner envelopes can recover locators without receiving authority."""

from pathlib import Path

import pytest

from qqtools.plugins.qexp.agent.bindings import ProjectBinding
from qqtools.plugins.qexp.agent.process_reservation_recovery import ProcessReservationRecovery, registration_reservation
from qqtools.plugins.qexp.runtime.paths import local_paths, machine_project_paths
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json


def case(tmp_path: Path, *, state="active", cpu=False, manifest=True):
    root = tmp_path / "machine"
    binding = ProjectBinding("project", tmp_path / "shared", "gpu-1")
    paths = machine_project_paths(root, binding.project_id)
    source = {
        "protocol_version": 1,
        "task_id": "task",
        "attempt_id": "task-attempt-1",
        "fencing_token": 7,
        "machine_name": "gpu-1",
        "wrapper_pid": 101,
        "wrapper_start_time_ticks": 202,
        "process_group_id": 303,
        "process_group_start_time_ticks": 404,
        "process_created_at": "2026-09-20T00:00:00Z",
        "gpu_ids": [] if cpu else [0],
    }
    reservation = {
        "reservation_id": "reservation",
        "acquisition_id": "acquisition",
        "project_id": binding.project_id,
        "shared_root": str(binding.shared_root),
        "task_id": "task",
        "attempt_id": "task-attempt-1",
        "fencing_token": 7,
        "machine_name": "gpu-1",
        "state": state,
        **({"cpu_slots": 1} if cpu else {"gpu_ids": [0]}),
    }
    record_path = local_paths(root)[("cpu_" if cpu else "") + state] / "reservation.json"
    registration = paths["registrations"] / "task-attempt-1.json"
    process = paths["processes"] / registration.name
    atomic_replace(record_path, {"reservation": reservation})
    atomic_replace(registration, {"process_registration": source})
    if manifest:
        atomic_replace(process, {"process": {**source, "observed_state": "running", "authority_state": "isolated"}})
    return root, binding, paths, source, reservation, record_path, registration, process


@pytest.mark.parametrize("state", ["active", "released"])
@pytest.mark.parametrize("cpu", [False, True])
@pytest.mark.parametrize("manifest", [False, True])
def test_recovers_exact_gpu_cpu_locator_without_mutating_registration_or_authority(tmp_path, state, cpu, manifest):
    root, binding, paths, source, _, record_path, registration, process = case(
        tmp_path, state=state, cpu=cpu, manifest=manifest
    )
    immutable = registration.read_bytes(), record_path.read_bytes()
    recovery = ProcessReservationRecovery(root)
    try:
        recovery.advance([binding])
        value = read_json(process)["process"]
        assert value["reservation_id"] == "reservation"
        assert value["authority_state"] == "isolated"
        assert value["observed_state"] == "running"
        assert registration_reservation(source, paths["root"]) == "reservation"
        before = process.read_bytes()
        recovery.advance([binding])
        assert process.read_bytes() == before
        assert (registration.read_bytes(), record_path.read_bytes()) == immutable
    finally:
        recovery.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("project_id", "other"),
        ("shared_root", "/other"),
        ("machine_name", "other"),
        ("task_id", "other"),
        ("attempt_id", "other-attempt-1"),
        ("fencing_token", 8),
        ("gpu_ids", [1]),
    ],
)
def test_different_reservation_identity_cannot_repair_a_process(tmp_path, field, value):
    root, binding, _, _, reservation, record_path, _, process = case(tmp_path)
    reservation[field] = value
    atomic_replace(record_path, {"reservation": reservation})
    before = process.read_bytes()
    recovery = ProcessReservationRecovery(root)
    try:
        recovery.advance([binding])
        assert process.read_bytes() == before
    finally:
        recovery.close()


@pytest.mark.parametrize("field", ["wrapper_pid", "wrapper_start_time_ticks", "process_group_id", "fencing_token"])
def test_conflicting_manifest_cannot_supply_a_registration_locator(tmp_path, field):
    root, binding, paths, source, _, _, _, process = case(tmp_path)
    value = read_json(process)["process"]
    value[field] += 1
    atomic_replace(process, {"process": value})
    before = process.read_bytes()
    recovery = ProcessReservationRecovery(root)
    try:
        recovery.advance([binding])
        assert process.read_bytes() == before
        value["reservation_id"] = "reservation"
        atomic_replace(process, {"process": value})
        assert registration_reservation(source, paths["root"]) is None
    finally:
        recovery.close()


@pytest.mark.parametrize("existing", [None, "different"])
def test_explicit_manifest_locator_is_never_replaced(tmp_path, existing):
    root, binding, _, _, _, _, _, process = case(tmp_path)
    value = read_json(process)["process"]
    value["reservation_id"] = existing
    atomic_replace(process, {"process": value})
    before = process.read_bytes()
    recovery = ProcessReservationRecovery(root)
    try:
        recovery.advance([binding])
        assert process.read_bytes() == before
    finally:
        recovery.close()


def test_current_registration_is_ignored_after_one_immutable_check(tmp_path, monkeypatch):
    root, binding, _, source, _, _, registration, _ = case(tmp_path)
    source["reservation_id"] = "reservation"
    atomic_replace(registration, {"process_registration": source})
    calls = []
    original = ProcessReservationRecovery._recover

    def recover(recovery, path, state, bindings):
        calls.append(path)
        return original(recovery, path, state, bindings)

    monkeypatch.setattr(ProcessReservationRecovery, "_recover", recover)
    recovery = ProcessReservationRecovery(root)
    try:
        recovery.advance([binding])
        recovery.advance([binding])
        assert calls == [local_paths(root)["active"] / "reservation.json"]
    finally:
        recovery.close()


def test_immutable_registration_ignore_cache_is_bounded(tmp_path):
    recovery = ProcessReservationRecovery(tmp_path / "machine")
    try:
        for index in range(1_025):
            recovery._ignore_path(Path(f"reservation-{index}.json"))
        assert len(recovery._ignored_paths) == 1_024
        assert Path("reservation-0.json") not in recovery._ignored_paths
        assert Path("reservation-1024.json") in recovery._ignored_paths
    finally:
        recovery.close()


def test_recovery_bounds_each_scan_and_continues_after_unreadable_record(tmp_path, monkeypatch):
    root, binding, _, _, _, record_path, _, process = case(tmp_path)
    for index in range(18):
        (record_path.parent / f"bad-{index}.json").write_text("{")
    recovery = ProcessReservationRecovery(root)
    counts = []
    from qqtools.plugins.qexp.runtime.authority_scan import EvidenceScan

    original = EvidenceScan.take

    def take(scan, limit, **kwargs):
        counts.append(limit)
        return original(scan, limit, **kwargs)

    monkeypatch.setattr(EvidenceScan, "take", take)
    try:
        for _ in range(4):
            recovery.advance([binding])
        assert counts == [8] * 16
        assert read_json(process)["process"]["reservation_id"] == "reservation"
    finally:
        recovery.close()


@pytest.mark.parametrize("field", ["gpu_ids", "cpu_slots"])
def test_conflicting_manifest_resources_cannot_be_repaired_or_resolve_registration(tmp_path, field):
    root, binding, paths, source, _, _, _, process = case(tmp_path)
    value = read_json(process)["process"]
    value[field] = [1] if field == "gpu_ids" else 2
    atomic_replace(process, {"process": value})
    before = process.read_bytes()
    recovery = ProcessReservationRecovery(root)
    try:
        recovery.advance([binding])
        assert process.read_bytes() == before
        value["reservation_id"] = "reservation"
        atomic_replace(process, {"process": value})
        assert registration_reservation(source, paths["root"]) is None
    finally:
        recovery.close()
