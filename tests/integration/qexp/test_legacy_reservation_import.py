"""Reservation migration keeps occupancy and durable ownership across restart."""

import multiprocessing
import os
from pathlib import Path
from threading import Event, Thread

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent import project_admin
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.project_admin import migrate_project
from qqtools.plugins.qexp.runtime.locks import exclusive
from qqtools.plugins.qexp.runtime.paths import local_paths
from qqtools.plugins.qexp.runtime.resources.cpu_lane import (
    attach_cpu,
    cpu_reservation_snapshot,
    release_cpu,
    reserve_cpu,
    set_cpu_lane_capacity,
)
from qqtools.plugins.qexp.runtime.resources.reservations import attach, release, reserve
from qqtools.plugins.qexp.runtime.responsibility_store import DurableIO
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json

pytestmark = pytest.mark.integration


def migration_fixture(tmp_path, is_cpu, is_active):
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "old")
    machine = cfg.shared_root / "machines" / cfg.machine_name / "machine.json"
    value = read_json(machine)
    value["machine"].pop("agent_runtime")
    atomic_replace(machine, value)
    runtime = MachineRuntime(tmp_path / "machine")
    runtime.ensure_layout()
    if is_cpu:
        set_cpu_lane_capacity(cfg.runtime_root, capacity=2)
        value = reserve_cpu(cfg.runtime_root, "old-task", 1, attempt_id="old-task-attempt-1", fencing_token=1)
        if is_active:
            attach_cpu(cfg.runtime_root, value["reservation"]["reservation_id"], "old-task-attempt-1", 1)
    else:
        value = reserve(cfg.runtime_root, "old-task", [0], attempt_id="old-task-attempt-1", fencing_token=1)
        if is_active:
            attach(cfg.runtime_root, value["reservation"]["reservation_id"], "old-task-attempt-1", 1)
    lane = ("cpu_" if is_cpu else "") + ("active" if is_active else "provisional")
    source = local_paths(cfg.runtime_root)[lane] / f"{value['reservation']['reservation_id']}.json"
    destination = runtime.paths[lane] / source.name
    return cfg, runtime, source, destination


@pytest.mark.parametrize("is_cpu", [False, True])
@pytest.mark.parametrize("is_active", [False, True])
def test_migration_moves_both_capacity_domains_without_changing_cpu_policy(tmp_path, is_cpu, is_active):
    cfg, runtime, source, destination = migration_fixture(tmp_path, is_cpu, is_active)
    original = read_json(source)
    set_cpu_lane_capacity(runtime.root, capacity=1)
    other = reserve_cpu(runtime.root, "other-task", 1, project_id="other-project")
    policy = runtime.paths["cpu_policy"].read_bytes()

    binding = migrate_project(runtime, cfg)

    expected = original["reservation"] | {
        "project_id": binding.project_id,
        "shared_root": str(cfg.shared_root),
        "machine_name": cfg.machine_name,
    }
    assert binding.enabled
    assert not source.exists()
    assert read_json(destination)["reservation"] == expected
    assert runtime.paths["cpu_policy"].read_bytes() == policy
    _, occupancy = cpu_reservation_snapshot(runtime.root)
    assert {item["reservation_id"] for item in occupancy} == {
        other["reservation"]["reservation_id"],
        *([source.stem] if is_cpu else []),
    }
    with pytest.raises(ValueError, match="insufficient"):
        reserve_cpu(runtime.root, "new-task", 1)


@pytest.mark.parametrize("is_cpu", [False, True])
@pytest.mark.parametrize("boundary", ["destination", "source"])
def test_directory_barrier_failure_retains_recoverable_occupancy(tmp_path, monkeypatch, is_cpu, boundary):
    cfg, runtime, source, destination = migration_fixture(tmp_path, is_cpu, True)
    original_sync = DurableIO.sync_directory

    def fail_sync(io, path, label):
        expected = destination.parent if boundary == "destination" else source.parent
        if path == expected and label.startswith("migration_reservation_"):
            raise OSError("reservation barrier interrupted")
        return original_sync(io, path, label)

    with monkeypatch.context() as patch:
        patch.setattr(DurableIO, "sync_directory", fail_sync)
        with pytest.raises(OSError, match="reservation barrier interrupted"):
            migrate_project(runtime, cfg)
    binding = runtime.matching_binding(cfg)
    assert not binding.enabled
    assert read_json(runtime.migration_path(binding.project_id))["migration"]["state"] == "blocked"
    assert destination.exists()
    assert source.exists() == (boundary == "destination")

    synced = []

    def track_sync(io, path, label):
        if label.startswith("migration_reservation_"):
            synced.append(path)
        return original_sync(io, path, label)

    monkeypatch.setattr(DurableIO, "sync_directory", track_sync)
    assert migrate_project(MachineRuntime(runtime.root), cfg).enabled
    assert source.parent in synced
    if boundary == "destination":
        assert all(path in synced for path in destination.resolve().parents)
    assert not source.exists() and destination.exists()


def crash_during_migration(cfg, runtime_root, source, destination, boundary):
    original_replace = project_admin.atomic_replace
    original_delete = DurableIO.delete

    def copy(path, value):
        original_replace(path, value)
        if boundary == "copy" and path == destination:
            os._exit(86)

    def delete(io, path, **kwargs):
        original_delete(io, path, **kwargs)
        if boundary == "unlink" and path == source:
            os._exit(86)

    project_admin.atomic_replace = copy
    DurableIO.delete = delete
    migrate_project(MachineRuntime(runtime_root), cfg)


@pytest.mark.parametrize("is_cpu", [False, True])
@pytest.mark.parametrize("boundary", ["copy", "unlink"])
def test_real_process_crash_resumes_reservation_migration(tmp_path, is_cpu, boundary):
    cfg, runtime, source, destination = migration_fixture(tmp_path, is_cpu, True)
    process = multiprocessing.get_context("fork").Process(
        target=crash_during_migration, args=(cfg, runtime.root, source, destination, boundary)
    )
    process.start()
    try:
        process.join(10)
        assert process.exitcode == 86
    finally:
        if process.is_alive():
            process.kill()
        process.join(5)
        process.close()
    assert destination.exists()
    assert not runtime.matching_binding(cfg).enabled
    assert migrate_project(MachineRuntime(runtime.root), cfg).enabled
    assert not source.exists()
    assert read_json(destination)["reservation"]["attempt_id"] == "old-task-attempt-1"


@pytest.mark.parametrize("is_cpu", [False, True])
@pytest.mark.parametrize("has_retained_copy", [False, True], ids=["removed", "retained"])
@pytest.mark.parametrize("should_interrupt_barrier", [False, True], ids=["durable", "barrier_failure"])
def test_released_import_is_not_resurrected_when_migration_retries(
    tmp_path, monkeypatch, is_cpu, has_retained_copy, should_interrupt_barrier
):
    cfg, runtime, source, destination = migration_fixture(tmp_path, is_cpu, True)
    original_delete = DurableIO.delete

    def fail_unlink(io, path, **kwargs):
        if path == source:
            raise OSError("before source unlink")
        return original_delete(io, path, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(DurableIO, "delete", fail_unlink)
        with pytest.raises(OSError, match="before source unlink"):
            migrate_project(runtime, cfg)
    release_reservation = release_cpu if is_cpu else release
    if has_retained_copy:
        original_unlink = Path.unlink

        def fail_release_unlink(path, *args, **kwargs):
            if path == destination:
                raise OSError("release unlink interrupted")
            return original_unlink(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "unlink", fail_release_unlink)
            with pytest.raises(OSError, match="release unlink interrupted"):
                release_reservation(runtime.root, source.stem)
    else:
        release_reservation(runtime.root, source.stem)
    if not is_cpu and not has_retained_copy:
        # Reusing the released GPU must not make the old source look like new occupancy.
        reserve(runtime.root, "successor-task", [0], project_id="other-project")
    released = runtime.paths["cpu_released" if is_cpu else "released"] / source.name
    receipt = released.read_bytes()
    assert source.exists() and destination.exists() == has_retained_copy

    if should_interrupt_barrier:
        original_sync = DurableIO.sync_directory

        def fail_retire_barrier(io, path, label):
            if label == "migration_reservation_retired":
                raise OSError("retire barrier interrupted")
            return original_sync(io, path, label)

        with monkeypatch.context() as patch:
            patch.setattr(DurableIO, "sync_directory", fail_retire_barrier)
            with pytest.raises(OSError, match="retire barrier interrupted"):
                migrate_project(runtime, cfg)
        assert source.exists() and not destination.exists()
        assert not runtime.matching_binding(cfg).enabled

    assert migrate_project(runtime, cfg).enabled

    assert not source.exists() and not destination.exists()
    assert released.read_bytes() == receipt


@pytest.mark.parametrize("is_cpu", [False, True])
@pytest.mark.parametrize("is_parent", [False, True])
def test_redirected_release_receipt_cannot_retire_migration_source(tmp_path, monkeypatch, is_cpu, is_parent):
    cfg, runtime, source, destination = migration_fixture(tmp_path, is_cpu, True)
    original_delete = DurableIO.delete

    def interrupted(io, path, **kwargs):
        if path == source:
            raise OSError("source retained")
        return original_delete(io, path, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(DurableIO, "delete", interrupted)
        with pytest.raises(OSError, match="source retained"):
            migrate_project(runtime, cfg)
    if is_cpu:
        release_cpu(runtime.root, source.stem)
    else:
        release(runtime.root, source.stem)
    receipt = runtime.paths["cpu_released" if is_cpu else "released"] / source.name
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    copy = foreign / receipt.name
    copy.write_bytes(receipt.read_bytes())
    receipt.unlink()
    if is_parent:
        receipt.parent.rmdir()
        receipt.parent.symlink_to(foreign, target_is_directory=True)
    else:
        receipt.symlink_to(copy)
    before = copy.read_bytes()

    with pytest.raises(OSError, match="not a (regular file|real directory)"):
        migrate_project(runtime, cfg)

    assert source.exists() and not destination.exists()
    assert copy.read_bytes() == before
    assert not runtime.matching_binding(cfg).enabled


@pytest.mark.parametrize("is_cpu", [False, True])
def test_conflict_in_another_phase_keeps_source_and_existing_reservation(tmp_path, is_cpu):
    cfg, runtime, source, destination = migration_fixture(tmp_path, is_cpu, False)
    other = runtime.paths["cpu_active" if is_cpu else "active"] / source.name
    value = read_json(source)
    value["reservation"].update(project_id="another-project", state="active")
    atomic_replace(other, value)

    with pytest.raises(RuntimeError, match="ID conflicts"):
        migrate_project(runtime, cfg)

    assert source.exists() and not destination.exists()
    assert read_json(other) == value
    assert not runtime.matching_binding(cfg).enabled


@pytest.mark.parametrize("is_cpu", [False, True])
def test_released_source_cannot_delete_conflicting_target_copy(tmp_path, monkeypatch, is_cpu):
    cfg, runtime, source, destination = migration_fixture(tmp_path, is_cpu, True)
    original_delete = DurableIO.delete

    def fail_unlink(io, path, **kwargs):
        if path == source:
            raise OSError("before source unlink")
        return original_delete(io, path, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(DurableIO, "delete", fail_unlink)
        with pytest.raises(OSError, match="before source unlink"):
            migrate_project(runtime, cfg)
    conflict = read_json(destination)
    (release_cpu if is_cpu else release)(runtime.root, source.stem)
    conflict["reservation"]["task_id"] = "another-task"
    atomic_replace(destination, conflict)
    released = runtime.paths["cpu_released" if is_cpu else "released"] / source.name
    before = {path: path.read_bytes() for path in (source, destination, released)}

    with pytest.raises(RuntimeError, match="ID conflicts"):
        migrate_project(runtime, cfg)

    assert {path: path.read_bytes() for path in before} == before
    assert not runtime.matching_binding(cfg).enabled


@pytest.mark.parametrize("is_nested", [False, True])
def test_unreadable_legacy_evidence_does_not_complete_migration(tmp_path, monkeypatch, is_nested):
    cfg, runtime, source, destination = migration_fixture(tmp_path, False, True)
    paths = local_paths(cfg.runtime_root)
    directory = paths["termination_decisions"] / "old-task-attempt-1" if is_nested else paths["observations"]
    evidence = directory / "retained.json"
    atomic_replace(evidence, {"fixture": "must remain discoverable"})
    original = os.scandir

    def unavailable(path):
        if Path(path) == directory:
            raise PermissionError("legacy evidence inaccessible")
        return original(path)

    with monkeypatch.context() as patch:
        patch.setattr(os, "scandir", unavailable)
        with pytest.raises(PermissionError, match="legacy evidence inaccessible"):
            migrate_project(runtime, cfg)
    binding = runtime.matching_binding(cfg)
    assert not binding.enabled
    assert read_json(runtime.migration_path(binding.project_id))["migration"]["state"] == "blocked"
    assert evidence.exists() and destination.exists()


@pytest.mark.parametrize("is_cpu", [False, True])
@pytest.mark.parametrize("is_source", [False, True])
def test_invalid_reservation_lane_stops_import_before_source_deletion(tmp_path, is_cpu, is_source):
    cfg, runtime, source, destination = migration_fixture(tmp_path, is_cpu, True)
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name, enabled=False, adopt_existing=True)
    lane = "cpu_provisional" if is_cpu else "provisional"
    directory = local_paths(cfg.runtime_root)[lane] if is_source else runtime.paths[lane]
    directory.rmdir()
    directory.write_text("damaged reservation lane")

    with pytest.raises(NotADirectoryError):
        project_admin._import_legacy_reservations(runtime, binding, cfg)

    assert source.exists() and not destination.exists()
    assert not runtime.matching_binding(cfg).enabled


@pytest.mark.parametrize("is_cpu", [False, True])
@pytest.mark.parametrize("field", ["project_id", "shared_root", "machine_name"])
def test_migration_cannot_reassign_another_owners_source(tmp_path, is_cpu, field):
    cfg, runtime, source, destination = migration_fixture(tmp_path, is_cpu, True)
    value = read_json(source)
    value["reservation"][field] = "other-owner"
    atomic_replace(source, value)

    with pytest.raises(RuntimeError, match="another project or machine"):
        migrate_project(runtime, cfg)

    assert read_json(source) == value
    assert not destination.exists()
    assert not runtime.matching_binding(cfg).enabled


def test_cpu_import_failure_keeps_previously_moved_gpu_occupancy(tmp_path, monkeypatch):
    cfg, runtime, source, destination = migration_fixture(tmp_path, True, True)
    gpu = reserve(cfg.runtime_root, "gpu-task", [0])["reservation"]
    gpu_source = local_paths(cfg.runtime_root)["provisional"] / f"{gpu['reservation_id']}.json"
    gpu_destination = runtime.paths["provisional"] / gpu_source.name
    original_replace = project_admin.atomic_replace

    def fail_cpu(path, value):
        if path == destination:
            raise OSError("CPU import interrupted")
        return original_replace(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(project_admin, "atomic_replace", fail_cpu)
        with pytest.raises(OSError, match="CPU import interrupted"):
            migrate_project(runtime, cfg)
    assert not gpu_source.exists() and gpu_destination.exists()
    assert source.exists() and not destination.exists()
    assert not runtime.matching_binding(cfg).enabled
    assert migrate_project(runtime, cfg).enabled
    assert gpu_destination.exists() and destination.exists()
    assert not source.exists()


def test_migration_waits_for_source_cpu_lock(tmp_path, monkeypatch):
    cfg, runtime, source, destination = migration_fixture(tmp_path, True, True)
    reached, finished = Event(), Event()
    errors = []
    original_import = project_admin._import_reservation_lane

    def lane(*args):
        if args[4] == "cpu-lane.lock":
            reached.set()
        return original_import(*args)

    def migrate():
        try:
            migrate_project(runtime, cfg)
        except Exception as exc:
            errors.append(exc)
        finally:
            finished.set()

    monkeypatch.setattr(project_admin, "_import_reservation_lane", lane)
    thread = Thread(target=migrate)
    try:
        with exclusive(local_paths(cfg.runtime_root)["locks"] / "cpu-lane.lock"):
            thread.start()
            assert reached.wait(5)
            assert not finished.wait(0.1)
            assert source.exists() and not destination.exists()
    finally:
        thread.join(10)
    assert not thread.is_alive() and errors == []
    assert not source.exists() and destination.exists()


@pytest.mark.parametrize("is_cpu", [False, True])
def test_duplicate_source_id_in_two_phases_is_rejected(tmp_path, is_cpu):
    cfg, runtime, source, destination = migration_fixture(tmp_path, is_cpu, True)
    duplicate = local_paths(cfg.runtime_root)["cpu_provisional" if is_cpu else "provisional"] / source.name
    value = read_json(source)
    value["reservation"]["state"] = "provisional"
    atomic_replace(duplicate, value)
    with pytest.raises(RuntimeError, match="multiple phases"):
        migrate_project(runtime, cfg)
    assert source.exists() and duplicate.exists() and not destination.exists()
    assert not runtime.matching_binding(cfg).enabled


@pytest.mark.parametrize("is_cpu", [False, True])
@pytest.mark.parametrize("is_source", [False, True])
def test_nested_reservation_does_not_hide_occupancy(tmp_path, is_cpu, is_source):
    cfg, runtime, source, destination = migration_fixture(tmp_path, is_cpu, True)
    root = source.parent if is_source else destination.parent
    nested = root / "leftover" / "nested.json"
    atomic_replace(nested, read_json(source))
    with pytest.raises(RuntimeError, match="nested path"):
        migrate_project(runtime, cfg)
    assert source.exists() and nested.exists() and not destination.exists()
    assert not runtime.matching_binding(cfg).enabled


@pytest.mark.parametrize("slots", [None, 0, -1, True, "1"])
def test_invalid_unrelated_cpu_occupancy_blocks_import(tmp_path, slots):
    cfg, runtime, source, destination = migration_fixture(tmp_path, True, True)
    other = runtime.paths["cpu_active"] / "other.json"
    atomic_replace(
        other,
        {
            "reservation": {
                "reservation_id": "other",
                "task_id": "other-task",
                "project_id": "other-project",
                "state": "active",
                "cpu_slots": slots,
            }
        },
    )
    before = other.read_bytes()
    with pytest.raises(RuntimeError, match="invalid slots"):
        migrate_project(runtime, cfg)
    assert source.exists() and not destination.exists() and other.read_bytes() == before
