"""Cleanup acknowledgement must follow durable local responsibility handoff."""

import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.commands.cleanup import clean, reconcile_cleanup_operations
from qqtools.plugins.qexp.runner import _process_start_time_ticks, _publish_exit_observation
from qqtools.plugins.qexp.runtime.paths import attempt_path, local_paths, task_path
from qqtools.plugins.qexp.runtime.resources.cpu_lane import (
    attach_cpu,
    cpu_reservation_snapshot,
    reserve_cpu,
    set_cpu_lane_capacity,
)
from qqtools.plugins.qexp.runtime.responsibility import responsibility_root
from qqtools.plugins.qexp.runtime.responsibility_cleanup import CLEANUP_FORMAT
from qqtools.plugins.qexp.runtime.responsibility_store import Conflict, DurableIO, Ledger, Unavailable
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.scheduler import claim_task, fail_attempt

pytestmark = pytest.mark.integration


def cleanup_capture_fixture(tmp_path, *, is_legacy_source):
    from dataclasses import replace

    from qqtools.plugins.qexp.agent.context import MachineRuntime
    from qqtools.plugins.qexp.runtime.responsibility_capture import WriterCaptureCheckpoint

    cfg, task, attempt = terminal_task(tmp_path)
    source = cfg.runtime_root
    reservation_root = cfg.runtime_root
    if is_legacy_source:
        runtime = MachineRuntime(tmp_path / "machine")
        binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
        cfg = replace(cfg, runtime_root=runtime.project_paths(binding.project_id)["root"])
        cfg.runtime_root.mkdir(parents=True, exist_ok=True)
        atomic_replace(
            runtime.migration_path(binding.project_id),
            {"migration": {"state": "active", "legacy_runtime_root": str(source)}},
        )
        reservation_root = runtime.root
        ledger = Ledger.open_or_create(responsibility_root(cfg.runtime_root))
        ledger.capture_source(
            attempt.attempt_id, {"task_id": task.task_id, "attempt_number": attempt.attempt_number}, source
        )
        assert source.parent != cfg.runtime_root.parent
    ledger = Ledger.open_or_create(responsibility_root(source))
    return cfg, task, attempt, reservation_root, WriterCaptureCheckpoint(ledger, source)


@pytest.mark.parametrize("is_legacy_source", [False, True])
def test_capture_cannot_begin_between_cleanup_acknowledgement_and_finalization(tmp_path, monkeypatch, is_legacy_source):
    from qqtools.plugins.qexp.commands import cleanup as cleanup_module
    from qqtools.plugins.qexp.runtime.responsibility_capture import has_pending_writer_capture

    cfg, task, _, reservation_root, checkpoint = cleanup_capture_fixture(tmp_path, is_legacy_source=is_legacy_source)
    original_cleanup = cleanup_module._cleanup_local_resources
    original_write = cleanup_module.atomic_replace
    original_finalize = cleanup_module._finalize_cleanup_operation
    checked = set()

    def assert_excluded(boundary):
        with pytest.raises(Conflict, match="busy"):
            with checkpoint.observe():
                pytest.fail(f"capture overlapped {boundary}")
        checked.add(boundary)

    def cleanup(*args, **kwargs):
        result = original_cleanup(*args, **kwargs)
        assert_excluded("local_return")
        return result

    def write(path, value):
        if cfg.machine_name in value.get("cleanup", {}).get("acknowledgements", {}):
            assert_excluded("acknowledgement")
        return original_write(path, value)

    def finalize(*args, **kwargs):
        assert_excluded("finalization")
        return original_finalize(*args, **kwargs)

    monkeypatch.setattr(cleanup_module, "_cleanup_local_resources", cleanup)
    monkeypatch.setattr(cleanup_module, "atomic_replace", write)
    monkeypatch.setattr(cleanup_module, "_finalize_cleanup_operation", finalize)
    clean(cfg, task_id=task.task_id, reservation_runtime_root=reservation_root)
    assert checked == {"local_return", "acknowledgement", "finalization"}
    assert not task_path(cfg.shared_root, task.task_id).exists()
    assert not has_pending_writer_capture(checkpoint.runtime_root)


@pytest.mark.parametrize("is_legacy_source", [False, True])
def test_pending_capture_blocks_finalization_of_previously_acknowledged_cleanup(
    tmp_path, monkeypatch, is_legacy_source
):
    from qqtools.plugins.qexp.commands import cleanup as cleanup_module

    cfg, task, attempt, reservation_root, checkpoint = cleanup_capture_fixture(
        tmp_path, is_legacy_source=is_legacy_source
    )
    with monkeypatch.context() as patch:
        patch.setattr(cleanup_module, "_finalize_cleanup_if_ready", lambda *_args: [])
        result = clean(cfg, task_id=task.task_id, reservation_runtime_root=reservation_root)
    assert result["operations"][task.task_id]["pending_machines"] == []
    assert task_path(cfg.shared_root, task.task_id).exists()
    assert Ledger(responsibility_root(cfg.runtime_root)).find(attempt.attempt_id) is None
    with checkpoint.observe():
        pass
    result = reconcile_cleanup_operations(cfg, reservation_runtime_root=reservation_root)
    assert result[0]["state"] == "waiting_ack" and result[0]["blockers"] == ["writer_capture_pending"]
    assert task_path(cfg.shared_root, task.task_id).exists()
    assert attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number).exists()


def test_pending_writer_capture_blocks_acknowledgement_without_visible_evidence(tmp_path):
    from qqtools.plugins.qexp.runtime.responsibility_capture import WriterCaptureCheckpoint

    cfg, task, attempt = terminal_task(tmp_path)
    ledger = Ledger.open_or_create(responsibility_root(cfg.runtime_root))
    with WriterCaptureCheckpoint(ledger, cfg.runtime_root).observe():
        pass
    result = clean(cfg, task_id=task.task_id, reservation_runtime_root=cfg.runtime_root)
    operation = result["operations"][task.task_id]
    assert operation["state"] == "waiting_ack"
    assert operation["pending_machines"] == [cfg.machine_name]
    assert task_path(cfg.shared_root, task.task_id).exists()
    assert attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number).exists()


@pytest.mark.parametrize("is_acknowledged", [False, True])
def test_legacy_cleanup_honors_target_capture_before_source_hold(tmp_path, monkeypatch, is_acknowledged):
    from dataclasses import replace

    from qqtools.plugins.qexp.commands import cleanup as cleanup_module
    from qqtools.plugins.qexp.runtime.responsibility_capture import WriterCaptureCheckpoint, has_pending_writer_capture

    cfg, task, attempt, reservation_root, source_checkpoint = cleanup_capture_fixture(tmp_path, is_legacy_source=True)
    source_cfg = replace(cfg, runtime_root=source_checkpoint.runtime_root)
    if is_acknowledged:
        with monkeypatch.context() as patch:
            patch.setattr(cleanup_module, "_finalize_cleanup_if_ready", lambda *_args: [])
            result = clean(source_cfg, task_id=task.task_id, reservation_runtime_root=reservation_root)
        assert result["operations"][task.task_id]["pending_machines"] == []
    ledger = Ledger(responsibility_root(cfg.runtime_root))
    checkpoint = WriterCaptureCheckpoint(ledger, cfg.runtime_root, legacy_source=source_cfg.runtime_root)

    def fail_source_hold():
        raise OSError("source hold interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(checkpoint, "_retain_source", fail_source_hold)
        with pytest.raises(OSError, match="source hold interrupted"):
            with checkpoint.observe():
                pytest.fail("partial hold establishment admitted an observation")
    assert has_pending_writer_capture(cfg.runtime_root)
    assert not has_pending_writer_capture(source_cfg.runtime_root)

    if is_acknowledged:
        operation = reconcile_cleanup_operations(source_cfg, reservation_runtime_root=reservation_root)[0]
    else:
        result = clean(source_cfg, task_id=task.task_id, reservation_runtime_root=reservation_root)
        operation = result["operations"][task.task_id]
    assert operation["state"] == "waiting_ack"
    assert operation["blockers"] == ["writer_capture_pending"]
    assert operation["removed"] == []
    assert task_path(cfg.shared_root, task.task_id).exists()
    assert attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number).exists()
    with WriterCaptureCheckpoint(ledger, cfg.runtime_root, legacy_source=source_cfg.runtime_root).observe():
        pass
    assert has_pending_writer_capture(source_cfg.runtime_root)


@pytest.mark.parametrize("source", [None, "", [], "relative"])
def test_managed_cleanup_rejects_unclassifiable_migration_source(tmp_path, source):
    cfg, task, attempt, reservation_root, _ = cleanup_capture_fixture(tmp_path, is_legacy_source=True)
    atomic_replace(
        cfg.runtime_root / "migration.json", {"migration": {"state": "active", "legacy_runtime_root": source}}
    )

    with pytest.raises(RuntimeError, match="legacy source"):
        clean(cfg, task_id=task.task_id, reservation_runtime_root=reservation_root)
    assert task_path(cfg.shared_root, task.task_id).exists()
    assert attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number).exists()


@pytest.mark.parametrize("has_explicit_runtime", [False, True])
def test_previously_acknowledged_cleanup_cannot_continue_after_binding_removal(
    tmp_path, monkeypatch, has_explicit_runtime
):
    from qqtools.plugins.qexp.agent.context import MachineRuntime
    from qqtools.plugins.qexp.commands import cleanup as cleanup_module

    cfg, task, attempt, reservation_root, _ = cleanup_capture_fixture(tmp_path, is_legacy_source=True)
    runtime = MachineRuntime(reservation_root)
    with monkeypatch.context() as patch:
        patch.setattr(cleanup_module, "_finalize_cleanup_if_ready", lambda *_args: [])
        result = clean(cfg, task_id=task.task_id, reservation_runtime_root=reservation_root)
    assert result["operations"][task.task_id]["pending_machines"] == []
    binding = runtime.matching_binding(cfg)
    runtime.set_enabled(binding.project_id, False)
    runtime.remove_binding(binding.project_id)
    assert not cfg.runtime_root.exists()
    monkeypatch.setenv("QEXP_MACHINE_RUNTIME_ROOT", str(reservation_root))

    with pytest.raises(RuntimeError, match="registered project binding"):
        reconcile_cleanup_operations(cfg, reservation_runtime_root=reservation_root if has_explicit_runtime else None)
    assert task_path(cfg.shared_root, task.task_id).exists()
    assert attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number).exists()
    assert not cfg.runtime_root.exists()


def test_cleanup_cursor_publication_excludes_binding_removal(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.agent.context import MachineRuntime
    from qqtools.plugins.qexp.runtime import operation_store

    cfg, task, _, reservation_root, _ = cleanup_capture_fixture(tmp_path, is_legacy_source=True)
    runtime = MachineRuntime(reservation_root)
    binding = runtime.matching_binding(cfg)
    runtime.set_enabled(binding.project_id, False)
    original = operation_store.atomic_replace
    checked = []

    def write(path, value):
        if path == local_paths(cfg.runtime_root)["maintenance_cursors"] / "cleanup.json":
            with pytest.raises(RuntimeError, match="writer capture"):
                runtime.remove_binding(binding.project_id)
            assert cfg.runtime_root.exists()
            checked.append(path)
        return original(path, value)

    monkeypatch.setattr(operation_store, "atomic_replace", write)
    result = clean(cfg, task_id=task.task_id, reservation_runtime_root=reservation_root)
    assert checked and result["operations"][task.task_id]["state"] == "completed"
    assert runtime.matching_binding(cfg) is not None


def terminal_task(tmp_path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "finished"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert fail_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "test_failure")
    return cfg, task, attempt


def registration(cfg, task, attempt, pid):
    path = local_paths(cfg.runtime_root)["registrations"] / f"{attempt.attempt_id}.json"
    atomic_replace(
        path,
        {
            "process_registration": {
                "protocol_version": 1,
                "task_id": task.task_id,
                "attempt_id": attempt.attempt_id,
                "wrapper_pid": pid,
                "wrapper_start_time_ticks": _process_start_time_ticks(pid),
            }
        },
    )
    return path


def exited_pid():
    with subprocess.Popen([sys.executable, "-c", "pass"]) as process:
        assert process.wait(timeout=10) == 0
        return process.pid


def test_cleanup_waits_for_live_wrapper_and_preserves_its_late_write(tmp_path):
    cfg, task, attempt = terminal_task(tmp_path)
    path = registration(cfg, task, attempt, os.getpid())

    result = clean(cfg, task_id=task.task_id, reservation_runtime_root=cfg.runtime_root)

    operation = result["operations"][task.task_id]
    assert operation["state"] == "waiting_ack"
    assert operation["pending_machines"] == [cfg.machine_name]
    assert operation["blockers"] == [f"local_writer_unresolved:{attempt.attempt_id}"]
    assert path.exists()
    assert task_path(cfg.shared_root, task.task_id).exists()
    assert attempt_path(cfg.shared_root, task.task_id, 1).exists()
    _publish_exit_observation(cfg, attempt.attempt_id, 1, task_id=task.task_id)
    value = read_json(path)
    value["process_registration"].update(wrapper_pid=exited_pid(), wrapper_start_time_ticks=1)
    atomic_replace(path, value)

    result = reconcile_cleanup_operations(cfg, reservation_runtime_root=cfg.runtime_root)

    assert result[0]["state"] == "completed"
    assert not path.exists()
    assert not (local_paths(cfg.runtime_root)["observations"] / path.name).exists()
    assert not task_path(cfg.shared_root, task.task_id).exists()


@pytest.mark.parametrize("cut", ["before_evidence", "after_evidence"])
def test_cleanup_storage_failure_keeps_receipt_and_delays_ack(tmp_path, monkeypatch, cut):
    cfg, task, attempt = terminal_task(tmp_path)
    path = registration(cfg, task, attempt, exited_pid())
    value = read_json(path)
    value["process_registration"]["wrapper_start_time_ticks"] = 1
    atomic_replace(path, value)
    _publish_exit_observation(cfg, attempt.attempt_id, 1, task_id=task.task_id)
    observation = local_paths(cfg.runtime_root)["observations"] / path.name
    ledger_root = responsibility_root(cfg.runtime_root)
    original = DurableIO.delete

    def interrupted_delete(io, target, **kwargs):
        if target == path and cut == "before_evidence":
            raise OSError("interrupted cleanup")
        result = original(io, target, **kwargs)
        if target == observation and cut == "after_evidence":
            raise OSError("interrupted cleanup")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(DurableIO, "delete", interrupted_delete)
        with pytest.raises(OSError, match="interrupted cleanup"):
            clean(cfg, task_id=task.task_id, reservation_runtime_root=cfg.runtime_root)

    entry = Ledger(ledger_root).lookup(attempt.attempt_id)
    assert entry["stage"] == "maintenance"
    assert entry["cleanup_receipt"]["format"] == CLEANUP_FORMAT
    assert entry["payload"] == {"task_id": task.task_id, "attempt_number": 1}
    operation_path = cfg.shared_root / "operations" / "cleanup" / f"{task.task_id}.json"
    assert read_json(operation_path)["cleanup"]["acknowledgements"] == {}
    assert task_path(cfg.shared_root, task.task_id).exists()
    assert path.exists() == (cut == "before_evidence")

    result = reconcile_cleanup_operations(cfg, reservation_runtime_root=cfg.runtime_root)

    assert result[0]["state"] == "completed"
    assert not task_path(cfg.shared_root, task.task_id).exists()
    assert not path.exists() and not observation.exists()
    assert Ledger(ledger_root).find(attempt.attempt_id) is None


@pytest.mark.parametrize("is_active", [False, True])
def test_cleanup_releases_cpu_reservations_without_touching_other_tasks(tmp_path, is_active):
    cfg, task, attempt = terminal_task(tmp_path)
    set_cpu_lane_capacity(cfg.runtime_root, capacity=2)
    reservation = reserve_cpu(
        cfg.runtime_root,
        task.task_id,
        1,
        attempt_id=attempt.attempt_id,
        fencing_token=attempt.current_fencing_token,
    )["reservation"]
    other = reserve_cpu(cfg.runtime_root, "unrelated-task", 1)["reservation"]
    if is_active:
        attach_cpu(cfg.runtime_root, reservation["reservation_id"], attempt.attempt_id, attempt.current_fencing_token)
    try:
        result = clean(cfg, task_id=task.task_id, reservation_runtime_root=cfg.runtime_root)

        assert result["operations"][task.task_id]["state"] == "completed"
        _policy, remaining = cpu_reservation_snapshot(cfg.runtime_root)
        assert [item["reservation_id"] for item in remaining] == [other["reservation_id"]]
        archived = local_paths(cfg.runtime_root)["cpu_released"] / f"{reservation['reservation_id']}.json"
        assert read_json(archived)["reservation"]["release_reason"] == "task_cleanup"
    finally:
        from qqtools.plugins.qexp.runtime.resources.reservations import release

        release(cfg.runtime_root, other["reservation_id"], "test_cleanup")


def test_cleanup_slices_decisions_before_acknowledging_and_deleting_truth(tmp_path):
    cfg, task, attempt = terminal_task(tmp_path)
    directory = local_paths(cfg.runtime_root)["termination_decisions"] / attempt.attempt_id
    for index in range(9):
        atomic_replace(directory / f"{index}.json", {"termination_decision": {"attempt_id": attempt.attempt_id}})

    result = clean(cfg, task_id=task.task_id, reservation_runtime_root=cfg.runtime_root)

    assert result["operations"][task.task_id]["state"] == "waiting_ack"
    assert result["operations"][task.task_id]["blockers"] == [f"local_cleanup_pending:{attempt.attempt_id}"]
    assert len(list(directory.iterdir())) == 1
    assert task_path(cfg.shared_root, task.task_id).exists()
    entry = Ledger(responsibility_root(cfg.runtime_root)).lookup(attempt.attempt_id)
    assert entry["cleanup_receipt"]["basis"] == "terminal_attempt"

    result = reconcile_cleanup_operations(cfg, reservation_runtime_root=cfg.runtime_root)

    assert result[0]["state"] == "completed"
    assert not directory.exists()
    assert not task_path(cfg.shared_root, task.task_id).exists()


@pytest.mark.parametrize("has_mismatched_receipt", [False, True])
def test_cleanup_rejects_a_membership_for_different_truth(tmp_path, has_mismatched_receipt):
    cfg, task, attempt = terminal_task(tmp_path)
    _publish_exit_observation(cfg, attempt.attempt_id, 1, task_id=task.task_id)
    path = local_paths(cfg.runtime_root)["observations"] / f"{attempt.attempt_id}.json"
    ledger = Ledger.open_or_create(responsibility_root(cfg.runtime_root))
    generation = ledger.publish(attempt.attempt_id, {"task_id": "different-task", "attempt_number": 1})
    if has_mismatched_receipt:
        ledger.handoff(
            attempt.attempt_id,
            generation,
            cleanup_receipt={
                "format": CLEANUP_FORMAT,
                "task_id": "different-task",
                "attempt_id": attempt.attempt_id,
                "basis": "terminal_attempt",
            },
        )

    with pytest.raises(Conflict, match="does not match Attempt truth"):
        clean(cfg, task_id=task.task_id, reservation_runtime_root=cfg.runtime_root)

    assert path.exists()
    assert task_path(cfg.shared_root, task.task_id).exists()


def test_cleanup_discovers_noncanonical_attempt_and_preserves_other_members(tmp_path):
    cfg, task, attempt = terminal_task(tmp_path)
    path = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    value = read_json(path)
    value["attempt"]["attempt_id"] = "historical-noncanonical-id"
    atomic_replace(path, value)
    identity = value["attempt"]["attempt_id"]
    _publish_exit_observation(cfg, identity, 1, task_id=task.task_id)
    observation = local_paths(cfg.runtime_root)["observations"] / f"{identity}.json"
    ledger = Ledger.open_or_create(responsibility_root(cfg.runtime_root))
    ledger.publish("other-member", {"task_id": "other-task", "attempt_number": 1})

    result = clean(cfg, task_id=task.task_id, reservation_runtime_root=cfg.runtime_root)

    assert result["operations"][task.task_id]["state"] == "completed"
    assert not observation.exists()
    assert ledger.find(identity) is None
    assert ledger.lookup("other-member")["stage"] == "active"


def test_cleanup_resolves_imported_noncanonical_locator_from_attempt_truth(tmp_path):
    from qqtools.plugins.qexp.runtime.responsibility_import import move_legacy_record

    cfg, task, attempt = terminal_task(tmp_path)
    path = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    value = read_json(path)
    identity = "historical-imported-id"
    value["attempt"]["attempt_id"] = identity
    atomic_replace(path, value)
    legacy_root = tmp_path / "old"
    source = local_paths(legacy_root)["observations"] / f"{identity}.json"
    observation = {"exit_observation": {"protocol_version": 1, "attempt_id": identity, "observed_exit_code": 1}}
    atomic_replace(source, observation)
    destination = local_paths(cfg.runtime_root)["observations"] / source.name
    move_legacy_record(
        cfg.runtime_root, legacy_root, "observations", source, destination, is_destination_authoritative=True
    )
    ledger = Ledger(responsibility_root(cfg.runtime_root))
    assert ledger.lookup(identity)["payload"] == {"task_id": None, "attempt_number": None}
    # A retained late record forces a second slice after the durable handoff.
    atomic_replace(source, observation)

    result = clean(cfg, task_id=task.task_id, reservation_runtime_root=cfg.runtime_root)

    assert result["operations"][task.task_id]["state"] == "waiting_ack"
    assert not source.exists() and destination.exists()
    entry = ledger.lookup(identity)
    assert entry["payload"] == {"task_id": task.task_id, "attempt_number": attempt.attempt_number}
    assert entry["legacy_source"] == str(legacy_root)
    assert entry["cleanup_receipt"]["basis"] == "terminal_attempt"
    assert task_path(cfg.shared_root, task.task_id).exists()

    result = reconcile_cleanup_operations(cfg, reservation_runtime_root=cfg.runtime_root)

    assert result[0]["state"] == "completed"
    assert ledger.find(identity) is None
    assert not destination.exists()
    assert not task_path(cfg.shared_root, task.task_id).exists()


def test_membership_find_distinguishes_absence_from_unavailable_bucket(tmp_path):
    from qqtools.plugins.qexp.runtime.responsibility_store import identity_key

    ledger = Ledger.open_or_create(tmp_path / "members")
    assert ledger.find("absent") is None
    bucket = int(identity_key("absent")[0], 16)
    (ledger.root / str(bucket) / "header").unlink()

    with pytest.raises(Unavailable, match="incomplete bucket"):
        ledger.find("absent")


@pytest.mark.parametrize("has_final_write", [False, True])
def test_missing_attempt_truth_does_not_allow_deleting_a_live_writer(tmp_path, has_final_write):
    cfg, task, attempt = terminal_task(tmp_path)
    path = registration(cfg, task, attempt, os.getpid())
    attempt_path(cfg.shared_root, task.task_id, 1).unlink()
    if has_final_write:
        _publish_exit_observation(cfg, attempt.attempt_id, 1, task_id=task.task_id)

    result = clean(cfg, task_id=task.task_id, reservation_runtime_root=cfg.runtime_root)

    assert result["operations"][task.task_id]["state"] == "waiting_ack"
    assert result["operations"][task.task_id]["blockers"] == [f"local_writer_unresolved:{attempt.attempt_id}"]
    assert path.exists()
    assert task_path(cfg.shared_root, task.task_id).exists()


@pytest.mark.parametrize("is_canonical", [False, True])
@pytest.mark.parametrize("has_task_truth", [False, True])
def test_unmatched_cleanup_persists_operation_proof_before_deletion(
    tmp_path, monkeypatch, is_canonical, has_task_truth
):
    cfg, task, attempt = terminal_task(tmp_path)
    identity = attempt.attempt_id if is_canonical else "historical-opaque-attempt"
    attempt_path(cfg.shared_root, task.task_id, 1).unlink()
    _publish_exit_observation(cfg, identity, 1, task_id=task.task_id)
    path = local_paths(cfg.runtime_root)["observations"] / f"{identity}.json"
    original = DurableIO.delete

    def interrupted(io, target, **kwargs):
        result = original(io, target, **kwargs)
        if target == path:
            raise OSError("unmatched evidence deletion interrupted")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(DurableIO, "delete", interrupted)
        with pytest.raises(OSError, match="unmatched evidence deletion interrupted"):
            clean(cfg, task_id=task.task_id, reservation_runtime_root=cfg.runtime_root)

    ledger = Ledger(responsibility_root(cfg.runtime_root))
    entry = ledger.lookup(identity)
    assert entry["stage"] == "maintenance"
    assert entry["payload"] == {"task_id": task.task_id, "attempt_number": 1 if is_canonical else None}
    operation = read_json(cfg.shared_root / "operations" / "cleanup" / f"{task.task_id}.json")["cleanup"]
    assert entry["cleanup_receipt"]["basis"] == "task_cleanup"
    assert entry["cleanup_receipt"]["operation_id"] == operation["operation_id"]
    assert operation["acknowledgements"] == {}
    assert not path.exists()
    if not has_task_truth:
        task_path(cfg.shared_root, task.task_id).unlink()

    # Replay has only the receipt, no original local record or shared Attempt.
    result = reconcile_cleanup_operations(cfg, reservation_runtime_root=cfg.runtime_root)
    assert result[0]["state"] == "completed"
    assert ledger.find(identity) is None
    assert not task_path(cfg.shared_root, task.task_id).exists()


def test_known_attempt_resolves_local_backfill_without_legacy_source(tmp_path):
    cfg, task, attempt = terminal_task(tmp_path)
    path = attempt_path(cfg.shared_root, task.task_id, 1)
    value = read_json(path)
    identity = "opaque-local-backfill"
    value["attempt"]["attempt_id"] = identity
    atomic_replace(path, value)
    _publish_exit_observation(cfg, identity, 1, task_id=task.task_id)
    ledger = Ledger.open_or_create(responsibility_root(cfg.runtime_root))
    ledger.capture_local(identity, {"task_id": task.task_id, "attempt_number": None})

    result = clean(cfg, task_id=task.task_id, reservation_runtime_root=cfg.runtime_root)

    assert result["operations"][task.task_id]["state"] == "completed"
    assert ledger.find(identity) is None


def test_unmatched_cleanup_cannot_use_advisory_membership_as_task_ownership(tmp_path):
    cfg, task, attempt = terminal_task(tmp_path)
    attempt_path(cfg.shared_root, task.task_id, 1).unlink()
    identity = "opaque-unclassified-evidence"
    _publish_exit_observation(cfg, identity, 1)
    observation = local_paths(cfg.runtime_root)["observations"] / f"{identity}.json"
    ledger = Ledger.open_or_create(responsibility_root(cfg.runtime_root))
    ledger.capture_local(identity, {"task_id": task.task_id, "attempt_number": None})

    result = clean(cfg, task_id=task.task_id, reservation_runtime_root=cfg.runtime_root)

    assert result["operations"][task.task_id]["state"] == "waiting_ack"
    assert observation.exists()
    assert ledger.lookup(identity)["stage"] == "active"


def test_unmatched_cleanup_waits_for_final_write_even_after_wrapper_exit(tmp_path):
    cfg, task, attempt = terminal_task(tmp_path)
    path = registration(cfg, task, attempt, exited_pid())
    value = read_json(path)
    value["process_registration"]["wrapper_start_time_ticks"] = 1
    atomic_replace(path, value)
    attempt_path(cfg.shared_root, task.task_id, 1).unlink()

    result = clean(cfg, task_id=task.task_id, reservation_runtime_root=cfg.runtime_root)

    assert result["operations"][task.task_id]["state"] == "waiting_ack"
    assert path.exists()
    _publish_exit_observation(cfg, attempt.attempt_id, 1, task_id=task.task_id)
    assert reconcile_cleanup_operations(cfg, reservation_runtime_root=cfg.runtime_root)[0]["state"] == "completed"
    assert not path.exists()


def crash_unmatched_cleanup(cfg, task_id, path, boundary):
    original_publish = Ledger.publish
    original_handoff = Ledger.handoff
    original_delete = DurableIO.delete

    def publish(ledger, *args, **kwargs):
        result = original_publish(ledger, *args, **kwargs)
        if boundary == "capture":
            os._exit(86)
        return result

    def handoff(ledger, *args, **kwargs):
        result = original_handoff(ledger, *args, **kwargs)
        if boundary == "handoff":
            os._exit(86)
        return result

    def delete(io, target, **kwargs):
        result = original_delete(io, target, **kwargs)
        if boundary == "delete" and target == path:
            os._exit(86)
        return result

    Ledger.publish, Ledger.handoff, DurableIO.delete = publish, handoff, delete
    clean(cfg, task_id=task_id, reservation_runtime_root=cfg.runtime_root)


@pytest.mark.parametrize("boundary", ["capture", "handoff", "delete"])
def test_process_crash_keeps_unmatched_cleanup_discoverable(tmp_path, boundary):
    cfg, task, attempt = terminal_task(tmp_path)
    attempt_path(cfg.shared_root, task.task_id, 1).unlink()
    identity = "opaque-old-attempt"
    _publish_exit_observation(cfg, identity, 1, task_id=task.task_id)
    path = local_paths(cfg.runtime_root)["observations"] / f"{identity}.json"
    process = multiprocessing.get_context("fork").Process(
        target=crash_unmatched_cleanup, args=(cfg, task.task_id, path, boundary)
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
    ledger = Ledger(responsibility_root(cfg.runtime_root))
    entry = ledger.lookup(identity)
    assert entry["stage"] == ("active" if boundary == "capture" else "maintenance")
    assert path.exists() == (boundary != "delete")
    assert reconcile_cleanup_operations(cfg, reservation_runtime_root=cfg.runtime_root)[0]["state"] == "completed"
    assert not path.exists()
    assert ledger.find(identity) is None


def test_cpu_cleanup_in_machine_runtime_is_project_scoped(tmp_path):
    from qqtools.plugins.qexp.runtime.resources.reservations import release

    cfg, task, attempt = terminal_task(tmp_path)
    machine_root = tmp_path / "machine"
    project_id = read_json(cfg.shared_root / "project" / "identity.json")["project"]["project_id"]
    set_cpu_lane_capacity(machine_root, capacity=2)
    own = reserve_cpu(machine_root, task.task_id, 1, project_id=project_id)["reservation"]
    other = reserve_cpu(machine_root, task.task_id, 1, project_id="another-project")["reservation"]
    try:
        result = clean(cfg, task_id=task.task_id, reservation_runtime_root=machine_root)

        assert result["operations"][task.task_id]["state"] == "completed"
        _policy, remaining = cpu_reservation_snapshot(machine_root)
        assert [item["reservation_id"] for item in remaining] == [other["reservation_id"]]
    finally:
        release(machine_root, own["reservation_id"], "test_cleanup")
        release(machine_root, other["reservation_id"], "test_cleanup")
