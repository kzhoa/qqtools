"""Legacy inbox moves retain responsibility across interruption and late writes."""

from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.runtime.paths import local_paths
from qqtools.plugins.qexp.runtime.responsibility import responsibility_root
from qqtools.plugins.qexp.runtime.responsibility_cleanup import CLEANUP_FORMAT, CleanupRequest, complete_cleanup
from qqtools.plugins.qexp.runtime.responsibility_import import move_legacy_record
from qqtools.plugins.qexp.runtime.responsibility_store import Conflict, DurableIO, Ledger
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json

pytestmark = pytest.mark.integration


def test_retained_inbox_refresh_copies_direct_records_during_pending_capture(tmp_path, monkeypatch):
    import os

    from qqtools.plugins.qexp.runtime import responsibility_import as transport
    from qqtools.plugins.qexp.runtime.responsibility_capture import WriterCaptureCheckpoint

    cfg, _, _, target, source, value = migration_fixture(tmp_path)
    ledger = Ledger.open_or_create(responsibility_root(target))
    identity = source.stem
    ledger.capture_source(identity, {"task_id": "task", "attempt_number": 1}, cfg.runtime_root)
    with WriterCaptureCheckpoint(ledger, target, legacy_source=cfg.runtime_root).observe():
        pass
    for number in range(100):
        atomic_replace(source.with_name(f"unrelated-{number}.json"), {"unrelated": number})
    observation = local_paths(cfg.runtime_root)["observations"] / source.name
    atomic_replace(observation, {"exit_observation": {"attempt_id": identity, "observed_exit_code": 0}})
    originals = {path: path.read_bytes() for path in (source, observation)}
    with monkeypatch.context() as patch:
        patch.setattr(os, "scandir", lambda *_a: pytest.fail("retained inbox refresh enumerated history"))
        assert transport.refresh_legacy_inbox(ledger, target, identity) == 2
    assert read_json(local_paths(target)["registrations"] / source.name) == value
    assert read_json(local_paths(target)["observations"] / source.name) == read_json(observation)
    assert {path: path.read_bytes() for path in originals} == originals
    with monkeypatch.context() as patch:
        patch.setattr(transport, "atomic_replace", lambda *_a: pytest.fail("exact inbox copy was rewritten"))
        patch.setattr(DurableIO, "replace", lambda *_a: pytest.fail("exact membership was rewritten"))
        assert transport.refresh_legacy_inbox(ledger, target, identity) == 0
    assert not responsibility_root(cfg.runtime_root).exists()


@pytest.mark.parametrize("boundary", ["membership", "copy"])
def test_retained_inbox_refresh_recovers_failed_publication_without_source_loss(tmp_path, monkeypatch, boundary):
    from qqtools.plugins.qexp.runtime import responsibility_import as transport

    cfg, _, _, target, source, value = migration_fixture(tmp_path)
    ledger = Ledger.open_or_create(responsibility_root(target))
    ledger.capture_source(source.stem, {"task_id": "task", "attempt_number": 1}, cfg.runtime_root)
    original = transport.atomic_replace

    def fail(*_args, **_kwargs):
        raise OSError("retained refresh interrupted")

    def fail_copy(path, value):
        original(path, value)
        fail()

    with monkeypatch.context() as patch:
        if boundary == "membership":
            patch.setattr(ledger, "capture_source", fail)
        else:
            patch.setattr(transport, "atomic_replace", fail_copy)
        with pytest.raises(OSError, match="retained refresh interrupted"):
            transport.refresh_legacy_inbox(ledger, target, source.stem)
    assert read_json(source) == value
    transport.refresh_legacy_inbox(Ledger(ledger.root), target, source.stem)
    assert read_json(source) == read_json(local_paths(target)["registrations"] / source.name) == value


def test_retained_inbox_refresh_rejects_conflict_and_does_not_resurrect_cleanup(tmp_path):
    from qqtools.plugins.qexp.runtime.responsibility_import import refresh_legacy_inbox

    cfg, _, _, target, source, value = migration_fixture(tmp_path)
    ledger = Ledger.open_or_create(responsibility_root(target))
    identity = source.stem
    generation = ledger.capture_source(identity, {"task_id": "task", "attempt_number": 1}, cfg.runtime_root)
    destination = local_paths(target)["registrations"] / source.name
    conflicting = {"process_registration": {**value["process_registration"], "wrapper_pid": 99999992}}
    atomic_replace(destination, conflicting)
    with pytest.raises(Conflict, match="conflicts during retained refresh"):
        refresh_legacy_inbox(ledger, target, identity)
    assert read_json(source) == value and read_json(destination) == conflicting
    destination.unlink()
    ledger.handoff(identity, generation, cleanup_receipt={"retained": "test"})
    assert refresh_legacy_inbox(ledger, target, identity) == 0
    assert not destination.exists() and source.exists()


def test_active_capture_excludes_retained_inbox_copy(tmp_path):
    from qqtools.plugins.qexp.runtime.responsibility_capture import WriterCaptureCheckpoint
    from qqtools.plugins.qexp.runtime.responsibility_import import refresh_legacy_inbox

    cfg, _, _, target, source, _ = migration_fixture(tmp_path)
    ledger = Ledger.open_or_create(responsibility_root(target))
    ledger.capture_source(source.stem, {"task_id": "task", "attempt_number": 1}, cfg.runtime_root)
    with WriterCaptureCheckpoint(ledger, target, legacy_source=cfg.runtime_root).observe():
        with pytest.raises(Conflict, match="busy"):
            refresh_legacy_inbox(ledger, target, source.stem)
    assert source.exists() and not (local_paths(target)["registrations"] / source.name).exists()


@pytest.mark.parametrize("lane", ["registrations", "events"])
@pytest.mark.parametrize("location", ["source", "destination"])
def test_pending_capture_blocks_record_and_event_source_deletion(tmp_path, lane, location):
    from qqtools.plugins.qexp.runtime.responsibility_capture import WriterCaptureCheckpoint

    cfg, runtime, binding, target, source, value = migration_fixture(tmp_path)
    if lane == "events":
        source = local_paths(cfg.runtime_root)[lane] / "machine" / "late.json"
        value = {"event_id": "late", "event_type": "late_diagnostic"}
        atomic_replace(source, value)
    root = cfg.runtime_root if location == "source" else target
    root.mkdir(parents=True, exist_ok=True)
    ledger = Ledger.open_or_create(responsibility_root(root))
    with runtime.migration_guard():
        with WriterCaptureCheckpoint(ledger, root).observe():
            pass
    with pytest.raises(RuntimeError, match="writer capture"):
        runtime.drain_legacy_runner_evidence(binding)
    assert read_json(source) == value


def test_late_legacy_event_transport_retains_source_until_target_is_durable(tmp_path, monkeypatch):
    cfg, runtime, binding, target, registration, value = migration_fixture(tmp_path)
    runtime.import_legacy_evidence(binding)
    event = {"event_id": "late", "timestamp": "2026-09-18T00:00:00Z", "event_type": "late_diagnostic"}
    source = local_paths(cfg.runtime_root)["events"] / "machine" / "late.json"
    destination = local_paths(target)["events"] / "machine" / source.name
    atomic_replace(source, event)
    original = DurableIO.sync_directory

    def interrupted(io, path, label):
        if label == "legacy_event_destination":
            raise OSError("event destination interrupted")
        return original(io, path, label)

    with monkeypatch.context() as patch:
        patch.setattr(DurableIO, "sync_directory", interrupted)
        with pytest.raises(OSError, match="event destination interrupted"):
            runtime.drain_legacy_runner_evidence(binding)
    assert read_json(source) == read_json(destination) == event
    runtime.drain_legacy_runner_evidence(binding)
    assert not source.exists() and read_json(destination) == event


def migration_fixture(tmp_path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "old")
    runtime = MachineRuntime(tmp_path / "machine")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    atomic_replace(
        runtime.migration_path(binding.project_id), {"migration": {"legacy_runtime_root": str(cfg.runtime_root)}}
    )
    target = runtime.project_paths(binding.project_id)["root"]
    source = local_paths(cfg.runtime_root)["registrations"] / "task-attempt-1.json"
    value = {
        "process_registration": {
            "protocol_version": 1,
            "task_id": "task",
            "attempt_id": "task-attempt-1",
            "wrapper_pid": 99999991,
            "wrapper_start_time_ticks": 1,
        }
    }
    atomic_replace(source, value)
    return cfg, runtime, binding, target, source, value


@pytest.mark.parametrize("lane", ["registrations", "events"])
@pytest.mark.parametrize("is_parent", [False, True])
def test_import_rejects_redirected_destination_without_losing_source(tmp_path, lane, is_parent):
    cfg, runtime, binding, target, source, value = migration_fixture(tmp_path)
    if lane == "events":
        runtime.import_legacy_evidence(binding)
        source = local_paths(cfg.runtime_root)[lane] / "machine" / "late.json"
        value = {"event_id": "late", "event_type": "late_diagnostic"}
        atomic_replace(source, value)
        destination = local_paths(target)[lane] / "machine" / source.name
    else:
        destination = local_paths(target)[lane] / source.name
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    copy = foreign / source.name
    atomic_replace(copy, value)
    if is_parent:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.parent.rmdir()
        destination.parent.symlink_to(foreign, target_is_directory=True)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(copy)
    before = copy.read_bytes()

    with pytest.raises(OSError, match="not a (regular file|real directory)"):
        runtime.drain_legacy_runner_evidence(binding)

    assert read_json(source) == value
    assert copy.read_bytes() == before
    assert destination.parent.is_symlink() if is_parent else destination.is_symlink()


@pytest.mark.parametrize("boundary", ["capture", "copied", "source_unlink"])
def test_interrupted_move_preserves_source_or_durable_destination_and_membership(tmp_path, monkeypatch, boundary):
    from qqtools.plugins.qexp.runtime import responsibility_import

    cfg, runtime, binding, target, source, value = migration_fixture(tmp_path)
    destination = local_paths(target)["registrations"] / source.name
    original_replace = responsibility_import.atomic_replace
    original_sync = DurableIO.sync_directory

    def fail_capture(*_args, **_kwargs):
        raise OSError("capture interrupted")

    def fail_copy(path, content):
        original_replace(path, content)
        if path == destination:
            raise OSError("copy interrupted")

    def fail_source_sync(io, path, label):
        if path == source.parent:
            raise OSError("source unlink interrupted")
        return original_sync(io, path, label)

    with monkeypatch.context() as patch:
        if boundary == "capture":
            patch.setattr(Ledger, "capture_source", fail_capture)
        elif boundary == "copied":
            patch.setattr(responsibility_import, "atomic_replace", fail_copy)
        else:
            patch.setattr(DurableIO, "sync_directory", fail_source_sync)
        with pytest.raises(OSError, match="interrupted"):
            runtime.import_legacy_evidence(binding)

    ledger = Ledger(responsibility_root(target))
    if boundary == "capture":
        assert source.exists() and not destination.exists()
        assert ledger.find("task-attempt-1") is None
    else:
        assert read_json(destination) == value
        entry = ledger.lookup("task-attempt-1")
        assert entry["legacy_source"] == str(cfg.runtime_root)
        assert entry["payload"] == {"task_id": "task", "attempt_number": 1}
        assert source.exists() == (boundary == "copied")

    restarted = MachineRuntime(runtime.root)
    restarted.import_legacy_evidence(binding)
    assert not source.exists()
    assert read_json(destination) == value
    generation = ledger.lookup("task-attempt-1")["generation"]
    late = local_paths(cfg.runtime_root)["observations"] / source.name
    observation = {"exit_observation": {"protocol_version": 1, "attempt_id": "task-attempt-1", "observed_exit_code": 0}}
    atomic_replace(late, observation)
    restarted.drain_legacy_runner_evidence(binding)
    assert not late.exists()
    assert read_json(local_paths(target)["observations"] / source.name) == observation
    assert ledger.lookup("task-attempt-1")["generation"] == generation


def test_conflicting_late_evidence_retains_both_copies_and_source_coverage(tmp_path):
    cfg, runtime, binding, target, source, value = migration_fixture(tmp_path)
    runtime.import_legacy_evidence(binding)
    conflicting = {"process_registration": {**value["process_registration"], "wrapper_pid": 99999992}}
    atomic_replace(source, conflicting)

    with pytest.raises(Conflict, match="conflicts during migration"):
        runtime.drain_legacy_runner_evidence(binding)

    assert read_json(source) == conflicting
    assert read_json(local_paths(target)["registrations"] / source.name) == value
    assert Ledger(responsibility_root(target)).lookup("task-attempt-1")["legacy_source"] == str(cfg.runtime_root)


@pytest.mark.parametrize("is_canonical", [False, True])
@pytest.mark.parametrize(
    "field, conflicting", [("attempt_id", "another-attempt"), ("task_id", "another-task"), ("attempt_number", 2)]
)
def test_initial_import_preserves_both_copies_when_destination_identity_conflicts(
    tmp_path, field, conflicting, is_canonical
):
    source_root, target = tmp_path / "old", tmp_path / "target"
    identity = "task-attempt-1" if is_canonical else "custom-attempt"
    source = local_paths(source_root)["observations"] / f"{identity}.json"
    destination = local_paths(target)["observations"] / source.name
    value = {
        "exit_observation": {"attempt_id": identity, "task_id": "task", "attempt_number": 1, "observed_exit_code": 0}
    }
    other = {"exit_observation": {**value["exit_observation"], field: conflicting}}
    atomic_replace(source, value)
    atomic_replace(destination, other)

    with pytest.raises(Conflict):
        move_legacy_record(target, source_root, "observations", source, destination, is_destination_authoritative=True)

    assert read_json(source) == value
    assert read_json(destination) == other
    # Do not publish a cleanup owner for a path known to contain conflicting truth.
    assert not responsibility_root(target).exists()


def test_initial_import_keeps_same_identity_destination_precedence(tmp_path):
    cfg, runtime, binding, target, source, value = migration_fixture(tmp_path)
    destination = local_paths(target)["registrations"] / source.name
    preferred = {"process_registration": {**value["process_registration"], "wrapper_pid": 99999992}}
    atomic_replace(destination, preferred)

    runtime.import_legacy_evidence(binding)

    assert not source.exists()
    assert read_json(destination) == preferred
    assert Ledger(responsibility_root(target)).lookup("task-attempt-1")["legacy_source"] == str(cfg.runtime_root)


@pytest.mark.parametrize("is_canonical", [False, True])
@pytest.mark.parametrize("side", ["source", "destination"])
@pytest.mark.parametrize("envelope", [None, "process_registration"])
def test_initial_import_rejects_missing_or_wrong_record_envelope(tmp_path, is_canonical, side, envelope):
    source_root, target = tmp_path / "old", tmp_path / "target"
    identity = "task-attempt-1" if is_canonical else "historical-attempt"
    source = local_paths(source_root)["observations"] / f"{identity}.json"
    destination = local_paths(target)["observations"] / source.name
    record = {"attempt_id": identity, "observed_exit_code": 0}
    valid = {"exit_observation": record}
    invalid = {} if envelope is None else {envelope: record}
    source_value = invalid if side == "source" else valid
    destination_value = invalid if side == "destination" else valid
    atomic_replace(source, source_value)
    atomic_replace(destination, destination_value)

    with pytest.raises((ValueError, Conflict), match="identity does not match"):
        move_legacy_record(target, source_root, "observations", source, destination, is_destination_authoritative=True)

    assert read_json(source) == source_value
    assert read_json(destination) == destination_value
    assert not responsibility_root(target).exists()


def test_matching_destination_supplies_missing_noncanonical_locator(tmp_path):
    source_root, target = tmp_path / "old", tmp_path / "target"
    identity = "historical-attempt"
    source = local_paths(source_root)["observations"] / f"{identity}.json"
    destination = local_paths(target)["observations"] / source.name
    atomic_replace(source, {"exit_observation": {"attempt_id": identity, "observed_exit_code": 0}})
    preferred = {
        "exit_observation": {"attempt_id": identity, "task_id": "task", "attempt_number": 3, "observed_exit_code": 0}
    }
    atomic_replace(destination, preferred)

    move_legacy_record(target, source_root, "observations", source, destination, is_destination_authoritative=True)

    assert not source.exists()
    assert read_json(destination) == preferred
    assert Ledger(responsibility_root(target)).lookup(identity)["payload"] == {"task_id": "task", "attempt_number": 3}


@pytest.mark.parametrize("ancestor_offset", [0, 1, 2])
def test_destination_ancestor_sync_failure_retains_source_and_retry_finishes(tmp_path, monkeypatch, ancestor_offset):
    cfg, runtime, binding, target, source, value = migration_fixture(tmp_path)
    destination = local_paths(target)["registrations"] / source.name
    failed_directory = target if ancestor_offset == 0 else target.parents[ancestor_offset - 1]
    original_sync = DurableIO.sync_directory

    def fail_ancestor(io, path, label):
        if path == failed_directory and label == "legacy_destination_ancestor":
            raise OSError("ancestor barrier interrupted")
        return original_sync(io, path, label)

    with monkeypatch.context() as patch:
        patch.setattr(DurableIO, "sync_directory", fail_ancestor)
        with pytest.raises(OSError, match="ancestor barrier interrupted"):
            runtime.import_legacy_evidence(binding)
    assert read_json(source) == read_json(destination) == value
    assert Ledger(responsibility_root(target)).lookup("task-attempt-1")["legacy_source"] == str(cfg.runtime_root)

    # Retry must sync ancestors even though every directory and target now exists.
    synced = []

    def record_sync(io, path, label):
        if label.startswith("legacy_destination"):
            assert source.exists()
            synced.append(path)
        return original_sync(io, path, label)

    monkeypatch.setattr(DurableIO, "sync_directory", record_sync)
    runtime.import_legacy_evidence(binding)
    assert synced == list(destination.resolve().parents)
    assert not source.exists()
    assert read_json(destination) == value


@pytest.mark.parametrize("lane", ["cpu_active", "cpu_provisional"])
def test_cpu_reservation_only_binding_cannot_be_removed(tmp_path, lane):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "local")
    runtime = MachineRuntime(tmp_path / "machine")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    target = runtime.project_paths(binding.project_id)["root"]
    target.mkdir(parents=True)
    reservation = runtime.paths[lane] / "cpu-pending.json"
    atomic_replace(reservation, {"reservation": {"reservation_id": "cpu-pending", "project_id": binding.project_id}})
    unrelated = runtime.paths[lane] / "cpu-other.json"
    atomic_replace(unrelated, {"reservation": {"reservation_id": "cpu-other", "project_id": "another-project"}})
    disabled = runtime.set_enabled(binding.project_id, False)

    assert runtime.binding_blockers(disabled) == ["reservation:cpu-pending"]
    with pytest.raises(RuntimeError, match="reservation:cpu-pending"):
        runtime.remove_binding(disabled.project_id)
    assert runtime.binding_state(disabled) == "draining"
    assert target.exists() and reservation.exists() and unrelated.exists()

    reservation.unlink()
    runtime.remove_binding(disabled.project_id)
    assert unrelated.exists()


def test_cleanup_covers_late_source_evidence_before_retiring_target(tmp_path):
    cfg, runtime, binding, target, source, value = migration_fixture(tmp_path)
    runtime.import_legacy_evidence(binding)
    # A late source write can exist before the next periodic import pass.
    late = local_paths(cfg.runtime_root)["observations"] / source.name
    atomic_replace(
        late, {"exit_observation": {"protocol_version": 1, "attempt_id": "task-attempt-1", "observed_exit_code": 0}}
    )
    ledger = Ledger(responsibility_root(target))
    request = CleanupRequest(
        "task-attempt-1",
        {"task_id": "task", "attempt_number": 1},
        {"format": CLEANUP_FORMAT, "task_id": "task", "attempt_id": "task-attempt-1", "basis": "terminal_attempt"},
    )

    assert not complete_cleanup(ledger, target, request)
    assert not late.exists()
    assert (local_paths(target)["registrations"] / source.name).exists()
    entry = ledger.lookup(request.identity)
    assert entry["cleanup_receipt"] == request.receipt
    assert complete_cleanup(ledger, target, CleanupRequest.from_entry(entry))
    assert not (local_paths(target)["registrations"] / source.name).exists()
    assert ledger.find(request.identity) is None


def test_source_cleanup_barrier_failure_keeps_receipt_and_destination(tmp_path, monkeypatch):
    cfg, runtime, binding, target, source, value = migration_fixture(tmp_path)
    runtime.import_legacy_evidence(binding)
    atomic_replace(source, value)
    ledger = Ledger(responsibility_root(target))
    request = CleanupRequest(
        "task-attempt-1",
        {"task_id": "task", "attempt_number": 1},
        {"format": CLEANUP_FORMAT, "task_id": "task", "attempt_id": "task-attempt-1", "basis": "terminal_attempt"},
    )
    original_sync = ledger.io.sync_directory

    def fail_sync(path, label):
        if path == source.parent:
            raise OSError("source cleanup barrier interrupted")
        return original_sync(path, label)

    with monkeypatch.context() as patch:
        patch.setattr(ledger.io, "sync_directory", fail_sync)
        with pytest.raises(OSError, match="source cleanup barrier interrupted"):
            complete_cleanup(ledger, target, request)
    assert not source.exists()
    assert (local_paths(target)["registrations"] / source.name).exists()
    entry = ledger.lookup(request.identity)
    assert entry["cleanup_receipt"] == request.receipt
    assert complete_cleanup(ledger, target, CleanupRequest.from_entry(entry))
    assert ledger.find(request.identity) is None


def test_import_does_not_recreate_destination_after_cleanup_handoff(tmp_path):
    cfg, runtime, binding, target, source, value = migration_fixture(tmp_path)
    ledger = Ledger.open_or_create(responsibility_root(target))
    identity = "task-attempt-1"
    generation = ledger.capture_source(identity, {"task_id": "task", "attempt_number": 1}, cfg.runtime_root)
    receipt = {"format": CLEANUP_FORMAT, "task_id": "task", "attempt_id": identity, "basis": "terminal_attempt"}
    generation = ledger.handoff(identity, generation, cleanup_receipt=receipt)

    runtime.import_legacy_evidence(binding)

    assert read_json(source) == value
    assert not (local_paths(target)["registrations"] / source.name).exists()
    assert ledger.lookup(identity)["generation"] == generation
    assert ledger.lookup(identity)["cleanup_receipt"] == receipt


def test_source_capture_can_fill_unknown_locator_but_cannot_change_known_identity(tmp_path):
    ledger = Ledger.open_or_create(tmp_path / "members")
    generation = ledger.capture_source("custom", {"task_id": None, "attempt_number": None}, tmp_path / "old")
    resolved = ledger.resolve_locator("custom", {"task_id": "task", "attempt_number": 2})
    assert resolved > generation
    assert ledger.lookup("custom")["legacy_source"] == str(tmp_path / "old")
    assert ledger.capture_source("custom", {"task_id": None, "attempt_number": None}, tmp_path / "old") == resolved
    with pytest.raises(Conflict, match="different identity"):
        ledger.resolve_locator("custom", {"task_id": "another", "attempt_number": 2})
    with pytest.raises(Conflict, match="different legacy"):
        ledger.capture_source("custom", {"task_id": "task", "attempt_number": 2}, tmp_path / "another")
    assert ledger.lookup("custom")["generation"] == resolved


@pytest.mark.parametrize("is_corrupt", [False, True])
def test_binding_removal_cannot_discard_receipt_only_or_unavailable_membership(tmp_path, is_corrupt):
    cfg, runtime, binding, target, source, value = migration_fixture(tmp_path)
    # No local evidence is copied: membership itself is the remaining owner.
    ledger = Ledger.open_or_create(responsibility_root(target))
    generation = ledger.publish("pending", {"task_id": "task", "attempt_number": 1})
    ledger.handoff("pending", generation)
    if is_corrupt:
        (ledger.root / "marker").write_text("broken")
    disabled = runtime.set_enabled(binding.project_id, False)

    with pytest.raises(RuntimeError, match="recovery_responsibilities"):
        runtime.remove_binding(disabled.project_id)

    assert target.exists()
    assert runtime.binding_state(disabled) == "draining"
    assert source.exists()


def test_import_rejects_same_source_and_destination_without_deleting_evidence(tmp_path):
    root = tmp_path / "runtime"
    path = local_paths(root)["observations"] / "task-attempt-1.json"
    atomic_replace(path, {"exit_observation": {"attempt_id": "task-attempt-1"}})
    with pytest.raises(ValueError, match="cannot be the destination"):
        move_legacy_record(root, root, "observations", path, path, is_destination_authoritative=True)
    assert path.exists()


def crash_during_move(target, source_root, source, destination, boundary):
    import os

    from qqtools.plugins.qexp.runtime import responsibility_import

    original_capture = Ledger.capture_source
    original_copy = responsibility_import.atomic_replace
    original_delete = DurableIO.delete

    def capture(ledger, *args, **kwargs):
        result = original_capture(ledger, *args, **kwargs)
        if boundary == "captured":
            os._exit(86)
        return result

    def copy(path, value):
        original_copy(path, value)
        if path == destination and boundary == "copied":
            os._exit(86)

    def delete(io, path, **kwargs):
        original_delete(io, path, **kwargs)
        if path == source and boundary == "source_deleted":
            os._exit(86)

    Ledger.capture_source = capture
    responsibility_import.atomic_replace = copy
    DurableIO.delete = delete
    move_legacy_record(target, source_root, "registrations", source, destination, is_destination_authoritative=True)


@pytest.mark.parametrize("boundary", ["captured", "copied", "source_deleted"])
def test_process_crash_during_legacy_move_keeps_a_discoverable_owner(tmp_path, boundary):
    import multiprocessing

    cfg, runtime, binding, target, source, value = migration_fixture(tmp_path)
    destination = local_paths(target)["registrations"] / source.name
    process = multiprocessing.get_context("fork").Process(
        target=crash_during_move,
        args=(target, cfg.runtime_root, source, destination, boundary),
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
    entry = Ledger(responsibility_root(target)).lookup("task-attempt-1")
    assert entry["legacy_source"] == str(cfg.runtime_root)
    assert source.exists() or destination.exists()
    runtime.import_legacy_evidence(binding)
    assert not source.exists()
    assert read_json(destination) == value


def test_cleanup_waits_for_a_writer_still_present_in_legacy_source(tmp_path):
    import os

    from qqtools.plugins.qexp.runner import _process_start_time_ticks

    cfg, runtime, binding, target, source, value = migration_fixture(tmp_path)
    runtime.import_legacy_evidence(binding)
    source_value = {
        "process_registration": {
            **value["process_registration"],
            "wrapper_pid": os.getpid(),
            "wrapper_start_time_ticks": _process_start_time_ticks(os.getpid()),
        }
    }
    atomic_replace(source, source_value)
    ledger = Ledger(responsibility_root(target))
    request = CleanupRequest(
        "task-attempt-1",
        {"task_id": "task", "attempt_number": 1},
        {"format": CLEANUP_FORMAT, "task_id": "task", "attempt_id": "task-attempt-1", "basis": "terminal_attempt"},
    )

    assert not complete_cleanup(ledger, target, request)
    assert read_json(source) == source_value
    assert (local_paths(target)["registrations"] / source.name).exists()
    assert ledger.lookup(request.identity)["stage"] == "active"
    assert "cleanup_receipt" not in ledger.lookup(request.identity)


def test_source_attachment_cannot_expand_an_existing_cleanup_receipt(tmp_path):
    ledger = Ledger.open_or_create(tmp_path / "members")
    payload = {"task_id": "task", "attempt_number": 1}
    generation = ledger.publish("attempt", payload)
    receipt = {"format": CLEANUP_FORMAT, "task_id": "task", "attempt_id": "attempt", "basis": "terminal_attempt"}
    generation = ledger.handoff("attempt", generation, cleanup_receipt=receipt)

    with pytest.raises(Conflict, match="cleanup already owns"):
        ledger.capture_source("attempt", payload, tmp_path / "old")

    assert ledger.lookup("attempt")["generation"] == generation
    assert "legacy_source" not in ledger.lookup("attempt")
