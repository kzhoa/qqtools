"""Bounded committed-Submission visibility and recoverable source proofs."""

import json
import os
from copy import deepcopy

import pytest

from qqtools.plugins.qexp import init_shared_root, scheduler, submit
from qqtools.plugins.qexp.runtime import dependencies
from qqtools.plugins.qexp.runtime import submission_control as control
from qqtools.plugins.qexp.runtime.availability.transitions import _submission_committed
from qqtools.plugins.qexp.runtime.paths import submission_path
from qqtools.plugins.qexp.runtime.ready import classify_ready_marker, routes
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json

pytestmark = pytest.mark.integration


@pytest.fixture
def submitted(tmp_path):
    cfg = init_shared_root(tmp_path / "project/.qexp", "worker", runtime_root=tmp_path / "runtime")
    task = submit(cfg, ["true"], task_id="survivor")
    path = submission_path(cfg.shared_root, task.submission_operation_id)
    return cfg, task, read_json(path)


def large_operation(operation):
    value = deepcopy(operation)
    # A retained specification can contain a long command or other user payload.
    value["submission"]["resolved_context"]["retained_payload"] = "x" * 180_000
    return value


def source_path(cfg, operation):
    return submission_path(cfg.shared_root, operation["submission"]["operation_id"])


def reference(cfg, task):
    result = routes.reference_for_generation(cfg, task.task_id, task.ready_generation)
    assert result is not None
    return result


def test_large_proof_consumers_do_not_open_bulk_source(submitted, monkeypatch):
    cfg, task, operation = submitted
    operation = large_operation(operation)
    control.publish_submission(cfg, operation)
    source = source_path(cfg, operation)
    original_open = type(source).open
    original_os_open = os.open

    def guarded_open(path, *args, **kwargs):
        assert path != source, "healthy committed-state read opened retained bulk payload"
        return original_open(path, *args, **kwargs)

    def guarded_os_open(path, *args, **kwargs):
        assert os.fspath(path) != os.fspath(source), "healthy reader opened retained bulk source"
        return original_os_open(path, *args, **kwargs)

    ready_ref = reference(cfg, task)
    with monkeypatch.context() as guard:
        guard.setattr(type(source), "open", guarded_open)
        guard.setattr(os, "open", guarded_os_open)
        assert control.read_submission_state(cfg, task.submission_operation_id) == "committed"
        assert classify_ready_marker(cfg, ready_ref).classification == "claimable"
        assert scheduler._eligible(cfg, task)
        assert dependencies.is_committed_submission_task(cfg, task)
        assert _submission_committed(cfg, task)
        attempt = scheduler.claim_task(cfg, task.task_id, [0])
        assert attempt is not None
        assert attempt.task_id == task.task_id


@pytest.mark.parametrize("state", ["preparing", "committing", "aborted", "blocked"])
def test_precommit_task_and_marker_do_not_authorize_dispatch(submitted, state):
    cfg, task, operation = submitted
    operation = large_operation(operation)
    operation["submission"]["state"] = state
    control.publish_submission(cfg, operation)
    assert control.read_submission_state(cfg, task.submission_operation_id) == state
    assert not scheduler._eligible(cfg, task)
    assert not dependencies.is_committed_submission_task(cfg, task)
    assert not _submission_committed(cfg, task)
    assert classify_ready_marker(cfg, reference(cfg, task)).classification != "claimable"


@pytest.mark.parametrize("state", ["committing", "aborted"])
def test_external_replacement_invalidates_old_committed_receipt(submitted, state):
    cfg, task, operation = submitted
    operation = large_operation(operation)
    control.publish_submission(cfg, operation)
    operation["submission"]["state"] = state
    atomic_replace(source_path(cfg, operation), operation)
    with pytest.raises(control.SubmissionControlUnavailable):
        control.read_submission_state(cfg, task.submission_operation_id)
    assert not scheduler._eligible(cfg, task)
    assert control.pending_path(cfg, task.submission_operation_id).exists()


def test_small_unindexed_source_has_bounded_truth_fallback(submitted):
    cfg, task, operation = submitted
    operation["submission"]["state"] = "aborted"
    atomic_replace(source_path(cfg, operation), operation)
    assert control.read_submission_state(cfg, task.submission_operation_id) == "aborted"
    assert not scheduler._eligible(cfg, task)


def test_deleted_source_cannot_be_resurrected_by_receipt(submitted):
    cfg, task, operation = submitted
    control.publish_submission(cfg, large_operation(operation))
    source_path(cfg, operation).unlink()
    with pytest.raises(FileNotFoundError):
        control.read_submission_state(cfg, task.submission_operation_id)
    assert not scheduler._eligible(cfg, task)


def test_truth_commit_survives_proof_write_failure_and_retains_repair(submitted, monkeypatch):
    cfg, task, operation = submitted
    operation = large_operation(operation)

    def unavailable(*args, **kwargs):
        raise OSError("receipt storage unavailable")

    with monkeypatch.context() as guard:
        guard.setattr(control, "write_receipt", unavailable)
        control.publish_submission(cfg, operation)
    assert read_json(source_path(cfg, operation))["submission"]["state"] == "committed"
    assert control.pending_path(cfg, task.submission_operation_id).exists()
    with pytest.raises(control.SubmissionControlUnavailable):
        control.read_submission_state(cfg, task.submission_operation_id)


def test_pending_failure_prevents_truth_mutation(submitted, monkeypatch):
    cfg, task, operation = submitted
    path = source_path(cfg, operation)
    before = path.read_bytes()
    operation = large_operation(operation)
    operation["submission"]["state"] = "aborted"
    real_replace = control.atomic_replace

    def fail_intent(target, value, **kwargs):
        if target == control.pending_path(cfg, task.submission_operation_id):
            raise OSError("intent fsync failed")
        return real_replace(target, value, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(control, "atomic_replace", fail_intent)
        with pytest.raises(OSError, match="intent fsync"):
            control.publish_submission(cfg, operation)
    assert path.read_bytes() == before


@pytest.mark.parametrize("damage", ["missing", "malformed", "foreign", "oversized", "symlink"])
def test_invalid_large_source_proof_never_enables_legacy_scan(submitted, damage, tmp_path):
    cfg, task, operation = submitted
    control.publish_submission(cfg, large_operation(operation))
    proof = control.record_path(cfg, task.submission_operation_id)
    if damage == "missing":
        proof.unlink()
    elif damage == "malformed":
        proof.write_text("{")
    elif damage == "foreign":
        value = read_json(proof)
        value["operation_id"] = "foreign-operation"
        atomic_replace(proof, value)
    elif damage == "oversized":
        proof.write_text(" " * 9000)
    else:
        target = tmp_path / "foreign-proof.json"
        target.write_bytes(proof.read_bytes())
        proof.unlink()
        proof.symlink_to(target)
    with pytest.raises(control.SubmissionControlUnavailable):
        control.read_submission_state(cfg, task.submission_operation_id)
    assert not scheduler._eligible(cfg, task)


def test_malformed_activation_state_never_selects_legacy_full_read(submitted):
    cfg, task, operation = submitted
    operation = large_operation(operation)
    atomic_replace(source_path(cfg, operation), operation)
    (cfg.shared_root / "indexes/submission-control/state.json").write_text(json.dumps({"state": "building"}))
    with pytest.raises(control.SubmissionControlUnavailable):
        control.read_submission_state(cfg, task.submission_operation_id)


def test_small_submission_publication_keeps_single_truth_write(submitted, monkeypatch):
    cfg, _task, operation = submitted
    writes = []
    real_replace = control.atomic_replace

    def record_write(path, value):
        writes.append(path)
        return real_replace(path, value)

    with monkeypatch.context() as guard:
        guard.setattr(control, "atomic_replace", record_write)
        control.publish_submission(cfg, operation)
    assert writes == [source_path(cfg, operation)]


def test_shared_source_revision_does_not_bind_client_mount_device(submitted):
    from types import SimpleNamespace

    cfg, _task, operation = submitted
    original = source_path(cfg, operation).stat()
    other_mount = SimpleNamespace(
        st_dev=original.st_dev + 1,
        st_ino=original.st_ino,
        st_size=original.st_size,
        st_mtime_ns=original.st_mtime_ns,
        st_ctime_ns=original.st_ctime_ns,
    )
    assert control.source_revision(original) == control.source_revision(other_mount)
    other_mount.st_ino += 1
    assert control.source_revision(original) != control.source_revision(other_mount)


@pytest.mark.parametrize("damage", ["malformed", "missing", "foreign"])
def test_small_source_remains_claimable_with_unavailable_derived_state(submitted, damage):
    cfg, task, operation = submitted
    assert source_path(cfg, operation).stat().st_size < control.SUBMISSION_SOURCE_LIMIT
    state_path = control.control_paths(cfg)["state"]
    if damage == "missing":
        state_path.unlink()
    elif damage == "malformed":
        state_path.write_text('{"version":true}')
    else:
        state = read_json(state_path)
        state["root"] = str(cfg.shared_root.parent / "foreign")
        atomic_replace(state_path, state)
    assert control.read_submission_state(cfg, task.submission_operation_id) == "committed"
    assert classify_ready_marker(cfg, reference(cfg, task)).classification == "claimable"
    assert scheduler.claim_task(cfg, task.task_id, [0]) is not None
