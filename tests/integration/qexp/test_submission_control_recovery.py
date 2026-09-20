"""Automatic Submission proof capture, crash recovery, and source replacement."""

from copy import deepcopy

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.runtime import submission_control as control
from qqtools.plugins.qexp.runtime.paths import submission_path
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.submission_control_maintenance import SubmissionControlMaintenance

pytestmark = pytest.mark.integration


def source_path(cfg, operation):
    return submission_path(cfg.shared_root, operation["submission"]["operation_id"])


def setup_source(tmp_path):
    cfg = init_shared_root(tmp_path / "project/.qexp", "worker", runtime_root=tmp_path / "runtime")
    task = submit(cfg, ["true"], task_id="survivor")
    operation = read_json(source_path(cfg, {"submission": {"operation_id": task.submission_operation_id}}))
    operation["submission"]["resolved_context"]["retained_payload"] = "x" * 180_000
    atomic_replace(source_path(cfg, operation), operation)
    control.request_repair(cfg, task.submission_operation_id)
    return cfg, task, operation


def finish(cfg, operation_id, *, restart=False):
    worker = SubmissionControlMaintenance(cfg)
    try:
        for _ in range(1000):
            worker.advance()
            if restart:
                worker.close()
                worker = SubmissionControlMaintenance(cfg)
            try:
                return control.read_submission_state(cfg, operation_id)
            except control.SubmissionControlUnavailable:
                continue
        pytest.fail("bounded Submission proof recovery did not converge")
    finally:
        worker.close()


@pytest.mark.parametrize("restart", [False, True])
def test_source_backfill_converges_across_every_step_restart(tmp_path, restart):
    cfg, task, operation = setup_source(tmp_path)
    assert finish(cfg, task.submission_operation_id, restart=restart) == "committed"
    assert not control.pending_path(cfg, task.submission_operation_id).exists()
    assert not (
        cfg.shared_root / "indexes/submission-control/checkpoints" / f"{task.submission_operation_id}.json"
    ).exists()
    assert source_path(cfg, operation).exists()


def test_replaced_source_revokes_partial_parser_checkpoint(tmp_path):
    cfg, task, operation = setup_source(tmp_path)
    worker = SubmissionControlMaintenance(cfg)
    try:
        for _ in range(3):
            worker.advance()
    finally:
        worker.close()
    replacement = deepcopy(operation)
    replacement["submission"]["state"] = "aborted"
    atomic_replace(source_path(cfg, operation), replacement)
    assert finish(cfg, task.submission_operation_id, restart=True) == "aborted"


def test_repair_worker_does_not_full_read_large_json(tmp_path, monkeypatch):
    cfg, task, _operation = setup_source(tmp_path)
    import qqtools.plugins.qexp.runtime.store as store

    original = store.read_json

    def guarded(path):
        assert path.parent.name != "submissions", "background source build used unbounded json.load"
        return original(path)

    with monkeypatch.context() as guard:
        guard.setattr(store, "read_json", guarded)
        guard.setattr(control, "read_json", guarded)
        assert finish(cfg, task.submission_operation_id) == "committed"


def test_deleted_source_reclaims_only_derived_repair_files(tmp_path):
    cfg, task, operation = setup_source(tmp_path)
    source_path(cfg, operation).unlink()
    worker = SubmissionControlMaintenance(cfg)
    try:
        for _ in range(30):
            worker.advance()
            if not control.pending_path(cfg, task.submission_operation_id).exists():
                break
        assert not control.pending_path(cfg, task.submission_operation_id).exists()
        assert not control.record_path(cfg, task.submission_operation_id).exists()
    finally:
        worker.close()
    assert not source_path(cfg, operation).exists()


def test_old_root_automatically_builds_and_activates(tmp_path):
    cfg, task, operation = setup_source(tmp_path)
    state_path = cfg.shared_root / "indexes/submission-control/state.json"
    state_path.unlink()
    assert control.read_submission_state(cfg, task.submission_operation_id) == "committed"
    worker = SubmissionControlMaintenance(cfg)
    try:
        for _ in range(1000):
            worker.advance()
            state = control.read_control_state(cfg)
            if state and state["state"] == "active":
                break
        else:
            pytest.fail("old root Submission capture did not activate")
    finally:
        worker.close()
    assert control.read_submission_state(cfg, task.submission_operation_id) == "committed"
    assert control.record_path(cfg, task.submission_operation_id).exists()
    assert read_json(source_path(cfg, operation))["submission"]["state"] == "committed"


def test_process_exit_after_truth_before_receipt_recovers(tmp_path, checkout_subprocess_env):
    import subprocess
    import sys

    cfg, task, operation = setup_source(tmp_path)
    operation_file = tmp_path / "next-operation.json"
    atomic_replace(operation_file, operation)
    program = """
import os
from pathlib import Path
from qqtools.plugins.qexp.layout import load_root_config
from qqtools.plugins.qexp.runtime import submission_control as control
from qqtools.plugins.qexp.runtime.store import read_json
import sys
cfg = load_root_config(Path(sys.argv[1]), 'worker', runtime_root=Path(sys.argv[2]))
def crash(*args, **kwargs):
    os._exit(73)
control.write_receipt = crash
control.publish_submission(cfg, read_json(Path(sys.argv[3])))
raise AssertionError('crash boundary not reached')
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(cfg.shared_root), str(cfg.runtime_root), str(operation_file)],
        env=checkout_subprocess_env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 73, result.stderr
    assert read_json(source_path(cfg, operation))["submission"]["state"] == "committed"
    assert control.pending_path(cfg, task.submission_operation_id).exists()
    assert finish(cfg, task.submission_operation_id, restart=True) == "committed"


def test_committed_afterimage_recovers_without_parsing_bulk_source(tmp_path, monkeypatch):
    cfg, task, operation = setup_source(tmp_path)

    def fail_receipt(*args, **kwargs):
        raise OSError("receipt fsync failed")

    with monkeypatch.context() as guard:
        guard.setattr(control, "write_receipt", fail_receipt)
        control.publish_submission(cfg, operation)
    pending = read_json(control.pending_path(cfg, task.submission_operation_id))
    assert pending["state"] == "committed"
    assert len(pending["after_image"]) == 4

    from qqtools.plugins.qexp.runtime.group_discovery.source_revision import BoundSource

    def forbid_parse(*args, **kwargs):
        raise AssertionError("ordinary publication recovery reread bulk source")

    with monkeypatch.context() as guard:
        guard.setattr(BoundSource, "open", forbid_parse)
        assert finish(cfg, task.submission_operation_id, restart=True) == "committed"


def test_prepared_afterimage_cannot_certify_an_unrenamed_source(tmp_path, monkeypatch):
    cfg, task, operation = setup_source(tmp_path)
    original = deepcopy(operation)
    original["submission"]["state"] = "aborted"
    control.publish_submission(cfg, original)
    real_replace = __import__("os").replace

    def fail_source_rename(source, destination, *args, **kwargs):
        if destination == source_path(cfg, operation):
            raise OSError("source rename failed")
        return real_replace(source, destination, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr("os.replace", fail_source_rename)
        with pytest.raises(OSError, match="source rename failed"):
            control.publish_submission(cfg, operation)
    assert read_json(source_path(cfg, operation))["submission"]["state"] == "aborted"
    assert finish(cfg, task.submission_operation_id, restart=True) == "aborted"


@pytest.mark.parametrize("version", [True, 2])
def test_invalid_receipt_version_is_repaired_instead_of_retiring_debt(tmp_path, version):
    cfg, task, operation = setup_source(tmp_path)
    operation_id = task.submission_operation_id
    control.write_receipt(cfg, operation_id, "committed", control.source_revision(source_path(cfg, operation).stat()))
    receipt = read_json(control.record_path(cfg, operation_id))
    receipt["version"] = version
    atomic_replace(control.record_path(cfg, operation_id), receipt)
    assert finish(cfg, operation_id) == "committed"
    assert type(read_json(control.record_path(cfg, operation_id))["version"]) is int
    assert read_json(control.record_path(cfg, operation_id))["version"] == 1


def test_doctor_repairs_damaged_control_state_without_changing_truth(tmp_path):
    from qqtools.plugins.qexp.doctor import repair_metadata, verify_integrity

    cfg, task, operation = setup_source(tmp_path)
    source = source_path(cfg, operation)
    original = source.read_bytes()
    control.control_paths(cfg)["state"].write_text('{"version":true}')
    assert verify_integrity(cfg)["submission_control"]["state"] == "unavailable"
    result = repair_metadata(cfg)
    assert result["submission_control"]["state"] == "building"
    assert source.read_bytes() == original
    assert finish(cfg, task.submission_operation_id) == "committed"
