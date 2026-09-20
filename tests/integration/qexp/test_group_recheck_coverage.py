"""Characterize Group cancellation rechecking after an interrupted visit."""

from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.commands import group as group_commands
from qqtools.plugins.qexp.commands.group import create_group, group_control, reconcile_group_cancel_operations
from qqtools.plugins.qexp.commands.task import retry
from qqtools.plugins.qexp.runtime.operation_store import active_operation_path, archived_operation_path
from qqtools.plugins.qexp.runtime.paths import attempt_path, group_path
from qqtools.plugins.qexp.runtime.records import AttemptRecord
from qqtools.plugins.qexp.runtime.resources.reservations import reserved_gpu_ids
from qqtools.plugins.qexp.runtime.store import read_json
from qqtools.plugins.qexp.runtime.tasks import load_task
from qqtools.plugins.qexp.scheduler import authorize_launch, claim_task, fail_attempt

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


@pytest.mark.parametrize("terminate_running", [False, True])
def test_group_cancel_rechecks_retried_failed_task_after_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminate_running: bool
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    create_group(cfg, "exp")
    original = submit(cfg, ["true"], group="exp")
    assert original.group_membership_sequence == 1
    original_identity = (original.task_id, original.submission_operation_id)

    attempt = claim_task(cfg, original.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, original.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert fail_attempt(cfg, original.task_id, attempt.attempt_id, attempt.current_fencing_token, "test_failure")
    original_attempt_path = attempt_path(cfg.shared_root, original.task_id, attempt.attempt_number)
    original_attempt = AttemptRecord.from_dict(read_json(original_attempt_path))
    assert original_attempt.phase == "failed"

    visited: list[tuple[str, int | None]] = []
    real_apply = group_commands._apply_group_cancel_locked

    def interrupt_after_terminal_visit(cfg_arg, task_arg, control_arg):
        progress_key, terminal = real_apply(cfg_arg, task_arg, control_arg)
        assert task_arg.task_id == original.task_id
        assert task_arg.group_membership_sequence == original.group_membership_sequence
        assert progress_key == "already_terminal"
        assert terminal is None
        visited.append((task_arg.task_id, task_arg.group_membership_sequence))
        raise OSError("after terminal visit")

    with monkeypatch.context() as patch:
        patch.setattr(group_commands, "_apply_group_cancel_locked", interrupt_after_terminal_visit)
        with pytest.raises(OSError, match="after terminal visit"):
            group_control(cfg, "exp", "cancel", terminate_running=terminate_running)

    assert visited == [(original.task_id, 1)]
    operation_id = read_json(group_path(cfg.shared_root, "exp"))["cancellation_operation"]["operation_id"]
    active_path = active_operation_path(cfg, "group_control", operation_id)
    interrupted = read_json(active_path)["group_control"]
    assert interrupted["state"] == "converging"
    assert interrupted["membership_high_watermark"] == 1

    retried = retry(cfg, original.task_id)
    assert retried.state["projection"] == "queued"
    assert (retried.task_id, retried.submission_operation_id) == original_identity
    assert retried.group_membership_sequence == 1
    assert retried.attempt_control["current_attempt_id"] is None
    assert AttemptRecord.from_dict(read_json(original_attempt_path)) == original_attempt

    later = submit(cfg, ["true"], group="exp")
    assert later.group_membership_sequence == 2
    later_before_reconcile = load_task(cfg, later.task_id).to_dict()

    assert len(reconcile_group_cancel_operations(cfg)) == 1
    operation_archive = archived_operation_path(cfg, "group_control", operation_id)
    completed_record = read_json(operation_archive)
    completed = completed_record["group_control"]
    cancelled = load_task(cfg, original.task_id)
    assert cancelled.state["projection"] == "cancelled"
    assert cancelled.control["cancellation_operation_id"] == operation_id
    assert cancelled.control["terminate_running"] is terminate_running
    assert load_task(cfg, later.task_id).state["projection"] == "queued"
    assert load_task(cfg, later.task_id).to_dict() == later_before_reconcile
    assert not active_path.exists()
    assert completed["state"] == "completed"
    assert completed["membership_high_watermark"] == 1
    assert completed["progress"]["target_tasks"] == 1
    assert completed["pending_machine_acknowledgements"] == {}
    assert cancelled.claim_control["active_claim"] is None
    assert reserved_gpu_ids(cfg.runtime_root) == set()
    assert sorted(original_attempt_path.parent.glob("*.json")) == [original_attempt_path]
    assert AttemptRecord.from_dict(read_json(original_attempt_path)) == original_attempt

    cancelled_before_repeat = cancelled.to_dict()
    assert reconcile_group_cancel_operations(cfg) == []
    assert load_task(cfg, original.task_id).to_dict() == cancelled_before_repeat
    assert load_task(cfg, later.task_id).to_dict() == later_before_reconcile
    assert read_json(operation_archive) == completed_record
    assert AttemptRecord.from_dict(read_json(original_attempt_path)) == original_attempt
