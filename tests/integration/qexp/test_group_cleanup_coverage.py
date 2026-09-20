"""Characterize Group cancellation around terminal Task cleanup."""

from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.commands import cleanup as cleanup_commands
from qqtools.plugins.qexp.commands import task as task_commands
from qqtools.plugins.qexp.commands.group import create_group, group_control, reconcile_group_cancel_operations
from qqtools.plugins.qexp.runtime.operation_store import (
    active_operation_path,
    archived_operation_path,
    locate_operation_path,
)
from qqtools.plugins.qexp.runtime.paths import group_path, submission_path, task_path
from qqtools.plugins.qexp.runtime.store import read_json

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


@pytest.mark.parametrize("cleanup_timing", ["before_cancel", "during_cancel"])
def test_group_cancel_preserves_cleanup_membership_boundary(tmp_path: Path, monkeypatch, cleanup_timing: str):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    create_group(cfg, "exp")

    old = submit(cfg, ["echo", "old"], group="exp")
    old_submission_id = old.submission_operation_id
    assert old_submission_id is not None
    assert old.group_membership_sequence == 1
    task_commands.cancel(cfg, old.task_id, terminate_running=False)
    keep = submit(cfg, ["echo", "keep"], group="exp")
    assert keep.group_membership_sequence == 2

    with monkeypatch.context() as patch:
        patch.setattr(cleanup_commands, "_finalize_cleanup_if_ready", lambda *_args: [])
        cleanup_commands.clean(cfg, task_id=old.task_id, reservation_runtime_root=cfg.runtime_root)

    old_path = task_path(cfg.shared_root, old.task_id)
    cleanup_path = locate_operation_path(cfg, "cleanup", old.task_id)
    cleanup = read_json(cleanup_path)["cleanup"]
    assert old_path.exists()
    assert cleanup["state"] == "waiting_ack"
    assert cleanup["required_machines"] == ["g1"]
    assert cleanup["acknowledgements"]["g1"]["acknowledged_at"]
    assert isinstance(cleanup["acknowledgements"]["g1"]["removed"], list)
    assert cleanup["pending_machines"] == []

    if cleanup_timing == "before_cancel":
        cleanup_commands.reconcile_cleanup_operations(cfg, reservation_runtime_root=cfg.runtime_root)
        assert not old_path.exists()
        group = group_control(cfg, "exp", "cancel")
        operation_id = group["cancellation_operation"]["operation_id"]
    else:
        from qqtools.plugins.qexp.commands import group as group_commands

        original_write = group_commands.write_active_operation

        def fail_before_first_task(cfg_, kind, operation_id_, value):
            result = original_write(cfg_, kind, operation_id_, value)
            if kind == "group_control" and value["group_control"]["state"] == "converging":
                raise OSError("interrupted before first Group member")
            return result

        with monkeypatch.context() as patch:
            patch.setattr(group_commands, "write_active_operation", fail_before_first_task)
            with pytest.raises(OSError, match="interrupted before first Group member"):
                group_control(cfg, "exp", "cancel")

        group_snapshot = read_json(group_path(cfg.shared_root, "exp"))
        cancellation_snapshot = group_snapshot["cancellation_operation"]
        operation_id = cancellation_snapshot["operation_id"]
        assert cancellation_snapshot["membership_high_watermark"] == 2
        assert any(
            barrier["operation_id"] == operation_id and barrier["membership_high_watermark"] == 2
            for barrier in group_snapshot["group"]["cancellation_barriers"]
        )
        control_path = locate_operation_path(cfg, "group_control", operation_id)
        control = read_json(control_path)["group_control"]
        assert control["state"] == "converging"
        assert control["membership_high_watermark"] == 2
        assert old_path.exists()

        cleanup_commands.reconcile_cleanup_operations(cfg, reservation_runtime_root=cfg.runtime_root)
        assert not old_path.exists()
        outstanding = read_json(locate_operation_path(cfg, "group_control", operation_id))["group_control"]
        assert outstanding["state"] == "converging"
        reconcile_group_cancel_operations(cfg)

    cleanup_archive = archived_operation_path(cfg, "cleanup", old.task_id)
    assert cleanup_archive.is_file()
    assert not cleanup_archive.is_symlink()
    assert not active_operation_path(cfg, "cleanup", old.task_id).exists()
    cleanup_record = read_json(cleanup_archive)
    cleanup = cleanup_record["cleanup"]
    assert cleanup["state"] == "completed"
    assert cleanup["task_id"] == old.task_id
    assert cleanup["group_name"] == "exp"
    assert cleanup["submission_operation_id"] == old_submission_id
    assert cleanup["terminal_state"] == "cancelled"
    assert not old_path.exists()

    submission = read_json(submission_path(cfg.shared_root, old_submission_id))["submission"]
    assert submission["state"] == "committed"
    assert submission["target_group"] == "exp"
    assert submission["resolved_context"]["task_ids"] == [old.task_id]
    assert submission["commit_plan"]["group_membership_sequences"] == [1]

    keep_record = read_json(task_path(cfg.shared_root, keep.task_id))["task"]
    assert keep_record["state"]["projection"] == "cancelled"

    control_record = read_json(locate_operation_path(cfg, "group_control", operation_id))
    control = control_record["group_control"]
    assert control["state"] == "completed"
    assert control["membership_high_watermark"] == 2
    assert control["progress"]["target_tasks"] == 1
    assert control["pending_machine_acknowledgements"] == {}

    assert cleanup_commands.reconcile_cleanup_operations(cfg, reservation_runtime_root=cfg.runtime_root) == []
    assert reconcile_group_cancel_operations(cfg) == []
    assert read_json(locate_operation_path(cfg, "cleanup", old.task_id)) == cleanup_record
    assert read_json(locate_operation_path(cfg, "group_control", operation_id)) == control_record
    assert not old_path.exists()
    assert read_json(task_path(cfg.shared_root, keep.task_id))["task"]["state"]["projection"] == "cancelled"
