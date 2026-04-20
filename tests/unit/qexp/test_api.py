from __future__ import annotations

import json

import pytest
import yaml

from qqtools.plugins.qexp.api import (
    batch_retry_cancelled,
    batch_retry_failed,
    batch_submit,
    cancel,
    clean,
    get_log_path,
    resubmit,
    retry,
    submit,
)
from qqtools.plugins.qexp.doctor import repair_metadata
from qqtools.plugins.qexp.indexes import load_index
from qqtools.plugins.qexp.layout import init_shared_root, resubmit_operation_path
from qqtools.plugins.qexp.models import (
    BATCH_COMMIT_ABORTED,
    BATCH_COMMIT_COMMITTED,
    BATCH_COMMIT_PREPARING,
    PHASE_CANCELLED,
    PHASE_FAILED,
    PHASE_QUEUED,
    PHASE_RUNNING,
    PHASE_SUCCEEDED,
)
from qqtools.plugins.qexp.storage import (
    cas_update_task,
    load_batch,
    load_resubmit_operation,
    load_task,
    save_resubmit_operation,
)


@pytest.fixture()
def cfg(tmp_path):
    return init_shared_root(tmp_path / ".qexp", "dev1", runtime_root=tmp_path / "runtime")


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


class TestSubmit:
    def test_basic_submit(self, cfg):
        t = submit(cfg, command=["python", "train.py"])
        assert t.status.phase == PHASE_QUEUED
        assert t.machine_name == "dev1"
        assert t.spec.command == ["python", "train.py"]
        assert t.meta.revision == 1

    def test_explicit_task_id(self, cfg):
        t = submit(cfg, command=["echo"], task_id="my-task")
        assert t.task_id == "my-task"

    def test_persisted(self, cfg):
        t = submit(cfg, command=["echo"])
        loaded = load_task(cfg, t.task_id)
        assert loaded.task_id == t.task_id

    def test_indexed(self, cfg):
        t = submit(cfg, command=["echo"])
        assert t.task_id in load_index(cfg, "state", PHASE_QUEUED)
        assert t.task_id in load_index(cfg, "machine", "dev1")

    def test_custom_gpus(self, cfg):
        t = submit(cfg, command=["echo"], requested_gpus=4)
        assert t.spec.requested_gpus == 4

    def test_invalid_task_id(self, cfg):
        with pytest.raises(ValueError):
            submit(cfg, command=["echo"], task_id="../bad")

    def test_rejects_existing_task_id(self, cfg):
        submit(cfg, command=["echo"], task_id="dup1")
        with pytest.raises(ValueError, match="already exists"):
            submit(cfg, command=["echo"], task_id="dup1")

    def test_with_name(self, cfg):
        t = submit(cfg, command=["echo"], name="test run")
        assert t.name == "test run"

    def test_with_group(self, cfg):
        t = submit(cfg, command=["echo"], group="contract_n_4and6")
        assert t.group == "contract_n_4and6"
        assert t.task_id in load_index(cfg, "group", "contract_n_4and6")

    def test_rejects_reserved_group(self, cfg):
        with pytest.raises(ValueError, match="reserved"):
            submit(cfg, command=["echo"], group="experiments")

    def test_submit_rejects_uninitialized_root(self, tmp_path):
        from qqtools.plugins.qexp.layout import RootConfig, ensure_machine_layout, ensure_runtime_layout, ensure_shared_layout

        cfg = RootConfig(
            shared_root=tmp_path / ".qexp",
            project_root=tmp_path,
            machine_name="dev1",
            runtime_root=tmp_path / "runtime",
        )
        ensure_shared_layout(cfg)
        ensure_machine_layout(cfg)
        ensure_runtime_layout(cfg)

        with pytest.raises(FileNotFoundError, match="Root manifest not found"):
            submit(cfg, command=["echo"])


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


class TestCancel:
    def test_cancel_queued(self, cfg):
        t = submit(cfg, command=["echo"])
        cancelled = cancel(cfg, t.task_id)
        assert cancelled.status.phase == PHASE_CANCELLED
        assert cancelled.status.reason == "cancelled_by_user"
        assert cancelled.timestamps.finished_at is not None

    def test_cancel_non_cancellable(self, cfg):
        t = submit(cfg, command=["echo"])
        # Force to succeeded
        t.status.phase = PHASE_SUCCEEDED
        t.timestamps.finished_at = "2026-01-01T00:00:00Z"
        from qqtools.plugins.qexp.indexes import update_index_on_phase_change
        cas_update_task(cfg, t, t.meta.revision)
        update_index_on_phase_change(cfg, t.task_id, PHASE_QUEUED, PHASE_SUCCEEDED)

        with pytest.raises(ValueError, match="Cannot cancel"):
            cancel(cfg, t.task_id)

    def test_cancel_updates_index(self, cfg):
        t = submit(cfg, command=["echo"])
        cancel(cfg, t.task_id)
        assert t.task_id not in load_index(cfg, "state", PHASE_QUEUED)
        assert t.task_id in load_index(cfg, "state", PHASE_CANCELLED)


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------


class TestRetry:
    def _make_failed_task(self, cfg):
        t = submit(cfg, command=["python", "train.py"], name="exp1")
        from qqtools.plugins.qexp.indexes import update_index_on_phase_change
        t.status.phase = PHASE_FAILED
        t.result.exit_code = 1
        t.result.terminal_reason = "nonzero_exit"
        t.timestamps.finished_at = "2026-01-01T00:00:00Z"
        cas_update_task(cfg, t, t.meta.revision)
        update_index_on_phase_change(cfg, t.task_id, PHASE_QUEUED, PHASE_FAILED)
        return t

    def test_retry_creates_new_task(self, cfg):
        original = self._make_failed_task(cfg)
        new = retry(cfg, original.task_id)
        assert new.task_id != original.task_id
        assert new.lineage.retry_of == original.task_id
        assert new.attempt == 2
        assert new.status.phase == PHASE_QUEUED

    def test_retry_preserves_command(self, cfg):
        original = self._make_failed_task(cfg)
        new = retry(cfg, original.task_id)
        assert new.spec.command == original.spec.command

    def test_retry_preserves_name(self, cfg):
        original = self._make_failed_task(cfg)
        new = retry(cfg, original.task_id)
        assert new.name == "exp1"

    def test_retry_preserves_group_by_default(self, cfg):
        original = submit(cfg, command=["python", "train.py"], name="exp1", group="contract_n_4and6")
        from qqtools.plugins.qexp.indexes import update_index_on_phase_change
        original.status.phase = PHASE_FAILED
        original.result.exit_code = 1
        original.timestamps.finished_at = "2026-01-01T00:00:00Z"
        cas_update_task(cfg, original, original.meta.revision)
        update_index_on_phase_change(cfg, original.task_id, PHASE_QUEUED, PHASE_FAILED)

        new = retry(cfg, original.task_id)
        assert new.group == "contract_n_4and6"

    def test_retry_allows_group_override(self, cfg):
        original = submit(cfg, command=["python", "train.py"], group="contract_n_4and6")
        from qqtools.plugins.qexp.indexes import update_index_on_phase_change
        original.status.phase = PHASE_FAILED
        original.result.exit_code = 1
        original.timestamps.finished_at = "2026-01-01T00:00:00Z"
        cas_update_task(cfg, original, original.meta.revision)
        update_index_on_phase_change(cfg, original.task_id, PHASE_QUEUED, PHASE_FAILED)

        new = retry(cfg, original.task_id, group="regrouped_debug")
        assert new.group == "regrouped_debug"

    def test_retry_non_terminal_fails(self, cfg):
        t = submit(cfg, command=["echo"])
        with pytest.raises(ValueError, match="Only terminal"):
            retry(cfg, t.task_id)

    def test_retry_indexed(self, cfg):
        original = self._make_failed_task(cfg)
        new = retry(cfg, original.task_id)
        assert new.task_id in load_index(cfg, "state", PHASE_QUEUED)


class TestResubmit:
    def _make_terminal_task(self, cfg, phase=PHASE_FAILED, *, task_id="t1", batch_id=None):
        t = submit(cfg, command=["python", "train.py"], task_id=task_id, name="exp1", group="contract_n_4and6")
        from qqtools.plugins.qexp.indexes import update_index_on_phase_change

        t = load_task(cfg, t.task_id)
        t.batch_id = batch_id
        t.status.phase = phase
        t.status.reason = "terminal"
        t.result.exit_code = 1 if phase == PHASE_FAILED else None
        t.result.terminal_reason = "nonzero_exit" if phase == PHASE_FAILED else None
        t.runtime.wrapper_pid = 999 if phase == PHASE_FAILED else None
        t.timestamps.started_at = "2026-01-01T00:00:00Z"
        t.timestamps.finished_at = "2026-01-01T00:10:00Z"
        cas_update_task(cfg, t, t.meta.revision)
        update_index_on_phase_change(cfg, t.task_id, PHASE_QUEUED, phase)
        return load_task(cfg, t.task_id)

    def _save_resubmit_op(self, cfg, task, *, command):
        from qqtools.plugins.qexp.api import _build_resubmit_operation

        operation = _build_resubmit_operation(
            cfg,
            task,
            command=command,
            requested_gpus=task.spec.requested_gpus,
            name=task.name,
            group=task.group,
        )
        save_resubmit_operation(cfg, operation)
        return load_resubmit_operation(cfg, task.task_id)

    def test_resubmit_failed_task_replaces_truth(self, cfg):
        original = self._make_terminal_task(cfg, PHASE_FAILED)
        new = resubmit(cfg, original.task_id, command=["python", "train.py", "--fresh"])

        assert new.task_id == original.task_id
        assert new.status.phase == PHASE_QUEUED
        assert new.spec.command == ["python", "train.py", "--fresh"]
        assert new.lineage.retry_of is None
        assert new.result.exit_code is None
        assert new.result.terminal_reason is None
        assert new.runtime.wrapper_pid is None
        assert new.timestamps.finished_at is None
        assert new.group == "contract_n_4and6"
        assert new.task_id in load_index(cfg, "state", PHASE_QUEUED)
        assert new.task_id not in load_index(cfg, "state", PHASE_FAILED)
        assert not resubmit_operation_path(cfg, new.task_id).exists()

    def test_resubmit_cancelled_task_is_fresh_first_submit(self, cfg):
        original = self._make_terminal_task(cfg, PHASE_CANCELLED, task_id="cancelled1")
        new = resubmit(cfg, original.task_id, command=["echo", "rerun"])

        assert new.task_id == "cancelled1"
        assert new.attempt == 1
        assert new.lineage.retry_of is None
        assert new.status.phase == PHASE_QUEUED

    def test_resubmit_rejects_non_terminal_task(self, cfg):
        submit(cfg, command=["echo"], task_id="live1")

        with pytest.raises(ValueError, match="Cannot resubmit"):
            resubmit(cfg, "live1", command=["echo", "again"])

        assert not resubmit_operation_path(cfg, "live1").exists()

    def test_resubmit_rejects_batch_member(self, cfg):
        task = self._make_terminal_task(cfg, PHASE_FAILED, task_id="batch-task", batch_id="b1")

        with pytest.raises(ValueError, match="Batch task resubmit"):
            resubmit(cfg, task.task_id, command=["echo", "again"])

        assert load_task(cfg, task.task_id).batch_id == "b1"
        assert not resubmit_operation_path(cfg, task.task_id).exists()

    def test_repair_completes_resubmit_after_delete_old(self, cfg):
        task = self._make_terminal_task(cfg, PHASE_FAILED, task_id="repair1")
        operation = self._save_resubmit_op(cfg, task, command=["echo", "new"])
        from qqtools.plugins.qexp.api import _delete_task_truth, _advance_resubmit_operation

        _advance_resubmit_operation(cfg, operation, "deleting_old")
        _delete_task_truth(cfg, task)

        result = repair_metadata(cfg)
        repaired = load_task(cfg, "repair1")
        assert "repair1" in result["repaired_resubmits"]
        assert repaired.status.phase == PHASE_QUEUED
        assert repaired.spec.command == ["echo", "new"]
        assert not resubmit_operation_path(cfg, "repair1").exists()

    def test_repair_commits_when_new_truth_already_exists(self, cfg):
        task = self._make_terminal_task(cfg, PHASE_FAILED, task_id="repair2")
        operation = self._save_resubmit_op(cfg, task, command=["echo", "newer"])
        from qqtools.plugins.qexp.api import _advance_resubmit_operation, _create_task

        _advance_resubmit_operation(cfg, operation, "creating_new")
        new_task = _create_task(
            cfg=cfg,
            command=["echo", "newer"],
            requested_gpus=1,
            task_id="repair2",
            name=task.name,
            batch_id=None,
            group=task.group,
            machine_name=cfg.machine_name,
            attempt=1,
        )
        from qqtools.plugins.qexp.api import _persist_submitted_task_truth

        delete_then_ignore = resubmit_operation_path(cfg, "repair2")
        from qqtools.plugins.qexp.api import _delete_task_truth
        _delete_task_truth(cfg, task)
        _persist_submitted_task_truth(cfg, new_task)

        result = repair_metadata(cfg)
        repaired = load_task(cfg, "repair2")
        assert "repair2" in result["repaired_resubmits"]
        assert repaired.spec.command == ["echo", "newer"]
        assert not delete_then_ignore.exists()

    def test_resubmit_failure_points_to_doctor_repair(self, cfg, monkeypatch):
        task = self._make_terminal_task(cfg, PHASE_FAILED, task_id="broken1")
        import qqtools.plugins.qexp.api as api_mod

        def boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(api_mod, "_persist_submitted_task_truth", boom)

        with pytest.raises(RuntimeError, match="qexp doctor repair"):
            resubmit(cfg, task.task_id, command=["echo", "new"])

        operation = load_resubmit_operation(cfg, task.task_id)
        assert operation.state == "creating_new"


# ---------------------------------------------------------------------------
# Batch submit
# ---------------------------------------------------------------------------


class TestBatchSubmit:
    def _write_manifest(self, tmp_path, content):
        p = tmp_path / "manifest.yaml"
        p.write_text(yaml.dump(content), encoding="utf-8")
        return p

    def test_basic_batch(self, cfg, tmp_path):
        manifest = self._write_manifest(tmp_path, {
            "batch": {"name": "sweep"},
            "defaults": {"requested_gpus": 1},
            "tasks": [
                {"name": "t1", "command": ["echo", "1"]},
                {"name": "t2", "command": ["echo", "2"]},
            ],
        })
        batch = batch_submit(cfg, manifest)
        assert batch.name == "sweep"
        assert len(batch.task_ids) == 2
        assert batch.commit_state == BATCH_COMMIT_COMMITTED
        assert batch.expected_task_count == 2
        assert batch.summary.total == 2
        assert batch.summary.queued == 2

    def test_tasks_persisted(self, cfg, tmp_path):
        manifest = self._write_manifest(tmp_path, {
            "tasks": [{"command": ["echo"]}],
        })
        batch = batch_submit(cfg, manifest)
        t = load_task(cfg, batch.task_ids[0])
        assert t.batch_id == batch.batch_id
        assert t.spec.command == ["echo"]

    def test_gpu_override(self, cfg, tmp_path):
        manifest = self._write_manifest(tmp_path, {
            "defaults": {"requested_gpus": 1},
            "tasks": [
                {"command": ["echo"], "requested_gpus": 4},
            ],
        })
        batch = batch_submit(cfg, manifest)
        t = load_task(cfg, batch.task_ids[0])
        assert t.spec.requested_gpus == 4

    def test_empty_tasks_fails(self, cfg, tmp_path):
        manifest = self._write_manifest(tmp_path, {"tasks": []})
        with pytest.raises(ValueError, match="at least one task"):
            batch_submit(cfg, manifest)

    def test_missing_command_fails(self, cfg, tmp_path):
        manifest = self._write_manifest(tmp_path, {
            "tasks": [{"name": "oops"}],
        })
        with pytest.raises(ValueError, match="command"):
            batch_submit(cfg, manifest)

    def test_duplicate_task_id_fails_before_persist(self, cfg, tmp_path):
        manifest = self._write_manifest(tmp_path, {
            "tasks": [
                {"task_id": "dup", "command": ["echo", "1"]},
                {"task_id": "dup", "command": ["echo", "2"]},
            ],
        })
        with pytest.raises(ValueError, match="Duplicate task_id"):
            batch_submit(cfg, manifest)

    def test_batch_indexed(self, cfg, tmp_path):
        manifest = self._write_manifest(tmp_path, {
            "tasks": [{"command": ["echo"]}],
        })
        batch = batch_submit(cfg, manifest)
        ids = load_index(cfg, "batch", batch.batch_id)
        assert batch.task_ids[0] in ids

    def test_batch_group_inheritance_and_override(self, cfg, tmp_path):
        manifest = self._write_manifest(tmp_path, {
            "batch": {"name": "sweep", "group": "contract_n_4and6"},
            "tasks": [
                {"task_id": "t1", "command": ["echo", "1"]},
                {"task_id": "t2", "group": "regrouped_debug", "command": ["echo", "2"]},
            ],
        })
        batch = batch_submit(cfg, manifest)
        assert batch.group == "contract_n_4and6"
        assert batch.batch_id in load_index(cfg, "batch_group", "contract_n_4and6")
        assert load_task(cfg, "t1").group == "contract_n_4and6"
        assert load_task(cfg, "t2").group == "regrouped_debug"
        assert "t1" in load_index(cfg, "group", "contract_n_4and6")
        assert "t2" in load_index(cfg, "group", "regrouped_debug")

    def test_existing_task_id_conflict_fails_before_batch_write(self, cfg, tmp_path):
        submit(cfg, command=["echo"], task_id="taken")
        manifest = self._write_manifest(tmp_path, {
            "tasks": [{"task_id": "taken", "command": ["echo", "1"]}],
        })
        with pytest.raises(ValueError, match="already exists"):
            batch_submit(cfg, manifest)

    def test_partial_failure_marks_batch_aborted(self, cfg, tmp_path, monkeypatch):
        manifest = self._write_manifest(tmp_path, {
            "tasks": [
                {"task_id": "ok-task", "command": ["echo", "1"]},
                {"task_id": "boom-task", "command": ["echo", "2"]},
            ],
        })

        from qqtools.plugins.qexp import api as api_mod

        original_save_task = api_mod.save_task

        def _boom(current_cfg, task):
            if task.task_id == "boom-task":
                raise RuntimeError("boom")
            return original_save_task(current_cfg, task)

        monkeypatch.setattr(api_mod, "save_task", _boom)
        with pytest.raises(RuntimeError, match="boom"):
            batch_submit(cfg, manifest)

        batches = list(load_index(cfg, "batch_group", "missing"))
        assert batches == []
        batch_files = list((cfg.shared_root / "global" / "batches").glob("*.json"))
        assert len(batch_files) == 1
        aborted = load_batch(cfg, batch_files[0].stem)
        assert aborted.commit_state == BATCH_COMMIT_ABORTED
        assert aborted.expected_task_count == 0
        assert aborted.task_ids == []
        assert load_index(cfg, "state", PHASE_QUEUED) == []
        with pytest.raises(FileNotFoundError):
            load_task(cfg, "ok-task")


# ---------------------------------------------------------------------------
# Batch retry
# ---------------------------------------------------------------------------


class TestBatchRetry:
    def _setup_batch_with_failures(self, cfg, tmp_path):
        manifest_path = tmp_path / "m.yaml"
        manifest_path.write_text(yaml.dump({
            "batch": {"name": "test"},
            "tasks": [
                {"name": "ok", "command": ["echo", "ok"]},
                {"name": "fail", "command": ["echo", "fail"]},
                {"name": "cancel", "command": ["echo", "cancel"]},
            ],
        }), encoding="utf-8")
        batch = batch_submit(cfg, manifest_path)

        # Mark second as failed
        from qqtools.plugins.qexp.indexes import update_index_on_phase_change
        t_fail = load_task(cfg, batch.task_ids[1])
        t_fail.status.phase = PHASE_FAILED
        cas_update_task(cfg, t_fail, t_fail.meta.revision)
        update_index_on_phase_change(cfg, t_fail.task_id, PHASE_QUEUED, PHASE_FAILED)

        # Mark third as cancelled
        t_cancel = load_task(cfg, batch.task_ids[2])
        t_cancel.status.phase = PHASE_CANCELLED
        cas_update_task(cfg, t_cancel, t_cancel.meta.revision)
        update_index_on_phase_change(cfg, t_cancel.task_id, PHASE_QUEUED, PHASE_CANCELLED)

        return batch

    def test_retry_failed(self, cfg, tmp_path):
        batch = self._setup_batch_with_failures(cfg, tmp_path)
        new_tasks = batch_retry_failed(cfg, batch.batch_id)
        assert len(new_tasks) == 1
        assert new_tasks[0].lineage.retry_of == batch.task_ids[1]

    def test_retry_cancelled(self, cfg, tmp_path):
        batch = self._setup_batch_with_failures(cfg, tmp_path)
        new_tasks = batch_retry_cancelled(cfg, batch.batch_id)
        assert len(new_tasks) == 1
        assert new_tasks[0].lineage.retry_of == batch.task_ids[2]

    def test_retry_failed_blocked_by_policy(self, cfg, tmp_path):
        manifest_path = tmp_path / "m2.yaml"
        import yaml
        manifest_path.write_text(yaml.dump({
            "batch": {
                "name": "strict",
                "policy": {"allow_retry_failed": False},
            },
            "tasks": [{"command": ["echo"]}],
        }), encoding="utf-8")
        batch = batch_submit(cfg, manifest_path)

        # Force task to failed
        from qqtools.plugins.qexp.indexes import update_index_on_phase_change
        t = load_task(cfg, batch.task_ids[0])
        t.status.phase = PHASE_FAILED
        cas_update_task(cfg, t, t.meta.revision)
        update_index_on_phase_change(cfg, t.task_id, PHASE_QUEUED, PHASE_FAILED)

        with pytest.raises(ValueError, match="disallows retrying failed"):
            batch_retry_failed(cfg, batch.batch_id)

    def test_retry_cancelled_blocked_by_policy(self, cfg, tmp_path):
        manifest_path = tmp_path / "m3.yaml"
        import yaml
        manifest_path.write_text(yaml.dump({
            "batch": {
                "name": "strict",
                "policy": {"allow_retry_cancelled": False},
            },
            "tasks": [{"command": ["echo"]}],
        }), encoding="utf-8")
        batch = batch_submit(cfg, manifest_path)

        from qqtools.plugins.qexp.indexes import update_index_on_phase_change
        t = load_task(cfg, batch.task_ids[0])
        t.status.phase = PHASE_CANCELLED
        cas_update_task(cfg, t, t.meta.revision)
        update_index_on_phase_change(cfg, t.task_id, PHASE_QUEUED, PHASE_CANCELLED)

        with pytest.raises(ValueError, match="disallows retrying cancelled"):
            batch_retry_cancelled(cfg, batch.batch_id)

    def test_retry_rejects_non_committed_batch(self, cfg, tmp_path):
        manifest_path = tmp_path / "m4.yaml"
        manifest_path.write_text(yaml.dump({
            "tasks": [{"command": ["echo"]}],
        }), encoding="utf-8")
        batch = batch_submit(cfg, manifest_path)
        batch.commit_state = BATCH_COMMIT_PREPARING
        from qqtools.plugins.qexp.storage import save_batch
        save_batch(cfg, batch)

        with pytest.raises(ValueError, match="not committed"):
            batch_retry_failed(cfg, batch.batch_id)


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------


class TestLogs:
    def test_read_logs(self, cfg):
        from qqtools.plugins.qexp.api import read_logs
        from qqtools.plugins.qexp.layout import runtime_log_path

        t = submit(cfg, command=["echo"], task_id="log-task")
        log_path = runtime_log_path(cfg, "log-task")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("line 1\nline 2\n", encoding="utf-8")

        content = read_logs(cfg, "log-task")
        assert "line 1" in content
        assert "line 2" in content

    def test_read_logs_missing(self, cfg):
        from qqtools.plugins.qexp.api import read_logs

        t = submit(cfg, command=["echo"], task_id="no-log")
        with pytest.raises(FileNotFoundError):
            read_logs(cfg, "no-log")

    def test_get_log_path(self, cfg):
        submit(cfg, command=["echo"], task_id="some-task")
        path = get_log_path(cfg, "some-task")
        assert "some-task" in str(path)
        assert "logs" in str(path)

    def test_get_log_path_uses_task_machine_runtime_root(self, tmp_path):
        cfg_a = init_shared_root(tmp_path / ".qexp", "dev1", runtime_root=tmp_path / "runtime-dev1")
        cfg_b = init_shared_root(tmp_path / ".qexp", "dev2", runtime_root=tmp_path / "runtime-dev2")
        submit(cfg_b, command=["echo"], task_id="remote-log-task")
        path = get_log_path(cfg_a, "remote-log-task")
        assert str(path).startswith(str(cfg_b.runtime_root))


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------


class TestClean:
    def _make_old_succeeded_task(self, cfg, task_id: str):
        from qqtools.plugins.qexp.indexes import update_index_on_phase_change
        from qqtools.plugins.qexp.layout import runtime_log_path

        t = submit(cfg, command=["echo"], task_id=task_id)
        t.status.phase = PHASE_SUCCEEDED
        t.timestamps.finished_at = "2020-01-01T00:00:00Z"  # very old
        cas_update_task(cfg, t, t.meta.revision)
        update_index_on_phase_change(cfg, task_id, PHASE_QUEUED, PHASE_SUCCEEDED)

        # Create a log file
        log_path = runtime_log_path(cfg, task_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("log content", encoding="utf-8")
        return t

    def test_clean_dry_run(self, cfg):
        self._make_old_succeeded_task(cfg, "old-1")
        result = clean(cfg, dry_run=True)
        assert result["dry_run"] is True
        assert result["planned_task_count"] == 1
        assert result["deleted_task_count"] == 0
        assert "old-1" in result["task_ids"]
        # Task should still exist (dry run)
        loaded = load_task(cfg, "old-1")
        assert loaded is not None

    def test_clean_real(self, cfg):
        from qqtools.plugins.qexp.layout import runtime_log_path

        self._make_old_succeeded_task(cfg, "old-2")
        result = clean(cfg, dry_run=False)
        assert result["deleted_task_count"] == 1

        with pytest.raises(FileNotFoundError):
            load_task(cfg, "old-2")

        log_path = runtime_log_path(cfg, "old-2")
        assert not log_path.is_file()

    def test_clean_skips_recent(self, cfg):
        from qqtools.plugins.qexp.indexes import update_index_on_phase_change
        from qqtools.plugins.qexp.models import utc_now_iso

        t = submit(cfg, command=["echo"], task_id="recent")
        t.status.phase = PHASE_SUCCEEDED
        t.timestamps.finished_at = utc_now_iso()  # just now
        cas_update_task(cfg, t, t.meta.revision)
        update_index_on_phase_change(cfg, t.task_id, PHASE_QUEUED, PHASE_SUCCEEDED)

        result = clean(cfg, dry_run=False)
        assert result["deleted_task_count"] == 0

    def test_clean_skips_failed_by_default(self, cfg):
        from qqtools.plugins.qexp.indexes import update_index_on_phase_change

        t = submit(cfg, command=["echo"], task_id="fail-old")
        t.status.phase = PHASE_FAILED
        t.timestamps.finished_at = "2020-01-01T00:00:00Z"
        cas_update_task(cfg, t, t.meta.revision)
        update_index_on_phase_change(cfg, t.task_id, PHASE_QUEUED, PHASE_FAILED)

        result = clean(cfg, dry_run=False)
        assert result["deleted_task_count"] == 0

    def test_clean_includes_failed(self, cfg):
        from qqtools.plugins.qexp.indexes import update_index_on_phase_change

        t = submit(cfg, command=["echo"], task_id="fail-old2")
        t.status.phase = PHASE_FAILED
        t.timestamps.finished_at = "2020-01-01T00:00:00Z"
        cas_update_task(cfg, t, t.meta.revision)
        update_index_on_phase_change(cfg, t.task_id, PHASE_QUEUED, PHASE_FAILED)

        result = clean(cfg, dry_run=False, include_failed=True)
        assert result["deleted_task_count"] == 1

    def test_single_task_clean_updates_batch_truth_and_indexes(self, cfg, tmp_path):
        manifest = tmp_path / "batch.yaml"
        manifest.write_text(yaml.dump({
            "batch": {"name": "sweep"},
            "tasks": [
                {"task_id": "keep-me", "command": ["echo", "1"]},
                {"task_id": "drop-me", "command": ["echo", "2"]},
            ],
        }), encoding="utf-8")
        batch = batch_submit(cfg, manifest)

        from qqtools.plugins.qexp.indexes import update_index_on_phase_change
        from qqtools.plugins.qexp.layout import runtime_log_path

        doomed = load_task(cfg, "drop-me")
        doomed.status.phase = PHASE_SUCCEEDED
        doomed.timestamps.finished_at = "2020-01-01T00:00:00Z"
        cas_update_task(cfg, doomed, doomed.meta.revision)
        update_index_on_phase_change(cfg, doomed.task_id, PHASE_QUEUED, PHASE_SUCCEEDED)
        runtime_log_path(cfg, doomed.task_id).write_text("done", encoding="utf-8")

        result = clean(cfg, task_id="drop-me")
        assert result["mode"] == "single_task"
        assert result["deleted_task_count"] == 1
        assert result["repaired_batches"] == [batch.batch_id]

        with pytest.raises(FileNotFoundError):
            load_task(cfg, "drop-me")

        reloaded_batch = load_batch(cfg, batch.batch_id)
        assert reloaded_batch.task_ids == ["keep-me"]
        assert reloaded_batch.summary.total == 1
        assert reloaded_batch.summary.queued == 1
        assert "drop-me" not in load_index(cfg, "batch", batch.batch_id)

    def test_single_task_clean_rejects_non_terminal(self, cfg):
        submit(cfg, command=["echo"], task_id="active-task")
        with pytest.raises(ValueError, match="Only terminal tasks can be cleaned"):
            clean(cfg, task_id="active-task")

    def test_single_task_clean_rejects_broken_batch_reference(self, cfg, tmp_path):
        manifest = tmp_path / "broken-batch.yaml"
        manifest.write_text(yaml.dump({
            "batch": {"name": "broken"},
            "tasks": [{"task_id": "broken-task", "command": ["echo"]}],
        }), encoding="utf-8")
        batch = batch_submit(cfg, manifest)

        from qqtools.plugins.qexp.indexes import update_index_on_phase_change
        from qqtools.plugins.qexp.layout import batch_path

        task = load_task(cfg, "broken-task")
        task.status.phase = PHASE_SUCCEEDED
        task.timestamps.finished_at = "2020-01-01T00:00:00Z"
        cas_update_task(cfg, task, task.meta.revision)
        update_index_on_phase_change(cfg, task.task_id, PHASE_QUEUED, PHASE_SUCCEEDED)
        batch_path(cfg, batch.batch_id).unlink()

        with pytest.raises(FileNotFoundError):
            clean(cfg, task_id="broken-task")

        assert load_task(cfg, "broken-task").task_id == "broken-task"

    def test_single_task_clean_reports_unresolved_remote_log(self, tmp_path):
        cfg_a = init_shared_root(tmp_path / ".qexp", "dev1", runtime_root=tmp_path / "runtime-dev1")
        cfg_b = init_shared_root(tmp_path / ".qexp", "dev2", runtime_root=tmp_path / "runtime-dev2")
        submit(cfg_b, command=["echo"], task_id="remote-task")

        from qqtools.plugins.qexp.indexes import update_index_on_phase_change
        task = load_task(cfg_b, "remote-task")
        task.status.phase = PHASE_SUCCEEDED
        task.timestamps.finished_at = "2020-01-01T00:00:00Z"
        cas_update_task(cfg_b, task, task.meta.revision)
        update_index_on_phase_change(cfg_b, task.task_id, PHASE_QUEUED, PHASE_SUCCEEDED)

        machine_path = cfg_b.shared_root / "machines" / "dev2" / "machine.json"
        machine_path.unlink()

        result = clean(cfg_a, task_id="remote-task")
        assert result["deleted_task_count"] == 1
        assert result["log_results"][0]["status"] == "unresolved"
