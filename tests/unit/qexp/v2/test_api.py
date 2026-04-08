from __future__ import annotations

import json

import pytest
import yaml

from qqtools.plugins.qexp.v2.api import (
    batch_retry_cancelled,
    batch_retry_failed,
    batch_submit,
    cancel,
    retry,
    submit,
)
from qqtools.plugins.qexp.v2.indexes import load_index
from qqtools.plugins.qexp.v2.layout import init_shared_root
from qqtools.plugins.qexp.v2.models import (
    PHASE_CANCELLED,
    PHASE_FAILED,
    PHASE_QUEUED,
    PHASE_RUNNING,
    PHASE_SUCCEEDED,
)
from qqtools.plugins.qexp.v2.storage import cas_update_task, load_batch, load_task


@pytest.fixture()
def cfg(tmp_path):
    return init_shared_root(tmp_path / "shared", "dev1")


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

    def test_with_name(self, cfg):
        t = submit(cfg, command=["echo"], name="test run")
        assert t.name == "test run"


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
        from qqtools.plugins.qexp.v2.indexes import update_index_on_phase_change
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
        from qqtools.plugins.qexp.v2.indexes import update_index_on_phase_change
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

    def test_retry_non_terminal_fails(self, cfg):
        t = submit(cfg, command=["echo"])
        with pytest.raises(ValueError, match="Only terminal"):
            retry(cfg, t.task_id)

    def test_retry_indexed(self, cfg):
        original = self._make_failed_task(cfg)
        new = retry(cfg, original.task_id)
        assert new.task_id in load_index(cfg, "state", PHASE_QUEUED)


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

    def test_batch_indexed(self, cfg, tmp_path):
        manifest = self._write_manifest(tmp_path, {
            "tasks": [{"command": ["echo"]}],
        })
        batch = batch_submit(cfg, manifest)
        ids = load_index(cfg, "batch", batch.batch_id)
        assert batch.task_ids[0] in ids


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
        from qqtools.plugins.qexp.v2.indexes import update_index_on_phase_change
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
        from qqtools.plugins.qexp.v2.indexes import update_index_on_phase_change
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

        from qqtools.plugins.qexp.v2.indexes import update_index_on_phase_change
        t = load_task(cfg, batch.task_ids[0])
        t.status.phase = PHASE_CANCELLED
        cas_update_task(cfg, t, t.meta.revision)
        update_index_on_phase_change(cfg, t.task_id, PHASE_QUEUED, PHASE_CANCELLED)

        with pytest.raises(ValueError, match="disallows retrying cancelled"):
            batch_retry_cancelled(cfg, batch.batch_id)


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------


class TestLogs:
    def test_read_logs(self, cfg):
        from qqtools.plugins.qexp.v2.api import read_logs
        from qqtools.plugins.qexp.v2.layout import runtime_log_path

        t = submit(cfg, command=["echo"], task_id="log-task")
        log_path = runtime_log_path(cfg, "log-task")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("line 1\nline 2\n", encoding="utf-8")

        content = read_logs(cfg, "log-task")
        assert "line 1" in content
        assert "line 2" in content

    def test_read_logs_missing(self, cfg):
        from qqtools.plugins.qexp.v2.api import read_logs

        t = submit(cfg, command=["echo"], task_id="no-log")
        with pytest.raises(FileNotFoundError):
            read_logs(cfg, "no-log")

    def test_get_log_path(self, cfg):
        from qqtools.plugins.qexp.v2.api import get_log_path

        path = get_log_path(cfg, "some-task")
        assert "some-task" in str(path)
        assert "logs" in str(path)


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------


class TestClean:
    def _make_old_succeeded_task(self, cfg, task_id: str):
        from qqtools.plugins.qexp.v2.indexes import update_index_on_phase_change
        from qqtools.plugins.qexp.v2.layout import runtime_log_path

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
        from qqtools.plugins.qexp.v2.api import clean

        self._make_old_succeeded_task(cfg, "old-1")
        result = clean(cfg, dry_run=True)
        assert result["dry_run"] is True
        assert result["deleted_task_count"] == 1
        assert "old-1" in result["task_ids"]
        # Task should still exist (dry run)
        loaded = load_task(cfg, "old-1")
        assert loaded is not None

    def test_clean_real(self, cfg):
        from qqtools.plugins.qexp.v2.api import clean
        from qqtools.plugins.qexp.v2.layout import runtime_log_path

        self._make_old_succeeded_task(cfg, "old-2")
        result = clean(cfg, dry_run=False)
        assert result["deleted_task_count"] == 1

        with pytest.raises(FileNotFoundError):
            load_task(cfg, "old-2")

        log_path = runtime_log_path(cfg, "old-2")
        assert not log_path.is_file()

    def test_clean_skips_recent(self, cfg):
        from qqtools.plugins.qexp.v2.api import clean
        from qqtools.plugins.qexp.v2.indexes import update_index_on_phase_change
        from qqtools.plugins.qexp.v2.models import utc_now_iso

        t = submit(cfg, command=["echo"], task_id="recent")
        t.status.phase = PHASE_SUCCEEDED
        t.timestamps.finished_at = utc_now_iso()  # just now
        cas_update_task(cfg, t, t.meta.revision)
        update_index_on_phase_change(cfg, t.task_id, PHASE_QUEUED, PHASE_SUCCEEDED)

        result = clean(cfg, dry_run=False)
        assert result["deleted_task_count"] == 0

    def test_clean_skips_failed_by_default(self, cfg):
        from qqtools.plugins.qexp.v2.api import clean
        from qqtools.plugins.qexp.v2.indexes import update_index_on_phase_change

        t = submit(cfg, command=["echo"], task_id="fail-old")
        t.status.phase = PHASE_FAILED
        t.timestamps.finished_at = "2020-01-01T00:00:00Z"
        cas_update_task(cfg, t, t.meta.revision)
        update_index_on_phase_change(cfg, t.task_id, PHASE_QUEUED, PHASE_FAILED)

        result = clean(cfg, dry_run=False)
        assert result["deleted_task_count"] == 0

    def test_clean_includes_failed(self, cfg):
        from qqtools.plugins.qexp.v2.api import clean
        from qqtools.plugins.qexp.v2.indexes import update_index_on_phase_change

        t = submit(cfg, command=["echo"], task_id="fail-old2")
        t.status.phase = PHASE_FAILED
        t.timestamps.finished_at = "2020-01-01T00:00:00Z"
        cas_update_task(cfg, t, t.meta.revision)
        update_index_on_phase_change(cfg, t.task_id, PHASE_QUEUED, PHASE_FAILED)

        result = clean(cfg, dry_run=False, include_failed=True)
        assert result["deleted_task_count"] == 1
