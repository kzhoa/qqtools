from __future__ import annotations

import json
import time

import pytest

from qqtools.plugins.qexp.v2.api import submit
from qqtools.plugins.qexp.v2.doctor import (
    cleanup_stale_locks,
    rebuild_indexes,
    repair_orphans,
    verify_integrity,
)
from qqtools.plugins.qexp.v2.indexes import load_index, update_index_on_phase_change
from qqtools.plugins.qexp.v2.layout import (
    global_locks_dir,
    global_tasks_dir,
    init_shared_root,
    task_path,
)
from qqtools.plugins.qexp.v2.models import PHASE_QUEUED, PHASE_RUNNING
from qqtools.plugins.qexp.v2.storage import cas_update_task, load_task


@pytest.fixture()
def cfg(tmp_path):
    return init_shared_root(tmp_path / "shared", "dev1")


class TestRebuildIndexes:
    def test_rebuild(self, cfg):
        submit(cfg, command=["echo", "1"])
        submit(cfg, command=["echo", "2"])
        stats = rebuild_indexes(cfg)
        assert stats["total_tasks"] == 2
        assert stats["states"].get("queued") == 2

    def test_rebuild_empty(self, cfg):
        stats = rebuild_indexes(cfg)
        assert stats["total_tasks"] == 0


class TestCleanupStaleLocks:
    def test_removes_old_locks(self, cfg):
        locks_dir = global_locks_dir(cfg)
        locks_dir.mkdir(parents=True, exist_ok=True)
        stale = locks_dir / "old.lock"
        stale.touch()
        import os
        os.utime(stale, (time.time() - 600, time.time() - 600))

        cleaned = cleanup_stale_locks(cfg, max_age_seconds=300.0)
        assert len(cleaned) == 1
        assert not stale.exists()

    def test_preserves_fresh_locks(self, cfg):
        locks_dir = global_locks_dir(cfg)
        locks_dir.mkdir(parents=True, exist_ok=True)
        fresh = locks_dir / "fresh.lock"
        fresh.touch()
        cleaned = cleanup_stale_locks(cfg, max_age_seconds=300.0)
        assert len(cleaned) == 0
        assert fresh.exists()


class TestVerifyIntegrity:
    def test_healthy(self, cfg):
        submit(cfg, command=["echo"])
        result = verify_integrity(cfg)
        assert result["ok"] is True
        assert result["tasks_checked"] == 1

    def test_id_mismatch(self, cfg):
        t = submit(cfg, command=["echo"])
        p = task_path(cfg, t.task_id)
        data = json.loads(p.read_text(encoding="utf-8"))
        data["task"]["task_id"] = "wrong-id"
        p.write_text(json.dumps(data), encoding="utf-8")
        result = verify_integrity(cfg)
        assert not result["ok"]
        assert any("mismatch" in i for i in result["issues"])

    def test_corrupt_file(self, cfg):
        t = submit(cfg, command=["echo"])
        p = task_path(cfg, t.task_id)
        p.write_text("not json", encoding="utf-8")
        result = verify_integrity(cfg)
        assert not result["ok"]

    def test_empty_dir(self, cfg):
        result = verify_integrity(cfg)
        assert result["ok"]
        assert result["tasks_checked"] == 0


class TestRepairOrphans:
    def test_no_orphans_when_healthy(self, cfg):
        submit(cfg, command=["echo"])
        orphaned = repair_orphans(cfg)
        assert orphaned == []

    def test_marks_orphaned(self, cfg):
        t = submit(cfg, command=["echo"])
        # Force to running
        t = load_task(cfg, t.task_id)
        t.status.phase = PHASE_RUNNING
        cas_update_task(cfg, t, t.meta.revision)
        update_index_on_phase_change(cfg, t.task_id, PHASE_QUEUED, PHASE_RUNNING)

        # Machine has no agent state -> stale
        orphaned = repair_orphans(cfg)
        assert t.task_id in orphaned
