from __future__ import annotations

import json
import time

import pytest

from qqtools.plugins.qexp.api import submit
from qqtools.plugins.qexp.agent import start_agent_record, stop_agent_record
from qqtools.plugins.qexp.doctor import (
    build_verify_jsonl_records,
    cleanup_stale_locks,
    repair_metadata,
    rebuild_indexes,
    repair_orphans,
    resolve_verify_exit_code,
    verify_integrity,
)
from qqtools.plugins.qexp.indexes import load_index, update_index_on_phase_change
from qqtools.plugins.qexp.lifecycle import read_agent_snapshot
from qqtools.plugins.qexp.layout import (
    agent_state_path,
    batch_path,
    global_locks_dir,
    global_tasks_dir,
    init_shared_root,
    resubmit_operation_path,
    task_path,
)
from qqtools.plugins.qexp.models import PHASE_QUEUED, PHASE_RUNNING, PHASE_SUCCEEDED
from qqtools.plugins.qexp.storage import (
    cas_update_task,
    load_claim,
    load_batch,
    load_task,
    save_batch,
    save_claim,
    save_resubmit_operation,
    write_atomic_json,
)


@pytest.fixture()
def cfg(tmp_path):
    return init_shared_root(tmp_path / ".qexp", "dev1", runtime_root=tmp_path / "runtime")


class TestRebuildIndexes:
    def test_rebuild(self, cfg):
        submit(cfg, command=["echo", "1"])
        submit(cfg, command=["echo", "2"])
        stats = rebuild_indexes(cfg)
        assert stats["index_stats"]["total_tasks"] == 2
        assert stats["index_stats"]["states"].get("queued") == 2
        assert stats["governance"]["total_tasks"] == 2

    def test_rebuild_empty(self, cfg):
        stats = rebuild_indexes(cfg)
        assert stats["index_stats"]["total_tasks"] == 0


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
        assert result["severity"] == "ok"
        assert result["recommended_actions"] == []
        assert result["issues"] == []
        assert result["messages"] == []
        assert result["issue_count_by_category"] == {}
        assert result["issue_count_by_code"] == {}
        assert result["tasks_checked"] == 1
        assert result["root_manifest"]["project_root"] == str(cfg.project_root)
        assert result["governance"]["total_tasks"] == 1

    def test_id_mismatch(self, cfg):
        t = submit(cfg, command=["echo"])
        p = task_path(cfg, t.task_id)
        data = json.loads(p.read_text(encoding="utf-8"))
        data["task"]["task_id"] = "wrong-id"
        p.write_text(json.dumps(data), encoding="utf-8")
        result = verify_integrity(cfg)
        assert not result["ok"]
        assert any(issue["code"] == "task_filename_id_mismatch" for issue in result["issues"])
        assert any("mismatch" in message for message in result["messages"])

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
        assert result["root_manifest"]["shared_root"] == str(cfg.shared_root)

    def test_detects_batch_dangling_reference(self, cfg, tmp_path):
        from qqtools.plugins.qexp.api import batch_submit
        import yaml

        manifest = tmp_path / "batch.yaml"
        manifest.write_text(yaml.dump({
            "tasks": [{"task_id": "t1", "command": ["echo"]}],
        }), encoding="utf-8")
        batch = batch_submit(cfg, manifest)
        task_path(cfg, "t1").unlink()

        result = verify_integrity(cfg)
        assert not result["ok"]
        assert any(issue["code"] == "batch_missing_task_reference" for issue in result["issues"])

    def test_detects_preparing_batch_that_should_be_committed(self, cfg, tmp_path):
        from qqtools.plugins.qexp.api import batch_submit
        import yaml

        manifest = tmp_path / "batch-preparing.yaml"
        manifest.write_text(yaml.dump({
            "tasks": [{"task_id": "t1", "command": ["echo"]}],
        }), encoding="utf-8")
        batch = batch_submit(cfg, manifest)
        batch.commit_state = "preparing"
        save_batch(cfg, batch)

        result = verify_integrity(cfg)
        assert not result["ok"]
        assert any(issue["code"] == "batch_preparing_complete_task_set" for issue in result["issues"])

    def test_detects_invalid_resubmit_operation(self, cfg):
        from qqtools.plugins.qexp.api import _build_resubmit_operation

        task = submit(cfg, command=["echo"], task_id="bad-op")
        task = load_task(cfg, "bad-op")
        task.status.phase = PHASE_SUCCEEDED
        task.timestamps.finished_at = "2026-01-01T00:00:00Z"
        cas_update_task(cfg, task, task.meta.revision)
        update_index_on_phase_change(cfg, "bad-op", PHASE_QUEUED, PHASE_SUCCEEDED)

        operation = _build_resubmit_operation(
            cfg,
            task,
            command=["echo", "new"],
            requested_gpus=1,
            name=task.name,
            group=task.group,
        )
        operation.old_task_summary.batch_id = "batch-x"
        save_resubmit_operation(cfg, operation)

        result = verify_integrity(cfg)
        assert not result["ok"]
        assert any(issue["code"] == "resubmit_illegal_batch_target" for issue in result["issues"])

    def test_detects_index_drift(self, cfg):
        task = submit(cfg, command=["echo"], task_id="drift-task")
        from qqtools.plugins.qexp.layout import index_by_state_dir

        write_atomic_json(index_by_state_dir(cfg) / f"{PHASE_QUEUED}.json", {"task_ids": []})

        result = verify_integrity(cfg)
        assert not result["ok"]
        assert result["severity"] == "high"
        assert result["diagnosis"]["index_drift_only"] is True
        assert result["index_drift"]["ok"] is False
        assert any(
            action["command"] == "qexp doctor rebuild-index"
            for action in result["recommended_actions"]
        )
        assert any(
            action["action_code"] == "run_doctor_rebuild_index_state"
            and action["blocking"] is True
            for action in result["recommended_actions"]
        )
        assert result["issue_count_by_category"]["derived_index"] == 1
        assert result["issue_count_by_code"]["derived_index_state_drift"] == 1
        assert any(
            issue["code"] == "derived_index_state_drift" and issue["details"]["family"] == "state"
            for issue in result["issues"]
        )

    def test_verify_recommendations_follow_explicit_policy_table(self, cfg):
        from qqtools.plugins.qexp.api import _build_resubmit_operation
        from qqtools.plugins.qexp.layout import index_by_state_dir

        task = submit(cfg, command=["echo"], task_id="policy-broken")
        task_path(cfg, task.task_id).write_text("not json", encoding="utf-8")
        write_atomic_json(index_by_state_dir(cfg) / f"{PHASE_QUEUED}.json", {"task_ids": []})

        healthy = submit(cfg, command=["echo"], task_id="policy-batch-task")
        healthy = load_task(cfg, healthy.task_id)
        healthy.status.phase = PHASE_SUCCEEDED
        healthy.timestamps.finished_at = "2026-01-01T00:00:00Z"
        cas_update_task(cfg, healthy, healthy.meta.revision)
        update_index_on_phase_change(cfg, healthy.task_id, PHASE_QUEUED, PHASE_SUCCEEDED)

        operation = _build_resubmit_operation(
            cfg,
            healthy,
            command=["echo", "new"],
            requested_gpus=1,
            name=healthy.name,
            group=healthy.group,
        )
        operation.old_task_summary.batch_id = "batch-z"
        save_resubmit_operation(cfg, operation)

        result = verify_integrity(cfg)
        actions = {action["action_code"]: action for action in result["recommended_actions"]}
        assert "manual_fix_truth_corruption" in actions
        assert "run_doctor_repair_resubmit" in actions
        assert actions["manual_fix_truth_corruption"]["blocking"] is True
        assert actions["run_doctor_repair_resubmit"]["blocking"] is True

    def test_corrupt_truth_requires_manual_fix_recommendation(self, cfg):
        task = submit(cfg, command=["echo"], task_id="broken-truth")
        task_path(cfg, task.task_id).write_text("not json", encoding="utf-8")

        result = verify_integrity(cfg)
        assert result["severity"] == "high"
        assert result["diagnosis"]["truth_ok"] is False
        assert any(
            action["command"] == "manual_fix_required"
            for action in result["recommended_actions"]
        )
        assert any(
            action["action_code"] == "manual_fix_truth_corruption"
            and action["blocking"] is True
            for action in result["recommended_actions"]
        )

    def test_detects_creating_new_snapshot_mismatch(self, cfg):
        from qqtools.plugins.qexp.api import _build_resubmit_operation, _advance_resubmit_operation

        task = submit(cfg, command=["echo"], task_id="mismatch1")
        task = load_task(cfg, "mismatch1")
        task.status.phase = PHASE_SUCCEEDED
        task.timestamps.finished_at = "2026-01-01T00:00:00Z"
        cas_update_task(cfg, task, task.meta.revision)
        update_index_on_phase_change(cfg, "mismatch1", PHASE_QUEUED, PHASE_SUCCEEDED)

        operation = _build_resubmit_operation(
            cfg,
            task,
            command=["echo", "fresh"],
            requested_gpus=1,
            name=task.name,
            group=task.group,
        )
        save_resubmit_operation(cfg, operation)
        _advance_resubmit_operation(cfg, operation, "creating_new")

        # Keep the old visible truth to simulate an ambiguous conflict.
        result = verify_integrity(cfg)
        assert not result["ok"]
        assert any(
            issue["code"] == "resubmit_visible_truth_snapshot_mismatch"
            for issue in result["issues"]
        )

    def test_verify_exit_policy_defaults_to_observe_only(self, cfg):
        task = submit(cfg, command=["echo"], task_id="broken-policy")
        task_path(cfg, task.task_id).write_text("not json", encoding="utf-8")

        result = verify_integrity(cfg)
        assert result["severity"] == "high"
        assert resolve_verify_exit_code(result) == 0
        assert resolve_verify_exit_code(result, strict=True) == 2
        assert resolve_verify_exit_code(result, fail_on="high") == 2
        assert resolve_verify_exit_code(result, fail_on="medium") == 2

    def test_verify_exit_policy_respects_threshold(self, cfg):
        task = submit(cfg, command=["echo"], task_id="drift-threshold")
        from qqtools.plugins.qexp.layout import index_by_state_dir

        write_atomic_json(index_by_state_dir(cfg) / f"{PHASE_QUEUED}.json", {"task_ids": []})

        result = verify_integrity(cfg)
        assert result["severity"] == "high"
        assert resolve_verify_exit_code(result, fail_on="high") == 2
        assert resolve_verify_exit_code(result, fail_on="medium") == 2
        assert resolve_verify_exit_code(result, strict=True) == 2

    def test_verify_jsonl_records_include_summary_issue_and_result(self, cfg):
        task = submit(cfg, command=["echo"], task_id="jsonl-broken")
        task_path(cfg, task.task_id).write_text("not json", encoding="utf-8")

        result = verify_integrity(cfg)
        records = build_verify_jsonl_records(result, strict=True)
        assert records[0]["type"] == "verify_summary"
        assert records[0]["exit_code"] == 2
        assert records[0]["fail_on"] == "low"
        issue_records = [record for record in records if record["type"] == "verify_issue"]
        assert issue_records
        assert issue_records[0]["issue_code"] == "task_truth_unreadable"
        assert issue_records[0]["category"] == "task_truth"
        recommendation_records = [
            record for record in records if record["type"] == "verify_recommendation"
        ]
        assert recommendation_records
        assert any(
            record["action_code"] == "manual_fix_truth_corruption"
            and record["blocking"] is True
            for record in recommendation_records
        )
        assert records[-1]["type"] == "verify_result"
        assert records[-1]["exit_code"] == 2


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

    def test_running_task_with_active_agent_and_claim_is_not_orphaned(self, cfg):
        t = submit(cfg, command=["echo"], task_id="alive-task")
        t = load_task(cfg, t.task_id)
        t.status.phase = PHASE_RUNNING
        cas_update_task(cfg, t, t.meta.revision)
        update_index_on_phase_change(cfg, t.task_id, PHASE_QUEUED, PHASE_RUNNING)

        start_agent_record(cfg)
        snapshot = read_agent_snapshot(cfg)
        snapshot.agent_state = "active"
        write_atomic_json(agent_state_path(cfg), snapshot.to_dict())
        save_claim(cfg, t.task_id, "2026-04-20T00:00:00Z", t.meta.revision)

        orphaned = repair_orphans(cfg, heartbeat_stale_seconds=9999.0)
        assert orphaned == []

    def test_stopped_agent_is_treated_as_stale_immediately(self, cfg):
        t = submit(cfg, command=["echo"])
        t = load_task(cfg, t.task_id)
        t.status.phase = PHASE_RUNNING
        cas_update_task(cfg, t, t.meta.revision)
        update_index_on_phase_change(cfg, t.task_id, PHASE_QUEUED, PHASE_RUNNING)

        start_agent_record(cfg)
        stop_agent_record(cfg, reason="manual_stop")

        orphaned = repair_orphans(cfg, heartbeat_stale_seconds=9999.0)
        assert t.task_id in orphaned

    def test_remote_recent_heartbeat_is_not_orphaned_by_foreign_pid(self, cfg):
        from qqtools.plugins.qexp.lifecycle import read_agent_snapshot

        other_cfg = init_shared_root(
            cfg.shared_root,
            "gpu2",
            runtime_root=cfg.runtime_root.parent / "runtime2",
        )
        t = submit(other_cfg, command=["echo"])
        t = load_task(other_cfg, t.task_id)
        t.status.phase = PHASE_RUNNING
        cas_update_task(other_cfg, t, t.meta.revision)
        update_index_on_phase_change(other_cfg, t.task_id, PHASE_QUEUED, PHASE_RUNNING)

        start_agent_record(other_cfg)
        snapshot = read_agent_snapshot(other_cfg)
        snapshot.pid = 99999999
        snapshot.agent_state = "draining"
        snapshot.last_heartbeat = "2099-04-14T00:00:00Z"
        write_atomic_json(agent_state_path(other_cfg), snapshot.to_dict())
        save_claim(other_cfg, t.task_id, "2026-04-20T00:00:00Z", t.meta.revision)

        orphaned = repair_orphans(cfg, heartbeat_stale_seconds=1.0)
        assert t.task_id not in orphaned

    def test_releases_active_claim_when_task_becomes_orphaned(self, cfg):
        t = submit(cfg, command=["echo"], task_id="claim-release")
        t = load_task(cfg, t.task_id)
        t.status.phase = PHASE_RUNNING
        cas_update_task(cfg, t, t.meta.revision)
        update_index_on_phase_change(cfg, t.task_id, PHASE_QUEUED, PHASE_RUNNING)
        save_claim(cfg, t.task_id, "2026-04-20T00:00:00Z", t.meta.revision)

        orphaned = repair_orphans(cfg)
        assert t.task_id in orphaned
        with pytest.raises(FileNotFoundError):
            load_claim(cfg, t.task_id)


class TestRepairMetadata:
    def test_prunes_missing_task_refs_and_rebuilds_summary(self, cfg, tmp_path):
        from qqtools.plugins.qexp.api import batch_submit
        import yaml

        manifest = tmp_path / "batch.yaml"
        manifest.write_text(yaml.dump({
            "tasks": [
                {"task_id": "keep", "command": ["echo"]},
                {"task_id": "gone", "command": ["echo"]},
            ],
        }), encoding="utf-8")
        batch = batch_submit(cfg, manifest)

        task_path(cfg, "gone").unlink()

        result = repair_metadata(cfg)
        assert result["repaired_batch_count"] == 1
        assert result["governance"]["total_tasks"] == 1
        assert result["index_drift"]["ok"] is True

        repaired = load_batch(cfg, batch.batch_id)
        assert repaired.task_ids == ["keep"]
        assert repaired.summary.total == 1
        assert repaired.summary.queued == 1

    def test_promotes_complete_preparing_batch_to_committed(self, cfg, tmp_path):
        from qqtools.plugins.qexp.api import batch_submit
        import yaml

        manifest = tmp_path / "promote.yaml"
        manifest.write_text(yaml.dump({
            "tasks": [{"task_id": "ready", "command": ["echo"]}],
        }), encoding="utf-8")
        batch = batch_submit(cfg, manifest)
        batch.commit_state = "preparing"
        save_batch(cfg, batch)

        result = repair_metadata(cfg)
        assert batch.batch_id in result["committed_batches"]

        repaired = load_batch(cfg, batch.batch_id)
        assert repaired.commit_state == "committed"
        assert repaired.summary.total == 1

    def test_marks_incomplete_preparing_batch_aborted(self, cfg, tmp_path):
        from qqtools.plugins.qexp.api import batch_submit
        import yaml

        manifest = tmp_path / "abort.yaml"
        manifest.write_text(yaml.dump({
            "tasks": [
                {"task_id": "keep", "command": ["echo"]},
                {"task_id": "drop", "command": ["echo"]},
            ],
        }), encoding="utf-8")
        batch = batch_submit(cfg, manifest)
        batch.commit_state = "preparing"
        save_batch(cfg, batch)
        task_path(cfg, "drop").unlink()

        result = repair_metadata(cfg)
        assert batch.batch_id in result["aborted_batches"]

        repaired = load_batch(cfg, batch.batch_id)
        assert repaired.commit_state == "aborted"
        assert repaired.task_ids == ["keep"]

    def test_repairs_resubmit_gap(self, cfg):
        from qqtools.plugins.qexp.api import _advance_resubmit_operation, _build_resubmit_operation, _delete_task_truth

        task = submit(cfg, command=["echo"], task_id="repair-gap")
        task = load_task(cfg, "repair-gap")
        task.status.phase = PHASE_SUCCEEDED
        task.timestamps.finished_at = "2026-01-01T00:00:00Z"
        cas_update_task(cfg, task, task.meta.revision)
        update_index_on_phase_change(cfg, "repair-gap", PHASE_QUEUED, PHASE_SUCCEEDED)

        operation = _build_resubmit_operation(
            cfg,
            task,
            command=["echo", "new"],
            requested_gpus=1,
            name=task.name,
            group=task.group,
        )
        save_resubmit_operation(cfg, operation)
        _advance_resubmit_operation(cfg, operation, "deleting_old")
        _delete_task_truth(cfg, task)

        result = repair_metadata(cfg)
        repaired = load_task(cfg, "repair-gap")
        assert "repair-gap" in result["repaired_resubmits"]
        assert repaired.status.phase == PHASE_QUEUED
        assert repaired.spec.command == ["echo", "new"]
        assert not resubmit_operation_path(cfg, "repair-gap").exists()

    def test_repairs_resubmit_when_replacement_task_has_started(self, cfg):
        from qqtools.plugins.qexp.api import (
            _advance_resubmit_operation,
            _build_resubmit_operation,
            _delete_task_truth,
            _materialize_resubmit_task,
            _persist_submitted_task_truth,
        )

        task = submit(cfg, command=["echo"], task_id="repair-running")
        task = load_task(cfg, "repair-running")
        task.status.phase = PHASE_SUCCEEDED
        task.timestamps.finished_at = "2026-01-01T00:00:00Z"
        cas_update_task(cfg, task, task.meta.revision)
        update_index_on_phase_change(cfg, "repair-running", PHASE_QUEUED, PHASE_SUCCEEDED)

        operation = _build_resubmit_operation(
            cfg,
            task,
            command=["echo", "replacement"],
            requested_gpus=1,
            name=task.name,
            group=task.group,
        )
        save_resubmit_operation(cfg, operation)
        _advance_resubmit_operation(cfg, operation, "deleting_old")
        _delete_task_truth(cfg, task)
        _advance_resubmit_operation(cfg, operation, "creating_new")

        replacement = _materialize_resubmit_task(operation)
        _persist_submitted_task_truth(cfg, replacement)
        replacement = load_task(cfg, "repair-running")
        replacement.status.phase = PHASE_RUNNING
        replacement.timestamps.started_at = "2026-01-01T00:01:00Z"
        cas_update_task(cfg, replacement, replacement.meta.revision)
        update_index_on_phase_change(cfg, "repair-running", PHASE_QUEUED, PHASE_RUNNING)

        result = repair_metadata(cfg)
        repaired = load_task(cfg, "repair-running")
        assert "repair-running" in result["repaired_resubmits"]
        assert repaired.status.phase == PHASE_RUNNING
        assert repaired.spec.command == ["echo", "replacement"]
        assert not resubmit_operation_path(cfg, "repair-running").exists()
