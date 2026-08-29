from __future__ import annotations

import json
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.cli import _split_machine_list, main
from qqtools.plugins.qexp.commands import task as task_commands
from qqtools.plugins.qexp.commands.group import change_worker
from qqtools.plugins.qexp.doctor import repair_metadata, verify_integrity
from qqtools.plugins.qexp.project_maintenance import offer_due_tasks
from qqtools.plugins.qexp.runtime.active_operations import active_operation_path, write_active_operation
from qqtools.plugins.qexp.runtime import availability as availability_runtime
from qqtools.plugins.qexp.runtime.availability import rebuild_deadline_indexes
from qqtools.plugins.qexp.runtime.paths import shared_paths
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task
from qqtools.plugins.qexp.scheduler import claim_task

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]

def _base_args(cfg) -> list[str]:
    return ["--shared-root", str(cfg.shared_root), "--machine", cfg.machine_name,
            "--runtime-root", str(cfg.runtime_root)]


def test_share_now_persists_journal_audit_and_keeps_home_eligible(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"], group="exp", sharing_mode="private",
                  fallback_machines="group")

    result = task_commands.share(cfg, task.task_id)

    stored = load_task(cfg, task.task_id)
    assert result.action == "share_now"
    assert stored.placement_policy["sharing_mode"] == "spillover"
    assert stored.placement_runtime["queue_scope"] == "shared"
    assert stored.meta["revision"] == task.meta["revision"] + 1
    assert (shared_paths(cfg.shared_root)["availability"] / f"{result.operation_id}.json").exists()
    event_files = list(shared_paths(cfg.shared_root)["events"].glob(f"*/{result.operation_id}.json"))
    assert event_files
    assert claim_task(cfg, task.task_id, [0]) is not None


def test_share_without_helpers_replaces_stale_private_fallback_with_group(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"], group="exp", sharing_mode="private",
                  fallback_machines=["legacy-helper"])

    task_commands.share(cfg, task.task_id)

    assert load_task(cfg, task.task_id).placement_policy["fallback_constraint"] == "group"


def test_repeated_same_share_is_idempotent_without_revision_change(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"], group="exp")
    change_worker(cfg, "exp", "g2", "add")

    first = task_commands.share(cfg, task.task_id, helper_machines=["g2"])
    revision = load_task(cfg, task.task_id).meta["revision"]
    second = task_commands.share(cfg, task.task_id, helper_machines=["g2"])

    assert first.operation_id != second.operation_id
    assert second.idempotent is True
    assert load_task(cfg, task.task_id).meta["revision"] == revision


def test_keep_local_is_idempotent_for_private_standalone_task(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    revision = load_task(cfg, task.task_id).meta["revision"]

    result = task_commands.keep_local(cfg, task.task_id)

    assert result.idempotent is True
    assert load_task(cfg, task.task_id).meta["revision"] == revision


def test_share_after_writes_and_repairs_deadline_index(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"], group="exp")

    result = task_commands.share(cfg, task.task_id, after_seconds=0)
    index_path = shared_paths(cfg.shared_root)["offer_deadlines"] / f"{task.task_id}.json"
    assert index_path.exists()
    assert read_json(index_path)["offer_deadline"]["operation_id"] == result.operation_id

    index_path.unlink()
    issues = {issue["code"] for issue in verify_integrity(cfg)["issues"]}
    assert "offer_deadline_index_missing" in issues
    repaired = repair_metadata(cfg)
    assert any(item.startswith("offer_deadline_indexes:") for item in repaired["repaired"])
    assert index_path.exists()


def test_doctor_replays_prepared_availability_operation(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"], group="exp")
    operation_id = "repair-share"
    operation_path = shared_paths(cfg.shared_root)["availability"] / f"{operation_id}.json"
    atomic_replace(operation_path, {"meta": {"schema_version": 6, "revision": 1,
        "created_at": "2026-08-06T00:00:00Z", "updated_at": "2026-08-06T00:00:00Z",
        "updated_by": {"actor_type": "cli", "machine_name": "g1", "process_id": "test"}},
        "availability_operation": {"operation_id": operation_id, "operation_type": "share_now",
        "task_id": task.task_id, "state": "prepared", "requested_by": "g1", "reason": "manual",
        "helper_machines": None, "after_seconds": None, "created_at": "2026-08-06T00:00:00Z",
        "updated_at": "2026-08-06T00:00:00Z", "completed_at": None, "blocked_reason": None,
        "task_revision_before": None, "task_revision_after": None, "result": None}})

    repair_metadata(cfg)

    stored = load_task(cfg, task.task_id)
    assert stored.placement_policy["sharing_mode"] == "spillover"
    assert stored.placement_runtime["queue_scope"] == "shared"
    assert read_json(operation_path)["availability_operation"]["state"] == "completed"


def test_doctor_archives_blocked_availability_operation_with_reason(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"], group="exp")
    operation_id = "blocked-share"
    operation_path = active_operation_path(cfg, "availability", operation_id)
    write_active_operation(cfg, "availability", operation_id, {"meta": {"schema_version": 6, "revision": 1,
        "created_at": "2026-08-06T00:00:00Z", "updated_at": "2026-08-06T00:00:00Z",
        "updated_by": {"actor_type": "cli", "machine_name": "g1", "process_id": "test"}},
        "availability_operation": {"operation_id": operation_id, "operation_type": "share_now",
        "task_id": task.task_id, "state": "blocked", "requested_by": "g1", "reason": "manual",
        "helper_machines": None, "after_seconds": None, "created_at": "2026-08-06T00:00:00Z",
        "updated_at": "2026-08-06T00:00:00Z", "completed_at": None,
        "blocked_reason": "placement can only change while a Task is queued and unclaimed.",
        "task_revision_before": None, "task_revision_after": None, "result": None}})

    archived_path = shared_paths(cfg.shared_root)["availability"] / f"{operation_id}.json"
    assert archived_path.is_symlink()

    repair_metadata(cfg)

    assert not operation_path.exists()
    assert archived_path.exists()
    assert not archived_path.is_symlink()
    assert read_json(archived_path)["availability_operation"]["state"] == "blocked"


def test_doctor_completes_operation_after_post_task_side_effect_failure(
        tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"], group="exp")
    original = availability_runtime.sync_deadline_index
    calls = 0

    def fail_once(cfg, task):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("index unavailable")
        return original(cfg, task)

    monkeypatch.setattr(availability_runtime, "sync_deadline_index", fail_once)
    with pytest.raises(OSError, match="index unavailable"):
        task_commands.share(cfg, task.task_id)

    stored = load_task(cfg, task.task_id)
    operation_id = stored.placement_runtime["availability_operation_id"]
    operation_path = shared_paths(cfg.shared_root)["availability"] / f"{operation_id}.json"
    assert read_json(operation_path)["availability_operation"]["state"] == "prepared"

    repair_metadata(cfg)

    assert read_json(operation_path)["availability_operation"]["state"] == "completed"


def test_share_rejects_claimed_task_without_mutation(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"], group="exp")
    assert claim_task(cfg, task.task_id, [0]) is not None
    revision = load_task(cfg, task.task_id).meta["revision"]

    with pytest.raises(ValueError, match="queued and unclaimed"):
        task_commands.share(cfg, task.task_id)

    stored = load_task(cfg, task.task_id)
    assert stored.meta["revision"] == revision
    assert stored.placement_policy["sharing_mode"] == "private"


def test_failed_availability_operation_does_not_make_doctor_unhealthy(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])

    with pytest.raises(ValueError, match="does not belong to a Group"):
        task_commands.share(cfg, task.task_id)

    assert verify_integrity(cfg)["healthy"] is True


def test_cli_availability_json_and_human_outputs(tmp_path: Path, monkeypatch, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"], group="exp")
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.ensure_local_agent_active",
        lambda cfg, *, reason: True,
    )

    assert main([*_base_args(cfg), "task", "share", task.task_id, "--format=json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["action"] == "share_now"
    assert payload["task_id"] == task.task_id
    assert payload["resulting_state"] == "shared"

    assert main([*_base_args(cfg), "task", "keep-local", task.task_id]) == 0
    assert "restricted to its home machine" in capsys.readouterr().out


def test_cli_share_accepts_comma_separated_helper_machines(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"], group="exp")
    change_worker(cfg, "exp", "g2", "add")
    change_worker(cfg, "exp", "g3", "add")
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.ensure_local_agent_active",
        lambda cfg, *, reason: True,
    )

    assert main([
        *_base_args(cfg), "task", "share", task.task_id, "--with", "g2,g3",
    ]) == 0

    assert load_task(cfg, task.task_id).placement_policy["fallback_constraint"] == ["g2", "g3"]


def test_cli_share_rejects_empty_comma_separated_helper_machine():
    with pytest.raises(ValueError, match="non-empty"):
        _split_machine_list(["g2,,g3"])


def test_rebuild_deadline_indexes_removes_stale_index(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    stale = shared_paths(cfg.shared_root)["offer_deadlines"] / "missing.json"
    atomic_replace(stale, {"offer_deadline": {"task_id": "missing"}})

    assert rebuild_deadline_indexes(cfg) == 1
    assert not stale.exists()


def test_agent_removes_stale_deadline_index_without_skipping_remaining_work(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    stale = shared_paths(cfg.shared_root)["offer_deadlines"] / "missing.json"
    atomic_replace(stale, {"offer_deadline": {"task_id": "missing"}})
    task = submit(cfg, ["echo", "ok"], group="exp")
    task_commands.share(cfg, task.task_id, after_seconds=0)

    offer_due_tasks(cfg)

    assert not stale.exists()
    assert load_task(cfg, task.task_id).placement_runtime["queue_scope"] == "shared"
