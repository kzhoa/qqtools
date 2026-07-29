import multiprocessing
import time
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.commands.cleanup import clean, reconcile_cleanup_operations
from qqtools.plugins.qexp.commands.group import (group_control,
                                                reconcile_group_cancel_operations)
from qqtools.plugins.qexp.commands.task import offer, retry
from qqtools.plugins.qexp.doctor import repair_metadata, repair_orphans, verify_integrity
from qqtools.plugins.qexp.config_types import RootConfig
from qqtools.plugins.qexp.runtime.locks import group_lock, task_lock
from qqtools.plugins.qexp.runtime.paths import attempt_path, group_path, task_path
from qqtools.plugins.qexp.runtime.reservations import attach, reserve, reserved_gpu_ids
from qqtools.plugins.qexp.runtime.records import new_id, utc_now
from qqtools.plugins.qexp.runtime.claims import reconcile_claim_archives
from qqtools.plugins.qexp.runtime import submission as submission_runtime
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task
from qqtools.plugins.qexp.scheduler import (authorize_launch, claim_task, expire_claim,
                                             cancel_task, fail_attempt,
                                             reconcile_running_tasks)


def _run_cleanup_reconcile(
        shared_root: str, project_root: str, runtime_root: str, machine_name: str,
        connection) -> None:
    cfg = RootConfig(Path(shared_root), Path(project_root), machine_name, Path(runtime_root))
    try:
        connection.send({"ok": True, "result": reconcile_cleanup_operations(cfg)})
    except BaseException as exc:
        connection.send({"ok": False, "type": type(exc).__name__, "message": str(exc)})
    finally:
        connection.close()


def _recv_cleanup_reconcile_result(process, parent_connection, *, timeout: float = 20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if parent_connection.poll(0.1):
            payload = parent_connection.recv()
            process.join(5)
            return payload
        if process.exitcode is not None:
            break
    if process.is_alive():
        process.terminate()
        process.join(5)
    raise AssertionError("cleanup reconciliation subprocess did not report before timeout")


def _failed_task(cfg, command: str = "failed"):
    task = submit(cfg, ["echo", command])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert fail_attempt(
        cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "test_failure"
    )
    return task


def test_clean_exact_terminal_task_supports_dry_run_and_audit(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = _failed_task(cfg)
    preview = clean(cfg, task_id=task.task_id, dry_run=True)
    assert preview["candidates"] == [task.task_id]
    assert preview["removed"] == []
    assert task_path(cfg.shared_root, task.task_id).exists()
    result = clean(cfg, task_id=task.task_id)
    assert result["removed"]
    assert not task_path(cfg.shared_root, task.task_id).exists()
    assert not (cfg.shared_root / "attempts" / task.task_id).exists()
    cleanup_operation = read_json(
        cfg.shared_root / "operations" / "cleanup" / f"{task.task_id}.json"
    )["cleanup"]
    assert cleanup_operation["state"] == "completed"
    events = [read_json(path) for path in (cfg.shared_root / "events").glob("*/*.json")]
    assert any(event.get("event_type") == "task_cleaned" for event in events)
    assert verify_integrity(cfg)["healthy"] is True


def test_clean_bulk_is_bounded_by_retention_and_limit(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    tasks = [_failed_task(cfg, str(index)) for index in range(2)]
    for task in tasks:
        path = task_path(cfg.shared_root, task.task_id)
        data = read_json(path)
        data["meta"]["updated_at"] = "2000-01-01T00:00:00Z"
        atomic_replace(path, data)
    preview = clean(cfg, older_than_days=30, limit=1, dry_run=True)
    assert len(preview["candidates"]) == 1
    assert preview["removed"] == []


def test_clean_releases_proven_local_reservation_before_deleting_truth(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = _failed_task(cfg)
    reservation = reserve(cfg.runtime_root, task.task_id, [3])
    reservation_id = reservation["reservation"]["reservation_id"]
    attach(cfg.runtime_root, reservation_id, "stale-attempt", 99)

    result = clean(cfg, task_id=task.task_id)

    assert result["operations"][task.task_id]["state"] == "completed"
    assert reserved_gpu_ids(cfg.runtime_root) == set()
    assert not task_path(cfg.shared_root, task.task_id).exists()


def test_clean_waits_for_remote_machine_local_cleanup(tmp_path: Path):
    shared_root = tmp_path / ".qexp"
    cfg1 = init_shared_root(shared_root, "gpu-1", runtime_root=tmp_path / "rt-1")
    cfg2 = init_shared_root(shared_root, "gpu-2", runtime_root=tmp_path / "rt-2")
    task = _failed_task(cfg1)
    manifest = cfg1.runtime_root / "processes" / "old-attempt.json"
    atomic_replace(manifest, {"process": {"task_id": task.task_id,
                                           "attempt_id": "old-attempt",
                                           "observed_state": "exited"}})
    log = shared_root / "logs" / task.task_id / "old-attempt.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("finished", encoding="utf-8")

    result = clean(cfg2, task_id=task.task_id)

    assert result["operations"][task.task_id]["state"] == "waiting_ack"
    assert result["operations"][task.task_id]["pending_machines"] == ["gpu-1"]
    assert task_path(shared_root, task.task_id).exists()
    assert manifest.exists()
    reconcile_cleanup_operations(cfg1)
    assert not task_path(shared_root, task.task_id).exists()
    assert not manifest.exists()
    assert not log.exists()


def test_cleanup_waiting_ack_blocks_retry_and_late_attempt_deletion(tmp_path: Path):
    shared_root = tmp_path / ".qexp"
    cfg1 = init_shared_root(shared_root, "gpu-1", runtime_root=tmp_path / "rt-1")
    cfg2 = init_shared_root(shared_root, "gpu-2", runtime_root=tmp_path / "rt-2")
    task = _failed_task(cfg1)
    result = clean(cfg2, task_id=task.task_id)
    assert result["operations"][task.task_id]["state"] == "waiting_ack"
    blocked_task = load_task(cfg2, task.task_id)
    assert blocked_task.control["cleanup_operation_id"]
    assert blocked_task.control["cleanup_state"] == "waiting_ack"

    with pytest.raises(ValueError, match="being cleaned"):
        retry(cfg2, task.task_id)
    assert claim_task(cfg1, task.task_id, [0]) is None

    reconcile_cleanup_operations(cfg1)

    assert not task_path(shared_root, task.task_id).exists()
    assert not (shared_root / "attempts" / task.task_id).exists()


def test_cleanup_waiting_ack_blocks_cancel_offer_and_claim_even_if_task_is_queued(
        tmp_path: Path):
    shared_root = tmp_path / ".qexp"
    cfg1 = init_shared_root(shared_root, "gpu-1", runtime_root=tmp_path / "rt-1")
    cfg2 = init_shared_root(shared_root, "gpu-2", runtime_root=tmp_path / "rt-2")
    task = submit(cfg1, ["echo", "ok"], group="exp", sharing_mode="spillover")
    attempt = claim_task(cfg1, task.task_id, [0])
    assert attempt is not None
    assert fail_attempt(
        cfg1, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "test_failure"
    )
    clean(cfg2, task_id=task.task_id)
    task_data = read_json(task_path(shared_root, task.task_id))
    task_data["task"]["state"] = {"projection": "queued", "reason": None}
    task_data["task"]["placement_runtime"]["queue_scope"] = "home"
    atomic_replace(task_path(shared_root, task.task_id), task_data)

    with pytest.raises(ValueError, match="being cleaned"):
        cancel_task(cfg1, task.task_id)
    with pytest.raises(ValueError, match="being cleaned"):
        offer(cfg1, task.task_id)
    assert claim_task(cfg1, task.task_id, [0]) is None


def test_cleaned_task_id_cannot_be_submitted_again(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = _failed_task(cfg)
    clean(cfg, task_id=task.task_id)

    with pytest.raises(ValueError, match="cannot be reused"):
        submit(cfg, ["echo", "new"], task_id=task.task_id)


def test_submission_rechecks_cleanup_tombstone_before_task_creation(
        tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task_id = "race-task"
    original = submission_runtime._reject_cleanup_tombstones
    calls = 0

    def inject_tombstone(cfg_value, resolved):
        nonlocal calls
        calls += 1
        if calls == 2:
            now = utc_now()
            atomic_replace(
                cfg.shared_root / "operations" / "cleanup" / f"{task_id}.json",
                {"meta": {"schema_version": 5, "revision": 1, "created_at": now,
                          "updated_at": now, "updated_by": {"actor_type": "test",
                          "machine_name": cfg.machine_name, "process_id": "0"}},
                 "cleanup": {"operation_id": new_id(), "task_id": task_id,
                             "state": "completed", "group_name": None,
                             "submission_operation_id": None,
                             "terminal_state": "failed", "created_at": now,
                             "required_machines": [cfg.machine_name],
                             "acknowledgements": {cfg.machine_name:
                                 {"acknowledged_at": now, "removed": []}},
                             "pending_machines": [], "completed_at": now}},
            )
        original(cfg_value, resolved)

    monkeypatch.setattr(submission_runtime, "_reject_cleanup_tombstones", inject_tombstone)

    with pytest.raises(ValueError, match="cannot be reused"):
        submit(cfg, ["echo", "new"], task_id=task_id)

    assert not task_path(cfg.shared_root, task_id).exists()
    tombstone = read_json(
        cfg.shared_root / "operations" / "cleanup" / f"{task_id}.json"
    )["cleanup"]
    assert tombstone["state"] == "completed"


def test_cleanup_required_machines_excludes_unrelated_registered_workers(tmp_path: Path):
    shared_root = tmp_path / ".qexp"
    cfg1 = init_shared_root(shared_root, "gpu-1", runtime_root=tmp_path / "rt-1")
    cfg2 = init_shared_root(shared_root, "gpu-2", runtime_root=tmp_path / "rt-2")
    init_shared_root(shared_root, "gpu-3", runtime_root=tmp_path / "rt-3")
    task = _failed_task(cfg1)

    result = clean(cfg2, task_id=task.task_id)

    cleanup = result["operations"][task.task_id]
    assert cleanup["state"] == "waiting_ack"
    assert cleanup["pending_machines"] == ["gpu-1"]
    operation = read_json(shared_root / "operations" / "cleanup" / f"{task.task_id}.json")
    assert operation["cleanup"]["required_machines"] == ["gpu-1", "gpu-2"]


def test_public_cleanup_finalization_does_not_block_on_group_lock(tmp_path: Path):
    shared_root = tmp_path / ".qexp"
    cfg1 = init_shared_root(shared_root, "gpu-1", runtime_root=tmp_path / "rt-1")
    cfg2 = init_shared_root(shared_root, "gpu-2", runtime_root=tmp_path / "rt-2")
    task = submit(cfg1, ["echo", "ok"], group="exp")
    attempt = claim_task(cfg1, task.task_id, [0])
    assert attempt is not None
    assert fail_attempt(
        cfg1, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "test_failure"
    )
    clean(cfg2, task_id=task.task_id)
    cleanup_path = shared_root / "operations" / "cleanup" / f"{task.task_id}.json"
    assert read_json(cleanup_path)["cleanup"]["state"] == "waiting_ack"
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)

    with group_lock(shared_root, "exp"):
        process = context.Process(
            target=_run_cleanup_reconcile,
            args=(str(cfg1.shared_root), str(cfg1.project_root), str(cfg1.runtime_root),
                  cfg1.machine_name, child_connection),
        )
        process.start()
        child_connection.close()
        assert _recv_cleanup_reconcile_result(process, parent_connection)["ok"] is True
        assert read_json(cleanup_path)["cleanup"]["state"] == "waiting_ack"
        assert task_path(shared_root, task.task_id).exists()

    try:
        assert reconcile_cleanup_operations(cfg1)[0]["state"] == "completed"
    finally:
        parent_connection.close()
        if process.is_alive():
            process.terminate()
            process.join(5)

    assert process.exitcode == 0
    assert read_json(cleanup_path)["cleanup"]["state"] == "completed"
    assert not task_path(shared_root, task.task_id).exists()


def test_public_cleanup_ack_does_not_block_on_task_lock(tmp_path: Path):
    shared_root = tmp_path / ".qexp"
    cfg1 = init_shared_root(shared_root, "gpu-1", runtime_root=tmp_path / "rt-1")
    cfg2 = init_shared_root(shared_root, "gpu-2", runtime_root=tmp_path / "rt-2")
    task = _failed_task(cfg1)
    clean(cfg2, task_id=task.task_id)
    cleanup_path = shared_root / "operations" / "cleanup" / f"{task.task_id}.json"
    assert read_json(cleanup_path)["cleanup"]["state"] == "waiting_ack"
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)

    with task_lock(shared_root, task.task_id):
        process = context.Process(
            target=_run_cleanup_reconcile,
            args=(str(cfg1.shared_root), str(cfg1.project_root), str(cfg1.runtime_root),
                  cfg1.machine_name, child_connection),
        )
        process.start()
        child_connection.close()
        payload = _recv_cleanup_reconcile_result(process, parent_connection)
        assert payload["ok"] is True
        assert payload["result"][0]["blockers"] == ["task_lock_busy"]

    parent_connection.close()
    assert reconcile_cleanup_operations(cfg1)[0]["state"] == "completed"


def test_cleaned_event_is_written_only_after_shared_deletion_succeeds(
        tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = _failed_task(cfg)
    target = task_path(cfg.shared_root, task.task_id)
    original_unlink = Path.unlink

    def fail_task_unlink(path: Path, *args, **kwargs):
        if path == target:
            raise OSError("injected deletion failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_task_unlink)
    with pytest.raises(OSError, match="injected deletion failure"):
        clean(cfg, task_id=task.task_id)

    events = [read_json(path) for path in (cfg.shared_root / "events").glob("*/*.json")]
    assert any(event.get("event_type") == "task_cleanup_started" for event in events)
    assert not any(event.get("event_type") == "task_cleaned" for event in events)


def test_doctor_finishes_interrupted_cleanup_operation(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = _failed_task(cfg)
    number = load_task(cfg, task.task_id).attempt_control["current_attempt_number"]
    attempt_file = attempt_path(cfg.shared_root, task.task_id, number)
    attempt_data = read_json(attempt_file)
    clean(cfg, task_id=task.task_id)
    atomic_replace(attempt_file, attempt_data)
    cleanup_path = cfg.shared_root / "operations" / "cleanup" / f"{task.task_id}.json"
    cleanup_data = read_json(cleanup_path)
    cleanup_data["cleanup"]["state"] = "preparing"
    cleanup_data["cleanup"]["completed_at"] = None
    atomic_replace(cleanup_path, cleanup_data)
    codes = {issue["code"] for issue in verify_integrity(cfg)["issues"]}
    assert "cleanup_operation_incomplete" in codes
    assert "cleaned_task_attempt_residual" in codes
    repair_metadata(cfg)
    assert not attempt_file.exists()
    assert read_json(cleanup_path)["cleanup"]["state"] == "completed"


def test_clean_rejects_nonterminal_task(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "queued"])
    with pytest.raises(ValueError, match="task_state:queued"):
        clean(cfg, task_id=task.task_id)


def test_group_prelaunch_cancellation_archives_claim(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"], group="exp")
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None

    group_control(cfg, "exp", "cancel")

    archive = read_json(
        cfg.shared_root / "claims" / "archive" / task.task_id / f"{attempt.current_fencing_token}.json"
    )["claim_archive"]
    assert archive["reason"] == "group_cancelled_before_launch"
    assert archive["claim"]["attempt_id"] == attempt.attempt_id


def test_claim_archive_io_failure_does_not_block_terminal_task(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    from qqtools.plugins.qexp.runtime import claims
    original_create_if_absent = claims.create_if_absent

    def fail_archive_once(path, value):
        if path.parent.parent.name == "archive":
            raise OSError("archive unavailable")
        original_create_if_absent(path, value)

    monkeypatch.setattr(claims, "create_if_absent", fail_archive_once)

    assert fail_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "test_failure")
    stored = load_task(cfg, task.task_id)
    assert stored.state["projection"] == "failed"
    assert stored.claim_control["active_claim"] is None
    pending = cfg.shared_root / "claims" / "pending" / task.task_id / "1.json"
    assert pending.exists()

    monkeypatch.setattr(claims, "create_if_absent", original_create_if_absent)
    assert reconcile_claim_archives(cfg, task.task_id)
    assert not pending.exists()
    assert (cfg.shared_root / "claims" / "archive" / task.task_id / "1.json").exists()


def test_missing_group_barrier_remains_blocked_during_reconciliation(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    submit(cfg, ["echo", "ok"], group="exp")
    group = group_control(cfg, "exp", "cancel")
    operation_id = group["cancellation_operation"]["operation_id"]
    path = group_path(cfg.shared_root, "exp")
    data = read_json(path)
    data["group"]["cancellation_barriers"] = []
    atomic_replace(path, data)
    operation_path = cfg.shared_root / "operations" / "group-control" / f"{operation_id}.json"
    operation = read_json(operation_path)
    operation["group_control"]["state"] = "blocked"
    operation["group_control"]["blocked_reason"] = "cancellation_barrier_missing"
    atomic_replace(operation_path, operation)
    reconcile_group_cancel_operations(cfg)
    control = read_json(operation_path)["group_control"]
    assert control["state"] == "blocked"
    assert control["blocked_reason"] == "cancellation_barrier_missing"
    task_id = next(iter((cfg.shared_root / "tasks").glob("*.json"))).stem
    preview = clean(cfg, task_id=task_id, dry_run=True)
    assert preview["skipped"][task_id] == [f"group_control_barrier_missing:{operation_id}"]


def test_doctor_detects_claim_attempt_token_mismatch(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    path = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    data = read_json(path)
    data["attempt"]["current_fencing_token"] += 1
    atomic_replace(path, data)
    codes = {issue["code"] for issue in verify_integrity(cfg)["issues"]}
    assert "claim_attempt_token_mismatch" in codes


def test_doctor_reports_corrupt_cross_truth_without_crashing(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    (cfg.shared_root / "groups" / "broken.json").write_text("{", encoding="utf-8")
    codes = {issue["code"] for issue in verify_integrity(cfg)["issues"]}
    assert "group_invalid" in codes


def test_repair_orphan_keeps_blocked_on_identity_mismatch(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    path = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    attempt_data = read_json(path)
    attempt_data["attempt"]["process"].update({"process_group_id": 111,
        "process_group_start_time_ticks": 10})
    atomic_replace(path, attempt_data)
    manifest_path = cfg.runtime_root / "processes" / f"{attempt.attempt_id}.json"
    atomic_replace(manifest_path, {"process": {"task_id": task.task_id,
        "attempt_id": attempt.attempt_id, "fencing_token": attempt.current_fencing_token,
        "process_group_id": 222, "process_group_start_time_ticks": 20}})
    assert expire_claim(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    reconcile_running_tasks(cfg)
    assert load_task(cfg, task.task_id).state["projection"] == "blocked"
    assert reserved_gpu_ids(cfg.runtime_root) == {0}
    result = repair_orphans(cfg)
    assert result["blocked"] == [{"task_id": task.task_id,
                                  "reason": "process_identity_mismatch"}]
    assert load_task(cfg, task.task_id).state["projection"] == "blocked"
    assert reserved_gpu_ids(cfg.runtime_root) == {0}


def test_repair_orphan_finalizes_identity_matched_absent_process(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    path = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    attempt_data = read_json(path)
    attempt_data["attempt"]["process"].update({"process_group_id": 111,
        "process_group_start_time_ticks": 10})
    atomic_replace(path, attempt_data)
    manifest_path = cfg.runtime_root / "processes" / f"{attempt.attempt_id}.json"
    atomic_replace(manifest_path, {"process": {"task_id": task.task_id,
        "attempt_id": attempt.attempt_id, "fencing_token": attempt.current_fencing_token,
        "process_group_id": 111, "process_group_start_time_ticks": 10,
        "exit_code": None}})
    assert expire_claim(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    monkeypatch.setattr("qqtools.plugins.qexp.scheduler._process_start_time_ticks",
                        lambda pid: None)
    monkeypatch.setattr("qqtools.plugins.qexp.scheduler._is_process_group_alive",
                        lambda pid: False)
    result = repair_orphans(cfg)
    assert result["repaired"] == [task.task_id]
    assert load_task(cfg, task.task_id).state["projection"] == "failed"
    assert reserved_gpu_ids(cfg.runtime_root) == set()
