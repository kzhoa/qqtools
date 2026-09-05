from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.authority import AuthoritySupervisor
from qqtools.plugins.qexp.commands.group import (
    change_worker,
    create_group,
    group_control,
    reconcile_group_cancel_operations,
    show_group,
)
from qqtools.plugins.qexp.doctor import repair_metadata
from qqtools.plugins.qexp.project_maintenance import offer_due_tasks, reconcile_project_reservations
from qqtools.plugins.qexp.runner import run_attempt
from qqtools.plugins.qexp.runtime.active_operations import active_operation_path, write_active_operation
from qqtools.plugins.qexp.runtime.paths import attempt_path
from qqtools.plugins.qexp.runtime.records import AttemptRecord
from qqtools.plugins.qexp.runtime.recovery import recover_running_attempt
from qqtools.plugins.qexp.runtime.reservations import reserve, reserved_gpu_ids
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task
from qqtools.plugins.qexp.scheduler import (
    authorize_launch,
    cancel_task,
    claim_task,
    expire_claim,
    reconcile_running_tasks,
    run_dispatch_cycle,
)

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _existing_group(cfg) -> None:
    create_group(cfg, "exp")

class RecordingExecutor:
    def __init__(self):
        self.attempts = []

    def launch_attempt(self, cfg, task_id, attempt):
        self.attempts.append(attempt)


class FakeChild:
    pid = 9876
    returncode = 0

    def poll(self):
        return 0

    def wait(self):
        return 0


def test_scheduler_commits_launch_gate_before_starting_runner(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    executor = RecordingExecutor()
    run_dispatch_cycle(cfg, available_gpus=[0], executor=executor)
    stored = load_task(cfg, task.task_id)
    assert stored.claim_control["active_claim"]["launch_state"] == "starting"
    assert executor.attempts
    attempt = executor.attempts[0]
    launch_id = read_json(attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number))["attempt"][
        "authorization"
    ]["launch_id"]
    assert (
        run_attempt(
            cfg,
            task.task_id,
            attempt.attempt_id,
            attempt.current_fencing_token,
            launch_id,
            popen_factory=lambda *args, **kwargs: FakeChild(),
        )
        == 0
    )
    AuthoritySupervisor(cfg).tick()
    assert load_task(cfg, task.task_id).state["projection"] == "succeeded"


def test_dispatch_resumes_starting_attempt_after_authorization_write_crash(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    path = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    original_atomic_replace = atomic_replace

    def crash_before_attempt_write(target, value):
        if target == path:
            raise SystemExit("simulated crash after Task launch authorization")
        original_atomic_replace(target, value)

    monkeypatch.setattr("qqtools.plugins.qexp.scheduler.atomic_replace", crash_before_attempt_write)
    with pytest.raises(SystemExit):
        authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)

    stranded = load_task(cfg, task.task_id)
    stale_launch_id = stranded.claim_control["active_claim"]["launch_id"]
    assert stranded.claim_control["active_claim"]["launch_state"] == "starting"
    assert AttemptRecord.from_dict(read_json(path)).phase == "claimed"
    monkeypatch.setattr("qqtools.plugins.qexp.scheduler.atomic_replace", original_atomic_replace)

    executor = RecordingExecutor()
    assert run_dispatch_cycle(cfg, available_gpus=[0], executor=executor) == [task.task_id]
    resumed = executor.attempts[0]
    stored = AttemptRecord.from_dict(read_json(path))
    assert resumed.attempt_id == attempt.attempt_id
    assert resumed.attempt_number == attempt.attempt_number
    assert resumed.reservation_id == attempt.reservation_id
    assert resumed.current_fencing_token == attempt.current_fencing_token
    assert stored.phase == "starting"
    assert stored.authorization["launch_id"] != stale_launch_id
    assert load_task(cfg, task.task_id).claim_control["fencing_epoch"] == attempt.current_fencing_token

    with pytest.raises(RuntimeError, match="not authorized"):
        run_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, stale_launch_id)
    assert not (cfg.runtime_root / "launch-intents" / f"{attempt.attempt_id}.json").exists()


def test_cancelling_stranded_starting_attempt_prevents_relaunch(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    path = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    original_atomic_replace = atomic_replace

    def crash_before_attempt_write(target, value):
        if target == path:
            raise SystemExit("simulated crash after Task launch authorization")
        original_atomic_replace(target, value)

    monkeypatch.setattr("qqtools.plugins.qexp.scheduler.atomic_replace", crash_before_attempt_write)
    with pytest.raises(SystemExit):
        authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    monkeypatch.setattr("qqtools.plugins.qexp.scheduler.atomic_replace", original_atomic_replace)

    cancel_task(cfg, task.task_id)
    executor = RecordingExecutor()
    assert run_dispatch_cycle(cfg, available_gpus=[0], executor=executor) == []
    assert executor.attempts == []
    assert load_task(cfg, task.task_id).state["projection"] == "cancelled"
    assert AttemptRecord.from_dict(read_json(path)).phase == "cancelled"
    assert reserved_gpu_ids(cfg.runtime_root) == set()


def test_starting_attempt_with_launch_evidence_is_not_resumed(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    intent = cfg.runtime_root / "launch-intents" / f"{attempt.attempt_id}.json"
    atomic_replace(intent, {"launch_intent": {"attempt_id": attempt.attempt_id}})

    executor = RecordingExecutor()
    assert run_dispatch_cycle(cfg, available_gpus=[0], executor=executor) == []
    assert executor.attempts == []


def test_expired_provisional_is_reclaimed_before_next_reservation(tmp_path: Path):
    runtime = tmp_path / "rt"
    first = reserve(runtime, "one", [0])
    path = runtime / "reservations" / "provisional" / f"{first['reservation']['reservation_id']}.json"
    data = read_json(path)
    data["reservation"]["expires_at"] = "2000-01-01T00:00:00Z"
    atomic_replace(path, data)
    second = reserve(runtime, "two", [0])
    assert second["reservation"]["task_id"] == "two"


def test_recovery_cas_issues_new_fencing_token(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    task = submit(cfg, ["echo", "ok"], group="exp", sharing_mode="spillover")
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    manifest_path = cfg.runtime_root / "processes" / f"{attempt.attempt_id}.json"
    atomic_replace(
        manifest_path,
        {
            "process": {
                "task_id": task.task_id,
                "attempt_id": attempt.attempt_id,
                "fencing_token": attempt.current_fencing_token,
                "process_group_id": 9876,
                "observed_state": "running",
            }
        },
    )
    assert expire_claim(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    token = recover_running_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert token == attempt.current_fencing_token + 1
    assert read_json(manifest_path)["process"]["fencing_token"] == token
    recovered = AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)))
    assert recovered.phase == "running"
    assert recovered.timestamps["orphaned_at"] is not None
    assert recovered.timestamps["recovered_at"] is not None
    assert recovered.timestamps["finished_at"] is None
    assert recovered.result["reason"] is None
    reconcile_project_reservations(cfg)
    assert reserved_gpu_ids(cfg.runtime_root) == {0}


def test_blocked_orphan_with_missing_process_finalizes_and_releases_gpu(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    task = submit(cfg, ["echo", "ok"], group="exp", sharing_mode="spillover")
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    manifest_path = cfg.runtime_root / "processes" / f"{attempt.attempt_id}.json"
    atomic_replace(
        manifest_path,
        {
            "process": {
                "task_id": task.task_id,
                "attempt_id": attempt.attempt_id,
                "fencing_token": attempt.current_fencing_token,
                "process_group_id": 9876,
                "observed_state": "running",
                "exit_code": None,
            }
        },
    )
    assert expire_claim(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    orphaned = AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)))
    assert orphaned.phase == "orphaned"
    assert orphaned.timestamps["orphaned_at"] is not None
    assert orphaned.timestamps["finished_at"] is None
    monkeypatch.setattr("qqtools.plugins.qexp.scheduler._process_evidence_state", lambda *args: "absent")
    reconcile_running_tasks(cfg)
    stored = load_task(cfg, task.task_id)
    assert stored.state == {"projection": "failed", "reason": "process_exited_without_status"}
    assert reserved_gpu_ids(cfg.runtime_root) == set()


def test_partial_recovery_finalize_preserves_monotonic_fencing_epoch(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    task = submit(cfg, ["echo", "ok"], group="exp", sharing_mode="spillover")
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    manifest_path = cfg.runtime_root / "processes" / f"{attempt.attempt_id}.json"
    atomic_replace(
        manifest_path,
        {
            "process": {
                "task_id": task.task_id,
                "attempt_id": attempt.attempt_id,
                "fencing_token": attempt.current_fencing_token,
                "process_group_id": 9876,
                "observed_state": "running",
                "exit_code": None,
            }
        },
    )
    assert expire_claim(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    path = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    attempt_data = read_json(path)
    recovered_token = attempt.current_fencing_token + 1
    attempt_data["attempt"]["phase"] = "running"
    attempt_data["attempt"]["current_fencing_token"] = recovered_token
    attempt_data["attempt"]["token_history"].append(recovered_token)
    atomic_replace(path, attempt_data)
    monkeypatch.setattr("qqtools.plugins.qexp.scheduler._process_evidence_state", lambda *args: "absent")
    reconcile_running_tasks(cfg)
    stored = load_task(cfg, task.task_id)
    assert stored.state["projection"] == "failed"
    assert stored.claim_control["fencing_epoch"] == recovered_token


def test_reconcile_finishes_recovery_when_manifest_write_was_interrupted(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    task = submit(cfg, ["echo", "ok"], group="exp", sharing_mode="spillover")
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    manifest_path = cfg.runtime_root / "processes" / f"{attempt.attempt_id}.json"
    atomic_replace(
        manifest_path,
        {
            "process": {
                "task_id": task.task_id,
                "attempt_id": attempt.attempt_id,
                "fencing_token": attempt.current_fencing_token,
                "process_group_id": 9876,
                "observed_state": "running",
            }
        },
    )
    assert expire_claim(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    token = recover_running_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    manifest = read_json(manifest_path)
    manifest["process"]["fencing_token"] = attempt.current_fencing_token
    atomic_replace(manifest_path, manifest)
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr("qqtools.plugins.qexp.scheduler._process_evidence_state", lambda *args: "alive")
    monkeypatch.setattr("qqtools.plugins.qexp.scheduler.os.killpg", lambda pid, sig: calls.append((pid, sig)))
    reconcile_running_tasks(cfg)
    repaired_manifest = read_json(manifest_path)["process"]
    assert repaired_manifest["fencing_token"] == token
    assert repaired_manifest["supervisor"] == "agent"
    assert calls == []


def test_agent_supervises_recovered_child_without_runner(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    task = submit(cfg, ["echo", "ok"], group="exp", sharing_mode="spillover")
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    manifest_path = cfg.runtime_root / "processes" / f"{attempt.attempt_id}.json"
    atomic_replace(
        manifest_path,
        {
            "process": {
                "task_id": task.task_id,
                "attempt_id": attempt.attempt_id,
                "fencing_token": attempt.current_fencing_token,
                "process_group_id": 9876,
                "observed_state": "running",
            }
        },
    )
    assert expire_claim(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert recover_running_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    renewed: list[int] = []
    monkeypatch.setattr("qqtools.plugins.qexp.scheduler._process_evidence_state", lambda *args: "alive")
    monkeypatch.setattr(
        "qqtools.plugins.qexp.scheduler.renew_attempt_lease", lambda *args: renewed.append(args[3]) or True
    )
    reconcile_running_tasks(cfg)
    assert renewed == [load_task(cfg, task.task_id).claim_control["active_claim"]["fencing_token"]]


def test_elapsed_offer_is_applied_by_home_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    task = submit(cfg, ["echo", "ok"], group="exp", sharing_mode="spillover", offer_after_seconds=0)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.project_maintenance.elapsed_offer_is_proven",
        lambda *_args: True,
    )
    offer_due_tasks(cfg)
    assert load_task(cfg, task.task_id).placement_runtime["queue_scope"] == "shared"


def test_worker_removal_drains_before_reporting_home_queue_blocker(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    submit(cfg, ["echo", "ok"], group="exp")
    group = change_worker(cfg, "exp", "g1", "remove")
    assert group["group"]["worker_set"]["g1"]["state"] == "draining"
    assert group["worker_control"]["state"] == "waiting_ack"


def test_worker_removal_operation_completes_after_blocker_clears(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    task = submit(cfg, ["echo", "ok"], group="exp")
    group = change_worker(cfg, "exp", "g1", "remove")
    operation_id = group["worker_control"]["operation_id"]
    active_path = active_operation_path(cfg, "group_control", operation_id)
    stable_path = cfg.shared_root / "operations" / "group-control" / f"{operation_id}.json"
    assert active_path.exists()
    assert stable_path.is_symlink()

    cancel_task(cfg, task.task_id, terminate_running=False)
    reconciled = reconcile_group_cancel_operations(cfg)

    assert any(item["operation_id"] == operation_id and item["state"] == "completed"
               for item in reconciled)
    assert not active_path.exists()
    assert stable_path.exists()
    assert not stable_path.is_symlink()
    assert read_json(stable_path)["group_control"]["state"] == "completed"
    assert show_group(cfg, "exp")["group"]["worker_set"]["g1"]["state"] == "removing"


def test_missing_group_barrier_blocked_operation_is_archived(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    submit(cfg, ["echo", "ok"], group="exp")
    operation_id = "missing-barrier"
    active_path = active_operation_path(cfg, "group_control", operation_id)
    write_active_operation(cfg, "group_control", operation_id, {"meta": {
        "schema_version": 6, "revision": 1, "created_at": "2026-08-06T00:00:00Z",
        "updated_at": "2026-08-06T00:00:00Z",
        "updated_by": {"actor_type": "cli", "machine_name": "g1", "process_id": "test"}},
        "group_control": {"operation_id": operation_id, "operation_type": "cancel",
        "group_name": "exp", "state": "converging", "group_revision_at_start": 1,
        "dispatch_epoch_at_start": 0, "membership_high_watermark": 1,
        "terminate_running": False, "progress": {"target_tasks": 0,
        "already_terminal": 0, "queued_cancelled": 0, "prelaunch_cancelled": 0,
        "running_allowed": 0, "termination_pending": 0, "termination_acknowledged": 0,
        "blocked": 0}, "pending_machine_acknowledgements": {},
        "created_at": "2026-08-06T00:00:00Z", "updated_at": "2026-08-06T00:00:00Z",
        "completed_at": None, "blocked_reason": None}})
    stable_path = cfg.shared_root / "operations" / "group-control" / f"{operation_id}.json"
    assert stable_path.is_symlink()

    reconcile_group_cancel_operations(cfg)

    assert not active_path.exists()
    assert stable_path.exists()
    assert not stable_path.is_symlink()
    control = read_json(stable_path)["group_control"]
    assert control["state"] == "blocked"
    assert control["blocked_reason"] == "cancellation_barrier_missing"


def test_group_cancel_persists_snapshot_operation(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    submit(cfg, ["echo", "ok"], group="exp")
    group = group_control(cfg, "exp", "cancel")
    operation = group["cancellation_operation"]
    operation_path = cfg.shared_root / "operations" / "group-control" / f"{operation['operation_id']}.json"
    assert operation_path.exists()
    assert operation["membership_high_watermark"] == 1


def test_default_group_cancel_does_not_block_authorized_attempt_recovery(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    task = submit(cfg, ["echo", "ok"], group="exp", sharing_mode="spillover")
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    manifest_path = cfg.runtime_root / "processes" / f"{attempt.attempt_id}.json"
    atomic_replace(
        manifest_path,
        {
            "process": {
                "task_id": task.task_id,
                "attempt_id": attempt.attempt_id,
                "fencing_token": attempt.current_fencing_token,
                "process_group_id": 9876,
                "observed_state": "running",
            }
        },
    )
    assert expire_claim(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    group_control(cfg, "exp", "cancel")
    assert recover_running_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)


def test_doctor_restores_terminating_group_cancel_intent(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    task = submit(cfg, ["echo", "ok"], group="exp", sharing_mode="spillover")
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    group = group_control(cfg, "exp", "cancel", terminate_running=True)
    operation = group["cancellation_operation"]
    task_path = cfg.shared_root / "tasks" / f"{task.task_id}.json"
    task_data = read_json(task_path)
    task_data["task"]["control"].update(
        {"cancellation_requested_at": None, "cancellation_operation_id": None, "terminate_running": False}
    )
    atomic_replace(task_path, task_data)
    operation_path = cfg.shared_root / "operations" / "group-control" / f"{operation['operation_id']}.json"
    operation_data = read_json(operation_path)
    operation_data["group_control"]["state"] = "converging"
    atomic_replace(operation_path, operation_data)
    repair_metadata(cfg)
    repaired = load_task(cfg, task.task_id)
    assert repaired.control["terminate_running"] is True
    assert repaired.control["cancellation_operation_id"] == operation["operation_id"]


def test_agent_reconciliation_completes_group_cancel_operation(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    task = submit(cfg, ["echo", "ok"], group="exp", sharing_mode="spillover")
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    manifest_path = cfg.runtime_root / "processes" / f"{attempt.attempt_id}.json"
    atomic_replace(
        manifest_path,
        {
            "process": {
                "task_id": task.task_id,
                "attempt_id": attempt.attempt_id,
                "fencing_token": attempt.current_fencing_token,
                "process_group_id": 9876,
                "observed_state": "running",
                "supervisor": "agent",
            }
        },
    )
    group = group_control(cfg, "exp", "cancel", terminate_running=True)
    operation_id = group["cancellation_operation"]["operation_id"]
    monkeypatch.setattr("qqtools.plugins.qexp.scheduler._process_evidence_state", lambda *args: "absent")
    reconcile_running_tasks(cfg)
    reconcile_group_cancel_operations(cfg)
    operation_path = cfg.shared_root / "operations" / "group-control" / f"{operation_id}.json"
    assert read_json(operation_path)["group_control"]["state"] == "completed"


def test_group_show_reconciles_waiting_cancel_operation(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    task = submit(cfg, ["echo", "ok"], group="exp", sharing_mode="spillover")
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    manifest_path = cfg.runtime_root / "processes" / f"{attempt.attempt_id}.json"
    atomic_replace(
        manifest_path,
        {
            "process": {
                "task_id": task.task_id,
                "attempt_id": attempt.attempt_id,
                "fencing_token": attempt.current_fencing_token,
                "process_group_id": 9876,
                "observed_state": "running",
                "supervisor": "agent",
            }
        },
    )
    group_control(cfg, "exp", "cancel", terminate_running=True)
    monkeypatch.setattr("qqtools.plugins.qexp.scheduler._process_evidence_state", lambda *args: "absent")
    reconcile_running_tasks(cfg)
    assert show_group(cfg, "exp")["cancellation_operation"]["state"] == "completed"


def test_scheduler_rejects_task_before_submission_commit(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    operation_path = cfg.shared_root / "operations" / "submissions" / f"{task.submission_operation_id}.json"
    operation = read_json(operation_path)
    operation["submission"]["state"] = "committing"
    atomic_replace(operation_path, operation)
    assert claim_task(cfg, task.task_id, [0]) is None
