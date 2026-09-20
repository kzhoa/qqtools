from pathlib import Path

import pytest

import qqtools.plugins.qexp.scheduler as scheduler_module
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
from qqtools.plugins.qexp.runtime.attempt_recovery import recover_running_attempt
from qqtools.plugins.qexp.runtime.operation_store import active_operation_path, write_active_operation
from qqtools.plugins.qexp.runtime.paths import attempt_path
from qqtools.plugins.qexp.runtime.records import AttemptRecord
from qqtools.plugins.qexp.runtime.resources.reservations import reserve, reserved_gpu_ids
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task
from qqtools.plugins.qexp.scheduler import (
    authorize_launch,
    cancel_task,
    claim_task,
    expire_claim,
    reconcile_running_tasks,
    resume_starting_attempt,
    run_dispatch_cycle,
)
from tests.helpers.qexp.clock import set_offer_evaluation_time

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


def test_fresh_launch_persists_matching_starting_pair_in_order(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    original_revision = load_task(cfg, task.task_id).meta["revision"]
    path = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    launch_id = "fresh-launch-id"
    authorized_at = "2026-09-20T12:34:56Z"
    writes = []
    original_save_task = scheduler_module.save_task
    original_atomic_replace = scheduler_module.atomic_replace

    def record_task_write(cfg_arg, task_arg):
        original_save_task(cfg_arg, task_arg)
        stored_attempt = AttemptRecord.from_dict(read_json(path))
        writes.append(("task", task_arg.claim_control["active_claim"].copy(), stored_attempt.phase))

    def record_attempt_write(target, value):
        original_atomic_replace(target, value)
        if target == path:
            writes.append(("attempt", value["attempt"]["authorization"].copy(), value["attempt"]["phase"]))

    monkeypatch.setattr(scheduler_module, "save_task", record_task_write)
    monkeypatch.setattr(scheduler_module, "atomic_replace", record_attempt_write)
    monkeypatch.setattr(scheduler_module, "utc_now", lambda: authorized_at)
    monkeypatch.setattr(scheduler_module.uuid, "uuid4", lambda: type("LaunchId", (), {"hex": launch_id})())

    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)

    stored_task = load_task(cfg, task.task_id)
    stored_attempt = AttemptRecord.from_dict(read_json(path))
    claim = stored_task.claim_control["active_claim"]
    assert [write[0] for write in writes] == ["task", "attempt"]
    assert writes[0][2] == "claimed"
    assert claim["launch_state"] == stored_attempt.phase == "starting"
    assert claim["launch_id"] == stored_attempt.authorization["launch_id"] == launch_id
    assert claim["launch_authorized_at"] == stored_attempt.timestamps["launch_authorized_at"] == authorized_at
    assert stored_task.meta["revision"] == original_revision + 1
    assert stored_task.meta["updated_at"] == authorized_at


@pytest.mark.parametrize("identity", ["attempt", "token"])
def test_fresh_launch_rejects_mismatched_attempt_identity(tmp_path: Path, identity: str):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    task_before = load_task(cfg, task.task_id).to_dict()
    attempt_file = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    attempt_before = read_json(attempt_file)

    attempt_id = "different-attempt" if identity == "attempt" else attempt.attempt_id
    fencing_token = attempt.current_fencing_token + 1 if identity == "token" else attempt.current_fencing_token

    assert not authorize_launch(cfg, task.task_id, attempt_id, fencing_token)
    assert load_task(cfg, task.task_id).to_dict() == task_before
    assert read_json(attempt_file) == attempt_before


@pytest.mark.parametrize(
    ("field", "mismatched_value"),
    [
        ("task_id", "different-task"),
        ("attempt_number", "../../tasks/unrelated"),
        ("attempt_number", True),
        ("attempt_number", 1.0),
    ],
)
def test_fresh_launch_rejects_mismatched_embedded_attempt_path(
    tmp_path: Path,
    field: str,
    mismatched_value: object,
):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    attempt_file = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    damaged_attempt = read_json(attempt_file)
    damaged_attempt["attempt"][field] = mismatched_value
    atomic_replace(attempt_file, damaged_attempt)
    task_before = load_task(cfg, task.task_id).to_dict()

    assert not authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert load_task(cfg, task.task_id).to_dict() == task_before
    assert read_json(attempt_file) == damaged_attempt
    assert not (cfg.shared_root / "tasks" / "unrelated.json").exists()
    assert {path.name for path in attempt_file.parent.iterdir()} == {attempt_file.name}


def test_fresh_launch_rejects_mismatched_embedded_task_path(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    task_file = cfg.shared_root / "tasks" / f"{task.task_id}.json"
    damaged_task = read_json(task_file)
    damaged_task["task"]["task_id"] = "different-task"
    atomic_replace(task_file, damaged_task)
    attempt_file = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    attempt_before = read_json(attempt_file)

    assert not authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert read_json(task_file) == damaged_task
    assert read_json(attempt_file) == attempt_before
    assert not (cfg.shared_root / "tasks" / "different-task.json").exists()
    assert not (cfg.shared_root / "attempts" / "different-task").exists()


def test_committed_starting_recovery_preserves_pair_and_write_order(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    before = load_task(cfg, task.task_id)
    original_claim = before.claim_control["active_claim"].copy()
    original_revision = before.meta["revision"]
    path = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    writes = []
    original_save_task = scheduler_module.save_task
    original_atomic_replace = scheduler_module.atomic_replace

    def record_task_write(cfg_arg, task_arg):
        original_save_task(cfg_arg, task_arg)
        writes.append("task")

    def record_attempt_write(target, value):
        original_atomic_replace(target, value)
        if target == path:
            writes.append("attempt")

    monkeypatch.setattr(scheduler_module, "save_task", record_task_write)
    monkeypatch.setattr(scheduler_module, "atomic_replace", record_attempt_write)

    resumed = resume_starting_attempt(cfg, task.task_id)

    assert resumed is not None
    stored_task = load_task(cfg, task.task_id)
    stored_attempt = AttemptRecord.from_dict(read_json(path))
    claim = stored_task.claim_control["active_claim"]
    assert writes == ["task", "attempt"]
    assert claim["launch_id"] == stored_attempt.authorization["launch_id"] == original_claim["launch_id"]
    assert (
        claim["launch_authorized_at"]
        == stored_attempt.timestamps["launch_authorized_at"]
        == original_claim["launch_authorized_at"]
    )
    assert stored_task.meta["revision"] == original_revision + 1
    assert stored_task.meta["updated_at"] == original_claim["launch_authorized_at"]


@pytest.mark.parametrize(
    ("field", "mismatched_value"),
    [
        ("task_id", "different-task"),
        ("attempt_number", "../../tasks/unrelated"),
        ("attempt_number", True),
        ("attempt_number", 1.0),
    ],
)
def test_starting_recovery_rejects_mismatched_embedded_attempt_path(
    tmp_path: Path,
    field: str,
    mismatched_value: object,
):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    attempt_file = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    damaged_attempt = read_json(attempt_file)
    damaged_attempt["attempt"][field] = mismatched_value
    atomic_replace(attempt_file, damaged_attempt)
    task_before = load_task(cfg, task.task_id).to_dict()

    assert resume_starting_attempt(cfg, task.task_id) is None
    assert load_task(cfg, task.task_id).to_dict() == task_before
    assert read_json(attempt_file) == damaged_attempt
    assert not (cfg.shared_root / "tasks" / "unrelated.json").exists()
    assert {path.name for path in attempt_file.parent.iterdir()} == {attempt_file.name}


def test_starting_recovery_rejects_mismatched_embedded_task_path(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    task_file = cfg.shared_root / "tasks" / f"{task.task_id}.json"
    damaged_task = read_json(task_file)
    damaged_task["task"]["task_id"] = "different-task"
    atomic_replace(task_file, damaged_task)
    attempt_file = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    attempt_before = read_json(attempt_file)

    assert resume_starting_attempt(cfg, task.task_id) is None
    assert read_json(task_file) == damaged_task
    assert read_json(attempt_file) == attempt_before
    assert not (cfg.shared_root / "tasks" / "different-task.json").exists()
    assert not (cfg.shared_root / "attempts" / "different-task").exists()


def test_launch_authorization_replays_after_task_write_interruption(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    path = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    original_save_task = scheduler_module.save_task

    def interrupt_task_write(*args, **kwargs):
        raise OSError("simulated Task write interruption")

    monkeypatch.setattr(scheduler_module, "save_task", interrupt_task_write)
    with pytest.raises(OSError, match="Task write interruption"):
        authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)

    assert load_task(cfg, task.task_id).claim_control["active_claim"]["launch_state"] == "claimed"
    assert AttemptRecord.from_dict(read_json(path)).phase == "claimed"
    monkeypatch.setattr(scheduler_module, "save_task", original_save_task)

    executor = RecordingExecutor()
    assert run_dispatch_cycle(cfg, available_gpus=[0], executor=executor) == [task.task_id]
    assert [launched.attempt_id for launched in executor.attempts] == [attempt.attempt_id]
    stored_task = load_task(cfg, task.task_id)
    stored_attempt = AttemptRecord.from_dict(read_json(path))
    claim = stored_task.claim_control["active_claim"]
    assert claim["launch_state"] == stored_attempt.phase == "starting"
    assert claim["launch_id"] == stored_attempt.authorization["launch_id"]
    assert claim["launch_authorized_at"] == stored_attempt.timestamps["launch_authorized_at"]


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


def test_orphan_partial_terminal_replays_persisted_result(tmp_path: Path, monkeypatch):
    from qqtools.plugins.qexp import lifecycle
    from qqtools.plugins.qexp.scheduler import finalize_orphaned_attempt

    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert expire_claim(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    original = lifecycle.save_task

    def unavailable(*args, **kwargs):
        raise OSError("Task publication interrupted")

    monkeypatch.setattr(lifecycle, "save_task", unavailable)
    with pytest.raises(OSError, match="publication interrupted"):
        finalize_orphaned_attempt(
            cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, exit_code=0, was_terminated=False
        )
    monkeypatch.setattr(lifecycle, "save_task", original)
    assert not finalize_orphaned_attempt(
        cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token + 1, exit_code=9, was_terminated=True
    )
    assert load_task(cfg, task.task_id).state["projection"] == "blocked"
    assert finalize_orphaned_attempt(
        cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, exit_code=9, was_terminated=True
    )
    assert load_task(cfg, task.task_id).state["projection"] == "succeeded"
    stored = read_json(attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number))["attempt"]
    assert stored["result"]["exit_code"] == 0
    assert not reserved_gpu_ids(cfg.runtime_root)


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
    monkeypatch.setattr(
        "qqtools.plugins.qexp.infrastructure.process.os.killpg", lambda pid, sig: calls.append((pid, sig))
    )
    reconcile_running_tasks(cfg)
    repaired_manifest = read_json(manifest_path)["process"]
    assert repaired_manifest["fencing_token"] == token
    assert repaired_manifest["supervisor"] == "agent"
    assert calls == []


@pytest.mark.parametrize("boundary", ["task", "manifest"])
@pytest.mark.parametrize("exit_code", [None, 0, 1])
def test_authority_restart_finishes_interrupted_recovery(tmp_path: Path, monkeypatch, boundary, exit_code):
    from qqtools.plugins.qexp.runtime import attempt_recovery

    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    task = submit(cfg, ["echo", "ok"], group="exp", sharing_mode="spillover")
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    old_token = attempt.current_fencing_token
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, old_token)
    path = cfg.runtime_root / "processes" / f"{attempt.attempt_id}.json"
    atomic_replace(
        path,
        {
            "process": {
                "protocol_version": 1,
                "task_id": task.task_id,
                "attempt_id": attempt.attempt_id,
                "fencing_token": old_token,
                "process_group_id": 9876,
                "process_group_start_time_ticks": 123,
                "observed_state": "running",
                "supervisor": "agent",
            }
        },
    )
    stored_path = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    record = read_json(stored_path)
    record["attempt"]["process"].update(process_group_id=9876, process_group_start_time_ticks=123)
    atomic_replace(stored_path, record)
    assert expire_claim(cfg, task.task_id, attempt.attempt_id, old_token)
    monkeypatch.setattr("qqtools.plugins.qexp.scheduler._process_start_time_ticks", lambda _: 123)
    monkeypatch.setattr("qqtools.plugins.qexp.scheduler._is_process_group_alive", lambda _: True)

    class RecoveryCrash(BaseException):
        pass

    with monkeypatch.context() as crash:
        if boundary == "task":

            def fail_save(*args):
                raise RecoveryCrash()

            crash.setattr(attempt_recovery, "save_task", fail_save)
        else:
            original = attempt_recovery.atomic_replace

            def fail_manifest(target, value):
                if target == path:
                    raise RecoveryCrash()
                return original(target, value)

            crash.setattr(attempt_recovery, "atomic_replace", fail_manifest)
        first = AuthoritySupervisor(cfg)
        with pytest.raises(RecoveryCrash):
            first.recover_startup()
    assert read_json(path)["process"]["fencing_token"] == old_token
    new_token = read_json(stored_path)["attempt"]["current_fencing_token"]
    assert new_token > old_token
    with monkeypatch.context() as mismatch:
        mismatch.setattr("qqtools.plugins.qexp.scheduler._process_start_time_ticks", lambda _: 456)
        rejected = AuthoritySupervisor(cfg)
        rejected.recover_startup()
        rejected.tick()
        assert read_json(path)["process"]["fencing_token"] == old_token
    if exit_code is not None:
        monkeypatch.setattr("qqtools.plugins.qexp.scheduler._process_start_time_ticks", lambda _: None)
        monkeypatch.setattr("qqtools.plugins.qexp.scheduler._is_process_group_alive", lambda _: False)
        observation_path = cfg.runtime_root / "process-observations" / f"{attempt.attempt_id}.json"
        observation = {
            "protocol_version": 1,
            "task_id": task.task_id,
            "attempt_id": "wrong-attempt",
            "observed_exit_code": exit_code,
        }
        atomic_replace(observation_path, {"exit_observation": observation})
        rejected = AuthoritySupervisor(cfg)
        rejected.recover_startup()
        rejected.tick()
        assert read_json(path)["process"]["fencing_token"] == old_token
        expected_projection = "blocked" if boundary == "task" else "running"
        assert load_task(cfg, task.task_id).state["projection"] == expected_projection
        observation["attempt_id"] = attempt.attempt_id
        atomic_replace(observation_path, {"exit_observation": observation})
    renewed = []
    from qqtools.plugins.qexp import authority

    original_renew = authority.renew_attempt_lease

    def record_renewal(*args, **kwargs):
        renewed.append(args[3])
        return original_renew(*args, **kwargs)

    monkeypatch.setattr(
        "qqtools.plugins.qexp.authority.renew_attempt_lease",
        record_renewal,
    )
    restarted = AuthoritySupervisor(cfg)
    restarted.recover_startup()
    restarted.tick()
    if exit_code is None:
        assert read_json(path)["process"]["fencing_token"] == new_token
        assert load_task(cfg, task.task_id).claim_control["active_claim"]["fencing_token"] == new_token
    if exit_code is not None:
        if boundary == "manifest":
            restarted.tick()
        expected_phase = "succeeded" if exit_code == 0 else "failed"
        assert load_task(cfg, task.task_id).state["projection"] == expected_phase
        assert not load_task(cfg, task.task_id).claim_control.get("active_claim")
        assert read_json(stored_path)["attempt"]["phase"] == expected_phase
        assert read_json(stored_path)["attempt"]["token_history"] == [old_token, new_token]
        restarted.tick()
        assert load_task(cfg, task.task_id).state["projection"] == expected_phase
        if boundary == "manifest":
            assert not path.exists()
        else:
            # No claim was published for the recovered token in this window.
            # Terminal reconciliation retains the exited manifest as evidence.
            process = read_json(path)["process"]
            assert process["observed_state"] == "exited"
            assert process["observed_exit_code"] == exit_code
        return
    assert renewed == [new_token]
    assert read_json(stored_path)["attempt"]["token_history"] == [old_token, new_token]


def test_terminal_accounting_read_failure_remains_pending(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    supervisor = AuthoritySupervisor(cfg)

    def unavailable(*args):
        raise OSError("temporary shared read failure")

    monkeypatch.setattr("qqtools.plugins.qexp.authority.load_task", unavailable)
    assert supervisor._reconcile_terminal_accounting("task", "attempt") is False


def test_terminal_accounting_transient_failure_retries_next_tick(tmp_path: Path, monkeypatch):
    from qqtools.plugins.qexp import authority

    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    process = {
        "protocol_version": 1,
        "task_id": task.task_id,
        "attempt_id": attempt.attempt_id,
        "fencing_token": attempt.current_fencing_token,
    }
    path = cfg.runtime_root / "processes" / f"{attempt.attempt_id}.json"
    atomic_replace(path, {"process": process})
    supervisor = AuthoritySupervisor(cfg)
    original_load = authority.load_task
    with monkeypatch.context() as outage:

        def unavailable(*args):
            raise OSError("temporary shared read failure")

        outage.setattr(authority, "load_task", unavailable)
        supervisor.tick()
    assert path.exists()
    assert original_load(cfg, task.task_id).claim_control["active_claim"]
    renewed = []
    original_renew = authority.renew_attempt_lease

    def record_renewal(*args):
        renewed.append(args[3])
        return original_renew(*args)

    monkeypatch.setattr(authority, "renew_attempt_lease", record_renewal)
    supervisor.tick()
    assert renewed == [attempt.current_fencing_token]
    assert path.exists()


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
    set_offer_evaluation_time(monkeypatch, cfg, task.task_id, seconds_after_deadline=-1)
    offer_due_tasks(cfg)
    assert load_task(cfg, task.task_id).placement_runtime["queue_scope"] == "home"
    set_offer_evaluation_time(monkeypatch, cfg, task.task_id)
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

    assert any(item["operation_id"] == operation_id and item["state"] == "completed" for item in reconciled)
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
    write_active_operation(
        cfg,
        "group_control",
        operation_id,
        {
            "meta": {
                "schema_version": 6,
                "revision": 1,
                "created_at": "2026-08-06T00:00:00Z",
                "updated_at": "2026-08-06T00:00:00Z",
                "updated_by": {"actor_type": "cli", "machine_name": "g1", "process_id": "test"},
            },
            "group_control": {
                "operation_id": operation_id,
                "operation_type": "cancel",
                "group_name": "exp",
                "state": "converging",
                "group_revision_at_start": 1,
                "dispatch_epoch_at_start": 0,
                "membership_high_watermark": 1,
                "terminate_running": False,
                "progress": {
                    "target_tasks": 0,
                    "already_terminal": 0,
                    "queued_cancelled": 0,
                    "prelaunch_cancelled": 0,
                    "running_allowed": 0,
                    "termination_pending": 0,
                    "termination_acknowledged": 0,
                    "blocked": 0,
                },
                "pending_machine_acknowledgements": {},
                "created_at": "2026-08-06T00:00:00Z",
                "updated_at": "2026-08-06T00:00:00Z",
                "completed_at": None,
                "blocked_reason": None,
            },
        },
    )
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


@pytest.mark.parametrize("crash_point", ["barrier", "converging"])
@pytest.mark.parametrize("phase", ["queued", "claimed", "authorized", "rejected"])
@pytest.mark.parametrize("terminate_running", [False, True])
@pytest.mark.parametrize("is_legacy_doctor", [False, True])
def test_group_cancel_replays_after_crash_before_first_task(
    tmp_path, monkeypatch, crash_point, phase, terminate_running, is_legacy_doctor
):
    from qqtools.plugins.qexp.commands import group as group_commands
    from qqtools.plugins.qexp.runtime.paths import group_path

    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    task = submit(cfg, ["echo", "recover"], group="exp")
    reservation_root = tmp_path / "doctor-reservations" if is_legacy_doctor else cfg.runtime_root
    if phase != "queued":
        attempt = claim_task(cfg, task.task_id, [0], reservation_runtime_root=reservation_root)
        assert attempt is not None
        if phase == "authorized":
            assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
        elif phase == "rejected":
            attempt_file = attempt_path(cfg.shared_root, task.task_id, 1)
            value = read_json(attempt_file)
            value["attempt"]["current_fencing_token"] += 1
            atomic_replace(attempt_file, value)
    before = load_task(cfg, task.task_id).to_dict()
    write_operation = group_commands.write_active_operation
    replace = group_commands.atomic_replace

    def fail_after_operation(cfg, kind, operation_id, value):
        result = write_operation(cfg, kind, operation_id, value)
        if crash_point == "converging" and value["group_control"]["state"] == "converging":
            raise OSError("interrupted before first Group member")
        return result

    def fail_after_barrier(path, value):
        replace(path, value)
        if crash_point == "barrier" and path == group_path(cfg.shared_root, "exp"):
            raise OSError("interrupted before first Group member")

    with monkeypatch.context() as crashing:
        crashing.setattr(group_commands, "write_active_operation", fail_after_operation)
        crashing.setattr(group_commands, "atomic_replace", fail_after_barrier)
        with pytest.raises(OSError, match="interrupted before first Group member"):
            group_control(cfg, "exp", "cancel", terminate_running=terminate_running)
    assert load_task(cfg, task.task_id).to_dict() == before
    later = submit(cfg, ["echo", "later"], group="exp")
    other = submit(cfg, ["echo", "unrelated"])
    group_data = read_json(group_path(cfg.shared_root, "exp"))
    operation_id = group_data["cancellation_operation"]["operation_id"]

    operation_path = cfg.shared_root / "operations/group-control" / f"{operation_id}.json"
    if is_legacy_doctor:
        active = active_operation_path(cfg, "group_control", operation_id)
        atomic_replace(operation_path, read_json(active))
        active.unlink()
        assert not operation_path.is_symlink()
        repair_result = repair_metadata(cfg, reservation_runtime_root=reservation_root)
    else:
        reconcile_group_cancel_operations(cfg)

    current = load_task(cfg, task.task_id)
    control = read_json(operation_path)["group_control"]
    if phase == "rejected":
        assert current.to_dict() == before
        assert control["state"] == "blocked"
        assert control["completed_at"] is None
        assert reserved_gpu_ids(reservation_root) == {0}
        if is_legacy_doctor:
            assert operation_id in repair_result["blocked"]
            assert operation_id not in repair_result["repaired"]
    elif phase == "authorized":
        assert current.state["projection"] == "running"
        assert current.control["terminate_running"] is terminate_running
        assert control["state"] == ("waiting_ack" if terminate_running else "completed")
        if terminate_running:
            assert current.control["cancellation_operation_id"] == operation_id
    else:
        assert current.state["projection"] == "cancelled"
        assert current.control["cancellation_operation_id"] == operation_id
        assert control["state"] == "completed"
        if phase == "claimed":
            stored = read_json(attempt_path(cfg.shared_root, task.task_id, 1))["attempt"]
            assert stored["phase"] == "cancelled"
            assert not reserved_gpu_ids(reservation_root)
    assert load_task(cfg, later.task_id).state["projection"] == "queued"
    assert load_task(cfg, other.task_id).state["projection"] == "queued"


def test_replaying_default_group_cancel_preserves_later_termination(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.commands import group as group_commands

    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    task = submit(cfg, ["echo", "running"], group="exp")
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    write_operation = group_commands.write_active_operation

    def interrupted(cfg, kind, operation_id, value):
        result = write_operation(cfg, kind, operation_id, value)
        if value["group_control"]["state"] == "converging":
            raise OSError("before default cancellation converged")
        return result

    with monkeypatch.context() as crashing:
        crashing.setattr(group_commands, "write_active_operation", interrupted)
        with pytest.raises(OSError, match="before default cancellation converged"):
            group_control(cfg, "exp", "cancel")
    terminating = group_control(cfg, "exp", "cancel", terminate_running=True)["cancellation_operation"]
    requested = load_task(cfg, task.task_id).to_dict()
    for _ in range(3):
        reconcile_group_cancel_operations(cfg)
    assert load_task(cfg, task.task_id).to_dict() == requested
    assert requested["task"]["control"]["cancellation_operation_id"] == terminating["operation_id"]


def test_group_cancel_missing_group_remains_durably_blocked(tmp_path):
    from qqtools.plugins.qexp.runtime.paths import group_path

    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    task = submit(cfg, ["echo", "running"], group="exp")
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    control = group_control(cfg, "exp", "cancel", terminate_running=True)["cancellation_operation"]
    before = load_task(cfg, task.task_id).to_dict()
    group_path(cfg.shared_root, "exp").unlink()

    result = reconcile_group_cancel_operations(cfg)

    assert len(result) == 1
    assert result[0]["blocked_reason"] == "group_missing"
    path = active_operation_path(cfg, "group_control", control["operation_id"])
    stored = read_json(path)["group_control"]
    assert stored["state"] == "blocked"
    assert stored["blocked_reason"] == "group_missing"
    assert stored["completed_at"] is None
    assert load_task(cfg, task.task_id).to_dict() == before


def _enable_bound_worker_removal(cfg):
    from qqtools.plugins.qexp.runtime.group_namespace import activate_group_authority_locked
    from qqtools.plugins.qexp.runtime.locks import schema_lock

    with schema_lock(cfg.shared_root):
        path = cfg.shared_root / "schema/version.json"
        value = read_json(path)
        value["schema"]["required_capabilities"].append("local-recovery-v1")
        atomic_replace(path, value)
        assert activate_group_authority_locked(cfg)


def _converge_worker_removal(cfg, control, *, state="completed"):
    from qqtools.plugins.qexp.runtime.operation_store import locate_operation_path
    from tests.helpers.qexp_discovery import discover_group

    discover_group(cfg, control["group_name"])
    for _ in range(100):
        path = locate_operation_path(cfg, "group_control", control["operation_id"])
        current = read_json(path)["group_control"]
        if current["state"] == state:
            return current
        reconcile_group_cancel_operations(cfg)
    raise AssertionError(f"Worker removal did not reach {state}: {current}")


def test_worker_remove_recovers_crash_before_draining_publication(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.commands import worker_removal as group_commands
    from qqtools.plugins.qexp.runtime.paths import group_path

    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _enable_bound_worker_removal(cfg)
    _existing_group(cfg)
    task = submit(cfg, ["echo", "queued"], group="exp")
    write_operation = group_commands.write_active_operation
    operation_ids = []

    def interrupted(cfg, kind, operation_id, value):
        result = write_operation(cfg, kind, operation_id, value)
        if value["group_control"]["operation_type"] == "worker_remove_v2":
            operation_ids.append(operation_id)
            raise OSError("before draining Group publication")
        return result

    with monkeypatch.context() as crashing:
        crashing.setattr(group_commands, "write_active_operation", interrupted)
        with pytest.raises(OSError, match="before draining Group publication"):
            change_worker(cfg, "exp", "g1", "remove")
    assert read_json(group_path(cfg.shared_root, "exp"))["group"]["worker_set"]["g1"]["state"] == "active"

    _converge_worker_removal(cfg, {"group_name": "exp", "operation_id": operation_ids[0]}, state="waiting_ack")

    group = read_json(group_path(cfg.shared_root, "exp"))
    assert group["group"]["worker_set"]["g1"]["state"] == "draining"
    operation = read_json(active_operation_path(cfg, "group_control", operation_ids[0]))["group_control"]
    assert operation["state"] == "waiting_ack"
    assert task.task_id in operation["blockers"]
    assert load_task(cfg, task.task_id).state["projection"] == "queued"


@pytest.mark.parametrize("should_drain_again", [False, True])
@pytest.mark.parametrize("terminate_running", [False, True])
def test_worker_remove_does_not_follow_reactivated_worker(tmp_path, should_drain_again, terminate_running):
    from qqtools.plugins.qexp.runtime.paths import group_path

    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _enable_bound_worker_removal(cfg)
    _existing_group(cfg)
    original = submit(cfg, ["echo", "old"], group="exp")
    change_worker(cfg, "exp", "g1", "remove", terminate_running=terminate_running)
    change_worker(cfg, "exp", "g1", "add")
    cancel_task(cfg, original.task_id)
    if terminate_running:
        later = submit(cfg, ["echo", "new"], group="exp")
        attempt = claim_task(cfg, later.task_id, [0])
        assert attempt is not None
        assert authorize_launch(cfg, later.task_id, attempt.attempt_id, attempt.current_fencing_token)
        task_before = load_task(cfg, later.task_id).to_dict()
    if should_drain_again:
        change_worker(cfg, "exp", "g1", "drain")
    before = read_json(group_path(cfg.shared_root, "exp"))["group"]["worker_set"]["g1"]

    reconcile_group_cancel_operations(cfg)

    after = read_json(group_path(cfg.shared_root, "exp"))["group"]["worker_set"]["g1"]
    assert after == before
    if terminate_running:
        assert load_task(cfg, later.task_id).to_dict() == task_before


def test_worker_remove_replays_when_final_group_publication_fails(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.commands import worker_removal as group_commands
    from qqtools.plugins.qexp.runtime.paths import group_path

    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _enable_bound_worker_removal(cfg)
    _existing_group(cfg)
    replace = group_commands.atomic_replace
    path = group_path(cfg.shared_root, "exp")

    def interrupted(target, value):
        if target == path and value["group"]["worker_set"]["g1"]["state"] == "removing":
            raise OSError("before removing Group publication")
        return replace(target, value)

    with monkeypatch.context() as crashing:
        crashing.setattr(group_commands, "atomic_replace", interrupted)
        with pytest.raises(OSError, match="before removing Group publication"):
            change_worker(cfg, "exp", "g1", "remove")

    reconcile_group_cancel_operations(cfg)

    group = read_json(path)
    assert group["group"]["worker_set"]["g1"]["state"] == "removing"
    assert group["worker_control"]["state"] == "completed"


def test_group_replay_maintenance_uses_explicit_reservation_backend(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.commands import group as group_commands
    from qqtools.plugins.qexp.project_maintenance import maintain_project

    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _existing_group(cfg)
    task = submit(cfg, ["echo", "claimed"], group="exp")
    root = tmp_path / "reservations"
    project_id = read_json(cfg.shared_root / "project/identity.json")["project"]["project_id"]
    attempt = claim_task(cfg, task.task_id, [0], reservation_runtime_root=root, project_id=project_id)
    assert attempt is not None
    write_operation = group_commands.write_active_operation

    def interrupted(cfg, kind, operation_id, value):
        result = write_operation(cfg, kind, operation_id, value)
        if value["group_control"]["state"] == "converging":
            raise OSError("before member cancellation")
        return result

    with monkeypatch.context() as crashing:
        crashing.setattr(group_commands, "write_active_operation", interrupted)
        with pytest.raises(OSError, match="before member cancellation"):
            group_control(cfg, "exp", "cancel", reservation_runtime_root=root)

    maintain_project(cfg, reservation_runtime_root=root, project_id=project_id, should_reconcile_reservations=False)

    assert load_task(cfg, task.task_id).state["projection"] == "cancelled"
    assert not reserved_gpu_ids(root)


@pytest.mark.parametrize("cut", ["draining", "converging", "removing", "archived"])
def test_bound_worker_remove_replays_each_durable_boundary(tmp_path, monkeypatch, cut):
    from qqtools.plugins.qexp.commands import worker_removal
    from qqtools.plugins.qexp.runtime.paths import group_path

    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _enable_bound_worker_removal(cfg)
    _existing_group(cfg)
    group_file = group_path(cfg.shared_root, "exp")
    replace = worker_removal.atomic_replace
    write = worker_removal.write_active_operation
    archive = worker_removal.archive_operation

    def interrupted_group(path, value):
        replace(path, value)
        if path == group_file and value["group"]["worker_set"]["g1"]["state"] == cut:
            raise OSError("after durable boundary")

    def interrupted_operation(cfg, kind, operation_id, value):
        result = write(cfg, kind, operation_id, value)
        if cut == value["group_control"]["state"]:
            raise OSError("after durable boundary")
        return result

    def interrupted_archive(cfg, kind, operation_id, value):
        if cut == "archived":
            # A completed historical copy is durable but the active copy remains.
            atomic_replace(cfg.shared_root / "operations/group-control" / f"{operation_id}.json", value)
            raise OSError("after durable boundary")
        return archive(cfg, kind, operation_id, value)

    with monkeypatch.context() as crashing:
        crashing.setattr(worker_removal, "atomic_replace", interrupted_group)
        crashing.setattr(worker_removal, "write_active_operation", interrupted_operation)
        crashing.setattr(worker_removal, "archive_operation", interrupted_archive)
        with pytest.raises(OSError, match="after durable boundary"):
            change_worker(cfg, "exp", "g1", "remove")
    reconcile_group_cancel_operations(cfg)
    group = read_json(group_file)
    control = group["worker_control"]
    assert group["group"]["worker_set"]["g1"]["state"] == "removing"
    assert control["state"] == "completed"
    assert not active_operation_path(cfg, "group_control", control["operation_id"]).exists()
    assert (
        read_json(cfg.shared_root / "operations/group-control" / f"{control['operation_id']}.json")["group_control"][
            "state"
        ]
        == "completed"
    )


def test_bound_worker_remove_survives_role_and_quota_updates(tmp_path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _enable_bound_worker_removal(cfg)
    _existing_group(cfg)
    task = submit(cfg, ["echo", "queued"], group="exp")
    first = change_worker(cfg, "exp", "g1", "remove")["worker_control"]
    change_worker(cfg, "exp", "g1", "set", role="borrow", has_gpu_limit=True, gpu_limit_gpus=2)
    cancel_task(cfg, task.task_id)
    _converge_worker_removal(cfg, first)
    group = show_group(cfg, "exp")
    worker = group["group"]["worker_set"]["g1"]
    assert worker["state"] == "removing"
    assert worker["scheduling_role"] == "borrow"
    assert worker["gpu_limit_gpus"] == 2
    assert worker["removal_operation_id"] == first["operation_id"]
    assert group["worker_control"]["state"] == "completed"


def test_bound_worker_remove_reuses_operation_and_only_escalates(tmp_path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _enable_bound_worker_removal(cfg)
    _existing_group(cfg)
    task = submit(cfg, ["echo", "running"], group="exp")
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    first = change_worker(cfg, "exp", "g1", "remove")["worker_control"]
    assert not load_task(cfg, task.task_id).control.get("terminate_running")
    second = change_worker(cfg, "exp", "g1", "remove", terminate_running=True)["worker_control"]
    _converge_worker_removal(cfg, second, state="waiting_ack")
    for _ in range(10):
        if load_task(cfg, task.task_id).control.get("terminate_running"):
            break
        reconcile_group_cancel_operations(cfg)
    requested = load_task(cfg, task.task_id).to_dict()
    third = change_worker(cfg, "exp", "g1", "remove")["worker_control"]
    assert first["operation_id"] == second["operation_id"] == third["operation_id"]
    assert second["terminate_running"] and third["terminate_running"]
    assert requested["task"]["control"]["terminate_running"]
    assert load_task(cfg, task.task_id).to_dict() == requested


def test_bound_worker_removals_have_independent_ownership(tmp_path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _enable_bound_worker_removal(cfg)
    _existing_group(cfg)
    task = submit(cfg, ["echo", "queued"], group="exp")
    first = change_worker(cfg, "exp", "g1", "remove")["worker_control"]
    change_worker(cfg, "exp", "g2", "add")
    second = change_worker(cfg, "exp", "g2", "remove")["worker_control"]
    second = _converge_worker_removal(cfg, second)
    assert second["state"] == "completed"
    cancel_task(cfg, task.task_id)
    _converge_worker_removal(cfg, first)
    group = show_group(cfg, "exp")
    for machine in ("g1", "g2"):
        assert group["group"]["worker_set"][machine]["state"] == "removing"
    assert group["worker_control"]["operation_id"] == second["operation_id"]
    path = cfg.shared_root / "operations/group-control" / f"{first['operation_id']}.json"
    assert read_json(path)["group_control"]["state"] == "completed"


@pytest.mark.parametrize(
    "field,value", [("worker_before", None), ("draining_state_epoch", True), ("terminate_running", 1)]
)
def test_bound_worker_remove_rejects_malformed_proof_without_effects(tmp_path, field, value):
    from qqtools.plugins.qexp.runtime.paths import group_path

    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _enable_bound_worker_removal(cfg)
    _existing_group(cfg)
    task = submit(cfg, ["echo", "queued"], group="exp")
    control = change_worker(cfg, "exp", "g1", "remove")["worker_control"]
    path = active_operation_path(cfg, "group_control", control["operation_id"])
    operation = read_json(path)
    operation["group_control"][field] = value
    atomic_replace(path, operation)
    group_before = group_path(cfg.shared_root, "exp").read_bytes()
    task_before = load_task(cfg, task.task_id).to_dict()
    reconcile_group_cancel_operations(cfg)
    assert read_json(path)["group_control"]["blocked_reason"] == "worker_removal_proof_invalid"
    assert group_path(cfg.shared_root, "exp").read_bytes() == group_before
    assert load_task(cfg, task.task_id).to_dict() == task_before


@pytest.mark.parametrize("cut_before_archive", [False, True])
def test_explicit_drain_after_removing_allows_fresh_removal(tmp_path, monkeypatch, cut_before_archive):
    from qqtools.plugins.qexp.commands import worker_removal
    from qqtools.plugins.qexp.runtime.paths import group_path

    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _enable_bound_worker_removal(cfg)
    _existing_group(cfg)
    with monkeypatch.context() as cut:
        if cut_before_archive:

            def interrupted(*_args, **_kwargs):
                raise OSError("before archive")

            cut.setattr(worker_removal, "archive_operation", interrupted)
            with pytest.raises(OSError, match="before archive"):
                change_worker(cfg, "exp", "g1", "remove")
        else:
            change_worker(cfg, "exp", "g1", "remove")
    first = read_json(group_path(cfg.shared_root, "exp"))
    assert first["group"]["worker_set"]["g1"]["state"] == "removing"
    first_id = first["worker_control"]["operation_id"]
    drained = change_worker(cfg, "exp", "g1", "drain")
    assert drained["group"]["worker_set"]["g1"]["state"] == "draining"
    assert "removal_operation_id" not in drained["group"]["worker_set"]["g1"]
    reconcile_group_cancel_operations(cfg)
    removed = change_worker(cfg, "exp", "g1", "remove")
    assert removed["group"]["worker_set"]["g1"]["state"] == "removing"
    assert removed["worker_control"]["state"] == "completed"
    assert removed["worker_control"]["operation_id"] != first_id


@pytest.mark.parametrize("phase", ["failed", "orphaned"])
def test_repeated_removal_rechecks_tasks_retried_after_completion(tmp_path, phase):
    from qqtools.plugins.qexp.commands.task import retry
    from qqtools.plugins.qexp.scheduler import fail_attempt

    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    _enable_bound_worker_removal(cfg)
    _existing_group(cfg)
    task = submit(cfg, ["true"], group="exp")
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    if phase == "failed":
        assert fail_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "test_failure")
    else:
        assert expire_claim(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    first = change_worker(cfg, "exp", "g1", "remove")["worker_control"]
    first = _converge_worker_removal(cfg, first)
    assert first["state"] == "completed"
    retried = retry(cfg, task.task_id)
    assert retried.state["projection"] == "queued"
    second = change_worker(cfg, "exp", "g1", "remove")
    second["worker_control"] = _converge_worker_removal(cfg, second["worker_control"], state="waiting_ack")
    assert second["worker_control"]["state"] == "waiting_ack"
    assert second["worker_control"]["blockers"] == [task.task_id]
    assert second["worker_control"]["operation_id"] != first["operation_id"]
    assert second["group"]["worker_set"]["g1"]["state"] == "draining"
