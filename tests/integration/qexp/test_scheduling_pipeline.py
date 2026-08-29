import signal
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.commands.group import create_group
from qqtools.plugins.qexp.commands.task import cancel, retry
from qqtools.plugins.qexp.runtime.paths import attempt_path, shared_paths
from qqtools.plugins.qexp.runtime.records import AttemptRecord
from qqtools.plugins.qexp.runtime.reservations import reserved_gpu_ids
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task
from qqtools.plugins.qexp.scheduler import (
    _manifest_supervisor,
    _terminate_process_group,
    authorize_launch,
    claim_task,
    expire_claim,
    reconcile_running_tasks,
    run_dispatch_cycle,
)

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]

class RecordingExecutor:
    def __init__(self):
        self.launched = []

    def launch_attempt(self, cfg, task_id, attempt):
        self.launched.append((task_id, attempt.attempt_id))


def test_dispatch_claims_and_launches_with_reservation(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    submit(cfg, ["echo", "ok"])
    executor = RecordingExecutor()
    assert run_dispatch_cycle(cfg, available_gpus=[0], executor=executor) == [next(iter((cfg.shared_root / "tasks").glob("*.json"))).stem]
    assert executor.launched
    assert reserved_gpu_ids(cfg.runtime_root) == {0}


def test_worker_set_blocks_intruder_claim(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    create_group(cfg, "exp")
    task = submit(cfg, ["echo", "ok"], group="exp", sharing_mode="spillover")
    other = cfg.__class__(cfg.shared_root, cfg.project_root, "intruder", tmp_path / "intruder-rt")
    assert claim_task(other, task.task_id, [0]) is None


def test_cancelled_prelaunch_claim_is_not_launchable(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    cancelled = cancel(cfg, task.task_id)
    assert cancelled.state["projection"] == "cancelled"
    assert reserved_gpu_ids(cfg.runtime_root) == set()


def test_prelaunch_cancel_with_missing_attempt_still_clears_claim_and_reservation(
        tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number).unlink()

    cancelled = cancel(cfg, task.task_id)

    assert cancelled.state == {"projection": "cancelled", "reason": "cancelled_before_launch"}
    assert cancelled.claim_control["active_claim"] is None
    assert cancelled.attempt_control["current_attempt_id"] is None
    assert reserved_gpu_ids(cfg.runtime_root) == set()


def test_prelaunch_cancel_increments_revision_once_and_event_matches_final_task(
        tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    revision_before_cancel = load_task(cfg, task.task_id).meta["revision"]
    events = []
    monkeypatch.setattr(
        "qqtools.plugins.qexp.scheduler.dispatch_task_lifecycle_hooks_noexcept",
        lambda _cfg, event: events.append(event),
    )

    cancelled = cancel(cfg, task.task_id)

    assert cancelled.meta["revision"] == revision_before_cancel + 1
    assert len(events) == 1
    assert events[0].task_revision == cancelled.meta["revision"]


def test_cancel_running_task_persists_termination_intent_by_default(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    cancelled = cancel(cfg, task.task_id)
    assert cancelled.state["projection"] == "running"
    assert cancelled.control["terminate_running"] is True


def test_blocked_orphan_retry_supersedes_orphan_and_records_audit_event(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert expire_claim(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)

    queued = retry(cfg, task.task_id)

    current = AttemptRecord.from_dict(read_json(attempt_path(
        cfg.shared_root, task.task_id, attempt.attempt_number
    )))
    events = [read_json(path) for path in shared_paths(cfg.shared_root)["events"].glob("*/*.json")]
    audit_event = next(event for event in events if event["event_type"] == "orphan_superseded_by_retry")
    assert queued.state == {"projection": "queued", "reason": "orphan_superseded_by_retry"}
    assert queued.claim_control["active_claim"] is None
    assert queued.claim_control["fencing_epoch"] == attempt.current_fencing_token + 1
    assert queued.attempt_control["current_attempt_id"] is None
    assert queued.attempt_control["current_attempt_number"] == attempt.attempt_number
    assert current.phase == "orphaned"
    assert audit_event["task_id"] == task.task_id
    assert audit_event["details"]["attempt_id"] == attempt.attempt_id
    assert audit_event["details"]["fencing_token"] == attempt.current_fencing_token
    assert audit_event["details"]["operator"] == cfg.machine_name
    assert isinstance(audit_event["details"]["timestamp"], str)


def test_agent_does_not_acknowledge_unconfirmed_termination(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    manifest_path = cfg.runtime_root / "processes" / f"{attempt.attempt_id}.json"
    atomic_replace(manifest_path, {"process": {"task_id": task.task_id,
        "attempt_id": attempt.attempt_id, "fencing_token": attempt.current_fencing_token,
        "process_group_id": 4321, "observed_state": "running", "supervisor": "agent"}})
    cancel(cfg, task.task_id)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.scheduler._process_evidence_state", lambda *args: "alive"
    )
    monkeypatch.setattr(
        "qqtools.plugins.qexp.scheduler._terminate_process_group", lambda pid: False
    )
    monkeypatch.setattr("qqtools.plugins.qexp.scheduler.renew_attempt_lease", lambda *args: True)
    reconcile_running_tasks(cfg)
    stored = load_task(cfg, task.task_id)
    assert stored.state["projection"] == "running"
    assert stored.control["termination_acknowledged_at"] is None


def test_agent_acknowledges_termination_only_after_process_disappears(
        tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    manifest_path = cfg.runtime_root / "processes" / f"{attempt.attempt_id}.json"
    atomic_replace(manifest_path, {"process": {"task_id": task.task_id,
        "attempt_id": attempt.attempt_id, "fencing_token": attempt.current_fencing_token,
        "process_group_id": 4321, "observed_state": "running", "supervisor": "agent"}})
    cancel(cfg, task.task_id)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.scheduler._process_evidence_state", lambda *args: "alive"
    )
    monkeypatch.setattr("qqtools.plugins.qexp.scheduler._terminate_process_group", lambda pid: True)
    reconcile_running_tasks(cfg)
    stored = load_task(cfg, task.task_id)
    assert stored.state["projection"] == "cancelled"
    assert stored.control["termination_result"] == "terminated"
    assert stored.control["termination_acknowledged_at"] is not None
    assert reserved_gpu_ids(cfg.runtime_root) == set()


def test_agent_finalizes_missing_recovered_process_and_releases_gpu(
        tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    manifest_path = cfg.runtime_root / "processes" / f"{attempt.attempt_id}.json"
    atomic_replace(manifest_path, {"process": {"task_id": task.task_id,
        "attempt_id": attempt.attempt_id, "fencing_token": attempt.current_fencing_token,
        "process_group_id": 4321, "observed_state": "running", "supervisor": "agent"}})
    monkeypatch.setattr(
        "qqtools.plugins.qexp.scheduler._process_evidence_state", lambda *args: "absent"
    )
    reconcile_running_tasks(cfg)
    stored = load_task(cfg, task.task_id)
    assert stored.state == {"projection": "failed", "reason": "process_exited_without_status"}
    assert reserved_gpu_ids(cfg.runtime_root) == set()


def test_agent_termination_escalates_when_sigterm_does_not_stop_process(monkeypatch):
    signals: list[int] = []
    monkeypatch.setattr("qqtools.plugins.qexp.scheduler._is_process_group_alive", lambda pid: True)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.scheduler.os.killpg",
        lambda pid, sent_signal: signals.append(sent_signal),
    )
    assert _terminate_process_group(4321, grace_seconds=0) is False
    assert signals == [signal.SIGTERM, signal.SIGKILL]


def test_reused_wrapper_pid_is_not_treated_as_runner(monkeypatch):
    monkeypatch.setattr("qqtools.plugins.qexp.scheduler._is_process_alive", lambda pid: True)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.scheduler._process_start_time_ticks", lambda pid: 222
    )
    manifest = {"wrapper_pid": 4321, "wrapper_start_time_ticks": 111}
    assert _manifest_supervisor(manifest) == "agent"


def test_reused_process_group_pid_is_not_renewed_or_signalled(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "ok"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    path = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    attempt_data = read_json(path)
    attempt_data["attempt"]["process"].update({"process_group_id": 4321,
        "process_group_start_time_ticks": 111})
    atomic_replace(path, attempt_data)
    manifest_path = cfg.runtime_root / "processes" / f"{attempt.attempt_id}.json"
    atomic_replace(manifest_path, {"process": {"task_id": task.task_id,
        "attempt_id": attempt.attempt_id, "fencing_token": attempt.current_fencing_token,
        "process_group_id": 4321, "process_group_start_time_ticks": 111,
        "observed_state": "running", "supervisor": "agent"}})
    monkeypatch.setattr(
        "qqtools.plugins.qexp.scheduler._process_start_time_ticks", lambda pid: 222
    )
    monkeypatch.setattr(
        "qqtools.plugins.qexp.scheduler._is_process_group_alive",
        lambda pid: pytest.fail("reused process group must not be probed or signalled"),
    )
    monkeypatch.setattr(
        "qqtools.plugins.qexp.scheduler.renew_attempt_lease",
        lambda *args: pytest.fail("reused process group must not be renewed"),
    )
    reconcile_running_tasks(cfg)
    assert load_task(cfg, task.task_id).state["projection"] == "running"
