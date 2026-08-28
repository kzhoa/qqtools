from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.machine_agent import dispatch_machine_cycle_locked
from qqtools.plugins.qexp.machine_runtime import MachineRuntime
from qqtools.plugins.qexp.runtime.reservations import (
    ReservationIdentity,
    active_reservations,
    attach,
    release_if_matches,
    reserve,
    reserved_gpu_ids,
)
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task
from qqtools.plugins.qexp.scheduler import authorize_launch, claim_task


class _RecordingExecutor:
    def __init__(self) -> None:
        self.launched: list[tuple[str, str]] = []

    def launch_attempt(self, _cfg, task_id, attempt) -> None:
        self.launched.append((task_id, attempt.attempt_id))


def test_full_capacity_skips_scheduler_work_but_runs_maintenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1")
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    reservation = reserve(
        runtime.root,
        "unknown-task",
        [0],
        attempt_id="unknown-attempt",
        fencing_token=1,
        project_id="unknown-project",
    )
    attach(runtime.root, reservation["reservation"]["reservation_id"], "unknown-attempt", 1)
    maintenance_calls: list[str] = []

    monkeypatch.setattr(
        "qqtools.plugins.qexp.machine_agent.maintain_project",
        lambda _cfg, **_kwargs: maintenance_calls.append(binding.project_id),
    )

    def fail_dispatch(*_args, **_kwargs):
        raise AssertionError("scheduler work must not run at full capacity")

    monkeypatch.setattr("qqtools.plugins.qexp.machine_agent.run_dispatch_cycle", fail_dispatch)

    results = dispatch_machine_cycle_locked(
        runtime,
        available_gpus=[0],
        supervise=False,
        publish_snapshots=False,
    )

    assert maintenance_calls == [binding.project_id]
    assert results == [{"project_id": binding.project_id, "launched": [], "status": "dispatched"}]
    assert reserved_gpu_ids(runtime.root) == {0}
    diagnostic = read_json(runtime.paths["diagnostics"] / "scheduler-cycle.json")
    assert diagnostic["scheduler_diagnostic"]["counters"]["scheduler.work.skipped_no_capacity"] == 1


def test_stale_reservation_is_released_before_capacity_gate(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1")
    stale = submit(cfg, ["echo", "stale"], working_dir=work_dir)
    ready = submit(cfg, ["echo", "ready"], working_dir=work_dir)
    stale_path = cfg.shared_root / "tasks" / f"{stale.task_id}.json"
    stale_value = read_json(stale_path)
    stale_value["task"]["state"] = {"projection": "failed", "reason": "test_failure"}
    atomic_replace(stale_path, stale_value)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    reservation = reserve(
        runtime.root,
        stale.task_id,
        [0],
        attempt_id="stale-attempt",
        fencing_token=1,
        project_id=binding.project_id,
    )
    attach(runtime.root, reservation["reservation"]["reservation_id"], "stale-attempt", 1)
    executor = _RecordingExecutor()

    results = dispatch_machine_cycle_locked(
        runtime,
        available_gpus=[0],
        executor=executor,
        supervise=False,
        publish_snapshots=False,
    )

    assert results == [
        {
            "project_id": binding.project_id,
            "launched": [ready.task_id],
            "status": "dispatched",
        }
    ]
    assert executor.launched[0][0] == ready.task_id
    assert load_task(cfg, stale.task_id).state["projection"] == "failed"
    assert active_reservations(runtime.root)[0]["task_id"] == ready.task_id


def test_identity_fenced_release_rejects_replaced_reservation(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    value = reserve(runtime_root, "task-1", [0], attempt_id="attempt-1", fencing_token=1)
    reservation_id = value["reservation"]["reservation_id"]
    attach(runtime_root, reservation_id, "attempt-1", 1)
    identity = ReservationIdentity.from_record(active_reservations(runtime_root)[0])
    path = runtime_root / "reservations" / "active" / f"{reservation_id}.json"
    replacement = read_json(path)
    replacement["reservation"]["acquisition_id"] = "replacement-acquisition"
    atomic_replace(path, replacement)

    assert release_if_matches(runtime_root, identity, "stale") is False
    assert reserved_gpu_ids(runtime_root) == {0}


def test_starting_recovery_uses_exact_active_reservation(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1")
    task = submit(cfg, ["echo", "recover"], working_dir=work_dir)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    attempt = claim_task(
        cfg,
        task.task_id,
        [0],
        reservation_runtime_root=runtime.root,
        project_id=binding.project_id,
    )
    assert attempt is not None
    assert authorize_launch(
        cfg,
        task.task_id,
        attempt.attempt_id,
        attempt.current_fencing_token,
        reservation_runtime_root=runtime.root,
    )
    executor = _RecordingExecutor()

    results = dispatch_machine_cycle_locked(
        runtime,
        available_gpus=[0],
        executor=executor,
        supervise=False,
        publish_snapshots=False,
    )

    assert results == [
        {
            "project_id": binding.project_id,
            "launched": [task.task_id],
            "status": "dispatched",
        }
    ]
    assert executor.launched == [(task.task_id, attempt.attempt_id)]
    diagnostic = read_json(runtime.paths["diagnostics"] / "scheduler-cycle.json")
    counters = diagnostic["scheduler_diagnostic"]["counters"]
    assert counters["recovery.starting.checked"] == 1
    assert counters["recovery.starting.launched"] == 1
    assert counters["scheduler.work.skipped_no_capacity"] == 1


def test_reservation_verification_error_is_fail_closed_and_project_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1")
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    reservation = reserve(
        runtime.root,
        "task-1",
        [0],
        attempt_id="attempt-1",
        fencing_token=1,
        project_id=binding.project_id,
    )
    attach(runtime.root, reservation["reservation"]["reservation_id"], "attempt-1", 1)

    def fail_verification(*_args, **_kwargs):
        raise OSError("shared root unavailable")

    monkeypatch.setattr(
        "qqtools.plugins.qexp.machine_agent.reconcile_reservation",
        fail_verification,
    )

    results = dispatch_machine_cycle_locked(
        runtime,
        available_gpus=[0],
        supervise=False,
        publish_snapshots=False,
    )

    assert results == [{"project_id": binding.project_id, "launched": [], "status": "dispatched"}]
    assert reserved_gpu_ids(runtime.root) == {0}
    diagnostic = read_json(runtime.paths["diagnostics"] / "scheduler-cycle.json")
    counters = diagnostic["scheduler_diagnostic"]["counters"]
    assert counters["recovery.reservations.errors"] == 1
    assert counters["recovery.reservations.isolated"] == 1
