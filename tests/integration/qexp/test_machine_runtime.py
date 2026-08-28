import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Barrier, Event, Thread

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.commands.cleanup import clean
from qqtools.plugins.qexp.commands.task import cancel, submit
from qqtools.plugins.qexp.machine_agent import (
    _MachineControlPlane,
    _publish_project_snapshots,
    dispatch_machine_cycle,
    dispatch_machine_cycle_locked,
    ensure_machine_agent_started,
    get_machine_agent_status,
    migrate_project,
    restart_machine_agent,
    run_machine_agent_loop,
    start_machine_agent,
    stop_machine_agent,
)
from qqtools.plugins.qexp.machine_runtime import (
    MACHINE_RUNTIME_ENV,
    MachineRuntime,
    ProjectBinding,
    resolve_machine_runtime_root,
)
from qqtools.plugins.qexp.project_maintenance import maintain_project
from qqtools.plugins.qexp.runtime.locks import exclusive
from qqtools.plugins.qexp.runtime.paths import machine_project_paths, machine_runtime_paths
from qqtools.plugins.qexp.runtime.reservations import active_reservations, attach, reserve, reserved_gpu_ids
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]

def test_resolve_machine_runtime_root_prefers_explicit_override_and_uses_safe_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = tmp_path / "configured-runtime"
    environment = tmp_path / "environment-runtime"
    monkeypatch.setenv(MACHINE_RUNTIME_ENV, str(environment))

    assert resolve_machine_runtime_root(configured) == configured.resolve()
    assert not configured.exists()

    assert resolve_machine_runtime_root() == environment.resolve()
    assert not environment.exists()

    monkeypatch.delenv(MACHINE_RUNTIME_ENV)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    expected_default = tmp_path / "home" / ".qqtools" / "qexp-machine"
    assert resolve_machine_runtime_root() == expected_default
    assert not expected_default.exists()

    runtime = MachineRuntime(configured)
    runtime.ensure_layout()
    assert configured.is_dir()


def test_resolve_machine_runtime_root_rejects_project_and_file_roots(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="project .qexp root"):
        resolve_machine_runtime_root(tmp_path / ".qexp")

    file_root = tmp_path / "runtime-file"
    file_root.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a directory"):
        resolve_machine_runtime_root(file_root)


def test_machine_project_paths_are_isolated_and_validate_project_ids(tmp_path: Path) -> None:
    root = tmp_path / "machine-runtime"
    runtime_paths = machine_runtime_paths(root)
    first = machine_project_paths(root, "project-one")
    second = machine_project_paths(root, "project-two")

    assert runtime_paths["root"] == root.resolve()
    assert first["root"] == root.resolve() / "projects" / "project-one"
    assert second["root"] == root.resolve() / "projects" / "project-two"
    assert first["processes"] != second["processes"]
    assert first["clock_health"].parent == first["agent"]

    with pytest.raises(ValueError, match="project_id is invalid"):
        machine_project_paths(root, "../escape")


def test_registry_add_list_disable_and_remove_project_binding(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime")
    runtime = MachineRuntime(tmp_path / "machine-runtime")

    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    revision, bindings = runtime.load_registry()
    assert revision == 1
    assert bindings == [binding]
    assert runtime.binding_state(binding) == "enabled"

    disabled = runtime.set_enabled(binding.project_id, False)
    revision, bindings = runtime.load_registry()
    assert revision == 2
    assert bindings == [disabled]
    assert runtime.binding_state(disabled) == "disabled"

    removed = runtime.remove_binding(disabled.shared_root)
    assert removed == disabled
    assert runtime.load_registry() == (3, [])


def test_remove_project_deletes_its_disposable_runtime_partition(tmp_path: Path) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime"
    )
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    project_root = runtime.project_paths(binding.project_id)["root"]
    atomic_replace(project_root / "diagnostic.json", {"diagnostic": {}})

    disabled = runtime.set_enabled(binding.project_id, False)
    runtime.remove_binding(disabled.project_id)

    assert not project_root.exists()


def test_migrate_project_imports_legacy_reservation_and_marks_machine_runtime(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime")
    record_path = cfg.shared_root / "machines" / cfg.machine_name / "machine.json"
    record = read_json(record_path)
    record["machine"].pop("agent_runtime")
    atomic_replace(record_path, record)
    provisional = reserve(cfg.runtime_root, "legacy-task", [0])
    runtime = MachineRuntime(tmp_path / "machine-runtime")

    binding = migrate_project(runtime, cfg)

    assert binding.enabled
    assert reserved_gpu_ids(runtime.root) == {0}
    assert reserved_gpu_ids(cfg.runtime_root) == set()
    imported = read_json(runtime.paths["provisional"] / f"{provisional['reservation']['reservation_id']}.json")
    assert imported["reservation"]["project_id"] == binding.project_id
    assert read_json(record_path)["machine"]["agent_runtime"] == "machine"


def test_migration_moves_agent_evidence_and_drains_only_late_runner_records(
    tmp_path: Path,
) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime"
    )
    record_path = cfg.shared_root / "machines" / cfg.machine_name / "machine.json"
    record = read_json(record_path)
    record["machine"].pop("agent_runtime")
    atomic_replace(record_path, record)
    source_process = cfg.runtime_root / "processes" / "attempt-1.json"
    source_decision = cfg.runtime_root / "termination-decisions" / "attempt-1" / "decision.json"
    source_registration = cfg.runtime_root / "process-registrations" / "attempt-1.json"
    atomic_replace(source_process, {"process": {"attempt_id": "attempt-1"}})
    atomic_replace(source_decision, {"termination_decision": {"attempt_id": "attempt-1"}})
    atomic_replace(source_registration, {"process_registration": {"attempt_id": "attempt-1"}})
    runtime = MachineRuntime(tmp_path / "machine-runtime")

    binding = migrate_project(runtime, cfg)
    destination = runtime.project_paths(binding.project_id)

    assert not source_process.exists()
    assert not source_decision.exists()
    assert not source_registration.exists()
    assert (destination["processes"] / "attempt-1.json").exists()
    assert (destination["termination_decisions"] / "attempt-1" / "decision.json").exists()
    assert (destination["registrations"] / "attempt-1.json").exists()

    late_observation = cfg.runtime_root / "process-observations" / "attempt-1.json"
    atomic_replace(late_observation, {"exit_observation": {"attempt_id": "attempt-1"}})
    runtime.drain_legacy_runner_evidence(binding)

    assert not late_observation.exists()
    assert (destination["observations"] / "attempt-1.json").exists()

    (destination["processes"] / "attempt-1.json").unlink()
    runtime.drain_legacy_runner_evidence(binding)
    assert not (destination["processes"] / "attempt-1.json").exists()


def test_project_snapshots_preserve_agent_start_and_continuous_idle_times(tmp_path: Path) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "project-runtime"
    )
    readable = {"project-1": cfg}
    agent_path = cfg.shared_root / "machines" / cfg.machine_name / "state" / "agent.json"

    _publish_project_snapshots(
        readable,
        instance_id="agent-1",
        pid=123,
        visible=[0],
        reservations=[],
        started_at="2026-08-27T00:00:00Z",
    )
    first = read_json(agent_path)["agent"]
    _publish_project_snapshots(
        readable,
        instance_id="agent-1",
        pid=123,
        visible=[0],
        reservations=[],
        started_at="2026-08-27T00:00:00Z",
    )
    second = read_json(agent_path)["agent"]

    assert first["started_at"] == second["started_at"] == "2026-08-27T00:00:00Z"
    assert first["idle_since_at"] == second["idle_since_at"]


def test_migrate_project_preserves_intentionally_disabled_active_binding(tmp_path: Path) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime"
    )
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name, enabled=False)
    atomic_replace(
        runtime.migration_path(binding.project_id),
        {
            "migration": {
                "legacy_runtime_root": str(cfg.runtime_root),
                "project_id": binding.project_id,
                "shared_root": str(cfg.shared_root),
                "machine_name": cfg.machine_name,
                "state": "active",
                "prepared_at": "2026-01-01T00:00:00Z",
            }
        },
    )

    migrated = migrate_project(runtime, cfg)

    assert not migrated.enabled
    assert runtime.matching_binding(cfg) == migrated


def test_migration_waits_for_legacy_reservation_lock_before_import(tmp_path: Path) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime"
    )
    record_path = cfg.shared_root / "machines" / cfg.machine_name / "machine.json"
    record = read_json(record_path)
    record["machine"].pop("agent_runtime")
    atomic_replace(record_path, record)
    source = reserve(cfg.runtime_root, "legacy-task", [0])
    reservation_id = source["reservation"]["reservation_id"]
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    errors: list[Exception] = []

    def run_migration() -> None:
        try:
            migrate_project(runtime, cfg)
        except Exception as exc:  # pragma: no cover - asserted after joining the thread
            errors.append(exc)

    source_lock = cfg.runtime_root / "locks" / "gpu-reservations.lock"
    with exclusive(source_lock):
        thread = Thread(target=run_migration)
        thread.start()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            binding = runtime.matching_binding(cfg)
            if binding is not None and runtime.migration_path(binding.project_id).exists():
                state = read_json(runtime.migration_path(binding.project_id))["migration"]["state"]
                if state == "legacy_agent_stopped":
                    break
            time.sleep(0.01)
        else:
            pytest.fail("migration did not reach the source reservation lock")
        assert not (runtime.paths["provisional"] / f"{reservation_id}.json").exists()
        source_path = (
            cfg.runtime_root / "reservations" / "provisional" / f"{reservation_id}.json"
        )
        assert source_path.exists()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert errors == []
    assert (runtime.paths["provisional"] / f"{reservation_id}.json").exists()


def test_migration_disables_binding_when_final_state_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import qqtools.plugins.qexp.machine_agent as machine_agent

    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime"
    )
    record_path = cfg.shared_root / "machines" / cfg.machine_name / "machine.json"
    record = read_json(record_path)
    record["machine"].pop("agent_runtime")
    atomic_replace(record_path, record)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    original_save = machine_agent._save_migration_state

    def fail_active_state(*args: object, state: str, **kwargs: object) -> None:
        if state == "active":
            raise OSError("injected final state failure")
        original_save(*args, state=state, **kwargs)

    monkeypatch.setattr(machine_agent, "_save_migration_state", fail_active_state)

    with pytest.raises(OSError, match="injected final state failure"):
        migrate_project(runtime, cfg)

    binding = runtime.matching_binding(cfg)
    assert binding is not None and not binding.enabled
    assert read_json(runtime.migration_path(binding.project_id))["migration"]["state"] == "blocked"


def test_legacy_agent_stop_treats_reused_pid_as_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qqtools.plugins.qexp.layout import runtime_pid_path
    from qqtools.plugins.qexp.machine_agent import _stop_verified_legacy_agent

    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime"
    )
    pid_path = runtime_pid_path(cfg)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text("1234", encoding="utf-8")
    start_ticks = iter([100, 200, 200])
    monkeypatch.setattr(
        "qqtools.plugins.qexp.machine_agent.get_agent_status",
        lambda _cfg: {"pid": 1234, "is_running": True},
    )
    monkeypatch.setattr(
        "qqtools.plugins.qexp.machine_agent._legacy_pid_matches",
        lambda _cfg, _pid: True,
    )
    monkeypatch.setattr(
        "qqtools.plugins.qexp.machine_agent._pid_start_time_ticks",
        lambda _pid: next(start_ticks),
    )
    monkeypatch.setattr("qqtools.plugins.qexp.machine_agent.os.kill", lambda _pid, _signal: None)

    assert _stop_verified_legacy_agent(cfg, timeout=0.0) == 1234
    assert not pid_path.exists()


def test_migration_reservation_conflict_keeps_source_and_disables_project(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime")
    record_path = cfg.shared_root / "machines" / cfg.machine_name / "machine.json"
    record = read_json(record_path)
    record["machine"].pop("agent_runtime")
    atomic_replace(record_path, record)
    provisional = reserve(cfg.runtime_root, "legacy-task", [0])
    reservation_id = provisional["reservation"]["reservation_id"]
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.ensure_layout()
    atomic_replace(
        runtime.paths["provisional"] / f"{reservation_id}.json",
        {"reservation": {"reservation_id": reservation_id, "task_id": "other-task", "gpu_ids": [1]}},
    )

    with pytest.raises(RuntimeError, match="conflicts during migration"):
        migrate_project(runtime, cfg)

    binding = runtime.matching_binding(cfg)
    assert binding is not None and not binding.enabled
    assert (cfg.runtime_root / "reservations" / "provisional" / f"{reservation_id}.json").exists()
    assert read_json(runtime.migration_path(binding.project_id))["migration"]["state"] == "blocked"


def test_migration_gpu_conflict_keeps_source_and_disables_project(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime")
    record_path = cfg.shared_root / "machines" / cfg.machine_name / "machine.json"
    record = read_json(record_path)
    record["machine"].pop("agent_runtime")
    atomic_replace(record_path, record)
    source = reserve(cfg.runtime_root, "legacy-task", [0])
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    reserve(runtime.root, "other-task", [0], project_id="other-project")

    with pytest.raises(RuntimeError, match="GPUs conflict during migration"):
        migrate_project(runtime, cfg)

    binding = runtime.matching_binding(cfg)
    assert binding is not None and not binding.enabled
    assert (cfg.runtime_root / "reservations" / "provisional" / f"{source['reservation']['reservation_id']}.json").exists()


def test_machine_recovery_retags_global_reservation(tmp_path: Path) -> None:
    from qqtools.plugins.qexp.runtime.recovery import recover_running_attempt
    from qqtools.plugins.qexp.scheduler import authorize_launch, claim_task

    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime")
    task = submit(cfg, ["echo", "ok"])
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    context = runtime.execution_context(cfg)
    attempt = claim_task(
        context.local_cfg,
        task.task_id,
        [0],
        reservation_runtime_root=runtime.root,
        project_id=binding.project_id,
    )
    assert attempt is not None
    assert authorize_launch(
        context.local_cfg,
        task.task_id,
        attempt.attempt_id,
        attempt.current_fencing_token,
        reservation_runtime_root=runtime.root,
    )
    process_path = context.local_cfg.runtime_root / "processes" / f"{attempt.attempt_id}.json"
    atomic_replace(process_path, {"process": {
        "task_id": task.task_id,
        "attempt_id": attempt.attempt_id,
        "fencing_token": attempt.current_fencing_token,
            "process_group_id": 9876,
            "observed_state": "running",
        }})
    attempt_path = cfg.shared_root / "attempts" / task.task_id / "1.json"
    attempt_value = read_json(attempt_path)
    attempt_value["attempt"]["phase"] = "orphaned"
    atomic_replace(attempt_path, attempt_value)
    task_value = load_task(cfg, task.task_id)
    task_value.state.update({"projection": "blocked", "reason": "test"})
    task_value.claim_control["active_claim"] = None
    atomic_replace(cfg.shared_root / "tasks" / f"{task.task_id}.json", task_value.to_dict())

    token = recover_running_attempt(
        context.local_cfg,
        task.task_id,
        attempt.attempt_id,
        attempt.current_fencing_token,
        reservation_runtime_root=runtime.root,
    )

    assert token is not None
    assert active_reservations(runtime.root)[0]["fencing_token"] == token


def test_registry_rejects_duplicate_project_id_and_shared_root(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    first = init_shared_root(tmp_path / "first" / ".qexp", "gpu-1", runtime_root=tmp_path / "first-runtime")
    second = init_shared_root(tmp_path / "second" / ".qexp", "gpu-1", runtime_root=tmp_path / "second-runtime")
    binding = runtime.add_binding(first.shared_root, first.machine_name)

    identity_path = second.shared_root / "project" / "identity.json"
    atomic_replace(identity_path, {"project": {"project_id": binding.project_id, "shared_root": str(second.shared_root)}})
    with pytest.raises(ValueError, match="already registered"):
        runtime.add_binding(second.shared_root, second.machine_name)

    atomic_replace(identity_path, {"project": {"project_id": "other-project", "shared_root": str(second.shared_root)}})
    first_identity_path = first.shared_root / "project" / "identity.json"
    atomic_replace(first_identity_path, {"project": {"project_id": "other-project", "shared_root": str(first.shared_root)}})
    with pytest.raises(ValueError, match="project root.*already registered"):
        runtime.add_binding(first.shared_root, first.machine_name)


def test_disabled_binding_is_draining_while_project_evidence_exists(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime")
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    disabled = runtime.set_enabled(binding.project_id, False)

    process_path = runtime.project_paths(disabled.project_id)["processes"] / "process-1.json"
    atomic_replace(process_path, {"process": {"process_id": "process-1"}})

    assert runtime.binding_state(disabled) == "draining"
    with pytest.raises(RuntimeError, match="active local evidence"):
        runtime.remove_binding(disabled.project_id)

    process_path.unlink()
    decision_path = (
        runtime.project_paths(disabled.project_id)["termination_decisions"]
        / "attempt-1"
        / "decision-1.json"
    )
    atomic_replace(decision_path, {"termination_decision": {"state": "signal_committed"}})
    assert runtime.binding_state(disabled) == "draining"
    with pytest.raises(RuntimeError, match="active local evidence"):
        runtime.remove_binding(disabled.project_id)

    decision_path.unlink()
    assert runtime.binding_state(disabled) == "disabled"
    assert runtime.remove_binding(disabled.project_id) == disabled


class _RecordingExecutor:
    def __init__(self) -> None:
        self.launched: list[tuple[str, str]] = []

    def launch_attempt(self, _cfg, task_id, attempt) -> None:
        self.launched.append((task_id, attempt.attempt_id))


class _FailingExecutor:
    def launch_attempt(self, _cfg, _task_id, _attempt) -> None:
        raise RuntimeError("launch failed")


def test_machine_dispatch_uses_one_reservation_root_and_each_task_preflight(tmp_path: Path) -> None:
    first_dir = tmp_path / "first-work"
    second_dir = tmp_path / "second-work"
    first_dir.mkdir()
    second_dir.mkdir()
    first = init_shared_root(tmp_path / "first" / ".qexp", "gpu-1", runtime_root=tmp_path / "first-runtime")
    second = init_shared_root(tmp_path / "second" / ".qexp", "gpu-1", runtime_root=tmp_path / "second-runtime")
    first_task = submit(first, ["echo", "first"], working_dir=first_dir)
    second_task = submit(second, ["echo", "second"], working_dir=second_dir)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    first_binding = runtime.add_binding(first.shared_root, first.machine_name)
    second_binding = runtime.add_binding(second.shared_root, second.machine_name)
    executor = _RecordingExecutor()

    first_results = dispatch_machine_cycle(runtime, available_gpus=[0, 1], executor=executor)

    assert {item["project_id"] for item in first_results} == {
        first_binding.project_id, second_binding.project_id,
    }
    assert {task_id for task_id, _ in executor.launched} == {first_task.task_id, second_task.task_id}
    assert reserved_gpu_ids(runtime.root) == {0, 1}
    assert reserved_gpu_ids(runtime.project_paths(first_binding.project_id)["root"]) == set()
    assert reserved_gpu_ids(runtime.project_paths(second_binding.project_id)["root"]) == set()
    reservations = active_reservations(runtime.root)
    assert {(item["project_id"], item["shared_root"], item["machine_name"])
            for item in reservations} == {
        (first_binding.project_id, str(first.shared_root), first.machine_name),
        (second_binding.project_id, str(second.shared_root), second.machine_name),
    }

    missing_task = submit(first, ["echo", "missing"], working_dir=tmp_path / "missing")
    dispatch_machine_cycle(runtime, available_gpus=[2], executor=executor)
    diagnostic = read_json(runtime.paths["diagnostics"] / f"bad-task-spec-{first_binding.project_id}-{missing_task.task_id}.json")
    assert diagnostic["machine_diagnostic"] == {
        "kind": "bad_task_spec_working_directory", "project_id": first_binding.project_id,
        "task_id": missing_task.task_id, "path": str(tmp_path / "missing"), "reason": "missing",
    }


def test_managed_task_cancel_releases_machine_reservation(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime")
    task = submit(cfg, ["echo", "ok"])
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.add_binding(cfg.shared_root, cfg.machine_name)
    monkeypatch.setenv(MACHINE_RUNTIME_ENV, str(runtime.root))

    from qqtools.plugins.qexp.scheduler import claim_task

    attempt = claim_task(cfg, task.task_id, [0], reservation_runtime_root=runtime.root)
    assert attempt is not None
    assert reserved_gpu_ids(runtime.root) == {0}

    cancelled = cancel(cfg, task.task_id)

    assert cancelled.state["projection"] == "cancelled"
    assert reserved_gpu_ids(runtime.root) == set()
    assert reserved_gpu_ids(cfg.runtime_root) == set()


def test_managed_launch_failure_releases_machine_reservation(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime")
    submit(cfg, ["echo", "ok"])
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.add_binding(cfg.shared_root, cfg.machine_name)

    results = dispatch_machine_cycle(runtime, available_gpus=[0], executor=_FailingExecutor())

    assert results[0]["status"] == "dispatched"
    assert reserved_gpu_ids(runtime.root) == set()
    assert reserved_gpu_ids(cfg.runtime_root) == set()


def test_machine_dispatch_isolates_unreadable_roots_and_publishes_project_views(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    first = init_shared_root(tmp_path / "first" / ".qexp", "gpu-1", runtime_root=tmp_path / "first-runtime")
    second = init_shared_root(tmp_path / "second" / ".qexp", "gpu-1", runtime_root=tmp_path / "second-runtime")
    task = submit(second, ["echo", "second"], working_dir=work_dir)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    first_binding = runtime.add_binding(first.shared_root, first.machine_name)
    second_binding = runtime.add_binding(second.shared_root, second.machine_name)
    original = __import__("qqtools.plugins.qexp.machine_agent", fromlist=["_binding_config"])._binding_config

    def unreadable_first(machine_runtime, binding):
        if binding == first_binding:
            raise RuntimeError("root unreadable")
        return original(machine_runtime, binding)

    monkeypatch.setattr("qqtools.plugins.qexp.machine_agent._binding_config", unreadable_first)
    results = dispatch_machine_cycle(runtime, available_gpus=[0], executor=_RecordingExecutor())

    assert results[0]["project_id"] == first_binding.project_id
    assert results[0]["status"] == "error"
    assert results[1] == {"project_id": second_binding.project_id, "launched": [task.task_id], "status": "dispatched"}
    second_state = second.shared_root / "machines" / "gpu-1" / "state"
    second_summary = read_json(second_state / "summary.json")["summary"]
    assert second_summary["machine_reservation_count"] == 1
    assert len(second_summary["machine_reservation_ids"]) == 1
    assert read_json(second_state / "agent.json")["agent"]["active_attempt_ids"] == [f"{task.task_id}-attempt-1"]


@pytest.mark.parametrize("failure_stage", ["legacy_drain", "maintenance", "supervision"])
def test_machine_dispatch_does_not_claim_after_project_precondition_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime"
    )
    task = submit(cfg, ["echo", "queued"], working_dir=tmp_path)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    executor = _RecordingExecutor()

    def fail(*_args, **_kwargs):
        raise RuntimeError(f"{failure_stage} failed")

    if failure_stage == "legacy_drain":
        monkeypatch.setattr(runtime, "drain_legacy_runner_evidence", fail)
    elif failure_stage == "maintenance":
        monkeypatch.setattr("qqtools.plugins.qexp.machine_agent.maintain_project", fail)
    else:
        monkeypatch.setattr("qqtools.plugins.qexp.machine_agent.AuthoritySupervisor.tick", fail)

    results = dispatch_machine_cycle(runtime, available_gpus=[0], executor=executor)

    assert results == [{
        "project_id": binding.project_id,
        "launched": [],
        "status": "error",
        "error": f"{failure_stage} failed",
    }]
    assert load_task(cfg, task.task_id).state["projection"] == "queued"
    assert executor.launched == []
    assert active_reservations(runtime.root) == []


def test_machine_dispatch_supervises_draining_but_revalidates_enabled_claims(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime")
    task = submit(cfg, ["echo", "queued"], working_dir=work_dir)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    disabled = runtime.set_enabled(binding.project_id, False)
    atomic_replace(runtime.project_paths(disabled.project_id)["processes"] / "active.json", {"process": {}})
    maintained: list[Path] = []
    monkeypatch.setattr(
        "qqtools.plugins.qexp.machine_agent.maintain_project",
        lambda value, **_kwargs: maintained.append(value.shared_root),
    )
    monkeypatch.setattr("qqtools.plugins.qexp.machine_agent.AuthoritySupervisor.tick", lambda _self: None)

    assert dispatch_machine_cycle(runtime, available_gpus=[0], executor=_RecordingExecutor()) == []
    assert maintained == [cfg.shared_root]
    assert not active_reservations(runtime.root)

    runtime.set_enabled(disabled.project_id, True)
    @contextmanager
    def deny_claim(_binding):
        yield False

    monkeypatch.setattr(runtime, "enabled_claim_guard", deny_claim)
    dispatch_machine_cycle(runtime, available_gpus=[0], executor=_RecordingExecutor())
    assert not active_reservations(runtime.root)
    assert read_json(cfg.shared_root / "tasks" / f"{task.task_id}.json")["task"]["state"]["projection"] == "queued"


def test_machine_dispatch_reuses_supervisor_for_each_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime"
    )
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    created: list[object] = []

    class RecordingSupervisor:
        def __init__(self, _cfg, *, reservation_runtime_root):
            self.reservation_runtime_root = reservation_runtime_root
            self.ticks = 0
            created.append(self)

        def tick(self) -> None:
            self.ticks += 1

    monkeypatch.setattr("qqtools.plugins.qexp.machine_agent.AuthoritySupervisor", RecordingSupervisor)
    supervisors = {}

    dispatch_machine_cycle_locked(runtime, available_gpus=[], supervisors=supervisors)
    dispatch_machine_cycle_locked(runtime, available_gpus=[], supervisors=supervisors)

    assert len(created) == 1
    assert supervisors[binding.project_id] is created[0]
    assert created[0].ticks == 2


def test_machine_project_paths_match_the_local_runtime_layout(tmp_path: Path) -> None:
    root = tmp_path / "machine-runtime"
    paths = machine_project_paths(root, "project-one")

    from qqtools.plugins.qexp.runtime.paths import local_paths

    project_root = root.resolve() / "projects" / "project-one"
    assert paths == {"root": project_root, **local_paths(project_root)}


def test_project_binding_rejects_non_boolean_enabled(tmp_path: Path) -> None:
    value = {"project_id": "project-one", "shared_root": str(tmp_path), "machine_name": "gpu-1", "enabled": 1}

    with pytest.raises(ValueError, match="enabled must be a bool"):
        ProjectBinding.from_dict(value)


def test_machine_dispatch_continues_after_a_project_dispatch_error(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = init_shared_root(tmp_path / "first" / ".qexp", "gpu-1", runtime_root=tmp_path / "first-runtime")
    second = init_shared_root(tmp_path / "second" / ".qexp", "gpu-1", runtime_root=tmp_path / "second-runtime")
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    first_binding = runtime.add_binding(first.shared_root, first.machine_name)
    second_binding = runtime.add_binding(second.shared_root, second.machine_name)

    def fail_first(cfg, **_kwargs):
        if cfg.shared_root == first.shared_root:
            raise RuntimeError("dispatch failed")
        return []

    monkeypatch.setattr("qqtools.plugins.qexp.machine_agent.run_dispatch_cycle", fail_first)

    results = dispatch_machine_cycle(runtime, available_gpus=[0], executor=_RecordingExecutor())
    result_by_project = {result["project_id"]: result for result in results}

    assert result_by_project[first_binding.project_id] == {
        "project_id": first_binding.project_id,
        "launched": [],
        "status": "error",
        "error": "dispatch failed",
    }
    assert result_by_project[second_binding.project_id] == {
        "project_id": second_binding.project_id,
        "launched": [],
        "status": "dispatched",
    }


def test_machine_cleanup_keeps_other_project_reservation_with_same_task_id(tmp_path: Path) -> None:
    first = init_shared_root(tmp_path / "first" / ".qexp", "gpu-1", runtime_root=tmp_path / "first-runtime")
    second = init_shared_root(tmp_path / "second" / ".qexp", "gpu-1", runtime_root=tmp_path / "second-runtime")
    task_id = "shared-task-id"
    first_task = submit(first, ["echo", "first"], task_id=task_id)
    second_task = submit(second, ["echo", "second"], task_id=task_id)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    first_binding = runtime.add_binding(first.shared_root, first.machine_name)
    second_binding = runtime.add_binding(second.shared_root, second.machine_name)
    for cfg, task in ((first, first_task), (second, second_task)):
        task_data = read_json(cfg.shared_root / "tasks" / f"{task.task_id}.json")
        task_data["task"]["state"] = {"projection": "failed", "reason": "test_failure"}
        atomic_replace(cfg.shared_root / "tasks" / f"{task.task_id}.json", task_data)
    first_reservation = reserve(runtime.root, task_id, [0], project_id=first_binding.project_id)
    second_reservation = reserve(runtime.root, task_id, [1], project_id=second_binding.project_id)
    attach(runtime.root, first_reservation["reservation"]["reservation_id"], "first-attempt", 1)
    attach(runtime.root, second_reservation["reservation"]["reservation_id"], "second-attempt", 1)

    clean(first, task_id=task_id, reservation_runtime_root=runtime.root)

    assert reserved_gpu_ids(runtime.root) == {1}
    assert [item["project_id"] for item in active_reservations(runtime.root)] == [second_binding.project_id]


def test_machine_maintenance_offers_elapsed_work(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime")
    task = submit(
        cfg,
        ["echo", "ok"],
        group="exp",
        sharing_mode="spillover",
        offer_after_seconds=0,
    )
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.add_binding(cfg.shared_root, cfg.machine_name)

    dispatch_machine_cycle(runtime, available_gpus=[], executor=_RecordingExecutor())

    assert load_task(cfg, task.task_id).placement_runtime["queue_scope"] == "shared"


def test_machine_maintenance_releases_only_its_project_stale_reservations(tmp_path: Path) -> None:
    first = init_shared_root(tmp_path / "first" / ".qexp", "gpu-1", runtime_root=tmp_path / "first-runtime")
    second = init_shared_root(tmp_path / "second" / ".qexp", "gpu-1", runtime_root=tmp_path / "second-runtime")
    first_task = submit(first, ["echo", "first"])
    second_task = submit(second, ["echo", "second"])
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    first_binding = runtime.add_binding(first.shared_root, first.machine_name)
    second_binding = runtime.add_binding(second.shared_root, second.machine_name)
    first_reservation = reserve(runtime.root, first_task.task_id, [0], project_id=first_binding.project_id)
    second_reservation = reserve(runtime.root, second_task.task_id, [1], project_id=second_binding.project_id)
    attach(runtime.root, first_reservation["reservation"]["reservation_id"], "first-attempt", 1)
    attach(runtime.root, second_reservation["reservation"]["reservation_id"], "second-attempt", 1)

    maintain_project(
        first,
        reservation_runtime_root=runtime.root,
        project_id=first_binding.project_id,
    )

    assert reserved_gpu_ids(runtime.root) == {1}
    assert [item["project_id"] for item in active_reservations(runtime.root)] == [second_binding.project_id]


def test_managed_clean_checks_project_local_process_blockers(tmp_path: Path) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp",
        "gpu-1",
        runtime_root=tmp_path / "legacy-runtime",
    )
    task = submit(cfg, ["echo", "ok"])
    task_path = cfg.shared_root / "tasks" / f"{task.task_id}.json"
    task_data = read_json(task_path)
    task_data["task"]["state"] = {"projection": "failed", "reason": "test_failure"}
    atomic_replace(task_path, task_data)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    context = runtime.execution_context(cfg)
    process_path = runtime.project_paths(binding.project_id)["processes"] / "active.json"
    atomic_replace(process_path, {
        "process": {
            "task_id": task.task_id,
            "attempt_id": "active-attempt",
            "observed_state": "running",
        }
    })

    result = clean(
        context.local_cfg,
        task_id=task.task_id,
        dry_run=True,
        reservation_runtime_root=context.reservation_root,
    )

    assert result["skipped"][task.task_id] == ["local_process:active-attempt"]


def test_reserved_gpu_ids_reads_each_provisional_record_once(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_root = tmp_path / "runtime"
    reservation = reserve(runtime_root, "task-1", [0])
    reservation_path = runtime_root / "reservations" / "provisional" / f"{reservation['reservation']['reservation_id']}.json"
    from qqtools.plugins.qexp.runtime import reservations

    original_read_json = reservations.read_json
    reads = 0

    def count_reads(path):
        nonlocal reads
        if path == reservation_path:
            reads += 1
        return original_read_json(path)

    monkeypatch.setattr(reservations, "read_json", count_reads)

    assert reserved_gpu_ids(runtime_root) == {0}
    assert reads == 1


def test_background_machine_agent_publishes_pid_only_after_acquiring_authority(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    source_root = Path(__file__).parents[3] / "src"
    monkeypatch.setenv("PYTHONPATH", str(source_root))

    process = start_machine_agent(runtime)
    try:
        active = get_machine_agent_status(runtime)
        assert active["is_running"] is True
        assert active["pid"] == process.pid
        assert runtime.paths["pid"].read_text(encoding="utf-8") == str(process.pid)
    finally:
        stop_machine_agent(runtime)
        process.wait(timeout=1.0)


def test_machine_agent_loop_rejects_non_main_thread_without_publishing_identity(
    tmp_path: Path,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    errors: list[Exception] = []

    def run() -> None:
        try:
            run_machine_agent_loop(runtime, loop_interval=0.01)
        except Exception as exc:
            errors.append(exc)

    thread = Thread(target=run)
    thread.start()
    thread.join()

    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "main thread" in str(errors[0])
    assert not runtime.paths["pid"].exists()
    assert not (runtime.paths["agent"] / "status.json").exists()


def test_machine_control_heartbeat_is_not_blocked_by_slow_maintenance(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime"
    )
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.add_binding(cfg.shared_root, cfg.machine_name)
    control_plane = _MachineControlPlane(
        runtime,
        instance_id="test-agent",
        loop_interval=0.02,
        started_at="2026-01-01T00:00:00Z",
        available_gpus=[],
    )
    heartbeat_path = cfg.shared_root / "machines" / cfg.machine_name / "state" / "agent.json"
    maintenance_started = Event()

    def slow_maintenance(*_args, **_kwargs) -> None:
        maintenance_started.set()
        time.sleep(0.15)

    monkeypatch.setattr("qqtools.plugins.qexp.machine_agent.maintain_project", slow_maintenance)
    control_plane.start()
    try:
        assert heartbeat_path.exists()
        heartbeat_path.unlink()
        worker = Thread(
            target=dispatch_machine_cycle_locked,
            kwargs={
                "runtime": runtime,
                "available_gpus": [],
                "supervise": False,
                "publish_snapshots": False,
            },
        )
        worker.start()
        assert maintenance_started.wait(timeout=1.0)
        deadline = time.monotonic() + 0.1
        while not heartbeat_path.exists() and time.monotonic() < deadline:
            time.sleep(0.005)
        worker.join(timeout=1.0)
        assert not worker.is_alive()
        assert heartbeat_path.exists()
    finally:
        control_plane.stop()


def test_machine_control_authority_is_not_blocked_by_slow_project_maintenance(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = init_shared_root(
        tmp_path / "first" / ".qexp", "gpu-1", runtime_root=tmp_path / "first-runtime"
    )
    second = init_shared_root(
        tmp_path / "second" / ".qexp", "gpu-1", runtime_root=tmp_path / "second-runtime"
    )
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.add_binding(first.shared_root, first.machine_name)
    runtime.add_binding(second.shared_root, second.machine_name)
    second_tick = Event()
    maintenance_started = Event()
    class RecordingSupervisor:
        def __init__(self, cfg, *, reservation_runtime_root) -> None:
            del reservation_runtime_root
            self.cfg = cfg

        @property
        def renewal_interval_seconds(self) -> float:
            return 0.02

        def recover_startup(self) -> None:
            return None

        def tick(self) -> None:
            if self.cfg.shared_root == second.shared_root:
                second_tick.set()

    def slow_maintenance(*_args, **_kwargs) -> None:
        maintenance_started.set()
        time.sleep(0.15)

    monkeypatch.setattr("qqtools.plugins.qexp.machine_agent.AuthoritySupervisor", RecordingSupervisor)
    monkeypatch.setattr("qqtools.plugins.qexp.machine_agent.maintain_project", slow_maintenance)
    control_plane = _MachineControlPlane(
        runtime,
        instance_id="test-agent",
        loop_interval=0.2,
        started_at="2026-01-01T00:00:00Z",
        available_gpus=[],
    )
    control_plane.start()
    try:
        second_tick.clear()
        worker = Thread(
            target=dispatch_machine_cycle_locked,
            kwargs={
                "runtime": runtime,
                "available_gpus": [],
                "supervise": False,
                "publish_snapshots": False,
            },
        )
        worker.start()
        assert maintenance_started.wait(timeout=1.0)
        assert second_tick.wait(timeout=0.3)
        worker.join(timeout=1.0)
        assert not worker.is_alive()
    finally:
        control_plane.stop()


def test_restart_and_activation_share_one_lifecycle_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    barrier = Barrier(3)
    state = {"is_running": False, "active_calls": 0, "max_active_calls": 0, "next_pid": 1000}
    errors: list[Exception] = []

    def enter() -> None:
        state["active_calls"] += 1
        state["max_active_calls"] = max(state["max_active_calls"], state["active_calls"])
        time.sleep(0.03)

    def leave() -> None:
        state["active_calls"] -= 1

    def get_status(_runtime):
        return {
            "is_running": state["is_running"],
            "pid": state["next_pid"] if state["is_running"] else None,
        }

    def start(_runtime, **_kwargs):
        enter()
        try:
            state["next_pid"] += 1
            state["is_running"] = True
            return type("Process", (), {"pid": state["next_pid"]})()
        finally:
            leave()

    def stop(_runtime, *, timeout):
        del timeout
        enter()
        try:
            was_running = state["is_running"]
            state["is_running"] = False
            return was_running
        finally:
            leave()

    monkeypatch.setattr("qqtools.plugins.qexp.machine_agent.get_machine_agent_status", get_status)
    monkeypatch.setattr("qqtools.plugins.qexp.machine_agent._start_machine_agent_locked", start)
    monkeypatch.setattr("qqtools.plugins.qexp.machine_agent._stop_machine_agent_locked", stop)

    def call(operation) -> None:
        barrier.wait()
        try:
            operation(runtime)
        except Exception as exc:
            errors.append(exc)

    threads = [
        Thread(target=call, args=(restart_machine_agent,)),
        Thread(target=call, args=(ensure_machine_agent_started,)),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert errors == []
    assert state["max_active_calls"] == 1
    assert state["is_running"] is True


def test_restart_waits_for_old_agent_process_after_identity_is_cleared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    source_root = Path(__file__).parents[3] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root)
    helper = """
import os
import signal
import sys
import time

from qqtools.plugins.qexp.machine_agent import _pid_start_time_ticks
from qqtools.plugins.qexp.machine_runtime import MachineRuntime
from qqtools.plugins.qexp.runtime.store import atomic_replace

runtime = MachineRuntime(sys.argv[1])
runtime.ensure_layout()
instance_id = "delayed-stop-agent"
start_ticks = _pid_start_time_ticks(os.getpid())
is_stopping = False

def stop(_signum, _frame):
    global is_stopping
    runtime.paths["pid"].unlink(missing_ok=True)
    atomic_replace(runtime.paths["agent"] / "status.json", {
        "machine_agent": {"instance_id": instance_id, "pid": None, "state": "stopped"}
    })
    is_stopping = True

signal.signal(signal.SIGTERM, stop)
with runtime.scheduler_authority(blocking=True):
    runtime.paths["pid"].write_text(str(os.getpid()), encoding="utf-8")
    atomic_replace(runtime.paths["agent"] / "status.json", {
        "machine_agent": {
            "instance_id": instance_id,
            "pid": os.getpid(),
            "pid_start_time_ticks": start_ticks,
            "state": "active",
        }
    })
    while not is_stopping:
        time.sleep(0.01)
    time.sleep(0.3)
"""
    old_process = subprocess.Popen(
        [sys.executable, "-c", helper, str(runtime.root)],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 5.0
    while not get_machine_agent_status(runtime)["is_running"] and time.monotonic() < deadline:
        if old_process.poll() is not None:
            pytest.fail(f"delayed-stop helper exited early: {old_process.stderr.read()}")
        time.sleep(0.02)
    assert get_machine_agent_status(runtime)["pid"] == old_process.pid

    reaper = Thread(target=old_process.wait)
    reaper.start()
    started_at = time.monotonic()
    new_process = restart_machine_agent(runtime)
    try:
        assert time.monotonic() - started_at >= 0.25
        assert old_process.returncode == 0
        assert get_machine_agent_status(runtime)["pid"] == new_process.pid
    finally:
        stop_machine_agent(runtime)
        new_process.wait(timeout=1.0)
        reaper.join()


def test_machine_agent_ignores_a_reused_or_stale_pid_record(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.ensure_layout()
    runtime.paths["pid"].write_text(str(os.getpid()), encoding="utf-8")
    atomic_replace(runtime.paths["agent"] / "status.json", {
        "machine_agent": {
            "instance_id": "old-agent",
            "pid": os.getpid(),
            "pid_start_time_ticks": -1,
            "state": "active",
        }
    })

    assert get_machine_agent_status(runtime)["is_running"] is False
    assert stop_machine_agent(runtime) is False
    assert not runtime.paths["pid"].exists()
