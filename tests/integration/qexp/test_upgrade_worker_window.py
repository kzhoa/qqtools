from __future__ import annotations

from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.agent import project_io_worker as worker
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.runtime.dependency_projection import (
    mark_degraded,
    projected_dependents,
    read_projection_state,
)
from qqtools.plugins.qexp.runtime.locks import dependency_lock, exclusive
from qqtools.plugins.qexp.runtime.paths import shared_paths
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.upgrade import UpgradeCoordinator, upgrade_journal_path
from qqtools.plugins.qexp.runtime.upgrade.task_dependencies_v2 import TaskDependenciesV2Migration

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


# QQTOOLS-COMPAT-0023: exercise automatic windows on a retained v1 dependency projection.
@pytest.fixture
def legacy(tmp_path):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1")
    for index in range(12):
        submit(
            cfg,
            ["echo", str(index)],
            task_id=f"task-{index}",
            working_dir=tmp_path,
            depends_on_task_ids=[f"task-{index - 1}"] if index else [],
        )
    schema_path = cfg.shared_root / "schema/version.json"
    schema = read_json(schema_path)
    schema["schema"]["required_capabilities"].remove("task-dependencies-v2")
    atomic_replace(schema_path, schema)
    with dependency_lock(cfg.shared_root):
        mark_degraded(cfg, "legacy checkpoint")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    revision, bindings = runtime.load_registry()
    coordinator = UpgradeCoordinator(cfg)
    coordinator.discover()
    for _ in range(30):
        status = coordinator.status()
        if status["phase"] == "backfill":
            break
        coordinator.advance()
    assert status["phase"] == "backfill"
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    request = executor.prepare_upgrade_service(binding, revision)
    try:
        yield cfg, runtime, executor, request
    finally:
        executor.shutdown()


def _item(cfg):
    return read_json(upgrade_journal_path(cfg))["upgrade"]["migrations"]["task-dependencies-v2"]


def test_production_window_commits_one_record_slices_and_releases_lock(legacy, monkeypatch):
    cfg, runtime, executor, request = legacy
    original = UpgradeCoordinator.advance
    usages = []

    def advance(self, **kwargs):
        result = original(self, **kwargs)
        with exclusive(shared_paths(cfg.shared_root)["locks"] / "upgrade.lock", blocking=False) as acquired:
            assert acquired
        usages.append(_item(cfg)["last_slice_usage"]["records"])
        return result

    monkeypatch.setattr(UpgradeCoordinator, "advance", advance)
    monkeypatch.setattr(worker, "time", SimpleNamespace(monotonic=lambda: 0.0))
    result = worker._upgrade_service(request, runtime.root, executor.paths, [False])
    assert usages == [1] * 8
    assert result["state"] == "runnable"
    assert _item(cfg)["work_items"] == 9  # readiness plus eight backfill slices


@pytest.mark.parametrize("change", ["phase", "migration", "pause", "retry", "probe", "waiting", "repair"])
def test_continuation_rechecks_peer_changes_under_lock(legacy, monkeypatch, change):
    from contextlib import contextmanager

    from qqtools.plugins.qexp.runtime.upgrade import framework

    cfg, runtime, executor, request = legacy
    original = UpgradeCoordinator.advance
    phase = TaskDependenciesV2Migration.backfill
    phase_calls = []

    def backfill(self, context):
        phase_calls.append(1)
        return phase(self, context)

    monkeypatch.setattr(TaskDependenciesV2Migration, "backfill", backfill)
    calls = []

    @contextmanager
    def lock_with_peer(path, **kwargs):
        # Interleave after advance's pre-lock status read, before its lock acquisition.
        if len(calls) == 2:
            with exclusive(shared_paths(cfg.shared_root)["locks"] / "upgrade.lock") as acquired:
                assert acquired
                journal = read_json(upgrade_journal_path(cfg))
                item = journal["upgrade"]["migrations"]["task-dependencies-v2"]
                if change == "pause":
                    atomic_replace(
                        shared_paths(cfg.shared_root)["upgrade_pause_intent"], {"pause_intent": {"state": "requested"}}
                    )
                elif change in {"retry", "probe"}:
                    item[f"next_{change}_at"] = "2999-01-01T00:00:00Z"
                    item["next_probe_at" if change == "retry" else "next_retry_at"] = "2000-01-01T00:00:00Z"
                elif change == "phase":
                    item.update(phase="audit", phase_index=2)
                elif change == "migration":
                    item["name"] = "peer-migration"
                    journal["upgrade"]["migrations"] = {"peer-migration": item}
                else:
                    item["state"] = "waiting" if change == "waiting" else "repair_required"
                atomic_replace(upgrade_journal_path(cfg), journal)
        with exclusive(path, **kwargs) as acquired:
            yield acquired

    def advance(self, **kwargs):
        calls.append(kwargs)
        return original(self, **kwargs)

    monkeypatch.setattr(framework, "exclusive", lock_with_peer)
    monkeypatch.setattr(UpgradeCoordinator, "advance", advance)
    monkeypatch.setattr(worker, "time", SimpleNamespace(monotonic=lambda: 0.0))
    result = worker._upgrade_service(request, runtime.root, executor.paths, [False])
    assert len(calls) == 2
    assert phase_calls == [1]
    assert result["state"] == UpgradeCoordinator(cfg).status()["state"]
    if change in {"retry", "probe"}:
        assert result["next_probe_at"] == "2999-01-01T00:00:00Z"


def test_manual_advance_keeps_single_slice_and_probe_semantics(legacy):
    cfg, *_ = legacy
    journal = read_json(upgrade_journal_path(cfg))
    journal["upgrade"]["migrations"]["task-dependencies-v2"]["next_probe_at"] = "2999-01-01T00:00:00Z"
    atomic_replace(upgrade_journal_path(cfg), journal)
    before = _item(cfg)["work_items"]
    result = UpgradeCoordinator(cfg).advance()
    assert result["invocation"]["slice_committed"]
    assert _item(cfg)["work_items"] == before + 1


def test_revoked_binding_fences_next_shared_mutation(legacy, monkeypatch):
    cfg, runtime, executor, request = legacy
    original = UpgradeCoordinator.advance
    calls = []

    def advance(self, **kwargs):
        if calls:
            monkeypatch.setattr(worker, "_claim_binding_is_current", lambda *a: False)
        calls.append(kwargs)
        return original(self, **kwargs)

    monkeypatch.setattr(UpgradeCoordinator, "advance", advance)
    monkeypatch.setattr(worker, "time", SimpleNamespace(monotonic=lambda: 0.0))
    with pytest.raises(worker._BindingAuthorityChanged):
        worker._upgrade_service(request, runtime.root, executor.paths, [False])
    assert _item(cfg)["work_items"] == 2


def _wait(executor, request, *, consume=False):
    import time

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        executor.poll()
        result = executor.consume(request.request_id, request) if consume else executor.load_result(request.request_id)
        if result is not None:
            return result
        time.sleep(0.02)
    pytest.fail("upgrade worker did not publish within 15 seconds")


def _instrument_children(monkeypatch, tmp_path, body):
    import subprocess

    from qqtools.plugins.qexp.agent import project_io_executor as executor_module

    script = tmp_path / "instrument_worker.py"
    script.write_text(
        "from qqtools.plugins.qexp.agent import project_io_worker as worker\n"
        "from qqtools.plugins.qexp.runtime.upgrade import UpgradeCoordinator\n"
        + body
        + "\nraise SystemExit(worker.main())\n"
    )
    popen = subprocess.Popen

    def start(command, **kwargs):
        if len(command) > 2 and command[1:3] == ["-m", "qqtools.plugins.qexp.agent.project_io_worker"]:
            command = [command[0], str(script), *command[3:]]
        return popen(command, **kwargs)

    monkeypatch.setattr(executor_module.subprocess, "Popen", start)


def test_interrupted_committed_worker_recovers_through_ambiguous_request(legacy, monkeypatch, tmp_path):
    cfg, runtime, executor, request = legacy
    before = _item(cfg)["work_items"]
    with monkeypatch.context() as patch:
        _instrument_children(
            patch,
            tmp_path,
            """import os
original = UpgradeCoordinator.advance
def advance(self, **kwargs):
    result = original(self, **kwargs)
    if result['invocation']['slice_committed']:
        os._exit(91)
    return result
UpgradeCoordinator.advance = advance""",
        )
        assert executor.start(request.request_id) is not None
        result = _wait(executor, request)
    assert result.status == "outcome_unknown"
    assert _item(cfg)["work_items"] == before + 1
    assert executor.consume(request.request_id, request) is None
    assert executor.reset_ambiguous_upgrade_service_for_retry(request.request_id, request)
    assert executor.start(request.request_id) is not None
    assert _wait(executor, request, consume=True).status == "completed"
    coordinator = UpgradeCoordinator(cfg)
    for _ in range(100):
        if not coordinator.status()["pending"]:
            break
        coordinator.advance()
    assert not coordinator.status()["pending"]
    assert read_projection_state(cfg)["state"] == "active"
    assert projected_dependents(cfg, "task-0") == ["task-1"]


def test_controller_rotates_backlogged_projects_with_one_slot(tmp_path, monkeypatch):
    import time

    from qqtools.plugins.qexp.agent.project_io_controller import ProjectIOController

    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    for name in ("a", "b"):
        cfg = init_shared_root(tmp_path / name / ".qexp", "gpu-1")
        runtime.add_binding(cfg.shared_root, cfg.machine_name)
    revision, bindings = runtime.load_registry()
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor, monotonic=lambda: time.monotonic() + 10)
    original_view = executor.status_view

    def view():
        return {**original_view(), "capacity": 1}

    monkeypatch.setattr(executor, "status_view", view)
    _instrument_children(
        monkeypatch,
        tmp_path,
        """def service(*args):
    return {'state':'runnable','pending':True,'can_run':True,'admission_blocked':False,
            'idle_blocking':True,'next_probe_at':None}
worker._upgrade_service = service""",
    )
    admitted = []
    original_start = executor.start

    def start(request_id, **kwargs):
        result = original_start(request_id, **kwargs)
        if result is not None:
            admitted.append(result.request.project_id)
        return result

    monkeypatch.setattr(executor, "start", start)
    try:
        deadline = time.monotonic() + 15
        while len(admitted) < 6 and time.monotonic() < deadline:
            controller.advance_upgrade_work(bindings, revision)
            # Make both backlogs eligible after each completed window without waiting on wall time.
            controller._upgrade_due.clear()
            time.sleep(0.02)
        assert len(admitted) >= 6
        assert admitted[:6] == [admitted[0], admitted[1]] * 3
        assert admitted[0] != admitted[1]
    finally:
        executor.shutdown()


@pytest.mark.stress
def test_daemon_window_throughput_from_identical_checkpoint(legacy, monkeypatch, tmp_path):
    """Compare the former one-call policy with bounded windows on identical data."""
    import json
    import shutil
    import time

    from qqtools.plugins.qexp.agent.project_io_controller import ProjectIOController

    cfg, runtime, executor, _request = legacy
    revision, bindings = runtime.load_registry()
    checkpoints = [tmp_path / "single-checkpoint", tmp_path / "window-checkpoint"]
    for checkpoint in checkpoints:
        shutil.copytree(cfg.shared_root, checkpoint)
    measurements = []
    for maximum, checkpoint in zip((1, 8), checkpoints, strict=True):
        shutil.rmtree(cfg.shared_root)
        shutil.copytree(checkpoint, cfg.shared_root)
        log = tmp_path / f"slices-{maximum}.jsonl"
        body = f"""import json, time
from pathlib import Path
worker._UPGRADE_WINDOW_MAX_CALLS = {maximum}
original = UpgradeCoordinator.advance
def advance(self, **kwargs):
    started = time.monotonic()
    result = original(self, **kwargs)
    ended = time.monotonic()
    with Path({str(log)!r}).open('a') as output:
        output.write(json.dumps({{'start':started, 'end':ended,
            'committed':result['invocation']['slice_committed']}}) + '\\n')
    return result
UpgradeCoordinator.advance = advance"""
        controller = ProjectIOController(runtime, executor)
        admissions = []
        original_start = executor.start

        def start(request_id, **kwargs):
            process = original_start(request_id, **kwargs)
            if process is not None:
                admissions.append(request_id)
            return process

        baseline_work = _item(cfg)["work_items"]
        started = time.monotonic()
        with monkeypatch.context() as patch:
            _instrument_children(patch, tmp_path, body)
            patch.setattr(executor, "start", start)
            while time.monotonic() - started < 120:
                controller.advance_upgrade_work(bindings, revision)
                if _item(cfg)["work_items"] - baseline_work >= 16:
                    for request in executor.unresolved_requests():
                        assert _wait(executor, request, consume=True).status == "completed"
                    break
                time.sleep(0.02)
            else:
                pytest.fail("daemon benchmark did not advance 16 items within 120 seconds")
        elapsed = time.monotonic() - started
        items = _item(cfg)["work_items"] - baseline_work
        slices = [json.loads(line) for line in log.read_text().splitlines()]
        committed = [row for row in slices if row["committed"]]
        measurements.append(
            {
                "max_calls": maximum,
                "items": items,
                "seconds": elapsed,
                "items_per_second": items / elapsed,
                "admissions_per_item": len(admissions) / items,
                "slice_seconds_mean": sum(row["end"] - row["start"] for row in committed) / len(committed),
                "between_slices_seconds_mean": sum(b["start"] - a["end"] for a, b in zip(committed, committed[1:]))
                / (len(committed) - 1),
            }
        )
    print("\nUPGRADE_WINDOW_BENCHMARK=" + json.dumps(measurements, sort_keys=True))
    assert measurements[1]["admissions_per_item"] < measurements[0]["admissions_per_item"]
    assert measurements[1]["items_per_second"] > measurements[0]["items_per_second"]


@pytest.mark.parametrize("interruption", ["contention", "io_failure", "no_progress"])
def test_window_stops_on_real_coordinator_interruption(legacy, monkeypatch, interruption):
    from contextlib import nullcontext
    from dataclasses import replace

    cfg, runtime, executor, request = legacy
    original_advance = UpgradeCoordinator.advance
    original_backfill = TaskDependenciesV2Migration.backfill
    calls = []
    phase_calls = []

    def backfill(self, context):
        phase_calls.append(1)
        if len(phase_calls) == 2 and interruption == "io_failure":
            raise OSError("injected projection I/O failure")
        result = original_backfill(self, context)
        if interruption == "no_progress":
            return replace(result, meaningful_progress=False)
        return result

    def advance(self, **kwargs):
        calls.append(kwargs)
        lock = (
            exclusive(shared_paths(cfg.shared_root)["locks"] / "upgrade.lock")
            if len(calls) == 2 and interruption == "contention"
            else nullcontext()
        )
        with lock:
            result = original_advance(self, **kwargs)
        if interruption == "contention" and len(calls) == 2:
            assert result["invocation"]["contended_lock"] == "upgrade"
            assert result["invocation"]["next_probe_at"] is not None
        return result

    monkeypatch.setattr(TaskDependenciesV2Migration, "backfill", backfill)
    monkeypatch.setattr(UpgradeCoordinator, "advance", advance)
    monkeypatch.setattr(worker, "time", SimpleNamespace(monotonic=lambda: 0.0))
    result = worker._upgrade_service(request, runtime.root, executor.paths, [False])
    assert len(calls) == (1 if interruption == "no_progress" else 2)
    assert len(phase_calls) == (2 if interruption == "io_failure" else 1)
    if interruption == "io_failure":
        assert result["state"] == "repair_required"
    elif interruption == "contention":
        assert result["next_probe_at"] is not None


def test_continuation_checks_completed_prerequisite_drift(legacy, monkeypatch):
    from qqtools.plugins.qexp.runtime.group_discovery.activation import (
        activation_path,
        advance_group_service_activation,
    )
    from qqtools.plugins.qexp.runtime.upgrade.framework import _new_migration_state
    from qqtools.plugins.qexp.runtime.upgrade.group_service import GroupServiceMigration

    cfg, runtime, executor, request = legacy
    from qqtools.plugins.qexp.runtime.group_namespace import activate_group_authority_locked
    from qqtools.plugins.qexp.runtime.locks import schema_lock

    with schema_lock(cfg.shared_root):
        schema_path = cfg.shared_root / "schema/version.json"
        schema = read_json(schema_path)
        if "local-recovery-v1" not in schema["schema"]["required_capabilities"]:
            schema["schema"]["required_capabilities"].append("local-recovery-v1")
        atomic_replace(schema_path, schema)
        assert activate_group_authority_locked(cfg)
    journal = read_json(upgrade_journal_path(cfg))
    for _ in range(100):
        if GroupServiceMigration().terminal_invariant_holds(cfg):
            break
        advance_group_service_activation(cfg)
    assert GroupServiceMigration().terminal_invariant_holds(cfg)
    prerequisite = _new_migration_state(GroupServiceMigration.spec)
    prerequisite["state"] = "completed"
    journal["upgrade"]["migrations"]["group-service-v1"] = prerequisite
    atomic_replace(upgrade_journal_path(cfg), journal)
    original = UpgradeCoordinator.advance
    calls = []

    def advance(self, **kwargs):
        if calls:
            # Evidence disappears independently of the upgrade journal.
            activation_path(cfg.shared_root).unlink()
        calls.append(kwargs)
        return original(self, **kwargs)

    monkeypatch.setattr(UpgradeCoordinator, "advance", advance)
    monkeypatch.setattr(worker, "time", SimpleNamespace(monotonic=lambda: 0.0))
    result = worker._upgrade_service(request, runtime.root, executor.paths, [False])
    assert len(calls) == 2
    assert _item(cfg)["work_items"] == 2
    assert result["state"] == "repair_required"
    journal = read_json(upgrade_journal_path(cfg))
    assert journal["upgrade"]["migrations"]["group-service-v1"]["state"] == "repair_required"


@pytest.mark.parametrize("revoke_before", ["generation", "records"])
def test_projection_directories_recheck_binding_after_inflight_commit(legacy, monkeypatch, revoke_before):
    from qqtools.plugins.qexp.runtime import dependency_projection as projection

    cfg, runtime, executor, request = legacy
    original_advance = UpgradeCoordinator.advance
    original_ensure = projection._ensure_generation
    calls = []
    attempted = []

    def ensure(root, generation):
        if len(calls) == 2:
            generation_path = projection.task_dependency_generation_path(root, generation)
            attempted.append(generation_path)
            if revoke_before == "generation":
                monkeypatch.setattr(worker, "_claim_binding_is_current", lambda *a: False)
            else:
                original_mkdir = type(generation_path).mkdir

                def mkdir(path, *args, **kwargs):
                    result = original_mkdir(path, *args, **kwargs)
                    if path == generation_path:
                        monkeypatch.setattr(worker, "_claim_binding_is_current", lambda *a: False)
                    return result

                monkeypatch.setattr(type(generation_path), "mkdir", mkdir)
        return original_ensure(root, generation)

    def advance(self, **kwargs):
        if calls:
            with dependency_lock(cfg.shared_root):
                mark_degraded(cfg, "legacy writer invalidation between slices")
        calls.append(kwargs)
        return original_advance(self, **kwargs)

    monkeypatch.setattr(projection, "_ensure_generation", ensure)
    monkeypatch.setattr(UpgradeCoordinator, "advance", advance)
    monkeypatch.setattr(worker, "time", SimpleNamespace(monotonic=lambda: 0.0))
    with pytest.raises(worker._BindingAuthorityChanged):
        worker._upgrade_service(request, runtime.root, executor.paths, [False])
    assert len(attempted) == 1
    assert attempted[0].exists() is (revoke_before == "records")
    assert not (attempted[0] / "records").exists()
    assert _item(cfg)["work_items"] == 2
