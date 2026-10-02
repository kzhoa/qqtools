from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, observer, submit
from qqtools.plugins.qexp.agent import project_io_worker as worker
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.project_io_controller import ProjectIOController
from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor
from qqtools.plugins.qexp.agent.project_io_protocol import ProjectIOProcess, ProjectIORequest, ProjectIOResult
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.runtime.observation import maintenance, projection
from qqtools.plugins.qexp.runtime.store import atomic_replace, fenced_mutations

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _case(tmp_path: Path, runtime: MachineRuntime, name: str):
    cfg = init_shared_root(tmp_path / name / ".qexp", "gpu-1")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    revision, _bindings = runtime.load_registry()
    return cfg, binding, revision


def _consume(executor: ProjectIOExecutor, request: ProjectIORequest) -> ProjectIOResult:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        executor.poll()
        result = executor.consume(request.request_id, request)
        if result is not None:
            return result
        time.sleep(0.01)
    raise AssertionError("observation slice did not finish")


def test_observation_gc_converges_with_fresh_owners_and_one_mutation_per_slice(tmp_path: Path, monkeypatch):
    cfg = init_shared_root(tmp_path / "project/.qexp", "gpu-1")
    keep = projection.read_state(cfg)["generation"]
    generations = projection.observation_path(cfg) / "generations"
    obsolete = generations / "obsolete" / "partitions" / "pages"
    for index in range(8):
        atomic_replace(obsolete / f"{index}.json", {"obsolete": index})
    before_fds = len(os.listdir("/proc/self/fd"))
    mutations = []
    real_unlink, real_rmdir = os.unlink, os.rmdir

    def unlink(path, *args, **kwargs):
        mutations.append(path)
        return real_unlink(path, *args, **kwargs)

    def rmdir(path, *args, **kwargs):
        mutations.append(path)
        return real_rmdir(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)
    monkeypatch.setattr(os, "rmdir", rmdir)
    for _ in range(20):
        before = len(mutations)
        owner = maintenance.ObservationMaintenance(cfg)
        try:
            result = owner.advance()
        finally:
            owner.close()
        assert len(mutations) - before <= 1
        assert len(os.listdir("/proc/self/fd")) == before_fds
        assert (generations / keep).is_dir()
        if result["reason"] == "idle":
            break
    else:
        raise AssertionError("fresh GC owners lost their deletion progress")
    assert not (generations / "obsolete").exists()


def test_observation_gc_fences_each_delete_and_retries_without_private_cursor(tmp_path: Path):
    cfg = init_shared_root(tmp_path / "project/.qexp", "gpu-1")
    leaf = projection.observation_path(cfg) / "generations/obsolete/leaf.json"
    atomic_replace(leaf, {"obsolete": True})

    def revoked():
        raise RuntimeError("controller epoch revoked")

    owner = maintenance.ObservationMaintenance(cfg)
    try:
        with fenced_mutations(cfg.shared_root, revoked):
            result = owner.advance()
        assert result["state"] == "degraded"
        assert leaf.exists()
    finally:
        owner.close()
    owner = maintenance.ObservationMaintenance(cfg)
    try:
        assert owner.advance()["reason"] == "garbage_collecting"
        assert not leaf.exists()
    finally:
        owner.close()


def test_observation_service_rebuilds_and_cleans_with_fresh_subprocesses(tmp_path: Path):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg, binding, revision = _case(tmp_path, runtime, "project")
    tasks = [submit(cfg, ["true"], working_dir=tmp_path) for _ in range(4)]
    before = projection.read_state(cfg)["generation"]
    maintenance.request_rebuild(cfg)
    local_root = runtime.project_paths(binding.project_id)["root"]
    local_before = {path.relative_to(local_root): path.read_bytes() for path in local_root.rglob("*") if path.is_file()}
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    try:
        for _ in range(100):
            request = executor.prepare_observation_service(binding, revision)
            assert executor.start(request.request_id) is not None
            result = _consume(executor, request)
            assert result.status == "completed"
            if result.evidence["quiescent"]:
                break
        else:
            raise AssertionError("fresh isolated observation workers did not finish the rebuild")
        state = projection.read_state(cfg)
        assert state["state"] == "active" and not state["dirty"]
        assert state["generation"] != before
        assert {path.name for path in (projection.observation_path(cfg) / "generations").iterdir()} == {
            state["generation"]
        }
        page = observer.list_tasks_page(cfg, page_size=100)
        assert {value["task_id"] for value in page["items"]} == {task.task_id for task in tasks}
        assert {
            path.relative_to(local_root): path.read_bytes() for path in local_root.rglob("*") if path.is_file()
        } == local_before
        assert not executor.unresolved_requests()
    finally:
        executor.shutdown()


def test_observation_service_blocked_read_does_not_delay_healthy_rebuild(tmp_path: Path, monkeypatch):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg_blocked, blocked, _revision = _case(tmp_path, runtime, "blocked")
    cfg_healthy, healthy, revision = _case(tmp_path, runtime, "healthy")
    _revision, bindings = runtime.load_registry()
    task = submit(cfg_healthy, ["true"], working_dir=tmp_path)
    maintenance.request_rebuild(cfg_healthy)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    runtime.working_set.reconcile(bindings, revision=revision)
    # Observation rejects non-regular state files before reading them. Block a
    # regular service read instead, after registration validation has completed.
    state_path = cfg_blocked.shared_root / "schema" / "version.json"
    saved = state_path.with_suffix(".saved")
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            controller.advance_binding_validation(bindings, revision)
            if all(controller.validated_config(binding, revision) is not None for binding in bindings):
                break
            time.sleep(0.01)
        else:
            raise AssertionError("observation bindings did not validate")
        monkeypatch.setattr(
            maintenance.ObservationMaintenance, "advance", lambda self: pytest.fail("inline Project observation")
        )
        state_path.rename(saved)
        os.mkfifo(state_path)
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            controller.advance_observation_maintenance(bindings, revision)
            if healthy.project_id in {identity.project_id for identity in controller._observation_due}:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("blocked observation withheld the healthy Project")
        state = projection.read_state(cfg_healthy)
        assert state["state"] == "active" and not state["dirty"]
        assert observer.list_tasks_page(cfg_healthy)["items"][0]["task_id"] == task.task_id
        assert (
            len([request for request in executor.unresolved_requests() if request.project_id == blocked.project_id])
            == 1
        )
        assert executor.status_view()["active_worker_count"] <= 4
    finally:
        if saved.exists():
            state_path.unlink()
            saved.rename(state_path)
        executor.shutdown()


@pytest.mark.parametrize("failure", ["worker_absence", "late_epoch"])
def test_observation_service_preserves_possible_write_ambiguity(tmp_path: Path, failure: str):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    _cfg, binding, revision = _case(tmp_path, runtime, "project")
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    try:
        request = executor.prepare_observation_service(binding, revision)
        process = ProjectIOProcess(request, 2_000_000_000, 1, request.prepared_at, "running")
        atomic_replace(executor.paths["project_io_processes"] / f"{request.request_id}.json", process.to_dict())
        if failure == "late_epoch":
            executor.fence_epoch()
            assert worker._publish_result(
                executor.root,
                executor.paths,
                request,
                process,
                "completed",
                None,
                {"state": "active", "quiescent": True, "reason_code": "idle"},
                shared_write_possible=True,
            )
        else:
            executor.poll()
        result = executor.load_result(request.request_id)
        assert result is not None and result.status == "outcome_unknown"
        assert executor.consume(request.request_id, request) is None
        if failure == "worker_absence":
            assert executor.reset_ambiguous_observation_service_for_retry(request.request_id, request)
            assert executor.start(request.request_id) is not None
            assert _consume(executor, request).status == "completed"
    finally:
        executor.shutdown()
