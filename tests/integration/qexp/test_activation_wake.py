"""Local activation must fence asynchronous idle proofs without shared I/O."""

from __future__ import annotations

from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.activation import ensure_managed_project_agent_active
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.helpers import _machine_is_true_idle
from qqtools.plugins.qexp.agent.lifecycle import _confirm_idle_shutdown, ensure_machine_agent_started
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _runtime(tmp_path: Path) -> MachineRuntime:
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    revision, bindings = runtime.load_registry_snapshot()
    runtime.working_set.reconcile(bindings, revision=revision)
    runtime.activation_wake.capture(revision, bindings)
    return runtime


def _confirm(runtime: MachineRuntime):
    return _confirm_idle_shutdown(
        runtime,
        has_consumed_binding=True,
        available_gpus=[],
        executor=None,
        instance_id="wake-test",
        loop_interval=0.1,
        started_at="2026-10-01T00:00:00Z",
    )


def test_running_agent_activation_invalidates_captured_idle_generation(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path)
    status = {"is_running": True, "pid": 123}

    def running(_runtime):
        with runtime.agent_lifecycle_guard(blocking=False) as acquired:
            assert not acquired
        return status

    monkeypatch.setattr("qqtools.plugins.qexp.agent.lifecycle.get_machine_agent_status", running)
    assert runtime.activation_wake.is_current()
    assert ensure_machine_agent_started(runtime) == (None, status)
    assert not runtime.activation_wake.is_current()
    first = read_json(runtime.paths["agent"] / "activation-wake.json")
    assert first["machine_activation_wake"]["runtime_id"] == runtime.instance_id
    revision, bindings = runtime.load_registry_snapshot()
    runtime.activation_wake.capture(revision, bindings)
    assert runtime.activation_wake.is_current()
    assert ensure_machine_agent_started(runtime) == (None, status)
    assert read_json(runtime.paths["agent"] / "activation-wake.json") != first
    assert not runtime.activation_wake.is_current()


def test_managed_running_fast_path_participates_in_activation_fence(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path)
    cfg = init_shared_root(tmp_path / "project/.qexp", "gpu-1")
    runtime.add_binding(cfg.shared_root, cfg.machine_name)
    revision, bindings = runtime.load_registry_snapshot()
    runtime.working_set.reconcile(bindings, revision=revision)
    runtime.activation_wake.capture(revision, bindings)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.agent.lifecycle.get_machine_agent_status",
        lambda _runtime: {"is_running": True, "pid": 123},
    )
    monkeypatch.setattr(
        "qqtools.plugins.qexp.activation.get_machine_agent_status",
        lambda _runtime: {"is_running": True, "pid": 123},
    )
    action, status = ensure_managed_project_agent_active(cfg, machine_runtime=runtime)
    assert action == "already_running" and status["is_running"]
    assert status["managed_by_machine"] and status["project_id"] == bindings[0].project_id
    assert not runtime.activation_wake.is_current()


@pytest.mark.parametrize("change", ["wake", "registry", "malformed", "foreign", "oversize"])
def test_local_idle_fence_rejects_changed_or_unreadable_capture(tmp_path, monkeypatch, change):
    runtime = _runtime(tmp_path)
    runtime.project_io_executor = SimpleNamespace(has_unfinished_work=lambda: False)
    runtime.last_cycle_had_demand = False

    def forbidden(*_args, **_kwargs):
        raise AssertionError("isolated final idle check attempted synchronous shared I/O")

    monkeypatch.setattr("qqtools.plugins.qexp.agent.lifecycle._dispatch.dispatch_machine_cycle_locked", forbidden)
    monkeypatch.setattr(runtime, "iter_recovery_blockers", forbidden)
    assert _machine_is_true_idle(runtime, has_consumed_binding=True)
    if change == "wake":
        with runtime.agent_lifecycle_guard():
            runtime.activation_wake.publish_locked()
    elif change == "registry":
        cfg = init_shared_root(tmp_path / "project/.qexp", "gpu-1")
        runtime.add_binding(cfg.shared_root, cfg.machine_name)
    else:
        record = {"version": 1, "runtime_id": runtime.instance_id, "generation": "a" * 32}
        if change == "malformed":
            record["generation"] = "invalid"
        elif change == "foreign":
            record["runtime_id"] = "another-runtime"
        else:
            record["generation"] = "a" * 1024
        atomic_replace(runtime.paths["agent"] / "activation-wake.json", {"machine_activation_wake": record})
    assert not runtime.activation_wake.is_current()
    assert not _machine_is_true_idle(runtime, has_consumed_binding=True)
    assert _confirm(runtime) is None
    with runtime.agent_lifecycle_guard(blocking=False) as acquired:
        assert acquired


def test_isolated_final_idle_holds_guard_without_dispatch_or_shared_reads(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path)
    runtime.project_io_executor = SimpleNamespace(has_unfinished_work=lambda: False)
    runtime.last_cycle_had_demand = False

    def forbidden(*_args, **_kwargs):
        raise AssertionError("isolated final idle check performed shared work")

    monkeypatch.setattr("qqtools.plugins.qexp.agent.lifecycle._dispatch.dispatch_machine_cycle_locked", forbidden)
    monkeypatch.setattr(runtime, "iter_recovery_blockers", forbidden)
    guard = _confirm(runtime)
    assert guard is not None
    try:
        with runtime.agent_lifecycle_guard(blocking=False) as acquired:
            assert not acquired
    finally:
        guard.__exit__(None, None, None)
    with runtime.agent_lifecycle_guard(blocking=False) as acquired:
        assert acquired


def test_activation_waiting_for_final_exit_starts_the_successor(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path)
    runtime.project_io_executor = SimpleNamespace(has_unfinished_work=lambda: False)
    runtime.last_cycle_had_demand = False
    status = {"is_running": True, "pid": 123}
    entered, finished = Event(), Event()
    results, failures = [], []

    def start(_runtime, **_kwargs):
        with runtime.agent_lifecycle_guard(blocking=False) as acquired:
            assert not acquired
        status.update(is_running=True, pid=456)
        return SimpleNamespace(pid=456)

    def activate():
        entered.set()
        try:
            results.append(ensure_machine_agent_started(runtime))
        except BaseException as exc:
            failures.append(exc)
        finally:
            finished.set()

    monkeypatch.setattr("qqtools.plugins.qexp.agent.lifecycle.get_machine_agent_status", lambda _runtime: dict(status))
    monkeypatch.setattr("qqtools.plugins.qexp.agent.lifecycle._start_machine_agent_locked", start)
    guard = _confirm(runtime)
    assert guard is not None
    thread = Thread(target=activate, daemon=True)
    try:
        thread.start()
        assert entered.wait(1.0)
        assert not finished.wait(0.05)
        status.update(is_running=False, pid=None)
    finally:
        guard.__exit__(None, None, None)
        thread.join(timeout=2.0)
    assert not thread.is_alive() and not failures
    assert len(results) == 1
    process, successor_status = results[0]
    assert process.pid == 456 and successor_status["pid"] == 456
    assert successor_status["is_running"]
    assert not runtime.activation_wake.is_current()


def test_activation_capture_invalidates_binding_turns_before_new_services(tmp_path):
    runtime = _runtime(tmp_path)
    cfg = init_shared_root(tmp_path / "project/.qexp", "gpu-1")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    revision, bindings = runtime.load_registry_snapshot()
    runtime.working_set.reconcile(bindings, revision=revision)
    runtime.activation_wake.capture(revision, bindings)
    old_turn = runtime.working_set.begin_turn(binding, "scheduler")
    with runtime.agent_lifecycle_guard():
        runtime.activation_wake.publish_locked()
    assert not runtime.activation_wake.is_current()
    runtime.activation_wake.capture(revision, bindings)
    new_turn = runtime.working_set.begin_turn(binding, "scheduler")
    assert new_turn.wake_generation > old_turn.wake_generation
    assert not runtime.working_set.acknowledge(old_turn, quiescent=True)
    assert runtime.activation_wake.is_current()


def test_isolated_idle_requires_all_service_lanes_to_settle(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path)
    cfg = init_shared_root(tmp_path / "project/.qexp", "gpu-1")
    runtime.add_binding(cfg.shared_root, cfg.machine_name)
    revision, bindings = runtime.load_registry_snapshot()
    runtime.working_set.reconcile(bindings, revision=revision)
    runtime.activation_wake.capture(revision, bindings)
    runtime.project_io_executor = SimpleNamespace(has_unfinished_work=lambda: False)
    runtime.last_cycle_had_demand = False

    def forbidden(*_args, **_kwargs):
        raise AssertionError("missing asynchronous idle proof was replaced by shared I/O")

    monkeypatch.setattr(runtime, "iter_recovery_blockers", forbidden)
    assert not _machine_is_true_idle(runtime, has_consumed_binding=True)
