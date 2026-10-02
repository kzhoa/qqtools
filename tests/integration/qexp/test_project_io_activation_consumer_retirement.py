from __future__ import annotations

import time
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.dispatch_loop import _advance_activation_working_set, dispatch_machine_cycle_locked
from qqtools.plugins.qexp.agent.project_io_controller import ProjectIOController
from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.runtime.project_activation_consumers import read_consumer_progress, register_consumer

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _case(tmp_path: Path):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    register_consumer(
        cfg.shared_root,
        runtime_id=runtime.instance_id,
        project_id=binding.project_id,
        registration_generation=binding.registration_generation,
        process_fence="process-a",
    )
    disabled = runtime.set_enabled(binding.project_id, False)
    return runtime, cfg, disabled


def _advance_retirement(runtime: MachineRuntime, controller: ProjectIOController, *, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    selected = ()
    while time.monotonic() < deadline:
        revision, snapshot = runtime.load_registry()
        selected = runtime.activation_consumer_retirements.select_pending(snapshot, limit=64)
        completions = controller.advance_activation_consumer_retirements(selected, revision)
        runtime.activation_consumer_retirements.apply_completions(selected, completions)
        if completions:
            return completions
        time.sleep(0.02)
    unresolved = controller.executor.unresolved_requests()
    raise AssertionError(
        f"activation consumer retirement did not complete: {unresolved!r}; "
        f"selected={selected!r}; results={[controller.executor.load_result(request.request_id) for request in unresolved]!r}"
    )


def test_binding_removal_defers_shared_consumer_retirement_to_isolated_worker(
    tmp_path: Path,
) -> None:
    runtime, cfg, binding = _case(tmp_path)
    removed = runtime.remove_binding(binding.project_id)
    assert removed == binding
    assert (
        read_consumer_progress(
            cfg.shared_root,
            runtime_id=runtime.instance_id,
            project_id=binding.project_id,
            registration_generation=binding.registration_generation,
        )["project_activation_consumer"]["state"]
        == "active"
    )

    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    try:
        completion = _advance_retirement(runtime, controller)[binding.project_id]
        assert completion == {"outcome": "retired", "consumer_existed": True}
        assert (
            read_consumer_progress(
                cfg.shared_root,
                runtime_id=runtime.instance_id,
                project_id=binding.project_id,
                registration_generation=binding.registration_generation,
            )["project_activation_consumer"]["state"]
            == "retired"
        )
        assert runtime.activation_consumer_retirements.select_pending((), limit=64) == ()
        assert not executor.has_unfinished_work()
    finally:
        executor.shutdown()


def test_pending_retirement_forces_new_generation_and_retires_only_old_consumer(tmp_path: Path) -> None:
    runtime, cfg, old = _case(tmp_path)
    runtime.remove_binding(old.project_id)

    replacement = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    assert replacement.registration_generation != old.registration_generation
    register_consumer(
        cfg.shared_root,
        runtime_id=runtime.instance_id,
        project_id=replacement.project_id,
        registration_generation=replacement.registration_generation,
        process_fence="process-b",
    )

    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    try:
        completion = _advance_retirement(runtime, controller)[old.project_id]
        assert completion["outcome"] == "retired"
        old_progress = read_consumer_progress(
            cfg.shared_root,
            runtime_id=runtime.instance_id,
            project_id=old.project_id,
            registration_generation=old.registration_generation,
        )
        replacement_progress = read_consumer_progress(
            cfg.shared_root,
            runtime_id=runtime.instance_id,
            project_id=replacement.project_id,
            registration_generation=replacement.registration_generation,
        )
        assert old_progress["project_activation_consumer"]["state"] == "retired"
        assert replacement_progress["project_activation_consumer"]["state"] == "active"
    finally:
        executor.shutdown()


def test_project_keyed_completion_clears_only_one_of_two_pending_generations(tmp_path: Path) -> None:
    runtime, cfg, first = _case(tmp_path)
    runtime.remove_binding(first.project_id)
    replacement = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    second = runtime.set_enabled(replacement.project_id, False)
    runtime.remove_binding(second.project_id)

    runtime.activation_consumer_retirements.apply_completions(
        (first, second),
        {first.project_id: {"outcome": "retired", "consumer_existed": True}},
    )

    assert (
        runtime.activation_consumer_retirements.select_exact(
            runtime_id=first.runtime_instance_id,
            project_id=first.project_id,
            registration_generation=first.registration_generation,
            shared_root=first.shared_root,
            machine_name=first.machine_name,
        )
        is None
    )
    assert (
        runtime.activation_consumer_retirements.select_exact(
            runtime_id=second.runtime_instance_id,
            project_id=second.project_id,
            registration_generation=second.registration_generation,
            shared_root=second.shared_root,
            machine_name=second.machine_name,
        )
        == second
    )


def test_dispatch_never_batches_two_retirement_generations_for_one_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, cfg, first = _case(tmp_path)
    runtime.remove_binding(first.project_id)
    replacement = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    second = runtime.set_enabled(replacement.project_id, False)
    runtime.remove_binding(second.project_id)
    revision, bindings = runtime.load_registry()
    assert bindings == []

    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    executor.prepare_activation_consumer_retire(first, revision)
    selected: list[tuple] = []
    monkeypatch.setattr(
        runtime.activation_consumer_retirements,
        "select_pending",
        lambda *_args, **_kwargs: (second,),
    )
    monkeypatch.setattr(
        controller,
        "advance_activation_consumer_retirements",
        lambda candidates, _revision: selected.append(tuple(candidates)) or {},
    )
    try:
        _advance_activation_working_set(runtime, controller, revision, bindings)
        assert selected == [(first,)]
    finally:
        executor.shutdown()


def test_production_dispatch_advances_pending_consumer_retirement(tmp_path: Path) -> None:
    runtime, cfg, binding = _case(tmp_path)
    runtime.remove_binding(binding.project_id)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    runtime.project_io_executor = executor
    runtime.project_io_controller = ProjectIOController(runtime, executor)
    deadline = time.monotonic() + 5.0
    try:
        while time.monotonic() < deadline:
            dispatch_machine_cycle_locked(runtime, available_gpus=[], supervise=True, publish_snapshots=False)
            progress = read_consumer_progress(
                cfg.shared_root,
                runtime_id=runtime.instance_id,
                project_id=binding.project_id,
                registration_generation=binding.registration_generation,
            )
            if (
                progress["project_activation_consumer"]["state"] == "retired"
                and runtime.activation_consumer_retirements.select_pending((), limit=64) == ()
                and not executor.has_unfinished_work()
            ):
                break
            time.sleep(0.02)
        else:
            raise AssertionError("production dispatch did not retire the removed activation consumer")

        assert progress["project_activation_consumer"]["state"] == "retired"
    finally:
        coordinator = getattr(runtime, "attempt_supervision_coordinator", None)
        if coordinator is not None:
            coordinator.close()
        executor.shutdown()


def test_retirement_worker_rejects_an_exact_binding_that_is_still_registered(tmp_path: Path) -> None:
    runtime, cfg, binding = _case(tmp_path)
    revision, _snapshot = runtime.load_registry()
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    try:
        request = executor.prepare_activation_consumer_retire(binding, revision)
        executor.start(request.request_id)
        deadline = time.monotonic() + 5.0
        result = None
        while result is None and time.monotonic() < deadline:
            executor.poll()
            result = executor.load_result(request.request_id)
            time.sleep(0.02)
        assert result is not None
        assert result.status == "completed"
        assert result.evidence == {"outcome": "stale", "consumer_existed": False}
        assert (
            read_consumer_progress(
                cfg.shared_root,
                runtime_id=runtime.instance_id,
                project_id=binding.project_id,
                registration_generation=binding.registration_generation,
            )["project_activation_consumer"]["state"]
            == "active"
        )
    finally:
        executor.shutdown()
