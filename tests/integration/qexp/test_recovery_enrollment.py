"""Machine-owned registration preparation progresses without foreground shared I/O."""

import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.helpers import _machine_is_true_idle
from qqtools.plugins.qexp.agent.project_io_controller import ProjectIOController
from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor
from qqtools.plugins.qexp.agent.recovery_enrollment import RecoveryEnrollment
from qqtools.plugins.qexp.layout import LOCAL_RECOVERY_CAPABILITY
from qqtools.plugins.qexp.runtime.store import read_json
from tests.helpers.qexp.lifecycle import wait_until

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


@pytest.fixture(autouse=True)
def empty_process_census(tmp_path, monkeypatch):
    """Keep pass-count tests deterministic; real child agents use actual /proc."""
    from qqtools.plugins.qexp.runtime import responsibility_process_capture as processes

    proc = tmp_path / "proc"
    (proc / "sys/kernel/random").mkdir(parents=True)
    (proc / "sys/kernel/random/boot_id").write_text(Path("/proc/sys/kernel/random/boot_id").read_text())
    (proc / "self/ns").mkdir(parents=True)
    (proc / "self/ns/pid").symlink_to("/proc/self/ns/pid")
    monkeypatch.setattr(processes, "PROC_ROOT", proc)


def register(runtime, tmp_path, name):
    cfg = init_shared_root(tmp_path / name / ".qexp", "gpu-1", runtime_root=tmp_path / f"legacy-{name}")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    return binding, cfg.shared_root / "machines" / cfg.machine_name / "registration.json"


@contextmanager
def coordinated_enrollment(runtime):
    executor = ProjectIOExecutor(runtime)
    enrollment = RecoveryEnrollment(runtime)
    try:
        with runtime.scheduler_authority() as acquired:
            assert acquired
            executor.begin_epoch()
            controller = ProjectIOController(runtime, executor)
            try:
                yield enrollment, controller
            finally:
                executor.fence_epoch()
    finally:
        enrollment.stop()
        executor.shutdown()


def advance_coordinated(enrollment, controller):
    revision, bindings = enrollment.runtime.load_registry_snapshot()
    with controller.admission_turn():
        enrollment.advance(controller, bindings, revision)


def settle_coordinated(enrollment, controller, *, timeout=20):
    def completed():
        advance_coordinated(enrollment, controller)
        return not enrollment.runtime.recovery_enrollment_pending_projects

    wait_until("enrollment-settled", completed, stage="recovery-enrollment:completion", timeout=timeout)


def test_all_registered_projects_are_prepared_in_fair_bounded_passes(tmp_path, monkeypatch):
    runtime = MachineRuntime(tmp_path / "machine")
    entries = [register(runtime, tmp_path, f"project-{index}") for index in range(9)]
    monkeypatch.setattr(
        runtime,
        "prepare_recovery_registration",
        lambda *_args, **_kwargs: pytest.fail("registration preparation ran in the coordinator"),
    )
    with coordinated_enrollment(runtime) as (enrollment, controller):
        visited = set()
        calls = []
        advance = enrollment._advance_local

        def observe(capture):
            calls.append(capture.binding)
            visited.add(capture.binding)
            return advance(capture)

        monkeypatch.setattr(enrollment, "_advance_local", observe)

        def completed():
            calls.clear()
            advance_coordinated(enrollment, controller)
            assert len(calls) <= 4
            if runtime.recovery_enrollment_pending_projects:
                assert not _machine_is_true_idle(runtime, has_consumed_binding=True)
            return not runtime.recovery_enrollment_pending_projects

        wait_until("nine-projects-settled", completed, stage="recovery-enrollment:fair-passes", timeout=30)
        assert visited == {binding for binding, _path in entries}
        assert enrollment._settled == visited
        assert all(read_json(path)["registration"]["version"] == 2 for _binding, path in entries)
        advance_coordinated(enrollment, controller)
        assert not runtime.recovery_enrollment_pending_projects


def test_pending_project_retries_without_starving_new_or_disabled_bindings(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runtime.locks import machine_lock

    runtime = MachineRuntime(tmp_path / "machine")
    first, first_path = register(runtime, tmp_path, "first")
    others = [register(runtime, tmp_path, f"other-{index}") for index in range(5)]
    disabled = runtime.set_enabled(others[-1][0].project_id, False)
    with coordinated_enrollment(runtime) as (enrollment, controller):
        with machine_lock(first.shared_root, first.machine_name) as acquired:
            assert acquired

            def others_completed():
                advance_coordinated(enrollment, controller)
                return runtime.recovery_enrollment_pending_projects == {first.project_id}

            wait_until("peers-settled", others_completed, stage="recovery-enrollment:blocked-peer", timeout=20)
            assert runtime.recovery_enrollment_pending_projects == {first.project_id}
            assert read_json(first_path)["registration"]["version"] == 1
            assert all(read_json(path)["registration"]["version"] == 2 for _binding, path in others)
            assert disabled in runtime.load_registry()[1]
            added, added_path = register(runtime, tmp_path, "added")

            def added_completed():
                advance_coordinated(enrollment, controller)
                return added in enrollment._settled

            wait_until("new-peer-settled", added_completed, stage="recovery-enrollment:new-peer", timeout=20)
            assert first not in enrollment._settled
        settle_coordinated(enrollment, controller)
        assert read_json(first_path)["registration"]["version"] == 2
        assert read_json(added_path)["registration"]["version"] == 2
        assert added in enrollment._settled


def test_binding_changes_invalidate_success_and_superseded_generation_settles(tmp_path, monkeypatch):
    runtime = MachineRuntime(tmp_path / "machine")
    binding, _path = register(runtime, tmp_path, "project")
    with coordinated_enrollment(runtime) as (enrollment, controller):
        settle_coordinated(enrollment, controller)
        assert enrollment._settled == {binding}
        disabled = runtime.set_enabled(binding.project_id, False)
        runtime._supersede_registration(disabled, "replacement")
        enrollment.poll()
        assert runtime.recovery_enrollment_pending_projects == {disabled.project_id}
        assert not enrollment._settled
        settle_coordinated(enrollment, controller)
        assert enrollment._settled == {disabled}
        assert runtime.registration_status(disabled)["state"] == "superseded"


def test_slow_worker_does_not_block_poll_or_start_duplicate_worker(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runtime.locks import machine_lock

    runtime = MachineRuntime(tmp_path / "machine")
    binding, _path = register(runtime, tmp_path, "project")
    with coordinated_enrollment(runtime) as (enrollment, controller):
        calls = []
        start = controller.executor.start

        def observe_start(request_id, **kwargs):
            if controller.executor._load_request(request_id).operation_kind == "recovery_admission":
                calls.append(request_id)
            return start(request_id, **kwargs)

        monkeypatch.setattr(controller.executor, "start", observe_start)
        with machine_lock(binding.shared_root, binding.machine_name) as acquired:
            assert acquired
            advance_coordinated(enrollment, controller)
            (request,) = controller.executor.unresolved_requests()
            assert request.operation_kind == "recovery_admission"
            for _ in range(3):
                started = time.monotonic()
                enrollment.poll()
                advance_coordinated(enrollment, controller)
                assert time.monotonic() - started < 1.0
                assert controller.executor.unresolved_requests() == (request,)
                assert enrollment._settled == frozenset()
            assert calls == [request.request_id]
        settle_coordinated(enrollment, controller)
        assert enrollment._settled == {binding}
        assert calls == [request.request_id]


def test_real_global_agent_prepares_registered_roots_before_on_demand_exit(tmp_path):
    from qqtools.plugins.qexp.agent.recovery_capture import inspect_recovery_capture
    from qqtools.plugins.qexp.runtime.group_namespace import is_group_authority_isolated
    from qqtools.plugins.qexp.runtime.responsibility_capture import CAPTURE_FILE
    from qqtools.plugins.qexp.runtime.responsibility_completion import is_source_released, read_capture_completion

    runtime = MachineRuntime(tmp_path / "machine")
    entries = [register(runtime, tmp_path, f"project-{index}") for index in range(5)]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[3] / "src")
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os, sys; os.fsync = lambda _descriptor: None; "
            "from pathlib import Path; "
            "from qqtools.plugins.qexp.runtime import responsibility_process_capture as process_capture; "
            "process_capture.PROC_ROOT = Path(sys.argv[2]); "
            "from qqtools.plugins.qexp.agent.lifecycle import run_machine_agent_loop; "
            "run_machine_agent_loop(sys.argv[1], loop_interval=0.05, available_gpus=[])",
            str(runtime.root),
            str(tmp_path / "proc"),
        ],
        env=environment,
        start_new_session=True,
    )

    def enrollment_has_settled(binding):
        if inspect_recovery_capture(runtime, binding)["state"] != "captured":
            return False
        root = runtime.project_paths(binding.project_id)["root"]
        proof = read_capture_completion(root, should_sync=False)
        return (
            proof is not None and is_source_released(root, proof) and is_group_authority_isolated(binding.shared_root)
        )

    try:
        wait_until(
            "all-registrations-prepared",
            lambda: all(read_json(path)["registration"]["version"] == 2 for _binding, path in entries),
            stage="recovery-enrollment:automatic-preparation",
            on_timeout=lambda: {
                "process_returncode": process.poll(),
                "requests": [
                    read_json(p).get("project_io_request", {}).get("operation_kind")
                    for p in runtime.paths["project_io_requests"].glob("*.json")
                ],
                "results": [
                    (
                        read_json(p).get("project_io_result", {}).get("status"),
                        read_json(p).get("project_io_result", {}).get("reason_code"),
                    )
                    for p in runtime.paths["project_io_results"].glob("*.json")
                ],
                "versions": [read_json(path)["registration"]["version"] for _binding, path in entries],
                "registration_status": [runtime.registration_status(binding) for binding, _path in entries],
                "checkpoints": [
                    [p.name for p in runtime.project_paths(binding.project_id)["root"].glob("responsibility-*.json")]
                    for binding, _path in entries
                ],
            },
        )
        # Capture proof precedes Group activation and retained-source release.
        # Keep those obligations inside the existing 15-second enrollment budget.
        wait_until(
            "all-enrollment-obligations-settled",
            lambda: all(enrollment_has_settled(binding) for binding, _path in entries),
            stage="recovery-enrollment:capture",
            timeout=15,
            on_timeout=lambda: {
                binding.project_id: {
                    "capture": inspect_recovery_capture(runtime, binding),
                    "writer_capture": (
                        read_json(runtime.project_paths(binding.project_id)["root"] / CAPTURE_FILE)
                        if (runtime.project_paths(binding.project_id)["root"] / CAPTURE_FILE).exists()
                        else None
                    ),
                    "blockers": runtime.binding_blockers(binding),
                }
                for binding, _path in entries
            },
        )
        # Idle shutdown performs one final multi-Project dispatch while holding
        # the activation fence. Bound the five-root traversal and shutdown to
        # 30 seconds independently of the completed enrollment obligations.
        assert process.wait(timeout=30) == 0
        for binding, _path in entries:
            assert read_capture_completion(runtime.project_paths(binding.project_id)["root"]) is not None
            schema = read_json(binding.shared_root / "schema" / "version.json")["schema"]
            assert LOCAL_RECOVERY_CAPABILITY in schema["required_capabilities"]
        for binding, _path in entries:
            assert runtime.registration_status(binding)["write_eligible"]
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def test_failed_worker_start_is_retryable(tmp_path, monkeypatch):
    runtime = MachineRuntime(tmp_path / "machine")
    binding, path = register(runtime, tmp_path, "project")
    with coordinated_enrollment(runtime) as (enrollment, controller):
        start = controller.executor.start

        def unavailable(_request_id, **_kwargs):
            raise RuntimeError("cannot start worker")

        monkeypatch.setattr(controller.executor, "start", unavailable)
        advance_coordinated(enrollment, controller)
        assert runtime.recovery_enrollment_pending_projects == {binding.project_id}
        assert read_json(path)["registration"]["version"] == 1
        monkeypatch.setattr(controller.executor, "start", start)
        settle_coordinated(enrollment, controller)
        assert read_json(path)["registration"]["version"] == 2


def test_background_preparation_waits_for_migration_then_resumes(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runtime.store import atomic_replace

    runtime = MachineRuntime(tmp_path / "machine")
    binding, path = register(runtime, tmp_path, "project")
    journal_path = runtime.migration_path(binding.project_id)
    original = {
        "migration": {
            "state": "active",
            "project_id": binding.project_id,
            "shared_root": str(binding.shared_root),
            "machine_name": binding.machine_name,
            "legacy_runtime_root": str(tmp_path / "legacy-project"),
        }
    }
    pending = {"migration": {**original["migration"], "state": "blocked"}}
    atomic_replace(journal_path, pending)
    with coordinated_enrollment(runtime) as (enrollment, controller):
        observed = []
        advance = controller.advance_recovery_admission

        def capture_result(*args, **kwargs):
            results = advance(*args, **kwargs)
            observed.extend(results.values())
            return results

        monkeypatch.setattr(controller, "advance_recovery_admission", capture_result)

        def migration_deferred():
            advance_coordinated(enrollment, controller)
            return any(value.get("state") == "waiting" for value in observed)

        wait_until("migration-deferred", migration_deferred, stage="recovery-enrollment:migration")
        assert read_json(path)["registration"]["version"] == 1
        assert runtime.recovery_enrollment_pending_projects == {binding.project_id}
        assert read_json(journal_path) == pending
        atomic_replace(journal_path, original)
        settle_coordinated(enrollment, controller)
        assert read_json(path)["registration"]["version"] == 2


def test_malformed_binding_does_not_abort_other_projects_in_pass(tmp_path):
    from qqtools.plugins.qexp.runtime.store import atomic_replace

    runtime = MachineRuntime(tmp_path / "machine")
    damaged, damaged_path = register(runtime, tmp_path, "damaged")
    healthy, healthy_path = register(runtime, tmp_path, "healthy")
    atomic_replace(damaged_path.parent / "machine.json", {"machine": None})
    with coordinated_enrollment(runtime) as (enrollment, controller):

        def healthy_completed():
            advance_coordinated(enrollment, controller)
            return healthy in enrollment._settled

        wait_until("healthy-settled", healthy_completed, stage="recovery-enrollment:damaged-peer", timeout=20)
        assert runtime.recovery_enrollment_pending_projects == {damaged.project_id}
        assert read_json(damaged_path)["registration"]["version"] == 1
        assert read_json(healthy_path)["registration"]["version"] == 2


def test_paused_machine_rollout_waits_without_stalling_other_projects(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.agent.lifecycle import get_machine_agent_status

    runtime = MachineRuntime(tmp_path / "machine")
    first, _path = register(runtime, tmp_path, "first")
    other, _other_path = register(runtime, tmp_path, "other")
    peer_cfg = init_shared_root(first.shared_root, "peer", runtime_root=tmp_path / "peer-local")
    peer = MachineRuntime(tmp_path / "peer-machine")
    peer_binding = peer.add_binding(peer_cfg.shared_root, peer_cfg.machine_name)
    with coordinated_enrollment(runtime) as (enrollment, controller):

        def other_completed():
            advance_coordinated(enrollment, controller)
            return other in enrollment._settled

        wait_until("other-settled", other_completed, stage="recovery-enrollment:paused-rollout", timeout=20)
        assert runtime.recovery_enrollment_pending_projects == {first.project_id}
        projects = {item["project_id"]: item for item in get_machine_agent_status(runtime)["projects"]}
        assert projects[first.project_id]["recovery_enrollment"] == {
            "state": "waiting",
            "blockers": ["registration_not_prepared:peer"],
            "diagnostic_only": True,
        }
        assert projects[other.project_id]["recovery_enrollment"]["state"] == "admission_fenced"
        assert projects[first.project_id]["write_eligible"]
    with peer.scheduler_authority() as acquired:
        assert acquired
        assert peer.prepare_recovery_registration(peer_binding)
    with coordinated_enrollment(runtime) as (enrollment, controller):
        settle_coordinated(enrollment, controller)
        projects = {item["project_id"]: item for item in get_machine_agent_status(runtime)["projects"]}
        assert projects[first.project_id]["recovery_enrollment"]["state"] == "admission_fenced"
