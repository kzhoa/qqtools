"""Machine-owned registration preparation progresses without foreground shared I/O."""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, get_ident
from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent import recovery_enrollment
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.helpers import _machine_is_true_idle
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


def collect(enrollment):
    wait_until(
        "enrollment-settled",
        lambda: (
            enrollment._settled == frozenset(enrollment.runtime.load_registry()[1])
            and (enrollment._thread is None or not enrollment._thread.is_alive())
        ),
        stage="recovery-enrollment:completion",
    )
    enrollment.poll()


def run_pass(enrollment):
    # Scheduling tests drive one bounded service turn with an explicit join.
    with ThreadPoolExecutor(max_workers=1) as pool:
        delay = pool.submit(enrollment._advance_pass).result(timeout=5)
    enrollment._refresh_pending()
    return delay


def test_all_registered_projects_are_prepared_in_fair_bounded_passes(tmp_path, monkeypatch):
    runtime = MachineRuntime(tmp_path / "machine")
    entries = [register(runtime, tmp_path, f"project-{index}") for index in range(9)]
    clock = [0.0]
    monkeypatch.setattr(recovery_enrollment, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    worker_calls = []
    original = runtime.prepare_recovery_registration
    foreground = get_ident()

    def prepare(binding, *, blocking=False):
        assert get_ident() != foreground
        worker_calls.append(binding)
        return original(binding, blocking=blocking)

    monkeypatch.setattr(runtime, "prepare_recovery_registration", prepare)
    advance = recovery_enrollment.RecoveryCapture.advance_step

    def after_registration(capture):
        assert all(read_json(path)["registration"]["version"] == 2 for _binding, path in entries)
        return advance(capture)

    monkeypatch.setattr(recovery_enrollment.RecoveryCapture, "advance_step", after_registration)
    enrollment = RecoveryEnrollment(runtime)
    try:
        with runtime.scheduler_authority() as acquired:
            assert acquired
            for expected in (4, 8, 9):
                before = len(worker_calls)
                run_pass(enrollment)
                assert len(worker_calls) - before <= 4
                assert len(set(worker_calls)) == expected
                assert sum(read_json(path)["registration"]["version"] == 2 for _binding, path in entries) == expected
                assert not _machine_is_true_idle(runtime, has_consumed_binding=True)
                clock[0] += 1
            for _ in range(3):
                if not runtime.recovery_enrollment_pending_projects:
                    break
                before = len(worker_calls)
                run_pass(enrollment)
                assert len(worker_calls) - before <= 4
                clock[0] += 1
            assert not runtime.recovery_enrollment_pending_projects
            assert enrollment._settled == {binding for binding, _path in entries}
            run_pass(enrollment)
            assert enrollment._thread is None
    finally:
        enrollment.stop()


def test_pending_project_retries_without_starving_new_or_disabled_bindings(tmp_path, monkeypatch):
    runtime = MachineRuntime(tmp_path / "machine")
    first, first_path = register(runtime, tmp_path, "first")
    others = [register(runtime, tmp_path, f"other-{index}") for index in range(5)]
    disabled = runtime.set_enabled(others[-1][0].project_id, False)
    clock = [0.0]
    monkeypatch.setattr(recovery_enrollment, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    original = runtime.prepare_recovery_registration
    unavailable = [True]

    def prepare(binding, *, blocking=False):
        if binding == first and unavailable[0]:
            raise OSError("first shared root is unavailable")
        return original(binding, blocking=blocking)

    monkeypatch.setattr(runtime, "prepare_recovery_registration", prepare)
    enrollment = RecoveryEnrollment(runtime)
    try:
        with runtime.scheduler_authority() as acquired:
            assert acquired
            for _ in range(4):
                run_pass(enrollment)
                clock[0] += 1
            assert runtime.recovery_enrollment_pending_projects == {first.project_id}
            assert read_json(first_path)["registration"]["version"] == 1
            assert all(read_json(path)["registration"]["version"] == 2 for _binding, path in others)
            assert disabled in runtime.load_registry()[1]
            added, added_path = register(runtime, tmp_path, "added")
            unavailable[0] = False
            clock[0] += 1
            run_pass(enrollment)
            assert not runtime.recovery_enrollment_pending_projects
            assert read_json(first_path)["registration"]["version"] == 2
            assert read_json(added_path)["registration"]["version"] == 2
            assert added in enrollment._settled
    finally:
        enrollment.stop()


def test_binding_changes_invalidate_success_and_superseded_generation_settles(tmp_path, monkeypatch):
    runtime = MachineRuntime(tmp_path / "machine")
    binding, _path = register(runtime, tmp_path, "project")
    clock = [0.0]
    monkeypatch.setattr(recovery_enrollment, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    enrollment = RecoveryEnrollment(runtime)
    try:
        with runtime.scheduler_authority() as acquired:
            assert acquired
            run_pass(enrollment)
            assert enrollment._settled == {binding}
            disabled = runtime.set_enabled(binding.project_id, False)
            runtime._supersede_registration(disabled, "replacement")
            clock[0] += 1
            run_pass(enrollment)
            assert enrollment._settled == {disabled}
            assert not runtime.recovery_enrollment_pending_projects
            assert runtime.registration_status(disabled)["state"] == "superseded"
    finally:
        enrollment.stop()


def test_slow_worker_does_not_block_poll_or_start_duplicate_worker(tmp_path, monkeypatch):
    runtime = MachineRuntime(tmp_path / "machine")
    binding, _path = register(runtime, tmp_path, "project")
    started = Event()
    release = Event()
    calls = []
    original = runtime.prepare_recovery_registration

    def slow_prepare(candidate, *, blocking=False):
        started.set()
        assert release.wait(5)
        calls.append(candidate)
        return original(candidate, blocking=blocking)

    monkeypatch.setattr(runtime, "prepare_recovery_registration", slow_prepare)
    enrollment = RecoveryEnrollment(runtime)
    try:
        with runtime.scheduler_authority() as acquired:
            assert acquired
            enrollment.poll()
            assert started.wait(5)
            first_thread = enrollment._thread
            for _ in range(3):
                enrollment.poll()
                assert enrollment._thread is first_thread
            assert not calls
            release.set()
            collect(enrollment)
            assert calls == [binding]
    finally:
        release.set()
        enrollment.stop()


def test_real_global_agent_prepares_registered_roots_before_on_demand_exit(tmp_path):
    from qqtools.plugins.qexp.agent.recovery_capture import inspect_recovery_capture
    from qqtools.plugins.qexp.runtime.group_namespace import is_group_authority_isolated
    from qqtools.plugins.qexp.runtime.responsibility_completion import is_source_released, read_capture_completion

    runtime = MachineRuntime(tmp_path / "machine")
    entries = [register(runtime, tmp_path, f"project-{index}") for index in range(5)]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[3] / "src")
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; from qqtools.plugins.qexp.agent.lifecycle import run_machine_agent_loop; "
            "run_machine_agent_loop(sys.argv[1], loop_interval=0.05, available_gpus=[])",
            str(runtime.root),
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
                "versions": [read_json(path)["registration"]["version"] for _binding, path in entries],
                "registration_status": [runtime.registration_status(binding) for binding, _path in entries],
                "checkpoints": [
                    [p.name for p in runtime.project_paths(binding.project_id)["root"].glob("responsibility-*.json")]
                    for binding, _path in entries
                ],
            },
        )
        # Capture proof precedes Group activation and retained-source release.
        # Keep those obligations inside the existing 15-second enrollment budget;
        # the unchanged five-second idle budget starts after all three finish.
        wait_until(
            "all-enrollment-obligations-settled",
            lambda: all(enrollment_has_settled(binding) for binding, _path in entries),
            stage="recovery-enrollment:capture",
            timeout=15,
            on_timeout=lambda: {
                binding.project_id: {
                    "capture": inspect_recovery_capture(runtime, binding),
                    "blockers": runtime.binding_blockers(binding),
                }
                for binding, _path in entries
            },
        )
        assert process.wait(timeout=5) == 0
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


def test_failed_thread_start_is_retryable(tmp_path, monkeypatch):
    runtime = MachineRuntime(tmp_path / "machine")
    binding, path = register(runtime, tmp_path, "project")
    enrollment = RecoveryEnrollment(runtime)
    start = recovery_enrollment.Thread.start

    def unavailable(_thread):
        raise RuntimeError("cannot start thread")

    monkeypatch.setattr(recovery_enrollment.Thread, "start", unavailable)
    try:
        with runtime.scheduler_authority() as acquired:
            assert acquired
            with pytest.raises(RuntimeError, match="cannot start"):
                enrollment.poll()
            assert enrollment._thread is None
            assert runtime.recovery_enrollment_pending_projects == {binding.project_id}
            monkeypatch.setattr(recovery_enrollment.Thread, "start", start)
            enrollment.poll()
            collect(enrollment)
            assert read_json(path)["registration"]["version"] == 2
    finally:
        enrollment.stop()


def test_background_preparation_waits_for_migration_then_finishes_same_pass(tmp_path, monkeypatch):
    from contextlib import contextmanager

    runtime = MachineRuntime(tmp_path / "machine")
    _binding, path = register(runtime, tmp_path, "project")
    attempted = Event()
    guard = runtime.migration_guard
    read_guard = runtime.migration_read_guard

    @contextmanager
    def observe_guard(*, blocking=True):
        assert blocking
        attempted.set()
        with read_guard(blocking=blocking) as acquired:
            yield acquired

    monkeypatch.setattr(runtime, "migration_read_guard", observe_guard)
    enrollment = RecoveryEnrollment(runtime)
    try:
        with runtime.scheduler_authority() as acquired:
            assert acquired
            with guard():
                enrollment.poll()
                assert attempted.wait(5)
                assert enrollment._thread.is_alive()
                assert read_json(path)["registration"]["version"] == 1
            collect(enrollment)
            assert not runtime.recovery_enrollment_pending_projects
            assert read_json(path)["registration"]["version"] == 2
    finally:
        enrollment.stop()


def test_malformed_binding_does_not_abort_other_projects_in_pass(tmp_path):
    from qqtools.plugins.qexp.runtime.store import atomic_replace

    runtime = MachineRuntime(tmp_path / "machine")
    damaged, damaged_path = register(runtime, tmp_path, "damaged")
    _healthy, healthy_path = register(runtime, tmp_path, "healthy")
    atomic_replace(damaged_path.parent / "machine.json", {"machine": None})
    enrollment = RecoveryEnrollment(runtime)
    try:
        with runtime.scheduler_authority() as acquired:
            assert acquired
            run_pass(enrollment)
            assert runtime.recovery_enrollment_pending_projects == {damaged.project_id}
            assert read_json(damaged_path)["registration"]["version"] == 1
            assert read_json(healthy_path)["registration"]["version"] == 2
    finally:
        enrollment.stop()


def test_paused_machine_rollout_waits_without_stalling_other_projects(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.agent.lifecycle import get_machine_agent_status

    runtime = MachineRuntime(tmp_path / "machine")
    first, _path = register(runtime, tmp_path, "first")
    other, _other_path = register(runtime, tmp_path, "other")
    peer_cfg = init_shared_root(first.shared_root, "peer", runtime_root=tmp_path / "peer-local")
    peer = MachineRuntime(tmp_path / "peer-machine")
    peer_binding = peer.add_binding(peer_cfg.shared_root, peer_cfg.machine_name)
    clock = [0.0]
    monkeypatch.setattr(recovery_enrollment, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    enrollment = RecoveryEnrollment(runtime)
    try:
        with runtime.scheduler_authority() as acquired:
            assert acquired
            run_pass(enrollment)
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
        clock[0] += 1
        with runtime.scheduler_authority() as acquired:
            assert acquired
            run_pass(enrollment)
            assert not runtime.recovery_enrollment_pending_projects
        projects = {item["project_id"]: item for item in get_machine_agent_status(runtime)["projects"]}
        assert projects[first.project_id]["recovery_enrollment"]["state"] == "admission_fenced"
    finally:
        enrollment.stop()
