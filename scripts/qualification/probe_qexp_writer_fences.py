"""Exercise released writer gates and lightweight process continuity in isolated roots."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread

CASES = (
    "worker_cached_legacy",
    "worker_cached_group_floor",
    "worker_raced_group_floor",
    "worker_raced_namespace",
    "worker_unknown_operation",
    "baseline_dispatch",
    "cached_dispatch",
    "cached_claim",
    "cached_runner",
    "cached_save_task",
    "raced_claim",
    "raced_runner",
    "late_exit",
    "cached_save_task_ready_gate",
    "raced_save_task_ready_gate",
    "absent_save_task_ready_gate",
    "live_namespace_upgrade",
    "paused_namespace_upgrade",
    "live_runner_exit",
    "live_delayed_registration",
    "missing_task_cleanup",
    "live_erased_evidence",
    "registration_baseline",
    "registration_dispatch",
    "registration_raced_dispatch",
    "registration_reactivation",
    "registration_runner",
    "registration_peer",
)
CAPABILITY = "local-recovery-v1"
WORKLOAD = """
import os
import sys
import time
from pathlib import Path
root = Path(sys.argv[1])
with (root / "starts.txt").open("a") as stream:
    stream.write(str(os.getpid()) + "\\n")
deadline = time.monotonic() + 20
while not (root / "release").exists():
    if time.monotonic() >= deadline:
        raise TimeoutError("probe did not release its workload")
    time.sleep(0.01)
(root / "finished").write_text(str(os.getpid()))
"""


def begin_capture_with_current_code(cfg, root, *, runtime_root=None, legacy_source=None) -> dict:
    """Durably retain discovery before the released fixture enumerates processes."""
    source = Path(__file__).resolve().parents[2] / "src"
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--begin-writer-capture",
            str(runtime_root or cfg.runtime_root),
            str(source),
            *([str(legacy_source)] if legacy_source is not None else []),
        ],
        env=dict(os.environ, PYTHONPATH=str(source)),
        capture_output=True,
        text=True,
        timeout=10,
    )
    (root / "capture-begin.stdout").write_text(completed.stdout)
    (root / "capture-begin.stderr").write_text(completed.stderr)
    completed.check_returncode()
    return json.loads(completed.stdout)


def begin_capture_child(runtime_root: Path, source: Path, legacy_source: Path | None = None) -> dict:
    from qqtools.plugins.qexp.runtime import responsibility_capture
    from qqtools.plugins.qexp.runtime.locks import exclusive
    from qqtools.plugins.qexp.runtime.responsibility import responsibility_root
    from qqtools.plugins.qexp.runtime.responsibility_store import Ledger
    from qqtools.plugins.qexp.runtime.store import read_json

    assert Path(responsibility_capture.__file__).resolve().is_relative_to(source.resolve())
    with exclusive(runtime_root / "locks" / "responsibility-initialize.lock"):
        ledger = Ledger.open_or_create(responsibility_root(runtime_root))
    # This fixture owns the isolated runtime's lifetime and has no real binding.
    with responsibility_capture.WriterCaptureCheckpoint(ledger, runtime_root, legacy_source=legacy_source).observe():
        pass
    return read_json(runtime_root / responsibility_capture.CAPTURE_FILE)


def capture_with_current_code(
    cfg, task, attempt, root, label, *, writers=None, runtime_root=None, legacy_source=None
) -> dict:
    """Import legacy evidence using this checkout, isolated from released imports."""
    source = Path(__file__).resolve().parents[2] / "src"
    environment = dict(os.environ, PYTHONPATH=str(source))
    writer_arguments = []
    if writers is not None:
        writer_file = root / f"captured-writers-{label}.json"
        writer_file.write_text(
            json.dumps(
                {
                    "writers": writers,
                    "shared_root": str(cfg.shared_root),
                    "machine_name": cfg.machine_name,
                    "legacy_source": str(legacy_source) if legacy_source is not None else None,
                }
            )
        )
        writer_arguments = [str(writer_file)]
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--capture-child",
            str(runtime_root or cfg.runtime_root),
            attempt.attempt_id,
            task.task_id,
            str(source),
            *writer_arguments,
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    (root / f"capture-{label}.stdout").write_text(completed.stdout)
    (root / f"capture-{label}.stderr").write_text(completed.stderr)
    completed.check_returncode()
    return json.loads(completed.stdout)


def capture_child(
    runtime_root: Path, attempt_id: str, task_id: str, source: Path, writer_file: Path | None = None
) -> dict:
    from qqtools.plugins.qexp.runtime import responsibility_backfill
    from qqtools.plugins.qexp.runtime.locks import exclusive
    from qqtools.plugins.qexp.runtime.responsibility import responsibility_root
    from qqtools.plugins.qexp.runtime.responsibility_capture import CAPTURE_FILE
    from qqtools.plugins.qexp.runtime.responsibility_cleanup import captured_writers_are_quiescent
    from qqtools.plugins.qexp.runtime.responsibility_store import Ledger

    module = Path(responsibility_backfill.__file__).resolve()
    assert module.is_relative_to(source.resolve()), module
    process_pages = []
    if writer_file is not None:
        from qqtools.plugins.qexp.config_types import RootConfig
        from qqtools.plugins.qexp.runtime import responsibility_process_capture

        assert Path(responsibility_process_capture.__file__).resolve().is_relative_to(source.resolve())
        with exclusive(runtime_root / "locks" / "responsibility-initialize.lock"):
            ledger = Ledger.open_or_create(responsibility_root(runtime_root))
        inputs = json.loads(writer_file.read_text())
        shared_root = Path(inputs["shared_root"])
        cfg = RootConfig(shared_root, shared_root.parent, inputs["machine_name"], runtime_root)
        legacy_source = Path(inputs["legacy_source"]) if inputs["legacy_source"] else None
        census = responsibility_process_capture.RunnerProcessCapture(cfg, ledger, legacy_source=legacy_source)
        try:
            for _ in range(10000):
                started = time.perf_counter()
                page = census.take(7)
                process_pages.append(
                    {
                        "entries_visited": page.entries_visited,
                        "writers_recorded": page.writers_recorded,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
                assert page.entries_visited <= 7
                if page.is_sweep_complete:
                    break
            else:
                raise AssertionError("bounded runner process census did not finish")
            for observed in inputs["writers"]:
                assert observed["attempt_id"] == attempt_id and observed["task_id"] == task_id
                writer = {
                    key: observed[key] for key in ("host_id", "boot_id", "pid_namespace", "pid", "start_time_ticks")
                }
                assert writer in ledger.lookup(attempt_id)["captured_writers"]
            if legacy_source is not None:
                assert ledger.lookup(attempt_id)["legacy_source"] == str(legacy_source)
        finally:
            census.close()
    capture = responsibility_backfill.ResponsibilityBackfill(runtime_root)
    visited = 0
    try:
        for _ in range(100):
            progress = capture.take(2)
            assert progress is not None
            visited += progress.entries_visited
            if progress.is_sweep_complete:
                break
        else:
            raise AssertionError("bounded legacy capture did not finish")
    finally:
        capture.close()
    member = Ledger(responsibility_root(runtime_root)).lookup(attempt_id)
    assert member["payload"] == {"task_id": task_id, "attempt_number": 1}
    assert member["stage"] == "active"
    return {
        "process_pages": process_pages,
        "source": str(module),
        "capture_id": progress.capture_id,
        "entries_visited": visited,
        "member": member,
        "captured_writers_quiescent": captured_writers_are_quiescent(member),
        "writer_capture_checkpoint": (
            json.loads((runtime_root / CAPTURE_FILE).read_text()) if (runtime_root / CAPTURE_FILE).exists() else None
        ),
    }


def delayed_runner(arguments: list[str]) -> int:
    """Pause the actual released registration writer after real process creation."""
    from qqtools.plugins.qexp import runner

    root = Path(arguments[0])
    original = runner._publish_registration

    def publish(*args, **kwargs):
        (root / "registration-paused").touch()
        deadline = time.monotonic() + 15
        while not (root / "registration-release").exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("probe did not release registration")
            time.sleep(0.01)
        return original(*args, **kwargs)

    runner._publish_registration = publish
    return runner.main(arguments[1:])


def runner_process_inventory(cfg) -> dict:
    """Characterize CLI runner locators and count Linux process inventory cost.

    This is a read-only experiment, not a durable migration or completeness
    certificate. It neither captures embedded Python writers nor fences launches.
    """
    from qqtools.plugins.qexp.infrastructure.host import host_instance_id

    matches = []
    host_id = host_instance_id()
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    pid_namespace = Path("/proc/self/ns/pid").stat().st_ino
    pages = []
    visited = 0
    with os.scandir("/proc") as entries:
        for entry in entries:
            visited += 1
            if visited == 64:
                pages.append(visited)
                visited = 0
            if not entry.name.isascii() or not entry.name.isdecimal():
                continue
            process = Path(entry.path)
            try:
                if entry.stat(follow_symlinks=False).st_uid != os.getuid():
                    continue
                before = (process / "stat").read_text().rsplit(")", 1)[1].split()
                with (process / "cmdline").open("rb") as stream:
                    raw = stream.read(65537)
                after = (process / "stat").read_text().rsplit(")", 1)[1].split()
            except (FileNotFoundError, ProcessLookupError):
                continue
            if before[19] != after[19] or after[0] == "Z":
                continue
            if len(raw) > 65536:
                raise RuntimeError("process command exceeds qualification inventory bound")
            arguments = os.fsdecode(raw).rstrip("\0").split("\0")
            if arguments[1:3] != ["-m", "qqtools.plugins.qexp.runner"]:
                continue
            if arguments[3:4] == ["--guardian"]:
                continue
            options = arguments[3:]
            if len(options) % 2 or len(set(options[::2])) != len(options[::2]):
                raise RuntimeError("unclassified released runner arguments")
            fields = dict(zip(options[::2], options[1::2]))
            if (
                fields.get("--shared-root") != str(cfg.shared_root)
                or fields.get("--runtime-root") != str(cfg.runtime_root)
                or fields.get("--machine") != cfg.machine_name
            ):
                continue
            matches.append(
                {
                    "host_id": host_id,
                    "boot_id": boot_id,
                    "pid_namespace": pid_namespace,
                    "pid": int(entry.name),
                    "start_time_ticks": int(after[19]),
                    "task_id": fields["--task-id"],
                    "attempt_id": fields["--attempt-id"],
                    "fencing_token": int(fields["--fencing-token"]),
                    "launch_id": fields["--launch-id"],
                }
            )
    if visited:
        pages.append(visited)
    return {"matches": matches, "directory_entries_per_64_entry_group": pages}


def observe_live_runner(
    cfg, task, attempt, root, transition, *, should_delay_registration=False, should_erase_evidence=False
) -> dict:
    """Keep a real released wrapper/guardian/child alive across the writer fence."""
    from qqtools.plugins.qexp.runner import observation_path, registration_path
    from qqtools.plugins.qexp.runtime.store import read_json

    command = [
        sys.executable,
        "-m",
        "qqtools.plugins.qexp.runner",
        "--shared-root",
        str(cfg.shared_root),
        "--runtime-root",
        str(cfg.runtime_root),
        "--machine",
        cfg.machine_name,
        "--task-id",
        task.task_id,
        "--attempt-id",
        attempt.attempt_id,
        "--fencing-token",
        str(attempt.current_fencing_token),
        "--launch-id",
        attempt.authorization["launch_id"],
    ]
    if should_delay_registration:
        command[1:3] = [str(Path(__file__).resolve()), "--delayed-runner", str(root)]
    with (root / "runner.stdout").open("w") as stdout, (root / "runner.stderr").open("w") as stderr:
        child = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            deadline = time.monotonic() + 10
            registration = registration_path(cfg, attempt.attempt_id)
            signal = root / "registration-paused" if should_delay_registration else registration
            while not signal.exists() or not (root / "starts.txt").exists():
                assert child.poll() is None, "released runner exited before registration"
                if time.monotonic() >= deadline:
                    raise TimeoutError("released runner did not register its workload")
                time.sleep(0.01)
            starts = (root / "starts.txt").read_text().splitlines()
            assert len(starts) == 1 and starts[0].isdigit(), starts
            captured = None
            if should_delay_registration:
                assert not registration.exists()
                captured = capture_with_current_code(cfg, task, attempt, root, "before")
                assert not registration.exists() and not observation_path(cfg, attempt.attempt_id).exists()
            else:
                assert read_json(registration)["process_registration"]["wrapper_pid"] == child.pid
            erased_inventory = None
            if should_erase_evidence:
                from qqtools.plugins.qexp.commands.cleanup import clean
                from qqtools.plugins.qexp.runtime.paths import attempt_path, local_paths, task_path
                from qqtools.plugins.qexp.scheduler import fail_attempt

                # Actual released public compensation and cleanup erase the
                # pre-spawn intent/registration even though the wrapper is live.
                assert fail_attempt(
                    cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "handoff_failed"
                )
                clean(cfg, task_id=task.task_id)
                assert not task_path(cfg.shared_root, task.task_id).exists()
                assert not attempt_path(cfg.shared_root, task.task_id, 1).exists()
                paths = local_paths(cfg.runtime_root)
                for name in ("processes", "registrations", "observations", "launch_intents", "wrappers"):
                    assert not (paths[name] / f"{attempt.attempt_id}.json").exists()
                captured_runtime = root / "captured/runtime"
                pending_capture = begin_capture_with_current_code(
                    cfg, root, runtime_root=captured_runtime, legacy_source=cfg.runtime_root
                )
                assert pending_capture["phase"] == "pending" and pending_capture["pending"] == []
                erased_inventory = runner_process_inventory(cfg)
                assert len(erased_inventory["matches"]) == 1, erased_inventory
                member = erased_inventory["matches"][0]
                assert member["pid"] == child.pid
                assert member["task_id"] == task.task_id and member["attempt_id"] == attempt.attempt_id
                assert member["fencing_token"] == attempt.current_fencing_token
                assert member["launch_id"] == attempt.authorization["launch_id"]
                captured = capture_with_current_code(
                    cfg,
                    task,
                    attempt,
                    root,
                    "before",
                    writers=erased_inventory["matches"],
                    runtime_root=captured_runtime,
                    legacy_source=cfg.runtime_root,
                )
                assert not captured["captured_writers_quiescent"]
                assert captured["writer_capture_checkpoint"]["capture_id"] == pending_capture["capture_id"]
            transition()
            assert child.poll() is None, "writer transition terminated the released runner"
            os.kill(int(starts[0]), 0)
            assert not (root / "finished").exists()
            (root / "registration-release").touch()
            (root / "release").touch()
            assert child.wait(timeout=10) == 0
            assert (root / "finished").read_text() == starts[0]
            assert (root / "starts.txt").read_text().splitlines() == starts
            observation = read_json(observation_path(cfg, attempt.attempt_id))["exit_observation"]
            assert observation["observed_exit_code"] == 0 and observation["task_id"] == task.task_id
            if not should_erase_evidence:
                assert read_json(registration)["process_registration"]["wrapper_pid"] == child.pid
            else:
                assert not registration.exists()
            if captured is not None:
                resumed = capture_with_current_code(
                    cfg, task, attempt, root, "after", runtime_root=captured_runtime if should_erase_evidence else None
                )
                assert resumed["entries_visited"] == 0
                assert resumed["capture_id"] == captured["capture_id"]
                assert resumed["member"] == captured["member"]
                if should_erase_evidence:
                    assert resumed["captured_writers_quiescent"]
                    assert resumed["writer_capture_checkpoint"] == captured["writer_capture_checkpoint"]
            return {
                "wrapper_pid": child.pid,
                "workload_pid": int(starts[0]),
                "launch_count": 1,
                "exit_code": 0,
                "captured_before_registration": captured,
                "inventory_after_old_cleanup": erased_inventory,
            }
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)


def across_schema_transition(operation, transition):
    """Pause released code after its initial check, before its shared fence."""
    from qqtools.plugins.qexp.runtime import locks

    entered, resume = Event(), Event()
    original = locks.schema_reader_lock
    outcomes = []

    @contextmanager
    def delayed(*args, **kwargs):
        entered.set()
        if not resume.wait(5):
            raise TimeoutError("schema transition did not release the writer")
        with original(*args, **kwargs) as acquired:
            yield acquired

    def writer():
        try:
            operation()
        except BaseException as exc:
            outcomes.append(exc)

    locks.schema_reader_lock = delayed
    thread = Thread(target=writer)
    try:
        thread.start()
        assert entered.wait(5), outcomes
        transition()
    finally:
        resume.set()
        thread.join(5)
        locks.schema_reader_lock = original
    assert not thread.is_alive(), "test-owned released writer did not finish"
    if outcomes:
        raise outcomes[0]


def probe_missing_task_cleanup(cfg, module: Path) -> dict:
    """Characterize an old public reconciler after Task deletion but before Attempt deletion."""
    from qqtools.plugins.qexp import submit
    from qqtools.plugins.qexp.commands import cleanup
    from qqtools.plugins.qexp.runtime.locks import schema_lock
    from qqtools.plugins.qexp.runtime.paths import attempt_path, ready_state_path, task_path
    from qqtools.plugins.qexp.runtime.ready import advance_ready_index_build
    from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
    from qqtools.plugins.qexp.scheduler import claim_task, fail_attempt

    task = submit(cfg, ["true"])
    advance_ready_index_build(cfg)
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert fail_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "fixture")
    remaining = attempt_path(cfg.shared_root, task.task_id, 1)
    before = remaining.read_bytes()
    original = cleanup.shutil.rmtree

    def interrupt(directory, *args, **kwargs):
        if directory == remaining.parent:
            raise OSError("qualification cleanup interrupted after Task deletion")
        return original(directory, *args, **kwargs)

    cleanup.shutil.rmtree = interrupt
    try:
        cleanup.clean(cfg, task_id=task.task_id)
    except OSError as exc:
        assert str(exc) == "qualification cleanup interrupted after Task deletion"
    else:
        raise AssertionError("cleanup did not reach the injected interruption")
    finally:
        cleanup.shutil.rmtree = original
    assert not task_path(cfg.shared_root, task.task_id).exists()
    assert remaining.read_bytes() == before
    with schema_lock(cfg.shared_root):
        path = ready_state_path(cfg.shared_root)
        value = read_json(path)
        assert value["ready_index"]["state"] in {"building", "active"}
        value["ready_index"]["writer_capability"] = "qualification-ready-writer-v2"
        atomic_replace(path, value)
        path = cfg.shared_root / "schema" / "version.json"
        value = read_json(path)
        value["schema"]["required_capabilities"].append(CAPABILITY)
        atomic_replace(path, value)
    result = cleanup.reconcile_cleanup_operations(cfg)
    assert any(item["state"] == "completed" for item in result), result
    assert not remaining.exists(), "released cleanup unexpectedly honored the writer gate"
    return {
        "case": "missing_task_cleanup",
        "source": str(module),
        "error": None,
        "task_changed": False,
        "remaining_attempt_removed": True,
    }


def probe_registration_boundary(case: str, cfg, root: Path, module: Path) -> dict:
    """Characterize machine admission separately from passive runner admission."""
    from types import SimpleNamespace

    from qqtools.plugins.qexp import init_shared_root, submit
    from qqtools.plugins.qexp.agent.context import MachineRuntime
    from qqtools.plugins.qexp.agent.lifecycle import dispatch_machine_cycle
    from qqtools.plugins.qexp.layout import load_machine_registration, save_machine_registration
    from qqtools.plugins.qexp.runner import run_attempt
    from qqtools.plugins.qexp.runtime.paths import attempt_path, task_path
    from qqtools.plugins.qexp.runtime.store import read_json
    from qqtools.plugins.qexp.scheduler import authorize_launch, claim_task

    runtime = MachineRuntime(root / "registered-machine")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    assert runtime.registration_status(binding)["write_eligible"]
    if case == "registration_peer":
        selected_cfg = init_shared_root(cfg.shared_root, "peer-machine", runtime_root=root / "peer-local")
        selected_runtime = MachineRuntime(root / "peer-runtime")
        selected_runtime.add_binding(cfg.shared_root, selected_cfg.machine_name)
    else:
        selected_cfg, selected_runtime = cfg, runtime
    task = submit(selected_cfg, ["true"])
    attempt = None
    if case == "registration_runner":
        attempt = claim_task(cfg, task.task_id, [0])
        assert attempt is not None
        assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    before = task_path(cfg.shared_root, task.task_id).read_bytes()
    original_guard = runtime._registration_guard

    def publish_registration():
        with original_guard(cfg):
            registration = load_machine_registration(cfg)
            registration["registration"].update(version=2, protocol_version=2)
            save_machine_registration(cfg, registration)

    if case not in {"registration_baseline", "registration_raced_dispatch"}:
        publish_registration()
    registration_before = load_machine_registration(cfg)
    launches = []

    class RecordingExecutor:
        def launch_attempt(self, _cfg, task_id, _attempt):
            launches.append(task_id)

    def spawn(*_args, **_kwargs):
        launches.append(task.task_id)
        return SimpleNamespace(pid=99999991, wait=lambda: 0)

    error = None
    result = None
    transition = None
    entered, release = Event(), Event()

    @contextmanager
    def delayed_guard(*args, **kwargs):
        entered.set()
        if not release.wait(10):
            raise TimeoutError("registration fence transition did not finish")
        with original_guard(*args, **kwargs):
            yield

    def transition_registration():
        if entered.wait(10):
            publish_registration()
            release.set()

    try:
        if case == "registration_raced_dispatch":
            runtime._registration_guard = delayed_guard
            transition = Thread(target=transition_registration)
            transition.start()
        if case == "registration_reactivation":
            result = runtime.reactivate_binding(binding)
        elif case == "registration_runner":
            value = read_json(attempt_path(cfg.shared_root, task.task_id, 1))["attempt"]
            result = run_attempt(
                cfg,
                task.task_id,
                attempt.attempt_id,
                attempt.current_fencing_token,
                value["authorization"]["launch_id"],
                popen_factory=spawn,
            )
        else:
            result = dispatch_machine_cycle(selected_runtime, available_gpus=[0], executor=RecordingExecutor())
    except RuntimeError as exc:
        if "unsupported protocol version" not in str(exc):
            raise
        error = str(exc)
    finally:
        runtime._registration_guard = original_guard
        if transition is not None:
            release.set()
            transition.join(15)
            assert not transition.is_alive() and entered.is_set()
    if case == "registration_raced_dispatch":
        registration_before = {
            "registration": {**registration_before["registration"], "version": 2, "protocol_version": 2}
        }
    should_launch = case in {"registration_baseline", "registration_runner", "registration_peer"}
    assert len(launches) == int(should_launch), (case, launches, result, error)
    assert (error is not None) == (case == "registration_reactivation"), (case, error)
    changed = task_path(cfg.shared_root, task.task_id).read_bytes() != before
    assert changed == (case in {"registration_baseline", "registration_peer"}), (case, changed)
    if case != "registration_baseline":
        assert load_machine_registration(cfg) == registration_before
    registration_status = runtime.registration_status(binding)
    if case in {"registration_dispatch", "registration_raced_dispatch"}:
        assert result == [
            {
                "project_id": binding.project_id,
                "launched": [],
                "status": "registration_ineligible",
                "error": (
                    "current registration generation or machine write eligibility is unavailable; "
                    "new admission is blocked while retained execution evidence is reconciled."
                ),
            }
        ], result
        assert registration_status["state"] == "invalid"
        assert registration_status["error"] == "project machine registration uses an unsupported protocol version."
    return {
        "case": case,
        "module": str(module),
        "task_changed": changed,
        "launches": len(launches),
        "error": error,
        "result": result,
        "registration_status": registration_status,
        "execution_mode": (
            "injected_process_factory"
            if case == "registration_runner"
            else "registration_only"
            if case == "registration_reactivation"
            else "recording_executor"
        ),
        "child_process_created": False,
        "training_executed": False,
    }


def probe_worker_boundary(case, cfg, module):
    from qqtools.plugins.qexp.commands.group import (
        _reconcile_worker_remove_operation,
        change_worker,
        create_group,
        reconcile_group_cancel_operations,
    )
    from qqtools.plugins.qexp.commands.task import cancel, submit
    from qqtools.plugins.qexp.runtime.locks import schema_lock
    from qqtools.plugins.qexp.runtime.operation_store import active_operation_path
    from qqtools.plugins.qexp.runtime.paths import group_path
    from qqtools.plugins.qexp.runtime.ready import advance_ready_index_build, read_ready_index_state
    from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json

    while read_ready_index_state(cfg) != "active":
        advance_ready_index_build(cfg)
    create_group(cfg, "experiment")
    task = submit(cfg, ["true"], group="experiment")
    control = change_worker(cfg, "experiment", cfg.machine_name, "remove")["worker_control"]
    operation_path = active_operation_path(cfg, "group_control", control["operation_id"])
    cached = read_json(operation_path)
    change_worker(cfg, "experiment", cfg.machine_name, "add")
    cancel(cfg, task.task_id)
    with schema_lock(cfg.shared_root):
        path = cfg.shared_root / "schema/version.json"
        value = read_json(path)
        value["schema"]["required_capabilities"].append(CAPABILITY)
        atomic_replace(path, value)
        if case == "worker_cached_group_floor":
            path = cfg.shared_root / "indexes/ready/group-members/state.json"
            value = read_json(path)
            value["group_ready_members"]["schema_version"] = 2
            atomic_replace(path, value)
        elif case == "worker_unknown_operation":
            cached["group_control"]["operation_type"] = "worker_remove_v2"
            atomic_replace(operation_path, cached)

    def publish_group_floor():
        with schema_lock(cfg.shared_root):
            floor_path = cfg.shared_root / "indexes/ready/group-members/state.json"
            value = read_json(floor_path)
            value["group_ready_members"]["schema_version"] = 2
            atomic_replace(floor_path, value)

    def isolate_namespace():
        source = Path(__file__).resolve().parents[2] / "src"
        code = """
from pathlib import Path
import sys
from qqtools.plugins.qexp.config_types import RootConfig
from qqtools.plugins.qexp.runtime.group_namespace import activate_group_authority_locked
from qqtools.plugins.qexp.runtime.locks import schema_lock
cfg = RootConfig(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]))
with schema_lock(cfg.shared_root):
    assert activate_group_authority_locked(cfg)
"""
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                str(cfg.shared_root),
                str(cfg.project_root),
                cfg.machine_name,
                str(cfg.runtime_root),
            ],
            env=dict(os.environ, PYTHONPATH=str(source)),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr)

    path = group_path(cfg.shared_root, "experiment")
    before = path.read_bytes()
    error = None
    try:
        if case == "worker_raced_namespace":
            across_schema_transition(
                lambda: _reconcile_worker_remove_operation(cfg, operation_path, cached, None), isolate_namespace
            )
        elif case == "worker_raced_group_floor":
            across_schema_transition(
                lambda: _reconcile_worker_remove_operation(cfg, operation_path, cached, None), publish_group_floor
            )
        elif case == "worker_unknown_operation":
            reconcile_group_cancel_operations(cfg)
        else:
            _reconcile_worker_remove_operation(cfg, operation_path, cached, None)
    except RuntimeError as exc:
        if "ordinary mutation is disabled" not in str(exc):
            raise
        error = str(exc)
    if case == "worker_raced_namespace":
        path = cfg.shared_root / "groups-v2/experiment.json"
    changed = path.read_bytes() != before
    assert changed == (case in {"worker_cached_legacy", "worker_raced_group_floor"}), (case, changed, error)
    assert (error is not None) == (case == "worker_cached_group_floor"), (case, error)
    return {"case": case, "source": str(module), "error": error, "task_changed": False, "group_changed": changed}


def probe(case: str, root: Path, source: Path) -> dict:
    from qqtools.plugins.qexp import init_shared_root, layout, submit
    from qqtools.plugins.qexp.runner import _publish_exit_observation, observation_path, run_attempt
    from qqtools.plugins.qexp.runtime.locks import schema_lock, schema_writer_lock
    from qqtools.plugins.qexp.runtime.paths import ready_state_path, shared_paths, task_path
    from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
    from qqtools.plugins.qexp.runtime.tasks import load_task, save_task
    from qqtools.plugins.qexp.scheduler import authorize_launch, claim_task, run_dispatch_cycle

    module = Path(layout.__file__).resolve()
    if not module.is_relative_to(source.resolve()):
        raise RuntimeError(f"probe imported the wrong checkout: {module}")
    cfg = init_shared_root(root / ".qexp", "probe-machine", runtime_root=root / "local")
    if case in {"live_namespace_upgrade", "paused_namespace_upgrade"}:
        from qexp_namespace_rollout import probe_live_namespace_upgrade

        return probe_live_namespace_upgrade(
            cfg, root, module, WORKLOAD, should_pause=case == "paused_namespace_upgrade"
        )
    if case.startswith("worker_"):
        return probe_worker_boundary(case, cfg, module)
    if case.startswith("registration_"):
        return probe_registration_boundary(case, cfg, root, module)
    if case == "missing_task_cleanup":
        return probe_missing_task_cleanup(cfg, module)
    is_live = case in {"live_runner_exit", "live_delayed_registration", "live_erased_evidence"}
    command = [sys.executable, "-c", WORKLOAD, str(root)] if is_live else ["true"]
    task = submit(cfg, command)
    if (case.endswith("ready_gate") and not case.startswith("absent_")) or is_live:
        from qqtools.plugins.qexp.runtime.ready import advance_ready_index_build, read_ready_index_state

        advance_ready_index_build(cfg)
        assert read_ready_index_state(cfg) in {"building", "active"}
    cached = load_task(cfg, task.task_id)
    launched = []

    class NoProcessExecutor:
        def launch_attempt(self, _cfg, task_id, _attempt):
            launched.append(task_id)

    def forbidden_spawn(*_args, **_kwargs):
        raise AssertionError("writer fence probe attempted to create a workload")

    attempt = None
    if case in {"cached_runner", "raced_runner", "late_exit"} or is_live:
        attempt = claim_task(cfg, task.task_id, [0])
        assert attempt is not None
        assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
        from qqtools.plugins.qexp.runtime.paths import attempt_path
        from qqtools.plugins.qexp.runtime.records import AttemptRecord

        attempt = AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task.task_id, 1)))
    before = task_path(cfg.shared_root, task.task_id).read_bytes()

    def publish_capability():
        with schema_lock(cfg.shared_root):
            if case.endswith("ready_gate") or is_live:
                path = ready_state_path(cfg.shared_root)
                value = read_json(path)
                value["ready_index"]["writer_capability"] = "qualification-ready-writer-v2"
                atomic_replace(path, value)
                if not is_live:
                    return
            path = shared_paths(cfg.shared_root)["schema"] / "version.json"
            value = read_json(path)
            value["schema"]["required_capabilities"].append(CAPABILITY)
            atomic_replace(path, value)

    continuity = None
    if is_live:
        continuity = observe_live_runner(
            cfg,
            task,
            attempt,
            root,
            publish_capability,
            should_delay_registration=case == "live_delayed_registration",
            should_erase_evidence=case == "live_erased_evidence",
        )
    elif case != "baseline_dispatch" and not case.startswith("raced_"):
        publish_capability()

    def operation():
        if case in {"baseline_dispatch", "cached_dispatch"}:
            run_dispatch_cycle(cfg, available_gpus=[0], executor=NoProcessExecutor(), should_recover_starting=False)
        elif case in {"cached_claim", "raced_claim"}:
            claim_task(cfg, task.task_id, [0])
        elif case in {"cached_runner", "raced_runner"} or is_live:
            assert attempt is not None
            run_attempt(
                cfg,
                task.task_id,
                attempt.attempt_id,
                attempt.current_fencing_token,
                attempt.authorization["launch_id"],
                popen_factory=forbidden_spawn,
            )
        elif case in {
            "cached_save_task",
            "cached_save_task_ready_gate",
            "raced_save_task_ready_gate",
            "absent_save_task_ready_gate",
        }:
            with schema_writer_lock(cfg, require_narrow=True):
                cached.meta["revision"] += 1
                save_task(cfg, cached)
        elif case == "late_exit":
            assert attempt is not None
            _publish_exit_observation(cfg, attempt.attempt_id, 0, task_id=task.task_id)
        else:
            raise ValueError(case)

    error = None
    try:
        if case.startswith("raced_"):
            across_schema_transition(operation, publish_capability)
        else:
            operation()
    except RuntimeError as exc:
        if "unsupported capabilities" not in str(exc) and "ready index requires writer capability" not in str(exc):
            raise
        error = str(exc)
    except FileNotFoundError as exc:
        if case != "live_erased_evidence" or Path(exc.filename) != task_path(cfg.shared_root, task.task_id):
            raise
        error = "runner reentry rejected because the Task was deleted"
    has_changed_truth = (
        not task_path(cfg.shared_root, task.task_id).exists()
        or task_path(cfg.shared_root, task.task_id).read_bytes() != before
    )
    should_reject = case in {
        "cached_dispatch",
        "cached_claim",
        "cached_runner",
        "raced_claim",
        "raced_runner",
        "cached_save_task_ready_gate",
        "raced_save_task_ready_gate",
        "live_runner_exit",
        "live_delayed_registration",
        "live_erased_evidence",
    }
    if case != "cached_dispatch":
        assert (error is not None) == should_reject, (case, error)
    assert has_changed_truth == (
        case in {"baseline_dispatch", "cached_save_task", "absent_save_task_ready_gate", "live_erased_evidence"}
    ), (
        case,
        has_changed_truth,
    )
    assert len(launched) == (1 if case == "baseline_dispatch" else 0)
    if case == "late_exit":
        observation = read_json(observation_path(cfg, attempt.attempt_id))["exit_observation"]
        assert observation["observed_exit_code"] == 0 and observation["task_id"] == task.task_id
    return {
        "case": case,
        "source": str(module),
        "error": error,
        "task_changed": has_changed_truth,
        "continuity": continuity,
    }


def run(output: Path, refs: list[str], cases: tuple[str, ...] = CASES) -> None:
    output.mkdir()  # Refuse to reuse any existing runtime or evidence directory.
    repo = Path(__file__).resolve().parents[2]
    probe_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rollout_sha256 = hashlib.sha256(Path(__file__).with_name("qexp_namespace_rollout.py").read_bytes()).hexdigest()
    source_digest = hashlib.sha256()
    for source_path in sorted((repo / "src/qqtools/plugins/qexp").rglob("*.py")):
        source_digest.update(str(source_path.relative_to(repo)).encode() + b"\0" + source_path.read_bytes())
    target_source_sha256 = source_digest.hexdigest()
    target_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    target_changes = subprocess.check_output(["git", "status", "--porcelain", "--", "src"], cwd=repo, text=True)
    results = []
    for index, ref in enumerate(refs):
        revision = subprocess.check_output(
            ["git", "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"], cwd=repo, text=True
        ).strip()
        checkout = output / f"source-{index}"
        checkout.mkdir()
        archive = subprocess.check_output(["git", "archive", revision, "src"], cwd=repo)
        with tarfile.open(fileobj=io.BytesIO(archive)) as source_archive:
            source_archive.extractall(checkout, filter="data")
        for case in cases:
            root = output / f"probe-{index}-{case}"
            root.mkdir()
            environment = dict(os.environ)
            for key in ("TMUX", "TMUX_PANE", "QQTOOLS_TEST_SOURCE_ROOT"):
                environment.pop(key, None)
            for key, relative in {
                "HOME": "home",
                "XDG_CACHE_HOME": "xdg/cache",
                "XDG_CONFIG_HOME": "xdg/config",
                "XDG_DATA_HOME": "xdg/data",
                "TMPDIR": "tmp",
                "TMP": "tmp",
                "TEMP": "tmp",
                "TMUX_TMPDIR": "tmux",
                "QEXP_MACHINE_RUNTIME_ROOT": "machine",
            }.items():
                directory = root / relative
                directory.mkdir(parents=True, exist_ok=True)
                environment[key] = str(directory)
            environment["PYTHONPATH"] = str(checkout / "src")
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--child", case, str(root), str(checkout / "src")],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=90 if case in {"live_namespace_upgrade", "paused_namespace_upgrade"} else 60,
            )
            (root / "stdout.txt").write_text(completed.stdout)
            (root / "stderr.txt").write_text(completed.stderr)
            completed.check_returncode()
            result = json.loads(completed.stdout)
            results.append(
                {
                    "ref": ref,
                    "revision": revision,
                    "probe_sha256": probe_sha256,
                    "rollout_sha256": rollout_sha256,
                    "target_source_sha256": target_source_sha256,
                    "required_capability": CAPABILITY,
                    "target_revision": target_revision,
                    "target_source_changes": target_changes.splitlines(),
                    **result,
                }
            )
            (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            outcome = "Task mutated" if result["task_changed"] else "Task unchanged"
            if case.startswith("worker_"):
                outcome = "Group mutated" if result["group_changed"] else "Group unchanged"
            elif case in {"live_namespace_upgrade", "paused_namespace_upgrade"}:
                outcome = "live released workload survived automatic cutover and offline completion"
            elif result.get("remaining_attempt_removed"):
                outcome = "remaining Attempt truth deleted despite both writer gates"
            elif case == "registration_runner":
                outcome = "runner reached injected process factory; no child or training was created"
            print(f"{ref}: {case}: {outcome}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--begin-writer-capture":
        print(
            json.dumps(
                begin_capture_child(
                    Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]) if len(sys.argv) > 4 else None
                )
            )
        )
    elif len(sys.argv) > 1 and sys.argv[1] == "--child":
        print(json.dumps(probe(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]))))
    elif len(sys.argv) > 1 and sys.argv[1] == "--capture-child":
        print(
            json.dumps(
                capture_child(
                    Path(sys.argv[2]),
                    sys.argv[3],
                    sys.argv[4],
                    Path(sys.argv[5]),
                    Path(sys.argv[6]) if len(sys.argv) > 6 else None,
                )
            )
        )
    elif len(sys.argv) > 1 and sys.argv[1] == "--delayed-runner":
        raise SystemExit(delayed_runner(sys.argv[2:]))
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--ref", action="append", dest="refs")
        parser.add_argument("--case", action="append", choices=CASES, dest="cases")
        args = parser.parse_args()
        run(args.output.resolve(), args.refs or ["v1.3.18"], tuple(args.cases) if args.cases else CASES)
