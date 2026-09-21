"""Automatic retained capture completes durably without surrendering live writers."""

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.helpers import _machine_is_true_idle
from qqtools.plugins.qexp.agent.recovery_capture import RecoveryCapture
from qqtools.plugins.qexp.runtime import responsibility_completion as completion
from qqtools.plugins.qexp.runtime import responsibility_process_capture as processes
from qqtools.plugins.qexp.runtime.paths import local_paths
from qqtools.plugins.qexp.runtime.recovery_admission import fence_recovery_admission
from qqtools.plugins.qexp.runtime.responsibility import responsibility_root
from qqtools.plugins.qexp.runtime.responsibility_capture import (
    CAPTURE_FILE,
    capture_cleanup_guard,
    has_pending_writer_capture,
)
from qqtools.plugins.qexp.runtime.responsibility_cleanup import CLEANUP_FORMAT, CleanupRequest, complete_cleanup
from qqtools.plugins.qexp.runtime.responsibility_store import Conflict, DurableIO, Ledger, Unavailable
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json

pytestmark = pytest.mark.integration


@pytest.fixture
def project(tmp_path, monkeypatch):
    cfg = init_shared_root(tmp_path / "project/.qexp", "worker", runtime_root=tmp_path / "legacy")
    runtime = MachineRuntime(tmp_path / "machine")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    root = runtime.project_paths(binding.project_id)["root"]
    root.mkdir(parents=True, exist_ok=True)
    proc = tmp_path / "proc"
    (proc / "sys/kernel/random").mkdir(parents=True)
    (proc / "sys/kernel/random/boot_id").write_text(Path("/proc/sys/kernel/random/boot_id").read_text())
    (proc / "self/ns").mkdir(parents=True)
    (proc / "self/ns/pid").symlink_to("/proc/self/ns/pid")
    monkeypatch.setattr(processes, "PROC_ROOT", proc)
    with runtime.scheduler_authority() as owns:
        assert owns
        assert runtime.prepare_recovery_registration(binding)
        assert fence_recovery_admission(runtime, binding).is_fenced
        yield runtime, binding, cfg, root, proc


def legacy(project):
    runtime, binding, cfg, _root, _proc = project
    atomic_replace(
        runtime.migration_path(binding.project_id),
        {
            "migration": {
                "state": "active",
                "legacy_runtime_root": str(cfg.runtime_root),
                "project_id": binding.project_id,
                "shared_root": str(cfg.shared_root),
                "machine_name": cfg.machine_name,
            }
        },
    )


def observation(root, identity="task-attempt-1"):
    path = local_paths(root)["observations"] / f"{identity}.json"
    atomic_replace(
        path,
        {
            "exit_observation": {
                "protocol_version": 1,
                "task_id": identity.rsplit("-attempt-", 1)[0],
                "attempt_id": identity,
                "observed_exit_code": 0,
            }
        },
    )
    return path


def advance_and_release(capture):
    """Exercise coverage and source reclamation as separate lifecycle steps."""
    return capture.advance() and capture.release_source()


def finish(capture):
    for _ in range(40):
        settled = advance_and_release(capture)
        if completion.read_capture_completion(capture.root) is not None:
            return settled
    pytest.fail("bounded capture did not finish finite fixture")


def request(identity="task-attempt-1"):
    return CleanupRequest(
        identity,
        {"task_id": "task", "attempt_number": 1},
        {
            "format": CLEANUP_FORMAT,
            "task_id": "task",
            "attempt_id": identity,
            "basis": "terminal_attempt",
        },
    )


def test_completion_retains_source_until_owned_cleanup_finishes(project):
    runtime, binding, cfg, root, _proc = project
    legacy(project)
    path = observation(cfg.runtime_root)
    capture = RecoveryCapture(runtime, binding)
    assert not finish(capture)
    assert not has_pending_writer_capture(root)
    assert has_pending_writer_capture(cfg.runtime_root)
    with capture_cleanup_guard(cfg.runtime_root) as allowed:
        assert not allowed
    with capture_cleanup_guard(root, cfg.runtime_root) as allowed:
        assert allowed
    ledger = Ledger(responsibility_root(root))
    assert ledger.lookup("task-attempt-1")["legacy_source"] == str(cfg.runtime_root)
    assert not complete_cleanup(ledger, root, request())
    assert not path.exists()
    assert complete_cleanup(ledger, root, request())
    assert advance_and_release(capture)
    assert not (cfg.runtime_root / CAPTURE_FILE).exists()
    disabled = runtime.set_enabled(binding.project_id, False)
    assert runtime.remove_binding(binding.project_id) == disabled
    assert not root.exists()
    assert not has_pending_writer_capture(cfg.runtime_root)


def test_empty_completion_restarts_without_process_or_history_enumeration(project, monkeypatch):
    runtime, binding, _cfg, root, _proc = project
    assert finish(RecoveryCapture(runtime, binding))
    proof = (root / completion.COMPLETION_FILE).read_bytes()
    with monkeypatch.context() as guarded:
        guarded.setattr(processes.os, "scandir", lambda *_args: pytest.fail("completed capture enumerated history"))
        assert advance_and_release(RecoveryCapture(runtime, binding))
    assert (root / completion.COMPLETION_FILE).read_bytes() == proof


def test_unrelated_process_does_not_require_runner_identity_stat(project):
    _runtime, _binding, cfg, root, proc = project
    fake = proc / "123456"
    fake.mkdir()
    (fake / "cmdline").write_bytes(b"python\0-c\0pass\0")
    (fake / "stat").write_text("not a process stat")
    ledger = Ledger.open_or_create(responsibility_root(root))
    scanner = processes.RunnerProcessCapture(replace(cfg, runtime_root=root), ledger)

    assert scanner.take().is_sweep_complete


def test_admission_restarts_completed_earlier_census_and_evidence(project):
    from qqtools.plugins.qexp.runtime.responsibility_backfill import ResponsibilityBackfill

    runtime, binding, cfg, root, proc = project
    ledger = Ledger.open_or_create(responsibility_root(root))
    scanner = processes.RunnerProcessCapture(replace(cfg, runtime_root=root), ledger)
    assert scanner.take().is_sweep_complete
    backfill = ResponsibilityBackfill(root, process_capture=scanner)
    assert backfill.take(64, should_cross_lanes=True).is_sweep_complete
    old_revision = read_json(root / CAPTURE_FILE)["progress"]["revision"]
    backfill.close()
    observation(root)
    pid = 123456
    fake = proc / str(pid)
    fake.mkdir()
    (fake / "stat").write_text(f"{pid} (runner) " + " ".join(["S", *(["0"] * 18), "123"]))
    args = ["python", "-m", processes.RUNNER_MODULE]
    fields = {
        "--shared-root": str(cfg.shared_root),
        "--runtime-root": str(root),
        "--machine": cfg.machine_name,
        "--task-id": "late",
        "--attempt-id": "late-attempt-1",
        "--fencing-token": "1",
        "--launch-id": "late-launch",
    }
    args.extend(value for pair in fields.items() for value in pair)
    (fake / "cmdline").write_bytes(("\0".join(args) + "\0").encode())
    assert finish(RecoveryCapture(runtime, binding))
    assert ledger.lookup("task-attempt-1")
    assert ledger.lookup("late-attempt-1")["captured_writers"][0]["pid"] == pid
    assert read_json(root / CAPTURE_FILE)["progress"]["revision"] > old_revision


def test_uncertain_admission_commit_resumes_without_restarting_again(project, monkeypatch):
    from qqtools.plugins.qexp.runtime import responsibility_capture as journal

    runtime, binding, _cfg, root, _proc = project
    publish = journal.atomic_replace

    def uncertain(path, value):
        publish(path, value)
        if "admission" in value:
            raise OSError("admission barrier interrupted")

    monkeypatch.setattr(journal, "atomic_replace", uncertain)
    with pytest.raises(OSError, match="admission barrier"):
        RecoveryCapture(runtime, binding)._begin()
    initial = read_json(root / CAPTURE_FILE)
    monkeypatch.setattr(journal, "atomic_replace", publish)
    capture = RecoveryCapture(runtime, binding)
    capture._begin()
    assert read_json(root / CAPTURE_FILE) == initial
    assert finish(capture)


def test_admission_replays_prior_pending_evidence_before_fresh_sweep(project, monkeypatch):
    from qqtools.plugins.qexp.runtime import responsibility_backfill as backfill

    runtime, binding, cfg, root, _proc = project
    ledger = Ledger.open_or_create(responsibility_root(root))
    scanner = processes.RunnerProcessCapture(replace(cfg, runtime_root=root), ledger)
    assert scanner.take().is_sweep_complete
    observation(root)
    old = backfill.ResponsibilityBackfill(root, process_capture=scanner)
    original = backfill.capture_local_record

    def interrupt(*args, **kwargs):
        raise OSError("capture interrupted with durable pending record")

    monkeypatch.setattr(backfill, "capture_local_record", interrupt)
    with pytest.raises(OSError, match="durable pending"):
        old.take(64, should_cross_lanes=True)
    pending = read_json(old.path)
    assert pending["pending"] == ["task-attempt-1.json"]
    old.close()
    observation(root, "late-attempt-1")
    observed = []

    def record(*args, **kwargs):
        observed.append(str(args[3]))
        return original(*args, **kwargs)

    monkeypatch.setattr(backfill, "capture_local_record", record)
    assert finish(RecoveryCapture(runtime, binding))
    assert observed[0] == "task-attempt-1.json"
    assert ledger.lookup("task-attempt-1") and ledger.lookup("late-attempt-1")
    assert read_json(old.path)["writer_sweep_revision"] > pending["writer_sweep_revision"]


def test_completion_requires_matching_admission_even_with_updated_digest(project):
    runtime, binding, _cfg, root, _proc = project
    assert finish(RecoveryCapture(runtime, binding))
    capture = read_json(root / CAPTURE_FILE)
    capture["admission"]["registration_generation"] = "another-generation"
    atomic_replace(root / CAPTURE_FILE, capture)
    proof = read_json(root / completion.COMPLETION_FILE)
    proof["capture_digest"] = completion._digest(capture)
    atomic_replace(root / completion.COMPLETION_FILE, proof)
    with pytest.raises(Unavailable, match="invalid retained capture"):
        completion.read_capture_completion(root)
    assert has_pending_writer_capture(root)


def test_completion_cannot_cover_a_later_registration_generation(project):
    from qqtools.plugins.qexp.agent.recovery_capture import inspect_recovery_capture

    runtime, binding, _cfg, root, _proc = project
    assert finish(RecoveryCapture(runtime, binding))
    old = (root / completion.COMPLETION_FILE).read_bytes()
    adopted = replace(binding, registration_generation="later-generation")
    with pytest.raises(Conflict, match="binding changed"):
        advance_and_release(RecoveryCapture(runtime, adopted))
    assert inspect_recovery_capture(runtime, adopted)["state"] == "unavailable"
    assert (root / completion.COMPLETION_FILE).read_bytes() == old


def readopt(project):
    """Exercise real registration A -> B -> A after explicit eligibility expiry."""
    from qqtools.plugins.qexp.layout import load_machine_registration, save_machine_registration

    runtime, _binding, cfg, _root, _proc = project

    def expire():
        value = load_machine_registration(cfg)
        value["registration"]["eligibility_expires_at"] = "2000-01-01T00:00:00Z"
        save_machine_registration(cfg, value)

    expire()
    other = MachineRuntime(runtime.root.parent / "other-owner")
    other.ensure_binding(cfg.shared_root, cfg.machine_name, adopt_existing=True)
    expire()
    adopted = runtime.ensure_binding(cfg.shared_root, cfg.machine_name, adopt_existing=True)[0]
    assert runtime.prepare_recovery_registration(adopted)
    assert fence_recovery_admission(runtime, adopted).is_fenced
    return adopted


@pytest.mark.parametrize("has_source", [False, True])
def test_readoption_recaptures_after_completed_generation(project, has_source):
    runtime, binding, cfg, root, _proc = project
    if has_source:
        legacy(project)
    assert finish(RecoveryCapture(runtime, binding))
    old = completion.read_capture_completion(root)
    revision = read_json(root / CAPTURE_FILE)["progress"]["revision"]
    adopted = readopt(project)
    assert adopted.registration_generation != binding.registration_generation
    observation(cfg.runtime_root if has_source else root)
    capture = RecoveryCapture(runtime, adopted)
    assert finish(capture) is (not has_source)
    proof = completion.read_capture_completion(root)
    assert proof["registration_generation"] == adopted.registration_generation
    assert proof != old
    assert read_json(root / CAPTURE_FILE)["progress"]["revision"] > revision
    assert Ledger(responsibility_root(root)).lookup("task-attempt-1")
    if has_source:
        assert has_pending_writer_capture(cfg.runtime_root)
        assert not (root / completion.SOURCE_RELEASE_FILE).exists()


def test_readoption_resets_unfinished_capture_and_rejects_old_iterator(project):
    runtime, binding, _cfg, root, proc = project
    for number in range(130):
        (proc / f"unrelated-{number}").touch()
    old = RecoveryCapture(runtime, binding)
    assert not advance_and_release(old)
    revision = read_json(root / CAPTURE_FILE)["progress"]["revision"]
    adopted = readopt(project)
    current = RecoveryCapture(runtime, adopted)
    current._begin()
    assert read_json(root / CAPTURE_FILE)["progress"]["revision"] == revision + 1
    with pytest.raises(Unavailable, match="admission changed"):
        old.processes.take()
    observation(root)
    assert finish(current)
    assert Ledger(responsibility_root(root)).lookup("task-attempt-1")


def test_readoption_preserves_unreleased_source_and_existing_members(project):
    runtime, binding, cfg, root, _proc = project
    legacy(project)
    observation(cfg.runtime_root)
    assert not finish(RecoveryCapture(runtime, binding))
    original = Ledger(responsibility_root(root)).lookup("task-attempt-1")
    adopted = readopt(project)
    observation(cfg.runtime_root, "late-attempt-1")
    assert not finish(RecoveryCapture(runtime, adopted))
    ledger = Ledger(responsibility_root(root))
    assert ledger.lookup("task-attempt-1")["legacy_source"] == original["legacy_source"]
    assert ledger.lookup("late-attempt-1")
    assert has_pending_writer_capture(cfg.runtime_root)


def test_readoption_does_not_recreate_lost_unreleased_retention(project):
    runtime, binding, cfg, root, _proc = project
    legacy(project)
    observation(cfg.runtime_root)
    assert not finish(RecoveryCapture(runtime, binding))
    old = (root / completion.COMPLETION_FILE).read_bytes()
    (cfg.runtime_root / CAPTURE_FILE).unlink()
    adopted = readopt(project)
    with pytest.raises(Unavailable, match="lost unreleased"):
        advance_and_release(RecoveryCapture(runtime, adopted))
    assert (root / completion.COMPLETION_FILE).read_bytes() == old
    assert not (cfg.runtime_root / CAPTURE_FILE).exists()


@pytest.mark.parametrize("boundary", ["source", "intent"])
def test_another_readoption_resumes_interrupted_generation_transition(project, monkeypatch, boundary):
    from qqtools.plugins.qexp.runtime import responsibility_generation as generation
    from qqtools.plugins.qexp.runtime.responsibility_capture import GENERATION_FILE

    runtime, binding, cfg, root, _proc = project
    legacy(project)
    assert finish(RecoveryCapture(runtime, binding))
    adopted = readopt(project)
    publish = generation.atomic_replace

    def interrupted(path, value):
        publish(path, value)
        if (boundary == "intent" and path.name == GENERATION_FILE) or (
            boundary == "source" and path.parent == cfg.runtime_root
        ):
            raise OSError("interrupted generation intent")

    monkeypatch.setattr(generation, "atomic_replace", interrupted)
    with pytest.raises(OSError, match="generation intent"):
        advance_and_release(RecoveryCapture(runtime, adopted))
    later = readopt(project)
    monkeypatch.setattr(generation, "atomic_replace", publish)
    observation(cfg.runtime_root)
    assert not finish(RecoveryCapture(runtime, later))
    assert completion.read_capture_completion(root)["registration_generation"] == later.registration_generation
    assert Ledger(responsibility_root(root)).lookup("task-attempt-1")


@pytest.mark.parametrize("damage", ["journal", "checkpoint", "ledger", "owner", "source_hold"])
def test_damaged_generation_transition_stays_unavailable(project, monkeypatch, damage):
    from qqtools.plugins.qexp.runtime import responsibility_generation as generation
    from qqtools.plugins.qexp.runtime.responsibility_capture import GENERATION_FILE

    runtime, binding, cfg, root, _proc = project
    legacy(project)
    assert finish(RecoveryCapture(runtime, binding))
    adopted = readopt(project)
    publish = generation.atomic_replace

    def interrupted(path, value):
        publish(path, value)
        if path.name == GENERATION_FILE:
            raise OSError("interrupted generation intent")

    monkeypatch.setattr(generation, "atomic_replace", interrupted)
    with pytest.raises(OSError, match="generation intent"):
        advance_and_release(RecoveryCapture(runtime, adopted))
    monkeypatch.setattr(generation, "atomic_replace", publish)
    if damage == "journal":
        (root / GENERATION_FILE).write_text("broken")
    elif damage == "checkpoint":
        (root / CAPTURE_FILE).write_text("broken")
    elif damage == "ledger":
        (responsibility_root(root) / "marker").write_text("broken")
    elif damage == "owner":
        journal = read_json(root / GENERATION_FILE)
        journal["owner"]["owner_instance"] = "different-owner"
        atomic_replace(root / GENERATION_FILE, journal)
    else:
        atomic_replace(cfg.runtime_root / CAPTURE_FILE, {"another": "capture"})
    proof = (root / completion.COMPLETION_FILE).read_bytes()
    with pytest.raises((Unavailable, ValueError)):
        advance_and_release(RecoveryCapture(runtime, adopted))
    assert (root / completion.COMPLETION_FILE).read_bytes() == proof
    with capture_cleanup_guard(root, cfg.runtime_root, should_remove_roots=True) as can_remove:
        assert not can_remove


def test_source_release_rechecks_completion_after_generation_transition(project, monkeypatch):
    runtime, binding, cfg, root, _proc = project
    legacy(project)
    assert finish(RecoveryCapture(runtime, binding))
    adopted = readopt(project)
    capture = RecoveryCapture(runtime, adopted)
    read = completion.read_capture_completion
    has_restarted = False

    def race(path, **kwargs):
        nonlocal has_restarted
        proof = read(path, **kwargs)
        if not has_restarted:
            has_restarted = True
            capture._restart_generation()
        return proof

    monkeypatch.setattr(completion, "read_capture_completion", race)
    with pytest.raises(Unavailable, match="changed before source release"):
        completion.release_completed_source(root)
    assert not (root / completion.SOURCE_RELEASE_FILE).exists()
    assert has_pending_writer_capture(cfg.runtime_root)


@pytest.mark.parametrize(
    "boundary",
    ["intent", "source", "completion_delete", "release_delete", "checkpoint", "source_ready", "intent_delete"],
)
def test_generation_transition_child_crash_replays_exact_revision(project, boundary):
    from qqtools.plugins.qexp.runtime import responsibility_generation as generation
    from qqtools.plugins.qexp.runtime.responsibility_capture import GENERATION_FILE

    runtime, binding, cfg, root, _proc = project
    legacy(project)
    assert finish(RecoveryCapture(runtime, binding))
    before = read_json(root / CAPTURE_FILE)
    adopted = readopt(project)
    capture = RecoveryCapture(runtime, adopted)
    pid = os.fork()
    if pid == 0:
        publish, delete = generation.atomic_replace, generation.DurableIO.delete

        def crash_publish(path, value):
            publish(path, value)
            kind = (
                "intent"
                if path.name == GENERATION_FILE
                else ("source" if value["phase"] == "generation_pending" else "source_ready")
                if path.parent == cfg.runtime_root
                else "checkpoint"
            )
            if kind == boundary:
                os._exit(47)

        def crash_delete(io, path, **kwargs):
            delete(io, path, **kwargs)
            kind = {
                completion.COMPLETION_FILE: "completion_delete",
                completion.SOURCE_RELEASE_FILE: "release_delete",
                GENERATION_FILE: "intent_delete",
            }[path.name]
            if kind == boundary:
                os._exit(47)

        generation.atomic_replace, generation.DurableIO.delete = crash_publish, crash_delete
        try:
            # Exercise the durable primitive in a real process; the parent owns
            # the isolated scheduler fence and no competing lifecycle runs here.
            generation.restart_capture_generation(root, legacy_source=cfg.runtime_root, owner=capture._owner())
        finally:
            os._exit(48)
    assert os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]) == 47
    assert has_pending_writer_capture(cfg.runtime_root)
    with capture_cleanup_guard(cfg.runtime_root, should_remove_roots=True) as can_remove:
        assert not can_remove, "standalone cleanup must not remove the source during a crashed transition"
    if boundary == "source":
        assert not (root / GENERATION_FILE).exists()
        assert completion.read_capture_completion(root) is not None
        with pytest.raises(Unavailable, match="does not match completed capture"):
            completion.release_completed_source(root)
        assert has_pending_writer_capture(cfg.runtime_root)
        with capture_cleanup_guard(cfg.runtime_root, should_remove_roots=True) as can_remove:
            assert not can_remove, "old completion must not release the generation hold"
    else:
        assert has_pending_writer_capture(root)
    observation(cfg.runtime_root)
    if (root / GENERATION_FILE).exists():
        with pytest.raises(Unavailable, match="generation transition"):
            completion.read_capture_completion(root)
    if boundary != "intent_delete":
        capture._restart_generation()
    after = read_json(root / CAPTURE_FILE)
    assert after["progress"]["revision"] == before["progress"]["revision"] + 1
    assert after["admission"]["registration_generation"] == adopted.registration_generation
    assert not finish(capture)
    assert Ledger(responsibility_root(root)).lookup("task-attempt-1")
    assert completion.read_capture_completion(root)["registration_generation"] == adopted.registration_generation


@pytest.mark.parametrize("damage", ["certificate", "capture", "evidence", "ledger", "checkpoint_missing"])
def test_invalid_completion_preserves_cleanup_barrier(project, damage):
    runtime, binding, _cfg, root, _proc = project
    assert finish(RecoveryCapture(runtime, binding))
    paths = {
        "certificate": root / completion.COMPLETION_FILE,
        "capture": root / CAPTURE_FILE,
        "evidence": root / "responsibility-capture-backfill.json",
        "ledger": responsibility_root(root) / "marker",
    }
    if damage == "checkpoint_missing":
        (root / CAPTURE_FILE).unlink()
    else:
        paths[damage].write_text("broken")
    assert has_pending_writer_capture(root)
    with capture_cleanup_guard(root) as allowed:
        assert not allowed
    with pytest.raises((OSError, ValueError, RuntimeError)):
        advance_and_release(RecoveryCapture(runtime, binding))


def test_visible_completion_retries_durability_without_recapture(project, monkeypatch):
    runtime, binding, _cfg, root, _proc = project
    replace_json = completion.atomic_replace

    def uncertain(path, value):
        replace_json(path, value)
        raise OSError("uncertain commit")

    monkeypatch.setattr(completion, "atomic_replace", uncertain)
    with pytest.raises(OSError, match="uncertain commit"):
        finish(RecoveryCapture(runtime, binding))
    sync = DurableIO.sync_directory

    def fail_barrier(self, path, label):
        if label == "capture_completion":
            raise OSError("completion barrier unavailable")
        return sync(self, path, label)

    monkeypatch.setattr(DurableIO, "sync_directory", fail_barrier)
    assert has_pending_writer_capture(root)
    monkeypatch.setattr(DurableIO, "sync_directory", sync)
    with monkeypatch.context() as guarded:
        guarded.setattr(processes.os, "scandir", lambda *_args: pytest.fail("completion retry recaptured"))
        assert advance_and_release(RecoveryCapture(runtime, binding))


def test_completion_survives_reboot_but_rejects_foreign_host(project, monkeypatch):
    runtime, binding, _cfg, _root, proc = project
    assert finish(RecoveryCapture(runtime, binding))
    (proc / "sys/kernel/random/boot_id").write_text("10000000-0000-0000-0000-000000000001")
    assert advance_and_release(RecoveryCapture(runtime, binding))
    monkeypatch.setattr(processes, "host_instance_id", lambda: "foreign")
    with pytest.raises(Unavailable, match="host"):
        advance_and_release(RecoveryCapture(runtime, binding))


def test_malformed_migration_is_not_treated_as_absent(project):
    runtime, binding, _cfg, root, _proc = project
    atomic_replace(runtime.migration_path(binding.project_id), {"migration": {"state": "active"}})
    with pytest.raises(Unavailable, match="migration"):
        advance_and_release(RecoveryCapture(runtime, binding))
    assert not (root / CAPTURE_FILE).exists()


def test_capture_slices_release_machine_lifecycle_locks(project, monkeypatch):
    runtime, binding, _cfg, _root, proc = project
    for number in range(130):
        (proc / f"other-{number}").touch()
    capture = RecoveryCapture(runtime, binding)
    take = processes.RunnerProcessCapture.take

    def check_locks(scanner, limit):
        with runtime.migration_guard(blocking=False) as migrated, runtime.registry_guard(blocking=False) as registered:
            assert migrated and registered
        return take(scanner, limit)

    monkeypatch.setattr(processes.RunnerProcessCapture, "take", check_locks)
    assert not advance_and_release(capture)
    assert not advance_and_release(capture)
    changed = runtime.set_enabled(binding.project_id, False)
    with pytest.raises(Conflict, match="binding changed"):
        finish(capture)
    assert finish(RecoveryCapture(runtime, changed))


def test_captured_live_writer_blocks_cleanup_until_real_process_exits(project):
    runtime, binding, cfg, root, proc = project
    legacy(project)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        fake = proc / str(child.pid)
        fake.mkdir()
        (fake / "stat").write_bytes((Path("/proc") / str(child.pid) / "stat").read_bytes())
        arguments = [
            "python",
            "-m",
            processes.RUNNER_MODULE,
            "--shared-root",
            str(cfg.shared_root),
            "--runtime-root",
            str(cfg.runtime_root),
            "--machine",
            cfg.machine_name,
            "--task-id",
            "task",
            "--attempt-id",
            "task-attempt-1",
            "--fencing-token",
            "1",
            "--launch-id",
            "launch",
        ]
        (fake / "cmdline").write_bytes(("\0".join(arguments) + "\0").encode())
        capture = RecoveryCapture(runtime, binding)
        assert not finish(capture)
        ledger = Ledger(responsibility_root(root))
        assert ledger.lookup("task-attempt-1")["captured_writers"][0]["pid"] == child.pid
        assert not complete_cleanup(ledger, root, request())
        assert not completion.release_completed_source(root)
        path = observation(cfg.runtime_root)
        child.terminate()
        child.wait(timeout=5)
        assert not complete_cleanup(ledger, root, request())
        assert not path.exists()
        assert complete_cleanup(ledger, root, request())
        assert advance_and_release(capture)
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=5)


def test_process_crash_after_completion_rename_recovers(tmp_path):
    cfg = init_shared_root(tmp_path / "project/.qexp", "worker", runtime_root=tmp_path / "legacy")
    runtime = MachineRuntime(tmp_path / "machine")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    root = runtime.project_paths(binding.project_id)["root"]
    pid = os.fork()
    if pid == 0:
        publish = completion.atomic_replace

        def crash(path, value):
            publish(path, value)
            os._exit(47)

        completion.atomic_replace = crash
        child_runtime = MachineRuntime(runtime.root)
        try:
            with child_runtime.scheduler_authority() as owns:
                assert owns
                assert child_runtime.prepare_recovery_registration(binding)
                assert fence_recovery_admission(child_runtime, binding).is_fenced
                capture = RecoveryCapture(child_runtime, binding)
                for _ in range(100):
                    advance_and_release(capture)
        finally:
            os._exit(48)
    assert os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]) == 47
    assert completion.read_capture_completion(root) is not None
    with runtime.scheduler_authority() as owns:
        assert owns
        assert advance_and_release(RecoveryCapture(runtime, binding))


def test_each_evidence_slice_keeps_its_record_budget_across_lanes(project, monkeypatch):
    from qqtools.plugins.qexp.runtime import responsibility_backfill as backfill

    runtime, binding, _cfg, root, _proc = project
    for number in range(130):
        observation(root, f"task-{number}-attempt-1")
    original = backfill.capture_local_record
    captures = []

    def observed(*args, **kwargs):
        captures.append(args[3])
        return original(*args, **kwargs)

    monkeypatch.setattr(backfill, "capture_local_record", observed)
    capture = RecoveryCapture(runtime, binding)
    for _ in range(10):
        previous = len(captures)
        is_settled = advance_and_release(capture)
        assert len(captures) - previous <= 64
        if is_settled:
            break
    else:
        pytest.fail("finite evidence capture did not finish")
    assert len(captures) == 130
    ledger = Ledger(responsibility_root(root))
    assert all(ledger.lookup(f"task-{number}-attempt-1") for number in range(130))


def test_retained_capture_displaces_only_the_advisory_backfill(project, monkeypatch):
    from qqtools.plugins.qexp.runtime.responsibility import ResponsibilityReader
    from qqtools.plugins.qexp.runtime.responsibility_backfill import ResponsibilityBackfill

    runtime, binding, _cfg, root, _proc = project
    capture = RecoveryCapture(runtime, binding)
    capture._begin()
    take = ResponsibilityBackfill.take

    def guarded(scanner, *args, **kwargs):
        assert scanner.process_capture is not None, "advisory history scan competed with retained capture"
        return take(scanner, *args, **kwargs)

    monkeypatch.setattr(ResponsibilityBackfill, "take", guarded)
    reader = ResponsibilityReader(root)
    try:
        assert reader.take() is None
        assert reader._done.wait(5)
        assert reader._error is None
        assert finish(capture)
    finally:
        reader.close()
        capture.close()


def test_capture_status_never_finishes_an_uncertain_barrier(project, monkeypatch):
    from qqtools.plugins.qexp.agent.recovery_capture import inspect_recovery_capture

    runtime, binding, _cfg, root, _proc = project
    assert inspect_recovery_capture(runtime, binding)["state"] == "not_started"
    capture = RecoveryCapture(runtime, binding)
    capture._begin()
    assert inspect_recovery_capture(runtime, binding)["state"] == "capturing_processes"
    assert finish(capture)
    monkeypatch.setattr(DurableIO, "sync_directory", lambda *_args: pytest.fail("status attempted a durability repair"))
    assert inspect_recovery_capture(runtime, binding) == {"state": "captured", "diagnostic_only": True}
    (root / completion.COMPLETION_FILE).write_text("broken")
    assert inspect_recovery_capture(runtime, binding)["state"] == "unavailable"


def test_source_release_retries_after_uncertain_unlink(project, monkeypatch):
    runtime, binding, cfg, root, _proc = project
    legacy(project)
    delete = DurableIO.delete

    def uncertain(self, path, **kwargs):
        delete(self, path, **kwargs)
        if path == cfg.runtime_root / CAPTURE_FILE:
            raise OSError("source unlink uncertainty")

    monkeypatch.setattr(DurableIO, "delete", uncertain)
    with pytest.raises(OSError, match="unlink uncertainty"):
        finish(RecoveryCapture(runtime, binding))
    assert completion.read_capture_completion(root) is not None
    assert not (cfg.runtime_root / CAPTURE_FILE).exists()
    barriers = []
    sync = DurableIO.sync_directory

    def observe(self, directory, label):
        if label == "capture_source_release":
            barriers.append(directory)
        return sync(self, directory, label)

    monkeypatch.setattr(DurableIO, "sync_directory", observe)
    assert advance_and_release(RecoveryCapture(runtime, binding))
    assert barriers == [cfg.runtime_root]


def test_released_source_stays_released_when_current_launches_publish(project):
    runtime, binding, cfg, root, _proc = project
    legacy(project)
    assert finish(RecoveryCapture(runtime, binding))
    assert not (cfg.runtime_root / CAPTURE_FILE).exists()
    ledger = Ledger(responsibility_root(root))
    ledger.publish("new-attempt-1", {"task_id": "new", "attempt_number": 1})
    assert advance_and_release(RecoveryCapture(runtime, binding))
    assert ledger.lookup("new-attempt-1")["stage"] == "active"


def test_missing_source_hold_cannot_settle_unresolved_capture(project):
    runtime, binding, cfg, root, _proc = project
    legacy(project)
    observation(cfg.runtime_root)
    assert not finish(RecoveryCapture(runtime, binding))
    (cfg.runtime_root / CAPTURE_FILE).unlink()
    with pytest.raises(Unavailable, match="disappeared"):
        advance_and_release(RecoveryCapture(runtime, binding))
    assert not (root / completion.SOURCE_RELEASE_FILE).exists()


def test_binding_removal_finishes_empty_source_release_without_another_agent_pass(project):
    runtime, binding, cfg, root, _proc = project
    legacy(project)
    observation(cfg.runtime_root)
    assert not finish(RecoveryCapture(runtime, binding))
    ledger = Ledger(responsibility_root(root))
    assert not complete_cleanup(ledger, root, request())
    assert complete_cleanup(ledger, root, request())
    assert (cfg.runtime_root / CAPTURE_FILE).exists()
    disabled = runtime.set_enabled(binding.project_id, False)
    assert runtime.remove_binding(binding.project_id) == disabled
    assert not (cfg.runtime_root / CAPTURE_FILE).exists()
    assert not root.exists()


@pytest.mark.parametrize("has_source", [False, True])
def test_qualified_idle_and_inbox_do_not_scan_attempt_history_or_sync(project, monkeypatch, has_source):
    from qqtools.plugins.qexp.agent import context

    runtime, binding, cfg, root, _proc = project
    if has_source:
        legacy(project)
    assert finish(RecoveryCapture(runtime, binding))
    runtime.last_cycle_had_demand = False
    original = context.iter_evidence_files
    allowed = {local_paths(root)["events"]}
    if has_source:
        allowed.update(
            local_paths(cfg.runtime_root)[name]
            for name in ("events", "active", "provisional", "cpu_active", "cpu_provisional")
        )

    def guarded(directory, **kwargs):
        assert directory in allowed, f"qualified routine check enumerated {directory}"
        return original(directory, **kwargs)

    def no_sync(*args, **kwargs):
        pytest.fail("healthy idle/inbox check synchronized unchanged capture")

    with monkeypatch.context() as patch:
        patch.setattr(context, "iter_evidence_files", guarded)
        patch.setattr(DurableIO, "sync_directory", no_sync)
        for _ in range(3):
            assert _machine_is_true_idle(runtime, has_consumed_binding=True)
            runtime.drain_legacy_runner_evidence(binding)


@pytest.mark.parametrize("stage", ["active", "maintenance"])
def test_qualified_idle_retains_each_membership_stage(project, stage):
    runtime, binding, _cfg, root, _proc = project
    assert finish(RecoveryCapture(runtime, binding))
    runtime.last_cycle_had_demand = False
    ledger = Ledger(responsibility_root(root))
    generation = ledger.publish("task-attempt-1", {"task_id": "task", "attempt_number": 1})
    if stage == "maintenance":
        ledger.handoff("task-attempt-1", generation)
    assert not _machine_is_true_idle(runtime, has_consumed_binding=True)
    assert "recovery_responsibilities" in runtime.iter_recovery_blockers(binding, should_use_capture=True)


@pytest.mark.parametrize("lane", ["active", "provisional", "cpu_active", "cpu_provisional"])
def test_qualified_idle_preserves_legacy_reservation_blockers(project, lane):
    runtime, binding, cfg, _root, _proc = project
    legacy(project)
    assert finish(RecoveryCapture(runtime, binding))
    runtime.last_cycle_had_demand = False
    atomic_replace(local_paths(cfg.runtime_root)[lane] / "reservation.json", {"reservation": {}})
    assert not _machine_is_true_idle(runtime, has_consumed_binding=True)
    assert f"legacy:{lane}:reservation" in runtime.iter_recovery_blockers(binding, should_use_capture=True)


@pytest.mark.parametrize("damage", ["certificate", "source", "migration", "ledger", "capability", "generation"])
def test_qualified_idle_and_import_fail_closed_on_invalid_capture(project, damage):
    runtime, binding, cfg, root, _proc = project
    legacy(project)
    assert finish(RecoveryCapture(runtime, binding))
    runtime.last_cycle_had_demand = False
    if damage == "certificate":
        (root / completion.COMPLETION_FILE).write_text("broken")
    elif damage in {"source", "migration"}:
        path = runtime.migration_path(binding.project_id)
        value = read_json(path)
        if damage == "source":
            value["migration"]["legacy_runtime_root"] = str(cfg.runtime_root.parent / "different")
        else:
            value["migration"]["state"] = "incomplete"
        atomic_replace(path, value)
    elif damage == "ledger":
        (responsibility_root(root) / "marker").write_text("broken")
    elif damage == "capability":
        path = cfg.shared_root / "schema/version.json"
        value = read_json(path)
        value["schema"]["required_capabilities"].remove("local-recovery-v1")
        atomic_replace(path, value)
    else:
        binding = replace(binding, registration_generation="other-generation")
    assert list(runtime.iter_recovery_blockers(binding, should_use_capture=True)) == ["recovery_capture_unavailable"]
    with pytest.raises((OSError, RuntimeError, ValueError)):
        runtime.drain_legacy_runner_evidence(binding)


def test_qualified_inbox_preserves_late_event_transport_and_barrier_retry(project, monkeypatch):
    runtime, binding, cfg, root, _proc = project
    legacy(project)
    assert finish(RecoveryCapture(runtime, binding))
    runtime.last_cycle_had_demand = False
    source = local_paths(cfg.runtime_root)["events"] / "machine/late.json"
    target = local_paths(root)["events"] / "machine/late.json"
    event = {"event_id": "late", "event_type": "diagnostic"}
    atomic_replace(source, event)
    assert not _machine_is_true_idle(runtime, has_consumed_binding=True)
    original = DurableIO.sync_directory

    def fail_destination(io, directory, label):
        if label == "legacy_event_destination":
            raise OSError("destination barrier interrupted")
        return original(io, directory, label)

    with monkeypatch.context() as patch:
        patch.setattr(DurableIO, "sync_directory", fail_destination)
        with pytest.raises(OSError, match="destination barrier interrupted"):
            runtime.drain_legacy_runner_evidence(binding)
    assert read_json(source) == read_json(target) == event
    runtime.drain_legacy_runner_evidence(binding)
    assert not source.exists() and read_json(target) == event
    assert not _machine_is_true_idle(runtime, has_consumed_binding=True)


def test_qualified_inbox_uses_direct_capture_membership_for_late_runner_exit(project, monkeypatch):
    from qqtools.plugins.qexp.agent import context
    from qqtools.plugins.qexp.agent.recovery_capture import recovery_owner
    from qqtools.plugins.qexp.runtime.responsibility import ResponsibilityReader

    runtime, binding, cfg, root, _proc = project
    legacy(project)
    identity = "task-attempt-1"
    atomic_replace(
        local_paths(cfg.runtime_root)["registrations"] / f"{identity}.json",
        {"process_registration": {"protocol_version": 1, "task_id": "task", "attempt_id": identity}},
    )
    assert not finish(RecoveryCapture(runtime, binding))
    source = observation(cfg.runtime_root)
    target = local_paths(root)["observations"] / source.name
    reader = ResponsibilityReader(root, owner=recovery_owner(runtime, binding))
    original = context.iter_evidence_files

    def guarded(directory, **kwargs):
        assert directory == local_paths(cfg.runtime_root)["events"]
        return original(directory, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(context, "iter_evidence_files", guarded)
            runtime.drain_legacy_runner_evidence(binding)
        for _ in range(40):
            if reader.take() is not None:
                reader.acknowledge()
            if reader._done is not None:
                assert reader._done.wait(5)
            if target.exists():
                break
        assert read_json(target) == read_json(source)
        assert source.exists()
    finally:
        reader.close()
        if reader._done is not None:
            assert reader._done.wait(5)


@pytest.mark.parametrize("damage", ["missing", "malformed", "different_completion"])
def test_qualified_idle_requires_matching_release_receipt_after_source_hold_removal(project, damage):
    runtime, binding, cfg, root, _proc = project
    legacy(project)
    assert finish(RecoveryCapture(runtime, binding))
    runtime.last_cycle_had_demand = False
    assert not (cfg.runtime_root / CAPTURE_FILE).exists()
    receipt = root / completion.SOURCE_RELEASE_FILE
    if damage == "missing":
        receipt.unlink()
    elif damage == "malformed":
        receipt.write_text("broken")
    else:
        value = read_json(receipt)
        value["completion_digest"] = "different"
        atomic_replace(receipt, value)
    assert not _machine_is_true_idle(runtime, has_consumed_binding=True)
    assert "legacy:writer_capture_unavailable" in runtime.iter_recovery_blockers(binding, should_use_capture=True)


def test_group_isolation_activates_while_captured_source_remains_owned(project):
    from qqtools.plugins.qexp.agent.recovery_enrollment import RecoveryEnrollment
    from qqtools.plugins.qexp.runtime.group_namespace import is_group_authority_isolated

    runtime, binding, cfg, root, _proc = project
    legacy(project)
    observation(cfg.runtime_root)
    capture = RecoveryCapture(runtime, binding)
    assert not finish(capture)
    assert completion.read_capture_completion(root) is not None
    enrollment = RecoveryEnrollment(runtime)
    enrollment._captures[binding] = capture
    try:
        from tests.helpers.qexp.lifecycle import wait_until

        enrollment.poll()
        wait_until(
            "group-authority-isolated",
            lambda: is_group_authority_isolated(cfg.shared_root),
            stage="recovery-enrollment:source-retained",
        )
        enrollment.poll()
        assert is_group_authority_isolated(cfg.shared_root)
        assert runtime.recovery_enrollment_pending_projects == {binding.project_id}
        assert has_pending_writer_capture(cfg.runtime_root)
        assert Ledger(responsibility_root(root)).has_members()
    finally:
        enrollment.stop()


def test_capture_completion_does_not_release_source_without_explicit_cleanup(project):
    runtime, binding, cfg, root, _proc = project
    legacy(project)
    capture = RecoveryCapture(runtime, binding)
    for _ in range(40):
        if capture.advance():
            break
    else:
        pytest.fail("bounded capture did not finish")
    assert completion.read_capture_completion(root) is not None
    assert has_pending_writer_capture(cfg.runtime_root)
    assert capture.activate_group_authority()
    assert has_pending_writer_capture(cfg.runtime_root)
    assert capture.release_source()
    assert not has_pending_writer_capture(cfg.runtime_root)
