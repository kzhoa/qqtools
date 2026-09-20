"""Bounded process observation, source retention and restart replay."""

import multiprocessing
import os
import shutil
from pathlib import Path

import pytest

from qqtools.plugins.qexp.config_types import RootConfig
from qqtools.plugins.qexp.runtime import responsibility_capture as journal
from qqtools.plugins.qexp.runtime import responsibility_process_capture as capture
from qqtools.plugins.qexp.runtime.responsibility import responsibility_root
from qqtools.plugins.qexp.runtime.responsibility_store import Conflict, Ledger, Unavailable
from qqtools.plugins.qexp.runtime.store import read_json

pytestmark = pytest.mark.integration


def fixture(tmp_path, monkeypatch):
    root = tmp_path / "target/runtime"
    root.mkdir(parents=True)
    source = tmp_path / "legacy/runtime"
    source.mkdir(parents=True)
    proc = tmp_path / "proc"
    (proc / "sys/kernel/random").mkdir(parents=True)
    (proc / "sys/kernel/random/boot_id").write_text("10000000-0000-0000-0000-000000000001")
    (proc / "self/ns").mkdir(parents=True)
    (proc / "self/ns/pid").touch()
    monkeypatch.setattr(capture, "PROC_ROOT", proc)
    cfg = RootConfig(tmp_path / "project/.qexp", tmp_path / "project", "fixture", root)
    ledger = Ledger.open_or_create(responsibility_root(root))
    return cfg, ledger, source, proc


def process(proc, cfg, pid, *, runtime_root=None, task_id="task", suffix="1", command=None):
    root = proc / str(pid)
    root.mkdir()
    # Fields after comm start at Linux stat field3; starttime is field22.
    (root / "stat").write_text(f"{pid} (fixture ) name) " + " ".join(["S", *(["0"] * 18), str(pid * 10)]))
    arguments = [
        "python",
        "-m",
        capture.RUNNER_MODULE,
        "--shared-root",
        str(cfg.shared_root),
        "--runtime-root",
        str(runtime_root or cfg.runtime_root),
        "--machine",
        cfg.machine_name,
        "--task-id",
        task_id,
        "--attempt-id",
        f"{task_id}-attempt-{suffix}",
        "--fencing-token",
        "1",
        "--launch-id",
        "fixture-launch",
    ]
    (root / "cmdline").write_bytes(command if command is not None else ("\0".join(arguments) + "\0").encode())
    return root


def finish(scanner, *, limit=7):
    pages = []
    try:
        for _ in range(1000):
            progress = scanner.take(limit)
            pages.append(progress)
            assert progress.entries_visited <= limit and progress.writers_recorded <= limit
            if progress.is_sweep_complete:
                return pages
        pytest.fail("finite process fixture did not finish")
    finally:
        scanner.close()


@pytest.mark.parametrize("limit", [1, 7, 63, 64])
def test_process_pages_capture_current_and_legacy_without_history(tmp_path, monkeypatch, limit):
    cfg, ledger, source, proc = fixture(tmp_path, monkeypatch)
    for number in range(70):
        process(proc, cfg, 100 + number, runtime_root=source if number % 2 else None, task_id=f"task-{number}")
    for number in range(5):
        (proc / f"non-process-{number}").touch()
    assert not cfg.shared_root.exists()
    scanner = capture.RunnerProcessCapture(cfg, ledger, legacy_source=source)
    pages = finish(scanner, limit=limit)
    assert sum(page.entries_visited for page in pages) == 77
    assert sum(page.writers_recorded for page in pages) == 70
    for number in range(70):
        entry = Ledger(ledger.root).lookup(f"task-{number}-attempt-1")
        assert entry["payload"] == {"task_id": f"task-{number}", "attempt_number": 1}
        assert entry["captured_writers"][0]["pid"] == 100 + number
        assert entry.get("legacy_source") == (str(source) if number % 2 else None)
    assert not cfg.shared_root.exists()
    target = read_json(cfg.runtime_root / journal.CAPTURE_FILE)
    source_hold = read_json(source / journal.CAPTURE_FILE)
    assert source_hold["capture_id"] == target["capture_id"]
    assert source_hold["target_root"] == str(cfg.runtime_root)
    assert source_hold["instance"] == ledger.instance
    assert target["phase"] == source_hold["phase"] == "pending"
    assert target["pending"] == [] and target["progress"]["is_sweep_complete"]
    with journal.capture_cleanup_guard(source) as can_cleanup:
        assert not can_cleanup
    with journal.capture_cleanup_guard(cfg.runtime_root) as can_cleanup:
        assert not can_cleanup
    with monkeypatch.context() as patch:
        patch.setattr(capture.os, "scandir", lambda *_args: pytest.fail("completed process sweep was enumerated"))
        repeated = capture.RunnerProcessCapture(cfg, Ledger(ledger.root), legacy_source=source).take(limit)
    assert repeated.is_sweep_complete and repeated.entries_visited == repeated.writers_recorded == 0


def test_unfinished_sweep_restarts_and_preserves_prior_membership(tmp_path, monkeypatch):
    cfg, ledger, source, proc = fixture(tmp_path, monkeypatch)
    for pid in range(100, 170):
        process(proc, cfg, pid, task_id=f"task-{pid}")
    scanner = capture.RunnerProcessCapture(cfg, ledger, legacy_source=source)
    first = scanner.take(64)
    scanner.close()
    assert not first.is_sweep_complete
    pages = finish(capture.RunnerProcessCapture(cfg, Ledger(ledger.root), legacy_source=source))
    assert sum(page.entries_visited for page in pages) == 72
    for pid in range(100, 170):
        entry = ledger.lookup(f"task-{pid}-attempt-1")
        assert len(entry["captured_writers"]) == 1


def test_alternating_scanners_do_not_restart_each_others_finite_traversal(tmp_path, monkeypatch):
    cfg, ledger, source, proc = fixture(tmp_path, monkeypatch)
    for pid in range(100, 170):
        process(proc, cfg, pid, task_id=f"task-{pid}")
    scanners = [capture.RunnerProcessCapture(cfg, Ledger(ledger.root), legacy_source=source) for _ in range(2)]
    try:
        for _ in range(12):
            pages = [scanner.take(7) for scanner in scanners]
            if all(page.is_sweep_complete for page in pages):
                break
        else:
            pytest.fail("checkpoint revisions starved concurrent process traversals")
        assert ledger.lookup("task-169-attempt-1")["captured_writers"][0]["pid"] == 169
    finally:
        for scanner in scanners:
            scanner.close()


@pytest.mark.parametrize("boundary", ["target_hold", "source_hold", "batch", "member", "cleared"])
def test_process_crash_replays_exact_legacy_writer_after_process_disappears(tmp_path, monkeypatch, boundary):
    cfg, ledger, source, proc = fixture(tmp_path, monkeypatch)
    process(proc, cfg, 100, runtime_root=source)

    def crash():
        original = journal.atomic_replace

        def write(path, value):
            original(path, value)
            if (
                (boundary == "target_hold" and path.parent == cfg.runtime_root and "progress" not in value)
                or (boundary == "source_hold" and path.parent == source)
                or (boundary == "batch" and value.get("pending"))
                or (boundary == "cleared" and value.get("progress") and value.get("pending") == [])
            ):
                os._exit(86)

        journal.atomic_replace = write
        original_capture = ledger.capture_writer

        def publish(*args, **kwargs):
            value = original_capture(*args, **kwargs)
            if boundary == "member":
                os._exit(86)
            return value

        ledger.capture_writer = publish
        capture.RunnerProcessCapture(cfg, ledger, legacy_source=source).take(64)

    child = multiprocessing.get_context("fork").Process(target=crash)
    child.start()
    try:
        child.join(10)
        assert not child.is_alive() and child.exitcode == 86
    finally:
        if child.is_alive():
            child.kill()
        child.join(5)
        child.close()
    if boundary in {"batch", "member", "cleared"}:
        shutil.rmtree(proc / "100")
    pages = finish(capture.RunnerProcessCapture(cfg, Ledger(ledger.root), legacy_source=source))
    assert pages[-1].is_sweep_complete
    member = ledger.lookup("task-attempt-1")
    assert member["legacy_source"] == str(source)
    assert member["captured_writers"][0]["pid"] == 100
    assert member["captured_writers"][0]["start_time_ticks"] == 1000
    assert read_json(cfg.runtime_root / journal.CAPTURE_FILE)["phase"] == "pending"
    assert read_json(source / journal.CAPTURE_FILE)["phase"] == "pending"


def test_source_hold_failure_admits_no_process_reads_and_preserves_target_hold(tmp_path, monkeypatch):
    cfg, ledger, source, proc = fixture(tmp_path, monkeypatch)
    process(proc, cfg, 100, runtime_root=source)
    original = journal.atomic_replace

    def fail_source(path, value):
        if path.parent == source:
            raise OSError("source barrier failed")
        return original(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(journal, "atomic_replace", fail_source)
        patch.setattr(capture.os, "scandir", lambda *_args: pytest.fail("observed before source retention"))
        with pytest.raises(OSError, match="source barrier"):
            capture.RunnerProcessCapture(cfg, ledger, legacy_source=source).take()
    assert journal.has_pending_writer_capture(cfg.runtime_root)
    assert ledger.find("task-attempt-1") is None
    finish(capture.RunnerProcessCapture(cfg, ledger, legacy_source=source))
    assert ledger.lookup("task-attempt-1")["legacy_source"] == str(source)


def test_another_target_cannot_overwrite_a_source_hold(tmp_path, monkeypatch):
    cfg, ledger, source, _ = fixture(tmp_path, monkeypatch)
    finish(capture.RunnerProcessCapture(cfg, ledger, legacy_source=source))
    before = read_json(source / journal.CAPTURE_FILE)
    other = tmp_path / "other/runtime"
    other.mkdir(parents=True)
    other_ledger = Ledger.open_or_create(responsibility_root(other))
    with pytest.raises(Conflict, match="another writer capture"):
        with journal.WriterCaptureCheckpoint(other_ledger, other, legacy_source=source).observe():
            pytest.fail("source capture was taken over")
    assert read_json(source / journal.CAPTURE_FILE) == before
    assert journal.has_pending_writer_capture(other)


@pytest.mark.parametrize("phase", ["sweeping", "complete", "pending"])
def test_lost_source_hold_cannot_silently_reuse_prior_progress(tmp_path, monkeypatch, phase):
    cfg, ledger, source, proc = fixture(tmp_path, monkeypatch)
    process(proc, cfg, 100, runtime_root=source)
    scanner = capture.RunnerProcessCapture(cfg, ledger, legacy_source=source)
    try:
        if phase == "complete":
            assert scanner.take(64).is_sweep_complete
        elif phase == "pending":
            checkpoint = journal.WriterCaptureCheckpoint(ledger, cfg.runtime_root, legacy_source=source)

            def fail(*_args, **_kwargs):
                raise OSError("injected membership failure")

            with monkeypatch.context() as patch:
                patch.setattr(ledger, "capture_writer", fail)
                with checkpoint.observe() as current:
                    with pytest.raises(OSError, match="injected"):
                        current.record(
                            [
                                {
                                    "identity": "task-attempt-1",
                                    "payload": {"task_id": "task", "attempt_number": 1},
                                    "writer": {**capture._process_scope(), "pid": 100, "start_time_ticks": 1000},
                                    "legacy_source": str(source),
                                }
                            ]
                        )
        else:
            assert not scanner.take(1).is_sweep_complete
    finally:
        scanner.close()
    before = read_json(cfg.runtime_root / journal.CAPTURE_FILE)
    (source / journal.CAPTURE_FILE).unlink()
    with pytest.raises(Unavailable, match="source hold disappeared"):
        capture.RunnerProcessCapture(cfg, Ledger(ledger.root), legacy_source=source).take()
    assert read_json(cfg.runtime_root / journal.CAPTURE_FILE) == before
    assert not (source / journal.CAPTURE_FILE).exists()
    if phase == "pending":
        assert ledger.find("task-attempt-1") is None


@pytest.mark.parametrize("fault", ["reused_pid", "disappeared", "denied", "oversized", "malformed"])
def test_process_faults_cannot_publish_a_false_identity(tmp_path, monkeypatch, fault):
    cfg, ledger, source, proc = fixture(tmp_path, monkeypatch)
    path = process(proc, cfg, 100, runtime_root=source)
    if fault == "oversized":
        with (path / "cmdline").open("ab") as stream:
            stream.write(b"x" * capture.COMMAND_BYTES)
    elif fault == "malformed":
        (path / "stat").write_text("broken")
    else:
        original = capture._process_stat
        reads = 0

        def changed(path, pid):
            nonlocal reads
            reads += 1
            if reads == 2:
                if fault == "disappeared":
                    raise FileNotFoundError()
                if fault == "denied":
                    raise PermissionError("denied")
                return "S", 2000
            return original(path, pid)

        monkeypatch.setattr(capture, "_process_stat", changed)
    scanner = capture.RunnerProcessCapture(cfg, ledger, legacy_source=source)
    try:
        if fault in {"denied", "oversized", "malformed"}:
            with pytest.raises((Unavailable, PermissionError)):
                scanner.take(64)
        else:
            assert scanner.take(64).is_sweep_complete
        assert ledger.find("task-attempt-1") is None
        assert journal.has_pending_writer_capture(cfg.runtime_root)
        assert journal.has_pending_writer_capture(source)
    finally:
        scanner.close()


@pytest.mark.parametrize("field,value", [("host_id", "different"), ("pid_namespace", 123), ("boot_id", "changed")])
def test_changed_process_scope_cannot_reuse_a_completed_sweep(tmp_path, monkeypatch, field, value):
    cfg, ledger, source, _ = fixture(tmp_path, monkeypatch)
    finish(capture.RunnerProcessCapture(cfg, ledger, legacy_source=source))
    before = read_json(cfg.runtime_root / journal.CAPTURE_FILE)
    scope = {**capture._process_scope(), field: value}
    monkeypatch.setattr(capture, "_process_scope", lambda: scope)
    with pytest.raises(Unavailable, match="context changed"):
        capture.RunnerProcessCapture(cfg, ledger, legacy_source=source).take()
    assert read_json(cfg.runtime_root / journal.CAPTURE_FILE) == before


def test_reboot_restart_retains_membership_and_requires_a_new_process_sweep(tmp_path, monkeypatch):
    cfg, ledger, source, proc = fixture(tmp_path, monkeypatch)
    process(proc, cfg, 100, runtime_root=source)
    scanner = capture.RunnerProcessCapture(cfg, ledger, legacy_source=source)
    finish(scanner)
    original = read_json(scanner.checkpoint.path)
    member = ledger.lookup("task-attempt-1")
    source_hold = (source / journal.CAPTURE_FILE).read_bytes()
    shutil.rmtree(proc / "100")
    (proc / "sys/kernel/random/boot_id").write_text("20000000-0000-0000-0000-000000000002")
    process(proc, cfg, 200, task_id="new")
    assert scanner.restart_after_reboot()
    restarted = read_json(scanner.checkpoint.path)
    assert restarted["capture_id"] == original["capture_id"]
    assert restarted["progress"]["revision"] > original["progress"]["revision"]
    assert not restarted["progress"]["is_sweep_complete"]
    assert ledger.lookup("task-attempt-1") == member
    assert (source / journal.CAPTURE_FILE).read_bytes() == source_hold
    with pytest.raises(Unavailable, match="completed process sweep"):
        with scanner.completed_sweep():
            pytest.fail("reboot reused prior process completion")
    finish(scanner)
    assert ledger.lookup("new-attempt-1")["captured_writers"][0]["pid"] == 200
    complete = scanner.checkpoint.path.read_bytes()
    assert not scanner.restart_after_reboot()
    assert scanner.checkpoint.path.read_bytes() == complete


@pytest.mark.parametrize("change", ["host", "namespace", "malformed_boot"])
def test_reboot_restart_does_not_adopt_a_foreign_or_unknown_scope(tmp_path, monkeypatch, change):
    cfg, ledger, source, _ = fixture(tmp_path, monkeypatch)
    scanner = capture.RunnerProcessCapture(cfg, ledger, legacy_source=source)
    finish(scanner)
    before = scanner.checkpoint.path.read_bytes()
    scope = capture._process_scope()
    if change == "host":
        scope.update(host_id="another-host", boot_id="20000000-0000-0000-0000-000000000002")
    elif change == "namespace":
        scope["pid_namespace"] += 1
    else:
        scope["boot_id"] = "invalid"
    monkeypatch.setattr(capture, "_process_scope", lambda: scope)
    with pytest.raises((Unavailable, ValueError)):
        scanner.restart_after_reboot()
    assert scanner.checkpoint.path.read_bytes() == before


@pytest.mark.parametrize("boundary", ["replayed_writer", "reset_commit"])
def test_reboot_restart_crash_replays_old_writer_before_reset(tmp_path, monkeypatch, boundary):
    cfg, ledger, source, proc = fixture(tmp_path, monkeypatch)
    process(proc, cfg, 100, runtime_root=source)
    scanner = capture.RunnerProcessCapture(cfg, ledger, legacy_source=source)
    with monkeypatch.context() as patch:

        def fail(*_args, **_kwargs):
            raise OSError("capture interrupted before membership")

        patch.setattr(ledger, "capture_writer", fail)
        with pytest.raises(OSError, match="before membership"):
            scanner.take(64)
    assert read_json(scanner.checkpoint.path)["pending"]
    shutil.rmtree(proc / "100")
    new_boot = "20000000-0000-0000-0000-000000000002"
    (proc / "sys/kernel/random/boot_id").write_text(new_boot)
    process(proc, cfg, 200, task_id="new")

    def crash():
        original_capture = ledger.capture_writer
        original_write = journal.atomic_replace

        def publish(*args, **kwargs):
            result = original_capture(*args, **kwargs)
            if boundary == "replayed_writer":
                os._exit(86)
            return result

        def write(path, value):
            original_write(path, value)
            if boundary == "reset_commit" and value.get("progress", {}).get("boot_id") == new_boot:
                os._exit(86)

        ledger.capture_writer = publish
        journal.atomic_replace = write
        scanner.restart_after_reboot()

    child = multiprocessing.get_context("fork").Process(target=crash)
    child.start()
    try:
        child.join(10)
        assert child.exitcode == 86
    finally:
        if child.is_alive():
            child.kill()
        child.join(5)
        child.close()
    resumed = capture.RunnerProcessCapture(cfg, Ledger(ledger.root), legacy_source=source)
    resumed.restart_after_reboot()
    finish(resumed)
    old = ledger.lookup("task-attempt-1")
    assert old["captured_writers"][0]["pid"] == 100
    assert old["captured_writers"][0]["boot_id"] != new_boot
    assert old["legacy_source"] == str(source)
    assert ledger.lookup("new-attempt-1")["captured_writers"][0]["boot_id"] == new_boot
    assert read_json(resumed.checkpoint.path)["pending"] == []
    assert read_json(source / journal.CAPTURE_FILE)["phase"] == "pending"


def test_unrelated_and_guardian_commands_are_not_runner_writers(tmp_path, monkeypatch):
    cfg, ledger, source, proc = fixture(tmp_path, monkeypatch)
    process(proc, cfg, 100, command=b"python\0-m\0other.module\0" + b"x" * capture.COMMAND_BYTES)
    guardian = "\0".join(["python", "-m", capture.RUNNER_MODULE, "--guardian", "100", "command", ""])
    process(proc, cfg, 101, command=guardian.encode())
    process(proc, cfg, 102, runtime_root=tmp_path / "unrelated")
    pages = finish(capture.RunnerProcessCapture(cfg, ledger, legacy_source=source))
    assert sum(page.writers_recorded for page in pages) == 0 and not ledger.has_members()


def test_progress_and_pending_batch_are_owned_copies(tmp_path, monkeypatch):
    cfg, ledger, source, _ = fixture(tmp_path, monkeypatch)
    checkpoint = journal.WriterCaptureCheckpoint(ledger, cfg.runtime_root, legacy_source=source)
    progress = {"nested": {"position": 1}}
    with checkpoint.observe() as current:
        current.record([], progress=progress)
        progress["nested"]["position"] = 2
        read = current.progress
        read["nested"]["position"] = 3
        assert current.progress == {"nested": {"position": 1}}
    assert read_json(checkpoint.path)["progress"] == {"nested": {"position": 1}}
