import os
import time
from dataclasses import replace
from pathlib import Path

import pytest

from qqtools.plugins.qexp.runtime import responsibility_backfill as backfill
from qqtools.plugins.qexp.runtime.paths import local_paths
from qqtools.plugins.qexp.runtime.responsibility import responsibility_root
from qqtools.plugins.qexp.runtime.responsibility_import import RECORD_KEYS
from qqtools.plugins.qexp.runtime.responsibility_store import Ledger, Unavailable
from qqtools.plugins.qexp.runtime.store import atomic_replace

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("lane", RECORD_KEYS)
def test_source_locator_read_and_local_apply_have_disjoint_io(tmp_path, monkeypatch, lane):
    source = tmp_path / "source"
    target = tmp_path / "target"
    relative = Path("task-attempt-1/decision.json" if lane == "termination_decisions" else "task-attempt-1.json")
    path = local_paths(source)[lane] / relative
    atomic_replace(
        path,
        {RECORD_KEYS[lane]: {"attempt_id": "task-attempt-1", "command": "private", "env": {"SECRET": "private"}}},
    )
    original = path.read_bytes()
    with monkeypatch.context() as guarded:
        guarded.setattr(backfill, "evidence_write_guard", lambda *_: pytest.fail("source read took target lock"))
        captured = backfill.read_capture_locator(source, lane, relative, should_require_record=True)
    assert captured.payload == {"task_id": "task", "attempt_number": 1}
    target.mkdir()
    ledger = Ledger.open_or_create(responsibility_root(target))
    with monkeypatch.context() as guarded:
        for name in ("resolve", "stat", "lstat", "open"):
            original_method = getattr(Path, name)

            def local_only(self, *args, _method=original_method, **kwargs):
                if self == source or source in self.parents:
                    pytest.fail("local apply accessed source filesystem")
                return _method(self, *args, **kwargs)

            guarded.setattr(Path, name, local_only)
        backfill.apply_capture_locator(ledger, target, captured)
        first = ledger.lookup(captured.identity)
        backfill.apply_capture_locator(ledger, target, captured)
        assert ledger.lookup(captured.identity) == first
    assert first["legacy_source"] == str(source)
    assert path.read_bytes() == original


def test_missing_retained_record_is_not_empty_evidence(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    assert backfill.read_capture_locator(source, "observations", Path("task-attempt-1.json")) is None
    with pytest.raises(Unavailable):
        backfill.read_capture_locator(source, "observations", Path("task-attempt-1.json"), should_require_record=True)


def test_source_locator_read_has_a_record_size_limit(tmp_path, monkeypatch):
    source = tmp_path / "source"
    relative = Path("task-attempt-1.json")
    atomic_replace(local_paths(source)["observations"] / relative, {"exit_observation": {"padding": "x" * 2048}})
    monkeypatch.setattr(backfill, "_MAX_EVIDENCE_RECORD_BYTES", 1024)
    with pytest.raises((ValueError, RuntimeError)):
        backfill.read_capture_locator(source, "observations", relative, should_require_record=True)


@pytest.mark.parametrize("kind", ["symlink", "directory", "fifo", "parent_symlink"])
def test_source_locator_read_rejects_nonregular_evidence_without_opening_it(tmp_path, kind):
    source = tmp_path / "source"
    relative = Path("task-attempt-1.json")
    directory = local_paths(source)["observations"]
    directory.parent.mkdir(parents=True)
    if kind == "parent_symlink":
        other = tmp_path / "other"
        other.mkdir()
        atomic_replace(other / relative, {"exit_observation": {"attempt_id": "task-attempt-1"}})
        directory.symlink_to(other, target_is_directory=True)
    else:
        directory.mkdir()
        path = directory / relative
        if kind == "symlink":
            other = tmp_path / "other.json"
            atomic_replace(other, {"exit_observation": {"attempt_id": "task-attempt-1"}})
            path.symlink_to(other)
        elif kind == "directory":
            path.mkdir()
        else:
            os.mkfifo(path)
    with pytest.raises(Unavailable):
        backfill.read_capture_locator(source, "observations", relative, should_require_record=True)


@pytest.mark.parametrize("change", ["none", "disabled", "common_admission", "batch_changed", "source_missing"])
def test_actual_source_worker_reads_without_machine_guards_or_target_writes(tmp_path, monkeypatch, change):
    from qqtools.plugins.qexp import init_shared_root
    from qqtools.plugins.qexp.agent.context import MachineRuntime
    from qqtools.plugins.qexp.agent.project_io_controller import ProjectIOController
    from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor
    from qqtools.plugins.qexp.agent.recovery_capture import recovery_owner
    from qqtools.plugins.qexp.agent.setup import initialize_machine
    from qqtools.plugins.qexp.runtime.responsibility_capture import (
        CAPTURE_FILE,
        CAPTURE_FORMAT,
        SOURCE_CAPTURE_FORMAT,
        capture_admission,
    )
    from qqtools.plugins.qexp.runtime.responsibility_process_capture import SCAN_FORMAT

    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg = init_shared_root(tmp_path / "project/.qexp", "gpu-1")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    if change == "disabled":
        binding = runtime.set_enabled(binding.project_id, False)
    revision, _ = runtime.load_registry()
    target = runtime.project_paths(binding.project_id)["root"]
    target.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source"
    relative = Path("task-attempt-1.json")
    source_record = local_paths(source)["observations"] / relative
    atomic_replace(source_record, {"exit_observation": {"attempt_id": "task-attempt-1", "env": {"SECRET": "private"}}})
    instance = Ledger.open_or_create(responsibility_root(target)).instance
    capture_id = "d" * 32
    backfill_id = "e" * 32
    atomic_replace(
        target / CAPTURE_FILE,
        {
            "format": CAPTURE_FORMAT,
            "phase": "pending",
            "runtime_root": str(target),
            "legacy_source": str(source),
            "instance": instance,
            "capture_id": capture_id,
            "admission": capture_admission(recovery_owner(runtime, binding)),
            "pending": [],
            "progress": {"format": SCAN_FORMAT, "revision": 1, "is_sweep_complete": True},
        },
    )
    pending = {
        "format": backfill.FORMAT,
        "instance": instance,
        "capture_id": backfill_id,
        "writer_capture_id": capture_id,
        "writer_sweep_revision": 1,
        "revision": 1 if change != "batch_changed" else 2,
        "lane": len(backfill.LANES) + backfill.LANES.index("observations"),
        "sources": [str(target), str(source)],
        "pending": [str(relative)],
    }
    atomic_replace(target / "responsibility-capture-backfill.json", pending)
    atomic_replace(
        source / CAPTURE_FILE,
        {
            "format": SOURCE_CAPTURE_FORMAT,
            "phase": "pending",
            "runtime_root": str(source),
            "target_root": str(target),
            "instance": instance,
            "capture_id": capture_id,
        },
    )
    if change == "source_missing":
        source_record.unlink()
    before = {str(path.relative_to(target)): path.read_bytes() for path in target.rglob("*") if path.is_file()}
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    try:
        parameters = {
            "capture_id": capture_id,
            "backfill_id": backfill_id,
            "backfill_revision": 1,
            "lane": "observations",
            "relative": str(relative),
        }
        if change == "common_admission":
            from qqtools.plugins.qexp import layout
            from qqtools.plugins.qexp.agent import project_io_worker as worker

            def forbidden(*_args, **_kwargs):
                pytest.fail("local coordinator performed shared/source work inline")

            monkeypatch.setattr(layout, "load_root_config", forbidden)
            monkeypatch.setattr(worker, "_legacy_capture_read", forbidden)
            controller = ProjectIOController(runtime, executor)
            deadline = time.monotonic() + 5
            completed = {}
            while time.monotonic() < deadline:
                completed = controller.advance_legacy_capture_reads(
                    [binding], revision, {binding.project_id: parameters}
                )
                if binding.project_id in completed:
                    break
                time.sleep(0.01)
            assert completed[binding.project_id]["state"] == "observed"
            assert not executor.unresolved_requests()
            after = {str(path.relative_to(target)): path.read_bytes() for path in target.rglob("*") if path.is_file()}
            assert after == before
            return
        prepared = executor.prepare_legacy_capture_read(binding, revision, **parameters)
        with runtime.registry_guard(), runtime.binding_commit_guard(binding):
            assert executor.start(prepared.request_id) is not None
            deadline = time.monotonic() + 5
            result = None
            while time.monotonic() < deadline:
                executor.poll()
                result = executor.consume(prepared.request_id, prepared)
                if result is not None:
                    break
                time.sleep(0.01)
            assert result is not None, "source worker blocked behind parent MachineRuntime guards"
        if change == "source_missing":
            assert result.status == "retryable_error"
            assert result.reason_code == "project_io_legacy_capture_read_failed"
        elif change == "batch_changed":
            assert result.status == "completed"
            assert dict(result.evidence) == {"state": "stale", "locator": None}
        else:
            assert result.status == "completed"
            assert result.evidence["state"] == "observed"
            assert set(result.evidence["locator"]) == {
                "source_root",
                "lane",
                "relative",
                "identity",
                "task_id",
                "attempt_number",
            }
        after = {str(path.relative_to(target)): path.read_bytes() for path in target.rglob("*") if path.is_file()}
        assert after == before
        assert (source / CAPTURE_FILE).exists()
    finally:
        executor.shutdown()


def test_source_context_accepts_real_interrupted_retained_capture(tmp_path, monkeypatch):
    from qqtools.plugins.qexp import init_shared_root
    from qqtools.plugins.qexp.runtime import responsibility_process_capture as processes
    from qqtools.plugins.qexp.runtime.responsibility_source_read import (
        load_source_read_context,
        read_retained_source_locator,
    )
    from qqtools.plugins.qexp.runtime.store import read_json

    source = tmp_path / "source"
    target = tmp_path / "target"
    target.mkdir()
    cfg = init_shared_root(tmp_path / "project/.qexp", "gpu-1", runtime_root=source)
    relative = Path("task-attempt-1.json")
    atomic_replace(
        local_paths(source)["observations"] / relative, {"exit_observation": {"attempt_id": "task-attempt-1"}}
    )
    proc = tmp_path / "proc"
    (proc / "sys/kernel/random").mkdir(parents=True)
    (proc / "sys/kernel/random/boot_id").write_text(Path("/proc/sys/kernel/random/boot_id").read_text())
    (proc / "self/ns").mkdir(parents=True)
    (proc / "self/ns/pid").symlink_to("/proc/self/ns/pid")
    monkeypatch.setattr(processes, "PROC_ROOT", proc)
    ledger = Ledger.open_or_create(responsibility_root(target))
    scanner = processes.RunnerProcessCapture(replace(cfg, runtime_root=target), ledger, legacy_source=source)
    owner = {
        "project_id": "project-a",
        "shared_root": str(cfg.shared_root),
        "machine_name": cfg.machine_name,
        "owner_root": str(tmp_path / "machine"),
        "owner_instance": "a" * 64,
        "registration_generation": "generation-a",
    }
    scanner.prepare_admission(owner)
    assert scanner.take().is_sweep_complete
    capture = backfill.ResponsibilityBackfill(target, process_capture=scanner)
    original_capture = backfill.capture_local_record

    def interrupt_source(*args, **kwargs):
        if kwargs.get("source_root") == source:
            raise OSError("source read interrupted after pending publication")
        return original_capture(*args, **kwargs)

    try:
        with monkeypatch.context() as guarded:
            guarded.setattr(backfill, "capture_local_record", interrupt_source)
            with pytest.raises(OSError, match="source read interrupted"):
                capture.take(64, should_cross_lanes=True)
        pending = read_json(capture.path)
        assert pending["pending"] == [str(relative)]
        context = load_source_read_context(
            target,
            owner,
            capture_id=pending["writer_capture_id"],
            backfill_id=pending["capture_id"],
            backfill_revision=pending["revision"],
            lane="observations",
            relative=str(relative),
        )
        captured = read_retained_source_locator(context, "observations", str(relative))
        backfill.apply_capture_locator(ledger, target, captured)
        assert ledger.lookup(captured.identity)["legacy_source"] == str(source)
        assert read_json(capture.path) == pending
    finally:
        capture.close()
