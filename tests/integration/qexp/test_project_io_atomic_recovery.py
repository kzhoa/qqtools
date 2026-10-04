"""Real interrupted atomic writes must not become executor authority evidence."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor
from qqtools.plugins.qexp.agent.setup import initialize_machine

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]

_INTERRUPTED_WRITE = """
import os, signal, sys, time
from pathlib import Path
from qqtools.plugins.qexp.runtime.locks import exclusive
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json

lock, target, source, stage, marker = map(Path, sys.argv[1:])
value = read_json(source)
def observe(current):
    if current != str(stage):
        return
    if str(marker) == 'kill':
        os.kill(os.getpid(), signal.SIGKILL)
    marker.touch()
    deadline = time.monotonic() + 10
    while not marker.with_suffix('.release').exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('test did not release atomic writer')
        time.sleep(0.01)
with exclusive(lock):
    atomic_replace(target, value, io_step_observer=observe)
"""


@pytest.fixture
def executor_case(tmp_path):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    revision, _bindings = runtime.load_registry()
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    yield runtime, executor, binding, revision
    executor.shutdown()


def _writer_command(executor, target, source, stage, marker):
    return [
        sys.executable,
        "-c",
        _INTERRUPTED_WRITE,
        str(executor.paths["project_io_lock"]),
        str(target),
        str(source),
        stage,
        str(marker),
    ]


@pytest.mark.parametrize("lane", ["requests", "processes", "results", "resolved"])
@pytest.mark.parametrize("stage", ["temp_write", "replace"])
@pytest.mark.parametrize("should_restart", [False, True], ids=["poll", "restart"])
def test_sigkill_before_atomic_replace_recovers_executor(executor_case, lane, stage, should_restart):
    runtime, executor, binding, revision = executor_case
    request = executor.prepare_validate_binding(binding, revision)
    source = executor.paths["project_io_requests"] / f"{request.request_id}.json"
    committed_request = source.read_bytes()
    name = f"{request.request_id}.json"
    if lane == "resolved":
        name = f"{1:020d}-{name}"
    target = executor.paths[f"project_io_{lane}"] / name

    child = subprocess.run(
        _writer_command(executor, target, source, stage, "kill"),
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert child.returncode == -signal.SIGKILL, child.stderr.decode()
    temporary = next(target.parent.glob(f".{target.name}.*"))
    assert source.read_bytes() == committed_request
    if lane != "requests":
        assert not target.exists()
    # Read-only observation ignores a temporary without deleting or adopting it.
    assert executor.status_view()["free_slot_count"] == 3
    assert temporary.exists()

    if should_restart and lane == "resolved":
        # Also exercise sequence recovery when the epoch record is unavailable.
        executor.paths["project_io_epoch"].unlink()
    restarted = ProjectIOExecutor(runtime) if should_restart else executor
    try:
        if should_restart:
            restarted.begin_epoch()
        assert restarted.poll()["envelope"] != "unknown"
        assert not temporary.exists()
        if should_restart:
            assert not restarted.unresolved_requests()
            following = restarted.prepare_validate_binding(binding, revision)
        else:
            assert restarted.unresolved_requests() == (request,)
            following = request
        assert restarted.start(following.request_id) is not None
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            restarted.poll()
            result = restarted.consume(following.request_id, following)
            if result is not None:
                break
            time.sleep(0.02)
        else:
            pytest.fail("restarted executor did not finish a fresh binding validation")
        assert result.status == "completed"
        assert result.evidence["project_id"] == binding.project_id
        assert restarted.status_view()["free_slot_count"] == 4
    finally:
        restarted.shutdown()


def test_status_does_not_remove_live_atomic_writer_temporary(executor_case, tmp_path):
    _runtime, executor, binding, revision = executor_case
    request = executor.prepare_validate_binding(binding, revision)
    target = executor.paths["project_io_requests"] / f"{request.request_id}.json"
    marker = tmp_path / "writer-paused"
    child = subprocess.Popen(_writer_command(executor, target, target, "replace", marker))
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            assert child.poll() is None
            time.sleep(0.01)
        assert marker.exists()
        temporary = next(target.parent.glob(f".{target.name}.*"))
        assert executor.status_view()["free_slot_count"] == 3
        assert executor.poll()["envelope"] == "unknown"  # The writer still owns the lock.
        assert temporary.exists()
        marker.with_suffix(".release").touch()
        assert child.wait(timeout=5) == 0
        assert not temporary.exists()
        assert executor.poll()["envelope"] != "unknown"
        assert executor.unresolved_requests() == (request,)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "directory", "unknown_name", "malformed_record"])
def test_temporary_recovery_preserves_invalid_evidence_blockers(executor_case, tmp_path, kind):
    _runtime, executor, _binding, _revision = executor_case
    name = "." + "a" * 32 + ".json.abcdefgh"
    if kind == "unknown_name":
        name = ".unexpected.json.abcdefgh"
    elif kind == "malformed_record":
        name = "a" * 32 + ".json"
    path = executor.paths["project_io_results"] / name
    external = tmp_path / "external"
    external.write_text("preserve")
    if kind == "symlink":
        path.symlink_to(external)
    elif kind == "hardlink":
        os.link(external, path)
    elif kind == "directory":
        path.mkdir()
    else:
        path.write_text("not JSON")
    try:
        assert executor.poll()["envelope"] == "unknown"
        assert path.exists()
        assert external.read_text() == "preserve"
    finally:
        if kind == "directory":
            path.rmdir()
        else:
            path.unlink()
