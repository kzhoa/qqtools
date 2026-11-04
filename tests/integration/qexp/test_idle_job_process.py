"""Real Idle Job process ownership and child cleanup, without experiment records."""

import os
import subprocess
import sys
import time
import uuid

import pytest

from qqtools.plugins.qexp.agent.idle_job_process import (
    current_boot_id,
    idle_job_processes_exited,
    read_idle_job_process,
    request_idle_job_stop,
    spawn_idle_job,
)
from qqtools.plugins.qexp.infrastructure.process import process_start_time_ticks
from qqtools.plugins.qexp.runtime.store import atomic_replace

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _wait(predicate, description, seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    pytest.fail(description)


def _spawn(root, command, timeout=0.2):
    instance = uuid.uuid4().hex
    directory = root / instance
    directory.mkdir(parents=True)
    values = {
        "runtime_id": "test-runtime",
        "instance_id": instance,
        "command": tuple(command),
        "working_directory": root,
        "gpu_ids": (1, 3),
        "stop_timeout_seconds": timeout,
        "parent_pid": os.getpid(),
        "parent_start_ticks": process_start_time_ticks(os.getpid()),
    }
    intent = {
        **values,
        "version": 1,
        "boot_id": current_boot_id(),
        "command": list(command),
        "working_directory": str(root),
        "gpu_ids": [1, 3],
    }
    atomic_replace(directory / "intent.json", {"idle_job_intent": intent})
    return directory, spawn_idle_job(directory, **values)


def _cleanup(directory, process):
    request_idle_job_stop(directory, "test_cleanup")
    process.wait(timeout=10)
    _wait(lambda: idle_job_processes_exited(directory), "Idle Job left owned processes behind")


def test_command_receives_resolved_gpus_explicit_cwd_and_separate_log(tmp_path):
    command = [
        sys.executable,
        "-c",
        "import os,time; print(os.environ['CUDA_VISIBLE_DEVICES'],os.getcwd(),flush=True); time.sleep(60)",
    ]
    directory, process = _spawn(tmp_path, command)
    try:
        log = directory / "command.log"
        _wait(lambda: log.exists() and "1,3" in log.read_text(), "command did not start with resolved GPU visibility")
        assert str(tmp_path) in log.read_text()
        record = read_idle_job_process(directory)
        assert record["runtime_id"] == "test-runtime"
        assert record["process_group_id"] == record["pid"]
        assert record["start_time_ticks"] == process_start_time_ticks(record["pid"])
        assert not list(tmp_path.rglob("task.json"))
    finally:
        _cleanup(directory, process)


def test_forced_stop_reclaims_term_ignoring_child(tmp_path):
    marker = tmp_path / "child.pid"
    child = "import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); open('child.pid','w').write(str(os.getpid())); time.sleep(60)"
    parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(60)"
    directory, process = _spawn(tmp_path, [sys.executable, "-c", parent])
    try:
        _wait(lambda: marker.exists() and marker.read_text(), "grandchild did not start")
        request_idle_job_stop(directory, "training_demand")
        process.wait(timeout=10)
        _wait(lambda: idle_job_processes_exited(directory), "TERM-ignoring child survived forced cleanup")
    finally:
        _cleanup(directory, process)


def test_no_process_registration_is_not_cleanup_proof_for_live_parent(tmp_path):
    directory, process = _spawn(tmp_path, [sys.executable, "-c", "import time;time.sleep(60)"])
    try:
        assert not idle_job_processes_exited(directory)
    finally:
        _cleanup(directory, process)


def test_supervisor_death_reclaims_command_group(tmp_path):
    marker = tmp_path / "alive"
    command = "import pathlib,time; pathlib.Path('alive').touch(); time.sleep(60)"
    directory, process = _spawn(tmp_path, [sys.executable, "-c", command])
    try:
        _wait(marker.exists, "command did not start")
        process.kill()
        process.wait(timeout=5)
        _wait(lambda: idle_job_processes_exited(directory), "guardian did not reclaim group after supervisor death")
    finally:
        _cleanup(directory, process)


def test_normal_command_exit_reclaims_remaining_children(tmp_path):
    child = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
    parent = f"import subprocess,sys; subprocess.Popen([sys.executable,'-c',{child!r}])"
    directory, process = _spawn(tmp_path, [sys.executable, "-c", parent])
    try:
        process.wait(timeout=10)
        _wait(lambda: idle_job_processes_exited(directory), "normal command exit left a live descendant")
    finally:
        _cleanup(directory, process)


def test_command_output_retention_is_bounded(tmp_path):
    command = "import sys,time; sys.stdout.write('x' * (4 * 1024 * 1024)); sys.stdout.flush(); time.sleep(60)"
    directory, process = _spawn(tmp_path, [sys.executable, "-c", command])
    try:
        _wait(lambda: (directory / "command.log.1").exists(), "large output was not rotated")
        request_idle_job_stop(directory, "test_cleanup")
        process.wait(timeout=10)
        logs = list(directory.glob("command.log*"))
        assert len(logs) <= 2
        assert sum(path.stat().st_size for path in logs) <= 2 * 1024 * 1024
    finally:
        _cleanup(directory, process)


def test_abrupt_agent_death_reclaims_idle_job(tmp_path):
    directory = tmp_path / uuid.uuid4().hex
    directory.mkdir()
    script = """
import os, sys, time
from pathlib import Path
from qqtools.plugins.qexp.agent.idle_job_process import current_boot_id, spawn_idle_job
from qqtools.plugins.qexp.infrastructure.process import process_start_time_ticks
from qqtools.plugins.qexp.runtime.store import atomic_replace
directory = Path(sys.argv[1])
values = dict(runtime_id='test-runtime', instance_id=directory.name,
    command=(sys.executable, '-c', "import time;print('started',flush=True);time.sleep(60)"),
    working_directory=directory, gpu_ids=(0,), stop_timeout_seconds=0.2,
    parent_pid=os.getpid(), parent_start_ticks=process_start_time_ticks(os.getpid()))
intent = dict(values, version=1, boot_id=current_boot_id(), working_directory=str(directory),
    command=list(values['command']), gpu_ids=[0])
atomic_replace(directory / 'intent.json', {'idle_job_intent':intent})
spawn_idle_job(directory, **values)
time.sleep(60)
"""
    parent = subprocess.Popen([sys.executable, "-c", script, str(directory)])
    try:
        log = directory / "command.log"
        _wait(lambda: log.exists() and "started" in log.read_text(), "parent did not start Idle Job")
        parent.kill()
        parent.wait(timeout=5)
        _wait(lambda: idle_job_processes_exited(directory), "agent death left an unmanaged Idle Job")
    finally:
        request_idle_job_stop(directory, "test_cleanup")
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=5)
        _wait(lambda: idle_job_processes_exited(directory), "cleanup after parent-death test failed")


def test_log_write_failure_stops_owned_processes(tmp_path, monkeypatch):
    original_popen = subprocess.Popen
    bootstrap = """
import sys
from qqtools.plugins.qexp.agent import idle_job_process as module
def fail_write(self, value):
    raise OSError('injected disk full')
module._BoundedCommandLog.write = fail_write
raise SystemExit(module.main(sys.argv[1:]))
"""

    def inject_failure(argv, *args, **kwargs):
        if "--supervisor" in argv:
            argv = [sys.executable, "-c", bootstrap, *argv[3:]]
        return original_popen(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", inject_failure)
    command = "import pathlib,time;pathlib.Path('started').touch();print('output',flush=True);time.sleep(60)"
    directory, process = _spawn(tmp_path, [sys.executable, "-c", command])
    try:
        _wait(lambda: (tmp_path / "started").exists(), "command did not start before injected failure")
        assert process.wait(timeout=5) != 0
        _wait(lambda: idle_job_processes_exited(directory), "log failure stranded owned processes")
    finally:
        _cleanup(directory, process)
