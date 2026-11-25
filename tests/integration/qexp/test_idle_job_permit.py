"""A prepared guardian must have current one-shot machine permission to run a command."""

import json
import os
import signal
import socket
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

from qqtools.plugins.qexp.agent.idle_job_process import current_boot_id, idle_job_processes_exited, spawn_idle_job
from qqtools.plugins.qexp.infrastructure.process import process_start_time_ticks
from qqtools.plugins.qexp.runtime.resources.cpu_lane import reserve_cpu, set_cpu_lane_capacity
from qqtools.plugins.qexp.runtime.resources.idle_job import (
    IdleJobAdmissionBlocked,
    idle_job_guard,
    idle_job_instance_path,
    read_idle_job_demand,
    read_idle_job_ownership,
    write_idle_job_ownership,
)
from qqtools.plugins.qexp.runtime.resources.reservations import reserve
from qqtools.plugins.qexp.runtime.store import atomic_replace

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


@pytest.fixture
def prepared(tmp_path):
    instance_id = uuid.uuid4().hex
    directory = idle_job_instance_path(tmp_path, instance_id)
    directory.mkdir(parents=True)
    marker = tmp_path / "command-starts"
    command = (
        sys.executable,
        "-c",
        "from pathlib import Path; import sys,time; Path(sys.argv[1]).open('a').write('ran\\n'); time.sleep(20)",
        str(marker),
    )
    ident = dict(
        runtime_id="machine",
        instance_id=instance_id,
        parent_pid=os.getpid(),
        parent_start_ticks=process_start_time_ticks(os.getpid()),
    )
    args = dict(**ident, command=command, working_directory=tmp_path, gpu_ids=(0,), stop_timeout_seconds=0.2)
    intent = dict(**args, version=2, boot_id=current_boot_id())
    intent.update(command=list(command), working_directory=str(tmp_path), gpu_ids=[0])
    atomic_replace(directory / "intent.json", {"idle_job_intent": intent})
    owner = dict(
        **ident,
        version=2,
        phase="prepared",
        gpu_ids=[0],
        config_revision=1,
        boot_id=current_boot_id(),
        created_at="2026-11-25T00:00:00Z",
    )
    with idle_job_guard(tmp_path):
        write_idle_job_ownership(tmp_path, owner)
    set_cpu_lane_capacity(tmp_path, capacity=1)
    with _permit_pair() as (sender, receiver):
        helper = spawn_idle_job(directory, **args, permit_socket=receiver)
        try:
            sender.settimeout(8)
            assert sender.recv(32) == b"ready"
            assert not marker.exists()
            yield sender, owner, helper, directory, marker
        finally:
            sender.close()
            if helper.poll() is None:
                helper.send_signal(signal.SIGCONT)
                helper.terminate()
            helper.wait(timeout=10)
            assert idle_job_processes_exited(directory)


@contextmanager
def _permit_pair():
    left, right = socket.socketpair(type=socket.SOCK_SEQPACKET)
    with left, right:
        yield left, right


@pytest.mark.parametrize("lane", ["gpu", "cpu"])
def test_delivered_permit_cannot_overtake_effective_stop(tmp_path, prepared, lane):
    channel, owner, helper, directory, marker = prepared
    helper.send_signal(signal.SIGSTOP)
    _, status = os.waitpid(helper.pid, os.WUNTRACED)
    assert os.WIFSTOPPED(status)
    try:
        channel.send(f"{owner['instance_id']} {time.monotonic() + 10} {read_idle_job_demand(tmp_path)}".encode())
        with pytest.raises(IdleJobAdmissionBlocked):
            if lane == "cpu":
                reserve_cpu(tmp_path, "training", 1)
            else:
                reserve(tmp_path, "training", [0])
        assert read_idle_job_ownership(tmp_path)["phase"] == "stopping"
    finally:
        helper.send_signal(signal.SIGCONT)
    helper.wait(timeout=8)
    assert idle_job_processes_exited(directory)
    assert not marker.exists()


@pytest.mark.parametrize("mode", ["allow", "expired", "foreign", "eof"])
def test_prepared_instance_obeys_exact_permit(tmp_path, prepared, mode):
    channel, owner, helper, directory, marker = prepared
    if mode == "eof":
        channel.close()
    else:
        ident = uuid.uuid4().hex if mode == "foreign" else owner["instance_id"]
        expiry = time.monotonic() + (-1 if mode == "expired" else 10)
        channel.send(f"{ident} {expiry} {read_idle_job_demand(tmp_path)}".encode())
    if mode == "allow":
        deadline = time.monotonic() + 5
        while not marker.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.read_text().splitlines() == ["ran"]
        with pytest.raises(IdleJobAdmissionBlocked):
            reserve(tmp_path, "training", [0])
    helper.wait(timeout=8)
    assert idle_job_processes_exited(directory)
    assert marker.exists() == (mode == "allow")


def test_go_before_training_with_guardian_paused_before_command(tmp_path, prepared):
    channel, owner, helper, directory, marker = prepared
    # The supervisor accepts GO under the admission lock. Pause the registered
    # guardian to separate that decision from creation of the user process.
    record = json.loads((directory / "process.json").read_text())["idle_job_process"]
    guardian = record["pid"]
    os.kill(guardian, signal.SIGSTOP)
    try:
        end = time.monotonic() + 5
        while "T" not in Path(f"/proc/{guardian}/status").read_text().split("State:", 1)[1].splitlines()[0]:
            assert time.monotonic() < end
            time.sleep(0.005)
        channel.send(f"{owner['instance_id']} {end} {read_idle_job_demand(tmp_path)}".encode())
        while read_idle_job_ownership(tmp_path)["phase"] != "launching":
            assert time.monotonic() < end
            time.sleep(0.005)
        assert not marker.exists()
        with pytest.raises(IdleJobAdmissionBlocked):
            reserve(tmp_path, "training", [0])
        assert read_idle_job_ownership(tmp_path)["phase"] == "stopping"
        assert not idle_job_processes_exited(directory)
    finally:
        os.kill(guardian, signal.SIGCONT)
    helper.wait(timeout=8)
    assert idle_job_processes_exited(directory)


@pytest.mark.parametrize("lane", ["cpu", "gpu", "malformed"])
def test_go_rechecks_durable_demand_before_running(tmp_path, prepared, monkeypatch, lane):
    from qqtools.plugins.qexp.runtime.resources import idle_job as admission

    channel, owner, helper, directory, marker = prepared
    revision = read_idle_job_demand(tmp_path)
    helper.send_signal(signal.SIGSTOP)
    _, status = os.waitpid(helper.pid, os.WUNTRACED)
    assert os.WIFSTOPPED(status)
    try:
        expiry = time.monotonic() + 10
        channel.send(f"{owner['instance_id']} {expiry} {revision}".encode())
        if lane == "malformed":
            atomic_replace(tmp_path / "idle-job" / "demand.json", {"invalid": True})
        else:

            def fail_stop(path, value):
                if path == directory / "stop.json":
                    raise OSError("stop publication failed")
                atomic_replace(path, value)

            monkeypatch.setattr(admission, "atomic_replace", fail_stop)
            with pytest.raises(OSError, match="stop publication failed"):
                if lane == "cpu":
                    reserve_cpu(tmp_path, "training", 1)
                else:
                    reserve(tmp_path, "training", [0])
            assert read_idle_job_demand(tmp_path) == revision + 1
        assert read_idle_job_ownership(tmp_path)["phase"] == "prepared"
        assert not (directory / "stop.json").exists()
    finally:
        helper.send_signal(signal.SIGCONT)
    helper.wait(timeout=8)
    assert time.monotonic() < expiry
    assert idle_job_processes_exited(directory)
    assert read_idle_job_ownership(tmp_path)["phase"] == "prepared"
    assert not marker.exists()
