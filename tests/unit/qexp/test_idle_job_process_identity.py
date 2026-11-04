"""Idle Job never signals a PID after its ownership identity changes."""

import signal

import pytest

from qqtools.plugins.qexp.agent import idle_job_process as process
from qqtools.plugins.qexp.infrastructure.process import ProcessIdentityRead


@pytest.mark.parametrize("changes_after_open", [False, True])
def test_reused_pid_is_not_signalled(monkeypatch, changes_after_open):
    identities = iter(
        [ProcessIdentityRead("present", 10), ProcessIdentityRead("present", 11)]
        if changes_after_open
        else [ProcessIdentityRead("present", 11)]
    )
    monkeypatch.setattr(process, "current_boot_id", lambda: "boot")
    monkeypatch.setattr(process, "read_process_identity", lambda _pid: next(identities))
    opened = []
    closed = []
    sent = []
    monkeypatch.setattr(process.os, "pidfd_open", lambda pid, flags: opened.append(pid) or 99)
    monkeypatch.setattr(process.os, "close", closed.append)
    monkeypatch.setattr(process.signal, "pidfd_send_signal", lambda *args: sent.append(args))
    assert not process._signal_exact_process(123, 10, "boot", signal.SIGTERM)
    assert sent == []
    assert opened == ([123] if changes_after_open else [])
    assert closed == ([99] if changes_after_open else [])


def test_unknown_process_identity_is_not_signal_authority(monkeypatch):
    monkeypatch.setattr(process, "current_boot_id", lambda: "boot")
    monkeypatch.setattr(
        process, "read_process_identity", lambda _pid: ProcessIdentityRead("unknown", reason="read_failed")
    )
    sent = []
    monkeypatch.setattr(process.signal, "pidfd_send_signal", lambda *args: sent.append(args))
    assert not process._signal_exact_process(123, 10, "boot", signal.SIGTERM)
    assert sent == []


def test_orphan_reaping_preserves_leader_failure_status(monkeypatch):
    observations = iter([(123, 7 << 8), (456, 0), (0, 0)])
    monkeypatch.setattr(process.os, "waitpid", lambda _pid, _flags: next(observations))
    assert process._reap_orphan_children(123) == 7
