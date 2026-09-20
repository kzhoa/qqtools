import ast
import errno
import os
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from qqtools.plugins.qexp.infrastructure import process as process_infrastructure
from qqtools.plugins.qexp.infrastructure.process import (
    ProcessIdentityRead,
    ProcessPresence,
    read_process_identity,
    read_process_presence,
)
from qqtools.plugins.qexp.runtime import process_evidence
from qqtools.plugins.qexp.runtime.process_evidence import (
    ProcessEvidence,
    inspect_group_identity,
    inspect_local_group_identity,
    inspect_wrapper_identity,
)


def _identity(state: str, ticks: int | None = None, reason: str | None = None) -> ProcessIdentityRead:
    return ProcessIdentityRead(state=state, start_time_ticks=ticks, reason=reason)


def _presence(state: str, reason: str | None = None) -> ProcessPresence:
    return ProcessPresence(state=state, reason=reason)


def _stat_bytes(pid: int, ticks: object) -> bytes:
    fields = ["S", *("0" for _ in range(18)), str(ticks)]
    return f"{pid} (command with ) parentheses) {' '.join(fields)}".encode()


def test_evidence_results_are_immutable():
    identity = _identity("present", 10)
    presence = _presence("present")
    evidence = ProcessEvidence(state="alive")

    with pytest.raises(FrozenInstanceError):
        identity.state = "absent"
    with pytest.raises(FrozenInstanceError):
        presence.state = "absent"
    with pytest.raises(FrozenInstanceError):
        evidence.state = "absent"


@pytest.mark.parametrize(
    ("manifest", "reason"),
    [
        ({"wrapper_start_time_ticks": 1}, "missing_identity"),
        ({"wrapper_pid": None, "wrapper_start_time_ticks": 1}, "missing_identity"),
        ({"wrapper_pid": 1, "wrapper_start_time_ticks": None}, "missing_identity"),
        ({"wrapper_pid": True, "wrapper_start_time_ticks": 1}, "invalid_identity"),
        ({"wrapper_pid": "1", "wrapper_start_time_ticks": 1}, "invalid_identity"),
        ({"wrapper_pid": 0, "wrapper_start_time_ticks": 1}, "invalid_identity"),
        ({"wrapper_pid": -1, "wrapper_start_time_ticks": 1}, "invalid_identity"),
        ({"wrapper_pid": 1, "wrapper_start_time_ticks": True}, "invalid_identity"),
        ({"wrapper_pid": 1, "wrapper_start_time_ticks": "1"}, "invalid_identity"),
        ({"wrapper_pid": 1, "wrapper_start_time_ticks": -1}, "invalid_identity"),
    ],
)
def test_wrapper_validation_performs_no_probes(monkeypatch, manifest, reason):
    monkeypatch.setattr(
        process_evidence,
        "read_process_identity",
        lambda _pid: pytest.fail("invalid evidence must not read process identity"),
    )
    monkeypatch.setattr(
        process_evidence,
        "read_process_presence",
        lambda _pid, *, is_group: pytest.fail("invalid evidence must not probe presence"),
    )

    assert inspect_wrapper_identity(manifest) == ProcessEvidence(state="unknown", reason=reason)


@pytest.mark.parametrize(
    ("process", "reason"),
    [
        ({"process_group_start_time_ticks": 1}, "missing_identity"),
        ({"process_group_id": 1}, "missing_identity"),
        ({"process_group_id": False, "process_group_start_time_ticks": 1}, "invalid_identity"),
        ({"process_group_id": 1, "process_group_start_time_ticks": -1}, "invalid_identity"),
    ],
)
def test_local_group_validation_performs_no_probes(monkeypatch, process, reason):
    monkeypatch.setattr(
        process_evidence,
        "read_process_identity",
        lambda _pid: pytest.fail("invalid evidence must not read process identity"),
    )
    monkeypatch.setattr(
        process_evidence,
        "read_process_presence",
        lambda _pid, *, is_group: pytest.fail("invalid evidence must not probe presence"),
    )

    assert inspect_local_group_identity(process) == ProcessEvidence(state="unknown", reason=reason)


@pytest.mark.parametrize(
    ("recorded", "manifest", "reason"),
    [
        (
            {"process_group_id": 10, "process_group_start_time_ticks": 20},
            {"process_group_start_time_ticks": 20},
            "missing_identity",
        ),
        (
            {"process_group_id": 10, "process_group_start_time_ticks": 20},
            {"process_group_id": 10, "process_group_start_time_ticks": "20"},
            "invalid_identity",
        ),
        (
            {"process_group_id": 10, "process_group_start_time_ticks": 20},
            {"process_group_id": 11, "process_group_start_time_ticks": 20},
            "identity_mismatch",
        ),
        (
            {"process_group_id": 10, "process_group_start_time_ticks": 20},
            {"process_group_id": 10, "process_group_start_time_ticks": 21},
            "identity_mismatch",
        ),
    ],
)
def test_group_validation_and_disagreement_perform_no_probes(monkeypatch, recorded, manifest, reason):
    monkeypatch.setattr(
        process_evidence,
        "read_process_identity",
        lambda _pid: pytest.fail("unusable evidence must not read process identity"),
    )
    monkeypatch.setattr(
        process_evidence,
        "read_process_presence",
        lambda _pid, *, is_group: pytest.fail("unusable evidence must not probe presence"),
    )

    assert inspect_group_identity(recorded, manifest) == ProcessEvidence(state="unknown", reason=reason)


@pytest.mark.parametrize(
    ("reads", "presence", "expected"),
    [
        (
            [_identity("present", 20), _identity("present", 20)],
            _presence("present"),
            ProcessEvidence(state="alive"),
        ),
        (
            [_identity("absent"), _identity("absent")],
            _presence("absent"),
            ProcessEvidence(state="absent"),
        ),
        (
            [_identity("absent"), _identity("absent")],
            _presence("present"),
            ProcessEvidence(state="unknown", reason="inconsistent_observation"),
        ),
        (
            [_identity("present", 20), _identity("present", 20)],
            _presence("absent"),
            ProcessEvidence(state="unknown", reason="inconsistent_observation"),
        ),
        (
            [_identity("absent"), _identity("present", 20)],
            _presence("absent"),
            ProcessEvidence(state="unknown", reason="inconsistent_observation"),
        ),
        (
            [_identity("present", 20), _identity("absent")],
            _presence("present"),
            ProcessEvidence(state="unknown", reason="inconsistent_observation"),
        ),
        (
            [_identity("absent"), _identity("present", 20)],
            _presence("present"),
            ProcessEvidence(state="unknown", reason="inconsistent_observation"),
        ),
        (
            [_identity("present", 20), _identity("absent")],
            _presence("absent"),
            ProcessEvidence(state="unknown", reason="inconsistent_observation"),
        ),
        (
            [_identity("present", 20), _identity("present", 21)],
            _presence("present"),
            ProcessEvidence(state="unknown", reason="identity_mismatch"),
        ),
        (
            [_identity("present", 20), _identity("unknown", reason="read_failed")],
            _presence("present"),
            ProcessEvidence(state="unknown", reason="read_failed"),
        ),
    ],
)
def test_wrapper_sampling_truth_table_and_probe_budget(monkeypatch, reads, presence, expected):
    calls: list[tuple[str, int, bool | None]] = []
    remaining = iter(reads)

    def read_identity(pid):
        calls.append(("identity", pid, None))
        return next(remaining)

    def read_presence(pid, *, is_group):
        calls.append(("presence", pid, is_group))
        return presence

    monkeypatch.setattr(process_evidence, "read_process_identity", read_identity)
    monkeypatch.setattr(process_evidence, "read_process_presence", read_presence)

    assert inspect_wrapper_identity({"wrapper_pid": 10, "wrapper_start_time_ticks": 20}) == expected
    assert calls == [("identity", 10, None), ("presence", 10, False), ("identity", 10, None)]


@pytest.mark.parametrize("reason", ["read_failed", "invalid_process_stat", "unsupported_probe"])
def test_unknown_identity_stops_before_presence(monkeypatch, reason):
    monkeypatch.setattr(process_evidence, "read_process_identity", lambda _pid: _identity("unknown", reason=reason))
    monkeypatch.setattr(
        process_evidence,
        "read_process_presence",
        lambda _pid, *, is_group: pytest.fail("unknown identity must stop sampling"),
    )

    assert inspect_wrapper_identity({"wrapper_pid": 10, "wrapper_start_time_ticks": 20}) == ProcessEvidence(
        state="unknown", reason=reason
    )


def test_pid_reuse_stops_before_presence(monkeypatch):
    monkeypatch.setattr(process_evidence, "read_process_identity", lambda _pid: _identity("present", 21))
    monkeypatch.setattr(
        process_evidence,
        "read_process_presence",
        lambda _pid, *, is_group: pytest.fail("identity mismatch must stop sampling"),
    )

    assert inspect_wrapper_identity({"wrapper_pid": 10, "wrapper_start_time_ticks": 20}) == ProcessEvidence(
        state="unknown", reason="identity_mismatch"
    )


def test_unknown_presence_stops_before_identity_recheck(monkeypatch):
    identity_reads = 0

    def read_identity(_pid):
        nonlocal identity_reads
        identity_reads += 1
        return _identity("present", 20)

    monkeypatch.setattr(process_evidence, "read_process_identity", read_identity)
    monkeypatch.setattr(
        process_evidence,
        "read_process_presence",
        lambda _pid, *, is_group: _presence("unknown", "read_failed"),
    )

    assert inspect_wrapper_identity({"wrapper_pid": 10, "wrapper_start_time_ticks": 20}) == ProcessEvidence(
        state="unknown", reason="read_failed"
    )
    assert identity_reads == 1


def test_group_and_wrapper_presence_probes_stay_independent(monkeypatch):
    group_flags: list[bool] = []
    reads = iter([_identity("present", 20), _identity("present", 20), _identity("absent"), _identity("absent")])
    monkeypatch.setattr(process_evidence, "read_process_identity", lambda _pid: next(reads))

    def read_presence(_pid, *, is_group):
        group_flags.append(is_group)
        return _presence("present" if is_group else "absent")

    monkeypatch.setattr(process_evidence, "read_process_presence", read_presence)
    group = {"process_group_id": 10, "process_group_start_time_ticks": 20}

    assert inspect_group_identity(group, group) == ProcessEvidence(state="alive")
    assert inspect_wrapper_identity({"wrapper_pid": 11, "wrapper_start_time_ticks": 20}) == ProcessEvidence(
        state="absent"
    )
    assert group_flags == [True, False]


@pytest.mark.parametrize("pid", [0, -1, True, "1"])
def test_read_process_identity_rejects_invalid_pid_without_touching_proc(monkeypatch, pid):
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda _path: pytest.fail("invalid pid must not construct or read a proc path"),
    )

    assert read_process_identity(pid) == _identity("unknown", reason="invalid_identity")


def test_read_process_identity_parses_parenthesized_command(tmp_path, monkeypatch):
    proc_root = tmp_path / "proc"
    stat_path = proc_root / "123" / "stat"
    stat_path.parent.mkdir(parents=True)
    stat_path.write_bytes(_stat_bytes(123, 456))
    monkeypatch.setattr(process_infrastructure, "_PROC_ROOT", proc_root)

    assert read_process_identity(123) == _identity("present", 456)


@pytest.mark.parametrize(
    "contents",
    [
        b"123 missing parenthesis",
        b"123 (command) S 0",
        _stat_bytes(123, "not-an-int"),
        _stat_bytes(123, -1),
        b"\xff",
    ],
)
def test_read_process_identity_reports_malformed_stat(tmp_path, monkeypatch, contents):
    proc_root = tmp_path / "proc"
    stat_path = proc_root / "123" / "stat"
    stat_path.parent.mkdir(parents=True)
    stat_path.write_bytes(contents)
    monkeypatch.setattr(process_infrastructure, "_PROC_ROOT", proc_root)

    assert read_process_identity(123) == _identity("unknown", reason="invalid_process_stat")


def test_read_process_identity_distinguishes_missing_pid_from_unavailable_proc(tmp_path, monkeypatch):
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    monkeypatch.setattr(process_infrastructure, "_PROC_ROOT", proc_root)
    assert read_process_identity(123) == _identity("absent")

    monkeypatch.setattr(process_infrastructure, "_PROC_ROOT", tmp_path / "missing-proc")
    assert read_process_identity(123) == _identity("unknown", reason="unsupported_probe")


def test_read_process_identity_preserves_permission_failure(tmp_path, monkeypatch):
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    monkeypatch.setattr(process_infrastructure, "_PROC_ROOT", proc_root)

    def deny(_path):
        raise PermissionError(errno.EACCES, "denied")

    monkeypatch.setattr(Path, "read_bytes", deny)
    assert read_process_identity(123) == _identity("unknown", reason="read_failed")


def test_read_process_identity_preserves_nonpermission_io_failure(tmp_path, monkeypatch):
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    monkeypatch.setattr(process_infrastructure, "_PROC_ROOT", proc_root)

    def fail(_path):
        raise OSError(errno.EIO, "I/O failure")

    monkeypatch.setattr(Path, "read_bytes", fail)
    assert read_process_identity(123) == _identity("unknown", reason="read_failed")


@pytest.mark.parametrize("is_group", [False, True])
def test_read_process_presence_classifies_zero_signal_results(monkeypatch, is_group):
    probe = "killpg" if is_group else "kill"
    calls = []

    def record(pid, sent_signal):
        calls.append((pid, sent_signal))

    monkeypatch.setattr(process_infrastructure.os, probe, record)
    assert read_process_presence(123, is_group=is_group) == _presence("present")
    assert calls == [(123, 0)]

    monkeypatch.setattr(
        process_infrastructure.os,
        probe,
        lambda _pid, _signal: (_ for _ in ()).throw(ProcessLookupError(errno.ESRCH, "missing")),
    )
    assert read_process_presence(123, is_group=is_group) == _presence("absent")

    monkeypatch.setattr(
        process_infrastructure.os,
        probe,
        lambda _pid, _signal: (_ for _ in ()).throw(PermissionError(errno.EPERM, "denied")),
    )
    assert read_process_presence(123, is_group=is_group) == _presence("unknown", "read_failed")

    monkeypatch.setattr(
        process_infrastructure.os,
        probe,
        lambda _pid, _signal: (_ for _ in ()).throw(OSError(errno.EIO, "I/O failure")),
    )
    assert read_process_presence(123, is_group=is_group) == _presence("unknown", "read_failed")


@pytest.mark.parametrize("is_group", [False, True])
@pytest.mark.parametrize("failure", ["enosys", "not_implemented"])
def test_read_process_presence_reports_unsupported_probe(monkeypatch, is_group, failure):
    probe = "killpg" if is_group else "kill"

    def unsupported(_pid, _signal):
        if failure == "enosys":
            raise OSError(errno.ENOSYS, "unsupported")
        raise NotImplementedError

    monkeypatch.setattr(process_infrastructure.os, probe, unsupported)
    assert read_process_presence(123, is_group=is_group) == _presence("unknown", "unsupported_probe")


@pytest.mark.parametrize("pid", [0, -1, False, "1"])
def test_read_process_presence_rejects_invalid_pid_before_syscall(monkeypatch, pid):
    monkeypatch.setattr(
        process_infrastructure.os,
        "kill",
        lambda *_args: pytest.fail("invalid pid must not reach kill"),
    )
    monkeypatch.setattr(
        process_infrastructure.os,
        "killpg",
        lambda *_args: pytest.fail("invalid pid must not reach killpg"),
    )

    assert read_process_presence(pid, is_group=False) == _presence("unknown", "invalid_identity")
    assert read_process_presence(pid, is_group=True) == _presence("unknown", "invalid_identity")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux proc identity contract")
def test_real_local_wrapper_identity_is_alive():
    identity = read_process_identity(os.getpid())
    assert identity.state == "present"
    assert identity.start_time_ticks is not None
    assert inspect_wrapper_identity(
        {"wrapper_pid": os.getpid(), "wrapper_start_time_ticks": identity.start_time_ticks}
    ) == ProcessEvidence(state="alive")


def test_evidence_consumers_do_not_import_scheduler_private_process_helpers():
    repository_root = Path(__file__).parents[3]
    source_root = repository_root / "src" / "qqtools" / "plugins" / "qexp"
    forbidden = {
        "_is_process_alive",
        "_is_process_group_alive",
        "_manifest_supervisor",
        "_process_evidence_state",
        "_process_start_time_ticks",
    }
    scheduler_tree = ast.parse((source_root / "scheduler.py").read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in forbidden
        for node in scheduler_tree.body
    )
    scheduler_imports = {
        alias.asname or alias.name
        for node in ast.walk(scheduler_tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert not scheduler_imports & forbidden

    for relative_path in ("authority.py", "doctor.py", "runtime/attempt_recovery.py"):
        tree = ast.parse((source_root / relative_path).read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("scheduler")
            for alias in node.names
        }
        assert not imported & forbidden, relative_path
