import hashlib
from pathlib import Path

import pytest

from qqtools.plugins.qexp.agent.project_io import executor as project_io_executor
from qqtools.plugins.qexp.agent.project_io import transport
from qqtools.plugins.qexp.agent.project_io import worker as project_io_worker


def _stat_record(pid: int, ticks: object) -> str:
    fields = ["S", *("0" for _ in range(18)), str(ticks)]
    return f"{pid} (command with ) parentheses) {' '.join(fields)}"


@pytest.mark.parametrize("pid", [True, 0, -1, "1", None])
def test_process_start_ticks_reject_invalid_pid_without_reading(monkeypatch, pid):
    monkeypatch.setattr(
        transport.Path,
        "read_text",
        lambda *_args, **_kwargs: pytest.fail("invalid PID must not read proc"),
    )

    assert transport._process_start_time_ticks(pid) is None


@pytest.mark.parametrize(
    ("record_or_error", "expected"),
    [
        (_stat_record(17, 23), 23),
        ("17 malformed", None),
        (_stat_record(17, "invalid"), None),
        (_stat_record(17, 0), None),
        (PermissionError(), None),
        (UnicodeError(), None),
    ],
)
def test_process_start_ticks_preserve_proc_read_outcomes(monkeypatch, record_or_error, expected):
    def read_text(_path, *, encoding):
        assert encoding == "ascii"
        if isinstance(record_or_error, BaseException):
            raise record_or_error
        return record_or_error

    monkeypatch.setattr(transport.sys, "platform", "linux")
    monkeypatch.setattr(transport.Path, "read_text", read_text)

    assert transport._process_start_time_ticks(17) == expected


@pytest.mark.parametrize(
    ("record_or_error", "expected"),
    [
        (_stat_record(17, 23), transport._PROCESS_LIVE),
        (_stat_record(17, 24), transport._PROCESS_ABSENT),
        (FileNotFoundError(), transport._PROCESS_ABSENT),
        (PermissionError(), transport._PROCESS_UNVERIFIED),
        (UnicodeError(), transport._PROCESS_UNVERIFIED),
        ("17 malformed", transport._PROCESS_UNVERIFIED),
        (_stat_record(17, "invalid"), transport._PROCESS_UNVERIFIED),
    ],
)
def test_inspect_process_identity_preserves_live_absent_and_unverified(monkeypatch, record_or_error, expected):
    def read_text(path, *, encoding):
        assert path == Path("/proc/17/stat")
        assert encoding == "ascii"
        if isinstance(record_or_error, BaseException):
            raise record_or_error
        return record_or_error

    monkeypatch.setattr(transport.sys, "platform", "linux")
    monkeypatch.setattr(transport.Path, "is_dir", lambda path: path == Path("/proc"))
    monkeypatch.setattr(transport.Path, "read_text", read_text)

    assert transport.inspect_process_identity(17, 23) == expected


def test_inspect_process_identity_does_not_probe_unsupported_platform(monkeypatch):
    monkeypatch.setattr(transport.sys, "platform", "darwin")
    monkeypatch.setattr(
        transport.Path,
        "is_dir",
        lambda _path: pytest.fail("unsupported platform must not probe proc"),
    )

    assert transport.inspect_process_identity(17, 23) == transport._PROCESS_UNVERIFIED


@pytest.mark.parametrize(
    ("identity", "expected"),
    [
        ({"machine_runtime": {"instance_id": "seed", "runtime_id": "a" * 64}}, "a" * 64),
        (
            {"machine_runtime": {"instance_id": "seed"}},
            hashlib.sha256(b"seed\0host-a").hexdigest(),
        ),
        (
            {"machine_runtime": {"instance_id": "seed", "runtime_id": "A" * 64}},
            hashlib.sha256(b"seed\0host-a").hexdigest(),
        ),
    ],
)
def test_resolve_runtime_id_uses_bounded_identity_and_existing_fallback(monkeypatch, tmp_path, identity, expected):
    calls = []

    def read_json_limited(path, *, max_bytes, record_type):
        calls.append((path, max_bytes, record_type))
        return identity

    monkeypatch.setattr(transport, "read_json_limited", read_json_limited)
    monkeypatch.setattr(transport, "host_instance_id", lambda: "host-a")

    assert transport._resolve_runtime_id(tmp_path) == expected
    assert calls == [(tmp_path / "identity.json", 4096, "machine_runtime_identity")]


@pytest.mark.parametrize("record", [{}, {"machine_runtime": {}}, {"machine_runtime": {"instance_id": ""}}])
def test_resolve_runtime_id_rejects_invalid_seed(monkeypatch, tmp_path, record):
    monkeypatch.setattr(transport, "read_json_limited", lambda *_args, **_kwargs: record)

    with pytest.raises(transport.ProjectIOProtocolError, match="machine runtime identity is invalid"):
        transport._resolve_runtime_id(tmp_path)


def test_executor_and_worker_share_support_error_class():
    assert project_io_executor.ProjectIOProtocolError is transport.ProjectIOProtocolError
    assert project_io_worker.ProjectIOProtocolError is transport.ProjectIOProtocolError
