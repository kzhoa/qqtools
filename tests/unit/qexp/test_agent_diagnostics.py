from __future__ import annotations

import os
from pathlib import Path

import pytest

from qqtools.plugins.qexp.agent.config import load_agent_config, set_agent_config
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.diagnostics import (
    DEFAULT_LOG_MAX_BYTES,
    MAX_LOG_MAX_BYTES,
    MIN_LOG_MAX_BYTES,
    AgentLogService,
    diagnostics_status,
    format_log_size,
    parse_log_size,
    prepare_agent_diagnostics,
    read_diagnostic_record,
    reconcile_agent_diagnostics,
    summarize_diagnostic_record,
    update_diagnostic_evidence,
)
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json


@pytest.fixture
def diagnostic_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MachineRuntime:
    log_root = tmp_path / "tmp"
    log_root.mkdir()
    monkeypatch.setattr("qqtools.plugins.qexp.agent.diagnostics._TMP_ROOT", log_root)
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    initialize_machine(runtime, "gpu-1")
    return runtime


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (str(MIN_LOG_MAX_BYTES), MIN_LOG_MAX_BYTES),
        ("64KiB", 64 * 1024),
        ("10MiB", 10 * 1024 * 1024),
        ("1GiB", 1024 * 1024 * 1024),
        (MAX_LOG_MAX_BYTES, MAX_LOG_MAX_BYTES),
    ],
)
def test_log_size_parser_accepts_exact_bounded_binary_sizes(value: str | int, expected: int) -> None:
    assert parse_log_size(value) == expected


@pytest.mark.parametrize(
    "value",
    [True, 0, -1, "", "1", "+65536", "64KB", "0.5MiB", "64 KiB", "1025MiB", "1TiB"],
)
def test_log_size_parser_rejects_ambiguous_or_out_of_range_values(value: object) -> None:
    with pytest.raises(ValueError):
        parse_log_size(value)  # type: ignore[arg-type]


def test_log_size_formatter_keeps_human_and_exact_byte_values() -> None:
    assert format_log_size(10 * 1024 * 1024) == "10 MiB (10485760 bytes)"
    assert format_log_size(MIN_LOG_MAX_BYTES) == "64 KiB (65536 bytes)"


def test_old_agent_config_defaults_and_every_writer_preserves_log_size(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    initialize_machine(runtime, "gpu-1")
    atomic_replace(
        runtime.paths["global_config"],
        {
            "agent_config": {
                "version": 1,
                "name": "gpu-1",
                "agent_mode": "daemon",
                "revision": 2,
                "provenance": "configured",
            }
        },
    )

    assert load_agent_config(runtime).log_max_bytes == DEFAULT_LOG_MAX_BYTES
    set_agent_config(runtime, name="gpu-renamed")

    stored = read_json(runtime.paths["global_config"])["agent_config"]
    assert stored["name"] == "gpu-renamed"
    assert stored["log_max_bytes"] == DEFAULT_LOG_MAX_BYTES


def test_invalid_agent_log_size_leaves_durable_config_unchanged(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    initialize_machine(runtime, "gpu-1")
    before = runtime.paths["global_config"].read_bytes()

    with pytest.raises(ValueError):
        set_agent_config(runtime, log_max_bytes=MIN_LOG_MAX_BYTES - 1)

    assert runtime.paths["global_config"].read_bytes() == before


def test_prepared_log_is_private_named_and_runtime_isolated(diagnostic_runtime: MachineRuntime) -> None:
    first = prepare_agent_diagnostics(
        diagnostic_runtime,
        instance_id="instance-a",
        capture_mode="detached",
        log_max_bytes=MIN_LOG_MAX_BYTES,
    )
    second = prepare_agent_diagnostics(
        diagnostic_runtime,
        instance_id="instance-b",
        capture_mode="detached",
        log_max_bytes=MIN_LOG_MAX_BYTES,
    )
    try:
        assert first.log_path.name == "agent.log"
        assert first.log_path != second.log_path
        assert first.log_path.is_file()
        assert first.log_path.stat().st_mode & 0o777 == 0o600
        assert first.log_path.parent.stat().st_mode & 0o777 == 0o700
        assert first.startup_sequence < second.startup_sequence
    finally:
        first.close()
        second.close()


def test_writer_merge_is_order_independent_and_specific_agent_failure_wins(
    diagnostic_runtime: MachineRuntime,
) -> None:
    summaries = []
    for index, order in enumerate((("launcher", "agent"), ("agent", "launcher"))):
        prepared = prepare_agent_diagnostics(
            diagnostic_runtime,
            instance_id=f"merge-{index}",
            capture_mode="detached",
            log_max_bytes=MIN_LOG_MAX_BYTES,
        )
        prepared.close()
        evidence = {
            "launcher": {"startup_outcome": "failed", "wait_status": 1},
            "agent": {
                "phase": "stopped",
                "admitted": True,
                "stop_reason": "unhandled_exception",
                "primary_exception": {"exception_type": "RuntimeError", "line": 42},
                "cleanup_outcome": "succeeded",
            },
        }
        for writer in order:
            assert update_diagnostic_evidence(
                diagnostic_runtime,
                instance_id=prepared.instance_id,
                startup_sequence=prepared.startup_sequence,
                writer=writer,
                revision=1,
                evidence=evidence[writer],
            )
        record = read_diagnostic_record(diagnostic_runtime, prepared.instance_id)
        assert record is not None
        summaries.append(summarize_diagnostic_record(record))

    assert [item["reason"] for item in summaries] == ["unhandled_exception", "unhandled_exception"]
    assert all(item["wait_status"] == 1 for item in summaries)
    assert all(item["source"] == "agent" for item in summaries)


def test_writer_revision_is_monotonic_and_retry_is_idempotent(diagnostic_runtime: MachineRuntime) -> None:
    prepared = prepare_agent_diagnostics(
        diagnostic_runtime,
        instance_id="monotonic",
        capture_mode="detached",
        log_max_bytes=MIN_LOG_MAX_BYTES,
    )
    prepared.close()
    arguments = {
        "instance_id": prepared.instance_id,
        "startup_sequence": prepared.startup_sequence,
        "writer": "launcher",
    }
    assert update_diagnostic_evidence(
        diagnostic_runtime, revision=2, evidence={"startup_outcome": "started"}, **arguments
    )
    assert update_diagnostic_evidence(
        diagnostic_runtime, revision=2, evidence={"startup_outcome": "failed"}, **arguments
    )
    assert not update_diagnostic_evidence(
        diagnostic_runtime, revision=1, evidence={"startup_outcome": "failed"}, **arguments
    )

    record = read_diagnostic_record(diagnostic_runtime, prepared.instance_id)
    assert record is not None
    assert record["writers"]["launcher"]["startup_outcome"] == "started"


def test_capture_recovery_clears_the_previous_error(diagnostic_runtime: MachineRuntime) -> None:
    prepared = prepare_agent_diagnostics(
        diagnostic_runtime,
        instance_id="capture-recovery",
        capture_mode="detached",
        log_max_bytes=MIN_LOG_MAX_BYTES,
    )
    prepared.close()
    arguments = {
        "instance_id": prepared.instance_id,
        "startup_sequence": prepared.startup_sequence,
        "writer": "agent",
    }
    assert update_diagnostic_evidence(
        diagnostic_runtime,
        revision=1,
        evidence={"capture_health": "degraded", "capture_error": "capture_io_OSError"},
        **arguments,
    )
    assert update_diagnostic_evidence(
        diagnostic_runtime,
        revision=2,
        evidence={"capture_health": "healthy", "capture_error": None},
        **arguments,
    )

    record = read_diagnostic_record(diagnostic_runtime, prepared.instance_id)
    assert record is not None
    assert record["capture_health"] == "healthy"
    assert "capture_error" not in record
    assert record["writers"]["agent"]["capture_error"] is None


def test_failed_reconciliation_write_is_degraded_and_retried(
    diagnostic_runtime: MachineRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = prepare_agent_diagnostics(
        diagnostic_runtime,
        instance_id="reconcile-retry",
        capture_mode="detached",
        log_max_bytes=MIN_LOG_MAX_BYTES,
    )
    prepared.close()
    assert update_diagnostic_evidence(
        diagnostic_runtime,
        instance_id=prepared.instance_id,
        startup_sequence=prepared.startup_sequence,
        writer="agent",
        revision=1,
        evidence={"phase": "active", "admitted": True, "pid": 2**30, "pid_start_time_ticks": 1},
    )

    from qqtools.plugins.qexp.agent import diagnostics

    original_update = diagnostics._update_evidence_internal
    monkeypatch.setattr(diagnostics, "_update_evidence_internal", lambda *_args, **_kwargs: False)
    failed = reconcile_agent_diagnostics(diagnostic_runtime)
    assert failed["reconciled"] is False
    assert failed["update_failures"] == 1
    assert not diagnostic_runtime.paths["diagnostic_last_exit"].exists()

    monkeypatch.setattr(diagnostics, "_update_evidence_internal", original_update)
    recovered = reconcile_agent_diagnostics(diagnostic_runtime)
    assert recovered["reconciled"] is True
    assert recovered["update_failures"] == 0
    record = read_diagnostic_record(diagnostic_runtime, prepared.instance_id)
    assert record is not None
    assert record["writers"]["observer"]["reason"] == "verified_absence"


def test_foreign_host_record_never_uses_local_pid_liveness(
    diagnostic_runtime: MachineRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = prepare_agent_diagnostics(
        diagnostic_runtime,
        instance_id="foreign-host",
        capture_mode="detached",
        log_max_bytes=MIN_LOG_MAX_BYTES,
    )
    prepared.close()
    assert update_diagnostic_evidence(
        diagnostic_runtime,
        instance_id=prepared.instance_id,
        startup_sequence=prepared.startup_sequence,
        writer="agent",
        revision=1,
        evidence={"phase": "active", "admitted": True, "pid": 2**30, "pid_start_time_ticks": 1},
    )
    monkeypatch.setattr("qqtools.plugins.qexp.agent.diagnostics.host_instance_id", lambda: "other-host")
    monkeypatch.setattr(
        "qqtools.plugins.qexp.agent.diagnostics.read_process_identity",
        lambda _pid: (_ for _ in ()).throw(AssertionError("foreign PID must not be inspected locally")),
    )

    status = diagnostics_status(
        diagnostic_runtime,
        configured_log_max_bytes=DEFAULT_LOG_MAX_BYTES,
        active_identity=None,
    )
    assert status["summary"]["reason"] == "unknown"
    assert status["summary_synthetic"] is False

    result = reconcile_agent_diagnostics(diagnostic_runtime)

    assert result["reconciled"] is True
    assert result["liveness_unknown"] == 1
    record = read_diagnostic_record(diagnostic_runtime, prepared.instance_id)
    assert record is not None
    assert record["writers"]["observer"]["reason"] == "host_identity_mismatch"
    assert "abnormal_exit_unknown" not in record["writers"]["observer"]


def test_successor_reconciles_proven_absence_without_inventing_wait_evidence(
    diagnostic_runtime: MachineRuntime,
) -> None:
    previous = prepare_agent_diagnostics(
        diagnostic_runtime,
        instance_id="missing-agent",
        capture_mode="detached",
        log_max_bytes=MIN_LOG_MAX_BYTES,
    )
    previous.close()
    update_diagnostic_evidence(
        diagnostic_runtime,
        instance_id=previous.instance_id,
        startup_sequence=previous.startup_sequence,
        writer="agent",
        revision=1,
        evidence={
            "phase": "active",
            "admitted": True,
            "pid": 2**30,
            "pid_start_time_ticks": 1,
            "stop_reason": None,
        },
    )

    successor = prepare_agent_diagnostics(
        diagnostic_runtime,
        instance_id="successor",
        capture_mode="detached",
        log_max_bytes=MIN_LOG_MAX_BYTES,
    )
    successor.close()

    record = read_diagnostic_record(diagnostic_runtime, previous.instance_id)
    assert record is not None
    summary = summarize_diagnostic_record(record)
    assert summary["reason"] == "abnormal_exit_unknown"
    assert summary["source"] == "observer"
    assert summary.get("exit_code") is None
    assert summary.get("signal") is None
    assert summary.get("exited_at") is None


def test_terminal_and_unresolved_history_are_bounded_separately(diagnostic_runtime: MachineRuntime) -> None:
    for index in range(40):
        prepared = prepare_agent_diagnostics(
            diagnostic_runtime,
            instance_id=f"terminal-{index:02d}",
            capture_mode="foreground_managed_only",
            log_max_bytes=MIN_LOG_MAX_BYTES,
        )
        prepared.close()
        update_diagnostic_evidence(
            diagnostic_runtime,
            instance_id=prepared.instance_id,
            startup_sequence=prepared.startup_sequence,
            writer="agent",
            revision=1,
            evidence={
                "phase": "stopped",
                "admitted": True,
                "stop_reason": "idle",
                "cleanup_outcome": "succeeded",
            },
        )

    current = prepare_agent_diagnostics(
        diagnostic_runtime,
        instance_id="current",
        capture_mode="foreground_managed_only",
        log_max_bytes=MIN_LOG_MAX_BYTES,
    )
    current.close()

    records = list(diagnostic_runtime.paths["diagnostic_instances"].glob("*.json"))
    assert len(records) == 33
    last_exit = read_json(diagnostic_runtime.paths["diagnostic_last_exit"])
    assert last_exit["summary"]["instance_id"] == "terminal-39"
    metadata = read_json(diagnostic_runtime.paths["diagnostic_metadata"])
    assert metadata["coverage"]["evicted_count"] == 8


def test_unresolved_history_is_bounded_and_late_writer_cannot_recreate_eviction(
    diagnostic_runtime: MachineRuntime,
) -> None:
    first_sequence = None
    for index in range(40):
        prepared = prepare_agent_diagnostics(
            diagnostic_runtime,
            instance_id=f"unresolved-{index:02d}",
            capture_mode="detached",
            log_max_bytes=MIN_LOG_MAX_BYTES,
        )
        prepared.close()
        if first_sequence is None:
            first_sequence = prepared.startup_sequence

    records = list(diagnostic_runtime.paths["diagnostic_instances"].glob("*.json"))
    assert len(records) == 33
    assert not (diagnostic_runtime.paths["diagnostic_instances"] / "unresolved-00.json").exists()
    assert not update_diagnostic_evidence(
        diagnostic_runtime,
        instance_id="unresolved-00",
        startup_sequence=first_sequence,
        writer="launcher",
        revision=1,
        evidence={"startup_outcome": "failed", "wait_status": 1},
    )
    assert not (diagnostic_runtime.paths["diagnostic_instances"] / "unresolved-00.json").exists()
    metadata = read_json(diagnostic_runtime.paths["diagnostic_metadata"])
    assert metadata["coverage"]["evicted_count"] == 7


def test_status_checks_path_availability_without_reading_log_contents(
    diagnostic_runtime: MachineRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = prepare_agent_diagnostics(
        diagnostic_runtime,
        instance_id="status-instance",
        capture_mode="detached",
        log_max_bytes=MIN_LOG_MAX_BYTES,
    )
    prepared.close()
    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        if path == prepared.log_path:
            raise AssertionError("status opened diagnostic log contents")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)

    result = diagnostics_status(
        diagnostic_runtime,
        configured_log_max_bytes=DEFAULT_LOG_MAX_BYTES,
        active_identity=None,
    )

    assert result["instance_id"] == "status-instance"
    assert result["log_available"] is True
    assert result["configured_log_max_bytes"] == DEFAULT_LOG_MAX_BYTES
    assert result["effective_log_max_bytes"] == MIN_LOG_MAX_BYTES


def test_foreground_managed_log_rotates_without_rebinding_terminal_streams(
    diagnostic_runtime: MachineRuntime,
) -> None:
    prepared = prepare_agent_diagnostics(
        diagnostic_runtime,
        instance_id="rotation",
        capture_mode="foreground_managed_only",
        log_max_bytes=MIN_LOG_MAX_BYTES,
    )
    prepared.close()
    stdout_stat = os.fstat(1)
    service = AgentLogService(
        prepared.log_path,
        max_bytes=MIN_LOG_MAX_BYTES,
        capture_mode="foreground_managed_only",
    )
    service.start()
    service.write("x" * (MIN_LOG_MAX_BYTES + 1024))
    service.stop()

    archives = list(prepared.log_path.parent.glob("_agent_*.log"))
    assert len(archives) == 1
    assert archives[0].stat().st_size >= MIN_LOG_MAX_BYTES
    assert prepared.log_path.exists()
    assert (os.fstat(1).st_dev, os.fstat(1).st_ino) == (stdout_stat.st_dev, stdout_stat.st_ino)


def test_log_service_recovers_after_transient_open_failure(
    diagnostic_runtime: MachineRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = prepare_agent_diagnostics(
        diagnostic_runtime,
        instance_id="open-recovery",
        capture_mode="foreground_managed_only",
        log_max_bytes=MIN_LOG_MAX_BYTES,
    )
    prepared.close()
    from qqtools.plugins.qexp.agent import diagnostics

    original_open = diagnostics._open_log_file
    attempts = 0

    def fail_once(path: Path):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("transient open failure")
        return original_open(path)

    monkeypatch.setattr(diagnostics, "_open_log_file", fail_once)
    health: list[tuple[str, str | None]] = []
    service = AgentLogService(
        prepared.log_path,
        max_bytes=MIN_LOG_MAX_BYTES,
        capture_mode="foreground_managed_only",
        health_callback=lambda state, error: health.append((state, error)),
    )
    service.start()
    service._next_retry_at = 0.0
    service._check_once()
    assert service.write("recovered\n")
    assert service.stop()

    assert health[0][0] == "degraded"
    assert health[-1] == ("healthy", None)
    assert b"recovered" in prepared.log_path.read_bytes()
