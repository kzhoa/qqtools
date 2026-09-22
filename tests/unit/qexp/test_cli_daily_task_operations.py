from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path

import pytest

from qqtools.plugins.qexp.commands import logs
from qqtools.plugins.qexp.commands import wait as wait_commands
from qqtools.plugins.qexp.config_types import RootConfig
from qqtools.plugins.qexp.runtime.observation import api as observation_api
from qqtools.plugins.qexp.runtime.records import SCHEMA_VERSION, AttemptRecord, TaskRecord


def _cfg(tmp_path: Path) -> RootConfig:
    return RootConfig(tmp_path / ".qexp", tmp_path, "gpu-1", tmp_path / "runtime")


def _task(
    task_id: str = "task-1",
    *,
    name: str | None = "experiment",
    projection: str = "queued",
    next_attempt_number: int = 1,
    current_attempt_number: int | None = None,
    current_attempt_id: str | None = None,
    reason: str | None = None,
) -> TaskRecord:
    return TaskRecord.from_dict(
        {
            "meta": {
                "schema_version": SCHEMA_VERSION,
                "revision": 1,
                "created_at": "2026-09-21T00:00:00Z",
                "updated_at": "2026-09-21T00:00:00Z",
                "updated_by": {"actor_type": "test", "machine_name": "gpu-1", "process_id": "0"},
            },
            "task": {
                "task_id": task_id,
                "group_name": None,
                "group_membership_sequence": None,
                "submission_operation_id": None,
                "name": name,
                "depends_on_task_ids": [],
                "ready_generation": 0,
                "spec": {
                    "command": ["true"],
                    "working_directory": "/tmp",
                    "requested_gpus": 1,
                    "requested_cpus": None,
                },
                "placement_policy": {
                    "home_machine": "gpu-1",
                    "sharing_mode": "private",
                    "fallback_machines": [],
                },
                "placement_runtime": {"queue_scope": "home", "offer_clock_evidence": None},
                "state": {"projection": projection, "reason": reason},
                "control": {"cleanup_operation_id": None, "cleanup_state": None},
                "attempt_control": {
                    "next_attempt_number": next_attempt_number,
                    "current_attempt_number": current_attempt_number,
                    "current_attempt_id": current_attempt_id,
                },
                "claim_control": {"fencing_epoch": 0, "active_claim": None},
            },
        }
    )


def _attempt(task_id: str, number: int, attempt_id: str, phase: str, exit_code: int | None = None) -> AttemptRecord:
    task = _task(task_id, next_attempt_number=number)
    value = AttemptRecord.claimed(
        task,
        "gpu-1",
        [0],
        f"reservation-{number}",
        number,
        authority_mode="holder_bound",
        clock_evidence=None,
        attempt_id=attempt_id,
    )
    value.phase = phase
    value.result["exit_code"] = exit_code
    return value


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), (0, 0.0), ("0", 0.0), ("1.5s", 1.5), ("2m", 120.0), ("0.5h", 1800.0)],
)
def test_wait_timeout_parser_accepts_only_finite_single_unit_durations(value, expected) -> None:
    assert wait_commands.parse_wait_timeout(value) == expected


@pytest.mark.parametrize("value", [True, -1, "-1", "NaN", "inf", "1m30s", "", "  ", "1d"])
def test_wait_timeout_parser_rejects_invalid_values(value) -> None:
    with pytest.raises(ValueError):
        wait_commands.parse_wait_timeout(value)


def test_make_wait_result_keeps_schema_for_pre_resolution_errors() -> None:
    result = wait_commands.make_wait_result(
        outcome="invalid_input",
        reason="invalid_input",
        error={"code": "invalid_input", "message": "bad timeout"},
    )

    assert set(result) == {
        "schema_version",
        "task_id",
        "project",
        "selected_attempt_number",
        "selected_attempt_id",
        "outcome",
        "reason",
        "task_exit_code",
        "error",
    }
    assert result["schema_version"] == 1
    assert result["task_id"] is None
    assert result["project"] is None
    assert result["selected_attempt_number"] is None
    assert result["selected_attempt_id"] is None
    assert result["task_exit_code"] is None
    assert result["error"] == {"code": "invalid_input", "message": "bad timeout"}

    timeout = wait_commands.make_wait_result(
        project=Path("/project"),
        task_id="task-1",
        selected_attempt_number=2,
        selected_attempt_id="attempt-2",
        outcome="timeout",
        reason="timeout",
        error={"code": "ignored", "message": "lifecycle errors have no error object"},
    )
    assert timeout["project"] == "/project"
    assert timeout["selected_attempt_number"] == 2
    assert timeout["selected_attempt_id"] == "attempt-2"
    assert timeout["error"] is None


def test_wait_pins_queued_attempt_number_and_does_not_follow_retry(monkeypatch, tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    snapshots = iter(
        [
            _task(next_attempt_number=2),
            _task(
                projection="running", next_attempt_number=3, current_attempt_number=2, current_attempt_id="attempt-2"
            ),
            _task(projection="queued", next_attempt_number=3),
        ]
    )
    attempts = iter(
        [
            _attempt("task-1", 2, "attempt-2", "running"),
            _attempt("task-1", 2, "attempt-2", "failed", exit_code=17),
        ]
    )
    monkeypatch.setattr(wait_commands, "load_task", lambda _cfg, _task_id: next(snapshots))
    monkeypatch.setattr(wait_commands, "read_json", lambda _path: next(attempts).to_dict())
    ticks = iter([0.0, 0.0, 0.1, 0.2, 0.3])

    result, exit_code = wait_commands.wait_for_task(
        cfg,
        "task-1",
        timeout_seconds=10,
        poll_interval_seconds=0.01,
        monotonic=lambda: next(ticks),
        sleep=lambda _seconds: None,
    )

    assert exit_code == 1
    assert result["outcome"] == "failed"
    assert result["selected_attempt_number"] == 2
    assert result["selected_attempt_id"] == "attempt-2"
    assert result["task_exit_code"] == 17


def test_wait_zero_timeout_performs_one_observation(monkeypatch, tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    reads = 0

    def load(_cfg, _task_id):
        nonlocal reads
        reads += 1
        return _task()

    monkeypatch.setattr(wait_commands, "load_task", load)
    result, exit_code = wait_commands.wait_for_task(cfg, "task-1", timeout_seconds=0, sleep=lambda _: None)

    assert reads == 1
    assert exit_code == 4
    assert result["outcome"] == "timeout"
    assert result["selected_attempt_number"] == 1


def test_finite_log_tail_handles_zero_and_last_lines(monkeypatch, tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    path = tmp_path / "attempt.log"
    path.write_bytes(b"first\nsecond\nthird\n")
    monkeypatch.setattr(logs, "get_log_path", lambda _cfg, _task_id: path)

    assert logs.read_logs(cfg, "task-1", tail_lines=0) == ""
    assert logs.read_logs(cfg, "task-1", tail_lines=2) == "second\nthird\n"
    assert logs.read_logs(cfg, "task-1") == "first\nsecond\nthird\n"
    with pytest.raises(ValueError):
        logs.read_logs(cfg, "task-1", tail_lines=-1)


def test_finite_log_writer_preserves_invalid_utf8_for_binary_stdout(monkeypatch, tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    path = tmp_path / "attempt.log"
    payload = b"before\xffafter\n"
    path.write_bytes(payload)
    monkeypatch.setattr(logs, "get_log_path", lambda _cfg, _task_id: path)
    stdout = BytesIO()

    logs.write_logs(cfg, "task-1", stdout=stdout)

    assert stdout.getvalue() == payload


def test_name_cursor_v2_binds_exact_filter_and_v1_stays_unchanged() -> None:
    v1 = observation_api._encode_cursor(project="project-1", phase=None, group=None, generation="0" * 32, last="task-1")
    v2 = observation_api._encode_cursor(
        project="project-1", phase=None, group=None, name="same", generation="0" * 32, last="task-1"
    )
    v1_payload = json.loads(base64.urlsafe_b64decode(v1 + "=" * (-len(v1) % 4)))
    v2_payload = json.loads(base64.urlsafe_b64decode(v2 + "=" * (-len(v2) % 4)))

    assert v1_payload["v"] == 1
    assert "name" not in v1_payload
    assert v2_payload["v"] == 2
    assert v2_payload["name"] == "same"
    with pytest.raises(observation_api.ObservationError, match="filters"):
        observation_api._decode_cursor(v2, project="project-1", phase=None, group=None, name="different")
