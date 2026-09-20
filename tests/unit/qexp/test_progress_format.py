import ast
import copy
import inspect
from datetime import datetime, timedelta, timezone

import pytest

from qqtools.plugins.qexp import progress_format
from qqtools.plugins.qexp.progress_format import format_progress_details


def _available(**payload_overrides):
    payload = {
        "stage": "train",
        "current": 2,
        "total": 8,
        "unit": "step",
        "message": "epoch 1",
    }
    payload.update(payload_overrides)
    return {
        "status": "available",
        "observation_state": "available",
        "reason": None,
        "protocol_version": 1,
        "task_id": "task-1",
        "attempt_id": "attempt-1",
        "attempt_number": 1,
        "machine_name": "gpu-1",
        "launch_id": None,
        "wrapper_pid": None,
        "wrapper_start_time_ticks": None,
        "registration_generation": "generation-1",
        "fencing_token": 1,
        "source_update_id": "update-1",
        "sequence": 1,
        "reported_at": "2026-09-18T16:00:25Z",
        "advanced_at": "2026-09-18T15:00:00Z",
        "progress": payload,
    }


@pytest.mark.parametrize(
    ("observation", "expected"),
    [
        (None, ("unavailable", "unavailable")),
        (
            {"status": "unavailable", "observation_state": "pending", "reason": "not_started"},
            ("pending", "pending (not started)"),
        ),
        (
            {"status": "unavailable", "observation_state": "no_report", "reason": "no_snapshot"},
            ("no_report", "unavailable (no report yet)"),
        ),
        (
            {"status": "unavailable", "observation_state": "unavailable", "reason": "cleanup"},
            ("unavailable", "unavailable (cleanup in progress)"),
        ),
        (
            {"status": "unavailable", "observation_state": "unavailable", "reason": "read_failed"},
            ("unavailable", "unavailable (read failed)"),
        ),
        (
            {"status": "unavailable", "observation_state": "unavailable", "reason": "invalid_snapshot"},
            ("unavailable", "unavailable (invalid snapshot)"),
        ),
        (
            {"status": "unavailable", "observation_state": "unavailable", "reason": "identity_mismatch"},
            ("unavailable", "unavailable (identity mismatch)"),
        ),
        (
            {"status": "unavailable", "observation_state": "unavailable", "reason": "unknown"},
            ("unavailable", "unavailable (unknown)"),
        ),
        (
            {"status": "unavailable", "observation_state": "unavailable", "reason": "unexpected"},
            ("unavailable", "unavailable"),
        ),
    ],
)
def test_missing_and_unavailable_observations_have_bounded_human_text(observation, expected):
    rows = format_progress_details(observation, now=datetime(2026, 9, 18, 16, 1, tzinfo=timezone.utc))

    assert rows == (("Progress status", expected[0]), ("Progress", expected[1]))


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"current": None, "total": None, "unit": None}, "unknown"),
        ({"current": None, "total": 8, "unit": "step"}, "unknown/8 step"),
        ({"current": 0, "total": 0, "unit": "step"}, "0/0 step"),
        ({"current": 2, "total": 8, "unit": "step"}, "2/8 step (25.0%)"),
        ({"current": 1, "total": 3, "unit": ""}, "1/3 (33.3%)"),
    ],
)
def test_available_progress_count_unit_and_percentage(payload, expected):
    rows = format_progress_details(
        _available(**payload),
        now=datetime(2026, 9, 18, 16, 1, tzinfo=timezone.utc),
    )

    assert [label for label, _value in rows] == [
        "Progress status",
        "Stage",
        "Progress",
        "Message",
        "Progress reported",
        "Progress advanced",
    ]
    assert dict(rows)["Progress"] == expected


@pytest.mark.parametrize("message", [None, ""])
def test_available_progress_always_retains_message_row(message):
    rows = format_progress_details(
        _available(message=message),
        now=datetime(2026, 9, 18, 16, 1, tzinfo=timezone.utc),
    )

    assert ("Message", message) in rows


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "0s ago"),
        (59.999, "59s ago"),
        (60, "1m ago"),
        (3599.999, "59m ago"),
        (3600, "1h ago"),
    ],
)
def test_relative_age_boundaries(seconds, expected):
    now = datetime(2026, 9, 18, 16, 0, tzinfo=timezone.utc)
    observation = _available()
    stamp = now - timedelta(seconds=seconds)
    observation["reported_at"] = stamp.isoformat()

    rows = dict(format_progress_details(observation, now=now))

    assert rows["Progress reported"].endswith(f"({expected})")


def test_invalid_naive_future_and_offset_timestamps():
    now = datetime(2026, 9, 19, 1, 0, tzinfo=timezone(timedelta(hours=9)))
    observation = _available()
    observation["reported_at"] = "invalid"
    observation["advanced_at"] = "2026-09-18T16:00:30+00:00"
    rows = dict(format_progress_details(observation, now=now))
    assert rows["Progress reported"] == "unknown"
    assert rows["Progress advanced"] == "2026-09-18 16:00:30 UTC (unknown (clock difference))"

    observation["reported_at"] = "2026-09-18T16:00:00"
    observation["advanced_at"] = "2026-09-18T17:00:00+01:00"
    rows = dict(format_progress_details(observation, now=now))
    assert rows["Progress reported"] == "unknown"
    assert rows["Progress advanced"] == "2026-09-18 16:00:00 UTC (0s ago)"


def test_explicit_now_must_be_aware_and_never_reads_the_clock(monkeypatch):
    class NoClockDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            raise AssertionError("explicit formatting time must not read the clock")

    monkeypatch.setattr(progress_format, "datetime", NoClockDatetime)
    aware = NoClockDatetime(2026, 9, 18, 16, 1, tzinfo=timezone.utc)

    assert dict(format_progress_details(_available(), now=aware))["Progress status"] == "available"
    with pytest.raises(ValueError, match="timezone"):
        format_progress_details(_available(), now=NoClockDatetime(2026, 9, 18, 16, 1))


def test_omitted_now_is_captured_once_for_both_timestamps(monkeypatch):
    class CountingDatetime(datetime):
        calls = 0

        @classmethod
        def now(cls, tz=None):
            cls.calls += 1
            value = cls(2026, 9, 18, 16, 1, tzinfo=timezone.utc)
            return value if tz is None else value.astimezone(tz)

    observation = _available()
    observation["reported_at"] = "2026-09-18T16:00:00Z"
    observation["advanced_at"] = "2026-09-18T16:00:00Z"
    monkeypatch.setattr(progress_format, "datetime", CountingDatetime)

    rows = dict(format_progress_details(observation))

    assert CountingDatetime.calls == 1
    assert rows["Progress reported"].endswith("(1m ago)")
    assert rows["Progress advanced"].endswith("(1m ago)")


def test_formatting_does_not_mutate_the_observation():
    observation = _available(unit="", message="")
    before = copy.deepcopy(observation)

    format_progress_details(observation, now=datetime(2026, 9, 18, 16, 1, tzinfo=timezone.utc))

    assert observation == before


def test_importing_progress_presentation_does_not_load_runtime_services():
    tree = ast.parse(inspect.getsource(progress_format))
    imports = {(node.level, node.module) for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    type_checking_guards = [
        node
        for node in tree.body
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"
    ]
    runtime_type_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.level, node.module) == (1, "runtime.progress_types")
    ]

    assert imports == {
        (0, "__future__"),
        (0, "datetime"),
        (0, "typing"),
        (1, "runtime.progress_types"),
    }
    assert len(type_checking_guards) == 1
    assert len(runtime_type_imports) == 1
    assert runtime_type_imports[0] in ast.walk(type_checking_guards[0])
    assert not any(isinstance(node, ast.Import) for node in ast.walk(tree))
