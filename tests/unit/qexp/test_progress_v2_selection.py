"""Whole-candidate selection and truthful metric presentation."""

from datetime import datetime, timezone

from qqtools.plugins.qexp.observer import _selected_progress_version
from qqtools.plugins.qexp.progress_format import format_progress_compact, format_progress_details


def _candidate(version: int, reported_at: str, attempt_id: str = "attempt-1") -> dict:
    progress = {"stage": "train", "current": 2, "total": 10, "unit": "step", "message": "live"}
    if version == 2:
        progress.update(
            metrics={"loss": 0.25, "lr": 0.001},
            completeness={"complete": True, "omitted_metrics": 0, "reasons": []},
        )
    return {
        "status": "available",
        "observation_state": "available",
        "reason": None,
        "protocol_version": version,
        "attempt_id": attempt_id,
        "attempt_number": 1,
        "reported_at": reported_at,
        "advanced_at": reported_at,
        "progress": progress,
    }


def test_newer_v1_selects_whole_base_without_old_v2_metrics():
    newer = _candidate(1, "2026-09-23T00:00:20Z")
    older = _candidate(2, "2026-09-23T00:00:10Z")
    selected = _selected_progress_version(newer, older, attempt_id="attempt-1", attempt_number=1)
    assert selected == 1
    fields = dict(
        format_progress_details(
            newer,
            progress_version=selected,
            include_metrics=True,
            now=datetime(2026, 9, 23, 0, 0, 30, tzinfo=timezone.utc),
        )
    )
    assert fields["Metrics"] == "unavailable (v1 progress selected)"
    assert "Metric loss" not in fields


def test_v2_wins_timestamp_tie_but_mismatched_attempt_is_never_candidate():
    v1 = _candidate(1, "2026-09-23T00:00:20Z")
    v2 = _candidate(2, "2026-09-23T00:00:20Z")
    assert _selected_progress_version(v1, v2, attempt_id="attempt-1", attempt_number=1) == 2
    assert _selected_progress_version(v1, {**v2, "attempt_id": "attempt-2"}, attempt_id="attempt-1") == 1


def test_v2_details_and_compact_show_same_current_values_and_completeness():
    v2 = _candidate(2, "2026-09-23T00:00:20Z")
    v2["progress"]["completeness"] = {
        "complete": False,
        "omitted_metrics": 1,
        "reasons": ["metric_limit"],
    }
    now = datetime(2026, 9, 23, 0, 0, 30, tzinfo=timezone.utc)
    details = dict(format_progress_details(v2, progress_version=2, include_metrics=True, now=now))
    compact = dict(format_progress_compact(v2, progress_version=2, now=now))
    assert details["Metric loss"] == 0.25
    assert details["Metric lr"] == 0.001
    assert details["Omitted metrics"] == 1
    assert details["Completeness reasons"] == "metric_limit"
    assert "loss=0.25" in compact["Metrics"]
    assert "incomplete" in compact["Metrics"]
