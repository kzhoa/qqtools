"""Cross-version observations select whole facts with strict same-source proof."""

from copy import deepcopy

import pytest

from qqtools.plugins.qexp.progress_selection import select_progress_protocol, selected_progress_observation


def candidate(version, second, *, update_id="same"):
    activity = {"stage": "validation", "current": 2, "total": 10, "unit": "batch", "message": None}
    extras = {"metrics": {"loss": 0.5}, "completeness": {"complete": True, "omitted_metrics": 0, "reasons": []}}
    data = activity if version == 1 else {**activity, **extras}
    if version == 3:
        data = {
            "activity": activity,
            "overall": {"current": 8, "total": 10, "unit": "step", "label": "Training"},
            **extras,
        }
    return {
        "status": "available",
        "observation_state": "available",
        "reason": None,
        "protocol_version": version,
        "task_id": "task-1",
        "attempt_id": "attempt-1",
        "attempt_number": 1,
        "machine_name": "g1",
        "launch_id": "launch-1",
        "wrapper_pid": 10,
        "wrapper_start_time_ticks": 20,
        "fencing_token": 1,
        "registration_generation": "generation-1",
        "source_update_id": update_id,
        "sequence": 1,
        "reported_at": f"2026-10-03T12:00:{second:02d}Z",
        "advanced_at": "2026-10-03T12:00:00Z",
        "progress": data,
    }


def test_delayed_flat_channel_preserves_same_source_v3_original_timestamp():
    v1, v2, v3 = candidate(1, 3), candidate(2, 2), candidate(3, 1)
    assert select_progress_protocol(v1, v2, v3) == 3
    selected = selected_progress_observation(v1, v2, v3, 3)
    assert selected == v3
    assert selected["reported_at"].endswith("01Z")


def test_newer_distinct_update_replaces_complete_scoped_report():
    v1, v3 = candidate(1, 3, update_id="new"), candidate(3, 1)
    assert select_progress_protocol(v1, None, v3) == 1
    assert "overall" not in selected_progress_observation(v1, None, v3, 1)["progress"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("task_id", "other"),
        ("attempt_id", "other"),
        ("attempt_number", 2),
        ("machine_name", "g2"),
        ("launch_id", "other"),
        ("wrapper_pid", 11),
        ("wrapper_start_time_ticks", 21),
        ("fencing_token", 2),
        ("registration_generation", "other"),
        ("source_update_id", ""),
    ],
)
def test_identity_mismatch_cannot_promote_older_richer_channel(field, value):
    v1, v3 = candidate(1, 3), candidate(3, 1)
    v3[field] = value
    assert select_progress_protocol(v1, None, v3) == 1


def test_activity_disagreement_and_missing_identity_cannot_prove_same_source():
    v1, v3 = candidate(1, 3), candidate(3, 1)
    v3["progress"]["activity"]["message"] = "different"
    assert select_progress_protocol(v1, None, v3) == 1
    v3 = candidate(3, 1)
    del v1["wrapper_pid"]
    del v3["wrapper_pid"]
    assert select_progress_protocol(v1, None, v3) == 1


def test_timestamp_tie_prefers_version_and_authorized_attempt_filter():
    v1, v2, v3 = [candidate(version, 1, update_id=f"update-{version}") for version in (1, 2, 3)]
    assert select_progress_protocol(v1, v2, v3) == 3
    v3["attempt_id"] = "old"
    assert select_progress_protocol(v1, v2, v3, attempt_id="attempt-1", attempt_number=1) == 2


@pytest.mark.parametrize("reason", ["invalid_snapshot", "read_failed", "identity_mismatch"])
def test_valid_base_wins_optional_channel_failure(reason):
    v1 = candidate(1, 1)
    failure = {"status": "unavailable", "observation_state": "unavailable", "reason": reason}
    assert select_progress_protocol(v1, None, failure) == 1
    assert selected_progress_observation(v1, None, failure, 1) == v1


def test_no_candidate_failure_precedence_over_absence():
    no_report = {"status": "unavailable", "observation_state": "no_report", "reason": "no_snapshot"}
    invalid = {"status": "unavailable", "observation_state": "unavailable", "reason": "invalid_snapshot"}
    read_failure = deepcopy(invalid) | {"reason": "read_failed"}
    assert selected_progress_observation(no_report, read_failure, invalid, None)["reason"] == "invalid_snapshot"
