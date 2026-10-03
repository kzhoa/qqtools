"""Human field selection preserves identities, scopes and absence distinctions."""

from datetime import datetime, timezone

import pytest

from qqtools.plugins.qexp.cli.output.task_list import render_task_list
from qqtools.plugins.qexp.task_list_fields import resolve_task_fields


def row(**updates):
    return {
        "task_id": "task-1",
        "name": "run",
        "phase": "running",
        "reason": None,
        "gpus": 2,
        "group": "training",
        "home_machine": "g1",
        "queue_scope": "home",
        "claim_machine": "g1",
        "location": {
            "status": "available",
            "reason": None,
            "machine_name": "g1",
            "assigned_gpus": [2, 0],
            "orphaned": False,
        },
        "reporting_policy": {"state": "enabled", "reason": None},
        "observation_time": "2026-10-03T12:01:00Z",
        **updates,
    }


def observation(current=999, total=1000, *, overall=True, stamp="2026-10-03T12:00:48Z"):
    activity = {"stage": "validation", "current": current, "total": total, "unit": "step", "message": None}
    return {
        "status": "available",
        "observation_state": "available",
        "reason": None,
        "protocol_version": 3,
        "reported_at": stamp,
        "advanced_at": stamp,
        "progress": {
            "activity": activity,
            "overall": {"current": current, "total": total, "unit": "step", "label": "Training"} if overall else None,
            "metrics": {},
            "completeness": {"complete": True, "omitted_metrics": 0, "reasons": []},
        },
    }


def rendered(items, fields, width=None):
    return render_task_list(
        items,
        {
            "fields": fields,
            "terminal_width": width,
            "details_command": "qexp --project '/projects/my run/.qexp' task show TASK_ID --details",
        },
    )


def test_presets_and_task_normalization():
    assert resolve_task_fields(None, None) is None
    assert resolve_task_fields("default", None) is None
    assert resolve_task_fields("progress", None) == (
        "task",
        "state",
        "overall-progress",
        "activity",
        "report-age",
        "location",
    )
    assert resolve_task_fields("placement", None) == ("task", "state", "requested-gpus", "home", "queue", "location")
    assert resolve_task_fields("overall", None) == ("task", "state", "overall-progress", "location")
    assert resolve_task_fields(None, " state , task, activity ") == ("task", "state", "activity")


@pytest.mark.parametrize(
    "fields",
    ["", " ", "state,", ",state", "state,state", "task,task", "current-activity", "reported-progress", "unknown"],
)
def test_bad_fields_rejected(fields):
    with pytest.raises(ValueError):
        resolve_task_fields(None, fields)


def test_conflicts_and_suggestion():
    with pytest.raises(ValueError):
        resolve_task_fields("progress", "state")
    with pytest.raises(ValueError, match="activity"):
        resolve_task_fields(None, "activty")


@pytest.mark.parametrize(
    "current,total,expected",
    [
        (999, 1000, "<100% · 999/1,000 step"),
        (1000, 1000, "100% · 1,000/1,000 step"),
        (1, 8, "13% · 1/8 step"),
        (0, 0, "0/0 step"),
        (3, None, "3 step"),
        (None, 4, "?/4 step"),
    ],
)
def test_counter_scope_and_rounding(current, total, expected):
    item = row(progress_scoped=observation(current, total), selected_progress_protocol_version=3)
    text = rendered([item], ("task", "overall-progress", "activity"))
    assert expected in text
    assert "validation" in text and "Training" in text
    if total in (None, 0) or current is None:
        assert "%" not in text
    assert text.count(expected) == 2


@pytest.mark.parametrize("width", [80, 120, 160, None])
def test_width_preserves_fields_ids_unicode_and_details(width):
    identifier = "task-" + "a" * 90
    item = row(
        task_id=identifier,
        name="界e\u0301" * 40,
        reason="failure " * 20,
        phase="failed",
        progress_scoped=observation(),
        selected_progress_protocol_version=3,
    )
    item["progress_scoped"]["progress"]["activity"]["message"] = "長いe\u0301 " * 30
    text = rendered([item], resolve_task_fields("progress", None), width)
    assert identifier in text
    assert "…" in text
    assert "State" in text and "Overall" in text and "Activity" in text and "Report age" in text and "Location" in text
    assert "failed" in text and "Reason" in text
    assert "--project" in text and "--details" in text
    assert "Report age: since agent acceptance; not a heartbeat." in text
    assert "GPU0,2" in text
    assert not any(line.startswith("\u0301") for line in text.splitlines())


@pytest.mark.parametrize(
    "seconds,age", [(0, "0s"), (59, "59s"), (60, "1m"), (3599, "59m"), (3600, "1h"), (86399, "23h"), (86400, "1d")]
)
def test_age_boundaries(seconds, age):
    stamp = datetime.fromtimestamp(
        datetime(2026, 10, 3, 12, 1, tzinfo=timezone.utc).timestamp() - seconds, timezone.utc
    ).isoformat()
    item = row(progress_scoped=observation(stamp=stamp), selected_progress_protocol_version=3)
    assert age in rendered([item], ("task", "report-age"))


def test_future_timestamp_is_not_zero_or_heartbeat():
    item = row(progress_scoped=observation(stamp="2026-10-03T12:02:00Z"), selected_progress_protocol_version=3)
    assert "Clock difference" in rendered([item], ("task", "report-age"))


def test_duplicate_failures_one_reason_legend_and_no_state_invention():
    unavailable = {"status": "unavailable", "observation_state": "unavailable", "reason": "read_failed"}
    items = [row(task_id=f"task-{i}", progress=unavailable, selected_progress_protocol_version=None) for i in range(2)]
    text = rendered(items, ("task", "state", "activity", "report-age"))
    assert text.count("Unavailable [1]") == 4
    assert text.count("Could not read observation") == 1
    assert "running" in text


@pytest.mark.parametrize(
    "policy,phase,report,expected",
    [
        ("disabled", "queued", None, "Not enabled"),
        ("enabled", "queued", None, "Not started"),
        ("unknown", "queued", None, "Not started"),
        ("enabled", "succeeded", None, "No report recorded"),
        ("enabled", "running", None, "No report yet"),
        ("enabled", "running", True, "Not provided"),
    ],
)
def test_overall_absence_is_evidence_based(policy, phase, report, expected):
    state = "pending" if phase == "queued" else "no_report"
    no_report = {
        "status": "unavailable",
        "observation_state": state,
        "reason": "not_started" if state == "pending" else "no_snapshot",
    }
    item = row(
        phase=phase,
        reporting_policy={"state": policy, "reason": "policy_selection_missing" if policy == "unknown" else None},
        progress=no_report,
        progress_scoped=observation(overall=False) if report else no_report,
        selected_progress_protocol_version=3 if report else None,
    )
    text = rendered([item], ("task", "overall-progress"))
    assert expected in text
    if policy == "unknown":
        assert "Reporting policy could not be verified" in text
        assert "Not enabled" not in text


def test_control_text_is_escaped_and_never_emits_ansi():
    item = row(name="evil\x1b[31m\nname")
    text = rendered([item], ("task", "name"))
    assert "\x1b" not in text
    assert "task-1" in text


@pytest.mark.parametrize(
    "options",
    [
        ["--fields", ""],
        ["--fields", "state,state"],
        ["--fields", "activty"],
        ["--view", "default", "--fields", "state"],
        ["--view", "default", "--format=json"],
        ["--fields", "state", "--format=json"],
        ["--view", "progress", "--limit", "0"],
        ["--view", "progress", "--limit", "1001"],
    ],
)
def test_invalid_list_options_fail_before_project_resolution(monkeypatch, options):
    from qqtools.plugins.qexp.cli import entrypoint

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid presentation must fail before Project reads")

    monkeypatch.setattr(entrypoint.context_commands, "resolve_project", forbidden)
    assert entrypoint.main(["task", "list", *options]) == 2


def test_list_help_exposes_complete_grouped_registry(capsys):
    from qqtools.plugins.qexp.cli.parser import build_parser
    from qqtools.plugins.qexp.task_list_fields import FIELD_HEADINGS

    with pytest.raises(SystemExit) as result:
        build_parser().parse_args(["task", "list", "--help"])
    assert result.value.code == 0
    help_text = capsys.readouterr().out
    assert help_text.index("Examples:") < help_text.index("Fields — identity/state:")
    for field in FIELD_HEADINGS:
        assert field in help_text
    assert "--live-progress" in help_text


@pytest.mark.parametrize("reason", ["cleanup", "attempt_unreadable", "identity_mismatch", "changed_during_read"])
def test_disabled_policy_never_hides_observation_failures(reason):
    failed = {"status": "unavailable", "observation_state": "unavailable", "reason": reason}
    text = rendered(
        [
            row(
                reporting_policy={"state": "disabled", "reason": None},
                progress=failed,
                progress_extended=failed,
                progress_scoped=failed,
            )
        ],
        ("task", "overall-progress"),
    )
    assert "Not enabled" not in text
    assert "Unavailable" in text


@pytest.mark.parametrize("token", ["machine" * 9, "训练" * 20, "e\u0301" * 60])
@pytest.mark.parametrize("width", [28, 48, 80])
def test_long_initial_nonidentity_tokens_fit_available_display_cells(token, width):
    from qqtools.plugins.qexp.cli.output.task_list import _display_width

    report = observation()
    report["progress"]["overall"]["label"] = token
    item = row(claim_machine=token, progress_scoped=report, selected_progress_protocol_version=3)
    text = rendered([item], ("task", "claimed-machine", "overall-progress"), width)
    # These rows have no diagnostic footer and no overlong Task ID.
    assert all(_display_width(line) <= width for line in text.splitlines()), text
    from qqtools.plugins.qexp.cli.output.task_list import _wrap_text

    assert "".join(_wrap_text(token, 12)) == token


def test_redirected_failure_reasons_immediately_follow_their_own_tasks():
    text = rendered(
        [
            row(task_id="task-a", phase="failed", reason="reason-a"),
            row(task_id="task-b", phase="running"),
            row(task_id="task-c", phase="blocked", reason="reason-c"),
        ],
        ("task", "state"),
    )
    assert (
        text.index("task-a")
        < text.index("Reason: reason-a")
        < text.index("task-b")
        < text.index("task-c")
        < text.index("Reason: reason-c")
    )
