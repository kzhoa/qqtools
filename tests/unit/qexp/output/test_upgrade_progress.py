from __future__ import annotations

import json
from copy import deepcopy

import pytest

from qqtools.plugins.qexp.cli.local_handlers import _upgrade_payload
from qqtools.plugins.qexp.cli.output import CliOutput, OutputKind, render
from qqtools.plugins.qexp.upgrade_progress import accepted_progress, semantic_delta, validate_progress


def _progress(**changes):
    return {
        "version": 1,
        "scope": "shared_project",
        "stage": "submissions",
        "stage_label": "Submission bootstrap",
        "stage_state": "processing",
        "scan_epoch": 3,
        "completed_units": 183,
        "inventoried_units": 237,
        "total_units": 237,
        "total_kind": "snapshot_exact",
        "unit": "sources",
        "remaining_stages": ["submission_control", "group_control", "cleanup", "discovery_debt", "stable_pass"],
        "last_progress_at": "2026-10-04T18:12:00Z",
        **changes,
    }


def _evidence(**changes):
    return {
        "version": 1,
        "journal_revision": 10,
        "identity": "a" * 64,
        "activation_revision": 350,
        "activation_cursor": 9223372036854775807,
        "source_checkpoint": None,
        **changes,
    }


def _status(**changes):
    return {
        "project_id": "482557a2a27c86ad",
        "state": "runnable",
        "phase": "activation",
        "pending": True,
        "admission_blocked": False,
        "migration_blocked": False,
        "blockers": [],
        "progress": _progress(),
        "progress_evidence": _evidence(),
        **changes,
    }


def test_progress_exact_and_nullable_contract_round_trips():
    for value in (
        _progress(),
        _progress(completed_units=None, last_progress_at=None),
        _progress(completed_units=0, total_units=0, inventoried_units=0),
        _progress(total_units=None, total_kind="dynamic"),
    ):
        assert validate_progress(json.loads(json.dumps(value))) == value


@pytest.mark.parametrize(
    "changes",
    [
        {"version": True},
        {"version": 2},
        {"scope": []},
        {"stage_state": {}},
        {"scan_epoch": 0},
        {"completed_units": True},
        {"completed_units": -1},
        {"completed_units": 238},
        {"inventoried_units": -1},
        {"total_units": None},
        {"total_kind": "dynamic"},
        {"stage": "../../submissions"},
        {"stage_label": "Submission\nsecret"},
        {"last_progress_at": "2026-10-04T12:00:00"},
        {"last_progress_at": "2026-10-04\n12:00:00Z"},
        {"remaining_stages": ["cleanup", "cleanup"]},
        {"current_item": {"kind": "submission", "id": "/private/source", "completed_bytes": 0, "total_bytes": 1}},
        {"current_item": {"kind": "submission", "id": "op", "completed_bytes": 2, "total_bytes": 1}},
        {"current_item": {"kind": "submission", "id": "op", "completed_bytes": 2}},
    ],
)
def test_malformed_progress_is_rejected(changes):
    with pytest.raises(ValueError):
        validate_progress(_progress(**changes))


@pytest.mark.parametrize("mutation", ["missing", "malformed", "old_revision", "in_flight", "wrong_bytes"])
def test_bad_publication_is_unavailable_without_mutating_durable_state(mutation):
    item = {"state": "runnable", "in_flight": False, "progress": _progress(), "progress_evidence": _evidence()}
    if mutation == "missing":
        item.pop("progress")
    elif mutation == "malformed":
        item["progress"] = {"version": 1}
    elif mutation == "old_revision":
        item["progress_evidence"]["journal_revision"] = 9
    elif mutation == "in_flight":
        item["in_flight"] = True
    else:
        item["progress"]["current_item"] = {"kind": "submission", "id": "op", "completed_bytes": 10, "total_bytes": 20}
        item["progress_evidence"]["source_checkpoint"] = {"identity": "b" * 64, "completed_bytes": 9}
    original = deepcopy(item)
    assert accepted_progress(item, 10) is None
    assert item == original


def test_publication_is_bound_to_exact_journal_revision():
    item = {"in_flight": False, "progress": _progress(), "progress_evidence": _evidence()}
    assert accepted_progress(item, 10) == _progress()
    assert accepted_progress(item, 11) is None


def test_semantic_comparison_never_uses_timestamp_revision_or_cookie_as_count():
    before = _status()
    after = deepcopy(before)
    after["progress"]["last_progress_at"] = "2026-10-04T19:12:00Z"
    after["progress_evidence"].update(activation_revision=999, activation_cursor=42, journal_revision=20)
    assert semantic_delta(before, after) is None
    after["progress"]["completed_units"] += 1
    assert semantic_delta(before, after)["kind"] == "units"
    after["progress"]["scan_epoch"] += 1
    assert semantic_delta(before, after) is None


def test_byte_comparison_requires_same_validated_source_identity():
    before = _status(
        progress=_progress(
            current_item={"kind": "submission", "id": "op", "completed_bytes": 16384, "total_bytes": 90000}
        )
    )
    before["progress_evidence"]["source_checkpoint"] = {"identity": "b" * 64, "completed_bytes": 16384}
    after = deepcopy(before)
    after["progress"]["current_item"]["completed_bytes"] = 32768
    after["progress_evidence"]["source_checkpoint"]["completed_bytes"] = 32768
    assert semantic_delta(before, after)["kind"] == "bytes"
    after["progress_evidence"]["source_checkpoint"]["identity"] = "c" * 64
    assert semantic_delta(before, after) is None


@pytest.mark.parametrize(
    ("progress", "expected", "percent"),
    [
        (_progress(), "183/237", True),
        (_progress(stage_state="complete", completed_units=237), "submissions 237/237", True),
        (
            _progress(stage_state="complete", completed_units=0, total_units=0, inventoried_units=0),
            "submissions 0/0",
            False,
        ),
        (
            _progress(
                stage="stable_pass",
                stage_state="complete",
                completed_units=None,
                total_units=None,
                total_kind="unknown",
            ),
            "stable pass stage complete",
            False,
        ),
        (_progress(completed_units=0, total_units=0, inventoried_units=0), "0/0", False),
        (_progress(total_units=None, total_kind="dynamic", completed_units=12), "12 processed", False),
        (_progress(stage_state="inventory", total_units=None, total_kind="unknown"), "inventory", False),
        (
            _progress(stage_state="recounting", completed_units=None, total_units=None, total_kind="dynamic"),
            "unavailable",
            False,
        ),
        (None, "unavailable", False),
    ],
)
def test_project_progress_renderer_preserves_json_and_exactness(progress, expected, percent):
    result = _upgrade_payload(_status(progress=progress), "status")
    human = render(CliOutput(OutputKind.UPGRADE_PROJECT, result), "human")
    assert "Progress:" in human
    assert expected in human
    assert ("%" in human) is percent
    assert json.loads(render(CliOutput(OutputKind.UPGRADE_PROJECT, result), "json")) == result


def test_machine_progress_details_are_wrapped_and_include_shared_scope():
    status = _status(
        progress=_progress(
            current_item={"kind": "submission", "id": "a" * 256, "completed_bytes": 49152, "total_bytes": 71234}
        )
    )
    result = {"projects": [{"project_id": status["project_id"], "upgrade": status}], "aggregate_state": "pending"}
    human = render(CliOutput(OutputKind.UPGRADE_REGISTRY_STATUS, result), "human")
    assert "Progress" in human
    assert "77.2%" in human
    assert "49152/71234 bytes" in human
    assert "shared Project" in human
    assert max(map(len, human.splitlines())) <= 160


@pytest.mark.parametrize("lock", ["upgrade", "schema"])
@pytest.mark.parametrize("observed", [False, True])
def test_invocation_waiting_has_no_own_delta_or_guessed_holder(lock, observed):
    status = _status(
        state="waiting",
        blockers=[f"{lock}_lock_busy"],
        invocation={
            "slice_committed": False,
            "contended_lock": lock,
            "next_probe_at": None,
            "observed_progress": observed,
            "delta": None,
        },
    )
    result = _upgrade_payload(status, "advance")
    assert result["outcome"] == "waiting"
    human = render(CliOutput(OutputKind.UPGRADE_PROJECT, result), "human")
    assert f"waiting for Project {lock} lock" in human
    assert ("another coordinator advanced shared Project work" in human) is observed
    assert "advanced Submission" not in human


@pytest.mark.parametrize("state", ["repair_required", "paused", "pause_pending", "validation_failed"])
def test_invocation_contention_cannot_hide_intervention(state):
    status = _status(
        state=state,
        blockers=["schema_lock_busy"],
        invocation={
            "slice_committed": False,
            "contended_lock": "schema",
            "next_probe_at": None,
            "observed_progress": False,
            "delta": None,
        },
    )
    result = _upgrade_payload(status, "advance")
    assert result["outcome"] == ("failed" if state == "validation_failed" else "blocked")
    assert result["next_action"] == "qexp admin upgrade status"


def test_invalid_progress_is_rejected_at_output_contract_boundary():
    with pytest.raises((ValueError, TypeError)):
        render(CliOutput(OutputKind.UPGRADE_PROJECT, _status(progress={"version": 1})), "json")


def test_own_byte_delta_describes_bytes_without_claiming_record_completion():
    result = _upgrade_payload(
        _status(
            invocation={
                "slice_committed": True,
                "contended_lock": None,
                "next_probe_at": None,
                "observed_progress": False,
                "delta": {
                    "kind": "bytes",
                    "stage": "submissions",
                    "unit": "sources",
                    "before": 32768,
                    "after": 49152,
                    "total": 71234,
                    "item_kind": "submission",
                    "item_id": "op-1",
                },
            }
        ),
        "advance",
    )
    human = render(CliOutput(OutputKind.UPGRADE_PROJECT, result), "human")
    assert "advanced Submission source bytes 32768 -> 49152" in human
    assert "completed Submission" not in human


@pytest.mark.parametrize("safety", [None, "admission_blocked", "migration_blocked", "repair_required"])
def test_machine_lock_only_waiting_preserves_safety_precedence(safety):
    status = _status(state="waiting", blockers=["schema_lock_busy"])
    if safety == "repair_required":
        status["state"] = safety
    elif safety:
        status[safety] = True
    result = _upgrade_payload(
        {"projects": [{"project_id": "p", "upgrade": status}], "pending_project_ids": ["p"]}, "status"
    )
    assert result["outcome"] == ("blocked" if safety else "waiting")


def test_machine_intervention_reason_wins_over_an_earlier_lock_wait():
    result = _upgrade_payload(
        {
            "projects": [
                {"project_id": "waiting", "upgrade": _status(state="waiting", blockers=["schema_lock_busy"])},
                {
                    "project_id": "damaged",
                    "upgrade": _status(state="repair_required", blockers=["invalid source JSON"]),
                },
            ],
            "pending_project_ids": ["waiting", "damaged"],
        },
        "status",
    )
    assert result["outcome"] == "blocked"
    assert result["reason"] == "invalid source JSON"
