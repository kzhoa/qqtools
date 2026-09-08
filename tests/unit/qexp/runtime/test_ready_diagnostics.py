from __future__ import annotations

from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp.runtime.ready.diagnostics import (
    MAX_REASON_BYTES,
    MAX_REASONS,
    ReadyDiagnostic,
    classification_diagnostic,
    diagnostic,
    parse_ready_reason,
    safe_reason_view,
    serialize_reason,
)
from qqtools.plugins.qexp.runtime.ready.state import degrade_state_record

# QQTOOLS-COMPAT-0010


class _Config:
    machine_name = "机器 A"


def test_v1_reason_round_trips_observed_marker_difference() -> None:
    value = diagnostic(
        "marker_invalid",
        stage="marker_schema",
        task_id="task-1",
        generation=1,
        missing_fields=[],
        unexpected_fields=["lane"],
    )

    reason = serialize_reason(_Config(), "marker_corrupt", value)

    assert len(reason.encode("utf-8")) <= MAX_REASON_BYTES
    assert parse_ready_reason(reason).normalized == reason
    assert "%20" in reason


def test_parser_rejects_duplicate_unknown_and_unescaped_fields() -> None:
    value = diagnostic("build_invalid", stage="build_state")
    reason = serialize_reason(_Config(), "build_invalid", value)

    for invalid in (
        reason + ";stage=build_cursor",
        reason + ";unknown=value",
        reason.replace("reader_machine=", "reader_machine=bad name"),
    ):
        with pytest.raises(ValueError):
            parse_ready_reason(invalid)


def test_long_dynamic_fields_are_bounded_and_marked() -> None:
    value = diagnostic(
        "incompatible_active_writers",
        stage="completion_writer_gate",
        incompatible_writers=["writer-" + "x" * 300 for _ in range(20)],
        omitted_count=4,
    )

    reason = serialize_reason(_Config(), "incompatible_active_writers", value)

    assert len(reason.encode("utf-8")) <= MAX_REASON_BYTES
    parsed = parse_ready_reason(reason).diagnostic.as_dict()
    assert parsed["incompatible_writers"]


def test_writer_budget_counts_items_removed_after_the_writer_cap() -> None:
    value = diagnostic(
        "incompatible_active_writers",
        stage="completion_writer_gate",
        incompatible_writers=["writer-" + "x" * 300 + str(index) for index in range(16)],
    )

    parsed = parse_ready_reason(serialize_reason(_Config(), "incompatible_active_writers", value)).diagnostic.as_dict()

    assert len(parsed["incompatible_writers"]) == 1
    assert parsed["omitted_count"] == 15


@pytest.mark.parametrize("schema_version", [True, -1, 10**1000])
def test_invalid_observed_marker_schema_version_cannot_escape_diagnostic_serialization(schema_version) -> None:
    value = classification_diagnostic(
        "marker_invalid",
        SimpleNamespace(task_id="task-1", generation=1),
        marker={"schema_version": schema_version},
    )

    reason = serialize_reason(_Config(), "marker_corrupt", value)

    assert parse_ready_reason(reason).normalized == reason
    assert "observed_schema_version=" not in reason


def test_task_route_diagnostic_reports_only_the_mismatched_field() -> None:
    value = classification_diagnostic(
        "route_mismatch",
        SimpleNamespace(
            task_id="task-1",
            generation=1,
            queue_scope="home",
            home_machine="gpu-1",
        ),
        task=SimpleNamespace(
            placement_runtime={"queue_scope": "shared"},
            placement_policy={"home_machine": "gpu-1"},
        ),
    )

    fields = value.as_dict()
    assert fields["mismatch_fields"] == ["queue_scope"]
    assert fields["expected_queue_scope"] == "home"
    assert fields["observed_queue_scope"] == "shared"
    assert "expected_home_machine" not in fields
    assert "observed_home_machine" not in fields


def test_long_task_identity_remains_parseable_after_bounded_hashing() -> None:
    value = diagnostic(
        "marker_missing",
        stage="marker_truth",
        task_id="task-" + "x" * 300,
        generation=1,
        indexed=False,
        task_projection="queued",
        active_claim=False,
    )

    reason = serialize_reason(_Config(), "marker_corrupt", value)

    assert parse_ready_reason(reason).normalized == reason


def test_unicode_dynamic_fields_obey_encoded_byte_limit() -> None:
    value = diagnostic(
        "build_failed",
        stage="build_backfill",
        exception_type="ValueError",
        object="build",
    )
    config = type("UnicodeConfig", (), {"machine_name": "机" * 300})()

    reason = serialize_reason(config, "build_failed", value)

    assert len(reason.encode("utf-8")) <= MAX_REASON_BYTES
    assert parse_ready_reason(reason).normalized == reason


def test_status_view_hides_unknown_history_and_allows_controlled_legacy() -> None:
    reasons = [
        "marker_corrupt:task-1.2",
        "unknown internal exception with /path and payload",
    ]

    displayed = safe_reason_view(reasons)

    assert displayed[0] == reasons[0]
    assert displayed[1].startswith("legacy_reason_unavailable;sha256=")
    assert "payload" not in displayed[1]


def test_diagnostic_is_typed() -> None:
    assert isinstance(diagnostic("build_invalid", stage="build_state"), ReadyDiagnostic)


def test_reason_limit_emits_fixed_event_without_persisting_an_extra_reason(caplog) -> None:
    record = {"state": "active", "degraded_reasons": []}
    for index in range(MAX_REASONS + 1):
        degrade_state_record(
            record,
            diagnostic(
                "marker_missing",
                stage="marker_truth",
                task_id=f"task-{index}",
                generation=1,
                indexed=False,
                task_projection="queued",
                active_claim=False,
            ),
            cfg=_Config(),
        )

    assert len(record["degraded_reasons"]) == MAX_REASONS
    assert "reason_limit_reached omitted_count=1" in caplog.messages[-1]
