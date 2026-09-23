from __future__ import annotations

import json

import pytest

from qqtools.plugins.qexp.runtime.ready.group_member_diagnostics import (
    CHECK_REGISTRY,
    PublicationTracker,
    ReadyMemberCheckError,
    diagnostic_for_exception,
    format_failure_diagnostic,
    parse_degraded_reason,
    serialize_degraded_reason,
    validate_failure_diagnostic,
)
from qqtools.plugins.qexp.runtime.store import JSONRecordSizeError

_STAGES = {
    "projection_check",
    "group_load",
    "locator_validate",
    "page_select",
    "entry_validate",
    "locator_write",
    "member_page_write",
    "member_catalog_write",
    "member_header_write",
    "global_state_commit",
}


def _stage(spec) -> str:
    return spec.stage if hasattr(spec, "stage") else spec["stage"]


def _diagnostic(*, stage: str = "entry_validate", check_id: str = "write.member_page.size") -> dict:
    return {
        "version": 1,
        "component": "group_ready_members",
        "operation": "publish",
        "stage": stage,
        "check_id": check_id,
        "reason_code": "unexpected_failure",
        "facts": {},
        "exception_type": "ValueError",
        "task_id": "task-42",
        "generation": 1,
        "group_name": "exp",
        "input_index": 3,
        "errno": None,
        "json_line": None,
        "json_column": None,
    }


def test_check_registry_exposes_every_protocol_stage_and_valid_fallback() -> None:
    assert {_stage(spec) for spec in CHECK_REGISTRY.values()} == _STAGES

    for check_id, spec in CHECK_REGISTRY.items():
        value = _diagnostic(stage=_stage(spec), check_id=check_id)
        assert validate_failure_diagnostic(value).to_dict() == value


def test_degraded_reason_is_canonical_bounded_and_omits_submission_index() -> None:
    value = _diagnostic()
    value.update(
        reason_code="member_page_too_large",
        facts={"record_type": "member_page", "actual_bytes": 65537, "limit_bytes": 65536},
        exception_type="JsonRecordSizeError",
        task_id="task-" + "x" * 240,
        errno=28,
    )
    validated = validate_failure_diagnostic(value)

    reason = serialize_degraded_reason(validated)

    assert len(reason.encode("utf-8")) <= 512
    assert "input_index" not in reason
    parsed = parse_degraded_reason(reason)
    assert parsed["version"] == 1
    assert parsed["reason_code"] == "member_page_too_large"
    assert parsed["stage"] == "entry_validate"
    assert parsed["check_id"] == "write.member_page.size"
    assert parsed["task_id_sha256"]


def test_diagnostic_validation_rejects_unknown_fields_and_unsafe_values() -> None:
    value = _diagnostic()
    value["secret"] = "do-not-persist"
    with pytest.raises((TypeError, ValueError)):
        validate_failure_diagnostic(value)

    for field, unsafe in (
        ("exception_type", "Bad Error /private/path"),
        ("task_id", "task with spaces"),
        ("generation", -1),
        ("errno", -1),
        ("facts", {"arbitrary": "SECRET"}),
    ):
        value = _diagnostic()
        value[field] = unsafe
        with pytest.raises((TypeError, ValueError)):
            validate_failure_diagnostic(value)


def test_degraded_reason_parser_rejects_noncanonical_or_incomplete_envelopes() -> None:
    reason = serialize_degraded_reason(validate_failure_diagnostic(_diagnostic()))
    parts = reason.split(";")

    with pytest.raises(ValueError):
        parse_degraded_reason(";".join((parts[0], parts[2], parts[1], *parts[3:])))
    with pytest.raises(ValueError):
        parse_degraded_reason(";".join(part for part in parts if not part.startswith("exception_type=")))


def test_full_diagnostic_and_human_line_are_bounded_and_safe() -> None:
    value = validate_failure_diagnostic(_diagnostic())

    assert len(json.dumps(value.to_dict(), sort_keys=True).encode("utf-8")) <= 4096
    rendered = format_failure_diagnostic(value)
    assert rendered.startswith("Diagnostic: ")
    assert "component=group_ready_members" in rendered
    assert "stage=entry_validate" in rendered
    assert "check=write.member_page.size" in rendered
    assert "task=task-42" in rendered
    assert "input_index=3" in rendered


def test_unexpected_exception_keeps_its_source_check_without_message_text() -> None:
    tracker = PublicationTracker("write.member_page.temp_write")
    error = ValueError("SECRET_EXCEPTION_SENTINEL /private/path")

    diagnostic = diagnostic_for_exception(
        error,
        tracker,
        task_id="task-42",
        generation=1,
        group_name="exp",
    )

    assert diagnostic.stage == "member_page_write"
    assert diagnostic.check_id == "write.member_page.temp_write"
    assert diagnostic.reason_code == "unexpected_failure"
    assert diagnostic.exception_type == "ValueError"
    assert "SECRET" not in json.dumps(diagnostic.to_dict(), sort_keys=True)
    assert "/private/path" not in json.dumps(diagnostic.to_dict(), sort_keys=True)


def test_malformed_known_facts_fall_back_without_masking_publication_site() -> None:
    tracker = PublicationTracker("entry.identifier_validate")
    error = ReadyMemberCheckError(
        "unsupported_identifier_encoding",
        {"field": "SECRET_UNREGISTERED_FIELD", "message": "SECRET"},
    )

    diagnostic = diagnostic_for_exception(
        error,
        tracker,
        task_id="task-42",
        generation=1,
        group_name="exp",
    )

    assert diagnostic.stage == "entry_validate"
    assert diagnostic.check_id == "entry.identifier_validate"
    assert diagnostic.reason_code == "unsupported_identifier_encoding"
    assert diagnostic.facts == {}
    assert "SECRET" not in json.dumps(diagnostic.to_dict(), sort_keys=True)


@pytest.mark.parametrize(
    ("check_id", "reason_code"),
    [
        ("write.member_page.temp_write", "storage_write_failure"),
        ("write.member_page.replace", "storage_replace_failure"),
        ("write.member_page.file_fsync", "storage_durability_failure"),
        ("write.member_page.directory_fsync", "storage_durability_failure"),
    ],
)
def test_storage_errors_use_the_active_source_check(check_id: str, reason_code: str) -> None:
    diagnostic = diagnostic_for_exception(
        OSError(5, "SECRET_STORAGE_MESSAGE"),
        PublicationTracker(check_id),
        task_id="task-42",
        generation=1,
        group_name="exp",
    )

    assert diagnostic.check_id == check_id
    assert diagnostic.reason_code == reason_code
    assert diagnostic.errno == 5
    assert "SECRET" not in json.dumps(diagnostic.to_dict(), sort_keys=True)


def test_json_metadata_and_oversized_record_facts_are_allowlisted() -> None:
    malformed = diagnostic_for_exception(
        json.JSONDecodeError("SECRET_JSON_MESSAGE", "{", 1),
        PublicationTracker("group.header_read"),
        task_id="task-42",
        generation=1,
        group_name="exp",
    )
    oversized = diagnostic_for_exception(
        JSONRecordSizeError(
            "SECRET_SIZE_MESSAGE",
            record_type="member_header",
            actual_bytes=4097,
            limit_bytes=4096,
        ),
        PublicationTracker("group.header_read"),
        task_id="task-42",
        generation=1,
        group_name="exp",
    )

    assert malformed.reason_code == "json_record_malformed"
    assert (malformed.json_line, malformed.json_column) == (1, 2)
    assert oversized.reason_code == "json_record_oversized"
    assert oversized.facts == {
        "record_type": "member_header",
        "actual_bytes": 4097,
        "limit_bytes": 4096,
    }
    assert "SECRET" not in json.dumps(malformed.to_dict(), sort_keys=True)
    assert "SECRET" not in json.dumps(oversized.to_dict(), sort_keys=True)


def test_encoder_failure_uses_source_owned_last_resort_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from qqtools.plugins.qexp.runtime.ready import group_member_diagnostics

    def fail_encoder(_value) -> str:
        raise ValueError("SECRET_ENCODER_FAILURE")

    monkeypatch.setattr(group_member_diagnostics, "_encode_diagnostic", fail_encoder)

    diagnostic = diagnostic_for_exception(
        OSError(5, "SECRET_PRIMARY_FAILURE"),
        PublicationTracker("write.member_page.temp_write"),
        task_id="task-42",
        generation=1,
        group_name="exp",
    )

    assert diagnostic.stage == "member_page_write"
    assert diagnostic.check_id == "write.member_page.temp_write"
    assert diagnostic.reason_code == "unexpected_failure"
    assert "SECRET" not in json.dumps(diagnostic.to_dict(), sort_keys=True)
