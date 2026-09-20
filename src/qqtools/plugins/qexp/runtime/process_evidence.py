"""Read-only process identity and presence evidence."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from ..infrastructure.process import ProcessIdentityRead, ProcessPresence, read_process_identity, read_process_presence

ProcessEvidenceState = Literal["alive", "absent", "unknown"]
ProcessEvidenceReason = Literal[
    "identity_mismatch",
    "missing_identity",
    "invalid_identity",
    "read_failed",
    "invalid_process_stat",
    "inconsistent_observation",
    "unsupported_probe",
]

_MISSING = object()


@dataclass(frozen=True, slots=True)
class ProcessEvidence:
    """Immutable classification of process identity and presence evidence."""

    state: ProcessEvidenceState
    reason: ProcessEvidenceReason | None = None

    def __post_init__(self) -> None:
        if self.state in {"alive", "absent"}:
            if self.reason is not None:
                raise ValueError("alive and absent process evidence cannot include a reason")
        elif self.state == "unknown":
            if self.reason not in {
                "identity_mismatch",
                "missing_identity",
                "invalid_identity",
                "read_failed",
                "invalid_process_stat",
                "inconsistent_observation",
                "unsupported_probe",
            }:
                raise ValueError("unknown process evidence requires a bounded reason")
        else:
            raise ValueError("invalid process evidence state")


def _valid_process_identifier(value: object) -> bool:
    return type(value) is int and value > 0


def _valid_start_time_ticks(value: object) -> bool:
    return type(value) is int and value >= 0


def _required_fields(
    process: Mapping[str, object],
    field_names: tuple[str, str],
) -> tuple[tuple[object, object], bool, bool]:
    values: list[object] = []
    has_missing = False
    has_invalid = False
    for field_name in field_names:
        value = process.get(field_name, _MISSING)
        values.append(value)
        if value is _MISSING or value is None:
            has_missing = True
        elif field_name.endswith(("_id", "_pid")):
            if not _valid_process_identifier(value):
                has_invalid = True
        elif not _valid_start_time_ticks(value):
            has_invalid = True
    return (values[0], values[1]), has_missing, has_invalid


def _validation_evidence(has_missing: bool, has_invalid: bool) -> ProcessEvidence | None:
    if has_missing:
        return ProcessEvidence(state="unknown", reason="missing_identity")
    if has_invalid:
        return ProcessEvidence(state="unknown", reason="invalid_identity")
    return None


def _unknown_evidence(observation: ProcessIdentityRead | ProcessPresence) -> ProcessEvidence:
    return ProcessEvidence(state="unknown", reason=observation.reason)


def _sample_process_evidence(pid: int, expected_ticks: int, *, is_group: bool) -> ProcessEvidence:
    first_identity = read_process_identity(pid)
    if first_identity.state == "unknown":
        return _unknown_evidence(first_identity)
    if first_identity.state == "present" and first_identity.start_time_ticks != expected_ticks:
        return ProcessEvidence(state="unknown", reason="identity_mismatch")

    presence = read_process_presence(pid, is_group=is_group)
    if presence.state == "unknown":
        return _unknown_evidence(presence)

    second_identity = read_process_identity(pid)
    if second_identity.state == "unknown":
        return _unknown_evidence(second_identity)
    if second_identity.state == "present" and second_identity.start_time_ticks != expected_ticks:
        return ProcessEvidence(state="unknown", reason="identity_mismatch")

    if first_identity.state == "present" and second_identity.state == "present" and presence.state == "present":
        return ProcessEvidence(state="alive")
    if first_identity.state == "absent" and second_identity.state == "absent" and presence.state == "absent":
        return ProcessEvidence(state="absent")
    return ProcessEvidence(state="unknown", reason="inconsistent_observation")


def inspect_group_identity(recorded: Mapping[str, object], manifest: Mapping[str, object]) -> ProcessEvidence:
    """Compare recorded and manifest process-group identity before sampling it."""
    recorded_fields, recorded_missing, recorded_invalid = _required_fields(
        recorded,
        ("process_group_id", "process_group_start_time_ticks"),
    )
    manifest_fields, manifest_missing, manifest_invalid = _required_fields(
        manifest,
        ("process_group_id", "process_group_start_time_ticks"),
    )
    validation = _validation_evidence(
        recorded_missing or manifest_missing,
        recorded_invalid or manifest_invalid,
    )
    if validation is not None:
        return validation
    if recorded_fields != manifest_fields:
        return ProcessEvidence(state="unknown", reason="identity_mismatch")
    return _sample_process_evidence(
        recorded_fields[0],
        recorded_fields[1],
        is_group=True,
    )


def inspect_wrapper_identity(manifest: Mapping[str, object]) -> ProcessEvidence:
    """Sample wrapper identity and process presence from a manifest."""
    fields, has_missing, has_invalid = _required_fields(
        manifest,
        ("wrapper_pid", "wrapper_start_time_ticks"),
    )
    validation = _validation_evidence(has_missing, has_invalid)
    if validation is not None:
        return validation
    return _sample_process_evidence(fields[0], fields[1], is_group=False)


def inspect_local_group_identity(process: Mapping[str, object]) -> ProcessEvidence:
    """Sample one local process group's identity and presence source."""
    fields, has_missing, has_invalid = _required_fields(
        process,
        ("process_group_id", "process_group_start_time_ticks"),
    )
    validation = _validation_evidence(has_missing, has_invalid)
    if validation is not None:
        return validation
    return _sample_process_evidence(fields[0], fields[1], is_group=True)
