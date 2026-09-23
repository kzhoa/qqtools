"""Bounded diagnostics for online Group ready-member publication."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import quote, unquote

MAX_DIAGNOSTIC_BYTES = 4096
MAX_DEGRADED_REASON_BYTES = 512
MAX_FACTS = 8
_MAX_SAFE_INTEGER = (1 << 63) - 1
_STAGES = frozenset(
    {
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
)
_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_EXCEPTION_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_SAFE_SECONDARIES = frozenset(
    {
        "degraded_reason_encoding_failed",
        "degraded_reason_persistence_failed",
        "submission_abort_persistence_failed",
        "submission_cleanup_failed",
    }
)
_REASON_CODES = frozenset(
    {
        "invalid_locator",
        "stale_locator",
        "conflicting_identity",
        "invalid_writable_page",
        "invalid_writable_queue",
        "unsupported_identifier_encoding",
        "member_page_too_large",
        "member_catalog_too_large",
        "member_locator_too_large",
        "member_header_too_large",
        "member_directory_too_large",
        "member_writable_index_too_large",
        "global_state_too_large",
        "invalid_member_count",
        "invalid_membership_digest",
        "invalid_revision",
        "invalid_page_contents",
        "projection_state_changed",
        "json_record_missing",
        "json_record_malformed",
        "json_record_oversized",
        "storage_read_failure",
        "storage_write_failure",
        "storage_replace_failure",
        "storage_durability_failure",
        "unexpected_failure",
    }
)


@dataclass(frozen=True, slots=True)
class FactRule:
    """A bounded, source-owned diagnostic fact."""

    kind: str
    maximum: int = 0
    choices: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CheckDefinition:
    """Stage and permitted reason/fact combinations for one publication check."""

    stage: str
    reasons: Mapping[str, Mapping[str, FactRule]]


_INTEGER_FACT = FactRule("integer", maximum=_MAX_SAFE_INTEGER)
_FIELD_FACT = FactRule(
    "choice",
    choices=("task_id", "queue_scope", "home_machine", "partition", "marker_name", "submission_operation_id"),
)
_RECORD_FACT = FactRule(
    "choice",
    choices=(
        "member_page",
        "member_catalog",
        "member_locator",
        "member_header",
        "member_directory",
        "member_writable_index",
        "member_global_state",
    ),
)


def _reasons(stage: str, **reasons: Mapping[str, FactRule] | None) -> CheckDefinition:
    normalized = {reason: MappingProxyType(dict(facts or {})) for reason, facts in reasons.items()}
    normalized.setdefault("unexpected_failure", MappingProxyType({}))
    return CheckDefinition(
        stage,
        MappingProxyType(normalized),
    )


def _io_checks(registry: dict[str, CheckDefinition], prefix: str, stage: str) -> None:
    registry[f"{prefix}.temp_write"] = _reasons(stage, storage_write_failure={})
    registry[f"{prefix}.file_fsync"] = _reasons(stage, storage_durability_failure={})
    registry[f"{prefix}.replace"] = _reasons(stage, storage_replace_failure={})
    registry[f"{prefix}.directory_fsync"] = _reasons(stage, storage_durability_failure={})


_registry: dict[str, CheckDefinition] = {
    "projection.layout_check": _reasons("projection_check", projection_state_changed={}, storage_read_failure={}),
    "projection.assert_writable": _reasons(
        "projection_check",
        projection_state_changed={},
        invalid_revision={"expected_revision": _INTEGER_FACT, "observed_revision": _INTEGER_FACT},
        json_record_missing={},
        json_record_malformed={},
        json_record_oversized={
            "record_type": _RECORD_FACT,
            "actual_bytes": _INTEGER_FACT,
            "limit_bytes": _INTEGER_FACT,
        },
        storage_read_failure={},
    ),
    "group.identity_validate": _reasons(
        "group_load",
        conflicting_identity={},
        invalid_revision={"expected_revision": _INTEGER_FACT, "observed_revision": _INTEGER_FACT},
    ),
    "group.header_read": _reasons(
        "group_load",
        json_record_missing={},
        json_record_malformed={},
        json_record_oversized={
            "record_type": _RECORD_FACT,
            "actual_bytes": _INTEGER_FACT,
            "limit_bytes": _INTEGER_FACT,
        },
        storage_read_failure={},
    ),
    "group.catalog_read": _reasons(
        "group_load",
        json_record_missing={},
        json_record_malformed={},
        json_record_oversized={
            "record_type": _RECORD_FACT,
            "actual_bytes": _INTEGER_FACT,
            "limit_bytes": _INTEGER_FACT,
        },
        storage_read_failure={},
    ),
    "group.partition_read": _reasons(
        "group_load",
        json_record_missing={},
        json_record_malformed={},
        json_record_oversized={
            "record_type": _RECORD_FACT,
            "actual_bytes": _INTEGER_FACT,
            "limit_bytes": _INTEGER_FACT,
        },
        storage_read_failure={},
    ),
    "group.state_validate": _reasons(
        "group_load",
        invalid_revision={"expected_revision": _INTEGER_FACT, "observed_revision": _INTEGER_FACT},
        invalid_member_count={"member_count": _INTEGER_FACT},
        invalid_membership_digest={},
        invalid_page_contents={},
        conflicting_identity={},
    ),
    "locator.exists": _reasons("locator_validate", storage_read_failure={}),
    "locator.read": _reasons(
        "locator_validate",
        json_record_missing={},
        json_record_malformed={},
        json_record_oversized={
            "record_type": _RECORD_FACT,
            "actual_bytes": _INTEGER_FACT,
            "limit_bytes": _INTEGER_FACT,
        },
        storage_read_failure={},
    ),
    "locator.validate": _reasons(
        "locator_validate",
        invalid_locator={},
        stale_locator={},
        conflicting_identity={},
        invalid_revision={"expected_revision": _INTEGER_FACT, "observed_revision": _INTEGER_FACT},
    ),
    "locator.page_header_read": _reasons(
        "locator_validate",
        json_record_missing={},
        json_record_malformed={},
        json_record_oversized={
            "record_type": _RECORD_FACT,
            "actual_bytes": _INTEGER_FACT,
            "limit_bytes": _INTEGER_FACT,
        },
        storage_read_failure={},
    ),
    "locator.page_catalog_read": _reasons(
        "locator_validate",
        json_record_missing={},
        json_record_malformed={},
        json_record_oversized={
            "record_type": _RECORD_FACT,
            "actual_bytes": _INTEGER_FACT,
            "limit_bytes": _INTEGER_FACT,
        },
        storage_read_failure={},
    ),
    "locator.page_partition_read": _reasons(
        "locator_validate",
        json_record_missing={},
        json_record_malformed={},
        json_record_oversized={
            "record_type": _RECORD_FACT,
            "actual_bytes": _INTEGER_FACT,
            "limit_bytes": _INTEGER_FACT,
        },
        storage_read_failure={},
    ),
    "locator.page_validate": _reasons(
        "locator_validate",
        invalid_revision={"expected_revision": _INTEGER_FACT, "observed_revision": _INTEGER_FACT},
        invalid_page_contents={},
        conflicting_identity={},
        invalid_member_count={"member_count": _INTEGER_FACT},
        invalid_membership_digest={},
        stale_locator={},
    ),
    "page.current_validate": _reasons("page_select", invalid_writable_page={"page": _INTEGER_FACT}),
    "page.current_header_read": _reasons(
        "page_select",
        json_record_missing={},
        json_record_malformed={},
        json_record_oversized={
            "record_type": _RECORD_FACT,
            "actual_bytes": _INTEGER_FACT,
            "limit_bytes": _INTEGER_FACT,
        },
        storage_read_failure={},
    ),
    "page.current_catalog_read": _reasons(
        "page_select",
        json_record_missing={},
        json_record_malformed={},
        json_record_oversized={
            "record_type": _RECORD_FACT,
            "actual_bytes": _INTEGER_FACT,
            "limit_bytes": _INTEGER_FACT,
        },
        storage_read_failure={},
    ),
    "page.current_partition_read": _reasons(
        "page_select",
        json_record_missing={},
        json_record_malformed={},
        json_record_oversized={
            "record_type": _RECORD_FACT,
            "actual_bytes": _INTEGER_FACT,
            "limit_bytes": _INTEGER_FACT,
        },
        storage_read_failure={},
    ),
    "page.current_entries": _reasons(
        "page_select",
        invalid_page_contents={},
        conflicting_identity={},
        invalid_member_count={"member_count": _INTEGER_FACT},
        invalid_membership_digest={},
        invalid_revision={"expected_revision": _INTEGER_FACT, "observed_revision": _INTEGER_FACT},
    ),
    "page.index_read": _reasons(
        "page_select",
        json_record_missing={},
        json_record_malformed={},
        json_record_oversized={
            "record_type": _RECORD_FACT,
            "actual_bytes": _INTEGER_FACT,
            "limit_bytes": _INTEGER_FACT,
        },
        storage_read_failure={},
    ),
    "page.directory_read": _reasons(
        "page_select",
        json_record_missing={},
        json_record_malformed={},
        json_record_oversized={
            "record_type": _RECORD_FACT,
            "actual_bytes": _INTEGER_FACT,
            "limit_bytes": _INTEGER_FACT,
        },
        storage_read_failure={},
    ),
    "page.directory_validate": _reasons(
        "page_select", invalid_writable_page={"page": _INTEGER_FACT}, invalid_page_contents={}
    ),
    "page.queue_validate": _reasons(
        "page_select",
        invalid_writable_queue={"queue_field": FactRule("choice", choices=("head", "tail", "free", "count"))},
        invalid_writable_page={"page": _INTEGER_FACT},
    ),
    "page.allocate_validate": _reasons("page_select", invalid_writable_page={"page": _INTEGER_FACT}),
    "page.directory_size": _reasons(
        "page_select",
        member_directory_too_large={
            "record_type": _RECORD_FACT,
            "actual_bytes": _INTEGER_FACT,
            "limit_bytes": _INTEGER_FACT,
        },
    ),
    "page.index_size": _reasons(
        "page_select",
        member_writable_index_too_large={
            "record_type": _RECORD_FACT,
            "actual_bytes": _INTEGER_FACT,
            "limit_bytes": _INTEGER_FACT,
        },
    ),
    "entry.identifier_validate": _reasons("entry_validate", unsupported_identifier_encoding={"field": _FIELD_FACT}),
    "entry.revision_validate": _reasons(
        "entry_validate", invalid_revision={"expected_revision": _INTEGER_FACT, "observed_revision": _INTEGER_FACT}
    ),
    "entry.count_validate": _reasons("entry_validate", invalid_member_count={"member_count": _INTEGER_FACT}),
    "entry.digest_validate": _reasons("entry_validate", invalid_membership_digest={}),
    "write.member_page.size": _reasons(
        "entry_validate",
        member_page_too_large={
            "record_type": _RECORD_FACT,
            "actual_bytes": _INTEGER_FACT,
            "limit_bytes": _INTEGER_FACT,
        },
    ),
    "write.member_catalog.size": _reasons(
        "entry_validate",
        member_catalog_too_large={
            "record_type": _RECORD_FACT,
            "actual_bytes": _INTEGER_FACT,
            "limit_bytes": _INTEGER_FACT,
        },
    ),
    "write.member_header.size": _reasons(
        "entry_validate",
        member_header_too_large={
            "record_type": _RECORD_FACT,
            "actual_bytes": _INTEGER_FACT,
            "limit_bytes": _INTEGER_FACT,
        },
    ),
    "locator.write.size": _reasons(
        "entry_validate",
        member_locator_too_large={
            "record_type": _RECORD_FACT,
            "actual_bytes": _INTEGER_FACT,
            "limit_bytes": _INTEGER_FACT,
        },
    ),
    "global.read": _reasons(
        "global_state_commit",
        json_record_missing={},
        json_record_malformed={},
        json_record_oversized={
            "record_type": _RECORD_FACT,
            "actual_bytes": _INTEGER_FACT,
            "limit_bytes": _INTEGER_FACT,
        },
        storage_read_failure={},
    ),
    "global.validate": _reasons(
        "global_state_commit",
        projection_state_changed={},
        invalid_revision={"expected_revision": _INTEGER_FACT, "observed_revision": _INTEGER_FACT},
    ),
    "global.size": _reasons(
        "global_state_commit",
        global_state_too_large={
            "record_type": _RECORD_FACT,
            "actual_bytes": _INTEGER_FACT,
            "limit_bytes": _INTEGER_FACT,
        },
    ),
}
for _prefix, _stage in (
    ("locator.write", "locator_write"),
    ("page.index_write", "page_select"),
    ("page.directory_write", "page_select"),
    ("write.member_page", "member_page_write"),
    ("write.member_catalog", "member_catalog_write"),
    ("write.member_header", "member_header_write"),
    ("global.write", "global_state_commit"),
):
    _io_checks(_registry, _prefix, _stage)
CHECK_REGISTRY: Mapping[str, CheckDefinition] = MappingProxyType(_registry)


class ReadyMemberCheckError(ValueError):
    """A publication validation failure with a stable reason and safe facts."""

    def __init__(self, reason_code: str, facts: Mapping[str, Any] | None = None) -> None:
        super().__init__("Group ready-member publication validation failed.")
        self.reason_code = reason_code
        self.facts = dict(facts or {})


class PublicationTracker:
    """Track the last registered publication boundary and active I/O step."""

    def __init__(self, check_id: str = "projection.layout_check") -> None:
        self.check_id = "projection.layout_check"
        self.stage = "projection_check"
        self.io_step: str | None = None
        self.enter(check_id)

    def enter(self, check_id: str) -> None:
        definition = CHECK_REGISTRY.get(check_id)
        if definition is None:
            raise ValueError("publication check is not registered")
        self.check_id = check_id
        self.stage = definition.stage
        self.io_step = None

    def observe_io(self, check_id: str, step: str) -> None:
        self.io_step = step
        suffix = {
            "temp_write": "temp_write",
            "file_fsync": "file_fsync",
            "replace": "replace",
            "directory_fsync": "directory_fsync",
        }.get(step)
        if suffix is not None:
            self.enter(f"{check_id}.{suffix}")
            self.io_step = step


@dataclass(frozen=True, slots=True)
class ReadyMemberFailureDiagnostic:
    """Immutable, validated publication evidence safe for durable/user output."""

    version: int
    component: str
    operation: str
    stage: str
    check_id: str
    reason_code: str
    facts: Mapping[str, Any]
    exception_type: str
    task_id: str
    generation: int
    group_name: str
    input_index: int | None = None
    errno: int | None = None
    json_line: int | None = None
    json_column: int | None = None

    def __post_init__(self) -> None:
        normalized = _validated_fields(self.to_dict(include_facts=False), self.facts)
        object.__setattr__(self, "facts", MappingProxyType(normalized["facts"]))
        for key in (
            "version",
            "component",
            "operation",
            "stage",
            "check_id",
            "reason_code",
            "exception_type",
            "task_id",
            "generation",
            "group_name",
            "input_index",
            "errno",
            "json_line",
            "json_column",
        ):
            object.__setattr__(self, key, normalized[key])
        if len(_encode_diagnostic(self.to_dict()).encode("utf-8")) > MAX_DIAGNOSTIC_BYTES:
            raise ValueError("failure diagnostic exceeds its byte budget")

    def to_dict(self, *, include_facts: bool = True) -> dict[str, Any]:
        value = {
            "version": self.version,
            "component": self.component,
            "operation": self.operation,
            "stage": self.stage,
            "check_id": self.check_id,
            "reason_code": self.reason_code,
            "facts": dict(self.facts) if include_facts else {},
            "exception_type": self.exception_type,
            "task_id": self.task_id,
            "generation": self.generation,
            "group_name": self.group_name,
            "input_index": self.input_index,
            "errno": self.errno,
            "json_line": self.json_line,
            "json_column": self.json_column,
        }
        return value

    def with_input_index(self, input_index: int | None) -> ReadyMemberFailureDiagnostic:
        return replace(self, input_index=input_index)


def _encode_diagnostic(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _valid_integer(value: object, *, nullable: bool = False) -> bool:
    return (nullable and value is None) or (type(value) is int and 0 <= value <= _MAX_SAFE_INTEGER)


def _validated_fields(base: Mapping[str, Any], facts: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "version",
        "component",
        "operation",
        "stage",
        "check_id",
        "reason_code",
        "facts",
        "exception_type",
        "task_id",
        "generation",
        "group_name",
        "input_index",
        "errno",
        "json_line",
        "json_column",
    }
    if set(base) != required or not isinstance(facts, Mapping) or len(facts) > MAX_FACTS:
        raise ValueError("failure diagnostic fields are invalid")
    check_id = base["check_id"]
    reason_code = base["reason_code"]
    stage = base["stage"]
    definition = CHECK_REGISTRY.get(check_id) if isinstance(check_id, str) else None
    if (
        base["version"] != 1
        or type(base["version"]) is not int
        or base["component"] != "group_ready_members"
        or base["operation"] != "publish"
        or not isinstance(reason_code, str)
        or not isinstance(stage, str)
        or definition is None
        or stage != definition.stage
        or reason_code not in definition.reasons
        or stage not in _STAGES
    ):
        raise ValueError("failure diagnostic protocol identity is invalid")
    exception_type = base["exception_type"]
    task_id = base["task_id"]
    group_name = base["group_name"]
    if (
        not isinstance(exception_type, str)
        or not _EXCEPTION_PATTERN.fullmatch(exception_type)
        or not isinstance(task_id, str)
        or not task_id
        or len(task_id.encode("utf-8")) > 256
        or not _ID_PATTERN.fullmatch(task_id)
        or not isinstance(group_name, str)
        or not group_name
        or len(group_name.encode("utf-8")) > 64
        or not _ID_PATTERN.fullmatch(group_name)
        or group_name[0] in ".-"
        or group_name in {"experiments", "qqtools_internal"}
    ):
        raise ValueError("failure diagnostic identifiers are invalid")
    for key in ("generation",):
        if not _valid_integer(base[key]):
            raise ValueError("failure diagnostic integer is invalid")
    for key in ("input_index", "errno", "json_line", "json_column"):
        if not _valid_integer(base[key], nullable=True):
            raise ValueError("failure diagnostic optional integer is invalid")
    allowed_facts = definition.reasons[base["reason_code"]]
    normalized_facts: dict[str, Any] = {}
    for key, value in facts.items():
        rule = allowed_facts.get(key)
        if rule is None:
            raise ValueError("failure diagnostic fact is not registered")
        if rule.kind == "integer":
            if not _valid_integer(value):
                raise ValueError("failure diagnostic fact integer is invalid")
        elif rule.kind == "choice":
            if not isinstance(value, str) or value not in rule.choices:
                raise ValueError("failure diagnostic fact value is invalid")
        elif rule.kind == "string":
            if not isinstance(value, str) or len(value.encode("utf-8")) > rule.maximum:
                raise ValueError("failure diagnostic fact string is invalid")
        else:
            raise ValueError("failure diagnostic fact rule is invalid")
        normalized_facts[key] = value
    result = dict(base)
    result["facts"] = normalized_facts
    return result


def validate_failure_diagnostic(value: object) -> ReadyMemberFailureDiagnostic:
    """Validate the exact v1 diagnostic contract and return its immutable model."""
    if isinstance(value, ReadyMemberFailureDiagnostic):
        return value
    if not isinstance(value, Mapping) or set(value) != {
        "version",
        "component",
        "operation",
        "stage",
        "check_id",
        "reason_code",
        "facts",
        "exception_type",
        "task_id",
        "generation",
        "group_name",
        "input_index",
        "errno",
        "json_line",
        "json_column",
    }:
        raise ValueError("failure diagnostic fields are invalid")
    base = {key: item for key, item in value.items() if key != "facts"}
    facts = value["facts"]
    normalized = _validated_fields({**base, "facts": {}}, facts)
    return ReadyMemberFailureDiagnostic(**normalized)


class ReadyMemberPublicationError(RuntimeError):
    """Public, bounded publication failure preserving the legacy error message."""

    def __init__(
        self,
        message: str,
        diagnostic: ReadyMemberFailureDiagnostic,
        secondary_failures: tuple[str, ...] | list[str] = (),
    ) -> None:
        super().__init__(message)
        self.diagnostic = validate_failure_diagnostic(diagnostic)
        self.secondary_failures = _normalize_secondaries(secondary_failures)

    def with_diagnostic(self, diagnostic: ReadyMemberFailureDiagnostic) -> ReadyMemberPublicationError:
        enriched = ReadyMemberPublicationError(str(self), diagnostic, self.secondary_failures)
        enriched.__cause__ = self.__cause__
        return enriched

    def with_input_index(self, input_index: int | None) -> ReadyMemberPublicationError:
        return self.with_diagnostic(self.diagnostic.with_input_index(input_index))

    def add_secondary(self, classification: str) -> None:
        self.secondary_failures = _normalize_secondaries((*self.secondary_failures, classification))


def _normalize_secondaries(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        if value in _SAFE_SECONDARIES and value not in normalized and len(normalized) < 8:
            normalized.append(value)
    return tuple(normalized)


def _safe_exception_type(exc: BaseException) -> str:
    name = type(exc).__name__
    return name if isinstance(name, str) and _EXCEPTION_PATTERN.fullmatch(name) else "Exception"


def _safe_identifier(value: object, label: str) -> str:
    if isinstance(value, str) and value and _ID_PATTERN.fullmatch(value):
        try:
            within_limit = len(value.encode("utf-8")) <= (256 if label == "task" else 64)
        except UnicodeError:
            within_limit = False
        valid_group = label != "group" or (value[0] not in ".-" and value not in {"experiments", "qqtools_internal"})
        if within_limit and valid_group:
            return value
    digest = hashlib.sha256(repr(type(value).__name__).encode("ascii", errors="ignore")).hexdigest()[:16]
    return f"invalid-{label}-{digest}"


def fallback_diagnostic_for_exception(
    exc: BaseException,
    tracker: PublicationTracker,
    *,
    task_id: object,
    generation: object,
    group_name: object,
) -> ReadyMemberFailureDiagnostic:
    """Build the bounded last-resort model without invoking diagnostic encoders."""
    check_id = tracker.check_id if tracker.check_id in CHECK_REGISTRY else "projection.layout_check"
    diagnostic = object.__new__(ReadyMemberFailureDiagnostic)
    values = {
        "version": 1,
        "component": "group_ready_members",
        "operation": "publish",
        "stage": CHECK_REGISTRY[check_id].stage,
        "check_id": check_id,
        "reason_code": "unexpected_failure",
        "facts": MappingProxyType({}),
        "exception_type": _safe_exception_type(exc),
        "task_id": _safe_identifier(task_id, "task"),
        "generation": generation if _valid_integer(generation) else 0,
        "group_name": _safe_identifier(group_name, "group"),
        "input_index": None,
        "errno": None,
        "json_line": None,
        "json_column": None,
    }
    for key, value in values.items():
        object.__setattr__(diagnostic, key, value)
    return diagnostic


_SIZE_REASONS = {
    "member_page": "member_page_too_large",
    "member_catalog": "member_catalog_too_large",
    "member_locator": "member_locator_too_large",
    "member_header": "member_header_too_large",
    "member_directory": "member_directory_too_large",
    "member_writable_index": "member_writable_index_too_large",
    "member_global_state": "global_state_too_large",
}


def diagnostic_for_exception(
    exc: BaseException,
    tracker: PublicationTracker,
    *,
    task_id: object,
    generation: object,
    group_name: object,
) -> ReadyMemberFailureDiagnostic:
    """Map an exception without inspecting its message or exposing its contents."""
    check_id = tracker.check_id if tracker.check_id in CHECK_REGISTRY else "projection.layout_check"
    definition = CHECK_REGISTRY[check_id]
    reason = "unexpected_failure"
    facts: dict[str, Any] = {}
    json_line = json_column = error_number = None
    candidate_facts: Mapping[str, Any] = {}
    if isinstance(exc, ReadyMemberCheckError):
        reason = exc.reason_code
        candidate_facts = exc.facts
    else:
        from .. import store

        size_error_type = getattr(store, "JSONRecordSizeError", ())
        if isinstance(size_error_type, type) and isinstance(exc, size_error_type):
            record_type = getattr(exc, "record_type", None)
            if isinstance(record_type, str) and record_type in _SIZE_REASONS:
                reason = _SIZE_REASONS[record_type]
                candidate_facts = {
                    "record_type": record_type,
                    "actual_bytes": getattr(exc, "actual_bytes", None),
                    "limit_bytes": getattr(exc, "limit_bytes", None),
                }
                if reason not in definition.reasons and "json_record_oversized" in definition.reasons:
                    reason = "json_record_oversized"
            else:
                reason = "json_record_oversized"
                candidate_facts = {
                    "record_type": record_type if record_type in _RECORD_FACT.choices else "member_header",
                    "actual_bytes": getattr(exc, "actual_bytes", None),
                    "limit_bytes": getattr(exc, "limit_bytes", None),
                }
        elif isinstance(exc, FileNotFoundError):
            reason = "json_record_missing"
        elif isinstance(exc, (json.JSONDecodeError, UnicodeDecodeError)):
            reason = "json_record_malformed"
            if isinstance(exc, json.JSONDecodeError):
                json_line, json_column = exc.lineno, exc.colno
        elif isinstance(exc, OSError):
            error_number = exc.errno if type(exc.errno) is int and exc.errno >= 0 else None
            io_step = tracker.io_step
            if io_step is None:
                if check_id.endswith((".file_fsync", ".directory_fsync")):
                    io_step = "file_fsync"
                elif check_id.endswith(".replace"):
                    io_step = "replace"
                elif check_id.endswith(".temp_write"):
                    io_step = "temp_write"
            if io_step in {"file_fsync", "directory_fsync"}:
                reason = "storage_durability_failure"
            elif io_step == "replace":
                reason = "storage_replace_failure"
            elif io_step == "temp_write":
                reason = "storage_write_failure"
            elif "storage_read_failure" in definition.reasons:
                reason = "storage_read_failure"
    if reason not in definition.reasons:
        reason = "unexpected_failure"
        candidate_facts = {}
    if reason in definition.reasons:
        try:
            facts = _validated_fields(
                {
                    "version": 1,
                    "component": "group_ready_members",
                    "operation": "publish",
                    "stage": definition.stage,
                    "check_id": check_id,
                    "reason_code": reason,
                    "facts": {},
                    "exception_type": _safe_exception_type(exc),
                    "task_id": _safe_identifier(task_id, "task"),
                    "generation": generation if _valid_integer(generation) else 0,
                    "group_name": _safe_identifier(group_name, "group"),
                    "input_index": None,
                    "errno": error_number,
                    "json_line": json_line,
                    "json_column": json_column,
                },
                candidate_facts,
            )["facts"]
        except (TypeError, ValueError):
            facts = {}
    try:
        return ReadyMemberFailureDiagnostic(
            version=1,
            component="group_ready_members",
            operation="publish",
            stage=definition.stage,
            check_id=check_id,
            reason_code=reason,
            facts=facts,
            exception_type=_safe_exception_type(exc),
            task_id=_safe_identifier(task_id, "task"),
            generation=generation if _valid_integer(generation) else 0,
            group_name=_safe_identifier(group_name, "group"),
            input_index=None,
            errno=error_number,
            json_line=json_line,
            json_column=json_column,
        )
    except (TypeError, ValueError):
        return fallback_diagnostic_for_exception(
            exc,
            tracker,
            task_id=task_id,
            generation=generation,
            group_name=group_name,
        )


def serialize_degraded_reason(value: object) -> str:
    """Encode the highest-priority diagnostic fields in a canonical <=512-byte string."""
    diagnostic = validate_failure_diagnostic(value)
    fields: list[tuple[str, str]] = [
        ("version", "1"),
        ("reason_code", diagnostic.reason_code),
        ("stage", diagnostic.stage),
        ("check_id", diagnostic.check_id),
        ("exception_type", diagnostic.exception_type),
    ]
    task_value = diagnostic.task_id
    if len(task_value) > 48:
        fields.append(("task_id_sha256", hashlib.sha256(task_value.encode("utf-8")).hexdigest()[:16]))
    else:
        fields.append(("task_id", task_value))
    fields.append(("generation", str(diagnostic.generation)))
    group_value = diagnostic.group_name
    if len(group_value) > 24:
        fields.append(("group_name_sha256", hashlib.sha256(group_value.encode("utf-8")).hexdigest()[:16]))
    else:
        fields.append(("group_name", group_value))
    for key, item in (
        ("errno", diagnostic.errno),
        ("json_line", diagnostic.json_line),
        ("json_column", diagnostic.json_column),
    ):
        if item is not None:
            fields.append((key, str(item)))
    for key in sorted(diagnostic.facts):
        value_text = str(diagnostic.facts[key])
        fields.append((f"fact.{key}", value_text))
    encoded_fields: list[str] = ["member_publish_failed"]
    for key, field_value in fields:
        candidate = f"{key}={quote(field_value, safe='-._~')}"
        next_value = ";".join((*encoded_fields, candidate))
        if len(next_value.encode("utf-8")) > MAX_DEGRADED_REASON_BYTES:
            if key.startswith("fact.") or key in {
                "errno",
                "json_line",
                "json_column",
                "group_name",
                "group_name_sha256",
            }:
                continue
            if key == "task_id":
                digest = hashlib.sha256(field_value.encode("utf-8")).hexdigest()[:16]
                candidate = f"task_id_sha256={digest}"
                next_value = ";".join((*encoded_fields, candidate))
            if len(next_value.encode("utf-8")) > MAX_DEGRADED_REASON_BYTES:
                raise ValueError("degraded reason required fields exceed their byte budget")
        encoded_fields.append(candidate)
    return ";".join(encoded_fields)


def parse_degraded_reason(value: str) -> dict[str, Any]:
    """Parse and validate one canonical v1 degraded-reason string."""
    if not isinstance(value, str):
        raise ValueError("degraded reason is invalid")
    try:
        if len(value.encode("utf-8")) > MAX_DEGRADED_REASON_BYTES:
            raise ValueError("degraded reason is invalid")
    except UnicodeError as exc:
        raise ValueError("degraded reason is invalid") from exc
    pieces = value.split(";")
    if not pieces or pieces[0] != "member_publish_failed":
        raise ValueError("degraded reason prefix is invalid")
    result: dict[str, Any] = {"component": "group_ready_members", "operation": "publish", "facts": {}}
    seen: set[str] = set()
    order = {
        "version": 0,
        "reason_code": 1,
        "stage": 2,
        "check_id": 3,
        "exception_type": 4,
        "task_id": 5,
        "task_id_sha256": 5,
        "generation": 6,
        "group_name": 7,
        "group_name_sha256": 7,
        "errno": 8,
        "json_line": 9,
        "json_column": 10,
    }
    previous_order = -1
    previous_fact = ""
    facts_started = False
    for piece in pieces[1:]:
        key, separator, encoded = piece.partition("=")
        if not separator or not key or key in seen:
            raise ValueError("degraded reason field is invalid")
        if not seen and key != "version":
            raise ValueError("degraded reason version must be first")
        try:
            decoded = unquote(encoded, errors="strict")
            normalized = quote(decoded, safe="-._~")
        except UnicodeError as exc:
            raise ValueError("degraded reason encoding is invalid") from exc
        if normalized != encoded:
            raise ValueError("degraded reason encoding is not canonical")
        seen.add(key)
        if key.startswith("fact."):
            facts_started = True
            fact_name = key[5:]
            if not fact_name or fact_name <= previous_fact:
                raise ValueError("degraded reason facts are not sorted")
            previous_fact = fact_name
            result["facts"][key[5:]] = decoded
        else:
            if facts_started or key not in order:
                raise ValueError("degraded reason fields are not in canonical order")
            if order[key] < previous_order:
                raise ValueError("degraded reason fields are not in canonical order")
            previous_order = order[key]
            if key in {"task_id", "task_id_sha256"} and seen.intersection({"task_id", "task_id_sha256"}) == {
                "task_id",
                "task_id_sha256",
            }:
                raise ValueError("degraded reason task identity is duplicated")
            if key in {"group_name", "group_name_sha256"} and seen.intersection(
                {"group_name", "group_name_sha256"}
            ) == {"group_name", "group_name_sha256"}:
                raise ValueError("degraded reason Group identity is duplicated")
            if key in {"version", "generation", "errno", "json_line", "json_column"}:
                if not decoded.isdigit() or str(int(decoded)) != decoded:
                    raise ValueError("degraded reason integer is invalid")
                result[key] = int(decoded)
            else:
                result[key] = decoded
    required = {"version", "reason_code", "stage", "check_id", "exception_type", "generation"}
    if (
        not required.issubset(seen)
        or not ("task_id" in seen or "task_id_sha256" in seen)
        or result.get("version") != 1
        or result.get("reason_code") not in _REASON_CODES
    ):
        raise ValueError("degraded reason protocol identity is invalid")
    definition = CHECK_REGISTRY.get(result.get("check_id"))
    if definition is None or result.get("stage") != definition.stage or result["reason_code"] not in definition.reasons:
        raise ValueError("degraded reason check is invalid")
    for integer_key in ("generation", "errno", "json_line", "json_column"):
        if integer_key in result and not _valid_integer(result[integer_key]):
            raise ValueError("degraded reason integer is out of range")
    if not _EXCEPTION_PATTERN.fullmatch(result["exception_type"]):
        raise ValueError("degraded reason exception type is invalid")
    if "task_id" in result and (not _ID_PATTERN.fullmatch(result["task_id"]) or len(result["task_id"]) > 48):
        raise ValueError("degraded reason task identifier is invalid")
    if "task_id_sha256" in result and not re.fullmatch(r"[a-f0-9]{16}", result["task_id_sha256"]):
        raise ValueError("degraded reason task hash is invalid")
    if "group_name_sha256" in result and not re.fullmatch(r"[a-f0-9]{16}", result["group_name_sha256"]):
        raise ValueError("degraded reason Group hash is invalid")
    fact_rules = definition.reasons[result["reason_code"]]
    if len(result["facts"]) > MAX_FACTS:
        raise ValueError("degraded reason has too many facts")
    for key, decoded in result["facts"].items():
        rule = fact_rules.get(key)
        if rule is None:
            raise ValueError("degraded reason fact is not registered")
        if rule.kind == "integer":
            if not decoded.isdigit() or str(int(decoded)) != decoded or not _valid_integer(int(decoded)):
                raise ValueError("degraded reason fact integer is invalid")
            result["facts"][key] = int(decoded)
        elif rule.kind == "choice" and decoded not in rule.choices:
            raise ValueError("degraded reason fact value is invalid")
    if "group_name" in result and (not _ID_PATTERN.fullmatch(result["group_name"]) or len(result["group_name"]) > 24):
        raise ValueError("degraded reason group identifier is invalid")
    if "group_name" in result and (
        result["group_name"][0] in ".-" or result["group_name"] in {"experiments", "qqtools_internal"}
    ):
        raise ValueError("degraded reason group identifier is invalid")
    return result


def format_failure_diagnostic(value: object) -> str:
    """Return one concise human-safe line without exception or record contents."""
    diagnostic = validate_failure_diagnostic(value)
    fields = [
        f"component={diagnostic.component}",
        f"operation={diagnostic.operation}",
        f"stage={diagnostic.stage}",
        f"check={diagnostic.check_id}",
        f"reason={diagnostic.reason_code}",
        f"exception={diagnostic.exception_type}",
        f"task={diagnostic.task_id}",
        f"generation={diagnostic.generation}",
    ]
    if diagnostic.input_index is not None:
        fields.append(f"input_index={diagnostic.input_index}")
    if diagnostic.errno is not None:
        fields.append(f"errno={diagnostic.errno}")
    if diagnostic.json_line is not None:
        fields.append(f"json_line={diagnostic.json_line}")
    if diagnostic.json_column is not None:
        fields.append(f"json_column={diagnostic.json_column}")
    if diagnostic.facts:
        facts = ",".join(f"{key}={diagnostic.facts[key]}" for key in sorted(diagnostic.facts))
        fields.append(f"facts={facts}")
    return "Diagnostic: " + " ".join(fields)


__all__ = [
    "CHECK_REGISTRY",
    "MAX_DEGRADED_REASON_BYTES",
    "MAX_DIAGNOSTIC_BYTES",
    "PublicationTracker",
    "ReadyMemberCheckError",
    "ReadyMemberFailureDiagnostic",
    "ReadyMemberPublicationError",
    "diagnostic_for_exception",
    "fallback_diagnostic_for_exception",
    "format_failure_diagnostic",
    "parse_degraded_reason",
    "serialize_degraded_reason",
    "validate_failure_diagnostic",
]
