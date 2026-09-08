"""Bounded, typed diagnostics for ready-index degradation reasons."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Mapping

from qqtools.version import __version__

# QQTOOLS-COMPAT-0010
DIAGNOSTIC_VERSION = 1
MAX_REASON_BYTES = 1024
MAX_REASONS = 32
MAX_DYNAMIC_BYTES = 128
MAX_WRITERS = 16
UNOBSERVED = "unobserved"

_UNRESERVED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
_PERCENT_ESCAPE = re.compile(r"%[0-9A-Fa-f]{2}")
_NON_NEGATIVE = re.compile(r"0|[1-9][0-9]*")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9._-]+$")
_BOUNDED_IDENTIFIER = re.compile(r"^[A-Za-z0-9._-]+(?:~h-[0-9a-f]{12})?$")


def _safe_identifier_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        result.append(value if _IDENTIFIER.fullmatch(value) else "invalid_field")
    return sorted(set(result))


def _safe_identifier(value: Any) -> str:
    return value if isinstance(value, str) and _IDENTIFIER.fullmatch(value) else UNOBSERVED


safe_identifier = _safe_identifier

_COMMON_FIELDS = {"reader_machine", "reader_version", "truncated_fields"}
_REASON_FIELDS: dict[str, tuple[set[str], set[str]]] = {
    "marker_invalid": (
        {"stage", "task_id", "generation"},
        {"unexpected_fields", "missing_fields", "observed_schema_version", "supported_schema_versions"},
    ),
    "marker_missing": (
        {"stage", "task_id", "generation"},
        {"indexed", "task_projection", "active_claim"},
    ),
    "identity_mismatch": (
        {"stage", "task_id", "generation", "mismatch_fields"},
        {
            "expected_task_id",
            "expected_generation",
            "expected_queue_scope",
            "expected_home_machine",
            "expected_schema_version",
            "expected_lane",
            "observed_task_id",
            "observed_generation",
            "observed_queue_scope",
            "observed_home_machine",
            "observed_schema_version",
            "observed_lane",
        },
    ),
    "record_invalid": (
        {"object", "stage", "issue_code"},
        {
            "task_id",
            "generation",
            "object_id",
            "exception_type",
            "errno",
            "json_line",
            "json_column",
        },
    ),
    "catalog_invalid": (
        {"route", "page", "stage"},
        {
            "exception_type",
            "errno",
            "json_line",
            "json_column",
            "unexpected_fields",
            "missing_fields",
            "mismatch_fields",
        },
    ),
    "partition_invalid": (
        {"route", "partition", "stage"},
        {
            "exception_type",
            "errno",
            "json_line",
            "json_column",
            "unexpected_fields",
            "missing_fields",
            "mismatch_fields",
        },
    ),
    "partition_missing": (
        {"route", "partition", "stage"},
        {
            "exception_type",
            "errno",
            "json_line",
            "json_column",
            "unexpected_fields",
            "missing_fields",
            "mismatch_fields",
        },
    ),
    "build_invalid": (
        {"stage"},
        {"object", "exception_type", "errno", "json_line", "json_column", "task_id", "generation"},
    ),
    "build_failed": (
        {"stage", "exception_type"},
        {"object", "errno", "json_line", "json_column", "task_id", "generation"},
    ),
    "doctor_projection_issue": (
        {"stage", "issue_code", "task_id"},
        {"generation", "object"},
    ),
    "incompatible_active_writers": (
        {"stage", "incompatible_writers"},
        {"omitted_count"},
    ),
    "state_invalid": (
        {"stage", "exception_type"},
        {"errno", "json_line", "json_column"},
    ),
}

_PREFIX_REASONS = {
    "marker_corrupt": {"marker_invalid", "marker_missing", "identity_mismatch", "record_invalid"},
    "catalog_invalid": {"catalog_invalid"},
    "partition_invalid": {"partition_invalid"},
    "partition_missing": {"partition_missing"},
    "build_invalid": {"build_invalid"},
    "build_failed": {"build_failed"},
    "incompatible_active_writers": {"incompatible_active_writers"},
    "doctor_repair": {"doctor_projection_issue"},
    "ready_state_invalid": {"state_invalid"},
}

_LIST_FIELDS = {
    "unexpected_fields",
    "missing_fields",
    "supported_schema_versions",
    "mismatch_fields",
    "incompatible_writers",
    "truncated_fields",
}
_BOOLEAN_FIELDS = {"indexed", "active_claim"}
_INTEGER_FIELDS = {
    "generation",
    "expected_generation",
    "observed_generation",
    "expected_schema_version",
    "observed_schema_version",
    "page",
    "errno",
    "json_line",
    "json_column",
    "omitted_count",
}
_ENUM_FIELDS = {
    "queue_scope": {"home", "shared"},
    "lane": {"cpu", "gpu"},
    "task_projection": {"queued", "running", "succeeded", "failed", "cancelled", "blocked", UNOBSERVED},
    "stage": {
        "marker_parse",
        "marker_schema",
        "marker_identity",
        "marker_truth",
        "partition_recheck",
        "catalog_read",
        "catalog_schema",
        "partition_read",
        "partition_schema",
        "slot_identity",
        "build_state",
        "build_watermark",
        "build_phase",
        "build_cursor",
        "build_backfill",
        "build_audit",
        "primary_rebuild",
        "completion_writer_gate",
        "doctor_projection_audit",
        "reason_list",
        "state_record",
    },
    "object": {
        "marker",
        "task",
        "submission",
        "group",
        "dependency",
        "build",
        "watermark",
        "cursor",
        "primary",
        "state",
    },
    "issue_code": {
        "marker_invalid",
        "marker_identity_invalid",
        "task_invalid",
        "submission_identity_missing",
        "submission_missing",
        "submission_invalid",
        "submission_state_invalid",
        "group_missing",
        "group_invalid",
        "dependency_invalid",
        "marker_missing_indexed",
        "marker_missing_unindexed",
        "marker_unindexed",
        "marker_stale",
        "marker_missing",
        "marker_corrupt",
        "generation_superseded",
        "task_not_queued",
        "task_controlled",
        "submission_aborted",
    },
}
_LIST_IDENTIFIER_FIELDS = {
    "unexpected_fields",
    "missing_fields",
    "mismatch_fields",
    "incompatible_writers",
    "truncated_fields",
}
_EXCEPTION_TYPES = {
    "FileNotFoundError",
    "PermissionError",
    "OSError",
    "JSONDecodeError",
    "UnicodeDecodeError",
    "KeyError",
    "TypeError",
    "ValueError",
    "InvalidReasonList",
    "RuntimeError",
}


class InvalidReasonList(ValueError):
    """Raised when a ready state contains a non-string degradation list."""


@dataclass(frozen=True, slots=True)
class ReadyDiagnostic:
    """A validated reason code and observed, protocol-owned fields."""

    reason_code: str
    fields: tuple[tuple[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return dict(self.fields)


@dataclass(frozen=True, slots=True)
class ParsedReadyReason:
    """A strictly parsed v1 reason."""

    prefix: str
    diagnostic: ReadyDiagnostic
    normalized: str


def diagnostic(reason_code: str, **fields: Any) -> ReadyDiagnostic:
    """Construct a diagnostic using only the registered reason schema."""
    if reason_code not in _REASON_FIELDS:
        raise ValueError(f"unknown ready diagnostic reason code: {reason_code}")
    required, optional = _REASON_FIELDS[reason_code]
    allowed = required | optional | _COMMON_FIELDS
    if set(fields) - allowed:
        unknown = sorted(set(fields) - allowed)[0]
        raise ValueError(f"unknown field {unknown!r} for ready diagnostic {reason_code!r}")
    if not required.issubset(fields):
        missing = sorted(required - set(fields))[0]
        raise ValueError(f"missing field {missing!r} for ready diagnostic {reason_code!r}")
    _validate_fields(reason_code, fields, require_reader=False)
    return ReadyDiagnostic(reason_code, tuple(sorted(fields.items())))


def _validate_fields(reason_code: str, fields: Mapping[str, Any], *, require_reader: bool) -> None:
    required, optional = _REASON_FIELDS[reason_code]
    allowed = required | optional | _COMMON_FIELDS
    if require_reader and not {"reader_machine", "reader_version"}.issubset(fields):
        raise ValueError("ready v1 reason must include reader_machine and reader_version")
    if set(fields) - allowed:
        raise ValueError(f"unknown ready v1 diagnostic field: {sorted(set(fields) - allowed)[0]}")
    if not required.issubset(fields):
        raise ValueError(f"ready v1 diagnostic is missing {sorted(required - set(fields))[0]}")
    for key, value in fields.items():
        if key in _LIST_FIELDS:
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(f"ready diagnostic field {key!r} must be a string list")
            if key in _LIST_IDENTIFIER_FIELDS and any(
                not re.fullmatch(r"[A-Za-z0-9._~-]+", item) for item in value if item
            ):
                raise ValueError(f"ready diagnostic field {key!r} contains an invalid identifier")
            if key == "supported_schema_versions" and any(not _NON_NEGATIVE.fullmatch(item) for item in value):
                raise ValueError("ready diagnostic schema versions must be non-negative integers")
        elif key in _BOOLEAN_FIELDS:
            if type(value) is not bool:
                raise ValueError(f"ready diagnostic field {key!r} must be boolean")
        elif key in _INTEGER_FIELDS:
            if type(value) is not int or value < 0:
                raise ValueError(f"ready diagnostic field {key!r} must be a non-negative integer")
        elif key in _ENUM_FIELDS:
            if value not in _ENUM_FIELDS[key]:
                raise ValueError(f"ready diagnostic field {key!r} has an unsupported value")
        elif key == "exception_type":
            if value not in _EXCEPTION_TYPES:
                raise ValueError("ready diagnostic exception type is not allowlisted")
        elif key in {"task_id", "object_id"}:
            if not _BOUNDED_IDENTIFIER.fullmatch(value):
                raise ValueError(f"ready diagnostic field {key!r} is not a safe identifier")
        elif key == "route":
            if value != "shared" and (
                not isinstance(value, str)
                or not value.startswith("home.")
                or not _BOUNDED_IDENTIFIER.fullmatch(value.removeprefix("home."))
            ):
                raise ValueError("ready diagnostic route is invalid")
        elif key == "partition":
            if not isinstance(value, str) or not _BOUNDED_IDENTIFIER.fullmatch(value):
                raise ValueError("ready diagnostic partition is invalid")
        elif key in {"reader_machine", "reader_version", "object", "issue_code"}:
            if not isinstance(value, str) or not value:
                raise ValueError(f"ready diagnostic field {key!r} must be non-empty text")
        elif key.startswith("expected_") or key.startswith("observed_"):
            identity_key = key.removeprefix("expected_").removeprefix("observed_")
            if identity_key == "task_id":
                if value != UNOBSERVED and not _BOUNDED_IDENTIFIER.fullmatch(value):
                    raise ValueError(f"ready diagnostic identity field {key!r} is invalid")
            elif identity_key == "home_machine":
                if not isinstance(value, str) or not value:
                    raise ValueError(f"ready diagnostic identity field {key!r} is invalid")
            elif identity_key in {"queue_scope", "lane"}:
                if value not in {"home", "shared", "cpu", "gpu"}:
                    raise ValueError(f"ready diagnostic identity field {key!r} is invalid")
            elif identity_key in {"generation", "schema_version"}:
                if type(value) is not int or value < 0:
                    raise ValueError(f"ready diagnostic identity field {key!r} is invalid")
            elif not isinstance(value, (str, int)) or (type(value) is int and value < 0):
                raise ValueError(f"ready diagnostic identity field {key!r} is invalid")
        else:
            raise ValueError(f"ready diagnostic field {key!r} is not supported")


def _encode(value: str) -> str:
    encoded: list[str] = []
    for char in value:
        if char in _UNRESERVED:
            encoded.append(char)
        else:
            encoded.extend(f"%{byte:02X}" for byte in char.encode("utf-8"))
    return "".join(encoded)


def _decode(value: str) -> str:
    output = bytearray()
    index = 0
    while index < len(value):
        char = value[index]
        if char in _UNRESERVED:
            output.extend(char.encode("ascii"))
            index += 1
            continue
        match = _PERCENT_ESCAPE.match(value, index)
        if match is None:
            raise ValueError("invalid percent escape in ready reason")
        output.append(int(value[index + 1 : index + 3], 16))
        index += 3
    try:
        return output.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("ready reason contains invalid UTF-8") from exc


def _truncate(value: str, limit: int) -> tuple[str, bool]:
    if len(_encode(value).encode("ascii")) <= limit:
        return value, False
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    suffix = f"~h-{digest}"
    prefix = value
    while prefix and len(_encode(prefix + suffix).encode("ascii")) > limit:
        prefix = prefix[:-1]
    candidate = prefix + suffix
    if len(_encode(candidate).encode("ascii")) <= limit:
        return candidate, True
    return suffix[:limit], True


def _field_value(value: Any, *, limit: int = MAX_DYNAMIC_BYTES) -> tuple[str, bool]:
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, int):
        text = str(value)
        if len(text.encode("ascii")) > limit:
            raise ValueError("ready diagnostic integer exceeds its bounded field size")
    elif isinstance(value, list):
        encoded_items: list[str] = []
        truncated = False
        for item in sorted(value):
            item_text, was_clipped = _truncate(str(item), min(MAX_DYNAMIC_BYTES, limit))
            encoded_items.append(_encode(item_text))
            truncated = truncated or was_clipped
        while len(",".join(encoded_items).encode("utf-8")) > limit and encoded_items:
            encoded_items.pop()
            truncated = True
        return ",".join(encoded_items), truncated
    else:
        text = str(value)
    text, truncated = _truncate(text, limit)
    return _encode(text), truncated


def _render(prefix: str, reason_code: str, fields: Mapping[str, Any]) -> str:
    parts = [prefix, "v=1", f"reason={reason_code}"]
    for key in sorted(fields):
        parts.append(f"{key}={_field_value(fields[key])[0]}")
    return ";".join(parts)


def serialize_reason(cfg: object, prefix: str, value: ReadyDiagnostic) -> str:
    """Serialize a diagnostic with the current reader identity and hard bounds."""
    if prefix not in _PREFIX_REASONS or value.reason_code not in _PREFIX_REASONS[prefix]:
        raise ValueError(f"ready diagnostic {value.reason_code!r} is not valid for {prefix!r}")
    fields = value.as_dict()
    fields.setdefault("reader_machine", getattr(cfg, "machine_name", UNOBSERVED))
    fields.setdefault("reader_version", getattr(cfg, "reader_version", __version__))
    _validate_fields(value.reason_code, fields, require_reader=True)

    bounded: dict[str, Any] = {}
    truncated: list[str] = []
    required, _optional = _REASON_FIELDS[value.reason_code]
    for key in sorted(fields):
        field_value = fields[key]
        if type(field_value) is int and field_value >= 10**MAX_DYNAMIC_BYTES:
            if key in required:
                raise ValueError("ready diagnostic required integer exceeds its bounded field size")
            truncated.append(key)
            continue
        if isinstance(field_value, list):
            bounded_value: list[str] = []
            source_values = sorted(field_value)
            omitted_writers = 0
            if value.reason_code == "incompatible_active_writers" and key == "incompatible_writers":
                omitted_writers = max(0, len(source_values) - MAX_WRITERS)
                source_values = source_values[:MAX_WRITERS]
                if omitted_writers:
                    truncated.append(key)
            for item in source_values:
                clipped, was_clipped = _truncate(item, MAX_DYNAMIC_BYTES)
                bounded_value.append(clipped)
                if was_clipped and key not in truncated:
                    truncated.append(key)
            while len(",".join(_encode(item) for item in bounded_value).encode("utf-8")) > MAX_DYNAMIC_BYTES:
                bounded_value.pop()
                if value.reason_code == "incompatible_active_writers" and key == "incompatible_writers":
                    omitted_writers += 1
                truncated.append(key)
            if omitted_writers:
                bounded["omitted_count"] = int(bounded.get("omitted_count", 0)) + omitted_writers
            field_value = bounded_value
        elif isinstance(field_value, str):
            field_value, was_clipped = _truncate(field_value, MAX_DYNAMIC_BYTES)
            if was_clipped:
                truncated.append(key)
        elif key == "omitted_count" and key in bounded:
            field_value = int(field_value) + int(bounded.pop(key))
        bounded[key] = field_value

    # Required fields are retained; optional fields are admitted in stable order.
    required_keys = {"reader_machine", "reader_version"} | required
    retained = {key: bounded[key] for key in required_keys}
    for key in sorted(set(bounded) - required_keys - {"truncated_fields"}):
        candidate = dict(retained)
        candidate[key] = bounded[key]
        if len(_render(prefix, value.reason_code, candidate).encode("utf-8")) <= MAX_REASON_BYTES:
            retained[key] = bounded[key]
        elif key not in truncated:
            truncated.append(key)
    bounded = retained
    if truncated:
        # A deterministic, bounded marker for omitted or shortened fields.
        marker = sorted(set(truncated))
        while marker:
            candidate = dict(bounded)
            candidate["truncated_fields"] = marker
            if len(rendered := _render(prefix, value.reason_code, candidate).encode("utf-8")) <= MAX_REASON_BYTES:
                bounded = candidate
                break
            marker.pop()
        if not marker:
            bounded.pop("truncated_fields", None)
    rendered = _render(prefix, value.reason_code, bounded)
    if len(rendered.encode("utf-8")) > MAX_REASON_BYTES:
        # The protocol's required fields are bounded individually. This final guard
        # keeps the persistence boundary fail-closed if a future schema grows.
        raise ValueError("ready diagnostic required fields exceed 1024 bytes")
    return rendered


def _parse_field(key: str, encoded: str) -> Any:
    if key in _LIST_FIELDS:
        if encoded == "":
            return []
        return [_decode(item) for item in encoded.split(",")]
    if key in _BOOLEAN_FIELDS:
        if encoded not in {"true", "false"}:
            raise ValueError("invalid boolean in ready reason")
        return encoded == "true"
    if key in _INTEGER_FIELDS:
        if not _NON_NEGATIVE.fullmatch(encoded):
            raise ValueError("invalid integer in ready reason")
        return int(encoded)
    return _decode(encoded)


def parse_reason(reason: str) -> ParsedReadyReason:
    """Strictly parse and normalize a persisted v1 reason."""
    if not isinstance(reason, str) or len(reason.encode("utf-8")) > MAX_REASON_BYTES:
        raise ValueError("ready reason is not a bounded string")
    parts = reason.split(";")
    if len(parts) < 3 or parts[1] != "v=1" or not parts[0]:
        raise ValueError("ready reason envelope is invalid")
    prefix = parts[0]
    if prefix not in _PREFIX_REASONS:
        raise ValueError("unknown ready reason prefix")
    if not parts[2].startswith("reason="):
        raise ValueError("ready reason code must be the first field")
    reason_code = parts[2][len("reason=") :]
    if reason_code not in _REASON_FIELDS or reason_code not in _PREFIX_REASONS[prefix]:
        raise ValueError("unknown or incompatible ready reason code")
    fields: dict[str, Any] = {}
    for item in parts[3:]:
        if "=" not in item:
            raise ValueError("ready reason field is malformed")
        key, encoded = item.split("=", 1)
        if key in fields or not key or key == "reason" or ";" in key:
            raise ValueError("ready reason contains a duplicate or invalid key")
        fields[key] = _parse_field(key, encoded)
    _validate_fields(reason_code, fields, require_reader=True)
    normalized = _render(prefix, reason_code, fields)
    if normalized != reason:
        raise ValueError("ready reason is not canonically encoded")
    return ParsedReadyReason(prefix, ReadyDiagnostic(reason_code, tuple(sorted(fields.items()))), normalized)


# Descriptive aliases are kept at the module boundary so callers do not need to
# know that the persisted representation is a reason string.
serialize_ready_reason = serialize_reason
parse_ready_reason = parse_reason


def _legacy_identifier(value: str) -> bool:
    return bool(_IDENTIFIER.fullmatch(value))


def is_safe_legacy_reason(reason: str) -> bool:
    """Return whether a bounded pre-v1 reason is safe to show during transition."""
    if not isinstance(reason, str) or len(reason.encode("utf-8")) > MAX_REASON_BYTES:
        return False
    marker = re.fullmatch(r"marker_corrupt:([A-Za-z0-9._-]+)\.([0-9]+)", reason)
    if marker:
        return True
    route_page = re.fullmatch(r"(?:catalog_invalid):([^:]+):([0-9]+)", reason)
    if route_page:
        return (route_page.group(1) == "shared" or route_page.group(1).startswith("home.")) and _legacy_identifier(
            route_page.group(1).removeprefix("home.")
        )
    partition = re.fullmatch(r"partition_(?:missing|invalid):([^:]+):([A-Za-z0-9._-]+)", reason)
    if partition:
        route = partition.group(1)
        return (route == "shared" or route.startswith("home.")) and _legacy_identifier(route.removeprefix("home."))
    writers = re.fullmatch(r"incompatible_active_writers:([^,]+(?:,[^,]+)*)", reason)
    return bool(
        writers
        and len(writers.group(1).split(",")) <= MAX_WRITERS
        and all(_legacy_identifier(item) for item in writers.group(1).split(","))
    )


def _digest(reason: str) -> str:
    return hashlib.sha256(reason.encode("utf-8")).hexdigest()


def safe_reason_view(reasons: Any) -> list[str]:
    """Return only canonical v1 or allowlisted legacy reasons for display."""
    if not isinstance(reasons, list) or not all(isinstance(item, str) for item in reasons):
        return [
            serialize_reason(
                _UnobservedConfig(),
                "ready_state_invalid",
                diagnostic("state_invalid", stage="reason_list", exception_type="InvalidReasonList"),
            )
        ]
    visible: list[str] = []
    overflow: list[str] = []
    raw_overflow: list[str] = []
    for reason in reasons:
        try:
            visible.append(parse_reason(reason).normalized)
        except ValueError:
            if is_safe_legacy_reason(reason):
                visible.append(reason)
            else:
                label = "v1_reason_unavailable" if ";v=1;" in reason else "legacy_reason_unavailable"
                visible.append(f"{label};sha256={_digest(reason)}")
    if len(visible) > MAX_REASONS:
        overflow = visible[MAX_REASONS - 1 :]
        raw_overflow = reasons[MAX_REASONS - 1 :]
        visible = visible[: MAX_REASONS - 1]
        encoded = b"".join(item.encode("utf-8") for item in raw_overflow)
        visible.append(
            f"legacy_reason_overflow;omitted_count={len(overflow)};sha256={hashlib.sha256(encoded).hexdigest()}"
        )
    return visible


display_ready_reasons = safe_reason_view


class _UnobservedConfig:
    machine_name = UNOBSERVED
    reader_version = UNOBSERVED


def exception_fields(exc: BaseException) -> dict[str, Any]:
    """Extract allowlisted structural exception facts without exposing its message."""
    name = type(exc).__name__ if type(exc).__name__ in _EXCEPTION_TYPES else "RuntimeError"
    result: dict[str, Any] = {"exception_type": name}
    errno = getattr(exc, "errno", None)
    if type(errno) is int and errno >= 0:
        result["errno"] = errno
    if isinstance(exc, JSONDecodeError):
        result["json_line"] = exc.lineno
        result["json_column"] = exc.colno
    return result


def classification_diagnostic(
    reason: str,
    reference: Any,
    *,
    task: Any = None,
    marker: Mapping[str, Any] | None = None,
    exception: BaseException | None = None,
    stage: str = "marker_truth",
) -> ReadyDiagnostic:
    """Map an existing classifier reason to a bounded diagnostic schema."""
    task_id = getattr(reference, "task_id", None)
    generation = getattr(reference, "generation", None)
    identity: dict[str, Any] = {}
    if isinstance(task_id, str) and _IDENTIFIER.fullmatch(task_id):
        identity["task_id"] = task_id
    if type(generation) is int and generation >= 0:
        identity["generation"] = generation
    if reason in {"marker_invalid", "marker_identity_invalid"}:
        if reason == "marker_identity_invalid":
            fields = {"object": "marker", "stage": "marker_identity", "issue_code": reason, **identity}
            if exception:
                fields.update(exception_fields(exception))
            return diagnostic("record_invalid", **fields)
        fields = {
            "stage": "marker_schema",
            **identity,
            "unexpected_fields": [],
            "missing_fields": [],
        }
        if marker is not None:
            common = {
                "schema_version",
                "task_id",
                "generation",
                "source_transition",
                "source_revision",
                "target_revision",
                "queue_scope",
                "home_machine",
                "group_name",
                "submission_operation_id",
                "created_at",
            }
            expected = common | (
                {"requested_gpus"}
                if marker.get("lane") is None
                else {"lane", "requested_cpus" if marker.get("lane") == "cpu" else "requested_gpus"}
            )
            fields["unexpected_fields"] = _safe_identifier_list(sorted(set(marker) - expected))
            fields["missing_fields"] = _safe_identifier_list(sorted(expected - set(marker)))
            observed_schema_version = marker.get("schema_version")
            if type(observed_schema_version) is int and observed_schema_version >= 0:
                fields["observed_schema_version"] = observed_schema_version
            fields["supported_schema_versions"] = ["1"]
        return diagnostic("marker_invalid", **fields)
    if reason == "route_mismatch":
        expected_queue_scope = getattr(reference, "queue_scope", "shared")
        expected_home_machine = getattr(reference, "home_machine", UNOBSERVED)
        source = marker or {}
        if task is not None:
            observed_queue_scope = task.placement_runtime.get("queue_scope")
            observed_home_machine = task.placement_policy.get("home_machine")
        else:
            observed_queue_scope = source.get("queue_scope")
            observed_home_machine = source.get("home_machine")
        mismatch_fields = []
        fields = {
            "stage": "marker_identity",
            **identity,
            "mismatch_fields": mismatch_fields,
        }
        if observed_queue_scope != expected_queue_scope:
            mismatch_fields.append("queue_scope")
            fields["expected_queue_scope"] = expected_queue_scope
            if observed_queue_scope in {"home", "shared"}:
                fields["observed_queue_scope"] = observed_queue_scope
        if observed_home_machine != expected_home_machine:
            mismatch_fields.append("home_machine")
            fields["expected_home_machine"] = expected_home_machine
            if isinstance(observed_home_machine, str) and observed_home_machine:
                safe_home_machine = _safe_identifier(observed_home_machine)
                if safe_home_machine != UNOBSERVED:
                    fields["observed_home_machine"] = safe_home_machine
        return diagnostic("identity_mismatch", **fields)
    if reason.startswith("marker_missing") or reason == "marker_unindexed":
        return diagnostic(
            "marker_missing",
            stage="marker_truth",
            **identity,
            indexed=reason.endswith("indexed") and not reason.endswith("unindexed"),
            task_projection=(task.state.get("projection") if task is not None else UNOBSERVED),
            active_claim=bool(task.claim_control.get("active_claim")) if task is not None else False,
        )
    object_name = {
        "task_invalid": "task",
        "submission_identity_missing": "submission",
        "submission_missing": "submission",
        "submission_invalid": "submission",
        "submission_state_invalid": "submission",
        "group_missing": "group",
        "group_invalid": "group",
        "dependency_invalid": "dependency",
    }.get(reason)
    if object_name:
        fields = {"object": object_name, "stage": stage, "issue_code": reason, **identity}
        if exception:
            fields.update(exception_fields(exception))
        return diagnostic("record_invalid", **fields)
    return diagnostic(
        "record_invalid",
        object="marker",
        stage=stage,
        issue_code="marker_corrupt",
        **identity,
        **(exception_fields(exception) if exception else {}),
    )


def storage_diagnostic(
    reason_code: str,
    *,
    route: str,
    location: int | str,
    stage: str,
    exception: BaseException | None = None,
    unexpected_fields: list[str] | None = None,
    missing_fields: list[str] | None = None,
    mismatch_fields: list[str] | None = None,
) -> ReadyDiagnostic:
    """Construct a catalog or partition diagnostic from observed storage facts."""
    fields: dict[str, Any] = {"route": route, "stage": stage}
    if reason_code == "catalog_invalid":
        fields["page"] = location
    else:
        safe_partition = _safe_identifier(location)
        fields["partition"] = safe_partition if safe_partition != UNOBSERVED else "invalid_field"
        if safe_partition != location:
            mismatch_fields = [*(mismatch_fields or []), "partition"]
    if exception:
        fields.update(exception_fields(exception))
    if unexpected_fields is not None:
        fields["unexpected_fields"] = _safe_identifier_list(unexpected_fields)
    if missing_fields is not None:
        fields["missing_fields"] = _safe_identifier_list(missing_fields)
    if mismatch_fields is not None:
        fields["mismatch_fields"] = _safe_identifier_list(mismatch_fields)
    return diagnostic(reason_code, **fields)


def build_diagnostic(
    reason_code: str,
    *,
    stage: str,
    object_name: str | None = None,
    task_id: str | None = None,
    generation: int | None = None,
    exception: BaseException | None = None,
) -> ReadyDiagnostic:
    """Construct a build or state diagnostic from structural facts."""
    fields: dict[str, Any] = {"stage": stage}
    if object_name is not None:
        fields["object"] = object_name
    if task_id is not None:
        fields["task_id"] = task_id
    if generation is not None:
        fields["generation"] = generation
    if exception:
        fields.update(exception_fields(exception))
    if reason_code == "build_failed" and "exception_type" not in fields:
        fields["exception_type"] = "RuntimeError"
    return diagnostic(reason_code, **fields)


def writer_diagnostic(writers: list[str], *, stage: str) -> ReadyDiagnostic:
    """Construct the bounded active-writer compatibility diagnostic."""
    ordered = sorted(writers)
    retained = ordered[:MAX_WRITERS]
    return diagnostic(
        "incompatible_active_writers",
        stage=stage,
        incompatible_writers=retained,
        **({"omitted_count": len(ordered) - len(retained)} if len(ordered) > len(retained) else {}),
    )


def doctor_diagnostic(task_id: str, issue_code: str, *, generation: int | None = None) -> ReadyDiagnostic:
    """Construct the diagnostic retained for a doctor projection finding."""
    fields: dict[str, Any] = {"stage": "doctor_projection_audit", "issue_code": issue_code, "task_id": task_id}
    if generation is not None:
        fields["generation"] = generation
    return diagnostic("doctor_projection_issue", **fields)
