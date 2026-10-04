"""Tolerant progress contracts for resumable qexp upgrades."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

_SAFE_TOKEN = re.compile(r"[A-Za-z0-9._-]+\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PROGRESS_VERSION = 1
_PROGRESS_SCOPES = frozenset({"shared_project"})
_STAGE_STATES = frozenset({"inventory", "processing", "recounting", "restarted", "complete"})
_TOTAL_KINDS = frozenset({"snapshot_exact", "dynamic", "unknown"})
_PROGRESS_FIELDS = frozenset(
    {
        "version",
        "scope",
        "stage",
        "stage_label",
        "stage_state",
        "scan_epoch",
        "completed_units",
        "inventoried_units",
        "total_units",
        "total_kind",
        "unit",
        "remaining_stages",
        "last_progress_at",
        "current_item",
    }
)
_EVIDENCE_FIELDS = frozenset(
    {
        "version",
        "journal_revision",
        "identity",
        "activation_revision",
        "activation_cursor",
        "source_checkpoint",
    }
)
_SOURCE_CHECKPOINT_FIELDS = frozenset({"identity", "completed_bytes"})


def _require_dict(value: object, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    return value


def _require_exact_int(value: object, label: str, *, positive: bool = False) -> int:
    if type(value) is not int or (value <= 0 if positive else value < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return value


def _require_safe_token(value: object, label: str, limit: int) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > limit
        or value in {".", ".."}
        or _SAFE_TOKEN.fullmatch(value) is None
    ):
        raise ValueError(f"{label} must be a bounded safe token")
    return value


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256 identity")
    return value


def _require_timestamp(value: object, label: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value or len(value) > 64 or not value.isprintable():
        raise ValueError(f"{label} must be an ISO timestamp or null")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO timestamp or null") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    return value


def _validate_current_item(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    item = _require_dict(value, "progress current_item")
    for key in ("kind", "id"):
        if key not in item:
            raise ValueError(f"progress current_item.{key} is required")
    result = {
        "kind": _require_safe_token(item["kind"], "progress current_item.kind", 32),
        "id": _require_safe_token(item["id"], "progress current_item.id", 256),
    }
    if ("completed_bytes" in item) != ("total_bytes" in item):
        raise ValueError("progress current_item byte counters must appear together")
    for key in ("completed_bytes", "total_bytes"):
        if key in item:
            result[key] = _require_exact_int(item[key], f"progress current_item.{key}")
    if "completed_bytes" in result and "total_bytes" in result and result["completed_bytes"] > result["total_bytes"]:
        raise ValueError("progress current_item.completed_bytes exceeds total_bytes")
    if "restarted" in item:
        if type(item["restarted"]) is not bool:
            raise ValueError("progress current_item.restarted must be a boolean")
        result["restarted"] = item["restarted"]
    return result


def validate_progress(value: Any) -> dict[str, Any]:
    """Validate and sanitize one version-1 upgrade progress envelope."""
    progress = _require_dict(value, "progress")
    required = _PROGRESS_FIELDS - {"current_item"}
    missing = required - set(progress)
    if missing:
        raise ValueError(f"progress is missing required fields: {sorted(missing)}")
    if type(progress["version"]) is not int or progress["version"] != _PROGRESS_VERSION:
        raise ValueError("progress version must be exactly 1")
    if type(progress["scope"]) is not str or progress["scope"] not in _PROGRESS_SCOPES:
        raise ValueError("progress scope is unsupported")
    result: dict[str, Any] = {
        "version": _PROGRESS_VERSION,
        "scope": progress["scope"],
        "stage": _require_safe_token(progress["stage"], "progress stage", 64),
        "stage_label": progress["stage_label"],
        "stage_state": progress["stage_state"],
        "scan_epoch": _require_exact_int(progress["scan_epoch"], "progress scan_epoch", positive=True),
        "completed_units": progress["completed_units"],
        "inventoried_units": _require_exact_int(progress["inventoried_units"], "progress inventoried_units"),
        "total_units": progress["total_units"],
        "total_kind": progress["total_kind"],
        "unit": _require_safe_token(progress["unit"], "progress unit", 32),
        "remaining_stages": progress["remaining_stages"],
        "last_progress_at": _require_timestamp(progress["last_progress_at"], "progress last_progress_at"),
    }
    stage_label = result["stage_label"]
    if type(stage_label) is not str or not stage_label or len(stage_label) > 80 or not stage_label.isprintable():
        raise ValueError("progress stage_label must be a printable single-line string")
    if type(result["stage_state"]) is not str or result["stage_state"] not in _STAGE_STATES:
        raise ValueError("progress stage_state is unsupported")
    completed_units = result["completed_units"]
    if completed_units is not None:
        result["completed_units"] = _require_exact_int(completed_units, "progress completed_units")
    total_units = result["total_units"]
    if total_units is not None:
        result["total_units"] = _require_exact_int(total_units, "progress total_units")
    if completed_units is not None and total_units is not None and completed_units > total_units:
        raise ValueError("progress completed_units exceeds total_units")
    if type(result["total_kind"]) is not str or result["total_kind"] not in _TOTAL_KINDS:
        raise ValueError("progress total_kind is unsupported")
    if result["total_kind"] == "snapshot_exact" and total_units is None:
        raise ValueError("snapshot_exact progress requires total_units")
    if result["total_kind"] != "snapshot_exact" and total_units is not None:
        raise ValueError("dynamic or unknown progress cannot declare total_units")
    remaining_stages = result["remaining_stages"]
    if type(remaining_stages) is not list or len(remaining_stages) > 16:
        raise ValueError("progress remaining_stages must be a list of at most 16 stages")
    sanitized_stages: list[str] = []
    for stage in remaining_stages:
        sanitized_stages.append(_require_safe_token(stage, "progress remaining_stages item", 64))
    if len(set(sanitized_stages)) != len(sanitized_stages):
        raise ValueError("progress remaining_stages must be unique")
    result["remaining_stages"] = sanitized_stages
    if "current_item" in progress:
        result["current_item"] = _validate_current_item(progress["current_item"])
    return result


def _validate_evidence(value: Any, *, journal_revision: int | None = None) -> dict[str, Any]:
    evidence = _require_dict(value, "progress_evidence")
    required = _EVIDENCE_FIELDS
    missing = required - set(evidence)
    if missing:
        raise ValueError(f"progress_evidence is missing required fields: {sorted(missing)}")
    if type(evidence["version"]) is not int or evidence["version"] != _PROGRESS_VERSION:
        raise ValueError("progress_evidence version must be exactly 1")
    bound_revision = _require_exact_int(
        evidence["journal_revision"], "progress_evidence journal_revision", positive=True
    )
    if journal_revision is not None and (type(journal_revision) is not int or bound_revision != journal_revision):
        raise ValueError("progress_evidence journal_revision does not match its container")
    result: dict[str, Any] = {
        "version": _PROGRESS_VERSION,
        "journal_revision": bound_revision,
        "identity": _require_sha256(evidence["identity"], "progress_evidence identity"),
        "activation_revision": _require_exact_int(
            evidence["activation_revision"], "progress_evidence activation_revision", positive=True
        ),
        "activation_cursor": _require_exact_int(evidence["activation_cursor"], "progress_evidence activation_cursor"),
    }
    source_checkpoint = evidence["source_checkpoint"]
    if source_checkpoint is None:
        result["source_checkpoint"] = None
    else:
        checkpoint = _require_dict(source_checkpoint, "progress_evidence source_checkpoint")
        for key in _SOURCE_CHECKPOINT_FIELDS:
            if key not in checkpoint:
                raise ValueError(f"progress_evidence source_checkpoint.{key} is required")
        result["source_checkpoint"] = {
            "identity": _require_sha256(checkpoint["identity"], "progress_evidence source_checkpoint.identity"),
            "completed_bytes": _require_exact_int(
                checkpoint["completed_bytes"], "progress_evidence source_checkpoint.completed_bytes"
            ),
        }
    return result


def _check_source_binding(progress: dict[str, Any], evidence: dict[str, Any]) -> None:
    checkpoint = evidence["source_checkpoint"]
    item = progress.get("current_item")
    item_completed = item.get("completed_bytes") if isinstance(item, dict) else None
    if checkpoint is not None:
        if not isinstance(item, dict) or type(item_completed) is not int:
            raise ValueError("progress source checkpoint requires current_item.completed_bytes")
        if item_completed != checkpoint["completed_bytes"]:
            raise ValueError("progress source checkpoint does not match current_item")
    elif isinstance(item, dict) and "completed_bytes" in item:
        raise ValueError("current_item.completed_bytes requires a source checkpoint")


def validate_progress_evidence(
    progress: Any,
    evidence: Any,
    *,
    journal_revision: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and sanitize an envelope together with its revision witness."""
    sanitized_progress = validate_progress(progress)
    sanitized_evidence = _validate_evidence(evidence, journal_revision=journal_revision)
    _check_source_binding(sanitized_progress, sanitized_evidence)
    return sanitized_progress, sanitized_evidence


def _accepted_progress_pair(item: Any, journal_revision: int) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if type(item) is not dict or type(journal_revision) is not int or journal_revision <= 0 or item.get("in_flight"):
        return None
    try:
        return validate_progress_evidence(
            item.get("progress"), item.get("progress_evidence"), journal_revision=journal_revision
        )
    except (TypeError, ValueError):
        return None


def accepted_progress(item: Any, journal_revision: int) -> dict[str, Any] | None:
    """Return a migration's sanitized, revision-bound progress or ``None``."""
    accepted = _accepted_progress_pair(item, journal_revision)
    return accepted[0] if accepted is not None else None


def _status_progress_pair(status: Any) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if type(status) is not dict:
        return None
    candidates: list[tuple[Any, Any]] = []
    if ("progress" in status or "progress_evidence" in status) and not status.get("in_flight"):
        candidates.append((status.get("progress"), status.get("progress_evidence")))
    migrations = status.get("migrations")
    if type(migrations) is list:
        active = next(
            (item for item in migrations if type(item) is dict and item.get("state") not in {"completed", "idle"}),
            next((item for item in migrations if type(item) is dict), None),
        )
        if type(active) is dict and not active.get("in_flight"):
            candidates.append((active.get("progress"), active.get("progress_evidence")))
    for progress, evidence in candidates:
        try:
            return validate_progress_evidence(progress, evidence)
        except (TypeError, ValueError):
            continue
    return None


def semantic_delta(before_status: Any, after_status: Any) -> dict[str, Any] | None:
    """Compare two valid committed observations and return this-observation's semantic delta."""
    before = _status_progress_pair(before_status)
    after = _status_progress_pair(after_status)
    if before is None or after is None:
        return None
    before_progress, before_evidence = before
    after_progress, after_evidence = after
    if (
        before_evidence["identity"] != after_evidence["identity"]
        or before_progress["scan_epoch"] != after_progress["scan_epoch"]
        or before_progress["stage"] != after_progress["stage"]
        or before_progress["unit"] != after_progress["unit"]
    ):
        return None
    before_units = before_progress["completed_units"]
    after_units = after_progress["completed_units"]
    if type(before_units) is int and type(after_units) is int and after_units > before_units:
        return {
            "kind": "units",
            "stage": after_progress["stage"],
            "unit": after_progress["unit"],
            "before": before_units,
            "after": after_units,
            "total": after_progress["total_units"],
        }
    before_item = before_progress.get("current_item")
    after_item = after_progress.get("current_item")
    before_checkpoint = before_evidence["source_checkpoint"]
    after_checkpoint = after_evidence["source_checkpoint"]
    if (
        isinstance(before_item, dict)
        and isinstance(after_item, dict)
        and before_item.get("kind") == after_item.get("kind")
        and before_item.get("id") == after_item.get("id")
        and isinstance(before_checkpoint, dict)
        and isinstance(after_checkpoint, dict)
        and before_checkpoint["identity"] == after_checkpoint["identity"]
        and type(before_item.get("completed_bytes")) is int
        and type(after_item.get("completed_bytes")) is int
        and after_item["completed_bytes"] > before_item["completed_bytes"]
    ):
        return {
            "kind": "bytes",
            "stage": after_progress["stage"],
            "unit": after_progress["unit"],
            "before": before_item["completed_bytes"],
            "after": after_item["completed_bytes"],
            "total": after_item.get("total_bytes"),
            "item_kind": after_item["kind"],
            "item_id": after_item["id"],
        }
    return None


__all__ = ["accepted_progress", "semantic_delta", "validate_progress", "validate_progress_evidence"]
