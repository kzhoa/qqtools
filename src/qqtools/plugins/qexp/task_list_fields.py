"""Field registry and parser for human ``qexp task list`` presentations."""

from __future__ import annotations

import difflib
import string
from collections.abc import Mapping

# Keep this order aligned with the public CLI help.  The mapping is deliberately
# the source of truth for accepted identifiers; there are no aliases.
FIELD_HEADINGS: Mapping[str, str] = {
    "task": "Task",
    "name": "Name",
    "state": "State",
    "requested-gpus": "GPUs",
    "group": "Group",
    "home": "Home",
    "queue": "Queue",
    "claimed-machine": "Claimed machine",
    "dependency": "Dependency",
    "reason": "Reason",
    "location": "Location",
    "overall-progress": "Overall",
    "activity": "Activity",
    "report-age": "Report age",
}

FIELD_PURPOSES: Mapping[str, str] = {
    "task": "stable Task identity (name and complete ID)",
    "name": "Task name",
    "state": "Task projection",
    "requested-gpus": "requested GPU count",
    "group": "Group name",
    "home": "home machine",
    "queue": "queue scope",
    "claimed-machine": "active claim machine",
    "dependency": "dependency gate and IDs",
    "reason": "Task-truth reason",
    "location": "selected current Attempt allocation",
    "overall-progress": "declared overall counter",
    "activity": "current stage, counter, and message",
    "report-age": "age since agent acceptance",
}

VIEWS: Mapping[str, tuple[str, ...] | None] = {
    "default": None,
    "progress": ("task", "state", "overall-progress", "activity", "report-age", "location"),
    "placement": ("task", "state", "requested-gpus", "home", "queue", "location"),
    "overall": ("task", "state", "overall-progress", "location"),
}

ENRICHED_FIELDS = frozenset({"location", "overall-progress", "activity", "report-age"})
_ASCII_WHITESPACE = string.whitespace


def _strip_ascii_whitespace(value: str) -> str:
    return value.strip(_ASCII_WHITESPACE)


def _unknown_field_message(identifier: str) -> str:
    matches = difflib.get_close_matches(identifier, FIELD_HEADINGS, n=2, cutoff=0.6)
    suggestion = ""
    if matches:
        best = matches[0]
        if len(matches) == 1 or difflib.SequenceMatcher(None, best, matches[1]).ratio() < 0.92:
            suggestion = f" Did you mean {best!r}?"
    alternatives = ", ".join(FIELD_HEADINGS)
    return f"unknown task field {identifier!r}; choose from: {alternatives}.{suggestion}"


def resolve_task_fields(view: str | None, fields: str | None) -> tuple[str, ...] | None:
    """Resolve one human view or explicit field list into ordered identifiers.

    ``None`` means the legacy no-option/default renderer.  ``default`` is an
    explicit human view but intentionally resolves to the same sentinel so its
    existing table remains byte-for-byte compatible.
    """
    if view is not None and fields is not None:
        raise ValueError("task list --view and --fields are mutually exclusive; use either a preset or fields.")
    if view is not None:
        if view not in VIEWS:
            alternatives = ", ".join(VIEWS)
            raise ValueError(f"unknown task list view {view!r}; choose from: {alternatives}.")
        return VIEWS[view]
    if fields is None:
        return None
    if not isinstance(fields, str):
        raise ValueError("task list --fields must be a comma-separated list of field identifiers.")

    values = fields.split(",")
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        identifier = _strip_ascii_whitespace(raw_value)
        if not identifier:
            raise ValueError("task list --fields cannot contain an empty field identifier.")
        if identifier not in FIELD_HEADINGS:
            raise ValueError(_unknown_field_message(identifier))
        if identifier in seen:
            raise ValueError(f"duplicate task field {identifier!r}; each field may appear only once.")
        seen.add(identifier)
        normalized.append(identifier)

    if "task" in seen:
        normalized.remove("task")
    return ("task", *normalized)


__all__ = ["ENRICHED_FIELDS", "FIELD_HEADINGS", "FIELD_PURPOSES", "VIEWS", "resolve_task_fields"]
