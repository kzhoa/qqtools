from __future__ import annotations

import pytest

from qqtools.plugins.qexp.agent.ready_cursor_validation import (
    validate_ready_cursor_position,
    validate_ready_cursor_transition,
)


def _position(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "catalog_page": 0,
        "partition": "00",
        "after_name": "task-a.1.json",
        "revision": 1,
    }
    value.update(updates)
    return value


def test_cursor_position_preserves_explicit_marker_compatibility_policy() -> None:
    legacy_marker = _position(after_name="nested/task-a.1.json")

    assert validate_ready_cursor_position(legacy_marker, "cursor")["after_name"] == "nested/task-a.1.json"
    with pytest.raises(ValueError, match="after_name"):
        validate_ready_cursor_position(legacy_marker, "cursor", require_safe_marker_name=True)


@pytest.mark.parametrize(
    "updates",
    [
        {"catalog_page": True},
        {"catalog_page": 1_000_000_000_000_001},
        {"partition": "../bad"},
        {"after_name": ""},
        {"revision": -1},
    ],
)
def test_cursor_position_rejects_invalid_shared_fields(updates: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        validate_ready_cursor_position(_position(**updates), "cursor")


def test_cursor_transition_requires_revision_advance_when_position_changes() -> None:
    observed = _position()
    changed = _position(after_name="task-b.1.json")

    with pytest.raises(ValueError, match="without advancing"):
        validate_ready_cursor_transition(observed, changed, "cursor")

    changed["revision"] = 2
    assert validate_ready_cursor_transition(observed, changed, "cursor")[1] == changed
