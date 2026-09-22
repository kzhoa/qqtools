"""Pure typed-configuration output contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .core import OutputContract, OutputKind
from .primitives import _details, _mapping, _operation


def _render_named_operation(result: Mapping[str, Any], default_action: str) -> str:
    action = result.get("action", default_action)
    status = result.get("status", result.get("state", "completed"))
    return _operation(
        action,
        status,
        tuple(
            (label.replace("_", " ").capitalize(), value)
            for label, value in result.items()
            if label not in {"action", "status", "state"}
        ),
    )


def _render_section(name: str, value: Mapping[str, Any]) -> str:
    if value.get("status") == "error":
        error = value.get("error")
        message = error.get("message") if isinstance(error, Mapping) else error
        return _details(
            (
                ("Section", name),
                ("Scope", value.get("scope")),
                ("Source", value.get("source")),
                ("Status", "unavailable"),
                ("Error", message),
            )
        )
    values = value.get("effective_values", value.get("values"))
    return _details(
        (
            ("Section", name),
            ("Scope", value.get("scope")),
            ("Source", value.get("source")),
            ("Applies to", value.get("applies_to")),
            ("Values", values),
        )
    )


def _render_config(result: Mapping[str, Any], _presentation: Mapping[str, object]) -> str:
    """Render typed configuration without flattening policy diagnostics."""
    sections = result.get("sections")
    if isinstance(sections, Mapping):
        blocks = []
        for name, value in sections.items():
            if isinstance(value, Mapping):
                blocks.append(_render_section(str(name), value))
            else:
                blocks.append(_details((("Section", name), ("Status", "unavailable"), ("Error", value))))
        summary = _details(
            (
                ("Action", result.get("action", "show")),
                ("Scope", result.get("scope")),
                ("Complete", result.get("complete")),
            )
        )
        return "\n\n".join(item for item in (summary, *blocks) if item)
    if result.get("action") in {"set", "reset"}:
        changed_fields = result.get("changed_fields")
        outcome = result.get("outcome", "updated" if result.get("changed") else "no_change")
        return _details(
            (
                ("Action", result.get("action")),
                ("Section", result.get("section")),
                ("Scope", result.get("scope")),
                ("Outcome", outcome),
                ("Changed fields", changed_fields),
                ("Source", result.get("source")),
                ("Applies to", result.get("applies_to")),
                ("Effective values", result.get("effective_values", result.get("values"))),
            )
        )
    if result.get("section"):
        return _details(
            (
                ("Action", result.get("action", "show")),
                ("Section", result.get("section")),
                ("Scope", result.get("scope")),
                ("Source", result.get("source")),
                ("Applies to", result.get("applies_to")),
                ("Values", result.get("effective_values", result.get("values"))),
            )
        )
    return _render_named_operation(result, "config")


def _validate_config(result: Any) -> None:
    _mapping(result, "config payload")


CONTRACTS = {
    OutputKind.CONFIG: OutputContract(_validate_config, _render_config),
}

__all__ = ["CONTRACTS"]
