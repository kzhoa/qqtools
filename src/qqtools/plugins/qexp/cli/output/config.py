"""Pure typed-configuration output contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .core import OutputContract, OutputKind
from .primitives import _details, _mapping, _operation, _present, _section, _value


def _label(key: object) -> str:
    acronyms = {"gpu": "GPU", "id": "ID", "ids": "IDs", "tmux": "TMUX", "ttl": "TTL"}
    words = [acronyms.get(word, word) for word in str(key).split("_")]
    if words and words[0] not in acronyms.values():
        words[0] = words[0].capitalize()
    return " ".join(words)


def _render_values(values: object, *, indent: str = "  ") -> str:
    if not isinstance(values, Mapping):
        return f"{indent}{_value(values)}"
    lines: list[str] = []
    for key, item in values.items():
        if not _present(item):
            continue
        label = _label(key)
        if isinstance(item, Mapping):
            nested = _render_values(item, indent=indent + "  ")
            if nested:
                lines.extend((f"{indent}{label}", nested))
        else:
            lines.append(f"{indent}{label}: {_value(item)}")
    return "\n".join(lines)


def _values_section(title: str, values: object, *, omit: frozenset[str] = frozenset()) -> str:
    if not _present(values):
        return ""
    if isinstance(values, Mapping) and omit:
        values = {key: item for key, item in values.items() if key not in omit}
    body = _render_values(values)
    return f"{title}\n{body}" if body else ""


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
    summary = _section(
        f"Configuration: {name}",
        (
            ("Scope", value.get("scope")),
            ("Source", value.get("source")),
            ("Applies to", value.get("applies_to")),
        ),
    )
    return "\n\n".join(
        item for item in (summary, _values_section("Values", values, omit=frozenset({"source", "applies_to"}))) if item
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
        summary = _section(
            "Configuration",
            (
                ("Action", result.get("action", "show")),
                ("Section", result.get("section")),
                ("Scope", result.get("scope")),
                ("Source", result.get("source")),
                ("Applies to", result.get("applies_to")),
            ),
        )
        return "\n\n".join(
            item
            for item in (
                summary,
                _values_section(
                    "Values",
                    result.get("effective_values", result.get("values")),
                    omit=frozenset({"source", "applies_to"}),
                ),
                _values_section("Retention", result.get("retention")),
            )
            if item
        )
    return _render_named_operation(result, "config")


def _validate_config(result: Any) -> None:
    _mapping(result, "config payload")


CONTRACTS = {
    OutputKind.CONFIG: OutputContract(_validate_config, _render_config),
}

__all__ = ["CONTRACTS"]
