"""Pure formatting and validation primitives used by qexp output families."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


def _value(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, Mapping):
        return "none" if not value else ", ".join(f"{key}={_value(item)}" for key, item in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return "none" if not value else ", ".join(_value(item) for item in value)
    return str(value)


def _present(value: Any) -> bool:
    """Return whether a value carries a useful human-facing fact.

    ``False`` and numeric zero are meaningful state, while empty optional
    collections and ``None`` are presentation noise.  Keeping this rule in
    the primitive lets family renderers omit empty rows without changing the
    canonical JSON payload.
    """
    return value is not None and value != "" and value != () and value != [] and value != {}


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]], *, empty_message: str = "No results.") -> str:
    if not rows:
        return empty_message
    text_rows = [[_value(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in text_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    render_row = lambda row: "  ".join(value.ljust(widths[index]) for index, value in enumerate(row)).rstrip()
    return "\n".join([render_row(headers), render_row(["-" * width for width in widths]), *map(render_row, text_rows)])


def _details(*sections: Sequence[tuple[str, Any]]) -> str:
    """Render conditional detail sections while retaining section spacing."""
    rendered_sections = []
    for section in sections:
        rows = [f"{label}: {_value(value)}" for label, value in section if _present(value)]
        if rows:
            rendered_sections.append("\n".join(rows))
    return "\n\n".join(rendered_sections)


def _operation(action: Any, status: Any, fields: Sequence[tuple[str, Any]]) -> str:
    return _details((("Action", action), ("Status", status), *fields))


def _task_summary(tasks: Sequence[Mapping[str, Any]], *, group: str | None = None, machine: str | None = None) -> str:
    selected = [
        task
        for task in tasks
        if (group is None or task.get("group") == group) and (machine is None or task.get("home_machine") == machine)
    ]
    counts = Counter(task.get("phase") or "unknown" for task in selected)
    return _value(dict(sorted(counts.items())))


def _queue_summary(tasks: Sequence[Mapping[str, Any]], group: str | None) -> str:
    counts = Counter(task.get("queue_scope") or "unknown" for task in tasks if task.get("group") == group)
    return _value(dict(sorted(counts.items())))


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping, got {type(value).__name__}")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{label} must be a sequence, got {type(value).__name__}")
    return value


def _required(value: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in value:
        raise ValueError(f"{label} is missing required key {key!r}")
    return value[key]


def _required_mapping(value: Mapping[str, Any], key: str, label: str) -> Mapping[str, Any]:
    return _mapping(_required(value, key, label), f"{label}.{key}")


def _required_sequence(value: Mapping[str, Any], key: str, label: str) -> Sequence[Any]:
    return _sequence(_required(value, key, label), f"{label}.{key}")


def _required_int(value: Mapping[str, Any], key: str, label: str) -> int:
    item = _required(value, key, label)
    if type(item) is not int:
        raise TypeError(f"{label}.{key} must be an integer")
    return item


def _required_bool(value: Mapping[str, Any], key: str, label: str) -> bool:
    item = _required(value, key, label)
    if type(item) is not bool:
        raise TypeError(f"{label}.{key} must be a boolean")
    return item


__all__ = [
    "_details",
    "_mapping",
    "_operation",
    "_present",
    "_queue_summary",
    "_required",
    "_required_bool",
    "_required_int",
    "_required_mapping",
    "_required_sequence",
    "_sequence",
    "_table",
    "_task_summary",
    "_value",
]
