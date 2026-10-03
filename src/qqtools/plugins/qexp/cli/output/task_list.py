"""Pure, width-aware human rendering for selected Task list fields."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ...task_list_fields import FIELD_HEADINGS

try:
    from ...progress_selection import selected_progress_observation as _select_progress_observation
except ImportError:
    _select_progress_observation = None


_EM_DASH = "—"
_ELLIPSIS = "…"
_TERMINAL_PHASES = frozenset({"succeeded", "failed", "cancelled", "completed"})
_UNAVAILABLE_EXPLANATIONS = {
    "cleanup": "Cleanup in progress",
    "attempt_unreadable": "Could not read observation",
    "read_failed": "Could not read observation",
    "invalid_snapshot": "Invalid report",
    "identity_mismatch": "Could not verify current Attempt",
    "changed_during_read": "Task changed; rerun this command",
    "concurrent_transition": "Task changed; rerun this command",
}


@dataclass
class _RenderContext:
    """Mutable bookkeeping for one pure render operation."""

    presentation: Mapping[str, object]
    shortened: bool = False
    notes: list[str] = field(default_factory=list)
    _note_keys: set[str] = field(default_factory=set)
    diagnostics: list[tuple[str, str, str]] = field(default_factory=list)
    _diagnostic_numbers: dict[tuple[str, str], int] = field(default_factory=dict)

    def add_note(self, key: str, text: str) -> None:
        if key not in self._note_keys:
            self._note_keys.add(key)
            self.notes.append(text)

    def diagnostic(self, reason: object, *, category: str = "observation") -> str:
        code = _bounded_reason(reason)
        key = (category, code)
        number = self._diagnostic_numbers.get(key)
        if number is None:
            number = len(self.diagnostics) + 1
            self._diagnostic_numbers[key] = number
            if category == "policy":
                text = f"Reporting policy could not be verified ({code})."
            else:
                explanation = _UNAVAILABLE_EXPLANATIONS.get(code, "Could not read observation")
                text = f"{explanation} ({code})."
            self.diagnostics.append((category, code, text))
        return f"Unavailable [{number}]"

    @property
    def details_command(self) -> str:
        command = self.presentation.get("details_command")
        return _sanitize(command) if isinstance(command, str) and command else "qexp task show TASK_ID --details"


@dataclass(frozen=True)
class _RenderedRow:
    values: tuple[str, ...]
    identity: tuple[str | None, str]
    task_id: str
    continuation_reason: str | None


def _bounded_reason(value: object) -> str:
    if isinstance(value, str) and value and all(char.isalnum() or char in "_-" for char in value):
        return value
    return "read_failed"


def _sanitize(value: object) -> str:
    """Return printable text with line separators and terminal controls escaped."""
    if value is None or value == "":
        return _EM_DASH
    text = str(value).replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    text = text.replace("\u2028", " ").replace("\u2029", " ")
    output: list[str] = []
    for char in text:
        codepoint = ord(char)
        category = unicodedata.category(char)
        if char == "\t":
            output.append("\\t")
        elif codepoint == 0x1B:
            output.append("\\x1b")
        elif char == "\u200d" or unicodedata.combining(char) or _is_variation_selector(char):
            output.append(char)
        elif codepoint < 0x20 or 0x7F <= codepoint <= 0x9F or category.startswith("C"):
            output.append(f"\\x{codepoint:02x}")
        else:
            output.append(char)
    return "".join(output)


def _is_variation_selector(char: str) -> bool:
    codepoint = ord(char)
    return 0xFE00 <= codepoint <= 0xFE0F or 0xE0100 <= codepoint <= 0xE01EF


def _is_emoji_modifier(char: str) -> bool:
    return 0x1F3FB <= ord(char) <= 0x1F3FF


def _graphemes(text: str) -> list[str]:
    """Group combining marks, variation selectors, and ZWJ sequences."""
    clusters: list[str] = []
    current = ""
    joined = False
    regional_count = 0
    for char in text:
        if not current:
            current = char
            joined = False
            regional_count = 1 if 0x1F1E6 <= ord(char) <= 0x1F1FF else 0
            continue
        if joined:
            current += char
            joined = False
            continue
        if char == "\u200d":
            current += char
            joined = True
            continue
        if (
            unicodedata.combining(char)
            or _is_variation_selector(char)
            or _is_emoji_modifier(char)
            or unicodedata.category(char) in {"Mn", "Mc", "Me"}
        ):
            current += char
            continue
        if 0x1F1E6 <= ord(char) <= 0x1F1FF and regional_count == 1:
            current += char
            regional_count = 2
            continue
        clusters.append(current)
        current = char
        regional_count = 1 if 0x1F1E6 <= ord(char) <= 0x1F1FF else 0
    if current:
        clusters.append(current)
    return clusters


def _char_width(char: str) -> int:
    if char == "\u200d" or unicodedata.combining(char) or _is_variation_selector(char) or _is_emoji_modifier(char):
        return 0
    if unicodedata.category(char).startswith("C"):
        return 0
    return 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1


def _cluster_width(cluster: str) -> int:
    if "\u200d" in cluster:
        return max((_char_width(char) for char in cluster), default=1)
    return sum(_char_width(char) for char in cluster)


def _display_width(text: str) -> int:
    return sum(_cluster_width(cluster) for cluster in _graphemes(text))


def _limit_text(text: str, limit: int, context: _RenderContext) -> str:
    if _display_width(text) <= limit:
        return text
    context.shortened = True
    if limit <= _display_width(_ELLIPSIS):
        return _ELLIPSIS
    remaining = limit - _display_width(_ELLIPSIS)
    selected: list[str] = []
    used = 0
    for cluster in _graphemes(text):
        width = _cluster_width(cluster)
        if used + width > remaining:
            break
        selected.append(cluster)
        used += width
    return "".join(selected) + _ELLIPSIS


def _human(value: object) -> str:
    if value is None or value == "":
        return _EM_DASH
    if isinstance(value, bool):
        return "yes" if value else "no"
    return _sanitize(value)


def _text_field(value: object, limit: int, context: _RenderContext) -> str:
    return _limit_text(_sanitize(value), limit, context)


def _is_mapping(value: object) -> bool:
    return isinstance(value, Mapping)


def _phase(row: Mapping[str, object]) -> str:
    value = row.get("phase", row.get("state"))
    if isinstance(value, Mapping):
        value = value.get("projection")
    return _human(value)


def _task_identity(row: Mapping[str, object], context: _RenderContext) -> tuple[str, tuple[str | None, str]]:
    task_id = _sanitize(row.get("task_id"))
    raw_name = row.get("name")
    name = None if raw_name is None or raw_name == "" else _limit_text(_sanitize(raw_name), 32, context)
    if name is None:
        return task_id, (None, task_id)
    return f"{name} ({task_id})", (name, task_id)


def _gpu_numbers(value: object) -> list[tuple[int | None, str]]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        values: Sequence[object] = value.split(",")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = value
    else:
        values = (value,)
    result: list[tuple[int | None, str]] = []
    for item in values:
        text = _sanitize(item).strip()
        if not text:
            continue
        if text[:3].upper() == "GPU":
            text = text[3:]
        try:
            number = int(text, 10)
        except (TypeError, ValueError):
            number = None
        result.append((number, text if number is None else str(number)))
    result.sort(key=lambda item: (item[0] is None, item[0] if item[0] is not None else item[1]))
    return result


def _location_value(row: Mapping[str, object], context: _RenderContext) -> str:
    location = row.get("location")
    if not isinstance(location, Mapping):
        return _EM_DASH
    status = location.get("status")
    if status == "absent":
        return _EM_DASH
    if status != "available":
        return context.diagnostic(location.get("reason"))
    machine = _human(location.get("machine_name"))
    assigned = _gpu_numbers(location.get("assigned_gpus"))
    if assigned:
        gpu_text = ",".join(text for _number, text in assigned)
        machine = f"{machine} · GPU{gpu_text}"
    if location.get("orphaned") is True:
        machine += " (orphaned)"
    return machine


def _dependency_value(row: Mapping[str, object]) -> str:
    state = _human(row.get("dependency_state"))
    dependencies = row.get("depends_on_task_ids")
    if isinstance(dependencies, Sequence) and not isinstance(dependencies, (str, bytes, bytearray)) and dependencies:
        return f"{state}: {', '.join(_sanitize(item) for item in dependencies)}"
    return state


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError, OverflowError):
        return None
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        return None
    return stamp.astimezone(timezone.utc)


def _observation_status(observation: Mapping[str, object] | None) -> str:
    if not isinstance(observation, Mapping):
        return "missing"
    status = observation.get("status")
    state = observation.get("observation_state")
    if status == "available" or (state == "available" and "progress" in observation):
        return "available"
    if status in {"pending", "no_report", "absent"} or state in {"pending", "no_report", "absent"}:
        return "missing"
    if status in {"unavailable", "error", "invalid"} or state in {"unavailable", "error", "invalid"}:
        return "unavailable"
    return "missing"


def _fallback_select_progress(
    progress: Mapping[str, object] | None,
    progress_extended: Mapping[str, object] | None,
    progress_scoped: Mapping[str, object] | None,
    selected_version: object,
) -> Mapping[str, object] | None:
    selected = {
        1: progress,
        2: progress_extended,
        3: progress_scoped,
    }.get(selected_version)
    if _observation_status(selected) == "available":
        return selected
    for candidate in (progress_scoped, progress_extended, progress):
        if _observation_status(candidate) == "available":
            return candidate
    if isinstance(selected, Mapping):
        return selected
    for candidate in (progress_scoped, progress_extended, progress):
        if isinstance(candidate, Mapping):
            return candidate
    return None


def _selected_observation(row: Mapping[str, object]) -> Mapping[str, object] | None:
    progress = row.get("progress") if isinstance(row.get("progress"), Mapping) else None
    extended = row.get("progress_extended") if isinstance(row.get("progress_extended"), Mapping) else None
    scoped = row.get("progress_scoped") if isinstance(row.get("progress_scoped"), Mapping) else None
    selected_version = row.get("selected_progress_protocol_version", row.get("selected_progress_version"))
    selector = _select_progress_observation
    if selector is not None:
        try:
            selected = selector(progress, extended, scoped, selected_version)
        except TypeError:
            try:
                selected = selector(progress, extended, selected_version)
            except TypeError:
                selected = None
        if isinstance(selected, Mapping):
            fallback = _fallback_select_progress(progress, extended, scoped, selected_version)
            if _observation_status(selected) != "available" and _observation_status(fallback) == "available":
                return fallback
            return selected
    return _fallback_select_progress(progress, extended, scoped, selected_version)


def _progress_parts(
    observation: Mapping[str, object],
) -> tuple[Mapping[str, object] | None, Mapping[str, object] | None]:
    payload = observation.get("progress")
    if not isinstance(payload, Mapping):
        payload = observation
    activity = payload.get("activity")
    if not isinstance(activity, Mapping):
        activity = payload if any(key in payload for key in ("stage", "current", "total", "unit", "message")) else None
    overall = payload.get("overall")
    if not isinstance(overall, Mapping):
        overall = observation.get("overall") if isinstance(observation.get("overall"), Mapping) else None
    return activity, overall


def _format_counter(counter: Mapping[str, object], *, include_label: bool = True) -> str:
    current = counter.get("current") if type(counter.get("current")) is int else None
    total = counter.get("total") if type(counter.get("total")) is int else None
    unit_value = counter.get("unit")
    unit = _sanitize(unit_value) if unit_value not in {None, ""} else ""
    if current is None:
        count = f"?/{total:,}" if total is not None else "?"
    elif total is not None:
        count = f"{current:,}/{total:,}"
    else:
        count = f"{current:,}"
    if unit:
        count = f"{count} {unit}"
    percent: str | None = None
    if current is not None and total is not None and total > 0:
        value = (100 * current + total // 2) // total
        if value >= 100 and current < total:
            percent = "<100%"
        else:
            percent = f"{value}%"
    text = f"{percent} · {count}" if percent is not None else count
    label = counter.get("label")
    if include_label and label not in {None, ""}:
        text = f"{_sanitize(label)}: {text}"
    return text


def _format_activity(activity: Mapping[str, object], context: _RenderContext) -> str:
    stage = _human(activity.get("stage"))
    has_counter = any(key in activity and activity.get(key) is not None for key in ("current", "total"))
    if has_counter:
        counter = {
            "current": activity.get("current"),
            "total": activity.get("total"),
            "unit": activity.get("unit"),
        }
        text = f"{stage} · {_format_counter(counter, include_label=False)}"
    else:
        text = stage
    message = activity.get("message")
    if message not in {None, ""}:
        text += f" · {_text_field(message, 48, context)}"
    return text


def _reporting_policy(row: Mapping[str, object], context: _RenderContext) -> str | None:
    policy = row.get("reporting_policy")
    if not isinstance(policy, Mapping):
        return None
    state = policy.get("state")
    if state == "unknown":
        context.diagnostic(policy.get("reason"), category="policy")
    return state if isinstance(state, str) else None


def _absence_value(row: Mapping[str, object], observation: Mapping[str, object] | None) -> str:
    phase = row.get("phase", row.get("state"))
    if isinstance(phase, Mapping):
        phase = phase.get("projection")
    if phase == "queued" and row.get("current_attempt_id") is None:
        return "Not started"
    if phase in _TERMINAL_PHASES:
        return "No report recorded"
    return "No report yet"


def _progress_value(
    row: Mapping[str, object],
    field_name: str,
    context: _RenderContext,
    row_state: dict[str, object],
) -> str:
    policy_state = _reporting_policy(row, context)
    observation = row_state.get("observation")
    if observation is None:
        observation = _selected_observation(row)
        row_state["observation"] = observation
    status = _observation_status(observation if isinstance(observation, Mapping) else None)
    if status == "unavailable":
        reason = observation.get("reason") if isinstance(observation, Mapping) else None
        return context.diagnostic(reason)
    if field_name == "overall-progress" and policy_state == "disabled":
        context.add_note(
            "not_enabled",
            "Overall reporting not enabled for this Task; future submissions may use --live-progress, "
            "which does not retrofit an existing process.",
        )
        return "Not enabled"
    if status != "available":
        if field_name == "report-age":
            return _EM_DASH
        return _absence_value(row, observation if isinstance(observation, Mapping) else None)
    activity, overall = _progress_parts(observation)
    if field_name == "activity":
        if activity is None:
            return _absence_value(row, observation)
        return _format_activity(activity, context)
    if field_name == "overall-progress":
        if overall is None:
            context.add_note("not_provided", "The selected report has no overall counter.")
            return "Not provided"
        return _format_counter(overall)

    reported_at = _parse_timestamp(observation.get("reported_at"))
    if reported_at is None:
        return context.diagnostic("invalid_snapshot")
    observation_stamp = _parse_timestamp(row.get("observation_time"))
    if observation_stamp is None:
        observation_stamp = _parse_timestamp(context.presentation.get("observation_time"))
    if observation_stamp is None:
        return context.diagnostic("invalid_snapshot")
    seconds = (observation_stamp - reported_at).total_seconds()
    if seconds < 0:
        context.add_note("clock_difference", "Clock difference: the selected report timestamp is in the future.")
        return "Clock difference"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"


def _value_for_field(
    row: Mapping[str, object],
    field_name: str,
    context: _RenderContext,
    row_state: dict[str, object],
) -> str:
    if field_name == "task":
        task_text, identity = row_state["identity"]
        return task_text
    if field_name == "name":
        return _text_field(row.get("name"), 32, context)
    if field_name == "state":
        return _phase(row)
    if field_name == "requested-gpus":
        return _human(row.get("gpus", row.get("requested_gpus")))
    if field_name == "group":
        return _human(row.get("group", row.get("group_name")))
    if field_name == "home":
        return _human(row.get("home_machine"))
    if field_name == "queue":
        return _human(row.get("queue_scope"))
    if field_name == "claimed-machine":
        return _human(row.get("claim_machine"))
    if field_name == "dependency":
        return _dependency_value(row)
    if field_name == "reason":
        return _text_field(row.get("reason"), 48, context)
    if field_name == "location":
        return _location_value(row, context)
    if field_name in {"overall-progress", "activity", "report-age"}:
        return _progress_value(row, field_name, context, row_state)
    return _EM_DASH


def _render_rows(
    tasks: Sequence[Mapping[str, object]], fields: tuple[str, ...], context: _RenderContext
) -> list[_RenderedRow]:
    rendered: list[_RenderedRow] = []
    for task in tasks:
        identity_text, identity = _task_identity(task, context)
        row_state: dict[str, object] = {"identity": (identity_text, identity), "observation": None}
        values = tuple(_value_for_field(task, field_name, context, row_state) for field_name in fields)
        reason = task.get("reason")
        phase = task.get("phase", task.get("state"))
        if isinstance(phase, Mapping):
            phase = phase.get("projection")
        continuation = (
            _text_field(reason, 48, context)
            if "state" in fields and "reason" not in fields and phase in {"blocked", "failed"} and reason
            else None
        )
        rendered.append(_RenderedRow(values, identity, identity[1], continuation))
    return rendered


def _wrap_text(text: str, width: int) -> list[str]:
    if width <= 0:
        return [text]
    if _display_width(text) <= width:
        return [text]
    lines: list[str] = []
    current = ""
    for word in text.split(" "):
        candidate = f"{current} {word}" if current else word
        if _display_width(candidate) <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
        while _display_width(current) > width:
            clusters = _graphemes(current)
            used = 0
            split_at = 0
            for index, cluster in enumerate(clusters):
                cluster_width = _cluster_width(cluster)
                if used + cluster_width > width:
                    break
                used += cluster_width
                split_at = index + 1
            if split_at == 0:
                split_at = 1
            lines.append("".join(clusters[:split_at]))
            current = "".join(clusters[split_at:])
    if current or not lines:
        lines.append(current)
    return lines


def _identity_lines(row: _RenderedRow, width: int) -> list[str]:
    name, task_id = row.identity
    if name is None:
        return [task_id]
    compact = f"{name} ({task_id})"
    if _display_width(compact) <= width:
        return [compact]
    return [*_wrap_text(name, width), task_id]


def _table_width(widths: Sequence[int]) -> int:
    return sum(widths) + 2 * max(0, len(widths) - 1)


def _pad(value: str, width: int) -> str:
    return value + (" " * max(0, width - _display_width(value)))


def _render_plain_table(fields: tuple[str, ...], rows: Sequence[_RenderedRow]) -> str:
    if not rows:
        return "No Tasks."
    headers = [FIELD_HEADINGS.get(field_name, field_name) for field_name in fields]
    widths = [_display_width(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row.values):
            widths[index] = max(widths[index], _display_width(value))

    def render_line(values: Sequence[str]) -> str:
        return "  ".join(_pad(value, widths[index]) for index, value in enumerate(values)).rstrip()

    lines = [render_line(headers), render_line(["-" * width for width in widths])]
    for row in rows:
        lines.append(render_line(row.values))
        if row.continuation_reason is not None:
            lines.append(f"Reason: {row.continuation_reason}")
    return "\n".join(lines)


def _render_wrapped_table(fields: tuple[str, ...], rows: Sequence[_RenderedRow], widths: Sequence[int]) -> str:
    headers = [FIELD_HEADINGS.get(field_name, field_name) for field_name in fields]
    wrapped_headers = [_wrap_text(header, widths[index]) for index, header in enumerate(headers)]
    lines = [
        "  ".join(
            _pad(parts[line] if line < len(parts) else "", widths[index]) for index, parts in enumerate(wrapped_headers)
        ).rstrip()
        for line in range(max(map(len, wrapped_headers)))
    ]
    lines.append("  ".join("-" * widths[index] for index in range(len(headers))).rstrip())
    for row in rows:
        wrapped: list[list[str]] = []
        for index, value in enumerate(row.values):
            wrapped.append(_identity_lines(row, widths[index]) if index == 0 else _wrap_text(value, widths[index]))
        height = max(len(value) for value in wrapped)
        for line_index in range(height):
            cells = [value[line_index] if line_index < len(value) else "" for value in wrapped]
            lines.append("  ".join(_pad(cells[index], widths[index]) for index in range(len(cells))).rstrip())
        if row.continuation_reason is not None:
            lines.append(f"Reason: {row.continuation_reason}")
    return "\n".join(lines)


def _render_blocks(fields: tuple[str, ...], rows: Sequence[_RenderedRow], terminal_width: int) -> str:
    blocks: list[str] = []
    for row in rows:
        lines: list[str] = []
        for field_name, value in zip(fields, row.values, strict=True):
            label = FIELD_HEADINGS.get(field_name, field_name)
            available = max(1, terminal_width - _display_width(label) - 2)
            if field_name == "task":
                value_lines = _identity_lines(row, available)
            else:
                value_lines = _wrap_text(value, available)
            lines.append(f"{label}: {value_lines[0]}")
            lines.extend(f"{' ' * (_display_width(label) + 2)}{part}" for part in value_lines[1:])
        if row.continuation_reason is not None:
            lines.append(f"Reason: {row.continuation_reason}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _render_table_or_blocks(fields: tuple[str, ...], rows: Sequence[_RenderedRow], terminal_width: int | None) -> str:
    if not rows:
        return "No Tasks."
    if terminal_width is None or terminal_width < 1:
        return _render_plain_table(fields, rows)
    headers = [FIELD_HEADINGS.get(field_name, field_name) for field_name in fields]
    widths = [_display_width(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row.values):
            widths[index] = max(widths[index], _display_width(value))
    widths[0] = min(widths[0], terminal_width)
    minimums = [min(12, width) for width in widths]
    while _table_width(widths) > terminal_width:
        candidates = [index for index in range(1, len(widths)) if widths[index] > minimums[index]]
        if not candidates:
            break
        index = max(candidates, key=lambda item: widths[item])
        widths[index] -= 1
    if _table_width(widths) > terminal_width:
        return _render_blocks(fields, rows, terminal_width)
    return _render_wrapped_table(fields, rows, widths)


def _footer(context: _RenderContext) -> list[str]:
    lines: list[str] = []
    if context.diagnostics:
        lines.append("Diagnostics:")
        for number, (category, _code, text) in enumerate(context.diagnostics, start=1):
            if category == "policy":
                lines.append(f"  Reporting policy [{number}]: {text}")
            else:
                lines.append(f"  Observation [{number}]: {text}")
    lines.extend(context.notes)
    if context.shortened or context.diagnostics:
        lines.append(f"Details: {context.details_command}")
    return lines


def render_task_list(tasks: Sequence[Mapping[str, object]], presentation: Mapping[str, object]) -> str:
    """Render selected human Task fields without terminal or domain I/O."""
    fields_value = presentation.get("fields")
    fields = (
        tuple(fields_value) if isinstance(fields_value, Sequence) and not isinstance(fields_value, str) else ("task",)
    )
    context = _RenderContext(presentation)
    rows = _render_rows(tasks, fields, context)
    if "report-age" in fields:
        context.add_note("report_age", "Report age: since agent acceptance; not a heartbeat.")
    rendered = _render_table_or_blocks(fields, rows, presentation.get("terminal_width"))
    footer = _footer(context)
    return "\n".join([rendered, *footer]) if footer else rendered


def render_task_page(page: Mapping[str, object], presentation: Mapping[str, object]) -> str:
    """Render a selected human Task page and preserve legacy continuation text."""
    items = page.get("items", ())
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
        items = ()
    if not items and page.get("next_cursor") is not None:
        rendered = "No matches in this page; more candidates remain."
    else:
        rendered = render_task_list(items, presentation)
    stop_reason = page.get("stop_reason")
    lines = [rendered]
    if page.get("next_cursor") is not None or stop_reason not in {"exhausted", "complete", "completed"}:
        lines.append(f"Stop reason: {_sanitize(stop_reason)}")
    elif not items:
        lines.append("End of results.")
    command = presentation.get("continuation_command")
    if command is not None:
        lines.append(f"Continue with: {_human(command)}")
    return "\n".join(lines)


__all__ = ["render_task_list", "render_task_page"]
