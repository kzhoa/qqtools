"""Pure formatting helpers for Project-related CLI presentation."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ProjectPresentationState",
    "encode_display_text",
    "abbreviate_home",
    "format_project_line",
]


@dataclass(slots=True)
class ProjectPresentationState:
    """Track one selected Project line through deferred CLI presentation."""

    line: str
    is_presented: bool = False

    def mark_presented(self) -> None:
        """Record that this invocation has shown its selected Project."""
        self.is_presented = True


def encode_display_text(value: str) -> str:
    """Escape control characters so displayed text stays on one physical line."""
    if not isinstance(value, str):
        raise TypeError("display text must be a string")

    encoded: list[str] = []
    for character in value:
        codepoint = ord(character)
        if character == "\\":
            encoded.append("\\\\")
        elif character == "\n":
            encoded.append("\\n")
        elif character == "\r":
            encoded.append("\\r")
        elif character == "\t":
            encoded.append("\\t")
        elif codepoint <= 0x1F or 0x7F <= codepoint <= 0x9F:
            encoded.append(f"\\x{codepoint:02x}")
        elif character in {"\u2028", "\u2029"} or unicodedata.category(character) == "Cf":
            escape = "u" if codepoint <= 0xFFFF else "U"
            width = 4 if codepoint <= 0xFFFF else 8
            encoded.append(f"\\{escape}{codepoint:0{width}x}")
        else:
            encoded.append(character)
    return "".join(encoded)


def abbreviate_home(path: str | Path, home: str | Path | None = None) -> str:
    """Return a canonical path with the canonical home directory shortened to ``~``."""
    canonical_path = Path(path).expanduser().resolve()
    home_path = Path.home() if home is None else Path(home)
    canonical_home = home_path.expanduser().resolve()
    try:
        relative_path = canonical_path.relative_to(canonical_home)
    except ValueError:
        return str(canonical_path)
    if not relative_path.parts:
        return "~"
    return f"~/{relative_path.as_posix()}"


def format_project_line(
    project_root: str | Path,
    source: str | None = None,
    invocation_cwd: str | Path | None = None,
    context_file: str | Path | None = None,
) -> str:
    """Format the canonical Project directory and its optional selection source."""
    canonical_root = Path(project_root).expanduser().resolve()
    line = f"Project: {encode_display_text(str(canonical_root))}"

    if source in (None, "explicit", "cli"):
        suffix = None
    elif source == "environment":
        suffix = "$QEXP_SHARED_ROOT"
    elif source in ("cwd", "cwd_ancestor"):
        if invocation_cwd is None:
            raise ValueError(f"invocation_cwd is required for source {source!r}")
        canonical_cwd = Path(invocation_cwd).expanduser().resolve()
        suffix = None if canonical_root == canonical_cwd else "parent directory"
    elif source == "manifest_ancestor":
        suffix = "manifest directory"
    elif source == "saved":
        if context_file is None:
            raise ValueError("context_file is required for source 'saved'")
        suffix = encode_display_text(abbreviate_home(context_file))
    else:
        raise ValueError(f"unknown Project source: {source!r}")

    if suffix is None:
        return line
    return f"{line} (from {suffix})"
