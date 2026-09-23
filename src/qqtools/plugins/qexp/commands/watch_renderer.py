"""Bounded terminal presentation for continuous qexp Task observation."""

from __future__ import annotations

import os
import shutil
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from errno import EPIPE
from typing import Any, TextIO


class WatchRenderer:
    """Render accepted Task frames using the best available terminal backend."""

    def __init__(self, stdout: TextIO, stderr: TextIO | None = None) -> None:
        self.stdout = stdout
        self.stderr = stderr if stderr is not None else sys.stderr
        self._cursor_up: str | None = None
        self._clear_line: str | None = None
        self._backend = "append"
        self._rich_console: Any = None
        self._rich_live: Any = None
        self._rich_table: Any = None
        self._rich_text: Any = None
        self._rich_started = False
        self._rich_diagnosed = False
        self._backend_diagnosed = False
        self._closed = False
        self._last_signature: tuple[Any, ...] | None = None
        self._last_dimensions: tuple[int, int] | None = None
        self._ansi_rows = 0

        capabilities = _cursor_capabilities(stdout)
        if capabilities is None:
            return
        self._cursor_up, self._clear_line = capabilities
        self._backend = "ansi"
        try:
            from rich.console import Console
            from rich.live import Live
            from rich.table import Table
            from rich.text import Text

            self._rich_console = Console(
                file=stdout,
                force_terminal=True,
                highlight=False,
                no_color=True,
                soft_wrap=True,
            )
            self._rich_live = Live(
                console=self._rich_console,
                auto_refresh=False,
                redirect_stdout=False,
                redirect_stderr=False,
                transient=False,
                vertical_overflow="crop",
            )
            self._rich_table = Table
            self._rich_text = Text
            self._backend = "rich"
        except Exception as exc:
            self._diagnose_rich_unavailable(exc)

    def __enter__(self) -> WatchRenderer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def render(self, payload: Mapping[str, Any], frame: str) -> None:
        """Render one canonical payload and its existing human frame."""
        if self._closed:
            raise RuntimeError("cannot render with a closed Task watch viewer")
        signature = _payload_signature(payload)
        safe_frame = _sanitize_frame(frame)
        columns, rows = _terminal_size()
        wrap_width = max(1, columns - 1)

        if self._backend == "append":
            self._append_if_changed(signature, safe_frame, wrap_width, rows)
            return

        lines = _viewport_lines(safe_frame, wrap_width, rows)
        try:
            if self._backend == "rich":
                self._render_rich(lines)
            else:
                self._render_ansi(lines, (columns, rows))
        except Exception as exc:
            if _is_closed_output_error(exc, self.stdout):
                raise
            self._handle_backend_failure(exc)
            self._append_if_changed(signature, safe_frame, wrap_width, rows)
            return
        self._last_signature = signature

    def close(self) -> None:
        """Restore Rich terminal state after normal exit or interruption."""
        if self._closed:
            return
        self._closed = True
        if self._rich_started:
            try:
                self._rich_live.stop()
            except Exception as exc:
                if not _is_closed_output_error(exc, self.stdout):
                    self._diagnose_backend_failure(exc)
                self._restore_rich_terminal_state()
            finally:
                self._rich_started = False

    def _render_rich(self, lines: Sequence[str]) -> None:
        table = self._rich_table.grid(padding=0, expand=True)
        table.add_column(no_wrap=True, overflow="crop")
        for line in lines:
            table.add_row(self._rich_text(line, no_wrap=True, overflow="crop"))
        if not self._rich_started:
            self._rich_started = True
            self._rich_live.start(refresh=False)
        self._rich_live.update(table, refresh=True)

    def _render_ansi(self, lines: Sequence[str], dimensions: tuple[int, int]) -> None:
        if self._ansi_rows:
            self.stdout.write("\r")
            self.stdout.write(self._clear_line or "")
            for _ in range(self._ansi_rows - 1):
                self.stdout.write(self._cursor_up or "")
                self.stdout.write("\r")
                self.stdout.write(self._clear_line or "")
        self.stdout.write("\r\n".join(lines))
        self.stdout.flush()
        self._ansi_rows = len(lines)
        self._last_dimensions = dimensions

    def _append_if_changed(
        self,
        signature: tuple[Any, ...],
        frame: str,
        width: int,
        height: int,
    ) -> None:
        if signature == self._last_signature:
            return
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        lines = _viewport_lines(f"[{timestamp}]\n{frame}", width, height)
        block = "\n".join((*lines, ""))
        self.stdout.write(block)
        self.stdout.flush()
        self._last_signature = signature

    def _handle_backend_failure(self, exc: Exception) -> None:
        if self._rich_started:
            try:
                self._rich_live.stop()
            except Exception:
                self._restore_rich_terminal_state()
            finally:
                self._rich_started = False
            self._restore_rich_terminal_state()
        self._backend = "append"
        self._ansi_rows = 0
        self._last_signature = None
        self._diagnose_backend_failure(exc)

    def _restore_rich_terminal_state(self) -> None:
        if self._cursor_up is None or self._clear_line is None:
            return
        try:
            self.stdout.write("\x1b[0m\x1b[?25h")
            self.stdout.flush()
        except Exception:
            pass

    def _diagnose_rich_unavailable(self, exc: Exception) -> None:
        if self._rich_diagnosed:
            return
        self._rich_diagnosed = True
        self._write_diagnostic(f"qexp task watch: Rich renderer unavailable ({type(exc).__name__}); using ANSI text.")

    def _diagnose_backend_failure(self, exc: Exception) -> None:
        if self._backend_diagnosed:
            return
        self._backend_diagnosed = True
        self._write_diagnostic(f"qexp task watch: terminal renderer failed ({type(exc).__name__}); appending updates.")

    def _write_diagnostic(self, message: str) -> None:
        try:
            self.stderr.write(message + "\n")
            self.stderr.flush()
        except Exception:
            pass


def _cursor_capabilities(stdout: TextIO) -> tuple[str, str] | None:
    """Return usable terminfo cursor-up and clear-line controls for stdout."""
    if os.environ.get("TERM") in {None, "", "dumb"}:
        return None
    try:
        import curses

        if not stdout.isatty():
            return None
        file_descriptor = stdout.fileno()
        curses.setupterm(term=os.environ["TERM"], fd=file_descriptor)
        cursor_up = curses.tigetstr("cuu1")
        clear_line = curses.tigetstr("el")
    except Exception:
        return None
    if not cursor_up or not clear_line:
        return None
    return cursor_up.decode("latin-1"), clear_line.decode("latin-1")


def _terminal_size() -> tuple[int, int]:
    try:
        size = shutil.get_terminal_size(fallback=(80, 24))
    except (OSError, ValueError):
        return (80, 24)
    return (max(1, size.columns), max(1, size.lines))


def _payload_signature(payload: Mapping[str, Any]) -> tuple[Any, ...]:
    """Identify displayed state and accepted reports, excluding lease revisions and age text."""
    return tuple(
        _freeze(payload.get(name))
        for name in (
            "task_id",
            "name",
            "phase",
            "reason",
            "terminal",
            "selected_attempt",
            "observation_state",
            "observation_reason",
            "progress",
            "progress_extended",
            "selected_progress_version",
        )
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted(((key, _freeze(item)) for key, item in value.items()), key=lambda pair: str(pair[0])))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze(item) for item in value)
    return value


def _sanitize_text(text: str, *, preserve_newlines: bool) -> str:
    safe_characters: list[str] = []
    for character in text:
        if character == "\n" and preserve_newlines:
            safe_characters.append(character)
            continue
        category = unicodedata.category(character)
        if character == "\t" or category in {"Cc", "Cs", "Zl", "Zp"}:
            safe_characters.append(" ")
        elif category == "Cf" and character not in {"\u200c", "\u200d"}:
            continue
        else:
            safe_characters.append(character)
    return "".join(safe_characters)


def _sanitize_frame(frame: str) -> str:
    if not isinstance(frame, str):
        raise TypeError(f"Task watch frame must be a string, got {type(frame).__name__}")
    return _sanitize_text(frame, preserve_newlines=True)


def _viewport_lines(frame: str, width: int, height: int) -> list[str]:
    wrapped: list[str] = []
    for line in frame.split("\n"):
        wrapped.extend(_wrap_line(line, width))
    if not wrapped:
        wrapped = [""]
    if len(wrapped) <= height:
        return wrapped
    if height == 1:
        return [_omission_line(len(wrapped), width)]
    visible_count = height - 1
    omitted_count = len(wrapped) - visible_count
    return [*wrapped[:visible_count], _omission_line(omitted_count, width)]


def _wrap_line(line: str, width: int) -> list[str]:
    rows: list[str] = []
    current: list[str] = []
    current_width = 0
    for character in line:
        character_width = _cell_width(character)
        if character_width > width:
            character = "�"
            character_width = _cell_width(character)
        if character_width and current_width + character_width > width:
            rows.append("".join(current))
            current = []
            current_width = 0
        current.append(character)
        current_width += character_width
    rows.append("".join(current))
    return rows


def _cell_width(character: str) -> int:
    category = unicodedata.category(character)
    if unicodedata.combining(character) or category in {"Mn", "Me", "Cf"}:
        return 0
    return 2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1


def _omission_line(count: int, width: int) -> str:
    label = f"... {count} rows omitted"
    if _display_width(label) <= width:
        return label
    compact = f"{count} omitted"
    return _clip_line(compact, width)


def _display_width(line: str) -> int:
    return sum(_cell_width(character) for character in line)


def _clip_line(line: str, width: int) -> str:
    clipped: list[str] = []
    current_width = 0
    for character in line:
        character_width = _cell_width(character)
        if current_width + character_width > width:
            break
        clipped.append(character)
        current_width += character_width
    return "".join(clipped)


def _is_closed_output_error(exc: Exception, stdout: TextIO) -> bool:
    if isinstance(exc, BrokenPipeError):
        return True
    if isinstance(exc, OSError) and exc.errno == EPIPE:
        return True
    return isinstance(exc, ValueError) and (bool(getattr(stdout, "closed", False)) or "closed file" in str(exc).lower())


def sanitize_payload_for_frame(value: Any) -> Any:
    """Copy canonical data while making producer text safe for a terminal frame."""
    if isinstance(value, str):
        return _sanitize_text(value, preserve_newlines=False)
    if isinstance(value, Mapping):
        return {
            sanitize_payload_for_frame(key) if isinstance(key, str) else key: sanitize_payload_for_frame(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_payload_for_frame(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_payload_for_frame(item) for item in value)
    return value


__all__ = ["WatchRenderer", "sanitize_payload_for_frame"]
