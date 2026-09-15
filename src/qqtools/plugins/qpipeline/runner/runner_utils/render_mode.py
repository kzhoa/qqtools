"""Terminal-aware renderer selection with explicit overrides preserved."""

from __future__ import annotations

import sys


def _isatty(stream: object) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, OSError, ValueError):
        return False


def resolve_render_mode(
    requested_mode: str | None, has_rich: bool, has_tqdm: bool, *, is_terminal: bool | None = None,
) -> tuple[str, str | None]:
    """Resolve auto against the renderer's output stream, not import availability.

    Rich uses stdout and tqdm uses stderr. ``is_terminal`` overrides both probes
    for deterministic callers/tests. Explicit rich/tqdm selections keep their
    existing dependency fallbacks even when output is redirected.
    """
    input_mode = requested_mode or "auto"
    if requested_mode is None or requested_mode == "auto":
        stdout_tty = _isatty(sys.stdout) if is_terminal is None else is_terminal
        stderr_tty = _isatty(sys.stderr) if is_terminal is None else is_terminal
        if has_rich and stdout_tty:
            mode = "rich"
        elif has_tqdm and stderr_tty:
            mode = "tqdm"
        else:
            mode = "plain"
    elif requested_mode == "rich":
        mode = "rich" if has_rich else "tqdm" if has_tqdm else "plain"
    elif requested_mode == "tqdm":
        mode = "tqdm" if has_tqdm else "plain"
    else:
        mode = "plain"
    return mode, f"Mode {input_mode} -> {mode}" if input_mode != mode else None
