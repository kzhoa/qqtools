"""Continuous Task observation commands for qexp."""

from __future__ import annotations

import errno
import sys
import time
from collections.abc import Callable
from typing import Any, TextIO

from .. import observer
from ..cli.output import CliOutput, OutputKind, render
from .watch_renderer import WatchRenderer, sanitize_payload_for_frame


def watch_task(
    cfg: Any,
    task_id: str,
    *,
    interval_seconds: int | float = 2,
    details: bool = False,
    follow_retries: bool = False,
    observer_attempt_id: str | None = None,
    stdout: TextIO = sys.stdout,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Refresh one compact Task frame until terminal or interruption."""
    if bool(getattr(stdout, "closed", False)):
        return 0
    renderer = WatchRenderer(stdout=stdout, stderr=sys.stderr)
    seen_observer_attempt = False
    try:
        while True:
            payload = observer.inspect_current_task(cfg, task_id)
            if observer_attempt_id is not None:
                selected = payload.get("selected_attempt")
                selected_id = selected.get("attempt_id") if isinstance(selected, dict) else None
                if selected_id is not None and selected_id != observer_attempt_id:
                    return 0
                if selected_id is None and seen_observer_attempt:
                    return 0
                seen_observer_attempt |= selected_id == observer_attempt_id
            presentation = {"details": details} if details else {}
            safe_payload = sanitize_payload_for_frame(payload)
            frame = render(CliOutput(OutputKind.TASK_WATCH, safe_payload, presentation), "human")
            try:
                renderer.render(payload, frame)
            except BrokenPipeError:
                return 0
            except OSError as exc:
                if exc.errno == errno.EPIPE:
                    return 0
                raise
            except ValueError as exc:
                if bool(getattr(stdout, "closed", False)) or "closed file" in str(exc).lower():
                    return 0
                raise
            is_terminal = bool(payload.get("terminal"))
            is_transition = payload.get("observation_reason") == "concurrent_transition"
            if is_terminal and not is_transition and not follow_retries:
                return 0
            sleep(interval_seconds)
    finally:
        renderer.close()


__all__ = ["watch_task"]
