"""Monotonic scheduling arithmetic; never a source of lease or fencing authority."""

from __future__ import annotations

import math


def advance_deadline(scheduled_at: float, interval: float, finished_at: float) -> tuple[float, int]:
    """Advance a periodic deadline, dropping missed slots rather than replaying them.

    The returned deadline is strictly after ``finished_at``. ``skipped`` counts
    scheduled slots after the one just executed that can no longer be served.
    No wall-clock timestamps or lease-expiry values belong in this function.
    """
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("control-plane interval must be finite and positive")
    if not math.isfinite(scheduled_at) or not math.isfinite(finished_at):
        raise ValueError("control-plane monotonic timestamps must be finite")
    deadline = scheduled_at + interval
    skipped = 0
    if deadline <= finished_at:
        skipped = int((finished_at - deadline) // interval) + 1
        deadline += skipped * interval
        # Floating-point rounding must not turn an overrun into a busy loop.
        if deadline <= finished_at:
            deadline = math.nextafter(finished_at, math.inf)
    return deadline, skipped
