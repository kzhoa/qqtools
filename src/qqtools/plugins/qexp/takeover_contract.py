"""Pure command and lifecycle values for a Project takeover."""

from __future__ import annotations

import re

TAKEOVER_PLAN_ID = re.compile(r"[0-9a-f]{64}")

TAKEOVER_TERMINAL_PHASES = {"completed", "cancelled", "stale"}
TAKEOVER_NEXT_PHASES = {
    "accepted": ("reserved", "aborting_stale", "cancelling"),
    "reserved": ("draining", "aborting_stale", "cancelling"),
    "draining": ("verifying_drain", "aborting_stale", "cancelling"),
    "verifying_drain": ("draining", "committing", "cancelling"),
    "committing": ("shared_committed", "aborting_stale", "cancelling"),
    "shared_committed": ("publishing_binding",),
    "publishing_binding": ("releasing",),
    "releasing": ("applying_groups", "completed"),
    "applying_groups": ("completed",),
    "cancelling": ("cancelled", "shared_committed"),
    "aborting_stale": ("stale", "shared_committed"),
}
TAKEOVER_PHASES = TAKEOVER_NEXT_PHASES.keys() | TAKEOVER_TERMINAL_PHASES
TAKEOVER_CANCELLABLE_PHASES = {"accepted", "reserved", "draining", "verifying_drain", "committing"}
