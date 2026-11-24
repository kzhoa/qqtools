"""Canonical activation checkpoint evidence for the Idle Job decision guard."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from operator import itemgetter

IDLE_DECISION_SECONDS = 0.05
LOCKS_PER_PROJECT = 3

IdleGuardCheckpoint = tuple[str, str | None, str, str | None, int | None]


def idle_guard_digest(checkpoints: Iterable[IdleGuardCheckpoint]) -> str:
    """Hash the exact roster's checkpoint rows ordered by unique Project ID."""
    encoded = json.dumps(sorted(checkpoints, key=itemgetter(0)), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()
