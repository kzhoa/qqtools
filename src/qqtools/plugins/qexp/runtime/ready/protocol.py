"""Shared protocol metadata for ready projections."""

from __future__ import annotations

from pathlib import Path

from ..paths import shared_paths

PRIMARY_READY_PROTOCOL_VERSION = 1


def projection_state_path(cfg: object) -> Path:
    """Return the durable state path for the primary-ready projection."""
    return shared_paths(cfg.shared_root)["ready_primary"] / "state.json"
