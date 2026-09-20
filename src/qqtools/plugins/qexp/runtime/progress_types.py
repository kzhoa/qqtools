"""Typed JSON shapes exchanged by qexp progress observation."""

from __future__ import annotations

from typing import Literal, TypeAlias, TypedDict


class ProgressPayload(TypedDict):
    """Normalized application progress payload."""

    stage: str
    current: int | None
    total: int | None
    unit: str | None
    message: str | None


class AvailableObservation(TypedDict):
    """An observation with a validated progress snapshot."""

    status: Literal["available"]
    observation_state: Literal["available"]
    reason: None
    protocol_version: int
    task_id: str
    attempt_id: str
    attempt_number: int
    machine_name: str
    launch_id: str | None
    wrapper_pid: int | None
    wrapper_start_time_ticks: int | None
    registration_generation: str
    fencing_token: int
    source_update_id: str
    sequence: int
    reported_at: str
    advanced_at: str
    progress: ProgressPayload


class PendingObservation(TypedDict):
    """An observation for a queued task that has not started."""

    status: Literal["unavailable"]
    observation_state: Literal["pending"]
    reason: Literal["not_started"]


class NoReportObservation(TypedDict):
    """An observation whose attempt has no accepted report."""

    status: Literal["unavailable"]
    observation_state: Literal["no_report"]
    reason: Literal["no_snapshot"]


class UnavailableObservation(TypedDict):
    """An observation that could not be safely read or matched."""

    status: Literal["unavailable"]
    observation_state: Literal["unavailable"]
    reason: Literal["cleanup", "read_failed", "invalid_snapshot", "identity_mismatch", "unknown"]


ProgressObservation: TypeAlias = (
    AvailableObservation | PendingObservation | NoReportObservation | UnavailableObservation
)
