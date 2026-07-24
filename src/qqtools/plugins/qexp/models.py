"""Public qexp constants and validation helpers."""
from __future__ import annotations

from .runtime.records import (ATTEMPT_PHASES, GROUP_ADMISSION_STATES, GROUP_DISPATCH_STATES,
    SCHEMA_VERSION, TASK_PHASES, AttemptRecord, TaskRecord, TaskSpec, new_id, utc_now,
    validate_group_name, validate_identifier)

PHASE_QUEUED = "queued"
PHASE_RUNNING = "running"
PHASE_SUCCEEDED = "succeeded"
PHASE_FAILED = "failed"
PHASE_CANCELLED = "cancelled"
PHASE_BLOCKED = "blocked"
PHASE_DISPATCHING = "running"
PHASE_STARTING = "running"
PHASE_ORPHANED = "blocked"

AGENT_MODE_ON_DEMAND = "on_demand"
AGENT_MODE_PERSISTENT = "persistent"

TERMINAL_PHASES = frozenset({PHASE_SUCCEEDED, PHASE_FAILED, PHASE_CANCELLED})

__all__ = ["ATTEMPT_PHASES", "AGENT_MODE_ON_DEMAND", "AGENT_MODE_PERSISTENT", "AttemptRecord",
           "PHASE_BLOCKED", "PHASE_CANCELLED", "PHASE_FAILED", "PHASE_QUEUED", "PHASE_RUNNING",
           "PHASE_SUCCEEDED", "SCHEMA_VERSION", "TASK_PHASES", "TaskRecord", "TaskSpec", "new_id",
           "utc_now", "validate_group_name", "validate_identifier"]
