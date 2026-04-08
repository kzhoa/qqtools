from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = "2.0"

# ---------------------------------------------------------------------------
# Task phases
# ---------------------------------------------------------------------------

PHASE_QUEUED = "queued"
PHASE_DISPATCHING = "dispatching"
PHASE_STARTING = "starting"
PHASE_RUNNING = "running"
PHASE_SUCCEEDED = "succeeded"
PHASE_FAILED = "failed"
PHASE_CANCELLED = "cancelled"
PHASE_BLOCKED = "blocked"
PHASE_ORPHANED = "orphaned"

ALL_PHASES = (
    PHASE_QUEUED,
    PHASE_DISPATCHING,
    PHASE_STARTING,
    PHASE_RUNNING,
    PHASE_SUCCEEDED,
    PHASE_FAILED,
    PHASE_CANCELLED,
    PHASE_BLOCKED,
    PHASE_ORPHANED,
)

TERMINAL_PHASES = frozenset({PHASE_SUCCEEDED, PHASE_FAILED, PHASE_CANCELLED})
ACTIVE_PHASES = frozenset({PHASE_DISPATCHING, PHASE_STARTING, PHASE_RUNNING})
CANCELLABLE_PHASES = frozenset({PHASE_QUEUED, PHASE_DISPATCHING, PHASE_RUNNING})

LEGAL_TRANSITIONS: frozenset[tuple[str, str]] = frozenset(
    {
        (PHASE_QUEUED, PHASE_DISPATCHING),
        (PHASE_DISPATCHING, PHASE_STARTING),
        (PHASE_STARTING, PHASE_RUNNING),
        (PHASE_RUNNING, PHASE_SUCCEEDED),
        (PHASE_RUNNING, PHASE_FAILED),
        (PHASE_QUEUED, PHASE_CANCELLED),
        (PHASE_DISPATCHING, PHASE_CANCELLED),
        (PHASE_RUNNING, PHASE_CANCELLED),
        (PHASE_DISPATCHING, PHASE_ORPHANED),
        (PHASE_STARTING, PHASE_ORPHANED),
        (PHASE_RUNNING, PHASE_ORPHANED),
    }
)

# ---------------------------------------------------------------------------
# Agent modes and states
# ---------------------------------------------------------------------------

AGENT_MODE_ON_DEMAND = "on_demand"
AGENT_MODE_PERSISTENT = "persistent"
AGENT_MODES = (AGENT_MODE_ON_DEMAND, AGENT_MODE_PERSISTENT)

AGENT_STATE_STOPPED = "stopped"
AGENT_STATE_STARTING = "starting"
AGENT_STATE_ACTIVE = "active"
AGENT_STATE_IDLE = "idle"
AGENT_STATE_STALE = "stale"
AGENT_STATE_FAILED = "failed"

AGENT_STATES = (
    AGENT_STATE_STOPPED,
    AGENT_STATE_STARTING,
    AGENT_STATE_ACTIVE,
    AGENT_STATE_IDLE,
    AGENT_STATE_STALE,
    AGENT_STATE_FAILED,
)

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
MACHINE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def generate_id() -> str:
    return uuid.uuid4().hex[:12]


def validate_task_id(task_id: str) -> str:
    if not task_id or not isinstance(task_id, str):
        raise ValueError("task_id must be a non-empty string.")
    if "/" in task_id or "\\" in task_id or ".." in task_id:
        raise ValueError(
            "task_id contains illegal path characters. "
            "Only letters, digits, '.', '_', and '-' are allowed."
        )
    if not TASK_ID_PATTERN.fullmatch(task_id):
        raise ValueError(
            "task_id contains illegal characters. "
            "Only letters, digits, '.', '_', and '-' are allowed."
        )
    return task_id


def validate_machine_name(name: str) -> str:
    if not name or not isinstance(name, str):
        raise ValueError("machine_name must be a non-empty string.")
    if "/" in name or "\\" in name or ".." in name:
        raise ValueError("machine_name contains illegal path characters.")
    if not MACHINE_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "machine_name contains illegal characters. "
            "Only letters, digits, '.', '_', and '-' are allowed."
        )
    return name


def validate_phase(phase: str) -> str:
    if phase not in ALL_PHASES:
        raise ValueError(f"Invalid phase {phase!r}. Must be one of {ALL_PHASES}.")
    return phase


def validate_phase_transition(from_phase: str, to_phase: str) -> None:
    if (from_phase, to_phase) not in LEGAL_TRANSITIONS:
        raise ValueError(
            f"Illegal phase transition: {from_phase!r} -> {to_phase!r}."
        )


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Meta:
    revision: int
    created_at: str
    updated_at: str
    updated_by_machine: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Meta:
        return cls(**data)

    @classmethod
    def new(cls, machine_name: str) -> Meta:
        now = utc_now_iso()
        return cls(
            revision=1,
            created_at=now,
            updated_at=now,
            updated_by_machine=machine_name,
        )


@dataclass(slots=True)
class TaskSpec:
    command: list[str]
    requested_gpus: int

    def __post_init__(self) -> None:
        if not isinstance(self.command, list) or not self.command:
            raise ValueError("command must be a non-empty list.")
        if any(not isinstance(a, str) for a in self.command):
            raise ValueError("command entries must be strings.")
        if not isinstance(self.requested_gpus, int) or self.requested_gpus <= 0:
            raise ValueError("requested_gpus must be a positive integer.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskSpec:
        return cls(**data)


@dataclass(slots=True)
class TaskStatus:
    phase: str
    reason: str | None = None
    category: str | None = None

    def __post_init__(self) -> None:
        validate_phase(self.phase)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskStatus:
        return cls(**data)


@dataclass(slots=True)
class TaskRuntime:
    assigned_gpus: list[int] = field(default_factory=list)
    process_group_id: int | None = None
    wrapper_pid: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskRuntime:
        return cls(**data)


@dataclass(slots=True)
class TaskTimestamps:
    created_at: str
    queued_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskTimestamps:
        return cls(**data)


@dataclass(slots=True)
class TaskResult:
    exit_code: int | None = None
    terminal_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskResult:
        return cls(**data)


@dataclass(slots=True)
class TaskLineage:
    retry_of: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskLineage:
        return cls(**data)


@dataclass(slots=True)
class Task:
    meta: Meta
    task_id: str
    name: str | None
    batch_id: str | None
    machine_name: str
    attempt: int
    spec: TaskSpec
    status: TaskStatus
    runtime: TaskRuntime
    timestamps: TaskTimestamps
    result: TaskResult
    lineage: TaskLineage

    def __post_init__(self) -> None:
        validate_task_id(self.task_id)
        validate_machine_name(self.machine_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta.to_dict(),
            "task": {
                "task_id": self.task_id,
                "name": self.name,
                "batch_id": self.batch_id,
                "machine_name": self.machine_name,
                "attempt": self.attempt,
                "spec": self.spec.to_dict(),
                "status": self.status.to_dict(),
                "runtime": self.runtime.to_dict(),
                "timestamps": self.timestamps.to_dict(),
                "result": self.result.to_dict(),
                "lineage": self.lineage.to_dict(),
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        meta = Meta.from_dict(data["meta"])
        t = data["task"]
        return cls(
            meta=meta,
            task_id=t["task_id"],
            name=t.get("name"),
            batch_id=t.get("batch_id"),
            machine_name=t["machine_name"],
            attempt=t.get("attempt", 1),
            spec=TaskSpec.from_dict(t["spec"]),
            status=TaskStatus.from_dict(t["status"]),
            runtime=TaskRuntime.from_dict(t.get("runtime", {})),
            timestamps=TaskTimestamps.from_dict(t["timestamps"]),
            result=TaskResult.from_dict(t.get("result", {})),
            lineage=TaskLineage.from_dict(t.get("lineage", {})),
        )


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BatchSummary:
    total: int = 0
    queued: int = 0
    running: int = 0
    succeeded: int = 0
    failed: int = 0
    cancelled: int = 0
    blocked: int = 0
    orphaned: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchSummary:
        return cls(**data)


@dataclass(slots=True)
class BatchPolicy:
    allow_retry_failed: bool = True
    allow_retry_cancelled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchPolicy:
        return cls(**data)


@dataclass(slots=True)
class Batch:
    meta: Meta
    batch_id: str
    name: str | None
    source_manifest: str | None
    machine_name: str
    task_ids: list[str]
    summary: BatchSummary
    policy: BatchPolicy

    def __post_init__(self) -> None:
        validate_task_id(self.batch_id)
        validate_machine_name(self.machine_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta.to_dict(),
            "batch": {
                "batch_id": self.batch_id,
                "name": self.name,
                "source_manifest": self.source_manifest,
                "machine_name": self.machine_name,
                "task_ids": list(self.task_ids),
                "summary": self.summary.to_dict(),
                "policy": self.policy.to_dict(),
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Batch:
        meta = Meta.from_dict(data["meta"])
        b = data["batch"]
        return cls(
            meta=meta,
            batch_id=b["batch_id"],
            name=b.get("name"),
            source_manifest=b.get("source_manifest"),
            machine_name=b["machine_name"],
            task_ids=b.get("task_ids", []),
            summary=BatchSummary.from_dict(b.get("summary", {})),
            policy=BatchPolicy.from_dict(b.get("policy", {})),
        )


# ---------------------------------------------------------------------------
# Machine
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GpuInventory:
    count: int = 0
    visible_gpu_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GpuInventory:
        return cls(**data)


@dataclass(slots=True)
class Machine:
    machine_name: str
    hostname: str | None
    shared_root: str
    runtime_root: str
    agent_mode: str
    agent_state: str
    last_heartbeat: str | None
    gpu_inventory: GpuInventory

    def __post_init__(self) -> None:
        validate_machine_name(self.machine_name)
        if self.agent_mode not in AGENT_MODES:
            raise ValueError(
                f"agent_mode must be one of {AGENT_MODES}, got {self.agent_mode!r}."
            )
        if self.agent_state not in AGENT_STATES:
            raise ValueError(
                f"agent_state must be one of {AGENT_STATES}, got {self.agent_state!r}."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "machine": {
                "machine_name": self.machine_name,
                "hostname": self.hostname,
                "shared_root": self.shared_root,
                "runtime_root": self.runtime_root,
                "agent_mode": self.agent_mode,
                "agent_state": self.agent_state,
                "last_heartbeat": self.last_heartbeat,
                "gpu_inventory": self.gpu_inventory.to_dict(),
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Machine:
        m = data["machine"]
        return cls(
            machine_name=m["machine_name"],
            hostname=m.get("hostname"),
            shared_root=m["shared_root"],
            runtime_root=m["runtime_root"],
            agent_mode=m.get("agent_mode", AGENT_MODE_ON_DEMAND),
            agent_state=m.get("agent_state", AGENT_STATE_STOPPED),
            last_heartbeat=m.get("last_heartbeat"),
            gpu_inventory=GpuInventory.from_dict(m.get("gpu_inventory", {})),
        )
