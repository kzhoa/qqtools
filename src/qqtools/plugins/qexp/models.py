from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = "4.0"
ROOT_SCOPE_PROJECT = "project"
ROOT_SCOPES = (ROOT_SCOPE_PROJECT,)

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
AGENT_STATE_DRAINING = "draining"
AGENT_STATE_IDLE = "idle"
AGENT_STATE_STALE = "stale"
AGENT_STATE_FAILED = "failed"

AGENT_STATES = (
    AGENT_STATE_STOPPED,
    AGENT_STATE_STARTING,
    AGENT_STATE_ACTIVE,
    AGENT_STATE_DRAINING,
    AGENT_STATE_IDLE,
    AGENT_STATE_STALE,
    AGENT_STATE_FAILED,
)

# ---------------------------------------------------------------------------
# Batch commit states
# ---------------------------------------------------------------------------

BATCH_COMMIT_PREPARING = "preparing"
BATCH_COMMIT_COMMITTED = "committed"
BATCH_COMMIT_ABORTED = "aborted"

BATCH_COMMIT_STATES = (
    BATCH_COMMIT_PREPARING,
    BATCH_COMMIT_COMMITTED,
    BATCH_COMMIT_ABORTED,
)

# ---------------------------------------------------------------------------
# Resubmit operation states
# ---------------------------------------------------------------------------

RESUBMIT_STATE_PREPARING = "preparing"
RESUBMIT_STATE_DELETING_OLD = "deleting_old"
RESUBMIT_STATE_CREATING_NEW = "creating_new"
RESUBMIT_STATE_COMMITTED = "committed"
RESUBMIT_STATE_ABORTED = "aborted"

RESUBMIT_STATES = (
    RESUBMIT_STATE_PREPARING,
    RESUBMIT_STATE_DELETING_OLD,
    RESUBMIT_STATE_CREATING_NEW,
    RESUBMIT_STATE_COMMITTED,
    RESUBMIT_STATE_ABORTED,
)

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
MACHINE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
GROUP_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
GROUP_MAX_LENGTH = 64
GROUP_RESERVED_NAMES = frozenset({"experiments", "qqtools_internal"})


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


def validate_group_name(group: str | None) -> str | None:
    if group is None:
        return None
    if not isinstance(group, str):
        raise ValueError(
            "group must be a string or null. "
            "group is the long-lived grouping key that runtime maps to a tmux session."
        )
    if not group:
        raise ValueError(
            "group must not be empty. "
            "Use null for no long-lived grouping, or pass a stable group name."
        )
    if len(group) > GROUP_MAX_LENGTH:
        raise ValueError(
            f"group length must be <= {GROUP_MAX_LENGTH}. "
            "group is a long-lived key, not a free-form description."
        )
    if group[0] in {".", "-"}:
        raise ValueError(
            "group must not start with '.' or '-'. "
            "It maps directly to a tmux session name."
        )
    if "/" in group or "\\" in group:
        raise ValueError(
            "group contains illegal path characters. "
            "It maps directly to a tmux session name."
        )
    if group in GROUP_RESERVED_NAMES:
        raise ValueError(
            f"group {group!r} is reserved. "
            "Choose another long-lived grouping key."
        )
    if not GROUP_PATTERN.fullmatch(group):
        raise ValueError(
            "group contains illegal characters. "
            "Only letters, digits, '.', '_', and '-' are allowed. "
            "group maps directly to a tmux session name."
        )
    return group


def validate_root_scope(scope: str) -> str:
    if scope not in ROOT_SCOPES:
        raise ValueError(f"root_scope must be one of {ROOT_SCOPES}, got {scope!r}.")
    return scope


def tmux_session_for_group(group: str | None) -> str:
    return group if group is not None else "experiments"


def validate_phase(phase: str) -> str:
    if phase not in ALL_PHASES:
        raise ValueError(f"Invalid phase {phase!r}. Must be one of {ALL_PHASES}.")
    return phase


def validate_phase_transition(from_phase: str, to_phase: str) -> None:
    if (from_phase, to_phase) not in LEGAL_TRANSITIONS:
        raise ValueError(
            f"Illegal phase transition: {from_phase!r} -> {to_phase!r}."
        )


def validate_batch_commit_state(state: str) -> str:
    if state not in BATCH_COMMIT_STATES:
        raise ValueError(
            f"Invalid batch commit_state {state!r}. "
            f"Must be one of {BATCH_COMMIT_STATES}."
        )
    return state


def validate_resubmit_state(state: str) -> str:
    if state not in RESUBMIT_STATES:
        raise ValueError(
            f"Invalid resubmit state {state!r}. Must be one of {RESUBMIT_STATES}."
        )
    return state


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
    group: str | None
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
        validate_group_name(self.group)
        validate_machine_name(self.machine_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta.to_dict(),
            "task": {
                "task_id": self.task_id,
                "name": self.name,
                "group": self.group,
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
            group=validate_group_name(t.get("group")),
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
    group: str | None
    source_manifest: str | None
    machine_name: str
    commit_state: str = BATCH_COMMIT_COMMITTED
    expected_task_count: int = 0
    task_ids: list[str] = field(default_factory=list)
    summary: BatchSummary = field(default_factory=BatchSummary)
    policy: BatchPolicy = field(default_factory=BatchPolicy)

    def __post_init__(self) -> None:
        validate_task_id(self.batch_id)
        validate_group_name(self.group)
        validate_machine_name(self.machine_name)
        validate_batch_commit_state(self.commit_state)
        if not isinstance(self.expected_task_count, int) or self.expected_task_count < 0:
            raise ValueError("expected_task_count must be a non-negative integer.")
        if len(self.task_ids) > self.expected_task_count:
            raise ValueError(
                "task_ids length must not exceed expected_task_count."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta.to_dict(),
            "batch": {
                "batch_id": self.batch_id,
                "name": self.name,
                "group": self.group,
                "source_manifest": self.source_manifest,
                "machine_name": self.machine_name,
                "commit_state": self.commit_state,
                "expected_task_count": self.expected_task_count,
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
            group=validate_group_name(b.get("group")),
            source_manifest=b.get("source_manifest"),
            machine_name=b["machine_name"],
            commit_state=validate_batch_commit_state(
                b.get("commit_state", BATCH_COMMIT_COMMITTED)
            ),
            expected_task_count=b.get("expected_task_count", len(b.get("task_ids", []))),
            task_ids=b.get("task_ids", []),
            summary=BatchSummary.from_dict(b.get("summary", {})),
            policy=BatchPolicy.from_dict(b.get("policy", {})),
        )


# ---------------------------------------------------------------------------
# Resubmit operation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ResubmitNewSubmission:
    command: list[str]
    requested_gpus: int
    name: str | None
    group: str | None
    machine_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.command, list) or not self.command:
            raise ValueError("resubmit new submission command must be a non-empty list.")
        if any(not isinstance(a, str) for a in self.command):
            raise ValueError("resubmit new submission command entries must be strings.")
        if not isinstance(self.requested_gpus, int) or self.requested_gpus <= 0:
            raise ValueError("requested_gpus must be a positive integer.")
        validate_group_name(self.group)
        validate_machine_name(self.machine_name)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResubmitNewSubmission:
        return cls(**data)


@dataclass(slots=True)
class ResubmitOldTaskSummary:
    phase: str
    machine_name: str
    attempt: int
    batch_id: str | None
    name: str | None
    group: str | None

    def __post_init__(self) -> None:
        validate_phase(self.phase)
        validate_machine_name(self.machine_name)
        validate_group_name(self.group)
        if not isinstance(self.attempt, int) or self.attempt <= 0:
            raise ValueError("attempt must be a positive integer.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResubmitOldTaskSummary:
        return cls(**data)


@dataclass(slots=True)
class ResubmitOperation:
    meta: Meta
    operation_type: str
    task_id: str
    state: str
    old_task_snapshot_path: str
    new_submission: ResubmitNewSubmission
    new_task_snapshot: dict[str, Any]
    old_task_summary: ResubmitOldTaskSummary

    def __post_init__(self) -> None:
        if self.operation_type != "resubmit":
            raise ValueError("operation_type must be 'resubmit'.")
        validate_task_id(self.task_id)
        validate_resubmit_state(self.state)
        Task.from_dict(self.new_task_snapshot)

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta.to_dict(),
            "operation": {
                "operation_type": self.operation_type,
                "task_id": self.task_id,
                "state": self.state,
                "old_task_snapshot_path": self.old_task_snapshot_path,
                "new_submission": self.new_submission.to_dict(),
                "new_task_snapshot": self.new_task_snapshot,
                "old_task_summary": self.old_task_summary.to_dict(),
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResubmitOperation:
        meta = Meta.from_dict(data["meta"])
        op = data["operation"]
        return cls(
            meta=meta,
            operation_type=op["operation_type"],
            task_id=op["task_id"],
            state=validate_resubmit_state(op["state"]),
            old_task_snapshot_path=op["old_task_snapshot_path"],
            new_submission=ResubmitNewSubmission.from_dict(op["new_submission"]),
            new_task_snapshot=op["new_task_snapshot"],
            old_task_summary=ResubmitOldTaskSummary.from_dict(op["old_task_summary"]),
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
    gpu_inventory: GpuInventory

    def __post_init__(self) -> None:
        validate_machine_name(self.machine_name)
        if self.agent_mode not in AGENT_MODES:
            raise ValueError(
                f"agent_mode must be one of {AGENT_MODES}, got {self.agent_mode!r}."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "machine": {
                "machine_name": self.machine_name,
                "hostname": self.hostname,
                "shared_root": self.shared_root,
                "runtime_root": self.runtime_root,
                "agent_mode": self.agent_mode,
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
            gpu_inventory=GpuInventory.from_dict(m.get("gpu_inventory", {})),
        )


@dataclass(slots=True)
class RootGovernanceSnapshot:
    total_tasks: int = 0
    terminal_tasks: int = 0
    total_batches: int = 0
    total_events: int = 0
    total_machines: int = 0
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RootGovernanceSnapshot:
        return cls(**data)


@dataclass(slots=True)
class RootLifecyclePolicy:
    clean_after_seconds: int = 7 * 24 * 3600
    archive_events_after_seconds: int | None = None
    prune_terminal_events_after_seconds: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RootLifecyclePolicy:
        return cls(**data)


@dataclass(slots=True)
class RootManifest:
    schema_version: str
    root_scope: str
    control_plane_id: str
    shared_root: str
    project_root: str
    layout_version: str
    created_at: str
    created_by_machine: str
    lifecycle_policy: RootLifecyclePolicy = field(default_factory=RootLifecyclePolicy)
    governance: RootGovernanceSnapshot = field(default_factory=RootGovernanceSnapshot)

    def __post_init__(self) -> None:
        validate_root_scope(self.root_scope)
        validate_machine_name(self.created_by_machine)
        validate_task_id(self.control_plane_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_manifest": {
                "schema_version": self.schema_version,
                "root_scope": self.root_scope,
                "control_plane_id": self.control_plane_id,
                "shared_root": self.shared_root,
                "project_root": self.project_root,
                "layout_version": self.layout_version,
                "created_at": self.created_at,
                "created_by_machine": self.created_by_machine,
                "lifecycle_policy": self.lifecycle_policy.to_dict(),
                "governance": self.governance.to_dict(),
            }
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RootManifest:
        root = data["root_manifest"]
        return cls(
            schema_version=root["schema_version"],
            root_scope=validate_root_scope(root["root_scope"]),
            control_plane_id=root["control_plane_id"],
            shared_root=root["shared_root"],
            project_root=root["project_root"],
            layout_version=root["layout_version"],
            created_at=root["created_at"],
            created_by_machine=root["created_by_machine"],
            lifecycle_policy=RootLifecyclePolicy.from_dict(
                root.get("lifecycle_policy", {})
            ),
            governance=RootGovernanceSnapshot.from_dict(root.get("governance", {})),
        )


@dataclass(slots=True)
class MachineWorkset:
    machine_name: str
    queued_count: int = 0
    dispatching_count: int = 0
    starting_count: int = 0
    running_count: int = 0
    terminal_count: int = 0
    has_launch_backlog: bool = False
    has_active_responsibility: bool = False
    updated_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        validate_machine_name(self.machine_name)
        counts = (
            self.queued_count,
            self.dispatching_count,
            self.starting_count,
            self.running_count,
            self.terminal_count,
        )
        if any((not isinstance(count, int)) or count < 0 for count in counts):
            raise ValueError("MachineWorkset counts must be non-negative integers.")
        expected_backlog = (
            self.queued_count > 0
            or self.dispatching_count > 0
            or self.starting_count > 0
        )
        expected_active = expected_backlog or self.running_count > 0
        if self.has_launch_backlog != expected_backlog:
            raise ValueError(
                "MachineWorkset.has_launch_backlog must match queued/dispatching/starting counts."
            )
        if self.has_active_responsibility != expected_active:
            raise ValueError(
                "MachineWorkset.has_active_responsibility must match launch backlog or running_count."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MachineWorkset:
        return cls(**data)


@dataclass(slots=True)
class AgentSnapshot:
    schema_version: str
    machine_name: str
    agent_mode: str
    agent_state: str
    pid: int | None
    started_at: str | None
    last_heartbeat: str | None
    last_transition_at: str | None
    idle_timeout_seconds: int
    idle_started_at: str | None
    idle_deadline_at: str | None
    drain_started_at: str | None
    last_exit_reason: str | None
    workset: MachineWorkset

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
        if not isinstance(self.idle_timeout_seconds, int) or self.idle_timeout_seconds < 0:
            raise ValueError("idle_timeout_seconds must be a non-negative integer.")
        if self.workset.machine_name != self.machine_name:
            raise ValueError("AgentSnapshot.workset.machine_name must match machine_name.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": {
                "schema_version": self.schema_version,
                "machine_name": self.machine_name,
                "agent_mode": self.agent_mode,
                "agent_state": self.agent_state,
                "pid": self.pid,
                "started_at": self.started_at,
                "last_heartbeat": self.last_heartbeat,
                "last_transition_at": self.last_transition_at,
                "idle_timeout_seconds": self.idle_timeout_seconds,
                "idle_started_at": self.idle_started_at,
                "idle_deadline_at": self.idle_deadline_at,
                "drain_started_at": self.drain_started_at,
                "last_exit_reason": self.last_exit_reason,
                "workset": self.workset.to_dict(),
            }
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentSnapshot:
        payload = data["agent"]
        workset_payload = payload.get("workset", {
            "machine_name": payload["machine_name"],
            "has_launch_backlog": False,
            "has_active_responsibility": False,
            "updated_at": payload.get("last_heartbeat") or utc_now_iso(),
        })
        return cls(
            schema_version=payload.get("schema_version", SCHEMA_VERSION),
            machine_name=payload["machine_name"],
            agent_mode=payload["agent_mode"],
            agent_state=payload["agent_state"],
            pid=payload.get("pid"),
            started_at=payload.get("started_at"),
            last_heartbeat=payload.get("last_heartbeat"),
            last_transition_at=payload.get("last_transition_at"),
            idle_timeout_seconds=payload.get("idle_timeout_seconds", 0),
            idle_started_at=payload.get("idle_started_at"),
            idle_deadline_at=payload.get("idle_deadline_at"),
            drain_started_at=payload.get("drain_started_at"),
            last_exit_reason=payload.get("last_exit_reason"),
            workset=MachineWorkset.from_dict(workset_payload),
        )


@dataclass(slots=True)
class MachineSummary:
    machine_name: str
    counts_by_phase: dict[str, int]
    updated_at: str

    def __post_init__(self) -> None:
        validate_machine_name(self.machine_name)
        cleaned_counts: dict[str, int] = {}
        for phase, count in self.counts_by_phase.items():
            validate_phase(phase)
            if not isinstance(count, int) or count < 0:
                raise ValueError("MachineSummary counts must be non-negative integers.")
            cleaned_counts[phase] = count
        self.counts_by_phase = cleaned_counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "machine_name": self.machine_name,
                "counts_by_phase": dict(self.counts_by_phase),
                "updated_at": self.updated_at,
            }
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MachineSummary:
        payload = data["summary"]
        return cls(
            machine_name=payload["machine_name"],
            counts_by_phase=payload.get("counts_by_phase", {}),
            updated_at=payload["updated_at"],
        )
