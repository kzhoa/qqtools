from __future__ import annotations

import json
from pathlib import Path

from .layout import RootConfig, agent_state_path, summary_state_path
from .indexes import load_index
from .models import (
    AGENT_MODE_ON_DEMAND,
    AGENT_STATE_ACTIVE,
    AGENT_STATE_DRAINING,
    AGENT_STATE_IDLE,
    AGENT_STATE_STOPPED,
    AGENT_STATES,
    ALL_PHASES,
    AgentSnapshot,
    MachineSummary,
    MachineWorkset,
    PHASE_DISPATCHING,
    PHASE_QUEUED,
    PHASE_RUNNING,
    PHASE_STARTING,
    SCHEMA_VERSION,
    utc_now_iso,
)
from .storage import load_task, read_json, write_atomic_json


def build_machine_workset(
    cfg: RootConfig,
    machine_name: str | None = None,
    updated_at: str | None = None,
) -> MachineWorkset:
    target_machine = machine_name or cfg.machine_name
    task_ids = load_index(cfg, "machine", target_machine)
    counts_by_phase = {phase: 0 for phase in ALL_PHASES}

    for task_id in task_ids:
        try:
            task = load_task(cfg, task_id)
        except FileNotFoundError:
            continue
        if task.machine_name != target_machine:
            continue
        counts_by_phase[task.status.phase] = counts_by_phase.get(task.status.phase, 0) + 1

    queued_count = counts_by_phase.get(PHASE_QUEUED, 0)
    dispatching_count = counts_by_phase.get(PHASE_DISPATCHING, 0)
    starting_count = counts_by_phase.get(PHASE_STARTING, 0)
    running_count = counts_by_phase.get(PHASE_RUNNING, 0)
    terminal_count = sum(
        count
        for phase, count in counts_by_phase.items()
        if phase not in {PHASE_QUEUED, PHASE_DISPATCHING, PHASE_STARTING, PHASE_RUNNING}
    )
    has_launch_backlog = queued_count > 0 or dispatching_count > 0 or starting_count > 0
    has_active_responsibility = has_launch_backlog or running_count > 0

    return MachineWorkset(
        machine_name=target_machine,
        queued_count=queued_count,
        dispatching_count=dispatching_count,
        starting_count=starting_count,
        running_count=running_count,
        terminal_count=terminal_count,
        has_launch_backlog=has_launch_backlog,
        has_active_responsibility=has_active_responsibility,
        updated_at=updated_at or utc_now_iso(),
    )


def derive_agent_state(workset: MachineWorkset) -> str:
    if workset.has_launch_backlog:
        return AGENT_STATE_ACTIVE
    if workset.running_count > 0:
        return AGENT_STATE_DRAINING
    return AGENT_STATE_IDLE


def build_summary_snapshot(workset: MachineWorkset) -> MachineSummary:
    counts_by_phase = {
        PHASE_QUEUED: workset.queued_count,
        PHASE_DISPATCHING: workset.dispatching_count,
        PHASE_STARTING: workset.starting_count,
        PHASE_RUNNING: workset.running_count,
    }
    return MachineSummary(
        machine_name=workset.machine_name,
        counts_by_phase=counts_by_phase,
        updated_at=workset.updated_at,
    )


def read_agent_snapshot(cfg: RootConfig) -> AgentSnapshot | None:
    return _read_snapshot(agent_state_path(cfg), AgentSnapshot)


def write_agent_snapshot(cfg: RootConfig, snapshot: AgentSnapshot) -> None:
    write_atomic_json(agent_state_path(cfg), snapshot.to_dict())


def read_summary_snapshot(cfg: RootConfig) -> MachineSummary | None:
    return _read_snapshot(summary_state_path(cfg), MachineSummary)


def write_summary_snapshot(cfg: RootConfig, summary: MachineSummary) -> None:
    write_atomic_json(summary_state_path(cfg), summary.to_dict())


def build_starting_agent_snapshot(
    cfg: RootConfig,
    persistent: bool,
    pid: int,
    idle_timeout_seconds: int = 600,
    started_at: str | None = None,
) -> AgentSnapshot:
    now = started_at or utc_now_iso()
    idle_timeout_seconds = 0 if persistent else idle_timeout_seconds
    return AgentSnapshot(
        schema_version=SCHEMA_VERSION,
        machine_name=cfg.machine_name,
        agent_mode="persistent" if persistent else AGENT_MODE_ON_DEMAND,
        agent_state="starting",
        pid=pid,
        started_at=now,
        last_heartbeat=now,
        last_transition_at=now,
        idle_timeout_seconds=idle_timeout_seconds,
        idle_started_at=None,
        idle_deadline_at=None,
        drain_started_at=None,
        last_exit_reason=None,
        workset=MachineWorkset(
            machine_name=cfg.machine_name,
            has_launch_backlog=False,
            has_active_responsibility=False,
            updated_at=now,
        ),
    )


def build_stopped_agent_snapshot(
    previous: AgentSnapshot | None,
    reason: str,
    stopped_at: str | None = None,
) -> AgentSnapshot:
    now = stopped_at or utc_now_iso()
    if previous is None:
        machine_name = "unknown"
        agent_mode = AGENT_MODE_ON_DEMAND
        idle_timeout_seconds = 600
        workset = MachineWorkset(
            machine_name=machine_name,
            has_launch_backlog=False,
            has_active_responsibility=False,
            updated_at=now,
        )
        started_at = None
    else:
        machine_name = previous.machine_name
        agent_mode = previous.agent_mode
        idle_timeout_seconds = previous.idle_timeout_seconds
        workset = previous.workset
        started_at = previous.started_at
    return AgentSnapshot(
        schema_version=SCHEMA_VERSION,
        machine_name=machine_name,
        agent_mode=agent_mode,
        agent_state=AGENT_STATE_STOPPED,
        pid=None,
        started_at=started_at,
        last_heartbeat=now,
        last_transition_at=now,
        idle_timeout_seconds=idle_timeout_seconds,
        idle_started_at=None,
        idle_deadline_at=None,
        drain_started_at=None,
        last_exit_reason=reason,
        workset=workset,
    )


def is_dynamic_agent_state(agent_state: str) -> bool:
    return agent_state in {
        "starting",
        AGENT_STATE_ACTIVE,
        AGENT_STATE_DRAINING,
        AGENT_STATE_IDLE,
    }


def summarize_agent_snapshot(snapshot: AgentSnapshot | None) -> dict[str, object]:
    if snapshot is None:
        return {
            "agent_state": AGENT_STATE_STOPPED,
            "pid": None,
            "machine_name": None,
            "last_heartbeat": None,
        }
    workset = snapshot.workset
    return {
        "schema_version": snapshot.schema_version,
        "machine_name": snapshot.machine_name,
        "agent_mode": snapshot.agent_mode,
        "agent_state": snapshot.agent_state,
        "pid": snapshot.pid,
        "started_at": snapshot.started_at,
        "last_heartbeat": snapshot.last_heartbeat,
        "last_transition_at": snapshot.last_transition_at,
        "idle_timeout_seconds": snapshot.idle_timeout_seconds,
        "idle_started_at": snapshot.idle_started_at,
        "idle_deadline_at": snapshot.idle_deadline_at,
        "drain_started_at": snapshot.drain_started_at,
        "last_exit_reason": snapshot.last_exit_reason,
        "workset": workset.to_dict(),
    }


def _read_snapshot(path: Path, cls: type[AgentSnapshot] | type[MachineSummary]):
    if not path.is_file():
        return None
    try:
        payload = read_json(path)
    except (json.JSONDecodeError, OSError):
        return None
    try:
        return cls.from_dict(payload)
    except (KeyError, TypeError, ValueError):
        return None
