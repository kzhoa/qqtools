"""Deterministic qexp test architecture primitives.

This module is test-only on purpose.  It models protocol decisions without adding a
test switch to the production qexp authority implementation.
"""
from __future__ import annotations

import json
import random
import select
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from tests.support.qexp.resources import TestResourceScope


class ProtocolPoint(StrEnum):
    """Stable protocol boundaries used by crash-window tests."""

    SNAPSHOT_READ = "snapshot_read"
    LOCK_ACQUIRE = "lock_acquire"
    AUTHORITATIVE_REVALIDATION = "authoritative_revalidation"
    CAS = "cas"
    TEMP_WRITE = "temp_write"
    FILE_FSYNC = "file_fsync"
    ATOMIC_REPLACE = "atomic_replace"
    DIRECTORY_FSYNC = "directory_fsync"
    INDEX_PUBLISH = "index_publish"
    INDEX_REMOVE = "index_remove"
    PROCESS_CREATE = "process_create"
    REGISTRATION_PUBLISH = "registration_publish"


class InjectedProtocolError(RuntimeError):
    """Raised when a scheduled protocol fault is reached."""


class SimulatedProtocolPause(RuntimeError):
    """Raised when an interleaving pauses one simulated participant."""


class SimulatedParticipantCrash(RuntimeError):
    """Raised when an interleaving crashes one simulated participant."""


class ProtocolDisposition(StrEnum):
    """The scheduled outcome for one explicit protocol yield point."""

    CONTINUE = "continue"
    CONFLICT = "conflict"
    PAUSE = "pause"
    CRASH = "crash"


class CrashWindow(StrEnum):
    """Runtime Spec crash windows covered by deterministic recovery decisions."""

    SUBMISSION_BEFORE_OPERATION = "CW-01"
    OPERATION_BEFORE_TASK_STAGING = "CW-02"
    PARTIAL_TASK_STAGING = "CW-03"
    WORKER_BEFORE_COMMIT = "CW-04"
    RESERVATION_BEFORE_CLAIM = "CW-05"
    CLAIM_BEFORE_ATTEMPT = "CW-06"
    ATTEMPT_BEFORE_LAUNCH_GATE = "CW-07"
    LAUNCH_GATE_BEFORE_PROCESS = "CW-08"
    PROCESS_BEFORE_METADATA = "CW-09"
    PROCESS_EXIT_BEFORE_TERMINAL = "CW-10"
    LEASE_EXPIRY_BEFORE_AUTHORIZATION = "CW-11"
    LEASE_EXPIRY_AFTER_AUTHORIZATION = "CW-12"
    ORPHAN_TOKEN_RECOVERY = "CW-13"
    SUCCESSOR_FENCING = "CW-14"
    WORKER_REMOVAL_RACE = "CW-15"
    CANCEL_LAUNCH_RACE = "CW-16"
    RECOVERY_DRAIN_RACE = "CW-17"
    CANCELLATION_RESTART = "CW-18"


@dataclass(frozen=True, slots=True)
class CrashWindowRecoveryPlan:
    """Pure convergence decision for one Runtime Spec crash window."""

    action: str
    invariant: str


_CRASH_WINDOW_RECOVERY_PLANS: dict[CrashWindow, CrashWindowRecoveryPlan] = {
    CrashWindow.SUBMISSION_BEFORE_OPERATION: CrashWindowRecoveryPlan(
        "preserve_absence", "no_operation_or_task"
    ),
    CrashWindow.OPERATION_BEFORE_TASK_STAGING: CrashWindowRecoveryPlan(
        "abort_or_resume_preparing_submission", "no_claimable_task"
    ),
    CrashWindow.PARTIAL_TASK_STAGING: CrashWindowRecoveryPlan(
        "complete_or_abort_submission", "no_partial_visibility"
    ),
    CrashWindow.WORKER_BEFORE_COMMIT: CrashWindowRecoveryPlan(
        "remove_inactive_worker_addition", "worker_inactive_until_commit"
    ),
    CrashWindow.RESERVATION_BEFORE_CLAIM: CrashWindowRecoveryPlan(
        "release_or_expire_reservation", "task_remains_queued"
    ),
    CrashWindow.CLAIM_BEFORE_ATTEMPT: CrashWindowRecoveryPlan(
        "release_claim_and_reservation", "no_launch_without_attempt"
    ),
    CrashWindow.ATTEMPT_BEFORE_LAUNCH_GATE: CrashWindowRecoveryPlan(
        "honor_pause_or_cancel", "no_process_without_authorization"
    ),
    CrashWindow.LAUNCH_GATE_BEFORE_PROCESS: CrashWindowRecoveryPlan(
        "recover_starting_or_fail_spawn", "no_untracked_process"
    ),
    CrashWindow.PROCESS_BEFORE_METADATA: CrashWindowRecoveryPlan(
        "reconcile_manifest_and_token", "one_attempt_identity"
    ),
    CrashWindow.PROCESS_EXIT_BEFORE_TERMINAL: CrashWindowRecoveryPlan(
        "republish_terminal_idempotently", "terminal_truth_converges"
    ),
    CrashWindow.LEASE_EXPIRY_BEFORE_AUTHORIZATION: CrashWindowRecoveryPlan(
        "return_claim_to_queue", "no_automatic_launch"
    ),
    CrashWindow.LEASE_EXPIRY_AFTER_AUTHORIZATION: CrashWindowRecoveryPlan(
        "block_task_as_orphaned", "no_automatic_replacement"
    ),
    CrashWindow.ORPHAN_TOKEN_RECOVERY: CrashWindowRecoveryPlan(
        "issue_successor_for_same_attempt", "attempt_identity_preserved"
    ),
    CrashWindow.SUCCESSOR_FENCING: CrashWindowRecoveryPlan(
        "reject_stale_writer", "stale_token_cannot_mutate_truth"
    ),
    CrashWindow.WORKER_REMOVAL_RACE: CrashWindowRecoveryPlan(
        "serialize_under_group_lock", "queued_work_not_stranded"
    ),
    CrashWindow.CANCEL_LAUNCH_RACE: CrashWindowRecoveryPlan(
        "serialize_final_launch_gate", "one_group_lock_order_wins"
    ),
    CrashWindow.RECOVERY_DRAIN_RACE: CrashWindowRecoveryPlan(
        "serialize_group_then_task_recovery", "forbidden_recovery_is_quarantined"
    ),
    CrashWindow.CANCELLATION_RESTART: CrashWindowRecoveryPlan(
        "resume_group_cancellation", "pending_machine_set_preserved"
    ),
}


def plan_crash_window_recovery(window: CrashWindow) -> CrashWindowRecoveryPlan:
    """Return the reference recovery decision for a Runtime Spec crash window."""
    try:
        return _CRASH_WINDOW_RECOVERY_PLANS[window]
    except KeyError as exc:
        raise ValueError(f"unsupported crash window: {window}") from exc


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One replayable test action or protocol observation."""

    sequence: int
    kind: str
    payload: Mapping[str, Any]
    logical_time: float
    participant: str | None = None


@dataclass(slots=True)
class TraceEnvelope:
    """Serializable, secret-safe trace shared by model and process tests."""

    scenario_version: str
    seed: int
    events: list[TraceEvent] = field(default_factory=list)

    def record(
        self,
        kind: str,
        payload: Mapping[str, Any],
        logical_time: float,
        participant: str | None = None,
    ) -> None:
        self.events.append(
            TraceEvent(len(self.events), kind, _redact_payload(payload), logical_time, participant)
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "scenario_version": self.scenario_version,
                "seed": self.seed,
                "events": [
                    {
                        "sequence": event.sequence,
                        "kind": event.kind,
                        "payload": dict(event.payload),
                        "logical_time": event.logical_time,
                        "participant": event.participant,
                    }
                    for event in self.events
                ],
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, value: str) -> "TraceEnvelope":
        data = json.loads(value)
        trace = cls(data["scenario_version"], data["seed"])
        for event in data["events"]:
            trace.events.append(TraceEvent(**event))
        return trace


def _redact_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    """Avoid persisting secret-bearing environment and command values in traces."""
    return {
        key: "<redacted>"
        if any(
            part in key.lower()
            for part in ("secret", "password", "credential", "api_key", "authorization")
        )
        else _redact_value(item)
        for key, item in value.items()
    }


def _redact_value(value: Any) -> Any:
    """Recursively redact a JSON-compatible trace value."""
    if isinstance(value, Mapping):
        return _redact_payload(value)
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class RevisionedValue:
    revision: int
    value: Any


ModelTaskState = Literal["queued", "claimed", "starting", "running", "cancelled", "finished"]
ReferenceCommandKind = Literal[
    "create_submission",
    "commit_submission",
    "abort_submission",
    "stage_task",
    "claim",
    "cancel",
    "authorize_launch",
    "publish_running",
    "publish_terminal",
]


@dataclass(frozen=True, slots=True)
class ModelEffectPlan:
    """One external effect that a reference-model command authorizes."""

    kind: Literal["launch", "terminate"]
    task_id: str
    fencing_token: int


@dataclass(slots=True)
class ModelTask:
    """The minimal observable Task truth used by the reference model."""

    task_id: str
    submission_id: str
    state: ModelTaskState = "queued"
    fencing_token: int = 0
    has_cancellation_requested: bool = False


@dataclass(frozen=True, slots=True)
class ReferenceCommand:
    """Serializable reference-model action used by generated scenarios and replay."""

    kind: ReferenceCommandKind
    task_id: str | None = None
    submission_id: str | None = None
    fencing_token: int | None = None
    is_cancelled: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "task_id": self.task_id,
            "submission_id": self.submission_id,
            "fencing_token": self.fencing_token,
            "is_cancelled": self.is_cancelled,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ReferenceCommand":
        return cls(
            kind=payload["kind"],
            task_id=payload.get("task_id"),
            submission_id=payload.get("submission_id"),
            fencing_token=payload.get("fencing_token"),
            is_cancelled=payload.get("is_cancelled", False),
        )


class QexpReferenceModel:
    """Independent model of claim, fencing, and launch-authority semantics."""

    def __init__(self) -> None:
        self._submission_states: dict[str, Literal["preparing", "committed", "aborted"]] = {}
        self._tasks: dict[str, ModelTask] = {}

    def create_submission(self, submission_id: str) -> None:
        if submission_id in self._submission_states:
            raise ValueError(f"submission already exists: {submission_id}")
        self._submission_states[submission_id] = "preparing"

    def commit_submission(self, submission_id: str) -> None:
        self._require_submission(submission_id, "preparing")
        self._submission_states[submission_id] = "committed"

    def abort_submission(self, submission_id: str) -> None:
        self._require_submission(submission_id, "preparing")
        self._submission_states[submission_id] = "aborted"

    def stage_task(self, task_id: str, submission_id: str) -> None:
        if task_id in self._tasks:
            raise ValueError(f"task already exists: {task_id}")
        if submission_id not in self._submission_states:
            raise ValueError(f"submission does not exist: {submission_id}")
        self._tasks[task_id] = ModelTask(task_id, submission_id)

    def claim(self, task_id: str) -> int | None:
        task = self._task(task_id)
        if (
            task.state != "queued"
            or task.has_cancellation_requested
            or self._submission_states[task.submission_id] != "committed"
        ):
            return None
        task.fencing_token += 1
        task.state = "claimed"
        self.assert_invariants()
        return task.fencing_token

    def cancel(self, task_id: str) -> ModelEffectPlan | None:
        task = self._task(task_id)
        task.has_cancellation_requested = True
        if task.state in {"queued", "claimed", "starting"}:
            task.state = "cancelled"
            self.assert_invariants()
            return None
        if task.state == "running":
            self.assert_invariants()
            return ModelEffectPlan("terminate", task.task_id, task.fencing_token)
        return None

    def authorize_launch(self, task_id: str, fencing_token: int) -> ModelEffectPlan | None:
        task = self._task(task_id)
        if (
            task.state != "claimed"
            or task.has_cancellation_requested
            or task.fencing_token != fencing_token
        ):
            return None
        task.state = "starting"
        self.assert_invariants()
        return ModelEffectPlan("launch", task.task_id, fencing_token)

    def publish_running(self, task_id: str, fencing_token: int) -> bool:
        task = self._task(task_id)
        if task.state != "starting" or task.fencing_token != fencing_token:
            return False
        task.state = "running"
        self.assert_invariants()
        return True

    def publish_terminal(self, task_id: str, fencing_token: int, *, cancelled: bool) -> bool:
        task = self._task(task_id)
        if task.state not in {"starting", "running"} or task.fencing_token != fencing_token:
            return False
        task.state = "cancelled" if cancelled else "finished"
        self.assert_invariants()
        return True

    def task_state(self, task_id: str) -> ModelTaskState:
        return self._task(task_id).state

    def available_commands(self) -> tuple[ReferenceCommand, ...]:
        """Return only model commands valid for the current state."""
        commands: list[ReferenceCommand] = []
        for submission_id, state in sorted(self._submission_states.items()):
            if state == "preparing":
                commands.extend(
                    (
                        ReferenceCommand("commit_submission", submission_id=submission_id),
                        ReferenceCommand("abort_submission", submission_id=submission_id),
                    )
                )
                for task_id, task in sorted(self._tasks.items()):
                    if task.submission_id == submission_id and task.state == "queued":
                        commands.append(ReferenceCommand("cancel", task_id=task_id))
        for task_id, task in sorted(self._tasks.items()):
            if (
                task.state == "queued"
                and self._submission_states[task.submission_id] == "committed"
            ):
                commands.extend(
                    (
                        ReferenceCommand("claim", task_id=task_id),
                        ReferenceCommand("cancel", task_id=task_id),
                    )
                )
            elif task.state == "claimed":
                commands.extend(
                    (
                        ReferenceCommand(
                            "authorize_launch",
                            task_id=task_id,
                            fencing_token=task.fencing_token,
                        ),
                        ReferenceCommand("cancel", task_id=task_id),
                    )
                )
            elif task.state == "starting":
                commands.extend(
                    (
                        ReferenceCommand(
                            "publish_running",
                            task_id=task_id,
                            fencing_token=task.fencing_token,
                        ),
                        ReferenceCommand("cancel", task_id=task_id),
                    )
                )
            elif task.state == "running":
                commands.extend(
                    (
                        ReferenceCommand(
                            "publish_terminal",
                            task_id=task_id,
                            fencing_token=task.fencing_token,
                            is_cancelled=False,
                        ),
                        ReferenceCommand("cancel", task_id=task_id),
                    )
                )
        return tuple(commands)

    def execute(self, command: ReferenceCommand) -> bool | int | ModelEffectPlan | None:
        """Apply one serializable command and return its normalized model outcome."""
        if command.kind == "create_submission":
            self.create_submission(self._required_text(command.submission_id, "submission_id"))
            return None
        if command.kind == "commit_submission":
            self.commit_submission(self._required_text(command.submission_id, "submission_id"))
            return None
        if command.kind == "abort_submission":
            self.abort_submission(self._required_text(command.submission_id, "submission_id"))
            return None
        if command.kind == "stage_task":
            self.stage_task(
                self._required_text(command.task_id, "task_id"),
                self._required_text(command.submission_id, "submission_id"),
            )
            return None
        task_id = self._required_text(command.task_id, "task_id")
        if command.kind == "claim":
            return self.claim(task_id)
        if command.kind == "cancel":
            return self.cancel(task_id)
        fencing_token = self._required_token(command.fencing_token)
        if command.kind == "authorize_launch":
            return self.authorize_launch(task_id, fencing_token)
        if command.kind == "publish_running":
            return self.publish_running(task_id, fencing_token)
        if command.kind == "publish_terminal":
            return self.publish_terminal(task_id, fencing_token, cancelled=command.is_cancelled)
        raise ValueError(f"unsupported reference command: {command.kind}")

    def assert_invariants(self) -> None:
        for task in self._tasks.values():
            submission_state = self._submission_states[task.submission_id]
            if task.state in {"claimed", "starting", "running"} and submission_state != "committed":
                raise AssertionError(f"active task has uncommitted submission: {task.task_id}")
            if task.fencing_token < 0:
                raise AssertionError(f"task has invalid fencing token: {task.task_id}")

    def _require_submission(
        self,
        submission_id: str,
        expected_state: Literal["preparing"],
    ) -> None:
        if self._submission_states.get(submission_id) != expected_state:
            raise ValueError(f"submission is not {expected_state}: {submission_id}")

    def _task(self, task_id: str) -> ModelTask:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise ValueError(f"task does not exist: {task_id}") from exc

    @staticmethod
    def _required_text(value: str | None, field_name: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError(f"reference command requires a non-empty {field_name}")
        return value

    @staticmethod
    def _required_token(value: int | None) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError("reference command requires a positive fencing_token")
        return value


class ReferenceScenarioRunner:
    """Runs and replays serialized reference-model commands with a trace envelope."""

    def __init__(self, *, seed: int) -> None:
        self.seed = seed
        self.trace = TraceEnvelope("qexp-reference-model/v1", seed)

    def run(self, commands: Iterable[ReferenceCommand]) -> QexpReferenceModel:
        model = QexpReferenceModel()
        for index, command in enumerate(commands):
            self.trace.record("model.command", command.to_payload(), float(index))
            result = model.execute(command)
            self.trace.record("model.result", _reference_result_payload(result), float(index))
        model.assert_invariants()
        return model

    def generate(self, *, maximum_actions: int) -> list[ReferenceCommand]:
        """Generate a bounded valid lifecycle sequence from the runner seed."""
        if maximum_actions < 4:
            raise ValueError("maximum_actions must allow submission staging and commit")
        generator = random.Random(self.seed)
        commands = [
            ReferenceCommand("create_submission", submission_id="submission-1"),
            ReferenceCommand("stage_task", task_id="task-1", submission_id="submission-1"),
            ReferenceCommand("commit_submission", submission_id="submission-1"),
        ]
        model = QexpReferenceModel()
        for command in commands:
            model.execute(command)
        while len(commands) < maximum_actions:
            available = model.available_commands()
            if not available:
                break
            command = generator.choice(available)
            commands.append(command)
            model.execute(command)
        return commands

    @classmethod
    def replay(cls, trace: TraceEnvelope) -> QexpReferenceModel:
        if trace.scenario_version != "qexp-reference-model/v1":
            raise ValueError("trace does not contain a reference-model scenario")
        commands = [
            ReferenceCommand.from_payload(event.payload)
            for event in trace.events
            if event.kind == "model.command"
        ]
        return cls(seed=trace.seed).run(commands)


def _reference_result_payload(result: bool | int | ModelEffectPlan | None) -> dict[str, Any]:
    """Convert a reference-model outcome into trace-safe JSON data."""
    if isinstance(result, ModelEffectPlan):
        return {
            "effect": result.kind,
            "task_id": result.task_id,
            "fencing_token": result.fencing_token,
        }
    return {"result": result}


class SimulatedRuntime:
    """Revisioned store, lock model, virtual clock, and fault-injection port."""

    def __init__(self, *, seed: int = 0) -> None:
        self.clock = 0.0
        self.seed = seed
        self.trace = TraceEnvelope("qexp-simulated-runtime/v1", seed)
        self._records: dict[str, RevisionedValue] = {}
        self._locks: set[str] = set()
        self._faults: dict[ProtocolPoint, list[BaseException]] = {}
        self._dispositions: dict[ProtocolPoint, list[ProtocolDisposition]] = {}

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("virtual clock cannot move backwards")
        self.clock += seconds
        self.trace.record("clock.advance", {"seconds": seconds}, self.clock)

    def inject_fault(self, point: ProtocolPoint, error: BaseException | None = None) -> None:
        self._faults.setdefault(point, []).append(error or InjectedProtocolError(point))

    def schedule(self, point: ProtocolPoint, disposition: ProtocolDisposition) -> None:
        """Schedule one deterministic interleaving outcome at a protocol point."""
        self._dispositions.setdefault(point, []).append(disposition)

    def yield_at(
        self,
        point: ProtocolPoint,
        *,
        participant: str | None = None,
    ) -> ProtocolDisposition:
        """Record and apply one scheduled protocol boundary outcome."""
        disposition = self._next_disposition(point)
        self.trace.record(
            "protocol.yield",
            {"point": point, "disposition": disposition},
            self.clock,
            participant,
        )
        faults = self._faults.get(point)
        if faults:
            raise faults.pop(0)
        if disposition == ProtocolDisposition.PAUSE:
            raise SimulatedProtocolPause(f"paused at {point}")
        if disposition == ProtocolDisposition.CRASH:
            raise SimulatedParticipantCrash(f"crashed at {point}")
        return disposition

    def read(self, key: str) -> RevisionedValue | None:
        self.yield_at(ProtocolPoint.SNAPSHOT_READ)
        return self._records.get(key)

    def cas(self, key: str, revision: int | None, value: Any) -> RevisionedValue | None:
        if self.yield_at(ProtocolPoint.CAS) == ProtocolDisposition.CONFLICT:
            self.trace.record("store.conflict", {"key": key, "revision": revision}, self.clock)
            return None
        current = self._records.get(key)
        if (current.revision if current else None) != revision:
            self.trace.record("store.conflict", {"key": key, "revision": revision}, self.clock)
            return None
        next_value = RevisionedValue(1 if current is None else current.revision + 1, value)
        self._records[key] = next_value
        self.trace.record("store.commit", {"key": key, "revision": next_value.revision}, self.clock)
        return next_value

    def atomic_replace(
        self,
        key: str,
        revision: int | None,
        value: Any,
        *,
        participant: str | None = None,
    ) -> RevisionedValue | None:
        """Model the durable write boundaries before committing a revisioned record."""
        for point in (
            ProtocolPoint.TEMP_WRITE,
            ProtocolPoint.FILE_FSYNC,
            ProtocolPoint.ATOMIC_REPLACE,
            ProtocolPoint.DIRECTORY_FSYNC,
        ):
            if self.yield_at(point, participant=participant) == ProtocolDisposition.CONFLICT:
                self.trace.record("store.conflict", {"key": key, "revision": revision}, self.clock)
                return None
        return self.cas(key, revision, value)

    def publish_index(self, key: str, *, participant: str | None = None) -> None:
        """Record an advisory index publication boundary."""
        self.yield_at(ProtocolPoint.INDEX_PUBLISH, participant=participant)
        self.trace.record("index.publish", {"key": key}, self.clock, participant)

    def remove_index(self, key: str, *, participant: str | None = None) -> None:
        """Record an advisory index removal boundary."""
        self.yield_at(ProtocolPoint.INDEX_REMOVE, participant=participant)
        self.trace.record("index.remove", {"key": key}, self.clock, participant)

    def create_process(
        self,
        attempt_id: str,
        *,
        participant: str | None = None,
    ) -> None:
        """Record a simulated process creation boundary."""
        self.yield_at(ProtocolPoint.PROCESS_CREATE, participant=participant)
        self.trace.record(
            "process.create",
            {"attempt_id": attempt_id},
            self.clock,
            participant,
        )

    def publish_registration(
        self,
        attempt_id: str,
        *,
        participant: str | None = None,
    ) -> None:
        """Record a simulated process-registration boundary."""
        self.yield_at(ProtocolPoint.REGISTRATION_PUBLISH, participant=participant)
        self.trace.record(
            "registration.publish",
            {"attempt_id": attempt_id},
            self.clock,
            participant,
        )

    def acquire(self, name: str) -> bool:
        if self.yield_at(ProtocolPoint.LOCK_ACQUIRE) == ProtocolDisposition.CONFLICT:
            return False
        if name in self._locks:
            return False
        self._locks.add(name)
        return True

    def release(self, name: str) -> None:
        self._locks.discard(name)

    def _next_disposition(self, point: ProtocolPoint) -> ProtocolDisposition:
        dispositions = self._dispositions.get(point)
        if not dispositions:
            return ProtocolDisposition.CONTINUE
        return dispositions.pop(0)


Action = Callable[[SimulatedRuntime], None]


class ScenarioRunner:
    """Executes replayable actions and minimizes a failing sequence by deletion."""

    def __init__(self, runtime: SimulatedRuntime) -> None:
        self.runtime = runtime

    def run(self, actions: Iterable[Action]) -> None:
        for index, action in enumerate(actions):
            self.runtime.trace.record(
                "action", {"index": index, "name": action.__name__}, self.runtime.clock
            )
            action(self.runtime)

    @staticmethod
    def shrink(actions: list[Action], fails: Callable[[list[Action]], bool]) -> list[Action]:
        """Return a deletion-minimal failing action sequence."""
        result = list(actions)
        index = 0
        while index < len(result):
            candidate = result[:index] + result[index + 1 :]
            if fails(candidate):
                result = candidate
            else:
                index += 1
        return result


@dataclass(slots=True)
class MachineParticipant:
    """A real child process with an intentionally tiny test-only control plane."""

    name: str
    scope: TestResourceScope
    process: subprocess.Popen[str]
    trace: TraceEnvelope

    def request(
        self,
        command: str,
        *,
        payload: Mapping[str, Any] | None = None,
        timeout_seconds: float = 2.0,
    ) -> dict[str, Any]:
        """Send one control command and wait for its structured response."""
        if not command:
            raise ValueError("participant command must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("participant request timeout must be positive")
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError(f"participant {self.name} has no IPC pipes")
        message = {"command": command, **dict(payload or {})}
        self.trace.record(
            "participant.command",
            {**message, "pid": self.process.pid},
            0.0,
            self.name,
        )
        try:
            self.process.stdin.write(json.dumps(message) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(
                f"participant {self.name} has a closed command pipe: {self._stderr_diagnostics()}"
            ) from exc
        ready, _, _ = select.select([self.process.stdout], [], [], timeout_seconds)
        if not ready:
            raise RuntimeError(
                f"participant {self.name} did not respond within {timeout_seconds} seconds"
            )
        response = self.process.stdout.readline()
        if not response:
            raise RuntimeError(
                f"participant {self.name} exited before responding: {self._stderr_diagnostics()}"
            )
        result = json.loads(response)
        self.trace.record("participant.response", result, 0.0, self.name)
        return result

    def close(self) -> None:
        if self.process.poll() is not None:
            self.scope.record_resource(
                "participant-exit",
                {"name": self.name, "pid": self.process.pid, "returncode": self.process.returncode},
            )
            return
        try:
            self.request("exit")
            self.process.wait(timeout=2.0)
        except (BrokenPipeError, RuntimeError, subprocess.TimeoutExpired) as exc:
            self.scope.record_cleanup_diagnostic(
                "participant-close",
                {"name": self.name, "pid": self.process.pid, "error": str(exc)},
            )
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=2.0)
        self.scope.record_resource(
            "participant-exit",
            {"name": self.name, "pid": self.process.pid, "returncode": self.process.returncode},
        )

    def kill(self) -> None:
        """Terminate this participant without attempting graceful protocol cleanup."""
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=2.0)
        self.scope.record_resource(
            "participant-exit",
            {"name": self.name, "pid": self.process.pid, "returncode": self.process.returncode},
        )

    def _stderr_diagnostics(self) -> str:
        if self.process.stderr is None or self.process.poll() is None:
            return "stderr unavailable while participant is running"
        return self.process.stderr.read().strip() or "no stderr output"


class SingleHostMachineLab:
    """Starts independently imported participants that share only a project root."""

    _PARTICIPANT = """
import json, os, sys
from pathlib import Path
for line in sys.stdin:
    message = json.loads(line)
    command = message["command"]
    if command == "exit":
        print(json.dumps({"exited": True}), flush=True)
        break
    if command == "identity":
        print(json.dumps({"pid": os.getpid(), "tmpdir": os.environ["TMPDIR"],
                          "runtime_root": os.environ["QEXP_MACHINE_RUNTIME_ROOT"]}), flush=True)
    elif command == "scheduler_authority":
        from qqtools.plugins.qexp.machine_runtime import MachineRuntime
        runtime = MachineRuntime()
        with runtime.scheduler_authority(blocking=False) as acquired:
            print(json.dumps({"acquired": acquired, "tmpdir": os.environ["TMPDIR"]}), flush=True)
    elif command == "claim_task":
        from qqtools.plugins.qexp.config_types import RootConfig
        from qqtools.plugins.qexp.scheduler import claim_task
        shared_root = Path(os.environ["QEXP_TEST_SHARED_ROOT"])
        cfg = RootConfig(
            shared_root,
            shared_root.parent,
            message["machine_name"],
            Path(os.environ["QEXP_MACHINE_RUNTIME_ROOT"]),
        )
        attempt = claim_task(cfg, message["task_id"], message["gpu_ids"])
        print(json.dumps({
            "claimed": attempt is not None,
            "attempt_id": attempt.attempt_id if attempt else None,
            "fencing_token": attempt.current_fencing_token if attempt else None,
        }), flush=True)
    elif command == "authorize_launch":
        from qqtools.plugins.qexp.config_types import RootConfig
        from qqtools.plugins.qexp.scheduler import authorize_launch
        shared_root = Path(os.environ["QEXP_TEST_SHARED_ROOT"])
        cfg = RootConfig(
            shared_root,
            shared_root.parent,
            message["machine_name"],
            Path(os.environ["QEXP_MACHINE_RUNTIME_ROOT"]),
        )
        authorized = authorize_launch(
            cfg,
            message["task_id"],
            message["attempt_id"],
            message["fencing_token"],
        )
        print(json.dumps({"authorized": authorized}), flush=True)
    elif command == "cancel_task":
        from qqtools.plugins.qexp.config_types import RootConfig
        from qqtools.plugins.qexp.scheduler import cancel_task
        shared_root = Path(os.environ["QEXP_TEST_SHARED_ROOT"])
        cfg = RootConfig(
            shared_root,
            shared_root.parent,
            message["machine_name"],
            Path(os.environ["QEXP_MACHINE_RUNTIME_ROOT"]),
        )
        task = cancel_task(cfg, message["task_id"])
        print(json.dumps({"state": task.state["projection"]}), flush=True)
    elif command == "fail_attempt":
        from qqtools.plugins.qexp.config_types import RootConfig
        from qqtools.plugins.qexp.scheduler import fail_attempt
        shared_root = Path(os.environ["QEXP_TEST_SHARED_ROOT"])
        cfg = RootConfig(
            shared_root,
            shared_root.parent,
            message["machine_name"],
            Path(os.environ["QEXP_MACHINE_RUNTIME_ROOT"]),
        )
        failed = fail_attempt(
            cfg,
            message["task_id"],
            message["attempt_id"],
            message["fencing_token"],
            "test_failure",
        )
        print(json.dumps({"failed": failed}), flush=True)
    elif command == "retry_task":
        from qqtools.plugins.qexp.commands.task import retry
        from qqtools.plugins.qexp.config_types import RootConfig
        shared_root = Path(os.environ["QEXP_TEST_SHARED_ROOT"])
        cfg = RootConfig(
            shared_root,
            shared_root.parent,
            message["machine_name"],
            Path(os.environ["QEXP_MACHINE_RUNTIME_ROOT"]),
        )
        task = retry(cfg, message["task_id"])
        print(json.dumps({"state": task.state["projection"]}), flush=True)
    elif command == "read_revision":
        from qqtools.plugins.qexp.runtime.store import read_json
        path = Path(os.environ["QEXP_TEST_SHARED_ROOT"]) / message["record_name"]
        print(json.dumps({"revision": read_json(path)["meta"]["revision"]}), flush=True)
    elif command == "cas_update":
        from qqtools.plugins.qexp.runtime.locks import exclusive
        from qqtools.plugins.qexp.runtime.store import CASConflict, cas_update, read_json
        path = Path(os.environ["QEXP_TEST_SHARED_ROOT"]) / message["record_name"]
        with exclusive(path.with_suffix(".lock")):
            value = read_json(path)
            value["value"] = message["value"]
            try:
                cas_update(path, message["expected_revision"], value)
            except CASConflict:
                print(json.dumps({"committed": False}), flush=True)
            else:
                print(json.dumps({"committed": True}), flush=True)
    elif command == "checkpoint":
        print(json.dumps({"checkpoint": "reached", "pid": os.getpid()}), flush=True)
        while True:
            follow_up = json.loads(sys.stdin.readline())["command"]
            if follow_up == "continue":
                print(json.dumps({"checkpoint": "continued", "pid": os.getpid()}), flush=True)
                break
            if follow_up == "exit":
                print(json.dumps({"exited": True}), flush=True)
                sys.exit(0)
            print(json.dumps({"error": "checkpoint_requires_continue"}), flush=True)
    else:
        print(json.dumps({"error": "unknown command"}), flush=True)
"""

    def __init__(self, root: Path, nodeid: str) -> None:
        self.root = root
        self.nodeid = nodeid
        self.shared_root = root / "shared-project" / ".qexp"
        self.shared_root.mkdir(parents=True, exist_ok=True)
        self.participants: list[MachineParticipant] = []
        self.trace = TraceEnvelope("qexp-single-host-machine-lab/v1", seed=0)

    def start(self, name: str) -> MachineParticipant:
        if any(item.name == name for item in self.participants):
            raise ValueError(f"duplicate machine participant: {name}")
        scope = TestResourceScope.create(self.root / "participants", f"{self.nodeid}-{name}")
        participant = self._start_with_scope(name, scope)
        self.participants.append(participant)
        return participant

    def restart(self, name: str) -> MachineParticipant:
        """Replace one terminated participant while preserving the shared project root."""
        for index, participant in enumerate(self.participants):
            if participant.name != name:
                continue
            if participant.process.poll() is None:
                raise RuntimeError(f"participant is still running: {name}")
            scope = TestResourceScope.create(
                self.root / "participants",
                f"{self.nodeid}-{name}-restart",
            )
            replacement = self._start_with_scope(name, scope)
            self.participants[index] = replacement
            return replacement
        raise ValueError(f"participant does not exist: {name}")

    def close(self) -> None:
        for participant in reversed(self.participants):
            participant.close()

    def _start_with_scope(self, name: str, scope: TestResourceScope) -> MachineParticipant:
        environment = scope.child_environment()
        environment["QEXP_TEST_SHARED_ROOT"] = str(self.shared_root)
        scope.record_resource("participant-intent", {"name": name})
        process = subprocess.Popen(
            [sys.executable, "-c", self._PARTICIPANT],
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        scope.record_resource("participant", {"name": name, "pid": process.pid})
        return MachineParticipant(name, scope, process, self.trace)
