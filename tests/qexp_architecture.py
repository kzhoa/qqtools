"""Deterministic qexp test architecture primitives.

This module is test-only on purpose.  It models protocol decisions without adding a
test switch to the production qexp authority implementation.
"""
from __future__ import annotations

import json
import select
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from tests.qexp_test_support import TestResourceScope


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

    def request(self, command: str, *, timeout_seconds: float = 2.0) -> dict[str, Any]:
        """Send one control command and wait for its structured response."""
        if not command:
            raise ValueError("participant command must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("participant request timeout must be positive")
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError(f"participant {self.name} has no IPC pipes")
        self.trace.record(
            "participant.command",
            {"command": command, "pid": self.process.pid},
            0.0,
            self.name,
        )
        self.process.stdin.write(json.dumps({"command": command}) + "\n")
        self.process.stdin.flush()
        ready, _, _ = select.select([self.process.stdout], [], [], timeout_seconds)
        if not ready:
            raise RuntimeError(
                f"participant {self.name} did not respond within {timeout_seconds} seconds"
            )
        response = self.process.stdout.readline()
        if not response:
            raise RuntimeError(f"participant {self.name} exited before responding")
        result = json.loads(response)
        self.trace.record("participant.response", result, 0.0, self.name)
        return result

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            self.request("exit")
            self.process.wait(timeout=2.0)
        except (RuntimeError, subprocess.TimeoutExpired):
            self.process.terminate()
            self.process.wait(timeout=2.0)

    def kill(self) -> None:
        """Terminate this participant without attempting graceful protocol cleanup."""
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=2.0)


class SingleHostMachineLab:
    """Starts independently imported participants that share only a project root."""

    _PARTICIPANT = """
import json, os, sys
for line in sys.stdin:
    command = json.loads(line)["command"]
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
        self.shared_root = root / "shared"
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
        process = subprocess.Popen(
            [sys.executable, "-c", self._PARTICIPANT],
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        return MachineParticipant(name, scope, process, self.trace)
