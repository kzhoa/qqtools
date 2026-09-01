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
from typing import Any

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


class SimulatedRuntime:
    """Revisioned store, lock model, virtual clock, and fault-injection port."""

    def __init__(self, *, seed: int = 0) -> None:
        self.clock = 0.0
        self.seed = seed
        self.trace = TraceEnvelope("qexp-simulated-runtime/v1", seed)
        self._records: dict[str, RevisionedValue] = {}
        self._locks: set[str] = set()
        self._faults: dict[ProtocolPoint, list[BaseException]] = {}

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("virtual clock cannot move backwards")
        self.clock += seconds
        self.trace.record("clock.advance", {"seconds": seconds}, self.clock)

    def inject_fault(self, point: ProtocolPoint, error: BaseException | None = None) -> None:
        self._faults.setdefault(point, []).append(error or InjectedProtocolError(point))

    def yield_at(self, point: ProtocolPoint, *, participant: str | None = None) -> None:
        self.trace.record("protocol.yield", {"point": point}, self.clock, participant)
        faults = self._faults.get(point)
        if faults:
            raise faults.pop(0)

    def read(self, key: str) -> RevisionedValue | None:
        self.yield_at(ProtocolPoint.SNAPSHOT_READ)
        return self._records.get(key)

    def cas(self, key: str, revision: int | None, value: Any) -> RevisionedValue | None:
        self.yield_at(ProtocolPoint.CAS)
        current = self._records.get(key)
        if (current.revision if current else None) != revision:
            self.trace.record("store.conflict", {"key": key, "revision": revision}, self.clock)
            return None
        next_value = RevisionedValue(1 if current is None else current.revision + 1, value)
        self._records[key] = next_value
        self.trace.record("store.commit", {"key": key, "revision": next_value.revision}, self.clock)
        return next_value

    def acquire(self, name: str) -> bool:
        self.yield_at(ProtocolPoint.LOCK_ACQUIRE)
        if name in self._locks:
            return False
        self._locks.add(name)
        return True

    def release(self, name: str) -> None:
        self._locks.discard(name)


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

    def request(self, command: str, *, timeout_seconds: float = 2.0) -> dict[str, Any]:
        """Send one control command and wait for its structured response."""
        if not command:
            raise ValueError("participant command must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("participant request timeout must be positive")
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError(f"participant {self.name} has no IPC pipes")
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
        return json.loads(response)

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            self.request("exit")
            self.process.wait(timeout=2.0)
        except (RuntimeError, subprocess.TimeoutExpired):
            self.process.terminate()
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
    else:
        print(json.dumps({"error": "unknown command"}), flush=True)
"""

    def __init__(self, root: Path, nodeid: str) -> None:
        self.root = root
        self.nodeid = nodeid
        self.shared_root = root / "shared"
        self.shared_root.mkdir(parents=True, exist_ok=True)
        self.participants: list[MachineParticipant] = []

    def start(self, name: str) -> MachineParticipant:
        if any(item.name == name for item in self.participants):
            raise ValueError(f"duplicate machine participant: {name}")
        scope = TestResourceScope.create(self.root / "participants", f"{self.nodeid}-{name}")
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
        participant = MachineParticipant(name, scope, process)
        self.participants.append(participant)
        return participant

    def close(self) -> None:
        for participant in reversed(self.participants):
            participant.close()
