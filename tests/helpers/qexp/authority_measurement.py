"""Instrument the real machine-agent entry point for isolated workload comparisons.

Run as a module with runtime root, output JSON path, and visible GPU count.
This changes observation only; it invokes the production control plane and runner.
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

IMPORT_STARTED = time.time()

from qqtools.plugins.qexp.agent.lifecycle import run_machine_agent_loop
from qqtools.plugins.qexp.runtime import store

IMPORT_FINISHED = time.time()


def pytest_addoption(parser):
    group = parser.getgroup("authority workload")
    group.addoption("--authority-workload-profile", default="{}", help="JSON workload dimensions")
    group.addoption("--authority-workload-output", default=None, help="Raw workload report path")
    group.addoption("--authority-profile-startup", action="store_true", help="Observe real runner launch phases")


class AgentMeasurements:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.operations = defaultdict(lambda: defaultdict(lambda: {"calls": 0, "errors": 0, "total_ns": 0}))
        self.events = []
        self.seen = set()
        self.samples = deque(maxlen=4096)
        self.samples_dropped = 0
        self.write_context = threading.local()
        self.writes = deque(maxlen=16384)
        self.writes_dropped = 0
        self.stop_sampling = threading.Event()
        self.sampler = threading.Thread(target=self._sample_loop, name="authority-measurement", daemon=True)

    def capture(self):
        """Copy cumulative counters under the same lock used by observations."""
        with self.lock:
            if len(self.samples) == self.samples.maxlen:
                self.samples_dropped += 1
            self.samples.append(
                {
                    "monotonic": time.monotonic(),
                    "operations": {
                        thread: {name: dict(value) for name, value in counters.items()}
                        for thread, counters in self.operations.items()
                    },
                }
            )

    def _sample_loop(self):
        while not self.stop_sampling.wait(0.25):
            self.capture()

    def start(self):
        self.capture()
        self.sampler.start()

    def stop(self):
        self.stop_sampling.set()
        self.sampler.join()
        self.capture()

    def observe(self, name, elapsed_ns, *, has_failed=False):
        with self.lock:
            value = self.operations[threading.current_thread().name][name]
            value["calls"] += 1
            value["errors"] += int(has_failed)
            value["total_ns"] += elapsed_ns

    def event(self, name, value):
        if not isinstance(value, dict):
            return
        for key in (
            "process_registration",
            "process",
            "exit_observation",
            "attempt",
            "reservation",
            "task",
            "registration",
        ):
            record = value.get(key)
            if not isinstance(record, dict):
                continue
            if key == "task":
                claim = record.get("claim_control", {}).get("active_claim") or {}
                record = dict(
                    record, phase=record.get("state", {}).get("projection"), launch_state=claim.get("launch_state")
                )
            identifier = record.get("attempt_id") or (record.get("task_id") if key == "task" else None)
            if key == "registration":
                identifier = record.get("project_id")
            if not identifier:
                continue
            if key == "reservation" and record.get("state") != "released":
                continue
            if name == "read_json" and key not in {"process_registration", "exit_observation"}:
                continue
            if name not in {"read_json", "atomic_replace", "create_if_absent"}:
                continue
            phase = record.get("phase") or record.get("state")
            lease = record.get("lease") or {}
            expiry = lease.get("expires_at") or record.get("eligibility_expires_at")
            token = record.get("current_fencing_token", record.get("fencing_token"))
            generation = record.get("generation")
            identity = (name, key, identifier, phase, record.get("launch_state"), expiry, token, generation)
            with self.lock:
                if identity in self.seen:
                    continue
                self.seen.add(identity)
                self.events.append(
                    {
                        "operation": name,
                        "record": key,
                        "attempt_id": record.get("attempt_id"),
                        "task_id": record.get("task_id"),
                        "project_id": record.get("project_id"),
                        "generation": generation,
                        "fencing_token": token,
                        "expires_at": expiry,
                        "renewed_at": lease.get("renewed_at"),
                        "phase": phase,
                        "authority_mode": record.get("authority_mode"),
                        "clock_error_bound_seconds": record.get("clock_error_bound_seconds"),
                        "launch_state": record.get("launch_state"),
                        "source_at": record.get("process_created_at") or record.get("observed_at"),
                        "at": datetime.now(timezone.utc).isoformat(),
                        "monotonic": time.monotonic(),
                        "thread": threading.current_thread().name,
                    }
                )

    def wrap(self, original, name, *, is_store=False):
        @wraps(original)
        def measured(*args, **kwargs):
            started = time.monotonic_ns()
            has_failed = True
            previous_write = getattr(self.write_context, "current", None)
            write = None
            if is_store and name in {"atomic_replace", "create_if_absent"}:
                write = {
                    "operation": name,
                    "path": str(args[0]),
                    "thread": threading.current_thread().name,
                    "started_monotonic": time.monotonic(),
                    "fsync_calls": 0,
                    "fsync_ns": 0,
                }
                self.write_context.current = write
            try:
                result = original(*args, **kwargs)
                has_failed = False
                if is_store:
                    value = result if name == "read_json" else (args[1] if len(args) > 1 else None)
                    if name != "create_if_absent" or result is not False:
                        self.event(name, value)
                        if name in {"atomic_replace", "create_if_absent"} and isinstance(value, dict):
                            for record in ("attempt", "registration", "process"):
                                if isinstance(value.get(record), dict):
                                    self.observe(f"record.{record}.publications", 0)
                return result
            finally:
                elapsed = time.monotonic_ns() - started
                if name == "os.fsync" and previous_write is not None:
                    previous_write["fsync_calls"] += 1
                    previous_write["fsync_ns"] += elapsed
                if write is not None:
                    self.write_context.current = previous_write
                    write.update(finished_monotonic=time.monotonic(), total_ns=elapsed, has_failed=has_failed)
                    with self.lock:
                        if len(self.writes) == self.writes.maxlen:
                            self.writes_dropped += 1
                        self.writes.append(write)
                self.observe(f"store.{name}" if is_store else name, elapsed, has_failed=has_failed)

        return measured

    def install(self):
        replacements = {}
        for name in ("read_json", "atomic_replace", "create_if_absent", "iter_json"):
            original = getattr(store, name)
            replacements[id(original)] = self.wrap(original, name, is_store=True)
        # Cover aliases imported before instrumentation as well as future imports.
        for module in tuple(sys.modules.values()):
            if not getattr(module, "__name__", "").startswith("qqtools.plugins.qexp"):
                continue
            for name, value in tuple(vars(module).items()):
                replacement = replacements.get(id(value))
                if replacement is not None:
                    setattr(module, name, replacement)
        for name in ("stat", "lstat", "fsync"):
            setattr(os, name, self.wrap(getattr(os, name), f"os.{name}"))
        fcntl.flock = self.wrap(fcntl.flock, "fcntl.flock")
        original_scandir = os.scandir
        measurements = self
        original_open = os.open

        def open_file(path, *args, **kwargs):
            if not isinstance(path, int) and Path(path).name.startswith("settled-history-"):
                measurements.observe("history.file_open", 0)
            return original_open(path, *args, **kwargs)

        os.open = open_file
        original_path_open = Path.open

        def path_open(path, *args, **kwargs):
            if path.name.startswith("settled-history-"):
                measurements.observe("history.file_open", 0)
            return original_path_open(path, *args, **kwargs)

        Path.open = path_open

        class Inventory:
            def __init__(self, entries):
                self.entries = entries

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

            def __iter__(self):
                return self

            def __next__(self):
                started = time.monotonic_ns()
                entry = next(self.entries)
                measurements.observe("os.scandir.entries", time.monotonic_ns() - started)
                return entry

            def close(self):
                self.entries.close()

        def scandir(*args, **kwargs):
            started = time.monotonic_ns()
            try:
                return Inventory(original_scandir(*args, **kwargs))
            finally:
                measurements.observe("os.scandir", time.monotonic_ns() - started)

        os.scandir = scandir


def main() -> None:
    runtime, output, gpu_count = sys.argv[1:]
    launch_calls = []
    if profile_root := os.environ.get("QEXP_TEST_STARTUP_PROFILE_ROOT"):
        from tests.helpers.qexp.startup_profile import install_launch_observer

        launch_calls = install_launch_observer(Path(profile_root))
    measurements = AgentMeasurements()
    measurements.install()
    measurements.start()
    try:
        run_machine_agent_loop(Path(runtime), available_gpus=list(range(int(gpu_count))), loop_interval=0.1)
    finally:
        measurements.stop()
        with measurements.lock:
            value = {
                "operations": dict(measurements.operations),
                "events": list(measurements.events),
                "operation_samples": list(measurements.samples),
                "operation_samples_dropped": measurements.samples_dropped,
                "operation_sample_interval_seconds": 0.25,
                "write_observations": list(measurements.writes),
                "write_observations_dropped": measurements.writes_dropped,
                "startup_import_seconds": IMPORT_FINISHED - IMPORT_STARTED,
                "launch_calls": launch_calls,
            }
        Path(output).write_text(json.dumps(value, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
