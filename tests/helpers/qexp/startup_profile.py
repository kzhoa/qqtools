"""Opt-in launch phase observer for the isolated authority workload.

Run this file directly to timestamp Python entry before importing qqtools. Records
are diagnostic only, flushed without fsync; production launch deadlines stay intact.
"""

from __future__ import annotations

import time

PYTHON_ENTERED = time.time()

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path


def collect_startup_profiles(
    root: Path, *, expected_source: Path, expected_home: Path, expected_count: int
) -> dict[str, object]:
    """Retain invalid/incomplete diagnostics without replacing a workload failure."""
    profiles = {}
    errors = []
    status = "valid"
    try:
        paths = sorted(root.iterdir())
    except FileNotFoundError:
        paths = []
    except OSError as error:
        paths = []
        errors.append(f"profile inventory: {type(error).__name__}")
        status = "invalid"
    for path in paths:
        try:
            profiles[path.name] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            errors.append(f"{path.name}: {type(error).__name__}")
            status = "invalid"
    records = [name for name in profiles if name.endswith(".json")]
    if len(records) != expected_count:
        errors.append(f"expected {expected_count} runner profiles, found {len(records)}")
        if status == "valid":
            status = "incomplete"
    for name in records:
        try:
            events = json.loads(profiles[name])
            if not isinstance(events, list) or not events or not all(isinstance(event, dict) for event in events):
                raise ValueError("invalid event list")
            if events[0].get("home") != str(expected_home) or not any(
                event.get("source_file") == str(expected_source) for event in events
            ):
                raise ValueError("runner escaped the profiled checkout or isolated HOME")
        except ValueError as error:
            errors.append(f"{name}: {error}")
            status = "invalid"
    return {
        "startup_profiles": profiles,
        "startup_profile_status": status,
        "startup_profile_errors": errors,
        "startup_environment_is_valid": status == "valid",
    }


def main() -> int:
    output = Path(sys.argv[1])
    events = [
        {
            "stage": "python_entered",
            "at": PYTHON_ENTERED,
            "home": os.environ.get("HOME"),
            "pythonpath": os.environ.get("PYTHONPATH"),
        }
    ]

    def event(stage):
        events.append({"stage": stage, "at": time.time()})

    try:
        event("import_started")
        from qqtools.plugins.qexp import runner

        event("import_finished")
        events[-1]["source_file"] = str(Path(runner.__file__).resolve())
        original_locks = runner.authority_locks

        @contextmanager
        def locks(*args, **kwargs):
            event("authority_lock_requested")
            with original_locks(*args, **kwargs):
                event("authority_lock_acquired")
                yield
            event("authority_lock_released")

        runner.authority_locks = locks
        for name in (
            "load_root_config",
            "load_task",
            "_load_attempt",
            "_publish_launch_intent",
            "_publish_registration",
        ):
            original = getattr(runner, name)

            def measured(*args, _original=original, _name=name, **kwargs):
                event(f"{_name}:started")
                try:
                    return _original(*args, **kwargs)
                finally:
                    event(f"{_name}:finished")

            setattr(runner, name, measured)
        return runner.main(sys.argv[2:])
    finally:
        output.write_text(json.dumps(events), encoding="utf-8")


def install_launch_observer(output: Path) -> list[dict[str, object]]:
    """Wrap only the test agent's tmux commands, preserving runner arguments."""
    import hashlib
    import shlex

    from qqtools.plugins.qexp.executor import Executor

    output.mkdir(parents=True, exist_ok=True)
    calls = []
    original = Executor.build_runner_command
    original_initiate = Executor.initiate_attempt

    def initiate(self, cfg, task_id, attempt, *args, **kwargs):
        create, send = self.create_window, self.send_command

        def measure(function, stage):
            def measured(*args, **kwargs):
                started = time.time()
                try:
                    return function(*args, **kwargs)
                finally:
                    calls.append({"stage": stage, "task_id": task_id, "started": started, "finished": time.time()})

            return measured

        self.create_window = measure(create, "tmux_create_window")
        self.send_command = measure(send, "tmux_send_command")
        try:
            return original_initiate(self, cfg, task_id, attempt, *args, **kwargs)
        finally:
            self.create_window, self.send_command = create, send

    def command(self, cfg, task_id, attempt_id, fencing_token, launch_id):
        argv = shlex.split(original(self, cfg, task_id, attempt_id, fencing_token, launch_id))
        identity = hashlib.sha256(str(cfg.shared_root).encode()).hexdigest()[:16] + "-" + launch_id
        path = output / f"{identity}.json"
        (output / f"{identity}.requested").write_text(str(time.time()), encoding="utf-8")
        # EPOCHREALTIME is supplied by the Linux test host's bash. An empty value
        # is retained as unavailable instead of fabricating a shell timestamp.
        shell = f'printf "%s\\n" "$EPOCHREALTIME" > {shlex.quote(str(path.with_suffix(".shell")))}'
        return shell + "; " + shlex.join([argv[0], str(Path(__file__).resolve()), str(path), *argv[3:]])

    Executor.build_runner_command = command
    Executor.initiate_attempt = initiate
    return calls


if __name__ == "__main__":
    raise SystemExit(main())
