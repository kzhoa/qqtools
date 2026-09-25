"""Background entrypoint for the qexp machine agent."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from ..runtime.records import utc_now
from .config import load_agent_config
from .context import MachineRuntime
from .diagnostics import parse_log_size, prepare_agent_diagnostics, update_diagnostic_evidence
from .lifecycle import MachineAgentStartError, run_machine_agent_loop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="qexp machine agent process")
    parser.add_argument("--machine-runtime-root", required=True)
    parser.add_argument("--loop-interval", type=float, default=5.0)
    parser.add_argument("--available-gpus", default=None)
    parser.add_argument("--instance-id")
    parser.add_argument("--log-path")
    parser.add_argument("--log-max-bytes", type=parse_log_size)
    parser.add_argument("--startup-sequence", type=int)
    parser.add_argument(
        "--capture-mode",
        choices=("detached", "foreground_managed_only"),
        default="detached",
    )
    parser.add_argument("--initial-capture-health", choices=("healthy", "degraded", "unavailable"))
    parser.add_argument("--initial-capture-error")
    parser.add_argument("--initial-reconciliation-degraded", action="store_true")
    return parser


def _startup_error_details(log_path: str | Path | None, startup_log) -> str:
    details: list[str] = []
    if startup_log is not None:
        try:
            startup_log.seek(0)
            lines = startup_log.read().strip().splitlines()
        except (OSError, ValueError):
            lines = []
        if lines:
            details.append(lines[-1])
    if log_path:
        try:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(log_path, flags)
            try:
                size = os.fstat(fd).st_size
                os.lseek(fd, max(0, size - 8192), os.SEEK_SET)
                lines = os.read(fd, 8192).decode("utf-8", errors="replace").strip().splitlines()
            finally:
                os.close(fd)
        except OSError:
            lines = []
        if lines and lines[-1] not in details:
            details.append(lines[-1])
        details.append(f"diagnostic log: {log_path}")
    return f": {'; '.join(details)}" if details else ""


def spawn_machine_agent_process(
    runtime: MachineRuntime | str | Path | None = None,
    *,
    available_gpus: list[int] | None = None,
    loop_interval: float | None = None,
    stdin=None,
    stdout=None,
    stderr=None,
) -> subprocess.Popen:
    if loop_interval is not None and loop_interval <= 0:
        raise ValueError("loop_interval must be positive.")
    machine_runtime = runtime if isinstance(runtime, MachineRuntime) else MachineRuntime(runtime)
    machine_runtime.require_initialized()
    machine_runtime.ensure_layout(create_identity=False)
    agent_config = load_agent_config(machine_runtime)
    instance_id = uuid.uuid4().hex
    capture_mode = "foreground_managed_only" if stdout is not None or stderr is not None else "detached"
    prepared = prepare_agent_diagnostics(
        machine_runtime,
        instance_id=instance_id,
        capture_mode=capture_mode,
        log_max_bytes=agent_config.log_max_bytes,
    )
    startup_sequence = prepared.startup_sequence
    log_path = prepared.log_path
    log_handle = prepared.handle
    startup_log = tempfile.TemporaryFile(mode="w+", encoding="utf-8") if stderr is None and log_handle is None else None
    launcher_revision = 0
    signal_attempts: list[dict[str, object]] = []

    def publish_launcher_evidence(**evidence: object) -> None:
        nonlocal launcher_revision
        launcher_revision += 1
        try:
            update_diagnostic_evidence(
                machine_runtime,
                instance_id=instance_id,
                startup_sequence=startup_sequence,
                writer="launcher",
                revision=launcher_revision,
                evidence=evidence,
            )
        except Exception:
            # Diagnostics must not prevent process activation.
            return

    if prepared.error is not None:
        message = (
            "persistent machine-agent log capture is unavailable"
            if log_handle is None
            else "machine-agent diagnostics preparation is degraded"
        )
        print(
            f"Warning: {message}; startup will continue with degraded capture.",
            file=stderr if stderr is not None else sys.stderr,
            flush=True,
        )

    command = [
        sys.executable,
        "-u",
        "-X",
        "faulthandler",
        "-m",
        "qqtools.plugins.qexp.agent.process",
        "--machine-runtime-root",
        str(machine_runtime.root),
        "--instance-id",
        instance_id,
        "--log-path",
        str(log_path) if log_path is not None else "",
        "--log-max-bytes",
        str(agent_config.log_max_bytes),
        "--startup-sequence",
        str(startup_sequence),
        "--capture-mode",
        capture_mode,
        "--initial-capture-health",
        prepared.capture_health,
    ]
    if prepared.error is not None:
        command.extend(("--initial-capture-error", prepared.error))
    if prepared.reconciliation_degraded:
        command.append("--initial-reconciliation-degraded")
    if available_gpus is not None:
        command.extend(("--available-gpus", ",".join(str(item) for item in available_gpus)))
    if loop_interval is not None:
        command.extend(("--loop-interval", str(loop_interval)))
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[4])
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root if not existing_pythonpath else f"{source_root}{os.pathsep}{existing_pythonpath}"
    )
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONFAULTHANDLER"] = "1"
    try:
        try:
            process = subprocess.Popen(
                command,
                env=environment,
                stdin=subprocess.DEVNULL if stdin is None else stdin,
                stdout=(stdout if stdout is not None else log_handle if log_handle is not None else subprocess.DEVNULL),
                stderr=(stderr if stderr is not None else log_handle if log_handle is not None else startup_log),
                start_new_session=True,
            )
        except OSError as exc:
            publish_launcher_evidence(
                startup_outcome="spawn_failed",
                observed_at=utc_now(),
            )
            details = _startup_error_details(log_path, startup_log)
            if startup_log is not None:
                startup_log.close()
            raise MachineAgentStartError(f"machine agent process could not be started: {exc}{details}") from exc
        publish_launcher_evidence(startup_outcome="started", observed_at=utc_now())
    finally:
        prepared.close()

    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            status_path = machine_runtime.paths["agent"] / "status.json"
            if status_path.exists():
                try:
                    status = json.loads(status_path.read_text(encoding="utf-8")).get("machine_agent", {})
                except (OSError, ValueError, TypeError, AttributeError):
                    status = {}
                if (
                    process.poll() is None
                    and status.get("state") == "active"
                    and status.get("pid") == process.pid
                    and status.get("instance_id") == instance_id
                ):
                    publish_launcher_evidence(
                        startup_outcome="started",
                        observed_at=utc_now(),
                    )
                    return process
            exit_code = process.poll()
            if exit_code is not None:
                publish_launcher_evidence(
                    startup_outcome="exited",
                    observed_at=utc_now(),
                    wait_status=exit_code,
                )
                raise MachineAgentStartError(
                    f"machine agent exited during startup with exit code {exit_code}"
                    f"{_startup_error_details(log_path, startup_log)}."
                )
            time.sleep(0.02)

        timeout_at = utc_now()
        publish_launcher_evidence(
            startup_outcome="timed_out",
            timeout_trigger="process_handshake",
            observed_at=timeout_at,
            timeout_triggered_at=timeout_at,
        )
        for signum in (signal.SIGTERM, signal.SIGKILL):
            attempt: dict[str, object] = {
                "signal": int(signum),
                "attempted_at": utc_now(),
                "delivered": False,
            }
            try:
                process.send_signal(signum)
            except OSError as exc:
                attempt["error_type"] = type(exc).__name__
            else:
                attempt["delivered"] = True
            signal_attempts.append(attempt)
            publish_launcher_evidence(
                startup_outcome="timed_out",
                timeout_trigger="process_handshake",
                timeout_triggered_at=timeout_at,
                signal_attempts=list(signal_attempts),
                observed_at=attempt["attempted_at"],
            )
            try:
                wait_status = process.wait(timeout=1.0 if signum == signal.SIGTERM else None)
            except subprocess.TimeoutExpired:
                continue
            publish_launcher_evidence(
                startup_outcome="timed_out",
                timeout_trigger="process_handshake",
                timeout_triggered_at=timeout_at,
                signal_attempts=list(signal_attempts),
                observed_at=utc_now(),
                wait_status=wait_status,
            )
            break
        raise MachineAgentStartError(
            f"machine agent did not acquire scheduler authority within 5 seconds"
            f"{_startup_error_details(log_path, startup_log)}."
        )
    finally:
        if startup_log is not None:
            startup_log.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    available_gpus = None
    if args.available_gpus is not None:
        try:
            available_gpus = [int(item) for item in args.available_gpus.split(",") if item.strip()]
        except ValueError as exc:
            raise RuntimeError("--available-gpus must be a comma-separated list of integers.") from exc
    run_machine_agent_loop(
        args.machine_runtime_root,
        loop_interval=args.loop_interval,
        available_gpus=available_gpus,
        instance_id=args.instance_id,
        log_path=args.log_path or None,
        effective_log_max_bytes=args.log_max_bytes,
        startup_sequence=args.startup_sequence,
        capture_mode=args.capture_mode,
        initial_capture_health=args.initial_capture_health,
        initial_capture_error=args.initial_capture_error,
        initial_reconciliation_degraded=args.initial_reconciliation_degraded,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
