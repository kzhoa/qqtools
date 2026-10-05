"""Real-process regression coverage for agent lifecycle independence (LI-01..LI-09)."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.lifecycle import (
    get_machine_agent_status,
    restart_machine_agent,
    start_machine_agent,
    stop_machine_agent,
)
from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor
from qqtools.plugins.qexp.agent.setup import initialize_machine, register_projects
from qqtools.plugins.qexp.config_types import RootConfig
from qqtools.plugins.qexp.layout import machine_state_path
from qqtools.plugins.qexp.runtime.paths import attempt_path, local_paths, machine_runtime_paths, shared_paths
from qqtools.plugins.qexp.runtime.project_activation import publish_project_activation
from qqtools.plugins.qexp.runtime.resources.cpu_lane import set_cpu_lane_capacity
from qqtools.plugins.qexp.runtime.resources.reservations import active_reservations
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task
from qqtools.plugins.qexp.scheduler import expire_claim
from qqtools.plugins.qexp.tmux import is_libtmux_available
from tests.fixtures.qexp_legacy_ownership import persist_legacy_orphan
from tests.helpers.qexp.lifecycle import LifecycleBranch, LifecycleLab, wait_all
from tests.helpers.qexp.worker_diagnostics import describe_project_io_workers

pytestmark = [pytest.mark.integration, pytest.mark.machine_lab]

CONVERGENCE_BUDGET_SECONDS = 15.0
PROCESS_START_BUDGET_SECONDS = 15.0
POLL_INTERVAL_SECONDS = 0.01


def _initialized_runtime(root: Path, *, agent_mode: str = "on_demand") -> MachineRuntime:
    runtime = MachineRuntime(root)
    initialize_machine(runtime, "gpu-1", agent_mode=agent_mode)
    return runtime


def _register_project(runtime: MachineRuntime, cfg: RootConfig):
    register_projects(runtime, [cfg.shared_root], machine_name=cfg.machine_name)
    return runtime.ensure_binding(cfg.shared_root, cfg.machine_name)[0]


@pytest.fixture(autouse=True)
def _isolated_visible_gpu(monkeypatch: pytest.MonkeyPatch, checkout_subprocess_env) -> None:
    """Give the real child agent one deterministic test-owned GPU."""
    monkeypatch.setenv("QEXP_VISIBLE_GPUS", "0")
    monkeypatch.setenv("PYTHONPATH", checkout_subprocess_env["PYTHONPATH"])


@pytest.fixture(autouse=True)
def _require_real_process_prerequisites() -> None:
    """Fail the lifecycle gate explicitly when its Linux/tmux boundary is unavailable."""
    if sys.platform != "linux":
        pytest.fail("LI-01..LI-08 unverified: qexp lifecycle requires a supported Linux host")
    if shutil.which("tmux") is None or not is_libtmux_available():
        pytest.fail("LI-01..LI-08 unverified: tmux and libtmux are required by the real-process gate")


@pytest.fixture(autouse=True)
def _cleanup_owned_tmux(qexp_resource_scope):
    """Close only servers inside this test's recorded tmux namespace."""
    from tests.helpers.qexp.resources import _has_active_unix_socket

    yield
    for socket_path in qexp_resource_scope.tmux_root.rglob("*"):
        if socket_path.is_socket():
            subprocess.run(
                ["tmux", "-S", str(socket_path), "kill-server"],
                check=False,
                capture_output=True,
                timeout=5,
            )
    _wait_for(lambda: not _has_active_unix_socket(qexp_resource_scope.tmux_root), timeout=5)


def _wait_for(
    predicate,
    *,
    timeout: float = CONVERGENCE_BUDGET_SECONDS,
    description: str = "lifecycle condition",
    on_timeout=None,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(POLL_INTERVAL_SECONDS)
    diagnostics = on_timeout() if on_timeout is not None else None
    suffix = f"; diagnostics={diagnostics}" if diagnostics is not None else ""
    raise AssertionError(f"{description} did not converge within {timeout:.1f}s{suffix}")


def _li03_timeout_diagnostics(cfg, runtime: MachineRuntime, task_id: str, process, *, observed: Path) -> dict:
    """Capture the state needed to distinguish lease recovery from agent startup delay."""
    try:
        task = load_task(cfg, task_id)
        task_state = {
            "projection": task.state.get("projection"),
            "reason": task.state.get("reason"),
            "attempt": task.attempt_control,
            "claim": task.claim_control.get("active_claim"),
        }
    except Exception as exc:  # diagnostics must not hide the original timeout
        task_state = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        agent = get_machine_agent_status(runtime)
    except Exception as exc:
        agent = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        reservations = active_reservations(runtime.root)
    except Exception as exc:
        reservations = {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "task": task_state,
        "agent": agent,
        "reservations": reservations,
        "process": {"pid": process.pid, "returncode": process.poll()},
        "peer_observed": observed.exists(),
    }


def _create_case(
    tmp_path: Path,
    *,
    exit_code: int = 0,
    seconds: float = 1.0,
    should_wait: bool = False,
):
    project_root = tmp_path / "project"
    shared_root = project_root / ".qexp"
    cfg = init_shared_root(shared_root, "gpu-1", agent_mode="daemon", runtime_root=tmp_path / "legacy")
    runtime = _initialized_runtime(tmp_path / "machine-runtime", agent_mode="daemon")
    _register_project(runtime, cfg)
    marker = tmp_path / "launch-count"
    command = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path\nimport time\n"
            f"p=Path({str(marker)!r})\n"
            "p.write_text(str(int(p.read_text()) + 1) if p.exists() else '1')\n"
            "deadline=time.monotonic()+60\n"
            f"while {should_wait!r} and not p.with_suffix('.finish').exists():\n"
            "    if time.monotonic()>deadline: raise SystemExit(99)\n"
            "    p.with_suffix('.progress').write_text(str(time.monotonic()))\n"
            "    time.sleep(0.01)\n"
            f"time.sleep({0 if should_wait else seconds!r})\nraise SystemExit({exit_code})"
        ),
    ]
    task = submit(cfg, command, working_dir=project_root)
    process = start_machine_agent(runtime, available_gpus=[0], loop_interval=0.1)
    return cfg, runtime, task, marker, process


def _start_case(
    tmp_path: Path,
    *,
    exit_code: int = 0,
    seconds: float = 1.0,
    should_wait: bool = False,
):
    return _create_case(tmp_path, exit_code=exit_code, seconds=seconds, should_wait=should_wait)


def _wait_running(cfg, task_id: str, marker: Path | None = None) -> None:
    def is_running_and_observed() -> bool:
        if load_task(cfg, task_id).state["projection"] != "running":
            return False
        return marker is None or marker.exists()

    def startup_diagnostics():
        # Resource teardown removes these records, so retain failure-boundary truth.
        try:
            task = load_task(cfg, task_id)
            number = task.attempt_control.get("current_attempt_number")
            path = attempt_path(cfg.shared_root, task_id, number) if number is not None else None
            panes = subprocess.run(
                ["tmux", "list-panes", "-a", "-F", "#{pane_id} #{window_name} #{pane_current_command}"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            pane_output = {}
            for line in panes.stdout.splitlines()[:8]:
                pane_id, _, _ = line.partition(" ")
                captured = subprocess.run(
                    ["tmux", "capture-pane", "-p", "-t", pane_id, "-S", "-20"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                )
                pane_output[line] = captured.stdout[-4000:]
            return {
                "tmux_panes": pane_output,
                "task_state": task.state,
                "claim": task.claim_control,
                "attempt_control": task.attempt_control,
                "attempt": read_json(path) if path is not None and path.exists() else None,
                "launch_marker": marker.read_text() if marker is not None and marker.exists() else None,
            }
        except Exception as exc:  # retain the original timeout if diagnostics fail
            return {"error": f"{type(exc).__name__}: {exc}"}

    _wait_for(
        is_running_and_observed,
        timeout=PROCESS_START_BUDGET_SECONDS,
        description=f"Task {task_id} running and observed",
        on_timeout=startup_diagnostics,
    )


def _wait_terminal(cfg, task_id: str) -> object:
    _wait_for(
        lambda: load_task(cfg, task_id).state["projection"] in {"succeeded", "failed", "cancelled"},
    )
    return load_task(cfg, task_id)


def test_real_agent_two_hung_worker_primary_and_renewal_qualification(tmp_path: Path) -> None:
    """Qualify the published two-hung-worker/default-loop latency baseline."""
    runtime = _initialized_runtime(tmp_path / "machine-runtime", agent_mode="daemon")
    healthy_cfg = init_shared_root(
        tmp_path / "healthy" / ".qexp",
        "gpu-1",
        agent_mode="daemon",
        runtime_root=tmp_path / "healthy-legacy",
    )
    healthy = _register_project(runtime, healthy_cfg)
    blocked: list[tuple[object, object, Path, Path]] = []
    for index in range(2):
        cfg = init_shared_root(
            tmp_path / f"blocked-{index}" / ".qexp",
            "gpu-1",
            agent_mode="daemon",
            runtime_root=tmp_path / f"blocked-{index}-legacy",
        )
        binding = _register_project(runtime, cfg)
        schema = shared_paths(cfg.shared_root)["schema"] / "version.json"
        blocked.append((cfg, binding, schema, schema.with_suffix(".qualification-held")))
    set_cpu_lane_capacity(runtime.root, capacity=1)

    running_marker = tmp_path / "running"
    running_finish = tmp_path / "running.finish"
    running = submit(
        healthy_cfg,
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path\nimport time\n"
                f"marker=Path({str(running_marker)!r}); finish=Path({str(running_finish)!r})\n"
                "marker.touch()\n"
                "deadline=time.monotonic()+90\n"
                "while not finish.exists():\n"
                "    if time.monotonic()>deadline: raise SystemExit(99)\n"
                "    time.sleep(0.02)\n"
            ),
        ],
        working_dir=healthy_cfg.project_root,
    )
    qualifying_agent: subprocess.Popen | None = None
    primary_finish = tmp_path / "primary.finish"
    foreground_hold = tmp_path / "foreground-hold"
    background_arm = tmp_path / "background-arm"
    background_started = tmp_path / "background-started"
    measurement_gate = tmp_path / "measurement-gate"
    trace = tmp_path / "worker-trace.jsonl"
    try:
        agent_script = r'''
import json, subprocess, sys, time
from pathlib import Path
from qqtools.plugins.qexp.agent.lifecycle import run_machine_agent_loop
from qqtools.plugins.qexp.agent.project_io_admission import ProjectIOAdmission
from qqtools.plugins.qexp.agent import project_io_executor
from qqtools.plugins.qexp.runtime.paths import shared_paths

root, healthy, foreground_hold, background_arm, background_started, measurement_gate, trace = sys.argv[1:]
foreground_hold = Path(foreground_hold)
background_arm = Path(background_arm)
background_started = Path(background_started)
measurement_gate = Path(measurement_gate)
trace = Path(trace)

real_offer = ProjectIOAdmission.offer
def offer(self, intent, action):
    if (
        foreground_hold.exists()
        and intent.owner[1] == healthy
        and intent.service_class in {"authority", "primary"}
        and not measurement_gate.exists()
    ):
        return None
    return real_offer(self, intent, action)
ProjectIOAdmission.offer = offer

worker_wrapper = r"""
import json, sys, time
from pathlib import Path
trace, project_id, operation_kind, began, started, gate, hold, *args = sys.argv[1:]
try:
    if hold == "1":
        Path(started).touch()
        deadline = time.monotonic() + 5
        while not Path(gate).exists():
            if time.monotonic() >= deadline:
                raise SystemExit(98)
            time.sleep(0.005)
    from qqtools.plugins.qexp.agent.project_io_worker import main
    code = main(args)
finally:
    with Path(trace).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "project_id": project_id,
            "operation_kind": operation_kind,
            "started_at": float(began),
            "qualification_background": hold == "1",
            "elapsed_seconds": time.monotonic() - float(began),
        }, sort_keys=True) + "\n")
raise SystemExit(code)
"""
background_operations = {
    "upgrade_service", "submission_control_service", "observation_service",
    "machine_snapshot_publish", "activation_observe",
    "activation_consumer_register", "activation_consumer_ack",
}
blocked_worker_wrapper = r"""
import sys
from pathlib import Path
schema, *args = sys.argv[1:]
# Every selected blocked-binding invocation crosses the injected I/O boundary,
# even when its particular operation would otherwise use cached schema state.
Path(schema).read_bytes()
from qqtools.plugins.qexp.agent.project_io_worker import main
raise SystemExit(main(args))
"""
real_popen = project_io_executor.subprocess.Popen
launching = [None]
background_selected = [False]
def popen(command, *args, **kwargs):
    item = launching[0]
    if item is not None and item[0].project_id == healthy:
        request, began = item
        hold = (
            request is not None
            and request.operation_kind in background_operations
            and background_arm.exists()
            and self_runtime.authority_ready_generations.get(healthy) == request.registration_generation
            and not background_selected[0]
        )
        if hold:
            background_selected[0] = True
        command = [
            command[0], "-c", worker_wrapper, trace, request.project_id, request.operation_kind,
            str(began), str(background_started), str(measurement_gate), "1" if hold else "0", *command[3:]
        ]
    elif item is not None:
        request, _began = item
        schema = shared_paths(Path(request.canonical_shared_root))["schema"] / "version.json"
        command = [command[0], "-c", blocked_worker_wrapper, str(schema), *command[3:]]
    return real_popen(command, *args, **kwargs)
project_io_executor.subprocess.Popen = popen

real_start = project_io_executor.ProjectIOExecutor.start
self_runtime = None
def start(self, request_id, **kwargs):
    global self_runtime
    self_runtime = self.runtime
    request = self._load_request(request_id)
    began = time.monotonic()
    launching[0] = (request, began)
    try:
        process = real_start(self, request_id, **kwargs)
    finally:
        launching[0] = None
    return process
project_io_executor.ProjectIOExecutor.start = start
run_machine_agent_loop(root, available_gpus=[0])
'''
        qualifying_agent = subprocess.Popen(
            [
                sys.executable,
                "-c",
                agent_script,
                str(runtime.root),
                healthy.project_id,
                str(foreground_hold),
                str(background_arm),
                str(background_started),
                str(measurement_gate),
                str(trace),
            ],
            start_new_session=True,
        )
        _wait_for(
            lambda: (
                load_task(healthy_cfg, running.task_id).state["projection"] == "running" and running_marker.exists()
            ),
            timeout=45,
            description="qualification setup running Attempt",
        )
        expected_projects = {healthy.project_id, *(item[1].project_id for item in blocked)}
        _wait_for(
            lambda: expected_projects.issubset(set(get_machine_agent_status(runtime)["reconciled_project_ids"])),
            timeout=45,
            description="current-agent recovery for every qualification binding",
        )
        foreground_hold.touch()
        executor_paths = machine_runtime_paths(runtime.root)

        def prior_renewal_is_drained() -> bool:
            for path in executor_paths["project_io_requests"].glob("*.json"):
                try:
                    request = read_json(path)["project_io_request"]
                except FileNotFoundError:
                    continue
                if request["project_id"] == healthy.project_id and request["operation_kind"] == "authority_renewal":
                    return False
            return True

        # A foreground grant already in flight when the hold is armed must not
        # satisfy the later measurement using a pre-measurement lease change.
        _wait_for(prior_renewal_is_drained, description="prior renewal fully consumed")
        before_expiry = load_task(healthy_cfg, running.task_id).claim_control["active_claim"]["lease_expires_at"]
        # The default lease policy renews after ten seconds. Hold only new
        # foreground grants so the exact running Attempt is due at measurement.
        time.sleep(10.1)

        primary_marker = tmp_path / "primary"
        primary = submit(
            healthy_cfg,
            [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path\nimport time\n"
                    f"marker=Path({str(primary_marker)!r}); finish=Path({str(primary_finish)!r})\n"
                    "marker.touch()\n"
                    "deadline=time.monotonic()+60\n"
                    "while not finish.exists():\n"
                    "    if time.monotonic()>deadline: raise SystemExit(99)\n"
                    "    time.sleep(0.02)\n"
                ),
            ],
            working_dir=healthy_cfg.project_root,
            requested_gpus=0,
            requested_cpus=1,
        )
        for blocked_cfg, _binding, schema, saved in blocked:
            schema.rename(saved)
            os.mkfifo(schema)
            publish_project_activation(blocked_cfg, "latency_qualification")
        with runtime.agent_lifecycle_guard():
            runtime.activation_wake.publish_locked()
        executor_paths = machine_runtime_paths(runtime.root)

        def held_project_ids() -> set[str]:
            held_ids: set[str] = set()
            for path in executor_paths["project_io_requests"].glob("*.json"):
                try:
                    request = read_json(path)["project_io_request"]
                except (OSError, KeyError, TypeError, ValueError):
                    continue
                if (executor_paths["project_io_processes"] / path.name).exists():
                    held_ids.add(request["project_id"])
            return held_ids

        blocked_ids = {item[1].project_id for item in blocked}
        _wait_for(
            lambda: blocked_ids.issubset(held_project_ids()),
            timeout=30,
            description="two held qualification workers",
            on_timeout=lambda: describe_project_io_workers(ProjectIOExecutor(runtime)),
        )
        background_arm.touch()
        _wait_for(
            background_started.exists,
            timeout=30,
            description="one already-running healthy background request",
        )
        assert blocked_ids.issubset(held_project_ids())
        measured_at = time.monotonic()
        measurement_gate.touch()

        def completed_healthy_timings() -> list[dict]:
            lines = trace.read_text().splitlines(keepends=True) if trace.exists() else []
            timings = [json.loads(line) for line in lines if line.endswith("\n")]
            return [
                item
                for item in timings
                if item["project_id"] == healthy.project_id
                and (item["qualification_background"] or item["started_at"] >= measured_at)
            ]

        def primary_admitted_and_running_renewed() -> bool:
            primary_task = load_task(healthy_cfg, primary.task_id)
            running_task = load_task(healthy_cfg, running.task_id)
            claim = primary_task.claim_control.get("active_claim")
            renewed_expiry = running_task.claim_control["active_claim"]["lease_expires_at"]
            reservations = active_reservations(runtime.root)
            process_path = local_paths(runtime.project_paths(healthy.project_id)["root"])["processes"] / (
                f"{running_task.attempt_control['current_attempt_id']}.json"
            )
            local_expiry = read_json(process_path)["process"].get("lease_expires_at")
            kinds = {item["operation_kind"] for item in completed_healthy_timings()}
            return (
                isinstance(claim, dict)
                and any(item.get("task_id") == primary.task_id for item in reservations)
                and renewed_expiry == before_expiry is None
                and running_task.claim_control["active_claim"]["authority_mode"] == "holder_bound"
                and local_expiry == renewed_expiry
                and {"scheduler_claim", "authority_renewal"} <= kinds
            )

        _wait_for(
            primary_admitted_and_running_renewed,
            timeout=15,
            description="exact primary admission and applied running-Attempt renewal",
            on_timeout=lambda: {
                "primary": load_task(healthy_cfg, primary.task_id).to_dict(),
                "running": load_task(healthy_cfg, running.task_id).to_dict(),
                "held_projects": sorted(held_project_ids()),
                "trace": trace.read_text() if trace.exists() else "",
            },
        )
        assert time.monotonic() - measured_at <= 15
        assert blocked_ids.issubset(held_project_ids())
        healthy_timings = completed_healthy_timings()
        assert healthy_timings
        assert max(item["elapsed_seconds"] for item in healthy_timings) <= 2.0
        kinds = {item["operation_kind"] for item in healthy_timings}
        assert kinds & {
            "upgrade_service",
            "submission_control_service",
            "observation_service",
            "machine_snapshot_publish",
            "activation_observe",
            "activation_consumer_register",
            "activation_consumer_ack",
        }
        assert "scheduler_claim" in kinds
        assert "authority_renewal" in kinds
    finally:
        measurement_gate = tmp_path / "measurement-gate"
        measurement_gate.touch()
        running_finish.touch()
        primary_finish.touch()
        try:
            stop_machine_agent(runtime)
        except (OSError, RuntimeError, ValueError):
            pass
        for process in (qualifying_agent,):
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=8)
        for _cfg, _binding, schema, saved in blocked:
            if schema.exists():
                schema.unlink()
            if saved.exists():
                saved.rename(schema)


@pytest.mark.parametrize(
    ("binding_count", "intent_window"),
    [
        pytest.param(1, 64, id="1"),
        pytest.param(2, 64, id="2"),
        pytest.param(3, 2, id="3-window-2"),
        pytest.param(8, 64, marks=pytest.mark.stress, id="8"),
        pytest.param(65, 64, marks=pytest.mark.stress, id="65"),
    ],
)
def test_real_agent_healthy_binding_scale_preserves_incumbent_progress(
    tmp_path: Path,
    binding_count: int,
    intent_window: int,
) -> None:
    """Qualify mixed scheduling, supervision, and background progress at scale."""
    from qqtools.plugins.qexp.lease import LeasePolicy, save_lease_policy

    runtime = _initialized_runtime(tmp_path / "machine-runtime", agent_mode="daemon")
    case_deadline = time.monotonic() + 600

    def remaining_seconds():
        return max(0.0, case_deadline - time.monotonic())

    configs = []
    first_markers = []
    for index in range(binding_count):
        cfg = init_shared_root(
            tmp_path / f"project-{index:02d}" / ".qexp",
            "gpu-1",
            agent_mode="daemon",
            runtime_root=tmp_path / f"legacy-{index:02d}",
        )
        if binding_count == 65:
            # Keep this scale qualification about bounded-window fairness. Each
            # fresh registration is still due immediately under this policy and
            # must complete one isolated renewal, while dedicated short-lease
            # tests retain the default-cadence saturation and expiry coverage.
            save_lease_policy(cfg, LeasePolicy(ttl_seconds=14_400, renew_interval_seconds=7_200.0))
        _register_project(runtime, cfg)
        configs.append(cfg)
    set_cpu_lane_capacity(runtime.root, capacity=min(4, binding_count))
    if intent_window == 64:
        process = start_machine_agent(runtime, available_gpus=[], loop_interval=0.05)
    else:
        # Exercise a real agent, worker processes, runners, and durable state
        # across the same bounded-window boundary without 130 training launches.
        agent_script = """
import sys
from qqtools.plugins.qexp.agent import project_io_admission, project_io_controller
from qqtools.plugins.qexp.agent.lifecycle import run_machine_agent_loop
project_io_admission._MAX_PENDING_INTENTS = int(sys.argv[2])
project_io_controller._MAX_RETAINED_SCHEDULER_INTENTS = int(sys.argv[2])
run_machine_agent_loop(sys.argv[1], available_gpus=[], loop_interval=0.05)
"""
        process = subprocess.Popen(
            [sys.executable, "-c", agent_script, str(runtime.root), str(intent_window)], start_new_session=True
        )
    first_tasks = []
    second_tasks = []
    try:
        expected = {binding.project_id for binding in runtime.load_registry()[1]}
        _wait_for(
            lambda: expected.issubset(set(get_machine_agent_status(runtime)["reconciled_project_ids"])),
            timeout=remaining_seconds(),
            description=f"real-agent recovery for every one of {binding_count} bindings",
        )
        first_entries = list(enumerate(configs))
        for index, cfg in first_entries:
            marker = tmp_path / f"first-{index:02d}"
            first_tasks.append(
                (
                    cfg,
                    submit(
                        cfg,
                        [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"],
                        working_dir=cfg.project_root,
                        requested_gpus=0,
                        requested_cpus=1,
                    ),
                    marker,
                )
            )
            first_markers.append(marker)
        # Add a second attempt-bearing Task while the first roster still
        # contains continuously eligible incumbents. Repeating every owner,
        # including the one initially outside the configured intent window, proves that
        # a continuing arrival wave cannot strand an incumbent after its first
        # scheduling/supervision cycle.
        _wait_for(
            lambda: any(marker.exists() for marker in first_markers),
            timeout=remaining_seconds(),
            description=f"first scheduling result across {binding_count} bindings",
        )
        second_entries = list(enumerate(configs))
        for index, cfg in second_entries:
            marker = tmp_path / f"second-{index:02d}"
            second_tasks.append(
                (
                    cfg,
                    submit(
                        cfg,
                        [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"],
                        working_dir=cfg.project_root,
                        requested_gpus=0,
                        requested_cpus=1,
                    ),
                    marker,
                )
            )

        def every_task_succeeded() -> bool:
            return all(
                load_task(cfg, task.task_id).state["projection"] == "succeeded" for cfg, task, _marker in first_tasks
            ) and all(load_task(cfg, task.task_id).state["projection"] == "succeeded" for cfg, task, _ in second_tasks)

        _wait_for(
            every_task_succeeded,
            timeout=remaining_seconds(),
            description=f"scale-{binding_count} scheduling and supervision results",
        )
        assert all(marker.exists() for marker in first_markers)
        assert all(marker.exists() for _cfg, _task, marker in second_tasks)

        # The submitted work above activates every binding. Prove background
        # publication reaches every incumbent while two full scheduling and
        # supervision waves share the bounded executor.
        _wait_for(
            lambda: all(machine_state_path(cfg, "agent.json").exists() for cfg in configs),
            timeout=remaining_seconds(),
            description=f"background snapshots for activated bindings at scale {binding_count}",
        )

        status = get_machine_agent_status(runtime)
        assert expected.issubset(set(status["reconciled_project_ids"]))
        selected_roots = {str(cfg.shared_root) for cfg, _task, _marker in first_tasks}
        _wait_for(
            lambda: (
                not any(record.get("shared_root") in selected_roots for record in active_reservations(runtime.root))
            ),
            timeout=min(30, remaining_seconds()),
            description="scale qualification target reservation cleanup",
        )
    finally:
        try:
            stop_machine_agent(runtime, timeout=10)
        except (OSError, RuntimeError, TimeoutError):
            if process.poll() is None:
                process.kill()
        process.wait(timeout=10)


def _cleanup(runtime: MachineRuntime, process: subprocess.Popen) -> None:
    try:
        if get_machine_agent_status(runtime)["is_running"]:
            stop_machine_agent(runtime, timeout=5.0)
    except (OSError, RuntimeError, TimeoutError):
        if process.poll() is None:
            process.kill()
    process.wait(timeout=5)


@pytest.mark.parametrize("is_sighup", [False, True], ids=["agent-stop", "sighup"])
def test_li01_training_remains_live_and_is_not_relaunched(tmp_path: Path, is_sighup: bool) -> None:
    cfg, runtime, task, marker, process = _start_case(tmp_path, should_wait=True)
    try:
        _wait_running(cfg, task.task_id, marker)
        attempt_id = load_task(cfg, task.task_id).attempt_control["current_attempt_id"]
        assert isinstance(attempt_id, str)
        attempt_file = attempt_path(cfg.shared_root, task.task_id, 1)
        _wait_for(lambda: read_json(attempt_file)["attempt"]["process"].get("process_group_id") is not None)
        original_process = read_json(attempt_file)["attempt"]["process"]
        _wait_for(marker.exists, timeout=5.0)
        if is_sighup:
            process.send_signal(signal.SIGHUP)
            assert process.wait(timeout=10) == 0
            summary = get_machine_agent_status(runtime)["diagnostics"]["summary"]
            assert summary["reason"] == "stopped_by_signal"
            assert summary["handled_signal"] == signal.SIGHUP
        else:
            stop_machine_agent(runtime)
        progress = marker.with_suffix(".progress")
        _wait_for(progress.exists)
        previous = progress.read_text()
        _wait_for(lambda: progress.read_text() not in {"", previous})
        assert len(active_reservations(runtime.root)) == 1
        assert marker.read_text(encoding="utf-8") == "1"
        restarted = restart_machine_agent(runtime, available_gpus=[0], loop_interval=0.1)
        marker.with_suffix(".finish").touch()
        terminal = _wait_terminal(cfg, task.task_id)
        assert terminal.attempt_control["next_attempt_number"] == 2
        assert marker.read_text(encoding="utf-8") == "1"
        assert restarted.pid != process.pid
        attempt = read_json(attempt_path(cfg.shared_root, task.task_id, 1))["attempt"]
        assert attempt["attempt_id"] == attempt_id
        for key in ("wrapper_pid", "wrapper_start_time_ticks", "process_group_id", "process_group_start_time_ticks"):
            assert attempt["process"][key] == original_process[key]
    finally:
        _cleanup(runtime, process)


def test_li02_offline_completion_preserves_exit_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QEXP_VISIBLE_GPUS", "0,1")
    project_root = tmp_path / "project"
    cfg = init_shared_root(project_root / ".qexp", "gpu-1", agent_mode="daemon", runtime_root=tmp_path / "legacy")
    runtime = _initialized_runtime(tmp_path / "machine-runtime", agent_mode="daemon")
    binding = _register_project(runtime, cfg)
    paths = local_paths(runtime.project_paths(binding.project_id)["root"])
    branches = []
    for name, exit_code, phase in (("success", 0, "succeeded"), ("failure", 7, "failed")):
        marker = tmp_path / name
        command = [
            sys.executable,
            "-c",
            (
                "from pathlib import Path\nimport time\n"
                f"p=Path({str(marker)!r})\np.write_text('1')\n"
                "deadline=time.monotonic()+60\n"
                "while not p.with_suffix('.finish').exists():\n"
                "    if time.monotonic()>deadline: raise SystemExit(99)\n"
                "    time.sleep(0.01)\n"
                f"raise SystemExit({exit_code})"
            ),
        ]
        task = submit(cfg, command, working_dir=project_root)
        branches.append((name, exit_code, phase, task, marker))
    process = start_machine_agent(runtime, available_gpus=[0, 1], loop_interval=0.1)
    try:
        wait_all(
            {
                f"running:{name}": lambda task=task, marker=marker: (
                    load_task(cfg, task.task_id).state["projection"] == "running" and marker.exists()
                )
                for name, _exit_code, _phase, task, marker in branches
            }
        )
        attempts = {
            name: load_task(cfg, task.task_id).attempt_control["current_attempt_id"]
            for name, _exit_code, _phase, task, _marker in branches
        }
        stop_machine_agent(runtime)
        for _name, _exit_code, _phase, _task, marker in branches:
            marker.with_suffix(".finish").touch()
        wait_all(
            {
                f"observation:{name}": lambda name=name: (paths["observations"] / f"{attempts[name]}.json").exists()
                for name, _exit_code, _phase, _task, _marker in branches
            }
        )
        assert all(load_task(cfg, task.task_id).state["projection"] == "running" for *_, task, _ in branches)
        start = time.monotonic()
        restarted = start_machine_agent(runtime, available_gpus=[0, 1], loop_interval=0.1)
        wait_all(
            {
                f"terminal:{name}": lambda task=task, phase=phase: (
                    load_task(cfg, task.task_id).state["projection"] == phase
                )
                for name, _exit_code, phase, task, _marker in branches
            }
        )
        _wait_for(
            lambda: not active_reservations(runtime.root),
            timeout=max(0, CONVERGENCE_BUDGET_SECONDS - (time.monotonic() - start)),
        )
        assert time.monotonic() - start <= CONVERGENCE_BUDGET_SECONDS
        for name, exit_code, phase, task, marker in branches:
            assert load_task(cfg, task.task_id).state["projection"] == phase
            attempt = read_json(attempt_path(cfg.shared_root, task.task_id, 1))["attempt"]
            assert attempt["result"]["exit_code"] == exit_code, name
            assert marker.read_text(encoding="utf-8") == "1", name
        assert active_reservations(runtime.root) == []
        stop_machine_agent(runtime)
        restarted.wait(timeout=5)
    finally:
        for _name, _exit_code, _phase, _task, marker in branches:
            marker.with_suffix(".finish").touch()
        _cleanup(runtime, process)


def test_agent_projects_both_progress_versions_through_isolated_transactions(tmp_path: Path) -> None:
    from qqtools.plugins.qexp.observer import inspect_task
    from qqtools.plugins.qexp.progress_policy import set_progress_policy
    from qqtools.plugins.qexp.runtime.submission import submit_specs

    project_root = tmp_path / "project"
    cfg = init_shared_root(project_root / ".qexp", "gpu-1", agent_mode="daemon", runtime_root=tmp_path / "legacy")
    runtime = _initialized_runtime(tmp_path / "machine-runtime", agent_mode="daemon")
    binding = _register_project(runtime, cfg)
    set_progress_policy(cfg.shared_root, 1)
    marker = tmp_path / "producer"
    command = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path\nimport time\nfrom qqtools.qexp import progress\n"
            "def report(number):\n"
            "    assert progress.update(stage='train' if number==1 else 'done', current=number, total=2, "
            "unit='step', metrics={'loss':1/number}, "
            "overall=progress.Counter(current=number,total=2,unit='step',label='Training'))\n"
            "    progress.flush(timeout=1)\n"
            f"marker=Path({str(marker)!r})\nreport(1)\nmarker.write_text('1')\n"
            "deadline=time.monotonic()+120\n"
            "while not marker.with_suffix('.finish').exists():\n"
            "    if time.monotonic()>deadline: raise SystemExit(99)\n"
            "    time.sleep(0.02)\n"
            "report(2)\n"
        ),
    ]
    task = submit_specs(cfg, [{"command": command, "working_directory": str(project_root), "live_progress": True}])[0]
    process = start_machine_agent(runtime, available_gpus=[0], loop_interval=0.1)
    try:
        _wait_running(cfg, task.task_id, marker)

        def all_reports(current: int) -> bool:
            view = inspect_task(cfg, task.task_id)
            return (
                view["progress"].get("progress", {}).get("current") == current
                and view["progress_extended"].get("progress", {}).get("current") == current
                and view["progress_scoped"].get("progress", {}).get("overall", {}).get("current") == current
            )

        # All three protocols share the binding slot with registration, scheduling,
        # supervision and maintenance. Their full cold-start chain is not the
        # single-peer primary/renewal 15-second qualification.
        _wait_for(
            lambda: all_reports(1),
            timeout=45,
            description="all three live progress projections",
            on_timeout=lambda: inspect_task(cfg, task.task_id),
        )
        view = inspect_task(cfg, task.task_id)
        assert view["progress_extended"]["progress"]["metrics"] == {"loss": 1.0}
        from datetime import datetime

        accepted_v1 = datetime.fromisoformat(view["progress"]["reported_at"].replace("Z", "+00:00"))
        accepted_v2 = datetime.fromisoformat(view["progress_extended"]["reported_at"].replace("Z", "+00:00"))
        # Legacy JSON remains timestamp-only even when the richer same-source
        # report was accepted first by the fair three-channel rotation.
        assert view["selected_progress_version"] == (2 if accepted_v2 >= accepted_v1 else 1)
        assert view["selected_progress_protocol_version"] == 3
        assert len({view[key]["source_update_id"] for key in ("progress", "progress_extended", "progress_scoped")}) == 1
        assert view["progress_scoped"]["progress"]["activity"] == view["progress"]["progress"]
        assert view["progress_scoped"]["progress"]["overall"]["label"] == "Training"
        assert marker.read_text() == "1"
        marker.with_suffix(".finish").touch()
        terminal = _wait_terminal(cfg, task.task_id)
        assert terminal.state["projection"] == "succeeded"
        _wait_for(
            lambda: all_reports(2),
            timeout=75,
            description="all three terminal progress projections",
            on_timeout=lambda: inspect_task(cfg, task.task_id),
        )
        assert load_task(cfg, task.task_id).attempt_control["next_attempt_number"] == 2
        local_root = runtime.project_paths(binding.project_id)["root"]
        _wait_for(
            lambda: (
                not any((local_root / "progress-contexts").glob("*.json"))
                and not any((local_root / "progress-v2-contexts").glob("*.json"))
                and not any((local_root / "progress-v3-contexts").glob("*.json"))
            ),
            timeout=15,
        )
    finally:
        marker.with_suffix(".finish").touch()
        _cleanup(runtime, process)


def test_li03_expired_claim_recovers_same_attempt_without_relaunch(tmp_path: Path) -> None:
    cfg, runtime, task, marker, process = _start_case(tmp_path, seconds=2.0)
    try:
        _wait_running(cfg, task.task_id, marker)
        stop_machine_agent(runtime)
        persist_legacy_orphan(cfg, task.task_id)
        start_machine_agent(runtime, available_gpus=[0], loop_interval=0.1)
        terminal = _wait_terminal(cfg, task.task_id)
        assert terminal.state["projection"] == "succeeded"
        assert marker.read_text(encoding="utf-8") == "1"
    finally:
        _cleanup(runtime, process)


def test_li04_sigkill_agent_does_not_kill_runner(tmp_path: Path) -> None:
    cfg, runtime, task, marker, process = _start_case(tmp_path, should_wait=True)
    try:
        _wait_running(cfg, task.task_id, marker)
        os.kill(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
        marker.with_suffix(".finish").touch()
        binding = runtime.load_registry()[1][0]
        paths = local_paths(runtime.project_paths(binding.project_id)["root"])
        attempt_id = load_task(cfg, task.task_id).attempt_control["current_attempt_id"]
        _wait_for((paths["observations"] / f"{attempt_id}.json").exists)
        assert marker.read_text(encoding="utf-8") == "1"
        start_machine_agent(runtime, available_gpus=[0], loop_interval=0.1)
        assert _wait_terminal(cfg, task.task_id).state["projection"] == "succeeded"
    finally:
        _cleanup(runtime, process)


@pytest.mark.parametrize("is_finished", [False, True])
def test_li03_real_peer_cannot_expire_durable_execution(tmp_path: Path, is_finished: bool) -> None:
    from qqtools.plugins.qexp.commands.group import change_worker, create_group
    from qqtools.plugins.qexp.commands.task import offer
    from qqtools.plugins.qexp.lease import LeasePolicy, save_lease_policy

    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", agent_mode="daemon", runtime_root=tmp_path / "legacy"
    )
    save_lease_policy(
        cfg,
        LeasePolicy(
            ttl_seconds=4,
            renew_interval_seconds=0.5,
            max_clock_skew_seconds=0.1,
            renewal_commit_margin_seconds=0.1,
            retry_initial_seconds=0.1,
            retry_max_seconds=0.2,
        ),
    )
    runtime = _initialized_runtime(tmp_path / "machine", agent_mode="daemon")
    binding = _register_project(runtime, cfg)
    create_group(cfg, "peers")
    change_worker(cfg, "peers", "gpu-1", "add")
    change_worker(cfg, "peers", "gpu-2", "add")
    marker, finish = tmp_path / "count", tmp_path / "finish"
    command = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path\nimport time\n"
            f"p=Path({str(marker)!r})\np.write_text(str(int(p.read_text())+1) if p.exists() else '1')\n"
            "deadline=time.monotonic()+45\n"
            f"while not Path({str(finish)!r}).exists():\n"
            "    if time.monotonic()>deadline: raise SystemExit(99)\n"
            "    time.sleep(0.02)\n"
        ),
    ]
    task = submit(cfg, command, working_dir=cfg.project_root, group="peers", sharing_mode="spillover")
    offer(cfg, task.task_id)
    process = start_machine_agent(runtime, available_gpus=[0], loop_interval=0.1)
    peer = None
    observed = tmp_path / "peer-observed"
    try:
        _wait_running(cfg, task.task_id, marker)
        claim = load_task(cfg, task.task_id).claim_control["active_claim"]
        assert claim["authority_mode"] == "holder_bound"
        assert claim["lease_expires_at"] is None
        stop_machine_agent(runtime)
        if is_finished:
            finish.touch()
            paths = local_paths(runtime.project_paths(binding.project_id)["root"])
            _wait_for((paths["observations"] / f"{claim['attempt_id']}.json").exists)
        script = """
import sys,time
from pathlib import Path
from qqtools.plugins.qexp.config_types import RootConfig
from qqtools.plugins.qexp.scheduler import expire_claim, claim_task
from qqtools.plugins.qexp.runtime.tasks import load_task
shared,root,task_id,attempt_id,token,observed = sys.argv[1:]
cfg=RootConfig(Path(shared),Path(shared).parent,"gpu-2",Path(root))
deadline=time.monotonic()+6
while time.monotonic()<deadline:
    assert not expire_claim(cfg,task_id,attempt_id,int(token))
    current=load_task(cfg,task_id)
    assert current.state["projection"]=="running"
    assert current.attempt_control["current_attempt_id"]==attempt_id
    assert current.claim_control["active_claim"]["fencing_token"]==int(token)
    assert claim_task(cfg,task_id,[0]) is None
    time.sleep(0.01)
Path(observed).touch()
"""
        peer = subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                str(cfg.shared_root),
                str(tmp_path / "peer"),
                task.task_id,
                claim["attempt_id"],
                str(claim["fencing_token"]),
                str(observed),
            ],
            start_new_session=True,
        )
        _wait_for(
            observed.exists,
            description="LI-03 peer preserves durable ownership beyond old TTL",
            on_timeout=lambda: _li03_timeout_diagnostics(cfg, runtime, task.task_id, peer, observed=observed),
        )
        assert peer.wait(timeout=5) == 0
        assert marker.read_text() == "1"
        restarted = start_machine_agent(runtime, available_gpus=[0], loop_interval=0.1)
        if not is_finished:
            _wait_for(
                lambda: load_task(cfg, task.task_id).state["projection"] == "running",
                description="LI-03 recovered task projection",
                on_timeout=lambda: _li03_timeout_diagnostics(cfg, runtime, task.task_id, restarted, observed=observed),
            )
            assert load_task(cfg, task.task_id).attempt_control["current_attempt_id"] == claim["attempt_id"]
            finish.touch()
        _wait_for(
            lambda: load_task(cfg, task.task_id).state["projection"] == "succeeded",
            description="LI-03 recovered task terminal projection",
            on_timeout=lambda: _li03_timeout_diagnostics(cfg, runtime, task.task_id, restarted, observed=observed),
        )
        assert load_task(cfg, task.task_id).state["projection"] == "succeeded"
        assert marker.read_text() == "1"
        stop_machine_agent(runtime)
        restarted.wait(timeout=5)
    finally:
        finish.touch()
        if peer is not None and peer.poll() is None:
            peer.kill()
            peer.wait(timeout=5)
        _cleanup(runtime, process)


@pytest.mark.parametrize("boundary", ["before_authorization", "after_authorization"])
def test_li05_launch_boundary_has_no_duplicate_authorized_process(tmp_path: Path, boundary: str) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", agent_mode="daemon", runtime_root=tmp_path / "legacy"
    )
    runtime = _initialized_runtime(tmp_path / "machine", agent_mode="daemon")
    _register_project(runtime, cfg)
    marker = tmp_path / "launch-count"
    task = submit(
        cfg,
        [
            sys.executable,
            "-c",
            f"from pathlib import Path; p=Path({str(marker)!r}); p.write_text(str(int(p.read_text())+1) if p.exists() else '1')",
        ],
        working_dir=cfg.project_root,
    )
    reached = tmp_path / "boundary"
    script = """
import os, sys
from pathlib import Path
from qqtools.plugins.qexp.agent.lifecycle import run_machine_agent_loop
from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor
boundary, root, reached = sys.argv[1:]
original_start = ProjectIOExecutor.start
original_consume = ProjectIOExecutor.consume
def start(self, request_id, *args, **kwargs):
    request = self._load_request(request_id)
    if boundary == "before_authorization" and request.operation_kind == "scheduler_launch_authorize":
        Path(reached).write_text(boundary)
        os._exit(73)
    return original_start(self, request_id, *args, **kwargs)
def consume(self, request_id, current_request):
    result = original_consume(self, request_id, current_request)
    if (
        boundary == "after_authorization"
        and current_request.operation_kind == "scheduler_launch_authorize"
        and result is not None
        and result.evidence.get("outcome") == "authorized"
    ):
        Path(reached).write_text(boundary)
        os._exit(73)
    return result
ProjectIOExecutor.start = start
ProjectIOExecutor.consume = consume
run_machine_agent_loop(root, loop_interval=0.1, available_gpus=[0])
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, boundary, str(runtime.root), str(reached)], start_new_session=True
    )
    lab = LifecycleLab(runtime, [0])
    lab.add_branch(LifecycleBranch(boundary, cfg, task, marker))
    lab.adopt_process(process)
    try:
        _wait_for(reached.exists)
        assert process.wait(timeout=5) == 73
        assert not marker.exists()
        original_attempt = load_task(cfg, task.task_id).attempt_control["current_attempt_id"]
        restarted = start_machine_agent(runtime, available_gpus=[0], loop_interval=0.1)
        _wait_for(lambda: marker.exists() or load_task(cfg, task.task_id).state["projection"] == "blocked")
        if marker.exists():
            assert _wait_terminal(cfg, task.task_id).state["projection"] == "succeeded"
            assert marker.read_text() == "1"
            assert (
                read_json(attempt_path(cfg.shared_root, task.task_id, 1))["attempt"]["attempt_id"] == original_attempt
            )
        else:
            assert load_task(cfg, task.task_id).state["projection"] == "blocked"
        assert load_task(cfg, task.task_id).attempt_control["next_attempt_number"] == 2
        stop_machine_agent(runtime)
        restarted.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        lab.close()


def test_li05_agent_crash_between_process_creation_and_registration(tmp_path: Path) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", agent_mode="daemon", runtime_root=tmp_path / "legacy"
    )
    runtime = _initialized_runtime(tmp_path / "machine", agent_mode="daemon")
    binding = _register_project(runtime, cfg)
    paths = local_paths(runtime.project_paths(binding.project_id)["root"])
    marker, reached, resume = (tmp_path / name for name in ("count", "created", "resume"))
    task = submit(
        cfg,
        [
            sys.executable,
            "-c",
            f"from pathlib import Path; p=Path({str(marker)!r}); p.write_text(str(int(p.read_text())+1) if p.exists() else '1')",
        ],
        working_dir=cfg.project_root,
    )
    runner_script = """
import sys, time
from pathlib import Path
from qqtools.plugins.qexp import runner
reached, resume = map(Path, sys.argv[1:3])
original = runner._publish_registration
def registration(*args, **kwargs):
    reached.touch()
    deadline = time.monotonic() + 30
    while not resume.exists():
        if time.monotonic() >= deadline: raise TimeoutError("registration barrier")
        time.sleep(0.02)
    return original(*args, **kwargs)
runner._publish_registration = registration
try:
    result = runner.main(sys.argv[3:])
except Exception:
    import traceback
    reached.with_suffix(".error").write_text(traceback.format_exc())
    raise
raise SystemExit(result)
"""
    agent_script = """
import sys
from qqtools.plugins.qexp.executor import Executor
from qqtools.plugins.qexp.agent.lifecycle import run_machine_agent_loop
original = Executor.build_runner_argv
def argv(self, *args, **kwargs):
    command = original(self, *args, **kwargs)
    return [sys.executable, "-c", "exec(" + repr(sys.argv[2]) + ")", sys.argv[3], sys.argv[4], *command[3:]]
Executor.build_runner_argv = argv
run_machine_agent_loop(sys.argv[1], loop_interval=0.1, available_gpus=[0])
"""
    process = subprocess.Popen(
        [sys.executable, "-c", agent_script, str(runtime.root), runner_script, str(reached), str(resume)],
        start_new_session=True,
    )
    lab = LifecycleLab(runtime, [0])
    lab.add_branch(LifecycleBranch("registration", cfg, task, marker))
    lab.adopt_process(process)
    try:
        _wait_for(lambda: reached.exists() or reached.with_suffix(".error").exists())
        assert not reached.with_suffix(".error").exists(), reached.with_suffix(".error").read_text()
        _wait_for(marker.exists)
        attempt_id = load_task(cfg, task.task_id).attempt_control["current_attempt_id"]
        assert not (paths["registrations"] / f"{attempt_id}.json").exists()
        process.kill()
        process.wait(timeout=5)
        restarted = start_machine_agent(runtime, available_gpus=[0], loop_interval=0.1)
        assert marker.read_text() == "1"
        resume.touch()
        assert _wait_terminal(cfg, task.task_id).state["projection"] == "succeeded"
        assert read_json(attempt_path(cfg.shared_root, task.task_id, 1))["attempt"]["attempt_id"] == attempt_id
        assert marker.read_text() == "1"
        stop_machine_agent(runtime)
        restarted.wait(timeout=5)
    finally:
        resume.touch()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        lab.close()


@pytest.mark.parametrize("boundary", ["attempt", "task", "reservation"])
@pytest.mark.parametrize("is_orphaned", [False, True])
def test_li06_terminal_publication_is_idempotent(tmp_path: Path, boundary: str, is_orphaned: bool) -> None:
    # Hold the workload until the original agent has stopped so terminal truth
    # cannot commit before the fault-injected agent reaches its boundary.
    cfg, runtime, task, marker, process = _start_case(tmp_path, should_wait=True)
    crashing = None
    try:
        _wait_running(cfg, task.task_id, marker)
        attempt_id = load_task(cfg, task.task_id).attempt_control["current_attempt_id"]
        stop_machine_agent(runtime)
        assert load_task(cfg, task.task_id).state["projection"] == "running"
        marker.with_suffix(".finish").touch()
        binding = runtime.load_registry()[1][0]
        paths = local_paths(runtime.project_paths(binding.project_id)["root"])
        observation = paths["observations"] / f"{attempt_id}.json"
        _wait_for(observation.exists)
        if is_orphaned:
            persist_legacy_orphan(cfg, task.task_id)
        reached = tmp_path / "crash-boundary"
        worker_script = """
import os, sys
from pathlib import Path
from qqtools.plugins.qexp import lifecycle
from qqtools.plugins.qexp.agent import project_io_worker
boundary, reached = sys.argv[1:3]
def crash():
    Path(reached).write_text(boundary)
    os._exit(73)
original_replace = lifecycle.atomic_replace
def replace(path, value, *args, **kwargs):
    result = original_replace(path, value, *args, **kwargs)
    if boundary == "attempt" and value.get("attempt", {}).get("phase") == "succeeded":
        crash()
    return result
lifecycle.atomic_replace = replace
original_save = lifecycle.save_task
def save(cfg, task, *args, **kwargs):
    result = original_save(cfg, task, *args, **kwargs)
    if boundary == "task" and task.state["projection"] == "succeeded":
        crash()
    return result
lifecycle.save_task = save
raise SystemExit(project_io_worker.main(sys.argv[3:]))
"""
        script = """
import os, sys
from pathlib import Path
from qqtools.plugins.qexp.agent import project_io_executor, project_io_supervision
from qqtools.plugins.qexp.agent.lifecycle import run_machine_agent_loop
boundary, root, reached, worker_script = sys.argv[1:]
def crash():
    Path(reached).write_text(boundary)
    os._exit(73)
original_popen = project_io_executor.subprocess.Popen
original_poll = project_io_executor.ProjectIOExecutor.poll
def popen(command, *args, **kwargs):
    if command[1:3] == ["-m", "qqtools.plugins.qexp.agent.project_io_worker"]:
        command = [command[0], "-c", worker_script, boundary, reached, *command[3:]]
    return original_popen(command, *args, **kwargs)
project_io_executor.subprocess.Popen = popen
def poll(self):
    result = original_poll(self)
    if boundary in {"attempt", "task"} and Path(reached).exists():
        crash()
    return result
project_io_executor.ProjectIOExecutor.poll = poll
original_apply = project_io_supervision._apply_terminal_local_effects
def apply(*args, **kwargs):
    result = original_apply(*args, **kwargs)
    if boundary == "reservation":
        crash()
    return result
project_io_supervision._apply_terminal_local_effects = apply
run_machine_agent_loop(root, loop_interval=0.1, available_gpus=[0])
"""
        crashing = subprocess.Popen(
            [sys.executable, "-c", script, boundary, str(runtime.root), str(reached), worker_script],
            start_new_session=True,
        )
        _wait_for(reached.exists)
        assert crashing.wait(timeout=5) == 73
        assert observation.exists(), "exit evidence must survive interrupted publication"
        for _ in range(2):
            restarted = restart_machine_agent(runtime, available_gpus=[0], loop_interval=0.1)
            assert _wait_terminal(cfg, task.task_id).state["projection"] == "succeeded"
            _wait_for(lambda: not active_reservations(runtime.root))
            stop_machine_agent(runtime)
            restarted.wait(timeout=5)
        attempt = read_json(attempt_path(cfg.shared_root, task.task_id, 1))["attempt"]
        assert attempt["attempt_id"] == attempt_id
        assert attempt["result"]["exit_code"] == 0
        assert marker.read_text(encoding="utf-8") == "1"
    finally:
        marker.with_suffix(".finish").touch()
        if crashing is not None and crashing.poll() is None:
            crashing.kill()
            crashing.wait(timeout=5)
        _cleanup(runtime, process)


def _create_li07_case(tmp_path: Path, runtime: MachineRuntime, name: str):
    root = tmp_path / name
    cfg = init_shared_root(root / ".qexp", "gpu-1", agent_mode="daemon", runtime_root=root / "legacy")
    binding = _register_project(runtime, cfg)
    marker, finish = root / "launch-count", root / "finish"
    command = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path\nimport time\n"
            f"marker = Path({str(marker)!r})\nfinish = Path({str(finish)!r})\n"
            "marker.write_text(str(int(marker.read_text()) + 1) if marker.exists() else '1')\n"
            "deadline = time.monotonic() + 60\n"
            "while not finish.exists():\n"
            "    if time.monotonic() > deadline: raise SystemExit(99)\n"
            "    time.sleep(0.01)\n"
        ),
    ]
    task = submit(cfg, command, working_dir=root)
    paths = local_paths(runtime.project_paths(binding.project_id)["root"])
    return cfg, binding, task, marker, finish, paths


def _wait_li07_first_launch(cases, runtime: MachineRuntime) -> None:
    wait_all(
        {
            f"running:{index}": lambda case=case: (
                load_task(case[0], case[2].task_id).state["projection"] == "running" and case[3].exists()
            )
            for index, case in enumerate(cases)
        },
        timeout=30.0,
        stage="multi-project:first-launch",
        on_timeout=lambda: {
            "workers": describe_project_io_workers(ProjectIOExecutor(runtime)),
            "tasks": {
                binding.project_id: {
                    "task": load_task(cfg, task.task_id).to_dict(),
                    "launch_marker": marker.read_text() if marker.exists() else None,
                }
                for cfg, binding, task, marker, _finish, _paths in cases
            },
        },
    )


def _assert_li07_results(cases, identities) -> None:
    for index, (cfg, _binding, task, marker, _finish, _paths) in enumerate(cases):
        attempt = read_json(attempt_path(cfg.shared_root, task.task_id, 1))["attempt"]
        assert attempt["attempt_id"] == identities[index]
        assert attempt["result"]["exit_code"] == 0
        assert marker.read_text() == "1"


def test_li07_cold_start_bindings_keep_identity_and_reservations_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_count = 4
    gpu_ids = list(range(4))
    monkeypatch.setenv("QEXP_VISIBLE_GPUS", ",".join(map(str, gpu_ids)))
    runtime = _initialized_runtime(tmp_path / "machine-runtime", agent_mode="daemon")
    cases = [_create_li07_case(tmp_path, runtime, str(index)) for index in range(project_count)]
    process = start_machine_agent(runtime, available_gpus=gpu_ids, loop_interval=0.1)
    try:
        _wait_li07_first_launch(cases, runtime)
        identities = [
            load_task(cfg, task.task_id).attempt_control["current_attempt_id"]
            for cfg, _binding, task, _marker, _finish, _paths in cases
        ]
        reservations = active_reservations(runtime.root)
        assert {item["project_id"] for item in reservations} == {case[1].project_id for case in cases}
        assert len(reservations) == project_count
        stop_machine_agent(runtime)
        cases[0][4].touch()
        _wait_for((cases[0][5]["observations"] / f"{identities[0]}.json").exists)
        assert not (cases[1][5]["observations"] / f"{identities[1]}.json").exists()
        assert len(active_reservations(runtime.root)) == project_count
        start = time.monotonic()
        restarted = start_machine_agent(runtime, available_gpus=gpu_ids, loop_interval=0.1)
        wait_all(
            {
                "terminal:offline": lambda: (
                    load_task(cases[0][0], cases[0][2].task_id).state["projection"] == "succeeded"
                ),
                "reservations:live": lambda: len(active_reservations(runtime.root)) == project_count - 1,
            }
        )
        assert {item["project_id"] for item in active_reservations(runtime.root)} == {
            case[1].project_id for case in cases[1:]
        }
        for index, case in enumerate(cases[1:], 1):
            assert load_task(case[0], case[2].task_id).attempt_control["current_attempt_id"] == identities[index]
        assert time.monotonic() - start <= CONVERGENCE_BUDGET_SECONDS
        for case in cases[1:]:
            case[4].touch()
        wait_all(
            {
                f"terminal:{index}": lambda case=case: (
                    load_task(case[0], case[2].task_id).state["projection"] == "succeeded"
                )
                for index, case in enumerate(cases[1:], 1)
            }
        )
        _wait_for(lambda: not active_reservations(runtime.root))
        _assert_li07_results(cases, identities)
        stop_machine_agent(runtime)
        restarted.wait(timeout=5)
    finally:
        for _cfg, _binding, _task, _marker, finish, _paths in cases:
            finish.touch()
        _cleanup(runtime, process)


def test_li07_running_agent_discovers_dynamic_bindings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gpu_ids = list(range(4))
    monkeypatch.setenv("QEXP_VISIBLE_GPUS", ",".join(map(str, gpu_ids)))
    runtime = _initialized_runtime(tmp_path / "machine-runtime", agent_mode="daemon")
    cases = [_create_li07_case(tmp_path, runtime, str(index)) for index in range(2)]
    process = start_machine_agent(runtime, available_gpus=gpu_ids, loop_interval=0.1)
    try:
        _wait_li07_first_launch(cases, runtime)
        initial_ids = {case[1].project_id for case in cases}
        initial_attempt_ids = [
            load_task(cfg, task.task_id).attempt_control["current_attempt_id"]
            for cfg, _binding, task, _marker, _finish, _paths in cases
        ]
        cases.extend(_create_li07_case(tmp_path, runtime, str(index)) for index in range(2, 4))
        _wait_li07_first_launch(cases[2:], runtime)
        identities = [
            load_task(cfg, task.task_id).attempt_control["current_attempt_id"]
            for cfg, _binding, task, _marker, _finish, _paths in cases
        ]
        reservations = active_reservations(runtime.root)
        assert {item["project_id"] for item in reservations} == {case[1].project_id for case in cases}
        assert initial_ids < {item["project_id"] for item in reservations}
        assert identities[:2] == initial_attempt_ids
        stop_machine_agent(runtime)
        for case in cases:
            case[4].touch()
        wait_all(
            {
                f"observation:{index}": lambda case=case, index=index: (
                    case[5]["observations"] / f"{identities[index]}.json"
                ).exists()
                for index, case in enumerate(cases)
            }
        )
        restarted = start_machine_agent(runtime, available_gpus=gpu_ids, loop_interval=0.1)
        wait_all(
            {
                **{
                    f"terminal:{index}": lambda case=case: (
                        load_task(case[0], case[2].task_id).state["projection"] == "succeeded"
                    )
                    for index, case in enumerate(cases)
                },
                "reservations:released": lambda: not active_reservations(runtime.root),
            }
        )
        _assert_li07_results(cases, identities)
        stop_machine_agent(runtime)
        restarted.wait(timeout=5)
    finally:
        for _cfg, _binding, _task, _marker, finish, _paths in cases:
            finish.touch()
        _cleanup(runtime, process)


def test_li08_mismatched_exit_evidence_is_retained_as_blocker(tmp_path: Path) -> None:
    cfg, runtime, task, marker, process = _start_case(tmp_path, should_wait=True)
    try:
        _wait_running(cfg, task.task_id, marker)
        binding = runtime.load_registry()[1][0]
        paths = local_paths(runtime.project_paths(binding.project_id)["root"])
        stop_machine_agent(runtime)
        marker.with_suffix(".finish").touch()
        observation = paths["observations"] / f"{task.task_id}-attempt-1.json"
        _wait_for(
            observation.exists,
            timeout=5.0,
            description="LI-08 durable exit observation",
            on_timeout=lambda: {
                "attempt": load_task(cfg, task.task_id).attempt_control,
                "agent": get_machine_agent_status(runtime),
                "process_returncode": process.poll(),
                "observation": str(observation),
            },
        )
        value = read_json(observation)
        value["exit_observation"]["task_id"] = "other-task"
        atomic_replace(observation, value)
        start_machine_agent(runtime, available_gpus=[0], loop_interval=0.1)
        diagnostic = paths["authority_diagnostics"] / observation.name
        _wait_for(
            lambda: (
                diagnostic.exists()
                and read_json(diagnostic)["authority_diagnostic"]["reason"] == "exit_observation_identity_mismatch"
            )
        )
        assert load_task(cfg, task.task_id).state["projection"] == "running"
        assert read_json(observation) == value
        assert marker.read_text() == "1"
    finally:
        _cleanup(runtime, process)


def test_li08_missing_exit_observation_is_diagnosed(tmp_path: Path) -> None:
    cfg, runtime, task, marker, process = _start_case(tmp_path, seconds=2)
    try:
        _wait_running(cfg, task.task_id, marker)
        stop_machine_agent(runtime)
        attempt_id = load_task(cfg, task.task_id).attempt_control["current_attempt_id"]
        binding = runtime.load_registry()[1][0]
        paths = local_paths(runtime.project_paths(binding.project_id)["root"])
        observation = paths["observations"] / f"{attempt_id}.json"
        _wait_for(observation.exists)
        observation.rename(tmp_path / "withheld-observation.json")
        restarted = start_machine_agent(runtime, available_gpus=[0], loop_interval=0.1)
        diagnostic = paths["authority_diagnostics"] / observation.name
        _wait_for(
            lambda: (
                diagnostic.exists()
                and read_json(diagnostic)["authority_diagnostic"]["reason"] == "exit_observation_missing"
            )
        )
        assert load_task(cfg, task.task_id).state["projection"] != "succeeded"
        assert (paths["registrations"] / observation.name).exists()
        assert marker.read_text() == "1"
        stop_machine_agent(runtime)
        restarted.wait(timeout=5)
    finally:
        _cleanup(runtime, process)


def test_li08_superseded_offline_attempt_preserves_evidence(tmp_path: Path) -> None:
    from qqtools.plugins.qexp.commands.task import retry

    cfg, runtime, task, marker, process = _start_case(tmp_path, seconds=2)
    try:
        _wait_running(cfg, task.task_id, marker)
        stop_machine_agent(runtime)
        claim = load_task(cfg, task.task_id).claim_control["active_claim"]
        binding = runtime.load_registry()[1][0]
        paths = local_paths(runtime.project_paths(binding.project_id)["root"])
        observation = paths["observations"] / f"{claim['attempt_id']}.json"
        _wait_for(observation.exists)
        persist_legacy_orphan(cfg, task.task_id)
        retry(cfg, task.task_id)
        runtime.set_enabled(binding.project_id, False)
        expected = load_task(cfg, task.task_id).to_dict()
        restarted = start_machine_agent(runtime, available_gpus=[0], loop_interval=0.1)
        diagnostic = paths["authority_diagnostics"] / observation.name
        _wait_for(
            lambda: (
                diagnostic.exists()
                and read_json(diagnostic)["authority_diagnostic"]["reason"] == "attempt_authority_superseded"
            )
        )
        assert load_task(cfg, task.task_id).to_dict() == expected
        assert observation.exists()
        assert marker.read_text() == "1"
        stop_machine_agent(runtime)
        restarted.wait(timeout=5)
    finally:
        _cleanup(runtime, process)


def test_li08_cancellation_while_agent_offline_is_honored(tmp_path: Path) -> None:
    from qqtools.plugins.qexp.commands.task import cancel

    cfg, runtime, task, marker, process = _start_case(tmp_path, should_wait=True)
    try:
        _wait_running(cfg, task.task_id, marker)
        attempt_id = load_task(cfg, task.task_id).attempt_control["current_attempt_id"]
        stop_machine_agent(runtime)
        cancel(cfg, task.task_id, reservation_runtime_root=runtime.root)
        restarted = start_machine_agent(runtime, available_gpus=[0], loop_interval=0.1)
        wait_all(
            {
                "task:cancelled": lambda: load_task(cfg, task.task_id).state["projection"] == "cancelled",
                "reservation:released": lambda: not active_reservations(runtime.root),
            }
        )
        assert load_task(cfg, task.task_id).state["projection"] == "cancelled"
        attempt = read_json(attempt_path(cfg.shared_root, task.task_id, 1))["attempt"]
        assert attempt["attempt_id"] == attempt_id
        assert marker.read_text() == "1"
        stop_machine_agent(runtime)
        restarted.wait(timeout=5)
    finally:
        marker.with_suffix(".finish").touch()
        _cleanup(runtime, process)


def test_li09_public_process_entrypoint_uses_real_python_runner(tmp_path: Path) -> None:
    cfg, runtime, task, marker, process = _start_case(tmp_path, seconds=0.2)
    try:
        terminal = _wait_terminal(cfg, task.task_id)
        assert terminal.state["projection"] == "succeeded"
        assert marker.read_text(encoding="utf-8") == "1"
    finally:
        _cleanup(runtime, process)


@pytest.mark.parametrize("global_mode", ["on_demand", "daemon"])
def test_global_idle_policy_uses_machine_global_config(tmp_path: Path, global_mode: str) -> None:
    runtime = _initialized_runtime(tmp_path / "machine", agent_mode=global_mode)
    configs = []
    for index, mode in enumerate(("daemon", "on_demand")):
        cfg = init_shared_root(
            tmp_path / str(index) / ".qexp", "gpu-1", agent_mode=mode, runtime_root=tmp_path / f"legacy-{index}"
        )
        configs.append(cfg)
        _register_project(runtime, cfg)
    script = """
import runpy
from pathlib import Path
from qqtools.plugins.qexp.agent import lifecycle
from tests.helpers.qexp.worker_diagnostics import trace_project_io_requests
def reject_direct_stop_publication(*args, **kwargs):
    raise AssertionError("agent cleanup performed direct shared stop publication")
lifecycle.publish_machine_stop_snapshot = reject_direct_stop_publication
real_confirm_idle_shutdown = lifecycle._confirm_idle_shutdown
def confirm_idle_shutdown(runtime, *args, **kwargs):
    guard = real_confirm_idle_shutdown(runtime, *args, **kwargs)
    if guard is not None:
        (Path(runtime.root) / "idle-confirmed").touch()
    return guard
lifecycle._confirm_idle_shutdown = confirm_idle_shutdown
trace_project_io_requests()
runpy.run_module("qqtools.plugins.qexp.agent.process", run_name="__main__")
"""
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            "--machine-runtime-root",
            str(runtime.root),
            "--loop-interval",
            "0.1",
            "--available-gpus",
            "0",
        ],
        start_new_session=True,
    )
    try:
        if global_mode == "on_demand":
            _wait_for(
                lambda: process.poll() is not None or (runtime.root / "idle-confirmed").exists(),
                timeout=30,
                description="idle-policy cold-start quiescence",
            )
            assert process.wait(timeout=10) == 0
            assert not get_machine_agent_status(runtime)["is_running"]
            assert all(
                read_json(machine_state_path(cfg, "agent.json"))["agent"]["observed_state"] == "stopped"
                for cfg in configs
            )
        else:
            _wait_for(lambda: get_machine_agent_status(runtime)["is_running"])
            with pytest.raises(subprocess.TimeoutExpired):
                process.wait(timeout=1)
    finally:
        _cleanup(runtime, process)


@pytest.mark.parametrize("failure_point", ["terminal_observe", "terminal_publish", "worker_start"])
def test_finished_process_releases_capacity_while_publication_is_unavailable(
    tmp_path: Path, failure_point: str
) -> None:
    # Publication failure is injected after local exit evidence exists; the
    # runner does not need two seconds of unrelated work to reach that boundary.
    cfg, runtime, task, marker, process = _start_case(tmp_path, should_wait=True)
    blocked_agent = None
    try:
        _wait_running(cfg, task.task_id, marker)
        attempt_id = load_task(cfg, task.task_id).attempt_control["current_attempt_id"]
        stop_machine_agent(runtime)
        binding = runtime.load_registry()[1][0]
        paths = local_paths(runtime.project_paths(binding.project_id)["root"])
        marker.with_suffix(".finish").touch()
        observation = paths["observations"] / f"{attempt_id}.json"
        _wait_for(
            observation.exists,
            description="durable exit observation before publication outage",
            on_timeout=lambda: {
                "failure_point": failure_point,
                "attempt": load_task(cfg, task.task_id).attempt_control,
                "agent": get_machine_agent_status(runtime),
                "process_returncode": process.poll(),
                "observation": str(observation),
            },
        )
        evidence = read_json(observation)
        reached = tmp_path / "publication-blocked"
        script = """
import sys
from pathlib import Path
from qqtools.plugins.qexp.agent.lifecycle import run_machine_agent_loop
from qqtools.plugins.qexp.agent.project_io_controller import ProjectIOController
from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor
def unavailable(self, *args, **kwargs):
    Path(sys.argv[2]).touch()
    raise OSError("injected shared terminal publication outage")
if sys.argv[3] == "terminal_observe":
    ProjectIOController.advance_authority_terminal_observations = unavailable
elif sys.argv[3] == "terminal_publish":
    ProjectIOController.advance_authority_terminal_publications = unavailable
else:
    original_start = ProjectIOExecutor.start
    def start(self, request_id, *args, **kwargs):
        request = self._load_request(request_id)
        if request.operation_kind == "authority_terminal_observe":
            Path(sys.argv[2]).touch()
            return None
        return original_start(self, request_id, *args, **kwargs)
    ProjectIOExecutor.start = start
run_machine_agent_loop(sys.argv[1], loop_interval=0.1, available_gpus=[0])
"""
        blocked_agent = subprocess.Popen(
            [sys.executable, "-c", script, str(runtime.root), str(reached), failure_point], start_new_session=True
        )
        _wait_for(reached.exists)
        _wait_for(lambda: not active_reservations(runtime.root))
        assert read_json(observation) == evidence
        assert load_task(cfg, task.task_id).state["projection"] == "running"
        stop_machine_agent(runtime)
        blocked_agent.wait(timeout=5)
        restarted = start_machine_agent(runtime, available_gpus=[0], loop_interval=0.1)
        assert _wait_terminal(cfg, task.task_id).state["projection"] == "succeeded"
        assert marker.read_text() == "1"
        stop_machine_agent(runtime)
        restarted.wait(timeout=5)
    finally:
        if blocked_agent is not None and blocked_agent.poll() is None:
            blocked_agent.terminate()
            blocked_agent.wait(timeout=5)
        _cleanup(runtime, process)


def test_global_idle_waits_for_unresolved_demand(tmp_path: Path) -> None:
    runtime = _initialized_runtime(tmp_path / "machine")
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy")
    _register_project(runtime, cfg)
    reached, resolve = tmp_path / "probe", tmp_path / "resolve"
    script = """
import sys
from pathlib import Path
from qqtools.plugins.qexp.agent import dispatch_loop as machine_agent
from qqtools.plugins.qexp.agent import lifecycle
from qqtools.plugins.qexp.agent.project_io_controller import ProjectIOController
original = ProjectIOController.advance_scheduler_observations
original_quiescence = ProjectIOController.advance_scheduler_quiescence
def probe(self, *args, **kwargs):
    Path(sys.argv[2]).touch()
    if not Path(sys.argv[3]).exists():
        # Missing closed observation evidence is unresolved, never no demand.
        return {}
    return original(self, *args, **kwargs)
ProjectIOController.advance_scheduler_observations = probe
def quiescence(self, *args, **kwargs):
    # Hold both independent demand-proof paths unresolved; neither supplies
    # a false empty Source result while the barrier is closed.
    if not Path(sys.argv[3]).exists():
        return None
    return original_quiescence(self, *args, **kwargs)
ProjectIOController.advance_scheduler_quiescence = quiescence
lifecycle.run_machine_agent_loop(sys.argv[1], loop_interval=0.1, available_gpus=[0])
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(runtime.root), str(reached), str(resolve)], start_new_session=True
    )
    try:
        _wait_for(reached.exists)
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(timeout=1)
        resolve.touch()
        assert process.wait(timeout=10) == 0
    finally:
        _cleanup(runtime, process)


@pytest.mark.parametrize("kind", ["availability", "group_control", "cleanup"])
def test_pending_repair_prevents_idle_exit(tmp_path: Path, kind: str) -> None:
    from qqtools.plugins.qexp.agent.lifecycle import _machine_is_true_idle
    from qqtools.plugins.qexp.runtime.operation_store import active_operation_path

    runtime = _initialized_runtime(tmp_path / "machine")
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy")
    _register_project(runtime, cfg)
    runtime.last_cycle_had_demand = False
    assert _machine_is_true_idle(runtime, has_consumed_binding=True)
    path = active_operation_path(cfg, kind, "pending")
    atomic_replace(path, {"operation": {"state": "pending"}})
    assert not _machine_is_true_idle(runtime, has_consumed_binding=True)
    path.unlink()
    assert _machine_is_true_idle(runtime, has_consumed_binding=True)


def test_failed_binding_is_not_consumed_for_idle_exit(tmp_path: Path) -> None:
    runtime = _initialized_runtime(tmp_path / "machine")
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy")
    binding = _register_project(runtime, cfg)
    reached, allow = tmp_path / "failed", tmp_path / "allow"
    script = """
import sys
from pathlib import Path
from qqtools.plugins.qexp.agent import lifecycle
from qqtools.plugins.qexp.agent.project_io_controller import ProjectIOController
original = ProjectIOController.advance_binding_validation
def validate(self, bindings, registry_revision):
    if not Path(sys.argv[3]).exists():
        Path(sys.argv[2]).touch()
        return {}
    return original(self, bindings, registry_revision)
ProjectIOController.advance_binding_validation = validate
lifecycle.run_machine_agent_loop(sys.argv[1], loop_interval=0.1, available_gpus=[0])
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(runtime.root), str(reached), str(allow)], start_new_session=True
    )
    try:
        _wait_for(reached.exists)
        runtime.set_enabled(binding.project_id, False)
        runtime.remove_binding(binding.project_id)
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(timeout=1)
        _register_project(runtime, cfg)
        allow.touch()
        assert process.wait(timeout=10) == 0
    finally:
        _cleanup(runtime, process)


def test_global_idle_does_not_reenter_registration_wait(tmp_path: Path) -> None:
    runtime = _initialized_runtime(tmp_path / "machine")
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", agent_mode="daemon", runtime_root=tmp_path / "legacy"
    )
    binding = _register_project(runtime, cfg)
    consumed = tmp_path / "consumed"
    script = """
import sys
from pathlib import Path
from qqtools.plugins.qexp.agent import dispatch_loop as machine_agent
from qqtools.plugins.qexp.agent import lifecycle
original = machine_agent.dispatch_machine_cycle_locked
def cycle(*args, **kwargs):
    result = original(*args, **kwargs)
    if args[0].last_cycle_consumed_binding:
        Path(sys.argv[2]).touch()
    return result
machine_agent.dispatch_machine_cycle_locked = cycle
lifecycle.run_machine_agent_loop(sys.argv[1], loop_interval=0.1, available_gpus=[0])
"""
    process = subprocess.Popen([sys.executable, "-c", script, str(runtime.root), str(consumed)], start_new_session=True)
    try:
        _wait_for(consumed.exists)
        runtime.set_enabled(binding.project_id, False)
        runtime.remove_binding(binding.project_id)
        assert process.wait(timeout=10) == 0
        assert not runtime.load_registry()[1]
    finally:
        _cleanup(runtime, process)
