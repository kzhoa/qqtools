"""Real-process regression coverage for agent lifecycle independence (LI-01..LI-09)."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.machine_agent import (
    get_machine_agent_status,
    restart_machine_agent,
    start_machine_agent,
    stop_machine_agent,
)
from qqtools.plugins.qexp.machine_runtime import MachineRuntime
from qqtools.plugins.qexp.runtime.paths import attempt_path, local_paths
from qqtools.plugins.qexp.runtime.resources.reservations import active_reservations
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task
from qqtools.plugins.qexp.scheduler import expire_claim
from qqtools.plugins.qexp.tmux import is_libtmux_available

pytestmark = [pytest.mark.integration, pytest.mark.machine_lab]

CONVERGENCE_BUDGET_SECONDS = 15.0
PROCESS_START_BUDGET_SECONDS = 15.0


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


def _wait_for(predicate, *, timeout: float = CONVERGENCE_BUDGET_SECONDS) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("lifecycle condition did not converge within the declared budget")


def _start_case(tmp_path: Path, *, exit_code: int = 0, seconds: float = 1.0, should_wait: bool = False):
    project_root = tmp_path / "project"
    shared_root = project_root / ".qexp"
    cfg = init_shared_root(shared_root, "gpu-1", agent_mode="daemon", runtime_root=tmp_path / "legacy")
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.ensure_binding(shared_root, "gpu-1")
    marker = tmp_path / "launch-count"
    command = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path\nimport time\n"
            f"p=Path({str(marker)!r})\np.write_text(str(int(p.read_text()) + 1) if p.exists() else '1')\n"
            "deadline=time.monotonic()+60\n"
            f"while {should_wait!r} and not p.with_suffix('.finish').exists():\n"
            "    if time.monotonic()>deadline: raise SystemExit(99)\n"
            "    p.with_suffix('.progress').write_text(str(time.monotonic()))\n"
            "    time.sleep(0.05)\n"
            f"time.sleep({0 if should_wait else seconds!r})\nraise SystemExit({exit_code})"
        ),
    ]
    task = submit(cfg, command, working_dir=project_root)
    process = start_machine_agent(runtime, available_gpus=[0])
    return cfg, runtime, task, marker, process


def _wait_running(cfg, task_id: str, marker: Path | None = None) -> None:
    _wait_for(lambda: load_task(cfg, task_id).state["projection"] == "running", timeout=PROCESS_START_BUDGET_SECONDS)
    if marker is not None:
        _wait_for(marker.exists, timeout=5.0)


def _wait_terminal(cfg, task_id: str) -> object:
    _wait_for(
        lambda: load_task(cfg, task_id).state["projection"] in {"succeeded", "failed", "cancelled"},
    )
    return load_task(cfg, task_id)


def _cleanup(runtime: MachineRuntime, process: subprocess.Popen) -> None:
    try:
        if get_machine_agent_status(runtime)["is_running"]:
            stop_machine_agent(runtime, timeout=5.0)
    except (OSError, RuntimeError, TimeoutError):
        if process.poll() is None:
            process.kill()
    process.wait(timeout=5)


def test_li01_training_remains_live_and_is_not_relaunched(tmp_path: Path) -> None:
    cfg, runtime, task, marker, process = _start_case(tmp_path, should_wait=True)
    try:
        _wait_running(cfg, task.task_id, marker)
        attempt_id = load_task(cfg, task.task_id).attempt_control["current_attempt_id"]
        assert isinstance(attempt_id, str)
        attempt_file = attempt_path(cfg.shared_root, task.task_id, 1)
        _wait_for(lambda: read_json(attempt_file)["attempt"]["process"].get("process_group_id") is not None)
        original_process = read_json(attempt_file)["attempt"]["process"]
        _wait_for(marker.exists, timeout=5.0)
        stop_machine_agent(runtime)
        progress = marker.with_suffix(".progress")
        _wait_for(progress.exists)
        previous = progress.read_text()
        _wait_for(lambda: progress.read_text() not in {"", previous})
        assert len(active_reservations(runtime.root)) == 1
        assert marker.read_text(encoding="utf-8") == "1"
        restarted = restart_machine_agent(runtime, available_gpus=[0])
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


@pytest.mark.parametrize(("exit_code", "phase"), ((0, "succeeded"), (7, "failed")))
def test_li02_offline_completion_preserves_exit_result(tmp_path: Path, exit_code: int, phase: str) -> None:
    cfg, runtime, task, marker, process = _start_case(tmp_path, exit_code=exit_code, should_wait=True)
    try:
        _wait_running(cfg, task.task_id, marker)
        stop_machine_agent(runtime)
        marker.with_suffix(".finish").touch()
        binding = runtime.load_registry()[1][0]
        paths = local_paths(runtime.project_paths(binding.project_id)["root"])
        attempt_id = load_task(cfg, task.task_id).attempt_control["current_attempt_id"]
        _wait_for((paths["observations"] / f"{attempt_id}.json").exists)
        assert load_task(cfg, task.task_id).state["projection"] == "running"
        start = time.monotonic()
        start_machine_agent(runtime, available_gpus=[0])
        terminal = _wait_terminal(cfg, task.task_id)
        _wait_for(
            lambda: not active_reservations(runtime.root),
            timeout=max(0, CONVERGENCE_BUDGET_SECONDS - (time.monotonic() - start)),
        )
        assert time.monotonic() - start <= CONVERGENCE_BUDGET_SECONDS
        assert terminal.state["projection"] == phase
        attempt = read_json(attempt_path(cfg.shared_root, task.task_id, 1))["attempt"]
        assert attempt["result"]["exit_code"] == exit_code
        assert marker.read_text(encoding="utf-8") == "1"
        assert active_reservations(runtime.root) == []
    finally:
        _cleanup(runtime, process)


def test_li03_expired_claim_recovers_same_attempt_without_relaunch(tmp_path: Path) -> None:
    cfg, runtime, task, marker, process = _start_case(tmp_path, seconds=2.0)
    try:
        _wait_running(cfg, task.task_id, marker)
        stored = load_task(cfg, task.task_id)
        claim = stored.claim_control["active_claim"]
        claim_path = cfg.shared_root / "tasks" / f"{task.task_id}.json"
        value = read_json(claim_path)
        value["task"]["claim_control"]["active_claim"]["lease_expires_at"] = "2000-01-01T00:00:00Z"
        atomic_replace(claim_path, value)
        stop_machine_agent(runtime)
        assert expire_claim(cfg, task.task_id, claim["attempt_id"], claim["fencing_token"])
        start_machine_agent(runtime, available_gpus=[0])
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
        start_machine_agent(runtime, available_gpus=[0])
        assert _wait_terminal(cfg, task.task_id).state["projection"] == "succeeded"
    finally:
        _cleanup(runtime, process)


@pytest.mark.parametrize("is_finished", [False, True])
def test_li03_real_peer_observes_natural_lease_expiry(tmp_path: Path, is_finished: bool) -> None:
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
    runtime = MachineRuntime(tmp_path / "machine")
    binding = runtime.ensure_binding(cfg.shared_root, "gpu-1")[0]
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
    process = start_machine_agent(runtime, available_gpus=[0])
    peer = None
    observed = tmp_path / "peer-observed"
    try:
        _wait_running(cfg, task.task_id, marker)
        claim = load_task(cfg, task.task_id).claim_control["active_claim"]
        assert claim["authority_mode"] == "bounded_lease"
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
deadline=time.monotonic()+15
while time.monotonic()<deadline:
    if expire_claim(cfg,task_id,attempt_id,int(token)):
        assert load_task(cfg,task_id).state["projection"]=="blocked"
        assert claim_task(cfg,task_id,[0]) is None
        Path(observed).touch()
        break
    time.sleep(0.05)
else: raise TimeoutError("peer did not observe natural expiry")
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
        _wait_for(observed.exists)
        assert peer.wait(timeout=5) == 0
        assert marker.read_text() == "1"
        restarted = start_machine_agent(runtime, available_gpus=[0])
        if not is_finished:
            _wait_for(lambda: load_task(cfg, task.task_id).state["projection"] == "running")
            assert load_task(cfg, task.task_id).attempt_control["current_attempt_id"] == claim["attempt_id"]
            finish.touch()
        assert _wait_terminal(cfg, task.task_id).state["projection"] == "succeeded"
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
    runtime = MachineRuntime(tmp_path / "machine")
    runtime.ensure_binding(cfg.shared_root, "gpu-1")
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
from qqtools.plugins.qexp import scheduler
from qqtools.plugins.qexp.machine_agent import run_machine_agent_loop
boundary, root, reached = sys.argv[1:]
original = scheduler.authorize_launch
def authorize(*args, **kwargs):
    if boundary == "before_authorization":
        Path(reached).write_text(boundary)
        os._exit(73)
    result = original(*args, **kwargs)
    if result:
        Path(reached).write_text(boundary)
        os._exit(73)
    return result
scheduler.authorize_launch = authorize
run_machine_agent_loop(root, loop_interval=0.1, available_gpus=[0])
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, boundary, str(runtime.root), str(reached)], start_new_session=True
    )
    try:
        _wait_for(reached.exists)
        assert process.wait(timeout=5) == 73
        assert not marker.exists()
        original_attempt = load_task(cfg, task.task_id).attempt_control["current_attempt_id"]
        restarted = start_machine_agent(runtime, available_gpus=[0])
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
        _cleanup(runtime, process)


def test_li05_agent_crash_between_process_creation_and_registration(tmp_path: Path) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", agent_mode="daemon", runtime_root=tmp_path / "legacy"
    )
    runtime = MachineRuntime(tmp_path / "machine")
    binding = runtime.ensure_binding(cfg.shared_root, "gpu-1")[0]
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
from qqtools.plugins.qexp.machine_agent import run_machine_agent_loop
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
    try:
        _wait_for(lambda: reached.exists() or reached.with_suffix(".error").exists())
        assert not reached.with_suffix(".error").exists(), reached.with_suffix(".error").read_text()
        _wait_for(marker.exists)
        attempt_id = load_task(cfg, task.task_id).attempt_control["current_attempt_id"]
        assert not (paths["registrations"] / f"{attempt_id}.json").exists()
        process.kill()
        process.wait(timeout=5)
        restarted = start_machine_agent(runtime, available_gpus=[0])
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
        _cleanup(runtime, process)


@pytest.mark.parametrize("boundary", ["attempt", "task", "reservation"])
@pytest.mark.parametrize("is_orphaned", [False, True])
def test_li06_terminal_publication_is_idempotent(tmp_path: Path, boundary: str, is_orphaned: bool) -> None:
    cfg, runtime, task, marker, process = _start_case(tmp_path, seconds=2.0)
    crashing = None
    try:
        _wait_running(cfg, task.task_id, marker)
        attempt_id = load_task(cfg, task.task_id).attempt_control["current_attempt_id"]
        stop_machine_agent(runtime)
        binding = runtime.load_registry()[1][0]
        paths = local_paths(runtime.project_paths(binding.project_id)["root"])
        observation = paths["observations"] / f"{attempt_id}.json"
        _wait_for(observation.exists)
        if is_orphaned:
            task_path = cfg.shared_root / "tasks" / f"{task.task_id}.json"
            stored = read_json(task_path)
            claim = stored["task"]["claim_control"]["active_claim"]
            claim["lease_expires_at"] = "2000-01-01T00:00:00Z"
            atomic_replace(task_path, stored)
            assert expire_claim(cfg, task.task_id, attempt_id, claim["fencing_token"])
        reached = tmp_path / "crash-boundary"
        script = """
import os, sys
from pathlib import Path
from qqtools.plugins.qexp import lifecycle, authority, scheduler
from qqtools.plugins.qexp.machine_agent import run_machine_agent_loop
boundary, root, reached = sys.argv[1:]
def crash():
    Path(reached).write_text(boundary)
    os._exit(73)
original_replace = lifecycle.atomic_replace
def replace(path, value):
    original_replace(path, value)
    if boundary == "attempt" and value.get("attempt", {}).get("phase") == "succeeded":
        crash()
lifecycle.atomic_replace = replace
original_save = lifecycle.save_task
def save(cfg, task):
    original_save(cfg, task)
    if boundary == "task" and task.state["projection"] == "succeeded":
        crash()
lifecycle.save_task = save
original_release = authority.release
def release(*args, **kwargs):
    result = original_release(*args, **kwargs)
    if boundary == "reservation":
        crash()
    return result
authority.release = release
original_scheduler_release = scheduler._release_task_reservation
def scheduler_release(*args, **kwargs):
    result = original_scheduler_release(*args, **kwargs)
    if boundary == "reservation":
        crash()
    return result
scheduler._release_task_reservation = scheduler_release
run_machine_agent_loop(root, loop_interval=0.1, available_gpus=[0])
"""
        crashing = subprocess.Popen(
            [sys.executable, "-c", script, boundary, str(runtime.root), str(reached)],
            start_new_session=True,
        )
        _wait_for(reached.exists)
        assert crashing.wait(timeout=5) == 73
        assert observation.exists(), "exit evidence must survive interrupted publication"
        for _ in range(2):
            restarted = restart_machine_agent(runtime, available_gpus=[0])
            assert _wait_terminal(cfg, task.task_id).state["projection"] == "succeeded"
            _wait_for(lambda: not active_reservations(runtime.root))
            stop_machine_agent(runtime)
            restarted.wait(timeout=5)
        attempt = read_json(attempt_path(cfg.shared_root, task.task_id, 1))["attempt"]
        assert attempt["attempt_id"] == attempt_id
        assert attempt["result"]["exit_code"] == 0
        assert marker.read_text(encoding="utf-8") == "1"
    finally:
        if crashing is not None and crashing.poll() is None:
            crashing.kill()
            crashing.wait(timeout=5)
        _cleanup(runtime, process)


@pytest.mark.parametrize("project_count,is_all_offline", [(2, False), (4, False), (4, True)])
def test_li07_multiple_bindings_keep_identity_and_reservations_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, project_count: int, is_all_offline: bool
) -> None:
    gpu_ids = list(range(project_count))
    monkeypatch.setenv("QEXP_VISIBLE_GPUS", ",".join(map(str, gpu_ids)))
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    cases = []
    for name in map(str, range(project_count)):
        root = tmp_path / name
        cfg = init_shared_root(root / ".qexp", "gpu-1", agent_mode="daemon", runtime_root=root / "legacy")
        binding = runtime.ensure_binding(cfg.shared_root, "gpu-1")[0]
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
                "    time.sleep(0.05)\n"
            ),
        ]
        task = submit(cfg, command, working_dir=root)
        paths = local_paths(runtime.project_paths(binding.project_id)["root"])
        cases.append((cfg, binding, task, marker, finish, paths))
    process = start_machine_agent(runtime, available_gpus=gpu_ids)
    try:
        identities = []
        for cfg, binding, task, marker, finish, paths in cases:
            _wait_running(cfg, task.task_id, marker)
            identities.append(load_task(cfg, task.task_id).attempt_control["current_attempt_id"])
        reservations = active_reservations(runtime.root)
        assert {item["project_id"] for item in reservations} == {case[1].project_id for case in cases}
        assert len(reservations) == project_count
        stop_machine_agent(runtime)
        cases[0][4].touch()
        _wait_for((cases[0][5]["observations"] / f"{identities[0]}.json").exists)
        if is_all_offline:
            for index, case in enumerate(cases[1:], 1):
                case[4].touch()
                _wait_for((case[5]["observations"] / f"{identities[index]}.json").exists)
        else:
            assert not (cases[1][5]["observations"] / f"{identities[1]}.json").exists()
        assert len(active_reservations(runtime.root)) == project_count
        start = time.monotonic()
        restarted = start_machine_agent(runtime, available_gpus=gpu_ids)
        assert _wait_terminal(cases[0][0], cases[0][2].task_id).state["projection"] == "succeeded"
        if is_all_offline:
            for case in cases[1:]:
                assert _wait_terminal(case[0], case[2].task_id).state["projection"] == "succeeded"
            _wait_for(lambda: not active_reservations(runtime.root))
        else:
            _wait_for(lambda: len(active_reservations(runtime.root)) == project_count - 1)
            assert {item["project_id"] for item in active_reservations(runtime.root)} == {
                case[1].project_id for case in cases[1:]
            }
            for index, case in enumerate(cases[1:], 1):
                assert load_task(case[0], case[2].task_id).attempt_control["current_attempt_id"] == identities[index]
        assert time.monotonic() - start <= CONVERGENCE_BUDGET_SECONDS
        print(
            f"lifecycle workload={project_count} all_offline={is_all_offline} return_seconds={time.monotonic() - start:.3f}"
        )
        for case in cases[1:]:
            case[4].touch()
            assert _wait_terminal(case[0], case[2].task_id).state["projection"] == "succeeded"
        _wait_for(lambda: not active_reservations(runtime.root))
        for index, (cfg, binding, task, marker, finish, paths) in enumerate(cases):
            attempt = read_json(attempt_path(cfg.shared_root, task.task_id, 1))["attempt"]
            assert attempt["attempt_id"] == identities[index]
            assert attempt["result"]["exit_code"] == 0
            assert marker.read_text() == "1"
        stop_machine_agent(runtime)
        restarted.wait(timeout=5)
    finally:
        for cfg, binding, task, marker, finish, paths in cases:
            finish.touch()
        _cleanup(runtime, process)


def test_li08_mismatched_exit_evidence_is_retained_as_blocker(tmp_path: Path) -> None:
    cfg, runtime, task, marker, process = _start_case(tmp_path, seconds=0.2)
    try:
        _wait_running(cfg, task.task_id, marker)
        binding = runtime.load_registry()[1][0]
        paths = local_paths(runtime.project_paths(binding.project_id)["root"])
        stop_machine_agent(runtime)
        observation = paths["observations"] / f"{task.task_id}-attempt-1.json"
        _wait_for(observation.exists, timeout=5.0)
        value = read_json(observation)
        value["exit_observation"]["task_id"] = "other-task"
        atomic_replace(observation, value)
        start_machine_agent(runtime, available_gpus=[0])
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
        restarted = start_machine_agent(runtime, available_gpus=[0])
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
        value = read_json(cfg.shared_root / "tasks" / f"{task.task_id}.json")
        value["task"]["claim_control"]["active_claim"]["lease_expires_at"] = "2000-01-01T00:00:00Z"
        atomic_replace(cfg.shared_root / "tasks" / f"{task.task_id}.json", value)
        assert expire_claim(cfg, task.task_id, claim["attempt_id"], claim["fencing_token"])
        retry(cfg, task.task_id)
        runtime.set_enabled(binding.project_id, False)
        expected = load_task(cfg, task.task_id).to_dict()
        restarted = start_machine_agent(runtime, available_gpus=[0])
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
        restarted = start_machine_agent(runtime, available_gpus=[0])
        assert _wait_terminal(cfg, task.task_id).state["projection"] == "cancelled"
        _wait_for(lambda: not active_reservations(runtime.root))
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


@pytest.mark.parametrize("modes", [("on_demand", "on_demand"), ("on_demand", "daemon"), ("daemon", "on_demand")])
def test_global_idle_policy_considers_every_binding(tmp_path: Path, modes: tuple[str, str]) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    for index, mode in enumerate(modes):
        cfg = init_shared_root(
            tmp_path / str(index) / ".qexp", "gpu-1", agent_mode=mode, runtime_root=tmp_path / f"legacy-{index}"
        )
        runtime.ensure_binding(cfg.shared_root, "gpu-1")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "qqtools.plugins.qexp.machine_agent_process",
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
        if "daemon" not in modes:
            assert process.wait(timeout=10) == 0
            assert not get_machine_agent_status(runtime)["is_running"]
        else:
            _wait_for(lambda: get_machine_agent_status(runtime)["is_running"])
            with pytest.raises(subprocess.TimeoutExpired):
                process.wait(timeout=1)
    finally:
        _cleanup(runtime, process)


@pytest.mark.parametrize("failure_point", ["_finalize", "_materialize_registrations", "binding"])
def test_finished_process_releases_capacity_while_publication_is_unavailable(
    tmp_path: Path, failure_point: str
) -> None:
    cfg, runtime, task, marker, process = _start_case(tmp_path, seconds=2)
    blocked_agent = None
    try:
        _wait_running(cfg, task.task_id, marker)
        attempt_id = load_task(cfg, task.task_id).attempt_control["current_attempt_id"]
        stop_machine_agent(runtime)
        binding = runtime.load_registry()[1][0]
        paths = local_paths(runtime.project_paths(binding.project_id)["root"])
        observation = paths["observations"] / f"{attempt_id}.json"
        _wait_for(observation.exists)
        evidence = read_json(observation)
        reached = tmp_path / "publication-blocked"
        script = """
import sys
from pathlib import Path
from qqtools.plugins.qexp.authority import AuthoritySupervisor
from qqtools.plugins.qexp import machine_agent
from qqtools.plugins.qexp.machine_agent import run_machine_agent_loop
def unavailable(self, *args, **kwargs):
    Path(sys.argv[2]).touch()
    raise OSError("injected shared terminal publication outage")
if sys.argv[3] == "binding":
    machine_agent._binding_config = unavailable
else:
    setattr(AuthoritySupervisor, sys.argv[3], unavailable)
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
        restarted = start_machine_agent(runtime, available_gpus=[0])
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
    runtime = MachineRuntime(tmp_path / "machine")
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy")
    runtime.ensure_binding(cfg.shared_root, "gpu-1")
    reached, resolve = tmp_path / "probe", tmp_path / "resolve"
    script = """
import sys
from pathlib import Path
from qqtools.plugins.qexp import machine_agent
original = machine_agent._probe_primary_demand
def probe(*args, **kwargs):
    Path(sys.argv[2]).touch()
    if not Path(sys.argv[3]).exists():
        return machine_agent.PrimaryDemandProbe("unresolved")
    return original(*args, **kwargs)
machine_agent._probe_primary_demand = probe
machine_agent.run_machine_agent_loop(sys.argv[1], loop_interval=0.1, available_gpus=[0])
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
    from qqtools.plugins.qexp.machine_agent import _machine_is_true_idle
    from qqtools.plugins.qexp.runtime.operation_store import active_operation_path

    runtime = MachineRuntime(tmp_path / "machine")
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy")
    runtime.ensure_binding(cfg.shared_root, "gpu-1")
    runtime.last_cycle_had_demand = False
    assert _machine_is_true_idle(runtime, has_consumed_binding=True)
    path = active_operation_path(cfg, kind, "pending")
    atomic_replace(path, {"operation": {"state": "pending"}})
    assert not _machine_is_true_idle(runtime, has_consumed_binding=True)
    path.unlink()
    assert _machine_is_true_idle(runtime, has_consumed_binding=True)


def test_failed_binding_is_not_consumed_for_idle_exit(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy")
    binding = runtime.ensure_binding(cfg.shared_root, "gpu-1")[0]
    reached, allow = tmp_path / "failed", tmp_path / "allow"
    script = """
import sys
from pathlib import Path
from qqtools.plugins.qexp import machine_agent
original = machine_agent._binding_config
def config(*args):
    if not Path(sys.argv[3]).exists():
        Path(sys.argv[2]).touch()
        raise OSError("unreadable first binding")
    return original(*args)
machine_agent._binding_config = config
machine_agent.run_machine_agent_loop(sys.argv[1], loop_interval=0.1, available_gpus=[0])
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
        runtime.ensure_binding(cfg.shared_root, "gpu-1")
        allow.touch()
        assert process.wait(timeout=10) == 0
    finally:
        _cleanup(runtime, process)


def test_global_idle_does_not_reenter_registration_wait(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine")
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", agent_mode="daemon", runtime_root=tmp_path / "legacy"
    )
    binding = runtime.ensure_binding(cfg.shared_root, "gpu-1")[0]
    consumed = tmp_path / "consumed"
    script = """
import sys
from pathlib import Path
from qqtools.plugins.qexp import machine_agent
original = machine_agent.dispatch_machine_cycle_locked
def cycle(*args, **kwargs):
    result = original(*args, **kwargs)
    Path(sys.argv[2]).touch()
    return result
machine_agent.dispatch_machine_cycle_locked = cycle
machine_agent.run_machine_agent_loop(sys.argv[1], loop_interval=0.1, available_gpus=[0])
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
