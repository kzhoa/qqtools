"""Real-runner authority workload with optional reproducible comparison profiles.

Load -p tests.helpers.qexp.authority_measurement to enable comparison options.
--authority-workload-profile accepts a JSON object with bindings (1..16),
attempts_per_binding (1..4), local_history (0..1024), shared_history (0..1024), and hold_seconds (0..30).
--authority-workload-output optionally retains the raw JSON report.
The default preserves the existing four-Attempt, 15-second convergence budget.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.lifecycle import stop_machine_agent
from qqtools.plugins.qexp.runtime.paths import attempt_path, local_paths
from qqtools.plugins.qexp.runtime.resources.reservations import active_reservations
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task
from qqtools.plugins.qexp.scheduler import cancel_task
from qqtools.plugins.qexp.tmux import is_libtmux_available
from tests.helpers.qexp.lifecycle import wait_all, wait_until
from tests.helpers.qexp.resources import _has_active_unix_socket
from tests.helpers.qexp.startup_profile import collect_startup_profiles

pytestmark = [pytest.mark.integration, pytest.mark.machine_lab]


@pytest.fixture(autouse=True)
def _checkout_environment(monkeypatch, checkout_subprocess_env):
    monkeypatch.setenv("PYTHONPATH", checkout_subprocess_env["PYTHONPATH"])


@pytest.fixture(autouse=True)
def _inherited_profile_environment(tmp_path, monkeypatch):
    """Neither opt-in mode may reuse a profile destination from its parent."""
    inherited = tmp_path / "inherited-profile"
    monkeypatch.setenv("QEXP_TEST_STARTUP_PROFILE_ROOT", str(inherited))
    yield
    assert not inherited.exists(), "workload reused an inherited profile destination"


@pytest.fixture(autouse=True)
def _owned_tmux_cleanup(qexp_resource_scope):
    yield
    for path in qexp_resource_scope.tmux_root.rglob("*"):
        if path.is_socket():
            subprocess.run(["tmux", "-S", str(path), "kill-server"], capture_output=True, timeout=5, check=False)
    wait_until("tmux closed", lambda: not _has_active_unix_socket(qexp_resource_scope.tmux_root))


def test_authority_workload(tmp_path, monkeypatch, request):
    assert sys.platform == "linux" and shutil.which("tmux") and is_libtmux_available()
    profile = {"bindings": 4, "attempts_per_binding": 1, "local_history": 0, "shared_history": 0, "hold_seconds": 0}
    overrides = json.loads(request.config.getoption("--authority-workload-profile", default="{}"))
    assert isinstance(overrides, dict) and overrides.keys() <= profile.keys()
    profile.update(overrides)
    for key, upper in (
        ("bindings", 16),
        ("attempts_per_binding", 4),
        ("local_history", 1024),
        ("shared_history", 1024),
        ("hold_seconds", 30),
    ):
        assert (
            type(profile[key]) is int
            and (1 if key in {"bindings", "attempts_per_binding"} else 0) <= profile[key] <= upper
        )
    gpu_count = profile["bindings"] * profile["attempts_per_binding"]
    monkeypatch.setenv("QEXP_VISIBLE_GPUS", ",".join(map(str, range(gpu_count))))
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    cases = []
    for project in range(profile["bindings"]):
        root = tmp_path / f"project-{project}"
        cfg = init_shared_root(root / ".qexp", "gpu-1", agent_mode="daemon", runtime_root=root / "legacy")
        binding = runtime.ensure_binding(cfg.shared_root, "gpu-1")[0]
        paths = local_paths(runtime.project_paths(binding.project_id)["root"])
        for index in range(profile["local_history"]):
            atomic_replace(
                paths["registrations"] / f"retained-{index}.json", {"process_registration": {"protocol_version": 0}}
            )
        for index in range(profile["shared_history"]):
            history = submit(cfg, [sys.executable, "-c", "pass"], working_dir=root)
            cancel_task(cfg, history.task_id, reservation_runtime_root=runtime.root)
        for index in range(profile["attempts_per_binding"]):
            marker, finish = root / f"started-{index}", root / f"finish-{index}"
            command = [
                sys.executable,
                "-c",
                "from pathlib import Path\nimport time\n"
                f"marker=Path({str(marker)!r})\nfinish=Path({str(finish)!r})\n"
                "marker.write_text(str(int(marker.read_text())+1) if marker.exists() else '1')\n"
                "deadline=time.monotonic()+60\n"
                "while not finish.exists():\n"
                "    if time.monotonic()>deadline: raise SystemExit(99)\n"
                "    time.sleep(0.01)\n",
            ]
            task = submit(cfg, command, working_dir=root)
            cases.append((cfg, task, marker, finish))
    repo = Path(__file__).resolve().parents[3]
    sources = sorted((repo / "src" / "qqtools" / "plugins" / "qexp").rglob("*.py"))
    sources.append(repo / "src" / "qqtools" / "__init__.py")
    report = {
        "profile": profile,
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "source_fingerprints": {
            str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        },
        "python": platform.python_version(),
        "kernel": platform.release(),
        "storage": "isolated local test temporary filesystem; fsync enabled",
        "instrumentation_sha256": hashlib.sha256(
            (repo / "tests/helpers/qexp/authority_measurement.py").read_bytes()
        ).hexdigest(),
        "workload_test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "startup_profiler_sha256": hashlib.sha256(
            (repo / "tests/helpers/qexp/startup_profile.py").read_bytes()
        ).hexdigest(),
        "task_count": len(cases),
        "poll_interval_seconds": 0.01,
        "stage_seen_seconds": {},
        "outcome": "incomplete",
    }
    measurements = tmp_path / "agent-measurements.json"
    startup_profiles = tmp_path / "startup-profiles"
    should_profile_startup = request.config.getoption("--authority-profile-startup", default=False)
    if should_profile_startup:
        monkeypatch.setenv("QEXP_TEST_STARTUP_PROFILE_ROOT", str(startup_profiles))
    else:
        monkeypatch.delenv("QEXP_TEST_STARTUP_PROFILE_ROOT", raising=False)
    started = time.monotonic()
    report["agent_requested_monotonic"] = started
    agent = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tests.helpers.qexp.authority_measurement",
            str(runtime.root),
            str(measurements),
            str(gpu_count),
        ],
        cwd=repo,
        start_new_session=True,
    )

    def running(case):
        cfg, task, marker, _finish = case
        path = attempt_path(cfg.shared_root, task.task_id, 1)
        if not path.exists() or not marker.exists() or read_json(path)["attempt"]["phase"] != "running":
            return False
        report["stage_seen_seconds"][f"running:{task.task_id}"] = time.monotonic() - started
        return True

    def terminal(case):
        cfg, task, _marker, _finish = case
        if load_task(cfg, task.task_id).state["projection"] != "succeeded":
            return False
        report["stage_seen_seconds"][f"terminal:{task.task_id}"] = time.monotonic() - started
        return True

    try:
        wait_all({str(index): lambda case=case: running(case) for index, case in enumerate(cases)}, stage="running")
        assert len(active_reservations(runtime.root)) == len(cases)
        report["all_running_seconds"] = time.monotonic() - started
        initial_attempts = {
            task.task_id: read_json(attempt_path(cfg.shared_root, task.task_id, 1))["attempt"]
            for cfg, task, _marker, _finish in cases
        }
        steady_started = time.monotonic()
        report["steady_started_monotonic"] = steady_started
        report["steady_observations"] = []
        while time.monotonic() - steady_started < profile["hold_seconds"]:
            assert agent.poll() is None, "machine agent exited during steady service"
            for cfg, task, marker, _finish in cases:
                attempt = read_json(attempt_path(cfg.shared_root, task.task_id, 1))["attempt"]
                initial = initial_attempts[task.task_id]
                assert attempt["phase"] == "running"
                assert attempt["current_fencing_token"] == initial["current_fencing_token"]
                assert marker.read_text() == "1"
                report["steady_observations"].append(
                    {"task_id": task.task_id, "seconds": time.monotonic() - started, "lease": attempt.get("lease")}
                )
            time.sleep(min(1.0, max(0.0, profile["hold_seconds"] - (time.monotonic() - steady_started))))
        report["steady_finished_monotonic"] = time.monotonic()
        report["finish_requested_seconds"] = time.monotonic() - started
        for _cfg, _task, _marker, finish in cases:
            finish.touch()
        wait_all({str(index): lambda case=case: terminal(case) for index, case in enumerate(cases)}, stage="terminal")
        wait_until("accounting", lambda: not active_reservations(runtime.root), timeout=15)
        report["accounting_seen_seconds"] = time.monotonic() - started
        for cfg, task, marker, _finish in cases:
            attempt = read_json(attempt_path(cfg.shared_root, task.task_id, 1))["attempt"]
            assert attempt["result"]["exit_code"] == 0
            assert marker.read_text() == "1"
        report["outcome"] = "passed"
    except Exception as exc:
        report["outcome"] = "failed"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        # Preserve the failure boundary before cleanup requests process exit and
        # before the resource fixture removes its isolated temporary roots.
        report["failure_tasks"] = []
        for cfg, task, marker, _finish in cases:
            snapshot = {"task_id": task.task_id, "project": cfg.shared_root.parent.name, "started": marker.exists()}
            try:
                snapshot["task"] = read_json(cfg.shared_root / "tasks" / f"{task.task_id}.json")
                attempt_file = attempt_path(cfg.shared_root, task.task_id, 1)
                snapshot["attempt"] = read_json(attempt_file) if attempt_file.exists() else None
            except (OSError, ValueError) as failure:
                snapshot["observation_error"] = {"type": type(failure).__name__, "message": str(failure)}
            report["failure_tasks"].append(snapshot)
        raise
    finally:
        for _cfg, _task, _marker, finish in cases:
            finish.touch()
        try:
            stop_machine_agent(runtime, timeout=5)
        finally:
            if agent.poll() is None:
                try:
                    agent.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    agent.kill()
                    agent.wait(timeout=5)
            if measurements.exists():
                report["agent"] = json.loads(measurements.read_text())
            has_invalid_profile = False
            if should_profile_startup:
                report.update(
                    collect_startup_profiles(
                        startup_profiles,
                        expected_source=repo / "src/qqtools/plugins/qexp/runner.py",
                        expected_home=request.getfixturevalue("qexp_resource_scope").home_root,
                        expected_count=len(cases),
                    )
                )
                report["workload_outcome"] = report["outcome"]
                has_invalid_profile = report["outcome"] == "passed" and not report["startup_environment_is_valid"]
                if has_invalid_profile:
                    report["outcome"] = "invalid_profile"
            snapshot = runtime.root / "authority_control_plane.json"
            if snapshot.exists():
                report["control_plane_snapshot"] = read_json(snapshot)
            output = request.config.getoption("--authority-workload-output", default=None)
            if output:
                Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
            assert not has_invalid_profile, "runner escaped the profiled checkout or isolated HOME"
