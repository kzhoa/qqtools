from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import pytest
from qexp_e2e import ensure_site_packages_import, is_machine_agent_running, jrun, make_env, run, stop_agent, wait_for

from qqtools.plugins.qexp.tmux import is_libtmux_available

pytestmark = [pytest.mark.e2e, pytest.mark.host_exclusive]


def test_installed_cli_stop_offline_completion_start(tmp_path: Path) -> None:
    if sys.platform != "linux" or shutil.which("tmux") is None or not is_libtmux_available():
        pytest.fail("LI-09 unverified: installed lifecycle E2E requires Linux, tmux, and libtmux")
    base = tmp_path / "installed-lifecycle"
    shared_root = base / ".qexp"
    runtime_root = base / "runtime"
    machine_runtime_root = base / "machine-runtime"
    marker = base / "launch-count"
    release = base / "allow-exit"
    finished = base / "training-finished"
    env = make_env(base)
    common = [
        "qexp",
        "--shared-root",
        str(shared_root),
        "--machine",
        "gpu-1",
        "--runtime-root",
        str(runtime_root),
        "--machine-runtime-root",
        str(machine_runtime_root),
    ]
    try:
        run([*common, "init", "--agent-mode", "daemon"], env=env)
        run([*common, "agent", "start"], env=env)
        wait_for(lambda: is_machine_agent_running(common, env=env), timeout=10, label="agent startup")
        command = [
            sys.executable,
            "-c",
            (
                "from pathlib import Path\nimport time\n"
                f"p=Path({str(marker)!r})\n"
                "p.write_text(str(int(p.read_text()) + 1) if p.exists() else '1')\n"
                "deadline=time.monotonic()+60\n"
                f"while not Path({str(release)!r}).exists():\n"
                "    if time.monotonic()>deadline: raise SystemExit(99)\n"
                "    time.sleep(0.05)\n"
                "print('offline lifecycle result', flush=True)\n"
                f"Path({str(finished)!r}).touch()\n"
            ),
        ]
        submit = run([*common, "submit", "--name", "installed-lifecycle", "--", *command], env=env)
        task_id = submit.stdout.strip()
        wait_for(
            lambda: jrun([*common, "task", "show", task_id], env=env)["task"]["state"]["projection"] == "running",
            timeout=30,
            label="installed task launch",
        )
        wait_for(marker.exists, timeout=10, label="training command started")
        original = jrun([*common, "task", "show", task_id], env=env)["task"]
        attempt_id = original["attempt_control"]["current_attempt_id"]
        run([*common, "agent", "stop"], env=env)
        assert not is_machine_agent_running(common, env=env)
        release.touch()
        wait_for(finished.exists, timeout=10, label="offline training completion")

        def exit_observations():
            return list(machine_runtime_root.glob(f"projects/*/process-observations/{attempt_id}.json"))

        wait_for(lambda: len(exit_observations()) == 1, timeout=10, label="runner durable exit observation")
        observation = json.loads(exit_observations()[0].read_text())["exit_observation"]
        assert observation["attempt_id"] == attempt_id
        assert observation["observed_exit_code"] == 0
        assert not is_machine_agent_running(common, env=env)
        started_at = time.monotonic()
        run([*common, "agent", "start"], env=env)
        wait_for(
            lambda: (
                jrun([*common, "task", "show", task_id], env=env)["task"]["state"]["projection"]
                in {"succeeded", "failed", "cancelled"}
            ),
            timeout=15,
            label="installed lifecycle convergence",
        )
        assert time.monotonic() - started_at <= 15
        terminal = jrun([*common, "task", "show", task_id], env=env)["task"]
        assert terminal["state"]["projection"] == "succeeded"
        assert terminal["claim_control"]["active_claim"] is None
        assert terminal["attempt_control"]["next_attempt_number"] == 2
        attempt = json.loads((shared_root / "attempts" / task_id / "1.json").read_text())["attempt"]
        assert attempt["attempt_id"] == attempt_id
        assert attempt["result"]["exit_code"] == 0
        archive = shared_root / "claims" / "archive" / task_id / f"{attempt['current_fencing_token']}.json"
        wait_for(archive.exists, timeout=5, label="claim archive")
        wait_for(
            lambda: not list((machine_runtime_root / "reservations" / "active").glob("*.json")),
            timeout=5,
            label="reservation release",
        )
        assert "offline lifecycle result" in run([*common, "logs", task_id], env=env).stdout
        assert marker.read_text(encoding="utf-8") == "1"
        assert "site-packages" in ensure_site_packages_import()
    finally:
        release.touch()
        stop_agent(common, env=env)
