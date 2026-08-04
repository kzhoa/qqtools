from __future__ import annotations

import json
import subprocess
from pathlib import Path

from qexp_e2e import (
    TASK_TERMINAL_TIMEOUT_SECONDS,
    ensure_site_packages_import,
    jrun,
    make_env,
    make_layout,
    run,
    stop_agent,
    wait_for,
)


def test_installed_wheel_cleanup_and_doctor_flow(tmp_path):
    base, shared_root, runtime_root = make_layout(tmp_path / "qexp-clean")
    env = make_env(base)
    imported_from = ensure_site_packages_import()
    common = [
        "qexp",
        "--shared-root",
        str(shared_root),
        "--machine",
        "gpu-1",
        "--runtime-root",
        str(runtime_root),
    ]
    pid_path = Path(runtime_root) / "agent" / "agent.pid"

    try:
        run([*common, "init", "--agent-mode", "daemon"], env=env)
        started = subprocess.run(
            [*common, "agent", "start"],
            env=env,
            text=True,
            capture_output=True,
        )
        assert started.returncode == 0, started.stderr
        wait_for(pid_path.exists, timeout=10, label="background agent pid file")

        submit = run(
            [
                *common,
                "submit",
                "--name",
                "failing-job",
                "--",
                "python",
                "-c",
                "import sys; print('fail ok'); sys.exit(7)",
            ],
            env=env,
        )
        task_id = submit.stdout.strip()

        def is_done() -> bool:
            task = jrun([*common, "task", "show", task_id], env=env)
            return task["task"]["state"]["projection"] in {"succeeded", "failed", "cancelled"}

        wait_for(is_done, timeout=TASK_TERMINAL_TIMEOUT_SECONDS, label="failing task terminal state")
        task = jrun([*common, "task", "show", task_id], env=env)
        logs = run([*common, "logs", task_id], env=env).stdout
        process_dir = Path(runtime_root) / "processes"

        def has_exited_process() -> bool:
            return any(
                json.loads(path.read_text(encoding="utf-8"))["process"].get("observed_state") == "exited"
                for path in process_dir.glob("*.json")
            )

        wait_for(has_exited_process, timeout=10, label="agent process-exit reconciliation")
        clean = jrun([*common, "clean", "--task-id", task_id], env=env)
        verify = jrun([*common, "doctor", "verify"], env=env)

        assert "site-packages" in imported_from
        assert task["task"]["state"]["projection"] == "failed"
        assert "fail ok" in logs
        assert task_id in clean["candidates"]
        assert clean["removed"]
        assert verify["healthy"] is True
    finally:
        stop_agent(common, env=env, pid_path=pid_path)
