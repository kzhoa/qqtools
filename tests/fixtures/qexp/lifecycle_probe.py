"""Measure four offline completions against an explicitly selected source tree."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    sys.path.insert(0, str(source))
    from tests.helpers.qexp.resources import TestResourceScope

    scope = TestResourceScope.create(args.work_root.resolve(), "four-offline-baseline")
    os.environ.update(scope.child_environment())
    os.environ["PYTHONPATH"] = str(source)
    os.environ["QEXP_VISIBLE_GPUS"] = "0,1,2,3"
    from qqtools.plugins.qexp import init_shared_root, submit
    from qqtools.plugins.qexp.agent.lifecycle import get_machine_agent_status, stop_machine_agent
    from qqtools.plugins.qexp.agent.context import MachineRuntime
    from qqtools.plugins.qexp.runtime.paths import local_paths
    from qqtools.plugins.qexp.runtime.resources.reservations import active_reservations
    from qqtools.plugins.qexp.runtime.tasks import load_task

    runtime = MachineRuntime(scope.runtime_root)
    cases = []
    agents = []

    def wait(predicate, timeout=15):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.05)
        return False

    def start():
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import sys; from qqtools.plugins.qexp.agent.lifecycle import run_machine_agent_loop; "
                "run_machine_agent_loop(sys.argv[1], available_gpus=[0,1,2,3])",
                str(runtime.root),
            ],
            start_new_session=True,
        )
        agents.append(process)
        assert wait(lambda: get_machine_agent_status(runtime)["is_running"])

    try:
        for index in range(4):
            root = scope.root / str(index)
            cfg = init_shared_root(root / ".qexp", "probe", agent_mode="daemon", runtime_root=root / "legacy")
            binding = runtime.ensure_binding(cfg.shared_root, "probe")[0]
            marker, finish = root / "count", root / "finish"
            code = (
                "from pathlib import Path\nimport time\n"
                f"p=Path({str(marker)!r})\np.write_text(str(int(p.read_text())+1) if p.exists() else '1')\n"
                "deadline=time.monotonic()+60\n"
                f"while not Path({str(finish)!r}).exists():\n"
                "    if time.monotonic()>deadline: raise SystemExit(99)\n"
                "    time.sleep(0.02)\n"
            )
            task = submit(cfg, [sys.executable, "-c", code], working_dir=root)
            paths = local_paths(runtime.project_paths(binding.project_id)["root"])
            cases.append((cfg, task, marker, finish, paths))
        start()
        assert wait(lambda: all(case[2].exists() for case in cases)), "verified launch timeout"
        stop_machine_agent(runtime)
        for cfg, task, marker, finish, paths in cases:
            finish.touch()
        assert wait(lambda: all(list(case[4]["observations"].glob("*.json")) for case in cases))
        evidence_bytes = sum(
            path.stat().st_size
            for case in cases
            for name in ("observations", "registrations", "launch_intents", "processes")
            for path in case[4][name].glob("*.json")
        )
        started = time.monotonic()
        start()
        is_converged = wait(
            lambda: (
                all(load_task(case[0], case[1].task_id).state["projection"] == "succeeded" for case in cases)
                and not active_reservations(runtime.root)
            ),
            max(0, 15 - (time.monotonic() - started)),
        )
        result = {
            "source_root": str(source),
            "workload": 4,
            "budget_seconds": 15,
            "is_converged": is_converged,
            "elapsed_seconds": time.monotonic() - started,
            "evidence_bytes": evidence_bytes,
            "launch_counts": [int(case[2].read_text()) for case in cases],
            "task_states": [load_task(case[0], case[1].task_id).state for case in cases],
        }
        output = scope.root / "result.json"
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"artifact": str(output), **result}))
    finally:
        for case in cases:
            case[3].touch()
        if get_machine_agent_status(runtime)["is_running"]:
            stop_machine_agent(runtime)
        for process in agents:
            process.wait(timeout=5)
        for path in scope.tmux_root.rglob("*"):
            if path.is_socket():
                subprocess.run(["tmux", "-S", str(path), "kill-server"], capture_output=True, timeout=5)


if __name__ == "__main__":
    main()
