"""Released-agent to current-agent rollout with a live released workload."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

_TARGET_AGENT = """
import json
import sys
from inspect import signature
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.lifecycle import start_machine_agent, stop_machine_agent
runtime = MachineRuntime(sys.argv[1])
if sys.argv[2] == "start":
    options = {"loop_interval": 0.05} if "loop_interval" in signature(start_machine_agent).parameters else {}
    process = start_machine_agent(runtime, available_gpus=[0], **options)
    print(json.dumps({"pid": process.pid}))
else:
    print(json.dumps({"stopped": stop_machine_agent(runtime, timeout=5)}))
"""


def _wait(predicate, stage: str) -> None:
    deadline = time.monotonic() + 15
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"namespace rollout did not converge: {stage}")
        time.sleep(0.02)


def _target_agent(runtime_root: Path, action: str, source: Path, root: Path) -> dict:
    temporary = runtime_root / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, "-c", _TARGET_AGENT, str(runtime_root), action],
        env=dict(os.environ, PYTHONPATH=str(source), TMPDIR=str(temporary), TMP=str(temporary), TEMP=str(temporary)),
        capture_output=True,
        text=True,
        timeout=10,
    )
    with (root / "target-agent.log").open("a") as stream:
        stream.write(f"{action}: {result.returncode}\n{result.stdout}\n{result.stderr}\n")
    result.check_returncode()
    return json.loads(result.stdout)


def probe_live_namespace_upgrade(cfg, root: Path, module: Path, workload: str, *, should_pause: bool = False) -> dict:
    """Restart actual machine agents while retaining one released runner identity."""
    from qqtools.plugins.qexp import init_shared_root, submit
    from qqtools.plugins.qexp.agent.context import MachineRuntime
    from qqtools.plugins.qexp.agent.lifecycle import start_machine_agent, stop_machine_agent
    from qqtools.plugins.qexp.commands.group import create_group
    from qqtools.plugins.qexp.runtime.paths import attempt_path, local_paths, task_path
    from qqtools.plugins.qexp.runtime.store import read_json
    from qqtools.plugins.qexp.runtime.tasks import load_task

    target_source = Path(__file__).resolve().parents[2] / "src"
    runtime = MachineRuntime(root / "machine")
    binding = runtime.ensure_binding(cfg.shared_root, cfg.machine_name)[0]
    peer_runtime = None
    if should_pause:
        peer_cfg = init_shared_root(cfg.shared_root, "peer-machine", runtime_root=root / "peer-local")
        peer_runtime = MachineRuntime(root / "peer-machine")
        peer_runtime.ensure_binding(peer_cfg.shared_root, peer_cfg.machine_name)
    create_group(cfg, "rollout")
    task = submit(cfg, [sys.executable, "-c", workload.replace("+ 20", "+ 60"), str(root)], group="rollout")
    paths = local_paths(runtime.project_paths(binding.project_id)["root"])
    process = None
    try:
        if peer_runtime is not None:
            _target_agent(peer_runtime.root, "start", module.parents[3], root)
        process = start_machine_agent(runtime, available_gpus=[0])
        _wait(
            lambda: (root / "starts.txt").exists() and load_task(cfg, task.task_id).state["projection"] == "running",
            "released workload launch",
        )
        starts = (root / "starts.txt").read_text().splitlines()
        assert len(starts) == 1 and starts[0].isdigit(), starts
        attempt_file = attempt_path(cfg.shared_root, task.task_id, 1)
        _wait(
            lambda: read_json(attempt_file)["attempt"]["process"].get("process_group_id") is not None,
            "released process registration",
        )
        original = read_json(attempt_file)["attempt"]
        attempt_id = original["attempt_id"]
        original_identity = dict(original["process"])
        assert original_identity.get("process_group_id") is not None
        stop_machine_agent(runtime, timeout=5)
        assert process.wait(timeout=5) == 0
        os.kill(int(starts[0]), 0)
        assert not (root / "finished").exists()
        _target_agent(runtime.root, "start", target_source, root)
        journal = cfg.shared_root / "schema/group-authority.json"
        if peer_runtime is not None:
            registration = cfg.shared_root / "machines" / cfg.machine_name / "registration.json"
            _wait(lambda: read_json(registration)["registration"]["version"] == 2, "first participant preparation")
            # Keep a real released participant unupgraded across multiple target
            # enrollment passes. Neither its absence nor elapsed time is retirement.
            pause_until = time.monotonic() + 2.5
            while time.monotonic() < pause_until:
                assert not journal.exists()
                assert (cfg.shared_root / "groups/rollout.json").is_file()
                assert (
                    read_json(peer_cfg.shared_root / "machines/peer-machine/registration.json")["registration"][
                        "version"
                    ]
                    == 1
                )
                assert (root / "starts.txt").read_text().splitlines() == starts
                assert not (root / "finished").exists()
                os.kill(int(starts[0]), 0)
                time.sleep(0.05)
            # An idle released peer may have exited on demand; its durable
            # registration still blocks activation until explicitly upgraded.
            _target_agent(peer_runtime.root, "stop", module.parents[3], root)
            _target_agent(peer_runtime.root, "start", target_source, root)
        _wait(
            lambda: journal.exists() and read_json(journal)["group_authority"]["phase"] == "completed",
            "automatic namespace activation with running released workload",
        )
        os.kill(int(starts[0]), 0)
        assert not (root / "finished").exists()
        assert (root / "starts.txt").read_text().splitlines() == starts
        assert read_json(cfg.shared_root / "indexes/ready/state.json")["ready_index"]["writer_capability"] == "ready-v2"
        assert (cfg.shared_root / "groups-v2/rollout.json").is_file()
        _target_agent(runtime.root, "stop", target_source, root)
        (root / "release").touch()
        observation = paths["observations"] / f"{attempt_id}.json"
        _wait(lambda: observation.exists(), "released runner exit while current agent is offline")
        assert read_json(observation)["exit_observation"]["observed_exit_code"] == 0
        assert (root / "finished").read_text() == starts[0]
        _target_agent(runtime.root, "start", target_source, root)
        # The released typed reader correctly rejects the activated capabilities.
        # Inspect persisted outcome without asking that old reader to admit writes.
        task_file = task_path(cfg.shared_root, task.task_id)
        _wait(lambda: read_json(task_file)["task"]["state"]["projection"] == "succeeded", "same Attempt reconciliation")
        terminal = read_json(task_file)["task"]
        after = read_json(attempt_file)["attempt"]
        assert terminal["attempt_control"]["next_attempt_number"] == 2
        assert after["attempt_id"] == attempt_id and after["result"]["exit_code"] == 0
        for key in ("wrapper_pid", "wrapper_start_time_ticks", "process_group_id", "process_group_start_time_ticks"):
            assert after["process"][key] == original_identity[key]
        assert (root / "starts.txt").read_text().splitlines() == starts
        return {
            "case": "paused_namespace_upgrade" if should_pause else "live_namespace_upgrade",
            "source": str(module),
            "task_changed": True,
            "continuity": {
                "launch_count": 1,
                "workload_pid": int(starts[0]),
                "attempt_id": attempt_id,
                "same_process_identity": True,
                "automatic_namespace_activation": True,
                "paused_rollout": should_pause,
                "logical_participants": 2 if should_pause else 1,
                "physical_hosts": 1,
                "offline_exit_code": 0,
                "terminal_state": terminal["state"]["projection"],
            },
        }
    finally:
        (root / "release").touch()
        try:
            _target_agent(runtime.root, "stop", target_source, root)
        finally:
            try:
                if peer_runtime is not None:
                    _target_agent(peer_runtime.root, "stop", target_source, root)
            finally:
                try:
                    if process is not None:
                        process.wait(timeout=5)
                finally:
                    # Every socket is under this probe's isolated TMUX_TMPDIR.
                    for path in Path(os.environ["TMUX_TMPDIR"]).rglob("*"):
                        if path.is_socket():
                            subprocess.run(
                                ["tmux", "-S", str(path), "kill-server"], capture_output=True, timeout=5, check=False
                            )
