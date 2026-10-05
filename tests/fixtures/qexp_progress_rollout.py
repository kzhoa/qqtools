"""Executable released-source fixture for scoped-progress rolling-upgrade qualification.

Run with Python 3.13, --ref v1.3.22 --output /tmp/unique-short-directory.
Each actor imports only its selected source tree; no installed packages are changed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

WORKLOAD = r"""
import json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
from qqtools.qexp import progress
root = Path(sys.argv[1])
root.mkdir(exist_ok=True)
(root / "started.json").write_text(json.dumps({"pid": os.getpid(), "v3": os.environ.get("QEXP_PROGRESS_V3_PATH"), "source": progress.__file__}))
deadline = time.monotonic() + 180
current = 0
while not (root / "release").exists():
    current += 1
    options = {"overall": progress.Counter(current=current, unit="step", label="Training")} if hasattr(progress, "Counter") else {}
    progress.update(stage="work", current=current, unit="batch", metrics={"loss": 0.25}, **options)
    progress.flush(timeout=0.1)
    if time.monotonic() > deadline:
        raise TimeoutError("qualification did not release its workload")
    time.sleep(0.1)
(root / "finished").write_text(str(os.getpid()))
"""


def actor(root: Path, action: str, task_id: str | None) -> dict:
    from qqtools.plugins.qexp import init_shared_root, observer
    from qqtools.plugins.qexp.agent.context import MachineRuntime
    from qqtools.plugins.qexp.agent.lifecycle import start_machine_agent, stop_machine_agent
    from qqtools.plugins.qexp.agent.setup import initialize_machine, register_projects
    from qqtools.plugins.qexp.commands.cleanup import clean
    from qqtools.plugins.qexp.config_types import RootConfig
    from qqtools.plugins.qexp.lease import LeasePolicy, save_lease_policy
    from qqtools.plugins.qexp.progress_policy import set_progress_policy
    from qqtools.plugins.qexp.runtime.submission import submit_specs

    runtime = MachineRuntime(root / "machine")
    if action in {"setup", "setup-short"}:
        cfg = init_shared_root(root / ".qexp", "rollout-machine", runtime_root=root / "local")
        if not runtime.has_identity:
            initialize_machine(runtime, cfg.machine_name, agent_mode="daemon")
        register_projects(runtime, [cfg.shared_root], machine_name=cfg.machine_name)
        binding = runtime.ensure_binding(cfg.shared_root, cfg.machine_name)[0]
        runtime_root = runtime.project_paths(binding.project_id)["root"]
        (root / "runtime-root").write_text(str(runtime_root))
        set_progress_policy(cfg.shared_root, 1)
        if action == "setup-short":
            save_lease_policy(
                cfg,
                LeasePolicy(
                    ttl_seconds=13,
                    renew_interval_seconds=1.5,
                    retry_initial_seconds=0.25,
                    retry_max_seconds=1.0,
                    renewal_commit_margin_seconds=2.0,
                ),
            )
        return {"source": observer.__file__}
    cfg = RootConfig(root / ".qexp", root, "rollout-machine", Path((root / "runtime-root").read_text()))
    if action == "start":
        process = start_machine_agent(runtime, available_gpus=[0, 1], loop_interval=0.05)
        return {"pid": process.pid}
    if action == "stop":
        return {"stopped": stop_machine_agent(runtime, timeout=10)}
    if action.startswith("submit-"):
        directory = root / action
        directory.mkdir()
        task = submit_specs(
            cfg,
            [
                {
                    "command": [sys.executable, "-c", WORKLOAD, str(directory)],
                    "working_directory": str(root),
                    "live_progress": action != "submit-disabled",
                }
            ],
        )[0]
        return {"task_id": task.task_id}
    if action == "show":
        return observer.inspect_task(cfg, task_id)
    if action == "probe-durable":
        from qqtools.plugins.qexp.runtime.paths import attempt_path
        from qqtools.plugins.qexp.runtime.store import read_json
        from qqtools.plugins.qexp.runtime.tasks import load_task
        from qqtools.plugins.qexp.scheduler import expire_claim, renew_attempt_lease

        task = load_task(cfg, task_id)
        claim = task.claim_control["active_claim"]
        path = attempt_path(cfg.shared_root, task_id, claim["attempt_number"])
        before = read_json(path)
        assert claim["authority_mode"] == "holder_bound"
        renewal = renew_attempt_lease(cfg, task_id, claim["attempt_id"], claim["fencing_token"])
        assert renewal.outcome.value == "not_required"
        assert not expire_claim(
            cfg, task_id, claim["attempt_id"], claim["fencing_token"], reservation_runtime_root=runtime.root
        )
        assert read_json(path) == before
        assert load_task(cfg, task_id).claim_control["active_claim"] == claim
        return {"source": observer.__file__, "renewal": renewal.outcome.value, "expiry_rejected": True}
    if action in {"clean", "clean-running"}:
        try:
            return clean(cfg, task_id=task_id, reservation_runtime_root=runtime.root)
        except ValueError as error:
            if action != "clean-running" or "task_state:running" not in str(error):
                raise
            return {"blocked_task_ids": [task_id], "reason": str(error)}
    raise ValueError(action)


def wait(predicate, stage: str):
    deadline = time.monotonic() + 30
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() > deadline:
            raise TimeoutError(stage)
        time.sleep(0.15)


def qualify(output: Path, ref: str, scenario: str = "rollout") -> None:
    repo = Path(__file__).resolve().parents[2]
    output.mkdir()
    revision = subprocess.check_output(["git", "rev-parse", f"{ref}^{{commit}}"], cwd=repo, text=True).strip()
    old = output / "released"
    old.mkdir()
    archive = subprocess.check_output(["git", "archive", revision, "src"], cwd=repo)
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        stream.extractall(old, filter="data")
    environment = dict(os.environ)
    for key in (
        "TMUX",
        "TMUX_PANE",
        "QQTOOLS_TEST_SOURCE_ROOT",
        "RANK",
        "SLURM_PROCID",
        "OMPI_COMM_WORLD_RANK",
        "QEXP_SHARED_ROOT",
    ):
        environment.pop(key, None)
    for key, relative in {
        "HOME": "home",
        "XDG_CACHE_HOME": "cache",
        "XDG_CONFIG_HOME": "config",
        "XDG_DATA_HOME": "data",
        "TMPDIR": "tmp",
        "TMP": "tmp",
        "TEMP": "tmp",
        "TMUX_TMPDIR": "tmux",
        "QEXP_MACHINE_RUNTIME_ROOT": "machine",
    }.items():
        directory = output / relative
        directory.mkdir(exist_ok=True)
        environment[key] = str(directory)
    clock_bin = output / "bin"
    clock_bin.mkdir()
    chronyc = clock_bin / "chronyc"
    chronyc.write_text(
        "#!/bin/sh\ncat <<'CLOCK'\nSystem time : 0.000001 seconds slow of NTP time\nRoot delay : 0.000002 seconds\nRoot dispersion : 0.000001 seconds\nSkew : 0.001 ppm\nLeap status : Normal\nCLOCK\n"
    )
    chronyc.chmod(0o755)
    environment.update(PATH=str(clock_bin) + os.pathsep + environment["PATH"], QEXP_VISIBLE_GPUS="0,1")
    fixture = Path(__file__).resolve()
    sources = {"old": old / "src", "new": repo / "src"}

    def call(version, action, task=None):
        result = subprocess.run(
            [sys.executable, str(fixture), "--actor", str(output), action, *([task] if task else [])],
            cwd=output,
            env=dict(environment, PYTHONPATH=str(sources[version])),
            capture_output=True,
            text=True,
            timeout=30,
        )
        with (output / "actors.log").open("a") as log:
            log.write(f"{version} {action}: {result.returncode}\n{result.stdout}\n{result.stderr}\n")
        result.check_returncode()
        return json.loads(result.stdout)

    def started(name):
        path = output / name / "started.json"
        return json.loads(path.read_text()) if path.exists() else None

    def snapshot(version, task):
        view = call(version, "show", task)
        for field in ("progress", "progress_extended"):
            report = view.get(field, {})
            if report.get("status") != "available" or report.get("observation_state") != "available":
                return None
            if not isinstance(report.get("source_update_id"), str) or not report["source_update_id"]:
                return None
            timestamp = report.get("reported_at")
            if not isinstance(timestamp, str) or not timestamp:
                return None
            if report.get("progress", {}).get("stage") != "work":
                return None
        return view

    results = {
        "released_ref": ref,
        "released_revision": revision,
        "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "clock": "deterministic bounded chronyc provider fixture; real process and lease logic",
        "candidate_source_sha256": hashlib.sha256(
            b"".join(
                str(p.relative_to(repo)).encode() + b"\0" + p.read_bytes() for p in sorted((repo / "src").rglob("*.py"))
            )
        ).hexdigest(),
    }
    try:
        original = None
        original_attempt = None
        if scenario in {"rollout", "durable"}:
            setup = call("old", "setup-short")
            assert str(old) in setup["source"], setup
            old_task = call("old", "submit-old")["task_id"]
            call("old", "start")
            original = wait(lambda: started("submit-old"), "released workload start")
            assert str(old) in original["source"]
            if scenario == "rollout":
                assert original["v3"] is None
            before = wait(lambda: snapshot("old", old_task), "released v2 report")
            original_attempt = before["task"]["attempt_control"]["current_attempt_id"]
            call("old", "stop")
            os.kill(original["pid"], 0)
            # Outlive the released source's valid short execution lease. The candidate must
            # preserve this exact process and Attempt without a timeout signal or replacement.
            time.sleep(15)
            os.kill(original["pid"], 0)
            call("new", "start")
            restarted_at = datetime.now(timezone.utc)
            after = wait(lambda: snapshot("new", old_task), "target observes released workload")
            assert after["task"]["attempt_control"]["current_attempt_id"] == original_attempt
            assert started("submit-old") == original
            os.kill(original["pid"], 0)
            wait(
                lambda: (
                    call("new", "show", old_task)["task"]["claim_control"]["active_claim"]["authority_mode"]
                    == "holder_bound"
                ),
                "legacy durable ownership adoption",
            )
            call("old", "probe-durable", old_task)
            decision_root = Path((output / "runtime-root").read_text()) / "termination-decisions" / original_attempt
            assert not decision_root.exists() or not any(decision_root.glob("*.json"))
            if scenario == "rollout":
                assert after["progress_scoped"]["status"] != "available"

            def continued_reporting():
                view = snapshot("new", old_task)
                if view is None:
                    return False
                for field in ("progress", "progress_extended"):
                    report = view[field]
                    previous = before[field]
                    accepted_at = datetime.fromisoformat(report["reported_at"].replace("Z", "+00:00"))
                    if accepted_at <= restarted_at or report["source_update_id"] == previous["source_update_id"]:
                        return False
                    if report["progress"]["current"] <= previous["progress"]["current"]:
                        return False
                return True

            wait(continued_reporting, "continued v1/v2 reporting accepted after agent restart")
        else:
            call("new", "setup")
        new_task = call("new", "submit-new")["task_id"]
        if scenario == "new-attempt":
            call("new", "start")
        new_process = wait(lambda: started("submit-new"), "target workload start")
        assert new_process["v3"] and str(repo / "src") in new_process["source"]
        scoped = wait(
            lambda: (
                view if (view := call("new", "show", new_task)).get("selected_progress_protocol_version") == 3 else None
            ),
            "target scoped report",
        )
        assert scoped["progress_scoped"]["progress"]["overall"]["label"] == "Training"
        legacy = wait(lambda: snapshot("old", new_task), "released reader sees new producer v2")
        assert legacy["progress_extended"]["progress"]["stage"] == "work"
        # Released reader/writer and target agent coexist against the same Project.
        assert new_task in call("old", "clean-running", new_task)["blocked_task_ids"]
        os.kill(new_process["pid"], 0)
        (output / "submit-new/release").touch()
        wait(lambda: call("new", "show", new_task)["task"]["state"]["projection"] == "succeeded", "target completion")
        disabled = call("new", "submit-disabled")["task_id"]
        disabled_process = wait(lambda: started("submit-disabled"), "opt-out workload")
        assert disabled_process["v3"] is None
        assert call("new", "show", disabled)["progress_scoped"]["status"] != "available"
        (output / "submit-disabled/release").touch()
        wait(lambda: call("new", "show", disabled)["task"]["state"]["projection"] == "succeeded", "opt-out completion")
        cleaned = call("old", "clean", new_task)
        assert cleaned["operations"][new_task]["state"] == "completed", cleaned
        # Old cleanup can leave unknown advisory sidecars, but cannot resurrect truth.
        assert not (output / ".qexp/tasks" / f"{new_task}.json").exists()
        if scenario in {"rollout", "durable"}:
            # Completion while the agent is offline exercises separate recovery proof.
            call("new", "stop")
            (output / "submit-old/release").touch()
            wait(lambda: (output / "submit-old/finished").exists(), "offline workload completion")
            observation_path = (
                Path((output / "runtime-root").read_text()) / "process-observations" / f"{original_attempt}.json"
            )
            wait(observation_path.exists, "durable offline exit observation")
            assert json.loads(observation_path.read_text())["exit_observation"]["observed_exit_code"] == 0
            call("new", "start")
            wait(
                lambda: call("new", "show", old_task)["task"]["state"]["projection"] == "succeeded",
                "offline completion recovery",
            )
        results.update(
            status="passed",
            scenario=scenario,
            old_pid=None if original is None else original["pid"],
            old_attempt_id=original_attempt,
            new_task_id=new_task,
            new_pid=new_process["pid"],
            checks=(
                [
                    "released-import provenance",
                    "same-PID same-Attempt rolling agent restart",
                    "TTL=13 agent outage beyond last lease expiry without signal",
                    "continued v1/v2 old producer",
                    "offline completion recovery",
                ]
                if scenario in {"rollout", "durable"}
                else []
            )
            + [
                "new v3 activation",
                "old reader of new v1/v2",
                "old cleanup cannot remove live task",
                "old terminal cleanup accepts new attempt",
                "frozen opt-out",
            ],
        )
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results, indent=2))
    except Exception as error:
        results.update(status="failed", scenario=scenario, error=f"{type(error).__name__}: {error}")
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        raise
    finally:
        for directory in output.glob("submit-*"):
            (directory / "release").touch()
        call("new", "stop")
        for path in (output / "tmux").rglob("*"):
            if path.is_socket():
                subprocess.run(["tmux", "-S", str(path), "kill-server"], capture_output=True, timeout=5, check=False)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--actor":
        print(json.dumps(actor(Path(sys.argv[2]), sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)))
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--ref", required=True)
        parser.add_argument("--scenario", choices=("rollout", "new-attempt", "durable"), default="rollout")
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args()
        qualify(args.output.resolve(), args.ref, args.scenario)
