"""End-to-end v2 observation by an application with no qPipeline imports."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.authority import AuthoritySupervisor
from qqtools.plugins.qexp.cli.output import CliOutput, OutputKind, render
from qqtools.plugins.qexp.commands.cleanup import clean
from qqtools.plugins.qexp.observer import inspect_task
from qqtools.plugins.qexp.runtime import submission as submission_runtime
from qqtools.plugins.qexp.runtime.paths import attempt_path
from qqtools.plugins.qexp.runtime.progress import ProgressProjector, shared_progress_path
from qqtools.plugins.qexp.runtime.progress_v2 import ProgressV2Projector
from qqtools.plugins.qexp.runtime.store import read_json
from qqtools.plugins.qexp.scheduler import authorize_launch, claim_task
from qqtools.qexp._progress_protocol import replace_advisory_snapshot

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def test_selected_third_party_producer_projects_both_versions_and_details(tmp_path, monkeypatch):
    source_root = str(Path(__file__).resolve().parents[3] / "src")
    monkeypatch.setenv("PYTHONPATH", source_root + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setenv("QEXP_PROGRESS_V2_PATH", "/must-not-be-inherited")
    for key in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        monkeypatch.delenv(key, raising=False)
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    code = (
        "from qqtools.qexp import progress\n"
        "assert progress.update(stage='custom-stage',current=3,total=None,unit='batch',"
        "message='ready',metrics={'loss':0.25,'lr':0.001})\n"
        "progress.flush(timeout=1)\n"
    )
    task = submission_runtime.submit_specs(
        cfg,
        [{"command": [sys.executable, "-c", code], "working_directory": str(tmp_path), "live_progress": True}],
    )[0]
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    launch_id = read_json(attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number))["attempt"][
        "authorization"
    ]["launch_id"]
    runner = subprocess.run(
        [
            sys.executable,
            "-m",
            "qqtools.plugins.qexp.runner",
            "--shared-root",
            str(cfg.shared_root),
            "--machine",
            cfg.machine_name,
            "--task-id",
            task.task_id,
            "--attempt-id",
            attempt.attempt_id,
            "--fencing-token",
            str(attempt.current_fencing_token),
            "--launch-id",
            launch_id,
            "--runtime-root",
            str(cfg.runtime_root),
        ],
        check=False,
        timeout=30,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    assert runner.returncode == 0, runner.stderr
    AuthoritySupervisor(cfg).tick()

    base_projector = ProgressProjector(cfg, registration_generation="test-generation")
    base_projector.tick()
    base_projector.close()
    extended_projector = ProgressV2Projector(cfg, registration_generation="test-generation")
    extended_projector.tick()
    extended_projector.close()

    view = inspect_task(cfg, task.task_id)
    assert view["task"]["state"]["projection"] == "succeeded"
    assert view["progress"]["progress"]["stage"] == "custom-stage"
    assert set(view["progress"]["progress"]) == {"stage", "current", "total", "unit", "message"}
    assert view["progress_extended"]["progress"]["metrics"] == {"loss": 0.25, "lr": 0.001}
    assert view["progress_extended"]["progress"]["completeness"]["complete"] is True
    assert view["selected_progress_version"] == 2
    assert "Task ID" in render(CliOutput(OutputKind.TASK_SHOW, view), "human")
    assert "Metric loss" in render(CliOutput(OutputKind.TASK_SHOW, view, {"details": True}), "human")

    reported_at = view["progress_extended"]["reported_at"]
    restarted_projector = ProgressV2Projector(cfg, registration_generation="restarted-agent")
    restarted_projector.tick()
    restarted_projector.close()
    assert inspect_task(cfg, task.task_id)["progress_extended"]["reported_at"] == reported_at

    original_v1_timestamp = view["progress"]["reported_at"]
    extended_path = cfg.shared_root / "progress-v2" / task.task_id / f"{attempt.attempt_id}.json"
    malformed = read_json(extended_path)
    malformed["progress"]["metrics"]["loss"] = True
    replace_advisory_snapshot(extended_path, malformed)
    fallback = inspect_task(cfg, task.task_id)
    assert fallback["progress_extended"]["reason"] == "invalid_snapshot"
    assert fallback["selected_progress_version"] == 1
    assert fallback["progress"]["reported_at"] == original_v1_timestamp
    assert "Metric loss" not in render(CliOutput(OutputKind.TASK_SHOW, fallback, {"details": True}), "human")

    cleanup_result = clean(cfg, task_id=task.task_id, reservation_runtime_root=cfg.runtime_root)
    assert cleanup_result["operations"][task.task_id]["state"] == "completed", (
        cleanup_result["operations"][task.task_id]["blockers"],
        cleanup_result["operations"][task.task_id]["pending_machines"],
    )
    assert not shared_progress_path(cfg.shared_root, task.task_id, attempt.attempt_id).exists()
    assert not (cfg.shared_root / "progress-v2" / task.task_id).exists()
