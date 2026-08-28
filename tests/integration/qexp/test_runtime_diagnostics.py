from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.machine_agent import MachineRuntime, dispatch_machine_cycle_locked
from qqtools.plugins.qexp.runtime.store import read_json
from qqtools.plugins.qexp.runtime.tasks import load_task
from qqtools.plugins.qexp.runtime.work_budget import RuntimeDiagnostics, activate_diagnostics

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]

def test_runtime_diagnostics_count_task_reads_and_machine_stages(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp",
        "gpu-1",
        runtime_root=tmp_path / "legacy-runtime",
    )
    task = submit(cfg, ["echo", "ok"], working_dir=work_dir)
    diagnostics = RuntimeDiagnostics()

    with activate_diagnostics(diagnostics):
        assert load_task(cfg, task.task_id).task_id == task.task_id

    assert diagnostics.snapshot()["counters"]["task_json_read.records"] == 1
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.add_binding(cfg.shared_root, cfg.machine_name)
    dispatch_machine_cycle_locked(
        runtime,
        available_gpus=[],
        supervise=False,
        publish_snapshots=False,
    )
    value = read_json(runtime.paths["diagnostics"] / "scheduler-cycle.json")
    cycle = value["scheduler_diagnostic"]
    assert cycle["counters"]["maintain_project.calls"] == 1
    assert cycle["counters"]["offer_due_tasks.calls"] == 1
    assert cycle["counters"]["scheduler.work.skipped_no_capacity"] == 1
    assert "run_dispatch_cycle.calls" not in cycle["counters"]
    assert cycle["counters"].get("task_json_read.records", 0) == 0
    assert "reservation_enumeration" in cycle["timings"]
