from __future__ import annotations

from pathlib import Path

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.commands.status import MAX_READ_ATTEMPTS, MAX_RECORD_BYTES, machine_detail, project_status


def test_project_status_is_useful_without_enumerating_large_histories(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    for index in range(500):
        (cfg.shared_root / "tasks" / f"task-{index}.json").write_text("{}", encoding="utf-8")
        machine = cfg.shared_root / "machines" / f"machine-{index}"
        machine.mkdir()

    result = project_status(cfg, selection_source="explicit", machine_runtime=runtime)

    assert result["project"]["path"] == str(cfg.project_root)
    assert result["project"]["selection_source"] == "explicit"
    assert result["totals"] == {"tasks": None, "machines": None, "reason": "not_available"}
    assert result["budget"]["read_attempts"] <= MAX_READ_ATTEMPTS
    assert result["next_actions"][:2] == [
        f"qexp task list --project {cfg.project_root}",
        f"qexp machine list --project {cfg.project_root}",
    ]


def test_project_status_marks_oversized_optional_record_unavailable(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.paths["registry"].parent.mkdir(parents=True, exist_ok=True)
    runtime.paths["registry"].write_bytes(b" " * (MAX_RECORD_BYTES + 1))

    result = project_status(cfg, selection_source="environment", machine_runtime=runtime)

    assert result["status"] == "partial"
    assert result["local_participation"]["state"] == "unavailable"
    assert any("record_oversized" in warning for warning in result["warnings"])
    assert result["budget"]["bytes_read"] <= result["budget"]["max_total_bytes"]


def test_machine_detail_reads_one_declared_machine_without_history(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")

    result = machine_detail(cfg, "gpu-1")

    assert result["machine_name"] == "gpu-1"
    assert result["declaration"]["machine_name"] == "gpu-1"
    assert result["budget"]["read_attempts"] == 4
