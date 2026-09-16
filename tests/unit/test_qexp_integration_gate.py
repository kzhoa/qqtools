from __future__ import annotations

import json
from pathlib import Path

from scripts import qexp_integration_gate


def _phase(name: str, status: str = "passed") -> dict[str, object]:
    return {"name": name, "status": status, "exit_code": 0 if status == "passed" else 1}


def test_report_dir_removes_stale_gate_outputs_but_preserves_unowned_files(tmp_path: Path) -> None:
    (tmp_path / "ordinary-junit.xml").write_text("stale", encoding="utf-8")
    (tmp_path / "lifecycle-stderr.log").write_text("stale", encoding="utf-8")
    (tmp_path / "summary.json").write_text("stale", encoding="utf-8")
    unrelated = tmp_path / "keep.txt"
    unrelated.write_text("keep", encoding="utf-8")

    assert qexp_integration_gate._report_dir(tmp_path) == tmp_path

    assert sorted(path.name for path in tmp_path.iterdir()) == ["keep.txt"]


def test_gate_runs_both_phases_and_writes_distinct_reports(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, list[str], Path, float]] = []

    def run_phase(name, command, report_dir, deadline):
        calls.append((name, command, report_dir, deadline))
        return _phase(name)

    monkeypatch.setattr(qexp_integration_gate, "_run_phase", run_phase)

    assert qexp_integration_gate.main(["--report-dir", str(tmp_path), "--durations=10"]) == 0

    assert [call[0] for call in calls] == ["ordinary", "lifecycle"]
    assert calls[0][3] == calls[1][3]
    commands = [call[1] for call in calls]
    assert any("ordinary-junit.xml" in argument for argument in commands[0])
    assert any("lifecycle-junit.xml" in argument for argument in commands[1])
    assert all("--durations=10" in command for command in commands)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "passed"


def test_gate_preserves_failed_phase_even_when_later_phase_passes(tmp_path: Path, monkeypatch) -> None:
    results = iter((_phase("ordinary", "failed"), _phase("lifecycle")))
    monkeypatch.setattr(qexp_integration_gate, "_run_phase", lambda *_args: next(results))

    assert qexp_integration_gate.main(["--report-dir", str(tmp_path)]) == 1

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert [phase["name"] for phase in summary["phases"]] == ["ordinary", "lifecycle"]


def test_gate_stops_before_second_phase_after_shared_deadline(tmp_path: Path, monkeypatch) -> None:
    times = iter((0.0, 0.0, 2.0, 2.0, 2.0))
    calls: list[str] = []
    monkeypatch.setattr(qexp_integration_gate.time, "monotonic", lambda: next(times))

    def run_phase(name, *_args):
        calls.append(name)
        return _phase(name, "timeout")

    monkeypatch.setattr(qexp_integration_gate, "_run_phase", run_phase)

    assert qexp_integration_gate.main(["--budget-seconds", "1", "--report-dir", str(tmp_path)]) == 1
    assert calls == ["ordinary"]
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "timeout"
