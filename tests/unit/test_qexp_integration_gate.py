from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts import qexp_integration_gate


def _phase(name: str, status: str = "passed") -> dict[str, object]:
    return {"name": name, "status": status, "exit_code": 0 if status == "passed" else 1}


def _write_phase_evidence(report_dir: Path, name: str) -> None:
    (report_dir / f"{name}-junit.xml").write_text(
        '<testsuites><testsuite tests="1"><testcase name="test_example" /></testsuite></testsuites>\n',
        encoding="utf-8",
    )
    (report_dir / f"{name}-collection.txt").write_text("tests/test_example.py::test_example\n", encoding="utf-8")
    (report_dir / f"{name}-timing.json").write_text(
        json.dumps(
            {
                "exit_code": 0,
                "duration_seconds": 0.1,
                "has_skipped": False,
                "reports": [{"nodeid": "tests/test_example.py::test_example"}],
            }
        ),
        encoding="utf-8",
    )


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
        _write_phase_evidence(report_dir, name)
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
    assert summary["budget_seconds"] == 600.0
    assert summary["hard_timeout_seconds"] == 1200.0


def test_gate_preserves_failed_phase_even_when_later_phase_passes(tmp_path: Path, monkeypatch) -> None:
    results = iter((_phase("ordinary", "failed"), _phase("lifecycle")))
    monkeypatch.setattr(qexp_integration_gate, "_run_phase", lambda *_args: next(results))

    assert qexp_integration_gate.main(["--report-dir", str(tmp_path)]) == 1

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "test_failed"
    assert [phase["name"] for phase in summary["phases"]] == ["ordinary", "lifecycle"]


def test_gate_prints_bounded_failed_phase_log_tails(tmp_path: Path, monkeypatch, capsys) -> None:
    results = iter((_phase("ordinary", "failed"), _phase("lifecycle")))

    def run_phase(name, *_args):
        if name == "ordinary":
            (tmp_path / "ordinary-stdout.log").write_text(
                "discarded\n" + "x" * qexp_integration_gate.FAILURE_LOG_TAIL_BYTES + "\nfailed-test\n",
                encoding="utf-8",
            )
            (tmp_path / "ordinary-stderr.log").write_text("traceback\n", encoding="utf-8")
        return next(results)

    monkeypatch.setattr(qexp_integration_gate, "_run_phase", run_phase)

    assert qexp_integration_gate.main(["--report-dir", str(tmp_path)]) == 1

    diagnostic = capsys.readouterr().err
    assert "discarded" not in diagnostic
    assert "failed-test" in diagnostic
    assert "traceback" in diagnostic


def test_gate_completes_both_phases_before_rejecting_soft_budget(tmp_path: Path, monkeypatch, capsys) -> None:
    times = iter((0.0, 2.0))
    calls: list[str] = []
    monkeypatch.setattr(qexp_integration_gate.time, "monotonic", lambda: next(times))

    def run_phase(name, *_args):
        calls.append(name)
        _write_phase_evidence(tmp_path, name)
        return _phase(name)

    monkeypatch.setattr(qexp_integration_gate, "_run_phase", run_phase)

    assert qexp_integration_gate.main(["--budget-seconds", "1", "--report-dir", str(tmp_path)]) == 1
    assert calls == ["ordinary", "lifecycle"]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "budget_exceeded"
    assert summary["budget_overrun_seconds"] == 1.0
    assert "qexp integration gate budget_exceeded" in capsys.readouterr().err


def test_gate_stops_after_hard_timeout(tmp_path: Path, monkeypatch) -> None:
    results = iter((_phase("ordinary", "hard_timeout"), _phase("lifecycle")))
    monkeypatch.setattr(qexp_integration_gate, "_run_phase", lambda *_args: next(results))

    assert qexp_integration_gate.main(["--report-dir", str(tmp_path)]) == 1

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "hard_timeout"
    assert [phase["name"] for phase in summary["phases"]] == ["ordinary"]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--budget-seconds", "nan"], "finite and positive"),
        (["--budget-seconds", "10", "--hard-timeout-seconds", "10"], "must be greater"),
        (["--hard-timeout-seconds", "inf"], "must be greater"),
    ],
)
def test_gate_rejects_invalid_time_limits(arguments: list[str], message: str) -> None:
    with pytest.raises(SystemExit, match=message):
        qexp_integration_gate.main(arguments)


def test_run_phase_rejects_success_without_complete_evidence(tmp_path: Path) -> None:
    result = qexp_integration_gate._run_phase(
        "ordinary",
        [sys.executable, "-c", "pass"],
        tmp_path,
        time.monotonic() + 10,
    )

    assert result["status"] == "evidence_failed"
    assert len(result["evidence_errors"]) == 3


@pytest.mark.parametrize("junit", ["<testsuites />", "<not-junit><testcase /></not-junit>"])
def test_run_phase_rejects_empty_or_non_junit_xml(tmp_path: Path, junit: str) -> None:
    _write_phase_evidence(tmp_path, "ordinary")
    (tmp_path / "ordinary-junit.xml").write_text(junit, encoding="utf-8")

    result = qexp_integration_gate._run_phase(
        "ordinary",
        [sys.executable, "-c", "pass"],
        tmp_path,
        time.monotonic() + 10,
    )

    assert result["status"] == "evidence_failed"
    assert result["evidence_errors"] == [f"incomplete JUnit report {tmp_path / 'ordinary-junit.xml'}"]


def test_run_phase_hard_timeout_kills_sigterm_ignoring_descendant(tmp_path: Path) -> None:
    pid_path = tmp_path / "child.pid"
    child_code = (
        "import os,signal,sys,time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "open(sys.argv[1], 'w').write(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    leader_code = (
        "import subprocess,sys,time\n"
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}, sys.argv[1]])\n"
        "while True: time.sleep(0.1)\n"
    )

    result = qexp_integration_gate._run_phase(
        "ordinary",
        [sys.executable, "-c", leader_code, str(pid_path)],
        tmp_path,
        time.monotonic() + 1,
    )

    assert result["status"] == "hard_timeout"
    child_pid = int(pid_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        stat_path = Path(f"/proc/{child_pid}/stat")
        if not stat_path.exists() or stat_path.read_text(encoding="utf-8").split()[2] == "Z":
            break
        time.sleep(0.01)
    else:
        os.kill(child_pid, 9)
        pytest.fail(f"hard-timeout descendant {child_pid} remained alive")
