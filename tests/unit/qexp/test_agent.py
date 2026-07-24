from pathlib import Path

from qqtools.plugins.qexp.agent import _runtime_pid_value


def test_runtime_pid_value_preserves_existing_live_pid(tmp_path: Path, monkeypatch):
    pid_path = tmp_path / "agent.pid"
    pid_path.write_text("4321", encoding="utf-8")
    monkeypatch.setattr("qqtools.plugins.qexp.agent._pid_alive", lambda pid: pid == 4321)

    assert _runtime_pid_value(pid_path) == 4321


def test_runtime_pid_value_falls_back_when_existing_pid_is_dead(tmp_path: Path, monkeypatch):
    pid_path = tmp_path / "agent.pid"
    pid_path.write_text("4321", encoding="utf-8")
    monkeypatch.setattr("qqtools.plugins.qexp.agent._pid_alive", lambda pid: False)
    monkeypatch.setattr("qqtools.plugins.qexp.agent.os.getpid", lambda: 9876)

    assert _runtime_pid_value(pid_path) == 9876
