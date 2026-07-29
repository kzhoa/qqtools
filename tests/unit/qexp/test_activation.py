from pathlib import Path

import pytest

from qqtools.plugins.qexp import AGENT_MODE_DAEMON, init_shared_root
from qqtools.plugins.qexp.activation import ensure_local_agent_active, stop_local_agent
from qqtools.plugins.qexp.agent_process import spawn_agent_process
from qqtools.plugins.qexp.machine_config import load_machine_policy
from qqtools.plugins.qexp.layout import runtime_pid_path
from qqtools.plugins.qexp.policy import resolve_machine_policy
from qqtools.plugins.qexp.runtime.paths import machine_path


def test_resolve_machine_policy_rejects_removed_persistent_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported qexp agent mode"):
        resolve_machine_policy("persistent")


def test_load_machine_policy_defaults_to_on_demand_when_machine_record_is_missing(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    machine_path(cfg.shared_root, cfg.machine_name).unlink()
    policy = load_machine_policy(cfg)
    assert policy.agent_mode == "on_demand"
    assert policy.exit_when_idle is True


def test_init_rejects_removed_persistent_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported qexp agent mode"):
        init_shared_root(tmp_path / ".qexp", "gpu-1", agent_mode="persistent", runtime_root=tmp_path / "rt")


def test_activation_launches_daemon_when_work_is_eligible(tmp_path: Path, monkeypatch) -> None:
    cfg = init_shared_root(
        tmp_path / ".qexp",
        "gpu-1",
        agent_mode=AGENT_MODE_DAEMON,
        runtime_root=tmp_path / "rt",
    )
    class FakeProcess:
        pid = 4321

    monkeypatch.setattr("qqtools.plugins.qexp.activation.get_agent_status", lambda cfg: {"is_running": False})
    monkeypatch.setattr("qqtools.plugins.qexp.activation.has_eligible_local_work", lambda cfg: True)
    monkeypatch.setattr("qqtools.plugins.qexp.activation.spawn_agent_process", lambda cfg: FakeProcess())

    assert ensure_local_agent_active(cfg, reason="submit") is True


def test_activation_launches_on_demand_agent_when_work_is_eligible(tmp_path: Path, monkeypatch) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    spawned = {"count": 0}

    class FakeProcess:
        pid = 4321

    monkeypatch.setattr("qqtools.plugins.qexp.activation.get_agent_status", lambda cfg: {"is_running": False})
    monkeypatch.setattr("qqtools.plugins.qexp.activation.has_eligible_local_work", lambda cfg: True)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.activation.spawn_agent_process",
        lambda cfg: spawned.__setitem__("count", spawned["count"] + 1) or FakeProcess(),
    )

    assert ensure_local_agent_active(cfg, reason="submit") is True
    assert spawned["count"] == 1


def test_activation_does_not_spawn_again_after_parent_registers_pid(tmp_path: Path, monkeypatch) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    spawned = {"count": 0}

    class FakeProcess:
        pid = 4321

    def status(cfg):
        return {"is_running": runtime_pid_path(cfg).exists()}

    def spawn(cfg):
        spawned["count"] += 1
        pid_path = runtime_pid_path(cfg)
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(FakeProcess.pid), encoding="utf-8")
        return FakeProcess()

    monkeypatch.setattr("qqtools.plugins.qexp.activation.get_agent_status", status)
    monkeypatch.setattr("qqtools.plugins.qexp.activation.has_eligible_local_work", lambda cfg: True)
    monkeypatch.setattr("qqtools.plugins.qexp.activation.spawn_agent_process", spawn)

    assert ensure_local_agent_active(cfg, reason="first") is True
    assert ensure_local_agent_active(cfg, reason="second") is False
    assert spawned["count"] == 1


def test_spawn_registers_pid_before_returning(tmp_path: Path, monkeypatch) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")

    class FakeProcess:
        pid = 4321

    monkeypatch.setattr(
        "qqtools.plugins.qexp.agent_process.subprocess.Popen",
        lambda *args, **kwargs: FakeProcess(),
    )

    assert spawn_agent_process(cfg).pid == 4321
    assert runtime_pid_path(cfg).read_text(encoding="utf-8") == "4321"


def test_activation_skips_when_agent_already_running(tmp_path: Path, monkeypatch) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    monkeypatch.setattr("qqtools.plugins.qexp.activation.get_agent_status", lambda cfg: {"is_running": True})
    monkeypatch.setattr("qqtools.plugins.qexp.activation.has_eligible_local_work", lambda cfg: True)

    assert ensure_local_agent_active(cfg, reason="submit") is False


def test_activation_skips_when_no_eligible_work_exists(tmp_path: Path, monkeypatch) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    monkeypatch.setattr("qqtools.plugins.qexp.activation.get_agent_status", lambda cfg: {"is_running": False})
    monkeypatch.setattr("qqtools.plugins.qexp.activation.has_eligible_local_work", lambda cfg: False)

    assert ensure_local_agent_active(cfg, reason="submit") is False


def test_activation_skips_when_lock_is_already_held(tmp_path: Path, monkeypatch) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    monkeypatch.setattr("qqtools.plugins.qexp.activation.get_agent_status", lambda cfg: {"is_running": False})
    monkeypatch.setattr("qqtools.plugins.qexp.activation.has_eligible_local_work", lambda cfg: True)
    lock_path = cfg.runtime_root / "locks" / "activation.lock"

    from qqtools.plugins.qexp.runtime.locks import exclusive

    with exclusive(lock_path, blocking=True):
        assert ensure_local_agent_active(cfg, reason="submit") is False


def test_stop_removes_stale_pid_file(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    pid_path = runtime_pid_path(cfg)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text("4321", encoding="utf-8")

    action, status = stop_local_agent(cfg)

    assert action == "already_stopped"
    assert status["is_running"] is False
    assert not pid_path.exists()
