from pathlib import Path

import pytest

from qqtools.plugins.qexp import AGENT_MODE_DAEMON, init_shared_root
from qqtools.plugins.qexp.activation import ensure_local_agent_active
from qqtools.plugins.qexp.agent_process import spawn_agent_process
from qqtools.plugins.qexp.machine_config import load_machine_policy
from qqtools.plugins.qexp.layout import runtime_pid_path
from qqtools.plugins.qexp.policy import resolve_machine_policy
from qqtools.plugins.qexp.runtime.paths import machine_path, shared_paths
from qqtools.plugins.qexp.runtime.store import iter_json, read_json


def test_resolve_machine_policy_rejects_removed_persistent_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported qexp agent mode"):
        resolve_machine_policy("persistent")


def test_load_machine_policy_defaults_to_on_demand_when_machine_record_is_missing(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    machine_path(cfg.shared_root, cfg.machine_name).unlink()
    policy = load_machine_policy(cfg)
    assert policy.agent_mode == "on_demand"
    assert policy.autostart_local is True
    assert policy.exit_when_idle is True


def test_init_rejects_removed_persistent_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported qexp agent mode"):
        init_shared_root(tmp_path / ".qexp", "gpu-1", agent_mode="persistent", runtime_root=tmp_path / "rt")


def test_activation_skips_for_daemon_mode(tmp_path: Path) -> None:
    cfg = init_shared_root(
        tmp_path / ".qexp",
        "gpu-1",
        agent_mode=AGENT_MODE_DAEMON,
        runtime_root=tmp_path / "rt",
    )
    assert ensure_local_agent_active(cfg, reason="submit") is False
    event_roots = sorted(path for path in shared_paths(cfg.shared_root)["events"].iterdir() if path.is_dir())
    events = [read_json(path) for path in iter_json(event_roots[-1])]
    event_types = {event["event_type"] for event in events}
    assert "agent_activation_requested" in event_types
    assert "agent_activation_skipped" in event_types
    skipped = [event for event in events if event["event_type"] == "agent_activation_skipped"]
    assert skipped[-1]["details"]["skip_reason"] == "manual_mode"


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
