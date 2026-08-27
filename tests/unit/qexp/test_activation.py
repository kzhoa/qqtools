import time
from pathlib import Path
from threading import Barrier, Thread

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.activation import ensure_local_agent_active
from qqtools.plugins.qexp.machine_runtime import MachineRuntime
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json


def test_unregistered_current_project_requires_explicit_add(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime")
    runtime = MachineRuntime(tmp_path / "machine-runtime")

    with pytest.raises(RuntimeError, match="qexp agent add-project"):
        ensure_local_agent_active(cfg, reason="submit", machine_runtime=runtime)


def test_legacy_project_requires_explicit_migration(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime")
    record_path = cfg.shared_root / "machines" / cfg.machine_name / "machine.json"
    record = read_json(record_path)
    record["machine"].pop("agent_runtime")
    atomic_replace(record_path, record)

    with pytest.raises(RuntimeError, match="qexp agent migrate-project"):
        ensure_local_agent_active(cfg, reason="submit", machine_runtime=MachineRuntime(tmp_path / "machine-runtime"))


def test_registered_project_does_not_start_a_second_machine_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime")
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.add_binding(cfg.shared_root, cfg.machine_name)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.machine_agent.get_machine_agent_status",
        lambda _runtime: {"is_running": True},
    )

    assert ensure_local_agent_active(cfg, reason="submit", machine_runtime=runtime) is False


def test_concurrent_activation_starts_only_one_machine_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime")
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.add_binding(cfg.shared_root, cfg.machine_name)
    is_running = False
    starts = 0
    barrier = Barrier(3)
    results: list[bool] = []

    def get_status(_runtime):
        return {"is_running": is_running, "pid": 1234 if is_running else None}

    def start(_runtime, **_kwargs):
        nonlocal is_running, starts
        starts += 1
        time.sleep(0.05)
        is_running = True
        return type("Process", (), {"pid": 1234})()

    def activate() -> None:
        barrier.wait()
        results.append(ensure_local_agent_active(cfg, reason="submit", machine_runtime=runtime))

    monkeypatch.setattr("qqtools.plugins.qexp.machine_agent.get_machine_agent_status", get_status)
    monkeypatch.setattr("qqtools.plugins.qexp.machine_agent._start_machine_agent_locked", start)
    threads = [Thread(target=activate), Thread(target=activate)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert starts == 1
    assert sorted(results) == [False, True]


def test_activation_accepts_an_agent_that_wins_during_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime")
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.add_binding(cfg.shared_root, cfg.machine_name)
    status_reads = 0

    def get_status(_runtime):
        nonlocal status_reads
        status_reads += 1
        return {"is_running": status_reads > 1, "pid": 1234 if status_reads > 1 else None}

    def lose_startup(_runtime, **_kwargs):
        raise RuntimeError("machine scheduler authority is already held")

    monkeypatch.setattr("qqtools.plugins.qexp.machine_agent.get_machine_agent_status", get_status)
    monkeypatch.setattr("qqtools.plugins.qexp.machine_agent._start_machine_agent_locked", lose_startup)

    assert ensure_local_agent_active(cfg, reason="submit", machine_runtime=runtime) is False
