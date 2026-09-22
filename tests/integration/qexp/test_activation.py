import shlex
import time
from pathlib import Path
from threading import Barrier, Thread

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.activation import AgentActivationError, ensure_local_agent_active
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.lifecycle import (
    MachineAgentStartBlockedError,
    MachineAgentStartError,
    restart_machine_agent,
)
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def test_unregistered_current_project_requires_explicit_registration(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime")
    runtime = MachineRuntime(tmp_path / "machine-runtime")

    with pytest.raises(RuntimeError, match="qexp project register"):
        ensure_local_agent_active(cfg, reason="submit", machine_runtime=runtime)


def test_legacy_project_requires_explicit_migration(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime")
    record_path = cfg.shared_root / "machines" / cfg.machine_name / "machine.json"
    record = read_json(record_path)
    record["machine"].pop("agent_runtime")
    atomic_replace(record_path, record)

    with pytest.raises(RuntimeError) as error:
        ensure_local_agent_active(cfg, reason="submit", machine_runtime=MachineRuntime(tmp_path / "machine-runtime"))
    command = str(error.value).partition("run '")[2].removesuffix("'.")
    assert shlex.split(command) == [
        "qexp",
        "admin",
        "migrate",
        "agent",
        "--project",
        str(cfg.shared_root),
        "--machine",
        cfg.machine_name,
    ]


def test_registered_project_does_not_start_a_second_machine_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime")
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.add_binding(cfg.shared_root, cfg.machine_name)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.agent.lifecycle.get_machine_agent_status",
        lambda _runtime: {"is_running": True},
    )

    assert ensure_local_agent_active(cfg, reason="submit", machine_runtime=runtime) is False


def test_pending_machine_replacement_is_a_named_activation_failure(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime")
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.add_binding(cfg.shared_root, cfg.machine_name)
    archive = runtime.paths["archives"] / "replacement"
    atomic_replace(
        runtime.paths["replacement_transaction"],
        {
            "replacement": {
                "version": 1,
                "target_name": "gpu-2",
                "new_runtime_id": "a" * 64,
                "policy": "daemon",
                "phase": "staged",
                "detach_old_runtime": True,
                "archive_path": str(archive),
            }
        },
    )

    with pytest.raises(AgentActivationError, match="pending machine replacement") as captured:
        ensure_local_agent_active(cfg, reason="submit", machine_runtime=runtime)
    assert captured.value.next_action is not None
    assert str(runtime.root) in captured.value.next_action
    assert "init --machine gpu-2 --agent-mode daemon --yes --detach-old-runtime" in captured.value.next_action


def test_activation_named_start_error_retries_without_replacing_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime")
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.add_binding(cfg.shared_root, cfg.machine_name)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.activation._ensure_machine_agent_started",
        lambda _runtime: (_ for _ in ()).throw(MachineAgentStartError("spawn denied")),
    )

    with pytest.raises(AgentActivationError, match="spawn denied") as captured:
        ensure_local_agent_active(cfg, reason="submit", machine_runtime=runtime)

    assert shlex.split(captured.value.next_action or "") == [
        "qexp",
        "--machine-runtime-root",
        str(runtime.root),
        "agent",
        "start",
    ]


@pytest.mark.parametrize("error", [ValueError("programming defect"), OSError("programming defect")])
def test_activation_unexpected_errors_are_not_downgraded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "legacy-runtime")
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.add_binding(cfg.shared_root, cfg.machine_name)
    monkeypatch.setattr(
        "qqtools.plugins.qexp.activation._ensure_machine_agent_started",
        lambda _runtime: (_ for _ in ()).throw(error),
    )

    with pytest.raises(type(error), match="programming defect"):
        ensure_local_agent_active(cfg, reason="submit", machine_runtime=runtime)


def test_restart_is_blocked_before_stopping_during_machine_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    initialize_machine(runtime, "gpu-1")
    runtime.paths["replacement_transaction"].write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "qqtools.plugins.qexp.agent.lifecycle._stop_machine_agent_locked",
        lambda *_args, **_kwargs: pytest.fail("restart stopped the agent before checking replacement state"),
    )

    with pytest.raises(MachineAgentStartBlockedError, match="pending machine replacement"):
        restart_machine_agent(runtime)


def test_concurrent_activation_starts_only_one_machine_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    monkeypatch.setattr("qqtools.plugins.qexp.agent.lifecycle.get_machine_agent_status", get_status)
    monkeypatch.setattr("qqtools.plugins.qexp.agent.lifecycle._start_machine_agent_locked", start)
    threads = [Thread(target=activate), Thread(target=activate)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert starts == 1
    assert sorted(results) == [False, True]


def test_activation_accepts_an_agent_that_wins_during_startup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    monkeypatch.setattr("qqtools.plugins.qexp.agent.lifecycle.get_machine_agent_status", get_status)
    monkeypatch.setattr("qqtools.plugins.qexp.agent.lifecycle._start_machine_agent_locked", lose_startup)

    assert ensure_local_agent_active(cfg, reason="submit", machine_runtime=runtime) is False
