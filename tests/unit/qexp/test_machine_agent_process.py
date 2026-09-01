from __future__ import annotations

import json
import subprocess
from pathlib import Path

from qqtools.plugins.qexp.machine_agent_process import spawn_machine_agent_process
from qqtools.plugins.qexp.machine_runtime import MachineRuntime


def test_machine_agent_default_stderr_does_not_create_an_unread_pipe(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 123
        stderr = None

        def poll(self) -> None:
            return None

    def start_process(*_args, **kwargs) -> FakeProcess:
        captured.update(kwargs)
        status_path = runtime.paths["agent"] / "status.json"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(
            json.dumps({"machine_agent": {"state": "active", "pid": FakeProcess.pid}}),
            encoding="utf-8",
        )
        return FakeProcess()

    monkeypatch.setattr(
        "qqtools.plugins.qexp.machine_agent_process.subprocess.Popen", start_process
    )

    assert spawn_machine_agent_process(runtime).pid == FakeProcess.pid
    assert captured["stderr"] is subprocess.DEVNULL
