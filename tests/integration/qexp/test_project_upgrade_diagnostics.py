"""The overview's generated upgrade action works with a nondefault runtime."""

import json
import shlex

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.project_admin import register_project
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.cli.entrypoint import main
from qqtools.plugins.qexp.commands.status import project_status
from qqtools.plugins.qexp.runtime.paths import shared_paths
from qqtools.plugins.qexp.runtime.store import atomic_replace

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def test_generated_upgrade_inspection_command_targets_registered_project(tmp_path, capsys):
    runtime = MachineRuntime(tmp_path / "nondefault machine's root")
    initialize_machine(runtime, "gpu-1", agent_mode="daemon")
    cfg = init_shared_root(tmp_path / "selected project" / ".qexp", "gpu-1", runtime_root=tmp_path / "partition")
    binding = register_project(runtime, cfg.shared_root, "gpu-1").binding
    journal = {
        "upgrade": {
            "version": 1,
            "project_id": binding.project_id,
            "shared_root": str(cfg.shared_root),
            "pause": {},
            "revision": 1,
            "updated_at": "2026-11-24T00:00:00Z",
            "migrations": {
                "task-dependencies-v2": {
                    "name": "task-dependencies-v2",
                    "state": "repair_required",
                    "phase": "backfill",
                    "admission_blocked": False,
                    "error": "diagnostic fixture requiring repair",
                }
            },
        }
    }
    atomic_replace(shared_paths(cfg.shared_root)["upgrade_journal"], journal)
    result = project_status(cfg, selection_source="explicit", machine_runtime=runtime)
    command = shlex.split(result["next_actions"][0])
    assert command[:3] == ["qexp", "--machine-runtime-root", str(runtime.root)]
    assert command[-2:] == ["--project", binding.project_id]
    assert main([*command[1:], "--format", "json"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert binding.project_id in json.dumps(inspected)
    assert "repair_required" in json.dumps(inspected)
    assert "diagnostic fixture requiring repair" in json.dumps(inspected)
