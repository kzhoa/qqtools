import json
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.cli import entrypoint as cli_entrypoint
from qqtools.plugins.qexp.commands.group import create_group

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _base_args(cfg) -> list[str]:
    machine_root = cfg.runtime_root.parent / "machine-runtime"
    MachineRuntime(machine_root).ensure_binding(cfg.shared_root, cfg.machine_name)
    return [
        "--project",
        str(cfg.shared_root),
        "--machine",
        cfg.machine_name,
        "--runtime-root",
        str(cfg.runtime_root),
        "--machine-runtime-root",
        str(machine_root),
    ]


def test_group_config_cli_shows_and_sets_future_default(tmp_path: Path, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "training")
    base = _base_args(cfg)

    assert cli_entrypoint.main([*base, "group", "config", "show", "training", "progress", "--format=json"]) == 0
    initial = json.loads(capsys.readouterr().out)
    assert initial["live_progress"] is False
    assert initial["source"] == "default"
    assert initial["revision"] == 0

    assert (
        cli_entrypoint.main(
            [*base, "group", "config", "set", "training", "progress", "--live-progress", "--format=json"]
        )
        == 0
    )
    changed = json.loads(capsys.readouterr().out)
    assert changed["live_progress"] is True
    assert changed["revision"] == 1
    assert changed["source"] == "configured"

    assert cli_entrypoint.main([*base, "group", "config", "show", "training", "progress", "--format=json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown == changed


def test_group_config_set_requires_one_selection(tmp_path: Path, capsys):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "training")
    base = _base_args(cfg)
    with pytest.raises(SystemExit) as exc_info:
        cli_entrypoint.main([*base, "group", "config", "set", "training", "progress"])
    assert exc_info.value.code == 2
    assert "required" in capsys.readouterr().err
