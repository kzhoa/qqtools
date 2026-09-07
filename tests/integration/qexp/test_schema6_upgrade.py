from pathlib import Path

import pytest

from qqtools.plugins.qexp.schema6_upgrade import (
    attest_schema6_upgrade,
    resume_schema6_upgrade,
    start_schema6_upgrade,
)
from qqtools.plugins.qexp.layout import validate_root_contract
from qqtools.plugins.qexp.machine_config import init_shared_root
from qqtools.plugins.qexp.machine_runtime import MachineRuntime
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json


def _legacy_root(tmp_path: Path):
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1")
    MachineRuntime(cfg.runtime_root).add_binding(cfg.shared_root, cfg.machine_name)
    path = cfg.shared_root / "schema" / "version.json"
    schema = read_json(path)
    del schema["schema"]["required_capabilities"]
    atomic_replace(path, schema)
    return cfg


def test_normal_access_rejects_a_legacy_cpu_lane_root(tmp_path: Path) -> None:
    cfg = _legacy_root(tmp_path)

    with pytest.raises(RuntimeError, match="requires cpu-lane-v1"):
        validate_root_contract(cfg)


def test_schema6_interruption_requires_fresh_bound_attestations(tmp_path: Path) -> None:
    cfg = _legacy_root(tmp_path)
    session = start_schema6_upgrade(cfg, machine_runtime_root=cfg.runtime_root)
    attest_schema6_upgrade(
        cfg,
        activation_id=session["activation_id"],
        machine_name="gpu-1",
        machine_runtime_root=cfg.runtime_root,
    )
    journal_path = cfg.shared_root / "schema" / "schema6-upgrade.json"
    journal = read_json(journal_path)
    journal["schema6_upgrade"]["phase"] = "normalizing"
    atomic_replace(journal_path, journal)

    with pytest.raises(RuntimeError, match="collect fresh"):
        resume_schema6_upgrade(
            cfg, activation_id=session["activation_id"], machine_runtime_root=cfg.runtime_root
        )

    recovered = read_json(journal_path)["schema6_upgrade"]
    assert recovered["phase"] == "awaiting_attestations"
    assert recovered["attestations"] == {}


@pytest.mark.parametrize("capabilities", [["cpu-lane-v1"], ["task-dependencies-v1"]])
def test_schema6_rejects_partial_capability_activation(
    tmp_path: Path, capabilities: list[str]
) -> None:
    cfg = _legacy_root(tmp_path)

    with pytest.raises(ValueError, match="requires cpu-lane-v1 and task-dependencies-v1 together"):
        start_schema6_upgrade(
            cfg, capabilities=capabilities, machine_runtime_root=cfg.runtime_root
        )
