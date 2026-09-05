from pathlib import Path

import pytest

from qqtools.plugins.qexp.cli import main
from qqtools.plugins.qexp.commands.task import submit
from qqtools.plugins.qexp.cpu_lane_upgrade import (
    attest_cpu_lane_upgrade,
    cpu_lane_upgrade_status,
    resume_cpu_lane_upgrade,
    start_cpu_lane_upgrade,
)
from qqtools.plugins.qexp.layout import is_cpu_lane_root
from qqtools.plugins.qexp.machine_config import init_shared_root
from qqtools.plugins.qexp.machine_runtime import MachineRuntime
from qqtools.plugins.qexp.runtime.reservations import reserve
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json


def _legacy_root(tmp_path: Path):
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1")
    MachineRuntime(cfg.runtime_root).add_binding(cfg.shared_root, cfg.machine_name)
    path = cfg.shared_root / "schema" / "version.json"
    schema = read_json(path)
    del schema["schema"]["required_capabilities"]
    atomic_replace(path, schema)
    return cfg


def test_drained_legacy_root_activates_cpu_lane_with_explicit_session(tmp_path: Path) -> None:
    cfg = _legacy_root(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    task = submit(cfg, ["echo", "gpu"], working_dir=work)

    session = start_cpu_lane_upgrade(cfg, machine_runtime_root=cfg.runtime_root)
    assert session["phase"] == "awaiting_attestations"
    with pytest.raises(RuntimeError, match="preparing"):
        submit(cfg, ["echo", "blocked"], working_dir=work)

    attested = attest_cpu_lane_upgrade(
        cfg, activation_id=session["activation_id"], machine_name="gpu-1",
        machine_runtime_root=cfg.runtime_root,
    )
    completed = resume_cpu_lane_upgrade(
        cfg, activation_id=session["activation_id"], machine_runtime_root=cfg.runtime_root
    )

    assert attested["attestations"]["gpu-1"]["machine_name"] == "gpu-1"
    assert completed["phase"] == "completed"
    assert is_cpu_lane_root(cfg)
    assert read_json(cfg.shared_root / "tasks" / f"{task.task_id}.json")["task"]["spec"]["lane"] == "gpu"
    assert cpu_lane_upgrade_status(cfg)["protocol_state"] == "canonical"


def test_upgrade_start_is_idempotent_for_a_legacy_root(tmp_path: Path) -> None:
    cfg = _legacy_root(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    submit(cfg, ["echo", "queued"], working_dir=work)

    first = start_cpu_lane_upgrade(cfg, machine_runtime_root=cfg.runtime_root)
    second = start_cpu_lane_upgrade(cfg, machine_runtime_root=cfg.runtime_root)
    assert second["activation_id"] == first["activation_id"]
    assert not is_cpu_lane_root(cfg)


def test_interrupted_normalization_requires_fresh_attestations(tmp_path: Path) -> None:
    cfg = _legacy_root(tmp_path)
    session = start_cpu_lane_upgrade(cfg, machine_runtime_root=cfg.runtime_root)
    attest_cpu_lane_upgrade(
        cfg, activation_id=session["activation_id"], machine_name="gpu-1",
        machine_runtime_root=cfg.runtime_root,
    )
    journal_path = cfg.shared_root / "schema" / "cpu-lane-upgrade.json"
    journal = read_json(journal_path)
    journal["cpu_lane_upgrade"]["phase"] = "normalizing"
    atomic_replace(journal_path, journal)

    with pytest.raises(RuntimeError, match="collect fresh"):
        resume_cpu_lane_upgrade(
            cfg, activation_id=session["activation_id"], machine_runtime_root=cfg.runtime_root
        )

    recovered = read_json(journal_path)["cpu_lane_upgrade"]
    assert recovered["phase"] == "awaiting_attestations"
    assert recovered["attestations"] == {}


def test_completed_upgrade_resume_is_idempotent_even_after_work_restarts(tmp_path: Path) -> None:
    cfg = _legacy_root(tmp_path)
    session = start_cpu_lane_upgrade(cfg, machine_runtime_root=cfg.runtime_root)
    attest_cpu_lane_upgrade(
        cfg, activation_id=session["activation_id"], machine_name="gpu-1",
        machine_runtime_root=cfg.runtime_root,
    )
    completed = resume_cpu_lane_upgrade(
        cfg, activation_id=session["activation_id"], machine_runtime_root=cfg.runtime_root
    )

    task = submit(cfg, ["echo", "gpu"], working_dir=tmp_path)
    task_path = cfg.shared_root / "tasks" / f"{task.task_id}.json"
    value = read_json(task_path)
    value["task"]["state"]["projection"] = "running"
    atomic_replace(task_path, value)

    assert resume_cpu_lane_upgrade(
        cfg, activation_id=session["activation_id"], machine_runtime_root=cfg.runtime_root
    ) == completed


def test_upgrade_cli_requires_explicit_shared_root(tmp_path: Path) -> None:
    cfg = _legacy_root(tmp_path)

    assert main([
        "--shared-root", str(cfg.shared_root), "--machine-runtime-root", str(cfg.runtime_root),
        "upgrade", "cpu-lane", "check",
    ]) == 0


def test_upgrade_checks_machine_runtime_reservations_not_legacy_runtime(tmp_path: Path) -> None:
    cfg = _legacy_root(tmp_path)
    runtime = MachineRuntime(cfg.runtime_root)
    binding = runtime.resolve_project_binding(cfg.shared_root)
    reserve(runtime.root, "running", [0], project_id=binding.project_id)

    from qqtools.plugins.qexp.cpu_lane_upgrade import check_cpu_lane_upgrade

    assert "runtime:provisional" in check_cpu_lane_upgrade(
        cfg, machine_runtime_root=runtime.root
    )["blockers"]


def test_upgrade_default_runtime_uses_machine_runtime_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _legacy_root(tmp_path)
    machine_runtime = tmp_path / "machine-runtime"
    MachineRuntime(machine_runtime).add_binding(cfg.shared_root, cfg.machine_name)
    monkeypatch.setenv("QEXP_MACHINE_RUNTIME_ROOT", str(machine_runtime))

    from qqtools.plugins.qexp.cpu_lane_upgrade import check_cpu_lane_upgrade

    assert check_cpu_lane_upgrade(cfg)["blockers"] == []


def test_attestation_requires_the_declared_local_machine_binding(tmp_path: Path) -> None:
    cfg = _legacy_root(tmp_path)
    init_shared_root(cfg.shared_root, "host-b", runtime_root=tmp_path / "host-b-project-runtime")
    host_a_runtime = tmp_path / "host-a-machine-runtime"
    MachineRuntime(host_a_runtime).add_binding(cfg.shared_root, "gpu-1")
    session = start_cpu_lane_upgrade(cfg, machine_runtime_root=host_a_runtime)

    with pytest.raises(ValueError, match="does not match the local machine binding"):
        attest_cpu_lane_upgrade(
            cfg,
            activation_id=session["activation_id"],
            machine_name="host-b",
            machine_runtime_root=host_a_runtime,
        )


def test_resume_rejects_tampered_attestation_binding(tmp_path: Path) -> None:
    cfg = _legacy_root(tmp_path)
    session = start_cpu_lane_upgrade(cfg, machine_runtime_root=cfg.runtime_root)
    attest_cpu_lane_upgrade(
        cfg, activation_id=session["activation_id"], machine_name="gpu-1",
        machine_runtime_root=cfg.runtime_root,
    )
    journal_path = cfg.shared_root / "schema" / "cpu-lane-upgrade.json"
    journal = read_json(journal_path)
    journal["cpu_lane_upgrade"]["attestations"]["gpu-1"]["binding_machine_name"] = "host-b"
    atomic_replace(journal_path, journal)

    with pytest.raises(ValueError, match="does not match machine binding"):
        resume_cpu_lane_upgrade(
            cfg, activation_id=session["activation_id"], machine_runtime_root=cfg.runtime_root
        )


def test_resume_rejects_attestation_from_replaced_machine_runtime(tmp_path: Path) -> None:
    cfg = _legacy_root(tmp_path)
    first_runtime = tmp_path / "first-machine-runtime"
    MachineRuntime(first_runtime).add_binding(cfg.shared_root, cfg.machine_name)
    replacement_runtime = tmp_path / "replacement-machine-runtime"
    MachineRuntime(replacement_runtime).add_binding(cfg.shared_root, cfg.machine_name)
    session = start_cpu_lane_upgrade(cfg, machine_runtime_root=first_runtime)
    attest_cpu_lane_upgrade(
        cfg,
        activation_id=session["activation_id"],
        machine_name=cfg.machine_name,
        machine_runtime_root=first_runtime,
    )
    with pytest.raises(ValueError, match="does not match the local machine runtime"):
        resume_cpu_lane_upgrade(
            cfg,
            activation_id=session["activation_id"],
            machine_runtime_root=replacement_runtime,
        )
