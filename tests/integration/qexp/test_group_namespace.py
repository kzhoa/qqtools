"""Group namespace activation isolates released writers without a history copy."""

import os
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.commands.group import (
    change_worker,
    create_group,
    reconcile_group_cancel_operations,
    show_group,
)
from qqtools.plugins.qexp.layout import ensure_shared_layout, validate_root_contract
from qqtools.plugins.qexp.observer import list_groups
from qqtools.plugins.qexp.runtime import group_namespace
from qqtools.plugins.qexp.runtime.group_namespace import activate_group_authority_locked, group_directory, read_group
from qqtools.plugins.qexp.runtime.locks import schema_lock
from qqtools.plugins.qexp.runtime.operation_store import active_operation_path
from qqtools.plugins.qexp.runtime.paths import group_path
from qqtools.plugins.qexp.runtime.ready import state
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


@pytest.fixture
def project(tmp_path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "experiment")
    with schema_lock(cfg.shared_root):
        path = cfg.shared_root / "schema/version.json"
        schema = read_json(path)
        schema["schema"]["required_capabilities"].append("local-recovery-v1")
        atomic_replace(path, schema)
    return cfg


def activate(cfg):
    with schema_lock(cfg.shared_root):
        return activate_group_authority_locked(cfg)


@pytest.mark.parametrize(
    "cut", ["schema", "ready", "identity", "prepared", "rename", "root_sync", "moved", "capability", "completed"]
)
def test_activation_recovers_every_persistence_boundary(project, monkeypatch, cut):
    cfg = project
    original = group_path(cfg.shared_root, "experiment").read_bytes()
    original_identity = (cfg.shared_root / "groups").stat().st_ino
    replace, ready_replace = group_namespace.atomic_replace, state.atomic_replace
    rename, sync = os.rename, group_namespace.DurableIO.sync_directory

    def interrupted_write(path, value):
        replace(path, value)
        phase = value.get("group_authority", {}).get("phase")
        schema = value.get("schema", {})
        has_capability = "group-authority-v2" in schema.get("required_capabilities", [])
        if (
            phase == cut
            or (cut == "identity" and "group_directory" in value)
            or (cut == "schema" and schema)
            or (cut == "capability" and has_capability)
        ):
            raise OSError("injected namespace crash")

    def interrupted_ready(path, value):
        ready_replace(path, value)
        if cut == "ready":
            raise OSError("injected namespace crash")

    def interrupted_rename(source, destination):
        rename(source, destination)
        if cut == "rename":
            raise OSError("injected namespace crash")

    def interrupted_sync(self, directory, label):
        sync(self, directory, label)
        if cut == "root_sync" and label == "group_authority_move":
            raise OSError("injected namespace crash")

    with monkeypatch.context() as crashing:
        crashing.setattr(group_namespace, "atomic_replace", interrupted_write)
        crashing.setattr(state, "atomic_replace", interrupted_ready)
        crashing.setattr(group_namespace.os, "rename", interrupted_rename)
        crashing.setattr(group_namespace.DurableIO, "sync_directory", interrupted_sync)
        with pytest.raises(OSError, match="injected namespace crash"):
            activate(cfg)
    assert activate(cfg)
    assert activate(cfg)
    assert group_directory(cfg.shared_root) == cfg.shared_root / "groups-v2"
    assert group_directory(cfg.shared_root).stat().st_ino == original_identity
    assert group_path(cfg.shared_root, "experiment").read_bytes() == original
    assert state.read_ready_index_state(cfg) == "active"
    assert state.read_state_record(cfg)[1]["writer_capability"] == "ready-v2"
    validate_root_contract(cfg)
    ensure_shared_layout(cfg)
    assert not (cfg.shared_root / "groups").exists()


def test_old_shadow_never_becomes_truth_after_interrupted_move(project, monkeypatch):
    cfg = project
    old = read_group(cfg.shared_root, "experiment")
    rename = os.rename

    def interrupted(source, destination):
        rename(source, destination)
        raise OSError("after rename")

    with monkeypatch.context() as crashing:
        crashing.setattr(group_namespace.os, "rename", interrupted)
        with pytest.raises(OSError, match="after rename"):
            activate(cfg)
    old["group"]["worker_set"]["g1"]["state"] = "removing"
    atomic_replace(cfg.shared_root / "groups/experiment.json", old)
    assert activate(cfg)
    assert read_group(cfg.shared_root, "experiment")["group"]["worker_set"]["g1"]["state"] == "active"
    assert list_groups(cfg)[0]["group"]["worker_set"]["g1"]["state"] == "active"
    assert (cfg.shared_root / "groups/experiment.json").exists()


def test_group_reader_retries_path_resolved_before_cutover(project, monkeypatch):
    cfg = project
    original = group_namespace.read_json
    has_activated = False

    def racing_read(path):
        nonlocal has_activated
        if path == cfg.shared_root / "groups/experiment.json" and not has_activated:
            has_activated = True
            assert activate(cfg)
        return original(path)

    monkeypatch.setattr(group_namespace, "read_json", racing_read)
    assert read_group(cfg.shared_root, "experiment")["group"]["name"] == "experiment"
    assert has_activated


@pytest.mark.parametrize("damage", ["missing_journal", "missing_directory", "different_directory", "symlink"])
def test_activated_namespace_damage_never_falls_back_to_old_truth(project, damage):
    cfg = project
    old = read_group(cfg.shared_root, "experiment")
    assert activate(cfg)
    atomic_replace(cfg.shared_root / "groups/experiment.json", old)
    if damage == "missing_journal":
        group_namespace.journal_path(cfg.shared_root).unlink()
    else:
        destination = cfg.shared_root / "groups-v2"
        destination.rename(cfg.shared_root / "displaced")
        if damage == "different_directory":
            destination.mkdir()
        elif damage == "symlink":
            destination.symlink_to(cfg.shared_root / "displaced", target_is_directory=True)
    with pytest.raises(RuntimeError, match="Group authority"):
        read_group(cfg.shared_root, "experiment")
    with pytest.raises(RuntimeError, match="Group authority"):
        activate(cfg)


def test_legacy_removal_blocks_without_changing_worker_task_or_capacity(project):
    cfg = project
    task = submit(cfg, ["true"], group="experiment")
    before_cutover = change_worker(cfg, "experiment", "g1", "remove", terminate_running=True)
    legacy = before_cutover["worker_control"]
    assert legacy["operation_type"] == "worker_remove"
    before_task = load_task(cfg, task.task_id).to_dict()
    before_worker = before_cutover["group"]["worker_set"]["g1"]
    assert activate(cfg)
    controls = reconcile_group_cancel_operations(cfg)
    assert controls[0]["blocked_reason"] == "legacy_worker_incarnation_unknown"
    assert controls[0]["state"] == "blocked"
    assert read_group(cfg.shared_root, "experiment")["group"]["worker_set"]["g1"] == before_worker
    assert load_task(cfg, task.task_id).to_dict() == before_task
    assert not active_operation_path(cfg, "group_control", legacy["operation_id"]).exists()
    assert show_group(cfg, "experiment")["worker_control"]["state"] == "blocked"
    current = change_worker(cfg, "experiment", "g1", "remove")["worker_control"]
    assert current["operation_type"] == "worker_remove_v2"
    assert current["operation_id"] != legacy["operation_id"]


@pytest.mark.parametrize("damage", ["degraded", "missing", "absent"])
def test_current_ready_repair_keeps_enforced_writer_floor(project, damage):
    from qqtools.plugins.qexp.runtime.ready.rebuild import begin_ready_index_build

    cfg = project
    assert activate(cfg)
    path = cfg.shared_root / "indexes/ready/state.json"
    value = read_json(path)
    if damage == "missing":
        path.unlink()
    else:
        value["ready_index"]["state"] = damage
        atomic_replace(path, value)
    status = begin_ready_index_build(cfg, is_repair=True)
    assert status["writer_capability"] == "ready-v2"
    with pytest.raises(RuntimeError, match="requires writer capability"):
        state.assert_ready_writer_compatible(cfg, "ready-v1")


@pytest.mark.parametrize("command", ["create", "set", "seal"])
def test_writer_resolves_namespace_after_acquiring_schema_lock(project, monkeypatch, command):
    from contextlib import contextmanager

    from qqtools.plugins.qexp.commands import group as commands

    cfg = project
    writer_lock = commands.group_writer_lock
    calls = 0

    @contextmanager
    def racing_lock(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == (1 if command == "create" else 2):
            assert activate(cfg)
        with writer_lock(*args, **kwargs) as acquired:
            yield acquired

    monkeypatch.setattr(commands, "group_writer_lock", racing_lock)
    if command == "create":
        commands.create_group(cfg, "later")
        assert read_group(cfg.shared_root, "later")["group"]["name"] == "later"
    elif command == "set":
        commands.change_worker(cfg, "experiment", "g1", "set", role="borrow")
        assert read_group(cfg.shared_root, "experiment")["group"]["worker_set"]["g1"]["scheduling_role"] == "borrow"
    else:
        commands.group_control(cfg, "experiment", "seal")
        assert read_group(cfg.shared_root, "experiment")["group"]["admission_state"] == "sealed"
    assert not (cfg.shared_root / "groups").exists()


def test_empty_project_dispatch_remains_idle_after_namespace_activation(project, tmp_path):
    from qqtools.plugins.qexp.agent.context import MachineRuntime
    from qqtools.plugins.qexp.agent.dispatch_loop import dispatch_machine_cycle

    cfg = project
    runtime = MachineRuntime(tmp_path / "machine")
    runtime.add_binding(cfg.shared_root, cfg.machine_name)
    assert activate(cfg)
    result = dispatch_machine_cycle(runtime, available_gpus=[])
    assert not runtime.last_cycle_had_demand, result


def test_namespace_identity_is_independent_of_host_device_numbers(project, monkeypatch):
    cfg = project
    assert activate(cfg)
    original = group_namespace._identity

    def another_mount(path):
        identity = original(path)
        return None if identity is None else {"device": identity["device"] + 100, "inode": identity["inode"] + 200}

    monkeypatch.setattr(group_namespace, "_identity", another_mount)
    assert activate(cfg)
    assert read_group(cfg.shared_root, "experiment")["group"]["name"] == "experiment"


@pytest.mark.parametrize("damage", ["missing", "absent"])
def test_lost_ready_floor_rejects_task_writes(project, damage):
    from qqtools.plugins.qexp.runtime.tasks import save_task

    cfg = project
    task = submit(cfg, ["true"], group="experiment")
    assert activate(cfg)
    path = cfg.shared_root / "indexes/ready/state.json"
    if damage == "missing":
        path.unlink()
    else:
        value = read_json(path)
        value["ready_index"]["state"] = "absent"
        atomic_replace(path, value)
    task.state["projection"] = "cancelled"
    with pytest.raises(RuntimeError, match="writer floor has no ready state"):
        save_task(cfg, task)
    assert load_task(cfg, task.task_id).state["projection"] == "queued"


def test_machine_status_separates_capture_from_group_activation(project, tmp_path, monkeypatch):
    from qqtools.plugins.qexp.agent.context import MachineRuntime
    from qqtools.plugins.qexp.agent.lifecycle import get_machine_agent_status

    cfg = project
    runtime = MachineRuntime(tmp_path / "machine")
    runtime.add_binding(cfg.shared_root, cfg.machine_name)
    before = get_machine_agent_status(runtime)["projects"][0]
    assert before["group_authority"]["state"] == "waiting"
    assert activate(cfg)
    monkeypatch.setattr(
        group_namespace.DurableIO, "sync_directory", lambda *_args: pytest.fail("status attempted durability repair")
    )
    active = get_machine_agent_status(runtime)["projects"][0]
    assert active["group_authority"] == {"state": "active", "directory": "groups-v2", "diagnostic_only": True}
    group_namespace.journal_path(cfg.shared_root).write_text("broken")
    assert group_namespace.inspect_group_authority(cfg)["state"] == "unavailable"


def test_empty_ready_activation_keeps_schema_writer_floor(project):
    cfg = project
    assert activate(cfg)
    (cfg.shared_root / "indexes/ready/state.json").unlink()
    state.ensure_ready_layout(cfg)
    state._activate_empty_ready_index(cfg, "recreated")
    assert state.read_state_record(cfg)[1]["writer_capability"] == "ready-v2"
    with pytest.raises(RuntimeError, match="requires writer capability"):
        state.assert_ready_writer_compatible(cfg, "ready-v1")


def test_unknown_ready_writer_is_rejected_before_projection_repair(project, monkeypatch):
    from qqtools.plugins.qexp.runtime.ready import rebuild

    cfg = project
    assert activate(cfg)
    path = cfg.shared_root / "indexes/ready/state.json"
    value = read_json(path)
    value["ready_index"].update(state="degraded", writer_capability="ready-future")
    atomic_replace(path, value)
    monkeypatch.setattr(
        rebuild, "_reset_ready_projection_for_repair", lambda *_args: pytest.fail("unsupported writer reset projection")
    )
    with pytest.raises(RuntimeError, match="unknown writer capability"):
        rebuild.begin_ready_index_build(cfg, is_repair=True)
    assert read_json(path) == value
