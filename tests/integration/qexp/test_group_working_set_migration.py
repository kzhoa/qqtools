"""QQTOOLS-COMPAT-0025. Qualify conversion, restart and durable replay."""

import shutil

import pytest

from qqtools.plugins.qexp import init_shared_root, observer, submit
from qqtools.plugins.qexp.commands.group import create_group
from qqtools.plugins.qexp.runtime.group_namespace import group_directory
from qqtools.plugins.qexp.runtime.group_working_set_migration import index_active
from qqtools.plugins.qexp.runtime.protocol_compatibility import GROUP_WORKING_SET_CAPABILITY
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.upgrade import MigrationRegistry, UpgradeCoordinator
from qqtools.plugins.qexp.runtime.upgrade.group_working_set import GroupWorkingSetMigration

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def test_conversion_restarts_from_shared_cursor_and_preserves_committed_submission(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runtime.upgrade import group_working_set as conversion

    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "machine")
    root = cfg.shared_root
    create_group(cfg, "existing")
    original = submit(cfg, ["true"], group="existing", idempotency_key="cutover")
    directory = group_directory(root)
    historical = read_json(directory / "existing.json")
    historical["group"]["admission_state"] = "sealed"
    atomic_replace(directory / "existing.json", historical)
    for i in range(65):
        from qqtools.plugins.qexp.runtime.records import new_group

        name = f"old-{i:03}"
        group = new_group(name, "g1")
        group["group"]["admission_state"] = "open"
        atomic_replace(directory / f"{name}.json", group)

    schema_path = root / "schema/version.json"
    schema = read_json(schema_path)
    schema["schema"]["required_capabilities"].remove(GROUP_WORKING_SET_CAPABILITY)
    atomic_replace(schema_path, schema)
    (root / "schema" / f"{GROUP_WORKING_SET_CAPABILITY}.json").unlink()
    shutil.rmtree(root / "indexes/group-working-set")
    # Only the 1.3.29 canonical Group and operation records remain as source.
    registry = MigrationRegistry([GroupWorkingSetMigration()])
    UpgradeCoordinator(cfg, registry=registry).discover()
    starts = []
    real_entry = conversion.read_directory_entry

    def counted(path, offset):
        starts.append(offset)
        return real_entry(path, offset)

    monkeypatch.setattr(conversion, "read_directory_entry", counted)
    expected = {"existing", *(f"old-{i:03}" for i in range(65))}
    checked_inventory = checked_audit = False
    for _ in range(130):
        coordinator = UpgradeCoordinator(cfg, registry=registry)
        result = coordinator.advance(force_retry=True)
        assert result["state"] != "repair_required", result
        if result["state"] == "completed":
            break
        marker = root / "schema" / f"{GROUP_WORKING_SET_CAPABILITY}.json"
        state = read_json(marker)["state"] if marker.exists() else None
        if state == "inventory" and not checked_inventory:
            with pytest.raises(RuntimeError, match="inventory"):
                create_group(cfg, "during-audit")
            with pytest.raises(RuntimeError, match="inventory|sealed"):
                submit(cfg, ["true"], group="existing")
            checked_inventory = True
        if state == "building":
            with pytest.raises(RuntimeError, match="sealed"):
                submit(cfg, ["true"], group="existing")
            if not checked_audit:
                create_group(cfg, "during-audit")
                expected.add("during-audit")
                checked_audit = True
        # Incomplete bootstrap still serves every published Group.
        page = observer.list_groups_page(cfg, page_size=1000)
        assert {g["group"]["name"] for g in page["items"]} == expected
    else:
        pytest.fail("bounded conversion never completed")
    assert checked_inventory and checked_audit
    assert starts.count(0) == 1
    assert index_active(root)
    assert "admission_state" not in read_json(directory / "existing.json")["group"]
    for all_groups in (False, True):
        groups = observer.list_groups_page(cfg, all_groups=all_groups, page_size=1000)["items"]
        assert [g["group"]["name"] for g in groups] == sorted(expected)
    replay = submit(cfg, ["true"], group="existing", idempotency_key="cutover")
    assert (replay.task_id, replay.submission_operation_id) == (original.task_id, original.submission_operation_id)
    assert submit(cfg, ["true"], group="existing").group_membership_sequence == 2


def test_remounted_project_with_completed_historical_journal_resumes(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.agent.context import MachineRuntime
    from qqtools.plugins.qexp.agent.setup import initialize_machine
    from qqtools.plugins.qexp.runtime.upgrade.machine import discover_registered_upgrades

    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "local")
    root = cfg.shared_root
    (root / "operations/upgrades/protocol-manifest.json").unlink()
    coordinator = UpgradeCoordinator(cfg)
    coordinator.discover()
    for _ in range(12):
        if coordinator.advance()["state"] == "completed":
            break
    assert coordinator.status()["state"] == "completed"
    schema_path = root / "schema/version.json"
    schema = read_json(schema_path)
    schema["schema"]["required_capabilities"].remove(GROUP_WORKING_SET_CAPABILITY)
    atomic_replace(schema_path, schema)
    (root / "schema" / f"{GROUP_WORKING_SET_CAPABILITY}.json").unlink()
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "g1")
    binding = runtime.add_binding(root, cfg.machine_name)
    with monkeypatch.context() as outage:

        def unavailable(*_args, **_kwargs):
            raise OSError("Project offline")

        outage.setattr(UpgradeCoordinator, "discover", unavailable)
        assert binding.project_id in discover_registered_upgrades(runtime)["pending_project_ids"]

    runtime.upgrade_probe_deadlines[binding.project_id] = 0.0
    discover_registered_upgrades(runtime)
    assert binding.project_id in runtime.upgrade_runnable_projects


def test_large_released_group_survives_interrupted_stream_copy(tmp_path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "machine")
    root = cfg.shared_root
    create_group(cfg, "large")
    file = group_directory(root) / "large.json"
    group = read_json(file)
    # The first field forces admission and provenance past the first 8 MiB.
    group["group"].update(aaa_note="漢" * (3 * 1024 * 1024), admission_state="sealed")
    atomic_replace(file, group)
    schema_path = root / "schema/version.json"
    schema = read_json(schema_path)
    schema["schema"]["required_capabilities"].remove(GROUP_WORKING_SET_CAPABILITY)
    atomic_replace(schema_path, schema)
    (root / "schema" / f"{GROUP_WORKING_SET_CAPABILITY}.json").unlink()
    shutil.rmtree(root / "indexes/group-working-set")
    registry = MigrationRegistry([GroupWorkingSetMigration()])
    UpgradeCoordinator(cfg, registry=registry).discover()
    from qqtools.plugins.qexp.runtime.upgrade import group_working_set as conversion

    real_publish = conversion.publish_group_copy
    lost_at_publication = False

    def lose_final_stage_once(context, path, checkpoint):
        nonlocal lost_at_publication
        if not lost_at_publication:
            checkpoint.with_suffix(".stage").unlink()
            lost_at_publication = True
        return real_publish(context, path, checkpoint)

    monkeypatch.setattr(conversion, "publish_group_copy", lose_final_stage_once)
    offsets = set()
    damaged_stage = False
    for _ in range(50):
        result = UpgradeCoordinator(cfg, registry=registry).advance(force_retry=True)
        assert result["state"] != "repair_required", result
        checkpoint = root / "operations/upgrades" / GROUP_WORKING_SET_CAPABILITY / "group-checkpoint.json"
        if checkpoint.exists():
            offsets.add(read_json(checkpoint)["offset"])
            if not damaged_stage:
                from qqtools.plugins.qexp.commands.group import group_control

                with pytest.raises(RuntimeError, match="being converted"):
                    group_control(cfg, "large", "pause")
                create_group(cfg, "writable")
                checkpoint.with_suffix(".stage").unlink()
                damaged_stage = True
        if result["state"] == "completed":
            break
    assert result["state"] == "completed"
    assert damaged_stage and lost_at_publication and len(offsets) >= 9
    del group["group"]["admission_state"]
    assert read_json(file) == group
    assert observer.list_groups_page(cfg)["items"][0]["group"]["name"] == "large"


def test_submission_projection_resumes_across_mount_devices_and_detects_replacement(tmp_path, monkeypatch):
    import json
    from dataclasses import replace

    from qqtools.plugins.qexp.runtime.group_discovery.activation_submission import advance_activation_submission_source
    from qqtools.plugins.qexp.runtime.upgrade import UpgradeSliceBudget, UpgradeStorage

    path = tmp_path / "op.json"
    checkpoint = tmp_path / "checkpoint.json"

    def write(state):
        record = {"padding": "x" * 90000, "submission": {"operation_id": "op", "state": state, "target_group": "g"}}
        temp = tmp_path / "replacement"
        temp.write_text(json.dumps(record))
        temp.replace(path)

    write("preparing")
    original = UpgradeStorage.read_regular_bytes
    count = 0

    def alternate_mount(storage, *args):
        nonlocal count
        data = original(storage, *args)
        count += 1
        return replace(data, revision=replace(data.revision, device=count % 2))

    monkeypatch.setattr(UpgradeStorage, "read_regular_bytes", alternate_mount)

    def step():
        storage = UpgradeStorage(UpgradeSliceBudget(1, 128, 262144))
        return advance_activation_submission_source(
            storage, path, "op", checkpoint, "submissions", mount_independent=True
        )

    assert step().state == "progressed"
    write("committed")
    for _ in range(14):
        result = step()
        if result.state != "progressed":
            break
    assert result.submission_state == "committed" and result.target_group == "g"
