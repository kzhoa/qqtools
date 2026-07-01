from __future__ import annotations

import pytest
import yaml

from qqtools.plugins.qexp.api import batch_submit, submit
from qqtools.plugins.qexp.agent import start_agent_record
from qqtools.plugins.qexp.layout import init_shared_root
from qqtools.plugins.qexp.models import BATCH_COMMIT_PREPARING, PHASE_QUEUED
from qqtools.plugins.qexp.observer import (
    inspect_batch,
    inspect_task,
    list_batches,
    list_machines,
    list_tasks,
    top_view,
)


@pytest.fixture()
def cfg(tmp_path):
    return init_shared_root(tmp_path / ".qexp", "dev1", runtime_root=tmp_path / "runtime")


class TestListTasks:
    def test_empty(self, cfg):
        assert list_tasks(cfg) == []

    def test_all_tasks(self, cfg):
        submit(cfg, command=["echo", "1"])
        submit(cfg, command=["echo", "2"])
        tasks = list_tasks(cfg)
        assert len(tasks) == 2

    def test_filter_by_phase(self, cfg):
        submit(cfg, command=["echo"])
        tasks = list_tasks(cfg, phase=PHASE_QUEUED)
        assert len(tasks) == 1

    def test_filter_by_machine(self, cfg):
        submit(cfg, command=["echo"])
        tasks = list_tasks(cfg, machine="dev1")
        assert len(tasks) == 1
        assert list_tasks(cfg, machine="other") == []

    def test_filter_by_group(self, cfg):
        submit(cfg, command=["echo"], group="contract_n_4and6")
        submit(cfg, command=["echo"], group="regrouped_debug")
        tasks = list_tasks(cfg, group="contract_n_4and6")
        assert len(tasks) == 1
        assert tasks[0]["group"] == "contract_n_4and6"

    def test_combines_group_and_phase_filters(self, cfg):
        submit(cfg, command=["echo"], group="contract_n_4and6")
        tasks = list_tasks(cfg, group="contract_n_4and6", phase=PHASE_QUEUED)
        assert len(tasks) == 1

    def test_invalid_group_filter_is_rejected(self, cfg):
        with pytest.raises(ValueError):
            list_tasks(cfg, group="../tasks_by_state/queued")

    def test_limit(self, cfg):
        for _ in range(10):
            submit(cfg, command=["echo"])
        tasks = list_tasks(cfg, limit=3)
        assert len(tasks) == 3

    def test_limit_preserves_index_order_under_additional_filters(self, cfg):
        submit(cfg, command=["echo"], task_id="z-last", group="contract_n_4and6")
        submit(cfg, command=["echo"], task_id="a-first", group="contract_n_4and6")
        submit(cfg, command=["echo"], task_id="m-middle", group="contract_n_4and6")
        tasks = list_tasks(cfg, phase=PHASE_QUEUED, group="contract_n_4and6", limit=2)
        assert [task["task_id"] for task in tasks] == ["z-last", "a-first"]


class TestInspectTask:
    def test_inspect(self, cfg):
        t = submit(cfg, command=["echo"], name="test")
        result = inspect_task(cfg, t.task_id)
        assert result["task"]["task_id"] == t.task_id
        assert result["task"]["name"] == "test"

    def test_missing(self, cfg):
        with pytest.raises(FileNotFoundError):
            inspect_task(cfg, "nosuch")


class TestBatchViews:
    def test_list_batches(self, cfg, tmp_path):
        manifest = tmp_path / "m.yaml"
        manifest.write_text(yaml.dump({
            "batch": {"name": "sweep"},
            "tasks": [{"command": ["echo"]}],
        }), encoding="utf-8")
        batch_submit(cfg, manifest)
        batches = list_batches(cfg)
        assert len(batches) == 1
        assert batches[0]["name"] == "sweep"

    def test_list_batches_includes_group(self, cfg, tmp_path):
        manifest = tmp_path / "m.yaml"
        manifest.write_text(yaml.dump({
            "batch": {"name": "sweep", "group": "contract_n_4and6"},
            "tasks": [{"command": ["echo"]}],
        }), encoding="utf-8")
        batch_submit(cfg, manifest)
        batches = list_batches(cfg)
        assert batches[0]["group"] == "contract_n_4and6"

    def test_inspect_batch(self, cfg, tmp_path):
        manifest = tmp_path / "m.yaml"
        manifest.write_text(yaml.dump({
            "tasks": [{"command": ["echo"]}],
        }), encoding="utf-8")
        b = batch_submit(cfg, manifest)
        result = inspect_batch(cfg, b.batch_id)
        assert result["batch"]["batch_id"] == b.batch_id

    def test_list_batches_hides_non_committed_batches(self, cfg, tmp_path):
        manifest = tmp_path / "m-hidden.yaml"
        manifest.write_text(yaml.dump({
            "tasks": [{"command": ["echo"]}],
        }), encoding="utf-8")
        batch = batch_submit(cfg, manifest)
        from qqtools.plugins.qexp.storage import save_batch

        batch.commit_state = BATCH_COMMIT_PREPARING
        save_batch(cfg, batch)

        assert list_batches(cfg) == []

    def test_inspect_batch_hides_non_committed_batch(self, cfg, tmp_path):
        manifest = tmp_path / "m-hidden2.yaml"
        manifest.write_text(yaml.dump({
            "tasks": [{"command": ["echo"]}],
        }), encoding="utf-8")
        batch = batch_submit(cfg, manifest)
        from qqtools.plugins.qexp.storage import save_batch

        batch.commit_state = BATCH_COMMIT_PREPARING
        save_batch(cfg, batch)

        with pytest.raises(FileNotFoundError, match="not committed"):
            inspect_batch(cfg, batch.batch_id)


class TestBatchLiveSummary:
    """Verify batch summary is computed from live task states, not stale."""

    def test_summary_reflects_phase_changes(self, cfg, tmp_path):
        from qqtools.plugins.qexp.indexes import update_index_on_phase_change
        from qqtools.plugins.qexp.models import PHASE_FAILED
        from qqtools.plugins.qexp.storage import cas_update_task, load_task

        manifest = tmp_path / "m.yaml"
        manifest.write_text(yaml.dump({
            "batch": {"name": "test"},
            "tasks": [
                {"command": ["echo", "1"]},
                {"command": ["echo", "2"]},
            ],
        }), encoding="utf-8")
        b = batch_submit(cfg, manifest)

        # Mark first task as failed
        t = load_task(cfg, b.task_ids[0])
        t.status.phase = PHASE_FAILED
        cas_update_task(cfg, t, t.meta.revision)
        update_index_on_phase_change(cfg, t.task_id, PHASE_QUEUED, PHASE_FAILED)

        # list_batches should show live summary
        batches = list_batches(cfg)
        assert batches[0]["failed"] == 1
        assert batches[0]["queued"] == 1
        assert batches[0]["total"] == 2

    def test_inspect_batch_live_summary(self, cfg, tmp_path):
        from qqtools.plugins.qexp.indexes import update_index_on_phase_change
        from qqtools.plugins.qexp.models import PHASE_SUCCEEDED
        from qqtools.plugins.qexp.storage import cas_update_task, load_task

        manifest = tmp_path / "m.yaml"
        manifest.write_text(yaml.dump({
            "tasks": [{"command": ["echo"]}],
        }), encoding="utf-8")
        b = batch_submit(cfg, manifest)

        t = load_task(cfg, b.task_ids[0])
        t.status.phase = PHASE_SUCCEEDED
        cas_update_task(cfg, t, t.meta.revision)
        update_index_on_phase_change(cfg, t.task_id, PHASE_QUEUED, PHASE_SUCCEEDED)

        result = inspect_batch(cfg, b.batch_id)
        assert result["batch"]["summary"]["succeeded"] == 1
        assert result["batch"]["summary"]["queued"] == 0


class TestMachineViews:
    def test_list_machines(self, cfg):
        start_agent_record(cfg)
        machines = list_machines(cfg)
        assert len(machines) == 1
        assert machines[0]["machine_name"] == "dev1"
        assert machines[0]["agent_state"] == "starting"
        assert machines[0]["gpu_status"] == "unknown"
        assert machines[0]["gpu_visible_count"] is None
        assert machines[0]["gpu_reserved_count"] is None
        assert machines[0]["gpu_free_count"] is None

    def test_agent_state_comes_from_agent_snapshot_not_summary(self, cfg):
        from qqtools.plugins.qexp.lifecycle import write_summary_snapshot
        from qqtools.plugins.qexp.models import MachineSummary

        start_agent_record(cfg)
        write_summary_snapshot(
            cfg,
            MachineSummary(
                machine_name="dev1",
                counts_by_phase={"queued": 99, "running": 88},
                updated_at="2026-04-14T00:00:00Z",
            ),
        )
        machines = list_machines(cfg)
        assert machines[0]["agent_state"] == "starting"
        assert machines[0]["counts_by_phase"]["queued"] == 99

    def test_remote_machine_is_not_forced_stale_by_local_pid_probe(self, cfg):
        from qqtools.plugins.qexp.layout import agent_state_path
        from qqtools.plugins.qexp.storage import write_atomic_json
        from qqtools.plugins.qexp.lifecycle import read_agent_snapshot

        other_cfg = init_shared_root(
            cfg.shared_root,
            "gpu2",
            runtime_root=cfg.runtime_root.parent / "runtime2",
        )
        start_agent_record(other_cfg)
        snapshot = read_agent_snapshot(other_cfg)
        snapshot.pid = 99999999
        snapshot.agent_state = "active"
        write_atomic_json(agent_state_path(other_cfg), snapshot.to_dict())

        machines = list_machines(cfg)
        remote = next(machine for machine in machines if machine["machine_name"] == "gpu2")
        assert remote["agent_state"] == "active"

    def test_gpu_counts_come_from_dynamic_gpu_snapshot(self, cfg):
        from qqtools.plugins.qexp.layout import gpu_state_path
        from qqtools.plugins.qexp.storage import write_atomic_json

        write_atomic_json(gpu_state_path(cfg), {
            "gpu_count": 4,
            "visible_gpu_ids": [0, 1, 2, 3],
            "reserved_gpu_ids": [1, 3],
            "task_to_gpu_ids": {"task-a": [1], "task-b": [3]},
            "backend": "stub",
            "probe_succeeded": True,
            "probe_error": None,
            "updated_at": "2099-01-01T00:00:00Z",
        })

        machines = list_machines(cfg)
        machine = machines[0]
        assert machine["gpu_status"] == "live"
        assert machine["gpu_visible_count"] == 4
        assert machine["gpu_reserved_count"] == 2
        assert machine["gpu_free_count"] == 2
        assert machine["gpu_backend"] == "stub"

    def test_gpu_snapshot_falls_back_to_machine_snapshot(self, cfg):
        from qqtools.plugins.qexp.models import GpuInventory
        from qqtools.plugins.qexp.storage import load_machine, save_machine

        machine = load_machine(cfg)
        machine.gpu_inventory = GpuInventory(count=3, visible_gpu_ids=[0, 1, 2])
        save_machine(cfg, machine)

        machines = list_machines(cfg)
        machine_view = machines[0]
        assert machine_view["gpu_status"] == "fallback"
        assert machine_view["gpu_visible_count"] == 3
        assert machine_view["gpu_reserved_count"] is None
        assert machine_view["gpu_free_count"] is None

    def test_gpu_snapshot_supports_cross_machine_summary(self, cfg):
        from qqtools.plugins.qexp.layout import gpu_state_path
        from qqtools.plugins.qexp.storage import write_atomic_json

        other_cfg = init_shared_root(
            cfg.shared_root,
            "gpu2",
            runtime_root=cfg.runtime_root.parent / "runtime2",
        )
        start_agent_record(other_cfg)
        write_atomic_json(gpu_state_path(other_cfg), {
            "gpu_count": 8,
            "visible_gpu_ids": list(range(8)),
            "reserved_gpu_ids": [0, 2, 4],
            "task_to_gpu_ids": {"r1": [0], "r2": [2], "r3": [4]},
            "backend": "stub",
            "probe_succeeded": True,
            "probe_error": None,
            "updated_at": "2099-01-01T00:00:00Z",
        })

        machines = list_machines(cfg)
        remote = next(machine for machine in machines if machine["machine_name"] == "gpu2")
        assert remote["agent_state"] == "starting"
        assert remote["gpu_visible_count"] == 8
        assert remote["gpu_reserved_count"] == 3
        assert remote["gpu_free_count"] == 5

    def test_gpu_error_uses_last_known_visible_count_without_reserved_or_free(self, cfg):
        from qqtools.plugins.qexp.layout import gpu_state_path
        from qqtools.plugins.qexp.models import GpuInventory
        from qqtools.plugins.qexp.storage import load_machine, save_machine, write_atomic_json

        machine = load_machine(cfg)
        machine.gpu_inventory = GpuInventory(count=4, visible_gpu_ids=[0, 1, 2, 3])
        save_machine(cfg, machine)

        write_atomic_json(gpu_state_path(cfg), {
            "gpu_count": None,
            "visible_gpu_ids": [],
            "reserved_gpu_ids": [],
            "task_to_gpu_ids": {},
            "backend": "pynvml",
            "probe_succeeded": False,
            "probe_error": "nvml exploded",
            "updated_at": "2099-01-01T00:00:00Z",
        })

        machines = list_machines(cfg)
        machine_view = machines[0]
        assert machine_view["gpu_status"] == "error"
        assert machine_view["gpu_visible_count"] == 4
        assert machine_view["gpu_reserved_count"] is None
        assert machine_view["gpu_free_count"] is None
        assert machine_view["gpu_probe_error"] == "nvml exploded"

    def test_malformed_gpu_snapshot_degrades_to_fallback_instead_of_crashing(self, cfg):
        from qqtools.plugins.qexp.layout import gpu_state_path
        from qqtools.plugins.qexp.models import GpuInventory
        from qqtools.plugins.qexp.storage import load_machine, save_machine, write_atomic_json

        machine = load_machine(cfg)
        machine.gpu_inventory = GpuInventory(count=2, visible_gpu_ids=[0, 1])
        save_machine(cfg, machine)

        write_atomic_json(gpu_state_path(cfg), {
            "gpu_count": 2,
            "visible_gpu_ids": [0, 1],
            "reserved_gpu_ids": 1,
            "task_to_gpu_ids": {},
            "backend": "stub",
            "probe_succeeded": True,
            "probe_error": None,
            "updated_at": "2099-01-01T00:00:00Z",
        })

        machines = list_machines(cfg)
        machine_view = machines[0]
        assert machine_view["gpu_status"] == "fallback"
        assert machine_view["gpu_visible_count"] == 2
        assert machine_view["gpu_reserved_count"] is None
        assert machine_view["gpu_free_count"] is None


class TestTopView:
    def test_basic(self, cfg):
        submit(cfg, command=["echo"])
        view = top_view(cfg)
        assert view["counts"][PHASE_QUEUED] == 1
        assert len(view["machines"]) == 1

    def test_all_machines(self, cfg):
        view = top_view(cfg, all_machines=True)
        assert len(view["machines"]) >= 1
