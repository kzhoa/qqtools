from __future__ import annotations

import pytest
import yaml

from qqtools.plugins.qexp.v2.api import batch_submit, submit
from qqtools.plugins.qexp.v2.agent import start_agent_record
from qqtools.plugins.qexp.v2.layout import init_shared_root
from qqtools.plugins.qexp.v2.models import PHASE_QUEUED
from qqtools.plugins.qexp.v2.observer import (
    inspect_batch,
    inspect_task,
    list_batches,
    list_machines,
    list_tasks,
    top_view,
)


@pytest.fixture()
def cfg(tmp_path):
    return init_shared_root(tmp_path / "shared", "dev1", runtime_root=tmp_path / "runtime")


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

    def test_limit(self, cfg):
        for _ in range(10):
            submit(cfg, command=["echo"])
        tasks = list_tasks(cfg, limit=3)
        assert len(tasks) == 3


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

    def test_inspect_batch(self, cfg, tmp_path):
        manifest = tmp_path / "m.yaml"
        manifest.write_text(yaml.dump({
            "tasks": [{"command": ["echo"]}],
        }), encoding="utf-8")
        b = batch_submit(cfg, manifest)
        result = inspect_batch(cfg, b.batch_id)
        assert result["batch"]["batch_id"] == b.batch_id


class TestBatchLiveSummary:
    """Verify batch summary is computed from live task states, not stale."""

    def test_summary_reflects_phase_changes(self, cfg, tmp_path):
        from qqtools.plugins.qexp.v2.indexes import update_index_on_phase_change
        from qqtools.plugins.qexp.v2.models import PHASE_FAILED
        from qqtools.plugins.qexp.v2.storage import cas_update_task, load_task

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
        from qqtools.plugins.qexp.v2.indexes import update_index_on_phase_change
        from qqtools.plugins.qexp.v2.models import PHASE_SUCCEEDED
        from qqtools.plugins.qexp.v2.storage import cas_update_task, load_task

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

    def test_agent_state_comes_from_agent_snapshot_not_summary(self, cfg):
        from qqtools.plugins.qexp.v2.lifecycle import write_summary_snapshot
        from qqtools.plugins.qexp.v2.models import MachineSummary

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
        from qqtools.plugins.qexp.v2.layout import agent_state_path
        from qqtools.plugins.qexp.v2.storage import write_atomic_json
        from qqtools.plugins.qexp.v2.lifecycle import read_agent_snapshot

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


class TestTopView:
    def test_basic(self, cfg):
        submit(cfg, command=["echo"])
        view = top_view(cfg)
        assert view["counts"][PHASE_QUEUED] == 1
        assert len(view["machines"]) == 1

    def test_all_machines(self, cfg):
        view = top_view(cfg, all_machines=True)
        assert len(view["machines"]) >= 1
