from __future__ import annotations

import json

import pytest

from qqtools.plugins.qexp.v2.layout import (
    RootConfig,
    agent_state_path,
    batch_path,
    ensure_machine_layout,
    ensure_shared_layout,
    global_batches_dir,
    global_events_dir,
    global_locks_dir,
    global_tasks_dir,
    gpu_state_path,
    index_by_batch_dir,
    index_by_machine_dir,
    index_by_state_dir,
    init_shared_root,
    load_root_config,
    machine_claims_active_dir,
    machine_claims_released_dir,
    machine_dir,
    machine_events_dir,
    machine_json_path,
    machine_state_dir,
    read_schema_version,
    schema_version_path,
    summary_state_path,
    task_path,
    write_schema_version,
)
from qqtools.plugins.qexp.v2.models import SCHEMA_VERSION


# ---------------------------------------------------------------------------
# RootConfig
# ---------------------------------------------------------------------------


class TestRootConfig:
    def test_construction(self, tmp_path):
        cfg = RootConfig(
            shared_root=tmp_path / "shared",
            machine_name="gpu1",
            runtime_root=tmp_path / "runtime",
        )
        assert cfg.machine_name == "gpu1"
        assert cfg.shared_root.is_absolute()
        assert cfg.runtime_root.is_absolute()

    def test_invalid_machine_name(self, tmp_path):
        with pytest.raises(ValueError):
            RootConfig(
                shared_root=tmp_path / "shared",
                machine_name="../evil",
                runtime_root=tmp_path / "runtime",
            )


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


class TestPathHelpers:
    @pytest.fixture()
    def cfg(self, tmp_path):
        return RootConfig(
            shared_root=tmp_path / "shared",
            machine_name="dev1",
            runtime_root=tmp_path / "runtime",
        )

    def test_global_tasks_dir(self, cfg):
        assert global_tasks_dir(cfg) == cfg.shared_root / "global" / "tasks"

    def test_global_batches_dir(self, cfg):
        assert global_batches_dir(cfg) == cfg.shared_root / "global" / "batches"

    def test_global_events_dir(self, cfg):
        assert global_events_dir(cfg) == cfg.shared_root / "global" / "events"

    def test_global_locks_dir(self, cfg):
        assert global_locks_dir(cfg) == cfg.shared_root / "global" / "locks"

    def test_machine_dir(self, cfg):
        assert machine_dir(cfg) == cfg.shared_root / "machines" / "dev1"

    def test_machine_state_dir(self, cfg):
        assert machine_state_dir(cfg) == machine_dir(cfg) / "state"

    def test_machine_claims(self, cfg):
        assert machine_claims_active_dir(cfg) == machine_dir(cfg) / "claims" / "active"
        assert machine_claims_released_dir(cfg) == machine_dir(cfg) / "claims" / "released"

    def test_task_path(self, cfg):
        assert task_path(cfg, "t1") == global_tasks_dir(cfg) / "t1.json"

    def test_batch_path(self, cfg):
        assert batch_path(cfg, "b1") == global_batches_dir(cfg) / "b1.json"

    def test_machine_json_path(self, cfg):
        assert machine_json_path(cfg) == machine_dir(cfg) / "machine.json"

    def test_index_dirs(self, cfg):
        base = cfg.shared_root / "global" / "indexes"
        assert index_by_state_dir(cfg) == base / "tasks_by_state"
        assert index_by_batch_dir(cfg) == base / "tasks_by_batch"
        assert index_by_machine_dir(cfg) == base / "tasks_by_machine"

    def test_state_file_paths(self, cfg):
        sd = machine_state_dir(cfg)
        assert agent_state_path(cfg) == sd / "agent.json"
        assert gpu_state_path(cfg) == sd / "gpu.json"
        assert summary_state_path(cfg) == sd / "summary.json"


# ---------------------------------------------------------------------------
# Layout creation
# ---------------------------------------------------------------------------


class TestEnsureLayout:
    @pytest.fixture()
    def cfg(self, tmp_path):
        return RootConfig(
            shared_root=tmp_path / "shared",
            machine_name="dev1",
            runtime_root=tmp_path / "runtime",
        )

    def test_ensure_shared_layout(self, cfg):
        ensure_shared_layout(cfg)
        assert global_tasks_dir(cfg).is_dir()
        assert global_batches_dir(cfg).is_dir()
        assert global_locks_dir(cfg).is_dir()
        assert global_events_dir(cfg).is_dir()
        assert index_by_state_dir(cfg).is_dir()
        assert index_by_batch_dir(cfg).is_dir()
        assert index_by_machine_dir(cfg).is_dir()

    def test_ensure_machine_layout(self, cfg):
        ensure_machine_layout(cfg)
        assert machine_claims_active_dir(cfg).is_dir()
        assert machine_claims_released_dir(cfg).is_dir()
        assert machine_events_dir(cfg).is_dir()
        assert machine_state_dir(cfg).is_dir()

    def test_idempotent(self, cfg):
        ensure_shared_layout(cfg)
        ensure_shared_layout(cfg)
        assert global_tasks_dir(cfg).is_dir()


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------


class TestSchemaVersion:
    @pytest.fixture()
    def cfg(self, tmp_path):
        return RootConfig(
            shared_root=tmp_path / "shared",
            machine_name="dev1",
            runtime_root=tmp_path / "runtime",
        )

    def test_write_and_read(self, cfg):
        ensure_shared_layout(cfg)
        write_schema_version(cfg)
        assert read_schema_version(cfg) == SCHEMA_VERSION

    def test_read_missing(self, cfg):
        assert read_schema_version(cfg) is None


# ---------------------------------------------------------------------------
# load_root_config
# ---------------------------------------------------------------------------


class TestLoadRootConfig:
    def test_default_runtime(self, tmp_path):
        cfg = load_root_config(tmp_path / "shared", "gpu1")
        assert cfg.machine_name == "gpu1"
        assert "qexp-runtime" in str(cfg.runtime_root)

    def test_default_runtime_includes_machine_name(self):
        """Default runtime_root must be isolated per machine."""
        cfg1 = load_root_config("/tmp/shared", "m1")
        cfg2 = load_root_config("/tmp/shared", "m2")
        assert cfg1.runtime_root != cfg2.runtime_root
        assert "m1" in str(cfg1.runtime_root)
        assert "m2" in str(cfg2.runtime_root)

    def test_custom_runtime(self, tmp_path):
        cfg = load_root_config(tmp_path / "s", "m1", tmp_path / "rt")
        assert cfg.runtime_root == (tmp_path / "rt").resolve()


# ---------------------------------------------------------------------------
# init_shared_root
# ---------------------------------------------------------------------------


class TestInitSharedRoot:
    def test_creates_full_layout(self, tmp_path):
        cfg = init_shared_root(tmp_path / "shared", "gpu2a")
        assert global_tasks_dir(cfg).is_dir()
        assert machine_state_dir(cfg).is_dir()
        assert cfg.runtime_root.is_dir()
        assert read_schema_version(cfg) == SCHEMA_VERSION

    def test_writes_machine_json(self, tmp_path):
        cfg = init_shared_root(tmp_path / "shared", "gpu2a")
        mpath = machine_json_path(cfg)
        assert mpath.is_file()
        data = json.loads(mpath.read_text(encoding="utf-8"))
        assert data["machine"]["machine_name"] == "gpu2a"
        assert data["machine"]["agent_mode"] == "on_demand"
        assert "agent_state" not in data["machine"]
        assert "last_heartbeat" not in data["machine"]

    def test_persistent_mode(self, tmp_path):
        cfg = init_shared_root(
            tmp_path / "shared", "gpu3", agent_mode="persistent"
        )
        data = json.loads(machine_json_path(cfg).read_text(encoding="utf-8"))
        assert data["machine"]["agent_mode"] == "persistent"

    def test_invalid_agent_mode(self, tmp_path):
        with pytest.raises(ValueError):
            init_shared_root(tmp_path / "shared", "gpu1", agent_mode="bad")

    def test_shared_mode_requires_machine(self, tmp_path):
        with pytest.raises(ValueError):
            init_shared_root(tmp_path / "shared", "")
