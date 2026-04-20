from __future__ import annotations

import json

import pytest

from qqtools.plugins.qexp.layout import (
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
    ensure_shared_layout,
    ensure_machine_layout,
    ensure_runtime_layout,
    read_root_manifest,
    read_schema_version,
    root_manifest_path,
    schema_version_path,
    summary_state_path,
    task_path,
    validate_root_contract,
    write_root_manifest,
    write_schema_version,
)
from qqtools.plugins.qexp.models import SCHEMA_VERSION


# ---------------------------------------------------------------------------
# RootConfig
# ---------------------------------------------------------------------------


class TestRootConfig:
    def test_construction(self, tmp_path):
        cfg = RootConfig(
            shared_root=tmp_path / ".qexp",
            project_root=tmp_path,
            machine_name="gpu1",
            runtime_root=tmp_path / "runtime",
        )
        assert cfg.machine_name == "gpu1"
        assert cfg.shared_root.is_absolute()
        assert cfg.project_root == tmp_path.resolve()
        assert cfg.runtime_root.is_absolute()

    def test_invalid_machine_name(self, tmp_path):
        with pytest.raises(ValueError):
            RootConfig(
                shared_root=tmp_path / ".qexp",
                project_root=tmp_path,
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
            shared_root=tmp_path / ".qexp",
            project_root=tmp_path,
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
            shared_root=tmp_path / ".qexp",
            project_root=tmp_path,
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
            shared_root=tmp_path / ".qexp",
            project_root=tmp_path,
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
        cfg = load_root_config(tmp_path / ".qexp", "gpu1")
        assert cfg.machine_name == "gpu1"
        assert cfg.project_root == tmp_path.resolve()
        assert "qexp-runtime" in str(cfg.runtime_root)

    def test_default_runtime_includes_machine_name(self):
        """Default runtime_root must be isolated per machine."""
        cfg1 = load_root_config("/tmp/.qexp", "m1")
        cfg2 = load_root_config("/tmp/.qexp", "m2")
        assert cfg1.runtime_root != cfg2.runtime_root
        assert "m1" in str(cfg1.runtime_root)
        assert "m2" in str(cfg2.runtime_root)

    def test_custom_runtime(self, tmp_path):
        cfg = load_root_config(tmp_path / ".qexp", "m1", tmp_path / "rt")
        assert cfg.runtime_root == (tmp_path / "rt").resolve()

    def test_rejects_non_project_control_root(self, tmp_path):
        with pytest.raises(ValueError, match="named '.qexp'"):
            load_root_config(tmp_path / "shared", "gpu1")


# ---------------------------------------------------------------------------
# init_shared_root
# ---------------------------------------------------------------------------


class TestInitSharedRoot:
    def test_creates_full_layout(self, tmp_path):
        cfg = init_shared_root(tmp_path / ".qexp", "gpu2a")
        assert global_tasks_dir(cfg).is_dir()
        assert machine_state_dir(cfg).is_dir()
        assert cfg.runtime_root.is_dir()
        assert read_schema_version(cfg) == SCHEMA_VERSION
        assert root_manifest_path(cfg).is_file()
        assert read_root_manifest(cfg).project_root == str(tmp_path.resolve())

    def test_writes_machine_json(self, tmp_path):
        cfg = init_shared_root(tmp_path / ".qexp", "gpu2a")
        mpath = machine_json_path(cfg)
        assert mpath.is_file()
        data = json.loads(mpath.read_text(encoding="utf-8"))
        assert data["machine"]["machine_name"] == "gpu2a"
        assert data["machine"]["agent_mode"] == "on_demand"
        assert "agent_state" not in data["machine"]
        assert "last_heartbeat" not in data["machine"]

    def test_persistent_mode(self, tmp_path):
        cfg = init_shared_root(
            tmp_path / ".qexp", "gpu3", agent_mode="persistent"
        )
        data = json.loads(machine_json_path(cfg).read_text(encoding="utf-8"))
        assert data["machine"]["agent_mode"] == "persistent"

    def test_invalid_agent_mode(self, tmp_path):
        with pytest.raises(ValueError):
            init_shared_root(tmp_path / ".qexp", "gpu1", agent_mode="bad")

    def test_shared_mode_requires_machine(self, tmp_path):
        with pytest.raises(ValueError):
            init_shared_root(tmp_path / ".qexp", "")

    def test_validate_root_contract_detects_forbidden_truth_dirs(self, tmp_path):
        cfg = init_shared_root(tmp_path / ".qexp", "gpu2a")
        forbidden = cfg.shared_root / "groups" / "exp1" / "tasks"
        forbidden.mkdir(parents=True)
        with pytest.raises(ValueError, match="Forbidden truth-layout"):
            validate_root_contract(cfg)

    def test_repeated_init_preserves_root_manifest_identity(self, tmp_path):
        cfg1 = init_shared_root(tmp_path / ".qexp", "gpu2a")
        manifest1 = read_root_manifest(cfg1)

        cfg2 = init_shared_root(tmp_path / ".qexp", "gpu3")
        manifest2 = read_root_manifest(cfg2)

        assert manifest2.control_plane_id == manifest1.control_plane_id
        assert manifest2.created_at == manifest1.created_at
        assert manifest2.created_by_machine == manifest1.created_by_machine

    def test_require_initialized_root_rejects_uninitialized_qexp_dir(self, tmp_path):
        cfg = RootConfig(
            shared_root=tmp_path / ".qexp",
            project_root=tmp_path,
            machine_name="dev1",
            runtime_root=tmp_path / "runtime",
        )
        ensure_shared_layout(cfg)
        ensure_machine_layout(cfg)
        ensure_runtime_layout(cfg)

        with pytest.raises(FileNotFoundError, match="Root manifest not found"):
            load_root_config(tmp_path / ".qexp", "dev1", require_initialized=True)
