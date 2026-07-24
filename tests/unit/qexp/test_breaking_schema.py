from pathlib import Path

import pytest

from qqtools.plugins.qexp.layout import load_root_config, validate_root_contract
from qqtools.plugins.qexp.machine_config import init_shared_root
from qqtools.plugins.qexp.runtime.paths import shared_paths
from qqtools.plugins.qexp.runtime.store import atomic_replace


def test_old_schema_fails_before_mutation(tmp_path: Path):
    root = tmp_path / ".qexp"
    root.mkdir()
    (root / "schema").mkdir()
    atomic_replace(root / "schema" / "version.json", {"schema": {"version": "4.1"}})
    cfg = load_root_config(root, "g1", tmp_path / "rt")
    with pytest.raises(RuntimeError, match="Unsupported qexp schema"):
        validate_root_contract(cfg)
    assert not (root / "tasks").exists()


def test_init_writes_schema5_layout(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    paths = shared_paths(cfg.shared_root)
    assert paths["groups"].is_dir()
    assert paths["submissions"].is_dir()
    assert (paths["schema"] / "version.json").read_text()
