from dataclasses import asdict, replace
from pathlib import Path

import pytest

from qqtools.plugins.qexp.agent.bindings import ProjectBinding
from qqtools.plugins.qexp.agent.inventory import ProjectInventoryEntry
from qqtools.plugins.qexp.config_types import RootConfig


def test_persisted_configuration_reconstruction_never_resolves_project_paths(monkeypatch):
    def forbid_resolution(*_args, **_kwargs):
        raise AssertionError("controller entered the shared filesystem")

    monkeypatch.setattr(Path, "resolve", forbid_resolution)
    binding = ProjectBinding.from_dict(
        {
            "project_id": "project-a",
            "shared_root": "/projects/a/.qexp",
            "machine_name": "gpu-1",
            "enabled": True,
        }
    )
    entry = ProjectInventoryEntry.from_dict(
        {
            "project_id": "project-a",
            "shared_root": "/projects/a/.qexp",
            "enabled": True,
        }
    )
    cfg = RootConfig.from_canonical_paths(
        binding.shared_root, binding.shared_root.parent, binding.machine_name, Path("/runtime/project-a")
    )
    disabled = replace(binding, enabled=False, _canonical_paths=True)
    assert entry.shared_root == cfg.shared_root == binding.shared_root
    assert disabled.enabled is False
    assert "_canonical_paths" not in asdict(binding)
    assert set(asdict(cfg)) == {"shared_root", "project_root", "machine_name", "runtime_root"}


@pytest.mark.parametrize("root", ["relative/.qexp", "/projects/../a/.qexp", "/projects/a\x00/.qexp"])
def test_persisted_noncanonical_project_path_fails_without_resolution(monkeypatch, root):
    def forbid_resolution(*_args, **_kwargs):
        raise AssertionError("invalid persisted path triggered filesystem access")

    monkeypatch.setattr(Path, "resolve", forbid_resolution)
    with pytest.raises(ValueError):
        ProjectBinding.from_dict(
            {
                "project_id": "project-a",
                "shared_root": root,
                "machine_name": "gpu-1",
                "enabled": True,
            }
        )
    with pytest.raises(ValueError):
        ProjectInventoryEntry.from_dict({"project_id": "project-a", "shared_root": root, "enabled": True})
    with pytest.raises(ValueError):
        RootConfig.from_canonical_paths(Path(root), Path("/projects/a"), "gpu-1", Path("/runtime/a"))


def test_explicit_configuration_construction_still_resolves_symlinks(tmp_path):
    target = tmp_path / "project"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    cfg = RootConfig(alias / ".qexp", alias, "gpu-1", tmp_path / "runtime")
    binding = ProjectBinding("project-a", alias / ".qexp", "gpu-1")
    inventory = ProjectInventoryEntry("project-a", alias / ".qexp")
    assert cfg.shared_root == binding.shared_root == inventory.shared_root == target / ".qexp"
    assert cfg.project_root == target
