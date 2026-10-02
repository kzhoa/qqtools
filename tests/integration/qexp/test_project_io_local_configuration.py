from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.inventory import ProjectInventoryEntry, load_inventory, save_inventory_locked
from qqtools.plugins.qexp.agent.project_io_controller import ProjectIOController
from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor
from qqtools.plugins.qexp.agent.setup import initialize_machine, set_project_enablement

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def test_registry_inventory_enablement_and_config_consumption_are_machine_local(tmp_path, monkeypatch):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "gpu-1")
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    with runtime.inventory_guard():
        save_inventory_locked(runtime, 1, [ProjectInventoryEntry(binding.project_id, binding.shared_root)])
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    original_resolve = Path.resolve

    def guarded_resolve(path, *args, **kwargs):
        if path == cfg.project_root or path.is_relative_to(cfg.project_root):
            raise AssertionError("machine-local metadata reconstruction resolved the shared Project")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", guarded_resolve)
    revision, bindings = runtime.load_registry_uncached()
    assert bindings == [binding]
    _inventory_revision, entries = load_inventory(runtime)
    assert entries[0].shared_root == cfg.shared_root
    consumed_cfg = controller._root_config(bindings[0])
    assert consumed_cfg.shared_root == cfg.shared_root
    assert consumed_cfg.runtime_root == runtime.project_paths(binding.project_id)["root"]
    result = set_project_enablement(runtime, binding.project_id, False)
    assert result["status"] == "committed"
    assert load_inventory(runtime)[1][0].enabled is False
    assert runtime.load_registry_uncached()[0] == revision + 1
    disabled = runtime.set_enabled(binding.project_id, False)
    assert disabled.enabled is False
    assert runtime.load_registry_uncached()[0] == revision + 2
    executor.shutdown()
