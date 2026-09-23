"""QQTOOLS-COMPAT-0016: Registered Project notification migration is bounded and resumable."""

import pytest

from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.machine_config import init_shared_root
from qqtools.plugins.qexp.notification_config import update_notifications
from qqtools.plugins.qexp.notification_migration import advance_notification_migration, notification_migration_status
from qqtools.plugins.qexp.notification_policy import load_policy

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _project(tmp_path, runtime, name):
    cfg = init_shared_root(tmp_path / name / ".qexp", "gpu-1", runtime_root=tmp_path / name / "local")
    runtime.ensure_binding(cfg.shared_root, cfg.machine_name)
    binding = runtime.resolve_project_binding(cfg.shared_root)
    update_notifications(
        cfg,
        lambda current: {
            "enabled": True,
            "providers": {"feishu": {"enabled": True, "webhook_env": "FEISHU_WEBHOOK"}},
        },
    )
    return cfg, binding


def test_registered_projects_import_automatically_in_bounded_slices(tmp_path):
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    projects = [_project(tmp_path, runtime, name) for name in ("first", "second")]
    assert notification_migration_status(runtime)["pending"]["count"] == 2

    first_pass = advance_notification_migration(runtime, limit=1)
    assert first_pass["inspected"]["count"] == 1
    assert first_pass["pending"]["count"] >= 1
    second_pass = advance_notification_migration(MachineRuntime(runtime.root), limit=1)
    assert second_pass["inspected"]["count"] == 1
    assert second_pass["inspected"]["ids"] != first_pass["inspected"]["ids"]
    assert notification_migration_status(runtime)["ready"]["count"] == 2
    for _cfg, binding in projects:
        assert load_policy(runtime.root, "project", binding.project_id)["override"]["enabled"] is True


def test_inaccessible_project_remains_pending_until_mount_returns(tmp_path):
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    cfg, binding = _project(tmp_path, runtime, "first")
    missing = cfg.shared_root.with_name("temporarily-unmounted")
    cfg.shared_root.rename(missing)
    try:
        result = advance_notification_migration(runtime, limit=1)
        assert binding.project_id in result["inaccessible"]["ids"]
        assert binding.project_id in result["pending"]["ids"]
    finally:
        missing.rename(cfg.shared_root)

    result = advance_notification_migration(runtime, limit=1)
    assert binding.project_id in result["imported"]["ids"]
    assert binding.project_id in notification_migration_status(runtime)["ready"]["ids"]
