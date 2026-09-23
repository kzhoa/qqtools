"""QQTOOLS-COMPAT-0016: Mixed-version writes reconcile without silent robot changes."""

import json

import pytest

from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.machine_config import init_shared_root
from qqtools.plugins.qexp.notification_cleanup import cleanup_credentials
from qqtools.plugins.qexp.notification_config import (
    shared_feishu_webhook_path,
    update_notifications,
    write_shared_feishu_webhook,
)
from qqtools.plugins.qexp.notification_credentials import credential_path
from qqtools.plugins.qexp.notification_policy import load_policy, replace_policy
from qqtools.plugins.qexp.notification_reconciliation import (
    LegacyConflictError,
    reconcile_legacy,
    resolve_legacy_conflict,
)
from qqtools.plugins.qexp.notification_resolver import resolve_delivery


def _fixture(tmp_path, monkeypatch):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "local")
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.root.mkdir()
    monkeypatch.setattr(
        "qqtools.plugins.qexp.notification_reconciliation._project_id",
        lambda _runtime, _cfg: "project-1",
    )
    return runtime, cfg


def _legacy_shared(cfg, webhook):
    update_notifications(
        cfg,
        lambda current: {
            "enabled": True,
            "providers": {"feishu": {"enabled": True, "credential_source": "shared_file"}},
        },
    )
    write_shared_feishu_webhook(cfg, webhook)


def test_initial_import_and_webhook_only_old_write(tmp_path, monkeypatch):
    runtime, cfg = _fixture(tmp_path, monkeypatch)
    first = "https://open.feishu.cn/open-apis/bot/v2/hook/one"
    second = "https://open.feishu.cn/open-apis/bot/v2/hook/two"
    _legacy_shared(cfg, first)

    assert reconcile_legacy(runtime, cfg) == "project-1"
    before = load_policy(runtime.root, "project", "project-1")
    assert resolve_delivery(runtime.root, "project-1")["webhook"] == first
    write_shared_feishu_webhook(cfg, second)

    assert reconcile_legacy(runtime, cfg) == "project-1"
    after = load_policy(runtime.root, "project", "project-1")
    assert after["revision"] == before["revision"] + 1
    assert resolve_delivery(runtime.root, "project-1")["webhook"] == second
    assert reconcile_legacy(runtime, cfg) == "project-1"
    assert load_policy(runtime.root, "project", "project-1")["revision"] == after["revision"]


def test_canonical_reset_then_old_write_requires_explicit_resolution(tmp_path, monkeypatch):
    runtime, cfg = _fixture(tmp_path, monkeypatch)
    reconcile_legacy(runtime, cfg)
    replace_policy(runtime.root, "project", 1, None, "project-1")
    _legacy_shared(cfg, "https://open.feishu.cn/open-apis/bot/v2/hook/older")

    with pytest.raises(LegacyConflictError, match="resolve"):
        reconcile_legacy(runtime, cfg)
    conflict = load_policy(runtime.root, "project", "project-1")
    assert conflict["override"] is None
    assert conflict["legacy"]["status"] == "legacy_conflict"
    with pytest.raises(LegacyConflictError):
        reconcile_legacy(runtime, cfg)

    resolve_legacy_conflict(runtime, cfg, prefer="canonical")
    assert reconcile_legacy(runtime, cfg) == "project-1"
    assert load_policy(runtime.root, "project", "project-1")["override"] is None


def test_legacy_resolution_imports_competing_old_write(tmp_path, monkeypatch):
    runtime, cfg = _fixture(tmp_path, monkeypatch)
    reconcile_legacy(runtime, cfg)
    replace_policy(runtime.root, "project", 1, {"enabled": False}, "project-1")
    _legacy_shared(cfg, "https://open.feishu.cn/open-apis/bot/v2/hook/older")

    with pytest.raises(LegacyConflictError):
        reconcile_legacy(runtime, cfg)
    resolve_legacy_conflict(runtime, cfg, prefer="legacy")
    assert resolve_delivery(runtime.root, "project-1")["webhook"].endswith("/older")


def test_invalid_legacy_source_is_durable_and_recovers_without_false_conflict(tmp_path, monkeypatch):
    runtime, cfg = _fixture(tmp_path, monkeypatch)
    _legacy_shared(cfg, "https://open.feishu.cn/open-apis/bot/v2/hook/one")
    shared_feishu_webhook_path(cfg).unlink()
    with pytest.raises(ValueError, match="unavailable or invalid"):
        reconcile_legacy(runtime, cfg)
    invalid = load_policy(runtime.root, "project", "project-1")
    assert invalid["legacy"]["status"] == "source_invalid"
    with pytest.raises(ValueError, match="unavailable or invalid"):
        reconcile_legacy(runtime, cfg)
    assert load_policy(runtime.root, "project", "project-1")["revision"] == invalid["revision"]

    write_shared_feishu_webhook(cfg, "https://open.feishu.cn/open-apis/bot/v2/hook/two")
    reconcile_legacy(runtime, cfg)
    assert resolve_delivery(runtime.root, "project-1")["webhook"].endswith("/two")


def test_previously_imported_invalid_source_recovers_without_conflict(tmp_path, monkeypatch):
    runtime, cfg = _fixture(tmp_path, monkeypatch)
    _legacy_shared(cfg, "https://open.feishu.cn/open-apis/bot/v2/hook/one")
    reconcile_legacy(runtime, cfg)
    shared_feishu_webhook_path(cfg).unlink()
    with pytest.raises(ValueError, match="unavailable or invalid"):
        reconcile_legacy(runtime, cfg)
    assert load_policy(runtime.root, "project", "project-1")["legacy"]["status"] == "source_invalid"
    write_shared_feishu_webhook(cfg, "https://open.feishu.cn/open-apis/bot/v2/hook/two")
    reconcile_legacy(runtime, cfg)
    assert resolve_delivery(runtime.root, "project-1")["webhook"].endswith("/two")


def test_disabled_shared_source_without_webhook_imports_as_disabled(tmp_path, monkeypatch):
    runtime, cfg = _fixture(tmp_path, monkeypatch)
    update_notifications(
        cfg,
        lambda _current: {
            "enabled": False,
            "providers": {"feishu": {"enabled": False, "credential_source": "shared_file"}},
        },
    )
    reconcile_legacy(runtime, cfg)
    override = load_policy(runtime.root, "project", "project-1")["override"]
    assert override["enabled"] is False
    assert "destination" not in override


def test_legacy_import_records_activated_credential_before_publish(tmp_path, monkeypatch):
    runtime, cfg = _fixture(tmp_path, monkeypatch)
    _legacy_shared(cfg, "https://open.feishu.cn/open-apis/bot/v2/hook/one")
    reconcile_legacy(runtime, cfg)
    imported = load_policy(runtime.root, "project", "project-1")
    credential_id = imported["override"]["destination"]["credential_id"]
    retention = json.loads((runtime.root / "notifications" / "retention.json").read_text())
    assert retention["credentials"][credential_id]["ever_referenced"] is True
    replace_policy(runtime.root, "project", imported["revision"], None, "project-1")
    assert cleanup_credentials(runtime.root, now=10**12)["deleted"] == 0
    assert credential_path(runtime.root, credential_id).exists()
