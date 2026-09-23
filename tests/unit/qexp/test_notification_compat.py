"""QQTOOLS-COMPAT-0016: Legacy notification import distinguishes edits and absence."""

import pytest

from qqtools.plugins.qexp.machine_config import init_shared_root
from qqtools.plugins.qexp.notification_compat import capture_legacy, legacy_override
from qqtools.plugins.qexp.notification_config import update_notifications, write_shared_feishu_webhook
from qqtools.plugins.qexp.runtime.locks import machine_lock


def _snapshot(cfg):
    with machine_lock(cfg.shared_root, cfg.machine_name):
        return capture_legacy(cfg)


def test_absent_section_and_explicit_disable_are_different(tmp_path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "local")
    absent = _snapshot(cfg)
    assert absent.is_absent
    assert legacy_override(absent) is None

    update_notifications(cfg, lambda current: {**current, "enabled": False})
    disabled = _snapshot(cfg)
    assert not disabled.is_absent
    assert disabled.fingerprint != absent.fingerprint
    assert legacy_override(disabled)["enabled"] is False


def test_shared_webhook_only_old_write_changes_fingerprint(tmp_path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "local")
    update_notifications(
        cfg,
        lambda current: {
            "enabled": True,
            "providers": {
                "feishu": {
                    "enabled": True,
                    "credential_source": "shared_file",
                    "secret_env": "SIGNING_SECRET",
                    "timeout_seconds": 9,
                }
            },
        },
    )
    write_shared_feishu_webhook(cfg, "https://open.feishu.cn/open-apis/bot/v2/hook/first")
    before = _snapshot(cfg)
    override = legacy_override(before, "a" * 32)
    assert override["enabled"] is True
    assert override["destination"]["signing"] == {"env": "SIGNING_SECRET"}
    assert override["timeout_seconds"] == 9
    assert "first" not in repr(override)

    write_shared_feishu_webhook(cfg, "https://open.feishu.cn/open-apis/bot/v2/hook/second")
    after = _snapshot(cfg)
    assert after.fingerprint != before.fingerprint
    assert after.webhook != before.webhook


def test_legacy_env_snapshot_preserves_effective_disable(tmp_path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "local")
    update_notifications(
        cfg,
        lambda current: {
            "enabled": False,
            "providers": {"feishu": {"enabled": True, "webhook_env": "CUSTOM_FEISHU"}},
        },
    )
    converted = legacy_override(_snapshot(cfg))
    assert converted["enabled"] is False
    assert converted["destination"]["webhook_env"] == "CUSTOM_FEISHU"
    assert converted["destination"]["signing"] == "unsigned"


def test_legacy_shared_source_requires_imported_private_id(tmp_path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "local")
    update_notifications(
        cfg,
        lambda current: {
            "enabled": True,
            "providers": {"feishu": {"enabled": True, "credential_source": "shared_file"}},
        },
    )
    write_shared_feishu_webhook(cfg, "https://open.feishu.cn/open-apis/bot/v2/hook/existing")
    snapshot = _snapshot(cfg)
    with pytest.raises(ValueError, match="credential"):
        legacy_override(snapshot)
