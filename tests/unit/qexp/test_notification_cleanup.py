"""Retain active private credentials and enforce orphan/retired expiry boundaries."""

import json
from datetime import datetime

from qqtools.plugins.qexp.notification_cleanup import cleanup_credentials, credential_retention_status, mark_reference
from qqtools.plugins.qexp.notification_credentials import credential_path, stage_webhook
from qqtools.plugins.qexp.notification_policy import policy_guard, policy_path, replace_policy_unlocked

WEBHOOK = "https://open.feishu.cn/open-apis/bot/v2/hook/retention-example"


def _created_at(runtime_root, credential_id):
    value = json.loads(credential_path(runtime_root, credential_id).read_text())["created_at"]
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _destination(credential_id):
    return {
        "provider": "feishu",
        "source": "private_file",
        "credential_id": credential_id,
        "signing": "unsigned",
    }


def test_staged_orphans_expire_at_24_hours_not_earlier(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    credential_id = stage_webhook(root, WEBHOOK)
    created = _created_at(root, credential_id)
    path = credential_path(root, credential_id)

    cleanup_credentials(root, now=created + 24 * 3600 - 1)
    assert path.exists()
    cleanup_credentials(root, now=created + 24 * 3600)
    assert not path.exists()


def test_read_only_status_caps_directory_scan_and_retains_unknown_credentials(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    root.mkdir()
    credential_id = stage_webhook(root, WEBHOOK)
    monkeypatch.setattr("qqtools.plugins.qexp.notification_cleanup._STATUS_SCAN_LIMIT", 1)

    status = credential_retention_status(root)

    assert "status limit" in status["diagnostic"]
    assert credential_path(root, credential_id).exists()
    assert not (root / "notifications" / "retention.json").exists()


def test_referenced_credentials_survive_and_retire_seven_days_after_reset(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    credential_id = stage_webhook(root, WEBHOOK)
    created = _created_at(root, credential_id)
    path = credential_path(root, credential_id)
    with policy_guard(root):
        mark_reference(root, credential_id)
        replace_policy_unlocked(root, "global", 0, {"enabled": True, "destination": _destination(credential_id)})

    cleanup_credentials(root, now=created + 25 * 3600)
    assert path.exists()
    with policy_guard(root):
        replace_policy_unlocked(root, "global", 1, None)
    unreferenced = created + 26 * 3600
    cleanup_credentials(root, now=unreferenced)
    cleanup_credentials(root, now=unreferenced + 7 * 86400 - 1)
    assert path.exists()
    cleanup_credentials(root, now=unreferenced + 7 * 86400)
    assert not path.exists()


def test_rereference_cancels_expiry_and_corrupt_policy_fails_closed(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    credential_id = stage_webhook(root, WEBHOOK)
    created = _created_at(root, credential_id)
    path = credential_path(root, credential_id)
    with policy_guard(root):
        mark_reference(root, credential_id)
        replace_policy_unlocked(root, "global", 0, {"enabled": True, "destination": _destination(credential_id)})
        replace_policy_unlocked(root, "global", 1, None)
    cleanup_credentials(root, now=created + 25 * 3600)
    with policy_guard(root):
        mark_reference(root, credential_id)
        replace_policy_unlocked(root, "global", 2, {"enabled": True, "destination": _destination(credential_id)})
    cleanup_credentials(root, now=created + 9 * 86400)
    assert path.exists()

    with policy_guard(root):
        replace_policy_unlocked(root, "global", 3, None)
    policy_path(root, "project", "unknown").write_text("invalid-json")
    result = cleanup_credentials(root, now=created + 30 * 86400)
    assert path.exists()
    assert result.get("error") or result.get("diagnostic")
