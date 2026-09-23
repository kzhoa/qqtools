"""Resolve sparse notification overrides without mixing credential bundles."""

import pytest

from qqtools.plugins.qexp.notification_credentials import credential_path, stage_webhook
from qqtools.plugins.qexp.notification_policy import replace_policy
from qqtools.plugins.qexp.notification_resolver import resolve_delivery, resolve_policy

WEBHOOK = "https://open.feishu.cn/open-apis/bot/v2/hook/global-robot"
PROJECT_WEBHOOK = "https://open.feishu.cn/open-apis/bot/v2/hook/project-robot"


def _destination(credential_id, signing="unsigned"):
    return {
        "provider": "feishu",
        "source": "private_file",
        "credential_id": credential_id,
        "signing": signing,
    }


def test_global_defaults_inherit_into_two_projects_and_new_project(tmp_path):
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    assert resolve_policy(runtime_root, "first")["enabled"] is False

    credential_id = stage_webhook(runtime_root, WEBHOOK)
    replace_policy(runtime_root, "global", 0, {"enabled": True, "destination": _destination(credential_id)})

    for project_id in ("first", "second", "registered-later"):
        effective = resolve_policy(runtime_root, project_id)
        assert effective["enabled"] is True
        assert effective["provenance"]["destination"] == "global"
        assert resolve_delivery(runtime_root, project_id, environ={})["webhook"] == WEBHOOK


def test_project_override_disable_and_reset_do_not_change_other_projects(tmp_path):
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    global_id = stage_webhook(runtime_root, WEBHOOK)
    project_id = stage_webhook(runtime_root, PROJECT_WEBHOOK)
    replace_policy(runtime_root, "global", 0, {"enabled": True, "destination": _destination(global_id)})
    replace_policy(runtime_root, "project", 0, {"destination": _destination(project_id)}, "first")

    assert resolve_delivery(runtime_root, "first")["webhook"] == PROJECT_WEBHOOK
    assert resolve_delivery(runtime_root, "second")["webhook"] == WEBHOOK
    assert resolve_policy(runtime_root, "first")["provenance"]["enabled"] == "global"

    replace_policy(runtime_root, "project", 1, {"enabled": False}, "first")
    assert resolve_policy(runtime_root, "first")["enabled"] is False
    with pytest.raises(ValueError, match="disabled"):
        resolve_delivery(runtime_root, "first")
    assert resolve_delivery(runtime_root, "second")["webhook"] == WEBHOOK

    replace_policy(runtime_root, "project", 2, None, "first")
    assert resolve_delivery(runtime_root, "first")["webhook"] == WEBHOOK


def test_explicit_project_credential_failure_never_sends_to_global(tmp_path):
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    global_id = stage_webhook(runtime_root, WEBHOOK)
    project_id = stage_webhook(runtime_root, PROJECT_WEBHOOK)
    replace_policy(runtime_root, "global", 0, {"enabled": True, "destination": _destination(global_id)})
    replace_policy(runtime_root, "project", 0, {"destination": _destination(project_id)}, "first")
    credential_path(runtime_root, project_id).unlink()

    with pytest.raises((ValueError, OSError)) as error:
        resolve_delivery(runtime_root, "first")
    assert WEBHOOK not in str(error.value)
    assert resolve_delivery(runtime_root, "second")["webhook"] == WEBHOOK


def test_project_destination_never_inherits_signing_from_global(tmp_path):
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    global_id = stage_webhook(runtime_root, WEBHOOK)
    project_id = stage_webhook(runtime_root, PROJECT_WEBHOOK)
    replace_policy(
        runtime_root,
        "global",
        0,
        {"enabled": True, "destination": _destination(global_id, {"env": "GLOBAL_SIGNING"})},
    )
    replace_policy(runtime_root, "project", 0, {"destination": _destination(project_id)}, "first")

    resolved = resolve_delivery(runtime_root, "first", environ={})
    assert resolved["secret"] is None
    assert resolved["webhook"] == PROJECT_WEBHOOK
    with pytest.raises(ValueError, match="GLOBAL_SIGNING"):
        resolve_delivery(runtime_root, "second", environ={})
