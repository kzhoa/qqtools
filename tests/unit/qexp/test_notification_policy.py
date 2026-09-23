"""Canonical notification policy is sparse, revisioned and private."""

import json
import stat

import pytest

from qqtools.plugins.qexp.notification_policy import load_policy, policy_path, replace_policy, validate_override
from qqtools.plugins.qexp.runtime.store import CASConflict

PRIVATE_DESTINATION = {
    "provider": "feishu",
    "source": "private_file",
    "credential_id": "a" * 32,
    "signing": "unsigned",
}


def test_global_and_project_records_are_revisioned_and_separate(tmp_path):
    runtime_root = tmp_path / "machine-runtime"
    runtime_root.mkdir()
    assert load_policy(runtime_root, "global")["revision"] == 0
    assert load_policy(runtime_root, "project", "project-1")["override"] is None

    global_record = replace_policy(runtime_root, "global", 0, {"enabled": True, "destination": PRIVATE_DESTINATION})
    project_record = replace_policy(runtime_root, "project", 0, {"enabled": False}, "project-1")

    assert global_record["revision"] == project_record["revision"] == 1
    assert load_policy(runtime_root, "global")["override"]["enabled"] is True
    assert load_policy(runtime_root, "project", "project-1")["override"] == {"enabled": False}
    assert stat.S_IMODE(policy_path(runtime_root, "global").stat().st_mode) == 0o600
    assert stat.S_IMODE(policy_path(runtime_root, "global").parent.stat().st_mode) == 0o700


def test_reset_is_a_revisioned_tombstone_and_conflict_preserves_prior_record(tmp_path):
    runtime_root = tmp_path / "machine-runtime"
    runtime_root.mkdir()
    replace_policy(runtime_root, "project", 0, {"enabled": False}, "project-1")

    with pytest.raises(CASConflict):
        replace_policy(runtime_root, "project", 0, {"enabled": True}, "project-1")
    assert load_policy(runtime_root, "project", "project-1")["override"] == {"enabled": False}

    reset = replace_policy(runtime_root, "project", 1, None, "project-1")
    assert reset["revision"] == 2
    assert load_policy(runtime_root, "project", "project-1")["override"] is None
    assert policy_path(runtime_root, "project", "project-1").exists()


@pytest.mark.parametrize(
    "override",
    [
        {"enabled": 1},
        {"timeout_seconds": float("nan")},
        {"destination": None},
        {"destination": {"provider": "feishu", "source": "private_file", "credential_id": "a" * 32}},
        {"destination": {**PRIVATE_DESTINATION, "webhook_env": "OTHER"}},
        {"destination": {**PRIVATE_DESTINATION, "signing": {"env": "INVALID-NAME"}}},
        {"acknowledge_shared_secret_risk": True},
    ],
)
def test_invalid_overrides_are_rejected(override):
    with pytest.raises(ValueError):
        validate_override(override)


def test_malformed_existing_policy_never_becomes_default(tmp_path):
    runtime_root = tmp_path / "machine-runtime"
    path = policy_path(runtime_root, "global")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema_version": 1, "revision": 1, "override": {"enabled": True}}))

    with pytest.raises(ValueError):
        load_policy(runtime_root, "global")


def test_project_id_cannot_escape_policy_directory(tmp_path):
    with pytest.raises(ValueError):
        policy_path(tmp_path, "project", "../escape")
