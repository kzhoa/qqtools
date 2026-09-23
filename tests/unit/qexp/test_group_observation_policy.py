import json
import threading
import time

import pytest

from qqtools.plugins.qexp.runtime import group_observation_policy as policy
from qqtools.plugins.qexp.runtime.records import new_group
from qqtools.plugins.qexp.runtime.store import atomic_replace


def _published_group(root, name="training", *, creation_operation_id=None):
    group = new_group(name, "machine-a", creation_operation_id=creation_operation_id)
    path = root / "groups" / f"{name}.json"
    atomic_replace(path, group)
    return group


def test_group_policy_defaults_then_set_increments_without_mutating_group(tmp_path):
    group = _published_group(tmp_path)
    initial = policy.inspect_group_policy(tmp_path, "training")
    assert initial["live_progress"] is False
    assert initial["source"] == "default"
    assert initial["revision"] == 0

    first = policy.set_group_policy(tmp_path, "training", True)
    second = policy.set_group_policy(tmp_path, "training", False)
    assert first["revision"] == 1
    assert second["revision"] == 2
    assert second["live_progress"] is False
    assert second["applies_to"] == "new_submissions"
    assert json.loads((tmp_path / "groups" / "training.json").read_text()) == group
    stored = policy.read_policy_record(tmp_path, "training")
    assert stored == {
        "version": 1,
        "revision": 2,
        "group_identity": policy.group_identity(group),
        "live_progress": False,
    }


def test_group_recreation_cannot_inherit_policy_at_same_timestamp(tmp_path, monkeypatch):
    previous = _published_group(tmp_path)
    policy.set_group_policy(tmp_path, "training", True)
    replacement = new_group("training", "machine-a", creation_operation_id="operation-b")
    replacement["meta"]["created_at"] = previous["meta"]["created_at"]
    monkeypatch.setattr(policy.group_namespace, "read_group", lambda root, name: replacement)

    inspected = policy.inspect_group_policy(tmp_path, "training")
    assert inspected["live_progress"] is False
    assert inspected["source"] == "default"
    assert inspected["revision"] == 0
    result = policy.set_group_policy(tmp_path, "training", True)
    assert result["revision"] == 1
    assert result["group_identity"]["creation_operation_id"] == "operation-b"


def test_malformed_policy_is_command_error_and_oversized_record_rejected(tmp_path):
    _published_group(tmp_path)
    path = tmp_path / "group-observation" / "training.json"
    path.parent.mkdir()
    path.write_text('{"version": 1, "revision": true}', encoding="utf-8")
    with pytest.raises(ValueError):
        policy.inspect_group_policy(tmp_path, "training")
    with pytest.raises(ValueError):
        policy.set_group_policy(tmp_path, "training", True)
    path.write_text("{" + " " * 4096 + "}", encoding="utf-8")
    with pytest.raises(ValueError):
        policy.read_policy_record(tmp_path, "training")


def test_submission_policy_read_times_out_without_replacing_blocked_worker(tmp_path, monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    def blocked(root, name):
        entered.set()
        release.wait(2)
        return None

    monkeypatch.setattr(policy, "read_policy_record", blocked)
    started = time.monotonic()
    first = policy.read_policy_snapshot(tmp_path, "training", timeout=0.1)
    assert time.monotonic() - started < 0.25
    assert entered.is_set()
    assert first["status"] == "unavailable"
    started = time.monotonic()
    second = policy.read_policy_snapshot(tmp_path, "training", timeout=0.1)
    assert time.monotonic() - started < 0.05
    assert second["status"] == "unavailable"
    job = policy._active_snapshot
    assert job is not None
    release.set()
    deadline = time.monotonic() + 1
    while policy._active_snapshot is not None and not policy._active_snapshot.completed.is_set():
        assert time.monotonic() < deadline
        time.sleep(0.001)
    assert job.result is None
    assert job.discarded is True


def test_policy_snapshot_permission_failure_is_advisory(tmp_path, monkeypatch):
    def forbidden(_root, _name):
        raise PermissionError("policy unreadable")

    monkeypatch.setattr(policy, "read_policy_record", forbidden)
    snapshot = policy.read_policy_snapshot(tmp_path, "training")
    assert snapshot["status"] == "unavailable"
    assert "PermissionError" in snapshot["reason"]


def test_policy_set_refuses_identity_change_before_replace(tmp_path, monkeypatch):
    original = _published_group(tmp_path)
    replacement = new_group("training", "machine-a", creation_operation_id="different-operation")
    replacement["meta"]["created_at"] = original["meta"]["created_at"]
    groups = iter((original, replacement))
    monkeypatch.setattr(policy.group_namespace, "read_group", lambda *_args: next(groups))

    with pytest.raises(RuntimeError, match="identity changed"):
        policy.set_group_policy(tmp_path, "training", True)
    assert not (tmp_path / "group-observation" / "training.json").exists()
