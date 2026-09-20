"""A durable cancellation barrier fences claims and fresh launch authorization."""

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.commands import group as group_commands
from qqtools.plugins.qexp.commands.group import create_group, group_control
from qqtools.plugins.qexp.commands.task import retry
from qqtools.plugins.qexp.runtime.operation_store import locate_operation_path
from qqtools.plugins.qexp.runtime.paths import group_path
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task
from qqtools.plugins.qexp.scheduler import authorize_launch, claim_task, fail_attempt, resume_starting_attempt

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def setup_task(tmp_path):
    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "rt")
    create_group(cfg, "exp")
    return cfg, submit(cfg, ["true"], group="exp")


def publish_barrier(cfg, monkeypatch, *, terminate=False):
    def interrupted(*args):
        raise OSError("cancel creator interrupted before first member")

    with monkeypatch.context() as patch:
        patch.setattr(group_commands, "_apply_group_cancel_locked", interrupted)
        with pytest.raises(OSError, match="creator interrupted"):
            group_control(cfg, "exp", "cancel", terminate_running=terminate)
    return read_json(group_path(cfg.shared_root, "exp"))["cancellation_operation"]["operation_id"]


def test_barrier_blocks_claim_but_not_members_added_after_watermark(tmp_path, monkeypatch):
    cfg, task = setup_task(tmp_path)
    publish_barrier(cfg, monkeypatch)
    assert claim_task(cfg, task.task_id, [0]) is None
    assert load_task(cfg, task.task_id).state["projection"] == "queued"
    later = submit(cfg, ["true"], group="exp")
    assert claim_task(cfg, later.task_id, [0]) is not None


@pytest.mark.parametrize("resume", [False, True])
def test_barrier_blocks_previously_claimed_but_unauthorized_launch(tmp_path, monkeypatch, resume):
    cfg, task = setup_task(tmp_path)
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    publish_barrier(cfg, monkeypatch)
    if resume:
        assert resume_starting_attempt(cfg, task.task_id) is None
    else:
        assert not authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert load_task(cfg, task.task_id).state["projection"] == "cancelled"


@pytest.mark.parametrize("terminate", [False, True])
def test_committed_launch_obeys_running_cancellation_semantics(tmp_path, monkeypatch, terminate):
    cfg, task = setup_task(tmp_path)
    attempt = claim_task(cfg, task.task_id, [0])
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    launch_id = load_task(cfg, task.task_id).claim_control["active_claim"]["launch_id"]
    publish_barrier(cfg, monkeypatch, terminate=terminate)
    resumed = resume_starting_attempt(cfg, task.task_id)
    if terminate:
        assert resumed is None
        assert load_task(cfg, task.task_id).state["projection"] == "cancelled"
    else:
        assert resumed is not None
        assert resumed.authorization["launch_id"] == launch_id


def test_default_cancel_request_does_not_revoke_committed_launch(tmp_path):
    cfg, task = setup_task(tmp_path)
    attempt = claim_task(cfg, task.task_id, [0])
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    launch_id = load_task(cfg, task.task_id).claim_control["active_claim"]["launch_id"]
    assert group_control(cfg, "exp", "cancel")["cancellation_operation"]["state"] == "completed"
    resumed = resume_starting_attempt(cfg, task.task_id)
    assert resumed is not None
    assert resumed.authorization["launch_id"] == launch_id


def test_completed_cancel_does_not_prohibit_independent_retry(tmp_path):
    cfg, task = setup_task(tmp_path)
    attempt = claim_task(cfg, task.task_id, [0])
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert fail_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "failure")
    assert group_control(cfg, "exp", "cancel")["cancellation_operation"]["state"] == "completed"
    retry(cfg, task.task_id)
    assert claim_task(cfg, task.task_id, [0]) is not None


def test_missing_operation_does_not_silently_disable_a_barrier(tmp_path, monkeypatch):
    cfg, task = setup_task(tmp_path)
    operation_id = publish_barrier(cfg, monkeypatch)
    locate_operation_path(cfg, "group_control", operation_id).unlink()
    assert claim_task(cfg, task.task_id, [0]) is None


@pytest.mark.parametrize("damage", ["non_object", "invalid_state"])
def test_malformed_operation_fails_closed_without_crashing_claim(tmp_path, monkeypatch, damage):
    cfg, task = setup_task(tmp_path)
    operation_id = publish_barrier(cfg, monkeypatch)
    path = locate_operation_path(cfg, "group_control", operation_id)
    if damage == "non_object":
        path.write_text("[]")
    else:
        record = read_json(path)
        record["group_control"]["state"] = []
        atomic_replace(path, record)
    assert claim_task(cfg, task.task_id, [0]) is None


def test_pause_after_committed_authorization_preserves_original_launch(tmp_path):
    cfg, task = setup_task(tmp_path)
    attempt = claim_task(cfg, task.task_id, [0])
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    launch_id = load_task(cfg, task.task_id).claim_control["active_claim"]["launch_id"]
    group_control(cfg, "exp", "pause")
    resumed = resume_starting_attempt(cfg, task.task_id)
    assert resumed is not None
    assert resumed.authorization["launch_id"] == launch_id
