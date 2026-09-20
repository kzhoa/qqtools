"""Recheck owners recover process death without retrying an old command."""

import multiprocessing
import os

import pytest

from qqtools.plugins.qexp import submit
from qqtools.plugins.qexp.commands import task as task_commands
from qqtools.plugins.qexp.runtime.authority_lock import authority_locks
from qqtools.plugins.qexp.runtime.group_discovery.changes import record_task_change, settle_task_change
from qqtools.plugins.qexp.runtime.group_discovery.rechecks import GroupRechecks
from qqtools.plugins.qexp.runtime.ready.routes import reference_for_generation
from qqtools.plugins.qexp.runtime.tasks import load_task
from qqtools.plugins.qexp.scheduler import authorize_launch, claim_task, fail_attempt
from tests.helpers.qexp_discovery import isolated_group

pytestmark = pytest.mark.integration


def failed_task(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    task = submit(cfg, ["true"], group="experiment")
    attempt = claim_task(cfg, task.task_id, [0])
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert fail_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "failure")
    task = load_task(cfg, task.task_id)
    journal = GroupRechecks(cfg.shared_root, "experiment")
    with authority_locks(cfg, task):
        journal.snapshot(initialize=True)
    return cfg, task, journal


def interrupted_retry(cfg, task_id, cut):
    target = "prepare_ready_transition" if cut == "before_task" else "save_task"
    original = getattr(task_commands, target)

    def crash(*args, **kwargs):
        original(*args, **kwargs)
        os._exit(73)

    setattr(task_commands, target, crash)
    task_commands.retry(cfg, task_id)


@pytest.mark.parametrize("cut", ["before_task", "after_task"])
def test_retry_owner_settles_after_real_process_death(tmp_path, cut):
    cfg, before, journal = failed_task(tmp_path)
    process = multiprocessing.get_context("fork").Process(target=interrupted_retry, args=(cfg, before.task_id, cut))
    process.start()
    try:
        process.join(10)
        assert not process.is_alive()
        assert process.exitcode == 73
    finally:
        if process.is_alive():
            process.kill()
            process.join(5)
        process.close()
    position = journal.snapshot()
    assert position.tail == 1
    event = journal.read(position, 1)
    assert event["state"] == "in_flight"
    current = load_task(cfg, before.task_id)
    with authority_locks(cfg, current):
        settled, results = settle_task_change(cfg, event)
    assert settled
    assert results == []
    resolved = journal.read(journal.snapshot(), 1)
    current = load_task(cfg, before.task_id)
    if cut == "before_task":
        assert resolved["state"] == "aborted"
        assert current.to_dict() == before.to_dict()
        assert reference_for_generation(cfg, before.task_id, before.ready_generation + 1) is None
    else:
        assert resolved["state"] == "committed"
        assert current.state["projection"] == "queued"
        assert current.ready_generation == before.ready_generation + 1
        assert reference_for_generation(cfg, before.task_id, current.ready_generation) is not None
    # An abandoned ticket is never reused by a later independently authorized attempt.
    if cut == "before_task":
        task_commands.retry(cfg, before.task_id)
        assert journal.snapshot().tail == 2
        assert journal.read(journal.snapshot(), 2)["state"] == "committed"


def test_terminal_ticket_settles_attempt_before_task_without_new_execution(tmp_path, monkeypatch):
    from qqtools.plugins.qexp import lifecycle

    cfg = isolated_group(tmp_path, tail=0)
    task = submit(cfg, ["true"], group="experiment")
    attempt = claim_task(cfg, task.task_id, [0])
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    journal = GroupRechecks(cfg.shared_root, "experiment")
    with authority_locks(cfg, load_task(cfg, task.task_id)):
        journal.snapshot(initialize=True)

    def crash(*args, **kwargs):
        raise OSError("Attempt is terminal, Task is still running")

    with monkeypatch.context() as patch:
        patch.setattr(lifecycle, "save_task", crash)
        with pytest.raises(OSError, match="still running"):
            fail_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "failure")
    event = journal.read(journal.snapshot(), 1)
    assert event["state"] == "in_flight"
    current = load_task(cfg, task.task_id)
    assert current.state["projection"] == "running"
    with authority_locks(cfg, current):
        settled, results = settle_task_change(cfg, event)
    assert settled
    assert len(results) == 1
    assert results[0].outcome == "committed"
    current = load_task(cfg, task.task_id)
    assert current.state["projection"] == "failed"
    assert current.claim_control["active_claim"] is None
    assert current.attempt_control["current_attempt_number"] == attempt.attempt_number
    assert journal.read(journal.snapshot(), 1)["state"] == "committed"


def test_lease_only_update_does_not_publish_rechecks(tmp_path):
    from qqtools.plugins.qexp.runtime.claims import renew_lease

    cfg = isolated_group(tmp_path, tail=0)
    task = submit(cfg, ["true"], group="experiment")
    attempt = claim_task(cfg, task.task_id, [0])
    journal = GroupRechecks(cfg.shared_root, "experiment")
    with authority_locks(cfg, load_task(cfg, task.task_id)):
        journal.snapshot(initialize=True)
    assert renew_lease(cfg, task.task_id, attempt.current_fencing_token, "2030-01-01T00:00:00Z")
    assert journal.snapshot().tail == 0


def test_publication_failure_invalidates_old_completion_proof_before_effect(tmp_path, monkeypatch):
    cfg, task, journal = failed_task(tmp_path)
    before = journal.snapshot()

    def unavailable(*args, **kwargs):
        raise OSError("journal unavailable")

    monkeypatch.setattr(GroupRechecks, "begin", unavailable)
    with authority_locks(cfg, task):
        with record_task_change(cfg, task, "task_cancel"):
            assert journal.snapshot().generation > before.generation


def test_failure_of_publication_and_invalidation_prevents_effect(tmp_path, monkeypatch):
    cfg, task, _ = failed_task(tmp_path)

    def unavailable(*args, **kwargs):
        raise OSError("storage unavailable")

    monkeypatch.setattr(GroupRechecks, "begin", unavailable)
    monkeypatch.setattr(GroupRechecks, "invalidate", unavailable)
    with authority_locks(cfg, task):
        with pytest.raises(OSError, match="storage unavailable"):
            with record_task_change(cfg, task, "task_cancel"):
                pytest.fail("effect was admitted without a publication or generation fence")


def test_retry_uncertain_task_commit_retains_ready_generation_for_settlement(tmp_path, monkeypatch):
    cfg, before, journal = failed_task(tmp_path)
    original = task_commands.save_task

    def uncertain_commit(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("Task rename committed before error")

    with monkeypatch.context() as patch:
        patch.setattr(task_commands, "save_task", uncertain_commit)
        with pytest.raises(OSError, match="before error"):
            task_commands.retry(cfg, before.task_id)
    current = load_task(cfg, before.task_id)
    assert current.state["projection"] == "queued"
    assert reference_for_generation(cfg, before.task_id, current.ready_generation) is not None
    event = journal.read(journal.snapshot(), 1)
    with authority_locks(cfg, current):
        settled, _ = settle_task_change(cfg, event)
    assert settled
    assert journal.read(journal.snapshot(), 1)["state"] == "committed"


def test_prelaunch_expiry_settles_ready_publication_after_process_death(tmp_path):
    from qqtools.plugins.qexp import scheduler

    cfg = isolated_group(tmp_path, tail=0)
    task = submit(cfg, ["true"], group="experiment")
    attempt = claim_task(cfg, task.task_id, [0])
    journal = GroupRechecks(cfg.shared_root, "experiment")
    with authority_locks(cfg, task):
        journal.snapshot(initialize=True)

    def interrupt_expiry():
        original = scheduler.save_task

        def crash(*args, **kwargs):
            original(*args, **kwargs)
            os._exit(75)

        scheduler.save_task = crash
        scheduler.expire_claim(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)

    process = multiprocessing.get_context("fork").Process(target=interrupt_expiry)
    process.start()
    try:
        process.join(10)
        assert not process.is_alive()
        assert process.exitcode == 75
    finally:
        if process.is_alive():
            process.kill()
            process.join(5)
        process.close()
    current = load_task(cfg, task.task_id)
    assert current.state["projection"] == "queued"
    event = journal.read(journal.snapshot(), 1)
    assert event["state"] == "in_flight"
    with authority_locks(cfg, current):
        assert settle_task_change(cfg, event) == (True, [])
    # Real scheduling checks publication, not merely the retained marker file.
    assert claim_task(cfg, task.task_id, [0]) is not None


def test_interrupted_cancel_is_superseded_by_later_terminal_and_retry(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.commands import group as group_commands

    cfg = isolated_group(tmp_path, tail=0)
    task = submit(cfg, ["true"], group="experiment")
    attempt = claim_task(cfg, task.task_id, [0])
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    journal = GroupRechecks(cfg.shared_root, "experiment")
    with authority_locks(cfg, task):
        journal.snapshot(initialize=True)
    original = group_commands.save_task

    def crash(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("cancel request committed")

    with monkeypatch.context() as patch:
        patch.setattr(group_commands, "save_task", crash)
        with authority_locks(cfg, task), pytest.raises(OSError, match="request committed"):
            group_commands._apply_group_cancel_locked(
                cfg, load_task(cfg, task.task_id), {"operation_id": "cancel-one", "terminate_running": False}
            )
    event = journal.read(journal.snapshot(), 1)
    assert event["state"] == "in_flight"
    assert fail_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "failure")
    task_commands.retry(cfg, task.task_id)
    current = load_task(cfg, task.task_id)
    assert current.state["projection"] == "queued"
    assert not current.control["cancellation_requested_at"]
    with authority_locks(cfg, current):
        assert settle_task_change(cfg, event) == (True, [])
    assert journal.read(journal.snapshot(), 1)["state"] == "committed"


def test_missing_group_provenance_does_not_settle_event(tmp_path):
    from qqtools.plugins.qexp.runtime.tasks import save_task

    cfg, task, journal = failed_task(tmp_path)
    with authority_locks(cfg, task):
        with pytest.raises(OSError):
            with record_task_change(cfg, task, "task_cancel"):
                raise OSError("before mutation")
        event = journal.read(journal.snapshot(), 1)
        task.group_name = None
        save_task(cfg, task)
        assert settle_task_change(cfg, event) == (False, [])
    assert journal.read(journal.snapshot(), 1)["state"] == "in_flight"


def test_released_claim_ticket_settles_after_task_write_error(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runtime import claims

    cfg = isolated_group(tmp_path, tail=0)
    task = submit(cfg, ["true"], group="experiment")
    attempt = claim_task(cfg, task.task_id, [0])
    journal = GroupRechecks(cfg.shared_root, "experiment")
    with authority_locks(cfg, task):
        journal.snapshot(initialize=True)
    original = claims.save_task

    def interrupt(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("released claim saved")

    with monkeypatch.context() as patch:
        patch.setattr(claims, "save_task", interrupt)
        with pytest.raises(OSError, match="released claim"):
            claims.release_claim(cfg, task.task_id, attempt.current_fencing_token, "manual_release")
    current = load_task(cfg, task.task_id)
    event = journal.read(journal.snapshot(), 1)
    with authority_locks(cfg, current):
        assert settle_task_change(cfg, event) == (True, [])
    assert journal.read(journal.snapshot(), 1)["state"] == "committed"


def test_retired_retry_settles_when_its_marker_was_already_consumed(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.commands.cleanup import clean

    cfg, task, journal = failed_task(tmp_path)

    def interrupt(*args, **kwargs):
        raise RuntimeError("exit before acknowledgement")

    with monkeypatch.context() as patch:
        patch.setattr(GroupRechecks, "resolve", interrupt)
        with pytest.raises(RuntimeError, match="before acknowledgement"):
            task_commands.retry(cfg, task.task_id)
    event = journal.read(journal.snapshot(), 1)
    assert event["state"] == "in_flight"
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert fail_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "failure")
    clean(cfg, task_id=task.task_id)
    assert reference_for_generation(cfg, task.task_id, event["evidence"]["details"]["reserved_generation"]) is None
    with authority_locks(cfg, task):
        assert settle_task_change(cfg, event) == (True, [])
    assert journal.read(journal.snapshot(), 1)["state"] == "committed"


def test_retry_abort_accepts_marker_already_removed_by_failed_call(tmp_path, monkeypatch):
    cfg, task, journal = failed_task(tmp_path)

    def interrupt(*args, **kwargs):
        raise OSError("before ready publication")

    with monkeypatch.context() as patch:
        patch.setattr(task_commands, "prepare_ready_transition", interrupt)
        with pytest.raises(OSError, match="before ready publication"):
            task_commands.retry(cfg, task.task_id)
    event = journal.read(journal.snapshot(), 1)
    assert reference_for_generation(cfg, task.task_id, event["evidence"]["details"]["reserved_generation"]) is None
    with authority_locks(cfg, task):
        assert settle_task_change(cfg, event) == (True, [])
    assert journal.read(journal.snapshot(), 1)["state"] == "aborted"
