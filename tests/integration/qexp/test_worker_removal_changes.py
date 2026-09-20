"""Task transition owners expose crash-safe removal recheck obligations."""

import multiprocessing
import os

import pytest

from qqtools.plugins.qexp import scheduler, submit
from qqtools.plugins.qexp.commands.task import share
from qqtools.plugins.qexp.runtime.authority_lock import authority_locks
from qqtools.plugins.qexp.runtime.availability import transitions
from qqtools.plugins.qexp.runtime.group_discovery.changes import settle_task_change
from qqtools.plugins.qexp.runtime.group_discovery.rechecks import GroupRechecks
from qqtools.plugins.qexp.runtime.paths import attempt_path
from qqtools.plugins.qexp.runtime.ready.routes import reference_for_generation
from qqtools.plugins.qexp.runtime.store import read_json
from qqtools.plugins.qexp.runtime.tasks import load_task
from tests.helpers.qexp_discovery import isolated_group

pytestmark = pytest.mark.integration


def prepare(tmp_path):
    from qqtools.plugins.qexp.commands.group import change_worker

    cfg = isolated_group(tmp_path, tail=0)
    change_worker(cfg, "experiment", "g2", "add")
    task = submit(cfg, ["true"], group="experiment")
    journal = GroupRechecks(cfg.shared_root, "experiment")
    with authority_locks(cfg, task):
        journal.snapshot(initialize=True)
    return cfg, task, journal


def kill_at_transition(cfg, task_id, transition, cut):
    if transition == "availability":
        module = transitions
        symbol = "prepare_ready_transition" if cut == "before_task" else "save_task"
        action = lambda: share(cfg, task_id, helper_machines=["g2"])
    elif transition == "claim":
        module = scheduler
        symbol = "save_task" if cut == "after_task" else "atomic_replace"
        action = lambda: scheduler.claim_task(cfg, task_id, [0])
    else:
        module = scheduler
        symbol = "save_task" if cut == "after_task" else "atomic_replace"
        task = load_task(cfg, task_id)
        claim = task.claim_control["active_claim"]
        action = lambda: scheduler.authorize_launch(cfg, task_id, claim["attempt_id"], claim["fencing_token"])
    original = getattr(module, symbol)

    def crash(*args, **kwargs):
        result = original(*args, **kwargs)
        if symbol != "atomic_replace" or "/attempts/" in str(args[0]):
            os._exit(79)
        return result

    setattr(module, symbol, crash)
    action()


def run_crash(cfg, task_id, transition, cut):
    process = multiprocessing.get_context("fork").Process(
        target=kill_at_transition, args=(cfg, task_id, transition, cut)
    )
    process.start()
    try:
        process.join(10)
        assert not process.is_alive()
        assert process.exitcode == 79
    finally:
        if process.is_alive():
            process.kill()
            process.join(5)
        process.close()


def settle(cfg, event):
    with authority_locks(cfg, load_task(cfg, event["task_id"])):
        return settle_task_change(cfg, event)


@pytest.mark.parametrize("cut", ["before_task", "after_task"])
def test_availability_ready_handoff_recovers_process_death(tmp_path, cut):
    cfg, task, journal = prepare(tmp_path)
    run_crash(cfg, task.task_id, "availability", cut)
    event = journal.read(journal.snapshot(), 1)
    assert event["owner"] == "availability"
    assert event["state"] == "in_flight"
    assert settle(cfg, event) == (True, [])
    state = journal.read(journal.snapshot(), 1)["state"]
    current = load_task(cfg, task.task_id)
    if cut == "before_task":
        assert state == "aborted"
        assert current.to_dict() == task.to_dict()
        assert reference_for_generation(cfg, task.task_id, task.ready_generation + 1) is None
    else:
        assert state == "committed"
        assert current.placement_runtime["queue_scope"] == "shared"
        assert reference_for_generation(cfg, task.task_id, current.ready_generation) is not None


@pytest.mark.parametrize("cut", ["after_task", "after_attempt"])
def test_claim_obligation_requires_materialized_matching_attempt(tmp_path, cut):
    cfg, task, journal = prepare(tmp_path)
    run_crash(cfg, task.task_id, "claim", cut)
    event = journal.read(journal.snapshot(), 1)
    assert event["owner"] == "claim"
    assert event["state"] == "in_flight"
    current = load_task(cfg, task.task_id)
    assert current.claim_control["active_claim"]
    if cut == "after_task":
        assert not attempt_path(cfg.shared_root, task.task_id, 1).exists()
        assert settle(cfg, event) == (False, [])
        assert journal.read(journal.snapshot(), 1)["state"] == "in_flight"
    else:
        assert settle(cfg, event) == (True, [])
        assert journal.read(journal.snapshot(), 1)["state"] == "committed"
        assert read_json(attempt_path(cfg.shared_root, task.task_id, 1))["attempt"]["phase"] == "claimed"


@pytest.mark.parametrize("cut", ["after_task", "after_attempt"])
def test_launch_obligation_never_invents_authorization(tmp_path, cut):
    cfg, task, journal = prepare(tmp_path)
    attempt = scheduler.claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    before_tail = journal.snapshot().tail
    run_crash(cfg, task.task_id, "launch", cut)
    event = journal.read(journal.snapshot(), before_tail + 1)
    assert event["owner"] == "launch"
    assert event["state"] == "in_flight"
    if cut == "after_task":
        assert settle(cfg, event) == (False, [])
        # The existing launch owner, not the discovery consumer, can repair this partial write.
        assert scheduler.resume_starting_attempt(cfg, task.task_id) is not None
    assert settle(cfg, event) == (True, [])
    assert journal.read(journal.snapshot(), before_tail + 1)["state"] == "committed"


def test_availability_uncertain_task_save_preserves_authoritative_ready_generation(tmp_path, monkeypatch):
    cfg, task, journal = prepare(tmp_path)
    original = transitions.save_task

    def uncertain(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("Task rename committed before error")

    with monkeypatch.context() as patch:
        patch.setattr(transitions, "save_task", uncertain)
        with pytest.raises(OSError, match="rename committed"):
            share(cfg, task.task_id, helper_machines=["g2"])
    current = load_task(cfg, task.task_id)
    assert current.ready_generation == task.ready_generation + 1
    assert reference_for_generation(cfg, task.task_id, current.ready_generation) is not None
    event = journal.read(journal.snapshot(), 1)
    assert settle(cfg, event) == (True, [])


def test_queued_cancel_publishes_a_removal_recheck(tmp_path):
    cfg, task, journal = prepare(tmp_path)
    scheduler.cancel_task(cfg, task.task_id)
    position = journal.snapshot()
    events = [journal.read(position, sequence) for sequence in range(1, position.tail + 1)]
    assert any(event["owner"] == "task_cancel" and event["state"] == "committed" for event in events)
    assert load_task(cfg, task.task_id).state["projection"] == "cancelled"


def test_missing_attempt_claim_can_settle_after_existing_prelaunch_cancellation(tmp_path):
    cfg, task, journal = prepare(tmp_path)
    run_crash(cfg, task.task_id, "claim", "after_task")
    event = journal.read(journal.snapshot(), 1)
    assert settle(cfg, event) == (False, [])
    scheduler.cancel_task(cfg, task.task_id)
    current = load_task(cfg, task.task_id)
    assert current.state["projection"] == "cancelled"
    assert current.claim_control["active_claim"] is None
    assert not attempt_path(cfg.shared_root, task.task_id, 1).exists()
    assert settle(cfg, event) == (True, [])


def test_interrupted_launch_obligation_accepts_fenced_recovery_successor(tmp_path):
    from qqtools.plugins.qexp.runtime.attempt_recovery import recover_running_attempt
    from qqtools.plugins.qexp.runtime.store import atomic_replace

    cfg, task, journal = prepare(tmp_path)
    attempt = scheduler.claim_task(cfg, task.task_id, [0])
    before_tail = journal.snapshot().tail
    run_crash(cfg, task.task_id, "launch", "after_attempt")
    event = journal.read(journal.snapshot(), before_tail + 1)
    assert event["state"] == "in_flight"
    manifest_path = cfg.runtime_root / "processes" / f"{attempt.attempt_id}.json"
    atomic_replace(
        manifest_path,
        {
            "process": {
                "task_id": task.task_id,
                "attempt_id": attempt.attempt_id,
                "fencing_token": attempt.current_fencing_token,
                "process_group_id": 9876,
                "observed_state": "running",
            }
        },
    )
    assert scheduler.expire_claim(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    token = recover_running_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert token == attempt.current_fencing_token + 1
    assert settle(cfg, event) == (True, [])
    assert journal.read(journal.snapshot(), before_tail + 1)["state"] == "committed"
    assert load_task(cfg, task.task_id).claim_control["active_claim"]["fencing_token"] == token


def test_claim_materialization_failure_does_not_leave_permanent_obligation(tmp_path, monkeypatch):
    cfg, task, journal = prepare(tmp_path)
    original = scheduler.atomic_replace

    def fail_attempt_write(path, value):
        if path == attempt_path(cfg.shared_root, task.task_id, 1):
            raise OSError("Attempt materialization failed")
        return original(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(scheduler, "atomic_replace", fail_attempt_write)
        with pytest.raises(OSError, match="materialization failed"):
            scheduler.claim_task(cfg, task.task_id, [0])
    current = load_task(cfg, task.task_id)
    assert current.state["projection"] == "queued"
    assert current.claim_control["active_claim"] is None
    event = journal.read(journal.snapshot(), 1)
    assert event["owner"] == "claim"
    assert event["state"] == "in_flight"
    assert settle(cfg, event) == (True, [])
    assert reference_for_generation(cfg, task.task_id, current.ready_generation) is not None
    assert scheduler.claim_task(cfg, task.task_id, [0]) is not None


class ProcessStopped(BaseException):
    """Abrupt owner death, without exception compensation or journal resolution."""


def interrupt_resolution(monkeypatch, owner):
    original = GroupRechecks.resolve

    def interrupted(self, ticket, **kwargs):
        event = self.read(self.snapshot(), ticket.sequence)
        if event["owner"] == owner:
            raise ProcessStopped
        return original(self, ticket, **kwargs)

    monkeypatch.setattr(GroupRechecks, "resolve", interrupted)


def running_attempt(cfg, task):
    attempt = scheduler.claim_task(cfg, task.task_id, [0])
    assert scheduler.authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    return attempt


def recover(cfg, task, attempt):
    from qqtools.plugins.qexp.runtime.attempt_recovery import recover_running_attempt
    from qqtools.plugins.qexp.runtime.store import atomic_replace

    atomic_replace(
        cfg.runtime_root / "processes" / f"{attempt.attempt_id}.json",
        {
            "process": {
                "task_id": task.task_id,
                "attempt_id": attempt.attempt_id,
                "fencing_token": attempt.current_fencing_token,
                "process_group_id": 9876,
                "observed_state": "running",
            }
        },
    )
    return recover_running_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)


def test_prelaunch_cancel_crash_has_reachable_terminal_owner(tmp_path, monkeypatch):
    from qqtools.plugins.qexp import lifecycle

    cfg, task, journal = prepare(tmp_path)
    scheduler.claim_task(cfg, task.task_id, [0])
    tail = journal.snapshot().tail
    original = lifecycle.atomic_replace

    def stop_after_attempt(path, value):
        original(path, value)
        if path == attempt_path(cfg.shared_root, task.task_id, 1):
            raise ProcessStopped

    with monkeypatch.context() as patch:
        patch.setattr(lifecycle, "atomic_replace", stop_after_attempt)
        with pytest.raises(ProcessStopped):
            scheduler.cancel_task(cfg, task.task_id)
    event = journal.read(journal.snapshot(), tail + 1)
    assert event["owner"] == "terminal"
    assert event["state"] == "in_flight"
    assert settle(cfg, event)[0]
    assert load_task(cfg, task.task_id).state["projection"] == "cancelled"


@pytest.mark.parametrize("owner", ["terminal", "claim_loss", "recovery"])
def test_old_obligation_accepts_identified_recovery_and_terminal_successors(tmp_path, monkeypatch, owner):
    from qqtools.plugins.qexp import lifecycle

    cfg, task, journal = prepare(tmp_path)
    attempt = running_attempt(cfg, task)
    if owner == "terminal":
        # Stop before the first Attempt write. A later recovered owner supersedes
        # the abandoned terminal request; its requested effect must not replay.
        original = lifecycle.atomic_replace

        def stop_before_attempt(path, value):
            if path == attempt_path(cfg.shared_root, task.task_id, 1):
                raise ProcessStopped
            return original(path, value)

        tail = journal.snapshot().tail
        with monkeypatch.context() as patch:
            patch.setattr(lifecycle, "atomic_replace", stop_before_attempt)
            with pytest.raises(ProcessStopped):
                scheduler.fail_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "old")
        assert scheduler.expire_claim(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
        token = recover(cfg, task, attempt)
    elif owner == "claim_loss":
        tail = journal.snapshot().tail
        with monkeypatch.context() as patch:
            interrupt_resolution(patch, owner)
            with pytest.raises(ProcessStopped):
                scheduler.expire_claim(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
        token = recover(cfg, task, attempt)
    else:
        assert scheduler.expire_claim(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
        tail = journal.snapshot().tail
        with monkeypatch.context() as patch:
            interrupt_resolution(patch, owner)
            with pytest.raises(ProcessStopped):
                recover(cfg, task, attempt)
        token = load_task(cfg, task.task_id).claim_control["active_claim"]["fencing_token"]
        assert scheduler.fail_attempt(cfg, task.task_id, attempt.attempt_id, token, "new_owner")
    event = journal.read(journal.snapshot(), tail + 1)
    assert event["owner"] == owner and event["state"] == "in_flight"
    before = load_task(cfg, task.task_id).to_dict()
    assert settle(cfg, event) == (True, [])
    assert load_task(cfg, task.task_id).to_dict() == before
    assert token == attempt.current_fencing_token + 1


@pytest.mark.parametrize("owner", ["claim", "launch"])
def test_interrupted_obligation_accepts_detached_orphan(tmp_path, owner):
    cfg, task, journal = prepare(tmp_path)
    if owner == "claim":
        tail = journal.snapshot().tail
        run_crash(cfg, task.task_id, owner, "after_attempt")
        claim = load_task(cfg, task.task_id).claim_control["active_claim"]
        assert scheduler.authorize_launch(cfg, task.task_id, claim["attempt_id"], claim["fencing_token"])
    else:
        scheduler.claim_task(cfg, task.task_id, [0])
        tail = journal.snapshot().tail
        run_crash(cfg, task.task_id, owner, "after_attempt")
        claim = load_task(cfg, task.task_id).claim_control["active_claim"]
    assert scheduler.expire_claim(cfg, task.task_id, claim["attempt_id"], claim["fencing_token"])
    assert load_task(cfg, task.task_id).state["projection"] == "blocked"
    event = journal.read(journal.snapshot(), tail + 1)
    assert event["owner"] == owner and event["state"] == "in_flight"
    assert settle(cfg, event) == (True, [])
