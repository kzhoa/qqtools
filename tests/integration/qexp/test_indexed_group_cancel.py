"""Group cancellation consumes certified membership and changes, not Task history."""

import multiprocessing
import os

import pytest

from qqtools.plugins.qexp import submit
from qqtools.plugins.qexp.commands.group import group_control, reconcile_group_cancel_operations
from qqtools.plugins.qexp.commands.task import retry
from qqtools.plugins.qexp.runtime.group_discovery.service import GroupDiscoveryService
from qqtools.plugins.qexp.runtime.operation_store import locate_operation_path
from qqtools.plugins.qexp.runtime.paths import group_path, task_path
from qqtools.plugins.qexp.runtime.process_evidence import ProcessEvidence
from qqtools.plugins.qexp.runtime.store import read_json
from qqtools.plugins.qexp.runtime.tasks import load_task
from qqtools.plugins.qexp.scheduler import authorize_launch, claim_task, fail_attempt
from tests.helpers.qexp_discovery import isolated_group

pytestmark = pytest.mark.integration


def discover(cfg):
    service = GroupDiscoveryService(cfg.shared_root, "experiment")
    try:
        for _ in range(20000):
            if service.advance()["state"] == "complete":
                return
        raise AssertionError("membership discovery did not converge")
    finally:
        service.request_close()
        for _ in range(10000):
            if service.is_closed:
                break
            service.advance()
        assert service.is_closed


def control(cfg, operation_id):
    return read_json(locate_operation_path(cfg, "group_control", operation_id))["group_control"]


def finish(cfg, operation_id):
    for _ in range(100):
        current = control(cfg, operation_id)
        if current["state"] == "completed":
            return current
        reconcile_group_cancel_operations(cfg, "experiment", include_legacy=False)
    raise AssertionError(f"indexed cancellation stalled: {current}")


def guard_task_scans(monkeypatch, cfg):
    original = os.scandir

    def guarded(path):
        if not isinstance(path, int):
            assert os.path.abspath(os.fspath(path)) != str(cfg.shared_root / "tasks"), (
                "cancellation enumerated Task history"
            )
        return original(path)

    monkeypatch.setattr(os, "scandir", guarded)


def test_cancel_uses_bounded_members_and_excludes_later_submissions(tmp_path, monkeypatch):
    cfg = isolated_group(tmp_path, tail=0)
    tasks = [submit(cfg, ["true"], group="experiment") for _ in range(4)]
    discover(cfg)
    # A unrelated malformed historical record must never enter candidate discovery.
    task_path(cfg.shared_root, "unrelated-history").write_text("not JSON")
    with monkeypatch.context() as guard:
        guard_task_scans(guard, cfg)
        data = group_control(cfg, "experiment", "cancel")
        operation_id = data["cancellation_operation"]["operation_id"]
        assert control(cfg, operation_id)["state"] != "completed"
        assert sum(load_task(cfg, task.task_id).state["projection"] == "cancelled" for task in tasks) <= 1
    task_path(cfg.shared_root, "unrelated-history").unlink()
    later = submit(cfg, ["true"], group="experiment")
    with monkeypatch.context() as guard:
        guard_task_scans(guard, cfg)
        completed = finish(cfg, operation_id)
    assert completed["progress"]["target_tasks"] == 4
    assert completed["progress"]["queued_cancelled"] == 4
    assert all(load_task(cfg, task.task_id).state["projection"] == "cancelled" for task in tasks)
    assert load_task(cfg, later.task_id).state["projection"] == "queued"


def test_unconfirmed_membership_waits_then_resumes_without_history_fallback(tmp_path, monkeypatch):
    cfg = isolated_group(tmp_path, tail=0)
    task = submit(cfg, ["true"], group="experiment")
    with monkeypatch.context() as guard:
        guard_task_scans(guard, cfg)
        result = group_control(cfg, "experiment", "cancel")
        operation_id = result["cancellation_operation"]["operation_id"]
        reconcile_group_cancel_operations(cfg, include_legacy=False)
    assert control(cfg, operation_id)["state"] != "completed"
    assert load_task(cfg, task.task_id).state["projection"] == "queued"
    discover(cfg)
    assert finish(cfg, operation_id)["state"] == "completed"


def test_retry_after_completed_member_receipt_is_rechecked(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    first = submit(cfg, ["true"], group="experiment")
    submit(cfg, ["true"], group="experiment")
    attempt = claim_task(cfg, first.task_id, [0])
    assert authorize_launch(cfg, first.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert fail_attempt(cfg, first.task_id, attempt.attempt_id, attempt.current_fencing_token, "failure")
    discover(cfg)
    result = group_control(cfg, "experiment", "cancel")
    operation_id = result["cancellation_operation"]["operation_id"]
    for _ in range(10):
        if control(cfg, operation_id)["discovery"]["member_cursor"] >= 1:
            break
        reconcile_group_cancel_operations(cfg, include_legacy=False)
    assert control(cfg, operation_id)["state"] != "completed"
    retry(cfg, first.task_id)
    assert claim_task(cfg, first.task_id, [0]) is None
    completed = finish(cfg, operation_id)
    assert completed["progress"]["target_tasks"] == 2
    assert load_task(cfg, first.task_id).state["projection"] == "cancelled"
    assert load_task(cfg, first.task_id).control["cancellation_operation_id"] == operation_id


def test_unexplained_missing_task_is_not_successful_cancellation(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    task = submit(cfg, ["true"], group="experiment")
    discover(cfg)
    task_path(cfg.shared_root, task.task_id).unlink()
    result = group_control(cfg, "experiment", "cancel")
    operation_id = result["cancellation_operation"]["operation_id"]
    for _ in range(8):
        reconcile_group_cancel_operations(cfg, include_legacy=False)
    current = control(cfg, operation_id)
    assert current["state"] != "completed"
    assert current["blocked_reason"]


def test_reconciliation_never_overwrites_newer_cancellation_snapshot(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    submit(cfg, ["true"], group="experiment")
    discover(cfg)
    first = group_control(cfg, "experiment", "cancel")["cancellation_operation"]["operation_id"]
    second = group_control(cfg, "experiment", "cancel", terminate_running=True)["cancellation_operation"][
        "operation_id"
    ]
    finish(cfg, first)
    finish(cfg, second)
    assert read_json(group_path(cfg.shared_root, "experiment"))["cancellation_operation"]["operation_id"] == second


def test_census_rejects_partial_terminal_transition_predating_journal(tmp_path, monkeypatch):
    from qqtools.plugins.qexp import lifecycle

    cfg = isolated_group(tmp_path, tail=0)
    task = submit(cfg, ["true"], group="experiment")
    attempt = claim_task(cfg, task.task_id, [0])
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    discover(cfg)

    def fail_task_write(*args, **kwargs):
        raise OSError("terminal Attempt committed before Task")

    with monkeypatch.context() as patch:
        patch.setattr(lifecycle, "save_task", fail_task_write)
        with pytest.raises(OSError, match="before Task"):
            fail_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "failure")
    assert load_task(cfg, task.task_id).state["projection"] == "running"
    result = group_control(cfg, "experiment", "cancel")
    operation_id = result["cancellation_operation"]["operation_id"]
    for _ in range(5):
        reconcile_group_cancel_operations(cfg, include_legacy=False)
    assert control(cfg, operation_id)["state"] != "completed"
    assert control(cfg, operation_id)["progress"]["running_allowed"] == 0
    assert fail_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "failure")
    assert finish(cfg, operation_id)["state"] == "completed"


def test_human_cancel_activates_pending_work_without_listing_task_history(tmp_path, monkeypatch, capsys):
    from qqtools.plugins.qexp.agent.context import MachineRuntime
    from qqtools.plugins.qexp.cli import entrypoint, project_handlers

    cfg = isolated_group(tmp_path, tail=0)
    submit(cfg, ["true"], group="experiment")
    machine_root = tmp_path / "machine-runtime"
    MachineRuntime(machine_root).ensure_binding(cfg.shared_root, cfg.machine_name)
    activated = []

    def activate(config, *, reason, **kwargs):
        snapshot = read_json(group_path(config.shared_root, "experiment"))["cancellation_operation"]
        assert snapshot["state"] == "converging"
        assert control(config, snapshot["operation_id"])["discovery"]
        activated.append(reason)

    def no_history(*args, **kwargs):
        pytest.fail("human cancellation rendered unused full-history Task summaries")

    with monkeypatch.context() as patch:
        patch.setattr(project_handlers, "ensure_local_agent_active", activate)
        patch.setattr(project_handlers.observer, "list_tasks", no_history)
        assert (
            entrypoint.main(
                [
                    "--project",
                    str(cfg.shared_root),
                    "--machine",
                    cfg.machine_name,
                    "--runtime-root",
                    str(cfg.runtime_root),
                    "--machine-runtime-root",
                    str(machine_root),
                    "group",
                    "cancel",
                    "experiment",
                    "--format=human",
                ]
            )
            == 0
        )
    assert activated == ["group-cancel"]
    assert "converging" in capsys.readouterr().out


def test_completed_cleanup_proves_missing_member_without_task_history(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.commands.cleanup import clean

    cfg = isolated_group(tmp_path, tail=0)
    first = submit(cfg, ["true"], group="experiment")
    second = submit(cfg, ["true"], group="experiment")
    discover(cfg)
    operation_id = group_control(cfg, "experiment", "cancel")["cancellation_operation"]["operation_id"]
    for _ in range(10):
        if load_task(cfg, first.task_id).state["projection"] == "cancelled":
            break
        reconcile_group_cancel_operations(cfg, include_legacy=False)
    assert control(cfg, operation_id)["state"] != "completed"
    clean(cfg, task_id=first.task_id)
    assert not task_path(cfg.shared_root, first.task_id).exists()
    with monkeypatch.context() as guard:
        guard_task_scans(guard, cfg)
        completed = finish(cfg, operation_id)
    assert completed["state"] == "completed"
    assert load_task(cfg, second.task_id).state["projection"] == "cancelled"
    assert completed["progress"]["target_tasks"] == 1
    assert completed["progress"]["queued_cancelled"] == 2


def test_invalidated_generation_revisits_members_before_completion(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runtime.group_discovery.rechecks import GroupRechecks

    cfg = isolated_group(tmp_path, tail=0)
    first = submit(cfg, ["true"], group="experiment")
    submit(cfg, ["true"], group="experiment")
    attempt = claim_task(cfg, first.task_id, [0])
    assert authorize_launch(cfg, first.task_id, attempt.attempt_id, attempt.current_fencing_token)
    assert fail_attempt(cfg, first.task_id, attempt.attempt_id, attempt.current_fencing_token, "failure")
    discover(cfg)
    operation_id = group_control(cfg, "experiment", "cancel")["cancellation_operation"]["operation_id"]
    for _ in range(10):
        if control(cfg, operation_id)["discovery"]["member_cursor"] >= 1:
            break
        reconcile_group_cancel_operations(cfg, include_legacy=False)
    assert control(cfg, operation_id)["state"] != "completed"
    generation = GroupRechecks(cfg.shared_root, "experiment").snapshot().generation

    def fail_publication(*args, **kwargs):
        raise OSError("event publication failed before retry")

    with monkeypatch.context() as patch:
        patch.setattr(GroupRechecks, "begin", fail_publication)
        retry(cfg, first.task_id)
    assert GroupRechecks(cfg.shared_root, "experiment").snapshot().generation != generation
    assert load_task(cfg, first.task_id).state["projection"] == "queued"
    assert finish(cfg, operation_id)["state"] == "completed"
    assert load_task(cfg, first.task_id).state["projection"] == "cancelled"


@pytest.mark.parametrize("after_receipt", [False, True])
def test_receipt_recovery_after_process_death_does_not_double_count(tmp_path, after_receipt):
    from qqtools.plugins.qexp.commands import group_cancel

    cfg = isolated_group(tmp_path, tail=0)
    tasks = [submit(cfg, ["true"], group="experiment") for _ in range(2)]
    discover(cfg)

    def interrupt_receipt():
        original = group_cancel._write_receipt

        def crash(*args, **kwargs):
            if after_receipt:
                original(*args, **kwargs)
            os._exit(74)

        group_cancel._write_receipt = crash
        group_control(cfg, "experiment", "cancel")

    process = multiprocessing.get_context("fork").Process(target=interrupt_receipt)
    process.start()
    try:
        process.join(10)
        assert not process.is_alive()
        assert process.exitcode == 74
    finally:
        if process.is_alive():
            process.kill()
            process.join(5)
        process.close()
    operation_id = read_json(group_path(cfg.shared_root, "experiment"))["cancellation_operation"]["operation_id"]
    assert control(cfg, operation_id)["discovery"]["pending_receipt"] is not None
    completed = finish(cfg, operation_id)
    assert completed["progress"]["target_tasks"] == 2
    assert completed["progress"]["queued_cancelled"] == 2
    assert all(load_task(cfg, task.task_id).state["projection"] == "cancelled" for task in tasks)
    assert completed["discovery"]["pending_receipt"] is None


def test_terminating_cancel_waits_for_runtime_ack_and_consumes_terminal_change(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runtime.store import atomic_replace
    from qqtools.plugins.qexp.scheduler import reconcile_running_tasks

    cfg = isolated_group(tmp_path, tail=0)
    task = submit(cfg, ["true"], group="experiment")
    attempt = claim_task(cfg, task.task_id, [0])
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    atomic_replace(
        cfg.runtime_root / "processes" / f"{attempt.attempt_id}.json",
        {
            "process": {
                "task_id": task.task_id,
                "attempt_id": attempt.attempt_id,
                "fencing_token": attempt.current_fencing_token,
                "process_group_id": 9876,
                "observed_state": "running",
                "supervisor": "agent",
            }
        },
    )
    discover(cfg)
    operation_id = group_control(cfg, "experiment", "cancel", terminate_running=True)["cancellation_operation"][
        "operation_id"
    ]
    for _ in range(5):
        reconcile_group_cancel_operations(cfg, include_legacy=False)
    pending = control(cfg, operation_id)
    assert pending["state"] == "waiting_ack"
    assert pending["pending_machine_acknowledgements"] == {cfg.machine_name: [task.task_id]}
    monkeypatch.setattr(
        "qqtools.plugins.qexp.scheduler.inspect_group_identity",
        lambda *_args: ProcessEvidence(state="absent"),
    )
    reconcile_running_tasks(cfg)
    completed = finish(cfg, operation_id)
    assert completed["pending_machine_acknowledgements"] == {}
    assert completed["progress"]["termination_pending"] == 0
    assert completed["progress"]["termination_acknowledged"] == 1


@pytest.mark.parametrize("invalidate", [False, True])
def test_cancel_resumes_its_own_attempt_before_task_terminal_write(tmp_path, monkeypatch, invalidate):
    from qqtools.plugins.qexp import lifecycle
    from qqtools.plugins.qexp.runtime.group_discovery.rechecks import GroupRechecks

    cfg = isolated_group(tmp_path, tail=0)
    task = submit(cfg, ["true"], group="experiment")
    attempt = claim_task(cfg, task.task_id, [0])
    discover(cfg)

    def interrupt(*args, **kwargs):
        raise OSError("cancel Attempt committed before Task")

    def fail_begin(*args, **kwargs):
        raise OSError("journal publication failed")

    with monkeypatch.context() as patch:
        patch.setattr(lifecycle, "save_task", interrupt)
        if invalidate:
            patch.setattr(GroupRechecks, "begin", fail_begin)
        with pytest.raises(OSError, match="before Task"):
            group_control(cfg, "experiment", "cancel")
    operation_id = read_json(group_path(cfg.shared_root, "experiment"))["cancellation_operation"]["operation_id"]
    assert load_task(cfg, task.task_id).state["projection"] == "running"
    completed = finish(cfg, operation_id)
    assert completed["progress"]["prelaunch_cancelled"] == 1
    current = load_task(cfg, task.task_id)
    assert current.state["projection"] == "cancelled"
    assert current.attempt_control["current_attempt_number"] == attempt.attempt_number


@pytest.mark.parametrize("reason", ["source_replaced", "member_conflict", "invalid_publication_cursor"])
def test_degraded_coverage_cannot_complete_at_fixed_watermark(tmp_path, monkeypatch, reason):
    from dataclasses import replace

    from qqtools.plugins.qexp.runtime.group_discovery.coverage import GroupCoverage

    cfg = isolated_group(tmp_path, tail=0)
    submit(cfg, ["true"], group="experiment")
    discover(cfg)
    operation_id = group_control(cfg, "experiment", "cancel")["cancellation_operation"]["operation_id"]
    assert control(cfg, operation_id)["discovery"]["member_cursor"] == 1
    original = GroupCoverage.status

    def degraded(self):
        return replace(original(self), reason=reason)

    with monkeypatch.context() as patch:
        patch.setattr(GroupCoverage, "status", degraded)
        for _ in range(5):
            reconcile_group_cancel_operations(cfg, include_legacy=False)
        assert control(cfg, operation_id)["state"] != "completed"
    assert finish(cfg, operation_id)["state"] == "completed"


def test_default_cancel_does_not_wait_for_later_terminating_operation(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    task = submit(cfg, ["true"], group="experiment")
    attempt = claim_task(cfg, task.task_id, [0])
    assert authorize_launch(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token)
    discover(cfg)
    default = group_control(cfg, "experiment", "cancel")["cancellation_operation"]["operation_id"]
    terminating = group_control(cfg, "experiment", "cancel", terminate_running=True)["cancellation_operation"][
        "operation_id"
    ]
    completed = finish(cfg, default)
    assert completed["progress"]["running_allowed"] == 1
    assert completed["pending_machine_acknowledgements"] == {}
    assert control(cfg, terminating)["state"] != "completed"
    assert load_task(cfg, task.task_id).control["terminate_running"] is True


def test_journal_member_identity_conflict_blocks_without_advancing_cursor(tmp_path):
    from qqtools.plugins.qexp.runtime.authority_lock import authority_locks
    from qqtools.plugins.qexp.runtime.group_discovery.rechecks import GroupRechecks

    cfg = isolated_group(tmp_path, tail=0)
    task = submit(cfg, ["true"], group="experiment")
    discover(cfg)
    operation_id = group_control(cfg, "experiment", "cancel")["cancellation_operation"]["operation_id"]
    journal = GroupRechecks(cfg.shared_root, "experiment")
    with authority_locks(cfg, task):
        ticket = journal.begin("wrong-task", task.submission_operation_id, 1, "task_cancel", {})
        journal.resolve(ticket)
    for _ in range(8):
        reconcile_group_cancel_operations(cfg, include_legacy=False)
    current = control(cfg, operation_id)
    assert current["state"] == "blocked"
    assert current["blocked_reason"] == "recheck_membership_identity_mismatch"
    assert current["discovery"]["journal_cursor"] < ticket.sequence


def test_cleanup_between_cancel_effect_and_receipt_transfers_proof(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.commands import group as group_commands
    from qqtools.plugins.qexp.commands.cleanup import clean

    cfg = isolated_group(tmp_path, tail=0)
    task = submit(cfg, ["true"], group="experiment")
    discover(cfg)
    original = group_commands.save_task

    def interrupt(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("cancel Task persisted without receipt")

    with monkeypatch.context() as patch:
        patch.setattr(group_commands, "save_task", interrupt)
        with pytest.raises(OSError, match="without receipt"):
            group_control(cfg, "experiment", "cancel")
    operation_id = read_json(group_path(cfg.shared_root, "experiment"))["cancellation_operation"]["operation_id"]
    clean(cfg, task_id=task.task_id)
    assert not task_path(cfg.shared_root, task.task_id).exists()
    completed = finish(cfg, operation_id)
    assert completed["progress"]["target_tasks"] == 0
    assert completed["progress"]["queued_cancelled"] == 1
