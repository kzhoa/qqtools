"""Reclamation retains live recovery proof and converges after interrupted deletion."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp.runtime.group_discovery import rechecks
from qqtools.plugins.qexp.runtime.group_discovery.coverage import GroupCoverage
from qqtools.plugins.qexp.runtime.group_discovery.rechecks import GroupRechecks
from qqtools.plugins.qexp.runtime.group_namespace import group_authority_identity
from qqtools.plugins.qexp.runtime.operation_store import archive_operation, write_active_operation
from qqtools.plugins.qexp.runtime.paths import shared_paths, submission_path
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from tests.helpers.qexp_discovery import discover_group, isolated_group, source_file

pytestmark = pytest.mark.integration


def journal_with_events(cfg, count=4, *, unresolved=None):
    journal = GroupRechecks(cfg.shared_root, "experiment")
    journal.snapshot(initialize=True)
    tickets = []
    for number in range(1, count + 1):
        ticket = journal.begin(f"task-{number}", "batch", number, "retry", {})
        if number != unresolved:
            journal.resolve(ticket)
        tickets.append(ticket)
    return journal, tickets


def event_path(cfg, position, sequence):
    return (
        GroupCoverage(cfg.shared_root, "experiment").directory
        / "rechecks"
        / str(position.generation)
        / f"{sequence}.json"
    )


def subscriber(cfg, position, operation="cancel-a", *, cursor=0, kind="cancel", state="converging"):
    value = {
        "group_control": {
            "operation_id": operation,
            "group_name": "experiment",
            "operation_type": kind,
            "state": state,
            "discovery": {
                "authority": group_authority_identity(cfg.shared_root),
                "generation": position.generation,
                "journal_cursor": cursor,
            },
        }
    }
    write_active_operation(cfg, "group_control", operation, value)
    return value


def maintenance(cfg):
    from qqtools.plugins.qexp.runtime.group_discovery.maintenance import GroupMaintenance

    return GroupMaintenance(cfg.shared_root, "experiment")


def advance_until(owner, predicate, *, steps=2000):
    result = None
    for _ in range(steps):
        result = owner.advance()
        if predicate():
            return result
    raise AssertionError(f"maintenance did not converge: {result}")


def close(owner):
    owner.request_close()
    for _ in range(100):
        if owner.is_closed:
            return
        owner.advance()
    raise AssertionError("maintenance retained open resources")


@pytest.fixture
def advancing_clock(monkeypatch):
    from qqtools.plugins.qexp.runtime.group_discovery import maintenance as module

    ticks = iter(range(1000000))
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: float(next(ticks))))


def test_journal_reclamation_is_bounded_and_preserves_unresolved_event(tmp_path):
    cfg = isolated_group(tmp_path)
    journal, tickets = journal_with_events(cfg, unresolved=3)
    position = journal.snapshot()
    assert journal.reclaim(position, 4, max_events=1) == 1
    assert not event_path(cfg, position, 1).exists()
    assert event_path(cfg, position, 2).exists()
    assert journal.reclaim(position, 4, max_events=64) == 1
    retained = journal.retention(position)
    assert (retained.floor, retained.deleted) == (2, 2)
    assert journal.read(position, 3)["state"] == "in_flight"
    assert journal.read(position, 4)["state"] == "committed"
    with pytest.raises(ValueError, match="retired"):
        journal.read(position, 1)
    journal.resolve(tickets[0])  # Late acknowledgement cannot recreate retired history.
    assert not event_path(cfg, position, 1).exists()
    journal.resolve(tickets[2])
    assert journal.reclaim(position, 4, max_events=64) == 2


@pytest.mark.parametrize("cut", ["floor", "unlink", "deleted"])
def test_reclaim_process_crash_recovers_exact_pending_deletion(tmp_path, cut):
    """Use abrupt real process exit so Python cleanup cannot repair the cut."""
    cfg = isolated_group(tmp_path)
    journal, _ = journal_with_events(cfg, count=2)
    position = journal.snapshot()
    child = os.fork()
    if child == 0:
        original_replace = rechecks.atomic_replace
        original_unlink = Path.unlink

        def replace(path, value):
            original_replace(path, value)
            if path.name == "retention.json":
                if cut == "floor" and value["floor"] == 1 and value["deleted"] == 0:
                    os._exit(79)
                if cut == "deleted" and value["deleted"] == 1:
                    os._exit(79)

        def unlink(path, *args, **kwargs):
            result = original_unlink(path, *args, **kwargs)
            if cut == "unlink" and path == event_path(cfg, position, 1):
                os._exit(79)
            return result

        rechecks.atomic_replace = replace
        Path.unlink = unlink
        journal.reclaim(position, 1)
        os._exit(3)
    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 79
    restored = GroupRechecks(cfg.shared_root, "experiment")
    restored.reclaim(restored.snapshot(), 1)
    assert not event_path(cfg, position, 1).exists()
    assert restored.retention(restored.snapshot()).deleted == 1
    assert restored.read(restored.snapshot(), 2)["task_id"] == "task-2"


def test_missing_unretired_event_is_not_success_and_generation_resets_floor(tmp_path):
    cfg = isolated_group(tmp_path)
    journal, _ = journal_with_events(cfg, count=2)
    old = journal.snapshot()
    event_path(cfg, old, 1).unlink()
    with pytest.raises((ValueError, OSError)):
        journal.reclaim(old, 2)
    assert journal.retention(old).floor == 0
    fresh = journal.invalidate()
    assert journal.retention(fresh).floor == 0
    with pytest.raises(ValueError):
        journal.reclaim(old, 2)


def test_slowest_subscriber_bounds_retirement_and_completion_releases_it(tmp_path, advancing_clock):
    cfg = isolated_group(tmp_path)
    journal, _ = journal_with_events(cfg)
    position = journal.snapshot()
    slow = subscriber(cfg, position, cursor=1)
    subscriber(cfg, position, operation="remove-b", cursor=3, kind="worker_remove_v2")
    owner = maintenance(cfg)
    try:
        advance_until(owner, lambda: journal.retention(position).deleted == 1)
        for _ in range(100):
            owner.advance()
        assert journal.retention(position).floor == 1
        assert journal.read(position, 2)["state"] == "committed"
        slow["group_control"]["state"] = "completed"
        archive_operation(cfg, "group_control", "cancel-a", slow)
        advance_until(owner, lambda: journal.retention(position).deleted == 3)
        assert event_path(cfg, position, 4).exists()
    finally:
        close(owner)


def test_new_subscriber_starts_at_tail_while_certified_backlog_is_reclaimed(tmp_path, advancing_clock):
    from qqtools.plugins.qexp.commands.group_cancel import initialize_cancel_discovery

    cfg = isolated_group(tmp_path)
    journal, _ = journal_with_events(cfg, count=12)
    position = journal.snapshot()
    owner = maintenance(cfg)
    try:
        advance_until(owner, lambda: journal.retention(position).deleted >= 1)
        control = {"group_name": "experiment"}
        initialize_cancel_discovery(cfg, control)
        assert control["discovery"]["journal_cursor"] == 12
        fresh = subscriber(cfg, position, operation="new", cursor=12)
        fresh["group_control"]["discovery"] = control["discovery"]
        write_active_operation(cfg, "group_control", "new", fresh)
        advance_until(owner, lambda: journal.retention(position).deleted == 12)
        ticket = journal.begin("later", "batch", 13, "retry", {})
        assert journal.read(journal.snapshot(), ticket.sequence)["state"] == "in_flight"
    finally:
        close(owner)


@pytest.mark.parametrize("damage", ["malformed", "no_discovery", "wrong_authority"])
def test_uncertain_subscriber_keeps_journal(tmp_path, advancing_clock, damage):
    cfg = isolated_group(tmp_path)
    journal, _ = journal_with_events(cfg)
    position = journal.snapshot()
    value = subscriber(cfg, position, cursor=3)
    if damage == "malformed":
        value = {"group_control": None}
    elif damage == "no_discovery":
        value["group_control"].pop("discovery")
    else:
        value["group_control"]["discovery"]["authority"] = {}
    write_active_operation(cfg, "group_control", "cancel-a", value)
    owner = maintenance(cfg)
    try:
        for _ in range(100):
            owner.advance()
        assert journal.retention(position).deleted == 0
        assert journal.read(position, 1)["state"] == "committed"
    finally:
        close(owner)


def test_completed_operation_receipts_reclaimed_without_touching_live_receipts(tmp_path, advancing_clock):
    cfg = isolated_group(tmp_path)
    journal, _ = journal_with_events(cfg)
    position = journal.snapshot()
    finished = subscriber(cfg, position, operation="done", cursor=4, state="completed")
    archive_operation(cfg, "group_control", "done", finished)
    subscriber(cfg, position, operation="live", cursor=0)
    coverage = GroupCoverage(cfg.shared_root, "experiment")
    for name in ("done", "live"):
        for sequence in range(10):
            atomic_replace(
                coverage.directory / "operations" / name / "receipt-generation" / f"{sequence}.json",
                {"receipt": sequence},
            )
    owner = maintenance(cfg)
    try:
        advance_until(owner, lambda: not (coverage.directory / "operations/done").exists())
        assert len(list((coverage.directory / "operations/live/receipt-generation").iterdir())) == 10
        archived = shared_paths(cfg.shared_root)["group_control"] / "done.json"
        assert read_json(archived) == finished
        assert journal.retention(position).deleted == 0
    finally:
        close(owner)


def test_invalidated_generation_is_removed_and_current_generation_remains_readable(tmp_path, advancing_clock):
    cfg = isolated_group(tmp_path)
    journal, _ = journal_with_events(cfg)
    old = journal.snapshot()
    journal.invalidate()
    ticket = journal.begin("current-task", "batch", 1, "retry", {})
    current = journal.snapshot()
    owner = maintenance(cfg)
    try:
        advance_until(owner, lambda: not event_path(cfg, old, 1).parent.exists())
        assert journal.read(current, ticket.sequence)["state"] == "in_flight"
    finally:
        close(owner)


def test_confirmed_source_audit_reclaimed_but_member_proof_survives_restart(tmp_path, advancing_clock):
    cfg = isolated_group(tmp_path, tail=3)
    source_file(
        submission_path(cfg.shared_root, "batch"), operation="batch", tasks=["a", "b", "c"], sequences=[1, 2, 3]
    )
    discover_group(cfg)
    coverage = GroupCoverage(cfg.shared_root, "experiment")
    debts = coverage.directory / "maintenance/sources"
    assert list(debts.glob("*.json")), "source completion must hand off audit cleanup durably"
    scratch = next((coverage.directory / "sources").glob("*/*/receipt.json")).parent
    assert (scratch / "qualification/task.digests").exists()
    owner = maintenance(cfg)
    try:
        advance_until(owner, lambda: not list(debts.glob("*.json")))
    finally:
        close(owner)
    assert not (scratch / "qualification/task.digests").exists()
    assert not (scratch / "qualification/task-audit").exists()
    assert not (scratch / "qualification/sequence-audit").exists()
    assert (scratch / "qualification/task.refs").exists()
    assert (scratch / "projection/events.jsonl").exists()
    assert (scratch / "receipt.json").exists()
    discover_group(cfg)
    assert [coverage.read_member(i).task_id for i in range(1, 4)] == ["a", "b", "c"]


def test_source_cleanup_handoff_failure_keeps_recovery_locator(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runtime.group_discovery import maintenance as module
    from qqtools.plugins.qexp.runtime.group_discovery.service import GroupDiscoveryService

    cfg = isolated_group(tmp_path)
    source_file(submission_path(cfg.shared_root, "batch"), operation="batch")
    owner = GroupDiscoveryService(cfg.shared_root, "experiment")
    original = module.enqueue_source_cleanup

    def crash(*args, **kwargs):
        raise OSError("cleanup handoff unavailable")

    with monkeypatch.context() as guard:
        guard.setattr(module, "enqueue_source_cleanup", crash)
        result = advance_until(owner, lambda: owner._last_status.get("state") == "error")
        assert "cleanup handoff unavailable" in result["reason"]
        locator = GroupCoverage(cfg.shared_root, "experiment").directory / "active-source.json"
        assert read_json(locator)["operation_id"] == "batch"
    close(owner)
    # A fresh service resumes the still-owned source and publishes its handoff.
    monkeypatch.setattr(module, "enqueue_source_cleanup", original)
    discover_group(cfg)
    assert not locator.exists()
    assert list((locator.parent / "maintenance/sources").glob("*.json"))


def test_damaged_source_receipt_retains_cleanup_debt_and_all_proof(tmp_path, advancing_clock):
    cfg = isolated_group(tmp_path)
    source_file(submission_path(cfg.shared_root, "batch"), operation="batch")
    discover_group(cfg)
    coverage = GroupCoverage(cfg.shared_root, "experiment")
    receipt = next((coverage.directory / "sources").glob("*/*/receipt.json"))
    receipt.write_text("{")
    owner = maintenance(cfg)
    try:
        for _ in range(150):
            owner.advance()
        assert list((coverage.directory / "maintenance/sources").glob("*.json"))
        assert (receipt.parent / "qualification/task.digests").exists()
        assert (receipt.parent / "qualification/task.refs").exists()
    finally:
        close(owner)


def test_maintenance_never_enumerates_task_or_archived_operation_history(tmp_path, monkeypatch, advancing_clock):
    cfg = isolated_group(tmp_path)
    journal, _ = journal_with_events(cfg)
    position = journal.snapshot()
    forbidden = {
        cfg.shared_root / "tasks",
        shared_paths(cfg.shared_root)["submissions"],
        shared_paths(cfg.shared_root)["group_control"],
    }
    original = os.scandir

    def guarded(path):
        if not isinstance(path, int):
            assert Path(path) not in forbidden, f"maintenance enumerated retained history: {path}"
        return original(path)

    owner = maintenance(cfg)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(os, "scandir", guarded)
            advance_until(owner, lambda: journal.retention(position).deleted == 4)
    finally:
        close(owner)


def test_real_cancel_then_removal_converge_while_maintenance_runs(tmp_path, advancing_clock):
    from qqtools.plugins.qexp import submit
    from qqtools.plugins.qexp.commands.group import change_worker, group_control, reconcile_group_cancel_operations
    from qqtools.plugins.qexp.runtime.operation_store import locate_operation_path
    from qqtools.plugins.qexp.runtime.tasks import load_task

    cfg = isolated_group(tmp_path, tail=0)
    tasks = [submit(cfg, ["true"], group="experiment") for _ in range(3)]
    discover_group(cfg)
    removal = change_worker(cfg, "experiment", "g1", "remove")["worker_control"]["operation_id"]
    cancellation = group_control(cfg, "experiment", "cancel")["cancellation_operation"]["operation_id"]
    owner = maintenance(cfg)
    try:
        for _ in range(500):
            owner.advance()
            reconcile_group_cancel_operations(cfg, "experiment", include_legacy=False)
            controls = [
                read_json(locate_operation_path(cfg, "group_control", key))["group_control"]
                for key in (removal, cancellation)
            ]
            if all(control["state"] == "completed" for control in controls):
                break
        assert all(control["state"] == "completed" for control in controls)
        assert all(load_task(cfg, task.task_id).state["projection"] == "cancelled" for task in tasks)
        journal = GroupRechecks(cfg.shared_root, "experiment")
        position = journal.snapshot()
        assert position.tail > 0
        advance_until(owner, lambda: journal.retention(position).deleted == position.tail)
        coverage = GroupCoverage(cfg.shared_root, "experiment")
        advance_until(owner, lambda: not (coverage.directory / "operations" / cancellation).exists())
        assert [coverage.read_member(i).task_id for i in range(1, 4)] == [task.task_id for task in tasks]
    finally:
        close(owner)


def test_source_cleanup_process_crash_keeps_handoff_and_resumes(tmp_path, advancing_clock):
    cfg = isolated_group(tmp_path)
    source_file(submission_path(cfg.shared_root, "batch"), operation="batch")
    discover_group(cfg)
    directory = GroupCoverage(cfg.shared_root, "experiment").directory
    child = os.fork()
    if child == 0:
        original = Path.unlink

        def unlink(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            if path.name == "task.digests":
                os._exit(79)
            return result

        Path.unlink = unlink
        owner = maintenance(cfg)
        for _ in range(1000):
            owner.advance()
        os._exit(3)
    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 79
    assert list((directory / "maintenance/sources").glob("*.json"))
    owner = maintenance(cfg)
    try:
        advance_until(owner, lambda: not list((directory / "maintenance/sources").glob("*.json")))
        assert GroupCoverage(cfg.shared_root, "experiment").read_member(1).task_id == "task-a"
    finally:
        close(owner)


def test_archive_before_active_retirement_does_not_release_receipts(tmp_path, advancing_clock):
    cfg = isolated_group(tmp_path)
    journal, _ = journal_with_events(cfg)
    position = journal.snapshot()
    value = subscriber(cfg, position, cursor=0)
    completed = {"group_control": dict(value["group_control"], state="completed")}
    archived = shared_paths(cfg.shared_root)["group_control"] / "cancel-a.json"
    # Crash after archive replacement, before retiring the previous active truth.
    atomic_replace(archived, completed)
    receipts = GroupCoverage(cfg.shared_root, "experiment").directory / "operations/cancel-a"
    atomic_replace(receipts / "generation/1.json", {"preserved": True})
    owner = maintenance(cfg)
    try:
        for _ in range(150):
            owner.advance()
        assert read_json(receipts / "generation/1.json") == {"preserved": True}
        assert journal.retention(position).floor == 0
        archive_operation(cfg, "group_control", "cancel-a", completed)
        advance_until(owner, lambda: not receipts.exists())
    finally:
        close(owner)


def test_existing_confirmed_audits_are_captured_once_without_new_submission(tmp_path, monkeypatch, advancing_clock):
    cfg = isolated_group(tmp_path)
    source_file(submission_path(cfg.shared_root, "batch"), operation="batch")
    discover_group(cfg)
    coverage = GroupCoverage(cfg.shared_root, "experiment")
    for debt in (coverage.directory / "maintenance/sources").glob("*.json"):
        debt.unlink()  # Baseline service had confirmed sources but no cleanup handoff.
    scratch = next((coverage.directory / "sources").glob("*/*/receipt.json")).parent
    owner = maintenance(cfg)
    capture = coverage.directory / "maintenance/source-capture.json"
    try:
        advance_until(owner, lambda: capture.exists() and read_json(capture).get("complete") is True)
        advance_until(owner, lambda: not (scratch / "qualification/task.digests").exists())
        advance_until(owner, lambda: not list((coverage.directory / "maintenance/sources").glob("*.json")))
    finally:
        close(owner)
    original = os.scandir

    def guarded(path):
        if not isinstance(path, int):
            assert Path(path) != coverage.directory / "sources", "completed capture scanned retained source index"
        return original(path)

    with monkeypatch.context() as guard:
        guard.setattr(os, "scandir", guarded)
        restored = maintenance(cfg)
        try:
            for _ in range(100):
                restored.advance()
            assert coverage.read_member(1).task_id == "task-a"
        finally:
            close(restored)


def test_missing_active_namespace_never_means_zero_subscribers(tmp_path, advancing_clock):
    cfg = isolated_group(tmp_path)
    journal, _ = journal_with_events(cfg)
    position = journal.snapshot()
    subscriber(cfg, position, cursor=0)
    active = shared_paths(cfg.shared_root)["group_control_active"]
    active.rename(active.with_name("unavailable-active"))
    owner = maintenance(cfg)
    try:
        for _ in range(100):
            owner.advance()
        assert journal.retention(position).floor == 0
        assert event_path(cfg, position, 1).exists()
    finally:
        close(owner)


def test_valid_large_active_operation_does_not_block_reclamation(tmp_path, advancing_clock):
    cfg = isolated_group(tmp_path)
    journal, _ = journal_with_events(cfg)
    position = journal.snapshot()
    value = subscriber(cfg, position, cursor=3)
    value["group_control"]["pending_machine_acknowledgements"] = {"g1": [f"pending-{i}" for i in range(10000)]}
    write_active_operation(cfg, "group_control", "cancel-a", value)
    owner = maintenance(cfg)
    try:
        advance_until(owner, lambda: journal.retention(position).deleted == 3)
        assert event_path(cfg, position, 4).exists()
    finally:
        close(owner)


def test_damaged_source_does_not_starve_later_cleanup_debt(tmp_path, advancing_clock):
    cfg = isolated_group(tmp_path, tail=2)
    source_file(submission_path(cfg.shared_root, "first"), operation="first")
    source_file(submission_path(cfg.shared_root, "second"), operation="second", tasks=["task-b"], sequences=[2])
    discover_group(cfg)
    coverage = GroupCoverage(cfg.shared_root, "experiment")
    owner = maintenance(cfg)
    try:
        advance_until(owner, lambda: owner._source_job is not None)
        damaged = owner._source_job.scratch
        (damaged / "receipt.json").write_text("{")
        good = next(
            path.parent for path in (coverage.directory / "sources").glob("*/*/receipt.json") if path.parent != damaged
        )
        advance_until(owner, lambda: not (good / "qualification/task.digests").exists())
        assert (damaged / "qualification/task.digests").exists()
        assert list((coverage.directory / "maintenance/sources").glob("*.json"))
    finally:
        close(owner)


def test_source_cleanup_rejects_symlink_ancestor(tmp_path, advancing_clock):
    cfg = isolated_group(tmp_path)
    source_file(submission_path(cfg.shared_root, "batch"), operation="batch")
    discover_group(cfg)
    directory = GroupCoverage(cfg.shared_root, "experiment").directory
    scratch = next((directory / "sources").glob("*/*/receipt.json")).parent
    qualification = scratch / "qualification"
    qualification.rename(scratch / "saved-qualification")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "task.digests").write_text("must survive")
    qualification.symlink_to(outside, target_is_directory=True)
    owner = maintenance(cfg)
    try:
        for _ in range(150):
            owner.advance()
        assert (outside / "task.digests").read_text() == "must survive"
        assert list((directory / "maintenance/sources").glob("*.json"))
    finally:
        close(owner)


def test_each_maintenance_advance_deletes_at_most_one_object(tmp_path, monkeypatch, advancing_clock):
    cfg = isolated_group(tmp_path, tail=3)
    source_file(
        submission_path(cfg.shared_root, "batch"), operation="batch", tasks=["a", "b", "c"], sequences=[1, 2, 3]
    )
    discover_group(cfg)
    journal, _ = journal_with_events(cfg, count=10)
    position = journal.snapshot()
    original_unlink = Path.unlink
    original_rmdir = Path.rmdir
    deleted = []

    def unlink(path, *args, **kwargs):
        deleted.append(path)
        return original_unlink(path, *args, **kwargs)

    def rmdir(path, *args, **kwargs):
        deleted.append(path)
        return original_rmdir(path, *args, **kwargs)

    owner = maintenance(cfg)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(Path, "unlink", unlink)
            patch.setattr(Path, "rmdir", rmdir)
            for _ in range(600):
                deleted.clear()
                owner.advance()
                assert len(deleted) <= 1, deleted
        assert journal.retention(position).deleted == 10
    finally:
        close(owner)


def test_missing_state_recovery_revokes_numerically_larger_retention_generation(tmp_path, monkeypatch, advancing_clock):
    cfg = isolated_group(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(rechecks.uuid, "uuid4", lambda: SimpleNamespace(int=1000))
        journal, tickets = journal_with_events(cfg)
    old = journal.snapshot()
    assert old.generation == 1000
    journal.reclaim(old, 1)
    directory = GroupCoverage(cfg.shared_root, "experiment").directory / "rechecks"
    (directory / "state.json").unlink()
    with monkeypatch.context() as patch:
        patch.setattr(rechecks.uuid, "uuid4", lambda: SimpleNamespace(int=100))
        fresh = journal.snapshot(initialize=True)
    assert fresh.generation == 100
    assert journal.retention(fresh).floor == 0
    journal.resolve(tickets[-1])
    current_ticket = journal.begin("current", "batch", 1, "retry", {})
    assert journal.read(journal.snapshot(), current_ticket.sequence)["state"] == "in_flight"
    owner = maintenance(cfg)
    try:
        advance_until(owner, lambda: not (directory / "1000").exists())
        assert journal.read(journal.snapshot(), current_ticket.sequence)["task_id"] == "current"
        journal.resolve(current_ticket)
        assert journal.read(journal.snapshot(), current_ticket.sequence)["state"] == "committed"
    finally:
        close(owner)


def test_unpublished_generation_is_reclaimable_and_invalidation_recreates_it(tmp_path, monkeypatch, advancing_clock):
    cfg = isolated_group(tmp_path)
    journal, _ = journal_with_events(cfg, count=1, unresolved=1)
    old = journal.snapshot()
    directory = GroupCoverage(cfg.shared_root, "experiment").directory / "rechecks"
    prepared = directory / str(old.generation + 1)
    original = rechecks.atomic_replace

    def fail_state(path, value):
        if path == directory / "state.json":
            raise OSError("state publication interrupted")
        original(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(rechecks, "atomic_replace", fail_state)
        with pytest.raises(OSError, match="publication interrupted"):
            journal.invalidate()
    assert prepared.is_dir()
    assert journal.snapshot() == old
    owner = maintenance(cfg)
    try:
        advance_until(owner, lambda: not prepared.exists())
        assert journal.read(old, 1)["state"] == "in_flight"
    finally:
        close(owner)
    fresh = journal.invalidate()
    assert fresh.generation == old.generation + 1
    assert prepared.is_dir()
    current = journal.begin("after-recovery", "batch", 1, "retry", {})
    journal.resolve(current)
    assert journal.read(journal.snapshot(), current.sequence)["state"] == "committed"
