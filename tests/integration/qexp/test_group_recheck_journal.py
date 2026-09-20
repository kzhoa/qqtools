"""Ordered recheck publication never makes an unpublished effect acknowledgeable."""

import pytest

from qqtools.plugins.qexp.runtime.group_discovery import rechecks
from qqtools.plugins.qexp.runtime.group_discovery.coverage import GroupCoverage
from qqtools.plugins.qexp.runtime.group_discovery.rechecks import GroupRechecks
from tests.helpers.qexp_discovery import isolated_group

pytestmark = pytest.mark.integration


def begin(journal, task="task-a"):
    return journal.begin(task, "submission-a", 1, "retry", {"task_revision_before": 3})


def test_journal_is_dormant_until_a_consumer_subscribes(tmp_path):
    cfg = isolated_group(tmp_path)
    journal = GroupRechecks(cfg.shared_root, "experiment")
    assert journal.snapshot() is None
    assert begin(journal) is None
    assert journal.snapshot() is None
    position = journal.snapshot(initialize=True)
    assert position.generation > 0
    assert position.tail == 0


def test_restart_preserves_inflight_and_exact_resolution(tmp_path):
    cfg = isolated_group(tmp_path)
    journal = GroupRechecks(cfg.shared_root, "experiment")
    journal.snapshot(initialize=True)
    ticket = begin(journal)
    restored = GroupRechecks(cfg.shared_root, "experiment")
    position = restored.snapshot()
    event = restored.read(position, ticket.sequence)
    assert event["state"] == "in_flight"
    assert (event["task_id"], event["submission_operation_id"], event["owner"]) == ("task-a", "submission-a", "retry")
    restored.resolve(ticket)
    restored.resolve(ticket)
    assert restored.read(restored.snapshot(), ticket.sequence)["state"] == "committed"
    with pytest.raises(ValueError):
        restored.resolve(ticket, outcome="aborted")


def test_aborted_attempt_cannot_cover_a_later_retry(tmp_path):
    cfg = isolated_group(tmp_path)
    journal = GroupRechecks(cfg.shared_root, "experiment")
    journal.snapshot(initialize=True)
    abandoned = begin(journal)
    journal.resolve(abandoned, outcome="aborted")
    fresh = begin(journal)
    assert fresh.sequence > abandoned.sequence
    assert journal.read(journal.snapshot(), fresh.sequence)["state"] == "in_flight"


def test_generation_invalidation_prevents_old_tail_completion(tmp_path):
    cfg = isolated_group(tmp_path)
    journal = GroupRechecks(cfg.shared_root, "experiment")
    journal.snapshot(initialize=True)
    old_ticket = begin(journal)
    old_position = journal.snapshot()
    fresh = journal.invalidate()
    assert fresh.generation == old_position.generation + 1
    assert fresh.tail == 0
    journal.resolve(old_ticket)
    with pytest.raises((ValueError, RuntimeError)):
        journal.read(old_position, old_ticket.sequence)
    new_ticket = begin(journal)
    assert new_ticket.generation == fresh.generation
    assert journal.read(journal.snapshot(), new_ticket.sequence)["state"] == "in_flight"


@pytest.mark.parametrize("cut", ["event", "tail"])
def test_publication_crashes_never_return_an_unreachable_ticket(tmp_path, monkeypatch, cut):
    cfg = isolated_group(tmp_path)
    journal = GroupRechecks(cfg.shared_root, "experiment")
    journal.snapshot(initialize=True)
    original = rechecks.atomic_replace

    def crash(path, value):
        original(path, value)
        if (cut == "event" and value.get("state") == "in_flight") or (
            cut == "tail" and path.name == "state.json" and value.get("tail") == 1
        ):
            raise OSError("simulated process cut")

    with monkeypatch.context() as patch:
        patch.setattr(rechecks, "atomic_replace", crash)
        with pytest.raises(OSError, match="process cut"):
            begin(journal)
    restored = GroupRechecks(cfg.shared_root, "experiment")
    position = restored.snapshot()
    if position.tail:
        assert restored.read(position, 1)["state"] == "in_flight"
    ticket = begin(restored, "task-b")
    assert restored.read(restored.snapshot(), ticket.sequence)["task_id"] == "task-b"


def test_no_directory_enumeration_for_new_or_resumed_events(tmp_path, monkeypatch):
    import os

    cfg = isolated_group(tmp_path)
    journal = GroupRechecks(cfg.shared_root, "experiment")
    journal.snapshot(initialize=True)

    def no_scan(*args, **kwargs):
        raise AssertionError("recheck publication scanned history")

    with monkeypatch.context() as patch:
        patch.setattr(os, "scandir", no_scan)
        ticket = begin(journal)
        journal.resolve(ticket)
        assert journal.read(journal.snapshot(), ticket.sequence)["state"] == "committed"


def test_missing_state_is_not_dormant_and_reinitialization_revokes_old_proof(tmp_path):
    cfg = isolated_group(tmp_path)
    journal = GroupRechecks(cfg.shared_root, "experiment")
    old = journal.snapshot(initialize=True)
    state = GroupCoverage(cfg.shared_root, "experiment").directory / "rechecks/state.json"
    state.unlink()
    with pytest.raises((ValueError, RuntimeError)):
        journal.snapshot()
    with pytest.raises((ValueError, RuntimeError)):
        begin(journal)
    new = journal.snapshot(initialize=True)
    assert new.generation != old.generation
    assert new.tail == 0
