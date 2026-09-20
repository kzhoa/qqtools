"""Consecutive membership coverage uses committed source provenance, not EOF."""

import pytest

from qqtools.plugins.qexp.runtime.group_discovery import coverage as coverage_module
from qqtools.plugins.qexp.runtime.group_discovery.coverage import GroupCoverage
from qqtools.plugins.qexp.runtime.group_namespace import group_directory
from tests.helpers.qexp_discovery import confirmed_source, isolated_group, set_tail

pytestmark = pytest.mark.integration


def publish_all(coverage, source):
    for _ in range(source.task_count + 20):
        step = coverage.publish(source, max_members=1)
        if step.state in {"complete", "blocked", "waiting"}:
            return step
    raise AssertionError("membership publication did not converge")


def advance_all(coverage, tail):
    for _ in range(tail + 3):
        status = coverage.advance(max_members=1)
        if status.is_complete:
            return status
    return status


def test_out_of_order_sources_never_certify_a_hole(tmp_path):
    cfg = isolated_group(tmp_path, tail=3)
    coverage = GroupCoverage(cfg.shared_root, "experiment")
    later = confirmed_source(cfg, coverage, "later", ["task-c"], [3])
    assert publish_all(coverage, later).state == "complete"
    assert coverage.advance().prefix == 0
    assert not coverage.status().is_complete
    earlier = confirmed_source(cfg, coverage, "earlier", ["task-a", "task-b"], [1, 2])
    assert publish_all(coverage, earlier).state == "complete"
    status = advance_all(coverage, 3)
    assert (status.prefix, status.tail, status.is_complete) == (3, 3, True)
    restored = GroupCoverage(cfg.shared_root, "experiment").status()
    assert (restored.prefix, restored.tail, restored.is_complete) == (3, 3, True)


def test_duplicate_publication_is_idempotent_and_conflict_blocks(tmp_path):
    cfg = isolated_group(tmp_path)
    coverage = GroupCoverage(cfg.shared_root, "experiment")
    first = confirmed_source(cfg, coverage, "first", ["task-a"], [1])
    assert publish_all(coverage, first).state == "complete"
    assert publish_all(coverage, first).state == "complete"
    assert coverage.advance().is_complete
    conflicting = confirmed_source(cfg, coverage, "conflicting", ["task-b"], [1])
    assert publish_all(coverage, conflicting).state == "blocked"
    assert not coverage.status().is_complete
    assert not GroupCoverage(cfg.shared_root, "experiment").status().is_complete


def test_crash_after_member_before_cursor_replays_exactly_once(tmp_path, monkeypatch):
    cfg = isolated_group(tmp_path, tail=2)
    coverage = GroupCoverage(cfg.shared_root, "experiment")
    source = confirmed_source(cfg, coverage, "batch", ["task-a", "task-b"], [1, 2])
    original = coverage_module.atomic_replace
    has_cut = False

    def cut(path, record):
        nonlocal has_cut
        original(path, record)
        if path.parent.name == "members" and not has_cut:
            has_cut = True
            raise OSError("member durable before cursor")

    with monkeypatch.context() as crashing:
        crashing.setattr(coverage_module, "atomic_replace", cut)
        with pytest.raises(OSError, match="before cursor"):
            coverage.publish(source, max_members=1)
    assert has_cut
    resumed = GroupCoverage(cfg.shared_root, "experiment")
    assert publish_all(resumed, source).state == "complete"
    assert advance_all(resumed, 2).is_complete
    assert len(list((resumed.directory / "members").glob("*.json"))) == 2


def test_pending_group_finalization_never_reports_complete(tmp_path):
    cfg = isolated_group(tmp_path)
    coverage = GroupCoverage(cfg.shared_root, "experiment")
    source = confirmed_source(cfg, coverage, "batch", ["task-a"], [1])
    assert publish_all(coverage, source).state == "complete"
    assert coverage.advance().is_complete
    set_tail(cfg, 1, pending={"operation_id": "new-batch", "membership_sequences": [2]})
    assert not coverage.status().is_complete
    set_tail(cfg, 2)
    assert not coverage.status().is_complete
    assert coverage.status().prefix == 1


def test_replaced_source_cannot_publish_members(tmp_path):
    cfg = isolated_group(tmp_path)
    coverage = GroupCoverage(cfg.shared_root, "experiment")
    source = confirmed_source(cfg, coverage, "batch", ["task-a"], [1])
    replacement = source.source.with_suffix(".new")
    replacement.write_bytes(source.source.read_bytes())
    replacement.replace(source.source)
    with pytest.raises((ValueError, RuntimeError)):
        coverage.publish(source)
    assert coverage.status().prefix == 0


def test_new_directory_identity_cannot_reuse_old_coverage(tmp_path):
    cfg = isolated_group(tmp_path)
    coverage = GroupCoverage(cfg.shared_root, "experiment")
    source = confirmed_source(cfg, coverage, "batch", ["task-a"], [1])
    publish_all(coverage, source)
    assert coverage.advance().is_complete
    directory = group_directory(cfg.shared_root)
    directory.rename(directory.with_name("retired-authority"))
    directory.mkdir()
    with pytest.raises((ValueError, RuntimeError, OSError)):
        coverage.status()


def test_large_task_identifier_stays_a_reference(tmp_path):
    cfg = isolated_group(tmp_path)
    coverage = GroupCoverage(cfg.shared_root, "experiment")
    source = confirmed_source(cfg, coverage, "large", ["x" * 160005], [1])
    assert publish_all(coverage, source).state == "complete"
    assert coverage.advance().is_complete
    slot = next((coverage.directory / "members").glob("*.json"))
    assert slot.stat().st_size < 16384


def test_sequence_beyond_current_tail_waits_for_group_finalization(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    coverage = GroupCoverage(cfg.shared_root, "experiment")
    source = confirmed_source(cfg, coverage, "batch", ["task-a"], [1])
    assert coverage.publish(source).state == "waiting"
    set_tail(cfg, 1)
    assert publish_all(coverage, source).state == "complete"
    assert coverage.advance().is_complete


def test_same_task_under_a_different_submission_is_a_conflict(tmp_path):
    cfg = isolated_group(tmp_path)
    coverage = GroupCoverage(cfg.shared_root, "experiment")
    first = confirmed_source(cfg, coverage, "first", ["task-a"], [1])
    second = confirmed_source(cfg, coverage, "second", ["task-a"], [1])
    assert publish_all(coverage, first).state == "complete"
    assert publish_all(coverage, second).state == "blocked"
    assert not GroupCoverage(cfg.shared_root, "experiment").status().is_complete


def test_empty_group_coverage_is_identity_bound(tmp_path):
    first = isolated_group(tmp_path / "first", tail=0)
    second = isolated_group(tmp_path / "second", tail=0)
    first_coverage = GroupCoverage(first.shared_root, "experiment")
    second_coverage = GroupCoverage(second.shared_root, "experiment")
    assert first_coverage.status().is_complete
    assert second_coverage.status().is_complete
    assert first_coverage.directory != second_coverage.directory
    set_tail(second, 1)
    assert first_coverage.status().is_complete
    assert not second_coverage.status().is_complete
