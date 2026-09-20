"""Certified member lookup preserves source provenance without history scans."""

import pytest

from qqtools.plugins.qexp.runtime.group_discovery.coverage import GroupCoverage
from qqtools.plugins.qexp.runtime.locks import exclusive
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from tests.helpers.qexp_discovery import confirmed_source, isolated_group, set_tail

pytestmark = pytest.mark.integration


def prepared(tmp_path, tasks=None):
    tasks = ["first", "second"] if tasks is None else tasks
    cfg = isolated_group(tmp_path, tail=len(tasks))
    coverage = GroupCoverage(cfg.shared_root, "experiment")
    source = confirmed_source(cfg, coverage, "batch", tasks, list(range(1, len(tasks) + 1)))
    for _ in range(len(tasks) + 1):
        coverage.publish(source)
    assert coverage.advance().is_complete
    return cfg, coverage, source


def test_lookup_directly_returns_exact_member_after_restart(tmp_path, monkeypatch):
    import os

    cfg, _, _ = prepared(tmp_path)

    def no_scan(*args, **kwargs):
        raise AssertionError("member lookup enumerated history")

    with monkeypatch.context() as patch:
        patch.setattr(os, "scandir", no_scan)
        coverage = GroupCoverage(cfg.shared_root, "experiment")
        member = coverage.read_member(2)
    assert (member.sequence, member.task_id, member.operation_id) == (2, "second", "batch")


def test_fixed_prefix_remains_readable_during_later_submission(tmp_path):
    cfg, coverage, _ = prepared(tmp_path)
    set_tail(cfg, 2, pending={"operation_id": "later"})
    assert coverage.read_member(1).task_id == "first"
    with pytest.raises((ValueError, RuntimeError)):
        coverage.read_member(3)


@pytest.mark.parametrize("sequence", [True, 0, -1, "1", 1.0])
def test_member_sequence_requires_positive_exact_integer(tmp_path, sequence):
    _, coverage, _ = prepared(tmp_path)
    with pytest.raises((TypeError, ValueError)):
        coverage.read_member(sequence)


@pytest.mark.parametrize("artifact", ["source", "spool", "task_references", "sequence_references"])
def test_same_bytes_replacement_invalidates_member_provenance(tmp_path, artifact):
    _, coverage, source = prepared(tmp_path)
    path = getattr(source, artifact)
    replacement = path.with_suffix(".replacement")
    replacement.write_bytes(path.read_bytes())
    replacement.replace(path)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        coverage.read_member(1)


def test_slot_cannot_redirect_to_another_valid_member(tmp_path):
    _, coverage, _ = prepared(tmp_path)
    first_path = coverage.directory / "members/1.json"
    first = read_json(first_path)
    second = read_json(coverage.directory / "members/2.json")
    first["task_ref"] = second["task_ref"]
    atomic_replace(first_path, first)
    with pytest.raises((ValueError, RuntimeError)):
        coverage.read_member(1)


def test_member_requires_whole_source_confirmation(tmp_path):
    _, coverage, source = prepared(tmp_path)
    (source.spool.parent.parent / "receipt.json").unlink()
    with pytest.raises((ValueError, RuntimeError, OSError)):
        coverage.read_member(1)


def test_single_large_identifier_is_not_truncated(tmp_path):
    task_id = "member_" + "x" * 160005
    _, coverage, _ = prepared(tmp_path, [task_id])
    assert coverage.read_member(1).task_id == task_id


def test_busy_member_publication_does_not_block_consumer(tmp_path):
    _, coverage, _ = prepared(tmp_path)
    with exclusive(coverage.directory / ".lock"):
        with pytest.raises(BlockingIOError):
            coverage.read_member(1)


def test_conflicted_coverage_cannot_supply_authoritative_candidates(tmp_path):
    cfg, coverage, _ = prepared(tmp_path)
    conflicting = confirmed_source(cfg, coverage, "conflict", ["other"], [1])
    assert coverage.publish(conflicting).state == "blocked"
    with pytest.raises((ValueError, RuntimeError)):
        coverage.read_member(1)


def test_receipt_changed_during_read_cannot_certify_member(tmp_path, monkeypatch):
    import os

    _, coverage, source = prepared(tmp_path)
    path = source.spool.parent.parent / "receipt.json"
    info = path.stat()
    original = os.read
    changed = False

    def changing_read(descriptor, size):
        nonlocal changed
        data = original(descriptor, size)
        if not changed and os.fstat(descriptor).st_ino == info.st_ino:
            changed = True
            os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000))
        return data

    monkeypatch.setattr(os, "read", changing_read)
    with pytest.raises(ValueError, match="revision changed"):
        coverage.read_member(1)
    assert changed


def test_receipt_replaced_during_artifact_read_invalidates_lookup(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runtime.group_discovery import coverage as module

    _, coverage, source = prepared(tmp_path)
    receipt = source.spool.parent.parent / "receipt.json"
    original = module._decode_task_id

    def replacing_receipt(*args, **kwargs):
        replacement = receipt.with_suffix(".replacement")
        replacement.write_bytes(receipt.read_bytes())
        replacement.replace(receipt)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_decode_task_id", replacing_receipt)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        coverage.read_member(1)
