"""Certified member lookup preserves source provenance without history scans."""

import ast
import sys
from pathlib import Path

import pytest

from qqtools.plugins.qexp.runtime.group_discovery.coverage import GroupCoverage
from qqtools.plugins.qexp.runtime.locks import exclusive
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from tests.helpers.qexp_discovery import (
    close_source,
    confirmed_source,
    finish_source,
    isolated_group,
    set_tail,
    source_file,
)

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
    from qqtools.plugins.qexp.runtime.group_discovery import member_reader as module

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


def test_member_reader_dependency_boundary_is_narrow():
    from qqtools.plugins.qexp.runtime.group_discovery import member_reader

    tree = ast.parse(Path(member_reader.__file__).read_text(encoding="utf-8"))
    imports = {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imports.update(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
    assert not any(
        forbidden in imported for imported in imports for forbidden in ("coverage", "recovery", "scheduler", "commands")
    )


def test_publication_reader_closes_an_earlier_reference_when_a_later_open_fails(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runtime.group_discovery import member_reader

    cfg = isolated_group(tmp_path)
    coverage = GroupCoverage(cfg.shared_root, "experiment")
    source = confirmed_source(cfg, coverage, "batch", ["task-a"], [1])
    real_open = member_reader.os.open
    real_close = member_reader.os.close
    closed = []

    def fail_sequence_open(path, flags, *args, **kwargs):
        if path == source.sequence_references:
            raise OSError("sequence reference open failed")
        return real_open(path, flags, *args, **kwargs)

    def recording_close(descriptor):
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(member_reader.os, "open", fail_sequence_open)
    monkeypatch.setattr(member_reader.os, "close", recording_close)
    with pytest.raises(OSError, match="sequence reference open failed"):
        with member_reader.open_publication_reader(
            source,
            group="experiment",
            coverage_directory=coverage.directory,
        ):
            pass
    assert len(closed) == 1


def test_publication_reader_attempts_every_close_and_preserves_body_error(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runtime.group_discovery import member_reader

    cfg = isolated_group(tmp_path)
    coverage = GroupCoverage(cfg.shared_root, "experiment")
    source = confirmed_source(cfg, coverage, "batch", ["task-a"], [1])
    real_close = member_reader.os.close
    closed = []

    def failing_close(descriptor):
        closed.append(descriptor)
        real_close(descriptor)
        raise OSError(f"close failed for {descriptor}")

    monkeypatch.setattr(member_reader.os, "close", failing_close)
    with pytest.raises(RuntimeError, match="primary read failure") as caught:
        with member_reader.open_publication_reader(
            source,
            group="experiment",
            coverage_directory=coverage.directory,
        ):
            raise RuntimeError("primary read failure")
    assert len(closed) == 2
    assert getattr(caught.value, "__notes__", ())
    assert all("close failed" in note for note in caught.value.__notes__)


def test_publication_reader_reports_multiple_close_failures(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runtime.group_discovery import member_reader

    cfg = isolated_group(tmp_path)
    coverage = GroupCoverage(cfg.shared_root, "experiment")
    source = confirmed_source(cfg, coverage, "batch", ["task-a"], [1])
    real_close = member_reader.os.close
    closed = []

    def failing_close(descriptor):
        closed.append(descriptor)
        real_close(descriptor)
        raise OSError(f"close failed for {descriptor}")

    monkeypatch.setattr(member_reader.os, "close", failing_close)
    with pytest.raises(ExceptionGroup, match="publication reader cleanup failed") as caught:
        with member_reader.open_publication_reader(
            source,
            group="experiment",
            coverage_directory=coverage.directory,
        ):
            pass
    assert len(closed) == 2
    assert len(caught.value.exceptions) == 2


def test_publication_reader_aggregates_base_cleanup_failures():
    from qqtools.plugins.qexp.runtime.group_discovery import member_reader

    failures = [KeyboardInterrupt("first close failed"), SystemExit("second close failed")]
    with pytest.raises(BaseExceptionGroup, match="publication reader cleanup failed") as caught:
        member_reader._finish_cleanup(failures, None, "publication reader cleanup failed")
    assert [type(error) for error in caught.value.exceptions] == [KeyboardInterrupt, SystemExit]


def test_direct_lookup_closes_every_open_resource_after_partial_open(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runtime.group_discovery import member_reader

    cfg, coverage, source = prepared(tmp_path)
    slot = read_json(coverage.directory / "members/1.json")
    receipt = source.spool.parent.parent / "receipt.json"
    real_open = member_reader.os.open
    real_close = member_reader.os.close
    opened = {}
    closed = []

    def fail_sequence_open(path, flags, *args, **kwargs):
        if path == source.sequence_references:
            raise OSError("sequence reference open failed")
        descriptor = real_open(path, flags, *args, **kwargs)
        opened[descriptor] = path
        return descriptor

    def recording_close(descriptor):
        closed.append(opened[descriptor])
        real_close(descriptor)

    monkeypatch.setattr(member_reader.os, "open", fail_sequence_open)
    monkeypatch.setattr(member_reader.os, "close", recording_close)
    with pytest.raises(OSError, match="sequence reference open failed"):
        member_reader.read_published_member(
            cfg.shared_root,
            coverage_directory=coverage.directory,
            group="experiment",
            slot=slot,
        )
    assert closed == [receipt, source.source, source.task_references]


def test_direct_lookup_attempts_every_close_and_preserves_read_error(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runtime.group_discovery import member_reader

    cfg, coverage, _ = prepared(tmp_path)
    slot = read_json(coverage.directory / "members/1.json")
    real_close = member_reader.os.close
    closed = []

    def failing_close(descriptor):
        closed.append(descriptor)
        real_close(descriptor)
        raise OSError(f"close failed for {descriptor}")

    def fail_read(*args, **kwargs):
        raise RuntimeError("primary member read failure")

    monkeypatch.setattr(member_reader.os, "close", failing_close)
    monkeypatch.setattr(member_reader, "_decode_task_id", fail_read)
    with pytest.raises(RuntimeError, match="primary member read failure") as caught:
        member_reader.read_published_member(
            cfg.shared_root,
            coverage_directory=coverage.directory,
            group="experiment",
            slot=slot,
        )
    assert len(closed) == 4
    assert getattr(caught.value, "__notes__", ())
    assert all("close failed" in note for note in caught.value.__notes__)


def test_publication_reader_rejects_huge_sequence_before_integer_construction(tmp_path):
    from qqtools.plugins.qexp.runtime.group_discovery.member_reader import SequenceBeyondLimit, open_publication_reader
    from qqtools.plugins.qexp.runtime.group_discovery.recovery import RecoverableSource
    from qqtools.plugins.qexp.runtime.paths import submission_path

    cfg = isolated_group(tmp_path, tail=0)
    coverage = GroupCoverage(cfg.shared_root, "experiment")
    digit_limit = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(0)
        huge_sequence = int("9" * 10_000)
        path = source_file(
            submission_path(cfg.shared_root, "batch"),
            operation="batch",
            tasks=["task-a"],
            sequences=[huge_sequence],
        )
    finally:
        sys.set_int_max_str_digits(digit_limit)

    session = RecoverableSource(path, coverage.source_scratch("batch"), "batch", "experiment")
    try:
        source = finish_source(session)
    finally:
        close_source(session)

    with open_publication_reader(source, group="experiment", coverage_directory=coverage.directory) as reader:
        with pytest.raises(SequenceBeyondLimit):
            reader.read_row(0, sequence_limit=0)
