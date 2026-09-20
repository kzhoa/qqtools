"""Source qualification joins complete positional fields and uniqueness audits."""

from __future__ import annotations

import base64
import json
import os
import struct

import pytest

from qqtools.plugins.qexp.runtime.group_discovery.driver import ProjectionDriver
from qqtools.plugins.qexp.runtime.group_discovery.slice_io import SliceIO
from qqtools.plugins.qexp.runtime.group_discovery.source_qualification import SourceQualification

pytestmark = pytest.mark.integration


def projected(tmp_path, tasks, sequences, *, reverse=False, state="committed", group="exp", parse_bytes=31):
    arrays = {
        "resolved_context": {"task_ids": tasks},
        "commit_plan": {"group_membership_sequences": sequences},
    }
    if reverse:
        arrays = dict(reversed(list(arrays.items())))
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "meta": {"schema_version": 6},
                "submission": {"operation_id": "op-1", "target_group": group, "state": state, **arrays},
            }
        )
    )
    driver = ProjectionDriver(source, tmp_path / "projection", "op-1", "exp")
    for _ in range(2000):
        driver.advance(SliceIO(), max_processed_bytes=parse_bytes)
        if driver.is_complete:
            return driver
    close(driver)
    raise AssertionError("test projection did not complete")


def close(driver):
    driver.request_close()
    for _ in range(200):
        if driver.is_closed:
            return
        driver.advance(SliceIO())
    raise AssertionError("source qualification leaked ownership")


def qualify(qualification, *, io_bytes=65536, operations=16):
    for _ in range(100000):
        io = SliceIO(max_io_bytes=io_bytes, max_operations=operations)
        qualification.advance(io, max_events=3)
        assert io.io_bytes_used <= io_bytes
        assert io.operations_used <= operations
        if qualification.is_complete:
            return qualification.result
        assert qualification.result is None
    raise AssertionError("source qualification did not converge")


@pytest.mark.parametrize("reverse", [False, True])
def test_qualified_references_pair_exact_members_in_either_field_order(tmp_path, reverse):
    driver = projected(tmp_path, ["task-a", "task-b"], [7, 8], reverse=reverse)
    qualification = SourceQualification(driver, tmp_path / "qualified")
    try:
        result = qualify(qualification)
        assert result.status == "qualified"
        assert result.projection.operation_id == "op-1"
        spool = result.projection.spool.read_bytes()
        paired = []
        for path in (result.task_references, result.sequence_references):
            fields = []
            rows = list(struct.iter_unpack(">QQQQ", path.read_bytes()))
            assert [row[0] for row in rows] == [0, 1]
            for _, start, end, size in rows:
                events = [json.loads(line) for line in spool[start:end].splitlines()]
                assert events[-1]["type"] == "end"
                assert events[-1]["decoded_size"] == size
                value = b"".join(base64.b64decode(event["data_b64"]) for event in events[:-1])
                assert len(value) == size
                fields.append(value)
            paired.append(fields)
        assert paired == [[b"task-a", b"task-b"], [b"7", b"8"]]
        assert not driver.is_closed  # The qualification owns no caller lifecycle.
    finally:
        close(qualification)
        close(driver)


@pytest.mark.parametrize("tasks,sequences", [(["task-a", "task-a"], [1, 2]), (["task-a", "task-b"], [1, 1])])
def test_duplicate_identity_or_sequence_never_qualifies(tmp_path, tasks, sequences):
    driver = projected(tmp_path, tasks, sequences)
    qualification = SourceQualification(driver, tmp_path / "qualified")
    try:
        result = qualify(qualification)
        assert result.status == "ambiguous"
        assert result.task_references is None
        assert result.sequence_references is None
    finally:
        close(qualification)
        close(driver)


@pytest.mark.parametrize("state,group", [("aborted", "exp"), ("committed", "other")])
def test_uncommitted_or_other_group_source_has_no_membership_result(tmp_path, state, group):
    driver = projected(tmp_path, ["task-a"], [1], state=state, group=group)
    qualification = SourceQualification(driver, tmp_path / "qualified")
    try:
        result = qualify(qualification)
        assert result.status == "irrelevant"
        assert result.task_references is None
        assert result.sequence_references is None
    finally:
        close(qualification)
        close(driver)


def test_one_byte_budget_keeps_all_outputs_provisional_until_complete(tmp_path):
    driver = projected(tmp_path, ["task-a"], [1])
    qualification = SourceQualification(driver, tmp_path / "qualified")
    try:
        result = qualify(qualification, io_bytes=1, operations=1)
        assert result.status == "qualified"
        assert result.task_references.stat().st_size == 32
        assert result.sequence_references.stat().st_size == 32
    finally:
        close(qualification)
        close(driver)


def test_source_replaced_after_projection_cannot_publish_qualification(tmp_path):
    driver = projected(tmp_path, ["task-a"], [1])
    manifest = driver.completed_projection
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(manifest.source.read_bytes())
    replacement.replace(manifest.source)
    qualification = SourceQualification(driver, tmp_path / "qualified")
    try:
        with pytest.raises(ValueError):
            qualify(qualification)
        assert qualification.result is None
    finally:
        close(qualification)
        close(driver)


def test_closing_projection_revokes_inflight_qualification(tmp_path):
    driver = projected(tmp_path, ["task-a"], [1])
    qualification = SourceQualification(driver, tmp_path / "qualified")
    qualification.advance(SliceIO(max_operations=1))
    close(driver)
    try:
        with pytest.raises(ValueError):
            qualify(qualification)
        assert qualification.result is None
    finally:
        close(qualification)


def test_empty_committed_source_qualifies_zero_pairs(tmp_path):
    driver = projected(tmp_path, [], [])
    qualification = SourceQualification(driver, tmp_path / "qualified")
    try:
        result = qualify(qualification)
        assert result.status == "qualified"
        assert result.task_references.read_bytes() == b""
        assert result.sequence_references.read_bytes() == b""
    finally:
        close(qualification)
        close(driver)


def test_incomplete_projection_cannot_start_qualification(tmp_path):
    driver = ProjectionDriver(tmp_path / "source.json", tmp_path / "projection", "op-1", "exp")
    try:
        assert driver.completed_projection is None
        with pytest.raises(ValueError):
            SourceQualification(driver, tmp_path / "qualified")
        assert not (tmp_path / "qualified").exists()
    finally:
        close(driver)


def test_large_task_identity_is_referenced_without_an_identifier_size_cap(tmp_path):
    task_id = "task-" + "a" * 160000
    driver = projected(tmp_path, [task_id], [1], parse_bytes=65536)
    qualification = SourceQualification(driver, tmp_path / "qualified")
    try:
        result = qualify(qualification)
        assert result.status == "qualified"
        ordinal, start, end, size = struct.unpack(">QQQQ", result.task_references.read_bytes())
        assert ordinal == 0 and size == len(task_id)
        with result.projection.spool.open("rb") as spool:
            spool.seek(start)
            events = [json.loads(line) for line in spool.read(end - start).splitlines()]
        assert b"".join(base64.b64decode(event["data_b64"]) for event in events[:-1]) == task_id.encode()
    finally:
        close(qualification)
        close(driver)


def test_existing_qualification_scratch_is_not_reused_or_overwritten(tmp_path):
    driver = projected(tmp_path, ["task-a"], [1])
    scratch = tmp_path / "qualified"
    scratch.mkdir()
    sentinel = scratch / "task.refs"
    sentinel.write_bytes(b"retained unfinished work")
    qualification = SourceQualification(driver, scratch)
    try:
        with pytest.raises(FileExistsError):
            qualify(qualification)
        assert sentinel.read_bytes() == b"retained unfinished work"
        assert qualification.result is None
    finally:
        close(qualification)
        close(driver)


def test_qualification_close_error_does_not_close_another_owners_reused_fd(tmp_path, monkeypatch):
    driver = projected(tmp_path, ["task-a"], [1])
    qualification = SourceQualification(driver, tmp_path / "qualified")
    opened = []
    replacements = []
    original_open = SliceIO.open
    original_close = SliceIO.close

    def tracked_open(self, path, flags, mode=0o600):
        result = original_open(self, path, flags, mode)
        if type(result) is int:
            opened.append(result)
        return result

    try:
        with monkeypatch.context() as patch:
            patch.setattr(SliceIO, "open", tracked_open)
            for _ in range(100):
                qualification.advance(SliceIO(max_operations=1))
                if opened:
                    break
            assert len(opened) == 1
            owned_fd = opened[0]

            def failing_close(self, descriptor):
                if descriptor == owned_fd and not replacements:

                    def fail():
                        os.close(descriptor)
                        replacement = os.open(tmp_path / "another-owner", os.O_CREAT | os.O_RDWR, 0o600)
                        replacements.append(replacement)
                        assert replacement == descriptor
                        raise OSError("close failed after release")

                    return self._metadata(fail)
                return original_close(self, descriptor)

            patch.setattr(SliceIO, "close", failing_close)
            qualification.request_close()
            with pytest.raises(OSError, match="close failed after release"):
                for _ in range(100):
                    qualification.advance(SliceIO(max_operations=1))
                    if qualification.is_closed:
                        break
            assert qualification.result is None
            assert qualification.is_closed
            os.fstat(replacements[0])
    finally:
        close(qualification)
        close(driver)
        for descriptor in replacements:
            try:
                os.close(descriptor)
            except OSError:
                pass


@pytest.mark.parametrize("name", ["task.digests", "task.refs", "sequence.digests", "sequence.refs"])
@pytest.mark.parametrize("failure", ["close", "fstat"])
def test_output_is_owned_before_suspendible_validation(tmp_path, monkeypatch, name, failure):
    driver = projected(tmp_path, ["task-a"], [1])
    qualification = SourceQualification(driver, tmp_path / "qualified")
    tracked_fds = []
    original_open = SliceIO.open
    original_stat = SliceIO.fstat

    def tracked_open(self, path, flags, mode=0o600):
        result = original_open(self, path, flags, mode)
        if path.name == name and type(result) is int:
            tracked_fds.append(result)
        return result

    monkeypatch.setattr(SliceIO, "open", tracked_open)
    try:
        for _ in range(100):
            qualification.advance(SliceIO(max_operations=1))
            if tracked_fds:
                break
        assert len(tracked_fds) == 1
        descriptor = tracked_fds[0]
        if failure == "close":
            close(qualification)
        else:

            def failed_stat(self, fd):
                if fd == descriptor:

                    def fail():
                        raise OSError("output fstat failed")

                    return self._metadata(fail)
                return original_stat(self, fd)

            monkeypatch.setattr(SliceIO, "fstat", failed_stat)
            with pytest.raises(OSError, match="output fstat failed"):
                qualify(qualification)
        assert qualification.result is None
        with pytest.raises(OSError):
            os.fstat(descriptor)
    finally:
        close(qualification)
        close(driver)
        for descriptor in tracked_fds:
            try:
                os.close(descriptor)
            except OSError:
                pass


@pytest.mark.parametrize("replace_spool", [False, True])
def test_bound_source_or_spool_replacement_cannot_publish_references(tmp_path, monkeypatch, replace_spool):
    driver = projected(tmp_path, ["task-a", "task-b"], [1, 2])
    qualification = SourceQualification(driver, tmp_path / "qualified")
    manifest = driver.completed_projection
    target = manifest.spool if replace_spool else manifest.source
    replacement = tmp_path / "replacement"
    replacement.write_bytes(target.read_bytes())
    original_read = SliceIO.read
    replaced = False

    def replace_after_bound_read(self, descriptor, size):
        nonlocal replaced
        result = original_read(self, descriptor, size)
        if isinstance(result, bytes) and result and not replaced:
            replacement.replace(target)
            replaced = True
        return result

    monkeypatch.setattr(SliceIO, "read", replace_after_bound_read)
    try:
        with pytest.raises(ValueError):
            qualify(qualification)
        assert replaced
        assert qualification.result is None
        assert not qualification.is_complete
    finally:
        close(qualification)
        close(driver)
