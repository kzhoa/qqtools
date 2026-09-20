"""Runtime uniqueness audit uses the caller's IO budget and keeps ambiguity."""

from __future__ import annotations

import hashlib
import os

import pytest

from qqtools.plugins.qexp.runtime.group_discovery.digest_audit import DigestAuditDriver
from qqtools.plugins.qexp.runtime.group_discovery.slice_io import SliceIO

pytestmark = pytest.mark.integration


def records(count):
    return [hashlib.sha256(str(index).encode()).digest() for index in range(count)]


def close(driver):
    driver.request_close()
    for _ in range(100):
        driver.advance(SliceIO(max_io_bytes=4096, max_operations=1))
        if driver.is_closed:
            return
    raise AssertionError("audit did not release descriptors")


def finish(driver, *, io_bytes=4096, operations=8):
    for _ in range(200000):
        io = SliceIO(max_io_bytes=io_bytes, max_operations=operations)
        driver.advance(io, max_records=7)
        assert io.io_bytes_used <= io_bytes
        assert io.operations_used <= operations
        if driver.is_complete:
            return
    raise AssertionError("audit did not converge")


@pytest.mark.parametrize("count", [0, 1, 64, 65, 129, 257])
def test_external_audit_produces_exact_sorted_digest_stream(tmp_path, count):
    values = records(count)
    source = tmp_path / "digests.bin"
    source.write_bytes(b"".join(reversed(values)))
    driver = DigestAuditDriver(source, tmp_path / "audit", expected_count=count)
    try:
        finish(driver)
        assert not driver.has_repeated_digest
        assert driver.result_path.read_bytes() == b"".join(sorted(values))
    finally:
        close(driver)


def test_equal_digests_in_different_runs_are_ambiguous(tmp_path):
    values = records(129)
    values[-1] = values[0]
    source = tmp_path / "digests.bin"
    source.write_bytes(b"".join(values))
    driver = DigestAuditDriver(source, tmp_path / "audit", expected_count=len(values))
    try:
        finish(driver)
        assert driver.has_repeated_digest
        assert driver.result_path is None
    finally:
        close(driver)


def test_one_byte_one_operation_slices_preserve_partial_records(tmp_path):
    values = records(9)
    source = tmp_path / "digests.bin"
    source.write_bytes(b"".join(values))
    driver = DigestAuditDriver(source, tmp_path / "audit", expected_count=len(values), run_records=2)
    try:
        finish(driver, io_bytes=1, operations=1)
        assert driver.result_path.read_bytes() == b"".join(sorted(values))
    finally:
        close(driver)


@pytest.mark.parametrize("payload,count", [(b"x", 1), (b"x" * 32, 2)])
def test_truncated_or_wrong_count_source_never_qualifies(tmp_path, payload, count):
    source = tmp_path / "digests.bin"
    source.write_bytes(payload)
    driver = DigestAuditDriver(source, tmp_path / "audit", expected_count=count)
    try:
        with pytest.raises(ValueError):
            finish(driver)
        assert not driver.is_complete
        assert driver.result_path is None
    finally:
        close(driver)


def test_existing_scratch_is_preserved(tmp_path):
    source = tmp_path / "digests.bin"
    source.write_bytes(b"")
    scratch = tmp_path / "audit"
    scratch.mkdir()
    sentinel = scratch / "keep"
    sentinel.write_text("untouched")
    driver = DigestAuditDriver(source, scratch, expected_count=0)
    try:
        with pytest.raises(FileExistsError):
            finish(driver)
        assert sentinel.read_text() == "untouched"
    finally:
        close(driver)


def test_constructor_and_early_close_perform_no_io(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected source IO")

    with monkeypatch.context() as patch:
        patch.setattr("os.open", forbidden)
        driver = DigestAuditDriver(tmp_path / "missing", tmp_path / "audit", expected_count=0)
        driver.request_close()
        driver.advance(SliceIO(max_operations=0, max_io_bytes=0))
        assert driver.is_closed
        assert not driver.is_complete


def test_source_replacement_after_audit_read_invalidates_result(tmp_path, monkeypatch):
    source = tmp_path / "digests.bin"
    payload = b"".join(records(65))
    source.write_bytes(payload)
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(payload)
    original = SliceIO.read
    replaced = False

    def replace_after_read(self, descriptor, size):
        nonlocal replaced
        result = original(self, descriptor, size)
        if isinstance(result, bytes) and result and not replaced:
            replacement.replace(source)
            replaced = True
        return result

    monkeypatch.setattr(SliceIO, "read", replace_after_read)
    driver = DigestAuditDriver(source, tmp_path / "audit", expected_count=65)
    try:
        with pytest.raises(ValueError):
            finish(driver)
        assert replaced
        assert not driver.is_complete
        assert driver.result_path is None
    finally:
        close(driver)


def test_close_after_empty_result_open_releases_its_descriptor(tmp_path, monkeypatch):
    source = tmp_path / "digests.bin"
    source.write_bytes(b"")
    original = SliceIO.open
    result_fds = []

    def tracked(self, path, flags, mode=0o600):
        result = original(self, path, flags, mode)
        if path.name == "result.bin" and type(result) is int:
            result_fds.append(result)
        return result

    monkeypatch.setattr(SliceIO, "open", tracked)
    driver = DigestAuditDriver(source, tmp_path / "audit", expected_count=0)
    try:
        for _ in range(100):
            driver.advance(SliceIO(max_operations=1))
            if result_fds:
                break
        assert len(result_fds) == 1
        assert not driver.is_complete
        close(driver)
        with pytest.raises(OSError):
            os.fstat(result_fds[0])
    finally:
        close(driver)
        for descriptor in result_fds:
            try:
                os.close(descriptor)
            except OSError:
                pass


@pytest.mark.parametrize("reuse_closed_fd", [False, True])
def test_close_error_never_retries_a_raw_descriptor_or_blocks_other_releases(tmp_path, monkeypatch, reuse_closed_fd):
    source = tmp_path / "digests.bin"
    source.write_bytes(b"".join(records(65)))
    opened = []
    attempted = []
    replacements = []
    original_open = SliceIO.open
    original_close = SliceIO.close

    def tracked_open(self, path, flags, mode=0o600):
        result = original_open(self, path, flags, mode)
        if type(result) is int:
            opened.append(result)
        return result

    driver = DigestAuditDriver(source, tmp_path / "audit", expected_count=65)
    with monkeypatch.context() as patch:
        patch.setattr(SliceIO, "open", tracked_open)
        for _ in range(1000):
            driver.advance(SliceIO(max_operations=1))
            if len(opened) == 2:
                break
        assert len(opened) == 2
        failed_fd = opened[-1]

        def failed_close(self, descriptor):
            attempted.append(descriptor)
            if descriptor == failed_fd and (not reuse_closed_fd or not replacements):

                def fail():
                    if reuse_closed_fd:
                        os.close(descriptor)
                        replacement = os.open(tmp_path / "other-owner", os.O_CREAT | os.O_RDWR, 0o600)
                        replacements.append(replacement)
                        assert replacement == descriptor
                    raise OSError("close reported failure")

                return self._metadata(fail)
            return original_close(self, descriptor)

        patch.setattr(SliceIO, "close", failed_close)
        driver.request_close()
        failure = None
        try:
            for _ in range(20):
                try:
                    driver.advance(SliceIO(max_operations=1))
                except OSError as error:
                    failure = error
                    break
            assert failure is not None, "one failed close prevented the remaining cleanup from finishing"
            assert str(failure) == "close reported failure"
            assert attempted.count(failed_fd) == 1
            assert set(attempted) == set(opened)
            assert driver.is_closed
            assert not driver.is_complete
            if reuse_closed_fd:
                os.fstat(replacements[0])
        finally:
            patch.setattr(SliceIO, "close", original_close)
            try:
                close(driver)
            except OSError:
                pass
            for descriptor in set(opened + replacements):
                try:
                    os.close(descriptor)
                except OSError:
                    pass
