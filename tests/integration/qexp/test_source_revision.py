"""Integration coverage for the stable-source binding prototype."""

from __future__ import annotations

import json
import os
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

import pytest

import qqtools.plugins.qexp.runtime.group_discovery.source_revision as source_revision
from qqtools.plugins.qexp.runtime.group_discovery.json_stream import Scanner
from qqtools.plugins.qexp.runtime.group_discovery.source_revision import BoundSource, SourceChangedError, SourceRevision

pytestmark = pytest.mark.integration


def test_scanner_reads_lexically_valid_json_through_bound_source_and_verifies(tmp_path: Path):
    payload = b'{"submission":{"id":"s-1"},"tasks":[{"id":"t-1"},{"id":"t-2"}]}'
    path = tmp_path / "payload.json"
    path.write_bytes(payload)
    spans = []

    with BoundSource.open(path) as source:
        scanner = Scanner(source, spans.append, chunk_bytes=7)
        while True:
            result = scanner.step(11)
            if result.is_complete:
                break
        assert source.tell() == len(payload)
        source.verify()

    scanned = b"".join(payload[span.start : span.end] for span in spans)
    assert json.loads(scanned) == json.loads(payload)


def test_revision_roundtrip_reopens_with_expected_revision_and_reads_scalars(tmp_path: Path):
    payload = b"0123456789abcdef"
    path = tmp_path / "scalar-source.bin"
    path.write_bytes(payload)

    with BoundSource.open(path) as source:
        serialized = json.dumps(asdict(source.revision))
        revision = SourceRevision(**json.loads(serialized))
        assert revision == source.revision

    with BoundSource.open(path, expected_revision=revision) as reopened:
        assert reopened.seek(5) == 5
        assert reopened.read(4) == payload[5:9]
        assert reopened.tell() == 9


def test_atomic_replace_with_identical_bytes_invalidates_old_source_and_expected_reopen(tmp_path: Path):
    path = tmp_path / "atomic.json"
    path.write_bytes(b'{"same":true}')
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(path.read_bytes())

    source = BoundSource.open(path)
    revision = source.revision
    os.replace(replacement, path)

    with pytest.raises(SourceChangedError):
        source.verify()
    assert source.closed
    with pytest.raises(SourceChangedError):
        BoundSource.open(path, expected_revision=revision)


@pytest.mark.parametrize("mutation", ["replace", "unlink", "rewrite", "truncate"])
def test_source_mutations_fail_verification_and_close_source(tmp_path: Path, mutation: str):
    path = tmp_path / f"{mutation}.bin"
    path.write_bytes(b"original-content")
    source = BoundSource.open(path)

    if mutation == "replace":
        replacement = tmp_path / "replacement.bin"
        replacement.write_bytes(path.read_bytes())
        os.replace(replacement, path)
    elif mutation == "unlink":
        path.unlink()
    elif mutation == "rewrite":
        path.write_bytes(b"modified-content")
        os.utime(path, ns=(source.revision.mtime_ns + 1, source.revision.mtime_ns + 1))
        assert path.stat().st_mtime_ns != source.revision.mtime_ns
    elif mutation == "truncate":
        os.truncate(path, 1)
    else:
        raise AssertionError(f"unknown mutation: {mutation}")

    with pytest.raises(SourceChangedError):
        source.verify()
    assert source.closed
    with pytest.raises(ValueError):
        source.read(1)


def test_expected_revision_rejects_missing_and_symlinked_reopens(tmp_path: Path):
    path = tmp_path / "expected.json"
    path.write_bytes(b'{"value":1}')
    with BoundSource.open(path) as source:
        revision = source.revision

    path.unlink()
    with pytest.raises(SourceChangedError):
        BoundSource.open(path, expected_revision=revision)

    target = tmp_path / "target.json"
    target.write_bytes(b'{"value":1}')
    path.symlink_to(target)
    with pytest.raises(SourceChangedError):
        BoundSource.open(path, expected_revision=revision)


def test_expected_revision_rejects_path_unlinked_between_open_and_lstat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "raced.json"
    path.write_bytes(b'{"value":1}')
    with BoundSource.open(path) as source:
        revision = source.revision

    original_lstat = source_revision.os.lstat
    original_open = source_revision.os.open
    opened: list[int] = []

    def capture_open(candidate, flags, *args, **kwargs):
        descriptor = original_open(candidate, flags, *args, **kwargs)
        opened.append(descriptor)
        return descriptor

    def unlink_before_lstat(candidate: os.PathLike[str] | str) -> os.stat_result:
        if Path(candidate) == path:
            path.unlink()
        return original_lstat(candidate)

    with monkeypatch.context() as context:
        context.setattr(source_revision.os, "open", capture_open)
        context.setattr(source_revision.os, "lstat", unlink_before_lstat)
        with pytest.raises(SourceChangedError):
            BoundSource.open(path, expected_revision=revision)
    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])


def test_initial_symlink_directory_and_fifo_are_rejected_without_blocking(tmp_path: Path):
    regular = tmp_path / "regular.bin"
    regular.write_bytes(b"data")
    symlink = tmp_path / "symlink.bin"
    symlink.symlink_to(regular)
    with pytest.raises(ValueError):
        BoundSource.open(symlink)

    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ValueError):
        BoundSource.open(directory)

    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError):
        BoundSource.open(fifo)


@pytest.mark.parametrize(
    "field, value",
    [
        ("device", True),
        ("inode", False),
        ("size", 1.0),
        ("mtime_ns", "1"),
        ("ctime_ns", None),
    ],
)
def test_revision_rejects_non_exact_integer_fields(field: str, value: object):
    values = {"device": 1, "inode": 2, "size": 3, "mtime_ns": -4, "ctime_ns": 5}
    values[field] = value
    with pytest.raises(TypeError):
        SourceRevision(**values)


@pytest.mark.parametrize("field", ["device", "inode", "size"])
def test_revision_rejects_negative_identity_or_size_fields(field: str):
    values = {"device": 1, "inode": 2, "size": 3, "mtime_ns": -4, "ctime_ns": -5}
    values[field] = -1
    with pytest.raises(ValueError):
        SourceRevision(**values)


def test_revision_allows_signed_timestamps_and_is_immutable():
    revision = SourceRevision(0, 0, 0, -1, -2)
    assert revision.mtime_ns == -1
    with pytest.raises(FrozenInstanceError):
        revision.size = 1


def test_read_and_seek_validate_bounds_and_boolean_or_float_inputs(tmp_path: Path):
    path = tmp_path / "bounds.bin"
    path.write_bytes(b"0123456789")

    with BoundSource.open(path) as source:
        for value in (0, 65_537, True, -1, 1.0):
            with pytest.raises((TypeError, ValueError)):
                source.read(value)
        for value in (-1, source.revision.size + 1, True, 1.0):
            with pytest.raises((TypeError, ValueError)):
                source.seek(value)
        assert source.seek(source.revision.size) == source.revision.size
        assert source.read(1) == b""


def test_read_requests_exact_size_without_hidden_prefetch(tmp_path: Path):
    path = tmp_path / "offset.bin"
    path.write_bytes(b"abcdef")

    with BoundSource.open(path) as source:
        assert source.tell() == 0
        assert source.read(1) == b"a"
        assert source.tell() == 1
        assert os.lseek(source._handle.fileno(), 0, os.SEEK_CUR) == 1
        assert source.read(3) == b"bcd"
        assert source.tell() == 4
        assert os.lseek(source._handle.fileno(), 0, os.SEEK_CUR) == 4


def test_close_is_idempotent_and_context_managers_close_on_success_and_error(tmp_path: Path):
    path = tmp_path / "lifecycle.bin"
    path.write_bytes(b"data")

    source = BoundSource.open(path)
    source.close()
    source.close()
    assert source.closed
    for operation in (lambda: source.read(1), source.tell, lambda: source.seek(0), source.verify):
        with pytest.raises(ValueError):
            operation()

    normal = None
    with BoundSource.open(path) as current:
        normal = current
    assert normal is not None and normal.closed

    failed = None
    with pytest.raises(RuntimeError):
        with BoundSource.open(path) as current:
            failed = current
            raise RuntimeError("test failure")
    assert failed is not None and failed.closed


@pytest.mark.parametrize("replace_original", [False, True])
def test_relative_source_remains_bound_to_original_path_after_chdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replace_original: bool
):
    original_dir = tmp_path / "original"
    other_dir = tmp_path / "other"
    original_dir.mkdir()
    other_dir.mkdir()
    original_path = original_dir / "payload"
    original_path.write_bytes(b"old")
    if replace_original:
        os.link(original_path, other_dir / "payload")

    with monkeypatch.context() as context:
        context.chdir(original_dir)
        with BoundSource.open(Path("payload")) as source:
            context.chdir(other_dir)
            if not replace_original:
                source.verify()
                assert source.read(3) == b"old"
                return

            # Moving the parent preserves the opened file's stamp; only the
            # originally named path now points at another file.
            original_dir.rename(tmp_path / "retained-original")
            original_dir.mkdir()
            original_path.write_bytes(b"new")
            assert SourceRevision.from_stat(os.fstat(source._handle.fileno())) == source.revision
            with pytest.raises(SourceChangedError):
                source.verify()
            assert source.closed
