"""Progress protocol tests require neither a GPU nor a running agent."""

import json
import os
from pathlib import Path

import pytest

from qqtools.qexp._progress_protocol import (
    MAX_PAYLOAD_BYTES,
    read_advisory_snapshot,
    replace_advisory_snapshot,
    semantic_key,
    validate_payload,
)


def payload(**overrides):
    return (
        dict(
            protocol_version=1,
            update_id="update-1",
            stage="train",
            current=1,
            total=10,
            unit="step",
            message=None,
            **{},
        )
        | overrides
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"protocol_version": True},
        {"protocol_version": 2},
        {"update_id": "../other"},
        {"current": True},
        {"current": -1},
        {"current": 11},
        {"current": 1.5},
        {"current": float("nan")},
        {"total": float("inf")},
        {"total": False},
        {"stage": ""},
        {"stage": "x" * 65},
        {"stage": "\x1b[2J"},
        {"unit": "x" * 33},
        {"message": "x" * 1025},
        {"message": "a\nb"},
        {"metrics": {"loss": 1.0}},
        {"task_id": "other"},
        {"current": 2**64},
    ],
)
def test_invalid_payload_is_rejected(overrides):
    with pytest.raises(ValueError):
        validate_payload(payload(**overrides))


def test_unknown_total_and_empty_work():
    assert validate_payload(payload(total=None))["total"] is None
    assert validate_payload(payload(current=0, total=0))["current"] == 0
    assert validate_payload(payload(current=None, total=None, unit=None))["current"] is None


def test_message_is_not_semantic_progress():
    assert semantic_key(payload(message="a")) == semantic_key(payload(message="b"))
    assert semantic_key(payload(current=1)) != semantic_key(payload(current=2))


def test_advisory_replace_is_latest_only_without_fsync(tmp_path, monkeypatch):
    def forbidden(*args):
        raise AssertionError("advisory progress must not fsync")

    monkeypatch.setattr(os, "fsync", forbidden)
    path = tmp_path / "progress.json"
    for i in range(1000):
        replace_advisory_snapshot(path, payload(current=i, total=1000))
    assert read_advisory_snapshot(path)["current"] == 999
    assert list(tmp_path.iterdir()) == [path]


def test_advisory_replace_does_not_recreate_cleaned_directory(tmp_path):
    path = tmp_path / "cleaned-task" / "progress.json"
    with pytest.raises(FileNotFoundError):
        replace_advisory_snapshot(path, payload())
    assert not path.parent.exists()


def test_failed_rename_retains_previous_snapshot_and_removes_temp(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    replace_advisory_snapshot(path, payload())

    def fail(*args):
        raise OSError("unavailable")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError):
        replace_advisory_snapshot(path, payload(current=2))
    assert read_advisory_snapshot(path)["current"] == 1
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("contents", [b"{", b"[]", b"\xff", b'{"a":1,"a":2}', b"x" * (MAX_PAYLOAD_BYTES + 1)])
def test_bad_file(tmp_path, contents):
    path = tmp_path / "progress.json"
    path.write_bytes(contents)
    with pytest.raises((ValueError, UnicodeError)):
        read_advisory_snapshot(path, max_bytes=MAX_PAYLOAD_BYTES)


def test_reader_rejects_fifo_and_symlink(tmp_path):
    path = tmp_path / "fifo"
    os.mkfifo(path)
    with pytest.raises(ValueError):
        read_advisory_snapshot(path)
    target = tmp_path / "target"
    target.write_text(json.dumps(payload()))
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(OSError):
        read_advisory_snapshot(link)
