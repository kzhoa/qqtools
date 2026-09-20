"""Real-directory discovery bounds and recovery of advisory authority cursors."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp.runtime import authority_scan
from qqtools.plugins.qexp.runtime.authority_scan import EvidenceScan, is_path_present, iter_evidence_files

pytestmark = pytest.mark.integration


def test_strict_presence_only_accepts_missing_metadata_as_absence(tmp_path, monkeypatch):
    target = tmp_path / "entry"
    assert not is_path_present(target)
    target.touch()
    assert is_path_present(target)
    original = Path.stat

    def unavailable(path, **kwargs):
        if path == target:
            raise OSError("metadata I/O failure")
        return original(path, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "stat", unavailable)
        with pytest.raises(OSError, match="metadata I/O failure"):
            is_path_present(target)


def test_strict_import_scan_distinguishes_missing_and_invalid_lanes(tmp_path):
    directory = tmp_path / "evidence"
    assert list(iter_evidence_files(directory, recursive=True)) == []
    directory.write_text("not a directory")
    with pytest.raises(NotADirectoryError):
        list(iter_evidence_files(directory, recursive=True))


@pytest.mark.parametrize("is_directory", [False, True])
def test_strict_import_scan_rejects_linked_evidence(tmp_path, is_directory):
    directory = tmp_path / "evidence"
    directory.mkdir()
    target = tmp_path / "target"
    if is_directory:
        target.mkdir()
    else:
        target.write_text("{}")
    (directory / "linked.json").symlink_to(target, target_is_directory=is_directory)
    with pytest.raises(OSError, match="symlink"):
        list(iter_evidence_files(directory, recursive=True))


def test_strict_import_scan_preserves_nested_paths(tmp_path):
    tmp_path = tmp_path / "evidence"
    nested = tmp_path / "attempt" / "decisions"
    nested.mkdir(parents=True)
    files = {tmp_path / "root.json", nested / "decision.json"}
    for path in files:
        path.write_text("{}")
    (tmp_path / "unrelated").write_text("ignored")
    assert set(iter_evidence_files(tmp_path, recursive=True)) == files
    assert list(iter_evidence_files(tmp_path)) == [tmp_path / "root.json"]


def test_strict_import_scan_propagates_root_metadata_failure(tmp_path, monkeypatch):
    original = Path.stat

    def unavailable(path, **kwargs):
        if path == tmp_path:
            raise PermissionError("metadata inaccessible")
        return original(path, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "stat", unavailable)
        with pytest.raises(PermissionError, match="metadata inaccessible"):
            list(iter_evidence_files(tmp_path))


def _sweep(scan: EvidenceScan, limit: int) -> tuple[set[str], int]:
    names: set[str] = set()
    visited = 0
    for _ in range(100):
        page = scan.take(limit)
        assert page.entries_visited <= limit
        assert len(page.paths) <= page.entries_visited
        names.update(path.name for path in page.paths)
        visited += page.entries_visited
        if page.is_complete:
            return names, visited
    pytest.fail("bounded discovery did not complete its sweep")


@pytest.mark.parametrize("limit", [1, 4, 64])
def test_discovery_counts_unrelated_entries_and_reaches_tail(tmp_path: Path, limit: int):
    tmp_path = tmp_path / "evidence"
    tmp_path.mkdir()
    for number in range(65):
        (tmp_path / f"{number}.json").write_text("{}")
    (tmp_path / "not-json").touch()
    (tmp_path / "directory.json").mkdir()
    (tmp_path / "link.json").symlink_to(tmp_path / "0.json")
    scan = EvidenceScan(tmp_path)
    try:
        names, visited = _sweep(scan, limit)
        assert names == {f"{number}.json" for number in range(65)}
        assert visited == 68
    finally:
        scan.close()


def test_replaced_directory_discards_old_cursor(tmp_path: Path):
    directory = tmp_path / "evidence"
    directory.mkdir()
    for number in range(4):
        (directory / f"old-{number}.json").touch()
    scan = EvidenceScan(directory)
    try:
        assert scan.take(1).entries_visited == 1
        directory.rename(tmp_path / "retired")
        directory.mkdir()
        (directory / "new.json").touch()
        assert _sweep(scan, 1) == ({"new.json"}, 1)
    finally:
        scan.close()


def test_restart_and_new_sweep_find_records_without_notifications(tmp_path: Path):
    tmp_path = tmp_path / "evidence"
    tmp_path.mkdir()
    (tmp_path / "old.json").touch()
    scan = EvidenceScan(tmp_path)
    assert _sweep(scan, 1) == ({"old.json"}, 1)
    (tmp_path / "new.json").touch()
    assert _sweep(scan, 1) == ({"old.json", "new.json"}, 2)
    scan.take(1)
    scan.close()
    restarted = EvidenceScan(tmp_path)
    try:
        assert _sweep(restarted, 1) == ({"old.json", "new.json"}, 2)
    finally:
        restarted.close()


def test_missing_directory_can_appear_later(tmp_path: Path):
    directory = tmp_path / "later"
    scan = EvidenceScan(directory)
    assert scan.take(1).is_complete
    directory.mkdir()
    (directory / "entry.json").touch()
    assert _sweep(scan, 1) == ({"entry.json"}, 1)


def test_directory_scan_does_not_follow_symlinks(tmp_path: Path):
    tmp_path = tmp_path / "evidence"
    tmp_path.mkdir()
    (tmp_path / "attempt").mkdir()
    (tmp_path / "file").touch()
    (tmp_path / "linked-attempt").symlink_to(tmp_path / "attempt", target_is_directory=True)
    scan = EvidenceScan(tmp_path, directories=True)
    assert _sweep(scan, 1) == ({"attempt"}, 3)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_budget_fails_before_discovery(tmp_path: Path, limit):
    scan = EvidenceScan(tmp_path / "absent")
    with pytest.raises(ValueError, match="positive integer"):
        scan.take(limit)


@pytest.mark.parametrize("limit", [1, 64])
def test_scan_never_enumerates_beyond_current_budget(tmp_path: Path, monkeypatch, limit):
    directory = tmp_path / "evidence"
    directory.mkdir()
    for number in range(256):
        (directory / f"{number}.json").touch()
    original = authority_scan.os.scandir
    calls = []
    handles = []

    class CountedDirectory:
        def __init__(self, path):
            self.entries = original(path)
            self.is_closed = False
            handles.append(self)

        def __next__(self):
            calls.append("entry")
            return next(self.entries)

        def close(self):
            self.is_closed = True
            self.entries.close()

    monkeypatch.setattr(authority_scan, "os", SimpleNamespace(scandir=CountedDirectory))
    scan = EvidenceScan(directory)
    try:
        first = scan.take(limit)
        assert len(calls) == limit
        second = scan.take(limit)
        assert len(calls) == 2 * limit
        assert set(first.paths).isdisjoint(second.paths)
        assert len(handles) == 1
    finally:
        scan.close()
    assert handles[0].is_closed


def test_discovery_retries_after_storage_failure(tmp_path: Path, monkeypatch):
    directory = tmp_path / "evidence"
    directory.mkdir()
    (directory / "retained.json").touch()
    original = authority_scan.os.scandir
    scan = EvidenceScan(directory)

    def unavailable(_path):
        raise OSError("storage unavailable")

    monkeypatch.setattr(authority_scan, "os", SimpleNamespace(scandir=unavailable))
    with pytest.raises(OSError, match="storage unavailable"):
        scan.take(1)
    monkeypatch.setattr(authority_scan, "os", SimpleNamespace(scandir=original))
    assert _sweep(scan, 1) == ({"retained.json"}, 1)


def test_slice_checks_identity_once_and_replacement_on_next_slice(tmp_path, monkeypatch):
    directory = tmp_path / "evidence"
    directory.mkdir()
    for number in range(8):
        (directory / f"old-{number}.json").touch()
    original = Path.stat
    checks = []

    def stat(path, *args, **kwargs):
        if path == directory:
            checks.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    scan = EvidenceScan(directory)
    token = object()
    try:
        for _ in range(4):
            assert scan.take(1, slice_token=token).entries_visited == 1
        assert len(checks) == 1
        directory.rename(tmp_path / "retired")
        directory.mkdir()
        (directory / "new.json").touch()
        page = scan.take(4, slice_token=object())
        assert page.paths == (directory / "new.json",)
        assert page.is_complete
        assert len(checks) == 2
    finally:
        scan.close()


@pytest.mark.parametrize("is_initially_missing", [False, True])
def test_slice_defers_new_sweep_after_empty_until_next_token(tmp_path, monkeypatch, is_initially_missing):
    directory = tmp_path / "evidence"
    if not is_initially_missing:
        directory.mkdir()
    scan = EvidenceScan(directory)
    token = object()
    assert scan.take(1, slice_token=token).is_complete
    directory.mkdir(exist_ok=True)
    (directory / "new.json").touch()
    for _ in range(8):
        page = scan.take(1, slice_token=token)
        assert page.is_complete and not page.paths
    assert scan.take(1, slice_token=object()).paths == (directory / "new.json",)
    scan.close()
