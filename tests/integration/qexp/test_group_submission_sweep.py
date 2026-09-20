"""Directory discovery is sliced and EOF never publishes membership proof."""

from __future__ import annotations

import time

import pytest

from qqtools.plugins.qexp.runtime.group_discovery.slice_io import SliceIO
from qqtools.plugins.qexp.runtime.group_discovery.source_sweep import SubmissionSourceSweep

pytestmark = pytest.mark.integration


def complete(sweep, *, operations=2, entries=3):
    candidates = []
    for _ in range(1000):
        io = SliceIO(max_io_bytes=0, max_operations=operations)
        step = sweep.advance(io, max_entries=entries)
        assert io.operations_used <= operations
        assert io.io_bytes_used == 0
        if step.state in {"page", "complete"}:
            candidates.extend(step.candidates)
        else:
            assert not step.candidates
        if step.state == "complete":
            return tuple(candidates)
    raise AssertionError("source enumeration did not complete")


def close(sweep):
    sweep.request_close()
    for _ in range(10):
        step = sweep.advance(SliceIO(max_operations=1))
        if step.state == "closed":
            assert sweep.is_closed
            return
    raise AssertionError("source enumeration leaked directory ownership")


def test_pages_visit_each_stable_source_once_with_tiny_budgets(tmp_path, monkeypatch):
    directory = tmp_path / "sources"
    directory.mkdir()
    expected = [f"op-{index:03}.json" for index in range(151)]
    for name in reversed(expected):
        (directory / name).write_text("{}")
    (directory / "unrelated.tmp").write_text("ignored")
    (directory / "bad name.json").write_text("ignored")
    original = SliceIO.next_entry
    visits = []

    def tracked(self, iterator):
        before = self.operations_used
        result = original(self, iterator)
        if self.operations_used > before:
            visits.append(result)
        return result

    monkeypatch.setattr(SliceIO, "next_entry", tracked)
    sweep = SubmissionSourceSweep(directory, page_size=17)
    try:
        visited = [path.name for path in complete(sweep)]
        assert sorted(visited) == expected
        assert len(visits) == 154  # 151 sources, two unrelated names, one EOF.
    finally:
        close(sweep)


def test_budget_deadline_and_close_before_open_do_no_io(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected directory IO")

    with monkeypatch.context() as patch:
        patch.setattr("os.scandir", forbidden)
        sweep = SubmissionSourceSweep(tmp_path)
        assert sweep.advance(SliceIO(max_operations=0)).state == "waiting"
        assert sweep.advance(SliceIO(), soft_deadline=time.monotonic() - 1).state == "waiting"
        close(sweep)


def test_discovery_returns_candidates_without_parsing_or_following_links(tmp_path):
    (tmp_path / "op-a.json").write_text("invalid JSON")
    (tmp_path / "op-b.json").mkdir()
    (tmp_path / "op-c.json").symlink_to(tmp_path / "missing")
    sweep = SubmissionSourceSweep(tmp_path)
    try:
        assert sorted(path.name for path in complete(sweep)) == ["op-a.json", "op-b.json", "op-c.json"]
    finally:
        close(sweep)


def test_partial_page_is_not_exposed_and_close_releases_handle(tmp_path, monkeypatch):
    (tmp_path / "op-a.json").write_text("{}")
    sweep = SubmissionSourceSweep(tmp_path)
    io = SliceIO(max_operations=1)
    assert not sweep.advance(io).candidates
    seen = []
    original = SliceIO.close_directory

    def tracked(self, iterator):
        seen.append(iterator)
        return original(self, iterator)

    monkeypatch.setattr(SliceIO, "close_directory", tracked)
    close(sweep)
    assert len(seen) == 1


def test_missing_directory_is_an_error_not_empty_complete_page(tmp_path):
    sweep = SubmissionSourceSweep(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        sweep.advance(SliceIO())
    close(sweep)


def test_directory_read_error_never_publishes_partial_page(tmp_path, monkeypatch):
    (tmp_path / "op-a.json").write_text("{}")
    sweep = SubmissionSourceSweep(tmp_path)
    # Open and consume one source, but do not reach EOF or publish a partial page.
    step = sweep.advance(SliceIO(max_operations=2), max_entries=1)
    assert not step.candidates

    def failed_read(self, iterator):
        raise OSError("directory storage unavailable")

    monkeypatch.setattr(SliceIO, "next_entry", failed_read)
    with pytest.raises(OSError, match="storage unavailable"):
        sweep.advance(SliceIO())
    close(sweep)


@pytest.mark.parametrize("explicit", [False, True])
def test_failed_close_retains_ownership_until_accounted_retry(tmp_path, monkeypatch, explicit):
    directory = tmp_path / "sources"
    directory.mkdir()
    (directory / "op-a.json").write_text("{}")
    sweep = SubmissionSourceSweep(directory)
    # Retain one unpublished candidate and the directory handle.
    sweep.advance(SliceIO(max_operations=2), max_entries=1)
    if explicit:
        sweep.request_close()
    original = SliceIO.close_directory
    attempts = []

    def close_once_failed(self, iterator):
        attempts.append(iterator)
        if len(attempts) == 1:

            def fail():
                raise OSError("close interrupted")

            return self._metadata(fail)
        return original(self, iterator)

    monkeypatch.setattr(SliceIO, "close_directory", close_once_failed)
    io = SliceIO(max_operations=2)
    try:
        with pytest.raises(OSError, match="close interrupted"):
            sweep.advance(io)
        assert io.operations_used == (1 if explicit else 2)
        assert not sweep.is_closed
        retry_io = SliceIO(max_operations=1)
        step = sweep.advance(retry_io)
        assert retry_io.operations_used == 1
        assert len(attempts) == 2 and attempts[0] is attempts[1]
        if explicit:
            assert step.state == "closed"
            assert not step.candidates
        else:
            assert step.state == "complete"
            assert step.candidates == (directory / "op-a.json",)
            assert sweep.advance(SliceIO(max_operations=0)) == step
    finally:
        close(sweep)
