"""Whole-source durable confirmation boundaries and interrupted replay."""

import os

import pytest

from qqtools.plugins.qexp.runtime.group_discovery.driver import ProjectionDriver
from qqtools.plugins.qexp.runtime.group_discovery.recovery import RecoverableSource
from qqtools.plugins.qexp.runtime.group_discovery.slice_io import SliceIO
from qqtools.plugins.qexp.runtime.group_discovery.source_qualification import SourceQualification
from tests.helpers.qexp_discovery import close_source, finish_source, source_file

pytestmark = pytest.mark.integration


def make_session(tmp_path, **kwargs):
    source = source_file(tmp_path / "op-1.json", **kwargs)
    return RecoverableSource(source, tmp_path / "work", "op-1", "experiment")


def reopen(tmp_path):
    return RecoverableSource(tmp_path / "op-1.json", tmp_path / "work", "op-1", "experiment")


def test_completed_confirmation_reopens_without_parsing_or_auditing(tmp_path, monkeypatch):
    session = make_session(tmp_path)
    try:
        result = finish_source(session)
        assert result.status == "qualified"
        assert result.task_count == 1
        assert (tmp_path / "work/receipt.json").is_file()
    finally:
        close_source(session)

    def forbidden(*args, **kwargs):
        pytest.fail("durably confirmed source was parsed/audited again")

    monkeypatch.setattr(ProjectionDriver, "advance", forbidden)
    monkeypatch.setattr(SourceQualification, "advance", forbidden)
    resumed = reopen(tmp_path)
    try:
        restored = finish_source(resumed, io_bytes=17, operations=2)
        assert restored == result
    finally:
        close_source(resumed)


def test_interrupted_qualification_reuses_completed_projection(tmp_path, monkeypatch):
    session = make_session(tmp_path, tasks=[f"task-{i}" for i in range(130)], sequences=list(range(1, 131)))
    called = 0
    advance = SourceQualification.advance

    def observed(self, *args, **kwargs):
        nonlocal called
        called += 1
        return advance(self, *args, **kwargs)

    monkeypatch.setattr(SourceQualification, "advance", observed)
    try:
        for _ in range(10000):
            session.advance(SliceIO(max_io_bytes=4096, max_operations=8))
            if called >= 3:
                break
        assert called >= 3
        assert not session.is_complete
        assert not (tmp_path / "work/receipt.json").exists()
    finally:
        close_source(session)
    resumed = reopen(tmp_path)
    try:
        result = finish_source(resumed)
        assert result.status == "qualified"
        assert result.task_count == 130
    finally:
        close_source(resumed)


@pytest.mark.parametrize("target", ["source", "spool", "task_references", "sequence_references"])
def test_replaced_source_or_confirmed_output_cannot_reuse_receipt(tmp_path, target):
    session = make_session(tmp_path)
    try:
        result = finish_source(session)
    finally:
        close_source(session)
    path = getattr(result, target)
    replacement = path.with_name(path.name + ".replacement")
    replacement.write_bytes(path.read_bytes())
    os.replace(replacement, path)
    resumed = reopen(tmp_path)
    try:
        with pytest.raises((ValueError, RuntimeError)):
            finish_source(resumed)
        assert not resumed.is_complete
        assert resumed.result is None
    finally:
        close_source(resumed)


@pytest.mark.parametrize(
    "state,tasks,sequences,status",
    [
        ("aborted", ["task-a"], [1], "irrelevant"),
        ("committed", ["task-a", "task-a"], [1, 2], "ambiguous"),
        ("committed", [], [], "qualified"),
    ],
)
def test_terminal_classifications_survive_restart(tmp_path, state, tasks, sequences, status):
    session = make_session(tmp_path, state=state, tasks=tasks, sequences=sequences)
    try:
        result = finish_source(session)
        assert result.status == status
    finally:
        close_source(session)
    resumed = reopen(tmp_path)
    try:
        assert finish_source(resumed) == result
    finally:
        close_source(resumed)


def test_small_slices_never_publish_partial_confirmation(tmp_path):
    session = make_session(tmp_path)
    try:
        for _ in range(10):
            session.advance(SliceIO(max_io_bytes=1, max_operations=1))
            assert session.result is None
        assert finish_source(session, io_bytes=31, operations=2).status == "qualified"
    finally:
        close_source(session)


def test_malformed_receipt_is_unavailable_not_empty_success(tmp_path):
    session = make_session(tmp_path)
    try:
        finish_source(session)
    finally:
        close_source(session)
    (tmp_path / "work/receipt.json").write_text('{"version": 999}')
    resumed = reopen(tmp_path)
    try:
        with pytest.raises((ValueError, RuntimeError)):
            finish_source(resumed)
        assert resumed.result is None
    finally:
        close_source(resumed)


@pytest.mark.parametrize("cut", ["qualification", "receipt"])
def test_process_exit_recovers_without_destructor_cleanup(tmp_path, cut):
    import subprocess
    import sys
    from pathlib import Path

    source_file(tmp_path / "op-1.json", tasks=[f"task-{i}" for i in range(130)], sequences=list(range(1, 131)))
    program = r"""
import os
import sys
from pathlib import Path
from qqtools.plugins.qexp.runtime.group_discovery.recovery import RecoverableSource
from qqtools.plugins.qexp.runtime.group_discovery.source_qualification import SourceQualification
from qqtools.plugins.qexp.runtime.group_discovery.slice_io import SliceIO
root = Path(sys.argv[1])
cut = sys.argv[2]
original = SourceQualification.advance
calls = 0
def interrupted(self, *args, **kwargs):
    global calls
    result = original(self, *args, **kwargs)
    calls += 1
    if cut == "qualification" and calls == 3:
        os._exit(91)
    return result
SourceQualification.advance = interrupted
session = RecoverableSource(root / "op-1.json", root / "work", "op-1", "experiment")
for _ in range(20000):
    session.advance(SliceIO(max_io_bytes=4096, max_operations=8))
    if session.is_complete:
        os._exit(92)
raise RuntimeError("child did not reach crash boundary")
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[3] / "src")
    child = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path), cut], env=environment, capture_output=True, timeout=30
    )
    assert child.returncode == (91 if cut == "qualification" else 92), child.stderr.decode()
    resumed = reopen(tmp_path)
    try:
        result = finish_source(resumed)
        assert result.status == "qualified"
        assert result.task_count == 130
    finally:
        close_source(resumed)
