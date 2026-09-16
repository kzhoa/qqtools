import json
from pathlib import Path
from types import SimpleNamespace

from tests.conftest import _CollectionManifest, _TimingJson


def test_collection_manifest_merges_worker_collections(tmp_path: Path) -> None:
    manifest = _CollectionManifest(tmp_path / "collection.txt")
    manifest.pytest_xdist_node_collection_finished(SimpleNamespace(), ["test_a.py::test_a"])
    manifest.pytest_xdist_node_collection_finished(SimpleNamespace(), ["test_a.py::test_a", "test_b.py::test_b"])
    session = SimpleNamespace(config=SimpleNamespace(), exitstatus=0)

    manifest.pytest_sessionfinish(session, 0)

    assert manifest.path.read_text(encoding="utf-8") == ("test_a.py::test_a\ntest_b.py::test_b\n")


def test_timing_json_records_each_test_phase(tmp_path: Path, monkeypatch) -> None:
    timing = _TimingJson(tmp_path / "timing.json")
    timing.pytest_runtest_logreport(
        SimpleNamespace(
            nodeid="test_a.py::test_a",
            when="call",
            outcome="passed",
            duration=0.25,
            skipped=False,
        )
    )
    session = SimpleNamespace(config=SimpleNamespace(), exitstatus=0)
    monkeypatch.setattr("tests.conftest.time.monotonic", lambda: timing.started_at + 0.5)

    timing.pytest_sessionfinish(session, 0)

    payload = json.loads(timing.path.read_text(encoding="utf-8"))
    assert payload["exit_code"] == 0
    assert payload["duration_seconds"] == 0.5
    assert payload["has_skipped"] is False
    assert payload["reports"][0]["nodeid"] == "test_a.py::test_a"


def test_timing_json_turns_a_skip_into_a_failed_phase(tmp_path: Path) -> None:
    timing = _TimingJson(tmp_path / "timing.json")
    timing.pytest_runtest_logreport(
        SimpleNamespace(
            nodeid="test_a.py::test_a",
            when="setup",
            outcome="skipped",
            duration=0.0,
            skipped=True,
        )
    )
    session = SimpleNamespace(config=SimpleNamespace(), exitstatus=0)

    timing.pytest_sessionfinish(session, 0)

    assert session.exitstatus == 1
    assert json.loads(timing.path.read_text(encoding="utf-8"))["has_skipped"] is True
