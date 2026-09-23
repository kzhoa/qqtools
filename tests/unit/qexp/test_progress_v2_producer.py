"""One public progress offer produces independent v1 and v2 snapshots."""

import math
import time

import pytest

from qqtools.qexp import progress
from qqtools.qexp._progress_protocol import read_advisory_snapshot
from qqtools.qexp._progress_protocol_v2 import validate_payload_v2


@pytest.fixture(autouse=True)
def reset_reporter(monkeypatch):
    progress.flush(timeout=0)
    for name in (
        "RANK",
        "SLURM_PROCID",
        "OMPI_COMM_WORLD_RANK",
        "QEXP_PROGRESS_PATH",
        "QEXP_PROGRESS_V2_PATH",
        "QEXP_PROGRESS_INTERVAL_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(progress, "_reporter", None)
    yield
    progress.flush(timeout=1)


def test_public_offer_writes_same_capture_and_update_id_to_both_mailboxes(tmp_path, monkeypatch):
    base_path = tmp_path / "latest.json"
    extended_path = tmp_path / "latest-v2.json"
    monkeypatch.setenv("QEXP_PROGRESS_PATH", str(base_path))
    monkeypatch.setenv("QEXP_PROGRESS_V2_PATH", str(extended_path))
    offered = {"loss": 0.284, "lr": 0.0001}
    assert progress.update(stage="custom-stage", current=4, total=None, unit="batch", message="ready", metrics=offered)
    offered["loss"] = 100
    progress.flush(timeout=1)
    base = read_advisory_snapshot(base_path)
    extended = validate_payload_v2(read_advisory_snapshot(extended_path))
    assert base == {key: value for key, value in extended.items() if key not in {"metrics", "completeness"}} | {
        "protocol_version": 1
    }
    assert extended["metrics"] == {"loss": 0.284, "lr": 0.0001}
    assert extended["completeness"] == {"complete": True, "omitted_metrics": 0, "reasons": []}


def test_without_v2_channel_metrics_are_not_traversed(tmp_path, monkeypatch):
    class HostileMetrics(dict):
        def __len__(self):
            raise AssertionError("metrics were inspected")

        def items(self):
            raise AssertionError("metrics were inspected")

    base_path = tmp_path / "latest.json"
    monkeypatch.setenv("QEXP_PROGRESS_PATH", str(base_path))
    assert progress.update(stage="train", current=1, metrics=HostileMetrics({"loss": math.nan}))
    progress.flush(timeout=1)
    assert read_advisory_snapshot(base_path)["current"] == 1


def test_invalid_metrics_keep_valid_base_and_mark_v2_incomplete(tmp_path, monkeypatch):
    base_path = tmp_path / "latest.json"
    extended_path = tmp_path / "latest-v2.json"
    monkeypatch.setenv("QEXP_PROGRESS_PATH", str(base_path))
    monkeypatch.setenv("QEXP_PROGRESS_V2_PATH", str(extended_path))
    assert progress.update(stage="train", current=1, metrics={"loss": 0.5, "bad": math.nan})
    progress.flush(timeout=1)
    assert read_advisory_snapshot(base_path)["current"] == 1
    extended = validate_payload_v2(read_advisory_snapshot(extended_path))
    assert extended["metrics"] == {"loss": 0.5}
    assert extended["completeness"] == {"complete": False, "omitted_metrics": 1, "reasons": ["invalid_metrics"]}


def test_failed_v2_mailbox_does_not_block_v1_or_repeat_successful_write(tmp_path, monkeypatch):
    base_path = tmp_path / "latest.json"
    broken_v2_path = tmp_path / "missing-parent" / "latest-v2.json"
    monkeypatch.setenv("QEXP_PROGRESS_PATH", str(base_path))
    monkeypatch.setenv("QEXP_PROGRESS_V2_PATH", str(broken_v2_path))
    writes = []
    original_replace = progress.replace_advisory_snapshot

    def record_replace(path, payload, *, max_bytes):
        writes.append(path)
        return original_replace(path, payload, max_bytes=max_bytes)

    monkeypatch.setattr(progress, "replace_advisory_snapshot", record_replace)
    assert progress.update(stage="train", current=1, metrics={"loss": 0.5})
    deadline = time.monotonic() + 1
    while not base_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert read_advisory_snapshot(base_path)["current"] == 1
    assert not broken_v2_path.exists()
    assert writes.count(base_path) == 1
    writer = progress._reporter
    assert progress.update(stage="train", current=1, metrics={"loss": 0.5})
    assert progress._reporter is writer
    broken_v2_path.parent.mkdir()
    deadline = time.monotonic() + 3
    while not broken_v2_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert (
        validate_payload_v2(read_advisory_snapshot(broken_v2_path))["update_id"]
        == read_advisory_snapshot(base_path)["update_id"]
    )
    progress.flush(timeout=1)
    assert writes.count(base_path) == 1
