"""One source offer fans out to independent bounded channels."""

import time

import pytest

from qqtools.qexp import progress
from qqtools.qexp._progress_protocol import read_advisory_snapshot


@pytest.fixture(autouse=True)
def reporter_scope(monkeypatch):
    progress.flush(timeout=1)
    for name in (
        "RANK",
        "SLURM_PROCID",
        "OMPI_COMM_WORLD_RANK",
        "QEXP_PROGRESS_PATH",
        "QEXP_PROGRESS_V2_PATH",
        "QEXP_PROGRESS_V3_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(progress, "_reporter", None)
    yield
    progress.flush(timeout=1)


def channels(tmp_path, monkeypatch, versions=(1, 2, 3)):
    paths = {}
    for version in versions:
        path = tmp_path / f"latest-{version}.json"
        variable = "QEXP_PROGRESS_PATH" if version == 1 else f"QEXP_PROGRESS_V{version}_PATH"
        monkeypatch.setenv(variable, str(path))
        paths[version] = path
    return paths


def test_one_offer_shared_identity_independent_payloads(tmp_path, monkeypatch):
    paths = channels(tmp_path, monkeypatch)
    counter = progress.Counter(current=33574, total=42000, unit="step", label="Training")
    assert progress.update(
        stage="validation", current=184, total=256, unit="batch", metrics={"loss": 0.3}, overall=counter
    )
    progress.flush(timeout=1)
    values = {v: read_advisory_snapshot(p) for v, p in paths.items()}
    assert len({item["update_id"] for item in values.values()}) == 1
    assert values[3]["activity"] == {k: values[1][k] for k in ("stage", "current", "total", "unit", "message")}
    assert values[3]["overall"] == {"current": 33574, "total": 42000, "unit": "step", "label": "Training"}
    assert "overall" not in values[1] and "overall" not in values[2]


@pytest.mark.parametrize("versions", [(1,), (1, 2), (3,), (1, 2, 3)])
def test_invalid_overall_rejects_before_any_channel_write(tmp_path, monkeypatch, versions):
    paths = channels(tmp_path, monkeypatch, versions)
    assert not progress.update(stage="train", overall=progress.Counter(current=True, unit="step"))
    progress.flush(timeout=1)
    assert not any(p.exists() for p in paths.values())


def test_no_v3_accepts_valid_overall_and_keeps_old_flat_shape(tmp_path, monkeypatch):
    paths = channels(tmp_path, monkeypatch, (1, 2))
    assert progress.update(
        stage="epoch", current=0, total=4, unit="epoch", overall=progress.Counter(current=0, unit="epoch")
    )
    progress.flush(timeout=1)
    assert read_advisory_snapshot(paths[1])["current"] == 0
    assert "overall" not in read_advisory_snapshot(paths[2])


def test_omitting_overall_replaces_instead_of_retaining_counter(tmp_path, monkeypatch):
    paths = channels(tmp_path, monkeypatch, (3,))
    assert progress.update(stage="train", overall=progress.Counter(current=1, unit="step"))
    deadline = time.monotonic() + 2
    while not paths[3].exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert read_advisory_snapshot(paths[3])["overall"]["current"] == 1
    assert progress.update(stage="evaluation")
    progress.flush(timeout=1)
    assert read_advisory_snapshot(paths[3])["overall"] is None


def test_failed_v3_channel_keeps_v1_v2_and_retry_source_id(tmp_path, monkeypatch):
    paths = channels(tmp_path, monkeypatch)
    broken = tmp_path / "missing" / "v3.json"
    monkeypatch.setenv("QEXP_PROGRESS_V3_PATH", str(broken))
    monkeypatch.setattr(progress, "_INITIAL_RETRY_DELAY_SECONDS", 0.01)
    # OutputState defaults are frozen at definition; set the actual state's retry after first offer.
    assert progress.update(stage="train", current=1, overall=progress.Counter(current=1, unit="step"))
    deadline = time.monotonic() + 2
    while not paths[2].exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert read_advisory_snapshot(paths[1])["current"] == 1
    assert read_advisory_snapshot(paths[2])["current"] == 1
    broken.parent.mkdir()
    progress.flush(timeout=1)
    assert read_advisory_snapshot(broken)["update_id"] == read_advisory_snapshot(paths[1])["update_id"]


def test_valid_input_no_channel_is_not_accepted():
    assert progress.validate_update(stage="working") == ()
    assert progress.update(stage="working") is False


def test_metrics_only_offer_is_not_deduplicated_by_v3_writer(tmp_path):
    path = tmp_path / "v3.json"
    reporter = progress._Reporter(path_v3=path, interval_seconds=1)
    try:
        assert reporter.update(stage="train", metrics={"loss": 0.8})
        deadline = time.monotonic() + 2
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(0.005)
        first = read_advisory_snapshot(path)
        assert first["metrics"] == {"loss": 0.8}
        assert reporter.update(stage="train", metrics={"loss": 0.4})
        reporter.close(timeout=1)
        second = read_advisory_snapshot(path)
        assert second["metrics"] == {"loss": 0.4}
        assert second["update_id"] != first["update_id"]
    finally:
        reporter.close(timeout=1)
