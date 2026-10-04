from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

import pytest

from qqtools.plugins.qexp.runtime.group_discovery import activation, activation_submission
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.upgrade import UpgradeCoordinator, upgrade_journal_path
from tests.helpers.qexp_discovery import isolated_group, source_file

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _prepared(tmp_path, sources=4, large=False):
    cfg = isolated_group(tmp_path, tail=0)
    directory = cfg.shared_root / "operations/submissions"
    for index in range(sources):
        path = source_file(directory / f"op-{index}.json", operation=f"op-{index}")
        if large:
            value = json.loads(path.read_text())
            value["submission"]["specifications"] = "x" * (96 * 1024)
            path.write_text(json.dumps(value))
    # Set up a real post-fence activation and its audited coordinator checkpoint.
    for _ in range(3):
        record = activation.advance_group_service_activation(cfg)
        if record["state"] == "building":
            break
    assert record["state"] == "building"
    coordinator = UpgradeCoordinator(cfg)
    coordinator.discover()
    journal = read_json(upgrade_journal_path(cfg))
    item = journal["upgrade"]["migrations"]["group-service-v1"]
    item.update(phase="activation", phase_index=2, audit_passed=True)
    atomic_replace(upgrade_journal_path(cfg), journal)
    return cfg


def _advance(cfg):
    status = UpgradeCoordinator(cfg).advance(force_retry=True)
    assert status["state"] != "repair_required", status["blockers"]
    item = next(item for item in status["migrations"] if item["name"] == "group-service-v1")
    usage = item.get("last_slice_usage", {})
    assert usage.get("records", 0) <= 1
    assert usage.get("metadata_ops", 0) <= 128
    assert usage.get("io_bytes", 0) <= 256 * 1024
    return status


def _until(cfg, predicate, limit=400):
    for _ in range(limit):
        status = _advance(cfg)
        if predicate(status):
            return status
    raise AssertionError(f"upgrade failed to reach expected progress: {status}")


def _stage(status, name):
    value = status.get("progress")
    return value if value and value["stage"] == name else None


def test_fixed_sources_inventory_counts_successful_entries_once_across_restart(tmp_path):
    cfg = _prepared(tmp_path, sources=6)
    directory = cfg.shared_root / "operations/submissions"
    (directory / ".metadata").write_text("not a source")
    snapshots = []
    for _ in range(200):
        status = _advance(cfg)  # Every call constructs a new coordinator.
        progress = _stage(status, "submissions")
        if progress:
            snapshots.append(progress)
        if not status["pending"]:
            break
    else:
        raise AssertionError("activation did not complete")
    assert snapshots
    assert any(item["inventoried_units"] != item["completed_units"] for item in snapshots)
    assert any(item["total_units"] == item["completed_units"] == 6 for item in snapshots)
    assert all(item["completed_units"] is None or item["completed_units"] <= 6 for item in snapshots)
    assert all(item["scope"] == "shared_project" for item in snapshots)
    exact = [item for item in snapshots if item["total_kind"] == "snapshot_exact"]
    assert exact and all(item["total_units"] == 6 for item in exact)
    assert activation.read_group_service_activation_record(cfg.shared_root)["revision"] > 6


def test_empty_and_missing_optional_namespaces_publish_exact_zero(tmp_path):
    cfg = _prepared(tmp_path, sources=0)
    optional = cfg.shared_root / "operations/group-discovery/active"
    if optional.exists():
        optional.rmdir()
    zeros = set()
    for _ in range(100):
        status = _advance(cfg)
        progress = status.get("progress")
        if progress and progress["total_kind"] == "snapshot_exact" and progress["total_units"] == 0:
            assert progress["completed_units"] == 0
            zeros.add(progress["stage"])
        if not status["pending"]:
            break
    assert not status["pending"]
    assert {"submissions", "submission_control", "group_control", "cleanup", "discovery_debt"} <= zeros


@pytest.mark.parametrize("change", ["add", "delete", "replace"])
def test_directory_change_restarts_epoch_without_inflating_completed_count(tmp_path, change):
    cfg = _prepared(tmp_path, sources=8)
    first = _until(cfg, lambda status: (p := _stage(status, "submissions")) and (p["completed_units"] or 0) >= 2)
    epoch = first["progress"]["scan_epoch"]
    directory = cfg.shared_root / "operations/submissions"
    if change == "add":
        source_file(directory / "added.json", operation="added")
        expected = 9
    elif change == "delete":
        (directory / "op-0.json").unlink()
        expected = 7
    else:
        directory.rename(directory.with_name("old-submissions"))
        directory.mkdir()
        for index in range(3):
            source_file(directory / f"new-{index}.json", operation=f"new-{index}")
        expected = 3
    seen = []
    for _ in range(100):
        status = _advance(cfg)
        if progress := _stage(status, "submissions"):
            seen.append(progress)
            assert progress["scan_epoch"] > epoch
            assert progress["completed_units"] is None or progress["completed_units"] <= expected
        if not status["pending"]:
            break
    assert seen
    assert any(p["total_units"] == p["completed_units"] == expected for p in seen)


def test_inventory_recovery_reconstructs_from_captured_cookie_without_freezing_activation(tmp_path):
    cfg = _prepared(tmp_path, sources=12)
    _until(cfg, lambda status: (p := _stage(status, "submissions")) and (p["completed_units"] or 0) >= 3)
    journal = read_json(upgrade_journal_path(cfg))
    item = journal["upgrade"]["migrations"]["group-service-v1"]
    for key in ("progress", "progress_evidence", "progress_state"):
        item.pop(key, None)
    atomic_replace(upgrade_journal_path(cfg), journal)
    assert UpgradeCoordinator(cfg).status()["progress"] is None
    snapshots = []
    unchanged_turns = 0
    previous = activation.read_group_service_activation_record(cfg.shared_root)["revision"]
    for _ in range(100):
        status = _advance(cfg)
        record = activation.read_group_service_activation_record(cfg.shared_root)
        unchanged_turns = unchanged_turns + 1 if record["revision"] == previous else 0
        assert unchanged_turns <= 1
        previous = record["revision"]
        if progress := _stage(status, "submissions"):
            snapshots.append(progress)
        if not status["pending"]:
            break
    assert any(p["completed_units"] is None for p in snapshots)
    known = [p["completed_units"] for p in snapshots if p["completed_units"] is not None]
    assert known and all(3 <= value <= 12 for value in known)
    assert not status["pending"]


@pytest.mark.parametrize("field", ["version", "completed_units", "target_ordinal"])
def test_malformed_inventory_state_is_discarded_without_resetting_activation(tmp_path, field):
    cfg = _prepared(tmp_path, sources=8)
    _until(cfg, lambda status: (p := _stage(status, "submissions")) and (p["completed_units"] or 0) >= 2)
    journal = read_json(upgrade_journal_path(cfg))
    state = journal["upgrade"]["migrations"]["group-service-v1"]["progress_state"]
    state[field] = True if field == "version" else 1000
    atomic_replace(upgrade_journal_path(cfg), journal)
    before = activation.read_group_service_activation_record(cfg.shared_root)
    status = _advance(cfg)
    after = activation.read_group_service_activation_record(cfg.shared_root)
    assert after["revision"] > before["revision"]
    assert after["activation_epoch"] == before["activation_epoch"]
    assert status["progress"]["completed_units"] is None
    assert not _until(cfg, lambda result: not result["pending"])["pending"]


def test_continuous_source_churn_never_schedules_two_inventory_only_slices(tmp_path):
    cfg = _prepared(tmp_path, sources=8)
    _until(cfg, lambda status: _stage(status, "submissions"))
    previous = activation.read_group_service_activation_record(cfg.shared_root)["revision"]
    consecutive_inventory = 0
    advances = 0
    for index in range(20):
        source_file(cfg.shared_root / f"operations/submissions/new-{index}.json", operation=f"new-{index}")
        status = _advance(cfg)
        revision = activation.read_group_service_activation_record(cfg.shared_root)["revision"]
        if revision == previous:
            consecutive_inventory += 1
        else:
            advances += 1
            consecutive_inventory = 0
        assert consecutive_inventory <= 1
        if progress := _stage(status, "submissions"):
            assert progress["total_kind"] != "snapshot_exact"
        previous = revision
    assert advances >= 10


def test_large_submission_reports_byte_checkpoints_then_one_source(tmp_path):
    cfg = _prepared(tmp_path, sources=1, large=True)
    positions = []
    completed = []
    for _ in range(100):
        status = _advance(cfg)
        if progress := _stage(status, "submissions"):
            if item := progress.get("current_item"):
                positions.append(item["completed_bytes"])
                delta = status["invocation"]["delta"]
                assert delta is not None and delta["kind"] == "bytes"
                assert delta["after"] == item["completed_bytes"]
                assert delta["after"] - delta["before"] == 16 * 1024
                assert item["total_bytes"] > 64 * 1024
                assert item["id"] == "op-0"
                assert progress["completed_units"] in {None, 0}
            completed.append(progress["completed_units"])
        if not status["pending"]:
            break
    assert len(set(positions)) >= 4
    assert positions == sorted(positions)
    assert all(value % (16 * 1024) == 0 for value in positions)
    assert 1 in completed
    assert all(value in {None, 0, 1} for value in completed)


def test_large_source_replacement_discards_byte_checkpoint(tmp_path):
    cfg = _prepared(tmp_path, sources=1, large=True)
    _until(
        cfg,
        lambda status: (
            (p := _stage(status, "submissions")) and p.get("current_item", {}).get("completed_bytes", 0) >= 32768
        ),
    )
    path = cfg.shared_root / "operations/submissions/op-0.json"
    # In-place content change keeps the directory revision unchanged.
    value = json.loads(path.read_text())
    value["submission"]["specifications"] = "y" * (100 * 1024)
    path.write_text(json.dumps(value))
    restarted = _until(
        cfg, lambda status: (p := _stage(status, "submissions")) and p.get("current_item", {}).get("restarted")
    )
    assert restarted["progress"]["current_item"]["completed_bytes"] == 0
    assert restarted["progress"]["completed_units"] in {None, 0}
    assert restarted["invocation"]["delta"] is None


@pytest.mark.parametrize("cut", ["activation", "checkpoint"])
def test_crash_after_authoritative_commit_recovers_without_double_counting(tmp_path, monkeypatch, cut):
    cfg = _prepared(tmp_path, sources=5, large=cut == "checkpoint")
    _until(cfg, lambda status: _stage(status, "submissions"))

    class Crash(BaseException):
        pass

    original = activation._write_record

    def crash_after_commit(root, record, **kwargs):
        result = original(root, record, **kwargs)
        if record["bootstrap"]["phase"] == "submissions":
            raise Crash
        return result

    with monkeypatch.context() as patch:
        if cut == "activation":
            patch.setattr(activation, "_write_record", crash_after_commit)
        else:
            persist_checkpoint = activation_submission._persist_checkpoint

            def crash_after_checkpoint(*args, **kwargs):
                persist_checkpoint(*args, **kwargs)
                raise Crash

            patch.setattr(activation_submission, "_persist_checkpoint", crash_after_checkpoint)
        with pytest.raises(Crash):
            for _ in range(3):
                _advance(cfg)
    assert UpgradeCoordinator(cfg).status()["progress"] is None
    for _ in range(160):
        status = _advance(cfg)
        if progress := _stage(status, "submissions"):
            assert progress["completed_units"] is None or progress["completed_units"] <= 5
        if not status["pending"]:
            break
    assert not status["pending"]


def test_large_inventory_has_constant_slice_cost_and_journal_size(tmp_path, monkeypatch, capsys):
    cfg = _prepared(tmp_path, sources=1000)
    _until(cfg, lambda status: _stage(status, "submissions"))
    reads = []
    original = activation.read_directory_entry

    def count_read(path, cookie):
        reads.append((path, cookie))
        return original(path, cookie)

    monkeypatch.setattr(activation, "read_directory_entry", count_read)
    from qqtools.plugins.qexp.runtime.upgrade import group_service_progress

    if hasattr(group_service_progress, "read_directory_entry"):
        monkeypatch.setattr(group_service_progress, "read_directory_entry", count_read)
    start = time.monotonic()
    sizes = []
    for _ in range(24):
        before = len(reads)
        _advance(cfg)
        assert len(reads) - before <= 1
        sizes.append(upgrade_journal_path(cfg).stat().st_size)
    elapsed = time.monotonic() - start
    assert max(sizes) < 32 * 1024
    assert max(sizes) - min(sizes) < 4096
    with capsys.disabled():
        print(
            f"\nprogress cost: 1000 sources, 24 slices, {len(reads)} directory reads, {elapsed:.3f}s, journal <= {max(sizes)} bytes (local test filesystem; fsync disabled)"
        )


def test_status_before_discovery_does_not_enumerate_history(tmp_path, monkeypatch):
    cfg = isolated_group(tmp_path, tail=0)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("status enumerated a source namespace")

    with monkeypatch.context() as patch:
        patch.setattr("os.scandir", forbidden)
        patch.setattr("os.listdir", forbidden)
        patch.setattr(activation, "read_directory_entry", forbidden)
        status = UpgradeCoordinator(cfg).status()
    assert status["state"] == "idle"
    assert "group-service-v1" in {item["name"] for item in status.get("available_migrations", [])}


def test_direct_activation_invalidates_existing_journal_observation(tmp_path):
    cfg = _prepared(tmp_path, sources=8)
    _until(cfg, lambda status: (p := _stage(status, "submissions")) and p["completed_units"] == 2)
    activation.advance_group_service_activation(cfg)
    status = UpgradeCoordinator(cfg).status()
    assert status["progress"] is None
    assert status["state"] == "runnable"


def test_stable_pass_restart_uses_new_epoch_and_recounts(tmp_path):
    cfg = _prepared(tmp_path, sources=3)
    before = _until(cfg, lambda status: _stage(status, "stable_pass"))
    source_file(cfg.shared_root / "operations/submissions/late.json", operation="late")
    after = _until(cfg, lambda status: _stage(status, "submissions"))
    assert after["progress"]["scan_epoch"] > before["progress"]["scan_epoch"]
    assert after["progress"]["total_kind"] != "snapshot_exact"
    final = _until(
        cfg, lambda status: (p := _stage(status, "submissions")) and p["completed_units"] == p["total_units"] == 4
    )
    assert final["progress"]["scope"] == "shared_project"


def test_unreachable_reconstruction_cookie_keeps_count_unavailable(tmp_path, monkeypatch):
    cfg = _prepared(tmp_path, sources=16)
    _until(cfg, lambda status: (p := _stage(status, "submissions")) and p["completed_units"] == 4)
    journal = read_json(upgrade_journal_path(cfg))
    item = journal["upgrade"]["migrations"]["group-service-v1"]
    for key in ("progress", "progress_evidence", "progress_state"):
        item.pop(key, None)
    atomic_replace(upgrade_journal_path(cfg), journal)
    from qqtools.plugins.qexp.runtime.upgrade import group_service_progress

    monkeypatch.setattr(group_service_progress, "read_directory_entry", lambda path, cookie: (None, cookie))
    before = activation.read_group_service_activation_record(cfg.shared_root)["revision"]
    for _ in range(8):
        status = _advance(cfg)
        if progress := _stage(status, "submissions"):
            assert progress["completed_units"] is None
    assert activation.read_group_service_activation_record(cfg.shared_root)["revision"] > before
    assert status["state"] == "runnable"


def test_inventory_does_not_refresh_semantic_timestamp(tmp_path, monkeypatch):
    cfg = _prepared(tmp_path, sources=8)
    previous = _until(cfg, lambda status: (p := _stage(status, "submissions")) and p["completed_units"] == 1)
    tick = 0

    def now():
        nonlocal tick
        tick += 1
        return (datetime(2026, 10, 5, tzinfo=timezone.utc) + timedelta(seconds=tick)).isoformat()

    monkeypatch.setattr(activation, "_utc_now", now)
    inventory_turns = 0
    for _ in range(10):
        revision = activation.read_group_service_activation_record(cfg.shared_root)["revision"]
        current = _advance(cfg)
        if activation.read_group_service_activation_record(cfg.shared_root)["revision"] == revision:
            inventory_turns += 1
            assert current["progress"]["last_progress_at"] == previous["progress"]["last_progress_at"]
        previous = current
    assert inventory_turns >= 4


def test_observation_io_failure_cannot_starve_authoritative_activation(tmp_path, monkeypatch):
    cfg = _prepared(tmp_path, sources=8)
    _until(cfg, lambda status: _stage(status, "submissions"))
    from qqtools.plugins.qexp.runtime.upgrade import group_service_progress

    def failed_observation(*_args):
        raise OSError("observation probe unavailable")

    monkeypatch.setattr(group_service_progress, "_metadata_revision", failed_observation)
    before = activation.read_group_service_activation_record(cfg.shared_root)["revision"]
    for _ in range(4):
        result = _advance(cfg)
        assert result["progress"] is None
        assert not result["admission_blocked"]
    assert activation.read_group_service_activation_record(cfg.shared_root)["revision"] == before + 4


def test_bad_source_still_requires_repair_with_progress_enabled(tmp_path):
    cfg = _prepared(tmp_path, sources=1, large=True)
    path = cfg.shared_root / "operations/submissions/op-0.json"
    path.write_bytes(b'{"submission":invalid' + b" " * (70 * 1024))
    for _ in range(40):
        status = UpgradeCoordinator(cfg).advance(force_retry=True)
        if status["state"] == "repair_required":
            break
    assert status["state"] == "repair_required"
    assert status["blockers"]
    assert status["progress"] is None
