"""Deterministic projector policy tests; no timers, agents or GPUs required."""

import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp.runtime import progress as runtime
from qqtools.qexp._progress_protocol import read_advisory_snapshot, replace_advisory_snapshot


@pytest.fixture
def channel(tmp_path):
    cfg = SimpleNamespace(runtime_root=tmp_path / "rt", shared_root=tmp_path / "shared", machine_name="m1")
    task = SimpleNamespace(task_id="task")
    attempt = SimpleNamespace(attempt_id="a1", attempt_number=1, authorization={"launch_id": "launch-1"})
    path = runtime.prepare_progress_channel(cfg, task, attempt, wrapper_start_time_ticks=100)
    assert path is not None
    clock = [0.0]
    flags = {"fencing_token": 1, "terminal": False, "retired": False}
    def resolve(cfg, context):
        if flags["retired"]:
            return None
        return {**context, "fencing_token": flags["fencing_token"], "terminal": flags["terminal"]}
    def wall():
        return (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=clock[0])).isoformat()
    def projector():
        return runtime.ProgressProjector(cfg, clock=lambda: clock[0], wall_clock=wall, resolver=resolve)
    def send(update_id, *, stage="train", current=1, total=10, message=None):
        replace_advisory_snapshot(runtime.local_progress_path(cfg.runtime_root, "a1"), {
            "protocol_version": 1, "update_id": update_id, "stage": stage,
            "current": current, "total": total, "unit": "step", "message": message,
        })
    shared = runtime.shared_progress_path(cfg.shared_root, "task", "a1")
    return SimpleNamespace(cfg=cfg, clock=clock, flags=flags, projector=projector, send=send, shared=shared)


def test_current_snapshot_and_restart_dedupe(channel):
    c = channel
    c.send("one")
    c.projector().observe("a1")
    first = read_advisory_snapshot(c.shared)
    c.clock[0] = 100
    c.projector().observe("a1")
    assert read_advisory_snapshot(c.shared) == first
    assert first["fencing_token"] == 1
    assert first["sequence"] == 1


def test_message_reports_without_advancing(channel):
    c = channel
    p = c.projector()
    c.send("one", message="a")
    p.observe("a1")
    first = read_advisory_snapshot(c.shared)
    c.clock[0] = 5
    c.send("two", message="b")
    p.observe("a1")
    second = read_advisory_snapshot(c.shared)
    assert second["reported_at"] != first["reported_at"]
    assert second["advanced_at"] == first["advanced_at"]
    c.clock[0] = 10
    c.send("three", current=2)
    p.observe("a1")
    assert read_advisory_snapshot(c.shared)["advanced_at"] != first["advanced_at"]


def test_shared_writes_are_coalesced_and_stage_bursts_bounded(channel, monkeypatch):
    c = channel
    p = c.projector()
    writes = []
    original = runtime.replace_advisory_snapshot
    def record(path, value, **kwargs):
        if path == c.shared:
            writes.append(c.clock[0])
        return original(path, value, **kwargs)
    monkeypatch.setattr(runtime, "replace_advisory_snapshot", record)
    for i in range(101):
        c.clock[0] = i / 20
        c.send(f"update-{i}", current=i, total=200)
        p.observe("a1")
    assert writes == [0, 5]
    writes.clear()
    for i in range(1, 101):
        c.clock[0] = 5 + i / 20
        c.send(f"stage-{i}", stage=f"stage-{i}")
        p.observe("a1")
    assert len(writes) <= 5
    assert all(b - a >= 1 for a, b in zip(writes, writes[1:]))


def test_stage_change_does_not_wait_normal_five_seconds(channel):
    c = channel
    p = c.projector()
    c.send("one")
    p.observe("a1")
    c.clock[0] = 1
    c.send("two", stage="validation", current=None, total=None)
    p.observe("a1")
    assert read_advisory_snapshot(c.shared)["progress"]["stage"] == "validation"


def test_fencing_recovery_preserves_freshness(channel):
    c = channel
    c.send("one")
    c.projector().observe("a1")
    first = read_advisory_snapshot(c.shared)
    c.clock[0] = 100
    c.flags["fencing_token"] = 2
    c.projector().observe("a1")
    second = read_advisory_snapshot(c.shared)
    assert second["fencing_token"] == 2
    assert second["reported_at"] == first["reported_at"]
    assert second["advanced_at"] == first["advanced_at"]
    assert second["sequence"] == first["sequence"]


def test_superseded_attempt_does_not_publish(channel):
    c = channel
    p = c.projector()
    c.send("one")
    p.observe("a1")
    first = read_advisory_snapshot(c.shared)
    c.flags["retired"] = True
    c.clock[0] = 10
    c.send("old-writer", current=9)
    p.observe("a1")
    assert read_advisory_snapshot(c.shared) == first
    assert not (c.cfg.runtime_root / "progress-contexts" / "a1.json").exists()


def test_fast_terminal_keeps_final_snapshot_without_process_records(channel):
    c = channel
    c.send("one", current=10)
    c.flags["terminal"] = True
    c.projector().tick()
    assert read_advisory_snapshot(c.shared)["progress"]["current"] == 10
    for name in runtime._LOCAL_DIRS:
        assert list((c.cfg.runtime_root / name).iterdir()) == []


def test_unintegrated_terminal_has_no_fabricated_progress(channel):
    c = channel
    c.flags["terminal"] = True
    c.projector().observe("a1")
    assert not c.shared.exists()


def test_projection_failure_recovers_latest_state(channel, monkeypatch):
    c = channel
    p = c.projector()
    c.send("one")
    p.observe("a1")
    original = runtime.replace_advisory_snapshot
    def fail_shared(path, *args, **kwargs):
        if path == c.shared:
            raise OSError("shared filesystem unavailable")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(runtime, "replace_advisory_snapshot", fail_shared)
    c.clock[0] = 5
    c.send("two", current=2)
    p.observe("a1")
    assert read_advisory_snapshot(c.shared)["progress"]["current"] == 1
    monkeypatch.setattr(runtime, "replace_advisory_snapshot", original)
    c.clock[0] = 10
    p = c.projector()
    p.observe("a1")
    assert read_advisory_snapshot(c.shared)["progress"]["current"] == 2


def test_bad_payload_preserves_last_good_value_and_bounded_diagnostic(channel):
    c = channel
    p = c.projector()
    c.send("one")
    p.observe("a1")
    path = runtime.local_progress_path(c.cfg.runtime_root, "a1")
    path.write_bytes(b"x" * 10000)
    for _ in range(100):
        p.observe("a1")
    assert read_advisory_snapshot(c.shared)["progress"]["current"] == 1
    assert len(list((c.cfg.runtime_root / "progress-diagnostics").iterdir())) == 1


def test_cleanup_and_no_recreation(channel):
    c = channel
    c.send("one")
    p = c.projector()
    p.observe("a1")
    runtime.cleanup_local_progress(c.cfg, "task", {"a1"})
    runtime.cleanup_shared_progress(c.cfg, "task")
    p.tick()
    assert not c.shared.parent.exists()
    for name in runtime._LOCAL_DIRS:
        assert list((c.cfg.runtime_root / name).iterdir()) == []


def test_zero_total_and_missing_progress_formatting():
    assert runtime.progress_details({}) == (("Progress", "unavailable"),)
    details = dict(runtime.progress_details({"progress": {
        "status": "available", "reported_at": "bad", "advanced_at": "bad",
        "progress": {"stage": "train", "current": 0, "total": 0, "unit": "step", "message": None},
    }}))
    assert details["Progress"] == "0/0 step"
    assert details["Progress reported"] == "unknown"


def test_cleanup_race_cannot_resurrect_local_observation(channel, monkeypatch):
    c = channel
    c.send("one")
    original = runtime.replace_advisory_snapshot
    observed = c.cfg.runtime_root / "progress-observed" / "a1.json"
    context = c.cfg.runtime_root / "progress-contexts" / "a1.json"
    def racing_write(path, value, **kwargs):
        if path == observed:
            context.unlink()
        return original(path, value, **kwargs)
    monkeypatch.setattr(runtime, "replace_advisory_snapshot", racing_write)
    c.projector().observe("a1")
    assert not observed.exists()
    assert not c.shared.exists()


@pytest.mark.parametrize("field,value", [("protocol_version", True), ("reported_at", None), ("status", "available")])
def test_corrupt_restore_does_not_prevent_new_report(channel, field, value):
    c = channel
    c.send("one")
    c.projector().observe("a1")
    local = c.cfg.runtime_root / "progress-observed" / "a1.json"
    bad = read_advisory_snapshot(local)
    bad[field] = value
    replace_advisory_snapshot(local, bad)
    c.clock[0] = 5
    c.send("two", current=2)
    c.projector().observe("a1")
    assert read_advisory_snapshot(c.shared)["progress"]["current"] == 2
