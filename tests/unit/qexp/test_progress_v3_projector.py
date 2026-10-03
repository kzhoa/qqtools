"""V3 acceptance clocks and scope replacement share established cadence guarantees."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp.runtime import progress_v3 as runtime
from qqtools.qexp._progress_protocol import read_advisory_snapshot, replace_advisory_snapshot


@pytest.fixture
def channel(tmp_path):
    cfg = SimpleNamespace(runtime_root=tmp_path / "runtime", shared_root=tmp_path / "shared", machine_name="g1")
    task = SimpleNamespace(task_id="task-1")
    attempt = SimpleNamespace(
        task_id="task-1", attempt_id="attempt-1", attempt_number=1, authorization={"launch_id": "launch-1"}
    )
    mailbox = runtime.prepare_progress_v3_channel(cfg, task, attempt, wrapper_start_time_ticks=1, interval_seconds=1)
    assert mailbox is not None
    clock = [0.0]
    flags = {"terminal": False, "retired": False}

    def resolve(_cfg, context):
        return None if flags["retired"] else {**context, "fencing_token": 1, "terminal": flags["terminal"]}

    def projector():
        return runtime.ProgressV3Projector(
            cfg,
            registration_generation="generation-1",
            resolver=resolve,
            clock=lambda: clock[0],
            wall_clock=lambda: (datetime(2026, 10, 3, tzinfo=timezone.utc) + timedelta(seconds=clock[0])).isoformat(),
        )

    def send(update_id, *, current=1, label="Training", message=None, overall=True, metrics=None):
        replace_advisory_snapshot(
            Path(mailbox),
            {
                "protocol_version": 3,
                "update_id": update_id,
                "activity": {"stage": "validation", "current": 1, "total": 4, "unit": "batch", "message": message},
                "overall": {"current": current, "total": 10, "unit": "step", "label": label} if overall else None,
                "metrics": metrics or {},
                "completeness": {"complete": True, "omitted_metrics": 0, "reasons": []},
            },
        )

    return SimpleNamespace(
        cfg=cfg,
        clock=clock,
        flags=flags,
        projector=projector,
        send=send,
        shared=runtime._shared_path(cfg, task.task_id, attempt.attempt_id),
        mailbox=Path(mailbox),
    )


def test_overall_only_advances_but_label_message_metrics_do_not(channel):
    c = channel
    p = c.projector()
    c.send("one")
    p._observe("attempt-1")
    initial = read_advisory_snapshot(c.shared)
    for number, kwargs in enumerate(
        [
            {"label": "New label"},
            {"label": "New label", "message": "note"},
            {"label": "New label", "message": "note", "metrics": {"loss": 0.3}},
        ],
        2,
    ):
        c.clock[0] = max(p._entries["attempt-1"]["cache_next_due"], p._entries["attempt-1"]["shared_next_due"])
        c.send(f"update-{number}", **kwargs)
        p._observe("attempt-1")
        snapshot = read_advisory_snapshot(c.shared)
        assert snapshot["sequence"] == number
        assert snapshot["advanced_at"] == initial["advanced_at"]
    c.clock[0] = max(p._entries["attempt-1"]["cache_next_due"], p._entries["attempt-1"]["shared_next_due"])
    c.send("overall-changed", current=2)
    p._observe("attempt-1")
    assert read_advisory_snapshot(c.shared)["advanced_at"] != initial["advanced_at"]


def test_dedup_restart_and_explicit_clear(channel):
    c = channel
    p = c.projector()
    c.send("one")
    p._observe("attempt-1")
    initial = read_advisory_snapshot(c.shared)
    c.clock[0] = 10
    c.send("same-content-new-id")
    p._observe("attempt-1")
    assert read_advisory_snapshot(c.shared) == initial
    restarted = c.projector()
    restarted._observe("attempt-1")
    assert read_advisory_snapshot(c.shared) == initial
    c.clock[0] = 20
    c.send("cleared", overall=False)
    restarted._observe("attempt-1")
    assert read_advisory_snapshot(c.shared)["progress"]["overall"] is None


@pytest.mark.parametrize(
    "field,bad",
    [
        ("wrapper_pid", True),
        ("wrapper_start_time_ticks", -1),
        ("attempt_number", 0),
        ("protocol_version", True),
        ("extra", 1),
    ],
)
def test_context_roundtrip_rejects_invalid_identity(channel, field, bad):
    value = read_advisory_snapshot(runtime._context_path(channel.cfg, "attempt-1"))
    assert runtime._validate_context(value, "attempt-1") == value
    value[field] = bad
    with pytest.raises(ValueError):
        runtime._validate_context(value, "attempt-1")


def test_snapshot_roundtrip_and_strict_identity(channel):
    c = channel
    c.send("one")
    c.projector()._observe("attempt-1")
    snapshot = read_advisory_snapshot(c.shared)
    assert runtime._validate_projection_v3(snapshot, snapshot, require_token=True, require_generation=True) == snapshot
    bad = deepcopy(snapshot)
    bad["attempt_number"] = True
    with pytest.raises(ValueError):
        runtime._validate_projection_v3(bad, snapshot)
    bad = deepcopy(snapshot)
    bad["progress"]["overall"]["unknown"] = 1
    with pytest.raises(ValueError):
        runtime._validate_projection_v3(bad, snapshot)


def test_retirement_removes_context_and_mailbox_without_recreation(channel):
    c = channel
    c.send("one")
    p = c.projector()
    p._observe("attempt-1")
    c.flags["retired"] = True
    p._observe("attempt-1")
    assert not c.mailbox.exists()
    assert not runtime._context_path(c.cfg, "attempt-1").exists()
