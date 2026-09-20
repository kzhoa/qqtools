"""Actual checkpointed source output feeds the bounded membership decoder."""

from __future__ import annotations

import json

import pytest

from qqtools.plugins.qexp.runtime.group_discovery.driver import ProjectionDriver
from qqtools.plugins.qexp.runtime.group_discovery.event_decoder import EventDecoder
from qqtools.plugins.qexp.runtime.group_discovery.slice_io import SliceIO
from qqtools.plugins.qexp.runtime.group_discovery.submission_projection import FieldChunk, FieldEnd

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("reverse", [False, True])
def test_driver_checkpointed_fields_decode_in_source_order(tmp_path, reverse):
    selected = {
        "resolved_context": {"task_ids": ["task-a", "task-b"]},
        "commit_plan": {"group_membership_sequences": [7, 8]},
    }
    if reverse:
        selected = dict(reversed(list(selected.items())))
    document = {
        "meta": {"schema_version": 6},
        "submission": {
            "operation_id": "op-1",
            "target_group": "exp",
            "state": "committed",
            **selected,
        },
    }
    source = tmp_path / "op-1.json"
    # Escaped ASCII IDs exercise normalized decoded bytes, rather than raw JSON equality.
    source.write_text(json.dumps(document).replace("task-a", "task-\\u0061"))
    scratch = tmp_path / "projection"
    driver = ProjectionDriver(source, scratch, "op-1", "exp")
    try:
        for _ in range(2000):
            driver.advance(SliceIO(), max_processed_bytes=5)
            if driver.is_complete:
                break
        assert driver.is_complete
        summary = driver.summary
        checkpoint = json.loads((scratch / "checkpoint.json").read_bytes())
        decoder = EventDecoder()
        values = {}
        ends = []
        with (scratch / "events.jsonl").open("rb") as spool:
            while data := spool.read(17):
                decoder.feed(data)
                while (item := decoder.pop()) is not None:
                    event = item.event
                    if isinstance(event, FieldChunk):
                        key = event.kind, event.ordinal
                        values[key] = values.get(key, b"") + event.data
                    elif isinstance(event, FieldEnd):
                        ends.append(event)
        decoder.finish(
            expected_events=checkpoint["spool"]["events"],
            task_count=summary.task_count,
            sequence_count=summary.sequence_count,
        )
        assert values == {
            ("task_id", 0): b"task-a",
            ("task_id", 1): b"task-b",
            ("sequence", 0): b"7",
            ("sequence", 1): b"8",
        }
        assert len(ends) == 4
    finally:
        driver.request_close()
        for _ in range(100):
            if driver.advance(SliceIO()).state == "closed":
                break
        assert driver.is_closed
