"""Projection spool validation preserves positional and byte-level evidence."""

from __future__ import annotations

import base64
import json

import pytest

from qqtools.plugins.qexp.runtime.group_discovery.event_decoder import EventDecoder
from qqtools.plugins.qexp.runtime.group_discovery.fingerprint import ChainedDigest
from qqtools.plugins.qexp.runtime.group_discovery.submission_projection import FieldChunk, FieldEnd


def field(kind, ordinal, data, *, start=0, chunks=None):
    digest = ChainedDigest()
    digest.update(data)
    pieces = chunks if chunks is not None else [data]
    events = [
        {
            "type": "chunk",
            "kind": kind,
            "ordinal": ordinal,
            "data_b64": base64.b64encode(piece).decode(),
            "is_final": i == len(pieces) - 1,
        }
        for i, piece in enumerate(pieces)
    ]
    events.append(
        {
            "type": "end",
            "kind": kind,
            "ordinal": ordinal,
            "digest": digest.hexdigest(),
            "decoded_size": len(data),
            "start": start,
            "end": start + len(data) + 2,
        }
    )
    return events


def encoded(events):
    return b"".join(json.dumps(event, separators=(",", ":")).encode() + b"\n" for event in events)


def decode(events, *, fragment=65536, tasks=1, sequences=1):
    payload = encoded(events)
    decoder = EventDecoder()
    result = []
    for offset in range(0, len(payload), fragment):
        decoder.feed(payload[offset : offset + fragment])
        while (item := decoder.pop()) is not None:
            result.append(item)
    decoder.finish(expected_events=len(events), task_count=tasks, sequence_count=sequences)
    return result


@pytest.mark.parametrize("fragment", [1, 7, 65536])
@pytest.mark.parametrize("reverse", [False, True])
def test_arrays_pair_by_ordinal_in_either_source_order(fragment, reverse):
    kinds = ["task_id", "sequence"]
    if reverse:
        kinds.reverse()
    events = []
    for index, kind in enumerate(kinds):
        for ordinal in range(2):
            value = f"task-{ordinal}".encode() if kind == "task_id" else str(ordinal + 1).encode()
            events.extend(field(kind, ordinal, value, start=index * 100 + ordinal * 20))
    result = decode(events, fragment=fragment, tasks=2, sequences=2)
    ends = [item for item in result if isinstance(item.event, FieldEnd)]
    assert [(item.event.kind, item.event.ordinal) for item in ends] == [
        (kind, ordinal) for kind in kinds for ordinal in range(2)
    ]
    payload = encoded(events)
    assert result[0].start == 0
    assert result[-1].end == len(payload)
    for left, right in zip(result, result[1:]):
        assert left.end == right.start
    for index in range(0, len(result), 2):
        assert result[index + 1].field_start == result[index].start


def test_large_scalar_is_streamed_without_integer_conversion():
    task = b"t" * 160000
    sequence = b"9" * 70000
    events = field("task_id", 0, task, chunks=[task[i : i + 65536] for i in range(0, len(task), 65536)])
    events += field("sequence", 0, sequence, start=200000, chunks=[sequence[:65536], sequence[65536:]])
    result = decode(events)
    assert sum(len(item.event.data) for item in result if isinstance(item.event, FieldChunk)) == 230000
    assert [item.event.decoded_size for item in result if isinstance(item.event, FieldEnd)] == [160000, 70000]


@pytest.mark.parametrize(
    "change",
    [
        lambda events: events[0].update(ordinal=True),
        lambda events: events[0].update(ordinal=1),
        lambda events: events[0].update(data_b64="dGFzaw==\n"),
        lambda events: events[0].update(is_final=False),
        lambda events: events[1].update(digest="0" * 64),
        lambda events: events[1].update(decoded_size=3),
        lambda events: events[1].update(start=True),
        lambda events: events[1].update(end=0),
        lambda events: events[1].update(extra="ignored"),
        lambda events: events.reverse(),
    ],
)
def test_malformed_spool_cannot_supply_field_evidence(change):
    events = field("task_id", 0, b"task")
    change(events)
    with pytest.raises(ValueError):
        decode(events, tasks=1, sequences=0)


@pytest.mark.parametrize(
    "kind,data",
    [
        ("task_id", b""),
        ("task_id", b"bad/name"),
        ("sequence", b"0"),
        ("sequence", b"01"),
        ("sequence", b"1e2"),
        ("sequence", b"-1"),
    ],
)
def test_invalid_decoded_identifier_or_sequence(kind, data):
    with pytest.raises(ValueError):
        decode(field(kind, 0, data), tasks=int(kind == "task_id"), sequences=int(kind == "sequence"))


@pytest.mark.parametrize("data", [b".", b".."])
def test_short_identifiers_preserve_existing_record_contract(data):
    result = decode(field("task_id", 0, data), tasks=1, sequences=0)
    assert result[0].event.data == data


def test_duplicate_keys_poison_decoder_and_discard_queued_results():
    decoder = EventDecoder()
    valid = encoded(field("task_id", 0, b"task"))
    malformed = b'{"type":"chunk","type":"end"}\n'
    with pytest.raises(ValueError):
        decoder.feed(valid + malformed)
    for action in (
        decoder.pop,
        lambda: decoder.feed(b""),
        lambda: decoder.finish(expected_events=2, task_count=1, sequence_count=0),
    ):
        with pytest.raises(ValueError):
            action()


def test_unfinished_line_and_incorrect_totals_never_finish():
    decoder = EventDecoder()
    decoder.feed(b'{"type":')
    with pytest.raises(ValueError):
        decoder.finish(expected_events=0, task_count=0, sequence_count=0)
    decoder = EventDecoder()
    decoder.feed(encoded(field("task_id", 0, b"task")))
    while decoder.pop() is not None:
        pass
    with pytest.raises(ValueError):
        decoder.finish(expected_events=2, task_count=2, sequence_count=0)


def test_bounded_input_and_output_backpressure():
    decoder = EventDecoder()
    with pytest.raises(ValueError):
        decoder.feed(b" " * 65537)
    decoder = EventDecoder()
    decoder.feed(encoded(field("task_id", 0, b"task")))
    with pytest.raises(RuntimeError):
        decoder.feed(b"\n")


@pytest.mark.parametrize("case", ["return_to_kind", "interleave", "overlap", "backward"])
def test_field_order_and_source_spans_cannot_conflict(case):
    first = field("task_id", 0, b"task-a", start=10)
    second = field("sequence", 0, b"1", start=30)
    if case == "return_to_kind":
        events = first + second + field("task_id", 1, b"task-b", start=50)
    elif case == "interleave":
        events = [first[0], *second, first[1]]
    else:
        second[-1].update(start=11 if case == "overlap" else 0, end=14 if case == "overlap" else 3)
        events = first + second
    with pytest.raises(ValueError):
        decode(events)


def test_multichunk_field_start_points_to_first_chunk_across_feeds():
    result = decode(
        field("task_id", 0, b"task-a", chunks=[b"ta", b"sk", b"-a"]),
        tasks=1,
        sequences=0,
        fragment=7,
    )
    assert len(result) == 4
    assert all(item.field_start == 0 for item in result)
    assert result[-1].start > result[0].end


def test_oversized_chunk_and_unterminated_line_are_rejected():
    with pytest.raises(ValueError):
        decode(field("task_id", 0, b"a" * 65537), tasks=1, sequences=0)
    decoder = EventDecoder()
    decoder.feed(b" " * 65536)
    with pytest.raises(ValueError, match="line"):
        decoder.feed(b" " * (90000 - 65536))
