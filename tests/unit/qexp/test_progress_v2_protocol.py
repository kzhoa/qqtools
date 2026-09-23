"""Strict, bounded framework-neutral metric protocol cases."""

import math

import pytest

from qqtools.qexp._progress_protocol_v2 import capture_metrics, read_payload_v2, validate_payload_v2


def _payload(metrics=None, completeness=None):
    return {
        "protocol_version": 2,
        "update_id": "publication-1",
        "stage": "custom-stage",
        "current": 7,
        "total": None,
        "unit": "batch",
        "message": "working",
        "metrics": {} if metrics is None else metrics,
        "completeness": {"complete": True, "omitted_metrics": 0, "reasons": []}
        if completeness is None
        else completeness,
    }


def test_capture_copies_only_first_32_built_in_scalars():
    offered = {f"metric_{index}": index for index in range(33)}
    captured, completeness = capture_metrics(offered)
    assert list(captured) == [f"metric_{index}" for index in range(32)]
    assert completeness == {"complete": False, "omitted_metrics": 1, "reasons": ["metric_limit"]}
    offered["metric_0"] = 999
    assert captured["metric_0"] == 0


def test_metric_only_errors_preserve_base_with_explicit_incompleteness():
    offered = {"okay": 1.25, "bool": True, "nan": math.nan, "nested": {"x": 1}, "bad name": 2}
    captured, completeness = capture_metrics(offered)
    assert captured == {"okay": 1.25}
    assert completeness == {"complete": False, "omitted_metrics": 4, "reasons": ["invalid_metrics"]}
    assert validate_payload_v2(_payload(captured, completeness))["metrics"] == captured


def test_invalid_container_never_invokes_custom_mapping_or_conversion():
    class HostileMapping:
        def __iter__(self):
            raise AssertionError("must not traverse")

    captured, completeness = capture_metrics(HostileMapping())
    assert captured == {}
    assert completeness == {"complete": False, "omitted_metrics": None, "reasons": ["invalid_metrics"]}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda item: item.update(extra=1),
        lambda item: item.update(protocol_version=True),
        lambda item: item.update(metrics={"bad name": 1}),
        lambda item: item.update(metrics={"tensor": object()}),
        lambda item: item.update(metrics={"nan": math.nan}),
        lambda item: item.update(completeness={"complete": True, "omitted_metrics": 1, "reasons": []}),
        lambda item: item.update(completeness={"complete": False, "omitted_metrics": 0, "reasons": []}),
        lambda item: item.update(completeness={"complete": False, "omitted_metrics": None, "reasons": ["size_limit"]}),
    ],
)
def test_wire_rejects_unknown_invalid_or_inconsistent_fields(mutation):
    payload = _payload()
    mutation(payload)
    with pytest.raises((TypeError, ValueError)):
        validate_payload_v2(payload)


def test_wire_accepts_maximum_finite_scalar_and_unknown_total():
    payload = _payload({"huge": 2**63 - 1, "tiny": -(2**63), "fraction": 0.5})
    assert validate_payload_v2(payload)["total"] is None
    payload["metrics"]["huge"] = 2**63
    with pytest.raises((TypeError, ValueError)):
        validate_payload_v2(payload)


@pytest.mark.parametrize(
    "metrics",
    [
        {"bool": True},
        {"infinite": math.inf},
        {"negative_infinite": -math.inf},
        {"too_negative": -(2**63) - 1},
        {"nested": [1]},
        {"bad/name?": 1},
        {f"m{index}": index for index in range(33)},
    ],
)
def test_wire_rejects_nonprimitive_or_oversized_metrics(metrics):
    with pytest.raises((TypeError, ValueError)):
        validate_payload_v2(_payload(metrics))


def test_wire_text_uses_utf8_bytes_and_rejects_controls():
    payload = _payload()
    payload["stage"] = "é" * 32
    assert validate_payload_v2(payload)["stage"] == payload["stage"]
    payload["stage"] += "é"
    with pytest.raises(ValueError):
        validate_payload_v2(payload)
    payload["stage"] = "train\x1b[31m"
    with pytest.raises(ValueError):
        validate_payload_v2(payload)


def test_wire_reader_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "latest-v2.json"
    path.write_text('{"protocol_version":2,"protocol_version":2}')
    with pytest.raises((TypeError, ValueError)):
        read_payload_v2(path)
