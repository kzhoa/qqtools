"""Scoped replacement reports preserve strict bounds and producer safety."""

from dataclasses import FrozenInstanceError

import pytest

from qqtools.qexp import progress
from qqtools.qexp._progress_protocol_v3 import semantic_key_v3, validate_payload_v3


def payload():
    return {
        "protocol_version": 3,
        "update_id": "update-1",
        "activity": {"stage": "validation", "current": 2, "total": 10, "unit": "batch", "message": None},
        "overall": {"current": 8, "total": 10, "unit": "step", "label": "Training"},
        "metrics": {"loss": 0.5},
        "completeness": {"complete": True, "omitted_metrics": 0, "reasons": []},
    }


@pytest.mark.parametrize("section", [None, "activity", "overall", "completeness"])
def test_unknown_fields_are_rejected(section):
    value = payload()
    (value if section is None else value[section])["extra"] = 1
    with pytest.raises(ValueError):
        validate_payload_v3(value)


@pytest.mark.parametrize("current,total", [(0, 0), (7, None), (2**63 - 1, None), (3, 3)])
def test_valid_counter_boundaries(current, total):
    value = payload()
    value["overall"].update(current=current, total=total)
    assert validate_payload_v3(value) == value
    assert (
        progress.validate_update(
            stage="arbitrary", overall=progress.Counter(current=current, total=total, unit="epoch")
        )
        == ()
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("current", True),
        ("current", -1),
        ("current", 2**63),
        ("total", False),
        ("total", 7),
        ("unit", ""),
        ("unit", "界" * 11),
        ("label", ""),
        ("label", "é" * 33),
        ("label", "bad\x1b"),
    ],
)
def test_invalid_overall_is_field_qualified(field, value):
    fields = {"current": 8, "total": 10, "unit": "step", "label": "Training"}
    fields[field] = value
    counter = progress.Counter(**fields)
    errors = progress.validate_update(stage="train", overall=counter)
    expected_field = "current" if field == "total" and value == 7 else field
    assert any(item.startswith(f"overall.{expected_field}:") for item in errors)
    wire = payload()
    wire["overall"] = fields
    with pytest.raises(ValueError):
        validate_payload_v3(wire)


def test_counter_is_keyword_only_immutable_and_construction_does_not_validate():
    with pytest.raises(TypeError):
        progress.Counter(1, "step")
    counter = progress.Counter(current=True, unit="")
    with pytest.raises(FrozenInstanceError):
        counter.current = 3
    assert progress.validate_update(stage="train", overall=counter)


def test_validation_is_pure_and_metric_omission_is_valid(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("validation touched runtime")

    monkeypatch.setattr(progress, "_get_reporter", forbidden)
    with monkeypatch.context() as isolated:
        isolated.setattr(progress.os, "environ", None)
        assert progress.validate_update(stage="custom", metrics={"bad": object()}) == ()
    errors = progress.validate_update(stage="", current=-1, total=False, unit=3, message=4, overall={})
    assert tuple(item.split(":")[0] for item in errors) == ("stage", "current", "total", "unit", "message", "overall")


def test_semantics_include_overall_label_and_metrics_but_not_update_id():
    value = payload()
    original = semantic_key_v3(value)
    value["update_id"] = "another"
    assert semantic_key_v3(value) == original
    value["overall"]["label"] = "Epochs"
    assert semantic_key_v3(value) != original
    value["overall"] = None
    assert validate_payload_v3(value)["overall"] is None
