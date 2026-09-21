from __future__ import annotations

import hashlib
import json

import pytest

from qqtools.plugins.qexp.runtime.records import new_submission
from qqtools.plugins.qexp.runtime.submission_plan import (
    decode_submission_plan,
    encode_submission_plan,
    normalize_submission_request,
)


def _canonical_context() -> dict:
    return {
        "task_ids": ["task-a", "task-b"],
        "task_specs": [
            {
                "task_id": "task-a",
                "name": "first",
                "home_machine": "g1",
                "command": ["echo", "first"],
                "working_directory": "/work/first",
                "requested_gpus": 1,
                "sharing_mode": "spillover",
                "fallback_machines": ["g2"],
                "offer_after_seconds": None,
                "depends_on_task_ids": [],
            },
            {
                "task_id": "task-b",
                "name": None,
                "home_machine": "g2",
                "command": ["echo", "second"],
                "working_directory": "/work/second",
                "requested_gpus": 1,
                "requested_cpus": None,
                "sharing_mode": "private",
                "fallback_machines": "group",
                "offer_after_seconds": None,
                "depends_on_task_ids": ["task-a"],
            },
        ],
        "create_group": False,
        "worker_set_additions": {
            "g2": {"scheduling_role": "borrow", "gpu_limit_gpus": 2},
        },
        "group_precondition": {"exists": True, "revision": 7, "worker_set_epoch": 3},
        "planned_worker_set": ["g1", "g2"],
    }


def _operation(context: dict | None = None) -> dict:
    return new_submission(
        operation_id="operation-a",
        kind="bulk",
        key="key-a",
        raw_digest="raw-a",
        machine="g1",
        target_group="experiment",
        resolved_context=context or _canonical_context(),
    )


def _resolved_digest(context: dict) -> str:
    return hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()


def test_request_digest_is_compatible_and_nested_input_is_snapshotted() -> None:
    command = ["echo", "before"]
    fallback = ["g2"]
    workers = {"g2": {"scheduling_role": "borrow", "gpu_limit_gpus": 2}}
    specs = [{"command": command, "fallback_machines": fallback}]
    request = normalize_submission_request(
        specs,
        group_name="experiment",
        kind="bulk",
        worker_set=workers,
    )
    expected_input = {
        "group": "experiment",
        "tasks": [
            {
                "command": ["echo", "before"],
                "fallback_machines": ["g2"],
                "home_machine": "current",
                "tmux_override": None,
            }
        ],
        "worker_set": {"g2": {"gpu_limit_gpus": 2, "scheduling_role": "borrow"}},
    }
    expected = hashlib.sha256(json.dumps(expected_input, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    command[1] = "after"
    fallback.append("g3")
    workers["g2"]["gpu_limit_gpus"] = 8
    specs[0]["home_machine"] = "g3"

    assert request.raw_request_digest == expected
    assert request.kind == "bulk"
    assert request.specs[0]["command"] == ("echo", "before")
    assert request.specs[0]["fallback_machines"] == ("g2",)
    assert request.worker_set_additions["g2"]["gpu_limit_gpus"] == 2
    with pytest.raises(TypeError):
        request.specs[0]["home_machine"] = "g3"


def test_current_canonical_operation_round_trips_with_independent_containers() -> None:
    source = _operation()
    plan = decode_submission_plan(source)
    first = encode_submission_plan(plan)
    second = encode_submission_plan(plan)

    assert first["submission"]["resolved_context"] == source["submission"]["resolved_context"]
    assert first["submission"]["resolved_context_digest"] == source["submission"]["resolved_context_digest"]
    first["submission"]["resolved_context"]["task_specs"][0]["command"][1] = "mutated"
    first["submission"]["resolved_context"]["worker_set_additions"]["g2"]["gpu_limit_gpus"] = 8

    assert second["submission"]["resolved_context"]["task_specs"][0]["command"] == ["echo", "first"]
    assert plan.task_specs[0]["command"] == ("echo", "first")
    assert plan.worker_set_additions["g2"]["gpu_limit_gpus"] == 2


def test_decoder_accepts_current_optional_task_fields_when_omitted() -> None:
    context = _canonical_context()
    first = context["task_specs"][0]
    assert "requested_cpus" not in first
    assert "offer_eligible_at" not in first
    assert "offer_clock_evidence" not in first

    plan = decode_submission_plan(_operation(context))

    assert plan.task_ids == ("task-a", "task-b")
    assert plan.task_specs[0]["requested_gpus"] == 1


@pytest.mark.parametrize(
    "damage",
    [
        "digest",
        "task-order",
        "missing-dependencies",
        "missing-offer-evidence",
        "invalid-offer-evidence-type",
        "unknown-task-field",
        "dependency-cycle",
        "contradictory-ungrouped-plan",
        "empty-plan",
    ],
)
def test_decoder_rejects_malformed_or_inconsistent_canonical_context(damage: str) -> None:
    operation = _operation()
    submission = operation["submission"]
    context = submission["resolved_context"]
    if damage == "digest":
        submission["resolved_context_digest"] = "0" * 64
    elif damage == "task-order":
        context["task_ids"] = ["task-b", "task-a"]
        submission["resolved_context_digest"] = _resolved_digest(context)
    elif damage == "missing-dependencies":
        del context["task_specs"][0]["depends_on_task_ids"]
        submission["resolved_context_digest"] = _resolved_digest(context)
    elif damage == "missing-offer-evidence":
        context["task_specs"][0]["offer_after_seconds"] = 30
        submission["resolved_context_digest"] = _resolved_digest(context)
    elif damage == "invalid-offer-evidence-type":
        spec = context["task_specs"][0]
        spec["offer_after_seconds"] = 30
        spec["offer_eligible_at"] = "2026-09-20T00:00:30Z"
        spec["offer_clock_evidence"] = {
            "creator_observation": {
                "observation_id": "observation-a",
                "provider": "chrony",
                "observed_at": "2026-09-20T00:00:00Z",
                "monotonic_observed_at": "invalid",
                "boot_id": "boot-a",
                "lower_error_seconds": -0.1,
                "upper_error_seconds": 0.1,
                "max_drift_rate": 0.0,
                "provider_margin_seconds": 0.0,
            },
            "deadline_monotonic_at": True,
        }
        submission["resolved_context_digest"] = _resolved_digest(context)
    elif damage == "unknown-task-field":
        context["task_specs"][0]["unexpected"] = True
        submission["resolved_context_digest"] = _resolved_digest(context)
    elif damage == "dependency-cycle":
        context["task_specs"][0]["depends_on_task_ids"] = ["task-b"]
        submission["resolved_context_digest"] = _resolved_digest(context)
    elif damage == "contradictory-ungrouped-plan":
        submission["target_group"] = None
        context["create_group"] = False
        submission["resolved_context_digest"] = _resolved_digest(context)
    else:
        context["task_ids"] = []
        context["task_specs"] = []
        submission["staged_task_count"] = 0
        submission["resolved_context_digest"] = _resolved_digest(context)

    with pytest.raises((RuntimeError, ValueError)):
        decode_submission_plan(operation)


def test_decoded_plan_is_recursively_immutable() -> None:
    plan = decode_submission_plan(_operation())

    with pytest.raises(TypeError):
        plan.task_specs[0]["command"][0] = "false"
    with pytest.raises(TypeError):
        plan.group_precondition["revision"] = 8
    with pytest.raises(TypeError):
        plan.worker_set_additions["g2"]["scheduling_role"] = "primary"
