from __future__ import annotations

import pytest

from qqtools.plugins.qexp.agent.group_service_transport import (
    GroupServiceRequestSpec,
    group_service_advance_evidence,
    group_service_advance_parameters,
    group_service_advance_request,
    group_service_candidate,
    group_service_probe_evidence,
    group_service_probe_parameters,
    group_service_probe_request,
    initial_group_service_continuation,
)
from qqtools.plugins.qexp.runtime.group_discovery.probe import initial_group_service_probe_state


def test_group_service_probe_contract_round_trips_closed_state():
    state = initial_group_service_probe_state()
    assert group_service_probe_parameters({"machine_name": "gpu-1", "probe_state": state}) == {
        "machine_name": "gpu-1",
        "probe_state": state,
    }
    assert group_service_probe_evidence({"state": "quiescent", "probe_state": state, "candidate": None}) == {
        "state": "quiescent",
        "probe_state": state,
        "candidate": None,
    }


def test_group_request_builders_preserve_the_closed_parameter_encoding():
    probe_state = initial_group_service_probe_state()
    probe = group_service_probe_request("gpu-1", probe_state)
    assert probe == GroupServiceRequestSpec(
        operation_kind="group_service_probe",
        parameters={"machine_name": "gpu-1", "probe_state": probe_state},
    )

    candidate = {"group": "experiment", "lane": "membership", "generation": 3}
    continuation = initial_group_service_continuation(candidate)
    advance = group_service_advance_request("gpu-1", candidate, continuation, probe_state)
    assert advance == GroupServiceRequestSpec(
        operation_kind="group_service_advance",
        parameters={
            "machine_name": "gpu-1",
            "candidate": candidate,
            "continuation": continuation,
            "probe_state": probe_state,
        },
    )


def test_group_request_spec_rejects_an_unsupported_operation():
    with pytest.raises(ValueError, match="operation"):
        GroupServiceRequestSpec("scheduler_claim", {"machine_name": "gpu-1"})


@pytest.mark.parametrize(
    "value",
    [
        {"state": "active", "probe_state": initial_group_service_probe_state(), "candidate": None},
        {
            "state": "pending",
            "probe_state": initial_group_service_probe_state(),
            "candidate": {"group": "experiment", "lane": "legacy", "generation": None},
        },
        {
            "state": "active",
            "probe_state": initial_group_service_probe_state(),
            "candidate": {"group": "experiment", "lane": "control", "generation": None},
        },
    ],
)
def test_group_service_probe_contract_rejects_inconsistent_candidate(value):
    with pytest.raises(ValueError):
        group_service_probe_evidence(value)


def test_group_candidate_validates_directly_without_probe_state() -> None:
    candidate = {"group": "experiment", "lane": "control", "generation": 2}
    assert group_service_candidate(candidate) == candidate

    with pytest.raises(ValueError, match="positive generation"):
        group_service_candidate({**candidate, "generation": None})


def test_group_advance_contract_round_trips_discovery_continuation():
    candidate = {"group": "experiment", "lane": "membership", "generation": 3}
    continuation = initial_group_service_continuation(candidate)
    parameters = group_service_advance_parameters(
        {
            "machine_name": "gpu-1",
            "candidate": candidate,
            "continuation": continuation,
            "probe_state": initial_group_service_probe_state(),
        }
    )

    assert parameters["candidate"] == candidate
    assert (
        group_service_advance_evidence({"state": "progress", "continuation": continuation, "probe": None}, candidate)[
            "continuation"
        ]
        == continuation
    )


def test_group_advance_contract_rejects_lane_mismatched_continuation():
    candidate = {"group": "experiment", "lane": "maintenance", "generation": 3}
    with pytest.raises(ValueError, match="cannot carry"):
        group_service_advance_parameters(
            {
                "machine_name": "gpu-1",
                "candidate": candidate,
                "continuation": {},
                "probe_state": initial_group_service_probe_state(),
            }
        )
