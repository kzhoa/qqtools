from __future__ import annotations

import uuid
from dataclasses import replace
from pathlib import Path

import pytest

from qqtools.plugins.qexp.agent.dispatch_probe import PrimaryProbeSession
from qqtools.plugins.qexp.agent.primary_probe_transport import encode_probe_session
from qqtools.plugins.qexp.agent.project_io_protocol import (
    PROJECT_IO_MAX_RECORD_BYTES,
    PROJECT_IO_PROTOCOL_VERSION,
    ProjectIORequest,
    ProjectIOResult,
)
from qqtools.plugins.qexp.runtime.store import JSONRecordSizeError


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("pending", 1),
        ("can_run", None),
        ("admission_blocked", "false"),
        ("idle_blocking", 0),
        ("state", "unknown"),
        ("next_probe_at", "tomorrow"),
    ],
)
def test_upgrade_summary_rejects_non_typed_evidence(tmp_path, field, invalid):
    request = replace(
        _request(tmp_path),
        operation_kind="upgrade_service",
        parameters={"machine_name": "gpu-1"},
        source_revisions={},
    )
    evidence = {
        "state": "completed",
        "pending": False,
        "can_run": False,
        "admission_blocked": False,
        "idle_blocking": False,
        "next_probe_at": None,
    }
    evidence[field] = invalid
    with pytest.raises(ValueError):
        ProjectIOResult(request, "completed", None, "2026-09-30T00:00:00Z", evidence)


def _terminal_observation_result(tmp_path: Path, cancel_requested: object) -> ProjectIOResult:
    parameters = {
        "machine_name": "gpu-1",
        "task_id": "task-a",
        "attempt_id": "task-a-attempt-1",
        "attempt_number": 1,
        "fencing_token": 1,
        "reservation_id": None,
        "process_identity": {
            "wrapper_pid": None,
            "wrapper_start_time_ticks": None,
            "process_group_id": None,
            "process_group_start_time_ticks": None,
        },
        "mode": "active",
    }
    request = replace(
        _request(tmp_path), operation_kind="authority_terminal_observe", parameters=parameters, source_revisions={}
    )
    return ProjectIOResult(
        request=request,
        status="completed",
        reason_code=None,
        completed_at="2026-09-28T00:00:01Z",
        evidence={
            **parameters,
            "outcome": "current",
            "reason": None,
            "task_phase": "running",
            "attempt_phase": "running",
            "attempt_result_reason": None,
            "attempt_exit_code": None,
            "execution_machine_name": "gpu-1",
            "reservation_machine_name": "gpu-1",
            "termination_result": None,
            "cancel_requested": cancel_requested,
            "source_revisions": {"task": 2, "attempt_digest": "a" * 64},
            "authority_granted": False,
            "local_effects": [],
        },
    )


@pytest.mark.parametrize("cancel_requested", [False, True])
def test_terminal_observation_carries_exact_boolean_cancellation_intent(tmp_path, cancel_requested):
    result = _terminal_observation_result(tmp_path, cancel_requested)
    restored = ProjectIOResult.from_dict(result.to_dict())
    assert restored.evidence["cancel_requested"] is cancel_requested


@pytest.mark.parametrize("cancel_requested", [None, 0, 1, "true", [], {}])
def test_terminal_observation_rejects_nonboolean_cancellation_intent(tmp_path, cancel_requested):
    with pytest.raises(ValueError, match="cancel_requested"):
        _terminal_observation_result(tmp_path, cancel_requested)


@pytest.mark.parametrize("task_phase", ["queued", "running", "failed"])
def test_settled_terminal_observation_round_trips_without_authority(tmp_path, task_phase):
    value = _terminal_observation_result(tmp_path, False).to_dict()
    evidence = value["project_io_result"]["evidence"]
    evidence.update(outcome="settled_terminal", task_phase=task_phase, attempt_phase="failed")
    result = ProjectIOResult.from_dict(value)
    assert result.evidence["outcome"] == "settled_terminal"
    assert result.evidence["task_phase"] == task_phase
    assert result.evidence["authority_granted"] is False
    assert result.evidence["local_effects"] == ()
    assert ProjectIOResult.from_dict(result.to_dict()) == result


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("task_phase", None),
        ("attempt_phase", "running"),
        ("source_revisions", {"task": None, "attempt_digest": "a" * 64}),
        ("source_revisions", {"task": 2, "attempt_digest": None}),
        ("execution_machine_name", "other-machine"),
        ("reservation_machine_name", None),
        ("authority_granted", True),
        ("local_effects", ["release_capacity"]),
    ],
)
def test_settled_terminal_observation_requires_exact_non_authorizing_evidence(tmp_path, field, invalid):
    value = _terminal_observation_result(tmp_path, False).to_dict()
    evidence = value["project_io_result"]["evidence"]
    evidence.update(outcome="settled_terminal", task_phase="queued", attempt_phase="failed")
    evidence[field] = invalid
    with pytest.raises(ValueError):
        ProjectIOResult.from_dict(value)


def _request(tmp_path: Path) -> ProjectIORequest:
    return ProjectIORequest(
        protocol_version=PROJECT_IO_PROTOCOL_VERSION,
        runtime_id="a" * 64,
        executor_epoch="b" * 32,
        request_id="c" * 32,
        operation_kind="validate_binding",
        project_id="project-a",
        canonical_shared_root=str((tmp_path / "project" / ".qexp").resolve()),
        registration_generation="registration-a",
        registry_revision=7,
        source_revisions={"inventory_revision": 4},
        provisional_offer_id=None,
        prepared_at="2026-09-28T00:00:00Z",
        parameters={"machine_name": "gpu-1"},
    )


def _submission_service_request(tmp_path: Path) -> ProjectIORequest:
    from qqtools.plugins.qexp.runtime.submission_control_continuation import submission_control_continuation

    return replace(
        _request(tmp_path),
        operation_kind="submission_control_service",
        source_revisions={},
        parameters={"machine_name": "gpu-1", "continuation": submission_control_continuation()},
    )


@pytest.mark.parametrize(
    "continuation",
    [
        None,
        {},
        {"payload": "private"},
        {"pending_lane": 1, "pending_offset": 0, "pending_inode": None, "pending_had_work": False},
        {"pending_lane": True, "pending_offset": -1, "pending_inode": None, "pending_had_work": False},
        {"pending_lane": True, "pending_offset": True, "pending_inode": None, "pending_had_work": False},
        {"pending_lane": True, "pending_offset": 1, "pending_inode": None, "pending_had_work": True},
        {"pending_lane": True, "pending_offset": 1, "pending_inode": 1, "pending_had_work": False},
        {"pending_lane": True, "pending_offset": 0, "pending_inode": True, "pending_had_work": False},
    ],
)
def test_submission_service_rejects_open_or_inconsistent_continuations(tmp_path, continuation):
    request = _submission_service_request(tmp_path)
    with pytest.raises(ValueError):
        replace(request, parameters={"machine_name": "gpu-1", "continuation": continuation})


@pytest.mark.parametrize(
    "field,value",
    [
        ("state", "unknown"),
        ("quiescent", 1),
        ("quiescent", False),
        ("reason_code", "free-form error"),
        ("reason_code", "blocked"),
        ("continuation", None),
    ],
)
def test_submission_service_rejects_unfounded_quiescence(tmp_path, field, value):
    request = _submission_service_request(tmp_path)
    evidence = {
        "state": "waiting",
        "quiescent": True,
        "reason_code": "idle",
        "continuation": dict(request.parameters["continuation"]),
    }
    evidence[field] = value
    with pytest.raises(ValueError):
        ProjectIOResult(request, "completed", None, "2026-09-28T00:00:01Z", evidence)


def test_submission_service_round_trips_closed_idle_evidence(tmp_path):
    request = _submission_service_request(tmp_path)
    evidence = {
        "state": "waiting",
        "quiescent": True,
        "reason_code": "idle",
        "continuation": dict(request.parameters["continuation"]),
    }
    result = ProjectIOResult(request, "completed", None, "2026-09-28T00:00:01Z", evidence)
    assert ProjectIOResult.from_dict(result.to_dict()) == result


def _primary_probe_request(tmp_path: Path) -> ProjectIORequest:
    return replace(
        _request(tmp_path),
        operation_kind="scheduler_primary_probe",
        source_revisions={},
        parameters={
            "machine_name": "gpu-1",
            "lane": "gpu",
            "round_id": "d" * 32,
            "phase": "scan",
            "capacity_digest": "e" * 64,
            "visible_capacity": 2,
            "free_capacity": 1,
            "group_gpu_usage": {"exp": 1},
            "probe_state": encode_probe_session(PrimaryProbeSession(), "project-a", "gpu"),
        },
    )


def _scheduler_quiescence_request(tmp_path: Path) -> ProjectIORequest:
    return replace(
        _request(tmp_path),
        operation_kind="scheduler_quiescence_probe",
        source_revisions={},
        parameters={
            "machine_name": "gpu-1",
            "probe_state": encode_probe_session(PrimaryProbeSession(), "project-a", "gpu"),
        },
    )


def test_scheduler_quiescence_probe_has_closed_non_authorizing_parameters(tmp_path):
    request = _scheduler_quiescence_request(tmp_path)
    assert ProjectIORequest.from_dict(request.to_dict()) == request
    with pytest.raises(ValueError, match="source_revisions"):
        replace(request, source_revisions={"task": 1})
    with pytest.raises(ValueError, match="provisional offer"):
        replace(request, provisional_offer_id="offer-a")
    with pytest.raises(ValueError, match="parameters"):
        replace(request, parameters={**request.parameters, "lane": "gpu"})


def test_scheduler_quiescence_probe_rejects_incomplete_absence_proof(tmp_path):
    request = _scheduler_quiescence_request(tmp_path)
    state = request.parameters["probe_state"]
    with pytest.raises(ValueError, match="both complete routes"):
        ProjectIOResult(
            request, "completed", None, "2026-09-28T00:00:01Z", {"state": "quiescent", "probe_state": state}
        )
    result = ProjectIOResult(
        request, "completed", None, "2026-09-28T00:00:01Z", {"state": "pending", "probe_state": state}
    )
    assert ProjectIOResult.from_dict(result.to_dict()) == result


def test_primary_probe_request_round_trips_with_closed_continuation(tmp_path):
    request = _primary_probe_request(tmp_path)
    assert ProjectIORequest.from_dict(request.to_dict()) == request
    with pytest.raises(ValueError, match="source_revisions"):
        replace(request, source_revisions={"ready_catalog": 1})
    with pytest.raises(ValueError, match="provisional offer"):
        replace(request, provisional_offer_id="offer-a")


@pytest.mark.parametrize(
    "field,value",
    [
        ("visible_capacity", True),
        ("free_capacity", 3),
        ("visible_capacity", 4097),
        ("round_id", "invalid"),
        ("capacity_digest", "invalid"),
        ("phase", "publish"),
        ("group_gpu_usage", {"exp": True}),
    ],
)
def test_primary_probe_request_rejects_invalid_inputs(tmp_path, field, value):
    request = _primary_probe_request(tmp_path)
    parameters = dict(request.parameters)
    parameters[field] = value
    with pytest.raises(ValueError):
        replace(request, parameters=parameters)


def test_primary_absence_requires_both_exact_completed_watermarks(tmp_path):
    request = _primary_probe_request(tmp_path)
    state = encode_probe_session(PrimaryProbeSession(), "project-a", "gpu")
    evidence = {"demand": "no_primary_demand", "probe_state": state, "route_revisions": {"home": 2, "shared": 3}}
    with pytest.raises(ValueError, match="completed baseline"):
        ProjectIOResult(request, "completed", None, "2026-09-28T00:00:01Z", evidence)
    for scope, revision in evidence["route_revisions"].items():
        state["routes"][scope].update(is_complete=True, revision=revision)
    result = ProjectIOResult(request, "completed", None, "2026-09-28T00:00:01Z", evidence)
    assert ProjectIOResult.from_dict(result.to_dict()) == result
    evidence["route_revisions"]["home"] = 4
    with pytest.raises(ValueError, match="completed baseline"):
        ProjectIOResult(request, "completed", None, "2026-09-28T00:00:01Z", evidence)


def _running_publication_request(tmp_path: Path) -> ProjectIORequest:
    return replace(
        _request(tmp_path),
        operation_kind="authority_running_publish",
        source_revisions={},
        parameters={
            "machine_name": "gpu-1",
            "task_id": "task-a",
            "attempt_id": "task-a-attempt-1",
            "attempt_number": 1,
            "fencing_token": 1,
            "reservation_id": "reservation-a",
            "process_identity": {
                "wrapper_pid": 123,
                "wrapper_start_time_ticks": 456,
                "process_group_id": None,
                "process_group_start_time_ticks": None,
            },
            "process_created_at": "2026-09-29T00:00:00Z",
        },
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("attempt_id", "task-b-attempt-1"),
        ("attempt_number", True),
        ("fencing_token", True),
        ("fencing_token", 0),
        ("reservation_id", None),
        ("process_created_at", "yesterday"),
        ("process_identity", {"wrapper_pid": 123}),
        ("manifest", "/arbitrary/local/path"),
    ],
)
def test_running_publication_rejects_invalid_or_unbounded_parameter_shape(tmp_path, field, value):
    request = _running_publication_request(tmp_path)
    with pytest.raises(ValueError):
        replace(request, parameters={**request.parameters, field: value})


def test_running_publication_has_no_revision_or_capacity_authority(tmp_path):
    request = _running_publication_request(tmp_path)
    with pytest.raises(ValueError):
        replace(request, source_revisions={"task": 1})
    with pytest.raises(ValueError):
        replace(request, provisional_offer_id="offer-a")


def test_activation_consumer_retirement_protocol_is_closed_and_non_authorizing(tmp_path):
    request = replace(
        _request(tmp_path),
        operation_kind="activation_consumer_retire",
        source_revisions={},
        parameters={"machine_name": "gpu-1"},
    )
    result = ProjectIOResult(
        request=request,
        status="completed",
        reason_code=None,
        completed_at="2026-09-29T00:00:01Z",
        evidence={"outcome": "retired", "consumer_existed": True},
    )

    assert ProjectIORequest.from_dict(request.to_dict()) == request
    assert ProjectIOResult.from_dict(result.to_dict()) == result
    with pytest.raises(ValueError, match="parameters"):
        replace(request, parameters={"machine_name": "gpu-1", "process_fence": "process-a"})
    with pytest.raises(ValueError, match="source_revisions"):
        replace(request, source_revisions={"membership": 1})
    with pytest.raises(ValueError, match="consumer_existed"):
        replace(result, evidence={"outcome": "retired", "consumer_existed": 1})
    with pytest.raises(ValueError, match="outcome"):
        replace(result, evidence={"outcome": "registered", "consumer_existed": True})


@pytest.mark.parametrize("transitioned", [False, True])
def test_running_publication_result_round_trip_does_not_grant_authority(tmp_path, transitioned):
    request = _running_publication_request(tmp_path)
    result = ProjectIOResult(
        request=request,
        status="completed",
        reason_code=None,
        completed_at="2026-09-29T00:00:01Z",
        evidence={
            **request.parameters,
            "outcome": "processed",
            "reason": None,
            "transitioned_to_running": transitioned,
            "authority_granted": False,
            "local_effects": [],
        },
    )
    assert ProjectIOResult.from_dict(result.to_dict()) == result
    assert result.evidence["authority_granted"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_id", "task-b"),
        ("fencing_token", True),
        ("attempt_number", True),
        ("transitioned_to_running", 1),
        ("authority_granted", True),
        ("local_effects", ["release"]),
        ("outcome", "current"),
        ("reason", "grant"),
    ],
)
def test_running_publication_result_rejects_changed_identity_and_authority(tmp_path, field, value):
    request = _running_publication_request(tmp_path)
    with pytest.raises(ValueError):
        ProjectIOResult(
            request=request,
            status="completed",
            reason_code=None,
            completed_at="2026-09-29T00:00:01Z",
            evidence={
                **request.parameters,
                "outcome": "processed",
                "reason": None,
                "transitioned_to_running": False,
                "authority_granted": False,
                "local_effects": [],
                field: value,
            },
        )


def test_project_io_request_round_trip_is_exact_and_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request(tmp_path)
    value = request.to_dict()

    def fail_resolve(*_args, **_kwargs):
        raise AssertionError("controller protocol parsing must not resolve the shared root")

    monkeypatch.setattr(Path, "resolve", fail_resolve)

    assert ProjectIORequest.from_dict(value) == request
    assert len(str(value).encode()) < PROJECT_IO_MAX_RECORD_BYTES


def test_project_io_request_rejects_unknown_fields_and_boolean_revisions(tmp_path: Path) -> None:
    value = _request(tmp_path).to_dict()
    value["project_io_request"]["unexpected"] = "field"

    with pytest.raises(ValueError, match="field|key|shape"):
        ProjectIORequest.from_dict(value)

    value = _request(tmp_path).to_dict()
    value["project_io_request"]["registry_revision"] = True
    with pytest.raises((TypeError, ValueError), match="registry_revision"):
        ProjectIORequest.from_dict(value)


def test_project_io_request_rejects_noncanonical_root_and_unbounded_revisions(tmp_path: Path) -> None:
    request = _request(tmp_path)
    value = request.to_dict()
    value["project_io_request"]["canonical_shared_root"] = "relative/.qexp"

    with pytest.raises(ValueError, match="canonical_shared_root"):
        ProjectIORequest.from_dict(value)

    value = request.to_dict()
    value["project_io_request"]["source_revisions"] = {f"revision_{index}": index for index in range(10_000)}
    with pytest.raises((JSONRecordSizeError, TypeError, ValueError)):
        ProjectIORequest.from_dict(value)


def test_project_io_result_repeats_request_identity_and_rejects_arbitrary_status(tmp_path: Path) -> None:
    request = _request(tmp_path)
    result = ProjectIOResult(
        request=request,
        status="completed",
        reason_code=None,
        completed_at="2026-09-28T00:00:01Z",
        evidence={
            "project_id": "project-a",
            "shared_root": request.canonical_shared_root,
            "schema_version": 6,
            "required_capabilities": [],
        },
    )

    assert ProjectIOResult.from_dict(result.to_dict()) == result

    value = result.to_dict()
    value["project_io_result"]["status"] = "successful-ish"
    with pytest.raises(ValueError, match="status"):
        ProjectIOResult.from_dict(value)


def test_project_io_result_rejects_oversized_evidence(tmp_path: Path) -> None:
    request = _request(tmp_path)

    with pytest.raises((JSONRecordSizeError, TypeError, ValueError)):
        ProjectIOResult(
            request=request,
            status="completed",
            reason_code=None,
            completed_at="2026-09-28T00:00:01Z",
            evidence={
                "project_id": request.project_id,
                "shared_root": request.canonical_shared_root,
                "schema_version": 6,
                "required_capabilities": ["x" * PROJECT_IO_MAX_RECORD_BYTES],
            },
        ).to_dict()


def test_registration_renew_protocol_is_closed_and_evidence_is_consistent(tmp_path: Path) -> None:
    base = _request(tmp_path)
    request = ProjectIORequest(
        protocol_version=base.protocol_version,
        runtime_id=base.runtime_id,
        executor_epoch=base.executor_epoch,
        request_id=base.request_id,
        operation_kind="registration_renew",
        project_id=base.project_id,
        canonical_shared_root=base.canonical_shared_root,
        registration_generation=base.registration_generation,
        registry_revision=base.registry_revision,
        source_revisions={},
        provisional_offer_id=None,
        prepared_at=base.prepared_at,
        parameters={"machine_name": "gpu-1", "renewal_horizon_seconds": 10.0},
    )
    completed = ProjectIOResult(
        request=request,
        status="completed",
        reason_code=None,
        completed_at="2026-09-28T00:00:01Z",
        evidence={
            "outcome": "eligible",
            "renewed": True,
            "eligibility_expires_at": "2026-09-28T00:02:00Z",
            "renew_after_seconds": 10.0,
            "reason": None,
        },
    )

    assert ProjectIORequest.from_dict(request.to_dict()) == request
    assert ProjectIOResult.from_dict(completed.to_dict()) == completed

    invalid_horizon = request.to_dict()
    invalid_horizon["project_io_request"]["parameters"]["renewal_horizon_seconds"] = True
    with pytest.raises(ValueError, match="renewal_horizon_seconds"):
        ProjectIORequest.from_dict(invalid_horizon)

    inconsistent_stale = completed.to_dict()
    evidence = inconsistent_stale["project_io_result"]["evidence"]
    evidence.update(
        {
            "outcome": "stale",
            "renewed": True,
            "eligibility_expires_at": None,
            "renew_after_seconds": None,
            "reason": "binding_fence",
        }
    )
    with pytest.raises(ValueError, match="registration_renew"):
        ProjectIOResult.from_dict(inconsistent_stale)

    for invalid_cadence in (True, -1, float("nan"), float("inf"), 86_401):
        invalid = completed.to_dict()
        invalid["project_io_result"]["evidence"]["renew_after_seconds"] = invalid_cadence
        with pytest.raises(ValueError, match="renew_after_seconds"):
            ProjectIOResult.from_dict(invalid)


def test_activation_io_protocol_closes_parameters_and_evidence(tmp_path: Path) -> None:
    base = _request(tmp_path)
    epoch = uuid.uuid4().hex

    def activation_request(kind: str, parameters: dict[str, object]) -> ProjectIORequest:
        return ProjectIORequest(
            protocol_version=base.protocol_version,
            runtime_id=base.runtime_id,
            executor_epoch=base.executor_epoch,
            request_id=base.request_id,
            operation_kind=kind,
            project_id=base.project_id,
            canonical_shared_root=base.canonical_shared_root,
            registration_generation=base.registration_generation,
            registry_revision=base.registry_revision,
            source_revisions={},
            provisional_offer_id=None,
            prepared_at=base.prepared_at,
            parameters=parameters,
        )

    observe = activation_request(
        "activation_observe",
        {"machine_name": "gpu-1", "replay_epoch": None, "replay_sequence": 0},
    )
    register = activation_request(
        "activation_consumer_register",
        {"machine_name": "gpu-1", "process_fence": "process-a"},
    )
    ack = activation_request(
        "activation_consumer_ack",
        {
            "machine_name": "gpu-1",
            "process_fence": "process-a",
            "epoch": epoch,
            "sequence": 3,
            "reconstructed_floor": None,
            "require_current": True,
        },
    )

    for request, evidence in (
        (
            observe,
            {
                "outcome": "observed",
                "checkpoint": {"epoch": epoch, "sequence": 3},
                "replay": {
                    "epoch": epoch,
                    "sequence": 3,
                    "reconstructed_floor": None,
                    "complete": True,
                },
            },
        ),
        (register, {"outcome": "registered", "acknowledgement": None}),
        (ack, {"outcome": "acknowledged", "acknowledgement": {"epoch": epoch, "sequence": 3}}),
    ):
        assert ProjectIORequest.from_dict(request.to_dict()) == request
        result = ProjectIOResult(
            request=request,
            status="completed",
            reason_code=None,
            completed_at="2026-09-28T00:00:01Z",
            evidence=evidence,
        )
        assert ProjectIOResult.from_dict(result.to_dict()) == result

    malformed = ack.to_dict()
    malformed["project_io_request"]["parameters"]["require_current"] = 1
    with pytest.raises(ValueError, match="require_current"):
        ProjectIORequest.from_dict(malformed)

    with pytest.raises(ValueError, match="differs from its request"):
        ProjectIOResult(
            request=ack,
            status="completed",
            reason_code=None,
            completed_at="2026-09-28T00:00:01Z",
            evidence={"outcome": "acknowledged", "acknowledgement": {"epoch": epoch, "sequence": 2}},
        )

    with pytest.raises(ValueError, match="sequence is invalid"):
        ProjectIOResult(
            request=observe,
            status="completed",
            reason_code=None,
            completed_at="2026-09-28T00:00:01Z",
            evidence={
                "outcome": "observed",
                "checkpoint": {"epoch": epoch, "sequence": 0},
                "replay": None,
            },
        )


def test_authority_service_is_point_in_time_observation_without_authority(tmp_path: Path) -> None:
    base = _request(tmp_path)
    request = ProjectIORequest(
        protocol_version=base.protocol_version,
        runtime_id=base.runtime_id,
        executor_epoch=base.executor_epoch,
        request_id=base.request_id,
        operation_kind="authority_service",
        project_id=base.project_id,
        canonical_shared_root=base.canonical_shared_root,
        registration_generation=base.registration_generation,
        registry_revision=base.registry_revision,
        source_revisions={},
        provisional_offer_id=None,
        prepared_at=base.prepared_at,
        parameters={
            "machine_name": "gpu-1",
            "service_action": "observe_current_attempt",
            "task_id": "task-a",
            "attempt_id": "task-a-attempt-1",
            "attempt_number": 1,
            "fencing_token": 1,
            "reservation_id": "reservation-a",
            "process_identity": {
                "wrapper_pid": 101,
                "wrapper_start_time_ticks": 0,
                "process_group_id": 303,
                "process_group_start_time_ticks": 0,
            },
        },
    )
    evidence = {
        "outcome": "observed_current",
        "reason": None,
        "runtime_id": request.runtime_id,
        "executor_epoch": request.executor_epoch,
        "project_id": request.project_id,
        "canonical_shared_root": request.canonical_shared_root,
        "registration_generation": request.registration_generation,
        "registry_revision": request.registry_revision,
        **dict(request.parameters),
        "source_revisions": {"task": 3, "attempt_digest": "d" * 64},
        "attempt_phase": "running",
        "authority_granted": False,
        "local_effects": [],
    }
    result = ProjectIOResult(
        request=request,
        status="completed",
        reason_code=None,
        completed_at="2026-09-28T00:00:01Z",
        evidence=evidence,
    )

    assert ProjectIORequest.from_dict(request.to_dict()) == request
    assert ProjectIOResult.from_dict(result.to_dict()) == result

    for field in ("wrapper_pid", "process_group_id"):
        malformed = request.to_dict()
        malformed["project_io_request"]["parameters"]["process_identity"][field] = 0
        with pytest.raises(ValueError, match="process_identity"):
            ProjectIORequest.from_dict(malformed)

    for field, value in (
        ("attempt_phase", "succeeded"),
        ("source_revisions", {"task": None, "attempt_digest": "d" * 64}),
        ("project_id", "other-project"),
        ("authority_granted", True),
        ("local_effects", ["terminate_process"]),
    ):
        malformed = result.to_dict()
        malformed["project_io_result"]["evidence"][field] = value
        with pytest.raises(ValueError, match="authority|current|local effects"):
            ProjectIOResult.from_dict(malformed)


def test_authority_renewal_protocol_requires_exact_observation_and_no_local_effects(tmp_path: Path) -> None:
    base = _request(tmp_path)
    parameters = {
        "machine_name": "gpu-1",
        "task_id": "task-a",
        "attempt_id": "task-a-attempt-1",
        "attempt_number": 1,
        "fencing_token": 1,
        "reservation_id": "reservation-a",
        "process_identity": {
            "wrapper_pid": 101,
            "wrapper_start_time_ticks": 0,
            "process_group_id": 303,
            "process_group_start_time_ticks": 0,
        },
    }
    source_revisions = {"task": 3, "attempt_digest": "d" * 64}
    request = ProjectIORequest(
        protocol_version=base.protocol_version,
        runtime_id=base.runtime_id,
        executor_epoch=base.executor_epoch,
        request_id=base.request_id,
        operation_kind="authority_renewal",
        project_id=base.project_id,
        canonical_shared_root=base.canonical_shared_root,
        registration_generation=base.registration_generation,
        registry_revision=base.registry_revision,
        source_revisions=source_revisions,
        provisional_offer_id=None,
        prepared_at=base.prepared_at,
        parameters=parameters,
    )
    evidence = {
        "outcome": "renewed",
        "reason": None,
        **parameters,
        "source_revisions": source_revisions,
        "committed_revisions": {"task": 4, "attempt_digest": "e" * 64},
        "lease_expires_at": "2026-09-28T00:02:00Z",
        "renew_after_seconds": 10.0,
        "authority_granted": False,
        "local_effects": [],
    }
    result = ProjectIOResult(
        request=request,
        status="completed",
        reason_code=None,
        completed_at="2026-09-28T00:00:01Z",
        evidence=evidence,
    )

    assert ProjectIORequest.from_dict(request.to_dict()) == request
    assert ProjectIOResult.from_dict(result.to_dict()) == result

    for field in ("wrapper_pid", "process_group_id"):
        malformed = request.to_dict()
        malformed["project_io_request"]["parameters"]["process_identity"][field] = 0
        with pytest.raises(ValueError, match="process_identity"):
            ProjectIORequest.from_dict(malformed)

    malformed = request.to_dict()
    malformed["project_io_request"]["source_revisions"]["attempt_digest"] = "short"
    with pytest.raises(ValueError, match="attempt_digest"):
        ProjectIORequest.from_dict(malformed)

    for field, value in (("authority_granted", True), ("local_effects", ["signal"])):
        malformed_result = result.to_dict()
        malformed_result["project_io_result"]["evidence"][field] = value
        with pytest.raises(ValueError, match="authority|local effects"):
            ProjectIOResult.from_dict(malformed_result)

    for field in ("task", "attempt_digest"):
        malformed_result = result.to_dict()
        malformed_result["project_io_result"]["evidence"]["committed_revisions"][field] = None
        with pytest.raises(ValueError, match="committed revisions"):
            ProjectIOResult.from_dict(malformed_result)

    for cadence in (0, float("nan")):
        malformed_result = result.to_dict()
        malformed_result["project_io_result"]["evidence"]["renew_after_seconds"] = cadence
        with pytest.raises(ValueError, match="renew_after_seconds"):
            ProjectIOResult.from_dict(malformed_result)


def test_scheduler_claim_request_freezes_nested_candidate_and_offer(tmp_path: Path) -> None:
    base = _request(tmp_path)
    candidate = {
        "task_id": "task-a",
        "task_revision": 4,
        "ready_identity": "task-a.2",
        "ready_generation": 2,
        "ready_scope": "home",
        "ready_revision": 3,
        "catalog_revision": 2,
        "catalog_page": 0,
        "partition": "00",
        "marker_name": "task-a.2.json",
        "home_machine": "gpu-1",
        "attempt_number": 1,
        "attempt_id": "task-a-attempt-1",
        "fencing_token": 1,
        "lane": "gpu",
        "requested_gpus": 1,
        "requested_cpus": 0,
        "admission_role": "primary",
        "group_name": None,
        "group_revision": None,
        "group_dispatch_epoch": None,
        "group_worker_set_epoch": None,
        "worker_state_epoch": None,
        "worker_scheduling_role": None,
        "gpu_limit_gpus": None,
    }
    offer = {
        "offer_id": "f" * 16,
        "acquisition_id": "e" * 16,
        "reservation_id": "f" * 16,
        "executor_epoch": base.executor_epoch,
        "request_id": base.request_id,
        "project_id": base.project_id,
        "shared_root": base.canonical_shared_root,
        "registration_generation": base.registration_generation,
        "task_id": "task-a",
        "attempt_id": "task-a-attempt-1",
        "attempt_number": 1,
        "fencing_token": 1,
        "lane": "gpu",
        "gpu_ids": [0],
        "cpu_slots": 0,
        "group_name": None,
        "group_dispatch_epoch": None,
        "group_worker_set_epoch": None,
        "worker_state_epoch": None,
        "worker_scheduling_role": None,
        "gpu_limit_gpus": None,
        "admitted_as_borrow": False,
    }
    request = ProjectIORequest(
        protocol_version=base.protocol_version,
        runtime_id=base.runtime_id,
        executor_epoch=base.executor_epoch,
        request_id=base.request_id,
        operation_kind="scheduler_claim",
        project_id=base.project_id,
        canonical_shared_root=base.canonical_shared_root,
        registration_generation=base.registration_generation,
        registry_revision=base.registry_revision,
        source_revisions=base.source_revisions,
        provisional_offer_id=offer["offer_id"],
        prepared_at=base.prepared_at,
        parameters={
            "machine_name": "gpu-1",
            "lane": "gpu",
            "admission_role": "primary",
            "candidate": candidate,
            "offer": offer,
            "cursor": {
                "namespace": "scheduler-primary-gpu",
                "routes": {
                    "home": {
                        "observed": {
                            "catalog_page": None,
                            "partition": None,
                            "after_name": None,
                            "revision": 0,
                        },
                        "next": {
                            "catalog_page": candidate["catalog_page"],
                            "partition": candidate["partition"],
                            "after_name": candidate["marker_name"],
                            "revision": 1,
                        },
                    },
                    "shared": {
                        "observed": {
                            "catalog_page": None,
                            "partition": None,
                            "after_name": None,
                            "revision": 0,
                        },
                        "next": {
                            "catalog_page": None,
                            "partition": None,
                            "after_name": None,
                            "revision": 0,
                        },
                    },
                },
            },
        },
    )

    candidate["task_revision"] = 99
    offer["gpu_ids"].append(1)

    assert request.parameters["candidate"]["task_revision"] == 4
    assert request.parameters["offer"]["gpu_ids"] == (0,)
    assert ProjectIORequest.from_dict(request.to_dict()) == request
    with pytest.raises(TypeError):
        request.parameters["candidate"]["task_revision"] = 5

    mismatched = request.to_dict()
    mismatched["project_io_request"]["parameters"]["offer"]["fencing_token"] = 2
    with pytest.raises(ValueError, match="offer identity"):
        ProjectIORequest.from_dict(mismatched)

    incomplete_group = request.to_dict()
    incomplete_group["project_io_request"]["parameters"]["candidate"]["group_name"] = "group-a"
    incomplete_group["project_io_request"]["parameters"]["offer"]["group_name"] = "group-a"
    with pytest.raises(ValueError, match="lacks Group"):
        ProjectIORequest.from_dict(incomplete_group)

    wrong_offer_id = request.to_dict()
    wrong_offer_id["project_io_request"]["provisional_offer_id"] = "a" * 32
    with pytest.raises(ValueError, match="provisional_offer_id"):
        ProjectIORequest.from_dict(wrong_offer_id)

    nonadvancing = request.to_dict()
    claim_parameters = nonadvancing["project_io_request"]["parameters"]
    claim_parameters["cursor"]["routes"]["home"]["observed"] = dict(
        claim_parameters["cursor"]["routes"]["home"]["next"]
    )
    with pytest.raises(ValueError, match="cursor does not advance"):
        ProjectIORequest.from_dict(nonadvancing)

    wrong_role = request.to_dict()
    wrong_role["project_io_request"]["parameters"]["candidate"]["admission_role"] = "borrow"
    wrong_role["project_io_request"]["parameters"]["admission_role"] = "borrow"
    with pytest.raises(ValueError, match="ungrouped candidate"):
        ProjectIORequest.from_dict(wrong_role)


def _scheduler_cursor_commit_request(tmp_path: Path) -> ProjectIORequest:
    base = _request(tmp_path)
    unchanged = {"catalog_page": None, "partition": None, "after_name": None, "revision": 0}
    advanced = {"catalog_page": 0, "partition": "00", "after_name": "task-a.1.json", "revision": 1}
    return ProjectIORequest(
        protocol_version=base.protocol_version,
        runtime_id=base.runtime_id,
        executor_epoch=base.executor_epoch,
        request_id=base.request_id,
        operation_kind="scheduler_cursor_commit",
        project_id=base.project_id,
        canonical_shared_root=base.canonical_shared_root,
        registration_generation=base.registration_generation,
        registry_revision=base.registry_revision,
        source_revisions=base.source_revisions,
        provisional_offer_id=None,
        prepared_at=base.prepared_at,
        parameters={
            "machine_name": "gpu-1",
            "cursor": {
                "namespace": "scheduler-primary-gpu",
                "routes": {
                    "home": {"observed": unchanged, "next": advanced},
                    "shared": {"observed": unchanged, "next": unchanged},
                },
            },
        },
    )


def test_scheduler_cursor_commit_protocol_is_strict_monotonic_and_bounded(tmp_path: Path) -> None:
    request = _scheduler_cursor_commit_request(tmp_path)
    assert ProjectIORequest.from_dict(request.to_dict()) == request
    result = ProjectIOResult(
        request=request,
        status="completed",
        reason_code=None,
        completed_at="2026-09-28T00:00:01Z",
        evidence={"routes": {"home": "committed", "shared": "already_applied"}},
    )
    assert ProjectIOResult.from_dict(result.to_dict()) == result

    regressed = request.to_dict()
    regressed["project_io_request"]["parameters"]["cursor"]["routes"]["home"]["next"]["revision"] = 0
    with pytest.raises(ValueError, match="revision|advance|monotonic"):
        ProjectIORequest.from_dict(regressed)

    unknown_outcome = result.to_dict()
    unknown_outcome["project_io_result"]["evidence"]["routes"]["home"] = "mostly_committed"
    with pytest.raises(ValueError, match="route|outcome|evidence"):
        ProjectIOResult.from_dict(unknown_outcome)


def test_scheduler_launch_authorize_protocol_rejects_identity_and_result_drift(tmp_path: Path) -> None:
    base = _request(tmp_path)
    identity = {
        "task_id": "task-a",
        "attempt_id": "task-a-attempt-1",
        "attempt_number": 1,
        "fencing_token": 3,
        "reservation_id": "reservation-a",
    }
    request = ProjectIORequest(
        protocol_version=base.protocol_version,
        runtime_id=base.runtime_id,
        executor_epoch=base.executor_epoch,
        request_id=base.request_id,
        operation_kind="scheduler_launch_authorize",
        project_id=base.project_id,
        canonical_shared_root=base.canonical_shared_root,
        registration_generation=base.registration_generation,
        registry_revision=base.registry_revision,
        source_revisions=base.source_revisions,
        provisional_offer_id=identity["reservation_id"],
        prepared_at=base.prepared_at,
        parameters={"machine_name": "gpu-1", "claim_identity": identity},
    )
    assert ProjectIORequest.from_dict(request.to_dict()) == request

    result = ProjectIOResult(
        request=request,
        status="completed",
        reason_code=None,
        completed_at="2026-09-28T00:00:01Z",
        evidence={
            "outcome": "authorized",
            "claim_identity": identity,
            "launch_id": "d" * 32,
            "launch_handoff_timeout_seconds": 10,
        },
    )
    assert ProjectIOResult.from_dict(result.to_dict()) == result

    mismatched = result.to_dict()
    mismatched["project_io_result"]["evidence"]["claim_identity"]["fencing_token"] = 4
    with pytest.raises(ValueError, match="identity"):
        ProjectIOResult.from_dict(mismatched)

    malformed_launch_id = result.to_dict()
    malformed_launch_id["project_io_result"]["evidence"]["launch_id"] = "not-a-launch-id"
    with pytest.raises(ValueError, match="launch_id"):
        ProjectIOResult.from_dict(malformed_launch_id)

    unknown_reason = ProjectIOResult(
        request=request,
        status="completed",
        reason_code=None,
        completed_at="2026-09-28T00:00:01Z",
        evidence={"outcome": "denied", "claim_identity": identity, "reason": "claim_not_current"},
    ).to_dict()
    unknown_reason["project_io_result"]["evidence"]["reason"] = "maybe"
    with pytest.raises(ValueError, match="reason"):
        ProjectIOResult.from_dict(unknown_reason)


def _reservation_reconcile_request(tmp_path: Path) -> ProjectIORequest:
    base = _request(tmp_path)
    identity = {
        "reservation_id": "reservation-a",
        "acquisition_id": "acquisition-a",
        "project_id": base.project_id,
        "task_id": "task-a",
        "attempt_id": "task-a-attempt-1",
        "fencing_token": 3,
        "gpu_ids": [0],
        "cpu_slots": None,
        "shared_root": base.canonical_shared_root,
        "registration_generation": None,
        "executor_epoch": None,
        "executor_request_id": None,
    }
    return replace(
        base,
        operation_kind="scheduler_reservation_reconcile",
        source_revisions={},
        parameters={"machine_name": "gpu-1", "reservation_identity": identity},
    )


def test_scheduler_reservation_reconcile_protocol_closes_identity_and_authority(tmp_path: Path) -> None:
    request = _reservation_reconcile_request(tmp_path)
    assert ProjectIORequest.from_dict(request.to_dict()) == request
    evidence = {
        "outcome": "retag",
        "reservation_identity": dict(request.parameters["reservation_identity"]),
        "reason": None,
        "target_attempt_id": "task-a-attempt-1",
        "target_fencing_token": 4,
    }
    result = ProjectIOResult(
        request=request,
        status="completed",
        reason_code=None,
        completed_at="2026-09-28T00:00:01Z",
        evidence=evidence,
    )
    assert ProjectIOResult.from_dict(result.to_dict()) == result

    stale_target = result.to_dict()
    stale_target["project_io_result"]["evidence"]["target_fencing_token"] = 3
    with pytest.raises(ValueError, match="retag"):
        ProjectIOResult.from_dict(stale_target)

    release_with_retag_authority = result.to_dict()
    release_evidence = release_with_retag_authority["project_io_result"]["evidence"]
    release_evidence.update(outcome="release", reason="claim_missing")
    with pytest.raises(ValueError, match="release"):
        ProjectIOResult.from_dict(release_with_retag_authority)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gpu_ids", []),
        ("gpu_ids", [0, 0]),
        ("fencing_token", True),
        ("executor_epoch", "epoch-without-other-owner-fields"),
    ],
)
def test_scheduler_reservation_reconcile_rejects_malformed_full_identity(tmp_path: Path, field, value) -> None:
    request = _reservation_reconcile_request(tmp_path).to_dict()
    identity = request["project_io_request"]["parameters"]["reservation_identity"]
    identity[field] = value
    with pytest.raises(ValueError, match="reservation_identity"):
        ProjectIORequest.from_dict(request)


def test_scheduler_reservation_reconcile_accepts_exact_cpu_identity(tmp_path: Path) -> None:
    request = _reservation_reconcile_request(tmp_path).to_dict()
    identity = request["project_io_request"]["parameters"]["reservation_identity"]
    identity.update(gpu_ids=None, cpu_slots=2)
    restored = ProjectIORequest.from_dict(request)
    assert restored.parameters["reservation_identity"]["gpu_ids"] is None
    assert restored.parameters["reservation_identity"]["cpu_slots"] == 2


def test_scheduler_due_offer_protocol_is_closed_and_non_authorizing(tmp_path: Path) -> None:
    request = replace(
        _request(tmp_path),
        operation_kind="scheduler_due_offer",
        source_revisions={},
        parameters={"machine_name": "gpu-1"},
    )
    assert ProjectIORequest.from_dict(request.to_dict()) == request
    result = ProjectIOResult(
        request=request,
        status="completed",
        reason_code=None,
        completed_at="2026-09-28T00:00:01Z",
        evidence={"outcome": "offered", "reason": "offered", "task_id": "task-a"},
    )
    assert ProjectIOResult.from_dict(result.to_dict()) == result

    with pytest.raises(ValueError, match="parameters"):
        replace(request, parameters={"machine_name": "gpu-1", "task_id": "task-a"})
    with pytest.raises(ValueError, match="source_revisions"):
        replace(request, source_revisions={"task": 1})
    with pytest.raises(ValueError, match="outcome and reason"):
        replace(result, evidence={"outcome": "noop", "reason": "offered", "task_id": "task-a"})
    with pytest.raises(ValueError, match="cannot name"):
        replace(
            result,
            evidence={"outcome": "noop", "reason": "no_due_deadline", "task_id": "task-a"},
        )


def test_scheduler_ready_index_build_protocol_is_closed_and_non_authorizing(tmp_path: Path) -> None:
    request = replace(
        _request(tmp_path),
        operation_kind="scheduler_ready_index_build",
        source_revisions={},
        parameters={"machine_name": "gpu-1"},
    )
    assert ProjectIORequest.from_dict(request.to_dict()) == request
    result = ProjectIOResult(
        request=request,
        status="completed",
        reason_code=None,
        completed_at="2026-09-28T00:00:01Z",
        evidence={
            "state": "building",
            "revision": 3,
            "build_id": "a" * 32,
            "phase": "inventory",
        },
    )
    assert ProjectIOResult.from_dict(result.to_dict()) == result
    active = replace(
        result,
        evidence={"state": "active", "revision": 4, "build_id": None, "phase": None},
    )
    assert ProjectIOResult.from_dict(active.to_dict()) == active

    with pytest.raises(ValueError, match="parameters"):
        replace(request, parameters={"machine_name": "gpu-1", "max_tasks": 64})
    with pytest.raises(ValueError, match="source_revisions"):
        replace(request, source_revisions={"ready_index": 1})
    with pytest.raises(ValueError, match="phase"):
        replace(result, evidence={**dict(result.evidence), "phase": "arbitrary"})
    with pytest.raises(ValueError, match="non-building"):
        replace(
            result,
            evidence={"state": "active", "revision": 4, "build_id": "a" * 32, "phase": None},
        )
    with pytest.raises(ValueError, match="requires build identity"):
        replace(
            result,
            evidence={"state": "building", "revision": 4, "build_id": None, "phase": None},
        )


def test_maintenance_descriptor_advance_protocol_is_closed_and_non_authorizing(tmp_path: Path) -> None:
    request = replace(
        _request(tmp_path),
        operation_kind="maintenance_descriptor_advance",
        source_revisions={},
        parameters={"machine_name": "gpu-1"},
    )
    assert ProjectIORequest.from_dict(request.to_dict()) == request
    waiting = ProjectIOResult(
        request=request,
        status="completed",
        reason_code=None,
        completed_at="2026-09-28T00:00:01Z",
        evidence={
            "maintenance_state": "waiting",
            "next_due_at": "2026-09-28T00:00:02Z",
            "more": False,
            "idle_blocking": True,
        },
    )
    assert ProjectIOResult.from_dict(waiting.to_dict()) == waiting
    running = replace(
        waiting,
        evidence={"maintenance_state": "running", "next_due_at": None, "more": True, "idle_blocking": False},
    )
    assert ProjectIOResult.from_dict(running.to_dict()) == running

    with pytest.raises(ValueError, match="parameters"):
        replace(request, parameters={"machine_name": "gpu-1", "max_scan": 1})
    with pytest.raises(ValueError, match="source_revisions"):
        replace(request, source_revisions={"descriptor": 1})
    with pytest.raises(ValueError, match="maintenance_state"):
        replace(waiting, evidence={**dict(waiting.evidence), "maintenance_state": "mostly_done"})
    with pytest.raises(ValueError, match="next_due_at"):
        replace(running, evidence={**dict(running.evidence), "next_due_at": "2026-09-28T00:00:02Z"})
    with pytest.raises(ValueError, match="more"):
        replace(running, evidence={**dict(running.evidence), "more": 1})
    with pytest.raises(ValueError, match="idle_blocking"):
        replace(waiting, evidence={**dict(waiting.evidence), "idle_blocking": 1})
    with pytest.raises(ValueError, match="requires waiting"):
        replace(running, evidence={**dict(running.evidence), "idle_blocking": True})
