from pathlib import Path

import pytest

from qqtools.plugins.qexp.agent.project_io_protocol import ProjectIORequest, ProjectIOResult


def _request(tmp_path: Path) -> ProjectIORequest:
    return ProjectIORequest(
        protocol_version=1,
        runtime_id="a" * 64,
        executor_epoch="b" * 32,
        request_id="c" * 32,
        operation_kind="observation_service",
        project_id="project-a",
        canonical_shared_root=str((tmp_path / "project/.qexp").resolve()),
        registration_generation="registration-a",
        registry_revision=1,
        source_revisions={},
        provisional_offer_id=None,
        prepared_at="2026-09-30T00:00:00Z",
        parameters={"machine_name": "gpu-1"},
    )


@pytest.mark.parametrize(
    "state,reason",
    [
        ("active", "idle"),
        ("active", "progress"),
        ("building", "progress"),
        ("degraded", "blocked"),
        ("waiting", "blocked"),
        ("closed", "closed"),
    ],
)
def test_observation_service_has_closed_typed_summary(tmp_path, state, reason):
    request = _request(tmp_path)
    result = ProjectIOResult(
        request,
        "completed",
        None,
        "2026-09-30T00:00:01Z",
        {"state": state, "quiescent": reason == "idle", "reason_code": reason},
    )
    assert ProjectIORequest.from_dict(request.to_dict()) == request
    assert ProjectIOResult.from_dict(result.to_dict()) == result


@pytest.mark.parametrize(
    "field,value",
    [
        ("state", "unknown"),
        ("quiescent", 1),
        ("quiescent", "true"),
        ("reason_code", "unknown"),
        ("reason_code", "blocked"),
        ("state", "building"),
        ("state", "degraded"),
        ("state", "closed"),
        ("arbitrary_payload", "ignored"),
    ],
)
def test_observation_service_rejects_invalid_or_unproven_idle_evidence(tmp_path, field, value):
    evidence = {"state": "active", "quiescent": True, "reason_code": "idle"}
    evidence[field] = value
    with pytest.raises(ValueError):
        ProjectIOResult(_request(tmp_path), "completed", None, "2026-09-30T00:00:01Z", evidence)


def test_observation_service_rejects_arbitrary_parameters_and_revisions(tmp_path):
    request = _request(tmp_path)
    for key, value in (("parameters", {"machine_name": "gpu-1", "path": "/tmp"}), ("source_revisions", {"task": 1})):
        record = request.to_dict()
        record["project_io_request"][key] = value
        with pytest.raises(ValueError):
            ProjectIORequest.from_dict(record)
