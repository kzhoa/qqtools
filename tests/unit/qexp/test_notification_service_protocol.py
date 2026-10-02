from dataclasses import replace
from pathlib import Path

import pytest

from qqtools.plugins.qexp.agent.project_io_protocol import ProjectIORequest, ProjectIOResult


def _request(tmp_path: Path) -> ProjectIORequest:
    return ProjectIORequest(
        protocol_version=1,
        runtime_id="a" * 64,
        executor_epoch="b" * 32,
        request_id="c" * 32,
        operation_kind="notification_service",
        project_id="project-a",
        canonical_shared_root=str((tmp_path / "project/.qexp").resolve()),
        registration_generation="registration-a",
        registry_revision=1,
        source_revisions={},
        provisional_offer_id=None,
        prepared_at="2026-09-30T00:00:00Z",
        parameters={"machine_name": "gpu-1"},
    )


@pytest.mark.parametrize("state", ["ready", "conflict", "source_invalid", "blocked"])
def test_notification_service_closed_summary_round_trip(tmp_path, state):
    request = _request(tmp_path)
    result = ProjectIOResult(request, "completed", None, request.prepared_at, {"state": state})
    assert ProjectIORequest.from_dict(request.to_dict()) == request
    assert ProjectIOResult.from_dict(result.to_dict()) == result


@pytest.mark.parametrize(
    "evidence",
    [{}, {"state": "unknown"}, {"state": "ready", "webhook": "secret"}, {"state": True}],
)
def test_notification_service_rejects_open_or_invalid_results(tmp_path, evidence):
    request = _request(tmp_path)
    with pytest.raises(ValueError):
        ProjectIOResult(request, "completed", None, request.prepared_at, evidence)


@pytest.mark.parametrize(
    "changes",
    [
        {"parameters": {"machine_name": "gpu-1", "webhook": "secret"}},
        {"parameters": {"machine_name": "gpu-1", "path": "/tmp"}},
        {"source_revisions": {"task": 1}},
        {"provisional_offer_id": "offer-a"},
    ],
)
def test_notification_service_rejects_credentials_paths_and_authority(tmp_path, changes):
    with pytest.raises(ValueError):
        replace(_request(tmp_path), **changes)
