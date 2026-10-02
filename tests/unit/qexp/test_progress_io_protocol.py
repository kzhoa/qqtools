from dataclasses import replace

import pytest

from qqtools.plugins.qexp.agent.project_io_protocol import ProjectIORequest, ProjectIOResult


def _context(version):
    context = {
        "protocol_version": version,
        "task_id": "task-a",
        "attempt_id": "attempt-a",
        "attempt_number": 1,
        "machine_name": "gpu-1",
        "launch_id": "launch-a",
        "wrapper_pid": 123,
        "wrapper_start_time_ticks": 456,
        "interval_seconds": 30,
    }
    if version == 1:
        context["reporting_policy_version"] = 1
    return context


def _snapshot(context):
    identity = {
        key: context[key]
        for key in (
            "task_id",
            "attempt_id",
            "attempt_number",
            "machine_name",
            "launch_id",
            "wrapper_pid",
            "wrapper_start_time_ticks",
        )
    }
    progress = {"stage": "train", "current": 1, "total": 10, "unit": "step", "message": None}
    if context["protocol_version"] == 2:
        progress.update(
            {"metrics": {"loss": 0.5}, "completeness": {"complete": True, "omitted_metrics": 0, "reasons": []}}
        )
    return {
        "protocol_version": context["protocol_version"],
        **identity,
        "registration_generation": "generation-a",
        "fencing_token": 1,
        "source_update_id": "update-a",
        "sequence": 1,
        "reported_at": "2026-09-30T00:00:00Z",
        "advanced_at": "2026-09-30T00:00:00Z",
        "progress": progress,
    }


def _request(tmp_path, version, *, publish=False):
    context = _context(version)
    return ProjectIORequest(
        protocol_version=1,
        runtime_id="a" * 64,
        executor_epoch="b" * 32,
        request_id="c" * 32,
        operation_kind="progress_projection",
        project_id="project-a",
        canonical_shared_root=str((tmp_path / "project/.qexp").resolve()),
        registration_generation="generation-a",
        registry_revision=1,
        source_revisions={},
        provisional_offer_id=None,
        prepared_at="2026-09-30T00:00:00Z",
        parameters={"machine_name": "gpu-1", "context": context, "projection": _snapshot(context) if publish else None},
    )


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("publish", [False, True])
def test_progress_protocol_round_trip_exact_identity_and_snapshot(tmp_path, version, publish):
    request = _request(tmp_path, version, publish=publish)
    snapshot = _snapshot(request.parameters["context"])
    binding = {
        key: snapshot[key]
        for key in snapshot
        if key not in {"protocol_version", "source_update_id", "sequence", "reported_at", "advanced_at", "progress"}
    }
    binding["terminal"] = False
    evidence = {"state": "published" if publish else "observed", "binding": binding, "snapshot": snapshot}
    result = ProjectIOResult(request, "completed", None, request.prepared_at, evidence)
    assert ProjectIORequest.from_dict(request.to_dict()) == request
    assert ProjectIOResult.from_dict(result.to_dict()) == result


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize(
    "field,value",
    [
        ("wrapper_pid", True),
        ("wrapper_start_time_ticks", -1),
        ("attempt_number", 0),
        ("machine_name", "other"),
        ("extra", "arbitrary"),
    ],
)
def test_progress_context_is_closed_and_typed(tmp_path, version, field, value):
    request = _request(tmp_path, version)
    parameters = {**request.parameters, "context": {**request.parameters["context"], field: value}}
    with pytest.raises(ValueError):
        replace(request, parameters=parameters)


@pytest.mark.parametrize("changes", [{"source_revisions": {"task": 1}}, {"provisional_offer_id": "offer-a"}])
def test_progress_cannot_transport_authority_or_resource_offer(tmp_path, changes):
    with pytest.raises(ValueError):
        replace(_request(tmp_path, 1), **changes)


@pytest.mark.parametrize("version", [1, 2])
def test_progress_success_cannot_acknowledge_a_different_publication(tmp_path, version):
    request = _request(tmp_path, version, publish=True)
    snapshot = _snapshot(request.parameters["context"])
    binding = {
        key: snapshot[key]
        for key in snapshot
        if key not in {"protocol_version", "source_update_id", "sequence", "reported_at", "advanced_at", "progress"}
    }
    binding["terminal"] = False
    snapshot["sequence"] += 1
    with pytest.raises(ValueError):
        ProjectIOResult(
            request,
            "completed",
            None,
            request.prepared_at,
            {"state": "published", "binding": binding, "snapshot": snapshot},
        )


@pytest.mark.parametrize("state", ["blocked", "retired", "stale"])
def test_progress_unavailable_cannot_return_reusable_identity(tmp_path, state):
    request = _request(tmp_path, 1)
    with pytest.raises(ValueError):
        ProjectIOResult(
            request,
            "completed",
            None,
            request.prepared_at,
            {"state": state, "binding": {"terminal": False}, "snapshot": None},
        )
