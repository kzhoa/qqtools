from dataclasses import replace
from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp.agent.project_io_protocol import ProjectIORequest, ProjectIOResult
from qqtools.plugins.qexp.runtime.responsibility_source_scan import initial_source_scan_cursor


def request(tmp_path):
    return ProjectIORequest(
        protocol_version=1,
        runtime_id="a" * 64,
        executor_epoch="b" * 32,
        request_id="c" * 32,
        operation_kind="legacy_capture_read",
        project_id="project-a",
        canonical_shared_root=str(tmp_path / "shared"),
        registration_generation="generation-a",
        registry_revision=1,
        source_revisions={},
        provisional_offer_id=None,
        prepared_at="2026-09-30T00:00:00Z",
        parameters={
            "machine_name": "gpu-1",
            "capture_id": "d" * 32,
            "backfill_id": "e" * 32,
            "backfill_revision": 1,
            "lane": "observations",
            "relative": "task-attempt-1.json",
        },
    )


def locator(tmp_path):
    return {
        "source_root": str(tmp_path / "source"),
        "lane": "observations",
        "relative": "task-attempt-1.json",
        "identity": "task-attempt-1",
        "task_id": "task",
        "attempt_number": 1,
    }


def hold_request(tmp_path):
    prepared = request(tmp_path)
    return replace(
        prepared,
        operation_kind="recovery_source_hold",
        parameters={"machine_name": prepared.parameters["machine_name"], "capture_id": "d" * 32},
    )


def hold_record(tmp_path):
    return {
        "format": "qexp-pending-writer-source-v1",
        "runtime_root": str(tmp_path / "source"),
        "target_root": str(tmp_path / "target"),
        "instance": "e" * 32,
        "capture_id": "d" * 32,
        "phase": "pending",
    }


def admission_request(tmp_path):
    prepared = request(tmp_path)
    return replace(
        prepared, operation_kind="recovery_admission", parameters={"machine_name": prepared.parameters["machine_name"]}
    )


def scan_request(tmp_path):
    prepared = request(tmp_path)
    parameters = {key: value for key, value in prepared.parameters.items() if key != "relative"}
    return replace(
        prepared,
        operation_kind="legacy_capture_scan",
        parameters={**parameters, "cursor": initial_source_scan_cursor()},
    )


@pytest.mark.parametrize(
    "operation,completed", [("recovery_source_release", "released"), ("recovery_group_authority", "active")]
)
@pytest.mark.parametrize("state", ["completed", "waiting", "stale"])
def test_recovery_completion_round_trip_is_non_authorizing(tmp_path, operation, completed, state):
    prepared = replace(
        request(tmp_path), operation_kind=operation, parameters={"machine_name": "gpu-1", "completion_digest": "d" * 64}
    )
    result = ProjectIOResult(
        prepared, "completed", None, prepared.prepared_at, {"state": completed if state == "completed" else state}
    )
    assert ProjectIORequest.from_dict(prepared.to_dict()) == prepared
    assert ProjectIOResult.from_dict(result.to_dict()) == result


@pytest.mark.parametrize("operation", ["recovery_source_release", "recovery_group_authority"])
@pytest.mark.parametrize("digest", ["d" * 32, "D" * 64, "../source", True])
def test_recovery_completion_rejects_weak_or_arbitrary_proof(tmp_path, operation, digest):
    with pytest.raises(ValueError):
        replace(
            request(tmp_path),
            operation_kind=operation,
            parameters={"machine_name": "gpu-1", "completion_digest": digest},
        )


@pytest.mark.parametrize("phase,state", [("retain", "retained"), ("normalize", "normalized")])
def test_capture_transition_protocol_preserves_its_source_phase(tmp_path, phase, state):
    prepared = replace(
        request(tmp_path),
        operation_kind="recovery_capture_transition",
        parameters={"machine_name": "gpu-1", "completion_digest": "d" * 64, "phase": phase},
    )
    result = ProjectIOResult(prepared, "completed", None, prepared.prepared_at, {"state": state})
    assert ProjectIORequest.from_dict(prepared.to_dict()) == prepared
    assert ProjectIOResult.from_dict(result.to_dict()) == result
    with pytest.raises(ValueError):
        ProjectIOResult(
            prepared,
            "completed",
            None,
            prepared.prepared_at,
            {"state": "normalized" if phase == "retain" else "retained"},
        )


@pytest.mark.parametrize("state", ["observed", "stale"])
def test_source_scan_protocol_round_trip(tmp_path, state):
    prepared = scan_request(tmp_path)
    scan = {
        "pending": ["task-attempt-1.json"],
        "cursor": initial_source_scan_cursor(),
        "at_end": False,
        "entries_visited": 1,
    }
    result = ProjectIOResult(
        prepared,
        "completed",
        None,
        prepared.prepared_at,
        {"state": state, "scan": scan if state == "observed" else None},
    )
    assert ProjectIORequest.from_dict(prepared.to_dict()) == prepared
    assert ProjectIOResult.from_dict(result.to_dict()) == result


@pytest.mark.parametrize(
    "key,value",
    [
        ("offset", -1),
        ("offset", True),
        ("child", "../other"),
        ("child", "/other"),
        ("directory_identity", [1]),
        ("child_offset", 1),
        ("source_root", "/arbitrary"),
    ],
)
def test_source_scan_rejects_arbitrary_or_inconsistent_cursors(tmp_path, key, value):
    prepared = scan_request(tmp_path)
    with pytest.raises(ValueError):
        replace(prepared, parameters={**prepared.parameters, "cursor": {**prepared.parameters["cursor"], key: value}})


@pytest.mark.parametrize(
    "key,value",
    [
        ("pending", ["../other.json"]),
        ("pending", ["task-attempt-1.json"] * 2),
        ("entries_visited", 65),
        ("entries_visited", True),
        ("at_end", 1),
        ("secret", "private"),
    ],
)
def test_source_scan_rejects_unbounded_or_malformed_evidence(tmp_path, key, value):
    prepared = scan_request(tmp_path)
    scan = {"pending": [], "cursor": initial_source_scan_cursor(), "at_end": False, "entries_visited": 2}
    with pytest.raises(ValueError):
        ProjectIOResult(
            prepared, "completed", None, prepared.prepared_at, {"state": "observed", "scan": {**scan, key: value}}
        )


@pytest.mark.parametrize(
    "state,prepared,fenced",
    [("ready", True, True), ("waiting", False, False), ("waiting", True, False), ("stale", False, False)],
)
def test_recovery_admission_protocol_separates_preparation_from_fencing(tmp_path, state, prepared, fenced):
    request = admission_request(tmp_path)
    evidence = {"state": state, "registration_prepared": prepared, "admission_fenced": fenced}
    result = ProjectIOResult(request, "completed", None, request.prepared_at, evidence)
    assert ProjectIORequest.from_dict(request.to_dict()) == request
    assert ProjectIOResult.from_dict(result.to_dict()) == result


@pytest.mark.parametrize(
    "state,prepared,fenced",
    [
        ("ready", False, True),
        ("ready", True, False),
        ("waiting", True, True),
        ("stale", True, False),
        ("ready", 1, True),
        ("unknown", False, False),
    ],
)
def test_recovery_admission_protocol_rejects_false_readiness(tmp_path, state, prepared, fenced):
    request = admission_request(tmp_path)
    with pytest.raises(ValueError):
        ProjectIOResult(
            request,
            "completed",
            None,
            request.prepared_at,
            {"state": state, "registration_prepared": prepared, "admission_fenced": fenced},
        )


@pytest.mark.parametrize("state", ["retained", "stale"])
def test_source_hold_protocol_round_trip(tmp_path, state):
    prepared = hold_request(tmp_path)
    evidence = {"state": state, "hold": hold_record(tmp_path) if state == "retained" else None}
    result = ProjectIOResult(prepared, "completed", None, prepared.prepared_at, evidence)
    assert ProjectIORequest.from_dict(prepared.to_dict()) == prepared
    assert ProjectIOResult.from_dict(result.to_dict()) == result


@pytest.mark.parametrize("key,value", [("source_root", "/arbitrary"), ("capture_id", True), ("capture_id", "invalid")])
def test_source_hold_request_cannot_invent_source_or_identity(tmp_path, key, value):
    prepared = hold_request(tmp_path)
    with pytest.raises(ValueError):
        replace(prepared, parameters={**prepared.parameters, key: value})


@pytest.mark.parametrize(
    "key,value",
    [
        ("runtime_root", "/"),
        ("runtime_root", "/source/../other"),
        ("runtime_root", "/source//other"),
        ("target_root", "/"),
        ("capture_id", "f" * 32),
        ("instance", "unknown"),
        ("instance", True),
        ("phase", "complete"),
        ("format", "unknown"),
        ("env", {"SECRET": "must-not-cross"}),
    ],
)
def test_source_hold_result_is_closed_and_exact(tmp_path, key, value):
    prepared = hold_request(tmp_path)
    with pytest.raises(ValueError):
        ProjectIOResult(
            prepared,
            "completed",
            None,
            prepared.prepared_at,
            {"state": "retained", "hold": {**hold_record(tmp_path), key: value}},
        )


def test_stale_source_hold_result_does_not_grant_retention(tmp_path):
    prepared = hold_request(tmp_path)
    with pytest.raises(ValueError):
        ProjectIOResult(
            prepared, "completed", None, prepared.prepared_at, {"state": "stale", "hold": hold_record(tmp_path)}
        )


@pytest.mark.parametrize("state", ["observed", "stale"])
def test_source_read_round_trip(tmp_path, state):
    prepared = request(tmp_path)
    evidence = {"state": state, "locator": locator(tmp_path) if state == "observed" else None}
    result = ProjectIOResult(prepared, "completed", None, prepared.prepared_at, evidence)
    assert ProjectIORequest.from_dict(prepared.to_dict()) == prepared
    assert ProjectIOResult.from_dict(result.to_dict()) == result


@pytest.mark.parametrize(
    "key,value",
    [
        ("source_root", "/arbitrary"),
        ("capture_id", "invalid"),
        ("backfill_revision", True),
        ("backfill_revision", 0),
        ("lane", []),
        ("lane", "unknown"),
        ("relative", "../task-attempt-1.json"),
        ("relative", "/task-attempt-1.json"),
        ("relative", "task.json/other.json"),
    ],
)
def test_source_read_rejects_unbounded_or_unretained_requests(tmp_path, key, value):
    prepared = request(tmp_path)
    with pytest.raises((ValueError, RuntimeError)):
        replace(prepared, parameters={**prepared.parameters, key: value})


@pytest.mark.parametrize(
    "key,value",
    [
        ("source_root", "/"),
        ("source_root", "/source/../other"),
        ("source_root", "/source//other"),
        ("identity", "other-attempt-1"),
        ("lane", "processes"),
        ("relative", "other-attempt-1.json"),
        ("task_id", "other"),
        ("attempt_number", True),
        ("attempt_number", 2),
        ("credentials", "must-not-cross-boundary"),
    ],
)
def test_source_read_rejects_forged_or_payload_bearing_results(tmp_path, key, value):
    prepared = request(tmp_path)
    evidence = {"state": "observed", "locator": {**locator(tmp_path), key: value}}
    with pytest.raises((ValueError, RuntimeError)):
        ProjectIOResult(prepared, "completed", None, prepared.prepared_at, evidence)


def test_stale_source_read_cannot_return_identity(tmp_path):
    prepared = request(tmp_path)
    with pytest.raises(ValueError):
        ProjectIOResult(
            prepared, "completed", None, prepared.prepared_at, {"state": "stale", "locator": locator(tmp_path)}
        )


@pytest.mark.parametrize("transition", ["unchanged", "batch_changed", "completed", "epoch_fenced", "source_missing"])
def test_source_worker_rechecks_capture_and_epoch_after_read(tmp_path, monkeypatch, transition):
    from qqtools.plugins.qexp.agent import project_io_worker as worker
    from qqtools.plugins.qexp.agent.recovery_transport import decode_capture_locator
    from qqtools.plugins.qexp.runtime import responsibility_source_read as source_read

    prepared = request(tmp_path)
    context = source_read.SourceReadContext(tmp_path / "source", {"capture": "retained"}, {"revision": 1})
    captured = decode_capture_locator(locator(tmp_path), prepared.parameters)
    calls = []

    def load_context(root, owner, **parameters):
        calls.append("context")
        assert owner["owner_instance"] == prepared.runtime_id
        assert parameters["backfill_id"] == prepared.parameters["backfill_id"]
        if calls.count("context") == 2:
            if transition == "completed":
                raise source_read.SourceCaptureStale("complete")
            if transition == "batch_changed":
                return replace(context, backfill={"revision": 2})
        return context

    def read_source(selected, lane, relative):
        calls.append("read")
        assert selected == context
        if transition == "source_missing":
            raise FileNotFoundError("retained source disappeared")
        return captured

    def epoch(_paths):
        return SimpleNamespace(
            active=not (transition == "epoch_fenced" and "read" in calls),
            runtime_id=prepared.runtime_id,
            executor_epoch=prepared.executor_epoch,
        )

    monkeypatch.setattr(worker, "load_root_config", lambda *_: object())
    monkeypatch.setattr(worker, "_read_epoch", epoch)
    monkeypatch.setattr(worker, "_claim_binding_is_current", lambda *_: True)
    monkeypatch.setattr(source_read, "load_source_read_context", load_context)
    monkeypatch.setattr(source_read, "read_retained_source_locator", read_source)
    if transition == "epoch_fenced":
        with pytest.raises(worker._ExecutorEpochFenced):
            worker._legacy_capture_read(prepared, tmp_path / "machine", {})
    elif transition == "source_missing":
        with pytest.raises(FileNotFoundError):
            worker._legacy_capture_read(prepared, tmp_path / "machine", {})
    else:
        result = worker._legacy_capture_read(prepared, tmp_path / "machine", {})
        assert result == (
            {"state": "observed", "locator": locator(tmp_path)}
            if transition == "unchanged"
            else {"state": "stale", "locator": None}
        )
        assert calls == ["context", "read", "context"]
