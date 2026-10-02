"""Recovery residency preserves finite work and admits beyond-window bindings."""

import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from qqtools.plugins.qexp.agent.bindings import ProjectBinding
from qqtools.plugins.qexp.agent.recovery_enrollment import RecoveryEnrollment, _Capture


def test_65th_binding_cannot_evict_unfinished_work_but_can_replace_a_parked_owner(monkeypatch):
    bindings = tuple(
        ProjectBinding.from_canonical_paths(
            project_id=f"project-{index}",
            shared_root=Path(f"/projects/{index}"),
            machine_name="gpu-1",
            registration_generation=f"generation-{index}",
            runtime_instance_id="runtime",
            runtime_root="/machine",
        )
        for index in range(65)
    )
    runtime = SimpleNamespace(
        _scheduler_authority_pid=os.getpid(),
        load_registry_snapshot=lambda: (0, bindings),
    )
    controller = SimpleNamespace(executor=SimpleNamespace(unresolved_requests=lambda: ()))
    for name in (
        "advance_recovery_admission",
        "advance_recovery_source_holds",
        "advance_legacy_capture_scans",
        "advance_legacy_capture_reads",
        "advance_recovery_capture_transitions",
        "advance_recovery_group_authority",
        "advance_recovery_source_releases",
    ):
        setattr(controller, name, lambda *_args: {})
    enrollment = RecoveryEnrollment(runtime)
    monkeypatch.setattr(enrollment, "_owned", lambda *_args: nullcontext())
    for _ in range(34):
        enrollment.advance(controller, bindings, 0)
        assert len(enrollment._captures) <= 64
    assert set(enrollment._captures) == set(bindings[:64])
    enrollment._captures[bindings[0]].is_parked = True
    for _ in range(17):
        enrollment.advance(controller, bindings, 0)
    assert bindings[-1] in enrollment._captures
    assert bindings[0] not in enrollment._captures
    assert len(enrollment._captures) == 64
    enrollment.stop()
    assert not enrollment._captures


def test_stale_domain_result_discards_cached_authority_and_restarts_from_durable_admission():
    binding = ProjectBinding.from_canonical_paths("project", Path("/project"), "gpu-1")
    enrollment = RecoveryEnrollment(SimpleNamespace())
    closed = []
    capture = _Capture(
        binding,
        operation="recovery_source_hold",
        parameters={"capture_id": "obsolete"},
        processes=SimpleNamespace(close=lambda: closed.append("processes")),
        evidence=SimpleNamespace(close=lambda: closed.append("evidence")),
        proof={"obsolete": True},
    )
    enrollment._apply(capture, {"state": "stale", "hold": None})
    assert closed == ["processes", "evidence"]
    assert capture.is_parked
    assert capture.operation == "recovery_admission" and capture.parameters == {}
    assert capture.processes is None and capture.evidence is None and capture.proof is None
