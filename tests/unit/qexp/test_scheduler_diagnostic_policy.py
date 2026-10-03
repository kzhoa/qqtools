from __future__ import annotations

from collections.abc import Mapping, Sequence

from qqtools.plugins.qexp.agent.scheduler_diagnostic_policy import (
    FindingObservationIntent,
    FindingObservationOmission,
    IgnoredFindingObservation,
    canonicalize_diagnostic_identity,
    iter_publication_probes,
    resolution_queries,
    select_resolution_candidate,
)


def _decision_identity(*, producer: str = "primary_probe", lane: str = "gpu") -> dict[str, object]:
    return {
        "producer": producer,
        "reason_code": "primary_demand",
        "component": "scheduler",
        "stage": "admission",
        "check": "primary_demand",
        "scope_type": "machine_lane",
        "runtime_id": "a" * 64,
        "resource_lane": lane,
    }


def _finding_identity(
    *,
    producer: str = "primary_probe",
    project_id: str = "project-a",
    registration_generation: str = "registration-1",
    lane: str = "gpu",
) -> dict[str, object]:
    return {
        "producer": producer,
        "reason_code": "ready_index_unreadable",
        "component": "scheduler",
        "stage": "admission",
        "check": "primary_demand",
        "scope_type": "project_route",
        "runtime_id": "a" * 64,
        "project_id": project_id,
        "registration_generation": registration_generation,
        "resource_lane": lane,
        "route_scope": "shared",
    }


def _complete_probe(
    *,
    producer: str = "primary_probe",
    lane: str = "gpu",
    registry_revision: int = 2,
    findings: Sequence[object] = (),
    covered_bindings: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    return {
        "identity": _decision_identity(producer=producer, lane=lane),
        "outcome": "no_primary_demand",
        "coverage": "complete",
        "source_revision": {"registry_revision": registry_revision},
        "covered_bindings": covered_bindings,
        "findings": findings,
    }


def _active_item(
    *,
    producer: str = "primary_probe",
    project_id: str = "project-a",
    registration_generation: str = "registration-1",
    lane: str = "gpu",
    registry_revision: int = 1,
    observation_revision: int = 3,
) -> dict[str, object]:
    identity, incomplete = canonicalize_diagnostic_identity(
        _finding_identity(
            producer=producer,
            project_id=project_id,
            registration_generation=registration_generation,
            lane=lane,
        )
    )
    assert not incomplete
    return {
        "identity": identity,
        "episode": "b" * 32,
        "observation_revision": observation_revision,
        "source_revision": {"registry_revision": registry_revision},
    }


def _active_view(*items: Mapping[str, object], coverage: str = "complete") -> dict[str, object]:
    return {"coverage": coverage, "items": items}


def test_publication_policy_preserves_order_defaults_and_explicit_malformed_finding() -> None:
    malformed = {"identity": _finding_identity()}
    valid = {
        "identity": _finding_identity(project_id="project-b"),
        "source_revision": {"registry_revision": 4},
    }
    probes = iter_publication_probes(
        (
            "ignored",
            {
                "identity": _decision_identity(),
                "coverage": 7,
                "findings": ("ignored", malformed, valid),
            },
        )
    )

    probe = next(probes)

    assert probe.coverage_complete is False
    assert probe.decision is not None
    assert probe.decision.outcome == "unknown_error"
    assert probe.decision.coverage == "unknown"
    findings = list(probe.findings)
    assert findings == [
        IgnoredFindingObservation(),
        FindingObservationOmission(reason="identity_incomplete"),
        FindingObservationIntent(
            identity=valid["identity"],
            severity="fault",
            source_revision=valid["source_revision"],
            details=None,
        ),
    ]


def test_publication_policy_stops_after_first_three_mapping_probes() -> None:
    raw = tuple({"identity": _decision_identity(lane=f"gpu_{index}"), "coverage": "complete"} for index in range(4))

    selected = list(iter_publication_probes(raw))

    assert len(selected) == 3
    assert [probe.decision.identity["resource_lane"] for probe in selected if probe.decision is not None] == [
        "gpu_0",
        "gpu_1",
        "gpu_2",
    ]


def test_resolution_policy_rejects_incomplete_and_malformed_probe_evidence() -> None:
    malformed_identity = _complete_probe()
    malformed_identity["identity"] = "not-a-mapping"
    missing_revision = _complete_probe()
    missing_revision["source_revision"] = None
    incomplete = _complete_probe()
    incomplete["coverage"] = "incomplete"

    assert resolution_queries((malformed_identity, missing_revision, incomplete)) == ()


def test_resolution_policy_prefers_enablement_and_selects_exact_candidate() -> None:
    binding = {"project_id": "project-a", "registration_generation": "registration-1"}
    enablement = _complete_probe(
        producer="enablement_reconciliation",
        registry_revision=8,
        covered_bindings=(binding,),
    )
    primary = _complete_probe(registry_revision=8, covered_bindings=(binding,))

    queries = resolution_queries((primary, enablement))
    candidate = select_resolution_candidate(
        queries[0],
        _active_view(_active_item(producer="enablement_reconciliation", registry_revision=7)),
    )

    assert [query.producer for query in queries] == ["enablement_reconciliation", "primary_probe"]
    assert candidate is not None
    assert candidate.identity["producer"] == "enablement_reconciliation"
    assert candidate.episode == "b" * 32
    assert candidate.observation_revision == 3
    assert candidate.source_revision == {"registry_revision": 7}


def test_resolution_policy_keeps_current_finding_and_rejects_newer_source_revision() -> None:
    identity = _finding_identity()
    current = {
        "identity": identity,
        "source_revision": {"registry_revision": 2},
    }
    binding = {"project_id": "project-a", "registration_generation": "registration-1"}
    current_query = resolution_queries(
        (_complete_probe(findings=(current,), covered_bindings=(binding,), registry_revision=2),)
    )[0]
    stale_query = resolution_queries((_complete_probe(covered_bindings=(binding,), registry_revision=1),))[0]

    assert select_resolution_candidate(current_query, _active_view(_active_item(registry_revision=1))) is None
    assert select_resolution_candidate(stale_query, _active_view(_active_item(registry_revision=2))) is None


def test_malformed_revision_probe_still_excludes_its_current_finding() -> None:
    identity = _finding_identity()
    malformed_current = _complete_probe(
        findings=({"identity": identity},),
        covered_bindings=(),
    )
    malformed_current["source_revision"] = None
    binding = {"project_id": "project-a", "registration_generation": "registration-1"}
    authorizing = _complete_probe(covered_bindings=(binding,), registry_revision=3)

    query = resolution_queries((malformed_current, authorizing))[0]

    assert select_resolution_candidate(query, _active_view(_active_item(registry_revision=2))) is None


def test_resolution_policy_requires_exact_lane_and_replacement_binding_generation() -> None:
    binding = {"project_id": "project-a", "registration_generation": "registration-2"}
    wrong_lane_query = resolution_queries(
        (_complete_probe(lane="cpu", covered_bindings=(binding,), registry_revision=3),)
    )[0]
    replacement_query = resolution_queries((_complete_probe(covered_bindings=(binding,), registry_revision=3),))[0]
    active = _active_view(_active_item(registration_generation="registration-1", registry_revision=2))

    assert select_resolution_candidate(wrong_lane_query, active) is None
    assert select_resolution_candidate(replacement_query, active) is None


def test_resolution_policy_uses_last_complete_probe_for_a_lane() -> None:
    old_binding = {"project_id": "project-a", "registration_generation": "registration-1"}
    replacement_binding = {"project_id": "project-a", "registration_generation": "registration-2"}
    queries = resolution_queries(
        (
            _complete_probe(covered_bindings=(old_binding,), registry_revision=2),
            _complete_probe(covered_bindings=(replacement_binding,), registry_revision=3),
        )
    )

    assert (
        select_resolution_candidate(
            queries[0],
            _active_view(_active_item(registration_generation="registration-1", registry_revision=1)),
        )
        is None
    )
