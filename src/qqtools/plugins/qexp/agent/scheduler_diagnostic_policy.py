"""Pure producer policy for the machine-local scheduler diagnostic store."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_MAX_INTEGER = (1 << 63) - 1
_BASE_IDENTITY_FIELDS = ("producer", "reason_code", "component", "stage", "check", "scope_type")
_IDENTITY_FIELDS = frozenset(
    {
        *_BASE_IDENTITY_FIELDS,
        "runtime_id",
        "project_id",
        "registration_generation",
        "task_id",
        "task_generation",
        "resource_lane",
        "route_scope",
        "admission_role",
        "projection_kind",
        "projection_target",
        "projection_generation",
        "build_id",
        "work_generation",
        "maintenance_kind",
        "target_id",
        "wake_epoch",
        "sequence_start",
        "sequence_end",
        "registry_revision",
        "ready_revision",
    }
)


@dataclass(frozen=True, slots=True)
class DecisionObservationIntent:
    """Normalized producer decision data awaiting store admission."""

    identity: Mapping[str, object]
    outcome: str
    capacity_context: Mapping[str, object] | None
    blocker_identities: Sequence[Mapping[str, object]] | None
    coverage: str
    source_revision: Mapping[str, object] | None
    details: Mapping[str, object] | None


@dataclass(frozen=True, slots=True)
class FindingObservationIntent:
    """Normalized producer finding data awaiting store admission."""

    identity: Mapping[str, object]
    severity: str
    source_revision: Mapping[str, object]
    details: Mapping[str, object] | None


@dataclass(frozen=True, slots=True)
class FindingObservationOmission:
    """Explicitly omitted mapping finding with incomplete observation identity."""

    reason: str


@dataclass(frozen=True, slots=True)
class IgnoredFindingObservation:
    """A raw non-mapping finding retained for publication-cap ordering."""


@dataclass(frozen=True, slots=True)
class PublicationProbe:
    """One bounded producer probe and its lazy publication intents."""

    coverage_complete: bool
    decision: DecisionObservationIntent | None
    findings: Iterable[FindingObservationIntent | FindingObservationOmission | IgnoredFindingObservation]

    @property
    def decision_intent(self) -> DecisionObservationIntent | None:
        """Return the normalized decision intent, if the probe has one."""
        return self.decision


@dataclass(frozen=True, slots=True)
class ResolutionQuery:
    """One ordered producer query plan for bounded active-store reconciliation."""

    producer: str
    probes: tuple[Mapping[str, object], ...]
    current_finding_digests: frozenset[str]


@dataclass(frozen=True, slots=True)
class ResolutionCandidate:
    """Exact active finding evidence selected for store-side revalidation."""

    identity: Mapping[str, object]
    episode: str
    observation_revision: int
    source_revision: Mapping[str, object]


def _safe_token(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    token = value.strip().lower()
    return token if _TOKEN_RE.fullmatch(token) else None


def _safe_identifier(value: object) -> str | None:
    return value if isinstance(value, str) and _IDENTIFIER_RE.fullmatch(value) else None


def canonicalize_diagnostic_identity(value: Mapping[str, object]) -> tuple[dict[str, Any], bool]:
    """Return the canonical allowlisted identity and whether input was incomplete."""
    safe: dict[str, Any] = {}
    omitted = False
    numeric_keys = {"sequence_start", "sequence_end", "registry_revision", "ready_revision", "task_generation"}
    token_keys = set(_BASE_IDENTITY_FIELDS) | {
        "resource_lane",
        "route_scope",
        "admission_role",
        "projection_kind",
        "maintenance_kind",
    }
    for key, raw in value.items():
        if key not in _IDENTITY_FIELDS:
            omitted = True
            continue
        if key in numeric_keys:
            if type(raw) is int and 0 <= raw <= _MAX_INTEGER:
                safe[key] = raw
            else:
                omitted = True
            continue
        sanitized = _safe_token(raw) if key in token_keys else _safe_identifier(raw)
        if sanitized is None:
            omitted = True
        else:
            safe[key] = sanitized
    for key in _BASE_IDENTITY_FIELDS:
        if key not in safe:
            safe[key] = "identity_incomplete"
            omitted = True
    return dict(sorted(safe.items())), omitted


def diagnostic_identity_digest(identity: Mapping[str, Any]) -> str:
    """Return the stable digest used by active and decision record paths."""
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _iter_probe_mappings(
    probes: Sequence[Mapping[str, object]] | Mapping[str, object] | None,
) -> Iterator[Mapping[str, object]]:
    if probes is None:
        return
    values = probes.values() if isinstance(probes, Mapping) else probes
    accepted = 0
    for value in values:
        if not isinstance(value, Mapping):
            continue
        yield value
        accepted += 1
        if accepted >= 3:
            break


def _iter_finding_intents(
    raw_findings: object,
) -> Iterator[FindingObservationIntent | FindingObservationOmission | IgnoredFindingObservation]:
    if not isinstance(raw_findings, Sequence) or isinstance(raw_findings, (str, bytes)):
        return
    for finding in raw_findings:
        if not isinstance(finding, Mapping):
            yield IgnoredFindingObservation()
            continue
        identity = finding.get("identity")
        source_revision = finding.get("source_revision")
        if not isinstance(identity, Mapping) or not isinstance(source_revision, Mapping):
            yield FindingObservationOmission(reason="identity_incomplete")
            continue
        yield FindingObservationIntent(
            identity=identity,
            severity=finding.get("severity") if isinstance(finding.get("severity"), str) else "fault",
            source_revision=source_revision,
            details=finding.get("details") if isinstance(finding.get("details"), Mapping) else None,
        )


def _publication_probe(probe: Mapping[str, object]) -> PublicationProbe:
    decision_identity = probe.get("identity")
    decision = None
    if isinstance(decision_identity, Mapping):
        blocker_identities = probe.get("blocker_identities")
        decision = DecisionObservationIntent(
            identity=decision_identity,
            outcome=probe.get("outcome") if isinstance(probe.get("outcome"), str) else "unknown_error",
            capacity_context=probe.get("capacity_context")
            if isinstance(probe.get("capacity_context"), Mapping)
            else None,
            blocker_identities=blocker_identities
            if isinstance(blocker_identities, Sequence) and not isinstance(blocker_identities, (str, bytes))
            else None,
            coverage=probe.get("coverage") if isinstance(probe.get("coverage"), str) else "unknown",
            source_revision=probe.get("source_revision") if isinstance(probe.get("source_revision"), Mapping) else None,
            details=probe.get("details") if isinstance(probe.get("details"), Mapping) else None,
        )
    return PublicationProbe(
        coverage_complete=probe.get("coverage") == "complete",
        decision=decision,
        findings=_iter_finding_intents(probe.get("findings", ())),
    )


def iter_publication_probes(
    probes: Sequence[Mapping[str, object]] | Mapping[str, object] | None,
) -> Iterator[PublicationProbe]:
    """Lazily normalize at most the first three mapping probes."""
    for probe in _iter_probe_mappings(probes):
        yield _publication_probe(probe)


def _valid_resolution_probe(probe: Mapping[str, object]) -> bool:
    return (
        probe.get("coverage") == "complete"
        and isinstance(probe.get("identity"), Mapping)
        and isinstance(probe.get("source_revision"), Mapping)
    )


def _iter_probe_findings(probe: Mapping[str, object]) -> Iterator[Mapping[str, object]]:
    findings = probe.get("findings", ())
    if not isinstance(findings, Sequence) or isinstance(findings, (str, bytes)):
        return
    for finding in findings:
        if isinstance(finding, Mapping):
            yield finding


def resolution_queries(
    probes: Sequence[Mapping[str, object]] | Mapping[str, object] | None,
) -> tuple[ResolutionQuery, ...]:
    """Build ordered bounded query plans from complete producer evidence."""
    normalized = tuple(_iter_probe_mappings(probes))
    complete_lanes: list[Mapping[str, object]] = []
    complete_enablement: list[Mapping[str, object]] = []
    current_digests: set[str] = set()
    for probe in normalized:
        identity = probe.get("identity")
        if probe.get("coverage") != "complete" or not isinstance(identity, Mapping):
            continue
        for finding in _iter_probe_findings(probe):
            finding_identity = finding.get("identity")
            if not isinstance(finding_identity, Mapping):
                continue
            safe_identity, incomplete = canonicalize_diagnostic_identity(finding_identity)
            if not incomplete:
                current_digests.add(diagnostic_identity_digest(safe_identity))
        if not isinstance(probe.get("source_revision"), Mapping):
            continue
        lane = identity.get("resource_lane")
        runtime_id = identity.get("runtime_id")
        if isinstance(lane, str) and isinstance(runtime_id, str):
            complete_lanes.append(probe)
        if identity.get("producer") == "enablement_reconciliation" and isinstance(runtime_id, str):
            complete_enablement.append(probe)
    plans: list[ResolutionQuery] = []
    if complete_enablement:
        plans.append(
            ResolutionQuery(
                producer="enablement_reconciliation",
                probes=tuple(complete_enablement),
                current_finding_digests=frozenset(current_digests),
            )
        )
    if complete_lanes:
        plans.append(
            ResolutionQuery(
                producer="primary_probe",
                probes=tuple(complete_lanes),
                current_finding_digests=frozenset(current_digests),
            )
        )
    return tuple(plans)


def _revision_covers(covered: object, observed: object) -> bool:
    if not isinstance(covered, Mapping) or not isinstance(observed, Mapping):
        return False
    for key, observed_value in observed.items():
        covered_value = covered.get(key)
        if type(observed_value) is int:
            if type(covered_value) is not int or covered_value < observed_value:
                return False
        elif covered_value != observed_value:
            return False
    return True


def _probe_covers_binding(
    probe: Mapping[str, object],
    *,
    project_id: str,
    registration_generation: object,
) -> bool:
    covered = probe.get("covered_bindings")
    if not isinstance(covered, Sequence) or isinstance(covered, (str, bytes)):
        return False
    return any(
        isinstance(binding, Mapping)
        and binding.get("project_id") == project_id
        and binding.get("registration_generation") == registration_generation
        for binding in covered
    )


def _candidate_from_item(item: Mapping[str, object]) -> ResolutionCandidate | None:
    identity = item.get("identity")
    episode = item.get("episode")
    observation_revision = item.get("observation_revision")
    source_revision = item.get("source_revision")
    if (
        not isinstance(identity, Mapping)
        or not isinstance(episode, str)
        or type(observation_revision) is not int
        or not isinstance(source_revision, Mapping)
    ):
        return None
    return ResolutionCandidate(identity, episode, observation_revision, source_revision)


def _matching_probe(
    query: ResolutionQuery,
    *,
    lane: str | None,
) -> Mapping[str, object] | None:
    if query.producer == "enablement_reconciliation":
        return None
    selected: Mapping[str, object] | None = None
    for probe in query.probes:
        if not _valid_resolution_probe(probe):
            continue
        identity = probe["identity"]
        if lane is not None and identity.get("resource_lane") == lane:
            selected = probe
    return selected


def _matches_candidate(
    item: Mapping[str, object],
    probe: Mapping[str, object],
) -> ResolutionCandidate | None:
    identity = item.get("identity")
    decision_identity = probe.get("identity")
    if not isinstance(identity, Mapping) or not isinstance(decision_identity, Mapping):
        return None
    if identity.get("runtime_id") != decision_identity.get("runtime_id"):
        return None
    source_revision = item.get("source_revision")
    if not isinstance(source_revision, Mapping) or not _revision_covers(probe.get("source_revision"), source_revision):
        return None
    project_id = identity.get("project_id")
    if isinstance(project_id, str) and not _probe_covers_binding(
        probe,
        project_id=project_id,
        registration_generation=identity.get("registration_generation"),
    ):
        return None
    return _candidate_from_item(item)


def select_resolution_candidate(
    query: ResolutionQuery,
    active_view: Mapping[str, object],
) -> ResolutionCandidate | None:
    """Select the first active item eligible under an ordered pure query plan."""
    if active_view.get("coverage") not in {"complete", "incomplete"}:
        return None
    items = active_view.get("items", ())
    if not isinstance(items, Iterable) or isinstance(items, (str, bytes)):
        return None
    for item in items:
        if not isinstance(item, Mapping):
            continue
        identity = item.get("identity")
        if not isinstance(identity, Mapping):
            continue
        if diagnostic_identity_digest(identity) in query.current_finding_digests:
            continue
        if query.producer == "enablement_reconciliation":
            for probe in query.probes:
                if not _valid_resolution_probe(probe):
                    continue
                candidate = _matches_candidate(item, probe)
                if candidate is not None:
                    return candidate
            continue
        if query.producer != "primary_probe":
            continue
        lane = identity.get("resource_lane")
        if not isinstance(lane, str):
            continue
        probe = _matching_probe(query, lane=lane)
        if probe is None:
            continue
        candidate = _matches_candidate(item, probe)
        if candidate is not None:
            return candidate
    return None


__all__ = [
    "DecisionObservationIntent",
    "FindingObservationIntent",
    "FindingObservationOmission",
    "IgnoredFindingObservation",
    "PublicationProbe",
    "ResolutionCandidate",
    "ResolutionQuery",
    "canonicalize_diagnostic_identity",
    "diagnostic_identity_digest",
    "iter_publication_probes",
    "resolution_queries",
    "select_resolution_candidate",
]
