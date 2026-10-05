"""Pure classification of durable local termination convergence evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TerminationConvergenceState = Literal["converged", "repairable", "invalid"]


@dataclass(frozen=True, slots=True)
class TerminationConvergenceEvidence:
    """Bounded raw evidence used to classify one exact termination Attempt."""

    decision_matches: bool
    decision_state: object
    shared_commitment: object
    confirmation: object
    manifest_state: object
    manifest_exit_code: object
    manifest_exited_at_valid: bool
    process_absent: bool
    registration_matches: bool
    exit_observation_state: object
    reservation_state: object


@dataclass(frozen=True, slots=True)
class TerminationConvergence:
    """Immutable result of one complete termination evidence classification."""

    state: TerminationConvergenceState
    reason: str
    manifest_exit_code: int | None
    process_absent: bool
    reservation_settled: bool


_NON_CONFIRMED_STATES = frozenset({"pending", "signal_committed", "sigterm_sent", "sigkill_sent"})
_OBSERVATION_STATES = frozenset({"absent", "available", "matching", "mismatch", "invalid"})
_RESERVATION_STATES = frozenset({"unassigned", "released", "active", "provisional", "release_pair", "conflict"})


def _invalid(
    reason: str, *, manifest_exit_code: int | None = None, process_absent: bool = False
) -> TerminationConvergence:
    return TerminationConvergence("invalid", reason, manifest_exit_code, process_absent, False)


def _valid_exit_code(value: object) -> tuple[bool, int | None]:
    if value is None:
        return True, None
    if type(value) is int:
        return True, value
    return False, None


def classify_termination_convergence(evidence: TerminationConvergenceEvidence) -> TerminationConvergence:
    """Classify complete, successfully-read termination evidence deterministically."""
    if not isinstance(evidence, TerminationConvergenceEvidence):
        return _invalid("termination_decision_invalid")
    if type(evidence.decision_matches) is not bool or not evidence.decision_matches:
        return _invalid("termination_decision_invalid")

    decision_state = evidence.decision_state
    if decision_state in _NON_CONFIRMED_STATES:
        return TerminationConvergence(
            "repairable",
            "termination_decision_progress_pending",
            None,
            bool(evidence.process_absent) if type(evidence.process_absent) is bool else False,
            False,
        )
    if decision_state != "confirmed":
        return _invalid("termination_decision_invalid")

    if evidence.shared_commitment != "committed":
        return _invalid("termination_decision_invalid")
    confirmation = evidence.confirmation
    if type(evidence.process_absent) is not bool:
        return _invalid("termination_decision_invalid")
    if not evidence.process_absent or confirmation in {"identity_present", "process_identity_present"}:
        return _invalid("termination_process_identity_present", process_absent=False)
    if confirmation not in {"identity_absent", "process_absent"}:
        return _invalid("termination_decision_invalid", process_absent=True)

    valid_exit_code, exit_code = _valid_exit_code(evidence.manifest_exit_code)
    if not valid_exit_code:
        return _invalid("termination_exit_code_invalid", process_absent=True)

    if type(evidence.registration_matches) is not bool or not evidence.registration_matches:
        return _invalid("termination_registration_mismatch", manifest_exit_code=exit_code, process_absent=True)

    observation_state = evidence.exit_observation_state
    if observation_state not in _OBSERVATION_STATES:
        return _invalid("termination_exit_observation_mismatch", manifest_exit_code=exit_code, process_absent=True)
    if observation_state in {"mismatch", "invalid"}:
        return _invalid("termination_exit_observation_mismatch", manifest_exit_code=exit_code, process_absent=True)

    reservation_state = evidence.reservation_state
    if reservation_state not in _RESERVATION_STATES or reservation_state == "conflict":
        return _invalid("termination_reservation_conflict", manifest_exit_code=exit_code, process_absent=True)

    manifest_state = evidence.manifest_state
    if manifest_state not in {"running", "exited"}:
        return _invalid("termination_decision_invalid", manifest_exit_code=exit_code, process_absent=True)
    if type(evidence.manifest_exited_at_valid) is not bool:
        return _invalid("termination_decision_invalid", manifest_exit_code=exit_code, process_absent=True)
    if manifest_state == "exited" and evidence.manifest_exited_at_valid and observation_state == "available":
        return _invalid("termination_exit_observation_mismatch", manifest_exit_code=exit_code, process_absent=True)
    if manifest_state != "exited" or not evidence.manifest_exited_at_valid:
        return TerminationConvergence(
            "repairable",
            "termination_manifest_completion_pending",
            exit_code,
            True,
            False,
        )

    if reservation_state in {"active", "provisional", "release_pair"}:
        return TerminationConvergence(
            "repairable",
            "termination_reservation_release_pending",
            exit_code,
            True,
            False,
        )
    return TerminationConvergence("converged", "termination_converged", exit_code, True, True)
