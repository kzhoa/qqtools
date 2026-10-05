from __future__ import annotations

from dataclasses import replace

import pytest

from qqtools.plugins.qexp.agent.project_io_termination_convergence import (
    TerminationConvergenceEvidence,
    classify_termination_convergence,
)


def _settled_evidence(exit_code: object = -15) -> TerminationConvergenceEvidence:
    return TerminationConvergenceEvidence(
        decision_matches=True,
        decision_state="confirmed",
        shared_commitment="committed",
        confirmation="identity_absent",
        manifest_state="exited",
        manifest_exit_code=exit_code,
        manifest_exited_at_valid=True,
        process_absent=True,
        registration_matches=True,
        exit_observation_state="matching" if exit_code is not None else "absent",
        reservation_state="released",
    )


@pytest.mark.parametrize("exit_code", [None, 0, 7, -15])
def test_convergence_accepts_exact_process_exit_results(exit_code: int | None) -> None:
    result = classify_termination_convergence(_settled_evidence(exit_code))

    assert result.state == "converged"
    assert result.reason == "termination_converged"
    assert result.manifest_exit_code == exit_code
    assert result.process_absent is True
    assert result.reservation_settled is True


@pytest.mark.parametrize("exit_code", [True, 1.5, "-15", [], {}])
def test_convergence_rejects_non_integer_process_exit_results(exit_code: object) -> None:
    result = classify_termination_convergence(_settled_evidence(exit_code))

    assert result.state == "invalid"
    assert result.reason == "termination_exit_code_invalid"
    assert result.manifest_exit_code is None
    assert result.reservation_settled is False


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"decision_matches": False}, "termination_decision_invalid"),
        ({"process_absent": False}, "termination_process_identity_present"),
        ({"registration_matches": False}, "termination_registration_mismatch"),
        ({"exit_observation_state": "mismatch"}, "termination_exit_observation_mismatch"),
        ({"exit_observation_state": "invalid"}, "termination_exit_observation_mismatch"),
        ({"reservation_state": "conflict"}, "termination_reservation_conflict"),
    ],
)
def test_convergence_rejects_contradictory_exact_evidence(
    change: dict[str, object],
    reason: str,
) -> None:
    result = classify_termination_convergence(replace(_settled_evidence(), **change))

    assert result.state == "invalid"
    assert result.reason == reason


@pytest.mark.parametrize("state", ["active", "provisional", "release_pair"])
def test_confirmed_decision_reports_repairable_reservation_release(state: str) -> None:
    result = classify_termination_convergence(replace(_settled_evidence(), reservation_state=state))

    assert result.state == "repairable"
    assert result.reason == "termination_reservation_release_pending"
    assert result.reservation_settled is False


@pytest.mark.parametrize("decision_state", ["pending", "signal_committed", "sigterm_sent", "sigkill_sent"])
def test_unconfirmed_decision_cannot_claim_local_effect_authority(decision_state: str) -> None:
    result = classify_termination_convergence(
        replace(
            _settled_evidence(),
            decision_state=decision_state,
            confirmation=None,
            manifest_state="running",
            manifest_exit_code=None,
            manifest_exited_at_valid=False,
            process_absent=False,
            exit_observation_state="absent",
            reservation_state="active",
        )
    )

    assert result.state == "repairable"
    assert result.reason == "termination_decision_progress_pending"
    assert result.process_absent is False
    assert result.reservation_settled is False


def test_confirmed_decision_can_repair_incomplete_manifest_without_losing_exit_result() -> None:
    result = classify_termination_convergence(
        replace(
            _settled_evidence(),
            manifest_state="running",
            manifest_exited_at_valid=False,
        )
    )

    assert result.state == "repairable"
    assert result.reason == "termination_manifest_completion_pending"
    assert result.manifest_exit_code == -15


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"registration_matches": False}, "termination_registration_mismatch"),
        ({"exit_observation_state": "mismatch"}, "termination_exit_observation_mismatch"),
        ({"reservation_state": "conflict"}, "termination_reservation_conflict"),
    ],
)
def test_incomplete_manifest_cannot_bypass_contradictory_evidence(
    change: dict[str, object],
    reason: str,
) -> None:
    result = classify_termination_convergence(
        replace(
            _settled_evidence(None),
            manifest_state="running",
            manifest_exited_at_valid=False,
            **change,
        )
    )

    assert result.state == "invalid"
    assert result.reason == reason


def test_incomplete_manifest_can_use_valid_exit_observation_as_repair_source() -> None:
    result = classify_termination_convergence(
        replace(
            _settled_evidence(None),
            manifest_state="running",
            manifest_exited_at_valid=False,
            exit_observation_state="available",
        )
    )

    assert result.state == "repairable"
    assert result.reason == "termination_manifest_completion_pending"


def test_null_exit_result_does_not_override_process_presence() -> None:
    result = classify_termination_convergence(
        replace(
            _settled_evidence(None),
            process_absent=False,
        )
    )

    assert result.state == "invalid"
    assert result.reason == "termination_process_identity_present"
