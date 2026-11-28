"""Wire contracts for due-offer and ready-index scheduling operations."""

from typing import Any, Mapping

from .fields import _require_exact_keys, _require_identifier

OPERATIONS = frozenset({"scheduler_due_offer", "scheduler_ready_index_build"})


def validate_parameters(kind: str, parameters: dict[str, Any]) -> None:
    _require_exact_keys(parameters, frozenset({"machine_name"}), f"{kind} parameters")
    parameters["machine_name"] = _require_identifier(parameters["machine_name"], "machine_name")


def validate_source(kind: str, revisions: Mapping[str, Any], offer_id: str | None) -> None:
    _require_exact_keys(revisions, frozenset(), f"{kind} source_revisions")
    if offer_id is not None:
        raise ValueError(f"{kind} cannot name a provisional offer.")


def validate_due_offer_evidence(evidence: dict[str, Any]) -> None:
    _require_exact_keys(
        evidence,
        frozenset({"outcome", "reason", "task_id"}),
        "scheduler_due_offer evidence",
    )
    reasons = {
        "offered",
        "no_due_deadline",
        "task_missing",
        "task_not_queued",
        "claim_active",
        "not_home_machine",
        "deadline_not_due",
        "elapsed_offer_unproven",
        "offer_rejected",
    }
    if evidence["outcome"] not in {"offered", "noop"} or evidence["reason"] not in reasons:
        raise ValueError("scheduler_due_offer outcome or reason is invalid.")
    if (evidence["outcome"] == "offered") != (evidence["reason"] == "offered"):
        raise ValueError("scheduler_due_offer outcome and reason do not agree.")
    task_id = evidence["task_id"]
    if evidence["reason"] == "no_due_deadline":
        if task_id is not None:
            raise ValueError("scheduler_due_offer empty result cannot name a Task.")
    else:
        evidence["task_id"] = _require_identifier(task_id, "scheduler_due_offer task_id")


def validate_ready_index_build_evidence(evidence: dict[str, Any]) -> None:
    _require_exact_keys(
        evidence,
        frozenset({"state", "revision", "build_id", "phase"}),
        "scheduler_ready_index_build evidence",
    )
    if evidence["state"] not in {"absent", "building", "active", "degraded"}:
        raise ValueError("scheduler_ready_index_build state is invalid.")
    revision = evidence["revision"]
    if type(revision) is not int or revision < 0:
        raise ValueError("scheduler_ready_index_build revision is invalid.")
    build_id = evidence["build_id"]
    phase = evidence["phase"]
    if build_id is not None:
        build_id = _require_identifier(build_id, "scheduler_ready_index_build build_id")
    if phase is not None and phase not in {
        "reset-projection",
        "inventory",
        "backfill",
        "audit",
        "primary-rebuild",
    }:
        raise ValueError("scheduler_ready_index_build phase is invalid.")
    if evidence["state"] != "building" and (build_id is not None or phase is not None):
        raise ValueError("scheduler_ready_index_build non-building evidence cannot name a build.")
    if evidence["state"] == "building" and (build_id is None or phase is None):
        raise ValueError("scheduler_ready_index_build building evidence requires build identity.")
    evidence["build_id"] = build_id
    evidence["phase"] = phase
