"""Confirmation identity and Group approvals for a Project takeover preview."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from ..runtime.records import validate_identifier
from ..runtime.takeover_registration import RegistrationWitness
from ..takeover_contract import TAKEOVER_PLAN_ID

_NONCE_HEX_LENGTH = 24


@dataclass(frozen=True, slots=True)
class TakeoverGroupApproval:
    """Describe one primary-worker addition and the exact Group impact approved."""

    action: str
    group_name: str
    machine_name: str
    group_revision: int
    worker_set_epoch: int
    affected_task_count: int
    affected_tasks_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.action, str) or self.action != "add_primary_worker":
            raise ValueError("action must be 'add_primary_worker'.")
        validate_identifier(self.group_name, "group_name")
        validate_identifier(self.machine_name, "machine_name")
        if type(self.group_revision) is not int or self.group_revision < 0:
            raise ValueError("group_revision must be a nonnegative integer.")
        if type(self.worker_set_epoch) is not int or self.worker_set_epoch < 0:
            raise ValueError("worker_set_epoch must be a nonnegative integer.")
        if type(self.affected_task_count) is not int or self.affected_task_count < 1:
            raise ValueError("affected_task_count must be a positive integer.")
        if not isinstance(self.affected_tasks_digest, str) or not TAKEOVER_PLAN_ID.fullmatch(
            self.affected_tasks_digest
        ):
            raise ValueError("affected_tasks_digest must be 64 lowercase hexadecimal characters.")

    @classmethod
    def from_dict(cls, value: object) -> TakeoverGroupApproval:
        """Validate the exact fields of a serialized Group approval."""
        if not isinstance(value, dict) or value.keys() != {field.name for field in fields(cls)}:
            raise ValueError("Project takeover Group approval has an invalid field set.")
        return cls(**value)

    def to_dict(self) -> dict[str, str | int]:
        """Serialize the Group approval fields for the plan or its journal."""
        return asdict(self)


def takeover_plan_nonce(plan_id: str) -> str:
    """Validate the plan ID format and recover its original preview nonce."""
    if not isinstance(plan_id, str) or not TAKEOVER_PLAN_ID.fullmatch(plan_id):
        raise ValueError("Project takeover plan ID must be 64 lowercase hexadecimal characters.")
    return plan_id[:_NONCE_HEX_LENGTH]


def takeover_plan_id(
    *,
    project_id: str,
    shared_root: Path,
    runtime_id: str,
    runtime_root: Path,
    target: RegistrationWitness,
    departing: RegistrationWitness | None,
    original_enabled: bool,
    groups: tuple[TakeoverGroupApproval, ...],
    nonce: str | None = None,
) -> str:
    """Fingerprint confirmation inputs with a fresh or supplied preview nonce.

    The caller supplies canonical Project and runtime roots. The fingerprint
    covers the target witness in full and tolerates routine renewal of the
    departing registration's eligibility. Use the ID to check that confirmation
    inputs are current.
    """
    if nonce is None:
        nonce = secrets.token_hex(12)
    elif not isinstance(nonce, str) or not re.fullmatch(r"[0-9a-f]{24}", nonce):
        raise ValueError("Project takeover plan nonce must be 24 lowercase hexadecimal characters.")

    if len({group.group_name for group in groups}) != len(groups):
        raise ValueError("Project takeover plan contains duplicate Group approvals.")
    departing_fields = None
    if departing is not None:
        departing_fields = departing.to_dict()
        del departing_fields["eligibility_expires_at"]

    payload = {
        "version": 1,
        "purpose": "qexp-project-takeover",
        "nonce": nonce,
        "project_id": project_id,
        "shared_root": str(shared_root),
        "runtime_id": runtime_id,
        "runtime_root": str(runtime_root),
        "target": target.to_dict(),
        "departing": departing_fields,
        "original_enabled": original_enabled,
        "groups": [group.to_dict() for group in sorted(groups, key=lambda group: group.group_name)],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return nonce + hashlib.sha256(encoded).hexdigest()[:40]
