"""Deployment-scoped qualification for cross-host filesystem coordination."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config_types import RootConfig
from ..layout import project_id, validate_root_contract
from .paths import shared_paths
from .records import utc_now
from .store import atomic_replace, read_json

FILESYSTEM_QUALIFICATION_VERSION = 1


@dataclass(frozen=True, slots=True)
class FilesystemProbeEvidence:
    """Observed results produced cooperatively by two distinct hosts."""

    initiator_host: str
    peer_host: str
    exclusive_lock: bool
    atomic_replace: bool
    fsync_visibility: bool
    failure_cleanup: bool


@dataclass(frozen=True, slots=True)
class FilesystemQualification:
    is_qualified: bool
    reasons: tuple[str, ...]


def _evidence_document(evidence: FilesystemProbeEvidence) -> dict[str, object]:
    """Normalize invalid probe values into persistable fail-closed evidence."""
    return {
        "initiator_host": (
            evidence.initiator_host.strip()
            if isinstance(evidence.initiator_host, str)
            else None
        ),
        "peer_host": (
            evidence.peer_host.strip()
            if isinstance(evidence.peer_host, str)
            else None
        ),
        "exclusive_lock": evidence.exclusive_lock is True,
        "atomic_replace": evidence.atomic_replace is True,
        "fsync_visibility": evidence.fsync_visibility is True,
        "failure_cleanup": evidence.failure_cleanup is True,
    }


def evaluate_filesystem_probe(evidence: FilesystemProbeEvidence) -> FilesystemQualification:
    """Fail closed unless every cross-host coordination property is proven."""
    reasons: list[str] = []
    if (
        not isinstance(evidence.initiator_host, str)
        or not evidence.initiator_host.strip()
        or not isinstance(evidence.peer_host, str)
        or not evidence.peer_host.strip()
    ):
        reasons.append("both host identities are required")
    elif evidence.initiator_host.strip() == evidence.peer_host.strip():
        reasons.append("probe hosts must be distinct")
    checks = (
        (evidence.exclusive_lock, "cross-host exclusive lock failed"),
        (evidence.atomic_replace, "atomic replace visibility failed"),
        (evidence.fsync_visibility, "fsync durability visibility failed"),
        (evidence.failure_cleanup, "failure cleanup behavior failed"),
    )
    reasons.extend(reason for is_safe, reason in checks if is_safe is not True)
    return FilesystemQualification(not reasons, tuple(reasons))


def filesystem_qualification_path(shared_root: Path) -> Path:
    """Return the deployment qualification record path for one project."""
    return shared_paths(shared_root)["project"] / "filesystem-qualification.json"


def record_filesystem_qualification(
    cfg: RootConfig,
    evidence: FilesystemProbeEvidence,
) -> FilesystemQualification:
    """Persist the latest evidence produced by a two-host deployment probe.

    Args:
        cfg: Initialized qexp project configuration.
        evidence: Results observed cooperatively by two distinct hosts.

    Returns:
        The successful qualification decision.

    Raises:
        RuntimeError: If the qexp project is not initialized.
        ValueError: If the persisted evidence does not prove every required property.
    """
    validate_root_contract(cfg)
    qualification = evaluate_filesystem_probe(evidence)
    atomic_replace(
        filesystem_qualification_path(cfg.shared_root),
        {
            "filesystem_qualification": {
                "version": FILESYSTEM_QUALIFICATION_VERSION,
                "project_id": project_id(cfg.shared_root),
                "shared_root": str(cfg.shared_root),
                "evidence": _evidence_document(evidence),
                "recorded_at": utc_now(),
            }
        },
    )
    if not qualification.is_qualified:
        raise ValueError(
            "cross-host filesystem qualification failed: "
            + "; ".join(qualification.reasons)
        )
    return qualification


def load_filesystem_qualification(cfg: RootConfig) -> FilesystemQualification:
    """Load and validate the qualification bound to this project deployment."""
    path = filesystem_qualification_path(cfg.shared_root)
    if not path.exists():
        return FilesystemQualification(False, ("qualification evidence is missing",))
    try:
        record = read_json(path)["filesystem_qualification"]
        if not isinstance(record, dict) or set(record) != {
            "version",
            "project_id",
            "shared_root",
            "evidence",
            "recorded_at",
        }:
            raise TypeError
        if record.get("version") != FILESYSTEM_QUALIFICATION_VERSION:
            return FilesystemQualification(
                False, ("qualification record version is unsupported",)
            )
        if record.get("project_id") != project_id(cfg.shared_root):
            return FilesystemQualification(False, ("qualification project identity mismatches",))
        if record.get("shared_root") != str(cfg.shared_root):
            return FilesystemQualification(False, ("qualification shared root mismatches",))
        if not isinstance(record.get("recorded_at"), str) or not record["recorded_at"]:
            raise TypeError
        raw_evidence = record["evidence"]
        if not isinstance(raw_evidence, dict) or set(raw_evidence) != {
            "initiator_host",
            "peer_host",
            "exclusive_lock",
            "atomic_replace",
            "fsync_visibility",
            "failure_cleanup",
        }:
            raise TypeError
        evidence = FilesystemProbeEvidence(**raw_evidence)
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return FilesystemQualification(False, ("qualification record is malformed",))
    return evaluate_filesystem_probe(evidence)


def require_cross_host_filesystem_qualification(cfg: RootConfig) -> None:
    """Fail closed unless this shared-root deployment has qualified evidence."""
    qualification = load_filesystem_qualification(cfg)
    if not qualification.is_qualified:
        raise RuntimeError(
            "cross-host scheduling requires qualified shared-filesystem evidence: "
            + "; ".join(qualification.reasons)
        )


def validate_existing_filesystem_qualification(cfg: RootConfig) -> None:
    """Reject a present but invalid qualification record during setup."""
    if not filesystem_qualification_path(cfg.shared_root).exists():
        return
    qualification = load_filesystem_qualification(cfg)
    invalid_record_reasons = {
        "qualification record version is unsupported",
        "qualification project identity mismatches",
        "qualification shared root mismatches",
        "qualification record is malformed",
    }
    if invalid_record_reasons.intersection(qualification.reasons):
        raise RuntimeError(
            "shared-filesystem qualification record is invalid: "
            + "; ".join(qualification.reasons)
        )
