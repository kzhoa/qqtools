"""Pure decision contract for a deployment-run two-host filesystem probe."""

from __future__ import annotations

from dataclasses import dataclass


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
