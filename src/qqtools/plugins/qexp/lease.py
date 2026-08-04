"""Authoritative lease policy, clock health, and structured lease outcomes."""
from __future__ import annotations

import random
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from .config_types import RootConfig
from .runtime.paths import local_paths, shared_paths
from .runtime.store import atomic_replace, read_json


class LeaseRenewalOutcome(str, Enum):
    RENEWED = "renewed"
    RETRYABLE_ERROR = "retryable_error"
    AUTHORITY_CHANGED = "authority_changed"
    ORPHANED_RECOVERY_REQUIRED = "orphaned_recovery_required"
    TERMINATION_REQUESTED = "termination_requested"
    INVALID_PROCESS_EVIDENCE = "invalid_process_evidence"


class AuthorityResolutionOutcome(str, Enum):
    RENEWED = "renewed"
    RECOVERED = "recovered"
    TERMINATION_REQUIRED = "termination_required"
    AUTHORITY_UNAVAILABLE = "authority_unavailable"
    QUARANTINE_REQUIRED = "quarantine_required"


@dataclass(frozen=True, slots=True)
class LeaseFailureDetails:
    error_type: str
    message: str
    errno: int | None = None


@dataclass(frozen=True, slots=True)
class LeaseRenewalResult:
    outcome: LeaseRenewalOutcome
    attempt_id: str
    observed_token: int | None = None
    lease_expires_at: str | None = None
    error: LeaseFailureDetails | None = None


@dataclass(frozen=True, slots=True)
class AuthorityResolution:
    outcome: AuthorityResolutionOutcome
    decision_id: str
    attempt_id: str
    previous_token: int
    effective_token: int | None = None
    lease_expires_at: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class LeasePolicy:
    ttl_seconds: int = 120
    renew_interval_seconds: float = 10.0
    retry_initial_seconds: float = 0.25
    retry_max_seconds: float = 5.0
    retry_jitter_ratio: float = 0.20
    max_clock_skew_seconds: float = 1.0
    renewal_commit_margin_seconds: float = 5.0
    clock_health_max_age_seconds: float = 30.0
    lease_loss_action: str = "isolate"

    def __post_init__(self) -> None:
        if self.ttl_seconds <= 0 or self.renew_interval_seconds <= 0:
            raise ValueError("lease TTL and renewal interval must be positive.")
        if self.renew_interval_seconds >= self.ttl_seconds:
            raise ValueError("renew interval must be less than the lease TTL.")
        if self.retry_initial_seconds <= 0 or self.retry_max_seconds < self.retry_initial_seconds:
            raise ValueError("lease retry bounds are invalid.")
        if not 0 <= self.retry_jitter_ratio <= 1:
            raise ValueError("retry jitter ratio must be between zero and one.")
        if self.max_clock_skew_seconds <= 0 or self.renewal_commit_margin_seconds <= 0:
            raise ValueError("clock skew and renewal commit margin must be positive.")
        minimum = 2 * self.max_clock_skew_seconds + self.renewal_commit_margin_seconds + self.retry_max_seconds
        if self.ttl_seconds <= minimum:
            raise ValueError("lease TTL has no safe clock and retry budget.")
        if self.lease_loss_action != "isolate":
            raise ValueError("only lease_loss_action='isolate' is supported.")

    def retry_delay(self, failure_count: int) -> float:
        capped = min(self.retry_max_seconds, self.retry_initial_seconds * 2 ** max(0, failure_count - 1))
        spread = capped * self.retry_jitter_ratio
        return max(0.0, capped + random.uniform(-spread, spread))


def default_lease_policy_document() -> dict[str, Any]:
    return {"lease_policy": asdict(LeasePolicy())}


def load_lease_policy(cfg: RootConfig) -> LeasePolicy:
    value = read_json(shared_paths(cfg.shared_root)["lease_policy"]).get("lease_policy")
    if not isinstance(value, dict):
        raise RuntimeError("qexp lease policy is malformed.")
    return LeasePolicy(**value)


def save_lease_policy(cfg: RootConfig, policy: LeasePolicy) -> None:
    atomic_replace(shared_paths(cfg.shared_root)["lease_policy"], {"lease_policy": asdict(policy)})


def lease_expiry(policy: LeasePolicy) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=policy.ttl_seconds)).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z")


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def holder_safe_deadline(expires_at: str, policy: LeasePolicy) -> datetime:
    return parse_utc(expires_at) - timedelta(seconds=policy.max_clock_skew_seconds)


def reclaim_allowed_at(expires_at: str, policy: LeasePolicy) -> datetime:
    return parse_utc(expires_at) + timedelta(seconds=policy.max_clock_skew_seconds)


_OFFSET_PATTERN = re.compile(r"^System time\s*:\s*([+-]?[0-9.]+)\s+seconds", re.MULTILINE)
_DISPERSION_PATTERN = re.compile(r"^Root dispersion\s*:\s*([0-9.]+)\s+seconds", re.MULTILINE)
_LEAP_PATTERN = re.compile(r"^Leap status\s*:\s*(.+)$", re.MULTILINE)


def chrony_health(policy: LeasePolicy, *, run: Callable[..., Any] = subprocess.run) -> tuple[bool, str]:
    """Return whether chrony supplies the configured bounded-clock guarantee."""
    try:
        completed = run(["chronyc", "tracking", "-n"], check=False, capture_output=True,
                        text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"chrony_unavailable:{type(exc).__name__}"
    if completed.returncode:
        return False, f"chrony_exit:{completed.returncode}"
    offset = _OFFSET_PATTERN.search(completed.stdout)
    dispersion = _DISPERSION_PATTERN.search(completed.stdout)
    leap = _LEAP_PATTERN.search(completed.stdout)
    if not offset or not dispersion or not leap or leap.group(1).strip() != "Normal":
        return False, "chrony_tracking_incomplete"
    if abs(float(offset.group(1))) > policy.max_clock_skew_seconds:
        return False, "chrony_offset_exceeds_policy"
    if float(dispersion.group(1)) > policy.max_clock_skew_seconds:
        return False, "chrony_dispersion_exceeds_policy"
    return True, "healthy"


def authority_clock_health(cfg: RootConfig, policy: LeasePolicy | None = None) -> tuple[bool, str]:
    """Return a bounded-age local clock-health observation for authority operations."""
    policy = policy or load_lease_policy(cfg)
    path = local_paths(cfg.runtime_root)["clock_health"]
    now = datetime.now(timezone.utc)
    try:
        cached = read_json(path).get("clock_health", {})
        checked_at = parse_utc(cached["checked_at"])
        age_seconds = (now - checked_at).total_seconds()
        if 0 <= age_seconds <= policy.clock_health_max_age_seconds:
            return bool(cached.get("healthy")), str(cached.get("reason", "missing_reason"))
    except (KeyError, OSError, TypeError, ValueError):
        pass
    healthy, reason = chrony_health(policy)
    try:
        atomic_replace(path, {"clock_health": {
            "healthy": healthy,
            "reason": reason,
            "checked_at": now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        }})
    except OSError:
        return False, "clock_health_persistence_failed"
    return healthy, reason
