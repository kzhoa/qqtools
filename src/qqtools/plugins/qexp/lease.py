"""Clock-capability evidence and fenced lease policy for qexp schema 6."""
from __future__ import annotations

import ctypes
import math
import os
import random
import re
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Iterable

from .config_types import RootConfig
from .runtime.paths import local_paths, shared_paths
from .runtime.records import utc_now
from .runtime.store import atomic_replace, read_json


AUTHORITY_MODES = ("bounded_lease", "holder_bound")


class LeaseRenewalOutcome(str, Enum):
    RENEWED = "renewed"
    NOT_REQUIRED = "not_required"
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
    clock_observation_max_age_seconds: float = 30.0
    clock_provider_margin_seconds: float = 0.001
    clock_provider_priority: tuple[str, ...] = ("chrony", "linux_adjtimex")
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
        if self.clock_observation_max_age_seconds <= 0 or self.clock_provider_margin_seconds < 0:
            raise ValueError("clock observation age and provider margin are invalid.")
        if not self.clock_provider_priority or len(set(self.clock_provider_priority)) != len(self.clock_provider_priority):
            raise ValueError("clock provider priority must contain unique providers.")
        if set(self.clock_provider_priority) - {"chrony", "linux_adjtimex"}:
            raise ValueError("clock provider priority contains an unsupported provider.")
        minimum = 2 * self.max_clock_skew_seconds + self.renewal_commit_margin_seconds + self.retry_max_seconds
        if self.ttl_seconds <= minimum:
            raise ValueError("lease TTL has no safe clock and retry budget.")
        if self.lease_loss_action != "isolate":
            raise ValueError("only lease_loss_action='isolate' is supported.")

    def retry_delay(self, failure_count: int) -> float:
        capped = min(self.retry_max_seconds, self.retry_initial_seconds * 2 ** max(0, failure_count - 1))
        spread = capped * self.retry_jitter_ratio
        return max(0.0, capped + random.uniform(-spread, spread))


@dataclass(frozen=True, slots=True)
class ClockObservation:
    """One provider's conservative system-clock-to-UTC interval observation."""

    observation_id: str
    provider: str
    observed_at: str
    monotonic_observed_at: float
    boot_id: str
    lower_error_seconds: float
    upper_error_seconds: float
    max_drift_rate: float
    provider_margin_seconds: float

    @property
    def observed_bound_seconds(self) -> float:
        return max(abs(self.lower_error_seconds), abs(self.upper_error_seconds))

    def bound_at(self, monotonic_now: float) -> float:
        age = max(0.0, monotonic_now - self.monotonic_observed_at)
        return self.observed_bound_seconds + age * self.max_drift_rate + self.provider_margin_seconds

    def interval_at(self, monotonic_now: float) -> tuple[float, float]:
        growth = max(0.0, monotonic_now - self.monotonic_observed_at) * self.max_drift_rate
        margin = growth + self.provider_margin_seconds
        return self.lower_error_seconds - margin, self.upper_error_seconds + margin

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ClockObservation":
        return cls(**value)


@dataclass(frozen=True, slots=True)
class ClockCapability:
    status: str
    reason: str
    observation: ClockObservation | None = None
    providers: tuple[str, ...] = ()

    @property
    def is_healthy(self) -> bool:
        return self.status == "healthy" and self.observation is not None


def default_lease_policy_document() -> dict[str, Any]:
    return {"lease_policy": asdict(LeasePolicy())}


def load_lease_policy(cfg: RootConfig) -> LeasePolicy:
    value = read_json(shared_paths(cfg.shared_root)["lease_policy"]).get("lease_policy")
    if not isinstance(value, dict):
        raise RuntimeError("qexp lease policy is malformed.")
    value = dict(value)
    # This supports only the pre-release source tree while it is being cut over; persisted
    # final schema-6 roots always contain the new key.
    if "clock_observation_max_age_seconds" not in value and "clock_health_max_age_seconds" in value:
        value["clock_observation_max_age_seconds"] = value.pop("clock_health_max_age_seconds")
    if isinstance(value.get("clock_provider_priority"), list):
        value["clock_provider_priority"] = tuple(value["clock_provider_priority"])
    return LeasePolicy(**value)


def save_lease_policy(cfg: RootConfig, policy: LeasePolicy) -> None:
    value = asdict(policy)
    value["clock_provider_priority"] = list(policy.clock_provider_priority)
    atomic_replace(shared_paths(cfg.shared_root)["lease_policy"], {"lease_policy": value})


def lease_expiry(policy: LeasePolicy) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=policy.ttl_seconds)).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z")


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def holder_safe_deadline(expires_at: str, holder_bound_seconds: float) -> datetime:
    return parse_utc(expires_at) - timedelta(seconds=holder_bound_seconds)


def reclaim_allowed_at(expires_at: str, holder_bound_seconds: float,
                       reclaimer_bound_seconds: float) -> datetime:
    return parse_utc(expires_at) + timedelta(
        seconds=holder_bound_seconds + reclaimer_bound_seconds
    )


def new_timed_offer_proof(observation: ClockObservation, after_seconds: int, *,
                          wall_now: datetime | None = None,
                          monotonic_now: float | None = None) -> tuple[str, dict[str, Any]]:
    """Create the immutable creator-side proof for a timed sharing deadline."""
    if not isinstance(after_seconds, int) or after_seconds < 0:
        raise ValueError("timed offer delay must be a non-negative integer.")
    wall_now = wall_now or datetime.now(timezone.utc)
    # Sample monotonic time after wall time.  That can only overstate the observation age
    # associated with the persisted deadline, which is conservative.
    monotonic_now = time.monotonic() if monotonic_now is None else monotonic_now
    if not isinstance(monotonic_now, (int, float)) or not math.isfinite(monotonic_now):
        raise ValueError("timed offer monotonic time is invalid.")
    deadline_monotonic_at = monotonic_now + after_seconds
    deadline = wall_now + timedelta(seconds=after_seconds)
    return (
        deadline.isoformat().replace("+00:00", "Z"),
        {
            "creator_observation": observation.to_dict(),
            "deadline_monotonic_at": deadline_monotonic_at,
        },
    )


def timed_offer_deadline_upper(deadline: str, proof: dict[str, Any]) -> datetime:
    """Return the latest UTC instant at which a creator's deadline may occur."""
    try:
        observation = ClockObservation.from_dict(proof["creator_observation"])
        deadline_monotonic_at = proof["deadline_monotonic_at"]
        deadline_at = parse_utc(deadline)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("timed offer proof is malformed.") from exc
    numeric = (observation.monotonic_observed_at, observation.lower_error_seconds,
               observation.upper_error_seconds, observation.max_drift_rate,
               observation.provider_margin_seconds, deadline_monotonic_at)
    if (any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in numeric)
            or observation.max_drift_rate < 0 or observation.provider_margin_seconds < 0
            or deadline_monotonic_at < observation.monotonic_observed_at):
        raise ValueError("timed offer deadline monotonic time is invalid.")
    try:
        return deadline_at + timedelta(seconds=observation.bound_at(deadline_monotonic_at))
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("timed offer deadline bound is invalid.") from exc


_OFFSET_PATTERN = re.compile(r"^System time\s*:\s*([+-]?[0-9.]+)\s+seconds", re.MULTILINE)
_ROOT_DELAY_PATTERN = re.compile(r"^Root delay\s*:\s*([+-]?[0-9.]+)\s+seconds", re.MULTILINE)
_DISPERSION_PATTERN = re.compile(r"^Root dispersion\s*:\s*([0-9.]+)\s+seconds", re.MULTILINE)
_SKEW_PATTERN = re.compile(r"^Skew\s*:\s*([0-9.]+)\s+ppm", re.MULTILINE)
_LEAP_PATTERN = re.compile(r"^Leap status\s*:\s*(.+)$", re.MULTILINE)


def _boot_id() -> str:
    try:
        return open("/proc/sys/kernel/random/boot_id", encoding="utf-8").read().strip()
    except OSError:
        return "unknown-boot"


def _chrony_observation(policy: LeasePolicy, *, run: Callable[..., Any] = subprocess.run) -> ClockObservation:
    try:
        completed = run(["chronyc", "tracking", "-n"], check=False, capture_output=True,
                        text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"chrony_unavailable:{type(exc).__name__}") from exc
    if completed.returncode:
        raise RuntimeError(f"chrony_exit:{completed.returncode}")
    offset = _OFFSET_PATTERN.search(completed.stdout)
    root_delay = _ROOT_DELAY_PATTERN.search(completed.stdout)
    dispersion = _DISPERSION_PATTERN.search(completed.stdout)
    skew = _SKEW_PATTERN.search(completed.stdout)
    leap = _LEAP_PATTERN.search(completed.stdout)
    if not all((offset, root_delay, dispersion, skew, leap)) or leap.group(1).strip() != "Normal":
        raise RuntimeError("chrony_tracking_incomplete")
    # chrony reports remaining system correction plus the NTP root distance components.
    # Root delay contributes half its round-trip delay to one-way UTC uncertainty.
    observed = abs(float(offset.group(1))) + abs(float(root_delay.group(1))) / 2 + float(dispersion.group(1))
    return ClockObservation(uuid.uuid4().hex, "chrony", utc_now(), time.monotonic(), _boot_id(),
                            -observed, observed, float(skew.group(1)) / 1_000_000,
                            policy.clock_provider_margin_seconds)


class _Timex(ctypes.Structure):
    _fields_ = [
        ("modes", ctypes.c_uint), ("offset", ctypes.c_long), ("freq", ctypes.c_long),
        ("maxerror", ctypes.c_long), ("esterror", ctypes.c_long), ("status", ctypes.c_int),
        ("constant", ctypes.c_long), ("precision", ctypes.c_long), ("tolerance", ctypes.c_long),
        ("time_sec", ctypes.c_long), ("time_usec", ctypes.c_long), ("tick", ctypes.c_long),
        ("ppsfreq", ctypes.c_long), ("jitter", ctypes.c_long), ("shift", ctypes.c_int),
        ("stabil", ctypes.c_long), ("jitcnt", ctypes.c_long), ("calcnt", ctypes.c_long),
        ("errcnt", ctypes.c_long), ("stbcnt", ctypes.c_long), ("tai", ctypes.c_int),
    ]


_TIME_ERROR = 5
_STA_UNSYNC = 0x0040
_STA_CLOCKERR = 0x1000


def _adjtimex_observation(policy: LeasePolicy) -> ClockObservation:
    if os.name != "posix":
        raise RuntimeError("linux_adjtimex_unavailable:unsupported_platform")
    value = _Timex()
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        state = libc.adjtimex(ctypes.byref(value))
    except (AttributeError, OSError) as exc:
        raise RuntimeError(f"linux_adjtimex_unavailable:{type(exc).__name__}") from exc
    if state < 0:
        raise RuntimeError(f"linux_adjtimex_errno:{ctypes.get_errno()}")
    if state == _TIME_ERROR or value.status & (_STA_UNSYNC | _STA_CLOCKERR):
        raise RuntimeError("linux_adjtimex_unsynchronized")
    if min(value.maxerror, value.esterror, value.precision, value.tolerance) < 0:
        raise RuntimeError("linux_adjtimex_invalid_metrics")
    # maxerror is the kernel's absolute maximum error.  The estimated error plus
    # clock precision is included as a second independent conservative lower bound.
    observed = max(value.maxerror / 1_000_000, (value.esterror + value.precision) / 1_000_000)
    # Linux stores tolerance in the same 16.16 ppm representation as frequency.
    return ClockObservation(uuid.uuid4().hex, "linux_adjtimex", utc_now(), time.monotonic(), _boot_id(),
                            -observed, observed, value.tolerance / (65_536 * 1_000_000),
                            policy.clock_provider_margin_seconds)


def chrony_health(policy: LeasePolicy, *, run: Callable[..., Any] = subprocess.run) -> tuple[bool, str]:
    """Compatibility probe retained for diagnostics and provider contract tests."""
    try:
        observation = _chrony_observation(policy, run=run)
    except RuntimeError as exc:
        return False, str(exc)
    if observation.bound_at(time.monotonic()) > policy.max_clock_skew_seconds:
        return False, "chrony_error_bound_exceeds_policy"
    return True, "healthy"


def _provider_observations(policy: LeasePolicy) -> tuple[list[ClockObservation], list[str]]:
    observations: list[ClockObservation] = []
    reasons: list[str] = []
    adapters: dict[str, Callable[[LeasePolicy], ClockObservation]] = {
        "chrony": _chrony_observation,
        "linux_adjtimex": _adjtimex_observation,
    }
    for provider in policy.clock_provider_priority:
        try:
            observations.append(adapters[provider](policy))
        except RuntimeError as exc:
            reasons.append(f"{provider}:{exc}")
    return observations, reasons


def clock_capability(cfg: RootConfig, policy: LeasePolicy | None = None) -> ClockCapability:
    """Return the current fail-closed authority capability, refreshing stale evidence."""
    policy = policy or load_lease_policy(cfg)
    path = local_paths(cfg.runtime_root)["clock_health"]
    now_mono = time.monotonic()
    try:
        cached = read_json(path).get("clock_capability", {})
        observation = ClockObservation.from_dict(cached["observation"])
        if (cached.get("status") == "healthy" and observation.boot_id == _boot_id()
                and now_mono - observation.monotonic_observed_at <= policy.clock_observation_max_age_seconds
                and observation.bound_at(now_mono) <= policy.max_clock_skew_seconds):
            return ClockCapability("healthy", "healthy", observation, tuple(cached.get("providers", [])))
    except (KeyError, OSError, TypeError, ValueError):
        pass
    observations, reasons = _provider_observations(policy)
    qualifying = [item for item in observations if item.bound_at(now_mono) <= policy.max_clock_skew_seconds]
    if not qualifying:
        status = "unavailable" if not observations else "unhealthy"
        capability = ClockCapability(status, ";".join(reasons) or "clock_error_bound_exceeds_policy", None,
                                      tuple(item.provider for item in observations))
    else:
        intervals = [item.interval_at(now_mono) for item in qualifying]
        lower = max(item[0] for item in intervals)
        upper = min(item[1] for item in intervals)
        if lower > upper:
            capability = ClockCapability("unhealthy", "provider_conflict", None,
                                          tuple(item.provider for item in qualifying))
        else:
            selected = next(item for name in policy.clock_provider_priority for item in qualifying
                            if item.provider == name)
            capability = ClockCapability("healthy", "healthy", selected,
                                          tuple(item.provider for item in qualifying))
    try:
        payload: dict[str, Any] = {"status": capability.status, "reason": capability.reason,
                                   "providers": list(capability.providers), "checked_at": utc_now()}
        if capability.observation:
            payload["observation"] = capability.observation.to_dict()
        atomic_replace(path, {"clock_capability": payload})
    except OSError:
        return ClockCapability("unavailable", "clock_capability_persistence_failed")
    return capability


def persist_clock_observation(cfg: RootConfig, observation: ClockObservation) -> None:
    """Persist immutable shared audit evidence before it is committed into authority."""
    path = shared_paths(cfg.shared_root)["clock_observations"] / cfg.machine_name / f"{observation.observation_id}.json"
    if path.exists():
        return
    atomic_replace(path, {"clock_observation": observation.to_dict()})


def authority_clock_health(cfg: RootConfig, policy: LeasePolicy | None = None) -> tuple[bool, str]:
    capability = clock_capability(cfg, policy)
    return capability.is_healthy, capability.reason
