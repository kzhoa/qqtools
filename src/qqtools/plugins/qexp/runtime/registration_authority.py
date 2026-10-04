"""Shared registration eligibility, independent of MachineRuntime locks."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from qqtools.version import __version__

from ..config_types import RootConfig
from ..layout import load_machine_registration, save_machine_registration
from ..lease import LeasePolicy, lease_expiry, load_lease_policy, parse_utc
from .locks import exclusive, machine_lock
from .paths import machine_registration_path, shared_paths
from .records import utc_now
from .responsibility_store import DurableIO
from .work_budget import diagnostic_increment, diagnostic_observe_ns

REGISTRATION_VERSION = 1
RECOVERY_REGISTRATION_VERSION = 2
RECOVERY_REGISTRATION_PROTOCOL = "qexp-local-responsibility-v1"


@dataclass(frozen=True, slots=True)
class RegistrationIdentity:
    """The captured ownership identity; it never grants reusable authority."""

    project_id: str
    generation: str
    runtime_id: str
    runtime_root: str


def publish_recovery_registration_locked(
    cfg: RootConfig,
    registration: dict[str, Any],
    *,
    before_shared_write: Callable[[], None] | None = None,
) -> None:
    """Publish the existing recovery fence under shared registration ownership.

    The caller must hold the exact owner's shared registration write guard and
    durably retire any local rollback snapshot before a version-1 conversion.
    This function owns no MachineRuntime lock, retention, or capture state.
    """
    validate_registration_record(registration, registration["project_id"], cfg)
    if registration["version"] == RECOVERY_REGISTRATION_VERSION:
        if before_shared_write is not None:
            before_shared_write()
        # Replay must complete the directory barrier after a possibly committed
        # rename, rather than interpreting visibility as durable preparation.
        DurableIO().sync_directory(
            machine_registration_path(cfg.shared_root, cfg.machine_name).parent,
            "recovery_registration_fence",
        )
        return
    updated = dict(registration)
    updated.update(
        version=RECOVERY_REGISTRATION_VERSION,
        protocol_version=RECOVERY_REGISTRATION_VERSION,
        recovery_protocol=RECOVERY_REGISTRATION_PROTOCOL,
        client_version=__version__,
        updated_at=utc_now(),
    )
    if before_shared_write is not None:
        before_shared_write()
    save_machine_registration(cfg, {"registration": updated})


def validate_registration_record(record: dict[str, Any], project_id: str, cfg: RootConfig) -> None:
    """Validate the existing shared ownership record without local state."""
    version = record.get("version")
    if type(version) is not int or version not in {REGISTRATION_VERSION, RECOVERY_REGISTRATION_VERSION}:
        raise RuntimeError("project machine registration uses an unsupported protocol version.")
    if version == RECOVERY_REGISTRATION_VERSION and (
        type(record.get("protocol_version")) is not int
        or record["protocol_version"] != version
        or record.get("recovery_protocol") != RECOVERY_REGISTRATION_PROTOCOL
    ):
        raise RuntimeError("project machine registration has an unsupported recovery protocol.")
    if record.get("project_id") != project_id or record.get("shared_root") != str(cfg.shared_root):
        raise RuntimeError("project machine registration does not match Project identity.")
    if record.get("machine_name") != cfg.machine_name:
        raise RuntimeError("project machine registration does not match the requested logical name.")
    for key in ("generation", "runtime_instance_id", "runtime_root", "eligibility_expires_at"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise RuntimeError("project machine registration is malformed.")


def registration_state(record: dict[str, Any]) -> str:
    """Classify only the current shared lease, not local scheduling evidence."""
    try:
        expires_at = parse_utc(record["eligibility_expires_at"])
    except (KeyError, TypeError, ValueError):
        return "invalid"
    if record.get("state") == "superseded":
        return "superseded"
    return "eligible" if expires_at > datetime.now(timezone.utc) else "expired"


def observe_registration_owner(
    cfg: RootConfig, identity: RegistrationIdentity, *, is_current: Callable[[], bool]
) -> str:
    """Read positive shared ownership evidence without local guards or renewal."""
    with exclusive(shared_paths(cfg.shared_root)["locks"] / "registrations.lock"):
        with machine_lock(cfg.shared_root, cfg.machine_name):
            if not is_current():
                raise RuntimeError("registration observation local identity changed")
            raw = load_machine_registration(cfg)
            record = raw.get("registration") if isinstance(raw, dict) else None
            if not isinstance(record, dict):
                return "unregistered"
            validate_registration_record(record, identity.project_id, cfg)
            state = registration_state(record)
            if state == "invalid":
                raise RuntimeError("registration observation has invalid lease evidence")
            if not is_current():
                raise RuntimeError("registration observation local identity changed")
            if (
                record["generation"] != identity.generation
                or record["runtime_instance_id"] != identity.runtime_id
                or record["runtime_root"] != identity.runtime_root
            ):
                return "superseded"
            return state


def _renewal_interval(policy: LeasePolicy) -> float:
    return min(policy.renew_interval_seconds, policy.ttl_seconds / 2)


def observe_registration_renewal(previous_expiry: str, policy: LeasePolicy, *, is_reactivation: bool) -> None:
    """Observe completed publication against the previous renewal target."""
    diagnostic_increment("registration.reactivation" if is_reactivation else "registration.renewal")
    try:
        target = parse_utc(previous_expiry) - timedelta(seconds=policy.ttl_seconds - _renewal_interval(policy))
        lateness = max(0.0, (datetime.now(timezone.utc) - target).total_seconds())
    except (ValueError, TypeError, OverflowError):
        diagnostic_increment("registration.renewal_lateness_unavailable")
        return
    diagnostic_observe_ns("registration.renewal_lateness", int(lateness * 1_000_000_000))


@contextmanager
def registration_write_guard(
    cfg: RootConfig,
    identity: RegistrationIdentity,
    *,
    is_current: Callable[[], bool],
    renewal_horizon_seconds: float = 0.0,
    before_shared_write: Callable[[], None] | None = None,
    allow_reactivation: bool = False,
    force_renewal: bool = False,
) -> Iterator[dict[str, Any] | None]:
    """Hold only shared registration locks; renew the exact eligible generation.

    ``is_current`` must recheck the caller's captured local identity without
    holding a MachineRuntime lock. Every successor mutation needs its own fence.
    """
    if (
        not isinstance(renewal_horizon_seconds, int | float)
        or isinstance(renewal_horizon_seconds, bool)
        or not math.isfinite(renewal_horizon_seconds)
        or renewal_horizon_seconds < 0
    ):
        raise ValueError("renewal_horizon_seconds must be a finite nonnegative number.")
    policy = load_lease_policy(cfg)
    with exclusive(shared_paths(cfg.shared_root)["locks"] / "registrations.lock"):
        with machine_lock(cfg.shared_root, cfg.machine_name):
            raw = load_machine_registration(cfg)
            record = raw.get("registration") if isinstance(raw, dict) else None
            if not isinstance(record, dict):
                yield None
                return
            validate_registration_record(record, identity.project_id, cfg)
            state = registration_state(record)
            # The explicit library reactivation also repairs an unreadable
            # expiry for its exact owner. Ordinary isolated renewals cannot
            # use malformed expiry as authority or repair a superseded record.
            can_repair_expiry = force_renewal and state == "invalid" and record.get("state") == "eligible"
            is_reactivation = allow_reactivation and (state == "expired" or can_repair_expiry)
            if (
                not (state == "eligible" or is_reactivation)
                or record.get("generation") != identity.generation
                or record.get("runtime_instance_id") != identity.runtime_id
                or record.get("runtime_root") != identity.runtime_root
                or not is_current()
            ):
                yield None
                return
            client_version_mismatch = record.get("client_version") != __version__
            previous_expiry = record["eligibility_expires_at"]
            expires_at = None if can_repair_expiry else parse_utc(previous_expiry)
            next_expiry = lease_expiry(policy)
            parsed_next_expiry = parse_utc(next_expiry)
            now = datetime.now(timezone.utc)
            renew_at = (
                now
                if expires_at is None
                else expires_at - timedelta(seconds=policy.ttl_seconds - _renewal_interval(policy))
            )
            horizon = now + timedelta(seconds=renewal_horizon_seconds)
            if (
                force_renewal
                or is_reactivation
                or client_version_mismatch
                or (
                    parsed_next_expiry != expires_at
                    and (now >= renew_at or expires_at <= horizon or parsed_next_expiry < expires_at)
                )
            ):
                record = dict(record)
                record["client_version"] = __version__
                record["eligibility_expires_at"] = next_expiry
                record["updated_at"] = utc_now()
                if not is_current():
                    yield None
                    return
                if before_shared_write is not None:
                    before_shared_write()
                save_machine_registration(cfg, {"registration": record})
                observe_registration_renewal(previous_expiry, policy, is_reactivation=is_reactivation or force_renewal)
            yield record if is_current() else None
