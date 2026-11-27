"""Machine-local observations of exact Project registration renewal."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import timedelta, timezone
from heapq import nsmallest
from threading import Lock
from typing import TYPE_CHECKING

from .snapshot_publication_health import observation_timestamp

if TYPE_CHECKING:
    from .project_io_controller import _ValidationIdentity

REGISTRATION_RENEWAL_HEALTH_ITEM_LIMIT = 8
_CAUSE_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,160}")


@dataclass(frozen=True, slots=True)
class _RenewalObservation:
    last_outcome: str | None
    blocking_reason: str | None
    cause: str | None
    retry_at: float | None
    next_retry_at: str | None


class RegistrationRenewalHealth:
    """Keep an advisory view independent of admission and retry decisions."""

    def __init__(self, *, monotonic: Callable[[], float], utc_clock: Callable[[], str]) -> None:
        self._monotonic = monotonic
        self._utc_clock = utc_clock
        self._lock = Lock()
        self._executor_epoch: str | None = None
        self._current: frozenset[_ValidationIdentity] = frozenset()
        self._occupied_owners: frozenset[tuple[str, str, str]] = frozenset()
        self._observations: dict[_ValidationIdentity, _RenewalObservation] = {}

    def reconcile(
        self,
        executor_epoch: str,
        occupied_owners: set[tuple[str, str, str]],
        current: frozenset[_ValidationIdentity] | None = None,
    ) -> None:
        with self._lock:
            if executor_epoch != self._executor_epoch:
                self._executor_epoch = executor_epoch
                self._current = frozenset()
                self._observations.clear()
            if current is not None:
                self._current = current
                retained = {}
                by_owner = {identity.request_owner: identity for identity in current}
                for identity, state in self._observations.items():
                    successor = by_owner.get(identity.request_owner)
                    if successor == identity:
                        retained[identity] = state
                    elif (
                        successor is not None
                        and replace(identity, registry_revision=successor.registry_revision) == successor
                    ):
                        # A roster edit invalidates scheduling state, while the same
                        # generation retains its observed outcome.
                        retained[successor] = _RenewalObservation(state.last_outcome, None, None, None, None)
                self._observations = retained
            self._occupied_owners = frozenset(occupied_owners)

    def record(
        self,
        identity: _ValidationIdentity,
        *,
        outcome: str | None = None,
        blocking_reason: str | None = None,
        cause: object = None,
        retry_at: float | None = None,
    ) -> None:
        exception_type = cause if isinstance(cause, str) and _CAUSE_PATTERN.fullmatch(cause) else None
        next_retry_at = self._retry_timestamp(retry_at) if retry_at is not None else None
        with self._lock:
            if identity not in self._current:
                return
            previous = self._observations.get(identity)
            last_outcome = previous.last_outcome if outcome is None and previous is not None else outcome
            self._observations[identity] = _RenewalObservation(
                last_outcome, blocking_reason, exception_type, retry_at, next_retry_at
            )

    def view(self, *, limit: int = REGISTRATION_RENEWAL_HEALTH_ITEM_LIMIT) -> dict[str, object]:
        """Return detached observations with active failures and other blockers first."""
        if type(limit) is not int or not 1 <= limit <= REGISTRATION_RENEWAL_HEALTH_ITEM_LIMIT:
            raise ValueError(f"limit must be an integer from 1 through {REGISTRATION_RENEWAL_HEALTH_ITEM_LIMIT}.")
        with self._lock:
            observations = tuple(self._observations.items())
            occupied_owners = self._occupied_owners
        now = self._monotonic()
        entries = (
            (identity, self._active_observation(identity, state, occupied_owners, now))
            for identity, state in observations
        )
        selected = nsmallest(
            limit,
            entries,
            key=lambda item: (
                item[1].blocking_reason is None,
                item[1].retry_at is None,
                item[0].project_id,
                item[0].registration_generation,
            ),
        )
        items = [
            {
                "project_id": identity.project_id,
                "registration_generation": identity.registration_generation,
                "last_outcome": state.last_outcome,
                "blocking_reason": state.blocking_reason,
                "cause": state.cause,
                "next_retry_at": state.next_retry_at,
            }
            for identity, state in selected
        ]
        return {"total": len(observations), "omitted_count": len(observations) - len(items), "items": items}

    @staticmethod
    def _active_observation(
        identity: _ValidationIdentity,
        state: _RenewalObservation,
        occupied_owners: frozenset[tuple[str, str, str]],
        now: float,
    ) -> _RenewalObservation:
        is_occupied = identity.request_owner in occupied_owners
        if state.retry_at is not None and state.retry_at <= now:
            return _RenewalObservation(state.last_outcome, None, None, None, None)
        if state.blocking_reason == "owner_occupied" and not is_occupied:
            return replace(state, blocking_reason=None)
        if state.blocking_reason == "awaiting_admission" and is_occupied:
            return replace(state, blocking_reason="owner_occupied")
        return state

    def _retry_timestamp(self, retry_at: float) -> str | None:
        try:
            remaining = retry_at - self._monotonic()
            wall_now = observation_timestamp(self._utc_clock())
            if wall_now is not None:
                deadline = wall_now + timedelta(seconds=remaining)
                return deadline.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        except (OSError, RuntimeError, ValueError, TypeError, OverflowError):
            return None
        return None
