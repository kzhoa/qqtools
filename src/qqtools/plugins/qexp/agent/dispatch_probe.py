"""In-memory ownership for machine primary-probe route state."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

from ..machine_dispatch_plan import (
    PrimaryProbeRouteDecision,
    PrimaryProbeRouteState,
    begin_primary_probe_route,
    finish_primary_probe_route,
)
from ..runtime.ready import ReadyCursor

RouteKey: TypeAlias = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class DependencyRecheck:
    """Temporary progress for a dependency-only primary candidate recheck."""

    cursor: ReadyCursor | None
    is_active: bool = False
    started_at_route_start: bool = False


@dataclass(frozen=True, slots=True)
class PrimaryProbeRoute:
    """Immutable baseline and dependency-recheck state for one route."""

    cursor: ReadyCursor | None
    revision: int | None
    is_complete: bool
    recheck: DependencyRecheck | None = None

    def __post_init__(self) -> None:
        if self.revision is not None and (not isinstance(self.revision, int) or self.revision < 0):
            raise ValueError("ready route revision must not be negative.")


@dataclass(slots=True)
class _LaneState:
    route_keys: tuple[RouteKey, ...] = ()
    pending: set[RouteKey] | None = None
    next_recheck_key: RouteKey | None = None

    def __post_init__(self) -> None:
        if self.pending is None:
            self.pending = set()


class PrimaryProbeSession:
    """Own process-local primary-probe route transitions for every dispatch lane."""

    def __init__(self) -> None:
        self._routes: dict[RouteKey, PrimaryProbeRoute] = {}
        self._lanes: dict[str, _LaneState] = {}

    def begin_round(self, lane: str, route_keys: Iterable[RouteKey]) -> tuple[RouteKey, ...]:
        """Reconcile one lane's participating routes and return its pending routes."""
        participating = tuple(dict.fromkeys(route_keys))
        state = self._lanes.setdefault(lane, _LaneState())
        state.route_keys = participating
        state.pending.intersection_update(participating)
        state.pending.update(key for key in participating if not self.route(key).is_complete)
        state.pending.update(
            key for key in participating if self.route(key).recheck is not None and self.route(key).recheck.is_active
        )
        if not state.pending:
            state.pending.update(participating)
        return tuple(key for key in participating if key in state.pending)

    def next_recheck(self, lane: str) -> RouteKey | None:
        """Fairly select and activate one dependency recheck once baselines complete."""
        state = self._lanes.get(lane)
        if state is None or not state.route_keys:
            return None
        if any(not self.route(key).is_complete for key in state.route_keys):
            return None
        active = [
            key for key in state.route_keys if self.route(key).recheck is not None and self.route(key).recheck.is_active
        ]
        if active:
            return active[0]
        candidates = [key for key in state.route_keys if self.route(key).recheck is not None]
        if not candidates:
            return None

        start_index = 0
        if state.next_recheck_key in state.route_keys:
            start_index = state.route_keys.index(state.next_recheck_key)
        for offset in range(len(state.route_keys)):
            index = (start_index + offset) % len(state.route_keys)
            key = state.route_keys[index]
            route = self.route(key)
            if route.recheck is None:
                continue
            next_index = (index + 1) % len(state.route_keys)
            state.next_recheck_key = state.route_keys[next_index]
            state.pending.discard(key)
            self._routes[key] = PrimaryProbeRoute(
                route.cursor,
                route.revision,
                route.is_complete,
                DependencyRecheck(
                    route.recheck.cursor,
                    is_active=True,
                    started_at_route_start=route.recheck.cursor is None,
                ),
            )
            return key
        return None

    def route(self, key: RouteKey) -> PrimaryProbeRoute:
        """Return an immutable route record, defaulting to an uninitialized route."""
        return self._routes.get(key, PrimaryProbeRoute(None, None, False))

    def begin_route(self, key: RouteKey, observed_revision: int | None) -> PrimaryProbeRouteDecision:
        """Apply the pure before-scan revision reduction and replace route state."""
        current = self.route(key)
        decision = begin_primary_probe_route(
            PrimaryProbeRouteState(current.cursor, current.revision, current.is_complete),
            observed_revision,
        )
        if decision.has_index_changed:
            recheck = None
        else:
            recheck = current.recheck
        self._routes[key] = PrimaryProbeRoute(
            decision.state.cursor,
            decision.state.revision,
            decision.state.is_complete,
            recheck,
        )
        state = self._lane_for_key(key)
        if decision.should_scan:
            state.pending.add(key)
        else:
            state.pending.discard(key)
        return decision

    def record_progress(self, key: RouteKey, cursor: ReadyCursor | None) -> None:
        """Record baseline or active-recheck progress without crossing their state boundary."""
        current = self.route(key)
        if current.recheck is not None and current.recheck.is_active:
            self._routes[key] = PrimaryProbeRoute(
                current.cursor,
                current.revision,
                current.is_complete,
                DependencyRecheck(
                    cursor,
                    is_active=True,
                    started_at_route_start=current.recheck.started_at_route_start,
                ),
            )
            return
        self._routes[key] = PrimaryProbeRoute(cursor, current.revision, False, current.recheck)
        self._lane_for_key(key).pending.add(key)

    def hold_candidate(self, key: RouteKey, cursor: ReadyCursor | None) -> None:
        """Retain a real primary-demand candidate at its resume position."""
        self.record_progress(key, cursor)

    def record_dependency_wait(self, key: RouteKey, cursor: ReadyCursor | None) -> None:
        """Retain or advance dependency-only recheck progress."""
        current = self.route(key)
        if current.recheck is not None and current.recheck.is_active:
            recheck = DependencyRecheck(
                cursor,
                is_active=True,
                started_at_route_start=current.recheck.started_at_route_start,
            )
        elif current.recheck is None:
            recheck = DependencyRecheck(cursor)
        else:
            recheck = current.recheck
        self._routes[key] = PrimaryProbeRoute(
            current.cursor,
            current.revision,
            current.is_complete,
            recheck,
        )

    def finish_route(self, key: RouteKey, observed_revision: int) -> PrimaryProbeRouteDecision:
        """Finalize a baseline scan and reconcile its revision atomically."""
        current = self.route(key)
        decision = finish_primary_probe_route(
            PrimaryProbeRouteState(current.cursor, current.revision, True),
            observed_revision,
        )
        recheck = None if decision.has_index_changed else current.recheck
        self._routes[key] = PrimaryProbeRoute(
            decision.state.cursor,
            decision.state.revision,
            decision.state.is_complete,
            recheck,
        )
        state = self._lane_for_key(key)
        if decision.state.is_complete:
            state.pending.discard(key)
        else:
            state.pending.add(key)
        return decision

    def finish_recheck(self, key: RouteKey, *, has_waiting_candidate: bool) -> None:
        """Finish active dependency progress while preserving the completed baseline."""
        current = self.route(key)
        if current.recheck is None or not current.recheck.is_active:
            raise ValueError("route does not have an active dependency recheck")
        if has_waiting_candidate:
            recheck = DependencyRecheck(current.recheck.cursor)
        elif current.recheck.started_at_route_start:
            recheck = None
        else:
            recheck = DependencyRecheck(None)
        self._routes[key] = PrimaryProbeRoute(
            current.cursor,
            current.revision,
            current.is_complete,
            recheck,
        )
        self._lane_for_key(key).pending.discard(key)

    def completed_revisions(
        self,
        lane: str,
        route_keys: Iterable[RouteKey],
    ) -> Mapping[RouteKey, int] | None:
        """Return an immutable revision snapshot when every required route is complete."""
        state = self._lanes.get(lane)
        if state is None:
            return None
        required = tuple(dict.fromkeys(route_keys))
        snapshot: dict[RouteKey, int] = {}
        for key in required:
            route = self.route(key)
            if not route.is_complete or not isinstance(route.revision, int):
                return None
            snapshot[key] = route.revision
        return MappingProxyType(snapshot)

    def invalidate_routes(self, route_keys: Iterable[RouteKey]) -> None:
        """Reset exactly the specified routes and make them pending in their lanes."""
        for key in dict.fromkeys(route_keys):
            self._routes[key] = PrimaryProbeRoute(None, None, False)
            self._lane_for_key(key).pending.add(key)

    def _lane_for_key(self, key: RouteKey) -> _LaneState:
        return self._lanes.setdefault(key[2], _LaneState())


__all__ = ["DependencyRecheck", "PrimaryProbeRoute", "PrimaryProbeSession", "RouteKey"]
