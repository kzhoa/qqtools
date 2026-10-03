"""Closed JSON continuation for isolated, independent primary-demand scans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..runtime.ready import ReadyCursor
from .dispatch_probe import DependencyRecheck, PrimaryProbeRoute, PrimaryProbeSession
from .ready_cursor_validation import validate_ready_cursor_position

SCOPES = ("shared", "home")


def _object(value: object, fields: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("primary probe continuation has missing or unknown fields.")
    return value


def _revision(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("primary probe revision must be a nonnegative integer.")
    return value


def _cursor(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    return validate_ready_cursor_position(
        value,
        "primary probe cursor",
        require_safe_marker_name=True,
        reject_marker_nul=False,
    )


def validate_probe_state(value: object) -> dict[str, Any]:
    """Validate and detach all continuation state at the protocol boundary."""
    state = _object(value, {"routes", "pending_scopes", "next_recheck_scope"})
    routes = _object(state["routes"], set(SCOPES))
    result: dict[str, Any] = {"routes": {}}
    for scope in SCOPES:
        route = _object(routes[scope], {"cursor", "revision", "is_complete", "recheck"})
        revision = None if route["revision"] is None else _revision(route["revision"])
        complete = route["is_complete"]
        if type(complete) is not bool or (complete and revision is None):
            raise ValueError("completed primary probe route requires its baseline revision.")
        recheck = route["recheck"]
        if recheck is not None:
            recheck = _object(recheck, {"cursor", "is_active", "started_at_route_start"})
            if type(recheck["is_active"]) is not bool or type(recheck["started_at_route_start"]) is not bool:
                raise ValueError("primary probe recheck flags must be booleans.")
            recheck = {
                "cursor": _cursor(recheck["cursor"]),
                "is_active": recheck["is_active"],
                "started_at_route_start": recheck["started_at_route_start"],
            }
        result["routes"][scope] = {
            "cursor": _cursor(route["cursor"]),
            "revision": revision,
            "is_complete": complete,
            "recheck": recheck,
        }
    pending = state["pending_scopes"]
    if (
        not isinstance(pending, Sequence)
        or isinstance(pending, str | bytes)
        or len(pending) > 2
        or any(not isinstance(scope, str) or scope not in SCOPES for scope in pending)
        or len(set(pending)) != len(pending)
    ):
        raise ValueError("primary probe pending scopes are invalid.")
    next_scope = state["next_recheck_scope"]
    if next_scope is not None and (not isinstance(next_scope, str) or next_scope not in SCOPES):
        raise ValueError("primary probe next recheck scope is invalid.")
    result.update(pending_scopes=list(pending), next_recheck_scope=next_scope)
    return result


def encode_probe_session(session: PrimaryProbeSession, project_id: str, lane: str) -> dict[str, Any]:
    routes, pending, next_key = session.snapshot_lane(lane)

    def cursor_value(cursor: ReadyCursor | None) -> dict[str, Any] | None:
        if cursor is None:
            return None
        return {
            "catalog_page": cursor.catalog_page,
            "partition": cursor.partition,
            "after_name": cursor.after_name,
            "revision": cursor.revision,
        }

    encoded = {}
    for scope in SCOPES:
        route = routes.get((project_id, scope, lane), PrimaryProbeRoute(None, None, False))
        recheck = route.recheck
        encoded[scope] = {
            "cursor": cursor_value(route.cursor),
            "revision": route.revision,
            "is_complete": route.is_complete,
            "recheck": None
            if recheck is None
            else {
                "cursor": cursor_value(recheck.cursor),
                "is_active": recheck.is_active,
                "started_at_route_start": recheck.started_at_route_start,
            },
        }
    return validate_probe_state(
        {
            "routes": encoded,
            "pending_scopes": [key[1] for key in pending],
            "next_recheck_scope": None if next_key is None else next_key[1],
        }
    )


def decode_probe_session(value: object, project_id: str, machine_name: str, lane: str) -> PrimaryProbeSession:
    state = validate_probe_state(value)

    def cursor_value(cursor: dict[str, Any] | None, scope: str) -> ReadyCursor | None:
        return None if cursor is None else ReadyCursor(project_id, machine_name, scope, **cursor)

    routes = {}
    for scope, route in state["routes"].items():
        recheck = route["recheck"]
        routes[(project_id, scope, lane)] = PrimaryProbeRoute(
            cursor_value(route["cursor"], scope),
            route["revision"],
            route["is_complete"],
            None
            if recheck is None
            else DependencyRecheck(
                cursor_value(recheck["cursor"], scope), recheck["is_active"], recheck["started_at_route_start"]
            ),
        )
    session = PrimaryProbeSession()
    session.restore_lane(
        lane,
        routes,
        [(project_id, scope, lane) for scope in state["pending_scopes"]],
        None if state["next_recheck_scope"] is None else (project_id, state["next_recheck_scope"], lane),
    )
    return session
