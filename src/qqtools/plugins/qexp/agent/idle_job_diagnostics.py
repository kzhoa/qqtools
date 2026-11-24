"""Bounded observations of the checks that prevent Idle Job demand proof."""

from __future__ import annotations

import re
import shlex
import time
from collections.abc import Collection, Mapping, Sequence
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any

from ..runtime.upgrade.contracts import PHASES
from ..runtime.upgrade.framework import UPGRADE_STATES
from ..upgrade_progress import validate_progress_evidence
from .bindings import ProjectBinding

DEMAND_DETAIL_MEMBER = "idle_job_demand_detail"
MAX_STATUS_BYTES = 65_536
_MAX_PROJECTS = 3
_MAX_LIVE_AGE_SECONDS = 60
_SAFE_TOKEN = re.compile(r"[A-Za-z0-9._-]{1,128}")
_UTC_TIMESTAMP = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?(?:Z|\+00:00)")
_CODES = frozenset(
    {
        "dispatch_cycle_incomplete",
        "enablement_unready",
        "bindings_unknown",
        "launch_handoff_pending",
        "upgrade_evidence_unknown",
        "upgrade_blocks_idle",
        "recovery_enrollment_pending",
        "maintenance_retry_pending",
        "partition_unknown",
        "recovery_unknown",
        "authority_unready",
        "demand_census_unknown",
        "executor_unready",
        "gpu_policy_unknown",
        "cpu_policy_unknown",
        "observation_failed",
    }
)
_DETAIL_FIELDS = frozenset({"code", "agent_instance_id", "observed_at", "since", "project_count", "projects"})
_PROJECT_FIELDS = frozenset({"project_id", "migration", "phase", "state", "last_progress_at", "detail_command"})


def _safe_token(value: object) -> bool:
    return isinstance(value, str) and value not in {".", ".."} and _SAFE_TOKEN.fullmatch(value) is not None


def _bounded_text(value: object, limit: int) -> bool:
    return isinstance(value, str) and 0 < len(value) <= limit and value.isprintable()


def _utc_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or len(value) > 32 or _UTC_TIMESTAMP.fullmatch(value) is None:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _validate_project(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict) or set(value) != _PROJECT_FIELDS:
        return None
    if not _bounded_text(value["project_id"], 256):
        return None
    if any(value[key] is not None and not _safe_token(value[key]) for key in ("migration", "phase", "state")):
        return None
    if value["phase"] is not None and value["phase"] not in PHASES:
        return None
    if value["state"] is not None and value["state"] not in UPGRADE_STATES:
        return None
    if value["last_progress_at"] is not None and _utc_timestamp(value["last_progress_at"]) is None:
        return None
    if value["detail_command"] is not None and not _bounded_text(value["detail_command"], 8192):
        return None
    return dict(value)


def validate_demand_detail(value: object) -> dict[str, Any] | None:
    """Copy valid optional detail independently of the required status member."""
    if not isinstance(value, dict) or set(value) != _DETAIL_FIELDS:
        return None
    if not isinstance(value["code"], str) or value["code"] not in _CODES:
        return None
    if not _safe_token(value["agent_instance_id"]) or _utc_timestamp(value["observed_at"]) is None:
        return None
    if value["since"] is not None and _utc_timestamp(value["since"]) is None:
        return None
    count, projects = value["project_count"], value["projects"]
    if count is not None and (type(count) is not int or count < 0):
        return None
    if not isinstance(projects, list) or len(projects) != min(count or 0, _MAX_PROJECTS):
        return None
    validated = []
    for project in projects:
        copied = _validate_project(project)
        if copied is None:
            return None
        validated.append(copied)
    if len({project["project_id"] for project in validated}) != len(validated):
        return None
    return {**value, "projects": validated}


def _select_migration(status: Mapping[str, Any]) -> dict[str, Any] | None:
    migrations = status.get("migrations")
    if not isinstance(migrations, list):
        return None
    active = None
    for item in migrations:
        if not isinstance(item, dict) or not all(_safe_token(item.get(key)) for key in ("name", "phase", "state")):
            continue
        if item["phase"] not in PHASES or item["state"] not in UPGRADE_STATES:
            continue
        if item["state"] == "repair_required":
            return item
        if active is None and item["state"] not in {"completed", "idle"}:
            active = item
    return active


def _last_progress_at(migration: dict[str, Any]) -> str | None:
    if migration.get("in_flight"):
        return None
    try:
        progress, _ = validate_progress_evidence(migration.get("progress"), migration.get("progress_evidence"))
    except (TypeError, ValueError):
        return None
    timestamp = progress["last_progress_at"]
    return timestamp if _utc_timestamp(timestamp) is not None else None


def _project_detail(
    project_id: str,
    *,
    binding: ProjectBinding | None,
    runtime_root: Path | None,
    cached_binding: object,
    cached_status: object,
) -> dict[str, Any]:
    migration = None
    command = None
    if binding is not None and runtime_root is not None:
        command = shlex.join(
            ["qexp", "--machine-runtime-root", str(runtime_root), "admin", "upgrade", "status", "--project", project_id]
        )
        if (
            cached_binding == binding
            and isinstance(cached_status, Mapping)
            and cached_status.get("project_id") == project_id
            and cached_status.get("shared_root") == str(binding.shared_root)
        ):
            migration = _select_migration(cached_status)
    return {
        "project_id": project_id,
        "migration": migration["name"] if migration is not None else None,
        "phase": migration["phase"] if migration is not None else None,
        "state": migration["state"] if migration is not None else None,
        "last_progress_at": _last_progress_at(migration) if migration is not None else None,
        "detail_command": command,
    }


def build_demand_observation(
    code: str,
    *,
    project_ids: Collection[str] | None = None,
    bindings: Sequence[ProjectBinding] = (),
    runtime_root: Path | None = None,
    upgrade_bindings: Mapping[str, ProjectBinding] | None = None,
    upgrade_projects: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Sample the existing blocker collection without sorting or copying it."""
    if project_ids is None:
        return {"code": code, "project_count": None, "projects": []}
    count = len(project_ids)
    ids = tuple(islice(project_ids, _MAX_PROJECTS))
    projects = []
    for project_id in ids:
        binding = None
        if code == "upgrade_blocks_idle":
            binding = next((item for item in bindings if item.project_id == project_id), None)
        projects.append(
            _project_detail(
                project_id,
                binding=binding,
                runtime_root=runtime_root,
                cached_binding=upgrade_bindings.get(project_id) if upgrade_bindings is not None else None,
                cached_status=upgrade_projects.get(project_id) if upgrade_projects is not None else None,
            )
        )
    return {"code": code, "project_count": count, "projects": projects}


def _scope(detail: Mapping[str, Any]) -> frozenset[tuple[str, str | None, str | None, str | None]]:
    return frozenset(
        (project["project_id"], project["migration"], project["phase"], project["state"])
        for project in detail["projects"]
    )


def observe_demand_detail(
    detail: Mapping[str, Any],
    *,
    agent_instance_id: str,
    observed_at: float,
    previous: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Stamp an observation and continue a wait only for a complete matching scope."""
    timestamp = datetime.fromtimestamp(observed_at, timezone.utc).isoformat().replace("+00:00", "Z")
    observation = validate_demand_detail(
        {**detail, "agent_instance_id": agent_instance_id, "observed_at": timestamp, "since": None}
    )
    if observation is None or (observation["project_count"] or 0) > _MAX_PROJECTS:
        return observation
    if (
        previous is not None
        and previous["agent_instance_id"] == agent_instance_id
        and previous["code"] == observation["code"]
        and previous["project_count"] == observation["project_count"]
        and _scope(previous) == _scope(observation)
    ):
        observation["since"] = previous["since"]
    else:
        observation["since"] = timestamp
    return observation


def display_demand_detail(
    detail: object,
    *,
    agent_instance_id: str | None,
    now: float | None = None,
) -> dict[str, Any] | None:
    """Add display-only freshness and duration for an already verified agent identity."""
    value = validate_demand_detail(detail)
    if value is None:
        return None
    observed = datetime.fromisoformat(value["observed_at"]).timestamp()
    display_time = time.time() if now is None else now
    age = display_time - observed
    live = agent_instance_id == value["agent_instance_id"] and 0 <= age <= _MAX_LIVE_AGE_SECONDS
    elapsed = None
    if live and value["since"] is not None:
        since = datetime.fromisoformat(value["since"]).timestamp()
        if since <= observed:
            elapsed = display_time - since
    return {**value, "live": live, "elapsed_seconds": elapsed}
