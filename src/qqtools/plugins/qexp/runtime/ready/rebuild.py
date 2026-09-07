"""Build, audit, and repair orchestration for the ready projection."""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..locks import exclusive, schema_lock, schema_writer_lock
from ..paths import ready_state_path, shared_paths, task_path
from ..records import TaskRecord, utc_now, validate_identifier
from ..store import atomic_replace, iter_json, read_json
from . import primary_candidates, routes, state
from .index import (
    begin_primary_ready_index_rebuild,
    classify_ready_marker,
    delete_ready_marker,
    prepare_ready_transition,
    retire_previous_ready_generation,
    task_should_have_ready_marker,
)

READY_BUILD_PAGE_SIZE = 64


def _build_root(cfg: object, build_id: str) -> Path:
    validate_identifier(build_id, "ready build id")
    return shared_paths(cfg.shared_root)["ready_builds"] / build_id


def _build_page_path(cfg: object, build_id: str, page: int) -> Path:
    return _build_root(cfg, build_id) / "watermark" / f"{page:016d}.json"


def _write_build_page(
    cfg: object,
    build_id: str,
    page: int,
    task_ids: list[str],
) -> None:
    atomic_replace(
        _build_page_path(cfg, build_id, page),
        {
            "ready_build_page": {
                "schema_version": state.READY_PROTOCOL_VERSION,
                "build_id": build_id,
                "page": page,
                "task_ids": list(task_ids),
            }
        },
    )


def _capture_build_watermark(cfg: object, record: dict[str, Any]) -> None:
    """Stream one immutable legacy inventory into bounded durable pages."""
    build = record["build"]
    build_id = build["build_id"]
    page = 0
    task_count = 0
    task_ids: list[str] = []
    tasks = shared_paths(cfg.shared_root)["tasks"]
    with os.scandir(tasks) as entries:
        for entry in entries:
            if not entry.is_file() or not entry.name.endswith(".json"):
                continue
            task_ids.append(entry.name[:-5])
            task_count += 1
            if len(task_ids) == READY_BUILD_PAGE_SIZE:
                _write_build_page(cfg, build_id, page, task_ids)
                page += 1
                task_ids = []
    if task_ids:
        _write_build_page(cfg, build_id, page, task_ids)
        page += 1
    build["watermark"] = {
        "page_count": page,
        "task_count": task_count,
        "captured_at": utc_now(),
        "is_complete": True,
    }
    build["phase"] = "backfill"


def _reset_ready_projection_for_repair(cfg: object, build_id: str) -> None:
    """Move the damaged advisory projection aside before a truth-based rebuild."""
    paths = shared_paths(cfg.shared_root)
    archive = _build_root(cfg, build_id) / "replaced-projection"
    archive.mkdir(parents=True, exist_ok=True)
    targets = {
        "home": paths["ready_home"],
        "shared": paths["ready_shared"],
        "catalogs": paths["ready_catalogs"],
        "reservations": paths["ready_reservations"],
        "cursors": paths["ready_cursors"],
        "allocators": paths["ready"] / "allocators",
    }
    for name, target in targets.items():
        archived = archive / name
        if target.exists():
            os.replace(target, archived)
        target.mkdir(parents=True, exist_ok=True)


def begin_ready_index_build(cfg: object, *, is_repair: bool = False) -> dict[str, Any]:
    """Start or resume the single durable ready-index build."""
    state.ensure_ready_layout(cfg)
    current_state = state.read_ready_index_state(cfg)
    if current_state == "active" or (current_state == "degraded" and not is_repair):
        return state.read_ready_index_status(cfg)
    if current_state in {"absent", "degraded"}:
        with schema_lock(cfg.shared_root):
            with exclusive(state.state_lock_path(cfg)):
                path = ready_state_path(cfg.shared_root)
                value, record = state.read_state_record(cfg)
                current_state = record["state"]
                if current_state == "active" or (current_state == "degraded" and not is_repair):
                    return record
                if current_state in {"absent", "degraded"}:
                    state.install_writer_capability_gate(cfg)
                    build_id = uuid.uuid4().hex
                    if current_state == "degraded" and is_repair:
                        _reset_ready_projection_for_repair(cfg, build_id)
                    record["state"] = "building"
                    record["writer_capability"] = state.READY_WRITER_CAPABILITY
                    record["build"] = {
                        "build_id": build_id,
                        "phase": "inventory",
                        "is_repair": is_repair,
                        "watermark": {
                            "page_count": 0,
                            "task_count": 0,
                            "captured_at": None,
                            "is_complete": False,
                        },
                        "cursor": {"page": 0, "offset": 0},
                        "audit_cursor": {"page": 0, "offset": 0},
                        "processed": 0,
                        "repaired": 0,
                        "stale_removed": 0,
                        "started_at": utc_now(),
                        "completed_at": None,
                    }
                    state.commit_state_under_lock(path, value, record)
    with exclusive(state.state_lock_path(cfg)):
        path = ready_state_path(cfg.shared_root)
        value, record = state.read_state_record(cfg)
        current_state = record["state"]
        if current_state == "active":
            return record
        if current_state == "degraded" and not is_repair:
            return record
        build = record.get("build")
        if not isinstance(build, dict):
            raise RuntimeError("ready index build state is missing.")
        if not build.get("watermark", {}).get("is_complete"):
            _capture_build_watermark(cfg, record)
            state.commit_state_under_lock(path, value, record)
        return record


def rebuild_primary_ready_index(cfg: object) -> None:
    """Rebuild primary candidates from authoritative queued Task records."""
    state.ensure_ready_layout(cfg)
    with primary_candidates.projection_rebuild_lock(cfg):
        build_id = uuid.uuid4().hex
        atomic_replace(
            primary_candidates.projection_state_path(cfg),
            {
                "primary_ready_index": {
                    "schema_version": primary_candidates.PRIMARY_READY_PROTOCOL_VERSION,
                    "state": "rebuilding",
                    "build_id": build_id,
                    "cleared": False,
                    "updated_at": utc_now(),
                }
            },
        )
        primary_candidates.park_projection_under_lock(cfg, build_id)
        atomic_replace(
            primary_candidates.projection_state_path(cfg),
            {
                "primary_ready_index": {
                    "schema_version": primary_candidates.PRIMARY_READY_PROTOCOL_VERSION,
                    "state": "rebuilding",
                    "build_id": build_id,
                    "cleared": True,
                    "updated_at": utc_now(),
                }
            },
        )
        for path in iter_json(shared_paths(cfg.shared_root)["tasks"]):
            task = TaskRecord.from_dict(read_json(path))
            if not task_should_have_ready_marker(task):
                continue
            reference = routes.reference_for_generation(cfg, task.task_id, task.ready_generation)
            if reference is not None:
                primary_candidates.sync_candidate_under_lock(cfg, task, reference, should_require_active=False)
        atomic_replace(
            primary_candidates.projection_state_path(cfg),
            {
                "primary_ready_index": {
                    "schema_version": primary_candidates.PRIMARY_READY_PROTOCOL_VERSION,
                    "state": "active",
                    "completed_build_id": build_id,
                    "updated_at": utc_now(),
                }
            },
        )


def _repair_task_ready_projection(cfg: object, task_id: str) -> tuple[int, int]:
    """Repair one Task projection under its authority lock."""
    from ..locks import task_writer_lock
    from ..tasks import load_task, save_task

    repaired = 0
    stale_removed = 0
    try:
        initial = load_task(cfg, task_id)
    except FileNotFoundError:
        return repaired, stale_removed
    with task_writer_lock(cfg, task_id, initial.group_name):
        try:
            task = load_task(cfg, task_id)
        except FileNotFoundError:
            return repaired, stale_removed
        reference = routes.reference_for_generation(cfg, task.task_id, task.ready_generation)
        classification = (
            classify_ready_marker(cfg, reference).classification
            if reference is not None and routes.is_reference_indexed(cfg, reference)
            else None
        )
        if task_should_have_ready_marker(task):
            if classification in {"claimable", "temporarily_unavailable"}:
                return repaired, stale_removed
            old_generation, _new_generation = prepare_ready_transition(cfg, task, "ready_index_rebuild")
            task.meta["revision"] += 1
            task.meta["updated_at"] = utc_now()
            save_task(cfg, task)
            retire_previous_ready_generation(cfg, old_generation, task)
            return 1, int(old_generation > 0)
        if reference is not None and delete_ready_marker(cfg, task.task_id, task.ready_generation):
            stale_removed += 1
        return repaired, stale_removed


def _load_build_page(cfg: object, build_id: str, page: int) -> list[str]:
    record = read_json(_build_page_path(cfg, build_id, page))["ready_build_page"]
    task_ids = record.get("task_ids")
    if (
        record.get("schema_version") != state.READY_PROTOCOL_VERSION
        or record.get("build_id") != build_id
        or record.get("page") != page
        or not isinstance(task_ids, list)
        or len(task_ids) > READY_BUILD_PAGE_SIZE
        or not all(isinstance(task_id, str) for task_id in task_ids)
    ):
        raise ValueError("ready build watermark page is invalid.")
    return task_ids


def _advance_build_cursor(
    cursor: dict[str, Any],
    *,
    item_count: int,
    page_count: int,
) -> bool:
    cursor["offset"] += 1
    if cursor["offset"] < item_count:
        return False
    cursor["page"] += 1
    cursor["offset"] = 0
    return cursor["page"] >= page_count


def _active_incompatible_writers(cfg: object) -> list[str]:
    """Return recently active machine agents that did not advertise ready-v1."""
    incompatible: list[str] = []
    machines = shared_paths(cfg.shared_root)["machines"]
    try:
        entries = os.scandir(machines)
    except FileNotFoundError:
        return incompatible
    now = datetime.now(timezone.utc)
    with entries:
        for entry in entries:
            if not entry.is_dir():
                continue
            path = Path(entry.path) / "state" / "agent.json"
            try:
                agent = read_json(path)["agent"]
                heartbeat = datetime.fromisoformat(agent["heartbeat_at"].replace("Z", "+00:00"))
                interval = float(agent["heartbeat_interval_seconds"])
            except (FileNotFoundError, KeyError, TypeError, ValueError):
                continue
            if agent.get("observed_state") not in {"active", "idle"}:
                continue
            if (now - heartbeat).total_seconds() > max(30.0, interval * 3.0):
                continue
            if agent.get("writer_capability") != state.READY_WRITER_CAPABILITY:
                incompatible.append(entry.name)
    return sorted(incompatible)


def _audit_task_ready_projection(cfg: object, task_id: str) -> str | None:
    try:
        task = TaskRecord.from_dict(read_json(task_path(cfg.shared_root, task_id)))
    except FileNotFoundError:
        return None
    except (KeyError, TypeError, ValueError):
        return f"task_invalid:{task_id}"
    reference = routes.reference_for_generation(cfg, task.task_id, task.ready_generation)
    if task_should_have_ready_marker(task):
        if reference is None:
            return f"marker_missing:{task_id}"
        if not routes.is_reference_indexed(cfg, reference):
            return f"marker_unindexed:{task_id}"
        result = classify_ready_marker(cfg, reference)
        if result.classification not in {"claimable", "temporarily_unavailable"}:
            return f"marker_{result.classification}:{task_id}:{result.reason}"
    elif reference is not None:
        return f"marker_stale:{task_id}"
    return None


def ready_task_projection_issue(cfg: object, task_id: str) -> str | None:
    """Return the ready projection defect for one authoritative Task, if any."""
    return _audit_task_ready_projection(cfg, task_id)


def advance_ready_index_build(
    cfg: object,
    *,
    max_tasks: int = READY_BUILD_PAGE_SIZE,
) -> dict[str, Any]:
    """Advance at most ``max_tasks`` durable rebuild or audit records."""
    if type(max_tasks) is not int or not 1 <= max_tasks <= READY_BUILD_PAGE_SIZE:
        raise ValueError(f"max_tasks must be between 1 and {READY_BUILD_PAGE_SIZE}.")
    begin_ready_index_build(cfg)
    # A rebuild can repair Task truth.  Hold the schema fence before the ready
    # state lock so its nested Task writer follows Schema -> state -> Group -> Task.
    with schema_writer_lock(cfg):
        with exclusive(state.state_lock_path(cfg)):
            path = ready_state_path(cfg.shared_root)
            value, record = state.read_state_record(cfg)
            if record["state"] != "building":
                return record
            build = record.get("build")
            if not isinstance(build, dict):
                state.degrade_state_record(record, "build_state_missing")
                state.commit_state_under_lock(path, value, record)
                return record
            watermark = build.get("watermark", {})
            page_count = watermark.get("page_count")
            if type(page_count) is not int or page_count < 0 or not watermark.get("is_complete"):
                state.degrade_state_record(record, "build_watermark_invalid")
                state.commit_state_under_lock(path, value, record)
                return record
            phase = build.get("phase")
            cursor_name = {
                "backfill": "cursor",
                "audit": "audit_cursor",
                "primary-rebuild": "primary_cursor",
            }.get(phase)
            if cursor_name is None:
                state.degrade_state_record(record, f"build_phase_invalid:{phase}")
                state.commit_state_under_lock(path, value, record)
                return record
            cursor = build.get(cursor_name)
            if not isinstance(cursor, dict):
                state.degrade_state_record(record, f"build_cursor_invalid:{cursor_name}")
                state.commit_state_under_lock(path, value, record)
                return record
            processed_now = 0
            try:
                if phase == "primary-rebuild":
                    begin_primary_ready_index_rebuild(cfg, build["build_id"])
                while processed_now < max_tasks and cursor["page"] < page_count:
                    task_ids = _load_build_page(cfg, build["build_id"], cursor["page"])
                    if cursor["offset"] >= len(task_ids):
                        cursor["page"] += 1
                        cursor["offset"] = 0
                        continue
                    task_id = task_ids[cursor["offset"]]
                    if phase == "backfill":
                        repaired, stale_removed = _repair_task_ready_projection(cfg, task_id)
                        build["repaired"] += repaired
                        build["stale_removed"] += stale_removed
                        build["processed"] += 1
                    elif phase == "audit":
                        issue = _audit_task_ready_projection(cfg, task_id)
                        if issue is not None:
                            state.degrade_state_record(record, issue)
                            break
                    else:
                        try:
                            task = TaskRecord.from_dict(read_json(task_path(cfg.shared_root, task_id)))
                        except FileNotFoundError:
                            task = None
                        if task is not None and task_should_have_ready_marker(task):
                            reference = routes.reference_for_generation(
                                cfg,
                                task.task_id,
                                task.ready_generation,
                            )
                            if reference is None:
                                state.degrade_state_record(record, f"marker_missing:{task.task_id}")
                                break
                            primary_candidates.rebuild_primary_ready_candidate(cfg, build["build_id"], task, reference)
                    processed_now += 1
                    _advance_build_cursor(cursor, item_count=len(task_ids), page_count=page_count)
                if record["state"] == "building" and cursor["page"] >= page_count:
                    if phase == "backfill":
                        build["phase"] = "audit"
                    elif phase == "audit":
                        build["phase"] = "primary-rebuild"
                        build["primary_cursor"] = {"page": 0, "offset": 0}
                    else:
                        primary_candidates.complete_primary_ready_index_rebuild(cfg, build["build_id"])
                        state.assert_ready_writer_compatible(cfg)
                        incompatible = _active_incompatible_writers(cfg)
                        if incompatible:
                            state.degrade_state_record(
                                record,
                                "incompatible_active_writers:" + ",".join(incompatible),
                            )
                        else:
                            record["state"] = "active"
                            record["degraded_reasons"] = []
                            build["phase"] = "completed"
                            build["completed_at"] = utc_now()
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                state.degrade_state_record(record, f"build_failed:{type(exc).__name__}:{exc}")
            state.commit_state_under_lock(path, value, record)
            return record


def repair_ready_index(
    cfg: object,
    *,
    max_tasks: int = READY_BUILD_PAGE_SIZE,
) -> dict[str, Any]:
    """Start degraded recovery and advance one bounded repair slice."""
    begin_ready_index_build(cfg, is_repair=True)
    return advance_ready_index_build(cfg, max_tasks=max_tasks)
