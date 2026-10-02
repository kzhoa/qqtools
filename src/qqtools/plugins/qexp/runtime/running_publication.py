"""Shared publication of a registered qexp process as running."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from pathlib import Path

from ..config_types import RootConfig
from .authority_lock import authority_locks
from .paths import attempt_path, shared_paths
from .records import AttemptRecord, utc_now
from .store import atomic_replace, read_json
from .tasks import load_task, save_task


def publish_running_registration(
    cfg: RootConfig,
    registration: Mapping[str, object],
    manifest: Path,
    *,
    mutation_fence: Callable[[], None] | None = None,
) -> bool:
    """Publish shared Attempt and Task truth for a valid process registration.

    The caller validates and captures the local registration before this shared transaction.
    Exact shared identity and lifecycle checks fence the commit. ``manifest`` is a local path
    locator only; this operation never accesses it and requires no local lock to remain held.
    A supplied fence is checked at every shared mutation, including Task indexes.

    Returns:
        Whether this call changed the Attempt phase to running. False also
        covers stale input, detached orphan evidence repair, and Task-only replay;
        it is not proof that the registration is current.
    """
    task_id = registration.get("task_id")
    attempt_id = registration.get("attempt_id")
    token = registration.get("fencing_token")
    reservation_id = registration.get("reservation_id")
    if (
        not isinstance(task_id, str)
        or not isinstance(attempt_id, str)
        or not isinstance(token, int)
        or not isinstance(reservation_id, str)
    ):
        return False

    task = load_task(cfg, task_id)
    with authority_locks(cfg, task):
        task = load_task(cfg, task_id)
        claim = task.claim_control.get("active_claim") or {}
        number = task.attempt_control.get("current_attempt_number")
        if not isinstance(number, int):
            return False
        try:
            attempt_file = attempt_path(cfg.shared_root, task_id, number)
            stored_attempt = read_json(attempt_file)
            attempt = AttemptRecord.from_dict(stored_attempt)
            original_attempt_value = copy.deepcopy(stored_attempt)
        except (FileNotFoundError, KeyError, ValueError):
            return False
        if (
            attempt.attempt_id != attempt_id
            or attempt.current_fencing_token != token
            or attempt.reservation_id != reservation_id
            or attempt.machine_name != cfg.machine_name
        ):
            return False
        is_detached = (
            not claim
            and task.state.get("projection") == "blocked"
            and task.state.get("reason") == "orphaned_attempt_requires_recovery"
            and task.attempt_control.get("current_attempt_id") is None
            and task.attempt_control.get("next_attempt_number") == number + 1
            and task.claim_control.get("fencing_epoch") == token
            and attempt.phase == "orphaned"
        )
        if is_detached:
            # Expiry may beat the asynchronous running publication. Preserve
            # the exact launch's process evidence, never resurrect its lease.
            paths = shared_paths(cfg.shared_root)
            expired_claim = None
            for directory in ("claim_archive", "claim_pending"):
                try:
                    archive = read_json(paths[directory] / task_id / f"{token}.json")["claim_archive"]
                except FileNotFoundError:
                    continue
                if (
                    archive.get("reason") == "lease_expired"
                    and archive.get("task_id") == task_id
                    and archive.get("attempt_id") == attempt_id
                    and archive.get("fencing_token") == token
                ):
                    expired_claim = archive.get("claim")
                    break
            launch_id = attempt.authorization.get("launch_id")
            if (
                not isinstance(expired_claim, dict)
                or not isinstance(launch_id, str)
                or not launch_id
                or expired_claim.get("launch_id") != launch_id
                or expired_claim.get("attempt_id") != attempt_id
                or expired_claim.get("attempt_number") != number
                or expired_claim.get("fencing_token") != token
                or expired_claim.get("reservation_id") != reservation_id
                or expired_claim.get("machine_name") != cfg.machine_name
                or expired_claim.get("launch_state") not in {"starting", "running"}
            ):
                return False
        elif (
            claim.get("attempt_id") != attempt_id
            or claim.get("fencing_token") != token
            or claim.get("reservation_id") != reservation_id
            or claim.get("machine_name") != cfg.machine_name
            or claim.get("launch_state") not in {"starting", "running"}
            or attempt.phase not in {"starting", "running"}
        ):
            return False
        for key in (
            "wrapper_pid",
            "wrapper_start_time_ticks",
            "process_group_id",
            "process_group_start_time_ticks",
        ):
            value = registration.get(key)
            if value is not None:
                existing = attempt.process.get(key)
                if existing is not None and existing != value:
                    return False
                attempt.process[key] = value
        attempt.process["local_process_manifest"] = str(manifest)
        created_at = registration.get("process_created_at")
        if not isinstance(created_at, str):
            return False
        existing_created = attempt.timestamps.get("process_created_at")
        if existing_created is not None and existing_created != created_at:
            return False
        attempt.timestamps["process_created_at"] = created_at
        if not is_detached and attempt.timestamps.get("running_at") is None:
            attempt.timestamps["running_at"] = utc_now()
        was_running = attempt.phase == "running"
        if not is_detached:
            attempt.phase = "running"
        attempt_value = attempt.to_dict()
        if attempt_value != original_attempt_value:
            atomic_replace(
                attempt_file,
                attempt_value,
                before_replace=(lambda _stat: mutation_fence()) if mutation_fence is not None else None,
            )
        if is_detached:
            return False
        if claim.get("launch_state") != "running":
            claim["launch_state"] = "running"
            task.meta["revision"] += 1
            task.meta["updated_at"] = utc_now()
            save_task(cfg, task, mutation_fence=mutation_fence)
        return not was_running
