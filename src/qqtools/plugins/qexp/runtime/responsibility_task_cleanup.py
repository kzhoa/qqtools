"""Explicit Task cleanup retains local obligations lacking shared Attempt truth."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .authority_scan import is_path_present, iter_evidence_files, validate_evidence_path
from .locks import exclusive
from .paths import local_paths, task_path
from .records import validate_identifier
from .responsibility import responsibility_root
from .responsibility_cleanup import (
    CLEANUP_FORMAT,
    FLAT_EVIDENCE,
    CleanupRequest,
    complete_cleanup,
    has_final_observation,
    writers_are_quiescent,
)
from .responsibility_import import RECORD_KEYS, recovery_locator
from .responsibility_store import BUCKETS, Conflict, Ledger
from .store import read_json
from .tasks import load_task

if TYPE_CHECKING:
    from ..config_types import RootConfig
    from .records import AttemptRecord


def _candidates(runtime_root: Path, ledger: Ledger | None) -> tuple[dict[str, dict], dict[str, str | None]]:
    """Inventory explicit maintenance work, including receipt-only responsibilities."""
    candidates: dict[str, dict] = {}

    def include(identity: str, value: dict) -> None:
        validate_identifier(identity, "attempt_id")
        payload = recovery_locator(identity, value)
        previous = candidates.setdefault(identity, {"task_id": None, "attempt_number": None})
        for key, known in payload.items():
            if known is not None:
                if previous[key] not in (None, known):
                    raise Conflict(f"local cleanup evidence has conflicting identity: {identity}")
                previous[key] = known

    paths = local_paths(runtime_root)
    for name, key in RECORD_KEYS.items():
        for path in iter_evidence_files(paths[name], recursive=name == "termination_decisions"):
            if not validate_evidence_path(path, runtime_root):
                continue
            try:
                value = read_json(path).get(key)
            except FileNotFoundError:
                continue
            identity = path.parent.name if name == "termination_decisions" else path.stem
            if not isinstance(value, dict) or value.get("attempt_id", identity) != identity:
                raise ValueError(f"local cleanup evidence does not match its path: {path}")
            include(identity, value)
    local_owners = {identity: payload["task_id"] for identity, payload in candidates.items()}
    if ledger is not None:
        for bucket in range(BUCKETS):
            cursor = None
            while True:
                entries, cursor = ledger.service_page(bucket, cursor)
                for entry in entries:
                    include(entry["identity"], entry["payload"])
                if cursor is None:
                    break
    return candidates, local_owners


def cleanup_unmatched_task_evidence(
    cfg: RootConfig, cleanup: dict, known_attempts: dict[str, AttemptRecord]
) -> tuple[list[str], list[str], set[str]]:
    """Transfer orphan evidence to a receipt under the caller's Task cleanup fence.

    The durable terminal cleanup operation excludes new authorization. Without
    matching Attempt truth, a final protocol-1 observation and writer quiescence
    are both required before handoff. Stored receipts own subsequent replay.
    """
    task_id = validate_identifier(cleanup.get("task_id"), "task_id")
    operation_id = validate_identifier(cleanup.get("operation_id"), "operation_id")
    if (
        cleanup.get("state") not in {"preparing", "waiting_ack"}
        or cleanup.get("terminal_state") not in {"succeeded", "failed", "cancelled"}
        or cfg.machine_name not in cleanup.get("required_machines", [])
    ):
        raise Conflict("unmatched cleanup requires a terminal Task cleanup operation")
    if is_path_present(task_path(cfg.shared_root, task_id)):
        task = load_task(cfg, task_id)
        if (
            task.control.get("cleanup_operation_id") != operation_id
            or task.state.get("projection") not in {"succeeded", "failed", "cancelled"}
            or task.claim_control.get("active_claim")
        ):
            raise Conflict("unmatched cleanup operation does not own terminal Task truth")
    root = responsibility_root(cfg.runtime_root)
    ledger = Ledger(root) if is_path_present(root) else None
    candidates, local_owners = _candidates(cfg.runtime_root, ledger)
    removed, blockers, identities = [], [], set()
    paths = local_paths(cfg.runtime_root)
    for identity, payload in candidates.items():
        if payload["task_id"] != task_id:
            continue
        identities.add(identity)
        known = known_attempts.get(identity)
        if known is not None:
            if known.machine_name != cfg.machine_name:
                blockers.append(f"local_attempt_owner_mismatch:{identity}")
            continue
        entry = ledger.find(identity) if ledger is not None else None
        request = CleanupRequest.from_entry(entry) if entry is not None else None
        if request is not None and request.payload != payload:
            raise Conflict("cleanup receipt does not match the observed local identity")
        if request is None:
            if (
                local_owners.get(identity) != task_id
                or not has_final_observation(cfg.runtime_root, task_id, identity)
                or not writers_are_quiescent(cfg.runtime_root, task_id, identity)
            ):
                blockers.append(f"local_writer_unresolved:{identity}")
                continue
            request = CleanupRequest(
                identity,
                payload,
                {
                    "format": CLEANUP_FORMAT,
                    "task_id": task_id,
                    "attempt_id": identity,
                    "basis": "task_cleanup",
                    "operation_id": operation_id,
                },
            )
        elif request.receipt.get("basis") == "task_cleanup" and request.receipt["operation_id"] != operation_id:
            raise Conflict("cleanup receipt belongs to another Task cleanup operation")
        evidence = [paths[name] / f"{identity}.json" for name in FLAT_EVIDENCE]
        evidence.append(paths["termination_decisions"] / identity)
        existing = [path for path in evidence if is_path_present(path)]
        if ledger is None:
            with exclusive(cfg.runtime_root / "locks" / "responsibility-initialize.lock"):
                ledger = Ledger.open_or_create(root)
        if not complete_cleanup(ledger, cfg.runtime_root, request):
            blockers.append(f"local_cleanup_pending:{identity}")
        removed.extend(str(path) for path in existing if not is_path_present(path))
    return removed, blockers, identities
