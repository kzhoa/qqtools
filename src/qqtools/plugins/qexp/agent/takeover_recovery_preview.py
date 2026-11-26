"""Read-only observation of a departing Project's recovery ledger membership."""

from __future__ import annotations

import base64
import hashlib
import json
import stat
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..lease import parse_utc
from ..runtime.authority_scan import is_path_present, validate_evidence_path
from ..runtime.paths import local_paths
from ..runtime.records import validate_identifier
from ..runtime.registration_protocol import _read_protocol_record
from ..runtime.responsibility import responsibility_root
from ..runtime.responsibility_cleanup import FLAT_EVIDENCE
from ..runtime.responsibility_import import recovery_locator
from ..runtime.responsibility_store import BUCKETS, MAX_COUNTER, Conflict, Ledger

if TYPE_CHECKING:
    from .bindings import ProjectBinding
    from .context import MachineRuntime

_PAGE_SIZE = 25
_CURSOR_LIMIT = 1024
_INVALID_CURSOR = "Recovery continuation is invalid. Start a fresh preview without --recovery-cursor."
_STALE_CURSOR = "Recovery continuation is stale. Start a fresh preview without --recovery-cursor."


class RecoveryPreviewCursorError(ValueError):
    """The advisory recovery continuation is malformed or no longer applicable."""


def _cursor_scope(binding: ProjectBinding, instance: str) -> str:
    # A fixed digest keeps the cursor bounded even for long registration generations.
    encoded = json.dumps([binding.project_id, binding.registration_generation, instance], separators=(",", ":"))
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _encode_cursor(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    return base64.urlsafe_b64encode(encoded).decode("ascii").rstrip("=")


def _decode_cursor(cursor: str, scope: str, instance: str) -> tuple[int, dict | None]:
    if not isinstance(cursor, str) or not 1 <= len(cursor) <= _CURSOR_LIMIT:
        raise RecoveryPreviewCursorError(_INVALID_CURSOR)
    try:
        raw = cursor.encode("ascii")
        decoded = base64.b64decode(raw + b"=" * (-len(raw) % 4), altchars=b"-_", validate=True)
        value = json.loads(decoded.decode("utf-8"))
        if (
            not isinstance(value, dict)
            or set(value) != {"scope", "bucket", "continuation"}
            or not isinstance(value["scope"], str)
            or _encode_cursor(value) != cursor
        ):
            raise ValueError("invalid recovery cursor encoding")
    except (ValueError, RecursionError) as exc:
        raise RecoveryPreviewCursorError(_INVALID_CURSOR) from exc

    if value["scope"] != scope:
        raise RecoveryPreviewCursorError(_STALE_CURSOR)
    bucket = value["bucket"]
    continuation = value["continuation"]
    if type(bucket) is not int or not 0 <= bucket < BUCKETS:
        raise RecoveryPreviewCursorError(_INVALID_CURSOR)
    if continuation is not None and (
        not isinstance(continuation, dict)
        or set(continuation) != {"instance", "bucket", "revision", "slot"}
        or continuation["instance"] != instance
        or type(continuation["bucket"]) is not int
        or continuation["bucket"] != bucket
        or type(continuation["revision"]) is not int
        or not 0 <= continuation["revision"] <= MAX_COUNTER
        or type(continuation["slot"]) is not int
        or not 1 <= continuation["slot"] < MAX_COUNTER
    ):
        raise RecoveryPreviewCursorError(_INVALID_CURSOR)
    return bucket, continuation


def _validate_local_id(value: str, label: str) -> str:
    validate_identifier(value, label)
    if value in {".", ".."}:
        raise ValueError(f"{label} cannot be '.' or '..'.")
    return value


def _local_directory_present(path: Path) -> bool:
    if not is_path_present(path):
        return False
    if not stat.S_ISDIR(path.stat(follow_symlinks=False).st_mode):
        raise RuntimeError(f"Recovery preview expected a local directory at {path}.")
    return True


def _is_control_plane_diagnostic(root: Path, entry: dict) -> bool:
    payload = entry["payload"]
    if payload != {"task_id": None, "attempt_number": None} or any(
        entry.get(field) is not None
        for field in ("legacy_source", "captured_writers", "writer_capture_incomplete", "cleanup_receipt")
    ):
        return False

    paths = local_paths(root)
    path = paths["authority_diagnostics"] / "control-plane.json"
    if not validate_evidence_path(path, root):
        return False
    document = _read_protocol_record(path)
    if document is None or set(document) != {"authority_diagnostic"}:
        return False
    diagnostic = document["authority_diagnostic"]
    if (
        not isinstance(diagnostic, dict)
        or set(diagnostic) != {"attempt_id", "reason", "at", "error_type", "error"}
        or diagnostic["attempt_id"] != "control-plane"
        or diagnostic["reason"] != "cleanup_replay_unavailable"
        or not all(isinstance(diagnostic[field], str) for field in ("at", "error_type", "error"))
        or not diagnostic["error_type"]
    ):
        return False
    try:
        timestamp = parse_utc(diagnostic["at"])
        if diagnostic["at"] != timestamp.isoformat(timespec="seconds").replace("+00:00", "Z"):
            return False
        if not diagnostic["at"].endswith("Z"):
            return False
    except ValueError:
        return False

    for name in FLAT_EVIDENCE:
        if (
            name != "authority_diagnostics"
            and _local_directory_present(paths[name])
            and is_path_present(paths[name] / "control-plane.json")
        ):
            return False
    decisions = paths["termination_decisions"]
    return not (_local_directory_present(decisions) and is_path_present(decisions / "control-plane"))


def _describe_record(root: Path, entry: dict) -> dict[str, Any]:
    identity = _validate_local_id(entry["identity"], "recovery identity")
    try:
        locator = recovery_locator(identity, entry["payload"])
        if locator["task_id"] is not None:
            _validate_local_id(locator["task_id"], "task_id")
    except (ValueError, Conflict):
        locator = {"task_id": None, "attempt_number": None}

    kind = "unidentified"
    if locator["task_id"] is not None and locator["attempt_number"] is not None:
        kind = "attempt"
    elif identity == "control-plane" and _is_control_plane_diagnostic(root, entry):
        kind = "control_plane_diagnostic"
    return {
        "identity": identity,
        **locator,
        "kind": kind,
        "cleanup_receipt_recorded": entry.get("cleanup_receipt") is not None,
    }


def observe_departing_recovery(
    runtime: MachineRuntime, binding: ProjectBinding, *, cursor: str | None = None
) -> dict[str, Any]:
    """Observe bounded ledger membership without granting cleanup authority."""
    if binding.project_id == "." or "\x00" in binding.project_id:
        raise ValueError("Project ID is not a valid path component.")
    project_root = runtime.project_paths(binding.project_id)["root"]
    root = responsibility_root(project_root)
    if not all(_local_directory_present(path) for path in (project_root.parent, project_root, root)):
        if cursor is not None:
            raise RecoveryPreviewCursorError(_STALE_CURSOR)
        return {"status": "absent", "count": 0, "records": [], "next_cursor": None}

    ledger = Ledger(root, is_observation=True)
    scope = _cursor_scope(binding, ledger.instance)
    start, continuation = _decode_cursor(cursor, scope, ledger.instance) if cursor is not None else (0, None)
    counts = [ledger.observe_count(bucket) for bucket in range(BUCKETS)]
    if continuation is not None and continuation["slot"] >= counts[start]:
        raise RecoveryPreviewCursorError(_STALE_CURSOR)

    records = []
    next_cursor = None
    for bucket in range(start, BUCKETS):
        if counts[bucket] == 0:
            continue
        if len(records) == _PAGE_SIZE:
            next_cursor = _encode_cursor({"scope": scope, "bucket": bucket, "continuation": None})
            break
        while len(records) < _PAGE_SIZE:
            try:
                entries, continuation, count = ledger.observe_page(
                    bucket, continuation, limit=_PAGE_SIZE - len(records)
                )
            except Conflict as exc:
                raise RecoveryPreviewCursorError(_STALE_CURSOR) from exc
            if count != counts[bucket]:
                raise RuntimeError(f"Recovery ledger membership changed during preview in bucket {bucket}.")
            records.extend(_describe_record(project_root, entry) for entry in entries)
            if continuation is None:
                break
        if continuation is not None:
            next_cursor = _encode_cursor({"scope": scope, "bucket": bucket, "continuation": continuation})
            break

    return {"status": "observed", "count": sum(counts), "records": records, "next_cursor": next_cursor}
