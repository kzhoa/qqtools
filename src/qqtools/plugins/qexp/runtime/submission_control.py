"""Bounded, revision-bound visibility for Submission operation state."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any, Callable

from ..config_types import RootConfig
from .locks import exclusive
from .paths import submission_path
from .records import validate_identifier
from .store import CASConflict, atomic_replace, create_if_absent, read_json

CONTROL_VERSION = 1
CONTROL_RECORD_LIMIT = 8192
SUBMISSION_SOURCE_LIMIT = 65536

_CONTROL_STATES = frozenset({"building", "active"})
_SUBMISSION_STATES = frozenset({"preparing", "committing", "committed", "aborted", "blocked"})
_CONTROL_FIELDS = frozenset({"version", "root", "state", "cursor"})
_RECEIPT_FIELDS = frozenset({"version", "root", "operation_id", "source_revision", "state"})

AtomicWriter = Callable[..., os.stat_result | None]


class SubmissionControlUnavailable(ValueError):
    """Derived Submission visibility cannot be established safely."""


def _root(cfg: RootConfig) -> Path:
    return Path(cfg.shared_root).expanduser().resolve()


def _control_root(cfg: RootConfig) -> Path:
    return _root(cfg) / "indexes" / "submission-control"


def control_paths(cfg: RootConfig) -> dict[str, Path]:
    """Return the durable Submission-control layout for one project root."""
    root = _control_root(cfg)
    return {
        "root": root,
        "state": root / "state.json",
        "records": root / "records",
        "pending": root / "pending",
        "checkpoints": root / "checkpoints",
        "locks": root / "locks",
    }


def _checked_operation_id(operation_id: str) -> str:
    return validate_identifier(operation_id, "operation_id")


def record_path(cfg: RootConfig, operation_id: str) -> Path:
    """Return the derived receipt path for an operation."""
    return control_paths(cfg)["records"] / f"{_checked_operation_id(operation_id)}.json"


def pending_path(cfg: RootConfig, operation_id: str) -> Path:
    """Return the replayable publication-intent path for an operation."""
    return control_paths(cfg)["pending"] / f"{_checked_operation_id(operation_id)}.json"


def operation_lock_path(cfg: RootConfig, operation_id: str) -> Path:
    """Return the per-operation control lock path."""
    return control_paths(cfg)["locks"] / f"{_checked_operation_id(operation_id)}.lock"


def source_revision(value: os.stat_result) -> list[int]:
    """Return the portable source revision fields bound into a derived receipt."""
    try:
        # Device numbers are host-local on shared filesystems.  Keep the five-field
        # shape while reserving the first field for a shared-root-relative identity.
        return [0, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]
    except AttributeError as exc:
        raise ValueError("source stat witness is invalid") from exc


def _encode(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")


def _require_size(value: dict[str, Any], maximum: int, label: str) -> None:
    if len(_encode(value)) > maximum:
        raise ValueError(f"{label} exceeds its {maximum}-byte limit.")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r} is not allowed")


def _unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def read_control_record(path: Path, max_bytes: int = CONTROL_RECORD_LIMIT) -> dict[str, Any]:
    """Read one regular, non-symlink JSON object within a structural byte bound."""
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer.")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"JSON record is not a regular file: {path}")
        chunks = bytearray()
        while len(chunks) <= max_bytes:
            chunk = os.read(descriptor, max_bytes + 1 - len(chunks))
            if not chunk:
                break
            chunks.extend(chunk)
            if len(chunks) > max_bytes:
                raise ValueError(f"JSON record exceeds its {max_bytes}-byte limit: {path}.")
    finally:
        os.close(descriptor)
    try:
        value = json.loads(
            bytes(chunks).decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_fields,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"JSON record is malformed: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object at {path}.")
    return value


def _ensure_control_directories(cfg: RootConfig) -> dict[str, Path]:
    paths = control_paths(cfg)
    for directory in (
        paths["root"].parent,
        paths["root"],
        *(paths[name] for name in ("records", "pending", "checkpoints", "locks")),
    ):
        try:
            directory.mkdir()
        except FileExistsError:
            if not stat.S_ISDIR(directory.lstat().st_mode):
                raise ValueError(f"Submission-control directory is not a regular directory: {directory}")
        else:
            for path in (directory, directory.parent):
                descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
    return paths


def _validate_revision(value: object) -> list[int]:
    if type(value) is not list or len(value) != 5 or any(type(item) is not int for item in value):
        raise ValueError("source_revision must contain exactly five integers.")
    if value[0] != 0 or value[1] < 0 or value[2] < 0:
        raise ValueError("source_revision has invalid shared-file identity fields.")
    return list(value)


def _validate_root(value: object, cfg: RootConfig) -> None:
    if type(value) is not str or value != str(_root(cfg)):
        raise ValueError("control record root does not match the configured project root.")


def _validate_control_state(value: object, cfg: RootConfig) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _CONTROL_FIELDS:
        raise ValueError("Submission-control state has invalid fields.")
    if type(value["version"]) is not int or value["version"] != CONTROL_VERSION:
        raise ValueError("Submission-control state version is invalid.")
    _validate_root(value["root"], cfg)
    if type(value["state"]) is not str or value["state"] not in _CONTROL_STATES:
        raise ValueError("Submission-control state is invalid.")
    if value["cursor"] is not None and not isinstance(value["cursor"], dict):
        raise ValueError("Submission-control cursor must be null or an object.")
    return value


def read_control_state(cfg: RootConfig) -> dict[str, Any] | None:
    """Read activation state, treating an absent file as a legacy building root."""
    path = control_paths(cfg)["state"]
    try:
        value = read_control_record(path)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise SubmissionControlUnavailable(f"Submission-control state is unavailable: {path}") from exc
    try:
        return _validate_control_state(value, cfg)
    except ValueError as exc:
        raise SubmissionControlUnavailable(f"Submission-control state is invalid: {path}") from exc


def write_control_state(cfg: RootConfig, value: dict[str, Any]) -> None:
    """Persist validated Submission-control activation state."""
    _validate_control_state(value, cfg)
    _require_size(value, CONTROL_RECORD_LIMIT, "Submission-control state")
    paths = _ensure_control_directories(cfg)
    atomic_replace(paths["state"], value)


def inspect_submission_control(cfg: RootConfig) -> dict[str, Any]:
    """Report bounded activation status without inspecting Submission history."""
    try:
        value = read_control_state(cfg)
    except SubmissionControlUnavailable:
        return {"state": "unavailable", "reason": "control_state_invalid"}
    if value is None:
        return {"state": "building", "reason": "bootstrap_pending"}
    return {"state": value["state"], "reason": None}


def request_control_rebuild(cfg: RootConfig) -> dict[str, Any]:
    """Restart derived certification under the background maintenance lock."""
    paths = _ensure_control_directories(cfg)
    with exclusive(paths["locks"] / "maintenance.lock", blocking=False) as acquired:
        if not acquired:
            return {"state": "waiting", "reason": "maintenance_busy"}
        write_control_state(
            cfg, {"version": CONTROL_VERSION, "root": str(_root(cfg)), "state": "building", "cursor": None}
        )
    return {"state": "building", "reason": "rebuild_requested"}


def _create_control_state_if_absent(cfg: RootConfig, state: str) -> None:
    value = {"version": CONTROL_VERSION, "root": str(_root(cfg)), "state": state, "cursor": None}
    _require_size(value, CONTROL_RECORD_LIMIT, "Submission-control state")
    paths = _ensure_control_directories(cfg)
    try:
        create_if_absent(paths["state"], value)
    except CASConflict:
        pass


def initialize_empty(cfg: RootConfig) -> None:
    """Create an active control layout for a newly initialized project root."""
    _create_control_state_if_absent(cfg, "active")


def initialize_build(cfg: RootConfig) -> None:
    """Create a building control layout without inspecting authoritative records."""
    _create_control_state_if_absent(cfg, "building")


def _source_lstat(path: Path) -> os.stat_result:
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"Submission source is not a regular file: {path}")
    return metadata


def _same_revision(first: os.stat_result, second: os.stat_result) -> bool:
    return source_revision(first) == source_revision(second)


def _require_same_source(path: Path, expected: os.stat_result) -> os.stat_result:
    try:
        current = _source_lstat(path)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SubmissionControlUnavailable(f"Submission source disappeared or became invalid: {path}") from exc
    if not _same_revision(expected, current):
        raise SubmissionControlUnavailable(f"Submission source changed during state read: {path}")
    return current


def _submission_state(value: dict[str, Any]) -> str:
    try:
        submission = value["submission"]
    except KeyError as exc:
        raise ValueError("Submission record has no submission object.") from exc
    if not isinstance(submission, dict):
        raise ValueError("Submission record has an invalid submission object.")
    state = submission.get("state")
    if type(state) is not str or state not in _SUBMISSION_STATES:
        raise ValueError("Submission state is invalid.")
    return state


def _read_receipt_state(cfg: RootConfig, operation_id: str, expected: os.stat_result) -> str | None:
    try:
        receipt = read_control_record(record_path(cfg, operation_id))
    except (FileNotFoundError, OSError, ValueError):
        return None
    if set(receipt) != _RECEIPT_FIELDS:
        return None
    if type(receipt["version"]) is not int or receipt["version"] != CONTROL_VERSION:
        return None
    if type(receipt["root"]) is not str or receipt["root"] != str(_root(cfg)):
        return None
    if receipt["operation_id"] != operation_id:
        return None
    try:
        revision = _validate_revision(receipt["source_revision"])
    except ValueError:
        return None
    if revision != source_revision(expected):
        return None
    receipt_state = receipt["state"]
    if type(receipt_state) is not str or receipt_state not in _SUBMISSION_STATES:
        return None
    try:
        _require_same_source(submission_path(_root(cfg), operation_id), expected)
    except SubmissionControlUnavailable:
        return None
    return receipt_state


def _read_source_state_bounded(path: Path, expected: os.stat_result) -> str:
    value = read_control_record(path, max_bytes=SUBMISSION_SOURCE_LIMIT)
    state = _submission_state(value)
    _require_same_source(path, expected)
    return state


def _read_source_state_legacy(path: Path, expected: os.stat_result) -> str:
    value = read_json(path)
    state = _submission_state(value)
    _require_same_source(path, expected)
    return state


def read_submission_state(cfg: RootConfig, operation_id: str) -> str:
    """Read one Submission state with a revision-bound derived-proof fast path."""
    operation_id = _checked_operation_id(operation_id)
    path = submission_path(_root(cfg), operation_id)
    source_before = _source_lstat(path)
    if source_before.st_size <= SUBMISSION_SOURCE_LIMIT:
        return _read_source_state_bounded(path, source_before)
    control_state = read_control_state(cfg)
    receipt_state = _read_receipt_state(cfg, operation_id, source_before)
    if receipt_state is not None:
        return receipt_state

    source_before = _source_lstat(path)
    if source_before.st_size <= SUBMISSION_SOURCE_LIMIT:
        return _read_source_state_bounded(path, source_before)

    if control_state is None or control_state["state"] == "building":
        return _read_source_state_legacy(path, source_before)

    request_repair(cfg, operation_id)
    raise SubmissionControlUnavailable(f"large Submission source requires repair: {operation_id}")


def _pending_value(cfg: RootConfig, operation_id: str) -> dict[str, Any]:
    return {"version": CONTROL_VERSION, "root": str(_root(cfg)), "operation_id": operation_id}


def _pending_after_image(
    cfg: RootConfig,
    operation_id: str,
    state: str,
    witness: os.stat_result,
) -> dict[str, Any]:
    value = _pending_value(cfg, operation_id)
    value.update({"state": state, "after_image": source_revision(witness)[:4]})
    _require_size(value, CONTROL_RECORD_LIMIT, "Submission publication intent")
    return value


def _ensure_pending(
    cfg: RootConfig,
    operation_id: str,
    writer: AtomicWriter,
) -> None:
    path = pending_path(cfg, operation_id)
    if not path.exists():
        writer(path, _pending_value(cfg, operation_id))


def request_repair(cfg: RootConfig, operation_id: str) -> bool:
    """Queue one replayable proof repair without blocking a foreground reader."""
    operation_id = _checked_operation_id(operation_id)
    source = submission_path(_root(cfg), operation_id)
    try:
        with exclusive(operation_lock_path(cfg, operation_id), blocking=False) as acquired:
            if not acquired:
                return False
            _source_lstat(source)
            _ensure_control_directories(cfg)
            _ensure_pending(cfg, operation_id, atomic_replace)
            return True
    except SubmissionControlUnavailable:
        raise
    except (OSError, ValueError) as exc:
        raise SubmissionControlUnavailable(f"could not request Submission repair: {operation_id}") from exc


def _receipt(
    cfg: RootConfig,
    operation_id: str,
    state: str,
    revision: list[int],
) -> dict[str, Any]:
    return {
        "version": CONTROL_VERSION,
        "root": str(_root(cfg)),
        "operation_id": operation_id,
        "source_revision": revision,
        "state": state,
    }


def write_receipt(
    cfg: RootConfig,
    operation_id: str,
    state: str,
    revision: list[int],
    *,
    _atomic_writer: AtomicWriter | None = None,
) -> None:
    """Write a proof only while the authoritative source keeps its revision."""
    operation_id = _checked_operation_id(operation_id)
    if type(state) is not str or state not in _SUBMISSION_STATES:
        raise ValueError("Submission receipt state is invalid.")
    revision = _validate_revision(revision)
    value = _receipt(cfg, operation_id, state, revision)
    if set(value) != _RECEIPT_FIELDS:
        raise ValueError("Submission receipt has invalid fields.")
    _require_size(value, CONTROL_RECORD_LIMIT, "Submission receipt")
    source = submission_path(_root(cfg), operation_id)
    try:
        before = _source_lstat(source)
    except (OSError, ValueError) as exc:
        raise SubmissionControlUnavailable(f"Submission source is unavailable: {source}") from exc
    if source_revision(before) != revision:
        raise SubmissionControlUnavailable(f"Submission source revision changed before receipt: {operation_id}")
    paths = _ensure_control_directories(cfg)
    writer = _atomic_writer or atomic_replace
    writer(paths["records"] / f"{operation_id}.json", value)
    try:
        after = _source_lstat(source)
    except (OSError, ValueError) as exc:
        raise SubmissionControlUnavailable(f"Submission source is unavailable after receipt: {source}") from exc
    if source_revision(after) != revision:
        raise SubmissionControlUnavailable(f"Submission source revision changed after receipt: {operation_id}")


def _retire_pending(cfg: RootConfig, operation_id: str) -> None:
    path = pending_path(cfg, operation_id)
    path.unlink(missing_ok=True)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_operation(operation: dict[str, Any]) -> tuple[str, str]:
    if not isinstance(operation, dict):
        raise ValueError("Submission operation must be a mapping.")
    submission = operation.get("submission")
    if not isinstance(submission, dict):
        raise ValueError("Submission operation has an invalid submission object.")
    operation_id = _checked_operation_id(submission.get("operation_id"))
    state = submission.get("state")
    if type(state) is not str or state not in _SUBMISSION_STATES:
        raise ValueError("Submission operation state is invalid.")
    return operation_id, state


def publish_submission(
    cfg: RootConfig,
    operation: dict[str, Any],
    *,
    _atomic_writer: AtomicWriter | None = None,
) -> None:
    """Publish Submission truth and, for large records, its replayable proof intent."""
    operation_id, state = _validate_operation(operation)
    writer = _atomic_writer or atomic_replace
    encoded_size = len(_encode(operation))
    source = submission_path(_root(cfg), operation_id)
    if encoded_size <= SUBMISSION_SOURCE_LIMIT:
        writer(source, operation)
        return

    _ensure_control_directories(cfg)
    with exclusive(operation_lock_path(cfg, operation_id)):
        _ensure_pending(cfg, operation_id, writer)

        def record_after_image(witness: os.stat_result) -> None:
            writer(pending_path(cfg, operation_id), _pending_after_image(cfg, operation_id, state, witness))

        witness = writer(source, operation, before_replace=record_after_image)
        if witness is None:
            return
        try:
            write_receipt(
                cfg,
                operation_id,
                state,
                source_revision(witness),
                _atomic_writer=writer,
            )
            _retire_pending(cfg, operation_id)
        except (OSError, ValueError):
            return
