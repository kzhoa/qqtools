"""Durable ordered Group recheck publication.

The journal records only the ordered recheck barrier and its caller-supplied
evidence.  It does not acquire the authoritative Group writer fence, inspect
or mutate Group or Task records, or replay any Task effect.  The caller owns
those responsibilities and must hold the schema and Group fences while using
this module.

Records use ordinary bounded-per-record JSON replacement.  State records are
limited to 64 KiB; event records are read with the regular JSON loader so the
caller can carry its bounded evidence without a second artificial token cap.
The filesystem layout is initialized and directory-synced explicitly because
``atomic_replace`` only makes the replaced file and its immediate parent
durable.
"""

from __future__ import annotations

import json
import math
import os
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..group_namespace import group_authority_identity
from ..records import validate_identifier
from ..store import atomic_replace, read_json, read_json_limited
from .coverage import GroupCoverage

_STATE_MAX_BYTES = 64 * 1024
_STATE_KEYS = frozenset({"version", "identity", "generation", "tail"})
_RETENTION_KEYS = frozenset({"version", "identity", "generation", "floor", "deleted"})
_EVENT_KEYS = frozenset(
    {
        "version",
        "identity",
        "generation",
        "sequence",
        "task_id",
        "submission_operation_id",
        "membership_sequence",
        "owner",
        "evidence",
        "state",
    }
)
_OWNERS = frozenset(
    {"retry", "claim", "claim_loss", "launch", "availability", "terminal", "recovery", "cleanup", "task_cancel"}
)
_EVENT_STATES = frozenset({"in_flight", "committed", "aborted"})


@dataclass(frozen=True, slots=True)
class RecheckPosition:
    """A snapshot of one journal generation and its durable event tail."""

    generation: int
    tail: int


@dataclass(frozen=True, slots=True)
class RecheckTicket:
    """The publication authority returned after an event and its tail commit."""

    generation: int
    sequence: int


@dataclass(frozen=True, slots=True)
class RecheckRetention:
    """The durable retirement boundary for one recheck journal generation."""

    generation: int
    floor: int
    deleted: int


class GroupRechecks:
    """Persist ordered recheck barriers for one Group authority identity."""

    def __init__(self, root: Path, group: str):
        # GroupCoverage validates the pure constructor inputs without touching
        # the authority namespace.  Its directory property is deliberately
        # deferred until the first operation so construction remains pure.
        if not isinstance(root, (Path, str)):
            raise TypeError("root must be a path")
        if not isinstance(group, str):
            raise TypeError("group must be a string")
        self._root = Path(os.path.abspath(os.fspath(root)))
        self._group = group
        self._coverage = GroupCoverage(self._root, group)
        self._identity: dict[str, Any] | None = None
        self._directory: Path | None = None

    def snapshot(self, *, initialize: bool = False) -> RecheckPosition | None:
        """Return the current position, optionally publishing the initial journal.

        An uninitialized journal is dormant.  ``initialize=True`` creates the
        fixed layout and a fresh positive generation, while retrying an
        interrupted layout that has directories but no committed state record.
        """

        if type(initialize) is not bool:
            raise TypeError("initialize must be a bool")
        paths, identity = self._bound_paths()
        journal_present = self._inspect_parent_layout(paths)
        if not _record_present(paths["state"], "recheck state"):
            if not initialize:
                if journal_present:
                    raise ValueError("recheck state missing")
                return None
            return self._initialize(paths, identity)
        return self._read_position(paths, identity)

    def begin(
        self,
        task_id: str,
        submission_operation_id: str,
        sequence: int,
        owner: str,
        evidence: dict[str, Any],
    ) -> RecheckTicket | None:
        """Publish an in-flight event before advancing the reachable tail.

        If the journal has not been initialized, the call is a no-op.  Once a
        journal exists, the event itself is durable before its state tail is
        advanced, so a returned ticket always has a reachable event.
        """

        paths, identity = self._bound_paths()
        journal_present = self._inspect_parent_layout(paths)
        if not _record_present(paths["state"], "recheck state"):
            if journal_present:
                raise ValueError("recheck state missing")
            return None
        position = self._read_position(paths, identity)
        validate_identifier(task_id, "task_id")
        validate_identifier(submission_operation_id, "submission_operation_id")
        _validate_positive_int(sequence, "membership_sequence")
        _validate_owner(owner)
        _validate_evidence(evidence)

        journal_sequence = position.tail + 1
        generation_path = paths["journal"] / str(position.generation)
        _require_directory(generation_path, "recheck generation")
        event_path = generation_path / f"{journal_sequence}.json"
        _validate_event_target_for_replace(event_path)
        event = {
            "version": 1,
            "identity": identity,
            "generation": position.generation,
            "sequence": journal_sequence,
            "task_id": task_id,
            "submission_operation_id": submission_operation_id,
            "membership_sequence": sequence,
            "owner": owner,
            "evidence": evidence,
            "state": "in_flight",
        }
        # The generation directory was made durable before this journal was
        # published; atomic_replace then fsyncs the event and that directory.
        atomic_replace(event_path, event)

        _require_regular_file(paths["state"], "recheck state")
        next_state = _state_record(identity, position.generation, journal_sequence)
        atomic_replace(paths["state"], next_state)
        return RecheckTicket(position.generation, journal_sequence)

    def read(self, position: RecheckPosition, sequence: int) -> dict[str, Any]:
        """Read one event visible from the supplied current-generation position."""

        _validate_position(position)
        _validate_positive_int(sequence, "sequence")
        paths, identity = self._bound_paths()
        self._inspect_parent_layout(paths)
        current = self._read_position(paths, identity)
        if position.generation != current.generation:
            raise ValueError("recheck position belongs to an inactive generation")
        retention = self._read_retention(paths, identity, current)
        if sequence <= retention.floor:
            raise ValueError("recheck sequence has been retired")
        if sequence > position.tail or sequence > current.tail:
            raise ValueError("recheck sequence is outside the supplied position")
        event_path = paths["journal"] / str(current.generation) / f"{sequence}.json"
        return self._read_event(event_path, identity, current.generation, sequence)

    def resolve(self, ticket: RecheckTicket, *, outcome: str = "committed") -> None:
        """Resolve an in-flight event, ignoring tickets from invalidated generations."""

        _validate_ticket(ticket)
        if type(outcome) is not str or outcome not in {"committed", "aborted"}:
            raise ValueError("recheck outcome must be 'committed' or 'aborted'")
        paths, identity = self._bound_paths()
        self._inspect_parent_layout(paths)
        current = self._read_position(paths, identity)
        if current.generation != ticket.generation:
            # Invalidation revokes publication authority for every prior
            # ticket.  Do not even inspect the old generation's event.
            return
        if ticket.sequence > current.tail:
            raise ValueError("recheck ticket is beyond the durable tail")
        retention = self._read_retention(paths, identity, current)
        if ticket.sequence <= retention.floor:
            return
        event_path = paths["journal"] / str(ticket.generation) / f"{ticket.sequence}.json"
        event = self._read_event(event_path, identity, ticket.generation, ticket.sequence)
        current_state = event["state"]
        if current_state == outcome:
            return
        if current_state != "in_flight":
            raise ValueError("recheck event has already been resolved differently")
        _validate_event_target_for_replace(event_path)
        event["state"] = outcome
        atomic_replace(event_path, event)

    def retention(self, position: RecheckPosition) -> RecheckRetention:
        """Return the durable retirement boundary for the supplied position."""

        _validate_position(position)
        paths, identity = self._bound_paths()
        self._inspect_parent_layout(paths)
        current = self._read_position(paths, identity)
        if position.generation != current.generation:
            raise ValueError("recheck position belongs to an inactive generation")
        return self._read_retention(paths, identity, current)

    def reclaim(self, position: RecheckPosition, through: int, *, max_events: int = 1) -> int:
        """Retire resolved events through a caller-certified subscriber cursor."""

        _validate_position(position)
        _validate_nonnegative_int(through, "through")
        if through > position.tail:
            raise ValueError("recheck retention cursor is beyond the supplied position")
        _validate_reclaim_limit(max_events)

        paths, identity = self._bound_paths()
        self._inspect_parent_layout(paths)
        current = self._read_position(paths, identity)
        if position.generation != current.generation:
            raise ValueError("recheck position belongs to an inactive generation")
        if position.tail > current.tail:
            raise ValueError("recheck position is beyond the durable tail")
        retention = self._read_retention(paths, identity, current)
        generation_path = paths["journal"] / str(current.generation)
        _require_directory(generation_path, "recheck generation")

        reclaimed = 0
        while reclaimed < max_events:
            if retention.floor > retention.deleted:
                sequence = retention.deleted + 1
                event_path = generation_path / f"{sequence}.json"
                _validate_event_target_for_replace(event_path)
                event_path.unlink(missing_ok=True)
                _sync_directory(generation_path)
                retention = RecheckRetention(current.generation, retention.floor, sequence)
                _write_retention(paths["retention"], identity, retention)
                reclaimed += 1
                continue

            sequence = retention.deleted + 1
            if sequence > through:
                break
            event_path = generation_path / f"{sequence}.json"
            event = self._read_event(event_path, identity, current.generation, sequence)
            if event["state"] == "in_flight":
                break

            _validate_event_target_for_replace(event_path)
            authorized = RecheckRetention(current.generation, sequence, retention.deleted)
            _write_retention(paths["retention"], identity, authorized)
            event_path.unlink(missing_ok=True)
            _sync_directory(generation_path)
            retention = RecheckRetention(current.generation, sequence, sequence)
            _write_retention(paths["retention"], identity, retention)
            reclaimed += 1
        return reclaimed

    def invalidate(self) -> RecheckPosition:
        """Revoke the current generation and publish a fresh empty generation."""

        paths, identity = self._bound_paths()
        self._inspect_parent_layout(paths)
        current = self._read_position(paths, identity)
        next_generation = current.generation + 1
        generation_path = paths["journal"] / str(next_generation)
        _ensure_directory(generation_path)
        _sync_layout(paths, generation_path)
        _require_regular_file(paths["state"], "recheck state")
        atomic_replace(paths["state"], _state_record(identity, next_generation, 0))
        return RecheckPosition(next_generation, 0)

    def _bound_paths(self) -> tuple[dict[str, Path], dict[str, Any]]:
        directory = self._coverage.directory
        authority = group_authority_identity(self._root)
        identity = _identity_record(authority, self._group)
        if self._identity is None:
            self._identity = identity
            self._directory = directory
        elif identity != self._identity or directory != self._directory:
            raise RuntimeError("Group authority identity changed while rechecks were bound")
        if self._directory is None:
            raise RuntimeError("recheck directory was not initialized")
        paths = _layout_paths(self._root, self._directory)
        return paths, self._identity

    def _initialize(self, paths: dict[str, Path], identity: dict[str, Any]) -> RecheckPosition:
        _ensure_directory(paths["indexes"])
        _ensure_directory(paths["group_discovery"])
        _ensure_directory(paths["coverage"])
        _ensure_directory(paths["journal"])
        generation = uuid.uuid4().int
        if generation <= 0:
            raise RuntimeError("initial recheck generation must be positive")
        generation_path = paths["journal"] / str(generation)
        _ensure_directory(generation_path)
        _sync_layout(paths, generation_path)

        # A concurrent retry may have published state while the directories
        # were being repaired.  Never overwrite such a state record.
        if _record_present(paths["state"], "recheck state"):
            return self._read_position(paths, identity)
        atomic_replace(paths["state"], _state_record(identity, generation, 0))
        return RecheckPosition(generation, 0)

    def _inspect_parent_layout(self, paths: dict[str, Path]) -> bool:
        # A read-only operation may observe an absent namespace.  Existing
        # components are still checked so a symlink or regular-file substitute
        # is never mistaken for an absent journal.
        for path, label in (
            (paths["root"], "shared root"),
            (paths["indexes"], "indexes directory"),
            (paths["group_discovery"], "group-discovery directory"),
            (paths["coverage"], "coverage directory"),
            (paths["journal"], "rechecks directory"),
        ):
            present = _directory_present(path, label)
            if not present:
                return False
        return True

    def _read_position(self, paths: dict[str, Path], identity: dict[str, Any]) -> RecheckPosition:
        _require_regular_file(paths["state"], "recheck state")
        try:
            value = read_json_limited(paths["state"], max_bytes=_STATE_MAX_BYTES)
        except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("recheck state is malformed") from exc
        if type(value) is not dict or set(value) != _STATE_KEYS:
            raise ValueError("recheck state has unknown or missing fields")
        if type(value["version"]) is not int or value["version"] != 1:
            raise ValueError("recheck state version is invalid")
        if value["identity"] != identity:
            raise ValueError("recheck state identity does not match the Group authority")
        generation = _validate_positive_int(value["generation"], "state.generation")
        tail = _validate_nonnegative_int(value["tail"], "state.tail")
        _require_directory(paths["journal"] / str(generation), "recheck generation")
        return RecheckPosition(generation, tail)

    def _read_retention(
        self,
        paths: dict[str, Path],
        identity: dict[str, Any],
        current: RecheckPosition,
    ) -> RecheckRetention:
        path = paths["retention"]
        if not _record_present(path, "recheck retention"):
            return RecheckRetention(current.generation, 0, 0)
        _require_regular_file(path, "recheck retention")
        try:
            value = read_json_limited(path, max_bytes=_STATE_MAX_BYTES)
        except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("recheck retention is malformed") from exc
        if type(value) is not dict or set(value) != _RETENTION_KEYS:
            raise ValueError("recheck retention has unknown or missing fields")
        if type(value["version"]) is not int or value["version"] != 1:
            raise ValueError("recheck retention version is invalid")
        if value["identity"] != identity:
            raise ValueError("recheck retention identity does not match the Group authority")
        generation = _validate_positive_int(value["generation"], "retention.generation")
        floor = _validate_nonnegative_int(value["floor"], "retention.floor")
        deleted = _validate_nonnegative_int(value["deleted"], "retention.deleted")
        if deleted > floor:
            raise ValueError("recheck retention deleted boundary exceeds its floor")
        if floor - deleted > 1:
            raise ValueError("recheck retention has more than one pending deletion")
        # Missing-state recovery selects a fresh UUID, which is not ordered
        # relative to old UUIDs. Only equality with published state grants
        # authority; every other sidecar belongs to a revoked generation.
        if generation != current.generation:
            return RecheckRetention(current.generation, 0, 0)
        if floor > current.tail:
            raise ValueError("recheck retention floor exceeds the durable tail")
        return RecheckRetention(generation, floor, deleted)

    def _read_event(
        self,
        path: Path,
        identity: dict[str, Any],
        generation: int,
        sequence: int,
    ) -> dict[str, Any]:
        _require_regular_file(path, "recheck event")
        try:
            value = read_json(path)
        except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("recheck event is malformed") from exc
        if type(value) is not dict or set(value) != _EVENT_KEYS:
            raise ValueError("recheck event has unknown or missing fields")
        if type(value["version"]) is not int or value["version"] != 1:
            raise ValueError("recheck event version is invalid")
        if value["identity"] != identity:
            raise ValueError("recheck event identity does not match the Group authority")
        if _validate_positive_int(value["generation"], "event.generation") != generation:
            raise ValueError("recheck event generation is invalid")
        if _validate_positive_int(value["sequence"], "event.sequence") != sequence:
            raise ValueError("recheck event sequence is invalid")
        validate_identifier(value["task_id"], "event.task_id")
        validate_identifier(value["submission_operation_id"], "event.submission_operation_id")
        _validate_positive_int(value["membership_sequence"], "event.membership_sequence")
        _validate_owner(value["owner"])
        _validate_evidence(value["evidence"])
        if type(value["state"]) is not str or value["state"] not in _EVENT_STATES:
            raise ValueError("recheck event state is invalid")
        return value


def _layout_paths(root: Path, coverage: Path) -> dict[str, Path]:
    indexes = root / "indexes"
    group_discovery = indexes / "group-discovery"
    if coverage.parent != group_discovery:
        raise RuntimeError("GroupCoverage returned an unexpected directory")
    return {
        "root": root,
        "indexes": indexes,
        "group_discovery": group_discovery,
        "coverage": coverage,
        "journal": coverage / "rechecks",
        "state": coverage / "rechecks" / "state.json",
        "retention": coverage / "rechecks" / "retention.json",
    }


def _identity_record(authority: object, group: str) -> dict[str, Any]:
    if type(authority) is not dict or set(authority) != {"project_id", "directory_identity"}:
        raise RuntimeError("Group authority identity is malformed")
    project_id = authority["project_id"]
    directory_identity = authority["directory_identity"]
    if type(project_id) is not str or not project_id or type(directory_identity) is not dict:
        raise RuntimeError("Group authority identity is malformed")
    return {
        "project_id": project_id,
        "directory_identity": dict(directory_identity),
        "group": group,
    }


def _state_record(identity: dict[str, Any], generation: int, tail: int) -> dict[str, Any]:
    return {"version": 1, "identity": identity, "generation": generation, "tail": tail}


def _retention_record(identity: dict[str, Any], retention: RecheckRetention) -> dict[str, Any]:
    return {
        "version": 1,
        "identity": identity,
        "generation": retention.generation,
        "floor": retention.floor,
        "deleted": retention.deleted,
    }


def _write_retention(path: Path, identity: dict[str, Any], retention: RecheckRetention) -> None:
    _validate_retention_target_for_replace(path)
    atomic_replace(path, _retention_record(identity, retention))


def _validate_position(position: object) -> None:
    if not isinstance(position, RecheckPosition):
        raise TypeError("position must be a RecheckPosition")
    _validate_positive_int(position.generation, "position.generation")
    _validate_nonnegative_int(position.tail, "position.tail")


def _validate_ticket(ticket: object) -> None:
    if not isinstance(ticket, RecheckTicket):
        raise TypeError("ticket must be a RecheckTicket")
    _validate_positive_int(ticket.generation, "ticket.generation")
    _validate_positive_int(ticket.sequence, "ticket.sequence")


def _validate_positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be an exact positive integer")
    return value


def _validate_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be an exact nonnegative integer")
    return value


def _validate_reclaim_limit(value: object) -> int:
    if type(value) is not int or not 1 <= value <= 64:
        raise ValueError("max_events must be an exact integer from 1 through 64")
    return value


def _validate_owner(owner: object) -> str:
    if type(owner) is not str or owner not in _OWNERS:
        raise ValueError("recheck owner is invalid")
    return owner


def _validate_evidence(evidence: object) -> None:
    if type(evidence) is not dict:
        raise ValueError("recheck evidence must be a JSON object")
    _validate_json_value(evidence, "evidence")
    try:
        json.dumps(evidence, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("recheck evidence is not JSON encodable") from exc


def _validate_json_value(value: object, label: str) -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite JSON number")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_value(item, f"{label}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"{label} contains a non-string JSON object key")
            _validate_json_value(item, f"{label}.{key}")
        return
    raise ValueError(f"{label} contains a non-JSON value")


def _directory_present(path: Path, label: str) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(info.st_mode):
        raise ValueError(f"{label} must not be a symlink")
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{label} is not a directory")
    return True


def _require_directory(path: Path, label: str) -> None:
    if not _directory_present(path, label):
        raise ValueError(f"{label} is missing")


def _ensure_directory(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        path.mkdir()
        info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        raise ValueError(f"journal directory must not be a symlink: {path}")
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"journal path is not a directory: {path}")


def _sync_layout(paths: dict[str, Path], leaf: Path) -> None:
    """Sync a newly created leaf and each ancestor through the shared root."""

    for path, label in (
        (leaf, "recheck generation"),
        (paths["journal"], "rechecks directory"),
        (paths["coverage"], "coverage directory"),
        (paths["group_discovery"], "group-discovery directory"),
        (paths["indexes"], "indexes directory"),
        (paths["root"], "shared root"),
    ):
        _require_directory(path, label)
        _sync_directory(path)


def _sync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _record_present(path: Path, label: str) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(info.st_mode):
        raise ValueError(f"{label} must not be a symlink")
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} is not a regular file")
    return True


def _require_regular_file(path: Path, label: str) -> None:
    if not _record_present(path, label):
        raise ValueError(f"{label} is missing")


def _validate_event_target_for_replace(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(info.st_mode):
        raise ValueError("recheck event must not be a symlink")
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("recheck event path is not a regular file")


def _validate_retention_target_for_replace(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(info.st_mode):
        raise ValueError("recheck retention must not be a symlink")
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("recheck retention path is not a regular file")


__all__ = ["GroupRechecks", "RecheckPosition", "RecheckRetention", "RecheckTicket"]
