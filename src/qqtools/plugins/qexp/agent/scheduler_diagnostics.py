"""Bounded machine-local scheduler diagnostics storage."""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from ..runtime.locks import exclusive, shared
from ..runtime.paths import machine_runtime_paths
from ..runtime.records import utc_now
from ..runtime.store import JSONRecordSizeError, atomic_replace, json_encoded_size, read_json_limited
from .context import MachineRuntime
from .scheduler_diagnostic_policy import (
    FindingObservationOmission,
    IgnoredFindingObservation,
    canonicalize_diagnostic_identity,
    diagnostic_identity_digest,
    iter_publication_probes,
    resolution_queries,
    select_resolution_candidate,
)

SCHEMA_VERSION = 1

MAX_FINDING_BYTES = 4 * 1024
MAX_DECISION_BYTES = 4 * 1024
MAX_ACTIVE_RECORDS = 256
MAX_ACTIVE_BYTES = 1024 * 1024
MAX_DECISIONS = 64
MAX_DECISIONS_BYTES = 256 * 1024
MAX_HISTORY_ENTRIES = 1024
MAX_HISTORY_BYTES = 4 * 1024 * 1024
MAX_HISTORY_AGE_SECONDS = 30 * 24 * 60 * 60
MAX_SEGMENT_BYTES = 64 * 1024
MAX_LIVE_BYTES = 8 * 1024 * 1024
MAX_PHYSICAL_PEAK_BYTES = 9 * 1024 * 1024
MAX_ACTIVE_READ_RECORDS = 256
MAX_ACTIVE_READ_BYTES = 1024 * 1024
MAX_HISTORY_READ_FILES = 4
MAX_HISTORY_READ_BYTES = 512 * 1024
MAX_QUERY_RESULT_BYTES = 128 * 1024
MAX_QUERY_LIMIT = 32
DEFAULT_QUERY_LIMIT = 32
MAX_SUMMARY_LIMIT = 16
MAX_CURSOR_BYTES = 1024
MAX_PUBLICATION_OPERATIONS = 16
MAX_PUBLICATION_BYTES = 64 * 1024
MAX_PUBLICATION_ADMISSION_NS = 10_000_000

_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_EPOCH_RE = re.compile(r"^[0-9a-f]{32}$")
_CURSOR_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$")
_MAX_INTEGER = (1 << 63) - 1
_SATURATING_COUNT = (1 << 31) - 1
_EXCEPTION_TYPES = frozenset(
    {
        "BlockingIOError",
        "BrokenPipeError",
        "ChildProcessError",
        "ConnectionError",
        "FileExistsError",
        "FileNotFoundError",
        "InterruptedError",
        "IsADirectoryError",
        "JSONDecodeError",
        "KeyError",
        "NotADirectoryError",
        "OSError",
        "PermissionError",
        "ProcessLookupError",
        "RuntimeError",
        "TimeoutError",
        "TypeError",
        "UnicodeDecodeError",
        "ValueError",
    }
)
_DETAIL_FIELDS = frozenset(
    {
        "exception_type",
        "errno",
        "line",
        "column",
        "offset",
        "outcome",
        "probe_state",
        "coverage",
        "requested",
        "capacity",
        "available",
        "reserved",
        "group_limit",
        "group_usage",
        "diagnostic_code",
        "progress",
        "details_omitted",
    }
)
_REVISION_FIELD_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SEGMENT_NAME_RE = re.compile(r"^([1-9][0-9]*)-([1-9][0-9]*)\.json$")


class _StoreReadError(ValueError):
    """A malformed or unsupported scheduler diagnostic record."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _InvalidCursor(ValueError):
    """A structurally valid cursor makes an impossible claim about this store."""


class _StoreCapacityError(OSError):
    """A write would exceed one of the bounded store capacities."""


class _WriteBudget:
    """Bound one writer turn by storage operations, bytes, and admission time."""

    def __init__(self, live_bytes: int) -> None:
        self.started_ns = time.monotonic_ns()
        self.live_bytes = live_bytes
        self.operations = 0
        self.encoded_bytes = 0

    def admit(self, encoded_bytes: int, *, final: bool = False) -> None:
        if self.operations >= MAX_PUBLICATION_OPERATIONS:
            raise _StoreCapacityError("publication_operation_limit")
        if self.encoded_bytes + encoded_bytes > MAX_PUBLICATION_BYTES:
            raise _StoreCapacityError("publication_byte_limit")
        if not final and time.monotonic_ns() - self.started_ns > MAX_PUBLICATION_ADMISSION_NS:
            raise _StoreCapacityError("publication_admission_deadline")
        self.operations += 1
        self.encoded_bytes += encoded_bytes


class SchedulerDiagnosticStore:
    """MachineRuntime-local bounded store for active findings and resolved history."""

    def __init__(self, runtime_root: MachineRuntime | str | Path) -> None:
        selected_root = runtime_root.root if isinstance(runtime_root, MachineRuntime) else runtime_root
        self.runtime_root = Path(selected_root).expanduser().resolve()
        self.paths = machine_runtime_paths(self.runtime_root)

    @contextmanager
    def _writer_lock(self) -> Iterator[None]:
        with exclusive(self.paths["scheduler_diagnostics_lock"], blocking=False) as acquired:
            if not acquired:
                raise OSError("scheduler_diagnostics_lock_unavailable")
            yield

    @contextmanager
    def _reader_lock(self) -> Iterator[bool]:
        lock_path = self.paths["scheduler_diagnostics_lock"]
        try:
            if not _is_regular_file(lock_path):
                yield False
                return
            with shared(lock_path, blocking=False) as acquired:
                yield acquired
        except OSError:
            yield False

    def _new_budget(self) -> _WriteBudget:
        return _WriteBudget(_tree_bytes(self.paths["scheduler_diagnostics"]))

    def _write(self, path: Path, value: dict[str, Any], budget: _WriteBudget, *, final: bool = False) -> None:
        encoded_bytes = json_encoded_size(value)
        try:
            previous_size = path.lstat().st_size if _is_regular_file(path) else 0
        except OSError:
            previous_size = 0
        next_live = budget.live_bytes - previous_size + encoded_bytes
        if next_live > MAX_LIVE_BYTES or budget.live_bytes + encoded_bytes > MAX_PHYSICAL_PEAK_BYTES:
            raise _StoreCapacityError("store_capacity")
        budget.admit(encoded_bytes, final=final)
        atomic_replace(path, value)
        budget.live_bytes = next_live

    def _unlink(self, path: Path, budget: _WriteBudget, *, final: bool = False) -> None:
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return
        if not os.path.isfile(path) or path.is_symlink():
            raise _StoreReadError("store_corrupt")
        budget.admit(0, final=final)
        path.unlink()
        budget.live_bytes = max(0, budget.live_bytes - metadata.st_size)

    @staticmethod
    def _envelope(kind: str, epoch: str, **data: Any) -> dict[str, Any]:
        return {
            "diagnostics": {
                "schema_version": SCHEMA_VERSION,
                "kind": kind,
                "epoch": epoch,
                **data,
            }
        }

    def _read_envelope(
        self,
        path: Path,
        *,
        kind: str,
        max_bytes: int,
        epoch: str | None = None,
    ) -> dict[str, Any]:
        if not _is_regular_file(path):
            raise _StoreReadError("store_corrupt")
        try:
            value = read_json_limited(path, max_bytes=max_bytes, record_type="scheduler_diagnostics")
        except JSONRecordSizeError as exc:
            raise _StoreReadError("record_oversized") from exc
        except FileNotFoundError as exc:
            raise _StoreReadError("store_corrupt") from exc
        except (OSError, UnicodeDecodeError, ValueError, TypeError) as exc:
            raise _StoreReadError("store_corrupt") from exc
        record = value.get("diagnostics")
        if not isinstance(record, dict):
            raise _StoreReadError("store_corrupt")
        if record.get("schema_version") != SCHEMA_VERSION:
            raise _StoreReadError("store_unsupported_version")
        if record.get("kind") != kind:
            raise _StoreReadError("store_corrupt")
        actual_epoch = record.get("epoch")
        if not isinstance(actual_epoch, str) or not _EPOCH_RE.fullmatch(actual_epoch):
            raise _StoreReadError("store_corrupt")
        if epoch is not None and actual_epoch != epoch:
            raise _StoreReadError("store_epoch_mismatch")
        return record

    def _load_metadata(self) -> dict[str, Any]:
        metadata = self._read_envelope(
            self.paths["scheduler_diagnostics_metadata"],
            kind="metadata",
            max_bytes=64 * 1024,
        )
        if not _valid_metadata(metadata):
            raise _StoreReadError("store_corrupt")
        return metadata

    def _load_index(self, epoch: str) -> dict[str, Any]:
        index = self._read_envelope(
            self.paths["scheduler_diagnostics_history_index"],
            kind="history_index",
            max_bytes=256 * 1024,
            epoch=epoch,
        )
        if not _valid_index(index):
            raise _StoreReadError("store_corrupt")
        return index

    def _ensure_initialized(self, budget: _WriteBudget, observed_at: str) -> dict[str, Any]:
        for name in (
            "scheduler_diagnostics",
            "scheduler_diagnostics_active",
            "scheduler_diagnostics_decisions",
            "scheduler_diagnostics_history",
            "scheduler_diagnostics_history_segments",
            "scheduler_diagnostics_pending",
        ):
            self.paths[name].mkdir(parents=True, exist_ok=True)

        metadata_path = self.paths["scheduler_diagnostics_metadata"]
        index_path = self.paths["scheduler_diagnostics_history_index"]
        has_metadata = _is_regular_file(metadata_path)
        has_index = _is_regular_file(index_path)
        if has_metadata:
            metadata = self._load_metadata()
            if not has_index:
                raise _StoreReadError("store_corrupt")
            self._load_index(metadata["epoch"])
        elif has_index:
            index = self._read_envelope(index_path, kind="history_index", max_bytes=256 * 1024)
            if not _valid_index(index) or index.get("segments"):
                raise _StoreReadError("store_corrupt")
            metadata = _initial_metadata(index["epoch"], observed_at)
            self._write(
                metadata_path,
                self._envelope("metadata", metadata["epoch"], **_without_kind(metadata)),
                budget,
            )
        else:
            epoch = uuid.uuid4().hex
            metadata = _initial_metadata(epoch, observed_at)
            index = _initial_index(epoch, observed_at)
            self._write(
                index_path,
                self._envelope("history_index", epoch, **_without_kind(index)),
                budget,
            )
            self._write(
                metadata_path,
                self._envelope("metadata", epoch, **_without_kind(metadata)),
                budget,
            )

        summary_path = self.paths["scheduler_diagnostics_summary"]
        if not _is_regular_file(summary_path):
            summary = _initial_summary(metadata, observed_at)
            self._write(
                summary_path,
                self._envelope("summary", metadata["epoch"], **_without_kind(summary)),
                budget,
            )
        return metadata

    def _read_summary_locked(self, epoch: str) -> dict[str, Any]:
        path = self.paths["scheduler_diagnostics_summary"]
        if not _is_regular_file(path):
            return _initial_summary(_initial_metadata(epoch, utc_now()), utc_now())
        summary = self._read_envelope(path, kind="summary", max_bytes=256 * 1024, epoch=epoch)
        if not _valid_summary(summary):
            raise _StoreReadError("store_corrupt")
        return summary

    def _commit_state(
        self,
        metadata: dict[str, Any],
        summary: dict[str, Any],
        budget: _WriteBudget,
        *,
        observed_at: str,
        coverage: str | None = None,
        reason: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        next_metadata = dict(metadata)
        next_metadata["snapshot_revision"] += 1
        next_metadata["observed_at"] = observed_at
        if coverage is not None:
            next_metadata["coverage"] = coverage
            next_metadata["reason"] = reason
        next_summary = dict(summary)
        next_summary.update(
            {
                "snapshot_revision": next_metadata["snapshot_revision"],
                "observed_at": observed_at,
                "coverage": next_metadata["coverage"],
                "reason": next_metadata["reason"],
                "status": "available",
                "active_count": next_metadata["active_count"],
                "overflow_count": next_metadata["overflow_count"],
                "omitted_observations": next_metadata["omitted_observations"],
                "omitted_scopes": list(next_metadata["omitted_scopes"]),
            }
        )
        if next_metadata["coverage"] != "complete":
            next_summary["truncated"] = True
        self._write(
            self.paths["scheduler_diagnostics_metadata"],
            self._envelope("metadata", metadata["epoch"], **_without_kind(next_metadata)),
            budget,
            final=True,
        )
        self._write(
            self.paths["scheduler_diagnostics_summary"],
            self._envelope("summary", metadata["epoch"], **_without_kind(next_summary)),
            budget,
            final=True,
        )
        return next_metadata, next_summary

    def _observe_finding_locked(
        self,
        *,
        identity: Mapping[str, object],
        severity: str,
        source_revision: Mapping[str, object],
        details: Mapping[str, object] | None,
        observed_at: str,
        metadata: dict[str, Any],
        summary: dict[str, Any],
        budget: _WriteBudget,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        safe_identity, identity_incomplete = canonicalize_diagnostic_identity(identity)
        safe_revision, revision_omitted = _sanitize_revision(source_revision)
        safe_details, details_omitted = _sanitize_details(details)
        details_omitted = details_omitted or revision_omitted or identity_incomplete
        safe_severity = severity if severity in {"fault", "warning", "info"} else "fault"
        if severity not in {"fault", "warning", "info"}:
            details_omitted = True
        digest = diagnostic_identity_digest(safe_identity)
        active_path = self.paths["scheduler_diagnostics_active"] / f"{digest}.json"
        previous: dict[str, Any] | None = None
        if _is_regular_file(active_path):
            previous = self._read_envelope(
                active_path,
                kind="active",
                max_bytes=MAX_FINDING_BYTES,
                epoch=metadata["epoch"],
            )
            if previous.get("identity") != safe_identity or previous.get("identity_digest") != digest:
                raise _StoreReadError("store_corrupt")

        next_revision = metadata["snapshot_revision"] + 1
        if previous is None:
            active_count, active_bytes = _active_usage(self.paths["scheduler_diagnostics_active"])
            record = {
                "identity": safe_identity,
                "identity_digest": digest,
                "identity_incomplete": identity_incomplete,
                "episode": uuid.uuid4().hex,
                "severity": safe_severity,
                "first_observed_at": observed_at,
                "last_observed_at": observed_at,
                "observation_count": 1,
                "count_saturated": False,
                "observation_revision": next_revision,
                "source_revision": safe_revision,
                "details": safe_details,
                "details_omitted": details_omitted,
            }
            encoded_size = _encoded_size(self._envelope("active", metadata["epoch"], **record))
            if active_count >= MAX_ACTIVE_RECORDS or active_bytes + encoded_size > MAX_ACTIVE_BYTES:
                self._record_active_overflow(metadata, safe_identity, observed_at)
                return metadata, summary, None
        else:
            if (
                not isinstance(previous.get("observation_count"), int)
                or type(previous.get("observation_count")) is not int
                or type(previous.get("count_saturated")) is not bool
            ):
                raise _StoreReadError("store_corrupt")
            record = {
                key: previous[key]
                for key in (
                    "identity",
                    "identity_digest",
                    "identity_incomplete",
                    "episode",
                    "severity",
                    "first_observed_at",
                    "observation_count",
                    "count_saturated",
                )
            }
            count = record["observation_count"]
            saturated = record["count_saturated"]
            if count >= _SATURATING_COUNT:
                saturated = True
            else:
                count += 1
            record.update(
                {
                    "severity": safe_severity,
                    "last_observed_at": observed_at,
                    "observation_count": count,
                    "count_saturated": saturated,
                    "observation_revision": next_revision,
                    "source_revision": safe_revision,
                    "details": safe_details,
                    "details_omitted": details_omitted,
                }
            )

        active_envelope = self._fit_record("active", metadata["epoch"], record, MAX_FINDING_BYTES)
        self._write(active_path, active_envelope, budget)
        next_summary = _summary_upsert_active(summary, record, is_new=previous is None)
        next_metadata = dict(metadata)
        next_metadata["active_count"] = min(MAX_ACTIVE_RECORDS, next_metadata["active_count"] + int(previous is None))
        if next_metadata["coverage"] == "unknown" and not next_metadata["overflow_count"]:
            # A direct observation is complete for its explicit identity. Cycle
            # publication may subsequently narrow coverage when a bounded probe
            # cannot inspect every intended scope.
            next_metadata["coverage"] = "complete"
            next_metadata["reason"] = None
        return next_metadata, next_summary, record

    def _record_active_overflow(
        self,
        metadata: dict[str, Any],
        identity: Mapping[str, Any],
        observed_at: str,
    ) -> None:
        metadata["coverage"] = "incomplete"
        metadata["reason"] = "active_overflow"
        metadata["overflow_count"] = min(_SATURATING_COUNT, metadata["overflow_count"] + 1)
        metadata["omitted_observations"] = min(_SATURATING_COUNT, metadata["omitted_observations"] + 1)
        sample = {key: identity[key] for key in ("producer", "scope_type") if key in identity}
        if sample and sample not in metadata["omitted_scopes"] and len(metadata["omitted_scopes"]) < 16:
            metadata["omitted_scopes"].append(sample)
        metadata["observed_at"] = observed_at

    @staticmethod
    def _fit_record(kind: str, epoch: str, record: dict[str, Any], max_bytes: int) -> dict[str, Any]:
        envelope = {"diagnostics": {"schema_version": SCHEMA_VERSION, "kind": kind, "epoch": epoch, **record}}
        if _encoded_size(envelope) <= max_bytes:
            return envelope
        reduced = dict(record)
        reduced["details"] = {}
        reduced["details_omitted"] = True
        envelope["diagnostics"] = {
            "schema_version": SCHEMA_VERSION,
            "kind": kind,
            "epoch": epoch,
            **reduced,
        }
        if _encoded_size(envelope) > max_bytes:
            raise ValueError("scheduler diagnostic identity exceeds its encoded record limit")
        return envelope

    def _record_decision_locked(
        self,
        *,
        identity: Mapping[str, object],
        outcome: str,
        capacity_context: Mapping[str, object] | None,
        blocker_identities: Sequence[Mapping[str, object]] | None,
        coverage: str,
        source_revision: Mapping[str, object] | None,
        details: Mapping[str, object] | None,
        observed_at: str,
        metadata: dict[str, Any],
        summary: dict[str, Any],
        budget: _WriteBudget,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        safe_identity, identity_incomplete = canonicalize_diagnostic_identity(identity)
        digest = diagnostic_identity_digest(safe_identity)
        safe_revision, revision_omitted = _sanitize_revision(source_revision or {})
        safe_details, details_omitted = _sanitize_details(details)
        safe_capacity, capacity_omitted = _sanitize_facts(capacity_context)
        safe_blockers: list[dict[str, Any]] = []
        blockers_omitted = False
        for blocker in (blocker_identities or ())[:16]:
            if not isinstance(blocker, Mapping):
                blockers_omitted = True
                continue
            sanitized, incomplete = canonicalize_diagnostic_identity(blocker)
            safe_blockers.append(sanitized)
            blockers_omitted = blockers_omitted or incomplete
        if len(blocker_identities or ()) > 16:
            blockers_omitted = True
        valid_coverage = coverage if coverage in {"complete", "incomplete", "unknown"} else "unknown"
        valid_outcome = _safe_token(outcome) or "unknown_error"
        record = {
            "identity": safe_identity,
            "identity_digest": digest,
            "identity_incomplete": identity_incomplete,
            "observed_at": observed_at,
            "outcome": valid_outcome,
            "capacity_context": safe_capacity,
            "blocker_identities": safe_blockers,
            "coverage": valid_coverage,
            "source_revision": safe_revision,
            "details": safe_details,
            "details_omitted": (
                identity_incomplete or revision_omitted or capacity_omitted or blockers_omitted or details_omitted
            ),
            "snapshot_revision": metadata["snapshot_revision"] + 1,
        }
        path = self.paths["scheduler_diagnostics_decisions"] / f"{digest}.json"
        is_new = not _is_regular_file(path)
        envelope = self._fit_record("decision", metadata["epoch"], record, MAX_DECISION_BYTES)
        if is_new:
            count, total_bytes = _decision_usage(self.paths["scheduler_diagnostics_decisions"])
            while count >= MAX_DECISIONS or total_bytes + _encoded_size(envelope) > MAX_DECISIONS_BYTES:
                if count == 0:
                    raise _StoreCapacityError("decision_capacity")
                oldest_before = _decision_usage(self.paths["scheduler_diagnostics_decisions"])
                self._evict_oldest_decision(metadata, summary, budget)
                count, total_bytes = _decision_usage(self.paths["scheduler_diagnostics_decisions"])
                if count >= oldest_before[0] and total_bytes >= oldest_before[1]:
                    raise _StoreCapacityError("decision_capacity")
        self._write(path, envelope, budget)
        next_summary = _summary_upsert_decision(summary, record)
        next_metadata = dict(metadata)
        next_metadata["decision_count"] = min(MAX_DECISIONS, next_metadata["decision_count"] + int(is_new))
        if valid_coverage != "complete":
            next_metadata["coverage"] = "incomplete"
            next_metadata["reason"] = "probe_coverage_incomplete"
        return next_metadata, next_summary

    def _evict_oldest_decision(
        self,
        metadata: dict[str, Any],
        summary: dict[str, Any],
        budget: _WriteBudget,
    ) -> None:
        records = _enumerate_records(
            self.paths["scheduler_diagnostics_decisions"],
            max_records=MAX_DECISIONS,
        )
        if not records:
            return
        oldest = min(records, key=lambda item: (item[1].get("observed_at", ""), item[0]))
        digest = oldest[1].get("identity_digest")
        self._unlink(oldest[2], budget)
        summary["decision_samples"] = [
            item for item in summary.get("decision_samples", []) if item.get("identity_digest") != digest
        ]
        metadata["decision_count"] = max(0, metadata["decision_count"] - 1)

    def observe_finding(
        self,
        *,
        identity: Mapping[str, object],
        severity: str,
        source_revision: Mapping[str, object],
        details: Mapping[str, object] | None = None,
        observed_at: str | None = None,
    ) -> dict[str, object]:
        """Observe or coalesce one active finding under its exact canonical identity."""
        _require_mapping(identity, "identity")
        _require_mapping(source_revision, "source_revision")
        safe_time = _validated_timestamp(observed_at)
        with self._writer_lock():
            budget = self._new_budget()
            metadata = self._ensure_initialized(budget, safe_time)
            summary = self._read_summary_locked(metadata["epoch"])
            next_metadata, next_summary, record = self._observe_finding_locked(
                identity=identity,
                severity=severity,
                source_revision=source_revision,
                details=details,
                observed_at=safe_time,
                metadata=metadata,
                summary=summary,
                budget=budget,
            )
            self._commit_state(
                next_metadata,
                next_summary,
                budget,
                observed_at=safe_time,
            )
            if record is None:
                raise _StoreCapacityError("active_overflow")
            return dict(record)

    def record_decision(
        self,
        *,
        identity: Mapping[str, object],
        outcome: str,
        capacity_context: Mapping[str, object] | None = None,
        blocker_identities: Sequence[Mapping[str, object]] | None = None,
        coverage: str = "complete",
        source_revision: Mapping[str, object] | None = None,
        details: Mapping[str, object] | None = None,
        observed_at: str | None = None,
    ) -> dict[str, object]:
        """Replace the latest decision sample for one canonical decision identity."""
        _require_mapping(identity, "identity")
        safe_time = _validated_timestamp(observed_at)
        with self._writer_lock():
            budget = self._new_budget()
            metadata = self._ensure_initialized(budget, safe_time)
            summary = self._read_summary_locked(metadata["epoch"])
            safe_identity, _incomplete = canonicalize_diagnostic_identity(identity)
            digest = diagnostic_identity_digest(safe_identity)
            path = self.paths["scheduler_diagnostics_decisions"] / f"{digest}.json"
            metadata, summary = self._record_decision_locked(
                identity=identity,
                outcome=outcome,
                capacity_context=capacity_context,
                blocker_identities=blocker_identities,
                coverage=coverage,
                source_revision=source_revision,
                details=details,
                observed_at=safe_time,
                metadata=metadata,
                summary=summary,
                budget=budget,
            )
            metadata, summary = self._commit_state(
                metadata,
                summary,
                budget,
                observed_at=safe_time,
            )
            record = self._read_envelope(
                path,
                kind="decision",
                max_bytes=MAX_DECISION_BYTES,
                epoch=metadata["epoch"],
            )
            return {key: value for key, value in record.items() if key not in {"schema_version", "kind", "epoch"}}

    def resolve_finding(
        self,
        *,
        identity: Mapping[str, object],
        episode: str,
        observation_revision: int,
        source_revision: Mapping[str, object],
        resolved_at: str | None = None,
    ) -> bool:
        """Commit immutable history before removing the exact captured active record."""
        _require_mapping(identity, "identity")
        _require_mapping(source_revision, "source_revision")
        if not isinstance(episode, str) or not _EPOCH_RE.fullmatch(episode):
            raise ValueError("episode must be a 32-character lowercase UUID token")
        if type(observation_revision) is not int or observation_revision < 0:
            raise ValueError("observation_revision must be a nonnegative integer")
        safe_identity, identity_incomplete = canonicalize_diagnostic_identity(identity)
        safe_revision, revision_omitted = _sanitize_revision(source_revision)
        revisionless_unknown_registry = (
            safe_identity.get("producer") == "enablement_reconciliation"
            and safe_identity.get("reason_code") == "registry_enablement_unknown"
        )
        if identity_incomplete or revision_omitted or (not safe_revision and not revisionless_unknown_registry):
            return False
        safe_time = _validated_timestamp(resolved_at)
        digest = diagnostic_identity_digest(safe_identity)
        with self._writer_lock():
            metadata_path = self.paths["scheduler_diagnostics_metadata"]
            if not _is_regular_file(metadata_path):
                return False
            budget = self._new_budget()
            metadata = self._load_metadata()
            active_path = self.paths["scheduler_diagnostics_active"] / f"{digest}.json"
            try:
                active = self._read_envelope(
                    active_path,
                    kind="active",
                    max_bytes=MAX_FINDING_BYTES,
                    epoch=metadata["epoch"],
                )
            except _StoreReadError as exc:
                if exc.reason == "store_corrupt" and not _is_regular_file(active_path):
                    return False
                raise
            if (
                active.get("identity") != safe_identity
                or active.get("episode") != episode
                or active.get("observation_revision") != observation_revision
                or active.get("source_revision") != safe_revision
            ):
                return False

            index = self._load_index(metadata["epoch"])
            sequence = metadata["next_history_sequence"]
            entry = {
                "sequence": sequence,
                "resolved_at": safe_time,
                "identity": safe_identity,
                "identity_digest": digest,
                "episode": episode,
                "severity": active.get("severity"),
                "first_observed_at": active.get("first_observed_at"),
                "last_observed_at": active.get("last_observed_at"),
                "observation_count": active.get("observation_count"),
                "observation_revision": observation_revision,
                "source_revision": safe_revision,
                "details": active.get("details", {}),
                "details_omitted": active.get("details_omitted", False),
            }
            segment_name = f"{sequence}-{sequence}.json"
            segment_path = self.paths["scheduler_diagnostics_history_segments"] / segment_name
            segment = self._envelope("history_segment", metadata["epoch"], entries=[entry])
            segment_size = _encoded_size(segment)
            if segment_size > MAX_SEGMENT_BYTES:
                return False

            transition_id = uuid.uuid4().hex
            pending_path = self.paths["scheduler_diagnostics_pending"] / f"{transition_id}.json"
            pending = self._envelope(
                "pending_resolution",
                metadata["epoch"],
                transition_id=transition_id,
                identity=safe_identity,
                identity_digest=digest,
                episode=episode,
                observation_revision=observation_revision,
                source_revision=safe_revision,
                history_sequence=sequence,
                segment_name=segment_name,
                resolved_at=safe_time,
                entry=entry,
            )
            if _encoded_size(pending) > 16 * 1024:
                return False
            self._write(pending_path, pending, budget)

            if _is_regular_file(segment_path):
                existing_segment = self._read_envelope(
                    segment_path,
                    kind="history_segment",
                    max_bytes=MAX_SEGMENT_BYTES,
                    epoch=metadata["epoch"],
                )
                if existing_segment.get("entries") != [entry]:
                    raise _StoreReadError("store_corrupt")
            else:
                self._write(segment_path, segment, budget)

            next_index, evicted_segments = _append_history_index(
                index,
                segment_name=segment_name,
                sequence=sequence,
                byte_count=segment_size,
                observed_at=safe_time,
            )
            if _encoded_size(self._envelope("history_index", metadata["epoch"], **_without_kind(next_index))) > (
                256 * 1024
            ):
                return False
            # The index is the history commit point. Active removal always follows it.
            self._write(
                self.paths["scheduler_diagnostics_history_index"],
                self._envelope("history_index", metadata["epoch"], **_without_kind(next_index)),
                budget,
            )
            summary = self._read_summary_locked(metadata["epoch"])
            next_summary = _summary_remove_active(summary, digest, safe_identity)
            next_metadata = dict(metadata)
            next_metadata["next_history_sequence"] = sequence + 1
            next_metadata["eviction_count"] = min(
                _SATURATING_COUNT, next_metadata["eviction_count"] + len(evicted_segments)
            )
            _apply_retained_window(next_metadata, next_index)
            next_metadata["active_count"] = max(0, next_metadata["active_count"] - 1)
            next_summary["retention"] = _retention_summary(next_index)
            self._unlink(active_path, budget)
            self._commit_state(
                next_metadata,
                next_summary,
                budget,
                observed_at=safe_time,
                coverage=("incomplete" if metadata["overflow_count"] else metadata["coverage"]),
                reason=(metadata["reason"] if metadata["overflow_count"] else None),
            )
            for segment_meta in evicted_segments:
                old_name = segment_meta.get("name")
                if isinstance(old_name, str) and _SEGMENT_NAME_RE.fullmatch(old_name):
                    self._unlink(
                        self.paths["scheduler_diagnostics_history_segments"] / old_name,
                        budget,
                        final=True,
                    )
            # A failure before this unlink leaves enough pending evidence for recover().
            self._unlink(pending_path, budget, final=True)
            return True

    def active_view(
        self,
        *,
        project_id: str | None = None,
        reason: str | None = None,
        producer: str | None = None,
        scope: str | None = None,
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> dict[str, Any]:
        """Return a bounded live view of retained active findings."""
        _validate_filter_identifier(project_id, "project_id")
        _validate_filter_token(reason, "reason")
        _validate_filter_token(producer, "producer")
        _validate_filter_token(scope, "scope")
        _validate_query_limit(limit)
        base = _active_empty(self.runtime_root, project_id)
        if not _is_regular_file(self.paths["scheduler_diagnostics_metadata"]):
            base["reason"] = _missing_metadata_reason(self.paths)
            return base
        try:
            with self._reader_lock() as acquired:
                if not acquired:
                    base["reason"] = "store_unavailable"
                    return base
                metadata = self._load_metadata()
                base.update(_metadata_view(metadata))
                if _has_pending_transition(self.paths["scheduler_diagnostics_pending"]):
                    base["coverage"] = "incomplete"
                    base["reason"] = "recovery_pending"
                    base["truncated"] = True
                paths = _active_paths(self.paths["scheduler_diagnostics_active"])
                if len(paths) > MAX_ACTIVE_READ_RECORDS:
                    raise _StoreReadError("active_store_oversized")
                records: list[dict[str, Any]] = []
                bytes_read = _file_size(self.paths["scheduler_diagnostics_metadata"])
                for path in paths:
                    size = _file_size(path)
                    bytes_read += size
                    if size > MAX_FINDING_BYTES or bytes_read > MAX_ACTIVE_READ_BYTES:
                        raise _StoreReadError("active_store_oversized")
                    record = self._read_envelope(
                        path,
                        kind="active",
                        max_bytes=MAX_FINDING_BYTES,
                        epoch=metadata["epoch"],
                    )
                    identity = record.get("identity")
                    digest = record.get("identity_digest")
                    if (
                        not isinstance(identity, dict)
                        or digest != path.stem
                        or diagnostic_identity_digest(identity) != digest
                        or not _valid_active_record(record)
                    ):
                        raise _StoreReadError("store_corrupt")
                    if record["observation_revision"] > metadata["snapshot_revision"]:
                        base["coverage"] = "incomplete"
                        base["reason"] = "snapshot_incomplete"
                    if project_id is not None and identity.get("project_id") != project_id:
                        continue
                    if reason is not None and identity.get("reason_code") != reason:
                        continue
                    if producer is not None and identity.get("producer") != producer:
                        continue
                    if scope is not None and identity.get("scope_type") != scope:
                        continue
                    records.append(_active_item(record))
                records.sort(key=_active_sort_key)
                total = len(records)
                items = records[:limit]
                base.update(
                    {
                        "total": total,
                        "aggregates": _active_aggregates(records),
                        "items": items,
                        "truncated": total > len(items) or bool(base.get("truncated")),
                    }
                )
                while items and _query_size(base) > MAX_QUERY_RESULT_BYTES:
                    items.pop()
                    base["items"] = items
                    base["truncated"] = True
                if _query_size(base) > MAX_QUERY_RESULT_BYTES:
                    return _active_empty(self.runtime_root, project_id, reason="result_oversized")
                return base
        except _StoreReadError as exc:
            return _active_empty(self.runtime_root, project_id, reason=exc.reason)
        except (OSError, RuntimeError, TypeError, ValueError):
            return _active_empty(self.runtime_root, project_id, reason="store_corrupt")

    def history_view(
        self,
        *,
        project_id: str | None = None,
        cursor: str | None = None,
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> dict[str, Any]:
        """Return one immutable-history page bound to an epoch and captured sequence."""
        _validate_filter_identifier(project_id, "project_id")
        _validate_query_limit(limit)
        decoded_cursor = _decode_cursor(cursor) if cursor is not None else None
        if decoded_cursor is not None:
            cursor_project = decoded_cursor["project_id"]
            if project_id is not None and project_id != cursor_project:
                raise ValueError("project_id does not match the history cursor")
            if project_id is None:
                project_id = cursor_project
        base = _history_empty(self.runtime_root, project_id)
        if not _is_regular_file(self.paths["scheduler_diagnostics_metadata"]):
            base["reason"] = _missing_metadata_reason(self.paths)
            return base
        try:
            with self._reader_lock() as acquired:
                if not acquired:
                    base["reason"] = "store_unavailable"
                    return base
                metadata = self._load_metadata()
                index = self._load_index(metadata["epoch"])
                base.update(_metadata_view(metadata))
                if _has_pending_transition(self.paths["scheduler_diagnostics_pending"]):
                    base["coverage"] = "incomplete"
                    base["reason"] = "recovery_pending"
                    base["truncated"] = True
                captured_max = (
                    index["max_sequence"] if decoded_cursor is None else decoded_cursor["captured_max_sequence"]
                )
                retained_min = index["retained_min_sequence"]
                retained_max = index["retained_max_sequence"]
                last_examined = 0 if decoded_cursor is None else decoded_cursor["last_examined_sequence"]
                base.update(
                    {
                        "captured_at": _capture_time(index, captured_max),
                        "captured_max_sequence": captured_max,
                        "retention": _retention_view(metadata, index),
                    }
                )
                if decoded_cursor is not None and decoded_cursor["epoch"] != metadata["epoch"]:
                    return _expired_history(base)
                if decoded_cursor is not None and captured_max > index["max_sequence"]:
                    raise _InvalidCursor("history cursor captured sequence exceeds the current store")
                if decoded_cursor is not None and captured_max > 0:
                    if retained_min is None or retained_max is None:
                        return _expired_history(base)
                    if last_examined < retained_min - 1 and captured_max >= retained_min:
                        return _expired_history(base)
                    if retained_max < captured_max:
                        return _expired_history(base)
                if decoded_cursor is None:
                    last_examined = retained_min - 1 if retained_min is not None else captured_max

                segments = [
                    item
                    for item in index["segments"]
                    if item["last_sequence"] > last_examined and item["first_sequence"] <= captured_max
                ]
                items: list[dict[str, Any]] = []
                examined = last_examined
                file_reads = 2
                bytes_read = _file_size(self.paths["scheduler_diagnostics_metadata"]) + _file_size(
                    self.paths["scheduler_diagnostics_history_index"]
                )
                stop_for_output = False
                for segment_meta in segments:
                    if file_reads >= MAX_HISTORY_READ_FILES:
                        break
                    segment_path = self.paths["scheduler_diagnostics_history_segments"] / segment_meta["name"]
                    size = _file_size(segment_path)
                    if size > MAX_SEGMENT_BYTES or bytes_read + size > MAX_HISTORY_READ_BYTES:
                        break
                    file_reads += 1
                    bytes_read += size
                    segment = self._read_envelope(
                        segment_path,
                        kind="history_segment",
                        max_bytes=MAX_SEGMENT_BYTES,
                        epoch=metadata["epoch"],
                    )
                    entries = segment.get("entries")
                    if not isinstance(entries, list) or not entries:
                        raise _StoreReadError("store_corrupt")
                    for entry in entries:
                        if not isinstance(entry, dict):
                            raise _StoreReadError("store_corrupt")
                        sequence = entry.get("sequence")
                        if type(sequence) is not int or sequence <= examined:
                            raise _StoreReadError("store_corrupt")
                        if sequence > captured_max:
                            break
                        identity = entry.get("identity")
                        if not isinstance(identity, dict):
                            raise _StoreReadError("store_corrupt")
                        if project_id is not None and identity.get("project_id") != project_id:
                            examined = sequence
                            continue
                        candidate = _history_item(entry)
                        trial = dict(base)
                        trial["items"] = [*items, candidate]
                        trial["end_of_capture"] = False
                        trial["truncated"] = True
                        if _query_size(trial) > MAX_QUERY_RESULT_BYTES:
                            stop_for_output = True
                            break
                        items.append(candidate)
                        examined = sequence
                        if len(items) >= limit:
                            stop_for_output = True
                            break
                    if stop_for_output:
                        break
                    examined = max(examined, min(segment_meta["last_sequence"], captured_max))

                has_more = any(
                    segment["last_sequence"] > examined and segment["first_sequence"] <= captured_max
                    for segment in index["segments"]
                )
                end_of_capture = not has_more or examined >= captured_max
                next_cursor = None
                if not end_of_capture:
                    next_cursor = _encode_cursor(
                        {
                            "version": 1,
                            "epoch": metadata["epoch"],
                            "project_id": project_id,
                            "captured_max_sequence": captured_max,
                            "last_examined_sequence": examined,
                        }
                    )
                base.update(
                    {
                        "items": items,
                        "truncated": not end_of_capture,
                        "end_of_capture": end_of_capture,
                        "next_cursor": next_cursor,
                    }
                )
                if _query_size(base) > MAX_QUERY_RESULT_BYTES:
                    return _history_empty(self.runtime_root, project_id, reason="result_oversized")
                return base
        except _InvalidCursor:
            raise
        except _StoreReadError as exc:
            return _history_empty(self.runtime_root, project_id, reason=exc.reason)
        except (OSError, RuntimeError, TypeError, ValueError):
            return _history_empty(self.runtime_root, project_id, reason="store_corrupt")

    def summary_view(
        self,
        *,
        project_id: str | None = None,
        limit: int = MAX_SUMMARY_LIMIT,
        read_json: Callable[[Path], dict[str, Any] | None] | None = None,
    ) -> dict[str, Any]:
        """Read the materialized summary without enumerating active or history files."""
        _validate_filter_identifier(project_id, "project_id")
        if type(limit) is not int or not 1 <= limit <= MAX_SUMMARY_LIMIT:
            raise ValueError(f"limit must be an integer from 1 through {MAX_SUMMARY_LIMIT}")
        summary_path = self.paths["scheduler_diagnostics_summary"]
        if not _is_regular_file(summary_path):
            return unavailable_summary(self.runtime_root, _missing_summary_reason(self.paths))
        try:
            with self._reader_lock() as acquired:
                if not acquired:
                    return unavailable_summary(self.runtime_root, "store_unavailable")
                summary_root = _read_summary_dependency(
                    summary_path,
                    read_json=read_json,
                    record_type="scheduler_diagnostics_summary",
                )
                metadata_root = _read_summary_dependency(
                    self.paths["scheduler_diagnostics_metadata"],
                    read_json=read_json,
                    record_type="scheduler_diagnostics_metadata",
                )
                index_root = _read_summary_dependency(
                    self.paths["scheduler_diagnostics_history_index"],
                    read_json=read_json,
                    record_type="scheduler_diagnostics_history_index",
                )
                record = summary_root.get("diagnostics") if isinstance(summary_root, dict) else None
                metadata = metadata_root.get("diagnostics")
                index = index_root.get("diagnostics")
                if not isinstance(record, dict) or not isinstance(metadata, dict) or not isinstance(index, dict):
                    raise _StoreReadError("store_corrupt")
                if any(item.get("schema_version") != SCHEMA_VERSION for item in (record, metadata, index)):
                    raise _StoreReadError("store_unsupported_version")
                if (
                    record.get("kind") != "summary"
                    or not _valid_summary(record)
                    or not _valid_metadata(metadata)
                    or not _valid_index(index)
                ):
                    raise _StoreReadError("store_corrupt")
                has_pending = _has_pending_transition(self.paths["scheduler_diagnostics_pending"])
                if record.get("epoch") != metadata.get("epoch") or index.get("epoch") != metadata.get("epoch"):
                    raise _StoreReadError("store_corrupt")
                if not has_pending and (
                    record.get("snapshot_revision") != metadata.get("snapshot_revision")
                    or record.get("active_count") != metadata.get("active_count")
                    or metadata.get("retained_min_sequence") != index.get("retained_min_sequence")
                    or metadata.get("retained_max_sequence") != index.get("retained_max_sequence")
                ):
                    raise _StoreReadError("store_corrupt")

                active_samples = record["active_samples"]
                decision_samples = record["decision_samples"]
                if project_id is not None:
                    active_samples = [item for item in active_samples if item.get("project_id") == project_id]
                    decision_samples = [
                        item for item in decision_samples if item.get("identity", {}).get("project_id") == project_id
                    ]
                    active_count = record["project_active_counts"].get(project_id, 0)
                else:
                    active_count = record["active_count"]
                active_samples = sorted(active_samples, key=_summary_active_sort_key)
                decision_samples = sorted(decision_samples, key=_summary_decision_sort_key)
                return {
                    "schema_version": SCHEMA_VERSION,
                    "status": "available",
                    "coverage": "incomplete" if has_pending else record["coverage"],
                    "observed_at": record["observed_at"],
                    "snapshot_revision": record["snapshot_revision"],
                    "truncated": has_pending
                    or bool(record.get("truncated"))
                    or active_count > min(len(active_samples), limit)
                    or len(decision_samples) > limit,
                    "reason": "recovery_pending" if has_pending else record.get("reason"),
                    "machine_runtime_root": str(self.runtime_root),
                    "project_id": project_id,
                    "active_count": active_count,
                    "active_findings": active_samples[:limit],
                    "decision_samples": decision_samples[:limit],
                    "counters": record.get("counters", {}),
                    "timings": record.get("timings", {}),
                    "working_set": record.get("working_set", {}),
                    "overflow": {
                        "count": record.get("overflow_count", 0),
                        "omitted_observations": record.get("omitted_observations", 0),
                        "omitted_scopes": record.get("omitted_scopes", []),
                    },
                    "retention": record.get("retention", {}),
                }
        except _StoreReadError as exc:
            return unavailable_summary(self.runtime_root, exc.reason)
        except (OSError, JSONRecordSizeError, RuntimeError, TypeError, ValueError):
            return unavailable_summary(self.runtime_root, "summary_unavailable")

    def publish_cycle(
        self,
        counters: Mapping[str, object] | None,
        timings: Mapping[str, object] | None,
        probes: Sequence[Mapping[str, object]] | Mapping[str, object] | None,
        working_set: Mapping[str, object] | None,
        observed_at: str | None = None,
    ) -> None:
        """Best-effort coalesced publication of bounded scheduler-cycle data."""
        safe_time = _validated_timestamp(observed_at)
        try:
            with self._writer_lock():
                budget = self._new_budget()
                metadata = self._ensure_initialized(budget, safe_time)
                summary = self._read_summary_locked(metadata["epoch"])
                raw_counters = counters if isinstance(counters, Mapping) else {}
                if isinstance(raw_counters.get("counters"), Mapping):
                    cycle_counters = raw_counters["counters"]
                    if timings is None and isinstance(raw_counters.get("timings"), Mapping):
                        timings = raw_counters["timings"]
                else:
                    cycle_counters = raw_counters
                summary["counters"], counters_omitted = _sanitize_numeric_map(cycle_counters)
                summary["timings"], timings_omitted = _sanitize_timings(timings)
                summary["working_set"], working_omitted = _sanitize_numeric_map(working_set)

                next_metadata = dict(metadata)
                next_summary = dict(summary)
                findings_written = 0
                decisions_written = 0
                complete = False
                reason = "probe_data_unavailable"
                saw_probe = False
                for probe in iter_publication_probes(probes):
                    if not saw_probe:
                        complete = True
                        reason = None
                        saw_probe = True
                    if not probe.coverage_complete:
                        complete = False
                        reason = "probe_coverage_incomplete"
                    decision = probe.decision
                    if decision is not None and decisions_written < 3:
                        try:
                            next_metadata, next_summary = self._record_decision_locked(
                                identity=decision.identity,
                                outcome=decision.outcome,
                                capacity_context=decision.capacity_context,
                                blocker_identities=decision.blocker_identities,
                                coverage=decision.coverage,
                                source_revision=decision.source_revision,
                                details=decision.details,
                                observed_at=safe_time,
                                metadata=next_metadata,
                                summary=next_summary,
                                budget=budget,
                            )
                            decisions_written += 1
                        except _StoreCapacityError:
                            complete = False
                            reason = "publication_budget_exhausted"
                    for finding in probe.findings:
                        if findings_written >= 6:
                            complete = False
                            reason = "publication_budget_exhausted"
                            break
                        if isinstance(finding, FindingObservationOmission):
                            complete = False
                            reason = "identity_incomplete"
                            continue
                        if isinstance(finding, IgnoredFindingObservation):
                            continue
                        try:
                            next_metadata, next_summary, record = self._observe_finding_locked(
                                identity=finding.identity,
                                severity=finding.severity,
                                source_revision=finding.source_revision,
                                details=finding.details,
                                observed_at=safe_time,
                                metadata=next_metadata,
                                summary=next_summary,
                                budget=budget,
                            )
                            if record is None:
                                break
                            findings_written += 1
                        except _StoreCapacityError:
                            complete = False
                            reason = "publication_budget_exhausted"
                            break
                if next_metadata["overflow_count"]:
                    complete = False
                    reason = "active_overflow"
                next_summary["truncated"] = bool(
                    counters_omitted
                    or timings_omitted
                    or working_omitted
                    or not complete
                    or next_metadata["active_count"] > len(next_summary.get("active_samples", ()))
                    or next_metadata["decision_count"] > len(next_summary.get("decision_samples", ()))
                )
                next_summary["retention"] = _retention_summary(self._load_index(next_metadata["epoch"]))
                next_metadata["cycle_published_at"] = safe_time
                next_metadata["cycle_findings_written"] = findings_written
                next_metadata["cycle_decisions_written"] = decisions_written
                self._commit_state(
                    next_metadata,
                    next_summary,
                    budget,
                    observed_at=safe_time,
                    coverage="complete" if complete else "incomplete",
                    reason=None if complete else reason,
                )
        except (OSError, RuntimeError, ValueError, TypeError, UnicodeError, OverflowError):
            return

    def reconcile_cycle(
        self,
        probes: Sequence[Mapping[str, object]] | Mapping[str, object] | None,
        *,
        resolved_at: str | None = None,
    ) -> bool:
        """Resolve at most one stale finding with complete current scope evidence."""
        safe_time = _validated_timestamp(resolved_at)
        for query in resolution_queries(probes):
            active = self.active_view(producer=query.producer, limit=MAX_QUERY_LIMIT)
            candidate = select_resolution_candidate(query, active)
            if candidate is None:
                continue
            return self.resolve_finding(
                identity=candidate.identity,
                episode=candidate.episode,
                observation_revision=candidate.observation_revision,
                source_revision=candidate.source_revision,
                resolved_at=safe_time,
            )
        return False

    def recover(self) -> dict[str, int]:
        """Finish exact indexed resolutions and remove a bounded set of orphan segments."""
        if not _is_regular_file(self.paths["scheduler_diagnostics_metadata"]):
            return {"recovered": 0, "discarded": 0}
        if not _has_pending_transition(self.paths["scheduler_diagnostics_pending"]):
            return {"recovered": 0, "discarded": 0}
        recovered = 0
        discarded = 0
        with self._writer_lock():
            safe_time = utc_now()
            budget = self._new_budget()
            metadata = self._load_metadata()
            index = self._load_index(metadata["epoch"])
            summary = self._read_summary_locked(metadata["epoch"])
            indexed_names = {item["name"] for item in index["segments"]}
            pending_paths = _pending_paths(self.paths["scheduler_diagnostics_pending"])
            for pending_path in pending_paths[:1]:
                pending = self._read_envelope(
                    pending_path,
                    kind="pending_resolution",
                    max_bytes=16 * 1024,
                    epoch=metadata["epoch"],
                )
                segment_name = pending.get("segment_name")
                sequence = pending.get("history_sequence")
                digest = pending.get("identity_digest")
                episode = pending.get("episode")
                identity = pending.get("identity")
                if (
                    not isinstance(segment_name, str)
                    or not _SEGMENT_NAME_RE.fullmatch(segment_name)
                    or type(sequence) is not int
                    or not isinstance(digest, str)
                    or not _DIGEST_RE.fullmatch(digest)
                    or not isinstance(episode, str)
                    or not _EPOCH_RE.fullmatch(episode)
                    or not isinstance(identity, dict)
                    or diagnostic_identity_digest(identity) != digest
                ):
                    raise _StoreReadError("pending_record_invalid")
                segment_path = self.paths["scheduler_diagnostics_history_segments"] / segment_name
                if segment_name not in indexed_names:
                    if _is_regular_file(segment_path):
                        self._unlink(segment_path, budget)
                    self._unlink(pending_path, budget, final=True)
                    discarded += 1
                    continue

                segment = self._read_envelope(
                    segment_path,
                    kind="history_segment",
                    max_bytes=MAX_SEGMENT_BYTES,
                    epoch=metadata["epoch"],
                )
                entries = segment.get("entries")
                match = next(
                    (
                        item
                        for item in entries or ()
                        if isinstance(item, dict)
                        and item.get("sequence") == sequence
                        and item.get("identity_digest") == digest
                        and item.get("identity") == identity
                        and item.get("episode") == episode
                        and item.get("observation_revision") == pending.get("observation_revision")
                        and item.get("source_revision") == pending.get("source_revision")
                    ),
                    None,
                )
                if match is None:
                    raise _StoreReadError("pending_history_mismatch")

                active_path = self.paths["scheduler_diagnostics_active"] / f"{digest}.json"
                active_matches = False
                if _is_regular_file(active_path):
                    active = self._read_envelope(
                        active_path,
                        kind="active",
                        max_bytes=MAX_FINDING_BYTES,
                        epoch=metadata["epoch"],
                    )
                    active_matches = (
                        active.get("identity") == identity
                        and active.get("episode") == episode
                        and active.get("observation_revision") == pending.get("observation_revision")
                        and active.get("source_revision") == pending.get("source_revision")
                    )
                next_metadata = dict(metadata)
                next_summary = dict(summary)
                if active_matches:
                    self._unlink(active_path, budget)
                    next_metadata["active_count"] = max(0, next_metadata["active_count"] - 1)
                    next_summary = _summary_remove_active(next_summary, digest, identity)
                    recovered += 1
                elif _is_regular_file(active_path):
                    next_metadata["coverage"] = "incomplete"
                    next_metadata["reason"] = "recovery_active_mismatch"
                elif metadata["next_history_sequence"] <= sequence:
                    # The active unlink committed, but metadata/summary did not.
                    next_metadata["active_count"] = max(0, next_metadata["active_count"] - 1)
                    next_summary = _summary_remove_active(next_summary, digest, identity)
                    recovered += 1
                next_metadata["next_history_sequence"] = max(next_metadata["next_history_sequence"], sequence + 1)
                next_metadata["eviction_count"] = max(next_metadata["eviction_count"], index["evicted_entries"])
                _apply_retained_window(next_metadata, index)
                next_summary["retention"] = _retention_summary(index)
                self._commit_state(
                    next_metadata,
                    next_summary,
                    budget,
                    observed_at=safe_time,
                    coverage=next_metadata["coverage"],
                    reason=next_metadata["reason"],
                )
                orphan_paths = [
                    path
                    for path in _segment_paths(self.paths["scheduler_diagnostics_history_segments"])
                    if path.name not in indexed_names
                ][:5]
                for segment_path in orphan_paths[:4]:
                    self._unlink(segment_path, budget, final=True)
                    discarded += 1
                # Keep the transition marker until every segment evicted by the
                # transition is gone. A later bounded recovery pass can then
                # continue cleanup without scanning on ordinary empty cycles.
                if len(orphan_paths) <= 4:
                    self._unlink(pending_path, budget, final=True)
        return {"recovered": recovered, "discarded": discarded}


def _encoded_size(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _query_size(value: object) -> int:
    return _encoded_size(value)


def _is_regular_file(path: Path) -> bool:
    try:
        return path.is_file() and not path.is_symlink()
    except OSError:
        return False


def _file_size(path: Path) -> int:
    if not _is_regular_file(path):
        raise _StoreReadError("store_corrupt")
    try:
        return path.stat().st_size
    except OSError as exc:
        raise _StoreReadError("store_corrupt") from exc


def _missing_metadata_reason(paths: Mapping[str, Path]) -> str:
    return "store_corrupt" if paths["scheduler_diagnostics"].exists() else "store_not_initialized"


def _missing_summary_reason(paths: Mapping[str, Path]) -> str:
    return _missing_metadata_reason(paths)


def _has_pending_transition(directory: Path) -> bool:
    if not directory.exists():
        return False
    try:
        return next(directory.iterdir(), None) is not None
    except OSError as exc:
        raise _StoreReadError("store_corrupt") from exc


def _read_summary_dependency(
    path: Path,
    *,
    read_json: Callable[[Path], dict[str, Any] | None] | None,
    record_type: str,
) -> dict[str, Any]:
    if not _is_regular_file(path):
        raise _StoreReadError("store_corrupt")
    if read_json is None:
        return read_json_limited(
            path,
            max_bytes=256 * 1024,
            record_type=record_type,
        )
    value = read_json(path)
    if value is None:
        raise _StoreReadError("store_corrupt")
    return value


def _tree_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    entries = 0
    for path in root.rglob("*"):
        entries += 1
        if entries > MAX_ACTIVE_RECORDS + MAX_DECISIONS + MAX_HISTORY_ENTRIES + 32:
            raise _StoreReadError("store_oversized")
        if path.is_symlink():
            raise _StoreReadError("store_corrupt")
        if path.is_file():
            total += path.stat().st_size
            if total > MAX_PHYSICAL_PEAK_BYTES:
                raise _StoreReadError("store_oversized")
    return total


def _without_kind(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in {"schema_version", "kind", "epoch"}}


def _initial_metadata(epoch: str, observed_at: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "metadata",
        "epoch": epoch,
        "snapshot_revision": 0,
        "next_history_sequence": 1,
        "coverage": "unknown",
        "reason": "not_yet_observed",
        "observed_at": observed_at,
        "active_count": 0,
        "decision_count": 0,
        "overflow_count": 0,
        "omitted_observations": 0,
        "omitted_scopes": [],
        "eviction_count": 0,
        "retained_min_sequence": None,
        "retained_max_sequence": None,
        "cycle_published_at": None,
        "cycle_findings_written": 0,
        "cycle_decisions_written": 0,
    }


def _initial_index(epoch: str, observed_at: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "history_index",
        "epoch": epoch,
        "updated_at": observed_at,
        "segments": [],
        "entry_count": 0,
        "byte_count": 0,
        "max_sequence": 0,
        "retained_min_sequence": None,
        "retained_max_sequence": None,
        "evicted_entries": 0,
    }


def _initial_summary(metadata: Mapping[str, Any], observed_at: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "summary",
        "epoch": metadata["epoch"],
        "status": "available",
        "coverage": metadata["coverage"],
        "observed_at": observed_at,
        "snapshot_revision": metadata["snapshot_revision"],
        "truncated": False,
        "reason": metadata["reason"],
        "active_count": metadata["active_count"],
        "active_samples": [],
        "decision_samples": [],
        "project_active_counts": {},
        "counters": {},
        "timings": {},
        "working_set": {},
        "overflow_count": metadata["overflow_count"],
        "omitted_observations": metadata["omitted_observations"],
        "omitted_scopes": list(metadata["omitted_scopes"]),
        "retention": {},
    }


def _valid_optional_sequence_window(value: Mapping[str, Any]) -> bool:
    minimum = value.get("retained_min_sequence")
    maximum = value.get("retained_max_sequence")
    if minimum is None or maximum is None:
        return minimum is None and maximum is None
    return type(minimum) is int and type(maximum) is int and 1 <= minimum <= maximum


def _valid_metadata(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    integer_keys = (
        "snapshot_revision",
        "next_history_sequence",
        "active_count",
        "decision_count",
        "overflow_count",
        "omitted_observations",
        "eviction_count",
    )
    return (
        value.get("kind") == "metadata"
        and isinstance(value.get("epoch"), str)
        and bool(_EPOCH_RE.fullmatch(value["epoch"]))
        and all(type(value.get(key)) is int and value[key] >= 0 for key in integer_keys)
        and value.get("coverage") in {"complete", "incomplete", "unknown"}
        and (value.get("reason") is None or isinstance(value.get("reason"), str))
        and isinstance(value.get("omitted_scopes"), list)
        and _valid_optional_sequence_window(value)
    )


def _valid_index(value: object) -> bool:
    if not isinstance(value, dict) or value.get("kind") != "history_index":
        return False
    if not isinstance(value.get("epoch"), str) or not _EPOCH_RE.fullmatch(value["epoch"]):
        return False
    for key in ("entry_count", "byte_count", "max_sequence", "evicted_entries"):
        if type(value.get(key)) is not int or value[key] < 0:
            return False
    segments = value.get("segments")
    if not isinstance(segments, list) or len(segments) > MAX_HISTORY_ENTRIES:
        return False
    previous = 0
    counted_bytes = 0
    counted_entries = 0
    expected_keys = {
        "name",
        "first_sequence",
        "last_sequence",
        "entry_count",
        "byte_count",
        "resolved_at",
    }
    for segment in segments:
        if not isinstance(segment, dict) or set(segment) != expected_keys:
            return False
        first = segment["first_sequence"]
        last = segment["last_sequence"]
        if (
            not isinstance(segment["name"], str)
            or not _SEGMENT_NAME_RE.fullmatch(segment["name"])
            or type(first) is not int
            or type(last) is not int
            or first <= previous
            or last < first
            or type(segment["entry_count"]) is not int
            or segment["entry_count"] != last - first + 1
            or type(segment["byte_count"]) is not int
            or not 0 < segment["byte_count"] <= MAX_SEGMENT_BYTES
            or not isinstance(segment["resolved_at"], str)
        ):
            return False
        previous = last
        counted_entries += segment["entry_count"]
        counted_bytes += segment["byte_count"]
    if value["entry_count"] != counted_entries or value["byte_count"] != counted_bytes:
        return False
    if segments:
        if value.get("retained_min_sequence") != segments[0]["first_sequence"]:
            return False
        if value.get("retained_max_sequence") != segments[-1]["last_sequence"]:
            return False
        if value["max_sequence"] < segments[-1]["last_sequence"]:
            return False
    elif value.get("retained_min_sequence") is not None or value.get("retained_max_sequence") is not None:
        return False
    return True


def _valid_summary(value: object) -> bool:
    if not isinstance(value, dict) or value.get("kind") != "summary":
        return False
    return (
        isinstance(value.get("epoch"), str)
        and bool(_EPOCH_RE.fullmatch(value["epoch"]))
        and value.get("status") == "available"
        and value.get("coverage") in {"complete", "incomplete", "unknown"}
        and type(value.get("snapshot_revision")) is int
        and value["snapshot_revision"] >= 0
        and type(value.get("active_count")) is int
        and value["active_count"] >= 0
        and type(value.get("truncated")) is bool
        and isinstance(value.get("active_samples"), list)
        and isinstance(value.get("decision_samples"), list)
        and isinstance(value.get("project_active_counts"), dict)
        and isinstance(value.get("counters"), dict)
        and isinstance(value.get("timings"), dict)
        and isinstance(value.get("working_set"), dict)
    )


def _require_mapping(value: object, label: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")


def _safe_token(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    token = value.strip().lower()
    return token if _TOKEN_RE.fullmatch(token) else None


def _safe_identifier(value: object) -> str | None:
    return value if isinstance(value, str) and _IDENTIFIER_RE.fullmatch(value) else None


def _sanitize_facts(value: Mapping[str, object] | None) -> tuple[dict[str, Any], bool]:
    if value is None:
        return {}, False
    safe: dict[str, Any] = {}
    omitted = False
    for key, raw in value.items():
        if not _REVISION_FIELD_RE.fullmatch(key) or len(safe) >= 32:
            omitted = True
            continue
        if type(raw) is bool:
            safe[key] = raw
        elif type(raw) is int and -_MAX_INTEGER <= raw <= _MAX_INTEGER:
            safe[key] = raw
        elif key == "exception_type" and raw in _EXCEPTION_TYPES:
            safe[key] = raw
        elif key in {"outcome", "probe_state", "coverage", "diagnostic_code", "progress"}:
            token = _safe_token(raw)
            if token is None:
                omitted = True
            else:
                safe[key] = token
        else:
            omitted = True
    return safe, omitted


def _sanitize_details(value: Mapping[str, object] | None) -> tuple[dict[str, Any], bool]:
    if value is None:
        return {}, False
    selected = {key: raw for key, raw in value.items() if key in _DETAIL_FIELDS}
    safe, omitted = _sanitize_facts(selected)
    return safe, omitted or len(selected) != len(value)


def _sanitize_revision(value: Mapping[str, object]) -> tuple[dict[str, Any], bool]:
    return _sanitize_facts(value)


def _validated_timestamp(value: str | None) -> str:
    if value is None:
        return utc_now()
    if not isinstance(value, str) or not _TIMESTAMP_RE.fullmatch(value):
        raise ValueError("timestamp must be an RFC-3339 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp must be an RFC-3339 string") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return value


def _record_paths(directory: Path, pattern: re.Pattern[str], *, max_entries: int) -> list[Path]:
    if not directory.exists():
        return []
    paths: list[Path] = []
    for path in directory.iterdir():
        if len(paths) >= max_entries:
            raise _StoreReadError("store_oversized")
        if path.is_symlink():
            raise _StoreReadError("store_corrupt")
        if path.is_file():
            if not pattern.fullmatch(path.name):
                raise _StoreReadError("store_corrupt")
            paths.append(path)
        else:
            raise _StoreReadError("store_corrupt")
    return sorted(paths, key=lambda item: item.name)


def _active_paths(directory: Path) -> list[Path]:
    return _record_paths(
        directory,
        re.compile(r"^[0-9a-f]{64}\.json$"),
        max_entries=MAX_ACTIVE_RECORDS + 1,
    )


def _pending_paths(directory: Path) -> list[Path]:
    return _record_paths(
        directory,
        re.compile(r"^[0-9a-f]{32}\.json$"),
        max_entries=MAX_ACTIVE_RECORDS + 1,
    )


def _segment_paths(directory: Path) -> list[Path]:
    return _record_paths(directory, _SEGMENT_NAME_RE, max_entries=MAX_HISTORY_ENTRIES + 5)


def _active_usage(directory: Path) -> tuple[int, int]:
    paths = _active_paths(directory)
    return len(paths), sum(_file_size(path) for path in paths)


def _decision_usage(directory: Path) -> tuple[int, int]:
    paths = _record_paths(
        directory,
        re.compile(r"^[0-9a-f]{64}\.json$"),
        max_entries=MAX_DECISIONS + 1,
    )
    return len(paths), sum(_file_size(path) for path in paths)


def _enumerate_records(directory: Path, *, max_records: int) -> list[tuple[str, dict[str, Any], Path]]:
    records: list[tuple[str, dict[str, Any], Path]] = []
    for path in _record_paths(
        directory,
        re.compile(r"^[0-9a-f]{64}\.json$"),
        max_entries=max_records + 1,
    )[:max_records]:
        value = read_json_limited(path, max_bytes=MAX_DECISION_BYTES, record_type="scheduler_diagnostic")
        record = value.get("diagnostics")
        if not isinstance(record, dict):
            raise _StoreReadError("store_corrupt")
        records.append((path.name, record, path))
    return records


def _summary_active_sample(record: Mapping[str, Any]) -> dict[str, Any]:
    identity = record["identity"]
    return {
        "identity": identity,
        "identity_digest": record["identity_digest"],
        "project_id": identity.get("project_id"),
        "severity": record["severity"],
        "first_observed_at": record["first_observed_at"],
        "last_observed_at": record["last_observed_at"],
        "observation_count": record["observation_count"],
        "observation_revision": record["observation_revision"],
        "details": record.get("details", {}),
        "details_omitted": record.get("details_omitted", False),
    }


def _summary_upsert_active(
    summary: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    is_new: bool,
) -> dict[str, Any]:
    next_summary = dict(summary)
    digest = record["identity_digest"]
    old_samples = summary.get("active_samples", [])
    samples = [item for item in old_samples if item.get("identity_digest") != digest]
    samples.append(_summary_active_sample(record))
    samples.sort(key=_summary_active_sort_key)
    next_summary["active_samples"] = samples[:16]
    counts = dict(summary.get("project_active_counts", {}))
    project_id = record["identity"].get("project_id")
    if is_new and isinstance(project_id, str):
        counts[project_id] = counts.get(project_id, 0) + 1
    next_summary["project_active_counts"] = counts
    next_summary["active_count"] = int(summary.get("active_count", 0)) + int(is_new)
    return next_summary


def _summary_remove_active(summary: Mapping[str, Any], digest: str, identity: Mapping[str, Any]) -> dict[str, Any]:
    next_summary = dict(summary)
    old_samples = summary.get("active_samples", [])
    next_summary["active_samples"] = [item for item in old_samples if item.get("identity_digest") != digest]
    counts = dict(summary.get("project_active_counts", {}))
    project_id = identity.get("project_id")
    if isinstance(project_id, str) and project_id in counts:
        counts[project_id] = max(0, counts[project_id] - 1)
        if counts[project_id] == 0:
            del counts[project_id]
    next_summary["project_active_counts"] = counts
    next_summary["active_count"] = max(0, int(summary.get("active_count", 0)) - 1)
    return next_summary


def _summary_upsert_decision(summary: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    next_summary = dict(summary)
    digest = record["identity_digest"]
    samples = [item for item in summary.get("decision_samples", []) if item.get("identity_digest") != digest]
    samples.append(dict(record))
    samples.sort(key=_summary_decision_sort_key)
    next_summary["decision_samples"] = samples[-16:]
    return next_summary


def _summary_active_sort_key(item: Mapping[str, Any]) -> tuple[int, str, str]:
    order = {"fault": 0, "warning": 1, "info": 2}
    return (
        order.get(str(item.get("severity")), 3),
        str(item.get("first_observed_at", "")),
        str(item.get("identity_digest", "")),
    )


def _summary_decision_sort_key(item: Mapping[str, Any]) -> tuple[str, str]:
    return str(item.get("observed_at", "")), str(item.get("identity_digest", ""))


def _active_item(record: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "identity",
        "episode",
        "severity",
        "first_observed_at",
        "last_observed_at",
        "observation_count",
        "count_saturated",
        "observation_revision",
        "source_revision",
        "details",
        "details_omitted",
        "identity_incomplete",
    )
    return {key: record[key] for key in keys}


def _active_sort_key(item: Mapping[str, Any]) -> tuple[int, str, str]:
    order = {"fault": 0, "warning": 1, "info": 2}
    return (
        order.get(str(item.get("severity")), 3),
        str(item.get("first_observed_at", "")),
        diagnostic_identity_digest(item.get("identity", {})),
    )


def _active_aggregates(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "reason": dict(Counter(str(item["identity"].get("reason_code")) for item in records)),
        "producer": dict(Counter(str(item["identity"].get("producer")) for item in records)),
        "scope": dict(Counter(str(item["identity"].get("scope_type")) for item in records)),
    }


def _valid_active_record(record: Mapping[str, Any]) -> bool:
    identity = record.get("identity")
    return (
        isinstance(identity, dict)
        and isinstance(record.get("identity_digest"), str)
        and diagnostic_identity_digest(identity) == record["identity_digest"]
        and isinstance(record.get("episode"), str)
        and bool(_EPOCH_RE.fullmatch(record["episode"]))
        and record.get("severity") in {"fault", "warning", "info"}
        and type(record.get("observation_count")) is int
        and record["observation_count"] >= 1
        and type(record.get("observation_revision")) is int
        and record["observation_revision"] >= 0
        and isinstance(record.get("source_revision"), dict)
    )


def _active_empty(runtime_root: Path, project_id: str | None, reason: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "action": "diagnostics_active",
        "machine_runtime_root": str(runtime_root),
        "project_id": project_id,
        "coverage": "unknown",
        "observed_at": None,
        "snapshot_revision": None,
        "total": None,
        "aggregates": {},
        "items": [],
        "truncated": False,
        "reason": reason,
    }


def _history_empty(runtime_root: Path, project_id: str | None, reason: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "action": "diagnostics_history",
        "machine_runtime_root": str(runtime_root),
        "project_id": project_id,
        "coverage": "unknown",
        "observed_at": None,
        "snapshot_revision": None,
        "items": [],
        "truncated": False,
        "reason": reason,
        "captured_at": None,
        "captured_max_sequence": None,
        "end_of_capture": False,
        "next_cursor": None,
        "retention": {},
    }


def _metadata_view(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "coverage": metadata["coverage"],
        "observed_at": metadata["observed_at"],
        "snapshot_revision": metadata["snapshot_revision"],
        "truncated": metadata["coverage"] != "complete",
        "reason": metadata["reason"],
    }


def _validate_filter_identifier(value: str | None, label: str) -> None:
    if value is not None and _safe_identifier(value) is None:
        raise ValueError(f"{label} must be a bounded identifier")


def _validate_filter_token(value: str | None, label: str) -> None:
    if value is not None and _safe_token(value) is None:
        raise ValueError(f"{label} must be a bounded token")


def _validate_query_limit(limit: int) -> None:
    if type(limit) is not int or not 1 <= limit <= MAX_QUERY_LIMIT:
        raise ValueError(f"limit must be an integer from 1 through {MAX_QUERY_LIMIT}")


def _append_history_index(
    index: Mapping[str, Any], *, segment_name: str, sequence: int, byte_count: int, observed_at: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    segments = [dict(item) for item in index["segments"]]
    segments.append(
        {
            "name": segment_name,
            "first_sequence": sequence,
            "last_sequence": sequence,
            "entry_count": 1,
            "byte_count": byte_count,
            "resolved_at": observed_at,
        }
    )
    evicted: list[dict[str, Any]] = []
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=MAX_HISTORY_AGE_SECONDS)
    while segments:
        entry_count = sum(item["entry_count"] for item in segments)
        total_bytes = sum(item["byte_count"] for item in segments)
        try:
            first_time = datetime.fromisoformat(segments[0]["resolved_at"].replace("Z", "+00:00"))
        except ValueError:
            first_time = cutoff - timedelta(seconds=1)
        if entry_count <= MAX_HISTORY_ENTRIES and total_bytes <= MAX_HISTORY_BYTES and first_time >= cutoff:
            break
        evicted.append(segments.pop(0))
    next_index = dict(index)
    next_index.update(
        {
            "updated_at": observed_at,
            "segments": segments,
            "entry_count": sum(item["entry_count"] for item in segments),
            "byte_count": sum(item["byte_count"] for item in segments),
            "max_sequence": max(index["max_sequence"], sequence),
            "retained_min_sequence": segments[0]["first_sequence"] if segments else None,
            "retained_max_sequence": segments[-1]["last_sequence"] if segments else None,
            "evicted_entries": index["evicted_entries"] + sum(item["entry_count"] for item in evicted),
        }
    )
    return next_index, evicted


def _apply_retained_window(metadata: dict[str, Any], index: Mapping[str, Any]) -> None:
    metadata["retained_min_sequence"] = index["retained_min_sequence"]
    metadata["retained_max_sequence"] = index["retained_max_sequence"]


def _retention_summary(index: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "first_sequence": index["retained_min_sequence"],
        "last_sequence": index["retained_max_sequence"],
        "entry_count": index["entry_count"],
        "byte_count": index["byte_count"],
        "evicted_entries": index["evicted_entries"],
    }


def _retention_view(metadata: Mapping[str, Any], index: Mapping[str, Any]) -> dict[str, Any]:
    value = _retention_summary(index)
    value["eviction_count"] = metadata["eviction_count"]
    return value


def _capture_time(index: Mapping[str, Any], captured_max: int) -> str | None:
    if captured_max == 0:
        return index.get("updated_at") if isinstance(index.get("updated_at"), str) else None
    for segment in reversed(index["segments"]):
        if segment["last_sequence"] <= captured_max:
            return segment["resolved_at"]
    return None


def _history_item(entry: Mapping[str, Any]) -> dict[str, Any]:
    return dict(entry)


def _encode_cursor(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    token = base64.urlsafe_b64encode(encoded).rstrip(b"=").decode("ascii")
    if len(token.encode("ascii")) > MAX_CURSOR_BYTES:
        raise ValueError("history cursor exceeds its size limit")
    return token


def _decode_cursor(token: str) -> dict[str, Any]:
    if not isinstance(token, str) or not token or len(token.encode("utf-8")) > MAX_CURSOR_BYTES:
        raise ValueError("invalid history cursor")
    if not _CURSOR_RE.fullmatch(token):
        raise ValueError("invalid history cursor")
    try:
        padded = token + "=" * (-len(token) % 4)
        value = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid history cursor") from exc
    expected = {
        "version",
        "epoch",
        "project_id",
        "captured_max_sequence",
        "last_examined_sequence",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("invalid history cursor")
    if value["version"] != 1 or not isinstance(value["epoch"], str) or not _EPOCH_RE.fullmatch(value["epoch"]):
        raise ValueError("invalid history cursor")
    _validate_filter_identifier(value["project_id"], "cursor project_id")
    for key in ("captured_max_sequence", "last_examined_sequence"):
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError("invalid history cursor")
    if value["last_examined_sequence"] > value["captured_max_sequence"]:
        raise ValueError("invalid history cursor")
    return value


def _expired_history(base: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    result.update(
        {
            "coverage": "incomplete",
            "reason": "cursor_expired",
            "items": [],
            "truncated": True,
            "end_of_capture": False,
            "next_cursor": None,
        }
    )
    return result


def _sanitize_numeric_map(value: Mapping[str, object] | None) -> tuple[dict[str, int], bool]:
    if value is None:
        return {}, False
    result: dict[str, int] = {}
    omitted = False
    for key, raw in value.items():
        if len(result) >= 64 or not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9_.]{0,95}", key):
            omitted = True
        elif type(raw) is int and -_MAX_INTEGER <= raw <= _MAX_INTEGER:
            result[key] = raw
        else:
            omitted = True
    return result, omitted


def _sanitize_timings(value: Mapping[str, object] | None) -> tuple[dict[str, Any], bool]:
    if value is None:
        return {}, False
    result: dict[str, Any] = {}
    omitted = False
    for key, raw in value.items():
        if len(result) >= 32 or not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9_.]{0,95}", key):
            omitted = True
        elif isinstance(raw, Mapping):
            safe, dropped = _sanitize_facts(raw)
            result[key] = safe
            omitted = omitted or dropped
        elif type(raw) in {int, float} and not isinstance(raw, bool):
            result[key] = raw
        else:
            omitted = True
    return result, omitted


def unavailable_summary(runtime_root: str | Path, reason: str) -> dict[str, object]:
    """Return an unavailable, non-mutating diagnostics summary."""
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "unavailable",
        "coverage": "unknown",
        "observed_at": None,
        "snapshot_revision": None,
        "truncated": False,
        "reason": reason,
        "machine_runtime_root": str(Path(runtime_root).expanduser().resolve()),
        "project_id": None,
        "active_count": None,
        "active_findings": [],
        "decision_samples": [],
        "counters": {},
        "timings": {},
        "working_set": {},
        "overflow": {"count": None, "omitted_observations": None, "omitted_scopes": []},
        "retention": {},
    }


__all__ = [
    "DEFAULT_QUERY_LIMIT",
    "MAX_ACTIVE_BYTES",
    "MAX_ACTIVE_READ_BYTES",
    "MAX_ACTIVE_READ_RECORDS",
    "MAX_ACTIVE_RECORDS",
    "MAX_CURSOR_BYTES",
    "MAX_DECISION_BYTES",
    "MAX_DECISIONS",
    "MAX_DECISIONS_BYTES",
    "MAX_FINDING_BYTES",
    "MAX_HISTORY_AGE_SECONDS",
    "MAX_HISTORY_BYTES",
    "MAX_HISTORY_ENTRIES",
    "MAX_HISTORY_READ_BYTES",
    "MAX_HISTORY_READ_FILES",
    "MAX_LIVE_BYTES",
    "MAX_PHYSICAL_PEAK_BYTES",
    "MAX_PUBLICATION_ADMISSION_NS",
    "MAX_PUBLICATION_BYTES",
    "MAX_PUBLICATION_OPERATIONS",
    "MAX_QUERY_LIMIT",
    "MAX_QUERY_RESULT_BYTES",
    "MAX_SEGMENT_BYTES",
    "MAX_SUMMARY_LIMIT",
    "SCHEMA_VERSION",
    "SchedulerDiagnosticStore",
    "unavailable_summary",
]
