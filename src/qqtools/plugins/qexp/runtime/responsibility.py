"""Required launch publication and qualified background recovery discovery.

Machine-owned discovery becomes primary only after exact-generation retained
capture qualification. Membership never grants execution or mutation authority.
"""

from __future__ import annotations

import threading
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

from .locks import exclusive
from .records import validate_identifier
from .responsibility_cleanup import CleanupRequest, complete_cleanup
from .responsibility_store import BUCKETS, Ledger, ServiceTraversal
from .work_budget import diagnostic_increment

if TYPE_CHECKING:
    from ..config_types import RootConfig
    from .records import AttemptRecord
    from .responsibility_backfill import ResponsibilityBackfill

_READ_WORKERS = threading.BoundedSemaphore(4)
_ERRORS = (OSError, RuntimeError, ValueError, KeyError, TypeError)


def responsibility_root(runtime_root: Path) -> Path:
    return runtime_root / "recovery-responsibilities"


def require_launch_responsibility(cfg: RootConfig, task_id: str, attempt_id: str, attempt_number: int) -> None:
    """Persist launch discovery before process creation, outside authority locks."""
    try:
        validate_identifier(task_id, "task_id")
        validate_identifier(attempt_id, "attempt_id")
        if type(attempt_number) is not int or attempt_number < 1:
            raise ValueError("launch responsibility requires a positive Attempt number")
        root = responsibility_root(cfg.runtime_root)
        with exclusive(cfg.runtime_root / "locks" / "responsibility-initialize.lock"):
            ledger = Ledger.open_or_create(root)
        ledger.publish(attempt_id, {"task_id": task_id, "attempt_number": attempt_number})
    except _ERRORS as exc:
        diagnostic_increment("responsibility.publication_unavailable")
        raise RuntimeError(f"cannot persist launch responsibility for {attempt_id!r}") from exc
    diagnostic_increment("responsibility.published")


def publish_launch(cfg: RootConfig, task_id: str, attempt: AttemptRecord) -> bool:
    """Try advisory publication; launch callers must use the required variant."""
    try:
        require_launch_responsibility(cfg, task_id, attempt.attempt_id, attempt.attempt_number)
    except _ERRORS:
        return False
    return True


class ResponsibilityReader:
    """Prefetch at most 64 members without waiting for index I/O in supervision.

    Four daemon workers at most may run across the machine process. There is no
    executor queue; a busy pool defers this optional path to a later turn. Worker
    closures own only storage, never supervisor state or authority locks.
    """

    def __init__(self, runtime_root: Path, *, owner: dict | None = None) -> None:
        self.runtime_root = runtime_root
        self.root = responsibility_root(runtime_root)
        self._traversal: ServiceTraversal | None = None
        self._maintenance_traversal: ServiceTraversal | None = None
        self._stage_build_bucket = 0
        self._stage_build_complete: set[int] = set()
        self._backfill: ResponsibilityBackfill | None = None
        self._is_backfill_complete = False
        self._done: threading.Event | None = None
        self._result: list[dict] = []
        self._error: Exception | None = None
        self._entries: deque[dict] = deque()
        self._cleanups: deque[CleanupRequest] = deque(maxlen=64)
        self._cleanup_inflight: str | None = None
        self._is_closed = False
        self.failures = 0
        self.owner = dict(owner) if owner is not None else None
        self.discovery_mode = "legacy" if owner is None else "checking"
        self.discovery_error: str | None = None
        self._generation: str | None = None
        self._worker_generation: str | None = None
        self._last_qualified_generation: str | None = None
        self._result_mode = self.discovery_mode
        self._result_generation: str | None = None
        self._result_initial_complete = False
        self._result_active_error = False
        self._initial_acknowledged = False
        self._unacknowledged = 0
        self._initial_epoch = 0
        self._worker_epoch = -1
        self._result_epoch = -1
        self._collected_initial_complete = False
        self._collected_epoch = -1

    @property
    def is_initial_sweep_complete(self) -> bool:
        return self.discovery_mode == "primary" and self._initial_acknowledged

    def acknowledge(self, *, is_success: bool = True, stage: str = "active") -> None:
        """Acknowledge a successfully serviced candidate, never just a prefetch."""
        if self._unacknowledged:
            self._unacknowledged -= 1
        if not is_success:
            if self.discovery_mode == "primary" and stage == "active":
                self.invalidate_initial_sweep()
            return
        self._mark_initial_complete()

    def _mark_initial_complete(self) -> None:
        if (
            self.discovery_mode == "primary"
            and self._collected_initial_complete
            and self._collected_epoch == self._initial_epoch
            and not self._entries
            and not self._unacknowledged
        ):
            self._initial_acknowledged = True

    def invalidate_initial_sweep(self) -> None:
        self._initial_acknowledged = False
        self._initial_epoch += 1

    def close(self) -> None:
        # Do not wait on an unbounded filesystem operation. Late results have no
        # callback into the supervisor; any interrupted redo remains recoverable.
        self._is_closed = True
        self._entries.clear()
        self._cleanups.clear()
        if self._backfill is not None and (self._done is None or self._done.is_set()):
            self._backfill.close()

    def cleanup(self, request: CleanupRequest) -> bool:
        """Queue proof publication; until it succeeds, original evidence is retained."""
        if self._is_closed:
            return False
        if self.is_cleanup_pending(request.identity):
            return True
        if len(self._cleanups) >= 64:
            return False
        self._cleanups.append(request)
        return True

    def is_cleanup_pending(self, attempt_id: str) -> bool:
        return (self._cleanup_inflight is not None and self._cleanup_inflight == attempt_id) or any(
            item.identity == attempt_id for item in self._cleanups
        )

    def take(self) -> dict | None:
        self.poll()
        if not self._entries:
            return None
        if self.owner is not None:
            self._unacknowledged += 1
        return self._entries.popleft()

    def poll(self) -> None:
        """Collect qualification and prefetch results without consuming a candidate."""
        if self._is_closed:
            return
        if self._done is not None and self._done.is_set():
            self._collected_initial_complete = self._result_initial_complete
            self._collected_epoch = self._result_epoch
            if self._error is not None:
                self.failures += 1
            if self.owner is not None:
                if self._result_generation != self._generation or self._result_mode != self.discovery_mode:
                    self._initial_acknowledged = False
                self._generation = self._result_generation
                self.discovery_mode = self._result_mode
                self.discovery_error = str(self._error) if self._error is not None else None
                if self.discovery_mode == "primary" and self._result_active_error:
                    self.invalidate_initial_sweep()
            self._entries.extend(self._result)
            self._done = None
            self._cleanup_inflight = None
            self._result = []
            self._error = None
            if not self._entries:
                self._mark_initial_complete()
        if self._done is None and not self._entries:
            self._start()

    def _start(self) -> None:
        if not _READ_WORKERS.acquire(blocking=False):
            return
        done = threading.Event()
        self._done = done
        cleanup = self._cleanups.popleft() if self._cleanups else None
        self._cleanup_inflight = None if cleanup is None else cleanup.identity
        epoch = self._initial_epoch

        def read() -> None:
            try:
                self._result_initial_complete = False
                self._result_active_error = False
                self._result_epoch = epoch
                if self.owner is not None:
                    from .responsibility_qualification import qualify_discovery

                    try:
                        generation = qualify_discovery(
                            self.runtime_root, self.owner, previous_generation=self._last_qualified_generation
                        )
                        self._result_mode = "primary" if generation is not None else "legacy"
                        self._result_generation = generation
                        if generation is not None:
                            self._last_qualified_generation = generation
                            self._is_backfill_complete = True
                            if self._backfill is not None:
                                self._backfill.close()
                                self._backfill = None
                    except _ERRORS as exc:
                        self._result_mode = "unavailable"
                        self._result_generation = None
                        self._error = exc
                        self._result_active_error = True
                    if self._result_generation != self._worker_generation or epoch != self._worker_epoch:
                        self._traversal = None
                        self._maintenance_traversal = None
                        self._stage_build_complete.clear()
                        self._worker_generation = self._result_generation
                        self._worker_epoch = epoch
                if not self._is_backfill_complete:
                    from .authority_scan import is_path_present
                    from .responsibility_backfill import ResponsibilityBackfill
                    from .responsibility_capture import CAPTURE_FILE
                    from .responsibility_completion import COMPLETION_FILE, read_capture_completion

                    try:
                        if is_path_present(self.runtime_root / CAPTURE_FILE) or is_path_present(
                            self.runtime_root / COMPLETION_FILE
                        ):
                            # The retained sweep now owns capture. Competing
                            # advisory sweeps waste history I/O and can repeatedly
                            # steal its checkpoint lock between bounded slices.
                            if self._backfill is not None:
                                self._backfill.close()
                                self._backfill = None
                            self._is_backfill_complete = read_capture_completion(self.runtime_root) is not None
                        else:
                            if self._backfill is None:
                                self._backfill = ResponsibilityBackfill(self.runtime_root)
                            progress = self._backfill.take()
                            if progress is not None:
                                self._is_backfill_complete = progress.is_sweep_complete
                    except _ERRORS as exc:
                        # Incomplete capture never suppresses existing membership
                        # discovery or upgrades an advisory ledger to complete.
                        self._error = exc
                if cleanup is not None and self._traversal is None:
                    with exclusive(self.runtime_root / "locks" / "responsibility-initialize.lock"):
                        ledger = Ledger.open_or_create(self.root)
                    self._traversal = ServiceTraversal(ledger, stage="active")
                if self._traversal is None and not self.root.exists():
                    if self._result_mode == "primary":
                        raise RuntimeError("qualified responsibility Ledger disappeared")
                    return
                if self._traversal is None:
                    self._traversal = ServiceTraversal(Ledger(self.root), stage="active")
                ledger = self._traversal.ledger
                if self._maintenance_traversal is None:
                    self._maintenance_traversal = ServiceTraversal(ledger, stage="maintenance")
                if cleanup is not None:
                    try:
                        complete_cleanup(ledger, self.runtime_root, cleanup)
                    except _ERRORS as exc:
                        # A failed maintenance item must not consume every read
                        # turn or hide unrelated recovery candidates.
                        self._error = exc
                bucket = self._stage_build_bucket
                self._stage_build_bucket = (bucket + 1) % BUCKETS
                if bucket not in self._stage_build_complete:
                    try:
                        if ledger.build_stage_index(bucket, limit=8):
                            self._stage_build_complete.add(bucket)
                    except _ERRORS as exc:
                        self._error = exc
                self._result = []
                # Cover four active buckets per read. Otherwise a late launch
                # just behind the cursor waits sixteen complete machine cycles.
                # Preserve the 64-member bound and steady maintenance reservation.
                traversals = (
                    ((self._traversal, 16),) * 4
                    if self._result_mode == "primary" and not self._initial_acknowledged
                    else ((self._traversal, 12),) * 4 + ((self._maintenance_traversal, 16),)
                )
                for traversal, limit in traversals:
                    try:
                        entries = traversal.take(limit)
                    except _ERRORS as exc:
                        # Independent stage cursors keep a failed maintenance
                        # page from consuming active discovery's service turn.
                        self._error = exc
                        if traversal is self._traversal:
                            self._result_active_error = True
                        continue
                    for entry in entries:
                        if "legacy_source" in entry and entry.get("cleanup_receipt") is None:
                            from .responsibility_import import refresh_legacy_inbox

                            try:
                                refresh_legacy_inbox(ledger, self.runtime_root, entry["identity"])
                            except _ERRORS as exc:
                                self._error = exc
                                if traversal is self._traversal:
                                    self._result_active_error = True
                        self._result.append(entry)
                self._result_initial_complete = (
                    self._result_mode == "primary"
                    and not self._result_active_error
                    and self._traversal.has_completed_initial_sweeps
                )
            except Exception as exc:
                # Worker exceptions cannot propagate to the caller thread; retain
                # them as failed discovery instead of a successful empty result.
                self._error = exc
                self._result_active_error = True
            finally:
                if self._is_closed and self._backfill is not None:
                    self._backfill.close()
                _READ_WORKERS.release()
                done.set()

        thread = threading.Thread(target=read, name="qexp-responsibility-read", daemon=True)
        try:
            thread.start()
        except RuntimeError:
            self._done = None
            self._cleanup_inflight = None
            _READ_WORKERS.release()
            self.failures += 1
