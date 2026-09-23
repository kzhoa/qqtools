"""Durable pending writer observations and exclusion of destructive cleanup."""

from __future__ import annotations

import copy
import errno
import fcntl
import os
import uuid
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

from .authority_scan import is_path_present
from .responsibility_store import Conflict, DurableIO, Ledger, Unavailable, validate_captured_writer
from .store import atomic_replace, read_json_limited, require_json_size

CAPTURE_FILE = "responsibility-writer-capture.json"
CAPTURE_FORMAT = "qexp-pending-writer-capture-v1"
SOURCE_CAPTURE_FORMAT = "qexp-pending-writer-source-v1"
CAPTURE_BYTES = 1024 * 1024
CAPTURE_BATCH_SIZE = 64
GENERATION_FILE = "responsibility-capture-generation.json"
ADMISSION_FORMAT = "qexp-local-capture-admission-v1"
ADMISSION_OWNER_FIELDS = (
    "project_id",
    "shared_root",
    "machine_name",
    "owner_root",
    "owner_instance",
    "registration_generation",
)


def capture_admission(owner: dict) -> dict:
    if any(not isinstance(owner.get(key), str) or not owner[key] for key in ADMISSION_OWNER_FIELDS):
        raise Unavailable("capture admission requires an exact binding owner")
    return {"format": ADMISSION_FORMAT, **{key: owner[key] for key in ADMISSION_OWNER_FIELDS}}


def is_same_capture_owner(previous: dict, current: dict) -> bool:
    """Compare persistent ownership without treating a generation gap as continuity."""
    return all(
        previous.get(key) == current.get(key) for key in ADMISSION_OWNER_FIELDS if key != "registration_generation"
    )


class CaptureBusy(Conflict):
    """Capture retention defers destructive maintenance, not compatible dispatch."""


@contextmanager
def _capture_parent_guard(parent: Path, *, is_exclusive: bool) -> Iterator[bool]:
    # The parent inode survives removal of a runtime partition. Locking the
    # directory read-only also avoids requiring a writable sibling lock file.
    fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        mode = fcntl.LOCK_EX if is_exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(fd, mode | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                raise
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def has_pending_writer_capture(runtime_root: Path) -> bool:
    # Invalid/dangling checkpoints or completion proofs forbid destruction. A
    # source hold stays pending until its owning ledger has retired every member.
    from .responsibility_completion import COMPLETION_FILE, read_capture_completion

    if is_path_present(runtime_root / GENERATION_FILE):
        return True
    if not is_path_present(runtime_root / CAPTURE_FILE) and not is_path_present(runtime_root / COMPLETION_FILE):
        return False

    try:
        return read_capture_completion(runtime_root) is None
    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
        return True


@contextmanager
def cleanup_runtime_guard(runtime_root: Path) -> Iterator[None]:
    """Keep the partition alive during cleanup discovery, including cursor writes.

    Pending capture permits metadata work here; destructive steps still require
    capture_cleanup_guard. Busy capture or root removal admits no metadata writes.
    """
    with _capture_parent_guard(runtime_root.resolve().parent, is_exclusive=False) as acquired:
        if not acquired:
            raise CaptureBusy("writer capture or runtime removal is busy")
        yield


@contextmanager
def capture_cleanup_guard(
    *runtime_roots: Path, should_remove_roots: bool = False, owner_root: Path | None = None
) -> Iterator[bool]:
    """Guard deletion; root removal is exclusive and pending capture retains evidence."""
    from .responsibility_completion import is_source_capture_complete

    owner = (owner_root or runtime_roots[0]).resolve() if runtime_roots else None
    roots = sorted({root.resolve() for root in runtime_roots} | ({owner} if owner is not None else set()))
    with ExitStack() as stack:
        for parent in sorted({root.parent for root in roots}):
            if not stack.enter_context(_capture_parent_guard(parent, is_exclusive=should_remove_roots)):
                yield False
                return
        for root in roots:
            if has_pending_writer_capture(root):
                if not should_remove_roots and root != owner and owner in roots:
                    try:
                        if is_source_capture_complete(root, owner):
                            continue
                    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                        pass
                yield False
                return
        yield True


class WriterCaptureCheckpoint:
    """Persist pending capture before observing, then replay durable writer batches.

    Callers establish initial retention under runtime lifecycle ownership.
    Subsequent slices may rely on the durable hold and exclusive parent guards;
    they neither change ownership nor publish the final completeness certificate.
    The yielded object accepts bounded record() batches; it never launches work.
    Leaving observe() does not complete capture or permit cleanup. A separately
    qualified admission fence and final evidence sweep are required for that
    transition, which is intentionally not exposed by this primitive.
    """

    def __init__(self, ledger: Ledger, runtime_root: Path, *, legacy_source: Path | None = None) -> None:
        from .responsibility import responsibility_root

        self.runtime_root = runtime_root.resolve()
        if ledger.root.resolve() != responsibility_root(self.runtime_root):
            raise ValueError("writer capture ledger does not belong to this runtime")
        self.ledger = ledger
        if legacy_source is not None and not legacy_source.is_absolute():
            raise ValueError("writer capture source must be absolute")
        self.legacy_source = legacy_source.resolve() if legacy_source is not None else None
        if self.legacy_source == self.runtime_root:
            raise ValueError("writer capture source must differ from its target")
        self.path = self.runtime_root / CAPTURE_FILE
        self._state: dict | None = None

    def _observations(self, records: list[dict]) -> list[dict]:
        from .records import validate_identifier
        from .responsibility_import import recovery_locator

        if not isinstance(records, list) or len(records) > CAPTURE_BATCH_SIZE:
            raise ValueError("writer capture requires a batch of at most 64 observations")
        result = []
        for record in records:
            if not isinstance(record, dict) or set(record) not in (
                {"identity", "payload", "writer"},
                {"identity", "payload", "writer", "legacy_source"},
            ):
                raise ValueError("writer observation requires identity, payload and writer")
            identity = validate_identifier(record["identity"], "attempt_id")
            if not isinstance(record["payload"], dict):
                raise ValueError("writer observation requires a recovery locator")
            payload = recovery_locator(identity, record["payload"])
            validate_captured_writer(record["writer"])
            item = {"identity": identity, "payload": payload, "writer": dict(record["writer"])}
            if "legacy_source" in record:
                if self.legacy_source is None or record["legacy_source"] != str(self.legacy_source):
                    raise ValueError("writer observation belongs to an unguarded legacy source")
                item["legacy_source"] = record["legacy_source"]
            result.append(item)
        return result

    def _save(self, state: dict) -> None:
        require_json_size(state, max_bytes=CAPTURE_BYTES, record_type="pending_writer_capture")
        atomic_replace(self.path, state)

    def _load_or_begin(self) -> dict:
        # Reject a replaced/deleted ledger, including stale objects surviving a
        # binding lifecycle change. Never initialize a replacement here.
        if Ledger(self.ledger.root).instance != self.ledger.instance:
            raise Conflict("writer capture responsibility ledger was replaced")
        try:
            state = read_json_limited(self.path, max_bytes=CAPTURE_BYTES)
        except FileNotFoundError:
            if is_path_present(self.path):
                raise Unavailable("writer capture checkpoint is not readable") from None
            state = {
                "format": CAPTURE_FORMAT,
                "capture_id": uuid.uuid4().hex,
                "instance": self.ledger.instance,
                "runtime_root": str(self.runtime_root),
                "legacy_source": str(self.legacy_source) if self.legacy_source is not None else None,
                "phase": "pending",
                "pending": [],
            }
            self._save(state)
            return state
        if (
            state.get("format") != CAPTURE_FORMAT
            or state.get("instance") != self.ledger.instance
            or state.get("runtime_root") != str(self.runtime_root)
            or state.get("legacy_source") != (str(self.legacy_source) if self.legacy_source is not None else None)
            or state.get("phase") != "pending"
            or not isinstance(state.get("capture_id"), str)
            or len(state["capture_id"]) != 32
            or any(char not in "0123456789abcdef" for char in state["capture_id"])
        ):
            raise Unavailable("invalid pending writer capture checkpoint")
        if state.get("progress") is not None and not isinstance(state["progress"], dict):
            raise Unavailable("invalid writer capture progress")
        try:
            if self._observations(state.get("pending")) != state["pending"]:
                raise ValueError("noncanonical pending observations")
        except (ValueError, TypeError, KeyError) as exc:
            raise Unavailable("invalid pending writer observations") from exc
        # A prior rename may be visible after a failed directory fsync. Do not
        # admit another observation until that durability barrier has succeeded.
        DurableIO().sync_directory(self.runtime_root, "writer_capture_checkpoint")
        return state

    def _retain_source(self) -> None:
        if self.legacy_source is None:
            return
        if not self.legacy_source.is_dir():
            raise Unavailable("writer capture legacy source directory is unavailable")
        expected = {
            "format": SOURCE_CAPTURE_FORMAT,
            "runtime_root": str(self.legacy_source),
            "target_root": str(self.runtime_root),
            "instance": self.ledger.instance,
            "capture_id": self._state["capture_id"],
            "phase": "pending",
        }
        path = self.legacy_source / CAPTURE_FILE
        try:
            state = read_json_limited(path, max_bytes=CAPTURE_BYTES)
        except FileNotFoundError:
            if is_path_present(path):
                raise Unavailable("writer capture source hold is not readable") from None
            if self._state.get("progress") is not None or self._state["pending"]:
                raise Unavailable("writer capture source hold disappeared after observation") from None
            atomic_replace(path, expected)
        else:
            if state != expected:
                raise Conflict("legacy source already belongs to another writer capture")
            DurableIO().sync_directory(self.legacy_source, "writer_capture_source")

    @property
    def capture_id(self) -> str:
        """Return the enrollment identity only while retention is guarded."""
        if self._state is None:
            raise RuntimeError("writer capture requires an active observation scope")
        return self._state["capture_id"]

    @property
    def progress(self) -> dict | None:
        """Return replayed producer progress only inside an observation scope."""
        if self._state is None:
            raise RuntimeError("writer capture requires an active observation scope")
        return copy.deepcopy(self._state.get("progress"))

    @property
    def admission(self) -> dict | None:
        if self._state is None:
            raise RuntimeError("writer capture requires an active observation scope")
        return copy.deepcopy(self._state.get("admission"))

    def admit(self, owner: dict, *, progress: dict) -> None:
        """Atomically bind first admission and reset any pre-fence sweep progress."""
        if self._state is None:
            raise RuntimeError("writer capture requires an active observation scope")
        admission = capture_admission(owner)
        if self.admission is not None:
            if self.admission != capture_admission(self.admission) or not is_same_capture_owner(
                self.admission, admission
            ):
                raise Unavailable("capture admission belongs to another binding")
            if self.admission == admission:
                return
        try:
            self._replay()
            state = {**self._state, "admission": admission, "progress": copy.deepcopy(progress)}
            self._save(state)
            self._state = state
        except BaseException:
            self._state = None
            raise

    def _replay(self) -> None:
        from .responsibility_backfill import capture_local_writer

        state = self._state
        if state is None:
            raise RuntimeError("writer capture requires an active observation scope")
        for record in state["pending"]:
            source = record.get("legacy_source")
            capture_local_writer(
                self.ledger,
                self.runtime_root,
                record["identity"],
                record["payload"],
                record["writer"],
                source_root=Path(source) if source is not None else None,
            )
        if state["pending"]:
            cleared = {**state, "pending": []}
            self._save(cleared)
            self._state = cleared

    @contextmanager
    def observe(self) -> Iterator[WriterCaptureCheckpoint]:
        """Replay prior intent and establish durable retention before caller reads."""
        if self._state is not None:
            raise Conflict("writer observation scope is already active")
        roots = [self.runtime_root] + ([self.legacy_source] if self.legacy_source is not None else [])
        with ExitStack() as stack:
            for parent in sorted({root.parent for root in roots}):
                if not stack.enter_context(_capture_parent_guard(parent, is_exclusive=True)):
                    raise CaptureBusy("writer capture or cleanup is busy")
            try:
                from .responsibility_completion import COMPLETION_FILE

                if is_path_present(self.runtime_root / GENERATION_FILE):
                    raise Conflict("writer capture generation transition is pending")
                if is_path_present(self.runtime_root / COMPLETION_FILE):
                    raise Conflict("completed writer capture cannot be reopened")
                self._state = self._load_or_begin()
                self._retain_source()
                self._replay()
                yield self
            finally:
                self._state = None

    def record(self, observations: list[dict], *, progress: dict | None = None) -> None:
        """Journal exact observations before publishing their Ledger memberships."""
        if self._state is None:
            raise RuntimeError("writer capture requires an active observation scope")
        try:
            self._replay()
            pending = self._observations(observations)
            if progress is not None and not isinstance(progress, dict):
                raise ValueError("writer capture progress must be an object")
            if pending or progress is not None:
                state = {**self._state, "pending": pending}
                if progress is not None:
                    state["progress"] = copy.deepcopy(progress)
                self._save(state)
                self._state = state
                self._replay()
        except BaseException:
            # A failed rename barrier may still have published new intent. A
            # caller catching the error must reopen/replay before another batch.
            self._state = None
            raise
