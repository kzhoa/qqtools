"""Bounded advisory local membership; never a substitute for execution authority."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .authority_scan import is_path_present

BUCKETS = 16
PAGE_SIZE = 64
HEADER_BYTES = 8192
PAGE_BYTES = 65536
ENTRY_BYTES = 65536
INITIAL_ENTRY_BYTES = 61440  # Reserve room for slot/generation growth and handoff.
CLEANUP_RECEIPT_BYTES = 4096
CAPTURED_WRITER_LIMIT = 64
TRANSACTION_BYTES = 524288
MAX_COUNTER = 2**63 - 1
FORMAT = "qexp-local-responsibility-v1"
DATA_NAME = re.compile(r"(?:header|[pam][0-9]+|e[0-9a-f]{64})\Z")
STAGE_PREFIXES = {"active": "a", "maintenance": "m"}


def _empty_header() -> dict:
    return {
        "revision": 0,
        "count": 0,
        "stages": {"version": 1, "active": 0, "maintenance": 0, "before": 0},
    }


class Unavailable(RuntimeError):
    """Storage is incomplete or corrupt; do not infer absence of responsibility."""


class Conflict(RuntimeError):
    """A caller holds stale membership or pagination state."""


def encode(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def identity_key(identity: str) -> str:
    if not isinstance(identity, str) or not identity:
        raise ValueError("identity must be a nonempty string")
    return hashlib.sha256(identity.encode()).hexdigest()


def validate_captured_writer(writer: dict) -> None:
    """Require a boot- and namespace-scoped Linux process identity."""
    if not isinstance(writer, dict) or set(writer) != {
        "host_id",
        "boot_id",
        "pid_namespace",
        "pid",
        "start_time_ticks",
    }:
        raise ValueError("captured writer requires an exact process identity")
    if not isinstance(writer["host_id"], str) or not 1 <= len(writer["host_id"]) <= 128:
        raise ValueError("captured writer requires a bounded host identity")
    boot = writer["boot_id"]
    if not isinstance(boot, str) or str(uuid.UUID(boot)) != boot:
        raise ValueError("captured writer boot ID must be a canonical UUID")
    if any(type(writer[key]) is not int or writer[key] < 1 for key in ("pid_namespace", "pid")):
        raise ValueError("captured writer namespace and PID must be positive integers")
    if type(writer["start_time_ticks"]) is not int or writer["start_time_ticks"] < 0:
        raise ValueError("captured writer start time must be a nonnegative integer")


class DurableIO:
    """Same-directory atomic replacement and explicit grouped durability barriers."""

    @contextmanager
    def lock(self, path: Path) -> Iterator[None]:
        with path.open("rb") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def read(self, path: Path, limit: int) -> dict:
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise Unavailable(f"record exceeds byte limit: {path}")
        try:
            value = json.loads(data)
        except (ValueError, UnicodeError) as exc:
            raise Unavailable(f"invalid JSON: {path}") from exc
        if not isinstance(value, dict):
            raise Unavailable(f"expected object: {path}")
        return value

    def sync(self, fd: int, kind: str, label: str) -> None:
        os.fsync(fd)

    def sync_directory(self, path: Path, label: str) -> None:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            self.sync(fd, "directory_fsync", label)
        finally:
            os.close(fd)

    def replace(self, path: Path, value: dict, *, should_sync_directory: bool = True) -> None:
        scratch = path.parent / "scratch"
        with scratch.open("wb") as stream:
            stream.write(encode(value))
            stream.flush()
            self.sync(stream.fileno(), "file_fsync", path.name)
        os.replace(scratch, path)
        if should_sync_directory:
            self.sync_directory(path.parent, path.name)

    def delete(self, path: Path, *, should_sync_directory: bool = True) -> None:
        path.unlink(missing_ok=True)
        if should_sync_directory:
            self.sync_directory(path.parent, path.name)


class Ledger:
    """Paged active/maintenance membership, serialized by one flock per bucket.

    Create accepts a nonexistent path only. Reopen requires the completed marker.
    An entry generation changes on handoff/republication, but not on physical moves.
    Pagination is revision-checked per bucket, not a cross-bucket snapshot.
    """

    io_type = DurableIO

    def __init__(self, root: Path, io: DurableIO | None = None) -> None:
        self.root = root
        self.io = io if io is not None else self.io_type()
        try:
            marker = self.io.read(root / "marker", HEADER_BYTES)
            if marker["format"] != FORMAT or marker["buckets"] != BUCKETS:
                raise Unavailable("unsupported responsibility format")
            self.instance = marker["instance"]
            if not isinstance(self.instance, str) or re.fullmatch("[0-9a-f]{32}", self.instance) is None:
                raise Unavailable("invalid responsibility identity")
        except (OSError, KeyError, TypeError) as exc:
            raise Unavailable(f"incomplete responsibility initialization: {root}") from exc

    @classmethod
    def create(cls, root: Path) -> Ledger:
        io = cls.io_type()
        root.mkdir()  # Refuse to initialize over any existing directory.
        for bucket in range(BUCKETS):
            directory = root / str(bucket)
            directory.mkdir()
            with (directory / "lock").open("xb") as stream:
                io.sync(stream.fileno(), "file_fsync", "lock")
            io.replace(directory / "header", _empty_header())
        io.sync_directory(root, "buckets")
        io.replace(root / "marker", {"format": FORMAT, "buckets": BUCKETS, "instance": uuid.uuid4().hex})
        io.sync_directory(root.parent, "root")
        return cls(root)

    @classmethod
    def open_or_create(cls, root: Path) -> Ledger:
        """Publish an empty store atomically, resuming interrupted initialization.

        The caller must hold the root's initialization lock across this operation.
        The fixed sibling staging directory is private to initialization; readers
        and publishers must use only root. Existing published stores are never
        reset, even if their marker is missing or unreadable.
        """
        if is_path_present(root):
            ledger = cls(root)
            cls._finish_initialization(root, ledger.io)
            return ledger
        staging = root.with_name(f".{root.name}.building")
        io = cls.io_type()
        if staging.is_symlink():
            raise Unavailable("initialization staging directory must not be a symlink")
        staging.mkdir(exist_ok=True)
        cls._validate_initialization(staging, io)
        if is_path_present(staging / "marker"):
            cls(staging)
        for bucket in range(BUCKETS):
            directory = staging / str(bucket)
            directory.mkdir(exist_ok=True)
            with (directory / "lock").open("ab") as stream:
                io.sync(stream.fileno(), "file_fsync", "lock")
            io.replace(directory / "header", _empty_header())
        io.sync_directory(staging, "buckets")
        io.replace(staging / "initializing", {"format": FORMAT})
        io.replace(staging / "marker", {"format": FORMAT, "buckets": BUCKETS, "instance": uuid.uuid4().hex})
        # The initialization lock excludes another publisher of this root. No
        # caller can have acknowledged membership in the private staging store.
        os.rename(staging, root)
        cls._finish_initialization(root, io)
        return cls(root)

    @staticmethod
    def _finish_initialization(root: Path, io: DurableIO) -> None:
        try:
            pending = io.read(root / "initializing", HEADER_BYTES)
        except FileNotFoundError:
            return
        if pending != {"format": FORMAT}:
            raise Unavailable("invalid initialization completion record")
        # A rename visible after an interrupted parent fsync is not yet a durable
        # root. Retain this retry record until its parent barrier has succeeded.
        io.sync_directory(root.parent, "root")
        io.delete(root / "initializing")

    @staticmethod
    def _validate_initialization(root: Path, io: DurableIO) -> None:
        """Accept only the fixed empty layout, never reinterpret occupied storage."""
        allowed = {str(bucket) for bucket in range(BUCKETS)} | {"marker", "scratch", "initializing"}
        with os.scandir(root) as entries:
            for entry in entries:
                if entry.name not in allowed or entry.is_symlink():
                    raise Unavailable(f"unexpected initialization entry: {entry.path}")
                if entry.name in {"marker", "scratch", "initializing"}:
                    if not entry.is_file(follow_symlinks=False):
                        raise Unavailable(f"invalid initialization file: {entry.path}")
                    if entry.name == "initializing" and io.read(Path(entry.path), HEADER_BYTES) != {"format": FORMAT}:
                        raise Unavailable("invalid initialization completion record")
                    continue
                if not entry.is_dir(follow_symlinks=False):
                    raise Unavailable(f"invalid initialization bucket: {entry.path}")
                with os.scandir(entry.path) as files:
                    for file in files:
                        if file.name not in {"lock", "header", "scratch"} or not file.is_file(follow_symlinks=False):
                            raise Unavailable(f"unexpected initialization file: {file.path}")
                        if file.name == "header" and io.read(Path(file.path), HEADER_BYTES) not in (
                            {"revision": 0, "count": 0},
                            _empty_header(),
                        ):
                            raise Unavailable("initialization cannot reset an occupied bucket")
                        if file.name == "lock" and file.stat().st_size != 0:
                            raise Unavailable("invalid initialization lock")

    @contextmanager
    def _locked(self, bucket: int) -> Iterator[Path]:
        if type(bucket) is not int or not 0 <= bucket < BUCKETS:
            raise ValueError("invalid bucket")
        directory = self.root / str(bucket)
        with self.io.lock(directory / "lock"):
            try:
                self._recover(directory)
                yield directory
            except (OSError, KeyError, TypeError, IndexError) as exc:
                raise Unavailable(f"incomplete bucket {bucket}: {exc}") from exc

    def _header(self, directory: Path) -> dict:
        value = self.io.read(directory / "header", HEADER_BYTES)
        self._validate_header(value)
        return value

    @staticmethod
    def _validate_header(value: dict) -> None:
        if "writer_capture_incomplete" in value and value["writer_capture_incomplete"] is not True:
            raise Unavailable("invalid incomplete writer capture marker")
        if any(type(value.get(key)) is not int or not 0 <= value[key] <= MAX_COUNTER for key in ("revision", "count")):
            raise Unavailable("invalid bucket header")
        stages = value.get("stages")
        if "stages" in value and (
            not isinstance(stages, dict)
            or type(stages.get("version")) is not int
            or stages["version"] != 1
            or any(
                type(stages.get(key)) is not int or not 0 <= stages[key] <= MAX_COUNTER
                for key in ("active", "maintenance", "before")
            )
            or stages["active"] + stages["maintenance"] > value["count"]
        ):
            raise Unavailable("invalid stage index header")

    def _page(self, directory: Path, number: int, count: int, *, prefix: str = "p") -> list[str]:
        entries = self.io.read(directory / f"{prefix}{number}", PAGE_BYTES).get("entries")
        expected = min(PAGE_SIZE, count - number * PAGE_SIZE)
        if (
            not isinstance(entries, list)
            or not 0 < expected <= PAGE_SIZE
            or len(entries) != expected
            or len(set(entries)) != len(entries)
            or any(not isinstance(key, str) or re.fullmatch("[0-9a-f]{64}", key) is None for key in entries)
        ):
            raise Unavailable("invalid dense page")
        return entries

    def _entry(self, directory: Path, key: str, slot: int | None = None) -> dict:
        value = self.io.read(directory / f"e{key}", ENTRY_BYTES)
        if (
            not isinstance(value.get("identity"), str)
            or not value["identity"]
            or identity_key(value["identity"]) != key
            or type(value.get("generation")) is not int
            or value["generation"] <= 0
            or type(value.get("slot")) is not int
            or value["slot"] < 0
            or value.get("stage") not in {"active", "maintenance"}
            or ("stage_slot" in value and (type(value["stage_slot"]) is not int or value["stage_slot"] < 0))
            or not isinstance(value.get("payload"), dict)
            or (
                "legacy_source" in value
                and (not isinstance(value["legacy_source"], str) or not Path(value["legacy_source"]).is_absolute())
            )
            or (slot is not None and value["slot"] != slot)
        ):
            raise Unavailable("invalid membership locator")
        if "captured_writers" in value:
            writers = value["captured_writers"]
            if not isinstance(writers, list) or not 1 <= len(writers) <= CAPTURED_WRITER_LIMIT:
                raise Unavailable("invalid captured writer inventory")
            try:
                for writer in writers:
                    validate_captured_writer(writer)
            except ValueError as exc:
                raise Unavailable("invalid captured writer identity") from exc
        if "writer_capture_incomplete" in value and value["writer_capture_incomplete"] is not True:
            raise Unavailable("invalid incomplete writer capture marker")
        return value

    def _recover(self, directory: Path) -> None:
        try:
            transaction = self.io.read(directory / "pending", TRANSACTION_BYTES)
        except FileNotFoundError:
            if is_path_present(directory / "scratch"):
                self.io.delete(directory / "scratch")
            return
        header = self._header(directory)
        base, revision = transaction["base"], transaction["revision"]
        if type(base) is not int or type(revision) is not int or base < 0 or revision != base + 1:
            raise Unavailable("invalid transaction generation")
        if header["revision"] not in (base, revision):
            raise Unavailable("transaction would overwrite a different generation")
        images = transaction["images"]
        self._validate_images(images)
        if transaction.get("digest") != hashlib.sha256(encode(images)).hexdigest():
            raise Unavailable("transaction image checksum mismatch")
        if images["header"]["revision"] != revision:
            raise Unavailable("transaction header generation mismatch")
        self._apply(directory, images)
        self.io.delete(directory / "pending")

    @staticmethod
    def _validate_images(images: dict) -> None:
        if not isinstance(images, dict) or not 1 <= len(images) <= 8 or not isinstance(images.get("header"), dict):
            raise Unavailable("invalid transaction write set")
        Ledger._validate_header(images["header"])
        for name, value in images.items():
            if DATA_NAME.fullmatch(name) is None:
                raise Unavailable("invalid transaction target")
            limit = HEADER_BYTES if name == "header" else ENTRY_BYTES if name.startswith("e") else PAGE_BYTES
            if value is not None and (not isinstance(value, dict) or len(encode(value)) > limit):
                raise Unavailable("transaction image exceeds encoding budget")

    def _apply(self, directory: Path, images: dict) -> None:
        for name, value in images.items():
            if name == "header":
                continue
            if value is None:
                self.io.delete(directory / name, should_sync_directory=False)
            else:
                self.io.replace(directory / name, value, should_sync_directory=False)
        # Pending remains durable across any partial namespace persistence. One
        # directory barrier commits all after-images before pending can be cleared.
        self.io.replace(directory / "header", images["header"], should_sync_directory=False)
        self.io.sync_directory(directory, "images")

    def _commit(self, directory: Path, header: dict, images: dict, count: int) -> int:
        revision = header["revision"] + 1
        if "stages" in header and sum(header["stages"][stage] for stage in STAGE_PREFIXES) == count:
            header["stages"]["before"] = 0
        images["header"] = {**header, "count": count, "revision": revision}
        self._validate_images(images)
        transaction = {
            "base": header["revision"],
            "revision": revision,
            "images": images,
            "digest": hashlib.sha256(encode(images)).hexdigest(),
        }
        size = len(encode(transaction))
        if size > TRANSACTION_BYTES:
            raise ValueError("transaction exceeds transaction byte budget")
        self.io.replace(directory / "pending", transaction)
        self._apply(directory, images)
        self.io.delete(directory / "pending")
        return revision

    def publish(self, identity: str, payload: dict) -> int:
        """Publish before a covered side effect; exact retries perform no writes."""
        return self._publish(identity, payload)

    def capture_local(self, identity: str, payload: dict) -> int:
        """Capture existing evidence, retaining known locator fields on retry."""
        self._resolved_payload({"task_id": None, "attempt_number": None}, payload)
        return self._publish(identity, payload, is_capture=True)

    def capture_writer(self, identity: str, payload: dict, writer: dict, *, source_root: Path | None = None) -> int:
        """Retain an observed writer even if old cleanup removes its evidence.

        The caller must observe the real host, boot, namespace, PID and start identity
        and serialize capture with evidence cleanup. This only records known
        writers; it does not certify complete writer discovery or grant authority.
        """
        validate_captured_writer(writer)
        self._resolved_payload({"task_id": None, "attempt_number": None}, payload)
        if source_root is not None and not source_root.is_absolute():
            raise ValueError("captured writer source must be absolute")
        return self._publish(
            identity,
            payload,
            is_capture=True,
            captured_writer=dict(writer),
            legacy_source=str(source_root) if source_root is not None else None,
        )

    def capture_source(self, identity: str, payload: dict, source_root: Path) -> int:
        """Retain a legacy writer's inbox before moving any of its evidence.

        The caller serializes capture and cleanup with the evidence-write guard.
        Unknown locator fields may be filled later, never replace known values.
        """
        if not source_root.is_absolute():
            raise ValueError("legacy evidence source must be absolute")
        return self._publish(identity, payload, legacy_source=str(source_root))

    @staticmethod
    def _resolved_payload(previous: dict, incoming: dict) -> dict:
        if set(previous) != {"task_id", "attempt_number"} or set(incoming) != set(previous):
            raise Conflict("incompatible recovery locator")
        result = dict(previous)
        for key, value in incoming.items():
            if value is None:
                continue
            if previous[key] not in (None, value):
                raise Conflict("recovery locator already has a different identity")
            result[key] = value
        return result

    def resolve_locator(self, identity: str, payload: dict) -> int:
        """Fill missing imported locator fields without replacing known truth."""
        key = identity_key(identity)
        with self._locked(int(key[0], 16)) as directory:
            header = self._header(directory)
            entry = self._entry(directory, key)
            self._validate_slot(directory, key, entry, header)
            resolved = self._resolved_payload(entry["payload"], payload)
            if resolved == entry["payload"]:
                return entry["generation"]
            if entry["stage"] != "active":
                raise Conflict("only active locators may be resolved")
            entry.update(payload=resolved, generation=header["revision"] + 1)
            return self._commit(directory, header, {f"e{key}": entry}, header["count"])

    def _publish(
        self,
        identity: str,
        payload: dict,
        *,
        legacy_source: str | None = None,
        is_capture: bool = False,
        captured_writer: dict | None = None,
    ) -> int:
        key = identity_key(identity)
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        with self._locked(int(key[0], 16)) as directory:
            header = self._header(directory)
            if is_path_present(directory / f"e{key}"):
                entry = self._entry(directory, key)
                if entry["identity"] != identity:
                    raise Conflict("identity already has different responsibility")
                self._validate_slot(directory, key, entry, header)
                if legacy_source is not None or is_capture:
                    resolved = self._resolved_payload(entry["payload"], payload)
                    if legacy_source is not None and entry.get("legacy_source") not in (None, legacy_source):
                        raise Conflict("Attempt already has a different legacy evidence source")
                    has_new_source = legacy_source is not None and entry.get("legacy_source") != legacy_source
                    writers = entry.get("captured_writers", [])
                    has_new_writer = captured_writer is not None and captured_writer not in writers
                    if resolved != entry["payload"] or has_new_source or has_new_writer:
                        if entry["stage"] != "active":
                            raise Conflict("cleanup already owns this Attempt")
                        previous = dict(entry)
                        entry.update(payload=resolved, generation=header["revision"] + 1)
                        if has_new_source:
                            entry["legacy_source"] = legacy_source
                        if has_new_writer:
                            if len(writers) >= CAPTURED_WRITER_LIMIT:
                                self._reject_writer_capture(
                                    directory, header, previous, "writer inventory exceeds its bound"
                                )
                            entry["captured_writers"] = [*writers, captured_writer]
                            if len(encode(entry)) > INITIAL_ENTRY_BYTES:
                                self._reject_writer_capture(
                                    directory, header, previous, "writer inventory exceeds encoding budget"
                                )
                        return self._commit(directory, header, {f"e{key}": entry}, header["count"])
                elif entry["payload"] != payload:
                    raise Conflict("identity already has different responsibility")
                return entry["generation"]
            slot = header["count"]
            page = slot // PAGE_SIZE
            entries = self._page(directory, page, slot) if slot % PAGE_SIZE else []
            entries.append(key)
            entry = {
                "identity": identity,
                "generation": header["revision"] + 1,
                "slot": slot,
                "stage": "active",
                "payload": payload,
            }
            if legacy_source is not None:
                entry["legacy_source"] = legacy_source
            if captured_writer is not None:
                entry["captured_writers"] = [captured_writer]
            if len(encode(entry)) > INITIAL_ENTRY_BYTES:
                if captured_writer is not None:
                    self._reject_writer_capture(directory, header, None, "writer inventory exceeds encoding budget")
                raise Unavailable("entry exceeds encoding budget")
            images = {f"p{page}": {"entries": entries}, f"e{key}": entry}
            self._stage_add(directory, header, images, key, entry)
            return self._commit(directory, header, images, slot + 1)

    def _reject_writer_capture(self, directory: Path, header: dict, entry: dict | None, reason: str) -> None:
        """Retain a sticky incomplete marker before reporting a bounded refusal."""
        if header.get("writer_capture_incomplete") or (entry is not None and entry.get("writer_capture_incomplete")):
            raise Unavailable(reason)
        images = {}
        if entry is not None:
            marked = {**entry, "writer_capture_incomplete": True, "generation": header["revision"] + 1}
            if len(encode(marked)) <= ENTRY_BYTES:
                if entry.get("writer_capture_incomplete") is not True:
                    images[f"e{identity_key(entry['identity'])}"] = marked
            else:
                header["writer_capture_incomplete"] = True
        else:
            header["writer_capture_incomplete"] = True
        # A full entry cannot be required to grow in order to record failure.
        # The fixed header marker conservatively retains its entire bucket.
        self._commit(directory, header, images, header["count"])
        raise Unavailable(reason)

    def _validate_slot(self, directory: Path, key: str, entry: dict, header: dict) -> None:
        slot = entry["slot"]
        if (
            entry["generation"] > header["revision"]
            or slot >= header["count"]
            or self._page(directory, slot // PAGE_SIZE, header["count"])[slot % PAGE_SIZE] != key
        ):
            raise Unavailable("locator does not match page")
        stages = header.get("stages")
        if "stage_slot" in entry:
            if stages is None:
                raise Unavailable("stage index header is missing")
            stage_slot = entry["stage_slot"]
            count = stages[entry["stage"]]
            if (
                stage_slot >= count
                or self._page(directory, stage_slot // PAGE_SIZE, count, prefix=STAGE_PREFIXES[entry["stage"]])[
                    stage_slot % PAGE_SIZE
                ]
                != key
            ):
                raise Unavailable("stage locator does not match page")
        elif stages is not None and stages["active"] + stages["maintenance"] == header["count"]:
            raise Unavailable("complete stage index has an unindexed locator")

    def _stage_add(self, directory: Path, header: dict, images: dict, key: str, entry: dict) -> None:
        stages = header.setdefault("stages", {"version": 1, "active": 0, "maintenance": 0, "before": header["count"]})
        stage = entry["stage"]
        slot = stages[stage]
        prefix = STAGE_PREFIXES[stage]
        page = slot // PAGE_SIZE
        entries = self._page(directory, page, slot, prefix=prefix) if slot % PAGE_SIZE else []
        entries.append(key)
        entry["stage_slot"] = slot
        images[f"e{key}"] = entry
        images[f"{prefix}{page}"] = {"entries": entries}
        stages[stage] += 1

    def _stage_remove(self, directory: Path, header: dict, images: dict, entry: dict) -> None:
        if "stage_slot" not in entry:
            return
        stage = entry["stage"]
        count = header["stages"][stage]
        slot, tail = entry["stage_slot"], count - 1
        page, last_page = slot // PAGE_SIZE, tail // PAGE_SIZE
        prefix = STAGE_PREFIXES[stage]
        entries = self._page(directory, page, count, prefix=prefix)
        last = entries if page == last_page else self._page(directory, last_page, count, prefix=prefix)
        if slot != tail:
            moved_key = last[-1]
            moved = images.get(f"e{moved_key}")
            if moved is None:
                moved = self._entry(directory, moved_key)
            if moved["stage"] != stage or moved.get("stage_slot") != tail:
                raise Unavailable("stage compaction locator mismatch")
            moved["stage_slot"] = slot
            entries[slot % PAGE_SIZE] = moved_key
            images[f"e{moved_key}"] = moved
        last.pop()
        images[f"{prefix}{page}"] = {"entries": entries} if entries else None
        images[f"{prefix}{last_page}"] = {"entries": last} if last else None
        header["stages"][stage] -= 1
        entry.pop("stage_slot")

    def build_stage_index(self, bucket: int, limit: int = PAGE_SIZE) -> bool:
        """Resume a decreasing base-page sweep; every new publication is indexed."""
        if type(limit) is not int or not 1 <= limit <= PAGE_SIZE:
            raise ValueError("stage build limit must be in 1..64")
        with self._locked(bucket) as directory:
            header = self._header(directory)
            if "stages" in header and header["stages"]["active"] + header["stages"]["maintenance"] == header["count"]:
                return True
            stages = header.setdefault(
                "stages", {"version": 1, "active": 0, "maintenance": 0, "before": header["count"]}
            )
            end = min(header["count"], stages["before"])
            if end == 0:
                if header["count"]:
                    raise Unavailable("stage build ended with unindexed responsibilities")
                self._commit(directory, header, {}, 0)
                return True
            page = (end - 1) // PAGE_SIZE
            entries = self._page(directory, page, header["count"])
            start = max(page * PAGE_SIZE, end - limit)
            for slot in range(end - 1, start - 1, -1):
                key = entries[slot % PAGE_SIZE]
                entry = self._entry(directory, key, slot)
                self._validate_slot(directory, key, entry, header)
                images = {}
                if "stage_slot" not in entry:
                    self._stage_add(directory, header, images, key, entry)
                header["stages"]["before"] = slot
                if images or slot == start:
                    self._commit(directory, header, images, header["count"])
                    header = self._header(directory)
            return header["stages"]["active"] + header["stages"]["maintenance"] == header["count"]

    def handoff(self, identity: str, generation: int, *, cleanup_receipt: dict | None = None) -> int:
        """Transfer to maintenance, optionally retaining the caller's cleanup proof.

        The storage layer validates encoding, not external completion or process
        identity. A bare maintenance stage never proves that evidence can be deleted.
        """
        if cleanup_receipt is not None and (
            not isinstance(cleanup_receipt, dict) or len(encode(cleanup_receipt)) > CLEANUP_RECEIPT_BYTES
        ):
            raise Unavailable("cleanup receipt exceeds encoding budget")
        key = identity_key(identity)
        with self._locked(int(key[0], 16)) as directory:
            header = self._header(directory)
            entry = self._entry(directory, key)
            self._validate_slot(directory, key, entry, header)
            if header.get("writer_capture_incomplete") or entry.get("writer_capture_incomplete"):
                raise Conflict("writer capture is incomplete; cleanup handoff is disabled")
            if entry["generation"] != generation:
                raise Conflict("stale handoff generation")
            if cleanup_receipt is not None and entry.get("cleanup_receipt") not in (None, cleanup_receipt):
                raise Conflict("cleanup receipt is immutable")
            if entry["stage"] == "maintenance" and (
                cleanup_receipt is None or entry.get("cleanup_receipt") == cleanup_receipt
            ):
                return generation
            images = {}
            if entry["stage"] == "active":
                self._stage_remove(directory, header, images, entry)
            entry["stage"] = "maintenance"
            if "stage_slot" not in entry:
                self._stage_add(directory, header, images, key, entry)
            if cleanup_receipt is not None:
                entry["cleanup_receipt"] = cleanup_receipt
            entry["generation"] = header["revision"] + 1
            images[f"e{key}"] = entry
            return self._commit(directory, header, images, header["count"])

    def retire(self, identity: str, generation: int) -> bool:
        """CAS retirement after external obligations complete; never deletes evidence."""
        key = identity_key(identity)
        with self._locked(int(key[0], 16)) as directory:
            header = self._header(directory)
            if not is_path_present(directory / f"e{key}"):
                return False
            entry = self._entry(directory, key)
            if entry["generation"] != generation or entry["stage"] != "maintenance":
                raise Conflict("stale retirement or active responsibility")
            self._validate_slot(directory, key, entry, header)
            slot, tail = entry["slot"], header["count"] - 1
            page, last_page = slot // PAGE_SIZE, tail // PAGE_SIZE
            entries = self._page(directory, page, header["count"])
            last = entries if page == last_page else self._page(directory, last_page, header["count"])
            images: dict[str, Any] = {f"e{key}": None}
            if slot != tail:
                moved_key = last[-1]
                moved = self._entry(directory, moved_key, tail)
                moved["slot"] = slot
                entries[slot % PAGE_SIZE] = moved_key
                images[f"e{moved_key}"] = moved
            last.pop()
            images[f"p{page}"] = {"entries": entries} if entries else None
            images[f"p{last_page}"] = {"entries": last} if last else None
            self._stage_remove(directory, header, images, entry)
            self._commit(directory, header, images, tail)
            return True

    def page(self, bucket: int, cursor: dict | None = None, limit: int = PAGE_SIZE) -> tuple[list[dict], dict | None]:
        """Read at most one physical page; changed buckets invalidate continuation.

        A returned cursor must be retried from the bucket start after Conflict.
        This is deliberately not yet a fair scheduling cursor under continuous churn.
        """
        if type(limit) is not int or not 1 <= limit <= PAGE_SIZE:
            raise ValueError("limit must be in 1..64")
        with self._locked(bucket) as directory:
            header = self._header(directory)
            slot = 0
            if cursor is not None:
                if (
                    cursor.get("instance") != self.instance
                    or cursor.get("bucket") != bucket
                    or cursor.get("revision") != header["revision"]
                ):
                    raise Conflict("pagination changed; restart this bucket")
                slot = cursor.get("slot")
                if type(slot) is not int or not 0 <= slot < header["count"]:
                    raise ValueError("invalid cursor slot")
            if slot == header["count"]:
                return [], None
            entries = self._page(directory, slot // PAGE_SIZE, header["count"])
            end = min(header["count"], slot + limit, (slot // PAGE_SIZE + 1) * PAGE_SIZE)
            records = [self._entry(directory, entries[index % PAGE_SIZE], index) for index in range(slot, end)]
            next_cursor = None
            if end < header["count"]:
                next_cursor = {"instance": self.instance, "bucket": bucket, "revision": header["revision"], "slot": end}
            return records, next_cursor

    def lookup(self, identity: str) -> dict:
        key = identity_key(identity)
        with self._locked(int(key[0], 16)) as directory:
            header = self._header(directory)
            entry = self._entry(directory, key)
            self._validate_slot(directory, key, entry, header)
            return entry

    def find(self, identity: str) -> dict | None:
        """Find a locator under its bucket fence; unavailable storage still fails.

        No locator means no recorded membership, not absence of external work.
        """
        key = identity_key(identity)
        with self._locked(int(key[0], 16)) as directory:
            header = self._header(directory)
            if not is_path_present(directory / f"e{key}"):
                return None
            entry = self._entry(directory, key)
            self._validate_slot(directory, key, entry, header)
            return entry

    def has_members(self) -> bool:
        """Check the fixed bucket headers; incomplete storage cannot prove idle."""
        for bucket in range(BUCKETS):
            with self._locked(bucket) as directory:
                if self._header(directory)["count"]:
                    return True
        return False

    def service_page(
        self, bucket: int, cursor: dict | None = None, limit: int = PAGE_SIZE, *, stage: str | None = None
    ) -> tuple[list[dict], dict | None]:
        """Visit a finite reverse sweep without restarting after mutation.

        A retained member only moves left during compaction. An unvisited member
        therefore stays below the decreasing cursor until returned. Appends never
        extend the sweep. Previously visited members may be returned again after a
        move; consumers must revalidate truth and make reconciliation idempotent.
        """
        if type(limit) is not int or not 1 <= limit <= PAGE_SIZE:
            raise ValueError("limit must be in 1..64")
        if stage is not None and stage not in STAGE_PREFIXES:
            raise ValueError("invalid responsibility stage")
        with self._locked(bucket) as directory:
            header = self._header(directory)
            count = header["count"]
            prefix = "p"
            if stage is not None:
                stages = header.get("stages")
                if stages is None or stages["active"] + stages["maintenance"] != count:
                    raise Unavailable("responsibility stage index is incomplete")
                count, prefix = stages[stage], STAGE_PREFIXES[stage]
            end = count
            if cursor is not None:
                if (
                    cursor.get("instance") != self.instance
                    or cursor.get("bucket") != bucket
                    or cursor.get("stage") != stage
                ):
                    raise Conflict("service cursor belongs to another ledger or bucket")
                before = cursor.get("before")
                if type(before) is not int or not 0 < before <= MAX_COUNTER:
                    raise ValueError("invalid service cursor boundary")
                end = min(end, before)
            if end == 0:
                return [], None
            number = (end - 1) // PAGE_SIZE
            entries = self._page(directory, number, count, prefix=prefix)
            start = max(number * PAGE_SIZE, end - limit)
            records = []
            for index in range(end - 1, start - 1, -1):
                entry = self._entry(directory, entries[index % PAGE_SIZE], index if stage is None else None)
                if stage is not None and (entry["stage"] != stage or entry.get("stage_slot") != index):
                    raise Unavailable("stage page does not match its locator")
                records.append(entry)
            following = (
                {"instance": self.instance, "bucket": bucket, "stage": stage, "before": start} if start else None
            )
            return records, following


class ServiceTraversal:
    """One bounded bucket slice per turn, with independent recurring sweeps."""

    def __init__(self, ledger: Ledger, *, stage: str | None = None) -> None:
        self.ledger = ledger
        if stage is not None and stage not in STAGE_PREFIXES:
            raise ValueError("invalid responsibility stage")
        self.stage = stage
        self.cursors: list[dict | None] = [None] * BUCKETS
        self.completed: set[int] = set()
        self.next_bucket = 0

    def take(self, limit: int = PAGE_SIZE) -> list[dict]:
        bucket = self.next_bucket
        # Advance even on a failed bucket; its cursor is retained for recovery.
        self.next_bucket = (bucket + 1) % BUCKETS
        if self.stage is None:
            entries, cursor = self.ledger.service_page(bucket, self.cursors[bucket], limit)
        else:
            entries, cursor = self.ledger.service_page(bucket, self.cursors[bucket], limit, stage=self.stage)
        self.cursors[bucket] = cursor
        if cursor is None:
            self.completed.add(bucket)
        return entries

    @property
    def has_completed_initial_sweeps(self) -> bool:
        return len(self.completed) == BUCKETS
