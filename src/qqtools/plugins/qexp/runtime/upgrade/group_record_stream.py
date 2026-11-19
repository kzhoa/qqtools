"""QQTOOLS-COMPAT-0025. Stream Group documents emitted by the 1.3.29 atomic writer."""

from __future__ import annotations

import base64
import json
import os
import stat
from pathlib import Path

from ..responsibility_store import DurableIO
from .contracts import DeterministicUpgradeError, UpgradeContext

_CHUNK = 65536
_PREFIX = 2048


def _identity(path: Path) -> list[int]:
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode):
        raise DeterministicUpgradeError(f"Group source is not a regular file: {path}")
    return [st.st_ino, st.st_size, st.st_ctime_ns, st.st_mtime_ns]


def _line(state: dict, prefix: bytes, start: int, stop: int) -> None:
    # Released JSON uses indent=2, sorted keys and escaped control characters.
    # Selected scalar values are bounded identifiers, epochs and control states.
    # Unselected lines (including large free-text strings) need only a prefix.
    stripped = prefix.lstrip(b" ")
    depth = len(prefix) - len(stripped)
    if depth == 2:
        state["section"] = "group" if stripped == b'"group": {' else "meta" if stripped == b'"meta": {' else None
    key, sep, value = stripped.partition(b": ")
    if not sep:
        return
    field = key.strip(b'"').decode("ascii", errors="replace")
    value = value.removesuffix(b",")
    group = state["record"]["group"]
    if (
        state["section"] == "group"
        and depth == 4
        and field in {"name", "admission_state", "creation_operation_id", "dispatch_epoch"}
    ):
        group[field] = json.loads(value)
        if field == "admission_state":
            if not stripped.endswith(b","):
                raise DeterministicUpgradeError("Released Group admission field has unexpected encoding")
            state["omit"] = [start, stop]
    elif state["section"] == "meta" and depth == 4 and field == "revision":
        state["record"]["meta"]["revision"] = json.loads(value)


def advance_group_source(
    context: UpgradeContext, path: Path, checkpoint: Path, *, copy_record: bool = True
) -> dict | None:
    """Advance a revision-bound projection and (during audit) staged byte copy."""
    storage = context.storage
    storage.account_metadata_ops(4)
    identity = _identity(path)
    try:
        state = storage.read_json_limited(checkpoint, max_bytes=16384)
    except FileNotFoundError:
        state = {}
    stage = checkpoint.with_suffix(".stage")
    if copy_record and state.get("offset", 0):
        try:
            stage_witness = _identity(stage)
        except FileNotFoundError:
            stage_witness = None
        if state.get("stage_revision") != stage_witness or stage_witness is None:
            state = {}
    if (state.get("source"), state.get("identity"), state.get("copy_record")) != (str(path), identity, copy_record):
        state = {
            "source": str(path),
            "identity": identity,
            "copy_record": copy_record,
            "offset": 0,
            "line_start": 0,
            "prefix": "",
            "section": None,
            "omit": None,
            "record": {
                "meta": {},
                "group": {"creation_operation_id": None},
            },
        }
    offset = state["offset"]
    if offset < identity[1]:
        # At most 1 MiB of input and output per callback. Each kernel read has
        # its own local before/after descriptor witness.
        prefix = base64.b64decode(state["prefix"])
        data = bytearray()
        for _ in range(16):
            read = storage.read_regular_bytes(path, offset + len(data), _CHUNK)
            rev = read.revision
            if [rev.inode, rev.size, rev.ctime_ns, rev.mtime_ns] != identity:
                return None
            data.extend(read.data)
            if read.eof:
                break
        pos = 0
        while pos < len(data):
            nl = data.find(b"\n", pos)
            end = nl if nl >= 0 else len(data)
            prefix += data[pos : min(end, pos + _PREFIX - len(prefix))]
            if nl >= 0 or offset + end == identity[1]:
                stop = offset + end + (nl >= 0)
                _line(state, prefix, state["line_start"], stop)
                state["line_start"] = stop
                prefix = b""
            pos = end + 1
        if copy_record:
            # Stage an exact copy. Removing the short member in-place when it
            # straddles a boundary would make restart offsets ambiguous. Copy
            # subsequent chunks at their final offset after seeing the full line.
            omit = state["omit"]
            context.slice_budget.consume_io_bytes(len(data))
            storage.account_metadata_ops(8)
            flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
            fd = os.open(stage, flags, 0o600)
            with os.fdopen(fd, "r+b") as output:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise DeterministicUpgradeError("Group conversion stage is not a regular file")
                if omit and omit[1] > offset:
                    start, stop = omit
                    output.seek(min(offset, start))
                    output.write(data[: max(0, start - offset)])
                    output.write(data[stop - offset :])
                else:
                    output.seek(offset - (omit[1] - omit[0] if omit else 0))
                    output.write(data)
                output.truncate()
                output.flush()
                os.fsync(fd)
            DurableIO().sync_directory(stage.parent, "group_conversion")
            state["stage_revision"] = _identity(stage)
        state.update(prefix=base64.b64encode(prefix).decode(), offset=offset + len(data))
        storage.atomic_replace(checkpoint, state)
        if state["offset"] < identity[1]:
            return None
    if state["record"]["group"].get("name") != path.stem or "revision" not in state["record"]["meta"]:
        raise DeterministicUpgradeError(f"Cannot verify released Group identity at {path}")
    return state["record"]


def publish_group_copy(context: UpgradeContext, path: Path, checkpoint: Path) -> None:
    state = context.storage.read_json_limited(checkpoint, max_bytes=16384)
    context.storage.account_metadata_ops(10)
    if state["source"] != str(path) or state["identity"] != _identity(path) or state["offset"] != state["identity"][1]:
        raise DeterministicUpgradeError("Group conversion witness changed before publication")
    stage = checkpoint.with_suffix(".stage")
    expected_size = state["identity"][1] - (state["omit"][1] - state["omit"][0])
    if _identity(stage) != state["stage_revision"] or state["stage_revision"][1] != expected_size:
        raise DeterministicUpgradeError("Group conversion stage has an incomplete length")
    os.replace(stage, path)
    DurableIO().sync_directory(path.parent, "group_conversion")
