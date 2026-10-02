"""Read-only legacy evidence slices selected by existing retained capture."""

from __future__ import annotations

import stat
from dataclasses import dataclass
from pathlib import Path

from .authority_scan import is_path_present
from .responsibility import responsibility_root
from .responsibility_backfill import (
    FORMAT,
    LANES,
    STATE_BYTES,
    CapturedEvidenceLocator,
    _canonical_capture_root,
    _relative_record,
    read_capture_locator,
)
from .responsibility_capture import (
    CAPTURE_BYTES,
    CAPTURE_FILE,
    CAPTURE_FORMAT,
    GENERATION_FILE,
    capture_admission,
    is_same_capture_owner,
    source_hold_for_capture,
)
from .responsibility_completion import COMPLETION_FILE
from .responsibility_process_capture import SCAN_FORMAT
from .responsibility_store import Unavailable, read_ledger_instance
from .store import read_json_limited


class SourceCaptureStale(RuntimeError):
    """The exact retained pending batch no longer belongs to this read."""


@dataclass(frozen=True, slots=True)
class SourceReadContext:
    source_root: Path
    capture: dict
    backfill: dict


def load_source_hold_context(runtime_root: Path, owner: dict, *, capture_id: str) -> dict:
    """Read fixed target metadata for separate source retention, without source I/O."""
    runtime_root = _canonical_capture_root(runtime_root)
    if is_path_present(runtime_root / GENERATION_FILE) or is_path_present(runtime_root / COMPLETION_FILE):
        raise SourceCaptureStale("capture is transitioning or complete")
    path = runtime_root / CAPTURE_FILE
    if not stat.S_ISREG(path.lstat().st_mode):
        raise Unavailable("source retention requires regular target capture metadata")
    capture = read_json_limited(path, max_bytes=CAPTURE_BYTES, record_type="retained_capture")
    hold = source_hold_for_capture(capture)
    if hold.target_root != runtime_root:
        raise Unavailable("source retention target differs from its partition")
    if read_ledger_instance(responsibility_root(runtime_root)) != hold.instance:
        raise SourceCaptureStale("source retention target ledger identity changed")
    admission = capture.get("admission")
    if admission is None:
        if capture.get("progress") is not None or capture["pending"]:
            raise Unavailable("unadmitted source retention already contains writer observations")
    elif (
        not isinstance(admission, dict)
        or admission != capture_admission(admission)
        or not is_same_capture_owner(admission, capture_admission(owner))
    ):
        raise SourceCaptureStale("source retention belongs to another persistent binding owner")
    if capture["capture_id"] != capture_id:
        raise SourceCaptureStale("source retention capture identity changed")
    return capture


def load_source_read_context(
    runtime_root: Path,
    owner: dict,
    *,
    capture_id: str,
    backfill_id: str,
    backfill_revision: int,
    lane: str,
    relative: str | None,
) -> SourceReadContext:
    """Select one already-journaled source read using fixed local metadata only."""
    runtime_root = _canonical_capture_root(runtime_root)
    if lane not in LANES:
        raise ValueError("unknown source capture lane")
    if relative is not None:
        _relative_record(lane, relative)
    if is_path_present(runtime_root / GENERATION_FILE) or is_path_present(runtime_root / COMPLETION_FILE):
        raise SourceCaptureStale("capture is transitioning or complete")
    capture_path = runtime_root / CAPTURE_FILE
    backfill_path = runtime_root / "responsibility-capture-backfill.json"
    for path in (capture_path, backfill_path):
        if not stat.S_ISREG(path.lstat().st_mode):
            raise Unavailable("source read requires regular retained capture metadata")
    capture = read_json_limited(capture_path, max_bytes=CAPTURE_BYTES, record_type="retained_capture")
    backfill = read_json_limited(backfill_path, max_bytes=STATE_BYTES, record_type="capture_backfill")
    progress = capture.get("progress")
    source = capture.get("legacy_source")
    if (
        capture.get("format") != CAPTURE_FORMAT
        or capture.get("phase") != "pending"
        or capture.get("runtime_root") != str(runtime_root)
        or not isinstance(source, str)
        or not isinstance(capture.get("instance"), str)
        or not isinstance(progress, dict)
        or progress.get("format") != SCAN_FORMAT
        or type(progress.get("revision")) is not int
        or progress["revision"] < 1
        or type(backfill.get("writer_sweep_revision")) is not int
        or not 1 <= backfill["writer_sweep_revision"] <= progress["revision"]
        or backfill.get("format") != FORMAT
        or backfill.get("instance") != capture["instance"]
        or not isinstance(backfill.get("pending"), list)
        or len(backfill["pending"]) > 64
        or type(backfill.get("revision")) is not int
        or type(backfill.get("lane")) is not int
    ):
        raise Unavailable("source read capture metadata is invalid")
    source_root = _canonical_capture_root(Path(source))
    if source_root == runtime_root or backfill.get("sources") != [str(runtime_root), source]:
        raise Unavailable("source read capture roots are invalid")
    if read_ledger_instance(responsibility_root(runtime_root)) != capture["instance"]:
        raise SourceCaptureStale("source read target ledger identity changed")
    if (
        capture.get("capture_id") != capture_id
        or capture.get("admission") != capture_admission(owner)
        or progress.get("is_sweep_complete") is not True
        or backfill.get("writer_capture_id") != capture_id
        or backfill.get("capture_id") != backfill_id
        or backfill["revision"] != backfill_revision
        or backfill["lane"] != len(LANES) + LANES.index(lane)
        or (
            relative not in backfill["pending"]
            if relative is not None
            else bool(backfill["pending"])
            or backfill.get("at_end") is not False
            or backfill["writer_sweep_revision"] != progress["revision"]
        )
    ):
        raise SourceCaptureStale("source read does not match the exact retained pending batch")
    return SourceReadContext(source_root, capture, backfill)


def validate_source_hold(context: SourceReadContext) -> None:
    """Verify retention before reading or enumerating source evidence."""
    hold_path = context.source_root / CAPTURE_FILE
    if not stat.S_ISREG(hold_path.lstat().st_mode):
        raise Unavailable("source hold must be a regular record")
    hold = read_json_limited(hold_path, max_bytes=CAPTURE_BYTES, record_type="source_hold")
    expected = source_hold_for_capture(context.capture).to_dict()
    if hold != expected:
        raise Unavailable("source read hold differs from the retained target capture")


def read_retained_source_locator(context: SourceReadContext, lane: str, relative: str) -> CapturedEvidenceLocator:
    """Read only the held source; target capture and ledger remain untouched."""
    validate_source_hold(context)
    captured = read_capture_locator(context.source_root, lane, Path(relative), should_require_record=True)
    if captured is None:
        raise Unavailable("retained source read returned no locator")
    return captured
