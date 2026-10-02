"""Replayable recapture after the same persistent owner reacquires its registration."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from .authority_scan import is_path_present
from .responsibility_capture import (
    CAPTURE_BYTES,
    CAPTURE_FILE,
    GENERATION_FILE,
    SOURCE_CAPTURE_FORMAT,
    SourceHold,
    _capture_parent_guard,
    capture_admission,
    is_same_capture_owner,
)
from .responsibility_completion import (
    COMPLETION_BYTES,
    COMPLETION_FILE,
    SOURCE_RELEASE_FILE,
    _digest,
    read_capture_completion,
)
from .responsibility_process_capture import _process_scope
from .responsibility_store import Conflict, DurableIO, Ledger, Unavailable, read_ledger_instance
from .store import atomic_replace, read_json_limited, require_json_size

FORMAT = "qexp-capture-generation-transition-v1"
STATE_BYTES = CAPTURE_BYTES + 2 * COMPLETION_BYTES


@dataclass(frozen=True, slots=True)
class CaptureGenerationTransition:
    runtime_root: Path
    source_root: Path | None
    state: dict
    before: dict
    after: dict


def load_capture_generation_transition(
    runtime_root: Path, *, legacy_source: Path | None, owner: dict
) -> CaptureGenerationTransition:
    """Read a replayable transition using fixed local metadata only."""
    from .responsibility import responsibility_root
    from .responsibility_backfill import _canonical_capture_root

    runtime_root = _canonical_capture_root(runtime_root)
    source = str(_canonical_capture_root(legacy_source)) if legacy_source is not None else None
    if source == str(runtime_root):
        raise Unavailable("generation transition source equals its target")
    admission = capture_admission(owner)
    path = runtime_root / GENERATION_FILE
    if is_path_present(path):
        state = read_json_limited(path, max_bytes=STATE_BYTES)
    else:
        proof = read_capture_completion(runtime_root, should_sync=False)
        if proof is None:
            raise Unavailable("generation transition requires completed prior capture")
        release_path = runtime_root / SOURCE_RELEASE_FILE
        release = read_json_limited(release_path, max_bytes=COMPLETION_BYTES) if is_path_present(release_path) else None
        if release is not None and (source is None or release != _release(proof)):
            raise Unavailable("invalid previous source release")
        state = {
            "format": FORMAT,
            "owner": admission,
            "previous_completion": proof,
            "previous_capture": read_json_limited(runtime_root / CAPTURE_FILE, max_bytes=CAPTURE_BYTES),
            "source_release": release,
        }
    before, after = _validate(state, runtime_root, source, admission)
    current = read_json_limited(runtime_root / CAPTURE_FILE, max_bytes=CAPTURE_BYTES)
    if current not in (before, after) or read_ledger_instance(responsibility_root(runtime_root)) != before["instance"]:
        raise Unavailable("generation transition checkpoint or Ledger changed")
    for name, expected in (
        (COMPLETION_FILE, state["previous_completion"]),
        (SOURCE_RELEASE_FILE, state["source_release"]),
    ):
        record_path = runtime_root / name
        if is_path_present(record_path) and read_json_limited(record_path, max_bytes=COMPLETION_BYTES) != expected:
            raise Unavailable("generation transition metadata changed")
    return CaptureGenerationTransition(runtime_root, Path(source) if source is not None else None, state, before, after)


def generation_source_hold(transition: CaptureGenerationTransition) -> SourceHold | None:
    if transition.source_root is None:
        return None
    return SourceHold(
        transition.source_root, transition.runtime_root, transition.before["instance"], transition.before["capture_id"]
    )


def _retain_generation_source_locked(
    transition: CaptureGenerationTransition, before_source_write: Callable[[], None]
) -> None:
    source = transition.source_root
    if source is None:
        return
    if source.resolve() != source or not source.is_dir():
        raise Unavailable("generation transition lost its canonical source directory")
    hold = generation_source_hold(transition).to_dict()
    path = source / CAPTURE_FILE
    if is_path_present(path):
        stored = read_json_limited(path, max_bytes=CAPTURE_BYTES)
        normal = {key: value for key, value in stored.items() if key != "admission"}
        normal["phase"] = "pending"
        if stored != hold and (
            normal != hold
            or stored.get("phase") != "generation_pending"
            or not isinstance(stored.get("admission"), dict)
            or stored["admission"] != capture_admission(stored["admission"])
            or not is_same_capture_owner(stored["admission"], transition.state["owner"])
        ):
            raise Unavailable("generation transition source belongs to another capture")
    elif transition.state["source_release"] is None:
        raise Unavailable("generation transition lost unreleased source retention")
    before_source_write()
    atomic_replace(path, {**hold, "phase": "generation_pending", "admission": transition.state["owner"]})


def retain_generation_source(
    transition: CaptureGenerationTransition, *, before_source_write: Callable[[], None]
) -> SourceHold | None:
    """Fence old completion release with source-only effects before local reset."""
    source = transition.source_root
    if source is None:
        return None
    with _capture_parent_guard(source.parent, is_exclusive=True) as acquired:
        if not acquired:
            raise Conflict("capture generation source retention is busy")
        _retain_generation_source_locked(transition, before_source_write)
    return generation_source_hold(transition)


def _apply_generation_reset_locked(transition: CaptureGenerationTransition, source_hold: SourceHold | None) -> None:
    root, state = transition.runtime_root, transition.state
    if source_hold != generation_source_hold(transition):
        raise Unavailable("generation reset requires exact separate source retention")
    current_transition = load_capture_generation_transition(
        root, legacy_source=transition.source_root, owner=state["owner"]
    )
    if current_transition != transition:
        raise Unavailable("generation transition local context changed")
    require_json_size(state, max_bytes=STATE_BYTES, record_type="capture_generation_transition")
    path = root / GENERATION_FILE
    if not is_path_present(path):
        atomic_replace(path, state)
    else:
        DurableIO().sync_directory(root, "capture_generation_intent")
    io = DurableIO()
    io.delete(root / COMPLETION_FILE)
    io.delete(root / SOURCE_RELEASE_FILE)
    if read_json_limited(root / CAPTURE_FILE, max_bytes=CAPTURE_BYTES) != transition.after:
        atomic_replace(root / CAPTURE_FILE, transition.after)
    else:
        io.sync_directory(root, "capture_generation_checkpoint")


def apply_generation_reset(transition: CaptureGenerationTransition, source_hold: SourceHold | None) -> None:
    """Retain local intent and reset coverage using local-only I/O."""
    with _capture_parent_guard(transition.runtime_root.parent, is_exclusive=True) as acquired:
        if not acquired:
            raise Conflict("capture generation local reset is busy")
        _apply_generation_reset_locked(transition, source_hold)


def _normalize_generation_source_locked(
    transition: CaptureGenerationTransition, before_source_write: Callable[[], None]
) -> None:
    source = transition.source_root
    if source is None:
        return
    if source.resolve() != source or not source.is_dir():
        raise Unavailable("generation transition lost its canonical source directory")
    hold = generation_source_hold(transition).to_dict()
    path = source / CAPTURE_FILE
    stored = read_json_limited(path, max_bytes=CAPTURE_BYTES)
    expected = {**hold, "phase": "generation_pending", "admission": transition.state["owner"]}
    if stored not in (hold, expected):
        raise Unavailable("generation transition source belongs to another capture")
    before_source_write()
    atomic_replace(path, hold)


def normalize_generation_source(
    transition: CaptureGenerationTransition, *, before_source_write: Callable[[], None]
) -> SourceHold | None:
    """Restore ordinary source retention only after the local reset is durable."""
    if transition.source_root is None:
        return None
    with _capture_parent_guard(transition.source_root.parent, is_exclusive=True) as acquired:
        if not acquired:
            raise Conflict("capture generation source normalization is busy")
        _normalize_generation_source_locked(transition, before_source_write)
    return generation_source_hold(transition)


def _finish_generation_reset_locked(transition: CaptureGenerationTransition, source_hold: SourceHold | None) -> None:
    root = transition.runtime_root
    if source_hold != generation_source_hold(transition):
        raise Unavailable("generation completion requires exact separate source normalization")
    if (
        not is_path_present(root / GENERATION_FILE)
        or read_json_limited(root / GENERATION_FILE, max_bytes=STATE_BYTES) != transition.state
        or read_json_limited(root / CAPTURE_FILE, max_bytes=CAPTURE_BYTES) != transition.after
        or is_path_present(root / COMPLETION_FILE)
        or is_path_present(root / SOURCE_RELEASE_FILE)
    ):
        raise Unavailable("capture generation reset is not durably ready")
    # The exact capture and local marker were checked by reset; recheck the
    # marker without constructing or mutating a replacement responsibility store.
    from .responsibility import responsibility_root

    if read_ledger_instance(responsibility_root(root)) != transition.before["instance"]:
        raise Unavailable("generation transition Ledger changed")
    DurableIO().delete(root / GENERATION_FILE)


def finish_generation_reset(transition: CaptureGenerationTransition, source_hold: SourceHold | None) -> None:
    """Clear local transition exclusion only after exact source normalization."""
    with _capture_parent_guard(transition.runtime_root.parent, is_exclusive=True) as acquired:
        if not acquired:
            raise Conflict("capture generation local completion is busy")
        _finish_generation_reset_locked(transition, source_hold)


def _release(proof: dict) -> dict:
    return {
        "format": "qexp-local-source-release-v1",
        "completion_digest": _digest(proof),
        "legacy_source": proof["legacy_source"],
    }


def restart_capture_generation(runtime_root: Path, *, legacy_source: Path | None, owner: dict) -> None:
    """Reset completed coverage under the caller's current admission/ownership fence.

    An intent blocks coverage reuse and destructive cleanup throughout recovery.
    Ledger memberships survive, and retries publish exactly one newer revision.
    This accepts only generation changes of the same persistent binding owner.
    """
    runtime_root = runtime_root.resolve()
    roots = [runtime_root] + ([legacy_source] if legacy_source is not None else [])
    with ExitStack() as stack:
        for parent in sorted({root.parent for root in roots}):
            if not stack.enter_context(_capture_parent_guard(parent, is_exclusive=True)):
                raise Conflict("capture generation transition is busy")
        transition = load_capture_generation_transition(runtime_root, legacy_source=legacy_source, owner=owner)
        _retain_generation_source_locked(transition, lambda: None)
        hold = generation_source_hold(transition)
        _apply_generation_reset_locked(transition, hold)
        _normalize_generation_source_locked(transition, lambda: None)
        _finish_generation_reset_locked(transition, hold)


def _validate(state: dict, root: Path, source: str | None, admission: dict) -> tuple[dict, dict]:
    if (
        not isinstance(state, dict)
        or set(state) != {"format", "owner", "previous_completion", "previous_capture", "source_release"}
        or state["format"] != FORMAT
        or not isinstance(state["owner"], dict)
        or state["owner"] != capture_admission(state["owner"])
        or not is_same_capture_owner(state["owner"], admission)
    ):
        raise Unavailable("invalid capture generation transition")
    proof, before = state["previous_completion"], state["previous_capture"]
    if (
        not isinstance(proof, dict)
        or not isinstance(before, dict)
        or proof.get("runtime_root") != str(root)
        or proof.get("legacy_source") != source
        or proof.get("capture_digest") != _digest(before)
        or before.get("admission") != capture_admission(proof)
        or not is_same_capture_owner(proof, admission)
        or proof.get("registration_generation") == state["owner"]["registration_generation"]
        or (state["source_release"] is not None and state["source_release"] != _release(proof))
    ):
        raise Unavailable("capture generation transition belongs to another binding")
    progress = before.get("progress")
    scope = _process_scope()
    if (
        not isinstance(progress, dict)
        or progress.get("is_sweep_complete") is not True
        or type(progress.get("revision")) is not int
        or progress["revision"] < 1
        or progress.get("host_id") != scope["host_id"]
        or (progress.get("boot_id") == scope["boot_id"] and progress.get("pid_namespace") != scope["pid_namespace"])
    ):
        raise Unavailable("invalid generation transition process scope")
    after = {
        **before,
        # Finish the exact interrupted reset first. If current ownership advanced
        # again, ordinary admission then resets this unfinished sweep once more.
        "admission": state["owner"],
        "progress": {**progress, "revision": progress["revision"] + 1, "is_sweep_complete": False},
    }
    return before, after
