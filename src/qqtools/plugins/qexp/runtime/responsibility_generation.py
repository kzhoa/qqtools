"""Replayable recapture after the same persistent owner reacquires its registration."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

from .authority_scan import is_path_present
from .responsibility_capture import (
    CAPTURE_BYTES,
    CAPTURE_FILE,
    GENERATION_FILE,
    SOURCE_CAPTURE_FORMAT,
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
from .responsibility_store import Conflict, DurableIO, Ledger, Unavailable
from .store import atomic_replace, read_json_limited, require_json_size

FORMAT = "qexp-capture-generation-transition-v1"
STATE_BYTES = CAPTURE_BYTES + 2 * COMPLETION_BYTES


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
    from .responsibility import responsibility_root

    runtime_root = runtime_root.resolve()
    admission = capture_admission(owner)
    source = str(legacy_source) if legacy_source is not None else None
    roots = [runtime_root] + ([legacy_source] if legacy_source is not None else [])
    with ExitStack() as stack:
        for parent in sorted({root.parent for root in roots}):
            if not stack.enter_context(_capture_parent_guard(parent, is_exclusive=True)):
                raise Conflict("capture generation transition is busy")
        path = runtime_root / GENERATION_FILE
        if is_path_present(path):
            state = read_json_limited(path, max_bytes=STATE_BYTES)
        else:
            proof = read_capture_completion(runtime_root)
            if proof is None:
                raise Unavailable("generation transition requires completed prior capture")
            release_path = runtime_root / SOURCE_RELEASE_FILE
            release = (
                read_json_limited(release_path, max_bytes=COMPLETION_BYTES) if is_path_present(release_path) else None
            )
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
        hold = None
        if legacy_source is not None:
            hold = {
                "format": SOURCE_CAPTURE_FORMAT,
                "runtime_root": source,
                "target_root": str(runtime_root),
                "instance": before["instance"],
                "capture_id": before["capture_id"],
                "phase": "pending",
            }
            hold_path = legacy_source / CAPTURE_FILE
            if is_path_present(hold_path):
                stored_hold = read_json_limited(hold_path, max_bytes=CAPTURE_BYTES)
                normal_hold = {key: value for key, value in stored_hold.items() if key != "admission"}
                normal_hold["phase"] = "pending"
                if stored_hold != hold and (
                    normal_hold != hold
                    or stored_hold.get("phase") != "generation_pending"
                    or not isinstance(stored_hold.get("admission"), dict)
                    or stored_hold["admission"] != capture_admission(stored_hold["admission"])
                    or not is_same_capture_owner(stored_hold["admission"], admission)
                ):
                    raise Unavailable("generation transition source belongs to another capture")
            elif state["source_release"] is None or not legacy_source.is_dir():
                raise Unavailable("generation transition lost unreleased source retention")
        current = read_json_limited(runtime_root / CAPTURE_FILE, max_bytes=CAPTURE_BYTES)
        if current not in (before, after) or Ledger(responsibility_root(runtime_root)).instance != before["instance"]:
            raise Unavailable("generation transition checkpoint or Ledger changed")
        for name, expected in (
            (COMPLETION_FILE, state["previous_completion"]),
            (SOURCE_RELEASE_FILE, state["source_release"]),
        ):
            record_path = runtime_root / name
            if is_path_present(record_path) and read_json_limited(record_path, max_bytes=COMPLETION_BYTES) != expected:
                raise Unavailable("generation transition metadata changed")
        require_json_size(state, max_bytes=STATE_BYTES, record_type="capture_generation_transition")
        # Source-only cleanup cannot see the target intent. A distinct hold also
        # prevents old completed-capture release from undoing this first write.
        if hold is not None:
            atomic_replace(hold_path, {**hold, "phase": "generation_pending", "admission": state["owner"]})
        if not is_path_present(path):
            atomic_replace(path, state)
        else:
            DurableIO().sync_directory(runtime_root, "capture_generation_intent")
        io = DurableIO()
        io.delete(runtime_root / COMPLETION_FILE)
        io.delete(runtime_root / SOURCE_RELEASE_FILE)
        if current != after:
            atomic_replace(runtime_root / CAPTURE_FILE, after)
        else:
            io.sync_directory(runtime_root, "capture_generation_checkpoint")
        if hold is not None:
            # Old completion is durably gone. The normal pending hold can now
            # bind to the reset checkpoint while the target intent still guards it.
            atomic_replace(hold_path, hold)
        io.delete(path)


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
