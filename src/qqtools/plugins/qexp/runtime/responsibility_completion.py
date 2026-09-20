"""Durable retained-capture completion, independent of supervision readiness."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .authority_scan import is_path_present
from .responsibility_backfill import LANES, STATE_BYTES, ResponsibilityBackfill
from .responsibility_capture import (
    CAPTURE_BYTES,
    CAPTURE_FILE,
    CAPTURE_FORMAT,
    GENERATION_FILE,
    SOURCE_CAPTURE_FORMAT,
    capture_admission,
)
from .responsibility_process_capture import SCAN_FORMAT, _process_scope
from .responsibility_store import DurableIO, Ledger, Unavailable
from .store import atomic_replace, read_json_limited, require_json_size

COMPLETION_FILE = "responsibility-capture-complete.json"
COMPLETION_FORMAT = "qexp-local-capture-complete-v1"
COMPLETION_BYTES = 16384
SOURCE_RELEASE_FILE = "responsibility-source-release.json"


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_capture_completion(runtime_root: Path, *, should_sync: bool = True) -> dict | None:
    """Validate fixed metadata; absence is distinct from a broken completion proof.

    No history or membership enumeration is needed. A completed capture survives
    a same-host reboot; it does not transfer ownership to another host/namespace.
    Callers still validate their binding before using this as discovery coverage.
    """
    from .responsibility import responsibility_root

    runtime_root = runtime_root.resolve()
    if is_path_present(runtime_root / GENERATION_FILE):
        raise Unavailable("capture generation transition is pending")
    path = runtime_root / COMPLETION_FILE
    if not is_path_present(path):
        return None
    proof = read_json_limited(path, max_bytes=COMPLETION_BYTES)
    capture = read_json_limited(runtime_root / CAPTURE_FILE, max_bytes=CAPTURE_BYTES)
    evidence = read_json_limited(runtime_root / "responsibility-capture-backfill.json", max_bytes=STATE_BYTES)
    scope = capture.get("progress", {})
    source = capture.get("legacy_source")
    expected_lanes = len(LANES) * (2 if source is not None else 1)
    if (
        proof.get("format") != COMPLETION_FORMAT
        or proof.get("runtime_root") != str(runtime_root)
        or proof.get("instance") != Ledger(responsibility_root(runtime_root)).instance
        or capture.get("format") != CAPTURE_FORMAT
        or capture.get("phase") != "pending"
        or capture.get("runtime_root") != str(runtime_root)
        or capture.get("instance") != proof["instance"]
        or capture.get("pending") != []
        or capture.get("admission") != capture_admission(proof)
        or not isinstance(scope, dict)
        or scope.get("format") != SCAN_FORMAT
        or scope.get("is_sweep_complete") is not True
        or evidence.get("instance") != proof["instance"]
        or evidence.get("writer_capture_id") != capture.get("capture_id")
        or evidence.get("writer_sweep_revision") != scope.get("revision")
        or evidence.get("lane") != expected_lanes
        or evidence.get("pending") != []
        or evidence.get("at_end") is not False
        or proof.get("capture_digest") != _digest(capture)
        or proof.get("evidence_digest") != _digest(evidence)
        or proof.get("legacy_source") != source
        or proof.get("shared_root") != scope.get("shared_root")
        or proof.get("machine_name") != scope.get("machine_name")
        or any(
            not isinstance(proof.get(key), str) or not proof[key]
            for key in ("project_id", "owner_root", "owner_instance", "registration_generation")
        )
    ):
        raise Unavailable("invalid retained capture completion")
    if source is not None and (
        not isinstance(source, str)
        or not Path(source).is_absolute()
        or str(Path(source).resolve()) != source
        or source == str(runtime_root)
    ):
        raise Unavailable("invalid retained capture source")
    current_scope = _process_scope()
    if scope.get("host_id") != current_scope["host_id"] or (
        scope.get("boot_id") == current_scope["boot_id"]
        and scope.get("pid_namespace") != current_scope["pid_namespace"]
    ):
        raise Unavailable("capture completion belongs to another host or PID namespace")
    # Finish a visible rename whose publisher did not finish its directory sync.
    if should_sync:
        DurableIO().sync_directory(runtime_root, "capture_completion")
    return proof


def publish_capture_completion(backfill: ResponsibilityBackfill, *, owner: dict) -> dict:
    """Commit after the caller revalidates scheduler, binding and root admission."""
    with backfill.completed_capture():
        capture = read_json_limited(backfill.runtime_root / CAPTURE_FILE, max_bytes=CAPTURE_BYTES)
        if capture.get("admission") != capture_admission(owner):
            raise Unavailable("completion requires this binding's post-admission capture")
        evidence = read_json_limited(backfill.path, max_bytes=STATE_BYTES)
        proof = {
            **owner,
            "format": COMPLETION_FORMAT,
            "runtime_root": str(backfill.runtime_root),
            "instance": capture["instance"],
            "legacy_source": capture["legacy_source"],
            "capture_digest": _digest(capture),
            "evidence_digest": _digest(evidence),
        }
        require_json_size(proof, max_bytes=COMPLETION_BYTES, record_type="capture_completion")
        atomic_replace(backfill.runtime_root / COMPLETION_FILE, proof)
        return read_capture_completion(backfill.runtime_root)


def is_source_capture_complete(source: Path, owner: Path) -> bool:
    """Permit the owning target's cleanup only after its exact source capture."""
    proof = read_capture_completion(owner)
    if proof is None or proof["legacy_source"] != str(source):
        return False
    hold = read_json_limited(source / CAPTURE_FILE, max_bytes=CAPTURE_BYTES)
    capture = read_json_limited(owner / CAPTURE_FILE, max_bytes=CAPTURE_BYTES)
    return hold == {
        "format": SOURCE_CAPTURE_FORMAT,
        "runtime_root": str(source),
        "target_root": str(owner),
        "instance": proof["instance"],
        "capture_id": capture["capture_id"],
        "phase": "pending",
    }


def is_source_released(runtime_root: Path, proof: dict) -> bool:
    """Inspect release without treating a missing source hold as a valid receipt."""
    source = proof["legacy_source"]
    if source is None:
        return True
    if is_path_present(Path(source) / CAPTURE_FILE):
        return False
    expected = {
        "format": "qexp-local-source-release-v1",
        "completion_digest": _digest(proof),
        "legacy_source": source,
    }
    if read_json_limited(runtime_root / SOURCE_RELEASE_FILE, max_bytes=COMPLETION_BYTES) != expected:
        raise Unavailable("invalid retained source release proof")
    return True


def release_completed_source(runtime_root: Path) -> bool:
    """Release source retention only after all captured responsibilities retire."""
    from contextlib import ExitStack

    from .responsibility import responsibility_root
    from .responsibility_capture import _capture_parent_guard

    runtime_root = runtime_root.resolve()
    proof = read_capture_completion(runtime_root)
    if proof is None:
        return False
    if proof["legacy_source"] is None:
        return True
    source = Path(proof["legacy_source"])
    with ExitStack() as stack:
        for parent in sorted({runtime_root.parent, source.parent}):
            if not stack.enter_context(_capture_parent_guard(parent, is_exclusive=True)):
                return False
        if read_capture_completion(runtime_root) != proof:
            raise Unavailable("capture completion changed before source release")
        release_path = runtime_root / SOURCE_RELEASE_FILE
        expected_release = {
            "format": "qexp-local-source-release-v1",
            "completion_digest": _digest(proof),
            "legacy_source": str(source),
        }
        if is_path_present(release_path):
            if read_json_limited(release_path, max_bytes=COMPLETION_BYTES) != expected_release:
                raise Unavailable("invalid retained source release proof")
            DurableIO().sync_directory(runtime_root, "capture_source_release_intent")
        else:
            if Ledger(responsibility_root(runtime_root)).has_members():
                if not is_path_present(source / CAPTURE_FILE):
                    raise Unavailable("source retention disappeared before release")
                return False
            # Retire source ownership before removing its hold. Later current
            # launches may add target-only members without reopening this source.
            atomic_replace(release_path, expected_release)
        if not is_path_present(source / CAPTURE_FILE):
            if source.is_dir():
                DurableIO().sync_directory(source, "capture_source_release")
            return True
        if not is_source_capture_complete(source, runtime_root):
            raise Unavailable("retained source does not match completed capture")
        DurableIO().delete(source / CAPTURE_FILE)
        return True
