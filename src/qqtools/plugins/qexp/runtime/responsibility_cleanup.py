"""Durable local evidence cleanup after terminal and writer-quiescence proof."""

from __future__ import annotations

import errno
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from ..infrastructure.host import host_instance_id
from .authority_scan import EvidenceScan
from .locks import exclusive
from .paths import attempt_control_lock_path, local_paths
from .records import validate_identifier
from .responsibility_capture import capture_cleanup_guard
from .responsibility_store import Conflict, DurableIO, Ledger
from .store import read_json

CLEANUP_FORMAT = "qexp-local-cleanup-v1"
EVIDENCE_RECORDS = (
    ("processes", "process"),
    ("registrations", "process_registration"),
    ("launch_intents", "launch_intent"),
)
FLAT_EVIDENCE = ("processes", "registrations", "observations", "launch_intents", "wrappers", "authority_diagnostics")


@contextmanager
def evidence_write_guard(runtime_root: Path, attempt_id: str) -> Iterator[bool]:
    """Exclude cleanup from manifest creation without waiting on background I/O."""
    validate_identifier(attempt_id, "attempt_id")
    with exclusive(runtime_root / "locks" / f"evidence-{attempt_id}.lock", blocking=False) as acquired:
        yield acquired


@dataclass(frozen=True)
class CleanupRequest:
    """A proven terminal obligation; a generation denotes replay of stored proof."""

    identity: str
    payload: dict
    receipt: dict
    generation: int | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.identity, "attempt_id")
        task_id = validate_identifier(self.payload.get("task_id"), "task_id")
        number = self.payload.get("attempt_number")
        has_operation_proof = self.receipt.get("basis") == "task_cleanup"
        if has_operation_proof:
            validate_identifier(self.receipt.get("operation_id"), "operation_id")
        if not (has_operation_proof and number is None) and (type(number) is not int or number < 1):
            raise ValueError("cleanup requires an exact Attempt number")
        if (
            self.receipt.get("format") != CLEANUP_FORMAT
            or self.receipt.get("task_id") != task_id
            or self.receipt.get("attempt_id") != self.identity
            or self.receipt.get("basis") not in {"terminal_attempt", "completed_task_cleanup", "task_cleanup"}
        ):
            raise ValueError("cleanup receipt identity or format mismatch")
        if self.generation is not None and (type(self.generation) is not int or self.generation < 1):
            raise ValueError("cleanup replay requires a positive generation")

    @classmethod
    def from_entry(cls, entry: dict) -> CleanupRequest | None:
        receipt = entry.get("cleanup_receipt")
        if receipt is None:
            return None
        if entry.get("stage") != "maintenance" or not isinstance(receipt, dict):
            raise ValueError("invalid persisted cleanup handoff")
        return cls(entry["identity"], entry["payload"], receipt, entry["generation"])


def _wrapper_has_exited(pid: object, expected_start: object) -> bool:
    if type(pid) is not int or pid <= 0 or type(expected_start) is not int or expected_start < 0:
        return False
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[19]) != expected_start or fields[0] in {"Z", "X"}
    except FileNotFoundError:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except OSError:
            pass
        return False
    except (OSError, ValueError, IndexError):
        return False


def _group_has_exited(group: object) -> bool:
    if type(group) is not int or group <= 0:
        return False
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return True
    except OSError:
        pass
    return False


def captured_writers_are_quiescent(entry: dict) -> bool:
    """Lost files cannot erase a previously captured live or unresolved writer."""
    if entry.get("writer_capture_incomplete"):
        return False
    writers = entry.get("captured_writers", [])
    if not writers:
        return True
    try:
        host = host_instance_id()
        if any(writer["host_id"] != host for writer in writers):
            return False
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if str(uuid.UUID(boot)) != boot:
            return False
        # A different boot proves that captured processes cannot write again.
        # A different PID namespace on the same boot does not prove their exit.
        namespace = Path("/proc/self/ns/pid").stat().st_ino
    except (OSError, ValueError, RuntimeError):
        return False
    return all(
        writer["boot_id"] != boot
        or (writer["pid_namespace"] == namespace and _wrapper_has_exited(writer["pid"], writer["start_time_ticks"]))
        for writer in writers
    )


def has_final_observation(runtime_root: Path, task_id: str, attempt_id: str) -> bool:
    """Protocol 1 publishes exit only after its last intent/registration write."""
    try:
        value = read_json(local_paths(runtime_root)["observations"] / f"{attempt_id}.json")["exit_observation"]
    except FileNotFoundError:
        return False
    return (
        isinstance(value, dict)
        and value.get("protocol_version") == 1
        and value.get("task_id") in {None, task_id}
        and value.get("attempt_id") == attempt_id
        and type(value.get("observed_exit_code")) is int
    )


def writers_are_quiescent(runtime_root: Path, task_id: str, attempt_id: str) -> bool:
    """Prove known immutable evidence writers cannot publish another late record.

    Unknown identity or inaccessible process state is unresolved. A terminal Task
    alone does not show that a still-running wrapper has published its last exit.
    """
    paths = local_paths(runtime_root)
    # This legacy namespace has no current writer/identity contract. Capture its
    # format explicitly during migration before granting cleanup permission.
    if (paths["wrappers"] / f"{attempt_id}.json").exists():
        return False
    for name, key in EVIDENCE_RECORDS:
        try:
            value = read_json(paths[name] / f"{attempt_id}.json")[key]
        except FileNotFoundError:
            continue
        if not isinstance(value, dict) or value.get("task_id") != task_id or value.get("attempt_id") != attempt_id:
            return False
        pid, start = value.get("wrapper_pid"), value.get("wrapper_start_time_ticks")
        group = value.get("process_group_id")
        has_exited_group = group is not None and _group_has_exited(group)
        if group is not None and not has_exited_group:
            return False
        if pid is None and start is None:
            # Existing protocol-1 recovered manifests may retain only the
            # process group. Its absence plus the runner's final write proves
            # completion; a terminal state label alone is insufficient.
            if (
                value.get("protocol_version") != 1
                or not has_exited_group
                or not has_final_observation(runtime_root, task_id, attempt_id)
            ):
                return False
        elif not _wrapper_has_exited(pid, start):
            return False
    return True


def remove_evidence_slice(
    runtime_root: Path, attempt_id: str, io: DurableIO, *, owner_root: Path | None = None
) -> bool:
    """Delete at most eight decisions plus fixed evidence, syncing before retirement."""
    with capture_cleanup_guard(runtime_root, owner_root=owner_root) as can_cleanup:
        if not can_cleanup:
            return False
        return _remove_evidence_slice_locked(runtime_root, attempt_id, io)


def _remove_evidence_slice_locked(runtime_root: Path, attempt_id: str, io: DurableIO) -> bool:
    validate_identifier(attempt_id, "attempt_id")
    paths = local_paths(runtime_root)
    directory = paths["termination_decisions"] / attempt_id
    scan = EvidenceScan(directory)
    try:
        page = scan.take(8)
        for path in page.paths:
            io.delete(path, should_sync_directory=False)
        if directory.exists():
            io.sync_directory(directory, "cleanup_decisions")
            try:
                directory.rmdir()
            except OSError as exc:
                if exc.errno == errno.ENOTEMPTY:
                    return False
                if exc.errno != errno.ENOENT:
                    raise
    finally:
        scan.close()
    # Also sync on retry after an earlier uncommitted unlink/rmdir disappeared
    # from memory. The stored receipt survives until every deletion is durable.
    if paths["termination_decisions"].exists():
        io.sync_directory(paths["termination_decisions"], "cleanup_decision_directory")
    for name in FLAT_EVIDENCE:
        if paths[name].exists():
            io.delete(paths[name] / f"{attempt_id}.json")
    io.sync_directory(runtime_root, "cleanup_evidence_directories")
    return True


def complete_cleanup(ledger: Ledger, runtime_root: Path, request: CleanupRequest) -> bool:
    """Persist proof before deletion; replay never needs Task truth or launch authority."""
    with capture_cleanup_guard(runtime_root) as can_cleanup:
        if not can_cleanup:
            return False
        with evidence_write_guard(runtime_root, request.identity) as acquired:
            if not acquired:
                return False
            with exclusive(attempt_control_lock_path(runtime_root, request.identity), blocking=False) as acquired:
                if not acquired:
                    return False
                return _complete_cleanup_locked(ledger, runtime_root, request)


def _complete_cleanup_locked(ledger: Ledger, runtime_root: Path, request: CleanupRequest) -> bool:
    if request.generation is None:
        entry = ledger.find(request.identity)
        if entry is None:
            generation = ledger.publish(request.identity, request.payload)
        else:
            generation = ledger.resolve_locator(request.identity, request.payload)
        entry = ledger.lookup(request.identity)
        if not captured_writers_are_quiescent(entry):
            return False
        source = entry.get("legacy_source")
        if source is not None and not writers_are_quiescent(Path(source), request.payload["task_id"], request.identity):
            return False
        generation = ledger.handoff(request.identity, generation, cleanup_receipt=request.receipt)
    else:
        entry = ledger.lookup(request.identity)
        if (
            entry["generation"] != request.generation
            or entry["stage"] != "maintenance"
            or entry.get("cleanup_receipt") != request.receipt
            or entry["payload"] != request.payload
        ):
            raise Conflict("cleanup proof belongs to a different membership generation")
        generation = request.generation
        source = entry.get("legacy_source")
    if source is not None:
        source_root = Path(source)
        if source_root == runtime_root:
            raise Conflict("legacy source cannot be the current evidence root")
        source_paths = local_paths(source_root)
        has_source_evidence = any((source_paths[name] / f"{request.identity}.json").exists() for name in FLAT_EVIDENCE)
        has_source_evidence |= (source_paths["termination_decisions"] / request.identity).exists()
        if source_root.exists():
            if not remove_evidence_slice(source_root, request.identity, ledger.io, owner_root=runtime_root):
                return False
        else:
            ledger.io.sync_directory(source_root.parent, "cleanup_legacy_root")
        if has_source_evidence:
            # At most one populated root is cleaned per slice. Keep local
            # identifying evidence until the source's deletions are durable.
            return False
    if not remove_evidence_slice(runtime_root, request.identity, ledger.io):
        return False
    ledger.retire(request.identity, generation)
    return True
