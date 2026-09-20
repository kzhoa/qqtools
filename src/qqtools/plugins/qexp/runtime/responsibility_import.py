"""Durable coverage for evidence copied from a still-writable legacy inbox."""

from contextlib import ExitStack
from pathlib import Path

from .authority_scan import validate_evidence_path
from .locks import exclusive
from .paths import local_paths
from .records import validate_identifier
from .responsibility import responsibility_root
from .responsibility_capture import capture_cleanup_guard, cleanup_runtime_guard
from .responsibility_cleanup import evidence_write_guard
from .responsibility_store import Conflict, DurableIO, Ledger
from .store import atomic_replace, read_json

RECORD_KEYS = {
    "processes": "process",
    "registrations": "process_registration",
    "observations": "exit_observation",
    "launch_intents": "launch_intent",
    "termination_decisions": "termination_decision",
    "wrappers": "wrapper",
    "authority_diagnostics": "authority_diagnostic",
}
RUNNER_INBOX = ("launch_intents", "registrations", "observations")


def refresh_legacy_inbox(ledger: Ledger, runtime_root: Path, identity: str) -> int:
    """Copy three direct runner inbox records without retiring their source.

    Membership supplies the source and remains required throughout refresh.
    Pending capture allows copies, but its active observation scope excludes them.
    Exact copies need no writes: retained source and membership still own replay.
    """
    if ledger.root.resolve() != responsibility_root(runtime_root).resolve():
        raise ValueError("legacy inbox ledger does not belong to this runtime")
    validate_identifier(identity, "attempt_id")
    entry = ledger.find(identity)
    if entry is None or "legacy_source" not in entry or entry.get("cleanup_receipt") is not None:
        return 0
    source_root = Path(entry["legacy_source"])
    if source_root.resolve() == runtime_root.resolve():
        raise Conflict("legacy source cannot be the destination runtime")
    with ExitStack() as stack:
        for root in sorted({runtime_root.resolve(), source_root.resolve()}):
            stack.enter_context(cleanup_runtime_guard(root))
        with evidence_write_guard(runtime_root, identity) as acquired:
            if not acquired:
                raise Conflict(f"legacy evidence is busy: {identity}")
            current = ledger.find(identity)
            if (
                current is None
                or current.get("legacy_source") != str(source_root)
                or current.get("cleanup_receipt") is not None
            ):
                return 0
            copied = 0
            for name in RUNNER_INBOX:
                source = local_paths(source_root)[name] / f"{identity}.json"
                destination = local_paths(runtime_root)[name] / source.name
                if not validate_evidence_path(source, source_root):
                    continue
                value = read_json(source)
                record = value.get(RECORD_KEYS[name])
                if not isinstance(record, dict) or record.get("attempt_id", identity) != identity:
                    raise Conflict("legacy inbox identity does not match its path")
                payload = recovery_locator(identity, record)
                destination_value = (
                    read_json(destination) if validate_evidence_path(destination, runtime_root) else None
                )
                if destination_value is not None and destination_value != value:
                    raise Conflict(f"legacy evidence conflicts during retained refresh: {destination}")
                ledger.capture_source(identity, payload, source_root)
                if destination_value is None:
                    atomic_replace(destination, value)
                    copied += 1
            return copied


def recovery_locator(attempt_id: str, record: dict) -> dict:
    """Capture known fields without inventing a locator for unclassified evidence."""
    task_id = record.get("task_id")
    if task_id is not None:
        validate_identifier(task_id, "task_id")
    candidate, separator, suffix = attempt_id.rpartition("-attempt-")
    number = record.get("attempt_number")
    if number is not None and (type(number) is not int or number < 1):
        raise ValueError("legacy evidence Attempt number must be a positive integer")
    if separator and candidate and suffix.isascii() and suffix.isdecimal():
        parsed = int(suffix)
        if parsed > 0 and str(parsed) == suffix:
            if task_id is not None and task_id != candidate:
                raise Conflict("legacy evidence Task identity does not match its Attempt")
            if number is not None and number != parsed:
                raise Conflict("legacy evidence Attempt number does not match its identity")
            task_id, number = validate_identifier(candidate, "task_id"), parsed
    return {"task_id": task_id, "attempt_number": number}


def move_legacy_record(
    runtime_root: Path,
    source_root: Path,
    name: str,
    source: Path,
    destination: Path,
    *,
    is_destination_authoritative: bool,
) -> None:
    """Capture the writer, durably copy, then durably unlink its source record."""
    if source_root.resolve() == runtime_root.resolve():
        raise ValueError("legacy source cannot be the destination runtime")
    attempt_id = source.parent.name if name == "termination_decisions" else source.stem
    validate_identifier(attempt_id, "attempt_id")
    with capture_cleanup_guard(runtime_root, source_root) as can_cleanup:
        if not can_cleanup:
            raise Conflict("legacy evidence import is blocked by writer capture")
        with evidence_write_guard(runtime_root, attempt_id) as acquired:
            if not acquired:
                raise RuntimeError(f"legacy evidence is busy: {attempt_id}")
            if not validate_evidence_path(source, source_root):
                return
            try:
                value = read_json(source)
            except FileNotFoundError:
                # A receipt owner may have completed cleanup since enumeration.
                return
            record = value.get(RECORD_KEYS[name])
            if not isinstance(record, dict) or record.get("attempt_id", attempt_id) != attempt_id:
                raise ValueError("legacy evidence identity does not match its path")
            payload = recovery_locator(attempt_id, record)
            destination_value = read_json(destination) if validate_evidence_path(destination, runtime_root) else None
            if destination_value is not None:
                destination_record = destination_value.get(RECORD_KEYS[name])
                if (
                    not isinstance(destination_record, dict)
                    or destination_record.get("attempt_id", attempt_id) != attempt_id
                ):
                    raise Conflict("destination evidence identity does not match its path")
                destination_payload = recovery_locator(attempt_id, destination_record)
                for key, known in destination_payload.items():
                    if known is not None:
                        if payload[key] not in (None, known):
                            raise Conflict("destination evidence belongs to a different recovery identity")
                        payload[key] = known
            with exclusive(runtime_root / "locks" / "responsibility-initialize.lock"):
                ledger = Ledger.open_or_create(responsibility_root(runtime_root))
            ledger.capture_source(attempt_id, payload, source_root)
            if ledger.lookup(attempt_id).get("cleanup_receipt") is not None:
                # Stored cleanup proof already covers this immutable source. Leave
                # it to the cleanup owner instead of recreating destination evidence.
                return
            if destination_value is not None:
                if not is_destination_authoritative and destination_value != value:
                    raise Conflict(f"legacy evidence conflicts during migration: {destination}")
                # Retry may observe a rename whose directory barrier was interrupted.
                DurableIO().sync_directory(destination.parent, "legacy_destination")
            else:
                atomic_replace(destination, value)
            # atomic_replace syncs the leaf directory, not newly created ancestors.
            # Existing names on a retry may also follow an interrupted mkdir barrier.
            # Persist the whole path (including the membership root's ancestors)
            # before discarding the only independently durable source copy.
            for directory in destination.resolve().parents[1:]:
                DurableIO().sync_directory(directory, "legacy_destination_ancestor")
            DurableIO().delete(source)
