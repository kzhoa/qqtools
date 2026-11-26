"""Read-only drain evidence for the departing Project binding."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from ..config_types import RootConfig
from ..runtime.authority_scan import iter_evidence_files, validate_evidence_path
from ..runtime.paths import attempt_path, local_paths, machine_project_paths, shared_paths, task_path
from ..runtime.records import AttemptRecord
from ..runtime.responsibility import responsibility_root
from ..runtime.responsibility_capture import CAPTURE_FILE, GENERATION_FILE, _capture_parent_guard
from ..runtime.responsibility_cleanup import FLAT_EVIDENCE, captured_writers_are_quiescent, writers_are_quiescent
from ..runtime.responsibility_completion import (
    COMPLETION_FILE,
    SOURCE_RELEASE_FILE,
    is_source_released,
    read_capture_completion,
)
from ..runtime.responsibility_import import RECORD_KEYS, recovery_locator
from ..runtime.responsibility_process_capture import has_live_runner
from ..runtime.responsibility_store import BUCKETS, DurableIO, Ledger
from ..runtime.store import read_json
from ..runtime.takeover_registration import RegistrationWitness, TakeoverRegistration
from ..runtime.tasks import load_task
from ..runtime.terminal_evidence import load_settled_terminal_attempt
from .takeover_recovery_preview import _describe_record, _local_directory_present

DrainObservation = Literal["drained", "drain_pending", "drain_unavailable"]


def _capture_is_pending(root: Path, transition: TakeoverRegistration, departing: RegistrationWitness) -> bool:
    generation = root / GENERATION_FILE
    capture = root / CAPTURE_FILE
    if validate_evidence_path(generation, root):
        read_json(generation)
        return True
    if not validate_evidence_path(root / COMPLETION_FILE, root):
        if validate_evidence_path(capture, root):
            read_json(capture)
            return True
        # Unfenced legacy sources must settle their existing recovery protocol.
        # Target-only bindings can use the same local process observation
        # independently of a root-wide upgrade involving the expired name.
        return validate_evidence_path(root / "migration.json", root)

    for path in (
        capture,
        root / "responsibility-capture-backfill.json",
        responsibility_root(root) / "marker",
    ):
        if not validate_evidence_path(path, root):
            raise ValueError("Departing partition capture evidence is incomplete.")
    proof = read_capture_completion(root, should_sync=False)
    expected = {
        "project_id": transition.project_id,
        "shared_root": str(transition.shared_root),
        "machine_name": departing.machine_name,
        "owner_root": departing.runtime_root,
        "owner_instance": departing.runtime_id,
        "registration_generation": departing.generation,
    }
    if proof is None or any(proof[key] != value for key, value in expected.items()):
        raise ValueError("Departing partition capture evidence belongs to a different scope.")
    migration_path = root / "migration.json"
    source = proof["legacy_source"]
    if validate_evidence_path(migration_path, root):
        migration = read_json(migration_path)["migration"]
        if (
            not isinstance(source, str)
            or not Path(source).is_absolute()
            or Path(source).resolve() != Path(source)
            or Path(source) == root
            or any(
                migration.get(key) != value
                for key, value in {
                    "state": "active",
                    "project_id": transition.project_id,
                    "shared_root": str(transition.shared_root),
                    "machine_name": departing.machine_name,
                    "legacy_runtime_root": source,
                }.items()
            )
        ):
            raise ValueError("Departing Project migration does not match the capture.")
        if not validate_evidence_path(root / SOURCE_RELEASE_FILE, root):
            return True
        source_root = Path(source)
        with _capture_parent_guard(source_root.parent, is_exclusive=True) as acquired:
            if not acquired or not is_source_released(root, proof):
                return True
            # The local release receipt precedes the source unlink. Repair its
            # durability barrier before retirement discards the replay source.
            directory = source_root if _local_directory_present(source_root) else source_root.parent
            DurableIO().sync_directory(directory, "takeover_capture_source_release")
        return False
    if source is not None:
        raise ValueError("Departing Project capture refers to an untracked legacy source.")
    return False


def _has_reservation_record(root: Path, reservation_id: str) -> bool:
    if (
        not isinstance(reservation_id, str)
        or not reservation_id
        or reservation_id in {".", ".."}
        or Path(reservation_id).name != reservation_id
    ):
        raise ValueError("reservation_id must be a nonempty filename component.")
    paths = local_paths(root)
    # The ordinary lookup can create lane locks. A raw record stays pending
    # until the normal reservation owner clears it.
    present = False
    for name in ("active", "provisional", "cpu_active", "cpu_provisional"):
        path = paths[name] / f"{reservation_id}.json"
        if validate_evidence_path(path, root):
            if read_json(path)["reservation"]["reservation_id"] != reservation_id:
                raise ValueError("Reservation record does not match its filename.")
            present = True
    return present


def _attempt_is_settled(cfg: RootConfig, machine_root: Path, task_id: str, identity: str, number: int) -> bool:
    shared_root = cfg.shared_root
    attempt_file = attempt_path(shared_root, task_id, number)
    if not all(
        validate_evidence_path(path, shared_root)
        for path in (
            shared_paths(shared_root)["schema"] / "version.json",
            task_path(shared_root, task_id),
            attempt_file,
        )
    ):
        raise ValueError("Departing Attempt has missing shared evidence.")
    task = load_task(cfg, task_id)
    attempt = AttemptRecord.from_dict(read_json(attempt_file))
    if task.task_id != task_id or not _matches_attempt(attempt, task_id, identity, number, cfg.machine_name):
        raise ValueError("Departing Attempt has a mismatched shared locator.")
    current_number = task.attempt_control["current_attempt_number"]
    next_number = task.attempt_control["next_attempt_number"]
    if (
        type(current_number) is not int
        or type(next_number) is not int
        or not number <= current_number < next_number
        or (current_number == number and task.attempt_control["current_attempt_id"] not in (None, identity))
    ):
        raise ValueError("Departing Attempt does not match the Task's Attempt history.")

    settled = load_settled_terminal_attempt(cfg, task_id, identity, attempt_number=number)
    if settled is None:
        return False
    if not _matches_attempt(settled, task_id, identity, number, cfg.machine_name):
        raise ValueError("Settled Attempt does not match the departing locator.")
    requested_at = settled.termination.get("requested_at")
    acknowledged_at = settled.termination.get("acknowledged_at")
    if any(
        value is not None and (not isinstance(value, str) or not value) for value in (requested_at, acknowledged_at)
    ):
        raise ValueError("Departing Attempt has malformed termination evidence.")
    if requested_at is not None and acknowledged_at is None:
        return False
    return settled.reservation_id is None or not _has_reservation_record(machine_root, settled.reservation_id)


def _matches_attempt(attempt: AttemptRecord, task_id: str, identity: str, number: int, machine_name: str) -> bool:
    return type(attempt.attempt_number) is int and (
        attempt.task_id,
        attempt.attempt_id,
        attempt.attempt_number,
        attempt.machine_name,
    ) == (task_id, identity, number, machine_name)


def _flat_locators(root: Path, paths: dict[str, Path], machine_name: str) -> dict[str, list[dict]]:
    locators: dict[str, list[dict]] = {}
    for name in FLAT_EVIDENCE:
        directory = paths[name]
        for path in iter_evidence_files(directory, recursive=True):
            if path.parent != directory or not validate_evidence_path(path, root):
                raise ValueError("Unexpected departing recovery evidence path.")
            record = read_json(path)[RECORD_KEYS[name]]
            if (
                not isinstance(record, dict)
                or record.get("attempt_id") != path.stem
                or record.get("machine_name", machine_name) != machine_name
            ):
                raise ValueError("Departing recovery evidence does not match its locator.")
            locators.setdefault(path.stem, []).append(recovery_locator(path.stem, record))
    return locators


def _has_unsettled_records(root: Path, paths: dict[str, Path]) -> bool:
    present = False
    for name in ("events", "termination_decisions"):
        for path in iter_evidence_files(paths[name], recursive=True):
            if not validate_evidence_path(path, root):
                raise ValueError("Departing publication or termination evidence changed during observation.")
            read_json(path)
            present = True
    return present


def _observe_members(ledger: Ledger, cfg: RootConfig, machine_root: Path, locators: dict[str, list[dict]]) -> bool:
    counts = [ledger.observe_count(bucket) for bucket in range(BUCKETS)]
    identities: set[str] = set()
    pending = False
    for bucket, count in enumerate(counts):
        cursor = None
        while True:
            entries, cursor, observed_count = ledger.observe_page(bucket, cursor)
            if observed_count != count:
                raise ValueError("Departing recovery ledger membership changed during observation.")
            for entry in entries:
                record = _describe_record(cfg.runtime_root, entry)
                identity = record["identity"]
                if identity in identities:
                    raise ValueError("Departing recovery ledger contains a duplicate identity.")
                identities.add(identity)
                evidence_locators = locators.pop(identity, [])
                if record["kind"] == "control_plane_diagnostic":
                    continue
                if record["kind"] != "attempt" or entry.get("legacy_source") is not None:
                    raise ValueError("Departing recovery evidence cannot identify a local Attempt.")
                if any(
                    value is not None and value != record[key]
                    for locator in evidence_locators
                    for key, value in locator.items()
                ):
                    raise ValueError("Departing recovery evidence has a mismatched Attempt locator.")
                task_id, number = record["task_id"], record["attempt_number"]
                if not captured_writers_are_quiescent(entry) or not writers_are_quiescent(
                    cfg.runtime_root, task_id, identity
                ):
                    pending = True
                if not _attempt_is_settled(cfg, machine_root, task_id, identity, number):
                    pending = True
            if cursor is None:
                break
    if locators or counts != [ledger.observe_count(bucket) for bucket in range(BUCKETS)]:
        raise ValueError("Departing local recovery evidence does not match observed ledger membership.")
    return pending


def observe_takeover_drain(transition: TakeoverRegistration) -> DrainObservation:
    """Observe departed work after admission and Project I/O handoffs are fenced."""
    departing = transition.departing
    if departing is None:
        return "drained"
    machine_root = transition.successor_runtime_root
    paths = machine_project_paths(machine_root, transition.project_id)
    root = paths["root"]
    try:
        if not all(_local_directory_present(path) for path in reversed(root.parents)):
            return "drain_unavailable"
        shared_root = transition.shared_root
        cfg = RootConfig.from_canonical_paths(shared_root, shared_root.parent, departing.machine_name, root)
        if has_live_runner(cfg):
            return "drain_pending"
        if not _local_directory_present(root):
            return "drained"
        if _capture_is_pending(root, transition, departing):
            return "drain_pending"
        ledger_root = responsibility_root(root)
        locators = _flat_locators(root, paths, departing.machine_name)
        pending = _has_unsettled_records(root, paths)
        if _local_directory_present(ledger_root):
            if not all(_local_directory_present(path) for path in (*reversed(shared_root.parents), shared_root)):
                return "drain_unavailable"
            ledger = Ledger(ledger_root, is_observation=True)
            pending = _observe_members(ledger, cfg, machine_root, locators) or pending
        elif locators:
            return "drain_unavailable"
        return "drain_pending" if pending else "drained"
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError):
        return "drain_unavailable"
