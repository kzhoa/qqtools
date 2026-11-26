"""Shared registration transaction for one exact Project takeover."""

from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal

from qqtools.version import __version__

from ..config_types import RootConfig
from ..layout import SCHEMA_VERSION, load_machine_registration, save_machine_record, save_machine_registration
from ..lease import lease_expiry, load_lease_policy, parse_utc
from ..machine_config import MACHINE_AGENT_RUNTIME
from .authority_scan import validate_evidence_path
from .locks import exclusive, machine_lock, schema_writer_lock
from .paths import lock_path, machine_path, machine_project_paths, machine_registration_path, shared_paths
from .protocol_compatibility import REGISTRATION_REPLACEMENT_CAPABILITY
from .records import utc_now
from .registration_authority import registration_state
from .registration_protocol import (
    MAX_REGISTRATION_RECORD_BYTES,
    REGISTRATION_PROTOCOL,
    RESERVATION_VERSION,
    TARGET_REGISTRATION_VERSION,
    _read_protocol_record,
    _validate_reservation,
    reservation_path,
    validate_registration,
)
from .responsibility_store import DurableIO
from .store import atomic_replace, check_mutation_fence, fenced_mutations, require_json_size


class TakeoverRegistrationConflict(RuntimeError):
    """A required Project capability, registration owner, or reservation has changed."""


def _validate_component(value: str, field: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value == "."
        or any(token in value for token in ("/", "\\", "..", "\x00"))
    ):
        raise ValueError(f"{field} must be a safe path component.")


def _validate_id(value: str, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string.")


def _validate_absolute_path(value: str | Path, field: str) -> None:
    path = Path(value)
    if path.anchor != "/" or ".." in path.parts or "\x00" in str(value) or str(path) != str(value):
        raise ValueError(f"{field} must be a canonical absolute path.")


@dataclass(frozen=True, slots=True)
class RegistrationWitness:
    """The exact registration fields that authorize one takeover attempt."""

    machine_name: str
    generation: str
    runtime_id: str
    runtime_root: str
    eligibility_expires_at: str
    state: str

    def __post_init__(self) -> None:
        _validate_component(self.machine_name, "machine_name")
        _validate_id(self.generation, "generation")
        _validate_id(self.runtime_id, "runtime_id")
        _validate_id(self.runtime_root, "runtime_root")
        _validate_absolute_path(self.runtime_root, "runtime_root")
        _validate_id(self.eligibility_expires_at, "eligibility_expires_at")
        if parse_utc(self.eligibility_expires_at).utcoffset() is None:
            raise ValueError("eligibility_expires_at must include a time zone.")
        if self.state not in ("eligible", "superseded"):
            raise ValueError("state must be 'eligible' or 'superseded'.")

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> RegistrationWitness:
        """Capture the witness fields from a validated version-3 registration."""
        return cls(
            machine_name=record["machine_name"],
            generation=record["generation"],
            runtime_id=record["runtime_instance_id"],
            runtime_root=record["runtime_root"],
            eligibility_expires_at=record["eligibility_expires_at"],
            state=record["state"],
        )

    def to_dict(self) -> dict[str, str]:
        """Serialize the witness for an intent or a typed Project I/O request."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> RegistrationWitness:
        """Validate and reconstruct a serialized registration witness."""
        if not isinstance(value, dict) or value.keys() != {field.name for field in fields(cls)}:
            raise ValueError("registration witness has an invalid field set.")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class TakeoverRegistration:
    """The replayable shared inputs owned by one takeover transaction."""

    project_id: str
    shared_root: Path
    transaction_id: str
    successor_runtime_id: str
    successor_runtime_root: Path
    successor_generation: str
    target: RegistrationWitness
    departing: RegistrationWitness | None

    def __post_init__(self) -> None:
        _validate_component(self.project_id, "project_id")
        _validate_id(self.transaction_id, "transaction_id")
        _validate_id(self.successor_runtime_id, "successor_runtime_id")
        _validate_id(self.successor_generation, "successor_generation")
        if not isinstance(self.shared_root, Path) or self.shared_root.name != ".qexp":
            raise ValueError("shared_root must be a path to a '.qexp' directory.")
        _validate_absolute_path(self.shared_root, "shared_root")
        if not isinstance(self.successor_runtime_root, Path):
            raise ValueError("successor_runtime_root must be a path.")
        _validate_absolute_path(self.successor_runtime_root, "successor_runtime_root")
        if not isinstance(self.target, RegistrationWitness):
            raise ValueError("target must be a registration witness.")
        if self.successor_generation == self.target.generation:
            raise ValueError("successor_generation must differ from the target generation.")
        if self.departing is not None:
            if not isinstance(self.departing, RegistrationWitness):
                raise ValueError("departing must be a registration witness.")
            if self.departing.machine_name == self.target.machine_name:
                raise ValueError("departing and target machine names must differ.")
            if self.departing.generation == self.successor_generation:
                raise ValueError("successor_generation must differ from the departing generation.")
            if (self.departing.runtime_id, self.departing.runtime_root) != (
                self.successor_runtime_id,
                str(self.successor_runtime_root),
            ):
                raise ValueError("the departing registration must belong to the successor runtime.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the exact shared transaction for durable replay."""
        return {
            "project_id": self.project_id,
            "shared_root": str(self.shared_root),
            "transaction_id": self.transaction_id,
            "successor_runtime_id": self.successor_runtime_id,
            "successor_runtime_root": str(self.successor_runtime_root),
            "successor_generation": self.successor_generation,
            "target": self.target.to_dict(),
            "departing": self.departing.to_dict() if self.departing is not None else None,
        }

    @classmethod
    def from_dict(cls, value: object) -> TakeoverRegistration:
        """Validate and reconstruct the exact shared transaction."""
        if not isinstance(value, dict) or value.keys() != {field.name for field in fields(cls)}:
            raise ValueError("takeover registration has an invalid field set.")
        for field in ("shared_root", "successor_runtime_root"):
            if not isinstance(value[field], str):
                raise ValueError(f"{field} must be a path string.")
            _validate_absolute_path(value[field], field)
        return cls(
            project_id=value["project_id"],
            shared_root=Path(value["shared_root"]),
            transaction_id=value["transaction_id"],
            successor_runtime_id=value["successor_runtime_id"],
            successor_runtime_root=Path(value["successor_runtime_root"]),
            successor_generation=value["successor_generation"],
            target=RegistrationWitness.from_dict(value["target"]),
            departing=RegistrationWitness.from_dict(value["departing"]) if value["departing"] is not None else None,
        )


def _validate_shared_project(transition: TakeoverRegistration) -> None:
    paths = shared_paths(transition.shared_root)
    identity = _read_protocol_record(paths["project"] / "identity.json")
    project = identity.get("project") if identity is not None else None
    if not isinstance(project, dict) or any(
        not isinstance(project.get(field), str) or not project[field] for field in ("project_id", "shared_root")
    ):
        raise RuntimeError("Project identity is missing or malformed during takeover.")
    if (project["project_id"], project["shared_root"]) != (
        transition.project_id,
        str(transition.shared_root),
    ):
        raise TakeoverRegistrationConflict("Project identity changed during takeover.")

    schema_document = _read_protocol_record(paths["schema"] / "version.json")
    schema = schema_document.get("schema") if schema_document is not None else {}
    if not isinstance(schema, dict) or type(schema.get("version")) is not int or schema["version"] != SCHEMA_VERSION:
        raise TakeoverRegistrationConflict("Project schema version is unsupported during takeover.")
    capabilities = schema.get("required_capabilities", [])
    if not isinstance(capabilities, list) or not all(isinstance(item, str) for item in capabilities):
        raise RuntimeError("Project schema capability state is malformed during takeover.")
    if REGISTRATION_REPLACEMENT_CAPABILITY not in capabilities:
        raise TakeoverRegistrationConflict("Project registration replacement capability is inactive.")


def _directory_identity(path: Path) -> tuple[int, int] | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISDIR(metadata.st_mode):
        raise TakeoverRegistrationConflict(f"Project directory is redirected or unavailable: {path}")
    return metadata.st_dev, metadata.st_ino


class _RegistrationScope:
    """Verify that mutation remains within the Project directories we locked."""

    def __init__(self, root: Path, configs: list[RootConfig], reservation_directory: Path) -> None:
        self.root = root
        self.reservation_directory = reservation_directory
        paths = shared_paths(root)
        required = (
            root,
            paths["locks"],
            paths["locks"] / "machines",
            paths["project"],
            paths["schema"],
            paths["machines"],
            *(paths["machines"] / cfg.machine_name for cfg in configs),
        )
        self.directories = {}
        if root.resolve(strict=True) != root:
            raise TakeoverRegistrationConflict("Project root is redirected during takeover.")
        for path in required:
            identity = _directory_identity(path)
            if identity is None:
                raise TakeoverRegistrationConflict(f"Project directory is missing: {path}")
            self.directories[path] = identity
        self.directories[reservation_directory] = _directory_identity(reservation_directory)
        self.lock_paths = (
            lock_path(root, "schema"),
            paths["locks"] / "registrations.lock",
            *(lock_path(root, "machines", cfg.machine_name) for cfg in configs),
        )
        self.locks = {path: self._lock_identity(path) for path in self.lock_paths}

    def _lock_identity(self, path: Path) -> tuple[int, int] | None:
        if not validate_evidence_path(path, self.root):
            return None
        metadata = path.lstat()
        return metadata.st_dev, metadata.st_ino

    def record_acquired_locks(self) -> None:
        for path, expected in self.locks.items():
            current = self._lock_identity(path)
            if current is None or (expected is not None and expected != current):
                raise TakeoverRegistrationConflict("Project registration lock identity changed.")
            self.locks[path] = current

    def check_current(self) -> None:
        if self.root.resolve(strict=True) != self.root or any(
            _directory_identity(path) != identity for path, identity in self.directories.items()
        ):
            raise TakeoverRegistrationConflict("Project directories changed during takeover.")
        if any(self._lock_identity(path) != identity for path, identity in self.locks.items()):
            raise TakeoverRegistrationConflict("Project registration lock identity changed.")

    def prepare_reservations(self, before_mutation: Callable[[], None]) -> None:
        if self.directories[self.reservation_directory] is None:
            before_mutation()
            self.check_current()
            self.reservation_directory.mkdir()
            self.directories[self.reservation_directory] = _directory_identity(self.reservation_directory)

    def mutation_fence(self, before_mutation: Callable[[], None]) -> Callable[[], None]:
        def verify() -> None:
            before_mutation()
            self.check_current()

        return verify


def _require_mutation_fence(before_mutation: Callable[[], None]) -> None:
    if not callable(before_mutation):
        raise ValueError("Project takeover writes require a current mutation fence.")


@contextmanager
def _registration_locks(
    transition: TakeoverRegistration,
) -> Iterator[tuple[RootConfig, RootConfig | None, _RegistrationScope]]:
    root = transition.shared_root
    partition = machine_project_paths(transition.successor_runtime_root, transition.project_id)["root"]
    target_cfg = RootConfig.from_canonical_paths(root, root.parent, transition.target.machine_name, partition)
    departing_cfg = (
        RootConfig.from_canonical_paths(root, root.parent, transition.departing.machine_name, partition)
        if transition.departing is not None
        else None
    )
    configs = sorted(
        (cfg for cfg in (target_cfg, departing_cfg) if cfg is not None),
        key=lambda cfg: cfg.machine_name,
    )
    scope = _RegistrationScope(root, configs, reservation_path(target_cfg).parent)
    _validate_shared_project(transition)
    with (
        schema_writer_lock(target_cfg),
        exclusive(shared_paths(root)["locks"] / "registrations.lock"),
        ExitStack() as stack,
    ):
        for cfg in configs:
            stack.enter_context(machine_lock(root, cfg.machine_name))
        scope.record_acquired_locks()
        scope.check_current()
        _validate_shared_project(transition)
        yield target_cfg, departing_cfg, scope


def _reservation_document(transition: TakeoverRegistration) -> dict[str, Any]:
    predecessor = transition.departing or transition.target
    witness = json.dumps(transition.to_dict(), sort_keys=True, separators=(",", ":")).encode("ascii")
    return {
        "registration_reservation": {
            "version": RESERVATION_VERSION,
            "transaction_id": transition.transaction_id,
            "project_id": transition.project_id,
            "shared_root": str(transition.shared_root),
            "runtime_id": transition.successor_runtime_id,
            "runtime_root": str(transition.successor_runtime_root),
            "old_machine_name": predecessor.machine_name,
            "old_generation": predecessor.generation,
            "requested_machine_name": transition.target.machine_name,
            "target_generation": transition.successor_generation,
            "takeover_witness_digest": hashlib.sha256(witness).hexdigest(),
        }
    }


def _read_reservation(cfg: RootConfig) -> dict[str, Any] | None:
    document = _read_protocol_record(reservation_path(cfg))
    if document is not None:
        _validate_reservation(document, cfg)
    return document


def _read_registration(cfg: RootConfig, project_id: str) -> dict[str, Any] | None:
    document = load_machine_registration(cfg)
    if document is not None:
        validate_registration(document.get("registration"), project_id, cfg)
    return document


def _require_original_target(record: dict[str, Any] | None, transition: TakeoverRegistration) -> None:
    if record is None or RegistrationWitness.from_record(record) != transition.target:
        raise TakeoverRegistrationConflict("Target registration changed during takeover.")
    if registration_state(record) == "eligible":
        raise TakeoverRegistrationConflict("Target registration still has write eligibility.")


def _is_successor(record: dict[str, Any], transition: TakeoverRegistration) -> bool:
    return (
        record["generation"] == transition.successor_generation
        and record["runtime_instance_id"] == transition.successor_runtime_id
        and record["runtime_root"] == str(transition.successor_runtime_root)
        and record["state"] == "eligible"
    )


def _require_departing_registration(
    cfg: RootConfig,
    transition: TakeoverRegistration,
    departing: RegistrationWitness,
    *,
    has_committed: bool,
) -> dict[str, Any]:
    document = _read_registration(cfg, transition.project_id)
    if document is None:
        raise TakeoverRegistrationConflict("Departing registration disappeared during takeover.")
    record = document["registration"]
    current = RegistrationWitness.from_record(record)
    if (
        current.generation != departing.generation
        or current.runtime_id != departing.runtime_id
        or current.runtime_root != departing.runtime_root
        or departing.state != "eligible"
        or not (
            current.state == "eligible"
            or (
                has_committed
                and current.state == "superseded"
                and record.get("superseded_by_generation") == transition.successor_generation
            )
        )
    ):
        raise TakeoverRegistrationConflict("Departing registration changed during takeover.")
    return document


def _require_protocol_size(document: dict[str, Any], record_type: str) -> None:
    require_json_size(document, max_bytes=MAX_REGISTRATION_RECORD_BYTES, record_type=record_type)


def reserve_takeover_registration(transition: TakeoverRegistration, *, before_mutation: Callable[[], None]) -> None:
    """Reserve the exact ineligible target for this transaction, or repair its barrier."""
    _require_mutation_fence(before_mutation)
    expected = _reservation_document(transition)
    with _registration_locks(transition) as (target_cfg, departing_cfg, scope):
        current = _read_reservation(target_cfg)
        if current is not None and current != expected:
            raise TakeoverRegistrationConflict("Target name is reserved by another transaction.")
        document = _read_registration(target_cfg, transition.project_id)
        record = document["registration"] if document is not None else None
        _require_original_target(record, transition)
        if transition.departing is not None and departing_cfg is not None:
            if _read_reservation(departing_cfg) is not None:
                raise TakeoverRegistrationConflict("Departing name is reserved by another registration transaction.")
            _require_departing_registration(departing_cfg, transition, transition.departing, has_committed=False)
        path = reservation_path(target_cfg)
        if current is None:
            _require_protocol_size(expected, "registration_reservation")
            scope.prepare_reservations(before_mutation)
            with fenced_mutations(transition.shared_root, scope.mutation_fence(before_mutation)):
                atomic_replace(path, expected)
        durable_io = DurableIO()
        durable_io.sync_directory(path.parent, "takeover_registration_reservation")
        durable_io.sync_directory(path.parent.parent, "takeover_registration_reservation_parent")


def _successor_document(
    transition: TakeoverRegistration, cfg: RootConfig, predecessor: dict[str, Any]
) -> dict[str, Any]:
    policy = load_lease_policy(cfg)
    now = utc_now()
    record = {
        "version": TARGET_REGISTRATION_VERSION,
        "project_id": transition.project_id,
        "shared_root": str(transition.shared_root),
        "machine_name": transition.target.machine_name,
        "generation": transition.successor_generation,
        "protocol_version": TARGET_REGISTRATION_VERSION,
        "registration_protocol": REGISTRATION_PROTOCOL,
        "recovery_protocol": predecessor.get("recovery_protocol"),
        "client_version": __version__,
        "runtime_instance_id": transition.successor_runtime_id,
        "runtime_root": str(transition.successor_runtime_root),
        "state": "eligible",
        "eligibility_expires_at": lease_expiry(policy),
        "created_at": now,
        "updated_at": now,
    }
    validate_registration(record, transition.project_id, cfg)
    document = {"registration": record}
    _require_protocol_size(document, "registration")
    return document


def _target_machine_document(
    current: dict[str, Any] | None, transition: TakeoverRegistration, cfg: RootConfig
) -> dict[str, Any]:
    identity = {
        "machine_name": cfg.machine_name,
        "project_id": transition.project_id,
        "shared_root": str(transition.shared_root),
        "agent_runtime": MACHINE_AGENT_RUNTIME,
    }
    if current is None:
        current = {}
        machine = {**identity, "agent_mode": "on_demand"}
    else:
        machine = current.get("machine")
        if not isinstance(machine, dict):
            raise RuntimeError("Target machine metadata is malformed during takeover.")
        if any(machine.get(key) != expected for key, expected in identity.items()):
            raise TakeoverRegistrationConflict("Target machine metadata changed during takeover.")
    document = {**current, "machine": {**machine, "runtime_root": str(cfg.runtime_root)}}
    _require_protocol_size(document, "machine")
    return document


def _departing_supersession(
    cfg: RootConfig,
    transition: TakeoverRegistration,
    departing: RegistrationWitness,
    *,
    has_committed: bool,
) -> dict[str, Any] | None:
    document = _require_departing_registration(cfg, transition, departing, has_committed=has_committed)
    record = document["registration"]
    if record["state"] == "superseded":
        return None
    document = {
        **document,
        "registration": {
            **record,
            "state": "superseded",
            "superseded_by_generation": transition.successor_generation,
            "updated_at": utc_now(),
        },
    }
    validate_registration(document["registration"], transition.project_id, cfg)
    _require_protocol_size(document, "registration")
    return document


def commit_takeover_registration(
    transition: TakeoverRegistration,
    *,
    before_mutation: Callable[[], None],
    is_replay_only: bool = False,
) -> None:
    """Grant the preallocated target generation and finish its shared metadata."""
    _require_mutation_fence(before_mutation)
    with _registration_locks(transition) as (target_cfg, departing_cfg, scope):
        if _read_reservation(target_cfg) != _reservation_document(transition):
            raise TakeoverRegistrationConflict("Exact takeover registration reservation is unavailable.")
        document = _read_registration(target_cfg, transition.project_id)
        record = document["registration"] if document is not None else None
        has_committed = record is not None and _is_successor(record, transition)
        if is_replay_only and not has_committed:
            raise TakeoverRegistrationConflict("Committed takeover successor no longer matches shared registration.")
        root = transition.shared_root
        target_directory = machine_registration_path(root, target_cfg.machine_name).parent
        durable_io = DurableIO()
        if has_committed:
            # Visibility after an interrupted rename does not prove durability.
            durable_io.sync_directory(target_directory, "takeover_registration_grant")
        else:
            _require_original_target(record, transition)
        current_machine = _read_protocol_record(machine_path(root, target_cfg.machine_name))
        target_machine = _target_machine_document(current_machine, transition, target_cfg)
        retiring = (
            _departing_supersession(departing_cfg, transition, transition.departing, has_committed=has_committed)
            if departing_cfg is not None and transition.departing is not None
            else None
        )
        if retiring is not None and _read_reservation(departing_cfg) is not None:
            raise TakeoverRegistrationConflict("Departing name is reserved by another registration transaction.")
        with fenced_mutations(root, scope.mutation_fence(before_mutation)):
            if not has_committed:
                assert record is not None
                successor = _successor_document(transition, target_cfg, record)
                # The departing/metadata reads and the temporary-file sync can
                # outlast a wall-clock correction. Recheck at grant publication.
                with fenced_mutations(root, lambda: _require_original_target(record, transition)):
                    save_machine_registration(target_cfg, successor)
            if target_machine != current_machine:
                save_machine_record(target_cfg, target_machine)
            if departing_cfg is not None:
                if retiring is not None:
                    save_machine_registration(departing_cfg, retiring)
                else:
                    durable_io.sync_directory(
                        machine_registration_path(root, departing_cfg.machine_name).parent,
                        "takeover_registration_departing",
                    )


def _release_locked_reservation(
    transition: TakeoverRegistration,
    target_cfg: RootConfig,
    scope: _RegistrationScope,
    before_mutation: Callable[[], None],
) -> bool:
    path = reservation_path(target_cfg)
    current = _read_reservation(target_cfg)
    if current is None:
        try:
            DurableIO().sync_directory(path.parent, "takeover_registration_release")
        except FileNotFoundError:
            pass
        return False
    if current != _reservation_document(transition):
        return False
    with fenced_mutations(transition.shared_root, scope.mutation_fence(before_mutation)):
        check_mutation_fence(path)
        path.unlink()
    DurableIO().sync_directory(path.parent, "takeover_registration_release")
    return True


def release_takeover_registration(transition: TakeoverRegistration, *, before_mutation: Callable[[], None]) -> bool:
    """Remove only this transaction's reservation and repair a prior unlink barrier."""
    _require_mutation_fence(before_mutation)
    with _registration_locks(transition) as (target_cfg, _departing_cfg, scope):
        return _release_locked_reservation(transition, target_cfg, scope, before_mutation)


def abort_takeover_registration(
    transition: TakeoverRegistration, *, before_mutation: Callable[[], None]
) -> Literal["committed", "departing_current", "departing_changed"]:
    """Prove the grant or release this reservation and check the departing registration.

    A committed result preserves the reservation so commit replay can finish.
    """
    _require_mutation_fence(before_mutation)
    with _registration_locks(transition) as (target_cfg, departing_cfg, scope):
        document = _read_registration(target_cfg, transition.project_id)
        if document is not None:
            record = document["registration"]
            if _is_successor(record, transition):
                return "committed"
            # A different target must still be a valid takeover witness.
            RegistrationWitness.from_record(record)

        outcome: Literal["departing_current", "departing_changed"] = "departing_current"
        if transition.departing is not None and departing_cfg is not None:
            try:
                _require_departing_registration(departing_cfg, transition, transition.departing, has_committed=False)
            except TakeoverRegistrationConflict:
                outcome = "departing_changed"
            if _read_reservation(departing_cfg) is not None:
                outcome = "departing_changed"
        _release_locked_reservation(transition, target_cfg, scope, before_mutation)
        return outcome


def observe_takeover_registration(
    transition: TakeoverRegistration,
) -> Literal["committed", "unchanged", "changed"]:
    """Classify the shared target identity under the registration fence."""
    with _registration_locks(transition) as (target_cfg, _departing_cfg, _scope):
        document = _read_registration(target_cfg, transition.project_id)
        if document is None:
            return "changed"
        record = document["registration"]
        if _is_successor(record, transition):
            return "committed"
        if RegistrationWitness.from_record(record) == transition.target:
            return "unchanged"
        return "changed"


@contextmanager
def takeover_successor_guard(
    transition: TakeoverRegistration, *, before_mutation: Callable[[], None]
) -> Iterator[bool]:
    """Protect approved post-grant writes against concurrent registration replacement."""
    _require_mutation_fence(before_mutation)
    with _registration_locks(transition) as (target_cfg, _departing_cfg, scope):
        document = _read_registration(target_cfg, transition.project_id)
        if document is None or not _is_successor(document["registration"], transition):
            yield False
            return
        with fenced_mutations(transition.shared_root, scope.mutation_fence(before_mutation)):
            yield True
