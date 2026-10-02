"""Machine-local process identity and immutable exit observations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..runtime.authority_scan import validate_evidence_path
from ..runtime.paths import machine_project_paths
from ..runtime.records import validate_identifier
from ..runtime.store import read_json_limited
from .context import MachineRuntime, ProjectBinding

PROCESS_IDENTITY_FIELDS = (
    "wrapper_pid",
    "wrapper_start_time_ticks",
    "process_group_id",
    "process_group_start_time_ticks",
)


def binding_signature(binding: ProjectBinding, registry_revision: int) -> tuple[str, ...]:
    return (
        binding.project_id,
        str(binding.shared_root),
        binding.machine_name,
        binding.registration_generation or "",
        binding.runtime_instance_id or "",
        binding.runtime_root or "",
        str(registry_revision),
    )


def binding_prefix(binding: ProjectBinding) -> tuple[str, ...]:
    """Return the binding ownership fields that should survive registry revisions."""
    return (
        binding.project_id,
        str(binding.shared_root),
        binding.machine_name,
        binding.registration_generation or "",
        binding.runtime_instance_id or "",
        binding.runtime_root or "",
    )


def read_process_manifest(
    runtime: MachineRuntime,
    binding: ProjectBinding,
    path: Path,
    signature: tuple[str, ...],
) -> dict[str, Any] | None:
    """Read and validate one machine-local process manifest."""
    paths = machine_project_paths(runtime.root, binding.project_id)
    try:
        if not validate_evidence_path(path, paths["root"]):
            return None
        envelope = read_json_limited(path, max_bytes=65_536, record_type="process_manifest")
    except FileNotFoundError:
        return None
    process = envelope.get("process")
    if not isinstance(process, dict) or path.stem != process.get("attempt_id"):
        return None
    task_id = process.get("task_id")
    attempt_id = process.get("attempt_id")
    if not isinstance(task_id, str) or not isinstance(attempt_id, str):
        return None
    try:
        validate_identifier(task_id, "process task_id")
        validate_identifier(attempt_id, "process attempt_id")
    except ValueError:
        return None
    prefix = f"{task_id}-attempt-"
    if not attempt_id.startswith(prefix):
        return None
    suffix = attempt_id[len(prefix) :]
    if not suffix.isascii() or not suffix.isdigit() or suffix.startswith("0"):
        return None
    attempt_number = int(suffix)
    if attempt_id != f"{task_id}-attempt-{attempt_number}":
        return None
    fencing_token = process.get("fencing_token")
    if (
        type(fencing_token) is not int
        or fencing_token < 1
        or process.get("machine_name") != binding.machine_name
        or process.get("observed_state") != "running"
    ):
        return None
    reservation_id = process.get("reservation_id")
    if reservation_id is not None:
        try:
            validate_identifier(reservation_id, "process reservation_id")
        except ValueError:
            return None
    process_identity: dict[str, int | None] = {}
    for field in PROCESS_IDENTITY_FIELDS:
        value = process.get(field)
        positive = field in {"wrapper_pid", "process_group_id"}
        if value is not None and (type(value) is not int or value < (1 if positive else 0)):
            return None
        process_identity[field] = value
    identity = (
        task_id,
        attempt_id,
        str(attempt_number),
        str(fencing_token),
        reservation_id or "",
        *("" if process_identity[field] is None else str(process_identity[field]) for field in PROCESS_IDENTITY_FIELDS),
    )
    return {
        "binding": binding,
        "binding_signature": signature,
        "intent_signature": identity,
        "due_key": (*binding_prefix(binding), *identity),
        "path": path,
        "parameters": {
            "task_id": task_id,
            "attempt_id": attempt_id,
            "attempt_number": attempt_number,
            "fencing_token": fencing_token,
            "reservation_id": reservation_id,
            "process_identity": process_identity,
        },
        "process": process,
    }


def read_exit_code(runtime: MachineRuntime, binding: ProjectBinding, entry: dict[str, Any]) -> int | None:
    """Read a bounded exit observation for one validated local process entry."""
    try:
        parameters = entry["parameters"]
        attempt_id = parameters["attempt_id"]
        task_id = parameters["task_id"]
        paths = machine_project_paths(runtime.root, binding.project_id)
        path = paths["observations"] / f"{attempt_id}.json"
        if not validate_evidence_path(path, paths["root"]):
            return None
        envelope = read_json_limited(path, max_bytes=65_536, record_type="exit_observation")
        observation = envelope.get("exit_observation")
        if not isinstance(observation, dict):
            return None
        if observation.get("protocol_version", 1) != 1:
            return None
        if observation.get("attempt_id") != attempt_id or observation.get("task_id") not in {None, task_id}:
            return None
        code = observation.get("observed_exit_code")
        if type(code) is not int:
            return None
        return code
    except (OSError, ValueError, TypeError, KeyError, RuntimeError):
        return None
