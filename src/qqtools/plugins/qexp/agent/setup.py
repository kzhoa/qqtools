"""Machine setup, Project enrollment and recoverable identity replacement."""

from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from ..config_types import RootConfig
from ..infrastructure.host import host_instance_id
from ..layout import initialize_shared_root, load_root_config, project_id, validate_root_contract
from ..runtime.authority_scan import iter_evidence_files
from ..runtime.paths import shared_paths
from ..runtime.process_evidence import inspect_local_group_identity, inspect_wrapper_identity
from ..runtime.store import atomic_replace, read_json
from .config import (
    DEFAULT_AGENT_MODE,
    AgentConfig,
    agent_config_payload,
    initialize_agent_config,
    load_agent_config,
    set_agent_config,
    validate_agent_mode,
    validate_agent_name,
)
from .context import MachineRuntime
from .inventory import (
    ProjectInventoryEntry,
    canonical_shared_root,
    find_inventory_entries,
    inventory_status,
    load_inventory,
    save_inventory_locked,
)
from .project_admin import enable_project, register_project, set_project_enabled, unregister_project


class SetupUsageError(ValueError):
    """A setup request contains an invalid user-controlled value."""


class SetupOperationalError(RuntimeError):
    """A recognized machine or Project state blocks setup progress."""


class MachineResetConfirmationRequired(SetupOperationalError):
    """Raised before a replacement mutates state when confirmation is absent."""

    def __init__(self, facts: dict[str, Any]):
        self.facts = facts
        super().__init__("machine identity replacement requires confirmation or --yes.")


def _runtime(value: MachineRuntime | str | Path | None) -> MachineRuntime:
    return value if isinstance(value, MachineRuntime) else MachineRuntime(value)


def _identity_record(runtime: MachineRuntime) -> dict[str, Any] | None:
    path = runtime.paths["identity"]
    if not path.exists():
        return None
    value = read_json(path)
    if not isinstance(value, dict):
        raise SetupOperationalError("machine runtime identity is malformed.")
    record = value.get("machine_runtime")
    if not isinstance(record, dict):
        raise SetupOperationalError("machine runtime identity is malformed.")
    seed = record.get("instance_id")
    effective = record.get("runtime_id")
    if not isinstance(seed, str) or not seed:
        raise SetupOperationalError("machine runtime identity is malformed.")
    if effective is not None and (
        not isinstance(effective, str)
        or len(effective) != 64
        or any(char not in "0123456789abcdef" for char in effective)
    ):
        raise SetupOperationalError("machine runtime identity has an invalid public runtime ID.")
    return record


def _effective_runtime_id(seed: str) -> str:
    if len(seed) == 64 and all(char in "0123456789abcdef" for char in seed):
        return seed
    return hashlib.sha256(f"{seed}\0{host_instance_id()}".encode()).hexdigest()


def _current_runtime_id(runtime: MachineRuntime) -> str | None:
    record = _identity_record(runtime)
    if record is None:
        return None
    effective = record.get("runtime_id")
    return effective if isinstance(effective, str) else _effective_runtime_id(record["instance_id"])


def _read_pid(runtime: MachineRuntime) -> int | None:
    try:
        return int(runtime.paths["pid"].read_text(encoding="utf-8").strip())
    except (FileNotFoundError, OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _active_identity(runtime: MachineRuntime) -> tuple[int, str, int] | None:
    from .helpers import _active_machine_identity

    try:
        return _active_machine_identity(runtime)
    except (OSError, ValueError, TypeError):
        return None


def _execution_evidence_blockers(paths: dict[str, Path], *, prefix: str = "") -> list[str]:
    """Return live or unverifiable runner evidence without mutating it."""
    blockers: list[str] = []
    record_keys = {
        "processes": "process",
        "registrations": "process_registration",
        "launch_intents": "launch_intent",
        "wrappers": None,
    }
    for name, record_key in record_keys.items():
        root = paths.get(name)
        if root is None:
            continue
        try:
            evidence_paths = list(iter_evidence_files(root, recursive=True))
        except OSError:
            blockers.append(f"{prefix}{name}:unavailable")
            continue
        for path in evidence_paths:
            if record_key is None:
                blockers.append(f"{prefix}{name}:{path.stem}:ambiguous")
                continue
            try:
                value = read_json(path)
                record = value.get(record_key) if isinstance(value, dict) else None
            except (OSError, ValueError, TypeError):
                record = None
            if not isinstance(record, dict):
                blockers.append(f"{prefix}{name}:{path.stem}:unavailable")
                continue
            observations = []
            if "process_group_id" in record or "process_group_start_time_ticks" in record:
                observations.append(inspect_local_group_identity(record))
            if "wrapper_pid" in record or "wrapper_start_time_ticks" in record:
                observations.append(inspect_wrapper_identity(record))
            if not observations:
                blockers.append(f"{prefix}{name}:{path.stem}:missing_identity")
                continue
            unresolved = next((item for item in observations if item.state != "absent"), None)
            if unresolved is not None:
                detail = unresolved.state if unresolved.reason is None else unresolved.reason
                blockers.append(f"{prefix}{name}:{path.stem}:{detail}")
    return blockers


def _local_obligations(runtime: MachineRuntime) -> tuple[list[str], bool]:
    """Return durable local obligations and whether local execution is excluded."""
    obligations: list[str] = []
    active = _active_identity(runtime)
    if active is not None:
        obligations.append(f"agent:{active[0]}")
    pid = _read_pid(runtime)
    if pid is not None and _pid_alive(pid) and active is None:
        obligations.append(f"ambiguous_agent_pid:{pid}")

    try:
        _revision, bindings = runtime.load_registry()
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        return [f"registry_unavailable:{exc}"], False
    execution_blockers: list[str] = []
    for binding in bindings:
        try:
            blockers = runtime.binding_blockers(binding)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            blockers = [f"{binding.project_id}:recovery_unavailable:{exc}"]
        obligations.extend(f"{binding.project_id}:{blocker}" for blocker in blockers)
        execution_blockers.extend(
            f"{binding.project_id}:{blocker}"
            for blocker in _execution_evidence_blockers(runtime.project_paths(binding.project_id))
        )
        try:
            legacy_roots = runtime._legacy_evidence_roots(binding)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            execution_blockers.append(f"{binding.project_id}:legacy:execution_evidence_unavailable")
        else:
            if legacy_roots is not None:
                execution_blockers.extend(
                    f"{binding.project_id}:{blocker}"
                    for blocker in _execution_evidence_blockers(legacy_roots[0], prefix="legacy:")
                )
    obligations.extend(item for item in execution_blockers if item not in obligations)
    # A copied runtime can contain local evidence even when its old registry was
    # lost.  Treat readable evidence as an unresolved obligation; detachment
    # preserves it while ordinary replacement rejects it.
    for name in (
        "active",
        "provisional",
        "cpu_active",
        "cpu_provisional",
        "processes",
        "registrations",
        "launch_intents",
    ):
        path = runtime.paths.get(name)
        if path is None:
            continue
        try:
            if next(iter_evidence_files(path, recursive=True), None) is not None:
                marker = f"runtime:{name}"
                if marker not in obligations:
                    obligations.append(marker)
        except OSError:
            obligations.append(f"runtime:{name}:unavailable")
    local_execution_excluded = not active and not (pid is not None and _pid_alive(pid)) and not execution_blockers
    return obligations, local_execution_excluded


def machine_init_facts(runtime: MachineRuntime | str | Path | None) -> dict[str, Any]:
    """Read replacement facts without creating a machine runtime."""
    machine_runtime = _runtime(runtime)
    old_id = _current_runtime_id(machine_runtime)
    config: AgentConfig | None = None
    try:
        config = load_agent_config(machine_runtime, require_initialized=False)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
        config = None
    obligations, execution_excluded = _local_obligations(machine_runtime) if old_id else ([], True)
    return {
        "old_runtime_id": old_id,
        "current_name": config.name if config else None,
        "current_agent_mode": config.agent_mode if config else None,
        "obligations": obligations,
        "local_execution_excluded": execution_excluded,
    }


def _read_replacement(runtime: MachineRuntime) -> dict[str, Any] | None:
    path = runtime.paths["replacement_transaction"]
    if not path.exists():
        return None
    value = read_json(path)
    transaction = value.get("replacement")
    if not isinstance(transaction, dict):
        raise SetupOperationalError("machine replacement transaction is malformed.")
    for key in ("version", "target_name", "new_runtime_id", "policy", "phase", "detach_old_runtime"):
        if key not in transaction:
            raise SetupOperationalError("machine replacement transaction is malformed.")
    if (
        type(transaction["version"]) is not int
        or transaction["version"] != 1
        or transaction["phase"] not in {"staged", "archived", "published"}
    ):
        raise SetupOperationalError("machine replacement transaction is unsupported.")
    try:
        validate_agent_name(transaction["target_name"])
        validate_agent_mode(transaction["policy"])
    except (TypeError, ValueError) as exc:
        raise SetupOperationalError("machine replacement transaction is malformed.") from exc
    new_runtime_id = transaction["new_runtime_id"]
    if (
        not isinstance(new_runtime_id, str)
        or len(new_runtime_id) != 64
        or any(char not in "0123456789abcdef" for char in new_runtime_id)
    ):
        raise SetupOperationalError("machine replacement transaction has an invalid target runtime ID.")
    if type(transaction["detach_old_runtime"]) is not bool:
        raise SetupOperationalError("machine replacement transaction has an invalid reset policy.")
    archive = Path(transaction.get("archive_path", ""))
    try:
        if archive.parent.resolve() != runtime.paths["archives"].resolve():
            raise ValueError
    except (OSError, RuntimeError, ValueError) as exc:
        raise SetupOperationalError("machine replacement transaction has an invalid archive path.") from exc
    return transaction


def pending_machine_replacement(runtime: MachineRuntime) -> dict[str, Any] | None:
    """Return validated facts needed to resume a pending identity replacement."""
    transaction = _read_replacement(runtime)
    return dict(transaction) if transaction is not None else None


def _validate_pending_target(transaction: dict[str, Any], name: str, mode: str | None, detach: bool) -> str:
    if transaction["target_name"] != name:
        raise SetupOperationalError(
            f"machine replacement is pending for target {transaction['target_name']!r}; "
            "retry that target before choosing another name."
        )
    requested = mode if mode is not None else transaction["policy"]
    if requested != transaction["policy"]:
        raise SetupOperationalError("machine replacement is pending with a different agent mode.")
    if transaction.get("detach_old_runtime") and not detach:
        raise SetupOperationalError("the pending replacement requires --detach-old-runtime to resume.")
    return transaction["new_runtime_id"]


def _archive_tree(runtime: MachineRuntime, archive: Path, *, transaction: dict[str, Any] | None = None) -> None:
    """Copy all old local authority state into an immutable sibling archive."""
    staging = archive.parent / f".{archive.name}.staging-{uuid.uuid4().hex}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        excluded = {"archives", "locks", "config.json", "inventory.json"}
        for child in runtime.root.iterdir():
            if child.name in excluded or child == staging:
                continue
            target = staging / child.name
            if child.is_dir():
                shutil.copytree(child, target, symlinks=True)
            elif child.is_file():
                shutil.copy2(child, target)
            else:
                raise SetupOperationalError(f"cannot archive non-regular machine runtime entry: {child}")
        manifest = {
            "archive": {
                "version": 1,
                "old_runtime_id": (
                    transaction.get("old_runtime_id")
                    if transaction and transaction.get("old_runtime_id")
                    else _current_runtime_id(runtime)
                ),
                "new_runtime_id": transaction.get("new_runtime_id") if transaction else None,
                "archive_path": str(archive),
                "policy": transaction.get("policy") if transaction else None,
                "obligations": list(transaction.get("obligations", ())) if transaction else [],
                "created_at": time.time(),
            }
        }
        atomic_replace(staging / "manifest.json", manifest)
        archive.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, archive)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _clear_old_local_state(runtime: MachineRuntime) -> None:
    preserved = {
        "archives",
        "locks",
        "config.json",
        "inventory.json",
        "replacement-transaction.json",
    }
    for child in list(runtime.root.iterdir()):
        if child.name in preserved:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _ensure_machine_dirs(runtime: MachineRuntime) -> None:
    runtime.root.mkdir(parents=True, exist_ok=True)
    if not os.access(runtime.root, os.W_OK | os.X_OK):
        raise SetupOperationalError(f"machine runtime root is not writable: {runtime.root}")
    for name in (
        "locks",
        "agent",
        "provisional",
        "active",
        "released",
        "cpu_provisional",
        "cpu_active",
        "cpu_released",
        "projects",
        "upgrades",
        "diagnostics",
        "archives",
        "generations",
        "scheduler",
    ):
        runtime.paths[name].mkdir(parents=True, exist_ok=True)
    runtime.paths["cpu_policy"].parent.mkdir(parents=True, exist_ok=True)


def initialize_machine(
    runtime: MachineRuntime | str | Path | None,
    name: str,
    *,
    agent_mode: str | None = None,
    detach_old_runtime: bool = False,
    confirmed: bool = False,
    expected_old_runtime_id: str | None = None,
    yes: bool | None = None,
) -> dict[str, Any]:
    """Initialize or replace one machine identity with a staged commit."""
    machine_runtime = _runtime(runtime)
    try:
        target_name = validate_agent_name(name)
        selected_mode = validate_agent_mode(agent_mode) if agent_mode is not None else None
    except ValueError as exc:
        raise SetupUsageError(str(exc)) from exc
    if yes is not None:
        confirmed = confirmed or yes

    with machine_runtime.agent_lifecycle_guard():
        transaction = _read_replacement(machine_runtime)
        if transaction is not None:
            new_id = _validate_pending_target(transaction, target_name, selected_mode, detach_old_runtime)
            selected_mode = transaction["policy"]
        else:
            new_id = None
        old_id = _current_runtime_id(machine_runtime)
        if expected_old_runtime_id is not None and old_id != expected_old_runtime_id:
            raise SetupOperationalError(
                "machine identity changed after confirmation was requested; inspect the current identity and retry."
            )
        if old_id is None and transaction is not None:
            old_id = transaction.get("old_runtime_id")
        reported_old_id = transaction.get("old_runtime_id") if transaction is not None else old_id
        if old_id is None and transaction is None:
            # Empty detachment is intentionally a no-op and is never persisted.
            _ensure_machine_dirs(machine_runtime)
            if selected_mode is None:
                selected_mode = DEFAULT_AGENT_MODE
            seed = uuid.uuid4().hex
            new_id = _effective_runtime_id(seed)
            initialize_agent_config(machine_runtime, target_name, agent_mode=selected_mode)
            atomic_replace(
                machine_runtime.paths["identity"],
                {"machine_runtime": {"instance_id": seed, "runtime_id": new_id, "created_at": time.time()}},
            )
            atomic_replace(
                machine_runtime.paths["current_generation"],
                {"current_generation": {"version": 1, "runtime_id": new_id, "published_at": time.time()}},
            )
            if not machine_runtime.paths["registry"].exists():
                atomic_replace(
                    machine_runtime.paths["registry"], {"registry": {"version": 1, "revision": 0, "bindings": []}}
                )
            if not machine_runtime.paths["inventory"].exists():
                atomic_replace(
                    machine_runtime.paths["inventory"], {"inventory": {"version": 1, "revision": 0, "entries": []}}
                )
            return {
                "action": "initialized",
                "machine_runtime_root": str(machine_runtime.root),
                "agent_name": target_name,
                "agent_mode": selected_mode,
                "old_runtime_id": None,
                "new_runtime_id": new_id,
                "detached": False,
                "archive_path": None,
                "obligations": [],
            }

        facts = machine_init_facts(machine_runtime)
        if transaction is None:
            if not facts["local_execution_excluded"]:
                raise SetupOperationalError(
                    "cannot reinitialize machine while local agent/process ownership is live or ambiguous: "
                    + ", ".join(facts["obligations"])
                )
            if facts["obligations"] and not detach_old_runtime:
                raise SetupOperationalError(
                    "cannot reinitialize machine with unresolved recovery obligations: "
                    + ", ".join(facts["obligations"])
                )
            if not confirmed:
                facts.update({"requested_name": target_name, "requested_agent_mode": selected_mode})
                raise MachineResetConfirmationRequired(facts)
            if selected_mode is None:
                try:
                    current = load_agent_config(machine_runtime)
                except RuntimeError:
                    # A runtime can have an identity without a feature-era
                    # config (for example an interrupted first setup).  The
                    # production default remains daemon in that case.
                    selected_mode = DEFAULT_AGENT_MODE
                else:
                    selected_mode = current.agent_mode
            new_id = _effective_runtime_id(uuid.uuid4().hex)
            archive = machine_runtime.paths["archives"] / f"{old_id}-{new_id}"
            transaction = {
                "version": 1,
                "phase": "staged",
                "old_runtime_id": old_id,
                "new_runtime_id": new_id,
                "target_name": target_name,
                "policy": selected_mode,
                "detach_old_runtime": detach_old_runtime,
                "archive_path": str(archive),
                "obligations": facts["obligations"],
                "created_at": time.time(),
            }
            atomic_replace(machine_runtime.paths["replacement_transaction"], {"replacement": transaction})
        else:
            # Pending transitions only resume after the same target has been
            # confirmed; no second random ID is ever allocated.
            if not confirmed:
                facts.update({"requested_name": target_name, "requested_agent_mode": transaction["policy"]})
                raise MachineResetConfirmationRequired(facts)
            new_id = transaction["new_runtime_id"]
            selected_mode = transaction["policy"]

        archive = Path(transaction["archive_path"])
        if transaction["phase"] == "staged":
            if archive.exists() and not (archive / "manifest.json").is_file():
                if archive.is_dir():
                    shutil.rmtree(archive)
                else:
                    archive.unlink()
            if not archive.exists():
                _archive_tree(machine_runtime, archive, transaction=transaction)
            transaction["phase"] = "archived"
            atomic_replace(machine_runtime.paths["replacement_transaction"], {"replacement": transaction})
        if transaction["phase"] == "archived":
            # Archive must contain the manifest before any current authority is
            # removed.  A failed archive therefore leaves the old identity intact.
            if not (archive / "manifest.json").is_file():
                raise SetupOperationalError(
                    "machine replacement archive is incomplete; current identity remains active."
                )
            _clear_old_local_state(machine_runtime)
            _ensure_machine_dirs(machine_runtime)
            old_revision = 0
            try:
                old_revision = read_json(archive / "registry.json").get("registry", {}).get("revision", 0)
            except (OSError, ValueError, TypeError, KeyError):
                pass
            atomic_replace(
                machine_runtime.paths["registry"],
                {"registry": {"version": 1, "revision": old_revision + 1, "bindings": []}},
            )
            # Keep the reusable inventory, but update global config only after
            # all old effective authority has been isolated.
            initialize_agent_config(machine_runtime, target_name, agent_mode=selected_mode)
            atomic_replace(
                machine_runtime.paths["current_generation"],
                {
                    "current_generation": {
                        "version": 1,
                        "runtime_id": new_id,
                        "old_runtime_id": old_id,
                        "published_at": time.time(),
                    }
                },
            )
            seed = uuid.uuid4().hex
            calculated = _effective_runtime_id(seed)
            if calculated != new_id:
                # Keep the staged public ID stable across retries; this branch is
                # only relevant to an injected host-identity change.
                atomic_replace(
                    machine_runtime.paths["identity"],
                    {
                        "machine_runtime": {
                            "instance_id": seed,
                            "runtime_id": new_id,
                            "created_at": time.time(),
                        }
                    },
                )
            else:
                atomic_replace(
                    machine_runtime.paths["identity"],
                    {"machine_runtime": {"instance_id": seed, "runtime_id": new_id, "created_at": time.time()}},
                )
            transaction["phase"] = "published"
            atomic_replace(machine_runtime.paths["replacement_transaction"], {"replacement": transaction})
        machine_runtime.paths["replacement_transaction"].unlink(missing_ok=True)
        return {
            "action": "reinitialized",
            "machine_runtime_root": str(machine_runtime.root),
            "agent_name": target_name,
            "agent_mode": selected_mode,
            "old_runtime_id": reported_old_id,
            "new_runtime_id": new_id,
            "detached": bool(transaction.get("detach_old_runtime")),
            "archive_path": str(archive),
            "obligations": list(transaction.get("obligations", [])),
        }


def initialize_project(shared_root: str | Path | None = None) -> dict[str, Any]:
    """Create shared Project truth without machine enrollment or activation."""
    root = canonical_shared_root(shared_root or Path.cwd())
    already_exists = root.exists()
    if already_exists:
        cfg = load_root_config(root, "project-init", require_initialized=False)
        validate_root_contract(cfg)
        identity = read_json(shared_paths(root)["project"] / "identity.json").get("project", {})
        stable_id = identity.get("project_id")
        if not isinstance(stable_id, str) or not stable_id:
            raise SetupOperationalError("qexp Project identity is malformed.")
        return {
            "action": "project_already_initialized",
            "project_id": stable_id,
            "shared_root": str(root),
            "created": False,
        }
    temp_runtime = root.parent / f".{root.name}.project-init-runtime"
    cfg = RootConfig(root, root.parent, "project-init", temp_runtime)
    initialize_shared_root(cfg)
    # ``initialize_shared_root`` predates the split and creates only empty
    # machine-local scaffolding. Remove that scaffolding immediately; no
    # machine record, binding, context or activation is published here.
    machine_dir = shared_paths(root)["machines"] / cfg.machine_name
    if machine_dir.exists():
        shutil.rmtree(machine_dir)
    if temp_runtime.exists():
        shutil.rmtree(temp_runtime)
    stable_id = project_id(root)
    return {"action": "project_initialized", "project_id": stable_id, "shared_root": str(root), "created": True}


def _entry_by_identity(
    entries: list[ProjectInventoryEntry], stable_id: str, root: Path
) -> ProjectInventoryEntry | None:
    for entry in entries:
        if entry.project_id == stable_id or entry.shared_root == root:
            return entry
    return None


def _result_for_binding(
    entry: ProjectInventoryEntry,
    binding: Any | None,
    *,
    status: str,
    reason: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "project_id": entry.project_id,
        "shared_root": str(entry.shared_root),
        "enabled": entry.enabled,
        "name_source": entry.name_source,
        "name_override": entry.name_override,
        "status": status,
    }
    if binding is not None:
        result.update(
            {
                "machine_name": binding.machine_name,
                "enabled": binding.enabled,
                "registration_generation": binding.registration_generation,
                "runtime_instance_id": binding.runtime_instance_id,
            }
        )
    elif entry.name_override is not None:
        result["machine_name"] = entry.name_override
    if reason is not None:
        result["reason"] = reason
    return result


def _load_project_identity(root: Path) -> tuple[str, RootConfig]:
    if not root.is_dir():
        raise SetupUsageError(f"Project path is missing or not mounted: {root}")
    identity_path = shared_paths(root)["project"] / "identity.json"
    if not identity_path.is_file():
        raise SetupUsageError(f"Project is not initialized: {root}; run 'qexp project init {root.parent}'.")
    identity = read_json(identity_path).get("project")
    if not isinstance(identity, dict) or identity.get("shared_root") != str(root):
        raise SetupUsageError(f"Project identity is malformed: {identity_path}")
    stable_id = identity.get("project_id")
    if not isinstance(stable_id, str) or not stable_id:
        raise SetupUsageError(f"Project identity has no stable ID: {identity_path}")
    # Root validation is independent of local machine records.
    cfg = load_root_config(root, "project-enrollment", require_initialized=True)
    return stable_id, cfg


def _legacy_project(cfg: RootConfig) -> bool:
    from ..machine_config import has_legacy_agent_metadata

    return has_legacy_agent_metadata(cfg)


def _name_intent(
    runtime: MachineRuntime,
    config: AgentConfig,
    entry: ProjectInventoryEntry | None,
    binding: Any | None,
    *,
    machine_name: str | None,
    name_source: str | None,
) -> tuple[str, str, str | None]:
    if machine_name is not None and name_source == "default":
        raise SetupUsageError("--machine and --name-source default are mutually exclusive.")
    requested_source = "explicit" if machine_name is not None else name_source
    if requested_source is not None and requested_source not in {"default", "explicit"}:
        raise SetupUsageError("--name-source must be default or explicit for registration.")
    if entry is not None and entry.name_source == "unresolved" and requested_source is None:
        raise SetupUsageError("name_source_unresolved")
    source = requested_source or (entry.name_source if entry is not None else "default")
    override = machine_name if machine_name is not None else (entry.name_override if entry is not None else None)
    if source == "default":
        override = None
    if binding is not None:
        if machine_name is not None and machine_name != binding.machine_name:
            raise SetupUsageError(
                f"Project {binding.project_id!r} is already registered as {binding.machine_name!r}; "
                "ordinary registration cannot adopt or rename it."
            )
        effective = binding.machine_name
        # Source confirmation changes only future inventory intent; effective
        # registration name remains frozen.
        return effective, source, override
    if source == "explicit" and override is None:
        effective = config.name
    else:
        effective = override or config.name
    return validate_agent_name(effective), source, override


def register_projects(
    runtime: MachineRuntime | str | Path | None,
    paths: Iterable[str | Path] = (),
    *,
    from_pool: bool = False,
    machine_name: str | None = None,
    name_source: str | None = None,
) -> dict[str, Any]:
    """Enroll explicit Projects or the saved local inventory incrementally."""
    machine_runtime = _runtime(runtime)
    machine_runtime.require_initialized()
    selected_paths = list(paths)
    if from_pool and selected_paths:
        raise SetupUsageError("Project register paths and --from-pool are mutually exclusive.")
    if not from_pool and not selected_paths:
        raise SetupUsageError("project register requires at least one Project path or --from-pool.")
    if machine_name is not None and (from_pool or len(selected_paths) != 1):
        raise SetupUsageError("--machine requires exactly one explicit Project path.")
    if machine_name is not None and name_source == "default":
        raise SetupUsageError("--machine and --name-source default are mutually exclusive.")
    if from_pool and (machine_name is not None or name_source is not None):
        raise SetupUsageError("--from-pool does not accept --machine or --name-source.")
    config = load_agent_config(machine_runtime)
    with machine_runtime.agent_lifecycle_guard():
        with machine_runtime.inventory_guard():
            revision, entries = load_inventory(machine_runtime)
            selection_failures: list[dict[str, Any]] = []
            if from_pool:
                selected_entries = list(entries)
            else:
                selected_entries = []
                for value in selected_paths:
                    root = canonical_shared_root(value)
                    known_entry = next((item for item in entries if item.shared_root == root), None)
                    try:
                        stable_id, cfg = _load_project_identity(root)
                    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                        selection_failures.append(
                            {
                                "project_id": known_entry.project_id if known_entry is not None else None,
                                "shared_root": str(root),
                                "enabled": known_entry.enabled if known_entry is not None else True,
                                "name_source": (
                                    known_entry.name_source
                                    if known_entry is not None
                                    else ("explicit" if machine_name is not None else (name_source or "default"))
                                ),
                                "name_override": (
                                    known_entry.name_override if known_entry is not None else machine_name
                                ),
                                "status": "conflicting",
                                "reason": str(exc),
                            }
                        )
                        continue
                    current = _entry_by_identity(entries, stable_id, root)
                    try:
                        binding = next(
                            (
                                item
                                for item in machine_runtime.load_registry()[1]
                                if item.project_id == stable_id or item.shared_root == root
                            ),
                            None,
                        )
                    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                        binding = None
                    effective, source, override = _name_intent(
                        machine_runtime,
                        config,
                        current,
                        binding,
                        machine_name=machine_name,
                        name_source=name_source,
                    )
                    updated = ProjectInventoryEntry(
                        stable_id, root, current.enabled if current else True, source, override
                    )
                    entries = [updated if item == current else item for item in entries]
                    if current is None:
                        entries.append(updated)
                    selected_entries.append(updated)
            if entries != load_inventory(machine_runtime)[1]:
                revision += 1
                save_inventory_locked(machine_runtime, revision, entries)

            results: list[dict[str, Any]] = list(selection_failures)
            for selected in selected_entries:
                entry = next((item for item in entries if item.project_id == selected.project_id), selected)
                binding = next(
                    (
                        item
                        for item in machine_runtime.load_registry()[1]
                        if item.project_id == entry.project_id or item.shared_root == entry.shared_root
                    ),
                    None,
                )
                if from_pool and entry.name_source == "unresolved":
                    results.append(
                        _result_for_binding(entry, binding, status="conflicting", reason="name_source_unresolved")
                    )
                    continue
                if not entry.shared_root.is_dir():
                    results.append(_result_for_binding(entry, binding, status="inventory_only", reason="missing_mount"))
                    continue
                try:
                    stable_id, cfg = _load_project_identity(entry.shared_root)
                    if _legacy_project(cfg):
                        migration_machine = binding.machine_name if binding is not None else config.name
                        command = shlex.join(
                            [
                                "qexp",
                                "admin",
                                "migrate",
                                "agent",
                                "--project",
                                str(entry.shared_root),
                                "--machine",
                                migration_machine,
                            ]
                        )
                        raise SetupUsageError(f"legacy project metadata requires '{command}'.")
                    effective, source, override = _name_intent(
                        machine_runtime,
                        config,
                        entry,
                        binding,
                        machine_name=machine_name if not from_pool else None,
                        name_source=name_source if not from_pool else None,
                    )
                    if source != entry.name_source or override != entry.name_override:
                        replacement = ProjectInventoryEntry(
                            entry.project_id, entry.shared_root, entry.enabled, source, override
                        )
                        entries = [replacement if item == entry else item for item in entries]
                        entry = replacement
                        revision += 1
                        save_inventory_locked(machine_runtime, revision, entries)
                    registration = register_project(
                        machine_runtime,
                        entry.shared_root,
                        effective,
                        enabled=entry.enabled,
                        adopt_existing=False,
                    )
                    result_status = "registered" if registration.binding.enabled else "disabled"
                    results.append(_result_for_binding(entry, registration.binding, status=result_status))
                except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                    results.append(_result_for_binding(entry, binding, status="conflicting", reason=str(exc)))
            return {
                "revision": revision,
                "runtime_id": machine_runtime.instance_id,
                "agent_name": config.name,
                "projects": results,
            }


def list_projects(runtime: MachineRuntime | str | Path | None) -> dict[str, Any]:
    machine_runtime = _runtime(runtime)
    return inventory_status(machine_runtime)


def _selected_entry(
    runtime: MachineRuntime, selector: str | Path
) -> tuple[int, list[ProjectInventoryEntry], ProjectInventoryEntry]:
    revision, entries = load_inventory(runtime)
    matches = find_inventory_entries(entries, selector)
    if len(matches) != 1:
        if not matches:
            raise SetupUsageError(f"machine Project inventory has no entry for {selector!r}.")
        raise SetupUsageError(f"machine Project inventory selector {selector!r} is ambiguous.")
    return revision, entries, matches[0]


def remove_project(runtime: MachineRuntime | str | Path | None, selector: str | Path) -> dict[str, Any]:
    """Remove one local enrollment entry, preserving shared Project truth."""
    machine_runtime = _runtime(runtime)
    machine_runtime.require_initialized()
    with machine_runtime.agent_lifecycle_guard():
        with machine_runtime.inventory_guard():
            revision, entries, entry = _selected_entry(machine_runtime, selector)
            if machine_runtime.paths["replacement_transaction"].exists():
                raise SetupOperationalError("Project removal is blocked by a pending machine replacement.")
            if machine_runtime.paths["registration_transaction"].exists():
                raise SetupOperationalError("Project removal is blocked by pending registration recovery.")
            binding = next(
                (
                    item
                    for item in machine_runtime.load_registry()[1]
                    if item.project_id == entry.project_id or item.shared_root == entry.shared_root
                ),
                None,
            )
            if binding is not None:
                unregister_project(machine_runtime, binding.project_id)
            entries = [item for item in entries if item != entry]
            save_inventory_locked(machine_runtime, revision + 1, entries)
            return {
                "action": "project_removed",
                "project_id": entry.project_id,
                "shared_root": str(entry.shared_root),
                "local_only": True,
                "inventory_revision": revision + 1,
            }


def set_project_enablement(
    runtime: MachineRuntime | str | Path | None,
    selector: str | Path,
    enabled: bool,
) -> dict[str, Any]:
    machine_runtime = _runtime(runtime)
    machine_runtime.require_initialized()
    with machine_runtime.agent_lifecycle_guard():
        with machine_runtime.inventory_guard():
            revision, entries, entry = _selected_entry(machine_runtime, selector)
            binding = next(
                (
                    item
                    for item in machine_runtime.load_registry()[1]
                    if item.project_id == entry.project_id or item.shared_root == entry.shared_root
                ),
                None,
            )
            if binding is None:
                raise SetupUsageError(f"Project {entry.project_id!r} is not registered.")
            updated_binding = (
                enable_project(machine_runtime, binding.project_id)
                if enabled
                else set_project_enabled(machine_runtime, binding.project_id, False)
            )
            updated_entry = ProjectInventoryEntry(
                entry.project_id,
                entry.shared_root,
                enabled,
                entry.name_source,
                entry.name_override,
            )
            save_inventory_locked(
                machine_runtime,
                revision + 1,
                [updated_entry if item == entry else item for item in entries],
            )
            result = _result_for_binding(
                updated_entry,
                updated_binding,
                status="registered" if enabled else "disabled",
            )
            result["action"] = "project_enabled" if enabled else "project_disabled"
            return result


def get_agent_config(runtime: MachineRuntime | str | Path | None) -> dict[str, Any]:
    return agent_config_payload(_runtime(runtime))


__all__ = [
    "MachineResetConfirmationRequired",
    "SetupOperationalError",
    "SetupUsageError",
    "get_agent_config",
    "initialize_machine",
    "initialize_project",
    "list_projects",
    "machine_init_facts",
    "register_projects",
    "remove_project",
    "set_agent_config",
    "set_project_enablement",
]
