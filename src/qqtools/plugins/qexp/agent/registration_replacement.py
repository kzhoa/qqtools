"""Durable Project registration-name replacement state machine."""

from __future__ import annotations

import json
import shlex
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from qqtools.version import __version__

from ..config_types import RootConfig
from ..layout import (
    load_machine_record,
    load_machine_registration,
    load_root_config,
    save_machine_record,
    save_machine_registration,
)
from ..lease import lease_expiry, load_lease_policy
from ..machine_config import MACHINE_AGENT_RUNTIME
from ..runtime.paths import shared_paths
from ..runtime.protocol_compatibility import REGISTRATION_REPLACEMENT_CAPABILITY
from ..runtime.records import utc_now
from ..runtime.registration_protocol import (
    REGISTRATION_PROTOCOL,
    TARGET_REGISTRATION_VERSION,
    _read_protocol_record,
    assert_name_unreserved,
    convert_registration,
    reservation_path,
    validate_registration,
)
from ..runtime.responsibility_store import DurableIO
from ..runtime.store import atomic_replace, read_json
from .bindings import ProjectBinding
from .config import validate_agent_name
from .inventory import (
    ProjectInventoryEntry,
    canonical_shared_root,
    exact_binding_for_entry,
    load_inventory,
    save_inventory_locked,
)

INTENT_VERSION = 1
MAX_INTENT_BYTES = 64 * 1024
INTENT_KEY = "project_registration_replacement"
PHASES = frozenset(
    {
        "preparing",
        "reserved",
        "draining",
        "old_retirement_committed",
        "target_published",
        "binding_published",
        "completed",
        "cancelling",
        "reservation_release_committed",
    }
)
CANCELLABLE_PHASES = frozenset({"preparing", "reserved", "draining"})


class ReplacementUsageError(ValueError):
    """A replacement request has invalid user-controlled options."""


class ReplacementOperationalError(RuntimeError):
    """A durable replacement state cannot advance safely."""


def _intent_path(runtime: Any) -> Path:
    return runtime.root / "project-registration-replacement.json"


def _read_bounded_json(path: Path) -> dict[str, Any] | None:
    return _read_protocol_record(path)


def read_replacement_intent(runtime: Any) -> dict[str, Any] | None:
    """Read and validate the one machine-local replacement intent."""
    value = _read_bounded_json(_intent_path(runtime))
    if value is None:
        return None
    intent = value.get(INTENT_KEY)
    if not isinstance(intent, dict) or type(intent.get("version")) is not int:
        raise ReplacementOperationalError("project registration replacement record is malformed.")
    if intent["version"] != INTENT_VERSION:
        raise ReplacementOperationalError("project registration replacement record version is unsupported.")
    required_strings = (
        "transaction_id",
        "project_id",
        "shared_root",
        "runtime_id",
        "runtime_root",
        "old_machine_name",
        "old_generation",
        "requested_machine_name",
        "target_generation",
        "old_name_source",
    )
    for field in required_strings:
        if not isinstance(intent.get(field), str) or not intent[field]:
            raise ReplacementOperationalError(f"replacement record has an invalid {field}.")
    for field in ("old_machine_name", "requested_machine_name"):
        validate_agent_name(intent[field])
    if intent["old_name_source"] not in {"default", "explicit", "unresolved"}:
        raise ReplacementOperationalError("replacement record has an invalid old_name_source.")
    for field in ("registry_revision", "inventory_revision"):
        if type(intent.get(field)) is not int or intent[field] < 0:
            raise ReplacementOperationalError(f"replacement record has an invalid {field}.")
    if type(intent.get("original_enabled")) is not bool:
        raise ReplacementOperationalError("replacement record has an invalid original_enabled value.")
    if intent.get("old_name_override") is not None and not isinstance(intent["old_name_override"], str):
        raise ReplacementOperationalError("replacement record has an invalid old_name_override.")
    if intent.get("phase") not in PHASES:
        raise ReplacementOperationalError("replacement record has an unsupported phase.")
    try:
        root = Path(intent["shared_root"])
        runtime_root = Path(intent["runtime_root"])
    except (TypeError, ValueError) as exc:
        raise ReplacementOperationalError("replacement record has invalid paths.") from exc
    if not root.is_absolute() or root.name != ".qexp" or not runtime_root.is_absolute():
        raise ReplacementOperationalError("replacement record has invalid paths.")
    return dict(intent)


def _write_intent(runtime: Any, intent: dict[str, Any], phase: str) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"unsupported replacement phase: {phase}")
    updated = dict(intent)
    updated["version"] = INTENT_VERSION
    updated["phase"] = phase
    updated["updated_at"] = utc_now()
    encoded = json.dumps({INTENT_KEY: updated}, ensure_ascii=False).encode("utf-8")
    if len(encoded) > MAX_INTENT_BYTES:
        raise ReplacementOperationalError("project registration replacement record exceeds 64 KiB.")
    atomic_replace(_intent_path(runtime), {INTENT_KEY: updated})
    return updated


def _clear_intent(runtime: Any) -> None:
    path = _intent_path(runtime)
    path.unlink(missing_ok=True)
    DurableIO().sync_directory(path.parent, "project_registration_replacement_cleared")


def _reservation_document(intent: dict[str, Any]) -> dict[str, Any]:
    return {
        "registration_reservation": {
            "version": 1,
            "transaction_id": intent["transaction_id"],
            "project_id": intent["project_id"],
            "shared_root": intent["shared_root"],
            "runtime_id": intent["runtime_id"],
            "runtime_root": intent["runtime_root"],
            "old_machine_name": intent["old_machine_name"],
            "old_generation": intent["old_generation"],
            "requested_machine_name": intent["requested_machine_name"],
            "target_generation": intent["target_generation"],
        }
    }


def _reservation_matches(value: dict[str, Any] | None, intent: dict[str, Any]) -> bool:
    if not isinstance(value, dict):
        return False
    reservation = value.get("registration_reservation")
    if not isinstance(reservation, dict) or type(reservation.get("version")) is not int or reservation["version"] != 1:
        return False
    expected = _reservation_document(intent)["registration_reservation"]
    return all(reservation.get(key) == expected[key] for key in expected)


def _read_reservation(cfg: RootConfig) -> dict[str, Any] | None:
    return _read_bounded_json(reservation_path(cfg))


def _require_exact_reservation(cfg: RootConfig, intent: dict[str, Any]) -> dict[str, Any]:
    value = _read_reservation(cfg)
    if value is None:
        raise ReplacementOperationalError("replacement reservation is missing; preserve the intent and retry.")
    if not _reservation_matches(value, intent):
        raise ReplacementOperationalError("replacement reservation is owned by another transaction.")
    return value


def _release_exact_reservation(cfg: RootConfig, intent: dict[str, Any], *, allow_missing: bool) -> None:
    path = reservation_path(cfg)
    value = _read_reservation(cfg)
    if value is None:
        if allow_missing:
            if path.parent.exists():
                DurableIO().sync_directory(path.parent, "project_registration_replacement_release_replay")
            return
        raise ReplacementOperationalError("replacement reservation is missing; preserve the intent and retry.")
    if not _reservation_matches(value, intent):
        if allow_missing:
            return
        raise ReplacementOperationalError("replacement reservation is owned by another transaction.")
    path.unlink(missing_ok=True)
    DurableIO().sync_directory(path.parent, "project_registration_replacement_reservation_released")


def _project_capability_active(root: Path) -> bool:
    schema_path = shared_paths(root)["schema"] / "version.json"
    try:
        schema = read_json(schema_path).get("schema")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ReplacementOperationalError("Project schema capability state is unavailable.") from exc
    capabilities = schema.get("required_capabilities") if isinstance(schema, dict) else None
    if not isinstance(capabilities, list) or not all(isinstance(item, str) for item in capabilities):
        raise ReplacementOperationalError("Project schema capability state is malformed.")
    return REGISTRATION_REPLACEMENT_CAPABILITY in capabilities


def _identity(root: Path) -> str:
    path = shared_paths(root)["project"] / "identity.json"
    try:
        value = read_json(path).get("project")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ReplacementOperationalError(f"Project identity is unavailable: {path}") from exc
    project_id = value.get("project_id") if isinstance(value, dict) else None
    if not isinstance(project_id, str) or not project_id:
        raise ReplacementOperationalError(f"Project identity is malformed: {path}")
    if value.get("shared_root") != str(root):
        raise ReplacementOperationalError(f"Project identity does not match {root}.")
    return project_id


def _target_config(root: Path, name: str) -> RootConfig:
    return load_root_config(root, name, require_initialized=False)


def _current_binding(project_id: str, root: Path, bindings: list[ProjectBinding]) -> ProjectBinding:
    matches = [item for item in bindings if item.project_id == project_id or item.shared_root == root]
    if len(matches) != 1 or matches[0].project_id != project_id or matches[0].shared_root != root:
        raise ReplacementOperationalError("Project registry identity and shared path conflict.")
    return matches[0]


def _intent_matches_request(intent: dict[str, Any], root: Path, project_id: str, requested_name: str | None) -> None:
    if intent["project_id"] != project_id or intent["shared_root"] != str(root):
        raise ReplacementOperationalError("a different Project registration replacement is already pending.")
    if requested_name is not None and intent["requested_machine_name"] != requested_name:
        raise ReplacementOperationalError(
            f"replacement already targets {intent['requested_machine_name']!r}; resume or cancel that intent first."
        )


def _replacement_result(
    runtime: Any,
    intent: dict[str, Any],
    *,
    status: str,
    phase: str,
    blockers: list[str] | tuple[str, ...] = (),
    cancellable: bool,
    generation: str | None = None,
) -> dict[str, Any]:
    bounded = []
    for blocker in blockers:
        text = str(blocker)
        if text and text not in bounded:
            bounded.append(text[:256])
        if len(bounded) >= 32:
            break
    root = Path(intent["shared_root"])
    path = root.parent
    replace_command = shlex.join(
        [
            "qexp",
            "--machine-runtime-root",
            str(runtime.root),
            "project",
            "register",
            str(path),
            "--machine",
            intent["requested_machine_name"],
            "--replace-current",
        ]
    )
    cancel_command = shlex.join(
        ["qexp", "--machine-runtime-root", str(runtime.root), "project", "register", str(path), "--cancel-replacement"]
    )
    next_actions = [replace_command] if status == "replacement_pending" else []
    if status == "replacement_pending" and cancellable:
        next_actions.append(cancel_command)
    result: dict[str, Any] = {
        "action": "project_registration_replacement",
        "status": status,
        "project_id": intent["project_id"],
        "shared_root": str(root),
        "runtime_id": intent["runtime_id"],
        "runtime_root": intent["runtime_root"],
        "old_machine_name": intent["old_machine_name"],
        "old_generation": intent["old_generation"],
        "requested_machine_name": intent["requested_machine_name"],
        "target_generation": intent["target_generation"],
        "phase": phase,
        "cancellable": cancellable,
        "blockers": bounded,
        "next_actions": next_actions,
        "follow_up_actions": next_actions,
        "original_enabled": intent["original_enabled"],
        "registry_revision": intent["registry_revision"],
        "inventory_revision": intent["inventory_revision"],
    }
    if generation is not None:
        result["registration_generation"] = generation
    return result


def _raise_if_machine_replacement_pending(runtime: Any) -> None:
    if runtime.paths["replacement_transaction"].exists():
        raise ReplacementOperationalError(
            "Project registration replacement is blocked by a pending machine replacement."
        )


def _load_state(
    runtime: Any, root: Path, project_id: str
) -> tuple[int, list[Any], int, list[ProjectBinding], ProjectBinding, ProjectInventoryEntry]:
    runtime.registration.prepare_registry_locked()
    if runtime.paths["registration_transaction"].exists():
        raise ReplacementOperationalError("pending registration recovery must be resolved before replacement.")
    inventory_revision, entries = load_inventory(runtime)
    registry_revision, bindings = runtime.load_registry_uncached()
    binding = _current_binding(project_id, root, bindings)
    entry = next((item for item in entries if item.project_id == project_id and item.shared_root == root), None)
    if entry is None:
        raise ReplacementOperationalError("Project inventory has no matching entry.")
    mirrored, conflict = exact_binding_for_entry(entry, bindings)
    if conflict or mirrored != binding:
        raise ReplacementOperationalError("Project inventory does not exactly mirror the current binding.")
    return inventory_revision, entries, registry_revision, bindings, binding, entry


def _validate_target_absence(
    root: Path,
    requested_name: str,
    project_id: str,
    old: ProjectBinding,
    entries: list[ProjectInventoryEntry],
    bindings: list[ProjectBinding],
    target_cfg: RootConfig,
) -> None:
    del root, requested_name, project_id, old, entries, bindings
    target_directory = shared_paths(target_cfg.shared_root)["machines"] / target_cfg.machine_name
    if target_directory.exists() or target_directory.is_symlink():
        raise ReplacementOperationalError(
            f"logical machine name {target_cfg.machine_name!r} already has shared Project state; resolve it first."
        )
    reservation = _read_reservation(target_cfg)
    if reservation is not None:
        raise ReplacementOperationalError(
            f"logical machine name {target_cfg.machine_name!r} is reserved by another replacement."
        )


def _reserve_target(runtime: Any, intent: dict[str, Any], old_cfg: RootConfig, target_cfg: RootConfig) -> None:
    with runtime._registration_guards(old_cfg, target_cfg):
        _require_old_registration(intent, old_cfg)
        existing = _read_reservation(target_cfg)
        if existing is not None:
            if not _reservation_matches(existing, intent):
                raise ReplacementOperationalError(
                    f"logical machine name {target_cfg.machine_name!r} is reserved by another replacement."
                )
        else:
            _validate_target_absence(
                Path(intent["shared_root"]),
                intent["requested_machine_name"],
                intent["project_id"],
                ProjectBinding.from_canonical_paths(
                    intent["project_id"],
                    Path(intent["shared_root"]),
                    intent["old_machine_name"],
                    intent["original_enabled"],
                    intent["old_generation"],
                    intent["runtime_id"],
                    intent["runtime_root"],
                ),
                [],
                [],
                target_cfg,
            )
        atomic_replace(reservation_path(target_cfg), _reservation_document(intent))
        _write_intent(runtime, intent, "reserved")


def _disable_old_admission(
    runtime: Any,
    intent: dict[str, Any],
    entry: ProjectInventoryEntry,
    old: ProjectBinding,
) -> dict[str, Any]:
    with runtime.inventory_guard():
        with runtime.registry_guard():
            revision, bindings = runtime.load_registry_uncached()
            current = _current_binding(intent["project_id"], Path(intent["shared_root"]), bindings)
            if (
                current.machine_name != intent["old_machine_name"]
                or current.registration_generation != intent["old_generation"]
            ):
                raise ReplacementOperationalError("old Project binding identity changed while replacement was pending.")
            current = replace(current, enabled=False, _canonical_paths=True)
            bindings = [current if item.project_id == current.project_id else item for item in bindings]
            runtime.registration.save_registry_locked(revision + 1, bindings)
            revision += 1
            inventory_revision, entries = load_inventory(runtime)
            updated_entry = replace(entry, enabled=False, _canonical_paths=True)
            if updated_entry != entry:
                entries = [updated_entry if item == entry else item for item in entries]
                save_inventory_locked(runtime, inventory_revision + 1, entries)
                inventory_revision += 1
            intent["registry_revision"] = revision
            intent["inventory_revision"] = inventory_revision
            return _write_intent(runtime, intent, "draining")


def _old_binding_from_registry(runtime: Any, intent: dict[str, Any]) -> ProjectBinding | None:
    _revision, bindings = runtime.load_registry_uncached()
    for binding in bindings:
        if (
            binding.project_id == intent["project_id"]
            and binding.shared_root == Path(intent["shared_root"])
            and binding.machine_name == intent["old_machine_name"]
        ):
            return binding
    return None


def _require_old_registration(intent: dict[str, Any], old_cfg: RootConfig) -> dict[str, Any]:
    if _identity(old_cfg.shared_root) != intent["project_id"]:
        raise ReplacementOperationalError("Project identity changed during replacement.")
    raw = load_machine_registration(old_cfg)
    record = raw.get("registration") if isinstance(raw, dict) else None
    if not isinstance(record, dict):
        raise ReplacementOperationalError("old shared registration is missing.")
    validate_registration(record, intent["project_id"], old_cfg, allow_source=True)
    is_retired = intent["phase"] in {"old_retirement_committed", "target_published", "binding_published", "completed"}
    if (
        record["generation"] != intent["old_generation"]
        or record["runtime_instance_id"] != intent["runtime_id"]
        or record["runtime_root"] != intent["runtime_root"]
        or (
            record["state"] != "eligible"
            and not (
                is_retired
                and record["state"] == "superseded"
                and record.get("superseded_by_generation") == intent["target_generation"]
            )
        )
    ):
        raise ReplacementOperationalError("old shared registration identity changed.")
    return record


def _retirement_callback(runtime: Any, intent: dict[str, Any], old_cfg: RootConfig, target_cfg: RootConfig) -> None:
    with runtime._registration_guards(old_cfg, target_cfg):
        _require_exact_reservation(target_cfg, intent)
        _require_old_registration(intent, old_cfg)
        _write_intent(runtime, intent, "old_retirement_committed")


def _retire_old(runtime: Any, intent: dict[str, Any], old_cfg: RootConfig, target_cfg: RootConfig) -> list[str] | None:
    old = _old_binding_from_registry(runtime, intent)
    if old is None:
        if intent["phase"] == "draining":
            raise ReplacementOperationalError("old Project binding disappeared before retirement committed.")
        return None
    if (
        old.registration_generation != intent["old_generation"]
        or old.runtime_instance_id != intent["runtime_id"]
        or old.runtime_root not in {None, intent["runtime_root"]}
        or old.enabled
    ):
        raise ReplacementOperationalError("old Project binding identity changed before retirement.")
    from ..authority import AuthoritySupervisor

    supervisor = AuthoritySupervisor(
        replace(old_cfg, runtime_root=runtime.project_paths(old.project_id)["root"]),
        reservation_runtime_root=runtime.root,
    )
    try:
        supervisor.reconcile_terminal_cleanup()
    finally:
        supervisor.close()
    try:
        runtime._remove_binding_once(
            old.project_id,
            before_retire=lambda: _retirement_callback(runtime, intent, old_cfg, target_cfg),
            replacement_fence=True,
        )
    except Exception as exc:
        from ..runtime.responsibility_capture import CaptureBusy

        blockers: list[str] = []
        try:
            blockers = runtime.binding_blockers(old)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            pass
        if isinstance(exc, CaptureBusy) or blockers:
            if not blockers:
                blockers = [str(exc)]
            return blockers
        raise
    return None


def _target_registration(runtime: Any, intent: dict[str, Any], target_cfg: RootConfig) -> None:
    existing = load_machine_registration(target_cfg)
    if existing is not None:
        record = existing.get("registration") if isinstance(existing, dict) else None
        if not isinstance(record, dict):
            raise ReplacementOperationalError("target shared registration is malformed.")
        validate_registration(record, intent["project_id"], target_cfg)
        if (
            record.get("generation") != intent["target_generation"]
            or record.get("runtime_instance_id") != intent["runtime_id"]
            or record.get("runtime_root") != intent["runtime_root"]
            or record.get("state") != "eligible"
        ):
            raise ReplacementOperationalError("target shared registration belongs to another owner.")
        record.update(eligibility_expires_at=lease_expiry(load_lease_policy(target_cfg)), updated_at=utc_now())
        save_machine_registration(target_cfg, existing)
        return
    if intent["phase"] != "old_retirement_committed":
        raise ReplacementOperationalError("published target registration disappeared; preserve the intent.")
    assert_name_unreserved(target_cfg, transaction_id=intent["transaction_id"])
    now = utc_now()
    record = {
        "version": TARGET_REGISTRATION_VERSION,
        "project_id": intent["project_id"],
        "shared_root": intent["shared_root"],
        "machine_name": intent["requested_machine_name"],
        "generation": intent["target_generation"],
        "protocol_version": TARGET_REGISTRATION_VERSION,
        "registration_protocol": REGISTRATION_PROTOCOL,
        "recovery_protocol": None,
        "client_version": __version__,
        "runtime_instance_id": intent["runtime_id"],
        "runtime_root": intent["runtime_root"],
        "state": "eligible",
        "eligibility_expires_at": lease_expiry(load_lease_policy(target_cfg)),
        "created_at": now,
        "updated_at": now,
    }
    save_machine_registration(target_cfg, {"registration": record})


def _publish_target_shared(runtime: Any, intent: dict[str, Any], old_cfg: RootConfig, target_cfg: RootConfig) -> None:
    with runtime._registration_guards(old_cfg, target_cfg):
        _require_exact_reservation(target_cfg, intent)
        old_record = _require_old_registration(intent, old_cfg)
        target_record = load_machine_record(target_cfg)
        if target_record is None:
            target_record = {
                "machine": {
                    "machine_name": target_cfg.machine_name,
                    "project_id": intent["project_id"],
                    "shared_root": intent["shared_root"],
                    "runtime_root": str(runtime.project_paths(intent["project_id"])["root"]),
                    "agent_mode": "on_demand",
                    "agent_runtime": MACHINE_AGENT_RUNTIME,
                }
            }
        machine = target_record.get("machine")
        if not isinstance(machine, dict) or any(
            machine.get(key) != expected
            for key, expected in {
                "project_id": intent["project_id"],
                "shared_root": intent["shared_root"],
                "machine_name": intent["requested_machine_name"],
                "agent_runtime": MACHINE_AGENT_RUNTIME,
            }.items()
        ):
            raise ReplacementOperationalError("target machine metadata does not match the replacement intent.")
        save_machine_record(target_cfg, target_record)
        old_record = convert_registration(old_record, intent["project_id"], old_cfg)
        old_record.update(
            state="superseded", superseded_by_generation=intent["target_generation"], updated_at=utc_now()
        )
        save_machine_registration(old_cfg, {"registration": old_record})
        _target_registration(runtime, intent, target_cfg)


def _publish_target_binding(runtime: Any, intent: dict[str, Any]) -> dict[str, Any]:
    with runtime.inventory_guard():
        with runtime.registry_guard():
            revision, bindings = runtime.load_registry_uncached()
            old = _old_binding_from_registry(runtime, intent)
            if old is not None:
                raise ReplacementOperationalError("old Project binding remains after retirement; retry replacement.")
            target = next(
                (
                    item
                    for item in bindings
                    if item.project_id == intent["project_id"]
                    and item.shared_root == Path(intent["shared_root"])
                    and item.machine_name == intent["requested_machine_name"]
                ),
                None,
            )
            if target is None:
                if any(
                    item.project_id == intent["project_id"] or item.shared_root == Path(intent["shared_root"])
                    for item in bindings
                ):
                    raise ReplacementOperationalError("target local registry identity conflicts with another binding.")
                target = ProjectBinding.from_canonical_paths(
                    intent["project_id"],
                    Path(intent["shared_root"]),
                    intent["requested_machine_name"],
                    False,
                    intent["target_generation"],
                    intent["runtime_id"],
                    intent["runtime_root"],
                )
                runtime.registration.save_registry_locked(revision + 1, [*bindings, target])
                revision += 1
            elif (
                target.registration_generation != intent["target_generation"]
                or target.runtime_instance_id != intent["runtime_id"]
                or target.runtime_root != intent["runtime_root"]
                or target.enabled
            ):
                raise ReplacementOperationalError("target local binding identity is inconsistent.")
            inventory_revision, entries = load_inventory(runtime)
            current_entry = next(
                (
                    item
                    for item in entries
                    if item.project_id == intent["project_id"] and item.shared_root == Path(intent["shared_root"])
                ),
                None,
            )
            if current_entry is None:
                raise ReplacementOperationalError("Project inventory entry disappeared during replacement.")
            intent["registry_revision"] = revision
            intent["inventory_revision"] = inventory_revision
            return _write_intent(runtime, intent, "binding_published")


def _publish_enablement(runtime: Any, intent: dict[str, Any]) -> dict[str, Any]:
    old_cfg = load_root_config(Path(intent["shared_root"]), intent["old_machine_name"], require_initialized=False)
    target_cfg = _target_config(Path(intent["shared_root"]), intent["requested_machine_name"])
    with runtime.inventory_guard():
        with runtime.registry_guard(), runtime._registration_guards(old_cfg, target_cfg):
            _require_old_registration(intent, old_cfg)
            _target_registration(runtime, intent, target_cfg)
            revision, bindings = runtime.load_registry_uncached()
            target = next(
                (
                    item
                    for item in bindings
                    if item.project_id == intent["project_id"]
                    and item.shared_root == Path(intent["shared_root"])
                    and item.machine_name == intent["requested_machine_name"]
                ),
                None,
            )
            if target is None or (
                target.registration_generation != intent["target_generation"]
                or target.runtime_instance_id != intent["runtime_id"]
                or target.runtime_root != intent["runtime_root"]
            ):
                raise ReplacementOperationalError("target local binding identity changed during replacement.")
            inventory_revision, entries = load_inventory(runtime)
            current = next(
                (
                    item
                    for item in entries
                    if item.project_id == intent["project_id"] and item.shared_root == Path(intent["shared_root"])
                ),
                None,
            )
            if current is None:
                raise ReplacementOperationalError("Project inventory entry is missing during replacement.")
            updated = ProjectInventoryEntry(
                intent["project_id"],
                Path(intent["shared_root"]),
                intent["original_enabled"],
                "explicit",
                intent["requested_machine_name"],
            )
            if updated != current:
                save_inventory_locked(
                    runtime, inventory_revision + 1, [updated if item == current else item for item in entries]
                )
                inventory_revision += 1
            enabled_target = replace(target, enabled=intent["original_enabled"], _canonical_paths=True)
            if enabled_target != target:
                runtime.registration.save_registry_locked(
                    revision + 1,
                    [enabled_target if item == target else item for item in bindings],
                )
                revision += 1
            intent["registry_revision"] = revision
            intent["inventory_revision"] = inventory_revision
            return _write_intent(runtime, intent, intent["phase"])


def _finish_replacement(
    runtime: Any, intent: dict[str, Any], old_cfg: RootConfig, target_cfg: RootConfig
) -> dict[str, Any]:
    if intent["phase"] == "old_retirement_committed":
        blockers = _retire_old(runtime, intent, old_cfg, target_cfg)
        latest = read_replacement_intent(runtime) or intent
        if blockers:
            return _replacement_result(
                runtime,
                latest,
                status="replacement_pending",
                phase=latest["phase"],
                blockers=blockers,
                cancellable=latest["phase"] in CANCELLABLE_PHASES,
            )
        _publish_target_shared(runtime, latest, old_cfg, target_cfg)
        intent = read_replacement_intent(runtime) or latest
        intent = _write_intent(runtime, intent, "target_published")
    if intent["phase"] in {"target_published", "binding_published"}:
        with runtime._registration_guards(old_cfg, target_cfg):
            _require_old_registration(intent, old_cfg)
            _target_registration(runtime, intent, target_cfg)
    if intent["phase"] == "target_published":
        intent = _publish_target_binding(runtime, intent)
    if intent["phase"] == "binding_published":
        intent = _publish_enablement(runtime, intent)
    target_cfg = _target_config(Path(intent["shared_root"]), intent["requested_machine_name"])
    old_cfg = load_root_config(Path(intent["shared_root"]), intent["old_machine_name"], require_initialized=False)
    with runtime._registration_guards(old_cfg, target_cfg):
        _release_exact_reservation(target_cfg, intent, allow_missing=True)
    intent = _write_intent(runtime, intent, "completed")
    result = _replacement_result(
        runtime,
        intent,
        status="replacement_completed",
        phase="completed",
        cancellable=False,
        generation=intent["target_generation"],
    )
    _clear_intent(runtime)
    return result


def _validate_old_local_for_cancellation(runtime: Any, intent: dict[str, Any]) -> None:
    root = Path(intent["shared_root"])
    with runtime.inventory_guard():
        with runtime.registry_guard():
            _revision, bindings = runtime.load_registry_uncached()
            old = _current_binding(intent["project_id"], root, bindings)
            if (
                old.machine_name != intent["old_machine_name"]
                or old.registration_generation != intent["old_generation"]
                or old.runtime_instance_id != intent["runtime_id"]
                or old.runtime_root not in {None, intent["runtime_root"]}
            ):
                raise ReplacementOperationalError("old Project binding changed; cancellation cannot restore it.")
            _inventory_revision, entries = load_inventory(runtime)
            entry = next(
                (item for item in entries if item.project_id == intent["project_id"] and item.shared_root == root), None
            )
            if entry is None:
                raise ReplacementOperationalError(
                    "Project inventory entry disappeared; cancellation cannot restore it."
                )


def _restore_cancellation(runtime: Any, intent: dict[str, Any]) -> dict[str, Any]:
    root = Path(intent["shared_root"])
    old_cfg = load_root_config(root, intent["old_machine_name"], require_initialized=False)
    target_cfg = _target_config(root, intent["requested_machine_name"])
    if intent["phase"] not in CANCELLABLE_PHASES | {"cancelling", "reservation_release_committed"}:
        raise ReplacementOperationalError("old retirement has committed; cancellation is unavailable.")
    with runtime._registration_guards(old_cfg, target_cfg):
        _require_old_registration(intent, old_cfg)
    _validate_old_local_for_cancellation(runtime, intent)
    if intent["phase"] in {"preparing", "reserved", "draining"}:
        with runtime._registration_guards(old_cfg, target_cfg):
            value = _read_reservation(target_cfg)
            if intent["phase"] == "preparing" and not _reservation_matches(value, intent):
                _clear_intent(runtime)
                return _replacement_result(
                    runtime,
                    intent,
                    status="replacement_cancelled",
                    phase="preparing",
                    cancellable=False,
                )
            _require_exact_reservation(target_cfg, intent)
    if intent["phase"] == "cancelling":
        with runtime._registration_guards(old_cfg, target_cfg):
            _require_exact_reservation(target_cfg, intent)
    if intent["phase"] in {"preparing", "reserved", "draining"}:
        intent = _write_intent(runtime, intent, "cancelling")
    if intent["phase"] == "cancelling":
        intent = _write_intent(runtime, intent, "reservation_release_committed")
    if intent["phase"] == "reservation_release_committed":
        with runtime._registration_guards(old_cfg, target_cfg):
            _release_exact_reservation(target_cfg, intent, allow_missing=True)
        with runtime.inventory_guard():
            with runtime.registry_guard(), runtime._registration_guards(old_cfg, target_cfg):
                _require_old_registration(intent, old_cfg)
                revision, bindings = runtime.load_registry_uncached()
                old = _current_binding(intent["project_id"], root, bindings)
                if (
                    old.machine_name != intent["old_machine_name"]
                    or old.registration_generation != intent["old_generation"]
                    or old.runtime_instance_id != intent["runtime_id"]
                    or old.runtime_root not in {None, intent["runtime_root"]}
                ):
                    raise ReplacementOperationalError("old Project binding changed; cancellation cannot restore it.")
                restored = replace(old, enabled=intent["original_enabled"], _canonical_paths=True)
                if restored != old:
                    runtime.registration.save_registry_locked(
                        revision + 1,
                        [restored if item == old else item for item in bindings],
                    )
                    revision += 1
                inventory_revision, entries = load_inventory(runtime)
                current = next(
                    (item for item in entries if item.project_id == intent["project_id"] and item.shared_root == root),
                    None,
                )
                if current is None:
                    raise ReplacementOperationalError("Project inventory entry disappeared during cancellation.")
                restored_entry = ProjectInventoryEntry(
                    intent["project_id"],
                    root,
                    intent["original_enabled"],
                    intent["old_name_source"],
                    intent["old_name_override"],
                )
                if restored_entry != current:
                    save_inventory_locked(
                        runtime,
                        inventory_revision + 1,
                        [restored_entry if item == current else item for item in entries],
                    )
                    inventory_revision += 1
                intent["registry_revision"] = revision
                intent["inventory_revision"] = inventory_revision
                intent = _write_intent(runtime, intent, "reservation_release_committed")
    result = _replacement_result(
        runtime,
        intent,
        status="replacement_cancelled",
        phase="reservation_release_committed",
        cancellable=False,
    )
    _clear_intent(runtime)
    return result


def _replace_locked(
    runtime: Any,
    shared_root: str | Path,
    requested_name: str | None,
    *,
    cancel_replacement: bool = False,
) -> dict[str, Any]:
    """Start or resume one durable Project registration-name replacement."""
    machine_runtime = runtime
    machine_runtime.require_initialized()
    root = canonical_shared_root(shared_root)
    project_id = _identity(root)
    intent = read_replacement_intent(machine_runtime)
    if intent is not None:
        _intent_matches_request(intent, root, project_id, requested_name)
        if intent["runtime_id"] != machine_runtime.instance_id or intent["runtime_root"] != str(machine_runtime.root):
            raise ReplacementOperationalError("replacement runtime identity changed; preserve the intent.")
        requested_name = intent["requested_machine_name"]
        if cancel_replacement or intent["phase"] in {"cancelling", "reservation_release_committed"}:
            return _restore_cancellation(machine_runtime, intent)
        if intent["phase"] == "completed":
            _publish_enablement(machine_runtime, intent)
            result = _replacement_result(
                machine_runtime,
                intent,
                status="replacement_completed",
                phase="completed",
                cancellable=False,
                generation=intent["target_generation"],
            )
            _clear_intent(machine_runtime)
            return result
    elif cancel_replacement:
        raise ReplacementOperationalError("no pending Project registration replacement exists for this path.")

    _raise_if_machine_replacement_pending(machine_runtime)
    try:
        requested_name = validate_agent_name(requested_name)
    except (TypeError, ValueError) as exc:
        raise ReplacementUsageError("--machine NAME is invalid.") from exc

    if intent is None:
        with machine_runtime.inventory_guard(), machine_runtime.registry_guard():
            inventory_revision, entries, registry_revision, bindings, old, entry = _load_state(
                machine_runtime, root, project_id
            )
            if old.machine_name == requested_name:
                if not machine_runtime.binding_write_eligible(old, renew=True):
                    raise ReplacementOperationalError(
                        "current registration is not eligible; ordinary registration is required."
                    )
                unchanged = {
                    "action": "project_registration_replacement",
                    "status": "unchanged",
                    "project_id": project_id,
                    "shared_root": str(root),
                    "runtime_id": machine_runtime.instance_id,
                    "runtime_root": str(machine_runtime.root),
                    "old_machine_name": old.machine_name,
                    "old_generation": old.registration_generation,
                    "requested_machine_name": requested_name,
                    "target_generation": old.registration_generation,
                    "phase": "unchanged",
                    "cancellable": False,
                    "blockers": [],
                    "next_actions": [],
                    "follow_up_actions": [],
                    "original_enabled": old.enabled,
                    "registry_revision": registry_revision,
                    "inventory_revision": inventory_revision,
                    "registration_generation": old.registration_generation,
                }
                return unchanged
            if not _project_capability_active(root):
                raise ReplacementOperationalError("Project registration replacement is pending upgrade activation.")
            old_cfg = old.root_config()
            target_cfg = _target_config(root, requested_name)
            intent = {
                "version": INTENT_VERSION,
                "transaction_id": uuid.uuid4().hex,
                "project_id": project_id,
                "shared_root": str(root),
                "runtime_id": machine_runtime.instance_id,
                "runtime_root": str(machine_runtime.root),
                "old_machine_name": old.machine_name,
                "old_generation": old.registration_generation,
                "requested_machine_name": requested_name,
                "target_generation": uuid.uuid4().hex,
                "original_enabled": old.enabled,
                "old_name_source": entry.name_source,
                "old_name_override": entry.name_override,
                "registry_revision": registry_revision,
                "inventory_revision": inventory_revision,
                "phase": "preparing",
            }
            with machine_runtime._registration_guards(old_cfg, target_cfg):
                _require_old_registration(intent, old_cfg)
                _validate_target_absence(root, requested_name, project_id, old, entries, bindings, target_cfg)
                _write_intent(machine_runtime, intent, "preparing")
    else:
        old_cfg = load_root_config(root, intent["old_machine_name"], require_initialized=False)
        target_cfg = _target_config(root, intent["requested_machine_name"])
        old = ProjectBinding.from_canonical_paths(
            intent["project_id"],
            root,
            intent["old_machine_name"],
            intent["original_enabled"],
            intent["old_generation"],
            intent["runtime_id"],
            intent["runtime_root"],
        )

    if intent["phase"] == "preparing":
        _reserve_target(machine_runtime, intent, old_cfg, target_cfg)
        intent = read_replacement_intent(machine_runtime) or intent
    if intent["phase"] == "reserved":
        with machine_runtime.inventory_guard(), machine_runtime.registry_guard():
            _, entries, _, bindings, old, entry = _load_state(machine_runtime, root, project_id)
        with machine_runtime._registration_guards(old_cfg, target_cfg):
            _require_exact_reservation(target_cfg, intent)
            _require_old_registration(intent, old_cfg)
        intent = _disable_old_admission(machine_runtime, intent, entry, old)
    if intent["phase"] == "draining":
        with machine_runtime._registration_guards(old_cfg, target_cfg):
            _require_exact_reservation(target_cfg, intent)
            _require_old_registration(intent, old_cfg)
        blockers = _retire_old(machine_runtime, intent, old_cfg, target_cfg)
        latest = read_replacement_intent(machine_runtime) or intent
        if blockers:
            return _replacement_result(
                machine_runtime,
                latest,
                status="replacement_pending",
                phase=latest["phase"],
                blockers=blockers,
                cancellable=latest["phase"] in CANCELLABLE_PHASES,
            )
        intent = latest
    if intent["phase"] in {"old_retirement_committed", "target_published", "binding_published"}:
        return _finish_replacement(
            machine_runtime, read_replacement_intent(machine_runtime) or intent, old_cfg, target_cfg
        )
    raise ReplacementOperationalError(f"replacement stopped in unsupported phase {intent['phase']!r}.")


def replace_current_project_registration(
    runtime: Any, shared_root: str | Path, requested_name: str | None, *, cancel_replacement: bool = False
) -> dict[str, Any]:
    """Advance one exact replacement while owning machine lifecycle changes."""
    with runtime.agent_lifecycle_guard():
        try:
            return _replace_locked(runtime, shared_root, requested_name, cancel_replacement=cancel_replacement)
        except ReplacementUsageError:
            raise
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            suffix = " No earlier replacement intent is active."
            try:
                intent = read_replacement_intent(runtime)
                if intent is not None:
                    command = shlex.join(
                        [
                            "qexp",
                            "--machine-runtime-root",
                            str(runtime.root),
                            "project",
                            "register",
                            str(Path(intent["shared_root"]).parent),
                            "--machine",
                            intent["requested_machine_name"],
                            "--replace-current",
                        ]
                    )
                    suffix = f" Earlier replacement intent remains active at {intent['phase']}; resume with: {command}"
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                suffix = " Earlier replacement intent is unreadable; preserve its evidence."
            raise ReplacementOperationalError(str(exc) + suffix) from exc
