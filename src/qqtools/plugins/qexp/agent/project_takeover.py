"""Read-only takeover preview and confirmation for one Project and logical name."""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config_types import RootConfig
from ..runtime.authority_scan import is_path_present
from ..runtime.paths import shared_paths
from ..runtime.records import validate_identifier
from ..runtime.registration_authority import registration_state
from ..runtime.registration_protocol import (
    _read_protocol_record,
    assert_name_unreserved,
    read_registration_document,
    validate_registration,
)
from ..runtime.takeover_registration import RegistrationWitness, TakeoverRegistrationConflict, has_takeover_capability
from .bindings import ProjectBinding
from .config import validate_agent_name
from .context import MachineRuntime
from .inventory import canonical_shared_root
from .registration_replacement import (
    _read_reservation,
    _require_exact_reservation,
    _reservation_matches,
    read_registration_intents,
)
from .takeover_binding import observe_takeover_inventory
from .takeover_plan import TakeoverGroupApproval, takeover_plan_id, takeover_plan_nonce
from .takeover_recovery_preview import RecoveryPreviewCursorError, observe_departing_recovery
from .takeover_task_preview import TaskImpactCursorError, observe_takeover_group_selection, observe_takeover_tasks


class TakeoverUsageError(ValueError):
    """A Project takeover argument or confirmation is invalid."""


@dataclass(frozen=True, slots=True)
class TakeoverPlanInputs:
    """The registration and Group evidence bound to a takeover plan."""

    project_id: str
    shared_root: Path
    runtime_id: str
    runtime_root: Path
    target: RegistrationWitness
    departing: RegistrationWitness | None
    original_enabled: bool
    groups: tuple[TakeoverGroupApproval, ...]

    def plan_id(self, nonce: str | None = None) -> str:
        """Fingerprint this evidence with a fresh or supplied preview nonce."""
        return takeover_plan_id(
            project_id=self.project_id,
            shared_root=self.shared_root,
            runtime_id=self.runtime_id,
            runtime_root=self.runtime_root,
            target=self.target,
            departing=self.departing,
            original_enabled=self.original_enabled,
            groups=self.groups,
            nonce=nonce,
        )


def _observe_project_id(root: Path) -> str:
    path = shared_paths(root)["project"] / "identity.json"
    try:
        envelope = _read_protocol_record(path)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"Project identity cannot be read at {path}: {exc}") from exc
    record = envelope.get("project") if envelope is not None else None
    stable_id = record.get("project_id") if isinstance(record, dict) else None
    if (
        not isinstance(stable_id, str)
        or not stable_id
        or stable_id == "."
        or any(token in stable_id for token in ("/", "\\", "..", "\x00"))
        or record.get("shared_root") != str(root)
    ):
        raise RuntimeError(f"Project identity does not match the selected Project at {path}.")
    return stable_id


def _observe_binding(
    runtime: MachineRuntime, root: Path, selected_project_id: str, runtime_id: str
) -> ProjectBinding | None:
    _, bindings = runtime.load_registry_uncached()
    matches = [
        binding for binding in bindings if binding.project_id == selected_project_id or binding.shared_root == root
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError("Local registry has multiple bindings matching the selected Project.")
    binding = matches[0]
    if (
        binding.project_id != selected_project_id
        or binding.shared_root != root
        or binding.runtime_instance_id != runtime_id
        or binding.runtime_root not in {None, str(runtime.root)}
        or not binding.registration_generation
        or binding.machine_name == "."
        or "\x00" in binding.machine_name
    ):
        raise RuntimeError("Local registry binding does not match the selected Project and machine runtime.")
    return binding


def _observe_registration(cfg: RootConfig, selected_project_id: str) -> dict[str, Any] | None:
    try:
        envelope = read_registration_document(cfg)
        if envelope is None:
            return None
        record = envelope.get("registration")
        validate_registration(record, selected_project_id, cfg)
        runtime_root = record["runtime_root"]
        runtime_path = Path(runtime_root)
        if (
            runtime_path.anchor != "/"
            or ".." in runtime_path.parts
            or "\x00" in runtime_root
            or str(runtime_path) != runtime_root
        ):
            raise RuntimeError("registration runtime_root is not a canonical absolute path.")
    except (OSError, RuntimeError, TypeError) as exc:
        raise RuntimeError(f"Project registration for {cfg.machine_name!r} cannot be read or validated: {exc}") from exc
    return record


def _inspect_project_takeover(
    runtime: MachineRuntime,
    path: str | Path,
    machine_name: str,
    *,
    recovery_cursor: str | None = None,
    after_task_id: str | None = None,
    selected_groups: tuple[str, ...] = (),
) -> tuple[dict[str, Any], TakeoverPlanInputs | None]:
    try:
        machine_name = validate_agent_name(machine_name)
    except ValueError as exc:
        raise TakeoverUsageError(str(exc)) from exc
    if machine_name == "." or "\x00" in machine_name:
        raise TakeoverUsageError("machine name must not be '.' or contain NUL.")
    if not str(path) or "\x00" in str(path):
        raise TakeoverUsageError("Project path must be non-empty and must not contain NUL.")

    root = canonical_shared_root(Path(path).expanduser())
    selected_project_id = _observe_project_id(root)
    runtime_id = runtime.observe_instance_id()
    runtime_root = str(runtime.root)
    binding = _observe_binding(runtime, root, selected_project_id, runtime_id)
    target_cfg = RootConfig.from_canonical_paths(root, root.parent, machine_name, runtime.root)
    blockers: list[dict[str, str]] = []
    try:
        inventory = observe_takeover_inventory(runtime, selected_project_id, root)
        if binding is not None and inventory is None:
            raise RuntimeError("The departing Project binding is missing its inventory entry.")
    except (OSError, RuntimeError, ValueError) as exc:
        blockers.append({"code": "takeover_inventory_unavailable", "message": str(exc)})
        inventory = None

    try:
        has_capability = has_takeover_capability(root)
    except TakeoverRegistrationConflict as exc:
        blockers.append({"code": "takeover_schema_unsupported", "message": str(exc)})
    else:
        if not has_capability:
            blockers.append(
                {
                    "code": "registration_replacement_inactive",
                    "message": "Upgrade the Project to enable the registration replacement capability before takeover.",
                }
            )

    current_binding = None
    departing = None
    if binding is not None:
        if binding.machine_name == machine_name:
            blockers.append(
                {"code": "target_is_current_binding", "message": f"{machine_name!r} is already the local binding."}
            )
        else:
            current_cfg = RootConfig.from_canonical_paths(root, root.parent, binding.machine_name, runtime.root)
            current = _observe_registration(current_cfg, selected_project_id)
            if (
                current is None
                or current["generation"] != binding.registration_generation
                or current["runtime_instance_id"] != runtime_id
                or current["runtime_root"] != runtime_root
                or registration_state(current) == "superseded"
            ):
                blockers.append(
                    {
                        "code": "current_registration_changed",
                        "message": f"The shared registration for {binding.machine_name!r} differs from the local binding.",
                    }
                )
            else:
                departing = RegistrationWitness.from_record(current)
                try:
                    runtime.activation_consumer_retirements.validate_retirement(binding)
                except ValueError as exc:
                    blockers.append({"code": "departing_retirement_unsupported", "message": str(exc)})
        current_binding = {
            "machine_name": binding.machine_name,
            "registration_generation": binding.registration_generation,
            "enabled": binding.enabled,
        }

    target = _observe_registration(target_cfg, selected_project_id)
    target_witness = None
    if target is None:
        target_registration = {
            "machine_name": machine_name,
            "state": "missing",
            "generation": None,
            "runtime_id": None,
            "runtime_root": None,
            "eligibility_expires_at": None,
        }
        blockers.append(
            {"code": "target_registration_missing", "message": f"No shared registration exists for {machine_name!r}."}
        )
    else:
        target_witness = RegistrationWitness.from_record(target)
        state = registration_state(target)
        target_registration = {
            "machine_name": machine_name,
            "state": state,
            "generation": target["generation"],
            "runtime_id": target["runtime_instance_id"],
            "runtime_root": target["runtime_root"],
            "eligibility_expires_at": target["eligibility_expires_at"],
        }
        if state == "eligible":
            blockers.append(
                {
                    "code": "target_registration_eligible",
                    "message": f"The shared registration for {machine_name!r} still has write eligibility.",
                }
            )

    for path_key, code, reason in (
        (
            "registration_transaction",
            "registration_recovery_pending",
            "Resolve pending registration recovery before a takeover.",
        ),
        (
            "replacement_transaction",
            "machine_replacement_pending",
            "Resolve the pending machine identity replacement before a takeover.",
        ),
    ):
        if is_path_present(runtime.paths[path_key]):
            blockers.append({"code": code, "message": f"{reason} Evidence: {runtime.paths[path_key]}"})

    try:
        intent, pending_takeover = read_registration_intents(runtime)
    except TypeError as exc:
        raise RuntimeError(f"Project registration replacement record is malformed: {exc}") from exc
    has_local_reservation_owner = False
    if intent is not None:
        blockers.append(
            {
                "code": "registration_replacement_pending",
                "message": f"The machine runtime has a pending registration replacement in phase {intent['phase']!r}.",
            }
        )
        if (
            intent["project_id"] == selected_project_id
            and intent["shared_root"] == str(root)
            and intent["runtime_id"] == runtime_id
            and intent["runtime_root"] == runtime_root
            and intent["requested_machine_name"] == machine_name
        ):
            has_local_reservation_owner = True
            if intent["phase"] in {"reserved", "draining", "old_retirement_committed", "cancelling"}:
                _require_exact_reservation(target_cfg, intent)
            else:
                reservation = _read_reservation(target_cfg)
                if reservation is not None and not _reservation_matches(reservation, intent):
                    raise RuntimeError("The target reservation does not match the pending Project replacement.")
    if pending_takeover is not None:
        blockers.append(
            {
                "code": "project_takeover_pending",
                "message": f"The machine runtime has a Project takeover at phase {pending_takeover.phase!r}.",
            }
        )
    elif not has_local_reservation_owner:
        assert_name_unreserved(target_cfg)

    departing_recovery = None
    if binding is None:
        if recovery_cursor is not None:
            raise TakeoverUsageError("There is no departing Project binding for --recovery-cursor.")
    else:
        try:
            departing_recovery = observe_departing_recovery(runtime, binding, cursor=recovery_cursor)
        except RecoveryPreviewCursorError as exc:
            raise TakeoverUsageError(str(exc)) from exc
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            departing_recovery = {"status": "unavailable", "count": None, "records": [], "next_cursor": None}
            blockers.append(
                {
                    "code": "departing_recovery_unavailable",
                    "message": f"Recovery records for local binding {binding.machine_name!r} cannot be verified.",
                }
            )

    task_impact = None
    if target_registration["state"] in {"expired", "superseded"}:
        try:
            task_impact = observe_takeover_tasks(target_cfg, after_task_id=after_task_id)
        except TaskImpactCursorError as exc:
            raise TakeoverUsageError(str(exc)) from exc
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            task_impact = {"status": "unavailable", "count": None, "tasks": [], "next_task": None}
            blockers.append(
                {
                    "code": "target_task_impact_unavailable",
                    "message": f"Task impact for the target registration {machine_name!r} could not be verified.",
                }
            )
    elif after_task_id is not None:
        raise TakeoverUsageError("--tasks-after requires an expired or superseded target registration.")

    result = {
        "action": "project_takeover_preview",
        "status": "blocked" if blockers else "preview",
        "project_id": selected_project_id,
        "shared_root": str(root),
        "runtime_id": runtime_id,
        "machine_runtime_root": runtime_root,
        "current_binding": current_binding,
        "target_registration": target_registration,
        "departing_recovery": departing_recovery,
        "task_impact": task_impact,
        "blockers": blockers,
    }
    if blockers:
        return result, None

    assert target_witness is not None
    try:
        validate_identifier(target_witness.machine_name, "Target machine name")
        if len(target_witness.machine_name) > 128:
            raise ValueError("Target machine name exceeds the Project I/O limit.")
    except ValueError as exc:
        result["status"] = "blocked"
        blockers.append({"code": "target_name_unsupported", "message": str(exc)})
        return result, None
    try:
        selected_group_changes = observe_takeover_group_selection(target_cfg, selected_groups)
    except ValueError as exc:
        raise TakeoverUsageError(str(exc)) from exc
    groups = tuple(TakeoverGroupApproval.from_dict(group) for group in selected_group_changes)
    return result, TakeoverPlanInputs(
        project_id=selected_project_id,
        shared_root=root,
        runtime_id=runtime_id,
        runtime_root=runtime.root,
        target=target_witness,
        departing=departing,
        original_enabled=binding.enabled if binding is not None else inventory is None or inventory.enabled,
        groups=groups,
    )


def preview_project_takeover(
    runtime: MachineRuntime,
    path: str | Path,
    machine_name: str,
    *,
    recovery_cursor: str | None = None,
    after_task_id: str | None = None,
    selected_groups: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Observe takeover evidence and issue a plan ID when the preview is clear."""
    result, inputs = _inspect_project_takeover(
        runtime,
        path,
        machine_name,
        recovery_cursor=recovery_cursor,
        after_task_id=after_task_id,
        selected_groups=selected_groups,
    )
    result["plan_id"] = inputs.plan_id() if inputs is not None else None
    result["selected_group_changes"] = [group.to_dict() for group in inputs.groups] if inputs is not None else []
    return result


def confirm_project_takeover(
    runtime: MachineRuntime,
    path: str | Path,
    machine_name: str,
    plan_id: str,
    *,
    selected_groups: tuple[str, ...] = (),
) -> TakeoverPlanInputs:
    """Reobserve a plan and return its inputs only if the evidence still matches."""
    try:
        nonce = takeover_plan_nonce(plan_id)
    except ValueError as exc:
        raise TakeoverUsageError(str(exc)) from exc
    try:
        result, inputs = _inspect_project_takeover(runtime, path, machine_name, selected_groups=selected_groups)
    except TakeoverUsageError as exc:
        raise TakeoverUsageError(f"{exc} Get a fresh --plan.") from exc
    if inputs is None:
        reasons = " ".join(blocker["message"] for blocker in result["blockers"])
        raise TakeoverUsageError(f"Project takeover is blocked. {reasons} Get a fresh --plan.")
    if not hmac.compare_digest(plan_id, inputs.plan_id(nonce)):
        raise TakeoverUsageError("Project takeover plan has changed. Get a fresh --plan.")
    return inputs
