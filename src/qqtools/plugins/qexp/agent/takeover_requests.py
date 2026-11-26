"""Machine-local acceptance and cancellation requests for Project takeovers."""

from __future__ import annotations

import uuid
from dataclasses import replace
from pathlib import Path

from ..runtime.authority_scan import is_path_present
from ..runtime.responsibility_store import DurableIO
from ..runtime.takeover_registration import TakeoverRegistration
from .config import validate_agent_name
from .context import MachineRuntime
from .inventory import canonical_shared_root
from .project_takeover import TakeoverUsageError, confirm_project_takeover
from .takeover_binding import observe_takeover_inventory
from .takeover_intent import (
    _CANCELLABLE_PHASES,
    TakeoverIntent,
    TakeoverIntentConflict,
    _result_path,
    read_takeover_intent,
    read_takeover_result,
    save_takeover_intent,
)
from .takeover_plan import takeover_plan_nonce


def _normalize_request_scope(path: str | Path, machine_name: str, plan_id: str) -> tuple[Path, str]:
    try:
        takeover_plan_nonce(plan_id)
        machine_name = validate_agent_name(machine_name)
    except ValueError as exc:
        raise TakeoverUsageError(str(exc)) from exc
    if machine_name == "." or "\x00" in machine_name:
        raise TakeoverUsageError("machine name must not be '.' or contain NUL.")
    if not isinstance(path, (str, Path)) or not str(path) or "\x00" in str(path):
        raise TakeoverUsageError("Project path must be non-empty and must not contain NUL.")
    try:
        shared_root = canonical_shared_root(Path(path).expanduser())
    except OSError as exc:
        raise RuntimeError(f"Project path cannot be resolved: {exc}") from exc
    return shared_root, machine_name


def _read_requested_takeover(
    runtime: MachineRuntime,
    shared_root: Path,
    machine_name: str,
    plan_id: str,
    *,
    acceptance_groups: tuple[str, ...] | None = None,
) -> TakeoverIntent | None:
    """Read and durably acknowledge the exact plan under the lifecycle guard."""
    active = read_takeover_intent(runtime.root)
    receipt = read_takeover_result(runtime.root, plan_id)
    intent = receipt
    if intent is None and active is not None:
        if active.plan_id == plan_id:
            intent = active
        elif acceptance_groups is not None:
            raise TakeoverIntentConflict(f"Project takeover plan {active.plan_id} is already active.")
    if intent is None:
        return None

    transition = intent.transition
    if runtime.observe_instance_id() != transition.successor_runtime_id:
        raise TakeoverUsageError("Project takeover plan belongs to a different machine runtime identity.")
    if transition.shared_root != shared_root or transition.target.machine_name != machine_name:
        raise TakeoverUsageError("Project takeover plan does not match the selected Project and target machine.")
    if acceptance_groups is not None and acceptance_groups != tuple(
        approval.group_name for approval in intent.group_approvals
    ):
        raise TakeoverUsageError("Project takeover plan does not match the selected Groups.")
    if receipt is not None:
        durable_io = DurableIO()
        durable_io.sync_directory(_result_path(runtime.root, plan_id).parent, "takeover_result")
        durable_io.sync_directory(runtime.root, "takeover_result_parent")
    else:
        save_takeover_intent(runtime.root, intent, expected=intent)
    return intent


def _require_local_acceptance_state(runtime: MachineRuntime, intent: TakeoverIntent, runtime_id: str) -> None:
    runtime.assert_no_pending_project_registration()
    for key, operation in (
        ("registration_transaction", "registration recovery"),
        ("replacement_transaction", "machine identity replacement"),
    ):
        path = runtime.paths[key]
        if is_path_present(path):
            raise TakeoverIntentConflict(f"Resolve pending {operation} before a takeover. Evidence: {path}")

    transition = intent.transition
    if runtime_id != transition.successor_runtime_id:
        raise TakeoverUsageError("Machine runtime identity changed. Get a fresh --plan.")

    _, bindings = runtime.load_registry_uncached()
    matches = [
        binding
        for binding in bindings
        if binding.project_id == transition.project_id or binding.shared_root == transition.shared_root
    ]
    departing = transition.departing
    inventory = observe_takeover_inventory(runtime, transition.project_id, transition.shared_root)
    if departing is None and not matches and (inventory is None or inventory.enabled) == intent.original_enabled:
        return
    if departing is not None and len(matches) == 1 and inventory is not None:
        binding = matches[0]
        if (
            binding.project_id == transition.project_id
            and binding.shared_root == transition.shared_root
            and binding.machine_name == departing.machine_name
            and binding.registration_generation == departing.generation
            and binding.enabled == intent.original_enabled
            and binding.runtime_instance_id == departing.runtime_id
            and binding.runtime_root in {None, departing.runtime_root}
        ):
            return
    raise TakeoverUsageError("Local Project binding changed. Get a fresh --plan.")


def accept_project_takeover(
    runtime: MachineRuntime,
    path: str | Path,
    machine_name: str,
    plan_id: str,
    *,
    selected_groups: tuple[str, ...] = (),
) -> TakeoverIntent:
    """Confirm a preview and durably accept it without starting takeover work."""
    shared_root, machine_name = _normalize_request_scope(path, machine_name, plan_id)
    if not isinstance(selected_groups, tuple) or any(not isinstance(name, str) for name in selected_groups):
        raise TakeoverUsageError("Group selection must contain Group names.")
    groups = tuple(sorted(selected_groups))

    try:
        with runtime.agent_lifecycle_guard():
            intent = _read_requested_takeover(runtime, shared_root, machine_name, plan_id, acceptance_groups=groups)
            if intent is not None:
                return intent
        try:
            inputs = confirm_project_takeover(runtime, shared_root, machine_name, plan_id, selected_groups=groups)
        except (OSError, RuntimeError, ValueError):
            # A concurrent acceptance may make the same preview fail confirmation.
            with runtime.agent_lifecycle_guard():
                intent = _read_requested_takeover(runtime, shared_root, machine_name, plan_id, acceptance_groups=groups)
                if intent is not None:
                    return intent
            raise

        transition = TakeoverRegistration(
            project_id=inputs.project_id,
            shared_root=inputs.shared_root,
            transaction_id=uuid.uuid4().hex,
            successor_runtime_id=inputs.runtime_id,
            successor_runtime_root=inputs.runtime_root,
            successor_generation=uuid.uuid4().hex,
            target=inputs.target,
            departing=inputs.departing,
        )
        prepared = TakeoverIntent(
            plan_id=plan_id,
            transition=transition,
            original_enabled=inputs.original_enabled,
            group_approvals=inputs.groups,
        )
        with runtime.agent_lifecycle_guard():
            # The registry guard prepares the local layout, so validate identity first.
            runtime_id = runtime.observe_instance_id()
            with runtime.registry_guard():
                intent = _read_requested_takeover(runtime, shared_root, machine_name, plan_id, acceptance_groups=groups)
                if intent is not None:
                    return intent
                _require_local_acceptance_state(runtime, prepared, runtime_id)
                save_takeover_intent(runtime.root, prepared, expected=None)
        return prepared
    except TakeoverUsageError:
        raise
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Project takeover acceptance failed: {exc}") from exc


def request_takeover_cancellation(
    runtime: MachineRuntime,
    path: str | Path,
    machine_name: str,
    plan_id: str,
) -> TakeoverIntent:
    """Record cancellation for the coordinator or return the current outcome."""
    shared_root, machine_name = _normalize_request_scope(path, machine_name, plan_id)
    try:
        with runtime.agent_lifecycle_guard():
            intent = _read_requested_takeover(runtime, shared_root, machine_name, plan_id)
            if intent is None:
                raise TakeoverUsageError(f"Unknown Project takeover plan {plan_id}.")
            if intent.is_cancel_requested or intent.phase not in _CANCELLABLE_PHASES:
                return intent
            requested = replace(intent, is_cancel_requested=True)
            save_takeover_intent(runtime.root, requested, expected=intent)
            return requested
    except TakeoverUsageError:
        raise
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Project takeover cancellation failed: {exc}") from exc


def read_project_takeover_operation(
    runtime: MachineRuntime,
    path: str | Path,
    machine_name: str,
    plan_id: str,
) -> TakeoverIntent:
    """Observe one exact takeover in the active journal or terminal receipts."""
    shared_root, machine_name = _normalize_request_scope(path, machine_name, plan_id)
    intent = read_takeover_result(runtime.root, plan_id)
    if intent is None:
        active = read_takeover_intent(runtime.root)
        if active is not None and active.plan_id == plan_id:
            intent = active
        else:
            # A terminal receipt can appear before the active record clears.
            intent = read_takeover_result(runtime.root, plan_id)
    if intent is None:
        raise TakeoverUsageError(f"Unknown Project takeover plan {plan_id}.")

    transition = intent.transition
    if runtime.observe_instance_id() != transition.successor_runtime_id:
        raise TakeoverUsageError("Project takeover plan belongs to a different machine runtime identity.")
    if transition.shared_root != shared_root or transition.target.machine_name != machine_name:
        raise TakeoverUsageError("Project takeover plan does not match the selected Project and target machine.")
    return intent
