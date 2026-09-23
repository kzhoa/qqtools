"""Handlers for machine-local and explicit Project setup commands."""

from __future__ import annotations

import argparse
import math
import os
import shlex
import sys
from pathlib import Path

from ..agent.config import agent_config_payload, set_agent_config
from ..agent.context import MachineRuntime, MachineRuntimeUninitializedError
from ..agent.lifecycle import (
    MachineAgentStartBlockedError,
    MachineAgentStartError,
    MachineAgentStopError,
    ensure_machine_agent_started,
    get_machine_agent_status,
    restart_machine_agent,
    stop_machine_agent,
)
from ..agent.project_admin import migrate_project, project_migration_state
from ..agent.readiness import capture_readiness_snapshot, evaluate_readiness, wait_for_readiness
from ..agent.setup import (
    SetupOperationalError,
    SetupUsageError,
    initialize_machine,
    initialize_project,
    list_projects,
    machine_init_facts,
    register_projects,
    remove_project,
    set_project_enablement,
)
from ..commands import configuration as configuration_commands
from ..commands import context as context_commands
from ..gpu_policy import parse_gpu_id_list, reset_gpu_policy, set_gpu_policy, show_gpu_policy
from ..layout import clear_context, load_context, load_root_config, migrate_schema5_to_schema6, save_context
from ..runtime.resources.cpu_lane import get_cpu_lane_policy, set_cpu_lane_capacity
from ..runtime.store import read_json
from ..runtime.upgrade.framework import UpgradeCoordinator
from ..runtime.upgrade.machine import advance_registered_upgrades, inspect_registered_upgrades
from ..schema6_upgrade import (
    attest_schema6_upgrade,
    check_schema6_upgrade,
    resume_schema6_upgrade,
    schema6_upgrade_status,
    start_schema6_upgrade,
)
from .errors import CliOperationalError, CliUsageError
from .outcome import CommandOutcome
from .output import CliOutput, OutputKind

LOCAL_HANDLERS = frozenset(
    {
        "init",
        "project_init",
        "project_register",
        "project_list",
        "project_enable",
        "project_disable",
        "project_remove",
        "use",
        "agent_start",
        "agent_run",
        "agent_name",
        "agent_status",
        "agent_stop",
        "agent_restart",
        "agent_config_gpus_show",
        "agent_config_gpus_set",
        "agent_config_gpus_reset",
        "agent_config_cpu_show",
        "agent_config_cpu_set",
        "config_show",
        "config_set",
        "config_reset",
        "admin_upgrade_status",
        "admin_upgrade_advance",
        "admin_upgrade_pause",
        "admin_upgrade_plan",
        "admin_upgrade_apply",
        "admin_upgrade_validate",
        "admin_upgrade_resume",
        "admin_migrate_schema",
        "admin_migrate_schema6_check",
        "admin_migrate_schema6_start",
        "admin_migrate_schema6_status",
        "admin_migrate_schema6_attest",
        "admin_migrate_schema6_resume",
        "admin_migrate_agent",
        "admin_repair",
        "retired_add-project",
        "retired_list-projects",
        "retired_enable-project",
        "retired_disable-project",
        "retired_remove-project",
        "retired_migrate-project",
    }
)


def _try_save_context(shared_root: str) -> None:
    try:
        save_context(shared_root)
    except OSError as exc:
        print(
            "qexp: initialized successfully, but failed to save CLI context "
            f"at {exc.filename or '~/.qqtools/qexp-context.json'}: {exc}",
            file=sys.stderr,
        )


def _upgrade_project_config(runtime: MachineRuntime, identifier: str):
    """Resolve an explicit registered project for a repair or project-scoped retry."""
    candidate = str(identifier)
    canonical = Path(candidate).expanduser().resolve() if candidate.endswith(".qexp") or "/" in candidate else None
    _revision, bindings = runtime.load_registry()
    matches = [
        binding
        for binding in bindings
        if binding.project_id == candidate or (canonical is not None and binding.shared_root == canonical)
    ]
    if len(matches) != 1:
        raise CliUsageError(f"machine registry must identify exactly one project for {identifier!r}.")
    return matches[0].root_config()


def _upgrade_payload(
    result: dict[str, object],
    action: str,
    *,
    project_root: str | Path | None = None,
) -> dict[str, object]:
    """Move upgrade operation facts into the canonical payload envelope."""
    payload = dict(result)
    payload["action"] = action
    blockers = payload.get("blockers")
    blocker_values = list(blockers) if isinstance(blockers, (list, tuple)) else []
    inaccessible = payload.get("inaccessible_projects")
    inaccessible_values = list(inaccessible) if isinstance(inaccessible, (list, tuple)) else []
    pending = bool(payload.get("pending")) or bool(payload.get("pending_project_ids"))
    aggregate_state = payload.get("aggregate_state")
    state = payload.get("state")
    if payload.get("error") or state == "validation_failed":
        outcome = "failed"
    elif aggregate_state == "inaccessible" or inaccessible_values:
        outcome = "blocked"
    elif state in {"repair_required", "paused", "pause_pending", "blocked"} or payload.get("admission_blocked"):
        outcome = "blocked"
    elif pending:
        outcome = "waiting"
    else:
        outcome = "completed"
    payload.setdefault("outcome", outcome)

    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason:
        if blocker_values:
            reason = str(blocker_values[0])
        elif inaccessible_values:
            first = inaccessible_values[0]
            if isinstance(first, dict):
                reason = first.get("reason") or first.get("error") or "registered project is inaccessible"
            else:
                reason = "registered project is inaccessible"
        elif isinstance(payload.get("pause"), dict) and payload["pause"].get("reason"):
            reason = str(payload["pause"]["reason"])
        elif isinstance(payload.get("repair"), dict) and payload["repair"].get("error"):
            reason = str(payload["repair"]["error"])
        elif pending:
            reason = "upgrade work remains pending"
        elif payload.get("error"):
            reason = str(payload["error"])
        elif outcome == "completed":
            reason = "no upgrade work remains" if action != "status" else "upgrade status inspected"
    payload.setdefault("reason", reason)

    follow_up: str | None = None
    if action == "validate" and outcome == "failed" and project_root is not None and payload.get("target"):
        follow_up = (
            f"qexp admin upgrade plan --project {shlex.quote(str(project_root))} "
            f"--target {shlex.quote(str(payload['target']))}"
        )
    elif pending or inaccessible_values or outcome in {"blocked", "failed"}:
        if project_root is None:
            follow_up = "qexp admin upgrade status"
        else:
            follow_up = f"qexp admin upgrade status --project {shlex.quote(str(project_root))}"
    payload.setdefault("next_action", follow_up)
    return payload


def _schema_migration_journal_path(cfg: object) -> Path:
    shared_root = getattr(cfg, "shared_root")
    return shared_root.parent / f".{shared_root.name}.schema6-migration.json"


def _schema_version(cfg: object) -> int | None:
    try:
        schema = read_json(getattr(cfg, "shared_root") / "schema" / "version.json")
    except (OSError, TypeError, ValueError, KeyError):
        return None
    value = schema.get("schema", {}).get("version") if isinstance(schema, dict) else None
    return value if type(value) is int else None


def _schema_migration_payload(
    cfg: object,
    *,
    source_schema: int | None,
    error: BaseException | None = None,
) -> dict[str, object]:
    """Describe the durable schema-5-to-6 cutover without inventing completion."""
    journal_path = _schema_migration_journal_path(cfg)
    migration: dict[str, object] = {}
    if journal_path.exists():
        try:
            value = read_json(journal_path)
            raw = value.get("migration") if isinstance(value, dict) else None
            if isinstance(raw, dict):
                migration = raw
        except (OSError, TypeError, ValueError):
            migration = {}
    current_schema = _schema_version(cfg)
    journal_phase = migration.get("phase")
    if current_schema == 6 and source_schema == 6:
        phase = "already_current"
    elif isinstance(journal_phase, str) and journal_phase:
        phase = journal_phase
    elif current_schema == 6:
        phase = "already_current"
    elif error is not None:
        phase = "blocked"
    else:
        phase = "staging"
    if type(source_schema) is not int:
        source_schema = migration.get("from_schema") if type(migration.get("from_schema")) is int else current_schema
    target_schema = migration.get("to_schema") if type(migration.get("to_schema")) is int else 6

    blockers_value = migration.get("blockers")
    blockers = list(blockers_value) if isinstance(blockers_value, list) else []
    if error is not None and not blockers:
        message = str(error)
        if "blockers:" in message:
            blockers = [item.strip() for item in message.split("blockers:", 1)[1].split(",") if item.strip()]
        elif message:
            blockers = [message]
    destructive_reached = phase in {"source_parked", "committed"}
    if phase in {"committed", "already_current"}:
        outcome = "no_change" if phase == "already_current" else "completed"
    elif error is not None or phase in {"blocked", "source_parked"} or blockers:
        outcome = "blocked"
    else:
        outcome = "waiting"
    next_action = None
    if phase not in {"committed", "already_current"}:
        next_action = (
            f"qexp admin migrate schema --project {shlex.quote(str(getattr(cfg, 'project_root')))} --to-schema 6"
        )
    result: dict[str, object] = {
        "action": "migrate_schema",
        "outcome": outcome,
        "phase": phase,
        "project": str(getattr(cfg, "project_root")),
        "shared_root": str(getattr(cfg, "shared_root")),
        "source_schema": source_schema,
        "target_schema": target_schema,
        "from_schema": source_schema,
        "to_schema": target_schema,
        "blockers": blockers,
        "destructive_boundary": {
            "name": "schema_root_replacement",
            "reached": destructive_reached,
            "recoverable": not destructive_reached,
        },
        "destructive_boundary_reached": destructive_reached,
        "next_action": next_action,
        "journal_path": str(journal_path),
        "migration_journal_phase": journal_phase,
    }
    if error is not None:
        result["error"] = {"code": "schema_migration_blocked", "message": str(error)}
        result["reason"] = blockers[0] if blockers else str(error)
    elif phase == "already_current":
        result["reason"] = "schema is already current"
    elif phase == "committed":
        result["reason"] = "schema root replacement committed"
    return result


def dispatch_local(
    handler: str,
    args: argparse.Namespace,
) -> CommandOutcome:
    """Dispatch a command that does not use ordinary Project selection."""
    if handler not in LOCAL_HANDLERS:
        raise ValueError(f"unsupported local handler {handler!r}.")
    if handler == "admin_repair" and getattr(args, "repair_target", None) == "identity":
        if not args.dry_run:
            raise CliUsageError("automatic restoration is unavailable; pass --dry-run for identity diagnosis.")
        from ..agent.identity_diagnosis import diagnose_machine_identity

        if args.machine_runtime_root is not None:
            selection_source = "--machine-runtime-root"
        elif os.environ.get("QEXP_MACHINE_RUNTIME_ROOT"):
            selection_source = "QEXP_MACHINE_RUNTIME_ROOT"
        else:
            selection_source = "default"
        result = diagnose_machine_identity(args.machine_runtime_root, selection_source=selection_source)
        exit_code = 0 if result["outcome"] == "healthy" else 1
        return CommandOutcome(exit_code, CliOutput(OutputKind.MACHINE_IDENTITY_DIAGNOSIS, result))
    if handler == "init":
        # QQTOOLS-COMPAT-0014: the former project-bound init spelling is
        # retained only as a bounded diagnostic during the transition.
        if args.init_shared_root or args.runtime_root or args.cpu_lane_capacity is not None:
            print(
                "qexp: QQTOOLS-COMPAT-0014: qexp init is machine-only. "
                "Use 'qexp project init PATH' to create shared Project truth, then "
                "'qexp project register PATH --machine NAME' to enroll it.",
                file=sys.stderr,
            )
            return CommandOutcome(2)
        target_name = args.init_machine or args.machine
        if not target_name:
            raise CliUsageError("init requires explicit --machine NAME.")
        runtime = MachineRuntime(args.machine_runtime_root)
        facts = machine_init_facts(runtime)
        confirmed = bool(args.yes)
        expected_old_runtime_id = None
        if args.detach_old_runtime and facts.get("old_runtime_id"):
            print(
                "WARNING: --detach-old-runtime preserves old local evidence in an isolated archive "
                "without completing recovery or changing shared Tasks, claims, or registrations.",
                file=sys.stderr,
            )
        if facts.get("old_runtime_id") and not confirmed:
            if args.format == "json":
                raise CliOperationalError("machine identity replacement requires --yes when --format=json is selected.")
            isatty = getattr(sys.stdin, "isatty", None)
            if not callable(isatty) or not isatty():
                raise CliUsageError("machine identity replacement requires --yes in noninteractive input.")
            print("Machine identity replacement will:", file=sys.stderr)
            print(f"  old runtime ID: {facts.get('old_runtime_id')}", file=sys.stderr)
            print(f"  current name: {facts.get('current_name') or '-'}", file=sys.stderr)
            print(f"  requested name: {target_name}", file=sys.stderr)
            print(
                f"  requested mode: {args.agent_mode or facts.get('current_agent_mode') or 'daemon'}",
                file=sys.stderr,
            )
            print(f"  obligations: {facts.get('obligations') or 'none'}", file=sys.stderr)
            print("  this creates a new runtime ID and replaces local authority.", file=sys.stderr)
            print(
                f"  shell-safe command: qexp init --machine {shlex.quote(target_name)} --yes",
                file=sys.stderr,
            )
            answer = input("Continue? [y/N] ")
            if answer.strip().lower() not in {"y", "yes"}:
                raise CliUsageError("machine identity replacement cancelled; no state was changed.")
            confirmed = True
            expected_old_runtime_id = facts["old_runtime_id"]
        try:
            result = initialize_machine(
                runtime,
                target_name,
                agent_mode=args.agent_mode,
                detach_old_runtime=args.detach_old_runtime,
                confirmed=confirmed,
                expected_old_runtime_id=expected_old_runtime_id,
            )
        except SetupUsageError as exc:
            raise CliUsageError(str(exc)) from exc
        except SetupOperationalError as exc:
            raise CliOperationalError(str(exc)) from exc
        return CommandOutcome(0, CliOutput(OutputKind.MACHINE_INIT, result))
    if handler.startswith("project_"):
        runtime = MachineRuntime(args.machine_runtime_root)
        if handler == "project_init":
            try:
                result = initialize_project(args.path)
            except SetupUsageError as exc:
                raise CliUsageError(str(exc)) from exc
            except SetupOperationalError as exc:
                raise CliOperationalError(str(exc)) from exc
            return CommandOutcome(0, CliOutput(OutputKind.PROJECT_OPERATION, result))
        if handler == "project_register":
            project_machine = args.project_machine
            if project_machine is not None and args.machine is not None and project_machine != args.machine:
                raise CliUsageError("project register --machine conflicts with the global --machine assertion.")
            if project_machine is None:
                project_machine = args.machine
            try:
                result = register_projects(
                    runtime,
                    args.paths,
                    from_pool=args.from_pool,
                    machine_name=project_machine,
                    name_source=args.name_source,
                )
            except SetupUsageError as exc:
                raise CliUsageError(str(exc)) from exc
            except (SetupOperationalError, MachineRuntimeUninitializedError) as exc:
                raise CliOperationalError(str(exc)) from exc
            statuses = [item.get("status") for item in result.get("projects", ())]
            exit_code = 0 if not statuses or all(item in {"registered", "disabled"} for item in statuses) else 2
            return CommandOutcome(exit_code, CliOutput(OutputKind.PROJECT_REGISTER, result))
        try:
            if handler == "project_list":
                return CommandOutcome(0, CliOutput(OutputKind.PROJECT_LIST, list_projects(runtime)))
            if handler == "project_remove":
                result = remove_project(runtime, args.selector)
            else:
                result = set_project_enablement(runtime, args.selector, handler == "project_enable")
        except SetupUsageError as exc:
            raise CliUsageError(str(exc)) from exc
        except (SetupOperationalError, MachineRuntimeUninitializedError) as exc:
            raise CliOperationalError(str(exc)) from exc
        return CommandOutcome(0, CliOutput(OutputKind.PROJECT_OPERATION, result))
    if handler in {"config_show", "config_set", "config_reset"} and getattr(args, "section", None) == "agent":
        runtime = MachineRuntime(args.machine_runtime_root)
        if handler == "config_show":
            result = configuration_commands.show_config("agent", cfg=None, runtime=runtime)
        elif handler == "config_set":
            values = {
                key: value
                for key, value in {"name": args.name, "agent_mode": args.agent_mode}.items()
                if value is not None
            }
            result = configuration_commands.set_config(
                "agent", cfg=None, runtime=runtime, provider=args.provider, values=values
            )
        else:
            # Agent configuration is machine-global and has no reset
            # default.  Reject it without resolving a Project context.
            raise CliUsageError("agent configuration cannot be reset")
        return CommandOutcome(0, CliOutput(OutputKind.CONFIG, result))
    if handler == "agent_name":
        runtime = MachineRuntime(args.machine_runtime_root)
        changed = args.set_to is not None
        if args.set_to is not None:
            set_agent_config(runtime, name=args.set_to)
        result = agent_config_payload(runtime)
        result["action"] = "updated" if changed else "shown"
        return CommandOutcome(0, CliOutput(OutputKind.AGENT_CONFIG, result))
    if handler.startswith("retired_"):
        # QQTOOLS-COMPAT-0014: retired executing routes fail before any
        # runtime/configuration resolution or state creation.
        replacement = {
            "add-project": "qexp project register PATH",
            "list-projects": "qexp project list",
            "enable-project": "qexp project enable SELECTOR",
            "disable-project": "qexp project disable SELECTOR",
            "remove-project": "qexp project remove SELECTOR",
            "migrate-project": "qexp admin migrate agent --project PATH --machine NAME",
        }[handler.removeprefix("retired_")]
        print(
            f"qexp: QQTOOLS-COMPAT-0014: retired command; use '{replacement}'.",
            file=sys.stderr,
        )
        return CommandOutcome(2)
    if handler in {"agent_start", "agent_run"}:
        runtime = MachineRuntime(args.machine_runtime_root)
        if handler == "agent_run":
            from ..agent.lifecycle import run_machine_agent_loop

            try:
                run_machine_agent_loop(runtime)
            except (MachineAgentStartBlockedError, MachineAgentStartError, MachineRuntimeUninitializedError) as exc:
                raise CliOperationalError(str(exc)) from exc
            return CommandOutcome(0)
        if handler == "agent_start" and (
            isinstance(args.timeout, bool) or not math.isfinite(args.timeout) or args.timeout <= 0
        ):
            raise CliUsageError("agent start timeout must be a positive finite number of seconds.")
        try:
            snapshot = capture_readiness_snapshot(runtime)
            _registry_revision, current_bindings = runtime.load_registry()
        except (MachineAgentStartBlockedError, MachineRuntimeUninitializedError) as exc:
            raise CliOperationalError(str(exc)) from exc
        registered_ids = {binding.project_id for binding in current_bindings if binding.enabled}
        if not snapshot["project_ids"] or not registered_ids.intersection(snapshot["project_ids"]):
            result = evaluate_readiness(runtime, snapshot)
            return CommandOutcome(2, CliOutput(OutputKind.AGENT_READINESS, result))
        ready = evaluate_readiness(runtime, snapshot)
        started_process = None
        if not ready["ready"]:
            try:
                started_process, _status = ensure_machine_agent_started(runtime)
            except (MachineAgentStartBlockedError, MachineAgentStartError, MachineRuntimeUninitializedError) as exc:
                raise CliOperationalError(str(exc)) from exc
            ready = wait_for_readiness(runtime, snapshot, timeout_seconds=args.timeout)
        if not ready["ready"]:
            return CommandOutcome(2, CliOutput(OutputKind.AGENT_READINESS, ready))
        action = "started" if started_process is not None else "already_running"
        operation = {
            **ready,
            "action": action,
            "outcome": action,
            "is_running": True,
            "ready": True,
        }
        return CommandOutcome(0, CliOutput(OutputKind.AGENT_OPERATION, operation))
    if handler == "use":
        use_project = args.use_project if args.use_project is not None else getattr(args, "project", None)
        is_selecting = use_project is not None
        if sum((is_selecting, args.show, args.clear)) != 1:
            raise CliUsageError("use requires exactly one of --project, --show, or --clear.")
        if args.machine is not None or args.runtime_root is not None or args.machine_runtime_root is not None:
            raise CliUsageError("qexp use accepts only --project, --show, or --clear.")
        if args.clear:
            changed = clear_context()
            return CommandOutcome(
                0,
                CliOutput(
                    OutputKind.CONTEXT,
                    {"action": "cleared", "shared_root": None, "changed": changed},
                ),
            )
        if args.show:
            context = load_context()
            return CommandOutcome(
                0,
                CliOutput(OutputKind.CONTEXT, {"shared_root": context["shared_root"] if context else None}),
            )
        if not use_project:
            raise CliUsageError("use requires a non-empty --project.")
        previous = load_context()
        normalized = context_commands.normalize_project_path(use_project)
        try:
            save_context(normalized)
        except OSError as exc:
            raise CliOperationalError(
                str(exc),
                code="context_write_failed",
                next_action="Check permissions for the qexp saved-context file.",
            ) from exc
        previous_root = previous.get("shared_root") if previous is not None else None
        selected_root = str(normalized)
        return CommandOutcome(
            0,
            CliOutput(
                OutputKind.CONTEXT,
                {
                    "action": "selected",
                    "shared_root": selected_root,
                    "changed": previous_root != selected_root,
                },
            ),
        )
    if handler in {"agent_status", "agent_stop", "agent_restart"}:
        runtime = MachineRuntime(args.machine_runtime_root)
        if handler == "agent_status":
            try:
                status = get_machine_agent_status(runtime)
            except MachineRuntimeUninitializedError as exc:
                raise CliOperationalError(str(exc)) from exc
            output = CliOutput(OutputKind.AGENT_STATUS, {"action": "status", **status})
        elif handler == "agent_stop":
            try:
                stopped = stop_machine_agent(runtime)
            except (MachineAgentStopError, MachineRuntimeUninitializedError) as exc:
                raise CliOperationalError(str(exc)) from exc
            status = get_machine_agent_status(runtime)
            action = "stopped" if stopped else "already_stopped"
            output = CliOutput(OutputKind.AGENT_OPERATION, {"action": action, "outcome": action, **status})
        else:
            try:
                process = restart_machine_agent(runtime)
            except (
                MachineAgentStartBlockedError,
                MachineAgentStartError,
                MachineAgentStopError,
                MachineRuntimeUninitializedError,
            ) as exc:
                raise CliOperationalError(str(exc)) from exc
            status = get_machine_agent_status(runtime)
            ready = status.get("ready")
            output = CliOutput(
                OutputKind.AGENT_OPERATION,
                {
                    "action": "restarted",
                    "outcome": "restarted",
                    **status,
                    "pid": process.pid,
                    "previous_pid": getattr(process, "previous_pid", None),
                    "ready": True if ready else None,
                    "is_running": True,
                },
            )
        return CommandOutcome(0, output)
    if handler.startswith("agent_config_"):
        runtime = MachineRuntime(args.machine_runtime_root)
        if handler.startswith("agent_config_gpus_"):
            runtime.ensure_layout()
            if handler == "agent_config_gpus_show":
                result = show_gpu_policy(runtime)
                action = "shown"
            elif handler == "agent_config_gpus_reset":
                result = reset_gpu_policy(runtime, expected_revision=args.expected_revision)
                action = "reset"
            else:
                try:
                    configured = () if args.none else parse_gpu_id_list(args.visible)
                except ValueError as exc:
                    raise CliUsageError(str(exc)) from exc
                result = set_gpu_policy(runtime, configured, expected_revision=args.expected_revision)
                action = "updated"
            result["machine_runtime_root"] = str(runtime.root)
            result["action"] = action
            return CommandOutcome(0, CliOutput(OutputKind.GPU_POLICY, result))
        runtime.ensure_layout()
        policy = (
            set_cpu_lane_capacity(runtime.root, capacity=args.capacity)
            if handler == "agent_config_cpu_set"
            else get_cpu_lane_policy(runtime.root)
        )
        return CommandOutcome(
            0,
            CliOutput(
                OutputKind.CPU_LANE,
                {
                    "machine_runtime_root": str(runtime.root),
                    "cpu_lane": policy.to_dict,
                    "action": "updated" if handler == "agent_config_cpu_set" else "shown",
                },
            ),
        )

    if handler.startswith("admin_upgrade_"):
        runtime = MachineRuntime(args.machine_runtime_root)
        action = handler.removeprefix("admin_upgrade_")
        project = getattr(args, "project", None)
        if action in {"status", "advance"}:
            if project is None:
                result = (
                    inspect_registered_upgrades(runtime)
                    if action == "status"
                    else advance_registered_upgrades(runtime, force_discovery=True)
                )
                result = _upgrade_payload(result, action)
                kind = OutputKind.UPGRADE_REGISTRY_STATUS if action == "status" else OutputKind.UPGRADE_ADVANCE
            else:
                selected = _upgrade_project_config(runtime, project)
                coordinator = UpgradeCoordinator(selected)
                result = coordinator.status() if action == "status" else coordinator.advance(force_retry=True)
                result = _upgrade_payload(result, action, project_root=selected.shared_root)
                kind = OutputKind.UPGRADE_PROJECT
            return CommandOutcome(0, CliOutput(kind, result))
        if project is None:
            raise CliUsageError(f"admin upgrade {action} requires explicit --project PATH.")
        selected = _upgrade_project_config(runtime, project)
        coordinator = UpgradeCoordinator(selected)
        if action == "pause":
            result = coordinator.request_pause(args.reason)
        elif action == "plan":
            result = coordinator.inspect_repair(args.target)
        elif action == "apply":
            result = coordinator.apply_repair(args.repair_id)
        elif action == "validate":
            result = coordinator.validate_repair(args.repair_id)
        else:
            result = coordinator.resume()
        result = _upgrade_payload(result, action, project_root=selected.shared_root)
        kind = OutputKind.UPGRADE_REPAIR if action in {"plan", "apply", "validate"} else OutputKind.UPGRADE_PROJECT
        return CommandOutcome(0, CliOutput(kind, result))

    if handler.startswith("admin_migrate_"):
        if args.project is None:
            raise CliUsageError("admin migrate requires explicit --project PATH.")
        selection_path = context_commands.normalize_project_path(args.project)
        machine = args.machine or "upgrade-coordinator"
        cfg = load_root_config(
            selection_path,
            machine,
            getattr(args, "runtime_root", None),
            require_initialized=False,
        )
        if handler == "admin_migrate_schema":
            if args.to_schema != 6:
                raise CliUsageError("only --to-schema 6 is supported.")
            source_schema = _schema_version(cfg)
            try:
                migrate_schema5_to_schema6(cfg)
            except (OSError, RuntimeError) as exc:
                return CommandOutcome(
                    1,
                    CliOutput(
                        OutputKind.SCHEMA6_UPGRADE,
                        _schema_migration_payload(cfg, source_schema=source_schema, error=exc),
                    ),
                )
            return CommandOutcome(
                0,
                CliOutput(
                    OutputKind.SCHEMA6_UPGRADE,
                    _schema_migration_payload(cfg, source_schema=source_schema),
                ),
            )
        if handler == "admin_migrate_agent":
            if args.machine is None:
                raise CliUsageError("admin migrate agent requires --machine NAME.")
            runtime = MachineRuntime(args.machine_runtime_root)
            binding = migrate_project(runtime, cfg)
            migration = project_migration_state(runtime, binding)
            try:
                _process, agent_status = ensure_machine_agent_started(runtime)
            except (
                MachineAgentStartBlockedError,
                MachineAgentStartError,
                MachineRuntimeUninitializedError,
                OSError,
            ) as exc:
                try:
                    agent_status = get_machine_agent_status(runtime)
                except (MachineRuntimeUninitializedError, OSError, RuntimeError, ValueError, KeyError, TypeError):
                    agent_status = {
                        "machine_runtime_root": str(runtime.root),
                        "agent_state": "unavailable",
                        "pid": None,
                        "is_running": False,
                        "ready": False,
                        "projects": [],
                    }
                return CommandOutcome(
                    1,
                    CliOutput(
                        OutputKind.AGENT_OPERATION,
                        {
                            "action": "project_migrated",
                            "outcome": "partial",
                            **binding.to_dict(),
                            **agent_status,
                            "migration_state": migration.get("state"),
                            "migration": migration,
                            "reason": "project migration committed but machine agent start failed",
                            "error": {"code": "agent_start_failed", "message": str(exc)},
                            "follow_up_command": "qexp agent start",
                            "next_action": "qexp agent start",
                            "migration_candidates": [],
                        },
                    ),
                )
            return CommandOutcome(
                0,
                CliOutput(
                    OutputKind.AGENT_OPERATION,
                    {
                        "action": "project_migrated",
                        "outcome": "completed",
                        **binding.to_dict(),
                        **agent_status,
                        "migration_state": migration.get("state"),
                        "migration": migration,
                        "migration_candidates": [],
                    },
                ),
            )
        action = handler.removeprefix("admin_migrate_schema6_")
        if action == "check":
            result = check_schema6_upgrade(
                cfg,
                capabilities=args.capabilities,
                machine_runtime_root=args.machine_runtime_root,
            )
        elif action == "status":
            result = schema6_upgrade_status(cfg)
        elif action == "start":
            result = start_schema6_upgrade(
                cfg,
                capabilities=args.capabilities,
                machine_runtime_root=args.machine_runtime_root,
            )
        elif action == "resume":
            result = resume_schema6_upgrade(
                cfg,
                activation_id=args.activation_id,
                machine_runtime_root=args.machine_runtime_root,
            )
        else:
            if not args.confirm_clients_stopped or args.machine is None:
                raise CliUsageError("attest requires --machine and --confirm-clients-stopped.")
            result = attest_schema6_upgrade(
                cfg,
                activation_id=args.activation_id,
                machine_name=args.machine,
                machine_runtime_root=args.machine_runtime_root,
            )
        return CommandOutcome(0, CliOutput(OutputKind.SCHEMA6_UPGRADE, result))

    return CommandOutcome(0)
