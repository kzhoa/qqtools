"""Handlers for machine-local and explicit Project setup commands."""

from __future__ import annotations

import argparse
import math
import shlex
import sys
from pathlib import Path
from typing import Callable

from ..agent.config import agent_config_payload, set_agent_config
from ..agent.context import MachineRuntime
from ..agent.lifecycle import (
    ensure_machine_agent_started,
    get_machine_agent_status,
    restart_machine_agent,
    stop_machine_agent,
)
from ..agent.project_admin import migrate_project
from ..agent.readiness import capture_readiness_snapshot, evaluate_readiness, wait_for_readiness
from ..agent.setup import (
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
from ..formatter import CliOutput, OutputKind
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
        raise ValueError(f"machine registry must identify exactly one project for {identifier!r}.")
    return matches[0].root_config()


def dispatch_local(
    handler: str,
    args: argparse.Namespace,
    *,
    emitter: Callable[..., None],
) -> int:
    """Dispatch a command that does not use ordinary Project selection."""
    if handler not in LOCAL_HANDLERS:
        raise ValueError(f"unsupported local handler {handler!r}.")
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
            return 2
        target_name = args.init_machine or args.machine
        if not target_name:
            raise ValueError("init requires explicit --machine NAME.")
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
                raise RuntimeError("machine identity replacement requires --yes when --format=json is selected.")
            isatty = getattr(sys.stdin, "isatty", None)
            if not callable(isatty) or not isatty():
                raise RuntimeError("machine identity replacement requires --yes in noninteractive input.")
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
                raise RuntimeError("machine identity replacement cancelled; no state was changed.")
            confirmed = True
            expected_old_runtime_id = facts["old_runtime_id"]
        result = initialize_machine(
            runtime,
            target_name,
            agent_mode=args.agent_mode,
            detach_old_runtime=args.detach_old_runtime,
            confirmed=confirmed,
            expected_old_runtime_id=expected_old_runtime_id,
        )
        emitter(CliOutput(OutputKind.MACHINE_INIT, result), args.format)
        return 0
    if handler.startswith("project_"):
        runtime = MachineRuntime(args.machine_runtime_root)
        if handler == "project_init":
            result = initialize_project(args.path)
            emitter(CliOutput(OutputKind.PROJECT_OPERATION, result), args.format)
            return 0
        if handler == "project_register":
            project_machine = args.project_machine
            if project_machine is not None and args.machine is not None and project_machine != args.machine:
                raise ValueError("project register --machine conflicts with the global --machine assertion.")
            if project_machine is None:
                project_machine = args.machine
            result = register_projects(
                runtime,
                args.paths,
                from_pool=args.from_pool,
                machine_name=project_machine,
                name_source=args.name_source,
            )
            emitter(CliOutput(OutputKind.PROJECT_REGISTER, result), args.format)
            statuses = [item.get("status") for item in result.get("projects", ())]
            return 0 if not statuses or all(item in {"registered", "disabled"} for item in statuses) else 2
        if handler == "project_list":
            emitter(CliOutput(OutputKind.PROJECT_LIST, list_projects(runtime)), args.format)
            return 0
        if handler == "project_remove":
            result = remove_project(runtime, args.selector)
        else:
            result = set_project_enablement(runtime, args.selector, handler == "project_enable")
        emitter(CliOutput(OutputKind.PROJECT_OPERATION, result), args.format)
        return 0
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
            result = configuration_commands.reset_config("agent", cfg=None, runtime=runtime)
        emitter(CliOutput(OutputKind.CONFIG, result), args.format)
        return 0
    if handler == "agent_name":
        runtime = MachineRuntime(args.machine_runtime_root)
        if args.set_to is not None:
            set_agent_config(runtime, name=args.set_to)
        result = agent_config_payload(runtime)
        emitter(CliOutput(OutputKind.AGENT_CONFIG, result), args.format)
        return 0
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
        return 2
    if handler in {"agent_start", "agent_run"}:
        runtime = MachineRuntime(args.machine_runtime_root)
        if handler == "agent_start" and (
            isinstance(args.timeout, bool) or not math.isfinite(args.timeout) or args.timeout <= 0
        ):
            raise ValueError("agent start timeout must be a positive finite number of seconds.")
        snapshot = capture_readiness_snapshot(runtime)
        _registry_revision, current_bindings = runtime.load_registry()
        registered_ids = {binding.project_id for binding in current_bindings if binding.enabled}
        if not snapshot["project_ids"] or not registered_ids.intersection(snapshot["project_ids"]):
            result = evaluate_readiness(runtime, snapshot)
            emitter(CliOutput(OutputKind.AGENT_READINESS, result), args.format)
            return 2
        if handler == "agent_run":
            from ..agent.lifecycle import run_machine_agent_loop

            run_machine_agent_loop(runtime)
            return 0
        ready = evaluate_readiness(runtime, snapshot)
        if not ready["ready"]:
            ensure_machine_agent_started(runtime)
            ready = wait_for_readiness(runtime, snapshot, timeout_seconds=args.timeout)
        emitter(CliOutput(OutputKind.AGENT_READINESS, ready), args.format)
        return 0 if ready["ready"] else 2
    if handler == "use":
        use_project = args.use_project if args.use_project is not None else getattr(args, "project", None)
        is_selecting = use_project is not None
        if sum((is_selecting, args.show, args.clear)) != 1:
            raise ValueError("use requires exactly one of --project, --show, or --clear.")
        if args.machine is not None or args.runtime_root is not None or args.machine_runtime_root is not None:
            raise ValueError("qexp use accepts only --project, --show, or --clear.")
        if args.clear:
            clear_context()
            return 0
        if args.show:
            context = load_context()
            emitter(
                CliOutput(
                    OutputKind.CONTEXT,
                    {"shared_root": context["shared_root"] if context else None},
                ),
                args.format or "human",
            )
            return 0
        if not use_project:
            raise ValueError("use requires a non-empty --project.")
        save_context(context_commands.normalize_project_path(use_project))
        return 0
    if handler in {"agent_status", "agent_stop", "agent_restart"}:
        runtime = MachineRuntime(args.machine_runtime_root)
        if handler == "agent_status":
            emitter(
                CliOutput(OutputKind.AGENT_STATUS, {"action": "status", **get_machine_agent_status(runtime)}),
                args.format,
            )
        elif handler == "agent_stop":
            stopped = stop_machine_agent(runtime)
            emitter(
                CliOutput(
                    OutputKind.AGENT_OPERATION,
                    {"action": "stopped" if stopped else "already_stopped", **get_machine_agent_status(runtime)},
                ),
                args.format,
            )
        else:
            process = restart_machine_agent(runtime)
            emitter(
                CliOutput(
                    OutputKind.AGENT_OPERATION,
                    {
                        "action": "restarted",
                        **get_machine_agent_status(runtime),
                        "pid": process.pid,
                        "previous_pid": getattr(process, "previous_pid", None),
                    },
                ),
                args.format,
            )
        return 0
    if handler.startswith("agent_config_"):
        runtime = MachineRuntime(args.machine_runtime_root)
        if handler.startswith("agent_config_gpus_"):
            runtime.ensure_layout()
            if handler == "agent_config_gpus_show":
                result = show_gpu_policy(runtime)
            elif handler == "agent_config_gpus_reset":
                result = reset_gpu_policy(runtime, expected_revision=args.expected_revision)
            else:
                configured = () if args.none else parse_gpu_id_list(args.visible)
                result = set_gpu_policy(runtime, configured, expected_revision=args.expected_revision)
            result["machine_runtime_root"] = str(runtime.root)
            emitter(CliOutput(OutputKind.GPU_POLICY, result), args.format)
            return 0
        runtime.ensure_layout()
        policy = (
            set_cpu_lane_capacity(runtime.root, capacity=args.capacity)
            if handler == "agent_config_cpu_set"
            else get_cpu_lane_policy(runtime.root)
        )
        emitter(
            CliOutput(
                OutputKind.CPU_LANE,
                {"machine_runtime_root": str(runtime.root), "cpu_lane": policy.to_dict},
            ),
            args.format,
        )
        return 0

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
                kind = OutputKind.UPGRADE_REGISTRY_STATUS if action == "status" else OutputKind.UPGRADE_ADVANCE
            else:
                selected = _upgrade_project_config(runtime, project)
                coordinator = UpgradeCoordinator(selected)
                result = coordinator.status() if action == "status" else coordinator.advance(force_retry=True)
                kind = OutputKind.UPGRADE_PROJECT
            emitter(CliOutput(kind, result), args.format)
            return 0
        if project is None:
            raise ValueError(f"admin upgrade {action} requires explicit --project PATH.")
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
        kind = OutputKind.UPGRADE_REPAIR if action in {"plan", "apply", "validate"} else OutputKind.UPGRADE_PROJECT
        emitter(CliOutput(kind, result, {"action": action}), args.format)
        return 0

    if handler.startswith("admin_migrate_"):
        if args.project is None:
            raise ValueError("admin migrate requires explicit --project PATH.")
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
                raise ValueError("only --to-schema 6 is supported.")
            migrate_schema5_to_schema6(cfg)
            emitter(
                CliOutput(OutputKind.SCHEMA6_UPGRADE, {"phase": "completed", "project": str(cfg.project_root)}),
                args.format,
            )
            return 0
        if handler == "admin_migrate_agent":
            if args.machine is None:
                raise ValueError("admin migrate agent requires --machine NAME.")
            runtime = MachineRuntime(args.machine_runtime_root)
            binding = migrate_project(runtime, cfg)
            _process, agent_status = ensure_machine_agent_started(runtime)
            emitter(
                CliOutput(
                    OutputKind.AGENT_OPERATION,
                    {
                        "action": "project_migrated",
                        **binding.to_dict(),
                        **agent_status,
                        "migration_candidates": [],
                    },
                ),
                args.format,
            )
            return 0
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
                raise ValueError("attest requires --machine and --confirm-clients-stopped.")
            result = attest_schema6_upgrade(
                cfg,
                activation_id=args.activation_id,
                machine_name=args.machine,
                machine_runtime_root=args.machine_runtime_root,
            )
        emitter(CliOutput(OutputKind.SCHEMA6_UPGRADE, result), args.format)
        return 0

    return 0
